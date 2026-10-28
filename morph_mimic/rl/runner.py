"""On-policy runner: rollout with planner-in-the-loop, hindsight buffers,
planner co-training, and dual-critic PPO (arXiv 2610.31608).

Stage 1: reference-only rollouts + PPO (no planner, no buffers).
Stage 2: planner runs during rollout (eta_obs/eta_rew annealing); the
    planner is supervised from the hindsight buffer of successful
    reference rollouts; 10 planner updates per PPO iteration.
Stage 3: mixed reference/goal-reaching rollouts; dual critics with
    warmup; both buffers feed the planner (proportional mixing by stored
    counts); successful goal rollouts bootstrap the planner.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from morph_mimic.contacts.labeling import hindsight_contacts
from morph_mimic.contacts.replay import HindsightBuffer, gaussian_targets, planner_loss
if TYPE_CHECKING:
    from morph_mimic.envs.acp_env import AcpEnv
from morph_mimic.rl.ppo import PPO, PPOCfg
from morph_mimic.rl.storage import RolloutStorage
from morph_mimic.utils.checkpoint import (
    atomic_torch_save, build_checkpoint_contract, capture_rng_state, file_hash,
    restore_rng_state, validate_checkpoint_contract,
)


@dataclass
class RunnerCfg:
    # rollout / PPO
    num_steps: int = 24
    ppo: PPOCfg = field(default_factory=PPOCfg)

    # planner co-training (Table S5)
    planner_updates_per_iter: int = 10
    planner_batch: int = 4096
    planner_grad_sub_batches: int = 8
    planner_lr: float = 1e-4
    planner_min_entries: int = 1000
    buffer_cap_steps: int = 40000

    # annealing schedules (Eq. 3); paper leaves the iteration numbers
    # configurable -- defaults below
    eta_obs_start: int = 100
    eta_rew_start: int = 400
    eta_anneal_steps: int = 2000

    # io
    save_interval: int = 200
    log_interval: int = 10
    max_iterations: int = 100000
    resume_path: str | None = None
    initial_checkpoint: str | None = None
    log_root: str = "runs"


class Runner:
    def __init__(self, env: AcpEnv, cfg: RunnerCfg, device: torch.device, seed: int = 0):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.stage = env.cfg.stage
        self.writer = None
        self.iteration = 0
        self.total_steps = 0
        self.contract = None
        self._resume_rng = None

    # ------------------------------------------------------------------
    def learn(self, actor, critic_m, critic_fs, planner=None, log_dir: str | None = None):
        cfg = self.cfg
        env = self.env
        self.contract = build_checkpoint_contract(env, planner, training_cfg=cfg)
        ppo = PPO(actor, critic_m, critic_fs, cfg.ppo, self.device)
        if cfg.initial_checkpoint is not None:
            initial = torch.load(cfg.initial_checkpoint, map_location=self.device, weights_only=False)
            ppo.log_std.data.copy_(initial["ppo"]["log_std"])
            ppo.constrain_noise()
        storage = RolloutStorage(
            env.num_envs, cfg.num_steps, env.cfg.observation_space, env.cfg.action_space,
            self.device, env.is_ref_env,
        )
        buffers = {"ref": HindsightBuffer(cfg.buffer_cap_steps), "goal": HindsightBuffer(cfg.buffer_cap_steps)}
        planner_opt = None
        if planner is not None:
            planner_opt = torch.optim.Adam(planner.parameters(), lr=cfg.planner_lr, eps=1e-8)

        if cfg.resume_path is not None:
            self.load(Path(cfg.resume_path), ppo, planner, planner_opt, buffers)

        if log_dir is not None:
            self.writer = SummaryWriter(log_dir)
            print(f"[runner] logging to {log_dir}")

        obs_dict, _ = env.reset()
        if self._resume_rng is not None:
            restore_rng_state(self._resume_rng)
            self._resume_rng = None
        obs = obs_dict["policy"]
        planner_aff_hist, planner_occ_hist = deque(maxlen=100), deque(maxlen=100)
        ep_rewards = torch.zeros(env.num_envs, device=self.device)
        ep_lengths = torch.zeros(env.num_envs, device=self.device)
        reference_episodes = deque(maxlen=1000)

        while self.iteration < cfg.max_iterations:
            t_start = time.time()
            storage.step = 0
            n_episodes = 0
            reward_telemetry = {}
            reward_telemetry_steps = 0

            self._set_annealing()
            for t in range(cfg.num_steps):
                if planner is not None:
                    pts, hts, pg = env.get_planner_obs()
                    with torch.no_grad():
                        aff, occ, contacts = planner(pts, hts, pg)
                    env.set_predicted_contacts(contacts, occ)
                    obs = env.refresh_observations()["policy"]
                    planner_aff_hist.append(float(aff.detach().mean()))
                    planner_occ_hist.append(float(occ.detach().mean()))

                with torch.no_grad():
                    vm_in, vfs_in = env.get_critic_obs()
                    vm = critic_m(vm_in)
                    vfs = critic_fs(vfs_in)
                action, log_prob = ppo.act(obs)
                obs2, rew, terminated, truncated, _info = env.step(action)
                terms = _info.get("reward_terms", {})
                if terms:
                    for name, value in terms.items():
                        value = torch.as_tensor(value, device=self.device).detach()
                        if name == "diagnostic/held_joint_error_max_rad":
                            reward_telemetry[name] = torch.maximum(reward_telemetry.get(name, value), value)
                        else:
                            reward_telemetry[name] = reward_telemetry.get(name, 0.0) + value
                    reward_telemetry_steps += 1
                dones = terminated | truncated
                if not torch.isfinite(rew).all() or not torch.isfinite(obs2["policy"]).all():
                    raise FloatingPointError("Nonfinite environment observation or reward")
                learning_rewards = rew.clone()
                timeout = _info.get("timeout_bootstrap")
                if timeout is not None:
                    # Isaac Lab has already auto-reset here. Only these saved
                    # post-physics, pre-reset observations describe the timeout.
                    ids = timeout["env_ids"]
                    with torch.no_grad():
                        final_vm = critic_m(timeout["critic_m"])
                        final_vfs = critic_fs(timeout["critic_fs"])
                        final_value = torch.where(env.is_ref_env[ids], final_vm, final_vfs)
                    learning_rewards[ids] += cfg.ppo.gamma * final_value
                storage.add(obs, action, log_prob, learning_rewards, dones, vm, vfs, vm_in, vfs_in,
                            ppo.action_mean, ppo.action_std)
                obs = obs2["policy"]
                ep_rewards += rew
                ep_lengths += 1
                self.total_steps += env.num_envs

                # drain finished episodes -> curriculum + hindsight buffers
                finished = env.finished_episodes
                env.finished_episodes = []
                if finished:
                    n_episodes += len(finished)
                    env.update_curriculum(finished)
                    reference_episodes.extend(ep for ep in finished if ep["is_ref"])
                    if planner is not None:
                        self._process_episodes(finished, buffers)
                    done_ids = [ep["env_id"] for ep in finished]
                    done_ids = torch.tensor(done_ids, device=self.device)
                    if len(done_ids):
                        self._log_episode_returns(ep_rewards[done_ids], ep_lengths[done_ids], finished)
                        ep_rewards[done_ids] = 0.0
                        ep_lengths[done_ids] = 0

            # ---- GAE
            with torch.no_grad():
                vm_in, vfs_in = env.get_critic_obs()
                last_vm = critic_m(vm_in)
                last_vfs = critic_fs(vfs_in)
            storage.compute_returns(last_vm, last_vfs, cfg.ppo.gamma, cfg.ppo.lam)

            # ---- planner updates (Stages 2-3)
            planner_stats = {}
            if planner is not None and planner_opt is not None:
                planner_stats = self._update_planner(planner, planner_opt, buffers)

            # ---- PPO
            ppo_stats = ppo.update(storage, self.iteration)

            # ---- logging
            if self.writer is not None and (self.iteration % cfg.log_interval == 0
                                            or self.iteration == cfg.max_iterations - 1):
                self.writer.add_scalar("train/iteration_time_s", time.time() - t_start, self.iteration)
                self.writer.add_scalar("train/total_steps", self.total_steps, self.iteration)
                for k, v in ppo_stats.items():
                    self.writer.add_scalar(f"train/{k}", v, self.iteration)
                for k, v in planner_stats.items():
                    self.writer.add_scalar(f"planner/{k}", v, self.iteration)
                if reward_telemetry_steps:
                    keys = list(reward_telemetry)
                    values = torch.stack([
                        reward_telemetry[key] if key == "diagnostic/held_joint_error_max_rad"
                        else reward_telemetry[key] / reward_telemetry_steps for key in keys
                    ])
                    for name, value in zip(keys, values.cpu().tolist()):
                        self.writer.add_scalar(f"reward/{name}", value, self.iteration)
                self.writer.add_scalar("planner/eta_obs", env.cfg.eta_obs, self.iteration)
                self.writer.add_scalar("planner/eta_rew", env.cfg.eta_rew, self.iteration)
                self.writer.add_scalar("buffer/ref_steps", buffers["ref"].size_steps(), self.iteration)
                self.writer.add_scalar("buffer/goal_steps", buffers["goal"].size_steps(), self.iteration)
                csi = getattr(env, "csi_min_phase", None)
                csi_mean = float(csi.mean()) if csi is not None else 0.0
                if csi is not None:
                    self.writer.add_scalar("curriculum/csi_min_phase_mean", csi_mean, self.iteration)
                    self.writer.add_scalar("curriculum/csi_min_phase_min", float(csi.min()), self.iteration)
                    self.writer.add_scalar("curriculum/phase_zero_fraction", float((csi <= 1e-6).float().mean()), self.iteration)
                if reference_episodes:
                    successes = [ep["end_phase"] >= .99 and not ep.get("terminated", True)
                                 and ep.get("root_position_error_m", float("inf")) <= getattr(env.cfg, "csi_success_root_error_m", .3)
                                 for ep in reference_episodes]
                    self.writer.add_scalar("episode/reference_completion_rate", float(np.mean(successes)), self.iteration)
                    self.writer.add_scalar("episode/reference_start_phase", float(np.mean([ep["init_phase"] for ep in reference_episodes])), self.iteration)
                    phase_zero = [ep for ep in reference_episodes if ep["init_phase"] <= 1e-6]
                    self.writer.add_scalar("episode/phase_zero_sample_fraction", len(phase_zero) / len(reference_episodes), self.iteration)
                    if phase_zero:
                        successes_zero = [ep["end_phase"] >= .99 and not ep.get("terminated", True)
                                          and ep.get("root_position_error_m", float("inf")) <= getattr(env.cfg, "csi_success_root_error_m", .3)
                                          for ep in phase_zero]
                        self.writer.add_scalar("episode/phase_zero_completion_rate", float(np.mean(successes_zero)), self.iteration)
                    errors = [ep["root_position_error_m"] for ep in reference_episodes if "root_position_error_m" in ep]
                    if errors:
                        self.writer.add_scalar("episode/terminal_root_error_m", float(np.mean(errors)), self.iteration)
                if planner_aff_hist:
                    self.writer.add_scalar("planner/affordance_mean", float(np.mean(planner_aff_hist)), self.iteration)
                    self.writer.add_scalar("planner/occupancy_mean", float(np.mean(planner_occ_hist)), self.iteration)
                print(
                    f"iter {self.iteration:6d} | t {time.time() - t_start:5.1f}s | "
                    f"eps {n_episodes:5d} | csi {csi_mean:.3f} | pol {ppo_stats['policy_loss']:7.3f} | "
                    f"val {ppo_stats['value_loss']:7.3f} | kl {ppo_stats['kl']:6.4f} | "
                    f"planner {format(planner_stats['loss'], '.3f') if 'loss' in planner_stats else 'n/a':>7} | "
                    f"buffers {buffers['ref'].size_steps()}/{buffers['goal'].size_steps()}"
                )

            if cfg.save_interval > 0 and self.iteration % cfg.save_interval == 0 and log_dir is not None:
                self.save(Path(log_dir) / f"iter_{self.iteration}.pt", ppo, planner, planner_opt, buffers)

            self.iteration += 1

        if log_dir is not None:
            self.save(Path(log_dir) / "final.pt", ppo, planner, planner_opt, buffers,
                      next_iteration=self.iteration)
        if self.writer is not None:
            self.writer.close()
        return ppo

    # ------------------------------------------------------------------
    def _set_annealing(self):
        cfg, env = self.cfg, self.env

        def ramp(u, start):
            if u <= start:
                return 0.0
            if u >= start + cfg.eta_anneal_steps:
                return 1.0
            return (u - start) / cfg.eta_anneal_steps

        if self.stage == 2:
            env.cfg.eta_obs = ramp(self.iteration, cfg.eta_obs_start)
            env.cfg.eta_ref_obs = env.cfg.eta_obs
            env.cfg.eta_rew = ramp(self.iteration, cfg.eta_rew_start)
        elif self.stage == 3:
            env.cfg.eta_obs = 1.0
            env.cfg.eta_ref_obs = 1.0
            env.cfg.eta_rew = 1.0

    def _process_episodes(self, finished: list[dict], buffers: dict):
        """Hindsight-label admitted episodes and add planner samples to buffers."""
        from morph_mimic.envs.acp_env import (
            REC_BASE, REC_BODY, REC_FORCE, REC_GOAL, REC_PRO, REC_PTS, REC_VEL, REC_YAW,
        )
        for ep in finished:
            rec = ep["record"]  # (n, 682)
            n = rec.shape[0]
            if ep["is_ref"]:
                admit = (not ep.get("terminated", False)
                         and ep["init_phase"] <= 0.1 and ep["end_phase"] >= 0.9)
            else:
                admit = ep["reached_goal"]
            if not admit:
                continue
            # truncate goal episodes at the first step within 0.2 m of the goal
            if not ep["is_ref"]:
                base = rec[:, REC_BASE : REC_BASE + 3]
                d_goal = (base[:, :2] - ep["goal_pos"][:2]).norm(dim=-1)
                near = (d_goal < 0.2).nonzero()
                if len(near):
                    n = int(near[0].item()) + 1
                    rec = rec[:n]
            if n < 20:
                continue

            forces = rec[:, REC_FORCE : REC_FORCE + 9]
            body = rec[:, REC_BODY : REC_BODY + 27].view(n, 9, 3)
            base = rec[:, REC_BASE : REC_BASE + 3]
            yaw = rec[:, REC_YAW]
            targets, flags, _ = hindsight_contacts(
                forces, body, base, yaw, horizon=75, min_seg=10, force_thr=1.0
            )
            valid = flags.any()
            if not valid and not ep["is_ref"]:
                continue
            # build planner samples
            points = rec[:, REC_PTS : REC_PTS + 561].view(n, 187, 3)
            heights = points[..., 2].view(n, 17, 11)
            proprio = rec[:, REC_PRO : REC_PRO + 75]
            goal = rec[:, REC_GOAL : REC_GOAL + 3]
            window = torch.zeros(n, 5, 75)
            for t in range(n):
                lo = max(0, t - 4)
                window[t, 5 - (t - lo + 1) :] = proprio[lo : t + 1]
            pg = torch.cat((window.flatten(1), goal), dim=-1)
            quality = 0.0
            if not ep["is_ref"]:
                fz = forces.sum(-1)  # (n,)
                vz = rec[:, REC_VEL + 2]
                power = (fz * vz).clamp(min=0) * 0.02  # dt = 0.02 s
                d0 = (base[0, :2] - ep["goal_pos"][:2]).norm().clamp(min=0.5)
                quality = float(power.sum() / d0)
            tname = self.env.cfg.terrain_types[ep["terrain_type"]]
            key = f"{tname}/tile{ep['terrain_level']}"
            buffers["ref" if ep["is_ref"] else "goal"].add(
                key,
                {
                    "points": points,
                    "heights": heights,
                    "pg": pg,
                    "targets": targets,
                    "flags": flags,
                },
                quality=quality,
            )

    def _update_planner(self, planner, optimizer, buffers: dict):
        cfg = self.cfg
        ref_steps = buffers["ref"].size_steps()
        goal_steps = buffers["goal"].size_steps()
        if ref_steps + goal_steps < cfg.planner_min_entries:
            return {"loss": float("nan")}
        # proportional mixing of the two buffers (paper: proportional to stored counts)
        w_ref = ref_steps / max(1, ref_steps + goal_steps)
        w_goal = 1.0 - w_ref
        losses = []
        for _ in range(cfg.planner_updates_per_iter):
            batch = []
            for buf_name, wgt in (("ref", w_ref), ("goal", w_goal)):
                n = int(cfg.planner_batch * wgt)
                if n <= 0:
                    continue
                s = buffers[buf_name].sample(n, self.device)
                if s is not None:
                    batch.append(s)
            if not batch:
                continue
            b = {k: torch.cat([x[k] for x in batch], dim=0) for k in ("points", "heights", "pg", "targets", "flags")}
            sub = max(1, b["points"].shape[0] // cfg.planner_grad_sub_batches)
            normalization = (int(b["flags"].sum()), len(b["points"]))
            iteration_loss = 0.0
            optimizer.zero_grad()
            for i in range(0, b["points"].shape[0], sub):
                pts = b["points"][i : i + sub]
                hts = b["heights"][i : i + sub]
                pg = b["pg"][i : i + sub]
                tg = b["targets"][i : i + sub]
                fl = b["flags"][i : i + sub]
                aff, occ, _ = planner(pts, hts, pg)
                p_star = gaussian_targets(pts, tg, fl, sigma_a=planner.sigma_a)
                loss, terms = planner_loss(aff, occ, p_star, fl, normalization=normalization)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite planner loss")
                loss.backward()
                iteration_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if not torch.stack([torch.isfinite(p).all() for p in planner.parameters()]).all():
                raise FloatingPointError("Nonfinite planner parameters")
            losses.append(iteration_loss)
        return {"loss": float(np.mean(losses)) if losses else float("nan"),
                "affordance_loss": float(np.mean([x for x in losses]))}

    def _log_episode_returns(self, rewards, lengths, finished):
        if self.writer is None:
            return
        self.writer.add_scalar("episode/mean_return", float(rewards.mean()), self.iteration)
        self.writer.add_scalar("episode/mean_length", float(lengths.float().mean()), self.iteration)

    # ------------------------------------------------------------------
    def save(self, path: Path, ppo: PPO, planner, planner_opt, buffers: dict,
             next_iteration: int | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema": 2,
            "contract": self.contract,
            "seed": self.seed,
            "rng_state": capture_rng_state(),
            "curriculum": {key: getattr(self.env, key).detach().cpu().clone()
                           for key in ("csi_min_phase", "terrain_level", "success_ema")
                           if hasattr(self.env, key)},
            "iteration": self.iteration,
            "next_iteration": self.iteration + 1 if next_iteration is None else next_iteration,
            "stage": self.stage,
            "robot": getattr(getattr(self.env, "robot_spec", None), "name", None),
            "total_steps": self.total_steps,
            "ppo": {
                "actor": ppo.actor.state_dict(),
                "critic_m": ppo.critic_m.state_dict(),
                "critic_fs": ppo.critic_fs.state_dict(),
                "log_std": ppo.log_std.detach(),
                "opt_main": ppo.opt_main.state_dict(),
                "opt_fs": ppo.opt_fs.state_dict(),
                "actor_lr": ppo.actor_lr,
            },
            "planner": planner.state_dict() if planner is not None else None,
            "planner_opt": planner_opt.state_dict() if planner_opt is not None else None,
        }
        # Publish an immutable replay generation first, then atomically commit
        # the checkpoint that names and hashes it. A failed write cannot pair
        # old network weights with a new buffer under the same filename.
        if planner is not None:
            bpath = path.with_name(f"{path.stem}.buffers.{uuid.uuid4().hex}.pt")
            atomic_torch_save({"ref": buffers["ref"].parts, "goal": buffers["goal"].parts}, bpath)
            state["buffer_file"] = bpath.name
            state["buffer_sha256"] = file_hash(bpath)
        atomic_torch_save(state, path)
        print(f"[runner] saved checkpoint to {path}")

    def load(self, path: Path, ppo: PPO, planner, planner_opt, buffers: dict):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("stage") not in (None, self.stage):
            raise ValueError("same-stage resume requires a checkpoint from the current stage")
        robot = getattr(getattr(self.env, "robot_spec", None), "name", None)
        if state.get("robot") not in (None, robot):
            raise ValueError("checkpoint belongs to a different robot")
        validate_checkpoint_contract(state, self.env, planner, mode="resume", training_cfg=self.cfg)
        b = None
        if state.get("buffer_file") is not None:
            filename = Path(state["buffer_file"])
            if filename.name != str(filename):
                raise ValueError("Checkpoint replay path must be a filename in its own directory")
            bpath = path.parent / filename
            if not bpath.is_file() or file_hash(bpath) != state.get("buffer_sha256"):
                raise ValueError("Checkpoint replay buffer is missing or its SHA256 differs")
            b = torch.load(bpath, map_location="cpu", weights_only=False)
        elif planner is not None:
            bpath = path.with_suffix(".buffers.pt")
            if not bpath.is_file():
                raise ValueError("Planner resume requires its replay buffer; it cannot be silently discarded")
            b = torch.load(bpath, map_location="cpu", weights_only=False)
        self.iteration = state.get("next_iteration", state["iteration"] + 1)
        self.total_steps = state["total_steps"]
        ppo.actor.load_state_dict(state["ppo"]["actor"])
        ppo.critic_m.load_state_dict(state["ppo"]["critic_m"])
        ppo.critic_fs.load_state_dict(state["ppo"]["critic_fs"])
        ppo.log_std.data.copy_(state["ppo"]["log_std"])
        ppo.constrain_noise()
        for key, opt in (("opt_main", ppo.opt_main), ("opt_fs", ppo.opt_fs)):
            if key in state["ppo"]:
                opt.load_state_dict(state["ppo"][key])
        ppo.actor_lr = float(state["ppo"].get("actor_lr", self.cfg.ppo.init_lr))
        for group in ppo.opt_main.param_groups:
            group["lr"] = ppo.actor_lr
        if planner is not None and state.get("planner") is not None:
            planner.load_state_dict(state["planner"])
            if state.get("planner_opt") is not None:
                planner_opt.load_state_dict(state["planner_opt"])
        if b is not None:
            buffers["ref"].parts = b["ref"]
            buffers["goal"].parts = b["goal"]
        for key, value in state.get("curriculum", {}).items():
            target = getattr(self.env, key, None)
            if target is None or target.shape != value.shape:
                raise ValueError(f"Checkpoint curriculum layout changed: {key}")
            target.copy_(value.to(target.device))
        self._resume_rng = state.get("rng_state")
        print(f"[runner] resumed from {path} at iteration {self.iteration}")
