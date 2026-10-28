"""Independent world-position teacher with physical box-top hold acceptance.

This is an engineering tracking experiment, not finite-observation ACP success.
The original teacher remains immutable and is refused as a resume checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
import json
import random
from types import SimpleNamespace

import numpy as np
import torch

from morph_mimic.scripts import train_reference_teacher as v1
from morph_mimic.scripts.train_until_validated import PROJECT_ROOT, CRITERIA, Supervisor, _sha256, acceptance
from morph_mimic.utils.evaluation import TrackingTrajectory

SCRIPT = "morph_mimic/scripts/train_reference_teacher_v2.py"
TASK_SCRIPT = "morph_mimic/utils/box_task.py"
HOLD_STEPS = 50
HOLD_SECONDS = 1.0
EXTRA_REWARDS = {
    "world_root_xyz": {"bandwidth_m": .3, "weight": 3.},
    "world_root_z": {"bandwidth_m": .15, "weight": 3.},
    "world_two_feet": {"bandwidth_m": .15, "weight": 2., "reduction": "mean of two per-foot exponentials"},
}
TEACHER_CONTRACT = deepcopy(v1.TEACHER_CONTRACT)
TEACHER_CONTRACT.update(
    experiment="privileged_world_tracking_box_hold_teacher_v2",
    reward={"base": "unchanged ACP Stage1 rewards and weights", "additive": EXTRA_REWARDS,
            "activation": "always on for reference environments, independent of chi_mov", "integration": "multiply by control_dt once"},
    end_hold={"seconds": HOLD_SECONDS, "control_endpoints": HOLD_STEPS,
              "reference": "clamp phase at one; hold final pose with zero reference velocities",
              "physical_reset": "only after 50 additional physical endpoints or original physical failure/timeout",
              "acceptance": "all 50 contiguous post-reference endpoints pass the hashed box task geometry/current-force gate",
              "force_scope": "Current positive normal Fz for the nine named contact bodies only; uninstrumented torso/hip links and friction support are not excluded",
              "mid_motion_phase": "fixed physical control time, no contact-dependent phase freeze"},
)


def additive_world_rewards(actual_root, reference_root, actual_feet, reference_feet):
    """Three bounded terms; no movement-speed gate can disable translation."""
    if (actual_root.shape != reference_root.shape or actual_root.ndim != 2 or actual_root.shape[1] != 3
            or actual_feet.shape != reference_feet.shape or actual_feet.shape != (len(actual_root), 2, 3)):
        raise ValueError("World rewards require matched [E,3] roots and [E,2,3] feet")
    return {
        "world_root_xyz": torch.exp(-(actual_root-reference_root).square().sum(-1)/.3**2),
        "world_root_z": torch.exp(-(actual_root[:, 2]-reference_root[:, 2]).square()/.15**2),
        "world_two_feet": torch.exp(-(actual_feet-reference_feet).square().sum(-1)/.15**2).mean(-1),
    }


def phase_and_hold(init_phase, controls, intervals):
    """Count only controls strictly after the endpoint, excluding endpoint 500."""
    phase = (init_phase + controls/intervals).clamp(max=1.)
    reference_controls = torch.ceil((1-init_phase)*intervals-1e-5)
    hold_controls = (controls-reference_controls).clamp(min=0).long()
    return phase, hold_controls


def teacher_environment_class(base, task_spec):
    parent = v1.teacher_environment_class(base)

    class BoxHoldTeacherEnv(parent):
        box_task_spec = task_spec

        def _pre_physics_step(self, actions):
            super()._pre_physics_step(actions)
            intervals = (self.clip_duration/self.step_dt).round().clamp(min=1)
            self.phase[:], self._box_hold_controls = phase_and_hold(
                self.init_phase, (self.episode_length_buf+1).float(), intervals)
            self._update_ref_contact_flags()

        def _ref_sample(self, env_ids=None):
            ref = super()._ref_sample(env_ids)
            phase = self.phase if env_ids is None else self.phase[env_ids]
            hold = phase >= 1.
            # Use actual final poses, rather than extrapolating velocity past
            # the motion. This changes reference commands only, never physics.
            if getattr(self, "motion_data", None) is not None:
                clip = next(iter(self.motion_data.clips.values()))
                origin = self.env_origin()+self.cell_offset
                if env_ids is not None:
                    origin = origin[env_ids]
                first = clip.root_pos[0].clone()
                first[2] = 0.
                final = {"root_pos": origin+clip.root_pos[-1]-first,
                    "root_quat": clip.root_quat[-1].expand(len(phase), -1),
                    "joint_pos": clip.joint_pos[-1].expand(len(phase), -1)}
                for name, value in final.items():
                    ref[name] = torch.where(hold[:, None], value, ref[name])
            for name in ("root_lin_vel", "root_ang_vel", "joint_vel"):
                ref[name] = torch.where(hold[:, None], torch.zeros_like(ref[name]), ref[name])
            return ref

        def _get_rewards(self):
            original = super()._get_rewards()
            ref = self._ref_sample()
            terms = additive_world_rewards(self.robot.data.root_pos_w, ref["root_pos"],
                self._contact_body_positions_w()[:, 6:8], self._ref_contact_pos(False)[:, 6:8])
            added = torch.zeros_like(original)
            telemetry = self.extras["reward_terms"]
            for name, value in terms.items():
                contribution = value * EXTRA_REWARDS[name]["weight"] * self.is_ref_env.float()
                added += contribution
                telemetry[f"raw/{name}"] = value.detach().mean()
                telemetry[f"weighted/{name}"] = contribution.detach().mean()*self.step_dt
            telemetry["diagnostic/box_hold_fraction"] = (self.phase >= 1.).float().mean().detach()
            metric = getattr(self, "_box_task_metrics", None)
            if metric is not None:
                active = (self._box_hold_controls > 0).float()
                denominator = active.sum().clamp(min=1.)
                telemetry["diagnostic/box_task_gate_fraction"] = metric["task_gate"].float().mean().detach()
                telemetry["diagnostic/box_hold_task_gate_fraction"] = (metric["task_gate"].float()*active).sum().detach()/denominator
                telemetry["diagnostic/box_root_height_error_mean_m"] = metric["rootheight_error_m"].mean().detach()
                telemetry["diagnostic/box_foot_support_margin_mean_m"] = metric["foot_support_margin_m"].mean().detach()
                telemetry["diagnostic/box_hold_completed_and_passed_fraction"] = (
                    (self._box_hold_controls >= HOLD_STEPS) & (self._box_task_hold_pass_count == HOLD_STEPS)).float().mean().detach()
            return original + added*self.step_dt

        def _get_dones(self):
            terminated, _ = super()._get_dones()
            hold = getattr(self, "_box_hold_controls", torch.zeros_like(self.episode_length_buf))
            complete = self.is_ref_env & (hold >= HOLD_STEPS)
            timeout = self.episode_length_buf >= self.max_episode_length-1
            truncated = timeout | complete
            from morph_mimic.utils.box_task import task_metrics
            data = self.robot.data
            origin = self.env_origin()+self.cell_offset
            feet = self._contact_body_positions_w()[:, 6:8]
            foot_quat = data.body_quat_w[:, self.contact_link_idx[6:8]]
            metric = task_metrics(data.root_pos_w-origin, data.root_quat_w,
                feet-origin[:, None], foot_quat, self._contact_forces_w(), self.box_task_spec)
            self._box_task_metrics = metric
            if not hasattr(self, "_box_task_hold_pass_count"):
                self._box_task_hold_pass_count = torch.zeros_like(hold)
            self._box_task_hold_pass_count += ((hold > 0) & metric["task_gate"]).long()
            if self.cfg.evaluation:
                snapshot = self.extras["final_tracking"]
                snapshot.update(truncated=truncated.detach().clone(), reached_end=complete.detach().clone(),
                    box_hold_controls=hold.detach().clone(), foot_pos_w=feet.detach().clone(),
                    foot_quat_w=foot_quat.detach().clone(),
                    **{f"box_task_{key}": value.detach().clone() for key, value in metric.items()})
            return terminated, truncated

        def _finish_episodes(self, env_ids):
            before = len(self.finished_episodes)
            super()._finish_episodes(env_ids)
            if hasattr(self, "_box_task_hold_pass_count"):
                for episode in self.finished_episodes[before:]:
                    env_id = episode["env_id"]
                    controls = int(self._box_hold_controls[env_id])
                    passed = int(self._box_task_hold_pass_count[env_id])
                    episode.update(box_task_hold_endpoints=controls, box_task_passing_hold_endpoints=passed,
                        box_task_passed=bool(controls >= HOLD_STEPS and passed == HOLD_STEPS and not episode["terminated"]))

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if hasattr(self, "_box_hold_controls"):
                if env_ids is None:
                    self._box_hold_controls.zero_()
                else:
                    self._box_hold_controls[env_ids] = 0
            if hasattr(self, "_box_task_hold_pass_count"):
                if env_ids is None:
                    self._box_task_hold_pass_count.zero_()
                else:
                    self._box_task_hold_pass_count[env_ids] = 0

    return BoxHoldTeacherEnv


TASK_FIELDS = {
    "rootheight_error_m": (), "root_tilt_rad": (), "foot_origin_margin_m": (2,),
    "foot_support_margin_m": (2,), "foot_sole_plane_error_m": (2,),
    "feet_current_positive_fz_n": (2,), "nonfoot_current_positive_fz_max_n": (),
    "foot_support_center_pos_rebased_m": (2, 3),
    "task_geometry_gate": (), "task_gate": (), "valid_state_gate": (),
    "root_height_gate": (), "root_tilt_gate": (), "foot_support_margin_gate": (),
    "foot_sole_plane_gate": (), "feet_current_support_gate": (), "nonfoot_current_support_gate": (),
}


def hold_summary(arrays):
    counts = np.asarray(arrays["box_hold_controls"])
    selected = np.flatnonzero(counts > 0)
    complete = (len(selected) == HOLD_STEPS and
        np.array_equal(counts[selected], np.arange(1, HOLD_STEPS+1)) and
        np.array_equal(selected, np.arange(len(counts)-HOLD_STEPS, len(counts))) and
        np.all(np.asarray(arrays["phase"])[selected] >= 1.))
    passed_steps = int(np.asarray(arrays["box_task_task_gate"])[selected].sum())
    return dict(passed=bool(complete and passed_steps == HOLD_STEPS), hold_complete=bool(complete),
        required_hold_endpoints=HOLD_STEPS, observed_hold_endpoints=int(len(selected)),
        passing_hold_endpoints=passed_steps,
        geometry_passing_hold_endpoints=int(np.asarray(arrays["box_task_task_geometry_gate"])[selected].sum()),
        sampling="Post-physics pre-reset endpoints; first phase-one endpoint excluded from the extra physical hold",
        final_values={key: np.asarray(arrays[f"box_task_{key}"])[-1].tolist() for key in TASK_FIELDS})


class BoxTaskTrajectory(TrackingTrajectory):
    _fields = TrackingTrajectory._fields | {"foot_pos_w": (2, 3), "foot_quat_w": (2, 4), "box_hold_controls": ()} | {
        f"box_task_{key}": shape for key, shape in TASK_FIELDS.items()}

    def summary(self):
        return super().summary() | {"box_task": hold_summary(self.arrays())}

    def write(self, path, *, metadata):
        metadata = deepcopy(metadata)
        metadata["result"].update(expected_control_steps=550, physical_hold_required_s=HOLD_SECONDS)
        metadata.update(reference_teacher=TEACHER_CONTRACT, box_task=self.box_task_spec)
        return super().write(path, metadata=metadata)


def augment_contract(contract, task_spec):
    result = v1.augment_contract(contract)
    result["observations"].update(policy_schema="ACP468_plus_privileged_reference116_box_hold_teacher_v2",
        reference_teacher=deepcopy(TEACHER_CONTRACT), box_task=deepcopy(task_spec))
    result["physics"]["joint_target_semantics"] = TEACHER_CONTRACT["action"]
    return result


def require_teacher_checkpoint(checkpoint):
    contract = checkpoint.get("contract") or {}
    obs = contract.get("observations", {})
    if (checkpoint.get("schema") != 2 or checkpoint.get("robot") != "g1" or checkpoint.get("stage") != 1
            or obs.get("reference_teacher") != TEACHER_CONTRACT or not isinstance(obs.get("box_task"), dict)
            or obs.get("policy_dim") != v1.OBS_DIM or obs.get("action_dim") != 23
            or contract.get("physics", {}).get("joint_target_semantics") != TEACHER_CONTRACT["action"]):
        raise ValueError("Requires v2 world-tracking/physical-box-hold teacher contract; v1/ACP/standing checkpoints refused")


def _install_runtime(args):
    # All class/build/contract extensions belong to this process, never source files.
    from morph_mimic.envs import acp_env
    from morph_mimic.scripts import common
    from morph_mimic.utils import checkpoint
    from morph_mimic.rl import runner
    from morph_mimic.utils.box_task import load_box_task_spec

    task_spec = load_box_task_spec(args.motion_dir/f"{args.clips[0]}.npz")
    acp_env.AcpEnv = teacher_environment_class(acp_env.AcpEnv, task_spec)
    original_build = common.build_env
    def build(*build_args, **kwargs):
        kwargs.update(use_usd=False, training_profile="nominal_pretrain", held_joint_mode="hard_limits_at_nominal",
            observation_space=v1.OBS_DIM, episode_length_s=12.,
            csi_enabled=not kwargs.get("evaluation", False), csi_sampling_mode="teacher_stratified_RSI_v1")
        env, spec = original_build(*build_args, **kwargs)
        if (env.cfg.stage != 1 or env.cfg.ref_env_ratio != 1 or len(env.motion_data.clip_names) != 1
                or not env.motion_data.scene_bindings or bool(env.action_delay.any()) or abs(env.step_dt-.02)>1e-9
                or any(abs(float(clip.duration_s)-10.) > .01 for clip in env.motion_data.clips.values())):
            env.close()
            raise ValueError("V2 teacher requires a nominal zero-delay 10s bound box clip at 50Hz")
        return env, spec
    common.build_env = build
    original_contract = checkpoint.build_checkpoint_contract
    def build_contract(*build_args, **kwargs):
        return augment_contract(original_contract(*build_args, **kwargs), task_spec)
    checkpoint.build_checkpoint_contract = runner.build_checkpoint_contract = build_contract
    return task_spec


def _train(args):
    from morph_mimic.scripts import common
    from morph_mimic.launcher import launch, close
    if args.resume is not None:
        require_teacher_checkpoint(torch.load(args.resume, map_location="cpu", weights_only=False))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    app, env = launch(headless=True, device=args.device), None
    try:
        _install_runtime(args)
        from morph_mimic.networks.tracker import Critic, TrackerActor
        from morph_mimic.rl.runner import Runner
        values = vars(args) | dict(continue_run=args.resume is not None,
            resume=str(args.resume) if args.resume else None, save_interval=args.chunk_iterations)
        cfg = common.configure_runner(SimpleNamespace(**values), 1)
        env, _ = common.build_env(1, "g1", args.num_envs, device=args.device, seed=args.seed,
            motion_dir=str(args.motion_dir), clips=args.clips)
        actor = TrackerActor(obs_dim=v1.OBS_DIM).to(args.device)
        critics = (Critic(353).to(args.device), Critic(298).to(args.device))
        Runner(env, cfg, torch.device(args.device), seed=args.seed).learn(actor, *critics, log_dir=str(args.run_dir))
    finally:
        if env is not None:
            env.close()
        close(app)


def task_acceptance(report):
    passed, reasons = acceptance(report)
    task = report["motions"][0].get("physical_diagnostics", {}).get("box_task", {})
    if report.get("task_passed") is not True or task.get("passed") is not True:
        reasons.append("physical_box_top_hold: all 50 consecutive extra physical endpoints must pass")
    return not reasons, reasons


def _evaluate(args):
    from morph_mimic.scripts import evaluate
    from morph_mimic.networks.tracker import TrackerActor
    from morph_mimic.launcher import launch, close
    from morph_mimic.utils.specs import RobotSpec
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    require_teacher_checkpoint(checkpoint)
    actor = TrackerActor(obs_dim=v1.OBS_DIM).to(args.device).eval()
    actor.load_state_dict(checkpoint["ppo"]["actor"])
    metadata = evaluate._clip_metadata(args.motion_dir, args.clips, RobotSpec.load("g1"))
    values = vars(args) | dict(robot="g1", stage=1, planner="acp", max_steps=552,
        start_phase=0., held_joint_mode="hard_limits_at_nominal")
    app = launch(headless=True, device=args.device)
    try:
        task_spec = _install_runtime(args)
        BoxTaskTrajectory.box_task_spec = task_spec
        evaluate.TrackingTrajectory = BoxTaskTrajectory
        result = evaluate._evaluate_clip(SimpleNamespace(**values), checkpoint, metadata[0], actor, None, app)
        result.update(expected_control_steps=550, physical_hold_required_s=HOLD_SECONDS)
        task = result["physical_diagnostics"]["box_task"]
        report = dict(robot="g1", stage=1, checkpoint=str(args.resume), checkpoint_sha256=_sha256(args.resume),
            scope=TEACHER_CONTRACT["scope"], reference_teacher=TEACHER_CONTRACT, box_task=task_spec,
            start_phase=0., evaluation_is_full=True, evaluation_is_partial=False,
            contract_verified=result["contract_verified"], expected_clips=1, evaluated_clips=1,
            completion_rate=float(result["completed"]), failures=int(result["failure"]), motions=[result],
            task_passed=task["passed"], completion_definition="Full phase-zero reference followed by 50 extra physical box-top hold endpoints")
        report["teacher_passed"], report["teacher_failure_reasons"] = task_acceptance(report)
        evaluate._write_report(args.out, report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        close(app)


class TeacherSupervisorV2(v1.TeacherSupervisor):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.status.update(reference_teacher=TEACHER_CONTRACT,
            success_definition="Three distinct checkpoints pass unchanged full-start tracking gates and all 50 physical box-top hold endpoints")

    def _sources(self, *, initialize=False):
        super()._sources(initialize=initialize)
        if initialize:
            for source in (SCRIPT, TASK_SCRIPT):
                self.status["source_sha256"].setdefault(source, _sha256(PROJECT_ROOT/source))
            super()._sources()

    def _command(self, mode):
        command = super()._command(mode)
        command[2] = "morph_mimic.scripts.train_reference_teacher_v2"
        return command

    def _evaluate(self, checkpoint, metadata, digest):
        iteration = metadata["next_iteration"]
        directory = self.args.run_dir/"evaluations"
        directory.mkdir(parents=True, exist_ok=True)
        report_path, trajectory = [directory/f"iter{iteration:06d}.{suffix}" for suffix in ("json", "npz")]
        self._child(self._command("evaluate") + ["--resume", str(checkpoint), "--out", str(report_path),
            "--trajectory", str(trajectory)], f"evaluate_iter{iteration:06d}.log", phase="evaluate")
        report = json.loads(report_path.read_text())
        if (report.get("checkpoint_sha256") != digest or report["motions"][0].get("clip") != self.args.clips[0]
                or not trajectory.is_file() or _sha256(trajectory) != report["motions"][0].get("trajectory_sha256")
                or report.get("reference_teacher") != TEACHER_CONTRACT):
            raise ValueError("V2 teacher evaluation checkpoint/clip/hold semantics/physical trajectory differ")
        passed, reasons = task_acceptance(report)
        if any(value["checkpoint_sha256"] == digest for value in self.status["evaluations"]):
            raise ValueError("A checkpoint cannot count twice toward v2 acceptance")
        self.status["consecutive_successes"] = self.status["consecutive_successes"]+1 if passed else 0
        self.status["evaluations"].append(dict(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            report=str(report_path), trajectory=str(trajectory), actual_total_steps=metadata["total_steps"],
            passed=passed, reasons=reasons, task_passed=report["task_passed"],
            box_task=report["motions"][0]["physical_diagnostics"]["box_task"],
            metrics={key: report["motions"][0][key] for key in CRITERIA},
            consecutive_successes=self.status["consecutive_successes"]))
        self.save()


def main():
    args = v1.arguments()
    if args.mode == "train":
        _train(args)
    elif args.mode == "evaluate":
        _evaluate(args)
    else:
        args.robot, args.training_profile, args.held_joint_mode = "g1", "nominal_pretrain", "hard_limits_at_nominal"
        args.required_successes, args.wait_pid, args.wait_min_iteration, args.resume_checkpoint = 3, None, 0, args.resume
        print(json.dumps(TeacherSupervisorV2(args).run(), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
