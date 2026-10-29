"""Bridge a frozen reference teacher to the finite-observation ACP controller.

DAgger is an engineering warm start, not an ACP paper stage. Every actuator
command uses nominal+0.25*action; only training queries a 584-feature teacher.
Student-only evaluation never loads or calls that teacher. A separate PPO run
uses fresh value networks/optimizer, paper CSI, and the base ACP rewards.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from morph_mimic.networks.student import ARCHITECTURE_CONTRACT, PaperCritic, PaperTrackerActor
from morph_mimic.scripts import train_reference_teacher as v1
from morph_mimic.scripts import train_reference_teacher_v3 as v3
from morph_mimic.scripts.train_until_validated import CRITERIA, PROJECT_ROOT, SOURCE_FILES, Supervisor, _atomic_json
from morph_mimic.utils.checkpoint import (
    atomic_torch_save, build_checkpoint_contract, capture_rng_state, file_hash,
    restore_rng_state, validate_checkpoint_contract,
)
from morph_mimic.utils.student_curriculum import (
    beta_schedule, student_to_teacher_action, teacher_observation, teacher_to_student_action,
)

SCRIPT = "morph_mimic/scripts/train_acp_student.py"
STUDENT_CONTRACT = dict(schema=1, name="finite_contact_student_v1", actor_observations=468,
    actor_actions=23, architecture=deepcopy(ARCHITECTURE_CONTRACT),
    action="q_target=q_nominal+0.25*clipped_student_action",
    actor_reference_kinematics=False, actor_reference_phase=False,
    history="five actual proprioceptive frames with executed nominal-offset actions",
    expert_history="separate five-frame equivalent reference-residual action history",
    expert_label="(post_advance_reference_q-q_nominal)/0.25+clipped_teacher_residual",
    evaluation="student-only; no teacher, critics, blend, or reference action baseline",
    scope="Single-clip finite actor bridge; DAgger/hold gates are engineering extensions, not full ACP reproduction")
DEFAULT_TEACHER = PROJECT_ROOT/"runs/g1_box_climb_contact_teacher_v3_1024/checkpoints/iter006000.pt"


def source_hashes():
    return {name: file_hash(PROJECT_ROOT/name) for name in SOURCE_FILES} | v3.source_hashes() | {name: file_hash(PROJECT_ROOT/name) for name in
        (v1.SCRIPT, v3.v2.SCRIPT, v3.v2.TASK_SCRIPT, SCRIPT,
         "morph_mimic/networks/student.py", "morph_mimic/utils/student_curriculum.py")}


def augment_contract(contract, task, surface, args):
    result = v3.augment_contract(contract, task, surface)
    result["observations"].pop("reference_teacher", None)
    result["observations"].update(policy_dim=468, action_dim=23,
        policy_schema="ACP468_nominal_three_ELU_student_v1", student_curriculum=deepcopy(STUDENT_CONTRACT))
    result["physics"]["joint_target_semantics"] = STUDENT_CONTRACT["action"]
    result["training"]["student_reward_profile"] = args.reward_profile
    result["training"]["student_curriculum_phase"] = args.phase
    return result


def require_student_checkpoint(value):
    obs = (value.get("contract") or {}).get("observations", {})
    if (value.get("schema") != 2 or value.get("robot") != "g1" or value.get("stage") != 1
            or obs.get("student_curriculum") != STUDENT_CONTRACT or obs.get("policy_dim") != 468
            or obs.get("action_dim") != 23 or "reference_teacher" in obs
            or value["contract"].get("physics", {}).get("joint_target_semantics") != STUDENT_CONTRACT["action"]):
        raise ValueError("Requires the finite 468D nominal-action student and three-hidden-ELU contract")


def input_hashes(args):
    paths = (args.teacher, args.motion_dir/f"{args.clips[0]}.npz",
             args.motion_dir/f"{args.clips[0]}_contacts.npz", args.motion_dir/f"{args.clips[0]}.scene.json")
    return {str(path): file_hash(path) for path in paths}


def require_expert_validation(path):
    report = path.parent.parent/"evaluations"/f"{path.stem}.json"
    if not report.is_file():
        raise ValueError("Expert needs its paired full-start validation report")
    value = json.loads(report.read_text())
    if (value.get("checkpoint_sha256") != file_hash(path) or value.get("teacher_passed") is not True
            or not v3.task_acceptance(value)[0]):
        raise ValueError("Expert checkpoint has not passed its bound full-start physical acceptance")


def require_ppo_admission(path, actor_path):
    value = json.loads(path.read_text())
    evaluations = value.get("evaluations", [])[-3:]
    if (value.get("state") != "validated" or value.get("consecutive_successes", 0) < 3
            or value.get("student_curriculum") != STUDENT_CONTRACT or len(evaluations) != 3
            or len({item.get("checkpoint_sha256") for item in evaluations}) != 3
            or evaluations[-1].get("checkpoint_sha256") != file_hash(actor_path)):
        raise ValueError("PPO admission requires three distinct validated student-only checkpoints")
    for item in evaluations:
        report = json.loads(Path(item["report"]).read_text())
        if (item.get("passed") is not True or item.get("teacher_beta_during_training") != 0.
                or report.get("teacher_loaded") is not False or report.get("critics_loaded") is not False
                or report.get("teacher_beta") != 0. or report.get("student_curriculum") != STUDENT_CONTRACT
                or report.get("checkpoint_sha256") != item["checkpoint_sha256"]
                or not v3.task_acceptance(report)[0]):
            raise ValueError("PPO admission includes an assisted, partial, or failed student evaluation")


def student_environment_class(base, task, motion, *, sampler, reward_profile):
    parent = v3.teacher_environment_class(base, task, motion)

    class StudentEnv(parent):
        # v1 calls this virtual method after constructing the ordinary 468 input.
        def _teacher_observation(self, observation):
            if observation.shape != (self.num_envs, 468):
                raise ValueError("Student policy must receive exactly 468 observations")
            return observation

        def _sample_csi_initial_phase(self, ids):
            if sampler == "paper_csi":
                return base._sample_csi_initial_phase(self, ids)
            return super()._sample_csi_initial_phase(ids)

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if hasattr(self, "_expert_history"):
                if ids is None:
                    self._expert_history.zero_()
                    self._executed_expert_residual.zero_()
                else:
                    self._expert_history[ids] = 0.
                    self._executed_expert_residual[ids] = 0.

        def _get_observations(self, record=True):
            observation = super()._get_observations(record=record)
            if not hasattr(self, "_expert_history"):
                self._expert_history = torch.zeros((self.num_envs, 5, 23), device=self.device)
                self._executed_expert_residual = torch.zeros((self.num_envs, 23), device=self.device)
            self._expert_history[:, :-1] = self._expert_history[:, 1:].clone()
            self._expert_history[:, -1] = self._executed_expert_residual
            return observation

        @torch.no_grad()
        def query_expert(self, observation, teacher, beta):
            if not 0. <= beta <= 1.:
                raise ValueError("Teacher mixture must be in [0,1]")
            features = v1.reference_features(self)
            expert_obs = teacher_observation(observation, features, self._expert_history)
            self._pending_expert = teacher(expert_obs).detach()
            self._expert_beta = beta

        def _pre_physics_step(self, actions):
            # Advance phase FIRST. The expert observed the pre-step state, but
            # its executed PD target uses the reference at this new phase.
            super()._pre_physics_step(actions)
            reference = self._ref_sample()["joint_pos"]
            nominal = self.default_pose[self.controlled_idx]
            pending = getattr(self, "_pending_expert", None)
            if pending is not None:
                self._expert_label = teacher_to_student_action(reference, nominal, pending,
                    scale=self.cfg.action_scale, teacher_clip=self.cfg.action_clip,
                    student_clip=self.cfg.action_clip).detach()
                beta = self._expert_beta
                selected = (torch.ones(self.num_envs, 1, device=self.device, dtype=torch.bool) if beta == 1.
                    else torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool) if beta == 0.
                    else torch.rand(self.num_envs, 1, device=self.device) < beta)
                self.actions = torch.where(selected, self._expert_label, self.actions)
                self._pending_expert = None
            self._executed_expert_residual = student_to_teacher_action(
                self.actions, reference, nominal, scale=self.cfg.action_scale).detach()

        def _apply_action(self):
            # Never inherit the teacher's q_reference actuator baseline.
            return base._apply_action(self)

        def _get_rewards(self):
            return base._get_rewards(self) if reward_profile == "paper_base" else super()._get_rewards()

    return StudentEnv


def install_runtime(args):
    from morph_mimic.envs import acp_env
    from morph_mimic.scripts import common
    from morph_mimic.utils import checkpoint
    from morph_mimic.rl import runner
    from morph_mimic.utils.box_task import load_box_task_spec
    from morph_mimic.utils.contact_surface import load_contact_surface_spec

    motion = args.motion_dir/f"{args.clips[0]}.npz"
    task, surface = load_box_task_spec(motion), load_contact_surface_spec(motion)
    base = acp_env.AcpEnv
    sampler = "stratified" if args.phase == "dagger" else "paper_csi"
    acp_env.AcpEnv = student_environment_class(base, task, motion, sampler=sampler, reward_profile=args.reward_profile)
    original_build = common.build_env
    def build(*positional, **kwargs):
        kwargs.update(use_usd=False, training_profile="nominal_pretrain",
            held_joint_mode="hard_limits_at_nominal", observation_space=468, episode_length_s=13.,
            csi_enabled=not kwargs.get("evaluation", False),
            csi_sampling_mode="curriculum_phase" if sampler == "paper_csi" else "student_dagger_stratified_v1")
        env, spec = original_build(*positional, **kwargs)
        if (env.cfg.stage != 1 or env.cfg.ref_env_ratio != 1. or len(env.motion_data.clip_names) != 1
                or not env.motion_data.scene_bindings or bool(env.action_delay.any())
                or abs(env.cfg.action_scale-.25) > 1e-9 or abs(env.step_dt-.02) > 1e-9):
            env.close()
            raise ValueError("Student bridge requires one nominal zero-delay bound G1 scene at 50 Hz")
        v3.ContactTrajectory.tracking, v3.ContactTrajectory.box_task_spec = env.tracking, task
        return env, spec
    common.build_env = build
    original_contract = checkpoint.build_checkpoint_contract
    def contract(*positional, **kwargs):
        return augment_contract(original_contract(*positional, **kwargs), task, surface, args)
    checkpoint.build_checkpoint_contract = runner.build_checkpoint_contract = contract
    return task


class Replay:
    def __init__(self, capacity, device):
        self.observations = torch.empty(capacity, 468, device=device)
        self.actions = torch.empty(capacity, 23, device=device)
        self.size, self.position, self.capacity = 0, 0, capacity

    def add(self, observation, action):
        if len(observation) > self.capacity:
            observation, action = observation[-self.capacity:], action[-self.capacity:]
        indices = (torch.arange(len(observation), device=observation.device)+self.position) % self.capacity
        self.observations[indices], self.actions[indices] = observation.detach(), action.detach()
        self.position = (self.position+len(observation)) % self.capacity
        self.size = min(self.capacity, self.size+len(observation))

    def state(self):
        return dict(observations=self.observations[:self.size].clone(), actions=self.actions[:self.size].clone(),
                    size=self.size, position=self.position, capacity=self.capacity)

    def load(self, value):
        if value["capacity"] != self.capacity:
            raise ValueError("DAgger replay capacity changed on resume")
        self.size, self.position = value["size"], value["position"]
        self.observations[:self.size] = value["observations"]
        self.actions[:self.size] = value["actions"]


def load_expert(args):
    from morph_mimic.networks.tracker import TrackerActor
    require_expert_validation(args.teacher)
    value = torch.load(args.teacher, map_location=args.device, weights_only=False)
    v3.require_teacher_checkpoint(value)
    teacher = TrackerActor(obs_dim=584).to(args.device).eval()
    teacher.load_state_dict(value["ppo"]["actor"])
    teacher.requires_grad_(False)
    return teacher


def train_dagger(args):
    from morph_mimic.launcher import launch, close
    from morph_mimic.scripts import common
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    teacher = load_expert(args)
    student = PaperTrackerActor().to(args.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    replay = Replay(args.replay_capacity, args.device)
    next_iteration, total_steps, restored_rng = 0, 0, None
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        require_student_checkpoint(saved)
        if saved.get("curriculum_kind") != "dagger" or saved.get("dagger_schedule") != schedule(args):
            raise ValueError("DAgger resume requires its original method and mixture schedule")
        if saved["teacher_sha256"] != file_hash(args.teacher):
            raise ValueError("DAgger teacher changed on resume")
        student.load_state_dict(saved["ppo"]["actor"])
        optimizer.load_state_dict(saved["dagger_optimizer"])
        replay.load(saved["dagger_replay"])
        next_iteration, total_steps, restored_rng = saved["next_iteration"], saved["total_steps"], saved["rng_state"]
        if next_iteration >= args.max_iterations:
            raise ValueError("Resume checkpoint already reaches the requested iteration target")
    app, env = launch(headless=True, device=args.device), None
    try:
        install_runtime(args)
        env, _ = common.build_env(1, "g1", args.num_envs, device=args.device, seed=args.seed,
            motion_dir=str(args.motion_dir), clips=args.clips)
        contract = common_contract(env)
        if saved is not None:
            validate_checkpoint_contract(saved, env, mode="resume")
        observation, _ = env.reset()
        episode_counts = dict(completed_from_start=0, box_and_contact_hold_passed=0, physical_failures=0)
        if restored_rng is not None:
            restore_rng_state(restored_rng)
        for iteration in range(next_iteration, args.max_iterations):
            started = time.monotonic()
            beta = beta_schedule(iteration, args.teacher_only_iterations, args.anneal_iterations)
            for _ in range(args.rollout_steps):
                before = observation["policy"]
                with torch.no_grad():
                    env.query_expert(before, teacher, beta)
                    action = student(before)
                    observation, reward, *_ = env.step(action)
                if not bool(torch.isfinite(reward).all() & torch.isfinite(observation["policy"]).all()):
                    raise FloatingPointError("Nonfinite student rollout")
                replay.add(before, env._expert_label)
                for episode in env.finished_episodes:
                    full = (episode["init_phase"] <= 1e-6 and episode["end_phase"] >= 1.
                            and not episode.get("terminated", True))
                    episode_counts["completed_from_start"] += int(full)
                    episode_counts["box_and_contact_hold_passed"] += int(full and episode.get("contact_surface_hold_passed", False))
                    episode_counts["physical_failures"] += int(episode.get("terminated", False))
                env.finished_episodes = []
                total_steps += args.num_envs
            losses = []
            for _ in range(args.updates_per_iteration):
                indices = torch.randint(replay.size, (min(replay.size, args.batch_size),), device=args.device)
                prediction = student(replay.observations[indices])
                loss = (prediction-replay.actions[indices]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite student imitation loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.detach())
            if iteration % 10 == 0 or iteration+1 == args.max_iterations:
                print(f"iter {iteration:6d} | t {time.monotonic()-started:5.1f}s | "
                      f"teacher_beta {beta:.3f} | replay {replay.size} | "
                      f"action_mse {float(torch.stack(losses).mean()):.6f} | "
                      f"q_target_rmse_rad {float(torch.stack(losses).mean().sqrt())*.25:.6f}", flush=True)
            if (iteration+1) % args.chunk_iterations == 0 or iteration+1 == args.max_iterations:
                state = dict(schema=2, stage=1, robot="g1", contract=contract, seed=args.seed,
                    next_iteration=iteration+1, total_steps=total_steps, iteration=iteration,
                    ppo=dict(actor=student.state_dict()), curriculum_kind="dagger",
                    dagger_schedule=schedule(args), teacher_beta=beta, teacher_sha256=file_hash(args.teacher),
                    dagger_optimizer=optimizer.state_dict(), dagger_replay=replay.state(), rng_state=capture_rng_state(),
                    source_sha256=source_hashes())
                atomic_torch_save(state, args.run_dir/"final.pt")
                print(f"[student] saved {args.run_dir/'final.pt'}", flush=True)
        if args.out:
            _atomic_json(args.out, dict(passed=True, kind="student_physical_pipeline_preflight",
                environments=args.num_envs, next_iteration=args.max_iterations, source_sha256=source_hashes(),
                teacher_sha256=file_hash(args.teacher), actor_input_dim=468, action_semantics=STUDENT_CONTRACT["action"],
                input_sha256=input_hashes(args),
                finite_rollout=True, finite_backward=True, physical_contacts_healthy=True,
                episodes=episode_counts,
                teacher_beta=beta, student_skill_validated=False, checkpoint=str(args.run_dir/"final.pt")))
    finally:
        if env is not None:
            env.close()
        close(app)


def common_contract(env):
    # Resolve the process-local wrapped builder, not the early imported symbol.
    from morph_mimic.utils import checkpoint
    return checkpoint.build_checkpoint_contract(env)


def schedule(args):
    return {key: getattr(args, key) for key in ("teacher_only_iterations", "anneal_iterations",
        "rollout_steps", "updates_per_iteration", "batch_size", "replay_capacity", "learning_rate")}


def evaluate(args):
    from morph_mimic.scripts import evaluate as evaluation
    from morph_mimic.launcher import launch, close
    from morph_mimic.utils.specs import RobotSpec
    value = torch.load(args.resume, map_location=args.device, weights_only=False)
    require_student_checkpoint(value)
    args.phase = value["contract"]["training"]["student_curriculum_phase"]
    args.reward_profile = value["contract"]["training"]["student_reward_profile"]
    student = PaperTrackerActor().to(args.device).eval()
    student.load_state_dict(value["ppo"]["actor"])
    metadata = evaluation._clip_metadata(args.motion_dir, args.clips, RobotSpec.load("g1"))
    values = vars(args) | dict(robot="g1", stage=1, planner="acp", max_steps=602,
        start_phase=0., held_joint_mode="hard_limits_at_nominal")
    app = launch(headless=True, device=args.device)
    try:
        task = install_runtime(args)
        evaluation.TrackingTrajectory = v3.ContactTrajectory
        result = evaluation._evaluate_clip(SimpleNamespace(**values), value, metadata[0], student, None, app)
        result.update(expected_control_steps=600, physical_hold_required_s=1.)
        report = dict(robot="g1", stage=1, checkpoint=str(args.resume), checkpoint_sha256=file_hash(args.resume),
            student_curriculum=STUDENT_CONTRACT, box_task=task, start_phase=0., evaluation_is_full=True,
            evaluation_is_partial=False, contract_verified=result["contract_verified"], expected_clips=1,
            evaluated_clips=1, completion_rate=float(result["completed"]), failures=int(result["failure"]),
            motions=[result], task_passed=result["physical_diagnostics"]["box_task"]["passed"],
            teacher_loaded=False, critics_loaded=False, teacher_beta=0.)
        report["student_passed"], report["student_failure_reasons"] = v3.task_acceptance(report)
        evaluation._write_report(args.out, report)
        print(json.dumps({key: report[key] for key in ("student_passed", "student_failure_reasons", "teacher_loaded")}), flush=True)
    finally:
        close(app)


def train_ppo(args):
    from morph_mimic.scripts import common
    from morph_mimic.launcher import launch, close
    from morph_mimic.rl.runner import Runner
    # Admission is based on a successful STUDENT rollout, not teacher success.
    if not args.resume:
        if args.initialize_actor is None or args.admission_report is None:
            raise ValueError("PPO needs an explicit student initialization and successful student-only admission report")
        require_ppo_admission(args.admission_report, args.initialize_actor)
    else:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        require_student_checkpoint(resumed)
        if resumed.get("curriculum_kind") == "dagger" or not {"critic_m", "critic_fs", "opt_main", "opt_fs"} <= resumed.get("ppo", {}).keys():
            raise ValueError("PPO resume requires its own PPO critics and optimizers; use explicit actor-only initialization for DAgger")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    app, env = launch(headless=True, device=args.device), None
    try:
        install_runtime(args)
        env, _ = common.build_env(1, "g1", args.num_envs, device=args.device, seed=args.seed,
            motion_dir=str(args.motion_dir), clips=args.clips)
        student = PaperTrackerActor().to(args.device)
        if args.initialize_actor:
            initial = torch.load(args.initialize_actor, map_location=args.device, weights_only=False)
            require_student_checkpoint(initial)
            student.load_state_dict(initial["ppo"]["actor"])
        cfg = common.configure_runner(SimpleNamespace(**(vars(args) | dict(
            continue_run=args.resume is not None, resume=str(args.resume) if args.resume else None,
            save_interval=args.chunk_iterations))), 1)
        Runner(env, cfg, torch.device(args.device), seed=args.seed).learn(student,
            PaperCritic(353).to(args.device), PaperCritic(298).to(args.device), log_dir=str(args.run_dir))
    finally:
        if env is not None:
            env.close()
        close(app)


class StudentSupervisor(Supervisor):
    _train = v1.TeacherSupervisor._train

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.status.update(student_curriculum=STUDENT_CONTRACT, curriculum_phase=args.phase,
            success_definition="Three distinct STUDENT-only full-start checkpoints; no teacher mixture during evaluation")
        self.inputs = input_hashes(args)
        self.status["frozen_inputs"] = self.inputs

    def _sources(self, *, initialize=False):
        if initialize:
            super()._sources(initialize=True)
            self.status["source_sha256"].update(source_hashes())
        super()._sources()
        if any(file_hash(path) != digest for path, digest in self.inputs.items()):
            raise ValueError("Frozen student curriculum input changed")

    def _command(self, mode):
        command = [str(self.args.python_executable), "-m", "morph_mimic.scripts.train_acp_student", mode,
            "--phase", self.args.phase, "--reward-profile", self.args.reward_profile,
            "--motion-dir", str(self.args.motion_dir), "--clips", *self.args.clips,
            "--teacher", str(self.args.teacher), "--device", self.args.device, "--seed", str(self.args.seed)]
        for key, value in schedule(self.args).items():
            command += ["--"+key.replace("_", "-"), str(value)]
        if self.args.initialize_actor:
            command += ["--initialize-actor", str(self.args.initialize_actor),
                        "--admission-report", str(self.args.admission_report)]
        return command

    def _evaluate(self, checkpoint, metadata, digest):
        iteration = metadata["next_iteration"]
        directory = self.args.run_dir/"evaluations"
        directory.mkdir(parents=True, exist_ok=True)
        report_path, trajectory = [directory/f"iter{iteration:06d}.{suffix}" for suffix in ("json", "npz")]
        self._child(self._command("evaluate")+["--resume", str(checkpoint), "--out", str(report_path),
            "--trajectory", str(trajectory)], f"evaluate_iter{iteration:06d}.log", phase="evaluate")
        report = json.loads(report_path.read_text())
        if (report.get("checkpoint_sha256") != digest or report.get("teacher_loaded") is not False
                or report.get("student_curriculum") != STUDENT_CONTRACT
                or not trajectory.is_file() or file_hash(trajectory) != report["motions"][0].get("trajectory_sha256")):
            raise ValueError("Student evaluation provenance or teacher-free execution differs")
        passed, reasons = v3.task_acceptance(report)
        # DAgger checkpoints produced with assistance never advance admission.
        beta = beta_schedule(iteration-1, self.args.teacher_only_iterations, self.args.anneal_iterations)
        if self.args.phase == "dagger" and beta > 0.:
            passed = False
            reasons.append("student_only_curriculum_not_reached")
        self.status["consecutive_successes"] = self.status["consecutive_successes"]+1 if passed else 0
        self.status["evaluations"].append(dict(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            report=str(report_path), trajectory=str(trajectory), actual_total_steps=metadata["total_steps"],
            passed=passed, reasons=reasons, task_passed=report["task_passed"],
            teacher_beta_during_training=beta if self.args.phase == "dagger" else 0.,
            metrics={key: report["motions"][0][key] for key in CRITERIA},
            consecutive_successes=self.status["consecutive_successes"]))
        self.save()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "evaluate", "supervise"))
    parser.add_argument("--phase", choices=("dagger", "ppo"), default="dagger")
    parser.add_argument("--reward-profile", choices=("contact_aux", "paper_base"), default=None)
    parser.add_argument("--motion-dir", type=Path, default=PROJECT_ROOT/"datasets/g1_box_climb_contact_v3")
    parser.add_argument("--clips", nargs=1, default=["climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3"])
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    for key in ("run-dir", "resume", "out", "trajectory", "initialize-actor", "admission-report", "readiness-report"):
        parser.add_argument("--"+key, type=Path)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=6000)
    parser.add_argument("--chunk-iterations", type=int, default=500)
    parser.add_argument("--teacher-only-iterations", type=int, default=200)
    parser.add_argument("--anneal-iterations", type=int, default=800)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--updates-per-iteration", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--replay-capacity", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    args.reward_profile = args.reward_profile or ("contact_aux" if args.phase == "dagger" else "paper_base")
    for key in ("motion_dir", "teacher", "run_dir", "resume", "out", "trajectory", "initialize_actor",
                "admission_report", "readiness_report", "python_executable"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    for key in ("num_envs", "max_iterations", "chunk_iterations", "anneal_iterations", "rollout_steps",
                "updates_per_iteration", "batch_size", "replay_capacity"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.teacher_only_iterations < 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Invalid mixture schedule or learning rate")
    if args.mode != "evaluate" and not args.teacher.is_file():
        parser.error("Frozen expert checkpoint is missing")
    if args.mode in ("preflight", "train", "supervise") and args.run_dir is None:
        parser.error("Choose an independent --run-dir")
    if args.mode in ("preflight", "evaluate") and (args.out is None or args.out.exists()):
        parser.error("Choose a new --out report")
    if args.mode == "evaluate" and (args.resume is None or args.trajectory is None or args.trajectory.exists()):
        parser.error("Evaluation needs a student checkpoint and new trajectory")
    if args.mode == "supervise" and (args.run_dir/"supervisor_status.json").exists():
        parser.error("Choose a new independent run")
    if args.resume and args.initialize_actor:
        parser.error("Strict resume and actor-only initialization cannot be combined")
    if args.mode in ("train", "preflight") and not args.resume and (args.run_dir/"final.pt").exists():
        parser.error("Run directory already contains weights; use explicit --resume or a new directory")
    if args.mode == "preflight" and (args.phase != "dagger" or args.max_iterations > 10):
        parser.error("Preflight requires dagger and at most 10 iterations")
    args.robot, args.required_successes, args.wait_pid = "g1", 3, None
    args.wait_min_iteration, args.resume_checkpoint = 0, args.resume
    args.training_profile, args.held_joint_mode = "nominal_pretrain", "hard_limits_at_nominal"
    return args


def main():
    args = arguments()
    if args.mode in ("train", "preflight"):
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (train_dagger if args.phase == "dagger" else train_ppo)(args)
    elif args.mode == "evaluate":
        with torch.no_grad():
            evaluate(args)
    else:
        if args.readiness_report is None:
            raise ValueError("Student supervision requires its own physical --readiness-report")
        ready = json.loads(args.readiness_report.read_text())
        if (ready.get("passed") is not True or ready.get("source_sha256") != source_hashes()
                or ready.get("environments") != args.num_envs or ready.get("teacher_sha256") != file_hash(args.teacher)
                or ready.get("input_sha256") != input_hashes(args)):
            raise ValueError("Student physical preflight is missing or differs from frozen code/inputs/environment count")
        StudentSupervisor(args).run()


if __name__ == "__main__":
    main()
