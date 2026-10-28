"""Independent privileged residual-reference teacher, not an ACP student.

Uses the unchanged ACP rewards/PPO and G1 23-axis model. The actor additionally
sees named reference kinematics and tracking errors; joint targets are current
reference positions plus residuals. RSI resets only at episode boundaries;
there is no root assistance during physical rollouts. Evaluate full phase zero.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

from morph_mimic.scripts.train_until_validated import (
    PROJECT_ROOT, CRITERIA, Supervisor, _sha256, acceptance,
)

SCRIPT = "morph_mimic/scripts/train_reference_teacher.py"
BASE_OBS_DIM = 468
FEATURES = (("reference_joint_position", 23), ("reference_joint_velocity", 23),
    ("reference_root_quaternion_wxyz", 4), ("reference_root_linear_velocity_body_yaw", 3),
    ("reference_root_angular_velocity_body_yaw", 3), ("root_position_error_body_yaw", 3),
    ("root_orientation_error_wxyz", 4), ("joint_position_error", 23),
    ("joint_velocity_error", 23), ("root_linear_velocity_error_body_yaw", 3),
    ("root_angular_velocity_error_body_yaw", 3), ("reference_phase", 1))
OBS_DIM = BASE_OBS_DIM + sum(width for _, width in FEATURES)
TEACHER_CONTRACT = dict(schema=1, experiment="privileged_reference_residual_teacher_v1",
    scope="Dynamics/tracking diagnostic; not ACP finite-observation skill validation",
    base_observation="unchanged ACP 468", extra_features=[list(value) for value in FEATURES],
    action="q_target=current_post_phase_advance_reference_q+0.25*clipped_residual",
    actions=23, residual_scale_rad=.25, action_delay_controls=0,
    target_reference_phase="same post-advance phase as reward reference, constant over physics substeps",
    rotation_error="conjugate(actual_world_from_link)*reference_world_from_link; nonnegative W",
    translation_error="reference minus actual; yaw-rotated vector with relative Z",
    root_linear_velocity_origin="base_link_frame_origin",
    sampling=dict(kind="stratified_RSI_v1", phase_zero_minimum_fraction=.15,
        barrier_fraction=.35, barrier_seconds=[3.2, 4.5], remainder="uniform phase [0,.98)",
        counts="ceil(.15*N) phase zero; round(.35*N) barrier capped by remainder; random permutation",
        csi_min_phase="unused; core CSI buffer updates do not affect this sampler"),
    initialization="fresh default actor/critics/PPO; no ACP or standing checkpoint transfer",
    reward="unchanged ACP Stage1 reward definitions and weights", root_assistance=False)


def sample_initial_phases(count, duration_s, *, device="cpu", generator=None):
    """Guaranteed phase-zero coverage per reset batch plus whole-clip/barrier RSI."""
    import torch

    if count < 0 or not math.isfinite(duration_s) or duration_s <= 4.5:
        raise ValueError("Teacher RSI needs nonnegative count and a clip longer than 4.5 seconds")
    if not count:
        return torch.empty(0, device=device)
    zero = math.ceil(.15 * count)
    barrier = min(count - zero, round(.35 * count))
    whole = count - zero - barrier
    phase = torch.cat((torch.zeros(zero, device=device),
        (3.2 + (4.5 - 3.2) * torch.rand(barrier, device=device, generator=generator)) / duration_s,
        .98 * torch.rand(whole, device=device, generator=generator)))
    return phase[torch.randperm(count, device=device, generator=generator)]


def residual_joint_targets(default_pose, controlled_idx, reference_q, residual):
    if reference_q.shape != residual.shape or reference_q.ndim != 2 or reference_q.shape[1] != 23:
        raise ValueError("Teacher requires matching [N,23] named reference/residual arrays")
    targets = default_pose.unsqueeze(0).expand(len(residual), -1).clone()
    targets[:, controlled_idx] = reference_q + .25 * residual
    return targets


def reference_features(env):
    """Current reference aligned with the current physical state; tile-invariant errors."""
    import torch
    from morph_mimic.utils.transforms import quat_apply, quat_conjugate, quat_mul, yaw_from_quat, yaw_quat

    ref, data = env._ref_sample(), env.robot.data
    current_q, current_qd = data.joint_pos[:, env.controlled_idx], data.joint_vel[:, env.controlled_idx]
    rotation = yaw_quat(-yaw_from_quat(data.root_quat_w))
    rotate = lambda value: quat_apply(rotation, value)
    error_quat = quat_mul(quat_conjugate(data.root_quat_w), ref["root_quat"])
    error_quat = torch.where(error_quat[:, :1] < 0, -error_quat, error_quat)
    result = torch.cat((ref["joint_pos"], ref["joint_vel"], ref["root_quat"],
        rotate(ref["root_lin_vel"]), rotate(ref["root_ang_vel"]),
        rotate(ref["root_pos"] - data.root_pos_w), error_quat,
        ref["joint_pos"] - current_q, ref["joint_vel"] - current_qd,
        rotate(ref["root_lin_vel"] - data.root_link_lin_vel_w),
        rotate(ref["root_ang_vel"] - data.root_ang_vel_w), env.phase[:, None]), dim=-1)
    if result.shape != (env.num_envs, OBS_DIM - BASE_OBS_DIM) or not torch.isfinite(result).all():
        raise ValueError("Teacher reference features are malformed/nonfinite")
    return result


def teacher_environment_class(base):
    """Create a process-local extension without editing the frozen core class."""
    class ReferenceTeacherEnv(base):
        def _sample_csi_initial_phase(self, env_ids):
            durations = self.clip_duration[env_ids]
            if not len(env_ids):
                return self.phase[env_ids]
            if not bool((durations == durations[0]).all()):
                raise ValueError("This teacher currently supports one bound reference clip")
            return sample_initial_phases(len(env_ids), float(durations[0]), device=self.device)

        def _apply_action(self):
            if self.cfg.replay:
                raise ValueError("Teacher is a learned residual controller, not replay mode")
            targets = residual_joint_targets(self.default_pose, self.controlled_idx,
                                              self._ref_sample()["joint_pos"], self.actions)
            self.robot.set_joint_position_target(targets)

        def _get_observations(self, record=True):
            observations = super()._get_observations(record=record)
            observations["policy"] = self._teacher_observation(observations["policy"])
            return observations

        def refresh_observations(self):
            observations = super().refresh_observations()
            observations["policy"] = self._teacher_observation(observations["policy"])
            return observations

        def _teacher_observation(self, observation):
            import torch
            if observation.shape != (self.num_envs, BASE_OBS_DIM):
                raise ValueError("Teacher expects the original ACP observation before augmentation")
            return torch.cat((observation, reference_features(self)), dim=-1)
    return ReferenceTeacherEnv


def augment_contract(contract):
    if contract is None:
        raise ValueError("Teacher checkpoints require a real robot/data contract")
    result = deepcopy(contract)
    result["observations"].update(policy_dim=OBS_DIM, action_dim=23,
        policy_schema="ACP468_plus_privileged_reference116_teacher_v1",
        reference_teacher=deepcopy(TEACHER_CONTRACT))
    result["physics"]["joint_target_semantics"] = TEACHER_CONTRACT["action"]
    return result


def require_teacher_checkpoint(checkpoint):
    contract = checkpoint.get("contract") or {}
    observations = contract.get("observations", {})
    if (checkpoint.get("schema") != 2 or checkpoint.get("robot") != "g1" or checkpoint.get("stage") != 1
            or observations.get("reference_teacher") != TEACHER_CONTRACT
            or observations.get("policy_dim") != OBS_DIM or observations.get("action_dim") != 23
            or contract.get("physics", {}).get("joint_target_semantics") != TEACHER_CONTRACT["action"]):
        raise ValueError("Requires this independent reference teacher contract; ACP/standing/legacy checkpoints are refused")


def _install_runtime():
    # Call only after SimulationApp has started. All mutations are process-local.
    from morph_mimic.envs import acp_env
    from morph_mimic.scripts import common
    from morph_mimic.utils import checkpoint
    from morph_mimic.rl import runner

    acp_env.AcpEnv = teacher_environment_class(acp_env.AcpEnv)
    original_build = common.build_env
    def build(*args, **kwargs):
        kwargs.update(use_usd=False, training_profile="nominal_pretrain",
            held_joint_mode="hard_limits_at_nominal", observation_space=OBS_DIM,
            csi_enabled=not kwargs.get("evaluation", False), csi_sampling_mode="teacher_stratified_RSI_v1")
        env, spec = original_build(*args, **kwargs)
        if (env.cfg.stage != 1 or env.cfg.ref_env_ratio != 1 or len(env.motion_data.clip_names) != 1
                or not env.motion_data.scene_bindings or bool(env.action_delay.any())):
            env.close()
            raise ValueError("Teacher needs nominal zero-delay Stage1 with one bound scene and 23 controlled joints")
        return env, spec
    common.build_env = build
    original_contract = checkpoint.build_checkpoint_contract
    def build_contract(*args, **kwargs):
        return augment_contract(original_contract(*args, **kwargs))
    checkpoint.build_checkpoint_contract = build_contract
    runner.build_checkpoint_contract = build_contract
    # Core validator resolves its global builder from checkpoint, including on resume.


def _train(args):
    import numpy as np
    import torch
    from morph_mimic.scripts import common
    from morph_mimic.launcher import launch, close

    if args.resume is not None:
        require_teacher_checkpoint(torch.load(args.resume, map_location="cpu", weights_only=False))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    app, env = launch(headless=True, device=args.device), None
    try:
        _install_runtime()
        from morph_mimic.networks.tracker import Critic, TrackerActor
        from morph_mimic.rl.runner import Runner
        values = vars(args) | dict(continue_run=args.resume is not None,
            resume=str(args.resume) if args.resume else None, save_interval=args.chunk_iterations)
        cfg = common.configure_runner(SimpleNamespace(**values), 1)
        env, _ = common.build_env(1, "g1", args.num_envs, device=args.device, seed=args.seed,
            motion_dir=str(args.motion_dir), clips=args.clips)
        actor = TrackerActor(obs_dim=OBS_DIM).to(args.device)
        critics = (Critic(353).to(args.device), Critic(298).to(args.device))
        Runner(env, cfg, torch.device(args.device), seed=args.seed).learn(actor, *critics, log_dir=str(args.run_dir))
    finally:
        if env is not None:
            env.close()
        close(app)


def _evaluate(args):
    import torch
    from morph_mimic.scripts import evaluate
    from morph_mimic.networks.tracker import TrackerActor
    from morph_mimic.launcher import launch, close
    from morph_mimic.utils.specs import RobotSpec

    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    require_teacher_checkpoint(checkpoint)
    actor = TrackerActor(obs_dim=OBS_DIM).to(args.device).eval()
    actor.load_state_dict(checkpoint["ppo"]["actor"])
    metadata = evaluate._clip_metadata(args.motion_dir, args.clips, RobotSpec.load("g1"))
    values = vars(args) | dict(robot="g1", stage=1, planner="acp", max_steps=None,
        start_phase=0., held_joint_mode="hard_limits_at_nominal")
    app = launch(headless=True, device=args.device)
    try:
        _install_runtime()
        result = evaluate._evaluate_clip(SimpleNamespace(**values), checkpoint, metadata[0], actor, None, app)
        report = dict(robot="g1", stage=1, checkpoint=str(args.resume), checkpoint_sha256=_sha256(args.resume),
            scope=TEACHER_CONTRACT["scope"], reference_teacher=TEACHER_CONTRACT,
            start_phase=0., evaluation_is_full=True, evaluation_is_partial=False,
            contract_verified=result["contract_verified"], expected_clips=1, evaluated_clips=1,
            completion_rate=float(result["completed"]), failures=int(result["failure"]), motions=[result])
        report["teacher_passed"], report["teacher_failure_reasons"] = acceptance(report)
        evaluate._write_report(args.out, report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        close(app)


class TeacherSupervisor(Supervisor):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.status.update(scope=TEACHER_CONTRACT["scope"], reference_teacher=TEACHER_CONTRACT,
            success_definition="Three distinct checkpoints passing the unchanged full phase-zero box reference gates")

    def _sources(self, *, initialize=False):
        super()._sources(initialize=initialize)
        if initialize:
            self.status["source_sha256"].setdefault(SCRIPT, _sha256(PROJECT_ROOT / SCRIPT))
            super()._sources()

    def _command(self, mode):
        return [str(self.args.python_executable), "-m", "morph_mimic.scripts.train_reference_teacher", mode,
            "--motion-dir", str(self.args.motion_dir), "--clips", *self.args.clips,
            "--device", self.args.device, "--seed", str(self.args.seed)]

    def _train(self, checkpoint, target):
        command = self._command("train") + ["--run-dir", str(self.args.run_dir), "--num-envs", str(self.args.num_envs),
            "--max-iterations", str(target), "--chunk-iterations", str(self.args.chunk_iterations)]
        if checkpoint is not None:
            command += ["--resume", str(checkpoint)]
        self.status["target_iteration"] = target
        self._child(command, f"train_until_iter{target:06d}.log", phase="train")
        return self._snapshot(self.args.run_dir / "final.pt", expected_iteration=target)

    def _evaluate(self, checkpoint, metadata, digest):
        iteration = metadata["next_iteration"]
        directory = self.args.run_dir / "evaluations"
        directory.mkdir(parents=True, exist_ok=True)
        report_path, trajectory = [directory / f"iter{iteration:06d}.{suffix}" for suffix in ("json", "npz")]
        self._child(self._command("evaluate") + ["--resume", str(checkpoint), "--out", str(report_path),
            "--trajectory", str(trajectory)], f"evaluate_iter{iteration:06d}.log", phase="evaluate")
        report = json.loads(report_path.read_text())
        if (report.get("checkpoint_sha256") != digest or report["motions"][0].get("clip") != self.args.clips[0]
                or not trajectory.is_file() or _sha256(trajectory) != report["motions"][0].get("trajectory_sha256")
                or report.get("reference_teacher") != TEACHER_CONTRACT):
            raise ValueError("Teacher evaluation checkpoint, clip, semantics or physical trajectory hash differs")
        contact_path = directory / f"iter{iteration:06d}_contacts.json"
        self._child([str(self.args.python_executable), "-m", "morph_mimic.scripts.evaluate_contact_trajectory",
            "--trajectory", str(trajectory), "--motion", str(self.args.motion_dir / f"{self.args.clips[0]}.npz"),
            "--labels", str(self.args.motion_dir / f"{self.args.clips[0]}_contacts.npz"), "--out", str(contact_path)],
            f"contacts_iter{iteration:06d}.log", phase="contact_metrics", cpu_only=True)
        contact = json.loads(contact_path.read_text())
        passed, reasons = acceptance(report)
        if any(value["checkpoint_sha256"] == digest for value in self.status["evaluations"]):
            raise ValueError("A checkpoint cannot count twice toward teacher acceptance")
        self.status["consecutive_successes"] = self.status["consecutive_successes"] + 1 if passed else 0
        self.status.update(latest_contact_report=str(contact_path), latest_contact_coverage=contact["coverage"])
        self.status["evaluations"].append(dict(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            report=str(report_path), trajectory=str(trajectory), contact_report=str(contact_path),
            contact_coverage=contact["coverage"], contact_metrics=contact["micro_force_history_vs_reference"],
            actual_total_steps=metadata["total_steps"], passed=passed, reasons=reasons,
            metrics={key: report["motions"][0][key] for key in CRITERIA},
            consecutive_successes=self.status["consecutive_successes"]))
        self.save()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "supervise"))
    parser.add_argument("--motion-dir", type=Path, default=PROJECT_ROOT / "datasets/g1_box_climb_seed_v2")
    parser.add_argument("--clips", nargs=1, default=["climb_03_z_scale_1.0_scene_fit_0000_0501"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=15000)
    parser.add_argument("--chunk-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    for key in ("motion_dir", "run_dir", "resume", "out", "trajectory", "python_executable"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if min(args.num_envs, args.max_iterations, args.chunk_iterations) < 1:
        parser.error("environment and iteration counts must be positive")
    if Path(args.clips[0]).name != args.clips[0] or not (args.motion_dir / f"{args.clips[0]}.npz").is_file():
        parser.error("selected box reference does not exist")
    if args.resume is not None and not args.resume.is_file():
        parser.error("resume checkpoint does not exist")
    if args.mode == "evaluate" and (args.resume is None or args.out is None or args.trajectory is None):
        parser.error("evaluate requires --resume, --out and --trajectory")
    if args.mode != "evaluate" and args.run_dir is None:
        parser.error("train/supervise requires --run-dir")
    if args.mode == "supervise" and (args.run_dir / "supervisor_status.json").exists():
        parser.error("choose a fresh independent teacher run directory")
    return args


def main():
    args = arguments()
    if args.mode == "train":
        _train(args)
    elif args.mode == "evaluate":
        _evaluate(args)
    else:
        args.robot, args.training_profile, args.held_joint_mode = "g1", "nominal_pretrain", "hard_limits_at_nominal"
        args.required_successes, args.wait_pid, args.wait_min_iteration, args.resume_checkpoint = 3, None, 0, args.resume
        print(json.dumps(TeacherSupervisor(args).run(), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
