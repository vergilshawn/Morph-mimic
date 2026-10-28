"""Contact-surface teacher with all-body measurements and a stable settle suffix.

An independent engineering experiment, not the finite-observation ACP policy.
The original ten-second motion is retained; the new reference appends one
second to settle before fifty additional physical hold controls. No root
assistance, mid-rollout pose writes or contact-dependent phase freezing.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from morph_mimic.scripts import train_reference_teacher as v1
from morph_mimic.scripts import train_reference_teacher_v2 as v2
from morph_mimic.scripts.train_until_validated import PROJECT_ROOT, CRITERIA, _sha256, _atomic_json
from morph_mimic.utils.contact_runtime import ContactRuntime, ContactRuntimeConfig, attach_contact_sensors
from morph_mimic.utils.contact_tracking import ContactTracking, CONTACT_TRACKING_CONTRACT, FIELDS
from morph_mimic.utils.contact_surface import load_contact_surface_spec
from morph_mimic.utils.evaluation import TrackingTrajectory

SCRIPT = "morph_mimic/scripts/train_reference_teacher_v3.py"
SOURCES = (SCRIPT, "morph_mimic/utils/contact_runtime.py", "morph_mimic/utils/contact_surface.py",
           "morph_mimic/utils/contact_tracking.py", "morph_mimic/utils/contact_events.py",
           "morph_mimic/utils/stable_hold.py", "morph_mimic/contacts/settle_reference.py")
MOTION_SECONDS = 11.
SENSOR_CONFIG = ContactRuntimeConfig(contacts_per_body=128)
REWARD_WEIGHTS = {"surface_orientation": 1., "contact_normal_distance": 1.,
                  "stance_tangent_slip": -.5, "unexpected_noncanonical_contact": -.5,
                  "hold_support_stability": 1.}
TEACHER_CONTRACT = deepcopy(v2.TEACHER_CONTRACT)
TEACHER_CONTRACT.update(experiment="all_body_contact_surface_settle_teacher_v3",
    initialization="Fresh optimization/progress; optional explicit v2 actor-only initialization; no optimizer/critic/iteration transfer",
    contact_measurements=json.loads(json.dumps(asdict(SENSOR_CONFIG))), contact_tracking=deepcopy(CONTACT_TRACKING_CONTRACT),
    additional_reward_weights=REWARD_WEIGHTS,
    landing_error="Policy-input body-origin anchors decoded at control start, compared with physical body origins at measured current-normal-load onset; surface event proxy reported separately",
    reference_extension="Original 501 frames followed by 50 smooth stable-settle frames; separate dataset and hashes",
    acceptance="Unchanged full-start tracking and box gates plus all-body force, measured support hull, COM speed and foot orientation at every physical hold endpoint")
TEACHER_CONTRACT["end_hold"]["force_scope"] = "All 30 rigid bodies terrain-filtered normal and friction force lever moments; independent free contact couples and robot self-contact forces unavailable"
COMMAND_FIELDS = {"policy_command_body_origin_rebased_m": (9, 3), "policy_command_valid": (9,),
                  "actual_contact_body_origin_rebased_m": (9, 3)}


def decode_policy_contact_anchors(targets, root_pos_w, root_quat_w, origin_w):
    """Invert the existing gravity-aligned XY-relative/absolute-Z policy frame."""
    from morph_mimic.utils.transforms import quat_apply, yaw_from_quat, yaw_quat
    result = quat_apply(yaw_quat(yaw_from_quat(root_quat_w))[:, None], targets)
    result[..., :2] += root_pos_w[:, None, :2]
    return result-origin_w[:, None]


def source_hashes():
    return {path: _sha256(PROJECT_ROOT/path) for path in SOURCES}


def require_training_readiness(args):
    if args.readiness_report is None:
        raise ValueError("Supervision requires an explicit --readiness-report from the completed checks")
    readiness = json.loads(args.readiness_report.read_text())
    if (readiness.get("training_ready") is not True or readiness.get("source_sha256") != source_hashes()
            or readiness.get("num_envs") != args.num_envs
            or readiness.get("motion_path") != str(args.motion_dir/f"{args.clips[0]}.npz")):
        raise ValueError("Training readiness is absent or belongs to different code, data or environment count")
    for path, digest in readiness.get("input_sha256", {}).items():
        if _sha256(path) != digest:
            raise ValueError(f"Preflight input changed: {path}")
    if len(readiness.get("input_sha256", {})) < 3:
        raise ValueError("Readiness requires the motion, contacts and scene hashes")


def augment_contract(contract, task_spec, surface_spec):
    result = v2.augment_contract(contract, task_spec)
    result["observations"].update(policy_schema="ACP468_plus_reference116_contact_surface_teacher_v3",
        reference_teacher=deepcopy(TEACHER_CONTRACT), contact_surface=deepcopy(surface_spec))
    return result


def require_teacher_checkpoint(checkpoint):
    obs = (checkpoint.get("contract") or {}).get("observations", {})
    if (checkpoint.get("schema") != 2 or checkpoint.get("robot") != "g1" or checkpoint.get("stage") != 1
            or obs.get("reference_teacher") != TEACHER_CONTRACT or obs.get("policy_dim") != v1.OBS_DIM
            or obs.get("action_dim") != 23 or not isinstance(obs.get("contact_surface"), dict)
            or not isinstance(obs.get("box_task"), dict)
            or (checkpoint.get("contract") or {}).get("physics", {}).get("joint_target_semantics") != TEACHER_CONTRACT["action"]):
        raise ValueError("A v3 contact-surface checkpoint is required for strict resume/evaluation")


def teacher_environment_class(base, task_spec, motion_path):
    parent = v2.teacher_environment_class(base, task_spec)

    class ContactTeacherEnv(parent):
        def _setup_scene(self):
            super()._setup_scene()
            attach_contact_sensors(self, SENSOR_CONFIG)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                self.contact_runtime = ContactRuntime(self)
                self._legacy_contact_orders = [self.contact_runtime.environment_order_for_sensor(sensor)
                    for sensor in self.scene_contact_sensors]
                self.tracking = ContactTracking(self, motion_path, task_spec)
                if self.tracking.reference_endpoint_potential.get("geometry_gate") is not True:
                    raise ValueError(f"Stable reference endpoint failed the pretraining conflict gate: {self.tracking.reference_endpoint_potential}")
                self._surface_hold_pass_count = torch.zeros_like(self.episode_length_buf)
            except BaseException:
                self.close()
                raise

        def _contact_forces_w(self):
            if not hasattr(self, "_legacy_contact_orders"):
                return super()._contact_forces_w()
            return torch.cat([sensor.data.force_matrix_w.index_select(0, order).sum(dim=-2)
                for sensor, order in zip(self.scene_contact_sensors, self._legacy_contact_orders)], dim=1)

        def _contact_force_history(self):
            if not hasattr(self, "_legacy_contact_orders"):
                return super()._contact_force_history()
            return torch.cat([sensor.data.force_matrix_w_history.index_select(0, order).sum(dim=-2)
                for sensor, order in zip(self.scene_contact_sensors, self._legacy_contact_orders)], dim=2)

        def _pre_physics_step(self, actions):
            self._policy_command_rebased = decode_policy_contact_anchors(self.contact_targets,
                self.robot.data.root_pos_w, self.robot.data.root_quat_w, self.env_origin()+self.cell_offset).clone()
            self._policy_command_valid = (self.contact_target_flags > .5).clone()
            super()._pre_physics_step(actions)

        def _get_dones(self):
            terminated, truncated = super()._get_dones()
            measured = self.contact_runtime.sample()
            if not bool(measured["all_body_health_gate"].all()):
                invalid = []
                for row, body in (~measured["health_gate"]).nonzero().detach().cpu().tolist()[:12]:
                    invalid.append(dict(env=row, body=self.robot.body_names[body],
                        **{key: measured[key][row, body].detach().cpu().tolist() for key in
                           ("buffer_invalid", "force_reconstruction_error_n", "normal_point_count", "friction_patch_count", "normal_load_n")}))
                raise RuntimeError(f"Unhealthy all-body contact measurement, env/body indices: {invalid}")
            values = self.tracking.update(measured)
            active = self._box_hold_controls > 0
            self._surface_hold_pass_count += (active & values["hold_gate"]).long()
            if self.cfg.evaluation:
                self.extras["final_tracking"].update(
                    {f"contact_tracking_{key}": values[key].detach().clone() for key in self.tracking.fields})
                self.extras["final_tracking"].update(
                    policy_command_body_origin_rebased_m=self._policy_command_rebased.detach().clone(),
                    policy_command_valid=self._policy_command_valid.detach().clone(),
                    actual_contact_body_origin_rebased_m=(self._contact_body_positions_w()
                        -(self.env_origin()+self.cell_offset)[:, None]).detach().clone())
            return terminated, truncated

        def _get_rewards(self):
            reward = super()._get_rewards()
            if self.tracking.last is None:
                raise RuntimeError("Contact rewards require the current post-physics pre-reset measurement")
            telemetry = self.extras["reward_terms"]
            for name, weight in REWARD_WEIGHTS.items():
                raw = self.tracking.last["reward_terms"][name]
                contribution = raw * weight * self.is_ref_env.float() * self.step_dt
                reward = reward + contribution
                telemetry[f"raw/{name}"] = raw.detach().mean()
                telemetry[f"weighted/{name}"] = contribution.detach().mean()
            telemetry["diagnostic/contact_surface_hold_gate_fraction"] = self.tracking.last["hold_gate"].float().mean().detach()
            telemetry["diagnostic/all_body_contact_health_fraction"] = self.tracking.last["measurement_health_gate"].float().mean().detach()
            return reward

        def _finish_episodes(self, env_ids):
            before = len(self.finished_episodes)
            super()._finish_episodes(env_ids)
            if hasattr(self, "_surface_hold_pass_count"):
                for episode in self.finished_episodes[before:]:
                    count = int(self._surface_hold_pass_count[episode["env_id"]])
                    episode.update(contact_surface_hold_passing_endpoints=count,
                        contact_surface_hold_passed=bool(count == v2.HOLD_STEPS and episode.get("box_task_passed")))

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            if hasattr(self, "_surface_hold_pass_count"):
                if env_ids is None:
                    self._surface_hold_pass_count.zero_()
                else:
                    self._surface_hold_pass_count[env_ids] = 0
    return ContactTeacherEnv


class ContactTrajectory(v2.BoxTaskTrajectory):
    _fields = v2.BoxTaskTrajectory._fields | {f"contact_tracking_{key}": shape for key, shape in FIELDS.items()} | COMMAND_FIELDS

    def summary(self):
        from morph_mimic.utils.contact_events import contact_event_report, commanded_landing_report
        result = super().summary()
        arrays = self.arrays()
        result["contact_surface"] = self.tracking.summary(arrays)
        result["commanded_landing"] = commanded_landing_report(arrays["elapsed_control_time_s"],
            arrays["policy_command_body_origin_rebased_m"], arrays["policy_command_valid"],
            arrays["actual_contact_body_origin_rebased_m"], arrays["contact_tracking_actual_normal_load_n"],
            control_dt_s=self.control_dt, body_names=CONTACT_TRACKING_CONTRACT["canonical_body_names"],
            metadata=dict(command_sampling="control_start_policy_input", command_anchor="body_origin",
                actual_sampling="post_physics_pre_reset_control_endpoints",
                positions_frame="scene_rebased_by_source_first_root_xy"))
        result["contact_events"] = contact_event_report(arrays["elapsed_control_time_s"], arrays["phase"],
            arrays["contact_tracking_actual_measured_point_rebased_m"], arrays["contact_tracking_actual_normal_load_n"],
            arrays["contact_tracking_actual_measured_normal_w"], **self.tracking.reference_event_inputs(),
            control_dt_s=self.control_dt, actual_valid=arrays["contact_tracking_actual_measured_point_valid"],
            body_names=CONTACT_TRACKING_CONTRACT["canonical_body_names"],
            metadata=dict(post_reference_hold_is_physical=True, sampling="post_physics_pre_reset_control_endpoints"))
        return result

    def write(self, path, *, metadata):
        metadata = deepcopy(metadata)
        metadata["result"].update(expected_control_steps=600, physical_hold_required_s=v2.HOLD_SECONDS)
        metadata.update(reference_teacher=TEACHER_CONTRACT, box_task=self.box_task_spec,
            contact_tracking=self.tracking.describe(), post_reference_hold_is_physical=True)
        return TrackingTrajectory.write(self, path, metadata=metadata)


def _install_runtime(args):
    from morph_mimic.envs import acp_env
    from morph_mimic.scripts import common
    from morph_mimic.utils import checkpoint
    from morph_mimic.rl import runner
    from morph_mimic.utils.box_task import load_box_task_spec
    motion = args.motion_dir/f"{args.clips[0]}.npz"
    task_spec, surface_spec = load_box_task_spec(motion), load_contact_surface_spec(motion)
    acp_env.AcpEnv = teacher_environment_class(acp_env.AcpEnv, task_spec, motion)
    original_build = common.build_env
    def build(*positional, **kwargs):
        kwargs.update(use_usd=False, training_profile="nominal_pretrain", held_joint_mode="hard_limits_at_nominal",
            observation_space=v1.OBS_DIM, episode_length_s=13.,
            csi_enabled=not kwargs.get("evaluation", False), csi_sampling_mode="teacher_stratified_RSI_v1")
        env, spec = original_build(*positional, **kwargs)
        if (env.cfg.stage != 1 or env.cfg.ref_env_ratio != 1 or len(env.motion_data.clip_names) != 1
                or not env.motion_data.scene_bindings or bool(env.action_delay.any())
                or abs(env.step_dt-.02) > 1e-9
                or any(abs(float(clip.duration_s)-MOTION_SECONDS) > .001 for clip in env.motion_data.clips.values())):
            env.close()
            raise ValueError("V3 requires one nominal, zero-delay 11-second bound G1 reference at 50Hz")
        ContactTrajectory.tracking = env.tracking
        ContactTrajectory.box_task_spec = task_spec
        return env, spec
    common.build_env = build
    original_contract = checkpoint.build_checkpoint_contract
    def contract(*positional, **kwargs):
        return augment_contract(original_contract(*positional, **kwargs), task_spec, surface_spec)
    checkpoint.build_checkpoint_contract = runner.build_checkpoint_contract = contract
    return task_spec


def _train(args):
    from morph_mimic.scripts import common
    from morph_mimic.launcher import launch, close
    if args.resume:
        require_teacher_checkpoint(torch.load(args.resume, map_location="cpu", weights_only=False))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    app, env = launch(headless=True, device=args.device), None
    try:
        _install_runtime(args)
        from morph_mimic.networks.tracker import Critic, TrackerActor
        from morph_mimic.rl.runner import Runner
        cfg = common.configure_runner(SimpleNamespace(**(vars(args) | dict(continue_run=args.resume is not None,
            resume=str(args.resume) if args.resume else None, save_interval=args.chunk_iterations))), 1)
        env, _ = common.build_env(1, "g1", args.num_envs, device=args.device, seed=args.seed,
            motion_dir=str(args.motion_dir), clips=args.clips)
        actor = TrackerActor(obs_dim=v1.OBS_DIM).to(args.device)
        if args.initialize_from is not None and args.resume is None:
            initial = torch.load(args.initialize_from, map_location=args.device, weights_only=False)
            v2.require_teacher_checkpoint(initial)
            actor.load_state_dict(initial["ppo"]["actor"], strict=True)
            _atomic_json(args.run_dir/"actor_initialization.json", dict(source=str(args.initialize_from),
                sha256=_sha256(args.initialize_from), source_iteration=initial["next_iteration"],
                transfer="actor weights only; new critics, optimizer, exploration noise, progress and RNG"))
        critics = (Critic(353).to(args.device), Critic(298).to(args.device))
        Runner(env, cfg, torch.device(args.device), seed=args.seed).learn(actor, *critics, log_dir=str(args.run_dir))
    finally:
        if env is not None:
            env.close()
        close(app)


def task_acceptance(report):
    _, reasons = v2.task_acceptance(report)
    surface = report["motions"][0].get("physical_diagnostics", {}).get("contact_surface", {})
    if surface.get("measurement_healthy_all") is not True:
        reasons.append("all_body_contact_measurement_health")
    if surface.get("hold_passed") is not True:
        reasons.append("all_body_contact_surface_hold: all 50 physical hold endpoints must pass")
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
    values = vars(args) | dict(robot="g1", stage=1, planner="acp", max_steps=602,
        start_phase=0., held_joint_mode="hard_limits_at_nominal")
    app = launch(headless=True, device=args.device)
    try:
        task_spec = _install_runtime(args)
        evaluate.TrackingTrajectory = ContactTrajectory
        result = evaluate._evaluate_clip(SimpleNamespace(**values), checkpoint, metadata[0], actor, None, app)
        result.update(expected_control_steps=600, physical_hold_required_s=1.)
        report = dict(robot="g1", stage=1, checkpoint=str(args.resume), checkpoint_sha256=_sha256(args.resume),
            scope=TEACHER_CONTRACT["scope"], reference_teacher=TEACHER_CONTRACT, box_task=task_spec,
            start_phase=0., evaluation_is_full=True, evaluation_is_partial=False,
            contract_verified=result["contract_verified"], expected_clips=1, evaluated_clips=1,
            completion_rate=float(result["completed"]), failures=int(result["failure"]), motions=[result],
            task_passed=result["physical_diagnostics"]["box_task"]["passed"],
            completion_definition="Original ten seconds + one-second settle + fifty extra physical hold endpoints")
        report["teacher_passed"], report["teacher_failure_reasons"] = task_acceptance(report)
        evaluate._write_report(args.out, report)
        print(json.dumps({"teacher_passed": report["teacher_passed"], "reasons": report["teacher_failure_reasons"], "out": str(args.out)}), flush=True)
    finally:
        close(app)


def _preflight(args):
    from morph_mimic.launcher import launch, close
    from morph_mimic.scripts import common
    app, env = launch(headless=True, device=args.device), None
    started = time.monotonic()
    try:
        _install_runtime(args)
        env, _ = common.build_env(1, "g1", args.num_envs, device=args.device, seed=args.seed,
            motion_dir=str(args.motion_dir), clips=args.clips)
        actor = None
        if args.initialize_from is not None:
            from morph_mimic.networks.tracker import TrackerActor
            initial = torch.load(args.initialize_from, map_location=args.device, weights_only=False)
            v2.require_teacher_checkpoint(initial)
            actor = TrackerActor(obs_dim=v1.OBS_DIM).to(args.device).eval()
            actor.load_state_dict(initial["ppo"]["actor"])
        obs, _ = env.reset()
        maxima, contact_endpoints = {}, 0
        for _ in range(args.steps):
            with torch.no_grad():
                action = actor(obs["policy"]) if actor is not None else torch.zeros((args.num_envs, 23), device=args.device)
                if args.preflight_noise:
                    action = action+args.preflight_noise*torch.randn_like(action)
                obs, reward, *_ = env.step(action)
            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                raise FloatingPointError("Preflight nonfinite observation/reward")
            state = env.tracking.last
            contact_endpoints += int((state["actual_normal_load_n"] > 1.).sum())
            for key in REWARD_WEIGHTS:
                maxima[key] = max(maxima.get(key, 0.), float(state["reward_terms"][key].abs().max()))
        if not contact_endpoints:
            raise RuntimeError("No measured terrain-bearing contact was observed")
        report = dict(passed=True, phase="raw_contact_physical_preflight", source_sha256=source_hashes(),
            environments=args.num_envs, steps=args.steps, seconds=round(time.monotonic()-started, 3),
            sensor=env.contact_runtime.describe(), reference_endpoint_potential=env.tracking.reference_endpoint_potential,
            measured_contact_endpoints=contact_endpoints, reward_max_abs=maxima,
            action_noise_std=args.preflight_noise, initialized_actor=bool(actor is not None),
            skill_validated=False, root_assistance=False, reference_teacher=TEACHER_CONTRACT)
        _atomic_json(args.out, report)
        print(json.dumps({key: report[key] for key in ("passed", "environments", "steps", "seconds", "measured_contact_endpoints")}), flush=True)
    except BaseException as error:
        _atomic_json(args.out, dict(passed=False, phase="raw_contact_physical_preflight",
            error_type=type(error).__name__, error=str(error), source_sha256=source_hashes()))
        raise
    finally:
        if env is not None:
            env.close()
        close(app)


class TeacherSupervisorV3(v2.TeacherSupervisorV2):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.status.update(reference_teacher=TEACHER_CONTRACT, success_definition=TEACHER_CONTRACT["acceptance"])

    def _sources(self, *, initialize=False):
        super()._sources(initialize=initialize)
        if initialize:
            self.status["source_sha256"].update(source_hashes())
            super()._sources()

    def _command(self, mode):
        command = super()._command(mode)
        command[2] = "morph_mimic.scripts.train_reference_teacher_v3"
        if mode == "train" and self.args.initialize_from is not None:
            command += ["--initialize-from", str(self.args.initialize_from)]
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
            raise ValueError("V3 checkpoint/reference/physical trajectory/measurement semantics differ")
        passed, reasons = task_acceptance(report)
        if any(value["checkpoint_sha256"] == digest for value in self.status["evaluations"]):
            raise ValueError("Cannot count the same checkpoint twice")
        self.status["consecutive_successes"] = self.status["consecutive_successes"]+1 if passed else 0
        self.status["evaluations"].append(dict(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            report=str(report_path), trajectory=str(trajectory), actual_total_steps=metadata["total_steps"],
            passed=passed, reasons=reasons, task_passed=report["task_passed"],
            contact_surface=report["motions"][0]["physical_diagnostics"]["contact_surface"],
            metrics={key: report["motions"][0][key] for key in CRITERIA},
            consecutive_successes=self.status["consecutive_successes"]))
        self.save()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train", "evaluate", "supervise"))
    parser.add_argument("--motion-dir", type=Path, default=PROJECT_ROOT/"datasets/g1_box_climb_contact_v3")
    parser.add_argument("--clips", nargs=1, default=["climb_03_z_scale_1.0_scene_fit_0000_0501_settle_v3"])
    for name in ("run-dir", "resume", "out", "trajectory", "initialize-from", "readiness-report"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=15000)
    parser.add_argument("--chunk-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--preflight-noise", type=float, default=0., help="explicit Gaussian action noise for physical stress checks")
    args = parser.parse_args(argv)
    for key in ("motion_dir", "run_dir", "resume", "out", "trajectory", "initialize_from", "readiness_report", "python_executable"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if min(args.num_envs, args.max_iterations, args.chunk_iterations, args.steps) < 1:
        parser.error("Counts must be positive")
    if not np.isfinite(args.preflight_noise) or args.preflight_noise < 0:
        parser.error("Preflight action noise must be finite and nonnegative")
    if Path(args.clips[0]).name != args.clips[0] or not (args.motion_dir/f"{args.clips[0]}.npz").is_file():
        parser.error("The selected contact-settle reference must exist")
    if args.resume is not None and not args.resume.is_file():
        parser.error("Resume checkpoint is missing")
    if args.initialize_from is not None and not args.initialize_from.is_file():
        parser.error("Explicit v2 initialization checkpoint is missing")
    if args.mode in ("preflight", "evaluate") and (args.out is None or args.out.exists()):
        parser.error("Choose a new --out report")
    if args.mode == "evaluate" and (args.resume is None or args.trajectory is None or args.trajectory.exists()):
        parser.error("Evaluation requires --resume and a new --trajectory")
    if args.mode in ("train", "supervise") and args.run_dir is None:
        parser.error("Training requires --run-dir")
    if args.mode == "supervise" and (args.run_dir/"supervisor_status.json").exists():
        parser.error("Choose a new independent run")
    return args


def main():
    args = arguments()
    if args.mode == "preflight":
        _preflight(args)
    elif args.mode == "train":
        _train(args)
    elif args.mode == "evaluate":
        _evaluate(args)
    else:
        require_training_readiness(args)
        args.robot, args.training_profile, args.held_joint_mode = "g1", "nominal_pretrain", "hard_limits_at_nominal"
        args.required_successes, args.wait_pid, args.wait_min_iteration, args.resume_checkpoint = 3, None, 0, args.resume
        print(json.dumps(TeacherSupervisorV3(args).run(), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
