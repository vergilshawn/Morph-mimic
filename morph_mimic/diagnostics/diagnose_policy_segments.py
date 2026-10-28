"""Inspect which reference segments a frozen policy can execute.

Each segment runs in a fresh simulator process, using the existing strict
evaluator. Optional phase-zero continuation suppresses only the root-error
*termination*, after recording the original decision; height, grace and
end-of-reference rules remain active. Diagnostic artifacts never qualify as
skill-acceptance evidence. The coordinating process imports only the standard
library and never creates a CUDA context.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASES = (0.0, 0.2, 0.34, 0.4, 0.5, 0.7, 0.9)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "mojia"), default="g1")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--phases", type=float, nargs="+", default=DEFAULT_PHASES)
    parser.add_argument("--continue-phase-zero", action="store_true",
                        help="also observe phase zero beyond its first root-error stop; diagnostic only")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--held-joint-mode", choices=("nominal_pd", "hard_limits_at_nominal"),
                        default="hard_limits_at_nominal")
    parser.add_argument("--planner", choices=("acp", "mlp"), default="acp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--job-timeout-s", type=float, default=600.,
                        help="maximum wall time for each isolated simulator process (default: 600)")
    parser.add_argument("--internal-single-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-job-label", help=argparse.SUPPRESS)
    parser.add_argument("--internal-continuation", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.resume.is_file():
        parser.error(f"checkpoint not found: {args.resume}")
    if not args.motion_dir.is_dir():
        parser.error(f"motion directory not found: {args.motion_dir}")
    if not (args.motion_dir / f"{args.clip}.npz").is_file():
        parser.error(f"motion clip not found: {args.clip}")
    if any(not math.isfinite(p) or not 0 <= p < 1 for p in args.phases):
        parser.error("all phases must be finite and in [0,1)")
    if len(args.phases) != len(set(args.phases)):
        parser.error("phases must be distinct")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not math.isfinite(args.job_timeout_s) or args.job_timeout_s <= 0:
        parser.error("--job-timeout-s must be finite and positive")
    if args.internal_single_job:
        if len(args.phases) != 1 or not args.internal_job_label or not args.out_dir.is_dir():
            parser.error("internal single job requires one phase, a label and an existing output directory")
        if Path(args.internal_job_label).name != args.internal_job_label:
            parser.error("invalid internal job label")
        if args.internal_continuation and args.phases[0] != 0.:
            parser.error("continuation is supported at phase zero only")
        for suffix in (".json", ".npz"):
            if (args.out_dir / f"{args.internal_job_label}{suffix}").exists():
                parser.error("refusing to overwrite single-job artifacts")
    elif args.out_dir.exists():
        parser.error(f"refusing to reuse output directory: {args.out_dir}")
    return args


def _continue_past_root_error(env):
    """Change the returned done mask, leaving configuration/contract unchanged."""
    import torch

    original = env._get_dones

    def get_dones(self):
        terminated, truncated = original()
        snapshot = self.extras.get("final_tracking")
        if snapshot is None:
            raise RuntimeError("Diagnostic continuation requires physical endpoint snapshots")
        if not bool(self.is_ref_env.all()):
            raise RuntimeError("Diagnostic continuation supports reference environments only")
        snapshot["strict_terminated"] = terminated.detach().clone()
        snapshot["strict_reference_error_exceeded"] = snapshot["reference_error_exceeded"].detach().clone()
        # grace_active means the settling grace period is still in force.
        height_failure = ((snapshot["root_pos_w"][:, 2] < snapshot["termination_height_threshold_m"])
                          & ~snapshot["grace_active"])
        snapshot["root_error_termination_enabled"] = torch.zeros_like(terminated)
        snapshot["terminated"] = height_failure.detach().clone()
        snapshot["died"] = height_failure.detach().clone()
        return height_failure, truncated

    env._get_dones = MethodType(get_dones, env)


def _diagnostic_result(result, *, continuation, first_original_termination):
    result.update(diagnostic_only=True, eligible_for_skill_acceptance=False,
                  root_error_termination_enabled=not continuation,
                  first_original_termination=first_original_termination)
    if continuation:
        # Retain what happened without using completion/success fields for a
        # trajectory whose original strict stopping rule has been bypassed.
        result["diagnostic_observed_outcome"] = {
            key: result.get(key) for key in ("completed", "segment_completed", "failure",
                                            "budget_limited", "terminated", "truncated", "end_phase")
        }
        result["observed_reference_end"] = bool(result.get("segment_completed", False))
        result["completed"] = False
        result["segment_completed"] = False
    return result


def _trajectory_type(continuation, source_hashes=None):
    import numpy as np
    import torch
    from morph_mimic.utils.evaluation import TrackingTrajectory

    class SegmentTrajectory(TrackingTrajectory):
        _fields = {**TrackingTrajectory._fields, "strict_terminated": (),
                   "strict_reference_error_exceeded": (), "root_error_termination_enabled": ()}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.first_original_termination = None

        def update(self, snapshot, *, action, reward, env_id=0):
            snapshot = dict(snapshot)
            snapshot.setdefault("strict_terminated", snapshot["terminated"])
            snapshot.setdefault("strict_reference_error_exceeded", snapshot["reference_error_exceeded"])
            snapshot.setdefault("root_error_termination_enabled", torch.ones_like(snapshot["terminated"]))
            super().update(snapshot, action=action, reward=reward, env_id=env_id)
            row = self._rows[-1]
            if self.first_original_termination is None and bool(row["strict_terminated"]):
                root_error = float(np.linalg.norm(row["root_pos_w"] - row["ref_root_pos_w"]))
                self.first_original_termination = dict(
                    step=len(self._rows), elapsed_control_time_s=len(self._rows) * self.control_dt,
                    phase=float(row["phase"]), root_position_error_m=root_error,
                    root_position_m=row["root_pos_w"].tolist(),
                    reference_root_position_m=row["ref_root_pos_w"].tolist(),
                    root_error_threshold_m=float(row["termination_root_error_threshold_m"]),
                    height_threshold_m=float(row["termination_height_threshold_m"]),
                    root_error_exceeded=bool(row["strict_reference_error_exceeded"]),
                    height_below_threshold=bool(row["root_pos_w"][2] < row["termination_height_threshold_m"]),
                    grace_active=bool(row["grace_active"]),
                    definition="First control endpoint where the unchanged original engineering termination rule fires",
                )

        def summary(self):
            summary = super().summary()
            summary.update(diagnostic_only=True, eligible_for_skill_acceptance=False,
                           root_error_termination_enabled=not continuation,
                           first_original_termination=self.first_original_termination)
            if continuation:
                last = self._rows[-1]
                summary["failure_reason"] = (
                    "root_below_height_threshold" if bool(last["died"]) else
                    "reference_end" if bool(last["reached_end"]) else
                    "episode_time_limit" if bool(last["time_out"]) else
                    "external_budget_or_interruption"
                )
            return summary

        def write(self, path, *, metadata):
            _diagnostic_result(metadata["result"], continuation=continuation,
                               first_original_termination=self.first_original_termination)
            metadata.update(diagnostic_only=True, eligible_for_skill_acceptance=False,
                            root_error_termination_enabled=not continuation,
                            first_original_termination=self.first_original_termination,
                            diagnostic_source_sha256_by_path=source_hashes or {},
                            threshold_semantics="Root threshold remains recorded; root_error_termination_enabled determines whether it stops this diagnostic")
            return super().write(path, metadata=metadata)

    return SegmentTrajectory


@contextmanager
def _evaluation_hooks(continuation, source_hashes=None):
    from morph_mimic.scripts import common, evaluate

    original_build = common.build_env
    original_trajectory = evaluate.TrackingTrajectory

    def build(*args, **kwargs):
        env, extra = original_build(*args, **kwargs)
        if continuation:
            _continue_past_root_error(env)
        return env, extra

    common.build_env = build
    evaluate.TrackingTrajectory = _trajectory_type(continuation, source_hashes)
    try:
        yield
    finally:
        common.build_env = original_build
        evaluate.TrackingTrajectory = original_trajectory


def _source_hashes():
    paths = sorted((PROJECT_ROOT / "morph_mimic").rglob("*.py"))
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _base_metadata(args, source_hashes):
    return dict(robot=args.robot, stage=args.stage, checkpoint=str(args.resume.resolve()),
                checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                motion_dir=str(args.motion_dir.resolve()), clip=args.clip,
                source_sha256=hashlib.sha256((args.motion_dir / f"{args.clip}.npz").read_bytes()).hexdigest(),
                seed=args.seed, device=args.device, diagnostic_only=True,
                eligible_for_skill_acceptance=False, source_sha256_by_path=source_hashes,
                scope="Frozen-policy segment diagnosis; suffix completion and relaxed continuation do not certify box climbing")


def _write_json(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".json.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _child_command(args, phase, continuation, label):
    command = [sys.executable, "-m", "morph_mimic.diagnostics.diagnose_policy_segments",
               "--robot", args.robot, "--stage", str(args.stage),
               "--resume", str(args.resume.resolve()), "--motion-dir", str(args.motion_dir.resolve()),
               "--clip", args.clip, "--phases", str(phase), "--out-dir", str(args.out_dir.resolve()),
               "--held-joint-mode", args.held_joint_mode, "--planner", args.planner,
               "--device", args.device, "--seed", str(args.seed),
               "--headless" if args.headless else "--no-headless",
               "--internal-single-job", "--internal-job-label", label]
    if args.max_steps is not None:
        command.extend(("--max-steps", str(args.max_steps)))
    if continuation:
        command.append("--internal-continuation")
    return command


def _run_isolated(args):
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "logs").mkdir()
    base = _base_metadata(args, _source_hashes())
    jobs = [(phase, False, f"segment_{index:02d}_phase_{phase:.6f}")
            for index, phase in enumerate(args.phases)]
    if args.continue_phase_zero:
        jobs.append((0., True, "phase_zero_continuation"))
    _write_json(args.out_dir / "manifest.json", dict(base, phases=list(args.phases),
                continue_phase_zero=args.continue_phase_zero,
                process_isolation="one fresh interpreter and SimulationApp per diagnostic job",
                job_timeout_s=args.job_timeout_s,
                commands=[_child_command(args, *job) for job in jobs]))
    reports, processes = [], []
    try:
        for phase, continuation, label in jobs:
            command = _child_command(args, phase, continuation, label)
            log_path = args.out_dir / "logs" / f"{label}.log"
            with log_path.open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
                process = dict(label=label, pid=child.pid, command=command, log=str(log_path.resolve()))
                processes.append(process)
                print(f"START {label} pid={child.pid} log={log_path}", flush=True)
                try:
                    process["returncode"] = child.wait(timeout=args.job_timeout_s)
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                    process["returncode"] = child.returncode
                    raise
            if process["returncode"] != 0:
                raise RuntimeError(f"{label}: simulator exited {process['returncode']}; see {log_path}")
            report_path = args.out_dir / f"{label}.json"
            trajectory_path = args.out_dir / f"{label}.npz"
            if not report_path.is_file() or not trajectory_path.is_file():
                raise RuntimeError(f"{label}: simulator did not publish both report and physical trajectory")
            report = json.loads(report_path.read_text())
            if report["checkpoint_sha256"] != base["checkpoint_sha256"]:
                raise RuntimeError(f"{label}: frozen checkpoint changed during diagnostic sweep")
            reports.append(report)
            print(json.dumps(dict(label=label, **report["result"]), ensure_ascii=False, allow_nan=False), flush=True)
            if report["result"]["interrupted"]:
                break
        _write_json(args.out_dir / "summary.json", dict(base, state="completed", evaluations=reports,
                                                       processes=processes))
        print(f"Saved isolated diagnostic sweep: {args.out_dir}", flush=True)
    except BaseException as error:
        _write_json(args.out_dir / "interrupted_summary.json", dict(
            base, state="incomplete", error=f"{type(error).__name__}: {error}",
            evaluations=reports, processes=processes))
        raise


def _run_single(args):
    import torch
    from morph_mimic.scripts import evaluate
    from morph_mimic.networks.acp import ACPPlanner, MLPPlanner
    from morph_mimic.networks.tracker import TrackerActor
    from morph_mimic.utils.specs import RobotSpec

    spec = RobotSpec.load(args.robot)
    metadata = evaluate._clip_metadata(args.motion_dir.resolve(), [args.clip], spec)[0]
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    if checkpoint.get("robot") != args.robot or checkpoint.get("stage") != args.stage:
        raise ValueError("Checkpoint robot and stage must match")
    if checkpoint.get("contract") is None:
        raise ValueError("Segment diagnostics require a physical checkpoint contract")
    actor = TrackerActor().to(args.device).eval()
    actor.load_state_dict(checkpoint["ppo"]["actor"])
    planner = None
    if args.stage == 2:
        if checkpoint.get("planner") is None:
            raise ValueError("Stage 2 requires a trained planner")
        planner = (ACPPlanner if args.planner == "acp" else MLPPlanner)().to(args.device).eval()
        planner.load_state_dict(checkpoint["planner"])
    source_hashes = _source_hashes()
    base = _base_metadata(args, source_hashes)
    from morph_mimic.launcher import launch, close
    app = launch(headless=args.headless, device=args.device)
    try:
        phase, continuation, label = args.phases[0], args.internal_continuation, args.internal_job_label
        call_args = SimpleNamespace(**vars(args))
        call_args.start_phase = phase
        call_args.trajectory = args.out_dir / f"{label}.npz"
        with _evaluation_hooks(continuation, source_hashes):
            result = evaluate._evaluate_clip(call_args, checkpoint, metadata, actor, planner, app)
        if not result["contract_verified"]:
            raise RuntimeError("Diagnostic requires verified physical contract")
        report = dict(base, start_phase=phase, root_error_termination_enabled=not continuation,
                      continuation=continuation,
                      first_original_termination=result["first_original_termination"], result=result)
        evaluate._write_report(args.out_dir / f"{label}.json", report)
        print(json.dumps(dict(label=label, **result), ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        close(app)


def main():
    args = _arguments()
    if args.internal_single_job:
        _run_single(args)
    else:
        _run_isolated(args)


if __name__ == "__main__":
    main()
