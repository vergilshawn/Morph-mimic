"""Independent nominal standing auxiliary experiment, not ACP box-climb success.

The standard-library supervisor runs Isaac training and evaluation in separate
children. Network architecture, PPO, rewards and frozen ACP core are reused.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from morph_mimic.scripts.train_until_validated import PROJECT_ROOT, Supervisor, _sha256

CRITERIA = {"max_root_xy_drift_m": .15, "max_root_height_drop_m": .10,
            "max_root_tilt_rad": .30, "max_held_joint_error_rad": .01}
EXTRA_SOURCES = ("morph_mimic/scripts/train_standing_baseline.py",
                 "morph_mimic/scripts/prepare_standing_reference.py")


def standing_metrics(path):
    """Measure the actual pre-reset trajectory against the static reference."""
    import numpy as np

    with np.load(path, allow_pickle=False) as arrays:
        positions, reference, quats, held, joints, elapsed = [np.asarray(arrays[key]) for key in
            ("root_pos_w", "ref_root_pos_w", "root_quat_w", "held_joint_position_error_rad",
             "all_joint_pos", "elapsed_control_time_s")]
    n = len(positions)
    if not n or any(a.shape != shape for a, shape in zip(
            (positions, reference, quats, held, joints, elapsed),
            ((n, 3), (n, 3), (n, 4), (n, 6), (n, 29), (n,)))):
        raise ValueError("Standing requires nonempty 29-axis pre-reset physical trajectory data")
    if not all(np.isfinite(a).all() for a in (positions, reference, quats, held, joints, elapsed)):
        raise ValueError("Standing trajectory contains nonfinite state")
    norms = np.linalg.norm(quats, axis=1)
    if (norms < 1e-8).any() or not np.allclose(reference, reference[0], atol=1e-6, rtol=0):
        raise ValueError("Standing requires valid quaternions and a static reference root")
    if not np.allclose(elapsed, (np.arange(n) + 1) * .02, atol=1e-6, rtol=0):
        raise ValueError("Standing requires uninterrupted 50 Hz control endpoints")
    q = quats / norms[:, None]
    return dict(samples=n, duration_s=float(elapsed[-1]),
        max_root_xy_drift_m=float(np.linalg.norm(positions[:, :2] - reference[0, :2], axis=1).max()),
        max_root_height_drop_m=max(0., float((reference[0, 2] - positions[:, 2]).max())),
        max_root_tilt_rad=float(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)).max()),
        max_held_joint_error_rad=float(np.abs(held).max()))


def standing_acceptance(report):
    motions = report.get("motions", [])
    if len(motions) != 1:
        raise ValueError("Standing requires one evaluated reference")
    motion, metrics = motions[0], report.get("standing_metrics", {})
    gates = dict(contract_verified=report.get("contract_verified") is True and motion.get("contract_verified") is True,
        full_evaluation=report.get("evaluation_is_full") is True,
        phase_zero=report.get("start_phase") == 0. and motion.get("start_phase") == 0.,
        completed=motion.get("completed") is True,
        no_failure=all(motion.get(key) is False for key in ("failure", "terminated", "budget_limited", "interrupted")),
        full_ten_seconds=metrics.get("samples") == 500 and math.isclose(metrics.get("duration_s", -1), 10., abs_tol=1e-6))
    failures = [key for key, passed in gates.items() if not passed]
    for key, limit in CRITERIA.items():
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Standing evaluation lacks finite nonnegative {key}")
        if value >= limit:
            failures.append(f"{key}={value:.8g} must be below {limit}")
    return not failures, failures


def _install_environment_override():
    from morph_mimic.scripts import common

    original = common.build_env
    def build(*args, **kwargs):
        kwargs.update(use_usd=False, csi_enabled=False, flat_reference=True,
                      training_profile="nominal_pretrain", held_joint_mode="hard_limits_at_nominal",
                      terrain_types=("flat",), ref_terrain_types=("flat",), num_terrain_levels=1)
        return original(*args, **kwargs)
    common.build_env = build


def _check_resume(args):
    if args.resume is None:
        return
    import torch

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract") or {}
    if (checkpoint.get("schema") != 2 or checkpoint.get("robot") != "g1" or checkpoint.get("stage") != 1
            or set(contract.get("clips", {})) != set(args.clips)
            or contract.get("training", {}).get("csi_enabled") is not False):
        raise ValueError("Resume requires this standing experiment's strict schema-2 contract; box-climb/legacy weights are refused")


def _train(args):
    from morph_mimic.scripts import common

    _check_resume(args)
    _install_environment_override()
    values = vars(args) | dict(robot="g1", motion_dir=str(args.motion_dir),
        resume=str(args.resume) if args.resume else None, continue_run=args.resume is not None,
        log_dir=str(args.run_dir), save_interval=args.chunk_iterations, flat_reference=True,
        training_profile="nominal_pretrain", held_joint_mode="hard_limits_at_nominal", headless=True)
    common.run_training(SimpleNamespace(**values), 1)


def _evaluate(args):
    from morph_mimic.scripts import evaluate

    _check_resume(args)
    _install_environment_override()
    original = evaluate._write_report
    def write(path, report):
        report.update(scope="Learned nominal standing auxiliary check; not dynamic tracking or ACP box-climb validation",
                      standing_criteria=CRITERIA, standing_metrics=standing_metrics(args.trajectory))
        report["standing_passed"], report["standing_failure_reasons"] = standing_acceptance(report)
        original(path, report)
    evaluate._write_report = write
    sys.argv = [sys.argv[0], "--robot", "g1", "--stage", "1", "--device", args.device,
        "--seed", str(args.seed), "--resume", str(args.resume), "--motion-dir", str(args.motion_dir),
        "--clips", *args.clips, "--out", str(args.out), "--trajectory", str(args.trajectory), "--start-phase", "0"]
    evaluate.main()


class StandingSupervisor(Supervisor):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.status.update(criteria=CRITERIA, scope="Nominal standing auxiliary; not ACP box climbing",
            success_definition="Three distinct checkpoints each standing continuously for 10 seconds from phase zero",
            contact_metrics_role="Diagnostic only; standing gates use actual pose and held coordinates")

    def _sources(self, *, initialize=False):
        super()._sources(initialize=initialize)
        if initialize:
            for name in EXTRA_SOURCES:
                self.status["source_sha256"].setdefault(name, _sha256(PROJECT_ROOT / name))
            super()._sources()

    def _command(self, mode):
        return [str(self.args.python_executable), "-m", "morph_mimic.scripts.train_standing_baseline", mode,
            "--motion-dir", str(self.args.motion_dir), "--clips", *self.args.clips,
            "--device", self.args.device, "--seed", str(self.args.seed)]

    def _train(self, checkpoint, target):
        command = self._command("train") + ["--run-dir", str(self.args.run_dir),
            "--num-envs", str(self.args.num_envs), "--max-iterations", str(target),
            "--chunk-iterations", str(self.args.chunk_iterations)]
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
                or not trajectory.is_file() or _sha256(trajectory) != report["motions"][0].get("trajectory_sha256")):
            raise ValueError("Standing evaluation checkpoint, clip or physical trajectory hash differs")
        passed, reasons = standing_acceptance(report)
        if any(item["checkpoint_sha256"] == digest for item in self.status["evaluations"]):
            raise ValueError("A checkpoint cannot count twice toward standing acceptance")
        self.status["consecutive_successes"] = self.status["consecutive_successes"] + 1 if passed else 0
        self.status["evaluations"].append(dict(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            report=str(report_path), trajectory=str(trajectory), actual_total_steps=metadata["total_steps"],
            passed=passed, reasons=reasons, metrics=report["standing_metrics"],
            consecutive_successes=self.status["consecutive_successes"]))
        self.save()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "supervise"))
    parser.add_argument("--motion-dir", type=Path, default=PROJECT_ROOT / "datasets/g1_standing_reference_v1")
    parser.add_argument("--clips", nargs=1, default=["standing_nominal_g1"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--chunk-iterations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    for name in ("motion_dir", "run_dir", "resume", "out", "trajectory", "python_executable"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if min(args.num_envs, args.max_iterations, args.chunk_iterations) < 1:
        parser.error("environment and iteration counts must be positive")
    if Path(args.clips[0]).name != args.clips[0] or not (args.motion_dir / f"{args.clips[0]}.npz").is_file():
        parser.error("selected standing motion does not exist")
    if args.resume is not None and not args.resume.is_file():
        parser.error("resume checkpoint does not exist")
    if args.mode == "evaluate" and (args.resume is None or args.out is None or args.trajectory is None):
        parser.error("evaluate requires --resume, --out and --trajectory")
    if args.mode != "evaluate" and args.run_dir is None:
        parser.error("train/supervise requires --run-dir")
    if args.mode == "supervise" and (args.run_dir / "supervisor_status.json").exists():
        parser.error("choose a fresh supervisor run directory")
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
        status = StandingSupervisor(args).run()
        print(json.dumps({key: status[key] for key in ("state", "current_iteration", "consecutive_successes")}), flush=True)


if __name__ == "__main__":
    main()
