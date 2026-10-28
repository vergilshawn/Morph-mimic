"""Run Stage 1 training and full-reference acceptance checks in serial chunks.

The supervisor uses only the Python standard library. Its child interpreter
loads checkpoints on CPU, trains in Isaac, and evaluates in Isaac separately.
Acceptance thresholds are local engineering criteria, not paper metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CRITERIA = {
    "root_position_rmse_m": 0.15,
    "root_position_max_error_m": 0.30,
    "joint_rmse_rad": 0.30,
    "root_orientation_rmse_rad": 0.50,
}
SOURCE_FILES = (
    "morph_mimic/envs/acp_env.py", "morph_mimic/envs/terrains.py",
    "morph_mimic/utils/motion.py", "morph_mimic/utils/reference_scene.py",
    "morph_mimic/utils/checkpoint.py", "morph_mimic/rl/runner.py",
    "morph_mimic/rl/ppo.py", "morph_mimic/rl/storage.py",
    "morph_mimic/scripts/common.py",
    "morph_mimic/utils/held_joints.py", "morph_mimic/utils/robot_cfg.py",
    "morph_mimic/utils/specs.py", "morph_mimic/cfg/configs.py",
    "morph_mimic/networks/tracker.py",
    "morph_mimic/scripts/evaluate_contact_trajectory.py",
    "morph_mimic/utils/evaluation.py",
    "morph_mimic/utils/training_profile.py", "morph_mimic/utils/transforms.py",
    "morph_mimic/scripts/evaluate.py", "morph_mimic/launcher.py",
    "morph_mimic/scripts/train_until_validated.py", "morph_mimic/scripts/train_stage1.py",
)
_METADATA_CODE = """import hashlib,json,sys,torch
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
t=c.get('contract',{}).get('training',{})
r=dict(schema=c.get('schema'),robot=c.get('robot'),stage=c.get('stage'),
 next_iteration=c.get('next_iteration'),total_steps=c.get('total_steps'),
 num_envs=t.get('num_envs'),seed=t.get('seed'),
 contract_sha256=hashlib.sha256(json.dumps(c.get('contract'),sort_keys=True).encode()).hexdigest())
print('CHECKPOINT_METADATA='+json.dumps(r,allow_nan=False),flush=True)
"""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "mojia"), default="g1")
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--chunk-iterations", type=int, default=1000)
    parser.add_argument("--required-successes", type=int, default=3)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-min-iteration", type=int, default=0,
                        help="minimum checkpoint iteration required after an existing external process exits")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-profile", choices=("acp_partial_dr", "nominal_pretrain"),
                        default="acp_partial_dr")
    parser.add_argument("--held-joint-mode", choices=("nominal_pd", "hard_limits_at_nominal"),
                        default="hard_limits_at_nominal")
    args = parser.parse_args(argv)
    for name in ("num_envs", "max_iterations", "chunk_iterations", "required_successes"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.wait_pid is not None and args.wait_pid < 1:
        parser.error("--wait-pid must be positive")
    if args.wait_min_iteration < 0 or (args.wait_min_iteration and args.wait_pid is None):
        parser.error("--wait-min-iteration requires --wait-pid and must be nonnegative")
    if len(args.clips) != 1 or Path(args.clips[0]).name != args.clips[0]:
        parser.error("exactly one clip name is required")
    for name in ("motion_dir", "run_dir", "python_executable", "resume_checkpoint"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if not args.python_executable.is_file() or not os.access(args.python_executable, os.X_OK):
        parser.error("--python-executable must be an executable interpreter file")
    if not (args.motion_dir / f"{args.clips[0]}.npz").is_file():
        parser.error("selected motion clip does not exist")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file() and args.wait_pid is None:
        parser.error("resume checkpoint does not exist")
    if (args.run_dir / "supervisor_status.json").exists():
        parser.error("refusing to overwrite an existing supervisor session; choose a fresh run directory")
    return args


def _process_identity(pid):
    """Return Linux PID start time, treating zombies as exited."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = data[data.rfind(")") + 2:].split()
    return None if fields[0] in ("Z", "X") else fields[19]


def acceptance(report):
    """A complete deterministic phase-zero evaluation must pass every gate."""
    if len(report.get("motions", [])) != 1:
        raise ValueError("Expected exactly one evaluated motion")
    motion = report["motions"][0]
    failures = []
    gates = {
        "contract_verified": report.get("contract_verified") is True and motion.get("contract_verified") is True,
        "full_evaluation": report.get("evaluation_is_full") is True,
        "phase_zero": motion.get("start_phase") == 0.0 and report.get("start_phase", 0.0) == 0.0,
        "completed": motion.get("completed") is True,
        "no_physical_failure": motion.get("failure") is False and motion.get("terminated") is False,
        "no_budget_or_interruption": motion.get("budget_limited") is False and motion.get("interrupted") is False,
    }
    failures.extend(name for name, passed in gates.items() if not passed)
    for name, limit in CRITERIA.items():
        value = motion.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Evaluation lacks a finite nonnegative {name}")
        if value > limit:
            failures.append(f"{name}={value:.8g} exceeds {limit}")
    if motion.get("held_joint_mode") == "hard_limits_at_nominal":
        error = motion.get("physical_diagnostics", {}).get("held_joint_position_error_max_rad")
        if isinstance(error, bool) or not isinstance(error, (float, int)) or not math.isfinite(error) or error < 0:
            raise ValueError("Held-joint constraints require finite pre-reset dynamic deviation diagnostics")
        if error > .01:
            failures.append(f"held_joint_position_error_max_rad={error:.8g} exceeds 0.01")
    return not failures, failures


def run_child(command, log_path, *, env, on_start):
    """Run one child with durable separate stdout and no background GPU work."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            on_start(process.pid)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


class Supervisor:
    def __init__(self, args, *, child_runner=run_child, sleeper=time.sleep):
        self.args = args
        self.child_runner = child_runner
        self.sleeper = sleeper
        self.path = args.run_dir / "supervisor_status.json"
        self.status = dict(
            schema=1, state="running", phase="setup", supervisor_pid=os.getpid(),
            supervisor_pgid=os.getpgrp(), child_pid=None,
            current_iteration=None, actual_total_steps=None, consecutive_successes=0,
            required_successes=args.required_successes, maximum_iterations=args.max_iterations,
            chunk_iterations=args.chunk_iterations, robot=args.robot, clips=args.clips,
            num_envs=args.num_envs, seed=args.seed, device=args.device,
            training_profile=getattr(args, "training_profile", "acp_partial_dr"),
            held_joint_mode=getattr(args, "held_joint_mode", "hard_limits_at_nominal"),
            training_profile_description=(
                "Nominal dynamics/observations pretraining; not the full ACP domain-randomized reproduction"
                if getattr(args, "training_profile", "acp_partial_dr") == "nominal_pretrain" else
                "ACP-aligned training with the project's implemented partial domain randomization"),
            motion_dir=str(args.motion_dir), run_dir=str(args.run_dir),
            wait_pid=args.wait_pid, criteria=CRITERIA,
            criteria_provenance="Local engineering acceptance thresholds, not paper-reported metrics",
            success_definition="Consecutive different checkpoints passing complete deterministic phase-zero reference evaluation",
            contact_metrics_role="Diagnostic only; contact F1 does not modify the full phase-zero acceptance gates",
            evaluations=[], children=[], source_sha256={},
        )
        self._probe_id = 0

    def save(self):
        self.status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.path, self.status)

    def _sources(self, *, initialize=False):
        if initialize:
            manifest = self.args.run_dir / "launch_manifest.json"
            expected = json.loads(manifest.read_text()).get("source_sha256", {}) if manifest.exists() else {}
            # Keep every supplied launch hash, and freeze newly covered source
            # files even when a compatible older manifest listed fewer files.
            self.status["source_sha256"] = {
                **{name: _sha256(PROJECT_ROOT / name) for name in SOURCE_FILES}, **expected,
            }
        changed = [name for name, digest in self.status["source_sha256"].items()
                   if _sha256(PROJECT_ROOT / name) != digest]
        if changed:
            raise ValueError(f"Frozen training source changed: {', '.join(changed)}")

    def _child(self, command, log_name, *, phase, cpu_only=False):
        self._sources()
        log_path = self.args.run_dir / "logs" / log_name
        record = dict(command=[str(item) for item in command], log=str(log_path), phase=phase, pid=None, returncode=None)
        self.status["children"].append(record)
        self.status["phase"] = phase
        self.save()

        def started(pid):
            record["pid"] = pid
            try:
                record["pgid"] = os.getpgid(pid)
            except ProcessLookupError:
                record["pgid"] = None
            self.status["child_pid"] = pid
            self.save()

        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if cpu_only:
            env["CUDA_VISIBLE_DEVICES"] = ""
        returncode = self.child_runner(record["command"], log_path, env=env, on_start=started)
        record["returncode"] = returncode
        self.status["child_pid"] = None
        self.save()
        if returncode != 0:
            raise RuntimeError(f"{phase} exited with status {returncode}; see {log_path}")
        self._sources()
        return log_path

    def _metadata(self, checkpoint):
        self._probe_id += 1
        path = self._child(
            [str(self.args.python_executable), "-c", _METADATA_CODE, str(checkpoint)],
            f"checkpoint_metadata_{self._probe_id:04d}.log", phase="checkpoint_metadata", cpu_only=True)
        lines = [line.partition("=")[2] for line in path.read_text().splitlines()
                 if line.startswith("CHECKPOINT_METADATA=")]
        if len(lines) != 1:
            raise ValueError(f"Checkpoint metadata probe did not return one result: {path}")
        metadata = json.loads(lines[0])
        if metadata.get("schema") != 2 or metadata.get("stage") != 1 or metadata.get("robot") != self.args.robot:
            raise ValueError("Checkpoint must have schema 2 and matching Stage 1 robot metadata")
        for name in ("next_iteration", "total_steps"):
            if type(metadata.get(name)) is not int or metadata[name] < 0:
                raise ValueError(f"Checkpoint has invalid {name}")
        if metadata.get("num_envs") != self.args.num_envs or metadata.get("seed") != self.args.seed:
            raise ValueError("Checkpoint environment count/seed differs from this strict continuation")
        return metadata

    def _snapshot(self, source, *, expected_iteration=None):
        metadata = self._metadata(source)
        iteration = metadata["next_iteration"]
        if expected_iteration is not None and iteration != expected_iteration:
            raise ValueError(f"Training ended at iteration {iteration}, expected {expected_iteration}")
        if iteration > self.args.max_iterations:
            raise ValueError("Initial checkpoint exceeds the requested maximum iteration budget")
        destination = self.args.run_dir / "checkpoints" / f"iter{iteration:06d}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Training writes final.pt atomically; linking retains that immutable inode.
        os.link(source, destination)
        digest = _sha256(destination)
        self.status.update(current_iteration=iteration, actual_total_steps=metadata["total_steps"],
                           checkpoint=str(destination), checkpoint_sha256=digest, checkpoint_metadata=metadata)
        self.save()
        return destination, metadata, digest

    def _evaluate(self, checkpoint, metadata, digest):
        iteration = metadata["next_iteration"]
        directory = self.args.run_dir / "evaluations"
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / f"iter{iteration:06d}.json"
        trajectory = directory / f"iter{iteration:06d}.npz"
        self._child([
            str(self.args.python_executable), "-m", "morph_mimic.scripts.evaluate",
            "--robot", self.args.robot, "--stage", "1", "--device", self.args.device,
            "--headless", "--seed", str(self.args.seed), "--resume", str(checkpoint),
            "--motion-dir", str(self.args.motion_dir), "--clips", *self.args.clips,
            "--held-joint-mode", getattr(self.args, "held_joint_mode", "hard_limits_at_nominal"),
            "--start-phase", "0", "--out", str(report_path), "--trajectory", str(trajectory),
        ], f"evaluate_iter{iteration:06d}.log", phase="evaluate")
        report = json.loads(report_path.read_text())
        if report.get("checkpoint_sha256") != digest or not trajectory.is_file():
            raise ValueError("Evaluation checkpoint hash or physical trajectory artifact is missing/mismatched")
        if report["motions"][0].get("clip") != self.args.clips[0]:
            raise ValueError("Evaluation selected a different reference clip")
        contact_report_path = directory / f"iter{iteration:06d}_contacts.json"
        self._child([
            str(self.args.python_executable), "-m", "morph_mimic.scripts.evaluate_contact_trajectory",
            "--trajectory", str(trajectory),
            "--motion", str(self.args.motion_dir / f"{self.args.clips[0]}.npz"),
            "--labels", str(self.args.motion_dir / f"{self.args.clips[0]}_contacts.npz"),
            "--out", str(contact_report_path),
        ], f"contacts_iter{iteration:06d}.log", phase="contact_metrics", cpu_only=True)
        contact_report = json.loads(contact_report_path.read_text())
        contact_coverage = contact_report["coverage"]
        self.status.update(latest_contact_report=str(contact_report_path),
                           latest_contact_coverage=contact_coverage)
        passed, reasons = acceptance(report)
        if any(item["checkpoint_sha256"] == digest for item in self.status["evaluations"]):
            raise ValueError("A checkpoint cannot count twice toward consecutive acceptance")
        count = self.status["consecutive_successes"] + 1 if passed else 0
        self.status["consecutive_successes"] = count
        self.status["evaluations"].append(dict(
            iteration=iteration, actual_total_steps=metadata["total_steps"], checkpoint=str(checkpoint),
            checkpoint_sha256=digest, report=str(report_path), trajectory=str(trajectory),
            contact_report=str(contact_report_path), contact_coverage=contact_coverage,
            contact_metrics=contact_report["micro_force_history_vs_reference"],
            passed=passed, reasons=reasons, consecutive_successes=count,
            metrics={name: report["motions"][0][name] for name in CRITERIA},
        ))
        self.save()

    def _train(self, checkpoint, target):
        command = [str(self.args.python_executable), "-m", "morph_mimic.scripts.train_stage1",
                   "--robot", self.args.robot, "--device", self.args.device, "--headless",
                   "--num_envs", str(self.args.num_envs), "--seed", str(self.args.seed),
                   "--motion_dir", str(self.args.motion_dir), "--clips", *self.args.clips,
                   "--max_iterations", str(target), "--save_interval", str(self.args.chunk_iterations),
                   "--log_dir", str(self.args.run_dir),
                   "--training-profile", getattr(self.args, "training_profile", "acp_partial_dr"),
                   "--held-joint-mode", getattr(self.args, "held_joint_mode", "hard_limits_at_nominal")]
        if checkpoint is not None:
            command += ["--resume", str(checkpoint), "--continue_run"]
        self.status["target_iteration"] = target
        self._child(command, f"train_until_iter{target:06d}.log", phase="train")
        return self._snapshot(self.args.run_dir / "final.pt", expected_iteration=target)

    def run(self):
        self.args.run_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.args.run_dir / ".train_until_validated.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self._sources(initialize=True)
                self.save()
                if self.args.wait_pid is not None:
                    self.status["phase"] = "waiting_for_existing_training"
                    self.save()
                    identity = _process_identity(self.args.wait_pid)
                    while identity is not None:
                        self.sleeper(10)
                        current = _process_identity(self.args.wait_pid)
                        if current is None:
                            break
                        if current != identity:
                            raise RuntimeError("The waited PID was reused; refusing ambiguous continuation")
                checkpoint = self.args.resume_checkpoint
                if checkpoint is None and (self.args.run_dir / "final.pt").is_file():
                    checkpoint = self.args.run_dir / "final.pt"
                if self.args.wait_pid is not None and checkpoint is None:
                    raise RuntimeError("Existing training exited without a final checkpoint")
                if checkpoint is None:
                    checkpoint, metadata, digest = self._train(None, min(self.args.chunk_iterations, self.args.max_iterations))
                else:
                    checkpoint, metadata, digest = self._snapshot(checkpoint)
                    if metadata["next_iteration"] < self.args.wait_min_iteration:
                        raise RuntimeError("External training ended before its required minimum iteration")
                while True:
                    self._evaluate(checkpoint, metadata, digest)
                    if self.status["consecutive_successes"] >= self.args.required_successes:
                        self.status.update(state="validated", phase="finished")
                        break
                    if metadata["next_iteration"] >= self.args.max_iterations:
                        self.status.update(state="budget_exhausted", phase="finished")
                        break
                    target = min(metadata["next_iteration"] + self.args.chunk_iterations, self.args.max_iterations)
                    checkpoint, metadata, digest = self._train(checkpoint, target)
                self.save()
            except BaseException as error:
                self.status.update(state="failed", phase="stopped", error_type=type(error).__name__,
                                   error=str(error), traceback=traceback.format_exc())
                self.save()
                raise
        return self.status


def main():
    args = _arguments()
    supervisor = Supervisor(args)
    try:
        status = supervisor.run()
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
    print(json.dumps({key: status[key] for key in
                      ("state", "current_iteration", "actual_total_steps", "consecutive_successes")}), flush=True)


if __name__ == "__main__":
    main()
