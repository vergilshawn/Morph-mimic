"""Supervisor control flow with real artifact/status I/O and fake GPU children."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from morph_mimic.scripts.train_until_validated import Supervisor


class FakeChildren:
    def __init__(self, args, outcomes, *, fail_train=False):
        self.args = args
        self.outcomes = iter(outcomes)
        self.fail_train = fail_train
        self.commands = []

    def checkpoint(self, path, iteration):
        path.parent.mkdir(parents=True, exist_ok=True)
        value = dict(schema=2, robot="g1", stage=1, next_iteration=iteration,
                     total_steps=iteration * 24 * 1024, num_envs=1024, seed=0)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(path)

    def __call__(self, command, log_path, *, env, on_start):
        self.commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        on_start(99999999)
        if command[1] == "-c":
            assert env["CUDA_VISIBLE_DEVICES"] == ""
            metadata = json.loads(Path(command[-1]).read_text())
            log_path.write_text("CHECKPOINT_METADATA=" + json.dumps(metadata) + "\n")
        elif command[2].endswith("train_stage1"):
            log_path.write_text("fake train\n")
            if self.fail_train:
                return 7
            target = int(command[command.index("--max_iterations") + 1])
            self.checkpoint(self.args.run_dir / "final.pt", target)
        elif command[2].endswith("evaluate"):
            log_path.write_text("fake evaluate\n")
            checkpoint = Path(command[command.index("--resume") + 1])
            report_path = Path(command[command.index("--out") + 1])
            trajectory = Path(command[command.index("--trajectory") + 1])
            passed = next(self.outcomes)
            motion = dict(clip="box", start_phase=0.0, completed=passed, failure=not passed,
                          terminated=not passed, budget_limited=False, interrupted=False,
                          contract_verified=True, root_position_rmse_m=.10 if passed else .4,
                          root_position_max_error_m=.2 if passed else .6,
                          joint_rmse_rad=.2, root_orientation_rmse_rad=.3)
            report = dict(contract_verified=True, evaluation_is_full=True,
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), motions=[motion])
            report_path.write_text(json.dumps(report))
            trajectory.write_bytes(b"fake-trajectory")
        else:
            raise AssertionError(command)
        return 0


class TestTrainingSupervisor(unittest.TestCase):
    def args(self, directory, **overrides):
        values = dict(run_dir=Path(directory), motion_dir=Path(directory) / "motions", clips=["box"],
                      python_executable=Path("/fake/isaac/python"), num_envs=1024, seed=0,
                      robot="g1", device="cuda:0", max_iterations=20000, chunk_iterations=1000,
                      required_successes=3, wait_pid=None, wait_min_iteration=0, resume_checkpoint=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_absolute_iteration_resume_and_three_distinct_successful_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            children = FakeChildren(args, [True, True, True])
            children.checkpoint(args.run_dir / "final.pt", 1000)
            result = Supervisor(args, child_runner=children).run()
            self.assertEqual(result["state"], "validated")
            self.assertEqual(result["current_iteration"], 3000)
            self.assertEqual(result["actual_total_steps"], 3000 * 24 * 1024)
            training = [c for c in children.commands if len(c) > 2 and c[2].endswith("train_stage1")]
            self.assertEqual([c[c.index("--max_iterations") + 1] for c in training], ["2000", "3000"])
            self.assertTrue(all("--continue_run" in c for c in training))
            self.assertEqual(len({item["checkpoint_sha256"] for item in result["evaluations"]}), 3)
            self.assertEqual(json.loads((args.run_dir / "supervisor_status.json").read_text())["state"], "validated")
            # Immutable initial snapshot survives final.pt being atomically replaced.
            initial = args.run_dir / "checkpoints/iter001000.pt"
            self.assertEqual(json.loads(initial.read_text())["next_iteration"], 1000)

    def test_failed_acceptance_resets_streak_and_upper_budget_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, max_iterations=3500)
            children = FakeChildren(args, [True, False, True, True])
            result = Supervisor(args, child_runner=children).run()
            self.assertEqual(result["state"], "budget_exhausted")
            self.assertEqual(result["current_iteration"], 3500)
            self.assertEqual([e["consecutive_successes"] for e in result["evaluations"]], [1, 0, 1, 2])
            self.assertEqual([e["iteration"] for e in result["evaluations"]], [1000, 2000, 3000, 3500])
            first_train = next(c for c in children.commands if len(c) > 2 and c[2].endswith("train_stage1"))
            self.assertNotIn("--resume", first_train)

    def test_child_failure_stops_without_evaluation_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            children = FakeChildren(args, [], fail_train=True)
            supervisor = Supervisor(args, child_runner=children)
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                supervisor.run()
            result = json.loads((args.run_dir / "supervisor_status.json").read_text())
            self.assertEqual(result["state"], "failed")
            self.assertEqual(len(children.commands), 1)
            self.assertEqual(result["evaluations"], [])


if __name__ == "__main__":
    unittest.main()
