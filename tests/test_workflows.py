"""Public check/diagnostic workflows, training status and isolated policy probes."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from morph_mimic.diagnostics.diagnose_policy_segments import (
    _arguments, _continue_past_root_error, _diagnostic_result, _run_isolated, _trajectory_type,
)
from morph_mimic.scripts import check, inspect
from morph_mimic.scripts.watch_training import TERMINAL_STATES, render, snapshot
from morph_mimic.utils.evaluation import TrackingTrajectory


# Public check ordering, isolated commands and strict saved-policy acceptance.

class TestCheckEntrypoint(unittest.TestCase):
    def arguments(self, *values):
        return check.parser().parse_args(values)

    def test_default_is_cpu_only_and_selected_stages_keep_dependency_order(self):
        self.assertEqual(check.stage_order(self.arguments()), ["regression", "assets", "reference"])
        self.assertEqual(check.stage_order(self.arguments("--only", "reference", "assets", "--gpu")),
                         ["assets", "reference", "gpu"])
        self.assertEqual(check.stage_order(self.arguments("--gpu-only")), ["gpu"])
        self.assertEqual(self.arguments().suite, ["basic"])

    def test_subprocess_uses_exact_interpreter_and_keeps_argument_boundaries(self):
        args = self.arguments("--pytest-args=-k 'contact or scene' -v", "--clip", "clip with spaces")
        command = check.child_command("regression", args, Path("result.json"))
        self.assertEqual(command[:6], [sys.executable, "-m", "pytest", "tests", "-q", "-rs"])
        self.assertEqual(command[6:], ["--suite", "basic", "-k", "contact or scene", "-v"])
        command = check.child_command("reference", args, Path("result.json"))
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[command.index("--clip") + 1], "clip with spaces")

    def test_optional_suites_are_forwarded_as_separate_arguments(self):
        args = self.arguments("--suite", "reference", "planner")
        command = check.child_command("regression", args, Path("result.json"))
        self.assertEqual(command[6:], ["--suite", "reference", "--suite", "planner"])

    def test_suite_collection_selects_essential_control_and_optional_cases_in_order(self):
        def collect(*suites):
            command = [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"]
            for suite in suites:
                command += ["--suite", suite]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                    capture_output=True, text=True, check=True)
            return [line for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line]
        basic = collect("basic")
        combined = collect("basic", "planner")
        all_cases = collect("all")
        self.assertTrue(basic)
        self.assertEqual(len(basic), len(set(basic)))
        self.assertTrue(set(basic) < set(combined) < set(all_cases))
        self.assertTrue(any("ContactTeacherContractTests" in node for node in basic))
        self.assertTrue(any("TestSceneIntegration" in node for node in basic))
        self.assertTrue(any("TestWorldTrackingTeacher" in node for node in basic))
        self.assertFalse(any("TestStandingSupervisor" in node or "TestPlannerAttention" in node for node in basic))
        self.assertFalse(any("TestExplicitMjcf" in node or "TestReferenceProjection" in node for node in basic))
        self.assertTrue(any("TestPlannerAttention" in node for node in combined))
        self.assertTrue(any("TestExplicitMjcf" in node for node in all_cases))
        expected = ["foundation", "reference", "contacts", "control", "training", "evaluation", "workflows"]
        groups = [expected.index(node.split("::")[0].removeprefix("tests/test_").removesuffix(".py")) for node in basic]
        self.assertEqual(groups, sorted(groups))

    def test_first_failure_stops_later_checks_and_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(check, "ROOT", Path(directory)):
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                kwargs["stdout"].write("meaningful failure\n")
                return subprocess.CompletedProcess(command, 7)
            status = check.run(self.arguments(), execute=execute)
            self.assertEqual(status, 1)
            self.assertEqual(len(calls), 1)
            latest = json.loads((Path(directory) / "runs/_reports/checks/latest.json").read_text())
            self.assertEqual(latest["state"], "failed")
            self.assertEqual(latest["stages"][0]["returncode"], 7)
            self.assertEqual(latest["selected"], ["regression", "assets", "reference"])

    def test_nonregression_exit_zero_requires_a_structured_success_result(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(check, "ROOT", Path(directory)):
            def execute(command, **kwargs):
                return subprocess.CompletedProcess(command, 0)
            self.assertEqual(check.run(self.arguments("--only", "assets"), execute=execute), 1)

    def test_selected_checks_execute_sequentially_and_keep_each_result(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(check, "ROOT", Path(directory)):
            names = []
            def execute(command, **kwargs):
                name = command[command.index("--_stage") + 1]
                names.append(name)
                Path(command[command.index("--_result") + 1]).write_text(json.dumps(dict(ok=True, marker=name)))
                return subprocess.CompletedProcess(command, 0)
            self.assertEqual(check.run(self.arguments("--only", "reference", "assets"), execute=execute), 0)
            self.assertEqual(names, ["assets", "reference"])
            latest = json.loads((Path(directory) / "runs/_reports/checks/latest.json").read_text())
            self.assertEqual([stage["result"]["marker"] for stage in latest["stages"]], names)

    def test_explicit_report_cannot_overwrite_a_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "old.json"
            existing.write_text("old")
            with self.assertRaises(FileExistsError):
                check.run(self.arguments("--out", str(existing)))
            self.assertEqual(existing.read_text(), "old")

    def test_gpu_checkpoint_kind_dispatch_has_no_training_command(self):
        for kind, module in [("teacher-v2", "train_reference_teacher_v2"),
                             ("teacher-v3", "train_reference_teacher_v3"),
                             ("acp-student", "train_acp_student"),
                             ("standing", "train_standing_baseline")]:
            with self.subTest(kind=kind):
                args = self.arguments("--checkpoint", "actor.pt", "--clip", "climb", "--checkpoint-kind", kind)
                command = check.checkpoint_command(args, Path("reports"))
                self.assertEqual(command[:4], [sys.executable, "-m", f"morph_mimic.scripts.{module}", "evaluate"])
                self.assertNotIn("train", command)
                self.assertIn("reports/policy_trajectory.npz", command)
                self.assertEqual(command[command.index("--clips")+1], "climb")

    def test_teacher_v3_uses_explicit_teacher_skill_acceptance(self):
        args = self.arguments("--checkpoint-kind", "teacher-v3", "--clip", "climb")
        report = dict(contract_verified=True, evaluation_is_full=True, evaluation_is_partial=False,
                      start_phase=0., expected_clips=1, evaluated_clips=1, completion_rate=1.,
                      teacher_passed=True, teacher_failure_reasons=[],
                      motions=[dict(clip="climb", contract_verified=True, start_phase=0., evaluation_is_partial=False,
                                    budget_limited=False, interrupted=False, completed=True,
                                    failure=False, terminated=False)])
        self.assertTrue(check.policy_evaluation_result(args, report)["policy_passed"])
        for key, value in [("teacher_passed", None), ("teacher_failure_reasons", None),
                           ("teacher_failure_reasons", ["physical_box_top_hold"]),
                           ("evaluation_is_full", False), ("start_phase", .34)]:
            with self.subTest(key=key):
                bad = deepcopy(report)
                bad[key] = value
                with self.assertRaises(ValueError):
                    check.policy_evaluation_result(args, bad)

    def test_checkpoint_requires_explicit_gpu_opt_in_before_any_import(self):
        with self.assertRaises(SystemExit) as caught:
            check.main(["--checkpoint", "missing.pt"])
        self.assertEqual(caught.exception.code, 2)

    def test_saved_policy_distinguishes_completed_evaluation_from_failed_skill(self):
        args = self.arguments("--checkpoint-kind", "standing", "--clip", "climb")
        report = dict(contract_verified=True, evaluation_is_full=True, evaluation_is_partial=False,
                      start_phase=0., expected_clips=1, evaluated_clips=1, completion_rate=0.,
                      standing_passed=False, standing_failure_reasons=["completed"],
                      motions=[dict(clip="climb", contract_verified=True, start_phase=0., evaluation_is_partial=False,
                                    budget_limited=False, interrupted=False, completed=False,
                                    failure=True, terminated=True)])
        result = check.policy_evaluation_result(args, report)
        self.assertTrue(result["evaluation_completed"])
        self.assertFalse(result["policy_passed"])
        for target, key, value in [("report", "evaluation_is_full", False),
                                   ("report", "evaluated_clips", 0),
                                   ("motion", "start_phase", .34),
                                   ("motion", "contract_verified", False),
                                   ("motion", "clip", "another_clip"),
                                   ("motion", "budget_limited", True),
                                   ("motion", "interrupted", True)]:
            with self.subTest(target=target, key=key):
                bad = deepcopy(report)
                (bad if target == "report" else bad["motions"][0])[key] = value
                with self.assertRaises(ValueError):
                    check.policy_evaluation_result(args, bad)

    def test_saved_policy_rejects_contradictory_success_flags(self):
        args = self.arguments("--checkpoint-kind", "standing", "--clip", "climb")
        report = dict(contract_verified=True, evaluation_is_full=True, evaluation_is_partial=False,
                      start_phase=0., expected_clips=1, evaluated_clips=1,
                      standing_passed=True, standing_failure_reasons=["full_ten_seconds"],
                      motions=[dict(clip="climb", contract_verified=True, start_phase=0., evaluation_is_partial=False,
                                    budget_limited=False, interrupted=False, completed=True,
                                    failure=False, terminated=False)])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            check.policy_evaluation_result(args, report)

    def test_failed_skill_exits_one_and_preserves_completed_evaluation_details(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, result = Path(directory) / "actor.pt", Path(directory) / "result.json"
            checkpoint.touch()
            details = dict(mode="saved_policy_evaluation", evaluation_completed=True,
                           policy_passed=False, policy_failure_reasons=["completed"],
                           report="policy_evaluation.json")
            with patch.object(check, "_gpu_check", return_value=details):
                status = check.main(["--_stage", "gpu", "--_result", str(result),
                                     "--checkpoint", str(checkpoint), "--clip", "climb",
                                     "--checkpoint-kind", "teacher-v3"])
            self.assertEqual(status, 1)
            saved = json.loads(result.read_text())
            self.assertFalse(saved["ok"])
            self.assertEqual(saved["details"], details)

    def test_checkpoint_without_one_clip_fails_before_simulator_import(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor.pt"
            checkpoint.touch()
            for kind in ("acp", "standing", "teacher-v2", "teacher-v3"):
                with self.subTest(kind=kind), self.assertRaises(SystemExit) as caught:
                    check.main(["--gpu-only", "--checkpoint", str(checkpoint), "--checkpoint-kind", kind])
                self.assertEqual(caught.exception.code, 2)

    def test_teacher_v3_rejects_other_robot_or_stage_before_simulator_import(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor.pt"
            checkpoint.touch()
            for values in [("--robot", "mojia"), ("--stage", "2")]:
                with self.subTest(values=values), self.assertRaises(SystemExit) as caught:
                    check.main(["--gpu-only", "--checkpoint", str(checkpoint), "--clip", "climb",
                                "--checkpoint-kind", "teacher-v3", *values])
                self.assertEqual(caught.exception.code, 2)


# Diagnostic dispatch and isolated interpreter boundaries.

def test_top_level_help_needs_no_simulator_or_diagnostic_import():
    probe = subprocess.run(
        [sys.executable, "-c", "import sys; from morph_mimic.scripts import inspect; "
         "assert not any(name.startswith(('torch', 'isaaclab', 'omni', 'morph_mimic.diagnostics')) "
         "for name in sys.modules); inspect.main(['--help'])"], capture_output=True, text=True,
    )
    assert probe.returncode == 0
    assert "No GPU job runs unless" in probe.stdout
    assert "Commands:" in probe.stdout
    assert "Starting SimulationApp" not in probe.stdout


def test_arguments_help_and_failure_are_forwarded_to_isolated_interpreter():
    with patch.object(inspect.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run:
        status = inspect.main(["frames", "--trajectory", "a path/trace.npz", "--help"])
    assert status == 7
    run.assert_called_once_with(
        [sys.executable, "-m", "morph_mimic.diagnostics.audit_tracking_frames",
         "--trajectory", "a path/trace.npz", "--help"], cwd=inspect.PROJECT_ROOT,
    )


def test_separator_and_frozen_evaluation_entry_point_remain_supported():
    with patch.object(inspect.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
        assert inspect.main(["evaluate", "--", "--robot", "g1", "--stage", "1"]) == 0
    assert run.call_args.args[0][2:] == ["morph_mimic.scripts.evaluate", "--robot", "g1", "--stage", "1"]


def test_diagnostic_implementations_are_internal_and_importable():
    # The old scattered script files should not remain as duplicate entry points.
    assert all(not (inspect.PROJECT_ROOT / "morph_mimic/scripts" / f"{module.rsplit('.', 1)[1]}.py").exists()
               for module, _ in inspect.COMMANDS.values() if module.startswith("morph_mimic.diagnostics."))
    assert importlib.util.find_spec("morph_mimic.diagnostics.diagnose_policy_segments") is not None


# Training status, rotating logs and explicit terminal states.

class TestTrainingWatcher(unittest.TestCase):
    def test_follows_new_child_and_distinguishes_live_iteration_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training.log").write_text("iter    990 | initial\n")
            status = dict(state="running", phase="waiting_for_existing_training", children=[],
                          current_iteration=None, maximum_iterations=20000,
                          chunk_iterations=1000, required_successes=3, evaluations=[])
            path = root / "supervisor_status.json"
            path.write_text(json.dumps(status))
            self.assertEqual(snapshot(root)["visible_iteration"], 990)
            child = root / "child2000.log"
            child.write_text("Isaac startup\niter   1040 | resumed\n")
            status.update(phase="train", current_iteration=1000, target_iteration=2000,
                          children=[dict(phase="train", log=str(child))])
            replacement = root / "replacement.json"
            replacement.write_text(json.dumps(status))
            replacement.replace(path)
            item = snapshot(root)
            self.assertEqual(item["completed_checkpoint_iteration"], 1000)
            self.assertEqual(item["visible_iteration"], 1040)
            self.assertEqual(item["active_log"], str(child))
            evaluation_log = root / "evaluate.log"
            evaluation_log.write_text("evaluating\n")
            status.update(phase="evaluate", current_iteration=2000)
            status["children"].append(dict(phase="evaluate", log=str(evaluation_log)))
            path.write_text(json.dumps(status))
            item = snapshot(root)
            self.assertEqual(item["visible_iteration"], 2000)
            self.assertEqual(item["active_log"], str(evaluation_log))
            self.assertEqual(item["last_training_line"], "iter   1040 | resumed")

    def test_stopping_states_explain_success_budget_and_failure_separately(self):
        base = dict(phase="finished", visible_iteration=20000, maximum_iterations=20000,
                    target_iteration=20000, consecutive_successes=0, required_successes=3,
                    evaluation_count=20, last_evaluation=None, last_training_line=None,
                    error="child exited 7", active_log="child.log")
        for state in ("validated", "budget_exhausted", "failed", "superseded"):
            with self.subTest(state=state):
                text = "\n".join(render(dict(base, state=state)))
                self.assertIn(f"STOP {state}", text)
        self.assertIn("验收尚未通过", "\n".join(render(dict(base, state="budget_exhausted"))))
        self.assertIn("child exited 7", "\n".join(render(dict(base, state="failed"))))

    def test_superseded_is_a_normal_stop_and_points_to_replacement(self):
        item = dict(state="superseded", phase="stopped", visible_iteration=7000,
                    maximum_iterations=20000, target_iteration=8000,
                    consecutive_successes=0, required_successes=3, evaluation_count=1,
                    last_evaluation=dict(iteration=7000, passed=False, consecutive_successes=0, reasons=[]),
                    last_training_line=None, active_log="old.log",
                    stop_reason="改为修正后的固定关节基线", superseded_by="runs/new_baseline")
        text = "\n".join(render(item))
        self.assertIn("superseded", TERMINAL_STATES)
        self.assertIn("人为停止", text)
        self.assertIn("改为修正后的固定关节基线", text)
        self.assertIn("runs/new_baseline", text)
        self.assertNotIn("继续训练", text)

    def test_contact_coverage_and_nominal_profile_are_visible_without_becoming_gates(self):
        item = dict(state="running", phase="contact_metrics", visible_iteration=1000,
                    maximum_iterations=20000, target_iteration=1000,
                    consecutive_successes=0, required_successes=3, evaluation_count=1,
                    last_evaluation=dict(iteration=1000, passed=False, consecutive_successes=0,
                                         reasons=[], contact_report="contacts.json",
                                         contact_coverage=dict(reference_duration_coverage_fraction=.076)),
                    training_profile="nominal_pretrain", last_training_line=None, active_log="contacts.log")
        text = "\n".join(render(item))
        self.assertIn("7.6%", text)
        self.assertIn("接触 F1 暂不参与验收", text)
        self.assertIn("尚非完整 ACP", text)


# Diagnostic continuation and original failure evidence.

def _snapshot(*, root_error=.0, height=.8, grace=False, phase=.1, end=False):
    snapshot = {key: torch.zeros((1, *shape)) for key, shape in TrackingTrajectory._fields.items()}
    snapshot.update(root_pos_w=torch.tensor([[root_error, 0., height]]),
                    ref_root_pos_w=torch.tensor([[0., 0., height]]),
                    phase=torch.tensor([phase]),
                    reference_error_exceeded=torch.tensor([root_error > .5]),
                    terminated=torch.tensor([(root_error > .5 or height < .3) and not grace]),
                    died=torch.tensor([(root_error > .5 or height < .3) and not grace]),
                    grace_active=torch.tensor([grace]),
                    reached_end=torch.tensor([end]), truncated=torch.tensor([end]),
                    time_out=torch.tensor([False]), clip_id=torch.tensor([0]),
                    termination_height_threshold_m=torch.tensor([.3]),
                    termination_root_error_threshold_m=torch.tensor([.5]),
                    all_joint_pos=torch.zeros(1, 29), held_joint_position_error_rad=torch.zeros(1, 6))
    return snapshot


class FakeEnv:
    def __init__(self):
        self.cfg = SimpleNamespace(max_reference_root_error_m=.5)
        self.is_ref_env = torch.tensor([True])
        self.extras = {}

    def _get_dones(self):
        snapshot = self.extras["final_tracking"]
        return snapshot["terminated"].clone(), snapshot["truncated"].clone()


def test_continuation_suppresses_only_root_failure_and_preserves_original_decision():
    env = FakeEnv()
    _continue_past_root_error(env)
    for error, height, grace, end, expect in [(.6, .8, False, False, False),
                                             (.6, .2, False, False, True),
                                             (.6, .2, True, False, False),
                                             (.6, .8, False, True, False)]:
        original = _snapshot(root_error=error, height=height, grace=grace, end=end)
        original_term = original["terminated"].clone()
        env.extras["final_tracking"] = original
        terminated, truncated = env._get_dones()
        assert bool(terminated[0]) is expect
        assert bool(truncated[0]) is end
        assert torch.equal(original["strict_terminated"], original_term)
        assert bool(original["strict_reference_error_exceeded"][0])
        assert not bool(original["root_error_termination_enabled"][0])
        assert env.cfg.max_reference_root_error_m == .5


def test_continued_reference_end_never_becomes_accepted_completion():
    result = dict(completed=True, segment_completed=True, failure=False, end_phase=1.)
    _diagnostic_result(result, continuation=True, first_original_termination={"step": 2})
    assert result["completed"] is False
    assert result["segment_completed"] is False
    assert result["observed_reference_end"] is True
    assert result["diagnostic_observed_outcome"]["completed"] is True
    assert result["diagnostic_only"] is True
    assert result["eligible_for_skill_acceptance"] is False


def test_trajectory_records_first_strict_failure_and_diagnostic_metadata(tmp_path):
    trajectory = _trajectory_type(True)(.02)
    env = FakeEnv()
    _continue_past_root_error(env)
    for error, height, phase in [(.1, .8, .002), (.6, .8, .004), (.8, .2, .006)]:
        env.extras["final_tracking"] = _snapshot(root_error=error, height=height, phase=phase)
        env._get_dones()
        trajectory.update(env.extras["final_tracking"], action=torch.zeros(1, 23), reward=torch.ones(1))
    first = trajectory.first_original_termination
    assert first["step"] == 2
    assert first["elapsed_control_time_s"] == .04
    assert first["root_error_exceeded"] is True
    assert first["height_below_threshold"] is False
    assert trajectory.summary()["failure_reason"] == "root_below_height_threshold"
    result = dict(completed=False, segment_completed=False, failure=True, end_phase=.006)
    path = tmp_path / "continued.npz"
    trajectory.write(path, metadata=dict(checkpoint_sha256="checkpoint", source_sha256="motion", result=result))
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        assert metadata["diagnostic_only"] is True
        assert metadata["eligible_for_skill_acceptance"] is False
        assert metadata["first_original_termination"]["step"] == 2
        assert metadata["result"]["eligible_for_skill_acceptance"] is False
        assert archive["all_joint_pos"].shape == (3, 29)
        assert archive["strict_terminated"].tolist() == [False, True, True]
        assert archive["terminated"].tolist() == [False, False, True]


def test_strict_trajectory_marks_suffix_as_diagnostic_without_relaxing_physics(tmp_path):
    trajectory = _trajectory_type(False)(.02)
    trajectory.update(_snapshot(root_error=.6), action=torch.zeros(1, 23), reward=torch.ones(1))
    result = dict(completed=False, segment_completed=False, failure=True)
    trajectory.write(tmp_path / "strict.npz", metadata=dict(result=result))
    assert result["root_error_termination_enabled"] is True
    assert result["first_original_termination"]["step"] == 1
    assert result["eligible_for_skill_acceptance"] is False
    assert "diagnostic_observed_outcome" not in result


def test_coordinator_module_does_not_import_cuda_or_array_libraries():
    result = subprocess.run([sys.executable, "-c",
                             "import sys; import morph_mimic.diagnostics.diagnose_policy_segments; "
                             "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def _coordinator_args(tmp_path):
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    motion = tmp_path / "clip.npz"
    motion.write_bytes(b"motion")
    return _arguments(["--resume", str(checkpoint), "--motion-dir", str(tmp_path),
                       "--clip", "clip", "--phases", "0", ".4", "--continue-phase-zero",
                       "--out-dir", str(tmp_path / "diagnosis")])


def test_each_segment_uses_fresh_process_and_aggregates_only_published_artifacts(tmp_path):
    args = _coordinator_args(tmp_path)
    commands = []

    class FakeChild:
        returncode = 0

        def __init__(self, command, **kwargs):
            import hashlib
            self.pid = 100 + len(commands)
            commands.append(command)
            self.label = command[command.index("--internal-job-label") + 1]
            report = dict(checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                          result=dict(interrupted=False))
            (args.out_dir / f"{self.label}.json").write_text(json.dumps(report))
            (args.out_dir / f"{self.label}.npz").write_bytes(b"published trajectory")

        def wait(self, timeout):
            return self.returncode

    with patch("morph_mimic.diagnostics.diagnose_policy_segments.subprocess.Popen", FakeChild), \
            patch("morph_mimic.diagnostics.diagnose_policy_segments._source_hashes", return_value={"source": "hash"}):
        _run_isolated(args)
    assert len(commands) == 3
    assert all("--internal-single-job" in command for command in commands)
    assert [command[command.index("--phases") + 1] for command in commands] == ["0.0", "0.4", "0.0"]
    assert "--internal-continuation" not in commands[0]
    assert "--internal-continuation" in commands[-1]
    summary = json.loads((args.out_dir / "summary.json").read_text())
    assert summary["state"] == "completed"
    assert len(summary["evaluations"]) == 3
    assert len({process["pid"] for process in summary["processes"]}) == 3
    assert summary["eligible_for_skill_acceptance"] is False


def test_child_failure_preserves_explicit_incomplete_summary(tmp_path):
    args = _coordinator_args(tmp_path)

    class FailedChild:
        pid = 999
        returncode = 2

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout):
            return self.returncode

    with patch("morph_mimic.diagnostics.diagnose_policy_segments.subprocess.Popen", FailedChild), \
            patch("morph_mimic.diagnostics.diagnose_policy_segments._source_hashes", return_value={"source": "hash"}):
        try:
            _run_isolated(args)
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed simulator must not be reported as a completed sweep")
    summary = json.loads((args.out_dir / "interrupted_summary.json").read_text())
    assert summary["state"] == "incomplete"
    assert summary["evaluations"] == []
    assert summary["processes"][0]["returncode"] == 2
    assert not (args.out_dir / "summary.json").exists()


if __name__ == "__main__":
    unittest.main()
