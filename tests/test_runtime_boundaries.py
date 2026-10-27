"""Check simulator shutdown status and scene-binding refusal without Kit."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from morph_mimic.launcher import close
from morph_mimic.scripts.prepare_motion import prepare_clip


class RecordingStream(io.StringIO):
    def __init__(self, events, name):
        super().__init__()
        self.events, self.name = events, name

    def flush(self):
        self.events.append("flush_" + self.name)
        super().flush()


class FakeApp:
    def __init__(self, events):
        self.events = events
        self.app = self
        self.quit_codes = []

    def post_quit(self, status):
        self.quit_codes.append(status)
        self.events.append("post_quit")

    def close(self):
        self.events.append("close")


class TestShutdownStatus(unittest.TestCase):
    def fixture(self):
        events = []
        return events, FakeApp(events), RecordingStream(events, "stdout"), RecordingStream(events, "stderr")

    def test_success_flushes_logs_before_close_without_requesting_failure(self):
        events, app, stdout, stderr = self.fixture()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print("validation completed")
            close(app)
        self.assertEqual(app.quit_codes, [])
        self.assertEqual(events[-3:], ["flush_stdout", "flush_stderr", "close"])
        self.assertEqual(stdout.getvalue(), "validation completed\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_exception_is_reported_and_preserved_with_failure_status(self):
        events, app, stdout, stderr = self.fixture()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(ValueError, "broken validation"):
                try:
                    raise ValueError("broken validation")
                finally:
                    close(app)
        self.assertEqual(app.quit_codes, [1])
        self.assertIn("ValueError: broken validation", stderr.getvalue())
        self.assertEqual(events[-3:], ["flush_stdout", "flush_stderr", "close"])

    def test_system_exit_matches_python_success_and_error_status_semantics(self):
        for argument, expected in [(None, 0), (0, 0), (7, 7), ("replay failed", 1)]:
            with self.subTest(argument=argument):
                events, app, stdout, stderr = self.fixture()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        try:
                            raise SystemExit(argument)
                        finally:
                            close(app)
                self.assertEqual(caught.exception.code, argument)
                self.assertEqual(app.quit_codes, [expected])
                self.assertEqual(events[-3:], ["flush_stdout", "flush_stderr", "close"])


class TestSceneConversionRefusal(unittest.TestCase):
    def source(self, directory, marker):
        source = Path(directory) / "walk_bound.npz"
        np.savez(source, requires_scene=np.array(marker), fps=np.array([50.]), joint_pos=np.zeros((3, 29)))
        return source

    def check_refusal(self, converter):
        for sidecar in (False, True):
            with self.subTest(sidecar=sidecar), tempfile.TemporaryDirectory() as directory:
                # Either marker independently requires scene-aware handling.
                source = self.source(directory, marker=not sidecar)
                if sidecar:
                    source.with_suffix(".scene.json").write_text('{"schema":1}')
                original = hashlib.sha256(source.read_bytes()).hexdigest()
                output = Path(directory) / "output"
                with self.assertRaisesRegex(ValueError, "scene-bound references require scene-aware conversion"):
                    converter(source, output)
                self.assertFalse(output.exists())
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)

    def test_generic_crop_refuses_either_scene_binding_marker_without_writing(self):
        self.check_refusal(lambda source, output: prepare_clip(source, output, 0., .02))

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is needed before projection input checks")
    def test_generic_projection_refuses_either_scene_binding_marker_without_writing(self):
        from morph_mimic.scripts.project_reference import project_clip
        from morph_mimic.utils.specs import RobotSpec

        spec = RobotSpec.load("g1")
        self.check_refusal(lambda source, output: project_clip(source, output, spec))


class TestEvaluationFastShutdown(unittest.TestCase):
    def test_complete_report_is_published_before_kit_fast_shutdown(self):
        from morph_mimic.scripts import evaluate

        class FakeActor:
            def to(self, device):
                return self

            def eval(self):
                return self

            def load_state_dict(self, state):
                self.loaded_state = state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "final.pt"
            checkpoint_path.write_bytes(b"test checkpoint bytes for provenance")
            output = root / "reports/evaluation.json"
            args = NS(robot="g1", stage=1, motion_dir=root / "motions", clips=None,
                      resume=checkpoint_path, device="cpu", planner="acp", headless=True,
                      seed=13, out=output)
            checkpoint = {"robot": "g1", "stage": 1, "ppo": {"actor": {"fixture": 1}}}
            metadata = [{"name": "complete_clip"}, {"name": "failed_clip"}]
            motions = [
                dict(name="complete_clip", interrupted=False, budget_limited=False,
                     contract_verified=True, completed=True, failure=False, end_phase=1., root_position_rmse_m=.01),
                dict(name="failed_clip", interrupted=False, budget_limited=False,
                     contract_verified=True, completed=False, failure=True, end_phase=.152, root_position_rmse_m=.4),
            ]
            app = object()
            reports_seen_at_close = []

            def fast_shutdown(closing_app):
                self.assertIs(closing_app, app)
                # A real Kit fast close can terminate Python immediately.
                # Read the actual published JSON here, before raising exit.
                self.assertTrue(output.is_file(), "evaluation report must exist before close")
                report = json.loads(output.read_text())
                self.assertEqual(report["motions"], motions)
                self.assertEqual(report["expected_clips"], 2)
                self.assertEqual(report["evaluated_clips"], 2)
                self.assertTrue(report["evaluation_is_full"])
                self.assertTrue(report["contract_verified"])
                self.assertEqual(report["completion_rate"], .5)
                self.assertEqual(report["failures"], 1)
                self.assertEqual(report["checkpoint_sha256"], hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
                self.assertEqual(report["seed"], 13)
                self.assertIn("bound scene", report["scope"])
                reports_seen_at_close.append(report)
                raise SystemExit(0)

            with patch.object(evaluate, "_arguments", return_value=args), \
                 patch.object(evaluate, "_clip_metadata", return_value=metadata), \
                 patch.object(evaluate.torch, "load", return_value=checkpoint), \
                 patch("morph_mimic.networks.tracker.TrackerActor", FakeActor), \
                 patch.object(evaluate, "_evaluate_clip", side_effect=motions) as evaluate_clip, \
                 patch("morph_mimic.launcher.launch", return_value=app), \
                 patch("morph_mimic.launcher.close", side_effect=fast_shutdown), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    evaluate.main()
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(evaluate_clip.call_count, 2)
            self.assertEqual(len(reports_seen_at_close), 1)
            self.assertEqual(json.loads(output.read_text()), reports_seen_at_close[0])


if __name__ == "__main__":
    unittest.main()
