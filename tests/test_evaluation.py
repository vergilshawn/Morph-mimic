"""Metrics distinguish tracking accuracy, physical failure, and time budgets."""

import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

from morph_mimic.utils.evaluation import ReferenceTrackingMetrics, TrackingTrajectory, reference_outcome


class TestReferenceEvaluation(unittest.TestCase):
    def snapshot(self, root=25.0, joint=4.0, body=9.0, orientation=0.25):
        return dict(
            root_squared_error_m2=torch.tensor([root]),
            joint_squared_error_rad2=torch.full((1, 23), joint),
            body_squared_error_m2=torch.full((1, 9), body),
            root_orientation_squared_error_rad2=torch.tensor([orientation]),
        )

    def test_distance_rms_is_not_per_coordinate_rms(self):
        metrics = ReferenceTrackingMetrics()
        # Position offsets [3,4,0] have Euclidean distance 5 m.
        metrics.update(self.snapshot())
        result = metrics.summary()
        self.assertEqual(result["root_position_rmse_m"], 5.0)
        self.assertEqual(result["joint_rmse_rad"], 2.0)
        self.assertEqual(result["tracked_body_position_rmse_m"], 3.0)
        self.assertEqual(result["root_orientation_rmse_rad"], 0.5)

    def test_terminal_sample_is_part_of_rms(self):
        metrics = ReferenceTrackingMetrics()
        metrics.update(self.snapshot(root=0.0))
        metrics.update(self.snapshot(root=100.0))
        self.assertAlmostEqual(metrics.summary()["root_position_rmse_m"], math.sqrt(50))
        self.assertEqual(metrics.summary()["tracking_samples"], 2)
        self.assertEqual(metrics.summary()["root_position_max_error_m"], 10.0)

    def test_rejects_nan_and_negative_squared_errors(self):
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReferenceTrackingMetrics().update(self.snapshot(root=value))

    def test_empty_metrics_cannot_claim_accuracy(self):
        with self.assertRaises(ValueError):
            ReferenceTrackingMetrics().summary()

    def test_final_frame_fall_is_failure(self):
        result = reference_outcome(terminated=True, truncated=True, reached_end=True, phase=1.0)
        self.assertTrue(result["failure"])
        self.assertFalse(result["completed"])
        self.assertFalse(result["budget_limited"])

    def test_short_budget_does_not_complete_motion(self):
        result = reference_outcome(terminated=False, truncated=True, reached_end=False, phase=0.7)
        self.assertTrue(result["budget_limited"])
        self.assertFalse(result["completed"])
        self.assertFalse(result["failure"])

    def test_full_motion_completion(self):
        result = reference_outcome(terminated=False, truncated=True, reached_end=True, phase=1.0)
        self.assertTrue(result["completed"])
        self.assertFalse(result["budget_limited"])

    def test_suffix_success_is_not_full_skill_completion(self):
        result = reference_outcome(terminated=False, truncated=True, reached_end=True,
                                   phase=1.0, start_phase=0.9)
        self.assertTrue(result["segment_completed"])
        self.assertTrue(result["evaluation_is_partial"])
        self.assertFalse(result["completed"])
        self.assertFalse(result["budget_limited"])

    def test_start_phase_must_not_select_empty_or_invalid_segments(self):
        for value in (-0.1, 1.0, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reference_outcome(terminated=False, truncated=True, reached_end=True,
                                  phase=1.0, start_phase=value)

    def test_old_point99_end_does_not_complete(self):
        result = reference_outcome(terminated=False, truncated=True, reached_end=True, phase=0.99)
        self.assertFalse(result["completed"])
        self.assertTrue(result["budget_limited"])

    def test_policy_evaluation_uses_terminal_snapshot_before_auto_reset(self):
        from morph_mimic.scripts.evaluate import _evaluate_clip

        test = self

        class FakeEnv:
            def __init__(self):
                self.clip_id = torch.tensor([0])
                self.phase = torch.tensor([0.0])
                self.cell_offset = torch.ones(1, 3)
                self.contact_link_names = [f"body{i}" for i in range(9)]
                self.controls = 0
                self.closed = False

            def reset(self):
                return {"policy": torch.zeros(1, 1)}, {}

            def step(self, action):
                test.assertEqual(float(action[0, 0]), 0.75)
                self.controls += 1
                done = self.controls == 2
                snapshot = test.snapshot(root=100.0 if done else 0.0)
                snapshot.update(clip_id=self.clip_id.clone(), phase=torch.tensor([1.0 if done else 0.5]),
                                reached_end=torch.tensor([done]))
                # Automatic reset has already returned phase to zero. Metrics
                # must use the endpoint snapshot rather than this current state.
                self.phase.zero_()
                return {"policy": torch.zeros(1, 1)}, torch.ones(1), torch.tensor([False]), torch.tensor([done]), {"final_tracking": snapshot}

            def close(self):
                self.closed = True

        env = FakeEnv()
        args = SimpleNamespace(seed=0, robot="g1", stage=1, device="cpu", motion_dir="/tmp",
                               max_steps=None)
        metadata = dict(name="walk1_subject1", duration_s=0.04, terrain_size=(9.0, 6.0),
                        source_sha256="source", frames=3)
        with patch("morph_mimic.scripts.common.build_env", return_value=(env, None)) as build, \
                patch("morph_mimic.scripts.evaluate.RobotSpec.load", return_value=SimpleNamespace(control_dt=0.02)), \
                patch("morph_mimic.utils.checkpoint.validate_checkpoint_contract", return_value=True):
            result = _evaluate_clip(args, {}, metadata, lambda obs: torch.full((1, 1), 0.75),
                                    None, SimpleNamespace(is_running=lambda: True))
        self.assertTrue(result["completed"])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["end_phase"], 1.0)
        self.assertAlmostEqual(result["root_position_rmse_m"], math.sqrt(50))
        self.assertTrue(env.closed)
        self.assertTrue(build.call_args.kwargs["evaluation"])
        self.assertFalse(build.call_args.kwargs["replay"])

    def test_partial_evaluation_uses_remaining_budget_and_never_claims_full_success(self):
        from morph_mimic.scripts.evaluate import _evaluate_clip

        test = self

        class FakeEnv:
            clip_id = torch.tensor([0])
            phase = torch.tensor([0.9])
            cell_offset = torch.zeros(1, 3)
            contact_link_names = [f"body{i}" for i in range(9)]

            def reset(self):
                return {"policy": torch.zeros(1, 23)}, {}

            def step(self, action):
                snapshot = test.snapshot(root=0.0)
                snapshot.update(clip_id=self.clip_id.clone(), phase=torch.tensor([1.0]),
                                reached_end=torch.tensor([True]))
                return {}, torch.ones(1), torch.tensor([False]), torch.tensor([True]), {"final_tracking": snapshot}

            def close(self):
                pass

        args = SimpleNamespace(seed=0, robot="g1", stage=1, device="cpu", motion_dir="/tmp",
                               start_phase=0.9, max_steps=None, trajectory=None)
        metadata = dict(name="box", duration_s=10.0, terrain_size=(9.0, 6.0), source_sha256="source", frames=501)
        with patch("morph_mimic.scripts.common.build_env", return_value=(FakeEnv(), None)) as build, \
                patch("morph_mimic.scripts.evaluate.RobotSpec.load", return_value=SimpleNamespace(control_dt=0.02)), \
                patch("morph_mimic.utils.checkpoint.validate_checkpoint_contract", return_value=True):
            result = _evaluate_clip(args, {}, metadata, lambda obs: torch.zeros(1, 23),
                                    None, SimpleNamespace(is_running=lambda: True))
        self.assertEqual(result["expected_control_steps"], 50)
        self.assertEqual(result["full_reference_control_steps"], 500)
        self.assertTrue(result["segment_completed"])
        self.assertFalse(result["completed"])
        self.assertEqual(build.call_args.kwargs["evaluation_start_phase"], 0.9)


class TestTrackingTrajectory(unittest.TestCase):
    def snapshot(self):
        snapshot = {name: torch.zeros((1,) + shape) for name, shape in TrackingTrajectory._fields.items()}
        snapshot["phase"][:] = 0.1
        snapshot["computed_torque"][:, 4] = 14.0
        snapshot["applied_torque"][:, 4] = 10.0
        snapshot["root_pos_w"][:, 2] = 0.2
        snapshot["ref_root_pos_w"][:, 2] = 0.8
        snapshot["termination_height_threshold_m"][:] = 0.3
        snapshot["joint_pos"][:] = 0.2
        snapshot["joint_ref_pos"][:] = 0.1
        snapshot["joint_target_pos"][:] = 0.5
        snapshot["ref_contact_flags"][:, 6] = 1.0
        snapshot["terminated"][:] = 1.0
        snapshot["died"][:] = 1.0
        return snapshot

    def test_copies_terminal_physical_values_and_diagnoses_failure(self):
        trajectory = TrackingTrajectory(0.02)
        snapshot = self.snapshot()
        trajectory.update(snapshot, action=torch.zeros(1, 23), reward=torch.ones(1))
        # Simulate the robot and the environment's buffers being reset in place.
        snapshot["root_pos_w"].zero_()
        snapshot["applied_torque"].zero_()
        a = trajectory.arrays()
        self.assertAlmostEqual(float(a["root_pos_w"][0, 2]), 0.2)
        self.assertEqual(float(a["applied_torque"][0, 4]), 10.0)
        summary = trajectory.summary()
        self.assertEqual(summary["failure_reason"], "root_below_height_threshold")
        self.assertEqual(summary["max_estimated_torque_clip_nm"], 4.0)
        self.assertEqual(summary["estimated_actuator_clipped_step_fraction"], 1.0)
        self.assertEqual(summary["contact_mismatch_fraction_per_body"][6], 1.0)
        self.assertAlmostEqual(summary["target_tracking_rmse_per_joint_rad"][0], 0.3)
        self.assertEqual(a["elapsed_control_time_s"].tolist(), [0.02])

    def test_rejects_missing_nonfinite_and_wrong_shape_snapshots(self):
        for problem in ("missing", "nonfinite", "shape"):
            snapshot = self.snapshot()
            if problem == "missing":
                del snapshot["contact_force_w"]
            elif problem == "nonfinite":
                snapshot["root_pos_w"][:, 0] = float("nan")
            else:
                snapshot["joint_pos"] = torch.zeros(1, 29)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                TrackingTrajectory(0.02).update(snapshot, action=torch.zeros(1, 23), reward=torch.ones(1))

    def test_rejects_auto_reset_phase_reversal(self):
        trajectory = TrackingTrajectory(0.02)
        snapshot = self.snapshot()
        trajectory.update(snapshot, action=torch.zeros(1, 23), reward=torch.ones(1))
        snapshot["phase"].zero_()
        with self.assertRaisesRegex(ValueError, "phase decreased"):
            trajectory.update(snapshot, action=torch.zeros(1, 23), reward=torch.ones(1))

    def test_npz_is_pickle_free_and_existing_artifacts_are_preserved(self):
        trajectory = TrackingTrajectory(0.02)
        trajectory.update(self.snapshot(), action=torch.zeros(1, 23), reward=torch.ones(1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            trajectory.write(path, metadata={"clip": "box", "schema": 1})
            original = path.read_bytes()
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(saved["joint_pos"].shape, (1, 23))
                self.assertEqual(saved["contact_force_w"].shape, (1, 9, 3))
                self.assertEqual(json.loads(saved["metadata_json"].item())["clip"], "box")
            with self.assertRaises(FileExistsError):
                trajectory.write(path, metadata={"clip": "other"})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_empty_trajectory_cannot_publish(self):
        trajectory = TrackingTrajectory(0.02)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            trajectory.write(Path(directory) / "empty.npz", metadata={})

    def test_evaluate_publishes_terminal_diagnostics_before_environment_close(self):
        from morph_mimic.scripts.evaluate import _evaluate_clip

        fixture = self.snapshot()
        fixture.update(
            root_squared_error_m2=torch.tensor([0.36]),
            joint_squared_error_rad2=torch.full((1, 23), 0.01),
            body_squared_error_m2=torch.zeros(1, 9),
            root_orientation_squared_error_rad2=torch.zeros(1),
        )
        test = self

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "final.pt"
            checkpoint.write_bytes(b"fixture-checkpoint")
            trajectory_path = root / "terminal.npz"

            class FakeEnv:
                clip_id = torch.tensor([0])
                phase = torch.tensor([0.0])
                cell_offset = torch.zeros(1, 3)
                contact_link_names = [f"body{i}" for i in range(9)]

                def reset(self):
                    return {"policy": torch.zeros(1, 23)}, {}

                def step(self, action):
                    return {}, torch.ones(1), torch.tensor([True]), torch.tensor([False]), {"final_tracking": fixture}

                def close(self):
                    test.assertTrue(trajectory_path.is_file(), "trajectory must be published before closing Isaac")

            args = SimpleNamespace(seed=0, robot="g1", stage=1, device="cpu", motion_dir=root,
                                   resume=checkpoint, start_phase=0.0, max_steps=None, trajectory=trajectory_path)
            metadata = dict(name="box", duration_s=10.0, terrain_size=(9.0, 6.0), source_sha256="source", frames=501)
            spec = SimpleNamespace(control_dt=0.02, controlled_joint_names=[f"joint{i}" for i in range(23)])
            with patch("morph_mimic.scripts.common.build_env", return_value=(FakeEnv(), None)) as build, \
                    patch("morph_mimic.scripts.evaluate.RobotSpec.load", return_value=spec), \
                    patch("morph_mimic.utils.checkpoint.validate_checkpoint_contract", return_value=True):
                result = _evaluate_clip(args, {}, metadata, lambda obs: torch.zeros(1, 23),
                                        None, SimpleNamespace(is_running=lambda: True))
            self.assertTrue(build.call_args.kwargs["evaluation_record_trajectory"])
            self.assertTrue(result["failure"])
            self.assertEqual(result["physical_diagnostics"]["failure_reason"], "root_below_height_threshold")
            self.assertEqual(len(result["trajectory_sha256"]), 64)
            with np.load(trajectory_path, allow_pickle=False) as saved:
                meta = json.loads(saved["metadata_json"].item())
                self.assertTrue(meta["result"]["contract_verified"])
                self.assertEqual(meta["contact_body_names"], FakeEnv.contact_link_names)
                self.assertEqual(meta["joint_names"], spec.controlled_joint_names)
                self.assertAlmostEqual(float(saved["root_pos_w"][0, 2]), 0.2)


if __name__ == "__main__":
    unittest.main()
