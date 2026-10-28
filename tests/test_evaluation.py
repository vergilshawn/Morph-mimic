"""Tracking, coordinate invariants, contact coverage and strict box task evaluation."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from morph_mimic.diagnostics.audit_tracking_frames import (
    SourceFK, finite_difference, frame_errors, multiply, named_columns,
    orientation_error, phase_coverage, rotate, yaw_quaternion,
)
from morph_mimic.diagnostics.evaluate_box_task import (
    evaluate_box_task_trace, source_foot_kinematics, summarize_box_hold,
)
from morph_mimic.scripts.evaluate_contact_trajectory import contact_report, confusion, _publish
from morph_mimic.utils.box_task import (
    CONTACT_BODY_NAMES, FOOT_BODY_NAMES, PROJECT_ROOT, _rectangle_top, load_box_task_spec, task_metrics,
)
from morph_mimic.utils.evaluation import ReferenceTrackingMetrics, TrackingTrajectory, reference_outcome


# Tracking accuracy, trajectory recording and full-motion completion.

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
                robot = SimpleNamespace(data=SimpleNamespace(joint_names=[f"joint{i}" for i in range(23)]))
                held_idx = torch.empty(0, dtype=torch.long)

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


# Contact report coverage and censored observation intervals.

class TestContactTrajectoryEvaluation(unittest.TestCase):
    def fixture(self, *, frames=11, fps=10.0, steps=10, start_phase=0.0, reference=None, actual=None):
        names = ["hand", "foot"]
        reference = np.zeros((frames, 2), dtype=bool) if reference is None else reference
        actual = np.zeros((steps, 2), dtype=bool) if actual is None else actual
        duration = (frames - 1) / fps
        elapsed = (np.arange(steps) + 1) / fps
        phase = (start_phase + elapsed / duration).astype(np.float32)
        indices = np.clip((phase * (frames - 1)).astype(np.int64), 0, frames - 1)
        trajectory = dict(phase=phase, measured_contact_flags=actual,
                          ref_contact_flags=reference[indices], elapsed_control_time_s=elapsed,
                          contact_force_w=np.zeros((steps, 2, 3)), initial_phase=np.asarray(start_phase))
        metadata = dict(contact_body_names=names, control_dt_s=1 / fps,
                        result=dict(start_phase=start_phase, completed=False))
        labels = dict(contact_flags=reference, body_names=np.asarray(names), fps=np.asarray(fps),
                      source_num_frames=np.asarray(frames), source_init_phase=np.asarray(0.0),
                      source_end_phase=np.asarray(1.0), method=np.asarray("reference_geometry_test"),
                      physics_validated=np.asarray(False), forces_used=np.asarray(False))
        return trajectory, metadata, labels, dict(frames=frames, fps=fps)

    def test_all_negative_has_no_invented_perfect_score(self):
        metrics = confusion(np.zeros(5, dtype=bool), np.zeros(5, dtype=bool))
        self.assertEqual(metrics["tn"], 5)
        for field in ("precision", "recall", "f1"):
            self.assertIsNone(metrics[field])

    def test_missed_positive_is_zero_recall_not_undefined(self):
        metrics = confusion([True, True], [False, False])
        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)
        self.assertEqual(metrics["fn"], 2)

    def test_early_failure_exposes_future_events_without_counting_them_negative(self):
        reference = np.zeros((11, 2), dtype=bool)
        reference[5:7, 0] = True
        args = self.fixture(steps=2, reference=reference)
        report = contact_report(*args[:3], **args[3])
        hand = report["bodies"]["hand"]
        self.assertEqual(hand["force_history_vs_reference"]["tn"], 2)
        self.assertEqual(hand["force_history_vs_reference"]["fn"], 0)
        self.assertIsNone(hand["force_history_vs_reference"]["recall"])
        self.assertEqual(hand["reference_positive_sample_coverage_fraction"], 0)
        self.assertEqual(hand["events"]["summary"]["reference_events_not_reached"], 1)
        self.assertEqual(hand["events"]["reference_events"][0]["detection_status"], "not_evaluated")
        self.assertAlmostEqual(report["coverage"]["reference_duration_coverage_fraction"], .2)

    def test_timing_offsets_only_for_observed_transitions(self):
        reference = np.zeros((11, 2), dtype=bool)
        reference[3:6, 0] = True
        actual = np.zeros((10, 2), dtype=bool)
        actual[3:6, 0] = True  # observed contact at .4,.5,.6; release at .7
        args = self.fixture(reference=reference, actual=actual)
        event = contact_report(*args[:3], **args[3])["bodies"]["hand"]["events"]["reference_events"][0]
        self.assertEqual(event["detection_status"], "matched")
        self.assertAlmostEqual(event["onset_error_s"], .1, places=6)
        self.assertAlmostEqual(event["end_error_s"], .1, places=6)

    def test_censored_transitions_remain_undefined(self):
        reference = np.ones((11, 2), dtype=bool)
        actual = np.ones((10, 2), dtype=bool)
        args = self.fixture(reference=reference, actual=actual)
        event = contact_report(*args[:3], **args[3])["bodies"]["hand"]["events"]["reference_events"][0]
        self.assertEqual(event["detection_status"], "matched")
        self.assertIsNone(event["onset_error_s"])
        self.assertIsNone(event["end_error_s"])

    def test_partial_suffix_separates_prior_events_and_full_skill_completion(self):
        reference = np.zeros((11, 2), dtype=bool)
        reference[1:3, 0] = True
        reference[8:10, 0] = True
        actual = np.ones((5, 2), dtype=bool)
        args = self.fixture(reference=reference, actual=actual, steps=5, start_phase=.5)
        report = contact_report(*args[:3], **args[3])
        self.assertTrue(report["coverage"]["reached_reference_end"])
        self.assertFalse(report["coverage"]["task_completed"])
        self.assertTrue(report["coverage"]["evaluation_is_partial"])
        self.assertEqual(report["bodies"]["hand"]["events"]["summary"]["reference_events_before_partial_reset"], 1)
        self.assertAlmostEqual(report["coverage"]["reference_duration_coverage_fraction"], .5)

    def test_body_order_is_resolved_by_names(self):
        reference = np.zeros((11, 2), dtype=bool)
        reference[:, 0] = True
        args = self.fixture(reference=reference)
        first = contact_report(*args[:3], **args[3])
        labels = copy.deepcopy(args[2])
        labels["body_names"] = labels["body_names"][::-1]
        labels["contact_flags"] = labels["contact_flags"][:, ::-1]
        reordered = contact_report(args[0], args[1], labels, **args[3])
        self.assertEqual(first, reordered)

    def test_reference_mismatch_is_rejected(self):
        args = self.fixture()
        args[0]["ref_contact_flags"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "do not match labels"):
            contact_report(*args[:3], **args[3])

    def test_missing_control_endpoint_is_rejected(self):
        args = self.fixture()
        args[0]["elapsed_control_time_s"][1:] += .1
        with self.assertRaisesRegex(ValueError, "contiguous control"):
            contact_report(*args[:3], **args[3])

    def test_nonfinite_force_and_nonbinary_flags_are_rejected(self):
        args = self.fixture()
        args[0]["contact_force_w"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite endpoint force"):
            contact_report(*args[:3], **args[3])
        args = self.fixture()
        args[0]["measured_contact_flags"] = np.full((10, 2), .5)
        with self.assertRaisesRegex(ValueError, "finite binary"):
            contact_report(*args[:3], **args[3])

    def test_source_label_phase_and_sampling_are_checked(self):
        args = self.fixture()
        args[2]["source_end_phase"] = np.asarray(.8)
        with self.assertRaisesRegex(ValueError, "complete motion"):
            contact_report(*args[:3], **args[3])

    def test_float32_floor_alignment_matches_runtime(self):
        reference = np.zeros((501, 2), dtype=bool)
        reference[250, 0] = True
        args = self.fixture(frames=501, fps=50, steps=1, start_phase=.5, reference=reference)
        self.assertEqual(int(args[0]["phase"][0] * np.float32(500)), 250)
        report = contact_report(*args[:3], **args[3])
        self.assertEqual(report["bodies"]["hand"]["force_history_vs_reference"]["fn"], 1)

    def test_final_frame_only_contact_can_be_detected(self):
        reference = np.zeros((11, 2), dtype=bool)
        reference[-1, 0] = True
        actual = np.zeros((10, 2), dtype=bool)
        actual[-1, 0] = True
        args = self.fixture(reference=reference, actual=actual)
        report = contact_report(*args[:3], **args[3])
        event = report["bodies"]["hand"]["events"]["reference_events"][0]
        self.assertEqual(event["detection_status"], "matched")
        self.assertIsNone(event["end_error_s"])

    def test_endpoint_force_is_separate_from_recorded_history(self):
        args = self.fixture(actual=np.ones((10, 2), dtype=bool))
        report = contact_report(*args[:3], **args[3])
        hand = report["bodies"]["hand"]
        self.assertEqual(hand["force_history_vs_reference"]["fp"], 10)
        self.assertEqual(hand["endpoint_abs_fz_vs_reference"]["fp"], 0)
        self.assertEqual(hand["history_endpoint_disagreement_samples"], 10)

    def test_continuous_contact_covering_two_reference_events_is_not_called_absent(self):
        reference = np.zeros((11, 2), dtype=bool)
        reference[2:4, 0] = True
        reference[6:8, 0] = True
        actual = np.zeros((10, 2), dtype=bool)
        actual[1:7, 0] = True
        args = self.fixture(reference=reference, actual=actual)
        events = contact_report(*args[:3], **args[3])["bodies"]["hand"]["events"]
        self.assertEqual(events["summary"]["matched_reference_events"], 1)
        self.assertEqual(events["summary"]["reference_events_with_contact_but_no_unique_match"], 1)
        self.assertEqual(events["summary"]["missed_reference_events_within_rollout"], 0)

    def test_complete_duration_does_not_invent_a_reset_force_sample(self):
        reference = np.ones((11, 2), dtype=bool)
        args = self.fixture(reference=reference)
        report = contact_report(*args[:3], **args[3])
        self.assertEqual(report["coverage"]["reference_duration_coverage_fraction"], 1.0)
        self.assertEqual(report["bodies"]["hand"]["unique_reference_positive_samples_visited"], 10)
        self.assertEqual(report["bodies"]["hand"]["full_reference_positive_samples"], 11)

    def test_atomic_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            _publish(path, {"recall": None})
            with self.assertRaises(FileExistsError):
                _publish(path, {"recall": 1.0})
            self.assertIsNone(json.loads(path.read_text())["recall"])


# Box geometry, current support and post-reference physical holds.

def specification():
    return dict(contact_body_names=list(CONTACT_BODY_NAMES),
        target_top=dict(top_z_m=.7, center_xy_m=[0., 0.], axes_xy=[[1., 0.], [0., 1.]], half_extents_m=[.5, .5]),
        foot_geometry=dict(capsule_endpoints_local_m=[[[[-.05, 0., -.025], [.13, 0., -.025]]]]*2,
            capsule_radii_m=[[.01], [.01]], support_center_site_local_m=[[.04, 0., -.037]]*2),
        final_reference=dict(root_pos_rebased_m=[0., 0., 1.45]),
        thresholds=dict(root_height_error_max_m=.1, root_tilt_max_rad=.3,
            foot_support_margin_min_m=.01, sole_plane_error_min_m=-.01, sole_plane_error_max_m=.02,
            foot_current_positive_fz_min_n=20., nonfoot_current_positive_fz_max_n=5., continuous_hold_s=1.))


def states(count=1):
    root = torch.tensor([[0., 0., 1.45]], dtype=torch.float64).repeat(count, 1)
    quat = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).repeat(count, 1)
    feet = torch.tensor([[[0., -.1, .735], [0., .1, .735]]], dtype=torch.float64).repeat(count, 1, 1)
    fq = quat[:, None, :].repeat(1, 2, 1)
    force = torch.zeros((count, 9, 3), dtype=torch.float64)
    force[:, [6, 7], 2] = 150.
    return [root, quat, feet, fq, force]


class TestBoxRuntimeMetrics(unittest.TestCase):
    def test_each_physical_failure_fails_its_gate(self):
        baseline = task_metrics(*states(), specification())
        self.assertTrue(baseline["task_gate"].item())
        cases = (
            ("root_height_gate", lambda a: a[0].__setitem__((0, 2), 1.34)),
            ("root_tilt_gate", lambda a: a[1].__setitem__(0, torch.tensor([np.cos(.16), 0., np.sin(.16), 0.]))),
            ("foot_support_margin_gate", lambda a: a[2].__setitem__((0, 1, 1), .495)),
            ("foot_sole_plane_gate", lambda a: a[2].__setitem__((0, 1, 2), .756)),
            ("feet_current_support_gate", lambda a: a[4].__setitem__((0, 7, 2), 0.)),
            ("nonfoot_current_support_gate", lambda a: a[4].__setitem__((0, 0, 2), 32.3)),
        )
        for key, mutate in cases:
            with self.subTest(key=key):
                arrays = states()
                mutate(arrays)
                value = task_metrics(*arrays, specification())
                self.assertFalse(value[key].item())
                self.assertFalse(value["task_gate"].item())

    def test_exact_rolled_capsule_lowest_point_and_positive_current_force(self):
        arrays = states()
        angle = .25
        arrays[3][0, 0] = torch.tensor([np.cos(angle/2), 0., np.sin(angle/2), 0.])
        value = task_metrics(*arrays, specification())
        # Forward endpoint lowers under positive Y rotation; site height is different.
        expected = .735 - .13*np.sin(angle) - .025*np.cos(angle) - .01 - .7
        self.assertAlmostEqual(value["foot_sole_plane_error_m"][0, 0].item(), expected, places=8)
        arrays = states()
        arrays[4][0, 7, 2] = -200.
        value = task_metrics(*arrays, specification())
        self.assertEqual(value["feet_current_positive_fz_n"][0, 1], 0.)
        self.assertFalse(value["task_gate"].item())

    def test_support_centre_allows_natural_overhang_and_uses_rotated_patch(self):
        arrays = states()
        arrays[2][0, 0, 0] = .44
        value = task_metrics(*arrays, specification())
        # Toe capsule ends at .57 > patch .5, but site centre .48 has 2cm margin.
        self.assertTrue(value["task_gate"].item())
        spec = specification()
        c, s = np.cos(.6), np.sin(.6)
        spec["target_top"].update(axes_xy=[[c, s], [-s, c]], half_extents_m=[.5, .1])
        arrays = states()
        arrays[2][0, :, :2] = torch.tensor([[.3, -.1], [.3, -.1]])
        self.assertFalse(task_metrics(*arrays, spec)["foot_support_margin_gate"].item())

    def test_malformed_or_nonfinite_states_do_not_pass(self):
        arrays = states()
        arrays[1][0, 0] = 2.
        self.assertFalse(task_metrics(*arrays, specification())["valid_state_gate"].item())
        arrays = states()
        arrays[0][0, 0] = float("nan")
        self.assertFalse(task_metrics(*arrays, specification())["task_gate"].item())
        arrays = states()
        arrays[-1] = torch.zeros((1, 2, 3), dtype=torch.float64)
        with self.assertRaises(ValueError):
            task_metrics(*arrays, specification())


class TestBoxTaskGeometry(unittest.TestCase):
    def test_mesh_top_is_verified_oriented_rectangle_not_aabb(self):
        mesh = trimesh.creation.box(extents=[1.2, .36, .7])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(.4, [0, 0, 1]))
        mesh.apply_translation([.3, -.2, .35])
        component = SimpleNamespace(mesh=mesh, name="rotated", link_name="box")
        top = _rectangle_top(component, np.array([.1, -.1]))
        self.assertAlmostEqual(top["top_z_m"], .7)
        self.assertTrue(np.allclose(top["center_xy_m"], [.2, -.1]))
        self.assertTrue(np.allclose(sorted(top["half_extents_m"]), [.18, .6]))
        self.assertAlmostEqual(np.prod(top["half_extents_m"])*4, .432)
        # Removing one top triangle leaves the same hull but not filled support.
        top_mask = np.all(np.isclose(mesh.triangles[..., 2], .7), axis=-1)
        mesh.update_faces(np.arange(len(mesh.faces)) != np.flatnonzero(top_mask)[0])
        with self.assertRaisesRegex(ValueError, "rectangular|rectangle"):
            _rectangle_top(component, np.zeros(2))

    def test_loader_parses_source_capsules_and_rejects_bad_reference_without_relaxing(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path, sidecar, urdf = [directory/name for name in ("motion.npz", "motion.scene.json", "scene.urdf")]
            sidecar.write_text("{}")
            urdf.write_text("<robot/>")
            body_names = np.array(["pelvis", *FOOT_BODY_NAMES])
            roots = np.tile([0., 0., 1.45], (2, 1))
            quats = np.tile([1., 0., 0., 0.], (2, 1))
            bodies = np.tile([[0., 0., 1.45], [0., -.1, .735], [0., .1, .735]], (2, 1, 1))
            bq = np.tile([1., 0., 0., 0.], (2, 3, 1))
            def write():
                np.savez(path, body_names=body_names, root_pos=roots, root_rot_wxyz=quats,
                    body_pos_w=bodies, body_quat_w=bq, fps=50.)
            write()
            component = SimpleNamespace(mesh=trimesh.creation.box(extents=[1., 1., .7]).apply_translation([0, 0, .35]),
                name="box", link_name="box")
            scene = SimpleNamespace(metadata=dict(robot="g1", collision_asset_sha256={}), source_first_root_xy=np.zeros(2),
                components=[component], ground=SimpleNamespace(top_z_m=0.), sidecar_path=sidecar,
                motion_sha256="fake", scene_urdf_path=urdf, scene_urdf_sha256="fake", fingerprint="bound")
            with patch("morph_mimic.utils.box_task.load_reference_scene", return_value=scene):
                spec = load_box_task_spec(path)
                self.assertTrue(spec["reference_endpoint_geometry_passed"])
                self.assertEqual(len(spec["foot_geometry"]["capsule_radii_m"][0]), 7)
                self.assertTrue(np.allclose(spec["foot_geometry"]["support_center_site_local_m"], [[.04, 0, -.037]]*2))
                self.assertFalse(spec["reference_endpoint_forces_verified"])
                json.dumps(spec, allow_nan=False)
                bodies[-1, 2, 2] += .04
                write()
                with self.assertRaisesRegex(ValueError, "supporting both reference endpoint feet"):
                    load_box_task_spec(path)


class TestBoxPhysicalHold(unittest.TestCase):
    def test_time_endpoint_alone_is_ineligible_and_one_bad_hold_endpoint_fails(self):
        def run(n, mutation=None, start=0.):
            times = (np.arange(n)+1)*.02
            phase = np.minimum(1., start+times/10.)
            metrics = {"task_gate": np.ones(n, dtype=bool), "feet_current_support_gate": np.ones(n, dtype=bool)}
            if mutation is not None:
                metrics["task_gate"][mutation] = False
                metrics["feet_current_support_gate"][mutation] = False
            return summarize_box_hold(metrics, times, phase, duration_s=10., start_phase=start, control_dt_s=.02)
        missing = run(500)
        self.assertFalse(missing["passed"])
        self.assertIn("missing_full_one_second_post_reference_physical_hold", missing["failure_reasons"])
        self.assertFalse(run(549)["passed"])
        passed = run(550)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["recorded_post_reference_control_endpoints"], 50)
        failed = run(550, -25)
        self.assertFalse(failed["passed"])
        self.assertIn("feet_current_support_gate", failed["failure_reasons"])
        self.assertFalse(run(380, start=.34)["passed"])

    def test_gapped_or_rewound_records_are_rejected(self):
        gates = {"task_gate": np.ones(3, dtype=bool)}
        for times, phase in (([.02, .04, .08], [.002, .004, .008]),
                             ([.02, .04, .06], [.002, 0, .006]),
                             ([10.02, 10.04, 10.06], [1., 1., 1.]),
                             ([.02, .04, .06], [1., 1., 1.])):
            with self.assertRaises(ValueError):
                summarize_box_hold(gates, times, phase, duration_s=10., start_phase=0., control_dt_s=.02)


class TestRecordedBoxAudit(unittest.TestCase):
    def test_current_force_and_full_hold_are_required_even_with_positive_history_flags(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            motion, trace = directory/"motion.npz", directory/"trace.npz"
            np.savez(motion, root_pos=np.tile([0., 0., 1.45], (501, 1)))
            spec = specification() | dict(motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),
                source_first_root_xy=[0., 0.], robot_xml=str(PROJECT_ROOT/"assets/g1/robot/g1.xml"),
                scope="test audit logic", unmonitored_contact_limit="named links only")
            spec["final_reference"]["duration_s"] = 10.
            arrays = states(550)
            root, quat, feet, foot_quat, forces = [value.numpy() for value in arrays]
            origin = np.array([7., -2., .4])
            names = json.loads((PROJECT_ROOT/"assets/g1/robot_spec.json").read_text())["joint_names"]
            metadata = dict(all_joint_names=names, contact_body_names=list(CONTACT_BODY_NAMES),
                control_dt_s=.02, source_sha256=spec["motion_sha256"],
                result=dict(start_phase=0., contract_verified=True, completed=True, failure=False,
                    terminated=False, budget_limited=False, interrupted=False))
            data = dict(root_pos_w=root+origin, ref_root_pos_w=root+origin, root_quat_w=quat,
                all_joint_pos=np.zeros((550, 29)), contact_force_w=forces,
                measured_contact_flags=np.ones((550, 9)), elapsed_control_time_s=(np.arange(550)+1)*.02,
                phase=np.minimum(1., (np.arange(550)+1)*.002), metadata_json=np.asarray(json.dumps(metadata)))
            def audit():
                np.savez(trace, **data)
                with patch("morph_mimic.diagnostics.evaluate_box_task.source_foot_kinematics",
                           return_value=(feet, foot_quat)) as fk:
                    report, _ = evaluate_box_task_trace(trace, motion, task_spec=spec)
                    self.assertTrue(np.allclose(fk.call_args.args[0], root))
                return report
            passed = audit()
            self.assertTrue(passed["task_passed"])
            self.assertFalse(passed["physics_stepped_by_this_script"])
            data["contact_force_w"][-20, 7, 2] = 0.
            failed = audit()
            self.assertFalse(failed["task_passed"])
            self.assertIn("feet_current_support_gate", failed["failure_reasons"])
            data["contact_force_w"][-20, 7, 2] = 150.
            data["metadata_json"] = np.asarray(json.dumps(metadata | dict(diagnostic_only=True)))
            self.assertIn("source_trace_is_diagnostic_only", audit()["failure_reasons"])


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "source-model FK needs optional MuJoCo")
class TestBoxSourceFk(unittest.TestCase):
    def test_source_capsule_formula_matches_mujoco_geom_support(self):
        import mujoco
        from morph_mimic.utils.box_task import _foot_geometry

        path = PROJECT_ROOT/"assets/g1/robot/g1.xml"
        geometry, _ = _foot_geometry(path)
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        data.qpos[2] = 1.4
        names = [model.joint(i).name for i in range(1, model.njnt)]
        joints = np.zeros((1, 29))
        joints[0, names.index("right_ankle_pitch_joint")] = .35
        fp, fq = source_foot_kinematics(np.array([[0., 0., 1.4]]), np.array([[1., 0., 0., 0.]]), joints, names, path)
        qids = [model.joint(name).qposadr[0] for name in names]
        data.qpos[qids] = joints[0]
        mujoco.mj_forward(model, data)
        spec = specification()
        spec["foot_geometry"] = geometry
        arrays = states()
        arrays[2], arrays[3] = torch.from_numpy(fp), torch.from_numpy(fq)
        value = task_metrics(*arrays, spec)
        for foot, capsule_names in enumerate(geometry["capsule_names"]):
            minimum = min(data.geom_xpos[model.geom(name).id, 2]
                - abs(data.geom_xmat[model.geom(name).id].reshape(3, 3)[2, 2])*model.geom(name).size[1]
                - model.geom(name).size[0] for name in capsule_names)
            self.assertAlmostEqual(value["foot_sole_plane_error_m"][0, foot].item(), minimum-.7, places=9)


# Common coordinate frames, derivatives and named source FK.

class TestFrameAudit(unittest.TestCase):
    def fixture(self):
        actual_root = np.array([[.2, -.1, 1.], [.3, -.2, 1.1]])
        reference_root = np.array([[0., 0., 1.2], [.1, -.1, 1.2]])
        actual_quat = yaw_quaternion(np.array([.1, .3]))
        reference_quat = yaw_quaternion(np.array([.2, .4]))
        actual = actual_root[:, None]+np.array([[[.2, .1, -.8], [-.1, .2, -.5]]])
        reference = reference_root[:, None]+np.array([[[.1, .1, -.8], [-.1, .1, -.5]]])
        return actual, reference, actual_root, reference_root, actual_quat, reference_quat

    def test_common_frame_invariant_to_simultaneous_world_yaw_and_translation(self):
        values = self.fixture()
        before = frame_errors(*values)
        rotation = yaw_quaternion(np.array(.87))
        translation = np.array([50., -80., 3.])
        transformed = [rotate(rotation, value)+translation for value in values[:4]]
        transformed += [multiply(rotation, value) for value in values[4:]]
        after = frame_errors(*transformed)
        for key in ("common_reference_yaw_error_m", "common_reference_full_root_error_m", "independent_root_yaw_shape_error_m"):
            np.testing.assert_allclose(after[key], before[key], atol=1e-13, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(after["world_error_m"], axis=-1),
            np.linalg.norm(before["world_error_m"], axis=-1), atol=1e-13, rtol=0)

    def test_own_root_shape_hides_horizontal_translation_but_common_frame_preserves_it(self):
        reference_root = np.array([[0., 0., 1.]])
        reference = reference_root[:, None]+np.array([[[.2, .1, -.8]]])
        displacement = np.array([.4, -.3, 0.])
        quaternion = yaw_quaternion(np.zeros(1))
        errors = frame_errors(reference+displacement, reference, reference_root+displacement,
            reference_root, quaternion, quaternion)
        np.testing.assert_allclose(errors["independent_root_yaw_shape_error_m"], 0., atol=1e-15)
        np.testing.assert_allclose(errors["common_reference_yaw_error_m"], displacement[None, None])
        elevated = frame_errors(reference+np.array([0., 0., .2]), reference,
            reference_root+np.array([0., 0., .2]), reference_root, quaternion, quaternion)
        np.testing.assert_allclose(elevated["independent_root_yaw_shape_error_m"][..., 2], .2)

    def test_antipodal_quaternions_have_zero_error(self):
        quaternion = yaw_quaternion(np.array([.2, -.4, 2.9]))
        relative, angle = orientation_error(quaternion, -quaternion)
        np.testing.assert_allclose(angle, 0., atol=1e-15)
        np.testing.assert_allclose(relative, np.broadcast_to([1., 0., 0., 0.], (3, 4)), atol=1e-15)

    def test_named_mapping_refuses_missing_and_duplicate_columns(self):
        np.testing.assert_array_equal(named_columns(["knee", "hip", "ankle"], ["hip", "ankle", "knee"]), [1, 2, 0])
        for names, requested in ((["hip", "hip"], ["hip"]), (["hip"], ["hip", "hip"]), (["hip"], ["knee"])):
            with self.subTest(names=names, requested=requested), self.assertRaises(ValueError):
                named_columns(names, requested)

    def test_derivatives_exclude_initial_and_missing_control_intervals(self):
        time = np.array([.02, .04, .08, .1])
        values = np.broadcast_to(time[:, None, None], (4, 2, 3)).copy()
        velocity, valid = finite_difference(values, time, .02)
        np.testing.assert_array_equal(valid, [False, True, False, True])
        np.testing.assert_allclose(velocity[valid], 1.)
        np.testing.assert_allclose(velocity[~valid], 0.)

    def test_initialization_phase_is_not_first_poststep_endpoint(self):
        phases = np.array([.342, .344, .346])
        coverage = phase_coverage({"result": {"start_phase": .34}}, phases)
        self.assertEqual(coverage["start_phase"], .34)
        self.assertEqual(coverage["initialization_phase"], .34)
        self.assertEqual(coverage["first_recorded_phase"], .342)
        self.assertIsNone(phase_coverage({}, phases)["initialization_phase"])
        with self.assertRaises(ValueError):
            phase_coverage({"result": {"start_phase": .4}}, phases)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "Optional CPU MuJoCo required")
    def test_source_fk_joint_permutation_keeps_body_and_joint_pivots(self):
        import mujoco

        path = Path(__file__).resolve().parents[1]/"assets/g1/robot/g1.xml"
        model = mujoco.MjModel.from_xml_path(str(path))
        names = [model.joint(i).name for i in range(model.njnt)
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        q = np.linspace(-.05, .05, 29)[None]
        first, second = SourceFK(path, names), SourceFK(path, names[::-1])
        args = (np.array([[.1, -.2, 1.]]), yaw_quaternion(np.array([.2])))
        direct = first.poses(*args, q, include_joint_anchors=True)
        permuted = second.poses(*args, q[:, ::-1], include_joint_anchors=True)
        for a, b in zip(direct, permuted):
            np.testing.assert_allclose(a, b, atol=0, rtol=0)
        self.assertEqual(direct[0].shape, (1, 30, 3))
        self.assertEqual(direct[2].shape, (1, 29, 3))


if __name__ == "__main__":
    unittest.main()
