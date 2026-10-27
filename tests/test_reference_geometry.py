"""Independent collision distances and safe geometric-label quality gates."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from morph_mimic.contacts.reference_geometry import (
    GeometryThresholds, assess_candidates, filter_segments, geometry_hindsight, sample_collision_geometry,
)
from morph_mimic.scripts.label_reference_geometry import write_outputs


class TestGeometricSchedule(unittest.TestCase):
    def test_default_preserves_single_geometric_contacts_and_optional_filter_is_explicit(self):
        flags = np.array([[0], [1], [0], [1], [1]], dtype=bool)
        np.testing.assert_array_equal(filter_segments(flags), flags)
        np.testing.assert_array_equal(filter_segments(flags, 2).ravel(), [0, 0, 0, 1, 1])

    def test_hindsight_uses_future_body_origin_query_yaw_and_absolute_height(self):
        flags = np.zeros((5, 1), dtype=bool)
        flags[2:4] = True
        body = np.tile([[[1, 3, 4]]], (5, 1, 1))
        root = np.tile([[1, 2, 3]], (5, 1))
        quat = np.tile([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]], (5, 1))
        thresholds = GeometryThresholds(footprint_x_m=(-2, 2), footprint_y_m=(-2, 2))
        targets, occupied = geometry_hindsight(flags, body, root, quat, 50, thresholds)
        np.testing.assert_allclose(targets[0, 0], [1, 0, 4], atol=1e-7)
        np.testing.assert_array_equal(occupied.ravel(), [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(targets[2:], 0)

    def test_penetration_and_contact_free_gaps_prevent_formal_labels(self):
        samples = dict(signed_distance_m=np.full((20, 3), 0.005), penetration_m=np.zeros((20, 3)),
                       all_robot_penetration_m=np.zeros(20))
        names = ["left_hand", "left_foot", "right_foot"]
        good, _ = assess_candidates(samples, names, 50, "box_climb", GeometryThresholds())
        self.assertTrue(good["accepted"])
        self.assertFalse(good["physics_validated"])
        self.assertIsNone(good["force_threshold_n"])
        samples["all_robot_penetration_m"][0] = 0.021
        penetrated, _ = assess_candidates(samples, names, 50, "box_climb", GeometryThresholds())
        self.assertFalse(penetrated["accepted"])
        self.assertTrue(any("penetration" in x for x in penetrated["rejection_reasons"]))
        samples["all_robot_penetration_m"][:] = 0
        samples["signed_distance_m"][:] = 0.1
        gap, _ = assess_candidates(samples, names, 50, "box_climb", GeometryThresholds())
        self.assertFalse(gap["accepted"])
        self.assertAlmostEqual(gap["maximum_no_contact_gap_s"], 0.4)

    def test_failed_write_request_saves_diagnostics_and_refuses_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, scene = root / "source.npz", root / "source.scene.json"
            source.write_bytes(b"original motion")
            scene.write_text("{}")
            report = dict(accepted=False, source_motion=str(source), scene_path=str(scene),
                          training_contact_labels_generated=False)
            contact = root / "source_contacts.npz"
            self.assertFalse(write_outputs(report, {"distance": np.array([0.2])}, {}, root / "diagnostic.json", root / "diagnostic.npz", contact))
            self.assertTrue(json.loads((root / "diagnostic.json").read_text())["requested_label_output_refused"])
            self.assertFalse(contact.exists())
            self.assertEqual(source.read_bytes(), b"original motion")
            with self.assertRaises(FileExistsError):
                write_outputs(report, {}, {}, root / "diagnostic.json", root / "diagnostic.npz")
            with self.assertRaisesRegex(ValueError, "source motion/scene"):
                write_outputs(report, {}, {}, source, root / "other.npz")

    def test_low_box_foot_only_support_is_allowed_unless_arm_support_is_requested(self):
        samples = dict(signed_distance_m=np.tile([0.25, 0.005, 0.005], (20, 1)),
                       penetration_m=np.zeros((20, 3)), all_robot_penetration_m=np.zeros(20))
        names = ["left_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"]
        ordinary, flags = assess_candidates(samples, names, 50, "box_climb", GeometryThresholds())
        self.assertTrue(ordinary["accepted"])
        self.assertFalse(flags[:, 0].any())
        self.assertTrue(flags[:, 1:].all())
        supported, _ = assess_candidates(samples, names, 50, "box_climb",
                                          GeometryThresholds(require_hand_contact=True))
        self.assertFalse(supported["accepted"])
        self.assertTrue(any("arm-support" in reason for reason in supported["rejection_reasons"]))


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is needed for collision-distance checks")
class TestSignedCollisionDistances(unittest.TestCase):
    def test_separation_surface_points_and_non_label_body_penetration(self):
        import mujoco

        xml = '''<mujoco><worldbody>
          <geom name="ground" type="box" size="2 2 .5" pos="0 0 -.5"/>
          <body name="label_body"><freejoint/><geom name="sphere" type="sphere" size=".1" mass="1"/>
            <body name="unlabeled_torso" pos="1 0 -.02"><geom type="sphere" size=".1" mass="1"/></body>
          </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        states = np.array([[0, 0, 0.105, 1, 0, 0, 0], [0, 0, 0.08, 1, 0, 0, 0]])
        measured = sample_collision_geometry(model, {model.geom("ground").id}, [model.body("label_body").id], states)
        np.testing.assert_allclose(measured["signed_distance_m"][:, 0], [0.005, -0.02], atol=1e-12)
        np.testing.assert_allclose(measured["scene_surface_point_w"][:, 0, 2], 0, atol=1e-12)
        np.testing.assert_allclose(measured["body_origin_w"][:, 0, 2], states[:, 2], atol=1e-12)
        np.testing.assert_allclose(measured["all_robot_penetration_m"], [0.015, 0.04], atol=1e-12)
        self.assertTrue(measured["surface_point_valid"].all())


if __name__ == "__main__":
    unittest.main()
