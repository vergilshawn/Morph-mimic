"""Optional reference/data regressions, including scene binding and source FK.

CPU checks retain the original scene/data invariants. MuJoCo-dependent classes
and methods explicitly skip when that optional backend is not installed.
"""

import ast
from copy import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from morph_mimic.contacts.reference_geometry import (
    GeometryThresholds, assess_candidates, filter_segments, geometry_hindsight,
    sample_collision_geometry,
)
from morph_mimic.contacts.self_collision import effective_robot_geom_pairs, sample_self_contacts
from morph_mimic.contacts.settle_reference import (
    _hermite, _multiply, _quat_exp, _rotation_vector, _world_angular_velocity,
    append_settle_reference,
)
from morph_mimic.diagnostics.validate_scene_reference import (
    _body_columns, build_model, scene_contacts, simulate,
)
from morph_mimic.scripts import common
from morph_mimic.scripts.common import configure_reference_scene
from morph_mimic.scripts.import_omniretarget import convert_clip, resample_qpos
from morph_mimic.scripts.label_reference_geometry import write_outputs
from morph_mimic.scripts.prepare_motion import prepare_clip
from morph_mimic.scripts.prepare_standing_reference import prepare_standing_reference
from morph_mimic.scripts.project_reference import project_clip
from morph_mimic.scripts.retarget_scene_reference import SceneFit
from morph_mimic.utils import checkpoint as cp
from morph_mimic.utils.motion import (
    MotionDataset, validate_contact_body_order, validate_motion_arrays,
)
from morph_mimic.utils.reference_scene import (
    load_reference_scene, reference_scene_grid_shape, reference_scene_mesh_copy,
)
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec
from morph_mimic.utils.transforms import to_body_yaw_frame, yaw_from_quat
from tests.reference_fixtures import named_motion, scene_motion


NS = SimpleNamespace



# Prepare motion.

class TestPrepareMotion(unittest.TestCase):
    def source(self, root):
        path = Path(root) / "walk1_subject1.npz"
        arrays = dict(fps=np.array([10.0]), joint_names=np.array(["joint"]), body_names=np.array(["base"]))
        for key, width in (("joint_pos", 1), ("joint_vel", 1), ("body_pos_w", 3), ("body_quat_w", 4),
                           ("body_lin_vel_w", 3), ("body_ang_vel_w", 3), ("extra_per_frame", 2)):
            arrays[key] = np.arange(21 * width, dtype=np.float32).reshape(21, width)
        np.savez(path, **arrays)
        return path

    def test_all_time_arrays_share_the_same_window_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            original = hashlib.sha256(source.read_bytes()).hexdigest()
            output, meta = prepare_clip(source, Path(directory) / "prepared", 0.5, 0.8)
            self.assertEqual(output.name, source.name)
            self.assertEqual(meta["prepared_num_frames"], 9)
            with np.load(output) as result, np.load(source) as full:
                for key in meta["sliced_fields"]:
                    np.testing.assert_array_equal(result[key], full[key][5:14])
                np.testing.assert_array_equal(result["joint_names"], full["joint_names"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)
            with self.assertRaises(FileExistsError):
                prepare_clip(source, output.parent, 0.5, 0.8)

    def test_invalid_range_and_non_frame_aligned_requests_leave_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(directory)
            output = Path(directory) / "prepared"
            for start, duration in ((1.5, 1.0), (0.05, 0.5), (0.0, -1.0)):
                with self.subTest(start=start, duration=duration), self.assertRaises(ValueError):
                    prepare_clip(source, output, start, duration)
            self.assertFalse(output.exists())


# Reference geometry.

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


# Reference projection.

class TestMotionStateContract(unittest.TestCase):
    def test_contact_columns_reject_swapped_hands_and_feet_and_allow_explicit_aliases(self):
        spec = RobotSpec.load("g1")
        bodies = ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_elbow_link", "right_elbow_link",
                  "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link", "pelvis"]
        validate_contact_body_order({"body_names": np.array(bodies)}, spec)
        swapped = bodies.copy()
        swapped[0], swapped[6] = swapped[6], swapped[0]
        with self.assertRaisesRegex(ValueError, "contact column 0.*expected left hands"):
            validate_contact_body_order({"body_names": np.array(swapped)}, spec)
        spec.asset_aliases = {"bodies": {"left_wrist_yaw_link": "custom_left_hand"}}
        bodies[0] = "custom_left_hand"
        validate_contact_body_order({"body_names": np.array(bodies)}, spec)

    def test_legacy_contact_column_order_is_explicitly_unverified(self):
        with self.assertWarnsRegex(RuntimeWarning, "column order is unverified"):
            validate_contact_body_order({}, RobotSpec.load("g1"))

    def test_raw_g1_reference_is_rejected_before_controlled_joint_selection(self):
        spec = RobotSpec.load("g1")
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "walk1_subject1.npz", **named_motion(moving_held=True))
            with self.assertRaisesRegex(ValueError, "held joints move.*Generate consistent FK"):
                MotionDataset(spec, directory, clip_names=["walk1_subject1"])

    def test_complete_names_and_shapes_are_validated_even_when_hold_check_is_disabled(self):
        spec = RobotSpec.load("g1")
        arrays = named_motion(moving_held=True)
        validate_motion_arrays(arrays, spec, require_nominal_held=False)
        arrays["joint_names"][1] = arrays["joint_names"][0]
        with self.assertRaisesRegex(ValueError, "duplicate joint_names"):
            validate_motion_arrays(arrays, spec, require_nominal_held=False)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is needed for reference projection")
class TestReferenceProjection(unittest.TestCase):
    def source(self, directory):
        arrays = named_motion(moving_held=True)
        # Exercise both independent body/joint permutations and moving roots.
        base = arrays["body_names"].tolist().index("pelvis")
        arrays["body_lin_vel_w"][:, base] = [0.2, -0.4, 0.1]
        arrays["body_ang_vel_w"][:, base] = [0.5, -0.3, 0.7]
        joint_order = np.arange(len(arrays["joint_names"]))[::-1]
        body_order = np.roll(np.arange(len(arrays["body_names"])), 7)
        for key in ("joint_pos", "joint_vel"):
            arrays[key] = arrays[key][:, joint_order]
        for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            arrays[key] = arrays[key][:, body_order]
        arrays["joint_names"] = arrays["joint_names"][joint_order]
        arrays["body_names"] = arrays["body_names"][body_order]
        arrays["contact_flags"] = np.ones((5, 9), dtype=bool)
        source = Path(directory) / "walk1_subject1.npz"
        np.savez(source, **arrays)
        np.savez(Path(directory) / "walk1_subject1_contacts.npz", contact_flags=np.ones((5, 9)))
        return source, arrays

    def test_projection_preserves_source_and_controlled_motion_but_rebuilds_held_fk(self):
        spec = RobotSpec.load("g1")
        with tempfile.TemporaryDirectory() as directory:
            source, original = self.source(directory)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            output, meta = project_clip(source, Path(directory) / "projected", spec)
            with np.load(output, allow_pickle=False) as result:
                names = result["joint_names"].tolist()
                controlled = [names.index(name) for name in spec.controlled_joint_names]
                held = [names.index(name) for name in spec.held_joint_names]
                nominal = np.asarray(spec.default_pose)[[spec.joint_names.index(name) for name in spec.held_joint_names]]
                np.testing.assert_array_equal(result["joint_pos"][:, controlled], original["joint_pos"][:, controlled])
                np.testing.assert_array_equal(result["joint_vel"][:, controlled], original["joint_vel"][:, controlled])
                np.testing.assert_allclose(result["joint_pos"][:, held], np.broadcast_to(nominal, (5, len(held))), atol=1e-7)
                np.testing.assert_array_equal(result["joint_vel"][:, held], 0)
                self.assertGreater(float(np.abs(result["body_pos_w"] - original["body_pos_w"]).max()), 1e-4)
                self.assertNotIn("contact_flags", result.files)
                self.assertNotIn("states", result.files)
            dataset = MotionDataset(spec, output.parent)
            self.assertEqual(dataset.clips[output.stem].num_frames, 5)
            self.assertAlmostEqual(dataset.clips[output.stem].duration_s, 4 / 50)
            self.assertEqual(dataset.motion_dir, output.parent.resolve())
            self.assertFalse((output.parent / "walk1_subject1_contacts.npz").exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            self.assertEqual(meta["source_sha256"], source_hash)
            self.assertEqual(meta["projected_motion_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            sidecar = json.loads(output.with_suffix(".projection.json").read_text())
            self.assertEqual(sidecar["profile_sha256"], meta["profile_sha256"])
            self.assertFalse(sidecar["physical_tracking_validated"])
            self.assertFalse(sidecar["contact_labels_inherited"])
            with self.assertRaises(FileExistsError):
                project_clip(source, output.parent, spec)

    def test_all_body_velocities_match_independent_integrated_position_differences(self):
        import mujoco

        spec = RobotSpec.load("g1")
        model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / spec.runtime_assets["mjcf"]))
        data = mujoco.MjData(model)
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self.source(directory)
            output, _ = project_clip(source, Path(directory) / "projected", spec)
            with np.load(output, allow_pickle=False) as result:
                joints, bodies = result["joint_names"].tolist(), result["body_names"].tolist()
                qids = [model.joint(name).qposadr[0] for name in joints]
                vids = [model.joint(name).dofadr[0] for name in joints]
                bids = [model.body(name).id for name in bodies]
                base = bodies.index(spec.base_link)
                for frame in range(len(result["joint_pos"])):
                    data.qpos[:7] = np.concatenate((result["root_pos"][frame], result["root_rot_wxyz"][frame]))
                    data.qpos[qids] = result["joint_pos"][frame]
                    data.qvel[:3] = result["body_lin_vel_w"][frame, base]
                    mujoco.mj_forward(model, data)
                    data.qvel[3:6] = data.xmat[model.body(spec.base_link).id].reshape(3, 3).T @ result["body_ang_vel_w"][frame, base]
                    data.qvel[vids] = result["joint_vel"][frame]
                    mujoco.mj_forward(model, data)
                    np.testing.assert_allclose(result["body_pos_w"][frame], data.xpos[bids], rtol=0, atol=2e-6)
                    before = data.xpos[bids].copy()
                    before_rotation = data.xmat[bids].reshape(-1, 3, 3).copy()
                    epsilon = 1e-6
                    mujoco.mj_integratePos(model, data.qpos, data.qvel, epsilon)
                    mujoco.mj_forward(model, data)
                    numerical_velocity = (data.xpos[bids] - before) / epsilon
                    np.testing.assert_allclose(result["body_lin_vel_w"][frame], numerical_velocity, rtol=0, atol=3e-5)
                    delta = data.xmat[bids].reshape(-1, 3, 3) @ np.swapaxes(before_rotation, -1, -2)
                    angular = np.stack((delta[:, 2, 1] - delta[:, 1, 2], delta[:, 0, 2] - delta[:, 2, 0],
                                        delta[:, 1, 0] - delta[:, 0, 1]), axis=-1) / (2 * epsilon)
                    np.testing.assert_allclose(result["body_ang_vel_w"][frame], angular, rtol=0, atol=3e-5)


# Reference scene.

def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestReferenceScene(unittest.TestCase):
    def test_replica_grid_has_one_distinct_cell_per_environment(self):
        for count in (1, 2, 3, 5, 16, 32, 128, 256, 512):
            with self.subTest(count=count):
                rows, cols = reference_scene_grid_shape(count)
                self.assertGreaterEqual(rows * cols, count)
                self.assertLess(rows * cols - count, rows)
                assigned = [(i // cols, i % cols) for i in range(count)]
                self.assertEqual(len(set(assigned)), count)
                self.assertLess(assigned[-1][0], rows)
        self.assertEqual(reference_scene_grid_shape(1), (1, 1))
        for count in (0, -1, 1.5, True, "2", None):
            with self.subTest(invalid=count), self.assertRaisesRegex(ValueError, "positive integer"):
                reference_scene_grid_shape(count)

    def test_replica_geometry_is_verified_once_and_each_tile_gets_an_independent_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar, _ = self.fixture(directory)
            scene = load_reference_scene(sidecar)
            expected = scene.rebased_mesh().vertices.copy()
            with patch("morph_mimic.utils.reference_scene.load_reference_scene", wraps=load_reference_scene) as loader:
                first = reference_scene_mesh_copy(sidecar, scene.fingerprint)
                first.apply_translation((100., 200., 0.))
                second = reference_scene_mesh_copy(sidecar, scene.fingerprint)
                self.assertEqual(loader.call_count, 1)
                np.testing.assert_allclose(second.vertices, expected)
                with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                    reference_scene_mesh_copy(sidecar, "0" * 64)
            # Unbound/direct callers cannot bypass source hashing through cache.
            with (Path(directory) / "obstacle.obj").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                reference_scene_mesh_copy(sidecar)

    def fixture(self, directory, urdf=None):
        directory = Path(directory)
        mesh = directory / "obstacle.obj"
        trimesh.creation.box(extents=(2, 4, 6)).export(mesh)
        motion = directory / "motion.npz"
        np.savez(motion, root_pos=np.array([[2., -3., 1.], [2.1, -3., 1.]]))
        source = directory / "source.npz"
        np.savez(source, qpos=np.zeros((2, 36)))
        terrain = directory / "terrain.urdf"
        terrain.write_text(urdf or '''<robot name="terrain">
          <link name="root"><collision name="scaled_mesh"><origin xyz="1 0 0"/>
            <geometry><mesh filename="obstacle.obj" scale="2 1 0.5"/></geometry></collision></link>
          <link name="child"><collision name="primitive"><geometry><box size="1 2 3"/></geometry></collision></link>
          <joint name="world_root" type="fixed"><parent link="world"/><child link="root"/>
            <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/></joint>
          <joint name="root_child" type="fixed"><parent link="root"/><child link="child"/>
            <origin xyz="0 1 0"/></joint></robot>''')
        metadata = dict(schema=1, robot="g1", motion_file="motion.npz", motion_sha256=_hash(motion),
                        source_motion={"path": "source.npz", "sha256": _hash(source)},
                        scene_urdf={"path": "terrain.urdf", "sha256": _hash(terrain)},
                        source_first_root_xy=[2, -3], world_position=[10, 0, 0],
                        world_quat_wxyz=[1, 0, 0, 0],
                        ground=dict(top_z_m=0., size_m=[10., 10.], thickness_m=1., center_xy_m=[0., 0.]),
                        collision_asset_sha256={"obstacle.obj": _hash(mesh)})
        sidecar = directory / "motion.scene.json"
        sidecar.write_text(json.dumps(metadata))
        return sidecar, metadata

    def rewrite(self, sidecar, metadata):
        sidecar.write_text(json.dumps(metadata))

    def test_fixed_tree_collision_origin_mesh_scale_and_rebase_share_one_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar, _ = self.fixture(directory)
            scene = load_reference_scene(sidecar)
            self.assertEqual(len(scene.components), 2)
            mesh, box = scene.components
            np.testing.assert_allclose(mesh.mesh.bounds, [[9, 1, 1.5], [13, 5, 4.5]], atol=1e-8)
            np.testing.assert_allclose(box.mesh.bounds, [[9, 1.5, 1.5], [11, 2.5, 4.5]], atol=1e-8)
            self.assertEqual(box.size_m, (1, 2, 3))
            np.testing.assert_allclose(mesh.transform[:3, 3], [11, 3, 3], atol=1e-8)
            np.testing.assert_allclose(scene.ground.mesh().bounds, [[-5, -5, -1], [5, 5, 0]])
            source = scene.combined_mesh()
            rebased = scene.rebased_mesh()
            np.testing.assert_allclose(rebased.vertices, source.vertices + [-2, 3, 0])
            np.testing.assert_array_equal(source.faces, rebased.faces)
            # Combining/rebasing must never mutate individual source components.
            np.testing.assert_allclose(mesh.mesh.bounds, [[9, 1, 1.5], [13, 5, 4.5]], atol=1e-8)

    def test_fingerprint_is_portable_but_binds_ground_and_motion_rebase(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            original.mkdir()
            sidecar, metadata = self.fixture(original)
            first = load_reference_scene(sidecar).fingerprint
            copied = Path(directory) / "copied"
            shutil.copytree(original, copied)
            self.assertEqual(first, load_reference_scene(copied / sidecar.name).fingerprint)
            metadata["ground"]["top_z_m"] = 0.2
            self.rewrite(sidecar, metadata)
            self.assertNotEqual(first, load_reference_scene(sidecar).fingerprint)
            metadata["source_first_root_xy"] = [99, 0]
            self.rewrite(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "source_first_root_xy differs"):
                load_reference_scene(sidecar)

    def test_all_source_hashes_are_checked_and_dependency_set_is_exact(self):
        for field in ("motion", "source_motion", "scene_urdf", "collision_asset"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                sidecar, _ = self.fixture(directory)
                filenames = {"motion": "motion.npz", "source_motion": "source.npz",
                             "scene_urdf": "terrain.urdf", "collision_asset": "obstacle.obj"}
                path = Path(directory) / filenames[field]
                with path.open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, field + ": SHA256 mismatch"):
                    load_reference_scene(sidecar)
        with tempfile.TemporaryDirectory() as directory:
            sidecar, metadata = self.fixture(directory)
            metadata["collision_asset_sha256"] = {}
            self.rewrite(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "missing from collision_asset_sha256"):
                load_reference_scene(sidecar)
            metadata["collision_asset_sha256"] = {"source.npz": _hash(Path(directory) / "source.npz"),
                                                   "obstacle.obj": _hash(Path(directory) / "obstacle.obj")}
            self.rewrite(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "files not used"):
                load_reference_scene(sidecar)

    def test_rejects_unsupported_dynamic_or_ambiguous_geometry(self):
        cases = [
            ("<sphere radius='1'/>", "unsupported geometry sphere"),
            ("<box size='1 0 2'/>", "positive box size"),
            ("<mesh filename='obstacle.obj' scale='-1 1 1'/>", "positive mesh scale"),
            ("<mesh filename='https://example.com/obstacle.obj'/>", "local file path"),
        ]
        for geometry, error in cases:
            with self.subTest(geometry=geometry), tempfile.TemporaryDirectory() as directory:
                sidecar, _ = self.fixture(directory, f"<robot><link name='root'><collision><geometry>{geometry}</geometry></collision></link></robot>")
                with self.assertRaisesRegex(ValueError, error):
                    load_reference_scene(sidecar)
        for joint, error in [
            ("<joint name='j' type='revolute'><parent link='world'/><child link='root'/></joint>", "fixed joints only"),
            ("<joint name='j' type='fixed'><parent link='missing'/><child link='root'/></joint>", "missing parent link"),
            ("<joint name='j' type='fixed'><parent link='root'/><child link='root'/></joint>", "cycle"),
        ]:
            with self.subTest(joint=joint), tempfile.TemporaryDirectory() as directory:
                sidecar, _ = self.fixture(directory, "<robot><link name='root'><collision><geometry><mesh filename='obstacle.obj'/></geometry></collision></link>" + joint + "</robot>")
                with self.assertRaisesRegex(ValueError, error):
                    load_reference_scene(sidecar)

    def test_implicit_world_accepts_multiple_fixed_children_and_world_quaternion(self):
        urdf = """<robot><link name='a'><collision><geometry><box size='1 1 1'/></geometry></collision></link>
        <link name='b'><collision><geometry><box size='1 1 1'/></geometry></collision></link>
        <joint name='a_j' type='fixed'><parent link='world'/><child link='a'/><origin xyz='1 0 0'/></joint>
        <joint name='b_j' type='fixed'><parent link='world'/><child link='b'/><origin xyz='0 2 0'/></joint></robot>"""
        with tempfile.TemporaryDirectory() as directory:
            sidecar, metadata = self.fixture(directory, urdf)
            metadata["collision_asset_sha256"] = {}
            metadata["world_quat_wxyz"] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
            self.rewrite(sidecar, metadata)
            scene = load_reference_scene(sidecar)
            np.testing.assert_allclose(scene.components[0].mesh.bounds.mean(axis=0), [10, 1, 0], atol=1e-8)
            np.testing.assert_allclose(scene.components[1].mesh.bounds.mean(axis=0), [8, 0, 0], atol=1e-8)
            metadata["world_quat_wxyz"] = [2, 0, 0, 0]
            self.rewrite(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "unit WXYZ"):
                load_reference_scene(sidecar)

    def test_json_duplicate_keys_and_motion_path_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar, _ = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "requested motion path differs"):
                load_reference_scene(sidecar, Path(directory) / "source.npz")
            sidecar.write_text('{"schema":1,"schema":1}')
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_reference_scene(sidecar)


# Scene integration.

CLIP = "climb_18_z_scale_1.0"


ISAAC_GENERATOR = Path("/home/user/IsaacLab/source/isaaclab/isaaclab/terrains/terrain_generator.py")


def _compile(nodes, path, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope


def _environment_class(base=object, terrain_builder=None):
    path = PROJECT_ROOT / "morph_mimic/envs/acp_env.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "AcpEnv")
    return _compile([cls], path, dict(np=np, torch=torch, DirectRLEnv=base,
                                     TERRAIN_GROUP={"reference_scene": "dense_whole_body"},
                                     build_terrain_cfg=terrain_builder,
                                     to_body_yaw_frame=to_body_yaw_frame, yaw_from_quat=yaw_from_quat))["AcpEnv"]


def _terrain_function():
    path = PROJECT_ROOT / "morph_mimic/envs/terrains.py"
    function = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "reference_scene_terrain")
    return _compile([function], path, dict(np=np))["reference_scene_terrain"]


def _terrain_builder():
    path = PROJECT_ROOT / "morph_mimic/envs/acp_env.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_terrain_cfg")
    factories = {node.id: NS for node in ast.walk(function)
                 if isinstance(node, ast.Name) and node.id.endswith("Cfg")}
    factories["sim_utils"] = NS(RigidBodyMaterialCfg=NS)
    return _compile([function], path, factories)["build_terrain_cfg"]


class SubCfg(NS):
    def copy(self):
        return copy(self)

    def to_dict(self):
        return {key: str(value) for key, value in vars(self).items()}


def _run_installed_generator_placement(terrain_function, sidecar, size, num_envs=1):
    """Run actual _get/_add methods and final __init__ recenter statements."""
    tree = ast.parse(ISAAC_GENERATOR.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TerrainGenerator")
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_get_terrain_mesh", "_add_sub_terrain"}]
    extracted = ast.ClassDef(name="GeneratorPlacement", bases=[], keywords=[], body=methods, decorator_list=[])
    scope = _compile([extracted], ISAAC_GENERATOR,
                     dict(np=np, torch=torch, trimesh=trimesh, os=os, dict_to_md5_hash=lambda value: "cache-disabled"))
    generator = scope["GeneratorPlacement"]()
    rows, cols = reference_scene_grid_shape(num_envs)
    generator.cfg = NS(seed=0, use_cache=False, cache_dir="/tmp/unused-reference-scene-cache",
                       size=size, num_rows=rows, num_cols=cols)
    generator.device = "cpu"
    generator.terrain_meshes = []
    generator.terrain_origins = np.zeros((rows, cols, 3))
    sub = SubCfg(function=terrain_function, sidecar_path=str(sidecar), size=size, flat_patch_sampling=None,
                 fingerprint=load_reference_scene(sidecar).fingerprint)
    for row in range(rows):
        for col in range(cols):
            mesh, origin = generator._get_terrain_mesh(0.0, sub)
            generator._add_sub_terrain(mesh, origin, row, col, sub)
    generator.terrain_mesh = trimesh.util.concatenate(generator.terrain_meshes)
    init = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    # The last assignment to local ``transform`` starts the generator-wide
    # recentering block. Execute those real operations, rather than duplicating
    # their arithmetic in a test implementation.
    starts = [index for index, node in enumerate(init.body) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "transform" for target in node.targets)]
    final_nodes = init.body[starts[-1]:starts[-1] + 4]
    _compile(final_nodes, ISAAC_GENERATOR, dict(self=generator, np=np))
    return generator


class TestSceneIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        scene_motion(cls.directory, CLIP)

    def dataset(self, directory=None):
        directory = self.directory if directory is None else directory
        return MotionDataset(RobotSpec.load("g1"), directory, clip_names=[CLIP])

    def cfg(self, **overrides):
        fields = dict(stage=1, ref_env_ratio=1., reference_scene_path=None, flat_reference=False,
                      terrain_size=(9., 6.), terrain_types=("flat",), ref_terrain_types=("flat",), num_terrain_levels=10)
        fields.update(overrides)
        return NS(**fields)

    @unittest.skipUnless(ISAAC_GENERATOR.is_file(), "installed IsaacLab terrain source is unavailable")
    def test_real_generator_recenter_keeps_motion_obstacle_relative_positions(self):
        motion = self.dataset()
        scene = motion.scene_bindings[CLIP]
        cfg = self.cfg()
        configure_reference_scene(cfg, motion, 1)
        placed = _run_installed_generator_placement(_terrain_function(), scene.sidecar_path, cfg.terrain_size)
        source = scene.combined_mesh()
        expected_vertices = source.vertices - [*scene.source_first_root_xy, 0.]
        np.testing.assert_allclose(placed.terrain_mesh.vertices, expected_vertices, atol=1e-8)
        np.testing.assert_array_equal(placed.terrain_mesh.faces, source.faces)
        np.testing.assert_allclose(placed.terrain_origins, 0., atol=1e-8)

        Env = _environment_class()
        env = Env.__new__(Env)
        env.cfg, env.motion_data, env.device, env.num_envs = cfg, motion, "cpu", 1
        env.terrain_origins = torch.tensor(placed.terrain_origins, dtype=torch.float32)
        env._init_terrain_mapping()
        np.testing.assert_array_equal(env.cell_offset.numpy(), 0.)
        env.is_ref_env = torch.ones(1, dtype=torch.bool)
        env.clip_id = torch.zeros(1, dtype=torch.long)
        for phase in (0., .3, .8):
            env.phase = torch.tensor([phase])
            reference = env._ref_sample()["root_pos"][0].numpy()
            raw_reference = motion.clips[CLIP].sample(env.phase)["root_pos"][0].numpy()
            # Compare every original obstacle vertex relative to the reference;
            # preserving this vector is the actual alignment invariant.
            np.testing.assert_allclose(placed.terrain_mesh.vertices - reference,
                                       source.vertices - raw_reference, atol=2e-7)

    def test_single_scene_stage_and_flat_replacement_gates(self):
        motion = self.dataset()
        for cfg, count, error in [
            (self.cfg(), 0, "positive integer"),
            (self.cfg(stage=3), 1, "Stage 1/2"),
            (self.cfg(ref_env_ratio=.5), 1, "reference environments"),
            (self.cfg(flat_reference=True), 1, "cannot replace"),
        ]:
            with self.subTest(config=vars(cfg), count=count), self.assertRaisesRegex(ValueError, error):
                configure_reference_scene(cfg, motion, count)
        unbound = NS(scene_bindings={}, clip_names=["flat"])
        with self.assertRaisesRegex(ValueError, "hashed motion scene"):
            configure_reference_scene(self.cfg(reference_scene_path="arbitrary.scene.json"), unbound, 1)
        with self.assertRaisesRegex(ValueError, "exactly one bound clip"):
            configure_reference_scene(self.cfg(), NS(scene_bindings=motion.scene_bindings, clip_names=[CLIP, "other"]), 1)

    def test_required_scene_is_not_silently_replaced_when_sidecar_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy2(self.directory / f"{CLIP}.npz", Path(directory) / f"{CLIP}.npz")
            with self.assertRaisesRegex(ValueError, "required scene sidecar is missing"):
                self.dataset(directory)

    def test_direct_environment_constructor_enforces_binding_before_simulation(self):
        motion = self.dataset()

        class BeforePhysics(Exception):
            pass

        class SimulatorBoundary:
            def __init__(self, *args, **kwargs):
                raise BeforePhysics()

        generated = []

        def terrain_builder(cfg):
            generated.append(cfg.reference_scene_path)
            return NS()

        Env = _environment_class(base=SimulatorBoundary, terrain_builder=terrain_builder)
        cfg = self.cfg(evaluation=False, scene=NS(num_envs=1), planner_scan_shape=(17, 11), tracker_scan_shape=(9, 7))
        with self.assertRaises(BeforePhysics):
            Env(cfg, motion.spec, motion)
        self.assertEqual(generated, [str(motion.scene_bindings[CLIP].sidecar_path)])
        self.assertEqual(cfg.terrain_types, ("reference_scene",))
        for overrides, error in [
            ({"terrain": NS()}, "preconfigured terrain importer"),
            ({"scene": NS(num_envs=0)}, "positive integer"),
            ({"stage": 3}, "Stage 1/2"),
        ]:
            cfg = self.cfg(evaluation=False, scene=NS(num_envs=1))
            for key, value in overrides.items():
                setattr(cfg, key, value)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, error):
                Env(cfg, motion.spec, motion)

    def test_terrain_builder_creates_replica_grid_without_changing_difficulty_levels(self):
        motion = self.dataset()
        builder = _terrain_builder()
        for count in (1, 5, 32, 512):
            with self.subTest(num_envs=count):
                cfg = self.cfg(scene=NS(num_envs=count))
                configure_reference_scene(cfg, motion, count)
                terrain = builder(cfg)
                generator = terrain.terrain_generator
                self.assertEqual((generator.num_rows, generator.num_cols), reference_scene_grid_shape(count))
                self.assertEqual(cfg.num_terrain_levels, 1)
                self.assertEqual(list(generator.sub_terrains), ["reference_scene"])
                self.assertEqual(generator.sub_terrains["reference_scene"].fingerprint,
                                 motion.scene_bindings[CLIP].fingerprint)
                for key, value in (("stage", 3), ("ref_env_ratio", .5), ("num_terrain_levels", 2),
                                   ("terrain_types", ("flat",)), ("ref_terrain_types", ("flat",))):
                    invalid = copy(cfg)
                    setattr(invalid, key, value)
                    with self.subTest(invalid=key), self.assertRaisesRegex(ValueError, "Stage 1/2"):
                        builder(invalid)

    @unittest.skipUnless(ISAAC_GENERATOR.is_file(), "installed IsaacLab terrain source is unavailable")
    def test_replicated_scene_tiles_keep_geometry_reference_and_contact_frames_aligned(self):
        motion = self.dataset()
        scene = motion.scene_bindings[CLIP]
        # The older geometry fixture predates contact labeling. Give its oracle
        # nonzero per-anatomy/per-frame targets so this check cannot pass merely
        # because all missing-label outputs happen to be zero.
        frames = motion.clips[CLIP].num_frames
        targets = torch.arange(27, dtype=torch.float32).reshape(1, 9, 3) / 30.
        targets = targets + torch.linspace(0., .1, frames)[:, None, None]
        flags = torch.ones(frames, 9)
        flags[:, ::2] = 0.
        motion.contact_targets[CLIP] = targets, flags
        motion.contact_target_frames[CLIP] = "reference_query_body_yaw_absolute_z"
        for count in (2, 5, 32):
            with self.subTest(num_envs=count):
                cfg = self.cfg()
                configure_reference_scene(cfg, motion, count)
                self.assertEqual(cfg.num_terrain_levels, 1)
                self.assertEqual(cfg.terrain_types, ("reference_scene",))
                self.assertEqual(cfg.reference_scene_fingerprint, scene.fingerprint)
                placed = _run_installed_generator_placement(
                    _terrain_function(), scene.sidecar_path, cfg.terrain_size, count)
                Env = _environment_class()
                env = Env.__new__(Env)
                env.cfg, env.motion_data, env.device, env.num_envs = cfg, motion, "cpu", count
                env.terrain_origins = torch.tensor(placed.terrain_origins, dtype=torch.float32)
                env._init_terrain_mapping()
                origins = env.env_origin()
                expected_origins = placed.terrain_origins.reshape(-1, 3)[:count]
                np.testing.assert_allclose(origins.numpy(), expected_origins, atol=2e-6)
                self.assertEqual(len(torch.unique(origins, dim=0)), count)
                np.testing.assert_array_equal(env.cell_offset.numpy(), 0.)
                env.is_ref_env = torch.ones(count, dtype=torch.bool)
                env.clip_id = torch.zeros(count, dtype=torch.long)
                env._contact_clip_body_idx = {}
                _, env.contact_motion_candidates = motion.spec.resolve_contact_bodies(motion.clips[CLIP].body_names)
                source = scene.combined_mesh()
                vertex_count = len(source.vertices)
                for phase in (0., .3, .8):
                    env.phase = torch.full((count,), phase)
                    sampled = env._ref_sample()
                    reference = sampled["root_pos"].numpy()
                    env.robot = NS(data=NS(root_pos_w=sampled["root_pos"], root_quat_w=sampled["root_quat"]))
                    raw_reference = motion.clips[CLIP].sample(env.phase)["root_pos"].numpy()
                    for cell in range(count):
                        vertices = placed.terrain_mesh.vertices[cell * vertex_count:(cell + 1) * vertex_count]
                        np.testing.assert_allclose(vertices - reference[cell], source.vertices - raw_reference[cell], atol=2e-5)
                    # Oracle targets are stored in the current reference yaw
                    # frame; horizontal tile translation must not alter them.
                    targets, flags = env._oracle_contacts(torch.arange(count))
                    torch.testing.assert_close(targets, targets[:1].expand_as(targets))
                    torch.testing.assert_close(flags, flags[:1].expand_as(flags))
                    contacts_world = env._ref_contact_pos(body_yaw=False) - origins[:, None, :]
                    contacts_body = env._ref_contact_pos(body_yaw=True)
                    torch.testing.assert_close(contacts_world, contacts_world[:1].expand_as(contacts_world), atol=2e-5, rtol=0.)
                    torch.testing.assert_close(contacts_body, contacts_body[:1].expand_as(contacts_body))
                # Every complete collision copy is separated by the configured
                # tile margin. Nearby robots cannot share the same obstacles.
                local_bounds = scene.rebased_mesh().bounds
                bounds = local_bounds[None, :, :2] + expected_origins[:, None, :2]
                for first in range(count):
                    for second in range(first + 1, count):
                        self.assertTrue(np.any(bounds[first, 1] < bounds[second, 0]) or
                                        np.any(bounds[second, 1] < bounds[first, 0]))

    def test_bound_scene_selects_local_robot_model_before_environment_creation(self):
        bound = self.dataset()
        model_requests = []

        def make_robot(spec, *, use_usd):
            model_requests.append(use_usd)
            return NS(local=not use_usd)

        def stage_cfg(count, *args):
            return self.cfg(sim=NS(device=None), episode_length_s=15., record_max_steps=500)

        fake_modules = {
            "morph_mimic.cfg.configs": NS(stage1_env_cfg=stage_cfg, stage2_env_cfg=stage_cfg, stage3_env_cfg=stage_cfg),
            "morph_mimic.envs.acp_env": NS(AcpEnv=lambda cfg, spec, motion: NS(cfg=cfg, motion=motion)),
            "morph_mimic.utils.robot_cfg": NS(make_robot_cfg=make_robot),
            "morph_mimic.utils.held_joints": NS(apply_held_joint_mode=lambda env: None),
        }
        with patch.dict(sys.modules, fake_modules):
            with patch.object(common, "MotionDataset", return_value=bound):
                env, _ = common.build_env(1, "g1", 1, device="cpu", require_contacts=False, use_usd=True)
                self.assertTrue(env.cfg.robot.local)
                self.assertEqual(env.cfg.terrain_types, ("reference_scene",))
            unbound = NS(scene_bindings={}, clip_names=["walk"], contact_flags={})
            with patch.object(common, "MotionDataset", return_value=unbound):
                env, _ = common.build_env(1, "g1", 1, device="cpu", require_contacts=False, use_usd=True)
                self.assertFalse(env.cfg.robot.local)
        self.assertEqual(model_requests, [False, True])

    def test_contact_labels_without_scene_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            shutil.copy2(self.directory / f"{CLIP}.npz", temporary / f"{CLIP}.npz")
            metadata = json.loads((self.directory / f"{CLIP}.scene.json").read_text())
            (temporary / f"{CLIP}.scene.json").write_text(json.dumps(metadata))
            original = self.dataset()
            clip = original.clips[CLIP]
            spec = original.spec
            names = []
            for group in spec.nine_body_groups():
                candidates = spec.contact_bodies[group]
                names.extend(candidates if len(candidates) == 1 else
                             [next(name for name in candidates if name.startswith(side)) for side in ("left", "right")])
            np.savez(temporary / f"{CLIP}_contacts.npz", fps=np.array([clip.fps]),
                     body_names=np.array(names), contact_flags=np.zeros((clip.num_frames, 9)),
                     contact_targets=np.zeros((clip.num_frames, 9, 3)),
                     contact_target_flags=np.zeros((clip.num_frames, 9)))
            with self.assertRaisesRegex(ValueError, "do not bind the current collision scene"):
                self.dataset(temporary)

    def test_checkpoint_rejects_same_motion_with_changed_ground_scene(self):
        motion = self.dataset()
        cfg = self.cfg()
        cfg.sim, cfg.decimation = NS(dt=.005), 4
        cfg.robot = NS(spawn=NS(usd_path="test://runtime"))
        env = NS(cfg=cfg, robot_spec=motion.spec, motion_data=motion, num_envs=1)
        expected = {"contract": cp.build_checkpoint_contract(env)}
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            shutil.copy2(self.directory / f"{CLIP}.npz", temporary / f"{CLIP}.npz")
            metadata = json.loads((self.directory / f"{CLIP}.scene.json").read_text())
            metadata["ground"]["top_z_m"] = .05
            (temporary / f"{CLIP}.scene.json").write_text(json.dumps(metadata))
            changed = self.dataset(temporary)
            self.assertNotEqual(motion.scene_bindings[CLIP].fingerprint, changed.scene_bindings[CLIP].fingerprint)
            env.motion_data = changed
            for mode in ("resume", "transfer", "evaluate"):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "physics"):
                    cp.validate_checkpoint_contract(expected, env, mode=mode)


class TestFilteredSceneContacts(unittest.TestCase):
    def test_scene_forces_use_only_filtered_obstacles_and_keep_anatomical_order(self):
        Env = _environment_class()
        env = Env.__new__(Env)
        env.cfg = NS(contact_force_threshold=1.)
        # The broad robot sensor includes large self-contact forces. Every
        # filtered anatomy sensor has its own reporter and opposing-body axis.
        env.contact_sensor_idx = torch.arange(9)
        env.contact_sensor = NS(data=NS(net_forces_w=torch.full((2, 9, 3), 99999.),
                                       net_forces_w_history=torch.full((2, 4, 9, 3), 99999.)))
        env.scene_contact_sensors = []
        expected, histories = [], []
        for anatomy in range(9):
            force = torch.zeros(2, 1, 2, 3)
            force[:, 0, 0] = torch.tensor([float(anatomy), -float(anatomy), .2])
            force[:, 0, 1] = torch.tensor([.5, .25, .3])
            # Instantaneous force remains below the vertical threshold; body
            # three had a >1 N filtered vertical contact in an earlier sample.
            history = force.unsqueeze(1).repeat(1, 4, 1, 1, 1)
            if anatomy == 3:
                history[:, 2, 0, 1, 2] = 2.
            env.scene_contact_sensors.append(NS(data=NS(force_matrix_w=force, force_matrix_w_history=history,
                                                       net_forces_w=torch.full((2, 1, 3), 99999.))))
            expected.append(force.sum(dim=-2))
            histories.append(history.sum(dim=-2))
        filtered = env._contact_forces_w()
        filtered_history = env._contact_force_history()
        self.assertEqual(tuple(filtered.shape), (2, 9, 3))
        self.assertEqual(tuple(filtered_history.shape), (2, 4, 9, 3))
        torch.testing.assert_close(filtered, torch.cat(expected, dim=1))
        torch.testing.assert_close(filtered_history, torch.cat(histories, dim=2))
        measured = env._contact_measured()
        expected_flags = torch.zeros(2, 9)
        expected_flags[:, 3] = 1.
        torch.testing.assert_close(measured, expected_flags)
        # If unfiltered net force leaked into either path all nine columns
        # would be active; changing it must have no effect on scene results.
        env.contact_sensor.data.net_forces_w.fill_(-7e8)
        env.contact_sensor.data.net_forces_w_history.fill_(-7e8)
        torch.testing.assert_close(env._contact_forces_w(), filtered)
        torch.testing.assert_close(env._contact_force_history(), filtered_history)
        torch.testing.assert_close(env._contact_measured(), expected_flags)

    def test_unbound_legacy_sensor_path_keeps_explicit_anatomical_permutation(self):
        Env = _environment_class()
        env = Env.__new__(Env)
        env.scene_contact_sensors = []
        env.contact_sensor_idx = torch.tensor([8, 2, 7, 1, 6, 0, 5, 3, 4])
        net = torch.arange(2 * 12 * 3, dtype=torch.float32).reshape(2, 12, 3)
        history = torch.arange(2 * 4 * 12 * 3, dtype=torch.float32).reshape(2, 4, 12, 3)
        env.contact_sensor = NS(data=NS(net_forces_w=net, net_forces_w_history=history))
        torch.testing.assert_close(env._contact_forces_w(), net[:, env.contact_sensor_idx])
        torch.testing.assert_close(env._contact_force_history(), history[:, :, env.contact_sensor_idx])


# Scene reference diagnostic.

@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is unavailable")
class TestSceneReferenceDiagnostic(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("trimesh"), "trimesh is unavailable")
    def test_disconnected_collision_mesh_keeps_gap_and_rejects_concave_surface(self):
        """Mesh collision must not fill the space between separate boxes."""
        import mujoco
        import trimesh

        first = trimesh.creation.box(extents=[1., 1., 1.])
        second = first.copy().apply_translation([3., 0., 0.])
        component = SimpleNamespace(name="two_boxes", kind="mesh",
                                    mesh=trimesh.util.concatenate([first, second]))
        ground = SimpleNamespace(size_m=[10., 10.], thickness_m=1.,
                                 center_xy_m=[0., 0.], top_z_m=-2.)
        scene = SimpleNamespace(components=[component], ground=ground)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.xml"
            model_path.write_text('<mujoco><worldbody><body><freejoint/>'
                                  '<geom type="sphere" size=".1" mass="1"/>'
                                  '</body></worldbody></mujoco>')
            model, geom_ids = build_model(model_path, scene, .005)
            self.assertEqual(len(geom_ids), 3)  # two convex pieces and ground
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            ray_geom = np.zeros(1, dtype=np.int32)
            # A vertical ray through the 2 m gap reaches the ground, not a hull.
            distance = mujoco.mj_ray(model, data, np.array([1.5, 0., 3.]),
                                     np.array([0., 0., -1.]), None, 1, -1, ray_geom)
            self.assertAlmostEqual(distance, 5., places=6)
            component.mesh = trimesh.creation.torus(major_radius=1., minor_radius=.2)
            with self.assertRaisesRegex(ValueError, "concave"):
                build_model(model_path, scene, .005)

    def test_free_root_falls_despite_stationary_reference(self):
        """A complete replay must fail tracking rather than teleport its root."""
        import mujoco

        model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.005" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="10 10 .1" pos="0 0 -50"/>
            <body name="pelvis"><freejoint/><geom type="sphere" size=".1" mass="1"/>
              <body name="limb" pos="0 0 -.2"><joint name="hinge"/>
                <geom type="sphere" size=".05" mass=".1"/></body>
            </body>
          </worldbody>
        </mujoco>""")
        spec = SimpleNamespace(base_link="pelvis", joint_names=["hinge"],
                               controlled_joint_names=["hinge"], default_pose=[0.],
                               stiffness=[10.], damping=[1.], effort_limits=[5.],
                               armature=[.01], control_dt=.02)
        frames = 51
        position = np.tile([0., 0., 1.], (frames, 1, 1))
        arrays = dict(fps=np.array([50.]), joint_names=np.array(["hinge"]),
                      joint_pos=np.zeros((frames, 1)), joint_vel=np.zeros((frames, 1)),
                      body_names=np.array(["pelvis"]), body_pos_w=position,
                      body_quat_w=np.tile([1., 0., 0., 0.], (frames, 1, 1)),
                      body_lin_vel_w=np.zeros((frames, 1, 3)),
                      body_ang_vel_w=np.zeros((frames, 1, 3)))
        report, trajectory = simulate(model, {model.geom("floor").id}, spec, arrays)
        self.assertTrue(report["complete"])
        self.assertEqual(report["recorded_num_frames"], frames)
        self.assertFalse(report["accepted"])
        self.assertGreater(report["root_position_max_error_m"], 4.)
        self.assertLess(trajectory["root_pos_w"][-1, 2], -3.)
        self.assertEqual(report["applied_root_force_max"], 0.)
        self.assertEqual(report["root_initializations"], 1)
        self.assertFalse(report["training_contact_labels_generated"])

    def test_contact_force_is_world_vertical_and_self_contacts_are_excluded(self):
        import mujoco

        model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep=".002"/><worldbody>
          <geom name="floor" type="plane" size="10 10 .1"/>
          <body name="robot" pos="0 0 .099"><freejoint/>
            <geom type="sphere" size=".1" mass="1"/></body>
          <body name="other_robot_link" pos="2 0 10"><freejoint/>
            <geom type="sphere" size=".1" mass="1"/></body>
          <body name="another_robot_link" pos="2.1 0 10"><freejoint/>
            <geom type="sphere" size=".1" mass="1"/></body>
        </worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertGreaterEqual(data.ncon, 2)  # includes a robot-robot collision
        forces, penetration, count = scene_contacts(
            model, data, {model.geom("floor").id},
            [model.body("robot").id, model.body("other_robot_link").id],
        )
        self.assertEqual(count, 1)
        self.assertGreater(forces[0, 2], 1.)
        np.testing.assert_allclose(forces[0, :2], 0., atol=1e-10)
        np.testing.assert_allclose(forces[1], 0., atol=1e-10)
        self.assertAlmostEqual(penetration, .001, places=6)


# Scene retarget.

@unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("scipy"), "MuJoCo and scipy required")
class TestSceneRetarget(unittest.TestCase):
    def model(self):
        spec = RobotSpec.load("g1")
        transform = np.eye(4)
        transform[:3, 3] = [.45, 0, .2]
        scene = SimpleNamespace(
            components=[SimpleNamespace(kind="box", transform=transform, size_m=[.4, .6, .4])],
            ground=SimpleNamespace(size_m=[10, 10], thickness_m=1, top_z_m=0, center_xy_m=[0, 0]),
        )
        model, scene_ids = build_model(PROJECT_ROOT / spec.runtime_assets["mjcf"], scene, spec.physics_dt)
        _, bodies = _body_columns(model, spec)
        return SceneFit(model, scene_ids, spec, bodies), spec

    def test_state_keeps_held_nominal_and_root_rotation_unit(self):
        fit, spec = self.model()
        controlled = np.asarray(spec.default_pose)[fit.controlled]
        variables = np.r_[controlled, [.02, -.01, .015], [.04, -.02, .03]]
        state = fit.state(variables, np.array([1, 2, .9]), np.array([1., 0, 0, 0]))
        np.testing.assert_allclose(state[:3], [1.02, 1.99, .915])
        self.assertAlmostEqual(float(np.linalg.norm(state[3:7])), 1)
        np.testing.assert_array_equal(state[fit.qids[fit.held]], np.asarray(spec.default_pose)[fit.held])
        np.testing.assert_array_equal(state[fit.qids[fit.controlled]], controlled)

    def test_aabb_pruning_keeps_every_pair_near_collision(self):
        import mujoco

        fit, spec = self.model()
        for height in (.7, .9, 1.5):
            variables = np.r_[np.asarray(spec.default_pose)[fit.controlled], np.zeros(6)]
            fit.data.qpos[:] = fit.state(variables, np.array([0, 0, height]), np.array([1., 0, 0, 0]))
            mujoco.mj_kinematics(fit.model, fit.data)
            pruned, _ = fit.distances(np.full(9, -1))
            brute = np.array([min(mujoco.mj_geomDistance(fit.model, fit.data, int(robot), int(scene), 5, None)
                                 for scene in fit.scene_ids) for robot in fit.robot_ids])
            close = brute <= .02
            np.testing.assert_allclose(pruned[close], brute[close], atol=1e-12)
            self.assertTrue((pruned[~close] >= .02 - 1e-12).all())
            self_pruned = fit.self_distances()
            self_brute = np.array([mujoco.mj_geomDistance(fit.model, fit.data, int(first), int(second), 5, None)
                                   for first, second in fit.self_pairs])
            close = self_brute <= .02
            np.testing.assert_allclose(self_pruned[close], self_brute[close], atol=1e-12)
            self.assertTrue((self_pruned[~close] >= .02 - 1e-12).all())

    def test_fit_preserves_input_and_respects_position_root_and_held_bounds(self):
        import mujoco

        fit, spec = self.model()
        root = np.array([0, 0, .9])
        variables = np.r_[np.asarray(spec.default_pose)[fit.controlled], np.zeros(6)]
        raw = np.tile(fit.state(variables, root, np.array([1., 0, 0, 0])), (2, 1))
        raw[:, fit.qids[spec.joint_names.index("waist_pitch_joint")]] = .2
        original = raw.copy()
        targets = []
        for state in raw:
            fit.data.qpos[:] = state
            mujoco.mj_kinematics(fit.model, fit.data)
            targets.append(fit.data.xpos[fit.body_ids].copy())
        fitted, _ = fit.optimize(raw, np.asarray(targets), np.full((2, 9), -1), 50, max_nfev=3)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(fitted[:, fit.qids[fit.held]], np.broadcast_to(np.asarray(spec.default_pose)[fit.held], (2, len(fit.held))))
        q = fitted[:, fit.qids]
        self.assertTrue((q >= np.asarray(spec.pos_limits_lower) - 1e-9).all())
        self.assertTrue((q <= np.asarray(spec.pos_limits_upper) + 1e-9).all())
        self.assertLessEqual(float(np.abs(fitted[:, :3] - root).max()), .03 + 1e-9)


# Self collision.

@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo required for collision filtering")
class TestEffectiveSelfCollision(unittest.TestCase):
    def model(self, exclude=False, masked=False):
        import mujoco

        contact = '<contact><exclude body1="left" body2="right"/></contact>' if exclude else ''
        mask = 'contype="0" conaffinity="0"' if masked else ''
        return mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <geom name="scene" type="sphere" size="2"/>
          <body name="root"><freejoint/>
            <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
            <body name="left" pos="-.04 0 0"><joint axis="0 0 1"/><geom name="left_geom" type="sphere" size=".1" mass="1"/></body>
            <body name="right" pos=".04 0 0"><joint axis="0 0 1"/><geom name="right_geom" type="sphere" size=".1" mass="1" {mask}/></body>
          </body></worldbody>{contact}</mujoco>''')

    def test_actual_collision_gate_and_pair_enumeration_agree(self):
        model = self.model()
        scene = {model.geom("scene").id}
        report = sample_self_contacts(model, model.qpos0[None, :], scene)
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(report["maximum_penetration_m"], .12)
        self.assertEqual(set(report["worst_contact"]["body_names"]), {"left", "right"})
        pairs = {tuple(pair) for pair in effective_robot_geom_pairs(model, scene)}
        expected = tuple(sorted((model.geom("left_geom").id, model.geom("right_geom").id)))
        self.assertIn(expected, pairs)

    def test_explicit_excludes_and_masks_are_preserved(self):
        for exclude, masked in ((True, False), (False, True)):
            with self.subTest(exclude=exclude, masked=masked):
                model = self.model(exclude, masked)
                scene = {model.geom("scene").id}
                before = model.geom_contype.copy(), model.geom_conaffinity.copy(), model.exclude_signature.copy()
                report = sample_self_contacts(model, model.qpos0[None, :], scene)
                self.assertTrue(report["accepted"])
                self.assertEqual(report["maximum_penetration_m"], 0)
                self.assertEqual(len(effective_robot_geom_pairs(model, scene)), 0)
                np.testing.assert_array_equal(model.geom_contype, before[0])
                np.testing.assert_array_equal(model.geom_conaffinity, before[1])
                np.testing.assert_array_equal(model.exclude_signature, before[2])

    def test_disabling_collision_cannot_create_a_passing_audit(self):
        import mujoco

        model = self.model()
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        with self.assertRaisesRegex(ValueError, "collision detection disabled"):
            sample_self_contacts(model, model.qpos0[None, :], {model.geom("scene").id})
        with self.assertRaisesRegex(ValueError, "collision detection to remain enabled"):
            effective_robot_geom_pairs(model, {model.geom("scene").id})


# Settle reference.

ROOT = Path(__file__).resolve().parents[1]


class SettleReferenceTests(unittest.TestCase):
    def test_hermite_preserves_start_derivative_and_stationary_endpoint(self):
        start,target,velocity = np.array([1.,2.,3.]),np.array([2.,1.,4.]),np.array([.1,.2,.3])
        pos,vel = _hermite(start,velocity,target,np.array([0.,1.]),1.)
        np.testing.assert_allclose(pos,[start,target])
        np.testing.assert_allclose(vel,[velocity,np.zeros(3)])

    def test_SO3_world_velocity_matches_finite_quaternion_derivative(self):
        phi,dphi = np.array([[.3,-.2,.4]]),np.array([[.2,.1,-.3]])
        epsilon = 1e-6
        plus,minus = _quat_exp(phi+epsilon*dphi)[0],_quat_exp(phi-epsilon*dphi)[0]
        relative = _multiply(plus,minus*np.array([1.,-1.,-1.,-1.]))
        finite = _rotation_vector(relative)/(2*epsilon)
        np.testing.assert_allclose(_world_angular_velocity(phi,dphi)[0],finite,atol=1e-9)

    def test_append_retains_every_source_array_prefix_and_rebinds_scene(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("CPU reference FK integration needs MuJoCo")
        motion = next(p for p in (ROOT/"datasets/g1_box_climb_seed_v2").glob("*.npz") if not p.stem.endswith("_contacts"))
        with np.load(motion,allow_pickle=False) as archive:
            source = {key:archive[key].copy() for key in archive.files}
        target = dict(joint_names=source["joint_names"].tolist(),joint_pos_rad=source["joint_pos"][-1].astype(float).tolist(),
            root_pos_world_m=source["root_pos"][-1].astype(float).tolist(),
            root_quat_wxyz=source["root_rot_wxyz"][-1].astype(float).tolist())
        payload = json.dumps(target,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        solution = dict(accepted=True,robot="g1",source_motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),
            target=target,target_sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            target_path = directory/"target.json"
            target_path.write_text(json.dumps(solution))
            output,metadata = append_settle_reference(motion,target_path,directory/"derived")
            self.assertTrue(metadata["source_prefix_preserved"])
            self.assertEqual(metadata["num_frames"],551)
            with np.load(output,allow_pickle=False) as derived:
                for key,value in source.items():
                    result = derived[key][:len(value)] if value.ndim and value.shape[0] == 501 else derived[key]
                    self.assertEqual(result.tobytes(),value.tobytes(),key)
                self.assertTrue(np.all(derived["joint_vel"][-1] == 0))
                self.assertTrue(np.all(derived["body_lin_vel_w"][-1] == 0))
                self.assertTrue(np.all(derived["body_ang_vel_w"][-1] == 0))
            self.assertEqual(load_reference_scene(output.with_suffix(".scene.json"),output).scene_urdf_sha256,
                load_reference_scene(motion.with_suffix(".scene.json"),motion).scene_urdf_sha256)
            with self.assertRaises(FileExistsError):
                append_settle_reference(motion,target_path,directory/"derived")


# Standing reference.

class TestStandingReference(unittest.TestCase):
    def test_invalid_parameters_refuse_output_without_mujoco(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "invalid"
            for kwargs in (dict(duration_s=-1), dict(duration_s=0.03), dict(clearance_m=0.02)):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    prepare_standing_reference(target, **kwargs)
            self.assertFalse(target.exists())

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "optional CPU MuJoCo required")
    def test_static_fk_is_complete_and_continuous_support_has_no_future_onset(self):
        import mujoco

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "standing"
            motion, meta = prepare_standing_reference(target)
            self.assertEqual(meta["num_frames"], 501)
            self.assertEqual(meta["foot_collision_count"], 14)
            self.assertAlmostEqual(meta["root_height_above_ground_m"], 0.7590215776517654, places=10)
            self.assertFalse(meta["physics_validated"])
            self.assertFalse(meta["forces_used"])
            spec = RobotSpec.load("g1")
            with np.load(motion) as data:
                self.assertEqual(data["joint_pos"].shape, (501, 29))
                for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                    self.assertFalse(data[key].any())
                model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / meta["source_mjcf"]))
                state = mujoco.MjData(model)
                state.qpos[:7] = [0, 0, meta["root_height_above_ground_m"], 1, 0, 0, 0]
                for name, value in zip(data["joint_names"], data["joint_pos"][0]):
                    state.qpos[model.jnt_qposadr[model.joint(str(name)).id]] = value
                mujoco.mj_forward(model, state)
                ids = [model.body(str(name)).id for name in data["body_names"]]
                np.testing.assert_allclose(data["body_pos_w"][0], state.xpos[ids], rtol=0, atol=1e-7)
            labels = motion.with_name(motion.stem + "_contacts.npz")
            with np.load(labels) as data:
                np.testing.assert_array_equal(data["contact_flags"].sum(axis=0), [0, 0, 0, 0, 0, 0, 501, 501, 0])
                self.assertFalse(data["contact_targets"].any())
                self.assertFalse(data["contact_target_flags"].any())
                self.assertEqual(str(data["source_sha256"]), hashlib.sha256(motion.read_bytes()).hexdigest())
                self.assertFalse(bool(data["physics_validated"]))
            dataset = MotionDataset(spec, target, clip_names=[motion.stem])
            self.assertEqual(dataset.contact_target_frames[motion.stem], "reference_query_body_yaw_absolute_z")
            before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()}
            with self.assertRaises(FileExistsError):
                prepare_standing_reference(target)
            self.assertEqual(before, {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()})


# Omniretarget import.

class TestOmniTimebase(unittest.TestCase):
    def test_slerp_uses_short_arc_and_floor_grid_without_time_scaling(self):
        q = np.zeros((3, 36))
        angles = np.array([0, 0.3, 0.6])
        q[:, 0], q[:, 3] = np.cos(angles / 2), np.sin(angles / 2)
        q[1, :4] *= -1  # A sign change must not change the rotation path.
        q[:, 4], q[:, 7] = [0, 1, 2], [0, 0.2, 0.4]
        result, times = resample_qpos(q, 30, 50)
        np.testing.assert_allclose(times, [0, 0.02, 0.04, 0.06], atol=1e-12)
        np.testing.assert_allclose(result[:, 4], times * 30, atol=1e-12)
        np.testing.assert_allclose(result[:, 7], times * 6, atol=1e-12)
        expected = np.stack([np.cos(times * 9 / 2), np.zeros(4), np.zeros(4), np.sin(times * 9 / 2)], axis=-1)
        np.testing.assert_allclose(np.abs(np.sum(result[:, :4] * expected, axis=-1)), 1, atol=1e-12)
        self.assertLess(2 / 30 - times[-1], 1 / 50)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is required for FK conversion")
class TestOmniConversion(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / "source"
        spec = RobotSpec.load("g1")
        # Reverse names to exercise named mapping rather than positional reuse
        # of the local MJCF's compiled joint order.
        names = list(reversed(spec.joint_names))
        robot = root / "models/g1/g1_29dof_spherehand.urdf"
        robot.parent.mkdir(parents=True)
        robot.write_text('<robot name="g1">' + ''.join(f'<joint name="{name}" type="revolute"/>' for name in names) + '</robot>')
        terrain = root / "models/terrain/climb_18/multi_boxes_z_scale_1.0.urdf"
        terrain.parent.mkdir(parents=True)
        (terrain.parent / "box.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        terrain.write_text('<robot name="terrain"><link name="box"><collision><geometry><mesh filename="box.obj"/></geometry></collision></link></robot>')
        times = np.arange(5) / 30
        q = np.zeros((5, 36))
        q[:, 0], q[:, 3] = np.cos(times / 2), np.sin(times / 2)
        q[1::2, :4] *= -1
        q[:, 4], q[:, 5], q[:, 6] = times, times * -0.5, 0.9
        for column, name in enumerate(names):
            q[:, 7 + column] = spec.default_pose[spec.joint_names.index(name)]
            q[:, 7 + column] += (0.1 if name in spec.held_joint_names else 0) + times * 0.2
        source = root / "climb_18_z_scale_1.0.npz"
        np.savez(source, qpos=q, fps=np.array(30.0))
        return root, source, q, spec

    def test_conversion_preserves_time_named_controlled_joints_and_scene_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, q, spec = self.fixture(directory)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            output, metadata = convert_clip(source, root, Path(directory) / "output", spec)
            self.assertEqual(metadata["num_frames"], 7)
            self.assertEqual(metadata["target_fps"], 50)
            self.assertAlmostEqual(metadata["duration_s"], 0.12)
            self.assertAlmostEqual(metadata["discarded_tail_s"], 4 / 30 - 0.12)
            with np.load(output, allow_pickle=False) as result:
                validate_motion_arrays(result, spec)
                self.assertEqual(result["category"].item(), "box_climb")
                self.assertTrue(result["requires_scene"].item())
                names, times = result["joint_names"].tolist(), np.arange(7) / 50
                np.testing.assert_allclose(result["root_pos"][:, 0], times, atol=1e-7)
                for name in spec.controlled_joint_names:
                    column = names.index(name)
                    expected = spec.default_pose[spec.joint_names.index(name)] + times * 0.2
                    np.testing.assert_allclose(result["joint_pos"][:, column], expected, atol=1e-7)
                held = [names.index(name) for name in spec.held_joint_names]
                np.testing.assert_array_equal(result["joint_vel"][:, held], 0)
                base = result["body_names"].tolist().index(spec.base_link)
                np.testing.assert_allclose(result["body_lin_vel_w"][:, base], np.tile([1, -0.5, 0], (7, 1)), atol=1e-6)
                np.testing.assert_allclose(result["body_ang_vel_w"][:, base], np.tile([0, 0, 1], (7, 1)), atol=1e-6)
            raw_path = output.parent / "raw_reference" / output.name
            with np.load(raw_path, allow_pickle=False) as raw:
                with self.assertRaisesRegex(ValueError, "held joints move"):
                    validate_motion_arrays(raw, spec)
            scene = json.loads(output.with_suffix(".scene.json").read_text())
            self.assertEqual(scene["motion_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(scene["source_motion"]["sha256"], source_hash)
            self.assertEqual(scene["motion_file"], output.name)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), source_hash)
            self.assertGreater(metadata["held_projection_body_position_max_delta_m"]["left_wrist_yaw_link"], 1e-3)
            self.assertFalse(metadata["physical_tracking_validated"])
            self.assertFalse(metadata["scene_contacts_preserved"])
            self.assertFalse(output.with_name(output.stem + "_contacts.npz").exists())
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                convert_clip(source, root, output.parent, spec)
            self.assertEqual(output.read_bytes(), original)

    def test_exported_body_origin_velocities_match_integrated_fk(self):
        import mujoco

        with tempfile.TemporaryDirectory() as directory:
            root, source, _, spec = self.fixture(directory)
            output, _ = convert_clip(source, root, Path(directory) / "output", spec)
            model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / spec.runtime_assets["mjcf"]))
            data = mujoco.MjData(model)
            with np.load(output, allow_pickle=False) as result:
                joints = result["joint_names"].tolist()
                qids = [model.joint(name).qposadr[0] for name in joints]
                vids = [model.joint(name).dofadr[0] for name in joints]
                bids = [model.body(name).id for name in result["body_names"].tolist()]
                base = result["body_names"].tolist().index(spec.base_link)
                frame = 3
                data.qpos[:3], data.qpos[3:7] = result["root_pos"][frame], result["root_rot_wxyz"][frame]
                data.qpos[qids] = result["joint_pos"][frame]
                mujoco.mj_forward(model, data)
                data.qvel[:3] = result["body_lin_vel_w"][frame, base]
                rotation = data.xmat[model.body(spec.base_link).id].reshape(3, 3)
                data.qvel[3:6] = rotation.T @ result["body_ang_vel_w"][frame, base]
                data.qvel[vids] = result["joint_vel"][frame]
                mujoco.mj_forward(model, data)
                before = data.xpos[bids].copy()
                before_rotation = data.xmat[bids].reshape(-1, 3, 3).copy()
                step = 1e-6
                mujoco.mj_integratePos(model, data.qpos, data.qvel, step)
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose((data.xpos[bids] - before) / step, result["body_lin_vel_w"][frame], atol=1e-5)
                delta = data.xmat[bids].reshape(-1, 3, 3) @ np.swapaxes(before_rotation, -1, -2)
                omega = np.stack((delta[:, 2, 1] - delta[:, 1, 2], delta[:, 0, 2] - delta[:, 2, 0], delta[:, 1, 0] - delta[:, 0, 1]), axis=-1) / (2 * step)
                np.testing.assert_allclose(omega, result["body_ang_vel_w"][frame], atol=1e-5)

    def test_object_augmented_qpos_and_invalid_quaternion_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, q, spec = self.fixture(directory)
            np.savez(source, qpos=np.pad(q, ((0, 0), (0, 7))), fps=30.0)
            with self.assertRaisesRegex(ValueError, r"\[T,36\]"):
                convert_clip(source, root, Path(directory) / "output", spec)
            q[:, :4] *= 2
            np.savez(source, qpos=q, fps=30.0)
            with self.assertRaisesRegex(ValueError, "unit WXYZ"):
                convert_clip(source, root, Path(directory) / "output", spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
