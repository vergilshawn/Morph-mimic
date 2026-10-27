"""Check held-DOF geometry, named mappings, and independent velocity differences."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from morph_mimic.scripts.project_reference import project_clip
from morph_mimic.utils.motion import MotionDataset, validate_contact_body_order, validate_motion_arrays
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


RAW_REFERENCE = PROJECT_ROOT / "datasets/g1_smoke/walk1_subject1.npz"
REFERENCE_UNAVAILABLE = "Optional local G1 smoke reference is not included in the source repository"


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

    @unittest.skipUnless(RAW_REFERENCE.is_file(), REFERENCE_UNAVAILABLE)
    def test_raw_g1_reference_is_rejected_before_controlled_joint_selection(self):
        spec = RobotSpec.load("g1")
        with self.assertRaisesRegex(ValueError, "held joints move.*Generate consistent FK"):
            MotionDataset(spec, PROJECT_ROOT / "datasets/g1_smoke", clip_names=["walk1_subject1"])

    @unittest.skipUnless(RAW_REFERENCE.is_file(), REFERENCE_UNAVAILABLE)
    def test_complete_names_and_shapes_are_validated_even_when_hold_check_is_disabled(self):
        spec = RobotSpec.load("g1")
        with np.load(PROJECT_ROOT / "datasets/g1_smoke/walk1_subject1.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        validate_motion_arrays(arrays, spec, require_nominal_held=False)
        arrays["joint_names"][1] = arrays["joint_names"][0]
        with self.assertRaisesRegex(ValueError, "duplicate joint_names"):
            validate_motion_arrays(arrays, spec, require_nominal_held=False)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is needed for reference projection")
@unittest.skipUnless(RAW_REFERENCE.is_file(), REFERENCE_UNAVAILABLE)
class TestReferenceProjection(unittest.TestCase):
    def source(self, directory):
        with np.load(PROJECT_ROOT / "datasets/g1_smoke/walk1_subject1.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            arrays[key] = arrays[key][10:15].copy()
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


if __name__ == "__main__":
    unittest.main()
