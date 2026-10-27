"""Check source-layout conversion, time preservation, and body-origin FK."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from morph_mimic.scripts.import_omniretarget import convert_clip, resample_qpos
from morph_mimic.utils.motion import validate_motion_arrays
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


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
    unittest.main()
