"""Exercise the actual oracle method with moving reference and robot frames."""

import ast
import hashlib
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from morph_mimic.utils.motion import MotionDataset
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec
from morph_mimic.utils.transforms import to_body_yaw_frame, yaw_from_quat, yaw_quat


def _oracle_method():
    path = Path(__file__).resolve().parents[1] / "morph_mimic/envs/acp_env.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "AcpEnv")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_oracle_contacts")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             method], type_ignores=[])
    scope = dict(torch=torch, to_body_yaw_frame=to_body_yaw_frame, yaw_from_quat=yaw_from_quat)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["_oracle_contacts"]


class TestOracleContactFrame(unittest.TestCase):
    def fixture(self):
        roots = torch.tensor([[3., -2., 1.], [3.2, -1.8, 1.2], [3.5, -1.4, 1.4]])
        yaw = torch.tensor([0., torch.pi / 3, torch.pi / 2])
        clip = NS(root_pos=roots, root_quat=yaw_quat(yaw))
        anchors = torch.tensor([3.8, -1.3, .6])[None, :].repeat(9, 1)
        anchors[:, 0] += torch.arange(9) * .015
        # One fixed source-world contact anchor is encoded using each discrete
        # query frame. The physical target must stay fixed between frames too.
        labels = to_body_yaw_frame(anchors[None].repeat(3, 1, 1), roots[:, None], yaw[:, None])
        flags = torch.ones(3, 9)
        flags[:, 1] = 0.
        labels[:, 1] = 1e5
        origins = torch.tensor([[0., 0., 0.], [20., -30., .2], [-50., 80., 0.], [100., 50., .5]])
        offsets = torch.tensor([[.1, -.1, 0.], [.2, 0., 0.], [0., .3, 0.], [-.2, -.2, 0.]])
        actual_root = origins + torch.tensor([.4, -.2, 9.])
        actual_yaw = torch.tensor([0., torch.pi / 2, -torch.pi / 2, torch.pi])
        motion = NS(clip_names=["reference"], clips={"reference": clip},
                    contact_targets={"reference": (labels, flags)},
                    contact_target_frames={"reference": "reference_query_body_yaw_absolute_z"})
        env = NS(device="cpu", num_envs=4, motion_data=motion, clip_id=torch.zeros(4, dtype=torch.long),
                 phase=torch.tensor([.25, .5, .75, 1.]), cell_offset=offsets,
                 env_origin=lambda: origins, robot=NS(data=NS(root_pos_w=actual_root, root_quat_w=yaw_quat(actual_yaw))))
        expected_world = anchors[None] - torch.tensor([*roots[0, :2], 0.])[None, None] + origins[:, None] + offsets[:, None]
        expected = to_body_yaw_frame(expected_world, actual_root[:, None], actual_yaw[:, None])
        expected[:, 1] = 0.
        return env, expected, flags

    def test_fixed_world_anchor_is_reexpressed_in_actual_robot_yaw_and_keeps_absolute_height(self):
        env, expected, flags = self.fixture()
        targets, actual_flags = _oracle_method()(env, torch.arange(4))
        torch.testing.assert_close(targets, expected, atol=1e-5, rtol=0.)
        torch.testing.assert_close(actual_flags, flags[0].expand(4, -1))
        torch.testing.assert_close(targets[:, 0, 2], torch.tensor([.6, .8, .6, 1.1]))
        self.assertTrue(torch.equal(targets[:, 1], torch.zeros(4, 3)))

    def test_label_uses_its_discrete_query_pose_and_does_not_slide_at_fractional_phase(self):
        env, expected, _ = self.fixture()
        method = _oracle_method()
        for phase in (.001, .249, .499, .501, .749, .999):
            with self.subTest(phase=phase):
                env.phase.fill_(phase)
                actual, _ = method(env, torch.arange(4))
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.)

    def test_subset_reset_order_and_multiple_clip_frames_do_not_mix_environments(self):
        env, expected, _ = self.fixture()
        legacy_targets = torch.full((3, 9, 3), .123)
        legacy_flags = torch.ones(3, 9)
        env.motion_data.clip_names.append("legacy")
        env.motion_data.contact_targets["legacy"] = legacy_targets, legacy_flags
        env.motion_data.contact_target_frames["legacy"] = "legacy_query_body_yaw_absolute_z_unverified"
        env.clip_id[2] = 1
        ids = torch.tensor([3, 2, 0])
        targets, flags = _oracle_method()(env, ids)
        torch.testing.assert_close(targets[[0, 2]], expected[[3, 0]], atol=1e-5, rtol=0.)
        torch.testing.assert_close(targets[1], legacy_targets[0])
        torch.testing.assert_close(flags[1], legacy_flags[0])

    def test_unidentified_legacy_frame_is_preserved_and_missing_labels_are_zero(self):
        env, _, _ = self.fixture()
        del env.motion_data.contact_target_frames
        labels, _ = env.motion_data.contact_targets["reference"]
        targets, _ = _oracle_method()(env, torch.tensor([0, 3]))
        # Unknown legacy files lack their generating query poses. Preserve their
        # old behavior instead of pretending reference inversion is exact.
        torch.testing.assert_close(targets[0], labels[0])
        torch.testing.assert_close(targets[1], labels[2])
        env.motion_data.contact_targets.clear()
        targets, flags = _oracle_method()(env, torch.tensor([3, 0]))
        self.assertTrue(torch.equal(targets, torch.zeros(2, 9, 3)))
        self.assertTrue(torch.equal(flags, torch.zeros(2, 9)))

    def test_unknown_bound_scene_frame_is_refused_instead_of_guessed(self):
        env, _, _ = self.fixture()
        env.motion_data.scene_bindings = {"reference": object()}
        env.motion_data.contact_target_frames.clear()
        with self.assertRaisesRegex(ValueError, "known reference-query"):
            _oracle_method()(env, torch.tensor([0]))


SEED = PROJECT_ROOT / "datasets/g1_box_climb_seed_v2"
CLIP = "climb_03_z_scale_1.0_scene_fit_0000_0501"


@unittest.skipUnless((SEED / f"{CLIP}_contacts.npz").is_file(), "box reference label fixture is absent")
class TestOracleContactFrameMetadata(unittest.TestCase):
    def test_existing_geometry_labels_are_recognized_without_modifying_their_bytes(self):
        path = SEED / f"{CLIP}_contacts.npz"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        motion = MotionDataset(RobotSpec.load("g1"), SEED, clip_names=[CLIP])
        self.assertEqual(motion.contact_target_frames[CLIP], "reference_query_body_yaw_absolute_z")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_scene_bound_unknown_or_unsupported_label_frame_is_rejected(self):
        with np.load(SEED / f"{CLIP}_contacts.npz", allow_pickle=False) as archive:
            labels = {key: archive[key] for key in archive.files}
        labels.pop("method", None)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for suffix in (".npz", ".scene.json"):
                shutil.copy2(SEED / f"{CLIP}{suffix}", directory / f"{CLIP}{suffix}")
            np.savez(directory / f"{CLIP}_contacts.npz", **labels)
            with self.assertRaisesRegex(ValueError, "bound scene training requires reference-query"):
                MotionDataset(RobotSpec.load("g1"), directory, clip_names=[CLIP])
            labels["contact_target_frame"] = np.array("world_xyz")
            np.savez(directory / f"{CLIP}_contacts.npz", **labels)
            with self.assertRaisesRegex(ValueError, "unsupported contact target coordinate frame"):
                MotionDataset(RobotSpec.load("g1"), directory, clip_names=[CLIP])

    def test_legacy_unbound_labels_keep_behavior_with_an_explicit_warning(self):
        with np.load(SEED / f"{CLIP}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["requires_scene"] = np.array(False)
        with np.load(SEED / f"{CLIP}_contacts.npz", allow_pickle=False) as archive:
            labels = {key: archive[key] for key in archive.files}
        labels.pop("method", None)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / f"{CLIP}.npz"
            np.savez(path, **arrays)
            labels["source_sha256"] = np.array(hashlib.sha256(path.read_bytes()).hexdigest())
            np.savez(directory / f"{CLIP}_contacts.npz", **labels)
            with self.assertWarnsRegex(RuntimeWarning, "cannot reproject.*preserving legacy unbound behavior"):
                motion = MotionDataset(RobotSpec.load("g1"), directory, clip_names=[CLIP])
            self.assertEqual(motion.contact_target_frames[CLIP], "legacy_query_body_yaw_absolute_z_unverified")


if __name__ == "__main__":
    unittest.main()
