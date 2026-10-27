"""Unit tests for torch-only modules (no Isaac Sim app needed)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from morph_mimic.contacts.labeling import hindsight_contacts
from morph_mimic.contacts.replay import HindsightBuffer, gaussian_targets, planner_loss
from morph_mimic.networks.acp import ACPPlanner, MLPPlanner
from morph_mimic.networks.tracker import Critic, TrackerActor
from morph_mimic.utils.transforms import encode_goal, quat_apply, to_body_yaw_frame, yaw_from_quat, yaw_quat


class TestTransforms(unittest.TestCase):
    def test_yaw_roundtrip(self):
        yaw = torch.tensor([0.0, 0.5, -2.0, 3.1])
        q = yaw_quat(yaw)
        self.assertTrue(torch.allclose(yaw_from_quat(q), yaw, atol=1e-5))

    def test_quat_apply_identity(self):
        q = torch.tensor([[1.0, 0, 0, 0], yaw_quat(torch.tensor(1.0))])
        v = torch.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]])
        out = quat_apply(q, v)
        self.assertTrue(torch.allclose(out[0], v[0]))
        self.assertTrue(torch.allclose(out[1], torch.tensor([0.5403023, 0.8414710, 0.0]), atol=1e-6))

    def test_body_yaw_frame_keeps_world_z(self):
        p = torch.tensor([[1.0, 0.0, 0.5], [0.0, 2.0, 1.2]])
        base = torch.tensor([[0.5, 0.5, 0.8]])
        yaw = torch.tensor([0.0])
        out = to_body_yaw_frame(p, base, yaw)
        self.assertAlmostEqual(out[0, 2].item(), 0.5)
        self.assertAlmostEqual(out[1, 2].item(), 1.2)
        self.assertAlmostEqual(out[0, 0].item(), 0.5)

    def test_goal_encoding_radius(self):
        base = torch.zeros(1, 3)
        yaw = torch.zeros(1)
        goal = torch.tensor([[0.05, 0.0, 0.0]])
        g = encode_goal(base, yaw, goal, torch.zeros(1), 0.5)
        # direction magnitude shrinks inside the radius
        self.assertLess(g[0, 0].item(), 0.2)


class TestHindsight(unittest.TestCase):
    def test_labeling(self):
        T, B = 200, 2
        forces = torch.zeros(T, B)
        forces[50:120, 0] = 5.0  # valid segment >= min_seg
        forces[130:135, 1] = 5.0  # too short -> discarded
        forces[150:200, 1] = 5.0
        pos = torch.zeros(T, B, 3)
        pos[:, 0, 0] = torch.arange(T) * 0.01
        pos[:, 1, 0] = torch.arange(T) * 0.01
        base = torch.zeros(T, 3)
        base[:, 0] = torch.arange(T) * 0.01
        yaw = torch.zeros(T)
        targets, flags, c_inst = hindsight_contacts(forces, pos, base, yaw, horizon=75)
        self.assertEqual(targets.shape, (T, B, 3))
        # at t=0, body 0's next contact onset is t=50 (within 75) -> flag
        self.assertTrue(flags[0, 0])
        # body 1 at t=0: next valid onset at 150 (>75) -> no flag
        self.assertFalse(flags[0, 1])
        # short segment excluded from instantaneous flags
        self.assertFalse(c_inst[130:135, 1].any())
        self.assertTrue(c_inst[50:120, 0].all())


class TestPlanner(unittest.TestCase):
    def test_acp_shapes_and_param_count(self):
        b = 8
        planner = ACPPlanner()
        n_params = sum(p.numel() for p in planner.parameters())
        points = torch.randn(b, 187, 3)
        heights = torch.randn(b, 17, 11)
        pg = torch.randn(b, 378)
        aff, occ, contacts = planner(points, heights, pg)
        self.assertEqual(aff.shape, (b, 9, 187))
        self.assertEqual(occ.shape, (b, 9))
        self.assertEqual(contacts.shape, (b, 9, 3))
        self.assertTrue(torch.allclose(aff.sum(-1), torch.ones(b, 9), atol=1e-4))
        print(f"\nACP planner parameters: {n_params} (paper: 786,993)")
        self.assertTrue(abs(n_params - 786_993) < 2000, f"param count {n_params} deviates from paper")

    def test_mlp_shapes_and_param_count(self):
        b = 8
        planner = MLPPlanner()
        n_params = sum(p.numel() for p in planner.parameters())
        points = torch.randn(b, 187, 3)
        heights = torch.randn(b, 17, 11)
        pg = torch.randn(b, 378)
        aff, occ, _ = planner(points, heights, pg)
        self.assertEqual(aff.shape, (b, 9, 187))
        self.assertEqual(occ.shape, (b, 9))
        print(f"MLP planner parameters: {n_params} (paper: 833,322)")

    def test_occupancy_gating(self):
        planner = ACPPlanner()
        points = torch.randn(4, 187, 3)
        heights = torch.randn(4, 17, 11)
        pg = torch.randn(4, 378)
        _, occ, contacts = planner(points, heights, pg)
        masked = contacts[(occ <= 0.5)]
        self.assertTrue((masked == 0).all())

    def test_planner_loss(self):
        b = 16
        aff = torch.rand(b, 9, 187)
        aff = aff / aff.sum(-1, keepdim=True)
        occ = torch.rand(b, 9).sigmoid()
        p_star = gaussian_targets(torch.randn(b, 187, 3), torch.randn(b, 9, 3), (torch.rand(b, 9) > 0.5).float())
        c_star = (torch.rand(b, 9) > 0.6).float()
        loss, terms = planner_loss(aff, occ, p_star, c_star)
        self.assertTrue(loss.isfinite())
        self.assertIn("affordance", terms)


class TestTracker(unittest.TestCase):
    def test_shapes(self):
        actor = TrackerActor()
        critic_m = Critic(353)
        critic_fs = Critic(298)
        obs = torch.randn(16, 468)
        self.assertEqual(actor(obs).shape, (16, 23))
        self.assertEqual(critic_m(torch.randn(16, 353)).shape, (16,))
        self.assertEqual(critic_fs(torch.randn(16, 298)).shape, (16,))


class TestBuffer(unittest.TestCase):
    def test_add_sample(self):
        buf = HindsightBuffer(cap_steps_per_partition=2000)
        for i in range(3):
            k = 100
            buf.add(
                f"box/tile{i}",
                {
                    "points": torch.randn(k, 187, 3),
                    "heights": torch.randn(k, 17, 11),
                    "pg": torch.randn(k, 378),
                    "targets": torch.randn(k, 9, 3),
                    "flags": torch.rand(k, 9) > 0.5,
                },
                quality=float(i),
            )
        self.assertEqual(buf.size_steps(), 300)
        s = buf.sample(64, torch.device("cpu"))
        self.assertIsNotNone(s)
        self.assertEqual(s["points"].shape, (64, 187, 3))
        # eviction keeps the best (lowest) quality episodes
        buf2 = HindsightBuffer(cap_steps_per_partition=150)
        for i in range(5):
            buf2.add("box/t", {"points": torch.randn(100, 187, 3), "heights": torch.randn(100, 17, 11),
                               "pg": torch.randn(100, 378), "targets": torch.randn(100, 9, 3),
                               "flags": torch.rand(100, 9) > 0.5}, quality=float(i))
        self.assertLessEqual(buf2.size_steps(), 150)
        self.assertEqual(buf2.parts["box/t"][0]["quality"], 0.0)  # best kept


if __name__ == "__main__":
    unittest.main(verbosity=2)
