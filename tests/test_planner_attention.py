"""Behavioral checks for contact-planner attention probabilities."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from morph_mimic.networks.acp import ACPPlanner, MLPPlanner


class _FixedQueries(nn.Module):
    def forward(self, context):
        return context.new_full((context.shape[0], 9, 96), 100.0)


class TestPlannerAttention(unittest.TestCase):
    def test_sharp_attention_preserves_contact_selection_confidence(self):
        planner = ACPPlanner()
        target_index = 42
        points = torch.randn(1, 187, 3)
        kv = torch.zeros(1, 187, 96)
        kv[:, target_index] = 1.0
        pooled = torch.zeros(1, 64)
        planner.film = _FixedQueries()

        # Make all 32 attention heads select the same terrain point. This uses
        # the real MultiheadAttention softmax, including its head averaging.
        with torch.no_grad():
            identity = torch.eye(96)
            planner.attn.in_proj_weight.copy_(identity.repeat(3, 1))
            planner.attn.in_proj_bias.zero_()
            planner.occupancy_head[-1].weight.zero_()
            planner.occupancy_head[-1].bias.fill_(10.0)
        with patch.object(planner, "_terrain_features", return_value=(kv, pooled)):
            affordance, occupancy, contacts = planner(
                points, torch.zeros(1, 17, 11), torch.zeros(1, 378)
            )

        self.assertTrue((affordance[..., target_index] > 0.99).all())
        self.assertTrue(torch.allclose(affordance.sum(-1), torch.ones(1, 9)))
        self.assertTrue((occupancy > 0.5).all())
        expected = points[:, target_index].unsqueeze(1).expand(-1, 9, -1)
        self.assertTrue(torch.equal(contacts, expected))

    def test_temperature_scales_log_probabilities(self):
        planner = ACPPlanner(temperature=0.5)
        weights = torch.full((1, 9, 187), 0.2 / 186)
        weights[..., 17] = 0.8
        expected = weights.square() / weights.square().sum(-1, keepdim=True)
        with patch.object(
            planner.attn, "forward", return_value=(torch.zeros(1, 9, 96), weights)
        ):
            affordance, _, _ = planner(
                torch.zeros(1, 187, 3),
                torch.zeros(1, 17, 11),
                torch.zeros(1, 378),
            )
        self.assertTrue(torch.allclose(affordance, expected, atol=1e-6))

    def test_mlp_planner_decodes_contacts_with_shared_terrain_decoder(self):
        planner = MLPPlanner()
        points = torch.randn(2, 187, 3)
        with torch.no_grad():
            planner.terrain_head.weight.zero_()
            planner.terrain_head.bias.zero_()
            planner.terrain_head.bias.view(9, 187)[:, 27] = 10.0
            planner.occ_head.weight.zero_()
            planner.occ_head.bias.fill_(10.0)
            affordance, occupancy, contacts = planner(
                points, torch.zeros(2, 17, 11), torch.zeros(2, 378)
            )
        expected = points[:, 27].unsqueeze(1).expand(-1, 9, -1)
        self.assertEqual(planner.sigma_a, 0.1)
        self.assertTrue(torch.equal(contacts, expected))
        self.assertTrue(torch.equal(
            contacts, ACPPlanner().decode(points, affordance, occupancy)
        ))
        self.assertTrue(torch.equal(
            planner.decode(points, affordance, torch.zeros_like(occupancy)),
            torch.zeros_like(contacts),
        ))


if __name__ == "__main__":
    unittest.main()
