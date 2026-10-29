"""Versioned ACP student networks with ELU after every hidden layer.

The legacy ``tracker`` module remains unchanged so validated reference teachers
keep their original forward function. Its state-dict tensor shapes also fit
these networks, but its missing final hidden activation makes the functions
different: checkpoints must carry and verify this architecture contract.
"""

from __future__ import annotations

import torch
from torch import nn


ARCHITECTURE_CONTRACT = {
    "schema": 1,
    "name": "acp_three_hidden_elu_v1",
    "actor_input_dim": 468,
    "actor_output_dim": 23,
    "critic_input_dims": [353, 298],
    "critic_output_dim": 1,
    "hidden_layers": [512, 256, 128],
    "hidden_activations": ["ELU", "ELU", "ELU"],
    "legacy_tracker_function_compatible": False,
}


def _hidden_body(input_dim: int) -> nn.Sequential:
    # All three entries are hidden layers. The output head is added separately.
    layers: list[nn.Module] = []
    for output_dim in ARCHITECTURE_CONTRACT["hidden_layers"]:
        layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
        input_dim = output_dim
    return nn.Sequential(*layers)


class PaperTrackerActor(nn.Module):
    """Contact-conditioned 468-feature ACP actor; emits 23 nominal offsets."""

    OBS_DIM = 468
    NUM_ACTIONS = 23

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = NUM_ACTIONS):
        super().__init__()
        if obs_dim != self.OBS_DIM or num_actions != self.NUM_ACTIONS:
            raise ValueError("ACP student requires 468 observations and 23 actions")
        self.body = _hidden_body(self.OBS_DIM)
        self.head = nn.Linear(128, self.NUM_ACTIONS)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim < 1 or obs.shape[-1] != self.OBS_DIM:
            raise ValueError("ACP student requires 468-feature observations; privileged teacher inputs are refused")
        return self.head(self.body(obs))


class PaperCritic(nn.Module):
    """Training-time value network with the paper's three hidden ELUs."""

    def __init__(self, obs_dim: int):
        super().__init__()
        if isinstance(obs_dim, bool) or not isinstance(obs_dim, int) or obs_dim <= 0:
            raise ValueError("Critic observation dimension must be a positive integer")
        self.obs_dim = obs_dim
        self.body = _hidden_body(obs_dim)
        self.head = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim < 1 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Critic requires {self.obs_dim}-feature observations")
        return self.head(self.body(obs)).squeeze(-1)
