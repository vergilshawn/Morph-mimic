"""Contact-conditioned whole-body tracker networks (actor + dual critics).

Paper (arXiv 2610.31608): "The tracker is a three-layer multilayer
perceptron with hidden dimensions {512, 256, 128} and ELU activations."
The actor receives 468 features and outputs 23 joint-position offsets.
The critics share the same MLP body; the reference-following critic
receives 353 features and the goal-reaching critic 298.
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(dims: list[int], activation) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class TrackerActor(nn.Module):
    """468 -> 23 joint-position offsets (before the 0.25 action scale)."""

    HIDDEN = [512, 256, 128]
    OBS_DIM = 468
    OBS_DIM_KIN = 468 + 52  # + reference kinematics variant (ablation)

    def __init__(self, num_actions: int = 23, obs_dim: int = OBS_DIM):
        super().__init__()
        self.body = _mlp([obs_dim, *self.HIDDEN], nn.ELU)
        self.head = nn.Linear(self.HIDDEN[-1], num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs))


class Critic(nn.Module):
    """Value network sharing the tracker's MLP body."""

    HIDDEN = [512, 256, 128]

    def __init__(self, obs_dim: int):
        super().__init__()
        self.body = _mlp([obs_dim, *self.HIDDEN], nn.ELU)
        self.head = nn.Linear(self.HIDDEN[-1], 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs)).squeeze(-1)
