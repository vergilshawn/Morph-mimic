"""Attention-based contact planner (ACP) and MLP baseline.

Implements Table S3 of arXiv 2610.31608 exactly:

Attention planner (target 786,993 parameters):
- Terrain-point embedding        3 -> 16   Linear, ELU
- Height-map encoder             1 -> 16 -> 48   5x5 kernels, padding 2, ELU
- Terrain-context encoder        96 -> 64 -> 64  pointwise MLP, max pooling
- Proprioception-goal-context    442 -> 128 -> 128 -> 96  MLP, ReLU
- Body embeddings                9 x 96   learned
- FiLM parameter generator       96 -> 64 -> 192  scale + shift
- Cross-attention                d = 96, 32 heads, 1 layer, no dropout
- Contact-occupancy head         160 -> 512 -> 512 -> 512 -> 1  sigmoid

The planner selects contacts from the perceived surface: inputs are the
187 height-scan points (17 x 11) in the body-yaw frame plus a 5-step
proprioceptive history (375) and goal command (3). Outputs are an
affordance distribution over the points per body and a per-body
occupancy probability. The decoder takes the top-k affordance points
(k = 1 in the paper's quantitative evaluations).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _FiLM(nn.Module):
    """Feature-wise linear modulation from a shared 96-d context to 9 body queries."""

    def __init__(self, dim: int = 96, num_bodies: int = 9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 2 * dim)
        )
        self.num_bodies = num_bodies
        self.dim = dim
        self.emb = nn.Parameter(torch.randn(num_bodies, dim) * 0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context (..., 96) -> queries (..., 9, 96)."""
        out = self.mlp(context)  # (..., 192)
        scale, shift = out.chunk(2, dim=-1)
        return self.emb * scale.unsqueeze(-2) + shift.unsqueeze(-2)


class ACPPlanner(nn.Module):
    """Attention-based contact planner (paper Table S3).

    Args:
        proprio_dim: proprioceptive history dim (375 = 5 x 75).
        goal_dim: goal command dim (3).
        scan_shape: (L, W) of the planner height scan (17, 11).
        num_bodies: number of contact bodies (9).
        attention_dim: d = 96.
        num_heads: 32.
        sigma_a: soft-target bandwidth (0.1 m), used by the training loss only.
        top_k: decoder k (1).
        temperature: temperature applied to attention probabilities (default 1.0).
    """

    def __init__(
        self,
        proprio_dim: int = 375,
        goal_dim: int = 3,
        scan_shape: tuple[int, int] = (17, 11),
        num_bodies: int = 9,
        attention_dim: int = 96,
        num_heads: int = 32,
        sigma_a: float = 0.1,
        top_k: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.L, self.W = scan_shape
        self.LW = self.L * self.W
        self.num_bodies = num_bodies
        self.dim = attention_dim
        self.top_k = top_k
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.sigma_a = sigma_a

        self.pt_embed = nn.Sequential(nn.Linear(3, 16), nn.ELU())
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 48, kernel_size=5, padding=2),
            nn.ELU(),
        )
        self.kv_proj = nn.Linear(64, self.dim)  # per-point features -> K/V
        self.terrain_ctx = nn.Sequential(
            nn.Linear(self.dim, 64), nn.ELU(), nn.Linear(64, 64)
        )  # pointwise 96->64->64, max-pooled outside

        # proprio + goal + pooled terrain context (375 + 3 + 64 = 442)
        self.prop_goal_ctx = nn.Sequential(
            nn.Linear(proprio_dim + goal_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.dim),
        )
        self.film = _FiLM(self.dim, num_bodies)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )
        self.occupancy_head = nn.Sequential(
            nn.Linear(self.dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def _terrain_features(self, points: torch.Tensor, heights: torch.Tensor):
        """points (B, LW, 3), heights (B, L, W) -> per-point K/V (B, LW, 96), pooled ctx (B, 64)."""
        b = points.shape[0]
        pt = self.pt_embed(points)  # (B, LW, 16)
        h = self.cnn(heights.unsqueeze(1))  # (B, 48, L, W)
        h = h.flatten(2).transpose(1, 2)  # (B, LW, 48)
        feats = torch.cat((pt, h), dim=-1)  # (B, LW, 64)
        kv = self.kv_proj(feats)  # (B, LW, 96)
        ctx = self.terrain_ctx(kv)  # (B, LW, 64)
        pooled = ctx.max(dim=1).values  # (B, 64)
        return kv, pooled

    def forward(
        self,
        points: torch.Tensor,
        heights: torch.Tensor,
        proprio_goal: torch.Tensor,
    ):
        """Predict contact plans.

        Args:
            points: (B, LW, 3) scan points in the body-yaw frame (z = world height).
            heights: (B, L, W) scan heights (relative to base; used only by the CNN).
            proprio_goal: (B, proprio_dim + goal_dim).

        Returns:
            affordance: (B, 9, LW) softmaxed per-body weights over scan points.
            occupancy: (B, 9) contact occupancy probabilities.
            contacts: (B, 9, 3) decoded contact targets (top-k; zero if occupancy <= 0.5).
        """
        kv, pooled = self._terrain_features(points, heights)
        context = self.prop_goal_ctx(torch.cat((proprio_goal, pooled), dim=-1))
        # shared feature vector: the 442->96 encoder output (proprio + goal +
        # pooled terrain context) directly drives the body queries via FiLM
        queries = self.film(context)  # (B, 9, 96)
        attn_out, attn_weights = self.attn(queries, kv, kv, need_weights=True, average_attn_weights=True)
        # MultiheadAttention already returns normalized probabilities. Applying
        # softmax to those probabilities again prevents sharp contact selection.
        affordance = attn_weights
        if self.temperature != 1.0:
            affordance = F.softmax(
                attn_weights.clamp_min(torch.finfo(attn_weights.dtype).tiny).log()
                / self.temperature,
                dim=-1,
            )

        occ_in = torch.cat((attn_out, pooled.unsqueeze(1).expand(-1, self.num_bodies, -1)), dim=-1)
        occupancy = self.occupancy_head(occ_in).squeeze(-1)  # (B, 9)
        occupancy = torch.sigmoid(occupancy)

        contacts = self.decode(points, affordance, occupancy)
        return affordance, occupancy, contacts

    def decode(self, points: torch.Tensor, affordance: torch.Tensor, occupancy: torch.Tensor):
        """Top-k decoder: contact targets coincide with observed terrain points."""
        k = min(self.top_k, self.LW)
        if k == 1:
            idx = affordance.argmax(dim=-1)  # (B, 9)
            contacts = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            topk = affordance.topk(k, dim=-1)  # values, indices (B, 9, k)
            w = topk.values / topk.values.sum(-1, keepdim=True).clamp(min=1e-9)
            sel = torch.gather(points, 1, topk.indices.unsqueeze(-1).expand(-1, -1, -1, 3))
            contacts = (sel * w.unsqueeze(-1)).sum(-2)
        mask = (occupancy <= 0.5).unsqueeze(-1)
        return torch.where(mask, torch.zeros_like(contacts), contacts)


class MLPPlanner(nn.Module):
    """MLP baseline (paper Table S3, 833,322 parameters).

    Input: 939 = 375 proprio + 3 goal + 561 flattened 3-D terrain points.
    """

    INPUT_DIM = 939

    def __init__(
        self,
        scan_shape: tuple[int, int] = (17, 11),
        num_bodies: int = 9,
        proprio_dim: int = 375,
        goal_dim: int = 3,
    ):
        super().__init__()
        self.L, self.W = scan_shape
        self.LW = self.L * self.W
        self.num_bodies = num_bodies
        self.sigma_a = 0.1
        self.top_k = 1
        in_dim = proprio_dim + goal_dim + 3 * self.LW
        assert in_dim == self.INPUT_DIM, in_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 285), nn.ELU(), nn.Linear(285, 285), nn.ELU()
        )
        self.terrain_head = nn.Linear(285, num_bodies * self.LW)
        self.occ_head = nn.Linear(285, num_bodies)

    # Both planners use the same terrain-point contact representation.
    decode = ACPPlanner.decode

    def forward(
        self,
        points: torch.Tensor,
        heights: torch.Tensor,
        proprio_goal: torch.Tensor,
    ):
        del heights  # MLP baseline sees the flattened 3-D points only
        x = torch.cat((proprio_goal, points.flatten(1)), dim=-1)
        h = self.backbone(x)
        logits = self.terrain_head(h).view(-1, self.num_bodies, self.LW)
        affordance = F.softmax(logits, dim=-1)
        occupancy = torch.sigmoid(self.occ_head(h))
        return affordance, occupancy, self.decode(points, affordance, occupancy)
