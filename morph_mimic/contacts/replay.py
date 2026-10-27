"""Planner supervision targets, loss (Eq. 2), and hindsight replay buffers.

The planner's targets come from the policy's own successful experience
rather than from the motion dataset, so that it learns contacts the
tracker can actually realize. Buffer sampling follows the paper's
hierarchy: uniform over terrain types, then tiles, then stored timesteps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_targets(
    points: torch.Tensor, targets_w: torch.Tensor, flags: torch.Tensor, sigma_a: float = 0.1
) -> torch.Tensor:
    """Soft Gaussian distribution over scan points per body (paper Eq. 1).

    Args:
        points: (B, LW, 3) scan points in the body-yaw frame (z = world height).
        targets_w: (B, 9, 3) hindsight contact targets in the same frame.
        flags: (B, 9) valid-target flags.

    Returns:
        p_star: (B, 9, LW) normalized soft distributions (uniform where invalid).
    """
    d = (points.unsqueeze(1) - targets_w.unsqueeze(2)).pow(2).sum(-1)  # (B, 9, LW)
    logits = -d / (2.0 * sigma_a**2)
    logits = torch.where(flags.unsqueeze(-1).bool(), logits, torch.full_like(logits, -1e9))
    return F.softmax(logits, dim=-1)


def planner_loss(
    affordance: torch.Tensor,
    occupancy: torch.Tensor,
    p_star: torch.Tensor,
    c_star: torch.Tensor,
    normalization: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked cross-entropy + binary cross-entropy (paper Eq. 2).

    Args:
        affordance: (B, 9, LW) predicted affordance distributions.
        occupancy: (B, 9) predicted occupancy probabilities.
        p_star: (B, 9, LW) soft Gaussian targets.
        c_star: (B, 9) target occupancy flags (float).

    Returns:
        loss, {"affordance": ..., "occupancy": ...}
    """
    c_star = c_star.to(dtype=occupancy.dtype)
    b = affordance.shape[0]
    valid = c_star.bool()
    denom = max(1, int(valid.sum()) if normalization is None else normalization[0])
    aff = -(p_star * (affordance + 1e-12).log()).sum(-1)  # (B, 9)
    aff = aff.masked_select(valid).sum() / denom
    occ_batch = b if normalization is None else normalization[1]
    occ = F.binary_cross_entropy(occupancy, c_star, reduction="sum") / (9 * occ_batch)
    return aff + occ, {"affordance": aff.detach(), "occupancy": occ.detach()}


class HindsightBuffer:
    """Hindsight replay buffer, partitioned by terrain type and tile.

    Episodes are admitted whole (only successful rollouts enter); within a
    partition, quality-based eviction drops the episodes with the highest
    normalized positive mechanical work (least efficient) first.

    Sampling: uniform over terrain types, then over tiles, then over
    timesteps (an episode's weight is proportional to its length).
    """

    def __init__(self, cap_steps_per_partition: int = 40000):
        self.cap_steps = cap_steps_per_partition
        # key -> list of episode dicts; each episode: fields (K, ...) + "quality" scalar
        self.parts: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ add
    def add(self, part_key: str, samples: dict[str, torch.Tensor], quality: float):
        ep = {k: v.detach().cpu() for k, v in samples.items()}
        ep["quality"] = quality
        self.parts.setdefault(part_key, []).append(ep)
        self._evict(part_key)

    def _evict(self, part_key: str):
        eps = self.parts[part_key]
        total = sum(e["points"].shape[0] for e in eps)
        if total <= self.cap_steps:
            return
        # drop worst (highest work-normalized) episodes until within cap
        eps.sort(key=lambda e: -e["quality"])
        while total > self.cap_steps and eps:
            total -= eps[0]["points"].shape[0]
            eps.pop(0)

    # ------------------------------------------------------------------ query
    def num_episodes(self, part_key: str | None = None) -> int:
        if part_key is not None:
            return len(self.parts.get(part_key, []))
        return sum(len(v) for v in self.parts.values())

    def size_steps(self, part_key: str | None = None) -> int:
        if part_key is not None:
            return sum(e["points"].shape[0] for e in self.parts.get(part_key, []))
        return sum(e["points"].shape[0] for v in self.parts.values() for e in v)

    def type_keys(self) -> dict[str, list[str]]:
        types: dict[str, list[str]] = {}
        for key in self.parts:
            types.setdefault(key.split("/")[0], []).append(key)
        return types

    # ------------------------------------------------------------------ sample
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        type_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Hierarchical-uniform sample across the whole buffer."""
        types = self.type_keys()
        # keep only types with data
        types = {t: [k for k in ks if self.parts[k]] for t, ks in types.items()}
        types = {t: ks for t, ks in types.items() if ks}
        if not types:
            return None
        tnames = list(types)
        if type_weights is None:
            tw = torch.ones(len(tnames), device=device) / len(tnames)
        else:
            tw = torch.tensor([type_weights.get(t, 0.0) for t in tnames], device=device)
            tw = tw / tw.sum().clamp(min=1e-9)
        t_draw = torch.multinomial(tw, batch_size, replacement=True)  # (B,)
        t_cnt = torch.bincount(t_draw, minlength=len(tnames))

        fields = ("points", "heights", "pg", "targets", "flags")
        gathered = {f: [] for f in fields}
        for ti, (tname, keys) in enumerate(types.items()):
            cnt = int(t_cnt[ti])
            if cnt == 0:
                continue
            # uniform over tiles of this type
            tile_draw = torch.randint(0, len(keys), (cnt,), device=device)
            tile_cnt = torch.bincount(tile_draw, minlength=len(keys))
            for ki, key in enumerate(keys):
                kcnt = int(tile_cnt[ki])
                if kcnt == 0:
                    continue
                eps = self.parts[key]
                lengths = torch.tensor([e["points"].shape[0] for e in eps], device=device, dtype=torch.float)
                ep_draw = torch.multinomial(lengths, kcnt, replacement=True)
                ep_cnt = torch.bincount(ep_draw, minlength=len(eps))
                for ei, e in enumerate(eps):
                    ecnt = int(ep_cnt[ei])
                    if ecnt == 0:
                        continue
                    n = e["points"].shape[0]
                    # Episodes live on CPU even when the learner runs on CUDA.
                    ix = torch.randint(0, n, (ecnt,), device="cpu")
                    for f in fields:
                        gathered[f].append(e[f][ix])
        if not gathered["points"]:
            return None
        return {f: torch.cat(gathered[f], dim=0).to(device) for f in fields}

    # ------------------------------------------------------------------ io
    def save(self, path):
        torch.save({"parts": self.parts}, path)

    def load(self, path):
        data = torch.load(path, weights_only=False)
        self.parts = data["parts"]
