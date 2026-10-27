"""Contact labeling: hindsight extraction of contact onsets and targets.

Paper (arXiv 2610.31608, Table S5):
- contact onset identified from the vertical contact force (> 1 N),
- segments shorter than 10 consecutive control steps are discarded,
- the next contact within a 1.5 s (75 steps) horizon becomes the target,
- targets are stored in the body-yaw frame of the querying timestep,
- targets outside the local scan footprint are invalid (flag = 0).
"""

from __future__ import annotations

import torch

from morph_mimic.utils.transforms import to_body_yaw_frame


@torch.jit.script
def _segment_contacts(forces: torch.Tensor, force_thr: float, min_seg: int):
    """Filter contact flags into segments lasting >= min_seg steps.

    Returns flags (T, B) where only frames inside valid segments are True.
    """
    t, b = forces.shape
    raw = forces > force_thr
    # segment ids via cumulative sum of transitions
    shifted = torch.zeros_like(raw)
    shifted[1:] = raw[:-1]
    starts = raw & ~shifted
    ids = starts.cumsum(0)  # (T, B) segment id per body per frame
    seg_len = torch.zeros(t, b, dtype=torch.long, device=forces.device)
    max_id = int(ids.max()) + 1
    for sid in range(max_id):
        m = ids == sid
        # segment length counts *contact* frames only (ids spans gaps too)
        ln = (m & raw).sum(0)
        seg_len = seg_len + m.to(torch.long) * ln.unsqueeze(0)
    return raw & (seg_len >= min_seg)


def hindsight_contacts(
    forces: torch.Tensor,
    body_pos_w: torch.Tensor,
    base_pos: torch.Tensor,
    base_yaw: torch.Tensor,
    horizon: int = 75,
    min_seg: int = 10,
    force_thr: float = 1.0,
    footprint_x: tuple[float, float] = (-0.8, 0.8),
    footprint_y: tuple[float, float] = (-0.5, 0.5),
    return_instantaneous: bool = True,
):
    """Label a recorded trajectory.

    Args:
        forces: (T, B) vertical contact forces per body (50 Hz control steps).
        body_pos_w: (T, B, 3) contact-body link positions in world frame.
        base_pos: (T, 3) base positions.
        base_yaw: (T,) base yaw angles.

    Returns:
        targets: (T, B, 3) next contact position within the horizon, in the
            body-yaw frame of the querying timestep (zero if none).
        flags: (T, B) occupancy flags (1 if a valid upcoming contact exists).
        c_inst: (T, B) instantaneous (segment-filtered) contact flags.
    """
    seg = _segment_contacts(forces, force_thr, min_seg)
    t, b = forces.shape

    # onset frame of each segment: first True after a False (or at frame 0)
    shifted = torch.zeros_like(seg)
    shifted[1:] = seg[:-1]
    onset = seg & ~shifted

    # next onset within horizon, per (t, b): index of first onset in (t, t+horizon]
    onset_idx = onset.nonzero()  # (K, 2) frame, body
    targets = torch.zeros(t, b, 3, device=forces.device, dtype=body_pos_w.dtype)
    flags = torch.zeros(t, b, device=forces.device, dtype=torch.bool)
    if onset_idx.numel() > 0:
        frames = onset_idx[:, 0]
        bodies = onset_idx[:, 1]
        # for efficiency, iterate over bodies
        for bi in range(b):
            m = bodies == bi
            if not m.any():
                continue
            f = frames[m]
            # for each query frame q, next onset = first f > q within q+horizon
            q = torch.arange(t, device=forces.device)
            idx = torch.searchsorted(f, q + 1)
            nxt = torch.where(idx < f.numel(), f[idx.clamp(max=f.numel() - 1)], torch.full_like(q, t + horizon))
            within = (nxt <= q + horizon) & (nxt < t)
            flags[:, bi] = within
            nxt_clamped = nxt.clamp(max=t - 1)
            pos_w = body_pos_w[nxt_clamped, bi]  # (T, 3) onset positions
            # transform onset position into body-yaw frame of the query step
            yaw = base_yaw[torch.arange(t, device=forces.device)]
            tg = to_body_yaw_frame(pos_w, base_pos, yaw)
            fx = (tg[..., 0] >= footprint_x[0]) & (tg[..., 0] <= footprint_x[1])
            fy = (tg[..., 1] >= footprint_y[0]) & (tg[..., 1] <= footprint_y[1])
            ok = within & fx & fy
            flags[:, bi] = ok
            targets[:, bi] = torch.where(ok.unsqueeze(-1), tg, torch.zeros_like(tg))

    out = [targets, flags]
    if return_instantaneous:
        out.append(seg)
    return out
