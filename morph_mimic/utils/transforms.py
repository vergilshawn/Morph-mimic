"""Utilities for quaternion math, body-yaw frames, and geometry.

Conventions (matching Isaac Lab / the ACP paper):
- Quaternions are WXYZ.
- The body-yaw frame is gravity-aligned, centered at the robot base,
  rotated by the base yaw: +x forward, +y left, z = absolute world height.
"""

import torch


@torch.jit.script
def quat_apply(q, v):
    """Rotate vectors v (..., 3) by quaternions q (..., 4) wxyz."""
    w, xyz = q[..., 0:1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


@torch.jit.script
def quat_conjugate(q):
    qc = q.clone()
    qc[..., 1:] = -qc[..., 1:]
    return qc


@torch.jit.script
def quat_mul(a, b):
    """Multiply quaternions a * b (wxyz)."""
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.cat((w, x, y, z), dim=-1)


@torch.jit.script
def quat_axis_angle(q):
    """Return axis-angle (angle * axis) for quaternion q (wxyz), angle in [0, pi]."""
    q = torch.where(q[..., 0:1] < 0, -q, q)
    vnorm = q[..., 1:].pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-9)
    angle = 2.0 * torch.atan2(vnorm.squeeze(-1), q[..., 0])
    axis = q[..., 1:] / vnorm
    return angle.unsqueeze(-1) * axis


@torch.jit.script
def quat_diff_rad(a, b):
    """Absolute angular difference between two orientations (radians)."""
    return quat_axis_angle(quat_mul(quat_conjugate(a), b)).pow(2).sum(-1).sqrt()


@torch.jit.script
def yaw_from_quat(q):
    """Yaw angle of quaternion q (wxyz)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@torch.jit.script
def yaw_quat(yaw):
    """Quaternion (wxyz) corresponding to a yaw rotation."""
    return torch.stack(
        (torch.cos(yaw / 2.0), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2.0)), dim=-1
    )


@torch.jit.script
def to_body_yaw_frame(points_w, base_pos, base_yaw):
    """Transform world points into the body-yaw frame.

    x, y are centered at the base and rotated by base yaw; z keeps its
    absolute world height (per the paper: contacts live on the terrain
    and keep absolute elevation so the planner supervises a consistent
    frame across the 1.5 s horizon).
    """
    rel = points_w - base_pos
    rel[..., 2] = points_w[..., 2]
    rot = yaw_quat(-base_yaw)
    return quat_apply(rot, rel)


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def wrap_to_pi(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


@torch.jit.script
def encode_goal(base_pos, base_yaw, goal_pos_w, goal_yaw_w, scale_radius: float):
    """Encode goal command g in the body-yaw frame.

    Returns (..., 3): bounded horizontal direction (2) + relative heading (1).
    d_xy / max(||d_xy||, r) with radius r; heading = wrap(goal_yaw - base_yaw).
    """
    rel = goal_pos_w - base_pos
    rel[..., 2] = 0.0
    rot = yaw_quat(-base_yaw)
    rel_body = quat_apply(rot, rel)
    d = rel_body[..., :2]
    norm = d.pow(2).sum(-1, keepdim=True).sqrt().clamp(min=scale_radius)
    direction = d / norm
    heading = wrap_to_pi(goal_yaw_w - base_yaw)
    return torch.cat((direction, heading.unsqueeze(-1)), dim=-1)
