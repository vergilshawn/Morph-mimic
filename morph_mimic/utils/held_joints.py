"""Explicit physical constraints for a reference projected onto fewer joints.

The legacy configuration only requests nominal PD targets on uncontrolled
joints. Those joints remain compliant physical degrees of freedom. The
``hard_limits_at_nominal`` experiment instead gives each explicitly held joint
equal lower and upper articulation limits. This preserves the robot's link,
collision, and inertia topology while constraining the omitted coordinate.
It is a project experiment, not a claim about the ACP authors' hardware.

Call :func:`apply_held_joint_mode` after constructing the environment and before
its first reset, checkpoint validation, or rollout. The Isaac Lab setter updates
the hard and soft limit buffers as well as PhysX. No states are overwritten each
step and no increased PD gain is used. Dynamic limit compliance still requires
a real simulation probe; successful parameter readback alone does not prove it.
"""

from __future__ import annotations

import torch


NOMINAL_PD = "nominal_pd"
HARD_LIMITS_AT_NOMINAL = "hard_limits_at_nominal"


def _held_indices(env):
    """Resolve the declared held joints and refuse unexplained extra DOFs."""
    names = list(env.robot.data.joint_names)
    spec = env.robot_spec
    aliases = (getattr(spec, "asset_aliases", None) or {}).get("joints", {})
    held_names = list(spec.held_joint_names)
    if not held_names or len(held_names) != len(set(held_names)):
        raise ValueError("hard held-joint constraints require distinct declared held joint names")
    ids = []
    for name in held_names:
        candidates = [name, aliases.get(name)]
        matches = [i for i, actual in enumerate(names) if actual in candidates]
        if not matches:
            normalized = name.removesuffix("_joint")
            matches = [i for i, actual in enumerate(names) if actual.removesuffix("_joint") == normalized]
        if len(matches) != 1:
            raise ValueError(f"held joint {name!r} does not resolve uniquely in the runtime articulation")
        ids.append(matches[0])
    if len(set(ids)) != len(ids):
        raise ValueError("declared held joints resolve to duplicate articulation indices")
    runtime_held = set(env.held_idx.detach().cpu().tolist())
    if set(ids) != runtime_held:
        raise ValueError("runtime uncontrolled joints differ from the spec's declared held joints")
    controlled = set(env.controlled_idx.detach().cpu().tolist())
    if controlled & runtime_held or controlled | runtime_held != set(range(len(names))):
        raise ValueError("controlled and held joints must partition the runtime articulation")
    return torch.tensor(ids, dtype=torch.long, device=env.device), held_names


def apply_held_joint_mode(env):
    """Apply and read back the configured held-joint constraint, once at startup.

    ``nominal_pd`` preserves the legacy behavior. ``hard_limits_at_nominal``
    requires the source articulation to already have active, finite limits on
    the held joints. PhysX accepts equal low/high articulation limit values;
    setting low > high through the runtime tensor API is invalid and is not
    used here. The equal limits retain the existing 29-joint tensor topology.
    """
    mode = getattr(env.cfg, "held_joint_mode", NOMINAL_PD)
    if mode not in {NOMINAL_PD, HARD_LIMITS_AT_NOMINAL}:
        raise ValueError(f"unknown held_joint_mode {mode!r}")
    if mode == NOMINAL_PD:
        return {"mode": mode, "physical_limits_changed": False}

    ids, held_names = _held_indices(env)
    robot = env.robot
    original = robot.data.joint_pos_limits.detach().clone()
    expected_shape = (env.num_envs, len(robot.data.joint_names), 2)
    if tuple(original.shape) != expected_shape:
        raise ValueError(f"unexpected articulation limit shape {tuple(original.shape)}; expected {expected_shape}")
    nominal = env.default_pose[ids]
    held_limits = original[:, ids]
    if not torch.isfinite(nominal).all() or not torch.isfinite(held_limits).all():
        raise ValueError("nominal held positions and existing physical limits must be finite")
    if not (held_limits[..., 0] < held_limits[..., 1]).all():
        raise ValueError("held joints must already have nondegenerate active articulation limits")
    if ((nominal < held_limits[..., 0]) | (nominal > held_limits[..., 1])).any():
        raise ValueError("held nominal pose lies outside the original articulation limits")

    limits = nominal[None, :, None].expand(env.num_envs, len(ids), 2).clone()
    robot.write_joint_position_limit_to_sim(limits, joint_ids=ids, warn_limit_violation=False)
    actual = robot.root_physx_view.get_dof_limits().to(original.device)
    expected = original.clone()
    expected[:, ids] = limits
    if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=0, atol=1e-7):
        raise RuntimeError("PhysX joint-limit readback differs from the requested held-only constraint")
    for name in ("joint_pos_limits", "soft_joint_pos_limits"):
        cached = getattr(robot.data, name)[:, ids]
        if not torch.allclose(cached, limits, rtol=0, atol=1e-7):
            raise RuntimeError(f"Isaac Lab {name} did not synchronize with the equal held-joint limits")
    # AcpEnv also holds a static asset-order limit table. Keep that description
    # consistent without affecting controlled-joint rewards or any asset file.
    env.joint_limits[ids] = torch.stack((nominal, nominal), dim=-1)
    report = {
        "mode": mode,
        "physical_limits_changed": True,
        "joint_names": held_names,
        "runtime_joint_indices": ids.detach().cpu().tolist(),
        "nominal_positions_rad": nominal.detach().cpu().tolist(),
        "original_limits_rad": held_limits[0].detach().cpu().tolist(),
        "num_envs": env.num_envs,
        "num_physical_joints": len(robot.data.joint_names),
        "physx_limit_readback_verified": True,
        "dynamic_constraint_compliance_validated": False,
        "provenance": "project controlled experiment; not a paper-specified locking implementation",
    }
    env.held_joint_constraint_report = report
    return report


def held_joint_deviation(env):
    """Measure physical held-coordinate drift for a real simulation probe.

    This reports actual articulation states rather than PD targets. Call before
    auto-reset if a terminal trajectory must also be included in the probe.
    """
    ids, held_names = _held_indices(env)
    position = env.robot.data.joint_pos[:, ids]
    velocity = env.robot.data.joint_vel[:, ids]
    if not torch.isfinite(position).all() or not torch.isfinite(velocity).all():
        raise FloatingPointError("nonfinite physical held-joint state")
    error = (position - env.default_pose[ids]).abs().max(dim=0).values
    speed = velocity.abs().max(dim=0).values
    return {
        "joint_names": held_names,
        "position_error_max_rad": error.detach().cpu().tolist(),
        "velocity_max_rad_s": speed.detach().cpu().tolist(),
        "maximum_position_error_rad": float(error.max()),
    }
