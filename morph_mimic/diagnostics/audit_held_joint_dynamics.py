"""Probe real held-coordinate constraints before auto-reset, without training.

Zero commands and reference-joint PD targets are tested from several reference
phases. Finite states and actual held deviations are recorded even if the robot
fails its tracking task. This audits constraints, not box-climbing success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np


def _fingerprint_collision_geometry(stage):
    from pxr import Usd, UsdPhysics

    descriptions = []
    # MJCF assets are instanceable: ordinary Stage.Traverse skips their
    # descendant collision prims and would accidentally fingerprint terrain
    # only. Include instance proxies explicitly.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            descriptions.append((str(prim.GetPath()), str(prim.GetTypeName()), [
                (str(attr.GetName()), repr(attr.Get())) for attr in prim.GetAttributes()
                if str(attr.GetName()) not in {"physics:lowerLimit", "physics:upperLimit"}
            ]))
    payload = json.dumps(descriptions, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "collision_prim_count": len(descriptions)}


def audit(args, app, report):
    import torch

    from morph_mimic.scripts.common import build_env
    from morph_mimic.utils import held_joints
    from morph_mimic.utils.checkpoint import file_hash

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    original_apply = held_joints.apply_held_joint_mode
    preservation = {}

    def observe_constraint_application(env):
        view = env.robot.root_physx_view
        masses, inertias = view.get_masses().clone(), view.get_inertias().clone()
        geometry = _fingerprint_collision_geometry(env.sim.stage)
        result = original_apply(env)
        preservation.update(
            masses_exactly_unchanged=bool(torch.equal(masses, view.get_masses())),
            inertias_exactly_unchanged=bool(torch.equal(inertias, view.get_inertias())),
            collision_geometry_before=geometry,
            collision_geometry_after=_fingerprint_collision_geometry(env.sim.stage),
            constraint_application=result,
        )
        preservation["collision_geometry_exactly_unchanged"] = (
            preservation["collision_geometry_before"] == preservation["collision_geometry_after"])
        if geometry["collision_prim_count"] < env.num_envs:
            raise RuntimeError("collision fingerprint did not include the replicated robot collision prims")
        return result

    env = None
    held_joints.apply_held_joint_mode = observe_constraint_application
    try:
        env, spec = build_env(
            1, args.robot, args.num_envs, ref_env_ratio=1., device=args.device,
            motion_dir=str(args.motion_dir), clips=[args.clip], training_profile="nominal_pretrain",
            held_joint_mode="hard_limits_at_nominal", evaluation=True,
            evaluation_start_phase=0., evaluation_record_trajectory=True,
            episode_length_s=15., csi_enabled=False, seed=args.seed,
        )
        held_joints.apply_held_joint_mode = original_apply
        report["asset_preservation"] = preservation
        report["provenance"] = {
            "motion_sha256": file_hash(args.motion_dir / f"{args.clip}.npz"),
            "contacts_sha256": file_hash(args.motion_dir / f"{args.clip}_contacts.npz"),
            "robot_spec_sha256": file_hash(Path(__file__).resolve().parents[2] / "assets" / args.robot / "robot_spec.json"),
            "source_mjcf_sha256": file_hash(Path(__file__).resolve().parents[2] / spec.runtime_assets["mjcf"]),
            "held_helper_sha256": file_hash(Path(held_joints.__file__)),
        }
        ids = env.held_idx
        names = [env.robot.data.joint_names[i] for i in ids.cpu().tolist()]
        report["held_joint_names_runtime_order"] = names
        nominal = env.default_pose[ids]
        current = {}
        original_dones = env._get_dones

        def capture_before_reset():
            # DirectRLEnv calls this after integration, before its reset handler.
            q = env.robot.data.joint_pos[:, ids]
            qd = env.robot.data.joint_vel[:, ids]
            finite = all(bool(torch.isfinite(value).all()) for value in (
                q, qd, env.robot.data.root_state_w,
                env.robot.data.joint_pos, env.robot.data.joint_vel,
            ))
            current["finite"] &= finite
            if finite:
                current["position_max"] = torch.maximum(current["position_max"], (q - nominal).abs().amax(0))
                current["velocity_max"] = torch.maximum(current["velocity_max"], qd.abs().amax(0))
            current["pre_reset_samples"] += args.num_envs
            result = original_dones()
            current["terminated_count"] += int(result[0].sum())
            current["truncated_count"] += int(result[1].sum())
            return result

        env._get_dones = capture_before_reset
        report["cases"] = []
        for mode in ("zero_action", "reference_joint_pd"):
            env.cfg.replay = mode == "reference_joint_pd"
            for phase in args.phases:
                env.cfg.evaluation_start_phase = phase
                observation, _ = env.reset()
                initial_phases = env.phase.detach().clone()
                initial_q = env.robot.data.joint_pos[:, ids].detach().clone()
                current = dict(position_max=(initial_q - nominal).abs().amax(0),
                               velocity_max=env.robot.data.joint_vel[:, ids].abs().amax(0),
                               pre_reset_samples=0, terminated_count=0, truncated_count=0, finite=True)
                reset_phase_max_error = float((initial_phases - phase).abs().max())
                actions = torch.zeros(args.num_envs, spec.n_controlled, device=env.device)
                for _ in range(args.steps):
                    if not app.is_running():
                        raise RuntimeError("simulation application stopped during the dynamics audit")
                    observation, reward, terminated, truncated, info = env.step(actions)
                    current["finite"] &= bool(torch.isfinite(reward).all())
                    current["finite"] &= all(bool(torch.isfinite(value).all()) for value in observation.values())
                case = {
                    "mode": mode, "requested_reset_phase": phase,
                    "actual_initial_phases": initial_phases.cpu().tolist(),
                    "reset_phase_max_error": reset_phase_max_error,
                    "steps": args.steps, "control_duration_s": args.steps * spec.control_dt,
                    "pre_reset_state_samples": current["pre_reset_samples"],
                    "all_states_rewards_observations_finite": current["finite"],
                    "held_position_error_max_rad": current["position_max"].cpu().tolist(),
                    "held_velocity_max_rad_s": current["velocity_max"].cpu().tolist(),
                    "maximum_held_position_error_rad": float(current["position_max"].max()),
                    "terminated_env_events": current["terminated_count"],
                    "truncated_env_events": current["truncated_count"],
                    "failures_not_hidden_by_reset": True,
                }
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
        report["completed"] = True
        report["maximum_held_position_error_rad"] = max(c["maximum_held_position_error_rad"] for c in report["cases"])
        report["accepted"] = (
            all(preservation[k] for k in ("masses_exactly_unchanged", "inertias_exactly_unchanged",
                                         "collision_geometry_exactly_unchanged"))
            and all(c["all_states_rewards_observations_finite"] and c["reset_phase_max_error"] < 1e-6
                    for c in report["cases"])
            and report["maximum_held_position_error_rad"] <= args.position_tolerance
        )
    finally:
        held_joints.apply_held_joint_mode = original_apply
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "mojia"), default="g1")
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--phases", type=float, nargs="+", default=[0., .5, .9])
    parser.add_argument("--position-tolerance", type=float, default=.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing audit")
    if args.steps < 50 or args.num_envs < 1 or any(not 0 <= p < 1 for p in args.phases):
        parser.error("require at least 50 steps, positive environment count, and phases in [0, 1)")
    args.motion_dir = args.motion_dir.resolve()
    report = dict(schema=1, completed=False, accepted=False, robot=args.robot, device=args.device,
                  num_envs=args.num_envs, training_profile="nominal_pretrain",
                  held_joint_mode="hard_limits_at_nominal", physical_skill_validated=False,
                  scope="physical held-coordinate compliance; this is not a tracking-success evaluation",
                  engineering_position_tolerance_rad=args.position_tolerance,
                  sample_timing="post-physics _get_dones entry, before all automatic resets")
    app = None
    try:
        from morph_mimic.launcher import launch
        app = launch(headless=True, device=args.device)
        audit(args, app, report)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: report.get(k) for k in ("completed", "accepted", "maximum_held_position_error_rad", "error")}), flush=True)
        status = 0 if report["completed"] and report["accepted"] else 1
        if app is not None:
            app.app.post_quit(status)
            from morph_mimic.launcher import close
            close(app)
        if status:
            raise SystemExit(status)


if __name__ == "__main__":
    main()
