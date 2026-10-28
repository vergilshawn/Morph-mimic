"""Audit independent bound-scene copies in an actual Isaac simulation.

This checks placement, sensors, finite zero-action stepping and subset-reset
isolation. It neither trains a policy nor validates execution of the skill.
Run with the Isaac Lab Python environment after preparing the runtime MJCF.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

import numpy as np


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "mojia"), default="g1")
    parser.add_argument("--motion-dir", "--motion_dir", dest="motion_dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="static placement/ray-height tolerance in metres")
    parser.add_argument("--synchronization-tolerance", type=float, default=1e-2,
                        help="zero-action inter-copy root-distance/joint-angle tolerance, m/rad")
    args = parser.parse_args(argv)
    if args.num_envs < 2:
        parser.error("--num-envs must be at least 2 to audit replication and reset isolation")
    if args.steps < 1:
        parser.error("--steps must be positive")
    for key in ("tolerance", "synchronization_tolerance"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if len(args.clips) != 1:
        parser.error("exactly one bound clip is required")
    if args.out.exists():
        parser.error(f"refusing to overwrite audit report: {args.out}")
    args.motion_dir = args.motion_dir.resolve()
    for name in args.clips:
        if Path(name).name != name or not (args.motion_dir / f"{name}.npz").is_file():
            parser.error(f"motion clip not found: {name}")
        if not (args.motion_dir / f"{name}.scene.json").is_file():
            parser.error(f"bound scene sidecar not found: {name}.scene.json")
    return args


def _numpy(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.asarray(value).copy()


def _finite(value):
    return bool(np.isfinite(_numpy(value)).all())


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _spread(value):
    """Maximum absolute component difference against copy zero."""
    a = _numpy(value)
    return float(np.max(np.abs(a - a[:1])))


def _check(report, name, accepted, **stats):
    report["checks"][name] = dict(accepted=bool(accepted), **stats)
    if not accepted:
        raise ValueError(f"Scene replication check failed: {name}")


def _capture_state(env):
    return {key: _numpy(value) for key, value in dict(
        root_state_w=env.robot.data.root_state_w,
        joint_pos=env.robot.data.joint_pos,
        joint_vel=env.robot.data.joint_vel,
        phase=env.phase,
        episode_length=env.episode_length_buf,
        origins=env.env_origin(),
        scene_origins=env.scene.env_origins,
        cell_offset=env.cell_offset,
    ).items()}


def audit(args, app, report):
    import torch

    from morph_mimic.scripts.common import build_env
    from morph_mimic.utils.checkpoint import build_checkpoint_contract, file_hash

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    clip = args.clips[0]
    motion_path = args.motion_dir / f"{clip}.npz"
    with np.load(motion_path, allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        duration = (len(data["joint_pos"]) - 1) / fps
    env = None
    try:
        env, spec = build_env(
            1, args.robot, args.num_envs, ref_env_ratio=1.0, device=args.device,
            motion_dir=str(args.motion_dir), clips=args.clips, evaluation=True,
            evaluation_record_trajectory=True, evaluation_start_phase=0.0,
            replay=False, csi_enabled=False, seed=args.seed,
            episode_length_s=max(duration + 0.1, (args.steps + 2) * 0.02),
            dr_torso_mass=(0.0, 0.0), dr_gain_scale=(1.0, 1.0),
            dr_joint_bias_nonankle=0.0, dr_joint_bias_ankle=0.0, dr_delay=(0, 0),
            obs_noise_angvel=0.0, obs_noise_gravity=0.0, obs_noise_jpos=0.0, obs_noise_jvel=0.0,
            planner_scan_noise=0.0, tracker_scan_noise=0.0,
            eta_obs=0.0, eta_ref_obs=0.0, eta_rew=0.0,
        )
        contract = build_checkpoint_contract(env)
        report["provenance"] = dict(
            motion_sha256=file_hash(motion_path),
            contacts_sha256=file_hash(args.motion_dir / f"{clip}_contacts.npz"),
            sidecar_sha256=file_hash(args.motion_dir / f"{clip}.scene.json"),
            scene_fingerprint=env.motion_data.scene_bindings[clip].fingerprint,
            checkpoint_contract=contract,
        )
        report["termination_settings"] = {
            key: getattr(env.cfg, key) for key in
            ("grace_steps", "reference_grace_steps", "min_height_ref", "max_reference_root_error_m")
        }
        observation, _ = env.reset()
        # Forward writes refresh transform/ray buffers without integrating time.
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(spec.control_dt)
        origins = env.env_origin().clone()
        origins_np = _numpy(origins)
        report["origins_w_m"] = origins_np.tolist()
        _check(report, "distinct_origins", origins_np.shape == (args.num_envs, 3)
               and np.isfinite(origins_np).all() and len(np.unique(origins_np, axis=0)) == args.num_envs,
               count=int(len(np.unique(origins_np, axis=0))), expected=args.num_envs)
        scene_origin_error = float(np.max(np.abs(_numpy(env.scene.env_origins) - origins_np)))
        _check(report, "scene_origins_match", scene_origin_error <= args.tolerance,
               max_abs_error_m=scene_origin_error)
        _check(report, "zero_cell_offsets", float(env.cell_offset.abs().max()) == 0.0,
               max_abs_offset_m=float(env.cell_offset.abs().max()))
        _check(report, "initial_reference_selection", bool((env.clip_id == 0).all() and (env.phase == 0).all()),
               phases=_numpy(env.phase).tolist(), clip_ids=_numpy(env.clip_id).tolist())

        root_local = env.robot.data.root_pos_w - origins
        body_local = env.robot.data.body_link_pos_w - origins[:, None, :]
        reference = env._ref_sample()
        ref_root_local = reference["root_pos"] - origins
        ref_contact_local = env._ref_contact_pos(False) - origins[:, None, :]
        first_root_xy = env.motion_data.clips[clip].root_pos[0].clone()
        first_root_xy[2] = 0.0
        reference_body_world = torch.stack([
            env.motion_data.sample_body_pos(env.clip_id, env.phase, name)
            for name in env.robot.body_names
        ], dim=1) + origins[:, None, :] - first_root_xy
        reference_body_local = reference_body_world - origins[:, None, :]
        ids = torch.arange(args.num_envs, device=env.device)
        oracle_targets, oracle_flags = env._oracle_contacts(ids)
        static_spreads = dict(
            root_position_m=_spread(root_local), body_position_m=_spread(body_local),
            reference_root_position_m=_spread(ref_root_local),
            reference_body_position_m=_spread(reference_body_local),
            reference_contact_position_m=_spread(ref_contact_local),
            oracle_contact_target_m=_spread(oracle_targets), oracle_contact_flags=_spread(oracle_flags),
            joint_position_rad=_spread(env.robot.data.joint_pos),
            root_quaternion=_spread(env.robot.data.root_quat_w),
        )
        _check(report, "translated_state_and_reference", _finite(reference_body_world)
               and all(v <= args.tolerance for v in static_spreads.values()),
               maximum_component_spreads=static_spreads,
               initial_reference_body_fk_max_error_m=float(
                   (env.robot.data.body_link_pos_w - reference_body_world).norm(dim=-1).max()))
        raw_ray_hits = env.ray_caster.data.ray_hits_w
        ray_finite = _finite(raw_ray_hits)
        ray_local = raw_ray_hits - origins[:, None, :]
        relative_heights = env._planner_heights_relative()
        ray_height_spread = _spread(relative_heights)
        ray_translation_spread = _spread(ray_local) if ray_finite else None
        _check(report, "translated_ray_geometry", ray_finite
               and ray_height_spread <= args.tolerance and ray_translation_spread <= args.tolerance,
               all_raw_hits_finite=ray_finite, raw_hits_shape=list(raw_ray_hits.shape),
               max_relative_height_spread_m=ray_height_spread,
               max_ray_position_spread_m=ray_translation_spread)
        sensors = env.scene_contact_sensors
        filters = [list(sensor.cfg.filter_prim_paths_expr) for sensor in sensors]
        forces = env._contact_forces_w()
        _check(report, "terrain_filtered_contact_sensors", len(sensors) == 9
               and all(value == ["/World/terrain/terrain/mesh"] for value in filters)
               and tuple(forces.shape) == (args.num_envs, 9, 3) and _finite(forces),
               sensor_count=len(sensors), filters=filters, force_shape=list(forces.shape))

        report["step_statistics"] = []
        actions = torch.zeros(args.num_envs, 23, device=env.device)
        max_force = 0.0
        for step in range(args.steps):
            if not app.is_running():
                raise RuntimeError("Simulation application stopped before the requested smoke steps")
            observation, reward, terminated, truncated, info = env.step(actions)
            snapshot = info.get("final_tracking")
            if snapshot is None:
                raise RuntimeError("Missing pre-reset final_tracking snapshot")
            finite = all(_finite(value) for value in observation.values()) and _finite(reward)
            finite = finite and all(_finite(value) for value in snapshot.values())
            forces = snapshot["contact_force_w"]
            root_local = snapshot["root_pos_w"] - origins
            root_spread = float(torch.linalg.vector_norm(root_local - root_local[:1], dim=-1).max())
            joint_spread = _spread(snapshot["joint_pos"])
            phase_spread = _spread(snapshot["phase"])
            force_max = float(forces.norm(dim=-1).max())
            max_force = max(max_force, force_max)
            row = dict(step=step + 1, all_finite=finite,
                       maximum_root_distance_between_copies_m=root_spread,
                       maximum_joint_component_spread_rad=joint_spread,
                       phase_spread=phase_spread,
                       terminated_count=int(terminated.sum()), truncated_count=int(truncated.sum()),
                       max_force_norm_n=force_max,
                       root_tracking_error_max_m=float(snapshot["root_squared_error_m2"].sqrt().max()))
            report["step_statistics"].append(row)
            _check(report, f"step_{step + 1:04d}", finite
                   and root_spread <= args.synchronization_tolerance
                   and joint_spread <= args.synchronization_tolerance
                   and phase_spread <= 1e-6
                   and int(terminated.sum()) in (0, args.num_envs)
                   and int(truncated.sum()) in (0, args.num_envs), **row)
        report["steps_completed"] = len(report["step_statistics"])
        _check(report, "contact_force_signal_observed", max_force > 1.0,
               maximum_force_norm_n=max_force, threshold_n=1.0)

        # Subset reset must not move another robot. No physics is integrated
        # between these captures; forward() refreshes the written PhysX poses.
        before = _capture_state(env)
        reset_ids = torch.arange(0, args.num_envs, 2, device=env.device)
        untouched_ids = np.arange(1, args.num_envs, 2)
        env._reset_idx(reset_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.0)
        after = _capture_state(env)
        differences = {}
        for key in before:
            indices = slice(None) if key in ("origins", "scene_origins", "cell_offset") else untouched_ids
            differences[key] = float(np.max(np.abs(after[key][indices] - before[key][indices])))
        reset_phase_max = float(env.phase[reset_ids].abs().max())
        _check(report, "subset_reset_isolation", all(v <= 1e-6 for v in differences.values())
               and reset_phase_max <= 1e-6,
               reset_env_ids=_numpy(reset_ids).tolist(), untouched_env_ids=untouched_ids.tolist(),
               maximum_absolute_change_in_untouched_state=differences,
               reset_env_maximum_phase=reset_phase_max,
               refresh="write_data_to_sim; sim.forward; scene.update(0); no physical step")
        report["completed"] = True
        report["accepted"] = all(value["accepted"] for value in report["checks"].values())
    finally:
        if env is not None:
            env.close()


def main():
    args = _arguments()
    report = dict(
        schema=1, completed=False, accepted=False, checks={},
        method="Isaac bound-scene placement, sensing, synchronized zero-action smoke and subset-reset isolation",
        robot=args.robot, motion_dir=str(args.motion_dir), clips=args.clips,
        num_envs=args.num_envs, requested_steps=args.steps, device=args.device, seed=args.seed,
        static_tolerance_m=args.tolerance, synchronization_tolerance_m_rad=args.synchronization_tolerance,
        physical_skill_validated=False, policy_trained=False,
        scope="Replication infrastructure audit only; zero-action survival is not a skill execution result",
    )
    app = None
    try:
        from morph_mimic.launcher import launch
        app = launch(headless=True, device=args.device)
        audit(args, app, report)
    except Exception as error:
        report.update(accepted=False, error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        # Publish before Kit fast shutdown, retaining errors and partial stats.
        from morph_mimic.scripts.evaluate import _write_report
        _write_report(args.out, _json_safe(report))
        print(json.dumps({key: report.get(key) for key in
                          ("completed", "accepted", "steps_completed", "error")}), flush=True)
        status = 0 if report["completed"] and report["accepted"] else 1
        if app is not None:
            app.app.post_quit(status)
            from morph_mimic.launcher import close
            close(app)
        if status:
            raise SystemExit(status)


if __name__ == "__main__":
    main()
