"""Deterministic complete-reference policy evaluation in its bound scene.

Stage 1 evaluates the actor; Stage 2 evaluates actor and learned planner with
full planner control. References without a bound scene use flat terrain.
This does not evaluate Stage 3 goal reaching, unseen-terrain generalization,
physical contact quality, or hardware performance.

Example (inside the Isaac Lab environment)::

    python -m morph_mimic.scripts.evaluate --robot g1 --stage 1 \\
        --resume runs/g1_stage1/seed0/final.pt --motion-dir datasets/g1_23dof_smoke \\
        --clips walk1_subject1 --out runs/g1_reference_evaluation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from morph_mimic.utils.evaluation import ReferenceTrackingMetrics, TrackingTrajectory, reference_outcome
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "mojia"), required=True)
    parser.add_argument("--stage", choices=(1, 2), type=int, required=True,
                        help="reference tracking only; Stage 3 goal evaluation is not implemented")
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--motion-dir", "--motion_dir", dest="motion_dir", type=Path)
    parser.add_argument("--clips", nargs="+", help="selected clips; default is all clips in motion-dir")
    parser.add_argument("--planner", choices=("acp", "mlp"), default="acp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--held-joint-mode", choices=("nominal_pd", "hard_limits_at_nominal"),
                        default="hard_limits_at_nominal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int,
                        help="optional per-clip budget; short budgets are reported as incomplete")
    parser.add_argument("--start-phase", type=float, default=0.0,
                        help="diagnostic reset phase in [0,1); nonzero evaluates a partial reference only")
    parser.add_argument("--trajectory", type=Path,
                        help="optional pre-reset physical endpoint NPZ; requires exactly one selected clip")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not math.isfinite(args.start_phase) or not 0 <= args.start_phase < 1:
        parser.error("--start-phase must be finite and lie in [0,1)")
    if not args.resume.is_file():
        parser.error(f"checkpoint not found: {args.resume}")
    if args.out.exists():
        parser.error(f"refusing to overwrite evaluation report: {args.out}")
    if args.trajectory is not None:
        if args.trajectory.suffix != ".npz":
            parser.error("--trajectory must have an .npz suffix")
        if args.trajectory.exists():
            parser.error(f"refusing to overwrite trajectory: {args.trajectory}")
        if args.trajectory.resolve() == args.out.resolve():
            parser.error("report and trajectory paths must differ")
    return args


def _clip_metadata(directory, names, spec):
    paths = {p.stem: p for p in sorted(directory.glob("*.npz")) if not p.stem.endswith("_contacts")}
    selected = list(dict.fromkeys(names)) if names is not None else list(paths)
    if not selected:
        raise ValueError(f"No motion clips selected in {directory}")
    missing = set(selected) - set(paths)
    if missing:
        raise ValueError(f"Unknown motion clips: {', '.join(sorted(missing))}")
    metadata = []
    for name in selected:
        path = paths[name]
        with np.load(path, allow_pickle=False) as data:
            fps_values = np.asarray(data["fps"]).reshape(-1)
            if fps_values.size != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0:
                raise ValueError(f"{name}: expected one finite positive fps")
            fps = float(fps_values[0])
            frames = len(data["joint_pos"])
            if frames < 2 or not np.isclose(fps * spec.control_dt, 1.0):
                raise ValueError(f"{name}: invalid frame count or fps/control frequency mismatch")
            if "root_pos" in data:
                xy = data["root_pos"][:, :2]
            else:
                body_id = data["body_names"].tolist().index(spec.base_link)
                xy = data["body_pos_w"][:, body_id, :2]
            if xy.shape != (frames, 2) or not np.isfinite(xy).all():
                raise ValueError(f"{name}: invalid reference root trajectory")
            terrain_size = tuple(np.maximum((9.0, 6.0), 2 * (np.abs(xy - xy[0]).max(axis=0) + 2.0)).tolist())
        metadata.append(dict(name=name, frames=frames, fps=fps, duration_s=(frames - 1) / fps,
                             source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             terrain_size=terrain_size))
    return metadata


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".json.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        # Publishing without overwrite also protects concurrent evaluation reports.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@torch.no_grad()
def _evaluate_clip(args, checkpoint, metadata, actor, planner, app):
    # The simulator is launched before this imports environment-dependent code.
    from morph_mimic.scripts.common import build_env
    from morph_mimic.utils.checkpoint import validate_checkpoint_contract

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    duration = metadata["duration_s"]
    start_phase = getattr(args, "start_phase", 0.0)
    spec = RobotSpec.load(args.robot)
    full_steps = round(duration / spec.control_dt)
    steps_required = math.ceil((1.0 - start_phase) * full_steps - 1e-9)
    # Two spare controls keep the episode timeout separate from end-of-clip.
    budget = steps_required + 2 if args.max_steps is None else args.max_steps
    env, _ = build_env(
        args.stage, args.robot, 1, ref_env_ratio=1.0, device=args.device,
        motion_dir=str(args.motion_dir), clips=[metadata["name"]],
        training_profile=checkpoint.get("contract", {}).get("training", {}).get("training_profile", "acp_partial_dr"),
        held_joint_mode=getattr(args, "held_joint_mode", "hard_limits_at_nominal"),
        evaluation=True, replay=False, seed=args.seed, csi_enabled=False,
        evaluation_start_phase=start_phase,
        evaluation_record_trajectory=getattr(args, "trajectory", None) is not None,
        episode_length_s=duration + 2 * spec.control_dt,
        terrain_types=("flat",), ref_terrain_types=("flat",), num_terrain_levels=1,
        terrain_size=metadata["terrain_size"],
        dr_torso_mass=(0.0, 0.0), dr_gain_scale=(1.0, 1.0),
        dr_joint_bias_nonankle=0.0, dr_joint_bias_ankle=0.0, dr_delay=(0, 0),
        obs_noise_angvel=0.0, obs_noise_gravity=0.0, obs_noise_jpos=0.0, obs_noise_jvel=0.0,
        eta_obs=float(args.stage > 1), eta_ref_obs=float(args.stage > 1),
        eta_rew=float(args.stage > 1),
    )
    try:
        contract_verified = bool(validate_checkpoint_contract(checkpoint, env, planner, mode="evaluate"))
        metrics = ReferenceTrackingMetrics()
        trajectory_path = getattr(args, "trajectory", None)
        trajectory = TrackingTrajectory(spec.control_dt) if trajectory_path is not None else None
        # The expanded tile is sized about its centre, not a training grid slot.
        env.cell_offset.zero_()
        observation, _ = env.reset()
        if int(env.clip_id[0]) != 0 or abs(float(env.phase[0]) - start_phase) > 1e-6:
            raise RuntimeError("Evaluation reset did not select the requested clip and start phase")
        terminated = truncated = reached_end = interrupted = False
        phase = start_phase
        total_reward = 0.0
        for _ in range(budget):
            if not app.is_running():
                interrupted = True
                break
            if planner is not None:
                points, heights, proprio_goal = env.get_planner_obs()
                _, occupancy, contacts = planner(points, heights, proprio_goal)
                env.set_predicted_contacts(contacts, occupancy)
                observation = env.refresh_observations()
            action = actor(observation["policy"])
            observation, reward, term, trunc, info = env.step(action)
            snapshot = info.get("final_tracking")
            if snapshot is None:
                raise RuntimeError("Evaluation requires pre-reset final_tracking snapshots from AcpEnv")
            if int(snapshot["clip_id"][0]) != 0:
                raise RuntimeError("Reference clip changed during evaluation")
            metrics.update(snapshot)
            if trajectory is not None:
                trajectory.update(snapshot, action=action, reward=reward)
            total_reward += float(reward[0])
            phase = float(snapshot["phase"][0])
            terminated = bool(term[0])
            truncated = bool(trunc[0])
            reached_end = bool(snapshot["reached_end"][0])
            # Do not consume the automatically reset episode's state or reward.
            if terminated or truncated or reached_end:
                break
        if not metrics.samples:
            raise RuntimeError("Simulation stopped before the first policy control step")
        result = dict(
            clip=metadata["name"], source_sha256=metadata["source_sha256"],
            source_frames=metadata["frames"], duration_s=duration,
            expected_control_steps=steps_required, steps=metrics.samples,
            full_reference_control_steps=full_steps,
            start_phase=start_phase, episode_reward=total_reward,
            contract_verified=contract_verified,
            held_joint_mode=getattr(env.cfg, "held_joint_mode", "nominal_pd") if hasattr(env, "cfg") else "unknown",
            body_names=list(env.contact_link_names),
            **reference_outcome(terminated=terminated, truncated=truncated,
                                reached_end=reached_end, phase=phase, interrupted=interrupted,
                                start_phase=start_phase),
            **metrics.summary(),
        )
        if trajectory is not None:
            result["physical_diagnostics"] = trajectory.summary()
            result["trajectory"] = str(trajectory_path.resolve())
            trajectory.write(trajectory_path, metadata=dict(
                schema=1, robot=args.robot, stage=args.stage, clip=metadata["name"],
                checkpoint=str(args.resume.resolve()),
                checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                source_sha256=metadata["source_sha256"], control_dt_s=spec.control_dt,
                joint_names=list(spec.controlled_joint_names),
                all_joint_names=list(env.robot.data.joint_names),
                held_joint_names=[env.robot.data.joint_names[i] for i in env.held_idx.tolist()],
                contact_body_names=list(env.contact_link_names),
                quaternion_convention="world_from_link_wxyz", positions_frame="world",
                sampling="Post-physics control endpoints before any automatic reset; initial reset state excluded",
                contact_force_sampling="Current control endpoint; measured flags may use contact-force history",
                result=result,
            ))
            result["trajectory_sha256"] = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
        return result
    finally:
        env.close()


def main():
    args = _arguments()
    start_phase = getattr(args, "start_phase", 0.0)
    trajectory_path = getattr(args, "trajectory", None)
    spec = RobotSpec.load(args.robot)
    directory = args.motion_dir if args.motion_dir is not None else Path(spec.motion_dir)
    args.motion_dir = directory if directory.is_absolute() else PROJECT_ROOT / directory
    metadata = _clip_metadata(args.motion_dir, args.clips, spec)
    if trajectory_path is not None and len(metadata) != 1:
        raise ValueError("--trajectory requires exactly one selected clip; use --clips")
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    if checkpoint.get("robot") != args.robot or checkpoint.get("stage") != args.stage:
        raise ValueError("Evaluation requires a checkpoint with matching robot and stage metadata")
    if args.stage == 2 and checkpoint.get("planner") is None:
        raise ValueError("Stage 2 evaluation requires a trained planner")
    from morph_mimic.networks.acp import ACPPlanner, MLPPlanner
    from morph_mimic.networks.tracker import TrackerActor

    actor = TrackerActor().to(args.device).eval()
    actor.load_state_dict(checkpoint["ppo"]["actor"])
    planner = None
    if args.stage == 2:
        planner = (ACPPlanner if args.planner == "acp" else MLPPlanner)().to(args.device).eval()
        planner.load_state_dict(checkpoint["planner"])
    from morph_mimic.launcher import launch

    app = launch(headless=args.headless, device=args.device)
    try:
        motions = []
        for clip in metadata:
            result = _evaluate_clip(args, checkpoint, clip, actor, planner, app)
            motions.append(result)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            if result["interrupted"]:
                break
        full = (start_phase == 0.0 and len(motions) == len(metadata)
                and not any(m["budget_limited"] for m in motions))
        report = dict(
            robot=args.robot, stage=args.stage, checkpoint=str(args.resume.resolve()),
            checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
            motion_dir=str(args.motion_dir.resolve()), seed=args.seed, device=args.device,
            planner=args.planner if planner is not None else None,
            scope="Stage 1/2 reference tracking in each bound scene, otherwise flat; Stage 3 goal evaluation is not implemented",
            completion_definition="Phase zero to reference phase one without physical termination; tracking accuracy is reported separately",
            position_metric_definition="sqrt(mean(sum_xyz(error**2))); global world-frame positions in metres",
            joint_metric_definition="sqrt(mean_time_and_controlled_joints(joint_error_rad**2))",
            orientation_metric_definition="RMS geodesic root-orientation angle in radians",
            sampling="Control-step endpoints, including the final physical state before automatic reset",
            deterministic_settings="One clip per environment, requested start phase, no CSI, observation noise, domain randomization, action delay or scan bias; Stage 2 planner control probability one",
            start_phase=start_phase, evaluation_is_partial=start_phase != 0.0,
            physical_contact_metrics_available=trajectory_path is not None,
            expected_clips=len(metadata), evaluated_clips=len(motions), evaluation_is_full=full,
            contract_verified=all(m["contract_verified"] for m in motions),
            completion_rate=(sum(m["completed"] for m in motions) / len(metadata)
                             if start_phase == 0.0 else None),
            segment_completion_rate=sum(m.get("segment_completed", m["completed"]) for m in motions) / len(metadata),
            failures=sum(m["failure"] for m in motions), motions=motions,
        )
        # Kit fast shutdown can terminate Python, so publish before close().
        _write_report(args.out, report)
        print(f"Saved {args.out}: {sum(m['completed'] for m in motions)}/{len(metadata)} completed; full={full}")
    finally:
        from morph_mimic.launcher import close

        close(app)


if __name__ == "__main__":
    main()
