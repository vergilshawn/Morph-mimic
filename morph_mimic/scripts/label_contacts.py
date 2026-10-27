"""Offline contact labeling: replay every clip with PD reference tracking and
record per-frame contact forces and link positions, then hindsight-label the
next-contact targets (1.5 s horizon, > 1 N, >= 10-step segments).

Labels require a complete replay and engineering tracking checks: maximum
root-position error <= 0.3 m and joint RMSE <= 0.5 rad by default. These
configurable acceptance thresholds are project checks, not paper thresholds.
Flat tiles expand to contain every selected root trajectory with 2 m margin.

Writes <clip>_contacts.npz next to each motion:
    contact_flags        (T, 9)  instantaneous segment-filtered contacts
    contact_targets      (T, 9, 3) next contact position in the body-yaw frame
    contact_target_flags (T, 9)  occupancy flags

Usage:
    conda run -n env_isaaclab_2 /home/user/IsaacLab/isaaclab.sh -p \
        morph_mimic/scripts/label_contacts.py --robot g1 [--clips walk1_subject1 ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from morph_mimic.contacts.labeling import hindsight_contacts
from morph_mimic.scripts.common import build_env
from morph_mimic.utils.specs import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="g1", choices=["g1", "mojia"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--motion_dir", type=str, default=None)
    parser.add_argument("--clips", nargs="*", default=None, help="clip names to label (default: all)")
    parser.add_argument("--episode_length_s", type=float, default=300.0)
    parser.add_argument("--min_height_ref", type=float, default=0.15)
    parser.add_argument("--record_max_steps", type=int, default=16000)
    parser.add_argument("--max_root_error", type=float, default=0.3,
                        help="engineering maximum root-position error in metres (not a paper threshold)")
    parser.add_argument("--max_joint_rmse", type=float, default=0.5,
                        help="engineering joint RMSE limit in radians (not a paper threshold)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing validated labels")
    parser.add_argument("--report", type=Path, help="save a new JSON report including rejected replays")
    parser.add_argument("--diagnose-only", action="store_true", help="validate physical replay without changing contact labels")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not np.isfinite(args.max_root_error) or args.max_root_error <= 0:
        parser.error("--max_root_error must be finite and positive")
    if not np.isfinite(args.max_joint_rmse) or args.max_joint_rmse <= 0:
        parser.error("--max_joint_rmse must be finite and positive")
    if args.report is not None and args.report.exists():
        parser.error(f"refusing to overwrite replay report: {args.report}")
    from morph_mimic.launcher import launch
    app = launch(headless=args.headless, device=args.device)
    try:
        _label(args, parser)
    finally:
        from morph_mimic.launcher import close

        close(app)


def _label(args, parser):
    from morph_mimic.envs.acp_env import REC_BASE, REC_BODY, REC_FORCE, REC_YAW

    # discover clips before building the env (one env per clip)
    from morph_mimic.utils.motion import validate_contact_labels
    from morph_mimic.utils.specs import RobotSpec
    from pathlib import Path

    spec = RobotSpec.load(args.robot)
    motion_dir = Path(args.motion_dir) if args.motion_dir is not None else Path(spec.motion_dir)
    if not motion_dir.is_absolute():
        motion_dir = PROJECT_ROOT / motion_dir
    all_names = sorted(p.stem for p in motion_dir.glob("*.npz") if not p.stem.endswith("_contacts"))
    clip_names = list(dict.fromkeys(args.clips if args.clips is not None else all_names))
    unknown = sorted(set(clip_names) - set(all_names))
    if unknown:
        parser.error(f"unknown clips: {', '.join(unknown)}")
    if not clip_names:
        parser.error("no motion clips selected")
    clip_meta = {}
    maximum_xy_displacement = np.zeros(2, dtype=np.float64)
    for name in clip_names:
        with np.load(motion_dir / f"{name}.npz", allow_pickle=False) as data:
            clip_meta[name] = (len(data["joint_pos"]), float(data["fps"].reshape(-1)[0]))
            if "root_pos" in data:
                root_xy = data["root_pos"][:, :2]
            else:
                body_names = data["body_names"].tolist()
                if spec.base_link not in body_names:
                    parser.error(f"{name}: base body {spec.base_link} is missing")
                root_xy = data["body_pos_w"][:, body_names.index(spec.base_link), :2]
            if not np.isfinite(root_xy).all():
                parser.error(f"{name}: root trajectory contains nonfinite values")
            maximum_xy_displacement = np.maximum(
                maximum_xy_displacement, np.abs(root_xy - root_xy[0]).max(axis=0)
            )
        if (motion_dir / f"{name}_contacts.npz").exists() and not args.overwrite and not args.diagnose_only:
            parser.error(f"{name}_contacts.npz already exists; use --overwrite to replace it")
        frames, fps = clip_meta[name]
        if not np.isclose(fps, 1.0 / spec.control_dt):
            parser.error(f"{name}: fps {fps} differs from control frequency {1.0 / spec.control_dt}")
        if frames > args.record_max_steps:
            parser.error(f"{name}: {frames} frames exceed --record_max_steps={args.record_max_steps}")
        if (frames - 1) / fps + 2 * spec.control_dt >= args.episode_length_s:
            parser.error(f"{name}: --episode_length_s must exceed the complete motion duration plus two steps")
    n_clips = len(clip_names)
    boundary_margin = 2.0
    terrain_size = tuple(np.maximum((9.0, 6.0), 2.0 * (maximum_xy_displacement + boundary_margin)).tolist())
    has_scene = any((motion_dir / f"{name}.scene.json").exists() for name in clip_names)
    print("[label] using bound obstacle scene" if has_scene else
          f"[label] flat terrain size {terrain_size}, {boundary_margin:g} m root-trajectory margin")
    env, spec = build_env(
        2, args.robot, num_envs=n_clips, device=args.device, replay=True,
        episode_length_s=args.episode_length_s, min_height_ref=args.min_height_ref,
        record_max_steps=args.record_max_steps,
        terrain_types=("flat",), ref_terrain_types=("flat",), num_terrain_levels=1,
        terrain_size=terrain_size,
        dr_torso_mass=(0.0, 0.0), dr_gain_scale=(1.0, 1.0),
        motion_dir=str(motion_dir), clips=clip_names,
    )
    terrain_size = env.cfg.terrain_size
    scenes = getattr(env.motion_data, "scene_bindings", {})
    label_method = "complete_reference_scene_pd_replay_vertical_force" if scenes else "complete_flat_ground_pd_replay_vertical_force"

    # Bind names to env IDs, independently of their IDs in the full dataset.
    # In particular, --clips walk1_subject1 has dataset ID > 0 but env ID 0.
    env_clip_names = dict(enumerate(clip_names))
    for i, name in enumerate(clip_names):
        cid = env.motion_data.clip_names.index(name)
        env.clip_id[i] = cid
        env.init_phase[i] = 0.0
        env.phase[i] = 0.0
        env.is_ref_env[i] = True
    env.cell_offset.zero_()
    # Set the actual simulated root/joints and clear records after assignment.
    env.reset()
    env.finished_episodes = []

    labeled = set()
    failed = {}
    clip_reports = {}
    processed = set()
    tracking_active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    root_error_max = torch.zeros(env.num_envs, device=env.device)
    joint_mse_sum = torch.zeros(env.num_envs, device=env.device)
    tracking_samples = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    steps = int(args.episode_length_s / spec.control_dt)
    print(f"[label] replaying {n_clips} clips ({steps} steps max each)")
    try:
        for step in range(steps):
            # Check the current physical state against the reference at its
            # current phase before step() can auto-reset finished episodes.
            reference = env._ref_sample()
            root_error = (env.robot.data.root_pos_w - reference["root_pos"]).norm(dim=-1)
            joint_mse = (env.robot.data.joint_pos[:, env.controlled_idx] - reference["joint_pos"]).square().mean(dim=-1)
            root_error = torch.where(torch.isfinite(root_error), root_error, torch.full_like(root_error, float("inf")))
            joint_mse = torch.where(torch.isfinite(joint_mse), joint_mse, torch.full_like(joint_mse, float("inf")))
            root_error_max[tracking_active] = torch.maximum(root_error_max[tracking_active], root_error[tracking_active])
            joint_mse_sum[tracking_active] += joint_mse[tracking_active]
            tracking_samples[tracking_active] += 1
            env.step(torch.zeros(env.num_envs, spec.n_controlled, device=env.device))
            finished = env.finished_episodes
            env.finished_episodes = []
            for ep in finished:
                name = env_clip_names.get(ep["env_id"])
                if name is None or name in processed:
                    continue
                processed.add(name)
                env_id = ep["env_id"]
                tracking_active[env_id] = False
                measured_root_error = float(root_error_max[env_id])
                measured_joint_rmse = float((joint_mse_sum[env_id] / tracking_samples[env_id].clamp(min=1)).sqrt())
                rec = ep["record"]
                n = rec.shape[0]
                expected_frames, fps = clip_meta[name]
                expected_cid = env.motion_data.clip_names.index(name)
                reasons = []
                if ep.get("clip_id") != expected_cid:
                    reasons.append("recorded clip ID differs from the assigned motion")
                if ep.get("terminated", True):
                    reasons.append("replay terminated before successful completion")
                if not np.isclose(ep.get("init_phase", -1.0), 0.0, atol=1e-6):
                    reasons.append("replay did not start at phase zero")
                if ep.get("end_phase", 0.0) < 1.0 - 1e-5:
                    reasons.append(f"replay stopped at phase {ep.get('end_phase', 0.0):.5f}")
                if n != expected_frames:
                    reasons.append(f"record has {n} frames; source motion has {expected_frames}")
                if not torch.isfinite(rec).all():
                    reasons.append("record contains nonfinite values")
                if not np.isfinite(measured_root_error) or measured_root_error > args.max_root_error:
                    reasons.append(f"root error {measured_root_error:.4f} m exceeds engineering limit {args.max_root_error:g} m")
                if not np.isfinite(measured_joint_rmse) or measured_joint_rmse > args.max_joint_rmse:
                    reasons.append(f"joint RMSE {measured_joint_rmse:.4f} rad exceeds engineering limit {args.max_joint_rmse:g} rad")
                clip_reports[name] = dict(
                    accepted=not reasons, rejection_reasons=reasons, source_num_frames=expected_frames,
                    recorded_num_frames=n, init_phase=float(ep.get("init_phase", -1)),
                    end_phase=float(ep.get("end_phase", 0)), terminated=bool(ep.get("terminated", True)),
                    tracking_root_max_error_m=measured_root_error if np.isfinite(measured_root_error) else None,
                    tracking_joint_rmse_rad=measured_joint_rmse if np.isfinite(measured_joint_rmse) else None,
                    source_sha256=hashlib.sha256((motion_dir / f"{name}.npz").read_bytes()).hexdigest(),
                    scene_fingerprint=scenes[name].fingerprint if name in scenes else None,
                )
                if reasons:
                    failed[name] = "; ".join(reasons)
                    print(f"[label] REJECTED {name}: {failed[name]}")
                    continue
                forces = rec[:, REC_FORCE : REC_FORCE + 9]
                body = rec[:, REC_BODY : REC_BODY + 27].reshape(n, 9, 3)
                base = rec[:, REC_BASE : REC_BASE + 3]
                yaw = rec[:, REC_YAW]
                targets, flags, c_inst = hindsight_contacts(
                    forces, body, base, yaw, horizon=75, min_seg=10, force_thr=1.0
                )
                labels = dict(
                    contact_flags=c_inst.cpu().numpy(),
                    contact_targets=targets.cpu().numpy(),
                    contact_target_flags=flags.cpu().numpy(),
                    fps=np.array([fps]),
                    source_num_frames=np.array(expected_frames),
                    source_init_phase=np.array(ep["init_phase"]),
                    source_end_phase=np.array(ep["end_phase"]),
                    source_sha256=np.array(hashlib.sha256((motion_dir / f"{name}.npz").read_bytes()).hexdigest()),
                    body_names=np.array(env.contact_link_names),
                    label_method=np.array(label_method),
                    force_threshold_n=np.array(1.0),
                    min_segment_steps=np.array(10),
                    horizon_steps=np.array(75),
                    tracking_root_max_error_m=np.array(measured_root_error),
                    tracking_joint_rmse_rad=np.array(measured_joint_rmse),
                    tracking_root_max_error_limit_m=np.array(args.max_root_error),
                    tracking_joint_rmse_limit_rad=np.array(args.max_joint_rmse),
                    tracking_sample_count=np.array(int(tracking_samples[env_id])),
                    tracking_threshold_basis=np.array("project_engineering_acceptance_not_paper_thresholds"),
                    terrain_size_m=np.array(terrain_size),
                    terrain_root_margin_m=np.array(boundary_margin),
                )
                if name in scenes:
                    labels["scene_fingerprint"] = np.array(scenes[name].fingerprint)
                    labels["force_source"] = np.array("per_body_terrain_filtered_normal_forces")
                validate_contact_labels(labels, expected_frames, fps, name)
                out = motion_dir / f"{name}_contacts.npz"
                if not args.diagnose_only:
                    temporary = out.with_suffix(".npz.tmp")
                    try:
                        with temporary.open("wb") as stream:
                            np.savez(stream, **labels)
                        os.replace(temporary, out)
                    finally:
                        temporary.unlink(missing_ok=True)
                labeled.add(name)
                n_contact_frames = int(c_inst.sum())
                print(f"[label] {name}: {n} frames, {n_contact_frames} contact frames, "
                      f"root max error {measured_root_error:.4f} m, joint RMSE {measured_joint_rmse:.4f} rad -> "
                      f"{'diagnostic only' if args.diagnose_only else out.name}")
            if len(processed) >= n_clips:
                break
            if step % 2000 == 0 and step > 0:
                print(f"[label] ... step {step}/{steps}, {len(labeled)}/{n_clips} labeled")
    finally:
        env.close()
    print(f"[label] done: {len(labeled)}/{n_clips} clips labeled")
    missing = set(clip_names) - labeled
    if args.report is not None:
        document = dict(
            schema=1, method=label_method, backend="IsaacLab/PhysX", robot=args.robot,
            motion_dir=str(motion_dir), terrain_size_m=list(terrain_size),
            accepted=not missing, labeled_clips=sorted(labeled), clips=clip_reports,
            root_max_error_limit_m=args.max_root_error, joint_rmse_limit_rad=args.max_joint_rmse,
            thresholds_basis="project_engineering_not_paper_thresholds",
            unprocessed_clips=sorted(set(clip_names) - set(clip_reports)),
            training_contact_labels_generated=bool(labeled) and not args.diagnose_only,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=args.report.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.link(temporary, args.report)
            finally:
                temporary.unlink(missing_ok=True)
    if missing:
        for name in sorted(missing):
            print(f"[label] incomplete {name}: {failed.get(name, 'no complete episode was recorded')}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
