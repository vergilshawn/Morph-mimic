"""Reference contact candidates from collision geometry, without force replay.

All distances are signed distances between the actual compiled robot and
environment collision geoms. Negative values mean overlap. These candidates
are geometric supervision, not measured forces, dynamic support, or a claim
that the motion can be tracked. The default 1 cm proximity and 2 cm maximum
penetration limits are project engineering choices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


METHOD = "reference_geometry_mujoco_collision_distance_v3"


@dataclass(frozen=True)
class GeometryThresholds:
    contact_distance_m: float = 0.01
    max_penetration_m: float = 0.02
    max_self_penetration_m: float = 0.02
    min_segment_frames: int = 1
    horizon_s: float = 1.5
    min_any_contact_fraction: float = 0.8
    max_no_contact_gap_s: float = 0.3
    min_foot_contact_s: float = 0.1
    min_hand_segment_s: float = 0.1
    require_hand_contact: bool = False
    footprint_x_m: tuple[float, float] = (-0.8, 0.8)
    footprint_y_m: tuple[float, float] = (-0.5, 0.5)

    def validate(self):
        for field in ("contact_distance_m", "max_penetration_m", "max_self_penetration_m", "horizon_s", "max_no_contact_gap_s",
                      "min_foot_contact_s", "min_hand_segment_s"):
            if not np.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not isinstance(self.min_segment_frames, int) or self.min_segment_frames < 1:
            raise ValueError("min_segment_frames must be a positive integer")
        if not isinstance(self.require_hand_contact, bool):
            raise ValueError("require_hand_contact must be boolean")
        if not 0 < self.min_any_contact_fraction <= 1:
            raise ValueError("min_any_contact_fraction must be in (0,1]")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _segments(flags):
    padded = np.pad(np.asarray(flags, dtype=np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def filter_segments(flags, min_frames=1):
    """Optional engineering denoising; this is not force-based hindsight."""
    result = np.asarray(flags, dtype=bool).copy()
    for body in range(result.shape[1]):
        for start, stop in _segments(result[:, body]):
            if stop - start < min_frames:
                result[start:stop, body] = False
    return result


def sample_collision_geometry(model, scene_geom_ids, body_ids, qpos_frames, distance_cap_m=5.0):
    """Set each reference pose once and query collision distances without steps.

    ``scene_surface_point_w`` is the scene-side point of the closest geom pair;
    ``body_origin_w`` remains a separate quantity for tracker-compatible targets.
    Every active robot collision participates in the global penetration gate,
    including the torso, head, hips, and wrists outside the nine label columns.
    """
    import mujoco

    scene_ids = sorted(int(i) for i in scene_geom_ids)
    if not scene_ids:
        raise ValueError("reference geometry needs scene collision geoms")
    body_ids = np.asarray(body_ids, dtype=int)
    column = {int(body): index for index, body in enumerate(body_ids)}
    robot_ids = [i for i in range(model.ngeom) if i not in scene_geom_ids and model.geom_bodyid[i] > 0
                 and (model.geom_contype[i] or model.geom_conaffinity[i])]
    pairs = [(robot, scene) for robot in robot_ids for scene in scene_ids
             if ((model.geom_contype[robot] & model.geom_conaffinity[scene])
                 or (model.geom_contype[scene] & model.geom_conaffinity[robot]))]
    body_geom_count = [sum(model.geom_bodyid[g] == body for g in robot_ids) for body in body_ids]
    if not pairs or not all(body_geom_count):
        raise ValueError("every contact body must own an active collision geom, and scene pairs must collide")
    qpos_frames = np.asarray(qpos_frames, dtype=float)
    if qpos_frames.ndim != 2 or qpos_frames.shape[1] != model.nq or not np.isfinite(qpos_frames).all():
        raise ValueError("reference qpos must be a finite frame-by-model-nq matrix")
    frames, bodies = len(qpos_frames), len(body_ids)
    samples = dict(
        signed_distance_m=np.full((frames, bodies), distance_cap_m),
        scene_surface_point_w=np.zeros((frames, bodies, 3)),
        robot_surface_point_w=np.zeros((frames, bodies, 3)),
        closest_robot_geom_id=np.full((frames, bodies), -1, dtype=int),
        closest_scene_geom_id=np.full((frames, bodies), -1, dtype=int),
        body_origin_w=np.empty((frames, bodies, 3)),
        all_robot_min_scene_distance_m=np.full(frames, distance_cap_m),
    )
    data, segment = mujoco.MjData(model), np.empty(6)
    for frame, state in enumerate(qpos_frames):
        data.qpos[:], data.qvel[:] = state, 0
        mujoco.mj_forward(model, data)
        samples["body_origin_w"][frame] = data.xpos[body_ids]
        for robot, scene in pairs:
            distance = float(mujoco.mj_geomDistance(model, data, robot, scene, distance_cap_m, segment))
            samples["all_robot_min_scene_distance_m"][frame] = min(
                samples["all_robot_min_scene_distance_m"][frame], distance)
            body_column = column.get(int(model.geom_bodyid[robot]))
            if body_column is not None and distance < samples["signed_distance_m"][frame, body_column]:
                samples["signed_distance_m"][frame, body_column] = distance
                samples["robot_surface_point_w"][frame, body_column] = segment[:3]
                samples["scene_surface_point_w"][frame, body_column] = segment[3:]
                samples["closest_robot_geom_id"][frame, body_column] = robot
                samples["closest_scene_geom_id"][frame, body_column] = scene
    samples["surface_point_valid"] = samples["closest_scene_geom_id"] >= 0
    samples["penetration_m"] = np.maximum(0, -samples["signed_distance_m"])
    samples["all_robot_penetration_m"] = np.maximum(0, -samples["all_robot_min_scene_distance_m"])
    return samples


def geometry_hindsight(flags, body_origins, root_positions, root_quaternions, fps, thresholds):
    """Future onset targets at body origins, in the query root-yaw frame.

    Z remains absolute world height, matching the current tracker's convention.
    Surface points are retained in diagnostics and are never substituted for a
    body-origin anchor without changing that supervision contract.
    """
    frames, bodies = flags.shape
    targets = np.zeros((frames, bodies, 3), dtype=np.float32)
    target_flags = np.zeros((frames, bodies), dtype=bool)
    shifted = np.zeros_like(flags)
    shifted[1:] = flags[:-1]
    onset = flags & ~shifted
    w, x, y, z = np.asarray(root_quaternions).T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    horizon = int(np.floor(thresholds.horizon_s * fps + 1e-9))
    query = np.arange(frames)
    for body in range(bodies):
        indices = np.flatnonzero(onset[:, body])
        if not len(indices):
            continue
        insertion = np.searchsorted(indices, query + 1)
        future = indices[np.minimum(insertion, len(indices) - 1)]
        within = (insertion < len(indices)) & (future <= query + horizon)
        positions = body_origins[future, body]
        delta = positions - root_positions
        local = np.stack((np.cos(yaw) * delta[:, 0] + np.sin(yaw) * delta[:, 1],
                          -np.sin(yaw) * delta[:, 0] + np.cos(yaw) * delta[:, 1], positions[:, 2]), axis=-1)
        within &= ((local[:, 0] >= thresholds.footprint_x_m[0]) & (local[:, 0] <= thresholds.footprint_x_m[1])
                   & (local[:, 1] >= thresholds.footprint_y_m[0]) & (local[:, 1] <= thresholds.footprint_y_m[1]))
        targets[:, body] = np.where(within[:, None], local, 0)
        target_flags[:, body] = within
    return targets, target_flags


def assess_candidates(samples, body_names, fps, category, thresholds):
    """Conservative geometric gates for a continuous-support box-climbing seed."""
    thresholds.validate()
    raw_flags = samples["signed_distance_m"] <= thresholds.contact_distance_m
    flags = filter_segments(raw_flags, thresholds.min_segment_frames)
    any_contact = flags.any(axis=1)
    gaps = _segments(~any_contact)
    max_gap = max((stop - start for start, stop in gaps), default=0) / fps
    coverage = float(any_contact.mean())
    max_penetration = float(samples["all_robot_penetration_m"].max())
    reasons = []
    if max_penetration > thresholds.max_penetration_m:
        reasons.append("robot-scene penetration exceeds the engineering limit")
    if coverage < thresholds.min_any_contact_fraction:
        reasons.append("too few frames have a geometric contact candidate")
    if max_gap > thresholds.max_no_contact_gap_s:
        reasons.append("geometric contact-free interval is too long for this climbing seed")
    feet = [i for i, name in enumerate(body_names) if "ankle" in name or "foot" in name]
    hands = [i for i, name in enumerate(body_names) if "wrist" in name or "hand" in name]
    # These are geometric coverage checks, not a force/balance stability test.
    for index in feet:
        if flags[:, index].sum() / fps < thresholds.min_foot_contact_s:
            reasons.append(f"insufficient geometric foot coverage: {body_names[index]}")
    longest_hand_segment = max((stop - start for i in hands for start, stop in _segments(flags[:, i])), default=0) / fps
    # ACP also steps onto low boxes with feet alone. A category name must not
    # force an arm-end contact; require it only for an explicitly chosen task.
    if thresholds.require_hand_contact and longest_hand_segment < thresholds.min_hand_segment_s:
        reasons.append("explicit arm-support requirement lacks a persistent geometric hand-contact candidate")
    schedule = {}
    for index, name in enumerate(body_names):
        segments = _segments(flags[:, index])
        schedule[name] = dict(
            contact_fraction=float(flags[:, index].mean()), contact_frames=int(flags[:, index].sum()),
            segment_count=len(segments),
            segments=[dict(start_frame=start, stop_frame_exclusive=stop,
                           start_s=start / fps, duration_s=(stop - start) / fps) for start, stop in segments],
            minimum_signed_distance_m=float(samples["signed_distance_m"][:, index].min()),
            maximum_penetration_m=float(samples["penetration_m"][:, index].max()),
        )
    report = dict(
        schema=1, method=METHOD, accepted=not reasons, rejection_reasons=reasons,
        thresholds=asdict(thresholds), threshold_basis="explicit project engineering choices, not paper force-label thresholds",
        instantaneous_contact_definition="minimum signed robot collision-to-scene distance <= contact_distance_m",
        force_threshold_n=None, forces_used=False, physics_validated=False,
        reference_pose_overwritten_each_frame=True, physical_tracking_required_for_this_method=False,
        contact_segment_filter="optional geometric denoising only; default one frame, no force-segment rule",
        any_contact_fraction=coverage, maximum_no_contact_gap_s=max_gap,
        robot_scene_max_penetration_m=max_penetration,
        geometric_coverage_is_dynamic_support=False, contact_body_names=list(body_names), schedule=schedule,
        label_target_anchor="future contact-onset body-frame origin, not scene surface; query root-yaw coordinates, absolute Z",
    )
    return report, flags


def diagnose_reference(motion_path, scene_path, robot="g1", thresholds=None, compare_raw=None):
    from morph_mimic.contacts.self_collision import sample_self_contacts
    from morph_mimic.diagnostics.validate_scene_reference import build_model, _body_columns, _joint_indices
    from morph_mimic.utils.motion import validate_motion_arrays, validate_contact_labels, validate_contact_body_order
    from morph_mimic.utils.reference_scene import load_reference_scene
    from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec
    import mujoco

    thresholds = thresholds or GeometryThresholds()
    thresholds.validate()
    motion_path, scene_path = Path(motion_path).resolve(), Path(scene_path).resolve()
    spec = RobotSpec.load(robot)
    with np.load(motion_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_motion_arrays(arrays, spec, motion_path)
    fps = float(np.asarray(arrays["fps"]).item())
    if not np.isclose(1 / fps, spec.control_dt):
        raise ValueError("reference rate must match the control interval")
    scene = load_reference_scene(scene_path, motion_path=motion_path)
    model_path = PROJECT_ROOT / spec.runtime_assets["mjcf"]
    model, scene_ids = build_model(model_path, scene, spec.physics_dt)
    body_names, body_ids = _body_columns(model, spec)
    qids, _ = _joint_indices(model, spec)

    def sample(arrays):
        order = [arrays["joint_names"].tolist().index(name) for name in spec.joint_names]
        base = arrays["body_names"].tolist().index(spec.base_link)
        frames = len(arrays["joint_pos"])
        states = np.zeros((frames, model.nq))
        states[:, :3], states[:, 3:7] = arrays["body_pos_w"][:, base], arrays["body_quat_w"][:, base]
        states[:, qids] = arrays["joint_pos"][:, order]
        measured = sample_collision_geometry(model, scene_ids, body_ids, states)
        return measured, states[:, :3], states[:, 3:7]

    samples, roots, quats = sample(arrays)
    category = str(np.asarray(arrays.get("category", "other")).item())
    report, flags = assess_candidates(samples, body_names, fps, category, thresholds)
    full_joint_order = [arrays["joint_names"].tolist().index(name) for name in spec.joint_names]
    self_states = np.zeros((len(roots), model.nq))
    self_states[:, :3], self_states[:, 3:7] = roots, quats
    self_states[:, qids] = arrays["joint_pos"][:, full_joint_order]
    self_report = sample_self_contacts(model, self_states, scene_ids, fps, thresholds.max_self_penetration_m)
    report["self_collision_check"] = {key: value for key, value in self_report.items() if key != "frames"}
    report["self_collision_max_penetration_m"] = self_report["maximum_penetration_m"]
    report["self_collision_gate_method"] = "independent mj_collision with unchanged model masks/excludes/parent/weld filters"
    if not self_report["accepted"]:
        report["accepted"] = False
        report["rejection_reasons"].append("robot self-collision penetration exceeds the engineering limit")
    samples["self_collision_max_penetration_m"] = np.array([frame["maximum_penetration_m"] for frame in self_report["frames"]])
    targets, target_flags = geometry_hindsight(flags, samples["body_origin_w"], roots, quats, fps, thresholds)
    report.update(robot=robot, category=category, num_frames=len(flags), fps=fps,
                  duration_s=(len(flags) - 1) / fps, source_motion=str(motion_path), source_sha256=_hash(motion_path),
                  scene_path=str(scene_path), scene_sha256=_hash(scene_path), scene_fingerprint=scene.fingerprint,
                  reference_model=str(model_path), reference_model_sha256=_hash(model_path),
                  robot_spec_sha256=_hash(PROJECT_ROOT / "assets" / robot / "robot_spec.json"),
                  mujoco_version=mujoco.__version__, training_contact_labels_generated=False)
    samples.update(contact_candidate_flags=flags, body_names=np.array(body_names),
                   source_sha256=np.array(report["source_sha256"]), scene_fingerprint=np.array(scene.fingerprint),
                   diagnostic_only=np.array(True), time_s=np.arange(len(flags)) / fps)
    if compare_raw is not None:
        compare_raw = Path(compare_raw).resolve()
        with np.load(compare_raw, allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        validate_motion_arrays(original, spec, compare_raw, require_nominal_held=False)
        if len(original["joint_pos"]) != len(flags) or float(np.asarray(original["fps"]).item()) != fps:
            raise ValueError("raw comparison needs the same frame count and sampling rate")
        raw_samples, raw_roots, raw_quats = sample(original)
        if not np.allclose(raw_roots, roots, atol=1e-6) or not np.allclose(np.abs(np.sum(raw_quats * quats, axis=-1)), 1, atol=1e-5):
            raise ValueError("raw comparison must share the same root trajectory and scene coordinates")
        raw_report, raw_flags = assess_candidates(raw_samples, body_names, fps, category, thresholds)
        report["raw_comparison"] = dict(
            motion_path=str(compare_raw), source_sha256=_hash(compare_raw), accepted=raw_report["accepted"],
            rejection_reasons=raw_report["rejection_reasons"],
            any_contact_fraction=raw_report["any_contact_fraction"],
            robot_scene_max_penetration_m=raw_report["robot_scene_max_penetration_m"],
            per_body={name: dict(
                raw_contact_fraction=float(raw_flags[:, i].mean()), projected_contact_fraction=float(flags[:, i].mean()),
                raw_contact_frames_lost=int(np.count_nonzero(raw_flags[:, i] & ~flags[:, i])),
                projected_contact_frames_gained=int(np.count_nonzero(flags[:, i] & ~raw_flags[:, i])),
                maximum_distance_increase_m=float((samples["signed_distance_m"][:, i] - raw_samples["signed_distance_m"][:, i]).max()),
            ) for i, name in enumerate(body_names)},
        )
        samples.update({f"raw_{key}": value for key, value in raw_samples.items()})
        samples["raw_contact_candidate_flags"] = raw_flags
    label_metadata = dict(method=METHOD, thresholds=asdict(thresholds), threshold_basis=report["threshold_basis"],
                          physics_validated=False, forces_used=False, force_threshold_n=None,
                          self_collision_checked=True,
                          self_collision_max_penetration_m=self_report["maximum_penetration_m"],
                          self_collision_gate_method=report["self_collision_gate_method"],
                          label_target_anchor=report["label_target_anchor"])
    labels = dict(contact_flags=flags, contact_targets=targets, contact_target_flags=target_flags,
                  fps=np.array(fps), body_names=np.array(body_names), source_num_frames=np.array(len(flags)),
                  source_init_phase=np.array(0.0), source_end_phase=np.array(1.0),
                  source_sha256=np.array(report["source_sha256"]), scene_fingerprint=np.array(scene.fingerprint),
                  method=np.array(METHOD), physics_validated=np.array(False), forces_used=np.array(False),
                  self_collision_checked=np.array(True),
                  self_collision_max_penetration_m=np.array(self_report["maximum_penetration_m"]),
                  geometry_metadata_json=np.array(json.dumps(label_metadata, sort_keys=True)),
                  scene_surface_point_w=samples["scene_surface_point_w"].astype(np.float32),
                  signed_distance_m=samples["signed_distance_m"].astype(np.float32))
    validate_contact_labels(labels, len(flags), fps)
    validate_contact_body_order(labels, spec)
    return report, samples, labels
