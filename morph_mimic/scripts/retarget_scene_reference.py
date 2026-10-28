"""Fit a nominal-held G1 reference to raw anchors and its actual collision scene.

    python -m morph_mimic.scripts.retarget_scene_reference \
        --clip climb_03_z_scale_1.0 --start-s 0 --duration-s 10 \
        --output-dir datasets/g1_box_climb_candidate

This is a bounded engineering optimization, not the authors' retargeter.
Every selected frame is retained. Small root corrections, 23 controlled joint
changes, original 29-DOF body anchors, original geometric contact components,
and all robot-scene collision pairs participate in the fit. Held joints stay
nominal. Candidate outputs remain explicitly unvalidated by physical replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from morph_mimic.contacts.reference_geometry import GeometryThresholds, diagnose_reference, sample_collision_geometry
from morph_mimic.contacts.self_collision import effective_robot_geom_pairs, sample_self_contacts
from morph_mimic.scripts.import_omniretarget import canonical_kinematics, convert_clip
from morph_mimic.scripts.label_reference_geometry import _write_new
from morph_mimic.diagnostics.validate_scene_reference import _body_columns, _joint_indices, build_model
from morph_mimic.utils.motion import validate_motion_arrays
from morph_mimic.utils.reference_scene import load_reference_scene
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SceneFit:
    """Finite-difference least squares with exact near-scene geom distances."""

    def __init__(self, model, scene_ids, spec, body_ids):
        import mujoco

        self.mj, self.model, self.spec = mujoco, model, spec
        self.data = mujoco.MjData(model)
        self.scene_ids = np.array(sorted(scene_ids), dtype=int)
        self.body_ids = np.asarray(body_ids)
        self.qids, _ = _joint_indices(model, spec)
        self.controlled = np.array([spec.joint_names.index(n) for n in spec.controlled_joint_names])
        self.held = np.array([spec.joint_names.index(n) for n in spec.held_joint_names])
        self.robot_ids = np.array([g for g in range(model.ngeom) if g not in scene_ids and model.geom_bodyid[g] > 0
                                   and (model.geom_contype[g] or model.geom_conaffinity[g])], dtype=int)
        self.body_geoms = [self.robot_ids[model.geom_bodyid[self.robot_ids] == body] for body in body_ids]
        if not all(len(geoms) for geoms in self.body_geoms):
            raise ValueError("all nine anchor bodies need actual collision geoms")
        self.compatible = np.array([
            [(model.geom_contype[r] & model.geom_conaffinity[s]) or (model.geom_contype[s] & model.geom_conaffinity[r])
             for s in self.scene_ids] for r in self.robot_ids], dtype=bool)
        self.segment = np.empty(6)
        self.anchor_sigma_m = np.array([.025, .025, .04, .04, .015, .015, .008, .008, .02])[:, None]
        self.n = len(self.controlled)
        self.self_pairs = effective_robot_geom_pairs(model, scene_ids)

    def state(self, variables, source_root, source_quat):
        q = np.zeros(self.model.nq)
        q[:3] = source_root + variables[self.n:self.n + 3]
        rotvec = variables[self.n + 3:]
        angle = np.linalg.norm(rotvec)
        delta = np.r_[np.cos(angle / 2), rotvec * (np.sin(angle / 2) / angle if angle > 1e-12 else .5)]
        self.mj.mju_mulQuat(q[3:7], np.asarray(source_quat, dtype=float), delta)
        q[self.qids] = np.asarray(self.spec.default_pose)
        q[self.qids[self.controlled]] = variables[:self.n]
        return q

    def distances(self, raw_contact_scene_ids):
        """AABB pruning only skips separated pairs with zero avoidance cost."""
        model, data = self.model, self.data
        rotation = data.geom_xmat.reshape(-1, 3, 3)
        centers = data.geom_xpos + np.einsum("gij,gj->gi", rotation, model.geom_aabb[:, :3])
        half = np.einsum("gij,gj->gi", np.abs(rotation), model.geom_aabb[:, 3:])
        self.world_geom_centers, self.world_geom_half = centers, half
        gap = np.maximum(np.abs(centers[self.robot_ids, None] - centers[self.scene_ids])
                         - half[self.robot_ids, None] - half[self.scene_ids], 0)
        lower_bound = np.linalg.norm(gap, axis=-1)
        query = (lower_bound <= .02) & self.compatible
        distance = lower_bound.copy()
        distance[~self.compatible] = 5.0
        cache = {}

        def exact(robot, scene):
            key = int(robot), int(scene)
            if key not in cache:
                cache[key] = float(self.mj.mj_geomDistance(model, data, key[0], key[1], 5.0, self.segment))
            return cache[key]

        for row, column in np.argwhere(query):
            distance[row, column] = exact(self.robot_ids[row], self.scene_ids[column])
        closest = distance.min(axis=1)
        contacts = np.zeros(9)
        for column, scene in enumerate(raw_contact_scene_ids):
            if scene >= 0:
                contacts[column] = min(exact(geom, scene) for geom in self.body_geoms[column])
        return closest, contacts

    def self_distances(self):
        """Fixed-dimension self-pair distances with conservative AABB pruning."""
        first, second = self.self_pairs.T
        centers, half = self.world_geom_centers, self.world_geom_half
        gap = np.maximum(np.abs(centers[first] - centers[second]) - half[first] - half[second], 0)
        distance = np.linalg.norm(gap, axis=-1)
        for index in np.flatnonzero(distance <= .02):
            distance[index] = self.mj.mj_geomDistance(self.model, self.data, int(first[index]), int(second[index]), 5, self.segment)
        return distance

    def optimize(self, raw_states, raw_body_positions, raw_contact_scene_ids, fps, max_nfev=30):
        from scipy.optimize import least_squares

        n, spec = self.n, self.spec
        lower = np.r_[np.asarray(spec.pos_limits_lower)[self.controlled], [-.03] * 3, [-.1] * 3]
        upper = np.r_[np.asarray(spec.pos_limits_upper)[self.controlled], [.03] * 3, [.1] * 3]
        all_states, diagnostics = [], []
        correction = np.zeros(n + 6)
        previous = None
        started = time.monotonic()
        for frame, raw in enumerate(raw_states):
            baseline = raw[self.qids[self.controlled]]
            initial = np.r_[baseline, np.zeros(6)] + correction
            lo, hi = lower.copy(), upper.copy()
            if previous is not None:
                max_step = np.asarray(spec.vel_limits)[self.controlled] / fps
                lo[:n] = np.maximum(lo[:n], previous[:n] - max_step)
                hi[:n] = np.minimum(hi[:n], previous[:n] + max_step)
                lo[n:n + 3], hi[n:n + 3] = np.maximum(lo[n:n + 3], previous[n:n + 3] - .008), np.minimum(hi[n:n + 3], previous[n:n + 3] + .008)
                lo[n + 3:], hi[n + 3:] = np.maximum(lo[n + 3:], previous[n + 3:] - .02), np.minimum(hi[n + 3:], previous[n + 3:] + .02)
            initial = np.clip(initial, lo + 1e-10, hi - 1e-10)
            active = raw_contact_scene_ids[frame] >= 0

            def residual(variables):
                self.data.qpos[:] = self.state(variables, raw[:3], raw[3:7])
                self.mj.mj_kinematics(self.model, self.data)
                separation, contacts = self.distances(raw_contact_scene_ids[frame])
                self_separation = self.self_distances()
                prior = np.r_[baseline, np.zeros(6)]
                # Strong obstacle and contact terms dominate small pose errors.
                return np.r_[
                    ((self.data.xpos[self.body_ids] - raw_body_positions[frame]) / self.anchor_sigma_m).ravel(),
                    (variables[:n] - baseline) / .5,
                    (variables[:n] - baseline - correction[:n]) / .2,
                    (variables[n:] - prior[n:]) / np.r_[[.03] * 3, [.1] * 3],
                    (variables[n:] - correction[n:]) / np.r_[[.03] * 3, [.1] * 3],
                    np.minimum(separation + .002, 0) / .002,
                    np.minimum(self_separation + .002, 0) / .002,
                    np.where(active, np.maximum(contacts - .003, 0), 0) / .003,
                ]

            result = least_squares(residual, initial, bounds=(lo, hi), max_nfev=max_nfev,
                                   ftol=2e-4, xtol=2e-5, gtol=1e-3,
                                   x_scale=np.r_[[.3] * n, [.02] * 3, [.1] * 3])
            previous = result.x.copy()
            correction = result.x - np.r_[baseline, np.zeros(6)]
            all_states.append(self.state(result.x, raw[:3], raw[3:7]))
            diagnostics.append(dict(frame=frame, success=bool(result.success), status=int(result.status),
                                    nfev=int(result.nfev), cost=float(result.cost), optimality=float(result.optimality)))
            if frame % 50 == 0 or frame == len(raw_states) - 1:
                print(f"Scene fit {frame + 1}/{len(raw_states)}: cost={result.cost:.3g}, nfev={result.nfev}, elapsed={time.monotonic() - started:.1f}s", flush=True)
        return np.asarray(all_states), diagnostics


def run_fit(dataset_root, clip, output_dir, start_s=0, duration_s=10, max_nfev=30):
    spec = RobotSpec.load("g1")
    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    sources = list(dataset_root.glob(f"extracted/robot-terrain/**/{clip}.npz"))
    if len(sources) != 1:
        raise ValueError("select exactly one downloaded robot-terrain clip")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("retarget output directory must be new or empty")
    fps = 1 / spec.control_dt
    start, length = round(start_s * fps), round(duration_s * fps)
    if start_s < 0 or duration_s <= 0 or not np.isclose(start, start_s * fps) or not np.isclose(length, duration_s * fps):
        raise ValueError("window start and duration must align with the control time grid")
    with tempfile.TemporaryDirectory(prefix="morph_scene_fit_", dir="/tmp") as directory:
        converted, provenance = convert_clip(sources[0], dataset_root, Path(directory))
        scene = load_reference_scene(converted.with_suffix(".scene.json"), motion_path=converted)
        with np.load(converted.parent / "raw_reference" / converted.name, allow_pickle=False) as archive:
            raw = {k: archive[k] for k in archive.files}
        if start + length >= len(raw["joint_pos"]):
            raise ValueError("selected window exceeds available source duration")
        indices = slice(start, start + length + 1)
        names = raw["joint_names"].tolist()
        joint_order = [names.index(name) for name in spec.joint_names]
        model, scene_ids = build_model(PROJECT_ROOT / spec.runtime_assets["mjcf"], scene, spec.physics_dt)
        body_names, body_ids = _body_columns(model, spec)
        qids, _ = _joint_indices(model, spec)
        raw_states = np.zeros((length + 1, model.nq))
        raw_states[:, :3], raw_states[:, 3:7] = raw["root_pos"][indices], raw["root_rot_wxyz"][indices]
        raw_states[:, qids] = raw["joint_pos"][indices][:, joint_order]
        raw_geometry = sample_collision_geometry(model, scene_ids, body_ids, raw_states)
        raw_scene_ids = raw_geometry["closest_scene_geom_id"].copy()
        raw_flags = raw_geometry["signed_distance_m"] <= GeometryThresholds().contact_distance_m
        raw_scene_ids[~raw_flags] = -1
        fitter = SceneFit(model, scene_ids, spec, body_ids)
        fitted, iterations = fitter.optimize(raw_states, raw_geometry["body_origin_w"], raw_scene_ids, fps, max_nfev)
        # Recompute every derivative and body FK after fitting, including endpoints.
        omni = np.c_[fitted[:, 3:7], fitted[:, :3], fitted[:, qids]]
        arrays = canonical_kinematics(model, omni, fps, spec.joint_names)
        raw_arrays = canonical_kinematics(model, np.c_[raw_states[:, 3:7], raw_states[:, :3], raw_states[:, qids]], fps, spec.joint_names)
        validate_motion_arrays(arrays, spec)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{clip}_scene_fit_{start:04}_{start + length + 1:04}"
        output = output_dir / f"{stem}.npz"
        raw_output = output_dir / "raw_reference" / output.name
        raw_output.parent.mkdir()
        optimizer_meta = dict(
            schema=2, method="bounded_scipy_least_squares_scene_self_collision_anchor_fit_v2",
            source_motion=str(sources[0]), source_sha256=_hash(sources[0]), source_revision=provenance["source_revision"],
            source_window_start_s=start / fps, source_window_end_s=(start + length) / fps,
            source_window_start_frame_50hz=start, num_frames=length + 1, fps=fps,
            complete_frames_preserved=True, source_pose_anchor="raw29 reference evaluated on local MJCF",
            root_translation_component_bound_m=.03, root_rotation_vector_component_bound_rad=.1,
            avoidance_target_penetration_m=.002, contact_target_max_gap_m=.003,
            self_collision_target_penetration_m=.002,
            self_collision_optimization_pair_count=len(fitter.self_pairs),
            self_collision_validation_method="independent mj_collision contacts with unchanged effective model filters",
            validation_max_self_penetration_m=.02,
            validation_contact_distance_m=.01, validation_max_penetration_m=.02,
            max_nfev=max_nfev, optimizer_success_frames=sum(x["success"] for x in iterations),
            held_joint_names=spec.held_joint_names, held_joint_target="spec nominal pose",
            physics_validated=False, isaac_runtime_validated=False, labels_written=False,
            full_source_conversion_provenance=provenance,
        )
        arrays["scene_retarget_metadata_json"] = np.array(json.dumps(optimizer_meta, sort_keys=True))
        raw_arrays["scene_retarget_metadata_json"] = arrays["scene_retarget_metadata_json"]
        for path, values in ((output, arrays), (raw_output, raw_arrays)):
            _write_new(path, lambda stream, values=values: np.savez(stream, **values))
        scene_document = json.loads(converted.with_suffix(".scene.json").read_text())
        scene_document.update(motion_file=output.name, motion_sha256=_hash(output),
                              source_first_root_xy=arrays["root_pos"][0, :2].tolist())
        _write_new(output.with_suffix(".scene.json"), lambda stream: stream.write(json.dumps(scene_document, indent=2).encode()))
    report, samples, _ = diagnose_reference(output, output.with_suffix(".scene.json"))
    fitted_geometry = sample_collision_geometry(model, scene_ids, body_ids, fitted)
    fitted_flags = fitted_geometry["signed_distance_m"] <= .01
    recalled = raw_flags & fitted_flags
    per_body_recall = {name: float(recalled[:, i].sum() / raw_flags[:, i].sum()) if raw_flags[:, i].any() else None
                       for i, name in enumerate(body_names)}
    recall = float(recalled.sum() / max(1, raw_flags.sum()))
    root_change = fitted[:, :3] - raw_states[:, :3]
    report.update(
        retarget=optimizer_meta, optimization_frames=iterations,
        raw_all_robot_max_penetration_m=float(raw_geometry["all_robot_penetration_m"].max()),
        raw_all_robot_frames_over_2cm=int((raw_geometry["all_robot_penetration_m"] > .02).sum()),
        root_translation_max_change_m=float(np.linalg.norm(root_change, axis=-1).max()),
        root_translation_component_max_change_m=np.abs(root_change).max(axis=0).tolist(),
        raw_geometric_contact_recall=recall, raw_geometric_contact_recall_per_body=per_body_recall,
        raw_contact_recall_engineering_minimum=.9,
        anchor_position_rms_error_m=float(np.sqrt(np.mean(np.sum((fitted_geometry["body_origin_w"] - raw_geometry["body_origin_w"]) ** 2, axis=-1)))),
    )
    failures = list(report["rejection_reasons"])
    if recall < .9 or any(per_body_recall[n] is not None and per_body_recall[n] < .9 for n in body_names[:2]):
        failures.append("raw geometric contact schedule recall is below the 90% engineering target")
    speeds = np.abs(arrays["joint_vel"])
    report["maximum_joint_speed_rad_s"] = float(speeds.max())
    if np.any(speeds > np.asarray(spec.vel_limits) + 1e-4):
        failures.append("recomputed joint velocities exceed robot spec limits")
    report["retarget_accepted"] = not failures
    report["retarget_rejection_reasons"] = failures
    report["candidate_motion_path"] = str(output)
    report["source_raw_window_path"] = str(raw_output)
    samples_path = output_dir / "diagnostics" / f"{output.stem}.retarget_samples.npz"
    report["diagnostic_samples"] = str(samples_path)
    _write_new(output.with_suffix(".retarget.json"), lambda stream: stream.write(json.dumps(report, indent=2).encode()))
    samples.update(raw_contact_candidate_flags=raw_flags, raw_signed_distance_m=raw_geometry["signed_distance_m"],
                   raw_scene_surface_point_w=raw_geometry["scene_surface_point_w"])
    _write_new(samples_path, lambda stream: np.savez_compressed(stream, **samples))
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets/external/omniretarget")
    parser.add_argument("--clip", default="climb_03_z_scale_1.0")
    parser.add_argument("--start-s", type=float, default=0)
    parser.add_argument("--duration-s", type=float, default=10)
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if Path(args.clip).name != args.clip or args.max_nfev < 1:
            raise ValueError("clip must be a source basename and max-nfev positive")
        output, report = run_fit(args.dataset_root, args.clip, args.output_dir, args.start_s, args.duration_s, args.max_nfev)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(output=str(output), accepted=report["retarget_accepted"],
                         max_penetration_m=report["robot_scene_max_penetration_m"],
                         raw_contact_recall=report["raw_geometric_contact_recall"],
                         rejection_reasons=report["retarget_rejection_reasons"])))
    raise SystemExit(0 if report["retarget_accepted"] else 1)


if __name__ == "__main__":
    main()
