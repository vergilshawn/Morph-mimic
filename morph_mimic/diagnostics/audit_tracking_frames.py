"""CPU-only, read-only frame diagnostics for saved physical tracking endpoints.

Reconstructs actual link origins/orientations from named 29-axis states using
the source G1 MJCF. This is forward kinematics, not physical replay, assistance,
contact-point reconstruction, or a new skill-acceptance criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def named_columns(names, requested):
    """Exact names only; reject duplicate, absent, or ambiguous columns."""
    names, requested = list(map(str, names)), list(map(str, requested))
    if len(set(names)) != len(names) or len(set(requested)) != len(requested):
        raise ValueError("Joint/body names must be unique")
    missing = set(requested)-set(names)
    if missing:
        raise ValueError(f"Missing named columns: {sorted(missing)}")
    return np.asarray([names.index(name) for name in requested], dtype=int)


def unit_quaternion(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if q.shape[-1] != 4 or not np.isfinite(q).all() or np.any(np.abs(norm-1) > 1e-3):
        raise ValueError("Expected finite unit WXYZ quaternions")
    return q/norm


def conjugate(q):
    out = np.asarray(q).copy()
    out[..., 1:] *= -1
    return out


def multiply(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate((aw*bw-np.sum(av*bv, axis=-1, keepdims=True),
        aw*bv+bw*av+np.cross(av, bv)), axis=-1)


def rotate(q, vectors):
    twice = 2*np.cross(q[..., 1:], vectors)
    return vectors+q[..., :1]*twice+np.cross(q[..., 1:], twice)


def yaw(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def yaw_quaternion(angle):
    return np.stack((np.cos(angle/2), np.zeros_like(angle),
        np.zeros_like(angle), np.sin(angle/2)), axis=-1)


def orientation_error(actual, reference):
    relative = multiply(conjugate(unit_quaternion(reference)), unit_quaternion(actual))
    relative = np.where(relative[..., :1] < 0, -relative, relative)
    angle = 2*np.arctan2(np.linalg.norm(relative[..., 1:], axis=-1), relative[..., 0])
    return relative, angle


def frame_errors(actual, reference, actual_root, reference_root, actual_quat, reference_quat):
    """Actual-minus-reference; shape retains absolute world Z as core ACP does."""
    actual_rotation = conjugate(yaw_quaternion(yaw(unit_quaternion(actual_quat))))
    reference_rotation = conjugate(yaw_quaternion(yaw(unit_quaternion(reference_quat))))
    world = actual-reference
    actual_shape, reference_shape = actual-actual_root[:, None], reference-reference_root[:, None]
    actual_shape[..., 2], reference_shape[..., 2] = actual[..., 2], reference[..., 2]
    shape = rotate(actual_rotation[:, None], actual_shape)-rotate(reference_rotation[:, None], reference_shape)
    return dict(world_error_m=world,
        common_reference_yaw_error_m=rotate(reference_rotation[:, None], world),
        common_reference_full_root_error_m=rotate(conjugate(reference_quat)[:, None], world),
        independent_root_yaw_shape_error_m=shape)


def interpolate(values, phases, *, quaternion=False):
    """Linear positions / antipodal-safe NLERP orientations at recorded phases."""
    position = np.clip(phases, 0., 1.)*(len(values)-1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower+1, len(values)-1)
    fraction = (position-lower).reshape((-1,)+(1,)*(values.ndim-1))
    a, b = values[lower], values[upper]
    if quaternion:
        b = np.where(np.sum(a*b, axis=-1, keepdims=True) < 0, -b, b)
    out = a+(b-a)*fraction
    return unit_quaternion(out) if quaternion else out


def finite_difference(values, time, dt):
    """Backward endpoint derivative; never synthesize an initial reset sample."""
    velocity, valid = np.zeros_like(values), np.zeros(len(values), dtype=bool)
    intervals = np.diff(time)
    valid[1:] = np.isclose(intervals, dt, rtol=0, atol=max(1e-7, dt*1e-4))
    safe = np.where(valid[1:], intervals, 1.)
    velocity[1:] = np.diff(values, axis=0)/safe.reshape((-1,)+(1,)*(values.ndim-1))
    velocity[~valid] = 0.
    return velocity, valid


class SourceFK:
    """Name-mapped source-model FK, without any dynamics step or force inference."""

    def __init__(self, xml_path, joint_names):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        hinges = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        if len(free) != 1 or len(hinges) != 29 or self.model.njnt != 30:
            raise ValueError("This diagnostic requires one free root and exactly 29 named hinge joints")
        self.joint_names = [self.model.joint(int(i)).name for i in hinges]
        if len(joint_names) != 29 or set(map(str, joint_names)) != set(self.joint_names):
            raise ValueError("Saved state must contain every source-model joint exactly once")
        self.joint_columns = named_columns(joint_names, self.joint_names)
        self.joint_ids = hinges
        self.joint_qadr = self.model.jnt_qposadr[hinges]
        self.root_qadr = int(self.model.jnt_qposadr[free[0]])
        self.root_body = int(self.model.jnt_bodyid[free[0]])
        self.body_ids = np.arange(1, self.model.nbody)
        self.body_names = [self.model.body(int(i)).name for i in self.body_ids]
        if len(self.body_names) != 30:
            raise ValueError("Expected the 30-link source G1 model")

    def poses(self, root_pos, root_quat, joints, *, include_joint_anchors=False):
        root_quat = unit_quaternion(root_quat)
        if (root_pos.shape != (len(joints), 3) or root_quat.shape != (len(joints), 4)
                or joints.shape != (len(joints), 29)
                or not np.isfinite(joints).all() or not np.isfinite(root_pos).all()):
            raise ValueError("Malformed complete source FK state")
        positions = np.empty((len(joints), len(self.body_ids), 3))
        orientations = np.empty((len(joints), len(self.body_ids), 4))
        anchors = np.empty((len(joints), len(self.joint_ids), 3)) if include_joint_anchors else None
        for row in range(len(joints)):
            self.data.qpos[self.root_qadr:self.root_qadr+7] = np.r_[root_pos[row], root_quat[row]]
            self.data.qpos[self.joint_qadr] = joints[row, self.joint_columns]
            self.mujoco.mj_forward(self.model, self.data)
            if not np.allclose(self.data.xpos[self.root_body], root_pos[row], atol=1e-9, rtol=0):
                raise ValueError("Free-root pose does not represent the saved root link origin")
            positions[row] = self.data.xpos[self.body_ids]
            orientations[row] = self.data.xquat[self.body_ids]
            if include_joint_anchors:
                anchors[row] = self.data.xanchor[self.joint_ids]
        return (positions, orientations, anchors) if include_joint_anchors else (positions, orientations)


def model_dependencies(xml_path):
    tree = ET.parse(xml_path).getroot()
    if tree.findall("include"):
        raise ValueError("MJCF includes require explicit dependency handling")
    compiler = tree.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    paths = [xml_path]
    for mesh in tree.findall("./asset/mesh"):
        if mesh.get("file"):
            paths.append((xml_path.parent/meshdir/mesh.get("file")).resolve())
    return {str(path): sha256(path) for path in sorted(set(paths))}


def error_summary(error):
    return dict(mean_signed_xyz_m=error.mean(axis=0).tolist(),
        rmse_xyz_m=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        rmse_norm_m=float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))),
        maximum_norm_m=float(np.linalg.norm(error, axis=-1).max()),
        endpoint_signed_xyz_m=error[-1].tolist())


def phase_coverage(metadata, phases):
    """Keep the recorded initialization distinct from the first physical endpoint."""
    initial = metadata.get("result", {}).get("start_phase")
    if initial is not None:
        initial = float(initial)
        if not np.isfinite(initial) or not 0 <= initial <= 1 or initial > float(phases[0])+1e-6:
            raise ValueError("Recorded initialization phase is invalid or after the first physical endpoint")
    return dict(start_phase=initial, initialization_phase=initial,
        first_recorded_phase=float(phases[0]), end_phase=float(phases[-1]))


def audit(trajectory_path, motion_path, *, robot_xml=None):
    trajectory_path, motion_path = Path(trajectory_path).resolve(), Path(motion_path).resolve()
    robot_xml = Path(robot_xml or PROJECT_ROOT/"assets/g1/robot/g1.xml").resolve()
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(trace.pop("metadata_json").item()))
    if metadata.get("robot") != "g1" or metadata.get("positions_frame") != "world" or metadata.get("quaternion_convention") != "world_from_link_wxyz":
        raise ValueError("Only explicitly named source G1 world/WXYZ endpoint trajectories are supported")
    if metadata.get("source_sha256") != sha256(motion_path):
        raise ValueError("Trajectory motion hash does not match the supplied source data")
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    phase, time = np.asarray(trace["phase"], float), np.asarray(trace["elapsed_control_time_s"], float)
    dt = float(metadata["control_dt_s"])
    if (len(phase) == 0 or phase.ndim != 1 or not np.isfinite(phase).all()
            or np.any(phase < 0) or np.any(phase > 1) or np.any(np.diff(phase) < -1e-7)
            or time.shape != phase.shape or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)
            or not np.isfinite(dt) or dt <= 0 or np.unique(trace["clip_id"]).size != 1):
        raise ValueError("Expected one finite, monotonic, uninterrupted physical clip")
    source_root = interpolate(motion["root_pos"], phase)
    offsets = trace["ref_root_pos_w"]-source_root
    origin = np.median(offsets, axis=0)
    origin_deviation = float(np.abs(offsets-origin).max())
    # The runtime's legacy phase-one sample is clamped to 1-1e-6; v2's
    # final root is exact. Both are below this explicit interpolation tolerance.
    if origin_deviation > 2e-5:
        raise ValueError(f"Reference-to-world translation changes within the trajectory ({origin_deviation:g} m)")
    reference_quat = unit_quaternion(trace["ref_root_quat_w"])
    source_quat = interpolate(motion["root_rot_wxyz"], phase, quaternion=True)
    if orientation_error(reference_quat, source_quat)[1].max() > 2e-4:
        raise ValueError("Saved reference root orientations differ from the source clip")
    fk = SourceFK(robot_xml, metadata["all_joint_names"])
    actual_pos, actual_quat, actual_anchors = fk.poses(trace["root_pos_w"], trace["root_quat_w"],
        trace["all_joint_pos"], include_joint_anchors=True)
    body_columns = named_columns(motion["body_names"], fk.body_names)
    reference_pos = interpolate(motion["body_pos_w"][:, body_columns], phase)+origin
    reference_body_quat = interpolate(motion["body_quat_w"][:, body_columns], phase, quaternion=True)
    # Validate that the supplied source-body FK is tied to this exact model.
    source_joint_columns = named_columns(motion["joint_names"], metadata["all_joint_names"])
    reference_joint_q = interpolate(motion["joint_pos"][:, source_joint_columns], phase)
    _, _, reference_anchors = fk.poses(trace["ref_root_pos_w"], reference_quat,
        reference_joint_q, include_joint_anchors=True)
    anchor_errors = frame_errors(actual_anchors, reference_anchors, trace["root_pos_w"],
        trace["ref_root_pos_w"], unit_quaternion(trace["root_quat_w"]), reference_quat)
    source_indices = np.unique(np.linspace(0, len(motion["root_pos"])-1, 5).astype(int))
    check_pos, check_quat = fk.poses(motion["root_pos"][source_indices],
        motion["root_rot_wxyz"][source_indices], motion["joint_pos"][source_indices][:, source_joint_columns])
    fk_error = float(np.linalg.norm(check_pos-motion["body_pos_w"][source_indices][:, body_columns], axis=-1).max())
    fk_orientation_error = float(orientation_error(check_quat, motion["body_quat_w"][source_indices][:, body_columns])[1].max())
    if fk_error > 2e-5 or fk_orientation_error > 2e-4:
        raise ValueError("Source motion FK does not match the supplied source MJCF")
    # Where recorded foot poses exist, independently verify actual-state FK.
    actual_fk_checks = {}
    if "foot_pos_w" in trace:
        feet = named_columns(fk.body_names, ["left_ankle_roll_link", "right_ankle_roll_link"])
        position_check = float(np.linalg.norm(actual_pos[:, feet]-trace["foot_pos_w"], axis=-1).max())
        orientation_check = float(orientation_error(actual_quat[:, feet], trace["foot_quat_w"])[1].max())
        actual_fk_checks = dict(max_recorded_foot_position_disagreement_m=position_check,
            max_recorded_foot_orientation_disagreement_rad=orientation_check)
        if position_check > 5e-4 or orientation_check > 5e-4:
            raise ValueError(f"Source-model actual FK disagrees with recorded foot poses: {actual_fk_checks}")
    errors = frame_errors(actual_pos, reference_pos, trace["root_pos_w"], trace["ref_root_pos_w"],
        unit_quaternion(trace["root_quat_w"]), reference_quat)
    body_rotation_error, body_angle = orientation_error(actual_quat, reference_body_quat)
    root_rotation_error, root_angle = orientation_error(trace["root_quat_w"], reference_quat)
    velocity, derivative_valid = finite_difference(actual_pos, time, dt)
    controlled_names = metadata["joint_names"]
    controlled_columns = named_columns(metadata["all_joint_names"], controlled_names)
    if len(controlled_names) != 23 or not np.allclose(trace["all_joint_pos"][:, controlled_columns], trace["joint_pos"], atol=1e-6, rtol=0):
        raise ValueError("Controlled joint state disagrees with full named state")
    qerror = trace["joint_pos"]-trace["joint_ref_pos"]
    wrapped_qerror = np.arctan2(np.sin(qerror), np.cos(qerror))
    qderror = trace["joint_vel"]-trace["joint_ref_vel"]
    root_world = trace["root_pos_w"]-trace["ref_root_pos_w"]
    root_common = rotate(conjugate(yaw_quaternion(yaw(reference_quat))), root_world)
    root_up = rotate(unit_quaternion(trace["root_quat_w"]), np.broadcast_to([0., 0., 1.], (len(phase), 3)))
    reference_up = rotate(reference_quat, np.broadcast_to([0., 0., 1.], (len(phase), 3)))
    body_report = {}
    for column, name in enumerate(fk.body_names):
        body_report[name] = {key.removesuffix("_m"): error_summary(value[:, column]) for key, value in errors.items()}
        body_report[name]["orientation"] = dict(rmse_rad=float(np.sqrt(np.mean(body_angle[:, column]**2))),
            max_rad=float(body_angle[:, column].max()), endpoint_rad=float(body_angle[-1, column]))
    contact_report = {}
    for column, name in enumerate(metadata["contact_body_names"]):
        body = int(named_columns(fk.body_names, [name])[0])
        current = np.asarray(trace["contact_force_w"][:, column], float)
        history_flags = trace["measured_contact_flags"][:, column] > .5
        valid_current = derivative_valid & (np.abs(current[:, 2]) > 1.)
        speed = np.linalg.norm(velocity[:, body], axis=-1)
        contact_report[name] = dict(endpoint_current_recorded_force_w_n=current[-1].tolist(),
            endpoint_positive_fz_n=float(max(current[-1, 2], 0.)),
            reference_geometry_flag_fraction=float((trace["ref_contact_flags"][:, column] > .5).mean()),
            actual_force_history_flag_fraction=float(history_flags.mean()),
            current_abs_fz_above_1n_fraction=float((np.abs(current[:, 2]) > 1.).mean()),
            moving_link_origin_under_current_fz_samples=int(valid_current.sum()),
            moving_link_origin_under_current_fz_speed_rms_mps=(float(np.sqrt(np.mean(speed[valid_current]**2))) if valid_current.any() else None),
            moving_link_origin_under_current_fz_speed_max_mps=(float(speed[valid_current].max()) if valid_current.any() else None))
    summary = dict(schema=1, scope="Read-only source-FK coordinate diagnostic; not skill acceptance or physical replay",
        trajectory=str(trajectory_path), trajectory_sha256=sha256(trajectory_path),
        motion=str(motion_path), motion_sha256=sha256(motion_path),
        checkpoint=metadata.get("checkpoint"), checkpoint_sha256=metadata.get("checkpoint_sha256"),
        source_sha256={str(Path(__file__).resolve()): sha256(__file__)} | model_dependencies(robot_xml),
        samples=len(phase), **phase_coverage(metadata, phase),
        observed_physical_endpoint_span_s=float(time[-1]-time[0]),
        elapsed_control_time_first_s=float(time[0]), elapsed_control_time_last_s=float(time[-1]), control_dt_s=dt,
        fixed_reference_to_world_translation_m=origin.tolist(), maximum_translation_inconsistency_m=origin_deviation,
        source_fk_validation=dict(checked_source_frames=source_indices.tolist(), max_link_position_error_m=fk_error,
            max_link_orientation_error_rad=fk_orientation_error, **actual_fk_checks),
        definitions=dict(signed_error="actual minus reference", world="Stored simulation world, including fixed tile origin",
            phase_coverage="start_phase/initialization_phase come only from trajectory metadata.result.start_phase, or null when unknown; first_recorded_phase/end_phase are post-physics endpoints, not reset-state samples",
            common_reference_yaw="Rotate world actual-minus-reference by inverse reference yaw; signed +X forward, +Y left, +Z up",
            common_reference_full_root="Rotate world actual-minus-reference by inverse full reference root orientation; origin shared",
            independent_root_yaw_shape="Each pose centers its own root XY and rotates by its own yaw, retains absolute world Z; horizontal root translation/yaw can cancel",
            reference_interpolation="Source body positions linear; body orientations antipodal-safe normalized linear interpolation at recorded phase; legacy endpoint clamp differs by at most reported tolerance",
            body_positions="MJCF link-frame origins, not every joint center or contact surface point",
            joint_pivots="All 29 source MJCF hinge xanchor positions; reference obtained by FK from interpolated 29 named source q and saved reference root, unlike linearly interpolated source body-position targets",
            joint_angles="Primary error is raw actual-minus-reference for limited hinge q; separately named wrapped error reproduces the periodic core reward convention, without treating finite hinges as continuous joints",
            quaternion_error="inverse(reference_world_from_link) * actual_world_from_link; WXYZ, nonnegative W",
            velocity="Backward difference of source-FK actual link origins between adjacent uninterrupted control endpoints; initial derivative invalid",
            force="Recorded current endpoint terrain-filtered normal force vectors; history flags are separate; no friction force reconstructed",
            geometric_reference_contacts="Recorded reference geometry/proximity flags, not measured successful source forces"),
        root=dict(world_error=error_summary(root_world), common_reference_yaw_error=error_summary(root_common),
            orientation_rmse_rad=float(np.sqrt(np.mean(root_angle**2))), orientation_endpoint_rad=float(root_angle[-1]),
            actual_tilt_endpoint_rad=float(np.arccos(np.clip(root_up[-1, 2], -1, 1))),
            reference_tilt_endpoint_rad=float(np.arccos(np.clip(reference_up[-1, 2], -1, 1)))),
        bodies=body_report,
        joint_pivots={name: {key.removesuffix("_m"): error_summary(value[:, i])
            for key, value in anchor_errors.items()} for i, name in enumerate(fk.joint_names)},
        joints={name: dict(position_rmse_rad=float(np.sqrt(np.mean(qerror[:, i]**2))),
            position_endpoint_signed_rad=float(qerror[-1, i]), velocity_rmse_rad_s=float(np.sqrt(np.mean(qderror[:, i]**2))),
            velocity_endpoint_signed_rad_s=float(qderror[-1, i]),
            wrapped_reward_position_rmse_rad=float(np.sqrt(np.mean(wrapped_qerror[:, i]**2)))) for i, name in enumerate(controlled_names)},
        contacts=contact_report, derivative_valid_samples=int(derivative_valid.sum()),
        limitations=["FK reconstructed poses are derived from saved states; this audit performs no dynamics rollout.",
            "Body origins and their finite-difference motion are not contact points or contact-point slip.",
            "Surface normals, tangent friction forces, contact moments, COP and full contact wrench are unavailable.",
            "No measured CoM dynamics, momentum balance, support equilibrium or dynamic stability is established.",
            "Only nine named links have recorded forces; uninstrumented contact cannot be excluded.",
            "Reference geometry flags and force-history flags have different physical meanings.",
            "Source body orientations are interpolated; orientation RMSE is an additional engineering diagnostic."])
    arrays = dict(phase=phase, elapsed_control_time_s=time, body_names=np.asarray(fk.body_names),
        actual_body_pos_w=actual_pos, reference_body_pos_w=reference_pos,
        actual_body_quat_w=actual_quat, reference_body_quat_w=reference_body_quat,
        body_rotation_error_wxyz=body_rotation_error, body_orientation_error_rad=body_angle,
        root_rotation_error_wxyz=root_rotation_error, root_orientation_error_rad=root_angle,
        root_world_error_m=root_world, root_common_reference_yaw_error_m=root_common,
        body_origin_velocity_w_mps=velocity, velocity_derivative_valid=derivative_valid,
        controlled_joint_names=np.asarray(controlled_names), joint_position_error_rad=qerror,
        wrapped_reward_joint_position_error_rad=wrapped_qerror, joint_velocity_error_rad_s=qderror, **errors)
    arrays.update(all_joint_pivot_names=np.asarray(fk.joint_names), actual_joint_pivot_pos_w=actual_anchors,
        reference_joint_pivot_pos_w=reference_anchors,
        **{f"joint_pivot_{key}": value for key, value in anchor_errors.items()})
    # Prove JSON serialization is finite before publishing either artifact.
    json.dumps(summary, ensure_ascii=False, allow_nan=False)
    return summary, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, default=PROJECT_ROOT/"assets/g1/robot/g1.xml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-npz", type=Path)
    args = parser.parse_args(argv)
    for path in (args.output, args.metrics_npz):
        if path is not None and path.exists():
            raise FileExistsError(f"Refusing to overwrite diagnostic output: {path}")
    report, arrays = audit(args.trajectory, args.motion, robot_xml=args.robot_xml)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.metrics_npz is not None:
        args.metrics_npz.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_npz.open("xb") as stream:
            np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(report, allow_nan=False)))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), samples=report["samples"],
        root=report["root"], source_fk_validation=report["source_fk_validation"]), allow_nan=False))


if __name__ == "__main__":
    main()
