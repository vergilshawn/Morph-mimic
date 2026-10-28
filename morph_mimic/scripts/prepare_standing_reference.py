"""Prepare an independent nominal-standing reference from source-model FK.

This auxiliary task tests learned balance with the existing 23-action ACP
controller. Its foot labels mean geometric proximity to a flat z=0 plane;
they are neither measured forces nor a successful physical replay.

    python -m morph_mimic.scripts.prepare_standing_reference \
        --output-dir datasets/g1_standing_reference_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from morph_mimic.contacts.reference_geometry import GeometryThresholds, geometry_hindsight
from morph_mimic.utils.motion import (
    validate_contact_body_order,
    validate_contact_labels,
    validate_motion_arrays,
)
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


METHOD = "engineering_static_reference_geometry_v1"
CLIP_NAME = "standing_nominal_g1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _minimum_geom_z(model, data, geom_id, mujoco):
    """Exact support height for the source robot's collision primitives."""
    center_z = float(data.geom_xpos[geom_id, 2])
    radius = float(model.geom_size[geom_id, 0])
    kind = model.geom_type[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return center_z - radius
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis_z = float(data.geom_xmat[geom_id].reshape(3, 3)[2, 2])
        return center_z - abs(axis_z) * float(model.geom_size[geom_id, 1]) - radius
    raise ValueError(f"unsupported collision primitive for exact plane distance: {model.geom(geom_id).name}")


def _dependencies(source):
    tree = ET.parse(source).getroot()
    if tree.findall("include"):
        raise ValueError("source includes need explicit dependency hashing before this preparation is supported")
    compiler = tree.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    paths = [source]
    for mesh in tree.findall("./asset/mesh"):
        if mesh.get("file"):
            paths.append((source.parent / meshdir / mesh.get("file")).resolve())
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in sorted(set(paths))}


def prepare_standing_reference(output_dir, *, duration_s=10.0, clearance_m=0.002):
    """Create a new, source-bound static FK package without overwriting data."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prepared output directory: {output_dir}")
    spec = RobotSpec.load("g1")
    fps = float(spec.motion_fps)
    intervals_float = duration_s * fps
    if (not np.isfinite(duration_s) or duration_s <= 0
            or not np.isclose(intervals_float, round(intervals_float), rtol=0, atol=1e-6)):
        raise ValueError("duration-s must be finite, positive and aligned to the motion sampling interval")
    if not np.isfinite(clearance_m) or not 0 <= clearance_m <= 0.005:
        raise ValueError("clearance-m must be finite and in [0, 0.005] metres")
    if not np.isclose(fps, 1.0 / spec.control_dt, rtol=0, atol=1e-6):
        raise ValueError("motion fps must match the controller frequency")

    import mujoco

    source = (PROJECT_ROOT / spec.runtime_assets["mjcf"]).resolve()
    source_dependencies = _dependencies(source)
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("static preparation requires exactly one free root")
    root_qadr = int(model.jnt_qposadr[free[0]])
    data.qpos[root_qadr:root_qadr + 7] = [0, 0, 0, 1, 0, 0, 0]
    for name, value in zip(spec.joint_names, spec.default_pose):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"missing single-hinge source joint: {name}")
        data.qpos[model.jnt_qposadr[joint]] = value
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    body_ids = list(range(1, model.nbody))
    body_names = [model.body(i).name for i in body_ids]
    contact_names, _ = spec.resolve_contact_bodies(body_names)
    contact_ids = [model.body(name).id for name in contact_names]
    collisions = [g for g in range(model.ngeom)
                  if model.geom_bodyid[g] > 0 and (model.geom_contype[g] or model.geom_conaffinity[g])]
    foot_ids = set(contact_ids[6:8])
    foot_geoms = [g for g in collisions if int(model.geom_bodyid[g]) in foot_ids]
    if not foot_geoms or {int(model.geom_bodyid[g]) for g in foot_geoms} != foot_ids:
        raise ValueError("both foot bodies must own collision geometry")
    if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE for g in foot_geoms):
        raise ValueError("G1 standing-height preparation expects source foot capsules")
    foot_support = [{"name": model.geom(g).name,
                     "minimum_z_at_root_z_zero_m": _minimum_geom_z(model, data, g, mujoco)}
                    for g in foot_geoms]
    minimum = min(item["minimum_z_at_root_z_zero_m"] for item in foot_support)
    root_height = -minimum + clearance_m
    data.qpos[root_qadr + 2] = root_height
    mujoco.mj_forward(model, data)

    # This is a pose reference, so all derivatives are exactly zero; no
    # dynamics step, contact force estimate or balance assistance is used.
    frames = round(intervals_float) + 1
    repeat = lambda values: np.broadcast_to(np.asarray(values), (frames,) + np.asarray(values).shape).copy().astype(np.float32)
    arrays = dict(
        fps=np.array(fps), joint_names=np.array(spec.joint_names), body_names=np.array(body_names),
        joint_pos=repeat(spec.default_pose), joint_vel=np.zeros((frames, spec.num_joints), np.float32),
        body_pos_w=repeat(data.xpos[body_ids]), body_quat_w=repeat(data.xquat[body_ids]),
        body_lin_vel_w=np.zeros((frames, len(body_ids), 3), np.float32),
        body_ang_vel_w=np.zeros((frames, len(body_ids), 3), np.float32),
        requires_scene=np.array(False),
        # The current loader has no static category. Reuse its flat locomotion
        # routing without claiming that this constant pose contains walking.
        category=np.array("walk"), motion_type=np.array("static_nominal_standing_auxiliary"),
    )
    validate_motion_arrays(arrays, spec, "generated static standing reference")
    signed_distance = []
    for body in contact_ids:
        owned = [g for g in collisions if model.geom_bodyid[g] == body]
        if not owned:
            raise ValueError(f"contact body owns no collision geometry: {model.body(body).name}")
        signed_distance.append(min(_minimum_geom_z(model, data, g, mujoco) for g in owned))
    threshold = GeometryThresholds(contact_distance_m=0.01)
    flags = repeat(np.array(signed_distance) <= threshold.contact_distance_m).astype(bool)
    if not np.all(flags[:, 6:8]) or flags[:, [0, 1, 2, 3, 4, 5, 8]].any():
        raise ValueError("nominal static pose must produce exactly the two foot proximity labels")
    contact_columns = [body_names.index(name) for name in contact_names]
    root_column = body_names.index(spec.base_link)
    targets, target_flags = geometry_hindsight(
        flags, arrays["body_pos_w"][:, contact_columns], arrays["body_pos_w"][:, root_column],
        arrays["body_quat_w"][:, root_column], fps, threshold,
    )
    labels = dict(
        contact_flags=flags, contact_targets=targets, contact_target_flags=target_flags,
        fps=np.array(fps), body_names=np.array(contact_names), source_num_frames=np.array(frames),
        source_init_phase=np.array(0.0), source_end_phase=np.array(1.0), method=np.array(METHOD),
        contact_target_frame=np.array("reference_query_body_yaw_absolute_z"),
        physics_validated=np.array(False), forces_used=np.array(False),
        signed_distance_m=repeat(signed_distance),
    )
    validate_contact_labels(labels, frames, fps, "static geometry labels")
    validate_contact_body_order(labels, spec, "static geometry labels")
    metadata = dict(
        schema=1, robot=spec.name, clip_name=CLIP_NAME, method=METHOD, duration_s=float(duration_s),
        num_frames=frames, fps=fps, source_mjcf=str(source.relative_to(PROJECT_ROOT)),
        source_mjcf_sha256=_sha256(source), source_dependencies=source_dependencies,
        robot_spec_sha256=_sha256(PROJECT_ROOT / "assets/g1/robot_spec.json"),
        preparation_script_sha256=_sha256(Path(__file__)), mujoco_version=mujoco.__version__,
        root_height_above_ground_m=root_height, minimum_foot_z_at_root_z_zero_m=minimum,
        initial_clearance_m=float(clearance_m), foot_collision_geoms=foot_support,
        contact_body_names=contact_names, contact_signed_plane_distance_m=signed_distance,
        geometric_contact_threshold_m=threshold.contact_distance_m,
        contact_target_frame="reference_query_body_yaw_absolute_z",
        target_semantics="strictly future contact onsets within 1.5 s; initial continuous foot support has no future onset",
        loader_category="walk", loader_category_reason="existing flat-locomotion routing; actual motion is static nominal standing",
        root_position=[0, 0, root_height], root_quaternion_wxyz=[1, 0, 0, 0],
        all_velocities_zero=True, model_body_count=len(body_names), foot_collision_count=len(foot_geoms),
        physics_validated=False, forces_used=False, self_collision_checked=False,
        learned_standing_validated=False, physical_box_climb_validated=False,
        scope="Independent static nominal-pose FK auxiliary task on flat z=0 terrain; geometry labels do not establish dynamic support",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()  # Exclusive publication; a concurrent creator is refused.
    try:
        motion_path = output_dir / f"{CLIP_NAME}.npz"
        with motion_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        labels["source_sha256"] = np.array(_sha256(motion_path))
        label_path = output_dir / f"{CLIP_NAME}_contacts.npz"
        with label_path.open("xb") as stream:
            np.savez_compressed(stream, **labels)
        metadata.update(source_sha256=_sha256(motion_path), contact_labels_sha256=_sha256(label_path))
        with (output_dir / f"{CLIP_NAME}.preparation.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return motion_path, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--clearance-m", type=float, default=0.002)
    args = parser.parse_args()
    try:
        path, metadata = prepare_standing_reference(args.output_dir, duration_s=args.duration_s, clearance_m=args.clearance_m)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(motion=str(path), num_frames=metadata["num_frames"],
                          root_height_above_ground_m=metadata["root_height_above_ground_m"],
                          physics_validated=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
