"""Convert one OmniRetarget robot-terrain clip and bind its original scene.

    python -m morph_mimic.scripts.import_omniretarget \
        --clip climb_18_z_scale_1.0 --output-dir datasets/g1_box_climb_imported

The source uses Drake's WXYZ + XYZ free-root layout, followed by the scalar
joints in its G1 URDF declaration order. This importer explicitly names those
columns, resamples at the controller's rate without changing playback speed,
and builds complete local-MJCF body kinematics. A raw_reference/ copy retains
the resampled 29-DOF trajectory for comparison; the main reference holds the
six uncontrolled joints at their controller nominal positions. The resulting
hand displacement is reported, not interpreted as preserved scene contacts.

No contact labels or physical tracking claims are produced by conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from morph_mimic.utils.motion import validate_motion_arrays
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resample_qpos(qpos, source_fps, target_fps):
    """Linear XYZ/joints and shortest-arc quaternion SLERP on a fixed grid."""
    source_duration = (len(qpos) - 1) / source_fps
    times = np.arange(int(np.floor(source_duration * target_fps + 1e-9)) + 1) / target_fps
    if len(times) < 2:
        raise ValueError("source duration is shorter than one controller interval")
    fractional = np.minimum(times * source_fps, len(qpos) - 1)
    low = np.floor(fractional).astype(int)
    high = np.minimum(low + 1, len(qpos) - 1)
    fraction = (fractional - low)[:, None]
    result = qpos[low] * (1 - fraction) + qpos[high] * fraction
    q0, q1 = qpos[low, :4].copy(), qpos[high, :4].copy()
    q0 /= np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 /= np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    denominator = np.sin(angle)
    # The linear limit avoids 0/0 and has the same shortest-arc limit.
    safe = np.where(dot > 1 - 1e-10, 1.0, denominator)
    spherical = np.sin((1 - fraction) * angle) / safe * q0 + np.sin(fraction * angle) / safe * q1
    quat = np.where(dot > 1 - 1e-10, (1 - fraction) * q0 + fraction * q1, spherical)
    result[:, :4] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return result, times


def canonical_kinematics(model, qpos, fps, joint_names):
    """Named FK with body-origin linear and angular velocities in world axes."""
    import mujoco

    data = mujoco.MjData(model)
    qids = np.array([model.joint(name).qposadr[0] for name in joint_names])
    vids = np.array([model.joint(name).dofadr[0] for name in joint_names])
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    q = np.zeros((len(qpos), model.nq), dtype=np.float64)
    q[:, :3], q[:, 3:7], q[:, qids] = qpos[:, 4:7], qpos[:, :4], qpos[:, 7:]
    velocity = np.empty((len(q), model.nv))
    for frame in range(len(q)):
        before, after = max(frame - 1, 0), min(frame + 1, len(q) - 1)
        mujoco.mj_differentiatePos(model, velocity[frame], (after - before) / fps, q[before], q[after])
    arrays = dict(fps=np.array(fps), category=np.array("box_climb"), requires_scene=np.array(True),
                  joint_names=np.array(joint_names), body_names=np.array(body_names),
                  joint_pos=qpos[:, 7:].astype(np.float32), joint_vel=velocity[:, vids].astype(np.float32),
                  root_pos=q[:, :3].astype(np.float32), root_rot_wxyz=q[:, 3:7].astype(np.float32))
    for name, width in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
        arrays[name] = np.empty((len(q), len(body_names), width), dtype=np.float32)
    spatial = np.empty(6)
    for frame in range(len(q)):
        data.qpos[:], data.qvel[:] = q[frame], velocity[frame]
        mujoco.mj_forward(model, data)
        arrays["body_pos_w"][frame], arrays["body_quat_w"][frame] = data.xpos[1:], data.xquat[1:]
        for index, body_id in enumerate(range(1, model.nbody)):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, spatial, 0)
            arrays["body_ang_vel_w"][frame, index] = spatial[:3]
            arrays["body_lin_vel_w"][frame, index] = spatial[3:] + np.cross(
                spatial[:3], data.xpos[body_id] - data.xipos[body_id])
    return arrays


def _publish(path, writer):
    """Publish a complete file without replacing concurrent/existing output."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def convert_clip(source_path, dataset_root, output_dir, spec=None):
    import mujoco

    spec = spec or RobotSpec.load("g1")
    if spec.name != "g1":
        raise ValueError("OmniRetarget's G1 references require separate retargeting for another robot")
    source_path, dataset_root, output_dir = map(lambda p: Path(p).resolve(), (source_path, dataset_root, output_dir))
    match = re.fullmatch(r"(climb_\d{2})(_z_scale_\d+(?:\.\d+)?)", source_path.stem)
    if not match:
        raise ValueError("only named climb_XX_z_scale_N.N robot-terrain clips have a verified scene binding")
    source_urdf = dataset_root / "models/g1/g1_29dof_spherehand.urdf"
    terrain_urdf = dataset_root / "models/terrain" / match[1] / f"multi_boxes{match[2]}.urdf"
    joints = [j for j in ET.parse(source_urdf).getroot().findall("joint") if j.get("type") != "fixed"]
    joint_names = [j.get("name") for j in joints]
    if len(joint_names) != spec.num_joints or set(joint_names) != set(spec.joint_names):
        raise ValueError("source URDF must name exactly the complete G1 joint set")
    if any(j.get("type") not in ("revolute", "continuous", "prismatic") for j in joints):
        raise ValueError("source qpos requires scalar URDF joints")
    collision_assets = {}
    for mesh in ET.parse(terrain_urdf).getroot().findall(".//collision/geometry/mesh"):
        path = (terrain_urdf.parent / mesh.attrib["filename"]).resolve()
        if not path.is_relative_to(dataset_root):
            raise ValueError("terrain mesh escapes the local dataset root")
        collision_assets[str(path)] = sha256(path)
    if not collision_assets:
        raise ValueError("terrain URDF has no collision meshes")
    with np.load(source_path, allow_pickle=False) as archive:
        source_qpos = np.asarray(archive["qpos"], dtype=np.float64)
        rate = np.asarray(archive["fps"]).reshape(-1)
    if source_qpos.ndim != 2 or source_qpos.shape[1] != 7 + spec.num_joints or len(source_qpos) < 2 or not np.isfinite(source_qpos).all():
        raise ValueError("source qpos must be finite [T,36] robot-terrain data; object trajectories are not supported")
    if rate.size != 1 or not np.isfinite(rate[0]) or rate[0] <= 0:
        raise ValueError("source fps must contain one finite positive value")
    norm_error = float(np.abs(np.linalg.norm(source_qpos[:, :4], axis=-1) - 1).max())
    if norm_error > 1e-3:
        raise ValueError("source root must use unit WXYZ quaternions")
    target_fps = 1.0 / spec.control_dt
    qpos, times = resample_qpos(source_qpos, float(rate[0]), target_fps)
    model_path = PROJECT_ROOT / spec.runtime_assets["mjcf"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    model_joints = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0 or model.jnt_dofadr[free[0]] != 0 or model.body(model.jnt_bodyid[free[0]]).name != spec.base_link:
        raise ValueError("reference model requires one leading free root at the specified base link")
    if set(model_joints) != set(joint_names) or len(model_joints) != len(joint_names):
        raise ValueError("reference model's complete joint set differs from the source")
    raw = canonical_kinematics(model, qpos, target_fps, joint_names)
    validate_motion_arrays(raw, spec, source_path, require_nominal_held=False)
    held = [joint_names.index(name) for name in spec.held_joint_names]
    nominal = np.asarray(spec.default_pose)[[spec.joint_names.index(name) for name in spec.held_joint_names]]
    held_max_change = np.max(np.abs(qpos[:, 7 + np.array(held)] - nominal), axis=0)
    projected_qpos = qpos.copy()
    projected_qpos[:, 7 + np.array(held)] = nominal
    projected = canonical_kinematics(model, projected_qpos, target_fps, joint_names)
    validate_motion_arrays(projected, spec, output_dir)
    displacement = np.linalg.norm(projected["body_pos_w"] - raw["body_pos_w"], axis=-1)
    source_duration = (len(source_qpos) - 1) / float(rate[0])
    upstream_report = dataset_root / "download_report.json"
    upstream = json.loads(upstream_report.read_text()) if upstream_report.exists() else {}
    destination = output_dir / source_path.name
    raw_destination = output_dir / "raw_reference" / source_path.name
    conversion_path, scene_path = destination.with_suffix(".conversion.json"), destination.with_suffix(".scene.json")
    for path in (destination, raw_destination, conversion_path, scene_path, output_dir / f"{destination.stem}_contacts.npz"):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite or reuse derived output: {path}")
    metadata = dict(
        schema=1, source="omniretarget", category="box_climb", robot="g1",
        source_motion=str(source_path), source_sha256=sha256(source_path),
        source_repository=upstream.get("repository"), source_revision=upstream.get("revision"),
        source_robot_urdf=str(source_urdf), source_robot_urdf_sha256=sha256(source_urdf),
        source_qpos_layout="WXYZ quaternion, XYZ root position, scalar joints in source URDF declaration order",
        source_joint_names=joint_names, source_num_frames=len(source_qpos), source_fps=float(rate[0]),
        source_duration_s=source_duration, target_fps=target_fps, num_frames=len(times), duration_s=float(times[-1]),
        source_times_s=times.tolist(), discarded_tail_s=max(0.0, source_duration - float(times[-1])),
        interpolation="linear XYZ and joint positions; shortest-arc quaternion SLERP; fixed controller time grid",
        velocity_convention="MuJoCo mj_differentiatePos central differences, one-sided endpoints; body-frame origin linear and angular velocities in world axes",
        root_trajectory_rebased=False, root_trajectory_time_scaled=False, joint_limits_clipped=False,
        source_quaternion_max_norm_error=norm_error,
        reference_model=str(model_path.resolve()), reference_model_sha256=sha256(model_path),
        robot_spec_sha256=sha256(PROJECT_ROOT / "assets/g1/robot_spec.json"),
        model_dependency_sha256={str(p.relative_to(model_path.parent)): sha256(p) for p in sorted(model_path.parent.rglob("*")) if p.is_file()},
        held_joint_nominal_rad=dict(zip(spec.held_joint_names, nominal.tolist())),
        held_joint_max_change_rad=dict(zip(spec.held_joint_names, held_max_change.tolist())),
        held_projection_body_position_max_delta_m=dict(zip(raw["body_names"].tolist(), displacement.max(axis=0).tolist())),
        held_projection_body_position_rms_delta_m=dict(zip(raw["body_names"].tolist(), np.sqrt(np.mean(displacement ** 2, axis=0)).tolist())),
        raw_reference=str(raw_destination), projected_reference=str(destination), scene_sidecar=str(scene_path),
        reference_only=True, contact_labels_generated=False, contact_labels_inherited=False,
        scene_contacts_preserved=False, physical_tracking_validated=False, isaac_runtime_asset_validated=False,
        mujoco_version=mujoco.__version__,
    )
    encoded = np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    raw["conversion_metadata_json"], projected["conversion_metadata_json"] = encoded, encoded
    scene = dict(
        schema=1, source="omniretarget", robot="g1", motion_file=destination.name,
        source_motion=dict(path=str(source_path), sha256=metadata["source_sha256"]),
        scene_urdf=dict(path=str(terrain_urdf), sha256=sha256(terrain_urdf)),
        source_first_root_xy=projected["root_pos"][0, :2].tolist(),
        ground=dict(top_z_m=0.0, size_m=[10.0, 10.0], thickness_m=1.0, center_xy_m=[0.0, 0.0]),
        collision_asset_sha256=collision_assets,
        world_position=[0.0, 0.0, 0.0], world_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
        contact_labels_generated=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_destination.parent.mkdir(exist_ok=True)
    created = []
    try:
        for path, arrays in ((raw_destination, raw), (destination, projected)):
            _publish(path, lambda stream, arrays=arrays: np.savez(stream, **arrays))
            created.append(path)
        metadata.update(raw_reference_sha256=sha256(raw_destination), projected_motion_sha256=sha256(destination))
        scene["motion_sha256"] = metadata["projected_motion_sha256"]
        for path, document in ((conversion_path, metadata), (scene_path, scene)):
            _publish(path, lambda stream, document=document: stream.write((json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()))
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return destination, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets/external/omniretarget")
    parser.add_argument("--clip", default="climb_18_z_scale_1.0", help="robot-terrain source name without extension")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if Path(args.clip).name != args.clip or args.clip.endswith(".npz"):
        parser.error("--clip must be a source name without a directory or .npz extension")
    try:
        paths = list(args.dataset_root.glob(f"extracted/robot-terrain/**/{args.clip}.npz"))
        if len(paths) != 1:
            raise ValueError(f"expected exactly one robot-terrain source for {args.clip}, found {len(paths)}")
        output, metadata = convert_clip(paths[0], args.dataset_root, args.output_dir)
    except (OSError, ValueError, KeyError, ET.ParseError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Converted {output}: {metadata['num_frames']} frames, {metadata['target_fps']:g} Hz, {metadata['duration_s']:g} s")
    for name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
        print(f"Nominal held-joint projection changes {name} by up to {metadata['held_projection_body_position_max_delta_m'][name]:.4f} m")
    print("Scene binding and raw 29-DOF comparison saved. Scene-aware contact labeling and physical tracking remain unvalidated.")


if __name__ == "__main__":
    main()
