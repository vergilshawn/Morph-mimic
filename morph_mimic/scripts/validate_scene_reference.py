"""Diagnose a scene-bound reference with free-root MuJoCo joint PD.

This records a physical baseline, not training contact labels. The root is
initialized once and is never assisted or reset while replaying the clip.
MuJoCo results do not establish Isaac Lab dynamics equivalence.

    python -m morph_mimic.scripts.validate_scene_reference --robot g1 \
        --motion datasets/g1_box_climb_smoke/climb_18_z_scale_1.0.npz \
        --scene datasets/g1_box_climb_smoke/climb_18_z_scale_1.0.scene.json \
        --output datasets/g1_box_climb_smoke/diagnostics/climb_18_pd.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _body_columns(model, spec):
    """Resolve the same nine anatomical columns as the tracker."""
    names = {model.body(i).name for i in range(1, model.nbody)}
    aliases = (getattr(spec, "asset_aliases", None) or {}).get("bodies", {})
    chosen = []
    for group in spec.nine_body_groups():
        candidates = spec.contact_bodies[group]
        sides = [candidates] if len(candidates) == 1 else [
            [name for name in candidates if side in name] for side in ("left", "right")
        ]
        for side in sides:
            body = next((candidate for name in side
                         for candidate in (name, aliases.get(name)) if candidate in names), None)
            if body is None:
                raise ValueError(f"reference MJCF has none of the contact bodies {side}")
            chosen.append(body)
    if len(chosen) != 9 or len(set(chosen)) != 9:
        raise ValueError("robot must resolve to nine distinct contact bodies")
    return chosen, np.array([model.body(name).id for name in chosen], dtype=int)


def _joint_indices(model, spec):
    import mujoco

    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if (len(free) != 1 or model.jnt_qposadr[free[0]] != 0
            or model.jnt_dofadr[free[0]] != 0
            or model.body(model.jnt_bodyid[free[0]]).name != spec.base_link):
        raise ValueError("reference model needs one leading free joint on the spec base link")
    scalar_names = [model.joint(i).name for i in range(model.njnt) if i != free[0]]
    if set(scalar_names) != set(spec.joint_names) or len(scalar_names) != len(spec.joint_names):
        raise ValueError("reference MJCF must contain exactly the full spec joint set")
    for name in spec.joint_names:
        if model.joint(name).type[0] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"only scalar robot joints are supported: {name}")
    return (np.array([model.joint(name).qposadr[0] for name in spec.joint_names]),
            np.array([model.joint(name).dofadr[0] for name in spec.joint_names]))


def build_model(model_path, scene, physics_dt, friction=1.0):
    """Add individually transformed scene collisions, preserving scene gaps.

    MuJoCo uses convex collision hulls for mesh geoms. Concave disconnected
    meshes are split; a connected concave mesh is rejected rather than silently
    replacing the source surface with a hull.
    """
    import mujoco

    source = mujoco.MjSpec.from_file(str(Path(model_path).resolve()))
    source.option.timestep = float(physics_dt)
    # A source model with actuators could otherwise apply additional force.
    if list(source.actuators):
        raise ValueError("PD diagnostic requires an unactuated reference MJCF")
    scene_names = []
    for index, component in enumerate(scene.components):
        prefix = f"reference_scene_{index}"
        common = dict(contype=1, conaffinity=1, condim=3, friction=[friction, 0.005, 0.0001])
        if component.kind == "box":
            transform = np.asarray(component.transform, dtype=float)
            rotation = np.ascontiguousarray(transform[:3, :3]).ravel()
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rotation)
            source.worldbody.add_geom(
                name=prefix, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(np.asarray(component.size_m) / 2).tolist(),
                pos=transform[:3, 3].tolist(), quat=quat.tolist(), **common,
            )
            scene_names.append(prefix)
        elif component.kind == "mesh":
            mesh = component.mesh
            pieces = list(mesh.split(only_watertight=False))
            if not pieces:
                raise ValueError(f"empty scene collision mesh: {component.name}")
            for part, piece in enumerate(pieces):
                if not piece.is_convex:
                    raise ValueError(
                        f"connected concave scene mesh {component.name}: supply validated convex "
                        "collision parts; a convex hull would alter the terrain"
                    )
                name = f"{prefix}_{part}"
                source.add_mesh(name=name, uservert=piece.vertices.ravel().tolist(),
                                userface=piece.faces.ravel().tolist())
                source.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                                          meshname=name, **common)
                scene_names.append(name)
        else:
            raise ValueError(f"unsupported scene collision kind: {component.kind}")
    ground = scene.ground
    name = "reference_scene_ground"
    source.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[ground.size_m[0] / 2, ground.size_m[1] / 2, ground.thickness_m / 2],
        pos=[*ground.center_xy_m, ground.top_z_m - ground.thickness_m / 2],
        contype=1, conaffinity=1, condim=3, friction=[friction, 0.005, 0.0001],
    )
    scene_names.append(name)
    model = source.compile()
    scene_geom_ids = {model.geom(name).id for name in scene_names}
    return model, scene_geom_ids


def scene_contacts(model, data, scene_geom_ids, contact_body_ids):
    """World forces per anatomical link and robot-scene penetration only."""
    import mujoco

    column = {int(body): i for i, body in enumerate(contact_body_ids)}
    world_forces = np.zeros((len(contact_body_ids), 3))
    max_penetration = 0.0
    contact_count = 0
    wrench = np.empty(6)
    for index, contact in enumerate(data.contact):
        geom0, geom1 = map(int, contact.geom)
        scene0, scene1 = geom0 in scene_geom_ids, geom1 in scene_geom_ids
        if scene0 == scene1:  # excludes self collisions and scene-scene pairs
            continue
        robot_geom = geom1 if scene0 else geom0
        robot_body = int(model.geom_bodyid[robot_geom])
        if robot_body == 0:  # another static/world geom is not the robot
            continue
        contact_count += 1
        max_penetration = max(max_penetration, -float(contact.dist))
        if robot_body in column:
            mujoco.mj_contactForce(model, data, index, wrench)
            # MuJoCo reports force exerted on geom1, in the contact frame.
            force_w = contact.frame.reshape(3, 3).T @ wrench[:3]
            world_forces[column[robot_body]] += force_w if scene0 else -force_w
    return world_forces, max_penetration, contact_count


def simulate(model, scene_geom_ids, spec, arrays, *, max_root_error=0.3,
             max_joint_rmse=0.5, max_penetration=0.03, contact_body_ids=None):
    """Run the entire reference without root assistance or intermediate resets."""
    import mujoco

    fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    ratio = (1 / fps) / model.opt.timestep
    substeps = round(ratio)
    if substeps < 1 or not np.isclose(ratio, substeps, atol=1e-8):
        raise ValueError("motion sampling interval must be an integer number of physics steps")
    if not np.isclose(1 / fps, spec.control_dt):
        raise ValueError("motion fps must match the robot control frequency")
    qids, vids = _joint_indices(model, spec)
    model.dof_armature[vids] = np.asarray(spec.armature)
    # No actuators or applied force on the leading six free-root DOFs.
    if model.nu:
        raise ValueError("diagnostic model must have no actuators")
    indices = [arrays["joint_names"].tolist().index(name) for name in spec.joint_names]
    positions = np.asarray(arrays["joint_pos"], dtype=float)[:, indices]
    velocities = np.asarray(arrays["joint_vel"], dtype=float)[:, indices]
    base_index = arrays["body_names"].tolist().index(spec.base_link)
    roots = np.asarray(arrays["body_pos_w"], dtype=float)[:, base_index]
    quats = np.asarray(arrays["body_quat_w"], dtype=float)[:, base_index]
    frames = len(positions)
    data = mujoco.MjData(model)
    data.qpos[:3], data.qpos[3:7], data.qpos[qids] = roots[0], quats[0], positions[0]
    data.qvel[:3] = arrays["body_lin_vel_w"][0, base_index]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quats[0])
    data.qvel[3:6] = rotation.reshape(3, 3).T @ arrays["body_ang_vel_w"][0, base_index]
    data.qvel[vids] = velocities[0]
    mujoco.mj_forward(model, data)
    controlled = np.array([spec.joint_names.index(name) for name in spec.controlled_joint_names], dtype=int)
    nominal = np.asarray(spec.default_pose, dtype=float)
    kp, kd, limits = map(np.asarray, (spec.stiffness, spec.damping, spec.effort_limits))
    body_ids = np.asarray(contact_body_ids if contact_body_ids is not None else [], dtype=int)
    recorded = {"root_pos_w": [], "root_quat_wxyz": [], "joint_pos": [],
                "joint_vel": [], "root_error_m": [], "controlled_joint_mse_rad2": [],
                "scene_contact_force_w_n": [], "scene_max_penetration_m": [],
                "scene_contact_count": [], "time_s": []}
    peak_penetration = 0.0
    torque_limit_hits, torque_samples = 0, 0
    root_force_max = 0.0
    warnings_before = np.array([warning.number for warning in data.warning])
    failure = None
    for frame in range(frames):
        if frame:
            target = nominal.copy()
            target[controlled] = positions[frame, controlled]
            for _ in range(substeps):
                tau = kp * (target - data.qpos[qids]) - kd * data.qvel[vids]
                torque_limit_hits += int(np.count_nonzero(np.abs(tau) > limits))
                torque_samples += len(tau)
                data.qfrc_applied[:] = 0.0
                data.xfrc_applied[:] = 0.0
                data.qfrc_applied[vids] = np.clip(tau, -limits, limits)
                root_force_max = max(root_force_max, float(np.abs(data.qfrc_applied[:6]).max()))
                mujoco.mj_step(model, data)
                # Forward kinematics/constraints at the final integrated state;
                # mj_step otherwise leaves some derived arrays at its pre-step state.
                mujoco.mj_forward(model, data)
                _, penetration, _ = scene_contacts(model, data, scene_geom_ids, body_ids)
                peak_penetration = max(peak_penetration, penetration)
                if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                    failure = "nonfinite physical state"
                    break
                if np.any(np.array([warning.number for warning in data.warning]) > warnings_before):
                    failure = "MuJoCo numerical warning; possible automatic state reset"
                    break
            if failure:
                break
        forces, penetration, count = scene_contacts(model, data, scene_geom_ids, body_ids)
        peak_penetration = max(peak_penetration, penetration)
        values = dict(root_pos_w=data.qpos[:3].copy(), root_quat_wxyz=data.qpos[3:7].copy(),
                      joint_pos=data.qpos[qids].copy(), joint_vel=data.qvel[vids].copy(),
                      root_error_m=float(np.linalg.norm(data.qpos[:3] - roots[frame])),
                      controlled_joint_mse_rad2=float(np.mean((data.qpos[qids[controlled]] - positions[frame, controlled]) ** 2)),
                      scene_contact_force_w_n=forces, scene_max_penetration_m=penetration,
                      scene_contact_count=count, time_s=float(data.time))
        for key, value in values.items():
            recorded[key].append(value)
    trajectory = {key: np.asarray(value) for key, value in recorded.items()}
    complete = len(trajectory["time_s"]) == frames and failure is None
    root_error = float(np.max(trajectory["root_error_m"])) if len(trajectory["root_error_m"]) else None
    rmse = (float(np.sqrt(np.mean(trajectory["controlled_joint_mse_rad2"])))
            if len(trajectory["controlled_joint_mse_rad2"]) else None)
    reasons = []
    if not complete:
        reasons.append(failure or "incomplete simulation")
    if root_error is None or root_error > max_root_error:
        reasons.append("root-position error exceeds engineering threshold")
    if rmse is None or rmse > max_joint_rmse:
        reasons.append("controlled-joint RMSE exceeds engineering threshold")
    if peak_penetration > max_penetration:
        reasons.append("robot-scene penetration exceeds engineering threshold")
    report = dict(
        schema=1, method="mujoco_free_root_joint_pd_scene_diagnostic",
        accepted=not reasons, rejection_reasons=reasons, complete=complete,
        source_num_frames=frames, recorded_num_frames=len(trajectory["time_s"]),
        fps=fps, physics_dt_s=float(model.opt.timestep), control_substeps=substeps,
        duration_s=(frames - 1) / fps, root_position_max_error_m=root_error,
        controlled_joint_rmse_rad=rmse, robot_scene_max_penetration_m=peak_penetration,
        torque_saturation_fraction=torque_limit_hits / max(1, torque_samples),
        applied_root_force_max=root_force_max, root_initializations=1,
        root_trajectory_overwritten=False, root_assistance=False,
        held_joints_target="spec nominal pose", joint_velocity_target="zero",
        armature="spec", scene_contact_forces="signed world net per body; robot-scene only",
        force_sampling="instantaneous at 50 Hz; not hindsight labels",
        thresholds=dict(root_position_max_error_m=max_root_error,
                        controlled_joint_rmse_rad=max_joint_rmse,
                        robot_scene_max_penetration_m=max_penetration),
        threshold_basis="project engineering checks; not paper acceptance thresholds",
        training_contact_labels_generated=False, isaac_runtime_validated=False,
        mujoco_version=mujoco.__version__,
    )
    return report, trajectory


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False, suffix=".json.tmp") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_reference(motion_path, scene_path, robot="g1", **thresholds):
    from morph_mimic.utils.motion import validate_motion_arrays
    from morph_mimic.utils.reference_scene import load_reference_scene
    from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec

    motion_path, scene_path = Path(motion_path).resolve(), Path(scene_path).resolve()
    spec = RobotSpec.load(robot)
    with np.load(motion_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    validate_motion_arrays(arrays, spec, motion_path)
    scene = load_reference_scene(scene_path, motion_path=motion_path)
    model_path = PROJECT_ROOT / spec.runtime_assets["mjcf"]
    model, scene_geom_ids = build_model(model_path, scene, spec.physics_dt)
    body_names, body_ids = _body_columns(model, spec)
    report, trajectory = simulate(model, scene_geom_ids, spec, arrays,
                                  contact_body_ids=body_ids, **thresholds)
    report.update(robot=robot, motion_path=str(motion_path), source_sha256=_sha256(motion_path),
                  scene_path=str(scene_path), scene_fingerprint=scene.fingerprint,
                  scene_sha256=_sha256(scene_path), reference_model=str(model_path),
                  reference_model_sha256=_sha256(model_path), contact_body_names=body_names,
                  scene_collision_geom_count=len(scene_geom_ids),
                  robot_spec_sha256=_sha256(PROJECT_ROOT / "assets" / robot / "robot_spec.json"),
                  controlled_joint_names=spec.controlled_joint_names,
                  held_joint_names=spec.held_joint_names,
                  joint_stiffness_nm_per_rad=spec.stiffness,
                  joint_damping_nms_per_rad=spec.damping,
                  joint_effort_limits_nm=spec.effort_limits,
                  scene_friction=[1.0, 0.005, 0.0001],
                  scene_coordinates="source world; no independent root/scene recentering")
    trajectory["body_names"] = np.array(body_names)
    trajectory["joint_names"] = np.array(spec.joint_names)
    trajectory["source_sha256"] = np.array(report["source_sha256"])
    trajectory["scene_fingerprint"] = np.array(scene.fingerprint)
    trajectory["diagnostic_only"] = np.array(True)
    return report, trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["g1", "mojia"], default="g1")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON diagnostic report")
    parser.add_argument("--trajectory", type=Path, help="optional raw diagnostic NPZ; never training labels")
    parser.add_argument("--max-root-error", type=float, default=0.3)
    parser.add_argument("--max-joint-rmse", type=float, default=0.5)
    parser.add_argument("--max-penetration", type=float, default=0.03)
    args = parser.parse_args()
    for name in ("max_root_error", "max_joint_rmse", "max_penetration"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.trajectory is not None and args.trajectory.stem.endswith("_contacts"):
        parser.error("diagnostic trajectory must not use the training _contacts filename")
    try:
        report, trajectory = validate_reference(
            args.motion, args.scene, args.robot, max_root_error=args.max_root_error,
            max_joint_rmse=args.max_joint_rmse, max_penetration=args.max_penetration,
        )
        if args.trajectory is not None:
            args.trajectory.parent.mkdir(parents=True, exist_ok=True)
            with args.trajectory.open("wb") as stream:
                np.savez_compressed(stream, **trajectory)
            report["diagnostic_trajectory"] = str(args.trajectory.resolve())
            report["diagnostic_trajectory_sha256"] = _sha256(args.trajectory)
        _atomic_json(args.output, report)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: report[key] for key in (
        "accepted", "complete", "recorded_num_frames", "root_position_max_error_m",
        "controlled_joint_rmse_rad", "robot_scene_max_penetration_m", "rejection_reasons",
    )}, ensure_ascii=False))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
