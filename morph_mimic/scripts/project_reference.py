"""Build a derived reference whose held joints match the controller.

    python -m morph_mimic.scripts.project_reference --robot g1 \
        --source-dir datasets/g1_walk_candidate --output-dir datasets/g1_walk_projected

The complete joint vector and all named body kinematics are retained. MuJoCo
provides reference FK only; this does not validate the Isaac Lab runtime asset,
contact dynamics, or tracking. Source assets and labels are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from morph_mimic.utils.motion import validate_motion_arrays
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_clip(source_path: Path, output_dir: Path, spec: RobotSpec):
    """Set held q to nominal and held dq to zero, then rebuild named FK.

    Input and output body velocities are world-frame velocities of the body
    frame origin. MuJoCo free-joint angular qvel is root-local, and
    mj_objectVelocity(mjOBJ_BODY) reports velocity at body COM, so both
    conversions are explicit below.
    """
    try:
        import mujoco
    except ImportError as exc:
        raise ValueError("reference projection requires MuJoCo; run in an environment with mujoco installed") from exc

    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    destination = output_dir / source_path.name
    provenance_path = destination.with_suffix(".projection.json")
    if source_path.stem.endswith("_contacts"):
        raise ValueError("select a motion reference, not contact labels")
    if not spec.held_joint_names:
        raise ValueError(f"{spec.name} has no held joints to project")
    for existing in (destination, provenance_path, output_dir / f"{source_path.stem}_contacts.npz"):
        if existing.exists():
            raise FileExistsError(f"refusing to overwrite or reuse labeled output: {existing}")
    with np.load(source_path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    if source_path.with_suffix(".scene.json").exists() or bool(np.asarray(source.get("requires_scene", False)).item()):
        raise ValueError("scene-bound references require scene-aware conversion; generic projection cannot preserve their binding")
    validate_motion_arrays(source, spec, source_path, require_nominal_held=False)
    model_relative = spec.runtime_assets.get("mjcf")
    if not model_relative:
        raise ValueError("projection requires the robot's explicit local MJCF reference model")
    model_path = (PROJECT_ROOT / model_relative).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_names, body_names = source["joint_names"].tolist(), source["body_names"].tolist()
    compiled_joints = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    compiled_bodies = [model.body(i).name for i in range(1, model.nbody)]
    if set(compiled_joints) != set(joint_names) or len(compiled_joints) != len(joint_names):
        raise ValueError("reference MJCF must contain exactly the complete named joint set")
    if set(compiled_bodies) != set(body_names) or len(compiled_bodies) != len(body_names):
        raise ValueError("reference MJCF and motion must contain exactly the same named bodies")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0 or model.jnt_dofadr[free[0]] != 0:
        raise ValueError("reference MJCF requires one leading free root")
    if model.body(model.jnt_bodyid[free[0]]).name != spec.base_link:
        raise ValueError("reference MJCF free-root body differs from robot spec")
    for name in joint_names:
        if model.joint(name).type[0] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"{name}: only scalar joints are supported")
    qids = np.array([model.joint(name).qposadr[0] for name in joint_names])
    vids = np.array([model.joint(name).dofadr[0] for name in joint_names])
    bids = np.array([model.body(name).id for name in body_names])
    held = [joint_names.index(name) for name in spec.held_joint_names]
    nominal = np.asarray(spec.default_pose)[[spec.joint_names.index(name) for name in spec.held_joint_names]]
    base = body_names.index(spec.base_link)
    root_pos = np.asarray(source.get("root_pos", source["body_pos_w"][:, base]), dtype=np.float64)
    root_quat = np.asarray(source.get("root_rot_wxyz", source["body_quat_w"][:, base]), dtype=np.float64)
    q, dq = source["joint_pos"].astype(np.float32), source["joint_vel"].astype(np.float32)
    held_max_change = np.max(np.abs(q[:, held] - nominal), axis=0)
    q[:, held], dq[:, held] = nominal, 0.0
    frames, bodies = len(q), len(body_names)
    arrays = dict(fps=source["fps"].copy(), joint_names=source["joint_names"].copy(),
                  body_names=source["body_names"].copy(), joint_pos=q, joint_vel=dq,
                  root_pos=root_pos.astype(np.float32), root_rot_wxyz=root_quat.astype(np.float32))
    for key, width in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
        arrays[key] = np.empty((frames, bodies, width), dtype=np.float32)
    spatial, rotation = np.empty(6), np.empty(9)
    for t in range(frames):
        data.qpos[:3], data.qpos[3:7], data.qpos[qids] = root_pos[t], root_quat[t], q[t]
        data.qvel[:3] = source["body_lin_vel_w"][t, base]
        mujoco.mju_quat2Mat(rotation, root_quat[t])
        data.qvel[3:6] = rotation.reshape(3, 3).T @ source["body_ang_vel_w"][t, base]
        data.qvel[vids] = dq[t]
        mujoco.mj_forward(model, data)
        arrays["body_pos_w"][t], arrays["body_quat_w"][t] = data.xpos[bids], data.xquat[bids]
        for body_index, body_id in enumerate(bids):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), spatial, 0)
            arrays["body_ang_vel_w"][t, body_index] = spatial[:3]
            arrays["body_lin_vel_w"][t, body_index] = spatial[3:] + np.cross(
                spatial[:3], data.xpos[body_id] - data.xipos[body_id])
    validate_motion_arrays(arrays, spec, destination)
    profile = dict(schema=1, name=f"{spec.name}_{spec.n_controlled}dof_nominal_hold_v1",
                   controlled_joint_names=spec.controlled_joint_names,
                   held_joint_nominal_rad=dict(zip(spec.held_joint_names, nominal.tolist())),
                   reference_model_sha256=_sha256(model_path),
                   velocity_convention="body-frame origin linear/world angular velocities; both in world axes")
    profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    spec_path = PROJECT_ROOT / "assets" / spec.name / "robot_spec.json"
    metadata = dict(
        schema=1, source_motion=str(source_path), source_sha256=_sha256(source_path),
        profile=profile, profile_sha256=profile_hash,
        robot_spec_sha256=_sha256(spec_path), reference_model=str(model_path),
        model_dependency_sha256={str(path.relative_to(model_path.parent)): _sha256(path)
                                 for path in sorted(model_path.parent.rglob("*")) if path.is_file()},
        held_joint_max_change_rad=dict(zip(spec.held_joint_names, held_max_change.tolist())),
        source_metadata={key: str(source[key].item()) for key in source if key.endswith("_metadata_json") and source[key].ndim == 0},
        source_fields_discarded=sorted(set(source) - set(arrays)),
        num_frames=frames, fps=float(np.asarray(source["fps"]).reshape(-1)[0]),
        duration_s=(frames - 1) / float(np.asarray(source["fps"]).reshape(-1)[0]),
        joint_order="source names preserved; complete physical state retained",
        root_trajectory_changed=False, controlled_joint_trajectory_changed=False,
        reference_only=True, contact_labels_generated=False, contact_labels_inherited=False,
        physical_tracking_validated=False, isaac_runtime_asset_validated=False, mujoco_version=mujoco.__version__,
    )
    arrays["projection_metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths, created = [], []
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz.tmp", delete=False) as stream:
            temporary_paths.append(Path(stream.name))
            np.savez(stream, **arrays)
        metadata["projected_motion_sha256"] = _sha256(temporary_paths[0])
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, suffix=".json.tmp", delete=False) as stream:
            temporary_paths.append(Path(stream.name))
            stream.write(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        for temporary, target in zip(temporary_paths, (destination, provenance_path)):
            os.link(temporary, target)
            created.append(target)
    except BaseException:
        for target in created:
            target.unlink()
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return destination, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["g1", "mojia"], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", help="optional clip names, without .npz")
    args = parser.parse_args()
    try:
        spec = RobotSpec.load(args.robot)
        files = [path for path in sorted(args.source_dir.glob("*.npz")) if not path.stem.endswith("_contacts")]
        if args.clips:
            missing = set(args.clips) - {path.stem for path in files}
            if missing:
                raise ValueError(f"unknown clips: {', '.join(sorted(missing))}")
            files = [path for path in files if path.stem in args.clips]
        if not files:
            raise ValueError("source directory contains no selected motion references")
        for source in files:
            output, metadata = project_clip(source, args.output_dir, spec)
            print(f"Projected {output}: {metadata['num_frames']} frames, {metadata['duration_s']:g} s, {metadata['profile']['name']}")
        print("Recompute force contact labels from complete physical replays before training.")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
