"""Append an explicitly derived smooth settling segment to a scene-bound clip.

The original arrays/prefix remain byte-identical.  A geometric IK target is
not a physical hold demonstration. This production helper runs CPU reference
FK only; callers must separately validate scene/self collisions and regenerate
the complete contact labels before training from its output.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from morph_mimic.utils.motion import validate_motion_arrays
from morph_mimic.utils.reference_scene import load_reference_scene
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _multiply(a, b):
    return np.concatenate((a[..., :1]*b[..., :1]-(a[..., 1:]*b[..., 1:]).sum(-1, keepdims=True),
        a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+np.cross(a[..., 1:], b[..., 1:])), -1)


def _rotation_vector(q):
    q = q/np.linalg.norm(q)
    q = q if q[0] >= 0 else -q
    norm = np.linalg.norm(q[1:])
    return 2*q[1:] if norm < 1e-10 else q[1:]*(2*np.arctan2(norm, q[0])/norm)


def _quat_exp(vectors):
    angle = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.full_like(angle, .5)
    np.divide(np.sin(angle/2), angle, out=scale, where=angle > 1e-10)
    return np.concatenate((np.cos(angle/2), vectors*scale), -1)


def _world_angular_velocity(phi, derivative):
    """SO(3) left Jacobian for q(t)=Exp(phi(t))*q_start, both in world axes."""
    theta2 = np.sum(phi*phi, axis=-1, keepdims=True)
    theta = np.sqrt(theta2)
    a, b = np.full_like(theta, .5), np.full_like(theta, 1/6)
    np.divide(1-np.cos(theta), theta2, out=a, where=theta2 > 1e-10)
    np.divide(theta-np.sin(theta), theta2*theta, out=b, where=theta2 > 1e-10)
    cross = np.cross(phi, derivative)
    return derivative+a*cross+b*np.cross(phi, cross)


def _hermite(start, start_velocity, target, times, duration):
    u = times[:, None]/duration
    position = ((2*u**3-3*u**2+1)*start+(u**3-2*u**2+u)*duration*start_velocity
        +(-2*u**3+3*u**2)*target)
    velocity = ((6*u**2-6*u)/duration*start+(3*u**2-4*u+1)*start_velocity
        +(-6*u**2+6*u)/duration*target)
    return position, velocity


def append_settle_reference(source_motion, target_json, output_dir, settle_seconds=1.):
    """Return ``(motion_path, provenance)`` without modifying any source file.

    Target schema: accepted=true, robot=g1, source_motion_sha256, target_sha256
    (canonical sorted compact JSON hash), and target dict with joint_names,
    joint_pos_rad, root_pos_world_m and root_quat_wxyz. Held coordinates must
    equal the source nominal. Source motion and its complete asset binding are
    verified before processing. Outputs refuse overwrite; labels are not copied.
    """
    import mujoco

    source_motion, target_json, output_dir = map(lambda p: Path(p).resolve(), (source_motion, target_json, output_dir))
    scene = load_reference_scene(source_motion.with_suffix(".scene.json"), source_motion)
    spec = RobotSpec.load("g1")
    solution = json.loads(target_json.read_text())
    if solution.get("accepted") is not True or solution.get("robot") != "g1":
        raise ValueError("settle target must be an explicitly accepted source-G1 solution")
    if solution.get("source_motion_sha256") != _sha(source_motion):
        raise ValueError("stable target is bound to a different source motion")
    target = solution["target"]
    payload = json.dumps(target, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(payload).hexdigest() != solution.get("target_sha256"):
        raise ValueError("stable target canonical payload SHA256 mismatch")
    for path, expected in solution.get("source_sha256", {}).items():
        if _sha(path) != expected:
            raise ValueError(f"stable target source dependency changed: {path}")
    with np.load(source_motion, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    validate_motion_arrays(source, spec, source_motion)
    fps = float(source["fps"])
    count = round(float(settle_seconds)*fps)
    if not np.isfinite(settle_seconds) or settle_seconds <= 0 or count < 1 or not np.isclose(count, settle_seconds*fps, rtol=0, atol=1e-8):
        raise ValueError("settle duration must be positive and an integer number of source-rate intervals")
    duration, original_frames = count/fps, len(source["joint_pos"])
    name = source_motion.stem+"_settle_v3"
    destination = output_dir/(name+".npz")
    sidecar_path = destination.with_suffix(".scene.json")
    provenance_path = destination.with_suffix(".settle.json")
    for path in (destination, sidecar_path, provenance_path, output_dir/(name+"_contacts.npz")):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite derived reference output: {path}")
    joint_names, body_names = source["joint_names"].tolist(), source["body_names"].tolist()
    if len(target["joint_names"]) != len(set(target["joint_names"])) or set(target["joint_names"]) != set(joint_names):
        raise ValueError("settle target must contain the complete unique source named joint set")
    target_columns = [target["joint_names"].index(name) for name in joint_names]
    target_q = np.asarray(target["joint_pos_rad"], dtype=float)[target_columns]
    target_root = np.asarray(target["root_pos_world_m"], dtype=float)
    target_quat = np.asarray(target["root_quat_wxyz"], dtype=float)
    if target_q.shape != (29,) or target_root.shape != (3,) or target_quat.shape != (4,) or not all(np.isfinite(v).all() for v in (target_q,target_root,target_quat)):
        raise ValueError("settle target requires finite 29-joint, XYZ and WXYZ state")
    if abs(np.linalg.norm(target_quat)-1) > 1e-5:
        raise ValueError("settle target quaternion must be normalized WXYZ")
    target_quat /= np.linalg.norm(target_quat)
    held = [joint_names.index(name) for name in spec.held_joint_names]
    nominal = np.asarray(spec.default_pose)[[spec.joint_names.index(name) for name in spec.held_joint_names]]
    if np.max(np.abs(target_q[held]-nominal)) > 1e-8:
        raise ValueError("settle target changes nominal held joints")
    model_path = PROJECT_ROOT/spec.runtime_assets["mjcf"]
    model, base = mujoco.MjModel.from_xml_path(str(model_path)), body_names.index(spec.base_link)
    data = mujoco.MjData(model)
    qids = np.asarray([model.joint(name).qposadr[0] for name in joint_names])
    vids = np.asarray([model.joint(name).dofadr[0] for name in joint_names])
    bids = np.asarray([model.body(name).id for name in body_names])
    if model.nq != 36 or model.nv != 35 or set(joint_names) != set(spec.joint_names):
        raise ValueError("settle FK requires the complete 29-axis free-root G1 model")
    times = np.arange(1,count+1)/fps
    q, dq = _hermite(source["joint_pos"][-1].astype(float), source["joint_vel"][-1].astype(float), target_q, times, duration)
    root, root_velocity = _hermite(source["root_pos"][-1].astype(float), source["body_lin_vel_w"][-1,base].astype(float), target_root, times, duration)
    q[:,held], dq[:,held] = nominal, 0.
    start_quat = source["root_rot_wxyz"][-1].astype(float)
    start_quat /= np.linalg.norm(start_quat)
    target_quat = target_quat if start_quat@target_quat >= 0 else -target_quat
    conjugate = start_quat*np.array([1.,-1.,-1.,-1.])
    rotation_delta = _rotation_vector(_multiply(target_quat, conjugate))
    phi, phi_dot = _hermite(np.zeros(3), source["body_ang_vel_w"][-1,base].astype(float), rotation_delta, times, duration)
    root_quat = _multiply(_quat_exp(phi), start_quat)
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
    root_angular_velocity = _world_angular_velocity(phi, phi_dot)
    # Exact stationary endpoint, including all hinge and free-root velocities.
    q[-1], root[-1], root_quat[-1] = target_q, target_root, target_quat
    q[-1,held], dq[-1], root_velocity[-1], root_angular_velocity[-1] = nominal, 0., 0., 0.
    arrays = {key: value.copy() for key,value in source.items()}
    extension = dict(joint_pos=q, joint_vel=dq, root_pos=root, root_rot_wxyz=root_quat)
    for key,width in (("body_pos_w",3),("body_quat_w",4),("body_lin_vel_w",3),("body_ang_vel_w",3)):
        extension[key] = np.empty((count,len(body_names),width))
    spatial, rotation = np.empty(6), np.empty(9)
    for frame in range(count):
        data.qpos[:3], data.qpos[3:7], data.qpos[qids] = root[frame], root_quat[frame], q[frame]
        data.qvel[:3] = root_velocity[frame]
        mujoco.mju_quat2Mat(rotation, root_quat[frame])
        data.qvel[3:6], data.qvel[vids] = rotation.reshape(3,3).T@root_angular_velocity[frame], dq[frame]
        mujoco.mj_forward(model,data)
        extension["body_pos_w"][frame], extension["body_quat_w"][frame] = data.xpos[bids],data.xquat[bids]
        for column,bi in enumerate(bids):
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,int(bi),spatial,0)
            extension["body_ang_vel_w"][frame,column] = spatial[:3]
            extension["body_lin_vel_w"][frame,column] = spatial[3:]+np.cross(spatial[:3],data.xpos[bi]-data.xipos[bi])
    for key,value in source.items():
        if value.ndim and value.shape[0] == original_frames and key not in extension:
            raise ValueError(f"unhandled frame-indexed source array {key}; refusing to truncate metadata")
    for key,value in extension.items():
        arrays[key] = np.concatenate((source[key],value.astype(source[key].dtype)),axis=0)
    validate_motion_arrays(arrays,spec,destination)
    preservation = {}
    for key,value in source.items():
        prefix = arrays[key][:original_frames] if key in extension else arrays[key]
        if prefix.dtype != value.dtype or prefix.shape != value.shape or prefix.tobytes() != value.tobytes():
            raise RuntimeError(f"settle extension altered original array {key}")
        preservation[key] = dict(dtype=str(value.dtype),shape=list(value.shape),sha256=_array_sha(value))
    metadata = dict(schema=1,method="source_prefix_plus_C1_Hermite_SO3_settle_v1",robot="g1",
        source_motion=str(source_motion),source_motion_sha256=_sha(source_motion),
        source_scene=str(scene.sidecar_path),source_scene_sha256=_sha(scene.sidecar_path),
        stable_target_json=str(target_json),stable_target_json_sha256=_sha(target_json),
        stable_target_sha256=solution["target_sha256"],source_prefix_preserved=True,
        source_array_prefix_sha256=preservation,source_frames=original_frames,added_settle_frames=count,
        num_frames=original_frames+count,fps=fps,settle_duration_s=duration,
        total_duration_s=(original_frames+count-1)/fps,
        reference_model=str(model_path),reference_model_sha256=_sha(model_path),
        algorithm_source_sha256=_sha(Path(__file__)),mujoco_version=mujoco.__version__,
        interpolation="C1 Cartesian/hinge cubic Hermite; antipodal-safe SO3 exponential Hermite and exact left-Jacobian world angular velocity",
        frame_convention="source world positions; origin linear/world angular velocities; free angular qvel transformed to root local",
        held_joints="all six source nominal, zero velocity",contact_labels_generated=False,
        geometric_collision_validated=False,physical_tracking_validated=False,
        scope="Local derived one-second settle reference; original 0-10s unchanged. Not paper data or a physical demonstration.")
    arrays["settle_metadata_json"] = np.array(json.dumps(metadata,sort_keys=True))
    sidecar = deepcopy(scene.metadata)
    sidecar.update(motion_file=destination.name,motion_sha256=None,contact_labels_generated=False,
        settle_extension={key:metadata[key] for key in ("source_motion","source_motion_sha256","stable_target_json",
            "stable_target_json_sha256","stable_target_sha256","source_prefix_preserved","source_frames",
            "added_settle_frames","settle_duration_s","scope")})
    # Preserve the bound original raw reference, including portable absolute
    # paths if the input sidecar happened to use relative dependencies.
    sidecar["source_motion"]["path"] = str((scene.sidecar_path.parent/sidecar["source_motion"]["path"]).resolve())
    sidecar["scene_urdf"]["path"] = str(scene.scene_urdf_path)
    sidecar["collision_asset_sha256"] = {str((scene.sidecar_path.parent/p).resolve()):sha for p,sha in sidecar["collision_asset_sha256"].items()}
    output_dir.mkdir(parents=True,exist_ok=True)
    temporaries,created = [],[]
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir,suffix=".npz.tmp",delete=False) as stream:
            temporaries.append(Path(stream.name))
            np.savez(stream,**arrays)
        metadata["derived_motion_sha256"] = _sha(temporaries[0])
        sidecar["motion_sha256"] = metadata["derived_motion_sha256"]
        for document in (sidecar,metadata):
            with tempfile.NamedTemporaryFile(mode="w",dir=output_dir,suffix=".json.tmp",delete=False) as stream:
                temporaries.append(Path(stream.name))
                stream.write(json.dumps(document,indent=2,sort_keys=True,allow_nan=False)+"\n")
        for temporary,destination_path in zip(temporaries,(destination,sidecar_path,provenance_path)):
            os.link(temporary,destination_path)
            created.append(destination_path)
        load_reference_scene(sidecar_path,destination)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    finally:
        for path in temporaries:
            path.unlink(missing_ok=True)
    return destination,metadata
