"""Audit imported Isaac body properties, USD collisions and kinematic FK.

The FK mode deliberately writes root/joint poses without integrating physics.
Passing this audit is not a physical tracking result or a training result.
Run in the Isaac Lab Python environment after preparing the explicit MJCF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from morph_mimic.diagnostics.validate_scene_reference import _atomic_json, _sha256


def inertia_in_link_frame(diagonal, quaternion_wxyz):
    """Rotate COM principal moments into the source link's actor frame."""
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    return rotation @ np.diag(diagonal) @ rotation.T


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _properties(env, profile):
    names = list(env.robot.body_names)
    masses = _numpy(env.robot.root_physx_view.get_masses())[0]
    inertia = _numpy(env.robot.root_physx_view.get_inertias())[0].reshape(len(names), 3, 3)
    com = _numpy(env.robot.root_physx_view.get_coms())[0]
    result = {}
    for index, name in enumerate(names):
        expected_mass = profile["compiled_body_mass_by_name"][name]
        expected_inertia = inertia_in_link_frame(profile["body_inertia_diagonal_by_name"][name],
                                                profile["body_inertia_quat_wxyz_by_name"][name])
        expected_com = np.array(profile["body_center_of_mass_local_m_by_name"][name])
        result[name] = dict(
            source_mass_kg=expected_mass, runtime_mass_kg=float(masses[index]),
            mass_abs_error_kg=float(abs(masses[index] - expected_mass)),
            mass_matches=bool(np.isclose(masses[index], expected_mass, rtol=1e-4, atol=1e-6)),
            source_inertia_link_frame_kgm2=expected_inertia.tolist(),
            runtime_inertia_link_frame_kgm2=inertia[index].tolist(),
            inertia_max_abs_error_kgm2=float(np.max(np.abs(inertia[index] - expected_inertia))),
            inertia_matches=bool(np.allclose(inertia[index], expected_inertia, rtol=1e-3, atol=1e-7)),
            source_com_local_m=expected_com.tolist(), runtime_com_local_m=com[index, :3].tolist(),
            com_error_m=float(np.linalg.norm(com[index, :3] - expected_com)),
            com_matches=bool(np.allclose(com[index, :3], expected_com, rtol=0, atol=1e-5)),
        )
    return dict(by_body=result, mass_matches=all(v["mass_matches"] for v in result.values()),
                inertia_matches=all(v["inertia_matches"] for v in result.values()),
                com_matches=all(v["com_matches"] for v in result.values()),
                runtime_total_mass_kg=float(masses.sum()),
                source_total_mass_kg=sum(profile["compiled_body_mass_by_name"].values()),
                inertia_convention="COM tensor in rigid-body prim axes; source principal moments rotated by inertial quaternion")


def _collisions(env, profile):
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    expected = {geometry["name"]: geometry for geometry in profile["collision_geoms"]}
    actual = []
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        ancestor = prim
        while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            ancestor = ancestor.GetParent()
        body = ancestor.GetName() if ancestor else None
        scale = np.abs(np.array(Gf.Transform(cache.GetLocalToWorldTransform(prim)).GetScale()))
        item = dict(path=str(prim.GetPath()), name=prim.GetName(), type=prim.GetTypeName(),
                    body=body, world_scale=scale.tolist(), instance_proxy=prim.IsInstanceProxy())
        if prim.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(prim)
            axis = str(capsule.GetAxisAttr().Get())
            axis_index = {"X": 0, "Y": 1, "Z": 2}[axis.upper()]
            radial = np.delete(scale, axis_index)
            item.update(radius_m=float(capsule.GetRadiusAttr().Get()) * float(radial.max()),
                        height_m=float(capsule.GetHeightAttr().Get()) * float(scale[axis_index]),
                        axis=axis, radial_scale_uniform=bool(np.isclose(radial[0], radial[1], atol=1e-6)))
        elif prim.IsA(UsdGeom.Sphere):
            sphere = UsdGeom.Sphere(prim)
            item.update(radius_m=float(sphere.GetRadiusAttr().Get()) * float(scale.max()),
                        scale_uniform=bool(np.allclose(scale, scale[0], atol=1e-6)))
        reference = expected.get(item["name"])
        if reference is not None:
            matches = body == reference["body"] and item["type"].lower() == reference["type"]
            if reference["type"] == "capsule":
                matches = matches and item.get("radial_scale_uniform", False)
                matches = matches and bool(np.isclose(item.get("radius_m", -1), reference["size"][0], atol=1e-5))
                matches = matches and bool(np.isclose(item.get("height_m", -1), 2*reference["size"][1], atol=1e-5))
            elif reference["type"] == "sphere":
                matches = matches and item.get("scale_uniform", False)
                matches = matches and bool(np.isclose(item.get("radius_m", -1), reference["size"][0], atol=1e-5))
            else:
                matches = False  # unsupported checks cannot count as verified
            item.update(source=reference, matches_source=bool(matches))
        else:
            item["matches_source"] = False
        actual.append(item)
    found = {item["name"] for item in actual}
    missing = sorted(set(expected) - found)
    extra = sorted(found - set(expected))
    return dict(actual=actual, missing=missing, unexpected=extra,
                expected_count=len(expected), runtime_count=len(actual),
                traverse_instance_proxies=True,
                shape_and_size_match=not missing and not extra and all(item["matches_source"] for item in actual))


def _kinematics(env, spec, motion_path, phases):
    import torch

    with np.load(motion_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    source_bodies, source_joints = arrays["body_names"].tolist(), arrays["joint_names"].tolist()
    body_order = [source_bodies.index(name) for name in env.robot.body_names]
    joint_order = [source_joints.index(name) for name in env.robot.joint_names]
    base = source_bodies.index(spec.base_link)
    translation = _numpy(env.env_origin()[0] + env.cell_offset[0]) - np.array([
        arrays["body_pos_w"][0, base, 0], arrays["body_pos_w"][0, base, 1], 0.])
    result = []
    for phase in phases:
        frame = int(round(phase * (len(arrays["joint_pos"]) - 1)))
        root_pose = np.concatenate([arrays["body_pos_w"][frame, base] + translation,
                                    arrays["body_quat_w"][frame, base]])
        joints = arrays["joint_pos"][frame, joint_order]
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=env.device)
        env.robot.write_root_pose_to_sim(tensor(root_pose[None]))
        env.robot.write_root_link_velocity_to_sim(torch.zeros(1, 6, device=env.device))
        env.robot.write_joint_state_to_sim(tensor(joints[None]), torch.zeros(1, len(joints), device=env.device))
        env.robot.set_joint_position_target(tensor(joints[None]))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(spec.control_dt)
        actual_pos = _numpy(env.robot.data.body_link_pos_w[0])
        expected_pos = arrays["body_pos_w"][frame, body_order] + translation
        errors = np.linalg.norm(actual_pos - expected_pos, axis=-1)
        result.append(dict(phase=phase, frame=frame,
                           max_body_position_error_m=float(errors.max()),
                           mean_body_position_error_m=float(errors.mean()),
                           body_position_error_m=dict(zip(env.robot.body_names, errors.tolist())),
                           position_matches=bool(errors.max() <= 1e-4)))
    return dict(mode="teleport reference root and all 29 joints; forward kinematics only; no physics integration",
                phases=result, source_to_runtime_translation_m=translation.tolist(),
                matches=all(item["position_matches"] for item in result),
                physical_tracking_validated=False)


def audit(args):
    from morph_mimic.scripts.common import build_env
    from morph_mimic.utils.specs import PROJECT_ROOT

    profile_path = PROJECT_ROOT / ".cache/mjcf" / args.robot / "g1_explicit.json"
    profile = json.loads(profile_path.read_text())
    env = None
    try:
        env, spec = build_env(1, args.robot, 1, device=args.device, use_usd=False,
                              require_contacts=False, motion_dir=str(args.motion_dir), clips=[args.clip],
                              evaluation=True, seed=0, dr_torso_mass=(0., 0.), dr_gain_scale=(1., 1.),
                              episode_length_s=20.)
        env.reset()
        body_names, joint_names = list(env.robot.body_names), list(env.robot.joint_names)
        report = dict(body_names=body_names, joint_names=joint_names,
                      body_names_match=set(body_names) == set(profile["body_names"]),
                      joint_names_match=set(joint_names) == set(profile["joint_names"]),
                      explicit_model_provenance=str(profile_path),
                      explicit_model_sha256=profile["derived_sha256"],
                      source_motion_sha256=_sha256(args.motion_dir / f"{args.clip}.npz"),
                      kinematic_only=True, physical_tracking_validated=False,
                      training_contact_labels_generated=False)
        if report["body_names_match"]:
            report["physical_properties"] = _properties(env, profile)
            report["kinematics"] = _kinematics(env, spec, args.motion_dir / f"{args.clip}.npz", args.phases)
        report["collision_geometry"] = _collisions(env, profile)
        report["accepted"] = bool(
            report["body_names_match"] and report["joint_names_match"]
            and report.get("physical_properties", {}).get("mass_matches", False)
            and report.get("physical_properties", {}).get("inertia_matches", False)
            and report.get("physical_properties", {}).get("com_matches", False)
            and report.get("kinematics", {}).get("matches", False)
            and report["collision_geometry"]["shape_and_size_match"])
        return report
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["g1"], default="g1")
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phases", type=float, nargs="+", default=[0., .25, .5, .75])
    args = parser.parse_args()
    if any(not 0 <= phase <= 1 for phase in args.phases):
        parser.error("phases must lie in [0,1]")
    args.motion_dir = args.motion_dir.resolve()
    report = dict(schema=1, completed=False, accepted=False,
                  method="Isaac runtime reference asset audit", physical_tracking_validated=False)
    _atomic_json(args.output, report)
    app = None
    try:
        from morph_mimic.launcher import launch
        app = launch(headless=True, device=args.device)
        report.update(audit(args), completed=True)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        _atomic_json(args.output, report)
        print(json.dumps({key: report.get(key) for key in ("completed", "accepted", "error")}), flush=True)
        if app is not None:
            # Fast Kit shutdown must retain the audit result and flush logs.
            app.app.post_quit(0 if report["completed"] and report["accepted"] else 1)
            from morph_mimic.launcher import close
            close(app)


if __name__ == "__main__":
    main()
