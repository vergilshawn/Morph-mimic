"""Explicit engineering completion checks for the bound G1 box-climb task.

These checks supplement trajectory tracking; they are not ACP paper rewards.
Positions use scene coordinates with the source first-root XY subtracted.
The loader verifies the actual mesh's rectangular, horizontal top instead of
using its world-axis bounding box. Runtime metrics never infer force from FK.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from morph_mimic.utils.reference_scene import load_reference_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
CONTACT_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_elbow_link",
    "right_elbow_link", "left_knee_link", "right_knee_link", *FOOT_BODY_NAMES, "pelvis")
BOX_TASK_THRESHOLDS = dict(root_height_error_max_m=.10, root_tilt_max_rad=.30,
    foot_support_margin_min_m=.01, sole_plane_error_min_m=-.01, sole_plane_error_max_m=.02,
    foot_current_positive_fz_min_n=20., nonfoot_current_positive_fz_max_n=5.,
    continuous_hold_s=1.)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(value, count, description):
    array = np.fromstring(value, sep=" ") if isinstance(value, str) else np.asarray(value, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{description} requires {count} finite numbers")
    return array


def _rotate_np(quaternion, vectors):
    xyz = quaternion[..., 1:]
    twice_cross = 2 * np.cross(xyz, vectors)
    return vectors + quaternion[..., :1] * twice_cross + np.cross(xyz, twice_cross)


def _convex_hull_2d(points):
    # Monotonic chain removes duplicate/collinear mesh vertices without scipy.
    points = sorted(set(map(tuple, np.round(points, 10))))
    if len(points) < 3:
        raise ValueError("top surface has fewer than three distinct XY vertices")
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for point in points:
        while len(lower) > 1 and cross(lower[-2], lower[-1], point) <= 1e-10:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) > 1 and cross(upper[-2], upper[-1], point) <= 1e-10:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _rectangle_top(component, first_xy):
    """Prove a filled rectangular top from mesh triangles, allowing yaw."""
    triangles = np.asarray(component.mesh.triangles, dtype=float)
    top_z = float(triangles[..., 2].max())
    top = triangles[np.all(np.isclose(triangles[..., 2], top_z, rtol=0, atol=1e-7), axis=-1)]
    if not len(top):
        raise ValueError(f"{component.name}: no horizontal top triangles")
    hull = _convex_hull_2d(top[..., :2].reshape(-1, 2))
    if len(hull) != 4:
        raise ValueError(f"{component.name}: horizontal top is not rectangular")
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=-1)
    unit = edges / lengths[:, None]
    if (np.any(lengths < 1e-4) or np.max(np.abs(np.sum(unit * np.roll(unit, -1, axis=0), axis=-1))) > 1e-5
            or not np.allclose(lengths[:2], lengths[2:], rtol=1e-5, atol=1e-7)):
        raise ValueError(f"{component.name}: top polygon does not have rectangle edges")
    rectangle_area = float(lengths[0] * lengths[1])
    triangle_area = float(np.linalg.norm(np.cross(top[:, 1]-top[:, 0], top[:, 2]-top[:, 0]), axis=-1).sum()/2)
    if not np.isclose(rectangle_area, triangle_area, rtol=1e-5, atol=1e-8):
        raise ValueError(f"{component.name}: top triangles do not fill one rectangle")
    return dict(component_name=component.name, component_link_name=component.link_name,
        top_z_m=top_z, center_xy_m=(hull.mean(axis=0)-first_xy).tolist(),
        axes_xy=unit[:2].tolist(), half_extents_m=(lengths[:2]/2).tolist(),
        vertices_xy_m=(hull-first_xy).tolist(), triangle_area_m2=triangle_area,
        rectangle_verified_from="coplanar top triangles, four orthogonal hull edges, matching surface area")


def _foot_geometry(robot_xml):
    source = robot_xml.read_text(encoding="utf-8")
    if "<!DOCTYPE" in source.upper() or "<!ENTITY" in source.upper():
        raise ValueError("robot XML DTD/entity declarations are unsupported")
    tree = ET.fromstring(source)
    if tree.findall("include"):
        raise ValueError("robot XML includes require explicit geometry/dependency handling")
    defaults = {}
    def visit_default(element, inherited):
        attributes = dict(inherited)
        if element.find("geom") is not None:
            attributes.update(element.find("geom").attrib)
        if element.get("class"):
            defaults[element.get("class")] = attributes
        for child in element.findall("default"):
            visit_default(child, attributes)
    for element in tree.findall("default"):
        visit_default(element, {})
    endpoints, radii, sites = [], [], []
    capsule_names = []
    for side, name in zip(("left", "right"), FOOT_BODY_NAMES):
        body = tree.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"source robot lacks foot body {name}")
        site = body.find(f"site[@name='{side}_foot']")
        if site is None:
            raise ValueError(f"source robot lacks {side}_foot support-centre site")
        sites.append(_vector(site.get("pos", "0 0 0"), 3, "foot site").tolist())
        local_endpoints, local_radii, names = [], [], []
        for geom in body.findall("geom"):
            properties = defaults.get(geom.get("class"), {}) | geom.attrib
            if properties.get("contype", "1") == "0" and properties.get("conaffinity", "1") == "0":
                continue
            if properties.get("type") != "capsule" or "fromto" not in properties:
                raise ValueError(f"{name}: collision geometry is not a supported explicit capsule")
            size = np.fromstring(properties.get("size", ""), sep=" ")
            if not len(size) or not np.isfinite(size).all() or size[0] <= 0:
                raise ValueError(f"{name}: foot capsule radius must be finite and positive")
            local_endpoints.append(_vector(properties["fromto"], 6, "capsule fromto").reshape(2, 3).tolist())
            local_radii.append(float(size[0]))
            names.append(geom.get("name"))
        if not local_endpoints:
            raise ValueError(f"{name}: no collision capsules")
        endpoints.append(local_endpoints)
        radii.append(local_radii)
        capsule_names.append(names)
    if len(endpoints[0]) != len(endpoints[1]):
        raise ValueError("left and right feet need the same capsule count for batched metrics")
    compiler = tree.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    dependencies = {str(robot_xml): _sha256(robot_xml)}
    for mesh in tree.findall("./asset/mesh"):
        if mesh.get("file"):
            path = (robot_xml.parent / meshdir / mesh.get("file")).resolve()
            dependencies[str(path)] = _sha256(path)
    return dict(body_names=list(FOOT_BODY_NAMES), capsule_names=capsule_names,
        capsule_endpoints_local_m=endpoints, capsule_radii_m=radii,
        support_center_site_local_m=sites), dependencies


def _geometry_np(root_quat, foot_pos, foot_quat, foot_geometry, top):
    endpoints = np.asarray(foot_geometry["capsule_endpoints_local_m"])
    radii = np.asarray(foot_geometry["capsule_radii_m"])
    world = foot_pos[:, None, None, :] + _rotate_np(foot_quat[:, None, None, :], endpoints)
    sole = (world[..., 2].min(axis=-1) - radii).min(axis=-1)
    sites = foot_pos + _rotate_np(foot_quat, np.asarray(foot_geometry["support_center_site_local_m"]))
    local = (sites[:, :2]-top["center_xy_m"]) @ np.asarray(top["axes_xy"]).T
    margin = (np.asarray(top["half_extents_m"])-np.abs(local)).min(axis=-1)
    tilt = float(np.arccos(np.clip(_rotate_np(root_quat, np.array([0., 0., 1.]))[2], -1, 1)))
    return dict(root_tilt_rad=tilt, foot_support_margin_m=margin.tolist(),
        foot_sole_plane_error_m=(sole-top["top_z_m"]).tolist(),
        foot_support_center_pos_rebased_m=sites.tolist())


def load_box_task_spec(motion_path, robot_xml=None):
    """Load a source-bound JSON specification; reject invalid reference endpoints.

    This checks reference *geometry* only. A feasible reference force solution,
    successful physical playback, or learned policy is not implied.
    """
    motion_path = Path(motion_path).resolve()
    scene = load_reference_scene(motion_path.with_suffix(".scene.json"), motion_path)
    if scene.metadata["robot"] != "g1":
        raise ValueError("box completion metrics currently support the source G1 model only")
    robot_xml = Path(robot_xml or PROJECT_ROOT / "assets/g1/robot/g1.xml").resolve()
    geometry, robot_dependencies = _foot_geometry(robot_xml)
    with np.load(motion_path, allow_pickle=False) as archive:
        names = archive["body_names"].tolist()
        columns = [names.index(name) for name in FOOT_BODY_NAMES]
        root_pos = np.asarray(archive["root_pos"][-1], dtype=float)
        root_quat = np.asarray(archive["root_rot_wxyz"][-1], dtype=float)
        foot_pos = np.asarray(archive["body_pos_w"][-1, columns], dtype=float)
        foot_quat = np.asarray(archive["body_quat_w"][-1, columns], dtype=float)
        fps = float(archive["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("reference motion FPS must be finite and positive")
        duration = (len(archive["root_pos"])-1)/fps
    if (not all(np.isfinite(value).all() for value in (root_pos, root_quat, foot_pos, foot_quat))
            or not np.isclose(np.linalg.norm(root_quat), 1, atol=1e-5)
            or not np.allclose(np.linalg.norm(foot_quat, axis=-1), 1, atol=1e-5)):
        raise ValueError("reference endpoint needs finite positions and unit WXYZ quaternions")
    root_pos[:2] -= scene.source_first_root_xy
    foot_pos[:, :2] -= scene.source_first_root_xy
    candidates, rejected = [], []
    thresholds = BOX_TASK_THRESHOLDS
    for component in scene.components:
        try:
            top = _rectangle_top(component, scene.source_first_root_xy)
            metrics = _geometry_np(root_quat, foot_pos, foot_quat, geometry, top)
            errors = np.asarray(metrics["foot_sole_plane_error_m"])
            if (top["top_z_m"] <= scene.ground.top_z_m + .05
                    or np.any(np.asarray(metrics["foot_support_margin_m"]) < thresholds["foot_support_margin_min_m"])
                    or np.any(errors < thresholds["sole_plane_error_min_m"])
                    or np.any(errors > thresholds["sole_plane_error_max_m"])):
                raise ValueError(f"endpoint feet fail target-top geometry: {metrics}")
            candidates.append((top, metrics))
        except ValueError as exc:
            rejected.append(dict(component=component.name, reason=str(exc)))
    if len(candidates) != 1:
        raise ValueError(f"expected one rectangular obstacle top supporting both reference endpoint feet; "
                         f"found {len(candidates)}; rejected={rejected}")
    top, reference_geometry = candidates[0]
    if reference_geometry["root_tilt_rad"] > thresholds["root_tilt_max_rad"]:
        raise ValueError(f"reference endpoint root tilt fails explicit task gate: {reference_geometry}")
    sources = robot_dependencies | {str(scene.sidecar_path): _sha256(scene.sidecar_path),
        str(motion_path): scene.motion_sha256, str(scene.scene_urdf_path): scene.scene_urdf_sha256,
        str(PROJECT_ROOT / "assets/g1/robot_spec.json"): _sha256(PROJECT_ROOT / "assets/g1/robot_spec.json")}
    sources.update(scene.metadata["collision_asset_sha256"])
    return dict(schema=1, method="source_geometry_box_top_and_current_force_task_gate_v1",
        scope="Engineering box completion supplement; not an ACP paper metric or dynamics proof",
        positions_frame="scene_rebased_by_source_first_root_xy", source_first_root_xy=scene.source_first_root_xy.tolist(),
        scene_fingerprint=scene.fingerprint, motion_path=str(motion_path), motion_sha256=scene.motion_sha256,
        robot_xml=str(robot_xml), source_sha256=sources, contact_body_names=list(CONTACT_BODY_NAMES),
        target_top=top, foot_geometry=geometry, thresholds=dict(thresholds),
        final_reference=dict(root_pos_rebased_m=root_pos.tolist(), root_quat_wxyz=root_quat.tolist(),
            foot_body_pos_rebased_m=foot_pos.tolist(), foot_body_quat_wxyz=foot_quat.tolist(), duration_s=duration),
        reference_endpoint_geometry=reference_geometry, reference_endpoint_geometry_passed=True,
        reference_endpoint_forces_verified=False,
        force_definition="current endpoint positive world Fz from terrain-filtered normal-force sensors; not history flags or full friction wrench",
        support_definition="source foot-site XY inside verified top rectangle by margin; natural foot overhang allowed; exact capsule minimum Z",
        unmonitored_contact_limit="Nonfoot force gate covers the six named hands/elbows/knees and pelvis sensors; other robot links are not measured by this input")


def task_metrics(rootpos, rootquat, footpos, footquat, contactforce, task_spec):
    """Return batched Torch metrics without CPU synchronization in the sim loop.

    Inputs: E×3, E×4, E×2×3, E×2×4, E×9×3, all on one device.
    Feet are left/right; force columns follow ``CONTACT_BODY_NAMES``. Invalid
    finite/unit-quaternion checks fail the tensor gate rather than normalizing
    malformed states. No force-history or FK-derived force is accepted here.
    """
    import torch
    from morph_mimic.utils.transforms import quat_apply

    count = rootpos.shape[0]
    shapes = ((count, 3), (count, 4), (count, 2, 3), (count, 2, 4), (count, 9, 3))
    values = (rootpos, rootquat, footpos, footquat, contactforce)
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
           or value.device != rootpos.device or value.dtype != rootpos.dtype
           for value, shape in zip(values, shapes)):
        raise ValueError("box metrics need same-device/dtype root E3/E4, feet E2x3/E2x4 and current forces E9x3")
    if task_spec.get("contact_body_names") != list(CONTACT_BODY_NAMES):
        raise ValueError("box task force columns differ from canonical nine-body order")
    constant = lambda value: torch.as_tensor(value, device=rootpos.device, dtype=rootpos.dtype)
    top, geometry, limits = task_spec["target_top"], task_spec["foot_geometry"], task_spec["thresholds"]
    valid = torch.ones(count, dtype=torch.bool, device=rootpos.device)
    for value in values:
        valid &= torch.isfinite(value).reshape(count, -1).all(dim=-1)
    valid &= (rootquat.norm(dim=-1)-1).abs() < 1e-3
    valid &= ((footquat.norm(dim=-1)-1).abs() < 1e-3).all(dim=-1)
    height_error = (rootpos[:, 2]-float(task_spec["final_reference"]["root_pos_rebased_m"][2])).abs()
    up = quat_apply(rootquat, constant([0., 0., 1.]).expand(count, -1))
    tilt = torch.acos(up[:, 2].clamp(-1., 1.))
    axes, half, center = constant(top["axes_xy"]), constant(top["half_extents_m"]), constant(top["center_xy_m"])
    origin_margin = (half-((footpos[..., :2]-center) @ axes.T).abs()).amin(dim=-1)
    sites = constant(geometry["support_center_site_local_m"])
    support_centers = footpos + quat_apply(footquat, sites.expand(count, -1, -1))
    support_margin = (half-((support_centers[..., :2]-center) @ axes.T).abs()).amin(dim=-1)
    endpoints = constant(geometry["capsule_endpoints_local_m"])
    radii = constant(geometry["capsule_radii_m"])
    capsules = endpoints.shape[1]
    world_endpoints = footpos[:, :, None, None, :] + quat_apply(
        footquat[:, :, None, None, :].expand(-1, -1, capsules, 2, -1),
        endpoints.unsqueeze(0).expand(count, -1, -1, -1, -1))
    minimum_sole_z = (world_endpoints[..., 2].amin(dim=-1)-radii).amin(dim=-1)
    sole_error = minimum_sole_z-float(top["top_z_m"])
    foot_fz = contactforce[:, [6, 7], 2].clamp_min(0.)
    nonfoot_fz = contactforce[:, [0, 1, 2, 3, 4, 5, 8], 2].clamp_min(0.).amax(dim=-1)
    gates = dict(valid_state_gate=valid,
        root_height_gate=height_error <= limits["root_height_error_max_m"],
        root_tilt_gate=tilt <= limits["root_tilt_max_rad"],
        foot_support_margin_gate=(support_margin >= limits["foot_support_margin_min_m"]).all(dim=-1),
        foot_sole_plane_gate=((sole_error >= limits["sole_plane_error_min_m"])
                              & (sole_error <= limits["sole_plane_error_max_m"])).all(dim=-1),
        feet_current_support_gate=(foot_fz >= limits["foot_current_positive_fz_min_n"]).all(dim=-1),
        nonfoot_current_support_gate=nonfoot_fz <= limits["nonfoot_current_positive_fz_max_n"])
    geometry_gate = gates["valid_state_gate"] & gates["root_height_gate"] & gates["root_tilt_gate"] \
        & gates["foot_support_margin_gate"] & gates["foot_sole_plane_gate"]
    return dict(rootheight_error_m=height_error, root_tilt_rad=tilt,
        foot_origin_margin_m=origin_margin, foot_support_margin_m=support_margin,
        foot_support_center_pos_rebased_m=support_centers, foot_sole_plane_error_m=sole_error,
        feet_current_positive_fz_n=foot_fz, nonfoot_current_positive_fz_max_n=nonfoot_fz,
        task_geometry_gate=geometry_gate,
        task_gate=geometry_gate & gates["feet_current_support_gate"] & gates["nonfoot_current_support_gate"], **gates)
