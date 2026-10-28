"""Source-bound contact geometry and explicitly limited stability diagnostics.

All positions are metres in one scene frame (source first-root XY removed),
velocities m/s, angles radians, quaternions WXYZ.  Collision candidates are
geometric surface points, NOT measured PhysX contact points or contact forces.
The signed distance to a triangle plane is not a global solid penetration SDF.
COM projection inside a force-supported hull is a quasi-static diagnostic,
not a proof of balance during motion, on walls, or with unknown friction.

The module imports no simulator and uses Torch tensor operations on CPU/GPU.
Spec loading uses the verified reference-scene parser and the source MJCF.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from morph_mimic.utils.reference_scene import load_reference_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _numbers(value, count, description):
    result = np.fromstring(value, sep=" ")
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{description}: expected {count} finite numbers")
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contact_surface_spec(motion_path, robot_xml=None, body_names=None):
    """Load collision primitives, explicit inertial COMs and bounded triangles.

    ``body_names`` determines the exact body-axis ordering, defaults to every
    source body. Unknown/duplicate names and unsupported collision geometry
    fail rather than silently replacing mesh geometry with link origins.
    COM data covers exactly the requested bodies: a whole-body COM requires
    all massive bodies. All source G1 inertials are explicit in this MJCF.
    """
    motion_path = Path(motion_path).resolve()
    robot_xml = Path(robot_xml or PROJECT_ROOT / "assets/g1/robot/g1.xml").resolve()
    scene = load_reference_scene(motion_path.with_suffix(".scene.json"), motion_path)
    source = robot_xml.read_text(encoding="utf-8")
    if "<!DOCTYPE" in source.upper() or "<!ENTITY" in source.upper():
        raise ValueError("MJCF DTD/entity declarations are unsupported")
    tree = ET.fromstring(source)
    if tree.findall("include"):
        raise ValueError("MJCF includes need explicit dependency handling")
    compiler = tree.find("compiler")
    if compiler is not None and compiler.get("angle", "degree") != "radian":
        raise ValueError("contact geometry requires a normalized radian MJCF")
    defaults = {"": {}}

    def default(element, inherited):
        properties = dict(inherited)
        geom = element.find("geom")
        if geom is not None:
            properties.update(geom.attrib)
        defaults[element.get("class", "")] = properties
        for child in element.findall("default"):
            default(child, properties)

    for element in tree.findall("default"):
        default(element, {})
    source_bodies = list(tree.findall(".//worldbody//body"))
    source_names = [body.get("name") for body in source_bodies]
    if None in source_names or len(set(source_names)) != len(source_names):
        raise ValueError("source body names must be explicit and unique")
    body_names = list(source_names if body_names is None else body_names)
    if len(set(body_names)) != len(body_names) or not body_names or set(body_names)-set(source_names):
        raise ValueError("requested body names must be known, nonempty and unique")
    columns = {name: i for i, name in enumerate(body_names)}
    masses, coms = [None]*len(body_names), [None]*len(body_names)
    geometries = []

    def body_visit(body, childclass=""):
        childclass = body.get("childclass", childclass)
        name = body.get("name")
        if name in columns:
            index = columns[name]
            inertial = body.find("inertial")
            if inertial is None:
                raise ValueError(f"{name}: explicit inertial mass/COM required")
            mass = float(inertial.get("mass", "nan"))
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError(f"{name}: inertial mass must be finite and positive")
            masses[index] = mass
            coms[index] = _numbers(inertial.get("pos", "0 0 0"), 3, name+" COM").tolist()
            for geom in body.findall("geom"):
                class_name = geom.get("class", childclass)
                if class_name not in defaults:
                    raise ValueError(f"unknown MJCF geom class {class_name}")
                props = defaults[class_name] | geom.attrib
                if int(props.get("contype", "1")) == 0 and int(props.get("conaffinity", "1")) == 0:
                    continue
                kind = props.get("type", "sphere")
                if kind not in ("sphere", "capsule", "box"):
                    raise ValueError(f"{name}: unsupported collision primitive {kind}")
                # Alternate orientation formats need proper compiler semantics;
                # silently ignoring one would give plausible but wrong points.
                if set(props) & {"axisangle", "euler", "xyaxes", "zaxis"}:
                    raise ValueError(f"{name}: normalized quat/fromto collision orientation required")
                pos = _numbers(props.get("pos", "0 0 0"), 3, name+" geom pos")
                quat = _numbers(props.get("quat", "1 0 0 0"), 4, name+" geom quat")
                if not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
                    raise ValueError(f"{name}: geom quaternion is not unit length")
                size = np.fromstring(props.get("size", ""), sep=" ")
                if not len(size) or not np.isfinite(size).all() or np.any(size <= 0):
                    raise ValueError(f"{name}: positive finite geometry sizes required")
                item = dict(name=geom.get("name", f"{name}_{len(geometries)}"),
                    body_index=index, body_name=name, type=kind, pos_local_m=pos.tolist(),
                    quat_local_wxyz=quat.tolist())
                if kind == "box":
                    if size.shape != (3,) or "fromto" in props:
                        raise ValueError(f"{name}: box requires three half-extents and no fromto")
                    item["half_extents_m"] = size.tolist()
                else:
                    item["radius_m"] = float(size[0])
                    if kind == "sphere":
                        if "fromto" in props:
                            raise ValueError(f"{name}: sphere fromto is unsupported")
                        endpoints = np.stack((pos, pos))
                    elif "fromto" in props:
                        if "pos" in geom.attrib or "quat" in geom.attrib:
                            raise ValueError(f"{name}: capsule fromto cannot be mixed with pos/quat")
                        endpoints = _numbers(props["fromto"], 6, name+" capsule").reshape(2, 3)
                    else:
                        if size.shape != (2,):
                            raise ValueError(f"{name}: capsule needs fromto or radius/half-length")
                        # MJCF capsule's intrinsic longitudinal axis is local Z.
                        q = torch.as_tensor(quat, dtype=torch.float64)
                        ends = torch.tensor([[0., 0., -size[1]], [0., 0., size[1]]], dtype=torch.float64)
                        endpoints = quat_rotate(q.expand(2, 4), ends).numpy()+pos
                    item["segment_endpoints_local_m"] = endpoints.tolist()
                geometries.append(item)
        for child in body.findall("body"):
            body_visit(child, childclass)

    for body in tree.findall("./worldbody/body"):
        body_visit(body)
    triangles, triangle_components = [], []
    components = [(c.name, c.mesh) for c in scene.components]+[("ground", scene.ground.mesh())]
    for name, mesh in components:
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError(f"{name}: closed consistently outward collision mesh required for signed normals")
        data = np.asarray(mesh.triangles, dtype=float).copy()
        data[..., :2] -= scene.source_first_root_xy
        if np.any(np.linalg.norm(np.cross(data[:, 1]-data[:, 0], data[:, 2]-data[:, 0]), axis=-1) < 1e-10):
            raise ValueError(f"{name}: degenerate collision triangle")
        triangles.extend(data.tolist())
        triangle_components.extend([name]*len(data))
    sources = {str(robot_xml): _sha(robot_xml), str(scene.sidecar_path): _sha(scene.sidecar_path),
        str(motion_path): scene.motion_sha256, str(scene.scene_urdf_path): scene.scene_urdf_sha256}
    sources.update(scene.metadata["collision_asset_sha256"])
    return dict(schema=1, positions_frame="scene_rebased_by_source_first_root_xy",
        quaternion_convention="world_from_body_wxyz", source_sha256=sources,
        source_first_root_xy=scene.source_first_root_xy.tolist(), scene_fingerprint=scene.fingerprint,
        body_names=body_names, masses_kg=masses, com_local_m=coms,
        whole_body_mass_covered=set(body_names)==set(source_names), geometries=geometries,
        triangles_m=triangles, triangle_components=triangle_components,
        limitation="Geometry candidates are not measured contacts. Signed triangle distance is local. "
            "COM/hull margin is quasi-static, not a dynamic balance or friction proof.")


def quat_rotate(quaternion, vector):
    """Rotate broadcastable [...,4] WXYZ and [...,3] vectors (unit quats)."""
    xyz, vector = torch.broadcast_tensors(quaternion[..., 1:], vector)
    twice = 2*torch.linalg.cross(xyz, vector)
    return vector+quaternion[..., :1]*twice+torch.linalg.cross(xyz, twice)


def rigid_point_velocity(linear_velocity_w, angular_velocity_w, point_offset_w):
    """v(origin)+omega cross (point-origin), [...,3], velocities m/s and rad/s.

    If linear velocity is the articulation's COM velocity, offset MUST be
    point minus the COM, not point minus the body's link-frame origin.
    """
    linear, angular, offset = torch.broadcast_tensors(linear_velocity_w, angular_velocity_w, point_offset_w)
    if linear.shape[-1] != 3:
        raise ValueError("rigid point velocities require final dimension 3")
    return linear+torch.linalg.cross(angular, offset)


def _unit(vector):
    return vector/torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-12)


def contact_point_metrics(point_pos_w, point_vel_w, surface_point_w, surface_normal_w,
                          actual_quat_wxyz=None, reference_quat_wxyz=None, valid_mask=None):
    """Metric dictionary with [...]-shaped valid mask, no inferred contact state.

    Surface normals must be finite unit normals. Non-contact/swing samples can
    be marked invalid by the caller; this helper never rewards proximity or
    asserts a contact because geometry happened to be close. Invalid results
    are zero plus an explicit false mask, and must not enter loss denominators.
    """
    pos, velocity, surface, normal = torch.broadcast_tensors(point_pos_w, point_vel_w,
        surface_point_w, surface_normal_w)
    if pos.shape[-1] != 3:
        raise ValueError("contact vectors require final dimension 3")
    valid = torch.isfinite(torch.stack((pos, velocity, surface, normal))).all(dim=(0, -1))
    norm = torch.linalg.vector_norm(normal, dim=-1)
    valid &= (norm-1).abs() <= 1e-4
    if valid_mask is not None:
        valid &= torch.broadcast_to(valid_mask.bool(), valid.shape)
    normal = _unit(torch.nan_to_num(normal))
    normal_distance = ((pos-surface)*normal).sum(-1)
    normal_speed = (velocity*normal).sum(-1)
    tangent = velocity-normal_speed[..., None]*normal
    out = dict(valid=valid, normal_distance_m=normal_distance,
        normal_speed_m_s=normal_speed, tangent_velocity_m_s=tangent,
        tangent_speed_m_s=torch.linalg.vector_norm(tangent, dim=-1))
    if (actual_quat_wxyz is None) != (reference_quat_wxyz is None):
        raise ValueError("orientation metric needs both actual and reference quaternion")
    if actual_quat_wxyz is not None:
        actual, reference = torch.broadcast_tensors(actual_quat_wxyz, reference_quat_wxyz)
        if actual.shape != (*valid.shape, 4):
            raise ValueError("orientation quaternions must match contact sample axes")
        valid &= torch.isfinite(actual).all(-1) & torch.isfinite(reference).all(-1)
        valid &= (torch.linalg.vector_norm(actual, dim=-1)-1).abs() <= 1e-4
        valid &= (torch.linalg.vector_norm(reference, dim=-1)-1).abs() <= 1e-4
        # Antipodal-safe geodesic distance in SO(3), range [0, pi].
        dot = (actual*reference).sum(-1).abs().clamp(0, 1)
        out["orientation_error_rad"] = 2*torch.atan2((1-dot.square()).clamp_min(0).sqrt(), dot)
    out["valid"] = valid
    for key in list(out):
        if key != "valid":
            value = out[key]
            mask = valid if value.ndim == valid.ndim else valid[..., None]
            out[key] = torch.where(mask, value, torch.zeros_like(value))
    return out


def _closest_triangle(points, triangles):
    """Broadcasted closest point on nondegenerate bounded triangles."""
    a, b, c = triangles.unbind(-2)
    ab, ac, ap = b-a, c-a, points-a
    d00, d01, d11 = (ab*ab).sum(-1), (ab*ac).sum(-1), (ac*ac).sum(-1)
    d20, d21 = (ap*ab).sum(-1), (ap*ac).sum(-1)
    denominator = (d00*d11-d01.square()).clamp_min(1e-20)
    v = (d11*d20-d01*d21)/denominator
    w = (d00*d21-d01*d20)/denominator
    inside = (v >= 0) & (w >= 0) & (v+w <= 1)
    projection = a+v[..., None]*ab+w[..., None]*ac
    edge_points = []
    for start, stop in ((a, b), (b, c), (c, a)):
        edge = stop-start
        ratio = (((points-start)*edge).sum(-1)/(edge*edge).sum(-1).clamp_min(1e-20)).clamp(0, 1)
        edge_points.append(start+ratio[..., None]*edge)
    candidates = torch.stack(edge_points, dim=-2)
    distances = ((candidates-points[..., None, :])**2).sum(-1)
    edge = candidates.gather(-2, distances.argmin(-1)[..., None, None].expand(*distances.shape[:-1], 1, 3)).squeeze(-2)
    return torch.where(inside[..., None], projection, edge)


def closest_scene_surface(points_w, spec):
    """Nearest point on verified bounded scene triangles for points [...,3].

    Returns point_w, outward normal_w, Euclidean distance_m, local signed
    normal_distance_m (alias signed_distance_m), triangle_index and valid.
    Closest on a box wall uses its wall normal, never global Z. A signed plane
    distance near a triangle edge is not the same as Euclidean clearance.
    """
    if points_w.shape[-1] != 3:
        raise ValueError("scene queries require [...,3] points")
    triangles = torch.as_tensor(spec["triangles_m"], dtype=points_w.dtype, device=points_w.device)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("scene spec needs nonempty [triangles,3,3] geometry")
    points = points_w[..., None, :]
    closest = _closest_triangle(points, triangles)
    distance2 = ((points-closest)**2).sum(-1)
    index = distance2.argmin(-1)
    selected = closest.gather(-2, index[..., None, None].expand(*index.shape, 1, 3)).squeeze(-2)
    normal = _unit(torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]))[index]
    valid = torch.isfinite(points_w).all(-1) & torch.isfinite(selected).all(-1)
    signed = ((points_w-selected)*normal).sum(-1)
    return dict(point_w=selected, normal_w=normal, distance_m=(points_w-selected).norm(dim=-1),
        signed_distance_m=signed, normal_distance_m=signed, triangle_index=index, valid=valid)


def _closest_segments(a, b, c, d):
    """Closest points between two finite segments, including point degeneracy."""
    u, v, w = b-a, d-c, a-c
    aa, bb, cc = (u*u).sum(-1), (u*v).sum(-1), (v*v).sum(-1)
    dd, ee = (u*w).sum(-1), (v*w).sum(-1)
    denom = aa*cc-bb.square()
    s = torch.where(denom > 1e-20, (bb*ee-cc*dd)/denom.clamp_min(1e-20), torch.zeros_like(denom)).clamp(0, 1)
    t = (bb*s+ee)/cc.clamp_min(1e-20)
    s = torch.where(t < 0, (-dd/aa.clamp_min(1e-20)).clamp(0, 1), s)
    s = torch.where(t > 1, ((bb-dd)/aa.clamp_min(1e-20)).clamp(0, 1), s)
    t = t.clamp(0, 1)
    # A degenerate first segment is a point; second degeneracy likewise.
    s = torch.where(aa <= 1e-20, torch.zeros_like(s), s)
    t = torch.where(aa <= 1e-20, (ee/cc.clamp_min(1e-20)).clamp(0, 1), t)
    t = torch.where(cc <= 1e-20, torch.zeros_like(t), t)
    s = torch.where(cc <= 1e-20, (-dd/aa.clamp_min(1e-20)).clamp(0, 1), s)
    return a+s[..., None]*u, c+t[..., None]*v


def _closest_segment_triangle(start, stop, triangles):
    """Exact closest pair for a segment and nondegenerate bounded triangle."""
    a, b, c = triangles.unbind(-2)
    normal = _unit(torch.linalg.cross(b-a, c-a))
    start, stop, normal = torch.broadcast_tensors(start, stop, normal)
    a = torch.broadcast_to(a, start.shape)
    b = torch.broadcast_to(b, start.shape)
    c = torch.broadcast_to(c, start.shape)
    endpoint_scene = (_closest_triangle(start, triangles), _closest_triangle(stop, triangles))
    segment_points = [start, stop]
    scene_points = list(endpoint_scene)
    for edge_start, edge_stop in ((a, b), (b, c), (c, a)):
        x, y = _closest_segments(start, stop, edge_start, edge_stop)
        segment_points.append(x)
        scene_points.append(y)
    # A segment crossing a triangle interior needs an explicit plane hit;
    # endpoint/edge pairs alone miss this zero-distance case.
    direction = stop-start
    denominator = (direction*normal).sum(-1)
    ratio = ((a-start)*normal).sum(-1)/torch.where(denominator.abs() > 1e-12, denominator, torch.ones_like(denominator))
    hit = start+ratio[..., None]*direction
    triangle_hit = _closest_triangle(hit, triangles)
    intersects = (denominator.abs() > 1e-12) & (ratio >= 0) & (ratio <= 1)
    intersects &= ((hit-triangle_hit)**2).sum(-1) <= 1e-14
    xs, ys = torch.stack(segment_points, -2), torch.stack(scene_points, -2)
    distance = ((xs-ys)**2).sum(-1)
    index = distance.argmin(-1)
    gather = index[..., None, None].expand(*index.shape, 1, 3)
    x, y = xs.gather(-2, gather).squeeze(-2), ys.gather(-2, gather).squeeze(-2)
    return torch.where(intersects[..., None], hit, x), torch.where(intersects[..., None], hit, y)


def nearest_collision_surface(body_pos_w, body_quat_wxyz, spec):
    """Nearest scene candidate for each sphere/capsule collider [...,G,3].

    Sphere/capsule centerline-to-bounded-triangle closest pairs are exact;
    signed gap subtracts the primitive radius from the signed pair distance.
    This is a local oriented-surface gap, not a union-of-solids penetration
    depth. ``point_w`` is an actual primitive surface point, ``surface_point_w``
    lies on a bounded scene triangle, and ``normal_w`` is its outward normal.
    Candidates do NOT assert contact or reconstruct a physical force point.
    For a box use ``collision_support_points`` with the known surface normal;
    this query rejects boxes rather than approximating their edge geometry.
    """
    if body_pos_w.shape[-2:] != (len(spec["body_names"]), 3) or body_quat_wxyz.shape != (*body_pos_w.shape[:-1], 4):
        raise ValueError("body poses must follow spec body_names with shapes [...,B,3/4]")
    triangles = torch.as_tensor(spec["triangles_m"], dtype=body_pos_w.dtype, device=body_pos_w.device)
    normals = _unit(torch.linalg.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]))
    outputs = {key: [] for key in ("point_w", "surface_point_w", "normal_w", "signed_clearance_m", "triangle_index", "valid")}
    for geom in spec["geometries"]:
        if geom["type"] not in ("sphere", "capsule"):
            raise ValueError("nearest_collision_surface supports exact sphere/capsule queries; boxes need support points")
        bi = geom["body_index"]
        local = torch.as_tensor(geom["segment_endpoints_local_m"], dtype=body_pos_w.dtype, device=body_pos_w.device)
        world = body_pos_w[..., bi, None, :]+quat_rotate(body_quat_wxyz[..., bi, None, :], local)
        x, y = _closest_segment_triangle(world[..., 0, None, :], world[..., 1, None, :], triangles)
        delta = x-y
        sign = torch.where((delta*normals).sum(-1) >= 0, torch.ones_like(delta[..., 0]), -torch.ones_like(delta[..., 0]))
        center_distance = delta.norm(dim=-1)
        gap = center_distance*sign-geom["radius_m"]
        index = gap.abs().argmin(-1)
        gather = index[..., None, None].expand(*index.shape, 1, 3)
        x, y = x.gather(-2, gather).squeeze(-2), y.gather(-2, gather).squeeze(-2)
        n = normals[index]
        selected_sign = sign.gather(-1, index[..., None]).squeeze(-1)
        d = x-y
        # Outside: toward scene; inside: away from scene into the solid. The
        # latter measures oriented penetration without flipping the radius.
        radial_direction = torch.where(d.norm(dim=-1, keepdim=True) > 1e-10,
            _unit(d)*selected_sign[..., None], n)
        collision_point = x-geom["radius_m"]*radial_direction
        outputs["point_w"].append(collision_point)
        outputs["surface_point_w"].append(y)
        outputs["normal_w"].append(n)
        outputs["signed_clearance_m"].append(gap.gather(-1, index[..., None]).squeeze(-1))
        outputs["triangle_index"].append(index)
        outputs["valid"].append(torch.isfinite(collision_point).all(-1) & torch.isfinite(y).all(-1))
    if not outputs["valid"]:
        raise ValueError("no collision primitives to query")
    result = {key: torch.stack(values, dim=-2 if key.endswith("_w") else -1) for key, values in outputs.items()}
    result["body_index"] = torch.tensor([g["body_index"] for g in spec["geometries"]], dtype=torch.long, device=body_pos_w.device)
    return result


def collision_support_points(body_pos_w, body_quat_wxyz, normal_w, spec):
    """Actual primitive surface points minimizing dot(point, outward normal).

    Body axes [...,B,3/4] follow spec body_names. ``normal_w`` is [...,3]
    (one common surface normal per sample). Returned points [...,P,3] have
    geom_index[P] and body_index[P]. Capsules yield BOTH endpoint support
    points, spheres one point, boxes all eight vertices tied for minimum.
    ``valid`` masks box vertices which are not on the support face. Thus a
    horizontal foot capsule's finite support patch is retained for hull tests.
    These are geometry candidates, never force-bearing contact assertions.
    """
    if body_pos_w.shape[-2:] != (len(spec["body_names"]), 3) or body_quat_wxyz.shape != (*body_pos_w.shape[:-1], 4):
        raise ValueError("body poses must follow spec body_names with shapes [...,B,3/4]")
    normal = _unit(normal_w)
    points, body_indices, geom_indices, validity = [], [], [], []
    for gi, geom in enumerate(spec["geometries"]):
        bi = geom["body_index"]
        quat, position = body_quat_wxyz[..., bi, :], body_pos_w[..., bi, :]
        if geom["type"] in ("sphere", "capsule"):
            endpoints = torch.as_tensor(geom["segment_endpoints_local_m"], dtype=position.dtype, device=position.device)
            if geom["type"] == "sphere":
                endpoints = endpoints[:1]
            world = position[..., None, :]+quat_rotate(quat[..., None, :], endpoints)
            support = world-geom["radius_m"]*normal[..., None, :]
            # For nonparallel capsules only the lower cap is a plane support;
            # the other cap remains a geometric surface point but is invalid.
            dots = (support*normal[..., None, :]).sum(-1)
            valid = dots <= dots.amin(-1, keepdim=True)+1e-6
        else:
            signs = torch.tensor([[-1., -1., -1.], [-1., -1., 1.], [-1., 1., -1.], [-1., 1., 1.],
                [1., -1., -1.], [1., -1., 1.], [1., 1., -1.], [1., 1., 1.]], dtype=position.dtype, device=position.device)
            half = torch.as_tensor(geom["half_extents_m"], dtype=position.dtype, device=position.device)
            localq = torch.as_tensor(geom["quat_local_wxyz"], dtype=position.dtype, device=position.device)
            localp = torch.as_tensor(geom["pos_local_m"], dtype=position.dtype, device=position.device)
            vertices = localp+quat_rotate(localq, signs*half)
            support = position[..., None, :]+quat_rotate(quat[..., None, :], vertices)
            dots = (support*normal[..., None, :]).sum(-1)
            valid = dots <= dots.amin(-1, keepdim=True)+1e-6
        points.append(support)
        validity.append(valid)
        body_indices.extend([bi]*support.shape[-2])
        geom_indices.extend([gi]*support.shape[-2])
    if not points:
        raise ValueError("no supported collision geometries in spec")
    point = torch.cat(points, dim=-2)
    valid = torch.cat(validity, dim=-1) & torch.isfinite(point).all(-1)
    normal_valid = torch.isfinite(normal_w).all(-1) & (normal_w.norm(dim=-1)-1).abs().le(1e-4)
    valid &= normal_valid[..., None]
    return dict(point_w=point, valid=valid,
        body_index=torch.tensor(body_indices, dtype=torch.long, device=point.device),
        geom_index=torch.tensor(geom_indices, dtype=torch.long, device=point.device))


def mass_weighted_com(body_pos_w, body_quat_wxyz, masses_kg, com_local_m):
    """Mass-weighted COM [...,3] from body origins and source inertial offsets.

    Inputs [B] masses and [B,3] local COM offsets must cover every body to call
    the result whole-body COM. A runtime backend's actual masses/COM offsets
    should replace source inertials when dynamics randomization is active.
    """
    masses = torch.as_tensor(masses_kg, dtype=body_pos_w.dtype, device=body_pos_w.device)
    local = torch.as_tensor(com_local_m, dtype=body_pos_w.dtype, device=body_pos_w.device)
    if (masses.ndim != 1 or local.shape != (len(masses), 3)
            or body_pos_w.shape[-2:] != (len(masses), 3)
            or body_quat_wxyz.shape != (*body_pos_w.shape[:-1], 4)):
        raise ValueError("COM input axes require body poses [...,B,3/4], masses [B], offsets [B,3]")
    if not torch.isfinite(masses).all() or not torch.all(masses > 0) or not torch.isfinite(local).all():
        raise ValueError("COM requires finite positive masses and finite offsets")
    centers = body_pos_w+quat_rotate(body_quat_wxyz, local)
    return (centers*masses[:, None]).sum(-2)/masses.sum()


def support_hull_margin(com_w, support_points_w, active_mask, plane_normal_w=None, *, coplanar_tolerance_m=.01):
    """Signed COM projection margin to a force-supported planar convex hull.

    com [...,3], points [...,P,3], mask [...,P]; positive means inside a 2D
    hull, negative outside. Exact edge distances, no axis-aligned box shortcut.
    Caller supplies a physically force-bearing mask; geometric nearness is
    insufficient. Coplanarity is required. Rank: -1 no support, 0 point,
    1 segment, 2 area. ``valid`` is true only for finite coplanar rank-2 hulls.
    For single-point/line support margin is minus distance to the point/segment
    and valid=false; no fabricated positive support area for single contact.
    This is a quasi-static projection diagnostic, not a dynamic stability gate.
    """
    if support_points_w.shape[-1] == 2:
        if plane_normal_w is not None:
            raise ValueError("2D support points already specify their plane; omit plane_normal_w")
        support_points_w = torch.cat((support_points_w, torch.zeros_like(support_points_w[..., :1])), -1)
        if com_w.shape[-1] != 2:
            raise ValueError("2D support points require 2D COM projection")
        com_w = torch.cat((com_w, torch.zeros_like(com_w[..., :1])), -1)
    if (support_points_w.ndim < 2 or support_points_w.shape[-1] != 3
            or com_w.shape != (*support_points_w.shape[:-2], 3)
            or active_mask.shape != support_points_w.shape[:-1] or support_points_w.shape[-2] < 1):
        raise ValueError("support inputs require [...,3], [...,P,3], [...,P] with P>=1")
    points = support_points_w
    if plane_normal_w is None:
        normal = torch.zeros_like(com_w)
        normal[..., 2] = 1.
    else:
        normal = torch.broadcast_to(plane_normal_w, com_w.shape)
    normal_valid = torch.isfinite(normal).all(-1) & (normal.norm(dim=-1)-1).abs().le(1e-4)
    normal = _unit(torch.nan_to_num(normal))
    # Stable plane tangent basis, including a horizontal wall normal.
    axis = torch.zeros_like(normal)
    use_x = normal[..., 0].abs() < .9
    axis[..., 0] = use_x.to(axis.dtype)
    axis[..., 1] = (~use_x).to(axis.dtype)
    u = _unit(torch.linalg.cross(normal, axis))
    v = torch.linalg.cross(normal, u)
    xy = torch.stack(((points*u[..., None, :]).sum(-1), (points*v[..., None, :]).sum(-1)), -1)
    center = torch.stack(((com_w*u).sum(-1), (com_w*v).sum(-1)), -1)
    active = active_mask.bool() & torch.isfinite(points).all(-1)
    count = active.sum(-1)
    depths = (points*normal[..., None, :]).sum(-1)
    depth_min = torch.where(active, depths, torch.inf).amin(-1)
    depth_max = torch.where(active, depths, -torch.inf).amax(-1)
    coplanar = (depth_max-depth_min) <= coplanar_tolerance_m
    # Batched monotonic chain uses O(batch*P) memory, avoiding a P^3 pair/point
    # tensor for large contact buffers. No host copy or scipy hull is needed.
    batch_shape, slots = xy.shape[:-2], xy.shape[-2]
    flat, flat_active = xy.reshape(-1, slots, 2), active.reshape(-1, slots)
    row = torch.arange(len(flat), device=xy.device)
    sortable = torch.where(flat_active[..., None], flat, torch.full_like(flat, torch.inf))
    order_y = sortable[..., 1].argsort(dim=-1, stable=True)
    sorted_y = sortable.gather(-2, order_y[..., None].expand(-1, -1, 2))
    order = order_y.gather(-1, sorted_y[..., 0].argsort(dim=-1, stable=True))
    sorted_points = flat.gather(-2, order[..., None].expand(-1, -1, 2))
    sorted_active = flat_active.gather(-1, order)
    unique = sorted_active.clone()
    unique[:, 1:] &= (sorted_points[:, 1:]-sorted_points[:, :-1]).norm(dim=-1) > 1e-7
    compact_order = (~unique).to(torch.int8).argsort(dim=-1, stable=True)
    sorted_points = sorted_points.gather(-2, compact_order[..., None].expand(-1, -1, 2))
    unique_count = unique.sum(-1)

    def chain(reverse):
        stack = torch.zeros_like(sorted_points)
        size = torch.zeros_like(unique_count)
        for i in range(slots):
            running = i < unique_count
            index = (unique_count-1-i).clamp_min(0) if reverse else torch.full_like(unique_count, i)
            point = sorted_points[row, index]
            for _ in range(slots):
                a = stack[row, (size-2).clamp_min(0)]
                b = stack[row, (size-1).clamp_min(0)]
                ab, bc = b-a, point-b
                cross = ab[:, 0]*bc[:, 1]-ab[:, 1]*bc[:, 0]
                pop = running & (size >= 2) & (cross <= 1e-10)
                if not bool(pop.any()):
                    break
                size = size-pop.to(size.dtype)
            stack[row, size.clamp_max(slots-1)] = torch.where(running[:, None], point, stack[row, size.clamp_max(slots-1)])
            size = size+running.to(size.dtype)
        return stack, size

    lower, lower_size = chain(False)
    upper, upper_size = chain(True)
    hull_count = torch.where(unique_count > 1, lower_size+upper_size-2, unique_count)
    area = (hull_count >= 3).reshape(batch_shape)
    flat_center = center.reshape(-1, 2)
    edge_margins = []
    for stack, size in ((lower, lower_size), (upper, upper_size)):
        delta = stack[:, 1:]-stack[:, :-1]
        offset = flat_center[:, None, :]-stack[:, :-1]
        distance = (delta[..., 0]*offset[..., 1]-delta[..., 1]*offset[..., 0])/delta.norm(dim=-1).clamp_min(1e-12)
        exists = torch.arange(slots-1, device=xy.device)[None, :] < (size-1)[:, None]
        edge_margins.append(torch.where(exists, distance, torch.inf).amin(-1) if slots > 1 else torch.full_like(flat_center[:, 0], torch.inf))
    margin = torch.minimum(*edge_margins).reshape(batch_shape)
    first = lower[:, 0]
    last = lower[row, (lower_size-1).clamp_min(0)]
    direction = last-first
    fraction = (((flat_center-first)*direction).sum(-1)/direction.square().sum(-1).clamp_min(1e-20)).clamp(0, 1)
    degenerate = (flat_center-(first+fraction[:, None]*direction)).norm(dim=-1).reshape(batch_shape)
    rank = torch.where(count > 0, torch.zeros_like(count), -torch.ones_like(count))
    rank = torch.where(unique_count.reshape(batch_shape) > 1, torch.ones_like(rank), rank)
    rank = torch.where(area, torch.full_like(rank, 2), rank)
    margin = torch.where(area, margin, -degenerate)
    finite_input = torch.isfinite(com_w).all(-1) & ((~active_mask.bool()) | torch.isfinite(points).all(-1)).all(-1)
    valid = area & coplanar & normal_valid & finite_input
    margin = torch.where(count > 0, margin, torch.full_like(margin, torch.nan))
    return dict(margin_m=margin, valid=valid, support_rank=rank, active_point_count=count,
        hull_point_count=hull_count.reshape(batch_shape),
        coplanar=coplanar & (count > 0), plane_normal_valid=normal_valid)
