"""Load a motion's hashed, static collision scene without starting a simulator.

Schema 1 sidecars bind one derived NPZ to an original fixed URDF and its mesh
dependencies. Geometry is retained in source world coordinates. Consumers that
rebase a reference by its first root XY must apply the same translation to the
scene, using ``rebased_mesh``. No source code or pickle data is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import operator
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def reference_scene_grid_shape(num_envs: int) -> tuple[int, int]:
    """Return a compact deterministic grid with one scene tile per environment.

    Grid coordinates describe copies of the same scene, not terrain difficulty
    or motion categories. Extra cells in the final row remain unassigned.
    """
    try:
        count = operator.index(num_envs)
    except TypeError as exc:
        raise ValueError("num_envs must be a positive integer") from exc
    if isinstance(num_envs, bool) or count < 1:
        raise ValueError("num_envs must be a positive integer")
    rows = math.isqrt(count)
    return rows, (count + rows - 1) // rows


@lru_cache(maxsize=8)
def _cached_rebased_scene_mesh(sidecar_path: str, fingerprint: str) -> trimesh.Trimesh:
    """Load and verify a content-keyed scene once for a terrain generation run."""
    scene = load_reference_scene(sidecar_path)
    if scene.fingerprint != fingerprint:
        raise ValueError("reference scene fingerprint differs from the motion's verified scene")
    return scene.rebased_mesh()


def reference_scene_mesh_copy(sidecar_path, fingerprint: str = "") -> trimesh.Trimesh:
    """Return an independently mutable scene copy for one generated tile.

    A motion dataset has already checked every bound source hash. Its verified
    fingerprint permits copies to reuse the first fully checked geometry load,
    instead of hashing and parsing the same source files hundreds of times.
    Direct terrain callers without that fingerprint are always revalidated.
    """
    if not fingerprint:
        return load_reference_scene(sidecar_path).rebased_mesh()
    return _cached_rebased_scene_mesh(str(Path(sidecar_path).resolve()), fingerprint).copy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(value, count: int, name: str) -> np.ndarray:
    if isinstance(value, str):
        value = value.split()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected {count} finite numbers") from exc
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected {count} finite numbers")
    return array


def _local_path(value, directory: Path, name: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{name}: expected a local file path")
    path = Path(value)
    if not path.is_absolute():
        path = directory / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{name}: file does not exist: {path}")
    return path


def _checked_file(value, directory: Path, name: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name}: requires path and sha256")
    path = _local_path(value["path"], directory, name)
    expected = value["sha256"]
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"{name}: invalid SHA256")
    if _sha256(path) != expected:
        raise ValueError(f"{name}: SHA256 mismatch: {path}")
    return path, expected


def _origin(element, name: str) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    if set(element.attrib) - {"xyz", "rpy"}:
        raise ValueError(f"{name}: unknown origin attributes")
    transform[:3, 3] = _vector(element.get("xyz", "0 0 0"), 3, name + " xyz")
    roll, pitch, yaw = _vector(element.get("rpy", "0 0 0"), 3, name + " rpy")
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    # URDF extrinsic fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll).
    transform[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return transform


def _world_transform(position, quaternion) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = _vector(position, 3, "world_position")
    w, x, y, z = _vector(quaternion, 4, "world_quat_wxyz")
    if not np.isclose(w*w + x*x + y*y + z*z, 1.0, rtol=0, atol=1e-6):
        raise ValueError("world_quat_wxyz: expected a unit WXYZ quaternion")
    transform[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ]
    return transform


@dataclass(frozen=True)
class GroundGeometry:
    """Explicit source-world ground slab; top surface is ``top_z_m``."""

    top_z_m: float
    size_m: tuple[float, float]
    thickness_m: float
    center_xy_m: tuple[float, float]

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.creation.box(extents=(*self.size_m, self.thickness_m)).apply_translation(
            (*self.center_xy_m, self.top_z_m - self.thickness_m / 2.0)
        )


@dataclass(frozen=True)
class CollisionGeometry:
    """One collision primitive in source world coordinates, excluding ground.

    ``mesh`` already includes scale and every transform. ``transform`` maps
    the scaled mesh/primitive's local frame into the source world. Keeping
    components separate prevents a physics backend from filling the gaps
    between obstacles with a single convex hull.
    """

    name: str
    link_name: str
    kind: str
    mesh: trimesh.Trimesh
    transform: np.ndarray
    size_m: tuple[float, float, float] | None = None
    source_path: Path | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class ReferenceScene:
    sidecar_path: Path
    motion_path: Path
    motion_sha256: str
    scene_urdf_path: Path
    scene_urdf_sha256: str
    source_first_root_xy: np.ndarray
    components: tuple[CollisionGeometry, ...]
    ground: GroundGeometry
    fingerprint: str
    metadata: dict

    def combined_mesh(self, rebased: bool = False) -> trimesh.Trimesh:
        mesh = trimesh.util.concatenate([component.mesh for component in self.components] + [self.ground.mesh()])
        if rebased:
            mesh.apply_translation((-self.source_first_root_xy[0], -self.source_first_root_xy[1], 0.0))
        return mesh

    def source_mesh(self) -> trimesh.Trimesh:
        return self.combined_mesh()

    def rebased_mesh(self) -> trimesh.Trimesh:
        return self.combined_mesh(rebased=True)

    @property
    def bounds(self) -> np.ndarray:
        return self.combined_mesh().bounds

    @property
    def obstacle_bounds(self) -> np.ndarray:
        return trimesh.util.concatenate([component.mesh for component in self.components]).bounds


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"scene JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _parse_ground(value) -> GroundGeometry:
    required = {"top_z_m", "size_m", "thickness_m", "center_xy_m"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("ground: requires top_z_m, size_m, thickness_m, center_xy_m")
    try:
        top, thickness = float(value["top_z_m"]), float(value["thickness_m"])
    except (ValueError, TypeError) as exc:
        raise ValueError("ground: invalid height or thickness") from exc
    size = _vector(value["size_m"], 2, "ground size_m")
    center = _vector(value["center_xy_m"], 2, "ground center_xy_m")
    if not np.isfinite([top, thickness]).all() or thickness <= 0 or (size <= 0).any():
        raise ValueError("ground: finite height and positive size/thickness required")
    return GroundGeometry(top, tuple(size), thickness, tuple(center))


def _read_collision_geometry(urdf_path: Path, dependencies: dict[Path, str], world: np.ndarray):
    source = urdf_path.read_text(encoding="utf-8")
    if "<!DOCTYPE" in source.upper() or "<!ENTITY" in source.upper():
        raise ValueError("scene URDF: DTD/entity declarations are unsupported")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise ValueError(f"scene URDF: invalid XML: {exc}") from exc
    if root.tag != "robot":
        raise ValueError("scene URDF: expected robot root")
    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name or name in links:
            raise ValueError("scene URDF: link names must be unique and nonempty")
        links[name] = link
    if not links:
        raise ValueError("scene URDF: no links")
    parents, joint_names = {}, set()
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name or name in joint_names or joint.get("type") != "fixed":
            raise ValueError("scene URDF: requires uniquely named fixed joints only")
        joint_names.add(name)
        parent, child = joint.find("parent"), joint.find("child")
        parent_name = None if parent is None else parent.get("link")
        child_name = None if child is None else child.get("link")
        if parent_name not in links and parent_name != "world":
            raise ValueError(f"scene URDF: missing parent link {parent_name}")
        if child_name not in links or child_name in parents or child_name == "world":
            raise ValueError(f"scene URDF: invalid or multiply parented child {child_name}")
        parents[child_name] = (parent_name, _origin(joint.find("origin"), f"joint {name}"))
    unparented = set(links) - set(parents) - {"world"}
    if len(unparented) > 1:
        raise ValueError("scene URDF: disconnected unparented link roots")
    transforms, visiting = {"world": np.eye(4)}, set()

    def link_transform(name):
        if name in transforms:
            return transforms[name]
        if name in visiting:
            raise ValueError("scene URDF: fixed-joint cycle")
        visiting.add(name)
        transform = np.eye(4) if name not in parents else link_transform(parents[name][0]) @ parents[name][1]
        visiting.remove(name)
        transforms[name] = transform
        return transform

    components, used = [], set()
    for link_name, link in links.items():
        link_world = world @ link_transform(link_name)
        for index, collision in enumerate(link.findall("collision")):
            name = collision.get("name", f"{link_name}_collision_{index}")
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"collision {name}: requires exactly one geometry")
            primitive = geometry[0]
            transform = link_world @ _origin(collision.find("origin"), f"collision {name}")
            size, path, digest = None, None, None
            if primitive.tag == "mesh":
                if set(primitive.attrib) - {"filename", "scale"}:
                    raise ValueError(f"collision {name}: unknown mesh attributes")
                path = _local_path(primitive.get("filename"), urdf_path.parent, f"collision {name}")
                if path not in dependencies:
                    raise ValueError(f"collision {name}: mesh missing from collision_asset_sha256: {path}")
                digest = dependencies[path]
                used.add(path)
                scale = _vector(primitive.get("scale", "1 1 1"), 3, f"collision {name} scale")
                if (scale <= 0).any():
                    raise ValueError(f"collision {name}: positive mesh scale required")
                mesh = trimesh.load_mesh(path, process=False, skip_materials=True)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise ValueError(f"collision {name}: mesh file must contain one triangle mesh")
                mesh.apply_scale(scale)
            elif primitive.tag == "box":
                if set(primitive.attrib) != {"size"}:
                    raise ValueError(f"collision {name}: box requires size only")
                box_size = _vector(primitive.get("size"), 3, f"collision {name} size")
                if (box_size <= 0).any():
                    raise ValueError(f"collision {name}: positive box size required")
                size = tuple(box_size)
                mesh = trimesh.creation.box(extents=box_size)
            else:
                raise ValueError(f"collision {name}: unsupported geometry {primitive.tag}")
            if not len(mesh.faces) or not len(mesh.vertices) or not np.isfinite(mesh.vertices).all():
                raise ValueError(f"collision {name}: empty or nonfinite triangle mesh")
            mesh.apply_transform(transform)
            components.append(CollisionGeometry(name, link_name, primitive.tag, mesh, transform.copy(), size, path, digest))
    if not components:
        raise ValueError("scene URDF: no collision geometry")
    if used != set(dependencies):
        raise ValueError("collision_asset_sha256 includes files not used by collision geometry")
    return tuple(components)


def load_reference_scene(sidecar_path: str | Path, motion_path: str | Path | None = None) -> ReferenceScene:
    """Validate a schema 1 binding and all files, then parse static geometry.

    All relative paths are relative to the sidecar. The motion must expose
    ``root_pos``; this loader deliberately does not guess its base-body index.
    Geometry hashes are checked before parsing. The fingerprint is portable
    across directory moves and binds motion, source geometry, ground and rebase.
    """
    sidecar = Path(sidecar_path).resolve()
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value {value}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read scene sidecar {sidecar}: {exc}") from exc
    required = {"schema", "robot", "motion_file", "motion_sha256", "source_motion", "scene_urdf",
                "source_first_root_xy", "ground", "collision_asset_sha256"}
    if not isinstance(metadata, dict) or required - metadata.keys():
        raise ValueError("scene sidecar: missing schema 1 binding fields")
    if type(metadata["schema"]) is not int or metadata["schema"] != 1:
        raise ValueError("scene sidecar: unsupported schema")
    if not isinstance(metadata["robot"], str) or not metadata["robot"]:
        raise ValueError("scene sidecar: robot must be explicit")
    bound_motion, motion_digest = _checked_file(
        {"path": metadata["motion_file"], "sha256": metadata["motion_sha256"]}, sidecar.parent, "motion")
    if motion_path is not None and Path(motion_path).resolve() != bound_motion:
        raise ValueError("scene sidecar: requested motion path differs from motion_file")
    _checked_file(metadata["source_motion"], sidecar.parent, "source_motion")
    urdf_path, urdf_digest = _checked_file(metadata["scene_urdf"], sidecar.parent, "scene_urdf")
    first_xy = _vector(metadata["source_first_root_xy"], 2, "source_first_root_xy")
    with np.load(bound_motion, allow_pickle=False) as motion:
        if "root_pos" not in motion:
            raise ValueError("scene motion: explicit root_pos required")
        root_pos = motion["root_pos"]
        if root_pos.ndim != 2 or root_pos.shape[1] != 3 or len(root_pos) < 2 or not np.isfinite(root_pos).all():
            raise ValueError("scene motion: root_pos must be finite (T, 3), T >= 2")
        if not np.allclose(root_pos[0, :2], first_xy, rtol=0, atol=1e-6):
            raise ValueError("scene motion: source_first_root_xy differs from bound motion")
    ground = _parse_ground(metadata["ground"])
    assets = metadata["collision_asset_sha256"]
    if not isinstance(assets, dict):
        raise ValueError("collision_asset_sha256: expected path-to-SHA256 mapping")
    dependencies = {}
    for source_path, expected in assets.items():
        path, digest = _checked_file({"path": source_path, "sha256": expected}, sidecar.parent, "collision_asset")
        if path in dependencies:
            raise ValueError("collision_asset_sha256: duplicate resolved path")
        dependencies[path] = digest
    world = _world_transform(metadata.get("world_position", [0, 0, 0]), metadata.get("world_quat_wxyz", [1, 0, 0, 0]))
    components = _read_collision_geometry(urdf_path, dependencies, world)
    identity = dict(schema=1, robot=metadata["robot"], motion_sha256=motion_digest,
                    source_motion_sha256=metadata["source_motion"]["sha256"], scene_urdf_sha256=urdf_digest,
                    collision_asset_sha256=sorted(dependencies.values()), source_first_root_xy=first_xy.tolist(),
                    ground=metadata["ground"], world_transform=world.tolist())
    # The source URDF hash binds dependency paths/transform order; additionally
    # bind the realized vertices/faces so future parser changes are detectable.
    geometry_digest = hashlib.sha256()
    for component in components:
        geometry_digest.update(np.asarray(component.mesh.vertices, dtype="<f8").tobytes())
        geometry_digest.update(np.asarray(component.mesh.faces, dtype="<i8").tobytes())
    identity["realized_collision_sha256"] = geometry_digest.hexdigest()
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return ReferenceScene(sidecar, bound_motion, motion_digest, urdf_path, urdf_digest, first_xy.copy(), components,
                          ground, fingerprint, metadata)
