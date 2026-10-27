"""Procedural terrain functions for reference-anchored and goal-reaching training.

Follows the Isaac Lab trimesh terrain convention: each function takes
(difficulty, cfg) and returns (meshes, origin). Terrains are grouped as
in the paper:

- reference-anchored: flat, rough (spatially aligned to motion clips)
- goal-reaching (dense foot):   stairs, walled stairs, slopes, platforms
- goal-reaching (sparse foot):  gaps, beams, pallets, stones (grid/paired/double)
- goal-reaching (dense whole-body):  boxes, uneven climbing
- goal-reaching (sparse whole-body): vaults  (ladders: TODO, need overhang geometry)
- evaluation: distractors, compositions (built from the above)

All tiles are laid out in the +x/+y quadrant with the base plane at z=0;
the origin returned is the lower-left corner. Spawns face +x.
"""

from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _heightmap_mesh(z: np.ndarray, size: tuple[float, float], origin_z: float = 0.0) -> trimesh.Trimesh:
    """Build a heightfield mesh from a (nx, ny) height grid."""
    nx, ny = z.shape
    xs = np.linspace(0.0, size[0], nx)
    ys = np.linspace(0.0, size[1], ny)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.stack((xx.flatten(), yy.flatten(), z.flatten()), axis=-1)
    f = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            f.append((a, a + 1, a + ny + 1))
            f.append((a, a + ny + 1, a + ny))
    return trimesh.Trimesh(vertices=vertices, faces=np.array(f))


def _make_plane(size, height=0.0):
    return trimesh.creation.box(extents=(size[0], size[1], 0.05)).apply_translation((size[0]/2.0, size[1]/2.0, height - 0.05/2))


def _box(cx, cy, cz, sx, sy, sz):
    """Create a box with XY center (cx, cy) and bottom elevation cz."""
    return trimesh.creation.box(extents=(sx, sy, sz)).apply_translation((cx, cy, cz + sz / 2.0))


def _border(size, height=0.4, thickness=0.15):
    """Four walls around the tile to contain the robot."""
    sx, sy = size
    return [
        _box(sx / 2.0, -thickness / 2.0, 0.0, sx, thickness, height),
        _box(sx / 2.0, sy + thickness / 2.0, 0.0, sx, thickness, height),
        _box(-thickness / 2.0, sy / 2.0, 0.0, thickness, sy, height),
        _box(sx + thickness / 2.0, sy / 2.0, 0.0, thickness, sy, height),
    ]


# ---------------------------------------------------------------------------
# reference-anchored terrains




def flat_terrain(difficulty: float, cfg: FlatCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    del difficulty
    return [_make_plane(cfg.size), *_border(cfg.size)], np.array(
        [cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0]
    )




def rough_terrain(difficulty: float, cfg: RoughCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = 64
    x = np.linspace(0, cfg.size[0], n)
    y = np.linspace(0, cfg.size[1], n)
    xx, yy = np.meshgrid(x, y)
    z = np.zeros_like(xx)
    for i in range(5):
        z += rng.uniform(-1, 1) * cfg.amplitude * difficulty * np.sin(
            rng.uniform(0.1, 1.5) * xx + rng.uniform(0, 6.28)
        ) * np.cos(rng.uniform(0.1, 1.5) * yy + rng.uniform(0, 6.28))
    mesh = _heightmap_mesh(z, cfg.size, origin_z=0.0)
    return [mesh, *_border(cfg.size)], np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])


def reference_scene_terrain(difficulty: float, cfg: ReferenceSceneCfg):
    """Use every source collision surface, with the motion's exact XY rebase.

    TerrainGenerator subtracts size/2 from meshes and origins before tiling.
    One merged mesh also lets RayCaster see all boxes and the explicit ground.
    """
    del difficulty
    from morph_mimic.utils.reference_scene import reference_scene_mesh_copy

    mesh = reference_scene_mesh_copy(cfg.sidecar_path, getattr(cfg, "fingerprint", ""))
    half = np.asarray(cfg.size, dtype=float) / 2.0
    if (np.abs(mesh.bounds[:, :2]) > half + 1e-6).any():
        raise ValueError("reference scene extends beyond the configured terrain tile")
    mesh.apply_translation((*half, 0.0))
    return [mesh], np.array([*half, 0.0])


# ---------------------------------------------------------------------------
# goal-reaching terrains: dense foot




def stairs_terrain(difficulty: float, cfg: StairsCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    h = cfg.step_height[0] + difficulty * (cfg.step_height[1] - cfg.step_height[0])
    d = cfg.step_depth[0] + difficulty * (cfg.step_depth[1] - cfg.step_depth[0])
    n = int(cfg.num_steps[0] + difficulty * (cfg.num_steps[1] - cfg.num_steps[0]))
    n = int(rng.integers(max(3, n - 2), n + 3))
    x0 = 2.0  # start of stairs
    width = 2.2
    y0 = (cfg.size[1] - width) / 2.0
    meshes = []
    for i in range(n):
        z_top = (i + 1) * h
        meshes.append(_box(x0 + (i + 0.5) * d, cfg.size[1] / 2.0, 0.0, d, width, z_top))
    top_len = max(1.5, cfg.size[0] - x0 - n * d)
    if top_len > 0.3:
        meshes.append(_box(x0 + n * d + top_len / 2.0, cfg.size[1] / 2.0, 0.0, top_len, width, n * h))
    if cfg.walled:
        meshes.append(_box(x0 + n * d / 2.0, y0 - 0.12, 0.0, n * d, 0.25, n * h))
        meshes.append(_box(x0 + n * d / 2.0, y0 + width + 0.12, 0.0, n * d, 0.25, n * h))
    meshes.extend(_border(cfg.size))
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])




def slope_terrain(difficulty: float, cfg: SlopeCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    s = cfg.slope[0] + difficulty * (cfg.slope[1] - cfg.slope[0])
    s *= rng.choice([-1.0, 1.0])  # up or down
    L = cfg.length[0] + difficulty * (cfg.length[1] - cfg.length[0])
    x0, width = 2.0, 2.2
    y0 = (cfg.size[1] - width) / 2.0
    # explicit ramp prism (8 vertices, 12 triangles)
    h = abs(s) * L
    t = 0.15  # thickness
    z_top = np.array([0.0, h, 0.0, h])
    z_bot = z_top - t
    corners = np.array(
        [
            [x0, y0, z_bot[0]], [x0 + L, y0, z_bot[1]], [x0, y0 + width, z_bot[2]], [x0 + L, y0 + width, z_bot[3]],
            [x0, y0, z_top[0]], [x0 + L, y0, z_top[1]], [x0, y0 + width, z_top[2]], [x0 + L, y0 + width, z_top[3]],
        ]
    )
    b0, b1, b2, b3 = 0, 1, 2, 3
    t0, t1, t2, t3 = 4, 5, 6, 7
    faces = [
        [b0, b2, b1], [b1, b2, b3],  # bottom
        [t0, t1, t2], [t1, t3, t2],  # top
        [b0, t1, t0], [b0, b1, t1],  # front (low)
        [b1, b3, t3], [b1, t3, t1],  # high end
        [b2, b0, t0], [b2, t0, t2],  # back
        [b3, b2, t2], [b3, t2, t3],  # side
    ]
    ramp = trimesh.Trimesh(vertices=corners, faces=faces)
    meshes = [ramp, *_border(cfg.size)]
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])




def platform_terrain(difficulty: float, cfg: PlatformCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    h = cfg.height[0] + difficulty * (cfg.height[1] - cfg.height[0])
    L = cfg.length[0] + difficulty * (cfg.length[1] - cfg.length[0])
    x0 = 2.0 + rng.uniform(0, 0.8)
    width = 2.2
    meshes = [_box(x0 + L / 2.0, cfg.size[1] / 2.0, 0.0, L, width, h), *_border(cfg.size)]
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])


# ---------------------------------------------------------------------------
# goal-reaching terrains: sparse foot




def gap_terrain(difficulty: float, cfg: GapCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    w = cfg.width[0] + difficulty * (cfg.width[1] - cfg.width[0])
    x0 = 2.0 + rng.uniform(0, 1.0)
    depth = 1.0
    side = 0.35
    left = _box(x0 - side / 2.0, cfg.size[1] / 2.0, 0.0, side, cfg.size[1], 0.35)
    right_x = x0 + w
    right = _box(right_x + (cfg.size[0] - right_x) / 2.0, cfg.size[1] / 2.0, 0.0, cfg.size[0] - right_x, cfg.size[1], 0.35)
    # pit floor at depth so falls into the gap count as failure via height check
    pit = _box((x0 + right_x) / 2.0, cfg.size[1] / 2.0, -depth - 0.05, w, cfg.size[1], 0.05)
    meshes = [left, right, pit, *_border(cfg.size)]
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])




def beam_terrain(difficulty: float, cfg: BeamCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    w = cfg.width[0] + difficulty * (cfg.width[1] - cfg.width[0])
    L = cfg.length[0] + difficulty * (cfg.length[1] - cfg.length[0])
    h = cfg.height[0] + difficulty * (cfg.height[1] - cfg.height[0])
    x0 = 2.0 + rng.uniform(0, 0.8)
    y0 = cfg.size[1] / 2.0 + rng.uniform(-0.8, 0.8)
    beam = _box(x0 + L / 2.0, y0, 0.0, L, w, h)
    # support pillars on both sides
    meshes = [
        _box(x0 - 0.15, y0, 0.0, 0.3, 0.3, h),
        _box(x0 + L + 0.15, y0, 0.0, 0.3, 0.3, h),
        beam,
        *_border(cfg.size),
    ]
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])




def pallet_terrain(difficulty: float, cfg: PalletCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    w = cfg.slat_width[0] + difficulty * (cfg.slat_width[1] - cfg.slat_width[0])
    g = cfg.gap[0] + difficulty * (cfg.gap[1] - cfg.gap[0])
    n = int(cfg.num_slats[0] + difficulty * (cfg.num_slats[1] - cfg.num_slats[0]))
    h = cfg.height[0] + difficulty * (cfg.height[1] - cfg.height[0])
    x0 = 2.0
    meshes = []
    for i in range(n):
        meshes.append(_box(x0 + i * (w + g) + w / 2.0, cfg.size[1] / 2.0, 0.0, w, 2.0, h))
    meshes.extend(_border(cfg.size))
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])




def stones_terrain(difficulty: float, cfg: StonesCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    s = cfg.stone_size[0] + difficulty * (cfg.stone_size[1] - cfg.stone_size[0])
    sp = cfg.spacing[0] + difficulty * (cfg.spacing[1] - cfg.spacing[0])
    h = cfg.height[0] + difficulty * (cfg.height[1] - cfg.height[0])
    n = int(cfg.num_stones[0] + difficulty * (cfg.num_stones[1] - cfg.num_stones[0]))
    x0, y0 = 2.0, cfg.size[1] / 2.0
    meshes = []
    if cfg.layout == "paired":
        for i in range(n):
            y_off = rng.uniform(-0.3, 0.3)
            meshes.append(_box(x0 + i * sp, y0 - 0.25 + y_off, 0.0, s, s, h))
            meshes.append(_box(x0 + i * sp, y0 + 0.25 + y_off, 0.0, s, s, h))
    elif cfg.layout == "double":
        for i in range(n):
            y_off = rng.uniform(-0.25, 0.25)
            meshes.append(_box(x0 + i * sp, y0 - 0.28 + y_off, 0.0, s, 1.4 * s, h))
            meshes.append(_box(x0 + i * sp, y0 + 0.28 + y_off, 0.0, s, 1.4 * s, h))
    else:  # grid: 3 rows (left/center/right)
        for i in range(n):
            col = i % 3 - 1
            x_off = rng.uniform(-0.05, 0.05)
            y_off = col * 0.45 + rng.uniform(-0.08, 0.08)
            meshes.append(_box(x0 + i * sp + x_off, y0 + y_off, 0.0, s, s, h))
    meshes.extend(_border(cfg.size))
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])


# ---------------------------------------------------------------------------
# goal-reaching terrains: dense whole-body




def box_terrain(difficulty: float, cfg: BoxCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    h = cfg.height[0] + difficulty * (cfg.height[1] - cfg.height[0])
    L = cfg.length[0] + difficulty * (cfg.length[1] - cfg.length[0])
    x0 = 2.0 + rng.uniform(0, 0.6)
    width = 1.6
    meshes = [_box(x0 + L / 2.0, cfg.size[1] / 2.0, 0.0, L, width, h), *_border(cfg.size)]
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])




def uneven_climb_terrain(difficulty: float, cfg: UnevenClimbCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    h = cfg.height[0] + difficulty * (cfg.height[1] - cfg.height[0])
    x0 = 2.0
    width = 2.0
    n = 5
    meshes = []
    for i in range(n):
        ih = h * (0.55 + 0.45 * rng.uniform(0, 1))
        il = (cfg.size[0] - x0) / n
        meshes.append(_box(x0 + i * il + il / 2.0, cfg.size[1] / 2.0, 0.0, il, width, ih))
    meshes.extend(_border(cfg.size))
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])


# ---------------------------------------------------------------------------
# goal-reaching terrains: sparse whole-body




def vault_terrain(difficulty: float, cfg: VaultCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    h = cfg.height[0] + difficulty * (cfg.height[1] - cfg.height[0])
    t = cfg.thickness[0] + difficulty * (cfg.thickness[1] - cfg.thickness[0])
    x0 = 2.0 + rng.uniform(0, 1.0)
    width = 2.4
    meshes = [_box(x0 + t / 2.0, cfg.size[1] / 2.0, 0.0, t, width, h), *_border(cfg.size)]
    return meshes, np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])


TERRAIN_FUNCS = {
    "flat": flat_terrain,
    "rough": rough_terrain,
    "stairs": stairs_terrain,
    "walled_stairs": stairs_terrain,
    "slope": slope_terrain,
    "platform": platform_terrain,
    "gap": gap_terrain,
    "beam": beam_terrain,
    "pallet": pallet_terrain,
    "stones_grid": stones_terrain,
    "stones_paired": stones_terrain,
    "stones_double": stones_terrain,
    "box": box_terrain,
    "uneven_climb": uneven_climb_terrain,
    "vault": vault_terrain,
}

# terrain groups used for buffer partitioning and style-term masks
REF_TERRAIN_TYPES = ("flat", "rough")
DENSE_FOOT = ("stairs", "walled_stairs", "slope", "platform", "flat", "rough")
SPARSE_FOOT = ("gap", "beam", "pallet", "stones_grid", "stones_paired", "stones_double")
DENSE_WHOLE_BODY = ("box", "uneven_climb", "reference_scene")
SPARSE_WHOLE_BODY = ("vault",)

TERRAIN_GROUP = {}
for t in DENSE_FOOT:
    TERRAIN_GROUP[t] = "dense_foot"
for t in SPARSE_FOOT:
    TERRAIN_GROUP[t] = "sparse_foot"
for t in DENSE_WHOLE_BODY:
    TERRAIN_GROUP[t] = "dense_whole_body"
for t in SPARSE_WHOLE_BODY:
    TERRAIN_GROUP[t] = "sparse_whole_body"


# ---------------------------------------------------------------------------
# configuration classes (function must live in the class body: isaaclab's
# configclass copies annotated fields into the instance dict)
from isaaclab.terrains import SubTerrainBaseCfg as _SBC  # noqa: E402


@configclass
class FlatCfg(_SBC):
    function = flat_terrain


@configclass
class RoughCfg(_SBC):
    amplitude: float = 0.05
    seed: int = 0
    function = rough_terrain


@configclass
class ReferenceSceneCfg(_SBC):
    sidecar_path: str = ""
    fingerprint: str = ""
    function = reference_scene_terrain


@configclass
class StairsCfg(_SBC):
    step_height: tuple[float, float] = (0.06, 0.18)
    step_depth: tuple[float, float] = (0.24, 0.32)
    num_steps: tuple[int, int] = (6, 14)
    walled: bool = False
    seed: int = 0
    function = stairs_terrain


@configclass
class SlopeCfg(_SBC):
    slope: tuple[float, float] = (0.15, 0.45)
    length: tuple[float, float] = (2.0, 4.0)
    seed: int = 0
    function = slope_terrain


@configclass
class PlatformCfg(_SBC):
    height: tuple[float, float] = (0.15, 0.6)
    length: tuple[float, float] = (1.0, 2.5)
    seed: int = 0
    function = platform_terrain


@configclass
class GapCfg(_SBC):
    width: tuple[float, float] = (0.25, 0.8)
    seed: int = 0
    function = gap_terrain


@configclass
class BeamCfg(_SBC):
    width: tuple[float, float] = (0.12, 0.3)
    length: tuple[float, float] = (1.5, 3.0)
    height: tuple[float, float] = (0.1, 0.4)
    seed: int = 0
    function = beam_terrain


@configclass
class PalletCfg(_SBC):
    slat_width: tuple[float, float] = (0.1, 0.25)
    gap: tuple[float, float] = (0.15, 0.4)
    num_slats: tuple[int, int] = (3, 6)
    height: tuple[float, float] = (0.08, 0.3)
    seed: int = 0
    function = pallet_terrain


@configclass
class StonesCfg(_SBC):
    stone_size: tuple[float, float] = (0.22, 0.5)
    spacing: tuple[float, float] = (0.35, 0.9)
    height: tuple[float, float] = (0.08, 0.35)
    num_stones: tuple[int, int] = (4, 9)
    layout: str = "grid"
    seed: int = 0
    function = stones_terrain


@configclass
class BoxCfg(_SBC):
    height: tuple[float, float] = (0.4, 1.0)
    length: tuple[float, float] = (0.5, 1.5)
    seed: int = 0
    function = box_terrain


@configclass
class UnevenClimbCfg(_SBC):
    height: tuple[float, float] = (0.3, 0.7)
    seed: int = 0
    function = uneven_climb_terrain


@configclass
class VaultCfg(_SBC):
    height: tuple[float, float] = (0.4, 0.6)
    thickness: tuple[float, float] = (0.12, 0.25)
    seed: int = 0
    function = vault_terrain


# default goal-reaching suite used when ref_env_ratio < 1
GOAL_TERRAIN_SUITE = tuple(t for t in TERRAIN_GROUP if t not in (*REF_TERRAIN_TYPES, "reference_scene"))
