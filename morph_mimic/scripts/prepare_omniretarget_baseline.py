"""Stage unchanged full-29-DoF source motion and source-world terrain for a tracker baseline.

This does not create Holosoma's FK/velocity training format or certify dynamics.
The existing scene sidecar supplies checked geometry, never the projected ACP pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from morph_mimic.utils.reference_scene import load_reference_scene
from morph_mimic.utils.specs import PROJECT_ROOT

SOURCE_URDF_SHA256 = "6d79140d335157ad026d24b79be6fbb161c997c01c3ffc37bcca282dc2240696"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(scene_path: Path, output: Path, duration_s: float) -> dict:
    import trimesh

    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration must be finite and positive")
    scene = load_reference_scene(scene_path)
    if scene.metadata["robot"] != "g1":
        raise ValueError("this baseline requires the G1 source scene")
    if (not np.allclose(scene.metadata.get("world_position", [0, 0, 0]), [0, 0, 0], rtol=0, atol=1e-10)
            or not np.allclose(scene.metadata.get("world_quat_wxyz", [1, 0, 0, 0]),
                               [1, 0, 0, 0], rtol=0, atol=1e-10)):
        raise ValueError("source motion is unchanged: its scene must have an identity world transform")
    source = scene.metadata["source_motion"]
    source_path = Path(source["path"])
    if not source_path.is_absolute():
        source_path = scene.sidecar_path.parent / source_path
    source_path = source_path.resolve()
    if sha256(source_path) != source["sha256"]:
        raise ValueError("original source motion hash differs from the scene binding")
    urdf = PROJECT_ROOT / "datasets/external/omniretarget/models/g1/g1_29dof_spherehand.urdf"
    # This exact model's source-column order was checked with Drake's Parser
    # and Joint.position_start(), not inferred just from the number of DoFs.
    if sha256(urdf) != SOURCE_URDF_SHA256:
        raise ValueError("source URDF changed; repeat the independent Drake state-column audit")
    root = ET.parse(urdf).getroot()
    names = [joint.attrib["name"] for joint in root.findall("joint")
             if joint.attrib["type"] not in {"fixed", "floating"}]
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("expected the verified 29 scalar joints of the source spherehand URDF")
    with np.load(source_path, allow_pickle=False) as data:
        fps_values = np.asarray(data["fps"]).reshape(-1)
        if fps_values.size != 1 or not np.isfinite(fps_values).all() or fps_values[0] <= 0:
            raise ValueError("source fps must be a positive finite scalar")
        fps = float(fps_values[0])
        qpos = np.asarray(data["qpos"])
        source_fps_array = data["fps"].copy()
    if qpos.ndim != 2 or qpos.shape[1] != 36 or not np.isfinite(qpos).all():
        raise ValueError("source motion must be finite [T,36] WXYZ,XYZ,29-joint data")
    if not np.allclose(np.linalg.norm(qpos[:, :4], axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("source root quaternions must be normalized")
    end_float = duration_s * fps
    end = round(end_float)
    if not np.isclose(end, end_float, rtol=0, atol=1e-7) or end < 1 or end >= len(qpos):
        raise ValueError("duration must select an available exact source-frame boundary")
    # No resampling, root translation, joint projection, or pose optimization.
    selected = qpos[:end + 1].copy()
    mesh = scene.source_mesh()
    obstacle_bounds = scene.obstacle_bounds.copy()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Refuse all existing destinations, including prior results.
    motion_path = output / f"{source_path.stem}_source_full29_0000_{end + 1:04d}.npz"
    terrain_path = output / f"{source_path.stem}_source_world.obj"
    np.savez_compressed(motion_path, qpos=selected, fps=source_fps_array,
                        joint_names=np.asarray(names))
    mesh.export(terrain_path)
    with np.load(motion_path, allow_pickle=False) as written:
        if not np.array_equal(written["qpos"], qpos[:end + 1]):
            raise ValueError("staged source pose differs from the original frames")
    loaded = trimesh.load_mesh(terrain_path, process=False, skip_materials=True)
    if (len(loaded.vertices) != len(mesh.vertices) or len(loaded.faces) != len(mesh.faces)
            or not np.allclose(loaded.vertices, mesh.vertices, rtol=0, atol=1e-7)
            or not np.array_equal(loaded.faces, mesh.faces)):
        raise ValueError("exported terrain changed vertex positions or topology")
    manifest = {
        "schema": 1,
        "purpose": "OmniRetarget/Holosoma full29 tracking baseline input preparation",
        "training_ready": False,
        "physics_validated": False,
        "source_motion": {"path": str(source_path), "sha256": sha256(source_path),
                          "total_frames": len(qpos), "fps": fps},
        "source_robot": {"path": str(urdf), "sha256": sha256(urdf), "joint_names": names},
        "scene_binding": {"path": str(scene.sidecar_path), "sha256": sha256(scene.sidecar_path),
                          "fingerprint": scene.fingerprint,
                          "use": "checked source geometry and engineering ground slab only; no ACP poses"},
        "source_scene_urdf": {"path": str(scene.scene_urdf_path), "sha256": scene.scene_urdf_sha256},
        "source_collision_assets": scene.metadata["collision_asset_sha256"],
        "motion": {"path": str(motion_path), "sha256": sha256(motion_path),
                   "format": "source qpos: WXYZ,XYZ,29 source-URDF joints; not Holosoma training format",
                   "frames": len(selected), "fps": fps, "source_frame_range_inclusive": [0, end],
                   "duration_s": end / fps, "root_initial_xyz_m": selected[0, 4:7].tolist(),
                   "source_values_exactly_preserved": True, "held_joint_projection": False},
        "terrain": {"path": str(terrain_path), "sha256": sha256(terrain_path),
                    "frame": "original source world; no XY rebase or recentering",
                    "bounds_m": loaded.bounds.tolist(), "obstacle_bounds_m": obstacle_bounds.tolist(),
                    "vertices": len(loaded.vertices), "triangles": len(loaded.faces),
                    "ground": scene.metadata["ground"], "roundtrip_geometry_verified": True},
        "pending": [
            "Use isolated training and conversion environments; do not install into the running ACP env.",
            "Select and audit the full29 runtime robot, including source spherehand collision/inertia differences.",
            "Convert to named full-body FK/velocity format; audit free-root angular frame and body velocity origin.",
            "Verify one-environment source-world scene and trajectory alignment in the training backend.",
            "Verify inter-environment collision filtering and layout before increasing the environment count.",
            "Run a separate full-reference training experiment and strict phase-zero complete-clip evaluation.",
        ],
    }
    (output / "preparation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=PROJECT_ROOT / (
        "datasets/g1_box_climb_seed_v2/climb_03_z_scale_1.0_scene_fit_0000_0501.scene.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    args = parser.parse_args()
    result = prepare(args.scene, args.output_dir, args.duration_s)
    print(json.dumps({"training_ready": result["training_ready"], "motion": result["motion"],
                      "terrain": result["terrain"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
