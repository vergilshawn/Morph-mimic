"""Read-only migration/data audit; needs NumPy, never Torch or Isaac Sim.

    python -m morph_mimic.scripts.audit_assets --verify-hashes
    python -m morph_mimic.scripts.audit_assets --require-contacts --json

Missing force-derived labels are warnings unless --require-contacts is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_tree(source, destination, verify_hashes):
    if not source.is_dir():
        return {"available": False, "source": str(source)}
    files = [p for p in sorted(source.rglob("*")) if p.is_file()]
    missing, changed = [], []
    for path in files:
        target = destination / path.relative_to(source)
        if not target.is_file():
            missing.append(str(path.relative_to(source)))
        elif path.stat().st_size != target.stat().st_size or (verify_hashes and sha256(path) != sha256(target)):
            changed.append(str(path.relative_to(source)))
    return dict(available=True, source=str(source), files=len(files), missing=missing,
                changed=changed, verification="sha256" if verify_hashes else "file_size")


def audit_robot(robot, source_root, verify_hashes, deep):
    directory = ROOT / "assets" / robot
    result = {"robot": robot, "errors": [], "warnings": []}
    errors, warnings = result["errors"], result["warnings"]
    try:
        spec = json.loads((directory / "robot_spec.json").read_text())
        control = json.loads((directory / "control.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return result
    result["migration"] = compare_tree(source_root / robot, directory, verify_hashes)
    if not result["migration"]["available"]:
        warnings.append("Original migration source is unavailable; local assets can still be checked.")
    elif result["migration"]["missing"] or result["migration"]["changed"]:
        errors.append("Migrated files are missing or differ from the selected source.")
    if robot == "mojia":
        result["raw_migration"] = compare_tree(source_root / "mojia_raw", directory / "raw", verify_hashes)
        if result["raw_migration"].get("missing") or result["raw_migration"].get("changed"):
            errors.append("Migrated raw motions differ from the selected source.")
    if spec["joint_names"] != control["joint_names"]:
        errors.append("Spec and control joint order differ.")
    result["joints"] = {"total": spec["num_joints"], "controlled": len(spec["controlled_joint_names"])}
    for key in ("default_pose", "stiffness", "damping", "effort_limits", "armature", "action_scale",
                "pos_limits_lower", "pos_limits_upper", "vel_limits"):
        values = np.asarray(spec[key])
        if values.shape != (spec["num_joints"],) or not np.isfinite(values).all():
            errors.append(f"Invalid spec vector: {key}.")
    result["runtime_assets"] = spec["runtime_assets"]
    for kind in ("urdf", "mjcf"):
        relative = spec["runtime_assets"].get(kind)
        if not relative:
            continue
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing runtime {kind}: {relative}")
            continue
        try:
            tree = ET.parse(path).getroot()
            mesh_dir = tree.find("compiler").get("meshdir", "") if tree.find("compiler") is not None else ""
            for mesh in tree.findall(".//mesh"):
                filename = mesh.get("filename") if kind == "urdf" else mesh.get("file")
                if filename and not (path.parent / mesh_dir / filename).is_file():
                    errors.append(f"Missing {kind} mesh: {filename}")
        except (OSError, ET.ParseError) as exc:
            errors.append(f"Invalid runtime {kind}: {exc}")
    if spec["runtime_assets"].get("usd_cfg"):
        warnings.append("G1 runtime uses an Isaac Lab USD configuration; local MJCF motions alone do not verify model equivalence or USD availability.")
    result["data_provenance"] = {key: manifest[key] for key in
        ("reference_data_only", "physical_tracking_validated", "hardware_validated", "foot_contact_labels", "official_supermimic_code")
        if key in manifest}
    if manifest.get("physical_tracking_validated") is False:
        warnings.append("Reference motions have not passed physical tracking validation.")
    if verify_hashes:
        manifest_errors = [name for name, expected in manifest.get("files", {}).items()
                           if not (directory / name).is_file() or sha256(directory / name) != expected]
        result["manifest_hash_mismatches"] = manifest_errors
        if manifest_errors:
            errors.append("Local assets differ from their migration manifest hashes.")
        dependency_errors = []
        urdf_relative = spec["runtime_assets"].get("urdf")
        if control.get("urdf_sha256") and urdf_relative:
            urdf_path = ROOT / urdf_relative
            if not urdf_path.is_file() or sha256(urdf_path) != control["urdf_sha256"]:
                dependency_errors.append(urdf_relative)
        for original, expected in manifest.get("asset_dependencies", {}).items():
            # Mojia's source manifest keeps provenance paths from supermimic;
            # the migrated runtime URDF/meshes live under assets/mojia/robot.
            try:
                relocated = directory / "robot" / Path(original).relative_to("assets/robots/mojia")
            except ValueError:
                dependency_errors.append(f"unmapped provenance path: {original}")
                continue
            if not relocated.is_file() or sha256(relocated) != expected:
                dependency_errors.append(str(relocated.relative_to(ROOT)))
        result["dependency_hash_mismatches"] = dependency_errors
        if dependency_errors:
            errors.append("Runtime robot dependencies differ from migration provenance hashes.")
    motion_dir = ROOT / spec["motion_dir"]
    paths = [p for p in sorted(motion_dir.glob("*.npz")) if not p.stem.endswith("_contacts")]
    frame_total, durations, root_min, missing_labels, invalid_labels = 0, [], [], [], []
    clip_summaries = []
    required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                names = data["joint_names"].tolist()
                bodies = data["body_names"].tolist()
                frames = len(data["joint_pos"])
                fps = float(np.asarray(data["fps"]).item())
                if fps <= 0 or not np.isfinite(fps) or not np.isclose(fps * spec["control_dt"], 1.0):
                    errors.append(f"{path.name}: invalid fps/control frequency.")
                    continue
                if frames < 2 or len(set(names)) != len(names) or set(spec["controlled_joint_names"]) - set(names):
                    errors.append(f"{path.name}: invalid frame count or controlled-joint mapping.")
                if spec["base_link"] not in bodies:
                    errors.append(f"{path.name}: base body is absent.")
                    continue
                if data["joint_pos"].shape != (frames, len(names)):
                    errors.append(f"{path.name}: joint_pos dimensions differ from joint metadata.")
                if deep:
                    shapes = {"joint_pos": (frames, len(names)), "joint_vel": (frames, len(names)),
                              **{key: (frames, len(bodies), 4 if key == "body_quat_w" else 3) for key in required[2:]}}
                    for key in required:
                        value = data[key]
                        if value.shape != shapes[key] or not np.isfinite(value).all():
                            errors.append(f"{path.name}: invalid {key} shape or nonfinite values.")
                    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1.0, atol=1e-3):
                        errors.append(f"{path.name}: body quaternions are not normalized.")
                root_min.append(float(data["body_pos_w"][:, bodies.index(spec["base_link"]), 2].min()))
                frame_total += frames
                durations.append((frames - 1) / fps)
                clip_summaries.append({"name": path.stem, "frames": frames, "fps": fps})
            label_path = path.with_name(path.stem + "_contacts.npz")
            if not label_path.is_file():
                missing_labels.append(path.stem)
                continue
            try:
                with np.load(label_path, allow_pickle=False) as labels:
                    shapes = {"contact_flags": (frames, 9), "contact_targets": (frames, 9, 3), "contact_target_flags": (frames, 9)}
                    for key, shape in shapes.items():
                        value = labels[key]
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError(f"{key}: expected finite values with shape {shape}, got {value.shape}")
                        if key != "contact_targets" and not np.isin(value, [0, 1]).all():
                            raise ValueError(f"{key}: flags must be binary")
                    if not np.isclose(float(np.asarray(labels["fps"]).item()), fps, rtol=0, atol=1e-6):
                        raise ValueError("fps differs from source motion")
                    if "source_end_phase" in labels and float(labels["source_end_phase"]) < 1 - 1e-5:
                        raise ValueError("replay ended before the source motion")
                    if verify_hashes and "source_sha256" in labels and str(labels["source_sha256"].item()) != sha256(path):
                        raise ValueError("source motion SHA256 differs")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                invalid_labels.append({"clip": path.stem, "reason": str(exc)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path.name}: {exc}")
    if not paths:
        errors.append("No motion clips found.")
    declared = {Path(clip["file"]).stem for clip in manifest.get("clips", [])}
    actual = {path.stem for path in paths}
    if declared != actual:
        errors.append("Motion file set differs from the migration manifest.")
    result["motions"] = {"clips": len(paths), "frames": frame_total, "duration_hours": sum(durations) / 3600,
                         "minimum_root_height_m": min(root_min) if root_min else None,
                         "maximum_frames": max((clip["frames"] for clip in clip_summaries), default=0)}
    result["clips"] = clip_summaries
    result["contacts"] = {"valid": len(paths) - len(missing_labels) - len(invalid_labels),
                          "missing": missing_labels, "invalid": invalid_labels}
    if missing_labels:
        warnings.append(f"{len(missing_labels)}/{len(paths)} clips lack force-derived nine-body contact labels.")
    if invalid_labels:
        errors.append("Contact labels do not match their source motions.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["g1", "mojia", "all"], default="all")
    parser.add_argument("--source-root", type=Path, default=Path("/media/user/DATA/supermimic/data"))
    parser.add_argument("--verify-hashes", action="store_true", help="compare original and manifest SHA256 hashes")
    parser.add_argument("--deep", action="store_true", help="also inspect every motion array for dimensions and finite values")
    parser.add_argument("--require-contacts", action="store_true", help="fail if any source clip lacks full contact labels")
    parser.add_argument("--json", action="store_true", help="print machine-readable results to stdout")
    args = parser.parse_args()
    robots = ["g1", "mojia"] if args.robot == "all" else [args.robot]
    results = [audit_robot(robot, args.source_root, args.verify_hashes, args.deep) for robot in robots]
    failed = any(r["errors"] or (args.require_contacts and r.get("contacts", {}).get("missing")) for r in results)
    if args.json:
        print(json.dumps({"ok": not failed, "robots": results}, indent=2, ensure_ascii=False))
    else:
        for result in results:
            print(f"[{result['robot']}] {len(result['errors'])} errors, {len(result['warnings'])} warnings")
            if "motions" in result:
                motion, contact = result["motions"], result["contacts"]
                print(f"  {motion['clips']} clips, {motion['frames']} frames, {motion['duration_hours']:.4f} h; contacts {contact['valid']}/{motion['clips']}")
            for message in result["errors"]:
                print(f"  ERROR: {message}")
            for message in result["warnings"]:
                print(f"  WARNING: {message}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
