"""Read-only self-collision audit using the model's effective collision filters.

    python -m morph_mimic.scripts.inspect self-collision \
        --motion datasets/g1_box_climb_seed_v2/climb_03_z_scale_1.0_scene_fit_0000_0501.npz \
        --scene datasets/g1_box_climb_seed_v2/climb_03_z_scale_1.0_scene_fit_0000_0501.scene.json \
        --raw-reference datasets/g1_box_climb_seed_v2/raw_reference/climb_03_z_scale_1.0_scene_fit_0000_0501.npz \
        --output runs/_reports/preflight/self_collision_recheck.json

The 2 cm maximum overlap gate is an engineering threshold. Collision masks,
explicit excludes, and MuJoCo's parent/weld filtering are preserved. Only
robot-robot contacts returned by mj_collision count; no integration or force
measurement is performed and no model or reference file is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from morph_mimic.contacts.self_collision import sample_self_contacts
from morph_mimic.scripts.label_reference_geometry import _write_new
from morph_mimic.diagnostics.validate_scene_reference import _joint_indices, build_model
from morph_mimic.utils.motion import validate_motion_arrays
from morph_mimic.utils.reference_scene import load_reference_scene
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def check_reference(motion_path, scene_path, raw_reference=None, robot="g1"):
    import mujoco

    motion_path, scene_path = Path(motion_path).resolve(), Path(scene_path).resolve()
    spec = RobotSpec.load(robot)
    scene = load_reference_scene(scene_path, motion_path=motion_path)
    model_path = PROJECT_ROOT / spec.runtime_assets["mjcf"]
    model, scene_ids = build_model(model_path, scene, spec.physics_dt)
    qids, _ = _joint_indices(model, spec)

    def check(path, require_nominal_held):
        path = Path(path).resolve()
        original_hash = _hash(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        validate_motion_arrays(arrays, spec, path, require_nominal_held=require_nominal_held)
        fps = float(np.asarray(arrays["fps"]).item())
        order = [arrays["joint_names"].tolist().index(name) for name in spec.joint_names]
        base = arrays["body_names"].tolist().index(spec.base_link)
        states = np.zeros((len(arrays["joint_pos"]), model.nq))
        states[:, :3], states[:, 3:7] = arrays["body_pos_w"][:, base], arrays["body_quat_w"][:, base]
        states[:, qids] = arrays["joint_pos"][:, order]
        result = sample_self_contacts(model, states, scene_ids, fps)
        result.update(source_motion=str(path), source_sha256=original_hash, source_unchanged=_hash(path) == original_hash)
        return result

    retarget = check(motion_path, True)
    raw = check(raw_reference, False) if raw_reference is not None else None
    if raw and (raw["num_frames"] != retarget["num_frames"] or raw["fps"] != retarget["fps"]):
        raise ValueError("raw comparison needs the same sampling rate and frame count")
    excluded = []
    for signature in model.exclude_signature:
        first, second = int(signature) >> 16, int(signature) & 65535
        excluded.append([model.body(first).name, model.body(second).name])
    return dict(
        schema=1, method="mujoco_mj_collision_effective_filters_reference_self_audit_v1",
        accepted=retarget["accepted"], robot=robot, max_penetration_threshold_m=.02,
        threshold_basis="project engineering limit; not a paper acceptance criterion",
        retarget_reference=retarget, raw29_reference=raw,
        scene_fingerprint=scene.fingerprint, scene_path=str(scene_path), scene_sha256=_hash(scene_path),
        reference_model=str(model_path), reference_model_sha256=_hash(model_path),
        explicit_excluded_body_pairs=excluded,
        model_disableflags=int(model.opt.disableflags), collision_masks_modified=False,
        exclusion_filters_modified=False, parent_weld_filtering="unchanged from compiled MuJoCo model",
        collision_scope="robot-robot contacts emitted by mj_collision; environment pairs excluded",
        physics_validated=False, forces_used=False, simulation_steps=0,
        reference_state_set_each_frame=True, mujoco_version=mujoco.__version__,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=["g1", "mojia"], default="g1")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--raw-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        protected = {args.motion.resolve(), args.scene.resolve()}
        if args.raw_reference:
            protected.add(args.raw_reference.resolve())
        if args.output.resolve() in protected:
            raise ValueError("output cannot replace source motion, scene, or raw reference")
        report = check_reference(args.motion, args.scene, args.raw_reference, args.robot)
        _write_new(args.output, lambda stream: stream.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode()))
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(accepted=report["accepted"], retarget_max_penetration_m=report["retarget_reference"]["maximum_penetration_m"],
                         retarget_frames_over_threshold=report["retarget_reference"]["frames_over_threshold"],
                         raw_max_penetration_m=report["raw29_reference"]["maximum_penetration_m"] if report["raw29_reference"] else None)))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
