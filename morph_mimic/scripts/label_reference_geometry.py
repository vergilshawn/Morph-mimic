"""Diagnose geometric reference contacts; write labels only after quality gates.

    python -m morph_mimic.scripts.label_reference_geometry --robot g1 \
        --motion datasets/g1_box_climb_seed_v2/climb_03_z_scale_1.0_scene_fit_0000_0501.npz \
        --scene datasets/g1_box_climb_seed_v2/climb_03_z_scale_1.0_scene_fit_0000_0501.scene.json \
        --compare-raw datasets/g1_box_climb_seed_v2/raw_reference/climb_03_z_scale_1.0_scene_fit_0000_0501.npz \
        --report datasets/g1_box_climb_seed_v2/diagnostics/climb_18_geometry.json

Default execution produces diagnostics, never training labels. --write-labels
requests a new *_contacts.npz only if all geometric gates pass. Failure still
saves diagnostics and exits nonzero. Geometry proximity uses an engineering
1 cm threshold, with a 2 cm maximum robot-scene penetration gate; no force
threshold, force proxy, PD bootstrap, or physics-validation claim is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from morph_mimic.contacts.reference_geometry import GeometryThresholds, diagnose_reference


def _write_new(path, writer):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
            stream.flush()
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_outputs(report, samples, labels, report_path, samples_path, label_path=None):
    """Preserve source/previous outputs, and refuse failed candidate labels."""
    report_path, samples_path = Path(report_path).resolve(), Path(samples_path).resolve()
    paths = [report_path, samples_path] + ([Path(label_path).resolve()] if label_path is not None else [])
    protected = {Path(report["source_motion"]).resolve(), Path(report["scene_path"]).resolve()}
    if report.get("raw_comparison"):
        protected.add(Path(report["raw_comparison"]["motion_path"]).resolve())
    if len(set(paths)) != len(paths) or any(path in protected for path in paths):
        raise ValueError("output paths must be distinct and cannot replace source motion/scene files")
    if report_path.stem.endswith("_contacts") or samples_path.stem.endswith("_contacts"):
        raise ValueError("diagnostics must not use the training _contacts filename")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    _write_new(samples_path, lambda stream: np.savez_compressed(stream, **samples))
    report["diagnostic_samples"] = str(samples_path)
    allowed = label_path is not None and report["accepted"]
    if allowed:
        _write_new(Path(label_path), lambda stream: np.savez_compressed(stream, **labels))
        report["training_contact_labels_generated"] = True
        report["contact_labels_path"] = str(Path(label_path).resolve())
    elif label_path is not None:
        report["requested_label_output_refused"] = True
    _write_new(report_path, lambda stream: stream.write((json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()))
    return allowed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=["g1", "mojia"], default="g1")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--compare-raw", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=Path, help="diagnostic NPZ; defaults to --report with .npz suffix")
    parser.add_argument("--min-segment-frames", type=int, default=1,
                        help="optional engineering denoising, default 1 preserves raw geometric contacts")
    parser.add_argument("--require-hand-contact", action="store_true",
                        help="require sustained arm-end support for this particular task; not required for all box climbs")
    parser.add_argument("--write-labels", action="store_true", help="write new motion-adjacent contacts only if geometry gates pass")
    args = parser.parse_args()
    try:
        thresholds = GeometryThresholds(min_segment_frames=args.min_segment_frames,
                                        require_hand_contact=args.require_hand_contact)
        report, samples, labels = diagnose_reference(args.motion, args.scene, args.robot, thresholds, args.compare_raw)
        label_path = args.motion.with_name(args.motion.stem + "_contacts.npz") if args.write_labels else None
        write_outputs(report, samples, labels, args.report, args.samples or args.report.with_suffix(".npz"), label_path)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: report[key] for key in ("accepted", "any_contact_fraction", "robot_scene_max_penetration_m",
                                                 "maximum_no_contact_gap_s", "training_contact_labels_generated", "rejection_reasons")}, ensure_ascii=False))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
