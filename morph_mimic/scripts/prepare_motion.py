"""Extract a reference-only motion window without altering source assets.

    python -m morph_mimic.scripts.prepare_motion --robot g1 --clip walk1_subject1 \
        --start-s 0 --duration-s 8 --output-dir datasets/g1_walk_candidate

An 8 s window at 50 Hz contains 401 samples, including both endpoints.
This operation does not generate force contacts or validate physical tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def prepare_clip(source_path: Path, output_dir: Path, start_s: float, duration_s: float):
    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    if not np.isfinite(start_s) or not np.isfinite(duration_s) or start_s < 0 or duration_s <= 0:
        raise ValueError("start-s must be nonnegative and duration-s must be positive; both must be finite")
    destination = output_dir / source_path.name
    provenance_path = destination.with_suffix(".preparation.json")
    if destination.exists() or provenance_path.exists():
        raise FileExistsError(f"refusing to overwrite prepared output: {destination}")
    if source_path.stem.endswith("_contacts"):
        raise ValueError("select a motion clip; contact labels must be generated from its own complete replay")
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if source_path.with_suffix(".scene.json").exists() or bool(np.asarray(arrays.get("requires_scene", False)).item()):
        raise ValueError("scene-bound references require scene-aware conversion; generic cropping cannot preserve their binding")
    fps_values = np.asarray(arrays["fps"]).reshape(-1)
    if len(fps_values) != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0:
        raise ValueError("source motion must declare one finite positive fps value")
    fps = float(fps_values[0])
    frames = len(arrays["joint_pos"])
    start_float, duration_float = start_s * fps, duration_s * fps
    start_frame, duration_frames = round(start_float), round(duration_float)
    if not np.isclose(start_float, start_frame, rtol=0, atol=1e-6) or not np.isclose(duration_float, duration_frames, rtol=0, atol=1e-6):
        raise ValueError(f"start-s and duration-s must align with the source sampling interval ({1 / fps:g} s)")
    if duration_frames < 1:
        raise ValueError("requested window must span at least one sampling interval")
    stop_frame = start_frame + duration_frames + 1
    if start_frame < 0 or stop_frame > frames:
        raise ValueError(f"requested [{start_s:g}, {start_s + duration_s:g}] s exceeds source range [0, {(frames - 1) / fps:g}] s")
    # Metadata vectors are independent of time, even if their length happens
    # to equal the number of motion samples in a tiny input file.
    non_time_fields = {"fps", "joint_names", "body_names"}
    sliced_fields = []
    for key, value in arrays.items():
        if key not in non_time_fields and value.ndim > 0 and value.shape[0] == frames:
            arrays[key] = value[start_frame:stop_frame].copy()
            sliced_fields.append(key)
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if key not in sliced_fields:
            raise ValueError(f"source motion lacks a correctly sized time-axis array: {key}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"source window contains nonfinite {key}")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    metadata = dict(
        schema=1,
        source_motion=str(source_path),
        source_sha256=source_hash,
        source_num_frames=frames,
        start_frame=start_frame,
        stop_frame_exclusive=stop_frame,
        start_time_s=start_frame / fps,
        duration_s=duration_frames / fps,
        prepared_num_frames=duration_frames + 1,
        fps=fps,
        sliced_fields=sorted(sliced_fields),
        reference_only=True,
        physical_tracking_validated=False,
        contact_labels_generated=False,
    )
    arrays["preparation_metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths, created = [], []
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz.tmp", delete=False) as stream:
            temporary_paths.append(Path(stream.name))
            np.savez(stream, **arrays)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, suffix=".json.tmp", delete=False) as stream:
            temporary_paths.append(Path(stream.name))
            stream.write(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        # Hard-link publication refuses to replace a destination created by
        # another process after the checks above.
        for temporary, target in zip(temporary_paths, (destination, provenance_path)):
            os.link(temporary, target)
            created.append(target)
    except BaseException:
        for target in created:
            target.unlink()
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return destination, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["g1", "mojia"], required=True)
    parser.add_argument("--clip", required=True, help="source motion name without .npz")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="optional motion directory instead of the robot's migrated assets")
    args = parser.parse_args()
    if Path(args.clip).name != args.clip or args.clip.endswith(".npz"):
        parser.error("--clip must be a motion name without a directory or .npz extension")
    spec = json.loads((ROOT / "assets" / args.robot / "robot_spec.json").read_text())
    source_dir = args.source_dir if args.source_dir is not None else ROOT / spec["motion_dir"]
    try:
        destination, metadata = prepare_clip(source_dir / f"{args.clip}.npz", args.output_dir, args.start_s, args.duration_s)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Prepared {destination}: {metadata['prepared_num_frames']} frames at {metadata['fps']:g} Hz, {metadata['duration_s']:g} s")
    print("Reference window only; force contact labeling and physical tracking validation are still required.")


if __name__ == "__main__":
    main()
