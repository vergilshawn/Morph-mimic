"""Evaluate recorded contact flags against a complete reference, without Isaac.

The primary comparison uses the environment's recorded force-history flags.
Reference labels may be geometric candidates, so agreement does not establish
load-bearing support. Unexecuted portions are coverage gaps, never true negatives.

Example::

    python -m morph_mimic.scripts.evaluate_contact_trajectory \
        --trajectory runs/example/from_start.npz \
        --motion datasets/example/clip.npz \
        --labels datasets/example/clip_contacts.npz --out runs/example/contacts.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


def _binary(value, name, shape=None):
    value = np.asarray(value)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    if not np.isfinite(value).all() or not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must contain finite binary flags")
    return value.astype(bool)


def _scalar(value, name):
    values = np.asarray(value).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"{name} must be scalar")
    return values[0].item()


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def confusion(reference, actual):
    """No-positive and no-prediction cases remain undefined, rather than 100%."""
    reference, actual = np.asarray(reference, dtype=bool), np.asarray(actual, dtype=bool)
    tp = int((reference & actual).sum())
    fp = int((~reference & actual).sum())
    fn = int((reference & ~actual).sum())
    tn = int((~reference & ~actual).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                reference_positive_samples=tp + fn, measured_positive_samples=tp + fp,
                precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
                f1=_ratio(2 * tp, 2 * tp + fp + fn))


def _runs(flags):
    edges = np.diff(np.r_[False, np.asarray(flags, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _event_report(reference, actual, times, indices, *, fps, duration, window_start):
    window_end = float(times[-1])
    expected = []
    for start, stop in _runs(reference):
        begin, end = start / fps, min(stop / fps, duration)
        overlap = max(0.0, min(end, window_end) - max(begin, window_start))
        status = ("not_evaluated_before_partial_reset" if end <= window_start else
                  "not_reached_after_rollout" if begin > window_end else
                  "observed_interval")
        expected.append(dict(start_frame=start, stop_frame_exclusive=stop,
                             start_s=begin, end_s=end, positive_samples=stop - start,
                             duration_s=end - begin, observed_duration_s=overlap,
                             observed_reference_positive_samples=int(((indices >= start) & (indices < stop)).sum()),
                             coincident_positive_samples=int((actual & (indices >= start) & (indices < stop)).sum()),
                             duration_coverage_fraction=_ratio(overlap, end - begin),
                             observation_status=status,
                             reference_onset_censored=start == 0,
                             reference_end_censored=stop == len(reference),
                             matched_measured_event=None, onset_error_s=None, end_error_s=None))
    observed = []
    for start, stop in _runs(actual):
        # Events are sampled at control endpoints. No force state is recorded at
        # reset, and the final positive endpoint does not reveal its release.
        observed.append(dict(start_s=float(times[start]),
                             end_s=float(times[stop]) if stop < len(times) else float(times[-1]),
                             first_positive_sample=start, stop_sample_exclusive=stop,
                             onset_censored=start == 0, end_censored=stop == len(times),
                             matched_reference_event=None))
    candidates = []
    for ri, ref in enumerate(expected):
        if ref["observation_status"] != "observed_interval":
            continue
        for ai, act in enumerate(observed):
            # Positive endpoint evidence can detect a one-sample event even
            # when its conservative sampled duration is zero.
            start, stop = act["first_positive_sample"], act["stop_sample_exclusive"]
            hits = int(((indices[start:stop] >= ref["start_frame"]) &
                        (indices[start:stop] < ref["stop_frame_exclusive"])).sum())
            if hits:
                candidates.append((hits, ri, ai))
    for _, ri, ai in sorted(candidates, reverse=True):
        ref, act = expected[ri], observed[ai]
        if ref["matched_measured_event"] is not None or act["matched_reference_event"] is not None:
            continue
        ref["matched_measured_event"] = ai
        act["matched_reference_event"] = ri
        if not act["onset_censored"] and not ref["reference_onset_censored"] and ref["start_s"] > window_start:
            ref["onset_error_s"] = act["start_s"] - ref["start_s"]
        if not act["end_censored"] and not ref["reference_end_censored"] and ref["end_s"] <= window_end:
            ref["end_error_s"] = act["end_s"] - ref["end_s"]
    for ref in expected:
        if ref["observation_status"] == "observed_interval":
            ref["detection_status"] = (
                "reference_event_not_sampled_at_control_endpoint" if not ref["observed_reference_positive_samples"] else
                "missed_within_observed_interval" if not ref["coincident_positive_samples"] else
                "matched" if ref["matched_measured_event"] is not None else
                "contact_observed_without_unique_event_match")
        else:
            ref["detection_status"] = "not_evaluated"
    total_duration = sum(event["duration_s"] for event in expected)
    observed_duration = sum(event["observed_duration_s"] for event in expected)
    summary = dict(
        reference_events=len(expected),
        reference_events_overlapping_rollout=sum(e["observation_status"] == "observed_interval" for e in expected),
        reference_events_not_reached=sum(e["observation_status"] == "not_reached_after_rollout" for e in expected),
        reference_events_before_partial_reset=sum(e["observation_status"] == "not_evaluated_before_partial_reset" for e in expected),
        matched_reference_events=sum(e["matched_measured_event"] is not None for e in expected),
        missed_reference_events_within_rollout=sum(e["detection_status"] == "missed_within_observed_interval" for e in expected),
        reference_events_with_contact_but_no_unique_match=sum(e["detection_status"] == "contact_observed_without_unique_event_match" for e in expected),
        measured_events=len(observed), unmatched_measured_events=sum(e["matched_reference_event"] is None for e in observed),
        full_reference_positive_duration_s=total_duration,
        observed_reference_positive_duration_s=observed_duration,
        reference_positive_duration_coverage_fraction=_ratio(observed_duration, total_duration),
    )
    return dict(summary=summary, reference_events=expected, measured_events=observed)


def contact_report(trajectory, metadata, labels, *, frames, fps, force_threshold_n=1.0):
    """Pure NumPy evaluation of one contiguous, pre-reset reference rollout."""
    if frames < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Motion requires at least two frames and finite positive fps")
    if not math.isfinite(force_threshold_n) or force_threshold_n <= 0:
        raise ValueError("Force threshold must be finite and positive")
    duration = (frames - 1) / fps
    names = list(metadata["contact_body_names"])
    label_names = np.asarray(labels["body_names"]).tolist()
    if not names or len(set(names)) != len(names) or len(set(label_names)) != len(label_names) or set(names) != set(label_names):
        raise ValueError("Reference and trajectory must have the same unique contact body names")
    order = [label_names.index(name) for name in names]
    reference = _binary(labels["contact_flags"], "contact_flags", (frames, len(names)))[:, order]
    phase_raw = np.asarray(trajectory["phase"])
    phase = phase_raw.astype(np.float64)
    if phase.ndim != 1 or not len(phase) or not np.isfinite(phase).all() or (phase < 0).any() or (phase > 1 + 1e-6).any() or (np.diff(phase) <= 0).any():
        raise ValueError("Trajectory phase must be a nonempty strictly increasing sequence in [0,1]")
    n = len(phase)
    actual = _binary(trajectory["measured_contact_flags"], "measured_contact_flags", (n, len(names)))
    recorded_ref = _binary(trajectory["ref_contact_flags"], "ref_contact_flags", (n, len(names)))
    # Preserve the stored phase dtype: the environment floors float32 phase
    # times frame-count; converting to float64 first changes boundary indices.
    indices = np.clip((phase_raw * (frames - 1)).astype(np.int64), 0, frames - 1)
    if not np.array_equal(reference[indices], recorded_ref):
        raise ValueError("Recorded reference flags do not match labels at the recorded phase")
    start_phase = float(metadata.get("result", {}).get("start_phase", trajectory.get("initial_phase", 0.0)))
    dt = float(metadata["control_dt_s"])
    if not math.isfinite(start_phase) or not 0 <= start_phase < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid initial phase or control timestep")
    elapsed = np.asarray(trajectory["elapsed_control_time_s"], dtype=np.float64)
    if elapsed.shape != (n,) or not np.isfinite(elapsed).all() or not np.allclose(elapsed, (np.arange(n) + 1) * dt, atol=1e-6, rtol=0):
        raise ValueError("Trajectory must contain every contiguous control endpoint")
    times = phase * duration
    if not np.allclose(times, np.minimum(start_phase * duration + elapsed, duration), atol=1e-5, rtol=0):
        raise ValueError("Reference phase and elapsed time disagree")
    if not np.isclose(float(_scalar(labels["fps"], "labels fps")), fps):
        raise ValueError("Label and motion fps disagree")
    for field, expected in (("source_num_frames", frames), ("source_init_phase", 0.0), ("source_end_phase", 1.0)):
        if field in labels and not np.isclose(float(_scalar(labels[field], field)), expected):
            raise ValueError(f"Labels do not describe the complete motion: {field}")
    forces = np.asarray(trajectory["contact_force_w"], dtype=np.float64)
    if forces.shape != (n, len(names), 3) or not np.isfinite(forces).all():
        raise ValueError("contact_force_w must contain finite endpoint force vectors")
    instantaneous = np.abs(forces[..., 2]) > force_threshold_n
    bodies = {}
    unique_indices = np.unique(indices)
    for b, name in enumerate(names):
        full_positives = int(reference[:, b].sum())
        covered_positives = int(reference[unique_indices, b].sum())
        bodies[name] = dict(
            force_history_vs_reference=confusion(recorded_ref[:, b], actual[:, b]),
            endpoint_abs_fz_vs_reference=confusion(recorded_ref[:, b], instantaneous[:, b]),
            history_endpoint_disagreement_samples=int((actual[:, b] != instantaneous[:, b]).sum()),
            full_reference_positive_samples=full_positives,
            unique_reference_positive_samples_visited=covered_positives,
            reference_positive_sample_coverage_fraction=_ratio(covered_positives, full_positives),
            events=_event_report(reference[:, b], actual[:, b], times, indices, fps=fps,
                                 duration=duration, window_start=start_phase * duration),
        )
    method = _scalar(labels.get("method", labels.get("label_method", np.asarray("unspecified"))), "label method")
    return dict(
        schema=1,
        definitions=dict(
            reference=f"Complete reference contact_flags; label method: {method}",
            measured="Recorded environment flags: maximum absolute world-Z force over sensor history exceeds threshold; not instantaneous load-bearing ground truth",
            instantaneous="Recorded endpoint absolute world-Z force exceeds threshold; does not reconstruct force history or surface-normal force",
            classification="Only recorded physical endpoints are classified; no unexecuted interval contributes TP/FP/FN/TN",
            undefined_metrics="null when the denominator is zero; absent positive contacts never receive an invented perfect score",
            events="Contiguous positive runs; one-to-one matching prioritizes the largest number of coincident positive endpoints",
            timing="Reference timeline in seconds; positive offset means late. Censored transitions have null offsets; resolution is the control timestep plus sensor-history lag",
            reference_sample_coverage="Unique floor-sampled source indices; reset frame is excluded from endpoint traces, so a complete rollout may not visit every source frame",
        ),
        reference_label_provenance=dict(
            method=method,
            physics_validated=bool(_scalar(labels["physics_validated"], "physics_validated")) if "physics_validated" in labels else None,
            forces_used=bool(_scalar(labels["forces_used"], "forces_used")) if "forces_used" in labels else None,
            scene_fingerprint=str(_scalar(labels["scene_fingerprint"], "scene_fingerprint")) if "scene_fingerprint" in labels else None,
        ),
        force_threshold_n=force_threshold_n,
        coverage=dict(
            duration_s=duration, rollout_start_reference_s=start_phase * duration,
            rollout_end_reference_s=min(float(times[-1]), duration),
            observed_reference_duration_s=min(float(times[-1]), duration) - start_phase * duration,
            reference_duration_coverage_fraction=(min(float(times[-1]), duration) - start_phase * duration) / duration,
            reference_frames=frames, observed_control_endpoints=n, unique_reference_frames_visited=len(unique_indices),
            evaluation_is_partial=start_phase != 0.0,
            reached_reference_end=bool(phase[-1] >= 1 - 1e-5),
            task_completed=bool(metadata.get("result", {}).get("completed", False)),
        ),
        micro_force_history_vs_reference=confusion(recorded_ref, actual),
        bodies=bodies,
        unavailable_metrics=dict(
            slip="Not measured: trace lacks per-contact tangential surface motion and contact-point trajectories",
            penetration="Not measured: recorded flags and forces do not establish signed runtime collision distance",
            load_bearing_support="Not established by a force threshold alone; no contact wrench or support equilibrium analysis",
        ),
    )


def _load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _publish(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--force-threshold-n", type=float, default=1.0,
                        help="threshold for the separate endpoint-Fz comparison; history flags are never recomputed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite report: {args.out}")
    trajectory, motion, labels = (_load(path) for path in (args.trajectory, args.motion, args.labels))
    metadata = json.loads(str(_scalar(trajectory["metadata_json"], "metadata_json")))
    motion_hash = hashlib.sha256(args.motion.read_bytes()).hexdigest()
    if metadata.get("source_sha256") != motion_hash or _scalar(labels["source_sha256"], "source_sha256") != motion_hash:
        raise ValueError("Trajectory, labels and supplied motion must have identical source SHA256")
    report = contact_report(trajectory, metadata, labels, frames=len(motion["joint_pos"]),
                            fps=float(_scalar(motion["fps"], "motion fps")),
                            force_threshold_n=args.force_threshold_n)
    report["artifacts"] = {key: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                           for key, path in (("trajectory", args.trajectory), ("motion", args.motion), ("labels", args.labels))}
    report["recorded_contact_force_sampling"] = metadata.get("contact_force_sampling")
    report["evaluation_result"] = metadata.get("result")
    _publish(args.out, report)
    print(json.dumps(dict(out=str(args.out), coverage=report["coverage"],
                          micro_force_history_vs_reference=report["micro_force_history_vs_reference"]), allow_nan=False))


if __name__ == "__main__":
    main()
