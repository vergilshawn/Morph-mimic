"""Contact onset timing and placement proxies from physical endpoint records.

Reference-onset matching is an engineering metric, not ACP Figure 7's error
between commanded and realized contacts. Both positions here are surface
points; a link origin must never be substituted for either one. Reference
events come from the complete clip, so an early failure cannot turn its
unexecuted suffix into negative contact evidence.
"""
from __future__ import annotations

import math

import numpy as np


def _binary(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or not np.isin(value, (False, True, 0, 1)).all():
        raise ValueError(f"{name} must be binary with shape {shape}")
    return value.astype(bool)


def _runs(flags):
    edges = np.diff(np.r_[False, flags, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _presence(reference, actual):
    tp, fp = int((reference & actual).sum()), int((~reference & actual).sum())
    fn, tn = int((reference & ~actual).sum()), int((~reference & ~actual).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def _geometry(points, normals, shape, validity, name):
    points, normals = np.asarray(points, dtype=float), np.asarray(normals, dtype=float)
    if points.shape != shape + (3,) or normals.shape != shape + (3,):
        raise ValueError(f"{name} points and normals must have shape {shape + (3,)}")
    valid = np.ones(shape, dtype=bool) if validity is None else _binary(validity, shape, name + " valid")
    norms = np.linalg.norm(normals, axis=-1)
    normal_valid = (valid & np.isfinite(normals).all(axis=-1)
                    & np.isclose(norms, 1., rtol=0., atol=1e-3))
    point_valid = valid & np.isfinite(points).all(axis=-1)
    normalized = np.zeros_like(normals)
    np.divide(normals, norms[..., None], out=normalized, where=normal_valid[..., None])
    return points, normalized, point_valid, normal_valid


def _match(reference, actual, tolerance, minimum_dot):
    """Ordered maximum-cardinality match, then minimum total timing error.

    A greedy nearest-event rule can consume the only feasible match for a
    later event. Dynamic programming retains one-to-one chronological matches
    and handles incompatible surface normals without a scipy dependency.
    """
    rows, columns = len(reference), len(actual)
    count = np.zeros((rows + 1, columns + 1), dtype=int)
    cost = np.zeros((rows + 1, columns + 1))
    choice = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    for r in range(1, rows + 1):
        for a in range(1, columns + 1):
            options = [(count[r-1, a], -cost[r-1, a], 1),
                       (count[r, a-1], -cost[r, a-1], 2)]
            ref, act = reference[r-1], actual[a-1]
            delta = abs(ref["start_s"] - act["start_s"])
            compatible = (ref["normal_valid"] and act["normal_valid"]
                          and np.dot(ref["normal"], act["normal"]) >= minimum_dot - 1e-12)
            if delta <= tolerance + 1e-9 and compatible:
                options.append((count[r-1, a-1] + 1, -(cost[r-1, a-1] + delta), 3))
            best = max(options)
            count[r, a], cost[r, a], choice[r, a] = best[0], -best[1], best[2]
    matches, r, a = [], rows, columns
    while r and a:
        selected = choice[r, a]
        if selected == 3:
            matches.append((r-1, a-1))
            r, a = r-1, a-1
        elif selected == 1:
            r -= 1
        else:
            a -= 1
    return list(reversed(matches))


def contact_event_report(times_s, phase, actual_points_w, normal_load_n,
                         actual_normals_w, reference_flags, reference_points_w,
                         reference_normals_w, *, reference_fps, control_dt_s,
                         start_phase=0., actual_valid=None, reference_valid=None,
                         body_names=None, onset_tolerance_s=.2,
                         normal_angle_tolerance_rad=math.pi/3,
                         force_threshold_n=1., metadata=None):
    """Summarize one contiguous physical rollout against a complete reference.

    Actual arrays are N x B (x 3); reference arrays are F x B (x 3) for the
    *full* clip, including both endpoints. ``times_s`` are elapsed physical
    control endpoints dt, 2dt, ... after reset, even while reference phase is
    clamped at one. Repeat phase-one samples require an explicit physical-hold
    declaration in metadata; this validates record semantics, not the physical
    authenticity of fabricated timestamps. Valid points and normals should
    describe the same robot/environment interface and consistently point out
    of the supporting surface. Loads are current positive normal loads, never
    force-history flags. Validity masks may disable unavailable contact data.

    All onset pairs are same-body, time-window and normal compatible. A pair's
    time error and Euclidean spatial error are reported separately. Missing
    spatial geometry does not become zero error. Initial or incomplete-window
    events are censored; a not-yet-reached reference event is never a miss.
    """
    scalars = (reference_fps, control_dt_s, onset_tolerance_s, force_threshold_n)
    if not all(np.isfinite(v) and v > 0 for v in scalars):
        raise ValueError("FPS, control dt, onset tolerance and force threshold must be positive finite")
    if not np.isfinite(start_phase) or not 0 <= start_phase < 1:
        raise ValueError("start_phase must lie in [0,1)")
    if not np.isfinite(normal_angle_tolerance_rad) or not 0 <= normal_angle_tolerance_rad <= math.pi:
        raise ValueError("normal angle tolerance must lie in [0,pi]")
    metadata = {} if metadata is None else dict(metadata)
    if metadata.get("sampling", "post_physics_pre_reset_control_endpoints") != "post_physics_pre_reset_control_endpoints":
        raise ValueError("Contact events require post-physics pre-reset control endpoints")
    for key, expected in (("control_dt_s", control_dt_s), ("start_phase", start_phase)):
        if key in metadata and (not np.isfinite(metadata[key]) or not np.isclose(metadata[key], expected, rtol=0, atol=1e-8)):
            raise ValueError(f"metadata {key} disagrees with arguments")
    phase_raw = np.asarray(phase)
    times, phase = np.asarray(times_s, dtype=float), np.asarray(phase_raw, dtype=float)
    if (times.ndim != 1 or not len(times) or phase.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(phase).all()):
        raise ValueError("time and phase must be nonempty finite vectors of equal size")
    if not np.allclose(times, (np.arange(len(times)) + 1) * control_dt_s, rtol=0, atol=max(1e-7, control_dt_s*1e-4)):
        raise ValueError("Physical endpoint times must advance exactly once per control dt from reset")
    ref_flags = np.asarray(reference_flags)
    if ref_flags.ndim != 2 or len(ref_flags) < 2 or not ref_flags.shape[1]:
        raise ValueError("Reference flags need the full clip with at least two frames and one body")
    frames, bodies = ref_flags.shape
    ref_flags = _binary(ref_flags, (frames, bodies), "reference flags")
    duration = (frames - 1) / reference_fps
    if "motion_duration_s" in metadata and (not np.isfinite(metadata["motion_duration_s"])
            or not np.isclose(metadata["motion_duration_s"], duration, rtol=0, atol=1e-7)):
        raise ValueError("metadata motion duration disagrees with the complete reference")
    reference_time = start_phase * duration + times
    expected_phase = np.minimum(reference_time / duration, 1.)
    if (np.any(phase < 0) or np.any(phase > 1. + 1e-6)
            or np.any(np.diff(phase) < -1e-7)
            or not np.allclose(phase, expected_phase, rtol=0, atol=1e-5)):
        raise ValueError("Reference phase must follow physical time, clamping only at one")
    repeated = np.diff(phase) <= 0
    if np.any(repeated & (phase[1:] < 1. - 1e-6)):
        raise ValueError("Repeated movement phases are invalid")
    physical_hold = reference_time > duration + max(1e-7, control_dt_s*1e-4)
    if physical_hold.any() and metadata.get("post_reference_hold_is_physical") is not True:
        raise ValueError("Post-reference samples require an explicit physical-hold declaration")
    load = np.asarray(normal_load_n, dtype=float)
    if load.shape != (len(times), bodies) or not np.isfinite(load).all() or np.any(load < 0):
        raise ValueError("Normal loads must be finite nonnegative current endpoint N x B values")
    ap, an, apv, anv = _geometry(actual_points_w, actual_normals_w, load.shape, actual_valid, "actual")
    rp, rn, rpv, rnv = _geometry(reference_points_w, reference_normals_w, ref_flags.shape, reference_valid, "reference")
    names = list(body_names) if body_names is not None else [f"body_{b}" for b in range(bodies)]
    if len(names) != bodies or len(set(names)) != bodies or not all(isinstance(n, str) and n for n in names):
        raise ValueError("body_names must contain one unique nonempty name per body")
    window_start, window_end = start_phase * duration, float(reference_time[-1])
    minimum_dot = math.cos(normal_angle_tolerance_rad)
    active = load > force_threshold_n
    # Preserve stored float32 phase arithmetic at label-frame boundaries,
    # matching the source tracker rather than introducing float64 floor drift.
    indices = np.clip((phase_raw*(frames-1)).astype(np.int64), 0, frames-1)
    observed_reference_flags = ref_flags[indices]
    result, errors, timing = {}, [], []
    totals = dict(reference_events=0, reference_onsets_observed=0, reference_onsets_fully_observed=0,
                  reference_events_not_reached=0, reference_events_before_reset=0,
                  reference_onsets_censored=0, actual_events=0, actual_onsets_observed=0,
                  actual_onsets_censored=0, matched_onsets=0, spatial_error_events=0,
                  reference_onsets_missed_with_complete_window=0, unmatched_actual_onsets=0,
                  reference_onsets_normal_incompatible=0, reference_onsets_geometry_unavailable=0,
                  reference_onsets_without_unique_match=0,
                  full_reference_contact_duration_s=0., observed_reference_contact_duration_s=0.)
    for b, name in enumerate(names):
        references, actuals = [], []
        for start, stop in _runs(ref_flags[:, b]):
            begin, end = float(start/reference_fps), float(min(stop/reference_fps, duration))
            status = ("before_reset" if end <= window_start and start != frames-1 else
                      "not_reached" if begin > window_end + 1e-8 else "overlaps_record")
            censored = start == 0 or begin <= window_start + 1e-8
            observed = status == "overlaps_record" and not censored
            complete_window = (observed and begin-onset_tolerance_s >= window_start-1e-8
                               and begin+onset_tolerance_s <= window_end+1e-8)
            references.append(dict(start_frame=int(start), stop_frame_exclusive=int(stop),
                start_s=begin, end_s=end, observation_status=status, onset_censored=bool(censored),
                onset_observed=bool(observed), complete_matching_window=bool(complete_window),
                point=rp[start, b], normal=rn[start, b], point_valid=bool(rpv[start, b]),
                normal_valid=bool(rnv[start, b]), matched_actual_event=None))
        for start, stop in _runs(active[:, b]):
            actuals.append(dict(first_sample=int(start), stop_sample_exclusive=int(stop),
                start_s=float(reference_time[start]), elapsed_onset_s=float(times[start]),
                onset_censored=bool(start == 0), point=ap[start, b], normal=an[start, b],
                point_valid=bool(apv[start, b]), normal_valid=bool(anv[start, b]),
                matched_reference_event=None))
        ref_candidates = [(i, e) for i, e in enumerate(references) if e["onset_observed"]]
        actual_candidates = [(i, e) for i, e in enumerate(actuals) if not e["onset_censored"]]
        for r, a in _match([e for _, e in ref_candidates], [e for _, e in actual_candidates],
                           onset_tolerance_s, minimum_dot):
            ri, ref = ref_candidates[r]
            ai, act = actual_candidates[a]
            ref["matched_actual_event"], act["matched_reference_event"] = ai, ri
            delta = float(act["start_s"]-ref["start_s"])
            distance = float(np.linalg.norm(act["point"]-ref["point"])) if act["point_valid"] and ref["point_valid"] else None
            ref.update(onset_error_s=delta, surface_placement_error_m=distance,
                       normal_alignment_cosine=float(np.dot(act["normal"], ref["normal"])))
            timing.append(delta)
            if distance is not None:
                errors.append(distance)
        counts = {key: 0 for key in totals}
        counts.update(reference_events=len(references), actual_events=len(actuals),
            actual_onsets_observed=len(actual_candidates), actual_onsets_censored=len(actuals)-len(actual_candidates),
            unmatched_actual_onsets=sum(e["matched_reference_event"] is None for _, e in actual_candidates))
        for event in references:
            matched = event["matched_actual_event"] is not None
            nearby = [e for _, e in actual_candidates
                      if abs(e["start_s"]-event["start_s"]) <= onset_tolerance_s + 1e-9]
            normal_compatible = [e for e in nearby if e["normal_valid"] and event["normal_valid"]
                and np.dot(e["normal"], event["normal"]) >= minimum_dot-1e-12]
            event["detection_status"] = ("matched" if matched else
                "not_evaluated" if event["observation_status"] in ("not_reached", "before_reset") else
                "onset_left_censored" if event["onset_censored"] else
                "matching_window_censored" if not event["complete_matching_window"] else
                "reference_normal_unavailable" if not event["normal_valid"] else "missed_observed_onset")
            if event["detection_status"] == "missed_observed_onset" and nearby:
                event["detection_status"] = ("onset_without_unique_match" if normal_compatible else
                    "actual_normal_unavailable" if any(not e["normal_valid"] for e in nearby) else
                    "normal_incompatible_contact_onset")
            counts["reference_onsets_observed"] += event["onset_observed"]
            counts["reference_onsets_fully_observed"] += event["complete_matching_window"]
            counts["reference_events_not_reached"] += event["observation_status"] == "not_reached"
            counts["reference_events_before_reset"] += event["observation_status"] == "before_reset"
            counts["reference_onsets_censored"] += event["detection_status"] in ("onset_left_censored", "matching_window_censored")
            counts["matched_onsets"] += matched
            counts["spatial_error_events"] += matched and event.get("surface_placement_error_m") is not None
            counts["reference_onsets_missed_with_complete_window"] += event["detection_status"] == "missed_observed_onset"
            counts["reference_onsets_normal_incompatible"] += event["detection_status"] == "normal_incompatible_contact_onset"
            counts["reference_onsets_geometry_unavailable"] += event["detection_status"] in ("reference_normal_unavailable", "actual_normal_unavailable")
            counts["reference_onsets_without_unique_match"] += event["detection_status"] == "onset_without_unique_match"
            counts["full_reference_contact_duration_s"] += event["end_s"]-event["start_s"]
            counts["observed_reference_contact_duration_s"] += max(0., min(event["end_s"], window_end)-max(event["start_s"], window_start))
        for key, value in counts.items():
            totals[key] += value
        # Preserve JSON-safe masks and unavailable values instead of exposing NaN.
        for event in references + actuals:
            point, normal = event.pop("point"), event.pop("normal")
            event["point_w_m"] = point.tolist() if event["point_valid"] else None
            event["surface_normal_w"] = normal.tolist() if event["normal_valid"] else None
        result[name] = dict(counts=counts,
            endpoint_contact_presence=_presence(observed_reference_flags[:, b], active[:, b]),
            reference_events=references, actual_events=actuals)
    coverage = totals["observed_reference_contact_duration_s"]/totals["full_reference_contact_duration_s"] if totals["full_reference_contact_duration_s"] else None
    return dict(schema=1, metric_kind="engineering_reference_surface_onset_proxy",
        paper_fig7_equivalent=False,
        definitions=dict(spatial="Euclidean surface-point distance at same-body time/normal-matched actual and reference onsets",
            timing="actual onset physical time minus reference onset time; positive means late",
            difference_from_paper="Reference event surface targets replace commanded contact plans; no claim of identical ACP Fig7 metric",
            load="current positive normal load, not force-history occupancy or an FK estimate",
            presence="Current positive-load flags versus geometric reference labels only at observed endpoints; no unexecuted suffix negatives",
            segment_filter="Contiguous positive-load runs; no minimum-duration filter is applied by this engineering proxy",
            coverage="Fraction of full reference contact duration lying in the physically observed time window, not fraction successfully realized",
            censored="Reset contacts and incomplete matching windows do not become observed missed onsets",
            hold="Clamped reference phase; physical endpoint time must continue at control_dt",
            confidence="Single-rollout descriptive metrics; no paper benchmark confidence interval"),
        parameters=dict(control_dt_s=float(control_dt_s), reference_fps=float(reference_fps),
            motion_duration_s=float(duration), start_phase=float(start_phase),
            onset_tolerance_s=float(onset_tolerance_s), normal_angle_tolerance_rad=float(normal_angle_tolerance_rad),
            force_threshold_n=float(force_threshold_n)),
        sampling=dict(samples=len(times), physical_elapsed_s=float(times[-1]),
            movement_samples=int((~physical_hold).sum()), physical_hold_samples=int(physical_hold.sum()),
            physical_hold_declared=metadata.get("post_reference_hold_is_physical") is True,
            record_semantics_validated=True, physical_authenticity_proven_by_this_report=False),
        summary=totals | dict(reference_contact_duration_coverage_fraction=coverage,
            endpoint_contact_presence=_presence(observed_reference_flags, active),
            spatial_error_available=bool(errors), mean_surface_placement_error_m=float(np.mean(errors)) if errors else None,
            max_surface_placement_error_m=float(np.max(errors)) if errors else None,
            mean_absolute_onset_error_s=float(np.mean(np.abs(timing))) if timing else None,
            mean_signed_onset_error_s=float(np.mean(timing)) if timing else None), bodies=result)


def commanded_landing_report(times_s, commanded_points_w, command_valid,
                              actual_body_origins_w, normal_load_n, *,
                              control_dt_s, body_names=None, force_threshold_n=1.,
                              actual_valid=None, metadata=None):
    """Euclidean command-to-realized body-origin error at physical load onsets.

    The command for endpoint i must be the exact policy input captured at the
    START of control i, reconstructed with that start state's root pose. Actual
    body origins are post-physics, pre-reset endpoints. All positions must use
    one shared frame. A scene surface point or weighted measured contact point
    is a different anchor and must not replace a body origin in this metric.

    The geometric operator matches the paper's commanded-versus-realized onset
    error definition. This local body-origin/normal-load convention and a single
    teacher rollout do not establish equivalence to the authors' implementation
    or completion of their episode/seed/confidence-interval benchmark protocol.
    """
    if (not np.isfinite(control_dt_s) or control_dt_s <= 0
            or not np.isfinite(force_threshold_n) or force_threshold_n <= 0):
        raise ValueError("Control dt and current normal-force threshold must be positive finite")
    metadata = {} if metadata is None else dict(metadata)
    if metadata.get("command_sampling") != "control_start_policy_input":
        raise ValueError("Command anchors must be declared control_start_policy_input")
    if metadata.get("command_anchor") != "body_origin":
        raise ValueError("Command and realized anchors must both be declared body_origin")
    if metadata.get("actual_sampling", "post_physics_pre_reset_control_endpoints") != "post_physics_pre_reset_control_endpoints":
        raise ValueError("Actual body origins must be post-physics pre-reset endpoints")
    if "control_dt_s" in metadata and (not np.isfinite(metadata["control_dt_s"])
            or not np.isclose(metadata["control_dt_s"], control_dt_s, rtol=0, atol=1e-8)):
        raise ValueError("Metadata and actual control dt disagree")
    times = np.asarray(times_s, dtype=float)
    commands, actual = np.asarray(commanded_points_w, dtype=float), np.asarray(actual_body_origins_w, dtype=float)
    load = np.asarray(normal_load_n, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or not np.allclose(times, (np.arange(len(times))+1)*control_dt_s,
                               rtol=0, atol=max(1e-7, control_dt_s*1e-4))):
        raise ValueError("Physical endpoint times must advance once per control dt from reset")
    if load.ndim != 2 or load.shape[0] != len(times) or not load.shape[1]:
        raise ValueError("Normal load must be a nonempty [endpoint,body] array")
    if commands.shape != load.shape+(3,) or actual.shape != commands.shape:
        raise ValueError("Commands and actual body origins must both have shape [endpoint,body,3]")
    if not np.isfinite(load).all() or np.any(load < 0):
        raise ValueError("Current normal loads must be finite and nonnegative")
    valid_command = _binary(command_valid, load.shape, "command validity")
    valid_actual = np.ones(load.shape, dtype=bool) if actual_valid is None else _binary(actual_valid, load.shape, "actual origin validity")
    if not np.isfinite(commands[valid_command]).all():
        raise ValueError("Valid policy command anchors must be finite; invalid policy output cannot enter contact error")
    if not np.isfinite(actual[valid_actual]).all():
        raise ValueError("Valid realized body origins must be finite")
    bodies = load.shape[1]
    names = list(body_names) if body_names is not None else [f"body_{b}" for b in range(bodies)]
    if len(names) != bodies or len(set(names)) != bodies or not all(isinstance(n, str) and n for n in names):
        raise ValueError("body_names must contain unique nonempty names in the exact array order")
    active = load > force_threshold_n
    totals = dict(actual_contact_events=0, initial_censored_events=0,
        observed_contact_onsets=0, onsets_with_command=0, onsets_without_command=0,
        onsets_without_actual_origin=0, valid_landing_error_events=0)
    per_body, all_errors = {}, []
    for b, name in enumerate(names):
        counts, events, errors = dict.fromkeys(totals, 0), [], []
        for start, stop in _runs(active[:, b]):
            start, stop = int(start), int(stop)
            initial = start == 0
            command_ok, actual_ok = bool(valid_command[start, b]), bool(valid_actual[start, b])
            eligible = not initial and command_ok and actual_ok
            distance = float(np.linalg.norm(commands[start, b]-actual[start, b])) if eligible else None
            counts["actual_contact_events"] += 1
            counts["initial_censored_events"] += initial
            counts["observed_contact_onsets"] += not initial
            counts["onsets_with_command"] += not initial and command_ok
            counts["onsets_without_command"] += not initial and not command_ok
            counts["onsets_without_actual_origin"] += not initial and not actual_ok
            counts["valid_landing_error_events"] += eligible
            if distance is not None:
                errors.append(distance)
            events.append(dict(first_sample=start, stop_sample_exclusive=stop,
                endpoint_time_s=float(times[start]), command_time_s=float(times[start]-control_dt_s),
                current_normal_load_n=float(load[start, b]), initial_onset_censored=initial,
                command_valid=command_ok, actual_origin_valid=actual_ok,
                status=("initial_onset_censored" if initial else "no_command" if not command_ok else
                        "actual_origin_unavailable" if not actual_ok else "valid_commanded_landing"),
                commanded_body_origin_w_m=commands[start, b].tolist() if command_ok else None,
                actual_body_origin_w_m=actual[start, b].tolist() if actual_ok else None,
                landing_error_m=distance))
        for key, value in counts.items():
            totals[key] += value
        all_errors.extend(errors)
        per_body[name] = dict(summary=counts | _landing_error_summary(errors), events=events)
    return dict(schema=1, kind="commanded_body_origin_onset_error",
        paper_metric_definition_compatible=True, paper_author_implementation_verified=False,
        paper_benchmark_complete=False,
        metadata=dict(kind="commanded_body_origin_onset_error", command_anchor="body_origin",
            actual_anchor="body_origin", command_sampling="control_start_policy_input",
            actual_sampling="post_physics_pre_reset_control_endpoints",
            positions_frame=metadata.get("positions_frame", "unspecified_shared_frame"),
            control_dt_s=float(control_dt_s), force_threshold_n=float(force_threshold_n)),
        definitions=dict(error="Euclidean distance in metres between actual policy-input command anchor and realized same-body origin at measured load onset",
            onset="First current-positive-normal-load endpoint after a noncontact endpoint; initial already-positive contact is censored",
            command_time="Command captured before its control step, using that start state's root pose; endpoint pose must not reconstruct an earlier command",
            missing="An onset without a command or valid realized body origin contributes no error value and is counted separately",
            minimum_duration="No contact-segment duration filter is applied",
            scope="One teacher rollout with an explicit local body-origin/normal-load convention; does not establish the paper's 3000-episode/seed benchmark or full ACP reproduction"),
        sampling=dict(samples=len(times), physical_elapsed_s=float(times[-1]),
            record_semantics_validated=True, physical_authenticity_proven_by_this_report=False),
        summary=totals | _landing_error_summary(all_errors), bodies=per_body)


def _landing_error_summary(errors):
    return dict(error_available=bool(errors),
        mean_landing_error_m=float(np.mean(errors)) if errors else None,
        rmse_landing_error_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        max_landing_error_m=float(np.max(errors)) if errors else None)
