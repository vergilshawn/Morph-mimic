"""Audit explicit box completion from recorded full-joint states/current forces.

MuJoCo is used only for source-model forward kinematics. This command never
steps dynamics, reruns a policy, invents support forces, or extends a short
recording. A missing one-second physical hold is explicitly ineligible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from morph_mimic.utils.box_task import (
    CONTACT_BODY_NAMES, FOOT_BODY_NAMES, PROJECT_ROOT, _sha256, load_box_task_spec, task_metrics,
)


def source_foot_kinematics(root_pos, root_quat, all_joint_pos, all_joint_names, robot_xml):
    """Source MJCF FK of every recorded joint, with no nominal reconstruction."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    expected = [model.joint(i).name for i in range(model.njnt) if i not in free]
    if (len(free) != 1 or model.jnt_qposadr[free[0]] != 0
            or len(all_joint_names) != 29 or len(set(all_joint_names)) != 29 or set(all_joint_names) != set(expected)):
        raise ValueError("box task audit requires all 29 distinct source G1 joints and one leading free root")
    if any(model.joint(name).type[0] != mujoco.mjtJoint.mjJNT_HINGE for name in all_joint_names):
        raise ValueError("box task source has unsupported non-scalar joints")
    qids = np.array([model.joint(name).qposadr[0] for name in all_joint_names])
    feet = np.array([model.body(name).id for name in FOOT_BODY_NAMES])
    data = mujoco.MjData(model)
    positions, orientations = [], []
    for pos, quat, joints in zip(root_pos, root_quat, all_joint_pos):
        data.qpos[:3], data.qpos[3:7], data.qpos[qids] = pos, quat, joints
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[feet].copy())
        orientations.append(data.xquat[feet].copy())
    return np.asarray(positions), np.asarray(orientations)


def summarize_box_hold(metrics, elapsed_times_s, phase, *, duration_s, start_phase, control_dt_s):
    """Require the final 1s to be recorded *after* the reference endpoint.

    Merely including the endpoint frame at t=duration is not a hold. Every
    control endpoint after that frame must satisfy every gate, continuously.
    This also rejects gaps, rewound phases, and partial initialization.
    """
    times, phase = np.asarray(elapsed_times_s, dtype=float), np.asarray(phase, dtype=float)
    if (times.ndim != 1 or phase.shape != times.shape or len(times) < 1
            or not np.isfinite(times).all() or not np.isfinite(phase).all()
            or not np.isfinite([duration_s, start_phase, control_dt_s]).all()
            or duration_s <= 0 or control_dt_s <= 0 or not 0 <= start_phase < 1):
        raise ValueError("hold audit needs finite aligned times/phases and positive duration/control interval")
    if np.any(np.diff(times) <= 0) or not np.allclose(np.diff(times), control_dt_s, rtol=0, atol=1e-6):
        raise ValueError("hold audit requires uninterrupted control endpoints, without resets or missing timestamps")
    if not np.isclose(times[0], control_dt_s, rtol=0, atol=1e-6):
        raise ValueError("hold audit needs the first post-reset control endpoint, not only a saved tail")
    if np.any(np.diff(phase) < -1e-6) or np.any(phase < start_phase-1e-6) or np.any(phase > 1+1e-6):
        raise ValueError("hold audit phases rewind or leave the selected reference interval")
    if not np.allclose(phase, np.minimum(1., start_phase+times/duration_s), rtol=0, atol=2e-5):
        raise ValueError("hold audit reference clock differs from elapsed physical control time")
    required_s = 1.
    samples_float = required_s/control_dt_s
    samples = round(samples_float)
    if not np.isclose(samples_float, samples, rtol=0, atol=1e-6):
        raise ValueError("one-second hold must align with the control timestep")
    end_time = (1-start_phase)*duration_s
    hold_mask = (times > end_time + 1e-6) & (phase >= 1-1e-6)
    hold_indices = np.flatnonzero(hold_mask)
    enough = bool(len(hold_indices) >= samples and times[-1] >= end_time+required_s-1e-6
        and np.all(hold_mask[-samples:]))
    final_window = np.arange(len(times)-samples, len(times)) if enough else hold_indices
    gates = {name: np.asarray(value, dtype=bool) for name, value in metrics.items() if name.endswith("_gate")}
    if "task_gate" not in gates or any(value.shape != times.shape for value in gates.values()):
        raise ValueError("hold audit needs one Boolean task/gate value per physical control endpoint")
    failures = []
    if start_phase != 0.:
        failures.append("partial_reference_initialization")
    if not enough:
        failures.append("missing_full_one_second_post_reference_physical_hold")
    if enough:
        failures.extend(name for name, values in gates.items()
                        if name != "task_gate" and not bool(values[final_window].all()))
    passed = enough and start_phase == 0. and bool(gates["task_gate"][final_window].all())
    if enough and not passed and not failures:
        failures.append("task_gate")
    return dict(required_hold_s=required_s, required_hold_control_endpoints=samples,
        recorded_post_reference_control_endpoints=int(len(hold_indices)),
        recorded_post_reference_hold_s=max(0., float(times[-1]-end_time)),
        uninterrupted_full_hold_recorded=enough, full_phase_zero=start_phase == 0.,
        hold_window_gate_pass_fraction={name: float(values[final_window].mean()) if len(final_window) else None
                                       for name, values in gates.items()},
        passed=passed, failure_reasons=failures)


def evaluate_box_task_trace(trajectory, motion, *, task_spec=None):
    """Return a JSON report and endpoint metric arrays for an existing trace."""
    import torch

    trajectory, motion = Path(trajectory).resolve(), Path(motion).resolve()
    spec = load_box_task_spec(motion) if task_spec is None else task_spec
    if spec["motion_sha256"] != _sha256(motion):
        raise ValueError("box task spec motion hash differs from requested motion")
    with np.load(trajectory, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays.get("metadata_json", "{}")))
    names = arrays.get("all_joint_names", metadata.get("all_joint_names"))
    if names is None or "all_joint_pos" not in arrays:
        raise ValueError("box task audit needs explicitly named all_joint_pos; held joints cannot be reconstructed")
    names = [str(value) for value in names]
    if metadata.get("contact_body_names") != list(CONTACT_BODY_NAMES):
        raise ValueError("trace must name the canonical nine current contact-force columns")
    result = metadata.get("result", {})
    source_digest = metadata.get("source_sha256", result.get("source_sha256"))
    if source_digest != spec["motion_sha256"]:
        raise ValueError("trace source motion hash is missing or differs from task spec")
    root = np.asarray(arrays["root_pos_w"], dtype=float)
    root_quat = np.asarray(arrays["root_quat_w"], dtype=float)
    reference_root = np.asarray(arrays["ref_root_pos_w"], dtype=float)
    joints = np.asarray(arrays["all_joint_pos"], dtype=float)
    force = np.asarray(arrays["contact_force_w"], dtype=float)
    phase = np.asarray(arrays["phase"], dtype=float)
    times = np.asarray(arrays["elapsed_control_time_s"], dtype=float)
    count = len(times)
    shapes = ((count, 3), (count, 4), (count, 3), (count, 29), (count, 9, 3), (count,))
    values = (root, root_quat, reference_root, joints, force, phase)
    if any(value.shape != shape or not np.isfinite(value).all() for value, shape in zip(values, shapes)):
        raise ValueError("trace full-joint/force/root/phase arrays have incompatible shapes or nonfinite values")
    if not np.allclose(np.linalg.norm(root_quat, axis=-1), 1, rtol=0, atol=1e-4):
        raise ValueError("trace root quaternions must be normalized WXYZ")
    with np.load(motion, allow_pickle=False) as archive:
        reference_positions = np.asarray(archive["root_pos"], dtype=float)
    reference_positions[:, :2] -= spec["source_first_root_xy"]
    query = np.clip(phase, 0, 1)*(len(reference_positions)-1)
    sampled = np.stack([np.interp(query, np.arange(len(reference_positions)), reference_positions[:, i])
                        for i in range(3)], axis=-1)
    origin = reference_root[0]-sampled[0]
    if not np.allclose(reference_root-sampled, origin, rtol=0, atol=2e-5):
        raise ValueError("trace reference does not have one fixed translation into the bound scene")
    # Convert the whole trace into the same rebased scene as the verified patch.
    rebased_root = root-origin
    feet, foot_quat = source_foot_kinematics(rebased_root, root_quat, joints, names, spec["robot_xml"])
    tensor_values = [torch.from_numpy(value) for value in (rebased_root, root_quat, feet, foot_quat, force)]
    metrics = {name: value.cpu().numpy() for name, value in task_metrics(*tensor_values, spec).items()}
    dt = float(metadata.get("control_dt_s", .02))
    start_phase = float(result.get("start_phase", arrays.get("initial_phase", 0.)))
    hold = summarize_box_hold(metrics, times, phase,
        duration_s=spec["final_reference"]["duration_s"], start_phase=start_phase, control_dt_s=dt)
    reasons = list(hold["failure_reasons"])
    if metadata.get("eligible_for_skill_acceptance") is False or metadata.get("diagnostic_only") is True:
        reasons.append("source_trace_is_diagnostic_only")
    if metadata.get("root_error_termination_enabled") is False:
        reasons.append("source_trace_disabled_root_tracking_termination")
    if not bool(result.get("contract_verified", metadata.get("contract_verified", False))):
        reasons.append("source_trace_contract_not_verified")
    if any(bool(result.get(key)) for key in ("failure", "terminated", "budget_limited", "interrupted")):
        reasons.append("source_trace_failed_or_was_incomplete")
    if not bool(result.get("completed")):
        reasons.append("source_trace_did_not_complete_full_reference")
    endpoint = {name: value[-1].tolist() for name, value in metrics.items()}
    endpoint_failures = [name for name, value in endpoint.items()
                         if name.endswith("_gate") and name not in ("task_gate", "task_geometry_gate") and value is False]
    reasons.extend(f"endpoint_{name}" for name in endpoint_failures)
    report = dict(schema=1, method="recorded_full29_source_model_fk_and_current_force_box_task_audit_v1",
        scope=spec["scope"], trajectory=str(trajectory), trajectory_sha256=_sha256(trajectory),
        task_spec=spec, physics_stepped_by_this_script=False, policy_executed_by_this_script=False,
        dynamics_validated_by_this_script=False, recorded_joint_count=29,
        held_joints_reconstructed=False, source_robot_geometry="source MJCF forward kinematics only",
        force_source="recorded current endpoint contact_force_w; no force-history flags or FK force inference",
        scene_origin_translation_removed_m=origin.tolist(), samples=count,
        final_recorded_elapsed_time_s=float(times[-1]), final_reference_phase=float(phase[-1]),
        endpoint=endpoint, endpoint_failure_gates=endpoint_failures,
        full_trace_task_gate_fraction=float(metrics["task_gate"].mean()),
        hold=hold, eligible_for_task_acceptance=not reasons, task_passed=not reasons,
        failure_reasons=reasons, monitored_force_limit=spec["unmonitored_contact_limit"])
    metrics.update(elapsed_control_time_s=times, phase=phase)
    return report, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or (args.metrics_out is not None and args.metrics_out.exists()):
        parser.error("refusing to overwrite an existing audit artifact")
    report, metrics = evaluate_box_task_trace(args.trajectory, args.motion)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("xb") as stream:
            np.savez_compressed(stream, **metrics, metadata_json=np.asarray(json.dumps(report, allow_nan=False)))
    print(json.dumps(dict(out=str(args.out), task_passed=report["task_passed"],
        failure_reasons=report["failure_reasons"], endpoint=report["endpoint"]), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
