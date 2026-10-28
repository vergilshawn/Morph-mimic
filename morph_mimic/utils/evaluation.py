"""Reference-tracking metrics independent of the simulator and training rewards.

Errors are sampled at control-step endpoints, including the terminal state
captured by the environment before its automatic reset. Position metrics use
Euclidean distance, rather than a per-coordinate RMS.
"""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


class ReferenceTrackingMetrics:
    """Accumulate one reference episode from reset-before-terminal snapshots."""

    def __init__(self):
        self.samples = 0
        self._sums = {}
        self._shapes = None
        self.root_position_max_error_m = 0.0

    def update(self, snapshot: dict, env_id: int = 0):
        fields = (
            "root_squared_error_m2", "joint_squared_error_rad2",
            "body_squared_error_m2", "root_orientation_squared_error_rad2",
        )
        values = {}
        for key in fields:
            value = torch.as_tensor(snapshot[key]).detach()[env_id].to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{key} must contain finite nonnegative squared errors")
            values[key] = value
        shapes = {key: tuple(value.shape) for key, value in values.items()}
        if shapes[fields[0]] != () or shapes[fields[3]] != ():
            raise ValueError("Root position and orientation errors must be scalar per environment")
        if values[fields[1]].ndim != 1 or values[fields[1]].numel() == 0:
            raise ValueError("Joint errors must have a nonempty joint axis")
        if shapes[fields[2]] != (9,):
            raise ValueError("Body errors must describe the nine ordered contact bodies")
        if self._shapes is not None and shapes != self._shapes:
            raise ValueError("Tracking-point or joint dimensions changed within the episode")
        self._shapes = shapes
        for key, value in values.items():
            self._sums[key] = self._sums.get(key, torch.zeros_like(value)) + value
        self.samples += 1
        self.root_position_max_error_m = max(
            self.root_position_max_error_m, math.sqrt(float(values[fields[0]]))
        )

    def summary(self):
        if not self.samples:
            raise ValueError("No physical tracking samples were collected")
        mean = {key: value / self.samples for key, value in self._sums.items()}
        return dict(
            tracking_samples=self.samples,
            root_position_rmse_m=math.sqrt(float(mean["root_squared_error_m2"])),
            root_position_max_error_m=self.root_position_max_error_m,
            joint_rmse_rad=math.sqrt(float(mean["joint_squared_error_rad2"].mean())),
            tracked_body_position_rmse_m=math.sqrt(float(mean["body_squared_error_m2"].mean())),
            tracked_body_position_rmse_per_body_m=mean["body_squared_error_m2"].sqrt().tolist(),
            root_orientation_rmse_rad=math.sqrt(float(mean["root_orientation_squared_error_rad2"])),
        )


def reference_outcome(*, terminated: bool, truncated: bool, reached_end: bool,
                      phase: float, interrupted: bool = False, start_phase: float = 0.0):
    """Distinguish completing a reference, physical failure and limited budgets.

    A fall on the final frame is a failure. An external step limit and a time
    limit are both incomplete evaluations; neither becomes a completed clip.
    """
    if not math.isfinite(phase) or phase < 0:
        raise ValueError("Reference phase must be finite and nonnegative")
    if not math.isfinite(start_phase) or not 0 <= start_phase < 1:
        raise ValueError("Start phase must lie in [0, 1)")
    segment_completed = bool(reached_end and phase >= 1.0 - 1e-5 and not terminated)
    partial = start_phase != 0.0
    completed = segment_completed and not partial
    failure = bool(terminated)
    budget_limited = not segment_completed and not failure
    return dict(
        completed=completed,
        segment_completed=segment_completed,
        evaluation_is_partial=partial,
        failure=failure,
        budget_limited=budget_limited,
        interrupted=bool(interrupted),
        terminated=bool(terminated),
        truncated=bool(truncated),
        end_phase=float(phase),
    )


class TrackingTrajectory:
    """Store physical endpoint snapshots without reading an auto-reset robot.

    ``computed_torque`` and ``applied_torque`` are Isaac actuator estimates,
    rather than joint force-sensor measurements. Their difference diagnoses
    actuator clipping; it does not measure all PhysX constraint impulses.
    """

    _fields = {
        "root_pos_w": (3,), "ref_root_pos_w": (3,),
        "root_quat_w": (4,), "ref_root_quat_w": (4,),
        "joint_pos": (23,), "joint_ref_pos": (23,), "joint_target_pos": (23,),
        "joint_vel": (23,), "joint_ref_vel": (23,),
        "applied_torque": (23,), "computed_torque": (23,),
        "measured_contact_flags": (9,), "ref_contact_flags": (9,),
        "contact_force_w": (9, 3), "phase": (), "clip_id": (),
        "terminated": (), "truncated": (), "reached_end": (),
        "died": (), "time_out": (), "grace_active": (),
        "termination_height_threshold_m": (),
        "reference_error_exceeded": (), "termination_root_error_threshold_m": (),
    }

    def __init__(self, control_dt: float, *, torque_clip_tolerance_nm: float = 1e-3):
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("Control dt must be positive and finite")
        if not math.isfinite(torque_clip_tolerance_nm) or torque_clip_tolerance_nm < 0:
            raise ValueError("Torque clipping tolerance must be finite and nonnegative")
        self.control_dt = control_dt
        self.torque_clip_tolerance_nm = torque_clip_tolerance_nm
        self._rows = []

    def update(self, snapshot: dict, *, action, reward, env_id: int = 0):
        missing = set(self._fields) - set(snapshot)
        if missing:
            raise ValueError(f"Missing pre-reset diagnostic fields: {', '.join(sorted(missing))}")
        row = {}
        for key, shape in self._fields.items():
            value = torch.as_tensor(snapshot[key]).detach()[env_id].cpu().numpy().copy()
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key}: expected finite endpoint data with shape {shape}, got {value.shape}")
            row[key] = value
        for key, value, shape in (("policy_action", action, (23,)), ("reward", reward, ())):
            value = torch.as_tensor(value).detach()[env_id].cpu().numpy().copy()
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key}: expected finite data with shape {shape}")
            row[key] = value
        if "joint_effort_limit" in snapshot:
            value = torch.as_tensor(snapshot["joint_effort_limit"]).detach()[env_id].cpu().numpy().copy()
            if value.shape != (23,) or not np.isfinite(value).all() or (value <= 0).any():
                raise ValueError("joint_effort_limit must contain 23 finite positive values")
            row["joint_effort_limit"] = value
        for key in ("all_joint_pos", "held_joint_position_error_rad"):
            if key in snapshot:
                value = torch.as_tensor(snapshot[key]).detach()[env_id].cpu().numpy().copy()
                if value.ndim != 1 or not np.isfinite(value).all():
                    raise ValueError(f"{key}: expected finite named joint-axis data")
                row[key] = value
        if self._rows and (row.keys() != self._rows[0].keys() or row["phase"] < self._rows[-1]["phase"]):
            raise ValueError("Diagnostic fields changed or reference phase decreased within an episode")
        self._rows.append(row)

    def arrays(self):
        if not self._rows:
            raise ValueError("No physical trajectory samples were collected")
        arrays = {key: np.stack([row[key] for row in self._rows]) for key in self._rows[0]}
        arrays["elapsed_control_time_s"] = (np.arange(len(self._rows)) + 1) * self.control_dt
        arrays["estimated_actuator_clipped"] = (
            np.abs(arrays["computed_torque"] - arrays["applied_torque"]) > self.torque_clip_tolerance_nm
        )
        return arrays

    def summary(self):
        a = self.arrays()
        delta = a["computed_torque"] - a["applied_torque"]
        joint_error = a["joint_pos"] - a["joint_ref_pos"]
        target_error = a["joint_target_pos"] - a["joint_pos"]
        mismatch = (a["measured_contact_flags"] > 0.5) != (a["ref_contact_flags"] > 0.5)
        last = self._rows[-1]
        reason = ("root_tracking_error_threshold" if bool(last["reference_error_exceeded"]) else
                  "root_below_height_threshold" if bool(last["died"]) else
                  "physical_termination_unspecified" if bool(last["terminated"]) else
                  "reference_end" if bool(last["reached_end"]) else
                  "episode_time_limit" if bool(last["time_out"]) else
                  "external_budget_or_interruption")
        result = dict(
            samples=len(self._rows), failure_reason=reason,
            measured_contact_definition="Environment force-history threshold; bound scenes use terrain-filtered sensors",
            torque_definition="Isaac implicit actuator estimated torques, not joint torque sensor measurements",
            torque_clip_tolerance_nm=self.torque_clip_tolerance_nm,
            estimated_actuator_clipped_step_fraction=float(a["estimated_actuator_clipped"].any(axis=1).mean()),
            estimated_actuator_clipped_fraction_per_joint=a["estimated_actuator_clipped"].mean(axis=0).tolist(),
            max_estimated_torque_clip_nm=float(np.abs(delta).max()),
            max_applied_torque_nm=float(np.abs(a["applied_torque"]).max()),
            joint_rmse_per_joint_rad=np.sqrt(np.mean(joint_error ** 2, axis=0)).tolist(),
            target_tracking_rmse_per_joint_rad=np.sqrt(np.mean(target_error ** 2, axis=0)).tolist(),
            contact_mismatch_fraction_per_body=mismatch.mean(axis=0).tolist(),
            endpoint_root_pos_w=last["root_pos_w"].tolist(),
            endpoint_ref_root_pos_w=last["ref_root_pos_w"].tolist(),
            endpoint_height_threshold_m=float(last["termination_height_threshold_m"]),
            endpoint_root_error_threshold_m=float(last["termination_root_error_threshold_m"]),
            endpoint_grace_active=bool(last["grace_active"]),
            termination_threshold_provenance="Local engineering safeguards, not claimed paper evaluation thresholds",
        )
        if "held_joint_position_error_rad" in a and a["held_joint_position_error_rad"].shape[1]:
            result["held_joint_position_error_max_rad"] = float(np.abs(a["held_joint_position_error_rad"]).max())
        return result

    def write(self, path, *, metadata: dict):
        """Atomically publish a pickle-free NPZ and refuse existing artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
