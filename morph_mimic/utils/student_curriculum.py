"""Explicit observation and actuator conversions for finite ACP students.

The privileged teacher predicts offsets about the post-advance reference pose;
the student predicts offsets about the robot's fixed nominal pose. These helpers
keep those controller meanings separate without changing either network.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


def _real(name: str, value: Real, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if result < 0 or (positive and result == 0):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {requirement}")
    return result


def _tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != shape or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor with shape {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite values")


def _same_storage_contract(reference: torch.Tensor, *values: torch.Tensor) -> None:
    if any(value.device != reference.device or value.dtype != reference.dtype for value in values):
        raise ValueError("Tensors must have the same device and dtype")


def _action_inputs(reference_q_post, nominal_q, action):
    if not isinstance(reference_q_post, torch.Tensor) or reference_q_post.ndim != 2:
        raise ValueError("reference_q_post must be a floating [N,23] tensor")
    shape = (reference_q_post.shape[0], 23)
    _tensor("reference_q_post", reference_q_post, shape)
    _tensor("action", action, shape)
    if not isinstance(nominal_q, torch.Tensor) or nominal_q.shape not in ((23,), shape):
        raise ValueError("nominal_q must have shape [23] or [N,23]")
    _tensor("nominal_q", nominal_q, tuple(nominal_q.shape))
    _same_storage_contract(reference_q_post, nominal_q, action)


def teacher_to_student_action(reference_q_post, nominal_q, teacher_action, *,
                              scale=.25, teacher_clip=20., student_clip=20.):
    """Convert a teacher command while preserving the executed PD joint target.

    ``reference_q_post`` must be sampled after the environment advances phase.
    Teacher clipping matches its actual actuator path. Student clipping would
    change that target, so an unrepresentable command fails instead of clipping.
    """
    scale = _real("scale", scale, positive=True)
    teacher_clip = _real("teacher_clip", teacher_clip, positive=True)
    student_clip = _real("student_clip", student_clip, positive=True)
    _action_inputs(reference_q_post, nominal_q, teacher_action)
    result = (reference_q_post - nominal_q) / scale + teacher_action.clamp(-teacher_clip, teacher_clip)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Converted student action must contain finite values")
    if bool((result.abs() > student_clip).any()):
        raise ValueError("Teacher target is outside student_clip; clipping would change the PD joint target")
    return result


def student_to_teacher_action(student_action, reference_q_post, nominal_q, *, scale=.25):
    """Express an actually executed student command as a teacher residual.

    No clipping is applied: this also represents off-policy student actions in
    the teacher's history, even when their equivalent residual exceeds its bound.
    """
    scale = _real("scale", scale, positive=True)
    _action_inputs(reference_q_post, nominal_q, student_action)
    result = student_action + (nominal_q - reference_q_post) / scale
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Converted teacher residual must contain finite values")
    return result


def teacher_observation(student_obs, reference_features, residual_history):
    """Build the teacher's 584 features without leaking reference into student.

    The first 375 student features contain five 75-feature proprioceptive frames.
    Replace only each frame's last 23 action values with its matching historical
    residual, leaving physical state, scan, contacts and goal features unchanged.
    All inputs remain unmodified.
    """
    if not isinstance(student_obs, torch.Tensor) or student_obs.ndim != 2:
        raise ValueError("student_obs must be a floating [N,468] tensor")
    count = student_obs.shape[0]
    _tensor("student_obs", student_obs, (count, 468))
    _tensor("reference_features", reference_features, (count, 116))
    _tensor("residual_history", residual_history, (count, 5, 23))
    _same_storage_contract(student_obs, reference_features, residual_history)
    prefix = student_obs.clone()
    prefix[:, :375].reshape(count, 5, 75)[:, :, -23:] = residual_history
    return torch.cat((prefix, reference_features), dim=-1)


def beta_schedule(iteration, teacher_only_iterations, anneal_iterations):
    """Teacher execution probability: warm start at one, then decay to zero."""
    iteration = _real("iteration", iteration)
    teacher_only_iterations = _real("teacher_only_iterations", teacher_only_iterations)
    anneal_iterations = _real("anneal_iterations", anneal_iterations, positive=True)
    return max(0., min(1., 1. - (iteration - teacher_only_iterations) / anneal_iterations))
