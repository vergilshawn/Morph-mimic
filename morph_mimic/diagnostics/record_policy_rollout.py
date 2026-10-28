"""Record a deterministic Stage-1 evaluation with every physical joint.

Reuse the evaluator and capture the physical state before its automatic reset.
All instrumentation is confined to this process; training files are unchanged.
The output is suitable for render_tracking_trajectory's offline video renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from morph_mimic.scripts import evaluate
from morph_mimic.utils.evaluation import TrackingTrajectory
from morph_mimic.utils.specs import RobotSpec


def install_capture(env):
    state = {"all_joint_names": list(env.robot.data.joint_names)}
    original_dones = env._get_dones
    original_reset = env.reset

    def capture_dones():
        state["all_joint_pos"] = env.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        return original_dones()

    def capture_reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        state["initial_all_joint_pos"] = env.robot.data.joint_pos[0].detach().cpu().numpy().copy()
        state["initial_root_pos_w"] = env.robot.data.root_pos_w[0].detach().cpu().numpy().copy()
        state["initial_root_quat_w"] = env.robot.data.root_quat_w[0].detach().cpu().numpy().copy()
        state["initial_phase"] = np.asarray(float(env.phase[0]))
        return result

    env._get_dones = capture_dones
    env.reset = capture_reset
    return state


class FullStateTrajectory(TrackingTrajectory):
    def __init__(self, control_dt, *, capture):
        super().__init__(control_dt)
        self.capture = capture
        self.full_joints = []

    def update(self, snapshot, *, action, reward, env_id=0):
        if env_id != 0:
            raise ValueError("Full-state recording supports a single environment")
        full = np.asarray(self.capture["all_joint_pos"])
        if full.shape != (len(self.capture["all_joint_names"]),) or not np.isfinite(full).all():
            raise ValueError("Invalid pre-reset full-joint state")
        super().update(snapshot, action=action, reward=reward, env_id=env_id)
        self.full_joints.append(full.copy())

    def arrays(self):
        arrays = super().arrays()
        if len(self.full_joints) != len(self._rows):
            raise ValueError("Physical endpoint and full-joint sample counts differ")
        arrays["all_joint_pos"] = np.stack(self.full_joints)
        arrays["all_joint_names"] = np.asarray(self.capture["all_joint_names"])
        for key in ("initial_all_joint_pos", "initial_root_pos_w", "initial_root_quat_w", "initial_phase"):
            if key in self.capture:
                arrays[key] = np.asarray(self.capture[key])
        return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "mojia"), default="g1")
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--motion-dir", dest="motion_dir", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--start-phase", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.start_phase) or not 0 <= args.start_phase < 1:
        parser.error("--start-phase must be in [0, 1)")
    if args.out.exists() or args.trajectory.exists():
        parser.error("refusing to overwrite an existing report or trajectory")
    if args.trajectory.suffix != ".npz":
        parser.error("--trajectory must have an .npz suffix")
    args.motion_dir = args.motion_dir.resolve()
    args.stage, args.max_steps, args.headless = 1, None, True
    spec = RobotSpec.load(args.robot)
    metadata = evaluate._clip_metadata(args.motion_dir, [args.clip], spec)[0]
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    if checkpoint.get("robot") != args.robot or checkpoint.get("stage") != 1:
        raise ValueError("Expected a matching Stage-1 robot checkpoint")
    from morph_mimic.launcher import launch, close
    from morph_mimic.networks.tracker import TrackerActor

    app = launch(headless=True, device=args.device)
    from morph_mimic.scripts import common

    original_build = common.build_env
    original_trajectory = evaluate.TrackingTrajectory
    capture = {}

    def build_recording_env(*positional, **keywords):
        env, robot_spec = original_build(*positional, **keywords)
        capture["state"] = install_capture(env)
        return env, robot_spec

    def trajectory_factory(control_dt):
        return FullStateTrajectory(control_dt, capture=capture["state"])

    try:
        common.build_env = build_recording_env
        evaluate.TrackingTrajectory = trajectory_factory
        actor = TrackerActor().to(args.device).eval()
        actor.load_state_dict(checkpoint["ppo"]["actor"])
        result = evaluate._evaluate_clip(args, checkpoint, metadata, actor, None, app)
        report = dict(schema=1, robot=args.robot, stage=1,
                      checkpoint=str(args.resume.resolve()),
                      checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                      full_physical_joint_count=len(capture["state"]["all_joint_names"]),
                      full_joint_sampling="Post-physics, before automatic reset; no nominal reconstruction",
                      start_phase=args.start_phase, evaluation_is_partial=args.start_phase != 0,
                      contract_verified=result["contract_verified"], motions=[result])
        evaluate._write_report(args.out, report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        common.build_env = original_build
        evaluate.TrackingTrajectory = original_trajectory
        close(app)


if __name__ == "__main__":
    main()
