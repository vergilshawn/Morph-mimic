"""Shared helpers for the training / labeling / play scripts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch

from morph_mimic.utils.motion import MotionDataset
from morph_mimic.utils.reference_scene import reference_scene_grid_shape
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def build_env(
    stage: int,
    robot: str,
    num_envs: int,
    ref_env_ratio: float = 0.3,
    device: str = "cuda:0",
    replay: bool = False,
    use_usd: bool = True,
    require_contacts: bool = True,
    motion_dir: str | None = None,
    clips: list[str] | None = None,
    **env_kwargs,
):
    from morph_mimic.cfg.configs import stage1_env_cfg, stage2_env_cfg, stage3_env_cfg
    from morph_mimic.envs.acp_env import AcpEnv
    from morph_mimic.utils.robot_cfg import make_robot_cfg

    spec = RobotSpec.load(robot)
    if stage == 1:
        cfg = stage1_env_cfg(num_envs)
    elif stage == 2:
        cfg = stage2_env_cfg(num_envs)
    else:
        cfg = stage3_env_cfg(num_envs, ref_env_ratio)
    cfg.sim.device = device
    cfg.replay = replay
    for k, v in env_kwargs.items():
        setattr(cfg, k, v)
    if stage >= 2 and not replay:
        cfg.record_max_steps = max(cfg.record_max_steps, math.ceil(cfg.episode_length_s / spec.control_dt) + 1)
    directory = Path(motion_dir) if motion_dir is not None else Path(spec.motion_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    motion = MotionDataset(spec, directory, device=device, clip_names=clips)
    configure_reference_scene(cfg, motion, num_envs)
    # A scene reference's FK/collision labels use this local model. The stock
    # G1 USD also has articulated fingers and cannot certify that contract.
    cfg.robot = make_robot_cfg(spec, use_usd=False if motion.scene_bindings else use_usd)
    if not replay and require_contacts and cfg.ref_env_ratio > 0:
        missing = set(motion.clip_names) - set(motion.contact_flags)
        if missing:
            raise ValueError(
                f"{robot}: {len(missing)} clips lack contact labels. "
                "Generate and validate labels with label_contacts before training. "
                "Smoke tests may explicitly pass require_contacts=False."
            )
        too_long = [clip.name for clip in motion.clips.values()
                    if (clip.num_frames - 1) / clip.fps > cfg.episode_length_s]
        if too_long:
            raise ValueError(
                f"{len(too_long)} clips exceed the {cfg.episode_length_s}s episode limit. "
                "Cut source recordings into complete short skills with prepare_motion; "
                "otherwise CSI cannot finish clips and stage 2 cannot admit successful rollouts."
            )
    env = AcpEnv(cfg, spec, motion)
    return env, spec


def configure_reference_scene(cfg, motion, num_envs):
    """Bind a single clip's source scene; refuse a flat or random replacement."""
    bindings = getattr(motion, "scene_bindings", {})
    if not bindings:
        if getattr(cfg, "reference_scene_path", None) is not None:
            raise ValueError("reference_scene_path requires a hashed motion scene sidecar")
        return
    if len(motion.clip_names) != 1 or len(bindings) != 1:
        raise ValueError("fixed scene ingestion currently supports exactly one bound clip")
    reference_scene_grid_shape(num_envs)  # Validate before allocating scene copies.
    if cfg.ref_env_ratio != 1.0 or cfg.stage not in (1, 2):
        raise ValueError("fixed scene ingestion requires Stage 1/2 reference environments")
    if getattr(cfg, "flat_reference", False):
        raise ValueError("--flat_reference cannot replace a bound obstacle scene")
    scene = bindings[motion.clip_names[0]]
    cfg.reference_scene_path = str(scene.sidecar_path)
    cfg.reference_scene_fingerprint = scene.fingerprint
    cfg.terrain_types = cfg.ref_terrain_types = ("reference_scene",)
    cfg.num_terrain_levels = 1
    # Symmetric tile around the first reference root XY, with no border walls.
    bounds = scene.rebased_mesh().bounds
    clip = motion.clips[motion.clip_names[0]]
    xy = clip.root_pos[:, :2].detach().cpu().numpy() - scene.source_first_root_xy
    extent = np.maximum(np.abs(bounds[:, :2]).max(axis=0), np.abs(xy).max(axis=0))
    cfg.terrain_size = tuple((2.0 * (extent + 1.0)).tolist())


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--robot", type=str, default="g1", choices=["g1", "mojia"])
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--continue_run", action="store_true", help="resume the same stage including optimizers and buffers")
    parser.add_argument("--motion_dir", type=str, default=None)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--flat_reference", action="store_true", help="use flat terrain only for stage 1/2 baseline validation")


def configure_runner(args, stage: int):
    from morph_mimic.cfg.configs import runner_cfg

    cfg = runner_cfg(stage)
    if args.max_iterations is not None:
        if args.max_iterations < 1:
            raise ValueError("--max_iterations must be positive")
        cfg.max_iterations = args.max_iterations
    if args.save_interval is not None:
        cfg.save_interval = args.save_interval
    if args.continue_run:
        if args.resume is None:
            raise ValueError("--continue_run requires --resume")
        cfg.resume_path = args.resume
    return cfg


def initialize_from_checkpoint(path, actor, critic_m, critic_fs, planner, device, stage, robot, env=None):
    """Transfer networks between stages; the new stage starts its own schedules."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("robot") not in (None, robot):
        raise ValueError(f"checkpoint robot {ckpt['robot']} does not match {robot}")
    expected = stage - 1
    if ckpt.get("stage") not in (None, expected):
        raise ValueError(f"stage {stage} requires a stage {expected} checkpoint; use --continue_run for the same stage")
    if env is not None:
        from morph_mimic.utils.checkpoint import validate_checkpoint_contract
        validate_checkpoint_contract(ckpt, env, planner, mode="transfer")
    actor.load_state_dict(ckpt["ppo"]["actor"])
    critic_m.load_state_dict(ckpt["ppo"]["critic_m"])
    critic_fs.load_state_dict(ckpt["ppo"]["critic_fs"])
    if planner is not None:
        if stage == 3 and ckpt.get("planner") is None:
            raise ValueError("stage 3 requires a trained planner from stage 2")
        if ckpt.get("planner") is not None:
            planner.load_state_dict(ckpt["planner"])
    return ckpt


def run_training(args, stage: int):
    """Launch SimulationApp before importing environment-dependent modules."""
    if stage > 1 and args.resume is None:
        raise ValueError(f"stage {stage} requires --resume from stage {stage - 1}")
    if args.resume is not None and not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)
    if stage == 1 and args.resume is not None:
        args.continue_run = True
    if args.flat_reference and stage == 3:
        raise ValueError("--flat_reference is only supported in stage 1/2")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    from morph_mimic.launcher import launch

    app = launch(headless=args.headless, device=args.device)
    env = None
    try:
        cfg = configure_runner(args, stage)
        from morph_mimic.networks.acp import ACPPlanner, MLPPlanner
        from morph_mimic.networks.tracker import Critic, TrackerActor
        from morph_mimic.rl.runner import Runner

        env, _ = build_env(
            stage, args.robot, resolve_num_envs(args, stage),
            ref_env_ratio=getattr(args, "ref_env_ratio", 0.3), device=args.device,
            seed=args.seed,
            flat_reference=args.flat_reference,
            motion_dir=args.motion_dir, clips=args.clips,
            **({"terrain_types": ("flat",), "ref_terrain_types": ("flat",), "num_terrain_levels": 1}
               if args.flat_reference else {}),
        )
        device = torch.device(args.device)
        actor = TrackerActor().to(device)
        critic_m, critic_fs = Critic(353).to(device), Critic(298).to(device)
        planner = None
        if stage > 1:
            planner = (ACPPlanner if args.planner == "acp" else MLPPlanner)().to(device)
        if args.resume is not None and not args.continue_run:
            initialize_from_checkpoint(
                args.resume, actor, critic_m, critic_fs, planner, device, stage, args.robot, env=env,
            )
            cfg.initial_checkpoint = args.resume
        runner = Runner(env, cfg, device, seed=args.seed)
        runner.learn(actor, critic_m, critic_fs, planner, log_dir_for(args, stage, args.robot))
    finally:
        if env is not None:
            env.close()
        from morph_mimic.launcher import close

        close(app)


def resolve_num_envs(args, stage: int) -> int:
    if args.num_envs is not None:
        return args.num_envs
    return {1: 8192, 2: 4096, 3: 4096}[stage]


def log_dir_for(args, stage: int, robot: str) -> str:
    if args.log_dir is not None:
        return args.log_dir
    return str(PROJECT_ROOT / "runs" / f"{robot}_stage{stage}" / f"seed{args.seed}")
