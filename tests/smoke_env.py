"""Isaac Sim smoke test: build the ACP env headless with a handful of
environments and step it briefly. Validates the whole env stack without
long training. GPU memory footprint is small (<= 64 envs).

Usage:
    cd /media/user/DATA/Morph_mimic
    PYTHONPATH=. conda run -n env_isaaclab_2 /home/user/IsaacLab/isaaclab.sh -p \
        tests/smoke_env.py --robot g1 --num_envs 64 --steps 20 [--stage 3]
"""

from __future__ import annotations

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="g1")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--stage", type=int, default=3)
parser.add_argument("--use_usd", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument("--clips", nargs="+", default=None)
args = parser.parse_args()

from morph_mimic.launcher import launch  # noqa: E402

simulation_app = launch(headless=args.headless, device=args.device)

import torch  # noqa: E402

from morph_mimic.networks.acp import ACPPlanner  # noqa: E402
from morph_mimic.networks.tracker import Critic, TrackerActor  # noqa: E402
from morph_mimic.scripts.common import build_env  # noqa: E402


def main():
    device = torch.device(args.device)
    t0 = time.time()
    env, spec = build_env(
        args.stage, args.robot, args.num_envs, ref_env_ratio=1.0 if args.stage < 3 else 0.25, use_usd=args.use_usd,
        device=args.device, require_contacts=False, motion_dir=args.motion_dir, clips=args.clips,
    )
    print(f"[smoke] env built in {time.time() - t0:.1f}s | joints: {env.robot.num_joints} | "
          f"bodies: {env.robot.num_bodies} | contact links: {env.contact_link_names[:3]}...")
    print(f"[smoke] controlled joints (23): {[env.robot.data.joint_names[i] for i in env.controlled_idx[:3].tolist()]}...")
    print(f"[smoke] ray hits: {env.ray_caster.data.ray_hits_w.shape}, dr_scan_bias: {env.dr_scan_bias.shape}")
    for sensor in getattr(env, "scene_contact_sensors", []):
        assert sensor.data.force_matrix_w.shape[-2] > 0, "terrain contact filter resolved no collision shapes"

    actor = TrackerActor().to(device)
    critic_m = Critic(353).to(device)
    critic_fs = Critic(298).to(device)
    planner = ACPPlanner().to(device)

    obs, _ = env.reset()
    print(f"[smoke] obs shape: {obs['policy'].shape}")
    t0 = time.time()
    for step in range(args.steps):
        pts, hts, pg = env.get_planner_obs()
        with torch.no_grad():
            aff, occ, contacts = planner(pts, hts, pg)
        assert torch.isfinite(pts).all(), "nonfinite terrain scan"
        env.set_predicted_contacts(contacts, occ)
        obs2 = env.refresh_observations()
        with torch.no_grad():
            vm, vfs = env.get_critic_obs()
            assert vm.shape[-1] == 353, vm.shape
            assert vfs.shape[-1] == 298, vfs.shape
            assert torch.isfinite(obs2["policy"]).all(), "nonfinite actor observation"
            a = actor(obs2["policy"])
        obs, rew, term, trunc, _ = env.step(a)
        assert torch.isfinite(rew).all(), "nonfinite reward"
        if step % 5 == 0:
            print(
                f"[smoke] step {step:3d} | rew {float(rew.mean()):8.3f} | "
                f"term {int(term.sum())}/{int(trunc.sum())} | base z {float(env.robot.data.root_pos_w[:, 2].mean()):5.2f}"
            )
            if getattr(env, "scene_contact_sensors", []):
                print(f"[smoke] terrain-filtered contact force {env._contact_forces_w().norm(dim=-1).sum().item():.3f} N")
    dt = (time.time() - t0) / args.steps
    print(f"[smoke] OK: {args.steps} steps @ {dt:.3f}s/step ({args.num_envs} envs, stage {args.stage})")
    env.close()
    from morph_mimic.launcher import close

    close(simulation_app)


if __name__ == "__main__":
    main()
