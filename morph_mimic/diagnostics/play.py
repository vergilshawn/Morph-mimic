"""Play a trained policy with the contact planner in the loop (GUI).

Usage:
    python -m morph_mimic.scripts.inspect play --robot g1 --resume <checkpoint> \
        [--num_envs 4] [--stage 3] [--max_steps 2000]
"""

from __future__ import annotations

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="g1", choices=["g1", "mojia"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--planner", type=str, default="acp", choices=["acp", "mlp"])
    parser.add_argument("--motion_dir", type=str, default=None)
    parser.add_argument("--clips", nargs="+", default=None)
    parser.add_argument("--flat_reference", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    from morph_mimic.launcher import launch
    app = launch(headless=args.headless, device=args.device)
    try:
        _play(args, app)
    finally:
        from morph_mimic.launcher import close

        close(app)


def _play(args, app):
    import torch

    from morph_mimic.networks.acp import ACPPlanner, MLPPlanner
    from morph_mimic.networks.tracker import TrackerActor
    from morph_mimic.scripts.common import build_env

    device = torch.device(args.device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    if ckpt.get("robot") not in (None, args.robot):
        raise ValueError(f"checkpoint robot {ckpt['robot']} does not match {args.robot}")
    if ckpt.get("stage") not in (None, args.stage):
        raise ValueError(f"checkpoint stage {ckpt['stage']} does not match --stage {args.stage}")
    env, _spec = build_env(
        args.stage, args.robot, args.num_envs, ref_env_ratio=0.0, device=args.device,
        motion_dir=args.motion_dir, clips=args.clips,
        **({"terrain_types": ("flat",), "ref_terrain_types": ("flat",), "num_terrain_levels": 1}
           if args.flat_reference and args.stage < 3 else {}),
    )
    if args.stage > 1:
        env.cfg.eta_obs = env.cfg.eta_ref_obs = env.cfg.eta_rew = 1.0
    actor = TrackerActor().to(device)
    planner = (ACPPlanner if args.planner == "acp" else MLPPlanner)().to(device) if args.stage > 1 else None
    from morph_mimic.utils.checkpoint import validate_checkpoint_contract
    validate_checkpoint_contract(ckpt, env, planner, mode="evaluate")

    actor.load_state_dict(ckpt["ppo"]["actor"])
    if planner is not None:
        if ckpt.get("planner") is None:
            env.close()
            raise ValueError("stage 2/3 playback requires a trained planner checkpoint")
        planner.load_state_dict(ckpt["planner"])
    log_std = ckpt["ppo"].get("log_std")
    actor.eval()
    if planner is not None:
        planner.eval()
    print(f"loaded {args.resume}")

    obs, _ = env.reset()
    with torch.no_grad():
        for _ in range(args.max_steps):
            if not app.is_running():
                break
            if planner is not None:
                pts, hts, pg = env.get_planner_obs()
                _, occ, contacts = planner(pts, hts, pg)
                env.set_predicted_contacts(contacts, occ)
                obs = env.refresh_observations()
            action = actor(obs["policy"])
            obs, rew, term, trunc, _ = env.step(action)
            if args.num_envs == 1:
                print(f"reward {float(rew.mean()):7.3f} | base z {env.robot.data.root_pos_w[0, 2].item():5.2f}", end="\r")
    env.close()


if __name__ == "__main__":
    main()
