"""Verify actual critic observation frames and per-regime PPO value clipping."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from morph_mimic.networks.tracker import Critic
from morph_mimic.rl.ppo import PPO, PPOCfg
from morph_mimic.rl.storage import RolloutStorage
from morph_mimic.utils import checkpoint as cp
from morph_mimic.utils.motion import MotionClip
from morph_mimic.utils.specs import RobotSpec
from morph_mimic.utils.transforms import quat_apply, to_body_yaw_frame, yaw_from_quat


def environment_class():
    source = Path(__file__).resolve().parents[1] / "morph_mimic/envs/acp_env.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "AcpEnv")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        cls,
    ], type_ignores=[])
    scope = dict(torch=torch, np=np, DirectRLEnv=object, _GRAV=(0., 0., -1.),
                 quat_apply=quat_apply, to_body_yaw_frame=to_body_yaw_frame,
                 yaw_from_quat=yaw_from_quat)
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope["AcpEnv"]


def test_critic_observations_ignore_scene_and_spawn_translation_in_xyz():
    env = environment_class().__new__(environment_class())
    env.num_envs, env.device = 2, "cpu"
    env.cfg = SimpleNamespace(reference_scene_path="bound", critic_reference_frame="scene_local")
    env.reference_scene_origins = torch.tensor([[0., 0., 0.], [150., -120., 3.]])
    env.cell_offset = torch.tensor([[0., 0., 0.], [2., -1., .2]])
    names = [f"contact_{i}" for i in range(9)]
    roots = torch.tensor([[.3, -.4, .85], [.35, -.4, .85], [.4, -.4, .85]])
    quats = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
    offsets = torch.arange(27, dtype=torch.float32).reshape(9, 3) / 100
    clip = MotionClip(
        "synthetic", "box_climb", 50., .04, roots, quats,
        torch.zeros(3, 3), torch.zeros(3, 3), torch.zeros(3, 23), torch.zeros(3, 23),
        names, roots[:, None, :] + offsets, quats[:, None, :].repeat(1, 9, 1),
    )
    env.motion_data = SimpleNamespace(clips={clip.name: clip}, clip_names=[clip.name],
                                      sample=lambda ids, phases: clip.sample(phases))
    env.clip_id = torch.zeros(2, dtype=torch.long)
    env.phase = torch.full((2,), .5)
    env.is_ref_env = torch.ones(2, dtype=torch.bool)
    env.contact_motion_candidates = [[name] for name in names]
    env._contact_clip_body_idx = {}
    reference = env._ref_sample()
    env.robot = SimpleNamespace(data=SimpleNamespace(
        root_ang_vel_b=torch.zeros(2, 3), root_quat_w=reference["root_quat"],
        root_pos_w=reference["root_pos"], root_link_lin_vel_b=torch.zeros(2, 3),
        joint_pos=reference["joint_pos"], joint_vel=reference["joint_vel"],
    ))
    env.controlled_idx = torch.arange(23)
    env.default_pose = torch.zeros(23)
    env.actions = torch.zeros(2, 23)
    env.ref_contact_flags = torch.zeros(2, 9)
    env.episode_length_buf = torch.zeros(2)
    env.max_episode_length = 750
    env.goal_pos = reference["root_pos"] + torch.tensor([.5, 0., .1])
    env.goal_height = reference["root_pos"][:, 2] + .2
    env.predicted_occupancy = torch.zeros(2, 9)
    env.predicted_contacts = torch.zeros(2, 9, 3)
    env.goal_command = lambda: torch.zeros(2, 3)
    env._planner_heights_relative = lambda: torch.zeros(2, 187)

    vm, vfs = env.get_critic_obs()
    assert vm.shape == (2, 353) and vfs.shape == (2, 298)
    torch.testing.assert_close(vm[0], vm[1], rtol=0, atol=1e-5)
    torch.testing.assert_close(vfs[0], vfs[1], rtol=0, atol=1e-5)
    expected_positions = (roots[1] - torch.tensor([roots[0, 0], roots[0, 1], 0.])) + offsets
    torch.testing.assert_close(vm[:, 317:344].reshape(2, 9, 3),
                               expected_positions.repeat(2, 1, 1), rtol=0, atol=1e-5)
    torch.manual_seed(3)
    with torch.no_grad():
        values = Critic(353)(vm)
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=1e-6)


class ConstantActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(23))

    def forward(self, obs):
        return self.mean.unsqueeze(0).expand(len(obs), -1)


class ConstantCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.))

    def forward(self, obs):
        return self.value.expand(len(obs))


@pytest.mark.parametrize("is_reference,iteration,expected_loss,actor_updates,critic_updates", [
    (True, 0, 3.24, True, False),
    (False, 0, 1., False, True),
    (False, 500, 3.24, True, False),
])
def test_ppo_clipping_and_actor_warmup_apply_per_regime(
        is_reference, iteration, expected_loss, actor_updates, critic_updates):
    actor, critic_m, critic_fs = ConstantActor(), ConstantCritic(), ConstantCritic()
    ppo = PPO(actor, critic_m, critic_fs,
              PPOCfg(epochs=1, num_minibatches=1, entropy_coef=0.), torch.device("cpu"))
    storage = RolloutStorage(1, 1, 468, 23, torch.device("cpu"),
                             torch.tensor([is_reference]))
    action = torch.full((1, 23), .5)
    with torch.no_grad():
        dist = ppo.evaluate(torch.zeros(1, 468))
        storage.actions[0] = action
        storage.log_probs[0] = dist.log_prob(action).sum(-1)
        storage.action_means[0] = dist.loc
        storage.action_stds[0] = dist.scale
    storage.has_policy_statistics = True
    # Stored value=0, current prediction=1, target=2. The clipped value=.2
    # incurs 3.24 loss; the unclipped value incurs 1.0 loss with nonzero grad.
    storage.values_m.zero_()
    storage.values_fs.zero_()
    storage.returns.fill_(2.)
    storage.advantages.fill_(1.)
    before = actor.mean.detach().clone()
    stats = ppo.update(storage, iteration)
    assert stats["value_loss"] == pytest.approx(expected_loss)
    assert (not torch.equal(actor.mean, before)) == actor_updates
    active_critic = critic_m if is_reference else critic_fs
    assert (active_critic.value.item() != 1.) == critic_updates


def test_reference_frame_is_a_training_contract_change_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "PROJECT_ROOT", tmp_path)
    model = tmp_path / "assets/g1/robot/model.xml"
    model.parent.mkdir(parents=True)
    model.write_text("robot model")
    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / "clip.npz").write_bytes(b"motion")
    (motions / "clip_contacts.npz").write_bytes(b"labels")
    env = SimpleNamespace(
        robot_spec=RobotSpec.load("g1"), num_envs=2, contact_link_names=[],
        motion_data=SimpleNamespace(motion_dir=motions, clip_names=["clip"]),
        cfg=SimpleNamespace(
            sim=SimpleNamespace(dt=.005), decimation=4,
            robot=SimpleNamespace(spawn=SimpleNamespace(usd_path="test://robot")),
            critic_reference_frame="scene_local",
        ),
    )
    current = cp.build_checkpoint_contract(env)
    assert current["training"]["critic_reference_frame"] == "scene_local"
    assert "critic_reference_frame" not in current["physics"]
    old = cp.build_checkpoint_contract(env)
    old["training"].pop("critic_reference_frame")
    assert cp.validate_checkpoint_contract({"contract": old}, env, mode="evaluate")
    with pytest.raises(ValueError, match="training"):
        cp.validate_checkpoint_contract({"contract": old}, env, mode="resume")
