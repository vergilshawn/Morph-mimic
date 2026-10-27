"""Behavioral regression tests for collection boundaries and numerical failures."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from morph_mimic.networks.tracker import Critic, TrackerActor
from morph_mimic.rl.ppo import PPO, PPOCfg
from morph_mimic.rl.runner import Runner, RunnerCfg
from morph_mimic.rl.storage import RolloutStorage
from morph_mimic.utils.transforms import quat_diff_rad
from tests.test_training_loop import TensorEnv


def tensor_environment_class():
    """Load real methods without starting Isaac Sim or importing GPU modules."""
    path = Path(__file__).resolve().parents[1] / "morph_mimic/envs/acp_env.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AcpEnv")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    scope = dict(torch=torch, np=np, DirectRLEnv=object, quat_diff_rad=quat_diff_rad)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["AcpEnv"]


def populated_rollout(ppo, env):
    storage = RolloutStorage(4, 2, 468, 23, torch.device("cpu"), env.is_ref_env)
    for _ in range(2):
        cm, cf = env.get_critic_obs()
        with torch.no_grad():
            vm, vfs = ppo.critic_m(cm), ppo.critic_fs(cf)
        action, logp = ppo.act(env.obs)
        storage.add(env.obs, action, logp, torch.ones(4), torch.ones(4, dtype=torch.bool),
                    vm, vfs, cm, cf, ppo.action_mean, ppo.action_std)
    storage.compute_returns(torch.zeros(4), torch.zeros(4), .99, .95)
    return storage


def make_ppo(**kwargs):
    return PPO(TrackerActor(), Critic(353), Critic(298),
               PPOCfg(epochs=1, num_minibatches=2, warmup_iterations=0, **kwargs), torch.device("cpu"))


def test_noise_can_learn_after_entropy_pushes_toward_upper_bound():
    ppo = make_ppo()
    for _ in range(3):
        ppo.opt_main.zero_grad()
        loss = -ppo.evaluate(torch.zeros(1, 468)).entropy().sum()
        loss.backward()
        assert ppo.log_std.grad.abs().sum() > 0
        ppo.opt_main.step()
        ppo.constrain_noise()
        assert ppo.log_std.max() <= 0
    with torch.no_grad():
        ppo.log_std.fill_(-100)
    ppo.constrain_noise()
    ppo.opt_main.zero_grad()
    ppo.evaluate(torch.zeros(1, 468)).entropy().sum().backward()
    assert ppo.log_std.grad.abs().sum() > 0


def test_exact_kl_matches_full_rollout_distribution_after_update():
    torch.manual_seed(11)
    ppo = make_ppo()
    storage = populated_rollout(ppo, TensorEnv())
    stats = ppo.update(storage, 1)
    with torch.no_grad():
        old = torch.distributions.Normal(storage.action_means.flatten(0, 1), storage.action_stds.flatten(0, 1))
        current = ppo.evaluate(storage.obs.flatten(0, 1))
        expected = torch.distributions.kl_divergence(old, current).sum(-1).mean()
    assert stats["kl"] >= 0
    assert stats["kl"] == pytest.approx(float(expected), rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_update_does_not_step_optimizer(failure):
    ppo = make_ppo()
    storage = populated_rollout(ppo, TensorEnv())
    before = [p.detach().clone() for p in ppo.actor.parameters()]
    if failure == "loss":
        storage.returns.fill_(float("nan"))
    else:
        ppo.actor.head.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises((FloatingPointError, RuntimeError), match="[Nn]onfinite|non-finite"):
        ppo.update(storage, 1)
    assert all(torch.equal(old, new) for old, new in zip(before, ppo.actor.parameters()))
    assert not ppo.opt_main.state


def test_only_artificial_timeout_exports_pre_reset_critic_observations():
    Env = tensor_environment_class()
    env = Env.__new__(Env)
    env.cfg = SimpleNamespace(stage=1, replay=False, evaluation=False, grace_steps=0,
                              reference_grace_steps=0, max_reference_root_error_m=.5,
                              min_height_ref=.3, min_height_goal=.45)
    env.device, env.num_envs, env.max_episode_length = "cpu", 4, 10
    env.episode_length_buf = torch.full((4,), 9)
    # Timeout, fallen+timeout, reference-end+timeout, goal reached+timeout.
    env.is_ref_env = torch.tensor([True, True, True, False])
    env.phase = torch.tensor([.5, .5, 1., 0.])
    env.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0., 1.], [0., 0., .1], [0., 0., 1.], [0., 0., 1.]])))
    env._ref_sample = lambda: {"root_pos": env.robot.data.root_pos_w.clone()}
    env.goal_pos = torch.tensor([[9., 0., 1.], [9., 0., 1.], [9., 0., 1.], [0., 0., 1.]])
    env.goal_height = torch.ones(4)
    env.cfg.success_horiz, env.cfg.success_vert = .5, .1
    env.ep_reached_goal = torch.zeros(4, dtype=torch.bool)
    env.ep_min_goal_dist = torch.full((4,), 1e9)
    env.extras = {}
    cm, cf = torch.full((4, 353), 7.), torch.full((4, 298), 11.)
    env.get_critic_obs = lambda: (cm, cf)
    term, trunc = env._get_dones()
    final = env.extras["timeout_bootstrap"]
    assert final["env_ids"].tolist() == [0]
    assert term.tolist() == [False, True, False, True]
    assert trunc.all()
    cm.fill_(99.)  # Simulate automatic reset mutating the live source buffers.
    assert final["critic_m"].eq(7).all()
    env.episode_length_buf.zero_()
    env._get_dones()
    assert "timeout_bootstrap" not in env.extras


def test_runner_uses_terminal_value_without_crossing_reset_boundary(monkeypatch):
    import morph_mimic.rl.runner as runner_module
    captured = {}

    class CapturingStorage(RolloutStorage):
        def compute_returns(self, *args):
            captured["rewards"] = self.rewards.clone()
            captured["dones"] = self.dones.clone()
            super().compute_returns(*args)

    class TimeoutEnv(TensorEnv):
        def step(self, actions):
            # Reset observations deliberately imply a very different value.
            self.critic_m.fill_(99)
            self.critic_fs.fill_(99)
            return ({"policy": self.obs.clone()}, torch.ones(4),
                    torch.tensor([True, False, False, True]),
                    torch.tensor([False, True, True, False]),
                    {"timeout_bootstrap": {"env_ids": torch.tensor([1, 2]),
                                           "critic_m": torch.full((2, 353), 7.),
                                           "critic_fs": torch.full((2, 298), 11.)}})

    class FirstFeatureCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
        def forward(self, x):
            return x[:, 0] * self.weight

    monkeypatch.setattr(runner_module, "RolloutStorage", CapturingStorage)
    cfg = RunnerCfg(num_steps=1, max_iterations=1, save_interval=0,
                    ppo=PPOCfg(epochs=1, num_minibatches=1, warmup_iterations=0))
    Runner(TimeoutEnv(), cfg, torch.device("cpu")).learn(TrackerActor(), FirstFeatureCritic(), FirstFeatureCritic())
    assert captured["rewards"][0].tolist() == pytest.approx([1., 1 + .99 * 7, 1 + .99 * 11, 1.])
    assert captured["dones"].all()


def test_action_bound_does_not_modify_raw_sample_and_rejects_nan():
    Env = tensor_environment_class()
    env = Env.__new__(Env)
    env.cfg = SimpleNamespace(action_clip=20., replay=False, evaluation=False)
    env.actions = torch.zeros(1, 23)
    env.prev_actions = torch.zeros(1, 2, 23)
    env.motion_data = None
    env._update_ref_contact_flags = lambda: None
    raw = torch.full((1, 23), 100.)
    env._pre_physics_step(raw)
    assert raw.eq(100).all() and env.actions.eq(20).all()
    before = env.actions.clone()
    with pytest.raises(FloatingPointError):
        env._pre_physics_step(torch.full((1, 23), float("nan")))
    assert torch.equal(env.actions, before)


def test_planner_nonfinite_gradient_is_rejected_before_optimizer_step():
    from morph_mimic.contacts.replay import HindsightBuffer
    from morph_mimic.networks.acp import ACPPlanner

    planner = ACPPlanner()
    before = [p.detach().clone() for p in planner.parameters()]
    next(planner.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
    optimizer = torch.optim.Adam(planner.parameters())
    buffers = {"ref": HindsightBuffer(), "goal": HindsightBuffer()}
    points = torch.randn(2, 187, 3)
    buffers["ref"].add("flat/tile0", {
        "points": points, "heights": points[..., 2].reshape(2, 17, 11),
        "pg": torch.randn(2, 378), "targets": torch.randn(2, 9, 3),
        "flags": torch.ones(2, 9, dtype=torch.bool),
    }, quality=0.)
    cfg = RunnerCfg(planner_min_entries=1, planner_updates_per_iter=1,
                    planner_batch=2, planner_grad_sub_batches=1)
    runner = Runner(TensorEnv(stage=2), cfg, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="non-finite"):
        runner._update_planner(planner, optimizer, buffers)
    assert not optimizer.state
    assert all(torch.equal(old, new) for old, new in zip(before, planner.parameters()))
