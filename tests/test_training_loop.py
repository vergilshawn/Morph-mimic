"""Exercise the real PPO/runner through an Isaac-compatible tensor-only stub.

These tests validate optimization and checkpoint wiring, not robot physics.
"""
from pathlib import Path
from types import SimpleNamespace

import torch

from morph_mimic.networks.tracker import Critic, TrackerActor
from morph_mimic.rl.ppo import PPO, PPOCfg
from morph_mimic.rl.runner import Runner, RunnerCfg
from morph_mimic.rl.storage import RolloutStorage


class TensorEnv:
    num_envs = 4

    def __init__(self, stage=1):
        self.cfg = SimpleNamespace(stage=stage, observation_space=468, action_space=23,
                                   eta_obs=0., eta_ref_obs=0., eta_rew=0.)
        self.robot_spec = SimpleNamespace(name="g1")
        self.is_ref_env = torch.tensor([True, True, False, False])
        self.finished_episodes = []
        self.obs = torch.randn(4, 468)
        self.critic_m = torch.randn(4, 353)
        self.critic_fs = torch.randn(4, 298)

    def reset(self):
        return {"policy": self.obs.clone()}, {}

    def get_critic_obs(self):
        return self.critic_m.clone(), self.critic_fs.clone()

    def step(self, actions):
        assert actions.shape == (4, 23)
        self.obs = torch.randn(4, 468)
        self.critic_m = torch.randn(4, 353)
        self.critic_fs = torch.randn(4, 298)
        return {"policy": self.obs.clone()}, torch.randn(4), torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), {}


def test_ppo_uses_distinct_privileged_observations():
    torch.manual_seed(7)
    env = TensorEnv()
    actor, vm, vfs = TrackerActor(), Critic(353), Critic(298)
    ppo = PPO(actor, vm, vfs, PPOCfg(epochs=1, num_minibatches=2, warmup_iterations=0), torch.device("cpu"))
    storage = RolloutStorage(4, 2, 468, 23, torch.device("cpu"), env.is_ref_env)
    actor_before = actor.head.weight.detach().clone()
    vm_before, vfs_before = vm.head.weight.detach().clone(), vfs.head.weight.detach().clone()
    obs = env.reset()[0]["policy"]
    for _ in range(2):
        cm, cf = env.get_critic_obs()
        with torch.no_grad():
            values_m, values_fs = vm(cm), vfs(cf)
        actions, logp = ppo.act(obs)
        nxt, rew, term, trunc, _ = env.step(actions)
        storage.add(obs, actions, logp, rew, term | trunc, values_m, values_fs, cm, cf)
        obs = nxt["policy"]
    storage.compute_returns(torch.zeros(4), torch.zeros(4), .99, .95)
    stats = ppo.update(storage, iteration=1)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    assert not torch.equal(actor.head.weight, actor_before)
    assert not torch.equal(vm.head.weight, vm_before)
    assert not torch.equal(vfs.head.weight, vfs_before)


def test_runner_checkpoint_continues_optimizer_and_iteration(tmp_path):
    cfg = RunnerCfg(num_steps=2, max_iterations=1, save_interval=0,
                    ppo=PPOCfg(epochs=1, num_minibatches=2, warmup_iterations=0))
    runner = Runner(TensorEnv(), cfg, torch.device("cpu"))
    ppo = runner.learn(TrackerActor(), Critic(353), Critic(298), log_dir=str(tmp_path))
    checkpoint = tmp_path / "final.pt"
    state = torch.load(checkpoint, weights_only=False)
    assert state["next_iteration"] == 1
    assert state["ppo"]["opt_main"]["state"]
    cfg.resume_path = str(checkpoint)
    cfg.max_iterations = 2
    resumed = Runner(TensorEnv(), cfg, torch.device("cpu"))
    ppo2 = resumed.learn(TrackerActor(), Critic(353), Critic(298), log_dir=str(tmp_path))
    assert resumed.iteration == 2
    assert resumed.total_steps == 16
    assert ppo2.opt_main.state_dict()["state"]
    assert not torch.equal(ppo.actor.head.weight, ppo2.actor.head.weight)


def test_one_sample_advantage_and_minibatches_stay_finite():
    storage = RolloutStorage(1, 1, 468, 23, torch.device("cpu"), torch.tensor([True]))
    storage.rewards.fill_(1.)
    storage.compute_returns(torch.zeros(1), torch.zeros(1), .99, .95)
    assert torch.isfinite(storage.advantages).all()
    batches = list(storage.minibatch_indices(8))
    assert len(batches) == 1 and len(batches[0][0]) == 1


def test_stage3_always_uses_predicted_contacts():
    env = TensorEnv(stage=3)
    Runner(env, RunnerCfg(), torch.device("cpu"))._set_annealing()
    assert (env.cfg.eta_obs, env.cfg.eta_ref_obs, env.cfg.eta_rew) == (1., 1., 1.)


def test_split_planner_gradients_equal_full_batch_with_uneven_contacts():
    from morph_mimic.contacts.replay import planner_loss

    torch.manual_seed(11)
    logits = torch.randn(4, 9, 7, requires_grad=True)
    occupancy_logits = torch.randn(4, 9, requires_grad=True)
    flags = torch.zeros(4, 9)
    flags[0, 0] = 1
    flags[2:] = 1
    flags = flags.bool()  # hindsight replay stores occupancy flags as bool
    targets = torch.randn(4, 9, 7).softmax(-1)
    full, _ = planner_loss(logits.softmax(-1), occupancy_logits.sigmoid(), targets, flags)
    expected = torch.autograd.grad(full, (logits, occupancy_logits))
    split = 0
    for sl in (slice(0, 2), slice(2, 4)):
        partial, _ = planner_loss(logits[sl].softmax(-1), occupancy_logits[sl].sigmoid(),
                                  targets[sl], flags[sl], normalization=(int(flags.sum()), 4))
        split = split + partial
    actual = torch.autograd.grad(split, (logits, occupancy_logits))
    assert torch.allclose(full, split, atol=1e-6)
    for first, second in zip(expected, actual):
        assert torch.allclose(first, second, atol=1e-7)
