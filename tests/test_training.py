"""CPU contracts for PPO, checkpointing, active teachers and serial supervision."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from morph_mimic.networks.tracker import Critic, TrackerActor
from morph_mimic.rl.ppo import PPO, PPOCfg
from morph_mimic.rl.runner import Runner, RunnerCfg
from morph_mimic.rl.storage import RolloutStorage
from morph_mimic.scripts import train_reference_teacher as v1
from morph_mimic.scripts import train_reference_teacher_v2 as v2
from morph_mimic.scripts import train_reference_teacher_v3 as v3
from morph_mimic.scripts.train_reference_teacher import (
    BASE_OBS_DIM, CRITERIA, FEATURES, OBS_DIM, SCRIPT as TEACHER_SCRIPT,
    TEACHER_CONTRACT as TEACHER_V1_CONTRACT, TeacherSupervisor,
    augment_contract as augment_teacher_contract, reference_features,
    require_teacher_checkpoint as require_v1_checkpoint, residual_joint_targets,
    sample_initial_phases, teacher_environment_class as teacher_v1_environment_class,
)
from morph_mimic.scripts.train_reference_teacher_v2 import (
    BoxTaskTrajectory, EXTRA_REWARDS, HOLD_STEPS, SCRIPT as WORLD_TEACHER_SCRIPT,
    TASK_SCRIPT, TASK_FIELDS, TEACHER_CONTRACT as TEACHER_V2_CONTRACT, TeacherSupervisorV2,
    additive_world_rewards, augment_contract as augment_world_contract, hold_summary,
    phase_and_hold, require_teacher_checkpoint as require_v2_checkpoint, task_acceptance,
    teacher_environment_class as teacher_v2_environment_class,
)
from morph_mimic.scripts.train_standing_baseline import (
    CRITERIA as STANDING_CRITERIA, EXTRA_SOURCES, StandingSupervisor, standing_acceptance, standing_metrics,
)
from morph_mimic.scripts.train_until_validated import Supervisor, SOURCE_FILES, _arguments, acceptance
from morph_mimic.utils import checkpoint as cp
from morph_mimic.utils.checkpoint import validate_checkpoint_contract
from morph_mimic.utils.contact_tracking import summary, FIELDS
from morph_mimic.utils.specs import RobotSpec
from morph_mimic.utils.training_profile import configure_training_profile
from morph_mimic.utils.transforms import quat_diff_rad

NS = SimpleNamespace


# Training loop

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


# Checkpoint contract

class Config(NS):
    def to_dict(self):
        return vars(self).copy()


@pytest.fixture
def physical_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "PROJECT_ROOT", tmp_path)
    asset = tmp_path / "assets/g1/robot"
    asset.mkdir(parents=True)
    (asset / "model.xml").write_text("original robot model")
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    for name in ("walk_a", "walk_b"):
        (motion_dir / f"{name}.npz").write_bytes(name.encode())
        (motion_dir / f"{name}_contacts.npz").write_bytes((name + " contacts").encode())
    spec = RobotSpec.load("g1")
    cfg = NS(sim=NS(dt=.005, gravity=(0., 0., -9.81)), decimation=4, action_scale=.25, action_clip=20.,
             robot=NS(spawn=Config(usd_path="test://robot", rigid_props=Config(disable_gravity=False),
                                  articulation_props=Config(enabled_self_collisions=False)),
                      actuators={"all": Config(stiffness=50., damping=2.)}), stage=1, ref_env_ratio=1.,
             episode_length_s=15., terrain_types=("flat",), seed=0)
    env = NS(robot_spec=spec, motion_data=NS(motion_dir=motion_dir, clip_names=["walk_a", "walk_b"]),
             cfg=cfg, contact_link_names=[f"body_{i}" for i in range(9)], num_envs=2)
    return env, {"contract": cp.build_checkpoint_contract(env)}


def test_resume_rejects_data_or_control_changes(physical_fixture):
    env, saved = physical_fixture
    assert cp.validate_checkpoint_contract(saved, env)
    path = env.motion_data.motion_dir / "walk_a.npz"
    path.write_bytes(b"changed motion with unchanged filename")
    with pytest.raises(ValueError, match="clips"):
        cp.validate_checkpoint_contract(saved, env)
    env.cfg.action_scale = .5
    with pytest.raises(ValueError, match="physics"):
        cp.validate_checkpoint_contract(saved, env, mode="transfer")


def test_evaluation_allows_subset_and_clean_settings_but_not_new_labels(physical_fixture):
    env, saved = physical_fixture
    env.motion_data.clip_names = ["walk_a"]
    env.cfg.episode_length_s = 8.04
    env.cfg.terrain_types = ("flat", "rough")
    env.num_envs = 1
    assert cp.validate_checkpoint_contract(saved, env, mode="evaluate")
    (env.motion_data.motion_dir / "walk_a_contacts.npz").write_bytes(b"reordered labels")
    with pytest.raises(ValueError, match="clips.walk_a"):
        cp.validate_checkpoint_contract(saved, env, mode="evaluate")


def test_model_hash_and_contact_order_are_enforced(physical_fixture, tmp_path):
    env, saved = physical_fixture
    env.contact_link_names[0], env.contact_link_names[6] = env.contact_link_names[6], env.contact_link_names[0]
    with pytest.raises(ValueError, match="contact_bodies"):
        cp.validate_checkpoint_contract(saved, env, mode="evaluate")
    (tmp_path / "assets/g1/robot/model.xml").write_text("new inertia under same path")
    with pytest.raises(ValueError, match="asset_sha256"):
        cp.validate_checkpoint_contract(saved, env, mode="transfer")


def test_resume_requires_training_semantics_but_transfer_allows_new_stage(physical_fixture):
    env, saved = physical_fixture
    env.cfg.stage = 2
    with pytest.raises(ValueError, match="training"):
        cp.validate_checkpoint_contract(saved, env)
    assert cp.validate_checkpoint_contract(saved, env, mode="transfer")


def test_legacy_checkpoint_is_explicitly_unverified(physical_fixture):
    env, _ = physical_fixture
    with pytest.warns(RuntimeWarning, match="unverified"):
        assert cp.validate_checkpoint_contract({}, env) is False


def test_atomic_save_failure_preserves_last_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    cp.atomic_torch_save({"accepted": 1}, path)
    previous = path.read_bytes()
    def failed_save(value, stream):
        stream.write(b"incomplete")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(cp.torch, "save", failed_save)
    with pytest.raises(OSError, match="disk failure"):
        cp.atomic_torch_save({"accepted": 2}, path)
    assert path.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_rng_restore_reproduces_python_numpy_and_torch_streams():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    saved = cp.capture_rng_state()
    expected = (random.random(), np.random.rand(3), torch.rand(3))
    cp.restore_rng_state(saved)
    assert random.random() == expected[0]
    np.testing.assert_array_equal(np.random.rand(3), expected[1])
    torch.testing.assert_close(torch.rand(3), expected[2])


def test_planner_resume_rejects_mismatched_buffer_before_changing_weights(tmp_path):
    from morph_mimic.contacts.replay import HindsightBuffer
    from morph_mimic.networks.acp import ACPPlanner
    from morph_mimic.networks.tracker import Critic, TrackerActor
    from morph_mimic.rl.ppo import PPO, PPOCfg
    from morph_mimic.rl.runner import Runner, RunnerCfg

    runner = Runner(TensorEnv(stage=2), RunnerCfg(), torch.device("cpu"))
    actor, vm, vfs = TrackerActor(), Critic(353), Critic(298)
    ppo = PPO(actor, vm, vfs, PPOCfg(), torch.device("cpu"))
    planner = ACPPlanner()
    optimizer = torch.optim.Adam(planner.parameters())
    buffers = {"ref": HindsightBuffer(), "goal": HindsightBuffer()}
    path = tmp_path / "state.pt"
    runner.save(path, ppo, planner, optimizer, buffers)
    state = torch.load(path, weights_only=False)
    buffer_path = path.parent / state["buffer_file"]
    buffer_path.write_bytes(b"different replay")
    before = actor.head.weight.detach().clone()
    with pytest.raises(ValueError, match="replay buffer"):
        runner.load(path, ppo, planner, optimizer, buffers)
    assert torch.equal(before, actor.head.weight)


def test_gravity_and_existing_planner_type_cannot_change_on_transfer(physical_fixture):
    env, saved = physical_fixture
    env.cfg.sim.gravity = (0., 0., -1.6)
    with pytest.raises(ValueError, match="physics"):
        cp.validate_checkpoint_contract(saved, env, mode="transfer")
    env.cfg.sim.gravity = (0., 0., -9.81)
    saved["contract"]["planner"] = "ACPPlanner"
    class MLPPlanner:
        pass
    with pytest.raises(ValueError, match="planner"):
        cp.validate_checkpoint_contract(saved, env, MLPPlanner(), mode="transfer")


def test_unknown_checkpoint_schema_is_rejected(physical_fixture):
    env, saved = physical_fixture
    saved['schema'] = 999
    with pytest.raises(ValueError, match='schema'):
        cp.validate_checkpoint_contract(saved, env)


def test_collision_and_actuator_configuration_are_checked(physical_fixture):
    env, saved = physical_fixture
    env.cfg.robot.spawn.articulation_props.enabled_self_collisions = True
    with pytest.raises(ValueError, match="physics"):
        cp.validate_checkpoint_contract(saved, env)
    env.cfg.robot.spawn.articulation_props.enabled_self_collisions = False
    env.cfg.robot.actuators["all"].stiffness = 500.
    with pytest.raises(ValueError, match="physics"):
        cp.validate_checkpoint_contract(saved, env)


def test_observation_schema_and_held_constraints_cannot_change_on_evaluation(physical_fixture):
    env, _ = physical_fixture
    env.cfg.observation_space, env.cfg.action_space = 468, 23
    env.cfg.held_joint_mode = "hard_limits_at_nominal"
    saved = {"contract": cp.build_checkpoint_contract(env)}
    env.cfg.observation_space = 520
    with pytest.raises(ValueError, match="observations"):
        cp.validate_checkpoint_contract(saved, env, mode="evaluate")
    env.cfg.observation_space = 468
    env.cfg.held_joint_mode = "nominal_pd"
    with pytest.raises(ValueError, match="physics"):
        cp.validate_checkpoint_contract(saved, env, mode="evaluate")


def test_nominal_and_partial_dr_profiles_are_not_strict_continuations(physical_fixture):
    env, _ = physical_fixture
    env.cfg.training_profile = "nominal_pretrain"
    env.cfg.csi_sampling_mode = "curriculum_phase"
    saved = {"contract": cp.build_checkpoint_contract(env)}
    env.cfg.training_profile = "acp_partial_dr"
    with pytest.raises(ValueError, match="training"):
        cp.validate_checkpoint_contract(saved, env)
    env.cfg.training_profile = "nominal_pretrain"
    env.cfg.csi_sampling_mode = "uniform_from_curriculum_phase"
    with pytest.raises(ValueError, match="training"):
        cp.validate_checkpoint_contract(saved, env)


# Training safeguards

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


# Training profile

def test_nominal_pretraining_keeps_task_and_control_but_disables_implemented_perturbations():
    cfg = SimpleNamespace(dr_torso_mass=(-2, 4), dr_gain_scale=(.9, 1.1),
                          dr_joint_bias_nonankle=.01, dr_joint_bias_ankle=.03,
                          dr_delay=(0, 1), obs_noise_jpos=.01,
                          tracker_scan_noise=.1, planner_scan_noise=.03,
                          action_scale=.25, w_track_contact_label=5.)
    configure_training_profile(cfg, "nominal_pretrain")
    assert cfg.dr_torso_mass == (0., 0.) and cfg.dr_gain_scale == (1., 1.)
    assert cfg.dr_delay == (0, 0) and cfg.obs_noise_jpos == 0
    assert cfg.tracker_scan_noise == cfg.planner_scan_noise == 0
    assert cfg.action_scale == .25 and cfg.w_track_contact_label == 5.
    assert cfg.training_profile == "nominal_pretrain"


def test_partial_profile_retains_configured_randomization_and_unknown_profile_fails():
    cfg = SimpleNamespace(dr_torso_mass=(-2, 4))
    configure_training_profile(cfg, "acp_partial_dr")
    assert cfg.dr_torso_mass == (-2, 4)
    with pytest.raises(ValueError, match="unknown training profile"):
        configure_training_profile(cfg, "unrecorded_experiment")


# Train until validated

class AcpFakeChildren:
    def __init__(self, args, outcomes, *, fail_train=False, fail_contacts=False):
        self.args = args
        self.outcomes = iter(outcomes)
        self.fail_train = fail_train
        self.fail_contacts = fail_contacts
        self.commands = []

    def checkpoint(self, path, iteration):
        path.parent.mkdir(parents=True, exist_ok=True)
        value = dict(schema=2, robot="g1", stage=1, next_iteration=iteration,
                     total_steps=iteration * 24 * 1024, num_envs=1024, seed=0)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(path)

    def __call__(self, command, log_path, *, env, on_start):
        self.commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        on_start(99999999)
        if command[1] == "-c":
            assert env["CUDA_VISIBLE_DEVICES"] == ""
            metadata = json.loads(Path(command[-1]).read_text())
            log_path.write_text("CHECKPOINT_METADATA=" + json.dumps(metadata) + "\n")
        elif command[2].endswith("train_stage1"):
            log_path.write_text("fake train\n")
            if self.fail_train:
                return 7
            target = int(command[command.index("--max_iterations") + 1])
            self.checkpoint(self.args.run_dir / "final.pt", target)
        elif command[2].endswith("evaluate"):
            log_path.write_text("fake evaluate\n")
            checkpoint = Path(command[command.index("--resume") + 1])
            report_path = Path(command[command.index("--out") + 1])
            trajectory = Path(command[command.index("--trajectory") + 1])
            passed = next(self.outcomes)
            motion = dict(clip="box", start_phase=0.0, completed=passed, failure=not passed,
                          terminated=not passed, budget_limited=False, interrupted=False,
                          contract_verified=True, root_position_rmse_m=.10 if passed else .4,
                          root_position_max_error_m=.2 if passed else .6,
                          joint_rmse_rad=.2, root_orientation_rmse_rad=.3)
            report = dict(contract_verified=True, evaluation_is_full=True,
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), motions=[motion])
            report_path.write_text(json.dumps(report))
            trajectory.write_bytes(b"fake-trajectory")
        elif command[2].endswith("evaluate_contact_trajectory"):
            assert env["CUDA_VISIBLE_DEVICES"] == ""
            log_path.write_text("fake CPU-only contact metrics\n")
            if self.fail_contacts:
                return 9
            report_path = Path(command[command.index("--out") + 1])
            # Deliberately bad contact F1 must not change the original gates.
            report_path.write_text(json.dumps(dict(
                coverage=dict(reference_duration_coverage_fraction=.25),
                micro_force_history_vs_reference=dict(tp=0, fp=10, fn=10, tn=0, f1=0.0),
            )))
        else:
            raise AssertionError(command)
        return 0


class TestTrainingSupervisor(unittest.TestCase):
    def test_dynamic_held_constraint_is_required_even_when_tracking_passes(self):
        motion = dict(start_phase=0.0, completed=True, failure=False, terminated=False,
                      budget_limited=False, interrupted=False, contract_verified=True,
                      root_position_rmse_m=.1, root_position_max_error_m=.2,
                      joint_rmse_rad=.2, root_orientation_rmse_rad=.3,
                      held_joint_mode="hard_limits_at_nominal",
                      physical_diagnostics={"held_joint_position_error_max_rad": .0008})
        report = dict(contract_verified=True, evaluation_is_full=True, motions=[motion])
        self.assertTrue(acceptance(report)[0])
        motion["physical_diagnostics"]["held_joint_position_error_max_rad"] = .4
        self.assertFalse(acceptance(report)[0])
        motion["physical_diagnostics"] = {}
        with self.assertRaisesRegex(ValueError, "pre-reset dynamic deviation"):
            acceptance(report)

    def args(self, directory, **overrides):
        values = dict(run_dir=Path(directory), motion_dir=Path(directory) / "motions", clips=["box"],
                      python_executable=Path("/fake/isaac/python"), num_envs=1024, seed=0,
                      robot="g1", device="cuda:0", max_iterations=20000, chunk_iterations=1000,
                      required_successes=3, wait_pid=None, wait_min_iteration=0, resume_checkpoint=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_absolute_iteration_resume_and_three_distinct_successful_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            children = AcpFakeChildren(args, [True, True, True])
            children.checkpoint(args.run_dir / "final.pt", 1000)
            result = Supervisor(args, child_runner=children).run()
            self.assertEqual(result["state"], "validated")
            self.assertEqual(result["current_iteration"], 3000)
            self.assertEqual(result["actual_total_steps"], 3000 * 24 * 1024)
            training = [c for c in children.commands if len(c) > 2 and c[2].endswith("train_stage1")]
            self.assertEqual([c[c.index("--max_iterations") + 1] for c in training], ["2000", "3000"])
            self.assertTrue(all("--continue_run" in c for c in training))
            self.assertEqual(len({item["checkpoint_sha256"] for item in result["evaluations"]}), 3)
            self.assertEqual(result["held_joint_mode"], "hard_limits_at_nominal")
            self.assertEqual(result["training_profile"], "acp_partial_dr")
            self.assertTrue(all(e["contact_metrics"]["f1"] == 0.0 and e["passed"] for e in result["evaluations"]))
            self.assertTrue(all(Path(e["contact_report"]).is_file() for e in result["evaluations"]))
            self.assertEqual(result["latest_contact_coverage"]["reference_duration_coverage_fraction"], .25)
            self.assertEqual(json.loads((args.run_dir / "supervisor_status.json").read_text())["state"], "validated")
            # Immutable initial snapshot survives final.pt being atomically replaced.
            initial = args.run_dir / "checkpoints/iter001000.pt"
            self.assertEqual(json.loads(initial.read_text())["next_iteration"], 1000)

    def test_failed_acceptance_resets_streak_and_upper_budget_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, max_iterations=3500)
            children = AcpFakeChildren(args, [True, False, True, True])
            result = Supervisor(args, child_runner=children).run()
            self.assertEqual(result["state"], "budget_exhausted")
            self.assertEqual(result["current_iteration"], 3500)
            self.assertEqual([e["consecutive_successes"] for e in result["evaluations"]], [1, 0, 1, 2])
            self.assertEqual([e["iteration"] for e in result["evaluations"]], [1000, 2000, 3000, 3500])
            first_train = next(c for c in children.commands if len(c) > 2 and c[2].endswith("train_stage1"))
            self.assertNotIn("--resume", first_train)

    def test_child_failure_stops_without_evaluation_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            children = AcpFakeChildren(args, [], fail_train=True)
            supervisor = Supervisor(args, child_runner=children)
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                supervisor.run()
            result = json.loads((args.run_dir / "supervisor_status.json").read_text())
            self.assertEqual(result["state"], "failed")
            self.assertEqual(len(children.commands), 1)
            self.assertEqual(result["evaluations"], [])

    def test_training_profile_and_held_mode_are_forwarded_to_correct_children(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, max_iterations=1000, training_profile="nominal_pretrain",
                             held_joint_mode="nominal_pd")
            children = AcpFakeChildren(args, [False])
            result = Supervisor(args, child_runner=children).run()
            train = next(c for c in children.commands if len(c) > 2 and c[2].endswith("train_stage1"))
            evaluate = next(c for c in children.commands if len(c) > 2 and c[2].endswith("evaluate"))
            self.assertEqual(train[train.index("--training-profile") + 1], "nominal_pretrain")
            self.assertEqual(train[train.index("--held-joint-mode") + 1], "nominal_pd")
            self.assertEqual(evaluate[evaluate.index("--held-joint-mode") + 1], "nominal_pd")
            self.assertEqual(result["training_profile"], "nominal_pretrain")
            self.assertIn("not the full ACP", result["training_profile_description"])
            phases = [c[2].rsplit(".", 1)[-1] for c in children.commands if c[1] == "-m"]
            self.assertEqual(phases, ["train_stage1", "evaluate", "evaluate_contact_trajectory"])

    def test_contact_diagnostic_failure_stops_without_counting_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            children = AcpFakeChildren(args, [True], fail_contacts=True)
            supervisor = Supervisor(args, child_runner=children)
            with self.assertRaisesRegex(RuntimeError, "status 9"):
                supervisor.run()
            result = json.loads((args.run_dir / "supervisor_status.json").read_text())
            self.assertEqual(result["state"], "failed")
            self.assertEqual(result["evaluations"], [])
            self.assertEqual(result["consecutive_successes"], 0)
            self.assertEqual(result["children"][-1]["phase"], "contact_metrics")

    def test_cli_exposes_profiles_without_needing_them_in_old_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "box.npz").touch()
            base = ["--motion-dir", str(root), "--clips", "box", "--run-dir", str(root / "run"),
                    "--python-executable", sys.executable]
            args = _arguments(base)
            self.assertEqual(args.training_profile, "acp_partial_dr")
            self.assertEqual(args.held_joint_mode, "hard_limits_at_nominal")
            args = _arguments(base + ["--training-profile", "nominal_pretrain", "--held-joint-mode", "nominal_pd"])
            self.assertEqual(args.training_profile, "nominal_pretrain")
            self.assertEqual(args.held_joint_mode, "nominal_pd")

    def test_runtime_and_contact_evaluator_sources_are_frozen(self):
        required = {
            "morph_mimic/utils/held_joints.py", "morph_mimic/utils/robot_cfg.py",
            "morph_mimic/utils/specs.py", "morph_mimic/cfg/configs.py",
            "morph_mimic/networks/tracker.py", "morph_mimic/utils/evaluation.py",
            "morph_mimic/scripts/evaluate_contact_trajectory.py",
        }
        self.assertTrue(required.issubset(SOURCE_FILES))


# Reference teacher

def contract():
    return dict(schema=1, robot={}, asset_sha256={}, clips={"clip": {}}, contact_bodies=[],
        physics={}, versions={}, observations=dict(policy_dim=468, action_dim=23), planner=None,
        training=dict(num_envs=1024, seed=29))


def teacher_checkpoint(value=None):
    return dict(schema=2, stage=1, robot="g1", contract=augment_teacher_contract(contract()) if value is None else value)


def physical_report(clip, digest, trajectory_hash, passed=True):
    motion = dict(clip=clip, start_phase=0., completed=passed, failure=not passed, terminated=not passed,
        budget_limited=False, interrupted=False, contract_verified=True,
        held_joint_mode="hard_limits_at_nominal", physical_diagnostics=dict(held_joint_position_error_max_rad=0.),
        trajectory_sha256=trajectory_hash, **{key: 0. for key in CRITERIA})
    return dict(start_phase=0., evaluation_is_full=True, contract_verified=True, motions=[motion],
        checkpoint_sha256=digest, reference_teacher=TEACHER_V1_CONTRACT)


class TestTeacherPhysicsContract(unittest.TestCase):
    def test_stratified_sampling_covers_starts_barrier_and_full_clip(self):
        generator = torch.Generator().manual_seed(29)
        phase = sample_initial_phases(10000, 10., generator=generator)
        self.assertEqual(int((phase == 0).sum()), 1500)
        self.assertGreaterEqual(int(((phase >= .32) & (phase <= .45)).sum()), 3500)
        self.assertGreater(int((phase > .8).sum()), 400)
        self.assertGreater(int(((phase > 0) & (phase < .2)).sum()), 400)
        self.assertTrue(bool(((phase >= 0) & (phase < .98)).all()))
        self.assertGreaterEqual(float((phase[:100] == 0).float().mean()), .05)
        for count in [0, 1, 3, 17, 1024]:
            sample = sample_initial_phases(count, 10.)
            self.assertEqual(len(sample), count)
            self.assertGreaterEqual(int((sample == 0).sum()), .15 * count)
        with self.assertRaises(ValueError):
            sample_initial_phases(3, 4.)

    def test_residual_targets_map_controlled_axes_and_current_reference_phase(self):
        controlled = torch.tensor([i for i in range(29) if i not in [1, 5, 10, 13, 17, 26]])
        default = torch.arange(29, dtype=torch.float32) / 100
        reference = torch.randn(2, 23)
        residual = torch.randn(2, 23)
        targets = residual_joint_targets(default, controlled, reference, residual)
        torch.testing.assert_close(targets[:, controlled], reference + .25 * residual)
        held = torch.tensor([1, 5, 10, 13, 17, 26])
        torch.testing.assert_close(targets[:, held], default[held].expand(2, -1))
        state = SimpleNamespace(target=None)
        robot = SimpleNamespace(set_joint_position_target=lambda target: setattr(state, "target", target.clone()))
        class Base:
            pass
        env = teacher_v1_environment_class(Base)()
        env.cfg, env.default_pose, env.controlled_idx = SimpleNamespace(replay=False), default, controlled
        env.robot, env.actions, env.phase = robot, residual, torch.tensor([.342, .452])
        env._ref_sample = lambda: dict(joint_pos=env.phase[:, None].expand(-1, 23))
        env._apply_action()
        torch.testing.assert_close(state.target[:, controlled], env.phase[:, None] + .25 * residual)
        # Reading at each apply uses the advanced phase, not a cached prior target.
        env.phase += .002
        env._apply_action()
        torch.testing.assert_close(state.target[:, controlled], env.phase[:, None] + .25 * residual)
        torch.testing.assert_close(env.actions, residual)

    def test_frozen_phase_advance_integrates_with_teacher_residual_control(self):
        # Exercise the actual frozen method without importing the Isaac runtime.
        path = Path(__file__).resolve().parents[1] / "morph_mimic/envs/acp_env.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AcpEnv")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pre_physics_step")
        scope = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
        class Base:
            _pre_physics_step = scope["_pre_physics_step"]
        env = teacher_v1_environment_class(Base)()
        env.cfg = SimpleNamespace(replay=False, evaluation=False, action_clip=20.)
        env.num_envs, env.device, env.step_dt = 1, "cpu", .02
        env.default_pose, env.controlled_idx = torch.zeros(29), torch.arange(23)
        env.phase, env.clip_duration, env.is_ref_env = torch.tensor([.34]), torch.tensor([10.]), torch.tensor([True])
        env.actions, env.prev_actions, env.motion_data = torch.zeros(1, 23), torch.zeros(1, 2, 23), object()
        env._update_ref_contact_flags = lambda: None
        env._ref_sample = lambda: dict(joint_pos=env.phase[:, None].expand(-1, 23))
        env.robot = SimpleNamespace(set_joint_position_target=lambda value: setattr(env, "target", value.clone()))
        env._pre_physics_step(torch.full((1, 23), 2.))
        env._apply_action()
        torch.testing.assert_close(env.phase, torch.tensor([.342]))
        torch.testing.assert_close(env.target[:, env.controlled_idx], torch.full((1, 23), .342 + .5))
        torch.testing.assert_close(env.actions, torch.full((1, 23), 2.))
        torch.testing.assert_close(env.prev_actions, torch.zeros(1, 2, 23))

    def test_reference_features_use_relative_height_named_joints_and_actual_yaw(self):
        yaw = torch.tensor(torch.pi / 2)
        quat = torch.tensor([[torch.cos(yaw/2), 0., 0., torch.sin(yaw/2)]])
        data = SimpleNamespace(root_quat_w=quat, root_pos_w=torch.tensor([[100., 200., 1.]]),
            root_link_lin_vel_w=torch.zeros(1, 3), root_ang_vel_w=torch.zeros(1, 3),
            joint_pos=torch.zeros(1, 29), joint_vel=torch.zeros(1, 29))
        ref = dict(root_pos=torch.tensor([[100., 201., 1.2]]), root_quat=-quat,
            root_lin_vel=torch.tensor([[0., 2., 0.]]), root_ang_vel=torch.zeros(1, 3),
            joint_pos=torch.ones(1, 23), joint_vel=torch.full((1, 23), 2.))
        env = SimpleNamespace(robot=SimpleNamespace(data=data), controlled_idx=torch.arange(23),
            num_envs=1, phase=torch.tensor([.34]), _ref_sample=lambda: ref)
        features = reference_features(env)
        self.assertEqual(features.shape, (1, OBS_DIM - BASE_OBS_DIM))
        blocks, start = {}, 0
        for key, width in FEATURES:
            blocks[key], start = features[:, start:start+width], start+width
        torch.testing.assert_close(blocks["root_position_error_body_yaw"], torch.tensor([[1., 0., .2]]), atol=1e-5, rtol=0)
        torch.testing.assert_close(blocks["reference_root_linear_velocity_body_yaw"], torch.tensor([[2., 0., 0.]]), atol=1e-5, rtol=0)
        torch.testing.assert_close(blocks["root_orientation_error_wxyz"], torch.tensor([[1., 0., 0., 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(blocks["joint_position_error"], torch.ones(1, 23))
        # Tile shifts do not alter errors or policy conditioning.
        data.root_pos_w += torch.tensor([[50., -80., 0.]])
        ref["root_pos"] += torch.tensor([[50., -80., 0.]])
        torch.testing.assert_close(reference_features(env), features)

    def test_separate_teacher_contract_rejects_acp_standing_and_semantic_changes(self):
        value = teacher_checkpoint()
        require_v1_checkpoint(value)
        for changed in [teacher_checkpoint(contract()), dict(value, schema=1), dict(value, robot="mojia")]:
            with self.assertRaisesRegex(ValueError, "teacher contract"):
                require_v1_checkpoint(changed)
        changed = deepcopy(value)
        changed["contract"]["observations"]["reference_teacher"]["sampling"]["barrier_fraction"] = .4
        with self.assertRaises(ValueError):
            require_v1_checkpoint(changed)
        # The unchanged core validator also refuses cross-use because observations
        # and physics semantics differ, including evaluation/transfer modes.
        for mode in ["resume", "evaluate", "transfer"]:
            with patch("morph_mimic.utils.checkpoint.build_checkpoint_contract", return_value=contract()):
                with self.assertRaisesRegex(ValueError, "observations"):
                    validate_checkpoint_contract(value, object(), mode=mode)
            with patch("morph_mimic.utils.checkpoint.build_checkpoint_contract", return_value=value["contract"]):
                self.assertTrue(validate_checkpoint_contract(value, object(), mode=mode))
        self.assertEqual(contract()["observations"]["policy_dim"], 468)


class TeacherFakeChildren:
    def __init__(self, args, outcomes):
        self.args, self.outcomes, self.commands = args, iter(outcomes), []

    def __call__(self, command, log_path, *, env, on_start):
        self.commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        on_start(99999999)
        if command[1] == "-c":
            assert env["CUDA_VISIBLE_DEVICES"] == ""
            log_path.write_text("CHECKPOINT_METADATA=" + Path(command[-1]).read_text() + "\n")
        elif command[2] == "morph_mimic.scripts.train_reference_teacher" and command[3] == "train":
            target = int(command[command.index("--max-iterations") + 1])
            path = self.args.run_dir / "final.pt"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(schema=2, robot="g1", stage=1, next_iteration=target,
                total_steps=target*24*1024, num_envs=1024, seed=29)))
            temporary.replace(path)
            log_path.write_text("fake training\n")
        elif command[2] == "morph_mimic.scripts.train_reference_teacher" and command[3] == "evaluate":
            path = Path(command[command.index("--out") + 1])
            trajectory = Path(command[command.index("--trajectory") + 1])
            cp = Path(command[command.index("--resume") + 1])
            trajectory.write_bytes(b"fake physical trajectory " + cp.read_bytes())
            value = physical_report(self.args.clips[0], hashlib.sha256(cp.read_bytes()).hexdigest(),
                hashlib.sha256(trajectory.read_bytes()).hexdigest(), next(self.outcomes))
            path.write_text(json.dumps(value))
            log_path.write_text("fake evaluation\n")
        elif command[2] == "morph_mimic.scripts.evaluate_contact_trajectory":
            assert env["CUDA_VISIBLE_DEVICES"] == ""
            path = Path(command[command.index("--out") + 1])
            path.write_text(json.dumps(dict(coverage={}, micro_force_history_vs_reference={})))
            log_path.write_text("fake offline contacts\n")
        else:
            raise AssertionError(command)
        return 0


class TestTeacherSupervisor(unittest.TestCase):
    def run_supervisor(self, directory, outcomes, budget):
        args = SimpleNamespace(run_dir=Path(directory), motion_dir=Path(directory)/"motions", clips=["clip"],
            python_executable=Path("/fake/python"), num_envs=1024, seed=29, robot="g1", device="cuda:0",
            max_iterations=budget, chunk_iterations=500, required_successes=3,
            wait_pid=None, wait_min_iteration=0, resume_checkpoint=None,
            training_profile="nominal_pretrain", held_joint_mode="hard_limits_at_nominal")
        children = TeacherFakeChildren(args, outcomes)
        frozen = tuple(SOURCE_FILES)
        status = TeacherSupervisor(args, child_runner=children).run()
        self.assertEqual(tuple(SOURCE_FILES), frozen)
        self.assertIn(TEACHER_SCRIPT, status["source_sha256"])
        self.assertEqual(status["reference_teacher"], TEACHER_V1_CONTRACT)
        return status, children

    def test_three_distinct_full_successes_and_strict_chunk_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            status, children = self.run_supervisor(directory, [True]*3, 15000)
            self.assertEqual(status["state"], "validated")
            self.assertEqual(status["current_iteration"], 1500)
            trains = [c for c in children.commands if len(c)>3 and c[3] == "train"]
            self.assertNotIn("--resume", trains[0])
            self.assertTrue(all("--resume" in c for c in trains[1:]))
            self.assertEqual(len({e["checkpoint_sha256"] for e in status["evaluations"]}), 3)

    def test_failed_rollout_resets_streak_and_budget_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _ = self.run_supervisor(directory, [True, False, True, True], 2000)
            self.assertEqual(status["state"], "budget_exhausted")
            self.assertEqual([e["consecutive_successes"] for e in status["evaluations"]], [1,0,1,2])


# Reference teacher v2

def bare_contract():
    return dict(observations=dict(policy_dim=468, action_dim=23), physics={})


def world_teacher_checkpoint(contract):
    return dict(schema=2, robot="g1", stage=1, contract=contract)


def arrays(steps=550):
    a = {f"box_task_{key}": np.ones((steps,)+shape) for key, shape in TASK_FIELDS.items()}
    a["box_hold_controls"] = np.maximum(0, np.arange(1, steps+1)-500)
    a["phase"] = np.minimum(1., np.arange(1, steps+1)/500)
    return a


class TestWorldTrackingTeacher(unittest.TestCase):
    def test_translation_and_feet_are_penalized_independently_of_motion_velocity(self):
        root = torch.zeros(2, 3)
        feet = torch.zeros(2, 2, 3)
        good = additive_world_rewards(root, root, feet, feet)
        self.assertTrue(all(torch.equal(value, torch.ones(2)) for value in good.values()))
        shifted_root, shifted_feet = root.clone(), feet.clone()
        shifted_root[:, 2] = .45
        shifted_feet[:, 0, 2] = .45  # One foot never makes it onto the obstacle.
        bad = additive_world_rewards(shifted_root, root, shifted_feet, feet)
        self.assertLess(float(bad["world_root_xyz"].max()), .11)
        self.assertLess(float(bad["world_root_z"].max()), .001)
        self.assertLess(float(bad["world_two_feet"].max()), .501)
        # World tile shifts do not change actual/reference differences.
        offset = torch.tensor([100., -200., .7])
        for key, value in bad.items():
            torch.testing.assert_close(value, additive_world_rewards(
                shifted_root+offset, root+offset, shifted_feet+offset, feet+offset)[key], atol=1e-5, rtol=1e-4)

    def test_additive_rewards_are_integrated_once_and_logged_as_contributions(self):
        class Base:
            def _get_rewards(self):
                self.extras["reward_terms"] = {}
                return torch.tensor([.7, .8])
        env = teacher_v2_environment_class(Base, {})()
        env.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3)))
        env._ref_sample = lambda: dict(root_pos=torch.zeros(2, 3))
        env._contact_body_positions_w = lambda: torch.zeros(2, 9, 3)
        env._ref_contact_pos = lambda _: torch.zeros(2, 9, 3)
        env.extras, env.phase = {}, torch.tensor([0., 1.])
        env.step_dt, env.is_ref_env = .02, torch.ones(2, dtype=torch.bool)
        reward = env._get_rewards()
        torch.testing.assert_close(reward, torch.tensor([.86, .96]))
        self.assertAlmostEqual(sum(float(env.extras["reward_terms"][f"weighted/{key}"])
            for key in EXTRA_REWARDS), .16, places=6)

    def test_full_start_has_500_motion_and_50_extra_hold_controls(self):
        for evaluating in [False, True]:
            for step, expected_hold in [(499, 0), (500, 0), (501, 1), (549, 49), (550, 50), (551, 51)]:
                phase, hold = phase_and_hold(torch.tensor([0.]), torch.tensor([float(step)]), torch.tensor([500.]))
                self.assertEqual(int(hold), expected_hold)
                self.assertEqual(float(phase), min(1., step/500)) if step >= 500 else self.assertLess(float(phase), 1.)
        # Noninteger RSI starts receive 50 complete controls after reaching the endpoint.
        phase, hold = phase_and_hold(torch.tensor([.3333]), torch.tensor([384.]), torch.tensor([500.]))
        self.assertEqual(float(phase), 1.)
        self.assertEqual(int(hold), 50)

    def test_environment_pre_step_clamps_training_and_evaluation_through_hold(self):
        class Base:
            def _pre_physics_step(self, actions):
                self.phase += .002  # Frozen training behavior can overshoot one.
        for evaluation in [False, True]:
            env = teacher_v2_environment_class(Base, {})()
            env.cfg = SimpleNamespace(evaluation=evaluation)
            env.phase, env.init_phase = torch.zeros(1), torch.zeros(1)
            env.clip_duration, env.step_dt = torch.tensor([10.]), .02
            env._update_ref_contact_flags = lambda: None
            for step in range(1, 551):
                env.episode_length_buf = torch.tensor([step-1])
                env._pre_physics_step(torch.zeros(1, 23))
                self.assertLessEqual(float(env.phase), 1.)
                self.assertEqual(int(env._box_hold_controls), max(0, step-500))
            self.assertEqual(float(env.phase), 1.)
            self.assertEqual(int(env._box_hold_controls), 50)

    def test_hold_never_resets_at_reference_end_and_preserves_original_failure(self):
        class Base:
            def _get_dones(self):
                return self.original_terminated.clone(), torch.ones(1, dtype=torch.bool)
        env = teacher_v2_environment_class(Base, {})()
        env.cfg, env.is_ref_env = SimpleNamespace(evaluation=False), torch.tensor([True])
        env.episode_length_buf, env.max_episode_length = torch.tensor([500]), 600
        env.original_terminated = torch.tensor([False])
        env.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(1, 3),
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]), body_quat_w=torch.zeros(1, 9, 4)))
        env.cell_offset, env.contact_link_idx = torch.zeros(1, 3), torch.arange(9)
        env.env_origin = lambda: torch.zeros(1, 3)
        env._contact_body_positions_w = lambda: torch.zeros(1, 9, 3)
        env._contact_forces_w = lambda: torch.zeros(1, 9, 3)
        with patch("morph_mimic.utils.box_task.task_metrics", return_value={"task_gate": torch.tensor([True])}):
            for count in [0, 1, 49]:
                env._box_hold_controls = torch.tensor([count])
                term, trunc = env._get_dones()
                self.assertFalse(bool(term))
                self.assertFalse(bool(trunc))
            env._box_hold_controls = torch.tensor([50])
            self.assertTrue(bool(env._get_dones()[1]))
            env._box_hold_controls, env.original_terminated = torch.tensor([12]), torch.tensor([True])
            self.assertTrue(bool(env._get_dones()[0]))

    def test_hold_zeroes_reference_velocities_without_touching_actual_state(self):
        class Base:
            def _ref_sample(self, ids=None):
                count = 2 if ids is None else len(ids)
                return {key: torch.ones(count, width) for key, width in
                    [("root_lin_vel", 3), ("root_ang_vel", 3), ("joint_vel", 23), ("root_pos", 3)]}
        env = teacher_v2_environment_class(Base, {})()
        env.phase = torch.tensor([.998, 1.])
        actual_velocity = torch.full((2, 3), 7.)
        env.robot = SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=actual_velocity))
        reference = env._ref_sample()
        for name in ["root_lin_vel", "root_ang_vel", "joint_vel"]:
            self.assertTrue(bool((reference[name][0] == 1).all()))
            self.assertTrue(bool((reference[name][1] == 0).all()))
        self.assertTrue(bool((env._ref_sample(torch.tensor([1]))["joint_vel"] == 0).all()))
        torch.testing.assert_close(actual_velocity, torch.full((2, 3), 7.))

    def test_hold_acceptance_requires_every_contiguous_pre_reset_endpoint(self):
        self.assertFalse(hold_summary(arrays(500))["passed"])
        self.assertFalse(hold_summary(arrays(549))["passed"])
        self.assertTrue(hold_summary(arrays(550))["passed"])
        bad = arrays()
        bad["box_task_task_gate"][517] = False
        self.assertFalse(hold_summary(bad)["passed"])
        bad = arrays()
        bad["box_hold_controls"][-1] = 49
        self.assertFalse(hold_summary(bad)["passed"])
        bad = arrays()
        bad["phase"][-1] = .99
        self.assertFalse(hold_summary(bad)["passed"])
        # A lone final successful pose is insufficient.
        bad = arrays()
        bad["box_task_task_gate"][500:549] = False
        self.assertEqual(hold_summary(bad)["passing_hold_endpoints"], 1)
        self.assertFalse(hold_summary(bad)["passed"])

    def test_v2_checkpoint_cannot_resume_v1_acp_or_standing(self):
        valid = world_teacher_checkpoint(augment_world_contract(bare_contract(), {"schema": 1}))
        require_v2_checkpoint(valid)
        for value in [world_teacher_checkpoint(bare_contract()), world_teacher_checkpoint(v1.augment_contract(bare_contract()))]:
            with self.assertRaisesRegex(ValueError, "v2"):
                require_v2_checkpoint(value)
        modified = deepcopy(valid)
        modified["contract"]["observations"]["reference_teacher"]["end_hold"]["control_endpoints"] = 1
        with self.assertRaises(ValueError):
            require_v2_checkpoint(modified)
        self.assertEqual(v1.TEACHER_CONTRACT["reward"], "unchanged ACP Stage1 reward definitions and weights")

    def test_strict_tracking_success_alone_cannot_validate_box_task(self):
        motion = dict(start_phase=0., completed=True, failure=False, terminated=False,
            budget_limited=False, interrupted=False, contract_verified=True,
            **{key: 0. for key in v1.CRITERIA}, physical_diagnostics=dict(box_task=dict(passed=False)))
        report = dict(start_phase=0., evaluation_is_full=True, contract_verified=True, motions=[motion], task_passed=False)
        self.assertFalse(task_acceptance(report)[0])
        report["task_passed"] = motion["physical_diagnostics"]["box_task"]["passed"] = True
        self.assertTrue(task_acceptance(report)[0])
        motion["root_position_rmse_m"] = .2
        self.assertFalse(task_acceptance(report)[0])

    def test_supervisor_freezes_v1_v2_and_task_geometry_source(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(run_dir=Path(directory), motion_dir=Path(directory), clips=["clip"],
                python_executable=Path("/fake/python"), num_envs=32, seed=29, robot="g1", device="cuda:0",
                max_iterations=10, chunk_iterations=5, required_successes=3,
                wait_pid=None, wait_min_iteration=0, resume_checkpoint=None,
                training_profile="nominal_pretrain", held_joint_mode="hard_limits_at_nominal")
            supervisor = TeacherSupervisorV2(args)
            supervisor._sources(initialize=True)
            for source in [v1.SCRIPT, WORLD_TEACHER_SCRIPT, TASK_SCRIPT]:
                self.assertIn(source, supervisor.status["source_sha256"])
            self.assertEqual(supervisor._command("train")[2], "morph_mimic.scripts.train_reference_teacher_v2")

    def test_training_episode_task_success_requires_all50_and_resets_by_index(self):
        class Base:
            def _finish_episodes(self, ids):
                for env_id in ids.tolist():
                    self.finished_episodes.append(dict(env_id=env_id, terminated=bool(env_id == 2)))
            def _reset_idx(self, ids):
                self._finish_episodes(ids)
        env = teacher_v2_environment_class(Base, {})()
        env.finished_episodes = []
        env._box_hold_controls = torch.tensor([50, 50, 50, 12])
        env._box_task_hold_pass_count = torch.tensor([50, 49, 50, 12])
        env._reset_idx(torch.tensor([0, 1, 2]))
        self.assertEqual([ep["box_task_passed"] for ep in env.finished_episodes], [True, False, False])
        torch.testing.assert_close(env._box_hold_controls, torch.tensor([0, 0, 0, 12]))
        torch.testing.assert_close(env._box_task_hold_pass_count, torch.tensor([0, 0, 0, 12]))
        env._finish_episodes(torch.tensor([3]))
        self.assertFalse(env.finished_episodes[-1]["box_task_passed"])

    def test_npz_metadata_describes_extra_physical_hold_before_atomic_write(self):
        trajectory = BoxTaskTrajectory(.02)
        trajectory.box_task_spec = {"source_sha256": {"model": "hash"}}
        with patch("morph_mimic.utils.evaluation.TrackingTrajectory.write") as write:
            original = dict(result={"expected_control_steps": 500})
            trajectory.write(Path("unused.npz"), metadata=original)
            metadata = write.call_args.kwargs["metadata"]
            self.assertEqual(metadata["result"]["expected_control_steps"], 550)
            self.assertEqual(metadata["result"]["physical_hold_required_s"], 1.)
            self.assertEqual(metadata["reference_teacher"], TEACHER_V2_CONTRACT)
            self.assertEqual(original["result"]["expected_control_steps"], 500)


# Reference teacher v3

class ContactTeacherContractTests(unittest.TestCase):
    def test_supervision_requires_successful_checks_for_exact_inputs_and_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            inputs=[root/"ref.npz",root/"ref_contacts.npz",root/"ref.scene.json"]
            for path in inputs:
                path.write_bytes(b"test input")
            report=dict(training_ready=True,source_sha256={"script":"known"},num_envs=1024,
                motion_path=str(inputs[0]),input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
            path=root/"checks.json"
            args=SimpleNamespace(readiness_report=path,num_envs=1024,motion_dir=root,clips=["ref"])
            with patch.object(v3,"source_hashes",return_value={"script":"known"}):
                path.write_text(json.dumps(report));v3.require_training_readiness(args)
                for key,value in (("training_ready",False),("num_envs",32),("source_sha256",{"script":"changed"})):
                    invalid=deepcopy(report);invalid[key]=value
                    path.write_text(json.dumps(invalid))
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        v3.require_training_readiness(args)
                path.write_text(json.dumps(report));inputs[0].write_bytes(b"replaced")
                with self.assertRaisesRegex(ValueError,"input changed"):
                    v3.require_training_readiness(args)

    def test_new_contract_rejects_old_weights_as_a_strict_resume(self):
        original = dict(observations=dict(policy_dim=468, action_dim=23), physics={})
        current = v3.augment_contract(original, {}, {})
        checkpoint = dict(schema=2, robot="g1", stage=1, contract=current)
        v3.require_teacher_checkpoint(checkpoint)
        with self.assertRaises(ValueError):
            v2.require_teacher_checkpoint(checkpoint)
        for change in ("action", "surface", "teacher"):
            broken = deepcopy(checkpoint)
            if change == "action":
                broken["contract"]["physics"]["joint_target_semantics"] = "nominal"
            elif change == "surface":
                del broken["contract"]["observations"]["contact_surface"]
            else:
                broken["contract"]["observations"]["reference_teacher"] = v2.TEACHER_CONTRACT
            with self.assertRaises(ValueError):
                v3.require_teacher_checkpoint(broken)

    def test_surface_reward_has_one_dt_and_costs_have_negative_weights(self):
        class Parent:
            def _get_rewards(self):
                return torch.tensor([.7, .8])
        with patch.object(v2, "teacher_environment_class", return_value=Parent):
            cls = v3.teacher_environment_class(object, {}, Path("unused"))
        env = cls.__new__(cls)
        env.step_dt, env.is_ref_env = .02, torch.tensor([True, False])
        env.extras = dict(reward_terms={})
        env.tracking = SimpleNamespace(last=dict(reward_terms={name: torch.ones(2) for name in v3.REWARD_WEIGHTS},
            hold_gate=torch.ones(2,dtype=torch.bool), measurement_health_gate=torch.ones(2,dtype=torch.bool)))
        torch.testing.assert_close(env._get_rewards(), torch.tensor([.74,.8]))
        self.assertAlmostEqual(sum(float(env.extras["reward_terms"]["weighted/"+key])
            for key in v3.REWARD_WEIGHTS), .02, places=6)
        self.assertLess(v3.REWARD_WEIGHTS["stance_tangent_slip"], 0)
        self.assertLess(v3.REWARD_WEIGHTS["unexpected_noncanonical_contact"], 0)

    def test_policy_anchors_use_control_start_yaw_and_preserve_absolute_height(self):
        from morph_mimic.utils.transforms import yaw_quat
        targets = torch.tensor([[[1., 0., 3.]]]).expand(2,9,3).clone()
        root = torch.tensor([[10.,20.,30.],[110.,-80.,230.]])
        origins = torch.tensor([[5.,6.,1.],[105.,-94.,201.]])
        targets[1,:,2] += 200.
        actual = v3.decode_policy_contact_anchors(targets, root,
            yaw_quat(torch.tensor([torch.pi/2,torch.pi/2])), origins)
        torch.testing.assert_close(actual[0], torch.tensor([[5.,15.,2.]]).expand(9,3), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual[0], actual[1], atol=1e-5, rtol=1e-5)

    def test_bad_raw_contact_cannot_be_consumed_for_learning(self):
        class Parent:
            def _get_dones(self):
                return torch.tensor([False]), torch.tensor([False])
        with patch.object(v2, "teacher_environment_class", return_value=Parent):
            cls=v3.teacher_environment_class(object, {}, Path("unused"))
        env=cls.__new__(cls)
        env.robot=SimpleNamespace(body_names=["pelvis"])
        env.contact_runtime=SimpleNamespace(sample=lambda:dict(all_body_health_gate=torch.tensor([False]),
            health_gate=torch.tensor([[False]]), **{key:torch.ones((1,1)) for key in
            ("buffer_invalid", "force_reconstruction_error_n", "normal_point_count", "friction_patch_count", "normal_load_n")}))
        with self.assertRaisesRegex(RuntimeError,"Unhealthy"):
            env._get_dones()

    def test_fifty_extra_physical_endpoints_are_required_for_contact_success(self):
        arrays={"contact_tracking_"+key:np.ones((600,)+shape) for key,shape in FIELDS.items()}
        arrays.update(phase=np.minimum(np.arange(1,601)/550,1.),
            box_hold_controls=np.maximum(np.arange(1,601)-550,0))
        self.assertTrue(summary(arrays)["hold_passed"])
        self.assertEqual(summary(arrays)["hold_observed_endpoints"],50)
        arrays["contact_tracking_hold_gate"][575]=False
        self.assertFalse(summary(arrays)["hold_passed"])
        arrays["contact_tracking_hold_gate"][575]=True
        arrays["box_hold_controls"][-1]=49
        self.assertFalse(summary(arrays)["hold_passed"])


# Standing baseline

def report(metrics=None, **motion_updates):
    motion = dict(clip="standing_nominal_g1", start_phase=0., completed=True,
        failure=False, terminated=False, budget_limited=False, interrupted=False, contract_verified=True)
    motion.update(motion_updates)
    return dict(start_phase=0., evaluation_is_full=True, contract_verified=True, motions=[motion],
        standing_metrics=metrics or dict(samples=500, duration_s=10., **{key: 0. for key in STANDING_CRITERIA}))


class TestStandingMetrics(unittest.TestCase):
    def test_full_physical_trace_accepts_and_catches_drift_tilt_and_held_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            positions = np.tile([0., 0., .759], (500, 1))
            arrays = dict(root_pos_w=positions.copy(), ref_root_pos_w=positions.copy(),
                root_quat_w=np.tile([1., 0., 0., 0.], (500, 1)),
                held_joint_position_error_rad=np.zeros((500, 6)), all_joint_pos=np.zeros((500, 29)),
                elapsed_control_time_s=(np.arange(500)+1)*.02)
            np.savez(path, **arrays)
            self.assertTrue(standing_acceptance(report(standing_metrics(path)))[0])
            arrays["root_pos_w"][-1] += [.151, 0., -.101]
            arrays["root_quat_w"][-1] = [np.cos(.16), 0., np.sin(.16), 0.]
            arrays["held_joint_position_error_rad"][-1, 0] = .011
            np.savez(path, **arrays)
            passed, reasons = standing_acceptance(report(standing_metrics(path)))
            self.assertFalse(passed)
            self.assertEqual(len(reasons), 4)
            arrays["ref_root_pos_w"][-1, 0] = .001
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "static reference"):
                standing_metrics(path)

    def test_short_partial_or_reset_rollout_cannot_pass(self):
        for mutation in (dict(completed=False), dict(start_phase=.5), dict(terminated=True)):
            self.assertFalse(standing_acceptance(report(**mutation))[0])
        short = report()
        short["standing_metrics"].update(samples=40, duration_s=.8)
        self.assertFalse(standing_acceptance(short)[0])
        for key in STANDING_CRITERIA:
            invalid = report()
            invalid["standing_metrics"][key] = float("nan")
            with self.assertRaises(ValueError):
                standing_acceptance(invalid)


class StandingFakeChildren:
    def __init__(self, args, outcomes):
        self.args, self.outcomes, self.commands = args, iter(outcomes), []

    def __call__(self, command, log_path, *, env, on_start):
        self.commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        on_start(99999999)
        if command[1] == "-c":
            assert env["CUDA_VISIBLE_DEVICES"] == ""
            log_path.write_text("CHECKPOINT_METADATA=" + Path(command[-1]).read_text() + "\n")
        elif command[3] == "train":
            target = int(command[command.index("--max-iterations") + 1])
            path = self.args.run_dir / "final.pt"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(schema=2, robot="g1", stage=1, next_iteration=target,
                total_steps=target*24*1024, num_envs=1024, seed=17)))
            temporary.replace(path)
            log_path.write_text("fake training\n")
        elif command[3] == "evaluate":
            path = Path(command[command.index("--out")+1])
            trajectory = Path(command[command.index("--trajectory")+1])
            checkpoint = Path(command[command.index("--resume")+1])
            trajectory.write_bytes(b"fake physical artifact " + checkpoint.read_bytes())
            value = report()
            if not next(self.outcomes):
                value["standing_metrics"]["max_root_tilt_rad"] = .5
            value["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            value["motions"][0]["trajectory_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
            path.write_text(json.dumps(value))
            log_path.write_text("fake evaluation\n")
        else:
            raise AssertionError(command)
        return 0


class TestStandingSupervisor(unittest.TestCase):
    def run_supervisor(self, directory, outcomes, budget):
        args = SimpleNamespace(run_dir=Path(directory), motion_dir=Path(directory)/"motions",
            clips=["standing_nominal_g1"], python_executable=Path("/fake/python"), num_envs=1024,
            seed=17, robot="g1", device="cuda:0", max_iterations=budget, chunk_iterations=250,
            required_successes=3, wait_pid=None, wait_min_iteration=0, resume_checkpoint=None,
            training_profile="nominal_pretrain", held_joint_mode="hard_limits_at_nominal")
        children = StandingFakeChildren(args, outcomes)
        frozen = tuple(SOURCE_FILES)
        status = StandingSupervisor(args, child_runner=children).run()
        self.assertEqual(tuple(SOURCE_FILES), frozen)
        self.assertTrue(set(EXTRA_SOURCES).issubset(status["source_sha256"]))
        return status, children

    def test_three_distinct_full_passes_stop_and_resume_strict_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            status, children = self.run_supervisor(directory, [True, True, True], 4000)
            self.assertEqual(status["state"], "validated")
            self.assertEqual(status["current_iteration"], 750)
            trains = [c for c in children.commands if len(c)>3 and c[3]=="train"]
            self.assertNotIn("--resume", trains[0])
            self.assertTrue(all("--resume" in c for c in trains[1:]))
            self.assertEqual(len({r["checkpoint_sha256"] for r in status["evaluations"]}), 3)
            self.assertEqual(status["criteria"], STANDING_CRITERIA)

    def test_failed_eval_resets_streak_and_budget_exhaustion_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _ = self.run_supervisor(directory, [True, False, True, True], 1000)
            self.assertEqual(status["state"], "budget_exhausted")
            self.assertEqual([r["consecutive_successes"] for r in status["evaluations"]], [1,0,1,2])
