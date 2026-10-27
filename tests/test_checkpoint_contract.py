"""Compatibility, atomic publication, and RNG behavior; no simulator needed."""
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from morph_mimic.utils import checkpoint as cp
from morph_mimic.utils.specs import RobotSpec


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
    from tests.test_training_loop import TensorEnv

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
