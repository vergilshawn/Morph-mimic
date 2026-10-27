"""Exercise real reference termination and CSI methods on CPU tensors.

These fixtures isolate episode bookkeeping. They do not establish physical
tracking success or replace the separate Isaac Lab rollout evaluation.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def environment_class():
    source = Path(__file__).resolve().parents[1] / "morph_mimic/envs/acp_env.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "AcpEnv")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        cls,
    ], type_ignores=[])
    scope = dict(torch=torch, np=np, DirectRLEnv=object)
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope["AcpEnv"]


def reference_environment(environment_class, *, step=45, phase=.991,
                          root=(0., 0., 1.), reference_root=(0., 0., 1.)):
    env = environment_class.__new__(environment_class)
    env.cfg = SimpleNamespace(
        stage=1, replay=False, evaluation=False, csi_enabled=True,
        csi_adjust=.05, csi_success_root_error_m=.3,
        grace_steps=75, reference_grace_steps=5,
        max_reference_root_error_m=.5, min_height_ref=.3,
        min_height_goal=.45, success_horiz=.5, success_vert=.1,
        record_max_steps=500,
    )
    env.device, env.num_envs, env.max_episode_length = "cpu", 1, 750
    env.episode_length_buf = torch.tensor([step])
    env.is_ref_env = torch.tensor([True])
    env.phase = torch.tensor([phase])
    env.init_phase = torch.tensor([.9])
    env.csi_min_phase = torch.tensor([.9])
    env.robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([root]),
    ))
    env._ref_sample = lambda env_ids=None: {"root_pos": torch.tensor([reference_root])}
    env.goal_pos = torch.tensor([[9., 0., 1.]])
    env.goal_height = torch.ones(1)
    env.ep_reached_goal = torch.zeros(1, dtype=torch.bool)
    env.ep_min_goal_dist = torch.full((1,), 1e9)
    env.extras = {}
    env.finished_episodes = []
    env.ep_record_step = torch.zeros(1, dtype=torch.long)
    env.clip_id = torch.zeros(1, dtype=torch.long)
    env.terrain_type_id = torch.zeros(1, dtype=torch.long)
    env.terrain_level = torch.zeros(1, dtype=torch.long)
    return env


def finish_episode(env):
    env.reset_terminated, env.reset_time_outs = env._get_dones()
    env._finish_episodes(torch.tensor([0]))
    assert len(env.finished_episodes) == 1
    return env.finished_episodes[0]


def test_fallen_reference_suffix_terminates_before_goal_grace_ends(environment_class):
    # The old 75-step grace covered the entire initial 0.9-to-0.99 suffix.
    # Root error stays below 0.5 here, so low height must detect this failure.
    env = reference_environment(environment_class, root=(0., 0., .08),
                                reference_root=(0., 0., .4))
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [True]
    assert truncated.tolist() == [True]
    assert env.episode_length_buf.item() < env.cfg.grace_steps


def test_goal_environment_retains_its_longer_grace(environment_class):
    env = reference_environment(environment_class, root=(0., 0., .08))
    env.is_ref_env.zero_()
    terminated, truncated = env._get_dones()
    assert not terminated.any()
    assert not truncated.any()


def test_reference_root_lost_detects_fall_on_elevated_support(environment_class):
    # Falling onto a box can leave the root above the absolute 0.3 m limit.
    env = reference_environment(environment_class, phase=.6, root=(.7, 0., .7),
                                reference_root=(0., 0., 1.45))
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [True]
    assert not truncated.any()


def test_simultaneous_failure_and_reference_end_does_not_advance_csi(environment_class):
    env = reference_environment(environment_class, root=(0., 0., .08))
    episode = finish_episode(env)
    assert episode["terminated"] and episode["end_phase"] >= .99
    env.update_curriculum([episode])
    assert env.csi_min_phase.item() == pytest.approx(.95)


def test_successful_reference_end_advances_csi(environment_class):
    env = reference_environment(environment_class, root=(.1, 0., 1.))
    episode = finish_episode(env)
    assert not episode["terminated"]
    env.update_curriculum([episode])
    assert env.csi_min_phase.item() == pytest.approx(.85)


def test_reference_end_with_poor_tracking_does_not_advance_csi(environment_class):
    # This is below the 0.5 m failure threshold, above the 0.3 m CSI threshold.
    env = reference_environment(environment_class, root=(.4, 0., 1.))
    episode = finish_episode(env)
    assert not episode["terminated"]
    env.update_curriculum([episode])
    assert env.csi_min_phase.item() == pytest.approx(.95)


def test_early_reference_failure_demotes_csi(environment_class):
    env = reference_environment(environment_class, step=40, phase=.98,
                                root=(.8, 0., 1.))
    episode = finish_episode(env)
    assert episode["terminated"]
    env.update_curriculum([episode])
    assert env.csi_min_phase.item() == pytest.approx(.95)
