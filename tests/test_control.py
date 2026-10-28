"""CPU control regressions, exercised without starting Isaac Sim.

The checks cover tensor contracts, contact coordinate frames, reference
termination, PPO semantics, shutdown boundaries, and terrain construction.
AST fixtures load real production methods; physical skill success is checked
separately by simulator rollouts.
"""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch
import trimesh
from torch import nn

from morph_mimic.launcher import close
from morph_mimic.networks.tracker import Critic
from morph_mimic.rl.ppo import PPO, PPOCfg
from morph_mimic.rl.storage import RolloutStorage
from morph_mimic.scripts.prepare_motion import prepare_clip
from morph_mimic.utils import checkpoint as cp
from morph_mimic.utils.motion import MotionClip, MotionDataset
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec
from morph_mimic.utils.transforms import (
    encode_goal,
    quat_apply,
    quat_diff_rad,
    to_body_yaw_frame,
    wrap_to_pi,
    yaw_from_quat,
    yaw_quat,
)


# Environment tensor contracts
# Consolidated from test_env_logic.py.

class TestEnvironmentTensorContracts(unittest.TestCase):
    def test_cpu_environment_contracts(self):
        path = Path(__file__).resolve().parents[1] / 'morph_mimic/envs/acp_env.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AcpEnv')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
        scope = dict(torch=torch, np=np, DirectRLEnv=object, RobotSpec=RobotSpec,
                     **{name: globals()[name] for name in ['encode_goal','quat_apply','quat_diff_rad','to_body_yaw_frame','wrap_to_pi','yaw_from_quat','yaw_quat']})
        for n in tree.body:
            if isinstance(n, ast.Assign):
                try:
                    exec(compile(ast.Module(body=[n], type_ignores=[]), str(path), 'exec'), scope)
                except NameError:
                    pass
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
        Env = scope['AcpEnv']
        cfg_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AcpEnvCfg')
        defaults = {}
        for n in cfg_node.body:
            if isinstance(n, ast.AnnAssign) and n.value is not None:
                try:
                    defaults[n.target.id] = ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    pass

        E, B, T = 4, 12, 6
        env = Env.__new__(Env)
        env.cfg = NS(**defaults)
        env.cfg.stage = 1
        env.cfg.planner_scan_noise = env.cfg.tracker_scan_noise = 0.0
        for k in ['obs_noise_angvel','obs_noise_gravity','obs_noise_jpos','obs_noise_jvel']:
            setattr(env.cfg, k, 0.0)
        env.device = 'cpu'
        env.num_envs = E
        env.extras = {}
        env.step_dt = .02
        env.max_episode_length = 750
        env.L, env.W, env.LW, env.TL, env.TW = 17, 11, 187, 9, 7
        env.robot_spec = RobotSpec.load('g1')
        canonical = []
        for group in env.robot_spec.nine_body_groups():
            cands = env.robot_spec.contact_bodies[group]
            if len(cands) == 1:
                canonical.extend(cands)
            else:
                canonical.extend([next(c for c in cands if c.startswith(side)) for side in ['left','right']])
        aliases = {'left_elbow_link':'left_elbow_pitch_link', 'right_elbow_link':'right_elbow_pitch_link'}
        env.robot_spec.asset_aliases = {'bodies': aliases}
        runtime_names = [aliases.get(c,c).replace('rubber_hand','wrist_yaw_link') for c in canonical] + ['extra1','extra2','extra3']
        motion_names = [c.replace('rubber_hand','wrist_yaw_link') for c in canonical]
        root_pos = torch.zeros(E,3)
        root_pos[:,2] = .75
        root_quat = torch.zeros(E,4)
        root_quat[:,0] = 1.
        data = NS(root_pos_w=root_pos,root_quat_w=root_quat,
                  root_lin_vel_b=torch.full((E,3),99.),root_ang_vel_b=torch.zeros(E,3),
                  root_lin_vel_w=torch.full((E,3),99.),root_ang_vel_w=torch.zeros(E,3),
                  root_link_lin_vel_b=torch.zeros(E,3),root_link_lin_vel_w=torch.zeros(E,3),
                  joint_pos=torch.zeros(E,23),joint_vel=torch.zeros(E,23),
                  applied_torque=torch.zeros(E,23),computed_torque=torch.zeros(E,23),
                  body_pos_w=torch.zeros(E,B,3),body_lin_vel_w=torch.full((E,B,3),99.),
                  body_link_lin_vel_w=torch.zeros(E,B,3),body_quat_w=torch.zeros(E,B,4))
        data.body_quat_w[:,:,0] = 1.
        env.torso_body_idx = B-1
        class Robot:
            num_joints = 23
            body_names = runtime_names
            def set_joint_position_target(self,targets,**kwargs):
                self.targets = targets
        env.robot = Robot()
        env.robot.data = data
        env.robot.actuators = {}
        env._resolve_contact_links()
        assert env.contact_motion_candidates[0][-1] == 'left_wrist_yaw_link'
        assert env.contact_link_names[0] == 'left_wrist_yaw_link'
        assert env.contact_link_names[2] == 'left_elbow_pitch_link'
        env.contact_sensor_idx = torch.tensor([3,1,9,0,7,5,8,2,4])
        hist = torch.zeros(E,10,B,3)
        hist[:,4,env.contact_sensor_idx[0],2] = 2.
        hist[:,2,11,2] = 100.
        env.contact_sensor = NS(data=NS(net_forces_w_history=hist, net_forces_w=hist[:,4].clone()))
        contacts = env._contact_measured()
        assert contacts.shape == (E,9)
        assert contacts[:,0].eq(1).all() and contacts[:,1:].eq(0).all()

        grid = torch.stack(torch.meshgrid(torch.linspace(-.8,.8,17),torch.linspace(-.5,.5,11),indexing='ij'),-1).reshape(187,2)
        rays = torch.cat((grid,torch.zeros(187,1)),-1)[None].repeat(E,1,1)
        hits = rays.clone()
        hits[:,5,:] = float('inf')
        env.ray_caster = NS(data=NS(ray_hits_w=hits),ray_starts=rays)
        env.dr_scan_bias = torch.zeros(E,187)
        scan = env._scan_points_body_yaw(noisy=False)
        assert scan.shape == (E,187,3) and torch.isfinite(scan).all()
        assert scan[:,5,2].eq(.75-4).all()
        assert env._tracker_scan().shape == (E,63)
        assert env._planner_heights_relative().shape == (E,187)

        env.controlled_idx = torch.arange(23)
        env.default_pose = torch.zeros(23)
        env.joint_limits = torch.tensor([[-1.,1.]]).repeat(23,1)
        env.vel_limits = torch.ones(23)*10
        env.actions = torch.zeros(E,23)
        env.prev_actions = torch.zeros(E,2,23)
        env.action_delay = torch.tensor([0,1,0,1])
        env.is_ref_env = torch.tensor([True,True,False,False])
        env.motion_data = None
        env.phase = torch.zeros(E)
        env.ref_contact_flags = torch.zeros(E,9)
        env._pre_physics_step(torch.ones(E,23))
        env._apply_action()
        assert env.robot.targets[[0,2]].eq(.25).all()
        assert env.robot.targets[[1,3]].eq(0).all()
        assert (env.actions-env.prev_actions[:,0]).square().sum(-1).eq(23).all()
        env._pre_physics_step(torch.ones(E,23)*2)
        env._apply_action()
        assert env.robot.targets[[0,2]].eq(.5).all()
        assert env.robot.targets[[1,3]].eq(.25).all()
        assert env._compute_proprio()[:,3:6].equal(torch.tensor([[0.,0.,-1.]]).repeat(E,1))

        env.contact_targets = torch.zeros(E,9,3)
        env.contact_target_flags = torch.zeros(E,9)
        env.predicted_contacts = torch.ones(E,9,3)*7
        env.predicted_occupancy = torch.ones(E,9)
        oracle = torch.ones(E,9,3)*3
        env._oracle_contacts = lambda ids: (oracle[ids],torch.ones(len(ids),9))
        env.cfg.eta_obs = env.cfg.eta_ref_obs = 0.
        env._update_contact_targets()
        assert env.contact_targets[:2].eq(3).all() and env.contact_targets[2:].eq(0).all()
        env.cfg.eta_obs = env.cfg.eta_ref_obs = 1.
        env._update_contact_targets()
        assert env.contact_targets.eq(7).all()
        env.contact_targets.zero_()
        env._update_contact_targets_for_reset(torch.arange(E))
        assert env.contact_targets.eq(7).all()

        body_pos = torch.zeros(T,9,3)
        body_pos[:,:,0] = torch.arange(T)[:,None]*.01 + 10
        body_pos[:,:,2] = torch.linspace(.1,.9,9)
        rp = torch.zeros(T,3)
        rp[:,0] = torch.arange(T)*.01+10
        rp[:,2] = .75
        rq = torch.zeros(T,4)
        rq[:,0] = 1
        clip = MotionClip('synthetic','walk',50.,T/50.,rp,rq,torch.zeros(T,3),torch.zeros(T,3),
                          torch.zeros(T,23),torch.zeros(T,23),motion_names,body_pos,torch.zeros(T,9,4))
        class Motion:
            clips = {'synthetic':clip}
            clip_names = ['synthetic']
            contact_flags = {}
            def sample(self,ids,phases):
                return clip.sample(phases)
        env.motion_data = Motion()
        env.clip_id = torch.zeros(E,dtype=torch.long)
        env.clip_duration = torch.ones(E)*clip.duration_s
        env.phase.zero_()
        env.terrain_origins = torch.zeros(10,13,3)
        env.terrain_origins[:,:,0] = 20
        env.terrain_origins[:,:,1] = 30
        env.terrain_type_id = torch.zeros(E,dtype=torch.long)
        env.terrain_level = torch.zeros(E,dtype=torch.long)
        env.cell_offset = torch.zeros(E,3)
        env._contact_clip_body_idx = {}
        ref = env._ref_sample()
        assert ref['root_pos'][:,2].eq(.75).all()
        assert ref['root_pos'][:,0].eq(20).all()
        rp_w = env._ref_contact_pos(False)
        rp_b = env._ref_contact_pos(True)
        assert torch.allclose(rp_w[:2,:,2],body_pos[0,:,2].repeat(2,1))
        assert torch.allclose(rp_w[:2,:,0],torch.ones(2,9)*20)
        assert torch.allclose(rp_b[:2,:,2],body_pos[0,:,2].repeat(2,1))

        env.proprio_history = torch.zeros(E,5,75)
        env.episode_length_buf = torch.ones(E,dtype=torch.long)
        env.ep_record_step = torch.zeros(E,dtype=torch.long)
        env.goal_pos = torch.zeros(E,3)
        env.goal_yaw = torch.zeros(E)
        env.goal_height = torch.ones(E)*.75
        env.ref_lookahead = torch.ones(E)
        env.terrain_group_id = torch.zeros(E,dtype=torch.long)
        env.contact_body_vel = torch.zeros(E,9,3)
        env.prev_contact_body_vel = torch.zeros(E,9,3)
        env.contact_duration = torch.zeros(E,9)
        env.cfg.eta_obs = env.cfg.eta_ref_obs = 0
        obs = env._get_observations()
        history = env.proprio_history.clone()
        env.refresh_observations()
        assert env.proprio_history.equal(history)
        assert obs['policy'].shape == (E,468)
        vm,vfs = env.get_critic_obs()
        assert vm.shape == (E,353) and vfs.shape == (E,298)
        # Legacy root_lin_vel aliases report COM velocity, whereas references
        # and root_pos are defined at the body origin.
        assert vm[:,262:265].eq(0).all() and vfs[:,262:265].eq(0).all()
        reward = env._get_rewards()
        assert reward.shape == (E,) and torch.isfinite(reward).all()
        assert env.contact_body_vel.eq(0).all()

        # Table 3 defines flat orientation using torso-frame gravity. A fixed
        # upright root with a rotated articulated torso must incur its own cost.
        original_weights = {key:value for key,value in vars(env.cfg).items() if key.startswith('w_')}
        for key in original_weights:
            setattr(env.cfg,key,0.)
        env.cfg.w_flat_orientation = 1.
        original_is_ref = env.is_ref_env.clone()
        env.is_ref_env.zero_()
        upright_reward = env._get_rewards()
        angle = torch.tensor(np.pi/4)
        data.body_quat_w[:,env.torso_body_idx,0] = torch.cos(angle/2)
        data.body_quat_w[:,env.torso_body_idx,1] = torch.sin(angle/2)
        tilted_reward = env._get_rewards()
        assert upright_reward.eq(0).all()
        assert torch.allclose(tilted_reward,torch.full((E,),-.5*env.step_dt),atol=1e-7)
        assert data.root_quat_w[:,0].eq(1).all() and data.root_quat_w[:,1:].eq(0).all()
        data.body_quat_w[:,env.torso_body_idx] = torch.tensor([1.,0.,0.,0.])
        env.is_ref_env.copy_(original_is_ref)
        for key,value in original_weights.items():
            setattr(env.cfg,key,value)

        env.finished_episodes = []
        env.ep_record = None
        env.init_phase = torch.zeros(E)
        env.ep_reached_goal = torch.zeros(E,dtype=torch.bool)
        env.ep_min_goal_dist = torch.ones(E)
        env.reset_terminated = torch.zeros(E,dtype=torch.bool)
        env.reset_time_outs = torch.ones(E,dtype=torch.bool)
        env.phase[:] = 1.
        # Completion fixtures represent tracking at the rebased tile origin.
        data.root_pos_w.copy_(env._ref_sample()['root_pos'])
        env._finish_episodes(torch.arange(E))
        assert len(env.finished_episodes) == E and 'record' not in env.finished_episodes[0]
        assert 'goal_pos' in env.finished_episodes[0] and 'clip_id' in env.finished_episodes[0]
        env.csi_min_phase = torch.ones(E)*.9
        env.success_ema = torch.zeros(E)
        env.update_curriculum(env.finished_episodes)
        assert torch.allclose(env.csi_min_phase[:2],torch.tensor([.85,.85]))

        # Replay records the initial frame and the post-physics terminal frame.
        env.cfg.stage = 2
        env.cfg.replay = True
        env.cfg.record_max_steps = T
        env.ep_record = torch.zeros(E,T,scope['RECORD_FIELDS'])
        env.ep_record_step.zero_()
        env.episode_length_buf.zero_()
        env.phase.zero_()
        env.clip_duration[:] = (T-1)/50.
        env.is_ref_env[:] = True
        env._get_observations()
        for frame in range(1,T):
            env._pre_physics_step(torch.zeros(E,23))
            env.episode_length_buf += 1
            terminated,truncated = env._get_dones()
            assert not terminated.any()
            if frame == T-1:
                assert truncated.all()
        assert env.ep_record_step.eq(T).all()
        assert env.phase.eq(1).all()
        assert torch.isfinite(env.ep_record).all()
        assert env.ep_record[:,:,scope['REC_VEL']:scope['REC_VEL']+3].eq(0).all()

        # Evaluation consumes every reference interval and includes terminal
        # metrics captured before auto-reset, even with no replay recording.
        env.cfg.replay = False
        env.cfg.evaluation = True
        env.phase.zero_()
        env.episode_length_buf.zero_()
        for frame in range(1,T):
            env._pre_physics_step(torch.zeros(E,23))
            env.episode_length_buf += 1
            _, truncated = env._get_dones()
            assert bool(truncated.all()) == (frame == T-1)
        final = env.extras['final_tracking']
        assert final['phase'].eq(1).all() and final['reached_end'].all()
        assert final['root_squared_error_m2'].shape == (E,)
        assert final['body_squared_error_m2'].shape == (E,9)
        assert final['joint_squared_error_rad2'].shape == (E,23)
        snapshot = final['root_squared_error_m2'].clone()
        env.robot.data.root_pos_w.fill_(999.)
        assert torch.equal(final['root_squared_error_m2'], snapshot)


# Reward, observation noise, and ACP training semantics
# Consolidated from test_acp_training_semantics.py.

@pytest.fixture
def environment():
    source = Path(__file__).resolve().parents[1] / "morph_mimic/envs/acp_env.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AcpEnv")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls,
    ], type_ignores=[])
    scope = dict(torch=torch, np=np, DirectRLEnv=object, quat_apply=quat_apply,
                 quat_diff_rad=quat_diff_rad, to_body_yaw_frame=to_body_yaw_frame,
                 wrap_to_pi=wrap_to_pi, yaw_from_quat=yaw_from_quat)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
            except NameError:
                pass
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    cfg_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AcpEnvCfg")
    defaults = {}
    for node in cfg_node.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            try:
                defaults[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    env = scope["AcpEnv"].__new__(scope["AcpEnv"])
    env.cfg = SimpleNamespace(**defaults)
    env.num_envs, env.device, env.step_dt = 2, "cpu", .02
    env.extras = {}
    return env


def test_csi_curriculum_phase_reaches_exact_start_and_legacy_retains_random_suffix(environment):
    env = environment
    env.csi_min_phase = torch.tensor([0., .9])
    ids = torch.arange(2)
    torch.testing.assert_close(env._sample_csi_initial_phase(ids), torch.tensor([0., .9]))
    env.cfg.csi_sampling_mode = "uniform_from_curriculum_phase"
    torch.manual_seed(12)
    sampled = env._sample_csi_initial_phase(ids)
    assert 0. < sampled[0] < .99 and .9 < sampled[1] < .99
    env.cfg.csi_sampling_mode = "unknown"
    with pytest.raises(ValueError, match="csi_sampling_mode"):
        env._sample_csi_initial_phase(ids)


def test_contact_acceleration_ignores_reset_velocity_and_keeps_real_differences(environment):
    env = environment
    velocities = torch.full((2, 9, 3), 3.)
    env.robot = SimpleNamespace(data=SimpleNamespace(body_link_lin_vel_w=velocities))
    env.contact_link_idx = torch.arange(9)
    env.contact_body_vel = torch.zeros_like(velocities)
    env.prev_contact_body_vel = torch.zeros_like(velocities)
    assert env._contact_acceleration().eq(0).all()
    assert env._contact_acceleration().eq(0).all()
    # A real 0.02 m/s change in one link over 0.02 s is 1 m/s^2.
    velocities[0, 0, 0] += .02
    torch.testing.assert_close(env._contact_acceleration(), torch.tensor([1., 0.]), atol=1e-5, rtol=1e-5)
    # Resetting one environment must not suppress the other's measurement.
    env._contact_velocity_history_valid[0] = False
    velocities[0].fill_(8.)
    velocities[1, 0, 1] += .04
    torch.testing.assert_close(env._contact_acceleration(), torch.tensor([0., 2.]), atol=1e-5, rtol=1e-5)


def test_tracker_noise_is_independent_bounded_and_does_not_inherit_planner_noise(environment):
    env = environment
    env.L, env.W, env.LW, env.TL, env.TW = 17, 11, 187, 9, 7
    env.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3)))
    noisy_calls = []

    def scan(*, noisy=True):
        noisy_calls.append(noisy)
        result = torch.zeros(2, 187, 3)
        if noisy:
            result[..., 2] = 100.  # Deliberately large planner noise/bias.
        return result

    env._scan_points_body_yaw = scan
    torch.manual_seed(13)
    observed = env._tracker_scan()
    assert noisy_calls == [False]
    assert observed.shape == (2, 63) and observed.abs().max() <= .1
    assert observed[0].std() > .02 and not torch.equal(observed[0], observed[1])
    env.cfg.tracker_scan_noise = 0.
    assert env._tracker_scan().eq(0).all()


def reward_environment(env):
    """An upright stationary reference with independently configurable limits."""
    n = env.num_envs
    zeros3 = torch.zeros(n, 3)
    quat = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    positions = torch.zeros(n, 9, 3)
    data = SimpleNamespace(
        root_pos_w=zeros3.clone(), root_quat_w=quat, root_link_lin_vel_b=zeros3.clone(),
        root_link_lin_vel_w=zeros3.clone(), root_ang_vel_b=zeros3.clone(), root_ang_vel_w=zeros3.clone(),
        joint_pos=torch.zeros(n, 23), joint_vel=torch.zeros(n, 23),
        applied_torque=torch.zeros(n, 23), computed_torque=torch.zeros(n, 23),
        soft_joint_pos_limits=torch.tensor([-0.95, .95]).repeat(n, 23, 1),
        body_pos_w=positions, body_link_lin_vel_w=positions.clone(),
        body_quat_w=quat[:, None, :].repeat(1, 9, 1),
    )
    env.robot = SimpleNamespace(data=data)
    env.controlled_idx = torch.arange(23)
    env.contact_link_idx = torch.arange(9)
    env.default_pose = torch.zeros(23)
    env.joint_limits = torch.tensor([[-1., 1.]]).repeat(23, 1)
    env.vel_limits = torch.ones(23) * 20
    env.is_ref_env = torch.tensor([True, False])
    env.init_phase = torch.tensor([0., .8])
    env.terrain_type_id = torch.zeros(n, dtype=torch.long)
    env.terrain_group_id = torch.zeros(n, dtype=torch.long)
    env.actions = torch.ones(n, 23) * .2
    env.prev_actions = torch.zeros(n, 2, 23)
    env.contact_duration = torch.zeros(n, 9)
    env.contact_body_vel = positions.clone()
    env.prev_contact_body_vel = positions.clone()
    env.ref_contact_flags = torch.zeros(n, 9)
    env.contact_target_flags = torch.zeros(n, 9)
    env.contact_targets = positions.clone()
    env.goal_pos = zeros3.clone()
    env.goal_height = torch.zeros(n)
    env.goal_yaw = torch.zeros(n)
    env.torso_body_idx = 0
    env.goal_command = lambda: zeros3.clone()
    env._ref_contact_pos = lambda body_yaw: positions.clone()
    env._contact_measured = lambda: torch.zeros(n, 9)
    env._contact_force_history = lambda: torch.zeros(n, 10, 9, 3)
    env._ref_sample = lambda: {
        "root_pos": zeros3.clone(), "root_quat": quat.clone(),
        "root_lin_vel": zeros3.clone(), "root_ang_vel": zeros3.clone(),
        "joint_pos": torch.zeros(n, 23), "joint_vel": torch.zeros(n, 23),
    }
    return env


def test_runtime_soft_limits_penalize_before_hard_stop(environment):
    env = reward_environment(environment)
    for name in vars(env.cfg):
        if name.startswith("w_"):
            setattr(env.cfg, name, 0.)
    env.cfg.w_dof_pos_limits = 10.
    env.robot.data.joint_pos[:, 0] = .96
    torch.testing.assert_close(env._get_rewards(), torch.full((2,), -.002), rtol=1e-5, atol=1e-7)
    env.cfg.joint_limit_reward_source = "legacy_spec_hard_limits"
    assert env._get_rewards().eq(0).all()


def test_weighted_telemetry_reconstructs_reward_and_respects_stage_masks(environment):
    env = reward_environment(environment)
    reward = env._get_rewards()
    telemetry = env.extras["reward_terms"]
    assert all(isinstance(value, torch.Tensor) and value.ndim == 0 for value in telemetry.values())
    assert all(not value.requires_grad for value in telemetry.values())
    weighted = [value for name, value in telemetry.items() if name.startswith("weighted/")]
    torch.testing.assert_close(sum(weighted), reward.mean())
    # Reference task terms apply to half the environments in this fixture.
    torch.testing.assert_close(telemetry["weighted/track_base_pos"], torch.tensor(3. * .5 * .02))
    assert telemetry["diagnostic/reference_initial_phase_mean"] == 0
    assert telemetry["diagnostic/reference_phase_zero_initialization_fraction"] == 1


# Critic coordinate frames and PPO clipping
# Consolidated from test_critic_training_semantics.py.

def _critic_environment_class():
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
    env = _critic_environment_class().__new__(_critic_environment_class())
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


# Reference termination and CSI curriculum
# Consolidated from test_reference_curriculum.py.

@pytest.fixture
def curriculum_environment_class():
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


def reference_environment(curriculum_environment_class, *, step=45, phase=.991,
                          root=(0., 0., 1.), reference_root=(0., 0., 1.)):
    env = curriculum_environment_class.__new__(curriculum_environment_class)
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


def test_fallen_reference_suffix_terminates_before_goal_grace_ends(curriculum_environment_class):
    # The old 75-step grace covered the entire initial 0.9-to-0.99 suffix.
    # Root error stays below 0.5 here, so low height must detect this failure.
    env = reference_environment(curriculum_environment_class, root=(0., 0., .08),
                                reference_root=(0., 0., .4))
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [True]
    assert truncated.tolist() == [True]
    assert env.episode_length_buf.item() < env.cfg.grace_steps


def test_goal_environment_retains_its_longer_grace(curriculum_environment_class):
    env = reference_environment(curriculum_environment_class, root=(0., 0., .08))
    env.is_ref_env.zero_()
    terminated, truncated = env._get_dones()
    assert not terminated.any()
    assert not truncated.any()


def test_reference_root_lost_detects_fall_on_elevated_support(curriculum_environment_class):
    # Falling onto a box can leave the root above the absolute 0.3 m limit.
    env = reference_environment(curriculum_environment_class, phase=.6, root=(.7, 0., .7),
                                reference_root=(0., 0., 1.45))
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [True]
    assert not truncated.any()


def test_simultaneous_failure_and_reference_end_does_not_advance_csi(curriculum_environment_class):
    env = reference_environment(curriculum_environment_class, root=(0., 0., .08))
    episode = finish_episode(env)
    assert episode["terminated"] and episode["end_phase"] >= .99
    env.update_curriculum([episode])
    assert env.csi_min_phase.item() == pytest.approx(.95)


def test_successful_reference_end_advances_csi(curriculum_environment_class):
    env = reference_environment(curriculum_environment_class, root=(.1, 0., 1.))
    episode = finish_episode(env)
    assert not episode["terminated"]
    env.update_curriculum([episode])
    assert env.csi_min_phase.item() == pytest.approx(.85)


def test_reference_end_with_poor_tracking_does_not_advance_csi(curriculum_environment_class):
    # This is below the 0.5 m failure threshold, above the 0.3 m CSI threshold.
    env = reference_environment(curriculum_environment_class, root=(.4, 0., 1.))
    episode = finish_episode(env)
    assert not episode["terminated"]
    env.update_curriculum([episode])
    assert env.csi_min_phase.item() == pytest.approx(.95)


def test_early_reference_failure_demotes_csi(curriculum_environment_class):
    env = reference_environment(curriculum_environment_class, step=40, phase=.98,
                                root=(.8, 0., 1.))
    episode = finish_episode(env)
    assert episode["terminated"]
    env.update_curriculum([episode])
    assert env.csi_min_phase.item() == pytest.approx(.95)


# Oracle contact coordinates and label metadata
# Consolidated from test_oracle_contact_frame.py.

def _oracle_method():
    path = Path(__file__).resolve().parents[1] / "morph_mimic/envs/acp_env.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "AcpEnv")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_oracle_contacts")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             method], type_ignores=[])
    scope = dict(torch=torch, to_body_yaw_frame=to_body_yaw_frame, yaw_from_quat=yaw_from_quat)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["_oracle_contacts"]


class TestOracleContactFrame(unittest.TestCase):
    def fixture(self):
        roots = torch.tensor([[3., -2., 1.], [3.2, -1.8, 1.2], [3.5, -1.4, 1.4]])
        yaw = torch.tensor([0., torch.pi / 3, torch.pi / 2])
        clip = NS(root_pos=roots, root_quat=yaw_quat(yaw))
        anchors = torch.tensor([3.8, -1.3, .6])[None, :].repeat(9, 1)
        anchors[:, 0] += torch.arange(9) * .015
        # One fixed source-world contact anchor is encoded using each discrete
        # query frame. The physical target must stay fixed between frames too.
        labels = to_body_yaw_frame(anchors[None].repeat(3, 1, 1), roots[:, None], yaw[:, None])
        flags = torch.ones(3, 9)
        flags[:, 1] = 0.
        labels[:, 1] = 1e5
        origins = torch.tensor([[0., 0., 0.], [20., -30., .2], [-50., 80., 0.], [100., 50., .5]])
        offsets = torch.tensor([[.1, -.1, 0.], [.2, 0., 0.], [0., .3, 0.], [-.2, -.2, 0.]])
        actual_root = origins + torch.tensor([.4, -.2, 9.])
        actual_yaw = torch.tensor([0., torch.pi / 2, -torch.pi / 2, torch.pi])
        motion = NS(clip_names=["reference"], clips={"reference": clip},
                    contact_targets={"reference": (labels, flags)},
                    contact_target_frames={"reference": "reference_query_body_yaw_absolute_z"})
        env = NS(device="cpu", num_envs=4, motion_data=motion, clip_id=torch.zeros(4, dtype=torch.long),
                 phase=torch.tensor([.25, .5, .75, 1.]), cell_offset=offsets,
                 env_origin=lambda: origins, robot=NS(data=NS(root_pos_w=actual_root, root_quat_w=yaw_quat(actual_yaw))))
        expected_world = anchors[None] - torch.tensor([*roots[0, :2], 0.])[None, None] + origins[:, None] + offsets[:, None]
        expected = to_body_yaw_frame(expected_world, actual_root[:, None], actual_yaw[:, None])
        expected[:, 1] = 0.
        return env, expected, flags

    def test_fixed_world_anchor_is_reexpressed_in_actual_robot_yaw_and_keeps_absolute_height(self):
        env, expected, flags = self.fixture()
        targets, actual_flags = _oracle_method()(env, torch.arange(4))
        torch.testing.assert_close(targets, expected, atol=1e-5, rtol=0.)
        torch.testing.assert_close(actual_flags, flags[0].expand(4, -1))
        torch.testing.assert_close(targets[:, 0, 2], torch.tensor([.6, .8, .6, 1.1]))
        self.assertTrue(torch.equal(targets[:, 1], torch.zeros(4, 3)))

    def test_label_uses_its_discrete_query_pose_and_does_not_slide_at_fractional_phase(self):
        env, expected, _ = self.fixture()
        method = _oracle_method()
        for phase in (.001, .249, .499, .501, .749, .999):
            with self.subTest(phase=phase):
                env.phase.fill_(phase)
                actual, _ = method(env, torch.arange(4))
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.)

    def test_subset_reset_order_and_multiple_clip_frames_do_not_mix_environments(self):
        env, expected, _ = self.fixture()
        legacy_targets = torch.full((3, 9, 3), .123)
        legacy_flags = torch.ones(3, 9)
        env.motion_data.clip_names.append("legacy")
        env.motion_data.contact_targets["legacy"] = legacy_targets, legacy_flags
        env.motion_data.contact_target_frames["legacy"] = "legacy_query_body_yaw_absolute_z_unverified"
        env.clip_id[2] = 1
        ids = torch.tensor([3, 2, 0])
        targets, flags = _oracle_method()(env, ids)
        torch.testing.assert_close(targets[[0, 2]], expected[[3, 0]], atol=1e-5, rtol=0.)
        torch.testing.assert_close(targets[1], legacy_targets[0])
        torch.testing.assert_close(flags[1], legacy_flags[0])

    def test_unidentified_legacy_frame_is_preserved_and_missing_labels_are_zero(self):
        env, _, _ = self.fixture()
        del env.motion_data.contact_target_frames
        labels, _ = env.motion_data.contact_targets["reference"]
        targets, _ = _oracle_method()(env, torch.tensor([0, 3]))
        # Unknown legacy files lack their generating query poses. Preserve their
        # old behavior instead of pretending reference inversion is exact.
        torch.testing.assert_close(targets[0], labels[0])
        torch.testing.assert_close(targets[1], labels[2])
        env.motion_data.contact_targets.clear()
        targets, flags = _oracle_method()(env, torch.tensor([3, 0]))
        self.assertTrue(torch.equal(targets, torch.zeros(2, 9, 3)))
        self.assertTrue(torch.equal(flags, torch.zeros(2, 9)))

    def test_unknown_bound_scene_frame_is_refused_instead_of_guessed(self):
        env, _, _ = self.fixture()
        env.motion_data.scene_bindings = {"reference": object()}
        env.motion_data.contact_target_frames.clear()
        with self.assertRaisesRegex(ValueError, "known reference-query"):
            _oracle_method()(env, torch.tensor([0]))


SEED = PROJECT_ROOT / "datasets/g1_box_climb_seed_v2"
CLIP = "climb_03_z_scale_1.0_scene_fit_0000_0501"


@unittest.skipUnless((SEED / f"{CLIP}_contacts.npz").is_file(), "box reference label fixture is absent")
class TestOracleContactFrameMetadata(unittest.TestCase):
    def test_existing_geometry_labels_are_recognized_without_modifying_their_bytes(self):
        path = SEED / f"{CLIP}_contacts.npz"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        motion = MotionDataset(RobotSpec.load("g1"), SEED, clip_names=[CLIP])
        self.assertEqual(motion.contact_target_frames[CLIP], "reference_query_body_yaw_absolute_z")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_scene_bound_unknown_or_unsupported_label_frame_is_rejected(self):
        with np.load(SEED / f"{CLIP}_contacts.npz", allow_pickle=False) as archive:
            labels = {key: archive[key] for key in archive.files}
        labels.pop("method", None)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for suffix in (".npz", ".scene.json"):
                shutil.copy2(SEED / f"{CLIP}{suffix}", directory / f"{CLIP}{suffix}")
            np.savez(directory / f"{CLIP}_contacts.npz", **labels)
            with self.assertRaisesRegex(ValueError, "bound scene training requires reference-query"):
                MotionDataset(RobotSpec.load("g1"), directory, clip_names=[CLIP])
            labels["contact_target_frame"] = np.array("world_xyz")
            np.savez(directory / f"{CLIP}_contacts.npz", **labels)
            with self.assertRaisesRegex(ValueError, "unsupported contact target coordinate frame"):
                MotionDataset(RobotSpec.load("g1"), directory, clip_names=[CLIP])

    def test_legacy_unbound_labels_keep_behavior_with_an_explicit_warning(self):
        with np.load(SEED / f"{CLIP}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["requires_scene"] = np.array(False)
        with np.load(SEED / f"{CLIP}_contacts.npz", allow_pickle=False) as archive:
            labels = {key: archive[key] for key in archive.files}
        labels.pop("method", None)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / f"{CLIP}.npz"
            np.savez(path, **arrays)
            labels["source_sha256"] = np.array(hashlib.sha256(path.read_bytes()).hexdigest())
            np.savez(directory / f"{CLIP}_contacts.npz", **labels)
            with self.assertWarnsRegex(RuntimeWarning, "cannot reproject.*preserving legacy unbound behavior"):
                motion = MotionDataset(RobotSpec.load("g1"), directory, clip_names=[CLIP])
            self.assertEqual(motion.contact_target_frames[CLIP], "legacy_query_body_yaw_absolute_z_unverified")


# Shutdown status and scene conversion boundaries
# Consolidated from test_runtime_boundaries.py.

class RecordingStream(io.StringIO):
    def __init__(self, events, name):
        super().__init__()
        self.events, self.name = events, name

    def flush(self):
        self.events.append("flush_" + self.name)
        super().flush()


class FakeApp:
    def __init__(self, events):
        self.events = events
        self.app = self
        self.quit_codes = []

    def post_quit(self, status):
        self.quit_codes.append(status)
        self.events.append("post_quit")

    def close(self):
        self.events.append("close")


class TestShutdownStatus(unittest.TestCase):
    def fixture(self):
        events = []
        return events, FakeApp(events), RecordingStream(events, "stdout"), RecordingStream(events, "stderr")

    def test_success_flushes_logs_before_close_without_requesting_failure(self):
        events, app, stdout, stderr = self.fixture()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print("validation completed")
            close(app)
        self.assertEqual(app.quit_codes, [])
        self.assertEqual(events[-3:], ["flush_stdout", "flush_stderr", "close"])
        self.assertEqual(stdout.getvalue(), "validation completed\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_exception_is_reported_and_preserved_with_failure_status(self):
        events, app, stdout, stderr = self.fixture()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(ValueError, "broken validation"):
                try:
                    raise ValueError("broken validation")
                finally:
                    close(app)
        self.assertEqual(app.quit_codes, [1])
        self.assertIn("ValueError: broken validation", stderr.getvalue())
        self.assertEqual(events[-3:], ["flush_stdout", "flush_stderr", "close"])

    def test_system_exit_matches_python_success_and_error_status_semantics(self):
        for argument, expected in [(None, 0), (0, 0), (7, 7), ("replay failed", 1)]:
            with self.subTest(argument=argument):
                events, app, stdout, stderr = self.fixture()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        try:
                            raise SystemExit(argument)
                        finally:
                            close(app)
                self.assertEqual(caught.exception.code, argument)
                self.assertEqual(app.quit_codes, [expected])
                self.assertEqual(events[-3:], ["flush_stdout", "flush_stderr", "close"])


class TestSceneConversionRefusal(unittest.TestCase):
    def source(self, directory, marker):
        source = Path(directory) / "walk_bound.npz"
        np.savez(source, requires_scene=np.array(marker), fps=np.array([50.]), joint_pos=np.zeros((3, 29)))
        return source

    def check_refusal(self, converter):
        for sidecar in (False, True):
            with self.subTest(sidecar=sidecar), tempfile.TemporaryDirectory() as directory:
                # Either marker independently requires scene-aware handling.
                source = self.source(directory, marker=not sidecar)
                if sidecar:
                    source.with_suffix(".scene.json").write_text('{"schema":1}')
                original = hashlib.sha256(source.read_bytes()).hexdigest()
                output = Path(directory) / "output"
                with self.assertRaisesRegex(ValueError, "scene-bound references require scene-aware conversion"):
                    converter(source, output)
                self.assertFalse(output.exists())
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)

    def test_generic_crop_refuses_either_scene_binding_marker_without_writing(self):
        self.check_refusal(lambda source, output: prepare_clip(source, output, 0., .02))

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is needed before projection input checks")
    def test_generic_projection_refuses_either_scene_binding_marker_without_writing(self):
        from morph_mimic.scripts.project_reference import project_clip
        from morph_mimic.utils.specs import RobotSpec

        spec = RobotSpec.load("g1")
        self.check_refusal(lambda source, output: project_clip(source, output, spec))


class TestEvaluationFastShutdown(unittest.TestCase):
    def test_complete_report_is_published_before_kit_fast_shutdown(self):
        from morph_mimic.scripts import evaluate

        class FakeActor:
            def to(self, device):
                return self

            def eval(self):
                return self

            def load_state_dict(self, state):
                self.loaded_state = state

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "final.pt"
            checkpoint_path.write_bytes(b"test checkpoint bytes for provenance")
            output = root / "reports/evaluation.json"
            args = NS(robot="g1", stage=1, motion_dir=root / "motions", clips=None,
                      resume=checkpoint_path, device="cpu", planner="acp", headless=True,
                      seed=13, out=output)
            checkpoint = {"robot": "g1", "stage": 1, "ppo": {"actor": {"fixture": 1}}}
            metadata = [{"name": "complete_clip"}, {"name": "failed_clip"}]
            motions = [
                dict(name="complete_clip", interrupted=False, budget_limited=False,
                     contract_verified=True, completed=True, failure=False, end_phase=1., root_position_rmse_m=.01),
                dict(name="failed_clip", interrupted=False, budget_limited=False,
                     contract_verified=True, completed=False, failure=True, end_phase=.152, root_position_rmse_m=.4),
            ]
            app = object()
            reports_seen_at_close = []

            def fast_shutdown(closing_app):
                self.assertIs(closing_app, app)
                # A real Kit fast close can terminate Python immediately.
                # Read the actual published JSON here, before raising exit.
                self.assertTrue(output.is_file(), "evaluation report must exist before close")
                report = json.loads(output.read_text())
                self.assertEqual(report["motions"], motions)
                self.assertEqual(report["expected_clips"], 2)
                self.assertEqual(report["evaluated_clips"], 2)
                self.assertTrue(report["evaluation_is_full"])
                self.assertTrue(report["contract_verified"])
                self.assertEqual(report["completion_rate"], .5)
                self.assertEqual(report["failures"], 1)
                self.assertEqual(report["checkpoint_sha256"], hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
                self.assertEqual(report["seed"], 13)
                self.assertIn("bound scene", report["scope"])
                reports_seen_at_close.append(report)
                raise SystemExit(0)

            with patch.object(evaluate, "_arguments", return_value=args), \
                 patch.object(evaluate, "_clip_metadata", return_value=metadata), \
                 patch.object(evaluate.torch, "load", return_value=checkpoint), \
                 patch("morph_mimic.networks.tracker.TrackerActor", FakeActor), \
                 patch.object(evaluate, "_evaluate_clip", side_effect=motions) as evaluate_clip, \
                 patch("morph_mimic.launcher.launch", return_value=app), \
                 patch("morph_mimic.launcher.close", side_effect=fast_shutdown), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    evaluate.main()
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(evaluate_clip.call_count, 2)
            self.assertEqual(len(reports_seen_at_close), 1)
            self.assertEqual(json.loads(output.read_text()), reports_seen_at_close[0])


# Terrain mesh geometry
# Consolidated from test_terrain_geometry.py.

def _load_generators():
    path = Path(__file__).resolve().parents[1] / "morph_mimic/envs/terrains.py"
    module = ast.parse(path.read_text())
    configurations = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields = {"size": (9.0, 6.0)}
        function_name = None
        for field in node.body:
            if isinstance(field, ast.AnnAssign) and field.value is not None:
                fields[field.target.id] = ast.literal_eval(field.value)
            elif isinstance(field, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "function"
                for target in field.targets
            ):
                function_name = field.value.id
        if function_name is not None:
            configurations[function_name] = fields

    module.body = [
        node
        for node in module.body
        if not isinstance(node, ast.ClassDef)
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("isaaclab")
        )
    ]
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["TERRAIN_FUNCS"], configurations


class TestTerrainGeometry(unittest.TestCase):
    def test_all_terrain_types_produce_flat_concatenable_mesh_lists(self):
        functions, configurations = _load_generators()
        for terrain_name, function in functions.items():
            fields = dict(configurations[function.__name__])
            if terrain_name == "walled_stairs":
                fields["walled"] = True
            if terrain_name.startswith("stones_"):
                fields["layout"] = terrain_name.removeprefix("stones_")
            for difficulty in (0.0, 0.5, 1.0):
                with self.subTest(terrain=terrain_name, difficulty=difficulty):
                    meshes, origin = function(difficulty, SimpleNamespace(**fields))
                    self.assertIsInstance(meshes, list)
                    self.assertTrue(meshes)
                    for mesh in meshes:
                        self.assertIsInstance(mesh, trimesh.Trimesh)
                        self.assertGreater(len(mesh.faces), 0)
                    combined = trimesh.util.concatenate(meshes)
                    self.assertTrue(np.isfinite(combined.vertices).all())
                    self.assertEqual(origin.shape, (3,))
                    self.assertTrue(np.isfinite(origin).all())
                    for wall in meshes[-4:]:
                        np.testing.assert_allclose(wall.bounds[:, 2], [0.0, 0.4], atol=1e-8)

    def test_obstacle_surfaces_match_configured_height(self):
        functions, configurations = _load_generators()
        names = (
            "platform", "beam", "pallet", "stones_grid", "stones_paired",
            "stones_double", "box", "uneven_climb", "vault",
        )
        for terrain_name in names:
            function = functions[terrain_name]
            fields = dict(configurations[function.__name__])
            if terrain_name.startswith("stones_"):
                fields["layout"] = terrain_name.removeprefix("stones_")
            for difficulty in (0.0, 0.5, 1.0):
                with self.subTest(terrain=terrain_name, difficulty=difficulty):
                    meshes, _ = function(difficulty, SimpleNamespace(**fields))
                    low, high = fields["height"]
                    height = low + difficulty * (high - low)
                    for obstacle in meshes[:-4]:
                        self.assertAlmostEqual(obstacle.bounds[0, 2], 0.0)
                        if terrain_name == "uneven_climb":
                            self.assertGreaterEqual(obstacle.bounds[1, 2], 0.55 * height)
                            self.assertLessEqual(obstacle.bounds[1, 2], height)
                        else:
                            self.assertAlmostEqual(obstacle.bounds[1, 2], height)

    def test_stairs_have_configured_rise_and_level_landing(self):
        functions, configurations = _load_generators()
        for terrain_name in ("stairs", "walled_stairs"):
            function = functions[terrain_name]
            fields = dict(configurations[function.__name__])
            fields["walled"] = terrain_name == "walled_stairs"
            for difficulty in (0.0, 0.5, 1.0):
                with self.subTest(terrain=terrain_name, difficulty=difficulty):
                    meshes, _ = function(difficulty, SimpleNamespace(**fields))
                    low, high = fields["step_height"]
                    rise = low + difficulty * (high - low)
                    step_count = len(meshes) - (7 if fields["walled"] else 5)
                    self.assertGreater(step_count, 0)
                    for index, step in enumerate(meshes[:step_count]):
                        np.testing.assert_allclose(
                            step.bounds[:, 2], [0.0, (index + 1) * rise], atol=1e-8
                        )
                    landing = meshes[step_count]
                    self.assertAlmostEqual(landing.bounds[1, 2], step_count * rise)

    def test_gap_floor_is_below_support_surfaces(self):
        functions, configurations = _load_generators()
        function = functions["gap"]
        for difficulty in (0.0, 0.5, 1.0):
            with self.subTest(difficulty=difficulty):
                meshes, _ = function(
                    difficulty, SimpleNamespace(**configurations[function.__name__])
                )
                left, right, pit = meshes[:3]
                self.assertLess(pit.bounds[1, 2], min(left.bounds[1, 2], right.bounds[1, 2]) - 0.5)
                self.assertAlmostEqual(pit.bounds[1, 2], -1.0)
