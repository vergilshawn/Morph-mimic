"""CPU regression checks of actual AcpEnv methods, without importing Isaac Sim.

The class is extracted through Python's AST and exercised with tensor fixtures
that follow the installed Isaac Lab sensor layouts. This checks indexing,
coordinate frames, and stage bookkeeping; it does not validate physics, sensor
initialization, USD/URDF import, or GPU execution. Run the Sim smoke separately.
"""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from morph_mimic.utils.motion import MotionClip
from morph_mimic.utils.specs import RobotSpec
from morph_mimic.utils.transforms import (
    encode_goal, quat_apply, quat_diff_rad, to_body_yaw_frame, wrap_to_pi,
    yaw_from_quat, yaw_quat,
)


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



if __name__ == "__main__":
    unittest.main()
