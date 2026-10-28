"""Foundation regressions: frames, robot contracts, contact labels and planners.

Tests needing MuJoCo keep an explicit dependency skip. Planner and replay cases
are also selectable as the optional planner suite from the public check entry.
"""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn

from morph_mimic.contacts.labeling import hindsight_contacts
from morph_mimic.contacts.replay import HindsightBuffer, gaussian_targets, planner_loss
from morph_mimic.diagnostics.audit_runtime_reference import inertia_in_link_frame
from morph_mimic.networks.acp import ACPPlanner, MLPPlanner
from morph_mimic.networks.tracker import Critic, TrackerActor
from morph_mimic.utils import explicit_mjcf
from morph_mimic.utils.explicit_mjcf import checked_explicit_mjcf, export_explicit_mjcf
from morph_mimic.utils.held_joints import apply_held_joint_mode, held_joint_deviation
from morph_mimic.utils.motion import validate_contact_labels
from morph_mimic.utils.transforms import (
    encode_goal, quat_apply, to_body_yaw_frame, yaw_from_quat, yaw_quat,
)


ROOT = Path(__file__).resolve().parents[1]
NS = SimpleNamespace



# Core.

class TestTransforms(unittest.TestCase):
    def test_yaw_roundtrip(self):
        yaw = torch.tensor([0.0, 0.5, -2.0, 3.1])
        q = yaw_quat(yaw)
        self.assertTrue(torch.allclose(yaw_from_quat(q), yaw, atol=1e-5))

    def test_quat_apply_identity(self):
        q = torch.tensor([[1.0, 0, 0, 0], yaw_quat(torch.tensor(1.0))])
        v = torch.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]])
        out = quat_apply(q, v)
        self.assertTrue(torch.allclose(out[0], v[0]))
        self.assertTrue(torch.allclose(out[1], torch.tensor([0.5403023, 0.8414710, 0.0]), atol=1e-6))

    def test_body_yaw_frame_keeps_world_z(self):
        p = torch.tensor([[1.0, 0.0, 0.5], [0.0, 2.0, 1.2]])
        base = torch.tensor([[0.5, 0.5, 0.8]])
        yaw = torch.tensor([0.0])
        out = to_body_yaw_frame(p, base, yaw)
        self.assertAlmostEqual(out[0, 2].item(), 0.5)
        self.assertAlmostEqual(out[1, 2].item(), 1.2)
        self.assertAlmostEqual(out[0, 0].item(), 0.5)

    def test_goal_encoding_radius(self):
        base = torch.zeros(1, 3)
        yaw = torch.zeros(1)
        goal = torch.tensor([[0.05, 0.0, 0.0]])
        g = encode_goal(base, yaw, goal, torch.zeros(1), 0.5)
        # direction magnitude shrinks inside the radius
        self.assertLess(g[0, 0].item(), 0.2)


class TestHindsight(unittest.TestCase):
    def test_labeling(self):
        T, B = 200, 2
        forces = torch.zeros(T, B)
        forces[50:120, 0] = 5.0  # valid segment >= min_seg
        forces[130:135, 1] = 5.0  # too short -> discarded
        forces[150:200, 1] = 5.0
        pos = torch.zeros(T, B, 3)
        pos[:, 0, 0] = torch.arange(T) * 0.01
        pos[:, 1, 0] = torch.arange(T) * 0.01
        base = torch.zeros(T, 3)
        base[:, 0] = torch.arange(T) * 0.01
        yaw = torch.zeros(T)
        targets, flags, c_inst = hindsight_contacts(forces, pos, base, yaw, horizon=75)
        self.assertEqual(targets.shape, (T, B, 3))
        # at t=0, body 0's next contact onset is t=50 (within 75) -> flag
        self.assertTrue(flags[0, 0])
        # body 1 at t=0: next valid onset at 150 (>75) -> no flag
        self.assertFalse(flags[0, 1])
        # short segment excluded from instantaneous flags
        self.assertFalse(c_inst[130:135, 1].any())
        self.assertTrue(c_inst[50:120, 0].all())


class TestPlanner(unittest.TestCase):
    def test_acp_shapes_and_param_count(self):
        b = 8
        planner = ACPPlanner()
        n_params = sum(p.numel() for p in planner.parameters())
        points = torch.randn(b, 187, 3)
        heights = torch.randn(b, 17, 11)
        pg = torch.randn(b, 378)
        aff, occ, contacts = planner(points, heights, pg)
        self.assertEqual(aff.shape, (b, 9, 187))
        self.assertEqual(occ.shape, (b, 9))
        self.assertEqual(contacts.shape, (b, 9, 3))
        self.assertTrue(torch.allclose(aff.sum(-1), torch.ones(b, 9), atol=1e-4))
        self.assertTrue(abs(n_params - 786_993) < 2000, f"param count {n_params} deviates from paper")


    def test_occupancy_gating(self):
        planner = ACPPlanner()
        points = torch.randn(4, 187, 3)
        heights = torch.randn(4, 17, 11)
        pg = torch.randn(4, 378)
        _, occ, contacts = planner(points, heights, pg)
        masked = contacts[(occ <= 0.5)]
        self.assertTrue((masked == 0).all())

    def test_planner_loss(self):
        b = 16
        aff = torch.rand(b, 9, 187)
        aff = aff / aff.sum(-1, keepdim=True)
        occ = torch.rand(b, 9).sigmoid()
        p_star = gaussian_targets(torch.randn(b, 187, 3), torch.randn(b, 9, 3), (torch.rand(b, 9) > 0.5).float())
        c_star = (torch.rand(b, 9) > 0.6).float()
        loss, terms = planner_loss(aff, occ, p_star, c_star)
        self.assertTrue(loss.isfinite())
        self.assertIn("affordance", terms)


class TestTracker(unittest.TestCase):
    def test_shapes(self):
        actor = TrackerActor()
        critic_m = Critic(353)
        critic_fs = Critic(298)
        obs = torch.randn(16, 468)
        self.assertEqual(actor(obs).shape, (16, 23))
        self.assertEqual(critic_m(torch.randn(16, 353)).shape, (16,))
        self.assertEqual(critic_fs(torch.randn(16, 298)).shape, (16,))


class TestBuffer(unittest.TestCase):
    def test_add_sample(self):
        buf = HindsightBuffer(cap_steps_per_partition=2000)
        for i in range(3):
            k = 100
            buf.add(
                f"box/tile{i}",
                {
                    "points": torch.randn(k, 187, 3),
                    "heights": torch.randn(k, 17, 11),
                    "pg": torch.randn(k, 378),
                    "targets": torch.randn(k, 9, 3),
                    "flags": torch.rand(k, 9) > 0.5,
                },
                quality=float(i),
            )
        self.assertEqual(buf.size_steps(), 300)
        s = buf.sample(64, torch.device("cpu"))
        self.assertIsNotNone(s)
        self.assertEqual(s["points"].shape, (64, 187, 3))
        # eviction keeps the best (lowest) quality episodes
        buf2 = HindsightBuffer(cap_steps_per_partition=150)
        for i in range(5):
            buf2.add("box/t", {"points": torch.randn(100, 187, 3), "heights": torch.randn(100, 17, 11),
                               "pg": torch.randn(100, 378), "targets": torch.randn(100, 9, 3),
                               "flags": torch.rand(100, 9) > 0.5}, quality=float(i))
        self.assertLessEqual(buf2.size_steps(), 150)
        self.assertEqual(buf2.parts["box/t"][0]["quality"], 0.0)  # best kept


# Contact labels.

class TestContactLabelValidation(unittest.TestCase):
    def labels(self, frames=20):
        return dict(
            contact_flags=np.zeros((frames, 9), dtype=bool),
            contact_targets=np.zeros((frames, 9, 3), dtype=np.float32),
            contact_target_flags=np.zeros((frames, 9), dtype=bool),
            fps=np.array([50.0]),
            source_num_frames=np.array(frames),
            source_init_phase=np.array(0.0),
            source_end_phase=np.array(1.0),
        )

    def test_complete_replay(self):
        validate_contact_labels(self.labels(), 20, 50.0)

    def test_failed_replay_cannot_stretch_over_source_motion(self):
        with self.assertRaisesRegex(ValueError, "complete motion"):
            validate_contact_labels(self.labels(10), 20, 50.0)

    def test_wrong_sampling_frequency(self):
        labels = self.labels()
        labels["fps"] = np.array([30.0])
        with self.assertRaisesRegex(ValueError, "fps"):
            validate_contact_labels(labels, 20, 50.0)

    def test_nonfinite_contact_target(self):
        labels = self.labels()
        labels["contact_targets"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_contact_labels(labels, 20, 50.0)

    def test_nonbinary_occupancy(self):
        labels = self.labels()
        labels["contact_target_flags"] = np.full((20, 9), 0.5)
        with self.assertRaisesRegex(ValueError, "binary"):
            validate_contact_labels(labels, 20, 50.0)

    def test_full_sized_record_with_wrong_phase_is_rejected(self):
        labels = self.labels()
        labels["source_end_phase"] = np.array(0.8)
        with self.assertRaisesRegex(ValueError, "ended before"):
            validate_contact_labels(labels, 20, 50.0)


# Held joint constraints.

class TestHeldJointConstraints(unittest.TestCase):
    def fixture(self):
        # Deliberately use different spec/runtime orders and a nonzero nominal.
        names = ["wrist_joint", "controlled_joint", "waist_joint"]
        hard = torch.tensor([[[-2., 2.], [-1., 1.], [-.5, .5]]]).repeat(2, 1, 1)
        data = NS(joint_names=names, joint_pos_limits=hard.clone(), soft_joint_pos_limits=hard.clone(),
                  joint_pos=torch.tensor([[.2, .0, .0], [.21, .0, -.01]]),
                  joint_vel=torch.tensor([[.0, .0, .0], [.1, .0, -.2]]))
        backend = hard.clone()

        def set_limits(limits, joint_ids, warn_limit_violation):
            data.joint_pos_limits[:, joint_ids] = limits
            data.soft_joint_pos_limits[:, joint_ids] = limits
            backend[:, joint_ids] = limits

        robot = NS(data=data, write_joint_position_limit_to_sim=set_limits,
                   root_physx_view=NS(get_dof_limits=lambda: backend.clone()))
        return NS(cfg=NS(held_joint_mode="hard_limits_at_nominal"), robot=robot, device="cpu", num_envs=2,
                  robot_spec=NS(held_joint_names=["waist_joint", "wrist_joint"], asset_aliases={}),
                  held_idx=torch.tensor([0, 2]), controlled_idx=torch.tensor([1]),
                  default_pose=torch.tensor([.2, .0, .0]), joint_limits=hard[0].clone())

    def test_only_declared_held_limits_change_and_no_states_are_teleported(self):
        env = self.fixture()
        initial_position = env.robot.data.joint_pos.clone()
        initial_velocity = env.robot.data.joint_vel.clone()
        report = apply_held_joint_mode(env)
        self.assertEqual(report["runtime_joint_indices"], [2, 0])
        self.assertTrue(report["physx_limit_readback_verified"])
        self.assertFalse(report["dynamic_constraint_compliance_validated"])
        torch.testing.assert_close(env.robot.data.joint_pos_limits[:, 1], torch.tensor([[-1., 1.], [-1., 1.]]))
        torch.testing.assert_close(env.robot.data.joint_pos_limits[:, 0], torch.full((2, 2), .2))
        torch.testing.assert_close(env.joint_limits[2], torch.zeros(2))
        torch.testing.assert_close(env.robot.data.joint_pos, initial_position)
        torch.testing.assert_close(env.robot.data.joint_vel, initial_velocity)

    def test_nominal_pd_keeps_legacy_physics(self):
        env = self.fixture()
        del env.cfg.held_joint_mode
        before = env.robot.data.joint_pos_limits.clone()
        self.assertFalse(apply_held_joint_mode(env)["physical_limits_changed"])
        torch.testing.assert_close(env.robot.data.joint_pos_limits, before)

    def test_refuses_unknown_mode_unexplained_joints_or_invalid_nominal(self):
        for failure in ("unknown", "extra", "nominal", "degenerate"):
            env = self.fixture()
            if failure == "unknown":
                env.cfg.held_joint_mode = "silent_gain_increase"
            elif failure == "extra":
                env.robot_spec.held_joint_names.pop()
            elif failure == "nominal":
                env.default_pose[2] = 99.
            else:
                env.robot.data.joint_pos_limits[:, 2] = 0.
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                apply_held_joint_mode(env)

    def test_refuses_backend_ignoring_limit_update(self):
        env = self.fixture()
        original = env.robot.data.joint_pos_limits.clone()
        env.robot.root_physx_view.get_dof_limits = lambda: original
        with self.assertRaisesRegex(RuntimeError, "readback"):
            apply_held_joint_mode(env)

    def test_reports_actual_coordinate_drift_and_nonfinite_states(self):
        env = self.fixture()
        report = held_joint_deviation(env)
        self.assertAlmostEqual(report["maximum_position_error_rad"], .01, places=6)
        self.assertAlmostEqual(report["velocity_max_rad_s"][0], .2, places=6)
        env.robot.data.joint_pos[0, 2] = torch.nan
        with self.assertRaises(FloatingPointError):
            held_joint_deviation(env)


# Local robot config.

class Config(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def copy(self):
        return copy.deepcopy(self)


class ArticulationConfig(Config):
    InitialStateCfg = Config


def config_factory():
    """Execute the real builder with inert config objects, no simulator imports."""
    source = ROOT / "morph_mimic/utils/robot_cfg.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "make_robot_cfg")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        function,
    ], type_ignores=[])
    scope = dict(PROJECT_ROOT=ROOT, ROBOT_PRIM_PATH="/World/envs/env_.*/Robot",
                 ArticulationCfg=ArticulationConfig, ImplicitActuatorCfg=Config,
                 sim_utils=SimpleNamespace(MjcfFileCfg=type("MjcfConfig", (Config,), {}),
                                           UrdfFileCfg=type("UrdfConfig", (Config,), {}),
                                           RigidBodyPropertiesCfg=Config,
                                           ArticulationRootPropertiesCfg=Config))
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope["make_robot_cfg"]


class TestLocalRobotConfig(unittest.TestCase):
    def spec(self, name="g1"):
        return SimpleNamespace(**json.loads((ROOT / "assets" / name / "robot_spec.json").read_text()))

    def test_g1_local_reference_model_has_complete_named_control_and_free_root(self):
        spec = self.spec()
        enable = Mock(return_value=True)
        derived = ROOT / ".cache/mjcf/g1/g1_explicit.xml"
        with patch.object(explicit_mjcf, "checked_explicit_mjcf", return_value=derived):
            with patch.dict(sys.modules, {"isaacsim.core.utils.extensions": SimpleNamespace(enable_extension=enable)}):
                cfg = config_factory()(spec, use_usd=False)
        enable.assert_called_once_with("isaacsim.asset.importer.mjcf")
        self.assertEqual(type(cfg.spawn).__name__, "MjcfConfig")
        self.assertEqual(cfg.articulation_root_prim_path, "/pelvis/pelvis")
        self.assertEqual(Path(cfg.spawn.asset_path), ROOT / ".cache/mjcf/g1/g1_explicit.xml")
        self.assertEqual(Path(cfg.spawn.usd_dir), ROOT / ".cache/usd/g1_mjcf")
        self.assertFalse(cfg.spawn.fix_base)
        self.assertTrue(cfg.spawn.activate_contact_sensors)
        self.assertTrue(cfg.spawn.import_inertia_tensor)
        self.assertTrue(cfg.spawn.self_collision)
        self.assertTrue(cfg.spawn.articulation_props.enabled_self_collisions)
        self.assertFalse(cfg.spawn.rigid_props.disable_gravity)
        self.assertEqual(set(cfg.init_state.joint_pos), set(spec.joint_names))
        self.assertEqual(len(cfg.init_state.joint_pos), 29)
        self.assertEqual(cfg.actuators["all"].effort_limit_sim,
                         dict(zip(spec.joint_names, spec.effort_limits)))

    def test_default_usd_selection_remains_independent_of_local_fallback(self):
        configured_usd = Config(prim_path="old", spawn=Config(activate_contact_sensors=False),
                                init_state=Config(rot=None, joint_pos={}))
        module = SimpleNamespace(G1_29DOF_CFG=configured_usd)
        with patch.dict(sys.modules, {"isaaclab_assets.robots.unitree": module}):
            cfg = config_factory()(self.spec(), use_usd=True)
        self.assertTrue(cfg.spawn.activate_contact_sensors)
        self.assertNotEqual(cfg.prim_path, "old")
        self.assertFalse(configured_usd.spawn.activate_contact_sensors)
        self.assertEqual(configured_usd.init_state.joint_pos, {})

    def test_existing_urdf_robot_still_uses_urdf_importer(self):
        spec = self.spec("mojia")
        cfg = config_factory()(spec, use_usd=False)
        self.assertEqual(type(cfg.spawn).__name__, "UrdfConfig")
        self.assertEqual(Path(cfg.spawn.asset_path), ROOT / spec.runtime_assets["urdf"])
        self.assertFalse(cfg.spawn.articulation_props.enabled_self_collisions)

    def test_missing_reference_file_fails_before_launching_import(self):
        spec = self.spec()
        spec.runtime_assets = dict(urdf=None, mjcf="assets/g1/missing_reference_model.xml")
        with self.assertRaisesRegex(ValueError, "MJCF not found"):
            config_factory()(spec, use_usd=False)

    def test_missing_mjcf_extension_fails_before_converter_is_constructed(self):
        enable = Mock(return_value=False)
        with patch.object(explicit_mjcf, "checked_explicit_mjcf", return_value=ROOT / "explicit.xml"):
            with patch.dict(sys.modules, {"isaacsim.core.utils.extensions": SimpleNamespace(enable_extension=enable)}):
                with self.assertRaisesRegex(RuntimeError, "Could not enable.*MJCF"):
                    config_factory()(self.spec(), use_usd=False)
        enable.assert_called_once_with("isaacsim.asset.importer.mjcf")


# Explicit mjcf.

@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is unavailable")
class TestExplicitMjcf(unittest.TestCase):
    def test_g1_materializes_capsules_without_moving_meshes_or_changing_inertia(self):
        import mujoco

        source = Path(__file__).resolve().parents[1] / "assets/g1/robot/g1.xml"
        with tempfile.TemporaryDirectory() as directory:
            output, provenance = export_explicit_mjcf(source, directory)
            tree = ET.parse(output)
            self.assertFalse(tree.findall("default"))
            self.assertTrue(all("class" not in node.attrib for node in tree.iter()))
            self.assertEqual(tree.find(".//geom[@name='left_hip_collision']").get("type"), "capsule")
            self.assertEqual(tree.find(".//geom[@name='left_foot1_collision']").get("type"), "capsule")
            for asset in tree.findall("asset/mesh"):
                path = Path(asset.get("file"))
                self.assertFalse(path.is_absolute())
                self.assertEqual(path.parts[0], "assets")
                self.assertTrue((output.parent / path).is_file())
                self.assertIn(str(output.parent / path), provenance["dependency_sha256"])
            original, derivative = (mujoco.MjModel.from_xml_path(str(path)) for path in (source, output))
            np.testing.assert_allclose(original.geom_pos, derivative.geom_pos, atol=1e-12)
            np.testing.assert_allclose(original.geom_quat, derivative.geom_quat, atol=1e-12)
            np.testing.assert_allclose(original.body_inertia, derivative.body_inertia, atol=1e-12)
            np.testing.assert_array_equal(original.geom_type, derivative.geom_type)
            np.testing.assert_array_equal(original.geom_contype, derivative.geom_contype)
            self.assertTrue(provenance["mujoco_equivalence_verified"])
            self.assertFalse(provenance["isaac_dynamics_equivalence_validated"])
            self.assertEqual(checked_explicit_mjcf(source, directory), output)

    def test_hash_checks_reject_source_or_derived_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "robot.xml"
            original = ('<mujoco><default><default class="contact"><geom type="capsule" size=".05"/>'
                        '</default></default><worldbody><body name="base"><freejoint/>'
                        '<geom class="contact" name="contact" fromto="0 0 0 0 0 .2"/>'
                        '</body></worldbody></mujoco>')
            source.write_text(original)
            output, _ = export_explicit_mjcf(source, Path(directory) / "cache")
            source.write_text(original + "<!-- source changed -->")
            with self.assertRaisesRegex(ValueError, "dependency changed"):
                checked_explicit_mjcf(source, output.parent)
            source.write_text(original)
            output.write_text(output.read_text() + "<!-- derivative changed -->")
            with self.assertRaisesRegex(ValueError, "cache hash mismatch"):
                checked_explicit_mjcf(source, output.parent)


# Runtime audit.

class TestRuntimeAudit(unittest.TestCase):
    def test_principal_tensor_rotates_into_link_axes_with_quaternion_sign_invariance(self):
        # A 45-degree rotation of principal moments 1,3 gives nonzero Ixy.
        quaternion = np.array([np.cos(np.pi/8), 0., 0., np.sin(np.pi/8)])
        expected = np.array([[2., -1., 0.], [-1., 2., 0.], [0., 0., 5.]])
        np.testing.assert_allclose(inertia_in_link_frame([1., 3., 5.], quaternion), expected, atol=1e-12)
        np.testing.assert_allclose(inertia_in_link_frame([1., 3., 5.], -quaternion), expected, atol=1e-12)


# Planner attention.

class _FixedQueries(nn.Module):
    def forward(self, context):
        return context.new_full((context.shape[0], 9, 96), 100.0)


class TestPlannerAttention(unittest.TestCase):
    def test_sharp_attention_preserves_contact_selection_confidence(self):
        planner = ACPPlanner()
        target_index = 42
        points = torch.randn(1, 187, 3)
        kv = torch.zeros(1, 187, 96)
        kv[:, target_index] = 1.0
        pooled = torch.zeros(1, 64)
        planner.film = _FixedQueries()

        # Make all 32 attention heads select the same terrain point. This uses
        # the real MultiheadAttention softmax, including its head averaging.
        with torch.no_grad():
            identity = torch.eye(96)
            planner.attn.in_proj_weight.copy_(identity.repeat(3, 1))
            planner.attn.in_proj_bias.zero_()
            planner.occupancy_head[-1].weight.zero_()
            planner.occupancy_head[-1].bias.fill_(10.0)
        with patch.object(planner, "_terrain_features", return_value=(kv, pooled)):
            affordance, occupancy, contacts = planner(
                points, torch.zeros(1, 17, 11), torch.zeros(1, 378)
            )

        self.assertTrue((affordance[..., target_index] > 0.99).all())
        self.assertTrue(torch.allclose(affordance.sum(-1), torch.ones(1, 9)))
        self.assertTrue((occupancy > 0.5).all())
        expected = points[:, target_index].unsqueeze(1).expand(-1, 9, -1)
        self.assertTrue(torch.equal(contacts, expected))

    def test_temperature_scales_log_probabilities(self):
        planner = ACPPlanner(temperature=0.5)
        weights = torch.full((1, 9, 187), 0.2 / 186)
        weights[..., 17] = 0.8
        expected = weights.square() / weights.square().sum(-1, keepdim=True)
        with patch.object(
            planner.attn, "forward", return_value=(torch.zeros(1, 9, 96), weights)
        ):
            affordance, _, _ = planner(
                torch.zeros(1, 187, 3),
                torch.zeros(1, 17, 11),
                torch.zeros(1, 378),
            )
        self.assertTrue(torch.allclose(affordance, expected, atol=1e-6))

    def test_mlp_planner_decodes_contacts_with_shared_terrain_decoder(self):
        planner = MLPPlanner()
        points = torch.randn(2, 187, 3)
        with torch.no_grad():
            planner.terrain_head.weight.zero_()
            planner.terrain_head.bias.zero_()
            planner.terrain_head.bias.view(9, 187)[:, 27] = 10.0
            planner.occ_head.weight.zero_()
            planner.occ_head.bias.fill_(10.0)
            affordance, occupancy, contacts = planner(
                points, torch.zeros(2, 17, 11), torch.zeros(2, 378)
            )
        self.assertEqual(affordance.shape, (2, 9, 187))
        self.assertEqual(occupancy.shape, (2, 9))
        expected = points[:, 27].unsqueeze(1).expand(-1, 9, -1)
        self.assertEqual(planner.sigma_a, 0.1)
        self.assertTrue(torch.equal(contacts, expected))
        self.assertTrue(torch.equal(
            contacts, ACPPlanner().decode(points, affordance, occupancy)
        ))
        self.assertTrue(torch.equal(
            planner.decode(points, affordance, torch.zeros_like(occupancy)),
            torch.zeros_like(contacts),
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
