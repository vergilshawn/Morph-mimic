"""Exercise real scene integration methods on CPU, without starting Isaac.

The installed TerrainGenerator's actual centering operations are executed via
AST extraction. These tests verify geometry/reference coordinates and refusal
gates; simulator import and physical tracking need separate validation.
"""

import ast
from copy import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from morph_mimic.scripts.common import configure_reference_scene
from morph_mimic.scripts import common
from morph_mimic.utils import checkpoint as cp
from morph_mimic.utils.motion import MotionDataset
from morph_mimic.utils.reference_scene import load_reference_scene, reference_scene_grid_shape
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec
from morph_mimic.utils.transforms import to_body_yaw_frame, yaw_from_quat


CLIP = "climb_18_z_scale_1.0"
DIRECTORY = PROJECT_ROOT / "datasets/g1_box_climb_smoke"
ISAAC_GENERATOR = Path("/home/user/IsaacLab/source/isaaclab/isaaclab/terrains/terrain_generator.py")


def _compile(nodes, path, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope


def _environment_class(base=object, terrain_builder=None):
    path = PROJECT_ROOT / "morph_mimic/envs/acp_env.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "AcpEnv")
    return _compile([cls], path, dict(np=np, torch=torch, DirectRLEnv=base,
                                     TERRAIN_GROUP={"reference_scene": "dense_whole_body"},
                                     build_terrain_cfg=terrain_builder,
                                     to_body_yaw_frame=to_body_yaw_frame, yaw_from_quat=yaw_from_quat))["AcpEnv"]


def _terrain_function():
    path = PROJECT_ROOT / "morph_mimic/envs/terrains.py"
    function = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "reference_scene_terrain")
    return _compile([function], path, dict(np=np))["reference_scene_terrain"]


def _terrain_builder():
    path = PROJECT_ROOT / "morph_mimic/envs/acp_env.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_terrain_cfg")
    factories = {node.id: NS for node in ast.walk(function)
                 if isinstance(node, ast.Name) and node.id.endswith("Cfg")}
    factories["sim_utils"] = NS(RigidBodyMaterialCfg=NS)
    return _compile([function], path, factories)["build_terrain_cfg"]


class SubCfg(NS):
    def copy(self):
        return copy(self)

    def to_dict(self):
        return {key: str(value) for key, value in vars(self).items()}


def _run_installed_generator_placement(terrain_function, sidecar, size, num_envs=1):
    """Run actual _get/_add methods and final __init__ recenter statements."""
    tree = ast.parse(ISAAC_GENERATOR.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TerrainGenerator")
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_get_terrain_mesh", "_add_sub_terrain"}]
    extracted = ast.ClassDef(name="GeneratorPlacement", bases=[], keywords=[], body=methods, decorator_list=[])
    scope = _compile([extracted], ISAAC_GENERATOR,
                     dict(np=np, torch=torch, trimesh=trimesh, os=os, dict_to_md5_hash=lambda value: "cache-disabled"))
    generator = scope["GeneratorPlacement"]()
    rows, cols = reference_scene_grid_shape(num_envs)
    generator.cfg = NS(seed=0, use_cache=False, cache_dir="/tmp/unused-reference-scene-cache",
                       size=size, num_rows=rows, num_cols=cols)
    generator.device = "cpu"
    generator.terrain_meshes = []
    generator.terrain_origins = np.zeros((rows, cols, 3))
    sub = SubCfg(function=terrain_function, sidecar_path=str(sidecar), size=size, flat_patch_sampling=None,
                 fingerprint=load_reference_scene(sidecar).fingerprint)
    for row in range(rows):
        for col in range(cols):
            mesh, origin = generator._get_terrain_mesh(0.0, sub)
            generator._add_sub_terrain(mesh, origin, row, col, sub)
    generator.terrain_mesh = trimesh.util.concatenate(generator.terrain_meshes)
    init = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    # The last assignment to local ``transform`` starts the generator-wide
    # recentering block. Execute those real operations, rather than duplicating
    # their arithmetic in a test implementation.
    starts = [index for index, node in enumerate(init.body) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "transform" for target in node.targets)]
    final_nodes = init.body[starts[-1]:starts[-1] + 4]
    _compile(final_nodes, ISAAC_GENERATOR, dict(self=generator, np=np))
    return generator


@unittest.skipUnless((DIRECTORY / f"{CLIP}.scene.json").is_file(), "converted box-climbing fixture is absent")
class TestSceneIntegration(unittest.TestCase):
    def dataset(self, directory=DIRECTORY):
        return MotionDataset(RobotSpec.load("g1"), directory, clip_names=[CLIP])

    def cfg(self, **overrides):
        fields = dict(stage=1, ref_env_ratio=1., reference_scene_path=None, flat_reference=False,
                      terrain_size=(9., 6.), terrain_types=("flat",), ref_terrain_types=("flat",), num_terrain_levels=10)
        fields.update(overrides)
        return NS(**fields)

    @unittest.skipUnless(ISAAC_GENERATOR.is_file(), "installed IsaacLab terrain source is unavailable")
    def test_real_generator_recenter_keeps_motion_obstacle_relative_positions(self):
        motion = self.dataset()
        scene = motion.scene_bindings[CLIP]
        cfg = self.cfg()
        configure_reference_scene(cfg, motion, 1)
        placed = _run_installed_generator_placement(_terrain_function(), scene.sidecar_path, cfg.terrain_size)
        source = scene.combined_mesh()
        expected_vertices = source.vertices - [*scene.source_first_root_xy, 0.]
        np.testing.assert_allclose(placed.terrain_mesh.vertices, expected_vertices, atol=1e-8)
        np.testing.assert_array_equal(placed.terrain_mesh.faces, source.faces)
        np.testing.assert_allclose(placed.terrain_origins, 0., atol=1e-8)

        Env = _environment_class()
        env = Env.__new__(Env)
        env.cfg, env.motion_data, env.device, env.num_envs = cfg, motion, "cpu", 1
        env.terrain_origins = torch.tensor(placed.terrain_origins, dtype=torch.float32)
        env._init_terrain_mapping()
        np.testing.assert_array_equal(env.cell_offset.numpy(), 0.)
        env.is_ref_env = torch.ones(1, dtype=torch.bool)
        env.clip_id = torch.zeros(1, dtype=torch.long)
        for phase in (0., .3, .8):
            env.phase = torch.tensor([phase])
            reference = env._ref_sample()["root_pos"][0].numpy()
            raw_reference = motion.clips[CLIP].sample(env.phase)["root_pos"][0].numpy()
            # Compare every original obstacle vertex relative to the reference;
            # preserving this vector is the actual alignment invariant.
            np.testing.assert_allclose(placed.terrain_mesh.vertices - reference,
                                       source.vertices - raw_reference, atol=2e-7)

    def test_single_scene_stage_and_flat_replacement_gates(self):
        motion = self.dataset()
        for cfg, count, error in [
            (self.cfg(), 0, "positive integer"),
            (self.cfg(stage=3), 1, "Stage 1/2"),
            (self.cfg(ref_env_ratio=.5), 1, "reference environments"),
            (self.cfg(flat_reference=True), 1, "cannot replace"),
        ]:
            with self.subTest(config=vars(cfg), count=count), self.assertRaisesRegex(ValueError, error):
                configure_reference_scene(cfg, motion, count)
        unbound = NS(scene_bindings={}, clip_names=["flat"])
        with self.assertRaisesRegex(ValueError, "hashed motion scene"):
            configure_reference_scene(self.cfg(reference_scene_path="arbitrary.scene.json"), unbound, 1)
        with self.assertRaisesRegex(ValueError, "exactly one bound clip"):
            configure_reference_scene(self.cfg(), NS(scene_bindings=motion.scene_bindings, clip_names=[CLIP, "other"]), 1)

    def test_required_scene_is_not_silently_replaced_when_sidecar_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy2(DIRECTORY / f"{CLIP}.npz", Path(directory) / f"{CLIP}.npz")
            with self.assertRaisesRegex(ValueError, "required scene sidecar is missing"):
                self.dataset(directory)

    def test_direct_environment_constructor_enforces_binding_before_simulation(self):
        motion = self.dataset()

        class BeforePhysics(Exception):
            pass

        class SimulatorBoundary:
            def __init__(self, *args, **kwargs):
                raise BeforePhysics()

        generated = []

        def terrain_builder(cfg):
            generated.append(cfg.reference_scene_path)
            return NS()

        Env = _environment_class(base=SimulatorBoundary, terrain_builder=terrain_builder)
        cfg = self.cfg(evaluation=False, scene=NS(num_envs=1), planner_scan_shape=(17, 11), tracker_scan_shape=(9, 7))
        with self.assertRaises(BeforePhysics):
            Env(cfg, motion.spec, motion)
        self.assertEqual(generated, [str(motion.scene_bindings[CLIP].sidecar_path)])
        self.assertEqual(cfg.terrain_types, ("reference_scene",))
        for overrides, error in [
            ({"terrain": NS()}, "preconfigured terrain importer"),
            ({"scene": NS(num_envs=0)}, "positive integer"),
            ({"stage": 3}, "Stage 1/2"),
        ]:
            cfg = self.cfg(evaluation=False, scene=NS(num_envs=1))
            for key, value in overrides.items():
                setattr(cfg, key, value)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, error):
                Env(cfg, motion.spec, motion)

    def test_terrain_builder_creates_replica_grid_without_changing_difficulty_levels(self):
        motion = self.dataset()
        builder = _terrain_builder()
        for count in (1, 5, 32, 512):
            with self.subTest(num_envs=count):
                cfg = self.cfg(scene=NS(num_envs=count))
                configure_reference_scene(cfg, motion, count)
                terrain = builder(cfg)
                generator = terrain.terrain_generator
                self.assertEqual((generator.num_rows, generator.num_cols), reference_scene_grid_shape(count))
                self.assertEqual(cfg.num_terrain_levels, 1)
                self.assertEqual(list(generator.sub_terrains), ["reference_scene"])
                self.assertEqual(generator.sub_terrains["reference_scene"].fingerprint,
                                 motion.scene_bindings[CLIP].fingerprint)
                for key, value in (("stage", 3), ("ref_env_ratio", .5), ("num_terrain_levels", 2),
                                   ("terrain_types", ("flat",)), ("ref_terrain_types", ("flat",))):
                    invalid = copy(cfg)
                    setattr(invalid, key, value)
                    with self.subTest(invalid=key), self.assertRaisesRegex(ValueError, "Stage 1/2"):
                        builder(invalid)

    @unittest.skipUnless(ISAAC_GENERATOR.is_file(), "installed IsaacLab terrain source is unavailable")
    def test_replicated_scene_tiles_keep_geometry_reference_and_contact_frames_aligned(self):
        motion = self.dataset()
        scene = motion.scene_bindings[CLIP]
        # The older geometry fixture predates contact labeling. Give its oracle
        # nonzero per-anatomy/per-frame targets so this check cannot pass merely
        # because all missing-label outputs happen to be zero.
        frames = motion.clips[CLIP].num_frames
        targets = torch.arange(27, dtype=torch.float32).reshape(1, 9, 3) / 30.
        targets = targets + torch.linspace(0., .1, frames)[:, None, None]
        flags = torch.ones(frames, 9)
        flags[:, ::2] = 0.
        motion.contact_targets[CLIP] = targets, flags
        motion.contact_target_frames[CLIP] = "reference_query_body_yaw_absolute_z"
        for count in (2, 5, 32):
            with self.subTest(num_envs=count):
                cfg = self.cfg()
                configure_reference_scene(cfg, motion, count)
                self.assertEqual(cfg.num_terrain_levels, 1)
                self.assertEqual(cfg.terrain_types, ("reference_scene",))
                self.assertEqual(cfg.reference_scene_fingerprint, scene.fingerprint)
                placed = _run_installed_generator_placement(
                    _terrain_function(), scene.sidecar_path, cfg.terrain_size, count)
                Env = _environment_class()
                env = Env.__new__(Env)
                env.cfg, env.motion_data, env.device, env.num_envs = cfg, motion, "cpu", count
                env.terrain_origins = torch.tensor(placed.terrain_origins, dtype=torch.float32)
                env._init_terrain_mapping()
                origins = env.env_origin()
                expected_origins = placed.terrain_origins.reshape(-1, 3)[:count]
                np.testing.assert_allclose(origins.numpy(), expected_origins, atol=2e-6)
                self.assertEqual(len(torch.unique(origins, dim=0)), count)
                np.testing.assert_array_equal(env.cell_offset.numpy(), 0.)
                env.is_ref_env = torch.ones(count, dtype=torch.bool)
                env.clip_id = torch.zeros(count, dtype=torch.long)
                env._contact_clip_body_idx = {}
                _, env.contact_motion_candidates = motion.spec.resolve_contact_bodies(motion.clips[CLIP].body_names)
                source = scene.combined_mesh()
                vertex_count = len(source.vertices)
                for phase in (0., .3, .8):
                    env.phase = torch.full((count,), phase)
                    sampled = env._ref_sample()
                    reference = sampled["root_pos"].numpy()
                    env.robot = NS(data=NS(root_pos_w=sampled["root_pos"], root_quat_w=sampled["root_quat"]))
                    raw_reference = motion.clips[CLIP].sample(env.phase)["root_pos"].numpy()
                    for cell in range(count):
                        vertices = placed.terrain_mesh.vertices[cell * vertex_count:(cell + 1) * vertex_count]
                        np.testing.assert_allclose(vertices - reference[cell], source.vertices - raw_reference[cell], atol=2e-5)
                    # Oracle targets are stored in the current reference yaw
                    # frame; horizontal tile translation must not alter them.
                    targets, flags = env._oracle_contacts(torch.arange(count))
                    torch.testing.assert_close(targets, targets[:1].expand_as(targets))
                    torch.testing.assert_close(flags, flags[:1].expand_as(flags))
                    contacts_world = env._ref_contact_pos(body_yaw=False) - origins[:, None, :]
                    contacts_body = env._ref_contact_pos(body_yaw=True)
                    torch.testing.assert_close(contacts_world, contacts_world[:1].expand_as(contacts_world), atol=2e-5, rtol=0.)
                    torch.testing.assert_close(contacts_body, contacts_body[:1].expand_as(contacts_body))
                # Every complete collision copy is separated by the configured
                # tile margin. Nearby robots cannot share the same obstacles.
                local_bounds = scene.rebased_mesh().bounds
                bounds = local_bounds[None, :, :2] + expected_origins[:, None, :2]
                for first in range(count):
                    for second in range(first + 1, count):
                        self.assertTrue(np.any(bounds[first, 1] < bounds[second, 0]) or
                                        np.any(bounds[second, 1] < bounds[first, 0]))

    def test_bound_scene_selects_local_robot_model_before_environment_creation(self):
        bound = self.dataset()
        model_requests = []

        def make_robot(spec, *, use_usd):
            model_requests.append(use_usd)
            return NS(local=not use_usd)

        def stage_cfg(count, *args):
            return self.cfg(sim=NS(device=None), episode_length_s=15., record_max_steps=500)

        fake_modules = {
            "morph_mimic.cfg.configs": NS(stage1_env_cfg=stage_cfg, stage2_env_cfg=stage_cfg, stage3_env_cfg=stage_cfg),
            "morph_mimic.envs.acp_env": NS(AcpEnv=lambda cfg, spec, motion: NS(cfg=cfg, motion=motion)),
            "morph_mimic.utils.robot_cfg": NS(make_robot_cfg=make_robot),
        }
        with patch.dict(sys.modules, fake_modules):
            with patch.object(common, "MotionDataset", return_value=bound):
                env, _ = common.build_env(1, "g1", 1, device="cpu", require_contacts=False, use_usd=True)
                self.assertTrue(env.cfg.robot.local)
                self.assertEqual(env.cfg.terrain_types, ("reference_scene",))
            unbound = NS(scene_bindings={}, clip_names=["walk"], contact_flags={})
            with patch.object(common, "MotionDataset", return_value=unbound):
                env, _ = common.build_env(1, "g1", 1, device="cpu", require_contacts=False, use_usd=True)
                self.assertFalse(env.cfg.robot.local)
        self.assertEqual(model_requests, [False, True])

    def test_contact_labels_without_scene_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            shutil.copy2(DIRECTORY / f"{CLIP}.npz", temporary / f"{CLIP}.npz")
            metadata = json.loads((DIRECTORY / f"{CLIP}.scene.json").read_text())
            (temporary / f"{CLIP}.scene.json").write_text(json.dumps(metadata))
            original = self.dataset()
            clip = original.clips[CLIP]
            spec = original.spec
            names = []
            for group in spec.nine_body_groups():
                candidates = spec.contact_bodies[group]
                names.extend(candidates if len(candidates) == 1 else
                             [next(name for name in candidates if name.startswith(side)) for side in ("left", "right")])
            np.savez(temporary / f"{CLIP}_contacts.npz", fps=np.array([clip.fps]),
                     body_names=np.array(names), contact_flags=np.zeros((clip.num_frames, 9)),
                     contact_targets=np.zeros((clip.num_frames, 9, 3)),
                     contact_target_flags=np.zeros((clip.num_frames, 9)))
            with self.assertRaisesRegex(ValueError, "do not bind the current collision scene"):
                self.dataset(temporary)

    def test_checkpoint_rejects_same_motion_with_changed_ground_scene(self):
        motion = self.dataset()
        cfg = self.cfg()
        cfg.sim, cfg.decimation = NS(dt=.005), 4
        cfg.robot = NS(spawn=NS(usd_path="test://runtime"))
        env = NS(cfg=cfg, robot_spec=motion.spec, motion_data=motion, num_envs=1)
        expected = {"contract": cp.build_checkpoint_contract(env)}
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            shutil.copy2(DIRECTORY / f"{CLIP}.npz", temporary / f"{CLIP}.npz")
            metadata = json.loads((DIRECTORY / f"{CLIP}.scene.json").read_text())
            metadata["ground"]["top_z_m"] = .05
            (temporary / f"{CLIP}.scene.json").write_text(json.dumps(metadata))
            changed = self.dataset(temporary)
            self.assertNotEqual(motion.scene_bindings[CLIP].fingerprint, changed.scene_bindings[CLIP].fingerprint)
            env.motion_data = changed
            for mode in ("resume", "transfer", "evaluate"):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "physics"):
                    cp.validate_checkpoint_contract(expected, env, mode=mode)


class TestFilteredSceneContacts(unittest.TestCase):
    def test_scene_forces_use_only_filtered_obstacles_and_keep_anatomical_order(self):
        Env = _environment_class()
        env = Env.__new__(Env)
        env.cfg = NS(contact_force_threshold=1.)
        # The broad robot sensor includes large self-contact forces. Every
        # filtered anatomy sensor has its own reporter and opposing-body axis.
        env.contact_sensor_idx = torch.arange(9)
        env.contact_sensor = NS(data=NS(net_forces_w=torch.full((2, 9, 3), 99999.),
                                       net_forces_w_history=torch.full((2, 4, 9, 3), 99999.)))
        env.scene_contact_sensors = []
        expected, histories = [], []
        for anatomy in range(9):
            force = torch.zeros(2, 1, 2, 3)
            force[:, 0, 0] = torch.tensor([float(anatomy), -float(anatomy), .2])
            force[:, 0, 1] = torch.tensor([.5, .25, .3])
            # Instantaneous force remains below the vertical threshold; body
            # three had a >1 N filtered vertical contact in an earlier sample.
            history = force.unsqueeze(1).repeat(1, 4, 1, 1, 1)
            if anatomy == 3:
                history[:, 2, 0, 1, 2] = 2.
            env.scene_contact_sensors.append(NS(data=NS(force_matrix_w=force, force_matrix_w_history=history,
                                                       net_forces_w=torch.full((2, 1, 3), 99999.))))
            expected.append(force.sum(dim=-2))
            histories.append(history.sum(dim=-2))
        filtered = env._contact_forces_w()
        filtered_history = env._contact_force_history()
        self.assertEqual(tuple(filtered.shape), (2, 9, 3))
        self.assertEqual(tuple(filtered_history.shape), (2, 4, 9, 3))
        torch.testing.assert_close(filtered, torch.cat(expected, dim=1))
        torch.testing.assert_close(filtered_history, torch.cat(histories, dim=2))
        measured = env._contact_measured()
        expected_flags = torch.zeros(2, 9)
        expected_flags[:, 3] = 1.
        torch.testing.assert_close(measured, expected_flags)
        # If unfiltered net force leaked into either path all nine columns
        # would be active; changing it must have no effect on scene results.
        env.contact_sensor.data.net_forces_w.fill_(-7e8)
        env.contact_sensor.data.net_forces_w_history.fill_(-7e8)
        torch.testing.assert_close(env._contact_forces_w(), filtered)
        torch.testing.assert_close(env._contact_force_history(), filtered_history)
        torch.testing.assert_close(env._contact_measured(), expected_flags)

    def test_unbound_legacy_sensor_path_keeps_explicit_anatomical_permutation(self):
        Env = _environment_class()
        env = Env.__new__(Env)
        env.scene_contact_sensors = []
        env.contact_sensor_idx = torch.tensor([8, 2, 7, 1, 6, 0, 5, 3, 4])
        net = torch.arange(2 * 12 * 3, dtype=torch.float32).reshape(2, 12, 3)
        history = torch.arange(2 * 4 * 12 * 3, dtype=torch.float32).reshape(2, 4, 12, 3)
        env.contact_sensor = NS(data=NS(net_forces_w=net, net_forces_w_history=history))
        torch.testing.assert_close(env._contact_forces_w(), net[:, env.contact_sensor_idx])
        torch.testing.assert_close(env._contact_force_history(), history[:, :, env.contact_sensor_idx])


if __name__ == "__main__":
    unittest.main()
