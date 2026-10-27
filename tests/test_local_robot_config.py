"""Select the reference MJCF without requiring a running SimulationApp."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from morph_mimic.utils import explicit_mjcf


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
