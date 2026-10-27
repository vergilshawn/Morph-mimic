"""Check actual mesh generators without starting Isaac Sim.

The terrain functions depend only on NumPy, trimesh and configuration fields.
Load their source while omitting Isaac Lab imports/configclass definitions, and
read literal configuration defaults from those definitions. These checks cover
geometry construction only; they do not validate simulator physics or alignment
between a motion clip and its terrain.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import trimesh


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


if __name__ == "__main__":
    unittest.main()
