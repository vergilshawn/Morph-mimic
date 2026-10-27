"""Verify scene/world alignment and reject stale or ambiguous bindings on CPU."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from morph_mimic.utils.reference_scene import (
    load_reference_scene,
    reference_scene_grid_shape,
    reference_scene_mesh_copy,
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestReferenceScene(unittest.TestCase):
    def test_replica_grid_has_one_distinct_cell_per_environment(self):
        for count in (1, 2, 3, 5, 16, 32, 128, 256, 512):
            with self.subTest(count=count):
                rows, cols = reference_scene_grid_shape(count)
                self.assertGreaterEqual(rows * cols, count)
                self.assertLess(rows * cols - count, rows)
                assigned = [(i // cols, i % cols) for i in range(count)]
                self.assertEqual(len(set(assigned)), count)
                self.assertLess(assigned[-1][0], rows)
        self.assertEqual(reference_scene_grid_shape(1), (1, 1))
        for count in (0, -1, 1.5, True, "2", None):
            with self.subTest(invalid=count), self.assertRaisesRegex(ValueError, "positive integer"):
                reference_scene_grid_shape(count)

    def test_replica_geometry_is_verified_once_and_each_tile_gets_an_independent_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar, _ = self.fixture(directory)
            scene = load_reference_scene(sidecar)
            expected = scene.rebased_mesh().vertices.copy()
            with patch("morph_mimic.utils.reference_scene.load_reference_scene", wraps=load_reference_scene) as loader:
                first = reference_scene_mesh_copy(sidecar, scene.fingerprint)
                first.apply_translation((100., 200., 0.))
                second = reference_scene_mesh_copy(sidecar, scene.fingerprint)
                self.assertEqual(loader.call_count, 1)
                np.testing.assert_allclose(second.vertices, expected)
                with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                    reference_scene_mesh_copy(sidecar, "0" * 64)
            # Unbound/direct callers cannot bypass source hashing through cache.
            with (Path(directory) / "obstacle.obj").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                reference_scene_mesh_copy(sidecar)

    def fixture(self, directory, urdf=None):
        directory = Path(directory)
        mesh = directory / "obstacle.obj"
        trimesh.creation.box(extents=(2, 4, 6)).export(mesh)
        motion = directory / "motion.npz"
        np.savez(motion, root_pos=np.array([[2., -3., 1.], [2.1, -3., 1.]]))
        source = directory / "source.npz"
        np.savez(source, qpos=np.zeros((2, 36)))
        terrain = directory / "terrain.urdf"
        terrain.write_text(urdf or '''<robot name="terrain">
          <link name="root"><collision name="scaled_mesh"><origin xyz="1 0 0"/>
            <geometry><mesh filename="obstacle.obj" scale="2 1 0.5"/></geometry></collision></link>
          <link name="child"><collision name="primitive"><geometry><box size="1 2 3"/></geometry></collision></link>
          <joint name="world_root" type="fixed"><parent link="world"/><child link="root"/>
            <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/></joint>
          <joint name="root_child" type="fixed"><parent link="root"/><child link="child"/>
            <origin xyz="0 1 0"/></joint></robot>''')
        metadata = dict(schema=1, robot="g1", motion_file="motion.npz", motion_sha256=_hash(motion),
                        source_motion={"path": "source.npz", "sha256": _hash(source)},
                        scene_urdf={"path": "terrain.urdf", "sha256": _hash(terrain)},
                        source_first_root_xy=[2, -3], world_position=[10, 0, 0],
                        world_quat_wxyz=[1, 0, 0, 0],
                        ground=dict(top_z_m=0., size_m=[10., 10.], thickness_m=1., center_xy_m=[0., 0.]),
                        collision_asset_sha256={"obstacle.obj": _hash(mesh)})
        sidecar = directory / "motion.scene.json"
        sidecar.write_text(json.dumps(metadata))
        return sidecar, metadata

    def rewrite(self, sidecar, metadata):
        sidecar.write_text(json.dumps(metadata))

    def test_fixed_tree_collision_origin_mesh_scale_and_rebase_share_one_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar, _ = self.fixture(directory)
            scene = load_reference_scene(sidecar)
            self.assertEqual(len(scene.components), 2)
            mesh, box = scene.components
            np.testing.assert_allclose(mesh.mesh.bounds, [[9, 1, 1.5], [13, 5, 4.5]], atol=1e-8)
            np.testing.assert_allclose(box.mesh.bounds, [[9, 1.5, 1.5], [11, 2.5, 4.5]], atol=1e-8)
            self.assertEqual(box.size_m, (1, 2, 3))
            np.testing.assert_allclose(mesh.transform[:3, 3], [11, 3, 3], atol=1e-8)
            np.testing.assert_allclose(scene.ground.mesh().bounds, [[-5, -5, -1], [5, 5, 0]])
            source = scene.combined_mesh()
            rebased = scene.rebased_mesh()
            np.testing.assert_allclose(rebased.vertices, source.vertices + [-2, 3, 0])
            np.testing.assert_array_equal(source.faces, rebased.faces)
            # Combining/rebasing must never mutate individual source components.
            np.testing.assert_allclose(mesh.mesh.bounds, [[9, 1, 1.5], [13, 5, 4.5]], atol=1e-8)

    def test_fingerprint_is_portable_but_binds_ground_and_motion_rebase(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            original.mkdir()
            sidecar, metadata = self.fixture(original)
            first = load_reference_scene(sidecar).fingerprint
            copied = Path(directory) / "copied"
            shutil.copytree(original, copied)
            self.assertEqual(first, load_reference_scene(copied / sidecar.name).fingerprint)
            metadata["ground"]["top_z_m"] = 0.2
            self.rewrite(sidecar, metadata)
            self.assertNotEqual(first, load_reference_scene(sidecar).fingerprint)
            metadata["source_first_root_xy"] = [99, 0]
            self.rewrite(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "source_first_root_xy differs"):
                load_reference_scene(sidecar)

    def test_all_source_hashes_are_checked_and_dependency_set_is_exact(self):
        for field in ("motion", "source_motion", "scene_urdf", "collision_asset"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                sidecar, _ = self.fixture(directory)
                filenames = {"motion": "motion.npz", "source_motion": "source.npz",
                             "scene_urdf": "terrain.urdf", "collision_asset": "obstacle.obj"}
                path = Path(directory) / filenames[field]
                with path.open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, field + ": SHA256 mismatch"):
                    load_reference_scene(sidecar)
        with tempfile.TemporaryDirectory() as directory:
            sidecar, metadata = self.fixture(directory)
            metadata["collision_asset_sha256"] = {}
            self.rewrite(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "missing from collision_asset_sha256"):
                load_reference_scene(sidecar)
            metadata["collision_asset_sha256"] = {"source.npz": _hash(Path(directory) / "source.npz"),
                                                   "obstacle.obj": _hash(Path(directory) / "obstacle.obj")}
            self.rewrite(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "files not used"):
                load_reference_scene(sidecar)

    def test_rejects_unsupported_dynamic_or_ambiguous_geometry(self):
        cases = [
            ("<sphere radius='1'/>", "unsupported geometry sphere"),
            ("<box size='1 0 2'/>", "positive box size"),
            ("<mesh filename='obstacle.obj' scale='-1 1 1'/>", "positive mesh scale"),
            ("<mesh filename='https://example.com/obstacle.obj'/>", "local file path"),
        ]
        for geometry, error in cases:
            with self.subTest(geometry=geometry), tempfile.TemporaryDirectory() as directory:
                sidecar, _ = self.fixture(directory, f"<robot><link name='root'><collision><geometry>{geometry}</geometry></collision></link></robot>")
                with self.assertRaisesRegex(ValueError, error):
                    load_reference_scene(sidecar)
        for joint, error in [
            ("<joint name='j' type='revolute'><parent link='world'/><child link='root'/></joint>", "fixed joints only"),
            ("<joint name='j' type='fixed'><parent link='missing'/><child link='root'/></joint>", "missing parent link"),
            ("<joint name='j' type='fixed'><parent link='root'/><child link='root'/></joint>", "cycle"),
        ]:
            with self.subTest(joint=joint), tempfile.TemporaryDirectory() as directory:
                sidecar, _ = self.fixture(directory, "<robot><link name='root'><collision><geometry><mesh filename='obstacle.obj'/></geometry></collision></link>" + joint + "</robot>")
                with self.assertRaisesRegex(ValueError, error):
                    load_reference_scene(sidecar)

    def test_implicit_world_accepts_multiple_fixed_children_and_world_quaternion(self):
        urdf = """<robot><link name='a'><collision><geometry><box size='1 1 1'/></geometry></collision></link>
        <link name='b'><collision><geometry><box size='1 1 1'/></geometry></collision></link>
        <joint name='a_j' type='fixed'><parent link='world'/><child link='a'/><origin xyz='1 0 0'/></joint>
        <joint name='b_j' type='fixed'><parent link='world'/><child link='b'/><origin xyz='0 2 0'/></joint></robot>"""
        with tempfile.TemporaryDirectory() as directory:
            sidecar, metadata = self.fixture(directory, urdf)
            metadata["collision_asset_sha256"] = {}
            metadata["world_quat_wxyz"] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
            self.rewrite(sidecar, metadata)
            scene = load_reference_scene(sidecar)
            np.testing.assert_allclose(scene.components[0].mesh.bounds.mean(axis=0), [10, 1, 0], atol=1e-8)
            np.testing.assert_allclose(scene.components[1].mesh.bounds.mean(axis=0), [8, 0, 0], atol=1e-8)
            metadata["world_quat_wxyz"] = [2, 0, 0, 0]
            self.rewrite(sidecar, metadata)
            with self.assertRaisesRegex(ValueError, "unit WXYZ"):
                load_reference_scene(sidecar)

    def test_json_duplicate_keys_and_motion_path_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sidecar, _ = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "requested motion path differs"):
                load_reference_scene(sidecar, Path(directory) / "source.npz")
            sidecar.write_text('{"schema":1,"schema":1}')
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_reference_scene(sidecar)


if __name__ == "__main__":
    unittest.main()
