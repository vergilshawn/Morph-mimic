"""Preserve compiled robot geometry/inertia while removing MJCF defaults."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from morph_mimic.utils.explicit_mjcf import checked_explicit_mjcf, export_explicit_mjcf


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


if __name__ == "__main__":
    unittest.main()
