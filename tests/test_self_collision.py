"""Actual MuJoCo collision filters must govern self-collision supervision."""

import importlib.util
import unittest

import numpy as np

from morph_mimic.contacts.self_collision import effective_robot_geom_pairs, sample_self_contacts


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo required for collision filtering")
class TestEffectiveSelfCollision(unittest.TestCase):
    def model(self, exclude=False, masked=False):
        import mujoco

        contact = '<contact><exclude body1="left" body2="right"/></contact>' if exclude else ''
        mask = 'contype="0" conaffinity="0"' if masked else ''
        return mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <geom name="scene" type="sphere" size="2"/>
          <body name="root"><freejoint/>
            <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
            <body name="left" pos="-.04 0 0"><joint axis="0 0 1"/><geom name="left_geom" type="sphere" size=".1" mass="1"/></body>
            <body name="right" pos=".04 0 0"><joint axis="0 0 1"/><geom name="right_geom" type="sphere" size=".1" mass="1" {mask}/></body>
          </body></worldbody>{contact}</mujoco>''')

    def test_actual_collision_gate_and_pair_enumeration_agree(self):
        model = self.model()
        scene = {model.geom("scene").id}
        report = sample_self_contacts(model, model.qpos0[None, :], scene)
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(report["maximum_penetration_m"], .12)
        self.assertEqual(set(report["worst_contact"]["body_names"]), {"left", "right"})
        pairs = {tuple(pair) for pair in effective_robot_geom_pairs(model, scene)}
        expected = tuple(sorted((model.geom("left_geom").id, model.geom("right_geom").id)))
        self.assertIn(expected, pairs)

    def test_explicit_excludes_and_masks_are_preserved(self):
        for exclude, masked in ((True, False), (False, True)):
            with self.subTest(exclude=exclude, masked=masked):
                model = self.model(exclude, masked)
                scene = {model.geom("scene").id}
                before = model.geom_contype.copy(), model.geom_conaffinity.copy(), model.exclude_signature.copy()
                report = sample_self_contacts(model, model.qpos0[None, :], scene)
                self.assertTrue(report["accepted"])
                self.assertEqual(report["maximum_penetration_m"], 0)
                self.assertEqual(len(effective_robot_geom_pairs(model, scene)), 0)
                np.testing.assert_array_equal(model.geom_contype, before[0])
                np.testing.assert_array_equal(model.geom_conaffinity, before[1])
                np.testing.assert_array_equal(model.exclude_signature, before[2])

    def test_disabling_collision_cannot_create_a_passing_audit(self):
        import mujoco

        model = self.model()
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        with self.assertRaisesRegex(ValueError, "collision detection disabled"):
            sample_self_contacts(model, model.qpos0[None, :], {model.geom("scene").id})
        with self.assertRaisesRegex(ValueError, "collision detection to remain enabled"):
            effective_robot_geom_pairs(model, {model.geom("scene").id})


if __name__ == "__main__":
    unittest.main()
