"""Physical checks for the CPU baseline, independent of Isaac/GPU availability."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from morph_mimic.scripts.validate_scene_reference import build_model, scene_contacts, simulate


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is unavailable")
class TestSceneReferenceDiagnostic(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("trimesh"), "trimesh is unavailable")
    def test_disconnected_collision_mesh_keeps_gap_and_rejects_concave_surface(self):
        """Mesh collision must not fill the space between separate boxes."""
        import mujoco
        import trimesh

        first = trimesh.creation.box(extents=[1., 1., 1.])
        second = first.copy().apply_translation([3., 0., 0.])
        component = SimpleNamespace(name="two_boxes", kind="mesh",
                                    mesh=trimesh.util.concatenate([first, second]))
        ground = SimpleNamespace(size_m=[10., 10.], thickness_m=1.,
                                 center_xy_m=[0., 0.], top_z_m=-2.)
        scene = SimpleNamespace(components=[component], ground=ground)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.xml"
            model_path.write_text('<mujoco><worldbody><body><freejoint/>'
                                  '<geom type="sphere" size=".1" mass="1"/>'
                                  '</body></worldbody></mujoco>')
            model, geom_ids = build_model(model_path, scene, .005)
            self.assertEqual(len(geom_ids), 3)  # two convex pieces and ground
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            ray_geom = np.zeros(1, dtype=np.int32)
            # A vertical ray through the 2 m gap reaches the ground, not a hull.
            distance = mujoco.mj_ray(model, data, np.array([1.5, 0., 3.]),
                                     np.array([0., 0., -1.]), None, 1, -1, ray_geom)
            self.assertAlmostEqual(distance, 5., places=6)
            component.mesh = trimesh.creation.torus(major_radius=1., minor_radius=.2)
            with self.assertRaisesRegex(ValueError, "concave"):
                build_model(model_path, scene, .005)

    def test_free_root_falls_despite_stationary_reference(self):
        """A complete replay must fail tracking rather than teleport its root."""
        import mujoco

        model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.005" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="10 10 .1" pos="0 0 -50"/>
            <body name="pelvis"><freejoint/><geom type="sphere" size=".1" mass="1"/>
              <body name="limb" pos="0 0 -.2"><joint name="hinge"/>
                <geom type="sphere" size=".05" mass=".1"/></body>
            </body>
          </worldbody>
        </mujoco>""")
        spec = SimpleNamespace(base_link="pelvis", joint_names=["hinge"],
                               controlled_joint_names=["hinge"], default_pose=[0.],
                               stiffness=[10.], damping=[1.], effort_limits=[5.],
                               armature=[.01], control_dt=.02)
        frames = 51
        position = np.tile([0., 0., 1.], (frames, 1, 1))
        arrays = dict(fps=np.array([50.]), joint_names=np.array(["hinge"]),
                      joint_pos=np.zeros((frames, 1)), joint_vel=np.zeros((frames, 1)),
                      body_names=np.array(["pelvis"]), body_pos_w=position,
                      body_quat_w=np.tile([1., 0., 0., 0.], (frames, 1, 1)),
                      body_lin_vel_w=np.zeros((frames, 1, 3)),
                      body_ang_vel_w=np.zeros((frames, 1, 3)))
        report, trajectory = simulate(model, {model.geom("floor").id}, spec, arrays)
        self.assertTrue(report["complete"])
        self.assertEqual(report["recorded_num_frames"], frames)
        self.assertFalse(report["accepted"])
        self.assertGreater(report["root_position_max_error_m"], 4.)
        self.assertLess(trajectory["root_pos_w"][-1, 2], -3.)
        self.assertEqual(report["applied_root_force_max"], 0.)
        self.assertEqual(report["root_initializations"], 1)
        self.assertFalse(report["training_contact_labels_generated"])

    def test_contact_force_is_world_vertical_and_self_contacts_are_excluded(self):
        import mujoco

        model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep=".002"/><worldbody>
          <geom name="floor" type="plane" size="10 10 .1"/>
          <body name="robot" pos="0 0 .099"><freejoint/>
            <geom type="sphere" size=".1" mass="1"/></body>
          <body name="other_robot_link" pos="2 0 10"><freejoint/>
            <geom type="sphere" size=".1" mass="1"/></body>
          <body name="another_robot_link" pos="2.1 0 10"><freejoint/>
            <geom type="sphere" size=".1" mass="1"/></body>
        </worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertGreaterEqual(data.ncon, 2)  # includes a robot-robot collision
        forces, penetration, count = scene_contacts(
            model, data, {model.geom("floor").id},
            [model.body("robot").id, model.body("other_robot_link").id],
        )
        self.assertEqual(count, 1)
        self.assertGreater(forces[0, 2], 1.)
        np.testing.assert_allclose(forces[0, :2], 0., atol=1e-10)
        np.testing.assert_allclose(forces[1], 0., atol=1e-10)
        self.assertAlmostEqual(penetration, .001, places=6)


if __name__ == "__main__":
    unittest.main()
