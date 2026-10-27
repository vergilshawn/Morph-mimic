"""Check bounded held-joint scene fitting and collision-pruning conservatism."""

import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

from morph_mimic.scripts.retarget_scene_reference import SceneFit
from morph_mimic.scripts.validate_scene_reference import _body_columns, build_model
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


@unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("scipy"), "MuJoCo and scipy required")
class TestSceneRetarget(unittest.TestCase):
    def model(self):
        spec = RobotSpec.load("g1")
        transform = np.eye(4)
        transform[:3, 3] = [.45, 0, .2]
        scene = SimpleNamespace(
            components=[SimpleNamespace(kind="box", transform=transform, size_m=[.4, .6, .4])],
            ground=SimpleNamespace(size_m=[10, 10], thickness_m=1, top_z_m=0, center_xy_m=[0, 0]),
        )
        model, scene_ids = build_model(PROJECT_ROOT / spec.runtime_assets["mjcf"], scene, spec.physics_dt)
        _, bodies = _body_columns(model, spec)
        return SceneFit(model, scene_ids, spec, bodies), spec

    def test_state_keeps_held_nominal_and_root_rotation_unit(self):
        fit, spec = self.model()
        controlled = np.asarray(spec.default_pose)[fit.controlled]
        variables = np.r_[controlled, [.02, -.01, .015], [.04, -.02, .03]]
        state = fit.state(variables, np.array([1, 2, .9]), np.array([1., 0, 0, 0]))
        np.testing.assert_allclose(state[:3], [1.02, 1.99, .915])
        self.assertAlmostEqual(float(np.linalg.norm(state[3:7])), 1)
        np.testing.assert_array_equal(state[fit.qids[fit.held]], np.asarray(spec.default_pose)[fit.held])
        np.testing.assert_array_equal(state[fit.qids[fit.controlled]], controlled)

    def test_aabb_pruning_keeps_every_pair_near_collision(self):
        import mujoco

        fit, spec = self.model()
        for height in (.7, .9, 1.5):
            variables = np.r_[np.asarray(spec.default_pose)[fit.controlled], np.zeros(6)]
            fit.data.qpos[:] = fit.state(variables, np.array([0, 0, height]), np.array([1., 0, 0, 0]))
            mujoco.mj_kinematics(fit.model, fit.data)
            pruned, _ = fit.distances(np.full(9, -1))
            brute = np.array([min(mujoco.mj_geomDistance(fit.model, fit.data, int(robot), int(scene), 5, None)
                                 for scene in fit.scene_ids) for robot in fit.robot_ids])
            close = brute <= .02
            np.testing.assert_allclose(pruned[close], brute[close], atol=1e-12)
            self.assertTrue((pruned[~close] >= .02 - 1e-12).all())
            self_pruned = fit.self_distances()
            self_brute = np.array([mujoco.mj_geomDistance(fit.model, fit.data, int(first), int(second), 5, None)
                                   for first, second in fit.self_pairs])
            close = self_brute <= .02
            np.testing.assert_allclose(self_pruned[close], self_brute[close], atol=1e-12)
            self.assertTrue((self_pruned[~close] >= .02 - 1e-12).all())

    def test_fit_preserves_input_and_respects_position_root_and_held_bounds(self):
        import mujoco

        fit, spec = self.model()
        root = np.array([0, 0, .9])
        variables = np.r_[np.asarray(spec.default_pose)[fit.controlled], np.zeros(6)]
        raw = np.tile(fit.state(variables, root, np.array([1., 0, 0, 0])), (2, 1))
        raw[:, fit.qids[spec.joint_names.index("waist_pitch_joint")]] = .2
        original = raw.copy()
        targets = []
        for state in raw:
            fit.data.qpos[:] = state
            mujoco.mj_kinematics(fit.model, fit.data)
            targets.append(fit.data.xpos[fit.body_ids].copy())
        fitted, _ = fit.optimize(raw, np.asarray(targets), np.full((2, 9), -1), 50, max_nfev=3)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(fitted[:, fit.qids[fit.held]], np.broadcast_to(np.asarray(spec.default_pose)[fit.held], (2, len(fit.held))))
        q = fitted[:, fit.qids]
        self.assertTrue((q >= np.asarray(spec.pos_limits_lower) - 1e-9).all())
        self.assertTrue((q <= np.asarray(spec.pos_limits_upper) + 1e-9).all())
        self.assertLessEqual(float(np.abs(fitted[:, :3] - root).max()), .03 + 1e-9)


if __name__ == "__main__":
    unittest.main()
