"""Contact geometry, force buffers, tracking events and physical hold contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from morph_mimic.utils.box_task import CONTACT_BODY_NAMES
from morph_mimic.utils.contact_events import contact_event_report, commanded_landing_report, _match
from morph_mimic.utils.contact_runtime import (
    ContactRuntime, ContactRuntimeConfig, aggregate_contact_buffers, contact_environment_order,
)
from morph_mimic.utils.contact_surface import (
    _closest_segment_triangle, closest_scene_surface, collision_support_points,
    contact_point_metrics, load_contact_surface_spec, mass_weighted_com,
    nearest_collision_surface, quat_rotate, rigid_point_velocity, support_hull_margin,
)
from morph_mimic.utils.contact_tracking import ContactTracking, FIELDS
from morph_mimic.utils.stable_hold import convex_margin, optimize_stable_hold


# Collision surfaces, point velocities and measured support geometry.

DTYPE = torch.float64


ROOT = Path(__file__).resolve().parents[1]


def tensor(value):
    return torch.tensor(value, dtype=DTYPE)


class ContactSurfaceTests(unittest.TestCase):
    def test_rotation_at_contact_uses_offset_from_velocity_origin(self):
        velocity = rigid_point_velocity(tensor([1., 0., 0.]), tensor([0., 0., 2.]), tensor([0., .5, 0.]))
        torch.testing.assert_close(velocity, torch.zeros(3, dtype=DTYPE))

    def test_vertical_wall_metrics_use_wall_normal_and_keep_swing_invalid(self):
        value = contact_point_metrics(tensor([[.02, 1., 2.], [.02, 1., 2.]]),
            tensor([[.3, .4, .5], [.3, .4, .5]]), tensor([[0., 1., 2.]]), tensor([[1., 0., 0.]]),
            valid_mask=torch.tensor([True, False]))
        self.assertAlmostEqual(value["normal_distance_m"][0].item(), .02)
        self.assertAlmostEqual(value["tangent_speed_m_s"][0].item(), np.hypot(.4, .5))
        self.assertEqual(value["normal_distance_m"][1].item(), 0.)
        self.assertEqual(value["valid"].tolist(), [True, False])

    def test_orientation_antipodal_and_invalid_normals(self):
        value = contact_point_metrics(torch.zeros((2, 3), dtype=DTYPE), torch.zeros((2, 3), dtype=DTYPE),
            torch.zeros((2, 3), dtype=DTYPE), tensor([[0., 0., 1.], [0., 0., 0.]]),
            tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]), tensor([[-1., 0., 0., 0.], [1., 0., 0., 0.]]))
        self.assertEqual(value["orientation_error_rad"].tolist(), [0., 0.])
        self.assertEqual(value["valid"].tolist(), [True, False])

    def test_bounded_triangle_nearest_does_not_extend_wall_infinitely(self):
        spec = dict(triangles_m=[[[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        result = closest_scene_surface(tensor([[.2, .25, .25], [.2, 2., 0.]]), spec)
        torch.testing.assert_close(result["point_w"], tensor([[0., .25, .25], [0., 1., 0.]]))
        torch.testing.assert_close(result["normal_w"], tensor([[1., 0., 0.], [1., 0., 0.]]))
        self.assertAlmostEqual(result["distance_m"][1].item(), np.hypot(.2, 1.))
        self.assertAlmostEqual(result["signed_distance_m"][1].item(), .2)

    def test_segment_triangle_handles_interior_crossing_and_parallel_edge(self):
        triangle = tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        x, y = _closest_segment_triangle(tensor([[.2, .2, -1.]]), tensor([[.2, .2, 1.]]), triangle)
        torch.testing.assert_close(x, tensor([[.2, .2, 0.]]))
        torch.testing.assert_close(x, y)
        x, y = _closest_segment_triangle(tensor([[-1., .3, .2]]), tensor([[1., .3, .2]]), triangle)
        self.assertAlmostEqual((x-y).norm().item(), .2)

    def test_capsule_nearest_is_surface_point_and_not_link_origin(self):
        spec = dict(body_names=["foot"], triangles_m=[[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]],
            geometries=[dict(type="capsule", body_index=0,
                segment_endpoints_local_m=[[-.1, 0., -.03], [.1, 0., -.03]], radius_m=.01)])
        result = nearest_collision_surface(tensor([[[.5, .5, .05]]]), tensor([[[1., 0., 0., 0.]]]), spec)
        self.assertAlmostEqual(result["point_w"][0, 0, 2].item(), .01)
        self.assertAlmostEqual(result["signed_clearance_m"][0, 0].item(), .01)
        self.assertEqual(result["body_index"].tolist(), [0])

    def test_capsule_support_patch_retains_both_flat_ends(self):
        spec = dict(body_names=["foot"], geometries=[dict(type="capsule", body_index=0,
            segment_endpoints_local_m=[[-.1, 0., -.03], [.1, 0., -.03]], radius_m=.01)])
        result = collision_support_points(tensor([[[0., 0., .04]]]), tensor([[[1., 0., 0., 0.]]]), tensor([[0., 0., 1.]]), spec)
        torch.testing.assert_close(result["point_w"][0], tensor([[-.1, 0., 0.], [.1, 0., 0.]]))
        self.assertTrue(result["valid"].all())
        tilted = collision_support_points(tensor([[[0., 0., .04]]]),
            tensor([[[np.cos(.2), 0., np.sin(.2), 0.]]]), tensor([[0., 0., 1.]]), spec)
        self.assertEqual(tilted["valid"].sum().item(), 1)

    def test_box_support_face_has_real_vertices(self):
        spec = dict(body_names=["box"], geometries=[dict(type="box", body_index=0,
            pos_local_m=[0., 0., 0.], quat_local_wxyz=[1., 0., 0., 0.], half_extents_m=[1., 2., 3.])])
        result = collision_support_points(tensor([[[0., 0., 3.]]]), tensor([[[1., 0., 0., 0.]]]), tensor([[0., 0., 1.]]), spec)
        self.assertEqual(result["valid"].sum().item(), 4)
        torch.testing.assert_close(result["point_w"][result["valid"]][:, 2], torch.zeros(4, dtype=DTYPE))
        with self.assertRaisesRegex(ValueError, "boxes need support points"):
            nearest_collision_surface(tensor([[[0., 0., 3.]]]), tensor([[[1., 0., 0., 0.]]]), spec | {"triangles_m": [[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]})

    def test_com_uses_mass_and_rotated_inertial_offsets(self):
        result = mass_weighted_com(tensor([[[0., 0., 0.], [2., 0., 0.]]]),
            tensor([[[0., 0., 0., 1.], [1., 0., 0., 0.]]]), [1., 3.], [[1., 0., 0.], [0., 0., 0.]])
        torch.testing.assert_close(result, tensor([[1.25, 0., 0.]]))

    def test_support_hull_rejects_bounding_box_false_positive(self):
        points = tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]*2)
        result = support_hull_margin(tensor([[.2, .2, 1.], [.8, .8, 1.]]), points, torch.ones((2, 3), dtype=torch.bool))
        self.assertTrue(result["valid"].all())
        self.assertAlmostEqual(result["margin_m"][0].item(), .2)
        self.assertAlmostEqual(result["margin_m"][1].item(), -.6/np.sqrt(2))

    def test_support_hull_rotation_and_permutation_invariant(self):
        points = tensor([[[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]]])
        center = tensor([[.4, 0., 2.]])
        angle = .64
        rotation = tensor([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        a = support_hull_margin(center, points, torch.ones((1, 4), dtype=torch.bool))
        b = support_hull_margin(center@rotation.T, (points@rotation.T)[:, [2, 0, 3, 1]], torch.ones((1, 4), dtype=torch.bool))
        torch.testing.assert_close(a["margin_m"], b["margin_m"])

    def test_degenerate_and_nonplanar_support_not_fabricated_valid(self):
        points = tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
            [[0., 0., 0.], [1., 0., 0.], [0., 1., .2]],
            [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        active = torch.tensor([[True, True, True], [True, True, True], [True, False, False], [False, False, False]])
        result = support_hull_margin(tensor([[.5, .2, 1.]]*4), points, active)
        self.assertEqual(result["support_rank"].tolist(), [1, 2, 0, -1])
        self.assertEqual(result["valid"].tolist(), [False]*4)
        self.assertAlmostEqual(result["margin_m"][0].item(), -.2)
        self.assertTrue(torch.isnan(result["margin_m"][3]))

    def test_single_foot_with_finite_force_supported_patch_is_valid(self):
        points = tensor([[[-.1, -.03, 0.], [.1, -.03, 0.], [.1, .03, 0.], [-.1, .03, 0.], [4., 4., 0.]]])
        result = support_hull_margin(tensor([[0., 0., 1.]]), points, torch.tensor([[True, True, True, True, False]]))
        self.assertTrue(result["valid"].item())
        self.assertAlmostEqual(result["margin_m"].item(), .03)

    def test_support_hull_2d_duplicates_and_single_point(self):
        points = tensor([[[-1., -1.], [1., -1.], [1., 1.], [-1., 1.], [1., 1.]]])
        value = support_hull_margin(tensor([[0., 0.]]), points, torch.ones((1, 5), dtype=torch.bool))
        self.assertEqual(value["hull_point_count"].item(), 4)
        self.assertAlmostEqual(value["margin_m"].item(), 1.)
        single = support_hull_margin(tensor([1., 0.]), tensor([[0., 0.]]), torch.tensor([True]))
        self.assertEqual(single["support_rank"].item(), 0)
        self.assertFalse(single["valid"].item())
        self.assertAlmostEqual(single["margin_m"].item(), -1.)

    def test_invalid_active_contact_cannot_silently_shrink_valid_hull(self):
        points = tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [float("nan"), 0., 0.]]])
        bad = support_hull_margin(tensor([[.2, .2, 1.]]), points, torch.ones((1, 4), dtype=torch.bool))
        self.assertFalse(bad["valid"].item())
        masked = support_hull_margin(tensor([[.2, .2, 1.]]), points, torch.tensor([[True, True, True, False]]))
        self.assertTrue(masked["valid"].item())
        self.assertAlmostEqual(masked["margin_m"].item(), .2)

    def test_source_spec_covers_massive_bodies_and_exact_contacts(self):
        motion = next((ROOT/"datasets/g1_box_climb_seed_v2").glob("*.npz"))
        # Avoid a contact-label NPZ if directory order changes.
        motion = next(path for path in motion.parent.glob("*.npz") if not path.stem.endswith("_contacts"))
        spec = load_contact_surface_spec(motion)
        self.assertEqual(len(spec["body_names"]), 30)
        self.assertTrue(spec["whole_body_mass_covered"])
        self.assertEqual(len(spec["triangles_m"]), 36)
        self.assertGreater(sum(spec["masses_kg"]), 30.)
        self.assertEqual({geom["type"] for geom in spec["geometries"]}, {"capsule", "sphere"})
        foot = [geom for geom in spec["geometries"] if geom["body_name"] == "left_ankle_roll_link"]
        self.assertEqual(len(foot), 7)
        self.assertTrue(all(geom["radius_m"] == .01 for geom in foot))
        partial = load_contact_surface_spec(motion, body_names=["left_ankle_roll_link"])
        self.assertFalse(partial["whole_body_mass_covered"])
        self.assertEqual(len(partial["geometries"]), 7)
        with self.assertRaises(ValueError):
            load_contact_surface_spec(motion, body_names=["no_such_body"])

    def test_unsupported_collision_type_fails_explicitly(self):
        motion = next(path for path in (ROOT/"datasets/g1_box_climb_seed_v2").glob("*.npz") if not path.stem.endswith("_contacts"))
        source = (ROOT/"assets/g1/robot/g1.xml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)/"unsupported.xml"
            model.write_text(source.replace('name="pelvis_collision" class="collision" type="sphere"', 'name="pelvis_collision" class="collision" type="ellipsoid"'))
            with self.assertRaisesRegex(ValueError, "unsupported collision primitive ellipsoid"):
                load_contact_surface_spec(motion, robot_xml=model)


# Contact buffer indexing, force reconstruction and runtime health gates.

def fixtures():
    # The two environments' pair ranges deliberately appear in reverse order.
    points = torch.full((8, 3), float('nan'))
    normals = points.clone()
    force = torch.full((8, 1), float('nan'))
    separation = force.clone()
    points[4], points[5], points[0] = torch.tensor([0., 0., 0.]), torch.tensor([2., 0., 0.]), torch.tensor([0., 0., 1.])
    normals[4] = normals[5] = torch.tensor([0., 0., 1.])
    normals[0] = torch.tensor([1., 0., 0.])
    force[4], force[5], force[0] = 5., 15., 10.
    separation[[0, 4, 5]] = 0.
    normal = [force, points, normals, separation,
              torch.tensor([[2], [1]], dtype=torch.int32), torch.tensor([[4], [0]], dtype=torch.int32)]
    friction = torch.full((8, 3), float('nan'))
    friction_points = friction.clone()
    friction[2], friction[6] = torch.tensor([-2., 0., 0.]), torch.tensor([0., -3., 0.])
    friction_points[2], friction_points[6] = torch.tensor([1., 0., 0.]), torch.tensor([0., 0., 1.])
    patch = [friction, friction_points, torch.tensor([[1], [1]], dtype=torch.int32), torch.tensor([[2], [6]], dtype=torch.int32)]
    kwargs = dict(com_pos_w=torch.tensor([[1., 0., 0.], [0., 0., 0.]]),
        com_lin_vel_w=torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
        angular_vel_w=torch.tensor([[0., 0., 2.], [0., 0., 0.]]),
        force_matrix_w=torch.tensor([[0., 0., 20.], [10., 0., 0.]]),
        config=ContactRuntimeConfig(contacts_per_body=4))
    return normal, patch, kwargs


class ContactBufferTests(unittest.TestCase):
    def test_measured_wrench_uses_each_buffers_own_patch_locations(self):
        normal, friction, kwargs = fixtures()
        result = aggregate_contact_buffers(normal, friction, **kwargs)
        self.assertTrue(result['health_gate'].all())
        torch.testing.assert_close(result['total_force_w'], torch.tensor([[-2., 0., 20.], [10., -3., 0.]]))
        torch.testing.assert_close(result['torque_about_com_w'], torch.tensor([[0., -10., 0.], [3., 10., 0.]]))
        torch.testing.assert_close(result['normal_force_weighted_point_w'], torch.tensor([[1.5, 0., 0.], [0., 0., 1.]]))
        torch.testing.assert_close(result['friction_power_w'], torch.tensor([-2., -3.]))

    def test_contact_velocity_uses_inertial_com_and_angular_offset(self):
        normal, friction, kwargs = fixtures()
        result = aggregate_contact_buffers(normal, friction, **kwargs)
        torch.testing.assert_close(result['load_weighted_point_velocity_w'], torch.tensor([[1., 1., 0.], [0., 1., 0.]]))
        torch.testing.assert_close(result['load_weighted_tangent_speed_m_s'], torch.tensor([5.**.5, 1.]))
        torch.testing.assert_close(result['normal_contact_velocities_w'][0, :2], torch.tensor([[1., -2., 0.], [1., 2., 0.]]))

    def test_no_contact_is_explicit_absence_not_fake_zero_position(self):
        normal, friction, kwargs = fixtures()
        normal[4].zero_()
        friction[2].zero_()
        kwargs['force_matrix_w'].zero_()
        result = aggregate_contact_buffers(normal, friction, **kwargs)
        self.assertTrue(result['health_gate'].all())
        self.assertFalse(result['contact_valid'].any())
        self.assertFalse(result['normal_contact_valid'].any())
        self.assertTrue(result['normal_force_weighted_point_w'].isnan().all())
        self.assertTrue((result['normal_force_w'] == 0).all())

    def test_invalid_normal_and_force_reconstruction_fail_gate(self):
        normal, friction, kwargs = fixtures()
        normal[2][4] *= 2.
        result = aggregate_contact_buffers(normal, friction, **kwargs)
        self.assertFalse(result['health_gate'][0])
        self.assertTrue(result['health_gate'][1])
        normal, friction, kwargs = fixtures()
        kwargs['force_matrix_w'][1, 0] += 1.
        result = aggregate_contact_buffers(normal, friction, **kwargs)
        self.assertTrue(result['health_gate'][0])
        self.assertFalse(result['health_gate'][1])

    def test_bad_ranges_overlap_and_full_capacity_fail(self):
        for counts, starts in (([[2], [2]], [[7], [0]]),
                               ([[2], [2]], [[0], [1]]),
                               ([[4], [4]], [[0], [4]])):
            normal, friction, kwargs = fixtures()
            normal[4] = torch.tensor(counts)
            normal[5] = torch.tensor(starts)
            result = aggregate_contact_buffers(normal, friction, **kwargs)
            self.assertFalse(result['health_gate'].all())

    def test_export_width_overflow_fails_instead_of_truncating_support(self):
        normal, friction, kwargs = fixtures()
        normal[4][0, 0] = 5
        normal[5][0, 0] = 1
        result = aggregate_contact_buffers(normal, friction, **kwargs)
        self.assertFalse(result['health_gate'][0])
        self.assertFalse(result['normal_contact_valid'][0].any())

    def test_nonfinite_active_friction_point_fails(self):
        normal, friction, kwargs = fixtures()
        friction[1][2, 0] = float('nan')
        result = aggregate_contact_buffers(normal, friction, **kwargs)
        self.assertFalse(result['health_gate'][0])
        self.assertTrue(result['health_gate'][1])

    def test_padded_points_preserve_terrain_points_without_fk(self):
        normal, friction, kwargs = fixtures()
        result = aggregate_contact_buffers(normal, friction, **kwargs)
        self.assertEqual(result['normal_contact_points_w'].shape, (2, 4, 3))
        torch.testing.assert_close(result['normal_contact_points_w'][0, :2], normal[1][4:6])
        torch.testing.assert_close(result['normal_contact_forces_n'][0], torch.tensor([5., 15., 0., 0.]))
        self.assertEqual(result['normal_contact_valid'].tolist(), [[True, True, False, False], [True, False, False, False]])


class RuntimeTests(unittest.TestCase):
    def test_authoritative_path_matching_handles_arbitrary_and_nested_orders(self):
        roots = ['/World/envs/env_2/Robot/pelvis/pelvis', '/World/envs/env_10/Robot/pelvis/pelvis', '/World/envs/env_1/Robot/pelvis/pelvis']
        sensors = ['/World/envs/env_1/Robot/pelvis/foot', '/World/envs/env_2/Robot/pelvis/foot', '/World/envs/env_10/Robot/pelvis/foot']
        self.assertEqual(contact_environment_order(roots, sensors, 'foot'), [1, 2, 0])
        for broken in ([sensors[0], sensors[0], sensors[2]], sensors[:-1],
                       [sensors[0], sensors[1], '/World/envs/env_99/Robot/pelvis/foot']):
            with self.assertRaises(RuntimeError):
                contact_environment_order(roots, broken, 'foot')

    def test_missing_body_coverage_is_rejected(self):
        env = SimpleNamespace(robot=SimpleNamespace(body_names=['foot', 'pelvis']))
        with self.assertRaisesRegex(RuntimeError, 'coverage'):
            ContactRuntime(env)

    def test_batched_views_preserve_articulation_body_order_and_buffer_offsets(self):
        normal, friction, kwargs = fixtures()
        class View:
            sensor_count = 2
            filter_count = 1
            max_contact_data_count = 8
            def __init__(self, scale, name):
                self.sensor_paths = [f'/World/envs/env_{i}/Robot/{name}' for i in range(2)]
                self.normal = [value.clone() for value in normal]
                self.friction = [value.clone() for value in friction]
                self.normal[0] *= scale
                self.matrix = kwargs['force_matrix_w'] * scale
            def get_contact_data(self, dt):
                return self.normal
            def get_friction_data(self, dt):
                return self.friction
            def get_contact_force_matrix(self, dt):
                return self.matrix[:, None]
        foot = SimpleNamespace(contact_physx_view=View(1., 'foot'), body_names=['foot'])
        pelvis = SimpleNamespace(contact_physx_view=View(2., 'pelvis'), body_names=['pelvis'])
        data = SimpleNamespace(**{name: kwargs[source][:, None].expand(-1, 2, -1)
            for name, source in [('body_com_pos_w', 'com_pos_w'),
                ('body_com_lin_vel_w', 'com_lin_vel_w'), ('body_com_ang_vel_w', 'angular_vel_w')]})
        env = SimpleNamespace(robot=SimpleNamespace(body_names=['pelvis', 'foot'], data=data, root_physx_view=SimpleNamespace(prim_paths=['/World/envs/env_0/Robot', '/World/envs/env_1/Robot'])), num_envs=2,
            physics_dt=.004, full_scene_contact_sensors=[('foot', foot), ('pelvis', pelvis)],
            full_scene_contact_config=kwargs['config'])
        result = ContactRuntime(env).sample()
        self.assertTrue(result['all_body_health_gate'].all())
        torch.testing.assert_close(result['normal_force_w'][:, 0], kwargs['force_matrix_w'] * 2.)
        torch.testing.assert_close(result['normal_force_w'][:, 1], kwargs['force_matrix_w'])
        torch.testing.assert_close(result['normal_contact_points_w'][0, 0, :2], normal[1][4:6])
        torch.testing.assert_close(result['normal_contact_points_w'][0, 1, :2], normal[1][4:6])
        self.assertEqual(result['normal_point_count'].tolist(), [[2, 2], [1, 1]])

    def test_raw_normal_count_buffers_are_snapshotted_before_friction(self):
        normal, friction, kwargs = fixtures()
        class View:
            sensor_count = 2
            filter_count = 1
            max_contact_data_count = 8
            sensor_paths = ['/World/envs/env_0/Robot/foot', '/World/envs/env_1/Robot/foot']
            sensor_names = ['foot', 'foot']
            def get_contact_data(self, dt):
                self.normal_dt = dt
                return normal
            def get_friction_data(self, dt):
                self.friction_dt = dt
                normal[4].copy_(friction[2])
                normal[5].copy_(friction[3])
                return friction
            def get_contact_force_matrix(self, dt):
                return kwargs['force_matrix_w'][:, None]
        view = View()
        sensor = SimpleNamespace(contact_physx_view=view, body_names=['foot'])
        data = SimpleNamespace(body_com_pos_w=kwargs['com_pos_w'].flip(0)[:, None],
            body_com_lin_vel_w=kwargs['com_lin_vel_w'].flip(0)[:, None], body_com_ang_vel_w=kwargs['angular_vel_w'].flip(0)[:, None])
        env = SimpleNamespace(robot=SimpleNamespace(body_names=['foot'], data=data, root_physx_view=SimpleNamespace(prim_paths=['/World/envs/env_1/Robot', '/World/envs/env_0/Robot'])), num_envs=2,
            physics_dt=.004, full_scene_contact_sensors=[('foot', sensor)],
            full_scene_contact_config=kwargs['config'])
        runtime = ContactRuntime(env)
        result = runtime.sample()
        self.assertTrue(result['all_body_health_gate'].all())
        self.assertEqual(view.normal_dt, .004)
        self.assertEqual(view.friction_dt, .004)
        self.assertEqual(result['normal_point_count'].tolist(), [[1], [2]])
        torch.testing.assert_close(result['normal_force_w'][:, 0], kwargs['force_matrix_w'].flip(0))
        self.assertEqual(runtime.environment_order_for_sensor(sensor).tolist(), [1, 0])
        self.assertEqual(result['normal_contact_points_w'].shape, (2, 1, 4, 3))


# Reference-aligned contact tracking and physical hold acceptance.

class ContactTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.motion = next(p for p in (ROOT/"datasets/g1_box_climb_seed_v2").glob("*.npz") if not p.stem.endswith("_contacts"))
        spec = load_contact_surface_spec(cls.motion)
        with np.load(cls.motion, allow_pickle=False) as archive:
            positions = torch.tensor(archive["body_pos_w"][-1:], dtype=torch.float64)
            positions[..., :2] -= torch.tensor(spec["source_first_root_xy"])
            quats = torch.tensor(archive["body_quat_w"][-1:], dtype=torch.float64)
        masses = torch.tensor(spec["masses_kg"], dtype=torch.float64)[None]
        localcom = torch.tensor(spec["com_local_m"], dtype=torch.float64)
        origin = torch.tensor([[7., -3., .2]], dtype=torch.float64)
        positions += origin[:, None]
        data = SimpleNamespace(body_pos_w=positions, body_quat_w=quats,
            body_com_pos_w=positions+quat_rotate(quats, localcom),
            body_com_lin_vel_w=torch.zeros_like(positions), body_com_ang_vel_w=torch.zeros_like(positions))
        robot = SimpleNamespace(body_names=spec["body_names"], data=data,
            root_physx_view=SimpleNamespace(get_masses=lambda: masses))
        env = SimpleNamespace(robot=robot, device="cpu", num_envs=1, phase=torch.ones(1, dtype=torch.float64),
            is_ref_env=torch.ones(1, dtype=torch.bool), contact_link_names=list(CONTACT_BODY_NAMES),
            cell_offset=torch.zeros((1, 3), dtype=torch.float64), env_origin=lambda: origin)
        cls.template = ContactTracking(env, cls.motion)
        cls.base_data = data
        cls.origin = origin
        cls.spec = spec
        cls.masses = masses

    def setUp(self):
        self.env = SimpleNamespace(**vars(self.template.env))
        self.env.robot = SimpleNamespace(**vars(self.template.env.robot))
        self.env.robot.data = SimpleNamespace(**{k: v.clone() for k, v in vars(self.base_data).items()})
        self.env.phase = self.env.phase.clone()
        self.tracking = object.__new__(ContactTracking)
        self.tracking.__dict__ = dict(self.template.__dict__)
        self.tracking.env = self.env
        self.tracking.last = None
        self.measured = self._measured_support()

    def _measured_support(self):
        dtype = torch.float64
        z = lambda *shape: torch.zeros(shape, dtype=dtype)
        b = lambda *shape: torch.zeros(shape, dtype=torch.bool)
        result = dict(health_gate=torch.ones((1, 30), dtype=torch.bool), all_body_health_gate=torch.ones(1, dtype=torch.bool),
            normal_load_n=z(1, 30), total_force_w=z(1, 30, 3), torque_about_com_w=z(1, 30, 3),
            contact_valid=b(1, 30), normal_force_weighted_point_w=torch.full((1,30,3), float("nan"), dtype=dtype),
            load_weighted_normal_w=torch.full((1,30,3), float("nan"), dtype=dtype), load_weighted_tangent_speed_m_s=z(1,30),
            normal_contact_points_w=torch.full((1,30,16,3), float("nan"), dtype=dtype),
            normal_contact_normals_w=torch.full((1,30,16,3), float("nan"), dtype=dtype),
            normal_contact_forces_n=z(1,30,16), normal_contact_valid=b(1,30,16))
        support = collision_support_points(self.env.robot.data.body_pos_w, self.env.robot.data.body_quat_w,
            torch.tensor([[0.,0.,1.]], dtype=dtype), self.tracking.surface_spec)
        for bi in self.tracking.feet.tolist():
            selected = (support["body_index"] == bi) & support["valid"][0]
            points = support["point_w"][0,selected]
            count = len(points)
            force_each = self.masses.sum()*9.81/(2*count)
            result["normal_contact_points_w"][0,bi,:count] = points
            result["normal_contact_normals_w"][0,bi,:count] = torch.tensor([0.,0.,1.],dtype=dtype)
            result["normal_contact_forces_n"][0,bi,:count] = force_each
            result["normal_contact_valid"][0,bi,:count] = True
            result["contact_valid"][0,bi] = True
            result["normal_load_n"][0,bi] = count*force_each
            result["total_force_w"][0,bi,2] = count*force_each
            result["normal_force_weighted_point_w"][0,bi] = points.mean(0)
            result["load_weighted_normal_w"][0,bi] = torch.tensor([0.,0.,1.],dtype=dtype)
            per_force = torch.tensor([0.,0.,force_each],dtype=dtype).expand_as(points)
            result["torque_about_com_w"][0,bi] = torch.linalg.cross(points-self.env.robot.data.body_com_pos_w[0,bi], per_force).sum(0)
        return result

    def test_exact_reference_has_finite_fields_and_flags_unstable_endpoint(self):
        value = self.tracking.update(self.measured)
        # The source kinematic endpoint is not a certified static stance:
        # its source-mass COM lies outside the foot support candidate hull.
        # Keep this measured conflict visible instead of weakening the gate.
        self.assertFalse(self.tracking.reference_endpoint_potential["geometry_gate"])
        self.assertLess(self.tracking.reference_endpoint_potential["support_hull_margin_m"], -.04)
        self.assertFalse(self.tracking.reference_endpoint_potential["physical_force_solution_verified"])
        for key, shape in FIELDS.items():
            self.assertEqual(value[key].shape, (1,*shape), key)
            self.assertTrue(torch.isfinite(value[key]).all(), key)
        self.assertLess(value["candidate_normal_distance_error_m"].abs().max().item(), 1e-6)
        self.assertTrue(value["measurement_health_gate"].item())
        self.assertTrue(value["support_hull_valid"].item())
        self.assertFalse(value["hold_gate"].item())
        self.assertFalse(value["cop_valid"].item())
        self.assertLess(value["orientation_error_rad"].abs().max().item(), 1e-3)

    def test_hold_gate_accepts_supported_slow_com_with_finite_foot_points(self):
        # This is an explicit synthetic COM/sensor contract test, not a
        # physically realized trajectory or a repaired source reference.
        points = self.measured["normal_contact_points_w"]
        active = self.measured["normal_contact_valid"]
        center = points[active].mean(0)
        self.env.robot.data.body_com_pos_w[:,:,:2] = center[:2]
        for bi in self.tracking.feet.tolist():
            selected = active[0,bi]
            points[0,bi,selected,2] = self.origin[0,2]+self.tracking.top_z
        value = self.tracking.update(self.measured)
        self.assertTrue(value["com_hull_gate"].item())
        self.assertTrue(value["hold_gate"].item())

    def test_hold_stability_reward_requires_measured_hull_and_stays_off_during_motion(self):
        points = self.measured["normal_contact_points_w"]
        active = self.measured["normal_contact_valid"].clone()
        center = points[active].mean(0)
        self.env.robot.data.body_com_pos_w[:,:,:2] = center[:2]
        stable = self.tracking.update(self.measured)["reward_terms"]["hold_support_stability"].item()
        self.assertGreater(stable, .5)
        self.env.robot.data.body_com_lin_vel_w[:,:,0] = .3
        fast = self.tracking.update(self.measured)["reward_terms"]["hold_support_stability"].item()
        self.assertLess(fast, stable)
        self.env.robot.data.body_com_lin_vel_w[:] = 0.
        self.env.robot.data.body_com_pos_w[:,:,0] += 1.
        outside = self.tracking.update(self.measured)["reward_terms"]["hold_support_stability"].item()
        self.assertLess(outside, stable)
        self.measured["normal_contact_valid"][:] = False
        self.assertEqual(self.tracking.update(self.measured)["reward_terms"]["hold_support_stability"].item(),0.)
        self.measured["normal_contact_valid"][:] = active
        self.env.phase[:] = .5
        self.assertEqual(self.tracking.update(self.measured)["reward_terms"]["hold_support_stability"].item(),0.)

    def test_same_local_offset_uses_actual_com_velocity_origin(self):
        actual = self.tracking.update(self.measured)
        bi = self.tracking.canonical[6].item()
        point = actual["actual_collision_candidate_rebased_m"][0,6]+self.origin[0]
        omega = torch.tensor([0.,0.,2.],dtype=torch.float64)
        self.env.robot.data.body_com_ang_vel_w[0,bi] = omega
        self.env.robot.data.body_com_lin_vel_w[0,bi] = -torch.linalg.cross(omega, point-self.env.robot.data.body_com_pos_w[0,bi])
        changed = self.tracking.update(self.measured)
        self.assertLess(changed["candidate_point_velocity_w_m_s"][0,6].norm().item(), 1e-10)

    def test_nonfoot_body_force_and_measurement_failure_prevent_hold(self):
        hip = self.env.robot.body_names.index("left_hip_roll_link")
        self.measured["normal_load_n"][0,hip] = 6.
        value = self.tracking.update(self.measured)
        self.assertFalse(value["nonfoot_force_gate"].item())
        self.assertFalse(value["hold_gate"].item())
        self.assertGreater(value["reward_terms"]["unexpected_noncanonical_contact"].item(), 0.)
        self.measured["normal_load_n"][0,hip] = 0.
        self.measured["health_gate"][0,hip] = False
        self.measured["all_body_health_gate"][0] = False
        unhealthy = self.tracking.update(self.measured)
        self.assertFalse(unhealthy["hold_gate"].item())
        self.assertTrue(all(v.item() == 0 for v in unhealthy["reward_terms"].values()))

    def test_com_speed_is_only_hold_gate(self):
        self.env.robot.data.body_com_lin_vel_w[:,:,0] = .21
        value = self.tracking.update(self.measured)
        self.assertFalse(value["com_speed_gate"].item())
        self.assertFalse(value["hold_gate"].item())
        self.env.phase[:] = .5
        value = self.tracking.update(self.measured)
        self.assertFalse(value["support_hull_valid"].item())
        self.assertEqual(value["support_point_count"].item(), 0)
        self.assertGreater(value["reward_terms"]["surface_orientation"].item(), 0.)

    def test_no_slip_penalty_for_reference_intentional_motion_or_missing_load(self):
        feet = self.tracking.feet
        self.measured["load_weighted_tangent_speed_m_s"][:,feet] = .5
        value = self.tracking.update(self.measured)
        self.assertGreater(value["reward_terms"]["stance_tangent_slip"].item(), 0.)
        self.measured["contact_valid"][:,feet] = False
        value = self.tracking.update(self.measured)
        self.assertEqual(value["reward_terms"]["stance_tangent_slip"].item(), 0.)
        self.measured["contact_valid"][:,feet] = True
        self.env.phase[:] = .5
        self.tracking.reference = dict(self.tracking.reference)
        self.tracking.reference["tangent_speed"] = torch.ones_like(self.tracking.reference["tangent_speed"])
        value = self.tracking.update(self.measured)
        self.assertEqual(value["reward_terms"]["stance_tangent_slip"].item(), 0.)

    def test_canonical_hand_and_knee_support_not_unexpected_during_motion(self):
        self.env.phase[:] = .4
        bi = self.tracking.canonical[0].item()
        self.measured["normal_load_n"][0,bi] = 100.
        value = self.tracking.update(self.measured)
        self.assertEqual(value["reward_terms"]["unexpected_noncanonical_contact"].item(), 0.)
        self.assertFalse(value["nonfoot_force_gate"].item())

    def test_degenerate_support_cannot_pass_and_missing_points_keep_valid_false(self):
        self.measured["normal_contact_valid"][:] = False
        bi = self.tracking.feet[0].item()
        self.measured["normal_contact_valid"][0,bi,0] = True
        value = self.tracking.update(self.measured)
        self.assertFalse(value["support_hull_valid"].item())
        self.assertEqual(value["support_hull_rank"].item(), 0)
        self.assertFalse(value["hold_gate"].item())
        self.assertEqual(value["actual_measured_point_valid"][0,:6].sum().item(), 0)
        self.assertTrue(torch.isfinite(value["actual_measured_point_rebased_m"]).all())

    def test_reference_events_are_complete_rebased_copied_arrays(self):
        event = self.tracking.reference_event_inputs()
        self.assertEqual(event["reference_flags"].shape, (501,9))
        self.assertEqual(event["reference_points_w"].shape, (501,9,3))
        self.assertTrue(event["reference_valid"].all())
        value = self.tracking.update(self.measured)
        np.testing.assert_allclose(event["reference_points_w"][-1], value["reference_surface_point_rebased_m"][0], atol=1e-8)
        event["reference_points_w"][:] = 999.
        self.assertLess(np.max(self.tracking.reference_event_inputs()["reference_points_w"]), 999.)

    def test_summary_requires_fifty_extra_distinct_physical_hold_controls(self):
        value = self.tracking.update(self.measured)
        arrays = {"contact_tracking_"+key:np.repeat(tensor.numpy(),51,axis=0)
            for key,tensor in value.items() if key in FIELDS}
        arrays["phase"] = np.ones(51)
        arrays["contact_tracking_hold_gate"][:] = True
        arrays["box_hold_controls"] = np.arange(51)
        passed = self.tracking.summary(arrays)
        self.assertTrue(passed["hold_passed"])
        self.assertEqual(passed["hold_observed_endpoints"],50)
        arrays["box_hold_controls"] = np.zeros(51)
        self.assertFalse(self.tracking.summary(arrays)["hold_passed"])
        del arrays["box_hold_controls"]
        self.assertFalse(self.tracking.summary(arrays)["hold_passed"])


# Contact onset coverage, timing and commanded landing errors.

class ContactEventsTest(unittest.TestCase):
    def fixture(self, samples=20, start_phase=0.):
        frames, bodies, fps, dt = 21, 2, 10., .1
        times = (np.arange(samples)+1)*dt
        flags = np.zeros((frames, bodies), dtype=bool)
        flags[5:9, 0], flags[14:19, 1] = True, True
        loads = np.zeros((samples, bodies))
        roots = start_phase*2+times
        loads[(roots >= .6-1e-8) & (roots < 1.-1e-8), 0] = 40.
        loads[(roots >= 1.5-1e-8) & (roots < 2.-1e-8), 1] = 40.
        normals = np.zeros((samples, bodies, 3))
        normals[..., 2] = 1.
        ref_normals = np.zeros((frames, bodies, 3))
        ref_normals[..., 2] = 1.
        points, ref_points = np.zeros_like(normals), np.zeros_like(ref_normals)
        points[..., 0] = .2
        return dict(times_s=times, phase=np.minimum(roots/2, 1.), actual_points_w=points,
            normal_load_n=loads, actual_normals_w=normals, reference_flags=flags,
            reference_points_w=ref_points, reference_normals_w=ref_normals,
            reference_fps=fps, control_dt_s=dt, start_phase=start_phase,
            body_names=['left_foot', 'right_foot'], onset_tolerance_s=.21,
            metadata=dict(sampling='post_physics_pre_reset_control_endpoints',
                control_dt_s=dt, start_phase=start_phase, motion_duration_s=2.))

    def test_onset_time_and_surface_distance_are_separate_and_json_safe(self):
        report = contact_event_report(**self.fixture())
        summary = report['summary']
        self.assertEqual(summary['matched_onsets'], 2)
        self.assertAlmostEqual(summary['mean_surface_placement_error_m'], .2)
        self.assertAlmostEqual(summary['mean_signed_onset_error_s'], .1)
        self.assertEqual(summary['reference_contact_duration_coverage_fraction'], 1.)
        self.assertFalse(report['paper_fig7_equivalent'])
        json.dumps(report, allow_nan=False)

    def test_early_failure_reports_future_events_without_false_negatives(self):
        report = contact_event_report(**self.fixture(samples=10))
        self.assertEqual(report['summary']['matched_onsets'], 1)
        self.assertEqual(report['summary']['reference_events_not_reached'], 1)
        self.assertEqual(report['summary']['reference_onsets_missed_with_complete_window'], 0)
        self.assertLess(report['summary']['reference_contact_duration_coverage_fraction'], 1.)
        future = report['bodies']['right_foot']['reference_events'][0]
        self.assertEqual(future['detection_status'], 'not_evaluated')

    def test_last_unmatched_onset_has_censored_matching_window(self):
        args = self.fixture(samples=15)
        args['normal_load_n'][:, 1] = 0
        report = contact_event_report(**args)
        event = report['bodies']['right_foot']['reference_events'][0]
        self.assertEqual(event['detection_status'], 'matching_window_censored')
        self.assertEqual(report['summary']['reference_onsets_missed_with_complete_window'], 0)

    def test_missing_force_is_observed_miss_when_window_is_complete(self):
        args = self.fixture()
        args['normal_load_n'][:] = 0
        report = contact_event_report(**args)
        self.assertEqual(report['summary']['reference_onsets_missed_with_complete_window'], 2)
        self.assertIsNone(report['summary']['mean_surface_placement_error_m'])
        self.assertFalse(report['summary']['spatial_error_available'])

    def test_normal_mismatch_preserves_contact_presence_separately(self):
        args = self.fixture()
        args['actual_normals_w'][:] = [1., 0., 0.]
        report = contact_event_report(**args)
        self.assertEqual(report['summary']['actual_onsets_observed'], 2)
        self.assertEqual(report['summary']['matched_onsets'], 0)
        self.assertEqual(report['summary']['reference_onsets_normal_incompatible'], 2)
        self.assertEqual(report['summary']['reference_onsets_missed_with_complete_window'], 0)
        self.assertGreater(report['summary']['endpoint_contact_presence']['tp'], 0)

    def test_missing_continuous_initial_support_is_visible_despite_censored_onset(self):
        args = self.fixture()
        args['reference_flags'][:] = True
        args['normal_load_n'][:] = 0
        report = contact_event_report(**args)
        self.assertEqual(report['summary']['reference_onsets_missed_with_complete_window'], 0)
        self.assertEqual(report['summary']['reference_onsets_censored'], 2)
        self.assertEqual(report['summary']['endpoint_contact_presence']['fn'], 40)
        self.assertEqual(report['summary']['endpoint_contact_presence']['recall'], 0.)

    def test_invalid_surface_normals_do_not_become_no_contact(self):
        args = self.fixture()
        args['actual_normals_w'][:] = np.nan
        report = contact_event_report(**args)
        self.assertEqual(report['summary']['actual_onsets_observed'], 2)
        self.assertEqual(report['summary']['reference_onsets_geometry_unavailable'], 2)
        self.assertEqual(report['summary']['reference_onsets_missed_with_complete_window'], 0)
        json.dumps(report, allow_nan=False)

    def test_unavailable_points_are_missing_error_instead_of_zero(self):
        args = self.fixture()
        args['actual_points_w'][:] = np.nan
        report = contact_event_report(**args)
        self.assertEqual(report['summary']['matched_onsets'], 2)
        self.assertEqual(report['summary']['spatial_error_events'], 0)
        self.assertIsNone(report['summary']['mean_surface_placement_error_m'])

    def test_physical_hold_accepts_clamped_phase_with_fresh_endpoint_time(self):
        args = self.fixture(samples=25)
        args['metadata']['post_reference_hold_is_physical'] = True
        report = contact_event_report(**args)
        self.assertEqual(report['sampling']['physical_hold_samples'], 5)
        self.assertEqual(report['sampling']['movement_samples'], 20)
        self.assertEqual(report['summary']['matched_onsets'], 2)
        self.assertFalse(report['sampling']['physical_authenticity_proven_by_this_report'])

    def test_hold_requires_declaration_and_cannot_reuse_frozen_time(self):
        args = self.fixture(samples=25)
        with self.assertRaisesRegex(ValueError, 'physical-hold'):
            contact_event_report(**args)
        args['metadata']['post_reference_hold_is_physical'] = True
        args['times_s'][20:] = 2.
        with self.assertRaisesRegex(ValueError, 'times must advance'):
            contact_event_report(**args)

    def test_movement_phase_cannot_stall_or_regress(self):
        args = self.fixture()
        args['phase'][5] = args['phase'][4]
        with self.assertRaisesRegex(ValueError, 'phase must follow|Repeated movement'):
            contact_event_report(**args)
        args = self.fixture()
        args['phase'][5] = args['phase'][4]-.1
        with self.assertRaisesRegex(ValueError, 'phase must follow'):
            contact_event_report(**args)

    def test_initial_contacts_and_partial_reset_onsets_are_censored(self):
        args = self.fixture(samples=8, start_phase=.6)
        report = contact_event_report(**args)
        self.assertEqual(report['summary']['reference_events_before_reset'], 1)
        self.assertEqual(report['summary']['matched_onsets'], 1)
        args = self.fixture()
        args['reference_flags'][:4, 0] = True
        args['normal_load_n'][:3, 0] = 40
        report = contact_event_report(**args)
        first = report['bodies']['left_foot']['reference_events'][0]
        self.assertEqual(first['detection_status'], 'onset_left_censored')
        self.assertEqual(report['summary']['actual_onsets_censored'], 1)

    def test_same_body_and_matching_use_maximum_cardinality(self):
        normal = np.array([0., 0., 1.])
        event = lambda t: dict(start_s=t, normal=normal, normal_valid=True)
        # A greedy .15 -> .2 match loses .3 -> .2; ordered DP preserves two.
        pairs = _match([event(.15), event(.3)], [event(0.), event(.2)], .16, .5)
        self.assertEqual(pairs, [(0, 0), (1, 1)])
        args = self.fixture()
        args['normal_load_n'][:] = args['normal_load_n'][:, ::-1]
        report = contact_event_report(**args)
        self.assertEqual(report['summary']['matched_onsets'], 0)

    def test_metadata_and_nonfinite_or_negative_loads_fail(self):
        for key, value in [('control_dt_s', .2), ('motion_duration_s', 3.), ('sampling', 'render_frames')]:
            args = self.fixture()
            args['metadata'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                contact_event_report(**args)
        for value in [-1., np.nan]:
            args = self.fixture()
            args['normal_load_n'][1, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                contact_event_report(**args)


class CommandedLandingTest(unittest.TestCase):
    def fixture(self):
        count, bodies = 8, 2
        command = np.zeros((count, bodies, 3))
        actual = command.copy()
        actual[:, 0] = [.3, .4, 0.]
        actual[:, 1] = [0., 0., 1.]
        load = np.zeros((count, bodies))
        load[2:4, 0], load[5:, 1] = 40., 50.
        return dict(times_s=(np.arange(count)+1)*.02, commanded_points_w=command,
            command_valid=np.ones((count, bodies), dtype=bool), actual_body_origins_w=actual,
            normal_load_n=load, control_dt_s=.02, body_names=['left_foot', 'right_foot'],
            metadata=dict(command_sampling='control_start_policy_input', command_anchor='body_origin',
                positions_frame='scene_rebased_by_source_first_root_xy'))

    def test_direct_command_comparison_has_no_reference_event_matching(self):
        report = commanded_landing_report(**self.fixture())
        summary = report['summary']
        self.assertEqual(summary['valid_landing_error_events'], 2)
        self.assertAlmostEqual(summary['mean_landing_error_m'], .75)
        self.assertAlmostEqual(summary['rmse_landing_error_m'], np.sqrt(.625))
        self.assertEqual(report['kind'], 'commanded_body_origin_onset_error')
        self.assertTrue(report['paper_metric_definition_compatible'])
        self.assertFalse(report['paper_benchmark_complete'])
        self.assertFalse(report['paper_author_implementation_verified'])
        event = report['bodies']['left_foot']['events'][0]
        self.assertAlmostEqual(event['command_time_s'], .04)
        self.assertAlmostEqual(event['endpoint_time_s'], .06)
        json.dumps(report, allow_nan=False)

    def test_absent_commands_are_counted_without_becoming_zero_error(self):
        args = self.fixture()
        args['command_valid'][:] = False
        args['commanded_points_w'][:] = np.nan
        summary = commanded_landing_report(**args)['summary']
        self.assertEqual(summary['observed_contact_onsets'], 2)
        self.assertEqual(summary['onsets_without_command'], 2)
        self.assertEqual(summary['valid_landing_error_events'], 0)
        self.assertIsNone(summary['mean_landing_error_m'])
        self.assertIsNone(summary['rmse_landing_error_m'])

    def test_initial_contact_is_censored_but_later_new_onsets_remain_measurable(self):
        args = self.fixture()
        args['normal_load_n'][:2, 1] = 20.
        report = commanded_landing_report(**args)
        self.assertEqual(report['summary']['actual_contact_events'], 3)
        self.assertEqual(report['summary']['initial_censored_events'], 1)
        self.assertEqual(report['summary']['valid_landing_error_events'], 2)
        self.assertIsNone(report['bodies']['right_foot']['events'][0]['landing_error_m'])

    def test_start_input_metadata_and_contiguous_endpoint_times_are_required(self):
        for key, value in [('command_sampling', 'post_physics_policy_input'),
                           ('command_anchor', 'scene_surface_point'),
                           ('actual_sampling', 'render_video_frames'), ('control_dt_s', .04)]:
            args = self.fixture()
            args['metadata'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                commanded_landing_report(**args)
        args = self.fixture()
        args['times_s'][4] = args['times_s'][3]
        with self.assertRaisesRegex(ValueError, 'times must advance'):
            commanded_landing_report(**args)

    def test_actual_unavailable_masks_and_invalid_policy_outputs_are_distinct(self):
        args = self.fixture()
        args['actual_valid'] = np.zeros((8, 2), dtype=bool)
        args['actual_body_origins_w'][:] = np.nan
        summary = commanded_landing_report(**args)['summary']
        self.assertEqual(summary['onsets_with_command'], 2)
        self.assertEqual(summary['onsets_without_actual_origin'], 2)
        self.assertIsNone(summary['mean_landing_error_m'])
        args = self.fixture()
        args['commanded_points_w'][2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Valid policy command anchors must be finite'):
            commanded_landing_report(**args)

    def test_common_rotation_translation_preserve_landing_distance(self):
        args = self.fixture()
        expected = commanded_landing_report(**args)['summary']['mean_landing_error_m']
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        for key in ['commanded_points_w', 'actual_body_origins_w']:
            args[key] = args[key] @ rotation.T + [12., -4., 3.]
        self.assertAlmostEqual(commanded_landing_report(**args)['summary']['mean_landing_error_m'], expected)

    def test_force_threshold_does_not_invent_onset_from_history_or_zero_load(self):
        args = self.fixture()
        args['normal_load_n'][:2, 0] = 1.
        report = commanded_landing_report(**args)
        self.assertEqual(report['summary']['initial_censored_events'], 0)
        self.assertEqual(report['summary']['observed_contact_onsets'], 2)


# Offline endpoint optimization with independent support verification.

class ConvexMarginTests(unittest.TestCase):
    def setUp(self):
        try:
            import scipy
        except ImportError:
            self.skipTest('SciPy is required for offline endpoint optimization')

    def test_signed_metric_margin_and_degenerate_support(self):
        points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        self.assertAlmostEqual(convex_margin([.5, .5], points)[0], .5)
        self.assertAlmostEqual(convex_margin([1.1, .5], points)[0], -.1)
        self.assertFalse(convex_margin([0., 0.], [[0., 0.], [1., 0.], [2., 0.]])[1])


class StableHoldIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mujoco
            import scipy
        except ImportError as exc:
            raise unittest.SkipTest('Source MuJoCo/SciPy required') from exc
        cls.motion = Path(__file__).resolve().parents[1]/'datasets/g1_box_climb_seed_v2/climb_03_z_scale_1.0_scene_fit_0000_0501.npz'
        if not cls.motion.is_file():
            raise unittest.SkipTest('Source-bound box reference unavailable')
        cls.result = optimize_stable_hold(cls.motion)

    def test_result_is_feasible_without_relaxing_limits_or_claiming_physics(self):
        from morph_mimic.utils.specs import RobotSpec
        result = self.result
        self.assertTrue(result['accepted'])
        self.assertTrue(result['source_unchanged'])
        self.assertGreaterEqual(result['after']['support_hull_margin_m'], .01)
        self.assertTrue(result['after']['support_hull_valid'])
        self.assertTrue(all(n >= 2 for n in result['after']['geometric_point_count_per_foot']))
        self.assertLess(result['changes']['root_translation_norm_m'], .10)
        self.assertLess(result['after']['root_tilt_rad'], .3)
        self.assertLessEqual(result['after']['scene_max_penetration_m'], .005)
        self.assertLessEqual(result['after']['self_max_penetration_m'], .005)
        self.assertEqual(result['constraints']['physics_steps'], 0)
        self.assertFalse(result['constraints']['collision_masks_modified'])
        spec = RobotSpec.load('g1')
        self.assertEqual(result['target']['joint_names'], spec.joint_names)
        q = np.asarray(result['target']['joint_pos_rad'])
        self.assertTrue(np.all(q >= np.asarray(spec.pos_limits_lower)-1e-8))
        self.assertTrue(np.all(q <= np.asarray(spec.pos_limits_upper)+1e-8))
        held = [spec.joint_names.index(n) for n in spec.held_joint_names]
        np.testing.assert_array_equal(q[held], np.asarray(spec.default_pose)[held])

    def test_independent_surface_support_and_com_match_report(self):
        import torch
        from morph_mimic.utils.contact_surface import (load_contact_surface_spec,
            collision_support_points, mass_weighted_com, support_hull_margin)
        from morph_mimic.utils.box_task import FOOT_BODY_NAMES, load_box_task_spec
        target = self.result['target']
        surface = load_contact_surface_spec(self.motion, body_names=target['body_names'])
        task = load_box_task_spec(self.motion)
        pos = torch.tensor(target['body_pos_world_m'], dtype=torch.float64)[None]
        pos[..., :2] -= torch.tensor(task['source_first_root_xy'])
        quat = torch.tensor(target['body_quat_wxyz'], dtype=torch.float64)[None]
        support = collision_support_points(pos, quat, torch.tensor([[0., 0., 1.]], dtype=torch.float64), surface)
        feet = torch.tensor([surface['body_names'].index(n) for n in FOOT_BODY_NAMES])
        mask = support['valid'] & (support['body_index'][None] == feet[:, None]).any(0)
        mask &= (support['point_w'][..., 2]-task['target_top']['top_z_m']).abs() <= .005
        com = mass_weighted_com(pos, quat, surface['masses_kg'], surface['com_local_m'])
        independently = support_hull_margin(com, support['point_w'], mask)
        self.assertTrue(independently['valid'][0])
        self.assertGreaterEqual(float(independently['margin_m'][0]), .01)
        np.testing.assert_allclose(com[0].numpy(), self.result['after']['com_rebased_m'], atol=1e-8)
        self.assertAlmostEqual(float(independently['margin_m'][0]), self.result['after']['support_hull_margin_m'], places=7)


if __name__ == "__main__":
    unittest.main()
