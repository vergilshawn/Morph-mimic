"""Measured contact and source-surface tracking additions for the v3 teacher.

This is an explicitly separate engineering supplement to ACP rewards. No
phase slowing, physical pose writing or contact inference from FK occurs.
Call update after physics and before reset; rewards read the resulting cache.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from morph_mimic.utils.box_task import CONTACT_BODY_NAMES, FOOT_BODY_NAMES, load_box_task_spec
from morph_mimic.utils.contact_surface import (
    collision_support_points, load_contact_surface_spec, mass_weighted_com,
    nearest_collision_surface, quat_rotate, rigid_point_velocity, support_hull_margin,
)


CONTACT_TRACKING_CONTRACT = dict(
    schema=1, scope="Engineering contact-surface tracking supplement, not original ACP rewards",
    positions_frame="scene_rebased_by_source_first_root_xy", sampling="post_physics_pre_reset_control_endpoints",
    canonical_body_names=list(CONTACT_BODY_NAMES), whole_body_count=30,
    normal_distance_bandwidth_m=.03, orientation_bandwidth_rad=.5,
    reference_stance_tangent_speed_max_m_s=.10, slip_bandwidth_m_s=.30,
    actual_contact_load_min_n=1., unexpected_force_scale_n=50.,
    hold_nonfoot_normal_load_max_n=5., hold_point_normal_force_min_n=.10,
    hold_point_upward_normal_min_cosine=.8, hold_point_plane_tolerance_m=.02,
    hold_com_hull_margin_min_m=.005, hold_com_xy_speed_max_m_s=.20,
    hold_foot_orientation_error_max_rad=.35,
    hold_stability_margin_bandwidth_m=.02, hold_stability_speed_bandwidth_m_s=.20,
    hold_stability_reward="phase>=1 and healthy measured rank-2 support only: sigmoid(margin/.02)*exp(-(COMxy_speed/.20)^2); otherwise zero",
    reward_masks="normal distance: reference label+valid surface; slip: reference stance+measured current load+health",
    normal_target="same reference bounded surface and normal, preserve reference collision-surface clearance",
    masses="actual backend masses snapshotted at initialization; refresh_masses required after mass randomization",
    weighted_point="normal-force-weighted measured surface point; not a general center of pressure",
    stability="COM projection hull is quasi-static; gates only during physical endpoint hold, never motion",
    cop="coplanar resultant-force lever-moment diagnostic, no independent free contact couple measured",
)


FIELDS = {
    "body_health_gate": (30,), "measurement_health_gate": (), "body_normal_load_n": (30,),
    "body_total_force_w_n": (30, 3), "body_torque_about_com_w_nm": (30, 3),
    "reference_contact_flags": (9,), "reference_stance_mask": (9,), "reference_surface_valid": (9,),
    "reference_surface_point_rebased_m": (9, 3), "reference_surface_normal_w": (9, 3),
    "reference_collision_point_rebased_m": (9, 3), "reference_point_tangent_speed_m_s": (9,),
    "actual_collision_candidate_rebased_m": (9, 3), "candidate_normal_distance_error_m": (9,),
    "candidate_point_velocity_w_m_s": (9, 3), "orientation_error_rad": (9,),
    "actual_measured_point_rebased_m": (9, 3), "actual_measured_normal_w": (9, 3),
    "actual_measured_point_valid": (9,), "actual_normal_load_n": (9,),
    "actual_tangent_speed_m_s": (9,), "stance_slip_valid": (9,),
    "com_rebased_m": (3,), "com_velocity_w_m_s": (3,), "com_valid": (), "com_xy_speed_m_s": (),
    "nonfoot_normal_load_max_n": (), "nonfoot_force_gate": (),
    "support_hull_margin_m": (), "support_hull_valid": (), "support_hull_rank": (),
    "support_point_count": (), "com_hull_gate": (), "com_speed_gate": (),
    "foot_orientation_error_rad": (2,), "foot_orientation_gate": (), "hold_gate": (),
    "cop_moment_derived_rebased_m": (3,), "cop_valid": (),
    "total_contact_force_w_n": (3,), "contact_moment_about_top_origin_w_nm": (3,),
}


def _conjugate(q):
    result = q.clone()
    result[..., 1:] *= -1
    return result


def _orientation_error(actual, reference):
    # Relative rotation avoids acos instability close to identity and is
    # antipodal-safe. Both source and backend use unit WXYZ quaternions.
    w = (actual*reference).sum(-1).abs().clamp(0, 1)
    return 2*torch.atan2((1-w.square()).clamp_min(0).sqrt(), w)


def _masked_mean(values, valid):
    return (torch.where(valid, values, torch.zeros_like(values))).sum(-1)/valid.sum(-1).clamp_min(1)


def _interpolate(values, phase, quaternion=False):
    position = phase.clamp(0, 1)*(len(values)-1)
    lower = position.floor().long()
    upper = (lower+1).clamp(max=len(values)-1)
    fraction = (position-lower).reshape((-1,)+(1,)*(values.ndim-1))
    a, b = values[lower], values[upper]
    if quaternion:
        b = torch.where((a*b).sum(-1, keepdim=True) < 0, -b, b)
    result = a+(b-a)*fraction
    if quaternion:
        result = result/result.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return result


class ContactTracking:
    fields = FIELDS

    def __init__(self, env, motion_path, task_spec=None):
        self.env = env
        self.body_names = list(env.robot.body_names)
        if len(self.body_names) != 30 or len(set(self.body_names)) != 30:
            raise ValueError("ContactTracking requires every one of the 30 named G1 bodies")
        if list(env.contact_link_names) != list(CONTACT_BODY_NAMES):
            raise ValueError("Canonical contact order differs from the explicit nine-body source contract")
        self.canonical = torch.tensor([self.body_names.index(n) for n in CONTACT_BODY_NAMES],
            device=env.device, dtype=torch.long)
        self.feet = torch.tensor([self.body_names.index(n) for n in FOOT_BODY_NAMES],
            device=env.device, dtype=torch.long)
        self.nonfoot = torch.tensor([i for i, n in enumerate(self.body_names) if n not in FOOT_BODY_NAMES],
            device=env.device, dtype=torch.long)
        self.noncanonical = torch.tensor([i for i, n in enumerate(self.body_names) if n not in CONTACT_BODY_NAMES],
            device=env.device, dtype=torch.long)
        self.surface_spec = load_contact_surface_spec(motion_path, body_names=self.body_names)
        self.task_spec = task_spec or load_box_task_spec(motion_path)
        self.top_z = float(self.task_spec["target_top"]["top_z_m"])
        self.motion_path = str(Path(motion_path).resolve())
        self._load_reference(motion_path)
        self.refresh_masses()
        self.last = None

    def refresh_masses(self, env_ids=None):
        """Snapshot actual backend masses; call after any mass randomization."""
        masses = self.env.robot.root_physx_view.get_masses().to(device=self.env.device,
            dtype=self.env.robot.data.body_com_pos_w.dtype)
        if masses.shape != (self.env.num_envs, len(self.body_names)):
            raise ValueError("Actual masses must have [environment,all-body] axes")
        if not torch.isfinite(masses).all() or not torch.all(masses > 0):
            raise ValueError("Actual body masses must be positive and finite")
        if env_ids is None or not hasattr(self, "masses"):
            self.masses = masses.clone()
        else:
            self.masses[env_ids] = masses[env_ids]

    def _load_reference(self, motion_path):
        with np.load(motion_path, allow_pickle=False) as archive:
            names = list(archive["body_names"])
            columns = [names.index(name) for name in self.body_names]
            pos = torch.tensor(archive["body_pos_w"][:, columns], dtype=torch.float64)
            quat = torch.tensor(archive["body_quat_w"][:, columns], dtype=torch.float64)
            linear = torch.tensor(archive["body_lin_vel_w"][:, columns], dtype=torch.float64)
            angular = torch.tensor(archive["body_ang_vel_w"][:, columns], dtype=torch.float64)
            self.reference_fps = float(archive["fps"])
        pos[..., :2] -= torch.tensor(self.surface_spec["source_first_root_xy"])
        candidates = nearest_collision_surface(pos, quat, self.surface_spec)
        canonical_cpu = self.canonical.cpu()
        frames, bodies = len(pos), len(canonical_cpu)
        offset, surface, normal, point = [torch.empty((frames, bodies, 3), dtype=pos.dtype) for _ in range(4)]
        valid = torch.zeros((frames, bodies), dtype=torch.bool)
        indices = torch.arange(frames)
        for column, bi in enumerate(canonical_cpu.tolist()):
            eligible = candidates["body_index"] == bi
            if not eligible.any():
                raise ValueError(f"No actual collision primitive for canonical body {self.body_names[bi]}")
            gap = torch.where(eligible[None], candidates["signed_clearance_m"].abs(), torch.inf)
            selected = gap.argmin(-1)
            point[:, column] = candidates["point_w"][indices, selected]
            surface[:, column] = candidates["surface_point_w"][indices, selected]
            normal[:, column] = candidates["normal_w"][indices, selected]
            valid[:, column] = candidates["valid"][indices, selected]
            offset[:, column] = quat_rotate(_conjugate(quat[:, bi]), point[:, column]-pos[:, bi])
        reference_velocity = rigid_point_velocity(linear[:, canonical_cpu], angular[:, canonical_cpu],
            point-pos[:, canonical_cpu])
        tangent_velocity = reference_velocity-(reference_velocity*normal).sum(-1, keepdim=True)*normal
        with np.load(Path(motion_path).with_name(Path(motion_path).stem+"_contacts.npz"), allow_pickle=False) as labels:
            label_names = list(labels["body_names"])
            flags = torch.tensor(labels["contact_flags"][:, [label_names.index(n) for n in CONTACT_BODY_NAMES]], dtype=torch.bool)
        if flags.shape != valid.shape:
            raise ValueError("Surface reference and contact labels must cover the same complete clip")
        valid &= torch.isfinite(offset).all(-1) & torch.isfinite(surface).all(-1) & torch.isfinite(normal).all(-1)
        valid &= torch.isfinite(reference_velocity).all(-1)
        # Offline feasibility is geometric potential only; actual force support
        # and dynamic balance cannot be certified from this reference cache.
        support = collision_support_points(pos[-1:], quat[-1:], torch.tensor([[0., 0., 1.]], dtype=pos.dtype), self.surface_spec)
        support_feet = (support["body_index"][None] == self.feet.cpu()[:, None]).any(0)
        support_mask = support["valid"] & support_feet
        support_mask &= (support["point_w"][..., 2]-self.top_z).abs() <= CONTACT_TRACKING_CONTRACT["hold_point_plane_tolerance_m"]
        com = mass_weighted_com(pos[-1:], quat[-1:], self.surface_spec["masses_kg"], self.surface_spec["com_local_m"])
        potential = support_hull_margin(com, support["point_w"], support_mask)
        self.reference_endpoint_potential = dict(
            com_rebased_m=com[0].tolist(), support_hull_margin_m=float(potential["margin_m"][0]),
            support_hull_valid=bool(potential["valid"][0]),
            geometry_gate=bool(potential["valid"][0] & (potential["margin_m"][0] >= CONTACT_TRACKING_CONTRACT["hold_com_hull_margin_min_m"])),
            physical_force_solution_verified=False)
        self._reference_event_cache = dict(reference_flags=flags.numpy(), reference_points_w=surface.numpy(),
            reference_normals_w=normal.numpy(), reference_valid=valid.numpy(), reference_fps=self.reference_fps)
        target_device = self.env.device
        target_dtype = self.env.robot.data.body_pos_w.dtype
        self.reference = {key: value.to(device=target_device, dtype=target_dtype) for key, value in
            dict(body_quat=quat[:, canonical_cpu], offset=offset, surface=surface, normal=normal,
                 point=point, tangent_speed=tangent_velocity.norm(dim=-1)).items()}
        self.reference["valid"] = valid.to(target_device)
        self.reference["flags"] = flags.to(target_device)

    def reference_event_inputs(self):
        """Complete reference arrays in scene-rebased coordinates, copied."""
        return {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._reference_event_cache.items()}

    def summary(self, arrays, prefix="contact_tracking_"):
        return summary(arrays, prefix=prefix)

    def describe(self):
        return dict(contract=deepcopy(CONTACT_TRACKING_CONTRACT), surface_spec=self.surface_spec,
            reference_endpoint_potential=self.reference_endpoint_potential,
            reference_frames=len(self.reference["offset"]), reference_fps=self.reference_fps)

    def update(self, measured=None):
        """Return fixed-shape finite tensors and positive unintegrated terms.

        ``reward_terms`` contains surface_orientation/contact_normal_distance
        bonuses, hold_support_stability bonus only during endpoint hold, and
        stance_tangent_slip/unexpected_noncanonical_contact costs.
        Caller assigns signs/weights and integrates control_dt exactly once.
        Geometric candidates and physical measured centroids remain separate.
        """
        env, data, c = self.env, self.env.robot.data, CONTACT_TRACKING_CONTRACT
        measured = env.contact_runtime.sample() if measured is None else measured
        phase = env.phase
        origin = env.env_origin()+env.cell_offset
        flags_index = (phase.clamp(0, 1)*(len(self.reference["flags"])-1)).floor().long()
        reference_flags = self.reference["flags"][flags_index]
        lower = flags_index
        upper = (lower+1).clamp(max=len(self.reference["flags"])-1)
        surface_valid = self.reference["valid"][lower] & self.reference["valid"][upper]
        # Interpolation must not invent a diagonal plane between a box wall
        # and its top. Adjacent coplanar mesh triangles are fine; a corner
        # transition's fractional sample has no unique surface target.
        previous_normal, next_normal = self.reference["normal"][lower], self.reference["normal"][upper]
        same_plane = (previous_normal*next_normal).sum(-1) >= 1.-1e-5
        same_plane &= (((self.reference["surface"][upper]-self.reference["surface"][lower])*previous_normal).sum(-1)).abs() <= 1e-5
        at_frame = ((phase.clamp(0, 1)*(len(self.reference["flags"])-1))-lower).abs() <= 1e-5
        surface_valid &= same_plane | at_frame[:, None]
        ref_quat = _interpolate(self.reference["body_quat"], phase, quaternion=True)
        offset = _interpolate(self.reference["offset"], phase)
        surface = _interpolate(self.reference["surface"], phase)
        normal = _interpolate(self.reference["normal"], phase)
        normal /= normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ref_point = _interpolate(self.reference["point"], phase)
        ref_speed = _interpolate(self.reference["tangent_speed"], phase)
        ref_speed = torch.where((phase >= 1.)[:, None], torch.zeros_like(ref_speed), ref_speed)
        actual_quat = data.body_quat_w[:, self.canonical]
        relative = quat_rotate(actual_quat, offset)
        actual_point = data.body_pos_w[:, self.canonical]+relative
        actual_velocity = rigid_point_velocity(data.body_com_lin_vel_w[:, self.canonical],
            data.body_com_ang_vel_w[:, self.canonical], actual_point-data.body_com_pos_w[:, self.canonical])
        actual_rebased = actual_point-origin[:, None]
        normal_error = ((actual_rebased-ref_point)*normal).sum(-1)
        orientation_error = _orientation_error(actual_quat, ref_quat)
        actual_valid = measured["contact_valid"][:, self.canonical] & measured["health_gate"][:, self.canonical]
        health = measured["all_body_health_gate"]
        surface_valid &= torch.isfinite(actual_point).all(-1) & torch.isfinite(actual_velocity).all(-1)
        surface_valid &= torch.isfinite(actual_quat).all(-1) & (actual_quat.norm(dim=-1)-1).abs().le(1e-3)
        stance = reference_flags & surface_valid & (ref_speed <= c["reference_stance_tangent_speed_max_m_s"])
        load = measured["normal_load_n"][:, self.canonical]
        slip_valid = stance & actual_valid & (load > c["actual_contact_load_min_n"])
        measured_slip = measured["load_weighted_tangent_speed_m_s"][:, self.canonical]
        rewards = dict(
            surface_orientation=_masked_mean(torch.exp(-orientation_error.square()/c["orientation_bandwidth_rad"]**2), surface_valid),
            contact_normal_distance=_masked_mean(torch.exp(-normal_error.square()/c["normal_distance_bandwidth_m"]**2), reference_flags & surface_valid),
            stance_tangent_slip=_masked_mean((measured_slip/c["slip_bandwidth_m_s"]).square().clamp(max=9.), slip_valid),
            unexpected_noncanonical_contact=((measured["normal_load_n"][:, self.noncanonical]/c["unexpected_force_scale_n"]).clamp(0, 1)).mean(-1),
        )
        rewards = {key: torch.where(health & env.is_ref_env, value, torch.zeros_like(value)) for key, value in rewards.items()}
        com = (data.body_com_pos_w*self.masses[..., None]).sum(1)/self.masses.sum(-1, keepdim=True)
        com_velocity = (data.body_com_lin_vel_w*self.masses[..., None]).sum(1)/self.masses.sum(-1, keepdim=True)
        com_valid = torch.isfinite(com).all(-1) & torch.isfinite(com_velocity).all(-1)
        nonfoot_load = measured["normal_load_n"][:, self.nonfoot].amax(-1)
        nonfoot_gate = health & (nonfoot_load <= c["hold_nonfoot_normal_load_max_n"])
        com_speed = com_velocity[:, :2].norm(dim=-1)
        foot_error = orientation_error[:, [6, 7]]
        foot_gate = surface_valid[:, [6, 7]].all(-1) & (foot_error <= c["hold_foot_orientation_error_max_rad"]).all(-1)
        values = dict(body_health_gate=measured["health_gate"], measurement_health_gate=health,
            body_normal_load_n=measured["normal_load_n"], body_total_force_w_n=measured["total_force_w"],
            body_torque_about_com_w_nm=measured["torque_about_com_w"],
            reference_contact_flags=reference_flags, reference_stance_mask=stance, reference_surface_valid=surface_valid,
            reference_surface_point_rebased_m=surface, reference_surface_normal_w=normal,
            reference_collision_point_rebased_m=ref_point, reference_point_tangent_speed_m_s=ref_speed,
            actual_collision_candidate_rebased_m=actual_rebased, candidate_normal_distance_error_m=normal_error,
            candidate_point_velocity_w_m_s=actual_velocity, orientation_error_rad=orientation_error,
            actual_measured_point_rebased_m=measured["normal_force_weighted_point_w"][:, self.canonical]-origin[:, None],
            actual_measured_normal_w=measured["load_weighted_normal_w"][:, self.canonical],
            actual_measured_point_valid=actual_valid, actual_normal_load_n=load,
            actual_tangent_speed_m_s=measured_slip, stance_slip_valid=slip_valid,
            com_rebased_m=com-origin, com_velocity_w_m_s=com_velocity, com_valid=com_valid,
            com_xy_speed_m_s=com_speed, nonfoot_normal_load_max_n=nonfoot_load,
            nonfoot_force_gate=nonfoot_gate, com_speed_gate=com_valid & (com_speed <= c["hold_com_xy_speed_max_m_s"]),
            foot_orientation_error_rad=foot_error, foot_orientation_gate=foot_gate,
        )
        count = env.num_envs
        scalar = lambda: torch.zeros(count, dtype=com.dtype, device=com.device)
        values.update(support_hull_margin_m=scalar(), support_hull_valid=torch.zeros_like(health),
            support_hull_rank=torch.full_like(health, -1, dtype=torch.long),
            support_point_count=torch.zeros_like(health, dtype=torch.long), com_hull_gate=torch.zeros_like(health),
            cop_moment_derived_rebased_m=torch.zeros_like(com), cop_valid=torch.zeros_like(health))
        plane_origin = origin.clone()
        plane_origin[:, 2] += self.top_z
        total_force = measured["total_force_w"].sum(1)
        moment = (measured["torque_about_com_w"]+torch.linalg.cross(
            data.body_com_pos_w-plane_origin[:, None], measured["total_force_w"])).sum(1)
        values["total_contact_force_w_n"] = total_force
        values["contact_moment_about_top_origin_w_nm"] = moment
        hold_ids = torch.nonzero(phase >= 1., as_tuple=False).flatten()
        if len(hold_ids):
            points = measured["normal_contact_points_w"][hold_ids][:, self.feet]
            point_normals = measured["normal_contact_normals_w"][hold_ids][:, self.feet]
            forces = measured["normal_contact_forces_n"][hold_ids][:, self.feet]
            valid = measured["normal_contact_valid"][hold_ids][:, self.feet].clone()
            valid &= (forces > c["hold_point_normal_force_min_n"])
            valid &= point_normals[..., 2] > c["hold_point_upward_normal_min_cosine"]
            valid &= (points[..., 2]-plane_origin[hold_ids, None, None, 2]).abs() <= c["hold_point_plane_tolerance_m"]
            valid &= measured["health_gate"][hold_ids][:, self.feet, None]
            flattened = points.reshape(len(hold_ids), -1, 3)
            mask = valid.reshape(len(hold_ids), -1)
            hull = support_hull_margin(com[hold_ids, :2], flattened[..., :2], mask)
            values["support_hull_margin_m"][hold_ids] = torch.nan_to_num(hull["margin_m"])
            values["support_hull_valid"][hold_ids] = hull["valid"] & health[hold_ids]
            values["support_hull_rank"][hold_ids] = hull["support_rank"]
            values["support_point_count"][hold_ids] = hull["active_point_count"]
            values["com_hull_gate"][hold_ids] = hull["valid"] & health[hold_ids] & com_valid[hold_ids] & (hull["margin_m"] >= c["hold_com_hull_margin_min_m"])
            # COP is only meaningful when all appreciably loaded normal points
            # share the chosen plane. Friction moments are already aggregated
            # from their independent patch application points by the runtime.
            all_points = measured["normal_contact_points_w"][hold_ids]
            all_normals = measured["normal_contact_normals_w"][hold_ids]
            all_loads = measured["normal_contact_forces_n"][hold_ids]
            all_valid = measured["normal_contact_valid"][hold_ids]
            loaded = all_valid & (all_loads > c["hold_point_normal_force_min_n"])
            on_plane = ((all_points[..., 2]-plane_origin[hold_ids, None, None, 2]).abs() <= c["hold_point_plane_tolerance_m"])
            on_plane &= all_normals[..., 2] > c["hold_point_upward_normal_min_cosine"]
            cop_valid = health[hold_ids] & ((~loaded) | on_plane).all(dim=(1, 2)) & (total_force[hold_ids, 2] > 20.)
            cop = plane_origin[hold_ids].clone()
            fz = total_force[hold_ids, 2].clamp_min(1e-9)
            cop[:, 0] += -moment[hold_ids, 1]/fz
            cop[:, 1] += moment[hold_ids, 0]/fz
            values["cop_moment_derived_rebased_m"][hold_ids] = cop-origin[hold_ids]
            values["cop_valid"][hold_ids] = cop_valid & torch.isfinite(cop).all(-1)
        values["hold_gate"] = (health & nonfoot_gate & values["com_hull_gate"]
            & values["com_speed_gate"] & foot_gate)
        stability_valid = ((phase >= 1.) & health & com_valid & values["support_hull_valid"]
            & (values["support_hull_rank"] == 2) & env.is_ref_env)
        stability_bonus = torch.sigmoid(values["support_hull_margin_m"]/c["hold_stability_margin_bandwidth_m"])
        stability_bonus *= torch.exp(-(com_speed/c["hold_stability_speed_bandwidth_m_s"]).square())
        rewards["hold_support_stability"] = torch.where(stability_valid, stability_bonus,
            torch.zeros_like(stability_bonus))
        # Export finite tensors while retaining validity masks for unavailable
        # measured points and degenerate hulls. No NaN becomes a valid contact.
        for key, shape in self.fields.items():
            if values[key].shape != (env.num_envs, *shape):
                raise RuntimeError(f"ContactTracking field {key} violates shape {(env.num_envs, *shape)}")
            if values[key].is_floating_point():
                values[key] = torch.nan_to_num(values[key], nan=0., posinf=0., neginf=0.)
        values["reward_terms"] = {key: torch.nan_to_num(value, nan=0., posinf=0., neginf=0.) for key, value in rewards.items()}
        self.last = values
        return values


def summary(arrays, prefix="contact_tracking_"):
    """Compact diagnosis from one recorded rollout; no success redefinition."""
    get = lambda name: np.asarray(arrays[prefix+name])
    phases = np.asarray(arrays["phase"])
    if "box_hold_controls" in arrays:
        controls = np.asarray(arrays["box_hold_controls"])
        selected = np.flatnonzero(controls > 0)
        hold = controls > 0
        hold_complete = bool(len(selected) == 50 and
            np.array_equal(controls[selected], np.arange(1, 51)) and
            np.array_equal(selected, np.arange(len(controls)-50, len(controls))) and
            np.all(phases[selected] >= 1.))
    else:
        # Phase-one endpoints alone cannot establish fifty EXTRA controls.
        hold = phases >= 1.
        hold_complete = False
    valid_slip = get("stance_slip_valid").astype(bool)
    slip = get("actual_tangent_speed_m_s")
    valid_normal = get("reference_surface_valid").astype(bool) & get("reference_contact_flags").astype(bool)
    normal_error = get("candidate_normal_distance_error_m")
    masked_mean = lambda value, valid: float(value[valid].mean()) if valid.any() else None
    return dict(measurement_healthy_all=bool(get("measurement_health_gate").all()),
        reference_stance_measured_slip_mean_m_s=masked_mean(slip, valid_slip),
        labeled_surface_normal_error_rmse_m=(float(np.sqrt(np.mean(normal_error[valid_normal]**2))) if valid_normal.any() else None),
        hold_complete=hold_complete,
        hold_passed=bool(hold_complete and get("hold_gate")[hold].all()),
        hold_observed_endpoints=int(hold.sum()),
        hold_contact_passing_endpoints=int(get("hold_gate")[hold].sum()),
        hold_support_hull_valid_endpoints=int(get("support_hull_valid")[hold].sum()),
        hold_support_hull_min_margin_m=(float(get("support_hull_margin_m")[hold & get("support_hull_valid").astype(bool)].min()) if (hold & get("support_hull_valid").astype(bool)).any() else None),
        physical_contact_point_semantics=CONTACT_TRACKING_CONTRACT["weighted_point"],
        com_hull_scope=CONTACT_TRACKING_CONTRACT["stability"],
        final_values={key: get(key)[-1].tolist() for key in ("hold_gate", "measurement_health_gate",
            "nonfoot_normal_load_max_n", "com_xy_speed_m_s", "support_hull_margin_m", "support_hull_valid",
            "foot_orientation_error_rad", "cop_valid")})
