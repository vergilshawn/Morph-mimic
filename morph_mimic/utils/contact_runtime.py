"""Measured terrain contacts for every robot rigid body.

The PhysX tensor API returns normal contact points and friction *patch* points
in separate buffers.  They are never paired by index here.  ``dt`` is the
physics step, so the API converts impulses to newtons exactly once.  Positions
and normals are world quantities; torques are measured-force lever moments
about each body's actual COM. The API does not expose an independent free
torsional/rolling contact couple, so that quantity is not claimed here.

No FK estimate is substituted for an unavailable contact measurement.  Missing
APIs, missing rigid bodies, invalid buffers, and capacity saturation fail the
measurement gate.  The capacity check is conservative: PhysX exposes no exact
"overflow occurred" counter, so a completely full buffer is rejected as well.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


SCHEMA = "physx_all_body_terrain_contact_v1"
DEFAULT_FILTER = "/World/terrain/terrain/mesh"


@dataclass(frozen=True)
class ContactRuntimeConfig:
    contacts_per_body: int = 16
    filter_prim_paths: tuple[str, ...] = (DEFAULT_FILTER,)
    force_tolerance_n: float = .05
    force_relative_tolerance: float = .002
    normal_tolerance: float = .02

    def __post_init__(self):
        if self.contacts_per_body < 2:
            raise ValueError("A positive contact buffer with headroom is required")
        if len(self.filter_prim_paths) != 1:
            raise ValueError("One explicitly selected terrain filter is required")
        if not all(math.isfinite(x) and x >= 0 for x in (
                self.force_tolerance_n, self.force_relative_tolerance, self.normal_tolerance)):
            raise ValueError("Contact tolerances must be finite and nonnegative")


def attach_contact_sensors(env: Any, config: ContactRuntimeConfig | None = None):
    """Call after ``super()._setup_scene()`` in an independent environment.

    One reporting body per view follows IsaacLab's filtered-contact limitation.
    Existing ACP nine-body sensors are retained.  Detailed buffers are fetched
    by :class:`ContactRuntime`, avoiding duplicate ContactSensor aggregation.
    """
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    import omni.usd
    from pxr import Usd, UsdPhysics

    config = config or ContactRuntimeConfig()
    if getattr(env, "full_scene_contact_sensors", None):
        raise RuntimeError("Full-body contact sensors are already attached")
    stage = omni.usd.get_context().get_stage()
    source_path = env.robot_body_root_pattern.replace("env_.*", "env_0")
    parent = stage.GetPrimAtPath(source_path)
    if not parent.IsValid():
        raise RuntimeError(f"Missing robot body namespace: {source_path}")
    body_names = sorted(prim.GetName() for prim in Usd.PrimRange(parent)
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI))
    if not body_names or len(body_names) != len(set(body_names)):
        raise RuntimeError("Cannot enumerate unique robot rigid bodies")
    env.full_scene_contact_sensors = []
    for index, name in enumerate(body_names):
        sensor = ContactSensor(ContactSensorCfg(
            prim_path=f"{env.robot_body_root_pattern}/{name}",
            update_period=0., history_length=0,
            filter_prim_paths_expr=list(config.filter_prim_paths),
            max_contact_data_count_per_prim=config.contacts_per_body,
            # Raw data retains normals and each point, unlike averaged fields.
            track_contact_points=False, track_friction_forces=False,
        ))
        env.full_scene_contact_sensors.append((name, sensor))
        env.scene.sensors[f"full_scene_contact_{index}"] = sensor
    env.full_scene_contact_config = config
    return tuple(body_names)


def contact_environment_order(articulation_paths, sensor_paths, body_name):
    """Map authoritative articulation rows to raw contact rows by USD paths.

    An MJCF articulation root can be the bodies' namespace or the root body
    inside that namespace.  All bodies in this project share one body parent.
    We therefore match exact ancestor relationships, including that shared
    body parent, rather than sorting environment numbers or trusting basenames.
    """
    roots = [str(path).rstrip('/') for path in articulation_paths]
    sensors = [str(path).rstrip('/') for path in sensor_paths]
    if (not roots or len(roots) != len(sensors) or len(set(roots)) != len(roots)
            or len(set(sensors)) != len(sensors)):
        raise RuntimeError("Contact/articulation paths must have equal nonempty unique coverage")
    raw_to_articulation = []
    for path in sensors:
        if not path.startswith('/') or path.rsplit('/', 1)[-1] != body_name:
            raise RuntimeError(f"Contact path does not name the requested body {body_name}: {path}")
        parent = path.rsplit('/', 1)[0]
        matches = [index for index, root in enumerate(roots)
            if path == root or path.startswith(root+'/') or root == parent or root.startswith(parent+'/')]
        if len(matches) != 1:
            raise RuntimeError(f"Contact body {path} has {len(matches)} authoritative articulation matches")
        raw_to_articulation.append(matches[0])
    if set(raw_to_articulation) != set(range(len(roots))):
        raise RuntimeError("Raw contact paths duplicate or omit an articulation instance")
    return [raw_to_articulation.index(index) for index in range(len(roots))]


def _buffer_owners(counts, starts, capacity: int):
    """Decode arbitrary sparse pair ranges without dynamic-size GPU gathers.

    Counts/start indices originate in PhysX uint32 buffers.  Empty groups have
    no meaningful start index.  Sorting nonempty ranges lets us map the fixed
    capacity buffer to its sensor group without per-view GPU-to-CPU sync.
    """
    counts = counts.reshape(-1).long()
    starts = starts.reshape(-1).long()
    active = counts > 0
    safe_starts = torch.where(active, starts, torch.full_like(starts, capacity))
    order = safe_starts.argsort(stable=True)
    sorted_starts = safe_starts[order]
    slots = torch.arange(capacity, device=starts.device)
    sorted_owner = torch.searchsorted(sorted_starts, slots, right=True) - 1
    owner = order[sorted_owner.clamp(min=0)]
    valid = (sorted_owner >= 0) & active[owner] & (slots >= starts[owner]) & (slots < starts[owner] + counts[owner])
    bad_group = (counts < 0) | (active & ((starts < 0) | (starts + counts > capacity)))
    # A full buffer may have silently truncated PhysX data; do not accept it.
    saturated = (counts.sum() >= capacity).expand_as(bad_group)
    # Overlapping ranges cannot be assigned to distinct sensor/filter pairs.
    ends = sorted_starts + counts[order]
    overlap = torch.zeros_like(active)
    if len(counts) > 1:
        overlap[order[1:]] = (sorted_starts[1:] < ends.cummax(0).values[:-1]) & active[order[1:]] & active[order[:-1]]
    # Spread zero-valued unused slots across groups instead of making every
    # unused buffer entry contend on the final sensor's atomic accumulator.
    owner = torch.where(valid, owner, slots.remainder(len(counts)))
    return owner, valid, bad_group | saturated | overlap


def _scatter(values, owner, groups: int):
    result = values.new_zeros((groups,) + values.shape[1:])
    return result.index_add_(0, owner, values)


def aggregate_contact_buffers(normal_data, friction_data, *, com_pos_w, com_lin_vel_w,
                              angular_vel_w, force_matrix_w, config=None):
    """Pure Torch aggregation for one reporting body per environment.

    ``normal_data`` is a six-item get_contact_data(dt) result whose count and
    start buffers have already been cloned before get_friction_data(dt).
    ``friction_data`` has four items.  Both APIs' forces are in N.  Every returned
    aggregate has leading [environment] dimension.  Missing contact points have
    NaN position/normal/velocity; absence is represented by explicit validity.
    """
    config = config or ContactRuntimeConfig()
    magnitude, points, normals, separation, counts, starts = normal_data
    friction, friction_points, friction_counts, friction_starts = friction_data
    groups = len(com_pos_w)
    if (counts.shape != (groups, 1) or friction_counts.shape != (groups, 1)
            or starts.shape != counts.shape or friction_starts.shape != counts.shape):
        raise RuntimeError("Full-body raw contact reporting requires one body and one terrain filter per env")
    if (magnitude.shape != (len(points), 1) or normals.shape != points.shape
            or separation.shape != magnitude.shape or points.shape[1:] != (3,)
            or friction.shape != friction_points.shape or friction.shape[1:] != (3,)):
        raise RuntimeError("Unexpected PhysX contact buffer shapes")
    for value in (com_pos_w, com_lin_vel_w, angular_vel_w, force_matrix_w):
        if value.shape != (groups, 3):
            raise RuntimeError("COM state and normal-force matrix must have shape [env,3]")

    owner, valid, bad_range = _buffer_owners(counts, starts, len(points))
    fowner, fvalid, fbad_range = _buffer_owners(friction_counts, friction_starts, len(friction_points))
    # Unused capacity is unspecified and may contain NaNs/old data.
    normal_finite = (torch.isfinite(magnitude).all(-1) & torch.isfinite(points).all(-1)
                     & torch.isfinite(normals).all(-1) & torch.isfinite(separation).all(-1))
    friction_finite = torch.isfinite(friction).all(-1) & torch.isfinite(friction_points).all(-1)
    normal_norm = normals.norm(dim=-1)
    point_bad = valid & (~normal_finite | ((normal_norm - 1.).abs() > config.normal_tolerance)
                         | (magnitude[:, 0] < -1.e-5))
    friction_bad = fvalid & ~friction_finite
    valid_numeric = valid & ~point_bad
    fvalid_numeric = fvalid & ~friction_bad
    weights = torch.where(valid_numeric, magnitude[:, 0].clamp(min=0.), 0.)
    safe_points = torch.where(valid_numeric[:, None], points, com_pos_w[owner])
    safe_normals = torch.where(valid_numeric[:, None], normals, 0.)
    normal_force = weights[:, None] * safe_normals
    velocity = com_lin_vel_w[owner] + torch.linalg.cross(angular_vel_w[owner], safe_points - com_pos_w[owner])
    tangent_velocity = velocity - (velocity * safe_normals).sum(-1, keepdim=True) * safe_normals
    safe_friction = torch.where(fvalid_numeric[:, None], friction, 0.)
    safe_friction_points = torch.where(fvalid_numeric[:, None], friction_points, com_pos_w[fowner])
    friction_velocity = (com_lin_vel_w[fowner]
                         + torch.linalg.cross(angular_vel_w[fowner], safe_friction_points - com_pos_w[fowner]))
    total_normal = _scatter(normal_force, owner, groups)
    total_friction = _scatter(safe_friction, fowner, groups)
    normal_load = _scatter(weights[:, None], owner, groups)[:, 0]
    weighted_position = _scatter(weights[:, None] * safe_points, owner, groups) / normal_load.clamp(min=1.e-12)[:, None]
    mean_normal = total_normal / total_normal.norm(dim=-1, keepdim=True).clamp(min=1.e-12)
    weighted_velocity = _scatter(weights[:, None] * velocity, owner, groups) / normal_load.clamp(min=1.e-12)[:, None]
    tangent_speed = _scatter((weights * tangent_velocity.norm(dim=-1))[:, None], owner, groups)[:, 0] / normal_load.clamp(min=1.e-12)
    normal_torque = _scatter(torch.linalg.cross(safe_points - com_pos_w[owner], normal_force), owner, groups)
    friction_torque = _scatter(torch.linalg.cross(safe_friction_points - com_pos_w[fowner], safe_friction), fowner, groups)
    friction_power = _scatter((safe_friction * friction_velocity).sum(-1, keepdim=True), fowner, groups)[:, 0]
    normal_bad_env = _scatter(point_bad[:, None].long(), owner, groups)[:, 0] > 0
    friction_bad_env = _scatter(friction_bad[:, None].long(), fowner, groups)[:, 0] > 0
    force_error = (total_normal - force_matrix_w).norm(dim=-1)
    force_tolerance = config.force_tolerance_n + config.force_relative_tolerance * force_matrix_w.norm(dim=-1)
    state_finite = (torch.isfinite(com_pos_w).all(-1) & torch.isfinite(com_lin_vel_w).all(-1)
                    & torch.isfinite(angular_vel_w).all(-1) & torch.isfinite(force_matrix_w).all(-1))
    # This is both the pool budget and the fixed exported point count.  A
    # single pair exceeding the export width is rejected rather than silently
    # truncating the support polygon used by downstream acceptance checks.
    point_width = config.contacts_per_body
    point_offsets = torch.arange(point_width, device=starts.device)
    point_indices = starts[:, 0].long()[:, None] + point_offsets[None]
    padded_valid = (point_offsets[None] < counts[:, :1].long()) & (point_indices >= 0) & (point_indices < len(points))
    point_indices = point_indices.clamp(min=0, max=len(points)-1)
    padded_valid &= valid_numeric[point_indices]
    padded_points = points[point_indices]
    padded_normals = normals[point_indices]
    padded_magnitude = magnitude[point_indices, 0]
    padded_velocity = (com_lin_vel_w[:, None] + torch.linalg.cross(
        angular_vel_w[:, None].expand_as(padded_points), padded_points-com_pos_w[:, None]))
    point_overflow = counts[:, 0] > point_width
    buffer_bad = bad_range | fbad_range | normal_bad_env | friction_bad_env | point_overflow
    health = state_finite & ~buffer_bad & (force_error <= force_tolerance)
    touching = normal_load > 1.e-6
    missing = torch.full_like(weighted_position, float("nan"))
    return {
        "health_gate": health, "buffer_invalid": buffer_bad,
        "force_reconstruction_error_n": force_error,
        "normal_force_w": total_normal, "friction_force_w": total_friction,
        "total_force_w": total_normal + total_friction,
        "torque_about_com_w": normal_torque + friction_torque,
        "normal_load_n": normal_load, "contact_valid": touching,
        "normal_force_weighted_point_w": torch.where(touching[:, None], weighted_position, missing),
        "load_weighted_normal_w": torch.where(touching[:, None], mean_normal, missing),
        "load_weighted_point_velocity_w": torch.where(touching[:, None], weighted_velocity, missing),
        "load_weighted_tangent_speed_m_s": torch.where(touching, tangent_speed, torch.zeros_like(tangent_speed)),
        "friction_power_w": friction_power,
        "normal_point_count": counts[:, 0].long(),
        "normal_contact_points_w": torch.where(padded_valid[..., None], padded_points, float("nan")),
        "normal_contact_normals_w": torch.where(padded_valid[..., None], padded_normals, float("nan")),
        "normal_contact_forces_n": torch.where(padded_valid, padded_magnitude, 0.),
        "normal_contact_valid": padded_valid & ~point_overflow[:, None],
        "normal_contact_velocities_w": torch.where(padded_valid[..., None], padded_velocity, float("nan")),
        "friction_patch_count": friction_counts[:, 0].long(),
    }


class ContactRuntime:
    """Capability-checked all-body, terrain-only measured contact snapshots."""
    def __init__(self, env: Any):
        self.env = env
        self.config = getattr(env, "full_scene_contact_config", ContactRuntimeConfig())
        self.sensors = tuple(getattr(env, "full_scene_contact_sensors", ()))
        body_names = tuple(env.robot.body_names)
        self.body_names = body_names
        if len(self.sensors) != len(body_names) or {n for n, _ in self.sensors} != set(body_names):
            raise RuntimeError("Full-body contact coverage does not match every articulation body")
        sensor_by_name = dict(self.sensors)
        self.sensors = tuple((name, sensor_by_name[name]) for name in body_names)
        self.physics_dt = float(env.physics_dt if hasattr(env, "physics_dt") else env.cfg.sim.dt)
        if not math.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise RuntimeError("A finite positive physics dt is required for impulse-to-force conversion")
        self.capacity_per_view = []
        self.environment_orders = []
        self.ordering_audit = []
        root_view = getattr(env.robot, "root_physx_view", None)
        articulation_paths = getattr(root_view, "prim_paths", None)
        if articulation_paths is None or len(articulation_paths) != env.num_envs:
            raise RuntimeError("Authoritative articulation prim_paths unavailable for contact ordering")
        self.articulation_paths = list(map(str, articulation_paths))
        order_device = env.robot.data.body_com_pos_w.device
        for name, sensor in self.sensors:
            view = sensor.contact_physx_view
            required = ("get_contact_data", "get_friction_data", "get_contact_force_matrix")
            if any(not callable(getattr(view, key, None)) for key in required):
                raise RuntimeError(f"Detailed contact API unavailable for {name}")
            if (view.sensor_count != env.num_envs or view.filter_count != 1
                    or tuple(sensor.body_names) != (name,)):
                raise RuntimeError(f"Invalid full-body contact view mapping for {name}")
            if view.max_contact_data_count < env.num_envs * self.config.contacts_per_body:
                raise RuntimeError(f"Insufficient detailed contact buffer for {name}")
            # sensor_names contains basenames, not paths, in PhysX 107.3.
            sensor_paths = getattr(view, "sensor_paths", None)
            if sensor_paths is None:
                raise RuntimeError(f"Raw contact sensor_paths unavailable for {name}")
            order = contact_environment_order(self.articulation_paths, sensor_paths, name)
            self.environment_orders.append(torch.tensor(order, dtype=torch.long, device=order_device))
            self.ordering_audit.append(dict(body_name=name, identity_order=order==list(range(env.num_envs)),
                articulation_paths_head=self.articulation_paths[:12], sensor_paths_head=list(sensor_paths)[:12],
                articulation_to_raw_rows_head=order[:12]))
            self.capacity_per_view.append(int(view.max_contact_data_count))
        for key in ("body_com_pos_w", "body_com_lin_vel_w", "body_com_ang_vel_w"):
            if not hasattr(env.robot.data, key):
                raise RuntimeError(f"Actual articulation COM state unavailable: {key}")

    def describe(self):
        return {"schema": SCHEMA, "body_names": list(self.body_names),
                "num_bodies": len(self.body_names), "num_envs": self.env.num_envs,
                "physics_dt_s": self.physics_dt, "filter_prim_paths": list(self.config.filter_prim_paths),
                "capacity_per_view": self.capacity_per_view,
                "environment_ordering": "Exact sensor_paths to root_physx_view.prim_paths one-to-one match; raw rows reordered to articulation state rows",
                "ordering_audit": self.ordering_audit,
                "legacy_contact_sensor_ordering_audit": self._legacy_ordering_audit(),
                "normal_source": "PhysX get_contact_data(physics_dt), per-point force*normal",
                "friction_source": "PhysX get_friction_data(physics_dt), separate patch point/force",
                "velocity_origin": "actual inertial COM, v_point=v_com+omega cross (point-com)",
                "torque_origin": "actual inertial COM",
                "torque_semantics": "sum of measured force lever moments; independent torsional/rolling free couple unavailable", "overflow_gate": "reject full, overlapping, out-of-range buffers and any pair wider than exported points",
                "exported_points_per_body": self.config.contacts_per_body,
                "scope": "all robot rigid bodies against selected terrain; robot self contacts excluded",
                "point_semantics": "normal-force-weighted measured point, not a general center of pressure"}

    def environment_order_for_sensor(self, sensor):
        """Return raw rows in articulation order for any single-body sensor.

        This also allows a new v3 environment to reorder its inherited nine
        sensors without editing the frozen ACP environment or reward formulas.
        """
        names = tuple(sensor.body_names)
        paths = getattr(sensor.contact_physx_view, "sensor_paths", None)
        if len(names) != 1 or paths is None:
            raise RuntimeError("Environment order requires a single-body contact view with sensor_paths")
        order = contact_environment_order(self.articulation_paths, paths, names[0])
        return torch.tensor(order, dtype=torch.long, device=self.environment_orders[0].device)

    def _legacy_ordering_audit(self):
        """Read-only audit; never change frozen ACP sensor aggregation."""
        audits = []
        for sensor in getattr(self.env, "scene_contact_sensors", ()):
            names = tuple(sensor.body_names)
            if len(names) != 1:
                audits.append(dict(body_names=list(names), verified=False, reason="not a single-body filtered view"))
                continue
            paths = getattr(sensor.contact_physx_view, "sensor_paths", None)
            try:
                order = contact_environment_order(self.articulation_paths, () if paths is None else paths, names[0])
                audits.append(dict(body_name=names[0], verified=True,
                    identity_order=order==list(range(self.env.num_envs)),
                    articulation_to_raw_rows_head=order[:12],
                    articulation_to_raw_rows=order, sensor_paths_head=list(paths)[:12]))
            except RuntimeError as error:
                audits.append(dict(body_name=names[0], verified=False, reason=str(error)))
        return audits

    def sample(self):
        data = self.env.robot.data
        normals, frictions, matrices, pool_health = [], [], [], []
        normal_offset = friction_offset = 0
        for sensor_index, (_, sensor) in enumerate(self.sensors):
            view = sensor.contact_physx_view
            normal = list(view.get_contact_data(dt=self.physics_dt))
            if len(normal) != 6:
                raise RuntimeError("PhysX normal contact API must return six buffers")
            # PhysX reuses count/start buffers for friction; snapshot these first.
            order = self.environment_orders[sensor_index]
            normal[4], normal[5] = normal[4].long().index_select(0, order), normal[5].long().index_select(0, order)
            friction = list(view.get_friction_data(dt=self.physics_dt))
            if len(friction) != 4:
                raise RuntimeError("PhysX friction API must return four buffers")
            friction[2], friction[3] = friction[2].long().index_select(0, order), friction[3].long().index_select(0, order)
            # Preserve separate per-view capacity gates when batching all views.
            ncounts, nstarts = normal[4].long(), normal[5].long()
            fcounts, fstarts = friction[2].long(), friction[3].long()
            normal_range_bad = (ncounts < 0) | ((ncounts > 0) & ((nstarts < 0) | (nstarts+ncounts > len(normal[1]))))
            friction_range_bad = (fcounts < 0) | ((fcounts > 0) & ((fstarts < 0) | (fstarts+fcounts > len(friction[1]))))
            pool_health.append((ncounts.sum() < len(normal[1]))
                               & (fcounts.sum() < len(friction[1]))
                               & ~normal_range_bad.any() & ~friction_range_bad.any())
            normal[5] = normal[5].long() + normal_offset
            friction[3] = friction[3].long() + friction_offset
            normal_offset += len(normal[1])
            friction_offset += len(friction[1])
            normals.append(normal)
            frictions.append(friction)
            matrices.append(view.get_contact_force_matrix(dt=self.physics_dt).reshape(self.env.num_envs, 3).index_select(0, order))
        # Aggregate once across bodies to avoid thousands of tiny GPU kernels.
        # The temporary flattened group order is [body,environment].
        flatten_state = lambda value: value.transpose(0, 1).reshape(-1, 3)
        normal = [torch.cat([values[index] for values in normals], dim=0) for index in range(6)]
        friction = [torch.cat([values[index] for values in frictions], dim=0) for index in range(4)]
        batched = aggregate_contact_buffers(normal, friction,
            com_pos_w=flatten_state(data.body_com_pos_w),
            com_lin_vel_w=flatten_state(data.body_com_lin_vel_w),
            angular_vel_w=flatten_state(data.body_com_ang_vel_w),
            force_matrix_w=torch.cat(matrices, dim=0), config=self.config)
        count, environments = len(self.sensors), self.env.num_envs
        result = {key: value.reshape(count, environments, *value.shape[1:]).transpose(0, 1)
                  for key, value in batched.items()}
        pool_health = torch.stack(pool_health)[None].expand(environments, -1)
        result["health_gate"] &= pool_health
        result["buffer_invalid"] |= ~pool_health
        result["all_body_health_gate"] = result["health_gate"].all(dim=1)
        return result
