"""Offline source-model endpoint optimization for an explicit settling suffix.

This changes a new endpoint target, never the original motion.  Constraints
establish geometric compatibility of a quasi-static candidate only. They do
not establish contact forces, actuator equilibrium, or physical hold success.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convex_margin(point_xy, points_xy):
    """Signed Euclidean margin to a rank-two convex hull (positive inside)."""
    from scipy.spatial import ConvexHull, QhullError
    points = np.unique(np.round(np.asarray(points_xy, dtype=float), 10), axis=0)
    if len(points) < 3 or np.linalg.matrix_rank(points-points.mean(0), tol=1e-9) < 2:
        return -1., False
    try:
        equations = ConvexHull(points).equations
    except QhullError:
        return -1., False
    margins = -(equations[:, :2] @ np.asarray(point_xy)+equations[:, 2])/np.linalg.norm(equations[:, :2], axis=-1)
    return float(margins.min()), True


def optimize_stable_hold(motion_path, *, output_path=None, max_iterations=300):
    """Find a minimally changed named-29 G1 endpoint using SLSQP and MuJoCo FK.

    Root translation is constrained below 10cm in Euclidean norm; original yaw
    is fixed and roll/pitch may each change at most .20rad. All 23 controlled
    joints stay within source limits and .6rad of the original final pose. Six
    held joints remain exactly at the robot spec's nominal values. Both foot
    origins preserve their original XY positions, with flat capsule soles on
    the verified box top. COM has at least 1cm convex-hull margin. Collision
    masks, excludes and parent filtering are retained from the source model.
    """
    import mujoco
    import scipy
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    from morph_mimic.utils.box_task import FOOT_BODY_NAMES, load_box_task_spec
    from morph_mimic.utils.contact_surface import load_contact_surface_spec
    from morph_mimic.utils.reference_scene import load_reference_scene
    from morph_mimic.utils.specs import RobotSpec, PROJECT_ROOT
    from morph_mimic.diagnostics.validate_scene_reference import build_model, _joint_indices

    motion_path = Path(motion_path).resolve()
    original_sha = _hash(motion_path)
    if output_path is not None and Path(output_path).resolve() == motion_path:
        raise ValueError("The optimization report cannot replace source motion")
    task = load_box_task_spec(motion_path)
    scene = load_reference_scene(motion_path.with_suffix('.scene.json'), motion_path)
    spec = RobotSpec.load('g1')
    xml = PROJECT_ROOT/spec.runtime_assets['mjcf']
    model, scene_ids = build_model(xml, scene, spec.physics_dt)
    disabled = int(model.opt.disableflags)
    if disabled & (int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)):
        raise ValueError("Static collision verification requires contact and constraints enabled")
    data = mujoco.MjData(model)
    qadr, _ = _joint_indices(model, spec)
    with np.load(motion_path, allow_pickle=False) as archive:
        names = list(archive['joint_names'])
        columns = [names.index(name) for name in spec.joint_names]
        q_original = archive['joint_pos'][-1, columns].astype(float)
        root_original = archive['root_pos'][-1].astype(float)
        quat_original = archive['root_rot_wxyz'][-1].astype(float)
    quat_original /= np.linalg.norm(quat_original)
    euler_original = Rotation.from_quat(quat_original[[1, 2, 3, 0]]).as_euler('xyz')
    controlled = np.array([spec.joint_names.index(n) for n in spec.controlled_joint_names])
    held = np.array([spec.joint_names.index(n) for n in spec.held_joint_names])
    if not (np.isfinite(q_original).all() and np.isfinite(root_original).all()
            and np.isfinite(quat_original).all()):
        raise ValueError("Source endpoint must be finite")
    if not np.allclose(q_original[held], np.asarray(spec.default_pose)[held], rtol=0, atol=1e-7):
        raise ValueError("Source endpoint must already respect nominal held joints")
    q_original[held] = np.asarray(spec.default_pose)[held]
    feet = np.array([model.body(name).id for name in FOOT_BODY_NAMES])
    body_ids = np.array([i for i in range(1, model.nbody)])
    body_names = [model.body(int(i)).name for i in body_ids]
    surface = load_contact_surface_spec(motion_path, body_names=body_names)
    masses = model.body_mass[body_ids]
    first_xy = np.asarray(task['source_first_root_xy'])
    endpoints = np.asarray(task['foot_geometry']['capsule_endpoints_local_m'])
    radii = np.asarray(task['foot_geometry']['capsule_radii_m'])
    site_local = np.asarray(task['foot_geometry']['support_center_site_local_m'])
    top = task['target_top']
    axes, center, half = [np.asarray(top[key]) for key in ('axes_xy', 'center_xy_m', 'half_extents_m')]
    top_z = float(top['top_z_m'])
    cached = [None, None]

    def unpack(x):
        root = root_original+x[:3]
        euler = euler_original+np.r_[x[3:5], 0.]
        quaternion = Rotation.from_euler('xyz', euler).as_quat()[[3, 0, 1, 2]]
        if quaternion@quat_original < 0:
            quaternion *= -1
        joints = q_original.copy()
        joints[controlled] += x[5:]
        return root, quaternion, joints

    def geometry(x):
        if cached[0] is not None and np.array_equal(x, cached[0]):
            return cached[1]
        root, quaternion, joints = unpack(x)
        data.qpos[:3], data.qpos[3:7], data.qpos[qadr] = root, quaternion, joints
        data.qvel[:] = 0
        mujoco.mj_fwdPosition(model, data)
        fp = data.xpos[feet].copy()
        rotations = data.xmat[feet].reshape(2, 3, 3).copy()
        world = fp[:, None, None]+np.einsum('bij,bckj->bcki', rotations, endpoints)
        points = world.copy()
        points[..., 2] -= radii[..., None]
        points[..., :2] -= first_xy
        com = (data.xipos[body_ids]*masses[:, None]).sum(0)/masses.sum()
        com[:2] -= first_xy
        flat_points = points.reshape(-1, 3)
        potential_margin, potential_valid = convex_margin(com[:2], flat_points[:, :2])
        local_points = (flat_points[:, :2]-center) @ axes.T
        point_top_margin = (half-np.abs(local_points)).min(-1)
        cap_low = points[..., 2] <= points[..., 2].min(-1, keepdims=True)+1e-6
        active = cap_low & (np.abs(points[..., 2]-top_z) <= .005) & (point_top_margin.reshape(2, -1, 2) >= 0.)
        support_margin, support_valid = convex_margin(com[:2], flat_points[active.reshape(-1), :2])
        sites = fp+np.einsum('bij,bj->bi', rotations, site_local)
        sites[:, :2] -= first_xy
        site_margin = (half-np.abs((sites[:, :2]-center) @ axes.T)).min(-1)
        root_up = data.xmat[model.body(spec.base_link).id].reshape(3, 3)[:, 2]
        tilt = float(np.arccos(np.clip(root_up[2], -1., 1.)))
        penetration = self_penetration = 0.
        collision_pairs = []
        for contact in data.contact:
            a, b = map(int, contact.geom)
            if int(model.geom_bodyid[a]) == 0 and int(model.geom_bodyid[b]) == 0:
                continue
            depth = max(0., -float(contact.dist))
            if a in scene_ids or b in scene_ids:
                penetration = max(penetration, depth)
            elif int(model.geom_bodyid[a]) and int(model.geom_bodyid[b]):
                self_penetration = max(self_penetration, depth)
            if depth > .0001:
                collision_pairs.append(dict(geom_names=[model.geom(a).name, model.geom(b).name], penetration_m=depth))
        value = dict(root=root, quaternion=quaternion, joints=joints, foot_pos=fp,
            foot_rotations=rotations, points=points, com=com,
            potential_margin=potential_margin, potential_valid=potential_valid,
            support_margin=support_margin, support_valid=support_valid,
            support_mask=active, site_margin=site_margin, point_top_margin=point_top_margin,
            tilt=tilt, scene_penetration=penetration, self_penetration=self_penetration,
            collision_pairs=collision_pairs)
        cached[:] = [x.copy(), value]
        return value

    x0 = np.zeros(5+len(controlled))
    before = geometry(x0)
    foot_xy_original = before['foot_pos'][:, :2].copy()
    def objective(x):
        # All terms dimensionless; target root/joint posture and feet are
        # preserved unless the explicit support constraints require a change.
        return float(np.sum((x[:3]/.10)**2)+np.sum((x[3:5]/.20)**2)
                     +np.sum((x[5:]/.30)**2)/3.)
    def equalities(x):
        g = geometry(x)
        foot_flatness = g['foot_rotations'][:, 2, :2].reshape(-1)
        foot_plane = g['points'][..., 2].mean(axis=(1, 2))-top_z
        foot_xy = (g['foot_pos'][:, :2]-foot_xy_original).reshape(-1)
        return np.r_[foot_flatness, foot_plane, foot_xy]
    def inequalities(x):
        g = geometry(x)
        return np.r_[.099-np.linalg.norm(x[:3]), .299-g['tilt'],
            g['site_margin']-.01, g['point_top_margin'], g['foot_rotations'][:, 2, 2]-.999,
            g['potential_margin']-.0101, .005-g['scene_penetration'], .005-g['self_penetration']]
    lower = np.maximum(np.asarray(spec.pos_limits_lower)[controlled]-q_original[controlled], -.60)
    upper = np.minimum(np.asarray(spec.pos_limits_upper)[controlled]-q_original[controlled], .60)
    bounds = [(-.099, .099)]*3+[(-.20, .20)]*2+list(zip(lower, upper))
    optimized = minimize(objective, x0, method='SLSQP', bounds=bounds,
        constraints=[dict(type='eq', fun=equalities), dict(type='ineq', fun=inequalities)],
        options=dict(maxiter=int(max_iterations), ftol=1.e-10, disp=False))
    after = geometry(optimized.x)
    eq_error = float(np.max(np.abs(equalities(optimized.x))))
    ineq_minimum = float(np.min(inequalities(optimized.x)))
    source_unchanged = original_sha == _hash(motion_path)
    accepted = bool(optimized.success and source_unchanged and eq_error <= 1.e-5
        and ineq_minimum >= -1.e-6 and after['support_valid'] and after['support_margin'] >= .01-1.e-6
        and np.all(after['support_mask'].reshape(2, -1).sum(-1) >= 2))
    def metrics(g):
        return dict(com_rebased_m=g['com'].tolist(), support_hull_margin_m=g['support_margin'],
            support_hull_valid=g['support_valid'], geometric_point_count_per_foot=g['support_mask'].reshape(2, -1).sum(-1).tolist(),
            potential_all_capsule_margin_m=g['potential_margin'], foot_site_top_margin_m=g['site_margin'].tolist(),
            foot_sole_plane_error_m=(g['points'][..., 2].min(axis=(1, 2))-top_z).tolist(),
            foot_max_capsule_plane_error_m=(g['points'][..., 2].max(axis=(1, 2))-top_z).tolist(),
            root_tilt_rad=g['tilt'], scene_max_penetration_m=g['scene_penetration'],
            self_max_penetration_m=g['self_penetration'], collision_pairs=g['collision_pairs'])
    result = dict(schema=1, robot='g1', accepted=accepted, algorithm_source_sha256=_hash(__file__), method='SLSQP_source_MuJoCo_FK_static_geometry_v1',
        scope='Geometric settling endpoint only; no force solution, actuator equilibrium or physical hold established',
        motion_path=str(motion_path), source_motion_sha256=original_sha, source_unchanged=source_unchanged,
        source_sha256=task['source_sha256'] | surface['source_sha256'],
        optimizer=dict(success=bool(optimized.success), message=str(optimized.message),
            iterations=int(optimized.nit), evaluations=int(optimized.nfev), objective=float(optimized.fun),
            max_equality_error=eq_error, min_inequality_slack=ineq_minimum),
        constraints=dict(root_translation_norm_max_m=.099, root_roll_pitch_change_max_rad=.20,
            controlled_joint_change_max_rad=.60, foot_xy_preserved=True, foot_flat_on_verified_top=True,
            minimum_com_hull_margin_m=.01, minimum_foot_site_top_margin_m=.01,
            max_self_and_scene_penetration_m=.005, held_joints='exact source nominal',
            collision_masks_modified=False, collision_excludes_modified=False, physics_steps=0),
        target=dict(root_pos_world_m=after['root'].tolist(), root_quat_wxyz=after['quaternion'].tolist(),
            joint_names=list(spec.joint_names), joint_pos_rad=after['joints'].tolist(),
            body_names=body_names, body_pos_world_m=data.xpos[body_ids].tolist(),
            body_quat_wxyz=data.xquat[body_ids].tolist()),
        changes=dict(root_translation_m=optimized.x[:3].tolist(), root_translation_norm_m=float(np.linalg.norm(optimized.x[:3])),
            root_roll_pitch_change_rad=optimized.x[3:5].tolist(),
            max_controlled_joint_change_rad=float(np.max(np.abs(optimized.x[5:])))),
        before=metrics(before), after=metrics(after), mujoco_version=mujoco.__version__, scipy_version=scipy.__version__)
    payload = json.dumps(result['target'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    result['target_sha256'] = hashlib.sha256(payload).hexdigest()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
