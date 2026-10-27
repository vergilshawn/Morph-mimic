"""Effective MuJoCo robot self-collision candidates and independent audits."""

from collections import defaultdict

import numpy as np


def effective_robot_geom_pairs(model, scene_geom_ids=()):
    """Fixed rigid geom pairs matching MuJoCo's default body/geom filters.

    Mirrors MuJoCo 3.6 filterBodyPair: same weld body is skipped; parent
    weld bodies are skipped unless FILTERPARENT is disabled. Explicit pairs
    bypass dynamic excludes/masks, as they do in the engine. The independent
    audit below always calls mj_collision instead of trusting this enumeration.
    """
    import mujoco

    if model.nflex:
        raise ValueError("fixed self-collision optimization supports rigid robots only")
    if hasattr(mujoco, "get_mjcb_contactfilter") and mujoco.get_mjcb_contactfilter() is not None:
        raise ValueError("custom collision callback cannot be represented by fixed default-filter pairs")
    disabled = int(model.opt.disableflags)
    if disabled & (int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)):
        raise ValueError("self-collision optimization requires collision detection to remain enabled")
    scenes = set(scene_geom_ids)
    robot_geoms = [g for g in range(model.ngeom) if g not in scenes and model.geom_bodyid[g] > 0]
    explicit = {tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    excluded = {int(signature) for signature in model.exclude_signature}
    parent_filter_disabled = bool(disabled & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    pairs = []
    for index, first in enumerate(robot_geoms):
        for second in robot_geoms[index + 1:]:
            pair = first, second
            if pair in explicit:
                pairs.append(pair)
                continue
            b0, b1 = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
            weld0, weld1 = int(model.body_weldid[b0]), int(model.body_weldid[b1])
            if weld0 == weld1:
                continue
            parent0 = int(model.body_weldid[model.body_parentid[weld0]])
            parent1 = int(model.body_weldid[model.body_parentid[weld1]])
            if not parent_filter_disabled and weld0 and weld1 and (weld0 == parent1 or weld1 == parent0):
                continue
            low, high = sorted((b0, b1))
            if (low << 16) + high in excluded:
                continue
            if not ((model.geom_contype[first] & model.geom_conaffinity[second])
                    or (model.geom_contype[second] & model.geom_conaffinity[first])):
                continue
            pairs.append(pair)
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def sample_self_contacts(model, states, scene_geom_ids=(), fps=50, max_penetration_m=.02):
    import mujoco

    if int(model.opt.disableflags) & (int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)):
        raise ValueError("cannot validate self-collisions with collision detection disabled")
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != model.nq or not np.isfinite(states).all():
        raise ValueError("states must be a finite [frames,model.nq] array")
    scene_geom_ids = set(scene_geom_ids)
    active = [i for i in range(model.ngeom) if i not in scene_geom_ids and model.geom_bodyid[i] > 0
              and (model.geom_contype[i] or model.geom_conaffinity[i])]
    if not active:
        raise ValueError("model contains no active robot collision geometry")
    data = mujoco.MjData(model)
    frames, pairs = [], defaultdict(lambda: dict(contact_samples=0, maximum_penetration_m=0, worst_frame=None))
    worst = None
    for frame, state in enumerate(states):
        data.qpos[:], data.qvel[:] = state, 0
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        contacts = []
        for contact in data.contact:
            g0, g1 = map(int, contact.geom)
            b0, b1 = int(model.geom_bodyid[g0]), int(model.geom_bodyid[g1])
            if g0 in scene_geom_ids or g1 in scene_geom_ids or b0 == 0 or b1 == 0:
                continue
            penetration = max(0.0, -float(contact.dist))
            item = dict(frame=frame, time_s=frame / fps, penetration_m=penetration,
                        signed_distance_m=float(contact.dist),
                        body_names=[model.body(b0).name, model.body(b1).name],
                        geom_names=[model.geom(g0).name, model.geom(g1).name],
                        contact_position_w=contact.pos.tolist())
            contacts.append(item)
            key = " <-> ".join(sorted(item["body_names"]))
            pair = pairs[key]
            pair["contact_samples"] += 1
            if penetration > pair["maximum_penetration_m"]:
                pair.update(maximum_penetration_m=penetration, worst_frame=frame)
            if worst is None or penetration > worst["penetration_m"]:
                worst = item
        peak = max((c["penetration_m"] for c in contacts), default=0.0)
        frames.append(dict(frame=frame, time_s=frame / fps, maximum_penetration_m=peak,
                           contact_count=len(contacts), over_threshold=peak > max_penetration_m,
                           contacts=contacts))
    maximum = max((f["maximum_penetration_m"] for f in frames), default=0.0)
    return dict(
        num_frames=len(states), fps=fps, duration_s=(len(states) - 1) / fps,
        accepted=maximum <= max_penetration_m, maximum_penetration_m=maximum,
        frames_over_threshold=sum(f["over_threshold"] for f in frames),
        frames_with_self_contact=sum(f["contact_count"] > 0 for f in frames),
        worst_contact=worst, body_pairs=dict(pairs), frames=frames,
        robot_active_collision_geom_count=len(active),
    )

