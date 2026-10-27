"""Checkpoint provenance and atomic writes; no Isaac imports.

A restart restores learning state and curriculum, not the PhysX trajectory.
Asset identifiers for remote USDs are recorded; their bytes cannot be hashed here.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import random
import tempfile
import warnings

import numpy as np
import torch

from morph_mimic.utils.specs import PROJECT_ROOT


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_rng_state():
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "python": random.getstate(),
    }


def restore_rng_state(state):
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda"):
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            warnings.warn("CUDA RNG topology differs; CUDA generators were not restored", RuntimeWarning)
        else:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    random.setstate(state["python"])


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    raise TypeError(f"Unsupported contract field {type(value).__name__}")


def _config_value(value):
    if hasattr(value, "to_dict"):
        return _config_value(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _config_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_config_value(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return _plain(value)


def build_checkpoint_contract(env, planner=None, *, training_cfg=None):
    spec = getattr(env, "robot_spec", None)
    motion = getattr(env, "motion_data", None)
    # Tensor-only test environments have no physical/data contract to certify.
    if not hasattr(spec, "joint_names") or motion is None:
        return None
    robot_fields = (
        "name", "num_joints", "joint_names", "controlled_joint_names", "held_joint_names",
        "default_pose", "stiffness", "damping", "effort_limits", "armature", "action_scale",
        "pos_limits_lower", "pos_limits_upper", "vel_limits", "control_dt", "physics_dt",
        "base_link", "contact_bodies", "runtime_assets", "asset_aliases",
    )
    robot = {name: _plain(getattr(spec, name)) for name in robot_fields}
    asset_root = PROJECT_ROOT / "assets" / spec.name
    extensions = {".xml", ".urdf", ".stl", ".obj", ".dae", ".usd", ".usda", ".usdc", ".json"}
    asset_hashes = {}
    for path in sorted(asset_root.rglob("*")):
        relative = path.relative_to(asset_root)
        if (path.is_file() and path.suffix.lower() in extensions
                and relative.name not in {"robot_spec.json", "manifest.json"}
                and not {"motions", "raw"}.intersection(relative.parts)):
            asset_hashes[str(relative)] = file_hash(path)
    directory = Path(getattr(motion, "motion_dir", PROJECT_ROOT / spec.motion_dir))
    clips = {}
    for name in motion.clip_names:
        path = directory / f"{name}.npz"
        label = directory / f"{name}_contacts.npz"
        clips[name] = {"motion_sha256": file_hash(path),
                       "contacts_sha256": file_hash(label) if label.exists() else None}
    cfg = env.cfg
    physics_fields = ("decimation", "action_scale", "action_clip", "contact_history", "contact_force_threshold",
                      "planner_scan_shape", "tracker_scan_shape", "scan_spacing", "scan_max_distance",
                      "oracle_contact_frame")
    physics = {key: _plain(getattr(cfg, key)) for key in physics_fields if hasattr(cfg, key)}
    physics["physics_dt"] = cfg.sim.dt
    physics["control_dt"] = cfg.sim.dt * cfg.decimation
    physics["backend"] = "IsaacLab/PhysX"
    physics["actuator_model"] = "implicit_PD_with_PhysX_effort_limits"
    physics["root_linear_velocity_origin"] = "base_link_frame_origin"
    physics["quaternion"] = "world_from_link_wxyz"
    physics["runtime_asset"] = getattr(cfg.robot.spawn, "usd_path", getattr(cfg.robot.spawn, "asset_path", None))
    asset_path = physics["runtime_asset"]
    if asset_path is not None and "://" not in str(asset_path) and Path(asset_path).is_file():
        physics["runtime_asset_sha256"] = file_hash(asset_path)
    bindings = getattr(motion, "scene_bindings", {})
    if bindings:
        physics["reference_scenes"] = {name: scene.fingerprint for name, scene in bindings.items()}
        physics["contact_force_source"] = "per_body_terrain_filtered_normal_forces"
    for key in ("gravity", "physx", "physics_material"):
        if hasattr(cfg.sim, key):
            physics[key] = _config_value(getattr(cfg.sim, key))
    # The actuator solver, collision settings, and mass/inertia overrides affect
    # physical trajectories even when the robot name and source meshes agree.
    if hasattr(cfg.robot.spawn, "to_dict"):
        spawn = cfg.robot.spawn.to_dict()
        spawn.pop("usd_dir", None)  # conversion cache location is not physics
        physics["spawn"] = _config_value(spawn)
    for key in ("actuators", "soft_joint_pos_limit_factor"):
        if hasattr(cfg.robot, key):
            physics[key] = _config_value(getattr(cfg.robot, key))
    versions = {}
    for name in ("isaaclab", "isaacsim", "torch"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    keys = {"stage", "ref_env_ratio", "episode_length_s", "terrain_types", "ref_terrain_types", "terrain_size",
            "num_terrain_levels", "csi_enabled", "csi_init_phase", "csi_adjust", "success_horiz", "success_vert",
            "min_height_ref", "min_height_goal", "goal_scale_radius", "ref_goal_scale_radius", "ref_goal_lookahead",
            "grace_steps", "reference_grace_steps", "max_reference_root_error_m",
            "csi_success_root_error_m", "critic_reference_frame", "min_goal_dist", "max_goal_dist", "seed"}
    keys.update(key for key in vars(cfg) if key.startswith(("w_", "dr_", "obs_noise")) or key.endswith("scan_noise"))
    training = {key: _plain(getattr(cfg, key)) for key in sorted(keys) if hasattr(cfg, key)}
    training["num_envs"] = env.num_envs
    if training_cfg is not None:
        settings = asdict(training_cfg)
        for key in ("resume_path", "initial_checkpoint", "save_interval", "log_interval", "max_iterations", "log_root"):
            settings.pop(key, None)
        training["runner"] = _plain(settings)
    contract = {
        "schema": 1, "robot": robot, "asset_sha256": asset_hashes, "clips": clips,
        "contact_bodies": list(getattr(env, "contact_link_names", [])),
        "physics": physics, "versions": versions,
        "planner": type(planner).__name__ if planner is not None else None,
        "training": training,
    }
    # Ensure the stored contract remains an ordinary JSON-compatible value.
    json.dumps(contract, sort_keys=True, allow_nan=False)
    return contract


def validate_checkpoint_contract(checkpoint, env, planner=None, *, mode="resume", training_cfg=None):
    if mode not in {"resume", "transfer", "evaluate"}:
        raise ValueError(f"Unknown checkpoint validation mode: {mode}")
    if checkpoint.get("schema") not in (None, 2):
        raise ValueError(f"Unsupported checkpoint schema {checkpoint['schema']}")
    expected = checkpoint.get("contract")
    if expected is None:
        if hasattr(getattr(env, "robot_spec", None), "joint_names"):
            warnings.warn("Legacy checkpoint has no robot/data/physics contract; compatibility is unverified", RuntimeWarning)
        return False
    actual = build_checkpoint_contract(env, planner, training_cfg=training_cfg)
    if actual is None:
        raise ValueError("Cannot validate a physical checkpoint without a robot/data environment")
    mismatches = []
    for key in ("schema", "robot", "asset_sha256", "contact_bodies", "physics", "versions"):
        if expected.get(key) != actual.get(key):
            mismatches.append(key)
    if ((mode != "transfer" or expected.get("planner") is not None)
            and expected.get("planner") != actual.get("planner")):
        mismatches.append("planner")
    if mode == "resume":
        for key in ("clips", "training"):
            if expected.get(key) != actual.get(key):
                mismatches.append(key)
    elif mode == "evaluate":
        for name, data in actual["clips"].items():
            if expected.get("clips", {}).get(name) != data:
                mismatches.append(f"clips.{name}")
    if mismatches:
        raise ValueError("Checkpoint contract differs in " + ", ".join(mismatches)
                         + "; model, data, or training semantics changed")
    return True
