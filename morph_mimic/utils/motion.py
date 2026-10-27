"""Motion dataset: load retargeted clips (npz) into canonical reference tensors.

Supports two local formats:
- supermimic G1: body kinematics (no explicit root fields; body 0 = pelvis)
- supermimic mojia: explicit root_pos / root_rot_wxyz + joint arrays

Canonical per-clip tensors (all torch, float32):
    root_pos      (T, 3)  world position of the base link
    root_quat     (T, 4)  wxyz orientation of the base link
    root_lin_vel  (T, 3)  base linear velocity (world)
    root_ang_vel  (T, 3)  base angular velocity (world)
    joint_pos     (T, J)  J = controlled joints, in spec order
    joint_vel     (T, J)
"""

from __future__ import annotations

import re
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from morph_mimic.utils.specs import RobotSpec

CLIP_CATEGORIES = ("walk", "run", "sprint", "dance", "fight", "fightAndSports", "jumps", "fallAndGetUp",
                   "box_climb", "stairs", "vault", "ladder", "sit", "get_up", "sparse_terrain")

_CLIP_CATEGORY_RE = re.compile(r"^([a-zA-Z]+?)\d*_")


@dataclass
class MotionClip:
    name: str
    category: str
    fps: float
    duration_s: float
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_names: list
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]

    def body_idx(self, name: str) -> int | None:
        """Index of a body by (suffix-normalized) name, or None."""
        if name in self.body_names:
            return self.body_names.index(name)
        norm = name.replace("_link", "").replace("_joint", "")
        for i, bn in enumerate(self.body_names):
            if bn.replace("_link", "").replace("_joint", "") == norm:
                return i
        return None

    def sample(self, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        """Interpolate the clip at phases in [0, 1).

        Returns dict of tensors with a leading batch dim.
        """
        phase = phase.clamp(0.0, 1.0 - 1e-6)
        pos = phase * (self.num_frames - 1)
        idx0 = pos.floor().long()
        idx1 = (idx0 + 1).clamp(max=self.num_frames - 1)
        frac = (pos - idx0).unsqueeze(-1)
        out = {}
        for key in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel"):
            v = getattr(self, key)
            if key == "root_quat":
                # slerp not needed for tiny phase steps; nlerp is fine and cheap
                q0, q1 = v[idx0], v[idx1]
                dot = (q0 * q1).sum(-1, keepdim=True)
                q1 = torch.where(dot < 0, -q1, q1)
                v = q0 + frac * (q1 - q0)
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            else:
                v = v[idx0] * (1 - frac) + v[idx1] * frac
            out[key] = v
        return out

    def sample_body_pos(self, phase: torch.Tensor, body_idx: int) -> torch.Tensor:
        """Interpolated world position of one body at phases (batch)."""
        phase = phase.clamp(0.0, 1.0 - 1e-6)
        pos = phase * (self.num_frames - 1)
        idx0 = pos.floor().long()
        idx1 = (idx0 + 1).clamp(max=self.num_frames - 1)
        frac = (pos - idx0).unsqueeze(-1)
        return self.body_pos_w[idx0, body_idx] * (1 - frac) + self.body_pos_w[idx1, body_idx] * frac


def validate_motion_arrays(data, spec: RobotSpec, source="motion", *, require_nominal_held=True):
    """Validate the complete named robot state before selecting controlled DOFs.

    A 23-action G1 still has 29 physical joints. Its reference FK must use the
    same nominal held joints as the controller; dropping six action columns
    from an arbitrary 29-DOF motion does not establish that contract.
    """
    names = {}
    for key in ("joint_names", "body_names"):
        value = np.asarray(data[key])
        if value.ndim != 1 or not len(value) or value.dtype.kind not in "US":
            raise ValueError(f"{source}: {key} must contain explicit names")
        labels = value.tolist()
        if len(set(labels)) != len(labels):
            raise ValueError(f"{source}: duplicate {key}")
        names[key] = labels
    if set(names["joint_names"]) != set(spec.joint_names):
        raise ValueError(f"{source}: full joint_names must match all {spec.num_joints} robot joints, including held joints")
    if spec.base_link not in names["body_names"]:
        raise ValueError(f"{source}: missing base link {spec.base_link}")
    fps = np.asarray(data["fps"]).reshape(-1)
    if fps.size != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
        raise ValueError(f"{source}: fps must be one finite positive value")
    joint_pos = np.asarray(data["joint_pos"])
    if joint_pos.ndim != 2 or len(joint_pos) < 2:
        raise ValueError(f"{source}: joint_pos must have at least two frames")
    frames, joints, bodies = len(joint_pos), spec.num_joints, len(names["body_names"])
    shapes = {
        "joint_pos": (frames, joints), "joint_vel": (frames, joints),
        "body_pos_w": (frames, bodies, 3), "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3), "body_ang_vel_w": (frames, bodies, 3),
    }
    explicit_root = [key in data for key in ("root_pos", "root_rot_wxyz")]
    if any(explicit_root) and not all(explicit_root):
        raise ValueError(f"{source}: root_pos and root_rot_wxyz must be supplied together")
    if all(explicit_root):
        shapes.update(root_pos=(frames, 3), root_rot_wxyz=(frames, 4))
    for key, expected in shapes.items():
        value = np.asarray(data[key])
        if value.shape != expected or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{source}: {key} must be finite numeric values with shape {expected}")
    for key in ("body_quat_w", "root_rot_wxyz"):
        if key in data and np.max(np.abs(np.linalg.norm(data[key], axis=-1) - 1.0)) > 1e-3:
            raise ValueError(f"{source}: {key} must contain unit WXYZ quaternions")
    if all(explicit_root):
        base = names["body_names"].index(spec.base_link)
        if not np.allclose(data["root_pos"], data["body_pos_w"][:, base], rtol=0, atol=1e-5):
            raise ValueError(f"{source}: explicit root position differs from base-link FK")
        dots = np.abs(np.sum(np.asarray(data["root_rot_wxyz"]) * np.asarray(data["body_quat_w"])[:, base], axis=-1))
        if not np.allclose(dots, 1.0, rtol=0, atol=1e-5):
            raise ValueError(f"{source}: explicit root orientation differs from base-link FK")
    if require_nominal_held and spec.held_joint_names:
        indices = [names["joint_names"].index(name) for name in spec.held_joint_names]
        nominal = np.asarray(spec.default_pose)[[spec.joint_names.index(name) for name in spec.held_joint_names]]
        max_error = float(np.max(np.abs(joint_pos[:, indices] - nominal)))
        max_speed = float(np.max(np.abs(np.asarray(data["joint_vel"])[:, indices])))
        if max_error > 1e-6 or max_speed > 1e-6:
            raise ValueError(
                f"{source}: held joints move in this reference (max nominal deviation {max_error:.6g} rad, "
                f"speed {max_speed:.6g} rad/s). Generate consistent FK first with "
                "python -m morph_mimic.scripts.project_reference --robot "
                f"{spec.name} --source-dir <motion-directory> --output-dir <new-directory>."
            )


def validate_contact_labels(data, num_frames: int, fps: float, source="contact labels"):
    """Reject labels that cannot be indexed one-to-one with the source motion.

    In particular, a failed replay's short record must never be stretched over
    a complete reference clip by normalized-phase lookup.
    """
    expected = {
        "contact_flags": (num_frames, 9),
        "contact_targets": (num_frames, 9, 3),
        "contact_target_flags": (num_frames, 9),
    }
    for key, shape in expected.items():
        if key not in data:
            raise ValueError(f"{source}: missing {key}")
        value = np.asarray(data[key])
        if value.shape != shape:
            raise ValueError(f"{source}: {key} shape {value.shape}, expected {shape}; replay must cover the complete motion")
        if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
            raise ValueError(f"{source}: {key} must contain numeric values")
        if not np.isfinite(value).all():
            raise ValueError(f"{source}: {key} contains nonfinite values")
        if key != "contact_targets" and not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{source}: {key} must contain binary flags")
    if "fps" not in data:
        raise ValueError(f"{source}: missing fps")
    label_fps = np.asarray(data["fps"]).reshape(-1)
    if label_fps.size != 1 or not np.isfinite(label_fps[0]) or not np.isclose(label_fps[0], fps, rtol=0, atol=1e-6):
        raise ValueError(f"{source}: label fps {label_fps.tolist()} differs from source fps {fps}")
    if "source_num_frames" in data and int(np.asarray(data["source_num_frames"]).item()) != num_frames:
        raise ValueError(f"{source}: source_num_frames does not match the motion")
    if "source_init_phase" in data and not np.isclose(float(np.asarray(data["source_init_phase"]).item()), 0.0, atol=1e-6):
        raise ValueError(f"{source}: replay did not start at phase zero")
    if "source_end_phase" in data:
        end_phase = float(np.asarray(data["source_end_phase"]).item())
        if not np.isfinite(end_phase) or end_phase < 1.0 - 1e-5:
            raise ValueError(f"{source}: replay ended before the motion finished")
    if "body_names" in data:
        bodies = np.asarray(data["body_names"])
        if bodies.shape != (9,) or len(set(bodies.tolist())) != 9:
            raise ValueError(f"{source}: body_names must identify nine distinct contact bodies")


def validate_contact_body_order(data, spec: RobotSpec, source="contact labels"):
    """Check every label column against the planner's canonical body order."""
    if "body_names" not in data:
        warnings.warn(
            f"{source}: legacy labels have no body_names; contact column order is unverified. "
            "Regenerate labels from a complete replay.", RuntimeWarning, stacklevel=2,
        )
        return
    body_names = np.asarray(data["body_names"])
    if body_names.shape != (9,) or body_names.dtype.kind not in "US":
        raise ValueError(f"{source}: body_names must contain nine explicit contact body names")
    aliases = (spec.asset_aliases or {}).get("bodies", {})
    expected = []
    for group in spec.nine_body_groups():
        candidates = spec.contact_bodies[group]
        if len(candidates) == 1:
            expected.append((group, candidates))
        else:
            for side in ("left", "right"):
                sided = [name for name in candidates if name.startswith(side)] or [name for name in candidates if side in name]
                if not sided:
                    raise ValueError(f"{source}: spec contact group {group} has no {side} candidates")
                expected.append((f"{side} {group}", sided))
    if len(expected) != 9:
        raise ValueError(f"{source}: robot spec must define exactly nine contact columns")
    normalize = lambda name: name.replace("_link", "").replace("_joint", "")
    for column, (actual, (label, candidates)) in enumerate(zip(body_names.tolist(), expected)):
        allowed = set(candidates) | {aliases[name] for name in candidates if name in aliases}
        if actual not in allowed and normalize(actual) not in {normalize(name) for name in allowed}:
            raise ValueError(f"{source}: contact column {column} is {actual}, expected {label} ({', '.join(sorted(allowed))}); regenerate labels")


class MotionDataset:
    """Collection of clips for one robot, in spec (controlled-joint) order."""

    def __init__(self, spec: RobotSpec, motion_dir: str | Path, device: str = "cpu", clip_names: list[str] | None = None):
        self.spec = spec
        self.device = device
        self.motion_dir = Path(motion_dir).resolve()
        self.clips: dict[str, MotionClip] = {}
        paths = [path for path in sorted(self.motion_dir.glob("*.npz")) if not path.stem.endswith("_contacts")]
        if clip_names is not None:
            selected = set(clip_names)
            missing = selected - {path.stem for path in paths}
            if missing:
                raise ValueError(f"unknown motion clips in {motion_dir}: {', '.join(sorted(missing))}")
            paths = [path for path in paths if path.stem in selected]
        if not paths:
            raise ValueError(f"no motion clips selected in {motion_dir}")
        for npz_path in paths:
            clip = self._load_clip(npz_path)
            self.clips[clip.name] = clip
        self.clip_names = list(self.clips)
        self.scene_bindings = {}
        for path in paths:
            sidecar = path.with_suffix(".scene.json")
            with np.load(path, allow_pickle=False) as archive:
                required = np.asarray(archive["requires_scene"]) if "requires_scene" in archive else np.array(False)
                if required.shape != () or required.dtype != np.bool_:
                    raise ValueError(f"{path}: requires_scene must be a scalar boolean")
            if sidecar.exists():
                from morph_mimic.utils.reference_scene import load_reference_scene

                scene = load_reference_scene(sidecar, path)
                if scene.metadata["robot"] != spec.name:
                    raise ValueError(f"{sidecar}: scene robot differs from {spec.name}")
                self.scene_bindings[path.stem] = scene
            elif required:
                raise ValueError(f"{path}: required scene sidecar is missing: {sidecar.name}")
        self.contact_flags: dict[str, torch.Tensor] = {}
        self.contact_targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.contact_target_frames: dict[str, str] = {}
        for name in self.clip_names:
            label_path = Path(motion_dir) / f"{name}_contacts.npz"
            if label_path.exists():
                clip = self.clips[name]
                with np.load(label_path, allow_pickle=False) as data:
                    validate_contact_labels(data, clip.num_frames, clip.fps, label_path)
                    validate_contact_body_order(data, spec, label_path)
                    if "source_sha256" in data:
                        source_hash = hashlib.sha256((Path(motion_dir) / f"{name}.npz").read_bytes()).hexdigest()
                        if str(np.asarray(data["source_sha256"]).item()) != source_hash:
                            raise ValueError(f"{label_path}: source motion SHA256 differs; regenerate its contact labels")
                    if name in self.scene_bindings:
                        expected = self.scene_bindings[name].fingerprint
                        if "scene_fingerprint" not in data or str(np.asarray(data["scene_fingerprint"]).item()) != expected:
                            raise ValueError(f"{label_path}: contact labels do not bind the current collision scene")
                    if "contact_target_frame" in data:
                        frame = str(np.asarray(data["contact_target_frame"]).item())
                    else:
                        method = str(np.asarray(data["method"]).item()) if "method" in data else ""
                        frame = ("reference_query_body_yaw_absolute_z"
                                 if method in {f"reference_geometry_mujoco_collision_distance_v{version}"
                                               for version in (1, 2, 3)}
                                 else "legacy_query_body_yaw_absolute_z_unverified")
                    if frame not in {"reference_query_body_yaw_absolute_z", "legacy_query_body_yaw_absolute_z_unverified"}:
                        raise ValueError(f"{label_path}: unsupported contact target coordinate frame {frame!r}")
                    if frame == "legacy_query_body_yaw_absolute_z_unverified":
                        message = (f"{label_path}: legacy contact targets lack their generating query pose; "
                                   "cannot reproject them exactly into the current robot body-yaw frame")
                        if name in self.scene_bindings:
                            raise ValueError(message + "; bound scene training requires reference-query labels")
                        warnings.warn(message + "; preserving legacy unbound behavior", RuntimeWarning)
                    self.contact_target_frames[name] = frame
                    self.contact_flags[name] = torch.from_numpy(
                        data["contact_flags"].astype(np.float32)
                    ).to(device)
                    self.contact_targets[name] = (
                        torch.from_numpy(data["contact_targets"].astype(np.float32)).to(device),
                        torch.from_numpy(data["contact_target_flags"].astype(np.float32)).to(device),
                    )
        if len(self.contact_flags) != len(self.clips):
            print(
                f"[MotionDataset] {len(self.clips) - len(self.contact_flags)}/{len(self.clips)} clips lack "
                f"*_contacts.npz labels in {motion_dir}; run scripts/label_contacts.py first "
                "(reference contact rewards are unavailable for these clips)"
            )

    @property
    def names(self) -> list[str]:
        return list(self.clips)

    def by_category(self, category: str) -> list[MotionClip]:
        return [c for c in self.clips.values() if c.category == category]

    def sample(self, clip_ids: torch.Tensor, phases: torch.Tensor) -> dict[str, torch.Tensor]:
        """Vectorized sampling: clip_ids (K,), phases (K,) -> dict of (K, ...)."""
        k = clip_ids.shape[0]
        out = {
            "root_pos": torch.zeros(k, 3, device=self.device),
            "root_quat": torch.zeros(k, 4, device=self.device),
            "root_lin_vel": torch.zeros(k, 3, device=self.device),
            "root_ang_vel": torch.zeros(k, 3, device=self.device),
            "joint_pos": torch.zeros(k, 23, device=self.device),
            "joint_vel": torch.zeros(k, 23, device=self.device),
        }
        for cid in torch.unique(clip_ids).tolist():
            m = clip_ids == cid
            s = self.clips[self.clip_names[cid]].sample(phases[m])
            for key in out:
                out[key][m] = s[key]
        out["root_quat"][out["root_quat"].norm(dim=-1) < 0.5, 0] = 1.0
        return out

    def sample_body_pos(
        self, clip_ids: torch.Tensor, phases: torch.Tensor, body_name: str
    ) -> torch.Tensor:
        """Vectorized body-position sampling; missing bodies yield NaN rows."""
        k = clip_ids.shape[0]
        out = torch.full((k, 3), float("nan"), device=self.device)
        for cid in torch.unique(clip_ids).tolist():
            clip = self.clips[self.clip_names[cid]]
            bi = clip.body_idx(body_name)
            if bi is None:
                continue
            m = clip_ids == cid
            out[m] = clip.sample_body_pos(phases[m], bi)
        return out

    def _load_clip(self, path: Path) -> MotionClip:
        spec = self.spec
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        validate_motion_arrays(data, spec, path)
        joint_names = [str(n) for n in data["joint_names"]]
        body_names = [str(n) for n in data["body_names"]]
        assert spec.base_link in body_names, f"base link {spec.base_link} not in {path.name}"
        base_idx = body_names.index(spec.base_link)

        if "root_pos" in data and "root_rot_wxyz" in data:
            root_pos = data["root_pos"]
            root_quat = data["root_rot_wxyz"]
        else:
            # G1 supermimic format: body 0 holds the base pose
            root_pos = data["body_pos_w"][:, base_idx]
            root_quat = data["body_quat_w"][:, base_idx]
        root_lin_vel = data["body_lin_vel_w"][:, base_idx]
        root_ang_vel = data["body_ang_vel_w"][:, base_idx]

        # reorder joints into spec order (controlled joints only)
        idx = [joint_names.index(nm) for nm in spec.controlled_joint_names]
        joint_pos = torch.from_numpy(data["joint_pos"][:, idx].astype(np.float32))
        joint_vel = torch.from_numpy(data["joint_vel"][:, idx].astype(np.float32))

        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        name = path.stem
        if "category" in data:
            value = np.asarray(data["category"])
            if value.shape != () or value.dtype.kind not in "US":
                raise ValueError(f"{path}: category must be one explicit string")
            category = str(value.item())
        else:
            m = _CLIP_CATEGORY_RE.match(name)
            category = m.group(1) if m else "other"
        if category not in CLIP_CATEGORIES:
            raise ValueError(f"unexpected clip category {category!r} in {name}")

        return MotionClip(
            name=name,
            category=category,
            fps=fps,
            duration_s=(joint_pos.shape[0] - 1) / fps,
            root_pos=torch.from_numpy(root_pos.astype(np.float32)).to(self.device),
            root_quat=torch.from_numpy(root_quat.astype(np.float32)).to(self.device),
            root_lin_vel=torch.from_numpy(root_lin_vel.astype(np.float32)).to(self.device),
            root_ang_vel=torch.from_numpy(root_ang_vel.astype(np.float32)).to(self.device),
            joint_pos=joint_pos.to(self.device),
            joint_vel=joint_vel.to(self.device),
            body_names=body_names,
            body_pos_w=torch.from_numpy(data["body_pos_w"].astype(np.float32)).to(self.device),
            body_quat_w=torch.from_numpy(data["body_quat_w"].astype(np.float32)).to(self.device),
        )


def check_root_height(clip: MotionClip) -> float:
    """Root height relative to the motion's minimum (sanity check)."""
    return float(clip.root_pos[:, 2].min())
