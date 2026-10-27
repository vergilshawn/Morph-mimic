"""Robot spec loading and joint-name mapping helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RobotSpec:
    name: str
    num_joints: int
    joint_names: list
    controlled_joint_names: list
    held_joint_names: list
    default_pose: list
    stiffness: list
    damping: list
    effort_limits: list
    armature: list
    action_scale: list
    pos_limits_lower: list
    pos_limits_upper: list
    vel_limits: list
    control_dt: float
    physics_dt: float
    base_link: str
    contact_bodies: dict  # body name -> candidate link names
    motion_fps: int
    runtime_assets: dict
    motion_dir: str
    source: str = ""
    urdf_joint_order: list | None = None
    asset_aliases: dict | None = None

    # derived (filled by finalize())
    n_controlled: int = 0
    joint_pos_limits: torch.Tensor | None = None
    pd_gains: torch.Tensor | None = None  # (num_joints, 2) [kp, kd]
    default_pose_t: torch.Tensor | None = None
    vel_limits_t: torch.Tensor | None = None
    action_scale_t: torch.Tensor | None = None
    effort_limits_t: torch.Tensor | None = None

    @staticmethod
    def load(robot: str) -> "RobotSpec":
        path = PROJECT_ROOT / "assets" / robot / "robot_spec.json"
        return RobotSpec.from_json(path)

    @classmethod
    def from_json(cls, path) -> "RobotSpec":
        data = json.loads(Path(path).read_text())
        data.pop("description", None)
        spec = cls(**data)
        spec.finalize()
        return spec

    def finalize(self):
        self.n_controlled = len(self.controlled_joint_names)
        self.joint_pos_limits = torch.tensor(
            [self.pos_limits_lower, self.pos_limits_upper], dtype=torch.float32
        ).T  # (J, 2)
        self.pd_gains = torch.tensor(
            [self.stiffness, self.damping], dtype=torch.float32
        ).T  # (J, 2)
        self.default_pose_t = torch.tensor(self.default_pose, dtype=torch.float32)
        self.vel_limits_t = torch.tensor(self.vel_limits, dtype=torch.float32)
        self.action_scale_t = torch.tensor(self.action_scale, dtype=torch.float32)
        self.effort_limits_t = torch.tensor(self.effort_limits, dtype=torch.float32)

    def index_map(self, other_names: list[str]) -> torch.Tensor:
        """Permutation aligning spec order to `other_names` (e.g. asset joint order)."""
        idx = [self.joint_names.index(nm) for nm in other_names]
        return torch.tensor(idx, dtype=torch.long)

    def controlled_idx(self, names: list[str]) -> torch.Tensor:
        """Indices of the 23 controlled joints inside `names`."""
        idx = [names.index(nm) for nm in self.controlled_joint_names]
        return torch.tensor(idx, dtype=torch.long)

    def nine_body_groups(self) -> list[str]:
        """Canonical order of the nine contact bodies: hands, elbows, knees, feet, pelvis.

        Each group expands to [left, right] links at asset-resolution time;
        the pelvis is a single link. Paper ordering: hands, elbows, knees,
        feet, pelvis.
        """
        return ["hands", "elbows", "knees", "feet", "pelvis"]

    def resolve_contact_bodies(self, body_names):
        """Resolve anatomical order before or after physics initialization."""
        aliases = (self.asset_aliases or {}).get("bodies", {})
        names, columns = [], []
        for group in self.nine_body_groups():
            candidates = self.contact_bodies[group]
            pairs = [candidates] if len(candidates) == 1 else [
                [name for name in candidates if side in name] for side in ("left", "right")
            ]
            for choices in pairs:
                match = None
                for candidate in choices:
                    normalized = candidate.replace("_link", "").replace("_joint", "")
                    if candidate in body_names:
                        match = candidate
                    elif aliases.get(candidate) in body_names:
                        match = aliases[candidate]
                    else:
                        match = next((name for name in body_names
                                      if name.replace("_link", "").replace("_joint", "") == normalized), None)
                    if match is not None:
                        break
                if match is None:
                    raise ValueError(f"contact candidates {choices} not found in asset bodies")
                names.append(match)
                columns.append(choices)
        if len(names) != 9 or len(set(names)) != 9:
            raise ValueError("robot must resolve to nine distinct contact bodies")
        return names, columns
