"""Generate robot_spec.json for each migrated robot asset.

Extracts joint order, limits, PD gains and default poses from the
migrated assets (MJCF/URDF + control.json) and writes a single spec
file consumed by the training pipeline.

Usage:
    python morph_mimic/scripts/make_robot_specs.py [--overwrite]
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Paper (arXiv 2610.31608) controls 23 joints on G1: legs (12), waist yaw,
# arms incl. wrist roll only (10). The 29-DOF MJCF additionally has waist
# roll/pitch and wrist pitch/yaw, which are held at their default pose.
G1_HELD_JOINTS = [
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Paper contact bodies: feet, knees, hands, elbows, pelvis. For each robot we
# list candidate link names (MJCF / USD / URDF variants); resolved at runtime
# against the actually loaded articulation.
G1_CONTACT_BODIES = {
    "feet": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "knees": ["left_knee_link", "right_knee_link"],
    "hands": ["left_rubber_hand", "left_hand_link", "left_wrist_yaw_link",
              "right_rubber_hand", "right_hand_link", "right_wrist_yaw_link"],
    "elbows": ["left_elbow_link", "right_elbow_link"],
    "pelvis": ["pelvis"],
}
MOJIA_CONTACT_BODIES = {
    "feet": ["left_ankle_roll", "right_ankle_roll"],
    "knees": ["left_knee_pitch", "right_knee_pitch"],
    # mojia has no hand link; the forearm (elbow_yaw) is the distal arm link
    "hands": ["left_elbow_yaw", "right_elbow_yaw"],
    "elbows": ["left_elbow_pitch", "right_elbow_pitch"],
    "pelvis": ["base_link"],
}


def load_mujoco_joints(mjcf_path):
    """Return joint names, limits (lower/upper) in MJCF DFS order."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    names, lower, upper, dclass = [], [], [], []
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = m.jnt_type[i]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(name)
        # ranges: hinge/slide have jnt_range; ball joints unsupported here
        lower.append(float(m.jnt_range[i][0]))
        upper.append(float(m.jnt_range[i][1]))
    return names, lower, upper


def load_urdf_joints(urdf_path):
    """Return joint names, limits, velocity limits, effort limits in URDF order."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    names, lower, upper, vel, effort = [], [], [], [], []
    for j in root.iter("joint"):
        if j.attrib.get("type") not in ("revolute", "continuous"):
            continue
        names.append(j.attrib["name"])
        lim = j.find("limit")
        if lim is None:
            lower.append(-np.pi)
            upper.append(np.pi)
            vel.append(100.0)
            effort.append(100.0)
        else:
            lo, hi = float(lim.attrib["lower"]), float(lim.attrib["upper"])
            lower.append(lo if np.isfinite(lo) else -np.pi)
            upper.append(hi if np.isfinite(hi) else np.pi)
            vel.append(float(lim.attrib.get("velocity", 100.0)))
            effort.append(float(lim.attrib.get("effort", 100.0)))
    return names, lower, upper, vel, effort


def make_g1_spec():
    asset_dir = ROOT / "assets" / "g1"
    control = json.loads((asset_dir / "control.json").read_text())
    names, lower, upper = load_mujoco_joints(asset_dir / "robot" / "g1.xml")

    assert names == control["joint_names"], "MJCF and control.json joint order mismatch"
    n = len(names)
    assert n == 29 and all(len(v) == n for v in (control["stiffness"], control["damping"],
                                                control["effort_limit"], control["armature"],
                                                control["nominal"], control["action_scale"]))

    controlled = [nm for nm in names if nm not in G1_HELD_JOINTS]
    held = [nm for nm in names if nm in G1_HELD_JOINTS]
    assert len(controlled) == 23, f"expected 23 controlled joints, got {len(controlled)}"

    spec = {
        "name": "g1",
        "description": "Unitree G1 29-DOF (MJCF from supermimic/data/g1); 23 joints controlled per ACP paper",
        "num_joints": n,
        "joint_names": names,
        "controlled_joint_names": controlled,
        "held_joint_names": held,
        "default_pose": control["nominal"],
        "stiffness": control["stiffness"],
        "damping": control["damping"],
        "effort_limits": control["effort_limit"],
        "armature": control["armature"],
        "action_scale": control["action_scale"],
        "pos_limits_lower": lower,
        "pos_limits_upper": upper,
        "vel_limits": [20.0] * n,
        "control_dt": control["control_dt"],
        "physics_dt": control["physics_dt"],
        "base_link": "pelvis",
        "contact_bodies": G1_CONTACT_BODIES,
        "motion_fps": 50,
        "runtime_assets": {
            "urdf": None,  # generated by scripts/convert_g1_mjcf_to_urdf.py if needed
            "usd_cfg": "isaaclab_assets:G1_29DOF_CFG",  # 29-DOF USD; dynamics still require validation
            "mjcf": "assets/g1/robot/g1.xml",
        },
        "motion_dir": "assets/g1/motions",
        "source": "supermimic/data/g1 (LAFAN1 retargeting dataset, 40 clips @50 Hz)",
    }
    (asset_dir / "robot_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"wrote {asset_dir}/robot_spec.json ({n} joints, {len(controlled)} controlled)")


def make_mojia_spec():
    asset_dir = ROOT / "assets" / "mojia"
    control = json.loads((asset_dir / "control.json").read_text())
    urdf = asset_dir / "robot" / "urdf" / "robot.urdf"
    urdf_names, lower, upper, vel, effort = load_urdf_joints(urdf)
    assert sorted(urdf_names) == sorted(control["joint_names"]), "URDF and control.json joint sets differ"

    # canonical order = control.json (MJCF DFS); reorder URDF-derived arrays to it
    idx = [urdf_names.index(nm) for nm in control["joint_names"]]
    names = [urdf_names[i] for i in idx]
    lower = [lower[i] for i in idx]
    upper = [upper[i] for i in idx]
    vel = [vel[i] for i in idx]
    effort = [effort[i] for i in idx]

    n = len(names)
    assert n == 23 and all(len(v) == n for v in (control["stiffness"], control["damping"],
                                                control["effort_limit"], control["armature"],
                                                control["nominal"], control["action_scale"]))

    spec = {
        "name": "mojia",
        "description": "Mojia v3.8 23-DOF humanoid (URDF from supermimic/assets/robots/mojia)",
        "num_joints": n,
        "joint_names": names,
        "urdf_joint_order": urdf_names,
        "controlled_joint_names": names,
        "held_joint_names": [],
        "default_pose": control["nominal"],
        "stiffness": control["stiffness"],
        "damping": control["damping"],
        "effort_limits": control["effort_limit"],
        "armature": control["armature"],
        "action_scale": control["action_scale"],
        "pos_limits_lower": lower,
        "pos_limits_upper": upper,
        "vel_limits": vel,
        "control_dt": control["control_dt"],
        "physics_dt": control["physics_dt"],
        "base_link": "base_link",
        "contact_bodies": MOJIA_CONTACT_BODIES,
        "motion_fps": 50,
        "runtime_assets": {
            "urdf": "assets/mojia/robot/urdf/robot.urdf",
            "usd_cfg": None,
            "mjcf": "assets/mojia/mjcf/mojia_v3_8.xml",
        },
        "motion_dir": "assets/mojia/motions",
        "source": "supermimic/data/mojia (retargeted from mojia_raw mocap, 40 clips @50 Hz)",
    }
    (asset_dir / "robot_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"wrote {asset_dir}/robot_spec.json ({n} joints, {len(names)} controlled)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    for fn in (make_g1_spec, make_mojia_spec):
        if (ROOT / "assets" / fn.__name__[5:] / "robot_spec.json").exists() and not args.overwrite:
            print(f"skip {fn.__name__[5:]} (exists, use --overwrite)")
            continue
        fn()


if __name__ == "__main__":
    main()
