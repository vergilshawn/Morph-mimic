"""Small deterministic named references generated without downloaded datasets.

These arrays exercise loader/frame contracts, not physical motion feasibility.
Projection tests independently verify their resulting MuJoCo FK.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def named_motion(*, frames=5, moving_held=False, category="walk"):
    spec = RobotSpec.load("g1")
    tree = ET.parse(PROJECT_ROOT / spec.runtime_assets["mjcf"])
    bodies = [node.get("name") for node in tree.findall(".//worldbody//body")]
    root = np.column_stack((2. + np.arange(frames) * .004,
                            -3. - np.arange(frames) * .002,
                            np.full(frames, .8)))
    yaw = .3 + np.arange(frames) * .01
    quat = np.column_stack((np.cos(yaw / 2), np.zeros((frames, 2)), np.sin(yaw / 2)))
    q = np.tile(np.asarray(spec.default_pose), (frames, 1))
    dq = np.zeros_like(q)
    q[:, 0] += np.linspace(0., .04, frames)
    dq[:, 0] = .5
    if moving_held:
        held = spec.joint_names.index(spec.held_joint_names[0])
        q[:, held] += np.linspace(.1, .2, frames)
        dq[:, held] = 1.25
    positions = root[:, None, :] + np.arange(len(bodies))[None, :, None] * np.array([.001, .002, -.003])
    positions[:, bodies.index(spec.base_link)] = root
    return dict(fps=np.array(50.), category=np.array(category),
                joint_names=np.array(spec.joint_names), body_names=np.array(bodies),
                joint_pos=q.astype(np.float32), joint_vel=dq.astype(np.float32), body_pos_w=positions,
                body_quat_w=np.tile(quat[:, None, :], (1, len(bodies), 1)),
                body_lin_vel_w=np.tile(np.array([.2, -.1, 0.]), (frames, len(bodies), 1)),
                body_ang_vel_w=np.tile(np.array([0., 0., .5]), (frames, len(bodies), 1)))


def scene_motion(directory, clip="box_fixture"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = named_motion(category="box_climb")
    base = arrays["body_names"].tolist().index("pelvis")
    arrays.update(root_pos=arrays["body_pos_w"][:, base].copy(),
                  root_rot_wxyz=arrays["body_quat_w"][:, base].copy(), requires_scene=np.array(True))
    motion = directory / f"{clip}.npz"
    np.savez(motion, **arrays)
    source = directory / "source.npz"
    np.savez(source, qpos=np.zeros((5, 36)))
    terrain = directory / "terrain.urdf"
    terrain.write_text('<robot name="fixture"><link name="box"><collision name="box">'
                       '<origin xyz="2.8 -3 0.25"/><geometry><box size="0.6 0.8 0.5"/>'
                       '</geometry></collision></link></robot>')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(schema=1, robot="g1", motion_file=motion.name, motion_sha256=digest(motion),
                    source_motion=dict(path=str(source), sha256=digest(source)),
                    scene_urdf=dict(path=str(terrain), sha256=digest(terrain)),
                    source_first_root_xy=arrays["root_pos"][0, :2].tolist(),
                    ground=dict(top_z_m=0., size_m=[10., 10.], thickness_m=1., center_xy_m=[0., 0.]),
                    collision_asset_sha256={})
    motion.with_suffix(".scene.json").write_text(json.dumps(metadata))
    return motion
