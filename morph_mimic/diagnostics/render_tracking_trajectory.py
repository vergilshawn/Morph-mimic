"""Render recorded PhysX tracking states and their reference without stepping physics.

This CPU software-EGL viewer does not execute the policy or establish MuJoCo /
PhysX dynamics equivalence. It applies recorded root and joint poses to the
verified local MJCF, then computes forward kinematics for display. Full recorded
joints are preferred; legacy controlled-joint traces explicitly disclose their
nominal reconstruction of unrecorded held joints in the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path, spec):
    """Normalize new full-joint and existing evaluation NPZ schemas."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays.get("metadata_json", "{}")))
    root = np.asarray(arrays["root_pos_w"], dtype=float)
    quat = np.asarray(arrays["root_quat_w"], dtype=float)
    phases = np.asarray(arrays["phase"], dtype=float)
    if "all_joint_pos" in arrays:
        joints = arrays["all_joint_pos"]
        names = arrays.get("all_joint_names", metadata.get("all_joint_names"))
        if names is None:
            raise ValueError("all_joint_pos requires explicit all_joint_names")
    else:
        joints = arrays["joint_pos"]
        names = arrays.get("joint_names", metadata.get("joint_names"))
        if names is None:
            raise ValueError("joint_pos requires joint_names in NPZ or metadata_json")
    names = [str(name) for name in names]
    if len(set(names)) != len(names) or set(names) - set(spec.joint_names):
        raise ValueError("trace has duplicate or unrecognized joint names")
    if np.shape(joints) != (len(root), len(names)):
        raise ValueError("recorded joint array dimensions differ from joint_names")
    missing = [name for name in spec.joint_names if name not in names]
    if set(missing) - set(spec.held_joint_names):
        raise ValueError("trace omits controlled joints")
    all_joints = np.tile(np.asarray(spec.default_pose, dtype=float), (len(root), 1))
    for column, name in enumerate(names):
        all_joints[:, spec.joint_names.index(name)] = joints[:, column]
    times = np.asarray(arrays.get("elapsed_control_time_s",
                       (np.arange(len(root)) + 1) * float(metadata.get("control_dt_s", spec.control_dt))),
                       dtype=float)
    if (root.shape != (len(times), 3) or quat.shape != (len(times), 4)
            or phases.shape != times.shape or len(times) < 1
            or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid root, quaternion, phase or increasing timestamp array")
    if not all(np.isfinite(value).all() for value in (root, quat, phases, all_joints, times)):
        raise ValueError("trace contains nonfinite values")
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 0.99) or np.any(norms > 1.01):
        raise ValueError("recorded quaternions are not normalized WXYZ")
    quat /= norms[:, None]
    terminal = bool(np.asarray(arrays.get("terminated", [False]))[-1])
    terminal |= bool(np.asarray(arrays.get("truncated", [False]))[-1])
    initial_included = False
    initial_fields = ("initial_all_joint_pos", "initial_root_pos_w", "initial_root_quat_w", "initial_phase")
    if all(key in arrays for key in initial_fields) and times[0] > 0:
        if missing:
            raise ValueError("initial_all_joint_pos requires a full-joint trace")
        initial_joints = np.asarray(arrays["initial_all_joint_pos"], dtype=float)
        if initial_joints.shape != (len(names),):
            raise ValueError("initial_all_joint_pos dimensions differ from all_joint_names")
        initial_joints = initial_joints[[names.index(name) for name in spec.joint_names]]
        root = np.vstack([arrays["initial_root_pos_w"], root])
        quat = np.vstack([arrays["initial_root_quat_w"], quat])
        all_joints = np.vstack([initial_joints, all_joints])
        phases = np.r_[float(arrays["initial_phase"]), phases]
        times = np.r_[0.0, times]
        if not all(np.isfinite(value).all() for value in (root, quat, all_joints, phases)):
            raise ValueError("initial state contains nonfinite values")
        if not np.isclose(np.linalg.norm(quat[0]), 1, atol=0.01):
            raise ValueError("initial root quaternion is not normalized")
        quat[0] /= np.linalg.norm(quat[0])
        initial_included = True
    return dict(root=root, quat=quat, joints=all_joints, phase=phases, times=times,
                metadata=metadata, missing_joints=missing, terminal=terminal,
                initial_state_included=initial_included)


def sample_pose(times, root, quat, joints, time):
    """Linear scalar interpolation and normalized shortest-path quaternion blend."""
    index = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 1))
    other = min(index + 1, len(times) - 1)
    weight = 0.0 if index == other else float(np.clip(
        (time - times[index]) / (times[other] - times[index]), 0, 1))
    quaternion = quat[other] * (-1 if np.dot(quat[index], quat[other]) < 0 else 1)
    quaternion = (1 - weight) * quat[index] + weight * quaternion
    quaternion /= np.linalg.norm(quaternion)
    return ((1 - weight) * root[index] + weight * root[other], quaternion,
            (1 - weight) * joints[index] + weight * joints[other])


def find_ffmpeg(explicit=None):
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg is required: pass --ffmpeg /path/to/ffmpeg") from None


class VideoWriter:
    def __init__(self, path, width, height, fps, ffmpeg):
        self.process = subprocess.Popen([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE)

    def append(self, array):
        self.process.stdin.write(np.ascontiguousarray(array, dtype=np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed with exit code {code}")


def render(args):
    # Force Mesa software rendering so this offline viewer does not consume the
    # training GPU. All MuJoCo imports must follow these environment settings.
    if args.backend == "software-egl":
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["EGL_PLATFORM"] = "surfaceless"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ.setdefault("LP_NUM_THREADS", "4")
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from morph_mimic.diagnostics.validate_scene_reference import build_model, _joint_indices
    from morph_mimic.utils.reference_scene import load_reference_scene
    from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec

    spec = RobotSpec.load(args.robot)
    trace = load_trace(args.trace, spec)
    scene_path = args.scene or args.motion.with_suffix(".scene.json")
    scene = load_reference_scene(scene_path, args.motion)
    with np.load(args.motion, allow_pickle=False) as archive:
        names = archive["joint_names"].tolist()
        columns = [names.index(name) for name in spec.joint_names]
        ref_joints = np.asarray(archive["joint_pos"][:, columns], dtype=float)
        ref_root = np.asarray(archive["root_pos"], dtype=float)
        ref_quat = np.asarray(archive["root_rot_wxyz"], dtype=float)
        ref_fps = float(archive["fps"])
    duration = (len(ref_root) - 1) / ref_fps
    reference_times = np.arange(len(ref_root)) / ref_fps
    if trace["metadata"].get("source_sha256") not in (None, sha256(args.motion)):
        raise ValueError("trace source_sha256 differs from supplied motion")
    if np.any(trace["phase"] < 0) or np.any(trace["phase"] > 1.00001):
        raise ValueError("trace phase must be in [0, 1]")
    # Traces are rebased world positions, while source scene geoms and original
    # references retain source coordinates. Shift only recorded root XY back.
    recorded_root = trace["root"].copy()
    recorded_root[:, :2] += np.asarray(scene.source_first_root_xy)
    model_path = args.model or PROJECT_ROOT / ".cache/mjcf" / args.robot / f"{args.robot}_explicit.xml"
    model, scene_geom_ids = build_model(model_path, scene, spec.physics_dt)
    model.vis.global_.offwidth = max(args.width, 640)
    model.vis.global_.offheight = max(args.height, 480)
    model.vis.quality.shadowsize = 1024
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    for geom in scene_geom_ids:
        model.geom_rgba[geom] = [0.35, 0.55, 0.66, 1.0]
    model.geom_rgba[model.geom("reference_scene_ground").id] = [0.77, 0.80, 0.80, 1]
    joint_qpos, _ = _joint_indices(model, spec)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    from OpenGL.GL import glGetString, GL_RENDERER
    gl_renderer = glGetString(GL_RENDERER)
    if isinstance(gl_renderer, bytes):
        gl_renderer = gl_renderer.decode("ascii", "replace")
    if args.backend == "software-egl" and gl_renderer and not any(
            name in gl_renderer.lower() for name in ("llvmpipe", "softpipe", "swrast")):
        renderer.close()
        raise RuntimeError(f"requested CPU software rendering, obtained {gl_renderer!r}")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Frame the complete ascent as well as the initial stance. A pelvis-height
    # target clips the head after mounting the high box.
    camera.lookat[:] = [ref_root[0, 0], ref_root[0, 1] + 0.20, 1.10]
    camera.distance = args.camera_distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0  # Source collision capsules are not visual surfaces.
    options.sitegroup[:] = 0
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small_font = ImageFont.load_default()

    def pose_image(pose, title, subtitle, status=""):
        data.qpos[:3], data.qpos[3:7], data.qpos[joint_qpos] = pose
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=options)
        frame = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, args.width, 61), fill=(20, 27, 35))
        draw.text((12, 7), title, font=font, fill=(255, 255, 255))
        draw.text((12, 33), subtitle, font=small_font, fill=(215, 225, 235))
        if status:
            draw.rectangle((0, args.height - 35, args.width, args.height), fill=(117, 31, 31))
            draw.text((12, args.height - 28), status, font=small_font, fill="white")
        return frame

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(args.ffmpeg)
    comparison = args.output_dir / "policy_vs_reference.mp4"
    terminal_time = float(trace["times"][-1])
    first_time = float(trace["times"][0])
    playback_duration = (terminal_time - first_time) * args.slowdown + args.hold_seconds
    frame_count = max(1, int(np.ceil(playback_duration * args.fps)) + 1)
    writer = VideoWriter(comparison, 2 * args.width, args.height, args.fps, ffmpeg)
    overview = []
    sample_frames = set(np.linspace(0, max(0, frame_count - int(args.hold_seconds * args.fps) - 1), 4).astype(int))
    result = trace["metadata"].get("result", {})
    failure = result.get("physical_diagnostics", {}).get("failure_reason", "episode ended")
    start_phase = float(result.get("start_phase", trace["phase"][0]))
    is_partial = bool(result.get("evaluation_is_partial", start_phase > 0.00001))
    title = (f"POLICY / PARTIAL INIT AT {100 * start_phase:.1f}%" if is_partial
             else "RECORDED POLICY / PhysX")
    diagnostic_continuation = trace["metadata"].get("root_error_termination_enabled") is False
    if diagnostic_continuation:
        title = "DIAGNOSTIC / ROOT ERROR STOP DISABLED"
    fallback_note = " [held joints reconstructed]" if trace["missing_joints"] else " [all joints recorded]"
    try:
        for index in range(frame_count):
            time = min(first_time + index / args.fps / args.slowdown, terminal_time)
            phase = float(np.interp(time, trace["times"], trace["phase"]))
            actual_pose = sample_pose(trace["times"], recorded_root, trace["quat"], trace["joints"], time)
            reference_pose = sample_pose(reference_times, ref_root, ref_quat, ref_joints, phase * duration)
            freeze = time >= terminal_time - 1e-9
            status = f"TERMINATED / FROZEN: {failure}" if freeze and trace["terminal"] else ""
            if freeze and result.get("segment_completed"):
                status = ("SEGMENT END / PARTIAL INIT: not a complete rollout" if is_partial
                          else "REFERENCE END / COMPLETE")
            left = pose_image(actual_pose, title, f"episode t={time:.2f}s   phase={phase * 100:.1f}%{fallback_note}", status)
            right = pose_image(reference_pose, "REFERENCE / same phase", f"target t={phase * duration:.2f}s / {duration:.2f}s   playback {args.slowdown:g}x slower")
            combined = Image.new("RGB", (args.width * 2, args.height))
            combined.paste(left, (0, 0))
            combined.paste(right, (args.width, 0))
            writer.append(np.asarray(combined))
            if index in sample_frames:
                overview.append(combined.copy())
        combined.save(args.output_dir / "terminal_frame.png")
    finally:
        writer.close()
    contact_sheet = Image.new("RGB", (args.width * 2, args.height * len(overview)))
    for index, frame in enumerate(overview):
        contact_sheet.paste(frame, (0, index * args.height))
    contact_sheet.save(args.output_dir / "overview.png")

    reference_path = args.output_dir / "reference_only.mp4"
    if not args.skip_reference:
        writer = VideoWriter(reference_path, args.width, args.height, args.fps, ffmpeg)
        try:
            for index in range(int(np.ceil(duration * args.fps)) + 1):
                time = min(index / args.fps, duration)
                pose = sample_pose(reference_times, ref_root, ref_quat, ref_joints, time)
                frame = pose_image(pose, "REFERENCE TARGET / kinematic playback", f"t={time:.2f}s / {duration:.2f}s   phase={100 * time / duration:.1f}%   real time")
                writer.append(np.asarray(frame))
        finally:
            writer.close()
    renderer.close()
    manifest = dict(
        schema=1, method="offline_kinematic_render_of_recorded_physx_rollout",
        policy_executed_by_this_script=False, physics_stepped_by_this_script=False,
        trace=str(args.trace.resolve()), trace_sha256=sha256(args.trace),
        motion=str(args.motion.resolve()), motion_sha256=sha256(args.motion),
        model=str(model_path.resolve()), model_sha256=sha256(model_path),
        scene=str(scene_path.resolve()), scene_fingerprint=scene.fingerprint,
        checkpoint=trace["metadata"].get("checkpoint"),
        checkpoint_sha256=trace["metadata"].get("checkpoint_sha256"),
        actual_recorded_joint_count=spec.num_joints - len(trace["missing_joints"]),
        held_joints_reconstructed_at_spec_nominal=trace["missing_joints"],
        root_frame_conversion="recorded rebased world XY + source_first_root_xy = original scene world",
        quaternion_convention="world_from_link_wxyz", samples=len(trace["times"]),
        initial_recorded_state_included=trace["initial_state_included"],
        evaluation_is_partial=is_partial, initialization_phase=start_phase,
        full_rollout_completed=bool(result.get("completed", False)),
        segment_completed=bool(result.get("segment_completed", False)),
        first_sample_time_s=first_time, terminal_time_s=terminal_time,
        terminal_phase=float(trace["phase"][-1]), terminated_or_truncated=trace["terminal"],
        failure_reason=failure, pose_interpolation="linear joints/root and shortest-path normalized quaternion blend",
        reset_or_loop_after_termination=False, terminal_hold_seconds=args.hold_seconds,
        comparison_slowdown=args.slowdown, fps=args.fps,
        diagnostic_only=bool(trace["metadata"].get("diagnostic_only", False)),
        root_error_termination_enabled=not diagnostic_continuation,
        eligible_for_skill_acceptance=trace["metadata"].get("eligible_for_skill_acceptance"),
        reference_only_playback="kinematic target, not policy control; full clip at real time",
        software_renderer=gl_renderer, rendering_backend=args.backend,
        comparison=str(comparison.resolve()), terminal_frame=str((args.output_dir / "terminal_frame.png").resolve()),
        overview=str((args.output_dir / "overview.png").resolve()),
        reference_only=None if args.skip_reference else str(reference_path.resolve()),
        camera=dict(lookat=camera.lookat.tolist(), distance=camera.distance,
                    azimuth=camera.azimuth, elevation=camera.elevation),
    )
    (args.output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--width", type=int, default=640, help="width of each comparison panel")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--slowdown", type=float, default=2)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--backend", choices=["software-egl", "configured"], default="software-egl")
    parser.add_argument("--camera-distance", type=float, default=3.6)
    parser.add_argument("--azimuth", type=float, default=145)
    parser.add_argument("--elevation", type=float, default=-15)
    args = parser.parse_args()
    if (args.width < 128 or args.height < 128 or args.width % 2 or args.height % 2
            or args.fps < 1 or args.slowdown <= 0 or args.hold_seconds < 0):
        parser.error("dimensions must be even and >=128; fps/slowdown positive; hold nonnegative")
    render(args)


if __name__ == "__main__":
    main()
