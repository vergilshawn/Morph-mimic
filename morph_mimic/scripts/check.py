"""Run ordered project checks; no simulator or training starts by default.

Default: basic regression tests -> local robot assets -> selected motion/contact/scene
contracts. Use --gpu to append a short Isaac smoke check, --gpu-only to run just
that check, or --checkpoint for a strict saved-policy skill check. Reports and logs live
under runs/_reports/checks/<UTC timestamp>; latest.json points to the last run.
Run with the Python environment containing the dependencies you want to check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
CPU_STAGES = ("regression", "assets", "reference")
DEFAULT_MOTION = ROOT / "datasets/g1_box_climb_contact_v3"


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--only", choices=CPU_STAGES, nargs="+", help="select CPU checks; canonical order is preserved")
    gpu = result.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", action="store_true", help="append explicit GPU smoke/evaluation")
    gpu.add_argument("--gpu-only", action="store_true", help="only run explicit GPU smoke/evaluation")
    result.add_argument("--robot", choices=("g1", "mojia"), default="g1")
    result.add_argument("--motion-dir", type=Path, default=DEFAULT_MOTION)
    result.add_argument("--clip", help="one motion stem; default selects all clips in motion-dir")
    result.add_argument("--out", type=Path, help="new JSON report path; existing reports are never overwritten")
    result.add_argument("--suite", choices=("basic", "reference", "planner", "auxiliary", "diagnostics", "physics", "all"),
                        nargs="+", default=["basic"],
                        help="regression suites: basic (default), reference preparation, Stage 2/3 planner, "
                             "auxiliary standing/old supervision, diagnostics, CPU MuJoCo physics, or all")
    result.add_argument("--pytest-args", default="", help="extra pytest arguments, e.g. --pytest-args='-k contact -v'")
    result.add_argument("--deep", action="store_true", help="also verify every manifest file, including all migrated motions")
    result.add_argument("--self-collision", action="store_true", help="append CPU MuJoCo geometry audit; requires one bound clip")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--num-envs", type=int, default=32, help="GPU smoke environments (1..64)")
    result.add_argument("--steps", type=int, default=20, help="GPU smoke control steps (1..1000)")
    result.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    result.add_argument("--checkpoint", type=Path,
                        help="check saved policy against its skill gates; failed skill exits 1; never train")
    result.add_argument("--checkpoint-kind", choices=("acp", "standing", "teacher-v2", "teacher-v3"), default="acp")
    result.add_argument("--_stage", choices=(*CPU_STAGES[1:], "self-collision", "gpu"), help=argparse.SUPPRESS)
    result.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    return result


def stage_order(args):
    if args.gpu_only:
        return ["gpu"]
    selected = set(args.only or CPU_STAGES)
    stages = [stage for stage in CPU_STAGES if stage in selected]
    if args.self_collision:
        stages.append("self-collision")
    if args.gpu:
        stages.append("gpu")
    return stages


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _asset_check(args):
    import numpy as np
    from morph_mimic.utils.specs import RobotSpec

    spec = RobotSpec.load(args.robot)
    if len(spec.joint_names) != spec.num_joints or len(set(spec.joint_names)) != spec.num_joints:
        raise ValueError("spec joint names/count are inconsistent")
    if (set(spec.controlled_joint_names) & set(spec.held_joint_names)
            or set(spec.controlled_joint_names) | set(spec.held_joint_names) != set(spec.joint_names)):
        raise ValueError("controlled and held joints must partition the full joint set")
    for field in ("default_pose", "stiffness", "damping", "effort_limits", "armature", "action_scale",
                  "pos_limits_lower", "pos_limits_upper", "vel_limits"):
        value = np.asarray(getattr(spec, field))
        if value.shape != (spec.num_joints,) or not np.isfinite(value).all():
            raise ValueError(f"spec {field} must have {spec.num_joints} finite values")
    if (np.any(np.asarray(spec.pos_limits_lower) > spec.default_pose)
            or np.any(np.asarray(spec.default_pose) > spec.pos_limits_upper)
            or np.any(np.asarray(spec.stiffness) <= 0) or np.any(np.asarray(spec.effort_limits) <= 0)
            or np.any(np.asarray(spec.damping) < 0) or np.any(np.asarray(spec.armature) < 0)):
        raise ValueError("invalid nominal pose, gains, armature or effort limits")
    ratio = spec.control_dt / spec.physics_dt
    if spec.physics_dt <= 0 or not np.isclose(ratio, round(ratio)) or ratio < 1:
        raise ValueError("control_dt must be a positive integer multiple of physics_dt")
    path = (ROOT / spec.runtime_assets["mjcf"]).resolve()
    tree = ET.parse(path)
    if tree.findall(".//include"):
        raise ValueError("included MJCF requires a dedicated model audit; lightweight check cannot certify it")
    joints = [node.get("name") for node in tree.findall(".//worldbody//joint") if node.get("type", "hinge") != "free"]
    bodies = [node.get("name") for node in tree.findall(".//worldbody//body")]
    if len(joints) != spec.num_joints or set(joints) != set(spec.joint_names):
        raise ValueError("MJCF scalar joints differ from the complete named spec")
    spec.resolve_contact_bodies(bodies)
    compiler = tree.find("compiler")
    compiler = {} if compiler is None else compiler.attrib
    dependencies = {path}
    for kind, directory in (("mesh", "meshdir"), ("texture", "texturedir")):
        for node in tree.findall(f"./asset/{kind}"):
            if node.get("file"):
                dependencies.add((path.parent / compiler.get(directory, "") / node.get("file")).resolve())
    asset_root = ROOT / "assets" / args.robot
    manifest = json.loads((asset_root / "manifest.json").read_text())
    expected = dict(manifest.get("files", {}))
    # Mojia's upstream manifest retains the pre-migration mesh paths.
    for original, digest in manifest.get("asset_dependencies", {}).items():
        try:
            relocated = str(Path(original).relative_to("assets/robots/mojia"))
        except ValueError:
            continue
        expected.setdefault(relocated, digest)
        expected.setdefault("robot/" + relocated, digest)
    if args.deep:
        dependencies.update((asset_root / name).resolve() for name in expected)
    hashes, unverified = {}, []
    for dependency in sorted(dependencies):
        if not dependency.is_file():
            raise FileNotFoundError(f"missing runtime dependency: {dependency}")
        relative = str(dependency.relative_to(asset_root))
        hashes[relative] = _hash(dependency)
        if relative not in expected:
            unverified.append(relative)
        elif hashes[relative] != expected[relative]:
            raise ValueError(f"asset SHA256 differs from migration manifest: {relative}")
    return dict(robot=args.robot, joints=len(joints), controlled=spec.n_controlled, bodies=len(bodies),
                checked_files=len(hashes), deep=args.deep, sha256=hashes,
                spec_sha256=_hash(asset_root / "robot_spec.json"), physical_runtime_validated=False,
                provenance_hashes_unavailable=unverified,
                warnings=(["Some derived runtime files have no historical manifest hash; current hashes are recorded"]
                          if unverified else []))


def _reference_check(args):
    import numpy as np
    from morph_mimic.utils.motion import MotionDataset
    from morph_mimic.utils.specs import RobotSpec

    spec = RobotSpec.load(args.robot)
    dataset = MotionDataset(spec, args.motion_dir, device="cpu", clip_names=[args.clip] if args.clip else None)
    clips = []
    for name in dataset.clip_names:
        clip = dataset.clips[name]
        if not np.isclose(clip.fps * spec.control_dt, 1.):
            raise ValueError(f"{name}: reference frequency differs from control frequency")
        if name not in dataset.contact_flags:
            raise ValueError(f"{name}: missing full contact labels")
        label = args.motion_dir / f"{name}_contacts.npz"
        with np.load(label, allow_pickle=False) as arrays:
            if "source_sha256" not in arrays or "body_names" not in arrays:
                raise ValueError(f"{name}: labels need source SHA256 and explicit anatomical names")
            method = str(arrays["method"].item()) if "method" in arrays else "unspecified"
        if dataset.contact_target_frames[name] != "reference_query_body_yaw_absolute_z":
            raise ValueError(f"{name}: unverified contact target coordinate frame")
        scene = dataset.scene_bindings.get(name)
        clips.append(dict(name=name, frames=clip.num_frames, duration_s=clip.duration_s, fps=clip.fps,
                          motion_sha256=_hash(args.motion_dir / f"{name}.npz"), contacts_sha256=_hash(label),
                          contact_method=method, scene_fingerprint=scene.fingerprint if scene else None,
                          contact_frame=dataset.contact_target_frames[name]))
    return dict(robot=args.robot, motion_dir=str(args.motion_dir), clips=clips,
                note="Schema/hash/coordinate contracts only; geometric labels do not prove physical task success")


def _self_collision_check(args):
    from morph_mimic.diagnostics.check_reference_self_collision import check_reference

    if not args.clip:
        raise ValueError("--self-collision requires --clip for one bound reference")
    report = check_reference(args.motion_dir / f"{args.clip}.npz", args.motion_dir / f"{args.clip}.scene.json", robot=args.robot)
    if not report["accepted"]:
        raise ValueError(f"reference self-collision audit failed: {report['retarget_reference']}")
    return report


def checkpoint_command(args, directory):
    common = ["--resume", str(args.checkpoint), "--motion-dir", str(args.motion_dir),
              "--device", args.device, "--out", str(directory / "policy_evaluation.json"),
              "--trajectory", str(directory / "policy_trajectory.npz")]
    if args.clip:
        common += ["--clips", args.clip]
    if args.checkpoint_kind == "acp":
        return [sys.executable, "-m", "morph_mimic.scripts.evaluate", "--robot", args.robot,
                "--stage", str(args.stage), *common]
    module = {"standing": "train_standing_baseline", "teacher-v2": "train_reference_teacher_v2",
              "teacher-v3": "train_reference_teacher_v3"}[args.checkpoint_kind]
    return [sys.executable, "-m", f"morph_mimic.scripts.{module}", "evaluate", *common]


def policy_evaluation_result(args, report):
    """Reject incomplete evaluations and report policy acceptance separately."""
    motions = report.get("motions", [])
    if (report.get("contract_verified") is not True
            or report.get("evaluation_is_full") is not True
            or report.get("evaluation_is_partial") is not False
            or report.get("start_phase") != 0.
            or report.get("expected_clips") != 1 or report.get("evaluated_clips") != 1
            or len(motions) != 1):
        raise ValueError("saved policy requires one complete phase-zero evaluation with verified contracts")
    motion = motions[0]
    if not args.clip or motion.get("clip") != args.clip:
        raise ValueError("saved policy report must match the one explicitly requested clip")
    if (motion.get("contract_verified") is not True
            or motion.get("start_phase") != 0.
            or motion.get("evaluation_is_partial") is not False
            or any(motion.get(key) is not False for key in ("budget_limited", "interrupted"))):
        raise ValueError("saved policy evaluation is partial, interrupted, budget-limited or unverified")
    if any(type(motion.get(key)) is not bool for key in ("completed", "failure", "terminated")):
        raise ValueError("saved policy evaluation lacks explicit completion and failure flags")
    if (motion["failure"] != motion["terminated"]
            or motion["completed"] == motion["failure"]):
        raise ValueError("saved policy evaluation must end in either reference completion or physical failure")
    if args.checkpoint_kind == "acp":
        from morph_mimic.scripts.train_until_validated import acceptance

        passed, reasons = acceptance(report)
    else:
        prefix = "standing" if args.checkpoint_kind == "standing" else "teacher"
        passed, reasons = report.get(f"{prefix}_passed"), report.get(f"{prefix}_failure_reasons")
        if type(passed) is not bool or not isinstance(reasons, list):
            raise ValueError(f"saved policy evaluation lacks explicit {prefix} acceptance results")
        if passed == bool(reasons):
            raise ValueError(f"saved policy evaluation has inconsistent {prefix} acceptance results")
    return dict(evaluation_completed=True, policy_passed=passed, policy_failure_reasons=reasons,
                completion_rate=report.get("completion_rate"), task_passed=report.get("task_passed"),
                note="Evaluation finished normally; policy_passed is the separate skill acceptance result")


def _gpu_check(args):
    if args.checkpoint:
        command = checkpoint_command(args, args._result.parent)
        subprocess.run(command, cwd=ROOT, check=True)
        evaluation = args._result.parent / "policy_evaluation.json"
        report = json.loads(evaluation.read_text())
        return dict(mode="saved_policy_evaluation", report=str(evaluation), checkpoint=str(args.checkpoint),
                    **policy_evaluation_result(args, report))
    from morph_mimic.launcher import launch, close

    app = launch(headless=True, device=args.device)
    env = None
    try:
        import torch
        from morph_mimic.networks.acp import ACPPlanner
        from morph_mimic.networks.tracker import Critic, TrackerActor
        from morph_mimic.scripts.common import build_env

        torch.manual_seed(0)
        env, spec = build_env(args.stage, args.robot, args.num_envs,
            ref_env_ratio=1. if args.stage < 3 else .25, device=args.device, use_usd=False,
            motion_dir=str(args.motion_dir), clips=[args.clip] if args.clip else None,
            training_profile="nominal_pretrain", evaluation=False)
        for sensor in getattr(env, "scene_contact_sensors", []):
            if sensor.data.force_matrix_w.shape[-2] <= 0:
                raise ValueError("scene contact filter resolved no collision shapes")
        actor = TrackerActor().to(args.device).eval()
        critics = (Critic(353).to(args.device).eval(), Critic(298).to(args.device).eval())
        planner = ACPPlanner().to(args.device).eval()
        observation, _ = env.reset()
        with torch.no_grad():
            for _ in range(args.steps):
                points, heights, goal = env.get_planner_obs()
                if not all(torch.isfinite(value).all() for value in (points, heights, goal)):
                    raise ValueError("nonfinite planner scan/observation")
                _, occupancy, contacts = planner(points, heights, goal)
                if args.stage >= 2:
                    env.set_predicted_contacts(contacts, occupancy)
                    observation = env.refresh_observations()
                if observation["policy"].shape != (args.num_envs, 468):
                    raise ValueError(f"unexpected actor observation shape {observation['policy'].shape}")
                values = env.get_critic_obs()
                if tuple(value.shape[-1] for value in values) != (353, 298):
                    raise ValueError("unexpected critic observation dimensions")
                action = actor(observation["policy"])
                checked = [observation["policy"], action, *(critic(value) for critic, value in zip(critics, values))]
                observation, reward, _, _, _ = env.step(action)
                checked += [reward, env.robot.data.root_pos_w, env.robot.data.joint_pos]
                if not all(torch.isfinite(value).all() for value in checked):
                    raise ValueError("nonfinite observations, actions, values, rewards or physical state")
        report = dict(mode="random_network_physical_smoke", robot=args.robot, stage=args.stage,
                      environments=args.num_envs, controls=args.steps, physical_tracking_validated=False,
                      joints=env.robot.num_joints, bodies=env.robot.num_bodies,
                      tracker_rays=list(env.ray_caster.data.ray_hits_w.shape),
                      note="A finite short smoke rollout is not a learned skill or task validation")
        _write(args._result, dict(ok=True, details=report))  # Kit may fast-exit during close().
        return report
    finally:
        if env is not None:
            env.close()
        close(app)


def child_command(stage, args, result):
    if stage == "regression":
        command = [sys.executable, "-m", "pytest", "tests", "-q", "-rs"]
        for suite in args.suite:
            command += ["--suite", suite]
        return [*command, *shlex.split(args.pytest_args)]
    command = [sys.executable, "-m", "morph_mimic.scripts.check", "--_stage", stage,
               "--_result", str(result), "--robot", args.robot, "--motion-dir", str(args.motion_dir),
               "--device", args.device, "--stage", str(args.stage), "--num-envs", str(args.num_envs),
               "--steps", str(args.steps), "--checkpoint-kind", args.checkpoint_kind]
    if args.clip:
        command += ["--clip", args.clip]
    if args.deep:
        command += ["--deep"]
    if args.checkpoint:
        command += ["--checkpoint", str(args.checkpoint)]
    return command


def run(args, *, execute=subprocess.run):
    started = datetime.now(timezone.utc)
    directory = ROOT / "runs/_reports/checks" / started.strftime("%Y%m%dT%H%M%S.%fZ")
    report_path = args.out or directory / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, ok=False, state="running", started_utc=started.isoformat(),
                  python=sys.executable, cwd=str(ROOT), selected=stage_order(args), stages=[],
                  logs_dir=str(directory), test_suites=args.suite, no_training=True)
    _write(report_path, report)
    for index, stage in enumerate(report["selected"], 1):
        log = directory / f"{index:02d}_{stage}.log"
        detail = directory / f"{index:02d}_{stage}.json"
        command = child_command(stage, args, detail)
        print(f"[check {index}/{len(report['selected'])}] {stage} | log: {log}", flush=True)
        start = time.monotonic()
        try:
            with log.open("w") as stream:
                result = execute(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            code = result.returncode
            details = json.loads(detail.read_text()) if detail.exists() else None
            if code == 0 and stage != "regression" and (not details or details.get("ok") is not True):
                code = 1
        except (OSError, ValueError) as error:
            code, details = 1, dict(error=str(error))
        report["stages"].append(dict(name=stage, ok=code == 0, returncode=code,
                                     duration_s=round(time.monotonic() - start, 3),
                                     command=command, log=str(log), result=details))
        _write(report_path, report)
        print(f"[check] {stage}: {'PASS' if code == 0 else 'FAIL'}", flush=True)
        policy = (details or {}).get("details", {})
        if policy.get("mode") == "saved_policy_evaluation":
            print(f"[check] POLICY {'PASS' if policy['policy_passed'] else 'FAIL'} | "
                  f"{policy['policy_failure_reasons']}", flush=True)
        if code:
            print(log.read_text(errors="replace")[-6000:], flush=True)
            break
    report.update(ok=len(report["stages"]) == len(report["selected"]) and all(item["ok"] for item in report["stages"]),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    report["state"] = "passed" if report["ok"] else "failed"
    _write(report_path, report)
    _write(ROOT / "runs/_reports/checks/latest.json", dict(report=str(report_path.resolve()), **report))
    print(f"[check] {report['state'].upper()} | report: {report_path}", flush=True)
    return 0 if report["ok"] else 1


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if not 1 <= args.num_envs <= 64 or not 1 <= args.steps <= 1000:
        arguments.error("GPU smoke requires 1..64 environments and 1..1000 steps")
    if args.gpu_only and (args.only or args.self_collision):
        arguments.error("--gpu-only cannot be combined with CPU-stage selectors")
    if args.self_collision and not args.clip:
        arguments.error("--self-collision requires --clip for one bound reference")
    if args.checkpoint and not (args.gpu or args.gpu_only or args._stage == "gpu"):
        arguments.error("--checkpoint requires explicit --gpu or --gpu-only")
    args.motion_dir = args.motion_dir.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
        if not args.checkpoint.is_file():
            arguments.error(f"checkpoint not found: {args.checkpoint}")
        if not args.clip:
            arguments.error("--checkpoint requires --clip because recorded policy evaluation selects one reference")
        if args.checkpoint_kind != "acp" and (args.robot != "g1" or args.stage != 1):
            arguments.error("standing/teacher evaluation requires --robot g1 --stage 1 --clip <name>")
        if args.checkpoint_kind == "acp" and args.stage == 3:
            arguments.error("ACP checkpoint reference evaluation supports Stage 1/2 only")
    if args._stage:
        if args._result is None:
            arguments.error("internal stage requires --_result")
        functions = dict(assets=_asset_check, reference=_reference_check,
                         **{"self-collision": _self_collision_check, "gpu": _gpu_check})
        try:
            details = functions[args._stage](args)
            ok = details.get("mode") != "saved_policy_evaluation" or details.get("policy_passed") is True
            _write(args._result, dict(ok=ok, details=details))
            return 0 if ok else 1
        except Exception as error:
            traceback.print_exc()
            _write(args._result, dict(ok=False, error_type=type(error).__name__, error=str(error)))
            return 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
