"""Unified entry point for optional diagnostics, evaluation and playback.

Every command runs in a fresh interpreter using the current Python environment.
Arguments following the command are forwarded unchanged; request its own help
with ``python -m morph_mimic.scripts.inspect COMMAND --help``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMANDS = {
    "assets": ("morph_mimic.diagnostics.audit_assets", "migrated asset hashes and motion arrays"),
    "runtime": ("morph_mimic.diagnostics.audit_runtime_reference", "Isaac model properties and FK (GPU)"),
    "held": ("morph_mimic.diagnostics.audit_held_joint_dynamics", "held-joint physical constraints (GPU)"),
    "scene": ("morph_mimic.diagnostics.audit_scene_replication", "scene copies, sensing and reset isolation (GPU)"),
    "frames": ("morph_mimic.diagnostics.audit_tracking_frames", "world/reference/local frame and joint errors"),
    "self-collision": ("morph_mimic.diagnostics.check_reference_self_collision", "reference self-collision geometry (MuJoCo)"),
    "segments": ("morph_mimic.diagnostics.diagnose_policy_segments", "isolated policy phase sweeps (GPU)"),
    "box": ("morph_mimic.diagnostics.evaluate_box_task", "recorded box-top completion and support"),
    "pd": ("morph_mimic.diagnostics.validate_scene_reference", "free-root reference PD replay (MuJoCo)"),
    "record": ("morph_mimic.diagnostics.record_policy_rollout", "record actual policy endpoints (GPU)"),
    "render": ("morph_mimic.diagnostics.render_tracking_trajectory", "offline policy/reference video (MuJoCo/EGL)"),
    "play": ("morph_mimic.diagnostics.play", "interactive policy playback (GPU)"),
    "contacts": ("morph_mimic.scripts.evaluate_contact_trajectory", "recorded contact events and coverage"),
    "evaluate": ("morph_mimic.scripts.evaluate", "strict full-reference policy evaluation (GPU)"),
}


def parser():
    command_help = "\n".join(f"  {name:15} {description}" for name, (_, description) in COMMANDS.items())
    result = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Commands:\n{command_help}\n\nNo GPU job runs unless its command is explicitly selected.",
    )
    result.add_argument("command", choices=COMMANDS, nargs="?", help="diagnostic or playback command")
    result.add_argument("arguments", nargs=argparse.REMAINDER, help="arguments for the selected command")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command is None:
        parser().print_help()
        return 0
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    module, _ = COMMANDS[args.command]
    completed = subprocess.run([sys.executable, "-m", module, *arguments], cwd=PROJECT_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
