"""Isaac Sim app launcher.

Must be called BEFORE any other isaaclab import in entry scripts:
``pxr`` (USD bindings) only becomes importable after the SimulationApp
is started. This mirrors the pattern of Isaac Lab's own examples.

Usage (first lines of a script):

    import argparse
    parser = argparse.ArgumentParser()
    ...  # parse args (plain argparse only, no isaaclab imports)
    from morph_mimic.launcher import launch
    simulation_app = launch(headless=args.headless)
    from morph_mimic.scripts.common import ...  # isaaclab imports OK now
"""

from __future__ import annotations


def launch(headless: bool = True, device: str = "cuda:0"):
    from isaaclab.app import AppLauncher

    launcher = AppLauncher({"headless": headless, "device": device})
    return launcher.app


def close(app):
    """Preserve validation output and error status before Kit fast shutdown."""
    import sys
    import traceback

    kind, error, tb = sys.exc_info()
    if kind is not None:
        traceback.print_exception(kind, error, tb)
        status = (0 if error.code is None else error.code if isinstance(error.code, int) else 1) if isinstance(error, SystemExit) else 1
        app.app.post_quit(status)
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()
