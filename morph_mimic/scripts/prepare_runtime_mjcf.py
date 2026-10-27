"""Prepare a hashed explicit MJCF cache before launching Isaac Sim."""

import argparse
from pathlib import Path

from morph_mimic.utils.explicit_mjcf import export_explicit_mjcf
from morph_mimic.utils.specs import PROJECT_ROOT, RobotSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["g1", "mojia"], required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    spec = RobotSpec.load(args.robot)
    source = spec.runtime_assets.get("mjcf")
    if not source:
        parser.error(f"{args.robot} has no reference MJCF")
    output, _ = export_explicit_mjcf(PROJECT_ROOT / source,
                                    args.output_dir or PROJECT_ROOT / ".cache/mjcf" / args.robot)
    print(output)


if __name__ == "__main__":
    main()
