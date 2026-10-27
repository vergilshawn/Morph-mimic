"""Stage 3 training. Run as python -m morph_mimic.scripts.train_stage3."""

from __future__ import annotations

import argparse

from morph_mimic.scripts.common import add_common_args, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--planner", default="acp", choices=["acp", "mlp"])
    parser.add_argument("--ref_env_ratio", type=float, default=0.3)
    args = parser.parse_args()
    run_training(args, 3)


if __name__ == "__main__":
    main()
