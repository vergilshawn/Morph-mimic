"""Stage 1 training. Run as python -m morph_mimic.scripts.train_stage1."""

from __future__ import annotations

import argparse

from morph_mimic.scripts.common import add_common_args, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    run_training(args, 1)


if __name__ == "__main__":
    main()
