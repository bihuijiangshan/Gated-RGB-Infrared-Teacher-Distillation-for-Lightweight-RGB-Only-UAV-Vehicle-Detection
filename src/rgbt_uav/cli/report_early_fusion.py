from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.early_fusion_report import (
    write_early_fusion_report,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a reproducible early-fusion evidence report"
    )
    parser.add_argument(
        "--completion",
        type=Path,
        required=True,
        help="Path to completion.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination report JSON path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)

    destination = write_early_fusion_report(
        args.completion,
        args.output,
    )

    print(destination)


if __name__ == "__main__":
    main()