"""CLI for inspecting the bounded formal calibration plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.formal_calibration_plan import (
    load_formal_calibration_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and print the bounded formal calibration plan"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_formal_calibration_plan(args.plan, args.matrix)
    print(json.dumps(plan.to_metadata(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
