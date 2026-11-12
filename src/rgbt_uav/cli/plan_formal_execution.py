"""Print the dependency-aware formal execution plan without training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.formal_execution_plan import (
    load_formal_execution_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the formal dependency plan; this command never trains"
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--resource-budget", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_formal_execution_plan(args.matrix, args.resource_budget)
    print(json.dumps(plan.to_metadata(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
