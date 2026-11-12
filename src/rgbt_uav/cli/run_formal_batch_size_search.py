"""Explicitly gated CLI for bounded formal batch-size search."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rgbt_uav.cli.run_formal_calibration import _git_metadata
from rgbt_uav.training.formal_batch_size_search import (
    load_formal_batch_size_search_plan,
    run_formal_batch_size_search,
)
from rgbt_uav.training.formal_calibration_registry import (
    build_production_calibration_workload_factory,
)
from rgbt_uav.training.formal_experiment_audit import (
    audit_formal_experiment_matrix,
)


class FormalBatchSizeSearchCLIError(RuntimeError):
    """Raised when an execution gate cannot be satisfied."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or execute bounded formal batch-size search"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    audit_fn: Callable[..., Any] = audit_formal_experiment_matrix,
    factory_builder: Callable[..., Any] = (
        build_production_calibration_workload_factory
    ),
    runner: Callable[..., dict[str, Any]] = run_formal_batch_size_search,
    git_metadata_fn: Callable[[], dict[str, object]] = _git_metadata,
) -> int:
    args = build_parser().parse_args(argv)
    plan = load_formal_batch_size_search_plan(args.plan, args.matrix)
    if not args.execute:
        print(json.dumps(plan.to_metadata(), sort_keys=True))
        return 0

    audit = audit_fn(args.protocol, args.matrix)
    if audit.protocol_id != plan.protocol_id:
        raise FormalBatchSizeSearchCLIError(
            "formal audit protocol_id does not match batch-size search plan"
        )
    result = runner(
        plan,
        workload_factory=factory_builder(),
        git_metadata=git_metadata_fn(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
