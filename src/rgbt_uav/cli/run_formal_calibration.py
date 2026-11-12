"""Explicitly gated CLI for bounded formal runtime calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rgbt_uav.training.formal_calibration_plan import (
    load_formal_calibration_plan,
)
from rgbt_uav.training.formal_calibration_registry import (
    build_production_calibration_workload_factory,
)
from rgbt_uav.training.formal_calibration_runner import run_formal_calibration
from rgbt_uav.training.formal_experiment_audit import (
    audit_formal_experiment_matrix,
)


class FormalCalibrationCLIError(RuntimeError):
    """Raised when calibration CLI preconditions are not satisfied."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or explicitly execute bounded formal training calibration"
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the bounded training workloads; omitted means plan-only",
    )
    return parser


def _git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FormalCalibrationCLIError("failed to read git metadata") from exc
    if status:
        raise FormalCalibrationCLIError(
            "tracked working tree must be clean before calibration"
        )
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise FormalCalibrationCLIError("git commit must be a full SHA-1")
    return {"commit": commit, "tracked_worktree_clean": True}


def main(
    argv: Sequence[str] | None = None,
    *,
    audit_fn: Callable[..., Any] = audit_formal_experiment_matrix,
    factory_builder: Callable[..., Any] = (
        build_production_calibration_workload_factory
    ),
    runner: Callable[..., dict[str, Any]] = run_formal_calibration,
    git_metadata_fn: Callable[[], dict[str, object]] = _git_metadata,
) -> int:
    args = build_parser().parse_args(argv)
    plan = load_formal_calibration_plan(args.plan, args.matrix)

    if not args.execute:
        print(json.dumps(plan.to_metadata(), sort_keys=True))
        return 0

    audit = audit_fn(args.protocol, args.matrix)
    if audit.protocol_id != plan.protocol_id:
        raise FormalCalibrationCLIError(
            "formal audit protocol_id does not match calibration plan"
        )
    git_metadata = git_metadata_fn()
    workload_factory = factory_builder()
    result = runner(
        plan,
        workload_factory=workload_factory,
        git_metadata=git_metadata,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
