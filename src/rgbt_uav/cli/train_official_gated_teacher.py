"""CLI for official gated dual-stream teacher training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.official_gated_teacher_config import (
    OfficialGatedTeacherConfig,
)
from rgbt_uav.training.official_gated_teacher_runner import (
    run_official_gated_teacher,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume official gated "
            "dual-stream teacher training"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to the official gated "
            "teacher YAML configuration"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(
        argv
    )

    config = (
        OfficialGatedTeacherConfig.from_yaml(
            args.config
        )
    )
    result = run_official_gated_teacher(
        config
    )

    print(
        json.dumps(
            {
                "best_weights": str(
                    result.best_weights
                ),
                "completed_epochs": list(
                    result.completed_epochs
                ),
                "completion_path": str(
                    result.completion_path
                ),
                "last_weights": str(
                    result.last_weights
                ),
                "metrics": (
                    result.metrics
                ),
                "state_path": str(
                    result.state_path
                ),
            },
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())