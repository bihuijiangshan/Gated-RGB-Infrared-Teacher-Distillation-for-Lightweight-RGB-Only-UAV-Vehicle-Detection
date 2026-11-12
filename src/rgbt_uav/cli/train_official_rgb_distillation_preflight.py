"""CLI for official RGB-T teacher to RGB student distillation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.official_rgb_distillation_config import (
    OfficialRGBDistillationConfig,
)
from rgbt_uav.training.official_rgb_distillation_runner import (
    run_official_rgb_distillation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume official RGB-T to RGB "
            "student distillation"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the official RGB distillation YAML config",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)

    config = OfficialRGBDistillationConfig.from_yaml(
        args.config
    )
    result = run_official_rgb_distillation(config)

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
                "metrics": result.metrics,
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