from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.resumable import ResumableTrainingConfig, run_resumable_training


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed resumable training preflight")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    result = run_resumable_training(ResumableTrainingConfig.from_yaml(args.config))
    print(
        json.dumps(
            {
                "best_weights": str(result.best_weights),
                "last_weights": str(result.last_weights),
                "pause_checkpoint": str(result.pause_checkpoint),
                "state_path": str(result.state_path),
                "completion_path": str(result.completion_path),
                "metrics": result.metrics,
                "completed_epochs": result.completed_epochs,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
