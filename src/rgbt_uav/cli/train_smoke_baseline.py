from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.smoke import SmokeTrainingConfig, run_smoke_training


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a verifiable one-epoch YOLO smoke baseline")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    result = run_smoke_training(SmokeTrainingConfig.from_yaml(args.config))
    print(
        json.dumps(
            {
                "best_weights": str(result.best_weights),
                "completion_path": str(result.completion_path),
                "last_weights": str(result.last_weights),
                "metrics": result.metrics,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
