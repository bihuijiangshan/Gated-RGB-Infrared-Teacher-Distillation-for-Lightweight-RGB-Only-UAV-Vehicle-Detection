from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.distillation_config import DistillationConfig
from rgbt_uav.training.distillation_evidence import (
    validate_distillation_completion,
)
from rgbt_uav.training.distillation_runner import (
    run_distillation_preflight,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the teacher-student distillation preflight"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the distillation YAML configuration",
    )
    return parser


def _result_payload(result: object) -> dict[str, object]:
    try:
        best_weights = Path(result.best_weights)
        last_weights = Path(result.last_weights)
        state_path = Path(result.state_path)
        completion_path = Path(result.completion_path)
        metrics = result.metrics
        completed_epochs = result.completed_epochs
    except AttributeError as exc:
        raise RuntimeError(
            "runner result does not satisfy the expected contract"
        ) from exc

    if not isinstance(metrics, dict):
        raise RuntimeError("runner result metrics must be a mapping")

    return {
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "state_path": str(state_path),
        "completion_path": str(completion_path),
        "metrics": metrics,
        "completed_epochs": list(completed_epochs),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    config = DistillationConfig.from_yaml(args.config)
    result = run_distillation_preflight(config)
    validate_distillation_completion(result.completion_path)

    print(
        json.dumps(
            _result_payload(result),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()