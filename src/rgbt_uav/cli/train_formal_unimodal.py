"""CLI for planned or explicit formal unimodal training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.formal_unimodal_config import (
    FormalUnimodalConfig,
)
from rgbt_uav.training.formal_unimodal_plan import (
    build_formal_unimodal_training_plan,
)
from rgbt_uav.training.formal_unimodal_runner import (
    run_formal_unimodal_training,
)


def _portable_path(
    path: Path,
) -> str:
    return str(path).replace(
        "\\",
        "/",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the formal unimodal training parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect a formal unimodal training plan; "
            "the default is plan-only and does not train"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Formal RGB-only or IR-only YAML configuration",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Explicitly execute the 100-epoch formal run; "
            "without this flag only the plan is printed"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an explicitly interrupted formal run; "
            "requires --execute and verified checkpoint history"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Print a plan or explicitly run formal training."""

    parser = build_parser()
    args = parser.parse_args(
        argv
    )

    if args.resume and not args.execute:
        parser.error("--resume requires --execute")

    config = FormalUnimodalConfig.from_yaml(
        args.config
    )
    plan = (
        build_formal_unimodal_training_plan(
            config
        )
    )

    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "execute": False,
                    "experiment_id": (
                        plan.experiment_id
                    ),
                    "modality": (
                        plan.modality
                    ),
                    "model_source": (
                        plan.model_source
                    ),
                    "epochs": (
                        plan.train_arguments[
                            "epochs"
                        ]
                    ),
                    "selection_split": (
                        "val"
                    ),
                    "test_evaluated": False,
                    "output_root": (
                        _portable_path(
                            config.output_root
                        )
                    ),
                    "training_root": (
                        _portable_path(
                            plan.training_root
                        )
                    ),
                    "best_weights": (
                        _portable_path(
                            plan.best_weights
                        )
                    ),
                    "last_weights": (
                        _portable_path(
                            plan.last_weights
                        )
                    ),
                    "train_arguments": (
                        plan.train_arguments
                    ),
                },
                sort_keys=True,
            )
        )
        return 0

    if args.resume:
        result = run_formal_unimodal_training(
            config,
            resume=True,
        )
    else:
        result = run_formal_unimodal_training(
            config
        )

    print(
        json.dumps(
            {
                "status": "completed",
                "execute": True,
                "experiment_id": (
                    config.experiment_id
                ),
                "completed_epochs": list(
                    result.completed_epochs
                ),
                "epoch_weights_count": len(
                    result.epoch_weights
                ),
                "best_weights": (
                    _portable_path(
                        result.best_weights
                    )
                ),
                "last_weights": (
                    _portable_path(
                        result.last_weights
                    )
                ),
                "completion_path": (
                    _portable_path(
                        result.completion_path
                    )
                ),
            },
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
