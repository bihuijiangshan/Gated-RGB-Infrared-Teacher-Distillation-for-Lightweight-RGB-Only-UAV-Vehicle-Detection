"""Command-line interface for official COCO small-object evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from rgbt_uav.evaluation.coco_dataset import DatasetContractError
from rgbt_uav.evaluation.coco_evidence import (
    EvaluationConfig,
    EvidencePublicationError,
    run_coco_small_evaluation,
)
from rgbt_uav.evaluation.coco_inference import InferenceContractError
from rgbt_uav.evaluation.coco_metrics import CocoMetricError


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate official COCO AP_small for one best checkpoint",
        allow_abbrev=False,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    return parser


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _load_published_metrics(metrics_path: Path, experiment_id: str) -> Mapping[str, object]:
    try:
        document = json.loads(
            metrics_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidencePublicationError(
            "published metrics are not strict readable JSON"
        ) from exc

    if not isinstance(document, Mapping):
        raise EvidencePublicationError("published metrics must be a JSON object")
    if document.get("status") != "passed":
        raise EvidencePublicationError("published metrics status is not passed")
    published_experiment = document.get("experiment_id")
    if not isinstance(published_experiment, str) or published_experiment != experiment_id:
        raise EvidencePublicationError("published metrics experiment_id does not match")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    print("Starting official COCO AP_small evaluation")
    config = EvaluationConfig(
        experiment_id=args.experiment_id,
        weights=args.weights,
        data=args.data,
        split=args.split,
        output=args.output,
    )
    try:
        metrics_path = run_coco_small_evaluation(config)
        _load_published_metrics(metrics_path, config.experiment_id)
    except (
        DatasetContractError,
        InferenceContractError,
        CocoMetricError,
        EvidencePublicationError,
    ) as exc:
        message = " ".join(str(exc).splitlines())
        print(f"Evaluation failed: {message}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "experiment_id": config.experiment_id,
                "metrics_path": str(metrics_path.absolute()),
                "status": "passed",
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
