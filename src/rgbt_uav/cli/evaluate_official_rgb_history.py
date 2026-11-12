"""Evaluate epoch history and select the best COCO mAP checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rgbt_uav.cli.evaluate_official_rgb_student import (
    main as evaluate_checkpoint,
)
from rgbt_uav.training.official_rgb_checkpoint_selection import (
    select_best_map_checkpoint,
)


class OfficialRGBHistoryEvaluationError(RuntimeError):
    """Raised when epoch-history evaluation cannot complete."""


def _student_model_from_completion(
    run_root: Path,
) -> str:
    completion_path = run_root / "completion.json"

    if not completion_path.is_file():
        raise OfficialRGBHistoryEvaluationError(
            "completion.json must be a regular file"
        )

    try:
        payload: Any = json.loads(
            completion_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise OfficialRGBHistoryEvaluationError(
            "failed to read completion.json"
        ) from exc

    if not isinstance(payload, dict):
        raise OfficialRGBHistoryEvaluationError(
            "completion root must be a mapping"
        )

    config = payload.get("config")
    if not isinstance(config, dict):
        raise OfficialRGBHistoryEvaluationError(
            "completion config must be a mapping"
        )

    model = config.get("student_model")
    if not isinstance(model, str) or not model.strip():
        raise OfficialRGBHistoryEvaluationError(
            "completion student_model must be a non-empty string"
        )

    return model

def evaluate_official_rgb_history(
    run_root: Path,
    *,
    data_root: Path,
    batch: int,
    workers: int,
    device: str,
    score_threshold: float,
    iou_threshold: float,
    max_det: int,
    model_name: str | None = None,
    evaluation_fn: Callable[..., int] = evaluate_checkpoint,
    selection_fn: Callable[..., dict[str, object]] = (select_best_map_checkpoint),
) -> dict[str, object]:
    """Evaluate every retained epoch and select maximum mAP50-95."""

    root = Path(run_root)
    epoch_root = root / "epochs"

    resolved_model = (
        _student_model_from_completion(root)
        if model_name is None
        else model_name
    )

    if (
        not isinstance(resolved_model, str)
        or not resolved_model.strip()
    ):
        raise OfficialRGBHistoryEvaluationError(
            "model_name must be a non-empty string"
        )

    if not epoch_root.is_dir():
        raise OfficialRGBHistoryEvaluationError(
            f"epoch directory does not exist: {epoch_root}"
        )

    checkpoints = sorted(
        epoch_root.glob("epoch-*.pt")
    )
    if not checkpoints:
        raise OfficialRGBHistoryEvaluationError(
            "epoch directory contains no checkpoints"
        )

    metrics_root = root / "epoch_metrics"
    metrics_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_paths: list[Path] = []

    for checkpoint in checkpoints:
        metrics_path = (
            metrics_root
            / f"{checkpoint.stem}.json"
        )

        exit_code = evaluation_fn(
            [
                "--checkpoint",
                str(checkpoint),
                "--model",
                resolved_model,
                "--output",
                str(metrics_path),
                "--data-root",
                str(data_root),
                "--batch",
                str(batch),
                "--workers",
                str(workers),
                "--device",
                device,
                "--score-threshold",
                str(score_threshold),
                "--iou-threshold",
                str(iou_threshold),
                "--max-det",
                str(max_det),
            ]
        )
        if exit_code != 0:
            raise OfficialRGBHistoryEvaluationError(
                f"evaluation failed for {checkpoint}"
            )
        if not metrics_path.is_file():
            raise OfficialRGBHistoryEvaluationError(
                f"evaluation did not write metrics: {metrics_path}"
            )

        metrics_paths.append(metrics_path)

    result = selection_fn(
        metrics_paths,
        destination=root / "best_map50_95.pt",
        evidence_path=root / "best_map50_95.json",
    )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all retained official RGB student "
            "epochs and select best mAP50-95"
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override student model YAML; by default "
            "read it from completion.json"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "data/processed/yolo_smoke_5pct"
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        default="0",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    result: dict[str, Any] = evaluate_official_rgb_history(
        args.run_root,
        data_root=args.data_root,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_det=args.max_det,
        model_name=args.model,
    )

    printable = {
        key: (
            str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in result.items()
    }
    print(
        json.dumps(
            printable,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())