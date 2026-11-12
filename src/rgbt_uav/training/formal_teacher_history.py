"""Validation-only checkpoint selection for formal teachers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from rgbt_uav.training.official_rgb_checkpoint_selection import (
    select_best_map_checkpoint,
)


class FormalTeacherHistoryError(RuntimeError):
    """Raised when retained teacher history is invalid."""


def evaluate_formal_teacher_history(
    run_root: Path,
    *,
    kind: str,
    data_root: Path,
    batch: int,
    workers: int,
    device: str,
    score_threshold: float,
    iou_threshold: float,
    max_det: int,
    evaluation_fn: Callable[..., dict[str, object]],
    selection_fn: Callable[..., dict[str, object]] = select_best_map_checkpoint,
) -> dict[str, object]:
    """Evaluate every retained epoch on val and select maximum mAP50-95."""

    root = Path(run_root)
    completion_path = root / "completion.json"
    if not completion_path.is_file() or completion_path.is_symlink():
        raise FormalTeacherHistoryError("completion.json must be a regular file")
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("status") != "completed":
        raise FormalTeacherHistoryError("completion evidence must be completed")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise FormalTeacherHistoryError("completion config must be a mapping")
    model_name = config.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise FormalTeacherHistoryError("completion model must be a string")
    epochs = payload.get("all_completed_epochs")
    if not isinstance(epochs, list) or not epochs:
        raise FormalTeacherHistoryError("completed epochs must be a list")
    expected_epochs = config.get('epochs')
    if (
        isinstance(expected_epochs, bool)
        or not isinstance(expected_epochs, int)
        or expected_epochs <= 0
    ):
        raise FormalTeacherHistoryError('config epochs must be a positive integer')
    if any(isinstance(epoch, bool) or not isinstance(epoch, int) for epoch in epochs):
        raise FormalTeacherHistoryError('completed epochs must contain integers')
    if epochs != list(range(1, expected_epochs + 1)):
        raise FormalTeacherHistoryError("completed epochs must match configured training epochs")

    checkpoints = [
        root / "epochs" / f"epoch-{epoch:03d}.pt" for epoch in epochs
    ]
    if any(not path.is_file() or path.is_symlink() for path in checkpoints):
        raise FormalTeacherHistoryError("epoch checkpoint history is incomplete")
    metrics_paths: list[Path] = []

    for checkpoint in checkpoints:
        output = root / "epoch_metrics" / f"{checkpoint.stem}.json"
        evaluation_fn(
            kind=kind,
            checkpoint=checkpoint,
            output=output,
            model_name=model_name,
            data_root=Path(data_root),
            batch=batch,
            workers=workers,
            device=device,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )
        if not output.is_file() or output.is_symlink():
            raise FormalTeacherHistoryError("evaluation metrics were not written")
        metrics_paths.append(output)

    result = selection_fn(
        metrics_paths,
        destination=root / "best_map50_95.pt",
        evidence_path=root / "best_map50_95.json",
    )
    return {
        **result,
        "kind": kind,
        "evaluated_epochs": list(epochs),
        "selection_split": "val",
        "test_evaluated": False,
    }
