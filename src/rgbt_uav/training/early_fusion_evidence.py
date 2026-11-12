from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path


class EarlyFusionEvidenceError(ValueError):
    """Raised when early-fusion completion evidence is invalid."""


_REQUIRED_FIELDS = frozenset(
    {
        "status",
        "experiment_id",
        "experiment_group",
        "config",
        "dataset",
        "completed_epochs",
        "all_completed_epochs",
        "best_weights",
        "last_weights",
        "metrics",
    }
)


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EarlyFusionEvidenceError(
            f"{field} must be a non-empty string"
        )
    return value


def _positive_int(value: object, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise EarlyFusionEvidenceError(
            f"{field} must be a positive integer"
        )
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise EarlyFusionEvidenceError(
            f"{field} must be a non-negative integer"
        )
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EarlyFusionEvidenceError(
            f"{field} must be a finite number"
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise EarlyFusionEvidenceError(
            f"{field} must be a finite number"
        )

    return converted


def _regular_file(
    declared: object,
    *,
    field: str,
    completion_path: Path,
) -> Path:
    declared_path = Path(_nonempty_string(declared, field))

    candidates: list[Path] = []

    if declared_path.is_absolute():
        candidates.append(declared_path)
    else:
        candidates.extend(
            [
                Path.cwd() / declared_path,
                completion_path.parent / declared_path,
            ]
        )

    for candidate in candidates:
        try:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
        except OSError:
            continue

    raise EarlyFusionEvidenceError(
        f"{field} must point to a regular file"
    )


def _epoch_list(value: object, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise EarlyFusionEvidenceError(
            f"{field} must be a non-empty list"
        )

    if any(
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch <= 0
        for epoch in value
    ):
        raise EarlyFusionEvidenceError(
            f"{field} must contain positive integers"
        )

    if value != sorted(set(value)):
        raise EarlyFusionEvidenceError(
            f"{field} must be sorted and unique"
        )

    return value


def validate_early_fusion_completion(
    path: Path,
) -> dict[str, object]:
    """Validate a completed early-fusion preflight evidence file."""

    completion_path = Path(path)

    if (
        not completion_path.is_file()
        or completion_path.is_symlink()
    ):
        raise EarlyFusionEvidenceError(
            f"completion must be a regular file: {completion_path}"
        )

    try:
        payload = json.loads(
            completion_path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise EarlyFusionEvidenceError(
            f"could not read completion: {completion_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise EarlyFusionEvidenceError(
            "completion root must be a JSON object"
        )

    missing = _REQUIRED_FIELDS - set(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise EarlyFusionEvidenceError(
            f"completion is missing required field(s): {names}"
        )

    if payload["status"] != "completed":
        raise EarlyFusionEvidenceError(
            "status must be completed"
        )

    _nonempty_string(payload["experiment_id"], "experiment_id")
    _nonempty_string(
        payload["experiment_group"],
        "experiment_group",
    )

    config = payload["config"]
    if not isinstance(config, Mapping):
        raise EarlyFusionEvidenceError(
            "config must be a mapping"
        )

    target_epochs = _positive_int(
        config.get("epochs"),
        "config.epochs",
    )
    _positive_int(config.get("batch"), "config.batch")
    _positive_int(config.get("imgsz"), "config.imgsz")
    _nonnegative_int(config.get("seed"), "config.seed")
    _nonnegative_int(config.get("workers"), "config.workers")
    _positive_int(config.get("num_classes"), "config.num_classes")
    _nonempty_string(config.get("model"), "config.model")
    _nonempty_string(config.get("device"), "config.device")
    _finite_number(config.get("lr"), "config.lr")

    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping):
        raise EarlyFusionEvidenceError(
            "dataset must be a mapping"
        )

    _nonnegative_int(
        dataset.get("train_count"),
        "dataset.train_count",
    )
    _nonnegative_int(
        dataset.get("val_count"),
        "dataset.val_count",
    )

    completed_epochs = _epoch_list(
        payload["completed_epochs"],
        "completed_epochs",
    )
    all_completed_epochs = _epoch_list(
        payload["all_completed_epochs"],
        "all_completed_epochs",
    )

    if all_completed_epochs != list(
        range(1, all_completed_epochs[-1] + 1)
    ):
        raise EarlyFusionEvidenceError(
            "all_completed_epochs must be contiguous from 1"
        )

    if all_completed_epochs[-1] > target_epochs:
        raise EarlyFusionEvidenceError(
            "all_completed_epochs cannot exceed config.epochs"
        )

    if any(epoch > target_epochs for epoch in completed_epochs):
        raise EarlyFusionEvidenceError(
            "completed_epochs cannot exceed config.epochs"
        )

    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping) or not metrics:
        raise EarlyFusionEvidenceError(
            "metrics must be a non-empty mapping"
        )

    for name, value in metrics.items():
        _nonempty_string(name, "metric name")
        _finite_number(value, f"metrics.{name}")

    _regular_file(
        payload["best_weights"],
        field="best_weights",
        completion_path=completion_path,
    )
    _regular_file(
        payload["last_weights"],
        field="last_weights",
        completion_path=completion_path,
    )

    return payload