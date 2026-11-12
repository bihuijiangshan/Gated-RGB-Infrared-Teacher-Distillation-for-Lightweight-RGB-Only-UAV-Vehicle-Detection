from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class DistillationEvidenceError(ValueError):
    """Raised when distillation completion evidence is invalid."""


_REQUIRED_FIELDS = {
    "status",
    "experiment_id",
    "experiment_group",
    "completed_epochs",
    "best_weights",
    "last_weights",
    "state_path",
    "metrics",
}


def _path_field(payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)

    if not isinstance(value, str) or not value.strip():
        raise DistillationEvidenceError(
            f"{key} must be a non-empty path"
        )

    path = Path(value)

    if not path.is_file():
        raise DistillationEvidenceError(
            f"{key} must be a regular file: {path}"
        )

    return path


def _validate_epochs(value: object) -> list[int]:
    if not isinstance(value, list) or not value:
        raise DistillationEvidenceError(
            "completed_epochs must be a non-empty list"
        )

    if any(
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch <= 0
        for epoch in value
    ):
        raise DistillationEvidenceError(
            "completed_epochs must contain positive integers"
        )

    if value != sorted(set(value)):
        raise DistillationEvidenceError(
            "completed_epochs must be strictly increasing"
        )

    return value


def _validate_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise DistillationEvidenceError(
            "metrics must be a mapping"
        )

    result: dict[str, float] = {}

    for key, metric in value.items():
        if not isinstance(key, str) or not key:
            raise DistillationEvidenceError(
                "metric names must be non-empty strings"
            )

        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise DistillationEvidenceError(
                f"metric {key!r} must be finite and numeric"
            )

        result[key] = float(metric)

    if "val_loss" not in result:
        raise DistillationEvidenceError(
            "metrics must contain val_loss"
        )

    return result


def validate_distillation_completion(
    completion_path: Path,
) -> dict[str, object]:
    """Validate completion metadata and referenced artifact files."""

    source = Path(completion_path)

    if not source.is_file():
        raise DistillationEvidenceError(
            f"completion must be a regular file: {source}"
        )

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistillationEvidenceError(
            f"could not read completion: {source}"
        ) from exc

    if not isinstance(payload, dict):
        raise DistillationEvidenceError(
            "completion root must be a JSON object"
        )

    missing = _REQUIRED_FIELDS - set(payload)
    if missing:
        raise DistillationEvidenceError(
            "completion is missing fields: "
            + ", ".join(sorted(missing))
        )

    if payload["status"] != "completed":
        raise DistillationEvidenceError(
            "completion status must be 'completed'"
        )

    for key in ("experiment_id", "experiment_group"):
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise DistillationEvidenceError(
                f"{key} must be a non-empty string"
            )

    _validate_epochs(payload["completed_epochs"])
    _path_field(payload, "best_weights")
    _path_field(payload, "last_weights")
    _path_field(payload, "state_path")
    _validate_metrics(payload["metrics"])

    return payload