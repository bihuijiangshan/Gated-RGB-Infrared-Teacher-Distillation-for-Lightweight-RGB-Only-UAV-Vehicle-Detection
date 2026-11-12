"""Checkpoint persistence for official RGB student distillation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class OfficialRGBDistillationCheckpointError(RuntimeError):
    """Raised when an official RGB checkpoint is invalid."""


_REQUIRED_FIELDS = {
    "format_version",
    "epoch",
    "student",
    "feature_distiller",
    "optimizer",
    "metrics",
}


def _validated_metrics(
    metrics: Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise OfficialRGBDistillationCheckpointError(
            "metrics must be a mapping"
        )

    result: dict[str, float] = {}

    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise OfficialRGBDistillationCheckpointError(
                "metric names must be non-empty strings"
            )
        if not isinstance(value, (int, float)):
            raise OfficialRGBDistillationCheckpointError(
                f"metric {key} must be numeric"
            )

        number = float(value)
        if not math.isfinite(number):
            raise OfficialRGBDistillationCheckpointError(
                f"metric {key} must be finite"
            )

        result[key] = number

    return result


def save_official_rgb_distillation_checkpoint(
    path: Path,
    *,
    epoch: int,
    student: nn.Module,
    feature_distiller: nn.Module,
    optimizer: Optimizer,
    metrics: Mapping[str, float],
) -> Path:
    """Atomically save official RGB student training state."""

    if not isinstance(epoch, int) or epoch <= 0:
        raise OfficialRGBDistillationCheckpointError(
            "epoch must be positive"
        )
    if not isinstance(student, nn.Module):
        raise OfficialRGBDistillationCheckpointError(
            "student must be a module"
        )
    if not isinstance(feature_distiller, nn.Module):
        raise OfficialRGBDistillationCheckpointError(
            "feature_distiller must be a module"
        )
    if not isinstance(optimizer, Optimizer):
        raise OfficialRGBDistillationCheckpointError(
            "optimizer must be an Optimizer"
        )

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    payload = {
        "format_version": 1,
        "epoch": epoch,
        "student": student.state_dict(),
        "feature_distiller": feature_distiller.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": _validated_metrics(metrics),
    }

    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        if temporary.is_file():
            temporary.unlink()
        raise OfficialRGBDistillationCheckpointError(
            f"failed to save checkpoint: {destination}"
        ) from exc

    return destination


def load_official_rgb_distillation_checkpoint(
    path: Path,
    *,
    student: nn.Module,
    feature_distiller: nn.Module,
    optimizer: Optimizer,
    map_location: torch.device | str,
) -> dict[str, Any]:
    """Restore official RGB student training state."""

    source = Path(path)
    if not source.is_file():
        raise OfficialRGBDistillationCheckpointError(
            f"checkpoint must be a regular file: {source}"
        )

    try:
        payload = torch.load(
            source,
            map_location=map_location,
            weights_only=False,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise OfficialRGBDistillationCheckpointError(
            f"failed to load checkpoint: {source}"
        ) from exc

    if not isinstance(payload, dict):
        raise OfficialRGBDistillationCheckpointError(
            "checkpoint payload must be a dictionary"
        )

    missing = _REQUIRED_FIELDS - set(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise OfficialRGBDistillationCheckpointError(
            f"checkpoint is missing fields: {names}"
        )

    if payload["format_version"] != 1:
        raise OfficialRGBDistillationCheckpointError(
            "unsupported checkpoint format version"
        )

    epoch = payload["epoch"]
    if not isinstance(epoch, int) or epoch <= 0:
        raise OfficialRGBDistillationCheckpointError(
            "checkpoint epoch must be positive"
        )

    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise OfficialRGBDistillationCheckpointError(
            "checkpoint metrics must be a mapping"
        )

    try:
        student.load_state_dict(
            payload["student"],
            strict=True,
        )
        feature_distiller.load_state_dict(
            payload["feature_distiller"],
            strict=True,
        )
        optimizer.load_state_dict(
            payload["optimizer"]
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise OfficialRGBDistillationCheckpointError(
            "checkpoint state is incompatible"
        ) from exc

    return {
        "format_version": 1,
        "epoch": epoch,
        "metrics": _validated_metrics(metrics),
    }