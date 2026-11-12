from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class DistillationCheckpointError(RuntimeError):
    """Raised when a distillation checkpoint is invalid or incompatible."""


_FORMAT_VERSION = 1


def _metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise DistillationCheckpointError(
                "metric names must be non-empty strings"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DistillationCheckpointError(f"metric {key!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise DistillationCheckpointError(f"metric {key!r} must be finite")
        result[key] = numeric
    return result


def _model_state(model: nn.Module, field: str) -> Mapping[str, Any]:
    if not isinstance(model, nn.Module):
        raise DistillationCheckpointError(f"{field} must be a torch.nn.Module")
    return model.state_dict()


def save_distillation_checkpoint(
    path: Path,
    *,
    teacher: nn.Module,
    student: nn.Module,
    feature_distiller: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Mapping[str, object],
) -> None:
    """Atomically save all trainable and frozen distillation state."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise DistillationCheckpointError(
            "epoch must be a non-negative integer"
        )
    if not isinstance(optimizer, Optimizer):
        raise DistillationCheckpointError("optimizer must be a torch optimizer")

    payload = {
        "format_version": _FORMAT_VERSION,
        "epoch": epoch,
        "teacher": _model_state(teacher, "teacher"),
        "student": _model_state(student, "student"),
        "feature_distiller": _model_state(feature_distiller, "feature_distiller"),
        "optimizer": optimizer.state_dict(),
        "metrics": _metrics(metrics),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".",
            suffix=".distillation.tmp",
            dir=destination.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _mapping(payload: object, field: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise DistillationCheckpointError(
            f"checkpoint field {field!r} must be a mapping"
        )
    return payload


def load_distillation_checkpoint(
    path: Path,
    *,
    teacher: nn.Module,
    student: nn.Module,
    feature_distiller: nn.Module,
    optimizer: Optimizer,
    map_location: torch.device | str | None = None,
) -> dict[str, object]:
    """Load a validated distillation checkpoint into all components."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise DistillationCheckpointError(
            f"checkpoint does not exist: {checkpoint_path}"
        )
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise DistillationCheckpointError(
            f"could not load checkpoint: {checkpoint_path}"
        ) from exc

    checkpoint = _mapping(payload, "root")
    if checkpoint.get("format_version") != _FORMAT_VERSION:
        raise DistillationCheckpointError("unsupported checkpoint format_version")
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise DistillationCheckpointError("epoch must be a non-negative integer")
    metrics = _metrics(_mapping(checkpoint.get("metrics"), "metrics"))

    try:
        teacher.load_state_dict(_mapping(checkpoint.get("teacher"), "teacher"))
        student.load_state_dict(_mapping(checkpoint.get("student"), "student"))
        feature_distiller.load_state_dict(
            _mapping(checkpoint.get("feature_distiller"), "feature_distiller")
        )
        optimizer.load_state_dict(_mapping(checkpoint.get("optimizer"), "optimizer"))
    except Exception as exc:
        raise DistillationCheckpointError(
            "checkpoint state is incompatible with distillation components"
        ) from exc

    return {
        "format_version": _FORMAT_VERSION,
        "epoch": epoch,
        "metrics": metrics,
    }

