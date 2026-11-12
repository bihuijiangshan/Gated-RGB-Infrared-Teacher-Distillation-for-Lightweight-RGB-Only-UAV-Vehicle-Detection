from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer


class TeacherCheckpointError(RuntimeError):
    """Raised when a teacher checkpoint is invalid."""


_FORMAT_VERSION = 1


def _validate_metrics(
    metrics: Mapping[str, object],
) -> dict[str, float]:
    result: dict[str, float] = {}

    for key, value in metrics.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TeacherCheckpointError(
                f"metric {key!r} must be a finite number"
            )

        result[key] = float(value)

    return result


def save_teacher_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Mapping[str, object],
) -> None:
    """Atomically save teacher model and optimizer state."""

    if not isinstance(model, nn.Module):
        raise TeacherCheckpointError(
            "model must be a torch.nn.Module"
        )

    if not isinstance(optimizer, Optimizer):
        raise TeacherCheckpointError(
            "optimizer must be a torch optimizer"
        )

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise TeacherCheckpointError(
            "epoch must be a non-negative integer"
        )

    payload = {
        "format_version": _FORMAT_VERSION,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": _validate_metrics(metrics),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".",
            suffix=".teacher.tmp",
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


def _mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TeacherCheckpointError(
            f"checkpoint field {field!r} must be a mapping"
        )

    return value


def load_teacher_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    map_location: torch.device | str | None = None,
) -> dict[str, object]:
    """Load teacher model and optimizer state."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise TeacherCheckpointError(
            f"checkpoint does not exist: {checkpoint_path}"
        )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise TeacherCheckpointError(
            f"could not load checkpoint: {checkpoint_path}"
        ) from exc

    checkpoint = _mapping(payload, field="root")

    if checkpoint.get("format_version") != _FORMAT_VERSION:
        raise TeacherCheckpointError(
            "unsupported checkpoint format_version"
        )

    epoch = checkpoint.get("epoch")

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise TeacherCheckpointError(
            "checkpoint epoch must be a non-negative integer"
        )

    metrics = _validate_metrics(
        _mapping(
            checkpoint.get("metrics"),
            field="metrics",
        )
    )

    try:
        model.load_state_dict(
            _mapping(
                checkpoint.get("model"),
                field="model",
            )
        )
        optimizer.load_state_dict(
            _mapping(
                checkpoint.get("optimizer"),
                field="optimizer",
            )
        )
    except Exception as exc:
        raise TeacherCheckpointError(
            "teacher checkpoint is incompatible with model or optimizer"
        ) from exc

    return {
        "format_version": _FORMAT_VERSION,
        "epoch": epoch,
        "metrics": metrics,
    }