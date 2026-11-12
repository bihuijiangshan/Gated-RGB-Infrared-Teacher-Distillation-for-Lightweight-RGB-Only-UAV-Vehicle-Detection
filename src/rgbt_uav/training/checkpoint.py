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


class EarlyFusionCheckpointError(RuntimeError):
    """Raised when an early-fusion checkpoint is invalid."""


_FORMAT_VERSION = 1


def _validate_epoch(epoch: object) -> int:
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise EarlyFusionCheckpointError(
            "epoch must be a non-negative integer"
        )

    return epoch


def _validate_metrics(
    metrics: Mapping[str, object],
) -> dict[str, float]:
    result: dict[str, float] = {}

    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise EarlyFusionCheckpointError(
                "metric names must be non-empty strings"
            )

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EarlyFusionCheckpointError(
                f"metric {key!r} must be numeric"
            )

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise EarlyFusionCheckpointError(
                f"metric {key!r} must be finite"
            )

        result[key] = numeric_value

    return result


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Mapping[str, object],
) -> dict[str, Any]:
    if not isinstance(model, nn.Module):
        raise EarlyFusionCheckpointError(
            "model must be a torch.nn.Module"
        )

    if not isinstance(optimizer, Optimizer):
        raise EarlyFusionCheckpointError(
            "optimizer must be a torch optimizer"
        )

    return {
        "format_version": _FORMAT_VERSION,
        "epoch": _validate_epoch(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": _validate_metrics(metrics),
    }


def save_early_fusion_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Mapping[str, object],
) -> None:
    """Atomically save model, optimizer, epoch, and metrics state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics=metrics,
    )

    temporary_path: Path | None = None

    try:
        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".",
            suffix=".checkpoint.tmp",
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


def _require_mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EarlyFusionCheckpointError(
            f"checkpoint field {field!r} must be a mapping"
        )

    return value


def load_early_fusion_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    map_location: torch.device | str | None = None,
) -> dict[str, object]:
    """Load and validate an early-fusion checkpoint."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise EarlyFusionCheckpointError(
            f"checkpoint does not exist: {checkpoint_path}"
        )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise EarlyFusionCheckpointError(
            f"could not load checkpoint: {checkpoint_path}"
        ) from exc

    checkpoint = _require_mapping(payload, field="root")

    format_version = checkpoint.get("format_version")
    if format_version != _FORMAT_VERSION:
        raise EarlyFusionCheckpointError(
            "unsupported checkpoint format_version"
        )

    epoch = _validate_epoch(checkpoint.get("epoch"))

    model_state = _require_mapping(
        checkpoint.get("model"),
        field="model",
    )
    optimizer_state = _require_mapping(
        checkpoint.get("optimizer"),
        field="optimizer",
    )
    metrics_mapping = _require_mapping(
        checkpoint.get("metrics"),
        field="metrics",
    )

    metrics = _validate_metrics(metrics_mapping)

    try:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    except Exception as exc:
        raise EarlyFusionCheckpointError(
            "checkpoint state is incompatible with model or optimizer"
        ) from exc

    return {
        "format_version": format_version,
        "epoch": epoch,
        "metrics": metrics,
    }