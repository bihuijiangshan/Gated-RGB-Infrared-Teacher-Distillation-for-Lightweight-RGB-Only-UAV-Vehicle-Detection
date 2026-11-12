from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from rgbt_uav.losses.detection import PairedDetectionLoss


class PairedTrainingStepError(ValueError):
    """Raised when a paired training batch is invalid."""


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise PairedTrainingStepError(
            "model must contain at least one parameter"
        ) from exc


def _batch_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise PairedTrainingStepError(
            f"batch field {key!r} must be a tensor"
        )
    return value


def run_paired_training_step(
    model: nn.Module,
    criterion: PairedDetectionLoss,
    optimizer: Optimizer,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Run one supervised optimization step for a paired RGB-IR batch."""

    target_device = (
        torch.device(device)
        if device is not None
        else _model_device(model)
    )

    model.to(target_device)
    model.train()

    rgb = _batch_tensor(batch, "rgb").to(target_device)
    ir = _batch_tensor(batch, "ir").to(target_device)

    boxes = batch.get("boxes")
    labels = batch.get("labels")

    if not isinstance(boxes, list) or not isinstance(labels, list):
        raise PairedTrainingStepError(
            "batch boxes and labels must be lists of tensors"
        )

    if len(boxes) != rgb.shape[0] or len(labels) != rgb.shape[0]:
        raise PairedTrainingStepError(
            "target batch length must match image batch length"
        )

    moved_boxes = [
        value.to(target_device)
        if isinstance(value, Tensor)
        else value
        for value in boxes
    ]
    moved_labels = [
        value.to(target_device)
        if isinstance(value, Tensor)
        else value
        for value in labels
    ]

    if not all(isinstance(value, Tensor) for value in moved_boxes):
        raise PairedTrainingStepError(
            "every boxes item must be a tensor"
        )

    if not all(isinstance(value, Tensor) for value in moved_labels):
        raise PairedTrainingStepError(
            "every labels item must be a tensor"
        )

    predictions = model(rgb, ir)
    losses = criterion(
        predictions["predictions"],
        moved_boxes,
        moved_labels,
    )

    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    optimizer.step()

    return {
        name: float(value.detach().cpu())
        for name, value in losses.items()
    }