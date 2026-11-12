from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from rgbt_uav.losses.detection import PairedDetectionLoss


class TeacherTrainingStepError(ValueError):
    """Raised when a teacher training batch is invalid."""


def _require_tensor(
    batch: Mapping[str, Any],
    key: str,
) -> Tensor:
    value = batch.get(key)

    if not isinstance(value, Tensor):
        raise TeacherTrainingStepError(
            f"batch field {key!r} must be a tensor"
        )

    return value


def run_teacher_training_step(
    model: nn.Module,
    criterion: PairedDetectionLoss,
    optimizer: Optimizer,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Run one supervised optimization step for the gated teacher."""

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration as exc:
            raise TeacherTrainingStepError(
                "model must contain at least one parameter"
            ) from exc
    else:
        target_device = torch.device(device)

    model.to(target_device)
    model.train()

    rgb = _require_tensor(batch, "rgb").to(target_device)
    ir = _require_tensor(batch, "ir").to(target_device)

    boxes = batch.get("boxes")
    labels = batch.get("labels")

    if not isinstance(boxes, list) or not isinstance(labels, list):
        raise TeacherTrainingStepError(
            "batch boxes and labels must be lists of tensors"
        )

    if len(boxes) != rgb.shape[0] or len(labels) != rgb.shape[0]:
        raise TeacherTrainingStepError(
            "target batch length must match image batch length"
        )

    moved_boxes: list[Tensor] = []
    moved_labels: list[Tensor] = []

    for index, (sample_boxes, sample_labels) in enumerate(
        zip(boxes, labels, strict=True)
    ):
        if not isinstance(sample_boxes, Tensor):
            raise TeacherTrainingStepError(
                f"boxes[{index}] must be a tensor"
            )

        if not isinstance(sample_labels, Tensor):
            raise TeacherTrainingStepError(
                f"labels[{index}] must be a tensor"
            )

        moved_boxes.append(sample_boxes.to(target_device))
        moved_labels.append(sample_labels.to(target_device))

    try:
        output = model(rgb, ir)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TeacherTrainingStepError(
            "teacher model forward failed"
        ) from exc

    if not isinstance(output, Mapping):
        raise TeacherTrainingStepError(
            "teacher model must return a mapping"
        )

    predictions = output.get("predictions")

    if not isinstance(predictions, (list, tuple)):
        raise TeacherTrainingStepError(
            "teacher output predictions must be a sequence"
        )

    losses = criterion(
        predictions,
        moved_boxes,
        moved_labels,
    )

    if "total" not in losses:
        raise TeacherTrainingStepError(
            "teacher criterion must return total loss"
        )

    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    optimizer.step()

    return {
        name: float(value.detach().cpu())
        for name, value in losses.items()
    }