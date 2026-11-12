from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class TeacherTrainingError(ValueError):
    """Raised when a teacher training epoch is invalid."""


def train_teacher_epoch(
    model: nn.Module,
    criterion: Any,
    optimizer: Optimizer,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Run one supervised teacher training epoch."""

    if not isinstance(model, nn.Module):
        raise TeacherTrainingError("model must be a torch.nn.Module")

    if not isinstance(optimizer, Optimizer):
        raise TeacherTrainingError("optimizer must be a torch optimizer")

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration as exc:
            raise TeacherTrainingError(
                "model must contain parameters"
            ) from exc
    else:
        target_device = torch.device(device)

    model.to(target_device)
    model.train()

    totals = {
        "total": 0.0,
        "classification": 0.0,
        "box": 0.0,
    }
    batches = 0
    samples = 0

    for batch_index, batch in enumerate(loader):
        try:
            rgb = batch["rgb"]
            ir = batch["ir"]
            boxes = batch["boxes"]
            labels = batch["labels"]
        except KeyError as exc:
            raise TeacherTrainingError(
                f"batch {batch_index} is missing field {exc.args[0]!r}"
            ) from exc

        if not isinstance(rgb, Tensor) or rgb.ndim != 4:
            raise TeacherTrainingError(
                "batch rgb must have shape [B, 3, H, W]"
            )

        if not isinstance(ir, Tensor) or ir.ndim != 4:
            raise TeacherTrainingError(
                "batch ir must have shape [B, 1, H, W]"
            )

        if not isinstance(boxes, list) or not isinstance(labels, list):
            raise TeacherTrainingError(
                "batch boxes and labels must be lists"
            )

        batch_size = int(rgb.shape[0])

        if batch_size <= 0:
            raise TeacherTrainingError(
                "batch size must be positive"
            )

        if len(boxes) != batch_size or len(labels) != batch_size:
            raise TeacherTrainingError(
                "target batch length must match image batch size"
            )

        try:
            output = model(
                rgb.to(target_device),
                ir.to(target_device),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise TeacherTrainingError(
                f"teacher forward failed at batch {batch_index}"
            ) from exc

        if not isinstance(output, Mapping):
            raise TeacherTrainingError(
                "teacher output must be a mapping"
            )

        predictions = output.get("predictions")

        if not isinstance(predictions, (list, tuple)):
            raise TeacherTrainingError(
                "teacher predictions must be a sequence"
            )

        moved_boxes = [
            value.to(target_device)
            for value in boxes
            if isinstance(value, Tensor)
        ]
        moved_labels = [
            value.to(target_device)
            for value in labels
            if isinstance(value, Tensor)
        ]

        if len(moved_boxes) != batch_size or len(moved_labels) != batch_size:
            raise TeacherTrainingError(
                "teacher targets must contain tensors for every sample"
            )

        metrics = criterion(
            predictions,
            moved_boxes,
            moved_labels,
        )

        total = metrics.get("total")
        classification = metrics.get("classification")
        box = metrics.get("box")

        if not all(
            isinstance(value, Tensor)
            for value in (total, classification, box)
        ):
            raise TeacherTrainingError(
                "teacher criterion must return total, classification, and box tensors"
            )

        if not all(
            torch.isfinite(value)
            for value in (total, classification, box)
        ):
            raise TeacherTrainingError(
                "teacher loss values must be finite"
            )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        totals["total"] += float(total.detach().cpu())
        totals["classification"] += float(
            classification.detach().cpu()
        )
        totals["box"] += float(box.detach().cpu())
        batches += 1
        samples += batch_size

    if batches == 0:
        raise TeacherTrainingError(
            "loader produced no batches"
        )

    return {
        "total": totals["total"] / batches,
        "classification": totals["classification"] / batches,
        "box": totals["box"] / batches,
        "batches": float(batches),
        "samples": float(samples),
    }