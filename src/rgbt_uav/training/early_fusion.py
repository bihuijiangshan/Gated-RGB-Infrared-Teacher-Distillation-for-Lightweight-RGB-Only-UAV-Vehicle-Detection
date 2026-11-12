from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from rgbt_uav.training.ultralytics_adapter import (
    UltralyticsAdapterError,
    run_ultralytics_early_fusion_step,
)


class EarlyFusionTrainingError(ValueError):
    """Raised when the early-fusion epoch configuration is invalid."""

TrainingStep = Callable[
    [
        nn.Module,
        Any,
        Optimizer,
        Mapping[str, object],
    ],
    dict[str, float],
]

def train_early_fusion_epoch(
    model: nn.Module,
    criterion: Any,
    optimizer: Optimizer,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device | str | None = None,
    step_fn: TrainingStep = run_ultralytics_early_fusion_step,
) -> dict[str, float]:
    """Run one complete training epoch over an early-fusion loader."""

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration as exc:
            raise EarlyFusionTrainingError(
                "model must contain parameters"
            ) from exc
    else:
        target_device = torch.device(device)

    model.to(target_device)
    model.train()

    total = 0.0
    box = 0.0
    classification = 0.0
    dfl = 0.0
    has_dfl = False
    batches = 0
    samples = 0

    for batch in loader:
        try:
            metrics = step_fn(
                model,
                criterion,
                optimizer,
                batch,
                device=target_device,
            )
        except (KeyError, TypeError, RuntimeError, UltralyticsAdapterError) as exc:
            raise EarlyFusionTrainingError(
                f"early-fusion batch failed at batch {batches}"
            ) from exc

        batch_size = batch["rgb"].shape[0]
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise EarlyFusionTrainingError(
                "batch rgb tensor must have a positive batch dimension"
            )

        try:
            total += metrics["total"]
            box += metrics["box"]
            classification += metrics["classification"]
        except KeyError as exc:
            raise EarlyFusionTrainingError(
                f"training step metrics missing required key: {exc.args[0]!r}"
            ) from exc

        if "dfl" in metrics:
            dfl += metrics["dfl"]
            has_dfl = True

        batches += 1
        samples += batch_size

    if batches == 0:
        raise EarlyFusionTrainingError("loader produced no batches")

    result = {
        "total": total / batches,
        "box": box / batches,
        "classification": classification / batches,
        "batches": float(batches),
        "samples": float(samples),
    }

    if has_dfl:
        result["dfl"] = dfl / batches

    return result