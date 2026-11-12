from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from rgbt_uav.training.distillation_step import (
    DistillationTrainingStepError,
    run_distillation_training_step,
)


class DistillationTrainingError(ValueError):
    """Raised when a teacher-student epoch cannot be completed."""


DistillationTrainingStep = Callable[..., dict[str, float]]


def train_distillation_epoch(
    teacher: nn.Module,
    student: nn.Module,
    detection_loss: Any,
    distillation_loss: Any,
    feature_distiller: nn.Module,
    optimizer: Optimizer,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device | str | None = None,
    step_fn: DistillationTrainingStep = run_distillation_training_step,
) -> dict[str, float]:
    """Run one complete frozen-teacher distillation epoch."""

    if not isinstance(teacher, nn.Module):
        raise DistillationTrainingError("teacher must be a torch.nn.Module")
    if not isinstance(student, nn.Module):
        raise DistillationTrainingError("student must be a torch.nn.Module")
    if not isinstance(feature_distiller, nn.Module):
        raise DistillationTrainingError(
            "feature_distiller must be a torch.nn.Module"
        )

    if device is None:
        try:
            target_device = next(student.parameters()).device
        except StopIteration as exc:
            raise DistillationTrainingError(
                "student must contain at least one parameter"
            ) from exc
    else:
        target_device = torch.device(device)

    teacher.to(target_device)
    student.to(target_device)
    feature_distiller.to(target_device)
    teacher.eval()
    student.train()
    feature_distiller.train()

    totals = {
        "total": 0.0,
        "detection": 0.0,
        "distillation": 0.0,
        "feature": 0.0,
        "logit": 0.0,
    }
    batches = 0
    samples = 0

    for batch in loader:
        try:
            rgb = batch["rgb"]
            if not isinstance(rgb, torch.Tensor) or rgb.ndim < 1:
                raise DistillationTrainingError(
                    "batch rgb tensor must have a batch dimension"
                )
            batch_size = rgb.shape[0]
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise DistillationTrainingError(
                    "batch rgb tensor must have a positive batch dimension"
                )

            metrics = step_fn(
                teacher,
                student,
                detection_loss,
                distillation_loss,
                feature_distiller,
                optimizer,
                batch,
                device=target_device,
            )
            if not isinstance(metrics, Mapping):
                raise DistillationTrainingError(
                    "distillation step must return a metrics mapping"
                )
            for key in totals:
                value = metrics.get(key)
                if not isinstance(value, (int, float)):
                    raise DistillationTrainingError(
                        f"training step metric {key!r} must be numeric"
                    )
                if not torch.isfinite(torch.tensor(float(value))):
                    raise DistillationTrainingError(
                        f"training step metric {key!r} must be finite"
                    )
                totals[key] += float(value)
        except DistillationTrainingError:
            raise
        except (KeyError, TypeError, RuntimeError, DistillationTrainingStepError) as exc:
            raise DistillationTrainingError(
                f"distillation batch failed at batch {batches}"
            ) from exc

        batches += 1
        samples += batch_size

    if batches == 0:
        raise DistillationTrainingError("loader produced no batches")

    return {
        **{key: value / batches for key, value in totals.items()},
        "batches": float(batches),
        "samples": float(samples),
    }

