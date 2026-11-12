"""Epoch orchestration for official RGB-T to RGB distillation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import Tensor

from rgbt_uav.training.official_teacher_student_step import (
    run_official_teacher_student_step,
)


class OfficialTeacherStudentEpochError(RuntimeError):
    """Raised when an official teacher-student epoch fails."""


_TRAIN_KEYS = (
    "total",
    "detection",
    "distillation",
    "feature",
    "logit",
    "box",
    "classification",
    "dfl",
)

_VALIDATION_KEYS = (
    "total",
    "box",
    "classification",
    "dfl",
)


def _batch_size(batch: Mapping[str, object]) -> int:
    rgb = batch.get("rgb")
    if not isinstance(rgb, Tensor) or rgb.ndim != 4:
        raise OfficialTeacherStudentEpochError(
            "batch.rgb must have shape [B, C, H, W]"
        )

    size = rgb.shape[0]
    if size <= 0:
        raise OfficialTeacherStudentEpochError(
            "batch size must be positive"
        )

    return size


def _finite_float(
    value: object,
    *,
    label: str,
) -> float:
    if isinstance(value, Tensor):
        if value.ndim != 0:
            raise OfficialTeacherStudentEpochError(
                f"{label} must be scalar"
            )
        result = float(value.detach().item())
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        raise OfficialTeacherStudentEpochError(
            f"{label} must be numeric"
        )

    if not torch.isfinite(torch.tensor(result)):
        raise OfficialTeacherStudentEpochError(
            f"{label} must be finite"
        )

    return result


def train_official_teacher_student_epoch(
    teacher: Any,
    student: Any,
    detection_criterion: Any,
    distillation_criterion: Any,
    optimizer: Any,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    num_classes: int,
    step_fn: Callable[..., Mapping[str, float]] = (
        run_official_teacher_student_step
    ),
) -> dict[str, float]:
    """Train the official RGB student for one epoch."""

    totals = {
        key: 0.0
        for key in _TRAIN_KEYS
    }
    batches = 0
    samples = 0

    for batch in loader:
        try:
            metrics = step_fn(
                teacher,
                student,
                detection_criterion,
                distillation_criterion,
                optimizer,
                batch,
                device=device,
                num_classes=num_classes,
            )
        except (
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OfficialTeacherStudentEpochError(
                f"training batch failed at batch {batches}"
            ) from exc

        if set(metrics) != set(_TRAIN_KEYS):
            raise OfficialTeacherStudentEpochError(
                "training step returned an invalid metric schema"
            )

        for key in _TRAIN_KEYS:
            totals[key] += _finite_float(
                metrics[key],
                label=f"training metric {key}",
            )

        samples += _batch_size(batch)
        batches += 1

    if batches == 0:
        raise OfficialTeacherStudentEpochError(
            "training loader produced no batches"
        )

    return {
        key: totals[key] / batches
        for key in _TRAIN_KEYS
    } | {
        "batches": float(batches),
        "samples": float(samples),
    }


def validate_official_rgb_student(
    student: Any,
    detection_criterion: Any,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate official student detection loss without updating it."""

    student.to(device)
    student.eval()

    totals = {
        key: 0.0
        for key in _VALIDATION_KEYS
    }
    batches = 0
    samples = 0

    with torch.no_grad():
        for batch in loader:
            try:
                rgb = batch["rgb"]
                if not isinstance(rgb, Tensor):
                    raise TypeError(
                        "batch.rgb must be a tensor"
                    )

                _, predictions = student(
                    rgb.to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                )
                metrics = detection_criterion(
                    predictions,
                    batch,
                )
            except (
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                raise OfficialTeacherStudentEpochError(
                    f"validation batch failed at batch {batches}"
                ) from exc

            if set(metrics) != set(_VALIDATION_KEYS):
                raise OfficialTeacherStudentEpochError(
                    "validation loss returned an invalid metric schema"
                )

            for key in _VALIDATION_KEYS:
                totals[key] += _finite_float(
                    metrics[key],
                    label=f"validation metric {key}",
                )

            samples += _batch_size(batch)
            batches += 1

    if batches == 0:
        raise OfficialTeacherStudentEpochError(
            "validation loader produced no batches"
        )

    return {
        key: totals[key] / batches
        for key in _VALIDATION_KEYS
    } | {
        "batches": float(batches),
        "samples": float(samples),
    }