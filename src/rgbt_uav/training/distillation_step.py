from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from rgbt_uav.losses.detection import PairedDetectionLoss
from rgbt_uav.losses.distillation import (
    DistillationLoss,
    FeatureDistiller,
)


class DistillationTrainingStepError(ValueError):
    """Raised when a teacher-student training batch is invalid."""


def _require_tensor(
    batch: Mapping[str, Any],
    key: str,
) -> Tensor:
    value = batch.get(key)

    if not isinstance(value, Tensor):
        raise DistillationTrainingStepError(
            f"batch field {key!r} must be a tensor"
        )

    return value


def _move_targets(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[Tensor], list[Tensor]]:
    boxes = batch.get("boxes")
    labels = batch.get("labels")

    if not isinstance(boxes, list) or not isinstance(labels, list):
        raise DistillationTrainingStepError(
            "batch boxes and labels must be lists"
        )

    if len(boxes) != batch_size or len(labels) != batch_size:
        raise DistillationTrainingStepError(
            "target batch length must match image batch length"
        )

    moved_boxes: list[Tensor] = []
    moved_labels: list[Tensor] = []

    for index, (sample_boxes, sample_labels) in enumerate(
        zip(boxes, labels, strict=True)
    ):
        if not isinstance(sample_boxes, Tensor):
            raise DistillationTrainingStepError(
                f"boxes[{index}] must be a tensor"
            )

        if not isinstance(sample_labels, Tensor):
            raise DistillationTrainingStepError(
                f"labels[{index}] must be a tensor"
            )

        moved_boxes.append(sample_boxes.to(device))
        moved_labels.append(sample_labels.to(device))

    return moved_boxes, moved_labels


def run_distillation_training_step(
    teacher: nn.Module,
    student: nn.Module,
    detection_loss: PairedDetectionLoss,
    distillation_loss: DistillationLoss,
    feature_distiller: FeatureDistiller,
    optimizer: Optimizer,
    batch: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Run one frozen-teacher, trainable-student distillation update."""

    if not isinstance(teacher, nn.Module):
        raise DistillationTrainingStepError(
            "teacher must be a torch.nn.Module"
        )

    if not isinstance(student, nn.Module):
        raise DistillationTrainingStepError(
            "student must be a torch.nn.Module"
        )

    if device is None:
        try:
            target_device = next(student.parameters()).device
        except StopIteration as exc:
            raise DistillationTrainingStepError(
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

    rgb = _require_tensor(batch, "rgb").to(target_device)
    ir = _require_tensor(batch, "ir").to(target_device)
    boxes, labels = _move_targets(
        batch,
        device=target_device,
        batch_size=rgb.shape[0],
    )

    with torch.no_grad():
        teacher_output = teacher(rgb, ir)

    student_output = student(rgb, ir)

    if not isinstance(teacher_output, Mapping):
        raise DistillationTrainingStepError(
            "teacher output must be a mapping"
        )

    if not isinstance(student_output, Mapping):
        raise DistillationTrainingStepError(
            "student output must be a mapping"
        )

    teacher_features = teacher_output.get("features")
    student_features = student_output.get("features")
    teacher_logits = teacher_output.get("predictions")
    student_logits = student_output.get("predictions")

    if not isinstance(teacher_features, (list, tuple)):
        raise DistillationTrainingStepError(
            "teacher features must be a sequence"
        )

    if not isinstance(student_features, (list, tuple)):
        raise DistillationTrainingStepError(
            "student features must be a sequence"
        )

    if not isinstance(teacher_logits, (list, tuple)):
        raise DistillationTrainingStepError(
            "teacher predictions must be a sequence"
        )

    if not isinstance(student_logits, (list, tuple)):
        raise DistillationTrainingStepError(
            "student predictions must be a sequence"
        )

    projected_features = [
        adapter(feature)
        for adapter, feature in zip(
            feature_distiller.adapters,
            student_features,
            strict=True,
        )
    ]

    detection = detection_loss(
        student_logits,
        boxes,
        labels,
    )

    class_teacher_logits = [value[:, : detection_loss.num_classes] for value in teacher_logits]
    class_student_logits = [value[:, : detection_loss.num_classes] for value in student_logits]

    distillation = distillation_loss(
        teacher_features,
        projected_features,
        class_teacher_logits,
        class_student_logits,
    )

    total = detection["total"] + distillation["total"]

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()

    return {
        "total": float(total.detach().cpu()),
        "detection": float(detection["total"].detach().cpu()),
        "distillation": float(distillation["total"].detach().cpu()),
        "feature": float(distillation["feature"].detach().cpu()),
        "logit": float(distillation["logit"].detach().cpu()),
    }