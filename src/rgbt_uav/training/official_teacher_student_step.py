"""One training step for an official RGB-T teacher and RGB student."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from rgbt_uav.training.official_distillation_student import (
    OfficialRGBStudent,
)
from rgbt_uav.training.official_distillation_teacher import (
    reorder_official_raw_predictions,
)
from rgbt_uav.training.official_student_loss import (
    OfficialStudentDetectionLoss,
)


class OfficialTeacherStudentStepError(RuntimeError):
    """Raised when the official distillation step cannot complete."""


def _require_image(
    batch: Mapping[str, object],
    key: str,
    *,
    channels: int,
    device: torch.device,
) -> Tensor:
    value = batch.get(key)

    if not isinstance(value, Tensor):
        raise OfficialTeacherStudentStepError(
            f"batch.{key} must be a tensor"
        )
    if value.ndim != 4 or value.shape[1] != channels:
        raise OfficialTeacherStudentStepError(
            f"batch.{key} has an invalid image shape"
        )

    return value.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

def _unpack_teacher_output(
    output: object,
) -> tuple[object, object]:
    if isinstance(output, Mapping):
        if "features" not in output:
            raise OfficialTeacherStudentStepError(
                "teacher output is missing features"
            )
        if "predictions" not in output:
            raise OfficialTeacherStudentStepError(
                "teacher output is missing predictions"
            )

        return (
            output["features"],
            output["predictions"],
        )

    if isinstance(output, Sequence) and len(output) == 2:
        return output[0], output[1]

    raise OfficialTeacherStudentStepError(
        "teacher returned an invalid output schema"
    )

def _require_three_levels(
    values: object,
    *,
    label: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(values, Sequence) or len(values) != 3:
        raise OfficialTeacherStudentStepError(
            f"{label} must contain three levels"
        )

    tensors = tuple(values)
    if not all(isinstance(value, Tensor) for value in tensors):
        raise OfficialTeacherStudentStepError(
            f"{label} levels must be tensors"
        )

    return tensors  # type: ignore[return-value]


def run_official_teacher_student_step(
    teacher: nn.Module,
    student: OfficialRGBStudent,
    detection_criterion: OfficialStudentDetectionLoss,
    distillation_criterion: Any,
    optimizer: Optimizer,
    batch: Mapping[str, object],
    *,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    """Update the RGB student while keeping the RGB-T teacher frozen."""

    if not isinstance(teacher, nn.Module):
        raise OfficialTeacherStudentStepError("teacher must be a torch module")
    if not isinstance(student, OfficialRGBStudent):
        raise OfficialTeacherStudentStepError(
            "student must be an OfficialRGBStudent"
        )
    if not isinstance(
        detection_criterion,
        OfficialStudentDetectionLoss,
    ):
        raise OfficialTeacherStudentStepError(
            "detection criterion has an invalid type"
        )
    if not isinstance(device, torch.device):
        raise OfficialTeacherStudentStepError(
            "device must be a torch.device"
        )
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise OfficialTeacherStudentStepError(
            "num_classes must be positive"
        )

    rgb = _require_image(
        batch,
        "rgb",
        channels=3,
        device=device,
    )
    ir = _require_image(
        batch,
        "ir",
        channels=1,
        device=device,
    )

    teacher.to(device)
    student.to(device)

    teacher.eval()
    student.train()

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        teacher_output = teacher(
            rgb,
            ir,
        )

    teacher_features, teacher_predictions = _unpack_teacher_output(teacher_output)

    student_features, student_raw_predictions = student(rgb)

    teacher_features = _require_three_levels(
        teacher_features,
        label="teacher features",
    )
    teacher_predictions = _require_three_levels(
        teacher_predictions,
        label="teacher predictions",
    )
    student_features = _require_three_levels(
        student_features,
        label="student features",
    )
    student_raw_predictions = _require_three_levels(
        student_raw_predictions,
        label="student predictions",
    )

    detection = detection_criterion(
        student_raw_predictions,
        batch,
    )

    student_predictions = reorder_official_raw_predictions(
        student_raw_predictions,
        num_classes=num_classes,
    )

    distillation = distillation_criterion(
        teacher_features,
        student_features,
        teacher_predictions,
        student_predictions,
    )

    required_detection = {
        "total",
        "box",
        "classification",
        "dfl",
    }
    if set(detection) != required_detection:
        raise OfficialTeacherStudentStepError(
            "detection loss returned an invalid schema"
        )

    required_distillation = {
        "total",
        "feature",
        "logit",
    }
    if set(distillation) != required_distillation:
        raise OfficialTeacherStudentStepError(
            "distillation loss returned an invalid schema"
        )

    total = detection["total"] + distillation["total"]

    if total.ndim != 0 or not torch.isfinite(total):
        raise OfficialTeacherStudentStepError(
            "combined loss must be a finite scalar"
        )

    total.backward()
    optimizer.step()

    metrics = {
        "total": total.detach().item(),
        "detection": detection["total"].detach().item(),
        "distillation": distillation["total"].detach().item(),
        "feature": distillation["feature"].detach().item(),
        "logit": distillation["logit"].detach().item(),
        "box": detection["box"].detach().item(),
        "classification": detection[
            "classification"
        ].detach().item(),
        "dfl": detection["dfl"].detach().item(),
    }

    if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise OfficialTeacherStudentStepError(
            "step metrics must all be finite"
        )

    return metrics