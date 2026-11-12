"""Official Ultralytics detection loss for the RGB student."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from ultralytics.utils.loss import v8DetectionLoss

from rgbt_uav.training.official_distillation_student import (
    OfficialRGBStudent,
)


class OfficialStudentLossError(RuntimeError):
    """Raised when an official student loss batch is invalid."""


def paired_batch_to_ultralytics_targets(
    batch: Mapping[str, object],
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    """Convert a collated paired batch to Ultralytics target tensors."""

    if not isinstance(device, torch.device):
        raise OfficialStudentLossError(
            "device must be a torch.device"
        )

    rgb = batch.get("rgb")
    boxes = batch.get("boxes")
    labels = batch.get("labels")

    if not isinstance(rgb, Tensor) or rgb.ndim != 4:
        raise OfficialStudentLossError(
            "batch.rgb must have shape [B, C, H, W]"
        )
    if not isinstance(boxes, Sequence):
        raise OfficialStudentLossError(
            "batch.boxes must be a sequence"
        )
    if not isinstance(labels, Sequence):
        raise OfficialStudentLossError(
            "batch.labels must be a sequence"
        )

    batch_size = rgb.shape[0]
    if len(boxes) != batch_size or len(labels) != batch_size:
        raise OfficialStudentLossError(
            "boxes and labels must match the batch size"
        )

    batch_indices: list[Tensor] = []
    class_targets: list[Tensor] = []
    box_targets: list[Tensor] = []

    for image_index, (image_boxes, image_labels) in enumerate(
        zip(boxes, labels, strict=True)
    ):
        if not isinstance(image_boxes, Tensor):
            raise OfficialStudentLossError(
                "each boxes item must be a tensor"
            )
        if not isinstance(image_labels, Tensor):
            raise OfficialStudentLossError(
                "each labels item must be a tensor"
            )
        if image_boxes.ndim != 2 or image_boxes.shape[1] != 4:
            raise OfficialStudentLossError(
                "each boxes tensor must have shape [N, 4]"
            )
        if image_labels.ndim != 1:
            raise OfficialStudentLossError(
                "each labels tensor must have shape [N]"
            )
        if image_boxes.shape[0] != image_labels.shape[0]:
            raise OfficialStudentLossError(
                "boxes and labels counts must match"
            )
        if not torch.isfinite(image_boxes).all():
            raise OfficialStudentLossError(
                "boxes must contain only finite values"
            )
        if image_boxes.numel() and (
            (image_boxes < 0).any()
            or (image_boxes > 1).any()
        ):
            raise OfficialStudentLossError(
                "boxes must use normalized xywh coordinates"
            )

        count = image_boxes.shape[0]
        if count == 0:
            continue

        batch_indices.append(
            torch.full(
                (count,),
                float(image_index),
                dtype=torch.float32,
                device=device,
            )
        )
        class_targets.append(
            image_labels.to(
                device=device,
                dtype=torch.float32,
            ).reshape(-1, 1)
        )
        box_targets.append(
            image_boxes.to(
                device=device,
                dtype=torch.float32,
            )
        )

    if batch_indices:
        return {
            "batch_idx": torch.cat(batch_indices, dim=0),
            "cls": torch.cat(class_targets, dim=0),
            "bboxes": torch.cat(box_targets, dim=0),
        }

    return {
        "batch_idx": torch.empty(
            (0,),
            dtype=torch.float32,
            device=device,
        ),
        "cls": torch.empty(
            (0, 1),
            dtype=torch.float32,
            device=device,
        ),
        "bboxes": torch.empty(
            (0, 4),
            dtype=torch.float32,
            device=device,
        ),
    }


class OfficialStudentDetectionLoss(nn.Module):
    """Apply Ultralytics' native YOLOv8 detection loss."""

    def __init__(
        self,
        student: OfficialRGBStudent,
    ) -> None:
        super().__init__()

        if not isinstance(student, OfficialRGBStudent):
            raise OfficialStudentLossError(
                "student must be an OfficialRGBStudent"
            )

        self.student = student
        self.loss = v8DetectionLoss(student.model)

    def forward(
        self,
        predictions: Sequence[Tensor],
        batch: Mapping[str, object],
    ) -> dict[str, Tensor]:
        if len(predictions) != 3:
            raise OfficialStudentLossError(
                "predictions must contain three levels"
            )
        if not all(
            isinstance(prediction, Tensor)
            for prediction in predictions
        ):
            raise OfficialStudentLossError(
                "prediction levels must be tensors"
            )

        device = predictions[0].device
        targets = paired_batch_to_ultralytics_targets(
            batch,
            device=device,
        )

        try:
            total, components = self.loss(
                list(predictions),
                targets,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise OfficialStudentLossError(
                "official student detection loss failed"
            ) from exc

        if not isinstance(total, Tensor) or total.ndim != 0:
            raise OfficialStudentLossError(
                "official detection total must be scalar"
            )
        if (
            not isinstance(components, Tensor)
            or components.numel() != 3
        ):
            raise OfficialStudentLossError(
                "official loss must return box, cls, and dfl components"
            )

        flat_components = components.reshape(-1)

        return {
            "total": total,
            "box": flat_components[0],
            "classification": flat_components[1],
            "dfl": flat_components[2],
        }