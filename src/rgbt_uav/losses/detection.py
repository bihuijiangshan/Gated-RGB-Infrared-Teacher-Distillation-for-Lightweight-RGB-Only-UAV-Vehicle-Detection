from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class PairedDetectionLossError(ValueError):
    """Raised when paired detection targets are invalid."""


class PairedDetectionLoss(nn.Module):
    """Minimal P3 dense loss for validating the paired training path.

    This is a development smoke loss. It is not the final YOLOv8 loss.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        reg_max: int = 4,
        classification_weight: float = 1.0,
        box_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise PairedDetectionLossError(
                "num_classes must be positive"
            )

        if reg_max <= 0:
            raise PairedDetectionLossError(
                "reg_max must be positive"
            )

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.classification_weight = classification_weight
        self.box_weight = box_weight

    def forward(
        self,
        predictions: Sequence[Tensor],
        boxes: Sequence[Tensor],
        labels: Sequence[Tensor],
    ) -> dict[str, Tensor]:
        if not predictions:
            raise PairedDetectionLossError("predictions must not be empty")

        if len(boxes) != len(labels):
            raise PairedDetectionLossError("boxes and labels must have the same batch length")

        level_classification_losses: list[Tensor] = []
        level_box_losses: list[Tensor] = []

        for level_index, prediction in enumerate(predictions):
            if prediction.ndim != 4:
                raise PairedDetectionLossError(
                    f"prediction level {level_index} must have shape "
                    "[batch, channels, height, width]"
                )

            batch_size, channels, height, width = prediction.shape
            expected_channels = self.num_classes + 4 * self.reg_max

            if channels != expected_channels:
                raise PairedDetectionLossError(
                    f"expected {expected_channels} prediction channels, "
                    f"got {channels} at level {level_index}"
                )

            if len(boxes) != batch_size:
                raise PairedDetectionLossError(
                    "target batch length must match prediction batch size"
                )

            target_classes = torch.zeros(
                (
                    batch_size,
                    self.num_classes,
                    height,
                    width,
                ),
                dtype=prediction.dtype,
                device=prediction.device,
            )

            target_boxes = torch.zeros(
                (
                    batch_size,
                    4,
                    height,
                    width,
                ),
                dtype=prediction.dtype,
                device=prediction.device,
            )

            positive = torch.zeros(
                (batch_size, height, width),
                dtype=torch.bool,
                device=prediction.device,
            )

            for batch_index, (sample_boxes, sample_labels) in enumerate(
                zip(boxes, labels, strict=True)
            ):
                self._validate_targets(
                    sample_boxes,
                    sample_labels,
                    batch_index=batch_index,
                )

                for box, label in zip(
                    sample_boxes,
                    sample_labels,
                    strict=True,
                ):
                    class_id = int(label.item())

                    center_x = min(
                        width - 1,
                        max(0, int(float(box[0]) * width)),
                    )
                    center_y = min(
                        height - 1,
                        max(0, int(float(box[1]) * height)),
                    )

                    target_classes[
                        batch_index,
                        class_id,
                        center_y,
                        center_x,
                    ] = 1.0

                    target_box = box.to(
                        dtype=prediction.dtype,
                        device=prediction.device,
                    ).clone()

                    target_box[0] = target_box[0] * width - center_x
                    target_box[1] = target_box[1] * height - center_y

                    target_boxes[
                        batch_index,
                        :,
                        center_y,
                        center_x,
                    ] = target_box

                    positive[
                        batch_index,
                        center_y,
                        center_x,
                    ] = True

            predicted_classes = prediction[:, : self.num_classes]
            predicted_boxes = prediction[
                :,
                self.num_classes : self.num_classes + 4,
            ]

            elementwise_classification = functional.binary_cross_entropy_with_logits(
                predicted_classes,
                target_classes,
                reduction="none",
            )

            positive_classes = target_classes > 0
            negative_classes = ~positive_classes

            if positive_classes.any():
                positive_classification = elementwise_classification[positive_classes].mean()
            else:
                positive_classification = prediction.sum() * 0.0

            if negative_classes.any():
                negative_classification = elementwise_classification[negative_classes].mean()
            else:
                negative_classification = prediction.sum() * 0.0

            classification_loss = positive_classification + negative_classification

            if positive.any():
                predicted_positive_boxes = torch.sigmoid(
                    predicted_boxes.permute(0, 2, 3, 1)[positive]
                )
                target_positive_boxes = target_boxes.permute(
                    0,
                    2,
                    3,
                    1,
                )[positive]

                box_loss = functional.smooth_l1_loss(
                    predicted_positive_boxes,
                    target_positive_boxes,
                )
            else:
                box_loss = prediction.sum() * 0.0

            level_classification_losses.append(classification_loss)
            level_box_losses.append(box_loss)

        classification = torch.stack(level_classification_losses).mean()

        box = torch.stack(level_box_losses).mean()

        total = self.classification_weight * classification + self.box_weight * box

        return {
            "total": total,
            "classification": classification,
            "box": box,
        }

    def _validate_targets(
        self,
        boxes: Tensor,
        labels: Tensor,
        *,
        batch_index: int,
    ) -> None:
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise PairedDetectionLossError(
                f"boxes at batch {batch_index} must have shape [num_boxes, 4]"
            )

        if labels.ndim != 1:
            raise PairedDetectionLossError(
                f"labels at batch {batch_index} must have shape [num_boxes]"
            )

        if boxes.shape[0] != labels.shape[0]:
            raise PairedDetectionLossError(
                f"box and label counts differ at batch {batch_index}"
            )

        if not torch.isfinite(boxes).all():
            raise PairedDetectionLossError(
                f"boxes at batch {batch_index} must be finite"
            )

        if boxes.numel() > 0 and (
            torch.any(boxes < 0.0) or torch.any(boxes > 1.0)
        ):
            raise PairedDetectionLossError(
                f"boxes at batch {batch_index} must lie in [0, 1]"
            )

        if labels.numel() > 0 and (
            torch.any(labels < 0)
            or torch.any(labels >= self.num_classes)
        ):
            raise PairedDetectionLossError(
                f"class IDs at batch {batch_index} are out of range"
            )