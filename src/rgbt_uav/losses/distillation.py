from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class FeatureDistiller(nn.Module):
    """Project student pyramid features and match detached teacher features."""

    def __init__(self, *, student_channels: list[int], teacher_channels: list[int]) -> None:
        super().__init__()
        if len(student_channels) != len(teacher_channels) or not student_channels:
            raise ValueError("student and teacher channel lists must have equal non-zero length")
        self.adapters = nn.ModuleList(
            nn.Conv2d(student, teacher, kernel_size=1, bias=False)
            for student, teacher in zip(student_channels, teacher_channels, strict=True)
        )

    def forward(self, student_features: list[Tensor], teacher_features: list[Tensor]) -> Tensor:
        if len(student_features) != len(self.adapters) or len(teacher_features) != len(
            self.adapters
        ):
            raise ValueError("feature levels must match configured adapters")
        losses: list[Tensor] = []
        for adapter, student, teacher in zip(
            self.adapters, student_features, teacher_features, strict=True
        ):
            projected = adapter(student)
            target = teacher.detach()
            if projected.shape != target.shape:
                raise ValueError("projected student and teacher feature shapes must match")
            projected = functional.normalize(projected, p=2, dim=1)
            target = functional.normalize(target, p=2, dim=1)
            losses.append(functional.mse_loss(projected, target))
        return torch.stack(losses).mean()


class DetectionDistillationLoss(nn.Module):
    """KL classification and Smooth-L1 localization distillation on raw pyramid outputs."""

    def __init__(
        self,
        *,
        num_classes: int,
        temperature: float = 2.0,
        classification_weight: float = 1.0,
        localization_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or temperature <= 0:
            raise ValueError("num_classes and temperature must be positive")
        self.num_classes = num_classes
        self.temperature = temperature
        self.classification_weight = classification_weight
        self.localization_weight = localization_weight

    def forward(
        self, student_predictions: list[Tensor], teacher_predictions: list[Tensor]
    ) -> dict[str, Tensor]:
        if len(student_predictions) != len(teacher_predictions) or not student_predictions:
            raise ValueError("student and teacher prediction levels must match and be non-empty")
        classification_losses: list[Tensor] = []
        localization_losses: list[Tensor] = []
        for student, teacher in zip(student_predictions, teacher_predictions, strict=True):
            target = teacher.detach()
            if student.shape != target.shape:
                raise ValueError("student and teacher prediction shapes must match")
            student_class = student[:, : self.num_classes]
            teacher_class = target[:, : self.num_classes]
            student_box = student[:, self.num_classes :]
            teacher_box = target[:, self.num_classes :]
            classification_losses.append(
                functional.kl_div(
                    functional.log_softmax(student_class / self.temperature, dim=1),
                    functional.softmax(teacher_class / self.temperature, dim=1),
                    reduction="batchmean",
                )
                * (self.temperature**2)
            )
            localization_losses.append(functional.smooth_l1_loss(student_box, teacher_box))

        classification = torch.stack(classification_losses).mean()
        localization = torch.stack(localization_losses).mean()
        total = (
            self.classification_weight * classification
            + self.localization_weight * localization
        )
        return {
            "classification": classification,
            "localization": localization,
            "total": total,
        }

class DistillationLossError(ValueError):
    """Raised when unified distillation inputs are invalid."""


class DistillationLoss(nn.Module):
    """Feature and logit distillation loss for aligned levels."""

    def __init__(
        self,
        *,
        feature_weight: float = 1.0,
        logit_weight: float = 1.0,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()

        if (
            isinstance(feature_weight, bool)
            or not isinstance(feature_weight, (int, float))
            or not torch.isfinite(torch.tensor(float(feature_weight)))
            or feature_weight < 0
        ):
            raise DistillationLossError(
                "feature_weight must be a finite non-negative number"
            )

        if (
            isinstance(logit_weight, bool)
            or not isinstance(logit_weight, (int, float))
            or not torch.isfinite(torch.tensor(float(logit_weight)))
            or logit_weight < 0
        ):
            raise DistillationLossError(
                "logit_weight must be a finite non-negative number"
            )

        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not torch.isfinite(torch.tensor(float(temperature)))
            or temperature <= 0
        ):
            raise DistillationLossError(
                "temperature must be positive"
            )

        self.feature_weight = float(feature_weight)
        self.logit_weight = float(logit_weight)
        self.temperature = float(temperature)

    @staticmethod
    def _validate_levels(
        teacher: list[Tensor] | tuple[Tensor, ...],
        student: list[Tensor] | tuple[Tensor, ...],
        field: str,
    ) -> None:
        if (
            not isinstance(teacher, (list, tuple))
            or not isinstance(student, (list, tuple))
            or not teacher
            or len(teacher) != len(student)
        ):
            raise DistillationLossError(
                f"teacher and student {field} level counts must match"
            )

        for index, (teacher_value, student_value) in enumerate(
            zip(teacher, student, strict=True)
        ):
            if not isinstance(teacher_value, Tensor):
                raise DistillationLossError(
                    f"teacher {field}[{index}] must be a tensor"
                )
            if not isinstance(student_value, Tensor):
                raise DistillationLossError(
                    f"student {field}[{index}] must be a tensor"
                )
            if teacher_value.shape != student_value.shape:
                raise DistillationLossError(
                    f"teacher and student {field}[{index}] shapes must match"
                )

    def forward(
        self,
        teacher_features: list[Tensor] | tuple[Tensor, ...],
        student_features: list[Tensor] | tuple[Tensor, ...],
        teacher_logits: list[Tensor] | tuple[Tensor, ...],
        student_logits: list[Tensor] | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        self._validate_levels(
            teacher_features,
            student_features,
            "features",
        )
        self._validate_levels(
            teacher_logits,
            student_logits,
            "logits",
        )

        feature_loss = torch.stack(
            [
                functional.mse_loss(
                    student,
                    teacher.detach(),
                )
                for teacher, student in zip(
                    teacher_features,
                    student_features,
                    strict=True,
                )
            ]
        ).mean()

        temperature = self.temperature
        logit_loss = torch.stack(
            [
                functional.kl_div(
                    functional.log_softmax(
                        student / temperature,
                        dim=1,
                    ),
                    functional.softmax(
                        teacher.detach() / temperature,
                        dim=1,
                    ),
                    reduction="none",
                )
                .sum(dim=1)
                .mean()
                * temperature**2
                for teacher, student in zip(
                    teacher_logits,
                    student_logits,
                    strict=True,
                )
            ]
        ).mean()

        total = (
            self.feature_weight * feature_loss
            + self.logit_weight * logit_loss
        )

        return {
            "total": total,
            "feature": feature_loss,
            "logit": logit_loss,
        }
