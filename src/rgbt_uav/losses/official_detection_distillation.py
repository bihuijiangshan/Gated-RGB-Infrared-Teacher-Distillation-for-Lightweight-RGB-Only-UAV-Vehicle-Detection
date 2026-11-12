from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from rgbt_uav.losses.distillation import DistillationLoss


class OfficialDetectionDistillationLossError(ValueError):
    """Raised when official detection distillation inputs are invalid."""


class OfficialDetectionDistillationLoss(nn.Module):
    """Distill official YOLO classification and DFL predictions.

    Official YOLO detection outputs use the channel layout:

        [4 * reg_max DFL channels, num_classes classification channels]

    The DFL channels are four independent categorical distributions, while
    the classification channels are independent binary logits. They must
    therefore be distilled separately.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        reg_max: int,
        classification_weight: float = 1.0,
        dfl_weight: float = 1.0,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()

        if (
            not isinstance(num_classes, int)
            or isinstance(num_classes, bool)
            or num_classes <= 0
        ):
            raise OfficialDetectionDistillationLossError(
                "num_classes must be a positive integer"
            )

        if (
            not isinstance(reg_max, int)
            or isinstance(reg_max, bool)
            or reg_max <= 0
        ):
            raise OfficialDetectionDistillationLossError(
                "reg_max must be a positive integer"
            )

        if (
            not isinstance(classification_weight, (int, float))
            or isinstance(classification_weight, bool)
            or classification_weight < 0
        ):
            raise OfficialDetectionDistillationLossError(
                "classification_weight must be a non-negative number"
            )

        if (
            not isinstance(dfl_weight, (int, float))
            or isinstance(dfl_weight, bool)
            or dfl_weight < 0
        ):
            raise OfficialDetectionDistillationLossError(
                "dfl_weight must be a non-negative number"
            )

        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or temperature <= 0
        ):
            raise OfficialDetectionDistillationLossError(
                "temperature must be a positive number"
            )

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.classification_weight = float(
            classification_weight
        )
        self.dfl_weight = float(dfl_weight)
        self.temperature = float(temperature)

    @property
    def expected_channels(self) -> int:
        """Return the expected official detection output channels."""
        return 4 * self.reg_max + self.num_classes

    def _validate_predictions(
        self,
        values: Sequence[Tensor],
        *,
        label: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != 3
        ):
            raise OfficialDetectionDistillationLossError(
                f"{label} predictions must contain three levels"
            )

        levels: list[Tensor] = []

        for level_index, prediction in enumerate(values):
            if not isinstance(prediction, Tensor):
                raise OfficialDetectionDistillationLossError(
                    f"{label} prediction level {level_index} "
                    "must be a tensor"
                )

            if prediction.ndim != 4:
                raise OfficialDetectionDistillationLossError(
                    f"{label} prediction level {level_index} "
                    "must have shape [B, C, H, W]"
                )

            if prediction.shape[1] != self.expected_channels:
                raise OfficialDetectionDistillationLossError(
                    f"{label} prediction level {level_index} "
                    f"must contain {self.expected_channels} channels"
                )

            if not torch.isfinite(prediction).all():
                raise OfficialDetectionDistillationLossError(
                    f"{label} prediction level {level_index} "
                    "must contain only finite values"
                )

            levels.append(prediction)

        return levels[0], levels[1], levels[2]

    def _classification_kl(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
    ) -> Tensor:
        temperature = self.temperature

        teacher_scaled = teacher_logits.detach() / temperature
        student_scaled = student_logits / temperature

        teacher_probability = torch.sigmoid(teacher_scaled)

        teacher_log_probability = functional.logsigmoid(
            teacher_scaled
        )
        teacher_log_inverse_probability = functional.logsigmoid(
            -teacher_scaled
        )

        student_log_probability = functional.logsigmoid(
            student_scaled
        )
        student_log_inverse_probability = functional.logsigmoid(
            -student_scaled
        )

        positive_kl = teacher_probability * (
            teacher_log_probability
            - student_log_probability
        )
        negative_kl = (1.0 - teacher_probability) * (
            teacher_log_inverse_probability
            - student_log_inverse_probability
        )

        loss = (
            positive_kl + negative_kl
        ).mean() * temperature**2

        # Floating-point cancellation can produce an extremely small
        # negative value when teacher and student logits are identical.
        return loss.clamp_min(0.0)

    def _dfl_kl(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
    ) -> Tensor:
        batch, _, height, width = teacher_logits.shape
        temperature = self.temperature

        teacher_distribution = (
            teacher_logits.detach()
            .reshape(
                batch,
                4,
                self.reg_max,
                height,
                width,
            )
            / temperature
        )
        student_distribution = (
            student_logits.reshape(
                batch,
                4,
                self.reg_max,
                height,
                width,
            )
            / temperature
        )

        teacher_probability = functional.softmax(
            teacher_distribution,
            dim=2,
        )
        student_log_probability = functional.log_softmax(
            student_distribution,
            dim=2,
        )

        loss = (
            functional.kl_div(
                student_log_probability,
                teacher_probability,
                reduction="none",
            )
            .sum(dim=2)
            .mean()
            * temperature**2
        )

        return loss.clamp_min(0.0)

    def forward(
        self,
        *,
        teacher_predictions: Sequence[Tensor],
        student_predictions: Sequence[Tensor],
    ) -> dict[str, Tensor]:
        teacher_levels = self._validate_predictions(
            teacher_predictions,
            label="teacher",
        )
        student_levels = self._validate_predictions(
            student_predictions,
            label="student",
        )

        classification_losses: list[Tensor] = []
        dfl_losses: list[Tensor] = []

        regression_channels = 4 * self.reg_max

        for level_index, (
            teacher_prediction,
            student_prediction,
        ) in enumerate(
            zip(
                teacher_levels,
                student_levels,
                strict=True,
            )
        ):
            if teacher_prediction.shape != student_prediction.shape:
                raise OfficialDetectionDistillationLossError(
                    "teacher and student prediction level "
                    f"{level_index} must have matching shapes"
                )

            if teacher_prediction.device != student_prediction.device:
                raise OfficialDetectionDistillationLossError(
                    "teacher and student prediction level "
                    f"{level_index} must use the same device"
                )

            if teacher_prediction.dtype != student_prediction.dtype:
                raise OfficialDetectionDistillationLossError(
                    "teacher and student prediction level "
                    f"{level_index} must use the same dtype"
                )

            teacher_dfl = teacher_prediction[
                :,
                :regression_channels,
            ]
            student_dfl = student_prediction[
                :,
                :regression_channels,
            ]

            teacher_classification = teacher_prediction[
                :,
                regression_channels:,
            ]
            student_classification = student_prediction[
                :,
                regression_channels:,
            ]

            classification_losses.append(
                self._classification_kl(
                    teacher_classification,
                    student_classification,
                )
            )
            dfl_losses.append(
                self._dfl_kl(
                    teacher_dfl,
                    student_dfl,
                )
            )

        classification_loss = torch.stack(
            classification_losses
        ).mean()
        dfl_loss = torch.stack(dfl_losses).mean()

        total = (
            self.classification_weight
            * classification_loss
            + self.dfl_weight
            * dfl_loss
        )

        return {
            "total": total,
            "classification": classification_loss,
            "dfl": dfl_loss,
        }


class OfficialTeacherStudentDistillationLoss(nn.Module):
    """Compose feature and detection-aware output distillation.

    This adapter preserves the existing distillation schema:

        total
        feature
        logit

    The ``logit`` value now represents the sum of classification-logit
    distillation and DFL distribution distillation. It no longer applies
    one softmax across all official YOLO output channels.
    """

    def __init__(
        self,
        *,
        feature_weight: float = 1.0,
        logit_weight: float = 1.0,
        temperature: float = 2.0,
        num_classes: int = 5,
        reg_max: int = 16,
        classification_distillation_weight: (
            float | None
        ) = None,
        dfl_distillation_weight: float | None = None,
    ) -> None:
        super().__init__()

        if (
            not isinstance(feature_weight, (int, float))
            or isinstance(feature_weight, bool)
            or feature_weight < 0
        ):
            raise OfficialDetectionDistillationLossError(
                "feature_weight must be a non-negative number"
            )

        if (
            not isinstance(logit_weight, (int, float))
            or isinstance(logit_weight, bool)
            or logit_weight < 0
        ):
            raise OfficialDetectionDistillationLossError(
                "logit_weight must be a non-negative number"
            )

        classification_weight = (
            float(logit_weight)
            if classification_distillation_weight is None
            else classification_distillation_weight
        )
        dfl_weight = (
            float(logit_weight)
            if dfl_distillation_weight is None
            else dfl_distillation_weight
        )

        self.feature_weight = float(feature_weight)
        self.logit_weight = float(logit_weight)
        self.classification_distillation_weight = float(
            classification_weight
        )
        self.dfl_distillation_weight = float(
            dfl_weight
        )

        self.feature_criterion = DistillationLoss(
            feature_weight=self.feature_weight,
            logit_weight=0.0,
            temperature=temperature,
        )
        self.detection_criterion = (
            OfficialDetectionDistillationLoss(
                num_classes=num_classes,
                reg_max=reg_max,
                classification_weight=(
                    self.classification_distillation_weight
                ),
                dfl_weight=(
                    self.dfl_distillation_weight
                ),
                temperature=temperature,
            )
        )

    def forward(
        self,
        teacher_features: Sequence[Tensor],
        student_features: Sequence[Tensor],
        teacher_predictions: Sequence[Tensor],
        student_predictions: Sequence[Tensor],
    ) -> dict[str, Tensor]:
        # This composite receives [classes, DFL] from the teacher/student
        # adapters. The underlying detection-aware loss expects [DFL, classes].
        teacher_levels = self.detection_criterion._validate_predictions(
            teacher_predictions,
            label="teacher",
        )
        student_levels = self.detection_criterion._validate_predictions(
            student_predictions,
            label="student",
        )
        num_classes = self.detection_criterion.num_classes

        teacher_native = tuple(
            torch.cat(
                (prediction[:, num_classes:], prediction[:, :num_classes]),
                dim=1,
            )
            for prediction in teacher_levels
        )
        student_native = tuple(
            torch.cat(
                (prediction[:, num_classes:], prediction[:, :num_classes]),
                dim=1,
            )
            for prediction in student_levels
        )

        detection_result = self.detection_criterion(
            teacher_predictions=teacher_native,
            student_predictions=student_native,
        )
        output_total = detection_result["total"]

        if self.feature_weight == 0.0:
            feature_loss = output_total.new_zeros(())
            feature_total = output_total.new_zeros(())
        else:
            feature_result = self.feature_criterion(
                teacher_features,
                student_features,
                teacher_predictions,
                student_predictions,
            )

            if set(feature_result) != {
                "total",
                "feature",
                "logit",
            }:
                raise OfficialDetectionDistillationLossError(
                    "feature criterion returned an invalid schema"
                )

            feature_loss = feature_result["feature"]
            feature_total = feature_result["total"]

        total = feature_total + output_total

        if total.ndim != 0 or not torch.isfinite(total):
            raise OfficialDetectionDistillationLossError(
                "combined distillation loss must be a finite scalar"
            )

        return {
            "total": total,
            "feature": feature_loss,
            "logit": (detection_result["classification"] + detection_result["dfl"]),
        }