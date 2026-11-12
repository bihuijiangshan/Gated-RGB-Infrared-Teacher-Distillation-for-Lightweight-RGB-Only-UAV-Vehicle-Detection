"""Ultralytics detection loss for the official gated teacher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from ultralytics.utils.loss import v8DetectionLoss

from rgbt_uav.models.official_gated_teacher import (
    OfficialGatedDualStreamTeacher,
)
from rgbt_uav.training.official_student_loss import (
    paired_batch_to_ultralytics_targets,
)


class OfficialGatedTeacherLossError(RuntimeError):
    """Raised when gated-teacher loss inputs are invalid."""


class _UltralyticsLossModelProxy:
    """Expose the model attributes required by v8DetectionLoss."""

    def __init__(
        self,
        teacher: OfficialGatedDualStreamTeacher,
    ) -> None:
        self.args = SimpleNamespace(
            box=float(teacher.args.box),
            cls=float(teacher.args.cls),
            dfl=float(teacher.args.dfl),
        )
        self.model = (
            teacher.detect,
        )
        self._teacher = teacher

    def parameters(
        self,
    ):
        return self._teacher.parameters()


def _validate_predictions(
    predictions: Sequence[Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(predictions, Sequence) or isinstance(
        predictions,
        (str, bytes, bytearray),
    ):
        raise OfficialGatedTeacherLossError(
            "predictions must be a sequence"
        )

    if len(predictions) != 3:
        raise OfficialGatedTeacherLossError(
            "predictions must contain three levels"
        )

    levels = tuple(predictions)

    if not all(
        isinstance(prediction, Tensor)
        and prediction.ndim == 4
        for prediction in levels
    ):
        raise OfficialGatedTeacherLossError(
            "prediction levels must be BCHW tensors"
        )

    first_device = levels[0].device
    first_dtype = levels[0].dtype
    batch_size = levels[0].shape[0]

    if batch_size <= 0:
        raise OfficialGatedTeacherLossError(
            "prediction batch must not be empty"
        )

    if not all(
        prediction.shape[0] == batch_size
        and prediction.device == first_device
        and prediction.dtype == first_dtype
        for prediction in levels
    ):
        raise OfficialGatedTeacherLossError(
            "prediction levels must share batch, device, and dtype"
        )

    if not all(
        torch.isfinite(prediction).all()
        for prediction in levels
    ):
        raise OfficialGatedTeacherLossError(
            "predictions must contain only finite values"
        )

    return levels


class OfficialGatedTeacherDetectionLoss(nn.Module):
    """Apply Ultralytics' native YOLOv8 loss to the gated teacher."""

    def __init__(
        self,
        teacher: OfficialGatedDualStreamTeacher,
    ) -> None:
        super().__init__()

        if not isinstance(
            teacher,
            OfficialGatedDualStreamTeacher,
        ):
            raise OfficialGatedTeacherLossError(
                "teacher must be an "
                "OfficialGatedDualStreamTeacher"
            )

        proxy = _UltralyticsLossModelProxy(
            teacher
        )

        try:
            self.loss = v8DetectionLoss(
                proxy
            )
        except (
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OfficialGatedTeacherLossError(
                "could not initialize official "
                "gated-teacher detection loss"
            ) from exc

    def forward(
        self,
        predictions: Sequence[Tensor],
        batch: Mapping[str, object],
    ) -> dict[str, Tensor]:
        levels = _validate_predictions(
            predictions
        )

        if not isinstance(batch, Mapping):
            raise OfficialGatedTeacherLossError(
                "batch must be a mapping"
            )

        targets = (
            paired_batch_to_ultralytics_targets(
                batch,
                device=levels[0].device,
            )
        )

        try:
            total, components = self.loss(
                list(levels),
                targets,
            )
        except (
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise OfficialGatedTeacherLossError(
                "official gated-teacher "
                "detection loss failed"
            ) from exc

        if (
            not isinstance(total, Tensor)
            or total.ndim != 0
            or not torch.isfinite(total)
        ):
            raise OfficialGatedTeacherLossError(
                "official detection total "
                "must be a finite scalar"
            )

        if (
            not isinstance(components, Tensor)
            or components.numel() != 3
            or not torch.isfinite(
                components
            ).all()
        ):
            raise OfficialGatedTeacherLossError(
                "official loss must return finite "
                "box, classification, and DFL components"
            )

        flat_components = (
            components.reshape(-1)
        )

        return {
            "total": total,
            "box": flat_components[0],
            "classification": (
                flat_components[1]
            ),
            "dfl": flat_components[2],
        }