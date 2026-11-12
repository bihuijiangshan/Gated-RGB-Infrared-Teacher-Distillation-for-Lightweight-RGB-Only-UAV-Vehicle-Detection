from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rgbt_uav.training.early_fusion_runtime import (
    build_early_fusion_model,
)


class OfficialDistillationTeacherError(RuntimeError):
    """Raised when the official early-fusion teacher output is invalid."""


class OfficialEarlyFusionTeacher(nn.Module):
    """Frozen official YOLOv8 early-fusion teacher with pyramid exposure."""

    def __init__(
        self,
        model: nn.Module,
        *,
        num_classes: int,
    ) -> None:
        super().__init__()

        if not isinstance(model, nn.Module):
            raise OfficialDistillationTeacherError(
                "model must be a torch.nn.Module"
            )

        model_layers = getattr(model, "model", None)

        if (
            not isinstance(model_layers, (nn.Sequential, nn.ModuleList))
            or len(model_layers) == 0
        ):
            raise OfficialDistillationTeacherError(
                "official model must expose a non-empty model sequence"
            )

        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 0
        ):
            raise OfficialDistillationTeacherError(
                "num_classes must be a positive integer"
            )

        self.num_classes = num_classes

        self.model = model
        self._features: list[Tensor] | None = None

        detect = model_layers[-1]
        detect.register_forward_pre_hook(
            self._capture_detect_inputs
        )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.eval()

    def _capture_detect_inputs(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
    ) -> None:
        del module

        if len(args) != 1:
            self._features = None
            return

        features = args[0]

        if not isinstance(features, (list, tuple)):
            self._features = None
            return

        if not all(
            isinstance(value, Tensor)
            for value in features
        ):
            self._features = None
            return

        self._features = list(features)

    def train(
        self,
        mode: bool = True,
    ) -> OfficialEarlyFusionTeacher:
        """Keep the wrapped teacher in evaluation mode."""

        del mode
        super().train(False)
        self.model.train(False)
        return self

    def forward(
        self,
        rgb: Tensor,
        ir: Tensor,
    ) -> dict[str, list[Tensor]]:
        if (
            not isinstance(rgb, Tensor)
            or not isinstance(ir, Tensor)
            or rgb.ndim != 4
            or ir.ndim != 4
        ):
            raise OfficialDistillationTeacherError(
                "rgb and ir must be BCHW tensors"
            )

        if (
            rgb.shape[0] != ir.shape[0]
            or rgb.shape[2:] != ir.shape[2:]
            or rgb.shape[1] != 3
            or ir.shape[1] != 1
        ):
            raise OfficialDistillationTeacherError(
                "expected aligned 3-channel RGB and 1-channel IR"
            )

        self._features = None

        fused = torch.cat(
            (
                rgb,
                ir.to(
                    device=rgb.device,
                    dtype=rgb.dtype,
                ),
            ),
            dim=1,
        )

        output = self.model(fused)
        raw_predictions = reorder_official_raw_predictions(
            _raw_predictions(output),
            num_classes=self.num_classes,
        )

        features = self._features

        if features is None:
            raise OfficialDistillationTeacherError(
                "official Detect inputs were not captured"
            )

        if len(features) != len(raw_predictions):
            raise OfficialDistillationTeacherError(
                "teacher feature and prediction level counts differ"
            )

        return {
            "features": features,
            "predictions": raw_predictions,
        }


def _raw_predictions(output: object) -> list[Tensor]:
    if not isinstance(output, tuple) or len(output) != 2:
        raise OfficialDistillationTeacherError(
            "official eval output must be a decoded/raw tuple"
        )

    raw = output[1]

    if isinstance(raw, Mapping):
        raw = raw.get("one2many")

    if not isinstance(raw, Sequence) or isinstance(
        raw,
        (str, bytes, bytearray),
    ):
        raise OfficialDistillationTeacherError(
            "official raw predictions must be a sequence"
        )

    predictions = list(raw)

    if not predictions or not all(
        isinstance(value, Tensor) and value.ndim == 4
        for value in predictions
    ):
        raise OfficialDistillationTeacherError(
            "official raw prediction levels must be BCHW tensors"
        )

    return predictions


def build_official_early_fusion_teacher(
    *,
    model_name: str,
    num_classes: int,
    device: torch.device | str,
) -> OfficialEarlyFusionTeacher:
    """Build a frozen four-channel official YOLOv8 teacher."""

    target_device = torch.device(device)

    model = build_early_fusion_model(
        model_name=model_name,
        num_classes=num_classes,
        device=target_device,
    )

    teacher = OfficialEarlyFusionTeacher(
        model,
        num_classes=num_classes,
    )
    teacher.to(target_device)
    teacher.eval()

    return teacher

def _checkpoint_metrics(
    value: object,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise OfficialDistillationTeacherError(
            "checkpoint metrics must be a mapping"
        )

    metrics: dict[str, float] = {}

    for key, metric in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
        ):
            raise OfficialDistillationTeacherError(
                f"checkpoint metric {key!r} must be finite and numeric"
            )

        metrics[key] = float(metric)

    return metrics


def load_official_early_fusion_teacher_checkpoint(
    path: Path,
    *,
    teacher: OfficialEarlyFusionTeacher,
    map_location: torch.device | str | None = None,
) -> dict[str, object]:
    """Load an official early-fusion checkpoint into a frozen teacher."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise OfficialDistillationTeacherError(
            f"teacher checkpoint does not exist: {checkpoint_path}"
        )

    if not isinstance(teacher, OfficialEarlyFusionTeacher):
        raise OfficialDistillationTeacherError(
            "teacher must be an OfficialEarlyFusionTeacher"
        )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise OfficialDistillationTeacherError(
            f"could not load teacher checkpoint: {checkpoint_path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise OfficialDistillationTeacherError(
            "teacher checkpoint root must be a mapping"
        )

    epoch = payload.get("epoch")

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise OfficialDistillationTeacherError(
            "teacher checkpoint epoch must be a non-negative integer"
        )

    model_state = payload.get("model")

    if not isinstance(model_state, Mapping):
        raise OfficialDistillationTeacherError(
            "teacher checkpoint model must be a state mapping"
        )

    metrics = _checkpoint_metrics(
        payload.get("metrics")
    )

    try:
        teacher.model.load_state_dict(
            model_state,
            strict=True,
        )
    except Exception as exc:
        raise OfficialDistillationTeacherError(
            "teacher checkpoint model state is incompatible"
        ) from exc

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    teacher.eval()

    return {
        "epoch": epoch,
        "metrics": metrics,
    }

def reorder_official_raw_predictions(
    predictions: Sequence[Tensor],
    *,
    num_classes: int,
) -> list[Tensor]:
    """Convert official [DFL, classes] outputs to [classes, DFL]."""

    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise OfficialDistillationTeacherError(
            "num_classes must be a positive integer"
        )

    reordered: list[Tensor] = []

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Tensor) or prediction.ndim != 4:
            raise OfficialDistillationTeacherError(
                f"prediction level {index} must be a BCHW tensor"
            )

        if prediction.shape[1] <= num_classes:
            raise OfficialDistillationTeacherError(
                f"prediction level {index} does not contain regression channels"
            )

        regression_channels = prediction.shape[1] - num_classes

        reordered.append(
            torch.cat(
                (
                    prediction[
                        :,
                        regression_channels:,
                    ],
                    prediction[
                        :,
                        :regression_channels,
                    ],
                ),
                dim=1,
            )
        )

    if not reordered:
        raise OfficialDistillationTeacherError(
            "predictions must not be empty"
        )

    return reordered