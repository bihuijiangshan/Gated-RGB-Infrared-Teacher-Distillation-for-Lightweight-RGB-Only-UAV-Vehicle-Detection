"""Official YOLO dual-stream teacher with gated pyramid fusion."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from ultralytics.nn.tasks import DetectionModel

from rgbt_uav.models.fusion import (
    ReliabilityGatedFusion,
)


class OfficialGatedTeacherError(ValueError):
    """Raised when the gated teacher contract is violated."""


def _validate_modalities(
    rgb: Tensor,
    ir: Tensor,
) -> None:
    if (
        not isinstance(rgb, Tensor)
        or not isinstance(ir, Tensor)
        or rgb.ndim != 4
        or ir.ndim != 4
    ):
        raise OfficialGatedTeacherError(
            "RGB and IR must be BCHW tensors"
        )

    if rgb.shape[1] != 3:
        raise OfficialGatedTeacherError(
            "RGB input must contain three channels"
        )

    if ir.shape[1] != 1:
        raise OfficialGatedTeacherError(
            "IR input must contain one channel"
        )

    if (
        rgb.shape[0] != ir.shape[0]
        or rgb.shape[2:] != ir.shape[2:]
    ):
        raise OfficialGatedTeacherError(
            "RGB and IR inputs must be aligned"
        )


def _rgb_quality(rgb: Tensor) -> Tensor:
    mean = rgb.mean(
        dim=(1, 2, 3)
    )
    contrast = rgb.std(
        dim=(1, 2, 3),
        unbiased=False,
    )
    return torch.stack(
        (
            mean,
            contrast,
        ),
        dim=1,
    )


def _routed_value(
    source: object,
    *,
    current: object,
    saved: list[object],
) -> object:
    if source == -1:
        return current

    if isinstance(source, int):
        try:
            value = saved[source]
        except IndexError as exc:
            raise OfficialGatedTeacherError(
                f"invalid official route index: {source}"
            ) from exc

        if value is None:
            raise OfficialGatedTeacherError(
                f"official route {source} "
                "was not retained"
            )
        return value

    if isinstance(source, Sequence) and not isinstance(
        source,
        (str, bytes, bytearray),
    ):
        values: list[object] = []

        for index in source:
            if not isinstance(index, int):
                raise OfficialGatedTeacherError(
                    "official route indices "
                    "must be integers"
                )

            if index == -1:
                values.append(current)
                continue

            try:
                value = saved[index]
            except IndexError as exc:
                raise OfficialGatedTeacherError(
                    "invalid official route "
                    f"index: {index}"
                ) from exc

            if value is None:
                raise OfficialGatedTeacherError(
                    f"official route {index} "
                    "was not retained"
                )

            values.append(value)

        return values

    raise OfficialGatedTeacherError(
        f"unsupported official route: {source!r}"
    )


def _extract_pyramid(
    stream: DetectionModel,
    image: Tensor,
    *,
    detect_from: object,
) -> list[Tensor]:
    current: object = image
    saved: list[object] = []

    for module in stream.model:
        source = getattr(module, "f", -1)
        routed = _routed_value(
            source,
            current=current,
            saved=saved,
        )
        current = module(routed)

        module_index = getattr(
            module,
            "i",
            len(saved),
        )
        saved.append(
            current
            if module_index in stream.save
            else None
        )

    routed_features = _routed_value(
        detect_from,
        current=current,
        saved=saved,
    )

    if not isinstance(
        routed_features,
        Sequence,
    ) or isinstance(
        routed_features,
        (str, bytes, bytearray),
    ):
        raise OfficialGatedTeacherError(
            "official Detect route must "
            "produce a feature sequence"
        )

    features = list(routed_features)

    if (
        len(features) != 3
        or not all(
            isinstance(feature, Tensor)
            and feature.ndim == 4
            for feature in features
        )
    ):
        raise OfficialGatedTeacherError(
            "official stream must expose "
            "three BCHW pyramid features"
        )

    return features


def _detect_input_channels(
    detect: nn.Module,
) -> list[int]:
    regression_heads = getattr(
        detect,
        "cv2",
        None,
    )

    if not isinstance(
        regression_heads,
        nn.ModuleList,
    ) or len(regression_heads) != 3:
        raise OfficialGatedTeacherError(
            "official Detect must expose "
            "three regression heads"
        )

    channels: list[int] = []

    for index, head in enumerate(
        regression_heads
    ):
        try:
            convolution = head[0].conv
            value = convolution.in_channels
        except (
            AttributeError,
            IndexError,
            TypeError,
        ) as exc:
            raise OfficialGatedTeacherError(
                "could not determine Detect "
                f"input channels at level {index}"
            ) from exc

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise OfficialGatedTeacherError(
                "Detect input channels "
                "must be positive integers"
            )

        channels.append(value)

    return channels


def _native_predictions(
    output: object,
) -> list[Tensor]:
    raw: object = output

    if (
        isinstance(output, tuple)
        and len(output) == 2
    ):
        raw = output[1]

    if not isinstance(raw, Sequence) or isinstance(
        raw,
        (str, bytes, bytearray),
    ):
        raise OfficialGatedTeacherError(
            "official Detect did not return "
            "raw prediction levels"
        )

    predictions = list(raw)

    if (
        len(predictions) != 3
        or not all(
            isinstance(prediction, Tensor)
            and prediction.ndim == 4
            for prediction in predictions
        )
    ):
        raise OfficialGatedTeacherError(
            "official Detect must return "
            "three BCHW prediction levels"
        )

    return predictions


def _reorder_predictions(
    predictions: Sequence[Tensor],
    *,
    num_classes: int,
) -> list[Tensor]:
    reordered: list[Tensor] = []

    for index, prediction in enumerate(
        predictions
    ):
        regression_channels = (
            prediction.shape[1]
            - num_classes
        )

        if regression_channels <= 0:
            raise OfficialGatedTeacherError(
                f"prediction level {index} "
                "does not contain regression channels"
            )

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

    return reordered


class OfficialGatedDualStreamTeacher(nn.Module):
    """Dual official YOLO streams fused before one Detect head."""

    def __init__(
        self,
        *,
        rgb_stream: DetectionModel,
        ir_stream: DetectionModel,
        detect: nn.Module,
        detect_from: object,
        num_classes: int,
    ) -> None:
        super().__init__()

        if not isinstance(
            rgb_stream,
            DetectionModel,
        ):
            raise OfficialGatedTeacherError(
                "rgb_stream must be a DetectionModel"
            )
        if not isinstance(
            ir_stream,
            DetectionModel,
        ):
            raise OfficialGatedTeacherError(
                "ir_stream must be a DetectionModel"
            )
        if not isinstance(detect, nn.Module):
            raise OfficialGatedTeacherError(
                "detect must be a torch module"
            )
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes <= 0
        ):
            raise OfficialGatedTeacherError(
                "num_classes must be positive"
            )

        channels = _detect_input_channels(
            detect
        )

        self.rgb_stream = rgb_stream
        self.ir_stream = ir_stream
        self.detect = detect
        self.detect_from = detect_from
        self.num_classes = num_classes

        self.fusions = nn.ModuleList(
            ReliabilityGatedFusion(channel)
            for channel in channels
        )

        self.args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )

    def forward(
        self,
        rgb: Tensor,
        ir: Tensor,
    ) -> dict[str, list[Tensor]]:
        _validate_modalities(
            rgb,
            ir,
        )

        ir = ir.to(
            device=rgb.device,
            dtype=rgb.dtype,
        )

        rgb_features = _extract_pyramid(
            self.rgb_stream,
            rgb,
            detect_from=self.detect_from,
        )
        ir_features = _extract_pyramid(
            self.ir_stream,
            ir,
            detect_from=self.detect_from,
        )

        quality = _rgb_quality(rgb)
        fused_features: list[Tensor] = []
        gates: list[Tensor] = []

        for (
            fusion,
            rgb_feature,
            ir_feature,
        ) in zip(
            self.fusions,
            rgb_features,
            ir_features,
            strict=True,
        ):
            if (
                rgb_feature.shape
                != ir_feature.shape
            ):
                raise OfficialGatedTeacherError(
                    "RGB and IR pyramid "
                    "features are not aligned"
                )

            fused, weights = fusion(
                rgb_feature,
                ir_feature,
                quality,
            )
            fused_features.append(fused)
            gates.append(weights)

        native_predictions = (
            _native_predictions(
                self.detect(
                    list(fused_features)
                )
            )
        )
        predictions = _reorder_predictions(
            native_predictions,
            num_classes=self.num_classes,
        )

        return {
            "features": fused_features,
            "gates": gates,
            "native_predictions": (
                native_predictions
            ),
            "predictions": predictions,
        }


def _build_stream(
    *,
    model_name: str,
    channels: int,
    num_classes: int,
) -> DetectionModel:
    try:
        model = DetectionModel(
            model_name,
            ch=channels,
            nc=num_classes,
            verbose=False,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherError(
            "could not construct official "
            f"{channels}-channel stream"
        ) from exc

    model.nc = num_classes
    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    return model


def build_official_gated_teacher(
    *,
    model_name: str,
    num_classes: int,
    device: torch.device | str,
) -> OfficialGatedDualStreamTeacher:
    """Build a trainable gated dual-stream official teacher."""

    if (
        not isinstance(model_name, str)
        or not model_name.strip()
    ):
        raise OfficialGatedTeacherError(
            "model_name must be a non-empty string"
        )

    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise OfficialGatedTeacherError(
            "num_classes must be positive"
        )

    target_device = torch.device(device)

    rgb_stream = _build_stream(
        model_name=model_name.strip(),
        channels=3,
        num_classes=num_classes,
    )
    ir_stream = _build_stream(
        model_name=model_name.strip(),
        channels=1,
        num_classes=num_classes,
    )

    rgb_detect = rgb_stream.model[-1]
    ir_detect = ir_stream.model[-1]

    detect_from = getattr(
        rgb_detect,
        "f",
        None,
    )
    ir_detect_from = getattr(
        ir_detect,
        "f",
        None,
    )

    if (
        detect_from is None
        or detect_from != ir_detect_from
    ):
        raise OfficialGatedTeacherError(
            "RGB and IR Detect routes differ"
        )

    rgb_stream.model[-1] = nn.Identity()
    ir_stream.model[-1] = nn.Identity()

    teacher = OfficialGatedDualStreamTeacher(
        rgb_stream=rgb_stream,
        ir_stream=ir_stream,
        detect=rgb_detect,
        detect_from=detect_from,
        num_classes=num_classes,
    )

    try:
        teacher.to(target_device)
    except (RuntimeError, TypeError) as exc:
        raise OfficialGatedTeacherError(
            "could not move gated teacher "
            f"to device: {target_device}"
        ) from exc

    teacher.train()

    for parameter in teacher.parameters():
        parameter.requires_grad_(True)

    return teacher