"""Inference adapter for the official gated dual-stream teacher."""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from typing import Any

import torch
from torch import Tensor, nn

from rgbt_uav.evaluation.paired import (
    ultralytics_decoded_to_paired_predictions,
)


class OfficialGatedTeacherInferenceError(
    RuntimeError
):
    """Raised when gated-teacher inference fails."""


def _pair_ids(
    batch: Mapping[str, object],
    *,
    batch_index: int,
    batch_size: int,
) -> tuple[str, ...]:
    value = batch.get(
        "pair_ids"
    )

    if value is None:
        value = batch.get(
            "pair_id"
        )

    if (
        isinstance(
            value,
            (str, bytes, Mapping),
        )
        or not isinstance(
            value,
            Iterable,
        )
    ):
        raise OfficialGatedTeacherInferenceError(
            f"batch {batch_index} pair_ids "
            "must be iterable"
        )

    try:
        normalized = tuple(value)
    except TypeError as exc:
        raise OfficialGatedTeacherInferenceError(
            f"batch {batch_index} pair_ids "
            "cannot be normalized"
        ) from exc

    if (
        len(normalized) != batch_size
        or not all(
            isinstance(pair_id, str)
            and pair_id
            for pair_id in normalized
        )
    ):
        raise OfficialGatedTeacherInferenceError(
            f"batch {batch_index} pair ids "
            "do not match the batch"
        )

    return normalized


def _image(
    batch: Mapping[str, object],
    key: str,
    *,
    channels: int,
    batch_index: int,
    device: torch.device,
) -> Tensor:
    value = batch.get(key)

    if not isinstance(value, Tensor):
        raise OfficialGatedTeacherInferenceError(
            f"batch {batch_index} {key} "
            "must be a tensor"
        )

    if (
        value.ndim != 4
        or value.shape[1] != channels
    ):
        raise OfficialGatedTeacherInferenceError(
            f"batch {batch_index} {key} "
            "has an invalid image shape"
        )

    if not torch.isfinite(value).all():
        raise OfficialGatedTeacherInferenceError(
            f"batch {batch_index} {key} "
            "must contain finite values"
        )

    return value.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


def _features(
    output: object,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(output, Mapping):
        raise OfficialGatedTeacherInferenceError(
            "gated teacher must return "
            "a mapping"
        )

    value = output.get(
        "features"
    )

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise OfficialGatedTeacherInferenceError(
            "teacher features must "
            "be a sequence"
        )

    levels = tuple(value)

    if (
        len(levels) != 3
        or not all(
            isinstance(level, Tensor)
            and level.ndim == 4
            for level in levels
        )
    ):
        raise OfficialGatedTeacherInferenceError(
            "teacher must expose three "
            "BCHW feature levels"
        )

    return levels  # type: ignore[return-value]


def _decoded_predictions(
    detect: nn.Module,
    features: tuple[
        Tensor,
        Tensor,
        Tensor,
    ],
) -> Tensor:
    try:
        output = detect(
            list(features)
        )
    except (
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherInferenceError(
            "shared Detect inference failed"
        ) from exc

    if (
        not isinstance(output, tuple)
        or len(output) != 2
    ):
        raise OfficialGatedTeacherInferenceError(
            "shared Detect must return "
            "decoded and raw predictions"
        )

    decoded = output[0]

    if (
        not isinstance(decoded, Tensor)
        or decoded.ndim != 3
    ):
        raise OfficialGatedTeacherInferenceError(
            "decoded predictions must "
            "have three dimensions"
        )

    return decoded


def run_official_gated_teacher_inference(
    teacher: nn.Module,
    loader: Iterable[
        Mapping[str, object]
    ],
    *,
    device: torch.device,
    score_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> list[dict[str, Any]]:
    """Run paired RGB/IR inference with the gated teacher."""

    if not isinstance(teacher, nn.Module):
        raise OfficialGatedTeacherInferenceError(
            "teacher must be a torch module"
        )

    if not isinstance(
        device,
        torch.device,
    ):
        raise OfficialGatedTeacherInferenceError(
            "device must be a torch.device"
        )

    detect = getattr(
        teacher,
        "detect",
        None,
    )

    if not isinstance(detect, nn.Module):
        raise OfficialGatedTeacherInferenceError(
            "teacher must expose "
            "a shared Detect module"
        )

    try:
        teacher.to(device)
    except (
        RuntimeError,
        TypeError,
    ) as exc:
        raise OfficialGatedTeacherInferenceError(
            "could not move teacher "
            "to device"
        ) from exc

    teacher.eval()
    detect.eval()

    results: list[
        dict[str, Any]
    ] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(
            loader
        ):
            if not isinstance(
                batch,
                Mapping,
            ):
                raise (
                    OfficialGatedTeacherInferenceError(
                        f"batch {batch_index} "
                        "must be a mapping"
                    )
                )

            rgb = _image(
                batch,
                "rgb",
                channels=3,
                batch_index=batch_index,
                device=device,
            )
            ir = _image(
                batch,
                "ir",
                channels=1,
                batch_index=batch_index,
                device=device,
            )

            if (
                rgb.shape[0] != ir.shape[0]
                or rgb.shape[2:]
                != ir.shape[2:]
            ):
                raise (
                    OfficialGatedTeacherInferenceError(
                        f"batch {batch_index} "
                        "modalities must be aligned"
                    )
                )

            normalized_pair_ids = (
                _pair_ids(
                    batch,
                    batch_index=batch_index,
                    batch_size=rgb.shape[0],
                )
            )

            try:
                output = teacher(
                    rgb,
                    ir,
                )
                features = _features(
                    output
                )
                decoded = (
                    _decoded_predictions(
                        detect,
                        features,
                    )
                )
                batch_results = (
                    ultralytics_decoded_to_paired_predictions(
                        decoded,
                        pair_ids=(
                            normalized_pair_ids
                        ),
                        score_threshold=(
                            score_threshold
                        ),
                        iou_threshold=(
                            iou_threshold
                        ),
                        max_det=max_det,
                    )
                )
            except (
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                raise (
                    OfficialGatedTeacherInferenceError(
                        "failed to process "
                        f"inference batch {batch_index}"
                    )
                ) from exc

            results.extend(
                batch_results
            )

    return results