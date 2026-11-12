"""Pretrained initialization for the official gated teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rgbt_uav.models.official_gated_teacher import (
    OfficialGatedDualStreamTeacher,
)
from rgbt_uav.training.official_distillation_student import (
    OfficialStudentError,
    _extract_state_dict,
)


class OfficialGatedTeacherPretrainedError(RuntimeError):
    """Raised when gated-teacher pretrained loading fails."""


_FIRST_CONV_KEY = "model.0.conv.weight"


def _compatible_state(
    source: dict[str, Tensor],
    target: dict[str, Tensor],
) -> dict[str, Tensor]:
    compatible: dict[str, Tensor] = {}

    for key, source_value in source.items():
        target_value = target.get(key)

        if target_value is None:
            continue

        if source_value.shape != target_value.shape:
            continue

        compatible[key] = (
            source_value.detach().clone()
        )

    return compatible


def _adapt_ir_state(
    source: dict[str, Tensor],
    target: dict[str, Tensor],
) -> tuple[
    dict[str, Tensor],
    tuple[str, ...],
]:
    compatible = _compatible_state(
        source,
        target,
    )
    adapted_keys: list[str] = []

    source_first_conv = source.get(
        _FIRST_CONV_KEY
    )
    target_first_conv = target.get(
        _FIRST_CONV_KEY
    )

    if not isinstance(source_first_conv, Tensor):
        raise OfficialGatedTeacherPretrainedError(
            "pretrained first convolution is missing"
        )

    if not isinstance(target_first_conv, Tensor):
        raise OfficialGatedTeacherPretrainedError(
            "IR first convolution is missing"
        )

    if (
        source_first_conv.ndim != 4
        or source_first_conv.shape[1] != 3
    ):
        raise OfficialGatedTeacherPretrainedError(
            "pretrained first convolution "
            "must have three input channels"
        )

    if (
        target_first_conv.ndim != 4
        or target_first_conv.shape[1] != 1
    ):
        raise OfficialGatedTeacherPretrainedError(
            "IR first convolution must "
            "have one input channel"
        )

    adapted_first_conv = (
        source_first_conv.detach()
        .to(
            dtype=target_first_conv.dtype,
        )
        .mean(
            dim=1,
            keepdim=True,
        )
        .clone()
    )

    if (
        adapted_first_conv.shape
        != target_first_conv.shape
    ):
        raise OfficialGatedTeacherPretrainedError(
            "adapted IR first convolution "
            "does not match the target shape"
        )

    if not torch.isfinite(
        adapted_first_conv
    ).all():
        raise OfficialGatedTeacherPretrainedError(
            "adapted IR first convolution "
            "must contain finite values"
        )

    compatible[_FIRST_CONV_KEY] = (
        adapted_first_conv
    )
    adapted_keys.append(_FIRST_CONV_KEY)

    return (
        compatible,
        tuple(sorted(adapted_keys)),
    )


def _detect_source_state(
    source: dict[str, Tensor],
    target: dict[str, Tensor],
) -> dict[str, Tensor]:
    compatible: dict[str, Tensor] = {}

    for source_key, source_value in source.items():
        parts = source_key.split(
            ".",
            maxsplit=2,
        )

        if (
            len(parts) != 3
            or parts[0] != "model"
            or not parts[1].isdigit()
        ):
            continue

        target_key = parts[2]
        target_value = target.get(
            target_key
        )

        if target_value is None:
            continue

        if source_value.shape != target_value.shape:
            continue

        compatible[target_key] = (
            source_value.detach().clone()
        )

    return compatible


def load_official_gated_teacher_pretrained(
    teacher: OfficialGatedDualStreamTeacher,
    checkpoint_path: Path,
) -> dict[str, object]:
    """Initialize RGB, IR, and the shared Detect head.

    RGB-compatible parameters are loaded directly. The IR first
    convolution is initialized by averaging the pretrained RGB
    convolution across its three input channels. The shared Detect
    head receives compatible parameters exactly once.
    """

    if not isinstance(
        teacher,
        OfficialGatedDualStreamTeacher,
    ):
        raise OfficialGatedTeacherPretrainedError(
            "teacher must be an "
            "OfficialGatedDualStreamTeacher"
        )

    path = Path(checkpoint_path)

    if not path.is_file():
        raise OfficialGatedTeacherPretrainedError(
            "pretrained checkpoint must be "
            f"a regular file: {path}"
        )

    try:
        payload: Any = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        source = _extract_state_dict(
            payload
        )
    except (
        OSError,
        OfficialStudentError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherPretrainedError(
            "could not read official "
            "pretrained checkpoint"
        ) from exc

    rgb_target = (
        teacher.rgb_stream.state_dict()
    )
    ir_target = (
        teacher.ir_stream.state_dict()
    )
    detect_target = (
        teacher.detect.state_dict()
    )

    rgb_compatible = _compatible_state(
        source,
        rgb_target,
    )
    ir_compatible, ir_adapted_keys = (
        _adapt_ir_state(
            source,
            ir_target,
        )
    )
    detect_compatible = (
        _detect_source_state(
            source,
            detect_target,
        )
    )

    if _FIRST_CONV_KEY not in rgb_compatible:
        raise OfficialGatedTeacherPretrainedError(
            "RGB first convolution is incompatible"
        )

    if not detect_compatible:
        raise OfficialGatedTeacherPretrainedError(
            "no compatible Detect parameters "
            "were found"
        )

    try:
        rgb_incompatible = (
            teacher.rgb_stream.load_state_dict(
                rgb_compatible,
                strict=False,
            )
        )
        ir_incompatible = (
            teacher.ir_stream.load_state_dict(
                ir_compatible,
                strict=False,
            )
        )
        detect_incompatible = (
            teacher.detect.load_state_dict(
                detect_compatible,
                strict=False,
            )
        )
    except (
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherPretrainedError(
            "pretrained state is incompatible "
            "with the gated teacher"
        ) from exc

    teacher.train()

    for parameter in teacher.parameters():
        parameter.requires_grad_(True)

    return {
        "checkpoint": str(path),
        "rgb_loaded_keys": tuple(
            sorted(rgb_compatible)
        ),
        "ir_loaded_keys": tuple(
            sorted(ir_compatible)
        ),
        "ir_adapted_keys": (
            ir_adapted_keys
        ),
        "detect_loaded_keys": tuple(
            sorted(detect_compatible)
        ),
        "rgb_missing_keys": tuple(
            rgb_incompatible.missing_keys
        ),
        "rgb_unexpected_keys": tuple(
            rgb_incompatible.unexpected_keys
        ),
        "ir_missing_keys": tuple(
            ir_incompatible.missing_keys
        ),
        "ir_unexpected_keys": tuple(
            ir_incompatible.unexpected_keys
        ),
        "detect_missing_keys": tuple(
            detect_incompatible.missing_keys
        ),
        "detect_unexpected_keys": tuple(
            detect_incompatible.unexpected_keys
        ),
    }