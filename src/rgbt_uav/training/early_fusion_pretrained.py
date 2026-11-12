from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

import torch
from torch import Tensor


class EarlyFusionPretrainedError(ValueError):
    """Raised when RGB pretrained weights cannot be adapted to RGB-IR input."""


_T = TypeVar("_T")


def adapt_first_conv_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    first_conv_key: str = "model.0.conv.weight",
) -> dict[str, Tensor]:
    """Adapt a 3-channel YOLO state dict to a 4-channel RGB-IR input."""

    if not isinstance(state_dict, Mapping):
        raise EarlyFusionPretrainedError(
            "state_dict must be a mapping"
        )

    if not isinstance(first_conv_key, str) or not first_conv_key:
        raise EarlyFusionPretrainedError(
            "first_conv_key must be a non-empty string"
        )

    if first_conv_key not in state_dict:
        raise EarlyFusionPretrainedError(
            f"first convolution is missing: {first_conv_key}"
        )

    first_weight = state_dict[first_conv_key]

    if not isinstance(first_weight, Tensor):
        raise EarlyFusionPretrainedError(
            "first convolution weight must be a tensor"
        )

    if first_weight.ndim != 4:
        raise EarlyFusionPretrainedError(
            "first convolution weight must have shape [out, in, height, width]"
        )

    if first_weight.shape[1] != 3:
        raise EarlyFusionPretrainedError(
            "first convolution must have exactly three input channels"
        )

    if not torch.isfinite(first_weight).all():
        raise EarlyFusionPretrainedError(
            "first convolution weight must be finite"
        )

    adapted = dict(state_dict)
    ir_weight = first_weight.mean(
        dim=1,
        keepdim=True,
    )
    adapted[first_conv_key] = torch.cat(
        (first_weight, ir_weight),
        dim=1,
    ).clone()

    return adapted


def load_adapted_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    first_conv_key: str = "model.0.conv.weight",
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Adapt and load RGB pretrained weights into a four-channel model."""

    if not isinstance(model, torch.nn.Module):
        raise EarlyFusionPretrainedError(
            "model must be a torch.nn.Module"
        )

    adapted = adapt_first_conv_state_dict(
        state_dict,
        first_conv_key=first_conv_key,
    )

    if not strict:
        model_state = model.state_dict()
        compatible: dict[str, Tensor] = {}

        for key, value in adapted.items():
            expected = model_state.get(key)

            if expected is None:
                continue

            if tuple(value.shape) != tuple(expected.shape):
                continue

            compatible[key] = value

        adapted = compatible

    try:
        incompatible = model.load_state_dict(
            adapted,
            strict=strict,
        )
    except (RuntimeError, TypeError) as exc:
        raise EarlyFusionPretrainedError(
            "adapted pretrained state is incompatible with model"
        ) from exc

    return (
        list(incompatible.missing_keys),
        list(incompatible.unexpected_keys),
    )