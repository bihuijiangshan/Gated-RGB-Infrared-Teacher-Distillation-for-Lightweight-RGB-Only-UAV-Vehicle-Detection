from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


class DistillationPretrainedError(RuntimeError):
    """Raised when a teacher checkpoint cannot be loaded."""


def _state_dict(payload: object) -> Mapping[str, Any]:
    if hasattr(payload, "state_dict"):
        value = payload.state_dict()
        if isinstance(value, Mapping):
            return value

    if isinstance(payload, Mapping):
        model_payload = payload.get("model")

        if hasattr(model_payload, "state_dict"):
            value = model_payload.state_dict()
            if isinstance(value, Mapping):
                return value

        if isinstance(model_payload, Mapping):
            return model_payload

        if all(isinstance(key, str) for key in payload):
            return payload

    raise DistillationPretrainedError(
        "teacher checkpoint does not contain a supported state_dict"
    )


def load_teacher_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    map_location: torch.device | str | None = None,
) -> dict[str, object]:
    """Load teacher weights and return validated incompatibility metadata."""

    checkpoint = Path(path)

    if not checkpoint.is_file():
        raise DistillationPretrainedError(
            f"teacher checkpoint does not exist: {checkpoint}"
        )

    if not isinstance(model, nn.Module):
        raise DistillationPretrainedError(
            "model must be a torch.nn.Module"
        )

    try:
        payload = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=False,
        )
        state_dict = _state_dict(payload)
        incompatible = model.load_state_dict(
            state_dict,
            strict=False,
        )
    except Exception as exc:
        raise DistillationPretrainedError(
            f"teacher checkpoint is incompatible: {checkpoint}"
        ) from exc

    missing_keys = tuple(incompatible.missing_keys)
    unexpected_keys = tuple(incompatible.unexpected_keys)

    if missing_keys or unexpected_keys:
        raise DistillationPretrainedError(
            "teacher checkpoint has missing or unexpected keys"
        )

    return {
        "checkpoint": checkpoint,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }