from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from ultralytics.nn.tasks import DetectionModel

from rgbt_uav.training.ultralytics_adapter import (
    UltralyticsAdapterError,
    to_ultralytics_early_fusion_batch,
)


class EarlyFusionRuntimeError(RuntimeError):
    """Raised when the early-fusion runtime cannot be constructed or evaluated."""


def resolve_early_fusion_device(value: str) -> torch.device:
    """Resolve a configured CPU or CUDA device."""

    if not isinstance(value, str) or not value.strip():
        raise EarlyFusionRuntimeError(
            "device must be a non-empty string"
        )

    normalized = value.strip().lower()

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized.isdigit():
        device = torch.device(f"cuda:{int(normalized)}")
    else:
        try:
            device = torch.device(normalized)
        except (RuntimeError, ValueError) as exc:
            raise EarlyFusionRuntimeError(
                f"invalid device: {value}"
            ) from exc

    if device.type != "cuda":
        raise EarlyFusionRuntimeError(
            "device must be 'cpu', a CUDA index, or a CUDA device"
        )

    if not torch.cuda.is_available():
        raise EarlyFusionRuntimeError(
            f"CUDA device requested but CUDA is unavailable: {value}"
        )

    index = device.index if device.index is not None else 0

    if index < 0 or index >= torch.cuda.device_count():
        raise EarlyFusionRuntimeError(
            f"CUDA device index is unavailable: {index}"
        )

    return torch.device(f"cuda:{index}")


def seed_early_fusion(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible run."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise EarlyFusionRuntimeError(
            "seed must be a non-negative integer"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_early_fusion_model(
    *,
    model_name: str,
    num_classes: int,
    device: torch.device | str,
) -> DetectionModel:
    """Build the fixed four-channel Ultralytics detection model."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise EarlyFusionRuntimeError(
            "model_name must be a non-empty string"
        )

    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise EarlyFusionRuntimeError(
            "num_classes must be a positive integer"
        )

    target_device = torch.device(device)

    try:
        model = DetectionModel(
            model_name.strip(),
            ch=4,
            nc=num_classes,
            verbose=False,
        )
    except Exception as exc:
        raise EarlyFusionRuntimeError(
            f"could not construct early-fusion model: {model_name}"
        ) from exc

    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    try:
        return model.to(target_device)
    except (RuntimeError, TypeError) as exc:
        raise EarlyFusionRuntimeError(
            f"could not move model to device: {target_device}"
        ) from exc


def _require_tensor(
    converted: Mapping[str, object],
    key: str,
) -> Tensor:
    value = converted.get(key)

    if not isinstance(value, Tensor):
        raise EarlyFusionRuntimeError(
            f"converted batch field {key!r} must be a tensor"
        )

    return value


def evaluate_early_fusion_loss(
    model: nn.Module,
    criterion: Any,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device | str,
) -> float:
    """Evaluate mean official detection loss without changing model state."""

    if not isinstance(model, nn.Module):
        raise EarlyFusionRuntimeError(
            "model must be a torch.nn.Module"
        )

    target_device = torch.device(device)
    was_training = model.training
    total = 0.0
    batches = 0

    model.to(target_device)
    model.eval()

    try:
        with torch.no_grad():
            for paired_batch in loader:
                try:
                    converted = to_ultralytics_early_fusion_batch(
                        paired_batch
                    )
                except UltralyticsAdapterError as exc:
                    raise EarlyFusionRuntimeError(
                        f"validation batch conversion failed at batch {batches}"
                    ) from exc

                images = _require_tensor(converted, "img").to(
                    target_device
                )
                batch_indices = _require_tensor(
                    converted,
                    "batch_idx",
                ).to(target_device)
                classes = _require_tensor(converted, "cls").to(
                    target_device
                )
                boxes = _require_tensor(converted, "bboxes").to(
                    target_device
                )

                validation_batch = {
                    "img": images,
                    "batch_idx": batch_indices,
                    "cls": classes,
                    "bboxes": boxes,
                }

                try:
                    predictions = model(images)
                    loss, _ = criterion(
                        predictions,
                        validation_batch,
                    )
                except (KeyError, TypeError, RuntimeError) as exc:
                    raise EarlyFusionRuntimeError(
                        f"validation loss failed at batch {batches}"
                    ) from exc

                if not isinstance(loss, Tensor) or loss.numel() != 1:
                    raise EarlyFusionRuntimeError(
                        "validation criterion must return a scalar loss tensor"
                    )

                numeric_loss = float(loss.detach().cpu())

                if not math.isfinite(numeric_loss):
                    raise EarlyFusionRuntimeError(
                        "validation loss must be finite"
                    )

                total += numeric_loss
                batches += 1
    finally:
        model.train(was_training)

    if batches == 0:
        raise EarlyFusionRuntimeError(
            "validation loader produced no batches"
        )

    return total / batches