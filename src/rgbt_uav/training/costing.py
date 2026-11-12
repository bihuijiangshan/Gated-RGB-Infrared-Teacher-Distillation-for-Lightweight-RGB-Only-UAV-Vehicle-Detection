"""Estimate full-data training GPU hours from preflight evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median


class TrainingCostEvidenceError(ValueError):
    """Raised when completion evidence cannot support a training cost estimate."""


@dataclass(frozen=True)
class TrainingCostEstimate:
    preflight_train_count: int
    preflight_val_count: int
    full_train_count: int
    full_val_count: int
    target_epochs: int
    median_train_epoch_seconds: float
    median_val_epoch_seconds: float
    one_time_seconds: float
    optimistic_gpu_hours: float
    neutral_gpu_hours: float
    conservative_gpu_hours: float
    reference_hardware_only: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "median_train_epoch_seconds",
            "median_val_epoch_seconds",
            "one_time_seconds",
            "optimistic_gpu_hours",
            "neutral_gpu_hours",
            "conservative_gpu_hours",
        ):
            _finite_nonnegative(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "preflight_train_count": self.preflight_train_count,
            "preflight_val_count": self.preflight_val_count,
            "full_train_count": self.full_train_count,
            "full_val_count": self.full_val_count,
            "target_epochs": self.target_epochs,
            "median_train_epoch_seconds": self.median_train_epoch_seconds,
            "median_val_epoch_seconds": self.median_val_epoch_seconds,
            "one_time_seconds": self.one_time_seconds,
            "optimistic_gpu_hours": self.optimistic_gpu_hours,
            "neutral_gpu_hours": self.neutral_gpu_hours,
            "conservative_gpu_hours": self.conservative_gpu_hours,
            "reference_hardware_only": self.reference_hardware_only,
        }


def _required(mapping: Mapping[str, object], key: str, path: str) -> object:
    for existing_key in mapping:
        if existing_key == key:
            return mapping[existing_key]
    raise TrainingCostEvidenceError(f"missing {path}.{key}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingCostEvidenceError(f"{path} must be a mapping")
    return value


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise TrainingCostEvidenceError(f"{path} must be a positive integer")
    return value


def _finite_nonnegative(value: object, path: str) -> float:
    if type(value) not in (int, float):
        raise TrainingCostEvidenceError(f"{path} must be an int or float")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise TrainingCostEvidenceError(f"{path} is too large") from exc
    if not math.isfinite(converted) or converted < 0:
        raise TrainingCostEvidenceError(f"{path} must be finite and nonnegative")
    return converted


def _finite_arithmetic(operation: Callable[[], int | float], path: str) -> float:
    try:
        result = operation()
    except OverflowError as exc:
        raise TrainingCostEvidenceError(f"{path} overflowed") from exc
    return _finite_nonnegative(result, path)


def _epoch_timings(value: object, path: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TrainingCostEvidenceError(f"{path} must be a sequence")
    if len(value) < 5:
        raise TrainingCostEvidenceError(f"{path} must contain at least five values")
    return tuple(_finite_nonnegative(item, f"{path}[{index}]") for index, item in enumerate(value))


def estimate_gpu_hours(
    completion: Mapping[str, object],
    *,
    full_train_count: int,
    full_val_count: int,
    target_epochs: int,
) -> TrainingCostEstimate:
    completion_mapping = _mapping(completion, "completion")
    dataset_provenance = _mapping(
        _required(completion_mapping, "dataset_provenance", "completion"),
        "completion.dataset_provenance",
    )
    counts = _mapping(
        _required(dataset_provenance, "counts", "completion.dataset_provenance"),
        "completion.dataset_provenance.counts",
    )
    preflight_train_count = _positive_int(
        _required(counts, "train", "completion.dataset_provenance.counts"),
        "completion.dataset_provenance.counts.train",
    )
    preflight_val_count = _positive_int(
        _required(counts, "val", "completion.dataset_provenance.counts"),
        "completion.dataset_provenance.counts.val",
    )
    checked_full_train_count = _positive_int(full_train_count, "full_train_count")
    checked_full_val_count = _positive_int(full_val_count, "full_val_count")
    if type(target_epochs) is not int or target_epochs != 100:
        raise TrainingCostEvidenceError("target_epochs must be exactly integer 100")

    timing = _mapping(
        _required(completion_mapping, "timing", "completion"),
        "completion.timing",
    )
    train_epoch_seconds = _epoch_timings(
        _required(timing, "train_epoch_seconds", "completion.timing"),
        "completion.timing.train_epoch_seconds",
    )
    val_epoch_seconds = _epoch_timings(
        _required(timing, "val_epoch_seconds", "completion.timing"),
        "completion.timing.val_epoch_seconds",
    )
    median_train_seconds = _finite_arithmetic(
        lambda: median(train_epoch_seconds),
        "median_train_epoch_seconds",
    )
    median_val_seconds = _finite_arithmetic(
        lambda: median(val_epoch_seconds),
        "median_val_epoch_seconds",
    )
    one_time_components = tuple(
        _finite_nonnegative(
            _required(timing, component, "completion.timing"),
            f"completion.timing.{component}",
        )
        for component in (
            "framework_final_validation_seconds",
            "final_validation_seconds",
            "reload_prediction_seconds",
            "fixed_seconds",
        )
    )
    one_time_seconds = _finite_arithmetic(
        lambda: sum(one_time_components),
        "one_time_seconds",
    )

    train_ratio = _finite_arithmetic(
        lambda: checked_full_train_count / preflight_train_count,
        "train_ratio",
    )
    val_ratio = _finite_arithmetic(
        lambda: checked_full_val_count / preflight_val_count,
        "val_ratio",
    )
    scaled_train_seconds = _finite_arithmetic(
        lambda: median_train_seconds * train_ratio,
        "scaled_train_epoch_seconds",
    )
    scaled_val_seconds = _finite_arithmetic(
        lambda: median_val_seconds * val_ratio,
        "scaled_val_epoch_seconds",
    )
    scaled_epoch_seconds = _finite_arithmetic(
        lambda: scaled_train_seconds + scaled_val_seconds,
        "scaled_epoch_seconds",
    )
    variable_seconds = _finite_arithmetic(
        lambda: target_epochs * scaled_epoch_seconds,
        "variable_seconds",
    )
    neutral_seconds = _finite_arithmetic(
        lambda: variable_seconds + one_time_seconds,
        "neutral_seconds",
    )
    optimistic_seconds = _finite_arithmetic(
        lambda: 0.85 * variable_seconds + one_time_seconds,
        "optimistic_seconds",
    )
    conservative_seconds = _finite_arithmetic(
        lambda: 1.25 * variable_seconds + one_time_seconds,
        "conservative_seconds",
    )
    optimistic_gpu_hours = _finite_arithmetic(
        lambda: optimistic_seconds / 3600,
        "optimistic_gpu_hours",
    )
    neutral_gpu_hours = _finite_arithmetic(
        lambda: neutral_seconds / 3600,
        "neutral_gpu_hours",
    )
    conservative_gpu_hours = _finite_arithmetic(
        lambda: conservative_seconds / 3600,
        "conservative_gpu_hours",
    )

    return TrainingCostEstimate(
        preflight_train_count=preflight_train_count,
        preflight_val_count=preflight_val_count,
        full_train_count=checked_full_train_count,
        full_val_count=checked_full_val_count,
        target_epochs=target_epochs,
        median_train_epoch_seconds=median_train_seconds,
        median_val_epoch_seconds=median_val_seconds,
        one_time_seconds=one_time_seconds,
        optimistic_gpu_hours=optimistic_gpu_hours,
        neutral_gpu_hours=neutral_gpu_hours,
        conservative_gpu_hours=conservative_gpu_hours,
    )


def apply_hourly_price(
    estimate: TrainingCostEstimate,
    *,
    price_per_gpu_hour: float,
) -> dict[str, float]:
    price = _finite_nonnegative(price_per_gpu_hour, "price_per_gpu_hour")
    return {
        "optimistic": _finite_arithmetic(
            lambda: estimate.optimistic_gpu_hours * price,
            "optimistic_price",
        ),
        "neutral": _finite_arithmetic(
            lambda: estimate.neutral_gpu_hours * price,
            "neutral_price",
        ),
        "conservative": _finite_arithmetic(
            lambda: estimate.conservative_gpu_hours * price,
            "conservative_price",
        ),
    }
