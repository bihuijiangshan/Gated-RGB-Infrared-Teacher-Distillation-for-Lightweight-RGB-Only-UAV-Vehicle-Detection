"""Statistics for gated-teacher lightweight distillation evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

METRICS = (
    "mAP50-95",
    "mAP50",
    "AP75",
    "AP_small",
    "AR_small",
)

METHODS = (
    "detection_only",
    "early_fusion_distillation",
    "gated_teacher_distillation",
)

PAIRED_COMPARISONS = {
    (
        "gated_teacher_distillation_"
        "minus_detection_only"
    ): (
        "gated_teacher_distillation",
        "detection_only",
    ),
    (
        "gated_teacher_distillation_"
        "minus_early_fusion_distillation"
    ): (
        "gated_teacher_distillation",
        "early_fusion_distillation",
    ),
}


class GatedDistillationReportError(RuntimeError):
    """Raised when multiseed evidence is invalid."""


def _finite_float(
    value: object,
    *,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise GatedDistillationReportError(
            f"{label} must be numeric"
        )

    number = float(value)

    if not math.isfinite(number):
        raise GatedDistillationReportError(
            f"{label} must be finite"
        )

    return number


def _validate_seeds(
    seeds: Sequence[int],
) -> tuple[int, ...]:
    normalized = tuple(seeds)

    if len(normalized) < 2:
        raise GatedDistillationReportError(
            "at least two seeds are required"
        )

    if any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        for seed in normalized
    ):
        raise GatedDistillationReportError(
            "seeds must be non-negative integers"
        )

    if len(set(normalized)) != len(normalized):
        raise GatedDistillationReportError(
            "seeds must be unique"
        )

    return normalized


def _validate_epoch(
    value: object,
    *,
    label: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise GatedDistillationReportError(
            f"{label} must be a positive integer"
        )

    return value


def _summary(
    values: Sequence[float],
) -> dict[str, object]:
    normalized = [
        _finite_float(
            value,
            label="summary value",
        )
        for value in values
    ]

    if len(normalized) < 2:
        raise GatedDistillationReportError(
            "summary requires at least two values"
        )

    return {
        "values": normalized,
        "mean": statistics.fmean(normalized),
        "sample_std": statistics.stdev(normalized),
        "min": min(normalized),
        "max": max(normalized),
    }


def _paired_summary(
    values: Sequence[float],
) -> dict[str, object]:
    result = _summary(values)

    positive = sum(
        value > 0.0
        for value in values
    )
    negative = sum(
        value < 0.0
        for value in values
    )
    zero = sum(
        value == 0.0
        for value in values
    )

    result.update(
        {
            "positive_seed_count": positive,
            "negative_seed_count": negative,
            "zero_seed_count": zero,
            "win_rate": positive / len(values),
        }
    )

    return result


def _validate_method_payload(
    payload: object,
    *,
    seed: int,
    method: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GatedDistillationReportError(
            f"seed {seed} method {method} "
            "must be a mapping"
        )

    normalized = dict(payload)

    _validate_epoch(
        normalized.get("selected_epoch"),
        label=(
            f"seed {seed} method {method} "
            "selected_epoch"
        ),
    )

    for metric in METRICS:
        if metric not in normalized:
            raise GatedDistillationReportError(
                f"seed {seed} method {method} "
                f"is missing {metric}"
            )

        _finite_float(
            normalized[metric],
            label=(
                f"seed {seed} method {method} "
                f"{metric}"
            ),
        )

    return normalized


def _validate_runs(
    runs: Mapping[int, Mapping[str, object]],
    *,
    seeds: tuple[int, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    if set(runs) != set(seeds):
        raise GatedDistillationReportError(
            "runs must contain exactly the requested seeds"
        )

    normalized: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for seed in seeds:
        seed_payload = runs[seed]

        if not isinstance(seed_payload, Mapping):
            raise GatedDistillationReportError(
                f"seed {seed} must be a mapping"
            )

        if set(seed_payload) != set(METHODS):
            expected = ", ".join(METHODS)
            raise GatedDistillationReportError(
                f"seed {seed} must contain exactly: "
                f"{expected}"
            )

        normalized[str(seed)] = {
            method: _validate_method_payload(
                seed_payload[method],
                seed=seed,
                method=method,
            )
            for method in METHODS
        }

    return normalized


def build_multiseed_report(
    *,
    runs: Mapping[int, Mapping[str, object]],
    seeds: Sequence[int],
) -> dict[str, object]:
    """Build aggregate and paired statistics for three methods."""

    normalized_seeds = _validate_seeds(seeds)
    normalized_runs = _validate_runs(
        runs,
        seeds=normalized_seeds,
    )

    aggregate: dict[
        str,
        dict[str, dict[str, object]],
    ] = {}

    for method in METHODS:
        aggregate[method] = {}

        for metric in METRICS:
            values = [
                _finite_float(
                    normalized_runs[str(seed)][
                        method
                    ][metric],
                    label=(
                        f"seed {seed} method "
                        f"{method} {metric}"
                    ),
                )
                for seed in normalized_seeds
            ]

            aggregate[method][metric] = _summary(
                values
            )

    paired: dict[
        str,
        dict[str, dict[str, object]],
    ] = {}

    for (
        comparison,
        (
            candidate_method,
            reference_method,
        ),
    ) in PAIRED_COMPARISONS.items():
        paired[comparison] = {}

        for metric in METRICS:
            values = [
                (
                    _finite_float(
                        normalized_runs[str(seed)][
                            candidate_method
                        ][metric],
                        label=(
                            f"seed {seed} "
                            f"{candidate_method} "
                            f"{metric}"
                        ),
                    )
                    - _finite_float(
                        normalized_runs[str(seed)][
                            reference_method
                        ][metric],
                        label=(
                            f"seed {seed} "
                            f"{reference_method} "
                            f"{metric}"
                        ),
                    )
                )
                for seed in normalized_seeds
            ]

            paired[comparison][metric] = (
                _paired_summary(values)
            )

    return {
        "format_version": 1,
        "status": "completed",
        "seeds": list(normalized_seeds),
        "metrics": list(METRICS),
        "methods": list(METHODS),
        "runs": normalized_runs,
        "aggregate": aggregate,
        "paired": paired,
    }