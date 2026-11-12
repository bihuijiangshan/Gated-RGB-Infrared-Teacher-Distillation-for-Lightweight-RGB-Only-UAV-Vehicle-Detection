"""Evaluation policies used to keep paper claims aligned with evidence."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_EXPORTS = {
    "ComplexityReport": ("rgbt_uav.evaluation.complexity", "ComplexityReport"),
    "EvaluationConfig": ("rgbt_uav.evaluation.coco_evidence", "EvaluationConfig"),
    "EvaluationEvidence": ("rgbt_uav.evaluation.coco_evidence", "EvaluationEvidence"),
    "EvidencePublicationError": (
        "rgbt_uav.evaluation.coco_evidence",
        "EvidencePublicationError",
    ),
    "profile_complexity": ("rgbt_uav.evaluation.complexity", "profile_complexity"),
    "run_coco_small_evaluation": (
        "rgbt_uav.evaluation.coco_evidence",
        "run_coco_small_evaluation",
    ),
}

__all__ = list(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
