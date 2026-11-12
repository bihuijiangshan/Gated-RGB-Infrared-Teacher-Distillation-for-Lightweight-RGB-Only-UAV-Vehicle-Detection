"""Evidence report for official RGB student distillation ablations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class OfficialRGBAblationReportError(RuntimeError):
    """Raised when ablation evidence is incomplete or inconsistent."""


_EXPECTED_WEIGHTS = {
    "detection_only": (0.0, 0.0),
    "feature_only": (1.0, 0.0),
    "logit_only": (0.0, 1.0),
    "full": (1.0, 1.0),
}

_METRIC_NAMES = (
    "mAP50-95",
    "mAP50",
    "AP75",
    "AP_small",
    "AR_small",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OfficialRGBAblationReportError(
            f"evidence must be a regular file: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise OfficialRGBAblationReportError(
            f"failed to read evidence: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise OfficialRGBAblationReportError(
            f"evidence root must be a mapping: {path}"
        )

    return payload


def _finite_metric(
    payload: Mapping[str, object],
    name: str,
) -> float:
    value = payload.get(name)

    if not isinstance(value, (int, float)):
        raise OfficialRGBAblationReportError(
            f"metric {name} must be numeric"
        )

    result = float(value)
    if not math.isfinite(result):
        raise OfficialRGBAblationReportError(
            f"metric {name} must be finite"
        )

    return result

def _validated_per_class(
    payload: object,
) -> dict[str, dict[str, float | None]]:
    if not isinstance(payload, Mapping) or not payload:
        raise OfficialRGBAblationReportError(
            "per_class metrics must be a non-empty mapping"
        )

    required_metrics = {
        "AP50-95",
        "AP50",
        "AP75",
        "AP_small",
    }
    result: dict[
        str,
        dict[str, float | None],
    ] = {}

    for class_name, class_payload in payload.items():
        if not isinstance(class_name, str) or not class_name:
            raise OfficialRGBAblationReportError(
                "per-class names must be non-empty strings"
            )
        if not isinstance(class_payload, Mapping):
            raise OfficialRGBAblationReportError(
                f"per-class metrics are invalid for {class_name}"
            )
        if set(class_payload) != required_metrics:
            raise OfficialRGBAblationReportError(
                f"per-class schema is invalid for {class_name}"
            )

        class_metrics: dict[str, float | None] = {}

        for metric_name in required_metrics:
            value = class_payload[metric_name]

            if value is None:
                class_metrics[metric_name] = None
                continue

            if not isinstance(value, (int, float)):
                raise OfficialRGBAblationReportError(
                    f"{class_name}.{metric_name} must be numeric or null"
                )

            number = float(value)
            if not math.isfinite(number):
                raise OfficialRGBAblationReportError(
                    f"{class_name}.{metric_name} must be finite"
                )

            class_metrics[metric_name] = number

        result[class_name] = class_metrics

    return result


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise OfficialRGBAblationReportError(
            f"weights must be a regular file: {path}"
        )

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_official_rgb_ablation_report(
    run_roots: Mapping[str, Path],
    destination: Path,
    *,
    git_commit: str,
    dataset_name: str,
) -> Path:
    """Validate four ablations and write one immutable-style report."""

    if set(run_roots) != set(_EXPECTED_WEIGHTS):
        raise OfficialRGBAblationReportError(
            "run roots must contain the four expected ablations"
        )
    if not isinstance(git_commit, str) or not git_commit.strip():
        raise OfficialRGBAblationReportError(
            "git_commit must be non-empty"
        )
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise OfficialRGBAblationReportError(
            "dataset_name must be non-empty"
        )

    runs: dict[str, dict[str, object]] = {}
    common_dataset: dict[str, int] | None = None
    common_classes: set[str] | None = None

    for run_name, expected_weights in _EXPECTED_WEIGHTS.items():
        root = Path(run_roots[run_name])
        completion = _read_json(
            root / "completion.json"
        )
        metrics_payload = _read_json(
            root / "paired_coco_metrics.json"
        )

        if completion.get("status") != "completed":
            raise OfficialRGBAblationReportError(
                f"{run_name} is not completed"
            )

        config = completion.get("config")
        dataset = completion.get("dataset")

        if not isinstance(config, dict):
            raise OfficialRGBAblationReportError(
                f"{run_name} config is invalid"
            )
        if not isinstance(dataset, dict):
            raise OfficialRGBAblationReportError(
                f"{run_name} dataset metadata is invalid"
            )

        actual_weights = (
            float(config.get("feature_weight", -1)),
            float(config.get("logit_weight", -1)),
        )
        if actual_weights != expected_weights:
            raise OfficialRGBAblationReportError(
                f"{run_name} weights do not match its ablation"
            )

        train_count = dataset.get("train_count")
        val_count = dataset.get("val_count")
        if (
            not isinstance(train_count, int)
            or train_count <= 0
            or not isinstance(val_count, int)
            or val_count <= 0
        ):
            raise OfficialRGBAblationReportError(
                f"{run_name} dataset counts are invalid"
            )

        run_dataset = {
            "train_count": train_count,
            "val_count": val_count,
        }
        if common_dataset is None:
            common_dataset = run_dataset
        elif run_dataset != common_dataset:
            raise OfficialRGBAblationReportError(
                "ablation dataset counts do not match"
            )

        best_weights_value = completion.get(
            "best_weights"
        )
        if (
            not isinstance(best_weights_value, str)
            or not best_weights_value
        ):
            raise OfficialRGBAblationReportError(
                f"{run_name} best_weights is invalid"
            )

        best_weights = Path(best_weights_value)
        metrics = {
            name: _finite_metric(
                metrics_payload,
                name,
            )
            for name in _METRIC_NAMES
        }

        per_class = _validated_per_class(
            metrics_payload.get("per_class")
        )

        run_classes = set(per_class)

        if common_classes is None:
            common_classes = run_classes
        elif run_classes != common_classes:
            raise OfficialRGBAblationReportError(
                "ablation per-class names do not match"
            )

        val_metrics = completion.get("metrics")
        if not isinstance(val_metrics, dict):
            raise OfficialRGBAblationReportError(
                f"{run_name} validation metrics are invalid"
            )

        runs[run_name] = {
            "experiment_id": completion.get("experiment_id"),
            "seed": config.get("seed"),
            "epochs": config.get("epochs"),
            "feature_weight": actual_weights[0],
            "logit_weight": actual_weights[1],
            "teacher_checkpoint": config.get("teacher_checkpoint"),
            "student_pretrained": config.get("student_pretrained"),
            "best_weights": str(best_weights),
            "best_weights_sha256": _sha256(best_weights),
            "restored_epoch": metrics_payload.get("restored_epoch"),
            "val_loss": _finite_metric(
                val_metrics,
                "val_loss",
            ),
            **metrics,
            "per_class": per_class,
        }

    assert common_dataset is not None
    assert common_classes is not None

    baseline = runs["detection_only"]
    comparisons: dict[str, dict[str, float]] = {}

    for name in (
        "feature_only",
        "logit_only",
        "full",
    ):
        comparisons[f"{name}_minus_detection_only"] = {
            "mAP50-95": (
                float(runs[name]["mAP50-95"])
                - float(baseline["mAP50-95"])
            ),
            "mAP50": (
                float(runs[name]["mAP50"])
                - float(baseline["mAP50"])
            ),
        }

    full_gain = comparisons[
        "full_minus_detection_only"
    ]
    feature_gain = comparisons[
        "feature_only_minus_detection_only"
    ]
    logit_gain = comparisons[
        "logit_only_minus_detection_only"
    ]

    comparisons["full_interaction"] = {
        "mAP50-95": (
            full_gain["mAP50-95"]
            - feature_gain["mAP50-95"]
            - logit_gain["mAP50-95"]
        ),
        "mAP50": (
            full_gain["mAP50"]
            - feature_gain["mAP50"]
            - logit_gain["mAP50"]
        ),
    }

    per_class_comparison: dict[
        str,
        dict[str, float | None],
    ] = {}

    for class_name in sorted(common_classes):
        baseline_class = baseline["per_class"]
        full_class = runs["full"]["per_class"]

        assert isinstance(baseline_class, dict)
        assert isinstance(full_class, dict)

        baseline_metrics = baseline_class[class_name]
        full_metrics = full_class[class_name]

        class_delta: dict[str, float | None] = {}

        for metric_name in (
            "AP50-95",
            "AP50",
            "AP75",
            "AP_small",
        ):
            baseline_value = baseline_metrics[
                metric_name
            ]
            full_value = full_metrics[metric_name]

            if (
                baseline_value is None
                or full_value is None
            ):
                class_delta[metric_name] = None
            else:
                class_delta[metric_name] = (
                    float(full_value)
                    - float(baseline_value)
                )

        per_class_comparison[class_name] = (
            class_delta
        )

    comparisons[
        "full_minus_detection_only_per_class"
    ] = per_class_comparison

    report = {
        "format_version": 1,
        "status": "completed",
        "scope": "5% preflight; not a formal thesis result",
        "git_commit": git_commit,
        "dataset": {
            "name": dataset_name,
            **common_dataset,
        },
        "evaluation": {
            "score_threshold": 0.001,
            "iou_threshold": 0.7,
            "max_det_before_coco": 300,
            "coco_max_dets": 100,
        },
        "runs": runs,
        "comparisons": comparisons,
    }

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(
        target.suffix + ".tmp"
    )

    try:
        temporary.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except (OSError, TypeError, ValueError) as exc:
        if temporary.is_file():
            temporary.unlink()
        raise OfficialRGBAblationReportError(
            f"failed to write report: {target}"
        ) from exc

    return target