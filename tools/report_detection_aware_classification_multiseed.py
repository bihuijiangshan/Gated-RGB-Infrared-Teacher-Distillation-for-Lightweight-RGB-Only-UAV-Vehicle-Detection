"""Report detection-aware classification distillation results."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from pathlib import Path

SEEDS = (3407, 3408, 3409)

CLASSIFICATION_RUNS = {
    seed: Path(
        "runs/paired/"
        "official_rgb_classification_distillation_"
        f"teacher_v2_seed{seed}"
    )
    for seed in SEEDS
}

BASELINE_REPORT = Path(
    "runs/archives/"
    "official_rgb_distillation_multiseed_20260827/"
    "summary.json"
)

DESTINATION = Path(
    "runs/archives/"
    "official_rgb_detection_aware_classification_"
    "multiseed_20260829/"
    "summary.json"
)

METRICS = (
    "mAP50-95",
    "mAP50",
    "AP75",
    "AP_small",
    "AR_small",
)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(
            f"required evidence is missing: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"evidence root must be a mapping: {path}"
        )

    return payload


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(
            f"checkpoint is missing: {path}"
        )

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _summary(
    values: list[float],
) -> dict[str, float]:
    if len(values) < 2:
        raise RuntimeError(
            "at least two values are required"
        )

    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _require_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(
            f"{label} must be a mapping"
        )
    return value


def main() -> None:
    dirty = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise RuntimeError(
            "tracked working tree must be clean"
        )

    git_commit = _git_output(
        "rev-parse",
        "HEAD",
    )

    baseline_report = _read_json(
        BASELINE_REPORT
    )
    baseline_runs = _require_mapping(
        baseline_report.get("runs"),
        label="baseline runs",
    )
    baseline_aggregate = _require_mapping(
        baseline_report.get("aggregate"),
        label="baseline aggregate",
    )
    baseline_paired = _require_mapping(
        baseline_report.get(
            "paired_full_minus_detection_only"
        ),
        label="baseline paired aggregate",
    )

    method_values = {
        "detection_only": {
            metric: []
            for metric in METRICS
        },
        "classification_distillation": {
            metric: []
            for metric in METRICS
        },
    }
    paired_values = {
        metric: []
        for metric in METRICS
    }
    seed_results: dict[str, object] = {}

    for seed in SEEDS:
        baseline_seed = _require_mapping(
            baseline_runs.get(str(seed)),
            label=f"baseline seed {seed}",
        )
        detection = _require_mapping(
            baseline_seed.get("detection_only"),
            label=f"detection-only seed {seed}",
        )

        root = CLASSIFICATION_RUNS[seed]
        completion = _read_json(
            root / "completion.json"
        )
        selection = _read_json(
            root / "best_map50_95.json"
        )

        config = _require_mapping(
            completion.get("config"),
            label=f"{root} config",
        )

        if config.get("seed") != seed:
            raise RuntimeError(
                f"{root} seed does not match {seed}"
            )
        if (
            config.get("output_distillation_kind")
            != "detection_aware"
        ):
            raise RuntimeError(
                f"{root} does not use detection-aware loss"
            )

        feature_weight = float(
            config["feature_weight"]
        )
        classification_weight = float(
            config[
                "classification_distillation_weight"
            ]
        )
        dfl_weight = float(
            config["dfl_distillation_weight"]
        )

        if (
            feature_weight,
            classification_weight,
            dfl_weight,
        ) != (0.0, 1.0, 0.0):
            raise RuntimeError(
                f"{root} has incorrect component weights"
            )

        selected_checkpoint_value = selection.get(
            "selected_checkpoint"
        )
        if not isinstance(
            selected_checkpoint_value,
            str,
        ):
            raise RuntimeError(
                f"{root} selected checkpoint is invalid"
            )

        selected_checkpoint = Path(
            selected_checkpoint_value
        )
        actual_hash = _sha256(
            selected_checkpoint
        )
        reported_hash = selection.get("sha256")

        if reported_hash != actual_hash:
            raise RuntimeError(
                f"{root} checkpoint SHA256 does not match"
            )

        selected_epoch = selection.get(
            "selected_epoch"
        )
        if (
            not isinstance(selected_epoch, int)
            or selected_epoch <= 0
        ):
            raise RuntimeError(
                f"{root} selected epoch is invalid"
            )

        detection_metrics = {
            metric: float(detection[metric])
            for metric in METRICS
        }
        classification_metrics = {
            metric: float(selection[metric])
            for metric in METRICS
        }

        for metric in METRICS:
            method_values[
                "detection_only"
            ][metric].append(
                detection_metrics[metric]
            )
            method_values[
                "classification_distillation"
            ][metric].append(
                classification_metrics[metric]
            )

        paired_delta = {
            metric: (
                classification_metrics[metric]
                - detection_metrics[metric]
            )
            for metric in METRICS
        }
        for metric, value in paired_delta.items():
            paired_values[metric].append(value)

        completion_metrics = _require_mapping(
            completion.get("metrics"),
            label=f"{root} completion metrics",
        )

        seed_results[str(seed)] = {
            "detection_only": {
                **detection_metrics,
                "best_weights": detection.get(
                    "best_weights"
                ),
                "best_weights_sha256": detection.get(
                    "best_weights_sha256"
                ),
                "restored_epoch": detection.get(
                    "restored_epoch"
                ),
            },
            "classification_distillation": {
                **classification_metrics,
                "selected_epoch": selected_epoch,
                "selected_checkpoint": str(
                    selected_checkpoint
                ),
                "selected_checkpoint_sha256": (
                    actual_hash
                ),
                "val_loss_selected_by_runner": float(
                    completion_metrics["val_loss"]
                ),
                "teacher_checkpoint": config[
                    "teacher_checkpoint"
                ],
                "feature_weight": feature_weight,
                "classification_distillation_weight": (
                    classification_weight
                ),
                "dfl_distillation_weight": dfl_weight,
                "temperature": float(
                    config["temperature"]
                ),
            },
            (
                "classification_distillation_"
                "minus_detection_only"
            ): paired_delta,
        }

    aggregate = {
        method: {
            metric: _summary(values)
            for metric, values in metrics.items()
        }
        for method, metrics in method_values.items()
    }

    paired_aggregate = {
        metric: {
            **_summary(values),
            "positive_seed_count": sum(
                value > 0
                for value in values
            ),
            "negative_seed_count": sum(
                value < 0
                for value in values
            ),
            "zero_seed_count": sum(
                value == 0
                for value in values
            ),
        }
        for metric, values in paired_values.items()
    }

    report = {
        "format_version": 1,
        "status": "completed",
        "scope": (
            "DroneVehicle 5% three-seed preflight; "
            "not a formal thesis result"
        ),
        "git_commit": git_commit,
        "baseline_report": str(BASELINE_REPORT),
        "baseline_report_git_commit": (
            baseline_report.get("git_commit")
        ),
        "dataset": {
            "name": "DroneVehicle smoke 5%",
            "train_count": 900,
            "val_count": 74,
        },
        "seeds": list(SEEDS),
        "evaluation": {
            "score_threshold": 0.001,
            "iou_threshold": 0.7,
            "max_det_before_coco": 300,
            "coco_max_dets": 100,
            "selection_metric": "mAP50-95",
        },
        "method": {
            "name": (
                "detection-aware classification "
                "distillation"
            ),
            "teacher": (
                "official RGB-T early fusion "
                "teacher v2, 10 epochs"
            ),
            "feature_weight": 0.0,
            "classification_distillation_weight": 1.0,
            "dfl_distillation_weight": 0.0,
            "temperature": 2.0,
        },
        "runs": seed_results,
        "aggregate": aggregate,
        (
            "paired_classification_distillation_"
            "minus_detection_only"
        ): paired_aggregate,
        "legacy_joint_reference": {
            "aggregate_full": baseline_aggregate.get(
                "full"
            ),
            "paired_full_minus_detection_only": (
                baseline_paired
            ),
        },
    }

    DESTINATION.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    DESTINATION.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("report_path=", DESTINATION)
    print("git_commit=", git_commit)
    print(
        "detection_map=",
        aggregate[
            "detection_only"
        ]["mAP50-95"],
    )
    print(
        "classification_distillation_map=",
        aggregate[
            "classification_distillation"
        ]["mAP50-95"],
    )
    print(
        "paired_map_gain=",
        paired_aggregate["mAP50-95"],
    )
    print(
        "paired_map50_gain=",
        paired_aggregate["mAP50"],
    )
    print(
        "paired_ap75_gain=",
        paired_aggregate["AP75"],
    )
    print(
        "detection_aware_classification_"
        "multiseed_report_ok=True"
    )


if __name__ == "__main__":
    main()