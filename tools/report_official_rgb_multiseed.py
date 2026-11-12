"""Generate paired multi-seed evidence for RGB distillation."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from pathlib import Path

RUNS = {
    3407: {
        "detection_only": Path(
            "runs/paired/official_rgb_detection_only_v1"
        ),
        "full": Path(
            "runs/paired/official_rgb_distillation_v1"
        ),
    },
    3408: {
        "detection_only": Path(
            "runs/paired/official_rgb_detection_only_seed3408"
        ),
        "full": Path(
            "runs/paired/official_rgb_distillation_seed3408"
        ),
    },
    3409: {
        "detection_only": Path(
            "runs/paired/official_rgb_detection_only_seed3409"
        ),
        "full": Path(
            "runs/paired/official_rgb_distillation_seed3409"
        ),
    },
}

DESTINATION = Path(
    "runs/archives/"
    "official_rgb_distillation_multiseed_20260827/"
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
            f"weights are missing: {path}"
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

    seed_results: dict[str, object] = {}
    method_values = {
        "detection_only": {
            metric: []
            for metric in METRICS
        },
        "full": {
            metric: []
            for metric in METRICS
        },
    }
    paired_values = {
        metric: []
        for metric in METRICS
    }

    for seed, methods in RUNS.items():
        per_seed: dict[str, object] = {}

        for method, root in methods.items():
            completion = _read_json(
                root / "completion.json"
            )
            metrics_payload = _read_json(
                root / "paired_coco_metrics.json"
            )

            config = completion.get("config")
            if not isinstance(config, dict):
                raise RuntimeError(
                    f"{root} config metadata is invalid"
                )
            if config.get("seed") != seed:
                raise RuntimeError(
                    f"{root} seed does not match {seed}"
                )

            expected_weights = (
                (0.0, 0.0)
                if method == "detection_only"
                else (1.0, 1.0)
            )
            actual_weights = (
                float(config["feature_weight"]),
                float(config["logit_weight"]),
            )
            if actual_weights != expected_weights:
                raise RuntimeError(
                    f"{root} has incorrect loss weights"
                )

            best_weights_value = completion.get(
                "best_weights"
            )
            if not isinstance(best_weights_value, str):
                raise RuntimeError(
                    f"{root} best_weights is invalid"
                )
            best_weights = Path(best_weights_value)

            metric_values = {
                metric: float(metrics_payload[metric])
                for metric in METRICS
            }
            for metric, value in metric_values.items():
                method_values[method][metric].append(
                    value
                )

            per_seed[method] = {
                **metric_values,
                "val_loss": float(
                    completion["metrics"]["val_loss"]
                ),
                "restored_epoch": int(
                    metrics_payload["restored_epoch"]
                ),
                "best_weights": str(best_weights),
                "best_weights_sha256": _sha256(
                    best_weights
                ),
            }

        detection = per_seed["detection_only"]
        full = per_seed["full"]
        assert isinstance(detection, dict)
        assert isinstance(full, dict)

        paired_delta = {
            metric: (
                float(full[metric])
                - float(detection[metric])
            )
            for metric in METRICS
        }
        for metric, value in paired_delta.items():
            paired_values[metric].append(value)

        seed_results[str(seed)] = {
            **per_seed,
            "full_minus_detection_only": paired_delta,
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
        }
        for metric, values in paired_values.items()
    }

    report = {
        "format_version": 1,
        "status": "completed",
        "scope": (
            "5% three-seed preflight; "
            "not a formal thesis result"
        ),
        "git_commit": git_commit,
        "dataset": {
            "name": "DroneVehicle smoke 5%",
            "train_count": 900,
            "val_count": 74,
        },
        "seeds": sorted(RUNS),
        "evaluation": {
            "score_threshold": 0.001,
            "iou_threshold": 0.7,
            "max_det_before_coco": 300,
            "coco_max_dets": 100,
        },
        "runs": seed_results,
        "aggregate": aggregate,
        "paired_full_minus_detection_only": (
            paired_aggregate
        ),
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
        aggregate["detection_only"]["mAP50-95"],
    )
    print(
        "full_map=",
        aggregate["full"]["mAP50-95"],
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
        "official_rgb_multiseed_report_ok=True"
    )


if __name__ == "__main__":
    main()