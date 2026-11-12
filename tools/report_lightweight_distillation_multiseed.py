"""Build the three-seed lightweight distillation evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEEDS = (3407, 3408, 3409)

METRICS = (
    "mAP50-95",
    "mAP50",
    "AP75",
    "AP_small",
    "AR_small",
)

BASELINE_TEMPLATE = (
    "runs/paired/"
    "official_rgb_lightweight_detection_only_seed{seed}"
)

DISTILLED_TEMPLATE = (
    "runs/paired/"
    "official_rgb_lightweight_"
    "classification_distillation_seed{seed}"
)

PROFILE_PATH = Path(
    "runs/archives/"
    "official_lightweight_teacher_student_"
    "profile_seed3407_20260830/profile.json"
)

DEFAULT_OUTPUT = Path(
    "runs/archives/"
    "official_rgb_lightweight_"
    "distillation_multiseed_20260831/"
    "summary.json"
)


class LightweightReportError(RuntimeError):
    """Raised when lightweight evidence is incomplete or inconsistent."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize three-seed lightweight "
            "detection-only and distillation runs"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=PROFILE_PATH,
    )
    return parser


def load_json(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise LightweightReportError(
            f"{label} must be a regular file: {path}"
        )

    try:
        payload: Any = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise LightweightReportError(
            f"failed to read {label}: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise LightweightReportError(
            f"{label} root must be a mapping"
        )

    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def finite_float(
    value: object,
    *,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise LightweightReportError(
            f"{label} must be numeric"
        )

    number = float(value)

    if not math.isfinite(number):
        raise LightweightReportError(
            f"{label} must be finite"
        )

    return number


def positive_int(
    value: object,
    *,
    label: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise LightweightReportError(
            f"{label} must be a positive integer"
        )

    return value


def aggregate(
    values: list[float],
) -> dict[str, float]:
    if len(values) < 2:
        raise LightweightReportError(
            "aggregate requires at least two values"
        )

    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def paired_aggregate(
    values: list[float],
) -> dict[str, float | int]:
    result: dict[str, float | int] = aggregate(
        values
    )

    result.update(
        {
            "positive_seed_count": sum(
                value > 0.0
                for value in values
            ),
            "negative_seed_count": sum(
                value < 0.0
                for value in values
            ),
            "zero_seed_count": sum(
                value == 0.0
                for value in values
            ),
        }
    )

    return result


def validate_completion(
    run_root: Path,
    *,
    seed: int,
    classification_weight: float,
) -> dict[str, Any]:
    completion = load_json(
        run_root / "completion.json",
        label=f"seed {seed} completion",
    )

    if completion.get("status") != "completed":
        raise LightweightReportError(
            f"seed {seed} run is not completed"
        )

    config = completion.get("config")
    if not isinstance(config, dict):
        raise LightweightReportError(
            f"seed {seed} completion config must be a mapping"
        )

    expected = {
        "epochs": 20,
        "seed": seed,
        "num_classes": 5,
        "resume": True,
        "student_model": (
            "configs/models/"
            "yolov8_rgb_student_tiny.yaml"
        ),
        "student_pretrained": "yolov8n.pt",
        "student_pretrained_kind": "width_reduced",
        "feature_weight": 0.0,
        "output_distillation_kind": "detection_aware",
        "classification_distillation_weight": (
            classification_weight
        ),
        "dfl_distillation_weight": 0.0,
    }

    # Fresh seed 3408/3409 runs retain resume=false in their
    # completion evidence, while seed 3407 was resumed to epoch 20.
    expected_resume = seed == 3407

    for key, expected_value in expected.items():
        if key == "resume":
            expected_value = expected_resume

        if config.get(key) != expected_value:
            raise LightweightReportError(
                f"seed {seed} config mismatch for {key}: "
                f"{config.get(key)!r} != {expected_value!r}"
            )

    dataset = completion.get("dataset")
    if not isinstance(dataset, dict):
        raise LightweightReportError(
            f"seed {seed} dataset metadata must be a mapping"
        )

    if dataset.get("train_count") != 900:
        raise LightweightReportError(
            f"seed {seed} train count mismatch"
        )
    if dataset.get("val_count") != 74:
        raise LightweightReportError(
            f"seed {seed} validation count mismatch"
        )

    student = completion.get("student")
    if not isinstance(student, dict):
        raise LightweightReportError(
            f"seed {seed} student metadata must be a mapping"
        )

    if student.get("pretrained_kind") != "width_reduced":
        raise LightweightReportError(
            f"seed {seed} did not use width-reduced initialization"
        )

    return completion


def load_selection(
    run_root: Path,
    *,
    seed: int,
    variant: str,
) -> dict[str, Any]:
    evidence_path = run_root / "best_map50_95.json"
    payload = load_json(
        evidence_path,
        label=f"seed {seed} {variant} selection",
    )

    if payload.get("status") != "completed":
        raise LightweightReportError(
            f"seed {seed} {variant} selection is incomplete"
        )

    epoch = positive_int(
        payload.get("selected_epoch"),
        label=f"seed {seed} {variant} epoch",
    )

    if epoch > 20:
        raise LightweightReportError(
            f"seed {seed} {variant} epoch exceeds 20"
        )

    checkpoint_value = payload.get(
        "selected_checkpoint"
    )
    if (
        not isinstance(checkpoint_value, str)
        or not checkpoint_value
    ):
        raise LightweightReportError(
            f"seed {seed} {variant} checkpoint is invalid"
        )

    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_file():
        raise LightweightReportError(
            f"seed {seed} {variant} checkpoint is missing: "
            f"{checkpoint}"
        )

    reported_hash = payload.get("sha256")
    if not isinstance(reported_hash, str):
        raise LightweightReportError(
            f"seed {seed} {variant} SHA256 is invalid"
        )

    actual_hash = sha256_file(checkpoint)
    if actual_hash != reported_hash.lower():
        raise LightweightReportError(
            f"seed {seed} {variant} checkpoint hash mismatch"
        )

    metrics = {
        name: finite_float(
            payload.get(name),
            label=f"seed {seed} {variant} {name}",
        )
        for name in METRICS
    }

    return {
        "selected_epoch": epoch,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": actual_hash,
        "selection_evidence": str(evidence_path),
        **metrics,
    }


def git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        status = subprocess.run(
            [
                "git",
                "status",
                "--short",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise LightweightReportError(
            "could not read git metadata"
        ) from exc

    return {
        "commit": commit,
        "worktree_dirty": bool(status),
        "status_short": (
            status.splitlines()
            if status
            else []
        ),
    }


def efficiency_summary(
    profile: dict[str, Any],
) -> dict[str, object]:
    if profile.get("status") != "completed":
        raise LightweightReportError(
            "profiling evidence is incomplete"
        )

    teacher = profile.get("teacher")
    student = profile.get("student")
    comparison = profile.get("comparison")

    if not all(
        isinstance(value, dict)
        for value in (
            teacher,
            student,
            comparison,
        )
    ):
        raise LightweightReportError(
            "profiling sections must be mappings"
        )

    teacher_parameters = positive_int(
        teacher["parameters"]["total"],
        label="teacher parameters",
    )
    student_parameters = positive_int(
        student["parameters"]["total"],
        label="student parameters",
    )

    teacher_checkpoint_bytes = positive_int(
        teacher["checkpoint_bytes"],
        label="teacher checkpoint bytes",
    )
    student_checkpoint_bytes = positive_int(
        student["checkpoint_bytes"],
        label="student checkpoint bytes",
    )

    teacher_peak_memory = finite_float(
        teacher["latency"]["peak_memory_mib"],
        label="teacher peak memory",
    )
    student_peak_memory = finite_float(
        student["latency"]["peak_memory_mib"],
        label="student peak memory",
    )

    return {
        "scope": profile.get("scope"),
        "device": profile.get("device"),
        "profiling": profile.get("profiling"),
        "teacher": {
            "model": teacher.get("model"),
            "parameters": teacher_parameters,
            "gflops": finite_float(
                teacher.get("gflops"),
                label="teacher GFLOPs",
            ),
            "checkpoint_bytes": teacher_checkpoint_bytes,
            "checkpoint_sha256": teacher.get(
                "checkpoint_sha256"
            ),
            "mean_latency_ms": finite_float(
                teacher["latency"]["mean_ms"],
                label="teacher latency",
            ),
            "peak_memory_mib": teacher_peak_memory,
        },
        "student": {
            "model": student.get("model"),
            "parameters": student_parameters,
            "gflops": finite_float(
                student.get("gflops"),
                label="student GFLOPs",
            ),
            "checkpoint_bytes": student_checkpoint_bytes,
            "checkpoint_sha256": student.get(
                "checkpoint_sha256"
            ),
            "mean_latency_ms": finite_float(
                student["latency"]["mean_ms"],
                label="student latency",
            ),
            "peak_memory_mib": student_peak_memory,
        },
        "comparison": {
            **comparison,
            "checkpoint_reduction_fraction": (
                1.0
                - student_checkpoint_bytes
                / teacher_checkpoint_bytes
            ),
            "peak_memory_reduction_fraction": (
                1.0
                - student_peak_memory
                / teacher_peak_memory
            ),
        },
    }


def main() -> None:
    args = build_parser().parse_args()

    runs: dict[str, object] = {}
    baseline_values = {
        name: []
        for name in METRICS
    }
    distilled_values = {
        name: []
        for name in METRICS
    }
    paired_values = {
        name: []
        for name in METRICS
    }

    for seed in SEEDS:
        baseline_root = Path(
            BASELINE_TEMPLATE.format(seed=seed)
        )
        distilled_root = Path(
            DISTILLED_TEMPLATE.format(seed=seed)
        )

        validate_completion(
            baseline_root,
            seed=seed,
            classification_weight=0.0,
        )
        validate_completion(
            distilled_root,
            seed=seed,
            classification_weight=1.0,
        )

        baseline = load_selection(
            baseline_root,
            seed=seed,
            variant="detection_only",
        )
        distilled = load_selection(
            distilled_root,
            seed=seed,
            variant="classification_distillation",
        )

        delta = {
            name: (
                float(distilled[name])
                - float(baseline[name])
            )
            for name in METRICS
        }

        for name in METRICS:
            baseline_values[name].append(
                float(baseline[name])
            )
            distilled_values[name].append(
                float(distilled[name])
            )
            paired_values[name].append(
                delta[name]
            )

        runs[str(seed)] = {
            "detection_only": baseline,
            "classification_distillation": distilled,
            "classification_distillation_minus_detection_only": (
                delta
            ),
        }

    profile = load_json(
        args.profile,
        label="profiling evidence",
    )

    seed3407_student_hash = runs[
        "3407"
    ]["classification_distillation"][
        "selected_checkpoint_sha256"
    ]

    profile_student_hash = profile[
        "student"
    ]["checkpoint_sha256"]

    if seed3407_student_hash != profile_student_hash:
        raise LightweightReportError(
            "profiled student does not match seed 3407 "
            "selected checkpoint"
        )

    aggregate_results = {
        "detection_only": {
            name: aggregate(values)
            for name, values in baseline_values.items()
        },
        "classification_distillation": {
            name: aggregate(values)
            for name, values in distilled_values.items()
        },
        "paired_classification_distillation_minus_detection_only": {
            name: paired_aggregate(values)
            for name, values in paired_values.items()
        },
    }

    paired_map = aggregate_results[
        "paired_classification_distillation_minus_detection_only"
    ]["mAP50-95"]

    report = {
        "format_version": 1,
        "status": "completed",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "scope": (
            "DroneVehicle 5% three-seed preflight; "
            "not a formal thesis result"
        ),
        "dataset": {
            "name": "DroneVehicle smoke 5%",
            "train_count": 900,
            "val_count": 74,
        },
        "seeds": list(SEEDS),
        "method": {
            "teacher": (
                "official RGB-T early-fusion teacher v2"
            ),
            "student": (
                "width-reduced official RGB student"
            ),
            "student_model": (
                "configs/models/"
                "yolov8_rgb_student_tiny.yaml"
            ),
            "epochs": 20,
            "student_pretrained": "yolov8n.pt",
            "student_pretrained_kind": "width_reduced",
            "feature_weight": 0.0,
            "classification_distillation_weight": 1.0,
            "dfl_distillation_weight": 0.0,
            "temperature": 2.0,
        },
        "evaluation": {
            "selection_metric": "mAP50-95",
            "score_threshold": 0.001,
            "iou_threshold": 0.7,
            "max_det_before_coco": 300,
            "coco_max_dets": 100,
        },
        "runs": runs,
        "aggregate": aggregate_results,
        "efficiency": efficiency_summary(profile),
        "profiling_evidence": {
            "path": str(args.profile),
            "sha256": sha256_file(args.profile),
        },
        "interpretation": {
            "stable_mAP50_95_improvement": (
                paired_map["positive_seed_count"]
                == len(SEEDS)
            ),
            "mAP50_95_positive_seed_count": (
                paired_map["positive_seed_count"]
            ),
            "mAP50_95_negative_seed_count": (
                paired_map["negative_seed_count"]
            ),
            "supported_claims": [
                (
                    "The lightweight student substantially "
                    "reduces parameters and GFLOPs."
                ),
                (
                    "The measured batch-1 FP32 CUDA latency "
                    "improvement is modest."
                ),
            ],
            "unsupported_claims": [
                (
                    "Classification distillation consistently "
                    "improves lightweight-student mAP50-95."
                ),
                (
                    "The lightweight student provides a large "
                    "batch-1 CUDA speedup."
                ),
            ],
        },
        "git": git_metadata(),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "baseline_mAP50_95": (
                    aggregate_results[
                        "detection_only"
                    ]["mAP50-95"]
                ),
                "distilled_mAP50_95": (
                    aggregate_results[
                        "classification_distillation"
                    ]["mAP50-95"]
                ),
                "paired_mAP50_95": paired_map,
                "parameter_reduction_fraction": (
                    report["efficiency"][
                        "comparison"
                    ][
                        "parameter_reduction_fraction"
                    ]
                ),
                "gflops_reduction_fraction": (
                    report["efficiency"][
                        "comparison"
                    ][
                        "gflops_reduction_fraction"
                    ]
                ),
                "latency_speedup": (
                    report["efficiency"][
                        "comparison"
                    ]["latency_speedup"]
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("lightweight_multiseed_report_ok=True")


if __name__ == "__main__":
    main()