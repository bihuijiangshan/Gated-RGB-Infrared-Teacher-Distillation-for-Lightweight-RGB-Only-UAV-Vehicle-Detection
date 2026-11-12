"""Build three-seed gated-teacher distillation evidence."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rgbt_uav.evaluation.gated_distillation_report import (
    METRICS,
    build_multiseed_report,
)

SEEDS = (3407, 3408, 3409)
EXPECTED_EPOCHS = list(range(1, 21))

RUN_TEMPLATES = {
    "detection_only": (
        "runs/paired/"
        "official_rgb_lightweight_"
        "detection_only_seed{seed}"
    ),
    "early_fusion_distillation": (
        "runs/paired/"
        "official_rgb_lightweight_"
        "classification_distillation_seed{seed}"
    ),
    "gated_teacher_distillation": (
        "runs/paired/"
        "official_rgb_lightweight_"
        "gated_classification_distillation_seed{seed}"
    ),
}

EARLY_TEACHER_CHECKPOINT = Path(
    "runs/paired/"
    "early_fusion_teacher_v2_10ep/"
    "best.pt"
)

GATED_TEACHER_CHECKPOINT = Path(
    "runs/paired/"
    "official_gated_teacher_v1/"
    "best_map50_95.pt"
)

OUTPUT_PATH = Path(
    "runs/archives/"
    "official_rgb_lightweight_"
    "gated_distillation_multiseed_20260908/"
    "summary.json"
)

METHOD_EXPECTATIONS = {
    "detection_only": {
        "teacher_kind": "official_early_fusion",
        "teacher_checkpoint": EARLY_TEACHER_CHECKPOINT,
        "teacher_restored_epoch": 10,
        "logit_weight": 0.0,
        "classification_distillation_weight": 0.0,
    },
    "early_fusion_distillation": {
        "teacher_kind": "official_early_fusion",
        "teacher_checkpoint": EARLY_TEACHER_CHECKPOINT,
        "teacher_restored_epoch": 10,
        "logit_weight": 1.0,
        "classification_distillation_weight": 1.0,
    },
    "gated_teacher_distillation": {
        "teacher_kind": "official_gated",
        "teacher_checkpoint": GATED_TEACHER_CHECKPOINT,
        "teacher_restored_epoch": 8,
        "logit_weight": 1.0,
        "classification_distillation_weight": 1.0,
    },
}


class GatedDistillationEvidenceError(RuntimeError):
    """Raised when real multiseed evidence is inconsistent."""


def read_json(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise GatedDistillationEvidenceError(
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
        raise GatedDistillationEvidenceError(
            f"failed to read {label}: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise GatedDistillationEvidenceError(
            f"{label} root must be a mapping"
        )

    return payload


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise GatedDistillationEvidenceError(
            f"required file is missing: {path}"
        )

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
        raise GatedDistillationEvidenceError(
            f"{label} must be numeric"
        )

    number = float(value)

    if not math.isfinite(number):
        raise GatedDistillationEvidenceError(
            f"{label} must be finite"
        )

    return number


def require_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GatedDistillationEvidenceError(
            f"{label} must be a mapping"
        )

    return value


def require_equal(
    actual: object,
    expected: object,
    *,
    label: str,
) -> None:
    if actual != expected:
        raise GatedDistillationEvidenceError(
            f"{label} mismatch: "
            f"{actual!r} != {expected!r}"
        )


def require_path_equal(
    actual: object,
    expected: Path,
    *,
    label: str,
) -> None:
    if not isinstance(actual, str):
        raise GatedDistillationEvidenceError(
            f"{label} must be a path string"
        )

    if Path(actual) != expected:
        raise GatedDistillationEvidenceError(
            f"{label} mismatch: "
            f"{Path(actual)!s} != {expected!s}"
        )


def git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise GatedDistillationEvidenceError(
            "failed to read git metadata"
        ) from exc

    if status:
        raise GatedDistillationEvidenceError(
            "tracked working tree must be clean"
        )

    return {
        "commit": commit,
        "tracked_worktree_clean": True,
    }


def validate_completion(
    run_root: Path,
    *,
    seed: int,
    method: str,
) -> dict[str, Any]:
    path = run_root / "completion.json"
    completion = read_json(
        path,
        label=f"seed {seed} {method} completion",
    )

    require_equal(
        completion.get("status"),
        "completed",
        label=f"seed {seed} {method} status",
    )

    config = require_mapping(
        completion.get("config"),
        label=f"seed {seed} {method} config",
    )

    common_expected = {
        "epochs": 20,
        "seed": seed,
        "num_classes": 5,
        "student_model": (
            "configs/models/"
            "yolov8_rgb_student_tiny.yaml"
        ),
        "student_pretrained": "yolov8n.pt",
        "student_pretrained_kind": "width_reduced",
        "feature_weight": 0.0,
        "temperature": 2.0,
        "output_distillation_kind": "detection_aware",
        "dfl_distillation_weight": 0.0,
    }

    for key, expected in common_expected.items():
        require_equal(
            config.get(key),
            expected,
            label=f"seed {seed} {method} config {key}",
        )

    expectation = METHOD_EXPECTATIONS[method]

    teacher_kind = config.get(
        "teacher_kind",
        "official_early_fusion",
    )
    require_equal(
        teacher_kind,
        expectation["teacher_kind"],
        label=f"seed {seed} {method} teacher kind",
    )

    require_path_equal(
        config.get("teacher_checkpoint"),
        expectation["teacher_checkpoint"],
        label=(
            f"seed {seed} {method} "
            "teacher checkpoint"
        ),
    )

    require_equal(
        config.get("logit_weight"),
        expectation["logit_weight"],
        label=f"seed {seed} {method} logit weight",
    )

    require_equal(
        config.get(
            "classification_distillation_weight"
        ),
        expectation[
            "classification_distillation_weight"
        ],
        label=(
            f"seed {seed} {method} "
            "classification distillation weight"
        ),
    )

    dataset = require_mapping(
        completion.get("dataset"),
        label=f"seed {seed} {method} dataset",
    )

    require_equal(
        dataset.get("train_count"),
        900,
        label=f"seed {seed} {method} train count",
    )
    require_equal(
        dataset.get("val_count"),
        74,
        label=f"seed {seed} {method} val count",
    )

    teacher = require_mapping(
        completion.get("teacher"),
        label=f"seed {seed} {method} teacher",
    )

    teacher_metadata_kind = teacher.get(
        "kind",
        "official_early_fusion",
    )
    require_equal(
        teacher_metadata_kind,
        expectation["teacher_kind"],
        label=(
            f"seed {seed} {method} "
            "teacher metadata kind"
        ),
    )

    require_path_equal(
        teacher.get("checkpoint"),
        expectation["teacher_checkpoint"],
        label=(
            f"seed {seed} {method} "
            "teacher metadata checkpoint"
        ),
    )

    require_equal(
        teacher.get("restored_epoch"),
        expectation["teacher_restored_epoch"],
        label=(
            f"seed {seed} {method} "
            "teacher restored epoch"
        ),
    )

    student = require_mapping(
        completion.get("student"),
        label=f"seed {seed} {method} student",
    )

    require_equal(
        student.get("pretrained_kind"),
        "width_reduced",
        label=(
            f"seed {seed} {method} "
            "student pretrained kind"
        ),
    )

    require_equal(
        completion.get("all_completed_epochs"),
        EXPECTED_EPOCHS,
        label=(
            f"seed {seed} {method} "
            "all completed epochs"
        ),
    )

    epoch_weights = completion.get("epoch_weights")
    if (
        not isinstance(epoch_weights, list)
        or len(epoch_weights) != 20
    ):
        raise GatedDistillationEvidenceError(
            f"seed {seed} {method} must retain "
            "20 epoch checkpoints"
        )

    metrics = require_mapping(
        completion.get("metrics"),
        label=(
            f"seed {seed} {method} "
            "completion metrics"
        ),
    )

    val_loss = finite_float(
        metrics.get("val_loss"),
        label=f"seed {seed} {method} val loss",
    )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "experiment_id": completion.get(
            "experiment_id"
        ),
        "val_loss_selected_by_runner": val_loss,
        "teacher_kind": teacher_kind,
        "teacher_checkpoint": str(
            expectation["teacher_checkpoint"]
        ),
        "teacher_checkpoint_sha256": sha256_file(
            expectation["teacher_checkpoint"]
        ),
        "teacher_restored_epoch": expectation[
            "teacher_restored_epoch"
        ],
    }


def validate_selection(
    run_root: Path,
    *,
    seed: int,
    method: str,
) -> dict[str, Any]:
    path = run_root / "best_map50_95.json"
    selection = read_json(
        path,
        label=f"seed {seed} {method} selection",
    )

    require_equal(
        selection.get("status"),
        "completed",
        label=(
            f"seed {seed} {method} "
            "selection status"
        ),
    )

    require_equal(
        selection.get("selection_metric"),
        "mAP50-95",
        label=(
            f"seed {seed} {method} "
            "selection metric"
        ),
    )

    require_equal(
        selection.get("evaluated_epochs"),
        EXPECTED_EPOCHS,
        label=(
            f"seed {seed} {method} "
            "evaluated epochs"
        ),
    )

    selected_epoch = selection.get("selected_epoch")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch not in EXPECTED_EPOCHS
    ):
        raise GatedDistillationEvidenceError(
            f"seed {seed} {method} "
            "selected epoch is invalid"
        )

    checkpoint_value = selection.get(
        "selected_checkpoint"
    )
    if (
        not isinstance(checkpoint_value, str)
        or not checkpoint_value
    ):
        raise GatedDistillationEvidenceError(
            f"seed {seed} {method} "
            "selected checkpoint is invalid"
        )

    checkpoint = Path(checkpoint_value)
    expected_checkpoint = (
        run_root / "best_map50_95.pt"
    )

    if checkpoint != expected_checkpoint:
        raise GatedDistillationEvidenceError(
            f"seed {seed} {method} "
            "selected checkpoint path mismatch"
        )

    actual_hash = sha256_file(checkpoint)
    reported_hash = selection.get("sha256")

    if (
        not isinstance(reported_hash, str)
        or reported_hash.lower() != actual_hash
    ):
        raise GatedDistillationEvidenceError(
            f"seed {seed} {method} "
            "checkpoint SHA256 mismatch"
        )

    candidates = selection.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 20
    ):
        raise GatedDistillationEvidenceError(
            f"seed {seed} {method} must contain "
            "20 evaluated candidates"
        )

    candidate_epochs = [
        candidate.get("epoch")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    require_equal(
        candidate_epochs,
        EXPECTED_EPOCHS,
        label=(
            f"seed {seed} {method} "
            "candidate epochs"
        ),
    )

    candidate_maps = [
        finite_float(
            candidate.get("mAP50-95"),
            label=(
                f"seed {seed} {method} "
                "candidate mAP50-95"
            ),
        )
        for candidate in candidates
        if isinstance(candidate, dict)
    ]

    selected_map = finite_float(
        selection.get("mAP50-95"),
        label=(
            f"seed {seed} {method} "
            "selected mAP50-95"
        ),
    )

    if not math.isclose(
        selected_map,
        max(candidate_maps),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise GatedDistillationEvidenceError(
            f"seed {seed} {method} checkpoint "
            "was not selected by maximum mAP50-95"
        )

    metrics = {
        metric: finite_float(
            selection.get(metric),
            label=(
                f"seed {seed} {method} {metric}"
            ),
        )
        for metric in METRICS
    }

    return {
        "selected_epoch": selected_epoch,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": actual_hash,
        "selection_evidence": str(path),
        "selection_evidence_sha256": sha256_file(path),
        **metrics,
    }


def add_relative_mean_gains(
    report: dict[str, object],
) -> None:
    aggregate = report["aggregate"]
    paired = report["paired"]

    references = {
        (
            "gated_teacher_distillation_"
            "minus_detection_only"
        ): "detection_only",
        (
            "gated_teacher_distillation_"
            "minus_early_fusion_distillation"
        ): "early_fusion_distillation",
    }

    for comparison, reference in references.items():
        for metric in METRICS:
            reference_mean = aggregate[
                reference
            ][metric]["mean"]
            delta_mean = paired[
                comparison
            ][metric]["mean"]

            paired[comparison][metric][
                "relative_mean_gain"
            ] = delta_mean / reference_mean


def main() -> None:
    git = git_metadata()

    runs: dict[
        int,
        dict[str, dict[str, object]],
    ] = {}

    evidence: dict[
        str,
        dict[str, dict[str, object]],
    ] = {}

    for seed in SEEDS:
        runs[seed] = {}
        evidence[str(seed)] = {}

        for method, template in RUN_TEMPLATES.items():
            run_root = Path(
                template.format(seed=seed)
            )

            completion = validate_completion(
                run_root,
                seed=seed,
                method=method,
            )
            selection = validate_selection(
                run_root,
                seed=seed,
                method=method,
            )

            runs[seed][method] = {
                **selection,
                "experiment_id": completion[
                    "experiment_id"
                ],
                "val_loss_selected_by_runner": (
                    completion[
                        "val_loss_selected_by_runner"
                    ]
                ),
                "teacher_kind": completion[
                    "teacher_kind"
                ],
                "teacher_checkpoint": completion[
                    "teacher_checkpoint"
                ],
                "teacher_checkpoint_sha256": (
                    completion[
                        "teacher_checkpoint_sha256"
                    ]
                ),
                "teacher_restored_epoch": completion[
                    "teacher_restored_epoch"
                ],
            }

            evidence[str(seed)][method] = {
                "run_root": str(run_root),
                "completion": completion,
                "selection_path": selection[
                    "selection_evidence"
                ],
                "selection_sha256": selection[
                    "selection_evidence_sha256"
                ],
            }

    report = build_multiseed_report(
        runs=runs,
        seeds=SEEDS,
    )

    add_relative_mean_gains(report)

    gated_vs_baseline = report["paired"][
        "gated_teacher_distillation_"
        "minus_detection_only"
    ]["mAP50-95"]

    gated_vs_early = report["paired"][
        "gated_teacher_distillation_"
        "minus_early_fusion_distillation"
    ]["mAP50-95"]

    report.update(
        {
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "scope": (
                "DroneVehicle 5% three-seed "
                "preflight; not a formal thesis result"
            ),
            "git": git,
            "dataset": {
                "name": "DroneVehicle smoke 5%",
                "train_count": 900,
                "val_count": 74,
                "test_used": False,
            },
            "evaluation": {
                "selection_split": "val",
                "selection_metric": "mAP50-95",
                "score_threshold": 0.001,
                "iou_threshold": 0.7,
                "max_det_before_coco": 300,
                "coco_max_dets": 100,
            },
            "method": {
                "student_model": (
                    "configs/models/"
                    "yolov8_rgb_student_tiny.yaml"
                ),
                "student_pretrained": "yolov8n.pt",
                "student_pretrained_kind": (
                    "width_reduced"
                ),
                "epochs": 20,
                "feature_weight": 0.0,
                "classification_distillation_weight": 1.0,
                "dfl_distillation_weight": 0.0,
                "temperature": 2.0,
                "early_fusion_teacher": str(
                    EARLY_TEACHER_CHECKPOINT
                ),
                "gated_teacher": str(
                    GATED_TEACHER_CHECKPOINT
                ),
            },
            "evidence": evidence,
            "interpretation": {
                "gated_minus_detection_only": {
                    "mean_mAP50_95_gain": (
                        gated_vs_baseline["mean"]
                    ),
                    "relative_mean_gain": (
                        gated_vs_baseline[
                            "relative_mean_gain"
                        ]
                    ),
                    "positive_seed_count": (
                        gated_vs_baseline[
                            "positive_seed_count"
                        ]
                    ),
                    "negative_seed_count": (
                        gated_vs_baseline[
                            "negative_seed_count"
                        ]
                    ),
                    "stable_across_all_seeds": (
                        gated_vs_baseline[
                            "positive_seed_count"
                        ]
                        == len(SEEDS)
                    ),
                },
                "gated_minus_early_fusion": {
                    "mean_mAP50_95_gain": (
                        gated_vs_early["mean"]
                    ),
                    "relative_mean_gain": (
                        gated_vs_early[
                            "relative_mean_gain"
                        ]
                    ),
                    "positive_seed_count": (
                        gated_vs_early[
                            "positive_seed_count"
                        ]
                    ),
                    "negative_seed_count": (
                        gated_vs_early[
                            "negative_seed_count"
                        ]
                    ),
                    "stable_across_all_seeds": (
                        gated_vs_early[
                            "positive_seed_count"
                        ]
                        == len(SEEDS)
                    ),
                },
                "supported_claims": [
                    (
                        "The gated-teacher variant has "
                        "higher mean validation mAP50-95 "
                        "than both comparison methods."
                    ),
                    (
                        "The gated-teacher variant improves "
                        "mAP50-95 in two of three student seeds."
                    ),
                ],
                "unsupported_claims": [
                    (
                        "Gated-teacher distillation improves "
                        "mAP50-95 for every random seed."
                    ),
                    (
                        "The 5% preflight results constitute "
                        "formal independent-test evidence."
                    ),
                ],
            },
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate = report["aggregate"]

    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "git_commit": git["commit"],
                "detection_only_mAP50_95": (
                    aggregate[
                        "detection_only"
                    ]["mAP50-95"]
                ),
                "early_fusion_distillation_mAP50_95": (
                    aggregate[
                        "early_fusion_distillation"
                    ]["mAP50-95"]
                ),
                "gated_teacher_distillation_mAP50_95": (
                    aggregate[
                        "gated_teacher_distillation"
                    ]["mAP50-95"]
                ),
                "gated_minus_detection_only": (
                    gated_vs_baseline
                ),
                "gated_minus_early_fusion": (
                    gated_vs_early
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        "gated_lightweight_distillation_"
        "multiseed_report_ok=True"
    )


if __name__ == "__main__":
    main()