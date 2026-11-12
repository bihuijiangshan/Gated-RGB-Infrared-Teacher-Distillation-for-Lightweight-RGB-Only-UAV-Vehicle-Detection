"""Evaluate an official gated RGB/IR teacher with COCO metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.evaluation.official_gated_teacher import (
    run_official_gated_teacher_inference,
)
from rgbt_uav.evaluation.paired import (
    evaluate_paired_coco,
    paired_samples_to_coco_ground_truth,
)
from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)

CLASS_NAMES = (
    "car",
    "truck",
    "bus",
    "van",
    "freight_car",
)

LABEL_TO_CATEGORY_ID = {
    index: index + 1
    for index in range(len(CLASS_NAMES))
}

CATEGORY_NAMES = {
    index + 1: name
    for index, name in enumerate(CLASS_NAMES)
}


class OfficialGatedTeacherEvaluationError(
    RuntimeError
):
    """Raised when gated-teacher evaluation cannot be completed."""


def _device(value: str) -> torch.device:
    normalized = value.strip().lower()

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized.startswith("cuda"):
        device = torch.device(normalized)
    elif normalized.isdigit():
        device = torch.device(
            f"cuda:{normalized}"
        )
    else:
        raise OfficialGatedTeacherEvaluationError(
            "device must be 'cpu', a CUDA index, "
            "or a CUDA device such as 'cuda:0'"
        )

    if not torch.cuda.is_available():
        raise OfficialGatedTeacherEvaluationError(
            "CUDA was requested but is unavailable"
        )

    index = (
        device.index
        if device.index is not None
        else 0
    )

    if index >= torch.cuda.device_count():
        raise OfficialGatedTeacherEvaluationError(
            f"CUDA device index does not exist: {index}"
        )

    return torch.device(
        f"cuda:{index}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(
            1024 * 1024
        ):
            digest.update(block)

    return digest.hexdigest()


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise OfficialGatedTeacherEvaluationError(
            f"checkpoint is not a regular file: {path}"
        )

    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherEvaluationError(
            f"could not load checkpoint: {path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise OfficialGatedTeacherEvaluationError(
            "checkpoint root must be a mapping"
        )

    required = {
        "format_version",
        "epoch",
        "experiment_id",
        "model_name",
        "pretrained",
        "teacher",
        "metrics",
    }

    missing = required - set(payload)

    if missing:
        raise OfficialGatedTeacherEvaluationError(
            "checkpoint is missing fields: "
            + ", ".join(sorted(missing))
        )

    if payload["format_version"] != 1:
        raise OfficialGatedTeacherEvaluationError(
            "unsupported checkpoint format version"
        )

    epoch = payload["epoch"]

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch <= 0
    ):
        raise OfficialGatedTeacherEvaluationError(
            "checkpoint epoch must be positive"
        )

    for field in (
        "experiment_id",
        "model_name",
        "pretrained",
    ):
        value = payload[field]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise OfficialGatedTeacherEvaluationError(
                f"checkpoint {field} must be non-empty"
            )

    if not isinstance(
        payload["teacher"],
        Mapping,
    ):
        raise OfficialGatedTeacherEvaluationError(
            "checkpoint teacher state must be a mapping"
        )

    return payload


def _prediction_category_counts(
    predictions: list[dict[str, object]],
) -> dict[str, int]:
    counts: Counter[int] = Counter()

    for result in predictions:
        labels = result.get("labels")

        if not isinstance(labels, list):
            raise OfficialGatedTeacherEvaluationError(
                "prediction labels must be a list"
            )

        for label in labels:
            if (
                isinstance(label, bool)
                or not isinstance(label, int)
                or label < 0
                or label >= len(CLASS_NAMES)
            ):
                raise OfficialGatedTeacherEvaluationError(
                    f"invalid prediction label: {label!r}"
                )

            counts[label] += 1

    return {
        name: counts[index]
        for index, name in enumerate(
            CLASS_NAMES
        )
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an official gated RGB/IR "
            "teacher with COCO metrics"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "data/processed/yolo_smoke_5pct"
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--device",
        default="0",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> None:
    args = build_parser().parse_args(argv)

    if args.batch <= 0:
        raise OfficialGatedTeacherEvaluationError(
            "batch must be positive"
        )

    if args.workers < 0:
        raise OfficialGatedTeacherEvaluationError(
            "workers must be non-negative"
        )

    if not 0.0 <= args.score_threshold <= 1.0:
        raise OfficialGatedTeacherEvaluationError(
            "score threshold must be between 0 and 1"
        )

    if not 0.0 <= args.iou_threshold <= 1.0:
        raise OfficialGatedTeacherEvaluationError(
            "IoU threshold must be between 0 and 1"
        )

    if args.max_det <= 0:
        raise OfficialGatedTeacherEvaluationError(
            "max-det must be positive"
        )

    device = _device(args.device)
    checkpoint_path = args.checkpoint
    payload = _load_checkpoint(
        checkpoint_path,
        device=device,
    )

    model_name = str(
        payload["model_name"]
    )

    teacher = build_official_gated_teacher(
        model_name=model_name,
        num_classes=len(CLASS_NAMES),
        device=device,
    )

    try:
        teacher.load_state_dict(
            payload["teacher"],
            strict=True,
        )
    except (
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise OfficialGatedTeacherEvaluationError(
            "checkpoint teacher state is "
            "incompatible with the model"
        ) from exc

    teacher.eval()

    dataset = PairedYoloDataset(
        args.data_root,
        split="val",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_paired_batch,
        pin_memory=device.type == "cuda",
    )

    predictions = (
        run_official_gated_teacher_inference(
            teacher,
            loader,
            device=device,
            score_threshold=(
                args.score_threshold
            ),
            iou_threshold=args.iou_threshold,
            max_det=args.max_det,
        )
    )

    if len(predictions) != len(dataset):
        raise OfficialGatedTeacherEvaluationError(
            "prediction image count does not "
            "match dataset count"
        )

    pair_ids = [
        result.get("pair_id")
        for result in predictions
    ]

    if (
        any(
            not isinstance(pair_id, str)
            or not pair_id
            for pair_id in pair_ids
        )
        or len(set(pair_ids))
        != len(pair_ids)
    ):
        raise OfficialGatedTeacherEvaluationError(
            "prediction pair IDs must be "
            "unique non-empty strings"
        )

    samples = [
        dataset[index]
        for index in range(len(dataset))
    ]

    ground_truth = (
        paired_samples_to_coco_ground_truth(
            samples,
            label_to_category_id=(
                LABEL_TO_CATEGORY_ID
            ),
            category_names=CATEGORY_NAMES,
        )
    )

    metrics = evaluate_paired_coco(
        samples,
        predictions,
        label_to_category_id=(
            LABEL_TO_CATEGORY_ID
        ),
        category_names=CATEGORY_NAMES,
    )

    prediction_boxes = sum(
        len(result["boxes"])
        for result in predictions
    )

    report = {
        "format_version": 1,
        "status": "completed",
        "experiment_id": str(
            payload["experiment_id"]
        ),
        "device": str(device),
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_sha256": _sha256(
            checkpoint_path
        ),
        "model": model_name,
        "pretrained": str(
            payload["pretrained"]
        ),
        "restored_epoch": int(
            payload["epoch"]
        ),
        "dataset": {
            "name": (
                "DroneVehicle smoke 5%"
            ),
            "split": "val",
            "images": len(dataset),
            "annotations": len(
                ground_truth["annotations"]
            ),
        },
        "evaluation": {
            "score_threshold": float(
                args.score_threshold
            ),
            "iou_threshold": float(
                args.iou_threshold
            ),
            "pre_coco_max_det": int(
                args.max_det
            ),
            "coco_max_det": 100,
        },
        "prediction_images": len(
            predictions
        ),
        "prediction_boxes": (
            prediction_boxes
        ),
        "prediction_category_counts": (
            _prediction_category_counts(
                predictions
            )
        ),
        "mAP50-95": metrics.ap,
        "mAP50": metrics.ap50,
        "AP75": metrics.ap75,
        "AP_small": metrics.ap_small,
        "AR_small": metrics.ar_small,
        "per_class_AP_small": (
            metrics.per_class_ap_small
        ),
    }

    numeric_metrics = (
        report["mAP50-95"],
        report["mAP50"],
        report["AP75"],
        report["AP_small"],
        report["AR_small"],
    )

    if not all(
        isinstance(value, float)
        and math.isfinite(value)
        and value >= 0.0
        for value in numeric_metrics
    ):
        raise OfficialGatedTeacherEvaluationError(
            "COCO metrics must be finite "
            "and non-negative"
        )

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
            report,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        f"metrics_path={args.output}"
    )
    print(
        "official_gated_teacher_"
        "coco_evaluation_ok=True"
    )


if __name__ == "__main__":
    main()