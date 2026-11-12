"""Evaluate the official RGB distilled student with COCO metrics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.evaluation.coco_per_class import (
    summarize_coco_per_class,
)
from rgbt_uav.evaluation.official_rgb_student import (
    run_official_rgb_student_inference,
)
from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
)

CLASS_NAMES = (
    "car",
    "truck",
    "bus",
    "van",
    "freight_car",
)
DATA_ROOT = Path(
    "data/processed/yolo_smoke_5pct"
)

NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 2
WORKERS = 0

SCORE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.7
MAX_DET = 300


def _tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise RuntimeError(
            f"{name} must be a tensor"
        )
    return value


def _build_coco_ground_truth(
    dataset: PairedYoloDataset,
) -> tuple[COCO, dict[str, int]]:
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    pair_to_image_id: dict[str, int] = {}

    annotation_id = 1

    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]

        pair_id = sample["pair_id"]
        if not isinstance(pair_id, str) or not pair_id:
            raise RuntimeError(
                "sample pair_id must be a non-empty string"
            )

        rgb = _tensor(
            sample["rgb"],
            name="sample.rgb",
        )
        boxes = _tensor(
            sample["boxes"],
            name="sample.boxes",
        )
        labels = _tensor(
            sample["labels"],
            name="sample.labels",
        )

        if rgb.ndim != 3:
            raise RuntimeError(
                "sample.rgb must have shape [C, H, W]"
            )
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise RuntimeError(
                "sample.boxes must have shape [N, 4]"
            )
        if labels.ndim != 1:
            raise RuntimeError(
                "sample.labels must have shape [N]"
            )
        if boxes.shape[0] != labels.shape[0]:
            raise RuntimeError(
                "sample box and label counts must match"
            )

        height = int(rgb.shape[-2])
        width = int(rgb.shape[-1])
        image_id = dataset_index + 1

        pair_to_image_id[pair_id] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": pair_id,
                "width": width,
                "height": height,
            }
        )

        for box, label in zip(
            boxes,
            labels,
            strict=True,
        ):
            class_index = int(label.item())
            if not 0 <= class_index < NUM_CLASSES:
                raise RuntimeError(
                    f"invalid target class: {class_index}"
                )

            center_x = float(box[0].item()) * width
            center_y = float(box[1].item()) * height
            box_width = float(box[2].item()) * width
            box_height = float(box[3].item()) * height

            x = center_x - box_width / 2.0
            y = center_y - box_height / 2.0

            x = max(0.0, min(x, float(width)))
            y = max(0.0, min(y, float(height)))
            box_width = max(
                0.0,
                min(box_width, float(width) - x),
            )
            box_height = max(
                0.0,
                min(box_height, float(height) - y),
            )

            if box_width <= 0 or box_height <= 0:
                raise RuntimeError(
                    "ground-truth box must have positive area"
                )

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_index + 1,
                    "bbox": [
                        x,
                        y,
                        box_width,
                        box_height,
                    ],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {
            "description": (
                "DroneVehicle smoke 5% paired validation"
            ),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": index + 1,
                "name": name,
            }
            for index, name in enumerate(CLASS_NAMES)
        ],
    }
    ground_truth.createIndex()

    return ground_truth, pair_to_image_id


def _build_coco_predictions(
    results: list[dict[str, object]],
    pair_to_image_id: dict[str, int],
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []

    for result_index, result in enumerate(results):
        pair_id = result.get("pair_id")
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")

        if not isinstance(pair_id, str):
            raise RuntimeError(
                f"results[{result_index}].pair_id is invalid"
            )
        if pair_id not in pair_to_image_id:
            raise RuntimeError(
                f"unknown prediction pair_id: {pair_id}"
            )
        if not isinstance(boxes, list):
            raise RuntimeError(
                f"results[{result_index}].boxes must be a list"
            )
        if not isinstance(scores, list):
            raise RuntimeError(
                f"results[{result_index}].scores must be a list"
            )
        if not isinstance(labels, list):
            raise RuntimeError(
                f"results[{result_index}].labels must be a list"
            )
        if not (
            len(boxes)
            == len(scores)
            == len(labels)
        ):
            raise RuntimeError(
                "prediction box, score, and label counts must match"
            )

        image_id = pair_to_image_id[pair_id]

        for box, score, label in zip(
            boxes,
            scores,
            labels,
            strict=True,
        ):
            if not isinstance(box, list) or len(box) != 4:
                raise RuntimeError(
                    "prediction box must be an xyxy list"
                )

            x1, y1, x2, y2 = (
                float(value)
                for value in box
            )
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            class_index = int(label)
            if not 0 <= class_index < NUM_CLASSES:
                raise RuntimeError(
                    f"invalid prediction class: {class_index}"
                )

            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": class_index + 1,
                    "bbox": [
                        x1,
                        y1,
                        width,
                        height,
                    ],
                    "score": float(score),
                }
            )

    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an official RGB distilled student "
            "with COCO metrics"
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Official RGB student checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination metrics JSON",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.yaml",
        help="Official RGB student model YAML",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Paired YOLO dataset root",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--device",
        default="0",
        help='CUDA index such as "0", or "cpu"',
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_THRESHOLD,
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=MAX_DET,
    )
    return parser


def _resolve_device(value: str) -> torch.device:
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
        raise ValueError(
            f"unsupported device: {value}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return device

def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.batch <= 0:
        raise ValueError("batch must be positive")
    if args.workers < 0:
        raise ValueError(
            "workers must be non-negative"
        )
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError(
            "score threshold must be between zero and one"
        )
    if not 0.0 < args.iou_threshold <= 1.0:
        raise ValueError(
            "IoU threshold must be in (0, 1]"
        )
    if args.max_det <= 0:
        raise ValueError(
            "max_det must be positive"
        )
    if (
        not isinstance(args.model, str)
        or not args.model.strip()
    ):
        raise ValueError(
            "model must be a non-empty string"
        )

    device = _resolve_device(args.device)

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
    )

    student = build_official_rgb_student(
        model_name=args.model,
        num_classes=NUM_CLASSES,
        device=device,
    )

    payload = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(
            "checkpoint must contain a dictionary"
        )
    if "student" not in payload:
        raise RuntimeError(
            "checkpoint is missing student state"
        )

    student.load_state_dict(
        payload["student"],
        strict=True,
    )
    restored_epoch = int(
        payload.get("epoch", -1)
    )

    results = run_official_rgb_student_inference(
        student,
        loader,
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_det=args.max_det,
    )

    ground_truth, pair_to_image_id = (
        _build_coco_ground_truth(dataset)
    )
    predictions = _build_coco_predictions(
        results,
        pair_to_image_id,
    )

    if predictions:
        detected = ground_truth.loadRes(predictions)
        evaluator = COCOeval(
            ground_truth,
            detected,
            "bbox",
        )
        evaluator.params.imgIds = sorted(
            ground_truth.getImgIds()
        )
        evaluator.params.catIds = list(
            range(1, NUM_CLASSES + 1)
        )
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        stats = evaluator.stats
        metrics = {
            "mAP50-95": float(stats[0]),
            "mAP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AR_small": float(stats[9]),
                        "per_class": summarize_coco_per_class(
                evaluator.eval["precision"],
                iou_thresholds=evaluator.params.iouThrs,
                class_names=CLASS_NAMES,
            ),
        }
    else:
        metrics = {
            "mAP50-95": 0.0,
            "mAP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AR_small": 0.0,
                        "per_class": {
                name: {
                    "AP50-95": None,
                    "AP50": None,
                    "AP75": None,
                    "AP_small": None,
                }
                for name in CLASS_NAMES
            },
        }

    report = {
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "restored_epoch": restored_epoch,
        "images": len(results),
        "prediction_boxes": len(predictions),
        **metrics,
        "evaluation": {
            "score_threshold": args.score_threshold,
            "iou_threshold": args.iou_threshold,
            "max_det_before_coco": args.max_det,
            "coco_max_dets": 100,
        },
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
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    assert restored_epoch > 0
    assert len(results) == len(dataset)
    for metric_name in (
        "mAP50-95",
        "mAP50",
        "AP75",
        "AP_small",
        "AR_small",
    ):
        metric_value = metrics[metric_name]
        assert isinstance(
            metric_value,
            (int, float),
        )
        assert np.isfinite(metric_value)

    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    assert set(per_class) == set(CLASS_NAMES)

    for class_name in CLASS_NAMES:
        class_metrics = per_class[class_name]
        assert isinstance(class_metrics, dict)
        assert set(class_metrics) == {
            "AP50-95",
            "AP50",
            "AP75",
            "AP_small",
        }

        for metric_value in class_metrics.values():
            assert metric_value is None or (
                isinstance(
                    metric_value,
                    (int, float),
                )
                and np.isfinite(metric_value)
            )

    print("official_rgb_student_coco_evaluation_ok=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())