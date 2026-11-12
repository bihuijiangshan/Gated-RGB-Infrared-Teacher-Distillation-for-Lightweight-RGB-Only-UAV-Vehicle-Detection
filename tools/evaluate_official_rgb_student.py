"""Evaluate the official RGB distilled student with COCO metrics."""

from __future__ import annotations

import json
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
CHECKPOINT_PATH = Path(
    "runs/paired/official_rgb_distillation_v1/best.pt"
)
METRICS_PATH = Path(
    "runs/paired/official_rgb_distillation_v1/"
    "paired_coco_metrics.json"
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


def _per_class_ap_small(
    evaluator: COCOeval,
) -> dict[str, float | None]:
    precision = evaluator.eval.get("precision")
    if not isinstance(precision, np.ndarray):
        raise RuntimeError(
            "COCO evaluation is missing precision data"
        )

    result: dict[str, float | None] = {}

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        values = precision[
            :,
            :,
            class_index,
            1,
            2,
        ]
        valid = values[values > -1]

        result[class_name] = (
            float(valid.mean())
            if valid.size
            else None
        )

    return result


def main() -> None:
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = PairedYoloDataset(
        DATA_ROOT,
        split="val",
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        collate_fn=collate_paired_batch,
    )

    student = build_official_rgb_student(
        model_name="yolov8n.yaml",
        num_classes=NUM_CLASSES,
        device=device,
    )

    payload = torch.load(
        CHECKPOINT_PATH,
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
        score_threshold=SCORE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        max_det=MAX_DET,
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
            "per_class_AP_small": (
                _per_class_ap_small(evaluator)
            ),
        }
    else:
        metrics = {
            "mAP50-95": 0.0,
            "mAP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AR_small": 0.0,
            "per_class_AP_small": {
                name: None
                for name in CLASS_NAMES
            },
        }

    report = {
        "device": str(device),
        "checkpoint": str(CHECKPOINT_PATH),
        "restored_epoch": restored_epoch,
        "images": len(results),
        "prediction_boxes": len(predictions),
        **metrics,
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    METRICS_PATH.write_text(
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
    assert all(
        np.isfinite(value)
        for key, value in metrics.items()
        if key != "per_class_AP_small"
    )

    print(
        "official_rgb_student_coco_evaluation_ok=True"
    )


if __name__ == "__main__":
    main()