from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.evaluation.paired import (
    ultralytics_decoded_to_paired_predictions,
)
from rgbt_uav.training.early_fusion_runtime import (
    build_early_fusion_model,
)

DATA_ROOT = Path("data/processed/yolo_smoke_5pct")
CHECKPOINT_PATH = Path(
    "runs/paired/early_fusion_teacher_v2_10ep/best.pt"
)
OUTPUT_PATH = Path(
    "runs/paired/early_fusion_teacher_v2_10ep/"
    "paired_coco_metrics.json"
)

MODEL_NAME = "yolov8n.yaml"
CLASS_NAMES = (
    "car",
    "truck",
    "bus",
    "van",
    "freight_car",
)
NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 2
WORKERS = 0
SCORE_THRESHOLD = 0.001
IOU_THRESHOLD = 0.7
MAX_DET = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _as_pair_ids(batch: dict[str, object]) -> tuple[str, ...]:
    value = batch.get("pair_ids")

    if value is None:
        value = batch.get("pair_id")

    if value is None:
        raise RuntimeError(
            "paired batch does not contain pair_ids or pair_id"
        )

    if isinstance(value, str):
        return (value,)

    return tuple(str(item) for item in value)


def _decoded_output(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        decoded = output
    elif isinstance(output, (tuple, list)) and output:
        decoded = output[0]
    else:
        raise RuntimeError(
            "early-fusion model returned an unsupported output"
        )

    if not isinstance(decoded, torch.Tensor):
        raise RuntimeError(
            "decoded model output must be a tensor"
        )

    if decoded.ndim != 3:
        raise RuntimeError(
            "decoded model output must have shape [B, C, N]"
        )

    return decoded


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            rgb = batch["rgb"]
            ir = batch["ir"]

            if not isinstance(rgb, torch.Tensor):
                raise RuntimeError(
                    f"batch {batch_index} rgb must be a tensor"
                )

            if not isinstance(ir, torch.Tensor):
                raise RuntimeError(
                    f"batch {batch_index} ir must be a tensor"
                )

            if rgb.ndim != 4 or rgb.shape[1] != 3:
                raise RuntimeError(
                    f"batch {batch_index} rgb must have shape "
                    "[B, 3, H, W]"
                )

            if ir.ndim != 4 or ir.shape[1] != 1:
                raise RuntimeError(
                    f"batch {batch_index} ir must have shape "
                    "[B, 1, H, W]"
                )

            if rgb.shape[0] != ir.shape[0]:
                raise RuntimeError(
                    f"batch {batch_index} RGB/IR batch sizes differ"
                )

            if rgb.shape[-2:] != ir.shape[-2:]:
                raise RuntimeError(
                    f"batch {batch_index} RGB/IR image sizes differ"
                )

            pair_ids = _as_pair_ids(batch)

            if len(pair_ids) != rgb.shape[0]:
                raise RuntimeError(
                    f"batch {batch_index} pair ID count does not "
                    "match the batch size"
                )

            fused = torch.cat(
                (
                    rgb.to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    ),
                    ir.to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    ),
                ),
                dim=1,
            )

            output = model(fused)
            decoded = _decoded_output(output)

            batch_results = (
                ultralytics_decoded_to_paired_predictions(
                    decoded,
                    pair_ids=pair_ids,
                    score_threshold=SCORE_THRESHOLD,
                    iou_threshold=IOU_THRESHOLD,
                    max_det=MAX_DET,
                )
            )

            results.extend(batch_results)

    return results


def build_coco_ground_truth(
    dataset: PairedYoloDataset,
) -> tuple[dict[str, object], dict[str, int]]:
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    pair_to_image_id: dict[str, int] = {}

    annotation_id = 1

    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]

        pair_id = str(sample["pair_id"])
        rgb = sample["rgb"]
        boxes = sample["boxes"]
        labels = sample["labels"]

        if not isinstance(rgb, torch.Tensor):
            raise RuntimeError(
                f"sample {dataset_index} rgb must be a tensor"
            )

        if not isinstance(boxes, torch.Tensor):
            raise RuntimeError(
                f"sample {dataset_index} boxes must be a tensor"
            )

        if not isinstance(labels, torch.Tensor):
            raise RuntimeError(
                f"sample {dataset_index} labels must be a tensor"
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
            boxes.detach().cpu(),
            labels.detach().cpu(),
            strict=True,
        ):
            center_x = float(box[0]) * width
            center_y = float(box[1]) * height
            box_width = float(box[2]) * width
            box_height = float(box[3]) * height

            x_min = center_x - box_width / 2.0
            y_min = center_y - box_height / 2.0

            x_min = max(0.0, min(x_min, float(width)))
            y_min = max(0.0, min(y_min, float(height)))
            box_width = max(
                0.0,
                min(box_width, float(width) - x_min),
            )
            box_height = max(
                0.0,
                min(box_height, float(height) - y_min),
            )

            if box_width <= 0.0 or box_height <= 0.0:
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [
                        x_min,
                        y_min,
                        box_width,
                        box_height,
                    ],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    categories = [
        {
            "id": index + 1,
            "name": name,
        }
        for index, name in enumerate(CLASS_NAMES)
    ]

    ground_truth: dict[str, object] = {
        "info": {
            "description": (
                "DroneVehicle smoke 5% paired validation set"
            ),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    return ground_truth, pair_to_image_id


def build_coco_predictions(
    results: list[dict[str, object]],
    *,
    pair_to_image_id: dict[str, int],
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []

    for result_index, result in enumerate(results):
        pair_id = str(result["pair_id"])

        if pair_id not in pair_to_image_id:
            raise RuntimeError(
                f"prediction {result_index} has unknown pair ID: "
                f"{pair_id}"
            )

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        if not isinstance(boxes, list):
            raise RuntimeError(
                f"prediction {result_index} boxes must be a list"
            )

        if not isinstance(scores, list):
            raise RuntimeError(
                f"prediction {result_index} scores must be a list"
            )

        if not isinstance(labels, list):
            raise RuntimeError(
                f"prediction {result_index} labels must be a list"
            )

        if not (
            len(boxes) == len(scores) == len(labels)
        ):
            raise RuntimeError(
                f"prediction {result_index} field lengths differ"
            )

        image_id = pair_to_image_id[pair_id]

        for box, score, label in zip(
            boxes,
            scores,
            labels,
            strict=True,
        ):
            if len(box) != 4:
                raise RuntimeError(
                    f"prediction {result_index} box must contain "
                    "four coordinates"
                )

            x_min = float(box[0])
            y_min = float(box[1])
            x_max = float(box[2])
            y_max = float(box[3])

            width = x_max - x_min
            height = y_max - y_min

            if width <= 0.0 or height <= 0.0:
                continue

            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [
                        x_min,
                        y_min,
                        width,
                        height,
                    ],
                    "score": float(score),
                }
            )

    return predictions


def _mean_valid(values: torch.Tensor) -> float | None:
    valid = values[values > -1.0]

    if valid.numel() == 0:
        return None

    return float(valid.mean().item())


def calculate_per_class_metrics(
    evaluator: COCOeval,
) -> dict[str, dict[str, float | None]]:
    precision = evaluator.eval["precision"]

    per_class: dict[str, dict[str, float | None]] = {}

    for class_index, class_name in enumerate(CLASS_NAMES):
        # precision dimensions:
        # [iou_threshold, recall, category, area, max_dets]
        #
        # area index 0 = all
        # area index 1 = small
        # max_dets index 2 = 100
        ap50_95 = _mean_valid(
            torch.as_tensor(
                precision[:, :, class_index, 0, 2],
                dtype=torch.float64,
            )
        )

        ap50 = _mean_valid(
            torch.as_tensor(
                precision[0, :, class_index, 0, 2],
                dtype=torch.float64,
            )
        )

        ap75 = _mean_valid(
            torch.as_tensor(
                precision[5, :, class_index, 0, 2],
                dtype=torch.float64,
            )
        )

        ap_small = _mean_valid(
            torch.as_tensor(
                precision[:, :, class_index, 1, 2],
                dtype=torch.float64,
            )
        )

        per_class[class_name] = {
            "AP50-95": ap50_95,
            "AP50": ap50,
            "AP75": ap75,
            "AP_small": ap_small,
        }

    return per_class


def evaluate_predictions(
    ground_truth: dict[str, object],
    predictions: list[dict[str, object]],
) -> tuple[dict[str, float], dict[str, dict[str, float | None]]]:
    coco_ground_truth = COCO()
    coco_ground_truth.dataset = ground_truth
    coco_ground_truth.createIndex()

    if predictions:
        coco_predictions = coco_ground_truth.loadRes(predictions)
    else:
        coco_predictions = COCO()
        coco_predictions.dataset = {
            "images": ground_truth["images"],
            "categories": ground_truth["categories"],
            "annotations": [],
        }
        coco_predictions.createIndex()

    evaluator = COCOeval(
        coco_ground_truth,
        coco_predictions,
        "bbox",
    )
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "mAP50-95": float(evaluator.stats[0]),
        "mAP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "AR_1": float(evaluator.stats[6]),
        "AR_10": float(evaluator.stats[7]),
        "AR_100": float(evaluator.stats[8]),
        "AR_small": float(evaluator.stats[9]),
        "AR_medium": float(evaluator.stats[10]),
        "AR_large": float(evaluator.stats[11]),
    }

    per_class = calculate_per_class_metrics(evaluator)

    return metrics, per_class


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"checkpoint does not exist: {CHECKPOINT_PATH}"
        )

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
        pin_memory=device.type == "cuda",
        collate_fn=collate_paired_batch,
    )

    model = build_early_fusion_model(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        device=device,
    )

    payload: Any = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            "checkpoint payload must be a dictionary"
        )

    model_state = payload.get("model")

    if not isinstance(model_state, dict):
        raise RuntimeError(
            "checkpoint does not contain a model state dictionary"
        )

    model.load_state_dict(model_state)

    restored_epoch = int(payload.get("epoch", 0))

    results = run_inference(
        model,
        loader,
        device=device,
    )

    ground_truth, pair_to_image_id = (
        build_coco_ground_truth(dataset)
    )
    coco_predictions = build_coco_predictions(
        results,
        pair_to_image_id=pair_to_image_id,
    )

    metrics, per_class = evaluate_predictions(
        ground_truth,
        coco_predictions,
    )

    label_counts = Counter(
        int(prediction["category_id"])
        for prediction in coco_predictions
    )

    report: dict[str, object] = {
        "format_version": 1,
        "status": "completed",
        "experiment_id": (
            "early-fusion-teacher-v2-10ep-seed-3407"
        ),
        "experiment_group": (
            "early-fusion-teacher-preflight-v2"
        ),
        "device": str(device),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "restored_epoch": restored_epoch,
        "dataset": {
            "name": "DroneVehicle smoke 5%",
            "split": "val",
            "images": len(dataset),
            "annotations": len(
                ground_truth["annotations"]  # type: ignore[arg-type]
            ),
        },
        "evaluation": {
            "score_threshold": SCORE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "pre_coco_max_det": MAX_DET,
            "coco_max_det": 100,
        },
        "prediction_images": len(results),
        "prediction_boxes": len(coco_predictions),
        "prediction_category_counts": {
            CLASS_NAMES[category_id - 1]: count
            for category_id, count in sorted(
                label_counts.items()
            )
        },
        **metrics,
        "per_class": per_class,
    }

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

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"metrics_path= {OUTPUT_PATH}")
    print("early_fusion_teacher_v2_coco_evaluation_ok=True")


if __name__ == "__main__":
    main()