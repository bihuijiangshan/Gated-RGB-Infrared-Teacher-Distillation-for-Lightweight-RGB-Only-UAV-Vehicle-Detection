from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn

from rgbt_uav.evaluation.coco_metrics import (
    CocoSmallMetrics,
    evaluate_coco_bbox,
)


class PairedEvaluationError(ValueError):
    """Raised when paired inference inputs or outputs are invalid."""


def _tensor(
    value: object,
    *,
    field: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise PairedEvaluationError(
            f"{field} must be a tensor"
        )

    if value.ndim != ndim:
        raise PairedEvaluationError(
            f"{field} must have {ndim} dimensions"
        )

    return value


def _validate_batch(batch: Mapping[str, object]) -> tuple[Tensor, Tensor, tuple[str, ...]]:
    rgb = _tensor(batch.get("rgb"), field="rgb", ndim=4)
    ir = _tensor(batch.get("ir"), field="ir", ndim=4)
    pair_ids = batch.get("pair_ids")

    if rgb.shape[1] != 3:
        raise PairedEvaluationError(
            "rgb must have shape [batch, 3, height, width]"
        )

    if ir.shape[1] != 1:
        raise PairedEvaluationError(
            "ir must have shape [batch, 1, height, width]"
        )

    if rgb.shape[0] != ir.shape[0] or rgb.shape[2:] != ir.shape[2:]:
        raise PairedEvaluationError(
            "rgb and ir batch/spatial shapes must match"
        )

    if not isinstance(pair_ids, tuple):
        raise PairedEvaluationError(
            "pair_ids must be a tuple"
        )

    if len(pair_ids) != rgb.shape[0]:
        raise PairedEvaluationError(
            "pair_ids length must match batch size"
        )

    if any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
        raise PairedEvaluationError(
            "pair_ids must contain non-empty strings"
        )

    return rgb, ir, pair_ids


def _validate_prediction(
    prediction: object,
    *,
    index: int,
    score_threshold: float,
) -> dict[str, list[float | int]]:
    if not isinstance(prediction, Mapping):
        raise PairedEvaluationError(
            f"prediction[{index}] must be a mapping"
        )

    boxes = _tensor(
        prediction.get("boxes"),
        field=f"prediction[{index}].boxes",
        ndim=2,
    )
    scores = _tensor(
        prediction.get("scores"),
        field=f"prediction[{index}].scores",
        ndim=1,
    )
    labels = _tensor(
        prediction.get("labels"),
        field=f"prediction[{index}].labels",
        ndim=1,
    )

    if boxes.shape[1] != 4:
        raise PairedEvaluationError(
            f"prediction[{index}].boxes must have shape [N, 4]"
        )

    if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != labels.shape[0]:
        raise PairedEvaluationError(
            f"prediction[{index}] detection fields must have equal length"
        )

    if not torch.isfinite(boxes).all():
        raise PairedEvaluationError(
            f"prediction[{index}].boxes must be finite"
        )

    if not torch.isfinite(scores).all():
        raise PairedEvaluationError(
            f"prediction[{index}].scores must be finite"
        )

    if not torch.isfinite(labels.to(dtype=torch.float32)).all():
        raise PairedEvaluationError(
            f"prediction[{index}].labels must be finite"
        )

    keep = scores >= score_threshold
    kept_boxes = boxes[keep].detach().cpu().tolist()
    kept_scores = scores[keep].detach().cpu().tolist()
    kept_labels = labels[keep].detach().cpu().tolist()

    return {
        "boxes": [[round(float(coordinate), 7) for coordinate in box] for box in kept_boxes],
        "scores": [round(float(score), 7) for score in kept_scores],
        "labels": [int(label) for label in kept_labels],
    }


def run_paired_inference(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device | str,
    score_threshold: float = 0.25,
) -> list[dict[str, object]]:
    """Run inference on paired RGB-IR batches and preserve pair IDs."""

    if not isinstance(model, nn.Module):
        raise PairedEvaluationError(
            "model must be a torch.nn.Module"
        )

    if (
        isinstance(score_threshold, bool)
        or not isinstance(score_threshold, (int, float))
        or not 0.0 <= float(score_threshold) <= 1.0
    ):
        raise PairedEvaluationError(
            "score_threshold must be between 0 and 1"
        )

    target_device = torch.device(device)
    was_training = model.training
    results: list[dict[str, object]] = []

    model.to(target_device)
    model.eval()

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if not isinstance(batch, Mapping):
                    raise PairedEvaluationError(
                        f"batch[{batch_index}] must be a mapping"
                    )

                rgb, ir, pair_ids = _validate_batch(batch)

                try:
                    predictions = model(
                        rgb.to(target_device),
                        ir.to(target_device),
                    )
                except (TypeError, RuntimeError, ValueError) as exc:
                    raise PairedEvaluationError(
                        f"paired model inference failed at batch {batch_index}"
                    ) from exc

                if not isinstance(predictions, (list, tuple)):
                    raise PairedEvaluationError(
                        "paired model must return one prediction per image"
                    )

                if len(predictions) != len(pair_ids):
                    raise PairedEvaluationError(
                        "prediction count must match batch size"
                    )

                for image_index, (pair_id, prediction) in enumerate(
                    zip(pair_ids, predictions, strict=True)
                ):
                    normalized = _validate_prediction(
                        prediction,
                        index=image_index,
                        score_threshold=float(score_threshold),
                    )
                    results.append(
                        {
                            "pair_id": pair_id,
                            **normalized,
                        }
                    )
    finally:
        model.train(was_training)

    return results

def paired_results_to_coco_predictions(
    results: list[dict[str, object]],
    *,
    pair_to_image_id: Mapping[str, int],
    label_to_category_id: Mapping[int, int],
) -> list[dict[str, object]]:
    """Convert paired xyxy detections to COCO xywh predictions."""

    if not isinstance(pair_to_image_id, Mapping):
        raise PairedEvaluationError(
            "pair_to_image_id must be a mapping"
        )

    if not isinstance(label_to_category_id, Mapping):
        raise PairedEvaluationError(
            "label_to_category_id must be a mapping"
        )

    predictions: list[dict[str, object]] = []

    for result_index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise PairedEvaluationError(
                f"results[{result_index}] must be a mapping"
            )

        pair_id = result.get("pair_id")

        if not isinstance(pair_id, str) or not pair_id:
            raise PairedEvaluationError(
                f"results[{result_index}].pair_id must be non-empty"
            )

        if pair_id not in pair_to_image_id:
            raise PairedEvaluationError(
                f"unknown pair_id: {pair_id}"
            )

        image_id = pair_to_image_id[pair_id]

        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id <= 0
        ):
            raise PairedEvaluationError(
                f"image ID for {pair_id!r} must be positive"
            )

        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")

        if not isinstance(boxes, list):
            raise PairedEvaluationError(
                f"results[{result_index}].boxes must be a list"
            )

        if not isinstance(scores, list) or not isinstance(labels, list):
            raise PairedEvaluationError(
                f"results[{result_index}] detection fields must be lists"
            )

        if len(boxes) != len(scores) or len(boxes) != len(labels):
            raise PairedEvaluationError(
                f"results[{result_index}] detection fields must have equal length"
            )

        for detection_index, (box, score, label) in enumerate(
            zip(boxes, scores, labels, strict=True)
        ):
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    for value in box
                )
            ):
                raise PairedEvaluationError(
                    f"results[{result_index}].boxes[{detection_index}] is invalid"
                )

            x1, y1, x2, y2 = (float(value) for value in box)
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                raise PairedEvaluationError(
                    f"results[{result_index}].boxes[{detection_index}] "
                    "must have positive width and height"
                )

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise PairedEvaluationError(
                    f"results[{result_index}].scores[{detection_index}] is invalid"
                )

            if (
                isinstance(label, bool)
                or not isinstance(label, int)
                or label not in label_to_category_id
            ):
                raise PairedEvaluationError(
                    f"results[{result_index}].labels[{detection_index}] is invalid"
                )

            category_id = label_to_category_id[label]

            if (
                isinstance(category_id, bool)
                or not isinstance(category_id, int)
                or category_id <= 0
            ):
                raise PairedEvaluationError(
                    f"category ID for label {label} must be positive"
                )

            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [
                        round(x1, 7),
                        round(y1, 7),
                        round(width, 7),
                        round(height, 7),
                    ],
                    "score": round(float(score), 7),
                }
            )

    return predictions

def paired_samples_to_coco_ground_truth(
    samples: Iterable[Mapping[str, object]],
    *,
    label_to_category_id: Mapping[int, int],
    category_names: Mapping[int, str],
) -> dict[str, object]:
    """Convert paired normalized targets into a COCO ground-truth object."""

    if not isinstance(label_to_category_id, Mapping):
        raise PairedEvaluationError(
            "label_to_category_id must be a mapping"
        )

    if not isinstance(category_names, Mapping):
        raise PairedEvaluationError(
            "category_names must be a mapping"
        )

    categories: list[dict[str, object]] = []

    for category_id, name in sorted(
        category_names.items(),
        key=lambda item: item[0],
    ):
        if (
            isinstance(category_id, bool)
            or not isinstance(category_id, int)
            or category_id <= 0
        ):
            raise PairedEvaluationError(
                "category IDs must be positive integers"
            )

        if not isinstance(name, str) or not name:
            raise PairedEvaluationError(
                "category names must be non-empty strings"
            )

        categories.append(
            {
                "id": category_id,
                "name": name,
            }
        )

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    seen_pair_ids: set[str] = set()
    annotation_id = 1

    for image_id, sample in enumerate(samples, start=1):
        if not isinstance(sample, Mapping):
            raise PairedEvaluationError(
                f"samples[{image_id - 1}] must be a mapping"
            )

        pair_id = sample.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise PairedEvaluationError(
                f"samples[{image_id - 1}].pair_id must be non-empty"
            )

        if pair_id in seen_pair_ids:
            raise PairedEvaluationError(
                f"duplicate pair_id: {pair_id}"
            )
        seen_pair_ids.add(pair_id)

        rgb = sample.get("rgb")
        if not isinstance(rgb, Tensor) or rgb.ndim != 3:
            raise PairedEvaluationError(
                f"samples[{image_id - 1}].rgb must be a CHW tensor"
            )

        if rgb.shape[0] != 3:
            raise PairedEvaluationError(
                f"samples[{image_id - 1}].rgb must have 3 channels"
            )

        height = int(rgb.shape[1])
        width = int(rgb.shape[2])

        if height <= 0 or width <= 0:
            raise PairedEvaluationError(
                f"samples[{image_id - 1}].rgb must have positive dimensions"
            )

        images.append(
            {
                "id": image_id,
                "file_name": pair_id,
                "width": width,
                "height": height,
            }
        )

        boxes = sample.get("boxes")
        labels = sample.get("labels")

        if not isinstance(boxes, Tensor) or boxes.ndim != 2:
            raise PairedEvaluationError(
                f"samples[{image_id - 1}].boxes must have shape [N, 4]"
            )

        if boxes.shape[1] != 4:
            raise PairedEvaluationError(
                f"samples[{image_id - 1}].boxes must have shape [N, 4]"
            )

        if not isinstance(labels, Tensor) or labels.ndim != 1:
            raise PairedEvaluationError(
                f"samples[{image_id - 1}].labels must have shape [N]"
            )

        if boxes.shape[0] != labels.shape[0]:
            raise PairedEvaluationError(
                f"samples[{image_id - 1}] box and label counts differ"
            )

        if not torch.isfinite(boxes).all():
            raise PairedEvaluationError(
                f"samples[{image_id - 1}].boxes must be finite"
            )

        for box, label in zip(boxes.tolist(), labels.tolist(), strict=True):
            if (
                isinstance(label, bool)
                or not isinstance(label, int)
                or label not in label_to_category_id
            ):
                raise PairedEvaluationError(
                    f"unknown label at image {image_id}: {label}"
                )

            x_center, y_center, box_width, box_height = (
                float(value) for value in box
            )

            if (
                box_width <= 0
                or box_height <= 0
                or x_center < 0
                or y_center < 0
                or x_center > 1
                or y_center > 1
                or box_width > 1
                or box_height > 1
            ):
                raise PairedEvaluationError(
                    f"invalid normalized box at image {image_id}"
                )

            x = (x_center - box_width / 2.0) * width
            y = (y_center - box_height / 2.0) * height
            pixel_width = box_width * width
            pixel_height = box_height * height

            if pixel_width <= 0 or pixel_height <= 0:
                raise PairedEvaluationError(
                    f"box has non-positive pixel size at image {image_id}"
                )

            category_id = label_to_category_id[label]

            if (
                isinstance(category_id, bool)
                or not isinstance(category_id, int)
                or category_id <= 0
            ):
                raise PairedEvaluationError(
                    f"category ID for label {label} must be positive"
                )

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [
                        round(x, 7),
                        round(y, 7),
                        round(pixel_width, 7),
                        round(pixel_height, 7),
                    ],
                    "area": round(pixel_width * pixel_height, 7),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

def evaluate_paired_coco(
    samples: Iterable[Mapping[str, object]],
    predictions: list[dict[str, object]],
    *,
    label_to_category_id: Mapping[int, int],
    category_names: Mapping[int, str],
) -> CocoSmallMetrics:
    """Evaluate paired detections with the existing official COCO metrics."""

    ground_truth = paired_samples_to_coco_ground_truth(
        samples,
        label_to_category_id=label_to_category_id,
        category_names=category_names,
    )

    pair_to_image_id = {
        str(image["file_name"]): int(image["id"])
        for image in ground_truth["images"]
    }

    coco_predictions = paired_results_to_coco_predictions(
        predictions,
        pair_to_image_id=pair_to_image_id,
        label_to_category_id=label_to_category_id,
    )

    return evaluate_coco_bbox(
        ground_truth,
        coco_predictions,
    )

def ultralytics_decoded_to_paired_predictions(
    decoded: Tensor,
    *,
    pair_ids: tuple[str, ...],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> list[dict[str, object]]:
    """Convert Ultralytics decoded ``xywh + class scores`` to paired detections."""

    if not isinstance(decoded, Tensor) or decoded.ndim != 3:
        raise PairedEvaluationError(
            "decoded must have shape [batch, 4 + classes, candidates]"
        )

    if decoded.shape[1] <= 4:
        raise PairedEvaluationError(
            "decoded must contain four box channels and at least one class"
        )

    if not isinstance(pair_ids, tuple) or len(pair_ids) != decoded.shape[0]:
        raise PairedEvaluationError(
            "pair_ids length must match decoded batch size"
        )

    if any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
        raise PairedEvaluationError(
            "pair_ids must contain non-empty strings"
        )

    if (
        isinstance(score_threshold, bool)
        or not isinstance(score_threshold, (int, float))
        or not 0.0 <= float(score_threshold) <= 1.0
    ):
        raise PairedEvaluationError(
            "score_threshold must be between 0 and 1"
        )

    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not 0.0 <= float(iou_threshold) <= 1.0
    ):
        raise PairedEvaluationError(
            "iou_threshold must be between 0 and 1"
        )

    if (
        isinstance(max_det, bool)
        or not isinstance(max_det, int)
        or max_det <= 0
    ):
        raise PairedEvaluationError(
            "max_det must be a positive integer"
        )

    if not torch.isfinite(decoded).all():
        raise PairedEvaluationError(
            "decoded predictions must be finite"
        )

    try:
        from ultralytics.utils.ops import non_max_suppression
    except Exception as exc:
        raise PairedEvaluationError(
            "Ultralytics non_max_suppression is unavailable"
        ) from exc

    try:
        detections = non_max_suppression(
            decoded,
            conf_thres=float(score_threshold),
            iou_thres=float(iou_threshold),
            max_det=max_det,
            nc=decoded.shape[1] - 4,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PairedEvaluationError(
            "Ultralytics NMS failed"
        ) from exc

    results: list[dict[str, object]] = []

    for pair_id, detection in zip(pair_ids, detections, strict=True):
        if not isinstance(detection, Tensor) or detection.ndim != 2:
            raise PairedEvaluationError(
                "Ultralytics NMS must return [N, 6] tensors"
            )

        if detection.shape[1] != 6:
            raise PairedEvaluationError(
                "Ultralytics NMS output must have six columns"
            )

        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[int] = []

        for row in detection.detach().cpu().tolist():
            x1, y1, x2, y2, score, label = row

            if (
                not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool) for value in row
                )
                or not torch.isfinite(torch.tensor(row, dtype=torch.float32)).all()
            ):
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(
                [
                    round(float(x1), 7),
                    round(float(y1), 7),
                    round(float(x2), 7),
                    round(float(y2), 7),
                ]
            )
            scores.append(round(float(score), 7))
            labels.append(int(label))

        results.append(
            {
                "pair_id": pair_id,
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        )

    return results