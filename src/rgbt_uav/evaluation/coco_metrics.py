"""Official COCO bounding-box metrics for small-object evaluation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from numbers import Real
from typing import Any

import numpy as np


class CocoMetricError(ValueError):
    """Raised when COCO evaluation inputs or outputs are invalid."""


@dataclass(frozen=True)
class CocoSmallMetrics:
    ap: float
    ap50: float
    ap75: float
    ap_small: float
    ar_small: float
    per_class_ap_small: dict[str, float | None]


def evaluate_coco_bbox(
    ground_truth: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> CocoSmallMetrics:
    """Evaluate COCO bounding-box predictions with official small-object settings."""
    _validate_inputs(ground_truth, predictions)
    coco_type, evaluator_type = _load_pycocotools()
    coco_gt = coco_type()
    coco_gt.dataset = copy.deepcopy(dict(ground_truth))
    coco_gt.createIndex()

    if predictions:
        coco_dt = coco_gt.loadRes(copy.deepcopy(list(predictions)))
    else:
        coco_dt = coco_type()
        coco_dt.dataset = {
            "images": copy.deepcopy(coco_gt.dataset["images"]),
            "categories": copy.deepcopy(coco_gt.dataset["categories"]),
            "annotations": [],
        }
        coco_dt.createIndex()

    evaluator = evaluator_type(coco_gt, coco_dt, "bbox")
    evaluator.params.iouThrs = np.linspace(0.5, 0.95, 10, endpoint=True)
    evaluator.params.areaRng = [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"]
    area_index = evaluator.params.areaRngLbl.index("small")
    det_index = evaluator.params.maxDets.index(100)
    per_class: dict[str, float | None] = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, category_index, area_index, det_index]
        valid = values[values > -1]
        category = coco_gt.cats[category_id]
        per_class[category["name"]] = (
            _required_metric(valid.mean(), category["name"]) if valid.size else None
        )

    ap, ap50, ap75, ap_small, ar_small = (
        _required_metric(evaluator.stats[index], f"stats[{index}]")
        for index in (0, 1, 2, 3, 9)
    )
    return CocoSmallMetrics(
        ap=ap,
        ap50=ap50,
        ap75=ap75,
        ap_small=ap_small,
        ar_small=ar_small,
        per_class_ap_small=per_class,
    )


def _load_pycocotools() -> tuple[type[Any], type[Any]]:
    required = "2.0.11"
    try:
        installed = metadata.version("pycocotools")
    except Exception as exc:
        raise CocoMetricError(
            f"pycocotools=={required} is required for COCO metric evaluation"
        ) from exc
    if installed != required:
        raise CocoMetricError(
            f"pycocotools=={required} is required for COCO metric evaluation; "
            f"found {installed}"
        )
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        raise CocoMetricError(
            f"pycocotools=={required} could not be imported for COCO metric evaluation"
        ) from exc
    return COCO, COCOeval


def _validate_inputs(
    ground_truth: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> None:
    if not isinstance(ground_truth, Mapping):
        raise CocoMetricError("ground_truth must be a mapping")
    _require_keys(ground_truth, ("images", "annotations", "categories"), "ground_truth")
    images = _require_list(ground_truth["images"], "ground_truth.images")
    annotations = _require_list(ground_truth["annotations"], "ground_truth.annotations")
    categories = _require_list(ground_truth["categories"], "ground_truth.categories")
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes, bytearray)):
        raise CocoMetricError("predictions must be a sequence of mappings")

    image_ids = _validate_images(images)
    category_ids = _validate_categories(categories)
    _validate_annotations(annotations, image_ids, category_ids)
    _validate_predictions(predictions, image_ids, category_ids)


def _validate_images(images: list[Any]) -> set[Real]:
    ids: set[Real] = set()
    for index, image in enumerate(images):
        location = f"ground_truth.images[{index}]"
        _require_mapping(image, location)
        _require_keys(image, ("id", "width", "height"), location)
        image_id = _exact_integer(image["id"], f"{location}.id")
        _unique_id(ids, image_id, f"{location}.id")
        _positive_number(image["width"], f"{location}.width")
        _positive_number(image["height"], f"{location}.height")
    return ids


def _validate_categories(categories: list[Any]) -> set[Real]:
    ids: set[Real] = set()
    for index, category in enumerate(categories):
        location = f"ground_truth.categories[{index}]"
        _require_mapping(category, location)
        _require_keys(category, ("id", "name"), location)
        category_id = _exact_integer(category["id"], f"{location}.id")
        _unique_id(ids, category_id, f"{location}.id")
        if not isinstance(category["name"], str):
            raise CocoMetricError(f"{location}.name must be a string")
    return ids


def _validate_annotations(
    annotations: list[Any], image_ids: set[Real], category_ids: set[Real]
) -> None:
    annotation_ids: set[Real] = set()
    for index, annotation in enumerate(annotations):
        location = f"ground_truth.annotations[{index}]"
        _require_mapping(annotation, location)
        _require_keys(
            annotation,
            ("id", "image_id", "category_id", "bbox", "area", "iscrowd"),
            location,
        )
        annotation_id = _exact_integer(annotation["id"], f"{location}.id")
        _unique_id(annotation_ids, annotation_id, f"{location}.id")
        _reference_id(annotation["image_id"], image_ids, f"{location}.image_id")
        _reference_id(annotation["category_id"], category_ids, f"{location}.category_id")
        _validate_bbox(annotation["bbox"], f"{location}.bbox")
        _positive_number(annotation["area"], f"{location}.area")
        crowd = _exact_integer(annotation["iscrowd"], f"{location}.iscrowd")
        if crowd not in (0, 1):
            raise CocoMetricError(f"{location}.iscrowd must be 0 or 1")


def _validate_predictions(
    predictions: Sequence[Mapping[str, Any]], image_ids: set[Real], category_ids: set[Real]
) -> None:
    for index, prediction in enumerate(predictions):
        location = f"predictions[{index}]"
        _require_mapping(prediction, location)
        _require_keys(prediction, ("image_id", "category_id", "bbox", "score"), location)
        _reference_id(prediction["image_id"], image_ids, f"{location}.image_id")
        _reference_id(prediction["category_id"], category_ids, f"{location}.category_id")
        _validate_bbox(prediction["bbox"], f"{location}.bbox")
        _finite_number(prediction["score"], f"{location}.score")


def _require_mapping(value: Any, location: str) -> None:
    if not isinstance(value, Mapping):
        raise CocoMetricError(f"{location} must be a mapping")


def _require_keys(mapping: Mapping[str, Any], keys: Sequence[str], location: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise CocoMetricError(f"{location} is missing required keys: {', '.join(missing)}")


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise CocoMetricError(f"{location} must be a list")
    return value


def _unique_id(ids: set[Real], value: Real, location: str) -> None:
    if value in ids:
        raise CocoMetricError(f"duplicate ID at {location}")
    ids.add(value)


def _reference_id(value: Any, valid_ids: set[Real], location: str) -> None:
    identifier = _exact_integer(value, location)
    if identifier not in valid_ids:
        raise CocoMetricError(f"{location} does not reference a declared ID")


def _exact_integer(value: Any, location: str) -> Real:
    number = _finite_number(value, location)
    if not float(number).is_integer():
        raise CocoMetricError(f"{location} must be an integer")
    return number


def _positive_number(value: Any, location: str) -> Real:
    number = _finite_number(value, location)
    if number <= 0:
        raise CocoMetricError(f"{location} must be positive")
    return number


def _finite_number(value: Any, location: str) -> Real:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise CocoMetricError(f"{location} must be a finite number")
    return value


def _validate_bbox(value: Any, location: str) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
    ):
        raise CocoMetricError(f"{location} must contain four numbers")
    for index, coordinate in enumerate(value):
        _finite_number(coordinate, f"{location}[{index}]")
    _positive_number(value[2], f"{location}[2]")
    _positive_number(value[3], f"{location}[3]")


def _required_metric(value: Any, location: str) -> float:
    metric = float(_finite_number(value, location))
    if metric < 0.0:
        raise CocoMetricError(f"{location} is an undefined required COCO aggregate")
    return metric
