from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from rgbt_uav.data.manifest import SampleRecord


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize_records(
    records: Sequence[SampleRecord], *, class_names: Sequence[str]
) -> dict[str, Any]:
    class_counts: Counter[str] = Counter({name: 0 for name in class_names})
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    empty_records = 0
    splits: Counter[str] = Counter()
    for record in records:
        splits[record.split] += 1
        empty_records += not record.boxes
        for box, label in zip(record.boxes, record.labels, strict=True):
            if not 0 <= label < len(class_names):
                raise ValueError(f"label {label} is outside class_names")
            box_width = float(box[2] - box[0])
            box_height = float(box[3] - box[1])
            widths.append(box_width)
            heights.append(box_height)
            areas.append(box_width * box_height)
            class_counts[class_names[label]] += 1
    return {
        "records": len(records),
        "boxes": len(areas),
        "empty_records": empty_records,
        "splits": dict(sorted(splits.items())),
        "class_counts": dict(class_counts),
        "small_boxes_coco": sum(area < 32**2 for area in areas),
        "box_width": _distribution(widths),
        "box_height": _distribution(heights),
        "box_area": _distribution(areas),
    }
