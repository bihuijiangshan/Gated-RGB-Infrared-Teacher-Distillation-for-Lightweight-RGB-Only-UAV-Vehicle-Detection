from __future__ import annotations

from collections.abc import Mapping, Sequence


class AnnotationError(ValueError):
    """Raised when an annotation cannot be converted without ambiguity."""


def obb_to_hbb(
    points: Sequence[float],
    *,
    crop_left: float,
    crop_top: float,
    width: float,
    height: float,
) -> list[float]:
    """Convert four OBB vertices to a clipped HBB after removing a crop offset."""
    if len(points) != 8:
        raise AnnotationError(f"expected 8 OBB coordinates, got {len(points)}")
    if width <= 0 or height <= 0:
        raise AnnotationError("output dimensions must be positive")

    xs = [float(value) - crop_left for value in points[0::2]]
    ys = [float(value) - crop_top for value in points[1::2]]
    x1 = min(max(min(xs), 0.0), float(width))
    y1 = min(max(min(ys), 0.0), float(height))
    x2 = min(max(max(xs), 0.0), float(width))
    y2 = min(max(max(ys), 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise AnnotationError("OBB becomes a degenerate HBB after crop and clipping")
    return [x1, y1, x2, y2]


def parse_obb_line(
    line: str,
    *,
    class_to_id: Mapping[str, int],
    crop_left: float,
    crop_top: float,
    width: float,
    height: float,
) -> tuple[list[float], int]:
    """Parse a DOTA-style `x1 y1 ... x4 y4 class [difficulty]` annotation line."""
    tokens = line.split()
    if len(tokens) < 9:
        raise AnnotationError("annotation line must contain 8 coordinates and a class")
    try:
        points = [float(token) for token in tokens[:8]]
    except ValueError as exc:
        raise AnnotationError("annotation coordinates must be numeric") from exc
    class_name = tokens[8]
    if class_name not in class_to_id:
        raise AnnotationError(f"unknown class: {class_name}")
    box = obb_to_hbb(
        points,
        crop_left=crop_left,
        crop_top=crop_top,
        width=width,
        height=height,
    )
    return box, int(class_to_id[class_name])

