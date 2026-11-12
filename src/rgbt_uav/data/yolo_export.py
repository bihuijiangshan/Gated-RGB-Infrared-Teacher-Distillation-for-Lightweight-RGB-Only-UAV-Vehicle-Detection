"""Export cropped RGB or IR records with YOLO-format labels."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

from PIL import Image

from rgbt_uav.data.manifest import SampleRecord

_SOURCE_SIZE = (840, 712)
_CROP_BOX = (100, 100, 740, 612)
_CROP_SIZE = (640, 512)


class YoloExportError(ValueError):
    """Raised when a record cannot be represented as a cropped YOLO sample."""


class YoloExportRollbackError(YoloExportError):
    """Raised when publishing fails and at least one rollback action also fails."""

    def __init__(
        self,
        *,
        publication_error: Exception,
        rollback_errors: list[tuple[Path, Exception]],
        preserved_backups: list[Path],
    ) -> None:
        self.publication_error = publication_error
        self.rollback_errors = rollback_errors
        self.preserved_backups = preserved_backups
        super().__init__("YOLO export publication and rollback both failed")


def _validated_dimension(value: object, *, name: str) -> float:
    invalid = (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    )
    if invalid:
        raise YoloExportError(f"{name} must be a positive finite number")
    return float(value)


def _validated_coordinate(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise YoloExportError(f"{name} must be a finite number")
    return float(value)


def hbb_to_yolo(
    box: Sequence[float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Normalize a validated HBB to YOLO center-x, center-y, width, height."""
    normalized_width = _validated_dimension(width, name="width")
    normalized_height = _validated_dimension(height, name="height")
    if len(box) != 4:
        raise YoloExportError("HBB must contain exactly four values")
    x1, y1, x2, y2 = (
        _validated_coordinate(value, name=f"box[{index}]") for index, value in enumerate(box)
    )
    if not (0 <= x1 < x2 <= normalized_width and 0 <= y1 < y2 <= normalized_height):
        raise YoloExportError(f"invalid HBB: {box}")
    return (
        (x1 + x2) / (2 * normalized_width),
        (y1 + y2) / (2 * normalized_height),
        (x2 - x1) / normalized_width,
        (y2 - y1) / normalized_height,
    )


def _validated_class_id(label: object) -> int:
    if isinstance(label, bool) or not isinstance(label, Integral) or label < 0:
        raise YoloExportError(f"invalid class ID: {label!r}")
    return int(label)


def format_yolo_labels(
    boxes: Sequence[Sequence[float]], labels: Sequence[int], *, width: int, height: int
) -> str:
    """Return one six-decimal YOLO label line per HBB and class ID."""
    if len(boxes) != len(labels):
        raise YoloExportError("box and label counts must match")
    lines: list[str] = []
    for box, label in zip(boxes, labels, strict=True):
        class_id = _validated_class_id(label)
        x_center, y_center, box_width, box_height = hbb_to_yolo(box, width=width, height=height)
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _temporary_sibling(destination: Path, *, purpose: str) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.{purpose}.",
        suffix=".tmp",
    )
    temporary.close()
    return Path(temporary.name)


def _normalized_resolved_path(path: Path) -> str:
    try:
        return os.path.normcase(str(path.resolve(strict=False)))
    except OSError as exc:
        raise YoloExportError(f"could not resolve path: {path}") from exc


def _validate_destinations(
    *, image_destination: Path, label_destination: Path, record: SampleRecord
) -> None:
    image_key = _normalized_resolved_path(image_destination)
    label_key = _normalized_resolved_path(label_destination)
    if image_key == label_key:
        raise YoloExportError("image and label destinations must differ")
    source_keys = {
        _normalized_resolved_path(Path(record.rgb_path)),
        _normalized_resolved_path(Path(record.ir_path)),
    }
    if image_key in source_keys or label_key in source_keys:
        raise YoloExportError("output destination must not overwrite a source image")


def _rollback_destination(
    destination: Path,
    *,
    existed: bool,
    backed_up: bool,
    published: bool,
    backup: Path | None,
) -> Exception | None:
    try:
        if backed_up:
            if destination.exists():
                destination.unlink()
            if backup is not None and backup.exists():
                backup.replace(destination)
        elif not existed and published and destination.exists():
            destination.unlink()
    except Exception as exc:
        return exc
    return None


def _preserved_backups(backups: Sequence[tuple[Path | None, bool]]) -> list[Path]:
    return [
        backup
        for backup, backed_up in backups
        if backed_up and backup is not None and backup.exists()
    ]


def _cleanup_siblings(paths: Sequence[Path | None], *, preserve: Sequence[Path] = ()) -> None:
    preserved = set(preserve)
    for path in paths:
        if path is None or path in preserved:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def export_yolo_record(
    record: SampleRecord,
    *,
    modality: Literal["rgb", "ir"],
    image_path: Path,
    label_path: Path,
) -> None:
    """Crop one record and publish its image and labels with rollback on ordinary exceptions."""
    if modality == "rgb":
        source_path = Path(record.rgb_path)
        output_mode = "RGB"
        output_format = "JPEG"
    elif modality == "ir":
        source_path = Path(record.ir_path)
        output_mode = "L"
        output_format = "PNG"
    else:
        raise YoloExportError(f"unsupported modality: {modality!r}")

    image_destination = Path(image_path)
    label_destination = Path(label_path)
    _validate_destinations(
        image_destination=image_destination,
        label_destination=label_destination,
        record=record,
    )
    labels = format_yolo_labels(
        record.boxes,
        record.labels,
        width=_CROP_SIZE[0],
        height=_CROP_SIZE[1],
    )
    try:
        with Image.open(source_path) as source_image:
            if source_image.size != _SOURCE_SIZE:
                raise YoloExportError(
                    f"source image must be {_SOURCE_SIZE}, got {source_image.size}: {source_path}"
                )
            cropped_image = source_image.convert(output_mode).crop(_CROP_BOX)
    except YoloExportError:
        raise
    except OSError as exc:
        raise YoloExportError(f"could not read source image: {source_path}") from exc

    image_destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    image_temporary: Path | None = None
    label_temporary: Path | None = None
    image_backup: Path | None = None
    label_backup: Path | None = None
    image_existed = image_destination.exists()
    label_existed = label_destination.exists()
    image_backed_up = False
    label_backed_up = False
    image_published = False
    label_published = False
    backups_to_preserve: list[Path] = []
    try:
        image_temporary = _temporary_sibling(image_destination, purpose="tmp")
        label_temporary = _temporary_sibling(label_destination, purpose="tmp")
        cropped_image.save(image_temporary, format=output_format)
        label_temporary.write_text(labels, encoding="utf-8", newline="\n")
        if image_existed:
            image_backup = _temporary_sibling(image_destination, purpose="backup")
            image_destination.replace(image_backup)
            image_backed_up = True
        if label_existed:
            label_backup = _temporary_sibling(label_destination, purpose="backup")
            label_destination.replace(label_backup)
            label_backed_up = True
        image_temporary.replace(image_destination)
        image_published = True
        label_temporary.replace(label_destination)
        label_published = True
    except Exception as publication_error:
        rollback_errors: list[tuple[Path, Exception]] = []
        image_rollback_error = _rollback_destination(
            image_destination,
            existed=image_existed,
            backed_up=image_backed_up,
            published=image_published,
            backup=image_backup,
        )
        if image_rollback_error is not None:
            rollback_errors.append((image_destination, image_rollback_error))
        label_rollback_error = _rollback_destination(
            label_destination,
            existed=label_existed,
            backed_up=label_backed_up,
            published=label_published,
            backup=label_backup,
        )
        if label_rollback_error is not None:
            rollback_errors.append((label_destination, label_rollback_error))
        preserved_backups = _preserved_backups(
            ((image_backup, image_backed_up), (label_backup, label_backed_up))
        )
        if rollback_errors:
            backups_to_preserve = preserved_backups
            raise YoloExportRollbackError(
                publication_error=publication_error,
                rollback_errors=rollback_errors,
                preserved_backups=preserved_backups,
            ) from publication_error
        raise
    finally:
        _cleanup_siblings((image_temporary, label_temporary))
        _cleanup_siblings(
            (image_backup, label_backup),
            preserve=backups_to_preserve,
        )
