from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rgbt_uav.data.annotations import AnnotationError, parse_obb_line
from rgbt_uav.data.dronevehicle_xml import parse_dronevehicle_xml
from rgbt_uav.data.manifest import SampleRecord, pair_modalities


class BuildError(ValueError):
    """Raised when a split cannot be converted into a complete manifest."""


_DOTA_HEADERS = ("imagesource:", "gsd:")


@dataclass(frozen=True)
class ManifestBuildResult:
    records: list[SampleRecord]
    ignored_names: dict[str, int]
    ignored_geometries: dict[str, int]


def build_dronevehicle_xml_manifest(
    *,
    rgb_dir: Path,
    ir_dir: Path,
    annotation_dir: Path,
    split: str,
    class_to_id: Mapping[str, int],
    crop_left: float,
    crop_top: float,
    width: float,
    height: float,
    source_width: int,
    source_height: int,
    class_aliases: Mapping[str, str] | None = None,
    ignored_names: set[str] | None = None,
    ignore_degenerate: bool = False,
    illumination: str = "unknown",
) -> ManifestBuildResult:
    """Build a globally keyed manifest from the dataset's native XML release."""

    annotation_root = Path(annotation_dir)
    if not annotation_root.is_dir():
        raise BuildError(f"not an annotation directory: {annotation_root}")

    records: list[SampleRecord] = []
    ignored_counter: Counter[str] = Counter()
    ignored_geometry_counter: Counter[str] = Counter()
    for local_id, rgb_path, ir_path in pair_modalities(Path(rgb_dir), Path(ir_dir)):
        annotation_path = annotation_root / f"{local_id}.xml"
        if not annotation_path.is_file():
            raise BuildError(f"missing annotation for {local_id}: {annotation_path}")
        parsed = parse_dronevehicle_xml(
            annotation_path,
            class_to_id=class_to_id,
            class_aliases=class_aliases,
            ignored_names=ignored_names,
            ignore_degenerate=ignore_degenerate,
            crop_left=crop_left,
            crop_top=crop_top,
            width=width,
            height=height,
            source_width=source_width,
            source_height=source_height,
        )
        ignored_counter.update(parsed.ignored_names)
        ignored_geometry_counter.update(parsed.ignored_geometries)
        records.append(
            SampleRecord(
                pair_id=f"{split}/{local_id}",
                rgb_path=str(rgb_path),
                ir_path=str(ir_path),
                boxes=parsed.boxes,
                labels=parsed.labels,
                split=split,
                illumination=illumination,
            )
        )
    return ManifestBuildResult(
        records=records,
        ignored_names=dict(ignored_counter),
        ignored_geometries=dict(ignored_geometry_counter),
    )


def build_split_manifest(
    *,
    rgb_dir: Path,
    ir_dir: Path,
    annotation_dir: Path,
    split: str,
    class_to_id: Mapping[str, int],
    crop_left: float,
    crop_top: float,
    width: float,
    height: float,
    illumination: str = "unknown",
) -> list[SampleRecord]:
    """Build records using one explicitly selected canonical annotation coordinate frame."""
    annotation_root = Path(annotation_dir)
    if not annotation_root.is_dir():
        raise BuildError(f"not an annotation directory: {annotation_root}")

    records: list[SampleRecord] = []
    for pair_id, rgb_path, ir_path in pair_modalities(Path(rgb_dir), Path(ir_dir)):
        annotation_path = annotation_root / f"{pair_id}.txt"
        if not annotation_path.is_file():
            raise BuildError(f"missing annotation for {pair_id}: {annotation_path}")
        boxes: list[list[float]] = []
        labels: list[int] = []
        for line_number, raw_line in enumerate(
            annotation_path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            line = raw_line.strip()
            if not line or line.lower().startswith(_DOTA_HEADERS):
                continue
            try:
                box, label = parse_obb_line(
                    line,
                    class_to_id=class_to_id,
                    crop_left=crop_left,
                    crop_top=crop_top,
                    width=width,
                    height=height,
                )
            except AnnotationError as exc:
                raise BuildError(
                    f"{annotation_path}:{line_number}: {exc}"
                ) from exc
            boxes.append(box)
            labels.append(label)
        records.append(
            SampleRecord(
                pair_id=pair_id,
                rgb_path=str(rgb_path),
                ir_path=str(ir_path),
                boxes=boxes,
                labels=labels,
                split=split,
                illumination=illumination,
            )
        )
    return records
