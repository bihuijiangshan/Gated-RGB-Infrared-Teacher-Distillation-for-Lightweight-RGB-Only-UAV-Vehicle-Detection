from __future__ import annotations

import xml.etree.ElementTree as element_tree
from collections.abc import Mapping, Set
from dataclasses import dataclass
from pathlib import Path

from rgbt_uav.data.annotations import AnnotationError, obb_to_hbb


class DroneVehicleXmlError(ValueError):
    """Raised when a native DroneVehicle XML annotation is invalid or ambiguous."""


@dataclass(frozen=True)
class ParsedDroneVehicleAnnotation:
    boxes: list[list[float]]
    labels: list[int]
    ignored_names: list[str]
    ignored_geometries: list[str]


def _required_text(parent: element_tree.Element, path: str, *, source: Path) -> str:
    value = parent.findtext(path)
    if value is None or not value.strip():
        raise DroneVehicleXmlError(f"{source}: missing XML value {path!r}")
    return value.strip()


def parse_dronevehicle_xml(
    path: Path,
    *,
    class_to_id: Mapping[str, int],
    crop_left: float,
    crop_top: float,
    width: float,
    height: float,
    source_width: int,
    source_height: int,
    class_aliases: Mapping[str, str] | None = None,
    ignored_names: Set[str] | None = None,
    ignore_degenerate: bool = False,
) -> ParsedDroneVehicleAnnotation:
    """Parse native polygon XML into cropped HBBs without hiding label anomalies."""

    source = Path(path)
    try:
        root = element_tree.parse(source).getroot()
    except (OSError, element_tree.ParseError) as exc:
        raise DroneVehicleXmlError(f"cannot parse {source}: {exc}") from exc

    try:
        xml_width = int(_required_text(root, "size/width", source=source))
        xml_height = int(_required_text(root, "size/height", source=source))
    except ValueError as exc:
        raise DroneVehicleXmlError(f"{source}: non-integer source dimensions") from exc
    if (xml_width, xml_height) != (source_width, source_height):
        raise DroneVehicleXmlError(
            f"{source}: source dimensions {(xml_width, xml_height)} do not match "
            f"expected {(source_width, source_height)}"
        )

    aliases = dict(class_aliases or {})
    ignored = set(ignored_names or set())
    boxes: list[list[float]] = []
    labels: list[int] = []
    ignored_found: list[str] = []
    ignored_geometries: list[str] = []
    for object_index, item in enumerate(root.findall("object")):
        raw_name = _required_text(item, "name", source=source)
        if raw_name in ignored:
            ignored_found.append(raw_name)
            continue
        canonical_name = aliases.get(raw_name, raw_name)
        if canonical_name not in class_to_id:
            raise DroneVehicleXmlError(
                f"{source}: unknown class {raw_name!r} on object {object_index}"
            )
        polygon = item.find("polygon")
        horizontal = item.find("bndbox")
        if polygon is None and horizontal is None and item.find("point") is not None:
            ignored_geometries.append("point")
            continue
        try:
            if polygon is not None:
                points: list[float] = []
                for index in range(1, 5):
                    points.extend(
                        [
                            float(_required_text(polygon, f"x{index}", source=source)),
                            float(_required_text(polygon, f"y{index}", source=source)),
                        ]
                    )
            else:
                if horizontal is None:
                    raise DroneVehicleXmlError(
                        f"{source}: object {object_index} has neither polygon nor bndbox"
                    )
                xmin = float(_required_text(horizontal, "xmin", source=source))
                ymin = float(_required_text(horizontal, "ymin", source=source))
                xmax = float(_required_text(horizontal, "xmax", source=source))
                ymax = float(_required_text(horizontal, "ymax", source=source))
                points = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
            box = obb_to_hbb(
                points,
                crop_left=crop_left,
                crop_top=crop_top,
                width=width,
                height=height,
            )
        except AnnotationError as exc:
            if ignore_degenerate:
                ignored_geometries.append("degenerate")
                continue
            raise DroneVehicleXmlError(
                f"{source}: invalid polygon on object {object_index}: {exc}"
            ) from exc
        except ValueError as exc:
            raise DroneVehicleXmlError(
                f"{source}: invalid polygon on object {object_index}: {exc}"
            ) from exc
        boxes.append(box)
        labels.append(int(class_to_id[canonical_name]))

    return ParsedDroneVehicleAnnotation(
        boxes=boxes,
        labels=labels,
        ignored_names=ignored_found,
        ignored_geometries=ignored_geometries,
    )
