from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.data.builder import (
    build_dronevehicle_xml_manifest,
    build_split_manifest,
)
from rgbt_uav.data.manifest import manifest_sha256, write_jsonl


def _class_alias(value: str) -> tuple[str, str]:
    source, separator, target = value.partition("=")
    if not separator or not source.strip() or not target.strip():
        raise argparse.ArgumentTypeError("class alias must use SOURCE=TARGET")
    return source.strip(), target.strip()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a paired RGB-IR JSONL manifest")
    parser.add_argument(
        "--format", choices=("dota", "dronevehicle-xml"), default="dota"
    )
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--class-names",
        default="car,truck,bus,van,freight_car",
        help="Comma-separated canonical class names in label-id order",
    )
    parser.add_argument("--crop-left", type=float, default=100)
    parser.add_argument("--crop-top", type=float, default=100)
    parser.add_argument("--width", type=float, default=640)
    parser.add_argument("--height", type=float, default=512)
    parser.add_argument("--source-width", type=int, default=840)
    parser.add_argument("--source-height", type=int, default=712)
    parser.add_argument("--class-alias", action="append", type=_class_alias, default=[])
    parser.add_argument("--ignore-name", action="append", default=[])
    parser.add_argument("--ignore-degenerate", action="store_true")
    parser.add_argument("--illumination", default="unknown")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    class_to_id = {name: index for index, name in enumerate(names)}
    ignored_counts: dict[str, int] = {}
    ignored_geometry_counts: dict[str, int] = {}
    if args.format == "dronevehicle-xml":
        result = build_dronevehicle_xml_manifest(
            rgb_dir=args.rgb_dir,
            ir_dir=args.ir_dir,
            annotation_dir=args.annotation_dir,
            split=args.split,
            class_to_id=class_to_id,
            class_aliases=dict(args.class_alias),
            ignored_names=set(args.ignore_name),
            ignore_degenerate=args.ignore_degenerate,
            crop_left=args.crop_left,
            crop_top=args.crop_top,
            width=args.width,
            height=args.height,
            source_width=args.source_width,
            source_height=args.source_height,
            illumination=args.illumination,
        )
        records = result.records
        ignored_counts = result.ignored_names
        ignored_geometry_counts = result.ignored_geometries
    else:
        records = build_split_manifest(
            rgb_dir=args.rgb_dir,
            ir_dir=args.ir_dir,
            annotation_dir=args.annotation_dir,
            split=args.split,
            class_to_id=class_to_id,
            crop_left=args.crop_left,
            crop_top=args.crop_top,
            width=args.width,
            height=args.height,
            illumination=args.illumination,
        )
    write_jsonl(records, args.output)
    print(f"wrote {len(records)} records to {args.output}")
    print(f"sha256={manifest_sha256(args.output)}")
    if ignored_counts:
        print(f"ignored_names={json.dumps(ignored_counts, ensure_ascii=False, sort_keys=True)}")
    if ignored_geometry_counts:
        print(
            "ignored_geometries="
            f"{json.dumps(ignored_geometry_counts, ensure_ascii=False, sort_keys=True)}"
        )


if __name__ == "__main__":
    main()
