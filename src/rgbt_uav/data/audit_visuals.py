from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

from rgbt_uav.data.manifest import SampleRecord


@dataclass(frozen=True)
class AuditSelection:
    record: SampleRecord
    stratum: str
    luminance: float


def compute_cropped_luminance(record: SampleRecord) -> float:
    with Image.open(record.rgb_path) as image:
        cropped = image.convert("L").crop((100, 100, 740, 612)).resize((64, 51))
        return float(ImageStat.Stat(cropped).mean[0])


def select_audit_records(
    records: list[SampleRecord],
    luminance_by_pair: dict[str, float],
    *,
    count: int,
    seed: int,
) -> list[AuditSelection]:
    if not 0 < count <= len(records):
        raise ValueError("count must be positive and no larger than records")
    missing = [record.pair_id for record in records if record.pair_id not in luminance_by_pair]
    if missing:
        raise ValueError(f"missing luminance for {missing[0]}")

    selected: list[AuditSelection] = []
    used: set[str] = set()

    def take(candidates: list[SampleRecord], amount: int, stratum: str) -> None:
        for record in candidates:
            if len([item for item in selected if item.stratum == stratum]) >= amount:
                break
            if record.pair_id in used:
                continue
            selected.append(
                AuditSelection(
                    record=record,
                    stratum=stratum,
                    luminance=luminance_by_pair[record.pair_id],
                )
            )
            used.add(record.pair_id)

    dark_count = max(1, round(count * 0.20))
    bright_count = max(1, round(count * 0.10))
    empty_count = max(1, round(count * 0.15))
    dense_count = max(1, round(count * 0.20))
    by_luminance = sorted(records, key=lambda item: luminance_by_pair[item.pair_id])
    take(by_luminance, dark_count, "dark")
    take(list(reversed(by_luminance)), bright_count, "bright")
    empty_records = sorted(
        (item for item in records if not item.boxes), key=lambda item: item.pair_id
    )
    take(empty_records, empty_count, "empty")
    dense_records = sorted(records, key=lambda item: (-len(item.boxes), item.pair_id))
    take(dense_records, dense_count, "dense")

    remaining = [item for item in records if item.pair_id not in used]
    random.Random(seed).shuffle(remaining)
    take(remaining, count - len(selected), "random")
    if len(selected) < count:
        backfill = [item for item in records if item.pair_id not in used]
        take(backfill, count - len(selected), "backfill")
    return selected


def _render_pair(selection: AuditSelection) -> Image.Image:
    panel_width = 320
    panel_height = 256
    title_height = 32
    with Image.open(selection.record.rgb_path) as image:
        rgb = image.convert("RGB").crop((100, 100, 740, 612)).resize((panel_width, panel_height))
    with Image.open(selection.record.ir_path) as image:
        infrared = (
            image.convert("L")
            .crop((100, 100, 740, 612))
            .resize((panel_width, panel_height))
            .convert("RGB")
        )

    colors = ("#ff3b30", "#34c759", "#007aff", "#ffcc00", "#af52de")
    for canvas in (rgb, infrared):
        draw = ImageDraw.Draw(canvas)
        for box, label in zip(selection.record.boxes, selection.record.labels, strict=True):
            scaled = tuple(value * 0.5 for value in box)
            draw.rectangle(scaled, outline=colors[label % len(colors)], width=2)

    row = Image.new("RGB", (panel_width * 2, panel_height + title_height), "#111111")
    row.paste(rgb, (0, title_height))
    row.paste(infrared, (panel_width, title_height))
    title = (
        f"{selection.record.pair_id} | {selection.stratum} | "
        f"Y={selection.luminance:.1f} | boxes={len(selection.record.boxes)} | RGB / IR"
    )
    ImageDraw.Draw(row).text((8, 8), title, fill="white")
    return row


def render_audit_sheets(
    selections: list[AuditSelection], *, output_dir: Path, per_sheet: int = 10
) -> list[Path]:
    if per_sheet <= 0:
        raise ValueError("per_sheet must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    row_height = 288
    for sheet_index in range(math.ceil(len(selections) / per_sheet)):
        group = selections[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        sheet = Image.new("RGB", (640, row_height * len(group)), "black")
        for row_index, selection in enumerate(group):
            sheet.paste(_render_pair(selection), (0, row_index * row_height))
        output = destination / f"audit_sheet_{sheet_index + 1:02d}.jpg"
        sheet.save(output, quality=92)
        outputs.append(output)
    return outputs
