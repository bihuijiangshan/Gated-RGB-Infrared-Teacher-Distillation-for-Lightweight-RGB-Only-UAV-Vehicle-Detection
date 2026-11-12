from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when modality pairing or manifest serialization is invalid."""


@dataclass(eq=True)
class SampleRecord:
    pair_id: str
    rgb_path: str
    ir_path: str
    boxes: list[list[float]]
    labels: list[int]
    split: str
    illumination: str = "unknown"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SampleRecord:
        return cls(
            pair_id=str(value["pair_id"]),
            rgb_path=str(value["rgb_path"]),
            ir_path=str(value["ir_path"]),
            boxes=[[float(number) for number in box] for box in value["boxes"]],
            labels=[int(label) for label in value["labels"]],
            split=str(value["split"]),
            illumination=str(value.get("illumination", "unknown")),
        )


def _files_by_stem(directory: Path, extensions: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise ManifestError(f"not a directory: {directory}")
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in result:
            raise ManifestError(f"duplicate stem {path.stem!r} in {directory}")
        result[path.stem] = path.resolve()
    return result


def pair_modalities(
    rgb_dir: Path,
    ir_dir: Path,
    *,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
) -> list[tuple[str, Path, Path]]:
    """Return strictly one-to-one RGB/IR pairs sorted by their shared filename stem."""
    normalized = {extension.lower() for extension in extensions}
    rgb = _files_by_stem(Path(rgb_dir), normalized)
    ir = _files_by_stem(Path(ir_dir), normalized)
    missing_ir = sorted(rgb.keys() - ir.keys())
    missing_rgb = sorted(ir.keys() - rgb.keys())
    problems: list[str] = []
    if missing_ir:
        problems.append(f"missing IR for {missing_ir[:10]}")
    if missing_rgb:
        problems.append(f"missing RGB for {missing_rgb[:10]}")
    if problems:
        raise ManifestError("; ".join(problems))
    return [(pair_id, rgb[pair_id], ir[pair_id]) for pair_id in sorted(rgb)]


def write_jsonl(records: list[SampleRecord], path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def read_jsonl(path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(SampleRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ManifestError(f"invalid JSONL record on line {line_number}") from exc
    return records


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronized_seed(pair_id: str, *, epoch: int, global_seed: int) -> int:
    """Derive a stable seed that both modalities use for geometric augmentation."""
    payload = f"{global_seed}:{epoch}:{pair_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def audit_records(
    records: list[SampleRecord], *, width: float, height: float, num_classes: int
) -> list[str]:
    issues: list[str] = []
    splits_by_pair: dict[str, set[str]] = {}
    occurrence_count: dict[tuple[str, str], int] = {}

    for record in records:
        splits_by_pair.setdefault(record.pair_id, set()).add(record.split)
        key = (record.pair_id, record.split)
        occurrence_count[key] = occurrence_count.get(key, 0) + 1
        if occurrence_count[key] > 1:
            issues.append(f"duplicate pair {record.pair_id!r} in split {record.split!r}")
        if not Path(record.rgb_path).is_file():
            issues.append(f"missing RGB file for {record.pair_id}: {record.rgb_path}")
        if not Path(record.ir_path).is_file():
            issues.append(f"missing IR file for {record.pair_id}: {record.ir_path}")
        if len(record.boxes) != len(record.labels):
            issues.append(f"box/label count mismatch for {record.pair_id}")
        for index, box in enumerate(record.boxes):
            valid = (
                len(box) == 4
                and 0 <= box[0] < box[2] <= width
                and 0 <= box[1] < box[3] <= height
            )
            if not valid:
                issues.append(f"invalid box {index} for {record.pair_id}: {box}")
        for index, label in enumerate(record.labels):
            if not 0 <= label < num_classes:
                issues.append(f"invalid label {index} for {record.pair_id}: {label}")

    for pair_id, splits in sorted(splits_by_pair.items()):
        if len(splits) > 1:
            issues.append(f"split leakage for {pair_id!r}: {sorted(splits)}")
    return issues
