from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rgbt_uav.data.manifest import audit_records, manifest_sha256, read_jsonl


@dataclass(frozen=True)
class SplitManifestReport:
    path: str
    sha256: str
    records: int
    boxes: int


@dataclass(frozen=True)
class ManifestSetReport:
    splits: dict[str, SplitManifestReport]
    issues: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "splits": {
                name: {
                    "path": item.path,
                    "sha256": item.sha256,
                    "records": item.records,
                    "boxes": item.boxes,
                }
                for name, item in sorted(self.splits.items())
            },
            "issues": list(self.issues),
        }


def audit_manifest_set(
    manifest_paths: Mapping[str, Path],
    *,
    width: float,
    height: float,
    num_classes: int,
) -> ManifestSetReport:
    required = {"train", "val", "test"}
    if set(manifest_paths) != required:
        raise ValueError(f"manifest set must contain {sorted(required)}")

    reports: dict[str, SplitManifestReport] = {}
    issues: list[str] = []
    owners_by_source: dict[Path, set[str]] = {}
    for split in sorted(required):
        path = Path(manifest_paths[split])
        records = read_jsonl(path)
        reports[split] = SplitManifestReport(
            path=str(path.resolve()),
            sha256=manifest_sha256(path),
            records=len(records),
            boxes=sum(len(record.boxes) for record in records),
        )
        issues.extend(
            f"{split}: {item}"
            for item in audit_records(
                records, width=width, height=height, num_classes=num_classes
            )
        )
        for record in records:
            if record.split != split:
                issues.append(
                    f"{split}: record {record.pair_id!r} declares split {record.split!r}"
                )
            for source in (record.rgb_path, record.ir_path):
                owners_by_source.setdefault(Path(source).resolve(), set()).add(split)

    for source, owners in sorted(owners_by_source.items(), key=lambda item: str(item[0])):
        if len(owners) > 1:
            issues.append(
                f"source file reused across splits: {source} in {sorted(owners)}"
            )

    return ManifestSetReport(splits=reports, issues=issues)
