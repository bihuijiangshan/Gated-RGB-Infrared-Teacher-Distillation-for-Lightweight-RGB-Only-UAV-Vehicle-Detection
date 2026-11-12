from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class LedgerError(ValueError):
    """Raised when an experiment record would break provenance guarantees."""


@dataclass(eq=True, frozen=True)
class ExperimentRecord:
    experiment_id: str
    group: str
    timestamp: str
    config_path: str
    git_commit: str
    manifest_sha256: str
    seed: int
    hardware: str
    metrics: dict[str, float]
    gpu_hours: float
    status: str
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentRecord:
        return cls(
            experiment_id=str(value["experiment_id"]),
            group=str(value["group"]),
            timestamp=str(value["timestamp"]),
            config_path=str(value["config_path"]),
            git_commit=str(value["git_commit"]),
            manifest_sha256=str(value["manifest_sha256"]),
            seed=int(value["seed"]),
            hardware=str(value["hardware"]),
            metrics={str(key): float(number) for key, number in value["metrics"].items()},
            gpu_hours=float(value["gpu_hours"]),
            status=str(value["status"]),
            notes=str(value.get("notes", "")),
        )


def load_records(path: Path) -> list[ExperimentRecord]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[ExperimentRecord] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(ExperimentRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise LedgerError(f"invalid ledger record on line {line_number}") from exc
    return records


def append_record(path: Path, record: ExperimentRecord) -> None:
    destination = Path(path)
    existing = load_records(destination)
    if any(item.experiment_id == record.experiment_id for item in existing):
        raise LedgerError(f"duplicate experiment_id: {record.experiment_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
        stream.write("\n")


def summarize_metrics(
    records: list[ExperimentRecord], *, group: str
) -> dict[str, dict[str, float | int]]:
    selected = [
        record for record in records if record.group == group and record.status == "completed"
    ]
    values_by_metric: dict[str, list[float]] = {}
    for record in selected:
        for metric, value in record.metrics.items():
            values_by_metric.setdefault(metric, []).append(float(value))

    summary: dict[str, dict[str, float | int]] = {}
    for metric, values in sorted(values_by_metric.items()):
        summary[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return summary
