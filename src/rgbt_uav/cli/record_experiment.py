from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rgbt_uav.experiments.ledger import ExperimentRecord, append_record


def _metrics(value: str) -> dict[str, float]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("metrics must be a JSON object")
    try:
        return {str(key): float(number) for key, number in parsed.items()}
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("all metric values must be numeric") from exc


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append an immutable experiment record")
    parser.add_argument("--ledger", type=Path, default=Path("runs/ledger.jsonl"))
    parser.add_argument("--id", required=True, dest="experiment_id")
    parser.add_argument("--group", required=True)
    parser.add_argument("--config", required=True, dest="config_path")
    parser.add_argument("--commit", required=True, dest="git_commit")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--metrics", required=True, type=_metrics)
    parser.add_argument("--gpu-hours", required=True, type=float)
    parser.add_argument("--status", choices=["completed", "failed", "interrupted"], required=True)
    parser.add_argument("--notes", default="")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    append_record(
        args.ledger,
        ExperimentRecord(
            experiment_id=args.experiment_id,
            group=args.group,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_path=args.config_path,
            git_commit=args.git_commit,
            manifest_sha256=args.manifest_sha256,
            seed=args.seed,
            hardware=args.hardware,
            metrics=args.metrics,
            gpu_hours=args.gpu_hours,
            status=args.status,
            notes=args.notes,
        ),
    )


if __name__ == "__main__":
    main()
