"""CLI for validation-only formal teacher checkpoint selection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.evaluation.formal_teacher import (
    evaluate_formal_teacher_checkpoint,
)
from rgbt_uav.training.formal_teacher_history import (
    evaluate_formal_teacher_history,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate formal teacher epoch history on val only"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=("early_fusion", "gated"), required=True
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_formal_teacher_history(
        args.run_root,
        kind=args.kind,
        data_root=args.data_root,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_det=args.max_det,
        evaluation_fn=evaluate_formal_teacher_checkpoint,
    )
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    print(json.dumps(printable, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
