from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbt_uav.data.manifest_set import audit_manifest_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an official RGB-IR manifest split set")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--width", type=float, default=640)
    parser.add_argument("--height", type=float, default=512)
    parser.add_argument("--num-classes", type=int, default=5)
    args = parser.parse_args()

    report = audit_manifest_set(
        {"train": args.train, "val": args.val, "test": args.test},
        width=args.width,
        height=args.height,
        num_classes=args.num_classes,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if report.issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
