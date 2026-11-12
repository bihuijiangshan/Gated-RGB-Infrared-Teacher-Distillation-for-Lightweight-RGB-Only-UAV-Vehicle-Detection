from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgbt_uav.data.manifest import audit_records, manifest_sha256, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a paired RGB-IR JSONL manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--width", type=float, default=640)
    parser.add_argument("--height", type=float, default=512)
    parser.add_argument("--num-classes", type=int, default=5)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    issues = audit_records(
        records, width=args.width, height=args.height, num_classes=args.num_classes
    )
    report = {
        "manifest": str(args.manifest.resolve()),
        "sha256": manifest_sha256(args.manifest),
        "records": len(records),
        "issues": issues,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

