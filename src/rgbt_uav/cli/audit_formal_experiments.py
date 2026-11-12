"""CLI for read-only formal experiment matrix auditing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.formal_experiment_audit import (
    audit_formal_experiment_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the formal experiment audit parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Perform a read-only audit of a formal "
            "experiment configuration matrix"
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Frozen formal experiment protocol YAML",
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
        help="Formal experiment matrix YAML",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the audit and print one JSON result."""

    args = build_parser().parse_args(argv)

    result = audit_formal_experiment_matrix(
        args.protocol,
        args.matrix,
    )

    print(
        json.dumps(
            {
                "status": "passed",
                "protocol_id": result.protocol_id,
                "run_count": result.run_count,
                "experiments": list(
                    result.experiments
                ),
                "seeds": list(
                    result.seeds
                ),
                "config_sha256": (
                    result.config_sha256
                ),
            },
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())