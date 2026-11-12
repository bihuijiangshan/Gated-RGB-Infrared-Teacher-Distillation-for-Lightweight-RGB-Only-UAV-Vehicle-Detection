"""Execute exactly one explicitly authorized formal experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rgbt_uav.training.formal_run_executor import (
    build_authorization_template,
    execute_formal_run,
    inspect_formal_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one formal experiment only after recorded authorization, "
            "an exact confirmation, and all safety checks"
        )
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--resource-budget", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(
        "configs/experiments/formal_experiment_protocol.yaml"
    ))
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-template", action="store_true",
                        help="Print an unapproved single-run authorization draft")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required explicit execution switch; otherwise no training runs",
    )
    parser.add_argument(
        "--confirm",
        help="Must exactly equal EXPERIMENT:SEED when --execute is supplied",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.authorization_template:
        if args.execute or args.authorization or args.confirm:
            parser.error("--authorization-template cannot execute or approve a run")
        draft = build_authorization_template(
            args.matrix, args.resource_budget, args.protocol,
            experiment=args.experiment, seed=args.seed,
        )
        print(json.dumps(draft, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if not args.execute:
        run = inspect_formal_run(
            args.matrix,
            args.resource_budget,
            experiment=args.experiment,
            seed=args.seed,
            authorization_path=args.authorization,
            protocol_path=args.protocol,
        )
        print(
            json.dumps(
                {
                    "status": "not_executed",
                    "execute": False,
                    "experiment": args.experiment,
                    "seed": args.seed,
                    "required_confirmation": (
                        f"{args.experiment}:{args.seed}"
                    ),
                    "authorized": (
                        "full_training_not_authorized" not in run.blockers
                    ),
                    "ready": run.ready,
                    "blockers": list(run.blockers),
                    "training_argv": list(run.training_argv),
                    "output_root": str(run.output_root).replace("\\", "/"),
                    "test_evaluated": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.confirm is None:
        parser.error("--execute requires --confirm")
    if args.authorization is None:
        parser.error("--execute requires --authorization")

    result = execute_formal_run(
        args.matrix,
        args.resource_budget,
        experiment=args.experiment,
        seed=args.seed,
        confirmation=args.confirm,
        authorization_path=args.authorization,
        protocol_path=args.protocol,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "execute": True,
                "experiment": result.experiment,
                "seed": result.seed,
                "config": str(result.config).replace("\\", "/"),
                "output_root": str(result.output_root).replace("\\", "/"),
                "git_commit": result.git_commit,
                "return_code": result.return_code,
                "test_evaluated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
