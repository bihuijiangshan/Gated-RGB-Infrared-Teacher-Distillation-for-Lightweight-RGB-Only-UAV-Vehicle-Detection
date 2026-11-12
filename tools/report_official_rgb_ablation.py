"""Generate the official RGB distillation ablation evidence report."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from rgbt_uav.training.official_rgb_ablation_report import (
    write_official_rgb_ablation_report,
)

RUN_ROOTS = {
    "detection_only": Path(
        "runs/paired/official_rgb_detection_only_v1"
    ),
    "feature_only": Path(
        "runs/paired/official_rgb_feature_only_v1"
    ),
    "logit_only": Path(
        "runs/paired/official_rgb_logit_only_v1"
    ),
    "full": Path(
        "runs/paired/official_rgb_distillation_v1"
    ),
}

DESTINATION = Path(
    "runs/archives/"
    "official_rgb_distillation_ablation_20260827/"
    "summary.json"
)


def _git_output(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            text=True,
            encoding="utf-8",
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise RuntimeError(
            f"git command failed: {' '.join(arguments)}"
        ) from exc


def main() -> None:
    dirty = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if dirty:
        raise RuntimeError(
            "tracked working tree must be clean before "
            "generating ablation evidence"
        )

    git_commit = _git_output(
        "rev-parse",
        "HEAD",
    )

    destination = write_official_rgb_ablation_report(
        RUN_ROOTS,
        DESTINATION,
        git_commit=git_commit,
        dataset_name="DroneVehicle smoke 5%",
    )

    report = json.loads(
        destination.read_text(encoding="utf-8")
    )

    print("report_path=", destination)
    print("git_commit=", report["git_commit"])
    print("dataset=", report["dataset"])

    for name in (
        "detection_only",
        "feature_only",
        "logit_only",
        "full",
    ):
        run = report["runs"][name]
        print(
            name,
            {
                "mAP50-95": run["mAP50-95"],
                "mAP50": run["mAP50"],
                "val_loss": run["val_loss"],
                "restored_epoch": run["restored_epoch"],
                "best_weights_sha256": (
                    run["best_weights_sha256"]
                ),
            },
        )

    for name, comparison in report[
        "comparisons"
    ].items():
        print(name, comparison)

    print(
        "official_rgb_ablation_report_ok=True"
    )


if __name__ == "__main__":
    main()