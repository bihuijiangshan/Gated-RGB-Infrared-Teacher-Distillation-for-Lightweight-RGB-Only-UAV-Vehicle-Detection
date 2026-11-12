"""Paired latency benchmark for gated teacher and RGB student."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from profile_official_gated_teacher_student import (
    GatedTeacherProfileAdapter,
)
from profile_official_teacher_student import (
    ProfilingError,
    extract_student_checkpoint,
    percentile,
    resolve_device,
    sha256_file,
)
from torch import Tensor, nn

from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)
from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
)
from rgbt_uav.training.official_gated_teacher_checkpoint import (
    load_official_gated_teacher_weights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an adjacent paired latency benchmark "
            "for the gated teacher and RGB student"
        )
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--teacher-model",
        default="yolov8n.yaml",
    )
    parser.add_argument(
        "--student-model",
        default=(
            "configs/models/"
            "yolov8_rgb_student_tiny.yaml"
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--device",
        default="0",
    )
    return parser


def git_metadata() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise ProfilingError(
            "failed to read git metadata"
        ) from exc

    if status:
        raise ProfilingError(
            "tracked working tree must be clean"
        )

    return {
        "commit": commit,
        "tracked_worktree_clean": True,
    }


def measure_once(
    model: nn.Module,
    sample: Tensor,
    *,
    device: torch.device,
) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(
            enable_timing=True
        )
        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()
        model(sample)
        end.record()
        end.synchronize()

        return float(
            start.elapsed_time(end)
        )

    started = time.perf_counter()
    model(sample)

    return (
        time.perf_counter() - started
    ) * 1000.0


def summary(
    values: list[float],
) -> dict[str, float]:
    if len(values) < 2:
        raise ProfilingError(
            "at least two latency values are required"
        )

    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "median": statistics.median(values),
        "p05": percentile(values, 0.05),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def main(
    argv: list[str] | None = None,
) -> None:
    args = build_parser().parse_args(argv)

    if args.num_classes <= 0:
        raise ProfilingError(
            "num_classes must be positive"
        )
    if args.height <= 0 or args.width <= 0:
        raise ProfilingError(
            "height and width must be positive"
        )
    if args.warmup < 0:
        raise ProfilingError(
            "warmup must be non-negative"
        )
    if args.pairs < 2:
        raise ProfilingError(
            "pairs must be at least two"
        )

    if not args.teacher_checkpoint.is_file():
        raise ProfilingError(
            "teacher checkpoint does not exist: "
            f"{args.teacher_checkpoint}"
        )
    if not args.student_checkpoint.is_file():
        raise ProfilingError(
            "student checkpoint does not exist: "
            f"{args.student_checkpoint}"
        )

    git = git_metadata()
    device = resolve_device(args.device)

    teacher = build_official_gated_teacher(
        model_name=args.teacher_model,
        num_classes=args.num_classes,
        device=device,
    )
    teacher_payload = (
        load_official_gated_teacher_weights(
            args.teacher_checkpoint,
            teacher=teacher,
            map_location=device,
        )
    )
    teacher.eval()

    teacher_adapter = (
        GatedTeacherProfileAdapter(
            teacher
        ).to(device)
    )
    teacher_adapter.eval()

    student = build_official_rgb_student(
        model_name=args.student_model,
        num_classes=args.num_classes,
        device=device,
    )
    student_epoch = extract_student_checkpoint(
        args.student_checkpoint,
        student=student,
        device=device,
    )
    student.model.eval()

    fused = torch.zeros(
        (
            1,
            4,
            args.height,
            args.width,
        ),
        device=device,
        dtype=torch.float32,
    )
    rgb = torch.zeros(
        (
            1,
            3,
            args.height,
            args.width,
        ),
        device=device,
        dtype=torch.float32,
    )

    teacher_latencies: list[float] = []
    student_latencies: list[float] = []
    paired_speedups: list[float] = []
    execution_orders: list[str] = []

    with torch.inference_mode():
        for index in range(args.warmup):
            if index % 2 == 0:
                teacher_adapter(fused)
                student.model(rgb)
            else:
                student.model(rgb)
                teacher_adapter(fused)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for index in range(args.pairs):
            if index % 2 == 0:
                execution_orders.append(
                    "teacher_then_student"
                )
                teacher_ms = measure_once(
                    teacher_adapter,
                    fused,
                    device=device,
                )
                student_ms = measure_once(
                    student.model,
                    rgb,
                    device=device,
                )
            else:
                execution_orders.append(
                    "student_then_teacher"
                )
                student_ms = measure_once(
                    student.model,
                    rgb,
                    device=device,
                )
                teacher_ms = measure_once(
                    teacher_adapter,
                    fused,
                    device=device,
                )

            if teacher_ms <= 0.0:
                raise ProfilingError(
                    "teacher latency must be positive"
                )
            if student_ms <= 0.0:
                raise ProfilingError(
                    "student latency must be positive"
                )

            teacher_latencies.append(
                teacher_ms
            )
            student_latencies.append(
                student_ms
            )
            paired_speedups.append(
                teacher_ms / student_ms
            )

    teacher_summary = summary(
        teacher_latencies
    )
    student_summary = summary(
        student_latencies
    )
    speedup_summary = summary(
        paired_speedups
    )

    ratio_of_means = (
        teacher_summary["mean"]
        / student_summary["mean"]
    )

    report = {
        "format_version": 1,
        "status": "completed",
        "scope": (
            "adjacent paired batch-1 FP32 "
            "network-forward latency; both models "
            "resident; preprocessing, transfer, "
            "decoding and NMS excluded"
        ),
        "git": git,
        "device": {
            "type": str(device),
            "name": (
                torch.cuda.get_device_name(
                    device
                )
                if device.type == "cuda"
                else "CPU"
            ),
        },
        "protocol": {
            "height": args.height,
            "width": args.width,
            "batch": 1,
            "dtype": "float32",
            "warmup_pairs": args.warmup,
            "measured_pairs": args.pairs,
            "execution_order": (
                "alternating teacher-first "
                "and student-first"
            ),
            "teacher_first_count": (
                execution_orders.count(
                    "teacher_then_student"
                )
            ),
            "student_first_count": (
                execution_orders.count(
                    "student_then_teacher"
                )
            ),
        },
        "teacher": {
            "kind": "official_gated",
            "model": args.teacher_model,
            "checkpoint": str(
                args.teacher_checkpoint
            ),
            "checkpoint_sha256": sha256_file(
                args.teacher_checkpoint
            ),
            "restored_epoch": (
                teacher_payload["epoch"]
            ),
            "experiment_id": (
                teacher_payload[
                    "experiment_id"
                ]
            ),
            "input_shape": list(
                fused.shape
            ),
            "latency_ms": teacher_summary,
            "fps_from_mean_latency": (
                1000.0
                / teacher_summary["mean"]
            ),
        },
        "student": {
            "kind": (
                "official_rgb_width_reduced"
            ),
            "model": args.student_model,
            "checkpoint": str(
                args.student_checkpoint
            ),
            "checkpoint_sha256": sha256_file(
                args.student_checkpoint
            ),
            "restored_epoch": student_epoch,
            "input_shape": list(
                rgb.shape
            ),
            "latency_ms": student_summary,
            "fps_from_mean_latency": (
                1000.0
                / student_summary["mean"]
            ),
        },
        "comparison": {
            "paired_speedup": speedup_summary,
            "ratio_of_mean_latencies": (
                ratio_of_means
            ),
            "mean_latency_reduction_fraction": (
                1.0
                - student_summary["mean"]
                / teacher_summary["mean"]
            ),
        },
        "notes": [
            (
                "Absolute latency may vary with laptop "
                "GPU power and scheduling state."
            ),
            (
                "The primary latency result is the "
                "paired speedup under alternating order."
            ),
            (
                "Both models remain resident because "
                "this tool measures latency only, not memory."
            ),
        ],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"metrics_path= {args.output}")
    print(
        "paired_gated_teacher_student_"
        "latency_benchmark_ok=True"
    )


if __name__ == "__main__":
    main()