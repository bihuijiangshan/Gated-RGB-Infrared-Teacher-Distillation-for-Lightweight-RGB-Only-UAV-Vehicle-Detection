"""Profile the gated RGB-T teacher and lightweight RGB student."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import torch
from profile_official_teacher_student import (
    ProfilingError,
    extract_student_checkpoint,
    model_state_bytes,
    parameter_statistics,
    percentile,
    resolve_device,
    sha256_file,
)
from torch import Tensor, nn
from ultralytics.utils.torch_utils import thop

from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)
from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
)
from rgbt_uav.training.official_gated_teacher_checkpoint import (
    load_official_gated_teacher_weights,
)


class GatedTeacherProfileAdapter(nn.Module):
    """Expose the two-input gated teacher as one four-channel model."""

    def __init__(
        self,
        teacher: nn.Module,
    ) -> None:
        super().__init__()
        self.teacher = teacher

    def forward(
        self,
        fused: Tensor,
    ) -> object:
        if (
            fused.ndim != 4
            or fused.shape[1] != 4
        ):
            raise ProfilingError(
                "gated teacher profiling input "
                "must have four channels"
            )

        return self.teacher(
            fused[:, :3],
            fused[:, 3:4],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the gated RGB-T teacher and "
            "lightweight RGB student"
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
        "--iterations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--device",
        default="0",
        help='CUDA index such as "0", or "cpu"',
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


def direct_gflops(
    model: nn.Module,
    sample: Tensor,
) -> float:
    if thop is None:
        raise ProfilingError(
            "THOP is required for GFLOPs profiling"
        )

    copied = deepcopy(model)
    copied.eval()

    try:
        with torch.inference_mode():
            operations = thop.profile(
                copied,
                inputs=[sample],
                verbose=False,
            )[0]
    except Exception as exc:
        raise ProfilingError(
            "failed to profile model GFLOPs"
        ) from exc
    finally:
        del copied

    return float(operations) / 1e9 * 2.0


def profile_latency(
    model: nn.Module,
    sample: Tensor,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
    reference_memory_bytes: int,
) -> dict[str, float]:
    if warmup < 0:
        raise ProfilingError(
            "warmup must be non-negative"
        )
    if iterations <= 0:
        raise ProfilingError(
            "iterations must be positive"
        )

    model.eval()

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()

            baseline_memory = int(
                torch.cuda.memory_allocated(device)
            )

            torch.cuda.reset_peak_memory_stats(
                device
            )

            timings: list[float] = []

            for _ in range(iterations):
                start = torch.cuda.Event(
                    enable_timing=True,
                )
                end = torch.cuda.Event(
                    enable_timing=True,
                )

                start.record()
                model(sample)
                end.record()
                end.synchronize()

                timings.append(
                    float(start.elapsed_time(end))
                )

            peak_memory = int(
                torch.cuda.max_memory_allocated(
                    device
                )
            )
        else:
            baseline_memory = 0
            peak_memory = 0
            timings = []

            for _ in range(iterations):
                started = time.perf_counter()
                model(sample)
                timings.append(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000.0
                )

    mean_ms = statistics.fmean(timings)
    incremental_peak = max(
        peak_memory - baseline_memory,
        0,
    )
    resident_memory = max(
        baseline_memory
        - reference_memory_bytes,
        0,
    )
    isolated_peak_memory = max(
        peak_memory
        - reference_memory_bytes,
        0,
    )

    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings),
        "p90_ms": percentile(
            timings,
            0.90,
        ),
        "p95_ms": percentile(
            timings,
            0.95,
        ),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "fps_from_mean_latency": (1000.0 / mean_ms),
        "baseline_memory_bytes": float(baseline_memory),
        "baseline_memory_mib": (baseline_memory / (1024.0**2)),
        "peak_memory_bytes": float(peak_memory),
        "peak_memory_mib": (peak_memory / (1024.0**2)),
        "incremental_peak_memory_bytes": (float(incremental_peak)),
        "incremental_peak_memory_mib": (incremental_peak / (1024.0**2)),
        "reference_memory_bytes": float(reference_memory_bytes),
        "reference_memory_mib": (reference_memory_bytes / (1024.0**2)),
        "resident_memory_bytes": float(resident_memory),
        "resident_memory_mib": (resident_memory / (1024.0**2)),
        "isolated_peak_memory_bytes": float(isolated_peak_memory),
        "isolated_peak_memory_mib": (isolated_peak_memory / (1024.0**2)),
    }


def model_profile(
    model: nn.Module,
    sample: Tensor,
    *,
    checkpoint: Path,
    warmup: int,
    iterations: int,
    device: torch.device,
    reference_memory_bytes: int,
) -> dict[str, object]:
    state_bytes = model_state_bytes(model)

    return {
        "input_shape": list(sample.shape),
        "parameters": parameter_statistics(model),
        "parameter_memory_bytes": (state_bytes),
        "parameter_memory_mib": (state_bytes / (1024.0**2)),
        "gflops": direct_gflops(
            model,
            sample,
        ),
        "training_checkpoint": str(checkpoint),
        "training_checkpoint_bytes": (checkpoint.stat().st_size),
        "training_checkpoint_mib": (checkpoint.stat().st_size / (1024.0**2)),
        "training_checkpoint_sha256": (sha256_file(checkpoint)),
        "latency": profile_latency(
            model,
            sample,
            warmup=warmup,
            iterations=iterations,
            device=device,
            reference_memory_bytes=(reference_memory_bytes),
        ),
    }


def safe_reduction(
    student_value: float,
    teacher_value: float,
) -> float | None:
    if teacher_value <= 0.0:
        return None

    return 1.0 - student_value / teacher_value


def main(
    argv: list[str] | None = None,
) -> None:
    args = build_parser().parse_args(argv)

    if args.num_classes <= 0:
        raise ProfilingError(
            "num_classes must be positive"
        )
    if args.height <= 0:
        raise ProfilingError(
            "height must be positive"
        )
    if args.width <= 0:
        raise ProfilingError(
            "width must be positive"
        )
    if args.warmup < 0:
        raise ProfilingError(
            "warmup must be non-negative"
        )
    if args.iterations <= 0:
        raise ProfilingError(
            "iterations must be positive"
        )

    teacher_checkpoint = (
        args.teacher_checkpoint
    )
    student_checkpoint = (
        args.student_checkpoint
    )

    if not teacher_checkpoint.is_file():
        raise ProfilingError(
            "teacher checkpoint does not exist: "
            f"{teacher_checkpoint}"
        )
    if not student_checkpoint.is_file():
        raise ProfilingError(
            "student checkpoint does not exist: "
            f"{student_checkpoint}"
        )

    git = git_metadata()
    device = resolve_device(args.device)
    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        teacher_reference_memory = int(
            torch.cuda.memory_allocated(
                device
            )
        )
    else:
        teacher_reference_memory = 0

    teacher = build_official_gated_teacher(
        model_name=args.teacher_model,
        num_classes=args.num_classes,
        device=device,  # noqa: F821
    )
    teacher_payload = (
        load_official_gated_teacher_weights(
            teacher_checkpoint,
            teacher=teacher,
            map_location=device,
        )
    )
    teacher.eval()

    teacher_epoch = int(
        teacher_payload["epoch"]
    )
    teacher_experiment_id = str(
        teacher_payload["experiment_id"]
    )

    teacher_adapter = (
        GatedTeacherProfileAdapter(
            teacher
        ).to(device)
    )
    teacher_adapter.eval()

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

    teacher_profile = model_profile(
        teacher_adapter,
        fused,
        checkpoint=teacher_checkpoint,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
        reference_memory_bytes=(teacher_reference_memory),
    )

    del fused
    del teacher_adapter
    del teacher

    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    remaining_after_teacher = (
        int(torch.cuda.memory_allocated(device))
        if device.type == "cuda"
        else 0
    )

    student_reference_memory = remaining_after_teacher
    student = build_official_rgb_student(
        model_name=args.student_model,
        num_classes=args.num_classes,
        device=device,
    )
    student_epoch = extract_student_checkpoint(
        student_checkpoint,
        student=student,
        device=device,
    )
    student.model.eval()

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

    student_profile = model_profile(
        student.model,
        rgb,
        checkpoint=student_checkpoint,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
        reference_memory_bytes=(student_reference_memory),
    )

    teacher_parameters = float(
        teacher_profile[
            "parameters"
        ]["total"]
    )
    student_parameters = float(
        student_profile[
            "parameters"
        ]["total"]
    )
    teacher_gflops = float(
        teacher_profile["gflops"]
    )
    student_gflops = float(
        student_profile["gflops"]
    )
    teacher_latency = float(
        teacher_profile[
            "latency"
        ]["mean_ms"]
    )
    student_latency = float(
        student_profile[
            "latency"
        ]["mean_ms"]
    )
    teacher_isolated_peak = float(teacher_profile["latency"]["isolated_peak_memory_mib"])
    student_isolated_peak = float(student_profile["latency"]["isolated_peak_memory_mib"])
    teacher_incremental_peak = float(
        teacher_profile[
            "latency"
        ][
            "incremental_peak_memory_mib"
        ]
    )
    student_incremental_peak = float(
        student_profile[
            "latency"
        ][
            "incremental_peak_memory_mib"
        ]
    )

    comparison = {
        "student_to_teacher_parameter_ratio": (
            student_parameters
            / teacher_parameters
        ),
        "parameter_reduction_fraction": (
            1.0
            - student_parameters
            / teacher_parameters
        ),
        "student_to_teacher_gflops_ratio": (
            student_gflops
            / teacher_gflops
        ),
        "gflops_reduction_fraction": (
            1.0
            - student_gflops
            / teacher_gflops
        ),
        "student_to_teacher_latency_ratio": (
            student_latency
            / teacher_latency
        ),
        "latency_speedup": (
            teacher_latency
            / student_latency
        ),
        (
            "isolated_peak_memory_"
            "reduction_fraction"
        ): safe_reduction(
            student_isolated_peak,
            teacher_isolated_peak,
        ),
        (
            "incremental_peak_memory_"
            "reduction_fraction"
        ): safe_reduction(
            student_incremental_peak,
            teacher_incremental_peak,
        ),
    }

    report = {
        "format_version": 1,
        "status": "completed",
        "scope": (
            "batch-1 FP32 network-forward "
            "profiling; preprocessing, host-to-device "
            "transfer, decoding and NMS excluded"
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
        "profiling": {
            "height": args.height,
            "width": args.width,
            "batch": 1,
            "dtype": "float32",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "gflops_backend": (
                "THOP MACs multiplied by 2"
            ),
        },
        "teacher": {
            "kind": "official_gated",
            "model": args.teacher_model,
            "restored_epoch": teacher_epoch,
            "experiment_id": (
                teacher_experiment_id
            ),
            **teacher_profile,
        },
        "student": {
            "kind": (
                "official_rgb_width_reduced"
            ),
            "model": args.student_model,
            "restored_epoch": (
                student_epoch
            ),
            **student_profile,
        },
        "comparison": comparison,
        "notes": [
            (
                "Training checkpoint sizes include "
                "training-state metadata and are not "
                "deployment model sizes."
            ),
            (
                "Latency represents model forward "
                "execution only."
            ),
            (
                "The gated teacher consumes RGB and IR; "
                "the deployed student consumes RGB only."
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
        "official_gated_teacher_"
        "lightweight_student_profile_ok=True"
    )


if __name__ == "__main__":
    main()