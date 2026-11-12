"""Profile the official RGB-T teacher and distilled RGB student."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from ultralytics.utils.torch_utils import get_flops

from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
)
from rgbt_uav.training.official_distillation_teacher import (
    build_official_early_fusion_teacher,
    load_official_early_fusion_teacher_checkpoint,
)


class ProfilingError(RuntimeError):
    """Raised when teacher/student profiling cannot be completed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the official RGB-T teacher and distilled RGB student"
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
        "--model",
        default="yolov8n.yaml",
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
    parser.add_argument(
        "--student-model",
        default="yolov8n.yaml",
        help="Official RGB student model YAML",
    )
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")

    if not value.isdigit():
        raise ProfilingError(
            'device must be "cpu" or a non-negative CUDA index'
        )

    if not torch.cuda.is_available():
        raise ProfilingError(
            "CUDA was requested but is not available"
        )

    index = int(value)
    if index >= torch.cuda.device_count():
        raise ProfilingError(
            f"CUDA device index does not exist: {index}"
        )

    return torch.device(f"cuda:{index}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def parameter_statistics(model: nn.Module) -> dict[str, int]:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
    }


def model_state_bytes(model: nn.Module) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
        if isinstance(value, Tensor)
    )


def extract_student_checkpoint(
    path: Path,
    *,
    student: nn.Module,
    device: torch.device,
) -> int:
    if not path.is_file():
        raise ProfilingError(
            f"student checkpoint does not exist: {path}"
        )

    try:
        payload: Any = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except Exception as exc:
        raise ProfilingError(
            f"could not load student checkpoint: {path}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ProfilingError(
            "student checkpoint root must be a mapping"
        )

    epoch = payload.get("epoch")
    state = payload.get("student")

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch <= 0
    ):
        raise ProfilingError(
            "student checkpoint epoch must be positive"
        )

    if not isinstance(state, Mapping):
        raise ProfilingError(
            "student checkpoint must contain a student state mapping"
        )

    try:
        student.load_state_dict(
            state,
            strict=True,
        )
    except Exception as exc:
        raise ProfilingError(
            "student checkpoint is incompatible with the model"
        ) from exc

    student.eval()
    return epoch


def percentile(
    values: list[float],
    fraction: float,
) -> float:
    if not values:
        raise ProfilingError(
            "latency values must not be empty"
        )

    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def profile_latency(
    model: nn.Module,
    sample: Tensor,
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
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
            torch.cuda.reset_peak_memory_stats(device)

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

            peak_memory_bytes = int(
                torch.cuda.max_memory_allocated(device)
            )
        else:
            timings = []

            for _ in range(iterations):
                started = time.perf_counter()
                model(sample)
                timings.append(
                    (time.perf_counter() - started) * 1000.0
                )

            peak_memory_bytes = 0

    mean_ms = statistics.fmean(timings)

    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings),
        "p90_ms": percentile(timings, 0.90),
        "p95_ms": percentile(timings, 0.95),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "fps_from_mean_latency": 1000.0 / mean_ms,
        "peak_memory_bytes": float(peak_memory_bytes),
        "peak_memory_mib": peak_memory_bytes / (1024.0**2),
    }


def model_profile(
    model: nn.Module,
    sample: Tensor,
    *,
    checkpoint: Path,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    parameters = parameter_statistics(model)

    gflops = float(
        get_flops(
            model,
            imgsz=[
                int(sample.shape[2]),
                int(sample.shape[3]),
            ],
        )
    )

    return {
        "input_shape": list(sample.shape),
        "parameters": parameters,
        "parameter_memory_bytes": model_state_bytes(model),
        "parameter_memory_mib": (
            model_state_bytes(model) / (1024.0**2)
        ),
        "gflops": gflops,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mib": (
            checkpoint.stat().st_size / (1024.0**2)
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "latency": profile_latency(
            model,
            sample,
            warmup=warmup,
            iterations=iterations,
            device=device,
        ),
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

    teacher_checkpoint = args.teacher_checkpoint
    student_checkpoint = args.student_checkpoint

    if not teacher_checkpoint.is_file():
        raise ProfilingError(
            f"teacher checkpoint does not exist: {teacher_checkpoint}"
        )
    if not student_checkpoint.is_file():
        raise ProfilingError(
            f"student checkpoint does not exist: {student_checkpoint}"
        )

    device = resolve_device(args.device)

    teacher = build_official_early_fusion_teacher(
        model_name=args.model,
        num_classes=args.num_classes,
        device=device,
    )
    teacher_payload = (
        load_official_early_fusion_teacher_checkpoint(
            teacher_checkpoint,
            teacher=teacher,
            map_location=device,
        )
    )

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

    teacher.model.eval()
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
        teacher.model,
        fused,
        checkpoint=teacher_checkpoint,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    student_profile = model_profile(
        student.model,
        rgb,
        checkpoint=student_checkpoint,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    teacher_parameters = int(
        teacher_profile["parameters"]["total"]
    )
    student_parameters = int(
        student_profile["parameters"]["total"]
    )
    teacher_gflops = float(
        teacher_profile["gflops"]
    )
    student_gflops = float(
        student_profile["gflops"]
    )
    teacher_latency = float(
        teacher_profile["latency"]["mean_ms"]
    )
    student_latency = float(
        student_profile["latency"]["mean_ms"]
    )

    comparison = {
        "student_to_teacher_parameter_ratio": (
            student_parameters / teacher_parameters
        ),
        "parameter_reduction_fraction": (
            1.0 - student_parameters / teacher_parameters
        ),
        "student_to_teacher_gflops_ratio": (
            student_gflops / teacher_gflops
            if teacher_gflops > 0.0
            else None
        ),
        "gflops_reduction_fraction": (
            1.0 - student_gflops / teacher_gflops
            if teacher_gflops > 0.0
            else None
        ),
        "student_to_teacher_latency_ratio": (
            student_latency / teacher_latency
        ),
        "latency_speedup": (
            teacher_latency / student_latency
        ),
    }

    report = {
        "format_version": 1,
        "status": "completed",
        "scope": (
            "batch-1 FP32 network-forward profiling; "
            "preprocessing and NMS excluded"
        ),
        "device": {
            "type": device.type,
            "name": (
                torch.cuda.get_device_name(device)
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
        },
        "teacher": {
            "model": args.model,
            "restored_epoch": teacher_payload["epoch"],
            **teacher_profile,
        },
        "student": {
            "model": args.student_model,
            "restored_epoch": student_epoch,
            **student_profile,
        },
        "comparison": comparison,
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
    print("official_teacher_student_profile_ok=True")


if __name__ == "__main__":
    main()