"""Audit the structural complexity of the lightweight RGB student."""

from pathlib import Path

import torch
from ultralytics.utils.torch_utils import get_flops

from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
)

LIGHTWEIGHT_MODEL = Path(
    "configs/models/yolov8_rgb_student_tiny.yaml"
)


def parameter_count(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    standard = build_official_rgb_student(
        model_name="yolov8n.yaml",
        num_classes=5,
        device=device,
    )
    lightweight = build_official_rgb_student(
        model_name=str(LIGHTWEIGHT_MODEL),
        num_classes=5,
        device=device,
    )

    standard_parameters = parameter_count(
        standard.model
    )
    lightweight_parameters = parameter_count(
        lightweight.model
    )

    standard_gflops = float(
        get_flops(
            standard.model,
            imgsz=[512, 640],
        )
    )
    lightweight_gflops = float(
        get_flops(
            lightweight.model,
            imgsz=[512, 640],
        )
    )

    parameter_ratio = (
        lightweight_parameters / standard_parameters
    )
    parameter_reduction = 1.0 - parameter_ratio

    gflops_ratio = (
        lightweight_gflops / standard_gflops
        if standard_gflops > 0.0
        else None
    )
    gflops_reduction = (
        1.0 - gflops_ratio
        if gflops_ratio is not None
        else None
    )

    lightweight.eval()

    sample = torch.zeros(
        1,
        3,
        512,
        640,
        device=device,
    )

    with torch.inference_mode():
        features, predictions = lightweight(sample)

    print("device=", device)
    print("model=", LIGHTWEIGHT_MODEL)
    print("standard_params=", standard_parameters)
    print("lightweight_params=", lightweight_parameters)
    print("parameter_ratio=", parameter_ratio)
    print("parameter_reduction=", parameter_reduction)
    print("standard_gflops=", standard_gflops)
    print("lightweight_gflops=", lightweight_gflops)
    print("gflops_ratio=", gflops_ratio)
    print("gflops_reduction=", gflops_reduction)
    print(
        "feature_shapes=",
        [tuple(value.shape) for value in features],
    )
    print(
        "prediction_shapes=",
        [tuple(value.shape) for value in predictions],
    )
    print(
        "predictions_finite=",
        all(
            torch.isfinite(value).all().item()
            for value in predictions
        ),
    )
    print("lightweight_student_build_ok=True")


if __name__ == "__main__":
    main()