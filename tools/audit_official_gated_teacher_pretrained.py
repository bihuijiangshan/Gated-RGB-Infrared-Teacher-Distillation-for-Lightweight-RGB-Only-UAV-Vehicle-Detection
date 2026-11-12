"""Audit pretrained initialization of the official gated teacher."""

from pathlib import Path

import torch

from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)
from rgbt_uav.training.official_gated_teacher_pretrained import (
    load_official_gated_teacher_pretrained,
)


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    teacher = build_official_gated_teacher(
        model_name="yolov8n.yaml",
        num_classes=5,
        device=device,
    )

    result = (
        load_official_gated_teacher_pretrained(
            teacher,
            Path("yolov8n.pt"),
        )
    )

    rgb_first = (
        teacher.rgb_stream.model[0]
        .conv.weight.detach()
    )
    ir_first = (
        teacher.ir_stream.model[0]
        .conv.weight.detach()
    )

    expected_ir_first = rgb_first.mean(
        dim=1,
        keepdim=True,
    )

    ir_mean_max_abs_error = float((ir_first - expected_ir_first).abs().max().cpu())

    ir_matches_rgb_mean = torch.allclose(
        ir_first,
        expected_ir_first,
        rtol=1e-6,
        atol=1e-7,
    )

    detect_cv3_missing = any(
        key.startswith("cv3.")
        for key in result[
            "detect_missing_keys"
        ]
    )

    teacher_trainable = all(
        parameter.requires_grad
        for parameter in teacher.parameters()
    )

    print("device=", device)
    print(
        "rgb_first_conv_shape=",
        tuple(rgb_first.shape),
    )
    print(
        "ir_first_conv_shape=",
        tuple(ir_first.shape),
    )
    print(
        "ir_matches_rgb_mean=",
        ir_matches_rgb_mean,
    )
    print(
        "ir_mean_max_abs_error=",
        ir_mean_max_abs_error,
    )
    print(
        "rgb_loaded_keys=",
        len(result["rgb_loaded_keys"]),
    )
    print(
        "rgb_missing_keys=",
        len(result["rgb_missing_keys"]),
    )
    print(
        "ir_loaded_keys=",
        len(result["ir_loaded_keys"]),
    )
    print(
        "ir_missing_keys=",
        len(result["ir_missing_keys"]),
    )
    print(
        "ir_adapted_keys=",
        result["ir_adapted_keys"],
    )
    print(
        "detect_loaded_keys=",
        len(result["detect_loaded_keys"]),
    )
    print(
        "detect_missing_keys=",
        len(result["detect_missing_keys"]),
    )
    print(
        "detect_cv3_missing=",
        detect_cv3_missing,
    )
    print(
        "teacher_trainable=",
        teacher_trainable,
    )

    assert rgb_first.shape[1] == 3
    assert ir_first.shape[1] == 1
    assert ir_matches_rgb_mean
    assert result["ir_adapted_keys"] == (
        "model.0.conv.weight",
    )
    assert result["rgb_loaded_keys"]
    assert result["ir_loaded_keys"]
    assert result["detect_loaded_keys"]
    assert detect_cv3_missing
    assert teacher_trainable

    print(
        "official_gated_teacher_pretrained_ok=True"
    )


if __name__ == "__main__":
    main()