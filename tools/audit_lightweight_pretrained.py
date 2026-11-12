"""Audit width-reduced initialization from real YOLOv8n weights."""

from pathlib import Path

import torch

from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
    load_official_rgb_student_width_reduced_pretrained,
)


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = Path("yolov8n.pt")
    model_path = Path(
        "configs/models/yolov8_rgb_student_tiny.yaml"
    )

    student = build_official_rgb_student(
        model_name=str(model_path),
        num_classes=5,
        device=device,
    )

    result = (
        load_official_rgb_student_width_reduced_pretrained(
            student,
            checkpoint,
        )
    )

    student.to(device)
    student.eval()

    sample = torch.zeros(
        1,
        3,
        512,
        640,
        device=device,
    )

    with torch.inference_mode():
        features, predictions = student(sample)

    print("device=", device)
    print("pretrained_path=", checkpoint)
    print("loaded_keys=", len(result["loaded_keys"]))
    print("exact_keys=", len(result["exact_keys"]))
    print("sliced_keys=", len(result["sliced_keys"]))
    print(
        "skipped_classification_keys=",
        len(result["skipped_classification_keys"]),
    )
    print("missing_keys=", len(result["missing_keys"]))
    print(
        "unexpected_keys=",
        len(result["unexpected_keys"]),
    )
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
    print("lightweight_pretrained_audit_ok=True")


if __name__ == "__main__":
    main()