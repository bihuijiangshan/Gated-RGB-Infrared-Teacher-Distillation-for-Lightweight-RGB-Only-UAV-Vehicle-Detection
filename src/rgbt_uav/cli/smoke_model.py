from __future__ import annotations

import argparse
import json

import torch

from rgbt_uav.evaluation.complexity import profile_complexity
from rgbt_uav.losses.distillation import DetectionDistillationLoss, FeatureDistiller
from rgbt_uav.models.detectors import StudentRGBTDetector, TeacherRGBTDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CPU teacher/student contract smoke test")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=8)
    args = parser.parse_args()

    rgb = torch.randn(1, 3, args.size, args.size)
    ir = torch.randn(1, 1, args.size, args.size)
    teacher = TeacherRGBTDetector(num_classes=5, base_channels=args.base_channels, reg_max=4)
    student = StudentRGBTDetector(num_classes=5, base_channels=args.base_channels, reg_max=4)

    teacher_output = teacher(rgb, ir)
    student_output = student(rgb, ir)
    channels = [feature.shape[1] for feature in student_output["features"]]
    feature_loss = FeatureDistiller(
        student_channels=channels,
        teacher_channels=channels,
    )(student_output["features"], teacher_output["features"])
    output_losses = DetectionDistillationLoss(num_classes=5)(
        student_output["predictions"], teacher_output["predictions"]
    )

    teacher_report = profile_complexity(teacher, (rgb, ir))
    student_report = profile_complexity(student, (rgb, ir))
    result = {
        "teacher_prediction_shapes": [list(value.shape) for value in teacher_output["predictions"]],
        "student_prediction_shapes": [list(value.shape) for value in student_output["predictions"]],
        "feature_distillation_loss": float(feature_loss.detach()),
        "output_distillation_loss": float(output_losses["total"].detach()),
        "teacher_complexity": teacher_report.__dict__,
        "student_complexity": student_report.__dict__,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
