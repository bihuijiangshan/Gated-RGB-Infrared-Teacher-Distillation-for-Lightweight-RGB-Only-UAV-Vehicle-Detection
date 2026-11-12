from pathlib import Path

import torch

from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
    load_official_rgb_student_pretrained,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

pretrained_path = Path("yolov8n.pt")
if not pretrained_path.is_file():
    raise FileNotFoundError(
        f"missing pretrained weights: {pretrained_path}"
    )

student = build_official_rgb_student(
    model_name="yolov8n.yaml",
    num_classes=5,
    device=device,
)

result = load_official_rgb_student_pretrained(
    student,
    pretrained_path,
)

first_conv = student.model.model[0].conv.weight
loaded_keys = result["loaded_keys"]
missing_keys = result["missing_keys"]
unexpected_keys = result["unexpected_keys"]

student.train()
rgb = torch.rand(
    2,
    3,
    512,
    640,
    device=device,
)

features, predictions = student(rgb)

print("device=", device)
print("pretrained_path=", pretrained_path)
print("first_conv_shape=", tuple(first_conv.shape))
print("loaded_keys=", len(loaded_keys))
print("missing_keys=", len(missing_keys))
print("unexpected_keys=", len(unexpected_keys))
print(
    "head_keys_skipped=",
    sum(
        key.startswith("model.22.")
        for key in missing_keys
    ),
)
print(
    "features=",
    [tuple(feature.shape) for feature in features],
)
print(
    "predictions=",
    [tuple(prediction.shape) for prediction in predictions],
)
print(
    "predictions_finite=",
    all(
        torch.isfinite(prediction).all().item()
        for prediction in predictions
    ),
)

assert first_conv.shape == (16, 3, 3, 3)
assert len(loaded_keys) > 250
assert len(unexpected_keys) == 0
assert len(features) == 3
assert len(predictions) == 3
assert all(
    prediction.shape[1] == 69
    for prediction in predictions
)
assert all(
    torch.isfinite(prediction).all()
    for prediction in predictions
)

print("official_rgb_student_pretrained_ok=True")