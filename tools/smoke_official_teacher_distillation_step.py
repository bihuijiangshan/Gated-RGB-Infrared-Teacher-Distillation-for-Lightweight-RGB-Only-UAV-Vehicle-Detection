from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.losses.detection import PairedDetectionLoss
from rgbt_uav.losses.distillation import (
    DistillationLoss,
    FeatureDistiller,
)
from rgbt_uav.models.detectors import StudentRGBTDetector
from rgbt_uav.training.distillation_step import (
    run_distillation_training_step,
)
from rgbt_uav.training.official_distillation_teacher import (
    build_official_early_fusion_teacher,
    load_official_early_fusion_teacher_checkpoint,
)

data_root = Path(
    "data/processed/yolo_smoke_5pct"
)
teacher_checkpoint = Path(
    "runs/paired/"
    "early_fusion_pretrained_preflight/"
    "best.pt"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = PairedYoloDataset(
    data_root,
    split="train",
)
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_paired_batch,
)
batch = next(iter(loader))

teacher = build_official_early_fusion_teacher(
    model_name="yolov8n.yaml",
    num_classes=5,
    device=device,
)

restored = load_official_early_fusion_teacher_checkpoint(
    teacher_checkpoint,
    teacher=teacher,
    map_location=device,
)

student = StudentRGBTDetector(
    num_classes=5,
    base_channels=16,
    reg_max=16,
).to(device)

feature_distiller = FeatureDistiller(
    student_channels=[64, 128, 256],
    teacher_channels=[64, 128, 256],
).to(device)

detection_loss = PairedDetectionLoss(
    num_classes=5,
    reg_max=16,
)

distillation_loss = DistillationLoss(
    feature_weight=1.0,
    logit_weight=1.0,
    temperature=2.0,
)

optimizer = torch.optim.Adam(
    list(student.parameters())
    + list(feature_distiller.parameters()),
    lr=1e-4,
)

teacher_before = {
    name: value.detach().clone()
    for name, value in teacher.state_dict().items()
}

with torch.no_grad():
    teacher_output = teacher(
        batch["rgb"].to(device),
        batch["ir"].to(device),
    )

student.eval()
with torch.no_grad():
    student_output = student(
        batch["rgb"].to(device),
        batch["ir"].to(device),
    )
student.train()

print("device=", device)
print("dataset_count=", len(dataset))
print("restored_teacher_epoch=", restored["epoch"])
print(
    "teacher_features=",
    [
        tuple(value.shape)
        for value in teacher_output["features"]
    ],
)
print(
    "student_features=",
    [
        tuple(value.shape)
        for value in student_output["features"]
    ],
)
print(
    "teacher_predictions=",
    [
        tuple(value.shape)
        for value in teacher_output["predictions"]
    ],
)
print(
    "student_predictions=",
    [
        tuple(value.shape)
        for value in student_output["predictions"]
    ],
)

metrics = run_distillation_training_step(
    teacher,
    student,
    detection_loss,
    distillation_loss,
    feature_distiller,
    optimizer,
    batch,
    device=device,
)

teacher_unchanged = all(
    torch.equal(
        value,
        teacher_before[name],
    )
    for name, value in teacher.state_dict().items()
)

metrics_finite = all(
    torch.isfinite(torch.tensor(value))
    for value in metrics.values()
)

print("metrics=", metrics)
print("teacher_unchanged=", teacher_unchanged)
print("metrics_finite=", metrics_finite)

assert teacher_unchanged
assert metrics_finite
assert len(teacher_output["features"]) == 3
assert len(teacher_output["predictions"]) == 3

for teacher_feature, student_feature in zip(
    teacher_output["features"],
    student_output["features"],
    strict=True,
):
    assert teacher_feature.shape == student_feature.shape

for teacher_prediction, student_prediction in zip(
    teacher_output["predictions"],
    student_output["predictions"],
    strict=True,
):
    assert teacher_prediction.shape == student_prediction.shape
    assert teacher_prediction.shape[1] == 69

print("official_teacher_distillation_step_ok=True")