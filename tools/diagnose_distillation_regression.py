from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.losses.detection import PairedDetectionLoss
from rgbt_uav.losses.distillation import FeatureDistiller
from rgbt_uav.models.detectors import (
    StudentRGBTDetector,
    TeacherRGBTDetector,
)
from rgbt_uav.training.distillation_checkpoint import (
    load_distillation_checkpoint,
)

root = Path("D:/Paper/rgbt-uav-thesis/data/processed/yolo_smoke_5pct")
dataset = PairedYoloDataset(root, split="val")
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_paired_batch,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

teacher = TeacherRGBTDetector(
    num_classes=5,
    base_channels=32,
    reg_max=16,
).to(device)

student = StudentRGBTDetector(
    num_classes=5,
    base_channels=16,
    reg_max=16,
).to(device)

distiller = FeatureDistiller(
    student_channels=[64, 128, 256],
    teacher_channels=[128, 256, 512],
).to(device)

optimizer = torch.optim.Adam(
    list(student.parameters()) + list(distiller.parameters()),
    lr=1e-4,
)

load_distillation_checkpoint(
    Path("D:/Paper/rgbt-uav-thesis/runs/paired/distillation_preflight_v3/best.pt"),
    teacher=teacher,
    student=student,
    feature_distiller=distiller,
    optimizer=optimizer,
    map_location=device,
)

batch = next(iter(loader))
rgb = batch["rgb"].to(device)
ir = batch["ir"].to(device)

boxes = [value.to(device) for value in batch["boxes"]]
labels = [value.to(device) for value in batch["labels"]]

student.eval()

with torch.no_grad():
    output = student(rgb, ir)

predictions = output["predictions"]

criterion = PairedDetectionLoss(
    num_classes=5,
    reg_max=16,
)

metrics = criterion(predictions, boxes, labels)

for level_index, level in enumerate(predictions):
    class_logits = level[:, :5]
    box_logits = level[:, 5:9]

    print("level=", level_index)
    print("shape=", tuple(level.shape))
    print("class_min=", float(class_logits.min()))
    print("class_max=", float(class_logits.max()))
    print("class_mean=", float(class_logits.mean()))
    print("score_min=", float(class_logits.sigmoid().min()))
    print("score_max=", float(class_logits.sigmoid().max()))
    print("box_min=", float(box_logits.min()))
    print("box_max=", float(box_logits.max()))
    print("box_mean=", float(box_logits.mean()))

# exit(0)
# print("device=", device)
# print("p3_shape=", tuple(p3.shape))
# print("raw_class_min=", float(raw_class.min()))
# print("raw_class_max=", float(raw_class.max()))
# print("raw_class_mean=", float(raw_class.mean()))
# print("raw_box_min=", float(raw_box.min()))
# print("raw_box_max=", float(raw_box.max()))
# print("raw_box_mean=", float(raw_box.mean()))
# print("sigmoid_box_min=", float(raw_box.sigmoid().min()))
# print("sigmoid_box_max=", float(raw_box.sigmoid().max()))
# print("sigmoid_box_mean=", float(raw_box.sigmoid().mean()))
# print("loss_total=", float(metrics["total"]))
# print("loss_classification=", float(metrics["classification"]))
# print("loss_box=", float(metrics["box"]))
# print("first_target_boxes=", boxes[0][:5])
# print("distillation_regression_diagnosis_ok=True")