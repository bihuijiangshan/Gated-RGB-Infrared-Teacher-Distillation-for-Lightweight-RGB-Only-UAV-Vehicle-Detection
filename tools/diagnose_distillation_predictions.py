from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.evaluation.distillation import (
    run_distillation_student_inference,
)
from rgbt_uav.losses.distillation import FeatureDistiller
from rgbt_uav.models.detectors import (
    StudentRGBTDetector,
    TeacherRGBTDetector,
)
from rgbt_uav.training.distillation_checkpoint import (
    load_distillation_checkpoint,
)

root = Path(
    r"D:\Paper\rgbt-uav-thesis\data\processed\yolo_smoke_5pct"
)
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
    Path(r"D:/Paper/rgbt-uav-thesis/runs/paired/distillation_preflight_v2/best.pt"),
    teacher=teacher,
    student=student,
    feature_distiller=distiller,
    optimizer=optimizer,
    map_location=device,
)

predictions = run_distillation_student_inference(
    student,
    loader,
    device=device,
    num_classes=5,
    reg_max=16,
    score_threshold=0.001,
)

sample = dataset[0]
first = predictions[0]

print("device=", device)
print("pair_id=", sample["pair_id"])
print("image_shape=", tuple(sample["rgb"].shape))
print("gt_min=", float(sample["boxes"].min()))
print("gt_max=", float(sample["boxes"].max()))
print("gt_first=", sample["boxes"][:5])
print("gt_labels=", sample["labels"][:20])
print("prediction_count=", len(first["boxes"]))

if len(first["boxes"]) > 0:
    print("pred_box_min=", float(first["boxes"].min()))
    print("pred_box_max=", float(first["boxes"].max()))
    print("pred_score_min=", float(first["scores"].min()))
    print("pred_score_max=", float(first["scores"].max()))
    print("pred_score_mean=", float(first["scores"].mean()))
    print("pred_first=", first["boxes"][:5])
    print("pred_labels=", first["labels"][:20])
else:
    print("pred_box_min=n/a")
    print("pred_box_max=n/a")
    print("pred_score_min=n/a")
    print("pred_score_max=n/a")