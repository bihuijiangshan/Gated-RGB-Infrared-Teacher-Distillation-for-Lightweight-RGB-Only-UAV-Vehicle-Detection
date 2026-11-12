from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.losses.distillation import DistillationLoss
from rgbt_uav.training.official_distillation_student import (
    build_official_rgb_student,
    load_official_rgb_student_pretrained,
)
from rgbt_uav.training.official_distillation_teacher import (
    build_official_early_fusion_teacher,
)
from rgbt_uav.training.official_student_loss import (
    OfficialStudentDetectionLoss,
)
from rgbt_uav.training.official_teacher_student_step import (
    run_official_teacher_student_step,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

data_root = Path("data/processed/yolo_smoke_5pct")
teacher_checkpoint = Path(
    "runs/paired/early_fusion_pretrained_preflight/best.pt"
)
student_pretrained = Path("yolov8n.pt")

for required_path in (
    teacher_checkpoint,
    student_pretrained,
):
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

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

teacher_payload = torch.load(
    teacher_checkpoint,
    map_location=device,
    weights_only=False,
)

if not isinstance(teacher_payload, dict):
    raise RuntimeError(
        "teacher checkpoint must contain a dictionary"
    )
if "model" not in teacher_payload:
    raise RuntimeError(
        "teacher checkpoint is missing model state"
    )

teacher.model.load_state_dict(
    teacher_payload["model"],
    strict=True,
)
restored_teacher_epoch = int(
    teacher_payload.get("epoch", -1)
)

student = build_official_rgb_student(
    model_name="yolov8n.yaml",
    num_classes=5,
    device=device,
)

pretrained_result = load_official_rgb_student_pretrained(
    student,
    student_pretrained,
)

detection_criterion = OfficialStudentDetectionLoss(
    student
)
distillation_criterion = DistillationLoss(
    feature_weight=1.0,
    logit_weight=1.0,
    temperature=2.0,
)

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=1e-4,
)

teacher_first_parameter = next(
    teacher.parameters()
)
teacher_before = teacher_first_parameter.detach().clone()

student_first_parameter = next(
    student.parameters()
)
student_before = student_first_parameter.detach().clone()

metrics = run_official_teacher_student_step(
    teacher,
    student,
    detection_criterion,
    distillation_criterion,
    optimizer,
    batch,
    device=device,
    num_classes=5,
)

teacher_unchanged = torch.equal(
    teacher_first_parameter.detach(),
    teacher_before,
)
student_changed = not torch.equal(
    student_first_parameter.detach(),
    student_before,
)

print("device=", device)
print("dataset_count=", len(dataset))
print("batch_rgb=", tuple(batch["rgb"].shape))
print("batch_ir=", tuple(batch["ir"].shape))
print(
    "restored_teacher_epoch=",
    restored_teacher_epoch,
)
print(
    "student_pretrained_loaded_keys=",
    len(pretrained_result["loaded_keys"]),
)
print(
    "student_pretrained_missing_keys=",
    len(pretrained_result["missing_keys"]),
)
print("metrics=", metrics)
print("teacher_unchanged=", teacher_unchanged)
print("student_changed=", student_changed)
print(
    "metrics_finite=",
    all(
        torch.isfinite(torch.tensor(value)).item()
        for value in metrics.values()
    ),
)

assert restored_teacher_epoch == 4
assert len(pretrained_result["loaded_keys"]) == 313
assert len(pretrained_result["missing_keys"]) == 66
assert teacher_unchanged
assert student_changed
assert metrics["total"] > 0
assert metrics["detection"] > 0
assert metrics["box"] >= 0
assert metrics["classification"] >= 0
assert metrics["dfl"] >= 0
assert metrics["distillation"] >= 0
assert all(
    torch.isfinite(torch.tensor(value))
    for value in metrics.values()
)

print("official_rgb_distillation_step_ok=True")