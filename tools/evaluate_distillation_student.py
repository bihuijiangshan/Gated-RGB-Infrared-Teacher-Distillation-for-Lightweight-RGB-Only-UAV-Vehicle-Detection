from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import PairedYoloDataset, collate_paired_batch
from rgbt_uav.evaluation.distillation import (
    run_distillation_student_inference,
)
from rgbt_uav.evaluation.paired import (
    evaluate_paired_coco,
    paired_samples_to_coco_ground_truth,
)
from rgbt_uav.losses.distillation import FeatureDistiller
from rgbt_uav.models.detectors import StudentRGBTDetector
from rgbt_uav.training.distillation_checkpoint import (
    load_distillation_checkpoint,
)
from rgbt_uav.training.official_distillation_teacher import (
    build_official_early_fusion_teacher,
)

ROOT = Path("D:/Paper/rgbt-uav-thesis/data/processed/yolo_smoke_5pct")
CHECKPOINT = Path(
    "runs/paired/distillation_detection_only_v1/best.pt"
)
OUTPUT = Path(
    "D:/Paper/rgbt-uav-thesis/runs/paired/distillation_preflight_v2/"
    "paired_coco_metrics.json"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = PairedYoloDataset(ROOT, split="val")
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_paired_batch,
)

teacher = build_official_early_fusion_teacher(
    model_name="yolov8n.yaml",
    num_classes=5,
    device=device,
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

optimizer = torch.optim.Adam(
    list(student.parameters()) + list(feature_distiller.parameters()),
    lr=1e-4,
)

restored = load_distillation_checkpoint(
    CHECKPOINT,
    teacher=teacher,
    student=student,
    feature_distiller=feature_distiller,
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

predictions = [
    {
        "pair_id": item["pair_id"],
        "boxes": [
            [float(value) for value in box]
            for box in item["boxes"].detach().cpu().tolist()
        ],
        "scores": [
            float(value)
            for value in item["scores"].detach().cpu().tolist()
        ],
        "labels": [
            int(value)
            for value in item["labels"].detach().cpu().tolist()
        ],
    }
    for item in predictions
]

ground_truth = paired_samples_to_coco_ground_truth(
    dataset,
    label_to_category_id={
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
    },
    category_names={
        1: "car",
        2: "truck",
        3: "bus",
        4: "van",
        5: "freight_car",
    },
)

metrics = evaluate_paired_coco(
    dataset,
    predictions,
    label_to_category_id={
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
    },
    category_names={
        1: "car",
        2: "truck",
        3: "bus",
        4: "van",
        5: "freight_car",
    },
)

payload = {
    "device": str(device),
    "checkpoint": str(CHECKPOINT),
    "restored_epoch": restored["epoch"],
    "images": len(dataset),
    "prediction_boxes": sum(
        len(item["boxes"])
        for item in predictions
    ),
    "mAP50-95": metrics.ap,
    "mAP50": metrics.ap50,
    "AP75": metrics.ap75,
    "AP_small": metrics.ap_small,
    "AR_small": metrics.ar_small,
    "per_class_AP_small": metrics.per_class_ap_small,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(json.dumps(payload, indent=2, ensure_ascii=False))
print("distillation_student_coco_evaluation_ok=True")