from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.losses.distillation import FeatureDistiller
from rgbt_uav.models.detectors import StudentRGBTDetector
from rgbt_uav.training.distillation_checkpoint import (
    load_distillation_checkpoint,
)
from rgbt_uav.training.official_distillation_teacher import (
    build_official_early_fusion_teacher,
)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ),
        dim=1,
    )


def aligned_iou(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    intersection_min = torch.maximum(
        first[:, :2],
        second[:, :2],
    )
    intersection_max = torch.minimum(
        first[:, 2:],
        second[:, 2:],
    )
    intersection_size = (
        intersection_max - intersection_min
    ).clamp(min=0)

    intersection = (
        intersection_size[:, 0]
        * intersection_size[:, 1]
    )

    first_size = (
        first[:, 2:] - first[:, :2]
    ).clamp(min=0)
    second_size = (
        second[:, 2:] - second[:, :2]
    ).clamp(min=0)

    first_area = first_size[:, 0] * first_size[:, 1]
    second_area = second_size[:, 0] * second_size[:, 1]
    union = first_area + second_area - intersection

    return intersection / union.clamp(min=1e-9)


data_root = Path("data/processed/yolo_smoke_5pct")
checkpoint = Path(
    "runs/paired/distillation_detection_only_v1/best.pt"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = PairedYoloDataset(
    data_root,
    split="val",
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
    list(student.parameters())
    + list(feature_distiller.parameters()),
    lr=1e-4,
)

restored = load_distillation_checkpoint(
    checkpoint,
    teacher=teacher,
    student=student,
    feature_distiller=feature_distiller,
    optimizer=optimizer,
    map_location=device,
)

student.eval()

with torch.no_grad():
    output = student(
        batch["rgb"].to(device),
        batch["ir"].to(device),
    )

target_boxes = batch["boxes"][0].to(device)
target_labels = batch["labels"][0].to(device)
target_xyxy = xywh_to_xyxy(target_boxes)

all_scores = []
records = []

for level_index, prediction in enumerate(
    output["predictions"]
):
    class_scores = prediction[0, :5].sigmoid()
    all_scores.append(class_scores.flatten())

    _, grid_height, grid_width = class_scores.shape

    predicted_boxes = []
    positive_scores = []

    for target_index, (box, label) in enumerate(
        zip(target_boxes, target_labels, strict=True)
    ):
        cell_x = min(
            grid_width - 1,
            max(0, int(float(box[0]) * grid_width)),
        )
        cell_y = min(
            grid_height - 1,
            max(0, int(float(box[1]) * grid_height)),
        )

        decoded = prediction[
            0,
            5:9,
            cell_y,
            cell_x,
        ].sigmoid()

        decoded = decoded.clone()
        decoded[0] = (
            cell_x + decoded[0]
        ) / grid_width
        decoded[1] = (
            cell_y + decoded[1]
        ) / grid_height

        predicted_boxes.append(decoded)

        score = class_scores[
            int(label),
            cell_y,
            cell_x,
        ]
        positive_scores.append(score)

        records.append(
            {
                "level": level_index,
                "target": target_index,
                "score": score,
            }
        )

    predicted_xywh = torch.stack(predicted_boxes)
    ious = aligned_iou(
        xywh_to_xyxy(predicted_xywh),
        target_xyxy,
    )
    scores = torch.stack(positive_scores)

    print("level=", level_index)
    print("positive_score_mean=", float(scores.mean()))
    print("positive_score_min=", float(scores.min()))
    print("positive_score_max=", float(scores.max()))
    print("positive_iou_mean=", float(ious.mean()))
    print("positive_iou_max=", float(ious.max()))
    print(
        "positive_iou_ge_050=",
        int((ious >= 0.5).sum()),
    )
    print(
        "predicted_wh_mean=",
        predicted_xywh[:, 2:].mean(dim=0).tolist(),
    )

candidate_scores = torch.cat(all_scores)
sorted_scores = torch.sort(
    candidate_scores,
    descending=True,
).values

ranks = torch.tensor(
    [
        int((sorted_scores > record["score"]).sum()) + 1
        for record in records
    ]
)

print("device=", device)
print("restored_epoch=", restored["epoch"])
print("pair_id=", batch["pair_ids"][0])
print("targets=", len(target_boxes))
print("all_candidates=", len(candidate_scores))
print("top_100_score=", float(sorted_scores[99]))
print("top_300_score=", float(sorted_scores[299]))
print("positive_rank_min=", int(ranks.min()))
print(
    "positive_rank_median=",
    int(ranks.float().median()),
)
print("positive_rank_max=", int(ranks.max()))
print(
    "positive_candidates_top_100=",
    int((ranks <= 100).sum()),
)
print(
    "positive_candidates_top_300=",
    int((ranks <= 300).sum()),
)
print("official_distilled_student_diagnosis_ok=True")