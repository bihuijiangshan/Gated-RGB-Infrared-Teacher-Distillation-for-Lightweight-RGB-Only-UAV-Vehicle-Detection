from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.models.detectors import TeacherRGBTDetector
from rgbt_uav.training.teacher_checkpoint import (
    load_teacher_training_checkpoint,
)

root = Path("data/processed/yolo_smoke_5pct")
checkpoint = Path(
    "runs/paired/teacher_ohem_v4/teacher-best.pt"
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

model = TeacherRGBTDetector(
    num_classes=5,
    base_channels=32,
    reg_max=16,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

restored = load_teacher_training_checkpoint(
    checkpoint,
    model=model,
    optimizer=optimizer,
    map_location=device,
)

batch = next(iter(loader))
rgb = batch["rgb"].to(device)
ir = batch["ir"].to(device)
boxes = batch["boxes"][0].to(device)
labels = batch["labels"][0].to(device)

model.eval()

with torch.no_grad():
    output = model(rgb, ir)

all_candidate_scores = []
positive_records = []

for level_index, prediction in enumerate(
    output["predictions"]
):
    class_scores = prediction[
        0,
        :5,
    ].sigmoid()

    flattened_scores = class_scores.flatten()
    all_candidate_scores.append(flattened_scores)

    _, grid_height, grid_width = class_scores.shape

    for target_index, (box, label) in enumerate(
        zip(boxes, labels, strict=True)
    ):
        cell_x = min(
            grid_width - 1,
            max(0, int(float(box[0]) * grid_width)),
        )
        cell_y = min(
            grid_height - 1,
            max(0, int(float(box[1]) * grid_height)),
        )

        positive_score = class_scores[
            int(label),
            cell_y,
            cell_x,
        ]

        positive_records.append(
            {
                "target": target_index,
                "level": level_index,
                "score": positive_score,
            }
        )

all_scores = torch.cat(all_candidate_scores)
sorted_scores = torch.sort(
    all_scores,
    descending=True,
).values

positive_ranks = []

for record in positive_records:
    score = record["score"]
    rank = int((sorted_scores > score).sum()) + 1
    positive_ranks.append(rank)

ranks = torch.tensor(positive_ranks)
positive_scores = torch.stack(
    [record["score"] for record in positive_records]
)

print("device=", device)
print("restored_epoch=", restored["epoch"])
print("pair_id=", batch["pair_ids"][0])
print("targets=", len(boxes))
print("all_candidates=", len(all_scores))
print("top_1_score=", float(sorted_scores[0]))
print("top_100_score=", float(sorted_scores[99]))
print("top_300_score=", float(sorted_scores[299]))
print("positive_score_mean=", float(positive_scores.mean()))
print("positive_score_min=", float(positive_scores.min()))
print("positive_score_max=", float(positive_scores.max()))
print("positive_rank_min=", int(ranks.min()))
print("positive_rank_median=", int(ranks.float().median()))
print("positive_rank_max=", int(ranks.max()))
print("positive_candidates_top_100=", int((ranks <= 100).sum()))
print("positive_candidates_top_300=", int((ranks <= 300).sum()))

for level_index in range(3):
    level_ranks = torch.tensor(
        [
            positive_ranks[index]
            for index, record in enumerate(positive_records)
            if record["level"] == level_index
        ]
    )

    print(
        "level=",
        level_index,
        "top_100=",
        int((level_ranks <= 100).sum()),
        "median_rank=",
        int(level_ranks.float().median()),
    )

print("teacher_candidate_ranking_diagnosis_ok=True")