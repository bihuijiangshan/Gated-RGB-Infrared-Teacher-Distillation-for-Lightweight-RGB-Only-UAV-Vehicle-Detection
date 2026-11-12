"""Run one real CUDA step for the official gated teacher."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rgbt_uav.data.paired import (
    PairedYoloDataset,
    collate_paired_batch,
)
from rgbt_uav.models.official_gated_teacher import (
    build_official_gated_teacher,
)
from rgbt_uav.training.official_gated_teacher_loss import (
    OfficialGatedTeacherDetectionLoss,
)
from rgbt_uav.training.official_gated_teacher_pretrained import (
    load_official_gated_teacher_pretrained,
)
from rgbt_uav.training.official_gated_teacher_step import (
    run_official_gated_teacher_step,
)


def _changed(
    before: torch.Tensor,
    after: torch.Tensor,
) -> bool:
    return not torch.equal(
        before.cpu(),
        after.detach().cpu(),
    )


def _finite_nonzero_gradient(
    parameters: list[torch.nn.Parameter],
) -> bool:
    return any(
        parameter.grad is not None
        and torch.isfinite(
            parameter.grad
        ).all()
        and parameter.grad.abs().sum() > 0
        for parameter in parameters
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this smoke test"
        )

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    dataset = PairedYoloDataset(
        Path(
            "data/processed/yolo_smoke_5pct"
        ),
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_paired_batch,
    )
    batch = next(iter(loader))

    teacher = build_official_gated_teacher(
        model_name="yolov8n.yaml",
        num_classes=5,
        device=device,
    )
    pretrained = (
        load_official_gated_teacher_pretrained(
            teacher,
            Path("yolov8n.pt"),
        )
    )

    criterion = (
        OfficialGatedTeacherDetectionLoss(
            teacher
        )
    )
    optimizer = torch.optim.AdamW(
        teacher.parameters(),
        lr=1e-4,
        weight_decay=5e-4,
    )

    rgb_parameters = list(
        teacher.rgb_stream.model[0]
        .conv.parameters()
    )
    ir_parameters = list(
        teacher.ir_stream.model[0]
        .conv.parameters()
    )
    gate_parameters = list(
        teacher.fusions.parameters()
    )

    rgb_before = (
        rgb_parameters[0]
        .detach()
        .clone()
    )
    ir_before = (
        ir_parameters[0]
        .detach()
        .clone()
    )
    gate_before = (
        gate_parameters[0]
        .detach()
        .clone()
    )

    metrics = run_official_gated_teacher_step(
        teacher,
        criterion,
        optimizer,
        batch,
        device=device,
    )

    torch.cuda.synchronize()

    rgb_changed = _changed(
        rgb_before,
        rgb_parameters[0],
    )
    ir_changed = _changed(
        ir_before,
        ir_parameters[0],
    )
    gate_changed = _changed(
        gate_before,
        gate_parameters[0],
    )

    rgb_gradient_ok = (
        _finite_nonzero_gradient(
            rgb_parameters
        )
    )
    ir_gradient_ok = (
        _finite_nonzero_gradient(
            ir_parameters
        )
    )
    gate_gradient_ok = (
        _finite_nonzero_gradient(
            gate_parameters
        )
    )

    peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

    print("device=", device)
    print("dataset_count=", len(dataset))
    print(
        "batch_rgb=",
        tuple(batch["rgb"].shape),
    )
    print(
        "batch_ir=",
        tuple(batch["ir"].shape),
    )
    print(
        "rgb_loaded_keys=",
        len(pretrained["rgb_loaded_keys"]),
    )
    print(
        "ir_loaded_keys=",
        len(pretrained["ir_loaded_keys"]),
    )
    print(
        "detect_loaded_keys=",
        len(pretrained["detect_loaded_keys"]),
    )
    print("metrics=", metrics)

    for level in ("p3", "p4", "p5"):
        gate_sum = (
            metrics[f"gate_rgb_{level}"]
            + metrics[f"gate_ir_{level}"]
        )
        print(
            f"{level}_gate_sum=",
            gate_sum,
        )

    print("rgb_changed=", rgb_changed)
    print("ir_changed=", ir_changed)
    print("gate_changed=", gate_changed)
    print(
        "rgb_gradient_ok=",
        rgb_gradient_ok,
    )
    print(
        "ir_gradient_ok=",
        ir_gradient_ok,
    )
    print(
        "gate_gradient_ok=",
        gate_gradient_ok,
    )
    print(
        "peak_memory_mib=",
        peak_memory_mib,
    )

    assert all(
        torch.isfinite(
            torch.tensor(value)
        )
        for value in metrics.values()
    )
    assert metrics["total"] > 0

    for level in ("p3", "p4", "p5"):
        assert abs(
            metrics[f"gate_rgb_{level}"]
            + metrics[f"gate_ir_{level}"]
            - 1.0
        ) < 1e-5

    assert rgb_changed
    assert ir_changed
    assert gate_changed
    assert rgb_gradient_ok
    assert ir_gradient_ok
    assert gate_gradient_ok

    print(
        "official_gated_teacher_step_ok=True"
    )


if __name__ == "__main__":
    main()