from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ComplexityReport:
    parameters: int
    flops: int
    model_size_mb: float


def profile_complexity(model: nn.Module, inputs: tuple[Tensor, ...]) -> ComplexityReport:
    """Profile common Conv/Linear FLOPs for one forward pass.

    FLOPs use the conventional two operations per multiply-accumulate. Elementwise,
    normalization, activation and indexing operations are intentionally excluded and
    this convention must be kept fixed across all comparison rows.
    """

    flops = 0

    def count_conv(module: nn.Conv2d, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal flops
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        kernel_ops *= module.in_channels // module.groups
        flops += output.numel() * kernel_ops * 2
        if module.bias is not None:
            flops += output.numel()

    def count_linear(module: nn.Linear, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal flops
        flops += output.numel() * module.in_features * 2
        if module.bias is not None:
            flops += output.numel()

    handles: list[Any] = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(count_conv))
        elif isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_hook(count_linear))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    state_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    return ComplexityReport(
        parameters=parameters,
        flops=flops,
        model_size_mb=state_bytes / (1024**2),
    )
