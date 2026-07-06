"""Phase 0 Causal HTQ-Transformer skeleton.

This file intentionally does not implement a complete Transformer. It provides
the public shape contract and zero-mean residual composition used by tests and
future implementation work.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HTQConfig:
    """Minimal HTQ shape configuration."""

    context_hours: int = 6
    target_steps: int = 6
    input_channels: int = 2
    output_channels: int = 2


class CausalHTQTransformer:
    """Skeleton model with semantic [B, L, H, C] inputs."""

    def __init__(self, config: HTQConfig | None = None) -> None:
        self.config = config or HTQConfig()

    def __call__(self, x_context):
        return self.forward(x_context)

    def forward(self, x_context):
        """Return current hourly profile repeated across target steps.

        The future implementation will replace the zero residual with decoder
        cross-attention outputs and enforce zero-mean residuals.
        """

        torch = _torch_from_value(x_context)
        if torch is None:
            raise TypeError("x_context must be a torch.Tensor for forward execution")
        if x_context.ndim != 4:
            raise ValueError("x_context must have shape [B, L, H, C]")

        current_hourly = x_context[:, -1:, :, :]
        residual = torch.zeros(
            (
                x_context.shape[0],
                self.config.target_steps,
                x_context.shape[2],
                self.config.output_channels,
            ),
            dtype=x_context.dtype,
            device=x_context.device,
        )
        if current_hourly.shape[-1] != self.config.output_channels:
            raise ValueError("Phase 0 skeleton requires input_channels == output_channels")
        pred = current_hourly.repeat(1, self.config.target_steps, 1, 1) + residual
        return {"pred": pred, "residual": residual}


def _torch_from_value(value):
    module = type(value).__module__.split(".")[0]
    if module != "torch":
        return None
    import torch

    return torch

