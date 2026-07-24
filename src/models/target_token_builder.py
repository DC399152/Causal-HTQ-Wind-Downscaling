"""Target time-height token construction for the encoder-only HTQ model."""

from __future__ import annotations

import torch
from torch import nn


class TargetTokenBuilder(nn.Module):
    """Build target tokens ordered as time-major then height-major."""

    def __init__(
        self,
        d_model: int,
        target_steps: int,
        height_levels: int,
        condition_on_current_height: bool = True,
        context_gate_init_bias: float = -1.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or target_steps <= 0 or height_levels <= 0:
            raise ValueError("d_model, target_steps, and height_levels must be positive")

        self.d_model = d_model
        self.target_steps = target_steps
        self.height_levels = height_levels
        self.condition_on_current_height = condition_on_current_height
        self.target_time_embedding = nn.Embedding(target_steps, d_model)
        self.target_type_embedding = nn.Embedding(1, d_model)

        if condition_on_current_height:
            self.context_norm = nn.LayerNorm(d_model)
            self.context_projection = nn.Linear(d_model, d_model)
            self.context_gate_logit = nn.Parameter(torch.tensor(float(context_gate_init_bias)))
        else:
            self.context_norm = None
            self.context_projection = None
            self.register_parameter("context_gate_logit", None)

    def forward(
        self,
        physical_height_embeddings: torch.Tensor,
        current_height_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return target tokens with shape ``[B, T*H, D]``."""

        if physical_height_embeddings.ndim != 3:
            raise ValueError("physical_height_embeddings must have shape [B, H, D]")
        batch_size, height_levels, d_model = physical_height_embeddings.shape
        if height_levels != self.height_levels or d_model != self.d_model:
            raise ValueError(
                "physical_height_embeddings shape does not match configured height_levels/d_model"
            )

        time_ids = torch.arange(self.target_steps, device=physical_height_embeddings.device)
        time_embedding = self.target_time_embedding(time_ids).view(1, self.target_steps, 1, self.d_model)
        height_embedding = physical_height_embeddings[:, None, :, :]
        type_embedding = self.target_type_embedding.weight.view(1, 1, 1, self.d_model)
        tokens = time_embedding + height_embedding + type_embedding

        if self.condition_on_current_height:
            if current_height_context is None:
                raise ValueError(
                    "current_height_context is required when condition_on_current_height=True"
                )
            if current_height_context.shape != physical_height_embeddings.shape:
                raise ValueError("current_height_context must have shape [B, H, D]")
            if self.context_norm is None or self.context_projection is None:
                raise RuntimeError("Target context modules are not initialized")
            context = self.context_projection(self.context_norm(current_height_context))
            gate = torch.sigmoid(self.context_gate_logit)
            tokens = tokens + gate * context[:, None, :, :]

        return tokens.reshape(batch_size, self.target_steps * self.height_levels, self.d_model)
