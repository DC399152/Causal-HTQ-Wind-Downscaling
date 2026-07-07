"""Multimodal fusion layers for Causal HTQ-Transformer."""

from __future__ import annotations

import torch
from torch import nn


class GatedCrossAttentionFusion(nn.Module):
    """Fuse wind tokens with auxiliary tokens using gated cross-attention."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.1,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_mlp[0].bias, gate_init_bias)

    def forward(
        self,
        wind_tokens: torch.Tensor,
        aux_tokens: torch.Tensor | None,
        aux_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Return fused wind tokens and gate diagnostics.

        wind_tokens: [B, N_w, D]
        aux_tokens: [B, N_aux, D]
        """

        if aux_tokens is None:
            return wind_tokens, {"skipped": True}

        no_valid_aux = None
        if aux_key_padding_mask is not None:
            aux_key_padding_mask = aux_key_padding_mask.to(dtype=torch.bool, device=aux_tokens.device)
            if aux_key_padding_mask.shape != aux_tokens.shape[:2]:
                raise ValueError("aux_key_padding_mask must have shape [B, N_aux]")
            no_valid_aux = aux_key_padding_mask.all(dim=1)
            if bool(no_valid_aux.any()):
                aux_key_padding_mask = aux_key_padding_mask.clone()
                aux_key_padding_mask[no_valid_aux] = False
                aux_tokens = aux_tokens.clone()
                aux_tokens[no_valid_aux] = 0.0

        aux_context, _ = self.cross_attention(
            query=wind_tokens,
            key=aux_tokens,
            value=aux_tokens,
            key_padding_mask=aux_key_padding_mask,
            need_weights=False,
        )
        if no_valid_aux is not None and bool(no_valid_aux.any()):
            aux_context = aux_context.clone()
            aux_context[no_valid_aux] = 0.0
        gate = self.gate_mlp(torch.cat([wind_tokens, aux_context], dim=-1))
        fused_tokens = wind_tokens + gate * aux_context
        fusion_info = {
            "skipped": False,
            "gate_mean": gate.detach().mean(),
            "gate_std": gate.detach().std(unbiased=False),
            "gate_min": gate.detach().min(),
            "gate_max": gate.detach().max(),
        }
        return fused_tokens, fusion_info
