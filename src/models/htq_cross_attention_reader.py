"""HTQ encoder with a single query-to-memory cross-attention reader."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.htq_transformer import CausalHTQTransformer, HTQConfig


@dataclass(frozen=True)
class CrossAttentionReaderConfig(HTQConfig):
    """Configuration for :class:`HTQCrossAttentionReader`."""

    architecture: str = "htq_cross_attention_reader"
    name: str = "htq_cross_attention_reader_v1"
    reader_num_layers: int = 1
    reader_nhead: int = 8
    reader_dim_feedforward: int = 512
    reader_dropout: float = 0.1
    reader_activation: str = "gelu"
    reader_final_norm: bool = True


class CrossAttentionReader(nn.Module):
    """Read encoder memory independently for every target query."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or nhead <= 0 or dim_feedforward <= 0:
            raise ValueError("Reader dimensions must be positive")
        if d_model % nhead != 0:
            raise ValueError("Reader d_model must be divisible by nhead")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Reader dropout must be in [0, 1)")
        if activation != "gelu":
            raise ValueError("CrossAttentionReader currently requires activation='gelu'")
        if not final_norm:
            raise ValueError("CrossAttentionReader requires reader_final_norm=True")

        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return final features, post-cross features, and optional attention."""

        if queries.ndim != 3 or memory.ndim != 3:
            raise ValueError("queries and memory must have shape [B, N, D]")
        if queries.shape[0] != memory.shape[0] or queries.shape[2] != memory.shape[2]:
            raise ValueError("queries and memory must share batch and feature dimensions")
        if memory_key_padding_mask is not None:
            expected = memory.shape[:2]
            if tuple(memory_key_padding_mask.shape) != expected:
                raise ValueError(
                    "memory_key_padding_mask must have shape "
                    f"{tuple(expected)}, got {tuple(memory_key_padding_mask.shape)}"
                )

        q = queries
        q_norm = self.query_norm(q)
        memory_norm = self.memory_norm(memory)
        context, attention = self.cross_attention(
            query=q_norm,
            key=memory_norm,
            value=memory_norm,
            key_padding_mask=memory_key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        post_cross = q + self.cross_dropout(context)
        post_ffn = post_cross + self.ffn_dropout(self.ffn(self.ffn_norm(post_cross)))
        output = self.output_norm(post_ffn)
        return output, post_cross, attention


class HTQCrossAttentionReader(CausalHTQTransformer):
    """Causal HTQ model using one cross-attention reader instead of a decoder."""

    def __init__(self, config: CrossAttentionReaderConfig | None = None) -> None:
        resolved = config or CrossAttentionReaderConfig()
        if resolved.architecture != "htq_cross_attention_reader":
            raise ValueError(
                "HTQCrossAttentionReader requires "
                "architecture='htq_cross_attention_reader'"
            )
        if resolved.reader_num_layers != 1:
            raise ValueError("The first Reader version requires reader_num_layers=1")
        super().__init__(resolved)
        self.config = resolved
        del self.decoder
        self.reader = CrossAttentionReader(
            d_model=resolved.d_model,
            nhead=resolved.reader_nhead,
            dim_feedforward=resolved.reader_dim_feedforward,
            dropout=resolved.reader_dropout,
            activation=resolved.reader_activation,
            final_norm=resolved.reader_final_norm,
        )

    def forward(
        self,
        x_hourly: torch.Tensor,
        x_mask: torch.Tensor,
        x_meteo: torch.Tensor | None = None,
        meteo_mask: torch.Tensor | None = None,
        x_static: torch.Tensor | None = None,
        current_hourly_reference: torch.Tensor | None = None,
        height_values: torch.Tensor | None = None,
        return_features: bool = False,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, object] | None]:
        """Return residual wind predictions and optional Reader diagnostics."""

        del height_values  # Kept for the shared training/evaluation interface.
        self._validate_inputs(x_hourly, x_mask)
        batch_size, context_hours, height_levels, _ = x_hourly.shape

        tokenized = self.tokenizer(x_hourly, x_mask)
        if tokenized.token_embeddings is None:
            raise RuntimeError("HTQ tokenizer must be configured with d_model")

        wind_tokens = tokenized.token_embeddings
        fusion_info = None
        aux_tokens_list: list[torch.Tensor] = []
        aux_padding_masks: list[torch.Tensor] = []
        if self.config.use_meteo:
            if x_meteo is None:
                raise ValueError("use_meteo=True but x_meteo is missing")
            if meteo_mask is None:
                raise ValueError("use_meteo=True but meteo_mask is missing")
            if self.meteo_encoder is None:
                raise RuntimeError("Meteo encoder is not initialized")
            meteo_tokens = self.meteo_encoder(x_meteo, meteo_mask)
            aux_tokens_list.append(meteo_tokens)
            meteo_token_valid = meteo_mask.any(dim=-1)
            aux_padding_masks.append(~meteo_token_valid.reshape(batch_size, -1))

        if self.config.use_static:
            if x_static is None:
                raise ValueError("use_static=True but x_static is missing")
            if self.static_encoder is None:
                raise RuntimeError("Static encoder is not initialized")
            static_tokens = self.static_encoder(x_static)
            aux_tokens_list.append(static_tokens)
            aux_padding_masks.append(
                torch.zeros(
                    static_tokens.shape[:2],
                    dtype=torch.bool,
                    device=static_tokens.device,
                )
            )

        if aux_tokens_list:
            if self.fusion is None:
                raise RuntimeError("Multimodal fusion module is not initialized")
            aux_tokens = torch.cat(aux_tokens_list, dim=1)
            aux_key_padding_mask = torch.cat(aux_padding_masks, dim=1)
            encoder_input, fusion_info = self.fusion(
                wind_tokens,
                aux_tokens,
                aux_key_padding_mask,
            )
        else:
            encoder_input = wind_tokens

        token_valid = tokenized.token_valid.reshape(
            batch_size,
            context_hours * height_levels,
        )
        memory_key_padding_mask = ~token_valid
        encoder_memory = self.encoder(
            encoder_input,
            src_key_padding_mask=memory_key_padding_mask,
        )
        target_queries = self.query_builder(
            encoder_memory,
            token_valid=token_valid.reshape(
                batch_size,
                context_hours,
                height_levels,
            ),
            height_levels=height_levels,
        )
        reader_output, reader_post_cross, attention = self.reader(
            target_queries,
            encoder_memory,
            memory_key_padding_mask,
            return_attention=return_attention,
        )
        target_features = reader_output.reshape(
            batch_size,
            self.config.target_steps,
            height_levels,
            self.config.d_model,
        )
        residual = self.residual_head(target_features)
        current_hourly = (
            current_hourly_reference
            if current_hourly_reference is not None
            else x_hourly[:, -1]
        )
        if current_hourly.shape != x_hourly[:, -1].shape:
            raise ValueError(
                "current_hourly_reference must have shape [B, H, C], "
                f"got {tuple(current_hourly.shape)}"
            )
        pred = current_hourly.unsqueeze(1) + residual

        output: dict[str, torch.Tensor | dict[str, object] | None] = {
            "pred": pred,
            "residual": residual,
            "encoder_memory": encoder_memory,
            "fusion_info": fusion_info,
        }
        if return_features or return_attention:
            output["target_queries"] = target_queries
            output["reader_post_cross"] = reader_post_cross
            output["target_features"] = target_features
        if return_attention:
            output["reader_attention_weights"] = attention
        return output
