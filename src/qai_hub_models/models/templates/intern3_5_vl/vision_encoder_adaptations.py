# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import nn
from transformers.models.internvl.modeling_internvl import (
    InternVLVisionAttention,
    InternVLVisionLayer,
)


class InternVLVisionAttentionAdaptation(nn.Module):
    """Adapter around InternVL attention that accepts legacy extra kwargs."""

    def __init__(self, visual: InternVLVisionAttention) -> None:
        super().__init__()
        self.attn = visual

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # InternVL attention does not consume cu_seqlens/rope inputs.
        del cu_seqlens, rotary_pos_emb, position_embeddings
        attn_output = self.attn(hidden_states, attention_mask=attention_mask)
        if isinstance(attn_output, tuple):
            return attn_output[0]
        return attn_output


class InternVLVisionLayerAdaptation(nn.Module):
    """Adapt InternVL vision layer to the Qwen-style block call signature."""

    def __init__(self, block: InternVLVisionLayer) -> None:
        super().__init__()
        self.layernorm_before = block.layernorm_before
        self.layernorm_after = block.layernorm_after
        self.attention = block.attention
        self.mlp = block.mlp
        self.lambda_1 = block.lambda_1
        self.lambda_2 = block.lambda_2
        self.dropout = block.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cu_seqlens, rotary_pos_emb, position_embeddings
        attn_output = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask=attention_mask,
        )
        if isinstance(attn_output, tuple):
            attention_output = attn_output[0]
        else:
            attention_output = attn_output

        attention_output = self.lambda_1 * attention_output
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.dropout(layer_output)
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output
        return layer_output + hidden_states


# Utility functions for replacing modules


def _rsetattr(obj: Any, attr: str, val: Any) -> None:
    pre, _, post = attr.rpartition(".")
    setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def replace_visual_attention_with_adaptation(
    model: nn.Module,
) -> nn.Module:
    """Replace InternVL vision layer/attention modules with adapted versions."""
    for name, module in model.named_modules():
        if isinstance(module, InternVLVisionLayer):
            _rsetattr(model, name, InternVLVisionLayerAdaptation(module))

    for name, module in model.named_modules():
        if isinstance(module, InternVLVisionAttention):
            _rsetattr(model, name, InternVLVisionAttentionAdaptation(module))

    return model
