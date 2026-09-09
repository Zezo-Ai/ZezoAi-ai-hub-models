# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Generator support for native KV cache (ScatterElements-based) models.

This lives downstream of the vendored ``lm_driver`` driver on purpose: it is an
export-shape concern, so GenAI Lab only provides the generic hooks it plugs into.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, cast

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from qai_hub_models.models.templates.llm.native_kv_cache import NativeKVCache
from qai_hub_models.models.templates.lm_driver.generator import (
    Generator,
    PrecomputedCosSinGeneratorMixin,
    _flatten_past_key_values,
)
from qai_hub_models.models.templates.lm_driver.utils.layer_cache import (
    build_layer_cache_descriptors,
)


class NativeKVGeneratorMixin:
    """Generator mixin for native KV cache (ScatterElements-based) models.

    Key differences from the standard left-pad + concat path:
    - Right-pads input tokens (real tokens first, then padding)
    - Passes full-size KV buffers directly (no shift/concat)
    - Builds causal mask manually for the fixed-size buffer
    - Appends cache_index tensor to inputs
    - Scatters new KV values into buffers at cache_index in output handling
    """

    # Supplied by Generator, which this mixin is always composed with.
    context_length: int
    device: torch.device
    layer_cache_descriptors: list

    @classmethod
    def prepare_inputs(
        cls,
        model: torch.nn.Module,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        past_key_values: list[torch.Tensor],
        sequence_length: int,
        context_length: int,
        pad_token: int = 0,
        attention_mask_min: int = -100,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.Tensor | None = None,
        layer_cache_descriptors: list | None = None,
        cache_index: int = 0,
        **kwargs: Any,
    ) -> OrderedDict[str, torch.Tensor]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if layer_cache_descriptors is None:
            if not hasattr(model, "config"):
                raise ValueError(
                    "Model config is required to build layer cache descriptors."
                )
            layer_cache_descriptors = build_layer_cache_descriptors(
                cast(PretrainedConfig, model.config)
            )

        input_tokens = input_ids if input_ids is not None else inputs_embeds
        assert input_tokens is not None  # guaranteed by the XOR check above
        input_tokens = input_tokens.to(
            dtype=torch.int32 if input_ids is not None else torch.float32
        )

        device = input_tokens.device
        batch_size = input_tokens.shape[0]
        input_length = input_tokens.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, input_length), dtype=torch.int32, device=device
            )

        # Right-pad input tokens to sequence_length
        if input_ids is not None:
            input_tokens_extension = torch.full(
                (batch_size, sequence_length - input_length),
                fill_value=pad_token,
                dtype=input_tokens.dtype,
                device=device,
            )
        else:
            embedding_dim = input_tokens.shape[2]
            input_tokens_extension = torch.zeros(
                (batch_size, sequence_length - input_length, embedding_dim),
                dtype=input_tokens.dtype,
                device=device,
            )

        padded_input_tokens = torch.cat((input_tokens, input_tokens_extension), dim=1)

        # Build attention mask: [cached positions | current tokens | unfilled]
        padded_attention_mask = torch.cat(
            (
                attention_mask,
                torch.zeros(
                    (batch_size, sequence_length - input_length),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ),
            dim=-1,
        )

        # KV buffers: pass directly (already full context_length size)
        # Native KV layout matches the ONNX input spec:
        #   key:   (num_kv_heads, 1, head_dim, context_length)
        #   value: (num_kv_heads, 1, context_length, head_dim)
        if cache_index > 0 and not past_key_values:
            raise ValueError(
                f"cache_index={cache_index} but no KV cache was passed. Falling back "
                "to zero buffers would silently drop all cached context."
            )
        if past_key_values and len(past_key_values) > 0:
            padded_past_key_values = [kv.to(device) for kv in past_key_values]
        else:
            padded_past_key_values = NativeKVCache.allocate(
                [(d.num_kv_heads, d.head_dim) for d in layer_cache_descriptors],
                context_length,
                device=device,
            ).to_flat_buffers()

        # The RoPE table has exactly context_length entries; padded positions
        # past the end are fully masked, so clamping them is safe.
        if position_ids is None:
            position_ids = (
                torch.arange(
                    cache_index,
                    cache_index + sequence_length,
                    dtype=torch.int32,
                    device=device,
                )
                .clamp(max=context_length - 1)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        else:
            padding_length = sequence_length - position_ids.shape[-1]
            if padding_length > 0:
                pad_shape = list(position_ids.shape)
                pad_shape[-1] = padding_length
                position_ids_padding = torch.zeros(
                    pad_shape, dtype=position_ids.dtype, device=device
                )
                position_ids = torch.cat((position_ids, position_ids_padding), dim=-1)

        # Build 4D causal mask for native KV
        # q at position p attends to k at position k iff k <= p and k is valid
        query_pos = torch.arange(sequence_length, device=device).view(1, 1, -1, 1)
        key_pos = torch.arange(context_length, device=device).view(1, 1, 1, -1)
        causal_mask = key_pos <= (cache_index + query_pos)

        if cache_index + input_length > context_length:
            raise ValueError(
                f"Context length exhausted: cache_index={cache_index} plus "
                f"{input_length} new token(s) exceeds context_length={context_length}. "
                "The native KV buffer is fixed-size and cannot slide."
            )

        # Valid mask: positions < cache_index are cached (valid), plus current tokens
        pre_cache_valid = torch.ones((1, 1, 1, cache_index), device=device)
        current_valid = padded_attention_mask.view(batch_size, 1, 1, -1)
        post_valid = torch.zeros(
            (1, 1, 1, max(0, context_length - cache_index - sequence_length)),
            device=device,
        )
        # The key axis is exactly context_length wide. Once cache_index plus the
        # padded sequence_length overruns it, the trailing slots are padding only.
        valid_mask = torch.cat((pre_cache_valid, current_valid, post_valid), dim=-1)
        valid_mask = valid_mask.narrow(-1, 0, context_length) > 0

        attend_mask = causal_mask & valid_mask
        cm_attention_mask = torch.zeros(
            (batch_size, 1, sequence_length, context_length),
            dtype=torch.float32,
            device=device,
        )
        cm_attention_mask.masked_fill_(~attend_mask, float("-inf"))
        cm_attention_mask = cm_attention_mask.clip(attention_mask_min, 0)

        # Build ordered dict
        input_key = "inputs_embeds" if input_ids is None else "input_ids"
        prepared = OrderedDict()
        prepared[input_key] = padded_input_tokens
        prepared["attention_mask"] = cm_attention_mask.to(
            dtype=cast(torch.dtype, model.dtype)
        )
        prepared["position_ids"] = position_ids
        for i, desc in enumerate(layer_cache_descriptors):
            li = desc.layer_idx
            prepared[f"past_nativekvcache__key_{li}_in"] = padded_past_key_values[i * 2]
            prepared[f"past_nativekvcache__value_{li}_in"] = padded_past_key_values[
                i * 2 + 1
            ]

        prepared["cache_index"] = torch.tensor(
            [cache_index], dtype=torch.int32, device=device
        )

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v

        return prepared

    def _output_names_from_descriptors(
        self,
        layer_cache_descriptors: list,
    ) -> list[str]:
        names = ["logits"]
        for desc in layer_cache_descriptors:
            i = desc.layer_idx
            names.append(f"past_nativekvcache__key_{i}_out")
            names.append(f"past_nativekvcache__value_{i}_out")
        return names

    def combine_local_and_global_outputs(
        self,
        num_valid_input_tokens: int,
        local_outputs: OrderedDict[str, torch.Tensor],
        global_outputs: dict[str, Any],
    ) -> None:
        # Strip logits from the left (right-padded, so valid tokens are first)
        local_logits = local_outputs["logits"]
        local_logits = torch.narrow(local_logits, 1, 0, num_valid_input_tokens)

        global_outputs["logits"] = (
            torch.cat((global_outputs["logits"], local_logits), dim=1)
            if "logits" in global_outputs
            else local_logits
        )

        cache = global_outputs["native_kv_cache"]
        assert isinstance(cache, NativeKVCache)

        # The graph emits a full sequence_length of KV; trim the right padding
        # before handing the states to the cache.
        seq_len = num_valid_input_tokens
        local_kv_list = [v for k, v in local_outputs.items() if k != "logits"]
        trimmed: list[torch.Tensor] = []
        for i in range(0, len(local_kv_list), 2):
            new_k = local_kv_list[i]
            new_v = local_kv_list[i + 1]
            if new_k.shape[3] > seq_len:
                new_k = new_k.narrow(3, 0, seq_len)
            if new_v.shape[2] > seq_len:
                new_v = new_v.narrow(2, 0, seq_len)
            trimmed += [new_k, new_v]

        cache.append(trimmed)
        global_outputs["past_key_values"] = cache.to_flat_buffers()

    def _init_global_outputs(
        self, past_key_values: DynamicCache | None
    ) -> dict[str, Any]:
        if isinstance(past_key_values, NativeKVCache):
            cache = past_key_values
        else:
            buffers = _flatten_past_key_values(past_key_values)
            if buffers:
                assert past_key_values is not None  # non-empty implies a real cache
                cache = NativeKVCache.from_buffers(
                    buffers,
                    self.context_length,
                    past_key_values.get_seq_length(),
                )
            else:
                cache = NativeKVCache.allocate(
                    [
                        (d.num_kv_heads, d.head_dim)
                        for d in self.layer_cache_descriptors
                    ],
                    self.context_length,
                    device=self.device,
                )
        return {
            "native_kv_cache": cache,
            "past_key_values": cache.to_flat_buffers(),
        }

    def _extra_prepare_kwargs(self, global_outputs: dict) -> dict[str, Any]:
        cache = global_outputs["native_kv_cache"]
        assert isinstance(cache, NativeKVCache)
        return {"cache_index": cache.get_seq_length()}

    def _wrap_kv_cache(
        self,
        past_key_values_list: list[torch.Tensor],
        global_outputs: dict,
    ) -> NativeKVCache:
        cache = global_outputs["native_kv_cache"]
        assert isinstance(cache, NativeKVCache)
        cache.adopt_flat_buffers(past_key_values_list)
        return cache


# mypy flags prepare_inputs as incompatible across the bases: the driver's mixins
# declare it as (cls, **kwargs) while Generator spells the arguments out.
class HubCompatibleNativeKVGenerator(  # type: ignore[misc]
    PrecomputedCosSinGeneratorMixin, NativeKVGeneratorMixin, Generator
):
    """Generator for native KV cache models.

    Composes PrecomputedCosSinGeneratorMixin (RoPE cos/sin) with
    NativeKVGeneratorMixin (scatter-based KV cache) and the base Generator.
    """
