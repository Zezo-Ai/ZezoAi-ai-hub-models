# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import gc
from enum import Enum

import torch
from packaging.version import Version

# Minimum torch version required for dynamic-shape ONNX export (dynamo export).
# Note that earlier versions did support dynamic shapes in general, but did
# not work well for LLMs until 2.10. torch >= 2.11 changes the exported graph
# in ways that break our split LLM pipeline (e.g. llama_v3_2_1b_instruct,
# qwen2_5_vl_7b_instruct), so we pin to 2.10.x.
TORCH_DYNAMIC_SHAPE_MIN_VERSION = "2.10"
TORCH_DYNAMIC_SHAPE_BELOW_VERSION = "2.11"
TORCH_SUPPORTS_DYNAMIC_SHAPE = (
    Version(TORCH_DYNAMIC_SHAPE_MIN_VERSION)
    <= Version(torch.__version__)
    < Version(TORCH_DYNAMIC_SHAPE_BELOW_VERSION)
)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class LLMIOType(Enum):
    # Genie-compatible input (with input token ids)
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    genie_input_ids = "genie_input_ids"

    # Genie-compatible input (with input token embeddings)
    # Inputs:
    # - input_embeds (post-Gather token embeddings)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    genie_input_embeds = "genie_inputs_embeds"

    # Genie-compatible input with native KV cache (right-padding with cache_index)
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    # - past_nativekvcache__key/value per layer
    # - cache_index (int32, shape [1])
    genie_input_ids_native_kv = "genie_input_ids_native_kv"

    # Hugging Face original input
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids (integer position ids)
    huggingface_input_ids = "huggingface_input_ids"


def is_native_kv(llm_io_type: LLMIOType) -> bool:
    """Check if the IO type uses native KV cache (right-padding with cache_index)."""
    return llm_io_type == LLMIOType.genie_input_ids_native_kv


def is_kv_key_name(name: str) -> bool:
    # Covers past_ and nativekvcache_ only; swa_ (sliding window) handled by callers.
    return "past_key" in name or "nativekvcache__key" in name


def is_kv_value_name(name: str) -> bool:
    # Covers past_ and nativekvcache_ only; swa_ (sliding window) handled by callers.
    return "past_value" in name or "nativekvcache__value" in name
