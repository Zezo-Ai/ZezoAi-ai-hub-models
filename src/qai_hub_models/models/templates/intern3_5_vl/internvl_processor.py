# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.models.internvl.processing_internvl import InternVLProcessor
from transformers.models.internvl.video_processing_internvl import (
    InternVLVideoProcessor,
)

from qai_hub_models.models.templates.vlm.processor_factory import VLMProcessorLike


def get_internvl_compatible_processor(hf_repo: str) -> VLMProcessorLike:
    """Return a processor that is compatible with InternVL."""
    cfg = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
    if getattr(cfg, "model_type", "") != "internvl_chat":
        return cast(
            VLMProcessorLike,
            AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True),
        )

    img_context_token = "<IMG_CONTEXT>"
    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo, use_fast=True, trust_remote_code=True
    )
    tokenizer.start_image_token = "<img>"
    tokenizer.end_image_token = "</img>"
    tokenizer.context_image_token = img_context_token
    tokenizer.start_image_token_id = tokenizer.convert_tokens_to_ids("<img>")
    tokenizer.end_image_token_id = tokenizer.convert_tokens_to_ids("</img>")
    tokenizer.context_image_token_id = tokenizer.convert_tokens_to_ids(
        img_context_token
    )
    tokenizer.video_token = "<video>"

    chat_template = tokenizer.chat_template
    if chat_template is not None:
        chat_template = chat_template.replace("'<image>", f"'{img_context_token}")

    image_processor = AutoImageProcessor.from_pretrained(
        hf_repo, trust_remote_code=True
    )
    return cast(
        VLMProcessorLike,
        InternVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=InternVLVideoProcessor(),
            chat_template=chat_template,
        ),
    )
