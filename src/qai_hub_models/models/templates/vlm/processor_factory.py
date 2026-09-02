# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Protocol, cast


class VLMProcessorLike(Protocol):
    """Structural type for VLM processors: callable with a ``tokenizer`` attribute."""

    tokenizer: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def get_default_vlm_processor(hf_repo_name: str) -> VLMProcessorLike:
    """Load the default processor for VLM families."""
    from transformers import AutoProcessor

    return cast(
        VLMProcessorLike,
        AutoProcessor.from_pretrained(hf_repo_name, trust_remote_code=True),
    )


class VLMProcessorHooksMixin:
    """Shared processor/prompt hooks for VLM families."""

    @classmethod
    def get_processor(cls, hf_repo_name: str) -> VLMProcessorLike:
        return get_default_vlm_processor(hf_repo_name)

    @classmethod
    def adapt_prompt_for_processor(
        cls,
        formatted_text: str,
        processor: VLMProcessorLike,
        num_images: int,
    ) -> str:
        return formatted_text

    @classmethod
    def get_image_placeholder_for_processor(
        cls, processor: VLMProcessorLike | None = None
    ) -> str:
        return "<image>"

    @classmethod
    def configure_generator_config(cls, model: Any, config: Any) -> Any:
        """Default no-op; model families can override when needed."""
        return config
