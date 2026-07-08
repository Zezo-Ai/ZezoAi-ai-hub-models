# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models._shared.llm.evaluate import llm_evaluate
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    HF_REPO_NAME,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
    Qwen2_5_VL_7B_VisionEncoder,
)

if __name__ == "__main__":
    llm_evaluate(
        quantized_model_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_model_cls=Qwen2_5_VL_7B_PreSplit,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        vision_encoder_cls=Qwen2_5_VL_7B_VisionEncoder,
        hf_repo_name=HF_REPO_NAME,
        vlm_image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
    )
