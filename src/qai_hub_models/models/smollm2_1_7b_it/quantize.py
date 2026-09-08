# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models.smollm2_1_7b_it.model import (
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Smollm2_1_7B_Instruct_PreSplit,
    Smollm2_1_7B_Instruct_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm.quantize import llm_quantize

if __name__ == "__main__":
    llm_quantize(
        quantized_model_cls=Smollm2_1_7B_Instruct_QuantizablePreSplit,
        fp_model_cls=Smollm2_1_7B_Instruct_PreSplit,
        model_id=MODEL_ID,
        supported_precisions=SUPPORTED_PRECISIONS,
    )
