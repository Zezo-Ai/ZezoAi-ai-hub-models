# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.intern3_5_vl.model import (
    Intern3_5VLPositionProcessor as PositionProcessor,
)

from .model import (
    DEFAULT_PRECISION,
    HF_REPO_NAME,
    HIDDEN_SIZE,
    MIN_MEMORY_RECOMMENDED,
    MODEL_ID,
    NUM_ATTN_HEADS,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
    FPSplitModelWrapper,
    Intern3_5_VL_2B_Collection,
    Intern3_5_VL_2B_Part1_Of_4,
    Intern3_5_VL_2B_Part2_Of_4,
    Intern3_5_VL_2B_Part3_Of_4,
    Intern3_5_VL_2B_Part4_Of_4,
    Intern3_5_VL_2B_PartBase,
    Intern3_5_VL_2B_PreSplit,
    Intern3_5_VL_2B_QuantizablePreSplit,
    Intern3_5_VL_2B_VisionEncoder,
    QuantizedSplitModelWrapper,
)

VisionEncoder = Intern3_5_VL_2B_VisionEncoder
Model = Intern3_5_VL_2B_Collection

__all__ = [
    "DEFAULT_PRECISION",
    "HF_REPO_NAME",
    "HIDDEN_SIZE",
    "MIN_MEMORY_RECOMMENDED",
    "MODEL_ID",
    "NUM_ATTN_HEADS",
    "NUM_KEY_VALUE_HEADS",
    "NUM_LAYERS",
    "NUM_LAYERS_PER_SPLIT",
    "NUM_SPLITS",
    "FPSplitModelWrapper",
    "Intern3_5_VL_2B_Collection",
    "Intern3_5_VL_2B_Part1_Of_4",
    "Intern3_5_VL_2B_Part2_Of_4",
    "Intern3_5_VL_2B_Part3_Of_4",
    "Intern3_5_VL_2B_Part4_Of_4",
    "Intern3_5_VL_2B_PartBase",
    "Intern3_5_VL_2B_PreSplit",
    "Intern3_5_VL_2B_QuantizablePreSplit",
    "Model",
    "PositionProcessor",
    "QuantizedSplitModelWrapper",
    "VisionEncoder",
]
