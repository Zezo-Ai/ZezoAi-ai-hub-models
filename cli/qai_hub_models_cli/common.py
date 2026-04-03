# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from enum import Enum

STORE_URL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com"
ASSET_FOLDER = "qai-hub-models/models/{model_id}/releases/v{version}"


# These are just syntactic sugar; you can use strings to get values from older versions.
class Precision(Enum):
    FLOAT = "float"
    W8A8 = "w8a8"
    W8A16 = "w8a16"
    W16A16 = "w16a16"
    W4A16 = "w4a16"
    W4 = "w4"
    W8A8_MIXED_INT16 = "w8a8_mixed_int16"
    W8A16_MIXED_INT16 = "w8a16_mixed_int16"
    W8A8_MIXED_FP16 = "w8a8_mixed_fp16"
    W8A16_MIXED_FP16 = "w8a16_mixed_fp16"
    MXFP4 = "mxfp4"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    MIXED = "mixed"
    MIXED_WITH_FLOAT = "mixed_with_float"


# These are just syntactic sugar; you can use strings to get values from older versions.
class TargetRuntime(Enum):
    TFLITE = "tflite"
    QNN_DLC = "qnn_dlc"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"
    GENIE = "genie"
    ONNXRUNTIME_GENAI = "onnxruntime_genai"
    VOICE_AI = "voice_ai"
    LLAMA_CPP_CPU = "llama_cpp_cpu"
    LLAMA_CPP_GPU = "llama_cpp_gpu"
    LLAMA_CPP_NPU = "llama_cpp_npu"
