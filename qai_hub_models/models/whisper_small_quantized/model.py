# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# flake8: noqa: E402
# (module level import not at top of file)

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
MODEL_ID = __name__.split(".")[-2]
from qai_hub_models.utils.quantization_aimet_onnx import ensure_aimet_onnx_installed

ensure_aimet_onnx_installed(model_id=MODEL_ID)
# isort: on

import base64
import os
from pathlib import Path

import torch
from transformers import WhisperConfig
from typing_extensions import Self

from qai_hub_models.configs.metadata_yaml import ModelMetadata
from qai_hub_models.models._shared.hf_whisper_quantized.model import (
    WhisperDecoderQuantizableBase,
    WhisperEncoderQuantizableBase,
)
from qai_hub_models.models.whisper_small.model import WhisperSmall
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
WHISPER_VERSION = "openai/whisper-small"
ENCODER_AIMET = "encoder.aimet"
DECODER_AIMET = "decoder.aimet"
TIKTOKEN_URL = CachedWebModelAsset(
    "https://raw.githubusercontent.com/openai/whisper/839639a223b92ad61851baae9ad8a695ccb41ce5/whisper/assets/multilingual.tiktoken",
    "whisper_small_quantized",
    1,
    "multilingual.tiktoken",
)


class WhisperSmallEncoderQuantizable(WhisperEncoderQuantizableBase):
    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        return WhisperSmall.from_pretrained().encoder

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[Path, Path]:
        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(ENCODER_AIMET, "model.onnx"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(ENCODER_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings


class WhisperSmallDecoderQuantizable(WhisperDecoderQuantizableBase):
    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        return WhisperSmall.from_pretrained().decoder

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[Path, Path]:
        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(DECODER_AIMET, "model.onnx"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(DECODER_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings


@CollectionModel.add_component(WhisperSmallEncoderQuantizable)
@CollectionModel.add_component(WhisperSmallDecoderQuantizable)
class WhisperSmallQuantized(CollectionModel):
    def __init__(
        self,
        encoder: WhisperEncoderQuantizableBase,
        decoder: WhisperDecoderQuantizableBase,
        config: WhisperConfig,
        hf_source: str,
    ) -> None:
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.hf_source = hf_source

    @classmethod
    def get_hf_whisper_version(cls) -> str:
        return WHISPER_VERSION

    @classmethod
    def from_pretrained(cls) -> Self:
        encoder = WhisperSmallEncoderQuantizable.from_pretrained()
        decoder = WhisperSmallDecoderQuantizable.from_pretrained()
        return cls(encoder, decoder, encoder.config, WHISPER_VERSION)

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        whisper_tiktoken = TIKTOKEN_URL.fetch()

        with open(whisper_tiktoken, "rb") as f:
            lines = f.readlines()

        with open(os.path.join(output_dir, "vocab.bin"), "wb") as f:
            for line in lines:
                l = line.split()
                if len(l) < 2:
                    continue
                token = base64.b64decode(line.split()[0])
                if b"\0" in token:
                    f.write(token)
                else:
                    f.write(token)
                    f.write(b"\0")
