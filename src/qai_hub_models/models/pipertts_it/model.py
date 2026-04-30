# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing_extensions import Self

from qai_hub_models.models._shared.pipertts.model import (
    SDP,
    Decoder,
    Encoder,
    Flow,
    PiperTTS,
    get_model,
)
from qai_hub_models.models._shared.voiceai_tts.language import TTSLanguage
from qai_hub_models.models._shared.voiceai_tts.t5_g2p import (
    T5Decoder,
    T5Encoder,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]


class Encoder_IT(Encoder):
    @staticmethod
    def component_precision() -> Precision:
        return Precision.w8a16


class SDP_IT(SDP):
    @staticmethod
    def component_precision() -> Precision:
        return Precision.float


class Flow_IT(Flow):
    @staticmethod
    def component_precision() -> Precision:
        return Precision.w8a16


class Decoder_IT(Decoder):
    @staticmethod
    def component_precision() -> Precision:
        return Precision.w8a16


class T5Encoder_Piper_IT(T5Encoder):
    @staticmethod
    def component_precision() -> Precision:
        return Precision.w8a16


class T5Decoder_Piper_IT(T5Decoder):
    @staticmethod
    def component_precision() -> Precision:
        return Precision.w8a16


@CollectionModel.add_component(Encoder_IT, "encoder")
@CollectionModel.add_component(SDP_IT, "sdp")
@CollectionModel.add_component(Flow_IT, "flow")
@CollectionModel.add_component(Decoder_IT, "decoder")
@CollectionModel.add_component(T5Encoder_Piper_IT, "charsiu_encoder")
@CollectionModel.add_component(T5Decoder_Piper_IT, "charsiu_decoder")
class PiperTTS_IT(PiperTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.ITALIAN

    @classmethod
    def from_pretrained(cls) -> Self:
        model_g = get_model(cls.get_language())
        return cls(
            Encoder_IT(model_g),
            SDP_IT(model_g),
            Flow_IT(model_g),
            Decoder_IT(model_g),
            T5Encoder_Piper_IT.from_pretrained(),
            T5Decoder_Piper_IT.from_pretrained(),
        )
