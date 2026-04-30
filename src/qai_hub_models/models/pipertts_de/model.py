# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.pipertts.model import (
    SDP,
    Decoder,
    Encoder,
    Flow,
    PiperTTS,
)
from qai_hub_models.models._shared.voiceai_tts.language import TTSLanguage
from qai_hub_models.models._shared.voiceai_tts.t5_g2p import (
    T5Decoder,
    T5Encoder,
)
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]


@CollectionModel.add_component(Encoder, "encoder")
@CollectionModel.add_component(SDP, "sdp")
@CollectionModel.add_component(Flow, "flow")
@CollectionModel.add_component(Decoder, "decoder")
@CollectionModel.add_component(T5Encoder, "charsiu_encoder")
@CollectionModel.add_component(T5Decoder, "charsiu_decoder")
class PiperTTS_DE(PiperTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.GERMAN
