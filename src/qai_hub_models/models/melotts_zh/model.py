# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing_extensions import Self

from qai_hub_models.models._shared.melotts.model import (
    BertWrapper,
    Decoder,
    Encoder,
    Flow,
    MeloTTS,
    get_tts_object,
)
from qai_hub_models.models._shared.voiceai_tts.language import TTSLanguage
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]


class Encoder_ZH(Encoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.CHINESE), speed_adjustment=0.85)


class Flow_ZH(Flow):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.CHINESE))


class Decoder_ZH(Decoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.CHINESE))


class BertWrapper_ZH(BertWrapper):
    @classmethod
    def from_pretrained(cls) -> Self:
        return super().from_pretrained(TTSLanguage.CHINESE)


@CollectionModel.add_component(Encoder_ZH, "encoder")
@CollectionModel.add_component(Flow_ZH, "flow")
@CollectionModel.add_component(Decoder_ZH, "decoder")
@CollectionModel.add_component(BertWrapper_ZH, "bert_wrapper")
class MeloTTS_ZH(MeloTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.CHINESE

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(
            Encoder_ZH.from_pretrained(),
            Flow_ZH.from_pretrained(),
            Decoder_ZH.from_pretrained(),
            BertWrapper_ZH.from_pretrained(),
        )
