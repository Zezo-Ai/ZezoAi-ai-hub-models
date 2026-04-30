# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import nltk
from typing_extensions import Self

from qai_hub_models.models._shared.melotts.model import (
    BertWrapper,
    Decoder,
    Encoder,
    Flow,
    MeloTTS,
    T5Decoder,
    T5Encoder,
    get_tts_object,
)
from qai_hub_models.models._shared.voiceai_tts.language import TTSLanguage
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]


class Encoder_EN(Encoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.ENGLISH), speed_adjustment=0.85)


class Flow_EN(Flow):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.ENGLISH))


class Decoder_EN(Decoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.ENGLISH))


class BertWrapper_EN(BertWrapper):
    @classmethod
    def from_pretrained(cls) -> Self:
        return super().from_pretrained(TTSLanguage.ENGLISH)


@CollectionModel.add_component(Encoder_EN, "encoder")
@CollectionModel.add_component(Flow_EN, "flow")
@CollectionModel.add_component(Decoder_EN, "decoder")
@CollectionModel.add_component(BertWrapper_EN, "bert_wrapper")
@CollectionModel.add_component(T5Encoder, "t5_encoder")
@CollectionModel.add_component(T5Decoder, "t5_decoder")
class MeloTTS_EN(MeloTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.ENGLISH

    @classmethod
    def from_pretrained(cls) -> Self:
        nltk.download("averaged_perceptron_tagger_eng")
        return cls(
            Encoder_EN.from_pretrained(),
            Flow_EN.from_pretrained(),
            Decoder_EN.from_pretrained(),
            BertWrapper_EN.from_pretrained(),
            T5Encoder.from_pretrained(),
            T5Decoder.from_pretrained(),
        )
