# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import re

import soundfile

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo


def assert_transcription_matches(
    audio_path: str | os.PathLike,
    expected_text: str,
) -> None:
    wav, sample_rate = soundfile.read(audio_path)

    model = WhisperLargeV3Turbo.from_pretrained()
    app = HfWhisperApp(
        model.encoder, model.decoder, WhisperLargeV3Turbo.get_hf_whisper_version()
    )
    transcription = app.transcribe(wav, sample_rate)

    trans = "".join(re.findall(r"\b\w+\b", transcription))
    expected = "".join(re.findall(r"\b\w+\b", expected_text))
    print(
        "\nOriginal_text: ",
        expected_text,
        "\n",
        "Transcription: ",
        transcription,
        sep="",
    )
    assert trans.lower() == expected.lower()
