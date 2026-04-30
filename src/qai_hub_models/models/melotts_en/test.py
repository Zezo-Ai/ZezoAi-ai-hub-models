# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.melotts.test_utils import (
    melotts_synthesize_and_verify,
)
from qai_hub_models.models.melotts_en.demo import main as demo_main
from qai_hub_models.models.melotts_en.model import MeloTTS_EN


def test_synthesized_audio() -> None:
    melotts_synthesize_and_verify(MeloTTS_EN)


def test_demo() -> None:
    demo_main(is_test=True)
