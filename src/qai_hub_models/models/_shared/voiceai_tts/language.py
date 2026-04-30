# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from enum import Enum


class TTSLanguage(Enum):
    ENGLISH = "ENGLISH"
    SPANISH = "SPANISH"
    ITALIAN = "ITALIAN"
    GERMAN = "GERMAN"
    CHINESE = "CHINESE"


LANG_CODE_MAP = {
    TTSLanguage.ENGLISH: "EN",
    TTSLanguage.SPANISH: "ES",
    TTSLanguage.ITALIAN: "IT",
    TTSLanguage.GERMAN: "DE",
    TTSLanguage.CHINESE: "ZH",
}

BERT_MODEL_IDS = {
    TTSLanguage.ENGLISH: "bert-base-uncased",
    TTSLanguage.SPANISH: "dccuchile/bert-base-spanish-wwm-uncased",
    TTSLanguage.ITALIAN: "bert-base-uncased",
    TTSLanguage.GERMAN: "bert-base-german-dbmdz-cased",
    TTSLanguage.CHINESE: "bert-base-multilingual-uncased",
}
