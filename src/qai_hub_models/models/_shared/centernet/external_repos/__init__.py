# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

from qai_hub_models.utils.external_repo import (
    IS_PIP_PACKAGE,
    get_repo_cache_paths,
    setup_external_repos,
)


MODEL_ID = "centernet"


if not TYPE_CHECKING:
    with FileLock(Path(__file__).resolve().parent / ".setup.lock"):
        setup_external_repos(MODEL_ID, shared=True)

    if IS_PIP_PACKAGE:
        __path__ = [str(p) for p in get_repo_cache_paths(MODEL_ID, shared=True)]
