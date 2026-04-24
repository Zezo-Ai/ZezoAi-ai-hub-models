# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from qai_hub_models.utils.external_repo import (
    IS_PIP_PACKAGE,
    RepoConfig,
    get_repo_cache_paths,
    setup_external_repos_impl,
)

MODEL_ID = "gkt"
EXTERNAL_REPOS_DIR = os.path.dirname(os.path.normpath(__file__))
LOCK = threading.Lock()

REPO_CONFIGS: dict[str, RepoConfig] = {
    "gkt": RepoConfig(
        repo_url="https://github.com/hustvl/GKT.git",
        commit_sha="104c27f66799f620e54eb0242509ee3b041ae426",
        patches_filename="gkt_patches.diff",
    ),
}

if not TYPE_CHECKING:
    with LOCK:
        setup_external_repos_impl(MODEL_ID, REPO_CONFIGS, EXTERNAL_REPOS_DIR)
    if IS_PIP_PACKAGE:
        __path__ = get_repo_cache_paths(MODEL_ID, REPO_CONFIGS, EXTERNAL_REPOS_DIR)
