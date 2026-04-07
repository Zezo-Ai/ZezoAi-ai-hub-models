# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Generator

import pytest

from qai_hub_models_cli.versions import get_published_versions, get_supported_versions


@pytest.fixture(autouse=True)
def _clear_version_cache() -> Generator[None, None, None]:
    """Clear lru_caches on version functions between tests."""
    get_published_versions.cache_clear()
    get_supported_versions.cache_clear()
    yield
    get_published_versions.cache_clear()
    get_supported_versions.cache_clear()
