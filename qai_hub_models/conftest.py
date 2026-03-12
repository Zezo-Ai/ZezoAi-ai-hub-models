# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest

# Import modules with @pytest_cli_envvar-decorated envvars to populate the registry
import qai_hub_models.scorecard.envvars
import qai_hub_models.test.utils.envvars  # noqa: F401
from qai_hub_models.utils.envvar_bases import PYTEST_CLI_ENVVAR_REGISTRY


def pytest_addoption(parser: pytest.Parser) -> None:
    # Get the underlying argparse parser and add a group for QAIHM options
    group = parser.getgroup(
        "AI Hub Models",
        "AI Hub Models test options (can also be set via environment variables)",
    )

    # Use each envvar's add_arg method to register it with setenv=True
    # This makes the envvar's custom action set the environment variable
    # during argument parsing.
    for envvar in PYTEST_CLI_ENVVAR_REGISTRY:
        envvar.add_arg(group, setenv=True)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "compile: Run compile tests.")
    config.addinivalue_line("markers", "quantize: Run quantize tests.")
    config.addinivalue_line("markers", "profile: Run profile tests.")
    config.addinivalue_line("markers", "inference: Run inference tests.")
    config.addinivalue_line("markers", "trace: Run trace accuracy tests.")
    config.addinivalue_line(
        "markers", "llm_perf: Run LLM performance collection tests."
    )


def pytest_collection_modifyitems(
    items: list[pytest.Item], config: pytest.Config
) -> None:
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("unmarked")
