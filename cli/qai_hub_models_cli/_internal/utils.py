# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from pathlib import Path

from qai_hub_models_cli.envvars import USE_INTERNAL_RELEASES_ENVVAR, bool_envvar_value

QAIHM_REPO_ROOT = Path(__file__).parent.parent.parent.parent


def use_internal_releases() -> bool:
    """
    Check if the internal (private) S3 release should be used instead of the public release.
    Returns True if the QAIHM_CLI_USE_INTERNAL_RELEASES env var is truthy.
    """
    return bool_envvar_value(USE_INTERNAL_RELEASES_ENVVAR)


def is_internal_repo() -> bool:
    """Check if running from the internal repository via git remote URL."""
    # Avoids circular import.
    from qai_hub_models_cli.versions import CURRENT_VERSION

    if not CURRENT_VERSION.is_devrelease:
        return False
    try:
        git_config = QAIHM_REPO_ROOT / ".git" / "config"
        return "ai-hub-models-internal" in git_config.read_text()
    except (OSError, ValueError):
        return False
