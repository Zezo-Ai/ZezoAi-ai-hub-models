# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Shim: this module has moved to qai_hub_models.common.
# This re-export exists for backwards compatibility and will be removed
# in a future release.
import warnings

warnings.warn(
    "qai_hub_models.models.common is deprecated. "
    "Import from qai_hub_models directly instead.",
    DeprecationWarning,
    stacklevel=2,
)

from qai_hub_models.common import *  # noqa: F403, E402
from qai_hub_models.common import __all__  # noqa: F401, E402
