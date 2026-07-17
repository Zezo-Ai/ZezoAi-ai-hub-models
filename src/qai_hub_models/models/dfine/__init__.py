# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.dfine.app import DFineApp as App
from qai_hub_models.models.dfine.model import DFine as Model

from .model import MODEL_ID

__all__ = ["MODEL_ID", "App", "Model"]
