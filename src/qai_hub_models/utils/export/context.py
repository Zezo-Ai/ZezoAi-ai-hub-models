# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Resolve a model's pipeline context (model class, display name, source dir)
from its ``model_id``. All four export pipelines call into here so per-model
``export.py`` shims don't have to pass the context explicitly.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


def resolve_model_cls(model_id: str) -> Any:
    """Return the ``Model`` class exported from ``models/<model_id>/__init__.py``."""
    return importlib.import_module(f"qai_hub_models.models.{model_id}").Model


def resolve_model_app_cls(model_id: str) -> Any | None:
    """Return the ``App`` class exported from the model's package, or ``None``."""
    module = importlib.import_module(f"qai_hub_models.models.{model_id}")
    return getattr(module, "App", None)


def resolve_model_display_name(model_id: str) -> str:
    """Resolve the human-readable model name from ``manifest.yaml``."""
    return QAIHMModelManifest.from_model(model_id).name or model_id


def resolve_model_dir(model_id: str) -> Path:
    """Return the on-disk path to ``models/<model_id>/``."""
    return QAIHM_MODELS_ROOT / model_id
