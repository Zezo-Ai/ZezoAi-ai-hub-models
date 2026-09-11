# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
- Patch flash-attn related dependencies and set attn mode to eager
- Stub out flash_attn to avoid requiring CUDA and a compiled build.
  flash_attn is a dependency in GR00T Eagle2 class but is not used at export time, so the real package is not needed.
Import this module before any transformer and groot imports.
"""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import sys
from typing import Any
from unittest.mock import MagicMock

import torch
from transformers import AutoModel
from transformers.models.siglip.modeling_siglip import SiglipAttention

# Prevents NameError / RuntimeError in transformers flash-attention checks
# when no CUDA device is available
if not torch.cuda.is_available():

    def _safe_get_device_capability(
        device: torch.device | str | int | None = None,
    ) -> tuple[int, int]:
        return (8, 0)  # Dummy Ampere; flash-attn swap won't affect export

    torch.cuda.get_device_capability = _safe_get_device_capability

# Forces eager attention for the SigLIP vision encoder so it doesn't
# try to dispatch to flash-attention at runtime.
_original_siglip_forward = SiglipAttention.forward


def _patched_siglip_forward(self: SiglipAttention, *args: Any, **kwargs: Any) -> Any:
    if hasattr(self, "config"):
        self.config._attn_implementation = "eager"
        self.config._attn_implementation_autoset = True
    return _original_siglip_forward(self, *args, **kwargs)


SiglipAttention.forward = _patched_siglip_forward

# Forces attn_implementation='eager' for all nested sub-configs loaded
# via AutoModel.from_config
_original_from_config = AutoModel.from_config


def _patched_from_config(cls: type, config: Any, **kwargs: Any) -> Any:
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
        config._attn_implementation_autoset = True
    return _original_from_config(config, **kwargs)


setattr(AutoModel, "from_config", classmethod(_patched_from_config))  # noqa: B010


# Stub out flash_attn to avoid requiring CUDA and a compiled build
def _make_stub(name: str) -> MagicMock:
    mod = MagicMock(spec=None)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__name__ = name
    mod.__package__ = name.split(".")[0]
    mod.__path__ = []
    mod.__loader__ = None
    return mod


_SUBMODULES = [
    "flash_attn",
    "flash_attn.flash_attn_interface",
    "flash_attn.bert_padding",
]
for _name in _SUBMODULES:
    if _name not in sys.modules:  # don't clobber a real install
        sys.modules[_name] = _make_stub(_name)

# Set __version__ on the stub so callers that do `flash_attn.__version__`
# get the right answer without touching importlib.metadata globally.
# We also attach metadata directly on the stub so that any call to
# importlib.metadata.distribution("flash_attn") is satisfied via the
# sys.modules entry
_flash_attn_stub = sys.modules["flash_attn"]
_flash_attn_stub.__version__ = "2.7.1"  # type: ignore[attr-defined]
_dist_mock = MagicMock()
_dist_mock.metadata = {"Name": "flash_attn", "Version": "2.7.1"}
_dist_mock.version = "2.7.1"
_dist_mock.name = "flash_attn"
_flash_attn_stub.__dist__ = _dist_mock  # type: ignore[attr-defined]
