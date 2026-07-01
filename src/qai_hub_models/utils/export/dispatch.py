# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Dispatch a ``model_id`` to the ``export_model`` function for its pipeline.

Lives in its own file so callers can import the dispatcher without
triggering top-level imports of every pipeline module. The pipelines
themselves must not import from this module -- doing so would re-create
the ``utils.export -> pipeline -> utils.args -> ...`` cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    PrecompiledCollectionModel,
)
from qai_hub_models.utils.base_model import PrecompiledWorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.context import resolve_model_cls

from .collection_pipeline import export_model as _collection_export_model
from .multi_graph_collection_pipeline import (
    export_model as _multi_graph_collection_export_model,
)
from .multi_graph_pipeline import export_model as _multi_graph_export_model
from .pipeline import export_model as _export_model
from .precompiled_pipeline import export_model as _precompiled_export_model


def resolve_export_model(model_id: str) -> Callable[..., Any]:
    """Return the ``export_model`` function for the pipeline matching this model."""
    model_cls = resolve_model_cls(model_id)
    # Precompiled (component or collection) -- already binary, no compile/quantize.
    if issubclass(model_cls, (PrecompiledWorkbenchModel, PrecompiledCollectionModel)):
        return _precompiled_export_model
    if issubclass(model_cls, MultiGraphCollectionModel):
        return _multi_graph_collection_export_model
    if issubclass(model_cls, CollectionModel):
        return _collection_export_model
    if issubclass(model_cls, MultiGraphWorkbenchModel):
        return _multi_graph_export_model
    return _export_model
