# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from functools import partial
from pathlib import Path
from typing import cast
from unittest.mock import Mock

from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.evaluate.dispatch import select_evaluate_pipeline
from qai_hub_models.utils.export.dispatch import ResolvedModel


def test_select_evaluate_pipeline_binds_resolved_context() -> None:
    """select_evaluate_pipeline picks a pipeline for the model_cls and binds resolved fields."""

    class FakeModel:
        pass

    resolved = ResolvedModel(
        model_id="foo",
        model_cls=FakeModel,
        manifest=Mock(),
        display_name="Foo",
        source_dir=Path("/fake"),
        supports_quant_cpu=True,
    )
    bound = select_evaluate_pipeline(resolved)
    # Bound keywords are the ResolvedModel fields accepted by the pipeline,
    # excluding model_id (kept unbound so positional callers don't collide).
    bound_partial = cast(partial, bound)
    assert "model_id" not in bound_partial.keywords
    assert bound_partial.keywords["model_cls"] is FakeModel
    assert bound_partial.keywords["supports_quant_cpu"] is True


def test_select_evaluate_pipeline_picks_collection_for_collection_model() -> None:
    """select_evaluate_pipeline picks collection_pipeline for CollectionModel subclasses."""

    class FakeCollectionModel(CollectionModel):
        pass

    resolved = ResolvedModel(
        model_id="fake_collection",
        model_cls=FakeCollectionModel,
        manifest=Mock(),
        display_name="Fake Collection",
        source_dir=Path("/fake"),
        app_cls=Mock(),
    )
    bound = select_evaluate_pipeline(resolved)
    # The collection pipeline requires app_cls in its signature
    bound_partial = cast(partial, bound)
    assert bound_partial.keywords["model_cls"] is FakeCollectionModel
    assert bound_partial.keywords["app_cls"] is resolved.app_cls
    # Verify the function name matches the collection pipeline
    assert (
        bound_partial.func.__module__
        == "qai_hub_models.utils.evaluate.collection_pipeline"
    )
