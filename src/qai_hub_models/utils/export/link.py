# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Link compiled DLCs into QNN context binaries (AOT runtimes only)."""

from __future__ import annotations

import qai_hub as hub

from qai_hub_models import TargetRuntime
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)


def run_link(
    compiled_model: hub.Model,
    device: hub.Device,
    model_name: str,
    model: BaseModel,
    target_runtime: TargetRuntime,
    extra_options: str = "",
) -> hub.client.LinkJob:
    """Link a compiled DLC into a context binary."""
    assert target_runtime.is_aot_compiled, (
        f"run_link() requires an AOT runtime, got {target_runtime}"
    )
    print(f"Linking {model_name} to context binary")
    return hub.submit_link_job(
        [compiled_model],
        device=device,
        name=model_name,
        options=model.get_hub_link_options(target_runtime, extra_options),
    )


def run_collection_link(
    compiled_models: ComponentGroup[hub.Model],
    device: hub.Device,
    model_name: str,
    model: CollectionModel,
    target_runtime: TargetRuntime,
    extra_options: str = "",
) -> ComponentGroup[hub.client.LinkJob]:
    """Submit one link job per component."""
    assert target_runtime.is_aot_compiled, (
        f"run_collection_link() requires an AOT runtime, got {target_runtime}"
    )
    link_jobs: ComponentGroup[hub.client.LinkJob] = ComponentGroup()
    for name, compiled_model in compiled_models.items():
        print(f"Linking {name} to context binary")
        link_jobs[name] = hub.submit_link_job(
            [compiled_model],
            device=device,
            name=f"{model_name}_{name}",
            options=model.get_component_hub_link_options(
                name,
                target_runtime,
                extra_options,
            ),
        )
    return link_jobs


def run_multi_graph_link(
    compiled_models: MultiGraphGroup[hub.Model],
    device: hub.Device,
    model_name: str,
    model: MultiGraphWorkbenchModel,
    target_runtime: TargetRuntime,
    extra_options: str = "",
) -> hub.client.LinkJob:
    """
    Submit a single link job consolidating every graph's compiled DLC into one
    QNN context binary.
    """
    assert target_runtime.is_aot_compiled, (
        f"run_multi_graph_link() requires an AOT runtime, got {target_runtime}"
    )
    print(f"Linking {model_name} to context binary")
    return hub.submit_link_job(
        list(compiled_models.values()),
        device=device,
        name=model_name,
        options=model.get_hub_link_options(target_runtime, extra_options),
    )


def run_multi_graph_collection_link(
    compiled_models: MultiGraphComponentGroup[hub.Model],
    device: hub.Device,
    model_name: str,
    model: MultiGraphCollectionModel,
    target_runtime: TargetRuntime,
    extra_options: str = "",
) -> ComponentGroup[hub.client.LinkJob]:
    """
    Submit one link job per component. The link job consolidates every graph
    of that component into a single QNN context binary.
    """
    assert target_runtime.is_aot_compiled, (
        f"run_multi_graph_collection_link() requires an AOT runtime, got {target_runtime}"
    )
    grouped: dict[str, list[hub.Model]] = {}
    for (comp_name, _graph), compiled_model in compiled_models.items():
        grouped.setdefault(comp_name, []).append(compiled_model)

    link_jobs: ComponentGroup[hub.client.LinkJob] = ComponentGroup()
    for name, models_for_component in grouped.items():
        print(f"Linking {name} to context binary")
        link_jobs[name] = hub.submit_link_job(
            models_for_component,  # type: ignore[arg-type]
            device=device,
            name=f"{model_name}_{name}",
            options=model.get_component_hub_link_options(
                name,
                target_runtime,
                extra_options,
            ),
        )
    return link_jobs
