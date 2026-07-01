# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Compile uploaded source models to a target on-device runtime."""

from __future__ import annotations

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_model import WorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)
from qai_hub_models.utils.input_spec import InputSpec, to_hub_input_specs


def run_compile(
    model: WorkbenchModel,
    model_name: str,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision,
    source_model: hub.Model,
    input_spec: InputSpec | None = None,
    extra_options: str = "",
    calibration_data: hub.Dataset | None = None,
) -> hub.client.CompileJob:
    """Submit a compile job that lowers ``source_model`` to ``target_runtime``."""
    input_spec = input_spec or model.get_input_spec()
    print(f"Optimizing model {model_name} to run on-device")
    return hub.submit_compile_job(
        model=source_model,
        input_specs=to_hub_input_specs(input_spec),
        device=device,
        name=model_name,
        calibration_data=calibration_data,
        options=model.get_hub_compile_options(
            target_runtime, precision, extra_options, device
        ),
    )


def run_collection_compile(
    model: CollectionModel,
    model_name: str,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision,
    source_models: ComponentGroup[hub.Model],
    input_specs: dict[str, InputSpec] | None = None,
    components: list[str] | None = None,
    extra_options: str = "",
) -> ComponentGroup[hub.client.CompileJob]:
    """Submit one compile job per component."""
    input_specs = input_specs or model.get_input_spec()
    components = components or model.component_names
    compile_jobs: ComponentGroup[hub.client.CompileJob] = ComponentGroup()
    for name in components:
        print(f"Optimizing model {name} to run on-device")
        compile_jobs[name] = hub.submit_compile_job(
            model=source_models[name],
            input_specs=to_hub_input_specs(input_specs[name]),
            device=device,
            name=f"{model_name}_{name}",
            options=model.get_component_hub_compile_options(
                name, target_runtime, precision, extra_options, device
            ),
        )
    return compile_jobs


def run_multi_graph_compile(
    model: MultiGraphWorkbenchModel,
    model_name: str,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision,
    source_models: MultiGraphGroup[hub.Model],
    input_specs: MultiGraphGroup[InputSpec] | None = None,
    extra_options: str = "",
) -> MultiGraphGroup[hub.client.CompileJob]:
    """Submit one compile job per graph."""
    input_specs = input_specs or model.get_input_spec()
    compile_options = model.get_hub_compile_options(
        target_runtime, precision, extra_options, device
    )
    compile_jobs: MultiGraphGroup[hub.client.CompileJob] = MultiGraphGroup()
    for graph_name, spec in input_specs.items():
        print(f"Optimizing {model_name} ({graph_name}) to run on-device")
        compile_jobs[graph_name] = hub.submit_compile_job(
            model=source_models[graph_name],
            input_specs=to_hub_input_specs(spec),
            device=device,
            name=f"{model_name}_{graph_name}",
            options=compile_options.get(graph_name, ""),
        )
    return compile_jobs


def run_multi_graph_collection_compile(
    model: MultiGraphCollectionModel,
    model_name: str,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision,
    source_models: MultiGraphComponentGroup[hub.Model],
    input_specs: MultiGraphComponentGroup[InputSpec] | None = None,
    components: list[str] | None = None,
    extra_options: str = "",
) -> MultiGraphComponentGroup[hub.client.CompileJob]:
    """Submit one compile job per ``(component, graph)`` pair."""
    input_specs = input_specs or model.get_input_spec()
    components = components or model.component_names
    compile_options = model.get_hub_compile_options(
        target_runtime, precision, extra_options, device
    )
    compile_jobs: MultiGraphComponentGroup[hub.client.CompileJob] = (
        MultiGraphComponentGroup()
    )
    for (comp_name, graph_name), spec in input_specs.items():
        if comp_name not in components:
            continue
        print(f"Optimizing model {comp_name} to run on-device")
        compile_jobs[(comp_name, graph_name)] = hub.submit_compile_job(
            model=source_models[(comp_name, graph_name)],
            input_specs=to_hub_input_specs(spec),
            device=device,
            name=f"{model_name}_{comp_name}",
            options=compile_options.get((comp_name, graph_name), ""),
        )
    return compile_jobs
