# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Profile compiled models on real devices."""

from __future__ import annotations

import qai_hub as hub

from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)


def run_profile(
    model_name: str,
    device: hub.Device,
    options: str,
    target_model: hub.Model,
) -> hub.client.ProfileJob:
    """Submit a profile job for ``target_model`` on ``device``."""
    print(f"Profiling model {model_name} on a hosted device.")
    return hub.submit_profile_job(
        model=target_model,
        device=device,
        name=model_name,
        options=options,
    )


def run_collection_profile(
    model_name: str,
    device: hub.Device,
    options_per_component: dict[str, str],
    target_models: ComponentGroup[hub.Model],
    components: list[str] | None = None,
) -> ComponentGroup[hub.client.ProfileJob]:
    """Submit one profile job per component."""
    components = components if components is not None else list(target_models)
    profile_jobs: ComponentGroup[hub.client.ProfileJob] = ComponentGroup()
    for name in components:
        print(f"Profiling model {name} on a hosted device.")
        profile_jobs[name] = hub.submit_profile_job(
            model=target_models[name],
            device=device,
            name=f"{model_name}_{name}",
            options=options_per_component.get(name, ""),
        )
    return profile_jobs


def run_multi_graph_profile(
    model_name: str,
    device: hub.Device,
    options_per_graph: MultiGraphGroup[str],
    target_model: hub.Model,
) -> MultiGraphGroup[hub.client.ProfileJob]:
    """
    Submit one profile job per graph against the single linked context binary.

    Each ``options`` string selects which graph inside the model to profile.
    """
    profile_jobs: MultiGraphGroup[hub.client.ProfileJob] = MultiGraphGroup()
    for graph_name, opts in options_per_graph.items():
        print(f"Profiling {model_name} ({graph_name}) on a hosted device.")
        profile_jobs[graph_name] = hub.submit_profile_job(
            model=target_model,
            device=device,
            name=f"{model_name}_{graph_name}",
            options=opts,
        )
    return profile_jobs


def run_multi_graph_collection_profile(
    model_name: str,
    device: hub.Device,
    options_per_graph: MultiGraphComponentGroup[str],
    target_models: ComponentGroup[hub.Model],
    components: list[str] | None = None,
) -> MultiGraphComponentGroup[hub.client.ProfileJob]:
    """
    Submit one profile job per ``(component, graph)``.

    The hub target model is selected per-component (link consolidates all graphs
    of a component into one model); ``options_per_graph`` selects the graph
    inside that model at profile time.
    """
    components = components if components is not None else list(target_models)
    profile_jobs: MultiGraphComponentGroup[hub.client.ProfileJob] = (
        MultiGraphComponentGroup()
    )
    for (comp_name, graph_name), opts in options_per_graph.items():
        if comp_name not in components:
            continue
        print(f"Profiling model {comp_name} on a hosted device.")
        profile_jobs[(comp_name, graph_name)] = hub.submit_profile_job(
            model=target_models[comp_name],
            device=device,
            name=f"{model_name}_{comp_name}_{graph_name}",
            options=opts,
        )
    return profile_jobs
