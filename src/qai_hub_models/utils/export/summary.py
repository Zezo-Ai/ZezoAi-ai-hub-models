# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tool-version extraction and end-of-run summary printers."""

from __future__ import annotations

from typing import Any

import qai_hub as hub

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_profile_metrics_from_job,
)


def extract_tool_versions(
    profile_job: hub.ProfileJob | None,
    inference_job: hub.InferenceJob | None,
    compile_job: hub.CompileJob | None,
) -> tuple[ToolVersions | None, bool]:
    """
    Return ``(tool_versions, came_from_device_job)``. Prefers profile, falls
    back to inference, then compile.
    """
    if profile_job is not None and profile_job.wait():
        return ToolVersions.from_job(profile_job), True
    if inference_job is not None and inference_job.wait():
        return ToolVersions.from_job(inference_job), True
    if compile_job is not None and compile_job.wait():
        return ToolVersions.from_job(compile_job), False
    return None, False


def print_profile_summary(profile_job: hub.ProfileJob) -> None:
    """Wait for the profile job and print latency / peak-memory metrics."""
    assert profile_job.wait().success, f"Job failed: {profile_job.url}"
    profile_data: dict[str, Any] = profile_job.download_profile()
    print_profile_metrics_from_job(profile_job, profile_data)


def print_inference_summary(
    model: BaseModel,
    inference_job: hub.InferenceJob,
    input_spec: InputSpec,
    target_runtime: TargetRuntime,
    outputs_to_skip: list[int] | None = None,
    metrics: str = "psnr",
) -> None:
    """Compare on-device outputs against a torch baseline."""
    sample_inputs = model.sample_inputs(input_spec, use_channel_last_format=False)
    torch_out = torch_inference(
        model,
        sample_inputs,
        return_channel_last_output=target_runtime.channel_last_native_execution,
    )
    assert inference_job.wait().success, f"Job failed: {inference_job.url}"
    ij_output = inference_job.download_output_data()
    assert ij_output is not None
    print_inference_metrics(
        inference_job,
        ij_output,
        torch_out,
        list(model.get_output_spec()),
        outputs_to_skip=outputs_to_skip or None,
        metrics=metrics,
    )
