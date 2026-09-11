# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from qai_hub_models import Precision
from qai_hub_models.models.qwen3_0_6b import Model
from qai_hub_models.models.qwen3_0_6b.demo import qwen3_0_6b_chat_demo
from qai_hub_models.models.qwen3_0_6b.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_0_6B_PreSplit,
    Qwen3_0_6B_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm import test
from qai_hub_models.models.templates.llm.llm_helpers import (
    log_perf_on_device_result,
)
from qai_hub_models.models.templates.llm.model import DEFAULT_CONTEXT_LENGTH
from qai_hub_models.models.templates.llm.perf_collection import (
    LLMPerfConfig,
    get_llm_perf_parametrization,
)
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.scorecard.device import cs_x_elite
from qai_hub_models.utils.checkpoint import CheckpointSpec

# Multi-sequence-length eval (matches qwen3_4b/8b/1.7b): prefill in the 2048
# bucket, decode in the 1 bucket.
DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


# Full model tests
@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen3_0_6B_PreSplit.release()
    Qwen3_0_6B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained(checkpoint)


# qwen3_1_7b is the qwen nightly canary (its SpinQuant R1 path surfaces
# quantization regressions). This model runs the full eval matrix weekly and
# keeps only one cheap row -- the W4A16 MMLU headline metric -- on
# @pytest.mark.nightly for a nightly regression signal.
@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        # Recipe: SpinQuant R2+R3 -> AdaScale -> Calibration. Baselines are
        # measured nightly values. `prompts` rows grade the deterministic FP
        # PreSplit regardless of checkpoint, so both share a conservative floor.
        ("DEFAULT_W4A16", "wikitext", 20.67, 0),
        pytest.param("DEFAULT_W4A16", "mmlu", 0.426, 1000, marks=pytest.mark.nightly),
        # FP PreSplit measures 41/50 on Grace2; the floor absorbs grader jitter.
        ("DEFAULT_W4A16", "grace2", 0.70, 5),
        # FP (unquantized): PPL 19.15, MMLU 47.07%.
        ("DEFAULT_UNQUANTIZED", "wikitext", 19.15, 0),
        ("DEFAULT_UNQUANTIZED", "mmlu", 0.4707, 1000),
        ("DEFAULT_UNQUANTIZED", "grace2", 0.70, 5),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
    tmp_path: Path,
) -> None:
    dataset_cls = next(
        d
        for d in FPSplitModelWrapper.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen3_0_6B_PreSplit.release()
    Qwen3_0_6B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    test.run_llm_evaluate_test(
        task=task,
        checkpoint=checkpoint,
        expected_metric=expected_metric,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        quantized_split_cls=QuantizedSplitModelWrapper,
        fp_split_cls=FPSplitModelWrapper,
        quantized_presplit_cls=Qwen3_0_6B_QuantizablePreSplit,
        fp_presplit_cls=Qwen3_0_6B_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        tmp_path=tmp_path,
        model_id=MODEL_ID,
    )


# Weekly-only (no @pytest.mark.nightly); nightly demo coverage is on qwen3_1_7b.
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Qwen3_0_6B_PreSplit.release()
    Qwen3_0_6B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    qwen3_0_6b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


def _get_llm_perf_params() -> list[tuple[Precision, ScorecardDevice]]:
    params = get_llm_perf_parametrization(
        MODEL_ID,
        default_devices=[cs_x_elite],
        default_precisions=[Precision.w4a16],
    )
    return params if params else [(Precision.w4a16, cs_x_elite)]


@pytest.fixture(scope="session")
def llm_perf_config() -> LLMPerfConfig:
    return LLMPerfConfig.from_environment()


@pytest.mark.skip(
    reason="On-device QDC perf is covered by the scorecard; skipped in the test suite."
)
@pytest.mark.llm_perf
@pytest.mark.skipif(
    not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test requires the qualcomm_device_cloud_sdk package.",
)
@pytest.mark.parametrize(("precision", "device"), _get_llm_perf_params())
def test_llm_perf(
    precision: Precision,
    device: ScorecardDevice,
    llm_perf_config: LLMPerfConfig,
) -> None:
    Qwen3_0_6B_PreSplit.release()
    Qwen3_0_6B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()

    tps, ttft, prefill_tps = test.run_llm_perf_test(
        model_id=MODEL_ID,
        device=device,
        precision=precision,
        output_dir=test.GENIE_BUNDLES_ROOT,
        qairt_sdk_path=llm_perf_config.qairt_sdk_path,
        skip_perf_update=llm_perf_config.skip_perf_update,
    )
    log_perf_on_device_result(
        model_name=MODEL_ID,
        precision=str(precision),
        device=device.name,
        tps=tps,
        prefill_tps=prefill_tps,
        ttft_ms=ttft,
    )
