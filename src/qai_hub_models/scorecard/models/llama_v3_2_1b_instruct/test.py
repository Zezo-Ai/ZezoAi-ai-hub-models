# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from qai_hub_models import Precision
from qai_hub_models.models.llama_v3_2_1b_instruct import Model
from qai_hub_models.models.llama_v3_2_1b_instruct.demo import llama_3_2_1b_chat_demo
from qai_hub_models.models.llama_v3_2_1b_instruct.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    Llama3_2_1B_PreSplit,
    Llama3_2_1B_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.models.templates.llm import test
from qai_hub_models.models.templates.llm.model import DEFAULT_CONTEXT_LENGTH
from qai_hub_models.utils.checkpoint import CheckpointSpec

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained(checkpoint)


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4", "wikitext", 16.74, 0),
        ("DEFAULT_W4", "mmlu", 0.399, 1000),
        ("DEFAULT_W4", "tiny_mmlu", 0.43, 0),
        ("DEFAULT_W4A16", "wikitext", 17.24, 0),
        ("DEFAULT_W4A16", "mmlu", 0.384, 1000),
        # Prompt-generation + LLM-grader smoke test (5 samples). The grader
        # label is an argmax over near-valued logits that can flip across hosts,
        # so expected_metric is a floor. FP PreSplit measures 41/50 on Grace2.
        ("DEFAULT_W4A16", "grace2", 0.70, 5),
        ("DEFAULT_UNQUANTIZED", "wikitext", 12.14, 0),
        ("DEFAULT_UNQUANTIZED", "mmlu", 0.482, 1000),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.41, 0),
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
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
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
        quantized_presplit_cls=Llama3_2_1B_QuantizablePreSplit,
        fp_presplit_cls=Llama3_2_1B_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        tmp_path=tmp_path,
        model_id=MODEL_ID,
    )


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_quantize_and_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quantize the model and verify it can respond with 'Paris'."""
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    # Calibrate on the PreSplit (monolithic QuantSim) like production; split
    # wrappers stack the Part sessions and OOM. Demo below still validates the split.
    checkpoint_path = test.setup_test_quantization(
        Llama3_2_1B_QuantizablePreSplit,
        Llama3_2_1B_PreSplit,
        str(tmp_path),
        precision=Precision.w4a16,
        checkpoint="DEFAULT",
        use_seq_mse=False,
    )
    llama_3_2_1b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint_path,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Llama3_2_1B_PreSplit.release()
    Llama3_2_1B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    llama_3_2_1b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
