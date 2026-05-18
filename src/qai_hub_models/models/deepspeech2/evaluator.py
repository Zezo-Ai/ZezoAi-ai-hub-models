# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import jiwer
import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.metrics import WORD_ERROR_RATE, MetricMetadata
from qai_hub_models.models.protocols import ExecutableModelProtocol


class DeepSpeech2Evaluator(BaseEvaluator):
    """WER evaluator for DeepSpeech2 on LibriSpeech test-clean.

    The LibriSpeechDataset yields raw waveforms. This evaluator converts each
    waveform to a spectrogram (matching the preprocessing in DeepSpeech2App),
    runs inference, decodes the CTC output, and accumulates WER.
    """

    def __init__(self, model: ExecutableModelProtocol) -> None:
        self.model = model
        self.reset()

    def add_batch(
        self,
        output: torch.Tensor,
        gt_tensor: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        output
            Ignored — inference is run internally from the raw waveform stored
            during preprocessing. This evaluator overrides __call__ instead.
        gt_tensor
            Tensor of shape [batch, max_text_length] with ASCII char codes,
            padded with zeros, as returned by LibriSpeechDataset.
        """

    def __call__(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
        gt_tensor: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        inputs
            Tuple of (waveform, attention_mask) from LibriSpeechDataset.
            waveform: [batch, sequence_length] float32 raw audio at 16 kHz.
        gt_tensor
            [batch, max_text_length] ASCII char codes padded with zeros.
        """
        from qai_hub_models.models.deepspeech2.app import (
            DeepSpeech2App,
            ctc_greedy_decode,
        )

        waveform, _ = inputs
        num_frames = 3500
        app = DeepSpeech2App(self.model, context_len=num_frames)

        for i in range(waveform.shape[0]):
            spec = app._waveform_to_spectrogram(waveform[i])
            current_len = spec.shape[1]
            if current_len > num_frames:
                spec = spec[:, :num_frames, :]
            elif current_len < num_frames:
                spec = torch.nn.functional.pad(
                    spec, (0, 0, 0, num_frames - current_len)
                )

            indices = self.model(spec)
            if isinstance(indices, tuple):
                indices = indices[0]
            prediction = ctc_greedy_decode(indices).lower()

            gt_chars = gt_tensor[i]
            reference = "".join(chr(int(c)) for c in gt_chars if int(c) != 0)

            self.predictions.append(prediction)
            self.references.append(reference)

    def reset(self) -> None:
        self.predictions: list[str] = []
        self.references: list[str] = []

    def get_accuracy_score(self) -> float:
        return jiwer.wer(self.references, self.predictions) * 100

    def formatted_accuracy(self) -> str:
        wer_score = self.get_accuracy_score()
        return f"Word Error Rate: {wer_score:.3f}"

    def get_metric_metadata(self) -> MetricMetadata:
        return WORD_ERROR_RATE
