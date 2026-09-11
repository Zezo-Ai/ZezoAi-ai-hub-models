# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import MetricMetadata


class LeRobotEvaluator(BaseEvaluator):
    """
    Measures Action RMSE between predicted and ground-truth action chunks.

    add_batch() accepts:
      pred : torch.Tensor (B, H, total_dof) — postprocessed, from either path
      gt   : torch.Tensor (B, H, total_dof) from GrootDataset, optionally in a 1-tuple
    """

    def __init__(
        self,
        dof_slices: list[tuple[str, slice]] | None = None,
    ) -> None:
        self._sum_sq_error: float = 0.0
        self._n_elements: int = 0
        self._n_steps: int = 0

        self._dof_slices: list[tuple[str, slice]] = dof_slices or []
        self._group_sum_sq: list[float] = [0.0] * len(self._dof_slices)
        self._group_n_elem: list[int] = [0] * len(self._dof_slices)

    def _to_array(self, x: Any) -> np.ndarray:
        if isinstance(x, (tuple, list)):
            x = x[0]  # framework wraps GT in a 1-tuple
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    def add_batch(self, pred: Any, gt: Any) -> None:
        pred_arr = self._to_array(pred)
        gt_arr = self._to_array(gt)

        # Flatten batch and horizon dims -> (B*H, total_dof)
        pred_flat = pred_arr.reshape(-1, pred_arr.shape[-1])
        gt_flat = gt_arr.reshape(-1, gt_arr.shape[-1])

        if pred_flat.shape != gt_flat.shape:
            raise ValueError(
                f"Shape mismatch after flatten: pred {pred_flat.shape} vs gt {gt_flat.shape}"
            )
        sq_err = (pred_flat - gt_flat) ** 2
        self._sum_sq_error += float(sq_err.sum())
        self._n_elements += sq_err.size
        self._n_steps += pred_flat.shape[0]  # count individual samples
        for i, (_, sl) in enumerate(self._dof_slices):
            group_sq = sq_err[..., sl]
            self._group_sum_sq[i] += float(group_sq.sum())
            self._group_n_elem[i] += group_sq.size

    def get_accuracy_score(self) -> float:
        """Returns RMSE (lower is better). Raises if no data has been accumulated."""
        if self._n_elements == 0:
            raise RuntimeError(
                "LeRobotEvaluator has no accumulated data. "
                "Ensure add_batch() was called at least once."
            )
        return float(np.sqrt(self._sum_sq_error / self._n_elements))

    def get_group_scores(self) -> list[tuple[str, float]]:
        """Per-modality-group RMSE. Returns [] if no dof_slices were provided."""
        results = []
        for i, (name, _) in enumerate(self._dof_slices):
            n = self._group_n_elem[i]
            rmse = float(np.sqrt(self._group_sum_sq[i] / n)) if n > 0 else 0.0
            results.append((name, rmse))
        return results

    def formatted_accuracy(self) -> str:
        lines = [
            f"Action RMSE: {self.get_accuracy_score():.6f} (over {self._n_steps} cadence-point * horizon steps)"
        ]
        group_scores = self.get_group_scores()
        if group_scores:
            lines.append("  Per-modality-group RMSE:")
            for name, rmse in group_scores:
                lines.append(f"    [{name}]: {rmse:.6f}")
        return "\n".join(lines)

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Action RMSE",
            unit="rad",
            description="Root mean squared error between model predicted actions and ground truth actions from the dataset",
            range=(0.0, float("inf")),
            higher_is_better=False,
        )

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._sum_sq_error = 0.0
        self._n_elements = 0
        self._n_steps = 0
        self._group_sum_sq = [0.0] * len(self._dof_slices)
        self._group_n_elem = [0] * len(self._dof_slices)
