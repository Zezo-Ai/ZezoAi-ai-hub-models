# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from qai_hub_models.models.grootn15.constants import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_EMBODIMENT_TAG,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.dataset import (
    LeRobotSingleDataset,
)
from qai_hub_models.models.grootn15.model import (
    DEFAULT_DATASET_ASSET,
    GrootCollection,
    load_checkpoint,
)
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit

_LANG_MAX_BYTES = 512  # max utf-8 bytes per language string
_SAMPLES_PER_JOB = 50


def _encode_strings(strings: list[str]) -> torch.Tensor:
    """Encode list[str] -> uint8 tensor of shape (n_strings, _LANG_MAX_BYTES), zero-padded."""
    out = torch.zeros(len(strings), _LANG_MAX_BYTES, dtype=torch.uint8)
    for i, s in enumerate(strings):
        b = s.encode("utf-8")[:_LANG_MAX_BYTES]
        out[i, : len(b)] = torch.frombuffer(b, dtype=torch.uint8)
    return out


def _build_dof_slices(
    modality_keys: list[str],
    dataset: LeRobotSingleDataset,
) -> list[tuple[str, slice]]:
    """
    Build dof slices for per-modality-group RMSE.
    Groups consecutive keys sharing the same suffix (arm/hand/etc).
    """
    traj_id = dataset.trajectory_ids[0]
    data_point = dataset.get_step_data(traj_id, 0)
    key_dims: list[tuple[str, int]] = []
    for key in modality_keys:
        val = np.asarray(data_point[f"action.{key}"])
        dim = val.shape[-1] if val.ndim > 1 else val.shape[0]
        key_dims.append((key, int(dim)))

    def _suffix(k: str) -> str:
        return k.rsplit("_", 1)[-1]

    offset = 0
    group_dim = 0
    group_keys: list[str] = []
    slices: list[tuple[str, slice]] = []
    cur_suffix = _suffix(key_dims[0][0]) if key_dims else ""

    for key, dim in key_dims:
        suf = _suffix(key)
        if suf != cur_suffix and group_keys:
            slices.append(("_".join(group_keys), slice(offset - group_dim, offset)))
            group_keys = []
            group_dim = 0
            cur_suffix = suf
        group_keys.append(key)
        group_dim += dim
        offset += dim

    if group_keys:
        slices.append(("_".join(group_keys), slice(offset - group_dim, offset)))

    return slices


class GrootDataset(BaseDataset):
    """
    LeRobotSingleDataset wrapper for GrootCollection evaluation.

    __getitem__ returns (field_tensors, gt_chunk) — model-agnostic.
    field_tensors is an ordered tuple matching get_field_keys() ordering.
    Numeric fields -> float32/uint8 tensors. Language list[str] -> zero-padded
    uint8 tensor of shape (n_strings, max_bytes) encoded as utf-8.
    Ground truth -> (action_horizon, total_dof) float32 tensor.
    """

    _cached_dof_slices: list[tuple[str, slice]] | None = None
    _field_keys: list[str] | None = None
    _cached_embodiment_tag: str | None = (
        None  # tracks which config the cache belongs to
    )

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        split: DatasetSplit = DatasetSplit.VAL,
        data_config: str = DEFAULT_DATA_CONFIG,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        num_trajectories: int = -1,
        input_spec: Any = None,
    ) -> None:
        if dataset_path is None:
            dataset_path = Path(DEFAULT_DATASET_ASSET.fetch(extract=True))

        super().__init__(dataset_path=dataset_path, split=split, input_spec=input_spec)

        policy = load_checkpoint(
            checkpoint="DEFAULT",
            data_config=data_config,
            embodiment_tag=embodiment_tag,
            device=GrootCollection._last_host_device,
        )
        self._action_horizon: int = policy.model.action_horizon
        self._modality_keys: list[str] = [
            key.split(".")[-1] for key in policy.modality_config["action"].modality_keys
        ]

        self._lerobot = LeRobotSingleDataset(
            dataset_path=str(self.dataset_path),
            modality_configs=policy.modality_config,
            video_backend="decord",
            video_backend_kwargs=None,
            transforms=None,
            embodiment_tag=embodiment_tag,
        )

        traj_ids = self._lerobot.trajectory_ids
        if num_trajectories != -1:
            traj_ids = traj_ids[:num_trajectories]

        # Pre-flatten: one entry per cadence point across selected trajectories.
        # Skip the last cadence point if the remaining steps are less than action_horizon
        # to avoid zero-padded GT chunks inflating RMSE.
        self._cadence_points: list[tuple[int, int]] = []
        for traj_id in traj_ids:
            traj_len = self._lerobot.trajectory_lengths[traj_id]
            for step in range(0, traj_len, self._action_horizon):
                if step + self._action_horizon <= traj_len:
                    self._cadence_points.append((traj_id, step))

        needs_refresh = GrootDataset._cached_embodiment_tag != embodiment_tag
        if GrootDataset._cached_dof_slices is None or needs_refresh:
            GrootDataset._cached_dof_slices = _build_dof_slices(
                self._modality_keys, self._lerobot
            )
        if GrootDataset._field_keys is None or needs_refresh:
            raw0 = self._lerobot.get_step_data(self._lerobot.trajectory_ids[0], 0)
            GrootDataset._field_keys = sorted(raw0.keys())
        GrootDataset._cached_embodiment_tag = embodiment_tag

    @classmethod
    def get_dof_slices(cls) -> list[tuple[str, slice]]:
        return cls._cached_dof_slices or []

    @classmethod
    def get_field_keys(cls) -> list[str]:
        return cls._field_keys or []

    def __len__(self) -> int:
        return len(self._cadence_points)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        traj_id, base_step = self._cadence_points[index]
        traj_len = self._lerobot.trajectory_lengths[traj_id]

        raw = self._lerobot.get_step_data(traj_id, base_step)

        assert GrootDataset._field_keys is not None
        tensors: list[torch.Tensor] = []
        for key in GrootDataset._field_keys:
            val = raw[key]
            if isinstance(val, list):
                tensors.append(_encode_strings(val))
            else:
                tensors.append(torch.as_tensor(np.asarray(val)))

        # Ground truth: (action_horizon, total_dof), zero-padded past end
        gt_rows: list[np.ndarray] = []
        for j in range(self._action_horizon):
            step = base_step + j
            if step < traj_len:
                dp = self._lerobot.get_step_data(traj_id, step)
                row = np.concatenate(
                    [
                        np.atleast_1d(arr[0] if arr.ndim > 1 else arr)
                        for key in self._modality_keys
                        for arr in (np.asarray(dp[f"action.{key}"]),)
                    ],
                    axis=0,
                ).astype(np.float32)
            else:
                row = np.zeros(gt_rows[0].shape, dtype=np.float32)
            gt_rows.append(row)

        gt_chunk = torch.from_numpy(np.stack(gt_rows))  # (action_horizon, total_dof)
        return tuple(tensors), gt_chunk

    def _download_data(self) -> None:
        DEFAULT_DATASET_ASSET.fetch(extract=True)

    def _validate_data(self) -> bool:
        return self.dataset_path.exists()

    @staticmethod
    def default_samples_per_job() -> int:
        return _SAMPLES_PER_JOB

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="",
            split_description="Robot manipulation trajectories (PickNPlace)",
        )
