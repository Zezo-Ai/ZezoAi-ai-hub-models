# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.models.grootn15.app import build_app, get_default_dataset_path
from qai_hub_models.models.grootn15.constants import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_EMBODIMENT_TAG,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.dataset import (
    LeRobotSingleDataset,
)
from qai_hub_models.models.grootn15.model import load_checkpoint
from qai_hub_models.utils.evaluate.helpers import EvalMode


def test_one_sample() -> None:
    """
    Run a single forward pass and assert:
    1. App-path and direct-policy-path produce close action predictions.
    2. Predicted actions are close to ground truth (RMSE sanity check).
    """
    torch.manual_seed(42)

    policy = load_checkpoint(
        checkpoint="DEFAULT",
        data_config=DEFAULT_DATA_CONFIG,
        embodiment_tag=DEFAULT_EMBODIMENT_TAG,
        device="cpu",
    )

    dataset = LeRobotSingleDataset(
        dataset_path=get_default_dataset_path(),
        modality_configs=policy.modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=DEFAULT_EMBODIMENT_TAG,
    )
    step_data = dataset.get_step_data(dataset.trajectory_ids[0], 0)

    cpu_rng_state = torch.get_rng_state()

    # Direct policy path
    pred_actions_policy = policy.get_action(step_data)

    # App path
    app = build_app(policy, EvalMode.FP)
    torch.set_rng_state(cpu_rng_state)
    pred_actions_app = app.predict_action_chunk(step_data)  # (1, H, total_dof)

    modality_keys = app.config.modality_keys

    # Parity check — app vs policy (both at horizon step 0)
    pred_app_concat = pred_actions_app[0, 0].numpy()  # (total_dof,)
    pred_policy_concat = np.concatenate(
        [np.atleast_1d(pred_actions_policy[f"action.{k}"][0]) for k in modality_keys],
        axis=0,
    )

    assert pred_app_concat.shape == pred_policy_concat.shape, (
        f"Shape mismatch: app={pred_app_concat.shape}, policy={pred_policy_concat.shape}"
    )
    np.testing.assert_allclose(
        pred_app_concat,
        pred_policy_concat,
        rtol=1e-2,
        atol=1e-2,
        err_msg="App path and direct policy path predictions differ beyond tolerance.",
    )

    # Ground-truth sanity check — RMSE of first horizon step should be reasonable
    gt_concat = np.concatenate(
        [step_data[f"action.{k}"] for k in modality_keys], axis=-1
    )  # (action_horizon, total_dof)
    rmse = float(np.sqrt(np.mean((pred_app_concat - gt_concat[0]) ** 2)))
    assert rmse < 0.1, f"App vs GT RMSE {rmse:.6f} exceeds threshold 0.1"

    print("PASS: test_one_sample")


if __name__ == "__main__":
    test_one_sample()
