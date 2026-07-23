# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm import tqdm

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.grootn15 import MODEL_ID
from qai_hub_models.models.grootn15.app import (
    GrootApp,
    build_app,
    get_default_dataset_path,
)
from qai_hub_models.models.grootn15.evaluator import LeRobotEvaluator
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.dataset import (
    LeRobotSingleDataset,
)
from qai_hub_models.models.grootn15.model import GrootCollection as Model
from qai_hub_models.models.grootn15.model import (
    load_checkpoint,
)
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)

# Type alias
InferFn = Callable[[list[dict[str, Any]]], list[dict]]


def _build_dof_slices(
    modality_keys: list[str],
    dataset: LeRobotSingleDataset,
) -> list[tuple[str, slice]]:
    """
    Build DOF slices for per-modality-group RMSE
    Groups consecutive keys sharing the same suffix (arm/hand/etc).
    e.g. ["right_arm","left_arm","right_hand","left_hand"]
      -> [("right_arm_left_arm", slice(0,14)), ("right_hand_left_hand", slice(14,26))]
    """
    # Get per-key DOF dims from first step of first trajectory
    traj_id = dataset.trajectory_ids[0]
    data_point = dataset.get_step_data(traj_id, 0)
    key_dims: list[tuple[str, int]] = []
    for key in modality_keys:
        val = np.asarray(data_point[f"action.{key}"])  # (action_horizon, dim)
        dim = val.shape[-1] if val.ndim > 1 else val.shape[0]
        key_dims.append((key, int(dim)))

    # Group consecutive keys by suffix (last _ token)
    def _suffix(k: str) -> str:
        return k.rsplit("_", 1)[-1]

    slices: list[tuple[str, slice]] = []
    offset = 0
    group_keys: list[str] = []
    group_dim = 0
    cur_suffix = _suffix(key_dims[0][0]) if key_dims else ""

    for key, dim in key_dims:
        suf = _suffix(key)
        if suf != cur_suffix and group_keys:
            slices.append(
                (
                    "_".join(group_keys),
                    slice(offset - group_dim, offset),
                )
            )
            group_keys = []
            group_dim = 0
            cur_suffix = suf
        group_keys.append(key)
        group_dim += dim
        offset += dim

    # flush last group
    if group_keys:
        slices.append(
            (
                "_".join(group_keys),
                slice(offset - group_dim, offset),
            )
        )

    return slices


# GT / pred helpers
def _get_gt_step(
    data_point: dict[str, Any],
    modality_keys: list[str],
) -> np.ndarray:
    """GT for a single step from a LeRobotSingleDataset row."""
    return np.concatenate(
        [
            np.atleast_1d(np.asarray(data_point[f"action.{key}"])[0])
            for key in modality_keys
        ],
        axis=0,
    )  # (total_dof,)


def _get_pred_steps(
    action_chunk: dict[str, Any],
    modality_keys: list[str],
    action_horizon: int,
) -> list[np.ndarray]:
    """Expand a single action_chunk dict into a list of action_horizon pred arrays."""
    return [
        np.concatenate(
            [
                np.atleast_1d(np.asarray(action_chunk[f"action.{key}"])[0][j])
                for key in modality_keys
            ],
            axis=0,
        )
        for j in range(action_horizon)
    ]  # (total_dof,)


# Core evaluation loop
def _evaluate(
    app: GrootApp,
    dataset: LeRobotSingleDataset,
    modality_keys: list[str],
    traj_ids: list[int],
    batch_size: int,
    infer_fn: InferFn,
    dof_slices: list[tuple[str, slice]],
) -> LeRobotEvaluator:
    """
    Step 1 — per trajectory:
      - collect GT at every step
      - collect obs + cadence points at every action_horizon steps

    Step 2 — batched inference:
      - batch cadence points across trajectories (batch_size at a time)
      - infer_fn returns list of action_chunk dicts

    Step 3 — expand + feed evaluator:
      - expand each chunk into action_horizon pred steps
      - feed VLARMSEEvaluator step by step
    """
    evaluator = LeRobotEvaluator(dof_slices=dof_slices)
    action_horizon = app.config.action_horizon

    # Step 1: collect GT + cadence obs
    cadence_map: dict[int, dict] = {}
    for traj_id in tqdm(traj_ids, desc="Collecting GT + cadence points"):
        traj_len = dataset.trajectory_lengths[traj_id]
        gt_steps: list[np.ndarray] = []
        base_steps: list[int] = []
        step_datas: list[dict] = []

        for step in range(traj_len):
            data_point = dataset.get_step_data(traj_id, step)

            # GT at every step
            gt_steps.append(_get_gt_step(data_point, modality_keys))

            # Cadence: collect obs for inference every action_horizon steps
            if step % action_horizon == 0:
                base_steps.append(step)
                step_datas.append(data_point)

        cadence_map[traj_id] = {
            "base_steps": base_steps,
            "step_datas": step_datas,
            "gt_steps": gt_steps,
        }

    # Step 2: batched inference
    flat_points: list[tuple[int, int]] = [
        (traj_id, cidx)
        for traj_id in traj_ids
        for cidx in range(len(cadence_map[traj_id]["base_steps"]))
    ]
    batches = [
        flat_points[i : i + batch_size] for i in range(0, len(flat_points), batch_size)
    ]

    pred_chunks_map: dict[int, dict[int, dict]] = {traj_id: {} for traj_id in traj_ids}
    for batch in tqdm(batches, desc="Inference"):
        batch_step_datas = [
            cadence_map[traj_id]["step_datas"][cidx] for traj_id, cidx in batch
        ]
        pred_chunks = infer_fn(batch_step_datas)
        for (traj_id, cidx), pred_chunk in zip(batch, pred_chunks, strict=False):
            pred_chunks_map[traj_id][cidx] = pred_chunk

    # Step 3: expand + feed evaluator
    for traj_id in tqdm(traj_ids, desc="Computing RMSE"):
        gt_steps = cadence_map[traj_id]["gt_steps"]
        base_steps = cadence_map[traj_id]["base_steps"]
        traj_len = len(gt_steps)

        # Build pred_action_across_time
        pred_steps: dict[int, np.ndarray] = {}
        for cidx, base_step in enumerate(base_steps):
            chunk_preds = _get_pred_steps(
                pred_chunks_map[traj_id][cidx], modality_keys, action_horizon
            )
            for j, pred in enumerate(chunk_preds):
                target = base_step + j
                if target >= traj_len:
                    break
                pred_steps[target] = pred

        # Feed evaluator step-by-step
        for step in range(traj_len):
            if step not in pred_steps:
                continue
            evaluator.add_batch(
                pred_steps[step][np.newaxis, :],
                gt_steps[step][np.newaxis, :],
            )

    return evaluator


def _make_infer_fn(app: GrootApp) -> InferFn:
    def infer_fn(step_datas: list[dict[str, Any]]) -> list[dict]:
        batched_pred = app.predict_action_chunk(step_datas)  # dict of (B, H, dim)
        return [
            {k: v[i : i + 1] for k, v in batched_pred.items()}
            for i in range(len(step_datas))
        ]

    return infer_fn


def main() -> None:
    parser = get_model_cli_parser(Model)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
        supported_precisions={Precision.float},
        available_target_runtimes=[TargetRuntime.QNN_CONTEXT_BINARY],
        default_device="Samsung Galaxy S25 (Family)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=get_default_dataset_path(),
        help="Path to LeRobot-format dataset directory.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=-1,
        help="Number of trajectories to evaluate. -1 = all. Default: -1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Cadence points per inference batch. ",
    )

    args = parser.parse_args()
    validate_on_device_demo_args(args, MODEL_ID)

    # Load policy
    policy = load_checkpoint(
        checkpoint=args.checkpoint,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        device=args.host_device,
    )

    # Load LeRobotSingleDataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )

    # modality_keys
    modality_keys: list[str] = [
        key.split(".")[-1] for key in policy.modality_config["action"].modality_keys
    ]

    # Trajectory selection
    all_ids = list(dataset.trajectory_ids)
    traj_ids = (
        all_ids if args.num_trajectories == -1 else all_ids[: args.num_trajectories]
    )

    # DOF slices for per-group RMSE
    dof_slices = _build_dof_slices(modality_keys, dataset)

    # Build GrootApp
    app = build_app(
        policy, args.eval_mode, args.device, args.hub_model_id, args.host_device
    )

    print(
        f"Evaluating {len(traj_ids)} trajectory/ies "
        f"| action_horizon={app.config.action_horizon} "
        f"| batch_size={args.batch_size}"
    )

    # Build infer_fn
    infer_fn = _make_infer_fn(app)

    # Run evaluation
    evaluator = _evaluate(
        app=app,
        dataset=dataset,
        modality_keys=modality_keys,
        traj_ids=traj_ids,
        batch_size=args.batch_size,
        infer_fn=infer_fn,
        dof_slices=dof_slices,
    )

    # Print results
    print(f"\n{'─' * 50}")
    print(f"Evaluation mode : {args.eval_mode.value.upper()}")
    print(f"Dataset         : {args.dataset_path}")
    print(f"Trajectories    : {len(traj_ids)}")
    print(evaluator.formatted_accuracy())
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()
