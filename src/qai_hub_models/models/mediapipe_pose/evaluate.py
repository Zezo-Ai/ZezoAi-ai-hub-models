# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import argparse
import warnings

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.mediapipe_pose import MODEL_ID, App, Model
from qai_hub_models.utils.args import evaluate_parser, get_model_kwargs
from qai_hub_models.utils.base_app import CollectionAppEvaluateProtocol
from qai_hub_models.utils.evaluate import _load_quant_cpu_onnx, evaluate_on_dataset
from qai_hub_models.utils.export.dispatch import resolve_export_model
from qai_hub_models.utils.inference import AsyncOnDeviceModel, compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec

SUPPORTED_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
    Precision.w8a8: [
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
}


DEFAULT_EVAL_DEVICE = "Samsung Galaxy S25 (Family)"


def build_parser(cli_mode: bool = False) -> argparse.ArgumentParser:
    """Build the argparse parser for this model's evaluate script.

    Exposed so the qai-hub-models CLI dispatcher can reuse the model's native
    parser without re-running main(). When *cli_mode* is True, runtime,
    precision, and device/chipset must be explicitly specified.
    """
    return evaluate_parser(
        model_cls=Model,
        supported_dataset_classes=Model.get_eval_dataset_classes(),
        supported_precision_runtimes=SUPPORTED_PRECISION_RUNTIMES,
        default_device=DEFAULT_EVAL_DEVICE,
        cli_mode=cli_mode,
    )


def main(args: argparse.Namespace | None = None) -> None:
    export_model = resolve_export_model(MODEL_ID)
    eval_dataset_classes = Model.get_eval_dataset_classes()
    if args is None:
        warnings.warn(
            "Running `python -m qai_hub_models.models.mediapipe_pose.evaluate` is "
            "deprecated and will be removed in a future release. "
            "Use `qai-hub-models evaluate mediapipe_pose` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = build_parser().parse_args()

    warnings.filterwarnings("ignore")
    model_kwargs = get_model_kwargs(Model, vars(args))

    if len(eval_dataset_classes) == 0:
        print(
            "Model does not have evaluation dataset specified. Evaluating PSNR on a single sample."
        )
        export_model(
            MODEL_ID,
            device=args.device,
            target_runtime=args.target_runtime,
            skip_downloading=True,
            skip_profiling=True,
            compile_options=args.compile_options,
            profile_options=args.profile_options,
            **model_kwargs,
        )
        return

    assert isinstance(App, CollectionAppEvaluateProtocol), (
        "App must implement CollectionAppEvaluateProtocol, when eval_datasets is specified"
    )

    if args.use_dataset_cache:
        raise ValueError("Collection models do not support use_dataset_cache.")

    collection_model = Model.from_pretrained(**model_kwargs)
    num_components = len(collection_model.component_names)
    input_spec: InputSpec | None = None
    torch_model_list = list(collection_model.components.values())
    model_executors: dict[str, CollectionAppEvaluateProtocol] = {}
    on_device_model_list: list[AsyncOnDeviceModel] = []
    if not args.skip_torch_accuracy:
        model_executors["torch"] = App.from_components(torch_model_list)
        input_spec = torch_model_list[0].get_input_spec()

    if not args.skip_device_accuracy or args.compute_quant_cpu_accuracy:
        if args.hub_model_id is not None:
            hub_model_id = args.hub_model_id.split(",")
            assert len(hub_model_id) == num_components, (
                f"Number of hub_model_ids ({len(hub_model_id)}) must equal "
                f"number of components ({num_components})"
            )
            compiled_model_list = [hub.get_model(model_id) for model_id in hub_model_id]
        else:
            compiled_model_list = compile_model_from_args(
                MODEL_ID,
                args,
                model_kwargs,
            )
            assert isinstance(compiled_model_list, list)
        for compiled_model in compiled_model_list:
            if compiled_model.get_producer() is None:
                raise ValueError(
                    "Compiled models must be compiled with AI Hub Workbench; they cannot be uploaded manually."
                )
            on_device_model_list.append(
                AsyncOnDeviceModel(
                    model=compiled_model,
                    input_names=None,
                    device=args.device,
                    inference_options=args.profile_options,
                )
            )
        if not args.skip_device_accuracy:
            model_executors["on-device"] = App.from_components(on_device_model_list)
        if args.compute_quant_cpu_accuracy and args.precision != Precision.float:
            quant_cpu_model_list = [
                _load_quant_cpu_onnx(model) for model in compiled_model_list
            ]
            model_executors["quant cpu"] = App.from_components(quant_cpu_model_list)

        input_spec = on_device_model_list[0].get_input_spec()

    if input_spec is None:
        raise ValueError("Cannot extract input spec.")

    evaluate_on_dataset(
        evaluator_func=collection_model.get_evaluator,
        dataset_cls=args.dataset_cls,
        model_executors=model_executors,
        input_spec=input_spec,
        samples_per_job=args.samples_per_job,
        num_samples=args.num_samples,
        seed=args.seed,
        use_cache=args.use_dataset_cache,
    )


if __name__ == "__main__":
    main()
