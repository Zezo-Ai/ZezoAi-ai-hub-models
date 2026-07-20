# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import onnx
import torch

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
)
from qai_hub_models.models._shared.llm.quantize import (
    _parse_spin_quant_args,
    quantize,
    save_command_args,
)
from qai_hub_models.utils.args import get_quantize_action_with_default
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset


def _quantize_vision_encoder(
    *,
    vision_encoder_cls: type,
    output_dir: str,
    image_height: int,
    image_width: int,
    num_calibration_samples: int = 100,
) -> None:
    """Quantize the VEG (Vision Embedding Generator).

    If a pre-exported vision_encoder.onnx exists in *output_dir* (produced by
    the export/SpinQuant phase), the VEG QuantSim is built on that graph.
    Otherwise, a fresh VEG ONNX is exported from the HF model.

    Produces vision_encoder.{onnx,data,encodings} in *output_dir*.
    """
    host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls: Any = vision_encoder_cls

    print(f"  Loading {num_calibration_samples} calibration images...")
    calibration_data = cls.get_calibration_data(
        num_calibration_samples, image_height, image_width
    )

    print("  Loading VEG from pretrained...")
    veg_model = cls.from_pretrained(
        device=host_device,
        image_height=image_height,
        image_width=image_width,
    )
    veg_model.eval()

    # Obtain the VEG ONNX graph: reuse the pre-exported one if the
    # export/SpinQuant phase already produced it (so the rotated graph is
    # used), otherwise export a fresh graph from the HF model.
    veg_onnx_path = Path(output_dir) / "vision_encoder.onnx"
    if veg_onnx_path.exists():
        print(f"  Loading pre-exported VEG ONNX from {veg_onnx_path}...")
        veg_onnx = onnx.load(str(veg_onnx_path), load_external_data=True)
    else:
        print("  Exporting VEG to ONNX...")
        veg_onnx = cls.export_to_onnx(veg_model, host_device)

    print("  Creating QuantSim from ONNX...")
    quant_sim, fixed_inputs = cls.create_quantsim_from_onnx(
        veg_onnx, veg_model, host_device
    )

    print(f"  Calibrating with {num_calibration_samples} images...")
    cls.calibrate(quant_sim, calibration_data, fixed_inputs)

    print(f"  Saving VEG to: {output_dir}")
    cls.save_quantized_checkpoint(quant_sim, output_dir)

    del veg_model
    del quant_sim
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("VEG quantization completed successfully.")


def quantize_vlm(
    *,
    quantized_model_cls: type,
    fp_model_cls: type,
    vision_encoder_cls: type,
    supported_precisions: list,
    description: str,
    model_id: str,
    sample_image: CachedWebModelAsset | str,
    default_image_height: int,
    default_image_width: int,
) -> None:
    """Run the VLM quantize flow.

    The shared quantize() function handles export, pre-sim transforms
    (SpinQuant co-rotation), QuantSim creation, and calibration for the
    backbone. VEG quantization runs as a separate step afterward.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length for the model",
    )
    parser.add_argument(
        "--calibration-sequence-length",
        type=int,
        default=DEFAULT_CALIBRATION_SEQ_LEN,
        help="Sequence length to be used during calibration.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to export the ONNX model and encodings.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Input directory with custom weights.",
    )
    parser.add_argument(
        "--use-seq-mse",
        action="store_true",
        default=False,
        help="Add to apply Sequential MSE.",
    )
    parser.add_argument(
        "--use-ada-scale",
        action="store_true",
        default=False,
        help="Add to apply AdaScale.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to be used for calibration.",
    )
    parser.add_argument(
        "--seq-mse-num-samples",
        type=int,
        default=None,
        help="Number of samples for sequential MSE.",
    )
    parser.add_argument(
        "--ada-scale-num-samples",
        type=int,
        default=None,
        help="Number of samples for AdaScale.",
    )
    parser.add_argument(
        "--ada-scale-num-iterations",
        type=int,
        default=None,
        help="Number of iterations for AdaScale.",
    )
    parser.add_argument(
        "--precision",
        default=Precision.parse(supported_precisions[0]),
        action=get_quantize_action_with_default(supported_precisions[0]),
        choices=[str(p) for p in supported_precisions],
        help="Pick the precision with which the model must be quantized.",
    )
    parser.add_argument(
        "--skip-veg",
        action="store_true",
        default=False,
        help="Skip vision encoder (VEG) quantization.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        help="Skip LLM text model quantization.",
    )
    parser.add_argument(
        "--veg-num-samples",
        type=int,
        default=100,
        help="Number of calibration samples for VEG quantization.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(default_image_height, default_image_width),
        help="Image size (height width) used to resize calibration images. "
        "Must match the size the model's input spec is built for.",
    )
    parser.add_argument(
        "--use-spin-quant",
        type=str,
        default=None,
        metavar="PASSES",
        help="Comma-separated SpinQuant passes to apply (r1,r2,r3). "
        "Example: --use-spin-quant r1,r3",
    )

    cli_args = sys.argv[1:]
    args = parser.parse_args(cli_args)

    spinquant_config = _parse_spin_quant_args(args)

    # LLM backbone quantization (export -> SpinQuant -> QuantSim -> calibrate -> save)
    if not args.skip_llm:
        quantize(
            quantized_model_cls=quantized_model_cls,
            fp_model_cls=fp_model_cls,
            context_length=args.context_length,
            seq_len=args.calibration_sequence_length,
            precision=args.precision,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            checkpoint=args.checkpoint,
            use_seq_mse=args.use_seq_mse,
            use_ada_scale=args.use_ada_scale,
            seq_mse_num_samples=args.seq_mse_num_samples,
            ada_scale_num_samples=args.ada_scale_num_samples,
            ada_scale_num_iterations=args.ada_scale_num_iterations,
            image_size=tuple(args.image_size),
            spinquant_config=spinquant_config,
        )
    else:
        print("Skipping LLM quantization as requested.")

    # VEG quantization (loads rotated VEG ONNX from disk if present)
    if not args.skip_veg:
        print()
        print("=" * 60)
        print("Vision Encoder (VEG) Quantization")
        print("=" * 60)
        _quantize_vision_encoder(
            vision_encoder_cls=vision_encoder_cls,
            output_dir=args.output_dir,
            image_height=args.image_size[0],
            image_width=args.image_size[1],
            num_calibration_samples=args.veg_num_samples,
        )
    else:
        print("Skipping VEG quantization as requested.")

    save_command_args(Path(args.output_dir) / "args.json", args, cli_args)

    print()
    print("All quantization completed.")
    print()
    print(
        "    If you are using custom weights via checkpoint folder, please add a copy "
        "of the model config to the output checkpoint folder."
    )
    print()
    fetched_sample_image = (
        sample_image if isinstance(sample_image, str) else sample_image.fetch()
    )
    print("Demo:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.demo "
        f"--checkpoint {args.output_dir} --image {fetched_sample_image} "
        "--prompt 'Describe this image'"
    )
    print()
    print("Export:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.export "
        f"--checkpoint {args.output_dir} --device 'Snapdragon 8 Elite QRD' "
        "--skip-profiling --skip-inferencing --output-dir output"
    )
