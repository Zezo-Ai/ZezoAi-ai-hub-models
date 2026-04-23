# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)

from qai_hub_models.models._shared.llm.demo import llm_chat_demo
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models._shared.qwen2_vl.model import (
    DEFAULT_USER_PROMPT,
    END_TOKENS,
    Qwen2VLTextBase,
)
from qai_hub_models.models.qwen2_5_vl_7b_instruct import MODEL_ID, VisionEncoder
from qai_hub_models.models.qwen2_5_vl_7b_instruct.model import (
    HF_REPO_NAME,
    HF_REPO_URL,
    HIDDEN_SIZE,
    SUPPORTED_PRECISIONS,
    Qwen2_5_VL_7B_PreSplit,
    Qwen2_5_VL_7B_QuantizablePreSplit,
    Qwen2_5_VL_7B_VisionEncoder,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec


def _fp_vlm_demo(prompt: str, image_path: str, max_output_tokens: int) -> None:
    """
    Run FP VLM demo using vanilla HF model.

    The Genie-adapted text model (SHA attention, 1D RoPE) is designed for
    on-device export, not FP inference. For FP demo, we use the vanilla HF
    model with VEG-produced vision embeddings.
    """
    device = torch.device("cpu")

    # Load and resize image to VEG's expected dimensions
    img = Image.open(image_path).convert("RGB")

    # Load VEG and compute vision embeddings
    print("Loading vision encoder (VEG)...")
    veg = VisionEncoder.from_pretrained(device=device)
    img_resized = img.resize((veg._image_width, veg._image_height))
    veg.eval()

    # Load processor and format prompt
    proc = AutoProcessor.from_pretrained(HF_REPO_NAME)
    formatted_text = Qwen2VLTextBase.get_input_prompt_with_tags(
        user_input_prompt=prompt,
        include_image=True,
    )
    processed = proc(
        text=[formatted_text],
        images=[img_resized],
        return_tensors="pt",
    ).to(device)

    pixel_values = processed["pixel_values"]
    input_ids = processed["input_ids"]
    attention_mask = processed["attention_mask"]

    print("Running vision encoder...")
    with torch.no_grad():
        vision_embeddings = veg(pixel_values=pixel_values)
    print(f"  Vision embeddings: {vision_embeddings.shape}")

    # Free VEG memory
    del veg

    # Load vanilla HF model (no Genie adaptations)
    print("Loading HF text model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_REPO_NAME,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    # Merge vision embeddings into text embeddings
    config = AutoConfig.from_pretrained(HF_REPO_NAME)
    image_token_id = config.image_token_id

    with torch.no_grad():
        text_embeddings = model.get_input_embeddings()(input_ids)

    image_mask = input_ids == image_token_id
    image_mask_expanded = image_mask.unsqueeze(-1).expand_as(text_embeddings)
    merged_embeddings = text_embeddings.clone()
    merged_embeddings = merged_embeddings.masked_scatter(
        image_mask_expanded,
        vision_embeddings.to(merged_embeddings.dtype),
    )

    # Generate
    print("Generating response...\n")
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=merged_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_output_tokens,
            do_sample=False,
        )

    tokenizer = proc.tokenizer
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"    + {response}")


def qwen2_5_vl_7b_instruct_chat_demo(
    test_checkpoint: CheckpointSpec | None = None,
) -> None:
    """Run Qwen2.5-VL-7B VLM demo."""
    # Check if this is an FP VLM demo — use vanilla HF model path
    if test_checkpoint is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--checkpoint", default="DEFAULT_UNQUANTIZED")
        parser.add_argument("--image", type=str, default=None)
        parser.add_argument("--prompt", type=str, default=DEFAULT_USER_PROMPT)
        parser.add_argument("--max-output-tokens", type=int, default=1000)
        args, _ = parser.parse_known_args()

        if args.checkpoint == "DEFAULT_UNQUANTIZED" and args.image is not None:
            print(f"\n{'-' * 85}")
            print(f"** Generating response via {MODEL_ID} **")
            print("Variant: FLOATING POINT (PyTorch)")
            print("    This runs the original unquantized model for baseline purposes.")
            print()
            print(f"Prompt: {args.prompt}")
            print(f"Image: {args.image}")
            print(f"Max output tokens: {args.max_output_tokens}")
            print(f"{'-' * 85}\n")
            _fp_vlm_demo(args.prompt, args.image, args.max_output_tokens)
            return

    llm_chat_demo(
        model_cls=Qwen2_5_VL_7B_QuantizablePreSplit,
        fp_model_cls=Qwen2_5_VL_7B_PreSplit,
        qnn_model_cls=LLM_QNN,
        model_id=MODEL_ID,
        end_tokens=END_TOKENS,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_url=HF_REPO_URL,
        default_prompt=DEFAULT_USER_PROMPT,
        supported_precisions=SUPPORTED_PRECISIONS,
        test_checkpoint=test_checkpoint,
        supports_thinking=False,
        # VLM parameters
        vision_encoder_cls=Qwen2_5_VL_7B_VisionEncoder,
        hidden_size=HIDDEN_SIZE,
    )


def main() -> None:
    qwen2_5_vl_7b_instruct_chat_demo()


if __name__ == "__main__":
    main()
