# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
import itertools
import logging
import math
import os
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import numpy as np
import onnx
import onnxruntime
import torch
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.models.qwen3 import modeling_qwen3

# isort: off
# This verifies aimet is installed, and this must be included first.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from aimet_onnx.common.defs import QuantScheme
    from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
# isort: on

from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
)
from qai_hub_models.configs.model_metadata import (
    GenieVisionPreprocessing,
)
from qai_hub_models.models.templates.intern3_5_vl.internvl_processor import (
    get_internvl_compatible_processor,
)
from qai_hub_models.models.templates.intern3_5_vl.vision_encoder import (
    Intern3_5VLVisionEncoder,
    Intern3_5VLVisionWrapper,
)
from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.llm.llm_helpers import (
    get_rope_scaling,
)
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    LLMDynamic_AIMETOnnx,
    LLMPartBase,
    SplitForwardMixin,
)
from qai_hub_models.models.templates.lm_driver.generator import (
    PrecomputedCosSinGeneratorMixin,
    TransposedKVGeneratorMixin,
)
from qai_hub_models.models.templates.lm_driver.internvl import (
    InternVL_VLM_Generator,
)
from qai_hub_models.models.templates.qwen3.model import (
    ORIG_QWEN3_ATTENTION,
    Qwen3PositionProcessor,
)
from qai_hub_models.models.templates.qwen3.model_adaptations import (
    QCQwen3MLP,
    SHAQwen3Attention,
)
from qai_hub_models.models.templates.qwen3_vl.model import (
    Qwen3VLCollectionBase,
    Qwen3VLPreSplitBase,
    Qwen3VLQuantizablePreSplitBase,
    Qwen3VLTextBase,
    Qwen3VLTextBase_AIMETOnnx,
    Qwen3VLTextBase_QNN,
    VisionEncoderCollectionProtocol,
    VisionEncoderExportProtocol,
    _vlm_eval_dataset_classes,
)
from qai_hub_models.models.templates.vlm.model import (
    DEFAULT_IMAGE_SIZE,
    VLMDynamic_AIMETOnnx,
)
from qai_hub_models.models.templates.vlm.processor_factory import VLMProcessorLike
from qai_hub_models.utils.base_multi_graph_model import (
    MultiGraphWorkbenchModel,
)
from qai_hub_models.utils.checkpoint import CheckpointType
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle, mock_torch_onnx_inference

if TYPE_CHECKING:
    from qai_hub.public_rest_api import DatasetEntries

    from qai_hub_models.utils.base_dataset import BaseDataset


logger = logging.getLogger(__name__)

END_TOKENS = {"<|im_end|>", "<|endoftext|>"}

DEFAULT_PROMPT_CONTEXT = "You are a helpful AI assistant."
DEFAULT_USER_PROMPT = "Give me a short introduction to large language model."


def apply_internvl_mm_token_ids(
    tokenizer: Any,
    text_config: Any,
    full_config: Any | None = None,
    image_token: str = "<IMG_CONTEXT>",
    video_token_id: int = -1,
) -> None:
    """Populate InternVL image/video token ids on text and full VLM configs."""
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    if isinstance(image_token_id, int) and image_token_id >= 0:
        text_config.image_token_id = image_token_id
        if full_config is not None:
            full_config.image_token_id = image_token_id

    text_config.video_token_id = video_token_id
    if full_config is not None:
        full_config.video_token_id = video_token_id


class HubCompatibleInternVLVLGenerator(  # type: ignore[misc]
    PrecomputedCosSinGeneratorMixin,
    TransposedKVGeneratorMixin,
    InternVL_VLM_Generator,
):
    pass


def resolve_text_config(llm_config: Any) -> Any:
    """Return the text-config view across supported multimodal config layouts."""
    if getattr(llm_config, "model_type", None) == "qwen3":
        return llm_config
    return (
        getattr(llm_config, "text_config", None)
        or getattr(llm_config, "llm_config", None)
        or getattr(llm_config, "language_config", None)
        or llm_config
    )


def get_vlm_config(model_ckpt: str | os.PathLike | Path | None) -> PretrainedConfig:
    """Construct and return a HuggingFace config for this VLM family."""
    from transformers import AutoConfig

    assert model_ckpt is not None
    print()
    print(f"Loading model config from {model_ckpt}")
    llm_config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    text_config = resolve_text_config(llm_config)
    text_config._attn_implementation = "eager"
    text_config._attn_implementation_internal = "eager"
    text_config.use_cache = True

    return llm_config


class Intern3_5VLTextBase(Qwen3VLTextBase):
    """
    Base class for InternVL text model (Qwen3 text backbone).

    Key differences from Qwen3VLTextBase:
    - Uses LLMIOType.genie_input_embeds
    - Input is embeddings, not token IDs
    - Loads from full VLM checkpoint and extracts text model
    - Handles deepstack visual embeddings injected at intermediate layers
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    GeneratorClass = HubCompatibleInternVLVLGenerator

    # InternVL3.5 text path uses Qwen3ForCausalLM under language_model.
    LMClass = modeling_qwen3.Qwen3ForCausalLM  # type: ignore[assignment, unused-ignore]

    VisionModelWrapper: type[torch.nn.Module] = Intern3_5VLVisionWrapper

    # Store reference to full VLM for embedding extraction
    _full_vlm: torch.nn.Module | None = None

    @classmethod
    def get_visual_output_names(cls, config: PretrainedConfig) -> tuple[str, ...]:
        return ("image_embeddings",)

    @classmethod
    def edit_llm_config(cls, llm_config: PretrainedConfig) -> PretrainedConfig:
        """Extract text config from supported multimodal config layouts."""
        text_cfg = resolve_text_config(llm_config)

        # Keep InternVL vision preprocessing contract explicit for Genie metadata/export.
        text_cfg.temporal_patch_size = 1
        text_cfg.spatial_merge_size = 2
        return text_cfg

    def _apply_monkey_patch(self) -> None:
        self.monkey_patch(skip_optimizations=self.skip_optimizations)

    def _load_tokenizer(self, checkpoint: str | os.PathLike | Path) -> Any:
        from transformers import AutoTokenizer

        # InternVL requires HF remote code tokenizers for special multimodal tokens.
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, is_fast=False, trust_remote_code=True
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.truncation_side = "left"
        return tokenizer

    def _post_configure_tokenizer_and_config(self) -> None:
        apply_internvl_mm_token_ids(
            tokenizer=self.tokenizer,
            text_config=self.llm_config,
            full_config=getattr(self, "_original_llm_config", None),
        )

    def _load_image_processor(self, checkpoint: str | os.PathLike | Path) -> Any:
        from transformers import AutoImageProcessor

        # InternVL's AutoProcessor can instantiate a tokenizer-coupled processor.
        # We only need vision preprocessing metadata in this path.
        image_processor = AutoImageProcessor.from_pretrained(
            checkpoint, trust_remote_code=True
        )
        if image_processor is None:
            hf_repo = getattr(self, "_hf_repo_name", None)
            assert hf_repo is not None, (
                "Checkpoint has no image processor and no _hf_repo_name fallback."
            )
            image_processor = AutoImageProcessor.from_pretrained(
                hf_repo, trust_remote_code=True
            )
        return image_processor

    def _extract_embedding_weights(
        self, full_vlm: torch.nn.Module | None
    ) -> torch.Tensor | None:
        if full_vlm is None:
            return None
        embedding_layer: torch.nn.Module | None = None
        get_input_embeddings = getattr(full_vlm, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            maybe_embedding_layer = get_input_embeddings()
            if isinstance(maybe_embedding_layer, torch.nn.Module):
                embedding_layer = maybe_embedding_layer
        if embedding_layer is None:
            language_model = getattr(full_vlm, "language_model", None)
            language_get_input_embeddings = getattr(
                language_model, "get_input_embeddings", None
            )
            if callable(language_get_input_embeddings):
                maybe_embedding_layer = language_get_input_embeddings()
                if isinstance(maybe_embedding_layer, torch.nn.Module):
                    embedding_layer = maybe_embedding_layer
        if embedding_layer is None:
            return None
        embed_weight = getattr(embedding_layer, "weight", None)
        if not isinstance(embed_weight, torch.Tensor):
            return None
        return embed_weight.detach().clone()

    @classmethod
    def load_llm_from_checkpoint(
        cls,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        load_pretrained: bool = True,
    ) -> tuple[torch.nn.Module, torch.nn.Module | None, torch.nn.Module | None]:
        """Load text model + lm_head from supported multimodal checkpoints.

        Returns (text_model, full_vlm, lm_head). The full_vlm is retained only
        for embedding extraction and released afterwards by callers.
        """
        if load_pretrained:
            full_cfg = get_vlm_config(checkpoint)
            if getattr(full_cfg, "model_type", "") == "internvl_chat":
                from transformers import AutoModel

                if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                    type.__setattr__(PreTrainedModel, "all_tied_weights_keys", {})
                full_vlm = AutoModel.from_pretrained(
                    checkpoint,
                    trust_remote_code=True,
                )
                if not hasattr(full_vlm, "language_model"):
                    raise ValueError(
                        "InternVL checkpoint does not expose `language_model`; "
                        "cannot extract text tower."
                    )
                language_model = full_vlm.language_model
                if not hasattr(language_model, "model") or not hasattr(
                    language_model, "lm_head"
                ):
                    raise ValueError(
                        "InternVL language model missing `model`/`lm_head` attributes."
                    )
                text_model = language_model.model
                lm_head = language_model.lm_head
                return text_model, full_vlm, lm_head

            raise ValueError(
                "intern3_5_vl only supports checkpoints with model_type='internvl_chat'. "
                f"Got model_type='{getattr(full_cfg, 'model_type', None)}'."
            )
        # Create uninitialized text model (Qwen3 text backbone)
        text_model = modeling_qwen3.Qwen3Model(llm_config)  # type: ignore[arg-type, unused-ignore]
        lm_head = torch.nn.Linear(
            llm_config.hidden_size, llm_config.vocab_size, bias=False
        )
        return text_model, None, lm_head

    @classmethod
    def get_image_placeholder_for_processor(
        cls, processor: VLMProcessorLike | None = None
    ) -> str:
        token = getattr(processor, "image_token", None)
        if not token and processor is not None and hasattr(processor, "tokenizer"):
            token = getattr(processor.tokenizer, "context_image_token", None)
        return token or "<IMG_CONTEXT>"

    @classmethod
    def get_processor(cls, hf_repo_name: str) -> VLMProcessorLike:
        from qai_hub_models.models.templates.intern3_5_vl.internvl_processor import (
            get_internvl_compatible_processor,
        )

        return get_internvl_compatible_processor(hf_repo_name)

    @classmethod
    def adapt_prompt_for_processor(
        cls,
        formatted_text: str,
        processor: VLMProcessorLike,
        num_images: int,
    ) -> str:
        image_token = cls.get_image_placeholder_for_processor(processor)
        formatted_text = formatted_text.replace("<image>", image_token)
        placeholder_count = formatted_text.count(image_token)
        if placeholder_count != num_images:
            raise ValueError(
                "InternVL prompt/image mismatch: "
                f"placeholder token={image_token!r}, "
                f"placeholder_count={placeholder_count}, "
                f"num_images={num_images}"
            )
        return formatted_text

    @classmethod
    def configure_generator_config(cls, model: Any, config: Any) -> Any:
        # InternVL full config nests text attributes under `.llm_config` (or
        # `.text_config` for some checkpoints). Flatten needed fields onto the
        # top-level config expected by VLM_Generator.
        text_cfg = resolve_text_config(config)
        if text_cfg is not None:
            for attr in (
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "hidden_size",
                "head_dim",
                "sliding_window",
                "layer_types",
            ):
                if not hasattr(config, attr) and hasattr(text_cfg, attr):
                    setattr(config, attr, getattr(text_cfg, attr))
        tok = getattr(model, "tokenizer", None)
        if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
            apply_internvl_mm_token_ids(tokenizer=tok, text_config=config)
        return config

    def _verify_ckpt(self) -> None:
        """Verify checkpoint is compatible with the InternVL3.5 text pipeline."""
        valid_model_types = {"qwen3", "internvl_chat"}
        architectures = getattr(self.llm_config, "architectures", None) or []
        if not (
            self.llm_config.model_type in valid_model_types
            or any("Qwen3" in arch for arch in architectures)
        ):
            raise ValueError(
                "Model config is not compatible with current text implementation. "
                f"Expected model_type in {valid_model_types}, got '{self.llm_config.model_type}'"
            )

    @staticmethod
    def monkey_patch(skip_optimizations: list[str] | None = None) -> None:
        """
        Apply monkey patches for InternVL3.5 text ONNX export.

        Adaptations applied:
        - SHA + MLP Conv2d on the Qwen3 text backbone used by InternVL3.5.
        """
        from qai_hub_models.models.templates.qwen3.model import Qwen3_Optimizations
        from qai_hub_models.models.templates.qwen3.model_adaptations import (
            QcQwen3_apply_rotary_pos_emb,
        )

        # SHA attention
        if (
            skip_optimizations
            and Qwen3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
            modeling_qwen3.Qwen3Attention = ORIG_QWEN3_ATTENTION  # type: ignore[misc, unused-ignore]
        elif modeling_qwen3.Qwen3Attention is not SHAQwen3Attention:
            modeling_qwen3.Qwen3Attention = SHAQwen3Attention  # type: ignore[misc, unused-ignore]

        # Use externally supplied compact RoPE (cos, sin) via position_ids tuple.
        def bypass_RotaryEmbedding(
            self: modeling_qwen3.Qwen3RotaryEmbedding,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        if not hasattr(modeling_qwen3.Qwen3RotaryEmbedding, "_original_forward"):
            modeling_qwen3.Qwen3RotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                modeling_qwen3.Qwen3RotaryEmbedding.forward
            )
            modeling_qwen3.Qwen3RotaryEmbedding.forward = bypass_RotaryEmbedding
        if modeling_qwen3.apply_rotary_pos_emb is not QcQwen3_apply_rotary_pos_emb:
            modeling_qwen3.apply_rotary_pos_emb = QcQwen3_apply_rotary_pos_emb
        if modeling_qwen3.Qwen3MLP is not QCQwen3MLP:
            modeling_qwen3.Qwen3MLP = QCQwen3MLP  # type: ignore[misc, unused-ignore]


class Intern3_5VLTextBase_AIMETOnnx(Qwen3VLTextBase_AIMETOnnx):
    """
    AIMET-ONNX quantized version of InternVL3.5 text model.

    Uses inputs_embeds instead of input_ids.
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    FPModel = Intern3_5VLTextBase

    get_input_prompt_with_tags = staticmethod(
        Intern3_5VLTextBase.get_input_prompt_with_tags
    )

    def _post_init_vlm_aimet(self) -> None:
        """InternVL-specific AIMET init: ensure image/video token ids are present."""
        apply_internvl_mm_token_ids(
            tokenizer=self.tokenizer,
            text_config=self.llm_config,
            full_config=getattr(self, "_original_llm_config", None),
        )


class InternVLDynamic_AIMETOnnx(VLMDynamic_AIMETOnnx):
    """InternVL-specific VLM dynamic base with custom processor creation."""

    def _load_calibration_vision_model(self) -> torch.nn.Module | None:
        """Load InternVL vision wrapper for multimodal calibration prefill."""
        logger = logging.getLogger(__name__)
        try:
            hf_repo = getattr(self, "_hf_repo_name", None)
            if hf_repo is None and self.checkpoint is not None:
                hf_repo = self.checkpoint
            if hf_repo is None:
                hf_repo = self.llm_config._name_or_path

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                type.__setattr__(PreTrainedModel, "all_tied_weights_keys", {})
            hf_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
            if hasattr(hf_model, "config"):
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_repo, use_fast=True, trust_remote_code=True
                )
                image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                if isinstance(image_token_id, int) and image_token_id >= 0:
                    hf_model.config.image_token_id = image_token_id
                hf_model.config.video_token_id = -1
            if hasattr(hf_model, "vision_model") and not hasattr(
                hf_model.vision_model, "projector"
            ):
                hf_model.vision_model.projector = hf_model.mlp1
                hf_model.vision_model.downsample_ratio = float(
                    getattr(hf_model, "downsample_ratio", 0.5)
                )
                hf_model.vision_model.select_layer = int(
                    getattr(hf_model, "select_layer", -1)
                )
                hf_model.vision_model.ps_version = str(
                    getattr(hf_model, "ps_version", "v2")
                )
            return Intern3_5VLVisionWrapper(hf_model).eval().to(device)
        except Exception as e:
            logger.warning(
                "Failed to load InternVL vision model for calibration; "
                "multimodal samples will use text-only prefill.",
                exc_info=True,
            )
            raise RuntimeError(
                "Failed to load InternVL vision model for calibration; "
                "multimodal samples will use text-only prefill."
            ) from e

    @staticmethod
    def _get_internvl_processor(hf_repo: str) -> VLMProcessorLike:
        return get_internvl_compatible_processor(hf_repo)

    def get_calibration_data(  # type: ignore[override]
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
        sequence_length: int = DEFAULT_CALIBRATION_SEQ_LEN,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        image_size: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
    ) -> DatasetEntries | None:
        from torch.utils.data import DataLoader

        from qai_hub_models.datasets import instantiate_dataset
        from qai_hub_models.datasets.wikitext.interleaved_aokvqa_wikitext import (
            InterleavedAOKVQAWikitext,
        )
        from qai_hub_models.models.templates.llm.generator_factory import make_generator
        from qai_hub_models.utils.base_dataset import DatasetSplit
        from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

        if num_samples == 0:
            num_samples = math.ceil(80000 / context_length)

        hf_repo = getattr(self, "_hf_repo_name", None)
        if hf_repo is None and self.checkpoint is not None:
            hf_repo = self.checkpoint
        if hf_repo is None:
            hf_repo = self.llm_config._name_or_path
        processor = self._get_internvl_processor(hf_repo)

        dataset = instantiate_dataset(
            InterleavedAOKVQAWikitext,
            DatasetSplit.TRAIN,
            input_spec=None,
            tokenizer=self.tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            num_samples=num_samples,
            processor=processor,
            image_size=image_size,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
            image_size=image_size,  # type: ignore[call-arg]
        )
        assert input_spec is not None

        vision_model = self._load_calibration_vision_model()
        generator = make_generator(
            self,
            sequence_length=sequence_length,
            context_length=context_length,
            vision_model=vision_model,
            model_cls=self.FPModel,
        )

        def multimodal_sample_to_kwargs(
            sample: tuple[torch.Tensor, ...], device: torch.device
        ) -> dict[str, torch.Tensor | None]:
            input_ids, attention_mask, *rest = sample
            kwargs: dict[str, torch.Tensor | None] = dict(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            pixel_values = rest[1] if len(rest) > 1 else None
            kwargs["pixel_values"] = (
                pixel_values.to(device) if pixel_values is not None else None
            )
            image_grid_thw = rest[2] if len(rest) > 2 else None
            kwargs["image_grid_thw"] = (
                image_grid_thw.to(device) if image_grid_thw is not None else None
            )
            return kwargs

        inputs = self._prefill_dataset(
            generator,
            dataloader,
            num_inputs=len(input_spec),
            sample_to_kwargs=multimodal_sample_to_kwargs,
            desc="Pre-filling calibration data (Interleaved WikiText/AOKVQA)",
        )
        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))


class Intern3_5VLDynamic_AIMETOnnx(
    InternVLDynamic_AIMETOnnx, Intern3_5VLTextBase_AIMETOnnx
):
    """Dynamic-shape variant of Intern3_5VLTextBase_AIMETOnnx.

    Inherits the VLM calibration / weight-optimization data pipeline from
    VLMDynamic_AIMETOnnx; only model-specific config lives here.
    """

    FPModel = Intern3_5VLTextBase

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return _vlm_eval_dataset_classes()


class Intern3_5VLTextBase_QNN(Qwen3VLTextBase_QNN):
    """QNN version of InternVL3.5 text model."""

    FPModel = Intern3_5VLTextBase


# Re-export position processor
Intern3_5VLPositionProcessor = Qwen3PositionProcessor


# ---------------------------------------------------------------------------
# Intern3_5VLPreSplitBase - FP PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Intern3_5VLPreSplitBase(Qwen3VLPreSplitBase, Intern3_5VLTextBase):
    """FP PreSplit base for InternVL3.5 models.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint. VLM uses split_embedding=False since inputs_embeds
    bypasses the embedding layer.

    Concrete subclasses set the architecture constants below.
    """

    GeneratorClass = HubCompatibleInternVLVLGenerator

    # --- per-model configuration (override in subclass) ---
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict = {}
    default_precision: Precision = Precision.w4a16
    min_memory_recommended: int = 0
    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0
    split_embedding = False

    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    num_deepstack_layers: int = 0

    vision_patch_size: int = 0
    spatial_merge_size: int = 0
    default_num_visual_tokens: int = 0

    _hf_repo_name: str = ""

    def _verify_ckpt(self) -> None:
        super()._verify_ckpt()
        text_config = self.llm_config
        if hasattr(self.llm_config, "text_config"):
            text_config = self.llm_config.text_config
        elif hasattr(self.llm_config, "llm_config"):
            text_config = self.llm_config.llm_config
        if not (
            text_config.num_hidden_layers == self.num_layers
            and text_config.hidden_size == self.hidden_size
            and text_config.num_attention_heads == self.num_attention_heads
            and text_config.num_key_value_heads == self.num_key_value_heads
        ):
            raise ValueError("Model config is not compatible with our implementation.")


Intern3_5VLPreSplitT = TypeVar("Intern3_5VLPreSplitT", bound=Intern3_5VLPreSplitBase)


# ---------------------------------------------------------------------------
# Intern3_5VLQuantizablePreSplitBase - Quantizable PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Intern3_5VLQuantizablePreSplitBase(
    Qwen3VLQuantizablePreSplitBase[Intern3_5VLPreSplitT],
    Generic[Intern3_5VLPreSplitT],
):
    """Quantizable PreSplit base for InternVL3.5 models.

    The S3 asset zip contains the FULL output of quantize.py (dynamic
    ONNX + weights + encodings + tokenizer + config + embedding_weights.raw).

    Concrete subclasses set ``FPModel`` and the config attributes below.
    """

    FPModel: type[Intern3_5VLPreSplitT]
    _hf_repo_name: str = ""

    # DynamicQuantizablePreSplitMixin config
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict = {}
    supported_precisions: list[Precision] = []
    default_precision: Precision = Precision.w4a16

    # DynamicPreSplitOnnxMixin config
    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0
    split_embedding = False

    num_layers: int = 0

    # SHA produces per-head q_norm/k_norm nodes in the ONNX graph.
    # Between block starts (input_layernorm): num_attention_heads q_norms
    # + num_key_value_heads k_norms + 1 post_attention_layernorm intermediate ops
    ada_scale_num_rmsnorm_per_blk: int | None = None

    # VLM: vision encoder class (set by leaf classes)
    vision_encoder_cls: type[VisionEncoderExportProtocol] | None = None

    # Keep InternVL-specific calibration behavior while reusing Qwen3VL
    # quantizable presplit plumbing.
    _load_calibration_vision_model = (
        InternVLDynamic_AIMETOnnx._load_calibration_vision_model
    )
    _get_internvl_processor = staticmethod(
        InternVLDynamic_AIMETOnnx._get_internvl_processor
    )

    def get_calibration_data(self, *args: Any, **kwargs: Any) -> DatasetEntries | None:
        """Compatibility wrapper for both AIMET and VLM calibration signatures."""
        input_spec = cast(InputSpec | None, kwargs.pop("input_spec", None))
        num_samples_opt = cast(int | None, kwargs.pop("num_samples", None))
        sequence_length = cast(
            int, kwargs.pop("sequence_length", DEFAULT_CALIBRATION_SEQ_LEN)
        )
        context_length = cast(int, kwargs.pop("context_length", DEFAULT_CONTEXT_LENGTH))
        image_size = cast(
            tuple[int, int] | None, kwargs.pop("image_size", DEFAULT_IMAGE_SIZE)
        )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        # Support both:
        # 1) (input_spec, num_samples) from AIMETOnnxQuantizableMixin
        # 2) (num_samples, input_spec, sequence_length, context_length, image_size)
        #    from VLMDynamic_AIMETOnnx
        if args:
            if isinstance(args[0], dict) or args[0] is None:
                input_spec = cast(InputSpec | None, args[0])
                if len(args) >= 2:
                    num_samples_opt = cast(int | None, args[1])
                if len(args) > 2:
                    raise TypeError(
                        "AIMET signature accepts at most 2 positional arguments."
                    )
            else:
                num_samples_opt = cast(int | None, args[0])
                if len(args) >= 2:
                    input_spec = cast(InputSpec | None, args[1])
                if len(args) >= 3:
                    sequence_length = cast(int, args[2])
                if len(args) >= 4:
                    context_length = cast(int, args[3])
                if len(args) >= 5:
                    image_size = cast(tuple[int, int] | None, args[4])
                if len(args) > 5:
                    raise TypeError(
                        "VLM signature accepts at most 5 positional arguments."
                    )

        internvl_self = cast(InternVLDynamic_AIMETOnnx, self)
        return internvl_self.get_calibration_data(
            num_samples=0 if num_samples_opt is None else num_samples_opt,
            input_spec=input_spec,
            sequence_length=sequence_length,
            context_length=context_length,
            image_size=image_size,
        )


# ---------------------------------------------------------------------------
# Vision Encoder Component
# ---------------------------------------------------------------------------


class Intern3_5VLVisionEncoderBase(Intern3_5VLVisionEncoder):
    """Vision encoder base for InternVL3.5 (adapted VEG for on-device deployment).

    Returns multiple outputs: image_embeddings + deepstack features.
    Supports both FP inference and quantized inference (via AIMET-ONNX QuantSim).

    Concrete subclasses set ``_hf_repo_name``, ``vision_patch_size``,
    ``vision_hidden_size``, ``vision_num_heads``,
    ``quant_presplit_cls``, ``default_image_height``/``default_image_width`` and
    ``DEFAULT_IMAGE_SIZE``.
    """

    _hf_repo_name: str = ""
    vision_patch_size: int = 0
    # Vision-tower attention dims, used to derive the RoPE embedding width
    # (rope_dim = (vision_hidden_size // vision_num_heads) // 2). These differ
    # per model (e.g. 8B: 1152/16 -> head_dim 72 -> rope_dim 36; 4B: 1280/16 ->
    # head_dim 80 -> rope_dim 40), so they must be set per subclass rather than
    # relying on a hardcoded default.
    vision_hidden_size: int = 0
    vision_num_heads: int = 0
    default_image_height: int = 0
    default_image_width: int = 0
    # Set by subclass to the model's Quantizable PreSplit class.
    quant_presplit_cls: type[Intern3_5VLQuantizablePreSplitBase]

    @classmethod
    def vision_rope_dim(cls) -> int:
        """RoPE embedding width for the vision tower = head_dim // 2.

        Derived from config-backed class attributes so each model variant gets
        the correct width (vs. a hardcoded default that only matched one model).
        """
        if not cls.vision_hidden_size or not cls.vision_num_heads:
            raise ValueError(
                f"{cls.__name__} must set vision_hidden_size and vision_num_heads "
                f"to derive the vision RoPE dim."
            )
        head_dim = cls.vision_hidden_size // cls.vision_num_heads
        return head_dim // 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._checkpoint: str | None = None
        self._precision: Precision = Precision.float
        self._quantized_session: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        device: torch.device | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        precision: Precision = Precision.float,
        **kwargs: Any,
    ) -> Self:
        if image_height is None:
            image_height = cls.default_image_height
        if image_width is None:
            image_width = cls.default_image_width
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if precision != Precision.float and (
            isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
        ):
            checkpoint = cls.quant_presplit_cls.fetch_default_checkpoint(precision)

        load_device = device if precision == Precision.float else torch.device("cpu")
        instance: Self = super().from_pretrained(  # type: ignore[assignment]
            checkpoint=cls._hf_repo_name,
            device=load_device,
            image_height=image_height,
            image_width=image_width,
        )
        instance._checkpoint = str(checkpoint)
        instance._precision = precision

        if precision != Precision.float:
            instance._init_quantized_session(Path(str(checkpoint)), device)

        return instance

    def _init_quantized_session(
        self,
        ckpt_path: Path,
        device: torch.device,
    ) -> None:
        """Create an AIMET-ONNX QuantSim session for quantized inference."""
        veg_onnx = ckpt_path / "vision_encoder.onnx"
        veg_enc = ckpt_path / "vision_encoder.encodings"

        onnx_model = onnx.load(str(veg_onnx), load_external_data=True)

        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")

        quant_logger = logging.getLogger("Quant")
        prev_level = quant_logger.level
        quant_logger.setLevel(logging.WARNING)
        try:
            quant_sim = QuantizationSimModel(
                model=onnx_model,
                quant_scheme=QuantScheme.min_max,
                param_type="int8",
                activation_type="int16",
                providers=providers,
            )
            if veg_enc.exists():
                load_encodings_to_sim(quant_sim, str(veg_enc), strict=False)
        finally:
            quant_logger.setLevel(prev_level)

        self._quantized_session = quant_sim

    def component_precision(self) -> Precision:
        return self._precision

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids_cos: torch.Tensor | None = None,
        position_ids_sin: torch.Tensor | None = None,
        window_attention_mask: torch.Tensor | None = None,
        full_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self._is_quantized:
            return self._forward_quantized(pixel_values)
        return super().forward(
            pixel_values=pixel_values,
            position_ids_cos=position_ids_cos,
            position_ids_sin=position_ids_sin,
            window_attention_mask=window_attention_mask,
            full_attention_mask=full_attention_mask,
        )

    def _forward_quantized(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run inference through the AIMET-ONNX QuantSim session."""
        assert self._quantized_session is not None
        results = mock_torch_onnx_inference(
            self._quantized_session.session,
            pixel_values,
            cast(torch.Tensor, self._pos_emb_cos),
            cast(torch.Tensor, self._pos_emb_sin),
            cast(torch.Tensor, self._window_attention_mask),
            cast(torch.Tensor, self._full_attention_mask),
        )
        if isinstance(results, torch.Tensor):
            return (results,)
        return tuple(results)

    def get_input_spec(
        self,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> InputSpec:
        if image_height is None:
            image_height = self.default_image_height
        if image_width is None:
            image_width = self.default_image_width
        return self.get_static_input_spec(image_height, image_width)

    @classmethod
    def get_static_input_spec(  # type: ignore[override]
        cls,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> InputSpec:
        if image_height is None:
            image_height = cls.default_image_height
        if image_width is None:
            image_width = cls.default_image_width
        return Intern3_5VLVisionEncoder.get_static_input_spec(
            image_height=image_height,
            image_width=image_width,
            patch_size=cls.vision_patch_size,
            rope_dim=cls.vision_rope_dim(),
        )

    def _get_onnx_bundle(self) -> ONNXBundle:
        if self._checkpoint is None:
            raise ValueError("No checkpoint provided for VisionEncoder.")
        ckpt = Path(self._checkpoint)
        return ONNXBundle(
            bundle_path=ckpt,
            onnx_graph_name="vision_encoder.onnx",
            onnx_weights_name="vision_encoder.data"
            if (ckpt / "vision_encoder.data").exists()
            else None,
            aimet_encodings_name="vision_encoder.encodings"
            if (ckpt / "vision_encoder.encodings").exists()
            else None,
        )

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        out_dir = Path(output_dir) / f"{model_name}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )
        return out_dir

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        spec = input_spec or self.get_input_spec()
        result: SampleInputsType = {}
        for name, (shape, dtype_str) in spec.items():
            np_dtype = np.float32 if dtype_str == "float32" else np.int64
            result[name] = [np.zeros(shape, dtype=np_dtype)]
        return result


# ---------------------------------------------------------------------------
# Unified Part Base
# ---------------------------------------------------------------------------


class Intern3_5VLPartBase(LLMPartBase, torch.nn.Module, MultiGraphWorkbenchModel):
    """Unified Part base: handles both FP and Quantizable modes based on precision.

    Spec derivation is inherited from ``LLMPartBase`` (head_dim attribute +
    ``_extra_graph_inputs`` hook); this class carries the family deploy/session
    plumbing (mirroring ``LlamaPartBase`` for text LLMs) plus the qwen3
    architecture constants and the deepstack graph-input override.

    Concrete subclasses set the architecture constants, the FP / Quantizable
    PreSplit classes, the export length lists, and ``part_id``.
    """

    # Architecture dims (LLMPartBase attribute names; head_dim is explicit
    # because hidden_size / num_attention_heads may not equal head_dim).
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    num_splits: int = 0
    num_deepstack_layers: int = 0
    default_precision: Precision = Precision.w4a16
    default_num_visual_tokens: int = 0
    part_id: int = 0

    # Set by subclass.
    fp_presplit_cls: type[Intern3_5VLPreSplitBase]
    quant_presplit_cls: type[Intern3_5VLQuantizablePreSplitBase]
    export_sequence_lengths: list[int] = []
    export_context_lengths: list[int] = []

    def __init__(
        self,
        presplit: Any,
        precision: Precision | None = None,
    ) -> None:
        super().__init__()
        if precision is None:
            precision = self.default_precision
        self._presplit = presplit
        self._precision = precision
        self._quant_sim: QuantizationSimModel | None = None
        self._fp_session: onnxruntime.InferenceSession | None = None
        self._graph_names: dict[str, tuple[int, int]] = {
            f"ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{self.num_splits}": (
                seq_len,
                ctx_len,
            )
            for seq_len, ctx_len in itertools.product(
                self.export_sequence_lengths, self.export_context_lengths
            )
        }

    @property
    def shared_source_model(self) -> bool:
        return True

    @property
    def graph_names(self) -> list[str]:
        return list(self._graph_names.keys())

    def component_precision(self) -> Precision:
        return self._precision

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create Part by getting or creating the appropriate PreSplit (cached)."""
        checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
        if not checkpoint_type.is_aimet_onnx():
            presplit: Any = cls.fp_presplit_cls.from_pretrained(
                host_device=host_device,
            )
            precision = Precision.float
        else:
            precision = checkpoint_type.precision(
                cls.default_precision, checkpoint=checkpoint
            )
            presplit = cls.quant_presplit_cls.from_pretrained(
                precision=precision,
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
            )
        return cls(presplit, precision=precision)

    def _extra_graph_inputs(
        self, name: str, sequence_length: int, context_length: int
    ) -> TensorSpec | None:
        # Deepstack-specific inputs (qwen3 VL only).
        if name == "visual_pos_masks":
            return TensorSpec(shape=(1, sequence_length), dtype="bool")
        if name.startswith("deepstack_visual_embeds_"):
            return TensorSpec(
                shape=(self.default_num_visual_tokens, self.hidden_size),
                dtype="float32",
            )
        return None

    def _get_onnx_input_names(self) -> list[str]:
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [i.name for i in onnx_model.graph.input]

    def _get_onnx_output_names(self) -> list[str]:
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [o.name for o in onnx_model.graph.output]

    def _get_onnx_bundle(self) -> ONNXBundle:
        return self._presplit.convert_to_onnx_and_split(part_id=self.part_id)

    def forward(
        self, *args: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor | Collection[torch.Tensor]:
        if self._is_quantized:
            quant_sim = self._get_quant_sim()
            return mock_torch_onnx_inference(quant_sim.session, *args, **kwargs)
        session = self._get_fp_session()
        return mock_torch_onnx_inference(session, *args, **kwargs)

    def _get_quant_sim(self) -> QuantizationSimModel:
        if self._quant_sim is not None:
            return self._quant_sim

        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=True
        )
        onnx_model.ir_version = min(onnx_model.ir_version, 11)

        assert isinstance(self._presplit, self.quant_presplit_cls)
        _hd = self._presplit.host_device
        host_device = _hd if isinstance(_hd, torch.device) else torch.device("cpu")
        providers = self._presplit.get_ort_providers(host_device)

        self._quant_sim = LLMDynamic_AIMETOnnx._build_quantsim(onnx_model, providers)
        LLMDynamic_AIMETOnnx._apply_precision_activations(
            self._quant_sim, self._precision
        )

        if onnx_bundle.aimet_encodings_path is not None:
            load_encodings_to_sim(
                self._quant_sim,
                str(onnx_bundle.aimet_encodings_path),
                strict=False,
            )

        return self._quant_sim

    def _get_fp_session(self) -> onnxruntime.InferenceSession:
        if self._fp_session is not None:
            return self._fp_session

        onnx_bundle = self._get_onnx_bundle()
        providers: list[str] = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        onnx_path = str(onnx_bundle.onnx_graph_path)
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        if onnx_model.ir_version > 10:
            onnx_model.ir_version = 10
            onnx.save(onnx_model, onnx_path)

        self._fp_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        return self._fp_session

    def serialize_graph(
        self,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        precision_suffix = f"_{self._precision}" if self._is_quantized else ""
        out_dir = Path(output_dir) / f"{model_name}{precision_suffix}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )

        return out_dir


# ---------------------------------------------------------------------------
# Split-Forward Mixin (for ONNX-based evaluation)
# ---------------------------------------------------------------------------


class Intern3_5VLSplitForwardMixin(SplitForwardMixin):
    """Split-forward mixin for InternVL3.5 eval via split Parts.

    Concrete wrappers set ``split_part_classes`` and ``default_num_visual_tokens``.
    """

    split_part_classes: list[type] = []
    default_num_visual_tokens: int = 0

    def get_split_part_classes(self) -> list[type]:
        return self.split_part_classes

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self._exporting_onnx or torch.compiler.is_compiling():
            return super(SplitForwardMixin, self).forward(  # type: ignore[misc]
                input_tokens, attention_mask, *args
            )
        self._ensure_parts()
        assert self._parts is not None
        assert self._input_names_for_parts is not None

        full_names = list(
            self.get_input_spec(  # type: ignore[attr-defined]
                sequence_length=DEFAULT_SEQUENCE_LENGTH,
                context_length=DEFAULT_CONTEXT_LENGTH,
            ).keys()
        )
        # Total positional args = input_tokens + attention_mask + *args
        num_provided = 2 + len(args)
        num_expected = len(full_names)

        # Pad missing deepstack inputs with zeros using actual runtime shapes.
        # visual_pos_masks=0 means no visual tokens, so deepstack is a no-op.
        if num_provided < num_expected:
            actual_seq_len = input_tokens.shape[1]
            device = input_tokens.device
            extra = []
            for name in full_names[num_provided:]:
                if name == "visual_pos_masks":
                    extra.append(
                        torch.zeros(1, actual_seq_len, dtype=torch.bool, device=device)
                    )
                elif name.startswith("deepstack_visual_embeds_"):
                    extra.append(
                        torch.zeros(1, self.default_num_visual_tokens, device=device)
                    )
                else:
                    extra.append(torch.zeros(1, device=device))
            args = (*args, *extra)

        return self._split_forward(
            self._parts,
            self._input_names_for_parts,
            input_tokens,
            attention_mask,
            *args,
            output_names_for_parts=self._output_names_for_parts,
        )


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class Intern3_5VLCollectionBase(Qwen3VLCollectionBase):
    """Collection base for InternVL3.5 deployment.

    Combines N text parts + 1 vision encoder for full VLM deployment.

    Concrete subclasses set ``_hf_repo_name``, the PreSplit / Part / VisionEncoder
    classes, the image/deepstack/patch constants, ``sample_image``, and the
    ordered ``parts`` mapping.
    """

    _checkpoint: str

    # Set by subclass.
    _hf_repo_name: str = ""
    fp_presplit_cls: type[Intern3_5VLPreSplitBase]
    quant_presplit_cls: type[Intern3_5VLQuantizablePreSplitBase]
    part_base_cls: type[LLMPartBase]
    vision_encoder_cls: type[VisionEncoderCollectionProtocol]
    num_deepstack_layers: int = 0
    vision_patch_size: int = 0
    default_image_height: int = 0
    default_image_width: int = 0
    default_precision: Precision = Precision.w4a16
    sample_image: Any = None
    parts: dict[str, type] = {}

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        **kwargs: Any,
    ) -> Self:
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            type.__setattr__(PreTrainedModel, "all_tied_weights_keys", {})
        return super().from_pretrained(
            checkpoint=checkpoint, host_device=host_device, **kwargs
        )

    @staticmethod
    def _canonical_io_name(name: str) -> str:
        return name.replace("/", "_").replace(".", "_")

    def remap_metadata_input_spec(
        self,
        component_name: str,
        graph_input_spec: InputSpec,
        compiled_input_names: set[str],
    ) -> InputSpec:
        """Remap InternVL input names to compiled metadata names when needed.

        ONNX-compiled metadata may sanitize names ("/" and "." -> "_") while
        graph input specs keep original names. Remap only unambiguous matches.
        """
        remapped: InputSpec = {}
        for input_name, spec in graph_input_spec.items():
            if input_name in compiled_input_names:
                remapped[input_name] = spec
                continue

            canonical = self._canonical_io_name(input_name)
            candidates = [
                candidate
                for candidate in compiled_input_names
                if self._canonical_io_name(candidate) == canonical
            ]
            if len(candidates) == 1:
                remapped[candidates[0]] = spec
            else:
                # Preserve strict merge behavior for unresolved/ambiguous names.
                remapped[input_name] = spec
        return remapped

    def _get_collection_processor(self) -> VLMProcessorLike:
        return get_internvl_compatible_processor(self._hf_repo_name)

    def _get_text_config_from_llm_config(self, llm_config: Any) -> Any:
        return resolve_text_config(llm_config)

    def _build_vlm_rope_config(
        self, text_config: Any, image_processor: Any
    ) -> dict[str, Any] | None:
        # Keep InternVL3.5 text generator on the standard rope config path.
        _ = get_rope_scaling(text_config)
        return None

    def _get_chat_template_spec(self) -> dict[str, str]:
        return Intern3_5VLTextBase.get_chat_template()

    def _resolve_vision_patch_size(self, image_processor: Any, llm_config: Any) -> int:
        vision_config = getattr(llm_config, "vision_config", None)
        assert vision_config is not None, "llm_config is missing vision_config"
        if vision_config is not None and hasattr(vision_config, "patch_size"):
            return int(vision_config.patch_size)
        return int(image_processor.patch_size)

    def _build_vision_preprocessing(
        self, image_processor: Any, llm_config: Any
    ) -> GenieVisionPreprocessing:
        text_cfg = Intern3_5VLTextBase.edit_llm_config(llm_config)
        return GenieVisionPreprocessing(
            image_width=self.default_image_width,
            image_height=self.default_image_height,
            patch_size=self._resolve_vision_patch_size(image_processor, llm_config),
            temporal_patch_size=int(text_cfg.temporal_patch_size),
            spatial_merge_size=int(text_cfg.spatial_merge_size),
            normalize_mean=image_processor.image_mean,
            normalize_std=image_processor.image_std,
        )

    def _get_veg_prompt_tokenizer(
        self, processor: VLMProcessorLike, tokenizer: Any
    ) -> Any:
        return getattr(processor, "tokenizer", tokenizer)

    def _adapt_dummy_veg_prompt(
        self,
        formatted_text: str,
        processor: VLMProcessorLike,
        num_images: int,
    ) -> str:
        return self.fp_presplit_cls.adapt_prompt_for_processor(
            formatted_text=formatted_text,
            processor=processor,
            num_images=num_images,
        )

    def _build_sample_veg_prompt_files(
        self,
        processor: VLMProcessorLike,
    ) -> tuple[str, str]:
        tok = getattr(processor, "tokenizer", None)
        start_image_token = getattr(tok, "start_image_token", "<img>")
        end_image_token = getattr(tok, "end_image_token", "</img>")

        prompt_prefix = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{start_image_token}"
        )
        prompt_suffix = (
            f"{end_image_token}Describe the image.<|im_end|>\n<|im_start|>assistant\n"
        )
        return prompt_prefix, prompt_suffix
