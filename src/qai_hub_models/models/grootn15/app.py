# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from types import GeneratorType
from typing import Any, cast

import numpy as np
import qai_hub as hub
import torch
from transformers import PreTrainedTokenizerBase

from qai_hub_models.models.grootn15.dataset import GrootDataset
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.dataset import (
    LeRobotSingleDataset,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.policy import (
    Gr00tPolicy,
    unsqueeze_dict_values,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.model.transforms import (
    GR00TTransform,
)
from qai_hub_models.models.grootn15.model import (
    DEFAULT_DATASET_ASSET,
    GrootQuantizedCollection,
    compute_vlm_seq_len,
    load_checkpoint,
    preprocess_dit,
    preprocess_llm,
    preprocess_vlm_proj,
)
from qai_hub_models.models.grootn15.model import GrootCollection as Model
from qai_hub_models.models.grootn15.utils import (
    ComponentIOSink,
    DataSplit,
)
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.evaluate.helpers import EvalMode
from qai_hub_models.utils.inference import (
    AsyncOnDeviceModel,
    AsyncOnDeviceResult,
    OnDeviceModel,
)


def _split_to_list(t: torch.Tensor, B: int) -> list[torch.Tensor]:
    """Split (B, ...) tensor into list of B tensors each (1, ...)"""
    return [t[i : i + 1] for i in range(B)]


class _PostprocessedAsyncResult(AsyncOnDeviceResult):
    """Wraps AsyncOnDeviceResult; applies postprocess_fn lazily in wait()."""

    def __init__(
        self,
        inner: AsyncOnDeviceResult,
        postprocess_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.__dict__ = inner.__dict__.copy()
        self._postprocess_fn = postprocess_fn

    def wait(self) -> torch.Tensor:
        raw = super().wait()
        raw_tensor = raw[0] if isinstance(raw, tuple) else raw
        return self._postprocess_fn(raw_tensor)


def _wait(
    result: torch.Tensor | tuple[torch.Tensor, ...] | AsyncOnDeviceResult,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Resolve an AsyncOnDeviceResult to tensors; pass through tensors unchanged."""
    if isinstance(result, AsyncOnDeviceResult):
        return result.wait()
    return result


def _call_component(
    component: ExecutableModelProtocol,
    *tensors: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, ...] | AsyncOnDeviceResult:
    """Submit a component call. Callers are responsible for calling .wait() on AsyncOnDeviceResult."""
    if isinstance(component, AsyncOnDeviceModel):
        B = tensors[0].shape[0]
        return component(*[_split_to_list(t, B) for t in tensors])  # type: ignore[arg-type]
    if isinstance(component, OnDeviceModel):
        device = tensors[0].device
        B = tensors[0].shape[0]
        result = component(*[_split_to_list(t, B) for t in tensors])  # type: ignore[arg-type]
        if isinstance(result, torch.Tensor):
            return result.to(device)
        if isinstance(result, tuple):
            return tuple(t.to(device) for t in result)
        return result
    return component(*tensors)


@dataclass
class GrootAppConfig:
    """
    Lightweight configuration for GrootApp, extracted from Gr00tPolicy
    once at build time to avoid storing Gr00tPolicy at inference time.
    """

    # Action config
    action_horizon: int
    action_dim: int

    # Padding
    vlm_seq_len: int
    pad_token_id: int

    # preprocess_llm
    image_token_index: int
    llm_embedding_weight: torch.Tensor

    # preprocess_vlm_proj
    vlm_proj_num_heads: int

    # preprocess_dit
    dit_num_heads: int
    num_target_vision_tokens: int

    # Transforms
    modality_transform: Any
    modality_config: dict
    dtype: torch.dtype
    modality_keys: list[str] = field(default_factory=list)

    @staticmethod
    def from_policy(policy: Gr00tPolicy) -> GrootAppConfig:
        """Extract all scalars and callables from a loaded Gr00tPolicy."""
        eagle_model = policy.model.backbone.eagle_model
        action_head = policy.model.action_head
        vl_self_attention = action_head.vl_self_attention

        transform = policy._modality_transform.transforms[-1]
        if not isinstance(transform, GR00TTransform):
            raise TypeError("GR00TTransform not found in modality_transform pipeline.")

        eagle_processor = cast(Any, transform.eagle_processor)
        tokenizer = cast(PreTrainedTokenizerBase, eagle_processor.tokenizer)

        return GrootAppConfig(
            action_horizon=policy.model.action_horizon,
            action_dim=policy.model.action_dim,
            vlm_seq_len=compute_vlm_seq_len(policy),
            pad_token_id=tokenizer.pad_token_id,
            image_token_index=eagle_model.image_token_index,
            llm_embedding_weight=eagle_model.language_model.get_input_embeddings()
            .weight.detach()
            .clone(),
            vlm_proj_num_heads=vl_self_attention.transformer_blocks[0].attn1.heads,
            dit_num_heads=action_head.model.transformer_blocks[0].attn1.heads,
            num_target_vision_tokens=action_head.config.num_target_vision_tokens,
            modality_transform=policy._modality_transform,
            modality_config=policy.modality_config,
            modality_keys=[
                k.split(".")[-1] for k in policy.modality_config["action"].modality_keys
            ],
            dtype=policy.model.action_head.dtype,
        )


class GrootApp:
    """
    Assembles GrootCollection components to reproduce Gr00tPolicy
    inference (ViT -> LLM -> VLMProjection -> DiT diffusion loop).
    """

    def __init__(
        self,
        config: GrootAppConfig,
        vit: ExecutableModelProtocol,
        llm: ExecutableModelProtocol,
        vlm_proj: ExecutableModelProtocol,
        dit: ExecutableModelProtocol,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.vit = vit
        self.llm = llm
        self.vlm_proj = vlm_proj
        self.dit = dit
        self.device = device

    @property
    def uses_ondevice_model(self) -> bool:
        return any(
            isinstance(m, AsyncOnDeviceModel)
            for m in (self.vit, self.llm, self.vlm_proj, self.dit)
        )

    def _preprocess(
        self,
        step_data: dict[str, Any] | list[dict[str, Any]] | tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply modality_transform per sample and prepare model-ready tensors.

        Accepts:
          - dict / list[dict]: raw step dicts from get_step_data (demo/calibration path)
          - tuple[Tensor, ...]: batched field tensors from DataLoader (eval path);
            ordered per GrootDataset.get_field_keys(), language encoded as uint8.

        Returns (pixel_values, input_ids, attention_mask, state, actions).
        """
        if isinstance(step_data, tuple) and all(
            isinstance(v, torch.Tensor) for v in step_data
        ):
            # Eval path: reconstruct per-sample step dicts from batched tensors
            field_keys = GrootDataset.get_field_keys()
            B = step_data[0].shape[0]
            step_dicts: list[dict[str, Any]] = []
            for i in range(B):
                sample: dict[str, Any] = {}
                for key, batched in zip(field_keys, step_data, strict=False):
                    item = batched[i]
                    if item.dtype == torch.uint8 and item.ndim == 2:
                        # Language: decode uint8 bytes -> list[str]
                        sample[key] = [
                            bytes(row[row != 0].tolist()).decode("utf-8")
                            for row in item
                        ]
                    else:
                        sample[key] = item.numpy()
                step_dicts.append(sample)
        else:
            step_dicts = [step_data] if isinstance(step_data, dict) else list(step_data)  # type: ignore[arg-type]

        per_sample = []
        for sd in step_dicts:
            sd = unsqueeze_dict_values(sd)
            inputs = self.config.modality_transform(sd)

            eagle_prefix = "eagle_"
            inputs = {
                (k.removeprefix(eagle_prefix) if k.startswith(eagle_prefix) else k): v
                for k, v in inputs.items()
            }
            inputs.pop("image_sizes", None)
            inputs.pop("action", None)

            cur_len = inputs["input_ids"].shape[1]
            pad_len = self.config.vlm_seq_len - cur_len
            if pad_len > 0:
                inputs["input_ids"] = torch.cat(
                    [
                        inputs["input_ids"],
                        torch.full(
                            (1, pad_len),
                            self.config.pad_token_id,
                            dtype=inputs["input_ids"].dtype,
                        ),
                    ],
                    dim=-1,
                )
                inputs["attention_mask"] = torch.cat(
                    [
                        inputs["attention_mask"].to(torch.int32),
                        torch.zeros((1, pad_len), dtype=torch.int32),
                    ],
                    dim=-1,
                )
            per_sample.append(inputs)

        dtype = self.config.dtype
        pixel_values = torch.cat([s["pixel_values"] for s in per_sample], dim=0).to(
            dtype
        )
        input_ids = torch.cat([s["input_ids"] for s in per_sample], dim=0)
        attention_mask = torch.cat([s["attention_mask"] for s in per_sample], dim=0)
        state = torch.cat([s["state"] for s in per_sample], dim=0).to(dtype)
        B = pixel_values.shape[0]
        actions = torch.randn(
            (B, self.config.action_horizon, self.config.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        return (
            pixel_values.to(self.device),
            input_ids.to(self.device),
            attention_mask.to(self.device),
            state.to(self.device),
            actions,
        )

    def _postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        """Unnormalize raw DiT outputs back to action space. Returns (B, H, total_dof)."""
        result = self.config.modality_transform.unapply(
            {"action": outputs.float().cpu()}
        )
        parts = [
            torch.as_tensor(np.asarray(result[f"action.{k}"]), dtype=torch.float32)
            for k in self.config.modality_keys
        ]
        return torch.cat(parts, dim=-1)  # (B, H, total_dof)

    def encode_vlm(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        component_io_sink: ComponentIOSink | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run ViT -> LLM -> VLMProjection to produce cross-attention KV pairs."""
        vit_embeds = _wait(_call_component(self.vit, pixel_values))
        assert isinstance(vit_embeds, torch.Tensor)
        vit_embeds = vit_embeds.to(pixel_values.device)

        if component_io_sink is not None:
            component_io_sink.dump(
                "vit",
                (pixel_values,),
                (vit_embeds,),
                list(self.vit.get_input_spec().keys()),  # type: ignore[attr-defined]
                cast(Any, self.vit).get_output_names(),
            )

        input_embeds, llm_attention_mask = preprocess_llm(
            input_ids,
            vit_embeds,
            attention_mask,
            self.config.llm_embedding_weight.to(input_ids.device),
            self.config.image_token_index,
        )
        vlm_embeds = _wait(_call_component(self.llm, input_embeds, llm_attention_mask))
        assert isinstance(vlm_embeds, torch.Tensor)
        vlm_embeds = vlm_embeds.to(input_embeds.device)

        if component_io_sink is not None:
            component_io_sink.dump(
                "llm",
                (input_embeds, llm_attention_mask),
                (vlm_embeds,),
                list(self.llm.get_input_spec().keys()),  # type: ignore[attr-defined]
                cast(Any, self.llm).get_output_names(),
            )

        vlm_attention_mask = preprocess_vlm_proj(
            attention_mask,
            self.config.vlm_proj_num_heads,
        )
        vlm_proj_out = _wait(
            _call_component(self.vlm_proj, vlm_embeds, vlm_attention_mask)
        )
        assert isinstance(vlm_proj_out, tuple)
        vlm_proj_out = tuple(t.to(vlm_embeds.device) for t in vlm_proj_out)

        if component_io_sink is not None:
            component_io_sink.dump(
                "vlm_proj",
                (vlm_embeds, vlm_attention_mask),
                vlm_proj_out,
                list(self.vlm_proj.get_input_spec().keys()),  # type: ignore[attr-defined]
                cast(Any, self.vlm_proj).get_output_names(),
            )

        return vlm_proj_out

    def denoise_steps(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        vlm_proj_kv_flat: tuple[torch.Tensor, ...],
        cross_attention_mask: torch.Tensor,
        component_io_sink: ComponentIOSink | None = None,
    ) -> torch.Tensor:
        """Run DiT denoising steps."""
        cross_attention_mask = preprocess_dit(
            cross_attention_mask,
            self.config.dit_num_heads,
            self.config.action_horizon,
            self.config.num_target_vision_tokens,
        )

        dit_inputs_tensors: tuple[torch.Tensor, ...] = (
            state,
            actions,
            cross_attention_mask,
            *vlm_proj_kv_flat,
        )

        actions_out = _call_component(self.dit, *dit_inputs_tensors)
        assert isinstance(actions_out, torch.Tensor)

        if component_io_sink is not None:
            component_io_sink.dump(
                "dit",
                dit_inputs_tensors,
                (actions_out,),
                list(self.dit.get_input_spec().keys()),  # type: ignore[attr-defined]
                cast(Any, self.dit).get_output_names(),
            )

        return actions_out

    def predict_action_chunk(
        self,
        step_data: dict[str, Any] | list[dict[str, Any]] | tuple[torch.Tensor, ...],
        component_io_sink: ComponentIOSink | None = None,
    ) -> torch.Tensor:
        """Preprocess, run full inference, postprocess. Returns (B, H, total_dof)."""
        pv, ids, mask, st, actions = self._preprocess(step_data)
        kv_flat = self.encode_vlm(pv, ids, mask, component_io_sink=component_io_sink)
        raw = self.denoise_steps(
            st, actions, kv_flat, mask, component_io_sink=component_io_sink
        )
        return self._postprocess(raw)

    @classmethod
    def from_components(
        cls,
        models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel],
    ) -> CollectionAppEvaluateProtocol:
        """Build a GrootApp from a list of [vit, llm, vlm_proj, dit] components."""
        vit, llm, vlm_proj, dit = models
        device = Model._last_host_device

        # Reuse the cached policy already loaded by from_pretrained
        # since from_pretrained always runs before from_components in the pipeline.
        policy = load_checkpoint(
            checkpoint=Model._last_checkpoint,
            data_config=Model._last_data_config,
            embodiment_tag=Model._last_embodiment_tag,
            device=device,
        )
        config = GrootAppConfig.from_policy(policy)
        return cls(
            config=config, vit=vit, llm=llm, vlm_proj=vlm_proj, dit=dit, device=device
        )

    def run_model_for_eval(
        self,
        model_input: Any,
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        """Run the full pipeline for evaluation.

        Torch path: predict_action_chunk -> yield result.
        On-device path: preprocess full batch -> encode_vlm sync ->
          submit DiT async -> yield _PostprocessedAsyncResult
        """
        if isinstance(model_input, GeneratorType):
            # On-device: stack all per-field chunks back into a full batch
            field_splits = list(model_input)
            stacked: tuple[torch.Tensor, ...] = tuple(
                torch.cat(list(chunks), dim=0) for chunks in field_splits
            )
            pv, ids, mask, st, actions = self._preprocess(stacked)

            # VIT, LLM, VLMProj: sequential deps — run synchronously
            kv_flat = self.encode_vlm(pv, ids, mask)

            # DiT: submit without waiting; framework drains .wait() later
            cross_mask = preprocess_dit(
                mask,
                self.config.dit_num_heads,
                self.config.action_horizon,
                self.config.num_target_vision_tokens,
            )
            async_result = _call_component(self.dit, st, actions, cross_mask, *kv_flat)
            assert isinstance(async_result, AsyncOnDeviceResult)
            result: _PostprocessedAsyncResult | torch.Tensor = (
                _PostprocessedAsyncResult(async_result, self._postprocess)
            )
        else:
            result = self.predict_action_chunk(model_input)

        yield result  # type: ignore[misc]
        return result  # type: ignore[return-value]

    @staticmethod
    def get_calibration_data(
        policy: Gr00tPolicy,
        dataset: LeRobotSingleDataset,
        component_name: str,
        num_samples: int = 100,
        host_device: str = "cpu",
        use_cache: bool = True,
        rng_seed: int = 42,
    ) -> tuple[ComponentIOSink, ComponentIOSink]:
        """
        Collect calibration + eval samples for `component_name`.

        Runs the FP pipeline over 2 * num_samples randomly sampled steps,
        dumps each sample using sinks as it is produced and returns the calib and eval sinks.

        If the cache already exists (same num_samples), it is loaded directly.
        """
        sink_calib = ComponentIOSink(eval_mode=EvalMode.FP, split=DataSplit.CALIB)
        sink_eval = ComponentIOSink(eval_mode=EvalMode.FP, split=DataSplit.EVAL)

        needs_dump = (
            sink_calib.sample_count(component_name) < num_samples
            or sink_eval.sample_count(component_name) < num_samples
        ) or not use_cache

        if needs_dump:
            rng = random.Random(rng_seed)
            sampled = rng.sample(
                range(len(dataset)), min(2 * num_samples, len(dataset))
            )
            calib_indices = sampled[:num_samples]
            calib_eval_indices = sampled[num_samples:]

            app = build_app(
                policy,
                eval_mode=EvalMode.FP,
                host_device=host_device,
            )

            def _run_and_dump(indices: list[int], sink: ComponentIOSink) -> None:
                print(f"Dumping {len(indices)} samples -> split={sink.split.value}")
                with torch.no_grad():
                    for idx in indices:
                        app.predict_action_chunk(dataset[idx], component_io_sink=sink)

            _run_and_dump(calib_indices, sink_calib)
            _run_and_dump(calib_eval_indices, sink_eval)

        return sink_calib, sink_eval


# App builder
def build_app(
    policy: Gr00tPolicy,
    eval_mode: EvalMode,
    device: hub.Device | None = None,
    hub_model_id: str | None = None,
    host_device: str = "cpu",
) -> GrootApp:
    """
    Build a GrootApp with either host PyTorch components (EvalMode.FP)
    or on-device hub.InferenceJob components (EvalMode.ON_DEVICE).
    """
    config = GrootAppConfig.from_policy(policy)

    if eval_mode == EvalMode.FP:
        collection = Model.from_policy(policy)
        return GrootApp(
            config=config,
            vit=collection.components["vit"],
            llm=collection.components["llm"],
            vlm_proj=collection.components["vlm_proj"],
            dit=collection.components["dit"],
            device=host_device,
        )

    # On-device path: wrap each component's InferenceJob as the callable.
    if device is None:
        raise ValueError("device must be provided for ON_DEVICE eval mode")
    if hub_model_id is None:
        raise ValueError("hub_model_id must be provided for ON_DEVICE eval mode")

    hub_ids = hub_model_id.split(",")
    component_names = list(GrootQuantizedCollection.component_classes.keys())
    if len(hub_ids) != len(component_names):
        raise ValueError(
            f"Expected {len(component_names)} comma-separated hub model IDs "
            f"({', '.join(component_names)}), got {len(hub_ids)}"
        )
    on_device_components = {}
    for name, hub_id in zip(component_names, hub_ids, strict=False):
        on_device_components[name] = OnDeviceModel(
            model=hub.get_model(hub_id),
            input_names=[],
            device=device,
        )

    return GrootApp(
        config=config,
        vit=on_device_components["vit"],
        llm=on_device_components["llm"],
        vlm_proj=on_device_components["vlm_proj"],
        dit=on_device_components["dit"],
        device=host_device,
    )


### Utils
def get_default_dataset_path() -> str:
    """Download & unpack the default dataset if not cached."""
    return str(DEFAULT_DATASET_ASSET.fetch(extract=True))
