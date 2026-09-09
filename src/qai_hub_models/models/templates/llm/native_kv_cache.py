# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


class NativeKVLayer:
    """One layer of a fixed-size KV buffer, laid out for the exported graph.

    The native KV graph consumes key as ``(kv_heads, batch, head_dim, context)``
    and value as ``(kv_heads, batch, context, head_dim)``, so the sequence axis
    is -1 for keys and -2 for values.
    """

    def __init__(self, max_cache_len: int) -> None:
        self.max_cache_len = max_cache_len
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.num_heads = 0
        self.max_batch_size = 0
        self.head_dim = 0
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        num_kv_heads, batch_size, head_dim, _ = key_states.shape
        self.num_heads = num_kv_heads
        self.max_batch_size = batch_size
        self.head_dim = head_dim
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.zeros(
            (num_kv_heads, batch_size, head_dim, self.max_cache_len),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (num_kv_heads, batch_size, self.max_cache_len, head_dim),
            dtype=self.dtype,
            device=self.device,
        )

    def adopt(self, keys: torch.Tensor, values: torch.Tensor, seq_length: int) -> None:
        """Take ownership of buffers the generator already scattered into."""
        self.num_heads, self.max_batch_size, self.head_dim = keys.shape[:3]
        self.dtype, self.device = keys.dtype, keys.device
        self.keys, self.values = keys, values
        self.cumulative_length = seq_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.lazy_initialization(key_states)
        assert self.keys is not None and self.values is not None

        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(key_states.shape[-1], device=self.keys.device)

        self.keys[..., cache_position] = key_states.to(self.keys.dtype)
        self.values[..., cache_position, :] = value_states.to(self.values.dtype)
        self.cumulative_length = max(
            self.cumulative_length, int(cache_position[-1]) + 1
        )
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def reset(self) -> None:
        if self.keys is not None:
            self.keys.zero_()
        if self.values is not None:
            self.values.zero_()
        self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError(
            "Native KV buffers are heads-first, so beam search would have to "
            "reorder dim 1 rather than dim 0."
        )


class NativeKVCache:
    """Cache over the fixed-size KV buffers consumed by the exported graph.

    Deliberately standalone rather than a ``transformers`` ``StaticCache``: the
    layer-based ``Cache`` API postdates the ``transformers==4.45.0`` that the
    llama recipes pin, and this only needs to duck-type the handful of methods
    the driver touches. Mirrors ``_FlatListCache`` in the driver.
    """

    def __init__(self, context_length: int, num_hidden_layers: int) -> None:
        self.layers = [NativeKVLayer(context_length) for _ in range(num_hidden_layers)]
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[tuple[torch.Tensor | None, torch.Tensor | None]]:
        for layer in self.layers:
            yield (layer.keys, layer.values)

    @classmethod
    def from_buffers(
        cls,
        buffers: list[torch.Tensor],
        context_length: int,
        seq_length: int,
    ) -> NativeKVCache:
        """Wrap flat ``[k0, v0, k1, v1, ...]`` buffers the generator wrote into."""
        cache = cls(context_length, len(buffers) // 2)
        for layer_idx, layer in enumerate(cache.layers):
            layer.adopt(buffers[layer_idx * 2], buffers[layer_idx * 2 + 1], seq_length)
        return cache

    @classmethod
    def allocate(
        cls,
        kv_shapes: list[tuple[int, int]],
        context_length: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> NativeKVCache:
        """Zeroed buffers from one ``(num_kv_heads, head_dim)`` pair per layer."""
        cache = cls(context_length, len(kv_shapes))
        for layer, (num_kv_heads, head_dim) in zip(
            cache.layers, kv_shapes, strict=True
        ):
            layer.lazy_initialization(
                torch.zeros(
                    (num_kv_heads, batch_size, head_dim, 0), dtype=dtype, device=device
                )
            )
        return cache

    def to_flat_buffers(self) -> list[torch.Tensor]:
        """The buffers in exported-graph order, ``[k0, v0, k1, v1, ...]``."""
        flat: list[torch.Tensor] = []
        for layer in self.layers:
            assert layer.keys is not None and layer.values is not None
            flat += [layer.keys, layer.values]
        return flat

    def adopt_flat_buffers(self, buffers: list[torch.Tensor]) -> None:
        """Re-point at equivalent buffers, e.g. after a device move."""
        for layer_idx, layer in enumerate(self.layers):
            layer.keys = buffers[layer_idx * 2]
            layer.values = buffers[layer_idx * 2 + 1]

    def append(self, flat_kv: list[torch.Tensor]) -> None:
        """Write one step of flat ``[k0, v0, ...]`` states at the cache's end."""
        start = self.get_seq_length()
        seq_length = flat_kv[0].shape[-1]
        if start + seq_length > self.context_length:
            raise ValueError(
                f"Context length exhausted: {start} cached token(s) plus "
                f"{seq_length} new exceeds context_length={self.context_length}. "
                "The native KV buffer is fixed-size and cannot slide."
            )
        cache_position = torch.arange(
            start, start + seq_length, device=flat_kv[0].device
        )
        for layer_idx, layer in enumerate(self.layers):
            layer.update(
                flat_kv[layer_idx * 2],
                flat_kv[layer_idx * 2 + 1],
                {"cache_position": cache_position},
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.context_length

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int = 0
    ) -> tuple[int, int]:
        # The buffer is always full-width, so the mask spans it with no offset.
        return self.context_length, 0

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * len(self.layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError(
            "Native KV buffers are heads-first, so beam search would have to "
            "reorder dim 1 rather than dim 0."
        )
