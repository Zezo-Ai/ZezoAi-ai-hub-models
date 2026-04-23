# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Tensor specification types for model I/O.

This module defines the core types for specifying tensor metadata:
- TensorSpec: Specification for a tensor (shape, dtype, metadata)
- IoType: Semantic type of a tensor (image, tensor, bbox)
- ColorFormat: Color format for image tensors
- ImageMetadata: Metadata for image tensors
- BboxFormat: Bounding box coordinate format
- BboxMetadata: Metadata for bounding box tensors

These types are used both in model.get_input_spec() and in metadata.yaml files.

Note: InputSpec is defined in qai_hub_models.utils.input_spec to keep
input specification types together with input-related utilities.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from typing import Any, Literal, overload

from qai_hub_models_cli.proto.shared import range_pb2, tensor_spec_pb2

from qai_hub_models.utils.base_config import BaseQAIHMConfig


# ---------------------------------------------------------------------------
# Enums for semantic metadata
# ---------------------------------------------------------------------------
class IoType(Enum):
    """Semantic type of an input/output tensor."""

    TENSOR = "tensor"
    IMAGE = "image"
    BBOX = "bbox"


class ColorFormat(Enum):
    """Color channel format for image tensors."""

    RGB = "rgb"
    BGR = "bgr"
    GRAYSCALE = "grayscale"


class BboxFormat(Enum):
    """Bounding box coordinate format."""

    XYXY = "xyxy"  # (x1, y1, x2, y2) - top-left and bottom-right corners
    XYWH = "xywh"  # (x, y, width, height) - top-left corner and size
    CXCYWH = "cxcywh"  # (cx, cy, width, height) - center and size


# ---------------------------------------------------------------------------
# Metadata classes
# ---------------------------------------------------------------------------
class ImageMetadata(BaseQAIHMConfig):
    """
    Metadata specific to image tensor inputs.

    This class groups image-related metadata fields like color format
    and value range together.
    """

    color_format: ColorFormat = ColorFormat.RGB
    value_range: tuple[float, float] = (0.0, 1.0)


class BboxMetadata(BaseQAIHMConfig):
    """
    Metadata specific to bounding box tensor outputs.

    This class groups bbox-related metadata fields like coordinate format
    and value range (pixel space vs normalized).
    """

    bbox_format: BboxFormat = BboxFormat.XYXY
    value_range: tuple[float, float] = (0.0, float("inf"))


# ---------------------------------------------------------------------------
# Quantization Parameters
# ---------------------------------------------------------------------------
_DTYPE_TO_PROTO: dict[str, int] = {
    "float16": tensor_spec_pb2.DTYPE_FLOAT16,
    "float32": tensor_spec_pb2.DTYPE_FLOAT32,
    "float64": tensor_spec_pb2.DTYPE_FLOAT64,
    "int8": tensor_spec_pb2.DTYPE_INT8,
    "int16": tensor_spec_pb2.DTYPE_INT16,
    "int32": tensor_spec_pb2.DTYPE_INT32,
    "int64": tensor_spec_pb2.DTYPE_INT64,
    "uint8": tensor_spec_pb2.DTYPE_UINT8,
    "uint16": tensor_spec_pb2.DTYPE_UINT16,
    "uint32": tensor_spec_pb2.DTYPE_UINT32,
    "uint64": tensor_spec_pb2.DTYPE_UINT64,
    "bool": tensor_spec_pb2.DTYPE_BOOL,
}

_IO_TYPE_TO_PROTO: dict[str, int] = {
    "tensor": tensor_spec_pb2.IO_TYPE_TENSOR,
    "image": tensor_spec_pb2.IO_TYPE_IMAGE,
    "bbox": tensor_spec_pb2.IO_TYPE_BBOX,
}

_COLOR_FORMAT_TO_PROTO: dict[str, int] = {
    "rgb": tensor_spec_pb2.COLOR_FORMAT_RGB,
    "bgr": tensor_spec_pb2.COLOR_FORMAT_BGR,
    "grayscale": tensor_spec_pb2.COLOR_FORMAT_GRAYSCALE,
}


class QuantizationParameters(BaseQAIHMConfig):
    """Quantization parameters for a tensor."""

    scale: float
    zero_point: int

    def to_proto(self) -> tensor_spec_pb2.QuantizationParameters:
        return tensor_spec_pb2.QuantizationParameters(
            scale=self.scale,
            zero_point=self.zero_point,
        )


# ---------------------------------------------------------------------------
# TensorSpec
# ---------------------------------------------------------------------------
class TensorSpec(BaseQAIHMConfig):
    """
    Specification for an input or output tensor.

    This class serves dual purposes:
    1. As the return type for model.get_input_spec() - supports tuple-like
       unpacking for backwards compatibility (shape, dtype = spec works)
    2. As the schema for metadata.yaml files - includes additional fields
       for documentation and quantization parameters.

    For use in get_input_spec(), create with shape, dtype, and optional metadata:
        TensorSpec(shape=(1, 3, 224, 224), dtype="float32")
        TensorSpec(
            shape=(1, 3, 224, 224),
            dtype="float32",
            io_type=IoType.IMAGE,
            image_metadata=ImageMetadata(color_format=ColorFormat.RGB),
        )

    Attributes
    ----------
    shape
        Tensor shape as a tuple of ints.
    dtype
        Data type string (e.g., "float32", "int64").
    description
        Human-readable description of the tensor's purpose.
    quantization_parameters
        Optional quantization scale/zero_point (populated from compiled model).
    io_type
        Semantic type: IMAGE for image tensors, BBOX for bounding box tensors,
        TENSOR for generic tensors.
    image_metadata
        Image-specific metadata (color_format, value_range). Only used when
        io_type is IMAGE.
    bbox_metadata
        Bbox-specific metadata (bbox_format). Only used when io_type is BBOX.
    value_range
        Value range for generic tensors (not images). Default is (None, None)
        meaning unbounded.
    """

    shape: tuple[int, ...] = ()
    dtype: str = ""
    description: str | None = None
    quantization_parameters: QuantizationParameters | None = None
    # Semantic metadata fields (populated from get_input_spec()/get_output_spec() metadata)
    io_type: IoType = IoType.TENSOR
    image_metadata: ImageMetadata | None = None
    bbox_metadata: BboxMetadata | None = None
    value_range: tuple[float, float] = (float("-inf"), float("inf"))

    # -----------------------------------------------------------------------
    # Tuple-like behavior for backwards compatibility with InputSpec usage
    # -----------------------------------------------------------------------
    @overload
    def __getitem__(self, idx: Literal[0]) -> tuple[int, ...]: ...

    @overload
    def __getitem__(self, idx: Literal[1]) -> str: ...

    def __getitem__(self, idx: int) -> tuple[int, ...] | str:
        """Allow indexing: spec[0] -> shape, spec[1] -> dtype"""
        if idx == 0:
            return self.shape
        if idx == 1:
            return self.dtype
        raise IndexError(f"TensorSpec index out of range: {idx}")

    def __len__(self) -> int:
        """Return 2 for compatibility with tuple length checks."""
        return 2

    def __iter__(self) -> Iterator[Any]:
        """Allow unpacking: shape, dtype = spec

        Returns Iterator[Any] because Python's type system cannot express
        "first yield is tuple[int, ...], second yield is str". Use indexing
        (spec[0], spec[1]) or attribute access (spec.shape, spec.dtype) for
        proper type inference.
        """
        yield self.shape
        yield self.dtype

    def to_proto(self, name: str) -> tensor_spec_pb2.TensorSpec:
        quant = None
        if self.quantization_parameters is not None:
            quant = self.quantization_parameters.to_proto()

        value_range = None
        if self.value_range != (float("-inf"), float("inf")):
            value_range = range_pb2.Range(
                double_r=range_pb2.DoubleRange(
                    min=self.value_range[0], max=self.value_range[1]
                )
            )

        image_metadata = None
        if self.image_metadata is not None:
            image_metadata = tensor_spec_pb2.ImageMetadata(
                color_format=_COLOR_FORMAT_TO_PROTO.get(
                    self.image_metadata.color_format.value,
                    tensor_spec_pb2.COLOR_FORMAT_UNSPECIFIED,
                ),
            )

        return tensor_spec_pb2.TensorSpec(
            name=name,
            shape=list(self.shape),
            dtype=_DTYPE_TO_PROTO.get(self.dtype, tensor_spec_pb2.DTYPE_UNSPECIFIED),
            quantization_parameters=quant,
            description=self.description,
            value_range=value_range,
            io_type=_IO_TYPE_TO_PROTO.get(
                self.io_type.value, tensor_spec_pb2.IO_TYPE_UNSPECIFIED
            ),
            image_metadata=image_metadata,
        )
