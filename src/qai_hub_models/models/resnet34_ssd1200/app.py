# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.app import YoloObjectDetectionApp


class ResNet34SSDApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        h, w = pixel_values.shape[-2:]
        assert h == 1200 and w == 1200, (
            f"ResNet34SSD only supports 1200x1200 input. Received: {h}x{w}"
        )
