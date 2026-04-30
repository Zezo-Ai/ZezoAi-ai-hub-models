# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from qai_hub_models.models.superpoint.model import HF_MODEL_ID


class SuperPointApp:
    """End-to-end inference wrapper for SuperPoint."""

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        height: int,
        width: int,
    ) -> None:
        self.model = model
        self.height = height
        self.width = width
        self.processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID, use_fast=True)

    def predict(
        self,
        image1: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        image2: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        raw_output: bool = False,
    ) -> (
        Image.Image
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        """Detect keypoints in two images, match them, and return a visualization.

        If ``raw_output=True``, returns
        ``(kp1, sc1, desc1, kp2, sc2, desc2)`` for both images instead.
        """
        inputs1 = self.processor(
            image1,
            return_tensors="pt",
            size={"height": self.height, "width": self.width},
            do_grayscale=True,
        )
        inputs2 = self.processor(
            image2,
            return_tensors="pt",
            size={"height": self.height, "width": self.width},
            do_grayscale=True,
        )
        # Processor always returns 3 channels with identical planes when do_grayscale=True;
        t1 = inputs1["pixel_values"][:, :1]
        t2 = inputs2["pixel_values"][:, :1]

        frame1 = (t1[0, 0].cpu().numpy() * 255).astype(np.uint8)
        frame2 = (t2[0, 0].cpu().numpy() * 255).astype(np.uint8)

        image_tensor = torch.cat([t1, t2], dim=0)
        batched = image_tensor.unsqueeze(0)
        kp_batch, sc_batch, desc_batch = self.model(batched)

        kp1 = kp_batch[0, 0].cpu().numpy()
        sc1 = sc_batch[0, 0].cpu().numpy()
        desc1 = desc_batch[0, 0].cpu().numpy()
        kp2 = kp_batch[0, 1].cpu().numpy()
        sc2 = sc_batch[0, 1].cpu().numpy()
        desc2 = desc_batch[0, 1].cpu().numpy()

        if raw_output:
            return (kp1, sc1, desc1, kp2, sc2, desc2)

        valid1, valid2 = sc1 > 0, sc2 > 0
        kp1, desc1 = kp1[valid1], desc1[valid1]
        kp2, desc2 = kp2[valid2], desc2[valid2]

        if len(kp1) >= 4 and len(kp2) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches: list = sorted(
                bf.match(desc1.astype(np.float32), desc2.astype(np.float32)),
                key=lambda x: x.distance,
            )
        else:
            matches = []

        return self._draw_matches(
            Image.fromarray(frame1),
            Image.fromarray(frame2),
            kp1,
            kp2,
            matches,
        )

    @staticmethod
    def _draw_matches(
        image0: Image.Image,
        image1: Image.Image,
        kp0: np.ndarray,
        kp1: np.ndarray,
        matches: list,
    ) -> Image.Image:
        """Draw matches between two images side-by-side."""
        img0_np = np.array(image0.convert("RGB"))
        img1_np = np.array(image1.convert("RGB"))

        kp0_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0]
        kp1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1]

        img_matches = cv2.drawMatches(  # type: ignore[call-overload]
            img0_np,
            kp0_cv,
            img1_np,
            kp1_cv,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        return Image.fromarray(img_matches)
