# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

# The mediapipe app tests used to compare the *rendered* output image pixel for
# pixel against a golden PNG. That is brittle: a handful of overlay pixels
# (landmarks / boxes) flip by up to 128 across environments while the model
# itself is unchanged, so the pixel comparison fails even though the detections
# are effectively identical. Instead we compare the structured output
# (bounding box + landmark coordinates), which is what the model actually
# predicts, with a tolerance that is robust to that cross-environment drift.

# Landmark / box coordinates live in input-image pixel space (hundreds to a few
# thousand), so a couple of pixels of drift is negligible.
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1.0


def _first_detection(batched: Any) -> np.ndarray:
    # The mediapipe apps run on a single image and return one selected detection
    # per output list. Pull that detection out as a numpy array.
    assert len(batched) >= 1, "Expected at least one batch element."
    tensor = batched[0]
    assert isinstance(tensor, torch.Tensor), "Expected a detection tensor."
    assert tensor.numel() > 0, "Expected a non-empty detection tensor."
    return tensor.detach().cpu().numpy()


def landmarks_from_raw_output(raw_output: Sequence[Any]) -> dict[str, np.ndarray]:
    """Extract the comparable structured output from an app's raw_output tuple.

    All mediapipe apps return
        (batched_selected_boxes, batched_selected_keypoints,
         batched_roi_4corners, batched_selected_landmarks, ...)
    We compare the selected box and the selected landmarks, which together
    determine everything drawn on the output image.
    """
    boxes = _first_detection(raw_output[0])
    landmarks = _first_detection(raw_output[3])
    return {"boxes": boxes, "landmarks": landmarks}


def assert_landmarks_close(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Assert two structured mediapipe outputs match within tolerance."""
    for key in ("boxes", "landmarks"):
        a = np.asarray(actual[key])
        e = np.asarray(expected[key])
        assert a.shape == e.shape, (
            f"{key}: shape mismatch, actual {a.shape} vs expected {e.shape}."
        )
        np.testing.assert_allclose(
            a, e, rtol=rtol, atol=atol, err_msg=f"{key} mismatch"
        )
