# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import qai_hub as hub

_CAN_ACCESS_HUB: bool | None = None


def can_access_qualcomm_ai_hub() -> bool:
    global _CAN_ACCESS_HUB  # noqa: PLW0603
    if _CAN_ACCESS_HUB is not None:
        return _CAN_ACCESS_HUB
    try:
        hub.get_frameworks()
    except Exception:
        _CAN_ACCESS_HUB = False
    else:
        _CAN_ACCESS_HUB = True
    return _CAN_ACCESS_HUB
