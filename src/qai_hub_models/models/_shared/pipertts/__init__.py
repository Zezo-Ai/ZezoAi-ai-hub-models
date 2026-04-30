# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import sys

import monotonic_align
import piper_train.vits

# from qai_hub_models.utils.asset_loaders import find_replace_in_repo

'''
# For w8a16 precision, QAIRT demands Conv1d to has bias
new_conv = """
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=True)
        self.conv_post.bias.data.zero_()
"""

find_replace_in_repo(
    piper_train.__path__[0],
    "vits/models.py",
    "self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)",
    new_conv,
)
'''
# piper_train.vits.monotonic_align expects a compiled Cython submodule
# (.monotonic_align.core) that is not shipped with the pip package.
# The standalone monotonic_align package provides the same functionality,
# so redirect the piper_train import to use it instead.
sys.modules["piper_train.vits.monotonic_align"] = monotonic_align
piper_train.vits.monotonic_align = monotonic_align  # type: ignore[attr-defined, unused-ignore]
