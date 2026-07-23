# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Prints "true" if the geniex_qairt QAIRT version QAIHM targets differs from the
# version pinned in intermediates/tool-versions.yaml, else "false". The weekend
# prod scorecard uses this to decide whether to rebuild all downloadable LLM assets.

from __future__ import annotations

from pathlib import Path

from qai_hub_models import TargetRuntime
from qai_hub_models.scorecard.artifacts import INTERMEDIATES_DIR
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.yaml import ToolVersionsByPathYaml


def qairt_geniex_changed(intermediates_dir: Path) -> bool:
    """
    True if the QAIRT version QAIHM targets for geniex_qairt differs from the version
    stamped in <intermediates_dir>/tool-versions.yaml. Returns False if no version is
    pinned yet. Constructs a QAIRTVersion, which may hit AI Hub Workbench when a
    client is configured; degrades gracefully to a literal-version parse otherwise.
    """
    pinned = ToolVersionsByPathYaml.from_dir(intermediates_dir).tool_versions.get(
        ScorecardProfilePath.GENIEX_QAIRT
    )
    pinned_qairt = pinned.qairt if pinned else None
    if pinned_qairt is None:
        return False
    return TargetRuntime.GENIEX_QAIRT.default_qairt_version != pinned_qairt


if __name__ == "__main__":
    print("true" if qairt_geniex_changed(INTERMEDIATES_DIR) else "false")
