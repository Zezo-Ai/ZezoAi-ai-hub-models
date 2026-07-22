# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for scorecard-history S3 filtering."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from qai_hub_models.scorecard.history import ScorecardManifest
from qai_hub_models.scripts import download_scorecard_results as mod


def _mk(run_id: str, deployment: str, run_name: str, date: str) -> ScorecardManifest:
    return ScorecardManifest(
        run_id=run_id, date=date, deployment=deployment, run_name=run_name
    )


def test_find_latest_run_skips_ad_hoc_dispatches_by_default() -> None:
    """The Scorecard Context grid and toolchain-diff baseline must pick the
    real weekly scorecards, not ad-hoc workflow_dispatches that a coworker
    might trigger with an arbitrary tableau_branch_name.
    """
    # Descending-by-date order matches list_runs()'s sort. An ad-hoc dispatch
    # sits at the top; the real weekly-prod is older.
    manifests = [
        _mk("adhoc-latest", "prod", "test-toolchain-diff-v2", "2026-07-13"),
        _mk("adhoc-mid", "prod", "shreya-experiment", "2026-07-13"),
        _mk("weekly-latest", "prod", "weekly-prod", "2026-07-06"),
    ]
    with mock.patch.object(mod, "list_runs", return_value=manifests):
        result = mod.find_latest_run("prod")
    assert result is not None
    assert result.run_id == "weekly-latest"


def test_find_latest_run_empty_prefix_includes_everything() -> None:
    """Passing run_name_prefix="" is the escape hatch when a caller genuinely
    wants the latest manifest of any shape (e.g. debugging).
    """
    manifests = [
        _mk("adhoc-latest", "prod", "test-toolchain-diff-v2", "2026-07-13"),
        _mk("weekly-latest", "prod", "weekly-prod", "2026-07-06"),
    ]
    with mock.patch.object(mod, "list_runs", return_value=manifests):
        result = mod.find_latest_run("prod", run_name_prefix="")
    assert result is not None
    assert result.run_id == "adhoc-latest"


def test_download_single_artifact_key_matches_upload_layout(tmp_path: Path) -> None:
    """The S3 key must be {PREFIX}/{run_id}-{run_name}/{artifact}, matching
    upload_scorecard_history.upload_scorecard_to_s3. A key mismatch here
    silently returns None on every lookup, and the toolchain diff / Scorecard
    Context grid fall back to intermediates without any error.
    """
    manifest = _mk("29003645353", "dev", "weekly-dev", "2026-07-09")
    dest = tmp_path / "tool-versions.yaml"

    with (
        mock.patch.object(mod, "get_qaihm_s3_or_exit", return_value=("bkt", None)),
        mock.patch.object(mod, "s3_file_exists", return_value=True) as exists,
        mock.patch.object(mod, "s3_download") as download,
    ):
        result = mod.download_single_artifact(manifest, "tool-versions.yaml", dest)

    expected_key = "scorecard-history/29003645353-weekly-dev/tool-versions.yaml"
    exists.assert_called_once_with("bkt", expected_key)
    download.assert_called_once_with("bkt", expected_key, dest, verbose=False)
    assert result == dest
