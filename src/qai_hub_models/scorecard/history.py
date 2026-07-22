# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Structured configs for scorecard historical storage and trend detection."""

from __future__ import annotations

from qai_hub_models.scorecard.results.performance_diff import SevereRegression
from qai_hub_models.utils.base_config import BaseQAIHMConfig


class ScorecardManifest(BaseQAIHMConfig):
    """Manifest for a single scorecard run stored in S3.

    Uploaded last as an atomic "commit" marker — its presence signals all
    artifacts were uploaded successfully.
    """

    run_id: str
    date: str
    branch: str = "main"
    deployment: str = "prod"
    # Distinguishes scheduled weekly runs ("weekly-prod", "weekly-dev") from
    # ad-hoc workflow_dispatches (which carry the dispatcher's chosen
    # tableau_branch_name). Callers can filter by this to skip test/manual
    # runs when picking the "previous" baseline for the toolchain-version
    # diff or the Scorecard Context grid. Empty string on manifests uploaded
    # before this field was introduced.
    run_name: str = ""
    commit_sha: str = ""
    github_run_url: str = ""
    artifacts: list[str] = []


class RecoveredRegression(BaseQAIHMConfig):
    """A regression that was sustained in history but not in the current run."""

    key: str
    count: int
    of: int


class TrendSummary(BaseQAIHMConfig):
    """Summary statistics for a trend report."""

    lookback: int
    threshold: int
    total_current: int
    sustained_count: int
    new_count: int
    flaky_count: int
    recovered_count: int


class TrendReport(BaseQAIHMConfig):
    """Classification of current regressions against historical data."""

    sustained: list[SevereRegression] = []
    new: list[SevereRegression] = []
    recovered: list[RecoveredRegression] = []
    flaky: list[SevereRegression] = []
    summary: TrendSummary
