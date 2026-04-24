# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Unit tests for qai_hub_models.scripts.notify_nightly_failures."""

import json
from pathlib import Path

from qai_hub_models.scripts.notify_nightly_failures import (
    categorize_failures,
    load_failed_jobs_json,
    render_issue_body,
)


def test_categorize_failures() -> None:
    """Splits workbench vs general failures by job name."""
    names = [
        "Run Tests (py 3.10) / Verify Model Tests",
        "Run Tests (py 3.10) / Run QAIHM Tests",
        "Run Tests (py 3.10) / Pre-commit",
    ]
    urls = [
        "https://github.com/org/repo/actions/runs/123/job/456",
        "https://github.com/org/repo/actions/runs/123/job/789",
        "https://github.com/org/repo/actions/runs/123/job/111",
    ]
    workbench, general = categorize_failures(names, urls)
    assert len(workbench) == 1
    assert "Verify Model Tests" in workbench[0]["name"]
    assert len(general) == 2


def test_load_failed_jobs_json(tmp_path: Path) -> None:
    """Loads structured JSON; returns empty dict for missing file."""
    assert load_failed_jobs_json(None) == {}
    assert load_failed_jobs_json(str(tmp_path / "nonexistent.json")) == {}

    data = {"compile/resnet50_TFLITE": "https://dev.aihub.qualcomm.com/jobs/j123"}
    json_path = tmp_path / "failed-workbench-jobs.json"
    json_path.write_text(json.dumps(data))
    assert load_failed_jobs_json(str(json_path)) == data


def test_render_workbench_issue() -> None:
    """Workbench template includes failure links and AI Hub job URLs."""
    body = render_issue_body(
        "workbench_issue.j2",
        today="2026-04-24",
        failures=[{"name": "Verify Model Tests", "url": "https://example.com/job/1"}],
        failed_aihub_jobs={
            "compile/resnet50": "https://dev.aihub.qualcomm.com/jobs/j1"
        },
        run_url="https://github.com/org/repo/actions/runs/123",
        repository="org/repo",
        ref_name="main",
    )
    assert "Workbench Job Failures" in body
    assert "Verify Model Tests" in body
    assert "https://dev.aihub.qualcomm.com/jobs/j1" in body
    assert "View Workflow Run" in body


def test_render_general_issue() -> None:
    """General template includes failure links and triage steps."""
    body = render_issue_body(
        "general_issue.j2",
        today="2026-04-24",
        failures=[{"name": "Run QAIHM Tests", "url": "https://example.com/job/2"}],
        run_url="https://github.com/org/repo/actions/runs/123",
        repository="org/repo",
        ref_name="main",
    )
    assert "Test Failures" in body
    assert "Run QAIHM Tests" in body
    assert "Nightly Failure Log" in body
