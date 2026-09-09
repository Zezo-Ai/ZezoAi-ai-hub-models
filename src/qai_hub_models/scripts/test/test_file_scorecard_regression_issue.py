# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.scripts.file_scorecard_regression_issue import build_issue_body

_RUN_URL = "https://example.com/run"
_PERF_URL = "https://example.com/perf"
_NUMERICS_URL = "https://example.com/numerics"


def test_body_includes_accuracy_sections_when_present() -> None:
    """Non-empty accuracy inputs render both new headings; job-id linkification
    does not run on rows that have no Job ID columns.
    """
    disabled = [
        {
            "Model ID": "fresh_model",
            "Runtime": "qnn_dlc",
            "FP Accuracy": "0.75",
            "Device Accuracy": "0.60",
            "Newly Disabled": "True",
        }
    ]
    benchmarks = [
        {
            "Model ID": "fresh_bench",
            "Accuracy Type": "device",
            "Actual Value": "0.30",
            "Benchmark Value": "0.70",
            "Newly Disabled": "True",
        }
    ]
    body = build_issue_body(
        perf_regressions=[],
        numerics_regressions=[],
        run_url=_RUN_URL,
        perf_diff_url=_PERF_URL,
        numerics_diff_url=_NUMERICS_URL,
        newly_disabled_configs=disabled,
        newly_failing_benchmarks=benchmarks,
    )
    assert "## Newly Disabled Configurations" in body
    assert "## Newly Failing Benchmarks" in body
    assert "fresh_model" in body
    assert "fresh_bench" in body
