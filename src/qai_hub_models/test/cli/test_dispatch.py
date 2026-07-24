# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for the heavy-side export/evaluate dispatcher."""

from __future__ import annotations

import argparse
from unittest.mock import Mock, patch

from qai_hub_models.cli.dispatch import run_model_script


def test_dispatch_export_builds_parser_and_runs() -> None:
    """Export path: parser is built once from resolved model and pipeline invoked."""
    fake_parser = argparse.ArgumentParser()
    fake_parser.add_argument("--device")
    fake_parsed = argparse.Namespace(device="S25")

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_model",
            return_value=Mock(model_id="fake_model"),
        ) as mock_resolve,
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ) as mock_build,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
        patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=True),
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("fake_model", "export", ["--device", "S25"])

        # resolve_model is called exactly once — no double resolution.
        mock_resolve.assert_called_once_with("fake_model")
        mock_build.assert_called_once()
        mock_select.assert_called_once()


def test_dispatch_export_prompts_for_unpublished_model() -> None:
    """Export for unpublished model prompts and exits early if user declines."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_model",
            return_value=Mock(model_id="sam"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_export_parser_for",
            return_value=fake_parser,
        ),
        patch("qai_hub_models.cli.dispatch.QAIHMModelManifest.from_model") as mock_info,
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_pipeline") as mock_select,
    ):
        mock_info.return_value.status.value = "pending"
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("sam", "export", [])

        mock_check.assert_called_once()
        mock_select.assert_not_called()


def test_dispatch_evaluate_builds_parser_and_runs() -> None:
    """Evaluate path: parser is built once from resolved model and pipeline invoked."""
    fake_parser = argparse.ArgumentParser()
    fake_parser.add_argument("--device")
    fake_parsed = argparse.Namespace(device="S25")

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_model",
            return_value=Mock(model_id="fake_model"),
        ) as mock_resolve,
        patch(
            "qai_hub_models.cli.dispatch.build_evaluate_parser_for",
            return_value=fake_parser,
        ) as mock_build,
        patch("qai_hub_models.cli.dispatch.select_evaluate_pipeline") as mock_select,
        patch("qai_hub_models.cli.dispatch._confirm_run_ok", return_value=True),
    ):
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("fake_model", "evaluate", ["--device", "S25"])

        mock_resolve.assert_called_once_with("fake_model")
        mock_build.assert_called_once()
        mock_select.assert_called_once()


def test_dispatch_evaluate_prompts_for_unpublished_model() -> None:
    """Evaluate for unpublished model prompts and exits early if user declines."""
    fake_parser = argparse.ArgumentParser()
    fake_parsed = argparse.Namespace()

    with (
        patch(
            "qai_hub_models.cli.dispatch.resolve_model",
            return_value=Mock(model_id="sam"),
        ),
        patch(
            "qai_hub_models.cli.dispatch.build_evaluate_parser_for",
            return_value=fake_parser,
        ),
        patch("qai_hub_models.cli.dispatch.QAIHMModelManifest.from_model") as mock_info,
        patch(
            "qai_hub_models.cli.dispatch.check_unpublished_model_warning",
            return_value=False,
        ) as mock_check,
        patch("qai_hub_models.cli.dispatch.select_evaluate_pipeline") as mock_select,
    ):
        mock_info.return_value.status.value = "pending"
        fake_parser.parse_args = Mock(return_value=fake_parsed)

        run_model_script("sam", "evaluate", [])

        mock_check.assert_called_once()
        mock_select.assert_not_called()
