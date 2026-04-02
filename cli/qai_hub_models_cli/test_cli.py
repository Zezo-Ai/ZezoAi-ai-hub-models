# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from qai_hub_models_cli.cli import _check_version_match, main


def test_cli_import() -> None:
    """Verify the CLI entry point is importable."""
    assert callable(main)


def test_cli_no_qai_hub_models(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the CLI prints help when qai_hub_models is not installed."""

    def _version_without_models(pkg: str) -> str:
        if pkg == "qai_hub_models":
            raise PackageNotFoundError(pkg)
        return "1.0.0"

    with (
        patch("qai_hub_models_cli.cli.version", side_effect=_version_without_models),
        patch.dict("sys.modules", {"qai_hub_models": None, "qai_hub_models.cli": None}),
    ):
        main([])
    captured = capsys.readouterr()
    assert "Qualcomm AI Hub Models CLI" in captured.out


def test_version_mismatch_raises() -> None:
    """Verify a RuntimeError is raised when package versions differ."""
    with (
        patch(
            "qai_hub_models_cli.cli.version",
            side_effect=lambda pkg: "1.0.0" if pkg == "qai_hub_models_cli" else "2.0.0",
        ),
        pytest.raises(RuntimeError, match="Version mismatch"),
    ):
        _check_version_match()


def test_version_match_ok() -> None:
    """Verify no error when package versions match."""
    with patch(
        "qai_hub_models_cli.cli.version",
        return_value="1.0.0",
    ):
        _check_version_match()


def test_version_check_models_not_installed() -> None:
    """Verify no error when qai_hub_models is not installed."""

    def _version_without_models(pkg: str) -> str:
        if pkg == "qai_hub_models":
            raise PackageNotFoundError(pkg)
        return "1.0.0"

    with patch("qai_hub_models_cli.cli.version", side_effect=_version_without_models):
        _check_version_match()
