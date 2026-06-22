# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Generator
from unittest.mock import patch

import pytest

from qai_hub_models_cli.cli import main
from qai_hub_models_cli.proto.platform_pb2 import PlatformInfo, RuntimeInfo
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime


def _fake_platform() -> PlatformInfo:
    # Display fields come straight from the proto (populated at release time
    # from TargetRuntime in qai_hub_models.common).
    return PlatformInfo(
        runtimes=[
            RuntimeInfo(
                runtime=Runtime.RUNTIME_QNN_DLC,
                file_extension=".dlc",
                is_aot_compiled=True,
                display_name="QAIRT DLC",
                description="Qualcomm AI Engine Direct deep-learning container.",
                documentation_url="https://example.com/qairt",
            ),
            RuntimeInfo(
                runtime=Runtime.RUNTIME_TFLITE,
                file_extension=".tflite",
                is_aot_compiled=False,
                display_name="TensorFlow Lite",
                description="LiteRT (TensorFlow Lite).",
                documentation_url="https://example.com/litert",
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _mocks() -> Generator[None]:
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.get_platform", return_value=_fake_platform()),
    ):
        yield


def test_runtimes_table(capsys: pytest.CaptureFixture[str]) -> None:
    main(["runtimes"])
    output = capsys.readouterr().out
    # Token + display name in the table, docs URL in the footnote.
    assert "tflite" in output and "TensorFlow Lite" in output
    assert "Learn more:" in output and "https://example.com/qairt" in output


def test_runtimes_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    main(["runtimes", "-q"])
    assert capsys.readouterr().out.strip().splitlines() == ["qnn_dlc", "tflite"]


def test_runtimes_old_version_omits_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Before 0.56.0 the display-metadata columns are omitted, but the table
    # (ID/Ext/Compiled) still renders.
    main(["runtimes", "-v", "0.55.0"])
    output = capsys.readouterr().out
    assert "qnn_dlc" in output and ".dlc" in output
    assert "Name" not in output and "Description" not in output
    assert "Learn more:" not in output
