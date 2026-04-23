# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from qai_hub_models._version import __version__
from qai_hub_models.scripts.build_release_proto import cmd_aws, cmd_website

SAMPLE_MODELS = {"mobilenet_v2", "aotgan"}


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "release_output"


def test_cmd_website(output_dir: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(output_dir), version=__version__, models=SAMPLE_MODELS
    )
    cmd_website(args)

    assert output_dir.exists()
    assert (output_dir / "asset_bases.yaml").exists()
    assert (output_dir / "devices_and_chipsets.yaml").exists()

    for model_id in sorted(SAMPLE_MODELS):
        model_dir = output_dir / "models" / model_id
        assert model_dir.exists()
        assert (model_dir / "info.yaml").exists()


def test_cmd_aws(output_dir: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(output_dir),
        version=__version__,
        models=SAMPLE_MODELS,
        upload=False,
    )
    cmd_aws(args)

    assert output_dir.exists()
    assert (output_dir / "platform.json").exists()
    assert (output_dir / "platform.pb").exists()

    for model_id in sorted(SAMPLE_MODELS):
        model_dir = output_dir / "models" / model_id
        assert model_dir.exists()
        assert (model_dir / "info.json").exists()
        assert (model_dir / "info.pb").exists()
