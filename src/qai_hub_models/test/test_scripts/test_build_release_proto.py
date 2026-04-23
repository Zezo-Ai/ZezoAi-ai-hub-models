# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from google.protobuf.json_format import Parse
from qai_hub_models_cli.proto import manifest_pb2

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
    manifest_json = output_dir / "manifest.json"
    manifest_pb = output_dir / "manifest.pb"
    assert manifest_json.exists()
    assert manifest_pb.exists()

    for manifest_path, ext in [(manifest_json, ".json"), (manifest_pb, ".pb")]:
        manifest = manifest_pb2.ReleaseManifest()
        if ext == ".json":
            Parse(manifest_path.read_text(), manifest)
        else:
            manifest.ParseFromString(manifest_path.read_bytes())

        assert manifest.version == __version__
        assert manifest.platform_url.endswith(f"platform{ext}")

        model_ids_in_manifest = {entry.id for entry in manifest.models}
        for model_id in sorted(SAMPLE_MODELS):
            assert model_id in model_ids_in_manifest

        for entry in manifest.models:
            assert entry.display_name
            assert entry.domain
            assert entry.manifest_urls.info.endswith(f"/info{ext}")

    for model_id in sorted(SAMPLE_MODELS):
        model_dir = output_dir / "models" / model_id
        assert model_dir.exists()
        assert (model_dir / "info.json").exists()
        assert (model_dir / "info.pb").exists()
