# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import requests
from packaging.version import Version

from qai_hub_models_cli.common import (
    AIHUB_MODELS_URL,
    ASSET_FOLDER,
    STORE_URL,
)
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers.platform import (
    precision_proto_to_str,
    runtime_proto_to_str,
)
from qai_hub_models_cli.proto_helpers.release_assets import get_model_asset_details
from qai_hub_models_cli.utils import download, get_next_free_path
from qai_hub_models_cli.versions import (
    CURRENT_VERSION,
)

ASSET_FILENAME = "{model_id}-{runtime}-{precision}.zip"
ASSET_CHIPSET_FILENAME = (
    "{model_id}-{runtime}-{precision}-{chipset_with_underscores}.zip"
)


def _normalize_runtime(runtime: Runtime.ValueType | str) -> str:
    if isinstance(runtime, int):
        return runtime_proto_to_str(runtime)
    return runtime


def _normalize_precision(precision: Precision.ValueType | str) -> str:
    if isinstance(precision, int):
        return precision_proto_to_str(precision)
    return precision


def _asset_url(
    model_id: str,
    runtime: str,
    precision: str,
    version: Version,
    chipset: str | None = None,
) -> tuple[str, str]:
    """Return (url, filename) for the asset."""
    model_id = model_id.lower()
    runtime_str = runtime.lower()
    precision_str = precision.lower()
    ver = str(version)
    if chipset is not None:
        filename = ASSET_CHIPSET_FILENAME.format(
            model_id=model_id,
            runtime=runtime_str,
            precision=precision_str,
            chipset_with_underscores=chipset.lower().replace("-", "_"),
        )
    else:
        filename = ASSET_FILENAME.format(
            model_id=model_id,
            runtime=runtime_str,
            precision=precision_str,
        )
    folder = ASSET_FOLDER.format(model_id=model_id, version=ver)
    url = f"{STORE_URL}/{folder}/{filename}"
    return url, filename


def get_asset_url(
    model: str,
    runtime: Runtime.ValueType | str,
    precision: Precision.ValueType | str,
    version: Version,
    chipset: str | None = None,
) -> str:
    """
    Resolve the download URL for a model asset.

    Parameters
    ----------
    model
        Model Name or ID (e.g. ``"mobilenet_v2"``).
    runtime
        Target runtime (e.g. ``RUNTIME_TFLITE`` or ``"tflite"``).
    precision
        Model precision (e.g. ``PRECISION_FLOAT`` or ``"float"``).
    version
        AI Hub Models version.
    chipset
        Optional chipset name.

    Returns
    -------
    str
        URL for the asset that exists.

    Raises
    ------
    FileNotFoundError
        If the asset does not exist on the server.
    """
    if version >= Version("0.50.1"):
        asset = get_model_asset_details(model, runtime, precision, chipset, version)
        return asset.download_url

    # Legacy: No manifest was published for these releases.
    def _head(url: str) -> int:
        resp = requests.head(url, timeout=10)
        if resp.status_code not in (200, 403, 404):
            raise ConnectionError(
                f"Unexpected response checking asset availability "
                f"(status {resp.status_code})."
            )
        return resp.status_code

    runtime_s = _normalize_runtime(runtime)
    precision_s = _normalize_precision(precision)
    if chipset is not None:
        url, _ = _asset_url(model, runtime_s, precision_s, version, chipset)
        if _head(url) == 200:
            return url

    url, _ = _asset_url(model, runtime_s, precision_s, version)
    if _head(url) == 200:
        return url

    chipset_msg = f", chipset={chipset!r}" if chipset else ""
    raise FileNotFoundError(
        f"No asset found for model={model!r}, runtime={runtime!r}, "
        f"precision={precision!r}, version={version!r}{chipset_msg}.\n"
        f"  - Browse available models: {AIHUB_MODELS_URL}\n"
        "  - List valid devices/chipsets: qai-hub list-devices (from the qai_hub package)"
    )


def fetch(
    model: str,
    runtime: Runtime.ValueType | str,
    output_dir: str | os.PathLike,
    precision: Precision.ValueType | str = "float",
    chipset: str | None = None,
    version: Version = CURRENT_VERSION,
    extract: bool = False,
    quiet: bool = False,
) -> Path:
    """
    Download a pre-compiled model asset from AI Hub Models.

    If a chipset is provided, the chipset-specific asset is tried first.
    If that does not exist, falls back to the generic asset.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``).
    runtime
        Target runtime (e.g. ``RUNTIME_TFLITE`` or ``"tflite"``).
    output_dir
        Output directory.
    precision
        Model precision (e.g. ``PRECISION_FLOAT`` or ``"float"``).
    chipset
        Chipset name for device-specific (AOT compiled) runtimes.
    version
        AI Hub Models version. Defaults to the installed CLI version.
    extract
        If True, extract the downloaded zip archive.
    quiet
        If True, suppress all output (progress bar, warnings, retry messages).

    Returns
    -------
    Path
        Path to the downloaded file (or extraction directory if *extract* is True).

    Raises
    ------
    FileNotFoundError
        If the asset does not exist on the server.
    """
    url = get_asset_url(model, runtime, precision, version, chipset)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = url.removeprefix("s3://").rsplit("/", 1)[-1]
    if extract:
        dst = get_next_free_path(out / Path(filename).stem)
    else:
        dst = get_next_free_path(out / filename)
    return download(url, dst, extract=extract, quiet=quiet)
