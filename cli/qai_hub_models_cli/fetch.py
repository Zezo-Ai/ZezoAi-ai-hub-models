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
    ASSET_FOLDER,
    STORE_URL,
    Precision,
    TargetRuntime,
)
from qai_hub_models_cli.utils import download, get_next_free_path
from qai_hub_models_cli.versions import (
    CURRENT_VERSION,
    verify_not_dev_release,
    verify_version_supported,
)

ASSET_FILENAME = "{model_id}-{runtime}-{precision}.zip"
ASSET_CHIPSET_FILENAME = (
    "{model_id}-{runtime}-{precision}-{chipset_with_underscores}.zip"
)


def _asset_url(
    model_id: str,
    runtime: TargetRuntime | str,
    precision: Precision | str,
    version: Version,
    chipset: str | None = None,
) -> tuple[str, str]:
    """Return (url, filename) for the asset."""
    model_id = model_id.lower()
    runtime_str = (
        runtime.value if isinstance(runtime, TargetRuntime) else runtime.lower()
    )
    precision_str = (
        precision.value if isinstance(precision, Precision) else precision.lower()
    )
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
    model_id: str,
    runtime: TargetRuntime | str,
    precision: Precision | str,
    version: Version,
    chipset: str | None = None,
) -> str:
    """
    Resolve the download URL for a model asset.

    Parameters
    ----------
    model_id
        Model ID (e.g. ``"mobilenet_v2"``).
    runtime
        Target runtime.
    precision
        Model precision.
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
    # TODO(#18390): llama_cpp runtimes are not
    # distributed as standard S3 zips. Check if the model exists and
    # supports llama_cpp, then direct the user to the model README
    # for download instructions instead of failing on a missing asset.
    verify_not_dev_release(version)
    verify_version_supported(version)

    # TODO(#18374): Read available assets from a manifest
    # instead of making HEAD requests.
    #
    # Not all runtimes produce chipset-specific assets. Try the chipset URL
    # first, then fall back to the generic (non-chipset) URL.
    def _head(url: str) -> int:
        resp = requests.head(url, timeout=10)
        if resp.status_code not in (200, 403, 404):
            raise ConnectionError(
                f"Unexpected response checking asset availability "
                f"(status {resp.status_code})."
            )
        return resp.status_code

    if chipset is not None:
        url, _ = _asset_url(model_id, runtime, precision, version, chipset)
        if _head(url) == 200:
            return url

    url, _ = _asset_url(model_id, runtime, precision, version)
    if _head(url) == 200:
        return url

    chipset_msg = f", chipset={chipset!r}" if chipset else ""
    raise FileNotFoundError(
        f"No asset found for model={model_id!r}, runtime={runtime!r}, "
        f"precision={precision!r}, version={version!r}{chipset_msg}.\n"
        "  - Browse available models: https://aihub.qualcomm.com/models\n"
        "  - List valid devices/chipsets: qai-hub list-devices (from the qai_hub package)"
    )


def fetch(
    model: str,
    runtime: TargetRuntime | str,
    output_dir: str | os.PathLike,
    precision: Precision | str = Precision.FLOAT,
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
        Target runtime (e.g. ``TargetRuntime.QNN_DLC`` or ``"qnn_dlc"``).
    output_dir
        Output directory.
    precision
        Model precision (e.g. ``Precision.FLOAT`` or ``"w8a8"``).
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
    filename = url.rsplit("/", 1)[-1]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if extract:
        dst = get_next_free_path(out / Path(filename).stem)
    else:
        dst = get_next_free_path(out / filename)

    return download(url, dst, extract=extract, quiet=quiet)
