# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import requests
from packaging.version import parse as parse_version
from tqdm import tqdm

from qai_hub_models_cli._version import __version__

MIN_SUPPORTED_VERSION = "0.44.0"


class UnsupportedVersionError(RuntimeError):
    pass


def normalize_version(version: str) -> str:
    """Lowercase and strip leading 'v' from a version string."""
    return version.lower().removeprefix("v")


def validate_version(version: str) -> None:
    """
    Validate that a requested version is readable by this version of the CLI.

    Ensures *version* is at or above ``MIN_SUPPORTED_VERSION``,
    which is the earliest release that published per-device assets to S3.

    Also ensures *version* does not exceed the currently installed package
    version, since assets for a newer release may rely on schema or runtime
    changes not present in the current install. This ceiling check is
    skipped on dev builds, which are always ahead of the latest release.

    Parameters
    ----------
    version
        Requested version string.

    Raises
    ------
    ValueError
        If *version* is below the minimum or above the installed version.
    """
    requested = parse_version(normalize_version(version))

    if requested.is_devrelease:
        raise UnsupportedVersionError(
            f"Version {version} is a dev version and has no published assets. "
            f"Provide a release version (e.g. v0.45.0)."
        )

    floor = parse_version(MIN_SUPPORTED_VERSION)
    if requested < floor:
        raise UnsupportedVersionError(
            f"Version {version} is not supported. Minimum supported version is v{MIN_SUPPORTED_VERSION}."
        )

    installed = parse_version(__version__)
    if not installed.is_devrelease and requested > installed:
        raise UnsupportedVersionError(
            f"Version {version} is newer than the installed version ({__version__}). "
            f"Upgrade the package or use version v{__version__} or earlier."
        )


def get_next_free_path(path: str | os.PathLike, delim: str = "-") -> Path:
    """Adds an incrementing number at the end of the given path until a non-existent filename is found."""
    path_without_ext, ext = os.path.splitext(path)

    counter = 1
    while os.path.exists(path):
        path = f"{path_without_ext}{delim}{counter}{ext}"
        counter += 1

    return Path(path)


_RETRYABLE_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ContentDecodingError,
    requests.exceptions.Timeout,
)


def _get_total_size(response: requests.Response, bytes_downloaded: int) -> int:
    """Parse total file size from Content-Range or Content-Length headers."""
    content_range = response.headers.get("content-range", "")
    match = re.search(r"/(\d+)\s*$", content_range)
    if match:
        return int(match.group(1))
    return bytes_downloaded + int(response.headers.get("content-length", 0))


def _download_to_file(
    url: str, tmp_filepath: str, num_retries: int, quiet: bool = False
) -> None:
    """Download *url* into *tmp_filepath* with resume and retry."""
    filename = url.rsplit("/", 1)[-1]

    for attempt in range(num_retries + 1):
        bytes_downloaded = (
            os.path.getsize(tmp_filepath) if os.path.exists(tmp_filepath) else 0
        )
        headers = {"Range": f"bytes={bytes_downloaded}-"} if bytes_downloaded else {}

        try:
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            if response.status_code not in (200, 206):
                raise requests.HTTPError(
                    f"Unable to download file at {url} (status {response.status_code})",
                    response=response,
                )

            # Server ignored range request — restart from scratch.
            if bytes_downloaded > 0 and response.status_code == 200:
                bytes_downloaded = 0

            mode = "ab" if bytes_downloaded > 0 else "wb"
            total_size = _get_total_size(response, bytes_downloaded)

            with tqdm(
                total=total_size,
                initial=bytes_downloaded,
                unit="B",
                unit_scale=True,
                disable=quiet,
            ) as bar:
                bar.set_description(f"Downloading {filename}")
                with open(tmp_filepath, mode) as f:
                    for chunk in response.iter_content(1024 * 1024):
                        bar.update(len(chunk))
                        f.write(chunk)
                bar.set_postfix_str("Done")

            actual_size = os.path.getsize(tmp_filepath)
            if not total_size:
                if not quiet:
                    print(
                        "Warning: server did not report file size; "
                        "cannot verify download completeness."
                    )
            elif actual_size < total_size:
                if attempt < num_retries:
                    if not quiet:
                        print(
                            f"Download incomplete ({actual_size}/{total_size} bytes), "
                            f"retrying ({attempt + 1}/{num_retries})..."
                        )
                    continue
                raise ConnectionError(
                    f"Download incomplete after {num_retries} retries: "
                    f"got {actual_size}/{total_size} bytes from {url}"
                )
            return

        except _RETRYABLE_ERRORS as e:
            if attempt < num_retries:
                if not quiet:
                    print(
                        f"Download interrupted ({e.__class__.__name__}), "
                        f"retrying ({attempt + 1}/{num_retries})..."
                    )
            else:
                raise


def download(
    url: str,
    dst_path: str | os.PathLike,
    num_retries: int = 4,
    extract: bool = False,
    quiet: bool = False,
) -> Path:
    """
    Download a file from *url* to *dst_path* with a progress bar.

    Supports resuming partial downloads on connection failures.
    If *extract* is True, the downloaded zip is extracted directly from the
    temp directory and the zip file is not kept.

    Parameters
    ----------
    url
        URL to download.
    dst_path
        Destination file path.
    num_retries
        Number of retry attempts on connection failure or incomplete download.
    extract
        If True, extract the downloaded zip and return the extraction directory
        (``dst_path`` without the ``.zip`` extension).
    quiet
        If True, suppress all output (progress bar, warnings, retry messages).

    Returns
    -------
    Path
        Path to the downloaded file, or extraction directory if *extract* is True.

    Raises
    ------
    FileExistsError
        If *dst_path* already exists.
    requests.HTTPError
        If the server returns a non-200/206 status.
    ConnectionError
        If the download is incomplete after all retries.
    """
    dst = Path(dst_path)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_filepath = os.path.join(tmp_dir, dst.name)
        _download_to_file(url, tmp_filepath, num_retries, quiet=quiet)

        if extract:
            return extract_zip_file(tmp_filepath, dst.parent / dst.stem)

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_filepath, dst)

    return dst


def extract_zip_file(
    filepath: str | os.PathLike,
    out_path: str | os.PathLike | None = None,
) -> Path:
    """
    Extract a zip file's contents into *out_path*.

    If the archive contains a single top-level directory, its contents
    are unwrapped directly into *out_path*. If the archive contains multiple
    top-level entries, they are all placed into *out_path* as-is.

    Parameters
    ----------
    filepath
        Path to the zip file.
    out_path
        Destination directory. If None, defaults to the zip file path
        without the .zip extension. Must not already exist.

    Returns
    -------
    Path
        Path to the extracted directory.
    """
    filepath = Path(filepath)
    out_path = Path(out_path if out_path else filepath.parent / filepath.stem)

    if out_path.exists():
        raise FileExistsError(f"Cannot extract to an existing directory: {out_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(filepath, "r") as zf:
            tmp_resolved = Path(tmp).resolve()
            for member in zf.namelist():
                if not (tmp_resolved / member).resolve().is_relative_to(tmp_resolved):
                    raise ValueError(f"Unsafe zip entry detected: {member!r}")
            zf.extractall(path=tmp)

        top_level_files = list(Path(tmp).iterdir())
        if len(top_level_files) == 1 and top_level_files[0].is_dir():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(top_level_files[0], out_path)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            for file in top_level_files:
                shutil.move(file, out_path / file.name)

    return out_path
