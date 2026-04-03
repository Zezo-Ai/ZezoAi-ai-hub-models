# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qai_hub_models_cli.utils import (
    MIN_SUPPORTED_VERSION,
    UnsupportedVersionError,
    download,
    extract_zip_file,
    get_next_free_path,
    normalize_version,
    validate_version,
)

# ── normalize_version ────────────────────────────────────────────────


def test_normalize_version_strips_v_prefix() -> None:
    assert normalize_version("v0.45.0") == "0.45.0"


def test_normalize_version_no_prefix() -> None:
    assert normalize_version("0.45.0") == "0.45.0"


def test_normalize_version_lowercases() -> None:
    assert normalize_version("V0.45.0") == "0.45.0"


def test_normalize_version_double_v() -> None:
    """Removeprefix only strips one leading 'v'."""
    assert normalize_version("vv0.45.0") == "v0.45.0"


# ── validate_version ────────────────────────────────────────────────


def test_validate_version_below_floor() -> None:
    with pytest.raises(UnsupportedVersionError, match="not supported"):
        validate_version("v0.43.0")


def test_validate_version_at_floor() -> None:
    validate_version(f"v{MIN_SUPPORTED_VERSION}")


def test_validate_version_dev_version_rejected() -> None:
    with pytest.raises(UnsupportedVersionError, match="dev version"):
        validate_version("0.45.0.dev1")


def test_validate_version_above_installed_on_release() -> None:
    with (
        patch("qai_hub_models_cli.utils.__version__", "0.45.0"),
        pytest.raises(UnsupportedVersionError, match="newer than the installed"),
    ):
        validate_version("v0.46.0")


def test_validate_version_above_installed_ok_on_dev() -> None:
    """Ceiling check is skipped on dev installs."""
    with patch("qai_hub_models_cli.utils.__version__", "0.45.0.dev1"):
        validate_version("v0.99.0")


def test_validate_version_at_installed() -> None:
    with patch("qai_hub_models_cli.utils.__version__", "0.45.0"):
        validate_version("v0.45.0")


# ── get_next_free_path ───────────────────────────────────────────────


def test_get_next_free_path_no_conflict(tmp_path: Path) -> None:
    result = get_next_free_path(tmp_path / "file.zip")
    assert result == tmp_path / "file.zip"


def test_get_next_free_path_increments(tmp_path: Path) -> None:
    (tmp_path / "file.zip").touch()
    result = get_next_free_path(tmp_path / "file.zip")
    assert result == tmp_path / "file-1.zip"


def test_get_next_free_path_increments_multiple(tmp_path: Path) -> None:
    (tmp_path / "file.zip").touch()
    (tmp_path / "file-1.zip").touch()
    (tmp_path / "file-2.zip").touch()
    result = get_next_free_path(tmp_path / "file.zip")
    assert result == tmp_path / "file-3.zip"


def test_get_next_free_path_directory(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    result = get_next_free_path(tmp_path / "dir")
    assert result == tmp_path / "dir-1"


# ── extract_zip_file ────────────────────────────────────────────────


def _create_zip(zip_path: Path, files: dict[str, str]) -> None:
    """Create a zip file with the given {path: content} mapping."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_extract_zip_single_dir_unwrap(tmp_path: Path) -> None:
    zip_path = tmp_path / "test.zip"
    _create_zip(zip_path, {"topdir/a.txt": "a", "topdir/b.txt": "b"})

    out = extract_zip_file(zip_path, tmp_path / "extracted")
    assert out == tmp_path / "extracted"
    assert (out / "a.txt").read_text() == "a"
    assert (out / "b.txt").read_text() == "b"


def test_extract_zip_multiple_top_level(tmp_path: Path) -> None:
    zip_path = tmp_path / "test.zip"
    _create_zip(zip_path, {"a.txt": "a", "b.txt": "b"})

    out = extract_zip_file(zip_path, tmp_path / "extracted")
    assert (out / "a.txt").read_text() == "a"
    assert (out / "b.txt").read_text() == "b"


def test_extract_zip_default_out_path(tmp_path: Path) -> None:
    zip_path = tmp_path / "archive.zip"
    _create_zip(zip_path, {"file.txt": "data"})

    out = extract_zip_file(zip_path)
    assert out == tmp_path / "archive"
    assert (out / "file.txt").read_text() == "data"


def test_extract_zip_raises_if_exists(tmp_path: Path) -> None:
    zip_path = tmp_path / "test.zip"
    _create_zip(zip_path, {"file.txt": "data"})
    (tmp_path / "extracted").mkdir()

    with pytest.raises(FileExistsError, match="Cannot extract"):
        extract_zip_file(zip_path, tmp_path / "extracted")


# ── download ─────────────────────────────────────────────────────────


def test_download_raises_if_exists(tmp_path: Path) -> None:
    dst = tmp_path / "file.bin"
    dst.touch()
    with pytest.raises(FileExistsError):
        download("http://example.com/file.bin", dst)


def test_download_basic(tmp_path: Path) -> None:
    content = b"hello world"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": str(len(content))}
    mock_response.iter_content = lambda chunk_size: [content]

    with patch("qai_hub_models_cli.utils.requests.get", return_value=mock_response):
        result = download("http://example.com/file.bin", tmp_path / "file.bin")

    assert result == tmp_path / "file.bin"
    assert result.read_bytes() == content


def test_download_with_extract(tmp_path: Path) -> None:
    # Create a real zip in memory.
    zip_path = tmp_path / "source.zip"
    _create_zip(zip_path, {"topdir/data.txt": "payload"})
    zip_bytes = zip_path.read_bytes()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": str(len(zip_bytes))}
    mock_response.iter_content = lambda chunk_size: [zip_bytes]

    with patch("qai_hub_models_cli.utils.requests.get", return_value=mock_response):
        result = download(
            "http://example.com/model.zip", tmp_path / "model.zip", extract=True
        )

    assert result == tmp_path / "model"
    assert (result / "data.txt").read_text() == "payload"
    # Zip should not be kept.
    assert not (tmp_path / "model.zip").exists()
