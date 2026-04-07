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
    download,
    extract_zip_file,
    get_next_free_path,
)

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


# ── extract_zip_file path traversal ─────────────────────────────────


def test_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../../etc/passwd", "pwned")

    with pytest.raises(ValueError, match="Unsafe zip entry"):
        extract_zip_file(zip_path, tmp_path / "extracted")
