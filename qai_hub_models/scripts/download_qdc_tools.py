# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path

from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_qaihm_s3,
    s3_download,
)

# S3 keys for QDC tools
QDC_WHEEL_S3_KEY = "qai-hub-models/qdc/qdc_public_api_client-0.0.12-py3-none-any.whl"
QAIRT_AUTO_SDK_S3_KEY = "qai-hub-models/qairt/2.42.0-auto-20260106/artifact.zip"

# Local filenames
QDC_WHEEL_FILENAME = "qdc_public_api_client-0.0.12-py3-none-any.whl"
QAIRT_AUTO_SDK_FILENAME = "qairt_auto_sdk.zip"


def download_qdc_wheel(local_path: str) -> None:
    """Download the QDC public API client wheel from S3."""
    bucket, _ = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)
    s3_download(bucket, QDC_WHEEL_S3_KEY, local_path)


def download_qairt_auto_sdk(local_path: str) -> None:
    """Download the QAIRT SDK for automotive devices from S3."""
    bucket, _ = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)
    s3_download(bucket, QAIRT_AUTO_SDK_S3_KEY, local_path)


def main() -> None:
    # Download the QDC wheel
    download_qdc_wheel(str(Path.home() / QDC_WHEEL_FILENAME))

    # Download the QAIRT SDK for automotive devices
    download_qairt_auto_sdk(str(Path.home() / QAIRT_AUTO_SDK_FILENAME))


if __name__ == "__main__":
    main()
