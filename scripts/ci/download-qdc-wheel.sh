# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
#
# Downloads the QDC SDK zip, verifies its checksum, extracts the wheel,
# copies it to the repo root, and cleans up temporary files.
#
# Usage: download-qdc-wheel.sh <repo_root>

set -euo pipefail

QDC_SDK_URL="https://softwarecenter.qualcomm.com/api/download/software/tools/Qualcomm_Device_Cloud_SDK/Windows/0.2.3/qualcomm_device_cloud_sdk-0.2.3.zip"
QDC_SDK_SHA256="ff14974c134dae8064ba15a8d78ebc62c480573c947c612f929718bd1c406d27"
REPO_ROOT="$1"

TMP_ZIP=/tmp/qualcomm_device_cloud_sdk.zip
TMP_DIR=/tmp/qualcomm_device_cloud_sdk

curl -fSL --max-time 120 -o "$TMP_ZIP" "$QDC_SDK_URL"
echo "$QDC_SDK_SHA256  $TMP_ZIP" | sha256sum -c -
unzip -q "$TMP_ZIP" -d "$TMP_DIR"

wheels=("$TMP_DIR"/*.whl)
[ "${#wheels[@]}" -eq 1 ] || { echo "Expected exactly one .whl, found ${#wheels[@]}"; exit 1; }
cp "${wheels[0]}" "$REPO_ROOT/"

rm -rf "$TMP_DIR" "$TMP_ZIP"
