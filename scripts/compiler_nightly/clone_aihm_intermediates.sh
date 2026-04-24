#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -e

AIHM_TAG="${1:-}"

if [ -z "$AIHM_TAG" ]; then
    echo "Fetching latest qai-hub-models version from PyPI"
    AIHM_TAG=$(curl -s https://pypi.org/pypi/qai-hub-models/json | jq -r '.info.version')
fi

if [[ ! "$AIHM_TAG" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] && [[ ! "$AIHM_TAG" =~ ^[0-9a-f]{7,40}$ ]]; then
  echo "ERROR: Invalid tag format (must be semver X.Y.Z or commit hash): $AIHM_TAG" >&2
  exit 1
fi

OUTPUT_DIR="intermediates_${AIHM_TAG}"
echo "Checking out intermediates from v${AIHM_TAG} to ${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

for file in compile-jobs.yaml link-jobs.yaml profile-jobs.yaml; do
    git show "v${AIHM_TAG}:src/qai_hub_models/scorecard/intermediates/${file}" > "${OUTPUT_DIR}/${file}"
done

echo "Done."
