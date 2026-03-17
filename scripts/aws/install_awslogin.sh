# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

REPO_ROOT=$(git rev-parse --show-toplevel)

. "${REPO_ROOT}/scripts/util/common.sh"

set_strict_mode

if [ -z "${QAIHM_AWSLOGIN_URL:-}" ]; then
    echo "Error: QAIHM_AWSLOGIN_URL is not set. Please set it to the awslogin package URL. See https://qualcomm-confluence.atlassian.net/wiki/spaces/ML/pages/3188064594/Private+AWS+Access+Setup for setup instructions." >&2
    exit 1
fi

if command -v uv &> /dev/null; then
    uv pip install "awslogin@${QAIHM_AWSLOGIN_URL}"
else
    pip install "awslogin@${QAIHM_AWSLOGIN_URL}"
fi
