#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Check if a GitHub user is a member of any specified team.
#
# Usage: check_github_team_membership.sh <username> <team1> [team2...]
#
# Optional env: GH_TOKEN (if not set, uses existing gh auth)
# Prints "true" if the user is in any listed team, "false" otherwise.

set -euo pipefail

USERNAME="$1"
ORG="qcom-ai-hub"

RESULT="false"
for team in "${@:2}"; do
  if [ "$(gh api "orgs/${ORG}/teams/${team}/memberships/${USERNAME}" --jq '.state' || true)" = "active" ]; then
    echo "Author '${USERNAME}' is in ${ORG}/${team}." >&2
    RESULT="true"
    break
  fi
done

echo "$RESULT"
