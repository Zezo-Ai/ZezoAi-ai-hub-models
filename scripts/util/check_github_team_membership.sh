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

_team_member_state() {
  local team="$1"
  local url="https://api.github.com/orgs/${ORG}/teams/${team}/memberships/${USERNAME}"
  if [ -n "${GH_TOKEN:-}" ]; then
    curl -s \
      -H "Authorization: Bearer ${GH_TOKEN}" \
      -H "Accept: application/vnd.github.v3+json" \
      "${url}" 2>/dev/null \
      | jq -r '.state // empty' 2>/dev/null || true
  else
    gh api "orgs/${ORG}/teams/${team}/memberships/${USERNAME}" --jq '.state' || true
  fi
}

RESULT="false"
# shellcheck disable=SC2068
for team in ${@:2}; do
  if [ "$(_team_member_state "$team")" = "active" ]; then
    echo "Author '${USERNAME}' is in ${ORG}/${team}." >&2
    RESULT="true"
    break
  fi
done

echo "$RESULT"
