#!/bin/bash

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Error: This script must be run, not sourced." >&2
  return 1
fi

# Arguments
deployments="$1" # List of deployments separated by commas (e.g., "prod,staging,dev")
prod_token=$2
staging_token=$3
dev_token=$4
sandbox_token=$5
user=${6:-"DEFAULT"}
venv_path=$7

user_lower=$(echo "$user" | tr '[:upper:]' '[:lower:]')

# Convert the deployments list to an array
IFS=',' read -r -a hub_deployments <<< "$deployments"

# Activate venv if provided (needed for qai-hub CLI)
if [ -n "$venv_path" ]; then
  # shellcheck disable=SC1091
  source "${venv_path}/bin/activate"
fi

# Configure a qai-hub profile for each deployment
configured_profiles=""
for i in "${!hub_deployments[@]}"; do
  deployment=$(echo "${hub_deployments[$i]}" | tr '[:upper:]' '[:lower:]')

  # Resolve token and URL for this deployment
  case "$deployment" in
    prod|workbench|app)
      token=$prod_token
      deployment_url_name="workbench"
      deployment="prod"
      ;;
    staging)
      token=$staging_token
      deployment_url_name="staging"
      ;;
    dev)
      token=$dev_token
      deployment_url_name="dev"
      ;;
    *)
      token=$sandbox_token
      if [[ "$deployment" == *.sandbox ]]; then
        deployment_url_name=$deployment
      else
        deployment_url_name="${deployment}.sandbox"
      fi
      ;;
  esac

  # Skip deployments without a token
  if [ -z "$token" ]; then
    echo "Skipping $deployment (no token)"
    continue
  fi

  url="https://${deployment_url_name}.aihub.qualcomm.com/"

  # Build profile name: "<deployment>" for default user, "<deployment>_<user>" otherwise
  if [ "$user_lower" == "default" ]; then
    profile_name="${deployment}"
  else
    profile_name="${deployment}_${user_lower}"
  fi

  # Skip duplicate profiles (e.g., "prod" and "workbench" both resolve to "prod")
  if [[ "$configured_profiles" == *"|$profile_name|"* ]]; then
    continue
  fi
  configured_profiles="${configured_profiles}|$profile_name|"

  echo "Configuring profile: $profile_name -> $url"
  qai-hub configure \
    --api_token "$token" \
    --api_url "$url" \
    --web_url "$url" \
    --profile "$profile_name" \
    --no-verbose

  # Set the first deployment as the default client
  if [ "$i" -eq 0 ]; then
    if [ "$user_lower" != "default" ] && [ -f "$HOME/.qai_hub/client.ini" ]; then
      echo "Skipping default client override for non-default user: $user"
    else
      cp "$HOME/.qai_hub/${profile_name}.ini" "$HOME/.qai_hub/client.ini"
      echo "Set default client to profile: $profile_name"
    fi
  fi
done
