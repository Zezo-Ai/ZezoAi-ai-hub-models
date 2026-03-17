# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import configparser
import contextlib
import getpass
import logging
import os
import sys
from typing import TypedDict

import boto3
import botocore
import botocore.exceptions
from awslogin.auth import IdP
from awslogin.profile import Profile


# This maintains a mapping from colloquial account names to:
# - account ID
# - the name of the AWS profile created locally. This is used by the AWS CLI and SDKs
# - the IAM role assumed in that account
#
# The account ID and roles are configured via environment variables:
#   QAIHM_AWS_ACCOUNT_ID  - AWS account ID
#   QAIHM_AWS_ROLE        - IAM role for the primary account
#   QAIHM_AWS_ADMIN_ROLE  - IAM role for the admin account
class AccountData(TypedDict):
    id: str
    profile: str
    role: str
    required: bool


def _load_accounts() -> dict[str, AccountData]:
    account_id = os.environ.get("QAIHM_AWS_ACCOUNT_ID", "")
    role = os.environ.get("QAIHM_AWS_ROLE", "")
    admin_role = os.environ.get("QAIHM_AWS_ADMIN_ROLE", "")

    if not account_id or not role or not admin_role:
        raise ValueError(
            "Missing required environment variables. Set "
            "QAIHM_AWS_ACCOUNT_ID, QAIHM_AWS_ROLE, and QAIHM_AWS_ADMIN_ROLE. "
            "See https://qualcomm-confluence.atlassian.net/wiki/spaces/ML/pages/3188064594/Private+AWS+Access+Setup for setup instructions."
        )

    return {
        "qaihm": {
            "id": account_id,
            "profile": "qaihm",
            "role": role,
            "required": True,
        },
        "qaihm-admin": {
            "id": account_id,
            "profile": "qaihm-admin",
            "role": admin_role,
            "required": False,
        },
    }


def profiles_exist(accounts: dict[str, AccountData]) -> bool:
    """
    Checks whether local AWS profile entries exist for all accounts we care about.

    Returns True if all profiles exist, False otherwise.
    """
    try:
        for account in accounts.values():
            boto3.Session(profile_name=account["profile"])
    except botocore.exceptions.ProfileNotFound as e:
        logging.warning(e)
        return False

    return True


def add_profiles(accounts: dict[str, AccountData]) -> None:
    """Adds local AWS profile entries for all accounts we care about."""
    config = configparser.ConfigParser()
    config_file = os.path.expanduser(f"~{os.sep}.aws{os.sep}config")

    # Make the .aws directory if it doesn't exist.
    config_dir = os.path.dirname(config_file)
    os.makedirs(config_dir, exist_ok=True)

    config.read(config_file)
    for profile in [a["profile"] for a in accounts.values()]:
        with contextlib.suppress(configparser.DuplicateSectionError):
            config.add_section(f"profile {profile}")

        config.set(f"profile {profile}", "region", "us-west-2")

    with open(config_file, "w") as f:
        config.write(f)


def prune_default() -> None:
    """
    Removes any duplicate profile entries for the "default" profile.

    AWS allows the default profile to be entered as [profile default] or [default] in the config file.
    The "add_profiles" method above adds all profiles with the same "profile name" section header,
    so if users have an existing [default] section, they will end up with two different default sections.
    This works with the AWS CLI, but CDK blows up, so delete the [default] section if it exists.
    """
    config = configparser.ConfigParser()
    config_file = os.path.expanduser(f"~{os.sep}.aws{os.sep}config")
    config.read(config_file)

    config.remove_section("default")

    with open(config_file, "w") as f:
        config.write(f)


def credentials_valid(accounts: dict[str, AccountData], all_accounts: bool) -> bool:
    """
    Checks whether valid AWS credentials exist for all accounts we care about.

    Parameters
    ----------
    accounts
        The accounts to check.
    all_accounts
        If True, checks all accounts instead of only the ones marked as required.

    Returns
    -------
    valid : bool
        True if valid credentials are found for all specified accounts, False otherwise.
    """
    for account in accounts.values():
        session = boto3.Session(profile_name=account["profile"])
        sts_client = session.client("sts")

        try:
            sts_client.get_caller_identity()
        except (
            botocore.exceptions.NoCredentialsError,
            botocore.exceptions.ClientError,
        ) as e:
            if account["required"] or all_accounts:
                logging.exception(
                    f"Could not get caller identity for aws profile '{account['profile']}'"
                )
                logging.exception(e)  # noqa: TRY401
                return False
            continue

    return True


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # The required accounts have IAM roles with a longer duration (8 hours) while the non-required ones have a shorter duration (1 hour).
    # The required roles are needed for build_and_test and for running Hub locally, while the others are just nice-to-have.
    # We want to ensure that valid credentials are always available for the required accounts, while credentials for the other accounts
    # are fetched only if the user explicitly asks for them.
    # This is so that most users can get away with needing to enter their password only once a day to get fresh credentials.
    argparser.add_argument("--all_accounts", action="store_true")
    args = argparser.parse_args()

    accounts = _load_accounts()

    if not profiles_exist(accounts):
        logging.info("Creating AWS profile entries")
        add_profiles(accounts)

    prune_default()

    if not credentials_valid(accounts, args.all_accounts):
        logging.info("Obtaining AWS credentials")

        if not sys.stdin.isatty():
            raise RuntimeError(
                "This is not a TTY and hence this script cannot prompt you for your password. Please re-run this in a different, interactive terminal."
            )

        password = getpass.getpass(prompt="Qualcomm password: ")
        username = getpass.getuser()

        idp = IdP(username, password)
        profile = Profile()

        for account in accounts.values():
            session = boto3.Session(profile_name=account["profile"])
            sts_client = session.client("sts")
            role_arn = f"arn:aws:iam::{account['id']}:role/{account['role']}"
            # Assume the role for 8 hours
            try:
                session = idp.assume_saml_role(role_arn, duration=8 * 60 * 60)
                logging.info(
                    f"Writing credentials for '{role_arn}' as profile '{account['profile']}'"
                )
                profile.set(account["profile"], session)
            except:  # noqa: E722
                if account["required"]:
                    raise
                else:
                    # External contributors don't have access to every account. If the account isn't required,
                    # continue without raising an exception.
                    continue

        profile.write()
