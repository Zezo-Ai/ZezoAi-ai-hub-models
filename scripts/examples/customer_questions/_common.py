# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Shared helpers for customer-question tooling.

Used by:
- customer_questions_slack_github_export.py (CSV exporter for human review)
- scripts/breeze_customer_questions/scrape_questions.py (Breeze agent input)

Keep helpers here narrow and side-effect-free. Anything Slack-write or
GitHub-write belongs in the calling scripts, not here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, TypedDict

from slack_sdk import WebClient
from slack_sdk.web.slack_response import SlackResponse

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Channels we never scrape (internal-only, automation noise, deprecated, etc.)
# -----------------------------------------------------------------------------
EXCLUDED_CHANNEL_IDS: set[str] = {
    "C07DMDFJAJX",
    "C07H7569R7H",
    "C0868KDMDEJ",
    "C08RV59CESC",
    "C089L6VEM28",
    "C08BNQTJLTW",
    "C09A5SF7HJA",
    "C08R9DQJKKM",
    "C09EVG667M0",
    "C09LLDL69QV",
    "C089S1F862K",
    "C099Y1GBDKK",
    "C07A5PJ7M8B",
    "C0840CTJGC8",  # lpcvc
}


# -----------------------------------------------------------------------------
# GitHub
# -----------------------------------------------------------------------------
GITHUB_REPOS: list[str] = [
    "qualcomm/ai-hub-models",
    "qualcomm/ai-hub-apps",
]

DEPENDABOT_LOGINS: set[str] = {"app/dependabot", "dependabot", "dependabot[bot]"}


# -----------------------------------------------------------------------------
# Slack types
# -----------------------------------------------------------------------------
class ChannelDict(TypedDict, total=False):
    id: str
    name: str


# Cache for user info: maps user_id -> (display_name, is_internal)
_user_cache: dict[str, tuple[str, bool]] = {}


def sanitize_text(text: str) -> str:
    """Strip Slack markdown and fix mojibake for clean downstream output."""
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"~(.+?)~", r"\1", text)
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", re.sub(r"```\w*\n?", "```", text))
    text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2 (\1)", text)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    # Mojibake fixup. Keys use explicit \uXXXX escapes so ruff RUF001
    # (ambiguous-unicode) is satisfied -- these characters are *deliberately*
    # 'ambiguous'; that is the whole point of mojibake.
    mojibake = {
        "\u00e2\u0080\u0094": "\u2014",  # em dash
        "\u00e2\u0080\u0093": "\u2013",  # en dash
        "\u00e2\u0080\u009c": '"',
        "\u00e2\u0080\u009d": '"',
        "\u00e2\u0080\u0099": "'",
        "\u00e2\u0080\u0098": "'",
        "\u00c2\u00a0": " ",
        "\u00e2\u0080\u00a6": "...",
        "\u201a\u00c4\u00ee": "\u2014",
        "\u201a\u00c4\u00f4": "'",
        "\u201a\u00c4\u00f2": '"',
        "\u201a\u00c4\u00fa": '"',
    }
    for bad, good in mojibake.items():
        text = text.replace(bad, good)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_qualcomm_user(user_data: Mapping[str, Any]) -> bool:
    """Return True if any profile field indicates a Qualcomm employee."""
    real_name = str(user_data.get("real_name", "")).lower()
    if real_name == "slackbot":
        return True
    profile = user_data.get("profile")
    if isinstance(profile, Mapping):
        display_name = str(profile.get("display_name", "")).lower()
        return "qualcomm" in real_name or "qualcomm" in display_name
    return "qualcomm" in real_name


def get_user_info(client: WebClient, user_id: str) -> tuple[str, bool]:
    """Return (display_name, is_internal) for a Slack user id. Caches results."""
    if not user_id:
        return ("Unknown", False)

    if user_id in _user_cache:
        return _user_cache[user_id]

    try:
        resp: SlackResponse = client.users_info(user=user_id)
        data = resp.data if hasattr(resp, "data") else resp  # type: ignore[assignment]
        if isinstance(data, Mapping):
            user = data.get("user")
            if isinstance(user, Mapping):
                internal = is_qualcomm_user(user)
                name = user.get("real_name")
                if isinstance(name, str) and name:
                    _user_cache[user_id] = (name, internal)
                    return (name, internal)
        return ("Unknown", False)
    except Exception as e:
        logger.warning(
            "Failed to resolve user_id=%s; returning 'Unknown'. Error: %s", user_id, e
        )
        return ("Unknown", False)
