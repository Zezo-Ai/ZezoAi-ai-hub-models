# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from slack_sdk import WebClient
from slack_sdk.web.slack_response import SlackResponse

# Run as a script → add this directory to sys.path so we can import _common.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    DEPENDABOT_LOGINS,
    EXCLUDED_CHANNEL_IDS,
    GITHUB_REPOS,
    ChannelDict,
    get_user_info,
    sanitize_text,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Configuration: read token from environment. Do NOT hard-code secrets.
# -----------------------------------------------------------------------------
SLACK_BOT_TOKEN_VAR = "SLACK_BOT_TOKEN"
slack_token = os.getenv(SLACK_BOT_TOKEN_VAR)
if not slack_token:
    raise RuntimeError(
        f"Environment variable {SLACK_BOT_TOKEN_VAR} is not set. "
        "Export your Slack bot token before running, e.g.: "
        'export SLACK_BOT_TOKEN="xoxb-***"'
    )

client = WebClient(token=slack_token)


def _tracker_row(
    question: str,
    date_submitted: str,
    status_or_channel: str,
    answered: str,
    medium: str,
    submitted_by: str,
) -> list[str]:
    """Build a single engagement-tracker row."""
    return [
        question,
        "",  # Subject Area
        date_submitted,
        status_or_channel,
        answered,
        "",  # Tetra PoC
        medium,
        submitted_by,
        "",  # Associated Issues or links
        "",  # Roadmap commitments
        "",  # Requests
        "",  # Last Updated Date
        "",  # Priority
    ]


def _parse_github_date(created: str) -> str:
    """Parse an ISO timestamp into YYYY-MM-DD."""
    if not created:
        return ""
    try:
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return created[:10]


def fetch_github_items(since_date: str) -> list[list[str]]:
    """Fetch issues and PRs created since *since_date* from GITHUB_REPOS.

    Returns rows matching the engagement tracker columns.
    Uses the ``gh`` CLI (must be authenticated).
    """
    rows: list[list[str]] = []
    for repo in GITHUB_REPOS:
        for kind in ("issue", "pr"):
            label = "issues" if kind == "issue" else "PRs"
            logger.info("Fetching GitHub %s from %s since %s", label, repo, since_date)
            try:
                result = subprocess.run(
                    [
                        "gh",
                        kind,
                        "list",
                        "--repo",
                        repo,
                        "--state",
                        "all",
                        "--search",
                        f"created:>={since_date}",
                        "--limit",
                        "200",
                        "--json",
                        "title,body,createdAt,url,comments,author",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning("Failed to fetch %s from %s: %s", label, repo, e)
                continue

            items = json.loads(result.stdout) if result.stdout.strip() else []
            for item in items:
                author_info = item.get("author", {})
                author_login = author_info.get("login", "")

                # Skip bot PRs
                if kind == "pr" and author_login in DEPENDABOT_LOGINS:
                    continue

                title = item.get("title", "")
                body = item.get("body", "")
                body_oneline = " ".join(body.split())
                question = sanitize_text(
                    f"{title} — {body_oneline}".strip() if body_oneline else title
                )

                comments = item.get("comments", [])
                comment_count = len(comments) if isinstance(comments, list) else 0
                answered = "N" if comment_count == 0 else "open items"

                author_name = author_info.get("name") or author_login or "Unknown"

                rows.append(
                    _tracker_row(
                        question=question,
                        date_submitted=_parse_github_date(item.get("createdAt", "")),
                        status_or_channel=item.get("url", ""),
                        answered=answered,
                        medium="GitHub",
                        submitted_by=author_name,
                    )
                )
            logger.info("Found %d %s from %s", len(items), label, repo)
    return rows


def main() -> None:
    # Fetch all channels
    channels: list[ChannelDict] = []

    cursor: str | None = None
    while True:
        resp: SlackResponse = client.conversations_list(
            types="public_channel", cursor=cursor
        )
        data = resp.data if hasattr(resp, "data") else resp  # type: ignore[assignment]
        if not isinstance(data, Mapping):
            logger.error("Unexpected conversations_list response type")
            break

        raw_channels = data.get("channels", [])
        if isinstance(raw_channels, list):
            for ch in raw_channels:
                if isinstance(ch, Mapping):
                    channels.append(
                        ChannelDict(
                            id=str(ch.get("id", "")),
                            name=str(ch.get("name", "")),
                        )
                    )

        meta = data.get("response_metadata") if isinstance(data, Mapping) else None
        next_cursor = None
        if isinstance(meta, Mapping):
            nc = meta.get("next_cursor")
            next_cursor = str(nc) if nc else None

        if next_cursor:
            cursor = next_cursor
        else:
            break

    # Calculate timestamp for 7 days ago
    one_week_ago = datetime.now() - timedelta(days=7)
    oldest_timestamp = one_week_ago.timestamp()

    output_path = "/Users/meghan/Downloads/slack_github_this_week.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Question",
                "Subject Area",
                "Date Submitted",
                "Status or channel",
                "Answered (y/n/open items)",
                "Tetra PoC",
                "Medium Submitted",
                "Submitted by",
                "Associated Issues or links",
                "Roadmap commitments",
                "Requests",
                "Last Updated Date",
                "Priority",
            ]
        )

        # Iterate through channels
        for channel in channels:
            channel_id = channel.get("id", "")
            channel_name = channel.get("name", "")

            # Skip excluded channels by ID
            if channel_id in EXCLUDED_CHANNEL_IDS:
                logger.info("Skipping channel ID: %s (%s)", channel_id, channel_name)
                continue

            logger.info("Processing channel: %s", channel_name)

            msg_cursor: str | None = None
            while True:
                hist_resp: SlackResponse = client.conversations_history(
                    channel=channel_id,
                    cursor=msg_cursor,
                    oldest=str(oldest_timestamp),
                )
                hdata = hist_resp.data if hasattr(hist_resp, "data") else hist_resp  # type: ignore[assignment]
                if not isinstance(hdata, Mapping):
                    logger.error("Unexpected conversations_history response type")
                    break

                msgs = hdata.get("messages", [])
                if isinstance(msgs, list):
                    for m in msgs:
                        if not isinstance(m, Mapping):
                            continue
                        text = str(m.get("text", ""))
                        ts_raw = m.get("ts")
                        # ts is a string like "1731349200.000100" — guard parse
                        try:
                            ts_float = float(ts_raw) if ts_raw is not None else 0.0
                        except (TypeError, ValueError):
                            ts_float = 0.0
                        date_submitted = (
                            datetime.fromtimestamp(ts_float).strftime("%Y-%m-%d")
                            if ts_float
                            else ""
                        )
                        reply_count = m.get("reply_count", 0)
                        replies = (
                            "Y"
                            if (isinstance(reply_count, int) and reply_count > 0)
                            else "N"
                        )
                        user_id = (
                            str(m.get("user", "")) if m.get("user") is not None else ""
                        )
                        user_name, is_internal = get_user_info(client, user_id)
                        if is_internal:
                            continue
                        writer.writerow(
                            _tracker_row(
                                question=sanitize_text(text),
                                date_submitted=date_submitted,
                                status_or_channel=channel_name,
                                answered=replies,
                                medium="Slack",
                                submitted_by=user_name,
                            )
                        )

                hmeta = (
                    hdata.get("response_metadata")
                    if isinstance(hdata, Mapping)
                    else None
                )
                next_cursor = None
                if isinstance(hmeta, Mapping):
                    nc = hmeta.get("next_cursor")
                    next_cursor = str(nc) if nc else None

                if next_cursor:
                    msg_cursor = next_cursor
                else:
                    break

        # -----------------------------------------------------------------
        # GitHub issues and PRs created this week
        # -----------------------------------------------------------------
        since_str = one_week_ago.strftime("%Y-%m-%d")
        github_rows = fetch_github_items(since_str)
        for row in github_rows:
            writer.writerow(row)
        logger.info("Appended %d GitHub items to export", len(github_rows))

    logger.info("✅ Export complete: %s", output_path)


if __name__ == "__main__":
    main()
