# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Render kb_candidates.json as a review-ready markdown document.

The output has one section per candidate: the verbatim thread (so the
reviewer can read the original Q&A), an approval checkbox, and an empty
YAML scaffold for the final ``customer-faq.md`` entry.

The reviewer's job is to either fill the scaffold (Q/A pair, tags) or
delete the section. A second script (``commit_approved_kb.py``, future
work) can read the reviewed markdown and append approved entries to
``customer-faq.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Slug-friendly characters; everything else becomes a hyphen.
_SLUG_BAD = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 50) -> str:
    s = _SLUG_BAD.sub("-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "entry"


def _suggest_id(question: dict[str, Any], index: int) -> str:
    """A stable, human-readable id hint the reviewer can edit."""
    channel = question.get("channel_or_repo", "unknown")
    date = (question.get("submitted_at") or "")[:10]
    # First 4 meaningful words of the question text.
    text = (question.get("question") or "").strip()
    words = [w for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 2][:4]
    word_slug = "-".join(words).lower() if words else f"entry-{index}"
    return f"{_slug(channel)}-{date}-{_slug(word_slug, 40)}"


def _quote(text: str, max_lines: int = 8) -> str:
    """Render *text* as a blockquote, capped at *max_lines* lines."""
    lines = (text or "").splitlines() or [""]
    if len(lines) > max_lines:
        lines = [*lines[:max_lines], f"... ({len(lines) - max_lines} more lines)"]
    return "\n".join(f"> {line}" for line in lines)


def _section(candidate: dict[str, Any], index: int) -> str:
    channel = candidate.get("channel_or_repo", "unknown")
    submitted_at = candidate.get("submitted_at", "?")
    submitted_by = candidate.get("submitted_by", "?")
    thread_url = candidate.get("thread_url", "")
    question_text = candidate.get("question") or "(empty)"
    replies = candidate.get("thread_replies", [])
    internal_replies = [r for r in replies if r.get("is_internal")]
    suggested_id = _suggest_id(candidate, index)

    parts = [
        f"## Candidate {index}: [#{channel}] {submitted_at[:10]} — {submitted_by}",
        "",
        f"**Thread:** {thread_url}",
        f"**Internal replies:** {len(internal_replies)} / {len(replies)} total",
        "",
        f"**Question ({submitted_by}):**",
        "",
        _quote(question_text),
        "",
        "**Thread replies:**",
        "",
    ]
    for r in replies:
        marker = "**[internal]**" if r.get("is_internal") else "_[customer]_"
        user = r.get("user", "?")
        parts.append(f"{marker} *{user}:*")
        parts.append("")
        parts.append(_quote(r.get("text") or "(empty)"))
        parts.append("")

    parts.extend(
        [
            "---",
            "",
            "### Reviewer action",
            "",
            "- [ ] **Approve** — fill the YAML below, then this section becomes a `customer-faq.md` entry.",
            "- [ ] **Reject** — delete this entire section.",
            "",
            "```yaml",
            f"id: {suggested_id}",
            f"asked_first: {submitted_at[:10]}",
            f"last_verified: {submitted_at[:10]}",
            "topic_tags: []   # e.g. [qwen3, release-assets, snapdragon-x-elite]",
            f"source_thread: {thread_url}",
            "```",
            "",
            "**Q:** _(write the canonical question — paraphrase the customer's wording)_",
            "",
            "**A:** _(write the canonical answer — paraphrase the engineer's reply, cite docs/code)_",
            "",
            "**See also:** _(optional: links to release-assets.yaml, info_yaml.py:N, etc.)_",
            "",
        ]
    )
    return "\n".join(parts)


def render(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text())
    candidates = payload.get("candidates", [])

    header = [
        "# KB Candidates — Customer Q&A Harvest",
        "",
        f"- **Harvested at:** {payload.get('harvested_at', '?')}",
        f"- **Source scrape:** {payload.get('source_scrape', '?')}",
        f"- **Source scraped at:** {payload.get('source_scraped_at', '?')}",
        f"- **Lookback days:** {payload.get('source_lookback_days', '?')}",
        f"- **Quality cut:** ≥{payload.get('min_internal_replies', '?')} internal replies",
        f"- **Total scraped:** {payload.get('total_scraped', 0)}",
        f"- **Candidate count:** {payload.get('candidate_count', 0)}",
        "",
        "## How to review",
        "",
        "1. Read each candidate's thread below.",
        "2. Check **Approve** and fill the YAML + Q/A scaffold, OR check **Reject** and delete the section.",
        "3. When done, hand the file to `commit_approved_kb.py` (future) — it'll append approved entries to `customer-faq.md`.",
        "",
        "---",
        "",
    ]

    sections = [_section(c, i + 1) for i, c in enumerate(candidates)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(header + sections))
    logger.info("Rendered %d candidates to %s", len(candidates), output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to kb_candidates.json from harvest_kb.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the review markdown.",
    )
    args = parser.parse_args()
    render(args.input, args.output)


if __name__ == "__main__":
    main()
