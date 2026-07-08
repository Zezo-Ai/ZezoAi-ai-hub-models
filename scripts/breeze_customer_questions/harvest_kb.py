# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Harvest customer Q&A threads from a scrape and emit KB candidates.

Reads questions.json produced by ``scrape_questions.py`` and filters to
threads worth proposing as ``customer-faq.md`` entries: a customer question
with at least N substantive internal replies. Optionally dedups against an
existing KB by ``thread_url``.

The output is a JSON file consumed by ``render_kb_candidates.py``, which
turns it into a review-friendly markdown document. A separate human-review
step approves which candidates land in ``customer-faq.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _internal_reply_count(question: dict[str, Any]) -> int:
    """Number of replies from Qualcomm internal users in a thread."""
    return sum(1 for r in question.get("thread_replies", []) if r.get("is_internal"))


def _existing_thread_urls(kb_path: Path | None) -> set[str]:
    """Extract thread URLs already cited in an existing customer-faq.md."""
    if kb_path is None or not kb_path.is_file():
        return set()
    text = kb_path.read_text()
    # Match both bare URLs and markdown links to qualcomm.slack.com / github.com.
    pattern = re.compile(
        r"(https?://(?:[^\s)]*qualcomm\.slack\.com|github\.com)/[^\s)]+)"
    )
    return set(pattern.findall(text))


def filter_candidates(
    questions: list[dict[str, Any]],
    min_internal_replies: int,
    seen_urls: set[str],
) -> list[dict[str, Any]]:
    """Filter scraped questions to KB candidates.

    A candidate must:
    - Have at least *min_internal_replies* internal replies in its thread.
    - Have a ``thread_url`` not present in *seen_urls*.
    """
    out = []
    for q in questions:
        url = q.get("thread_url", "")
        if url in seen_urls:
            continue
        if _internal_reply_count(q) < min_internal_replies:
            continue
        out.append(q)
    return out


def harvest(
    scrape_path: Path,
    output_path: Path,
    min_internal_replies: int,
    kb_path: Path | None,
) -> None:
    """Read a scrape file and write kb_candidates.json."""
    scrape = json.loads(scrape_path.read_text())
    questions = scrape.get("questions", [])
    seen = _existing_thread_urls(kb_path)

    candidates = filter_candidates(questions, min_internal_replies, seen)
    # Sort newest-first by ``submitted_at`` for stable, reviewable order.
    candidates.sort(key=lambda q: q.get("submitted_at", ""), reverse=True)

    payload = {
        "harvested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_scrape": str(scrape_path),
        "source_scraped_at": scrape.get("scraped_at"),
        "source_lookback_days": scrape.get("lookback_days"),
        "min_internal_replies": min_internal_replies,
        "kb_dedup_from": str(kb_path) if kb_path else None,
        "kb_dedup_skipped": len(questions)
        - len(candidates)
        - sum(
            1
            for q in questions
            if _internal_reply_count(q) < min_internal_replies
            and q.get("thread_url", "") not in seen
        ),
        "total_scraped": len(questions),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(
        "Harvested %d candidates from %d scraped questions (>=%d internal replies). "
        "Wrote %s",
        len(candidates),
        len(questions),
        min_internal_replies,
        output_path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scrape",
        type=Path,
        required=True,
        help="Path to questions.json produced by scrape_questions.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write kb_candidates.json.",
    )
    parser.add_argument(
        "--min-internal-replies",
        type=int,
        default=2,
        help=(
            "Quality cut. Threads with fewer internal replies are dropped. Default: 2."
        ),
    )
    parser.add_argument(
        "--kb",
        type=Path,
        default=None,
        help=(
            "Path to existing customer-faq.md. Threads whose URL is already "
            "cited are skipped (dedup)."
        ),
    )
    args = parser.parse_args()
    harvest(args.scrape, args.output, args.min_internal_replies, args.kb)


if __name__ == "__main__":
    main()
