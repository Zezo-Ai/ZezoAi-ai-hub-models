# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Merge LLM perf.yaml update logs from every runtime into the checkout.

Each perf job (genie, geniex) emits a JSON-lines log of its
update_perf_yaml calls (see _shared/llm/perf_collection.py). This script
replays all of them onto a fresh checkout with a single clear-then-upsert pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qai_hub_models import Precision
from qai_hub_models.models._shared.llm.perf_collection import (
    clear_llm_metrics_for_profile_path,
    update_perf_yaml,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath


def _load_updates_file(path: Path) -> list[dict]:
    """Load one JSON-lines updates file (one update dict per line)."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_updates(paths: list[Path]) -> list[dict]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        elif p.exists():
            files.append(p)
        else:
            print(f"Updates file {p} does not exist; skipping.")

    updates: list[dict] = []
    for f in files:
        entries = _load_updates_file(f)
        print(f"Loaded {len(entries)} updates from {f}")
        updates.extend(entries)
    return updates


def apply_updates(updates: list[dict]) -> int:
    if not updates:
        print("No perf updates to apply; nothing to do.")
        return 0

    # Clear each (model, profile_path, device, precision) bucket once before
    # rewriting so dropped context lengths / compute units don't leave orphans
    # behind. Scoping the clear to the (device, precision) pairs actually
    # present in the updates is what keeps this safe for partial runs: a run
    # that measured only a subset of devices/precisions (or where some jobs
    # flaked and emitted no update line) must not wipe committed metrics for
    # the buckets it didn't touch. Distinct runtimes use distinct profile_paths,
    # so clearing one runtime's buckets never touches another's -- the merge is
    # order-independent across runtimes and idempotent per bucket.
    seen: set[tuple[str, str, str, str]] = set()
    for u in updates:
        key = (u["model_id"], u["profile_path"], u["device_name"], u["precision"])
        if key in seen:
            continue
        seen.add(key)
        clear_llm_metrics_for_profile_path(
            model_id=u["model_id"],
            profile_path=ScorecardProfilePath(u["profile_path"]),
            device_name=u["device_name"],
            precision=Precision.parse(u["precision"]),
        )

    models: set[str] = set()
    for u in updates:
        update_perf_yaml(
            model_id=u["model_id"],
            device_name=u["device_name"],
            precision=Precision.parse(u["precision"]),
            context_length=u["context_length"],
            tps=u["tps"],
            ttft_ms=u["ttft_ms"],
            prefill_tps=u["prefill_tps"],
            ttft_max_ms=u["ttft_max_ms"],
            profile_path=ScorecardProfilePath(u["profile_path"]),
            desired_compute_unit=u["desired_compute_unit"],
        )
        models.add(u["model_id"])

    print(f"Applied {len(updates)} perf updates across {len(models)} models:")
    for model_id in sorted(models):
        print(f"  {model_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="+",
        help="Perf updates files and/or directories to scan for *.jsonl.",
    )
    args = ap.parse_args()
    return apply_updates(collect_updates([Path(p) for p in args.paths]))


if __name__ == "__main__":
    raise SystemExit(main())
