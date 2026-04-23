#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import csv
import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DISPLAY_SEPARATOR,
    JOB_STATUS_SUCCESS,
    get_date_str,
    load_yaml_safe,
    log_and_print,
    setup_script_logging,
    strip_device_suffix,
)

logger = logging.getLogger(__name__)

# Min perf delta (μs) for regressions/progressions (500 μs = 0.5 ms)
MIN_PERF_DELTA_US = 500


def calculate_speedups(prod_config: dict, dev_config: dict) -> dict:
    # Build a map from model name (without device) to prod info
    # Prod keys are like "model_name-device", dev keys are just "model_name"
    prod_by_model = {}
    for prod_key, prod_info in prod_config.items():
        model_name_only = strip_device_suffix(prod_key)
        prod_by_model[model_name_only] = prod_info

    speedups = {}

    for model_name, dev_info in dev_config.items():
        prod_info = prod_by_model.get(model_name, {})

        # Prod config has estimated_inference_time
        prod_latency = prod_info.get("estimated_inference_time")
        dev_latency = dev_info.get("estimated_inference_time")

        prod_success = prod_info.get("prod_job_status") == JOB_STATUS_SUCCESS
        dev_success = dev_info.get("profile_status") == JOB_STATUS_SUCCESS

        prod_job_url = prod_info.get("profile_job_url")
        if not prod_job_url:
            prod_job_url = prod_info.get("prod_job_url")

        speedups[model_name] = {
            "prod_latency": prod_latency,
            "dev_latency": dev_latency,
            "prod_success": prod_success,
            "dev_success": dev_success,
            "prod_job_url": prod_job_url,
            "dev_job_url": dev_info.get("profile_job_url"),
        }

        # Calculate speedup
        if prod_success and not dev_success:
            # Prod success, dev failure = infinite regression
            speedups[model_name]["speedup"] = float("-inf")
            speedups[model_name]["speedup_str"] = "INF"
        elif prod_latency and dev_latency:
            # Both have latencies - calculate speedup
            speedup = prod_latency / dev_latency
            speedups[model_name]["speedup"] = speedup
            speedups[model_name]["speedup_str"] = f"{speedup:.3f}"
        else:
            # Missing data
            speedups[model_name]["speedup"] = None
            speedups[model_name]["speedup_str"] = "N/A"

    return speedups


def get_progressions(speedups: dict) -> dict:
    """Get models with progressions (speedup > 1.10, i.e., >10% faster and delta > 0.5ms)."""
    progressions = {}

    for model_name, info in speedups.items():
        speedup = info["speedup"]
        prod_latency = info.get("prod_latency")
        dev_latency = info.get("dev_latency")

        # Progression if speedup > 1.10 (>10% faster) and delta > MIN_PERF_DELTA_US
        if speedup is not None and speedup > 1.10 and prod_latency and dev_latency:
            delta_us = abs(prod_latency - dev_latency)
            if delta_us > MIN_PERF_DELTA_US:
                progressions[model_name] = info

    return progressions


def _create_perf_table(
    models: dict,
    title: str,
    sort_key: Callable[[tuple[str, Any]], Any] | None = None,
    empty_message: str = "No models to display.",
    print_to_console: bool = True,
) -> None:
    """Helper to create and print a performance table."""
    if not models:
        if print_to_console:
            log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
            log_and_print(empty_message, logger)
            log_and_print(DISPLAY_SEPARATOR, logger)
        else:
            logger.info(f"\n{DISPLAY_SEPARATOR}")
            logger.info(empty_message)
            logger.info(DISPLAY_SEPARATOR)
        return

    table = PrettyTable()
    table.field_names = [
        "Model",
        "Prod (ms)",
        "Dev (ms)",
        "Speedup",
        "Prod URL",
        "Dev URL",
    ]
    table.align["Model"] = "l"
    table.align["Prod (ms)"] = "r"
    table.align["Dev (ms)"] = "r"
    table.align["Speedup"] = "r"
    table.align["Prod URL"] = "l"
    table.align["Dev URL"] = "l"

    # Sort models
    sorted_models = sorted(models.items(), key=sort_key) if sort_key else models.items()

    for model_name, info in sorted_models:
        prod_ms = (
            f"{info['prod_latency'] / 1000:.2f}" if info["prod_latency"] else "N/A"
        )
        dev_ms = f"{info['dev_latency'] / 1000:.2f}" if info["dev_latency"] else "N/A"

        prod_url = info.get("prod_job_url", "N/A")
        dev_url = info.get("dev_job_url", "N/A")

        table.add_row(
            [model_name, prod_ms, dev_ms, info["speedup_str"], prod_url, dev_url]
        )

    if print_to_console:
        log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
        log_and_print(title, logger)
        log_and_print(DISPLAY_SEPARATOR, logger)
        for line in str(table).split("\n"):
            log_and_print(line, logger)
    else:
        logger.info(f"\n{DISPLAY_SEPARATOR}")
        logger.info(title)
        logger.info(DISPLAY_SEPARATOR)
        for line in str(table).split("\n"):
            logger.info(line)


def print_progressions_table(progressions: dict) -> None:
    """Print table of progressions with URLs, sorted best to worst."""
    _create_perf_table(
        progressions,
        title=f"PROGRESSIONS: {len(progressions)} models >10% faster in dev",
        sort_key=lambda x: x[1]["speedup"],
        empty_message="No progressions found (>10% faster).",
        print_to_console=False,
    )


def calculate_geomean(speedups: dict) -> float | None:
    """Calculate geometric mean of speedups for models with both latencies."""
    valid_speedups = [
        info["speedup"]
        for info in speedups.values()
        if info["speedup"] is not None and info["speedup"] > 0
    ]

    if not valid_speedups:
        return None

    # Geomean = exp(mean(log(speedups)))
    log_sum = sum(math.log(s) for s in valid_speedups)
    return math.exp(log_sum / len(valid_speedups))


def get_regressions(speedups: dict) -> dict:
    """Get models with regressions (speedup < 0.90, i.e., >10% slower and delta > 0.5ms)."""
    regressions = {}

    for model_name, info in speedups.items():
        speedup = info["speedup"]
        prod_latency = info.get("prod_latency")
        dev_latency = info.get("dev_latency")

        # Regression if speedup < 0.90 (>10% slower) or -inf for prod success + dev failure
        if speedup is not None and speedup < 0.90:
            # For -inf (prod success, dev failure), always include
            if speedup == float("-inf"):
                regressions[model_name] = info
            elif prod_latency and dev_latency:
                delta_us = abs(dev_latency - prod_latency)
                if delta_us > MIN_PERF_DELTA_US:
                    regressions[model_name] = info

    return regressions


def print_regressions_table(regressions: dict) -> None:
    """Print table of regressions with URLs, sorted worst to best."""
    _create_perf_table(
        regressions,
        title=f"REGRESSIONS: {len(regressions)} models >10% slower in dev",
        sort_key=lambda x: (
            x[1]["speedup"] if x[1]["speedup"] != float("-inf") else -1000
        ),
        empty_message="No regressions found! 🎉",
        print_to_console=True,
    )


def save_full_table_csv(speedups: dict, output_dir: Path, date_str: str) -> Path:
    """Save full performance table to CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"performance-results-{date_str}.csv"

    # Sort by speedup descending (best first), with proper handling of None and -inf
    sorted_speedups = sorted(
        speedups.items(),
        key=lambda x: (
            x[1]["speedup"]
            if x[1]["speedup"] is not None and x[1]["speedup"] != float("-inf")
            else 1000
            if x[1]["speedup"] is None
            else -1000
        ),
        reverse=True,
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Prod Latency (ms)",
                "Dev Latency (ms)",
                "Speedup",
                "Prod Job URL",
                "Dev Job URL",
            ]
        )

        for model_name, info in sorted_speedups:
            prod_ms = (
                f"{info['prod_latency'] / 1000:.2f}" if info["prod_latency"] else ""
            )
            dev_ms = f"{info['dev_latency'] / 1000:.2f}" if info["dev_latency"] else ""

            writer.writerow(
                [
                    model_name,
                    prod_ms,
                    dev_ms,
                    info["speedup_str"],
                    info.get("prod_job_url", ""),
                    info.get("dev_job_url", ""),
                ]
            )

    log_and_print(f"Saved full performance table to: {csv_path}", logger)
    return csv_path


def print_summary(speedups: dict, geomean: float | None) -> None:
    """Print summary statistics."""
    total = len(speedups)
    with_data = sum(1 for s in speedups.values() if s["speedup"] is not None)
    regressions_count = sum(
        1 for s in speedups.values() if s["speedup"] is not None and s["speedup"] < 0.90
    )
    improvements_count = sum(
        1 for s in speedups.values() if s["speedup"] is not None and s["speedup"] > 1.10
    )
    neutral_count = with_data - regressions_count - improvements_count

    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]
    summary_table.align["Metric"] = "l"
    summary_table.align["Value"] = "r"

    summary_table.add_row(["Total Models", total])
    summary_table.add_row(["With Latency Data", with_data])
    summary_table.add_row(["Improvements (>10%)", improvements_count])
    summary_table.add_row(["Neutral (±10%)", neutral_count])
    summary_table.add_row(["Regressions (>10% slower)", regressions_count])

    if geomean:
        summary_table.add_row(["Geomean Speedup", f"{geomean:.3f}x"])
        improvement_pct = (geomean - 1.0) * 100
        summary_table.add_row(["Overall Improvement", f"{improvement_pct:+.1f}%"])

    log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
    log_and_print("Performance Summary", logger)
    log_and_print(DISPLAY_SEPARATOR, logger)
    for line in str(summary_table).split("\n"):
        log_and_print(line, logger)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze performance results and calculate speedups"
    )
    parser.add_argument(
        "--dev-profile-config",
        type=Path,
        required=True,
        help="Path to dev-profile-jobs-<date>.yaml with collected results",
    )
    parser.add_argument(
        "--prod-profile-config",
        type=Path,
        required=True,
        help="Path to AIHM profile-scorecard.yaml from prod",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for logs (default: results)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag used for output file identifier (default: current date)",
    )

    args = parser.parse_args()

    job_yaml_tag = args.tag or get_date_str()
    log_file = setup_script_logging(
        args.output_dir, "post-perf-results", args.verbose, job_yaml_tag
    )
    log_and_print(f"Full logs: {log_file}", logger)

    try:
        prod_config = load_yaml_safe(args.prod_profile_config)
        dev_config = load_yaml_safe(args.dev_profile_config)

        log_and_print(f"Loaded {len(prod_config)} prod profile jobs", logger)
        log_and_print(f"Loaded {len(dev_config)} dev profile jobs", logger)

        speedups = calculate_speedups(prod_config, dev_config)
        geomean = calculate_geomean(speedups)
        progressions = get_progressions(speedups)
        regressions = get_regressions(speedups)

        print_summary(speedups, geomean)
        print_progressions_table(progressions)
        print_regressions_table(regressions)
        save_full_table_csv(speedups, args.output_dir, job_yaml_tag)

        if regressions:
            log_and_print(
                f"✗ Found {len(regressions)} performance regressions.", logger
            )
            return 1

        log_and_print("✓ No performance regressions found.", logger)
        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
