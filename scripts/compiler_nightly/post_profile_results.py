#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import math
import sys
from pathlib import Path

from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DISPLAY_SEPARATOR,
    JOB_STATUS_SUCCESS,
    extract_tag_and_dir_from_yaml,
    load_yaml_safe,
    log_and_print,
    map_prod_by_model,
    print_results_table,
    save_results_csv,
    setup_script_logging,
)

logger = logging.getLogger(__name__)

# Min perf delta (μs) for regressions/progressions (500 μs = 0.5 ms)
MIN_PERF_DELTA_US = 500


def calculate_speedups(prod_config: dict, dev_config: dict) -> dict:
    prod_by_model = map_prod_by_model(prod_config)

    speedups = {}

    for model_name, dev_info in dev_config.items():
        prod_info = prod_by_model.get(model_name, {})

        # Prod config has estimated_inference_time
        prod_latency = prod_info.get("estimated_inference_time")
        dev_latency = dev_info.get("estimated_inference_time")

        # Support both prod scorecard format and dev-profile-jobs format for baseline
        prod_success = (
            prod_info.get("prod_job_status") == JOB_STATUS_SUCCESS
            or prod_info.get("profile_status") == JOB_STATUS_SUCCESS
        )
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


def _perf_row(model_name: str, info: dict, empty_value: str = "N/A") -> list:
    prod_ms = (
        f"{info['prod_latency'] / 1000:.2f}" if info["prod_latency"] else empty_value
    )
    dev_ms = f"{info['dev_latency'] / 1000:.2f}" if info["dev_latency"] else empty_value
    return [
        model_name,
        prod_ms,
        dev_ms,
        info["speedup_str"],
        info.get("prod_job_url", empty_value),
        info.get("dev_job_url", empty_value),
    ]


def print_progressions_table(progressions: dict) -> None:
    field_names = ["Model", "Prod (ms)", "Dev (ms)", "Speedup", "Prod URL", "Dev URL"]
    print_results_table(
        progressions,
        title=f"PROGRESSIONS: {len(progressions)} models >10% faster in dev",
        field_names=field_names,
        row_extractor=_perf_row,
        sort_key=lambda x: -x[1]["speedup"],
        empty_message="No progressions found (>10% faster).",
        print_to_console=True,
    )


def calculate_geomean(speedups: dict) -> float | None:
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
    field_names = ["Model", "Prod (ms)", "Dev (ms)", "Speedup", "Prod URL", "Dev URL"]
    print_results_table(
        regressions,
        title=f"REGRESSIONS: {len(regressions)} models >10% slower in dev",
        field_names=field_names,
        row_extractor=_perf_row,
        sort_key=lambda x: (
            x[1]["speedup"] if x[1]["speedup"] != float("-inf") else -1000
        ),
        empty_message="No regressions found! 🎉",
        print_to_console=True,
    )


def save_full_table_csv(speedups: dict, output_dir: Path, tag: str) -> Path:
    field_names = [
        "Model",
        "Prod Latency (ms)",
        "Dev Latency (ms)",
        "Speedup",
        "Prod Job URL",
        "Dev Job URL",
    ]
    csv_path = output_dir / f"performance-results__{tag}.csv"

    def sort_key(x: tuple) -> float:
        speedup = x[1]["speedup"]
        if speedup is not None and speedup != float("-inf"):
            return speedup
        if speedup is None:
            return 1000
        return -1000

    return save_results_csv(
        speedups,
        csv_path,
        field_names,
        lambda name, info: _perf_row(name, info, empty_value=""),
        sort_key=sort_key,
    )


def print_summary(
    speedups: dict, geomean: float | None, regressions: dict, progressions: dict
) -> None:
    total = len(speedups)
    with_data = sum(1 for s in speedups.values() if s["speedup"] is not None)
    regressions_count = len(regressions)
    improvements_count = len(progressions)
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
        help="Path to dev-profile-jobs__<tag>.yaml with collected results",
    )
    parser.add_argument(
        "--prod-profile-config",
        type=Path,
        required=True,
        help="Path to AIHM profile-scorecard.yaml from prod",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    tag, output_dir = extract_tag_and_dir_from_yaml(args.dev_profile_config)
    log_file = setup_script_logging(output_dir, "post-perf-results", args.verbose, tag)
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

        print_summary(speedups, geomean, regressions, progressions)
        print_progressions_table(progressions)
        print_regressions_table(regressions)
        save_full_table_csv(speedups, output_dir, tag)

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
