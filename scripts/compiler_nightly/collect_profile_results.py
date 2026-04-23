#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qai_hub import Client, JobType
from utils import (
    DEFAULT_MAX_WORKERS,
    JOB_STATUS_FAILED,
    JOB_STATUS_SUCCESS,
    get_date_str,
    load_client,
    load_yaml_safe,
    log_and_print,
    save_yaml_results,
    setup_script_logging,
)

logger = logging.getLogger(__name__)


def collect_profile_result(
    client: Client, model_name: str, job_info: dict
) -> tuple[str, dict]:
    """Collect profile job status and estimated inference time."""
    profile_job_id = job_info.get("profile_job")

    # If no profile job was submitted (compile failed), return as-is
    if not profile_job_id:
        return model_name, job_info

    try:
        job = client.get_job(profile_job_id, JobType.PROFILE)
        status = job.get_status()

        result = job_info.copy()
        result["profile_status"] = status.code

        # Get estimated inference time if job succeeded
        if status.success:
            try:
                profile = job.download_profile()
                result["estimated_inference_time"] = int(
                    profile["execution_summary"]["estimated_inference_time"]
                )
            except Exception as e:
                logger.warning(
                    f"{model_name}: Failed to extract estimated inference time: {e}"
                )

        return model_name, result

    except Exception:
        logger.exception(f"{model_name}: Failed to collect profile result")
        return model_name, job_info


def collect_profile_results(
    client: Client, job_map: dict, max_workers: int = DEFAULT_MAX_WORKERS
) -> dict:
    """Collect profile job results in parallel."""
    logger.info(f"Collecting results for {len(job_map)} profile jobs")

    updated_job_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(collect_profile_result, client, model_name, job_info)
            for model_name, job_info in job_map.items()
        ]

        completed = 0
        total = len(job_map)

        for future in as_completed(futures):
            try:
                model_name, result = future.result()
                updated_job_map[model_name] = result
                completed += 1

                status = result.get("profile_status", "N/A")
                estimated_inference_time = result.get("estimated_inference_time")

                if estimated_inference_time:
                    logger.info(
                        f"  [{completed}/{total}] {model_name}: {status}, "
                        f"latency={estimated_inference_time / 1000:.2f}ms"
                    )
                else:
                    logger.info(f"  [{completed}/{total}] {model_name}: {status}")

            except Exception:
                logger.exception("  Error collecting result")

    return updated_job_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect profile job results and update config with status and estimated inference time"
    )
    parser.add_argument(
        "--dev-profile",
        type=str,
        default="dev",
        help="Hub client profile for dev environment (default: dev)",
    )
    parser.add_argument(
        "--dev-profile-config",
        type=Path,
        required=True,
        help="Path to dev-profile-jobs-<date>.yaml file from run_profile_jobs.py",
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
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS})",
    )

    args = parser.parse_args()

    date_str = get_date_str()
    log_file = setup_script_logging(
        args.output_dir, "collect-profile-results", args.verbose, date_str
    )
    log_and_print(f"Full logs: {log_file}", logger)

    try:
        dev_client = load_client(args.dev_profile)
        job_map = load_yaml_safe(args.dev_profile_config)
        log_and_print(
            f"Loaded {len(job_map)} profile jobs from {args.dev_profile_config}", logger
        )

        updated_job_map = collect_profile_results(dev_client, job_map, args.workers)

        # Count results
        succeeded = sum(
            1
            for j in updated_job_map.values()
            if j.get("profile_status") == JOB_STATUS_SUCCESS
        )
        failed = sum(
            1
            for j in updated_job_map.values()
            if j.get("profile_status") == JOB_STATUS_FAILED
        )
        skipped = sum(1 for j in updated_job_map.values() if not j.get("profile_job"))

        log_and_print(
            f"Collected {len(updated_job_map)} results: "
            f"{succeeded} succeeded, {failed} failed, {skipped} skipped",
            logger,
        )

        # Save back to the same file
        save_yaml_results(updated_job_map, args.dev_profile_config)
        log_and_print(f"Updated config saved to {args.dev_profile_config}", logger)

        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
