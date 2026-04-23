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
from qai_hub import Client, Device, JobType
from utils import (
    DEFAULT_MAX_WORKERS,
    JOB_STATUS_SUCCESS,
    get_aihw_compiler_nightly_project,
    get_date_str,
    load_client,
    load_yaml_safe,
    log_and_print,
    merge_job_options,
    save_yaml_results,
    setup_script_logging,
    strip_device_suffix,
)

logger = logging.getLogger(__name__)


def submit_single_profile_job(
    client: Client,
    model_name: str,
    compile_job_info: dict,
    project_id: str,
    profile_by_model: dict,
    extra_profile_options: str | None,
) -> tuple[str, dict | None, Exception | None]:
    """Submit a single profile job.

    Returns
    -------
    tuple[str, dict | None, Exception | None]
        (model_name, job_result_dict, error)
        If successful or skipped, job_result_dict is populated and error is None.
        If failed, job_result_dict is None and error contains the exception.
    """
    compile_job_id = compile_job_info.get("dev_job")
    if not compile_job_id:
        logger.warning(f"{model_name}: No dev_job found, skipping")
        return (
            model_name,
            {
                "compile_job": None,
                "compile_job_url": None,
                "compile_status": "NO_DEV_JOB",
                "profile_job": None,
                "profile_job_url": None,
            },
            None,
        )

    try:
        # Get the compile job
        compile_job = client.get_job(compile_job_id, JobType.COMPILE)
        compile_status = compile_job.get_status()

        # Check if compile job succeeded
        if compile_status.code != JOB_STATUS_SUCCESS:
            logger.info(
                f"{model_name}: Compile job {compile_job_id} status is {compile_status.code}, skipping profile"
            )
            return (
                model_name,
                {
                    "compile_job": compile_job_id,
                    "compile_job_url": compile_job_info.get(
                        "dev_job_url", compile_job.url
                    ),
                    "compile_status": compile_status.code,
                    "profile_job": None,
                    "profile_job_url": None,
                },
                None,
            )

        # Get target model
        target_model = compile_job.get_target_model()
        if not target_model:
            logger.warning(
                f"{model_name}: No target model found for compile job {compile_job_id}"
            )
            return (
                model_name,
                {
                    "compile_job": compile_job_id,
                    "compile_job_url": compile_job_info.get(
                        "dev_job_url", compile_job.url
                    ),
                    "compile_status": compile_status.code,
                    "profile_job": None,
                    "profile_job_url": None,
                },
                None,
            )

        # Get device and options from profile config
        # If profile_config is provided, only profile models in the config
        if profile_by_model:
            if model_name not in profile_by_model:
                logger.debug(f"{model_name}: Not in profile config, skipping")
                return model_name, None, None  # Skip entirely
            profile_spec = profile_by_model[model_name]
            device = Device(profile_spec["device"])
            base_options = profile_spec.get("options", "")
        else:
            # No profile config - use device from compile job
            device = compile_job.device
            base_options = ""

        profile_options = merge_job_options(base_options, extra_profile_options)

        # Submit profile job
        profile_job = client.submit_profile_job(
            model=target_model,
            device=device,
            name=f"{model_name}_profile",
            options=profile_options,
            project=project_id,
        )

        logger.info(f"Submitted {model_name}: {profile_job.job_id}")
        result = {
            "compile_job": compile_job_id,
            "compile_job_url": compile_job_info.get("dev_job_url", compile_job.url),
            "compile_status": compile_status.code,
            "profile_job": profile_job.job_id,
            "profile_job_url": profile_job.url,
        }
        return model_name, result, None

    except Exception as e:
        logger.exception(f"Failed to submit profile job for {model_name}")
        return model_name, None, e


def submit_profile_jobs(
    client: Client,
    compile_jobs: dict,
    project_id: str,
    profile_config: dict | None = None,
    extra_profile_options: str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Submit profile jobs in parallel.

    Returns
    -------
    tuple[dict[str, dict], dict[str, dict]]
        (job_map, failed_jobs)
        job_map: successful job submissions and skipped jobs
        failed_jobs: dict mapping model_name to compile_job_info + error info
    """
    job_map = {}
    failed_jobs = {}
    logger.info(
        f"Processing {len(compile_jobs)} compile jobs for profiling with {max_workers} workers"
    )

    # Build a map from model_name (without device) to profile spec
    # Profile config keys are "model_name-device", compile_jobs keys are just "model_name"
    profile_by_model = {}
    if profile_config:
        for key, spec in profile_config.items():
            model_name_only = strip_device_suffix(key)
            profile_by_model[model_name_only] = spec

    if extra_profile_options:
        logger.info(f"Extra profile options: {extra_profile_options}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                submit_single_profile_job,
                client,
                model_name,
                compile_job_info,
                project_id,
                profile_by_model,
                extra_profile_options,
            ): model_name
            for model_name, compile_job_info in compile_jobs.items()
        }

        for future in as_completed(futures):
            model_name, result, error = future.result()
            if error is None:
                if result is not None:
                    job_map[model_name] = result
                # else: skipped entirely (not in profile config)
            else:
                failed_jobs[model_name] = {
                    **compile_jobs[model_name],
                    "error": str(error),
                }

    # Count submitted, skipped_failed, skipped_error
    submitted = skipped_failed = skipped_error = 0
    for v in job_map.values():
        if v.get("profile_job") is not None:
            submitted += 1
        else:
            compile_status = v.get("compile_status")
            if compile_status not in [JOB_STATUS_SUCCESS, "NO_DEV_JOB"]:
                skipped_failed += 1
            else:
                skipped_error += 1

    log_and_print(
        f"Profile job submission complete: {submitted} submitted, "
        f"{skipped_failed} skipped (compile failed), "
        f"{skipped_error} skipped (errors)",
        logger,
    )
    return job_map, failed_jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit profile jobs for compiled models"
    )
    parser.add_argument(
        "--dev-profile",
        type=str,
        default="dev",
        help="Hub client profile for dev environment (default: dev)",
    )
    parser.add_argument(
        "--dev-compile-config",
        type=Path,
        required=True,
        help="Path to dev-compile-jobs-<date>.yaml file from run_compile_jobs.py",
    )
    parser.add_argument(
        "--prod-profile-config",
        type=Path,
        default=None,
        help="Path to prod profile-scorecard.yaml (optional). If provided, device and options are taken from this file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--extra-profile-options",
        type=str,
        default=None,
        help='Extra profile options to append (e.g., "--compute_unit npu")',
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag used for output file identifier (default: current date)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers for job submission (default: {DEFAULT_MAX_WORKERS})",
    )

    args = parser.parse_args()

    job_yaml_tag = args.tag or get_date_str()
    log_file = setup_script_logging(
        args.output_dir, "run-profile-jobs", args.verbose, job_yaml_tag
    )
    log_and_print(f"Full logs: {log_file}", logger)

    try:
        project_id = get_aihw_compiler_nightly_project()
        dev_client = load_client(args.dev_profile)
        compile_jobs = load_yaml_safe(args.dev_compile_config)
        log_and_print(
            f"Loaded {len(compile_jobs)} compile jobs from {args.dev_compile_config}",
            logger,
        )

        profile_config = None
        if args.prod_profile_config:
            profile_config = load_yaml_safe(args.prod_profile_config)
            log_and_print(
                f"Loaded prod profile config with {len(profile_config)} entries from {args.prod_profile_config}",
                logger,
            )
            # Profile config keys have format "model_name-device"
            # Extract model names (strip device suffix) to match against compile_jobs
            profile_model_names = {strip_device_suffix(key) for key in profile_config}

            models_to_profile = profile_model_names & set(compile_jobs)
            log_and_print(
                f">>> This script will create {len(models_to_profile)} profile jobs <<<",
                logger,
            )
        else:
            log_and_print(
                f">>> This script will create up to {len(compile_jobs)} profile jobs (no filter) <<<",
                logger,
            )

        job_map, failed_jobs = submit_profile_jobs(
            dev_client,
            compile_jobs,
            project_id,
            profile_config,
            args.extra_profile_options,
            args.workers,
        )

        dev_profile_jobs_file = (
            args.output_dir / f"dev-profile-jobs-{job_yaml_tag}.yaml"
        )
        save_yaml_results(job_map, dev_profile_jobs_file)
        log_and_print(f"Saved to {dev_profile_jobs_file}", logger)

        if failed_jobs:
            failed_jobs_file = (
                args.output_dir
                / f"dev-profile-jobs-submission-failures-{job_yaml_tag}.yaml"
            )
            save_yaml_results(failed_jobs, failed_jobs_file)
            log_and_print(
                f"Failed to create {len(failed_jobs)} profile job(s). Saved to {failed_jobs_file}",
                logger,
            )

        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
