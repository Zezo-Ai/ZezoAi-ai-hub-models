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
    extract_tag_and_dir_from_yaml,
    get_aihw_compiler_nightly_project,
    load_client,
    load_yaml_safe,
    log_and_print,
    merge_job_options,
    save_yaml_results,
    setup_script_logging,
    strip_device_suffix,
)

logger = logging.getLogger(__name__)


def _skip_result(
    compile_job_id: str | None, compile_job_url: str | None, compile_status: str
) -> dict:
    return {
        "compile_job": compile_job_id,
        "compile_job_url": compile_job_url,
        "compile_status": compile_status,
        "link_job": None,
        "link_job_url": None,
    }


def submit_single_link_job(
    client: Client,
    model_name: str,
    compile_job_info: dict,
    project_id: str,
    link_by_model: dict,
    extra_link_options: str | None,
) -> tuple[str, dict | None, Exception | None]:
    compile_job_id = compile_job_info.get("dev_job")
    if not compile_job_id:
        logger.warning(f"{model_name}: No dev_job found, skipping")
        return model_name, _skip_result(None, None, "NO_DEV_JOB"), None

    try:
        # Get the compile job
        compile_job = client.get_job(compile_job_id, JobType.COMPILE)
        compile_url = compile_job_info.get("dev_job_url", compile_job.url)

        # Wait for compile job to finish
        compile_status = compile_job.wait()

        # Check if compile job succeeded
        if compile_status.code != JOB_STATUS_SUCCESS:
            logger.info(
                f"{model_name}: Compile job {compile_job_id} status is {compile_status.code}, skipping link"
            )
            return (
                model_name,
                _skip_result(compile_job_id, compile_url, compile_status.code),
                None,
            )

        # Get target model (if job is running)
        target_model = compile_job.get_target_model()
        if not target_model:
            logger.warning(
                f"{model_name}: No target model found for compile job {compile_job_id}"
            )
            return (
                model_name,
                _skip_result(compile_job_id, compile_url, compile_status.code),
                None,
            )

        # Get device and options from link config
        if link_by_model:
            if model_name not in link_by_model:
                logger.debug(f"{model_name}: Not in link config, skipping")
                return model_name, None, None  # Skip entirely
            link_spec = link_by_model[model_name]
            device = Device(link_spec["device"])
            base_options = link_spec.get("options", "")
        else:
            # No link config - use device from compile job
            device = compile_job.device
            base_options = ""

        link_options = merge_job_options(base_options, extra_link_options) or ""

        # Submit link job
        link_job = client.submit_link_job(
            models=target_model,
            device=device,
            name=f"{model_name}_link",
            options=link_options,
            project=project_id,
        )

        logger.info(f"Submitted {model_name}: {link_job.job_id}")
        result = {
            "compile_job": compile_job_id,
            "compile_job_url": compile_job_info.get("dev_job_url", compile_job.url),
            "compile_status": compile_status.code,
            "link_job": link_job.job_id,
            "link_job_url": link_job.url,
        }
        return model_name, result, None

    except Exception as e:
        logger.exception(f"Failed to submit link job for {model_name}")
        return model_name, None, e


def submit_link_jobs(
    client: Client,
    compile_jobs: dict,
    project_id: str,
    link_config: dict | None = None,
    extra_link_options: str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> tuple[dict[str, dict], dict[str, dict]]:
    job_map = {}
    failed_jobs = {}
    logger.info(
        f"Processing {len(compile_jobs)} compile jobs for linking with {max_workers} workers"
    )

    # Build a map from model_name (without device) to link spec
    # Link config keys are "model_name_QNN_CONTEXT_BINARY-device", compile_jobs keys are "model_name_QNN_DLC"
    link_by_model = {}
    if link_config:
        for key, spec in link_config.items():
            model_name_only = strip_device_suffix(key)
            # Replace QNN_CONTEXT_BINARY with QNN_DLC to match compile job keys
            model_name_only = model_name_only.replace("_QNN_CONTEXT_BINARY", "_QNN_DLC")
            link_by_model[model_name_only] = spec

    if extra_link_options:
        logger.info(f"Extra link options: {extra_link_options}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                submit_single_link_job,
                client,
                model_name,
                compile_job_info,
                project_id,
                link_by_model,
                extra_link_options,
            ): model_name
            for model_name, compile_job_info in compile_jobs.items()
        }

        for future in as_completed(futures):
            model_name, result, error = future.result()
            if error is None:
                if result is not None:
                    job_map[model_name] = result
                # else: skipped entirely (not in link config)
            else:
                failed_jobs[model_name] = {
                    **compile_jobs[model_name],
                    "error": str(error),
                }

    # Count submitted, skipped_failed, skipped_no_target
    submitted = skipped_failed = skipped_no_target = 0
    for v in job_map.values():
        if v.get("link_job") is not None:
            submitted += 1
        else:
            compile_status = v.get("compile_status")
            if compile_status == "NO_DEV_JOB":
                skipped_no_target += 1
            elif compile_status != JOB_STATUS_SUCCESS:
                skipped_failed += 1
            else:
                skipped_no_target += 1

    log_and_print(
        f"Link job submission complete: {submitted} submitted, "
        f"{skipped_failed} skipped (compile failed), "
        f"{skipped_no_target} skipped (no target model)",
        logger,
    )
    return job_map, failed_jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit link jobs for compiled models")
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
        "--prod-link-config",
        type=Path,
        default=None,
        help="Path to prod link-scorecard.yaml (optional). If provided, device and options are taken from this file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--extra-link-options",
        type=str,
        default=None,
        help='Extra link options to append (e.g., "--some_option value")',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers for job submission (default: {DEFAULT_MAX_WORKERS})",
    )

    args = parser.parse_args()

    tag, output_dir = extract_tag_and_dir_from_yaml(args.dev_compile_config)
    log_file = setup_script_logging(output_dir, "run-link-jobs", args.verbose, tag)
    log_and_print(f"Full logs: {log_file}", logger)

    try:
        project_id = get_aihw_compiler_nightly_project()
        dev_client = load_client(args.dev_profile)
        compile_jobs = load_yaml_safe(args.dev_compile_config)
        log_and_print(
            f"Loaded {len(compile_jobs)} compile jobs from {args.dev_compile_config}",
            logger,
        )

        link_config = None
        if args.prod_link_config:
            link_config = load_yaml_safe(args.prod_link_config)
            log_and_print(
                f"Loaded prod link config with {len(link_config)} entries from {args.prod_link_config}",
                logger,
            )

        job_map, failed_jobs = submit_link_jobs(
            dev_client,
            compile_jobs,
            project_id,
            link_config,
            args.extra_link_options,
            args.workers,
        )

        dev_link_jobs_file = output_dir / f"dev-link-jobs__{tag}.yaml"
        save_yaml_results(job_map, dev_link_jobs_file)
        log_and_print(f"Saved to {dev_link_jobs_file}", logger)

        if failed_jobs:
            failed_jobs_file = (
                output_dir / f"dev-link-jobs-submission-failures__{tag}.yaml"
            )
            save_yaml_results(failed_jobs, failed_jobs_file)
            log_and_print(
                f"Failed to create {len(failed_jobs)} link job(s). Saved to {failed_jobs_file}",
                logger,
            )

        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
