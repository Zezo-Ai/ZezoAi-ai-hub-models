#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qai_hub import Client, Device
from run_compile_jobs import format_input_specs
from utils import (
    get_aihw_compiler_nightly_project,
    load_client,
    load_yaml_safe,
    log_and_print,
    merge_job_options,
    save_yaml_results,
    setup_script_logging,
)

logger = logging.getLogger(__name__)

PROD_MODEL_SEPARATOR = "__prod__"


def format_dev_model_name(model_name: str, prod_model_id: str) -> str:
    return f"{model_name}{PROD_MODEL_SEPARATOR}{prod_model_id}"


def reupload_single_model(
    prod_client: Client,
    dev_client: Client,
    model_name: str,
    job_spec: dict,
    project_id: str,
) -> dict:
    """Upload a single model from prod to dev (always fresh upload).

    Returns
    -------
    dict
        Updated job_spec with new model_id
    """
    prod_job = prod_client.get_job(job_spec["prod_job"])
    prod_model_id = prod_job.model.model_id
    log_and_print(f"Uploading {model_name} (prod model: {prod_model_id})", logger)

    # Download from prod
    prod_model = prod_client.get_model(prod_model_id)
    model_name_with_prod_id = format_dev_model_name(model_name, prod_model_id)

    with tempfile.TemporaryDirectory(prefix=f"model_{model_name}_") as temp_dir:
        model_path = Path(temp_dir) / f"{model_name}_{prod_model_id}"
        prod_model.download(str(model_path))

        downloaded_files = list(Path(temp_dir).glob(f"{model_name}_{prod_model_id}.*"))
        if not downloaded_files:
            raise RuntimeError(f"No file found after download in {temp_dir}")

        # Always upload fresh to dev
        dev_model = dev_client.upload_model(
            str(downloaded_files[0]),
            name=model_name_with_prod_id,
            project=project_id,
        )

        # Update job spec with new model_id
        updated_spec = job_spec.copy()
        updated_spec["model_id"] = dev_model.model_id
        # Remove error field
        updated_spec.pop("error", None)

        logger.info(f"Success: {model_name} -> new model_id {dev_model.model_id}")
        return updated_spec


def reupload_failed_models(
    prod_client: Client,
    dev_client: Client,
    failed_jobs: dict,
    project_id: str,
) -> dict[str, dict]:
    """Upload failed models serially (always fresh upload). Fails on first error.

    Returns
    -------
    dict[str, dict]
        Uploaded jobs with new model_ids
    """
    successful_uploads = {}
    total = len(failed_jobs)

    log_and_print(f"Uploading {total} models", logger)
    for _idx, (model_name, job_spec) in enumerate(failed_jobs.items(), start=1):
        updated_spec = reupload_single_model(
            prod_client,
            dev_client,
            model_name,
            job_spec,
            project_id,
        )
        successful_uploads[model_name] = updated_spec

    log_and_print(
        f"Successfully uploaded {len(successful_uploads)}/{total} models", logger
    )
    return successful_uploads


def submit_single_compile_job(
    client: Client,
    model_name: str,
    job_spec: dict,
    project_id: str,
    extra_compiler_args: str | None,
) -> dict:
    """Submit a single compile job.

    Returns
    -------
    dict
        Job result dict with dev_job and dev_job_url
    """
    logger.info(f"Submitting compile job for {model_name}")
    compile_options = merge_job_options(
        job_spec["compile_options"], extra_compiler_args
    )
    formatted_input_specs = format_input_specs(job_spec["input_specs"])

    model = client.get_model(job_spec["model_id"])
    device = Device(job_spec["device"])

    job = client.submit_compile_job(
        model=model,
        name=model_name,
        device=device,
        input_specs=formatted_input_specs,
        options=compile_options,
        project=project_id,
    )
    return {
        "prod_job": job_spec["prod_job"],
        "prod_job_url": job_spec.get("prod_job_url", job_spec["prod_job"]),
        "prod_job_status": job_spec["prod_job_status"],
        "dev_job": job.job_id,
        "dev_job_url": job.url,
    }


def submit_compile_jobs(
    client: Client,
    job_specs: dict,
    project_id: str,
    extra_compiler_args: str | None,
) -> dict[str, dict]:
    """Submit compile jobs serially. Fails on first error.

    Returns
    -------
    dict[str, dict]
        Submitted jobs map
    """
    job_map = {}
    total = len(job_specs)

    logger.info(f"Submitting {total} compile jobs serially")

    for idx, (model_name, job_spec) in enumerate(job_specs.items(), start=1):
        result = submit_single_compile_job(
            client,
            model_name,
            job_spec,
            project_id,
            extra_compiler_args,
        )
        job_map[model_name] = result
        print(f"  [{idx}/{total}] {model_name}: submitted")

    log_and_print(f"Successfully submitted {len(job_map)}/{total} compile jobs", logger)
    return job_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload failed models (fresh) and submit compile jobs"
    )
    parser.add_argument(
        "--prod-profile",
        type=str,
        default="prod",
        help="Hub client profile for prod environment (default: prod)",
    )
    parser.add_argument(
        "--dev-profile",
        type=str,
        default="dev",
        help="Hub client profile for dev environment (default: dev)",
    )
    parser.add_argument(
        "--failed-jobs",
        type=Path,
        required=True,
        help="Path to dev-compile-jobs-submission-failures-*.yaml file",
    )
    parser.add_argument(
        "--dev-jobs-file",
        type=Path,
        required=True,
        help="Path to existing dev-compile-jobs-*.yaml file to update",
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
        "--extra-compiler-args",
        type=str,
        default=None,
        help='Additional compiler arguments as CLI string (e.g., "--flag1 value1 --flag2")',
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag for output files (default: extracted from input filename)",
    )

    args = parser.parse_args()

    # Extract tag from input filename if not provided
    if args.tag:
        tag = args.tag
    else:
        # Extract from filename like "dev-compile-jobs-submission-failures-04-21.yaml"
        stem = args.failed_jobs.stem
        if stem.startswith("dev-compile-jobs-submission-failures-"):
            tag = stem.replace("dev-compile-jobs-submission-failures-", "")
        else:
            tag = "reuploaded"

    log_file = setup_script_logging(
        args.output_dir, "resubmit-compile-job-failures", args.verbose, tag
    )
    log_and_print(f"Full logs: {log_file}", logger)

    try:
        project_id = get_aihw_compiler_nightly_project()
        prod_client = load_client(args.prod_profile)
        dev_client = load_client(args.dev_profile)
        failed_jobs = load_yaml_safe(args.failed_jobs)

        log_and_print(
            f"Loaded {len(failed_jobs)} failed jobs from {args.failed_jobs}", logger
        )

        # Always re-upload models
        successful_uploads = reupload_failed_models(
            prod_client, dev_client, failed_jobs, project_id
        )

        job_map = submit_compile_jobs(
            dev_client,
            successful_uploads,
            project_id,
            args.extra_compiler_args,
        )

        existing_jobs = load_yaml_safe(args.dev_jobs_file)
        num_existing = len(existing_jobs)
        log_and_print(
            f"Loaded {num_existing} existing jobs from {args.dev_jobs_file}",
            logger,
        )

        # Check how many are replacements vs additions
        replacements = sum(1 for name in job_map if name in existing_jobs)
        additions = len(job_map) - replacements

        # Merge: new jobs override existing ones with same model_name
        existing_jobs.update(job_map)
        save_yaml_results(existing_jobs, args.dev_jobs_file)

        log_and_print(
            f"Updated {args.dev_jobs_file}: {replacements} replaced, {additions} added (total: {len(existing_jobs)})",
            logger,
        )

        log_and_print(
            f"Summary: {len(job_map)} jobs submitted successfully",
            logger,
        )
        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
