# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import contextlib
import datetime
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pformat

import pytest
import qai_hub as hub

from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    InferenceScorecardJobYaml,
    ProfileScorecardJobYaml,
    QuantizeScorecardJobYaml,
)
from qai_hub_models.test.utils.envvars import DisableWorkbenchJobTimeoutEnvvar
from qai_hub_models.utils.testing_async_utils import (
    get_compile_job_ids_file,
    get_inference_job_ids_file,
    get_profile_job_ids_file,
    get_quantize_job_ids_file,
)

# Maximum time (minutes after job submission) to wait for a single job to finish
_JOB_TIMEOUT_SECONDS = 75 * 60
# 90-minute hard cap per thread to prevent indefinite hangs
_JOIN_TIMEOUT_SECONDS = 90 * 60


def _not_exists_or_empty(path: str | os.PathLike) -> bool:
    return not os.path.exists(path) or os.stat(path).st_size == 0


def _check_single_job(
    name: str,
    job_id: str,
    failed_jobs: dict[str, str],
    timeout_jobs: dict[str, str],
    lock: threading.Lock,
) -> None:
    """Check a single job's status, storing results in the shared dicts."""
    try:
        job = hub.get_job(job_id)

        status = job.get_status()
        if not status.finished:
            if DisableWorkbenchJobTimeoutEnvvar.get():
                status = job.wait()
            else:
                timemax = datetime.timedelta(seconds=_JOB_TIMEOUT_SECONDS)
                timediff = datetime.datetime.now() - job.date
                if timediff < timemax:
                    with contextlib.suppress(TimeoutError):
                        status = job.wait(int((timemax - timediff).total_seconds()))

        if not status.success:
            with lock:
                if not status.finished or (
                    status.failure
                    and status.message is not None
                    and "timed out" in status.message
                ):
                    timeout_jobs[name] = job.url
                else:
                    failed_jobs[name] = job.url
    except Exception as exc:
        with lock:
            failed_jobs[name] = f"<exception: {exc}>"


def _verify_jobs_successful(job_ids: dict[str, str], job_type: str) -> None:
    """
    Verifies that the jobs with the given job ids all succeeded.
    If any jobs fail, raises a ValueError with the failed job urls.

    Uses a thread pool to check results in parallel.
    """
    failed_jobs: dict[str, str] = {}
    timeout_jobs: dict[str, str] = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=min(64, len(job_ids))) as pool:
        futures = [
            pool.submit(
                _check_single_job, name, job_id, failed_jobs, timeout_jobs, lock
            )
            for name, job_id in job_ids.items()
        ]
        for f in as_completed(futures, timeout=_JOIN_TIMEOUT_SECONDS):
            f.result()

    error_strs = []
    if failed_jobs:
        error_strs.append(
            f"The following {job_type} jobs failed:\n{pformat(failed_jobs)}"
        )
    if timeout_jobs:
        error_strs.append(
            f"The following {job_type} jobs timed out:\n{pformat(timeout_jobs)}"
        )
    if len(error_strs) > 0:
        raise ValueError("\n".join(error_strs))


"""
When testing compilation in CI, synchronously waiting for each job to
finish is too slow. Instead, job ids are written to a file upon submission,
and success is validated all at once in the end using these tests.
"""


@pytest.mark.skipif(
    _not_exists_or_empty(get_quantize_job_ids_file()),
    reason="No quantize jobs file found",
)
def test_quantize_jobs_success() -> None:
    job_ids = QuantizeScorecardJobYaml.from_file(
        get_quantize_job_ids_file()
    ).job_id_mapping
    _verify_jobs_successful(job_ids, "quantize")


@pytest.mark.skipif(
    _not_exists_or_empty(get_compile_job_ids_file()),
    reason="No compile jobs file found",
)
def test_compile_jobs_success() -> None:
    job_ids = CompileScorecardJobYaml.from_file(
        get_compile_job_ids_file()
    ).job_id_mapping
    _verify_jobs_successful(job_ids, "compile")


@pytest.mark.skipif(
    _not_exists_or_empty(get_profile_job_ids_file()), reason="No profile jobs found"
)
def test_profile_jobs_success() -> None:
    job_ids = ProfileScorecardJobYaml.from_file(
        get_profile_job_ids_file()
    ).job_id_mapping
    _verify_jobs_successful(job_ids, "profile")


@pytest.mark.skipif(
    _not_exists_or_empty(get_inference_job_ids_file()), reason="No inference jobs found"
)
def test_inference_jobs_success() -> None:
    job_ids = InferenceScorecardJobYaml.from_file(
        get_inference_job_ids_file()
    ).job_id_mapping
    _verify_jobs_successful(job_ids, "inference")
