# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Upload scorecard artifacts to S3 for long-term historical storage.

Each upload produces a manifest.json that acts as both an index and an atomic
commit marker. The manifest is uploaded last — its presence in S3 signals that
all artifacts for a run were uploaded successfully. The query and trend-detection
scripts use manifests to enumerate available runs and discover their artifacts.

Manifest contents:
    run_id          GitHub Actions run ID
    date            Scorecard date (YYYY-MM-DD)
    branch          Source branch (e.g. "main")
    deployment      Hub deployment (e.g. "prod")
    commit_sha      Git commit at time of scorecard
    github_run_url  Link to the GitHub Actions run
    artifacts       List of uploaded file names (e.g. ["performance-summary.txt", ...])

Usage:
    python -m qai_hub_models.scripts.upload_scorecard_history \
        --run-id 25594376004 \
        --dry-run
"""

import argparse
import datetime
import logging
import os
import sys
import tempfile

import git  # noqa: TID251

from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    BranchEnvvar,
    DateFormatEnvvar,
    DeploymentEnvvar,
)
from qai_hub_models.scorecard.history import ScorecardManifest
from qai_hub_models.scripts.download_scorecard_results import S3_PREFIX
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_qaihm_s3_or_exit,
    s3_file_exists,
    s3_multipart_upload,
)

logger = logging.getLogger(__name__)

# Maps ScorecardArtifact entries to their canonical S3 file names.
# Artifacts with timestamps in their on-disk names are normalized to fixed names.
HISTORY_ARTIFACTS: list[tuple[ScorecardArtifact, str]] = [
    (ScorecardArtifact.PERFORMANCE_SUMMARY, "performance-summary.txt"),
    (ScorecardArtifact.NUMERICS_SUMMARY, "numerics-summary.txt"),
    (ScorecardArtifact.PERF_REGRESSIONS_2X, "perf-regressions-2x.json"),
    (ScorecardArtifact.NUMERICS_REGRESSIONS, "numerics-regressions.json"),
    (ScorecardArtifact.RESULTS_CSV, "results.csv"),
    (ScorecardArtifact.SCORECARD_FAILURE_ANALYSIS, "scorecard_failure_analysis.csv"),
    # Per-run toolchain snapshot. Used by the next run of the same deployment
    # as the "previous" side of its toolchain-version diff (so dev diffs
    # against dev, prod against prod). Without this, collect_scorecard_results
    # falls back to the checked-in intermediates copy, which only refreshes
    # when a prod scorecard PR merges to main.
    (ScorecardArtifact.TOOL_VERSIONS, "tool-versions.yaml"),
]


def upload_scorecard_to_s3(
    run_id: str,
    run_name: str,
    date: datetime.date,
    branch: str = "main",
    deployment: str = "prod",
    commit_sha: str = "",
    run_url: str = "",
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Upload scorecard artifacts to S3 under scorecard-history/{run_id}-{run_name}/."""
    s3_root = f"{S3_PREFIX}/{run_id}-{run_name}"
    manifest_key = f"{s3_root}/manifest.json"

    bucket = None
    if not dry_run:
        bucket, _ = get_qaihm_s3_or_exit(QAIHM_PRIVATE_S3_BUCKET)

        # Idempotency check
        if not force and s3_file_exists(bucket, manifest_key):
            logger.info(
                f"Run {run_id} already uploaded (manifest exists at {manifest_key}). "
                "Use --force to overwrite."
            )
            return

    # Discover and upload artifacts
    uploaded: list[str] = []
    for artifact, s3_name in HISTORY_ARTIFACTS:
        if not artifact.exists():
            logger.warning(f"Artifact not found: {artifact.value} (skipping)")
            continue
        local_path = artifact.path

        s3_key = f"{s3_root}/{s3_name}"
        if dry_run:
            print(
                f"[dry-run] Would upload {local_path} -> s3://{QAIHM_PRIVATE_S3_BUCKET}/{s3_key}"
            )
        else:
            assert bucket is not None
            s3_multipart_upload(bucket, s3_key, local_path, disable_progress=True)
        uploaded.append(s3_name)

    if not uploaded:
        logger.error("No artifacts found to upload. Check --artifacts-dir path.")
        sys.exit(1)

    # Build and upload manifest (last, as atomic "commit" marker)
    manifest = ScorecardManifest(
        run_id=run_id,
        date=date.isoformat(),
        branch=branch,
        deployment=deployment,
        run_name=run_name,
        commit_sha=commit_sha,
        github_run_url=run_url,
        artifacts=uploaded,
    )

    if dry_run:
        print(
            f"[dry-run] Would upload manifest.json -> s3://{QAIHM_PRIVATE_S3_BUCKET}/{manifest_key}"
        )
        print(manifest.model_dump_json(indent=2))
    else:
        assert bucket is not None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name
        manifest.to_json(tmp_path, exclude_defaults=False, exclude_none=False)
        try:
            s3_multipart_upload(bucket, manifest_key, tmp_path, disable_progress=True)
        finally:
            os.unlink(tmp_path)
        print(
            f"Uploaded {len(uploaded)} artifacts to s3://{QAIHM_PRIVATE_S3_BUCKET}/{s3_root}/"
        )


def _git_default_sha() -> str:
    """Get the current git commit SHA, or empty string if not in a repo."""
    try:
        return git.Repo(search_parent_directories=True).head.commit.hexsha
    except (git.InvalidGitRepositoryError, ValueError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload scorecard artifacts to S3 for long-term historical storage."
    )
    ArtifactsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--run-id", type=str, required=True, help="GitHub Actions run ID"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="weekly-prod",
        help="Run name for S3 path (e.g. 'weekly-prod', 'weekly-staging')",
    )
    DateFormatEnvvar.DateEnvvar.add_arg(parser)
    BranchEnvvar.add_arg(parser)
    DeploymentEnvvar.add_arg(parser)
    parser.add_argument(
        "--commit-sha", type=str, default=_git_default_sha(), help="Git commit SHA"
    )
    parser.add_argument("--run-url", type=str, default="")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing upload"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Parse date string from envvar into datetime.date
    date = datetime.datetime.strptime(
        args.date, DateFormatEnvvar.FormatEnvvar.get()
    ).date()

    upload_scorecard_to_s3(
        run_id=args.run_id,
        run_name=args.run_name,
        date=date,
        branch=args.branch,
        deployment=args.deployment,
        commit_sha=args.commit_sha,
        run_url=args.run_url,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
