# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Setup external repos for models.

Usage:
    python -m qai_hub_models.scripts.setup_external_repos -a
    python -m qai_hub_models.scripts.setup_external_repos -m gkt track_anything

Useful for:
    - CI setup step (clone all external repos before tests)
    - Local development (pre-populate all repos for IDE support)
"""

from __future__ import annotations

import argparse
import sys

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.utils.external_repo import (
    RepoConfig,
    setup_external_repos_impl,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup external repos for models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=str,
        help="Models for which to setup external repos.",
    )
    group.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Setup external repos for all models.",
    )
    args = parser.parse_args()
    models = args.models if args.models else MODEL_IDS

    failed = []
    for model_id in models:
        config = QAIHMModelCodeGen.from_model(model_id)
        if not config.external_repos:
            continue

        external_repos_dir = str(QAIHM_MODELS_ROOT / model_id / "external_repos")
        repo_configs = {
            name: RepoConfig(
                repo_url=cfg.repo_url,
                commit_sha=cfg.commit_sha,
                patches_filename=cfg.patches_filename,
            )
            for name, cfg in config.external_repos.items()
        }

        print(f"Setting up external repos for {model_id}...")
        try:
            setup_external_repos_impl(model_id, repo_configs, external_repos_dir)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(model_id)

    if failed:
        print(f"\nFailed to set up external repos for: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
