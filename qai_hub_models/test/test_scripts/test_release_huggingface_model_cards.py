# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.scripts.release_huggingface_model_cards import (
    HF_REPO_NAMES_TO_NEVER_DEPRECATE,
    release_hf_model_cards,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS

# Fake deprecated model name that doesn't exist in MODEL_IDS
FAKE_DEPRECATED_MODEL = "fake_deprecated_model_for_testing"
# Model from HF_REPO_NAMES_TO_NEVER_DEPRECATE that should be skipped
NEVER_DEPRECATE_MODEL = next(iter(HF_REPO_NAMES_TO_NEVER_DEPRECATE))


def test_generate_and_dry_run_release_hf_model_cards() -> None:
    """Test release_hf_model_cards generates model cards for multiple models in dry-run mode."""
    # Get all models
    model_ids = MODEL_IDS
    all_model_names = [
        QAIHMModelInfo.from_model(model_id).name for model_id in model_ids
    ]

    # Create mock HF models list that includes all real models plus a fake deprecated one
    mock_hf_models = [
        # All existing model names as if they were on Hugging Face
        # Leave out 1 model to simulate an un-published model.
        *[
            SimpleNamespace(id=f"qualcomm/{name}", tags=[])
            for name in all_model_names[:-1]
        ],
        # Fake deprecated model
        SimpleNamespace(id=f"qualcomm/{FAKE_DEPRECATED_MODEL}", tags=[]),
        # Model from HF_REPO_NAMES_TO_NEVER_DEPRECATE (should not be deprecated)
        SimpleNamespace(id=f"qualcomm/{NEVER_DEPRECATE_MODEL}", tags=[]),
    ]
    hf_list_models_patch = mock.patch(
        "qai_hub_models.scripts.release_huggingface_model_cards.HfApi.list_models",
        return_value=mock_hf_models,
    )

    with hf_list_models_patch, TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        release_hf_model_cards(
            model_list=model_ids,
            version="0.0.0",
            deprecate_removed_models=True,
            output_dir=output_dir,
            dry_run=True,
        )

        # Verify output for each model
        for model_id in model_ids:
            model_output_dir = output_dir / model_id
            assert model_output_dir.is_dir(), (
                f"Expected directory {model_output_dir} to exist"
            )
            assert (model_output_dir / "README.md").is_file(), (
                f"Expected {model_output_dir / 'README.md'} to exist"
            )
            assert (model_output_dir / "LICENSE").is_file(), (
                f"Expected {model_output_dir / 'LICENSE'} to exist"
            )

        # Verify deprecated model was processed
        deprecations_dir = output_dir / "deprecations"
        assert deprecations_dir.is_dir(), "Expected deprecations directory to exist"

        deprecated_model_dir = deprecations_dir / FAKE_DEPRECATED_MODEL
        assert deprecated_model_dir.is_dir(), (
            f"Expected deprecated model dir {deprecated_model_dir} to exist"
        )

        readme_path = deprecated_model_dir / "README.md"
        assert readme_path.is_file(), f"Expected {readme_path} to exist"

        readme_content = readme_path.read_text()
        assert "deprecated" in readme_content.lower(), (
            f"README.md for {FAKE_DEPRECATED_MODEL} should contain deprecation notice"
        )

        # Verify model from HF_REPO_NAMES_TO_NEVER_DEPRECATE was NOT deprecated
        never_deprecate_dir = deprecations_dir / NEVER_DEPRECATE_MODEL
        assert not never_deprecate_dir.exists(), (
            f"Model {NEVER_DEPRECATE_MODEL} should NOT be deprecated (in HF_REPO_NAMES_TO_NEVER_DEPRECATE)"
        )
