# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.scripts.generate_global_readme import generate_global_readme

TEST_MODELS = ["resnet50", "easyocr", "litehrnet"]


def test_generate_global_readme() -> None:
    # Verify the generated README contains the expected model table
    models = [QAIHMModelInfo.from_model(model_id) for model_id in TEST_MODELS]

    with TemporaryDirectory() as tmp_dir:
        readme_path = generate_global_readme(models, Path(tmp_dir))
        with open(readme_path) as f:
            readme = f.read()

    with open(
        os.path.join(os.path.dirname(__file__), "summary_table_expected.md")
    ) as expected_f:
        expected_table = expected_f.read()

    assert expected_table in readme
