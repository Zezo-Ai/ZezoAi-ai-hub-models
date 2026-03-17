# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    CachedWebAsset,
    ModelZooAssetConfig,
    download_from_private_s3,
)
from qai_hub_models.utils.aws import QAIHM_PRIVATE_S3_BUCKET
from qai_hub_models.utils.envvars import IsOnCIEnvvar


class CachedPrivateCIAsset(CachedWebAsset):
    """
    Cached asset that is only available via the private S3 bucket on CI.

    On CI (`QAIHM_CI=1`), the asset is downloaded from the private S3 bucket.
    Outside CI, :meth:`fetch` raises an error because the asset has no public URL.
    """

    def __init__(
        self,
        private_s3_key: str,
        local_cache_path: Path,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
        non_ci_error: Exception | None = None,
        local_cache_extracted_path: str | Path | None = None,
    ) -> None:
        """
        Parameters
        ----------
        private_s3_key
            Key inside the private S3 bucket
            (e.g. `"qai-hub-models/datasets/foo/bar.zip"`).
        local_cache_path
            Path to store the downloaded asset on disk.
        asset_config
            Asset config to use to save this file.
        non_ci_error
            Custom error to raise when fetching outside CI.
            If `None`, a default `ValueError` is used.
        local_cache_extracted_path
            Path where extracted archive contents will live.
            Defaults to ``local_cache_path`` without its extension.
        """
        self.non_ci_error = non_ci_error
        url = f"s3://{QAIHM_PRIVATE_S3_BUCKET}/{private_s3_key}"
        super().__init__(
            url,
            local_cache_path,
            asset_config,
            download_from_private_s3,
            local_cache_extracted_path=local_cache_extracted_path,
        )

    def fetch(
        self,
        extract: bool = False,
        local_path: str | Path | None = None,
    ) -> Path:
        if local_path is not None:
            return Path(super().fetch(extract=extract, local_path=local_path))
        if not IsOnCIEnvvar.get():
            raise self.non_ci_error or ValueError(
                f"CachedPrivateCIAsset ({self.url}) can only be fetched on CI "
                "(set QAIHM_CI=1 with valid AWS credentials)."
            )
        return Path(super().fetch(extract=extract))


class UnfetchableDatasetError(Exception):
    def __init__(self, dataset_name: str, installation_steps: list[str] | None) -> None:
        """
        Create an error for datasets that cannot be automatically fetched in code.
        These datasets often require a login, license agreement acceptance, etc., to download.

        Parameters
        ----------
        dataset_name
            The name of the dataset being fetched.

        installation_steps
            Steps required for a 3rd party user to install this dataset manually.
            If None, the dataset is assumed to be not publicly available.
        """
        self.dataset_name = dataset_name
        self.installation_steps = installation_steps
        if installation_steps is None:
            super().__init__(
                f"Dataset {dataset_name} is for Qualcomm-internal usage only. If you have reached this error message when running an export or evaluate script, please file an issue at https://github.com/qualcomm/ai-hub-models/issues."
            )
        else:
            super().__init__(
                f"To use dataset {dataset_name}, you must download it manually. Follow these steps:\n"
                + "\n".join(
                    [f"{i + 1}. {step}" for i, step in enumerate(installation_steps)]
                )
            )


class CachedPrivateCIDatasetAsset(CachedPrivateCIAsset):
    """Private S3 cached asset scoped to a dataset."""

    def __init__(
        self,
        private_s3_key: str,
        dataset_id: str,
        dataset_version: int | str,
        filename: str,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
        installation_steps: list[str] | None = None,
        local_cache_extracted_path: str | Path | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        extracted: Path | None = None
        if local_cache_extracted_path is not None:
            extracted = asset_config.get_local_store_dataset_path(
                dataset_id, dataset_version, local_cache_extracted_path
            )
        super().__init__(
            private_s3_key,
            asset_config.get_local_store_dataset_path(
                dataset_id, dataset_version, filename
            ),
            asset_config,
            UnfetchableDatasetError(
                dataset_name=dataset_id,
                installation_steps=installation_steps,
            ),
            local_cache_extracted_path=extracted,
        )
