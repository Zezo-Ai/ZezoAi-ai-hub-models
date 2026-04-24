# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import ruamel.yaml
from google.protobuf.json_format import MessageToDict, MessageToJson
from google.protobuf.message import Message
from mypy_boto3_s3.service_resource import Bucket
from qai_hub_models_cli.proto import manifest_pb2

from qai_hub_models._version import __version__
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.configs.proto_helpers import domain_to_proto
from qai_hub_models.configs.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.envvars import EnabledModelsEnvvar, SpecialModelSetting
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.aws import (
    QAIHM_PUBLIC_S3_BUCKET,
    get_qaihm_s3_or_exit,
    s3_multipart_upload,
)
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

Format = Literal["yaml", "json", "proto"]

MODEL_YAMLS_TO_COPY = [
    "info.yaml",
    "perf.yaml",
    "numerics.yaml",
]

GLOBAL_YAMLS_TO_COPY = [
    "asset_bases.yaml",
    "devices_and_chipsets.yaml",
]


_EXT: dict[Format, str] = {"yaml": ".yaml", "json": ".json", "proto": ".pb"}

# Proto enum prefixes from release_assets.proto (Precision, Runtime).
# Update this list if new enum types are added to the proto schema.
_ENUM_PREFIXES = ["PRECISION_", "RUNTIME_"]


def _simplify_enum_values_for_website_import(obj: Any) -> Any:
    # The website expects lowercase names without the proto prefix (e.g. "w8a8", not "PRECISION_W8A8").
    if isinstance(obj, dict):
        return {k: _simplify_enum_values_for_website_import(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_simplify_enum_values_for_website_import(v) for v in obj]
    if isinstance(obj, str):
        for prefix in _ENUM_PREFIXES:
            if obj.startswith(prefix):
                return obj[len(prefix) :].lower()
    return obj


def _make_yaml() -> ruamel.yaml.YAML:
    y = ruamel.yaml.YAML()
    y.width = 4096
    return y


def _write_proto(msg: Message, path: Path, fmt: Format) -> Path:
    path = path.with_suffix(_EXT[fmt])
    if fmt == "yaml":
        proto_dict = MessageToDict(msg, preserving_proto_field_name=True)
        with open(path, "w") as f:
            _make_yaml().dump(proto_dict, f)
    elif fmt == "json":
        path.write_text(MessageToJson(msg, preserving_proto_field_name=True))
    else:
        path.write_bytes(msg.SerializeToString())
    return path


def _write_proto_multi(msg: Message, path: Path, fmts: list[Format]) -> list[Path]:
    return [_write_proto(msg, path, fmt) for fmt in fmts]


def _build_release_assets_proto(
    model_id: str,
    aihm_version: str,
    output_dir: Path,
    fmts: list[Format],
) -> list[Path]:
    release_assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    if release_assets.empty:
        return []

    for precision, prec_details in release_assets.precisions.items():
        for path, asset in prec_details.universal_assets.items():
            asset.download_url = ASSET_CONFIG.get_release_asset_url(
                model_id, aihm_version, path.runtime, precision, None
            )
        for chipset, path_dict in prec_details.chipset_assets.items():
            for path, asset in path_dict.items():
                asset.download_url = ASSET_CONFIG.get_release_asset_url(
                    model_id, aihm_version, path.runtime, precision, chipset
                )

    proto_msg = release_assets.to_proto(aihm_version, model_id)
    result: list[Path] = []
    for fmt in fmts:
        if fmt == "yaml":
            out_path = (output_dir / "release-assets").with_suffix(".yaml")
            proto_dict = MessageToDict(proto_msg, preserving_proto_field_name=True)
            proto_dict = _simplify_enum_values_for_website_import(proto_dict)
            with open(out_path, "w") as f:
                _make_yaml().dump(proto_dict, f)
            result.append(out_path)
        else:
            result.append(_write_proto(proto_msg, output_dir / "release-assets", fmt))
    return result


def _build_model_protos(
    model_id: str,
    aihm_version: str,
    output_dir: Path,
    fmts: list[Format],
) -> tuple[list[Path], str | None]:
    info = QAIHMModelInfo.from_model(model_id)
    if info.status != MODEL_STATUS.PUBLISHED:
        return [], f"status={info.status.value}"

    written: list[Path] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    written.extend(
        _write_proto_multi(info.to_proto(aihm_version), output_dir / "info", fmts)
    )

    perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
    if not perf.empty:
        written.extend(
            _write_proto_multi(
                perf.to_proto(aihm_version, model_id), output_dir / "perf", fmts
            )
        )

    numerics = QAIHMModelNumerics.from_model(model_id, not_exists_ok=True)
    if numerics is not None:
        written.extend(
            _write_proto_multi(
                numerics.to_proto(aihm_version, model_id),
                output_dir / "numerics",
                fmts,
            )
        )

    written.extend(
        _build_release_assets_proto(model_id, aihm_version, output_dir, fmts)
    )

    return written, None


def _build_platform_proto(
    aihm_version: str,
    output_dir: Path,
    fmts: list[Format],
) -> list[Path]:
    platform = DevicesAndChipsetsYaml.load()
    return _write_proto_multi(
        platform.to_proto(aihm_version), output_dir / "platform", fmts
    )


def _copy_global_yamls(output_root: Path, version: str) -> None:
    version_tag = f"v{version}" if not version.startswith("v") else version

    src_root = QAIHM_MODELS_ROOT.parent
    for yaml_name in GLOBAL_YAMLS_TO_COPY:
        src = (src_root / yaml_name).resolve()
        dst = (output_root / yaml_name).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src != dst:
            shutil.copy2(src, dst)

    # Pin the models repo URL to the release tag (but not other repos like ai-hub-apps).
    asset_bases_dst = output_root / "asset_bases.yaml"
    if asset_bases_dst.exists():
        text = asset_bases_dst.read_text()
        text = text.replace(
            "ai-hub-models/blob/main", f"ai-hub-models/blob/{version_tag}"
        )
        text = text.replace(
            "ai-hub-models/tree/main", f"ai-hub-models/tree/{version_tag}"
        )
        asset_bases_dst.write_text(text)


def _for_each_model(
    fn: Callable[[str], tuple[int, str | None]], model_ids: list[str]
) -> None:
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {model_id: pool.submit(fn, model_id) for model_id in model_ids}
        for model_id, future in futures.items():
            written, skip_reason = future.result()
            if written:
                print(f"  {model_id}: wrote {written} files")
            else:
                print(f"  {model_id}: skipped ({skip_reason})")


def cmd_website(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_global_yamls(output_root, args.version)

    def build_model(model_id: str) -> tuple[int, str | None]:
        model_src = QAIHM_MODELS_ROOT / model_id
        model_dst = output_root / "models" / model_id
        model_dst.mkdir(parents=True, exist_ok=True)
        written = 0

        for yaml_name in MODEL_YAMLS_TO_COPY:
            src = (model_src / yaml_name).resolve()
            dst = (model_dst / yaml_name).resolve()
            if src.exists() and src != dst:
                shutil.copy2(src, dst)
                written += 1

        written += len(
            _build_release_assets_proto(model_id, args.version, model_dst, ["yaml"])
        )

        return written, (None if written else "no yamls found")

    enabled, _ = validate_and_split_enabled_models(args.models)
    model_ids = sorted(enabled)
    _for_each_model(build_model, model_ids)
    print(f"Built website yamls for {len(model_ids)} models in {output_root}")


def _build_manifest(
    version: str,
    model_written_paths: dict[str, list[Path]],
    platform_s3_url: str,
    output_dir: Path,
    fmts: list[Format],
) -> list[Path]:
    _STEM_TO_FIELD = [
        ("info", "info"),
        ("perf", "perf"),
        ("numerics", "numerics"),
        ("release-assets", "release_assets"),
    ]

    model_infos: list[tuple[str, set[str], QAIHMModelInfo, str]] = []
    for model_id, written in model_written_paths.items():
        info = QAIHMModelInfo.from_model(model_id)
        written_stems = {p.stem for p in written}
        s3_folder_url = ASSET_CONFIG.get_asset_url(
            ASSET_CONFIG.get_release_s3_folder(model_id, version)
        )
        model_infos.append((model_id, written_stems, info, s3_folder_url))

    result: list[Path] = []
    for fmt in fmts:
        ext = _EXT[fmt]
        manifest = manifest_pb2.ReleaseManifest(
            version=version,
            platform_url=platform_s3_url.replace(".json", ext),
        )
        for model_id, written_stems, info, s3_folder_url in model_infos:
            url_kwargs: dict[str, str] = {}
            for stem, field in _STEM_TO_FIELD:
                if stem in written_stems:
                    url_kwargs[field] = os.path.join(s3_folder_url, f"{stem}{ext}")
            manifest.models.append(
                manifest_pb2.ManifestModelEntry(
                    id=model_id,
                    display_name=info.name,
                    domain=domain_to_proto(info.domain),
                    manifest_urls=manifest_pb2.ModelManifestUrls(**url_kwargs),
                )
            )
        result.append(_write_proto(manifest, output_dir / "manifest", fmt))

    return result


def cmd_aws(args: argparse.Namespace) -> None:
    bucket: Bucket | None = None
    if args.upload:
        bucket, _ = get_qaihm_s3_or_exit(QAIHM_PUBLIC_S3_BUCKET)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    all_files: list[tuple[Path, str]] = []

    aws_fmts: list[Format] = ["json", "proto"]
    global_s3_folder = ASSET_CONFIG.get_global_release_s3_folder(args.version)
    platform_paths = _build_platform_proto(args.version, output_root, aws_fmts)
    all_files.extend(
        (path, os.path.join(global_s3_folder, path.name)) for path in platform_paths
    )

    model_written_paths: dict[str, list[Path]] = {}

    def build_model(model_id: str) -> tuple[int, str | None]:
        written, skip_reason = _build_model_protos(
            model_id, args.version, output_root / "models" / model_id, aws_fmts
        )
        if written:
            s3_folder = ASSET_CONFIG.get_release_s3_folder(model_id, args.version)
            all_files.extend(
                (path, os.path.join(s3_folder, path.name)) for path in written
            )
            model_written_paths[model_id] = written
        return len(written), skip_reason

    enabled, _ = validate_and_split_enabled_models(args.models)
    model_ids = sorted(enabled)
    _for_each_model(build_model, model_ids)

    platform_s3_url = ASSET_CONFIG.get_asset_url(
        os.path.join(global_s3_folder, "platform.json")
    )
    manifest_paths = _build_manifest(
        args.version, model_written_paths, platform_s3_url, output_root, aws_fmts
    )
    all_files.extend(
        (path, os.path.join(global_s3_folder, path.name)) for path in manifest_paths
    )

    print(f"Built proto JSON for {len(model_ids)} models in {output_root}")

    if args.upload:
        assert bucket
        with ThreadPoolExecutor(max_workers=64) as pool:
            futures = [
                pool.submit(
                    s3_multipart_upload,
                    bucket,
                    s3_key,
                    local_path,
                    make_public=True,
                    disable_progress=True,
                )
                for local_path, s3_key in all_files
            ]
            for f in futures:
                f.result()
        print(f"Uploaded {len(all_files)} files to s3://{QAIHM_PUBLIC_S3_BUCKET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build release proto files for website or AWS."
    )
    parser.add_argument(
        "mode",
        choices=["website", "aws"],
        help="Build mode: 'website' for YAML files, 'aws' for proto JSON.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Root output directory.",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default=__version__,
        help="AIHM version string. Defaults to installed version.",
    )
    EnabledModelsEnvvar.add_arg(parser, {SpecialModelSetting.PYTORCH})
    parser.add_argument(
        "--upload",
        "-u",
        action="store_true",
        help="Upload proto JSON files to S3 after building (aws mode only).",
    )

    args = parser.parse_args()
    if args.mode == "website":
        cmd_website(args)
    elif args.mode == "aws":
        cmd_aws(args)


if __name__ == "__main__":
    main()
