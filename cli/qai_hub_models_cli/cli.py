# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import contextlib
import sys
from importlib.metadata import PackageNotFoundError, version

from qai_hub_models_cli._version import __version__
from qai_hub_models_cli.common import Precision, TargetRuntime
from qai_hub_models_cli.fetch import fetch, get_asset_url
from qai_hub_models_cli.utils import UnsupportedVersionError


def _check_version_match() -> None:
    """Exit if qai_hub_models and qai_hub_models_cli versions differ."""
    try:
        cli_version = version("qai_hub_models_cli")
        models_version = version("qai_hub_models")
    except PackageNotFoundError:
        return
    if cli_version != models_version:
        print(
            f"Version mismatch: qai_hub_models_cli=={cli_version} "
            f"but qai_hub_models=={models_version}. "
            "Please reinstall both packages from the same version."
        )
        sys.exit(1)


def _run_fetch(args: argparse.Namespace) -> None:
    try:
        if args.url_only:
            url = get_asset_url(
                args.model,
                args.runtime,
                args.precision,
                args.qaihm_version,
                args.chipset,
            )
            print(url)
            return

        result = fetch(
            model=args.model,
            runtime=args.runtime,
            precision=args.precision,
            chipset=args.chipset,
            version=args.qaihm_version,
            extract=args.extract,
            output_dir=args.output,
            quiet=args.quiet,
        )
    except Exception as e:
        if args.quiet and not isinstance(
            e, (FileNotFoundError, UnsupportedVersionError)
        ):
            print(
                "Failed to fetch model. Consider excluding -q/--quiet from your command to reveal more logs."
            )
        raise

    if args.quiet:
        print(result)
    elif args.extract:
        print(f"Extracted to: {result}")
    else:
        print(f"Saved to: {result}")


def add_fetch_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fetch",
        help="Download a pre-compiled model asset.",
    )
    parser.add_argument("model", type=str.lower, help="Model ID (e.g. mobilenet_v2).")
    runtime_values = ", ".join(rt.value for rt in TargetRuntime)
    parser.add_argument(
        "-r",
        "--runtime",
        required=True,
        type=str.lower,
        help=f"Target runtime. Known values: {runtime_values}. "
        "Older releases may support different values.",
    )
    precision_values = ", ".join(p.value for p in Precision)
    parser.add_argument(
        "-p",
        "--precision",
        default="float",
        type=str.lower,
        help=f"Model precision. Known values: {precision_values}. "
        "Older releases may support different values. Default: float.",
    )
    # TODO(#18389): Add a list of valid chipsets
    # so the CLI can validate and suggest chipset names.
    parser.add_argument(
        "-c",
        "--chipset",
        default=None,
        type=str.lower,
        help="Chipset name for device-specific (AOT compiled) runtimes.",
    )
    parser.add_argument(
        "-v",
        "--version",
        default=__version__,
        type=str.lower,
        dest="qaihm_version",
        help=f"AI Hub Models version tag (e.g. v0.45.0 or 0.45.0). Default: {__version__}.",
    )
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract the downloaded zip archive (default: true). Use --no-extract to skip.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        help="Output directory. Default: current directory.",
    )
    parser.add_argument(
        "--url-only",
        action="store_true",
        help="Print the download URL only (do not download).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all output except the result path.",
    )
    parser.set_defaults(func=_run_fetch)
    return parser


def main(args: list[str] | None = None) -> None:
    _check_version_match()

    parser = argparse.ArgumentParser(
        prog="qai_hub_models",
        description="Qualcomm AI Hub Models CLI.",
    )

    subparsers = parser.add_subparsers()
    add_fetch_parser(subparsers)

    # Allow qai_hub_models to add subcommands if installed
    with contextlib.suppress(ImportError):
        from qai_hub_models.cli import configure_parser

        configure_parser(subparsers)

    parsed = parser.parse_args(args)
    if hasattr(parsed, "func"):
        try:
            parsed.func(parsed)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
