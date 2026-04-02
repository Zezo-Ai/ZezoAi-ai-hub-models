# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import contextlib
from importlib.metadata import PackageNotFoundError, version


def _check_version_match() -> None:
    """Raise if qai_hub_models and qai_hub_models_cli versions differ."""
    try:
        cli_version = version("qai_hub_models_cli")
        models_version = version("qai_hub_models")
    except PackageNotFoundError:
        return
    if cli_version != models_version:
        raise RuntimeError(
            f"Version mismatch: qai_hub_models_cli=={cli_version} "
            f"but qai_hub_models=={models_version}. "
            "Please reinstall both packages from the same version."
        )


def main(args: list[str] | None = None) -> None:
    _check_version_match()

    parser = argparse.ArgumentParser(
        prog="qai_hub_models",
        description="Qualcomm AI Hub Models CLI.",
    )

    subparsers = parser.add_subparsers()

    # Allow qai_hub_models to add subcommands if installed
    with contextlib.suppress(ImportError):
        from qai_hub_models.cli import configure_parser

        configure_parser(subparsers)

    parsed = parser.parse_args(args)
    if hasattr(parsed, "func"):
        parsed.func(parsed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
