#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Split the LLM perf workflow's `device` input into per-pool matrix entries.

Emits a single GITHUB_OUTPUT line `matrix_include=<json-array>` for the
matrix `include:` field. The dedicated QDC pool covers devices that use
QDC_PRIVATE_API_KEY; the aws pool covers devices that run through AWS
Device Farm instead of QDC (see ScorecardDevice.devicefarm_backend); everything
else uses the shared QDC pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEDICATED_DEVICES = {"cs_8_elite_qrd", "cs_x_elite", "cs_ventuno_q"}

# Keep in sync with ScorecardDevice instances whose devicefarm_backend="aws"
# (src/qai_hub_models/scorecard/device.py). This script runs before the venv
# is set up, so it can't import qai_hub_models to read that flag directly.
AWS_DEVICES = {"cs_8_elite", "cs_8_elite_gen_5"}

# Keep in sync with ALL_GENIEX_DEVICES in
# src/qai_hub_models/scripts/run_geniex_bench_benchmarks.py.
ALL_GENIEX_DEVICES = (
    "cs_x_elite",
    "cs_x2_elite",
    "cs_9075",
    "cs_8_elite",
    "cs_8_elite_gen_5",
)

# Superseded by the Samsung Galaxy S25/S26 (AWS) devices above; no longer part
# of the default "all" sweep, but still routable via explicit selection.--device for
RETIRED_FROM_ALL = {"cs_8_elite_qrd", "cs_8_elite_gen_5_qrd"}

# The LLM default device (DEFAULT_QDC_DEVICE in scorecard/device.py). The
# scorecard-wide is_default=True flag lives on cs_8_elite (Samsung Galaxy
# S25 Family), which is *not* the LLM default -- so if the token "default"
# reaches run_geniex_bench_benchmarks.py it would dispatch to the wrong
# device. Expand it here instead.
DEFAULT_LLM_DEVICE = "cs_x_elite"


def split(device_input: str) -> tuple[str, str, str]:
    device_input = (device_input or "all").strip()
    if device_input.lower() == "all":
        shared = ",".join(
            d
            for d in ALL_GENIEX_DEVICES
            if d not in DEDICATED_DEVICES and d not in AWS_DEVICES
        )
        return (
            shared,
            ",".join(sorted(DEDICATED_DEVICES - RETIRED_FROM_ALL)),
            ",".join(sorted(AWS_DEVICES)),
        )

    shared: list[str] = []
    dedicated: list[str] = []
    aws: list[str] = []
    for raw in device_input.split(","):
        d = raw.strip()
        if not d:
            continue
        if d.lower() == "default":
            d = DEFAULT_LLM_DEVICE
        if d in AWS_DEVICES:
            aws.append(d)
        elif d in DEDICATED_DEVICES:
            dedicated.append(d)
        else:
            shared.append(d)
    return ",".join(shared), ",".join(dedicated), ",".join(aws)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.environ.get("DEVICE_INPUT", "all"))
    args = parser.parse_args(argv)

    shared, dedicated, aws = split(args.device)
    entries = []
    if shared:
        entries.append({"pool": "shared", "devices": shared})
    if dedicated:
        entries.append({"pool": "dedicated", "devices": dedicated})
    if aws:
        entries.append({"pool": "aws", "devices": aws})

    matrix_include = json.dumps(entries)
    print(f"shared={shared}   dedicated={dedicated}   aws={aws}", file=sys.stderr)
    print(f"matrix include: {matrix_include}", file=sys.stderr)

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as f:
            f.write(f"matrix_include={matrix_include}\n")
    else:
        print(matrix_include)
    return 0


if __name__ == "__main__":
    sys.exit(main())
