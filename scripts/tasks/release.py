# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

from .task import CompositeTask
from .util import get_env_bool, on_ci
from .venv import CreateVenvTask, RunCommandsWithVenvTask


class CreateReleaseVenv(CompositeTask):
    """Create a venv for building and releasing the repository / wheel."""

    def __init__(
        self,
        venv_path: str | os.PathLike,
        python_executable: str | None = None,
    ) -> None:
        tasks = []
        self.venv_path = str(venv_path)
        # Create Venv
        tasks.append(CreateVenvTask(self.venv_path, python_executable))
        # Install minimal build dependencies only for the wheel building.
        tasks.append(
            RunCommandsWithVenvTask(
                "Install build dependencies",
                venv=self.venv_path,
                commands=["pip install build"],
            )
        )
        super().__init__(
            group_name=f"Build Release Venv at {self.venv_path}", tasks=tasks
        )


class BuildWheelTask(CompositeTask):
    """
    Creates a wheel from the provided directory.
    If no directory is provided, assumes the release directory defined above.
    """

    def __init__(
        self,
        venv: str | None,
        wheel_dir: str | os.PathLike,
        overwrite_if_exists: bool = True,
        release_wheel: bool | None = get_env_bool("QAIHM_RELEASE_BUILD", False),
    ) -> None:
        tasks = []

        os.makedirs(wheel_dir, exist_ok=True)
        if overwrite_if_exists or not os.listdir(wheel_dir):
            env = os.environ.copy()
            env["QAIHM_RELEASE_BUILD"] = "1" if release_wheel else "0"
            tasks.append(
                RunCommandsWithVenvTask(
                    "Build Wheel",
                    venv=venv,
                    env=env,
                    commands=[
                        f"rm -f {os.path.join(wheel_dir, 'qai_hub_models-*.whl')}",
                        f"python -m build --wheel --outdir {wheel_dir}"
                        + (" > /dev/null" if on_ci() else ""),
                        f"echo 'Wheel can be found at {wheel_dir}'",
                    ],
                )
            )

        super().__init__(f"Build Wheel to: {wheel_dir}", tasks)
