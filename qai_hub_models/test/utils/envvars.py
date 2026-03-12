# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.utils.envvar_bases import (
    QAIHMBoolEnvvar,
    pytest_cli_envvar,
)


@pytest_cli_envvar
class DisableWorkbenchJobTimeoutEnvvar(QAIHMBoolEnvvar):
    """
    If this is false, a 1 hr timeout is enforced on jobs, post submission time.
    This is a separate timeout than the timeout on AI Hub Workbench. This timeout is intended for
    PR tests, so users don't have to wait hours for hub to time out their job to know it's failing.

    If this is true, the override timeout is disabled for AI Hub Models testing.
    We will wait for the job to time out on AI Hub Workbench instead."
    """

    VARNAME = "QAIHM_TEST_DISABLE_WORKBENCH_JOB_TIMEOUT"
    CLI_ARGNAMES = ["--disable-workbench-timeout"]
    CLI_HELP_MESSAGE = "For testing, AI Hub Models enforces a 1 hour timeout on workbench jobs (after submission time) by default. If True, the QAIHM-specific 1hr timeout is disabled."

    @classmethod
    def default(cls) -> bool:
        return False


__all__ = ["DisableWorkbenchJobTimeoutEnvvar"]
