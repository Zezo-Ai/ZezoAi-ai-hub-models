# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import hashlib
import os
import platform
import shutil
import tempfile

try:
    import git  # noqa: TID251
except ImportError as e:
    raise ImportError(
        "GitPython is required for external repo support. "
        "Please install git (https://github.com/git-guides/install-git) "
        "and then run: pip install gitpython"
    ) from e

from qai_hub_models.utils.asset_loaders import (
    EXECUTING_IN_CI_ENVIRONMENT,
    LOCAL_STORE_DEFAULT_PATH,
    query_yes_no,
)

EXTERNAL_REPOS_CACHE_DIR = os.path.join(LOCAL_STORE_DEFAULT_PATH, "external_repos")
USE_SYMLINK_CACHE = platform.system() != "Windows"
IS_PIP_PACKAGE = "site-packages" in os.path.normpath(__file__).split(os.sep)
MYPY_IGNORE = "# mypy: ignore-errors\n"


def compute_content_hash(
    repo_url: str,
    commit_sha: str,
    patch_contents: str = "",
) -> str:
    """Compute a hash of the repo configuration that produces the clone."""
    data = repo_url + "\n" + commit_sha + "\n" + patch_contents
    return hashlib.shake_256(data.encode()).hexdigest(8)


def get_cache_dir(model_id: str, repo_name: str, content_hash: str) -> str:
    """Return the cache directory path for a given repo."""
    return os.path.join(EXTERNAL_REPOS_CACHE_DIR, model_id, repo_name, content_hash)


def _shallow_clone(url: str, commit_sha: str, target: str) -> None:
    """Shallow-fetch a single commit into a target directory.

    Falls back to a full clone if the remote doesn't support shallow fetch
    of arbitrary SHAs.
    """
    repo = git.Repo.init(target)
    try:
        repo.git.fetch(url, commit_sha, depth=1)
    except git.GitCommandError:
        # Remote doesn't support shallow fetch of arbitrary SHAs — full fetch
        repo.git.fetch(url, commit_sha)
    repo.git.checkout("FETCH_HEAD")


def _apply_patch(target: str, patch_path: str) -> None:
    """Apply a patch file to a cloned repo."""
    repo = git.Repo(target)
    if platform.system() == "Windows":
        repo.git.apply(patch_path, ignore_space_change=True)
    else:
        repo.git.apply(patch_path)


def postprocess_repo(directory: str) -> None:
    """Walk a directory tree, create missing __init__.py, and add mypy ignore to all .py files."""
    for dirpath, dirnames, filenames in os.walk(directory):
        # Skip hidden directories (like .git)
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if "__init__.py" not in filenames:
            init_path = os.path.join(dirpath, "__init__.py")
            with open(init_path, "w") as f:
                f.write(MYPY_IGNORE)
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            with open(fpath) as f:
                content = f.read()
            if not content.startswith(MYPY_IGNORE):
                with open(fpath, "w") as f:
                    f.write(MYPY_IGNORE + content)


def clone_and_patch(
    url: str,
    commit_sha: str,
    target: str,
    patch_path: str | None = None,
) -> None:
    """Clone a repo at a pinned commit and optionally apply a patch."""
    _shallow_clone(url, commit_sha, target)
    if patch_path and os.path.exists(patch_path):
        _apply_patch(target, patch_path)
    postprocess_repo(target)


def clone_and_link_repo(
    model_id: str,
    repo_name: str,
    repo_url: str,
    commit_sha: str,
    content_hash: str,
    external_repos_dir: str,
    patch_path: str | None = None,
    ask_to_clone: bool = not EXECUTING_IN_CI_ENVIRONMENT,
) -> None:
    """
    Ensures a single repo is cloned and available.

    Three modes:
    - Pip (any OS): clones into cache, caller uses __path__ redirect.
    - Dev + symlinks (unix): clones into cache, symlinks into source tree.
    - Dev + no symlinks (Windows): clones directly into source tree, no cache.

    Crash safety: clones into a temp directory, then renames atomically.
    """
    is_pip = IS_PIP_PACKAGE
    use_cache = is_pip or USE_SYMLINK_CACHE
    cache_dir = get_cache_dir(model_id, repo_name, content_hash)
    repo_in_cache = os.path.join(cache_dir, repo_name)
    local_dir = os.path.join(external_repos_dir, repo_name)

    if use_cache:
        # Fast path: cache already populated
        if os.path.isdir(repo_in_cache):
            # Dev unix: ensure symlink exists
            if not is_pip and not os.path.exists(local_dir):
                os.symlink(repo_in_cache, local_dir)
            return
    # Windows dev: repo lives directly in source tree
    elif os.path.isdir(local_dir):
        return

    # Prompt user if needed
    if ask_to_clone:
        should_clone = query_yes_no(
            f"Model {model_id} requires external repo {repo_url}. Ok to clone?",
            timeout=60,
        )
        if not should_clone:
            raise RuntimeError(
                f"Unable to load {model_id} without its required repository {repo_url}."
            )

    print(f"Cloning {repo_url} ({commit_sha[:8]}) for {model_id}/{repo_name}...")

    # dir= ensures temp dir is on the same filesystem for atomic rename.
    # ignore_cleanup_errors=True because on success we rename the dir away
    # before the context manager tries to clean it up.
    if use_cache:
        parent = os.path.dirname(cache_dir)
        os.makedirs(parent, exist_ok=True)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True, dir=parent) as tmp:
            clone_and_patch(
                repo_url, commit_sha, os.path.join(tmp, repo_name), patch_path
            )
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            os.rename(tmp, cache_dir)

        # Dev unix: create symlink
        if not is_pip and not os.path.exists(local_dir):
            os.symlink(repo_in_cache, local_dir)
    else:
        # Dev Windows: clone directly into source tree
        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True, dir=external_repos_dir
        ) as tmp:
            clone_and_patch(repo_url, commit_sha, tmp, patch_path)
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            os.rename(tmp, local_dir)

    print("Done")


@dataclasses.dataclass(frozen=True)
class RepoConfig:
    """Configuration for a single external repository dependency."""

    repo_url: str
    commit_sha: str
    patches_filename: str | None = None


def get_patch_contents(config: RepoConfig, external_repos_dir: str) -> str:
    """Read patch file contents for a repo config, or return empty string."""
    if config.patches_filename:
        patch_path = os.path.join(external_repos_dir, config.patches_filename)
        if os.path.exists(patch_path):
            with open(patch_path) as f:
                return f.read()
    return ""


def compute_repo_hashes(
    repo_configs: dict[str, RepoConfig],
    external_repos_dir: str,
) -> dict[str, str]:
    """Compute content hash for each configured repo."""
    return {
        repo_name: compute_content_hash(
            config.repo_url,
            config.commit_sha,
            get_patch_contents(config, external_repos_dir),
        )
        for repo_name, config in repo_configs.items()
    }


def get_repo_cache_paths(
    model_id: str,
    repo_configs: dict[str, RepoConfig],
    external_repos_dir: str,
) -> list[str]:
    """Return cache directory paths for all external repos (for __path__ redirect)."""
    return [
        get_cache_dir(model_id, repo_name, content_hash)
        for repo_name, content_hash in compute_repo_hashes(
            repo_configs, external_repos_dir
        ).items()
    ]


def setup_external_repos_impl(
    model_id: str,
    repo_configs: dict[str, RepoConfig],
    external_repos_dir: str,
) -> None:
    """
    Clone all external repos for a model if needed.

    Pip mode: cache dir existence is sufficient (hash is in the path).
    Dev mode: each repo has a <repo_name>_hash.txt in external_repos/.
        Hashes are computed live. If the hash file matches, skip.
        On mismatch or missing, re-clone and write hash file.
    """
    is_pip = IS_PIP_PACKAGE
    per_repo_hashes = compute_repo_hashes(repo_configs, external_repos_dir)
    for repo_name, config in repo_configs.items():
        content_hash = per_repo_hashes[repo_name]

        if not is_pip:
            hash_file = os.path.join(external_repos_dir, f"{repo_name}_hash.txt")
            local_dir = os.path.join(external_repos_dir, repo_name)

            # Fast path: hash file matches and local dir exists
            stored_hash = None
            if os.path.exists(hash_file):
                with open(hash_file) as f:
                    stored_hash = f.read().strip()
            if stored_hash == content_hash and os.path.exists(local_dir):
                continue
            # Stale or incomplete — clean up so we re-clone
            if os.path.exists(hash_file):
                os.remove(hash_file)
            if os.path.islink(local_dir):
                os.remove(local_dir)
            elif os.path.isdir(local_dir):
                shutil.rmtree(local_dir)

        patch_path = None
        if config.patches_filename:
            patch_path = os.path.join(external_repos_dir, config.patches_filename)
        clone_and_link_repo(
            model_id=model_id,
            repo_name=repo_name,
            repo_url=config.repo_url,
            commit_sha=config.commit_sha,
            content_hash=content_hash,
            external_repos_dir=external_repos_dir,
            patch_path=patch_path,
        )

        # Dev mode: write hash file after successful setup
        if not is_pip:
            with open(hash_file, "w") as f:
                f.write(content_hash)
