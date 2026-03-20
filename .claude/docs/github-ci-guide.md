# GitHub CLI: Common Mistakes to Avoid

When using `gh` to interact with PRs and CI, there are several pitfalls that waste time if you don't know about them upfront.

## 1. jq expressions with `!=` break in bash

Bash intercepts `!` inside double-quoted strings and in `--jq` arguments. This means `!=` silently gets mangled.

```bash
# BROKEN — bash escapes the !
gh pr checks 123 --jq '.[] | select(.state != "SUCCESS")'
# Error: unexpected token "\\"

# FIX — use positive matches instead
gh pr checks 123 --json name,state \
  --jq '.[] | select(.state == "FAILURE" or .state == "ERROR") | .name + ": " + .state'
```

This applies to all `gh` commands with `--jq`. Never use `!=` — always rewrite as positive conditions.

## 2. `--log-failed` returns nothing for in-progress runs

`gh run view {run_id} --log-failed` silently returns empty output (or a brief message) if the run hasn't completed. There's no obvious error — it just looks like there are no logs.

**Always check run status first:**
```bash
gh run view {run_id} --json jobs \
  --jq '.jobs[] | .name + ": " + .status + " / " + .conclusion'
```

If any job shows `in_progress`, wait before trying to read logs.

**Fallback for when logs aren't available via `--log-failed`:**
```bash
# Get the failed job ID
gh api repos/{owner}/{repo}/actions/runs/{run_id}/jobs \
  --jq '.jobs[] | select(.conclusion == "failure") | {name: .name, id: .id}'

# Fetch that job's logs directly
gh api repos/{owner}/{repo}/actions/jobs/{job_id}/logs 2>&1 | grep "FAILED src/" | head -10
```

## 3. Grepping for "FAILED" matches test names, not just failures

CI logs contain pytest test collection output like:
```
<Function test_user_can_get_public_inference_jobs[FAILED_JOBS-STATE_BY_STRING]>
```

The word "FAILED" here is part of the test *name* (a parametrize ID), not an actual failure. Grepping broadly for "FAILED" returns hundreds of these false positives.

**Always use this pattern to find actual test failures:**
```bash
gh run view {run_id} --log-failed 2>&1 | grep "FAILED src/" | head -20
```

The `FAILED src/` prefix is what pytest prints for actual failures. For build errors, grep for `WARNING:` (Sphinx) or `error:` (compilers).

## Quick reference: common tasks

```bash
# Find PR for current branch
gh pr list --head $(git branch --show-current) --json number,title,url

# Fetch inline PR review comments
gh api repos/{owner}/{repo}/pulls/{pr_number}/comments \
  --jq '[.[] | {user: .user.login, path: .path, line: (.line // .original_line), body: .body}]'

# List failed CI checks
gh pr checks {pr_number} --json name,state \
  --jq '.[] | select(.state == "FAILURE") | .name + ": " + .state'
```
