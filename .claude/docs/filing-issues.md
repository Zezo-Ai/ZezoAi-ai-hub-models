# Filing Issues in qcom-ai-hub/tetracode

All bugs and feature requests for AI Hub (workbench, models, SDK, infrastructure) are tracked in the `qcom-ai-hub/tetracode` GitHub repo.

## Defaults

- **Label**: `ai-hub-models` — always include this unless the issue is unrelated to this repo
- **Priority**: `P2` — use this unless the user specifies otherwise

## Creating an issue

Use the REST API to set issue type, labels, and assignee in one call:

```bash
gh api repos/qcom-ai-hub/tetracode/issues \
  --method POST \
  -f title="<short title>" \
  -f body="<body>" \
  -f type="Task" \
  -f 'labels[][name]=P2' \
  -f 'labels[][name]=ai-hub-models' \
  -f 'assignees[]=<github_username>'
```

## Required fields

Every issue must have:

1. **Title** — short, specific, searchable (e.g. "Link Jobs do not save I/O specs" not "bug in workbench")
2. **Priority label** — exactly one of:
   - `P0` — Burning fire. Drop everything.
   - `P1` — Urgent, top priority feature work
   - `P2` — Feature work that needs to get done
   - `P3` — Nice to have
3. **Component label** — at least one (see label list below)
4. **Assignee** — GitHub username of the person responsible
5. **Body** — structured with Summary, Expected Behavior, and Actual Behavior sections

## Issue type

Issue types are an org-level feature (not labels). Set them via the API when creating an issue:

| Type | When to use |
|------|-------------|
| `Bug` | Something is broken or behaving incorrectly |
| `Task` | Work item that isn't a bug or feature (refactor, cleanup, investigation) |
| `Feature` | A request, idea, or new functionality |

To create an issue with a type, use the REST API instead of `gh issue create`:

```bash
gh api repos/qcom-ai-hub/tetracode/issues \
  --method POST \
  -f title="<title>" \
  -f body="<body>" \
  -f type="Bug"
```

Labels and assignees can be added in the same call:

```bash
gh api repos/qcom-ai-hub/tetracode/issues \
  --method POST \
  -f title="<title>" \
  -f body="<body>" \
  -f type="Task" \
  -f 'labels[][name]=P2' \
  -f 'labels[][name]=ai-hub-models' \
  -f 'assignees[]=<github_username>'
```

If the user doesn't specify a type, infer it from context. Default to `Task` if ambiguous.

## Common labels

### Priority
| Label | Meaning |
|-------|---------|
| `P0` | Burning fire — drop everything |
| `P1` | Urgent, top priority |
| `P2` | Needs to get done |
| `P3` | Nice to have |

### Size (optional)
| Label | Meaning |
|-------|---------|
| `small` | ~1 day |
| `medium` | 2-3 days |
| `large` | 4+ days |

### Component / area
| Label | Use for |
|-------|---------|
| `ai-hub-models` | Issues in this repo (qaihm models) |
| `Compiler Service` | Compile job issues |
| `Devices` | Device issues, bring-up, AWS Device Farm, QDC |
| `Quantization` | Quantization-related |
| `qnn` | QNN-related issues |
| `scorecard` | Scorecard improvements |
| `Cloud services` | Cloud service tasks |
| `Docs` | Documentation |
| `Testing` | Test infrastructure |
| `Security` | Exploits or vulnerabilities |
| `perf` | Performance issues |

### Other useful labels
- `Blocked` — waiting on something external
- `untriaged` — needs triage
- `Backlog` — needs roadmap decision

## Cross-referencing issues

When two issues are related, add a comment linking them:

```bash
gh issue comment <issue_number> --repo qcom-ai-hub/tetracode \
  --body "Related: #<other_issue_number>"
```

GitHub auto-links `#NNN` within the same repo.

## Querying existing issues

```bash
# Search by label
gh issue list --repo qcom-ai-hub/tetracode --label "ai-hub-models" --label "P0"

# Search by assignee
gh issue list --repo qcom-ai-hub/tetracode --assignee "username_QCOM"

# Search by keyword
gh issue list --repo qcom-ai-hub/tetracode --search "link job I/O"

# View a specific issue
gh issue view <issue_number> --repo qcom-ai-hub/tetracode
```

## Finding assignees

To match a person's name to their GitHub username, search active contributors on this repo:

```bash
gh api repos/qcom-ai-hub/ai-hub-models-internal/contributors --jq '.[].login'
```

Or search by name across the org:

```bash
gh api 'search/users?q=<name>+org:qcom-ai-hub' --jq '.items[].login'
```
