# Research publication cadence

The user instructed on 2026-09-05 that completed research updates should be
pushed to GitHub `main` regularly. During active gravity research, commit and
publish each completed, validated milestone rather than accumulating a long
local-only history. This is a milestone workflow, not a scheduled automation.

Before publishing, inspect the working tree, include only the intended changes,
run the relevant checks, and fetch `origin/main`. Preserve remote changes and
use an ordinary fast-forward push, integrating concurrent work when necessary.
Do not force-push or discard another task's changes.

Keep large private/raw observational datasets outside Git. Publish source URLs,
hashes, reproducible scripts, findings and their scientific limitations. Do not
present unfinished experiments as completed results.
