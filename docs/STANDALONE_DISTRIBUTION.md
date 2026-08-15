# Standalone Invariant source release

Invariant has one supported standalone distribution contract: a versioned source-release ZIP.
It does not require an editable checkout. The archive contains:

- one ordinary `sigma-theory-compiler` wheel for non-editable installation;
- every Git-tracked config, formal asset, immutable receipt, example, script, source file, test,
  and document at the release commit;
- the hydrated bytes for every Git LFS path, not Git LFS pointer files; and
- a closed `RELEASE-MANIFEST.json` with the byte length and SHA-256 of every payload file, plus
  the Git LFS object SHA-256 for every LFS resource.

The wheel is the Python runtime boundary. The extracted archive root is the resource and
provenance boundary. Git metadata and Git LFS are not required after extraction.

## Build the release

Build only from a clean, fully hydrated checkout:

```powershell
python -m pip install --editable ".[release]"
git lfs pull
git lfs fsck
git status --short
python scripts/build_standalone_release.py `
  --output-directory dist
```

The builder rejects dirty checkouts, missing LFS objects, pointer files, symlinks, submodules,
case-insensitive path collisions, malformed Git state, and an existing output artifact. ZIP member
order, timestamps, modes, and storage method are fixed. The project version in `pyproject.toml`
names both the archive and its single root directory.

For repository tests only, `--allow-dirty` permits the test's staged implementation to be built.
Such a manifest records `source_tree_clean: false` and is not a release candidate.

## Install and verify

Extract `invariant-0.1.0-source-release.zip`, enter `Invariant-0.1.0`, and install the bundled
wheel normally. This is not an editable install:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install packages\sigma_theory_compiler-0.1.0-py3-none-any.whl
.venv\Scripts\sigma-release verify --release-root .
```

Installation may obtain the declared Python dependencies from the configured package index. The
Formula Discovery examples themselves perform no network access. For an offline environment that
already provides the declared dependencies, add `--no-deps --no-index` to the install command.

Verification is fail-closed. It checks the manifest checksum, the exact file set, every file size
and SHA-256, the single wheel, all manifest counts, and all hydrated LFS object IDs. A missing,
extra, or tampered resource fails before a formula run or provenance replay is trusted.

## Public PASS and REJECT examples

Use caller-owned output paths outside the release tree:

```powershell
New-Item -ItemType Directory -Force work\standalone-pass | Out-Null
.venv\Scripts\sigma-formula-discovery run `
  --problem examples\formula-discovery\pass-exact-polynomial.json `
  --result work\standalone-pass\result.json `
  --report work\standalone-pass\report.md
# exit 0; decision PASS; exact formula x**3 - 2*x + 5

New-Item -ItemType Directory -Force work\standalone-reject | Out-Null
.venv\Scripts\sigma-formula-discovery run `
  --problem examples\formula-discovery\reject-heldout-counterexample.json `
  --result work\standalone-reject\result.json `
  --report work\standalone-reject\report.md
# exit 10; decision REJECT; exact held-out counterexample documented
```

Replay each result without rewriting it by replacing `run` with `validate` and keeping the same
three paths. REJECT deliberately remains exit code 10 during successful replay because it is the
scientific decision, not an operational error.

## Provenance materialization fixed point

Materialize the registered historical byte forms from the extracted resource root, then run the
same command again:

```powershell
.venv\Scripts\python scripts\materialize_hash_bound_worktree.py --project-root .
.venv\Scripts\python scripts\materialize_hash_bound_worktree.py --project-root .
```

The second result must report `"files_rewritten": 0`. Release acceptance additionally hashes the
complete tracked resource set after each invocation and requires byte-for-byte equality between
the first fixed point and the second pass. Because materialization intentionally restores some
historical registered line endings, run `sigma-release verify` before materialization; retain the
original archive as the immutable distribution artifact.

## Decisive acceptance command

```powershell
python -m pytest tests/test_standalone_source_release.py -q
python -m ruff check `
  src/sigma_theory_compiler/standalone_release.py `
  scripts/build_standalone_release.py `
  tests/test_standalone_source_release.py
```

The isolated-install test builds the real release bundle, extracts it away from the checkout,
installs only its wheel into a new virtual environment, verifies all resources/LFS identities,
runs and replays both public decisions, proves second-pass materializer stability, and exercises
missing-resource and tampered-LFS failures.
