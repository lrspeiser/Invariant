from __future__ import annotations

import re
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "sigma-theory-compiler.yml"
BEGIN = "# BEGIN comprehensive-alpha-test-manifest"
END = "# END comprehensive-alpha-test-manifest"
PATH_PATTERN = re.compile(r"tests/test_[a-z0-9_]+\.py")
SLICE_PATTERN = re.compile(r"(?m)^\s{10}- slice: ([a-z0-9-]+)$")
EXPECTED_PATH_COUNT = 169
EXPECTED_PATH_SET_SHA256 = "706d4a489254a7b7360b9623af550fe0fb7eb6746ac8bdb30b0c15c63c264aa9"
EXPECTED_SLICE_COUNTS = {
    "discovery-and-proof": 48,
    "system7-coordinate-authority": 15,
    "system7-lower-p0": 1,
    "system7-lower-p1": 1,
    "system7-lower-p2": 1,
    "system7-lower-p3-and-q": 5,
    "system10-cylindrical": 29,
    "system8-system9-recurrence": 13,
    "pde-controls-and-85-state": 25,
    "math-controls-and-operations": 31,
}


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _manifest_block(text: str) -> str:
    assert text.count(BEGIN) == 1
    assert text.count(END) == 1
    return text.split(BEGIN, 1)[1].split(END, 1)[0]


def _slice_entries(block: str) -> dict[str, list[str]]:
    matches = list(SLICE_PATTERN.finditer(block))
    entries: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        entries[match.group(1)] = PATH_PATTERN.findall(block[match.end() : stop])
    return entries


def test_current_comprehensive_manifest_is_partitioned_exactly_once() -> None:
    entries = _slice_entries(_manifest_block(_workflow_text()))
    assert {name: len(paths) for name, paths in entries.items()} == EXPECTED_SLICE_COUNTS

    paths = [path for slice_paths in entries.values() for path in slice_paths]
    counts = Counter(paths)
    assert len(paths) == EXPECTED_PATH_COUNT
    assert len(counts) == EXPECTED_PATH_COUNT
    assert set(counts.values()) == {1}
    assert all((ROOT / path).is_file() for path in paths)

    encoded = ("\n".join(sorted(paths)) + "\n").encode()
    assert sha256(encoded).hexdigest() == EXPECTED_PATH_SET_SHA256


def test_common_gate_and_parallel_slices_keep_actionable_diagnostics() -> None:
    text = _workflow_text()
    common = text.split("\n  comprehensive_alpha_common:\n", 1)[1].split(
        "\n  comprehensive_alpha_slices:\n", 1
    )[0]
    matrix = text.split("\n  comprehensive_alpha_slices:\n", 1)[1].split(
        "\n  comprehensive-alpha:\n", 1
    )[0]
    aggregate = text.split("\n  comprehensive-alpha:\n", 1)[1].split(
        "\n  portable-continuous-and-gates:\n", 1
    )[0]

    assert "timeout-minutes: 20" in common
    assert "uses: leanprover/lean-action@v1" in common
    assert "python scripts/materialize_hash_bound_worktree.py" in common
    assert "Check comprehensive-alpha source" in common
    assert "tests/test_comprehensive_alpha_workflow_topology.py" in common

    assert "needs: comprehensive_alpha_common" in matrix
    assert "timeout-minutes: 120" in matrix
    assert "fail-fast: false" in matrix
    assert "uses: leanprover/lean-action@v1" in matrix
    assert (
        matrix.index("uses: actions/setup-python@v5")
        < matrix.index("uses: leanprover/lean-action@v1")
        < matrix.index("python scripts/materialize_hash_bound_worktree.py")
    )
    assert "python scripts/materialize_hash_bound_worktree.py" in matrix
    assert 'python -m pip install --editable ".[dev]"' in matrix
    assert "${{ matrix.test_files }}" in matrix
    assert matrix.count("-vv --tb=short --durations=20") == 1

    assert "name: comprehensive-alpha" in aggregate
    assert "if: ${{ always() }}" in aggregate
    assert "comprehensive_alpha_common" in aggregate
    assert "comprehensive_alpha_slices" in aggregate
    assert 'test "$COMMON_RESULT" = "success"' in aggregate
    assert 'test "$SLICES_RESULT" = "success"' in aggregate
    assert "Verify comprehensive-alpha vertical slices" not in text
