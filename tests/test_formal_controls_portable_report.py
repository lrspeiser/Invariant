from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.formal_controls_portable_report import (
    _content_sha,
    _windows_to_wsl_text,
    build_report,
    portable_projection,
    validate_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/formal_controls_portable_report.json"
ARTIFACT = ROOT / "runs/formal-controls-v1/formal-controls-portable.json"


def test_portable_report_is_deterministic_and_host_path_free() -> None:
    expected = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert build_report(CONFIG) == expected
    validate_artifact(expected, ROOT, CONFIG)
    assert expected["decision"] == "pass_portable_semantic_projection_118_controls"
    assert expected["semantic_report"]["counts"] == {"total": 118, "passed": 118, "failed": 0}
    assert len(expected["semantic_report"]["checks"]) == 118
    text = json.dumps(expected)
    assert "C:\\Users\\" not in text
    assert "/mnt/c/" not in text
    assert "created_utc" not in expected["semantic_report"]
    assert all(not value for value in expected["claim_seals"].values())


def test_projection_is_invariant_to_root_and_timestamp_changes() -> None:
    report = json.loads(
        (ROOT / "runs/formal-controls-v1/formal-controls.json").read_text(encoding="utf-8")
    )
    hashes = {
        "executable_sha256": "1" * 64,
        "python_module_sha256": "2" * 64,
    }
    first, _ = portable_projection(report, ROOT, backend_hashes=hashes)
    mutated = deepcopy(report)
    mutated["created_utc"] = "2099-01-01T00:00:00+00:00"
    old_root = str(ROOT.resolve())
    new_root = r"D:\portable\sigma-theory-compiler"
    old_cadabra = mutated["backends"]["cadabra2"]["root"]
    new_cadabra = r"D:\portable\cadabra-root"
    serialized = json.dumps(mutated).replace(
        old_root.replace("\\", "\\\\"), new_root.replace("\\", "\\\\")
    )
    serialized = serialized.replace(
        old_cadabra.replace("\\", "\\\\"), new_cadabra.replace("\\", "\\\\")
    )
    serialized = serialized.replace(
        _windows_to_wsl_text(old_cadabra), _windows_to_wsl_text(new_cadabra)
    )
    mutated = json.loads(serialized)
    second, _ = portable_projection(mutated, Path(new_root), backend_hashes=hashes)
    assert first == second


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(decision="candidate_formal_pass"),
        lambda value: value["claim_seals"].update(candidate_theory_validity_claimed=True),
        lambda value: value["portability"].update(absolute_windows_paths=1),
        lambda value: value["semantic_report"]["counts"].update(passed=117, failed=1),
        lambda value: value.update(extra_claim=False),
    ],
)
def test_rehashed_semantic_tampering_fails_closed(mutation) -> None:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    mutation(value)
    value["content_sha256"] = _content_sha(value)
    with pytest.raises(ValueError):
        validate_artifact(value, ROOT, CONFIG)
