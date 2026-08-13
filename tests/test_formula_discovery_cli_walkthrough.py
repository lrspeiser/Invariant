from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.formula_discovery_cli import EXIT_PASS, EXIT_REJECT, main
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
PASS_PROBLEM = ROOT / "examples/formula-discovery/pass-exact-polynomial.json"
REJECT_PROBLEM = ROOT / "examples/formula-discovery/reject-heldout-counterexample.json"
WALKTHROUGH = ROOT / "docs/formula-discovery-cli-walkthrough.md"


def _exercise(tmp_path: Path, problem_path: Path, expected_exit: int) -> dict[str, object]:
    stem = problem_path.stem
    result_path = tmp_path / f"{stem}-result.json"
    report_path = tmp_path / f"{stem}-report.md"
    arguments = (
        "--problem",
        str(problem_path),
        "--result",
        str(result_path),
        "--report",
        str(report_path),
    )
    assert main(("run", *arguments)) == expected_exit
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_bytes = result_path.read_bytes()
    report_bytes = report_path.read_bytes()
    assert main(("validate", *arguments)) == expected_exit
    assert result_path.read_bytes() == result_bytes
    assert report_path.read_bytes() == report_bytes
    return result


def test_public_walkthrough_pass_and_reject_use_actual_run_and_validate(tmp_path: Path) -> None:
    passed = _exercise(tmp_path, PASS_PROBLEM, EXIT_PASS)
    rejected = _exercise(tmp_path, REJECT_PROBLEM, EXIT_REJECT)

    assert passed["decision"] == "PASS"
    assert passed["discovery_job"]["synthesis"]["expression"] == "x**3 - 2*x + 5"
    assert passed["discovery_job"]["validation"]["checked_rows"] == 2
    assert passed["orchestration"]["counts"]["hard_gate_eligible"] == 1
    assert passed["orchestration"]["counts"]["pareto_fronts"] == 1

    assert rejected["decision"] == "REJECT"
    witness = rejected["discovery_job"]["validation"]["counterexample"]
    assert witness == {
        "expected": {"denominator": 1, "numerator": 62},
        "observed": {"denominator": 1, "numerator": 61},
        "point": {"denominator": 1, "numerator": 4},
        "reason": "exact_heldout_mismatch",
        "residual": {"denominator": 1, "numerator": -1},
        "row_index": 1,
    }
    assert rejected["orchestration"]["counts"]["hard_gate_eligible"] == 0
    assert rejected["orchestration"]["counts"]["pareto_fronts"] == 0
    statuses = [
        row["status"] for row in rejected["orchestration"]["evaluations"][0]["gate_outcomes"]
    ]
    assert statuses == ["pass", "reject"]


def test_walkthrough_inputs_are_closed_related_and_docs_bind_public_contract() -> None:
    passed = json.loads(PASS_PROBLEM.read_text(encoding="utf-8"))
    rejected = json.loads(REJECT_PROBLEM.read_text(encoding="utf-8"))
    assert (
        set(passed)
        == set(rejected)
        == {
            "constraints",
            "job_id",
            "limits",
            "proof",
            "schema_version",
            "solver",
            "validation",
            "variable",
            "variable_domain",
        }
    )
    comparable_pass = {
        key: value for key, value in passed.items() if key not in {"job_id", "validation"}
    }
    comparable_reject = {
        key: value for key, value in rejected.items() if key not in {"job_id", "validation"}
    }
    assert comparable_pass == comparable_reject
    assert passed["validation"]["rows"][0] == rejected["validation"]["rows"][0]
    assert passed["validation"]["rows"][1]["value"]["numerator"] == 61
    assert rejected["validation"]["rows"][1]["value"]["numerator"] == 62
    assert canonical_sha256(passed) != canonical_sha256(rejected)

    documentation = WALKTHROUGH.read_text(encoding="utf-8")
    for marker in (
        "pass-exact-polynomial.json",
        "reject-heldout-counterexample.json",
        "hard_structure",
        "hard_validation",
        "exact Pareto",
        "SCHEMA_ERROR",
        "OPERATIONAL_ERROR",
        "does not establish a scientific law or novelty",
    ):
        assert marker in documentation
