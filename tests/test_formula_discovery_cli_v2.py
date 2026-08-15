from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.formula_discovery_cli import (
    EXIT_OPERATIONAL_ERROR,
    EXIT_PASS,
    PUBLIC_RESULT_SCHEMA_V2,
    main,
    render_formula_discovery_markdown,
    run_formula_discovery_files,
    validate_formula_discovery_files,
)
from sigma_theory_compiler.formula_discovery_job_v2 import PROBLEM_SCHEMA
from sigma_theory_compiler.sigma_core import canonical_sha256


def _q(value: int) -> dict[str, int]:
    return {"numerator": value, "denominator": 1}


def _problem() -> dict[str, object]:
    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "public.v2.multivariate",
        "variables": [
            {"name": "x", "domain": "rational"},
            {"name": "y", "domain": "rational"},
        ],
        "premises": [],
        "solver": {
            "kind": "exact_polynomial_basis_v2",
            "basis": ["1", "x", "y", "x*y"],
        },
        "constraints": {
            "kind": "evaluations",
            "rows": [
                {"point": {"x": _q(0), "y": _q(0)}, "value": _q(1)},
                {"point": {"x": _q(1), "y": _q(0)}, "value": _q(3)},
                {"point": {"x": _q(0), "y": _q(1)}, "value": _q(4)},
                {"point": {"x": _q(1), "y": _q(1)}, "value": _q(10)},
            ],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [
                {"point": {"x": _q(2), "y": _q(3)}, "value": _q(38)},
                {"point": {"x": _q(-1), "y": _q(4)}, "value": _q(-5)},
            ],
        },
        "proof": {"kind": "none"},
        "limits": {
            "max_basis_terms": 8,
            "max_constraint_rows": 16,
            "max_expression_nodes": 64,
            "max_integer_bits": 64,
            "max_parameter_combinations": 32,
            "max_parameters": 3,
            "max_recurrence_order": 3,
            "max_validation_rows": 8,
            "max_variables": 3,
        },
    }


def _paths(tmp_path: Path, problem: dict[str, object]) -> tuple[Path, Path, Path]:
    problem_path = tmp_path / "problem-v2.json"
    result_path = tmp_path / "result-v2.json"
    report_path = tmp_path / "report-v2.md"
    problem_path.write_text(json.dumps(problem), encoding="utf-8")
    return problem_path, result_path, report_path


def test_v2_public_file_boundary_runs_orchestration_and_exact_replay(tmp_path: Path) -> None:
    paths = _paths(tmp_path, _problem())
    result = run_formula_discovery_files(*paths)
    assert result["schema_version"] == PUBLIC_RESULT_SCHEMA_V2
    assert result["decision"] == "PASS"
    assert result["discovery_job"]["class_id"] == "multivariate_polynomial"
    assert result["orchestration"]["counts"]["hard_gate_eligible"] == 1
    assert result["orchestration"]["pareto"]["counts"]["pareto_fronts"] == 1
    assert validate_formula_discovery_files(*paths) == result
    report = paths[2].read_text(encoding="utf-8")
    assert report == render_formula_discovery_markdown(result)
    assert "Discovery class: `multivariate_polynomial`" in report
    assert "Hard-gate eligible candidates: 1" in report


def test_v2_cli_run_and_validate_return_pass(tmp_path: Path) -> None:
    paths = _paths(tmp_path, _problem())
    arguments = (
        "--problem",
        str(paths[0]),
        "--result",
        str(paths[1]),
        "--report",
        str(paths[2]),
    )
    assert main(("run", *arguments)) == EXIT_PASS
    assert main(("validate", *arguments)) == EXIT_PASS


def test_v2_resealed_nested_tamper_fails_public_replay(tmp_path: Path) -> None:
    paths = _paths(tmp_path, _problem())
    result = run_formula_discovery_files(*paths)
    tampered = copy.deepcopy(result)
    tampered["discovery_job"]["class_id"] = "rational_function_with_domain"
    tampered["discovery_job"]["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered["discovery_job"].items() if key != "content_sha256"}
    )
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    paths[1].write_text(json.dumps(tampered), encoding="utf-8")
    assert (
        main(
            (
                "validate",
                "--problem",
                str(paths[0]),
                "--result",
                str(paths[1]),
                "--report",
                str(paths[2]),
            )
        )
        == EXIT_OPERATIONAL_ERROR
    )
