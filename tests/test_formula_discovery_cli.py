from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.formula_discovery_cli import (
    EXIT_BLOCK,
    EXIT_OPERATIONAL_ERROR,
    EXIT_PASS,
    EXIT_REJECT,
    EXIT_SCHEMA_ERROR,
    PUBLIC_RESULT_SCHEMA,
    main,
    render_formula_discovery_markdown,
    run_formula_discovery_files,
    validate_formula_discovery_files,
)
from sigma_theory_compiler.formula_discovery_job import PROBLEM_SCHEMA
from sigma_theory_compiler.sigma_core import canonical_sha256


def _q(value: int) -> dict[str, int]:
    return {"numerator": value, "denominator": 1}


def _problem(job_id: str = "public.example") -> dict[str, object]:
    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": job_id,
        "variable": "x",
        "variable_domain": "rational",
        "solver": {"kind": "exact_linear_basis_v1", "basis": ["1", "x"]},
        "constraints": {
            "kind": "evaluations",
            "rows": [
                {"point": _q(0), "value": _q(1)},
                {"point": _q(1), "value": _q(3)},
            ],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _q(4), "value": _q(9)}],
        },
        "proof": {"kind": "none"},
        "limits": {
            "max_basis_terms": 4,
            "max_constraint_rows": 8,
            "max_expression_nodes": 32,
            "max_integer_bits": 64,
            "max_validation_rows": 4,
        },
    }


def _paths(tmp_path: Path, problem: dict[str, object]):
    problem_path = tmp_path / "problem.json"
    result_path = tmp_path / "result.json"
    report_path = tmp_path / "report.md"
    problem_path.write_text(json.dumps(problem), encoding="utf-8")
    return problem_path, result_path, report_path


def test_pass_run_writes_new_sealed_outputs_and_validate_is_read_only(tmp_path: Path) -> None:
    paths = _paths(tmp_path, _problem())
    result = run_formula_discovery_files(*paths)
    assert result["schema_version"] == PUBLIC_RESULT_SCHEMA
    assert result["decision"] == "PASS"
    assert result["orchestration"]["counts"]["hard_gate_eligible"] == 1
    assert paths[1].is_file() and paths[2].is_file()
    assert json.loads(paths[1].read_text(encoding="utf-8")) == result
    assert paths[2].read_text(encoding="utf-8") == render_formula_discovery_markdown(result)
    before = (paths[1].read_bytes(), paths[2].read_bytes())
    assert validate_formula_discovery_files(*paths) == result
    assert (paths[1].read_bytes(), paths[2].read_bytes()) == before


def test_cli_exit_codes_distinguish_pass_reject_block_and_schema_error(tmp_path: Path) -> None:
    cases = []
    cases.append((_problem("case.pass"), EXIT_PASS, "PASS"))
    rejected = _problem("case.reject")
    rejected["validation"]["rows"][0]["value"] = _q(99)
    cases.append((rejected, EXIT_REJECT, "REJECT"))
    blocked = _problem("case.block")
    blocked["constraints"]["rows"] = blocked["constraints"]["rows"][:1]
    cases.append((blocked, EXIT_BLOCK, "BLOCK"))
    malformed = _problem("case.schema")
    malformed["unknown"] = True
    cases.append((malformed, EXIT_SCHEMA_ERROR, "SCHEMA_ERROR"))

    for index, (problem, exit_code, decision) in enumerate(cases):
        directory = tmp_path / str(index)
        directory.mkdir()
        problem_path, result_path, report_path = _paths(directory, problem)
        assert (
            main(
                (
                    "run",
                    "--problem",
                    str(problem_path),
                    "--result",
                    str(result_path),
                    "--report",
                    str(report_path),
                )
            )
            == exit_code
        )
        stored = json.loads(result_path.read_text(encoding="utf-8"))
        assert stored["decision"] == decision
        if decision in {"BLOCK", "SCHEMA_ERROR"}:
            assert stored["orchestration"] is None
        assert (
            main(
                (
                    "validate",
                    "--problem",
                    str(problem_path),
                    "--result",
                    str(result_path),
                    "--report",
                    str(report_path),
                )
            )
            == exit_code
        )


def test_reject_report_documents_hard_gate_and_exact_counterexample(tmp_path: Path) -> None:
    problem = _problem("report.reject")
    problem["validation"]["rows"][0]["value"] = _q(10)
    paths = _paths(tmp_path, problem)
    result = run_formula_discovery_files(*paths)
    report = paths[2].read_text(encoding="utf-8")
    assert result["decision"] == "REJECT"
    assert result["orchestration"]["pareto"]["counts"]["hard_gate_eligible"] == 0
    assert "## Exact counterexample" in report
    assert "`hard_validation`: **REJECT**" in report
    assert "Hard-gate eligible candidates: 0" in report


def test_outputs_are_immutable_and_existing_files_are_never_changed(tmp_path: Path) -> None:
    paths = _paths(tmp_path, _problem())
    paths[1].write_bytes(b"existing-result")
    assert (
        main(
            (
                "run",
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
    assert paths[1].read_bytes() == b"existing-result"
    assert not paths[2].exists()

    paths[1].unlink()
    paths[2].write_bytes(b"existing-report")
    assert (
        main(
            (
                "run",
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
    assert not paths[1].exists()
    assert paths[2].read_bytes() == b"existing-report"


def test_resealed_result_or_markdown_tamper_fails_validation(tmp_path: Path) -> None:
    paths = _paths(tmp_path, _problem())
    result = run_formula_discovery_files(*paths)
    tampered = copy.deepcopy(result)
    tampered["claims"]["promotion_authorized"] = True
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

    paths[1].write_text(json.dumps(result), encoding="utf-8")
    paths[2].write_text(paths[2].read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
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


def test_strict_json_duplicate_float_and_non_object_inputs_are_schema_errors(
    tmp_path: Path,
) -> None:
    payloads = (
        '{"job_id":"a","job_id":"b"}',
        '{"value": 1.5}',
        "[]",
    )
    for index, payload in enumerate(payloads):
        problem_path = tmp_path / f"problem-{index}.json"
        result_path = tmp_path / f"result-{index}.json"
        report_path = tmp_path / f"report-{index}.md"
        problem_path.write_text(payload, encoding="utf-8")
        assert (
            main(
                (
                    "run",
                    "--problem",
                    str(problem_path),
                    "--result",
                    str(result_path),
                    "--report",
                    str(report_path),
                )
            )
            == EXIT_SCHEMA_ERROR
        )
        assert not result_path.exists()
        assert not report_path.exists()


def test_public_surface_has_no_runtime_network_database_or_secret_dependencies() -> None:
    source = Path("src/sigma_theory_compiler/formula_discovery_cli.py").read_text(encoding="utf-8")
    forbidden = ("requests", "urllib", "socket", "subprocess", "sqlite", "token", "password")
    assert all(f"import {name}" not in source for name in forbidden)
