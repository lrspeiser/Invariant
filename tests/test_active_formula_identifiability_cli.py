from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.active_formula_identifiability import (
    build_query_answer,
    target_commitment,
)
from sigma_theory_compiler.active_formula_identifiability_cli import (
    EXIT_BLOCK,
    EXIT_OPERATIONAL_ERROR,
    EXIT_PASS,
    main,
    resume_files,
    start_files,
    validate_resume_files,
    validate_start_files,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

NONCE = "public-target-nonce-001"


def _q(value: int) -> dict[str, int]:
    return {"numerator": value, "denominator": 1}


def _problem() -> dict[str, object]:
    return {
        "schema_version": "sigma-active-formula-identifiability-problem-1.0",
        "session_id": "public.active.example",
        "variable": "n",
        "variable_domain": "integer",
        "hypotheses": [
            {"hypothesis_id": "formula.linear", "expression": "n"},
            {"hypothesis_id": "formula.square", "expression": "n**2"},
        ],
        "observations": [{"point": _q(0), "value": _q(0)}],
        "query_space": [_q(1), _q(2)],
        "query_budget": 2,
        "target_commitment": target_commitment("public.active.example", "formula.square", NONCE),
        "limits": {
            "max_expression_nodes": 32,
            "max_hypotheses": 4,
            "max_integer_bits": 32,
            "max_observations": 4,
            "max_query_space": 4,
        },
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_public_start_resume_and_both_read_only_validators(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.json"
    initial_path = tmp_path / "initial.json"
    initial_report = tmp_path / "initial.md"
    answer_path = tmp_path / "answer.json"
    result_path = tmp_path / "result.json"
    result_report = tmp_path / "result.md"
    problem = _problem()
    _write(problem_path, problem)

    initial = start_files(problem_path, initial_path, initial_report)
    assert initial["decision"] == "BLOCK"
    assert validate_start_files(problem_path, initial_path, initial_report) == initial
    answer = build_query_answer(
        problem,
        initial,
        target_hypothesis_id="formula.square",
        nonce=NONCE,
    )
    _write(answer_path, answer)
    result = resume_files(
        problem_path,
        initial_path,
        answer_path,
        result_path,
        result_report,
    )
    assert result["decision"] == "PASS"
    assert (
        validate_resume_files(
            problem_path,
            initial_path,
            answer_path,
            result_path,
            result_report,
        )
        == result
    )
    assert "Exact ambiguity witness" in initial_report.read_text(encoding="utf-8")
    assert "Identified expression: `n**2`" in result_report.read_text(encoding="utf-8")


def test_cli_exit_codes_initial_block_then_resumed_pass(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.json"
    initial_path = tmp_path / "initial.json"
    initial_report = tmp_path / "initial.md"
    answer_path = tmp_path / "answer.json"
    result_path = tmp_path / "result.json"
    result_report = tmp_path / "result.md"
    problem = _problem()
    _write(problem_path, problem)
    assert (
        main(
            (
                "start",
                "--problem",
                str(problem_path),
                "--result",
                str(initial_path),
                "--report",
                str(initial_report),
            )
        )
        == EXIT_BLOCK
    )
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    answer = build_query_answer(
        problem,
        initial,
        target_hypothesis_id="formula.square",
        nonce=NONCE,
    )
    _write(answer_path, answer)
    assert (
        main(
            (
                "resume",
                "--problem",
                str(problem_path),
                "--initial",
                str(initial_path),
                "--answer",
                str(answer_path),
                "--result",
                str(result_path),
                "--report",
                str(result_report),
            )
        )
        == EXIT_PASS
    )


def test_immutable_outputs_and_markdown_tamper_fail_closed(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.json"
    initial_path = tmp_path / "initial.json"
    initial_report = tmp_path / "initial.md"
    _write(problem_path, _problem())
    initial_path.write_bytes(b"existing")
    assert (
        main(
            (
                "start",
                "--problem",
                str(problem_path),
                "--result",
                str(initial_path),
                "--report",
                str(initial_report),
            )
        )
        == 30
    )
    assert initial_path.read_bytes() == b"existing"
    assert not initial_report.exists()

    initial_path.unlink()
    start_files(problem_path, initial_path, initial_report)
    initial_report.write_text(
        initial_report.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8"
    )
    assert (
        main(
            (
                "validate-start",
                "--problem",
                str(problem_path),
                "--result",
                str(initial_path),
                "--report",
                str(initial_report),
            )
        )
        == 30
    )


def test_resealed_answer_tamper_is_operationally_rejected_by_cli(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.json"
    initial_path = tmp_path / "initial.json"
    initial_report = tmp_path / "initial.md"
    answer_path = tmp_path / "answer.json"
    result_path = tmp_path / "result.json"
    result_report = tmp_path / "result.md"
    problem = _problem()
    _write(problem_path, problem)
    initial = start_files(problem_path, initial_path, initial_report)
    answer = build_query_answer(
        problem,
        initial,
        target_hypothesis_id="formula.square",
        nonce=NONCE,
    )
    answer["query_content_sha256"] = "0" * 64
    answer["content_sha256"] = canonical_sha256(
        {key: value for key, value in answer.items() if key != "content_sha256"}
    )
    _write(answer_path, answer)
    assert (
        main(
            (
                "resume",
                "--problem",
                str(problem_path),
                "--initial",
                str(initial_path),
                "--answer",
                str(answer_path),
                "--result",
                str(result_path),
                "--report",
                str(result_report),
            )
        )
        == EXIT_OPERATIONAL_ERROR
    )
    assert not result_path.exists()
    assert not result_report.exists()
