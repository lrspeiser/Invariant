from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.formula_discovery_job import (
    PROBLEM_SCHEMA,
    run_formula_discovery_job,
)
from sigma_theory_compiler.formula_discovery_lean_translation import (
    TRANSLATION_SCHEMA,
    FormulaDiscoveryLeanTranslationError,
    translate_formula_discovery_pass,
    validate_formula_discovery_lean_translation,
    write_lean_source,
)
from sigma_theory_compiler.math_lean_adapter import LeanAdapterConfig, run_lean_adapter
from sigma_theory_compiler.sigma_core import canonical_sha256


def _q(value: int, denominator: int = 1) -> dict[str, int]:
    return {"numerator": value, "denominator": denominator}


def _limits() -> dict[str, int]:
    return {
        "max_basis_terms": 8,
        "max_constraint_rows": 32,
        "max_expression_nodes": 128,
        "max_integer_bits": 128,
        "max_validation_rows": 16,
    }


def _integer_polynomial_problem() -> dict[str, object]:
    coefficients = (-11, 4, 0, 3)

    def target(x: int) -> int:
        return sum(value * x**power for power, value in enumerate(coefficients))

    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "translation.polynomial.alpha",
        "variable": "x",
        "variable_domain": "integer",
        "solver": {
            "kind": "exact_linear_basis_v1",
            "basis": ["1", "x", "x**2", "x**3"],
        },
        "constraints": {
            "kind": "evaluations",
            "rows": [{"point": _q(x), "value": _q(target(x))} for x in range(4)],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _q(x), "value": _q(target(x))} for x in (-4, 7)],
        },
        "proof": {"kind": "none"},
        "limits": _limits(),
    }


def _nat_recurrence_problem() -> dict[str, object]:
    coefficients = (5, 2, 1)

    def target(n: int) -> int:
        return sum(value * n**power for power, value in enumerate(coefficients))

    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "translation.recurrence.beta",
        "variable": "n",
        "variable_domain": "integer",
        "solver": {
            "kind": "exact_linear_basis_v1",
            "basis": ["1", "n", "n**2"],
        },
        "constraints": {
            "kind": "first_order_recurrence",
            "sequence": "values",
            "base": {"index": 0, "value": _q(5)},
            "successor_increment": "2*n + 3",
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _q(n), "value": _q(target(n))} for n in (1, 4, 9)],
        },
        "proof": {"kind": "induction"},
        "limits": _limits(),
    }


def _pass(problem: dict[str, object]) -> dict:
    result = run_formula_discovery_job(problem)
    assert result["decision"] == "PASS"
    return result


def _reseal_result(result: dict) -> None:
    result["content_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )


def _reseal_translation(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def _lean_environment() -> dict[str, str] | None:
    environment = dict(os.environ)
    if environment.get("INVARIANT_LEAN_EXECUTABLE") or shutil.which("lean"):
        return environment
    candidate = (
        Path.home()
        / ".cache"
        / "invariant"
        / "lean"
        / "v4.33.0"
        / "lean-4.33.0-windows"
        / "bin"
        / "lean.exe"
    )
    if not candidate.is_file():
        return None
    environment["INVARIANT_LEAN_EXECUTABLE"] = str(candidate)
    return environment


def test_integer_polynomial_pass_translates_without_embedded_benchmark_logic() -> None:
    problem = _integer_polynomial_problem()
    result = _pass(problem)
    translation = translate_formula_discovery_pass(result, problem)
    validate_formula_discovery_lean_translation(translation, result, problem)
    assert translation["schema_version"] == TRANSLATION_SCHEMA
    assert translation["translation_kind"] == "integer_polynomial_coefficient_identity"
    assert translation["proof_certificate_content_sha256"] is None
    assert translation["counts"]["candidate_polynomial_degree"] == 3
    assert "[-11, 4, 0, 3]" in translation["source"]
    assert "rfl" in translation["source"]
    assert "sorry" not in translation["source"].lower()
    assert "axiom " not in translation["source"].lower()
    assert (
        translation["source_sha256"]
        == __import__("hashlib").sha256(translation["source"].encode()).hexdigest()
    )


def test_nat_recurrence_pass_translates_candidate_and_induction_certificate() -> None:
    problem = _nat_recurrence_problem()
    result = _pass(problem)
    translation = translate_formula_discovery_pass(result, problem)
    validate_formula_discovery_lean_translation(translation, result, problem)
    assert translation["translation_kind"] == "first_order_nat_recurrence_closed_form"
    assert (
        translation["proof_certificate_content_sha256"]
        == result["proof_certificate"]["content_sha256"]
    )
    source = translation["source"]
    assert "n ^ 2 + 2 * n + 5" in source
    assert "2 * n + 3" in source
    assert "induction n with" in source
    assert "Lean.Parser.Tactic.omega" in source


@pytest.mark.parametrize("problem_factory", [_integer_polynomial_problem, _nat_recurrence_problem])
def test_translation_is_deterministic_and_manifest_is_closed(problem_factory) -> None:
    problem = problem_factory()
    result = _pass(problem)
    first = translate_formula_discovery_pass(result, problem)
    second = translate_formula_discovery_pass(result, problem)
    assert first == second
    manifest = first["premise_manifest"]
    assert manifest["target"] == first["target"]
    assert manifest["equivalent_targets"] == []
    assert manifest["forbidden_premises"] == ["Classical.choice", "False.elim"]
    assert manifest["forbidden_prefixes"] == ["KnownAnswer.", "Unsafe."]
    assert first["counts"]["allowed_premises"] == len(manifest["allowed_premises"])


def test_optional_generated_sources_execute_in_real_lean(tmp_path: Path) -> None:
    environment = _lean_environment()
    if environment is None:
        pytest.skip("Lean unavailable; deterministic source checks remain active")
    executable = environment.get("INVARIANT_LEAN_EXECUTABLE") or shutil.which("lean")
    assert executable is not None
    for problem in (_integer_polynomial_problem(), _nat_recurrence_problem()):
        result = _pass(problem)
        translation = translate_formula_discovery_pass(result, problem)
        source_path = write_lean_source(
            translation, tmp_path / f"{translation['translation_kind']}.lean"
        )
        manifest = translation["premise_manifest"]
        config = LeanAdapterConfig(
            target=translation["target"],
            allowed_premises=manifest["allowed_premises"],
            forbidden_premises=manifest["forbidden_premises"],
            forbidden_prefixes=[
                value.removesuffix(".") for value in manifest["forbidden_prefixes"]
            ],
            executable=Path(executable),
            timeout_seconds=30,
        )
        adapter = run_lean_adapter(config, source_path, environment={})
        assert adapter["decision"] == "pass_lean_checked_closed_premise", json.dumps(
            adapter, indent=2
        )
        assert adapter["source_sha256"] == translation["source_sha256"]


def test_nonpass_and_rational_domain_results_reject_before_source() -> None:
    problem = _integer_polynomial_problem()
    problem["validation"]["rows"][0]["value"] = _q(999)
    result = run_formula_discovery_job(problem)
    assert result["decision"] == "REJECT"
    with pytest.raises(FormulaDiscoveryLeanTranslationError, match="only validated PASS"):
        translate_formula_discovery_pass(result, problem)

    rational = _integer_polynomial_problem()
    rational["variable_domain"] = "rational"
    rational_result = _pass(rational)
    with pytest.raises(FormulaDiscoveryLeanTranslationError, match="rational-domain"):
        translate_formula_discovery_pass(rational_result, rational)


def test_rational_coefficients_and_unsafe_nat_constructs_reject() -> None:
    rational_coefficient = _integer_polynomial_problem()
    rational_coefficient["constraints"]["rows"] = [
        {"point": _q(x), "value": _q(2 * x + 1, 2)} for x in range(2)
    ]
    rational_coefficient["solver"]["basis"] = ["1", "x"]
    rational_coefficient["validation"]["rows"] = [
        {"point": _q(x), "value": _q(2 * x + 1, 2)} for x in (3, 5)
    ]
    rational_result = _pass(rational_coefficient)
    with pytest.raises(FormulaDiscoveryLeanTranslationError, match="integer-coefficient"):
        translate_formula_discovery_pass(rational_result, rational_coefficient)

    negative_nat = _nat_recurrence_problem()
    negative_nat["constraints"]["base"]["value"] = _q(-5)
    negative_nat["validation"]["rows"] = [
        {"point": _q(n), "value": _q(n * n + 2 * n - 5)} for n in (1, 4, 9)
    ]
    negative_result = _pass(negative_nat)
    with pytest.raises(FormulaDiscoveryLeanTranslationError, match="nonnegative integral"):
        translate_formula_discovery_pass(negative_result, negative_nat)


def test_missing_induction_certificate_and_candidate_tamper_fail_closed() -> None:
    problem = _nat_recurrence_problem()
    result = _pass(problem)
    no_proof = copy.deepcopy(result)
    no_proof["proof_certificate"] = None
    no_proof["counts"]["proof_certificates"] = 0
    _reseal_result(no_proof)
    with pytest.raises(FormulaDiscoveryLeanTranslationError, match="did not validate exactly"):
        translate_formula_discovery_pass(no_proof, problem)

    polynomial = _integer_polynomial_problem()
    polynomial_result = _pass(polynomial)
    tampered = copy.deepcopy(polynomial_result)
    tampered["candidate"]["representation"]["expression"] = "0"
    _reseal_result(tampered)
    with pytest.raises(FormulaDiscoveryLeanTranslationError, match="did not validate exactly"):
        translate_formula_discovery_pass(tampered, polynomial)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("source", value["source"] + "\n-- tamper\n"),
        lambda value: value["premise_manifest"]["allowed_premises"].append("False.elim"),
        lambda value: value["claims"].__setitem__("novelty_established", True),
        lambda value: value.__setitem__("unknown", True),
    ],
)
def test_resealed_translation_tampers_fail_closed(mutator) -> None:
    problem = _integer_polynomial_problem()
    result = _pass(problem)
    translation = translate_formula_discovery_pass(result, problem)
    tampered = copy.deepcopy(translation)
    mutator(tampered)
    tampered["premise_manifest"]["content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered["premise_manifest"].items()
            if key != "content_sha256"
        }
    )
    if isinstance(tampered.get("source"), str):
        tampered["source_sha256"] = (
            __import__("hashlib").sha256(tampered["source"].encode()).hexdigest()
        )
    _reseal_translation(tampered)
    with pytest.raises(FormulaDiscoveryLeanTranslationError):
        validate_formula_discovery_lean_translation(tampered, result, problem)
