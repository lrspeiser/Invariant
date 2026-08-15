from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.native_newton_blind_polynomial_tournament import (
    CLAIMS,
    CONFIG_PATH,
    OUTPUT_PATH,
    NativeNewtonTournamentError,
    build_campaign,
    validate_checked_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checked() -> dict:
    rebuilt = build_campaign(CONFIG, ROOT)
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_checked_campaign(value, CONFIG, ROOT)
    assert value == rebuilt
    return value


def _result(value: dict, family: str) -> dict:
    return next(row for row in value["candidate_results"] if row["candidate"]["family"] == family)


def _reseal(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_native_newton_constructs_exact_formula_without_generic_solver(checked: dict) -> None:
    native = _result(checked, "symbolic_newton")
    candidate = native["candidate"]
    assert candidate["method"] == (
        "native_forward_difference_to_newton_basis_to_monomial_coefficients"
    )
    assert candidate["forward_difference_first_column"] == [11, -2, -2, -24, -24]
    assert candidate["seed_coefficients_constant_first"] == [11]
    assert candidate["coefficients_constant_first"] == [11, -3, 0, 2, -1]
    assert candidate["changed_from_seed"] is True
    assert candidate["public_rows_used"] == 5
    assert candidate["target_fields_read"] == []
    assert candidate["generic_exact_solver_used"] is False
    assert native["status"] == "PASS"
    certificate = native["proof_certificate"]
    assert certificate["decision"] == "proved_exact_integer_polynomial_identity"
    assert certificate["cleared_coefficient_residuals"] == [0, 0, 0, 0, 0]


def test_blind_chronology_freezes_all_candidates_before_one_unseal(checked: dict) -> None:
    assert checked["phase_ledger"] == {
        "generation_events_before_unseal": 3,
        "pre_unseal_target_access_count": 0,
        "candidate_set_frozen_before_unseal": True,
        "frozen_candidates_sha256": checked["phase_ledger"]["frozen_candidates_sha256"],
        "atomic_unseal_batches": 1,
        "target_records_unsealed": 1,
        "post_unseal_generation_count": 0,
        "post_unseal_tuning_events": 0,
    }
    assert len(checked["phase_ledger"]["frozen_candidates_sha256"]) == 64
    assert checked["world"]["target_commitment_sha256"] == canonical_sha256(
        checked["unsealed_target"]
    )


def test_fixed_baselines_reject_with_exact_counterexamples(checked: dict) -> None:
    grammar = _result(checked, "grammar")
    egraph = _result(checked, "egraph")
    assert grammar["candidate"]["coefficients_constant_first"] == [11]
    assert grammar["status"] == "REJECT"
    assert grammar["counterexample"] == {
        "point": 1,
        "candidate_value": 11,
        "target_value": 9,
        "residual": 2,
    }
    assert egraph["candidate"]["coefficients_constant_first"] == [11, -2]
    assert egraph["status"] == "REJECT"
    assert egraph["counterexample"] == {
        "point": 2,
        "candidate_value": 7,
        "target_value": 5,
        "residual": 2,
    }
    assert all(row["proof_certificate"] is None for row in (grammar, egraph))


def test_counts_and_claims_are_narrow(checked: dict) -> None:
    assert checked["decision"] == "pass_native_newton_one_of_three_exact_two_rejected"
    assert checked["counts"] == {
        "worlds": 1,
        "generator_families": 3,
        "candidates": 3,
        "candidate_passes": 1,
        "candidate_rejects": 2,
        "candidate_blocks": 0,
        "exact_identity_certificates": 1,
        "exact_counterexamples": 2,
        "native_formula_constructions": 1,
        "generic_exact_solver_invocations": 0,
        "floating_point_operations": 0,
    }
    assert checked["claims"] == CLAIMS
    assert CLAIMS["native_non_bayesian_generator_constructed_formula"] is True
    assert CLAIMS["generic_exact_solver_used"] is False
    assert CLAIMS["general_formula_discovery_established"] is False
    assert CLAIMS["novelty_established"] is False
    assert CLAIMS["promotion_authorized"] is False


def test_source_has_no_generic_formula_discovery_solver_dependency() -> None:
    source = (
        ROOT / "src/sigma_theory_compiler/native_newton_blind_polynomial_tournament.py"
    ).read_text(encoding="utf-8")
    assert "formula_discovery_job" not in source
    assert "exact_linear" not in source
    assert "linsolve" not in source
    assert "sympy" not in source


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["phase_ledger"].__setitem__("pre_unseal_target_access_count", 1),
        lambda value: value["counts"].__setitem__("generic_exact_solver_invocations", 1),
        lambda value: value["candidate_results"][0]["candidate"].__setitem__(
            "coefficients_constant_first", [11]
        ),
        lambda value: value["candidate_results"][1]["counterexample"].__setitem__("residual", 0),
        lambda value: value["claims"].__setitem__("general_formula_discovery_established", True),
        lambda value: value.__setitem__("unknown_top_level_key", True),
    ],
)
def test_resealed_tampers_fail_closed(checked: dict, mutator) -> None:
    tampered = copy.deepcopy(checked)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(NativeNewtonTournamentError):
        validate_checked_campaign(tampered, CONFIG, ROOT)


def test_malformed_preregistration_fails_closed() -> None:
    tampered = copy.deepcopy(CONFIG)
    tampered["policies"]["pre_unseal_target_access_count"] = 1
    with pytest.raises(NativeNewtonTournamentError, match="config changed"):
        build_campaign(tampered, ROOT)
