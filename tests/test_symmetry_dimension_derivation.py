from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import symmetry_dimension_derivation as D
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _receipt() -> dict[str, object]:
    return D.build_receipt(ROOT)


def test_cross_domain_controls_recover_exact_expected_invariants() -> None:
    receipt = _receipt()
    assert receipt["summary"] == {
        "controls_passed": 4,
        "dimension_mutations_rejected": 4,
        "invariant_coordinates": 4,
        "status": "PASS_SYMMETRY_DIMENSION_FORCED_DERIVATION",
        "symmetry_mutations_rejected": 4,
    }
    observed = {
        result["problem_id"]: result["invariant"]["expression"]
        for result in receipt["results"]
    }
    assert observed == {
        "control.simple-pendulum-scaling": "period**2*gravity/length",
        "control.kepler-similarity": (
            "period**2*central_mass*gravitational_constant/semi_major_axis**3"
        ),
        "control.diffusion-similarity": "length_scale**2/(time_scale*diffusivity)",
        "control.reynolds-similarity": "density*speed*length_scale/dynamic_viscosity",
    }


def test_independent_exact_evaluators_agree_on_rank_and_nullspace() -> None:
    for result in _receipt()["results"]:
        evaluators = result["independent_evaluators"]
        assert evaluators["agreement"] is True
        assert evaluators["fraction_gaussian_elimination_rank"] == evaluators["sympy_rank"]
        assert evaluators["sympy_exact_nullspace_basis"] == [
            result["invariant"]["exponents"]
        ]
        assert result["search"]["nullity"] == 1


def test_dimension_exponent_mutations_are_rejected_exactly() -> None:
    for result in _receipt()["results"]:
        mutation = result["mutations"]["dimension_exponent_offset"]
        assert mutation["rejected"] is True
        assert any(mutation["row_residuals"])


def test_dropping_nuisance_symmetry_opens_a_spurious_coordinate() -> None:
    for result in _receipt()["results"]:
        mutation = result["mutations"]["drop_nuisance_symmetry"]
        assert mutation == {
            "baseline_nullity": 1,
            "mutated_nullity": 2,
            "nuisance_coordinate_admitted": True,
            "nuisance_exponents": [0] * (len(result["variables"]) - 1) + [1],
            "rejected": True,
        }


def test_forced_form_preserves_free_function_and_novelty_boundary() -> None:
    receipt = _receipt()
    assert not any(receipt["claims"].values())
    for result in receipt["results"]:
        assert result["forced_form"]["free_function_arity"] == 1
        assert result["forced_form"]["free_function_determined"] is False
        assert result["creative_brief"]["llm_origin_assessment_labels"] == [
            "known_rewrite",
            "cross_domain_synthesis",
            "proposed_new_construction",
            "uncertain",
        ]
        assert "neither empirical truth nor literature novelty" in result["creative_brief"][
            "novelty_caution"
        ]


def test_expected_vector_mutation_fails_closed() -> None:
    config = D.load_config(ROOT)
    changed = copy.deepcopy(config)
    changed["problems"][0]["expected_control_exponents"][0] += 1
    with pytest.raises(D.SymmetryDimensionError, match="nullspace"):
        D.evaluate_problem(changed["problems"][0], changed["search_policy"])


def test_claim_promotion_fails_even_after_resealing() -> None:
    receipt = _receipt()
    changed = copy.deepcopy(receipt)
    changed["claims"]["specific_law_discovered"] = True
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(D.SymmetryDimensionError, match="claim boundary"):
        D.validate_receipt(changed, ROOT)


def test_source_binding_mutation_fails_even_after_resealing() -> None:
    receipt = _receipt()
    changed = copy.deepcopy(receipt)
    changed["source_bindings"]["source"]["normalized_sha256"] = "0" * 64
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(D.SymmetryDimensionError, match="no longer reproduces"):
        D.validate_receipt(changed, ROOT)


def test_stored_receipt_validates_against_current_sources() -> None:
    receipt = json.loads((ROOT / D.OUTPUT_PATH).read_text(encoding="utf-8"))
    assert D.validate_receipt(receipt, ROOT) == receipt
