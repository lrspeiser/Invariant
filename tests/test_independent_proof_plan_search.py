from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import independent_proof_plan_search as P
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _receipt() -> dict[str, object]:
    return P.run_proof_plan_search(ROOT)


def test_six_independent_routes_close_and_their_mutations_stay_open() -> None:
    receipt = _receipt()
    P.validate_proof_plan_search(receipt, ROOT)
    assert receipt["summary"] == {
        "mechanisms": list(P.MECHANISMS),
        "mutation_controls_rejected": 6,
        "positive_routes_closed": 6,
        "status": "PASS_INDEPENDENT_PROOF_PLAN_SEARCH",
        "total_routes": 6,
    }
    assert all(row["positive_control"]["closed"] for row in receipt["routes"])
    assert all(not row["mutation_control"]["closed"] for row in receipt["routes"])


def test_routes_search_induction_invariants_bijections_descent_transforms_and_contradiction() -> None:
    receipt = _receipt()
    by_mechanism = {row["mechanism"]: row for row in receipt["routes"]}
    assert set(by_mechanism) == set(P.MECHANISMS)
    induction = by_mechanism["induction"]["positive_control"]
    assert induction["induction_variables"] == ["n"]
    assert "strengthened_invariant" in induction["facts"]
    assert by_mechanism["bijection_or_involution"]["positive_control"]["representation"] == (
        "combinatorial_objects"
    )
    assert by_mechanism["transform_and_extract"]["positive_control"]["representation"] == "native"


def test_rank_uses_search_metrics_and_is_not_persuasive_prose() -> None:
    templates = P.plan_templates(_receipt())
    assert [template["rank"] for template in templates] == list(range(1, 7))
    keys = [
        (
            -template["metrics"]["falsification_power"],
            template["metrics"]["premise_count"],
            template["metrics"]["proof_debt"],
            template["metrics"]["cost"],
            template["route_id"],
        )
        for template in templates
    ]
    assert keys == sorted(keys)


def test_capability_inference_only_marks_structural_applicability() -> None:
    capabilities = P.infer_candidate_capabilities(
        {
            "family": "partition recurrence",
            "falsifiers": ["boundary"],
            "invariants": ["parity"],
            "representation": "generating_function",
            "source_idea_domains": ["combinatorics"],
        }
    )
    assert set(capabilities) == {
        "combinatorial_structure",
        "declared_invariant",
        "discrete_domain",
        "falsifiable_boundary",
        "transformable_representation",
        "well_founded_order",
    }


def test_resealed_receipt_cannot_turn_an_abstract_route_into_a_candidate_proof() -> None:
    changed = copy.deepcopy(_receipt())
    changed["claims"]["closed_abstract_route_establishes_candidate_theorem"] = True
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(P.IndependentProofPlanSearchError, match="claim boundary"):
        P.validate_proof_plan_search(changed)


def test_stored_proof_plan_receipt_exactly_replays_current_sources() -> None:
    receipt = json.loads((ROOT / P.OUTPUT_PATH).read_text(encoding="utf-8"))
    P.validate_proof_plan_search(receipt, ROOT)
