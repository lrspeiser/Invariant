from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from sigma_theory_compiler.egraph_candidate_generator import (
    REGISTERED_RULE_IDS,
    EGraphBoundaryError,
    SaturationLimits,
    extract_candidate_artifacts,
    saturate_expressions,
    validate_replay,
    validate_saturation_result,
)
from sigma_theory_compiler.math_expression_ir import add, literal, multiply, power, symbol
from sigma_theory_compiler.math_types import RATIONAL
from sigma_theory_compiler.sigma_core import (
    CandidateArtifact,
    DomainPackRef,
    ProvenanceRecord,
    SchemaViolation,
    canonical_sha256,
)


def _provenance(result_sha256: str) -> ProvenanceRecord:
    descriptor = {
        "pack_id": "egraph.exact-algebra",
        "pack_version": "1.0",
        "rewrite_rules": list(REGISTERED_RULE_IDS),
    }
    return ProvenanceRecord.create(
        DomainPackRef("egraph.exact-algebra", "1.0", canonical_sha256(descriptor)),
        {"saturation_result_sha256": result_sha256},
    )


@pytest.fixture(scope="module")
def saturated() -> dict[str, object]:
    x, y, z = symbol("x"), symbol("y"), symbol("z")
    factored = multiply(x, add(y, z))
    expanded = add(multiply(x, y), multiply(x, z))
    result = saturate_expressions(
        (factored, expanded),
        limits=SaturationLimits(maximum_nodes=256, maximum_iterations=8, maximum_work_units=50_000),
    )
    validate_saturation_result(result)
    return result


def test_registered_exact_saturation_is_deterministic_and_replayable(
    saturated: dict[str, object],
) -> None:
    x, y, z = symbol("x"), symbol("y"), symbol("z")
    seeds = (
        multiply(x, add(y, z)),
        add(multiply(x, y), multiply(x, z)),
    )
    limits = SaturationLimits(256, 8, 50_000)
    assert saturated == saturate_expressions(seeds, limits=limits)
    validate_replay(saturated, seeds, limits=limits)
    assert saturated["decision"] == "saturated_registered_rules_fixed_point"
    assert saturated["fixed_point_complete"] is True
    assert saturated["counts"] == {
        "seed_inputs": 2,
        "unique_seed_hashes": 2,
        "unique_expression_nodes": 21,
        "canonical_eclasses": 7,
        "iterations_completed": 5,
        "work_units_consumed": 4158,
        "direct_rewrite_merges": 14,
        "congruence_merges": 0,
        "deduplicated_rewrite_results": 74,
    }
    class_by_member = {
        member: row for row in saturated["eclasses"] for member in row["member_expression_sha256s"]
    }
    assert (
        len({class_by_member[seed]["eclass_id"] for seed in saturated["seed_expression_sha256s"]})
        == 1
    )


def test_canonical_eclasses_extract_smallest_member_and_preserve_lineage(
    saturated: dict[str, object],
) -> None:
    expression_hashes = {row["expression_sha256"] for row in saturated["expressions"]}
    partition = [
        member for row in saturated["eclasses"] for member in row["member_expression_sha256s"]
    ]
    assert len(partition) == len(set(partition))
    assert set(partition) == expression_hashes
    assert all(
        row["eclass_id"] == f"eclass-{min(row['member_expression_sha256s'])[:24]}"
        for row in saturated["eclasses"]
    )
    assert saturated["rewrite_lineage_root_sha256"] == canonical_sha256(
        saturated["rewrite_lineage"]
    )
    assert {event["rule_id"] for event in saturated["rewrite_lineage"]} <= set(
        REGISTERED_RULE_IDS
    ) | {"registered_congruence_inference"}
    assert all(
        event["sequence"] == sequence for sequence, event in enumerate(saturated["rewrite_lineage"])
    )


def test_sigma_core_candidates_are_content_deduplicated_and_scope_limited(
    saturated: dict[str, object],
) -> None:
    provenance = _provenance(saturated["content_sha256"])
    candidates = extract_candidate_artifacts(saturated, provenance)
    assert len(candidates) == 1
    candidate = candidates[0]
    candidate.validate()
    assert candidate.claims == ("registered_rewrite_equivalence",)
    assert candidate.representation["saturation_result_sha256"] == saturated["content_sha256"]
    assert candidate.representation["fixed_point_complete"] is True
    assert candidate.assumptions == (
        "Equivalence is limited to the content-hash-bound registered rewrite manifest.",
    )

    mutable = candidate.to_dict()
    mutable["representation"]["fixed_point_complete"] = False
    with pytest.raises(SchemaViolation, match="canonical identity changed"):
        CandidateArtifact.from_dict(mutable)


def test_non_equivalent_seeds_remain_separate_and_no_broad_claim_is_emitted() -> None:
    x, y = symbol("x"), symbol("y")
    result = saturate_expressions((add(x, y), multiply(x, y)))
    class_by_member = {
        member: row["eclass_id"]
        for row in result["eclasses"]
        for member in row["member_expression_sha256s"]
    }
    assert len({class_by_member[seed] for seed in result["seed_expression_sha256s"]}) == 2
    assert result["claims"] == {
        "equivalence_scope": "registered_exact_rewrites_and_congruence_only",
        "unregistered_equivalence_claimed": False,
        "novelty_claimed": False,
        "promotion_authorized": False,
        "time_based_termination_used": False,
    }


def test_unregistered_rules_and_unsafe_expression_domains_fail_closed() -> None:
    x = symbol("x")
    with pytest.raises(EGraphBoundaryError, match="unregistered or unproved rewrite"):
        saturate_expressions((x,), rule_ids=(*REGISTERED_RULE_IDS, "divide_by_symbol"))
    with pytest.raises(EGraphBoundaryError, match="exact integer and rational"):
        saturate_expressions((literal(0.5),))
    with pytest.raises(EGraphBoundaryError, match="outside the closed rewrite domain"):
        saturate_expressions((power(x, 2),))
    with pytest.raises(EGraphBoundaryError, match="separately proved typed rewrite"):
        saturate_expressions((symbol("typed_x", RATIONAL),))
    with pytest.raises(EGraphBoundaryError, match="positive integer"):
        SaturationLimits(maximum_nodes=10, maximum_iterations=1, maximum_work_units=0)


def test_node_iteration_and_work_caps_are_deterministic_not_time_based() -> None:
    x, y, z = symbol("x"), symbol("y"), symbol("z")
    seed = multiply(x, add(y, z))
    node_limited = saturate_expressions(
        (seed,),
        limits=SaturationLimits(maximum_nodes=5, maximum_iterations=10, maximum_work_units=500),
    )
    assert node_limited["decision"] == "bounded_node_cap"
    assert node_limited["fixed_point_complete"] is False
    work_limited = saturate_expressions(
        (seed,),
        limits=SaturationLimits(maximum_nodes=100, maximum_iterations=10, maximum_work_units=1),
    )
    assert work_limited["decision"] == "bounded_work_unit_cap"
    assert work_limited["counts"]["work_units_consumed"] == 1
    iteration_limited = saturate_expressions(
        (seed,),
        limits=SaturationLimits(maximum_nodes=100, maximum_iterations=1, maximum_work_units=1000),
    )
    assert iteration_limited["decision"] == "bounded_iteration_cap"
    assert all("second" not in key and "time" not in key for key in iteration_limited["limits"])


def test_exact_rational_constant_rules_and_content_hash_dedup() -> None:
    x = symbol("x")
    seed = add(literal(Fraction(1, 3)), literal(Fraction(2, 3)), multiply(literal(1), x))
    result = saturate_expressions((seed, seed))
    assert result["counts"]["seed_inputs"] == 2
    assert result["counts"]["unique_seed_hashes"] == 1
    candidates = extract_candidate_artifacts(result, _provenance(result["content_sha256"]))
    assert len(candidates) == 1


def test_resealed_lineage_and_extraction_tampering_are_rejected(
    saturated: dict[str, object],
) -> None:
    tampered = copy.deepcopy(saturated)
    event = tampered["rewrite_lineage"][0]
    event["rule_id"] = "multiply_zero"
    tampered["rewrite_lineage_root_sha256"] = canonical_sha256(tampered["rewrite_lineage"])
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(EGraphBoundaryError, match="not replayable"):
        validate_saturation_result(tampered)

    tampered = copy.deepcopy(saturated)
    row = tampered["eclasses"][0]
    row["extracted_expression_sha256"] = row["member_expression_sha256s"][-1]
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(EGraphBoundaryError, match="canonical extraction changed"):
        validate_saturation_result(tampered)
