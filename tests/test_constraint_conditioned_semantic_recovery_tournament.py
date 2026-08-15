from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.constraint_conditioned_semantic_recovery_tournament import (
    CLAIMS,
    CONFIG_FILE_SHA256,
    CONFIG_PATH,
    FAMILIES,
    OUTPUT_PATH,
    WORLD_IDS,
    build_campaign,
    synthesize_from_constraints,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def campaign() -> dict[str, object]:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_campaign(value, ROOT)
    return value


def test_preregistration_is_exact_closed_and_answer_free(config: dict[str, object]) -> None:
    assert hashlib.sha256((ROOT / CONFIG_PATH).read_bytes()).hexdigest() == CONFIG_FILE_SHA256
    assert tuple(config["generator_families"]) == FAMILIES
    assert tuple(row["world_id"] for row in config["worlds"]) == WORLD_IDS
    assert config["preregistration_status"] == "sealed_before_generation"
    assert config["generic_grammar"] == {
        "coefficient_domain": "rationals",
        "expression_form": "linear_combination_of_declared_basis",
        "solver": "exact_rank_classification_and_unique_linear_solve",
        "target_blind": True,
        "outcomes": {
            "inconsistent": "REJECT",
            "malformed": "BLOCK",
            "underdetermined": "BLOCK",
            "unique": "CANDIDATE",
        },
    }
    serialized = json.dumps(config, sort_keys=True)
    assert "x**4 + 2*x**3 - x - 30" not in serialized
    assert "3/(x + 2) - 2/(x + 5) + 5/(x + 7)" not in serialized
    assert "2*n**3 + 2*n**2 + n + 7" not in serialized
    assert all("closed_form" not in row for row in config["worlds"])


def test_generic_solver_is_target_blind_deterministic_and_not_world_specific(
    config: dict[str, object],
) -> None:
    expressions = []
    for world in config["worlds"]:
        first = synthesize_from_constraints(
            world["variable"],
            world["basis"],
            world["constraints"],
            proposal_material="lineage-a",
        )
        second = synthesize_from_constraints(
            world["variable"],
            world["basis"],
            world["constraints"],
            proposal_material="lineage-b",
        )
        assert first.outcome == second.outcome == "CANDIDATE"
        assert first.expression == second.expression
        assert first.coefficients == second.coefficients
        assert first.rank == first.column_count
        expressions.append(str(first.expression))
    assert expressions == [
        "x**4 + 2*x**3 - x - 30",
        "(6*x**2 + 53*x + 127)/(x**3 + 14*x**2 + 59*x + 70)",
        "2*n**3 + 2*n**2 + n + 7",
    ]


def test_negative_controls_fail_closed_with_exact_rank_semantics(
    config: dict[str, object],
) -> None:
    observed = {}
    for control in config["controls"]:
        result = synthesize_from_constraints(
            control["variable"],
            control["basis"],
            control["constraints"],
            proposal_material=control["control_id"],
        )
        observed[control["control_id"]] = (result.outcome, result.reason)
    assert observed["malformed_unknown_symbol"][0] == "BLOCK"
    assert observed["malformed_unknown_symbol"][1].startswith("malformed_constraints:")
    assert observed["underdetermined_rank_deficient"] == (
        "BLOCK",
        "underdetermined_exact_constraints",
    )
    assert observed["noisy_inconsistent_duplicate"] == (
        "REJECT",
        "inconsistent_exact_constraints",
    )


def test_campaign_chronology_outcomes_proofs_and_pareto_are_exact(
    campaign: dict[str, object],
) -> None:
    assert campaign["decision"] == "pass_three_of_three_constraint_conditioned_semantic_worlds"
    assert campaign["counts"] == {
        "candidate_blocks": 0,
        "candidate_passes": 21,
        "candidate_rejects": 0,
        "control_blocks": 2,
        "control_rejects": 1,
        "generator_families": 7,
        "generic_synthesis_invocations": 21,
        "metric_receipts": 42,
        "native_generator_invocations": 21,
        "pareto_eligible_candidates": 21,
        "proof_certificates": 21,
        "world_passes": 3,
        "worlds": 3,
    }
    ledger = campaign["phase_ledger"]
    assert ledger["generation_and_solve_events_before_unseal"] == 21
    assert ledger["pre_unseal_target_access_count"] == 0
    assert ledger["atomic_unseal_batches"] == 1
    assert ledger["post_unseal_generation_count"] == 0
    assert ledger["post_unseal_tuning_events"] == 0
    assert ledger["events"][-1]["event"] == "targets_atomically_unsealed"
    assert all(row["event"] == "generated_and_constraint_solved" for row in ledger["events"][:-1])
    assert campaign["claims"] == CLAIMS
    for world in campaign["world_results"]:
        assert world["terminal_status_counts"] == {"pass": 7}
        assert world["pareto_eligible_families"] == list(FAMILIES)
        assert len(world["metric_receipts"]) == 14
        assert world["reference_proof_certificate"]["content_sha256"]
        assert all(
            row["proof_certificate"]["content_sha256"] for row in world["assessments"].values()
        )
        assert all(row["all_required_gates_passed"] for row in world["evaluations"].values())


def test_holdout_and_lineage_are_closed_and_native_apis_are_all_bound(
    campaign: dict[str, object],
) -> None:
    for world in campaign["world_results"]:
        assert set(world["native_generator_receipts"]) == set(FAMILIES)
        assert len(world["candidates"]) == 7
        for candidate in world["candidates"]:
            representation = candidate["representation"]
            assert representation["target_fields_read"] == []
            assert representation["solver_receipt"]["outcome"] == "CANDIDATE"
            assert representation["native_candidate"] in candidate["provenance"]["inputs"]
        cut = world["holdout"]["cut"]
        assert len(cut["visible_node_ids"]) == 1
        assert len(cut["forbidden_node_ids"]) == 2


def test_replay_is_deterministic_and_resealed_semantic_tamper_is_rejected(
    campaign: dict[str, object],
) -> None:
    assert campaign == build_campaign(ROOT)
    mutated = copy.deepcopy(campaign)
    mutated["claims"]["scientific_truth_established"] = True
    mutated["content_sha256"] = canonical_sha256(
        {key: value for key, value in mutated.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="claim boundary changed"):
        validate_campaign(mutated, ROOT)

    mutated = copy.deepcopy(campaign)
    mutated["world_results"][0]["candidates"][0]["representation"]["target_fields_read"] = [
        "closed_form"
    ]
    mutated["content_sha256"] = canonical_sha256(
        {key: value for key, value in mutated.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="exact replay mismatch"):
        validate_campaign(mutated, ROOT)
