from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.candidate_knowledge_graph import (
    CandidateKnowledgeGraph,
    HoldoutCut,
)
from sigma_theory_compiler.prospective_blind_cross_generator_tournament import (
    CLAIMS,
    CONFIG_PATH,
    FAMILIES,
    OUTPUT_PATH,
    WORLD_ROWS,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return build_campaign(ROOT)


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    )


def test_preregistration_is_exactly_three_unseen_worlds_with_fixed_budgets() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert [
        (row["world_id"], row["public_seed"], row["sealed_target_sha256"])
        for row in config["worlds"]
    ] == list(WORLD_ROWS)
    assert len({row["world_id"] for row in config["worlds"]}) == 3
    assert len({row["public_seed"] for row in config["worlds"]}) == 3
    assert config["generator_families"] == list(FAMILIES)
    assert config["budgets"] == {
        "candidates_per_family_per_world": 1,
        "generator_work_items_per_world": 128,
        "hypothesis_inventory_size": 11,
        "llm_calls_per_world": 1,
        "llm_maximum_micro_usd_per_world": 1000,
        "maximum_pareto_work_units_per_world": 512,
    }
    assert config["policies"]["post_unseal_tuning"] == "forbidden"
    assert config["policies"]["target_records_per_unseal"] == 3
    assert config["policies"]["target_unseal_batches"] == 1


def test_all_native_generation_precedes_the_single_unseal_phase(
    result: dict[str, object],
) -> None:
    assert result["counts"]["worlds"] == 3
    assert result["counts"]["generator_families"] == 7
    assert result["counts"]["selected_candidates"] == 21
    assert result["counts"]["evaluation_results"] == 21
    ledger = result["phase_ledger"]
    assert ledger["generation_events_before_first_unseal"] == 21
    assert ledger["pre_unseal_target_access_count"] == 0
    assert ledger["post_unseal_generation_count"] == 0
    assert ledger["post_unseal_tuning_events"] == 0
    assert ledger["target_records_unsealed"] == {row[0]: 1 for row in WORLD_ROWS}
    assert ledger["target_unseal_batches"] == 1
    assert [row["event"] for row in ledger["events"]] == ["generated"] * 21 + ["targets_unsealed"]
    assert result["claims"] == CLAIMS


def test_holdout_evaluation_and_pareto_are_common_and_fail_closed(
    result: dict[str, object],
) -> None:
    total_eligible = 0
    for world in result["world_results"]:
        graph = CandidateKnowledgeGraph.from_dict(world["holdout"]["graph"])
        cut = HoldoutCut.from_dict(world["holdout"]["cut"])
        cut.validate_against(graph)
        assert len(cut.visible_node_ids) == 1
        assert len(cut.forbidden_node_ids) == 2
        assert tuple(row["family"] for row in world["family_bindings"]) == FAMILIES
        assert set(world["native_generator_receipts"]) == set(FAMILIES)
        assert set(world["evaluations"]) == set(FAMILIES)
        eligible = set(world["pareto_eligible_families"])
        total_eligible += len(eligible)
        assert world["decision"] in {
            "pass_at_least_one_target_blind_candidate_survived",
            "reject_fixed_budget_exhausted_without_holdout_match",
            "block_missing_registered_holdout_evidence",
        }
        candidate_by_id = {row["artifact_id"]: row for row in world["candidates"]}
        public_sha256 = canonical_sha256(
            {
                "world_id": world["world_id"],
                "public_seed": world["public_seed"],
                "hypothesis_inventory": list(range(11)),
                "target_disclosed": False,
            }
        )
        for family, evaluation in world["evaluations"].items():
            assert evaluation["counts"]["registered_steps"] == 2
            assert evaluation["counts"]["attempted_gates"] == 2
            assert evaluation["gate_outcomes"][0]["status"] == "pass"
            expected = "pass" if family in eligible else "reject"
            assert evaluation["gate_outcomes"][1]["status"] == expected
            artifact_id = evaluation["artifact"]["artifact_id"]
            assert candidate_by_id[artifact_id]["representation"]["target_fields_read"] == []
            assert (
                candidate_by_id[artifact_id]["representation"]["public_world_sha256"]
                == public_sha256
            )
        assert len(world["metric_receipts"]) == 2 * len(eligible)
        if eligible:
            assert world["pareto"] is not None
            ranked = {
                row["artifact_id"] for front in world["pareto"]["pareto_fronts"] for row in front
            }
            expected_ranked = {
                row["candidate"]["artifact_id"]
                for row in world["family_bindings"]
                if row["family"] in eligible
            }
            assert ranked == expected_ranked
        else:
            assert world["pareto"] is None
    assert result["counts"]["pareto_eligible_candidates"] == total_eligible


def test_campaign_is_deterministic_validated_and_committed(result: dict[str, object]) -> None:
    replay = build_campaign(ROOT)
    assert replay == result
    validate_campaign(result, ROOT)
    checked = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert checked == result
    assert result["claims"]["generator_output_establishes_truth"] is False
    assert result["claims"]["promotion_authorized"] is False


def test_resealed_leakage_unseal_and_eligibility_tampers_fail_closed(
    result: dict[str, object],
) -> None:
    leaked = copy.deepcopy(result)
    leaked["world_results"][0]["candidates"][0]["representation"]["target_fields_read"] = [
        "hypothesis"
    ]
    _reseal(leaked)
    with pytest.raises(ValueError, match="immutable replay mismatch"):
        validate_campaign(leaked, ROOT)

    extra_unseal = copy.deepcopy(result)
    extra_unseal["phase_ledger"]["target_records_unsealed"][WORLD_ROWS[0][0]] = 2
    _reseal(extra_unseal)
    with pytest.raises(ValueError, match="immutable replay mismatch"):
        validate_campaign(extra_unseal, ROOT)

    promoted = copy.deepcopy(result)
    promoted["claims"]["promotion_authorized"] = True
    _reseal(promoted)
    with pytest.raises(ValueError, match="claim boundary changed"):
        validate_campaign(promoted, ROOT)
