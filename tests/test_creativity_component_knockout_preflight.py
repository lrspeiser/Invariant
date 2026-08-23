from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import creativity_component_knockout_preflight as K
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / K.OUTPUT_PATH


def test_stored_preflight_rebuilds_and_validates_from_current_sources() -> None:
    stored = json.loads(RECEIPT.read_text(encoding="utf-8"))
    K.validate_receipt(stored, ROOT)
    assert K.build_receipt(ROOT) == stored


def test_four_knockouts_remove_exactly_one_feature_under_matched_budgets() -> None:
    config, _ = K.load_config(ROOT)
    full = config["arms"][K.REFERENCE_ARM]["feature_flags"]
    assert all(full.values())
    assert len(config["experiments"]) == 4
    for experiment in config["experiments"]:
        knockout = config["arms"][experiment["knockout_arm"]]["feature_flags"]
        changed = {feature for feature in K.FEATURES if knockout[feature] != full[feature]}
        assert changed == {experiment["removed_feature"]}
        assert knockout[experiment["removed_feature"]] is False

    confirmatory = json.loads(
        (ROOT / config["confirmatory_binding"]["path"]).read_text(encoding="utf-8")
    )
    assert config["matched_resource_budget"] == confirmatory["matched_resource_budget"]
    assert config["attempt_policy"] == confirmatory["attempt_policy"]


def test_preflight_seals_all_slots_without_key_or_provider_access() -> None:
    receipt = K.build_receipt(ROOT)
    schedule = receipt["schedule"]
    assert receipt["design"]["live_executor_source_bound"] is True
    assert "live_generation_runner" in receipt["source_bindings"]
    assert schedule["total_scheduled_slots"] == 384
    assert len(schedule["slot_sha256s"]) == 384
    assert len(set(schedule["slot_sha256s"])) == 384
    assert {item["scheduled_slots"] for item in schedule["experiments"]} == {96}
    accounting = receipt["resource_accounting"]
    assert accounting["preflight_provider_calls"] == 0
    assert accounting["preflight_credential_accesses"] == 0
    assert accounting["maximum_provider_calls_if_all_four_runs_are_authorized"] == 384
    assert accounting["maximum_total_tokens_if_all_four_runs_are_authorized"] == 1_600_000
    assert receipt["release_gate"]["component_knockout_live_runs_complete"] is False
    assert receipt["claims"]["more_creative_established"] is False


def test_second_feature_removal_fails_closed() -> None:
    config, _ = K.load_config(ROOT)
    changed = copy.deepcopy(config)
    changed["arms"]["minus_expanded_grammar"]["feature_flags"][
        "independent_proof_recombination"
    ] = False
    with pytest.raises(K.ComponentKnockoutPreflightError, match="does not remove exactly"):
        K.validate_config(changed, ROOT)


def test_execution_semantics_enforce_each_registered_intervention() -> None:
    config, _ = K.load_config(ROOT)
    full = config["arms"][K.REFERENCE_ARM]["execution_semantics"]
    assert full["admitted_representations"] == list(K.FULL_REPRESENTATIONS)
    assert full["independent_proof_plans_per_hypothesis"] == 2
    assert full["post_generation_recombinations_per_task"] == 3
    assert full["origin_lineage_mode"] == "preserve"
    assert full["critic_reject_action"] == "retain_for_repair"

    assert config["arms"]["minus_expanded_grammar"]["execution_semantics"][
        "admitted_representations"
    ] == ["sympy_expression"]
    assert config["arms"]["minus_independent_proof_recombination"][
        "execution_semantics"
    ]["independent_proof_plans_per_hypothesis"] == 0
    assert config["arms"]["minus_independent_proof_recombination"][
        "execution_semantics"
    ]["post_generation_recombinations_per_task"] == 0
    assert config["arms"]["minus_lineage_labels"]["execution_semantics"][
        "origin_lineage_mode"
    ] == "normalize_uncertain_and_hide"
    assert config["arms"]["minus_non_pruning"]["execution_semantics"][
        "critic_reject_action"
    ] == "drop_before_expansion"

    changed = copy.deepcopy(config)
    changed["arms"]["minus_non_pruning"]["execution_semantics"][
        "critic_reject_action"
    ] = "retain_for_repair"
    with pytest.raises(K.ComponentKnockoutPreflightError, match="semantics changed"):
        K.validate_config(changed, ROOT)


def test_unmatched_resource_budget_fails_closed() -> None:
    config, _ = K.load_config(ROOT)
    changed = copy.deepcopy(config)
    changed["matched_resource_budget"]["grammar_depth"] += 1
    with pytest.raises(K.ComponentKnockoutPreflightError, match="resource budget is not matched"):
        K.validate_config(changed, ROOT)


def test_resealed_schedule_tamper_fails_against_deterministic_rebuild() -> None:
    changed = K.build_receipt(ROOT)
    changed["schedule"]["slot_sha256s"][0] = "0" * 64
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(K.ComponentKnockoutPreflightError, match="schedule changed"):
        K.validate_receipt(changed, ROOT)
