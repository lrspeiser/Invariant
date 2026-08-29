from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_lead_recombination as R

ROOT = Path(__file__).resolve().parents[1]


def test_stored_recombination_receipt_rebuilds_from_hardened_parent_registry() -> None:
    stored = json.loads((ROOT / R.OUTPUT_PATH).read_text(encoding="utf-8"))
    R.validate_receipt(stored, ROOT)
    assert R.build_receipt(ROOT) == stored
    assert stored["parent_registry"]["lead_count"] == 5
    assert stored["parent_registry"]["registered_evidence_files"] == 74


def test_all_pairs_triples_roles_and_interfaces_are_structurally_covered() -> None:
    config, _ = R.load_config(ROOT)
    plans = R.build_descendant_plans(config)
    assert len(config["pairwise_recombinations"]) == 10
    assert len(config["triple_recombinations"]) == 10
    assert len(plans) == 20
    assert len({row["descendant_id"] for row in plans}) == 20
    assert len({row["plan_sha256"] for row in plans}) == 20
    assert tuple(config["lead_roles"]) == R.LEAD_IDS
    assert tuple(
        row["interface_id"] for row in config["dimensionless_interfaces"]
    ) == R.INTERFACE_IDS
    assert all(row["dimensionless"] for row in config["dimensionless_interfaces"])
    assert all(row["target_data_bindings"] == [] for row in plans)
    assert all(row["scientific_payload_rows_read"] == 0 for row in plans)
    assert all(row["outcome_scores_computed"] == 0 for row in plans)
    assert all(row["execution_authorized"] is False for row in plans)


def test_top_ben_architecture_is_additive_with_age_nuisance_and_resonance_deferred() -> None:
    receipt = R.build_receipt(ROOT)
    top = receipt["top_architecture"]
    assert top["members"] == [
        "nonlocal_boundary_response",
        "baryonic_transition_variable",
        "emergent_gravity_transition",
    ]
    assert top["base_lead"] == "emergent_gravity_transition"
    assert top["gate_lead"] == "baryonic_transition_variable"
    assert top["additive_channel_lead"] == "nonlocal_boundary_response"
    assert top["nuisance_lead"] == "dynamical_age_spectral_clock"
    assert top["deferred_lead"] == "massive_field_orbital_resonance"
    assert "+" in top["formula_template"]
    assert top["children_empirically_work"] is False
    assert receipt["ablation_contract"]["composition_mode"] == (
        "additive_orthogonal_channels"
    )
    assert receipt["ablation_contract"]["products_of_channels_allowed"] is False
    assert receipt["ablation_contract"]["required_modes"] == list(R.ABLATION_MODES)


def test_controls_target_blindness_and_novelty_claim_boundary_are_frozen() -> None:
    receipt = R.build_receipt(ROOT)
    controls = receipt["control_contract"]
    assert controls["matched_complexity_required"] is True
    assert tuple(row["control_id"] for row in controls["controls"]) == R.CONTROL_IDS
    assert all(row["target_blind"] for row in controls["controls"])
    assert controls["single_counterexample_is_universal_veto"] is False
    novelty = receipt["novelty_policy"]
    assert novelty["allowed_labels"] == list(R.NOVELTY_LABELS)
    assert novelty["labels_are_authoritative"] is False
    assert novelty["labels_are_historical_novelty_findings"] is False
    assert novelty["specialist_prior_art_review_required"] is True
    assert all(
        gate["preflight_satisfies_gate"] is False
        for gate in receipt["publication_interest_gates"]
    )


def test_target_or_object_label_authority_tamper_fails_closed() -> None:
    config, _ = R.load_config(ROOT)
    changed = copy.deepcopy(config)
    changed["target_blind_generation"]["object_label_switches_allowed"] = True
    with pytest.raises(R.GravityLeadRecombinationError, match="seal changed"):
        R.validate_config(changed, ROOT)

    changed = copy.deepcopy(config)
    changed["safety_contract"]["scientific_payload_rows_allowed"] = 1
    with pytest.raises(R.GravityLeadRecombinationError, match="seal changed"):
        R.validate_config(changed, ROOT)


def test_matrix_role_or_forbidden_rule_tamper_fails_closed() -> None:
    config, _ = R.load_config(ROOT)
    changed = copy.deepcopy(config)
    changed["pairwise_recombinations"][0]["disposition"] = "unrestricted"
    with pytest.raises(R.GravityLeadRecombinationError, match="seal changed"):
        R.validate_config(changed, ROOT)

    changed = copy.deepcopy(config)
    changed["forbidden_combinations"].pop()
    with pytest.raises(R.GravityLeadRecombinationError, match="seal changed"):
        R.validate_config(changed, ROOT)


def test_parent_registry_binding_and_resealed_receipt_tamper_fail_closed() -> None:
    config, _ = R.load_config(ROOT)
    changed = copy.deepcopy(config)
    changed["parent_registry_binding"]["receipt_content_sha256"] = "0" * 64
    with pytest.raises(R.GravityLeadRecombinationError, match="seal changed"):
        R.validate_config(changed, ROOT)

    receipt = R.build_receipt(ROOT)
    receipt["descendant_plans"][0]["execution_authorized"] = True
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = R._canonical_sha256(body)
    with pytest.raises(R.GravityLeadRecombinationError, match="descendant plans changed"):
        R.validate_receipt(receipt, ROOT, rebuild=False)


def test_preflight_claims_no_execution_success_or_publication_result() -> None:
    receipt = R.build_receipt(ROOT)
    assert receipt["safety"] == {
        "metadata_only": True,
        "scientific_payload_rows_read": 0,
        "sealed_target_rows_opened": 0,
        "outcome_scores_computed": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "gpu_production_runs": 0,
        "children_executed": 0,
    }
    assert receipt["release_gate"]["status"] == (
        "PASS_PREFLIGHT_ONLY_EXECUTION_BLOCKED"
    )
    assert receipt["release_gate"]["child_execution_authorized"] is False
    assert receipt["claim_boundary"] == {
        "children_empirically_work": False,
        "physical_mechanism_established": False,
        "alternative_to_gr_established": False,
        "dark_matter_eliminated": False,
        "historical_novelty_established": False,
        "publication_gate_passed": False,
        "structural_preflight_only": True,
    }
