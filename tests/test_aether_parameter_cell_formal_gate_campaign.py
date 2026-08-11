from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

import sigma_theory_compiler.aether_parameter_cell_formal_gate_campaign as campaign_module
from sigma_theory_compiler.aether_parameter_cell_formal_gate_campaign import (
    _sha,
    build_aether_parameter_cell_formal_gate_campaign,
    build_aether_parameter_cell_formal_gate_status,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "aether_parameter_cell_formal_gate_campaign.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "aether-parameter-cell-formal-gate-status.json"
EXPECTED_CONFIG_FILE_SHA256 = "082028d898c69c79f9dd39d68500862279370b59af4d145495d56620280b880d"
EXPECTED_SOURCE_FILE_SHA256 = "fad67e2478690ef3cff51976e199b115ba05190ec1c1c0fdaf74d6dbe687cb0c"
EXPECTED_ARTIFACT_FILE_SHA256 = "3b0c213653fd744a0284b86c3a6f871246af59a037a5913f954402aef2390dc6"
EXPECTED_CAMPAIGN_CONTENT_SHA256 = (
    "b475905e2f18e8eb4cd82b2b54d02403d2ee7e126d5402a555348b5d71034acb"
)
EXPECTED_STATUS_CONTENT_SHA256 = "edbe0f8e888977a7a0c9ffea1f330b45c0a7259c2ee1251ed88cc101824dda1e"
EXPECTED_PREFLIGHT_ROOT_SHA256 = "88ae6d2cb555bdc41dd0854b174789825d2c2467338e7fdc643f71b557095366"
EXPECTED_CANDIDATE_RECORD_ROOT_SHA256 = (
    "383bdebe521da344eeb2deb1151d344a508823946ac3260ea3be57572a7256ca"
)
EXPECTED_CANDIDATE_BINDING_ROOT_SHA256 = (
    "9ce52240dd362b127a4cebf70235a6571fc4057152f1e19dbc52c568af05250c"
)


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def campaign() -> dict:
    return build_aether_parameter_cell_formal_gate_campaign(_load_config(), ROOT)


def test_replay_matches_portable_artifact_and_exact_file_bindings(campaign: dict) -> None:
    status = build_aether_parameter_cell_formal_gate_status(campaign)
    checked_in = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert status == checked_in
    assert campaign == build_aether_parameter_cell_formal_gate_campaign(_load_config(), ROOT)
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == EXPECTED_CONFIG_FILE_SHA256
    assert (
        hashlib.sha256(
            (
                ROOT / "src/sigma_theory_compiler/aether_parameter_cell_formal_gate_campaign.py"
            ).read_bytes()
        ).hexdigest()
        == EXPECTED_SOURCE_FILE_SHA256
    )
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == EXPECTED_ARTIFACT_FILE_SHA256
    assert campaign["content_sha256"] == EXPECTED_CAMPAIGN_CONTENT_SHA256
    assert status["content_sha256"] == EXPECTED_STATUS_CONTENT_SHA256


def test_exact_candidate_decisions_and_gate_finding_counts(campaign: dict) -> None:
    assert campaign["input_preflight_pass_count"] == campaign["candidate_count"] == 128
    assert campaign["decision_counts"] == {"blocked": 126, "reject": 2}
    assert campaign["formal_pass_count"] == campaign["solar_bundle_count"] == 0
    assert campaign["gate_finding_counts"] == {
        "principal_spin0_degeneracy_reject": 2,
        "finite_negative_local_density_witness": 79,
        "positive_at_every_finite_tilt_but_no_uniform_gap": 8,
        "uniform_positive_static_local_twist_gap": 39,
        "finite_characteristic_slicing_present": 121,
        "globally_noncharacteristic_for_finite_unit_tilt": 5,
    }
    assert (
        campaign["provenance"]["formal_preflight_aether_record_root_sha256"]
        == EXPECTED_PREFLIGHT_ROOT_SHA256
    )
    assert campaign["candidate_gate_record_root_sha256"] == EXPECTED_CANDIDATE_RECORD_ROOT_SHA256
    status = build_aether_parameter_cell_formal_gate_status(campaign)
    assert status["candidate_binding_root_sha256"] == EXPECTED_CANDIDATE_BINDING_ROOT_SHA256


def test_every_candidate_action_preflight_and_gate_record_is_hash_bound(campaign: dict) -> None:
    records = campaign["candidate_records"]
    for key in (
        "candidate_id",
        "action_sha256",
        "action_density_equivalence_sha256",
        "preflight_input_lineage_sha256",
        "preflight_result_sha256",
        "preflight_record_sha256",
        "content_sha256",
    ):
        assert len({record[key] for record in records}) == 128
    for record in records:
        body = {key: value for key, value in record.items() if key != "content_sha256"}
        assert record["content_sha256"] == _sha(body)
        specialization = record["exact_specialization"]
        specialization_body = {
            key: value for key, value in specialization.items() if key != "content_sha256"
        }
        assert specialization["content_sha256"] == _sha(specialization_body)
        provenance = record["provenance"]
        provenance_body = {
            key: value for key, value in provenance.items() if key != "binding_sha256"
        }
        assert provenance["binding_sha256"] == _sha(provenance_body)
        assert provenance["action_sha256"] == record["action_sha256"]
        assert provenance["preflight_result_sha256"] == record["preflight_result_sha256"]


def test_two_cells_have_exact_decisive_spin_zero_principal_degeneracy(campaign: dict) -> None:
    rejected = [
        record for record in campaign["candidate_records"] if record["decision"] == "reject"
    ]
    assert [record["parameters"] for record in rejected] == [
        {"c1": "1/16", "c2": "0", "c3": "-1/16", "c4": "0"},
        {"c1": "1/16", "c2": "0", "c3": "-1/16", "c4": "1/16"},
    ]
    for record in rejected:
        exact = record["exact_specialization"]
        assert exact["combinations"]["c123"] == "0"
        assert Fraction(exact["principal_speed_squared"]["spin_0"]) == 0
        assert record["blocker"] == "nonpositive_spin0_principal_numerator_c123"
        assert record["gate_ledger"]["aligned_minkowski_principal_and_linear_modes"] == {
            "status": "reject"
        }


def test_local_twist_findings_do_not_overclaim_total_energy_rejection(campaign: dict) -> None:
    negative = [
        record
        for record in campaign["candidate_records"]
        if record["gate_ledger"]["static_unit_reduced_pure_twist_local_energy"].get("finding")
        == "finite_negative_local_density_witness"
    ]
    assert len(negative) == 79
    assert {record["decision"] for record in negative} == {"blocked"}
    for record in negative:
        witness = record["exact_specialization"]["finite_negative_twist_witness"]
        assert witness["local_hamiltonian_density_negative"] is True
        assert witness["full_gravitational_constraint_embedding_proven"] is False
        assert witness["candidate_rejection_authorized_by_this_witness_alone"] is False
        assert Fraction(witness["C_y"]) < 0
        assert record["gate_ledger"]["global_positive_energy"] == {"status": "blocked"}
    representative = next(
        record
        for record in negative
        if record["parameters"] == {"c1": "1/4", "c2": "1/8", "c3": "0", "c4": "1/16"}
    )
    assert representative["exact_specialization"]["static_twist_large_tilt_limit"] == (
        "negative_infinity"
    )
    assert campaign["static_twist_asymptotic_correction"]["exact_limit"] == (
        "negative_infinity_for_c4>0; c1/2-c3_for_c4=0"
    )


def test_global_tilt_strata_are_exactly_conditional_not_physical_rejections(campaign: dict) -> None:
    conditional = [
        record
        for record in campaign["candidate_records"]
        if record["gate_ledger"]["global_unit_tilt_legendre_strata"]["status"] == "conditional"
    ]
    assert len(conditional) == 121
    for record in conditional:
        thresholds = record["gate_ledger"]["global_unit_tilt_legendre_strata"][
            "finite_characteristic_tilt_squared"
        ]
        assert thresholds
        speeds = record["exact_specialization"]["principal_speed_squared"]
        for sector, threshold in thresholds.items():
            speed = Fraction(speeds[sector])
            assert Fraction(threshold) == 1 / (speed - 1)
        assert record["decision"] == "blocked"


def test_tamper_missing_adapter_and_observation_opening_fail_closed(
    campaign: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered_campaign = copy.deepcopy(campaign)
    tampered_campaign["candidate_records"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign content binding changed"):
        build_aether_parameter_cell_formal_gate_status(tampered_campaign)

    missing = _load_config()
    missing["formal_adapters"] = missing["formal_adapters"][:-1]
    with pytest.raises(ValueError, match="required reviewed Aether formal adapter is missing"):
        build_aether_parameter_cell_formal_gate_campaign(missing, ROOT)

    opened = _load_config()
    opened["observational_authorization"] = True
    with pytest.raises(ValueError, match="opened observations"):
        build_aether_parameter_cell_formal_gate_campaign(opened, ROOT)

    original_compile = campaign_module._compile_action_ir

    def tampered_compile(*args, **kwargs):
        action = original_compile(*args, **kwargs)
        return {**action, "content_sha256": "0" * 64}

    monkeypatch.setattr(campaign_module, "_compile_action_ir", tampered_compile)
    with pytest.raises(ValueError, match="formal-preflight record chunk changed"):
        build_aether_parameter_cell_formal_gate_campaign(_load_config(), ROOT)


def test_checked_artifact_keeps_every_external_input_sealed(campaign: dict) -> None:
    assert campaign["observational_data_opened"] is False
    assert campaign["dark_matter_or_halo_inputs"] is False
    assert campaign["redshift_distance_inputs"] is False
    assert campaign["paid_llm_spend_usd"] == 0.0
    assert campaign["data_eligibility"] == {
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_calls": False,
    }
    assert all(record["formal_pass"] is False for record in campaign["candidate_records"])
    assert all(
        record["solar_bundle_generated"] is False for record in campaign["candidate_records"]
    )
