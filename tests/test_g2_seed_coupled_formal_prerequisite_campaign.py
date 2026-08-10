from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g2_seed_coupled_formal_prerequisite_campaign import (
    _sha,
    _validate_target,
    build_g2_seed_coupled_formal_prerequisite_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g2_seed_coupled_formal_prerequisite_campaign.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g2-seed-coupled-formal-prerequisite-campaign.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g2_seed_coupled_formal_prerequisite_campaign(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "86231cb76c45e3e49880aa0b37f0c74685efa5cc4d889d7d6dadf7c6efc427f2"
    )


def test_candidate_specific_local_and_common_cone_certificates_are_exact(rebuilt: dict) -> None:
    by_id = {item["seed_id"]: item for item in rebuilt["candidate_records"]}
    quarter = by_id["G3-a82c572555e5d79686bc4a4a"]["candidate_certificate"]
    eighth = by_id["G3-e8b35002cfc9c60691a2f67b"]["candidate_certificate"]
    assert quarter["uniform_margins"]["scalar_speed_squared_interval"] == ["3/5", "1"]
    assert eighth["uniform_margins"]["scalar_speed_squared_interval"] == ["5/7", "1"]
    assert quarter["unitary_lapse_pair"]["Delta_N_multiplication_factor"] == (
        "(N^2+(3/4))/N^5"
    )
    assert eighth["unitary_lapse_pair"]["Delta_N_multiplication_factor"] == (
        "(N^2+(3/8))/N^5"
    )
    for certificate in (quarter, eighth):
        assert certificate["uniform_margins"]["G2_X_minimum"] == "1"
        assert certificate["uniform_margins"]["G2_X_plus_2X_G2_XX_minimum"] == "1"
        assert certificate["common_time_cone"]["status"] == "pass"
        assert certificate["common_time_cone"]["covers_declared_causal_gradient_cell"] is True
        assert certificate["pointwise_scalar_hamiltonian"]["nonnegative_on_domain"] is True
        assert certificate["dominant_energy_condition"]["status_on_causal_gradient_domain"] == "pass"


def test_local_passes_do_not_promote_missing_global_prerequisites(rebuilt: dict) -> None:
    assert rebuilt["target_seed_count"] == 2
    assert rebuilt["decision_counts"] == {"blocked": 2}
    assert rebuilt["formal_pass_count"] == 0
    for record in rebuilt["candidate_records"]:
        gates = record["gate_ledger"]
        assert gates["coupled_adm_primary_and_legendre"]["status"] == "pass"
        assert gates["candidate_local_dirac_pair"]["status"] == "pass"
        assert gates["principal_symbol"]["status"] == "pass"
        assert gates["common_time_cone"]["status"] == "pass"
        assert gates["complete_distributed_dirac_boundary_contract"]["status"] == "blocked"
        assert gates["global_positive_energy"]["status"] == "blocked"
        assert record["decision"] == "blocked"
        assert record["negative_energy_counterexample_found"] is False
        assert record["solar_bundle"] == {"generated": False, "status": "blocked"}


def test_known_answer_and_eligibility_are_calibration_only_and_sealed(rebuilt: dict) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert rebuilt["invoked_adapter_entrypoints"] == [
        item["entrypoint"] for item in config["formal_adapters"]
    ]
    assert rebuilt["known_answer_control"]["eligible_as_candidate_evidence"] is False
    assert rebuilt["known_answer_control"]["pass_gate_count"] == 13
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_typed_action_binding_is_rejected() -> None:
    predecessor = json.loads(
        (ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-compilation-campaign.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target = copy.deepcopy(config["target_seeds"][0])
    record = next(item for item in predecessor["candidate_records"] if item["seed_id"] == target["seed_id"])
    target["action_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(record, target)
