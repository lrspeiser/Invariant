from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g3_componentwise_interval_domain_audit import (
    _sha,
    _validate_target,
    build_g3_componentwise_interval_domain_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g3_componentwise_interval_domain_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g3-componentwise-interval-domain-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g3_componentwise_interval_domain_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "854b2c9b5aa6c10663a4c0097eee1e87ad40dbad5774fed44618bf74c6352f08"
    )


def test_componentwise_box_is_explicit_nonzero_and_action_bound(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    domain = record["componentwise_domain"]
    assert domain["contract_kind"] == "pointwise_local_jet_box_not_evolution_invariant"
    assert domain["frame"] == "local_orthonormal_tetrad_e0_equals_BSSN_foliation_normal"
    assert domain["spatial_gradient_component_abs"] == "1/200"
    assert domain["symmetric_hessian_component_abs"] == "1/100"
    assert domain["riemann_tetrad_component_abs"] == "1/10000"
    assert domain["lapse_interval"] == ["99/100", "101/100"]
    assert domain["direction_sphere"].endswith("no sampling")
    ir = record["candidate_adapter_ir"]
    assert ir["source_action_sha256"] == record["action_sha256"]
    assert ir["formulation_classification"] == {
        "canonical_G2": "x",
        "canonical_G3": "x/100",
        "G4_X": "0",
    }


def test_existing_interval_machinery_certifies_uniform_principal_and_cones(
    rebuilt: dict,
) -> None:
    record = rebuilt["candidate_records"][0]
    certificate = record["principal_common_cone_certificate"]
    proof = certificate["uniform_proof"]
    assert certificate["status"] == "pass_uniform_local_jet_box"
    assert certificate["domain"]["derived_X"]["lower"] > 0.488
    assert proof["common_time_covector_upper_P00"] < -0.999
    assert proof["spatial_block_eigenvalue_lower"] > 0.998
    assert proof["characteristic_discriminant_lower"] > 0.997
    assert proof["slicing_cone_polynomial_upper"] < -0.995
    assert proof["direction_sphere_method"].endswith("no direction sampling")
    assert certificate["weak_field_diagnostic"]["maximum_derivative_ratio"] < 0.011
    assert record["gate_ledger"]["uniform_principal_symbol"]["status"] == "pass"
    assert record["gate_ledger"]["direction_sphere_coverage"] == {
        "status": "pass",
        "method": "no sampling",
    }


def test_interval_negative_controls_reject_invalid_domains(rebuilt: dict) -> None:
    controls = rebuilt["candidate_records"][0]["negative_controls"]
    gradient = controls["non_timelike_gradient_box"]
    slicing = controls["invalid_BSSN_sigma"]
    assert gradient["status"] == "reject"
    assert "non-timelike" in " ".join(gradient["errors"])
    assert slicing["status"] == "reject"
    assert "sigma>1/2" in " ".join(slicing["errors"])


def test_positive_g2_lapse_piece_does_not_promote_unknown_g3_operator(
    rebuilt: dict,
) -> None:
    record = rebuilt["candidate_records"][0]
    lapse = record["lapse_prerequisite"]
    assert lapse["positive_G2_multiplication_contribution"] == {
        "exact": "Delta_N^(G2)=N^-3",
        "lower": "1000000/1030301",
        "upper": "1000000/970299",
        "uniformly_positive": True,
    }
    assert lapse["G3_remainder"]["status"] == "blocked"
    assert lapse["local_full_operator_invertibility"] == "blocked"
    assert lapse["global_boundary_domain"]["status"] == "blocked"
    assert record["first_missing_premise"] == "candidate_specific_full_Delta_N_operator"
    assert record["decision"] == "blocked"
    assert record["necessary_condition_rejection_found"] is False


def test_principal_pass_does_not_open_global_or_external_gates(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    assert rebuilt["uniform_principal_common_cone_pass_count"] == 1
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert rebuilt["full_formal_pass_count"] == 0
    assert record["gate_ledger"]["global_hamiltonian_energy"]["status"] == "blocked"
    assert record["solar_bundle"] == {"generated": False, "status": "blocked"}
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_predecessor_action_is_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    predecessor = json.loads((ROOT / config["predecessor"]["path"]).read_text(encoding="utf-8"))
    target = copy.deepcopy(config["target_seed"])
    record = predecessor["candidate_records"][0]
    target["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(record, target)
