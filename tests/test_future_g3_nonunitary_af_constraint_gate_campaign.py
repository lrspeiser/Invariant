from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_nonunitary_af_constraint_gate_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_nonunitary_af_constraint_gate_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "future_g3_nonunitary_af_constraint_gate_campaign.json"
ARTIFACT = (
    ROOT / "runs" / "engine" / "future-g3-nonunitary-af-constraint-gate-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_nonunitary_af_constraint_gate_campaign(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "d7574dbc65b833a923ca653a9fc9c87b9fa872a24c6c89020afa995a7039e46a"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "cd569bb88453486ad02257009722f2c34ff1faee107b6b38ebe5e076db37cde9"
    )


def test_all_three_nonunitary_principal_formulations_are_candidate_bound(rebuilt: dict) -> None:
    expected = {
        "33/4000": {
            "spatial_eigenvalue_lower": "63998911/64000000",
            "time_space_norm_upper_squared": "1089/80000000000",
            "slicing_cone_separation": "99967/100000",
        },
        "17/2000": {
            "spatial_eigenvalue_lower": "15999711/16000000",
            "time_space_norm_upper_squared": "289/20000000000",
            "slicing_cone_separation": "49983/50000",
        },
        "9/1000": {
            "spatial_eigenvalue_lower": "3999919/4000000",
            "time_space_norm_upper_squared": "81/5000000000",
            "slicing_cone_separation": "24991/25000",
        },
    }
    for record in rebuilt["candidate_records"]:
        certificate = record["nonunitary_AF_principal_certificate"]
        assert certificate["candidate_id"] == record["candidate_id"]
        assert certificate["action_sha256"] == record["action_sha256"]
        assert certificate["direct_candidate_specialization"] is True
        assert certificate["family_label_used_as_equivalence_proof"] is False
        assert certificate["unitary_lapse_multiplier_or_inverse_used"] is False
        assert certificate["X_to_zero_limit"] == {
            "P00": "-1",
            "Pij_eigenvalue": "1",
            "physical_principal_degeneracy": False,
            "ordinary_lapse_coefficient_depends_on_X": False,
        }
        bounds = certificate["uniform_exact_bounds"]
        assert bounds["P00_upper"] == "-1"
        assert bounds["common_time_covector_margin"] == "1"
        for name, value in expected[record["beta"]].items():
            assert bounds[name] == value
        assert certificate["BSSN_gauge_roots_squared"] == {
            "transverse": "1",
            "momentum": "1",
            "slicing": "2",
            "longitudinal": "1",
        }
        assert certificate["status"] == (
            "pass_candidate_bound_nonunitary_AF_principal_formulation"
        )


def test_nontrivial_reference_constraint_failure_is_radial_and_exact(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        gate = record["Einstein_constraint_gate"]
        ansatz = gate["nontrivial_decaying_gradient_reference_ansatz"]
        assert ansatz["role"] == "principal_reference_and_rejected_constraint_ansatz_only"
        assert ansatz["canonical_energy_density"] == "v(r)^2/2"
        assert ansatz["cubic_G3_energy_density"] == "0"
        assert ansatz["cubic_G3_matter_flux_T_n_r"] == "-beta*v(r)^2*d_v_d_r"
        assert ansatz["cubic_G3_matter_flux_for_profile"] == (
            "2*beta*r^3/L^4/(1+(r/L)^4)^(5/2)"
        )
        assert ansatz["Hamiltonian_constraint_residual_LHS_minus_2rho"] == "-v**2"
        assert ansatz["Hamiltonian_constraint_residual_for_profile"] == (
            "-1/(1+(r/L)^4)"
        )
        assert ansatz["matter_flux_nonzero_for_every_r_greater_than_zero"] is True
        assert ansatz["status"] == "reject_reference_ansatz_as_Einstein_constraint_solution"
        assert ansatz["theory_rejected"] is False
        assert record["theory_rejected"] is False


def test_actual_vacuum_solution_is_separate_and_not_promoted(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        gate = record["Einstein_constraint_gate"]
        vacuum = gate["actual_AF_vacuum_constraint_solution"]
        assert vacuum["candidate_action_bound"] is True
        assert vacuum["asymptotically_flat"] is True
        assert vacuum["Hamiltonian_constraint_residual"] == "0"
        assert vacuum["momentum_constraint_residual"] == "0"
        assert vacuum["scalar_stress_tensor"] == "0"
        assert vacuum["status"] == "pass_actual_AF_vacuum_constraint_solution"
        assert vacuum["overlap_with_nontrivial_transition_profile"] == (
            "X=0_asymptotic_endpoint_only"
        )
        assert gate["candidate_nontrivial_AF_constraint_solution_available"] is False
        assert "reference AF constraint solution" in gate["constraint_solution_counting_contract"]
        assert record["gate_ledger"]["actual_AF_vacuum_constraint_reference"] == {
            "status": "pass_reference_only"
        }
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_counts_blocker_and_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["nonunitary_formulation_registration_pass_count"] == 3
    assert rebuilt["nonunitary_AF_principal_pass_count"] == 3
    assert rebuilt["flat_nontrivial_reference_constraint_ansatz_reject_count"] == 3
    assert rebuilt["actual_AF_vacuum_constraint_reference_pass_count"] == 3
    assert rebuilt["candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"] == 0
    assert rebuilt["global_hamiltonian_energy_pass_count"] == 0
    assert rebuilt["full_formal_pass_count"] == 0
    assert rebuilt["first_blocker_counts"] == {FIRST_BLOCKER: 3}
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["synthetic_fixture_role"] == "none_used"
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    for record in rebuilt["candidate_records"]:
        assert record["decision"] == "blocked"
        assert record["first_blocker"] == FIRST_BLOCKER


def test_action_method_formulation_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_nonunitary_af_constraint_gate_campaign(action, ROOT)

    method = copy.deepcopy(config)
    method["method_control_expectations"]["certificate_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="method control changed"):
        build_future_g3_nonunitary_af_constraint_gate_campaign(method, ROOT)

    formulation = copy.deepcopy(config)
    formulation["alternative_formulation"]["scalar_is_coordinate"] = True
    with pytest.raises(ValueError, match="formulation contract changed"):
        build_future_g3_nonunitary_af_constraint_gate_campaign(formulation, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_nonunitary_af_constraint_gate_campaign(source, ROOT)
