from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.g3_g4_nonunitary_gauge_bypass_audit import (
    _sha,
    _validate_formulations,
    build_g3_g4_nonunitary_gauge_bypass_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g3_g4_nonunitary_gauge_bypass_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g3-g4-nonunitary-gauge-bypass-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g3_g4_nonunitary_gauge_bypass_audit(config, ROOT)


def _record(result: dict, family: str) -> dict:
    return next(item for item in result["candidate_records"] if item["family"] == family)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "7f093215f5151ee0945e9f9cae28e0dc3133f54ce4bfda6f2764961a2cda8451"
    )


def test_g3_nonunitary_bssn_bypasses_chart_but_not_remaining_physics(rebuilt: dict) -> None:
    record = _record(rebuilt, "G3")
    certificate = record["nonunitary_bypass_certificate"]
    assert certificate["unitary_Delta_N_used"] is False
    assert certificate["covariant_scalar_retained_as_evolved_field"] is True
    assert certificate["X_to_zero_limit"] == {
        "P00": "-1",
        "Pij_eigenvalue": "1",
        "scalar_roots": ["-1", "1"],
        "BSSN_slicing_roots": ["-sqrt(2)", "sqrt(2)"],
        "ordinary_lapse_coefficient_depends_on_X": False,
        "physical_principal_degeneracy": False,
    }
    assert certificate["complete_AF_reference_profile_bounds"][
        "effective_spatial_lower"
    ] == "39999/40000"
    assert certificate["bypass_decision"] == "pass_for_principal_formulation_only"
    assert record["decision"] == "blocked"
    assert record["first_missing_premise"] == (
        "candidate_specific_asymptotically_flat_Einstein_constraint_solution"
    )
    assert record["gate_ledger"]["global_energy"]["status"] == "blocked"


def test_g4_global_einstein_scalar_gauge_is_regular_at_zero_gradient(rebuilt: dict) -> None:
    record = _record(rebuilt, "G4")
    certificate = record["nonunitary_bypass_certificate"]
    principal = certificate["generalized_harmonic_principal"]
    wave = sp.Symbol("wave")
    assert sp.sympify(principal["determinant"]) == wave**11
    assert principal["time_block_rank_in_local_orthonormal_frame"] == 11
    assert principal["depends_on_nabla_chi"] is False
    assert principal["regular_at_nabla_chi_zero"] is True
    assert certificate["gauge_constraint_propagation"]["scalar_clock_margin_required"] is False
    assert certificate["ADM_constraint_count"]["physical_phase_dimension"] == 6
    assert certificate["ADM_constraint_count"]["physical_configuration_dof"] == 3
    assert certificate["AF_and_energy_binding"]["candidate_specific_maximal_positive_mass"] == (
        "pass_from_predecessor"
    )
    assert record["decision"] == "pass"
    assert record["first_missing_premise"] is None
    assert record["gate_ledger"]["formal_prerequisite_completion"]["status"] == "pass"


def test_chart_obstruction_is_not_upgraded_to_physical_rejection(rebuilt: dict) -> None:
    assert rebuilt["unitary_chart_obstruction_count"] == 2
    assert rebuilt["nonunitary_bypass_pass_count"] == 2
    assert rebuilt["decision_counts"] == {"pass": 1, "blocked": 1}
    assert rebuilt["full_formal_pass_count"] == 1
    for record in rebuilt["candidate_records"]:
        assert record["unitary_obstruction_classification"] == (
            "chart_obstruction_not_physical_failure"
        )
        assert record["necessary_condition_rejection_found"] is False
        assert record["nonunitary_bypass_certificate"]["physical_principal_degeneracy"] is False


def test_no_unitary_coordinate_contract_can_reenter() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(config["alternative_formulations"])
    tampered["G4"]["scalar_is_coordinate"] = True
    with pytest.raises(ValueError, match="non-unitary formulation contract changed"):
        _validate_formulations(tampered)


def test_tampered_action_binding_is_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["targets"]["G3"]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="G3 action hash mismatch"):
        build_g3_g4_nonunitary_gauge_bypass_audit(config, ROOT)


def test_solar_and_observational_surfaces_remain_sealed(rebuilt: dict) -> None:
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    for record in rebuilt["candidate_records"]:
        assert record["solar_bundle"] == {
            "generated": False,
            "status": "sealed",
            "reason": "candidate_specific_Solar_prediction_bundle_outside_this_formal_audit",
        }
