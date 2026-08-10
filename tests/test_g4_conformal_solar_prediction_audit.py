from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.g4_conformal_solar_prediction_audit import (
    _sha,
    _validate_synthetic_source,
    build_g4_conformal_solar_prediction_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g4_conformal_solar_prediction_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g4-conformal-solar-prediction-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g4_conformal_solar_prediction_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "0036cf8501ae16fe31587f4609ec3872341438453e235a8140b07f8bfa8a7bb4"
    )


def test_exact_einstein_frame_coupling_is_candidate_specific(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0]["coupling_and_PPN_certificate"]
    phi = sp.Symbol("phi")
    alpha = sp.sympify(certificate["matter_coupling"]["alpha(phi)=d_ln_A/d_chi"])
    assert sp.simplify(alpha + phi / sp.sqrt(2500 + 56 * phi**2)) == 0
    assert certificate["matter_coupling"]["alpha_0_at_phi_infinity_zero"] == "0"
    assert certificate["matter_coupling"]["beta_0=d_alpha/d_chi"] == "-1/50"
    assert certificate["scope"].startswith("massless scalar-tensor weak-field expansion")


def test_newtonian_and_ppn_predictions_pass_on_declared_background(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0]["coupling_and_PPN_certificate"]
    newton = certificate["Newtonian_prediction"]
    ppn = certificate["PPN_prediction"]
    assert newton["G_cav_over_G_star"] == "1"
    assert newton["linear_scalar_source_alpha_0_T"] == "0"
    assert newton["exterior_potential"] == "U=G_star*M/r"
    assert ppn["gamma_minus_one"] == "0"
    assert ppn["gamma"] == "1"
    assert ppn["beta_minus_one"] == "0"
    assert ppn["beta"] == "1"


def test_exact_scalar_free_branch_reduces_to_gr_without_calibration_inference(
    rebuilt: dict,
) -> None:
    record = rebuilt["candidate_records"][0]
    branch = record["exact_scalar_free_branch_certificate"]
    assert set(branch["exact_field_equation_residuals"].values()) == {"0"}
    assert branch["vacuum_exterior"] == "Schwarzschild_is_exact_on_this_branch"
    assert "not uniqueness" in branch["branch_selection_warning"]
    calibration = record["GR_calibration_control"]
    assert calibration["role"] == "solver_calibration_control_not_candidate_evidence"
    assert set(calibration["statuses"].values()) == {"pass"}
    assert len(calibration["statuses"]) == 5
    assert "independently derived" in calibration["candidate_inference_rule"]


def test_synthetic_weak_source_control_does_not_claim_real_solar_uniqueness(
    rebuilt: dict,
) -> None:
    certificate = rebuilt["candidate_records"][0]["synthetic_uniform_sphere_certificate"]
    radial = certificate["uniform_sphere_radial_parameter"]
    assert radial["z_squared=3*abs(beta_0)*G_star*M/(R*c^2)"] == "3/50000"
    assert radial["first_zero_mode_threshold"] == "z=pi/2"
    assert certificate["linear_scalar_free_branch_unique"] is True
    assert certificate["status"] == "pass_synthetic_known_answer_only"
    assert "does not certify the real Sun" in certificate["non_extension"]


def test_real_solar_bundle_remains_inadmissible_before_data_opening(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    admissibility = record["real_solar_admissibility"]
    assert admissibility["admissible"] is False
    assert admissibility["decision"] == "blocked"
    assert admissibility["first_missing_premise"] == (
        "registered_candidate_specific_action_bound_Solar_bundle"
    )
    assert admissibility["current_evaluator_result"]["decision"] == "blocked"
    assert admissibility["current_evaluator_result"]["blocker"] == (
        "missing_exact_action_bound_solar_control_bundle"
    )
    assert admissibility["GR_control_bundle_reuse_negative"] == {
        "rejected": True,
        "reason": "Solar known-answer bundle cannot be attached to a discovery candidate",
    }
    assert admissibility["candidate_use_authorized"] is False
    assert admissibility["dataset_ready"] is False
    assert admissibility["primary_files_downloaded"] is False
    assert record["solar_bundle"] == {
        "analytic_known_answer_bundle_generated": True,
        "real_observational_bundle_generated": False,
        "real_observational_bundle_admissible": False,
        "status": "blocked_before_data_opening",
    }


def test_counts_and_fail_closed_seals_are_exact(rebuilt: dict) -> None:
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert rebuilt["gate_status_counts"] == {"pass": 6, "blocked": 2}
    assert rebuilt["calibration_control_status_counts"] == {"pass": 5}
    assert rebuilt["analytic_known_answer_bundle_count"] == 1
    assert rebuilt["real_solar_bundle_count"] == 0
    assert rebuilt["real_solar_bundle_admissible_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_action_and_synthetic_source_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(config)
    tampered["target"]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="formal G4 target identity"):
        build_g4_conformal_solar_prediction_audit(tampered, ROOT)

    source = copy.deepcopy(config["synthetic_source_contract"])
    source["role"] = "real_Solar_evidence"
    with pytest.raises(ValueError, match="synthetic source contract changed"):
        _validate_synthetic_source(source)
