from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.g4_real_sun_interval_source_audit import (
    _sha,
    build_g4_real_sun_interval_source_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g4_real_sun_interval_source_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g4-real-sun-interval-source-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g4_real_sun_interval_source_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "479bc2743c7ae8c49392126d00b3590db3a838e6ec2a0b32f60421d2d8c1b98f"
    )


def test_every_fact_has_one_required_class_and_no_raw_values_opened(rebuilt: dict) -> None:
    registry = rebuilt["authoritative_fact_registry"]
    facts = registry["facts"]
    assert registry["classification_counts"] == {
        "calibrated": 6,
        "model_dependent": 4,
        "raw": 0,
    }
    assert registry["raw_values_opened"] == 0
    assert registry["tracking_target_values_opened"] == 0
    assert {item["classification"] for item in facts} == {
        "calibrated",
        "model_dependent",
    }
    assert all(item["used_to_instantiate_theorem"] is False for item in facts)


def test_nominal_compactness_is_exact_but_calibration_only(rebuilt: dict) -> None:
    diagnostic = rebuilt["calibration_and_model_diagnostics"][
        "nominal_compactness_calibration"
    ]
    assert Fraction(diagnostic["exact_fraction"]) == Fraction(
        31_598_200_000_000, 7_443_618_783_895_286_097
    )
    assert 0 < Fraction(diagnostic["exact_fraction"]) < Fraction(1, 100_000)
    assert diagnostic["status"] == "pass_calibration_only"
    assert diagnostic["theorem_evidence"] is False
    assert "conversion factors" in diagnostic["reason"]


def test_model_central_values_are_diagnostics_not_source_bounds(rebuilt: dict) -> None:
    diagnostics = rebuilt["calibration_and_model_diagnostics"]
    density = diagnostics["central_density_model_diagnostic"]
    pressure = diagnostics["central_pressure_trace_model_diagnostic"]
    assert density["below_source_class_threshold_1_over_1000"] is True
    assert density["status"] == "model_dependent_counterfactual_only"
    assert pressure["status"] == "model_dependent_single_point_only"
    assert density["theorem_evidence"] is False
    assert pressure["theorem_evidence"] is False


def test_solar_wind_prevents_photospheric_compact_support_inference(
    rebuilt: dict,
) -> None:
    assessments = {
        item["requirement_id"]: item for item in rebuilt["interval_assessments"]
    }
    support = assessments["source_support_radius_upper"]
    assert support["status"] == "blocked"
    assert support["decisive_negative_control"] == (
        "photospheric_radius_is_not_total_trace_compact_support"
    )
    assert "exterior-tail" in support["missing"]
    assert rebuilt["first_missing_premise"] == (
        "registered_finite_trace_support_or_resolved_exterior_tail_Kato_bound"
    )


def test_all_six_source_theorem_requirements_remain_blocked(rebuilt: dict) -> None:
    assessments = rebuilt["interval_assessments"]
    assert {item["requirement_id"] for item in assessments} == {
        "source_support_radius_upper",
        "total_mass_and_compactness",
        "trace_density_or_concentration_upper",
        "pressure_trace_sign",
        "static_geometry_intervals",
        "scalar_boundary_and_topology",
    }
    assert all(item["status"] == "blocked" for item in assessments)
    assert rebuilt["theorem_requirement_counts"] == {"pass": 0, "blocked": 6}
    assert rebuilt["decision"] == "blocked"
    assert rebuilt["real_source_interval_certificate_admissible"] is False
    assert rebuilt["real_solar_bundle_admissible"] is False


def test_no_circular_candidate_evidence_or_observation_opening(rebuilt: dict) -> None:
    assert set(rebuilt["no_circularity_ledger"].values()) == {"reject"}
    assert "GR_fitted_ephemeris_residual_as_truth" in rebuilt[
        "no_circularity_ledger"
    ]
    assert "standard_solar_model_as_candidate_evidence" in rebuilt[
        "no_circularity_ledger"
    ]
    assert rebuilt["candidate_independent_source_audit"] is True
    assert rebuilt["candidate_rejection_found"] is False
    assert rebuilt["observational_authorization"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["tracking_target_values_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_fact_classification_and_predecessor_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tampered_fact = copy.deepcopy(config)
    tampered_fact["authoritative_fact_registry"][0]["classification"] = "raw"
    with pytest.raises(ValueError, match="classifications changed"):
        build_g4_real_sun_interval_source_audit(tampered_fact, ROOT)

    tampered_predecessor = copy.deepcopy(config)
    tampered_predecessor["source_bindings"]["predecessor"][
        "required_certificates"
    ]["coercivity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="certificate mismatch"):
        build_g4_real_sun_interval_source_audit(tampered_predecessor, ROOT)
