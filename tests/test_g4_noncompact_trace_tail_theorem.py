from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.g4_noncompact_trace_tail_theorem import (
    _sha,
    build_g4_noncompact_trace_tail_theorem,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g4_noncompact_trace_tail_theorem.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g4-noncompact-trace-tail-theorem.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g4_noncompact_trace_tail_theorem(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "e969149879f526161f7dddd220c11583bf42d729ab329b3bb00c1c09a8c8d3cc"
    )


def test_predecessor_real_sun_blocker_is_exactly_closed(rebuilt: dict) -> None:
    predecessor = rebuilt["source_bindings"]["predecessor"]
    provenance = rebuilt["candidate_records"][0]["provenance"]
    assert predecessor["content_sha256"] == (
        "14b1e1a0dc298cf402630a8daeed19f61a11c66ba0be010eba9ebb87cee93576"
    )
    assert provenance["predecessor_provenance_sha256"] == (
        "823f56025c1e59d6c07bed85a4732a54baf97b347f897c4497596be8b47dba4e"
    )
    assert provenance["predecessor_fact_registry_sha256"] == (
        "4e896660107ee9b4c3e0b82216b99bcbe098311460efcfccd9f7acf43daeb13e"
    )


def test_anisotropic_hardy_tail_has_exact_separated_margin(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0][
        "hardy_anisotropic_tail_certificate"
    ]
    bound = certificate["combined_relative_bound"]
    eta_in = Fraction(bound["eta_interior_upper"])
    eta_out = Fraction(bound["eta_exterior_upper"])
    eta_total = Fraction(bound["eta_total_upper"])
    margin = Fraction(bound["coercive_margin_lower"])
    assert eta_in == eta_out == Fraction(81_608, 77_182_875)
    assert eta_total == Fraction(163_216, 77_182_875)
    assert margin == Fraction(77_019_659, 77_182_875)
    assert eta_in + eta_out == eta_total
    assert eta_total + margin == 1
    assert 0 < eta_total < 1
    assert "angular_structure" in certificate["tail_implication"]
    assert certificate["status"] == "pass_conditional_noncompact_tail_class"


def test_kato_bound_separates_interior_and_tail_contributions(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0][
        "kato_birman_schwinger_tail_certificate"
    ]
    bound = certificate["Birman_Schwinger_bound"]
    assert Fraction(bound["kappa_interior_upper"]) == Fraction(11, 87_500)
    assert Fraction(bound["kappa_exterior_upper"]) == Fraction(11, 87_500)
    assert Fraction(bound["kappa_flat_upper"]) == Fraction(11, 43_750)
    assert Fraction(bound["flat_coercive_margin_lower"]) == Fraction(
        43_739, 43_750
    )
    geometry = certificate["general_geometry_route"]
    assert Fraction(geometry["kappa_upper"]) == Fraction(1_111, 4_331_250)
    assert Fraction(geometry["coercive_margin_lower"]) == Fraction(
        4_330_139, 4_331_250
    )
    assert geometry["status"] == "conditional_on_global_Green_kernel_domination"


def test_decay_threshold_distinguishes_form_bound_from_af_source(rebuilt: dict) -> None:
    thresholds = rebuilt["candidate_records"][0][
        "kato_birman_schwinger_tail_certificate"
    ]["integrability_thresholds"]
    assert thresholds["Kato_trace_tail"] == "p>2"
    assert thresholds["finite_total_trace_mass_necessary_for_AF"] == "p>3"
    assert thresholds["registered_class_uses"] == "p>=4"
    assert "r^-2" in thresholds["warning"]
    assert "does not by itself certify" in thresholds["warning"]


def test_minimal_tail_facts_are_measurable_and_explicit(rebuilt: dict) -> None:
    contract = rebuilt["candidate_records"][0][
        "minimal_real_source_tail_fact_contract"
    ]
    ids = {item["id"] for item in contract["required_registered_facts"]}
    assert ids == {
        "registered_reference_radius_and_center",
        "interior_trace_density_upper",
        "exterior_trace_amplitude_upper",
        "exterior_decay_exponent_lower",
        "composition_pressure_trace_transform",
        "angular_or_resolved_tail_coverage",
        "outer_transition_or_cutoff",
        "geometry_and_boundary_domain",
    }
    assert "one-point wind density or speed" in contract["insufficient_controls"]
    assert "a steady r^-2 fit extrapolated to infinity" in contract[
        "insufficient_controls"
    ]
    assert contract["current_status"] == "missing_no_real_Sun_tail_profile_opened"


def test_theorem_passes_conditionally_but_real_sun_remains_blocked(
    rebuilt: dict,
) -> None:
    record = rebuilt["candidate_records"][0]
    assert rebuilt["theorem_pass_count"] == 1
    assert rebuilt["real_source_instantiation_pass_count"] == 0
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert rebuilt["gate_status_counts"] == {"pass": 4, "blocked": 1}
    assert record["theorem_decision"] == "pass"
    assert record["real_Sun_instantiation_decision"] == "blocked"
    assert record["overall_decision"] == "blocked"
    assert record["first_missing_premise"] == (
        "registered_trace_tail_amplitude_decay_and_outer_transition"
    )
    assert record["candidate_rejection_found"] is False
    assert record["real_solar_bundle_admissible"] is False
    assert rebuilt["observational_authorization"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["tracking_target_values_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_tail_class_and_predecessor_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tampered_tail = copy.deepcopy(config)
    tampered_tail["tail_class"]["trace_envelope"]["p_lower"] = "2"
    with pytest.raises(ValueError, match="tail class changed"):
        build_g4_noncompact_trace_tail_theorem(tampered_tail, ROOT)

    tampered_predecessor = copy.deepcopy(config)
    tampered_predecessor["predecessor"]["fact_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fact registry mismatch"):
        build_g4_noncompact_trace_tail_theorem(tampered_predecessor, ROOT)
