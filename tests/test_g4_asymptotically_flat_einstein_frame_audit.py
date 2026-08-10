from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.g4_asymptotically_flat_einstein_frame_audit import (
    _sha,
    _validate_domain,
    _validate_target,
    build_g4_asymptotically_flat_einstein_frame_audit,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "g4_asymptotically_flat_einstein_frame_audit.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "g4-asymptotically-flat-einstein-frame-audit.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_g4_asymptotically_flat_einstein_frame_audit(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "e96237ac8af7dc5d1a4d6a379817d23c904c7f17b19cd2de0e390296fa5be7bc"
    )


def test_conformal_scalar_map_is_global_and_has_no_field_boundary(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0]["global_conformal_certificate"]
    y = sp.Symbol("y", nonnegative=True)
    kinetic = (2500 + 56 * y) / (50 + y) ** 2
    lapse_factor = sp.factor((50 + y) * kinetic / 50)
    assert certificate["f_positive_for_all_real_phi"] is True
    assert certificate["Einstein_scalar_kinetic"] == "(56*y + 2500)/(y + 50)**2"
    assert certificate["Einstein_scalar_kinetic_positive_for_all_real_phi"] is True
    assert sp.factor(lapse_factor - 2 * (14 * y + 625) / (25 * (y + 50))) == 0
    assert sp.factor(sp.diff(lapse_factor, y) - 6 / (y + 50) ** 2) == 0
    assert certificate["lapse_multiplier_factor"]["global_interval"] == ["1", "28/25"]
    assert certificate["global_field_range"]["phi_domain"] == "R"
    assert certificate["global_field_range"]["chi_domain"] == "R"
    assert certificate["global_field_range"]["domain_preservation"] == (
        "no finite field-value boundary exists to cross"
    )


def test_weighted_falloff_preserves_ADM_charge_and_scalar_boundaries(rebuilt: dict) -> None:
    certificate = rebuilt["candidate_records"][0]["boundary_positive_mass_certificate"]
    falloff = certificate["falloff_transport"]
    assert falloff["f_minus_one"] == "phi^2/50=O(r^-2)"
    assert falloff["D_i_f"] == "O(r^-3)"
    assert falloff["chi"] == "phi+O(phi^3)=O(r^-1)"
    charge = certificate["ADM_charge_transform"]
    assert charge["energy_correction_falloff"] == "r^2*O(r^-3)->0"
    assert charge["momentum_correction_falloff"] == "r^2*O(r^-3)->0"
    assert charge["E_E_equals_E_J"] is True
    assert charge["P_E_equals_P_J"] is True
    assert certificate["scalar_and_conformal_boundaries"]["status"] == "pass"


def test_candidate_specific_maximal_positive_mass_theorem_passes(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    certificate = record["boundary_positive_mass_certificate"]
    matter = certificate["Einstein_frame_matter"]
    assert matter["dominant_energy_condition"] == "pass_for_canonical_massless_scalar"
    theorem = certificate["maximal_constraint_reduction"]
    assert theorem["riemannian_positive_mass_core_applicable"] is True
    assert theorem["E_E_nonnegative"] is True
    assert theorem["therefore_E_J_nonnegative"] is True
    assert record["gate_ledger"]["candidate_specific_maximal_positive_mass"]["status"] == "pass"
    assert record["resolved_global_energy_followup"] == "pass_on_explicit_maximal_AF_domain"
    domain = record["asymptotically_flat_domain"]
    assert domain["constraint_subdomain"]["nonempty_witness"] == (
        "Minkowski_h_E_with_K_E=0_and_phi=Pi_phi=0"
    )


def test_AF_scalar_clock_dichotomy_blocks_global_unitary_delta(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    obstruction = record["global_unitary_lapse_obstruction"]
    assert obstruction["bounded_coefficient"] == "1<=f*K_E<28/25"
    assert obstruction["AF_clock_dichotomy"]["Delta_limit"] == "0"
    assert "nondecaying scalar stress" in obstruction["AF_clock_dichotomy"][
        "alternative_phi_equals_time"
    ]
    assert obstruction["annulus_sequence"]["conclusion"] == (
        "no_bounded_global_unitary_Delta_inverse"
    )
    assert obstruction["ordinary_ADM_lapse"]["does_not_control_scalar_clock_lapse"] is True
    assert obstruction["status"] == "blocked"
    assert record["gate_ledger"]["ordinary_ADM_lapse_bounds"]["status"] == "pass"
    assert record["gate_ledger"]["global_unitary_Delta_N_inverse"]["status"] == "blocked"


def test_no_formal_pass_means_no_solar_bundle_or_observations(rebuilt: dict) -> None:
    record = rebuilt["candidate_records"][0]
    assert rebuilt["candidate_specific_positive_mass_pass_count"] == 1
    assert rebuilt["global_unitary_lapse_pass_count"] == 0
    assert rebuilt["full_formal_pass_count"] == 0
    assert rebuilt["decision_counts"] == {"blocked": 1}
    assert record["decision"] == "blocked"
    assert record["first_missing_premise"] == (
        "global_unitary_Delta_N_inverse_compatible_with_AF_scalar_falloff"
    )
    assert record["necessary_condition_rejection_found"] is False
    assert record["solar_bundle"] == {
        "generated": False,
        "status": "blocked",
        "reason": "full_formal_pass_not_proven",
    }
    assert rebuilt["solar_bundle_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_domain_and_action_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    weakened = copy.deepcopy(config["asymptotically_flat_domain"])
    weakened["boundary_contract"]["conformal_total_divergence"] = "uncontrolled"
    with pytest.raises(ValueError, match="asymptotically flat domain changed"):
        _validate_domain(weakened)

    predecessor = json.loads((ROOT / config["predecessor"]["path"]).read_text(encoding="utf-8"))
    record = predecessor["candidate_records"][0]
    target = copy.deepcopy(config["target"])
    target["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="action hash mismatch"):
        _validate_target(record, target)
