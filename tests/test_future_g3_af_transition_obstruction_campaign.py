from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_af_transition_obstruction_campaign import (
    FIRST_BLOCKER,
    _principal_certificate,
    _sha,
    build_future_g3_af_transition_obstruction_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "future_g3_af_transition_obstruction_campaign.json"
ARTIFACT = ROOT / "runs" / "engine" / "future-g3-af-transition-obstruction-campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_af_transition_obstruction_campaign(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "c56768424a08f3666be10bb7a2e3c8c56e852f6629ae5cd3d8f7347b4ff6a375"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "18d8be9e4570a5117af0cd071f92e9d03252bcd65b1aaab08bc2dfdbc690b173"
    )


def test_symbolic_effective_metric_control_is_exact(rebuilt: dict) -> None:
    control = rebuilt["symbolic_profile_identity_control"]
    assert control["status"] == "pass"
    assert set(control["exact_residuals"]) == {"0"}
    assert control["effective_metric"] == {
        "P00": "-(1+3*beta^2*X^2)",
        "P0r": "-2*beta*d_v_d_r",
        "Pij": "(1-beta^2*X^2)*delta_ij",
    }
    stress = rebuilt["symbolic_g3_stress_center_control"]
    assert stress["status"] == "pass"
    assert stress["center_substitution"]["T_mu_nu_G3"] == "0"
    assert stress["center_substitution"]["nabla_mu_X"] == "0"


def test_smooth_profiles_reach_AF_with_finite_canonical_tail(rebuilt: dict) -> None:
    domain_hashes = set()
    for record in rebuilt["candidate_records"]:
        domain = record["AF_transition_domain_certificate"]
        profile = record["radial_profile_certificate"]
        assert domain["candidate_id"] == record["candidate_id"]
        assert domain["action_sha256"] == record["action_sha256"]
        assert domain["direct_action_bound_registration"] is True
        assert domain["method_control_domain_reused"] is False
        domain_hashes.add(domain["content_sha256"])
        assert profile["endpoint_and_falloff"] == {
            "X_at_r_zero": "1/2",
            "X_strictly_positive_at_finite_r": True,
            "X_limit_at_infinity": "0",
            "v": "L^2/r^2+O(r^-6)",
            "X": "L^4/(2*r^4)+O(r^-8)",
            "d_v_d_r": "-2*L^2/r^3+O(r^-7)",
        }
        maximum = profile["global_derivative_maximum_proof"]
        assert maximum["unique_positive_stationary_point"] == "r=L"
        assert maximum["maximum_abs_d_v_d_r"] == "1/(sqrt(2)*L)"
        assert profile["canonical_G2_energy_diagnostic"]["exact"] == ("500000*sqrt(2)*pi**2")
        assert profile["canonical_G2_energy_diagnostic"]["finite"] is True
        assert "not the complete constrained" in profile["canonical_G2_energy_diagnostic"]["scope"]
    assert len(domain_hashes) == 3


def test_candidate_specific_AF_principal_and_cone_margins(rebuilt: dict) -> None:
    expected = {
        "33/4000": (
            "63998911/64000000",
            "1089/80000000000",
            "-99967/100000",
            "99967/100000",
        ),
        "17/2000": (
            "15999711/16000000",
            "289/20000000000",
            "-49983/50000",
            "49983/50000",
        ),
        "9/1000": (
            "3999919/4000000",
            "81/5000000000",
            "-24991/25000",
            "24991/25000",
        ),
    }
    for record in rebuilt["candidate_records"]:
        principal = record["principal_common_cone_certificate"]
        bounds = principal["uniform_bounds_on_zero_less_equal_X_less_equal_one_half"]
        spatial, norm_squared, cone_upper, separation = expected[record["beta"]]
        assert bounds["P00_upper"] == "-1"
        assert bounds["common_time_covector_margin"] == "1"
        assert bounds["spatial_eigenvalue_lower"] == spatial
        assert bounds["time_space_norm_upper_squared"] == norm_squared
        assert bounds["characteristic_discriminant_lower"] == spatial
        assert bounds["slicing_cone_polynomial_upper"] == cone_upper
        assert bounds["slicing_cone_separation"] == separation
        assert principal["direct_candidate_recompute"] is True
        assert principal["prior_candidate_bound_reused"] is False
        assert principal["status"].endswith("including_X_limit_zero")


def test_flat_reference_is_rejected_without_rejecting_theory(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        audit = record["flat_reference_constraint_audit"]
        center = audit["center_jet_r_equals_zero"]
        assert center["rho_canonical_G2"] == "1/2"
        assert center["rho_cubic_G3"] == "0"
        assert center["Hamiltonian_constraint_residual_LHS_minus_2rho"] == "-1"
        assert audit["status"] == "reject_flat_reference_as_constraint_datum"
        assert audit["theory_rejected"] is False
        assert (
            audit["cubic_stress_center_control_sha256"]
            == (rebuilt["symbolic_g3_stress_center_control"]["content_sha256"])
        )
        assert "does not rule out a conformally deformed" in audit["scope"]
        assert record["theory_rejected"] is False
        assert record["negative_energy_counterexample_found"] is False


def test_annulus_modes_exactly_obstruct_bounded_unitary_inverse(rebuilt: dict) -> None:
    expected_n2 = {
        "33/4000": "8192003267/524288000000",
        "17/2000": "2048000867/131072000000",
        "9/1000": "512000243/32768000000",
    }
    for record in rebuilt["candidate_records"]:
        obstruction = record["unitary_lapse_obstruction"]
        assert obstruction["specialization_residual"] == "0"
        assert obstruction["direct_candidate_specialization"] is True
        assert obstruction["periodic_inverse_reused"] is False
        assert obstruction["pointwise_properties"]["limit_at_infinity"] == "0"
        sequence = obstruction["normalized_annulus_sequence"]
        assert sequence["bound_at_n_equals_2"] == expected_n2[record["beta"]]
        assert Fraction(sequence["bound_at_n_equals_2"]) > 0
        assert sequence["bound_limit"] == "0"
        assert sequence["conclusion"] == ("zero_in_approximate_spectrum_and_no_bounded_L2_inverse")
        assert obstruction["status"] == (
            "exact_obstruction_to_bounded_global_unitary_Delta_N_inverse"
        )
        assert "not a ghost" in obstruction["scope"]


def test_remaining_blockers_counts_and_seals_are_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["AF_decaying_gradient_profile_pass_count"] == 3
    assert rebuilt["AF_principal_common_cone_profile_pass_count"] == 3
    assert rebuilt["flat_reference_constraint_ansatz_reject_count"] == 3
    assert rebuilt["AF_unitary_lapse_Dirac_pass_count"] == 0
    assert rebuilt["AF_Einstein_constraint_solution_pass_count"] == 0
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
        assert record["full_formal_pass"] is False
        assert record["observational_data_opened"] is False


def test_action_domain_source_and_failed_cone_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_af_transition_obstruction_campaign(action, ROOT)

    domain = copy.deepcopy(config)
    domain["targets"][0]["transition_domain"]["transition_length_L"] = "10"
    with pytest.raises(ValueError, match="transition domain changed"):
        build_future_g3_af_transition_obstruction_campaign(domain, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_af_transition_obstruction_campaign(source, ROOT)

    with pytest.raises(ValueError, match="principal/common-cone bound failed"):
        _principal_certificate(Fraction(1), Fraction(1))
