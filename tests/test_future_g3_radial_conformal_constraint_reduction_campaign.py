from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_radial_conformal_constraint_reduction_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_radial_conformal_constraint_reduction_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "future_g3_radial_conformal_constraint_reduction_campaign.json"
ARTIFACT = (
    ROOT
    / "runs"
    / "engine"
    / "future-g3-radial-conformal-constraint-reduction-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_radial_conformal_constraint_reduction_campaign(
        _load(CONFIG), ROOT
    )


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "468500013c61dabda396a39ede9698c354311b6841cdf8ba1180cc997a1ed47d"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "729e8cf6434bf3c455c8aa8d37cfaa7391be08daf2e903b8fe4533bccb48d3cf"
    )


def test_symbolic_stress_momentum_and_hamiltonian_reduction_is_exact(rebuilt: dict) -> None:
    control = rebuilt["symbolic_constraint_reduction_control"]
    assert control["status"] == "pass"
    assert control["extrinsic_curvature_convention"] == (
        "K_ij=(dot(h)_ij-L_shift(h)_ij)/(2*N)"
    )
    assert control["stress_projections"] == {
        "rho_G2": "v**2/2",
        "rho_G3": "K*beta*v**3",
        "T_n_r_G3": "-beta*d_v*v**2",
    }
    assert control["pure_trace_momentum_solution"] == "K(r)=-(beta/2)*v(r)^3"
    assert control["momentum_residual_after_solution"] == "0"
    assert control["Hamiltonian_residual"] == (
        "(6*R3 + 7*beta**2*v**6 - 6*v**2)/6"
    )
    assert control["required_scalar_curvature"] == "-7*beta**2*v**6/6 + v**2"


def test_candidate_bound_exact_margins_and_momentum_solutions(rebuilt: dict) -> None:
    expected = {
        "33/4000": ("-33/8000", "31997459/32000000", "31997459/256000000"),
        "17/2000": ("-17/4000", "23997977/24000000", "23997977/192000000"),
        "9/1000": ("-9/2000", "1999811/2000000", "1999811/16000000"),
    }
    for record in rebuilt["candidate_records"]:
        certificate = record["constraint_reduction_certificate"]
        center_k, curvature_margin, q_margin = expected[record["beta"]]
        assert certificate["candidate_id"] == record["candidate_id"]
        assert certificate["action_sha256"] == record["action_sha256"]
        assert certificate["direct_action_binding"] is True
        assert certificate["family_label_used_as_constraint_evidence"] is False
        momentum = certificate["exact_momentum_constraint"]
        assert momentum["K_at_center"] == center_k
        assert momentum["residual"] == "0"
        assert momentum["K_falloff"] == "-(beta/2)*L^6/r^6+O(r^-10)"
        assert momentum["status"] == "pass_exact_radial_momentum_constraint_reduction"
        hamiltonian = certificate["exact_Hamiltonian_constraint"]
        assert hamiltonian["global_positive_factor_lower"] == curvature_margin
        assert hamiltonian["required_scalar_curvature_at_center"] == curvature_margin
        assert hamiltonian["flat_spatial_metric_residual_strictly_negative_at_finite_r"]
        assert hamiltonian["flat_pure_trace_completion_status"] == "reject_exact_ansatz"
        assert certificate["radial_Lichnerowicz_BVP"][
            "Q_beta_over_v_squared_lower"
        ] == q_margin


def test_radial_bvp_is_registered_but_not_misreported_as_a_solution(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["constraint_reduction_certificate"]
        bvp = certificate["radial_Lichnerowicz_BVP"]
        assert bvp["equation"] == (
            "psi_double_prime(r)+2*psi_prime(r)/r+Q_beta(r)*psi(r)^5=0"
        )
        assert bvp["Q_beta"] == (
            "(1/8)*v(r)^2*(1-(7/6)*beta^2*v(r)^4)"
        )
        assert bvp["Q_beta_strictly_positive_at_finite_r"] is True
        assert bvp["source_integrable_on_R3"] is True
        assert bvp["boundary_conditions"] == [
            "psi(r)>0",
            "psi_prime(0)=0",
            "psi(infinity)=1",
        ]
        assert bvp["positive_global_solution_proved"] is False
        assert bvp["status"] == "blocked_at_positive_global_BVP_solution"
        assert certificate[
            "candidate_nontrivial_AF_Einstein_constraint_solution_available"
        ] is False
        assert certificate["theory_rejected"] is False
        assert record["theory_rejected"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_counts_blocker_and_seals_are_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["radial_pure_trace_momentum_constraint_reduction_pass_count"] == 3
    assert rebuilt["positive_Hamiltonian_source_registration_pass_count"] == 3
    assert rebuilt["flat_pure_trace_completion_ansatz_reject_count"] == 3
    assert rebuilt["radial_Lichnerowicz_BVP_registration_pass_count"] == 3
    assert rebuilt["positive_global_radial_Lichnerowicz_solution_pass_count"] == 0
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


def test_action_ansatz_predecessor_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_radial_conformal_constraint_reduction_campaign(action, ROOT)

    ansatz = copy.deepcopy(config)
    ansatz["radial_conformal_ansatz"]["transition_length_L"] = "10"
    with pytest.raises(ValueError, match="ansatz contract changed"):
        build_future_g3_radial_conformal_constraint_reduction_campaign(ansatz, ROOT)

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_radial_conformal_constraint_reduction_campaign(predecessor, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_radial_conformal_constraint_reduction_campaign(source, ROOT)
