from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "engine"
    / "future-g3-nonradial-york-bounded-mean-curvature-no-go-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign(
        _load(CONFIG), ROOT
    )


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "3f366eeb681396f8ae2056f4414b31f3d03794473d27ac5f150a8e16f9b9ad0f"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "dec17893b285171a166203037a7bc74867955412f0e54a184107b9c3db33f8f2"
    )


def test_york_hamiltonian_reduction_keeps_tracefree_sign_exact(rebuilt: dict) -> None:
    control = rebuilt["symbolic_York_Hamiltonian_control"]
    assert control["constraint_normalization"] == (
        "R3+K^2-K_ij*K^ij=2*rho"
    )
    assert control["tracefree_norm_identity"] == (
        "K_ij*K^ij=A_ij*A^ij+K^2/3"
    )
    assert control["candidate_matter_density"] == "rho=v^2/2+beta*K*v^3"
    assert control["required_scalar_curvature"] == (
        "R3=v^2+2*beta*K*v^3-(2/3)*K^2+A_ij*A^ij"
    )
    assert control["reduced_equation"] == (
        "-Delta_delta(psi)=q(x)*psi(x)^5"
    )
    assert control["status"] == "pass"


def test_nonradial_green_comparison_does_not_assume_radial_fields(rebuilt: dict) -> None:
    control = rebuilt["universal_nonradial_green_control"]
    assert control["positivity_consequence"] == "psi(x)>=1"
    assert control["ball_kernel_bound"] == (
        "1/|x-z|>=1/(2*L)_for_x,z_in_B_L"
    )
    assert control["necessary_inequality"] == "m-1>=B_L*m^5"
    assert control["B_L_definition"] == "B_L=q_lower_on_B_L*L^2/6"
    assert control["universal_allowed_B_L_upper"] == "256/3125"
    assert control["unique_maximizer"] == "m=5/4"
    assert "No radial symmetry" in control["scope"]
    assert control["status"] == "pass"


def test_all_candidate_source_margins_and_green_obstructions_close(rebuilt: dict) -> None:
    expected = {
        "33/4000": ("101/5000", "101/80000", "101/48", "303337/150000"),
        "17/2000": ("49/2500", "49/40000", "49/24", "146981/75000"),
        "9/1000": ("23/1250", "23/20000", "23/12", "68803/37500"),
    }
    for record in rebuilt["candidate_records"]:
        certificate = record["nonradial_York_no_go_certificate"]
        factor, q_lower, green, excess = expected[record["beta"]]
        source = certificate["exact_source_bound"]
        comparison = certificate["exact_nonradial_green_comparison"]
        beta = Fraction(record["beta"])
        assert Fraction(factor) == (
            1 - 2 * beta * Fraction(6, 5) - Fraction(2, 3) * Fraction(6, 5) ** 2
        )
        assert source["mean_curvature_cap_kappa"] == "6/5"
        assert source["global_source_factor"] == factor
        assert source["q_lower_on_B_L"] == q_lower
        assert comparison["green_ball_coefficient_B_L_lower"] == green
        assert comparison["strict_excess"] == excess
        assert Fraction(green) > Fraction(256, 3125)
        assert comparison["B_L_lower_exceeds_universal_upper"] is True
        assert comparison[
            "positive_AF_conformal_factor_exists_in_declared_class"
        ] is False


def test_scope_rejects_ansatz_class_but_not_candidates(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["nonradial_York_no_go_certificate"]
        assert certificate["direct_action_binding"] is True
        assert certificate[
            "family_label_used_as_constraint_or_no_go_evidence"
        ] is False
        assert certificate["decision"] == (
            "reject_conformally_flat_bounded_mean_curvature_York_class"
        )
        assert certificate["momentum_constraint_status"] == (
            "not_reached_because_Hamiltonian_constraint_has_no_solution_in_class"
        )
        assert certificate[
            "candidate_nontrivial_AF_Einstein_constraint_solution_available"
        ] is False
        assert certificate["theory_rejected"] is False
        assert "non-conformally-flat" in certificate["scope"]
        assert record["theory_rejected"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_counts_blocker_and_data_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["nonradial_York_Hamiltonian_reduction_pass_count"] == 3
    assert rebuilt["bounded_mean_curvature_green_comparison_pass_count"] == 3
    assert rebuilt[
        "conformally_flat_bounded_mean_curvature_York_class_reject_count"
    ] == 3
    assert rebuilt["momentum_constraint_solution_pass_count"] == 0
    assert rebuilt[
        "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
    ] == 0
    assert rebuilt["theory_reject_count"] == 0
    assert rebuilt["global_hamiltonian_energy_pass_count"] == 0
    assert rebuilt["full_formal_pass_count"] == 0
    assert rebuilt["first_blocker_counts"] == {FIRST_BLOCKER: 3}
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["synthetic_fixture_role"] == "none_used"
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_action_domain_predecessor_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign(
            action, ROOT
        )

    domain = copy.deepcopy(config)
    domain["domain_contract"]["mean_curvature_bound"] = "abs(K)<=2*v"
    with pytest.raises(ValueError, match="nonradial York domain contract changed"):
        build_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign(
            domain, ROOT
        )

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign(
            predecessor, ROOT
        )

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_nonradial_york_bounded_mean_curvature_no_go_campaign(
            source, ROOT
        )
