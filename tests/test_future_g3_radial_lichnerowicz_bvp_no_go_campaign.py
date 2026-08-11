from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_radial_lichnerowicz_bvp_no_go_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_radial_lichnerowicz_bvp_no_go_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "future_g3_radial_lichnerowicz_bvp_no_go_campaign.json"
ARTIFACT = (
    ROOT
    / "runs"
    / "engine"
    / "future-g3-radial-lichnerowicz-bvp-no-go-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_radial_lichnerowicz_bvp_no_go_campaign(
        _load(CONFIG), ROOT
    )


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "0242110785e5475469b05597e06a1965926b6451b21d8f7babc1cd7418db8f9a"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "a207492d5848e5a2e01b0344903d1110ac901a18a75944807c88bcc8327b0d12"
    )


def test_universal_scalar_inequality_is_sharp(rebuilt: dict) -> None:
    control = rebuilt["universal_scalar_inequality_control"]
    assert control["function"] == "f(y)=(y-1)/y^5"
    assert control["derivative"] == "-(4*y - 5)/y**6"
    assert control["domain"] == "y>=1"
    assert control["unique_interior_maximizer"] == "5/4"
    assert control["exact_global_maximum"] == "256/3125"
    assert control["endpoint_values"] == {
        "f(1)": "0",
        "limit_y_to_infinity": "0",
    }
    assert control["status"] == "pass"


def test_all_three_candidate_comparison_obstructions_close_exactly(rebuilt: dict) -> None:
    expected = {
        "33/4000": (
            "31997459/32000000",
            "31997459/153600",
            "3998109511/19200000",
        ),
        "17/2000": (
            "23997977/24000000",
            "23997977/115200",
            "2998567477/14400000",
        ),
        "9/1000": (
            "1999811/2000000",
            "1999811/9600",
            "249878071/1200000",
        ),
    }
    for record in rebuilt["candidate_records"]:
        certificate = record["radial_Lichnerowicz_no_go_certificate"]
        margin, a_lower, excess = expected[record["beta"]]
        assert certificate["candidate_id"] == record["candidate_id"]
        assert certificate["action_sha256"] == record["action_sha256"]
        assert certificate["direct_action_binding"] is True
        assert certificate["family_label_used_as_existence_or_no_go_evidence"] is False
        bounds = certificate["exact_candidate_bounds"]
        assert bounds["one_minus_braiding_factor_lower"] == margin
        assert bounds["Q_beta_lower_on_0_to_L"] == str(Fraction(margin) / 16)
        assert bounds["A_L_lower"] == a_lower
        assert bounds["universal_allowed_A_L_upper"] == "256/3125"
        assert bounds["strict_excess"] == excess
        assert Fraction(a_lower) > Fraction(256, 3125)
        assert bounds["A_L_lower_exceeds_universal_upper"] is True


def test_monotonicity_tail_comparison_and_scope_are_explicit(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["radial_Lichnerowicz_no_go_certificate"]
        assert certificate["radial_monotonicity_identity"] == {
            "mass_function": "M(r)=integral_0^r s^2*Q_beta(s)*psi(s)^5 ds",
            "identity": "-r^2*psi_prime(r)=M(r)",
            "consequences": ["psi_prime(r)<=0", "psi(r)>=psi(infinity)=1"],
        }
        comparison = certificate["comparison_at_R_equals_L"]
        assert comparison["exact_tail_identity"] == (
            "psi(L)-1=integral_L^infinity M(t)/t^2 dt"
        )
        assert comparison["tail_lower_bound"] == "psi(L)-1>=M(L)/L"
        assert comparison["combined_necessary_inequality"] == (
            "y-1>=A_L*y^5_for_y=psi(L)>=1"
        )
        assert certificate["decision"] == "reject_radial_conformal_pure_trace_ansatz"
        assert certificate["positive_global_solution_exists_in_declared_class"] is False
        assert certificate["theory_rejected"] is False
        assert record["radial_conformal_pure_trace_ansatz_rejected"] is True
        assert record[
            "candidate_nontrivial_AF_Einstein_constraint_solution_available"
        ] is False
        assert record["theory_rejected"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_counts_blocker_and_seals_are_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["exact_comparison_inequality_pass_count"] == 3
    assert rebuilt["positive_global_radial_Lichnerowicz_solution_pass_count"] == 0
    assert rebuilt[
        "positive_global_radial_Lichnerowicz_solution_nonexistence_count"
    ] == 3
    assert rebuilt["radial_conformal_pure_trace_ansatz_reject_count"] == 3
    assert rebuilt[
        "candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"
    ] == 0
    assert rebuilt["global_hamiltonian_energy_pass_count"] == 0
    assert rebuilt["full_formal_pass_count"] == 0
    assert rebuilt["first_blocker_counts"] == {FIRST_BLOCKER: 3}
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["synthetic_fixture_role"] == "none_used"
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_action_contract_predecessor_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_radial_lichnerowicz_bvp_no_go_campaign(action, ROOT)

    contract = copy.deepcopy(config)
    contract["comparison_contract"]["transition_length_L"] = "10"
    with pytest.raises(ValueError, match="comparison proof contract changed"):
        build_future_g3_radial_lichnerowicz_bvp_no_go_campaign(contract, ROOT)

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_radial_lichnerowicz_bvp_no_go_campaign(predecessor, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_radial_lichnerowicz_bvp_no_go_campaign(source, ROOT)
