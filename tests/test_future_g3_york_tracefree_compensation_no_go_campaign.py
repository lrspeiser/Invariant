from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_york_tracefree_compensation_no_go_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_york_tracefree_compensation_no_go_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "future_g3_york_tracefree_compensation_no_go_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "engine"
    / "future-g3-york-tracefree-compensation-no-go-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_york_tracefree_compensation_no_go_campaign(
        _load(CONFIG), ROOT
    )


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "33cb23399cd5c0739ffb8e86dd59d554f09030b7918e3ff42933074db432356b"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "bf32d1aa6ee37a3666ee35ab6d3a03250afd9a14433fd100cf4326fc2b50b764"
    )


def test_exact_compensation_cases_restore_threshold_source(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["York_tracefree_compensation_certificate"]
        source = certificate["exact_compensated_source_certificate"]
        assert source["Hamiltonian_source"] == (
            "S_beta=v^2+2*beta*K*v^3-(2/3)*K^2+A_ij*A^ij"
        )
        assert source["compensation_deficit"] == (
            "D_beta=(2/3)*K^2-2*beta*K*v^3-(1-c_star)*v^2"
        )
        assert source["case_D_nonpositive"] == (
            "A_squared>=0_implies_S_beta>=c_star*v^2"
        )
        assert source["case_D_positive"] == (
            "A_squared>=D_beta_implies_S_beta>=c_star*v^2"
        )
        assert source["global_conclusion"] == "S_beta>=c_star*v^2"
        assert source["Green_coefficient"] == "L^2*c_star/96=256/3125"
        assert source[
            "positive_AF_conformal_factor_exists_in_compensated_class"
        ] is False


def test_entire_analytic_threshold_class_is_included(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        inclusion = record["York_tracefree_compensation_certificate"][
            "analytic_threshold_class_inclusion"
        ]
        assert inclusion["root_identity"] == (
            "(2/3)*kappa_star^2+2*beta*kappa_star=1-c_star"
        )
        assert inclusion["proof"] == (
            "if_abs_K<=kappa_star*v_then_D_beta<=0_so_A_squared>=0_suffices"
        )
        assert inclusion["entire_predecessor_closed_class_included"] is True


def test_controls_prove_strict_extension_without_claiming_constraint_solution(
    rebuilt: dict,
) -> None:
    expected = {
        "33/4000": ("303/250", "499647/500000", "5027/62500000"),
        "17/2000": ("303/250", "9999/10000", "21451/31250000"),
        "9/1000": ("1211/1000", "749609/750000", "24853/93750000"),
    }
    for record in rebuilt["candidate_records"]:
        witness = record["York_tracefree_compensation_certificate"][
            "strict_extension_witness_control"
        ]
        cap, a_sq, deficit = expected[record["beta"]]
        assert witness["role"] == (
            "algebraic_domain_nonemptiness_control_not_constraint_solution"
        )
        assert witness["kappa"] == cap
        assert witness["outside_predecessor_threshold"] is True
        assert witness["A_squared_at_center"] == a_sq
        assert witness["compensation_deficit_at_center"] == deficit
        assert Fraction(deficit) > 0
        assert witness["resulting_Hamiltonian_source"] == "S_beta=v^2"
        assert witness["momentum_constraint_solved"] is False
        assert witness["AF_constraint_solution_inferred"] is False


def test_undercompensated_controls_remain_inconclusive(rebuilt: dict) -> None:
    expected = {
        "33/4000": "353/500000",
        "17/2000": "1/10000",
        "9/1000": "391/750000",
    }
    for record in rebuilt["candidate_records"]:
        control = record["York_tracefree_compensation_certificate"][
            "undercompensated_negative_control"
        ]
        assert control["A_squared_at_center"] == "0"
        assert control["source_at_center"] == expected[record["beta"]]
        assert Fraction(control["source_at_center"]) < Fraction(1536, 1953125)
        assert control["source_below_c_star"] is True
        assert control["status"] == "outside_certificate_comparison_inconclusive"
        assert control["AF_solution_or_action_pass_inferred"] is False


def test_scope_counts_and_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["exact_tracefree_compensation_bound_pass_count"] == 3
    assert rebuilt["strict_domain_extension_control_pass_count"] == 3
    assert rebuilt["tracefree_compensated_York_class_reject_count"] == 3
    assert rebuilt["undercompensated_negative_control_inconclusive_count"] == 3
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
    assert rebuilt["synthetic_fixture_role"] == (
        "deterministic_symbolic_controls_only"
    )
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    for record in rebuilt["candidate_records"]:
        certificate = record["York_tracefree_compensation_certificate"]
        assert certificate["direct_action_binding"] is True
        assert certificate["family_label_used_as_compensation_evidence"] is False
        assert certificate["decision"] == (
            "reject_tracefree_compensated_conformally_flat_York_class"
        )
        assert certificate[
            "candidate_nontrivial_AF_Einstein_constraint_solution_available"
        ] is False
        assert certificate["theory_rejected"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_action_witness_contract_predecessor_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_york_tracefree_compensation_no_go_campaign(action, ROOT)

    witness = copy.deepcopy(config)
    witness["targets"][0]["witness_A_squared_at_center"] = "0"
    with pytest.raises(ValueError, match="compensation witness changed"):
        build_future_g3_york_tracefree_compensation_no_go_campaign(witness, ROOT)

    contract = copy.deepcopy(config)
    contract["compensation_contract"]["mean_curvature_scope"] = "bounded"
    with pytest.raises(ValueError, match="compensation contract changed"):
        build_future_g3_york_tracefree_compensation_no_go_campaign(contract, ROOT)

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_york_tracefree_compensation_no_go_campaign(predecessor, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_york_tracefree_compensation_no_go_campaign(source, ROOT)
