from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_general_geometry_surplus_mismatch_no_go_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_general_geometry_surplus_mismatch_no_go_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "future_g3_general_geometry_surplus_mismatch_no_go_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "engine"
    / "future-g3-general-geometry-surplus-mismatch-no-go-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
        _load(CONFIG), ROOT
    )


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "6fd80b865c02883a4d5b387573a7fdd8a13238e984edc2e61204b1b1d08019d7"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "932d0c6c76407ed786b0500c0cdb551e08e40e5ec6e04810eec71d3ec80b4e6a"
    )


def test_exact_surplus_identity_is_candidate_bound_and_geometry_general(
    rebuilt: dict,
) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["general_geometry_surplus_mismatch_certificate"]
        identity = certificate["exact_surplus_identity"]
        assert certificate["candidate_id"] == record["candidate_id"]
        assert certificate["action_sha256"] == record["action_sha256"]
        assert certificate["beta"] == record["beta"]
        assert certificate["direct_action_binding"] is True
        assert certificate["family_label_used_as_surplus_evidence"] is False
        assert identity["York_source_surplus"] == (
            "Y=A_ij*A^ij-D_beta=S_beta-c_star*v^2"
        )
        assert identity["curvature_surplus"] == "C=R3-c_star*v^2"
        assert identity["Hamiltonian_residual"] == "R3-S_beta=C-Y"
        assert identity["compensation_implies"] == "Y>=0"
        assert identity["pointwise_no_go_condition"] == "Y(x_star)>C(x_star)"
        assert identity["conformal_flatness_used"] is False
        assert identity["Cotton_tensor_restricted"] is False


def test_above_threshold_mismatch_control_is_exact_reject(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        control = record["general_geometry_surplus_mismatch_certificate"][
            "above_threshold_mismatch_control"
        ]
        assert control["curvature_ratio_R3_over_v_squared"] == "1537/1953125"
        assert control["source_ratio_S_beta_over_v_squared"] == "1538/1953125"
        assert control["curvature_surplus_ratio"] == "1/1953125"
        assert control["York_source_surplus_ratio"] == "2/1953125"
        assert control["Hamiltonian_residual_ratio"] == "-1/1953125"
        assert Fraction(control["Hamiltonian_residual_ratio"]) < 0
        assert control["decision"] == (
            "reject_by_exact_negative_Hamiltonian_residual"
        )
        assert control["metric_or_York_fields_constructed"] is False
        assert control["AF_constraint_solution_inferred"] is False


def test_matched_and_overcurvature_controls_do_not_promote(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["general_geometry_surplus_mismatch_certificate"]
        matched = certificate["matched_surplus_control"]
        over = certificate["overcurvature_negative_control"]
        assert matched["curvature_surplus_ratio"] == "1/1953125"
        assert matched["York_source_surplus_ratio"] == "1/1953125"
        assert matched["Hamiltonian_residual_ratio"] == "0"
        assert matched["status"] == "pointwise_Hamiltonian_match_only"
        assert matched["momentum_constraint_solved"] is False
        assert matched["AF_constraint_solution_inferred"] is False
        assert over["curvature_surplus_ratio"] == "3/1953125"
        assert over["York_source_surplus_ratio"] == "2/1953125"
        assert over["Hamiltonian_residual_ratio"] == "1/1953125"
        assert over["status"] == "not_excluded_by_surplus_lower_bound"
        assert over["AF_constraint_solution_or_action_pass_inferred"] is False


def test_scope_counts_and_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["exact_surplus_identity_pass_count"] == 3
    assert rebuilt["above_threshold_surplus_mismatch_class_reject_count"] == 3
    assert rebuilt["matched_surplus_necessary_control_count"] == 3
    assert rebuilt["overcurvature_not_excluded_control_count"] == 3
    assert rebuilt["registered_AF_metric_York_datum_pass_count"] == 0
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
        certificate = record["general_geometry_surplus_mismatch_certificate"]
        assert certificate["decision"] == (
            "reject_general_AF_geometry_surplus_mismatch_class"
        )
        assert certificate[
            "candidate_nontrivial_AF_Einstein_constraint_solution_available"
        ] is False
        assert certificate["theory_rejected"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_action_surplus_contract_predecessor_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
            action, ROOT
        )

    control = copy.deepcopy(config)
    control["targets"][0]["mismatch_York_surplus_ratio"] = "1/1953125"
    with pytest.raises(ValueError, match="surplus controls changed"):
        build_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
            control, ROOT
        )

    contract = copy.deepcopy(config)
    contract["surplus_contract"]["conformal_flatness_required"] = True
    with pytest.raises(ValueError, match="surplus-mismatch contract changed"):
        build_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
            contract, ROOT
        )

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
            predecessor, ROOT
        )

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_general_geometry_surplus_mismatch_no_go_campaign(
            source, ROOT
        )
