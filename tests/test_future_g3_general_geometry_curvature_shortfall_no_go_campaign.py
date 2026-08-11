from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_general_geometry_curvature_shortfall_no_go_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_general_geometry_curvature_shortfall_no_go_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "future_g3_general_geometry_curvature_shortfall_no_go_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "engine"
    / "future-g3-general-geometry-curvature-shortfall-no-go-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
        _load(CONFIG), ROOT
    )


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "1c06013ecad8dd73fd3a508b704c886e98ccde8c2c733eac2cd117f97807d809"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "001abc2313e0cc88b9a02c6af2043132e97baec069f891d19af001ca92b7059b"
    )


def test_pointwise_theorem_removes_conformal_flatness(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        theorem = record["general_geometry_curvature_shortfall_certificate"][
            "exact_pointwise_theorem"
        ]
        assert theorem["Hamiltonian_constraint"] == "R3=S_beta"
        assert theorem["compensated_source_lower"] == "S_beta>=c_star*v^2"
        assert theorem["finite_profile_positivity"] == (
            "v(x_star)^2>0_for_every_finite_x_star"
        )
        assert theorem["curvature_shortfall_hypothesis"] == (
            "R3(x_star)<c_star*v(x_star)^2_at_some_finite_x_star"
        )
        assert theorem["strict_residual_sign"] == "R3(x_star)-S_beta(x_star)<0"
        assert theorem[
            "Hamiltonian_constraint_solution_exists_in_declared_class"
        ] is False
        assert theorem["conformal_flatness_used"] is False
        assert theorem["Cotton_tensor_restricted"] is False


def test_exact_below_endpoint_and_above_controls(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["general_geometry_curvature_shortfall_certificate"]
        below = certificate["exact_below_threshold_control"]
        endpoint = certificate["sharp_endpoint_control"]
        above = certificate["above_threshold_negative_control"]
        assert below["R3_over_v_squared"] == "768/1953125"
        assert below["source_lower_ratio"] == "1536/1953125"
        assert below["strict_residual_upper_ratio"] == "-768/1953125"
        assert Fraction(below["strict_residual_upper_ratio"]) < 0
        assert below["decision"] == "reject_by_pointwise_Hamiltonian_residual"
        assert below["metric_constructed"] is False
        assert below["AF_constraint_solution_inferred"] is False
        assert endpoint["R3_over_v_squared"] == "1536/1953125"
        assert endpoint["Hamiltonian_residual_if_both_saturate"] == "0"
        assert endpoint["status"] == (
            "conditional_theorem_inconclusive_at_exact_endpoint"
        )
        assert endpoint["AF_constraint_solution_inferred"] is False
        assert above["R3_over_v_squared"] == "1537/1953125"
        assert above["difference_from_threshold"] == "1/1953125"
        assert above["status"] == "not_excluded_by_pointwise_lower_bound"
        assert above["AF_constraint_solution_or_action_pass_inferred"] is False


def test_candidate_binding_scope_and_blocker_are_honest(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        certificate = record["general_geometry_curvature_shortfall_certificate"]
        assert certificate["candidate_id"] == record["candidate_id"]
        assert certificate["action_sha256"] == record["action_sha256"]
        assert certificate["beta"] == record["beta"]
        assert certificate["direct_action_binding"] is True
        assert certificate["family_label_used_as_geometry_evidence"] is False
        assert certificate["decision"] == (
            "reject_general_AF_geometry_curvature_shortfall_class"
        )
        assert "nonconformally_flat" in certificate["excluded_class"]
        assert certificate[
            "candidate_nontrivial_AF_Einstein_constraint_solution_available"
        ] is False
        assert certificate["theory_rejected"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_counts_and_data_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["general_geometry_pointwise_theorem_pass_count"] == 3
    assert rebuilt["curvature_shortfall_constraint_class_reject_count"] == 3
    assert rebuilt["exact_curvature_endpoint_inconclusive_count"] == 3
    assert rebuilt["above_threshold_not_excluded_control_count"] == 3
    assert rebuilt["nonconformally_flat_metric_construction_pass_count"] == 0
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


def test_action_ratio_contract_predecessor_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
            action, ROOT
        )

    ratio = copy.deepcopy(config)
    ratio["targets"][0]["below_threshold_control_ratio"] = "1536/1953125"
    with pytest.raises(ValueError, match="curvature-ratio controls changed"):
        build_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
            ratio, ROOT
        )

    contract = copy.deepcopy(config)
    contract["general_geometry_contract"]["conformal_flatness_required"] = True
    with pytest.raises(ValueError, match="general geometry.*contract changed"):
        build_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
            contract, ROOT
        )

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
            predecessor, ROOT
        )

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_general_geometry_curvature_shortfall_no_go_campaign(
            source, ROOT
        )
