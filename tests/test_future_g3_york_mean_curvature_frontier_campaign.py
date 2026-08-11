from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_g3_york_mean_curvature_frontier_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_york_mean_curvature_frontier_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "future_g3_york_mean_curvature_frontier_campaign.json"
ARTIFACT = (
    ROOT / "runs" / "engine" / "future-g3-york-mean-curvature-frontier-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_york_mean_curvature_frontier_campaign(
        _load(CONFIG), ROOT
    )


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "be660d9cb540ef02d888b1413eee928335ba0e3f3cc31a82184538985cb82fc2"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "55478aef6bf81449ddad151c5516e5e8e71901c1b142ae4cbbd32e728524fe69"
    )


def test_each_candidate_extends_strictly_beyond_six_fifths(rebuilt: dict) -> None:
    expected = {
        "33/4000": ("1211/1000", "11/1000", "303/250"),
        "17/2000": ("1211/1000", "11/1000", "303/250"),
        "9/1000": ("121/100", "1/100", "1211/1000"),
    }
    for record in rebuilt["candidate_records"]:
        frontier = record["York_mean_curvature_frontier_certificate"][
            "exact_frontier"
        ]
        cap, extension, next_cap = expected[record["beta"]]
        assert frontier["predecessor_common_kappa_cap"] == "6/5"
        assert frontier["candidate_kappa_cap"] == cap
        assert frontier["strict_extension_beyond_predecessor"] == extension
        assert Fraction(cap) > Fraction(6, 5)
        assert frontier["grid_step"] == "1/1000"
        assert frontier["next_grid_cap"] == next_cap
        assert Fraction(next_cap) == Fraction(cap) + Fraction(1, 1000)


def test_exact_frontier_closes_and_next_grid_point_is_only_inconclusive(
    rebuilt: dict,
) -> None:
    expected = {
        "33/4000": (
            "14027/6000000",
            "14027/57600",
            "1163551/7200000",
            "-5027/600000",
        ),
        "17/2000": (
            "5197/3000000",
            "5197/28800",
            "354713/3600000",
            "-21451/300000",
        ),
        "9/1000": (
            "323/150000",
            "323/1440",
            "128147/900000",
            "-24853/900000",
        ),
    }
    for record in rebuilt["candidate_records"]:
        certificate = record["York_mean_curvature_frontier_certificate"]
        frontier = certificate["exact_frontier"]
        factor, coefficient, excess, next_excess = expected[record["beta"]]
        assert frontier["source_factor_at_cap"] == factor
        assert frontier["required_source_factor_for_strict_no_go"] == (
            "1536/1953125"
        )
        assert frontier["green_ball_coefficient_at_cap"] == coefficient
        assert frontier["universal_allowed_green_coefficient"] == "256/3125"
        assert frontier["strict_green_excess_at_cap"] == excess
        assert Fraction(excess) > 0
        assert frontier["green_excess_at_next_grid_cap"] == next_excess
        assert Fraction(next_excess) <= 0
        assert frontier["source_factor_monotonicity_for_nonnegative_kappa"] == (
            "d_source_factor/d_kappa=-2*beta-(4/3)*kappa<0"
        )
        assert frontier["cap_is_largest_certified_point_on_declared_grid"] is True
        assert frontier["next_grid_cap_comparison_status"] == "inconclusive"
        assert certificate["decision"] == (
            "reject_expanded_candidate_millicap_York_class"
        )
        assert certificate[
            "candidate_nontrivial_AF_Einstein_constraint_solution_available"
        ] is False
        assert certificate["theory_rejected"] is False


def test_counts_scope_and_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["candidate_millicap_frontier_registration_pass_count"] == 3
    assert rebuilt["strict_extension_beyond_kappa_6_over_5_pass_count"] == 3
    assert rebuilt["expanded_nonradial_York_class_reject_count"] == 3
    assert rebuilt["next_grid_cap_inconclusive_count"] == 3
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
    for record in rebuilt["candidate_records"]:
        certificate = record["York_mean_curvature_frontier_certificate"]
        assert certificate["direct_action_binding"] is True
        assert certificate["family_label_used_as_frontier_evidence"] is False
        assert "not evidence for existence" in certificate["scope"]
        assert record["theory_rejected"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_action_cap_contract_predecessor_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_york_mean_curvature_frontier_campaign(action, ROOT)

    cap = copy.deepcopy(config)
    cap["targets"][0]["kappa_cap"] = "6/5"
    with pytest.raises(ValueError, match="millicap frontier did not certify"):
        build_future_g3_york_mean_curvature_frontier_campaign(cap, ROOT)

    contract = copy.deepcopy(config)
    contract["frontier_contract"]["kappa_grid_step"] = "1/100"
    with pytest.raises(ValueError, match="frontier contract changed"):
        build_future_g3_york_mean_curvature_frontier_campaign(contract, ROOT)

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_york_mean_curvature_frontier_campaign(predecessor, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_york_mean_curvature_frontier_campaign(source, ROOT)
