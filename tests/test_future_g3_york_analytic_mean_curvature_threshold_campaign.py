from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.future_g3_york_analytic_mean_curvature_threshold_campaign import (
    FIRST_BLOCKER,
    _sha,
    build_future_g3_york_analytic_mean_curvature_threshold_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "future_g3_york_analytic_mean_curvature_threshold_campaign.json"
ARTIFACT = (
    ROOT / "runs" / "engine" / "future-g3-york-analytic-mean-curvature-threshold-campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_g3_york_analytic_mean_curvature_threshold_campaign(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: item for key, item in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert committed["content_sha256"] == (
        "911425c0710df598caa6a6f4021620a75559f089ad7b034ccdc9887af1e6ef70"
    )
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "3566816d29783f07b7361ea7e965e3cc0d45e8b7f4bb9291d2020dc55bbabd65"
    )


def test_exact_candidate_roots_polynomials_and_isolating_intervals(rebuilt: dict) -> None:
    expected = {
        "33/4000": (
            "-99/8000 + sqrt(59958939705)/200000",
            "11991787941/2000000000",
            [62500000, 1546875, -93676272],
            ["1211/1000", "303/250"],
        ),
        "17/2000": (
            "-51/4000 + sqrt(14989829145)/100000",
            "2997965829/500000000",
            [31250000, 796875, -46838136],
            ["1211/1000", "303/250"],
        ),
        "9/1000": (
            "-27/2000 + sqrt(3747506505)/50000",
            "749501301/125000000",
            [15625000, 421875, -23419068],
            ["121/100", "1211/1000"],
        ),
    }
    kappa = sp.Symbol("kappa")
    for record in rebuilt["candidate_records"]:
        threshold = record["York_analytic_threshold_certificate"]["exact_algebraic_threshold"]
        expression, radicand, coefficients, interval = expected[record["beta"]]
        assert threshold["candidate_root_expression"] == expression
        assert threshold["radicand"] == radicand
        assert threshold["radicand_is_not_a_rational_square"] is True
        assert threshold["primitive_integer_polynomial_coefficients_descending"] == coefficients
        assert threshold["isolating_interval"] == interval
        root = sp.sympify(expression)
        polynomial = sum(
            coefficient * kappa ** (2 - index) for index, coefficient in enumerate(coefficients)
        )
        assert sp.simplify(polynomial.subs(kappa, root)) == 0
        assert sp.Rational(interval[0]) < root < sp.Rational(interval[1])
        assert threshold["source_factor_at_threshold"] == "1536/1953125"
        assert threshold["Green_coefficient_at_threshold"] == "256/3125"
        assert threshold["unique_positive_root"] is True


def test_endpoint_is_excluded_by_strict_not_coarse_equality(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        endpoint = record["York_analytic_threshold_certificate"]["endpoint_certificate"]
        assert endpoint["coarse_coefficient_relation"] == ("B_beta(kappa_star)=256/3125")
        assert endpoint["strict_source_fact"] == "v(r)^2>1/2_for_|x|<L"
        assert "almost_everywhere" in endpoint["strict_kernel_fact"]
        assert endpoint["strict_endpoint_inequality"] == (
            "m-1>(256/3125)*m^5_for_m=min_B_L(psi)>=1"
        )
        assert endpoint["contradiction"] == ("(m-1)/m^5<=256/3125_for_all_m>=1")
        assert endpoint["threshold_endpoint_excluded"] is True


def test_above_threshold_controls_are_negative_and_inconclusive(rebuilt: dict) -> None:
    expected = {
        "33/4000": ("303/250", "-5027/600000"),
        "17/2000": ("303/250", "-21451/300000"),
        "9/1000": ("1211/1000", "-24853/900000"),
    }
    for record in rebuilt["candidate_records"]:
        control = record["York_analytic_threshold_certificate"]["above_threshold_negative_control"]
        cap, excess = expected[record["beta"]]
        assert control["kappa"] == cap
        assert control["strictly_above_kappa_star"] is True
        assert control["Green_excess"] == excess
        assert Fraction(excess) < 0
        assert control["status"] == "comparison_inconclusive"
        assert control["AF_solution_or_action_pass_inferred"] is False


def test_scope_counts_and_seals_remain_fail_closed(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 3
    assert rebuilt["decision_counts"] == {"blocked": 3}
    assert rebuilt["exact_algebraic_threshold_pass_count"] == 3
    assert rebuilt["closed_threshold_endpoint_reject_count"] == 3
    assert rebuilt["above_threshold_negative_control_inconclusive_count"] == 3
    assert rebuilt["candidate_nontrivial_AF_Einstein_constraint_solution_pass_count"] == 0
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
        certificate = record["York_analytic_threshold_certificate"]
        assert certificate["direct_action_binding"] is True
        assert certificate["family_label_used_as_threshold_evidence"] is False
        assert certificate["decision"] == "reject_closed_analytic_threshold_York_class"
        assert "abs_K_less_than_or_equal_to_kappa_star" in certificate["excluded_class"]
        assert (
            certificate["candidate_nontrivial_AF_Einstein_constraint_solution_available"] is False
        )
        assert certificate["theory_rejected"] is False
        assert record["global_energy_pass"] is False
        assert record["full_formal_pass"] is False


def test_action_root_contract_predecessor_and_source_tampering_fail_closed() -> None:
    config = _load(CONFIG)

    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target binding changed"):
        build_future_g3_york_analytic_mean_curvature_threshold_campaign(action, ROOT)

    root = copy.deepcopy(config)
    root["targets"][0]["radicand"] = "1"
    with pytest.raises(ValueError, match="analytic threshold certificate changed"):
        build_future_g3_york_analytic_mean_curvature_threshold_campaign(root, ROOT)

    contract = copy.deepcopy(config)
    contract["threshold_contract"]["endpoint_semantics"] = "inconclusive"
    with pytest.raises(ValueError, match="analytic threshold contract changed"):
        build_future_g3_york_analytic_mean_curvature_threshold_campaign(contract, ROOT)

    predecessor = copy.deepcopy(config)
    predecessor["bindings"]["predecessor"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound content hash mismatch"):
        build_future_g3_york_analytic_mean_curvature_threshold_campaign(predecessor, ROOT)

    source = copy.deepcopy(config)
    source["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="campaign source hash mismatch"):
        build_future_g3_york_analytic_mean_curvature_threshold_campaign(source, ROOT)
