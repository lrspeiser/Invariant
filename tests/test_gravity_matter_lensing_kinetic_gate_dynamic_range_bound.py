from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_matter_lensing_kinetic_gate_dynamic_range_bound as bound

ROOT = Path(__file__).resolve().parents[1]


def test_exact_design_values() -> None:
    assert bound.maximum_ratio(1.0) == pytest.approx(2.44140625)
    assert bound.maximum_ratio(0.5) == pytest.approx(5.0625)
    assert bound.maximum_initial_slope(1.0e4) == pytest.approx(1.0 / 36.0)
    assert bound.maximum_initial_slope(1.0e8) == pytest.approx(1.0 / 396.0)


def test_inverse_relation() -> None:
    for q0 in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
        assert bound.maximum_initial_slope(bound.maximum_ratio(q0)) == pytest.approx(q0)


def test_comparison_solution_grows_before_blowup() -> None:
    q0 = 0.25
    midpoint_ratio = bound.maximum_ratio(q0) ** 0.5
    assert bound.comparison_q(q0, midpoint_ratio) > q0
    assert bound.comparison_z_ratio(q0, midpoint_ratio) > 1.0
    with pytest.raises(bound.KineticGateDynamicRangeError):
        bound.comparison_q(q0, bound.maximum_ratio(q0))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_bad_slopes_fail_closed(bad: float) -> None:
    with pytest.raises(bound.KineticGateDynamicRangeError):
        bound.maximum_ratio(bad)


def test_receipt_is_deterministic_and_restrictive() -> None:
    first = bound.build_receipt(ROOT)
    second = bound.build_receipt(ROOT)
    assert first == second
    assert first["machine_checks_passed"] == 9
    assert first["content_sha256"] == bound._self_hash(first)
    assert first["claim_boundary"]["quantitative_corollary_machine_verified"] is True
    assert first["claim_boundary"]["unconditional_action_no_go"] is False
    assert first["claim_boundary"]["publication_ready"] is False


def test_config_mutation_rejected() -> None:
    config = copy.deepcopy(bound.load_config(ROOT))
    config["claim_boundary"]["publication_ready"] = True
    with pytest.raises(bound.KineticGateDynamicRangeError, match="semantics changed"):
        original = bound._read_json
        try:
            bound._read_json = lambda _path: config
            bound.load_config(ROOT)
        finally:
            bound._read_json = original


def test_predecessor_hashes_and_content_are_bound() -> None:
    config = bound.load_config(ROOT)
    for role in ("config", "module", "test", "receipt"):
        path = ROOT / config["predecessor"][f"{role}_path"]
        assert bound._sha256_file(path) == config["predecessor"][f"{role}_sha256"]
