from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    gravity_matter_lensing_kinetic_gate_publication_candidate as candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_dynamic_range_examples() -> None:
    assert candidate.maximum_ratio(1.0) == pytest.approx(2.44140625)
    assert candidate.maximum_ratio(0.5) == pytest.approx(5.0625)
    assert math.log10(candidate.maximum_ratio(0.01)) > 5.0


def test_shifted_power_thresholds() -> None:
    assert candidate.shifted_power_threshold(0.5) == pytest.approx(1.0)
    assert candidate.shifted_power_threshold(1.0) == pytest.approx(0.6)
    assert candidate.shifted_power_threshold(2.0) == pytest.approx(1.0 / 3.0)
    assert candidate.shifted_power_threshold(4.0) == pytest.approx(3.0 / 17.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_design_inputs_fail_closed(value: float) -> None:
    with pytest.raises(candidate.KineticGatePublicationCandidateError):
        candidate.maximum_ratio(value)
    with pytest.raises(candidate.KineticGatePublicationCandidateError):
        candidate.shifted_power_threshold(value)


def test_symbolic_proof_routes() -> None:
    assert candidate.symbolic_checks() == {
        "P02_M_IDENTITY": True,
        "P03_RICCATI_SOLUTION": True,
        "P04_FINITE_RANGE_BOUND": True,
        "P05_SHIFTED_POWER_THRESHOLD": True,
    }


def test_all_bound_bytes_match_commits() -> None:
    config = candidate.load_config(ROOT)
    for binding in config["bindings"]:
        for role in ("config", "module", "test", "receipt"):
            relative = binding[f"{role}_path"]
            expected = binding[f"{role}_sha256"]
            assert candidate._sha256_file(ROOT / relative) == expected
            assert (
                candidate._sha256_bytes(candidate._git_show(ROOT, binding["commit"], relative))
                == expected
            )


def test_receipt_is_restrictive_and_deterministic() -> None:
    first = candidate.build_receipt(ROOT)
    second = candidate.build_receipt(ROOT)
    assert first == second
    assert first["checks_passed"] == 11
    assert all(first["checks"].values())
    assert first["status"] == "PROMISING_ORIGINAL_THEOREM_CANDIDATE_NOT_PREPRINT_READY"
    assert first["claim_boundary"]["candidate_original_mathematical_result"] is True
    assert first["claim_boundary"]["historical_novelty_established"] is False
    assert first["claim_boundary"]["publication_ready"] is False
    assert first["witness_summary"]["sound_speed_squared_max"] > 1.0
    assert first["content_sha256"] == candidate._self_hash(first)
    assert first["implementation_binding"]["module_sha256"] == candidate._sha256_file(
        ROOT / first["implementation_binding"]["module_path"]
    )
    assert first["implementation_binding"]["test_sha256"] == candidate._sha256_file(
        ROOT / first["implementation_binding"]["test_path"]
    )


def test_config_mutations_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = copy.deepcopy(candidate.load_config(ROOT))
    config["claim_boundary"]["publication_ready"] = True
    monkeypatch.setattr(candidate, "_read_json", lambda _path: config)
    with pytest.raises(candidate.KineticGatePublicationCandidateError, match="config changed"):
        candidate.load_config(ROOT)


def test_no_observational_claim_or_access() -> None:
    receipt = candidate.build_receipt(ROOT)
    assert all(value == 0 for value in receipt["zero_access"].values())
    assert receipt["claim_boundary"]["observational_support"] is False
    assert receipt["claim_boundary"]["modified_gravity_success"] is False
    assert "not a full-action instability theorem" in receipt["maximal_theorem"]["strict_scope"]
