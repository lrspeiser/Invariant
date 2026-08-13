from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.barotropic_irrotational_constraint_propagation_gate import (
    BarotropicConstraintPropagationGateError,
    _canonical_sha,
    _exact_constraint_inventory_replay,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/barotropic_irrotational_constraint_propagation_gate.json"


def test_closes_fourth_gate_as_not_applicable_only() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_FOURTH_GATE_ZERO_INDEPENDENT_CONSTRAINTS"
    assert [item["outcome"] for item in receipt["gate_results"]] == [
        "PREDECESSOR_PASS",
        "PREDECESSOR_PASS",
        "PREDECESSOR_PASS",
        "PASS_NOT_APPLICABLE",
    ]
    assert receipt["counts"] == {
        "sectors": 1,
        "predecessor_gates": 3,
        "new_gates_passed_not_applicable": 1,
        "independent_primary_matter_constraints": 0,
        "independent_matter_gauge_generators": 0,
        "definitional_identities_replayed": 3,
        "exact_specialized_residuals": 5,
        "registered_negative_controls": 3,
        "specialized_negative_controls": 1,
        "blocks": 0,
        "rejects": 0,
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_exact_inventory_identities_and_boundary_negative() -> None:
    replay = _exact_constraint_inventory_replay()
    legendre = replay["legendre_map"]
    assert legendre["direct_coefficients"] == [3, -1]
    assert legendre["positive_decomposition_coefficients"] == [3, -1]
    assert legendre["identity_residual"] == [0, 0]
    assert legendre["hessian_rank"] == 1
    assert legendre["independent_primary_matter_constraints"] == 0
    assert legendre["independent_matter_gauge_generators"] == 0
    identities = replay["definitional_identities_not_constraints"]
    assert identities["velocity_normalization"]["numerator_residual"] == 0
    assert identities["irrotationality"]["residual"] == 0
    assert identities["barotropic_equation_of_state"]["residual"] == 0
    assert replay["propagation_system"]["independent_constraint_vector"] == []
    assert replay["negative_control"]["legendre_jacobian"] == 0
    assert replay["negative_control"]["rejected"] is True


def test_claims_remain_bounded() -> None:
    claims = build_receipt(CONFIG, root=ROOT)["claims"]
    assert claims["first_three_gates_closed_by_predecessors"] is True
    assert claims["matter_constraint_propagation_gate_closed_not_applicable"] is True
    for name, value in claims.items():
        if name not in {
            "first_three_gates_closed_by_predecessors",
            "matter_constraint_propagation_gate_closed_not_applicable",
        }:
            assert value is False, name


@pytest.mark.parametrize("binding", ["predecessor", "formal_controls", "legendre_source"])
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if binding == "predecessor":
        config["predecessor"]["file_sha256"] = "0" * 64
    else:
        config["evidence_bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BarotropicConstraintPropagationGateError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_policy_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["gravity_constraint_propagation"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BarotropicConstraintPropagationGateError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
