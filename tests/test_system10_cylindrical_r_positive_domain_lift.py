from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_domain_lift import (
    System10CylindricalDomainLiftError,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_domain_lift.json"
RECEIPT = ROOT / "runs/math/system10-cylindrical-r-positive-domain-lift/receipt.json"


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return build_receipt(CONFIG)


def test_committed_receipt_replays_exactly(receipt: dict[str, Any]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_all_1010_formulation_jets_are_exact_rational_functions(
    receipt: dict[str, Any],
) -> None:
    packet = receipt["materialization"]["formulation_jet_rational_functions"]
    entries = [entry for family in packet["families"] for entry in family["entries"]]
    assert len(entries) == 1010
    assert packet["nonzero_rational_functions"] == 22
    assert packet["r1_values_replayed_exactly"] == 1010
    assert packet["maximum_denominator_r_power"] == 4
    assert all(item["denominator_r_power"] >= 0 for item in entries)


def test_all_96_rows_lift_and_replay_at_r1(receipt: dict[str, Any]) -> None:
    materialization = receipt["materialization"]
    rows = materialization["sourced_rational_rows"]
    candidates = materialization["candidate_results"]
    assert len(materialization["shared_symbolic_gauge_rows"]) == 4
    assert len(rows) == 48
    assert len(candidates) == 12
    assert all(item["symbolic_gauge_rows"] == 4 for item in candidates)
    assert all(item["symbolic_hamiltonian_momentum_rows"] == 4 for item in candidates)
    assert receipt["counts"]["physical_gravity_rows_closed"] == 96
    assert receipt["counts"]["exact_r1_replays"] == 1062
    assert all(item["r1_predecessor_polynomial_sha256"] for item in rows)


def test_denominator_domain_and_time_differential_certificates(
    receipt: dict[str, Any],
) -> None:
    materialization = receipt["materialization"]
    domain = materialization["domain_certificate"]
    proof = materialization["acceleration_and_integrability_proof"]
    assert domain["domain"] == "r>0"
    assert domain["denominator_zero_set"] == ["r=0"]
    assert domain["all_denominators_monomials_in_r"] is True
    assert domain["maximum_denominator_r_power"] > 0
    assert proof["partial_0_v_nonzero_coefficients"] == 0
    assert proof["integrability_substitutions"] == 51
    assert proof["forbidden_time_differential_atoms_after_replacement"] == 0


def test_domain_sign_and_row_negative_controls_are_sealed(
    receipt: dict[str, Any],
) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "include_axis_r_zero",
        "corrupt_radial_derivative_sign",
        "drop_symbolic_row",
        "corrupt_r1_row_coefficient",
    }
    assert all(item["rejected"] is True for item in controls.values())
    assert controls["corrupt_radial_derivative_sign"]["r1_expected"] == "-2"
    assert controls["drop_symbolic_row"]["observed_rows"] == 95


def test_no_broader_pde_claim_is_made(receipt: dict[str, Any]) -> None:
    claims = receipt["claims"]
    assert claims["fixed_cylindrical_profile_r_positive_closed"] is True
    assert claims["arbitrary_formulation_functions_closed"] is False
    assert claims["sourced_constraint_propagation_closed"] is False
    assert claims["general_hyperbolicity_closed"] is False
    assert claims["general_common_time_positivity_closed"] is False
    assert claims["promotion_authorized"] is False


def test_tampered_domain_binding_and_claim_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["domain_contract"]["predicate"] = "r>=0"
    changed_domain = tmp_path / "changed-domain.json"
    changed_domain.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalDomainLiftError, match="domain contract"):
        build_receipt(changed_domain, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["r1_sourced_row_receipt"]["file_sha256"] = "0" * 64
    changed_binding = tmp_path / "changed-binding.json"
    changed_binding.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalDomainLiftError, match="hash mismatch"):
        build_receipt(changed_binding, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["sourced_constraint_propagation"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10CylindricalDomainLiftError, match="claims policy"):
        build_receipt(broadened, root=ROOT)
