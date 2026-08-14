from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_matter_dynamic_rhs_materializer import (
    System10MatterDynamicRHSError,
    _canonical_lf_sha,
    _fluid_exact_evaluation,
    _load_binding,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_matter_dynamic_rhs_materializer.json"
RECEIPT = (
    ROOT / "runs/math/system10-cylindrical-r-positive-matter-dynamic-rhs-materializer/receipt.json"
)


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return build_receipt(CONFIG)


def test_committed_receipt_replays_exactly(receipt: dict[str, Any]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["decision"] == "BOUNDED_PASS_2_MATTER_DYNAMIC_ROWS_BLOCK_15_DYNAMIC_ROWS"


def test_scalar_and_fluid_rows_have_exact_state_indices_and_origins(
    receipt: dict[str, Any],
) -> None:
    rows = receipt["materialization"]["rows"]
    assert [item["field_index"] for item in rows] == [11, 16]
    assert [item["lhs_state_index"] for item in rows] == [28, 33]
    assert rows[0]["state_atoms"] == {"q": 11, "v": 28, "w1": 45, "w2": 62, "w3": 79}
    assert rows[1]["state_atoms"] == {"q": 16, "v": 33, "w1": 50, "w2": 67, "w3": 84}
    assert rows[0]["equation_origin"]["action_sector_id"] == "canonical_minimally_coupled_scalar"
    assert rows[1]["equation_origin"]["action_sector_id"] == "barotropic_irrotational_fluid"
    assert rows[0]["solved_acceleration_certificate"]["substitution_residual"] == "0"
    assert rows[1]["solved_acceleration_certificate"]["substitution_residual"] == "0"
    assert rows[1]["solved_acceleration_certificate"]["dynamic_denominator_positive_proof"][-1] == (
        "therefore D=X+v**2>0"
    )


def test_scalar_cylindrical_row_contains_connection_and_mass_terms(
    receipt: dict[str, Any],
) -> None:
    scalar = receipt["materialization"]["rows"][0]
    assert scalar["rhs"] == (
        "partial_1 state[45]+state[45]/r+partial_2 state[62]/r**2+"
        "partial_3 state[79]-m_chi**2*state[11]"
    )
    assert scalar["solved_acceleration_certificate"]["maximum_coordinate_denominator_r_power"] == 2
    assert scalar["solved_acceleration_certificate"]["coordinate_pole_set"] == ["r=0"]


def test_fluid_row_exactly_solves_covariant_component_at_admitted_points(
    receipt: dict[str, Any],
) -> None:
    validation = receipt["materialization"]["independent_exact_validation"]
    assert validation["all_domain_admitted"] is True
    assert validation["all_residuals_zero"] is True
    assert len(validation["evaluations"]) == 2
    point = {
        "r": Fraction(2),
        "v": Fraction(3),
        "w1": Fraction(1),
        "w2": Fraction(2),
        "w3": Fraction(1),
        "dv1": Fraction(2),
        "dv2": Fraction(-1),
        "dv3": Fraction(3),
        "dw11": Fraction(1),
        "dw12": Fraction(2),
        "dw13": Fraction(-2),
        "dw22": Fraction(4),
        "dw23": Fraction(1),
        "dw33": Fraction(-1),
    }
    result = _fluid_exact_evaluation(point)
    assert result["X"] > 0
    assert result["D"] == result["X"] + point["v"] ** 2
    assert result["residual"] == 0
    fluid = receipt["materialization"]["rows"][1]
    assert fluid["definitions"]["H12"] == "partial_1 w2-w2/r"
    assert fluid["definitions"]["H22"] == "partial_2 w2+r*w1"
    assert fluid["equation_origin"]["kappa_cancellation"].endswith("kappa>0")


def test_all_candidates_advance_to_70_of_85_without_overclaim(
    receipt: dict[str, Any],
) -> None:
    counts = receipt["counts"]
    assert counts["candidate_dynamic_row_instances_registered"] == 24
    assert counts["total_rhs_rows_closed_per_candidate"] == 70
    assert counts["candidate_dynamic_rows_remaining"] == 180
    candidates = receipt["materialization"]["candidate_results"]
    assert len(candidates) == 12
    assert len({item["candidate_id"] for item in candidates}) == 12
    assert all(item["total_rhs_rows_closed"] == 70 for item in candidates)
    assert all(item["dynamic_rows_remaining"] == 15 for item in candidates)
    assert receipt["claims"]["full_85_state_rhs_closed"] is False
    assert receipt["claims"]["constraint_propagation_closed"] is False
    assert receipt["claims"]["hyperbolicity_closed"] is False


def test_every_remaining_row_has_an_exact_typed_missing_primitive(
    receipt: dict[str, Any],
) -> None:
    missing = receipt["materialization"]["missing_dynamic_rows"]
    assert len(missing) == 15
    assert [item["field_index"] for item in missing] == [*range(11), *range(12, 16)]
    assert sum(item["sector"] == "sourced_metric" for item in missing) == 10
    assert sum(item["sector"] == "candidate_gravity_scalar" for item in missing) == 1
    assert sum(item["sector"] == "source_free_maxwell" for item in missing) == 4
    assert all(item["missing_primitives"] for item in missing)
    assert receipt["materialization"]["next_missing_primitive"]["required_candidate_rows"] == 180


def test_connection_sign_denominator_zero_fill_and_overclaim_corruptions_reject(
    receipt: dict[str, Any],
) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "scalar_missing_cylindrical_connection",
        "scalar_mass_sign_flip",
        "fluid_wrong_acoustic_denominator",
        "fluid_missing_cylindrical_connection",
        "maxwell_zero_fill",
        "claim_full_rhs",
    }
    assert all(item["rejected"] is True for item in controls.values())
    assert controls["scalar_missing_cylindrical_connection"]["exact_residual"] == "3/2"
    assert controls["fluid_wrong_acoustic_denominator"]["exact_rhs_delta"] == "2/3"
    assert controls["fluid_missing_cylindrical_connection"]["exact_rhs_delta"] == "2/13"


def test_crlf_binding_is_portable_but_non_line_tamper_fails(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    binding = config["bindings"]["common_rhs"]
    source = ROOT / binding["path"]
    lf = source.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    assert hashlib.sha256(lf).hexdigest() == binding["canonical_lf_sha256"]
    copied_binding = {**binding, "path": "predecessor.json"}
    copied = tmp_path / copied_binding["path"]
    copied.write_bytes(crlf)
    _, loaded = _load_binding(tmp_path, copied_binding)
    assert loaded["content_sha256"] == binding["content_sha256"]
    copied.write_bytes(crlf + b" ")
    with pytest.raises(System10MatterDynamicRHSError, match="hash mismatch"):
        _load_binding(tmp_path, copied_binding)


def test_binding_frozen_claim_and_immutable_output_tamper_fail(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["total_matter_action"]["canonical_lf_sha256"] = "0" * 64
    tampered_binding = tmp_path / "tampered-binding.json"
    tampered_binding.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10MatterDynamicRHSError, match="hash mismatch"):
        build_receipt(tampered_binding, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["frozen_expectations"]["dynamic_rows_registered"] = 3
    tampered_frozen = tmp_path / "tampered-frozen.json"
    tampered_frozen.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10MatterDynamicRHSError, match="frozen"):
        build_receipt(tampered_frozen, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_85_state_rhs"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10MatterDynamicRHSError, match="claims policy"):
        build_receipt(broadened, root=ROOT)

    conflict = tmp_path / "receipt.json"
    conflict.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10MatterDynamicRHSError, match="immutable output conflict"):
        write_receipt(CONFIG, conflict, root=ROOT)


def test_self_evidence_uses_canonical_lf_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]
