from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_maxwell_dynamic_rhs_materializer import (
    System10MaxwellDynamicRHSError,
    _canonical_lf_sha,
    _independent_covariant_replay,
    _load_binding,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_maxwell_dynamic_rhs_materializer.json"
RECEIPT = (
    ROOT / "runs/math/system10-cylindrical-r-positive-maxwell-dynamic-rhs-materializer/receipt.json"
)


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return build_receipt(CONFIG)


def test_committed_receipt_replays_exactly(receipt: dict[str, Any]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["decision"] == "BOUNDED_PASS_4_MAXWELL_DYNAMIC_ROWS_BLOCK_11_GRAVITY_ROWS"


def test_all_four_covector_rows_have_exact_state_indices_and_origins(
    receipt: dict[str, Any],
) -> None:
    rows = receipt["materialization"]["rows"]
    assert [item["field_index"] for item in rows] == [12, 13, 14, 15]
    assert [item["lhs_state_index"] for item in rows] == [29, 30, 31, 32]
    assert [item["component"] for item in rows] == ["B_0", "B_1", "B_2", "B_3"]
    for component, row in enumerate(rows):
        field = 12 + component
        assert row["state_atoms"] == {
            "q": field,
            "v": 17 + field,
            "w1": 34 + field,
            "w2": 51 + field,
            "w3": 68 + field,
        }
        assert row["equation_origin"]["gauge_completion"] == "E_L_mu=E_mu+nabla_mu C"
        assert row["equation_origin"]["reduced_equation"] == "box_g B_mu-R_mu^nu B_nu=0"
        assert row["solved_acceleration_certificate"]["substitution_residual"] == "0"
        assert row["solved_acceleration_certificate"]["coordinate_pole_set"] == ["r=0"]


def test_radial_and_angular_covector_connection_couplings_are_retained(
    receipt: dict[str, Any],
) -> None:
    rows = receipt["materialization"]["rows"]
    terms = [{(term["coefficient"], term["atom"]) for term in row["rhs_terms"]} for row in rows]
    assert ("1/r", "state[46]") in terms[0]
    assert ("-1/r**2", "state[13]") in terms[1]
    assert ("-2/r**3", "state[65]") in terms[1]
    assert ("-1/r", "state[48]") in terms[2]
    assert ("2/r", "state[64]") in terms[2]
    assert ("1/r", "state[49]") in terms[3]
    assert rows[1]["solved_acceleration_certificate"]["maximum_coordinate_denominator_r_power"] == 3


def test_independent_levi_civita_covector_box_replay_is_exact(
    receipt: dict[str, Any],
) -> None:
    replay = receipt["materialization"]["independent_covariant_replay"]
    assert replay == _independent_covariant_replay()
    assert replay["nonzero_christoffel"] == {
        "Gamma^1_22": "-r",
        "Gamma^2_12": "1/r",
        "Gamma^2_21": "1/r",
    }
    assert replay["ricci_tensor_nonzero_entries"] == 0
    assert replay["component_residuals"] == ["0", "0", "0", "0"]
    assert replay["all_residuals_zero"] is True


def test_all_candidates_advance_to_74_of_85_without_propagation_overclaim(
    receipt: dict[str, Any],
) -> None:
    counts = receipt["counts"]
    assert counts["maxwell_dynamic_rows_registered"] == 4
    assert counts["candidate_row_instances_registered"] == 48
    assert counts["total_rhs_rows_closed_per_candidate"] == 74
    assert counts["candidate_dynamic_rows_remaining"] == 132
    candidates = receipt["materialization"]["candidate_results"]
    assert len(candidates) == 12
    assert len({item["candidate_id"] for item in candidates}) == 12
    assert all(item["total_rhs_rows_closed"] == 74 for item in candidates)
    assert all(item["dynamic_rows_remaining"] == 11 for item in candidates)
    assert receipt["claims"]["full_85_state_rhs_closed"] is False
    assert receipt["claims"]["constraint_propagation_closed"] is False
    assert receipt["claims"]["hyperbolicity_closed"] is False


def test_only_typed_gravity_acceleration_blocks_remain(receipt: dict[str, Any]) -> None:
    remaining = receipt["materialization"]["remaining_dynamic_rows"]
    assert len(remaining) == 11
    assert [item["field_index"] for item in remaining] == list(range(11))
    assert sum(item["sector"] == "sourced_metric" for item in remaining) == 10
    assert sum(item["sector"] == "candidate_gravity_scalar" for item in remaining) == 1
    assert all(item["missing_primitives"] for item in remaining)
    next_missing = receipt["materialization"]["next_missing_primitive"]
    assert next_missing["required_rows_per_candidate"] == 11
    assert next_missing["required_candidate_rows"] == 132


def test_connection_cross_zero_fill_and_overclaim_corruptions_reject(
    receipt: dict[str, Any],
) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "omit_B0_radial_connection",
        "omit_B1_algebraic_connection",
        "flip_B1_B2_cross_sign",
        "flip_B2_radial_connection_sign",
        "omit_B2_B1_cross_connection",
        "zero_fill_maxwell_rows",
        "claim_full_rhs_or_propagation",
    }
    assert all(item["rejected"] is True for item in controls.values())
    assert controls["omit_B0_radial_connection"]["exact_rhs_delta"] == "3/2"
    assert controls["flip_B1_B2_cross_sign"]["exact_rhs_delta"] == "2"
    assert controls["omit_B2_B1_cross_connection"]["exact_rhs_delta"] == "5"


def test_crlf_binding_is_portable_but_non_line_tamper_fails(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    binding = config["bindings"]["matter_dynamic_rhs"]
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
    with pytest.raises(System10MaxwellDynamicRHSError, match="hash mismatch"):
        _load_binding(tmp_path, copied_binding)


def test_binding_frozen_claim_and_immutable_output_tamper_fail(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["maxwell_mixed_principal"]["canonical_lf_sha256"] = "0" * 64
    tampered_binding = tmp_path / "tampered-binding.json"
    tampered_binding.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10MaxwellDynamicRHSError, match="hash mismatch"):
        build_receipt(tampered_binding, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["frozen_expectations"]["maxwell_rows_registered"] = 3
    tampered_frozen = tmp_path / "tampered-frozen.json"
    tampered_frozen.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10MaxwellDynamicRHSError, match="frozen"):
        build_receipt(tampered_frozen, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["constraint_propagation"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10MaxwellDynamicRHSError, match="claims policy"):
        build_receipt(broadened, root=ROOT)

    conflict = tmp_path / "receipt.json"
    conflict.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10MaxwellDynamicRHSError, match="immutable output conflict"):
        write_receipt(CONFIG, conflict, root=ROOT)


def test_self_evidence_uses_canonical_lf_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]
