from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_kinematic_rhs_materializer import (
    System10KinematicRHSError,
    _canonical_lf_sha,
    _load_binding,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_kinematic_rhs_materializer.json"
RECEIPT = ROOT / "runs/math/system10-cylindrical-r-positive-kinematic-rhs-materializer/receipt.json"


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return build_receipt(CONFIG)


def test_committed_receipt_replays_exactly(receipt: dict[str, Any]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["decision"] == (
        "BOUNDED_PASS_68_COMMON_KINEMATIC_RHS_ROWS_BLOCK_17_DYNAMIC_ROWS"
    )


def test_complete_q_and_w_kinematic_families_have_exact_state_indices(
    receipt: dict[str, Any],
) -> None:
    rows = receipt["materialization"]["rows"]
    assert len(rows) == 68
    assert [item["lhs_state_index"] for item in rows[:17]] == list(range(17))
    assert [item["rhs_terms"][0]["state_index"] for item in rows[:17]] == list(range(17, 34))
    assert all(item["rhs_terms"][0]["spatial_derivatives"] == [] for item in rows[:17])
    assert [item["lhs_state_index"] for item in rows[17:]] == list(range(34, 85))
    for item in rows[17:]:
        coordinate = item["equation_origin"]["spatial_coordinate"]
        field = item["equation_origin"]["field_index"]
        assert item["rhs_terms"][0]["state_index"] == 17 + field
        assert item["rhs_terms"][0]["spatial_derivatives"] == [coordinate]


def test_every_registered_row_has_a_bound_equation_origin(
    receipt: dict[str, Any],
) -> None:
    rows = receipt["materialization"]["rows"]
    assert {item["equation_origin"]["origin_type"] for item in rows[:17]} == {"state_definition"}
    assert {item["equation_origin"]["origin_type"] for item in rows[17:]} == {
        "commuting_coordinate_partial_integrability"
    }
    assert len({item["row_sha256"] for item in rows}) == 68
    assert len({item["equation_origin"]["origin_sha256"] for item in rows}) == 68
    assert all(item["maximum_denominator_r_power"] == 0 for item in rows)


def test_all_candidates_receive_the_common_slice_but_not_dynamic_rows(
    receipt: dict[str, Any],
) -> None:
    counts = receipt["counts"]
    assert counts["common_kinematic_rows_registered"] == 68
    assert counts["candidate_row_instances_closed"] == 816
    assert counts["candidate_dynamic_velocity_rows_required"] == 204
    assert counts["candidate_dynamic_velocity_rows_registered"] == 0
    candidates = receipt["materialization"]["candidate_results"]
    assert len(candidates) == 12
    assert len({item["candidate_id"] for item in candidates}) == 12
    assert all(item["complete_kinematic_rhs_rows"] == 68 for item in candidates)
    assert all(item["missing_candidate_dynamic_v_rows"] == 17 for item in candidates)
    assert receipt["claims"]["full_85_state_rhs_closed"] is False
    assert receipt["claims"]["constraint_propagation_closed"] is False


def test_dynamic_row_nonidentifiability_witness_is_exact(receipt: dict[str, Any]) -> None:
    witness = receipt["materialization"]["dynamic_row_nonidentifiability_witness"]
    assert witness["dynamic_row"] == "evolution_v[0]"
    assert witness["completion_0_lower_order_increment"] == "0"
    assert witness["completion_1_lower_order_increment"] == "q_0**2"
    assert witness["shared_registered_principal_A_Bi_Cij"] is True
    assert witness["shared_68_kinematic_rows"] is True
    assert witness["sourced_acceleration_solution_registered"] is False
    assert witness["exact_rhs_delta"] == "1"
    assert witness["nonzero"] is True
    missing = receipt["materialization"]["next_missing_primitive"]
    assert missing["required_rows_per_candidate"] == 17
    assert missing["required_candidate_rows"] == 204
    assert missing["registered_rows"] == 0


def test_omission_wrong_derivative_zero_fill_and_overclaim_reject(
    receipt: dict[str, Any],
) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "omit_last_integrability_row",
        "wrong_spatial_derivative",
        "zero_fill_dynamic_velocity_rows",
        "claim_full_rhs_from_kinematic_slice",
    }
    assert all(item["rejected"] is True for item in controls.values())
    assert controls["wrong_spatial_derivative"]["exact_residual"] == "5"


def test_crlf_bound_json_is_portable_but_non_line_tamper_fails(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    binding = config["bindings"]["first_order_reduction"]
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
    with pytest.raises(System10KinematicRHSError, match="hash mismatch"):
        _load_binding(tmp_path, copied_binding)


def test_binding_frozen_claim_and_output_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["divq_rows"]["canonical_lf_sha256"] = "0" * 64
    tampered_binding = tmp_path / "tampered-binding.json"
    tampered_binding.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10KinematicRHSError, match="hash mismatch"):
        build_receipt(tampered_binding, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["frozen_expectations"]["row_count"] = 67
    tampered_frozen = tmp_path / "tampered-frozen.json"
    tampered_frozen.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10KinematicRHSError, match="frozen"):
        build_receipt(tampered_frozen, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["constraint_propagation"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10KinematicRHSError, match="claims policy"):
        build_receipt(broadened, root=ROOT)

    conflict = tmp_path / "receipt.json"
    conflict.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10KinematicRHSError, match="immutable output conflict"):
        write_receipt(CONFIG, conflict, root=ROOT)


def test_self_evidence_uses_canonical_lf_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]
