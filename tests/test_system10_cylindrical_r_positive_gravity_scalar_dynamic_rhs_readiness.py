from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_dynamic_rhs_readiness import (
    System10GravityScalarReadinessError,
    _ambiguity_witness,
    _canonical_lf_sha,
    _independent_source_replay,
    _load_binding,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/system10_cylindrical_r_positive_gravity_scalar_dynamic_rhs_readiness.json"
)
RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-gravity-scalar-dynamic-rhs-readiness/receipt.json"
)


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return build_receipt(CONFIG)


def test_committed_block_receipt_replays_exactly(receipt: dict[str, Any]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["decision"] == "BLOCK_GRAVITY_SCALAR_REQUIRES_COUPLED_11X11_FIXED_R_SOLVE"


def test_each_candidate_has_two_exact_solutions_with_different_scalar_acceleration(
    receipt: dict[str, Any],
) -> None:
    blocks = receipt["materialization"]["candidate_blocks"]
    assert len(blocks) == 12
    assert len({item["candidate_id"] for item in blocks}) == 12
    for item in blocks:
        assert item["solution_A"]["exact_scalar_euler_residual"] == "0"
        assert item["solution_B"]["exact_scalar_euler_residual"] == "0"
        assert item["solution_A"]["aphi"] != item["solution_B"]["aphi"]
        assert item["scalar_acceleration_difference"] == item["candidate_coefficients"]["alpha"]
        assert item["candidate_coefficients"]["alpha"] != "0"
        assert item["conclusion"] == "SCALAR_ROW_ALONE_DOES_NOT_IDENTIFY_APHI"


def test_witness_is_the_exact_bound_scalar_row_at_an_admissible_cylindrical_point(
    receipt: dict[str, Any],
) -> None:
    for item in receipt["materialization"]["candidate_blocks"]:
        profile = item["witness_profile"]
        row = item["specialized_scalar_euler"]
        assert profile["coordinate_domain"] == "fixed cylindrical r>0"
        assert profile["radius"] == "1"
        assert profile["nonzero_connection"] == [
            "Gamma^1_22=-1",
            "Gamma^2_12=Gamma^2_21=1",
        ]
        assert row["equation"] == ("-alpha*ag22-alpha*ag33+(1+3*c20)*aphi-(1+c20)=0")
        alpha = item["candidate_coefficients"]["alpha"]
        assert row["ordered_acceleration_coefficients"][7] == _negate(alpha)
        assert row["ordered_acceleration_coefficients"][9] == _negate(alpha)
        assert item["source_equation_origin"]["lowered_row"] == 10


def test_independent_universal_constructor_replay_matches_specialized_row(
    receipt: dict[str, Any],
) -> None:
    replay = receipt["materialization"]["independent_source_replay"]
    assert replay == _independent_source_replay()
    assert replay["ordered_acceleration_coefficients"] == [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "-alpha",
        "0",
        "-alpha",
        "3*c20 + 1",
    ]
    assert replay["W_phi"] == "-c20 - 1"
    assert replay["affine_residual"] == "0"
    assert replay["exact_match"] is True


def _negate(value: str) -> str:
    if value.startswith("-"):
        return value[1:]
    return f"-{value}"


def test_candidate_block_rebuild_is_deterministic_and_coefficient_tamper_changes_seal(
    receipt: dict[str, Any],
) -> None:
    original = receipt["materialization"]["candidate_blocks"][0]
    coefficients = {
        "a10": original["candidate_coefficients"]["alpha"],
        "c20": original["candidate_coefficients"]["c20"],
    }
    rebuilt = _ambiguity_witness(
        original["candidate_id"], coefficients, original["source_equation_origin"]
    )
    assert rebuilt == original
    coefficients["a10"] = "1" if coefficients["a10"] != "1" else "-1"
    tampered = _ambiguity_witness(
        original["candidate_id"], coefficients, original["source_equation_origin"]
    )
    assert tampered["witness_sha256"] != original["witness_sha256"]


def test_coverage_remains_74_and_metric_extension_stops_after_scalar_block(
    receipt: dict[str, Any],
) -> None:
    counts = receipt["counts"]
    assert counts["gravity_scalar_rows_registered"] == 0
    assert counts["total_rhs_rows_closed_per_candidate"] == 74
    assert counts["dynamic_rows_remaining_per_candidate"] == 11
    assert counts["candidate_dynamic_rows_remaining"] == 132
    assert counts["metric_rows_attempted_after_block"] == 0
    stop = receipt["materialization"]["stop_condition"]
    assert stop["status"] == "BLOCK_BEFORE_METRIC_ROW_EXTENSION"
    assert stop["metric_rows_attempted_after_block"] == 0
    assert receipt["claims"]["gravity_scalar_dynamic_row_closed"] is False
    assert receipt["claims"]["metric_dynamic_rows_closed"] is False
    assert receipt["claims"]["full_85_state_rhs_closed"] is False


def test_exact_negative_controls_reject_zero_fill_point_reuse_and_overclaim(
    receipt: dict[str, Any],
) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "zero_metric_coupling",
        "promote_scalar_only_solution",
        "reuse_local_point_solve_as_fixed_r_row",
        "claim_75_rows",
    }
    assert all(item["rejected"] is True for item in controls.values())
    assert controls["zero_metric_coupling"]["all_candidates_have_nonzero_alpha"] is True
    assert controls["promote_scalar_only_solution"]["distinct_exact_solution_per_candidate"] is True
    assert (
        controls["reuse_local_point_solve_as_fixed_r_row"][
            "local_solve_has_serialized_fixed_r_component_10"
        ]
        is False
    )


def test_crlf_binding_is_portable_but_non_line_tamper_fails(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    binding = config["bindings"]["scalar_row_lowering"]
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
    with pytest.raises(System10GravityScalarReadinessError, match="hash mismatch"):
        _load_binding(tmp_path, copied_binding)


def test_binding_frozen_claim_and_immutable_output_tamper_fail(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["metric_tensor_dag"]["canonical_lf_sha256"] = "0" * 64
    tampered_binding = tmp_path / "tampered-binding.json"
    tampered_binding.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarReadinessError, match="hash mismatch"):
        build_receipt(tampered_binding, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["frozen_expectations"]["candidate_blocks"] = 11
    tampered_frozen = tmp_path / "tampered-frozen.json"
    tampered_frozen.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarReadinessError, match="frozen"):
        build_receipt(tampered_frozen, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["gravity_scalar_dynamic_row"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarReadinessError, match="claims policy"):
        build_receipt(broadened, root=ROOT)

    conflict = tmp_path / "receipt.json"
    conflict.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10GravityScalarReadinessError, match="immutable output conflict"):
        write_receipt(CONFIG, conflict, root=ROOT)


def test_self_evidence_uses_canonical_lf_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]
