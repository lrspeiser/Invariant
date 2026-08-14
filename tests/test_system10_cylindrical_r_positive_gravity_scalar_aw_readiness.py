from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.system10_cylindrical_r_positive_gravity_scalar_aw_readiness import (
    System10GravityScalarAWReadinessError,
    _canonical_lf_sha,
    _load_binding,
    _source_api_audit,
    build_receipt,
    write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system10_cylindrical_r_positive_gravity_scalar_aw_readiness.json"
RECEIPT = ROOT / (
    "runs/math/system10-cylindrical-r-positive-gravity-scalar-aw-readiness/receipt.json"
)
NONLINEAR_SOURCE = ROOT / "src/sigma_theory_compiler/quartic_nonlinear_evolution_campaign.py"


@pytest.fixture(scope="module")
def receipt() -> dict[str, Any]:
    return build_receipt(CONFIG)


def test_committed_readiness_receipt_replays_exactly(receipt: dict[str, Any]) -> None:
    assert receipt == json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["decision"] == "BLOCK_COORDINATE_ARITHMETIC_A_W_MATERIALIZER_MISSING"


def test_representative_manifest_has_every_A_and_W_root_exactly_once(
    receipt: dict[str, Any],
) -> None:
    manifest = receipt["materialization"]["semantic_A_W_manifest"]
    a_entries = manifest["semantic_A_entries"]
    w_entries = manifest["semantic_W_entries"]
    assert manifest["candidate_id"] == "quartic-symbol-06e267a9215345b6"
    assert len(a_entries) == 121
    assert len(w_entries) == 11
    assert {(item["row"], item["column"]) for item in a_entries} == {
        (row, column) for row in range(11) for column in range(11)
    }
    assert {item["row"] for item in w_entries} == set(range(11))
    assert len({item["entry_sha256"] for item in a_entries}) == 121
    assert len({item["entry_sha256"] for item in w_entries}) == 11
    assert manifest["entrywise_coordinate_arithmetic_A_entries"] == 0
    assert manifest["entrywise_coordinate_arithmetic_W_entries"] == 0


def test_manifest_roots_replay_the_bound_operational_packet(receipt: dict[str, Any]) -> None:
    upstream = json.loads(
        (
            ROOT / "runs/physics-language/quartic-metric-rows-tensor-dag-campaign/campaign.json"
        ).read_text(encoding="utf-8")
    )
    roots = upstream["common_explicit_tensor_dag_packet"]["root_packet"]
    manifest = receipt["materialization"]["semantic_A_W_manifest"]
    observed_a = [item["semantic_root"] for item in manifest["semantic_A_entries"]]
    observed_w = [item["semantic_root"] for item in manifest["semantic_W_entries"]]
    assert observed_a == [root for row in roots["time_block_A"] for root in row]
    assert observed_w == roots["acceleration_free_W"]
    assert manifest["semantic_root_packet_sha256"] == roots["content_sha256"]


def test_downstream_arithmetic_still_receives_opaque_component_inputs(
    receipt: dict[str, Any],
) -> None:
    boundary = receipt["materialization"]["arithmetic_component_boundary"]
    assert boundary["component_inputs"] == 132
    assert boundary["operation"] == "exact_component_input"
    assert boundary["coordinate_arithmetic_expressions_embedded"] == 0
    assert receipt["counts"]["coordinate_arithmetic_A_entries"] == 0
    assert receipt["counts"]["coordinate_arithmetic_W_entries"] == 0


def test_live_source_api_audit_replays_and_exposes_no_checkpoint_serializer(
    receipt: dict[str, Any],
) -> None:
    audit = receipt["materialization"]["live_source_api_audit"]
    assert audit == _source_api_audit(NONLINEAR_SOURCE)
    assert audit["action_factor_tensor_calls"] == 3
    assert audit["action_simplify_calls"] == 1
    assert audit["assembler_factor_calls"] == 2
    assert audit["checkpoint_parameters_present"] == []
    assert audit["per_row_return_before_full_tensor_factor"] is False
    assert audit["atomic_A_W_entry_serializer"] is False


def test_first_missing_primitive_has_decisive_acceptance_and_no_row_overclaim(
    receipt: dict[str, Any],
) -> None:
    missing = receipt["materialization"]["first_missing_primitive"]
    assert missing["primitive_id"] == (
        "checkpointable_unfactored_fixed_r_positive_coordinate_A_W_materializer_v1"
    )
    assert missing["required_output"]["A_entries"] == 121
    assert missing["required_output"]["W_entries"] == 11
    assert len(missing["acceptance"]) == 6
    assert missing["status"] == "BLOCK_FIRST_IMPLEMENTATION_PRIMITIVE_MISSING"
    counts = receipt["counts"]
    assert counts["candidate_dynamic_rows_closed"] == 0
    assert counts["rhs_rows_closed_per_candidate"] == 74
    assert counts["candidate_dynamic_rows_remaining"] == 132
    assert receipt["claims"]["coordinate_arithmetic_A_W_materialized"] is False
    assert receipt["claims"]["full_85_state_rhs_closed"] is False


def test_probe_observation_is_bounded_and_not_used_as_algebraic_proof(
    receipt: dict[str, Any],
) -> None:
    probe = receipt["materialization"]["bounded_probe_observation"]
    assert probe["wall_clock_cap_seconds"] == 300
    assert probe["outcome"] == "TIMEOUT_BEFORE_FIRST_RETURNED_ACTION_PACKET"
    assert probe["scientific_role"] == (
        "supplemental operational evidence; not used as an algebraic proof"
    )


def test_negative_controls_reject_root_placeholder_point_and_row_promotions(
    receipt: dict[str, Any],
) -> None:
    controls = receipt["materialization"]["negative_controls"]
    assert set(controls) == {
        "promote_semantic_root_to_arithmetic_value",
        "promote_component_placeholder",
        "reuse_local_point_determinant",
        "claim_dynamic_row",
    }
    assert all(item["rejected"] is True for item in controls.values())
    assert controls["reuse_local_point_determinant"]["local_point_nonzero"] is True
    assert controls["reuse_local_point_determinant"]["symbolic_r_positive_domain_nonzero"] is False


def test_crlf_binding_is_portable_but_non_line_tamper_fails(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    binding = config["bindings"]["gravity_scalar_blocker"]
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
    with pytest.raises(System10GravityScalarAWReadinessError, match="hash mismatch"):
        _load_binding(tmp_path, copied_binding)


def test_binding_frozen_claim_and_immutable_output_tamper_fail(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["metric_tensor_dag"]["canonical_lf_sha256"] = "0" * 64
    tampered_binding = tmp_path / "tampered-binding.json"
    tampered_binding.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWReadinessError, match="hash mismatch"):
        build_receipt(tampered_binding, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["frozen_expectations"]["semantic_A_entries"] = 120
    tampered_frozen = tmp_path / "tampered-frozen.json"
    tampered_frozen.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWReadinessError, match="frozen"):
        build_receipt(tampered_frozen, root=ROOT)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["solved_dynamic_rows"] = True
    broadened = tmp_path / "broadened.json"
    broadened.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(System10GravityScalarAWReadinessError, match="claims policy"):
        build_receipt(broadened, root=ROOT)

    conflict = tmp_path / "receipt.json"
    conflict.write_text("{}\n", encoding="utf-8")
    with pytest.raises(System10GravityScalarAWReadinessError, match="immutable output conflict"):
        write_receipt(CONFIG, conflict, root=ROOT)


def test_self_evidence_uses_canonical_lf_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_evidence"].values():
        assert _canonical_lf_sha(ROOT / binding["path"]) == binding["canonical_lf_sha256"]
