from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.system11_g2_solar_authorization_closure import (
    CAMPAIGN_ID,
    CONFIG_PATH,
    OPENING_SCHEMA,
    OUTPUT_PATH,
    System11SolarAuthorizationError,
    _semantic_sha256,
    build_receipt,
    validate_and_consume_opening_packet,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / OUTPUT_PATH
CANDIDATES = (
    (
        "G3A-2f8983c88f504150381064f2",
        "19f36a7c814ca11ace6de1270802a542872c35c27c7e64542eea672e16cbae88",
    ),
    (
        "G3A-58e59412e5fe77cd54caf863",
        "9457ba1ff99ecfdabc08200dda3ff15b8656b025d106fe2c2cd4abd77a01c3b5",
    ),
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _copy_bound_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config = _read(ROOT / CONFIG_PATH)
    _write(tmp_path / CONFIG_PATH, config)
    for descriptor in config["source_bindings"].values():
        relative = descriptor["path"]
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path, config


def _opening_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    domains: list[dict[str, object]] = []
    domain_hashes: dict[str, str] = {}
    for index, (candidate_id, action_sha) in enumerate(CANDIDATES):
        value = {
            "candidate_id": candidate_id,
            "action_sha256": action_sha,
            "status": "instantiated_before_target_access",
            "target_values_accessed": False,
            "domain": "registered_static_connected_regular_horizonless_source",
        }
        relative = f"opening/domain-{index}.json"
        _write(tmp_path / relative, value)
        semantic = canonical_sha256(value)
        domain_hashes[candidate_id] = semantic
        domains.append(
            {
                "candidate_id": candidate_id,
                "action_sha256": action_sha,
                "document": {"path": relative, "semantic_sha256": semantic},
            }
        )
    split = {
        "status": "committed_before_target_access",
        "target_values_accessed": False,
        "candidate_ids": [candidate_id for candidate_id, _ in CANDIDATES],
        "roles": [
            "state_and_calibration_training",
            "formula_selection_validation",
            "untouched_target_blind_test",
        ],
    }
    roots = {
        "status": "selected_and_hash_bound_before_target_access",
        "target_values_accessed": False,
        "primary_record_payloads_opened": False,
        "dataset_id": "CO-SS-RSS-1-SCE1-V1.0",
        "record_roots_sha256": ["a" * 64, "b" * 64],
    }
    _write(tmp_path / "opening/split.json", split)
    _write(tmp_path / "opening/roots.json", roots)
    split_hash = canonical_sha256(split)
    roots_hash = canonical_sha256(roots)
    authorization = {
        "status": "authorized",
        "independent_of_candidate_generation": True,
        "campaign_id": CAMPAIGN_ID,
        "candidate_actions": [
            {"candidate_id": candidate_id, "action_sha256": action_sha}
            for candidate_id, action_sha in CANDIDATES
        ],
        "candidate_source_domain_sha256": domain_hashes,
        "held_out_split_commitment_sha256": split_hash,
        "selected_primary_record_roots_sha256": roots_hash,
        "atomic_open_batches": 1,
        "candidate_evaluations": 2,
        "refits_after_open": 0,
        "promotion_actions": 0,
        "authority_id": "independent-review-board-fixture",
    }
    _write(tmp_path / "opening/authorization.json", authorization)
    packet = {
        "schema_version": OPENING_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "candidate_source_domains": domains,
        "held_out_split_commitment": {
            "path": "opening/split.json",
            "semantic_sha256": split_hash,
        },
        "selected_primary_record_roots": {
            "path": "opening/roots.json",
            "semantic_sha256": roots_hash,
        },
        "independent_authorization": {
            "path": "opening/authorization.json",
            "semantic_sha256": canonical_sha256(authorization),
        },
    }
    packet_path = tmp_path / "opening/opening-packet.json"
    _write(packet_path, packet)
    return packet, packet_path


def test_checked_receipt_replays_exactly() -> None:
    assert build_receipt(ROOT) == _read(RECEIPT)


def test_current_boundary_is_exactly_blocked_before_opening() -> None:
    result = build_receipt(ROOT)
    assert result["decision"] == "block"
    assert result["missing_opening_fields"] == [
        "source_branch_domain_instantiation_sha256",
        "held_out_split_commitment_sha256",
        "selected_primary_record_roots_sha256",
        "observation_opening_authorization_sha256",
    ]
    assert set(result["missing_obligations_by_candidate"]) == {
        candidate_id for candidate_id, _ in CANDIDATES
    }
    assert result["counts"]["missing_external_opening_obligations"] == 8


def test_no_target_or_primary_payload_was_read() -> None:
    result = build_receipt(ROOT)
    assert result["data_boundary"] == {
        "target_data_read": False,
        "primary_record_payloads_read": False,
        "held_out_target_access_count": 0,
        "primary_record_access_count": 0,
        "real_data_evaluation_count": 0,
    }
    assert result["authorization_inventory"]["already_authorized_observational_artifacts"] == 0


def test_one_shot_budget_and_launch_command_are_frozen() -> None:
    result = build_receipt(ROOT)
    assert result["one_shot_budget"] == {
        "atomic_open_batches": 1,
        "candidate_evaluations": 2,
        "refits_after_open": 0,
        "promotion_actions": 0,
    }
    launch = result["launch_contract"]
    assert launch["consume_packet_once"] is True
    assert launch["target_data_read_by_preflight"] is False
    assert "--consume-once-output" in launch["command"]
    audit = result["execution_implementation_audit"]
    assert audit["action_bound_real_record_likelihood_executor"] == ("block_not_registered")
    assert audit["preflight_command_executes_observational_evaluation"] is False


def test_bound_metadata_tamper_fails_closed(tmp_path: Path) -> None:
    root, _ = _copy_bound_root(tmp_path)
    receipt = _read(root / "runs/math/g2-solar-one-shot-observation-boundary/receipt.json")
    receipt["decision"] = "pass"
    _write(root / "runs/math/g2-solar-one-shot-observation-boundary/receipt.json", receipt)
    with pytest.raises(System11SolarAuthorizationError, match="semantic source changed"):
        build_receipt(root)


def test_resealed_false_local_authorization_fails_closed(tmp_path: Path) -> None:
    root, config = _copy_bound_root(tmp_path)
    relative = "runs/engine/g2-solar-heldout-transfer-registration.json"
    transfer = _read(root / relative)
    transfer["observational_authorization"] = True
    _write(root / relative, transfer)
    config["source_bindings"]["transfer_registration"]["semantic_sha256"] = _semantic_sha256(
        transfer
    )
    _write(root / CONFIG_PATH, config)
    with pytest.raises(System11SolarAuthorizationError, match="sealed boundary changed"):
        build_receipt(root)


def test_valid_external_opening_packet_is_consumed_once_without_target_read(
    tmp_path: Path,
) -> None:
    _, packet_path = _opening_fixture(tmp_path)
    output = tmp_path / "opening/authorization-receipt.json"
    result = validate_and_consume_opening_packet(tmp_path, packet_path, output)
    assert result == _read(output)
    assert result["decision"] == "pass_authorization_preflight_only"
    assert result["data_boundary"] == {
        "target_data_read": False,
        "primary_record_payloads_read": False,
        "authorization_metadata_consumptions": 1,
    }
    assert result["claims"]["one_shot_executed"] is False
    with pytest.raises(System11SolarAuthorizationError, match="already exists"):
        validate_and_consume_opening_packet(tmp_path, packet_path, output)


def test_authorization_sign_corruption_fails(tmp_path: Path) -> None:
    packet, packet_path = _opening_fixture(tmp_path)
    authorization_path = tmp_path / packet["independent_authorization"]["path"]
    authorization = _read(authorization_path)
    authorization["refits_after_open"] = 1
    _write(authorization_path, authorization)
    packet["independent_authorization"]["semantic_sha256"] = canonical_sha256(authorization)
    _write(packet_path, packet)
    with pytest.raises(System11SolarAuthorizationError, match="refits_after_open"):
        validate_and_consume_opening_packet(tmp_path, packet_path, tmp_path / "opening/result.json")


def test_empty_primary_root_manifest_fails(tmp_path: Path) -> None:
    packet, packet_path = _opening_fixture(tmp_path)
    roots_path = tmp_path / packet["selected_primary_record_roots"]["path"]
    roots = _read(roots_path)
    roots["record_roots_sha256"] = []
    _write(roots_path, roots)
    packet["selected_primary_record_roots"]["semantic_sha256"] = canonical_sha256(roots)
    authorization_path = tmp_path / packet["independent_authorization"]["path"]
    authorization = _read(authorization_path)
    authorization["selected_primary_record_roots_sha256"] = canonical_sha256(roots)
    _write(authorization_path, authorization)
    packet["independent_authorization"]["semantic_sha256"] = canonical_sha256(authorization)
    _write(packet_path, packet)
    with pytest.raises(System11SolarAuthorizationError, match="not admissible"):
        validate_and_consume_opening_packet(tmp_path, packet_path, tmp_path / "opening/result.json")


def test_opening_descriptor_cannot_escape_root(tmp_path: Path) -> None:
    packet, packet_path = _opening_fixture(tmp_path)
    packet["held_out_split_commitment"]["path"] = "../split.json"
    _write(packet_path, packet)
    with pytest.raises(System11SolarAuthorizationError, match="escapes project root"):
        validate_and_consume_opening_packet(tmp_path, packet_path, tmp_path / "opening/result.json")
