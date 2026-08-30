from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_3d_source_geometry_contract_v1 as contract


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict, list[dict], list[dict]]:
    config, predecessor, catalog = contract.load_metadata()
    rows = list(contract.iter_dimensionality_rows(config, predecessor, catalog))
    return config, predecessor, catalog, rows


def test_local_config_and_predecessor_seals(packet: tuple) -> None:
    config, _predecessor, _catalog, _rows = packet
    contract.validate_config(config)
    contract.validate_predecessors(config)


def test_catalog_and_object_counts(packet: tuple) -> None:
    _config, predecessor, catalog, _rows = packet
    assert len(catalog) == 420
    assert Counter(row["mechanism_family"] for row in catalog) == {
        "TWELL_ATOMIC": 380,
        "TWELL_COMPOUND": 20,
        "GP01": 7,
        "GRAVITY_LIGHT_ONTOLOGY": 13,
    }
    assert len(predecessor["objects"]["SPARC"]) == 139
    assert len(predecessor["objects"]["XCOP"]) == 8


def test_exact_dimensionality_matrix(packet: tuple) -> None:
    config, _predecessor, _catalog, rows = packet
    assert len(rows) == config["dimensionality_matrix_contract"]["expected_rows"] == 61_740
    assert rows[0]["mechanism_id"] == "TW2-A01-D01"
    assert rows[0]["domain"] == "SPARC"
    assert rows[-1]["mechanism_id"] == "QG13"
    assert rows[-1]["domain"] == "XCOP"
    assert len({row["object_id"] for row in rows}) == 147


def test_exact_row_schema_and_zero_authority(packet: tuple) -> None:
    config, _predecessor, _catalog, rows = packet
    expected = set(config["dimensionality_matrix_contract"]["row_fields"])
    assert all(set(row) == expected for row in rows)
    assert not any(row["data_eligible"] for row in rows)
    assert not any(row["scored"] for row in rows)


def test_no_real_object_is_overclaimed_as_full_3d_ready(packet: tuple) -> None:
    _config, _predecessor, _catalog, rows = packet
    counts = Counter(row["current_disposition"] for row in rows)
    assert counts["FULL_3D_SOURCE_READY"] == 0
    assert all(
        row["current_source_dimensionality"] == "RADIAL_SOURCE_CURVE_ONLY"
        for row in rows
        if row["domain"] == "SPARC"
    )
    assert all(
        row["current_source_dimensionality"] == "SPHERICAL_1D"
        for row in rows
        if row["domain"] == "XCOP"
    )


def test_ontology_and_action_are_not_forced_into_data_scores(packet: tuple) -> None:
    _config, _predecessor, _catalog, rows = packet
    ontology = [row for row in rows if row["mechanism_family"] == "GRAVITY_LIGHT_ONTOLOGY"]
    action = [row for row in rows if row["mechanism_id"] == "GP01-ACTION_PLACEHOLDER"]
    assert len(ontology) == 13 * 147
    assert {row["current_disposition"] for row in ontology} == {"THEORY_ONLY"}
    assert len(action) == 147
    assert {row["current_disposition"] for row in action} == {"INCOMPLETE_QUARANTINE"}


def test_history_and_environment_fail_for_the_right_reason(packet: tuple) -> None:
    _config, _predecessor, _catalog, rows = packet
    telegraph = [row for row in rows if row["mechanism_id"] == "GP01-TELEGRAPH"]
    environmental = [
        row
        for row in rows
        if "D12_ENV" in row["drivers"]
        and row["mechanism_family"].startswith("TWELL")
        and row["architecture"] not in contract._HISTORY_ARCHITECTURES
        and row["architecture"] not in contract._STOCHASTIC_ARCHITECTURES
        and not (set(row["drivers"]) & contract._HISTORY_DRIVERS)
    ]
    assert {row["current_disposition"] for row in telegraph} == {"SOURCE_BLOCKED_MISSING_HISTORY"}
    assert {row["current_disposition"] for row in environmental} == {
        "SOURCE_BLOCKED_MISSING_ENVIRONMENT"
    }


def test_source_contract_hash_is_object_independent_within_mechanism_domain(
    packet: tuple,
) -> None:
    _config, _predecessor, _catalog, rows = packet
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        grouped.setdefault((row["mechanism_id"], row["domain"]), set()).add(
            row["source_contract_sha256"]
        )
    assert len(grouped) == 840
    assert all(len(values) == 1 for values in grouped.values())


def test_stream_is_deterministic(packet: tuple) -> None:
    config, predecessor, catalog, rows = packet
    first = contract._stream_root(rows)
    second = contract._stream_root(contract.iter_dimensionality_rows(config, predecessor, catalog))
    assert first == second
    assert first[0] == 61_740


def test_all_synthetic_fixture_contract_checks_pass(packet: tuple) -> None:
    config, _predecessor, _catalog, _rows = packet
    checks = contract.synthetic_contract_checks(config)
    assert len(checks) == 15
    assert all(checks.values())


@pytest.mark.parametrize("section", contract._STRICT_SECTIONS)
def test_every_semantic_section_is_fail_closed(packet: tuple, section: str) -> None:
    config, _predecessor, _catalog, _rows = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(contract.ThreeDContractError, match=f"section {section} changed"):
        contract.validate_config(changed)


def test_section_seal_rebinding_is_rejected(packet: tuple) -> None:
    config, _predecessor, _catalog, _rows = packet
    changed = copy.deepcopy(config)
    changed["authority_boundary"]["campaign_execution_authorized"] = True
    changed["section_sha256"]["authority_boundary"] = contract.content_sha256(
        changed["authority_boundary"]
    )
    with pytest.raises(contract.ThreeDContractError, match="authority.*changed"):
        contract.validate_config(changed)


def test_config_raw_pin_rejects_coherent_file_replacement(
    packet: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _predecessor, _catalog, _rows = packet
    forged = copy.deepcopy(config)
    forged["purpose"] = "overclaim"
    path = tmp_path / contract.CONFIG_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr(contract, "_ROOT", tmp_path)
    with pytest.raises(contract.ThreeDContractError):
        contract.validate_local_integrity()


def test_noncanonical_output_is_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attacker = tmp_path / "response-bearing.json"
    attacker.write_text('{"private":"value"}', encoding="utf-8")
    reads = 0

    def forbidden_read(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(contract, "OUTPUT_PATH", attacker)
    monkeypatch.setattr(contract, "_read_json", forbidden_read)
    with pytest.raises(contract.ThreeDContractError, match="output path changed"):
        contract.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple) -> None:
    _config, _predecessor, _catalog, _rows = packet
    receipt = contract.build_receipt()
    contract.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["dimensionality_matrix"]["full_3d_source_ready_rows"] = 61_740
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = contract.content_sha256(body)
    with pytest.raises(contract.ThreeDContractError, match="deterministic rebuild"):
        contract.validate_receipt_payload(forged)


def test_receipt_claim_ceiling_and_zero_access(packet: tuple) -> None:
    _config, _predecessor, _catalog, _rows = packet
    receipt = contract.build_receipt()
    assert receipt["access_accounting"] == {
        "scientific_response_files_opened": 0,
        "scientific_response_rows_opened": 0,
        "source_payload_files_opened": 0,
        "source_payload_rows_opened": 0,
        "scores_computed": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
    }
    assert "any 3-D field solver" in receipt["claim_boundary"]["does_not_establish"]
