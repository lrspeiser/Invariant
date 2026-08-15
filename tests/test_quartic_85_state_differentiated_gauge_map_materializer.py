from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_differentiated_gauge_map_materializer import (
    Quartic85StateDifferentiatedGaugeMapError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_differentiated_gauge_map_materializer.json"
OUTPUT = ROOT / ("runs/math/quartic-85-state-differentiated-gauge-map-materializer/receipt.json")


def test_all_780_slots_are_checkpointed_without_overlap() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == ("PASS_EXACT_INDEXED_GAUGE_MAP_WITH_FORMAL_EXTERNAL_JET_PACKETS")
    materialization = receipt["materialization"]
    packets = materialization["checkpoint_packets"]
    assert len(packets) == 48
    assert sum(item["slot_count"] for item in packets) == 780
    assert packets[0]["prior_checkpoint_sha256"] == "0" * 64
    assert all(
        packets[index]["prior_checkpoint_sha256"] == packets[index - 1]["checkpoint_sha256"]
        for index in range(1, len(packets))
    )
    assert materialization["final_checkpoint_sha256"] == packets[-1]["checkpoint_sha256"]
    integrity = materialization["packet_integrity_controls"]
    assert integrity["complete_slot_count"] == integrity["unique_slot_count"] == 780
    assert integrity["omitted_last_slot_negative"]["rejected"] is True
    assert integrity["duplicated_first_slot_negative"]["rejected"] is True


def test_physical_metric_third_slots_map_to_85_state_operators() -> None:
    mapping = build_receipt(CONFIG, root=ROOT)["materialization"][
        "physical_metric_third_operator_map"
    ]
    assert len(mapping) == 200
    assert all(item["kind"] == "85_state_differential_operator" for item in mapping)
    all_time = next(item for item in mapping if item["key"] == "d3_g[0,0,0|0,0]")
    assert all_time["state_index"] == 17
    assert all_time["state_coordinate"] == "v[g_00]"
    assert all_time["remaining_derivative_operator"] == [0, 0]
    spatial = next(item for item in mapping if item["key"] == "d3_g[1,2,3|0,0]")
    assert spatial["state_index"] == 34
    assert spatial["state_coordinate"] == "w_1[g_00]"
    assert spatial["remaining_derivative_operator"] == [2, 3]


def test_exact_indexed_formula_program_and_candidate_bindings() -> None:
    materialization = build_receipt(CONFIG, root=ROOT)["materialization"]
    program = materialization["indexed_formula_program"]
    body = {key: value for key, value in program.items() if key != "program_sha256"}
    assert program["program_sha256"] == _canonical_sha(body)
    assert len(program["templates"]) == 17
    assert program["output_components"] == [
        "divQ_lower[0]",
        "divQ_lower[1]",
        "divQ_lower[2]",
        "divQ_lower[3]",
    ]
    candidates = materialization["candidate_results"]
    assert len(candidates) == 12
    assert len({item["manifest_sha256"] for item in candidates}) == 12
    assert all(item["formal_external_jet_atoms"] == 580 for item in candidates)
    assert all(item["fully_expanded_coefficient_rows"] == 0 for item in candidates)
    progress = materialization["readiness_unit_progress"]
    assert [item["unit_id"] for item in progress] == [
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
    ]
    assert progress[6]["status"] == "PASS_EXACT_INDEXED_FOUR_COMPONENT_MAP"
    assert progress[7]["status"].startswith("PARTIAL_")
    assert progress[8]["status"].startswith("PARTIAL_")


def test_claims_keep_external_values_and_propagation_open() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    claims = receipt["claims"]
    assert claims["formal_primitive_packet_registration_closed"] is True
    assert claims["physical_metric_third_to_85_state_operator_map_closed"] is True
    assert claims["exact_indexed_differentiated_gauge_map_closed"] is True
    assert claims["external_formulation_jet_values_certified"] is False
    assert claims["fully_expanded_85_state_coefficient_rows_closed"] is False
    assert claims["constraint_propagation_closed"] is False
    assert claims["gravity_h7_theorem_established"] is False
    boundary = receipt["materialization"]["scientific_boundary"]
    assert boundary["formal_external_jet_atoms_are_values"] is False
    assert boundary["constraint_propagation_inferred"] is False


def test_checked_receipt_is_path_free_and_content_addressed() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == receipt
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize(
    "binding",
    [
        "readiness_contract",
        "off_shell_divergence_gate",
        "gauge_source_formula",
        "constraint_coordinate_basis",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateDifferentiatedGaugeMapError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_broadened_expansion_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["fully_expanded_85_state_coefficient_rows"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateDifferentiatedGaugeMapError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
