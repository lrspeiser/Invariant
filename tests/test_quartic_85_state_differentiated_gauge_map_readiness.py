from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_differentiated_gauge_map_readiness import (
    Quartic85StateDifferentiatedGaugeReadinessError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_differentiated_gauge_map_readiness.json"
OUTPUT = ROOT / ("runs/math/quartic-85-state-differentiated-gauge-map-readiness/receipt.json")


def test_exact_missing_jet_inventory_and_chunk_plan() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == ("PASS_RESUMABLE_READINESS_CONTRACT_FIVE_PRIMITIVE_JET_BLOCKERS")
    inventory = receipt["materialization"]["primitive_inventory"]
    assert {item["family"]: item["missing_slots"] for item in inventory} == {
        "hat_inverse_first": 40,
        "tilde_inverse_second": 100,
        "reference_connection_second": 400,
        "gauge_source_second": 40,
        "physical_metric_third": 200,
    }
    assert sum(item["missing_slots"] for item in inventory) == 780
    assert sum(item["chunk_count"] for item in inventory) == 48
    assert all(item["status"] == "BLOCK_MISSING_REGISTRATION" for item in inventory)


def test_constructible_subset_and_product_rule_are_bounded() -> None:
    materialization = build_receipt(CONFIG, root=ROOT)["materialization"]
    constructible = materialization["constructible_subset"]
    assert len(constructible["exact_source_packets"]) == 3
    assert len(constructible["exact_algebraic_reductions"]) == 4
    assert constructible["cold_symbolic_work_executed"] is False
    shell = materialization["product_rule_shell"]
    assert shell["status"] == "PASS_DEPENDENCY_SHELL_ONLY"
    assert len(shell["branches"]) == 3
    metric_branch = next(
        item for item in shell["branches"] if item["branch"] == "physical_inverse_derivative"
    )
    assert metric_branch["constructible_subset"] == (
        "exactly zero by physical metric compatibility"
    )
    assert metric_branch["missing"] is None


def test_resume_dag_has_no_false_completion() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    units = receipt["materialization"]["resume_units"]
    assert [item["unit_id"] for item in units] == [
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
    assert units[3]["status"] == "BLOCK_PRIMITIVE_PACKETS_ABSENT"
    assert units[-1]["depends_on"] == ["R7"]
    assert receipt["counts"]["cold_symbolic_runs"] == 0
    claims = receipt["claims"]
    assert claims["differentiated_gauge_map_readiness_contract_closed"] is True
    assert claims["primitive_differentiated_formulation_jets_registered"] is False
    assert claims["differentiated_gauge_map_in_85_state_coordinates_closed"] is False
    assert claims["constraint_propagation_closed"] is False
    assert claims["gravity_h7_theorem_established"] is False


def test_checked_receipt_is_path_free_and_content_addressed() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    contract = {
        key: value
        for key, value in receipt["materialization"].items()
        if key != "readiness_contract_sha256"
    }
    assert receipt["materialization"]["readiness_contract_sha256"] == _canonical_sha(contract)
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
        "off_shell_divergence_gate",
        "gauge_source_formula",
        "constraint_coordinate_basis",
        "coordinate_two_jet_tube",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateDifferentiatedGaugeReadinessError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_invalid_chunk_plan_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["primitive_chunk_sizes"]["reference_connection_second"] = 21
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateDifferentiatedGaugeReadinessError, match="chunk plan"):
        build_receipt(candidate, root=ROOT)


def test_broadened_construction_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["differentiated_gauge_map_constructed"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateDifferentiatedGaugeReadinessError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
