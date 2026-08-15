from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_candidate_gravity_constraint_basis_gate import (
    Quartic85StateCandidateGravityConstraintBasisError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_candidate_gravity_constraint_basis_gate.json"
OUTPUT = ROOT / ("runs/math/quartic-85-state-candidate-gravity-constraint-basis-gate/receipt.json")


def test_exact_85_state_kinematic_matter_basis_is_hash_bound() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == (
        "BOUNDED_PASS_KINEMATIC_MATTER_BASIS_TYPED_BLOCK_GRAVITY_COORDINATE_MAP"
    )
    materialization = receipt["materialization"]
    shared = materialization["shared_kinematic_matter_basis"]
    assert len(shared["field_basis"]) == 17
    assert len(shared["state_coordinates"]) == 85
    assert len(shared["definition_rows"]) == 51
    assert len(shared["curl_rows"]) == 51
    assert materialization["shared_kinematic_matter_basis_sha256"] == _canonical_sha(shared)
    candidates = materialization["candidate_results"]
    assert len(candidates) == 12
    assert len({item["constraint_coordinate_manifest_sha256"] for item in candidates}) == 12
    assert all(item["kinematic_constraint_rows"] == 102 for item in candidates)
    assert all(item["gravity_sector_kinematic_rows"] == 66 for item in candidates)
    assert all(item["matter_sector_kinematic_rows"] == 36 for item in candidates)
    assert all(item["physical_gravity_constraint_coordinate_rows"] == 0 for item in candidates)


def test_state_indices_and_flat_maxwell_row_are_exact() -> None:
    shared = build_receipt(CONFIG, root=ROOT)["materialization"]["shared_kinematic_matter_basis"]
    coordinates = shared["state_coordinates"]
    assert [item["state_index"] for item in coordinates] == list(range(85))
    assert coordinates[0]["coordinate"] == "q[g_00]"
    assert coordinates[29]["coordinate"] == "v[B_0]"
    assert coordinates[47]["coordinate"] == "w_1[B_1]"
    assert coordinates[65]["coordinate"] == "w_2[B_2]"
    assert coordinates[83]["coordinate"] == "w_3[B_3]"
    maxwell = shared["flat_maxwell_constraint_row"]
    assert [item["state_index"] for item in maxwell["state_terms"]] == [29, 47, 65, 83]
    assert [item["coefficient"] for item in maxwell["state_terms"]] == ["-1", "1", "1", "1"]
    assert maxwell["corruption_negative"]["difference_from_registered_row"] == "-7"
    assert maxwell["corruption_negative"]["rejected"] is True


def test_physical_gravity_coordinate_map_remains_precisely_blocked() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    block = receipt["materialization"]["gravity_coordinate_map_block"]
    assert block["reason_code"] == ("missing_gravity_gauge_adm_to_85_state_coordinate_map")
    assert block["unregistered_gravity_rows_per_candidate"] == 8
    assert block["unregistered_gravity_rows_all_candidates"] == 96
    assert {item["map"] for item in block["missing"]} == {
        "modified_harmonic_C_mu",
        "Hamiltonian_momentum_constraints",
        "coordinate_to_candidate_jet",
    }
    claims = receipt["claims"]
    assert claims["candidate_gravity_constraint_coordinate_basis_closed"] is False
    assert claims["constraint_propagation_closed"] is False
    assert claims["candidate_jet_uniformity_closed"] is False
    assert claims["gravity_h7_theorem_established"] is False


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
        "sourced_constraint_blocker",
        "coupled_85_state_reduction",
        "vacuum_definition_curl",
        "matter_constraint_interface",
        "coupled_field_basis",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateCandidateGravityConstraintBasisError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["coupled_85_state_reduction"]["path"] = "runs/math/absent.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateCandidateGravityConstraintBasisError, match="cannot read"):
        build_receipt(candidate, root=ROOT)


def test_broadened_gravity_basis_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["candidate_gravity_constraint_coordinate_basis"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateCandidateGravityConstraintBasisError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
