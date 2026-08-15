from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_85_state_resonant_projector_compatibility_gate import (
    Quartic85StateResonantCompatibilityError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_85_state_resonant_projector_compatibility_gate.json"
OUTPUT = ROOT / "runs/math/quartic-85-state-resonant-projector-compatibility-gate/receipt.json"


def test_exact_flat_reference_resonant_compatibility_passes() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_EXACT_FLAT_E1_RESONANT_COMPATIBILITY_ALL_B_COMPONENTS"
    assert len(receipt["materialization"]["resonant_projections"]) == 8
    assert {
        (item["root"], item["potential_component"])
        for item in receipt["materialization"]["resonant_projections"]
    } == {(str(root), f"B_{component}") for root in (-1, 1) for component in range(4)}
    assert all(
        item["nonzero_entries"] == 0 for item in receipt["materialization"]["resonant_projections"]
    )


def test_basis_packets_and_counts_are_exact() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    material = receipt["materialization"]
    basis = material["basis"]
    assert sorted(basis["gravity_block_to_coupled85"] + basis["matter_block_to_coupled85"]) == list(
        range(85)
    )
    assert material["vacuum"]["K55"]["shape"] == [55, 55]
    assert material["vacuum"]["K55"]["nonzero_count"] == 131
    assert {
        key: packet["nonzero_count"] for key, packet in material["vacuum"]["projectors"].items()
    } == {"-1": 24, "1": 24}
    assert material["matter"]["Hm"]["shape"] == [30, 30]
    assert material["matter"]["Hm"]["nonzero_count"] == 30
    assert [
        packet["nonzero_count"] for packet in material["maxwell_cross_block"]["coefficient_packets"]
    ] == [10, 10, 4, 4]


def test_corruption_negative_is_constructive() -> None:
    negative = build_receipt(CONFIG, root=ROOT)["materialization"]["corruption_negative"]
    assert negative["rejected"] is True
    assert negative["mutation"] == "add 1 to C^(0)[19,40] in the registered block basis"
    assert [item["nonzero_entries"] for item in negative["results"]] == [8, 8]
    assert [item["first_nonzero"]["value"] for item in negative["results"]] == ["1/8", "1/8"]


def test_claims_and_content_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["counts"]["resonant_projection_entries_checked"] == 13200
    assert receipt["counts"]["resonant_projection_nonzero_entries"] == 0
    assert receipt["claims"]["exact_flat_reference_e1_basis_materialized"] is True
    assert receipt["claims"]["exact_resonant_compatibility_all_four_potential_components"] is True
    assert not any(
        value
        for name, value in receipt["claims"].items()
        if name
        not in {
            "exact_flat_reference_e1_basis_materialized",
            "exact_resonant_compatibility_all_four_potential_components",
        }
    )
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == _canonical_sha(body)


def test_checked_receipt_is_current_and_path_free() -> None:
    checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert checked == build_receipt(CONFIG, root=ROOT)
    rendered = json.dumps(checked, sort_keys=True)
    assert str(ROOT) not in rendered
    assert "\\\\" not in rendered


@pytest.mark.parametrize(
    "binding",
    [
        "coupled_85_state_reduction",
        "flat_vacuum_K55",
        "flat_vacuum_P55",
        "maxwell_mixed_principal",
        "symmetrizer_blocker",
    ],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateResonantCompatibilityError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["flat_vacuum_K55"]["path"] = "runs/math/missing-k55.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateResonantCompatibilityError, match="cannot read bound file"):
        build_receipt(candidate, root=ROOT)


def test_broadened_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["full_coupled_symmetrizer"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(Quartic85StateResonantCompatibilityError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
