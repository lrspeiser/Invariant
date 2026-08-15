from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_twelve_candidate_total_matter_action_binding import (
    QuarticTotalMatterActionBindingError,
    _canonical_sha,
    build_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/quartic_twelve_candidate_total_matter_action_binding.json"
OUTPUT = ROOT / "runs/math/quartic-twelve-candidate-total-matter-action-binding/receipt.json"


def test_all_twelve_total_actions_are_exactly_bound() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["decision"] == "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY"
    results = receipt["candidate_results"]
    assert len(results) == 12
    assert len({item["gravity_action_sha256"] for item in results}) == 12
    assert len({item["total_action_sha256"] for item in results}) == 12
    assert {item["shared_matter_action_sha256"] for item in results} == {
        receipt["shared_matter_action_sha256"]
    }
    for item in results:
        assert item["total_action_sha256"] == _canonical_sha(item["total_action_manifest"])
        assert [gate["outcome"] for gate in item["gate_results"]] == ["PASS", "BLOCK"]
        assert item["omitted_fluid_negative"]["hash_differs"] is True


def test_shared_matter_action_is_three_sector_and_namespace_safe() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    action = receipt["shared_matter_action"]
    assert action["physical_metric"] == "g_mu_nu"
    assert [item["field"] for item in action["components"]] == ["chi_m", "B_mu", "tau"]
    assert [item["sector_id"] for item in action["components"]] == [
        "canonical_minimally_coupled_scalar",
        "source_free_maxwell",
        "barotropic_irrotational_fluid",
    ]
    assert receipt["shared_matter_action_sha256"] == _canonical_sha(action)


def test_counts_claims_and_content_are_bounded() -> None:
    receipt = build_receipt(CONFIG, root=ROOT)
    assert receipt["counts"] == {
        "candidates": 12,
        "shared_matter_sectors": 3,
        "total_action_hash_bindings_passed": 12,
        "unique_gravity_action_hashes": 12,
        "unique_total_action_hashes": 12,
        "omitted_fluid_hash_negatives_passed": 12,
        "sourced_euler_bindings_passed": 0,
        "six_item_contract_items_fully_closed": 0,
        "blocks": 12,
        "rejects": 0,
    }
    claims = receipt["claims"]
    assert claims["all_twelve_total_actions_compositionally_hash_bound"] is True
    assert claims["shared_physical_metric_and_distinct_field_namespaces_bound"] is True
    assert not any(
        value
        for name, value in claims.items()
        if name
        not in {
            "all_twelve_total_actions_compositionally_hash_bound",
            "shared_physical_metric_and_distinct_field_namespaces_bound",
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
    ["census", "universal_matter", "fluid_action", "candidate_actions", "quartic_family_ir"],
)
def test_tampered_binding_fails_closed(tmp_path: Path, binding: str) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"][binding]["file_sha256"] = "0" * 64
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTotalMatterActionBindingError, match="hash mismatch"):
        build_receipt(candidate, root=ROOT)


def test_field_collision_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["field_namespace"]["matter_scalar"] = config["field_namespace"]["gravity_scalar"]
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTotalMatterActionBindingError, match="namespaces collide"):
        build_receipt(candidate, root=ROOT)


def test_missing_binding_fails_as_typed_error(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["bindings"]["fluid_action"]["path"] = "runs/math/missing-fluid-action.json"
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTotalMatterActionBindingError, match="cannot read bound file"):
        build_receipt(candidate, root=ROOT)


def test_broadened_euler_claim_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["claims_policy"]["sourced_gauge_fixed_euler_binding"] = True
    candidate = tmp_path / "config.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTotalMatterActionBindingError, match="claims policy"):
        build_receipt(candidate, root=ROOT)
