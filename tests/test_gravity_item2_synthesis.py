from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.gravity_item2_synthesis as synthesis
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict[str, object]:
    return json.loads((ROOT / synthesis.OUTPUT_PATH).read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def test_synthesis_is_exact_hash_bound_scoped_reject() -> None:
    stored = _load()
    assert synthesis.build_receipt(ROOT) == stored
    synthesis.validate_receipt(stored, root=ROOT)
    assert stored["status"] == "REJECT"
    assert stored["decision"] == "REJECT_ITEM2_TESTED_PROJECTED_SHAPE_FAMILIES_ADVANCE_ITEM3"
    assert len(stored["attempt_bindings"]) == 5
    assert all(stored["coverage_checks"].values())
    assert not any(stored["survivor_checks"].values())


def test_synthesis_preserves_confirmation_and_claim_boundaries() -> None:
    stored = _load()
    assert stored["boundary_checks"]["total_confirmation_target_accesses"] == 0
    assert stored["boundary_checks"]["all_confirmation_boundaries_untouched"] is True
    assert stored["claims"] == {
        "all_anisotropic_gravity_rejected": False,
        "alternative_to_gr_established": False,
        "dark_matter_eliminated": False,
        "historical_novelty_established": False,
        "item_2_tested_scope_complete": True,
        "roadmap_item_3_authorized_by_order": True,
        "shape_can_never_matter": False,
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("coverage_checks", "intermediate_scale_groups_tested", False),
        ("survivor_checks", "any_family_survives_all_frozen_robustness_controls", True),
        ("boundary_checks", "all_confirmation_boundaries_untouched", False),
        ("claims", "all_anisotropic_gravity_rejected", True),
        ("claims", "alternative_to_gr_established", True),
    ],
)
def test_resealed_invalid_closure_is_rejected(section: str, key: str, value: bool) -> None:
    stored = copy.deepcopy(_load())
    stored[section][key] = value
    with pytest.raises(synthesis.GravityItem2SynthesisError):
        synthesis.validate_receipt(_reseal(stored), root=ROOT)


def test_failure_space_retains_all_five_attempts() -> None:
    stored = _load()
    attempts = {
        attempt
        for family in stored["failure_space"]
        for attempt in family["attempts"]
    }
    assert attempts == {1, 2, 3, 4, 5}
    assert len(stored["scoped_rejection"]["not_rejected"]) >= 5
