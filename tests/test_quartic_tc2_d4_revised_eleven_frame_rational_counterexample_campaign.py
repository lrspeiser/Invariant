from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_revised_eleven_frame_rational_counterexample_campaign import (
    EXPECTED_EXACT_GATE_SHA256,
    FALSE_CLAIMS,
    TRUE_CLAIMS,
    RevisedElevenFrameRationalCounterexampleError,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-revised-eleven-frame-rational-counterexample-campaign/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_next_signed_height_one_point(artifact: dict) -> None:
    gate = artifact["exact_gate"]
    assert gate["search_protocol"]["points_evaluated"] == 1
    assert gate["search_protocol"]["stopped_at_first_regular_obstruction"] is True
    assert gate["search_protocol"]["remaining_bounded_selector_points_after_obstruction"] == 0
    assert gate["search_protocol"]["preregistered_signed_height_one_selector_exhausted"] is True
    selector = gate["first_obstruction"]["selector"]
    assert selector["chart_coordinates"] == ["-1", "-1"]
    assert selector["direction"] == ["-1/3", "-2/3", "-2/3"]


def test_full_recurrence_all_candidates_and_exact_hash(artifact: dict) -> None:
    first = artifact["exact_gate"]["first_obstruction"]
    assert first["full_recurrence"]["orders_checked"] == [1, 2, 3, 4]
    assert first["full_recurrence"]["directional_polarization_evaluations"] == 15
    obstruction = first["exact_rational_obstruction"]
    assert obstruction["candidate_conditions_checked"] == 12
    assert obstruction["candidate_compatibilities"] == 0
    assert obstruction["candidate_obstructions"] == 12
    assert obstruction["distinct_eta_normalized_targets"] == 1
    assert obstruction["eta_normalized_target_rank"] == 2
    assert obstruction["eta_normalized_target_nonzero_entries"] == 16
    assert all(
        set(row["nonzero_equal_eigenspace_compressions"]) == {"0"}
        and row["zero_speed_cleared_numerator"]["numerator_rank"] > 0
        for row in obstruction["candidate_records"]
    )
    assert _content_hash(artifact["exact_gate"]) == EXPECTED_EXACT_GATE_SHA256


def test_bounded_envelope_class_exhausted(artifact: dict) -> None:
    bounded = artifact["exact_gate"]["bounded_next_escape"]
    envelope = bounded["minimal_preserving_envelope"]
    assert envelope["support_search_maximum"] == 14
    assert envelope["total_supports_checked"] == 32766
    assert envelope["degree_four"]["prior_zero_constraint_rank"] == 11
    assert envelope["degree_four"]["prior_zero_nullity"] == 4
    assert envelope["bounded_class_exhausted"] is True
    assert envelope["repair_constructed"] is False
    assert all(value == 0 for value in envelope["feasible_envelopes_by_support_size"].values())
    assert envelope["zero_constraints_cover_all_eleven_predecessor_frames"] is True
    assert bounded["exact_range_classification"]["quotient_target_zero"] is True
    repair = bounded["bounded_repair_classification"]
    assert repair["transverse_curl_target_range_compatible"] is True
    assert repair["preserving_envelope_exists_in_declared_class"] is False
    assert repair["twelfth_local_certificate_constructed"] is False
    assert repair["total_local_direction_certificates"] == 11


def test_closed_world_claims_and_controls(artifact: dict) -> None:
    assert set(artifact["claims"]) == TRUE_CLAIMS | FALSE_CLAIMS
    assert all(artifact["claims"][key] is True for key in TRUE_CLAIMS)
    assert all(artifact["claims"][key] is False for key in FALSE_CLAIMS)
    assert len(artifact["negative_controls"]) == 10
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("counts", "candidate_obstructions"), 11),
        (("counts", "inferred_global_passes"), 1),
        (("exact_gate", "search_protocol", "stopped_at_first_regular_obstruction"), False),
        (("exact_gate", "first_obstruction", "selector", "direction"), ["1", "0", "0"]),
        (
            (
                "exact_gate", "first_obstruction", "full_recurrence",
                "revised_global_symbol_sha256",
            ),
            "0" * 64,
        ),
        (
            (
                "exact_gate", "first_obstruction", "exact_rational_obstruction",
                "candidate_records", 0, "zero_speed_cleared_numerator", "numerator_sha256",
            ),
            "0" * 64,
        ),
        (
            (
                "exact_gate", "bounded_next_escape", "minimal_preserving_envelope",
                "bounded_class_exhausted",
            ),
            False,
        ),
        (
            (
                "exact_gate", "bounded_next_escape", "bounded_repair_classification",
                "twelfth_local_certificate_constructed",
            ),
            True,
        ),
        (("claims", "full_direction_sphere_D4_compatibility_proved"), True),
    ],
)
def test_resealed_semantic_tamper_rejected(
    artifact: dict, path: tuple[object, ...], value: object
) -> None:
    mutated = copy.deepcopy(artifact)
    cursor: object = mutated
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    mutated["content_sha256"] = _content_hash(
        {key: item for key, item in mutated.items() if key != "content_sha256"}
    )
    with pytest.raises(RevisedElevenFrameRationalCounterexampleError):
        validate_campaign(mutated)


def test_unknown_claim_rejected(artifact: dict) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["claims"]["unknown"] = False
    mutated["content_sha256"] = _content_hash(
        {key: item for key, item in mutated.items() if key != "content_sha256"}
    )
    with pytest.raises(RevisedElevenFrameRationalCounterexampleError):
        validate_campaign(mutated)
