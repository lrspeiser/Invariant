from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_revised_nine_frame_rational_counterexample_campaign import (
    EXPECTED_EXACT_GATE_SHA256,
    FALSE_CLAIMS,
    TRUE_CLAIMS,
    RevisedNineFrameRationalCounterexampleError,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-revised-nine-frame-rational-counterexample-campaign/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_next_signed_height_one_fail_fast_point(artifact: dict) -> None:
    gate = artifact["exact_gate"]
    assert gate["atlas"]["union_covers_real_S2"] is True
    assert gate["search_protocol"]["points_evaluated"] == 1
    assert gate["search_protocol"]["stopped_at_first_regular_obstruction"] is True
    assert gate["search_protocol"]["remaining_bounded_selector_points_after_obstruction"] == 2
    selector = gate["first_obstruction"]["selector"]
    assert selector["chart_coordinates"] == ["1", "-1"]
    assert selector["direction"] == ["-1/3", "2/3", "-2/3"]


def test_full_recurrence_and_all_candidates(artifact: dict) -> None:
    first = artifact["exact_gate"]["first_obstruction"]
    assert first["full_recurrence"]["orders_checked"] == [1, 2, 3, 4]
    assert first["full_recurrence"]["directional_polarization_evaluations"] == 15
    obstruction = first["exact_rational_obstruction"]
    assert obstruction["candidate_conditions_checked"] == 12
    assert obstruction["candidate_compatibilities"] == 0
    assert obstruction["candidate_obstructions"] == 12
    assert obstruction["distinct_eta_normalized_targets"] == 1
    assert all(
        set(row["nonzero_equal_eigenspace_compressions"]) == {"0"}
        and row["zero_speed_cleared_numerator"]["numerator_rank"] > 0
        for row in obstruction["candidate_records"]
    )
    assert _content_hash(artifact["exact_gate"]) == EXPECTED_EXACT_GATE_SHA256


def test_bounded_preserving_envelope_classification(artifact: dict) -> None:
    envelope = artifact["exact_gate"]["bounded_next_escape"]["minimal_preserving_envelope"]
    assert envelope["minimal_even_homogeneous_degree"] == 4
    assert envelope["support_search_maximum"] == 10
    assert envelope["sparsest_support_size"] <= 10
    assert envelope["sparsest_support_feasible_envelopes"] > 0
    assert envelope["zero_at_all_nine_predecessor_frames"] is True


def test_bounded_local_escape_closes_only_new_frame(artifact: dict) -> None:
    bounded = artifact["exact_gate"]["bounded_next_escape"]
    assert bounded["exact_range_classification"]["target_in_full_transverse_curl_range"] is True
    assert bounded["exact_range_classification"]["quotient_target_zero"] is True
    completion = bounded["local_completion"]
    assert completion["candidate_conditions_checked"] == 12
    assert completion["candidate_compatibilities"] == 12
    assert completion["candidate_obstructions"] == 0
    assert completion["prior_nine_direction_certificates_preserved"] is True
    assert completion["total_local_direction_certificates"] == 10
    assert completion["constructed_completion_rank"] == 2
    assert completion["elementary_curl_channels"] == 2
    assert completion["gradient_residual_zero"] is True


def test_closed_world_claims_and_negative_controls(artifact: dict) -> None:
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
                "exact_gate",
                "first_obstruction",
                "full_recurrence",
                "revised_global_symbol_sha256",
            ),
            "0" * 64,
        ),
        (
            (
                "exact_gate",
                "first_obstruction",
                "exact_rational_obstruction",
                "candidate_records",
                0,
                "zero_speed_cleared_numerator",
                "numerator_sha256",
            ),
            "0" * 64,
        ),
        (
            (
                "exact_gate",
                "bounded_next_escape",
                "minimal_preserving_envelope",
                "sparsest_support_size",
            ),
            11,
        ),
        (
            (
                "exact_gate",
                "bounded_next_escape",
                "local_completion",
                "candidate_records",
                0,
                "corrected_residual_sha256",
            ),
            "0" * 64,
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
    body = {key: item for key, item in mutated.items() if key != "content_sha256"}
    mutated["content_sha256"] = _content_hash(body)
    with pytest.raises(RevisedNineFrameRationalCounterexampleError):
        validate_campaign(mutated)


def test_unknown_claim_rejected(artifact: dict) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["claims"]["unknown"] = False
    body = {key: item for key, item in mutated.items() if key != "content_sha256"}
    mutated["content_sha256"] = _content_hash(body)
    with pytest.raises(RevisedNineFrameRationalCounterexampleError):
        validate_campaign(mutated)
