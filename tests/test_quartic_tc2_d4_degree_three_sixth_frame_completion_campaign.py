from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_degree_three_sixth_frame_completion_campaign import (
    FALSE_CLAIMS,
    TRUE_CLAIMS,
    SixthFrameCompletionError,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/backgrounds/quartic_tc2_d4_degree_three_sixth_frame_completion_campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_campaign(ROOT, CONFIG)


def test_exact_selector_and_counts(artifact: dict) -> None:
    assert artifact["exact_completion"]["selector"]["direction"] == ["2/3", "1/3", "2/3"]
    assert artifact["counts"]["directional_recurrence_evaluations"] == 15
    assert artifact["counts"]["prior_candidate_obstructions"] == 12
    assert artifact["counts"]["total_certified_directions"] == 6


def test_exact_range_and_sharp_completion(artifact: dict) -> None:
    result = artifact["exact_completion"]
    assert result["exact_range_classification"]["normalized_target_rank"] == 4
    assert result["exact_range_classification"]["transverse_selector_rank"] == 22
    assert result["exact_range_classification"]["quotient_target_zero"] is True
    assert result["minimal_rank_two_completion"]["lower_bound_completion_rank"] == 2
    assert result["minimal_rank_two_completion"]["coordinate_pairs"] == [[11, 21], [15, 32]]


def test_unique_degree_three_preserving_extension(artifact: dict) -> None:
    extension = artifact["exact_completion"]["exact_sphere_extension"]
    assert extension["envelope"] == "a6(n)=(3/2)*n3*(4*n1+n2-3*n3)"
    assert extension["envelope_coefficient_system_rank"] == 6
    assert extension["minimal_total_extension_degree"] == 3
    assert extension["unique_under_five_zero_values_and_one_normalization"] is True
    assert extension["five_prior_direction_extensions_zero"] is True
    assert extension["physical_gradient_lift_annihilated_identically"] is True


def test_all_candidates_close(artifact: dict) -> None:
    result = artifact["exact_completion"]["corrected_result"]
    assert result["candidate_compatibilities"] == 12
    assert result["candidate_obstructions"] == 0
    assert len(result["candidate_records"]) == 12
    assert all(row["D4_Sylvester_solvable"] for row in result["candidate_records"])


def test_claims_are_closed_world_and_fail_closed(artifact: dict) -> None:
    assert set(artifact["claims"]) == TRUE_CLAIMS | FALSE_CLAIMS
    assert all(artifact["claims"][key] is True for key in TRUE_CLAIMS)
    assert all(artifact["claims"][key] is False for key in FALSE_CLAIMS)


def test_negative_controls(artifact: dict) -> None:
    assert len(artifact["negative_controls"]) == 9
    assert all(value["rejected"] for value in artifact["negative_controls"].values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("counts", "total_certified_directions"), 7),
        (("exact_completion", "selector", "direction"), ["1", "0", "0"]),
        (("exact_completion", "exact_range_classification", "normalized_target_sha256"), "0" * 64),
        (("exact_completion", "minimal_rank_two_completion", "coordinate_pairs"), [[11, 21]]),
        (("exact_completion", "corrected_result", "candidate_compatibilities"), 11),
        (("claims", "full_direction_sphere_D4_compatibility_proved"), True),
    ],
)
def test_semantic_tamper_rejected(artifact: dict, path: tuple[str, ...], value: object) -> None:
    mutated = copy.deepcopy(artifact)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    body = {key: item for key, item in mutated.items() if key != "content_sha256"}
    from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

    mutated["content_sha256"] = _content_hash(body)
    with pytest.raises(SixthFrameCompletionError):
        validate_campaign(mutated)


def test_unknown_claim_rejected(artifact: dict) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["claims"]["unknown"] = False
    body = {key: item for key, item in mutated.items() if key != "content_sha256"}
    from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

    mutated["content_sha256"] = _content_hash(body)
    with pytest.raises(SixthFrameCompletionError):
        validate_campaign(mutated)


def test_serialized_artifact_is_canonical(artifact: dict) -> None:
    encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    assert 'finite_selector_determines_full_direction_sphere":false' in encoded
