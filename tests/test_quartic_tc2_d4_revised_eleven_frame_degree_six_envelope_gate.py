from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_revised_eleven_frame_degree_six_envelope_gate import (
    EXPECTED_EXACT_GATE_SHA256,
    RevisedElevenFrameDegreeSixEnvelopeError,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs/physics-language/quartic-tc2-d4-revised-eleven-frame-degree-six-envelope-gate/campaign.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_single_preregistered_height_two_point(artifact: dict) -> None:
    gate = artifact["exact_gate"]
    assert gate["search_protocol"]["points_evaluated"] == 1
    assert gate["search_protocol"]["stopped_after_first_exact_point_and_class"] is True
    assert gate["selector"]["chart_coordinates"] == ["1", "2"]
    assert gate["selector"]["direction"] == ["-2/3", "1/3", "2/3"]


def test_full_recurrence_and_all_candidates(artifact: dict) -> None:
    gate = artifact["exact_gate"]
    assert gate["full_recurrence"]["orders_checked"] == [1, 2, 3, 4]
    assert gate["full_recurrence"]["directional_polarization_evaluations"] == 15
    classification = gate["exact_rational_classification"]
    assert classification["candidate_conditions_checked"] == 12
    assert classification["candidate_obstructions"] == 12
    assert classification["candidate_compatibilities"] == 0
    assert classification["eta_normalized_target_rank"] == 4
    assert classification["eta_normalized_target_nonzero_entries"] == 56
    assert _content_hash(gate) == EXPECTED_EXACT_GATE_SHA256


def test_bounded_degree_six_classification(artifact: dict) -> None:
    bounded = artifact["exact_gate"]["bounded_classification"]
    envelope = bounded["envelope"]
    assert envelope["declared_degree_ladder"] == [0, 2, 4, 6]
    assert envelope["degree_six_full_support_ceiling"] == 28
    assert envelope["minimal_feasible_even_degree"] == 4
    assert envelope["deterministic_support_indices"] == [1, 2, 5, 7, 9, 10]
    assert envelope["deterministic_support_size"] == 6
    assert envelope["zero_at_all_eleven_predecessor_frames"] is True
    assert bounded["range"]["quotient_target_zero"] is True
    assert bounded["repair"]["transverse_curl_target_range_compatible"] is True
    assert bounded["repair"]["local_certificate_constructed"] is True
    assert bounded["repair"]["total_local_direction_certificates"] == 12


def test_global_claims_fail_closed(artifact: dict) -> None:
    assert artifact["counts"]["inferred_global_passes"] == 0
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())
    for key in (
        "full_direction_sphere_D4_compatibility_proved",
        "TC2_closed",
        "full_tube_Sylvester_identity",
        "global_H7_closed",
        "lifespan_proved",
    ):
        assert artifact["claims"][key] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("config_sha256",), "0" * 64),
        (("scope",), "GLOBAL PDE PROVED"),
        (("next_gate",), "none; complete"),
        (("counts", "candidate_obstructions"), 11),
        (("counts", "inferred_global_passes"), 1),
        (("counts", "rational_SO3_charts"), 999),
        (("counts", "directional_polarization_evaluations"), 0),
        (("counts", "recurrence_orders_checked"), 0),
        (("counts", "envelope_degrees_checked"), 0),
        (("counts", "prior_direction_constraints"), 0),
        (("counts", "negative_controls"), 0),
        (("source_bindings", "exhaustion_predecessor", "path"), "evil.json"),
        (("source_bindings", "exhaustion_predecessor", "content_sha256"), "0" * 64),
        (("negative_controls", "infer_full_direction_sphere", "rejected"), False),
        (("exact_gate", "selector", "chart_coordinates"), ["2", "1"]),
        (("exact_gate", "full_recurrence", "orders_checked"), [4]),
        (("exact_gate", "bounded_classification", "envelope", "degree_six_full_support_ceiling"), 29),
        (("exact_gate", "bounded_classification", "range", "quotient_target_zero"), False),
        (("claims", "TC2_closed"), True),
    ],
)
def test_resealed_tamper_rejected(
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
    with pytest.raises(RevisedElevenFrameDegreeSixEnvelopeError):
        validate_campaign(mutated)


def test_resealed_unknown_top_level_key_rejected(artifact: dict) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["unknown_semantic_field"] = True
    mutated["content_sha256"] = _content_hash(
        {key: item for key, item in mutated.items() if key != "content_sha256"}
    )
    with pytest.raises(RevisedElevenFrameDegreeSixEnvelopeError):
        validate_campaign(mutated)


def test_resealed_negative_controls_wholesale_replacement_rejected(artifact: dict) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["negative_controls"] = {
        f"allow_{index}": {"rejected": False} for index in range(8)
    }
    mutated["content_sha256"] = _content_hash(
        {key: item for key, item in mutated.items() if key != "content_sha256"}
    )
    with pytest.raises(RevisedElevenFrameDegreeSixEnvelopeError):
        validate_campaign(mutated)
