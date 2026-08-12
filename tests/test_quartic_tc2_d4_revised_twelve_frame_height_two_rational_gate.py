from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_revised_twelve_frame_height_two_rational_gate import (
    EXPECTED_EXACT_GATE_SHA256,
    EXPECTED_EXTENSION_SHA256,
    EXPECTED_TARGET_SHA256,
    RevisedTwelveFrameHeightTwoRationalError,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-revised-twelve-frame-height-two-rational-gate/"
    "campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_one_preregistered_second_height_two_point(artifact: dict) -> None:
    gate = artifact["exact_gate"]
    assert gate["search_protocol"]["preregistered_points"] == [
        {"chart": "primary_e1_stereographic", "coordinates": ["1", "-2"]}
    ]
    assert gate["search_protocol"]["points_evaluated"] == 1
    assert gate["search_protocol"]["stopped_after_first_exact_point_and_class"] is True
    assert gate["selector"]["direction"] == ["-2/3", "1/3", "-2/3"]


def test_full_recurrence_and_all_candidates(artifact: dict) -> None:
    gate = artifact["exact_gate"]
    assert gate["full_recurrence"]["orders_checked"] == [1, 2, 3, 4]
    assert gate["full_recurrence"]["directional_polarization_evaluations"] == 15
    classification = gate["exact_rational_classification"]
    assert classification["candidate_conditions_checked"] == 12
    assert classification["candidate_obstructions"] == 12
    assert classification["candidate_compatibilities"] == 0
    assert classification["eta_normalized_target_sha256"] == EXPECTED_TARGET_SHA256
    assert _content_hash(gate) == EXPECTED_EXACT_GATE_SHA256


def test_preregistered_envelope_and_repair_boundary(artifact: dict) -> None:
    bounded = artifact["exact_gate"]["bounded_classification"]
    assert bounded["preregistered_repair_class"] == {
        "transverse_curl_state_indices": list(range(11, 33)),
        "deterministic_decomposition": "lexicographic_first_nonzero_skew_pivot",
        "maximum_elementary_curl_channels": 4,
        "scalar_envelope_degrees": [0, 2, 4, 6],
        "degree_six_full_support_ceiling": 28,
    }
    envelope = bounded["envelope"]
    assert envelope["declared_degree_ladder"] == [0, 2, 4, 6]
    assert envelope["zero_at_all_twelve_predecessor_frames"] is True
    repair = bounded["repair"]
    assert repair["classification"] == "bounded_local_repair_constructed"
    assert repair["extension_sha256"] == EXPECTED_EXTENSION_SHA256
    assert repair["local_certificate_constructed"] is True
    assert repair["total_local_direction_certificates"] == 13


def test_counts_boundary_and_global_claims_fail_closed(artifact: dict) -> None:
    assert artifact["counts"] == {
        "bound_predecessors": 1,
        "candidate_compatibilities": 0,
        "candidate_conditions_checked": 12,
        "candidate_obstructions": 12,
        "directional_polarization_evaluations": 15,
        "envelope_degrees_checked": 4,
        "inferred_global_passes": 0,
        "inherited_bound_artifacts_verified": 11,
        "negative_controls": 8,
        "new_local_direction_certificates": 1,
        "prior_direction_constraints": 12,
        "rational_SO3_charts": 2,
        "recurrence_orders_checked": 4,
        "regular_search_points_evaluated": 1,
        "total_local_direction_certificates": 13,
    }
    assert artifact["next_gate"].startswith("Preregister exactly one further bounded")
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())
    for key in (
        "finite_selector_determines_full_direction_sphere",
        "full_direction_sphere_D4_compatibility_proved",
        "TC2_closed",
        "full_tube_Sylvester_identity",
        "CK1_closed",
        "CK3_closed",
        "B7_closed",
        "global_H7_closed",
        "lifespan_proved",
    ):
        assert artifact["claims"][key] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("scope",), "global theorem"),
        (("next_gate",), "complete"),
        (("config_sha256",), "0" * 64),
        (("config_file_sha256",), "0" * 64),
        (("counts", "candidate_obstructions"), 11),
        (("counts", "inferred_global_passes"), 1),
        (("negative_controls", "infer_full_direction_sphere", "rejected"), False),
        (("source_bindings", "degree_six_predecessor", "content_sha256"), "0" * 64),
        (("source_bindings", "degree_six_predecessor", "file_sha256"), "0" * 64),
        (("source_bindings", "degree_six_predecessor", "path"), "evil.json"),
        (("source_bindings", "campaign_source", "file_sha256"), "0" * 64),
        (("source_bindings", "campaign_source", "path"), "evil.py"),
        (("source_bindings", "campaign_test", "file_sha256"), "0" * 64),
        (("source_bindings", "campaign_test", "path"), "evil.py"),
        (("exact_gate", "selector", "chart_coordinates"), ["-1", "2"]),
        (("exact_gate", "full_recurrence", "orders_checked"), [4]),
        (
            (
                "exact_gate",
                "bounded_classification",
                "preregistered_repair_class",
                "maximum_elementary_curl_channels",
            ),
            5,
        ),
        (
            ("exact_gate", "bounded_classification", "envelope", "declared_degree_ladder"),
            [0, 2, 4, 6, 8],
        ),
        (
            ("exact_gate", "bounded_classification", "envelope", "deterministic_support_indices"),
            [1, 2, 3],
        ),
        (
            ("exact_gate", "bounded_classification", "range", "transverse_selector_rank"),
            21,
        ),
        (
            (
                "exact_gate",
                "bounded_classification",
                "range",
                "selector_target_plane_intersection_dimension",
            ),
            3,
        ),
        (
            ("exact_gate", "bounded_classification", "repair", "elementary_curl_channels"),
            3,
        ),
        (
            ("exact_gate", "bounded_classification", "repair", "coordinate_pairs"),
            [[11, 21]],
        ),
        (
            ("exact_gate", "bounded_classification", "repair", "extension_sha256"),
            "0" * 64,
        ),
        (("claims", "TC2_closed"), True),
    ],
)
def test_resealed_semantic_mutation_rejected(
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
    with pytest.raises(RevisedTwelveFrameHeightTwoRationalError):
        validate_campaign(mutated)


def test_resealed_unknown_keys_rejected(artifact: dict) -> None:
    for path in (
        ("top",),
        ("gate",),
        ("candidate",),
        ("repair_candidate",),
        ("source_binding",),
        ("test_binding",),
        ("predecessor_binding",),
    ):
        mutated = copy.deepcopy(artifact)
        if path == ("top",):
            mutated["unknown"] = True
        elif path == ("gate",):
            mutated["exact_gate"]["unknown"] = True
        elif path == ("candidate",):
            mutated["exact_gate"]["exact_rational_classification"]["candidate_records"][0][
                "unknown"
            ] = True
        elif path == ("repair_candidate",):
            mutated["exact_gate"]["bounded_classification"]["repair"]["candidate_records"][0][
                "unknown"
            ] = True
        elif path == ("source_binding",):
            mutated["source_bindings"]["campaign_source"]["unknown"] = True
        elif path == ("test_binding",):
            mutated["source_bindings"]["campaign_test"]["unknown"] = True
        else:
            mutated["source_bindings"]["degree_six_predecessor"]["unknown"] = True
        mutated["content_sha256"] = _content_hash(
            {key: item for key, item in mutated.items() if key != "content_sha256"}
        )
        with pytest.raises(RevisedTwelveFrameHeightTwoRationalError):
            validate_campaign(mutated)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("global_claim_policy",), "allow_global"),
        (("search_schedule", 0, "coordinates"), ["-1", "2"]),
        (("bounded_envelope_degrees",), [0, 2, 4, 6, 8]),
        (("bounded_repair_class", "maximum_elementary_curl_channels"), 5),
    ],
)
def test_resealed_config_semantic_mutation_rejected(
    artifact: dict,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[object, ...],
    value: object,
) -> None:
    from sigma_theory_compiler import (
        quartic_tc2_d4_revised_twelve_frame_height_two_rational_gate as gate_module,
    )

    config_path = ROOT / gate_module.CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cursor: object = config
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    config["content_sha256"] = _content_hash(
        {key: item for key, item in config.items() if key != "content_sha256"}
    )
    temporary = ROOT / ".pytest-mutated-twelve-frame-config.json"
    temporary.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(gate_module, "CONFIG_PATH", temporary.name)
    try:
        with pytest.raises(RevisedTwelveFrameHeightTwoRationalError):
            validate_campaign(artifact)
    finally:
        temporary.unlink()


def test_resealed_config_unknown_key_rejected(
    artifact: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sigma_theory_compiler import (
        quartic_tc2_d4_revised_twelve_frame_height_two_rational_gate as gate_module,
    )

    config_path = ROOT / gate_module.CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["unknown"] = True
    config["content_sha256"] = _content_hash(
        {key: item for key, item in config.items() if key != "content_sha256"}
    )
    temporary = ROOT / ".pytest-mutated-twelve-frame-config.json"
    temporary.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(gate_module, "CONFIG_PATH", temporary.name)
    try:
        with pytest.raises(RevisedTwelveFrameHeightTwoRationalError):
            validate_campaign(artifact)
    finally:
        temporary.unlink()
