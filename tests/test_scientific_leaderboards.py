from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.scientific_leaderboards import (
    CATEGORIES,
    build_scientific_leaderboards,
    load_leaderboard_config,
    validate_scientific_leaderboards,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "scientific_leaderboards.json"


def test_category_local_rankings_keep_missing_evidence_unranked() -> None:
    board = build_scientific_leaderboards(ROOT, load_leaderboard_config(CONFIG))
    assert tuple(board["categories"]) == CATEGORIES
    formal = board["categories"]["formal_adm_dirac"]
    assert formal["top10"][0]["candidate_id"] == "KNOWN-ANSWER-EINSTEIN-AETHER"
    assert formal["top10"][0]["role"] == "known_answer_control"
    assert formal["top10"][0]["promotion_eligible"] is False
    assert formal["top10"][1]["candidate_id"] == "G3-f9c598b70a77ea54009d8f18"
    assert formal["top10"][1]["evidence_status"] == "pass"
    assert formal["top10"][1]["role"] == "generated_candidate"
    assert formal["top10"][2]["candidate_id"] == "GF-cb4ebf3da5a74582"
    assert formal["top10"][2]["evidence_status"] == "reject"
    assert all(row["rank"] is None for row in formal["unranked_blocked_or_untested"])

    solar = board["categories"]["solar_known_answer"]
    assert solar["top10"][0]["candidate_id"] == "KNOWN-ANSWER-EINSTEIN-HILBERT"
    assert solar["top10"][0]["metrics"] == {
        "passed_control_count": 5,
        "total_control_count": 5,
    }
    assert solar["unranked_blocked_or_untested"][0]["evidence_status"] == "blocked"
    g4_solar = next(
        row
        for row in solar["unranked_blocked_or_untested"]
        if row["candidate_id"] == "G3-f9c598b70a77ea54009d8f18"
    )
    assert g4_solar["rank"] is None
    assert g4_solar["metrics"]["analytic_prediction_bundle_count"] == 1
    assert g4_solar["metrics"]["real_solar_bundle_count"] == 0
    assert g4_solar["metrics"]["source_class_uniqueness_theorem"] == "pass"
    assert g4_solar["metrics"]["noncompact_trace_tail_theorem"] == (
        "pass_conditionally"
    )
    assert g4_solar["metrics"]["verified_registration_field_count"] == 3
    assert g4_solar["metrics"]["remaining_registration_field_count"] == 6
    assert g4_solar["metrics"]["primary_record_access_count"] == 0
    assert g4_solar["blocker"] == (
        "registered_trace_tail_amplitude_decay_and_outer_transition"
    )

    assert board["categories"]["lensing_cluster"]["ranked_count"] == 0
    assert board["categories"]["galaxy_direct_observable"]["ranked_count"] == 0
    g4_galaxy = next(
        row
        for row in board["categories"]["galaxy_direct_observable"][
            "unranked_blocked_or_untested"
        ]
        if row["candidate_id"] == "G3-f9c598b70a77ea54009d8f18"
    )
    assert g4_galaxy["metrics"]["filled_registration_hash_count"] == 11
    assert g4_galaxy["metrics"]["missing_registration_hash_count"] == 7
    assert g4_galaxy["metrics"][
        "analytic_rotation_lensing_control_pass_count"
    ] == 3
    assert g4_galaxy["metrics"]["object_specific_gravity_parameter_count"] == 0
    assert g4_galaxy["blocker"] == (
        "registered_real_source_manifest_and_selected_primary_roots"
    )
    assert board["categories"]["nonlinear_energy"]["ranked_count"] == 0
    assert board["categories"]["simplicity_complexity"]["ranked_count"] == 6
    assert board["categories"]["computational_robustness"]["ranked_count"] == 6

    all_rows = [
        row
        for category in board["categories"].values()
        for row in category["full_ranked"]
        + category["unranked_blocked_or_untested"]
    ]
    assert all(row["theory_formula"]["defining_action"] for row in all_rows)
    assert all(row["theory_formula"]["scope_note"] for row in all_rows)

    gr_formula = solar["top10"][0]["theory_formula"]
    assert gr_formula["title"] == "General relativity (Einstein–Hilbert)"
    assert "√(-g)" in gr_formula["defining_action"]

    g4_formula = g4_solar["theory_formula"]
    assert g4_formula["title"] == "Conformal scalar–tensor gravity"
    assert "φ²/100" in g4_formula["defining_action"]
    assert g4_formula["action_content_sha256"] == (
        "6ddd6502d110ead90ff494a6569213ec2e61a0b046dfa86344bb1980df6abc90"
    )
    assert len(g4_formula["operator_terms"]) == 2
    dossiers = board["theory_dossiers"]
    assert len(dossiers) == 7
    assert dossiers["G3-f9c598b70a77ea54009d8f18"]["hierarchy_status_counts"] == {
        "blocked": 1,
        "calibration_only": 1,
        "proven": 8,
    }
    assert dossiers["G3-f9c598b70a77ea54009d8f18"]["overall_status"] == (
        "blocked_after_formal_pass"
    )


def test_ranked_rows_are_comparable_and_history_replay_is_idempotent() -> None:
    config = load_leaderboard_config(CONFIG)
    first = build_scientific_leaderboards(ROOT, config)
    replay = build_scientific_leaderboards(ROOT, config, first)
    assert first["leaderboard_root_sha256"] == replay["leaderboard_root_sha256"]
    assert first["history"] == replay["history"]
    assert len(replay["history"]) == 1
    for category in replay["categories"].values():
        ranked_classes = {row["data_class"] for row in category["full_ranked"]}
        assert len(ranked_classes) <= 1
        assert all(row["gate_completeness"] == "complete_for_category" for row in category["full_ranked"])
        assert all(row["evidence_status"] != "blocked" for row in category["full_ranked"])
    encoded = json.dumps(replay).lower()
    assert "truth_score" not in encoded
    assert "overall_score" not in encoded
    assert replay["data_eligibility"] == {
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_calls": False,
    }


def test_source_tamper_and_forbidden_data_class_fail_closed() -> None:
    config = load_leaderboard_config(CONFIG)
    config["sources"]["solar_known_answer"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_scientific_leaderboards(ROOT, config)

    config = load_leaderboard_config(CONFIG)
    config["data_eligibility"]["dark_matter_or_halo_inputs"] = True
    with pytest.raises(ValueError, match="eligibility"):
        build_scientific_leaderboards(ROOT, config)


def test_negative_controls_reject_score_collapse_mixing_and_control_leakage() -> None:
    board = build_scientific_leaderboards(ROOT, load_leaderboard_config(CONFIG))

    tampered = copy.deepcopy(board)
    tampered["truth_score"] = 1.0
    with pytest.raises(ValueError, match="truth-score"):
        validate_scientific_leaderboards(tampered)

    tampered = copy.deepcopy(board)
    formal = tampered["categories"]["formal_adm_dirac"]
    formal["full_ranked"][0]["data_class"] = "solar_metric"
    with pytest.raises(ValueError, match="mixing"):
        validate_scientific_leaderboards(tampered)

    tampered = copy.deepcopy(board)
    tampered["categories"]["solar_known_answer"]["full_ranked"][0][
        "promotion_eligible"
    ] = True
    with pytest.raises(ValueError, match="promotion"):
        validate_scientific_leaderboards(tampered)

    for forbidden in (
        "observational_data_opened",
        "dark_matter_or_halo_inputs",
        "redshift_distance_inputs",
    ):
        tampered = copy.deepcopy(board)
        tampered["data_eligibility"][forbidden] = True
        with pytest.raises(ValueError, match="forbidden data"):
            validate_scientific_leaderboards(tampered)

    tampered = copy.deepcopy(board)
    tampered["categories"]["simplicity_complexity"]["full_ranked"][0]["metrics"][
        "halo_target"
    ] = 1
    with pytest.raises(ValueError, match="inferred target"):
        validate_scientific_leaderboards(tampered)

    tampered = copy.deepcopy(board)
    del tampered["categories"]["formal_adm_dirac"]["full_ranked"][0][
        "theory_formula"
    ]
    with pytest.raises(ValueError, match="theory formula"):
        validate_scientific_leaderboards(tampered)

    tampered = copy.deepcopy(board)
    tampered["theory_dossiers"]["G3-f9c598b70a77ea54009d8f18"][
        "hierarchy_nodes"
    ][0]["status"] = "truthy"
    with pytest.raises(ValueError, match="hierarchy"):
        validate_scientific_leaderboards(tampered)
