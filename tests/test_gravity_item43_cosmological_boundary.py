import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item43_cosmological_boundary import (
    admissible_candidates,
    age_ratio,
    boundary_bases,
    build_candidate_manifest,
    build_exposure_manifest,
    build_sample_manifest,
    decode_candidate,
    expansion_ratio,
    generate_raw_candidates,
    load_config,
    sersic_n4_fraction,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_counterexample_and_response_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 43
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["discovery_policy"]["counterexample_count_alone_is_never_decisive"]
    assert config["candidate_generator"]["post_response_cells"] == 0
    assert config["schema_audit_exposure"]["rows_with_response_seen"] == 5


def test_raw_grid_has_four_equal_cosmological_niches() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 262_144
    assert [int(np.sum(raw["lane"] == lane)) for lane in range(4)] == [65_536] * 4
    assert decode_candidate(0, config)["lane"] == "expansion_rate_running"
    assert decode_candidate(196_608, config)["lane"] == "finite_horizon_fraction"


def test_boundary_coordinates_have_present_epoch_and_monotone_limits() -> None:
    config = load_config(ROOT)
    z = np.asarray([0.0, 0.5, 1.0])
    radius = np.asarray([1.0, 10.0, 100.0])
    assert np.all(np.diff(expansion_ratio(z, config)) > 0.0)
    assert np.all(np.diff(age_ratio(z, config)) < 0.0)
    bases = boundary_bases(z, radius, config)
    assert bases.shape == (4, 3)
    assert np.allclose(bases[:3, 0], 1.0)
    assert np.all(bases > 0.0)


def test_de_vaucouleurs_aperture_fraction_is_half_at_re() -> None:
    config = load_config(ROOT)
    b4 = float(config["constants"]["sersic_n4_b"])
    values = sersic_n4_fraction(np.asarray([0.0, 1.0, 10.0]), b4)
    assert values[0] == 0.0
    assert abs(values[1] - 0.5) < 1e-8
    assert 0.9 < values[2] < 1.0


def test_admission_and_freeze_manifests_are_response_safe() -> None:
    config = load_config(ROOT)
    admitted, audit = admissible_candidates(config)
    assert audit["raw_candidates"] == 262_144
    assert audit["admitted_candidates"] == 186_989
    assert len(admitted["candidate_id"]) == audit["admitted_candidates"]
    candidate = build_candidate_manifest(ROOT)
    exposure = build_exposure_manifest(ROOT)
    assert candidate["response_accessed_during_generation"] is False
    assert candidate["confirmation_accessed"] is False
    assert exposure["counts"]["schema_rows_with_response_seen"] == 5
    assert exposure["counts"]["remaining_response_rows_read"] == 0
    assert exposure["counts"]["confirmation_rows_read"] == 0


def test_fresh_predictor_source_and_sample_keep_responses_sealed() -> None:
    config = load_config(ROOT)
    source_path = (
        ROOT / config["paths"]["source_dir"] / config["paths"]["predictor_source"]
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert source["counts"]["grade_a_before_schema_exclusion"] == 40
    assert source["counts"]["eligible_after_schema_exclusion"] == 35
    assert source["counts"]["response_values_read"] == 0
    assert not any("log10_einstein_mass_msun" in row for row in source["records"])
    sample = build_sample_manifest(ROOT)
    assert sample["counts"]["exploration_lenses"] == 28
    assert sample["counts"]["confirmation_lenses"] == 7
    assert sample["counts"]["response_rows_read"] == 0
    assert sample["counts"]["confirmation_rows_read"] == 0
    assert all(
        row["target"]
        not in config["schema_audit_exposure"]["excluded_from_every_item43_role"]
        for row in sample["objects"]
    )


def test_recorded_result_preserves_negative_transfer_and_formula_family() -> None:
    config = load_config(ROOT)
    source_dir = ROOT / config["paths"]["source_dir"]
    responses = json.loads(
        (source_dir / config["paths"]["response_source"]).read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (source_dir / config["paths"]["evaluation_result"]).read_text(encoding="utf-8")
    )
    transfer = json.loads(
        (source_dir / config["paths"]["clash_transfer_result"]).read_text(
            encoding="utf-8"
        )
    )
    aggregate = json.loads(
        (ROOT / config["paths"]["aggregate_result"]).read_text(encoding="utf-8")
    )
    assert responses["counts"]["exploration_response_rows"] == 28
    assert responses["counts"]["confirmation_response_rows"] == 0
    assert evaluation["selected_candidate"]["lane"] == "finite_horizon_fraction"
    assert evaluation["scores"]["cosmological_boundary"]["loss"] < evaluation[
        "scores"
    ]["matched_no_boundary"]["loss"]
    assert evaluation["scores"]["cosmological_boundary"]["loss"] > evaluation[
        "scores"
    ]["ordinary_ridge"]["loss"]
    assert len(evaluation["robustness"]["systematic_candidate_predictions"]) == 6
    assert evaluation["counterexample_policy_assessment"]["formula_family_pruned"] is False
    assert len(transfer["raw_counterexample_clusters"]) == 20
    assert transfer["aggregate_improvement_percent"] < 0.0
    assert aggregate["claims"]["formula_family_pruned"] is False
    assert aggregate["claims"]["single_counterexample_used_as_veto"] is False
