from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item45_universal_interactions import (
    AXES,
    GravityItem45Error,
    _devaucouleurs_enclosed_slope,
    build_interaction_features_from_item44,
    decode_candidate,
    generate_raw_candidates,
    interaction_bank,
    load_config,
    main_effect_bank,
    primitive_coordinates,
    recipe_catalog,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict:
    return json.loads(
        (ROOT / "configs/gravity_item45_universal_interactions_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_contract_and_counterexample_policy() -> None:
    config = load_config(ROOT)
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["discovery_policy"]["counterexample_count_alone_is_never_decisive"]
    assert not config["discovery_policy"]["finite_empirical_sample_may_prune_family"]
    assert config["gates"]["confirmation_response_rows"] == 0
    assert config["candidate_generator"]["post_evaluation_cells"] == 0


def test_dependency_mutation_is_rejected(tmp_path: Path) -> None:
    config = _raw_config()
    config["scientific_dependencies"] = {
        "docs/GRAVITY_COUNTEREXAMPLE_AND_DATA_QUALITY_POLICY_V2.md": "0" * 64
    }
    with pytest.raises(GravityItem45Error, match="scientific dependency changed"):
        validate_config(ROOT, config)


def test_generator_capacity_and_equal_niches() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 262144
    assert np.array_equal(np.unique(raw["recipe"]), np.arange(64))
    assert all(np.sum(raw["recipe"] // 16 == niche) == 65536 for niche in range(4))
    assert len(recipe_catalog(config)) == 64


def test_candidate_decode_preserves_creativity_provenance() -> None:
    config = load_config(ROOT)
    known = decode_candidate(0, config)
    synthesis = decode_candidate(32 * 4096, config)
    main = decode_candidate(63 * 4096, config, main_effect=True)
    assert "known_pairwise" in known["creativity_label"]
    assert "potentially_new" in synthesis["creativity_label"]
    assert synthesis["creativity_label"].endswith("not_historical_novelty")
    assert main["niche"] == "matched_unary_main_effect_control"


def test_de_vaucouleurs_slope_is_positive_and_decreases_outward() -> None:
    slope = _devaucouleurs_enclosed_slope(np.asarray([0.1, 1.0, 10.0]))
    assert np.all(slope > 0.0)
    assert np.all(np.diff(slope) < 0.0)


def test_primitive_and_recipe_banks_are_finite_and_bounded() -> None:
    config = load_config(ROOT)
    arrays = {
        "population": np.asarray(["S4TM", "CLASH", "CLASH"]),
        "object": np.asarray(["lens", "cluster", "cluster"]),
        "radius": np.asarray([5.0, 100.0, 200.0]),
        "size": np.asarray([4.0, 150.0, 150.0]),
        "redshift": np.asarray([0.2, 0.4, 0.4]),
        "u": np.asarray([2.0, 0.1, 0.2]),
        "horizon": np.asarray([4.0e6, 3.5e6, 3.5e6]),
        "schwarzschild": np.asarray([1e-8, 1e-3, 4e-3]),
    }
    raw, normalized = primitive_coordinates(arrays, config)
    interactions, catalog = interaction_bank(normalized, config)
    main = main_effect_bank(normalized, config)
    assert raw.shape == (3, len(AXES))
    assert normalized.shape == (3, len(AXES))
    assert interactions.shape == (3, 64)
    assert main.shape == (3, 64)
    assert len(catalog) == 64
    assert np.all(np.isfinite(interactions)) and np.all((interactions > 0) & (interactions < 1))
    assert np.all(np.isfinite(main)) and np.all((main > 0) & (main < 1))


def test_feature_synthesis_is_response_blind() -> None:
    config = load_config(ROOT)
    source = json.loads((ROOT / "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-scale-features.json").read_text(encoding="utf-8"))
    mutated = copy.deepcopy(source)
    for index, row in enumerate(mutated["records"]):
        row["log10_observed_quantity"] = 1000.0 + index
        row["log10_uncertainty"] = 0.000001
    before = build_interaction_features_from_item44(source, config)
    after = build_interaction_features_from_item44(mutated, config)
    assert before == after
    assert before["response_values_used"] == 0
    assert before["response_fields_read_by_feature_builder"] == []
    assert all("log10_observed_quantity" not in row for row in before["records"])


def test_every_required_axis_participates_in_interactions() -> None:
    config = load_config(ROOT)
    catalog = recipe_catalog(config)
    participating = {axis for recipe in catalog for axis in recipe["operands"]}
    assert participating == set(AXES)
    assert all(len(recipe["operands"]) >= 2 for recipe in catalog)


def test_recorded_result_preserves_nonpromotion_and_mismatches() -> None:
    aggregate_path = ROOT / "runs/gravity/roadmap/item-45-universal-interactions-v1.json"
    evaluation_path = ROOT / "runs/gravity/roadmap/item-45-universal-interactions-v1-source/joint-evaluation-result.json"
    if not aggregate_path.exists() or not evaluation_path.exists():
        pytest.skip("Item 45 evaluation artifacts are not installed")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert aggregate["decision"] == "NONPROMOTED_ITEM45_UNIVERSAL_INTERACTION_RESULT_RETAINED"
    assert not aggregate["gates"]["beats_item44_s4tm"]
    assert not aggregate["gates"]["paired_p_passes"]
    assert not aggregate["claims"]["formula_family_pruned"]
    assert not aggregate["claims"]["single_counterexample_used_as_veto"]
    assert evaluation["counterexample_policy_assessment"]["raw_counterexample_count"] == 19
    assert all(
        row["selected_interaction"]["interaction_expression"]
        == "geometry*tanh(2*density)"
        for row in evaluation["fold_ledger"]
    )
