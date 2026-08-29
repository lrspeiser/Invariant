from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item46_dimensionless_generator import (
    GravityItem46Error,
    _dimension_rows,
    _rank_fraction,
    build_dimensionless_features_from_sources,
    decode_candidate,
    generate_raw_candidates,
    load_config,
    pi_catalog,
    pi_vectors,
    primitive_basis_indices,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict:
    return json.loads(
        (ROOT / "configs/gravity_item46_dimensionless_generator_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_contract_preserves_counterexamples_and_sealed_data() -> None:
    config = load_config(ROOT)
    policy = config["discovery_policy"]
    assert policy["single_empirical_counterexample_is_not_a_formula_or_family_veto"]
    assert policy["counterexample_count_alone_is_never_decisive"]
    assert not policy["finite_empirical_sample_may_prune_family"]
    assert config["gates"]["confirmation_response_rows"] == 0
    assert config["candidate_generator"]["post_evaluation_cells"] == 0


def test_dependency_mutation_is_rejected() -> None:
    config = _raw_config()
    config["scientific_dependencies"] = {
        "docs/GRAVITY_COUNTEREXAMPLE_AND_DATA_QUALITY_POLICY_V2.md": "0" * 64
    }
    with pytest.raises(GravityItem46Error, match="scientific dependency changed"):
        validate_config(ROOT, config)


def test_exact_dimension_matrix_rank_nullity_and_bounded_span() -> None:
    config = load_config(ROOT)
    rows = _dimension_rows(config)
    vectors = pi_vectors(config)
    assert _rank_fraction(rows) == 3
    assert len(rows[0]) - _rank_fraction(rows) == 6
    assert len(vectors) == 184
    assert _rank_fraction(vectors) == 6
    assert all(
        sum(row[index] * vector[index] for index in range(len(vector))) == 0
        for row in rows
        for vector in vectors
    )


def test_minimum_complexity_basis_is_response_independent_and_named() -> None:
    config = load_config(ROOT)
    catalog = pi_catalog(config)
    indices = primitive_basis_indices(config)
    assert indices == [0, 1, 2, 5, 7, 47]
    assert [catalog[index]["expression"] for index in indices] == [
        "q",
        "H*t",
        "R/Rb",
        "a0/(c*H)",
        "Rb/(c*t)",
        "Rb*c**2/(M*G)",
    ]
    assert all(catalog[index]["primitive_basis_member"] for index in indices)


def test_every_dimension_breaking_negative_control_is_rejected() -> None:
    config = load_config(ROOT)
    catalog = pi_catalog(config)
    assert len(catalog) == 184
    assert all(item["negative_control"]["rejected"] for item in catalog)
    assert all(any(item["negative_control"]["dimension_residual"]) for item in catalog)
    assert all(item["dimension_residual"] == [0, 0, 0] for item in catalog)


def test_candidate_capacity_and_recipe_mapping_are_exact() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 753664
    assert np.array_equal(np.unique(raw["recipe"]), np.arange(184))
    assert all(np.sum(raw["recipe"] == recipe) == 4096 for recipe in range(184))
    basis = decode_candidate(0, config)
    combination = decode_candidate(183 * 4096, config)
    assert basis["creativity_label"] == "known_buckingham_pi_basis_or_physical_rewrite"
    assert combination["creativity_label"].endswith("historical_novelty_unestablished")


def test_feature_generation_is_strictly_response_blind() -> None:
    config = load_config(ROOT)
    item44 = json.loads(
        (ROOT / "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-scale-features.json").read_text(encoding="utf-8")
    )
    item45 = json.loads(
        (ROOT / "runs/gravity/roadmap/item-45-universal-interactions-v1-source/interaction-features.json").read_text(encoding="utf-8")
    )
    mutated = copy.deepcopy(item44)
    for index, row in enumerate(mutated["records"]):
        row["log10_observed_quantity"] = 10000.0 + index
        row["log10_uncertainty"] = 1e-12
    before = build_dimensionless_features_from_sources(item44, item45, config)
    after = build_dimensionless_features_from_sources(mutated, item45, config)
    assert before == after
    assert before["response_values_used"] == 0
    assert before["response_fields_read_by_feature_builder"] == []
    assert before["counts"]["pi_recipes"] == 184
    coordinates = np.asarray([row["pi_coordinates"] for row in before["records"]])
    assert coordinates.shape == (112, 184)
    assert np.all(np.isfinite(coordinates))
    assert np.all((coordinates > 0.0) & (coordinates <= 1.0))
    assert all("log10_observed_quantity" not in row for row in before["records"])
