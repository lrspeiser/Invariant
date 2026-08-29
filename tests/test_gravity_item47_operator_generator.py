from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item47_operator_generator import (
    GravityItem47Error,
    _operator_raw_values,
    _shape_by_object,
    build_operator_features_from_sources,
    decode_candidate,
    generate_raw_candidates,
    load_config,
    operator_catalog,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict:
    return json.loads(
        (ROOT / "configs/gravity_item47_operator_generator_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _synthetic_raw(
    config: dict, *, mass: np.ndarray | None = None, axis_ratio: float = 0.8, age: float = 0.6
) -> np.ndarray:
    radius = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    if mass is None:
        mass = np.asarray([1.0, 2.2, 4.8, 8.0, 10.0])
    return _operator_raw_values(
        radius,
        mass,
        radius=4.0,
        baryonic_size=3.0,
        u=0.4,
        axis_ratio=axis_ratio,
        age_over_t0=age,
        config=config,
    )


def test_frozen_contract_preserves_counterexample_and_history_boundaries() -> None:
    config = load_config(ROOT)
    policy = config["discovery_policy"]
    assert policy["single_empirical_counterexample_is_not_a_formula_or_family_veto"]
    assert policy["counterexample_count_alone_is_never_decisive"]
    assert not policy["finite_empirical_sample_may_prune_family"]
    assert not config["scope"]["measured_baryonic_history_available"]
    assert "constant-state closure" in config["profile_contract"]["history_claim_boundary"]
    assert config["gates"]["confirmation_response_rows"] == 0


def test_dependency_mutation_is_rejected() -> None:
    config = _raw_config()
    config["scientific_dependencies"] = {
        "docs/GRAVITY_COUNTEREXAMPLE_AND_DATA_QUALITY_POLICY_V2.md": "0" * 64
    }
    with pytest.raises(GravityItem47Error, match="scientific dependency changed"):
        validate_config(ROOT, config)


def test_operator_catalog_and_candidate_capacity_are_equal_by_class() -> None:
    config = load_config(ROOT)
    catalog = operator_catalog(config)
    raw = generate_raw_candidates(config)
    assert len(catalog) == 96
    assert len(raw["candidate_id"]) == 393216
    assert np.array_equal(np.unique(raw["recipe"]), np.arange(96))
    assert all(np.sum(raw["recipe"] // 16 == class_id) == 65536 for class_id in range(6))
    assert all(not recipe["historical_novelty_claimed"] for recipe in catalog)
    assert decode_candidate(0, config)["operator_class"] == "local"
    assert decode_candidate(95 * 4096, config)["operator_class"] == "causal_history"


def test_operator_evaluator_produces_all_six_typed_blocks() -> None:
    config = load_config(ROOT)
    raw = _synthetic_raw(config)
    assert raw.shape == (96,)
    assert np.all(np.isfinite(raw))
    assert all(np.ptp(raw[start : start + 16]) > 0.0 for start in range(0, 96, 16))


def test_interior_and_exterior_support_rules_are_executable() -> None:
    config = load_config(ROOT)
    baseline = _synthetic_raw(config)
    outer_changed = _synthetic_raw(
        config, mass=np.asarray([1.0, 2.2, 4.8, 15.0, 30.0])
    )
    assert np.allclose(baseline[32:48], outer_changed[32:48], atol=1e-12, rtol=0.0)
    assert not np.allclose(baseline[48:64], outer_changed[48:64])
    inner_changed = _synthetic_raw(
        config, mass=np.asarray([0.2, 1.1, 4.8, 8.0, 10.0])
    )
    assert np.allclose(baseline[48:64], inner_changed[48:64], atol=1e-12, rtol=0.0)
    assert not np.allclose(baseline[32:48], inner_changed[32:48])


def test_tensor_scalar_is_rotation_safe_and_history_memory_is_monotone() -> None:
    config = load_config(ROOT)
    spherical = _synthetic_raw(config, axis_ratio=1.0)
    assert np.allclose(spherical[64:80], 0.0, atol=1e-15, rtol=0.0)
    young = _synthetic_raw(config, age=0.2)
    old = _synthetic_raw(config, age=0.8)
    assert np.all(old[80:84] > young[80:84])
    assert np.all(np.isfinite(old[80:96]))


def test_operator_feature_generation_is_response_blind() -> None:
    config = load_config(ROOT)
    item44 = json.loads(
        (ROOT / "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-scale-features.json").read_text(encoding="utf-8")
    )
    item46 = json.loads(
        (ROOT / "runs/gravity/roadmap/item-46-dimensionless-generator-v1-source/dimensionless-features.json").read_text(encoding="utf-8")
    )
    shapes = _shape_by_object(
        ROOT, {"object": np.asarray([row["object"] for row in item44["records"]])}
    )
    mutated = copy.deepcopy(item44)
    for index, row in enumerate(mutated["records"]):
        row["log10_observed_quantity"] = 20000.0 + index
        row["log10_uncertainty"] = 1e-15
    before = build_operator_features_from_sources(item44, item46, shapes, config)
    after = build_operator_features_from_sources(mutated, item46, shapes, config)
    assert before == after
    assert before["response_values_used"] == 0
    assert before["response_fields_read_by_feature_builder"] == []
    assert before["counts"]["operator_recipes"] == 96
    coordinates = np.asarray([row["operator_coordinates"] for row in before["records"]])
    assert coordinates.shape == (112, 96)
    assert np.all((coordinates > 0.0) & (coordinates < 1.0))
    assert all("log10_observed_quantity" not in row for row in before["records"])


def test_recorded_result_retains_nonlocal_clue_without_promotion() -> None:
    aggregate_path = ROOT / "runs/gravity/roadmap/item-47-operator-generator-v1.json"
    evaluation_path = ROOT / "runs/gravity/roadmap/item-47-operator-generator-v1-source/joint-evaluation-result.json"
    if not aggregate_path.exists() or not evaluation_path.exists():
        pytest.skip("Item 47 evaluation artifacts are not installed")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert aggregate["decision"] == "NONPROMOTED_ITEM47_OPERATOR_RESULT_RETAINED"
    assert aggregate["gates"]["beats_item45_s4tm"]
    assert not aggregate["gates"]["beats_item45_clash"]
    assert not aggregate["claims"]["formula_family_pruned"]
    assert not aggregate["claims"]["single_counterexample_used_as_veto"]
    assert not aggregate["claims"]["measured_history_tested"]
    assert evaluation["selected_candidate"]["operator_class"] == "exterior_nonlocal"
    assert evaluation["selected_candidate"]["source"] == "kernel_outer_potential"
    assert evaluation["counterexample_policy_assessment"]["raw_counterexample_count"] == 23
    assert evaluation["counterexample_policy_assessment"]["single_object_sensitive"]
    assert all(
        row["selected_operator"]["source"] == "kernel_outer_potential"
        and row["selected_operator"]["scale"] == 0.6
        for row in evaluation["fold_ledger"]
    )
