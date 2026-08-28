from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_item1_effective_dimension as item1
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / item1.OUTPUT_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


@pytest.fixture(scope="module")
def prepared() -> tuple[dict[str, object], list[dict[str, object]]]:
    config = dict(item1.load_config(ROOT))
    return config, item1.prepare_objects(ROOT, config)


def test_dimension_feature_recovers_a_synthetic_mass_scaling_without_a_target() -> None:
    radius = np.geomspace(1.0, 100.0, 11)
    expected_dimension = 1.7
    log_gbar = (expected_dimension - 2.0) * np.log(radius)
    features = item1.dimension_features(radius, log_gbar, 2.0)
    assert features["profile_mass_dimension"] == pytest.approx(expected_dimension)
    assert features["local_dimension_median"] == pytest.approx(expected_dimension)
    assert features["local_dimension_iqr"] == pytest.approx(0.0, abs=1e-12)
    assert features["support_dimension"] == 2.0


def test_real_population_is_whole_object_and_uses_only_exploration_galaxies(
    prepared: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    config, objects = prepared
    assert len(objects) == 159
    assert sum(row["domain"] == "galaxy" for row in objects) == 139
    assert sum(row["domain"] == "cluster" for row in objects) == 20
    assert sum(row["point_count"] for row in objects if row["domain"] == "galaxy") == 2720
    assert sum(row["point_count"] for row in objects if row["domain"] == "cluster") == 84
    assert config["authorization"]["sparc_confirmation_evaluator_accesses_allowed"] == 0
    for row in objects:
        assert set(row["features"]) == {
            "local_dimension_iqr",
            "local_dimension_median",
            "profile_mass_dimension",
            "profile_mass_dimension_squared",
            "support_dimension",
        }
        assert "observed" not in row["features"]
        assert "sigma" not in row["features"]


def test_folds_hold_out_every_object_once_and_balance_both_populations(
    prepared: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    config, objects = prepared
    cv = config["cross_validation"]
    assignments = item1._fold_assignments(objects, salt=cv["fold_salt"], folds=cv["outer_folds"])
    assert set(assignments) == {row["key"] for row in objects}
    for fold in range(5):
        heldout = [row for row in objects if assignments[row["key"]] == fold]
        assert sum(row["domain"] == "cluster" for row in heldout) == 4
        assert sum(row["domain"] == "galaxy" for row in heldout) in {27, 28}


def test_fixed_dimension_rules_are_exact_and_posthoc_bridge_cannot_qualify(
    prepared: tuple[dict[str, object], list[dict[str, object]]],
) -> None:
    config, objects = prepared
    inverse = item1._fixed_betas(objects, "inverse_support_dimension")
    bridge = item1._fixed_betas(objects, "posthoc_support_bridge")
    for row in objects:
        key = row["key"]
        expected_inverse = 0.5 if row["domain"] == "galaxy" else 1.0 / 3.0
        expected_bridge = 0.5 if row["domain"] == "galaxy" else 2.0
        assert inverse[key] == pytest.approx(expected_inverse)
        assert bridge[key] == pytest.approx(expected_bridge)
    posthoc = next(row for row in config["fixed_rules"] if row["id"] == "posthoc_support_bridge")
    assert posthoc["origin_label"] == "post_target_new_synthesis"
    assert posthoc["qualifying_first_principles_rule"] is False


def test_receipt_rebuilds_exactly_and_records_an_inconclusive_item1_result() -> None:
    stored = _load(OUTPUT)
    rebuilt = item1.build_receipt(ROOT)
    assert rebuilt == stored
    item1.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "INCONCLUSIVE_ITEM1_EFFECTIVE_DIMENSION"
    assert stored["claims"]["roadmap_item_1_complete"] is False
    assert stored["claims"]["continuous_dimension_explains_cross_scale_response"] is False
    assert stored["specific_hypothesis_results"]["beta_equals_inverse_support_dimension"] == (
        "REJECTED_BY_CURRENT_CROSS_SCALE_DIAGNOSTIC"
    )
    assert stored["counts"]["sparc_confirmation_evaluator_accesses"] == 0
    assert stored["counts"]["direct_lensing_likelihood_evaluations"] == 0
    assert stored["counts"]["paid_model_calls"] == 0


def test_nested_predictions_are_complete_and_never_admit_the_posthoc_bridge() -> None:
    receipt = _load(OUTPUT)
    nested = receipt["nested_model"]
    assert len(receipt["per_object_diagnostics"]) == 159
    assert len(nested["fold_ledger"]) == 5
    assert all(
        fold["selected_model_id"] != "posthoc_support_bridge" for fold in nested["fold_ledger"]
    )
    assert receipt["gate_checks"]["whole_object_target_blind_outer_predictions_complete"] is True
    assert receipt["gate_checks"]["posthoc_support_bridge_excluded_from_admission"] is True


@pytest.mark.parametrize(
    "claim",
    [
        "alternative_to_gr_established",
        "binary_support_dimension_is_first_principles",
        "direct_lensing_test_completed",
        "historical_novelty_established",
        "sequential_G6_G7_G8_advanced",
        "sparc_confirmation_opened",
    ],
)
def test_resealed_overclaim_is_rejected(claim: str) -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["claims"][claim] = True
    with pytest.raises(item1.GravityItem1EffectiveDimensionError):
        item1.validate_receipt(_reseal(receipt), root=ROOT)


def test_resealed_posthoc_admission_or_nonzero_confirmation_access_is_rejected() -> None:
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["fixed_rule_results"]["posthoc_support_bridge"]["qualifying_first_principles_rule"] = (
        True
    )
    with pytest.raises(item1.GravityItem1EffectiveDimensionError):
        item1.validate_receipt(_reseal(receipt), root=ROOT)
    receipt = copy.deepcopy(_load(OUTPUT))
    receipt["counts"]["sparc_confirmation_evaluator_accesses"] = 1
    with pytest.raises(item1.GravityItem1EffectiveDimensionError):
        item1.validate_receipt(_reseal(receipt), root=ROOT)
