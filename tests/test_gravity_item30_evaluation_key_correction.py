from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item30_evaluation_key_correction import (
    _clean_corrected_scientific,
    _with_flexible_alias,
    validate_corrected_result,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item30_flexible_alias_is_the_same_prediction() -> None:
    prediction = np.asarray([1.0, 2.0, 3.0])
    original = {"baryonic_virial": prediction + 1.0, "flexible_nuisance": prediction}
    aliased = _with_flexible_alias(original)
    assert set(original) == {"baryonic_virial", "flexible_nuisance"}
    assert aliased["flexible"] is aliased["flexible_nuisance"]


def test_item30_key_cleanup_requires_exact_duplicate_metrics() -> None:
    scientific = {
        "metrics": {
            "baseline_mse": {
                "baryonic_virial": 2.0,
                "flexible_nuisance": 1.0,
                "flexible": 1.0,
            }
        },
        "broad_slices": {
            "low_mass": {
                "flexible_mse": 1.0,
                "flexible_nuisance_mse": 1.0,
                "improvement_vs_flexible": 0.25,
                "improvement_vs_flexible_nuisance": 0.25,
            }
        },
    }
    cleaned = _clean_corrected_scientific(scientific)
    assert "flexible" not in cleaned["metrics"]["baseline_mse"]
    assert "flexible_mse" not in cleaned["broad_slices"]["low_mass"]
    assert cleaned["broad_slices"]["low_mass"]["improvement_vs_flexible"] == 0.25


def test_item30_checked_result_binds_both_narrow_corrections() -> None:
    validate_corrected_result(ROOT)


def test_item30_checked_decision_is_quality_inconclusive_and_negative() -> None:
    result = json.loads(
        (ROOT / "runs/gravity/roadmap/item-30-screening-mechanisms-v1.json").read_text(
            encoding="utf-8"
        )
    )
    scientific = result["scientific"]
    assert result["decision"] == "INCONCLUSIVE_ITEM30_QUALITY"
    assert scientific["quality"]["complete_exploration_objects"] == 562
    assert scientific["quality"]["pass"] is False
    assert scientific["universal_gravity_track"]["decision"] == "NOT_PROMOTED"
    assert scientific["phenomenon_publication_track"]["decision"] == "NOT_PROMOTED"
    assert scientific["partial_track"]["retained_slices"] == []
    assert scientific["metrics"]["improvement_vs_baryonic_virial"] > 0.23
    assert scientific["metrics"]["improvement_vs_structural"] < -0.55
    assert scientific["metrics"]["improvement_vs_flexible"] < -1.41
    assert scientific["metrics"]["selection_aware_permutation_p"] == 1.0
    assert scientific["selected_niche_counts"] == {"0": 0, "1": 0, "2": 0, "3": 5}
    assert all(
        candidate["environment_coupling"] == 0.0
        for candidate in scientific["selected_candidates"]
    )
    assert scientific["controls"]["all_injected_niches_recovered"] is True
    assert scientific["controls"]["GR_control_candidate_improves"] is False
    assert result["frozen_boundary"]["confirmation_response_values_read"] == 0
