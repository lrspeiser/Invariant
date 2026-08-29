from __future__ import annotations

import numpy as np

from sigma_theory_compiler.gravity_item30_evaluation_key_correction import (
    _clean_corrected_scientific,
    _with_flexible_alias,
)


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
