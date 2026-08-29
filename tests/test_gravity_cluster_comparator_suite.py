from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_cluster_comparator_suite as suite

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stored() -> dict[str, object]:
    return json.loads((ROOT / suite.OUTPUT_PATH).read_text(encoding="utf-8"))


def test_stored_suite_rebuilds_and_preserves_development_only_boundary(
    stored: dict[str, object],
) -> None:
    suite.validate_receipt(stored, ROOT)
    assert stored["decision"] == (
        "CANDIDATE_SURVIVES_MATCHED_DEVELOPMENT_COMPARATORS_NOT_INDEPENDENT"
    )
    assert stored["sample"]["xcop_confirmation_rows_used"] is False
    assert stored["sample"]["independent_source_rows_used"] is False
    assert stored["sample"]["inferred_total_mass_target_rows"] == 0
    assert stored["counts"] == {
        "candidate_models": 1,
        "comparators": 6,
        "physical_halo_comparators": 2,
        "nonphysical_wrong_law_controls": 1,
        "generic_or_empirical_comparators": 3,
        "ablations": 4,
        "candidate_original_variants_screened": 2025,
        "new_parametric_variants_screened": 8262,
        "target_rows_opened": 0,
    }


def test_candidate_beats_frozen_halo_and_generic_comparators_on_development_holdout(
    stored: dict[str, object],
) -> None:
    candidate = stored["candidate"]
    models = stored["comparators"]
    assert candidate["holdout"]["score"] == pytest.approx(9.323376402702117)
    assert models["GR_PLUS_NFW"]["holdout"]["score"] == pytest.approx(
        28.438331523903912
    )
    assert models["GR_PLUS_EINASTO"]["holdout"]["score"] == pytest.approx(
        30.912589678016165
    )
    assert stored["ranking"]["candidate_rank"] == 1
    assert stored["claims"][
        "candidate_beats_strongest_conventional_on_development_holdout"
    ] is True
    assert stored["claims"][
        "candidate_beats_strongest_generic_on_development_holdout"
    ] is True
    assert stored["claims"]["independent_replication"] is False
    assert stored["claims"]["alternative_to_gr_established"] is False


def test_ablation_identifies_symmetric_channel_and_transition_as_important(
    stored: dict[str, object],
) -> None:
    candidate_score = stored["candidate"]["holdout"]["score"]
    ablations = stored["ablations"]
    assert ablations["REMOVE_INTERIOR_KERNEL_CHANNEL"]["holdout"]["score"] > candidate_score
    assert ablations["REMOVE_SYMMETRIC_KERNEL_CHANNEL"]["holdout"]["score"] > 40 * candidate_score
    assert ablations["REMOVE_OCCUPANCY_TRANSITION"]["holdout"]["score"] > 10 * candidate_score
    assert all(row["selection"]["refit"] is False for row in ablations.values())


def test_nfw_is_outward_cumulative_and_wrong_control_is_reversed() -> None:
    radius = np.asarray([0.1, 0.3, 1.0])
    parameters = {"c500": 3.0}
    nfw = suite._halo_fraction("GR_PLUS_NFW", radius, parameters)
    wrong = suite._halo_fraction("WRONG_REVERSED_NFW", radius, parameters)
    assert np.all(np.diff(nfw) > 0.0)
    assert np.all(np.diff(wrong) < 0.0)
    assert nfw[-1] == pytest.approx(1.0)
    assert wrong[-1] == pytest.approx(1.0)


def test_independent_or_confirmation_access_and_candidate_refit_fail_closed() -> None:
    config = suite.load_config(ROOT)
    opened = copy.deepcopy(config)
    opened["sample_contract"]["independent_source_rows_used"] = True
    with pytest.raises(suite.GravityClusterComparatorError, match="development-only"):
        suite.validate_config(opened, ROOT)

    refit = copy.deepcopy(config)
    refit["candidate"]["refit"] = True
    with pytest.raises(suite.GravityClusterComparatorError, match="candidate"):
        suite.validate_config(refit, ROOT)
