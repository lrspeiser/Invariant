from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.archimedes_real_data_confirmation import (
    RECEIPT_PATH,
    RealDataConfirmationError,
    build_receipt,
    validate_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_receipt(ROOT)


def test_target_blind_search_recovers_archimedes_relation(receipt: dict) -> None:
    split = receipt["initial_split"]
    assert split["discovery"]["candidate_count"] == 112
    assert split["discovery"]["winner"]["coefficients"] == [1, 1, -1, -1]
    assert split["discovery"]["winner"]["mean_absolute_residual_newton"] == "1/25"
    assert split["holdout_score"]["mean_absolute_residual_newton"] == "1/50"
    assert split["discovery"]["independent_exact_evaluators_agree"] is True


def test_same_relation_wins_every_leave_one_object_out_fold(receipt: dict) -> None:
    cross_validation = receipt["leave_one_object_out"]
    assert cross_validation["fold_count"] == 4
    assert cross_validation["same_archimedes_equivalent_winner_every_fold"] is True
    assert {
        tuple(fold["training_winner"]["coefficients"])
        for fold in cross_validation["folds"]
    } == {(1, 1, -1, -1)}


def test_claim_boundary_preserves_real_data_limitations(receipt: dict) -> None:
    assert receipt["interpretation"]["known_concept"] == "Archimedes' principle"
    assert receipt["interpretation"]["novel_theory"] is False
    assert receipt["measurement_resolution_control"]["strict_equality_confirmed"] is False
    assert receipt["measurement_resolution_control"][
        "rows_compatible_with_zero_at_display_quantization_only"
    ] == 2
    assert receipt["pairing_specificity_control"] == {
        "permutation_count": 24,
        "unique_mae_count": 1,
        "observed_mae_newton": "7/200",
        "best_permuted_mae_newton": "7/200",
        "permutations_no_worse_than_observed": 24,
        "status": "PAIRING_SPECIFICITY_UNIDENTIFIABLE",
        "reason": (
            "all 24 displaced-water permutations have the same aggregate absolute residual; "
            "the four low-variation rows cannot validate object-level pairing"
        ),
    }


def test_committed_receipt_replays_exactly(receipt: dict) -> None:
    committed = json.loads((ROOT / RECEIPT_PATH).read_text(encoding="utf-8"))
    assert committed == receipt
    assert validate_receipt(committed, ROOT)["status"] == (
        "PASS_REAL_MEASURED_BEST_RELATION_LIMITED_CONFIRMATION"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["initial_split"]["discovery"]["winner"].__setitem__(
            "coefficients", [1, 0, 0, 0]
        ),
        lambda value: value["interpretation"].__setitem__("novel_theory", True),
        lambda value: value["result"].__setitem__("strict_equality_confirmed", True),
        lambda value: value.__setitem__("content_sha256", "0" * 64),
    ],
)
def test_resealed_or_semantically_mutated_receipts_fail_closed(receipt: dict, mutation) -> None:
    changed = deepcopy(receipt)
    mutation(changed)
    with pytest.raises(RealDataConfirmationError, match="does not replay exactly"):
        validate_receipt(changed, ROOT)
