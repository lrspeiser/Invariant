from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler.dataset_challenge_suite import (
    run_dataset_challenges,
    validate_dataset_challenges,
)

ROOT = Path(__file__).resolve().parents[1]


def test_intervention_noise_shift_and_unidentifiability_are_executed() -> None:
    receipt = run_dataset_challenges(ROOT)
    validate_dataset_challenges(receipt)
    by_kind = {row["kind"]: row for row in receipt["results"]}
    assert by_kind["intervention"]["evidence"]["crossed_at_fixed_covariates"] is True
    assert by_kind["noisy"]["evidence"]["positive_uncertainty_declared"] is True
    assert by_kind["shifted"]["evidence"]["domain_shift_explicit"] is True
    assert (
        by_kind["unidentifiable"]["evidence"]["required_conclusion"]
        == "UNDERDETERMINED_RETAIN_MULTIPLE_MECHANISMS"
    )
