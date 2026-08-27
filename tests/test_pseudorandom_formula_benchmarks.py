from __future__ import annotations

import copy
from pathlib import Path

import pytest

from sigma_theory_compiler.pseudorandom_formula_benchmarks import (
    TRILLION_GRAMMAR_SIZE,
    PseudorandomFormulaBenchmarkError,
    build_benchmark_receipt,
    validate_benchmark_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def receipt() -> dict:
    return build_benchmark_receipt(ROOT)


def test_exact_polynomial_is_recovered_from_every_candidate_in_pseudorandom_order(
    receipt: dict,
) -> None:
    problem = receipt["problems"][0]

    assert problem["grammar"]["candidate_count"] == 14_641
    assert problem["search"]["candidates_tested"] == 14_641
    assert problem["winner"]["coefficients"] == [5, -2, 3, 0]
    assert problem["winner"]["exact_on_all_rows"] is True
    assert problem["classification"] == "known_planted_rediscovery_not_novel"


def test_archimedes_relation_is_recovered_from_real_measurements(receipt: dict) -> None:
    problem = receipt["problems"][1]

    assert problem["grammar"]["candidate_count"] == 112
    assert problem["winner"]["coefficients"] == [1, 1, -1, -1]
    assert problem["winner"]["training_mean_absolute_residual_newton"] == "1/25"
    assert problem["winner"]["holdout_mean_absolute_residual_newton"] == "1/50"
    assert problem["classification"] == "known_real_data_rediscovery_not_novel"


def test_kepler_equivalent_relation_is_recovered_from_anonymous_real_catalog_columns(
    receipt: dict,
) -> None:
    problem = receipt["problems"][2]

    assert problem["grammar"]["candidate_count"] == 6152
    assert problem["winner"]["exponents"] == [2, -3, 1]
    assert problem["source"]["training_rows"] == 2020
    assert problem["source"]["holdout_rows"] == 511
    assert float(problem["winner"]["holdout_within_1sigma_fraction"]) > 0.9
    assert float(problem["winner"]["holdout_within_2sigma_fraction"]) > 0.95
    assert problem["classification"] == "known_catalog_relation_rediscovery_not_novel"


def test_trillion_scale_claim_stays_inside_the_sampled_boundary(receipt: dict) -> None:
    probe = receipt["scale_probe"]

    ordinal = probe["ordinal_prefix_receipt"]
    assert ordinal["permutation"]["size"] == TRILLION_GRAMMAR_SIZE
    assert ordinal["sample"]["count"] == 10_000
    assert ordinal["sample"]["all_unique"] is True
    assert probe["gpu_chunk_schedule"]["descriptor"]["chunk_count"] == 212_774
    assert probe["gpu_chunk_schedule"]["sampled_chunks_sha256"]
    assert probe["gpu_chunk_schedule"]["sampled_chunk_ids_unique"] is True
    assert receipt["claims"]["trillion_formula_campaign_executed"] is False
    assert receipt["counts"]["formula_candidates_actually_tested"] == 20_905


def test_receipt_replays_and_tampering_fails_closed(receipt: dict) -> None:
    validate_benchmark_receipt(receipt, ROOT)

    tampered = copy.deepcopy(receipt)
    tampered["claims"]["new_formula_discovered"] = True
    with pytest.raises(PseudorandomFormulaBenchmarkError, match="seal"):
        validate_benchmark_receipt(tampered, ROOT)
