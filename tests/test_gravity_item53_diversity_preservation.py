from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler.gravity_item53_diversity_preservation import (
    _archive_diversity,
    _diversity_archive,
    _score_only_archive,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_pool() -> list[dict[str, int | float]]:
    rows = []
    ordinal = 0
    for operator in range(8):
        for pair in range(16):
            rows.append(
                {
                    "ordinal": ordinal,
                    "item52_full_data_training_loss": float(ordinal),
                    "operator_index": operator,
                    "source_pair_index": pair,
                    "operator_source_niche": operator * 16 + pair,
                    "transform_pair_index": ordinal % 64,
                    "outer_cell_index": ordinal,
                    "left_primitive_index": ordinal % 440,
                    "right_primitive_index": (ordinal + 1) % 440,
                }
            )
            ordinal += 1
    return rows


def test_item53_freezes_nondestructive_equal_size_archives() -> None:
    config = load_config(ROOT, require_bound=False)
    assert config["archives"]["slots"] == 64
    assert config["archives"]["selection_deduplicates_ordinals"] is True
    assert config["archives"]["selection_never_deletes_source_database_records"] is True
    assert config["policy"]["single_counterexample_terminal"] is False
    assert config["policy"]["finite_empirical_failure_prunes_niche"] is False
    assert config["policy"]["low_score_alone_deletes_candidate"] is False


def test_diversity_archive_covers_niches_score_only_misses() -> None:
    pool = _synthetic_pool()
    score_only = _score_only_archive(pool, 64)
    diverse = _diversity_archive(pool, 64)
    score_metrics = _archive_diversity(score_only)
    diverse_metrics = _archive_diversity(diverse)
    assert len(score_only) == len(diverse) == 64
    assert diverse_metrics["distinct_binary_operators"] == 8
    assert diverse_metrics["distinct_ordered_source_pairs"] == 16
    assert diverse_metrics["distinct_operator_source_niches"] == 64
    assert score_metrics["distinct_binary_operators"] < 8
    assert len({row["ordinal"] for row in diverse}) == 64
