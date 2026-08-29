from __future__ import annotations

from pathlib import Path
import json

from sigma_theory_compiler.gravity_item49_pseudorandom_exploration import (
    load_config as load_item49_config,
)
from sigma_theory_compiler.gravity_item52_failure_space import (
    _formal_outer_records,
    build_aggregate_result,
    build_query_test,
    load_config,
    query_database,
)


ROOT = Path(__file__).resolve().parents[1]


def test_item52_freezes_nonterminal_empirical_failure_language() -> None:
    config = load_config(ROOT, require_bound=False)
    policy = config["failure_policy"]
    assert policy["single_empirical_counterexample_terminal"] is False
    assert policy["empirical_counterexample_count_terminal"] is False
    assert policy["finite_empirical_region_underperformance_global_prune"] is False
    assert policy["retain_best_representative_for_every_empirical_region"] is True
    assert policy["retain_all_object_level_counterexamples"] is True


def test_formal_outer_gate_accounts_for_the_whole_bound_grammar() -> None:
    config49 = load_item49_config(ROOT)
    records = _formal_outer_records(config49)
    admitted = [row for row in records if row["passes_frozen_uniform_admissibility"]]
    excluded = [row for row in records if not row["passes_frozen_uniform_admissibility"]]
    assert len(records) == 4096
    assert len(admitted) == 336
    assert len(excluded) == 3760
    assert sum(row["full_grammar_ordinals_in_cell"] for row in records) == 6_496_138_035_200
    assert all(row["global_physical_impossibility_claimed"] is False for row in records)


def test_query_database_filters_and_preserves_best_representatives() -> None:
    database = {
        "empirical_region_records": [
            {
                "region_type": "binary_operator",
                "region_id": 0,
                "best_full_data_balanced_training_loss": 2.0,
                "tested_scope_status": "NO_SCHEDULED_MEMBER_BEATS_RETROSPECTIVE_THRESHOLD",
                "best_representative_retained": True,
            },
            {
                "region_type": "ordered_source_item_pair",
                "region_id": 0,
                "best_full_data_balanced_training_loss": 1.0,
                "tested_scope_status": "HAS_SCHEDULED_MEMBER_BEATING_RETROSPECTIVE_THRESHOLD",
                "best_representative_retained": True,
            },
        ]
    }
    rows = query_database(
        database,
        tested_scope_status="NO_SCHEDULED_MEMBER_BEATS_RETROSPECTIVE_THRESHOLD",
    )
    assert len(rows) == 1
    assert rows[0]["region_type"] == "binary_operator"
    assert rows[0]["best_representative_retained"] is True


def test_recorded_item52_database_and_queries_preserve_nonpruning_contract() -> None:
    config = load_config(ROOT)
    source = ROOT / config["paths"]["source_dir"]
    database = json.loads(
        (source / config["paths"]["database"]).read_text(encoding="utf-8")
    )
    query = json.loads(
        (source / config["paths"]["query_test"]).read_text(encoding="utf-8")
    )
    aggregate = json.loads(
        (ROOT / config["paths"]["aggregate_result"]).read_text(encoding="utf-8")
    )
    assert query == build_query_test(ROOT)
    assert aggregate == build_aggregate_result(ROOT)
    assert database["formal_scope"]["excluded_outer_parameter_cells"] == 3760
    assert database["empirical_scope"]["region_records"] == 1000
    assert database["empirical_scope"]["regions_with_scheduled_threshold_passer"] == 0
    assert all(
        row["best_representative_retained"] and not row["global_family_pruned"]
        for row in database["empirical_region_records"]
    )
    assert aggregate["claims"]["roadmap_item_52_complete"] is True
    assert aggregate["claims"]["formula_family_pruned"] is False
