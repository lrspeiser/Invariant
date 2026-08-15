import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_q_metric_leaf_candidate0_batch_gate import (
    OUTPUT_PATH,
    QMetricLeafBatchError,
    _content_sha,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _load():
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def test_replay():
    v = _load()
    assert build_campaign(root=ROOT) == v
    validate_campaign(v, root=ROOT)


def test_nine_units_and_shared_dag():
    v = _load()
    assert [u["unit_index"] for u in v["units"]] == list(range(1, 10))
    assert len({u["leaf_arithmetic_DAG_sha256"] for u in v["units"]}) == 1
    assert sum(u["leaf_derivative_roots"] for u in v["units"]) == 23760


def test_counts_and_seals():
    c = _load()["gate_counts"]
    assert (
        c["cumulative_completed_units"],
        c["remaining_units"],
        c["cumulative_leaf_roots"],
        c["remaining_leaf_roots"],
    ) == (10, 110, 26400, 290400)
    assert c["nonzero_leaf_roots"] + c["exact_zero_leaf_roots"] == 23760
    assert c["unique_registered_coordinate_columns_after"] == 143
    assert c["registered_D2_entries_per_candidate_after"] == 5324


def test_full_claims_false():
    s = _load()["claim_seals"]
    assert (
        not s["all_120_units_complete"]
        and not s["complete_q_metric_leaf_family_registered"]
        and not s["all_153_unique_coordinate_leaf_authorities_registered"]
        and not s["D2_entry_count_advanced"]
    )


def test_tamper_rejected():
    v = copy.deepcopy(_load())
    v["gate_counts"]["remaining_units"] = 0
    v["content_sha256"] = _content_sha(v)
    with pytest.raises(QMetricLeafBatchError, match="checked result changed"):
        validate_campaign(v, root=ROOT)
