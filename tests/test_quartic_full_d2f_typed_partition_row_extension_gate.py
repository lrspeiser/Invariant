from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_full_d2f_typed_partition_row_extension_gate import (
    CONFIG_PATH,
    OUTPUT_PATH,
    _csha,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result():
    return build_gate(ROOT / CONFIG_PATH)


def test_artifact(result):
    assert result == json.loads((ROOT / OUTPUT_PATH).read_text())


def test_partition(result):
    p = result["typed_full_domain_partition"]
    assert sum(x["per_candidate"] for x in p) == 257499
    assert [x["per_candidate"] for x in p] == [22, 220, 5082, 31702, 31702, 188771]


def test_counts(result):
    c = result["gate_counts"]
    assert c["newly_registered_per_candidate"] == 220
    assert c["registered_per_candidate"] == 242
    assert c["remaining_per_candidate"] == 257257
    assert c["newly_registered_all_candidates"] == 2640


def test_manifests(result):
    ids = set()
    for m in result["candidate_manifests"]:
        assert len(m["new_row_extension_records"]) == 220
        for r in m["new_row_extension_records"]:
            assert r["source_row"] in range(10)
            ids.add(r["record_id"])
    assert len(ids) == 2640


def test_claims_closed(result):
    s = result["claim_seals"]
    assert s["typed_full_domain_partition_complete"] is True
    assert s["maximal_same_direction_row_extension_registered"] is True
    for k in ("complete_D2F", "full_high_atom_identity", "physical_no_go", "candidate_rejected"):
        assert s[k] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("gate_counts", "registered_per_candidate", 257499),
        ("claim_seals", "complete_D2F", True),
        ("claim_seals", "physical_no_go", True),
    ],
)
def test_tamper(result, section, key, value):
    x = copy.deepcopy(result)
    x[section][key] = value
    x["content_sha256"] = _csha(x)
    assert x != build_gate(ROOT / CONFIG_PATH)


def test_unknown_key(result):
    x = copy.deepcopy(result)
    x["overclaim"] = True
    x["content_sha256"] = _csha(x)
    assert x != build_gate(ROOT / CONFIG_PATH)
