from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.quartic_principal_second_jet_covariant_projection_gate import (
    OUTPUT_PATH,
    PrincipalSecondJetProjectionError,
    _content_sha,
    _flat_substitution,
    _generic_einstein_tangent,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_variable_sylvester_campaign import (
    _linearized_einstein_upper,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value["content_sha256"] = _content_sha(value)


def test_checked_artifact_is_exact_live_replay() -> None:
    checked = _load()
    assert build_campaign(root=ROOT) == checked
    validate_campaign(checked, root=ROOT)
    assert checked["content_sha256"] == _content_sha(checked)


def test_all_99_principal_directions_are_registered_without_D2_overclaim() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    assert counts["principal_second_jet_projection_directions"] == 99
    assert counts["active_principal_projection_directions"] == 75
    assert counts["zero_principal_projection_directions"] == 24
    assert counts["covariant_projection_scalar_coefficients"] == 270
    assert counts["new_unique_projection_directions_registered_per_candidate"] == 79
    assert counts["remaining_lower_jet_projection_directions"] == 54
    assert counts["D2_entries_registered_per_candidate"] == 5324
    assert counts["D2_entries_remaining_per_candidate"] == 252175
    assert checked["claim_seals"]["D2_entry_count_advanced"] is False
    assert checked["claim_seals"]["complete_D2F"] is False


def test_arbitrary_metric_formula_replays_all_flat_metric_directions() -> None:
    registry = _load()["principal_projection_registry"]
    metric = [row for row in registry if row["field_index"] < 10]
    assert len(metric) == 90
    for row in metric:
        pair = tuple(row["derivative_pair"])
        field = row["field_index"]
        generic = _generic_einstein_tangent(pair, field)
        actual = {
            name: sp.factor(value.subs(_flat_substitution()))
            for name, value in generic.items()
            if sp.factor(value.subs(_flat_substitution())) != 0
        }
        assert actual == _linearized_einstein_upper(pair, field)


def test_formal_slot_aliases_are_reconciled_candidate_by_candidate() -> None:
    checked = _load()
    assert len(checked["candidate_manifests"]) == 12
    for candidate in checked["candidate_manifests"]:
        aliases = candidate["alias_reconciliation_records"]
        assert len(aliases) == 2
        assert all(row["same_coordinate_vector_same_covariant_projection"] for row in aliases)
        assert all(len(row["formal_coordinate_ordinals"]) == 2 for row in aliases)


def test_trace_omission_negative_is_exact_and_nonzero() -> None:
    negative = _load()["exact_controls"]["drop_trace_term_changes_flat_projection"]
    assert negative == {
        "exact_residual": "sqrt(2)/2",
        "rejected": True,
        "witness_atom": "s01[1]",
        "witness_output": "G_00",
    }


def test_resealed_projection_or_claim_tamper_fails_closed() -> None:
    for mutate in (
        lambda value: value["principal_projection_registry"][10].update(
            {"covariant_jet_entries": {}}
        ),
        lambda value: value["claim_seals"].update({"complete_D2F": True}),
        lambda value: value["gate_counts"].update({"D2_entries_registered_per_candidate": 257499}),
    ):
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        _reseal(corrupted)
        with pytest.raises(PrincipalSecondJetProjectionError):
            validate_campaign(corrupted, root=ROOT)


def test_unknown_key_and_raw_file_tamper_fail_closed() -> None:
    corrupted = copy.deepcopy(_load())
    corrupted["unknown"] = False
    _reseal(corrupted)
    with pytest.raises(PrincipalSecondJetProjectionError):
        validate_campaign(corrupted, root=ROOT)
    raw = (ROOT / OUTPUT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() != _load()["content_sha256"]
