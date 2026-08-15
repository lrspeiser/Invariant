from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_lower_coordinate_covariant_projection_gate import (
    OUTPUT_PATH,
    LowerCoordinateProjectionError,
    _content_sha,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value["content_sha256"] = _content_sha(value)


def test_checked_campaign_is_exact_live_replay() -> None:
    checked = _load()
    assert build_campaign(root=ROOT) == checked
    validate_campaign(checked, root=ROOT)
    assert checked["content_sha256"] == _content_sha(checked)


def test_all_54_lower_directions_have_exact_indexed_certificates() -> None:
    checked = _load()
    registry = checked["lower_projection_registry"]
    assert len(registry) == 54
    assert [row["coordinate_column"] for row in registry] == list(range(54))
    assert sum(row["family"] == "q_metric" for row in registry) == 10
    assert sum(row["family"] == "p_metric" for row in registry) == 40
    assert sum(row["family"] == "p_scalar_gradient" for row in registry) == 4
    program_sha = checked["indexed_formula_program"]["content_sha256"]
    assert all(row["indexed_formula_program_sha256"] == program_sha for row in registry)
    assert all(row["exact_projection_registered"] is True for row in registry)


def test_sqrt2_metric_encoding_and_scalar_gradient_seeds_are_explicit() -> None:
    by_atom = {row["coordinate_atom"]: row for row in _load()["lower_projection_registry"]}
    assert by_atom["q[1]"]["tangent_seed"]["dg_value"] == "sqrt(2)/2"
    assert by_atom["p1[1]"]["tangent_seed"]["dP_value"] == "sqrt(2)/2"
    assert by_atom["p3[10]"]["tangent_seed"] == {
        "dg": "0",
        "dP": "0",
        "dv_component": 3,
        "dv_value": "1",
    }


def test_formula_program_contains_full_lower_chain_rule() -> None:
    program = _load()["indexed_formula_program"]
    formulas = program["formulas_in_dependency_order"]
    assert len(formulas) == 17
    assert formulas[0] == "du^ab=-u^ac*dg_cd*u^db"
    assert any(formula.startswith("dGamma") for formula in formulas)
    assert any(formula.startswith("dR^r_smn") for formula in formulas)
    assert formulas[-1].startswith("dG^mn=")
    assert program["output_basis"][:4] == ["v_0", "v_1", "v_2", "v_3"]
    assert len(program["output_basis"]) == 24


def test_alias_union_is_candidate_bound_and_coordinate_aware() -> None:
    checked = _load()
    alias = checked["alias_reconciliation"]
    assert alias == {
        "coordinate_columns_are_authoritative": True,
        "lower_formal_slot_alias_groups": 0,
        "inherited_principal_alias_groups_per_candidate": 2,
        "inherited_principal_alias_columns": [97, 130],
        "unique_coordinate_directions_after_union": 153,
    }
    assert len(checked["candidate_manifests"]) == 12
    for candidate in checked["candidate_manifests"]:
        assert candidate["lower_coordinate_projection_certificates"] == 54
        assert candidate["total_unique_coordinate_directions_projected"] == 153
        assert candidate["inherited_principal_alias_columns"] == [97, 130]


def test_D1_DAG_audit_preserves_5324_of_257499() -> None:
    checked = _load()
    audit = checked["D1_DAG_audit"]
    counts = checked["gate_counts"]
    assert audit["full_D1_entries_per_candidate"] == 1683
    assert audit["lower_D1_entries_per_candidate"] == 594
    assert audit["missing_candidate_bound_leaf_derivatives"] == 31680
    assert audit["registered_leaf_derivative_roots"] == 0
    assert audit["new_ordered_D2_roots_legitimately_yielded"] == 0
    assert counts["D2_entries_registered_per_candidate_before"] == 5324
    assert counts["new_D2_entries_registered_per_candidate"] == 0
    assert counts["D2_entries_registered_per_candidate_after"] == 5324
    assert counts["full_D2_entries_per_candidate"] == 257499
    assert checked["claim_seals"]["D2_entry_count_advanced"] is False


def test_exact_connection_and_normalization_negatives_are_nonzero() -> None:
    controls = _load()["exact_controls"]
    assert controls["Minkowski_q0_inverse_identity"]["exact_residual"] == "0"
    assert controls["omit_inverse_metric_tangent"] == {
        "exact_residual": "-1",
        "rejected": True,
    }
    assert controls["drop_off_diagonal_sqrt2_normalization"]["rejected"] is True
    assert controls["cylindrical_p1_metric_H22"]["exact_projection"] == "1/2"
    assert controls["cylindrical_p1_scalar_H22"]["exact_projection"] == "1"


def test_resealed_projection_alias_or_D2_tamper_fails_closed() -> None:
    mutations = (
        lambda value: value["lower_projection_registry"][1]["tangent_seed"].update(
            {"dg_value": "1"}
        ),
        lambda value: value["alias_reconciliation"].update({"lower_formal_slot_alias_groups": 1}),
        lambda value: value["gate_counts"].update(
            {"D2_entries_registered_per_candidate_after": 5325}
        ),
        lambda value: value["claim_seals"].update({"complete_D2F": True}),
    )
    for mutate in mutations:
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        _reseal(corrupted)
        with pytest.raises(LowerCoordinateProjectionError, match="result changed"):
            validate_campaign(corrupted, root=ROOT)
