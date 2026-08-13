from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    quartic_fitted_output_connection_registered_variation_selection_audit as gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / gate.CONFIG_PATH
OUTPUT = ROOT / gate.OUTPUT_PATH


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = _sha(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_checked_in_artifact_matches_exact_rebuild() -> None:
    actual = json.loads(OUTPUT.read_text(encoding="utf-8"))
    expected = gate.build_gate(CONFIG)

    assert actual == expected
    gate._validate_result(actual, root=ROOT)


def test_closed_inventory_capability_classification_is_exact() -> None:
    result = gate.build_gate(CONFIG)
    rows = {row["evidence"]: row for row in result["evidence_capabilities"]}

    assert set(rows) == {
        "generic_G4_metric_variation",
        "generated_candidate_metric_variation",
        "universal_source_DAG",
        "full_source_D1",
    }
    assert rows["generic_G4_metric_variation"]["registered_units"] == 24
    assert rows["generated_candidate_metric_variation"]["registered_units"] == 163
    assert rows["generated_candidate_metric_variation"]["quartic_G4_X_grid_candidate_overlap"] == 0
    assert rows["universal_source_DAG"]["registered_units"] == 1056
    assert rows["universal_source_DAG"]["full_component_Frechet_tensors_complete"] is False
    assert rows["full_source_D1"]["registered_units"] == 20196
    assert rows["full_source_D1"]["complete_orders_2_to_4"] is False
    assert all(row["selector_equations_contributed"] == 0 for row in rows.values())
    assert all(row["map_to_22_output_connection_coordinates"] is False for row in rows.values())
    assert all(row["corrected_second_source_jet_values"] == 0 for row in rows.values())


def test_selection_matrix_is_exact_rank_zero_of_22() -> None:
    result = gate.build_gate(CONFIG)

    assert result["selection_matrix"] == {
        "rows": 0,
        "columns": 22,
        "rank": 0,
        "nullity": 22,
        "selected_parameters": 0,
        "unselected_parameters": 22,
    }
    records = result["coordinate_selection_records"]
    assert len(records) == 22
    assert [row["coordinate_ordinal"] for row in records] == list(range(22))
    assert [row["ambiguity_parameter"] for row in records] == [
        f"lambda_{index}" for index in range(22)
    ]
    assert all(row["eligible_selector_equations_registered"] == 0 for row in records)
    assert all(row["parameter_selected"] is False for row in records)


def test_candidate_sets_and_dispositions_remain_closed() -> None:
    result = gate.build_gate(CONFIG)

    assert len(result["candidate_ids"]) == len(set(result["candidate_ids"])) == 12
    assert result["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert result["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }
    assert result["gate_counts"]["registered_corrected_second_source_jet_entries"] == 0
    assert result["gate_counts"]["cross_slice_D2F_entries_admitted"] == 0


def test_theorem_is_explicitly_closed_world_not_physical_no_go() -> None:
    result = gate.build_gate(CONFIG)
    theorem = result["registered_selection_theorem"]

    assert "shape 0-by-22, rank zero, and nullity 22" in theorem["exact_result"]
    assert "neither a physical no-go" in theorem["boundary"]
    assert "invalidates the premise and requires a new gate" in theorem["boundary"]
    assert result["claim_seals"]["physical_covariant_variation_no_go_proved"] is False
    assert result["claim_seals"]["all_22_ambiguity_parameters_remain_unselected"] is True


def test_all_global_claims_and_candidate_rejection_remain_false() -> None:
    claims = gate.build_gate(CONFIG)["claim_seals"]

    for key in (
        "covariant_output_connection_derivation_registered",
        "corrected_second_source_jet_registered",
        "cross_slice_D2F_entries_admitted",
        "complete_ordered_D2F_tensor_registered",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "candidate_theory_rejected",
        "observational_claim_made",
    ):
        assert claims[key] is False
    assert all(
        control["rejected"] is True
        for control in gate.build_gate(CONFIG)["exact_controls"].values()
    )


def test_all_local_predecessor_and_inventory_bindings_are_live() -> None:
    bindings = gate.build_gate(CONFIG)["source_bindings"]

    for label in ("source", "config", "test"):
        binding = bindings[label]
        assert (
            binding["file_sha256"]
            == hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
        )
    predecessor = bindings["predecessor"]
    assert (
        predecessor["file_sha256"]
        == hashlib.sha256((ROOT / predecessor["path"]).read_bytes()).hexdigest()
    )
    for bundle in bindings["evidence_inventory"].values():
        for binding in bundle.values():
            assert (
                binding["file_sha256"]
                == hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            )


def test_unknown_top_key_and_content_hash_tamper_fail() -> None:
    value = gate.build_gate(CONFIG)
    value["unknown"] = True
    _reseal(value)
    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)

    value = gate.build_gate(CONFIG)
    value["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash changed"):
        gate._validate_result(value, root=ROOT)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("rows", 1),
        ("rank", 1),
        ("nullity", 21),
        ("selected_parameters", 1),
        ("unselected_parameters", 21),
    ],
)
def test_resealed_selection_matrix_tampers_fail(field: str, replacement: int) -> None:
    value = gate.build_gate(CONFIG)
    value["selection_matrix"][field] = replacement
    _reseal(value)

    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("eligible_selector_equations_registered", 1),
        ("parameter_selected", True),
        ("ambiguity_parameter", "lambda_tampered"),
    ],
)
def test_resealed_coordinate_selector_tampers_fail(field: str, replacement: object) -> None:
    value = gate.build_gate(CONFIG)
    value["coordinate_selection_records"][0][field] = replacement
    _reseal(value)

    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("selector_equations_contributed", 1),
        ("map_to_22_output_connection_coordinates", True),
        ("corrected_second_source_jet_values", 1),
    ],
)
def test_resealed_evidence_capability_tampers_fail(field: str, replacement: object) -> None:
    value = gate.build_gate(CONFIG)
    value["evidence_capabilities"][0][field] = replacement
    _reseal(value)

    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("claim_seals", "physical_covariant_variation_no_go_proved", True),
        ("claim_seals", "corrected_second_source_jet_registered", True),
        ("claim_seals", "candidate_theory_rejected", True),
        ("decision_counts", "reject", 1),
        ("downstream_admission_counts", "pass", 1),
    ],
)
def test_resealed_claim_and_disposition_tampers_fail(
    section: str, field: str, replacement: object
) -> None:
    value = gate.build_gate(CONFIG)
    value[section][field] = replacement
    _reseal(value)

    with pytest.raises(ValueError, match="result boundary changed"):
        gate._validate_result(value, root=ROOT)


def test_source_predecessor_and_inventory_binding_tampers_fail() -> None:
    value = gate.build_gate(CONFIG)
    value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    _reseal(value)
    with pytest.raises(ValueError, match="local source binding changed"):
        gate._validate_result(value, root=ROOT)

    value = gate.build_gate(CONFIG)
    value["source_bindings"]["predecessor"]["file_sha256"] = "0" * 64
    _reseal(value)
    with pytest.raises(ValueError, match="predecessor binding changed"):
        gate._validate_result(value, root=ROOT)

    value = gate.build_gate(CONFIG)
    value["source_bindings"]["evidence_inventory"]["full_source_D1"]["artifact"]["file_sha256"] = (
        "0" * 64
    )
    _reseal(value)
    with pytest.raises(ValueError, match="inventory binding changed"):
        gate._validate_result(value, root=ROOT)


def test_config_unknown_key_and_inventory_drift_fail() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["unknown"] = True
    with pytest.raises(ValueError, match="config boundary changed"):
        gate._validate_config(config)

    config = copy.deepcopy(config)
    config.pop("unknown")
    config["selection_contract"]["required_selection_rank"] = 21
    with pytest.raises(ValueError, match="config boundary changed"):
        gate._validate_config(config)
