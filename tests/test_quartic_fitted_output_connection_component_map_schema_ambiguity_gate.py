from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.quartic_fitted_output_connection_component_map_schema_ambiguity_gate import (
    CONFIG_PATH,
    EXPECTED_EVIDENCE,
    EXPECTED_PREDECESSOR,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUTPUT_PATH


def _load_artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expect_rejected(value: dict, match: str = "component map") -> None:
    _reseal(value)
    with pytest.raises(ValueError, match=match):
        _validate_result(value, root=ROOT)


def _matrix(records: list[dict], rows: int = 22, columns: int = 24) -> sp.Matrix:
    matrix = sp.zeros(rows, columns)
    for record in records:
        matrix[record["output_coordinate"], record["generic_term"]] = sp.sympify(record["value"])
    return matrix


def test_checked_artifact_is_exact_rebuild() -> None:
    checked = _load_artifact()
    assert build_gate(ROOT / CONFIG_PATH) == checked
    _validate_result(checked, root=ROOT)


def test_projection_witnesses_are_exact_and_distinct() -> None:
    projection = _load_artifact()["generic_term_projection_ambiguity"]
    coefficients = sp.Matrix(
        [sp.sympify(value) for value in projection["generic_coefficient_vector"]]
    )
    beta = sp.Matrix([sp.sympify(value) for value in projection["target_beta_vector"]])
    base = _matrix(projection["base_sparse_entries"])
    alternate = _matrix(projection["alternate_sparse_entries"])
    assert base * coefficients == beta
    assert alternate * coefficients == beta
    assert base != alternate
    assert projection["base_residual_nonzero_count"] == 0
    assert projection["alternate_residual_nonzero_count"] == 0


def test_projection_dimension_count_is_exact() -> None:
    projection = _load_artifact()["generic_term_projection_ambiguity"]
    assert projection["matrix_shape"] == [22, 24]
    assert projection["unknown_entries"] == 528
    assert projection["value_constraints"] == 22
    assert projection["constraint_rank"] == 22
    assert projection["affine_solution_dimension"] == 506
    assert len(projection["generic_term_ids"]) == 24


def test_every_target_D1_entry_is_registered() -> None:
    artifact = _load_artifact()
    records = artifact["coordinate_records"]
    assert len(records) == 22
    assert {row["coordinate_ordinal"] for row in records} == set(range(22))
    assert all(row["D1_arithmetic_root"] for row in records)
    assert all(row["D1_arithmetic_dag_sha256"] for row in records)
    assert artifact["gate_counts"]["target_D1_memberships_found"] == 22
    assert artifact["gate_counts"]["unique_target_D1_row_atom_entries"] == 20


def test_registered_D1_does_not_supply_mixed_D2() -> None:
    artifact = _load_artifact()
    ambiguity = artifact["mixed_D2_extension_ambiguity"]
    assert ambiguity["mixed_multi_index_components_completed"] == 0
    assert ambiguity["direction_tangent_embeddings_registered"] == 0
    assert ambiguity["target_ordered_mixed_D2F_roots_registered"] == 0
    assert ambiguity["independent_mixed_D2_extension_parameters"] == 22
    assert ambiguity["explicit_witness_completions"] == 23
    assert all(
        row["zero_extension_D2_value"] == "0"
        and row["unit_extension_D2_value"] == "1"
        and row["both_extensions_preserve_registered_D1_value"] is True
        for row in artifact["coordinate_records"]
    )


def test_dag_atoms_do_not_overlap_target_atoms() -> None:
    ambiguity = _load_artifact()["mixed_D2_extension_ambiguity"]
    assert ambiguity["pure_checkpoint_atoms"] == ["q[0]", "p0[10]"]
    assert ambiguity["pure_derivative_roots"] == 1056
    assert ambiguity["target_atom_overlap_with_pure_DAG_checkpoints"] == 0
    assert ambiguity["full_component_Frechet_tensors_complete"] is False


def test_candidate_and_downstream_counts_are_conservative() -> None:
    artifact = _load_artifact()
    assert artifact["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert artifact["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_missing_schema_is_closed_and_complete() -> None:
    missing = _load_artifact()["missing_schema"]
    assert {key for key, value in missing.items() if value is False} == {
        "generic_term_id_to_source_component",
        "P10_Pother_direction_to_153_state_tangent",
        "ordered_mixed_D2F_root_for_each_output_coordinate",
        "corrected_source_jet_to_output_bundle_connection",
    }
    assert set(missing["required_fields"]) == {
        "generic_term_id",
        "source_row",
        "coordinate_atom",
        "direction_tangent_coefficients_in_153_basis",
        "ordered_D2_arithmetic_root",
        "ordered_D2_arithmetic_dag_sha256",
        "output_bundle_projection_rule_id",
        "candidate_id",
    }


def test_claim_seals_distinguish_schema_obstruction_from_physical_no_go() -> None:
    claims = _load_artifact()["claim_seals"]
    assert claims["all_22_target_D1_row_atom_entries_registered"] is True
    assert claims["two_exact_term_projection_witnesses_constructed"] is True
    assert claims["term_projection_affine_dimension_506_proved"] is True
    assert claims["mixed_D2_22_parameter_ambiguity_constructed"] is True
    assert claims["registered_cross_registry_component_map_unique"] is False
    assert claims["physical_covariant_component_map_no_go_proved"] is False
    assert claims["candidate_theory_rejected"] is False
    assert claims["observational_claim_made"] is False


def test_all_global_claims_remain_closed() -> None:
    claims = _load_artifact()["claim_seals"]
    for key in (
        "covariant_output_connection_derivation_registered",
        "corrected_second_source_jet_registered",
        "cross_slice_D2F_entries_admitted",
        "complete_ordered_D2F_tensor_registered",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
    ):
        assert claims[key] is False


def test_live_bindings_are_exact() -> None:
    bindings = _load_artifact()["source_bindings"]
    assert bindings["source"] == {"path": SOURCE_PATH, "file_sha256": _sha(ROOT / SOURCE_PATH)}
    assert bindings["config"] == {"path": CONFIG_PATH, "file_sha256": _sha(ROOT / CONFIG_PATH)}
    assert bindings["test"] == {"path": TEST_PATH, "file_sha256": _sha(ROOT / TEST_PATH)}
    assert bindings["predecessor"] == EXPECTED_PREDECESSOR
    assert bindings["direct_evidence"] == EXPECTED_EVIDENCE


def test_unknown_top_level_key_is_rejected() -> None:
    value = _load_artifact()
    value["unknown"] = True
    _expect_rejected(value)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("generic_term_projection_ambiguity", "affine_solution_dimension"), 505),
        (("generic_term_projection_ambiguity", "base_residual_nonzero_count"), 1),
        (("mixed_D2_extension_ambiguity", "mixed_multi_index_components_completed"), 1),
        (("gate_counts", "target_D1_memberships_found"), 21),
        (("gate_counts", "target_ordered_mixed_D2F_roots_registered"), 1),
        (("missing_schema", "generic_term_id_to_source_component"), True),
        (("claim_seals", "physical_covariant_component_map_no_go_proved"), True),
        (("claim_seals", "complete_ordered_D2F_tensor_registered"), True),
        (("decision_counts", "reject"), 1),
        (("downstream_admission_counts", "pass"), 1),
    ],
)
def test_resealed_boundary_tampers_are_rejected(path: tuple[str, str], replacement: object) -> None:
    value = _load_artifact()
    value[path[0]][path[1]] = replacement
    _expect_rejected(value)


def test_resealed_projection_entry_tamper_is_rejected() -> None:
    value = _load_artifact()
    value["generic_term_projection_ambiguity"]["alternate_sparse_entries"][0]["value"] = "9"
    _expect_rejected(value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("D1_arithmetic_root", "tampered-root"),
        ("direction_state_tangent_registered", True),
        ("ordered_mixed_D2F_root_registered", True),
    ],
)
def test_resealed_coordinate_tampers_are_rejected(field: str, replacement: object) -> None:
    value = _load_artifact()
    value["coordinate_records"][0][field] = replacement
    _expect_rejected(value)


@pytest.mark.parametrize("binding", ["source", "config", "test", "predecessor"])
def test_resealed_direct_binding_tampers_are_rejected(binding: str) -> None:
    value = _load_artifact()
    value["source_bindings"][binding]["file_sha256"] = "0" * 64
    _expect_rejected(value)


@pytest.mark.parametrize("bundle", sorted(EXPECTED_EVIDENCE))
def test_resealed_evidence_binding_tampers_are_rejected(bundle: str) -> None:
    value = _load_artifact()
    value["source_bindings"]["direct_evidence"][bundle]["artifact"]["content_sha256"] = "0" * 64
    _expect_rejected(value)


def test_raw_content_hash_tamper_is_rejected() -> None:
    value = copy.deepcopy(_load_artifact())
    value["scope"] = "tampered"
    with pytest.raises(ValueError, match="content hash"):
        _validate_result(value, root=ROOT)
