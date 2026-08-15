from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_p10_pother_coordinate_tangent_embedding_gate import (
    BASIS_SHA256,
    CONFIG_PATH,
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


def _load() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(value: dict) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reject(value: dict) -> None:
    _reseal(value)
    with pytest.raises(ValueError, match="coordinate tangent"):
        _validate_result(value, root=ROOT)


def _records(artifact: dict) -> list[dict]:
    return [
        row for manifest in artifact["candidate_manifests"] for row in manifest["embedding_records"]
    ]


def test_checked_artifact_is_exact_rebuild() -> None:
    artifact = _load()
    assert build_gate(ROOT / CONFIG_PATH) == artifact
    _validate_result(artifact, root=ROOT)


def test_coordinate_basis_is_exact_bijection() -> None:
    registry = _load()["coordinate_basis_registry"]
    atoms = registry["coordinate_atom_basis"]
    assert registry["dimension"] == 153
    assert len(atoms) == len(set(atoms)) == 153
    assert registry["coordinate_atom_basis_sha256"] == BASIS_SHA256
    assert registry["column_bijection_proved"] is True
    assert registry["source_rows_per_column"] == 11


def test_all_264_embeddings_are_candidate_bound_and_unique() -> None:
    records = _records(_load())
    assert len(records) == 264
    assert len({row["embedding_id"] for row in records}) == 264
    assert all(row["candidate_bound"] is True for row in records)
    assert all(row["candidate_rejection_authorized"] is False for row in records)


def test_every_embedding_is_the_exact_registered_unit_vector() -> None:
    atoms = _load()["coordinate_basis_registry"]["coordinate_atom_basis"]
    for row in _records(_load()):
        column = row["coordinate_column"]
        assert atoms[column] == row["coordinate_atom"]
        assert row["basis_dimension"] == 153
        assert row["coefficient_field"] == "Q"
        assert row["sparse_entries"] == [{"coordinate_column": column, "coefficient": "1"}]
        assert row["support_size"] == 1
        assert row["exact_squared_norm"] == "1"
        assert row["canonical_coordinate_unit_tangent"] is True


def test_direction_labels_match_atom_field_classes() -> None:
    for row in _records(_load()):
        field_index = int(row["coordinate_atom"].split("[")[1][:-1])
        expected = "P10" if field_index == 10 else "Pother"
        assert row["direction_label"] == expected


def test_per_candidate_counts_are_exact() -> None:
    for manifest in _load()["candidate_manifests"]:
        assert manifest["registered_embeddings"] == 22
        assert manifest["P10_embedding_records"] == 7
        assert manifest["Pother_embedding_records"] == 15
        assert manifest["unique_coordinate_unit_vectors"] == 20
        assert manifest["candidate_decision"] == "pass_coordinate_tangents_downstream_blocked"


def test_global_counts_are_exact_and_fail_closed() -> None:
    artifact = _load()
    counts = artifact["gate_counts"]
    assert counts["registered_coordinate_tangent_embeddings"] == 264
    assert counts["P10_embedding_records"] == 84
    assert counts["Pother_embedding_records"] == 180
    assert counts["unique_coordinate_unit_vectors_global"] == 20
    assert counts["registered_ordered_mixed_D2_roots"] == 0
    assert counts["blocked_ordered_mixed_D2_roots"] == 264
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_decision_counts_do_not_promote_downstream() -> None:
    artifact = _load()
    assert artifact["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert artifact["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }


def test_claims_distinguish_coordinate_embedding_from_covariant_map() -> None:
    claims = _load()["claim_seals"]
    assert claims["all_264_coordinate_unit_tangents_registered"] is True
    assert claims["atom_column_bijection_replayed"] is True
    assert claims["P10_Pother_field_index_classification_replayed"] is True
    assert claims["generic_term_component_projection_registered"] is False
    assert claims["physical_covariant_component_map_proved"] is False
    assert claims["physical_covariant_component_map_no_go_proved"] is False


def test_all_broad_claims_remain_closed() -> None:
    claims = _load()["claim_seals"]
    for key in (
        "ordered_mixed_D2_values_registered",
        "corrected_second_source_jet_registered",
        "complete_ordered_D2F_tensor_registered",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "candidate_theory_rejected",
        "observational_claim_made",
    ):
        assert claims[key] is False


def test_live_bindings_are_exact() -> None:
    bindings = _load()["source_bindings"]
    assert bindings["source"] == {"path": SOURCE_PATH, "file_sha256": _file_sha(ROOT / SOURCE_PATH)}
    assert bindings["config"] == {"path": CONFIG_PATH, "file_sha256": _file_sha(ROOT / CONFIG_PATH)}
    assert bindings["test"] == {"path": TEST_PATH, "file_sha256": _file_sha(ROOT / TEST_PATH)}
    assert bindings["predecessor"] == EXPECTED_PREDECESSOR
    assert bindings["full_source_D1_artifact"]["content_sha256"] == (
        "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a"
    )


def test_raw_content_tamper_is_rejected() -> None:
    value = _load()
    value["scope"] = "tampered"
    with pytest.raises(ValueError, match="content hash"):
        _validate_result(value, root=ROOT)


def test_unknown_top_level_key_is_rejected() -> None:
    value = _load()
    value["unknown"] = True
    _reject(value)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("decision_counts", "reject", 1),
        ("downstream_admission_counts", "pass", 1),
        ("gate_counts", "registered_coordinate_tangent_embeddings", 263),
        ("gate_counts", "registered_ordered_mixed_D2_roots", 1),
        ("coordinate_basis_registry", "dimension", 152),
        ("claim_seals", "generic_term_component_projection_registered", True),
        ("claim_seals", "ordered_mixed_D2_values_registered", True),
        ("claim_seals", "physical_covariant_component_map_no_go_proved", True),
        ("claim_seals", "candidate_theory_rejected", True),
    ],
)
def test_resealed_summary_tampers_are_rejected(
    section: str, field: str, replacement: object
) -> None:
    value = _load()
    value[section][field] = replacement
    _reject(value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("embedding_id", "0" * 64),
        ("coordinate_atom", "s11[9]"),
        ("coordinate_column", 0),
        ("direction_label", "Pother"),
        ("support_size", 2),
        ("exact_squared_norm", "2"),
        ("ordered_D2_arithmetic_root_registered", True),
        ("candidate_rejection_authorized", True),
    ],
)
def test_resealed_embedding_tampers_are_rejected(field: str, replacement: object) -> None:
    value = _load()
    value["candidate_manifests"][0]["embedding_records"][0][field] = replacement
    _reject(value)


def test_resealed_sparse_coefficient_tamper_is_rejected() -> None:
    value = _load()
    value["candidate_manifests"][0]["embedding_records"][0]["sparse_entries"][0]["coefficient"] = (
        "2"
    )
    _reject(value)


@pytest.mark.parametrize("binding", ["source", "config", "test"])
def test_resealed_local_binding_tampers_are_rejected(binding: str) -> None:
    value = _load()
    value["source_bindings"][binding]["file_sha256"] = "0" * 64
    _reject(value)


@pytest.mark.parametrize("binding", ["source", "config", "test", "artifact"])
def test_resealed_predecessor_binding_tampers_are_rejected(binding: str) -> None:
    value = _load()
    value["source_bindings"]["predecessor"][binding]["file_sha256"] = "0" * 64
    _reject(value)


def test_resealed_full_source_binding_tamper_is_rejected() -> None:
    value = _load()
    value["source_bindings"]["full_source_D1_artifact"]["content_sha256"] = "0" * 64
    _reject(value)
