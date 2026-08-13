from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_ordered_mixed_d2_root_minimal_registration_contract import (
    CONFIG_PATH,
    EXPECTED_PREDECESSOR,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    REQUIRED_FIELDS,
    SOURCE_PATH,
    TEST_PATH,
    _validate_result,
    build_contract,
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
    with pytest.raises(ValueError, match="ordered mixed D2"):
        _validate_result(value, root=ROOT)


def test_checked_artifact_is_exact_rebuild() -> None:
    artifact = _load()
    assert build_contract(ROOT / CONFIG_PATH) == artifact
    _validate_result(artifact, root=ROOT)


def test_cartesian_manifest_is_complete_and_unique() -> None:
    artifact = _load()
    manifests = artifact["candidate_manifests"]
    obligations = [row for manifest in manifests for row in manifest["obligations"]]
    assert len(manifests) == 12
    assert len(obligations) == 264
    assert len({row["obligation_id"] for row in obligations}) == 264
    assert all(len(manifest["obligations"]) == 22 for manifest in manifests)
    assert all(manifest["required_obligations"] == 22 for manifest in manifests)


def test_each_candidate_has_every_coordinate_ordinal() -> None:
    for manifest in _load()["candidate_manifests"]:
        assert {row["coordinate_ordinal"] for row in manifest["obligations"]} == set(range(22))


def test_every_obligation_is_candidate_bound_and_D1_anchored() -> None:
    for manifest in _load()["candidate_manifests"]:
        for row in manifest["obligations"]:
            assert row["candidate_id"] == manifest["candidate_id"]
            assert row["registered_D1_arithmetic_root"]
            assert len(row["registered_D1_arithmetic_dag_sha256"]) == 64
            assert row["obligation_status"] == "required_unregistered"
            assert row["candidate_rejection_authorized"] is False


def test_coordinate_templates_preserve_the_exact_target_inventory() -> None:
    templates = _load()["coordinate_templates"]
    assert len(templates) == 22
    assert {row["coordinate_ordinal"] for row in templates} == set(range(22))
    assert {row["direction_label"] for row in templates} == {"P10", "Pother"}
    assert len({(row["source_row"], row["coordinate_atom"]) for row in templates}) == 20


def test_registration_schema_names_every_required_field() -> None:
    schema = _load()["registration_schema"]
    assert schema["required_registration_fields"] == REQUIRED_FIELDS
    assert schema["unique_obligation_key"] == ["candidate_id", "coordinate_ordinal"]
    assert schema["candidate_count"] == 12
    assert schema["coordinate_obligations_per_candidate"] == 22
    assert schema["obligation_count"] == 264


def test_no_missing_value_is_fabricated() -> None:
    artifact = _load()
    schema = artifact["registration_schema"]
    assert schema["registered_cross_registry_projection_records"] == 0
    assert schema["registered_direction_tangent_records"] == 0
    assert schema["registered_ordered_D2_root_records"] == 0
    for manifest in artifact["candidate_manifests"]:
        for row in manifest["obligations"]:
            for key in (
                "direction_tangent_basis_sha256_registered",
                "direction_tangent_sparse_coefficients_registered",
                "generic_term_component_projection_rule_id_registered",
                "output_bundle_projection_rule_id_registered",
                "ordered_D2_arithmetic_root_registered",
                "ordered_D2_arithmetic_dag_sha256_registered",
            ):
                assert row[key] is False


def test_counts_preserve_fail_closed_boundary() -> None:
    artifact = _load()
    assert artifact["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert artifact["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }
    assert artifact["gate_counts"]["required_ordered_mixed_D2_roots"] == 264
    assert artifact["gate_counts"]["registered_ordered_mixed_D2_roots"] == 0
    assert artifact["gate_counts"]["blocked_ordered_mixed_D2_roots"] == 264
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_claims_distinguish_contract_from_physical_no_go() -> None:
    claims = _load()["claim_seals"]
    assert claims["exact_264_slot_registration_contract_complete"] is True
    assert claims["all_slots_candidate_bound_and_D1_anchored"] is True
    assert claims["physical_covariant_component_map_no_go_proved"] is False
    assert claims["candidate_theory_rejected"] is False
    assert claims["observational_claim_made"] is False


def test_all_broad_claims_remain_closed() -> None:
    claims = _load()["claim_seals"]
    for key in (
        "generic_term_component_projection_registered",
        "direction_tangent_embeddings_registered",
        "ordered_mixed_D2_values_registered",
        "corrected_second_source_jet_registered",
        "complete_ordered_D2F_tensor_registered",
        "full_high_atom_good_unknown_identity_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
    ):
        assert claims[key] is False


def test_live_local_and_predecessor_bindings_are_exact() -> None:
    bindings = _load()["source_bindings"]
    assert bindings["source"] == {"path": SOURCE_PATH, "file_sha256": _file_sha(ROOT / SOURCE_PATH)}
    assert bindings["config"] == {"path": CONFIG_PATH, "file_sha256": _file_sha(ROOT / CONFIG_PATH)}
    assert bindings["test"] == {"path": TEST_PATH, "file_sha256": _file_sha(ROOT / TEST_PATH)}
    assert bindings["predecessor"] == EXPECTED_PREDECESSOR


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
        ("gate_counts", "required_ordered_mixed_D2_roots", 263),
        ("gate_counts", "registered_ordered_mixed_D2_roots", 1),
        ("registration_schema", "obligation_count", 263),
        ("registration_schema", "registered_direction_tangent_records", 1),
        ("claim_seals", "physical_covariant_component_map_no_go_proved", True),
        ("claim_seals", "ordered_mixed_D2_values_registered", True),
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
        ("obligation_id", "0" * 64),
        ("candidate_id", "unknown"),
        ("coordinate_ordinal", 22),
        ("registered_D1_arithmetic_root", "invented"),
        ("direction_tangent_sparse_coefficients_registered", True),
        ("ordered_D2_arithmetic_root_registered", True),
        ("candidate_rejection_authorized", True),
    ],
)
def test_resealed_obligation_tampers_are_rejected(field: str, replacement: object) -> None:
    value = _load()
    value["candidate_manifests"][0]["obligations"][0][field] = replacement
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


def test_resealed_full_D1_binding_tamper_is_rejected() -> None:
    value = _load()
    value["source_bindings"]["full_source_D1_artifact"]["content_sha256"] = "0" * 64
    _reject(value)
