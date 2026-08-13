from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_flat_factorized_leaf_jet_d2_specialization_gate import (
    CONFIG_PATH,
    EXPECTED_EVIDENCE,
    EXPECTED_PREDECESSOR,
    FIRST_BLOCKER,
    MAP_SHA256,
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
    with pytest.raises(ValueError, match="flat factorized D2"):
        _validate_result(value, root=ROOT)


def _records(artifact: dict) -> list[dict]:
    return [
        row for manifest in artifact["candidate_manifests"] for row in manifest["flat_D2_records"]
    ]


def test_checked_artifact_is_exact_rebuild() -> None:
    artifact = _load()
    assert build_gate(ROOT / CONFIG_PATH) == artifact
    _validate_result(artifact, root=ROOT)


def test_factorized_manifest_is_compact_and_complete() -> None:
    artifact = _load()
    manifest = artifact["factorized_leaf_derivative_manifest"]
    assert len(manifest) == 20
    assert sum(len(row["coordinate_ordinals"]) for row in manifest) == 22
    assert all(row["typed_jet_support_size"] <= 2 for row in manifest)
    assert len(artifact["factorized_manifest_sha256"]) == 64


def test_all_264_flat_D2_records_are_materialized() -> None:
    records = _records(_load())
    assert len(records) == 264
    assert len({(row["candidate_id"], row["coordinate_ordinal"]) for row in records}) == 264
    assert all(row["flat_D2_arithmetic_dag_sha256"] for row in records)
    assert all(type(row["flat_D2_arithmetic_root"]) is int for row in records)
    assert all(row["flat_typed_map_sha256"] == MAP_SHA256 for row in records)


def test_flat_value_census_is_exact() -> None:
    counts = Counter(row["flat_D2_value"] for row in _records(_load()))
    assert counts == {"0": 192, "-1": 18, "-1/2": 18, "1/2": 18, "1": 18}
    gate = _load()["gate_counts"]
    assert gate["flat_D2_roots_materialized"] == 264
    assert gate["flat_D2_nonzero_roots"] == 72
    assert gate["flat_D2_zero_roots"] == 192
    assert gate["flat_unique_exact_values"] == 5


def test_flat_specialized_arithmetic_DAG_replays_values() -> None:
    artifact = _load()
    dag = artifact["flat_specialized_D2_arithmetic_DAG"]
    assert dag["allowed_operations"] == ["exact_constant"]
    assert dag["node_count"] == 5
    values = {index: node["value"] for index, node in enumerate(dag["nodes"])}
    for row in _records(artifact):
        assert values[row["flat_D2_arithmetic_root"]] == row["flat_D2_value"]


def test_each_candidate_has_22_flat_and_zero_general_roots() -> None:
    for manifest in _load()["candidate_manifests"]:
        assert manifest["flat_D2_roots_materialized"] == 22
        assert manifest["flat_D2_nonzero_roots"] == 6
        assert manifest["general_background_D2_roots_registered"] == 0
        assert len(manifest["flat_D2_records"]) == 22
        assert manifest["candidate_rejection_authorized"] is False


def test_general_background_boundary_remains_closed() -> None:
    artifact = _load()
    counts = artifact["gate_counts"]
    assert counts["predecessor_leaf_obligations_factorized"] == 31680
    assert counts["general_background_leaf_derivative_roots_registered"] == 0
    assert counts["general_background_D2_roots_registered"] == 0
    assert counts["general_background_D2_roots_blocked"] == 264
    assert all(row["general_background_D2_root_registered"] is False for row in _records(artifact))
    assert artifact["first_blocker"] == FIRST_BLOCKER


def test_decision_counts_do_not_promote_general_background() -> None:
    artifact = _load()
    assert artifact["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert artifact["general_background_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }


def test_claims_distinguish_flat_specialization_from_general_map() -> None:
    claims = _load()["claim_seals"]
    assert claims["flat_typed_coordinate_map_replayed"] is True
    assert claims["flat_A_B_C_leaf_derivative_factorization_replayed"] is True
    assert claims["all_264_flat_D2_values_materialized"] is True
    assert claims["general_background_coordinate_map_registered"] is False
    assert claims["general_background_D2_values_registered"] is False
    assert claims["physical_no_go_proved"] is False


def test_all_broad_claims_remain_closed() -> None:
    claims = _load()["claim_seals"]
    for key in (
        "complete_ordered_D2F_tensor_registered",
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
    assert bindings["direct_evidence"] == EXPECTED_EVIDENCE


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
        ("general_background_admission_counts", "pass", 1),
        ("gate_counts", "flat_D2_roots_materialized", 263),
        ("gate_counts", "flat_D2_nonzero_roots", 71),
        ("gate_counts", "general_background_D2_roots_registered", 1),
        ("claim_seals", "general_background_coordinate_map_registered", True),
        ("claim_seals", "general_background_D2_values_registered", True),
        ("claim_seals", "physical_no_go_proved", True),
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
        ("flat_D2_value", "1"),
        ("flat_D2_arithmetic_root", 99),
        ("coordinate_atom", "unknown"),
        ("coordinate_column", 0),
        ("flat_typed_map_sha256", "0" * 64),
        ("general_background_D2_root_registered", True),
        ("candidate_rejection_authorized", True),
    ],
)
def test_resealed_flat_record_tampers_are_rejected(field: str, replacement: object) -> None:
    value = _load()
    value["candidate_manifests"][0]["flat_D2_records"][0][field] = replacement
    _reject(value)


def test_resealed_factorization_tamper_is_rejected() -> None:
    value = _load()
    value["factorized_leaf_derivative_manifest"][0]["typed_jet_support_size"] = 99
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


@pytest.mark.parametrize("bundle", sorted(EXPECTED_EVIDENCE))
def test_resealed_evidence_binding_tampers_are_rejected(bundle: str) -> None:
    value = _load()
    value["source_bindings"]["direct_evidence"][bundle]["artifact"]["content_sha256"] = "0" * 64
    _reject(value)
