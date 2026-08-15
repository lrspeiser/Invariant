from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_pother_arbitrary_background_leaf_derivative_gate import (
    CONFIG_PATH,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    _content_sha,
    _expression_DAG,
    _generic_derivatives,
    _second_metric_tangent,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return build_gate(ROOT / CONFIG_PATH)


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value["content_sha256"] = _content_sha(value)
    return value


def test_build_matches_immutable_artifact(result: dict[str, object]) -> None:
    stored = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert result == stored
    assert stored["content_sha256"] == _content_sha(stored)


def test_exact_leaf_and_replay_ready_census(result: dict[str, object]) -> None:
    assert result["decision"] == ("pass_23760_Pother_leaf_roots_all_180_D2_records_replay_ready")
    assert result["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    assert result["downstream_admission_counts"] == {
        "pass": 0,
        "blocked": 12,
        "reject": 0,
    }
    counts = result["gate_counts"]
    assert counts["P10_ordered_D2_roots_previously_sealed"] == 84
    assert counts["Pother_target_records"] == 180
    assert counts["registered_Pother_leaf_derivative_roots"] == 23760
    assert counts["nonzero_Pother_leaf_derivative_roots"] == 156
    assert counts["zero_Pother_leaf_derivative_roots"] == 23604
    assert counts["Pother_ordered_D2_roots_replay_ready"] == 180
    assert counts["Pother_ordered_D2_roots_registered"] == 0
    assert counts["all_target_ordered_D2_roots_registered"] == 84


def test_every_candidate_has_complete_Pother_leaf_packets(
    result: dict[str, object],
) -> None:
    manifests = result["candidate_manifests"]
    assert len(manifests) == 12
    for manifest in manifests:
        assert manifest["unique_Pother_directions"] == 15
        assert manifest["Pother_target_records"] == 15
        assert manifest["registered_leaf_derivative_roots"] == 1980
        assert manifest["nonzero_leaf_derivative_roots"] == 13
        assert manifest["zero_leaf_derivative_roots"] == 1967
        assert manifest["Pother_ordered_D2_roots_registered"] == 0
        assert manifest["Pother_ordered_D2_roots_replay_ready"] == 15
        assert manifest["candidate_rejection_authorized"] is False
        assert manifest["first_blocker"] == FIRST_BLOCKER
        assert len(manifest["direction_packets"]) == 15
        for packet in manifest["direction_packets"]:
            assert packet["total_leaf_derivative_roots"] == 132
            assert packet["registered_symbolic_background_scope"] is True
            assert packet["A_derivative_shape"] == [11, 11]
            assert packet["source_chunk_column_shape"] == [11]
            assert packet["G_upper_tangent_sha256"]
            assert packet["dense_root_manifest_sha256"]


def test_exact_sparse_support_has_thirteen_nonzero_and_two_zero_directions() -> None:
    packets = _generic_derivatives()
    assert len(packets) == 15
    assert sum(row["nonzero_leaf_derivatives"] for row in packets.values()) == 13
    assert sum(row["zero_leaf_derivatives"] for row in packets.values()) == 1967
    assert {atom for atom, row in packets.items() if row["nonzero_leaf_derivatives"] == 0} == {
        "s11[4]",
        "s22[7]",
    }
    assert all(not row["source_chunk_column_derivative_sparse_entries"] for row in packets.values())
    assert all(len(row["A_derivative_sparse_entries"]) in {0, 1} for row in packets.values())


def test_metric_component_weight_and_G_upper_tangent_are_not_flattened() -> None:
    _, _, _, diagonal = _second_metric_tangent("s11[0]")
    _, _, _, off_diagonal = _second_metric_tangent("s01[1]")
    assert any(value != 0 for value in diagonal.values())
    assert all("sqrt(2)" not in str(value) for value in diagonal.values())
    assert any("sqrt(2)" in str(value) for value in off_diagonal.values())
    assert any("g_" in str(value) for value in diagonal.values())
    assert any("gu_" in str(value) for value in diagonal.values())


def test_expression_DAG_rejects_unregistered_symbol() -> None:
    with pytest.raises(ValueError, match="expression symbol escaped"):
        _expression_DAG({"forbidden_background_symbol"})


def test_expression_DAG_has_closed_domain(result: dict[str, object]) -> None:
    dag = result["leaf_derivative_arithmetic_DAG"]
    assert dag["allowed_operations"] == ["exact_sympy_rational_expression"]
    contract = dag["background_symbol_contract"]
    assert len(contract["lower_metric_symbols"]) == 10
    assert len(contract["inverse_metric_symbols"]) == 10
    assert contract["coefficient_symbols"] == ["alpha"]
    assert contract["domain"] == "g_is_nonsingular_and_gu_is_its_exact_inverse"
    assert dag["node_count"] == len(dag["nodes"])
    assert all(node["srepr_sha256"] for node in dag["nodes"])


def test_claims_remain_fail_closed(result: dict[str, object]) -> None:
    seals = result["claim_seals"]
    assert seals["all_23760_Pother_leaf_derivative_roots_registered"] is True
    assert seals["all_180_Pother_ordered_D2_records_replay_ready"] is True
    for key in (
        "Pother_ordered_D2_roots_registered",
        "physical_no_go_proved",
        "complete_ordered_D2F_tensor_registered",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "candidate_theory_rejected",
        "observational_claim_made",
    ):
        assert seals[key] is False
    assert set(result["data_seals"].values()) == {False}
    assert all(value == {"rejected": True} for value in result["exact_controls"].values())


def test_source_config_test_predecessor_and_evidence_are_hash_bound(
    result: dict[str, object],
) -> None:
    bindings = result["source_bindings"]
    assert set(bindings) == {
        "source",
        "config",
        "test",
        "predecessor",
        "direct_evidence",
    }
    for label in ("source", "config", "test"):
        path = ROOT / bindings[label]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == bindings[label]["file_sha256"]
    for bundle in (
        bindings["predecessor"],
        bindings["direct_evidence"]["nonlinear_evolution"],
    ):
        for label in ("source", "config", "test", "artifact"):
            path = ROOT / bundle[label]["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle[label]["file_sha256"]


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("decision", "pass_complete_D2F"),
        ("first_blocker", "none"),
        ("manifest_sha256", "0" * 64),
        ("scope", "fully covariant theorem"),
    ],
)
def test_resealed_top_level_tamper_fails_closed(
    result: dict[str, object], key: str, replacement: object
) -> None:
    tampered = copy.deepcopy(result)
    tampered[key] = replacement
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("gate_counts", "registered_Pother_leaf_derivative_roots", 23759),
        ("gate_counts", "Pother_ordered_D2_roots_registered", 180),
        ("claim_seals", "Pother_ordered_D2_roots_registered", True),
        ("claim_seals", "complete_ordered_D2F_tensor_registered", True),
        ("claim_seals", "physical_no_go_proved", True),
        ("data_seals", "live_SQLite_opened", True),
    ],
)
def test_resealed_nested_tamper_fails_closed(
    result: dict[str, object], section: str, key: str, replacement: object
) -> None:
    tampered = copy.deepcopy(result)
    tampered[section][key] = replacement
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_resealed_tangent_hash_tamper_fails_closed(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["generic_tangent_packets"][0]["G_upper_tangent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_resealed_leaf_root_tamper_fails_closed(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    packet = tampered["candidate_manifests"][0]["direction_packets"][0]
    packet["A_derivative_sparse_entries"][0]["arithmetic_root"] += 1
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_resealed_inverse_domain_tamper_fails_closed(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["leaf_derivative_arithmetic_DAG"]["background_symbol_contract"]["domain"] = (
        "gu_independent"
    )
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_unknown_top_level_key_fails_closed(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["unregistered_claim"] = True
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)
