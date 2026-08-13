from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_pother_inverse_product_d2_replay_gate import (
    CONFIG_PATH,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    _bound_labels,
    _content_sha,
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


def test_build_matches_artifact(result: dict[str, object]) -> None:
    assert result == json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def test_exact_counts(result: dict[str, object]) -> None:
    assert result["decision"] == "pass_all_180_Pother_roots_all_264_bounded_targets_sealed"
    assert result["decision_counts"] == {"pass": 12, "blocked": 0, "reject": 0}
    counts = result["gate_counts"]
    assert counts["presented_Pother_leaf_derivative_roots"] == 23760
    assert counts["Pother_ordered_D2_roots_registered"] == 180
    assert counts["nonzero_Pother_ordered_D2_roots"] == 156
    assert counts["zero_Pother_ordered_D2_roots"] == 24
    assert counts["all_bounded_target_ordered_D2_roots_registered"] == 264
    assert counts["complete_ordered_D2F_tensors_registered"] == 0


def test_candidate_manifests(result: dict[str, object]) -> None:
    roots, records = set(), set()
    for manifest in result["candidate_manifests"]:
        assert manifest["Pother_ordered_D2_roots_registered"] == 15
        assert manifest["nonzero_Pother_ordered_D2_roots"] == 13
        assert manifest["zero_Pother_ordered_D2_roots"] == 2
        assert manifest["candidate_rejection_authorized"] is False
        assert manifest["first_blocker"] == FIRST_BLOCKER
        for packet in manifest["replay_packets"]:
            assert packet["D1_quotient_domain_assumption"] == "c11=(-1)^11 det(A) is nonzero"
            assert packet["leaf_background_domain_assumption"] == (
                "g_is_nonsingular_and_gu_is_its_exact_inverse"
            )
            roots.add(packet["D2_merkle_replay_root_sha256"])
        records.update(row["ordered_D2_record_id"] for row in manifest["ordered_Pother_D2_records"])
    assert len(roots) == 157
    assert len(records) == 180


def test_typed_B_and_C_labels_are_distinct() -> None:
    predecessor = json.loads(
        (
            ROOT
            / "runs/physics-language/quartic-pother-arbitrary-background-leaf-derivative-gate/campaign.json"
        ).read_text(encoding="utf-8")
    )
    packets = predecessor["candidate_manifests"][0]["direction_packets"]
    assert any(any(label.startswith("B_1[") for label in _bound_labels(row)) for row in packets)
    assert any(any(label.startswith("C_11[") for label in _bound_labels(row)) for row in packets)


def test_claims_fail_closed(result: dict[str, object]) -> None:
    seals = result["claim_seals"]
    assert seals["Pother_ordered_D2_roots_registered"] is True
    assert seals["all_264_bounded_target_ordered_D2_roots_registered"] is True
    for key in (
        "complete_ordered_D2F_tensor_registered",
        "full_high_atom_identity_closed",
        "physical_no_go_proved",
        "global_H7_energy_closed",
        "nonlinear_PDE_closed",
        "nonlinear_lifespan_proved",
        "candidate_theory_rejected",
        "observational_claim_made",
    ):
        assert seals[key] is False
    assert set(result["data_seals"].values()) == {False}


def test_bindings(result: dict[str, object]) -> None:
    bindings = result["source_bindings"]
    for label in ("source", "config", "test"):
        binding = bindings[label]
        assert (
            hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
            == binding["file_sha256"]
        )


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("gate_counts", "Pother_ordered_D2_roots_registered", 179),
        ("gate_counts", "complete_ordered_D2F_tensors_registered", 12),
        ("claim_seals", "complete_ordered_D2F_tensor_registered", True),
        ("claim_seals", "physical_no_go_proved", True),
        ("data_seals", "live_SQLite_opened", True),
    ],
)
def test_resealed_tamper_fails(
    result: dict[str, object], section: str, key: str, replacement: object
) -> None:
    tampered = copy.deepcopy(result)
    tampered[section][key] = replacement
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_resealed_root_tamper_fails(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["candidate_manifests"][0]["replay_packets"][0]["D2_merkle_replay_root_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)


def test_unknown_key_fails(result: dict[str, object]) -> None:
    tampered = copy.deepcopy(result)
    tampered["overclaim"] = True
    with pytest.raises(ValueError, match="result boundary changed"):
        _validate_result(_reseal(tampered), root=ROOT)
