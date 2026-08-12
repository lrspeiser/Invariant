from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.quartic_fitted_output_connection_action_feature_factorization_gate as lane
from sigma_theory_compiler.quartic_fitted_output_connection_action_feature_factorization_gate import (
    CLAIM_SEALS,
    CONFIG_PATH,
    EXPECTED_DIRECT_EVIDENCE,
    EXPECTED_PREDECESSORS,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    _load_bound,
    _validate_config,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_PATH
ARTIFACT = ROOT / OUTPUT_PATH


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _reseal(value: dict[str, object]) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


@pytest.fixture(scope="module")
def gate() -> dict[str, object]:
    return build_gate(CONFIG)


@pytest.fixture(scope="module")
def bound_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    origin = _load_bound(
        ROOT, EXPECTED_PREDECESSORS["fitted_output_connection_covariant_origin_audit"]
    )
    pother = _load_bound(ROOT, EXPECTED_PREDECESSORS["candidate_pother_one_form_connection"])
    action = lane._load_action(ROOT)
    return origin, pother, action


def test_exact_gate_matches_checked_artifact_and_replays(gate: dict[str, object]) -> None:
    checked = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert gate == checked
    assert checked["content_sha256"] == hashlib.sha256(
        _canonical({key: item for key, item in checked.items() if key != "content_sha256"})
    ).hexdigest()


def test_full_grid_gives_unique_exact_declared_affine_factorization(
    gate: dict[str, object],
) -> None:
    counts = gate["gate_counts"]
    assert counts["action_feature_grid_points"] == 12
    assert counts["action_feature_design_rank"] == 3
    assert counts["declared_affine_features"] == 3
    assert counts["fitted_connection_coordinates_per_candidate"] == 22
    assert counts["fitted_connection_values_checked"] == 264
    assert counts["factorization_residual_nonzero_count"] == 0
    assert len(gate["universal_coordinate_map"]) == 22
    assert {(row["action_features"]["G2_X2"], row["action_features"]["G4_X"])
            for row in gate["candidate_records"]} == {
        (x, y) for x in ("-1", "0", "1") for y in ("-1", "-1/2", "1/2", "1")
    }


def test_every_coordinate_factors_only_through_G4_X(gate: dict[str, object]) -> None:
    for row in gate["universal_coordinate_map"]:
        coefficients = row["affine_coefficients"]
        assert coefficients["constant"] == "0"
        assert coefficients["G2_X2"] == "0"
        assert coefficients["G4_X"] != "0"
    assert gate["gate_counts"]["coordinates_with_zero_constant_coefficient"] == 22
    assert gate["gate_counts"]["coordinates_with_zero_G2_X2_coefficient"] == 22
    assert gate["gate_counts"]["coordinates_with_nonzero_G4_X_coefficient"] == 22


def test_finite_factorization_passes_but_origin_and_downstream_remain_blocked(
    gate: dict[str, object],
) -> None:
    assert gate["decision_counts"] == {"pass": 12, "reject": 0, "blocked": 0}
    assert gate["downstream_admission_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert all(row["gate_decision"] == "pass" for row in gate["candidate_records"])
    assert all(row["candidate_decision"] == "blocked" for row in gate["candidate_records"])
    assert not any(row["candidate_rejection_authorized"] for row in gate["candidate_records"])
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "registered_action_feature_grid_bound",
        "candidate_independent_affine_factorization_exists",
        "factorization_unique_in_declared_three_feature_class",
        "all_264_fitted_values_factor_through_G4_X",
        "G2_X2_and_constant_coefficients_vanish_in_declared_fit",
    }
    assert gate["gate_counts"]["fitted_coefficients_with_action_root_provenance"] == 0
    assert gate["gate_counts"]["registered_covariant_derivation_functors"] == 0
    assert gate["gate_counts"]["cross_slice_D2F_entries_admitted"] == 0
    assert gate["gate_counts"]["principal_high_atom_entries_missing_per_candidate"] == 106920
    assert not any(gate["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("multiplier", "result boundary"),
        ("feature", "result boundary"),
        ("residual", "result boundary"),
        ("rank", "result boundary"),
        ("origin_claim", "result boundary"),
        ("provenance_claim", "result boundary"),
        ("second_source", "result boundary"),
        ("admit_D2F", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("unknown_key", "result boundary"),
        ("forge_origin", "predecessor binding"),
        ("forge_pother", "predecessor binding"),
        ("forge_action", "direct evidence binding"),
        ("forge_local", "local binding"),
    ],
)
def test_resealed_semantic_and_provenance_tampering_fails_closed(
    gate: dict[str, object],
    bound_inputs: tuple[dict[str, object], dict[str, object], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    # The exact replay test above exercises the complete live predecessor closure. Reuse
    # those already hash-checked inputs here so the tamper matrix isolates this lane's
    # closed-result comparison rather than repeating the costly two-sided solve 14 times.
    monkeypatch.setattr(lane, "_validated_inputs", lambda _root: bound_inputs)
    value = json.loads(json.dumps(gate))
    if mutation == "multiplier":
        value["universal_coordinate_map"][0]["affine_coefficients"]["G4_X"] = "0"
    elif mutation == "feature":
        value["candidate_records"][0]["action_features"]["G4_X"] = "0"
    elif mutation == "residual":
        value["candidate_records"][0]["factorization_residual_nonzero_count"] = 1
    elif mutation == "rank":
        value["gate_counts"]["action_feature_design_rank"] = 2
    elif mutation == "origin_claim":
        value["claim_seals"]["factorization_is_covariant_derivation"] = True
    elif mutation == "provenance_claim":
        value["claim_seals"]["fitted_coefficients_with_action_root_provenance"] = True
    elif mutation == "second_source":
        value["gate_counts"]["registered_corrected_second_source_jet_entries"] = 1
    elif mutation == "admit_D2F":
        value["claim_seals"]["cross_slice_D2F_entries_admitted"] = True
    elif mutation == "reject_candidate":
        value["candidate_records"][0]["candidate_rejection_authorized"] = True
    elif mutation == "unknown_key":
        value["promotion"] = True
    elif mutation == "forge_origin":
        value["source_bindings"]["fitted_output_connection_covariant_origin_audit"][
            "content_sha256"
        ] = "0" * 64
    elif mutation == "forge_pother":
        value["source_bindings"]["candidate_pother_one_form_connection"][
            "content_sha256"
        ] = "0" * 64
    elif mutation == "forge_action":
        value["source_bindings"]["direct_evidence"]["covariant_action"]["artifact"][
            "content_sha256"
        ] = "0" * 64
    else:
        value["source_bindings"]["test"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        _validate_result(_reseal(value), root=ROOT)


def test_config_paths_and_closed_bindings(gate: dict[str, object]) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["policies"]["global_H7"] = "pass"
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(config)
    with pytest.raises(ValueError, match="path escapes"):
        _load_bound(
            ROOT,
            {"path": "../outside.json", "file_sha256": "0" * 64, "content_sha256": "0" * 64},
        )
    assert gate["source_bindings"]["direct_evidence"] == EXPECTED_DIRECT_EVIDENCE
    for label, binding in EXPECTED_PREDECESSORS.items():
        assert gate["source_bindings"][label] == binding
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        assert gate["source_bindings"][label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
