from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_full_sphere_degree_seven_transverse_curl_gate import (
    DENSE_RATIONAL_ENTRY_CEILING,
    ODD_SPHERE_MODE_DIMENSION,
    SYMBOLIC_ROW_DESCRIPTORS,
    UNKNOWN_COLUMNS,
    FullSphereDegreeSevenGateError,
    _content_hash,
    build_campaign,
    solve_exact_sparse_system,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_d4_full_sphere_degree_seven_transverse_curl_gate.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-d4-full-sphere-degree-seven-transverse-curl-gate"
    / "campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_phase_one_seals_exact_sparse_topology(artifact: dict) -> None:
    phase = artifact["phase_one"]
    topology = phase["symbolic_sparse_topology"]
    assert phase["decision"] == "PASS_EXACT_PREREGISTRATION_AND_SPARSE_TOPOLOGY_READINESS"
    assert phase["upstream_seals_verified"] == 4
    assert topology["column_index"]["homogeneous_monomials"] == 28
    assert topology["column_index"]["deduplicated_descriptors"] == UNKNOWN_COLUMNS
    assert topology["row_index"]["odd_sphere_modes"] == ODD_SPHERE_MODE_DIMENSION
    assert topology["row_index"]["deduplicated_descriptors"] == SYMBOLIC_ROW_DESCRIPTORS
    assert topology["dense_rational_entry_ceiling"] == DENSE_RATIONAL_ENTRY_CEILING
    assert topology["column_index"]["duplicates_removed"] == 0
    assert topology["row_index"]["duplicates_removed"] == 0


def test_phase_two_is_honestly_blocked_before_solve(artifact: dict) -> None:
    phase = artifact["phase_two"]
    assert phase == {
        "BLOCK": True,
        "OBSTRUCTED_CLASS": False,
        "PASS": False,
        "attempted": False,
        "decision": "BLOCK",
        "dense_solve_admitted": False,
        "first_blocker": (
            "register_the_coordinate_free_D4_sparse_coefficient_map_and_exact_rhs_"
            "for_all_558_equal_eigenspace_cokernel_coordinates_and_210_odd_sphere_modes"
        ),
        "solve_admitted": False,
        "sparse_solve_required": True,
    }
    assert artifact["counts"]["coefficient_entries_materialized"] == 0
    assert artifact["counts"]["candidate_replays_attempted"] == 0
    assert artifact["counts"]["full_sphere_passes"] == 0


def test_exact_solver_emits_verified_pass() -> None:
    result = solve_exact_sparse_system(
        [{0: 1, 1: 1}, {0: 1, 1: -1}],
        [3, 1],
        2,
        maximum_rows=8,
        maximum_columns=8,
        maximum_nonzeros=16,
    )
    assert result["decision"] == "PASS"
    assert result["solution"] == ["2", "1"]
    assert result["exact_residuals"] == ["0", "0"]
    assert result["exact_solution_verified"] is True


def test_exact_solver_emits_verified_obstructed_class() -> None:
    result = solve_exact_sparse_system(
        [{0: 1}, {0: 1}],
        [0, 1],
        1,
        maximum_rows=8,
        maximum_columns=8,
        maximum_nonzeros=16,
    )
    assert result["decision"] == "OBSTRUCTED_CLASS"
    assert result["exact_witness_verified"] is True
    assert result["witness_times_matrix"] == ["0"]
    assert result["witness_times_rhs"] != "0"


def test_exact_solver_blocks_over_cap_without_attempt() -> None:
    result = solve_exact_sparse_system(
        [{0: 1}, {0: 1}],
        [0, 1],
        1,
        maximum_rows=1,
        maximum_columns=8,
        maximum_nonzeros=16,
    )
    assert result["decision"] == "BLOCK"
    assert result["attempted"] is False
    assert result["reason"] == "exact_solver_resource_cap_not_admitted"


def test_broad_claims_remain_false(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["phase_one_readiness_passed"] is True
    assert claims["exact_sparse_column_and_row_topology_constructed"] is True
    for claim in (
        "full_direction_sphere_D4_compatibility_proved",
        "complete_D2F_tensor_registered",
        "full_high_atom_identity_proved",
        "TC2_closed",
        "global_H7_closed",
        "full_tube_Sylvester_identity_proved",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
        "theory_candidate_rejected",
    ):
        assert claims[claim] is False


def test_all_negative_controls_reject(artifact: dict) -> None:
    controls = artifact["negative_controls"]
    assert len(controls) == 10
    assert all(control == {"rejected": True} for control in controls.values())


def test_artifact_replays_deterministically(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)


def test_semantic_artifact_tamper_fails_closed(artifact: dict) -> None:
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["full_sphere_passes"] = 1
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(FullSphereDegreeSevenGateError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)


def test_upstream_binding_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["upstream_bindings"]["full_Sylvester_reference"]["content_sha256"] = "0" * 64
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FullSphereDegreeSevenGateError, match="config upstream hash mismatch"):
        build_campaign(ROOT, path)


def test_resource_cap_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["resource_caps"]["maximum_dense_rational_entries"] += 1
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FullSphereDegreeSevenGateError, match="invalid full-sphere gate config"):
        build_campaign(ROOT, path)
