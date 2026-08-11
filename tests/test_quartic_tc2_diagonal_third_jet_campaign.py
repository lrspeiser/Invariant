import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import (
    _content_hash_matches,
    generic_diagonal_third_jet_control,
    run_quartic_tc2_diagonal_third_jet_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
ARTIFACT = RUNS / "quartic-tc2-diagonal-third-jet-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_artifacts() -> list[dict]:
    paths = [
        RUNS / "quartic-tc2-second-atom-chunk-campaign" / "campaign.json",
        *[
            RUNS / f"quartic-tc2-second-atom-chunk{offset}-campaign" / "campaign.json"
            for offset in range(64, 704, 64)
        ],
        *sorted((RUNS / "quartic-tc2-continuous-service" / "chunks").glob("offset-*.json")),
    ]
    return [_load(path) for path in paths]


def _inputs() -> tuple:
    return (
        _load(RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"),
        _load(RUNS / "quartic-tc2-quadratic-deltak-extension-campaign" / "campaign.json"),
        _canonical_artifacts(),
        _load(ROOT / "configs" / "backgrounds" / "quartic_tc2_diagonal_third_jet_campaign.json"),
    )


def test_generic_control_rejects_diagonal_to_mixed_or_tube_promotion() -> None:
    passed, control = generic_diagonal_third_jet_control()
    assert passed
    assert control["Taylor_coefficient_residual_zero"]
    assert control["third_derivative_multiplier"] == 6
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_41_diagonal_active_third_jets_are_exact_and_reproducible() -> None:
    result = run_quartic_tc2_diagonal_third_jet_campaign(*_inputs())
    assert result == _load(ARTIFACT)
    assert _content_hash_matches(result)
    assert result["status"] == (
        "pass_bounded_all_41_diagonal_active_coordinate_third_jet_audit_"
        "mixed_triples_full_tube_global_H7_fail_closed"
    )
    assert result["slice_contract"] == {
        "coordinate_sector": "canonical active affine second-partial atoms at fixed q,p",
        "active_coordinate_directions": 41,
        "tested_triples": "(A,A,A) for every one of the 41 active coordinate directions",
        "diagonal_triples": 41,
        "mixed_AAB_ABB_ABC_triples": 0,
        "full_symmetric_triples_in_41_direction_sector": 12341,
        "full_symmetric_triples_in_153_coordinate_basis": 608685,
        "factorial_normalized_internal_recurrence": True,
        "reported_deltaK_AAA_is_third_derivative": True,
    }
    assert result["counts"] == {
        "candidates": 12,
        "diagonal_direction_packets": 41,
        "symbolic_parameter_diagonal_third_jet_passes": 41,
        "candidate_direction_evaluations": 492,
        "candidate_direction_solvable": 492,
        "candidate_direction_obstructed": 0,
        "candidates_all_41_diagonal_third_jets_closed": 12,
        "mixed_third_jet_closures": 0,
        "full_tube_Sylvester_identities": 0,
        "TC2_closures": 0,
        "B7_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    records = result["direction_records"]
    assert len(records) == 41
    assert sum(item["D3P55_nonzero_entries"] > 0 for item in records) == 11
    assert sum(item["D3K55_nonzero_entries"] > 0 for item in records) == 26
    assert sum(item["D3TC2_nonzero_entries"] > 0 for item in records) == 0
    assert all(item["committed_D2_deltaK_reproduced"] for item in records)
    assert all(item["symbolic_equal_eigenspace_compressions_zero"] for item in records)
    assert all(item["symbolic_deltaK_AAA_Hermitian"] for item in records)
    assert sum(item["symbolic_deltaK_AAA_nonzero_entries"] > 0 for item in records) == 26
    assert max(item["symbolic_deltaK_AAA_nonzero_entries"] for item in records) == 80
    assert max(item["symbolic_deltaK_AAA_rank"] for item in records) == 6
    for key in (
        "D3P55_sha256",
        "D3K55_sha256",
        "D3TC2_sha256",
        "third_Sylvester_RHS_sha256",
        "symbolic_deltaK_AAA_sha256",
    ):
        assert all(len(item[key]) == 64 for item in records)
    candidate_results = [item for record in records for item in record["candidate_results"]]
    assert len(candidate_results) == 492
    assert all(item["solvable"] for item in candidate_results)
    assert all(item["deltaK_AAA_Hermitian"] for item in candidate_results)
    assert all(item["third_Sylvester_residual_zero"] for item in candidate_results)
    assert sum(item["deltaK_AAA_nonzero_entries"] > 0 for item in candidate_results) == 312
    assert max(item["deltaK_AAA_nonzero_entries"] for item in candidate_results) == 80
    assert max(item["deltaK_AAA_rank"] for item in candidate_results) == 6
    assert all(item["all_41_diagonal_third_jets_closed"] for item in result["certificates"])
    assert not result["first_remaining_blocker"]["closed"]


def test_hash_tamper_and_false_global_policy_reject() -> None:
    inputs = list(_inputs())
    corrupt = copy.deepcopy(inputs[1])
    corrupt["content_sha256"] = "0" * 64
    inputs[1] = corrupt
    rejected = run_quartic_tc2_diagonal_third_jet_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["diagonal_direction_packets"] == 0

    inputs = list(_inputs())
    config = copy.deepcopy(inputs[-1])
    config["global_H7_policy"] = "closed"
    inputs[-1] = config
    rejected = run_quartic_tc2_diagonal_third_jet_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["global_H7_closures"] == 0
