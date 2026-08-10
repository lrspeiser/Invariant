import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_second_atom_chunk64_campaign import (
    _validate_prior_chain,
)
from sigma_theory_compiler.quartic_tc2_second_atom_chunk640_campaign import (
    generic_second_atom_chunk640_boundary_control,
    run_quartic_tc2_second_atom_chunk640_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PRIOR = RUNS / "quartic-tc2-second-atom-chunk576-campaign" / "campaign.json"
VARIABLE = RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_second_atom_chunk640_campaign.json"
)
ARTIFACT = RUNS / "quartic-tc2-second-atom-chunk640-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def test_offset_576_chain_and_640_boundary_are_exact() -> None:
    chain_passed, chain = _validate_prior_chain(_load(PRIOR))
    boundary_passed, boundary = generic_second_atom_chunk640_boundary_control()
    assert chain_passed and chain["passed"] and chain["records_checked"] == 64
    assert boundary_passed and boundary["expected_next_offset"] == 640
    assert all(item["rejected"] for item in boundary["negative_controls"].values())


def test_offset_640_chunk_reaches_704_without_obstruction_but_stays_open() -> None:
    result = run_quartic_tc2_second_atom_chunk640_campaign(
        _load(PRIOR), _load(VARIABLE), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_cumulative_704_second_atom_pairs_no_obstruction_remaining_fail_closed"
    )
    assert result["first_exact_obstruction"] is None
    assert result["counts"] == {
        "total_unordered_coordinate_atom_pairs": 11781,
        "prior_cumulative_evaluated_coordinate_atom_pairs": 640,
        "current_evaluated_coordinate_atom_pairs": 64,
        "cumulative_evaluated_coordinate_atom_pairs": 704,
        "remaining_unevaluated_coordinate_atom_pairs": 11077,
        "candidates": 12,
        "current_evaluated_candidate_pairs": 768,
        "current_solvable_candidate_pairs": 768,
        "current_obstructed_candidate_pairs": 0,
        "cumulative_deltaK_AB_constructions": 8448,
        "TC2_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    assert result["exact_tensor_summary_current_chunk"] == {
        "unique_direction_pair_packets": 59,
        "nonzero_D2P55_packets": 35,
        "nonzero_D2K55_packets": 47,
        "nonzero_D2TC2_packets": 0,
        "nonzero_deltaK_AB_packets": 40,
        "maximum_deltaK_AB_rank": 6,
        "all_deltaK_AB_Hermitian": True,
        "all_second_Sylvester_residuals_zero": True,
    }
    assert result["chunk_contract"]["chunk_offset"] == 640
    assert result["chunk_contract"]["resume_after_record_sha256"] == (
        "40e13981bf06a03d0eff1d85c45afea9335806dd2520d801141aa23dd991e222"
    )
    assert result["pair_manifest"][0]["selector_pair_index"] == 640
    assert result["pair_manifest"][-1]["selector_pair_index"] == 703
    assert all(
        candidate["solvable"]
        and candidate["deltaK_AB_sha256"] is not None
        and candidate["Hermitian"]
        and candidate["second_Sylvester_residual_zero"]
        for record in result["pair_manifest"]
        for candidate in record["candidate_results"]
    )
    assert result == _load(ARTIFACT)


def test_prior_chain_tamper_wrong_tip_and_false_promotions_reject() -> None:
    prior = _load(PRIOR)
    variable = _load(VARIABLE)
    config = _load(CONFIG)
    corrupt = json.loads(json.dumps(prior))
    corrupt["pair_manifest"][15]["selector_pair_index"] += 1
    _rehash(corrupt)
    result = run_quartic_tc2_second_atom_chunk640_campaign(
        corrupt, variable, config
    )
    assert result["status"] == "reject"
    assert "prior record chain mismatch" in result["errors"][0]

    wrong_tip = dict(config)
    wrong_tip["prior_resume_sha256"] = "0" * 64
    result = run_quartic_tc2_second_atom_chunk640_campaign(
        prior, variable, wrong_tip
    )
    assert result["status"] == "reject"
    assert "unsupported continuation contract" in result["errors"][0]

    for policy in ("global_H7_policy", "lifespan_policy"):
        promoted = dict(config)
        promoted[policy] = "pass"
        result = run_quartic_tc2_second_atom_chunk640_campaign(
            prior, variable, promoted
        )
        assert result["status"] == "reject"
