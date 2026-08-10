import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_second_atom_chunk64_campaign import (
    _validate_prior_chain,
)
from sigma_theory_compiler.quartic_tc2_second_atom_chunk512_campaign import (
    generic_second_atom_chunk512_boundary_control,
    run_quartic_tc2_second_atom_chunk512_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PRIOR = RUNS / "quartic-tc2-second-atom-chunk448-campaign" / "campaign.json"
VARIABLE = RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_second_atom_chunk512_campaign.json"
)
ARTIFACT = RUNS / "quartic-tc2-second-atom-chunk512-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def test_offset_448_chain_and_512_boundary_are_exact() -> None:
    chain_passed, chain = _validate_prior_chain(_load(PRIOR))
    boundary_passed, boundary = generic_second_atom_chunk512_boundary_control()
    assert chain_passed and chain["passed"] and chain["records_checked"] == 64
    assert boundary_passed and boundary["expected_next_offset"] == 512
    assert all(item["rejected"] for item in boundary["negative_controls"].values())


def test_offset_512_chunk_reaches_576_without_obstruction_but_stays_open() -> None:
    result = run_quartic_tc2_second_atom_chunk512_campaign(
        _load(PRIOR), _load(VARIABLE), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_cumulative_576_second_atom_pairs_no_obstruction_remaining_fail_closed"
    )
    assert result["first_exact_obstruction"] is None
    assert result["counts"] == {
        "total_unordered_coordinate_atom_pairs": 11781,
        "prior_cumulative_evaluated_coordinate_atom_pairs": 512,
        "current_evaluated_coordinate_atom_pairs": 64,
        "cumulative_evaluated_coordinate_atom_pairs": 576,
        "remaining_unevaluated_coordinate_atom_pairs": 11205,
        "candidates": 12,
        "current_evaluated_candidate_pairs": 768,
        "current_solvable_candidate_pairs": 768,
        "current_obstructed_candidate_pairs": 0,
        "cumulative_deltaK_AB_constructions": 6912,
        "TC2_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    assert result["exact_tensor_summary_current_chunk"] == {
        "unique_direction_pair_packets": 55,
        "nonzero_D2P55_packets": 34,
        "nonzero_D2K55_packets": 46,
        "nonzero_D2TC2_packets": 0,
        "nonzero_deltaK_AB_packets": 44,
        "maximum_deltaK_AB_rank": 6,
        "all_deltaK_AB_Hermitian": True,
        "all_second_Sylvester_residuals_zero": True,
    }
    assert result["chunk_contract"]["chunk_offset"] == 512
    assert result["chunk_contract"]["resume_after_record_sha256"] == (
        "5c550896a25a8a8069166c1bee6952d4b00bdf335fd68c86cc40b9d8fc161747"
    )
    assert result["pair_manifest"][0]["selector_pair_index"] == 512
    assert result["pair_manifest"][-1]["selector_pair_index"] == 575
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
    result = run_quartic_tc2_second_atom_chunk512_campaign(
        corrupt, variable, config
    )
    assert result["status"] == "reject"
    assert "prior record chain mismatch" in result["errors"][0]

    wrong_tip = dict(config)
    wrong_tip["prior_resume_sha256"] = "0" * 64
    result = run_quartic_tc2_second_atom_chunk512_campaign(
        prior, variable, wrong_tip
    )
    assert result["status"] == "reject"
    assert "unsupported continuation contract" in result["errors"][0]

    for policy in ("global_H7_policy", "lifespan_policy"):
        promoted = dict(config)
        promoted[policy] = "pass"
        result = run_quartic_tc2_second_atom_chunk512_campaign(
            prior, variable, promoted
        )
        assert result["status"] == "reject"
