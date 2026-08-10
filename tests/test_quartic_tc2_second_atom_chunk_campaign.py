import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_second_atom_chunk_campaign import (
    _canonical_active_affine_pairs,
    generic_second_atom_sylvester_control,
    run_quartic_tc2_second_atom_chunk_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-variable-sylvester-campaign"
    / "campaign.json"
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_second_atom_chunk_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-second-atom-chunk-campaign"
    / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def test_mixed_recurrence_selector_and_negative_controls_are_exact() -> None:
    passed, control = generic_second_atom_sylvester_control()
    assert passed
    assert control["mixed_Sylvester_residual"] == "0"
    assert all(item["rejected"] for item in control["negative_controls"].values())

    pairs = _canonical_active_affine_pairs()
    assert len(pairs) == 861
    assert pairs[0]["global_pair_index"] == 7028
    assert (pairs[0]["left_atom"], pairs[0]["right_atom"]) == (
        "s01[2]",
        "s01[2]",
    )
    assert pairs[63]["global_pair_index"] == 7180


def test_first_64_pair_chunk_constructs_every_delta_but_stays_fail_closed() -> None:
    result = run_quartic_tc2_second_atom_chunk_campaign(
        _load(VARIABLE), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_first_64_second_atom_pairs_no_obstruction_remaining_fail_closed"
    )
    assert result["first_exact_obstruction"] is None
    assert result["counts"] == {
        "total_unordered_coordinate_atom_pairs": 11781,
        "evaluated_coordinate_atom_pairs": 64,
        "remaining_unevaluated_coordinate_atom_pairs": 11717,
        "candidates": 12,
        "evaluated_candidate_pairs": 768,
        "solvable_candidate_pairs": 768,
        "obstructed_candidate_pairs": 0,
        "deltaK_AB_constructions": 768,
        "TC2_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    assert result["exact_tensor_summary"] == {
        "unique_direction_pair_packets": 46,
        "nonzero_D2P55_packets": 15,
        "nonzero_D2K55_packets": 31,
        "nonzero_D2TC2_packets": 0,
        "nonzero_deltaK_AB_packets": 24,
        "maximum_deltaK_AB_rank": 2,
        "all_deltaK_AB_Hermitian": True,
        "all_second_Sylvester_residuals_zero": True,
    }
    assert len(result["pair_manifest"]) == 64
    assert len({item["global_pair_index"] for item in result["pair_manifest"]}) == 64
    assert all(
        candidate["solvable"]
        and candidate["deltaK_AB_sha256"] is not None
        and candidate["Hermitian"]
        and candidate["second_Sylvester_residual_zero"]
        for record in result["pair_manifest"]
        for candidate in record["candidate_results"]
    )
    previous = result["chunk_contract"]["chunk_seed_sha256"]
    for record in result["pair_manifest"]:
        assert record["previous_record_sha256"] == previous
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        assert record["record_sha256"] == _hash(body)
        previous = record["record_sha256"]
    assert previous == result["chunk_contract"]["resume_after_record_sha256"]
    assert result == _load(ARTIFACT)


def test_upstream_tamper_chunk_contract_and_false_promotion_reject() -> None:
    variable = _load(VARIABLE)
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(variable))
    corrupt["content_sha256"] = "0" * 64
    result = run_quartic_tc2_second_atom_chunk_campaign(corrupt, config)
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]

    wrong_size = dict(config)
    wrong_size["chunk_size"] = 63
    result = run_quartic_tc2_second_atom_chunk_campaign(variable, wrong_size)
    assert result["status"] == "reject"
    assert "unsupported second-atom chunk contract" in result["errors"][0]

    for policy in ("global_H7_policy", "lifespan_policy"):
        promoted = dict(config)
        promoted[policy] = "pass"
        result = run_quartic_tc2_second_atom_chunk_campaign(variable, promoted)
        assert result["status"] == "reject"
