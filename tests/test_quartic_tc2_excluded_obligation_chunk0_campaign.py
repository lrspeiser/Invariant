import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_excluded_obligation_chunk0_campaign import (
    run_quartic_tc2_excluded_obligation_chunk0_campaign,
)
from sigma_theory_compiler.quartic_tc2_second_atom_chunk_campaign import (
    _canonical_active_affine_pairs,
    _direction_key,
    _second_pair_symbolic_packet,
)
from sigma_theory_compiler.quartic_tc2_variable_sylvester_campaign import (
    _content_hash,
    _content_hash_matches,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
VARIABLE = RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"
CLASSIFICATION = (
    RUNS / "quartic-tc2-excluded-pair-classification-campaign" / "campaign.json"
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_excluded_obligation_chunk0_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-tc2-excluded-obligation-chunk0-campaign" / "campaign.json"
)
LEGACY_CHUNK = RUNS / "quartic-tc2-second-atom-chunk-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_chunk_replays_exact_artifact() -> None:
    result = run_quartic_tc2_excluded_obligation_chunk0_campaign(
        _load(VARIABLE), _load(CLASSIFICATION), _load(CONFIG)
    )
    assert result == _load(ARTIFACT)
    assert _content_hash_matches(result)


def test_first_64_exact_counts_corrections_and_restart_chain() -> None:
    result = _load(ARTIFACT)
    assert result["status"] == (
        "pass_first_64_excluded_obligations_no_obstruction_remaining_fail_closed"
    )
    assert result["first_exact_obstruction"] is None
    assert result["counts"] == {
        "full_unordered_coordinate_atom_pairs": 11781,
        "completed_canonical_active_pairs": 861,
        "classification_discharged_zero_pairs": 8245,
        "total_excluded_obligations": 2675,
        "current_evaluated_obligations": 64,
        "remaining_unevaluated_obligations": 2611,
        "candidates": 12,
        "current_candidate_checks": 768,
        "current_solvable_candidate_checks": 768,
        "current_obstructed_candidate_checks": 0,
        "TC2_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    contract = result["chunk_contract"]
    assert contract["chunk_seed_sha256"] == (
        "098a79ee4054e183ffd9524e0e8219a1fb5f4d86279a5dc065c1b4c796b10a68"
    )
    assert contract["resume_after_record_sha256"] == (
        "d437ea7885125f3d39ced4615be9ff1e6755ae4843d0a907f18e9f78478a3dd1"
    )
    manifest = result["pair_manifest"]
    assert len(manifest) == 64
    assert manifest[0]["global_pair_index"] == 55
    assert manifest[-1]["global_pair_index"] == 263
    previous = contract["chunk_seed_sha256"]
    for local_index, record in enumerate(manifest):
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        assert record["chunk_local_index"] == local_index
        assert record["previous_record_sha256"] == previous
        assert record["record_sha256"] == _content_hash(body)
        assert all(
            candidate["solvable"]
            and candidate["deltaK_AB_sha256"] is not None
            and candidate["Hermitian"]
            and candidate["second_Sylvester_residual_zero"]
            for candidate in record["candidate_results"]
        )
        previous = record["record_sha256"]
    assert previous == contract["resume_after_record_sha256"]


def test_coordinate_D2_pushforward_is_bound_and_omission_rejects() -> None:
    result = _load(ARTIFACT)
    control = result["omitted_D2J_negative_control"]
    assert control == {
        "pair_global_index": 55,
        "required_D2J_sha256": (
            "d447a5e1c067128a1d544451cfdf014624c9a3fef6158f51efd45e021602f776"
        ),
        "with_pushforward_packet_sha256": (
            "5a7bf078fda68a8156ee2748ad151989894589bd908b91797a763a75ade5ba14"
        ),
        "omitted_pushforward_packet_sha256": (
            "d9475d16d4199825e69f1fd387db919002d6d5781663a000c81d10c77f82d718"
        ),
        "packet_changes_when_D2J_omitted": True,
        "rejected": True,
    }
    packet = next(
        item
        for item in result["symbolic_pair_packets"]
        if item["content_sha256"] == control["with_pushforward_packet_sha256"]
    )
    assert packet["coordinate_D2_pushforward_included"]
    assert packet["second_coordinate_direction"] == {
        "G_22": "sqrt(2)/2",
        "G_33": "sqrt(2)/2",
    }


def test_optional_D2J_extension_preserves_legacy_packet_exactly() -> None:
    legacy = _load(LEGACY_CHUNK)
    pair = _canonical_active_affine_pairs()[0]
    packet = _second_pair_symbolic_packet(
        _direction_key(pair["left_direction"]),
        _direction_key(pair["right_direction"]),
    )
    assert packet["content_sha256"] == legacy["pair_manifest"][0][
        "symbolic_pair_packet_sha256"
    ]
    assert "coordinate_D2_pushforward_included" not in packet


def test_manifest_tamper_wrong_selector_and_false_promotion_reject() -> None:
    variable, classification, config = (
        _load(VARIABLE),
        _load(CLASSIFICATION),
        _load(CONFIG),
    )
    corrupt = json.loads(json.dumps(classification))
    corrupt["excluded_pair_manifest"][0]["global_pair_index"] += 1
    body = {key: value for key, value in corrupt.items() if key != "content_sha256"}
    corrupt["content_sha256"] = _content_hash(body)
    result = run_quartic_tc2_excluded_obligation_chunk0_campaign(
        variable, corrupt, config
    )
    assert result["status"] == "reject"

    wrong = dict(config)
    wrong["selector_sha256"] = "0" * 64
    result = run_quartic_tc2_excluded_obligation_chunk0_campaign(
        variable, classification, wrong
    )
    assert result["status"] == "reject"

    promoted = dict(config)
    promoted["global_H7_policy"] = "pass"
    result = run_quartic_tc2_excluded_obligation_chunk0_campaign(
        variable, classification, promoted
    )
    assert result["status"] == "reject"


def test_artifact_file_hash_is_stable() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "4c1e7abe67ea7369ca33893197c89cd6f1a446a9edfdf1dda55be27e3d495744"
    )
