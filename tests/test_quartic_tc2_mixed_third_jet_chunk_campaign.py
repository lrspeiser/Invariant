import copy
import json
from collections import Counter
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_mixed_third_jet_chunk_campaign import (
    _content_hash_matches,
    _mixed_selector,
    _record_hash_matches,
    _triple_kind,
    generic_mixed_third_polarization_control,
    run_quartic_tc2_mixed_third_jet_chunk_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
ARTIFACT = RUNS / "quartic-tc2-mixed-third-jet-chunk-campaign" / "campaign.json"


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
        _load(RUNS / "quartic-tc2-diagonal-third-jet-campaign" / "campaign.json"),
        _load(RUNS / "quartic-tc2-quadratic-deltak-extension-campaign" / "campaign.json"),
        _canonical_artifacts(),
        _load(ROOT / "configs" / "backgrounds" / "quartic_tc2_mixed_third_jet_chunk_campaign.json"),
    )


def test_generic_polarization_and_selector_controls() -> None:
    passed, control = generic_mixed_third_polarization_control()
    assert passed
    assert control["AAB_control"] == "0"
    assert control["ABB_control"] == "0"
    assert control["ABC_control"] == "0"
    assert all(item["rejected"] for item in control["negative_controls"].values())
    selector = _mixed_selector()
    assert len(selector) == 12300
    assert selector[0] == (0, 0, 1)
    assert selector[63] == (0, 1, 24)
    assert Counter(_triple_kind(triple) for triple in selector[:64]) == {
        "AAB": 40,
        "ABB": 1,
        "ABC": 23,
    }


def test_first_64_mixed_triples_are_exact_restartable_and_reproducible() -> None:
    result = run_quartic_tc2_mixed_third_jet_chunk_campaign(*_inputs())
    assert result == _load(ARTIFACT)
    assert _content_hash_matches(result)
    assert result["status"] == "pass_mixed_third_jet_chunk_64_global_closure_fail_closed"
    assert result["errors"] == []
    assert result["first_exact_obstruction"] is None
    assert result["counts"] == {
        "selected": 64,
        "triple_kind_counts": {"AAB": 40, "ABB": 1, "ABC": 23},
        "symbolic_parameter_compatible": 64,
        "candidate_evaluations": 768,
        "candidate_solvable": 768,
        "candidate_obstructed": 0,
        "mixed_triples_remaining": 12236,
        "full_tube_Sylvester_identities": 0,
        "TC2_closures": 0,
        "B7_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    contract = result["chunk_contract"]
    assert contract["chunk_offset"] == 0
    assert contract["processed_count"] == 64
    assert contract["next_offset"] == 64
    assert contract["prior_resume_sha256"] is None
    assert not contract["stopped_early"]
    manifest = result["triple_manifest"]
    assert len(manifest) == 64
    assert all(_record_hash_matches(record) for record in manifest)
    assert all(
        record["previous_record_sha256"]
        == (contract["resume_seed_sha256"] if index == 0 else manifest[index - 1]["record_sha256"])
        for index, record in enumerate(manifest)
    )
    assert manifest[-1]["record_sha256"] == contract["resume_tip_sha256"]
    assert manifest[0]["active_position_triple"] == [0, 0, 1]
    assert manifest[-1]["active_position_triple"] == [0, 1, 24]
    assert all(record["symbolic_parameter_compatible"] for record in manifest)
    assert all(record["obstructed_candidate_ids"] == [] for record in manifest)
    candidate_results = [
        candidate for record in manifest for candidate in record["candidate_results"]
    ]
    assert len(candidate_results) == 768
    assert all(candidate["solvable"] for candidate in candidate_results)
    assert all(candidate["deltaK_ABC_Hermitian"] for candidate in candidate_results)
    assert all(candidate["third_Sylvester_residual_zero"] for candidate in candidate_results)
    assert all(candidate["deltaK_ABC_nonzero_entries"] == 0 for candidate in candidate_results)
    assert all(
        not value
        for key, value in result["closure_ledger"].items()
        if key != "processed_mixed_third_jets_closed"
    )
    assert result["closure_ledger"]["processed_mixed_third_jets_closed"] == 64


def test_hash_tamper_and_false_global_policy_reject() -> None:
    inputs = list(_inputs())
    corrupt = copy.deepcopy(inputs[0])
    corrupt["content_sha256"] = "0" * 64
    inputs[0] = corrupt
    rejected = run_quartic_tc2_mixed_third_jet_chunk_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["selected"] == 0

    inputs = list(_inputs())
    config = copy.deepcopy(inputs[-1])
    config["TC2_policy"] = "closed"
    inputs[-1] = config
    rejected = run_quartic_tc2_mixed_third_jet_chunk_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["TC2_closures"] == 0
