import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_mixed_third_jet_chunk_campaign import (
    _content_hash,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_continuation_service import (
    QuarticTC2MixedThirdJetContinuationServiceError,
    _all_global_claims_false,
    _chunk_config,
    _hash_matches,
    _load_state,
    _validate_result,
    _with_hash,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_parallel_epoch import (
    _canonical_second_atom_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT / "runs" / "physics-language" / "quartic-tc2-mixed-third-jet-parallel-continuation-service"
)


def test_parallel_epoch_inputs_are_complete_and_hash_bound() -> None:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "backgrounds"
            / "quartic_tc2_mixed_third_jet_parallel_continuation_service.json"
        ).read_text(encoding="utf-8")
    )
    artifacts = _canonical_second_atom_artifacts(ROOT)
    assert (
        _content_hash([artifact["content_sha256"] for artifact in artifacts])
        == config["canonical_D2_artifact_sequence_sha256"]
    )
    assert config["initial_prior_offset"] == 128
    assert config["start_offset"] == 192
    assert config["parallel_worker_count"] == 8
    assert config["parallel_execution_policy"] == (
        "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
    )


def test_parallel_epoch_artifact_is_exact_ordered_and_fail_closed() -> None:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "backgrounds"
            / "quartic_tc2_mixed_third_jet_parallel_continuation_service.json"
        ).read_text(encoding="utf-8")
    )
    artifact = json.loads((OUTPUT / "chunks" / "offset-000192.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((OUTPUT / "checkpoint.json").read_text(encoding="utf-8"))
    status = json.loads((OUTPUT / "service-status.json").read_text(encoding="utf-8"))
    dynamic = _chunk_config(
        config,
        config["initial_prior_resume_tip_sha256"],
        192,
        64,
    )
    _validate_result(
        artifact,
        dynamic,
        192,
        64,
        config["canonical_D2_artifact_sequence_sha256"],
    )
    assert _load_state(OUTPUT / "checkpoint.json", config) == checkpoint
    assert _hash_matches(checkpoint) and _hash_matches(status)
    assert artifact["chunk_contract"]["parallel_worker_count"] == 8
    assert artifact["chunk_contract"]["processed_count"] == 64
    assert artifact["counts"]["candidate_solvable"] == 768
    assert artifact["counts"]["candidate_obstructed"] == 0
    assert artifact["counts"]["mixed_triples_remaining"] == 12_044
    assert checkpoint["next_offset"] == 256
    assert checkpoint["remaining_mixed_triples"] == 12_044
    assert artifact["first_exact_obstruction"] is None
    assert _all_global_claims_false(artifact["closure_ledger"])
    assert not any(checkpoint["claims"].values())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("parallel_worker_count", 7),
        ("parallel_execution_policy", None),
        ("bounded_speculative_evaluations_may_finish_after_first_obstruction", False),
        ("records_after_first_obstruction_committed_or_inferred", 9),
    ],
)
def test_parallel_epoch_contract_tamper_is_rejected(key: str, value: object) -> None:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "backgrounds"
            / "quartic_tc2_mixed_third_jet_parallel_continuation_service.json"
        ).read_text(encoding="utf-8")
    )
    artifact = json.loads((OUTPUT / "chunks" / "offset-000192.json").read_text(encoding="utf-8"))
    tampered = copy.deepcopy(artifact)
    tampered["chunk_contract"][key] = value
    tampered = _with_hash(
        {name: field for name, field in tampered.items() if name != "content_sha256"}
    )
    dynamic = _chunk_config(
        config,
        config["initial_prior_resume_tip_sha256"],
        192,
        64,
    )
    with pytest.raises(
        QuarticTC2MixedThirdJetContinuationServiceError,
        match="executor result contract mismatch",
    ):
        _validate_result(
            tampered,
            dynamic,
            192,
            64,
            config["canonical_D2_artifact_sequence_sha256"],
        )
