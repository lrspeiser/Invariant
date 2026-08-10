import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_continuous_service import (
    QuarticTC2ContinuousServiceError,
    _checkpoint_hash_matches,
    _chunk_config,
    _chunk_metadata,
    _json_bytes,
    run_continuous_tc2_service,
)
from sigma_theory_compiler.quartic_tc2_variable_sylvester_campaign import (
    _content_hash,
    _content_hash_matches,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PRIOR = RUNS / "quartic-tc2-second-atom-chunk640-campaign" / "campaign.json"
VARIABLE = RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_tc2_continuous_service.json"
REAL_SERVICE = RUNS / "quartic-tc2-continuous-service"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_result(
    prior: dict,
    _variable: dict,
    chunk_config: dict,
    metadata: dict,
    *,
    obstruct: bool = False,
    padding: int = 0,
) -> dict:
    offset = int(chunk_config["chunk_offset"])
    evaluated = 1 if obstruct else 64
    body = {
        "schema_version": metadata["schema_version"],
        "status": (
            metadata["obstruction_status"]
            if obstruct
            else metadata["success_status"]
        ),
        "errors": [],
        "upstream_sha256": {
            "prior_chunk": prior["content_sha256"],
            "prior_resume": prior["chunk_contract"]["resume_after_record_sha256"],
        },
        "chunk_contract": {
            "chunk_offset": offset,
            "resume_after_record_sha256": _content_hash(
                {"offset": offset, "prior": prior["content_sha256"]}
            ),
        },
        "counts": {
            "prior_cumulative_evaluated_coordinate_atom_pairs": offset,
            "cumulative_evaluated_coordinate_atom_pairs": offset + evaluated,
            "TC2_closures": 0,
            "global_H7_closures": 0,
            "lifespans_proved": 0,
        },
        "first_exact_obstruction": (
            {"selector_pair_index": offset, "candidate_id": "fake-0"}
            if obstruct
            else None
        ),
        "padding": "x" * padding,
    }
    return {**body, "content_sha256": _content_hash(body)}


def test_service_checkpoints_resumes_and_stops_before_partial_tail(tmp_path: Path) -> None:
    prior, variable, config = _load(PRIOR), _load(VARIABLE), _load(CONFIG)
    calls: list[int] = []

    def executor(p: dict, v: dict, c: dict, m: dict) -> dict:
        calls.append(int(c["chunk_offset"]))
        return _fake_result(p, v, c, m)

    first = run_continuous_tc2_service(
        prior, variable, config, tmp_path, executor=executor, monotonic=lambda: 0.0
    )
    assert first["status"] == "checkpointed"
    assert first["reason"] == "chunk_limit"
    assert first["chunks_advanced"] == 1
    assert first["next_offset"] == 768
    assert calls == [704]

    second = run_continuous_tc2_service(
        prior, variable, config, tmp_path, executor=executor, monotonic=lambda: 0.0
    )
    assert second["next_offset"] == 832
    assert second["checkpoint"]["completed_chunks"] == 2
    assert calls == [704, 768]

    tail = run_continuous_tc2_service(
        prior, variable, config, tmp_path, executor=executor, monotonic=lambda: 0.0
    )
    assert tail["chunks_advanced"] == 0
    assert tail["reason"] == "selector_tail_requires_partial_chunk_engine"
    assert calls == [704, 768]
    checkpoint = _load(tmp_path / "checkpoint.json")
    assert not any(checkpoint["claims"].values())


def test_orphan_artifact_is_recovered_without_reexecution(tmp_path: Path) -> None:
    prior, variable, config = _load(PRIOR), _load(VARIABLE), _load(CONFIG)
    metadata = _chunk_metadata(704)
    result = _fake_result(
        prior, variable, _chunk_config(config, prior, 704), metadata
    )
    orphan = tmp_path / "chunks" / "offset-000704.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(_json_bytes(result))

    def should_not_run(*_args: object) -> dict:
        raise AssertionError("valid orphan artifact must be recovered idempotently")

    recovered = run_continuous_tc2_service(
        prior,
        variable,
        config,
        tmp_path,
        executor=should_not_run,
        monotonic=lambda: 0.0,
    )
    assert recovered["chunks_advanced"] == 1
    assert recovered["next_offset"] == 768
    assert (tmp_path / "checkpoint.json").is_file()


def test_obstruction_stops_permanently_and_reentry_is_noop(tmp_path: Path) -> None:
    prior, variable, config = _load(PRIOR), _load(VARIABLE), _load(CONFIG)
    calls = 0

    def obstructing(p: dict, v: dict, c: dict, m: dict) -> dict:
        nonlocal calls
        calls += 1
        return _fake_result(p, v, c, m, obstruct=True)

    stopped = run_continuous_tc2_service(
        prior, variable, config, tmp_path, executor=obstructing, monotonic=lambda: 0.0
    )
    assert stopped["status"] == "stopped"
    assert stopped["reason"] == "exact_obstruction"
    assert stopped["chunks_advanced"] == 1
    again = run_continuous_tc2_service(
        prior, variable, config, tmp_path, executor=obstructing, monotonic=lambda: 0.0
    )
    assert again["status"] == "already_stopped"
    assert again["chunks_advanced"] == 0
    assert calls == 1


def test_wall_disk_tamper_and_false_promotion_controls(tmp_path: Path) -> None:
    prior, variable, config = _load(PRIOR), _load(VARIABLE), _load(CONFIG)
    clock = iter((0.0, 240.0))
    wall = run_continuous_tc2_service(
        prior,
        variable,
        config,
        tmp_path / "wall",
        executor=lambda p, v, c, m: _fake_result(p, v, c, m),
        monotonic=lambda: next(clock),
    )
    assert wall["chunks_advanced"] == 0 and wall["reason"] == "wall_time_limit"

    small = dict(config)
    small["max_artifact_bytes"] = 1024
    with pytest.raises(QuarticTC2ContinuousServiceError, match="artifact byte limit"):
        run_continuous_tc2_service(
            prior,
            variable,
            small,
            tmp_path / "disk",
            executor=lambda p, v, c, m: _fake_result(p, v, c, m, padding=2048),
            monotonic=lambda: 0.0,
        )
    assert not (tmp_path / "disk" / "checkpoint.json").exists()

    promoted = dict(config)
    promoted["global_H7_policy"] = "pass"
    with pytest.raises(QuarticTC2ContinuousServiceError, match="initial service contract"):
        run_continuous_tc2_service(
            prior, variable, promoted, tmp_path / "promoted", monotonic=lambda: 0.0
        )

    clean = tmp_path / "tamper"
    run_continuous_tc2_service(
        prior,
        variable,
        config,
        clean,
        executor=lambda p, v, c, m: _fake_result(p, v, c, m),
        monotonic=lambda: 0.0,
    )
    checkpoint = _load(clean / "checkpoint.json")
    checkpoint["next_offset"] += 64
    (clean / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(QuarticTC2ContinuousServiceError, match="checkpoint contract"):
        run_continuous_tc2_service(
            prior, variable, config, clean, monotonic=lambda: 0.0
        )


def test_real_offset_704_checkpoint_is_exact_and_fail_closed() -> None:
    checkpoint = _load(REAL_SERVICE / "checkpoint.json")
    artifact = _load(REAL_SERVICE / "chunks" / "offset-000704.json")
    assert _checkpoint_hash_matches(checkpoint)
    assert _content_hash_matches(artifact)
    assert checkpoint["next_offset"] == 768
    assert checkpoint["prior_resume_sha256"] == (
        "b54cf0318aea170f9f71a64f02d9777cce1a805877577b4a96ee94823fb7559e"
    )
    assert not checkpoint["permanently_stopped"]
    assert not any(checkpoint["claims"].values())
    assert artifact["status"] == (
        "pass_cumulative_768_second_atom_pairs_no_obstruction_remaining_fail_closed"
    )
    assert artifact["first_exact_obstruction"] is None
    assert artifact["counts"] == {
        "total_unordered_coordinate_atom_pairs": 11781,
        "prior_cumulative_evaluated_coordinate_atom_pairs": 704,
        "current_evaluated_coordinate_atom_pairs": 64,
        "cumulative_evaluated_coordinate_atom_pairs": 768,
        "remaining_unevaluated_coordinate_atom_pairs": 11013,
        "candidates": 12,
        "current_evaluated_candidate_pairs": 768,
        "current_solvable_candidate_pairs": 768,
        "current_obstructed_candidate_pairs": 0,
        "cumulative_deltaK_AB_constructions": 9216,
        "TC2_closures": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
    }
    assert artifact["exact_tensor_summary_current_chunk"] == {
        "unique_direction_pair_packets": 57,
        "nonzero_D2P55_packets": 40,
        "nonzero_D2K55_packets": 46,
        "nonzero_D2TC2_packets": 0,
        "nonzero_deltaK_AB_packets": 41,
        "maximum_deltaK_AB_rank": 6,
        "all_deltaK_AB_Hermitian": True,
        "all_second_Sylvester_residuals_zero": True,
    }
    assert artifact["pair_manifest"][0]["selector_pair_index"] == 704
    assert artifact["pair_manifest"][-1]["selector_pair_index"] == 767
    assert all(
        candidate["solvable"]
        and candidate["Hermitian"]
        and candidate["second_Sylvester_residual_zero"]
        for record in artifact["pair_manifest"]
        for candidate in record["candidate_results"]
    )
