import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_continuous_service import (
    _checkpoint_hash_matches,
    _json_bytes,
)
from sigma_theory_compiler.quartic_tc2_obligation_continuous_service import (
    QuarticTC2ObligationContinuousServiceError,
    _metadata,
    _validate_record_chain,
    _validate_result,
    run_obligation_continuous_service,
)
from sigma_theory_compiler.quartic_tc2_variable_sylvester_campaign import (
    _content_hash,
    _content_hash_matches,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
INITIAL = RUNS / "quartic-tc2-excluded-obligation-chunk0-campaign" / "campaign.json"
VARIABLE = RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"
CLASSIFICATION = (
    RUNS / "quartic-tc2-excluded-pair-classification-campaign" / "campaign.json"
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_obligation_continuous_service.json"
)
REAL_SERVICE = RUNS / "quartic-tc2-obligation-continuous-service"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_result(
    prior: dict,
    classification: dict,
    metadata: dict,
    *,
    obstruct: bool = False,
    padding: int = 0,
) -> dict:
    offset, requested = metadata["offset"], metadata["size"]
    evaluated = 1 if obstruct else requested
    seed = _content_hash({"fake_seed": offset, "prior": prior["content_sha256"]})
    previous = seed
    records = []
    candidates = [
        {
            "candidate_id": f"fake-{index}",
            "solvable": True,
            "Hermitian": True,
            "second_Sylvester_residual_zero": True,
        }
        for index in range(12)
    ]
    for local_index in range(evaluated):
        record_body = {
            "chunk_local_index": local_index,
            "obligation_selector_index": offset + local_index,
            "global_pair_index": 1000 + offset + local_index,
            "candidate_results": candidates,
            "previous_record_sha256": previous,
        }
        record_hash = _content_hash(record_body)
        records.append({**record_body, "record_sha256": record_hash})
        previous = record_hash
    body = {
        "schema_version": metadata["schema_version"],
        "status": (
            metadata["obstruction_status"]
            if obstruct
            else metadata["success_status"]
        ),
        "errors": [],
        "upstream_sha256": {
            "prior_artifact": prior["content_sha256"],
            "classification": classification["content_sha256"],
        },
        "chunk_contract": {
            "chunk_offset": offset,
            "requested_chunk_size": requested,
            "evaluated_chunk_size": evaluated,
            "chunk_seed_sha256": seed,
            "resume_after_record_sha256": previous,
        },
        "pair_manifest": records,
        "first_exact_obstruction": (
            {"obligation_selector_index": offset, "candidate_id": "fake-0"}
            if obstruct
            else None
        ),
        "counts": {
            "total_excluded_obligations": 2675,
            "current_evaluated_obligations": evaluated,
            "cumulative_evaluated_obligations": offset + evaluated,
            "remaining_unevaluated_obligations": 2675 - offset - evaluated,
            "TC2_closures": 0,
            "global_H7_closures": 0,
            "lifespans_proved": 0,
        },
        "padding": "x" * padding,
    }
    return {**body, "content_sha256": _content_hash(body)}


def test_service_checkpoints_and_resumes_without_duplicate_work(tmp_path: Path) -> None:
    initial, variable, classification, config = (
        _load(INITIAL),
        _load(VARIABLE),
        _load(CLASSIFICATION),
        _load(CONFIG),
    )
    calls: list[int] = []

    def executor(p: dict, _v: dict, c: dict, _cc: dict, m: dict) -> dict:
        calls.append(m["offset"])
        return _fake_result(p, c, m)

    first = run_obligation_continuous_service(
        initial,
        variable,
        classification,
        config,
        tmp_path,
        executor=executor,
        monotonic=lambda: 0.0,
    )
    assert first["chunks_advanced"] == 1 and first["next_offset"] == 128
    second = run_obligation_continuous_service(
        initial,
        variable,
        classification,
        config,
        tmp_path,
        executor=executor,
        monotonic=lambda: 0.0,
    )
    assert second["chunks_advanced"] == 1 and second["next_offset"] == 192
    assert second["checkpoint"]["completed_service_chunks"] == 2
    assert calls == [64, 128]
    assert not any(second["checkpoint"]["claims"].values())


def test_valid_orphan_is_recovered_without_executor(tmp_path: Path) -> None:
    initial, variable, classification, config = (
        _load(INITIAL),
        _load(VARIABLE),
        _load(CLASSIFICATION),
        _load(CONFIG),
    )
    metadata = _metadata(64, 64)
    orphan = _fake_result(initial, classification, metadata)
    path = tmp_path / "chunks" / "offset-000064.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(_json_bytes(orphan))

    def should_not_run(*_args: object) -> dict:
        raise AssertionError("valid orphan chunk must not be reexecuted")

    result = run_obligation_continuous_service(
        initial,
        variable,
        classification,
        config,
        tmp_path,
        executor=should_not_run,
        monotonic=lambda: 0.0,
    )
    assert result["next_offset"] == 128
    assert (tmp_path / "checkpoint.json").is_file()


def test_obstruction_is_permanent_and_reentry_is_noop(tmp_path: Path) -> None:
    initial, variable, classification, config = (
        _load(INITIAL),
        _load(VARIABLE),
        _load(CLASSIFICATION),
        _load(CONFIG),
    )
    calls = 0

    def executor(p: dict, _v: dict, c: dict, _cc: dict, m: dict) -> dict:
        nonlocal calls
        calls += 1
        return _fake_result(p, c, m, obstruct=True)

    stopped = run_obligation_continuous_service(
        initial,
        variable,
        classification,
        config,
        tmp_path,
        executor=executor,
        monotonic=lambda: 0.0,
    )
    assert stopped["status"] == "stopped"
    assert stopped["reason"] == "exact_obstruction"
    again = run_obligation_continuous_service(
        initial,
        variable,
        classification,
        config,
        tmp_path,
        executor=executor,
        monotonic=lambda: 0.0,
    )
    assert again["status"] == "already_stopped"
    assert again["chunks_advanced"] == 0 and calls == 1


def test_wall_disk_checkpoint_tamper_and_false_promotion_reject(tmp_path: Path) -> None:
    initial, variable, classification, config = (
        _load(INITIAL),
        _load(VARIABLE),
        _load(CLASSIFICATION),
        _load(CONFIG),
    )
    clock = iter((0.0, 240.0))
    wall = run_obligation_continuous_service(
        initial,
        variable,
        classification,
        config,
        tmp_path / "wall",
        executor=lambda p, _v, c, _cc, m: _fake_result(p, c, m),
        monotonic=lambda: next(clock),
    )
    assert wall["chunks_advanced"] == 0 and wall["reason"] == "wall_time_limit"

    small = dict(config)
    small["max_artifact_bytes"] = 1024
    with pytest.raises(
        QuarticTC2ObligationContinuousServiceError, match="artifact byte limit"
    ):
        run_obligation_continuous_service(
            initial,
            variable,
            classification,
            small,
            tmp_path / "disk",
            executor=lambda p, _v, c, _cc, m: _fake_result(
                p, c, m, padding=2048
            ),
            monotonic=lambda: 0.0,
        )

    promoted = dict(config)
    promoted["global_H7_policy"] = "pass"
    with pytest.raises(
        QuarticTC2ObligationContinuousServiceError,
        match="initial obligation service contract",
    ):
        run_obligation_continuous_service(
            initial,
            variable,
            classification,
            promoted,
            tmp_path / "promoted",
            monotonic=lambda: 0.0,
        )

    clean = tmp_path / "tamper"
    run_obligation_continuous_service(
        initial,
        variable,
        classification,
        config,
        clean,
        executor=lambda p, _v, c, _cc, m: _fake_result(p, c, m),
        monotonic=lambda: 0.0,
    )
    checkpoint = _load(clean / "checkpoint.json")
    checkpoint["next_offset"] += 64
    (clean / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(
        QuarticTC2ObligationContinuousServiceError,
        match="obligation checkpoint contract mismatch",
    ):
        run_obligation_continuous_service(
            initial,
            variable,
            classification,
            config,
            clean,
            monotonic=lambda: 0.0,
        )


def test_exact_final_partial_tail_contract_is_supported() -> None:
    initial, classification = _load(INITIAL), _load(CLASSIFICATION)
    metadata = _metadata(2624, 51)
    result = _fake_result(initial, classification, metadata)
    _validate_result(result, initial, classification, metadata)
    assert metadata["success_status"] == (
        "pass_cumulative_2675_excluded_obligations_no_obstruction_remaining_fail_closed"
    )
    assert result["counts"]["remaining_unevaluated_obligations"] == 0
    assert _validate_record_chain(result)


def test_real_offset64_checkpoint_is_exact_and_fail_closed() -> None:
    if not (REAL_SERVICE / "checkpoint.json").exists():
        pytest.skip("real bounded chunk has not been materialized yet")
    checkpoint = _load(REAL_SERVICE / "checkpoint.json")
    artifact = _load(REAL_SERVICE / "chunks" / "offset-000064.json")
    assert _checkpoint_hash_matches(checkpoint)
    assert _content_hash_matches(artifact)
    assert [item["offset"] for item in checkpoint["history"]][:3] == [64, 128, 192]
    assert checkpoint["next_offset"] == (
        checkpoint["history"][-1]["offset"] + checkpoint["history"][-1]["size"]
    )
    assert not checkpoint["permanently_stopped"]
    assert not any(checkpoint["claims"].values())
    assert artifact["status"] == (
        "pass_cumulative_128_excluded_obligations_no_obstruction_remaining_fail_closed"
    )
    assert artifact["first_exact_obstruction"] is None
    assert artifact["counts"]["current_evaluated_obligations"] == 64
    assert artifact["counts"]["remaining_unevaluated_obligations"] == 2547
    assert _validate_record_chain(artifact)
    assert all(
        candidate["solvable"]
        and candidate["Hermitian"]
        and candidate["second_Sylvester_residual_zero"]
        for record in artifact["pair_manifest"]
        for candidate in record["candidate_results"]
    )
