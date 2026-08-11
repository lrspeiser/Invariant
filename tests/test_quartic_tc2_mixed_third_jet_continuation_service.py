import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_mixed_third_jet_chunk_campaign import (
    DEFAULT_CHUNK_SIZE,
    TOTAL_MIXED_TRIPLES,
    _content_hash,
    _mixed_selector,
    _triple_kind,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_continuation_service import (
    QuarticTC2MixedThirdJetContinuationServiceError,
    _all_global_claims_false,
    _chunk_config,
    _hash_matches,
    _json_bytes,
    _load_state,
    _validate_result,
    _with_hash,
    run_mixed_third_jet_continuation_service,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
INITIAL = RUNS / "quartic-tc2-mixed-third-jet-chunk-campaign" / "campaign.json"
INITIAL_CONFIG = (
    ROOT / "configs" / "backgrounds" / "quartic_tc2_mixed_third_jet_chunk_campaign.json"
)
DIAGONAL = RUNS / "quartic-tc2-diagonal-third-jet-campaign" / "campaign.json"
QUADRATIC = RUNS / "quartic-tc2-quadratic-deltak-extension-campaign" / "campaign.json"
SERVICE_CONFIG = (
    ROOT / "configs" / "backgrounds" / "quartic_tc2_mixed_third_jet_continuation_service.json"
)
REAL_SERVICE = RUNS / "quartic-tc2-mixed-third-jet-continuation-service"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict]:
    return _load(DIAGONAL), _load(QUADRATIC), _load(SERVICE_CONFIG)


def _fake_result(
    chunk_config: dict,
    *,
    obstruct: bool = False,
    padding: int = 0,
) -> dict:
    offset = int(chunk_config["chunk_offset"])
    requested = int(chunk_config["chunk_size"])
    processed = 1 if obstruct else requested
    closed = 0 if obstruct else processed
    selector = _mixed_selector()[offset : offset + processed]
    seed = _content_hash(
        {
            "upstream": chunk_config["expected_upstream_content_sha256"],
            "canonical_D2_artifact_sequence_sha256": (
                "cc84bba0a54839b0228b07162a921ffdc1a893b6ff3de51d61dd0320ec95796d"
            ),
            "selector": "lexicographic_active_direction_multisets_excluding_AAA",
            "chunk_offset": offset,
            "prior_resume_sha256": chunk_config["expected_prior_resume_sha256"],
        }
    )
    previous = seed
    manifest = []
    for index, triple in enumerate(selector):
        record_body = {
            "chunk_index": index,
            "selector_index": offset + index,
            "active_position_triple": list(triple),
            "triple_kind": _triple_kind(triple),
            "previous_record_sha256": previous,
        }
        record = {**record_body, "record_sha256": _content_hash(record_body)}
        manifest.append(record)
        previous = record["record_sha256"]
    kinds = Counter(_triple_kind(triple) for triple in selector)
    partial = requested != DEFAULT_CHUNK_SIZE
    body = {
        "schema_version": "fake-executor-result",
        "status": (
            "stop_first_exact_mixed_third_jet_obstruction"
            if obstruct
            else (
                f"pass_mixed_third_jet_exact_partial_tail_{requested}_global_closure_fail_closed"
                if partial
                else "pass_mixed_third_jet_chunk_64_global_closure_fail_closed"
            )
        ),
        "errors": [],
        "upstream_sha256": chunk_config["expected_upstream_content_sha256"],
        "canonical_D2_artifact_sequence_sha256": (
            "cc84bba0a54839b0228b07162a921ffdc1a893b6ff3de51d61dd0320ec95796d"
        ),
        "config_sha256": _content_hash(chunk_config),
        "chunk_contract": {
            "selector": "lexicographic_active_direction_multisets_excluding_AAA",
            "global_mixed_triple_count": TOTAL_MIXED_TRIPLES,
            "chunk_offset": offset,
            "requested_chunk_size": requested,
            "processed_count": processed,
            "next_offset": offset + processed,
            "stop_on_first_obstruction": True,
            "stopped_early": obstruct,
            "resume_policy": "record_sha256_chain",
            "prior_resume_sha256": chunk_config["expected_prior_resume_sha256"],
            "resume_seed_sha256": seed,
            "resume_tip_sha256": previous,
            **({"exact_final_partial_tail": True} if partial else {}),
        },
        **(
            {
                "partial_tail_control": {
                    "selector_total": TOTAL_MIXED_TRIPLES,
                    "tail_offset": offset,
                    "tail_size": requested,
                    "tail_exhausts_selector_exactly": True,
                    "padded_or_inferred_triples": 0,
                    "passed": True,
                }
            }
            if partial
            else {}
        ),
        "counts": {
            "selected": processed,
            "triple_kind_counts": dict(sorted(kinds.items())),
            "symbolic_parameter_compatible": closed,
            "candidate_evaluations": processed * 12,
            "candidate_solvable": processed * 12 - (1 if obstruct else 0),
            "candidate_obstructed": 1 if obstruct else 0,
            "mixed_triples_remaining": TOTAL_MIXED_TRIPLES - offset - closed,
            "full_tube_Sylvester_identities": 0,
            "TC2_closures": 0,
            "B7_closures": 0,
            "global_H7_closures": 0,
            "lifespans_proved": 0,
        },
        "first_exact_obstruction": (
            {"selector_index": offset, "record_sha256": previous} if obstruct else None
        ),
        "triple_manifest": manifest,
        "closure_ledger": {
            "processed_mixed_third_jets_closed": closed,
            "all_12_300_mixed_third_jets_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "padding": "x" * padding,
    }
    return {**body, "content_sha256": _content_hash(body)}


def _run(tmp_path: Path, executor) -> dict:
    diagonal, quadratic, config = _inputs()
    return run_mixed_third_jet_continuation_service(
        INITIAL,
        INITIAL_CONFIG,
        diagonal,
        quadratic,
        [],
        config,
        tmp_path,
        executor=executor,
        monotonic=lambda: 0.0,
    )


def test_service_advances_exactly_one_chunk_and_persists_hash_chain(
    tmp_path: Path,
) -> None:
    calls = []

    def executor(_diagonal: dict, _quadratic: dict, _canonical: list, config: dict) -> dict:
        calls.append((config["chunk_offset"], config["chunk_size"]))
        return _fake_result(config)

    result = _run(tmp_path, executor)
    assert result["status"] == "checkpointed"
    assert result["reason"] == "chunk_limit"
    assert result["chunks_advanced"] == 1
    assert result["next_offset"] == 128
    assert calls == [(64, 64)]
    checkpoint = _load(tmp_path / "checkpoint.json")
    status = _load(tmp_path / "service-status.json")
    artifact = _load(tmp_path / "chunks" / "offset-000064.json")
    assert _hash_matches(checkpoint) and _hash_matches(status)
    assert (
        checkpoint["history"][0]["resume_tip_sha256"]
        == artifact["chunk_contract"]["resume_tip_sha256"]
    )
    assert checkpoint["remaining_mixed_triples"] == 12_172
    assert status["checkpoint_content_sha256"] == checkpoint["content_sha256"]
    assert not any(checkpoint["claims"].values())
    assert _all_global_claims_false(artifact["closure_ledger"])


def test_valid_orphan_is_adopted_without_reexecution(tmp_path: Path) -> None:
    _diagonal, _quadratic, config = _inputs()
    dynamic = _chunk_config(config, config["initial_prior_resume_tip_sha256"], 64, 64)
    artifact = tmp_path / "chunks" / "offset-000064.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(_json_bytes(_fake_result(dynamic)))

    def should_not_run(*_args: object) -> dict:
        raise AssertionError("a valid orphan must be adopted")

    recovered = _run(tmp_path, should_not_run)
    assert recovered["chunks_advanced"] == 1
    assert recovered["next_offset"] == 128


def test_obstruction_is_permanent_and_preserves_unclosed_remaining(
    tmp_path: Path,
) -> None:
    calls = 0

    def executor(_diagonal: dict, _quadratic: dict, _canonical: list, config: dict) -> dict:
        nonlocal calls
        calls += 1
        return _fake_result(config, obstruct=True)

    stopped = _run(tmp_path, executor)
    assert stopped["status"] == "stopped"
    assert stopped["reason"] == "exact_obstruction"
    assert stopped["next_offset"] == 65
    assert stopped["checkpoint"]["remaining_mixed_triples"] == 12_236
    again = _run(tmp_path, executor)
    assert again["status"] == "already_stopped"
    assert again["chunks_advanced"] == 0
    assert calls == 1


def test_exact_partial_tail_and_budget_tamper_controls(tmp_path: Path) -> None:
    diagonal, quadratic, config = _inputs()
    dynamic = _chunk_config(config, "prior-tip", 12_288, 12)
    tail = _fake_result(dynamic)
    _validate_result(
        tail,
        dynamic,
        12_288,
        12,
        config["canonical_D2_artifact_sequence_sha256"],
    )
    assert tail["chunk_contract"]["next_offset"] == TOTAL_MIXED_TRIPLES
    assert tail["partial_tail_control"]["padded_or_inferred_triples"] == 0

    wrong = _with_hash(
        {**{k: v for k, v in config.items() if k != "content_sha256"}, "TC2_policy": "pass"}
    )
    with pytest.raises(
        QuarticTC2MixedThirdJetContinuationServiceError,
        match="unsupported service contract",
    ):
        run_mixed_third_jet_continuation_service(
            INITIAL,
            INITIAL_CONFIG,
            diagonal,
            quadratic,
            [],
            wrong,
            tmp_path / "promotion",
            monotonic=lambda: 0.0,
        )

    tiny = _with_hash(
        {
            **{k: v for k, v in config.items() if k != "content_sha256"},
            "max_artifact_bytes": 1024,
        }
    )
    with pytest.raises(
        QuarticTC2MixedThirdJetContinuationServiceError,
        match="artifact byte budget exceeded",
    ):
        run_mixed_third_jet_continuation_service(
            INITIAL,
            INITIAL_CONFIG,
            diagonal,
            quadratic,
            [],
            tiny,
            tmp_path / "disk",
            executor=lambda _d, _q, _c, dynamic: _fake_result(dynamic, padding=2048),
            monotonic=lambda: 0.0,
        )
    assert not (tmp_path / "disk" / "checkpoint.json").exists()


def test_wall_limit_and_checkpoint_tamper_are_fail_closed(tmp_path: Path) -> None:
    diagonal, quadratic, config = _inputs()
    clock = iter((0.0, 901.0))
    wall = run_mixed_third_jet_continuation_service(
        INITIAL,
        INITIAL_CONFIG,
        diagonal,
        quadratic,
        [],
        config,
        tmp_path / "wall",
        executor=lambda _d, _q, _c, dynamic: _fake_result(dynamic),
        monotonic=lambda: next(clock),
    )
    assert wall["chunks_advanced"] == 0
    assert wall["reason"] == "wall_time_limit"

    clean = tmp_path / "tamper"
    _run(clean, lambda _d, _q, _c, dynamic: _fake_result(dynamic))
    checkpoint_path = clean / "checkpoint.json"
    checkpoint = _load(checkpoint_path)
    checkpoint["next_offset"] += 64
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(
        QuarticTC2MixedThirdJetContinuationServiceError,
        match="checkpoint contract mismatch",
    ):
        _run(clean, lambda _d, _q, _c, dynamic: _fake_result(dynamic))


def test_real_offset_64_service_artifact_is_exact_and_fail_closed() -> None:
    _diagonal, _quadratic, config = _inputs()
    checkpoint = _load(REAL_SERVICE / "checkpoint.json")
    status = _load(REAL_SERVICE / "service-status.json")
    artifact = _load(REAL_SERVICE / "chunks" / "offset-000064.json")
    dynamic = _chunk_config(config, config["initial_prior_resume_tip_sha256"], 64, 64)
    _validate_result(
        artifact,
        dynamic,
        64,
        64,
        config["canonical_D2_artifact_sequence_sha256"],
    )
    assert _load_state(REAL_SERVICE / "checkpoint.json", config) == checkpoint
    assert _hash_matches(checkpoint) and _hash_matches(status)
    assert artifact["chunk_contract"]["chunk_offset"] == 64
    assert artifact["chunk_contract"]["processed_count"] == 64
    assert artifact["chunk_contract"]["next_offset"] == 128
    assert artifact["counts"]["mixed_triples_remaining"] == 12_172
    assert artifact["first_exact_obstruction"] is None
    assert checkpoint["next_offset"] == 128
    assert checkpoint["remaining_mixed_triples"] == 12_172
    assert not any(checkpoint["claims"].values())
    assert _all_global_claims_false(artifact["closure_ledger"])
