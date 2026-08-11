import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash
from sigma_theory_compiler.quartic_tc2_fourth_jet_obligation_service import (
    CHUNK_SCHEMA,
    DEFAULT_CHUNK_SIZE,
    FINAL_TAIL_SIZE,
    TOTAL_OBLIGATIONS,
    _chunk_config,
    _chunk_seed,
    _global_claims,
    _initial_resume_sha256,
    _validate_chunk_result,
    run_fourth_jet_obligation_service,
)
from sigma_theory_compiler.quartic_tc2_fourth_jet_parallel_kernel import (
    QuarticTC2FourthJetParallelKernelError,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_continuation_service import (
    _with_hash,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_fourth_jet_obligation_service.json"
)
OUTPUT = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-fourth-jet-obligation-service"
)


def _inputs() -> tuple[dict, dict, dict, dict]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    campaign = json.loads((ROOT / config["fourth_campaign"]["path"]).read_text())
    predecessor = json.loads(
        (ROOT / config["third_jet_predecessor"]["path"]).read_text()
    )
    candidates = json.loads((ROOT / config["candidate_source"]["path"]).read_text())
    return config, campaign, predecessor, candidates


def _synthetic_result(
    campaign: dict,
    candidates: dict,
    chunk_config: dict,
    *,
    obstruction_index: int | None = None,
) -> dict:
    offset = chunk_config["obligation_offset"]
    size = chunk_config["requested_size"]
    selected = campaign["selector"]["records"][offset : offset + size]
    previous = _chunk_seed(
        campaign, offset, chunk_config["expected_prior_resume_sha256"]
    )
    manifest = []
    first_obstruction = None
    for local_index, selector_record in enumerate(selected):
        obstructed = local_index == obstruction_index
        candidate_results = [
            {
                "candidate_id": certificate["candidate_id"],
                "solvable": not (obstructed and candidate_index == 0),
            }
            for candidate_index, certificate in enumerate(candidates["certificates"])
        ]
        obstructed_ids = (
            [candidate_results[0]["candidate_id"]] if obstructed else []
        )
        body = {
            "obligation_offset": offset + local_index,
            "chunk_index": local_index,
            "selector_record_sha256": selector_record["record_sha256"],
            "active_indices": selector_record["active_indices"],
            "active_positions": selector_record["active_positions"],
            "multiplicity_partition": selector_record["multiplicity_partition"],
            "fourth_campaign_content_sha256": campaign["content_sha256"],
            "basis_positions": selector_record["active_positions"],
            "directional_evaluations": 1,
            "symbolic_parameter_compatible": not obstructed,
            "candidate_results": candidate_results,
            "obstructed_candidate_ids": obstructed_ids,
            "previous_record_sha256": previous,
        }
        record = {**body, "record_sha256": _content_hash(body)}
        manifest.append(record)
        previous = record["record_sha256"]
        if obstructed:
            first_obstruction = {
                "obligation_offset": offset + local_index,
                "selector_record_sha256": selector_record["record_sha256"],
                "record_sha256": record["record_sha256"],
                "active_indices": selector_record["active_indices"],
                "active_positions": selector_record["active_positions"],
                "obstructed_candidate_ids": obstructed_ids,
                "gate": "fourth-order equal-eigenspace Sylvester compatibility",
            }
            break
    processed = len(manifest)
    passed = processed - int(first_obstruction is not None)
    partial = size != DEFAULT_CHUNK_SIZE
    status = (
        "stop_first_exact_fourth_jet_obstruction"
        if first_obstruction
        else (
            f"pass_exact_fourth_jet_final_tail_{size}_tube_fail_closed"
            if partial
            else f"pass_exact_fourth_jet_chunk_{DEFAULT_CHUNK_SIZE}_tube_fail_closed"
        )
    )
    body = {
        "schema_version": CHUNK_SCHEMA,
        "status": status,
        "config_sha256": _content_hash(chunk_config),
        "upstream_sha256": {
            "fourth_campaign": campaign["content_sha256"],
            "candidate_source": candidates["content_sha256"],
        },
        "chunk_contract": {
            "selector": chunk_config["selector"],
            "global_obligation_count": TOTAL_OBLIGATIONS,
            "obligation_offset": offset,
            "requested_chunk_size": size,
            "processed_count": processed,
            "next_obligation_offset": offset + processed,
            "parallel_worker_count": 8,
            "parallel_execution_policy": chunk_config["parallel_execution_policy"],
            "bounded_speculative_evaluations_may_finish_after_first_obstruction": True,
            "records_after_first_obstruction_committed_or_inferred": 0,
            "stop_on_first_obstruction": True,
            "stopped_early": first_obstruction is not None,
            "resume_policy": "record_sha256_chain",
            "prior_resume_sha256": chunk_config["expected_prior_resume_sha256"],
            "resume_seed_sha256": _chunk_seed(
                campaign, offset, chunk_config["expected_prior_resume_sha256"]
            ),
            "resume_tip_sha256": previous,
            **({"exact_final_partial_tail": True} if partial else {}),
        },
        "counts": {
            "selected": processed,
            "partition_counts": {},
            "directional_evaluations": processed,
            "symbolic_parameter_compatible": passed,
            "candidate_evaluations": processed * 12,
            "candidate_solvable": processed * 12 - int(first_obstruction is not None),
            "candidate_obstructed": int(first_obstruction is not None),
            "fourth_obligations_remaining": TOTAL_OBLIGATIONS - offset - passed,
            "fourth_obligations_inferred_passed": 0,
        },
        "first_exact_obstruction": first_obstruction,
        "obligation_manifest": manifest,
        "closure_ledger": {
            "processed_fourth_obligations_closed": passed,
            "all_3060_fourth_obligations_closed": (
                first_obstruction is None and offset + passed == TOTAL_OBLIGATIONS
            ),
            **_global_claims(),
        },
        "scope": "synthetic exact service contract test",
        "errors": [],
    }
    return _with_hash(body)


def test_synthetic_chunk_checkpoint_and_global_fail_closed(tmp_path: Path) -> None:
    _, campaign, predecessor, _ = _inputs()

    def executor(campaign_arg: dict, candidates_arg: dict, chunk_arg: dict) -> dict:
        return _synthetic_result(campaign_arg, candidates_arg, chunk_arg)

    result = run_fourth_jet_obligation_service(
        ROOT, CONFIG_PATH, tmp_path / "service", executor=executor
    )
    checkpoint = result["checkpoint"]
    assert result["next_obligation_offset"] == 32
    assert result["remaining_obligations"] == 3_028
    assert checkpoint["completed_chunks"] == 1
    assert not any(checkpoint["claims"].values())
    assert checkpoint["history"][0]["closed_count"] == 32
    assert checkpoint["prior_resume_sha256"] != _initial_resume_sha256(
        campaign, predecessor
    )


def test_valid_orphan_is_adopted_without_executor_reentry(tmp_path: Path) -> None:
    source = tmp_path / "source"

    def executor(campaign_arg: dict, candidates_arg: dict, chunk_arg: dict) -> dict:
        return _synthetic_result(campaign_arg, candidates_arg, chunk_arg)

    run_fourth_jet_obligation_service(ROOT, CONFIG_PATH, source, executor=executor)
    orphan = tmp_path / "orphan"
    (orphan / "chunks").mkdir(parents=True)
    shutil.copy2(
        source / "chunks" / "obligation-offset-000000.json",
        orphan / "chunks" / "obligation-offset-000000.json",
    )

    def forbidden(*args: object) -> dict:
        raise AssertionError("valid orphan must not reenter executor")

    result = run_fourth_jet_obligation_service(
        ROOT, CONFIG_PATH, orphan, executor=forbidden
    )
    assert result["orphan_adopted"] is True
    assert result["next_obligation_offset"] == 32


def test_first_obstruction_is_permanent_and_has_no_post_records(tmp_path: Path) -> None:
    def executor(campaign_arg: dict, candidates_arg: dict, chunk_arg: dict) -> dict:
        return _synthetic_result(
            campaign_arg, candidates_arg, chunk_arg, obstruction_index=2
        )

    output = tmp_path / "stopped"
    result = run_fourth_jet_obligation_service(
        ROOT, CONFIG_PATH, output, executor=executor
    )
    assert result["status"] == "stopped"
    assert result["next_obligation_offset"] == 3
    artifact = json.loads(
        (output / "chunks" / "obligation-offset-000000.json").read_text()
    )
    assert len(artifact["obligation_manifest"]) == 3
    assert artifact["chunk_contract"][
        "records_after_first_obstruction_committed_or_inferred"
    ] == 0
    repeated = run_fourth_jet_obligation_service(
        ROOT, CONFIG_PATH, output, executor=executor
    )
    assert repeated["status"] == "already_stopped"
    assert repeated["chunks_advanced"] == 0


def test_infrastructure_failure_commits_no_lifecycle_state(tmp_path: Path) -> None:
    def failed_executor(*args: object) -> dict:
        raise QuarticTC2FourthJetParallelKernelError(
            "directional recurrence failed in mandatory orders one through three"
        )

    output = tmp_path / "infrastructure-failure"
    with pytest.raises(
        QuarticTC2FourthJetParallelKernelError,
        match="mandatory orders one through three",
    ):
        run_fourth_jet_obligation_service(
            ROOT, CONFIG_PATH, output, executor=failed_executor
        )
    assert not (output / "checkpoint.json").exists()
    assert not (output / "service-status.json").exists()
    assert not (output / "chunks" / "obligation-offset-000000.json").exists()


def test_exact_final_tail_is_twenty_without_padding() -> None:
    config, campaign, _, candidates = _inputs()
    dynamic = _chunk_config(config, campaign, 3_040, FINAL_TAIL_SIZE, "f" * 64)
    result = _synthetic_result(campaign, candidates, dynamic)
    _validate_chunk_result(result, dynamic, campaign, candidates)
    assert result["chunk_contract"]["exact_final_partial_tail"] is True
    assert result["chunk_contract"]["next_obligation_offset"] == 3_060
    assert result["counts"]["fourth_obligations_remaining"] == 0


def test_two_committed_exact_chunks_and_checkpoint_are_valid() -> None:
    config, campaign, predecessor, candidates = _inputs()
    artifact = json.loads(
        (OUTPUT / "chunks" / "obligation-offset-000000.json").read_text()
    )
    checkpoint = json.loads((OUTPUT / "checkpoint.json").read_text())
    status = json.loads((OUTPUT / "service-status.json").read_text())
    initial_resume = _initial_resume_sha256(campaign, predecessor)
    dynamic = _chunk_config(config, campaign, 0, 32, initial_resume)
    _validate_chunk_result(artifact, dynamic, campaign, candidates)
    assert artifact["content_sha256"] == (
        "c0a56d249afd246387477d1dca3b41f3133ad5e407a4e6c38763c477f119c55b"
    )
    assert artifact["counts"] == {
        "candidate_evaluations": 384,
        "candidate_obstructed": 0,
        "candidate_solvable": 384,
        "directional_evaluations": 283,
        "fourth_obligations_inferred_passed": 0,
        "fourth_obligations_remaining": 3_028,
        "partition_counts": {"AAAA": 1, "AAAB": 14, "AABB": 2, "AABC": 15},
        "selected": 32,
        "symbolic_parameter_compatible": 32,
    }
    second = json.loads(
        (OUTPUT / "chunks" / "obligation-offset-000032.json").read_text()
    )
    second_dynamic = _chunk_config(
        config,
        campaign,
        32,
        32,
        artifact["chunk_contract"]["resume_tip_sha256"],
    )
    _validate_chunk_result(second, second_dynamic, campaign, candidates)
    assert second["content_sha256"] == (
        "b6e03be15c3bfdceb69e71918b12eae0d117009ad43617830f6364ff886b4ee1"
    )
    assert second["counts"] == {
        "candidate_evaluations": 384,
        "candidate_obstructed": 0,
        "candidate_solvable": 384,
        "directional_evaluations": 346,
        "fourth_obligations_inferred_passed": 0,
        "fourth_obligations_remaining": 2_996,
        "partition_counts": {"AABB": 2, "AABC": 30},
        "selected": 32,
        "symbolic_parameter_compatible": 32,
    }
    assert checkpoint["completed_chunks"] == 2
    assert checkpoint["next_obligation_offset"] == 64
    assert checkpoint["remaining_obligations"] == 2_996
    assert checkpoint["prior_resume_sha256"] == (
        "82261f7d53a122efd79ce799539a018aca1db523a76f77c6e761d978c39d8da5"
    )
    assert checkpoint["permanently_stopped"] is False
    assert not any(checkpoint["claims"].values())
    assert status["checkpoint_content_sha256"] == checkpoint["content_sha256"]
