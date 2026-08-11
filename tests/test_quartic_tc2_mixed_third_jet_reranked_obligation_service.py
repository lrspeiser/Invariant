import copy
import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_basis_reduction_campaign import (
    _load_bound_json,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_parallel_epoch import (
    _canonical_second_atom_artifacts,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_reranked_obligation_service import (
    QuarticTC2RerankedObligationServiceError,
    _chunk_config,
    _claims,
    _hash_matches,
    _initial_resume_sha256,
    _load_state,
    _scope,
    _validate_chunk_result,
    _validate_config,
    _with_hash,
    run_reranked_obligation_service,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_mixed_third_jet_reranked_obligation_service.json"
)
OUTPUT = (
    ROOT / "runs" / "physics-language" / "quartic-tc2-mixed-third-jet-reranked-obligation-service"
)


def _inputs() -> tuple[dict, dict, dict, dict, dict, dict, list[dict]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    diagonal = _load_bound_json(ROOT, config["diagonal_third_jet"])
    quadratic = _load_bound_json(ROOT, config["quadratic_deltaK"])
    reduction = _load_bound_json(ROOT, config["reranked_reduction"])
    predecessor = _load_bound_json(ROOT, config["stopped_predecessor_checkpoint"])
    supervisor = _load_bound_json(ROOT, config["stopped_supervisor_state"])
    canonical = _canonical_second_atom_artifacts(ROOT)
    return config, diagonal, quadratic, reduction, predecessor, supervisor, canonical


def test_service_config_binds_stopped_predecessor_and_selector_chain() -> None:
    config, diagonal, quadratic, reduction, predecessor, supervisor, canonical = _inputs()
    selector = _validate_config(
        config,
        diagonal,
        quadratic,
        reduction,
        predecessor,
        supervisor,
        canonical,
    )
    assert _hash_matches(config)
    assert len(selector) == 447
    assert selector[0]["global_selector_index"] == 1634
    assert selector[-1]["global_selector_index"] == 12_269
    assert _initial_resume_sha256(reduction) == (
        "1c979dd6037a9841ef6dc7d506ecb2b51d3b42bec18a5655d0c7bc6acbe7fb5b"
    )


def test_seven_exact_chunks_checkpoint_and_status_are_valid() -> None:
    config, _, _, reduction, _, _, _ = _inputs()
    artifacts = []
    prior_resume = _initial_resume_sha256(reduction)
    for offset, size in (
        (0, 64),
        (64, 64),
        (128, 64),
        (192, 64),
        (256, 64),
        (320, 64),
        (384, 63),
    ):
        artifact = json.loads(
            (OUTPUT / "chunks" / f"obligation-offset-{offset:06d}.json").read_text(encoding="utf-8")
        )
        dynamic = _chunk_config(config, reduction, offset, size, prior_resume)
        _validate_chunk_result(artifact, dynamic, reduction, config, offset, size)
        artifacts.append(artifact)
        prior_resume = artifact["chunk_contract"]["resume_tip_sha256"]
    checkpoint = json.loads((OUTPUT / "checkpoint.json").read_text(encoding="utf-8"))
    status = json.loads((OUTPUT / "service-status.json").read_text(encoding="utf-8"))
    assert _load_state(OUTPUT / "checkpoint.json", config, reduction) == checkpoint
    initial = artifacts[0]
    latest = artifacts[-1]
    assert initial["content_sha256"] == (
        "cc0881c1e06a7f5fa308be071d950f2ddd3f9239f6b522a87cc13cd9d4c94ea7"
    )
    assert latest["content_sha256"] == (
        "a543a2a98d3cdab9bcfbe2e742e3853922d32b87784051dc7be2f87769724853"
    )
    assert initial["status"] == "pass_reranked_obligation_chunk_64_fail_closed"
    assert initial["counts"] == {
        "B7_closures": 0,
        "TC2_closures": 0,
        "candidate_evaluations": 768,
        "candidate_obstructed": 0,
        "candidate_solvable": 768,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "remaining_active_triples_inferred_passed": 0,
        "reranked_obligations_remaining": 383,
        "selected": 64,
        "stable_mixed_prefix_records": 1600,
        "symbolic_parameter_compatible": 64,
        "triple_kind_counts": {"AAB": 12, "ABB": 8, "ABC": 44},
    }
    assert all(artifact["first_exact_obstruction"] is None for artifact in artifacts)
    assert initial["obligation_manifest"][0]["global_selector_index"] == 1634
    assert initial["obligation_manifest"][-1]["global_selector_index"] == 2032
    assert initial["chunk_contract"]["resume_tip_sha256"] == (
        "b6d7222f845bd57d31be37e2237f20112e3b2be9c74e29f150aea13153ce0738"
    )
    assert latest["chunk_contract"]["resume_tip_sha256"] == (
        "36338e1d76f61acbbab4927f7fd38cc116defb3a5d0ccd2a73d45faafe726e55"
    )
    assert latest["scope"] == _scope(full_mixed_sector_closed=True)
    assert latest["closure_ledger"]["all_447_reranked_obligations_closed"] is True
    assert latest["closure_ledger"]["all_12_300_mixed_third_jets_closed"] is True
    assert checkpoint["completed_chunks"] == 7
    assert checkpoint["next_obligation_offset"] == 447
    assert checkpoint["remaining_obligations"] == 0
    assert checkpoint["permanently_stopped"] is False
    assert checkpoint["claims"]["full_mixed_sector_closed"] is True
    assert sum(checkpoint["claims"].values()) == 1
    assert _hash_matches(checkpoint) and _hash_matches(status)
    assert status["checkpoint_content_sha256"] == checkpoint["content_sha256"]
    assert status["decision"] == "completed"
    assert status["reason"] == "reranked_selector_complete_full_tube_still_open"


def test_selector_record_tamper_is_rejected_after_rehash() -> None:
    config, _, _, reduction, _, _, _ = _inputs()
    artifact = json.loads(
        (OUTPUT / "chunks" / "obligation-offset-000000.json").read_text(encoding="utf-8")
    )
    tampered = copy.deepcopy(artifact)
    last = tampered["obligation_manifest"][-1]
    last["selector_obligation_sha256"] = "0" * 64
    last_body = {key: value for key, value in last.items() if key != "record_sha256"}
    last["record_sha256"] = _content_hash(last_body)
    tampered["chunk_contract"]["resume_tip_sha256"] = last["record_sha256"]
    tampered = _with_hash(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    dynamic = _chunk_config(config, reduction, 0, 64, _initial_resume_sha256(reduction))
    with pytest.raises(
        QuarticTC2RerankedObligationServiceError, match="result record-chain mismatch"
    ):
        _validate_chunk_result(tampered, dynamic, reduction, config, 0, 64)


def test_orphan_artifact_is_validated_and_adopted_without_reexecution(tmp_path: Path) -> None:
    orphan_output = tmp_path / "reranked-service"
    orphan_chunk = orphan_output / "chunks" / "obligation-offset-000000.json"
    orphan_chunk.parent.mkdir(parents=True)
    shutil.copy2(OUTPUT / "chunks" / "obligation-offset-000000.json", orphan_chunk)
    calls = 0

    def forbidden_executor(*args: object) -> dict:
        nonlocal calls
        calls += 1
        raise AssertionError("valid orphan must be adopted without executor reentry")

    result = run_reranked_obligation_service(
        ROOT,
        CONFIG_PATH,
        orphan_output,
        executor=forbidden_executor,
    )
    assert calls == 0
    assert result["status"] == "checkpointed"
    assert result["chunks_advanced"] == 1
    assert result["next_obligation_offset"] == 64
    assert result["remaining_obligations"] == 383
    assert (orphan_output / "checkpoint.json").is_file()
    assert (orphan_output / "service-status.json").is_file()


def test_checkpoint_claim_tamper_is_rejected_after_rehash(tmp_path: Path) -> None:
    config, _, _, reduction, _, _, _ = _inputs()
    checkpoint = json.loads((OUTPUT / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["claims"]["TC2_closed"] = True
    checkpoint = _with_hash(
        {key: value for key, value in checkpoint.items() if key != "content_sha256"}
    )
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(QuarticTC2RerankedObligationServiceError, match="checkpoint contract"):
        _load_state(path, config, reduction)


def test_final_tail_size_is_exactly_63_without_padding() -> None:
    config, _, _, reduction, _, _, _ = _inputs()
    dynamic = _chunk_config(config, reduction, 384, 63, "f" * 64)
    assert dynamic["obligation_offset"] == 384
    assert dynamic["requested_size"] == 63
    assert dynamic["obligation_offset"] + dynamic["requested_size"] == 447
    assert config["partial_tail_policy"] == (
        "evaluate_exact_remaining_obligation_entries_without_padding_or_inference"
    )
    completed_claims = _claims(full_mixed_sector_closed=True)
    assert completed_claims["full_mixed_sector_closed"] is True
    assert not any(
        value for key, value in completed_claims.items() if key != "full_mixed_sector_closed"
    )
