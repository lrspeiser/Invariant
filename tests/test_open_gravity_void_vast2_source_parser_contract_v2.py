from __future__ import annotations

import inspect

import pytest

from sigma_theory_compiler import open_gravity_void_vast2_source_parser_contract_v1 as v1
from sigma_theory_compiler import open_gravity_void_vast2_source_parser_contract_v2 as contract


def test_v2_reuses_exact_frozen_parser_grammar() -> None:
    frame = v1._synthetic_frame(void=0)
    row = contract.parse_vast2_record(frame, source_index=0, framed_start=0)
    assert row["void"] == 0 and row["payload_bytes"] == 105
    assert contract.semantic_sphere_key(row) == v1.semantic_sphere_key(row)


def test_v1_failed_artifacts_are_preserved_and_bound() -> None:
    binding = contract.validate_frozen_v1_and_failure(contract.load_config())
    assert binding["v1_preserved"] is True
    assert binding["v1_failure_classification"] == "WINDOWS_TEXT_MODE_CRLF_EXPANSION_ARTIFACT_FREEZE_MISMATCH"
    failure = contract.load_config()["superseded_v1_build_failure"]
    assert failure["actual_ledger_bytes"] - failure["declared_ledger_bytes"] == 80080
    assert failure["source_or_parser_failure"] is False


def test_v2_package_hashes_and_claim_boundary_are_exact() -> None:
    receipt = contract.check_package()
    assert receipt["status"] == "PASS_VAST2_SOURCE_PARSER_CONTRACT_V2_AWAIT_DISTINCT_INDEPENDENT_AUDIT"
    assert receipt["decision"] == "BLOCK_LANE9_REAUTHORIZATION_PENDING_DISTINCT_VAST2_CONTRACT_V2_AUDIT"
    assert receipt["source_disposition"]["accepted_rows"] == 80080
    assert receipt["source_disposition"]["zero_identifier_sphere_rows"] == 263
    assert receipt["source_disposition"]["byte_duplicate_rows"] == 0
    assert receipt["source_disposition"]["semantic_duplicate_sphere_keys"] == 0
    assert receipt["claim_boundary"]["successor_independently_audited"] is False
    assert receipt["claim_boundary"]["self_review_claimed_independent"] is False
    assert receipt["claim_boundary"]["executor_run_or_authorized"] is False
    assert all(gate["passed"] for gate in receipt["conformance_gates"])


def test_v2_is_append_only_and_exposes_no_run_or_authorization_cli() -> None:
    source = inspect.getsource(contract.main)
    assert 'choices=("build", "check", "status")' in source
    for command in ("run-development", "score", "authorize", "reauthorize"):
        with pytest.raises(SystemExit):
            contract.main([command])
    with pytest.raises(contract.Vast2SourceParserContractV2Error, match="append-only v2 package already exists"):
        contract.build_package()
