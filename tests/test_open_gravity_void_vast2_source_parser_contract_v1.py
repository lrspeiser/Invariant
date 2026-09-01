from __future__ import annotations

import inspect

import pytest

from sigma_theory_compiler import open_gravity_void_vast2_source_parser_contract_v1 as contract


def test_exact_zero_based_record_and_provenance_are_accepted() -> None:
    frame = contract._synthetic_frame(void=0)
    row = contract.parse_vast2_record(frame, source_index=7, framed_start=742)
    assert row["Cosmo"] == "Planck2018"
    assert row["void"] == 0
    assert row["framed_bytes"] == 106 and row["payload_bytes"] == 105
    assert row["source_index"] == 7 and row["line_number"] == 8
    assert row["framed_start"] == 742 and row["framed_end_exclusive"] == 848
    assert len(row["framed_raw_sha256"]) == len(row["payload_raw_sha256"]) == 64


def test_exact_grammar_rejects_mutated_separator_label_framing_and_domain() -> None:
    frame = contract._synthetic_frame(void=1)
    cases = (
        frame[:10] + b"X" + frame[11:],
        contract._synthetic_frame(cosmo=b"Other     ", void=1),
        frame[:-1] + b"\r\n",
        contract._synthetic_frame(void=9999),
        frame[:101] + b"    " + frame[105:],
        frame[:-2] + b"\n",
        frame[:-1] + b"X\n",
    )
    for bad in cases:
        with pytest.raises(contract.Vast2SourceParserContractError):
            contract.parse_vast2_record(bad, source_index=0, framed_start=0)


def test_group_repetition_is_valid_but_full_semantic_duplicate_rejects() -> None:
    first = contract.parse_vast2_record(contract._synthetic_frame(void=0), source_index=0, framed_start=0)
    second = dict(first)
    second["y"] = 1.0
    assert contract.validate_no_semantic_sphere_duplicates([first, second]) == 2
    with pytest.raises(contract.Vast2SourceParserContractError, match="duplicate VAST2 semantic sphere key"):
        contract.validate_no_semantic_sphere_duplicates([first, dict(first)])
    negative_zero = dict(first)
    negative_zero["y"] = -0.0
    with pytest.raises(contract.Vast2SourceParserContractError, match="duplicate VAST2 semantic sphere key"):
        contract.validate_no_semantic_sphere_duplicates([first, negative_zero])


def test_frozen_v5_delegate_is_demonstrably_inexact_without_running_executor() -> None:
    zero = contract._synthetic_frame(void=0)
    with pytest.raises(contract.frozen_v1.DevelopmentReleaseV1Error, match="invalid VAST table2 radius or void"):
        contract.frozen_v1.parse_vast_table2_record(zero)
    valid_one = contract._synthetic_frame(void=1)
    bad_separator = valid_one[:10] + b"X" + valid_one[11:]
    assert contract.frozen_v1.parse_vast_table2_record(bad_separator)["void"] == 1
    assert contract.frozen_v1.parse_vast_table2_record(contract._synthetic_frame(cosmo=b"Other     ", void=1))["Cosmo"] == "Other"
    assert contract.frozen_v1.parse_vast_table2_record(valid_one[:-1] + b"\r\n")["void"] == 1
    assert contract.frozen_v1.parse_vast_table2_record(contract._synthetic_frame(void=9999))["void"] == 9999


def test_paths_authority_and_cli_are_fail_closed() -> None:
    for value in ("../secret", "/absolute", "work\\private\\source", "./relative"):
        with pytest.raises(contract.Vast2SourceParserContractError):
            contract.canonical_file(value)
    authority = contract.load_config()["authority"]
    assert authority["scoring_authority"] is False
    assert authority["development_run_authority"] is False
    assert authority["reauthorization_authority"] is False
    assert authority["may_mint_or_consume_authorization"] is False
    source = inspect.getsource(contract.main)
    assert 'choices=("build", "check", "status")' in source
    with pytest.raises(SystemExit):
        contract.main(["run-development"])
    with pytest.raises(SystemExit):
        contract.main(["authorize"])


def test_frozen_successor_package_is_exact_and_explicitly_unaudited() -> None:
    receipt = contract.check_package()
    assert receipt["status"] == "PASS_VAST2_SOURCE_PARSER_CONTRACT_AWAIT_DISTINCT_INDEPENDENT_AUDIT"
    assert receipt["decision"] == "BLOCK_LANE9_REAUTHORIZATION_PENDING_DISTINCT_SUCCESSOR_AUDIT"
    assert receipt["source_disposition"]["accepted_rows"] == 80080
    assert receipt["source_disposition"]["zero_identifier_sphere_rows"] == 263
    assert receipt["source_disposition"]["byte_duplicate_rows"] == 0
    assert receipt["source_disposition"]["semantic_duplicate_sphere_keys"] == 0
    assert receipt["claim_boundary"]["successor_independently_audited"] is False
    assert receipt["claim_boundary"]["self_review_claimed_independent"] is False
    assert receipt["claim_boundary"]["executor_run_or_authorized"] is False
    assert all(gate["passed"] for gate in receipt["conformance_gates"])


def test_append_only_build_refuses_to_replace_frozen_artifacts() -> None:
    with pytest.raises(contract.Vast2SourceParserContractError, match="append-only package already exists"):
        contract.build_package()
