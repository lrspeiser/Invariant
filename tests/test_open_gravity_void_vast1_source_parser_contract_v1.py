from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_vast1_source_parser_contract_v1 as contract


def _frames() -> list[bytes]:
    return (contract.REPO_ROOT / "work/private/open-gravity-void-source-v2/vast-table1.dat").read_bytes().splitlines(
        keepends=True
    )


def _replace(frame: bytes, start: int, end: int, token: bytes) -> bytes:
    assert len(token) == end - start
    return frame[:start] + token + frame[end:]


def test_exact_source_diagnosis_and_all_row_dispositions() -> None:
    config = contract.load_config()
    entries, summary, rows = contract.audit_vast1_source(config)
    assert len(entries) == len(rows) == 2347
    assert summary["status"] == "PASS_ALL_OFFICIAL_ROWS_ACCEPTED_FROZEN_V4_PARSER_WRONG_SOURCE_NOT_CORRUPT"
    assert summary["accepted_rows"] == 2347 and summary["rejected_rows"] == 0
    assert summary["zero_identifier_source_indices"] == [0, 1163]
    assert summary["edge_counts"] == {"0": 1622, "1": 679, "2": 46}
    assert summary["payload_length_counts"] == {"178": 6, "179": 58, "180": 534, "181": 1749}
    assert all(entry["parser_disposition"] == "ACCEPT_OFFICIAL_VAST1_ROW" for entry in entries)
    assert contract._ledger_root(entries) == summary["row_disposition_root_sha256"]


def test_exact_failure_row_and_neighboring_records() -> None:
    frames = _frames()
    offsets = [0, len(frames[0]), len(frames[0]) + len(frames[1])]
    rows = [contract.parse_vast1_record(frames[index], source_index=index, framed_start=offsets[index]) for index in range(3)]
    assert rows[0]["void"] == 0 and rows[0]["Rad"].hex() == "0x1.671135e46c159p+4"
    assert rows[0]["framed_raw_sha256"] == "e1aaeccae3e857121fd4b1b31895d21cf590e145f0421341e3c6ec7e6418a0a7"
    assert rows[0]["payload_raw_sha256"] == "3021b00c7baf0ffa9871d75d0d6e0f857cb4832c3566ec5049715c13214d49f2"
    assert (rows[1]["void"], rows[1]["payload_bytes"]) == (1, 181)
    assert rows[1]["framed_raw_sha256"] == "be5257d1df28514988b49c02d53d5140f29ed9c6bf06ca7a90bf89649b49c6d5"
    assert (rows[2]["void"], rows[2]["payload_bytes"]) == (2, 180)
    assert rows[2]["framed_raw_sha256"] == "df04a90248c65b081a34b64adf2f36ab2fd8775323c6562d5fffa00651bac486"


def test_zero_identifier_variable_final_width_and_edge_two_are_valid() -> None:
    frames = _frames()
    indices = (0, 2, 123, 148, 1163)
    offset = 0
    offsets: list[int] = []
    for frame in frames:
        offsets.append(offset)
        offset += len(frame)
    rows = [contract.parse_vast1_record(frames[index], source_index=index, framed_start=offsets[index]) for index in indices]
    assert rows[0]["void"] == 0
    assert rows[1]["payload_bytes"] == 180
    assert rows[2]["edge"] == 2
    assert rows[3]["payload_bytes"] == 178
    assert rows[4]["Cosmo"] == "WMAP5" and rows[4]["void"] == 0


def test_strict_record_grammar_rejects_adversaries() -> None:
    frame = _frames()[1]
    adversaries = [
        _replace(frame, 96, 100, b"  -1"),
        _replace(frame, 101, 102, b"3"),
        _replace(frame, 77, 95, b"        1.0000e+01"),
        _replace(frame, 10, 11, b"X"),
        _replace(frame, 0, 10, b"Planck2019"),
        frame[:-1] + b"\r\n",
        frame[:-1],
        frame[:-5] + b"\n",
        frame[:-1] + b"X\n",
    ]
    for adversary in adversaries:
        with pytest.raises(contract.Vast1SourceParserContractError):
            contract.parse_vast1_record(adversary, source_index=1, framed_start=182)
    short_frame = _frames()[2]
    leading_reff_space = short_frame[:163] + b" " + short_frame[163:]
    with pytest.raises(contract.Vast1SourceParserContractError, match="Reff"):
        contract.parse_vast1_record(leading_reff_space, source_index=2, framed_start=364)


def test_retained_failure_and_frozen_parser_behavior_are_bound() -> None:
    config = contract.load_config()
    binding = contract.validate_frozen_bindings(config)
    assert binding["failure_raw_sha256"] == "50faf79bbf7af6cf810eee5d1851087c02a9ab900045f0ed90718a8782e942fb"
    frames = _frames()
    with pytest.raises(contract.failed_v1.DevelopmentReleaseV1Error, match="radius or void"):
        contract.failed_v1.parse_vast_table1_record(frames[0])
    with pytest.raises(contract.failed_v1.DevelopmentReleaseV1Error, match="record length"):
        contract.failed_v1.parse_vast_table1_record(frames[2])


def test_only_vast1_and_readme_scientific_paths_are_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    prohibited = {
        (contract.REPO_ROOT / "work/private/open-gravity-void-source-v2/cf4-table4.dat.gz").resolve(),
        (contract.REPO_ROOT / "work/private/open-gravity-void-source-v2/vast-table2.dat.gz").resolve(),
        (contract.REPO_ROOT / "work/private/open-gravity-void-source-v2/vast-table3.dat.gz").resolve(),
        (contract.REPO_ROOT / "runs/gravity/open-gravity-void-geometry-source-completion-v2/artifacts/mask-u8.bin").resolve(),
    }
    original = Path.open
    opened_prohibited: list[Path] = []

    def guarded(path: Path, *args, **kwargs):
        resolved = path.resolve()
        if resolved in prohibited:
            opened_prohibited.append(resolved)
            raise AssertionError(f"prohibited source opened: {resolved}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    artifacts, receipt = contract.build_artifacts()
    assert set(artifacts) == {"artifacts/vast1-row-dispositions.jsonl", "artifacts/source-summary.json"}
    assert receipt["access_accounting"]["prohibited_source_files_opened"] == 0
    assert receipt["access_accounting"]["scores_computed"] == 0
    assert opened_prohibited == []


def test_documentation_block_and_official_source_hashes_are_exact() -> None:
    config = contract.load_config()
    readme = contract.canonical_file(config["documentation"]["path"]).read_bytes()
    block = b"".join(readme.splitlines(keepends=True)[80:105])
    assert len(block) == 1642
    assert hashlib.sha256(block).hexdigest() == "cfb52111b77e0f76b952e9c2cce8a1b70881c8fad775b357a6b5263ae8db0aed"
    assert b"void   [0/1183]?" in block
    assert b"edge   [0/2]" in block
    assert b"164- 181 F18.15" in block


def test_rehashed_ledger_summary_forgery_is_rejected() -> None:
    artifacts, receipt = contract.build_artifacts()
    ledger = [json.loads(line) for line in artifacts["artifacts/vast1-row-dispositions.jsonl"].splitlines()]
    ledger[0]["parser_disposition"] = "FORGED_ACCEPT"
    ledger[0]["content_sha256"] = contract._self_hash(ledger[0])
    forged_ledger = b"".join(contract._canonical(entry) + b"\n" for entry in ledger)
    payloads = {**artifacts, "receipt.json": contract._pretty(receipt)}
    payloads["artifacts/vast1-row-dispositions.jsonl"] = forged_ledger
    with pytest.raises(contract.Vast1SourceParserContractError):
        contract.validate_package_payloads(payloads)


def test_config_and_v4_failure_mutations_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = contract.file_sha256
    monkeypatch.setattr(
        contract,
        "file_sha256",
        lambda path: "0" * 64 if path == contract.CONFIG_PATH else original(path),
    )
    with pytest.raises(contract.Vast1SourceParserContractError, match="config raw drift"):
        contract.load_config()
    monkeypatch.undo()
    config = contract.load_config()
    failure_path = contract.canonical_file(config["retained_v4_failure"]["path"])
    monkeypatch.setattr(
        contract,
        "file_sha256",
        lambda path: "0" * 64 if path == failure_path else original(path),
    )
    with pytest.raises(contract.Vast1SourceParserContractError, match="bound raw drift"):
        contract.validate_frozen_bindings(config)


def test_arbitrary_paths_and_authority_escalation_are_rejected() -> None:
    for value in ("../secret", "/absolute", "work\\private\\source", "./relative"):
        with pytest.raises(contract.Vast1SourceParserContractError):
            contract.canonical_file(value)
    authority = contract.load_config()["authority"]
    assert authority["scoring_authority"] is False
    assert authority["development_run_authority"] is False
    assert authority["may_mint_or_consume_authorization"] is False
    source = inspect.getsource(contract.main)
    assert 'choices=("build", "check", "status")' in source
    with pytest.raises(SystemExit):
        contract.main(["run-development"])


def test_frozen_package_is_exact() -> None:
    receipt = contract.check_package()
    assert receipt["status"] == "PASS_VAST1_SOURCE_PARSER_CONTRACT_AWAIT_INDEPENDENT_AUDIT"
    assert receipt["decision"] == "REPAIRABLE_PARSER_CONTRACT_NO_NEW_DEVELOPMENT_AUTHORIZATION"
    assert all(gate["passed"] for gate in receipt["conformance_gates"])
    assert receipt["access_accounting"]["scores_computed"] == 0
