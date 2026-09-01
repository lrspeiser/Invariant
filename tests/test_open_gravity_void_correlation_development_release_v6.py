from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_correlation_development_release_v6 as release


def _fixture(payload: str | None = None) -> bytes:
    value = payload or release.load_config()["vast1_integration"]["fixture_raw_ascii"]
    return (value + "\n").encode("ascii")


def _vast2_fixture(payload: str | None = None) -> bytes:
    value = payload or release.load_config()["vast2_integration"]["fixture_raw_ascii"]
    return (value + "\n").encode("ascii")


def test_v1_through_v5_packets_and_v5_audit_are_byte_exact() -> None:
    config = release.load_config()
    bindings = release.validate_release_chain(config)
    assert (
        bindings["v4_failure_content_sha256"]
        == "e75f423476eb0f04c2b6ea9a01a0a1f9bd68dc74509d56e4df66b1ec1640fb83"
    )
    assert (
        bindings["v5_audit_content_sha256"]
        == "52f8dfece7f6f86d699abfa658b7a271c291a514bd4e4c2caa26d4936aa23e64"
    )
    modules = {
        "v1": release.v1,
        "v2": release.v2,
        "v3": release.v3,
        "v4": release.v4,
        "v5": release.v5,
    }
    for name, packet in config["development_packet_chain"].items():
        module = modules[name]
        assert (
            release.file_sha256(release.canonical_file(packet["config"]["path"]))
            == packet["config"]["raw_sha256"]
        )
        assert (
            release.file_sha256(release.canonical_file(packet["module"]["path"]))
            == packet["module"]["raw_sha256"]
        )
        assert (
            module.module_semantic_sha256(release.canonical_file(packet["module"]["path"]))
            == packet["module"]["semantic_sha256"]
        )
        assert (
            release.file_sha256(release.canonical_file(packet["test"]["path"]))
            == packet["test"]["raw_sha256"]
        )
        assert (
            release.file_sha256(release.canonical_file(packet["receipt"]["path"]))
            == packet["receipt"]["raw_sha256"]
        )


def test_source_free_build_never_resolves_or_opens_scientific_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = release.load_config()
    source_relatives = {section["path"] for section in config["sources"].values()}
    sources = {(release.REPO_ROOT / relative).absolute() for relative in source_relatives}
    original_open = Path.open
    original_canonical = release.canonical_file
    opened: list[Path] = []
    resolved: list[Path] = []

    def guarded_open(path: Path, *args, **kwargs):
        candidate = path.absolute()
        if candidate in sources:
            opened.append(candidate)
            raise AssertionError(f"scientific source opened: {candidate}")
        return original_open(path, *args, **kwargs)

    def guarded_canonical(relative: str) -> Path:
        if relative in source_relatives:
            candidate = (release.REPO_ROOT / relative).absolute()
            resolved.append(candidate)
            raise AssertionError(f"scientific source resolved: {candidate}")
        return original_canonical(relative)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(release, "canonical_file", guarded_canonical)
    receipt = release.build_receipt()
    assert release.main(["build"]) == 0
    assert release.main(["check"]) == 0
    assert release.main(["status"]) == 0
    assert opened == resolved == []
    assert receipt["access_accounting"]["scientific_source_paths_resolved"] == 0
    assert receipt["access_accounting"]["scientific_source_files_opened"] == 0


def test_audited_parser_accepts_zero_edge_two_and_all_four_payload_lengths() -> None:
    base = _fixture()
    row = release.parse_vast1_record_v6(base, source_index=0, framed_start=0)
    assert row["void"] == 0 and row["payload_bytes"] == 181
    edge_two = base[:101] + b"2" + base[102:]
    assert release.parse_vast1_record_v6(edge_two, source_index=0, framed_start=0)["edge"] == 2
    for remove, expected in ((0, 181), (1, 180), (2, 179), (3, 178)):
        frame = base if remove == 0 else base[: -1 - remove] + b"\n"
        assert (
            release.parse_vast1_record_v6(frame, source_index=0, framed_start=0)["payload_bytes"]
            == expected
        )


def test_audited_parser_rejects_old_and_out_of_contract_grammar() -> None:
    base = _fixture()
    with pytest.raises(release.vast1_contract.Vast1SourceParserContractError):
        release.parse_vast1_record_v6(
            base[:101] + b"3" + base[102:], source_index=0, framed_start=0
        )
    with pytest.raises(release.vast1_contract.Vast1SourceParserContractError):
        release.parse_vast1_record_v6(base[:-5] + b"\n", source_index=0, framed_start=0)
    with pytest.raises(release.vast1_contract.Vast1SourceParserContractError):
        release.parse_vast1_record_v6(base[:-1] + b"X\n", source_index=0, framed_start=0)


def test_v5_wrong_vast2_fixtures_now_pass_exact_audited_grammar() -> None:
    config = release.load_config()["vast2_integration"]
    fixtures = [
        (
            _vast2_fixture(),
            0,
            0,
            "fixture_framed_sha256",
            "fixture_payload_sha256",
            "Planck2018",
        ),
        (
            (config["wmap_zero_fixture_raw_ascii"] + "\n").encode("ascii"),
            39735,
            4211910,
            "wmap_zero_fixture_framed_sha256",
            "wmap_zero_fixture_payload_sha256",
            "WMAP5",
        ),
    ]
    for frame, index, offset, framed_key, payload_key, cosmology in fixtures:
        with pytest.raises(release.v1.DevelopmentReleaseV1Error):
            release.v5.v1.parse_vast_table2_record(frame)
        row = release.parse_vast2_record_v6(frame, source_index=index, framed_start=offset)
        assert row["Cosmo"] == cosmology and row["void"] == 0
        assert row["payload_bytes"] == 105 and row["framed_bytes"] == 106
        assert row["framed_raw_sha256"] == config[framed_key]
        assert row["payload_raw_sha256"] == config[payload_key]


def test_vast2_malformed_grammar_fails_closed() -> None:
    frame = _vast2_fixture()
    malformed = [
        frame[:-1] + b"\r\n",
        frame[:-2] + b"\n",
        frame[:10] + b"X" + frame[11:],
        frame[:101] + b"1184\n",
        b"BadCosmo  " + frame[10:],
    ]
    for value in malformed:
        with pytest.raises(release.vast2_contract.v1.Vast2SourceParserContractError):
            release.parse_vast2_record_v6(value, source_index=0, framed_start=0)


def test_repaired_geometry_join_accepts_zero_and_excludes_edge_two() -> None:
    sphere = ("Planck2018", 0, -1.0, 2.0, 3.0, 10.0)
    summary = release.validate_vast_duplicate_keys_v6([("Planck2018", 0, 2)], [sphere])
    assert summary == {
        "groups": 1,
        "sphere_rows": 1,
        "retained": 0,
        "excluded_edge": 1,
        "excluded_nonplanck": 0,
    }
    assert release.validate_vast_duplicate_keys_v6([("Planck2018", 0, 0)], [sphere]) == {
        "groups": 1,
        "sphere_rows": 1,
        "retained": 1,
        "excluded_edge": 0,
        "excluded_nonplanck": 0,
    }
    with pytest.raises(release.DevelopmentReleaseV6Error, match="VAST1 key"):
        release.validate_vast_duplicate_keys_v6([("Planck2018", 0, 3)], [sphere])


def test_vast2_union_groups_allow_repeated_membership_but_reject_exact_duplicate() -> None:
    first = ("Planck2018", 0, -1.0, 2.0, 3.0, 10.0)
    second = ("Planck2018", 0, -2.0, 2.5, 3.5, 9.0)
    summary = release.validate_vast_duplicate_keys_v6([("Planck2018", 0, 0)], [first, second])
    assert summary["groups"] == 1 and summary["sphere_rows"] == 2 and summary["retained"] == 2
    with pytest.raises(release.DevelopmentReleaseV6Error, match="semantic sphere key"):
        release.validate_vast_duplicate_keys_v6([("Planck2018", 0, 0)], [first, first])
    with pytest.raises(release.DevelopmentReleaseV6Error, match="group union mismatch"):
        release.validate_vast_duplicate_keys_v6([("Planck2018", 0, 0), ("WMAP5", 0, 0)], [first])


def test_owned_runner_is_fully_integrated_but_not_exposed_by_cli() -> None:
    assert list(inspect.signature(release.run_development_once).parameters) == []
    structure = release._runner_structure()
    assert all(structure.values())
    source = inspect.getsource(release.run_development_once)
    assert "parse_vast1_record_v6" in source
    assert "parse_vast2_record_v6" in source
    assert "v1.parse_vast_table1_record" not in source
    assert "v1.parse_vast_table2_record" not in source
    assert "prepare_vast_geometry_v6" in source
    assert "v1.prepare_vast_geometry" not in source
    assert source.index("_load_future_gates") < source.index("_consume_authorization")
    assert "v3.regenerate_permutations_from_rows" in source
    assert 'choices=("build", "check", "status")' in inspect.getsource(release.main)
    with pytest.raises(SystemExit):
        release.main(["run-development"])


def test_two_gate_paths_are_absent_and_no_run_side_effect_exists() -> None:
    config = release.load_config()
    assert not (release.REPO_ROOT / config["future_gates"]["independent_reaudit_path"]).exists()
    assert not (release.REPO_ROOT / config["future_gates"]["one_run_authorization_path"]).exists()
    assert not release.FINAL_DIRECTORY.exists()
    assert not release.STAGING_ROOT.exists()
    assert not release.FAILURE_DIRECTORY.exists()
    assert not release.CONSUMPTION_DIRECTORY.exists()


def test_parser_audit_and_retained_failure_are_exactly_bound() -> None:
    config = release.load_config()
    audit = release._load_bound_json(config["vast1_parser_contract"]["independent_audit"])
    assert audit["status"] == "PASS_EXECUTOR_SUCCESSOR_BUILD_ONLY_NO_SCIENTIFIC_RUN_AUTHORITY"
    assert audit["authority"]["scientific_development_runs_allowed"] == 0
    assert audit["authority"]["authorizations_may_be_consumed"] is False
    failure = release._load_bound_json(config["retained_v4_failure"])
    assert failure["stage"] == "VAST1_OWNED_STREAM"
    assert failure["access_counts"]["development_scores"] == 0
    v5_audit = release._load_bound_json(config["development_v5_independent_audit"])
    assert v5_audit["scientific_run_authority"] is False
    parser2_audit = release._load_bound_json(config["vast2_parser_contract"]["independent_audit"])
    assert (
        parser2_audit["status"]
        == "PASS_VAST2_SOURCE_PARSER_CONTRACT_V2_DISTINCT_INDEPENDENT_AUDIT_NO_RUN_AUTHORITY"
    )
    assert parser2_audit["authority"]["scientific_run_authority"] is False


def test_exact_10000_permutation_and_hard_seals_are_retained() -> None:
    config = release.load_config()
    assert release._PERMUTATIONS == 10000
    assert release.v3.np.__version__ == "2.2.6"
    assert config["future_executor"]["hard_seals"] == [
        "CF4_VALIDATION_SCIENTIFIC_FIELDS",
        "CF4_CONFIRMATION_SCIENTIFIC_FIELDS",
        "PANTHEON_PLUS_DATA",
        "PANTHEON_PLUS_COVARIANCE",
    ]
    source = inspect.getsource(release.validate_final_payloads_v6)
    assert "_exact_validate_regenerated" in source and "permutations=_PERMUTATIONS" in source


def test_runtime_config_and_rehashed_receipt_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(release, "run_development_once", lambda: "FORGED_RUN")
    with pytest.raises(release.DevelopmentReleaseV6Error, match="runner identity drift"):
        release.validate_code_pins()
    monkeypatch.undo()
    original = release.file_sha256
    monkeypatch.setattr(
        release,
        "file_sha256",
        lambda path: "0" * 64 if path == release.CONFIG_PATH else original(path),
    )
    with pytest.raises(release.DevelopmentReleaseV6Error, match="config raw drift"):
        release.load_config()
    monkeypatch.undo()
    receipt = release.check_receipt()
    receipt["decision"] = "REHASHED_FORGERY"
    receipt["content_sha256"] = release._self_hash(receipt)
    forged = tmp_path / "receipt.json"
    forged.write_bytes(release._pretty(receipt))
    monkeypatch.setattr(release, "OUTPUT_PATH", forged)
    with pytest.raises(release.DevelopmentReleaseV6Error, match="receipt drift"):
        release.check_receipt()


def test_arbitrary_path_rejected_before_open() -> None:
    for path in ("../source", "/source", "work\\private\\source", "./source"):
        with pytest.raises(release.DevelopmentReleaseV6Error):
            release.canonical_file(path)


def test_receipt_has_zero_authority_and_exact_fixture_only_access() -> None:
    receipt = release.check_receipt()
    assert (
        receipt["status"]
        == "PASS_SOURCE_FREE_VAST1_VAST2_PARSER_INTEGRATED_EXECUTOR_V6_AWAIT_INDEPENDENT_REAUDIT"
    )
    assert receipt["authority"]["scientific_runs_allowed"] == 0
    assert receipt["authority"]["authorizations_may_be_minted"] is False
    assert receipt["authority"]["authorizations_may_be_consumed"] is False
    assert receipt["access_accounting"]["allowed_vast1_fixture_rows_decoded"] == 1
    assert receipt["access_accounting"]["allowed_vast2_fixture_rows_decoded"] == 2
    assert receipt["access_accounting"]["scores_computed"] == 0
    assert all(gate["passed"] for gate in receipt["conformance_gates"])
