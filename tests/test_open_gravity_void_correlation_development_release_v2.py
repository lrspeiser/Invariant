from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_correlation_development_release_v2 as release


def _contract_receipt() -> dict:
    return release.check_receipt()


def _authorization() -> tuple[bytes, dict]:
    contract = _contract_receipt()
    config = release.load_config()
    value = {
        "schema": config["authorization_contract"]["schema"],
        "status": config["authorization_contract"]["status"],
        "decision": config["authorization_contract"]["decision"],
        "authorization_id": "a" * 64,
        "uses_allowed": 1,
        "hard_seals": config["authorization_contract"]["required_hard_seals"],
        "contract_binding": release._authorization_binding(contract),
        "content_sha256": "",
    }
    value["content_sha256"] = release._self_hash(value)
    return release._pretty(value), contract


def _failure_counts() -> dict[str, int]:
    return {key: 0 for key in release.load_config()["failure_contract"]["access_count_keys"]}


def _valid_profile_models() -> dict[str, dict]:
    template = {
        "best_delta_H": 0.0,
        "best_chi2": 1.0,
        "null_chi2": 1.0,
        "delta_chi2": 0.0,
        "one_sided_statistic": 0.0,
        "tied_delta_H": [0.0],
    }
    return {
        "PRIMARY_UNION_VOID_PATH": dict(template),
        "C00_FLRW_RADIAL_BULK_SHEAR_NULL": {"delta_chi2": 0.0, "best_delta_H": 0.0, "null_chi2": 1.0},
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID": dict(template),
        "C02_TARGET_ENDPOINT_LOCAL_VOID": dict(template),
        "C03_SINGLE_DOMINANT_VOID": dict(template),
    }


def _metadata_rows(ledger: list[dict]) -> list[dict]:
    return [
        {
            "identifier": entry["identifier"],
            "source_index": entry["source_index"],
            "bucket": entry["bucket"],
            "role": entry["role"],
        }
        for entry in ledger
        if entry["role"] == "development"
    ]


@pytest.fixture(scope="module")
def valid_package() -> tuple[dict[str, bytes], bytes, dict]:
    ledger = release.load_identifier_ledger()
    development = [entry for entry in ledger if entry["role"] == "development"]
    rows: list[dict] = []
    directions = (1.0, 0.0, 0.0)
    design_cache: dict[float, tuple[float, ...]] = {}
    for index, entry in enumerate(development):
        distance = 10.0 + float(index % 10)
        design_cache.setdefault(
            distance,
            release.v1.executor_v3.velocity_to_log_design(
                release.v1.executor_v3.nuisance_velocity_design(distance, directions)
            ),
        )
        eligible = index < 500
        rows.append(
            {
                "identifier": entry["identifier"],
                "source_index": entry["source_index"],
                "bucket": entry["bucket"],
                "role": "development",
                "eligible_primary": eligible,
                "reason_codes": [] if eligible else ["ANGULAR_MASK_FALSE", "UNOBSERVED_PATH"],
                "cf4": {
                    "1PGC": entry["identifier"],
                    "source_index": entry["source_index"],
                    "DMzp": 30.0,
                    "e_DMzp": 0.1,
                    "Dist": distance,
                    "V3k": 1000,
                    "RAdeg": 0.0,
                    "DEdeg": 0.0,
                },
                "z_D": 0.01,
                "D_path_Mpc": distance,
                "direction": directions,
                "mask_pixel": eligible,
                "L_void_Mpc": 0.0,
                "L_observed_matter_Mpc": distance if eligible else 0.0,
                "L_unobserved_Mpc": 0.0 if eligible else distance,
                "void_fraction": 0.0,
                "union_crossings": 0,
                "maximum_chord_Mpc": 0.0,
                "observer_endpoint_chord_Mpc": 0.0,
                "target_endpoint_chord_Mpc": 0.0,
                "y": 0.0,
                "sigma_s": 0.01,
                "nuisance_design_log": design_cache[distance],
                "law_column": 0.0,
            }
        )
    details = release.v1.profile_grid_details(rows)
    models = release.v1.score_countermodels(rows)
    permutation = {
        "observed": 0.0,
        "permutation_statistics": [0.0] * 10000,
        "tail_count": 10000,
        "p_value": 1.0,
    }
    artifacts = release.assemble_development_artifacts_v2(
        rows,
        ledger,
        details,
        permutation,
        models,
        release.expected_success_access_counts(),
        [],
    )
    authorization, contract = _authorization()
    payloads = release.finalize_development_artifacts(artifacts, authorization, contract)
    return payloads, authorization, contract


def test_blocked_v1_is_preserved_byte_exact() -> None:
    config = release.load_config()
    release.validate_blocked_v1(config)
    for section in config["blocked_v1"].values():
        assert release.canonical_file(section["path"]).read_bytes() == release.canonical_file(section["preserved_path"]).read_bytes()


def test_every_audited_ledger_leaf_and_offset_is_exact() -> None:
    ledger = release.load_identifier_ledger()
    assert len(ledger) == 38053
    assert sum(entry["role"] == "development" for entry in ledger) == 22897
    assert ledger[0]["framed_start"] == 0
    assert all(ledger[index]["framed_start"] == ledger[index - 1]["framed_end_exclusive"] for index in range(1, len(ledger)))


def test_stale_zero_and_rehashed_offset_ledger_entries_reject() -> None:
    entry = dict(release.load_identifier_ledger()[0])
    entry["leaf_sha256"] = "0" * 64
    with pytest.raises(release.DevelopmentReleaseV2Error, match="stale zero"):
        release.validate_ledger_entry(entry)
    entry = dict(release.load_identifier_ledger()[0])
    entry["payload_start"] += 1
    body = dict(entry)
    body.pop("leaf_sha256")
    entry["leaf_sha256"] = release.content_sha256(body)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="payload start"):
        release.validate_ledger_entry(entry)


def test_full_row_and_tail_hashes_are_checked_after_valid_leaf() -> None:
    line = release.v1._format_synthetic_record(
        release.v1.load_config()["fixed_width_schemas"]["CF4_TABLE4"],
        {"1PGC": 12, "DMzp": 30.0, "e_DMzp": 0.1, "Dist": 10.0, "V3k": 1000, "RAdeg": 0.0, "DEdeg": 0.0},
        157,
    )
    entry = release.v1._synthetic_ledger(line, 12)
    release.validate_ledger_entry(entry)
    mutated = bytearray(line)
    mutated[120] = ord("X")
    payload = release.v1._frame_payload(bytes(mutated), 157)
    entry["framed_raw_sha256"] = release.bytes_sha256(bytes(mutated))
    entry["payload_raw_sha256"] = release.bytes_sha256(payload)
    body = dict(entry)
    body.pop("leaf_sha256")
    entry["leaf_sha256"] = release.content_sha256(body)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="tail hash"):
        release.parse_cf4_development_record_v2(bytes(mutated), entry, source_index=0, framed_start=0)


def test_runtime_dependencies_are_raw_semantic_receipt_and_callable_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    config = release.load_config()
    release.validate_runtime_dependencies(config)
    monkeypatch.setattr(release.geometry_v3, "radec_to_xyz", lambda *args: None)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="identity drift"):
        release.validate_runtime_dependencies(config)


def test_runtime_nonfixture_file_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = release.load_config()
    target = release.canonical_file(config["runtime_dependencies"]["law_v4"]["module"]["path"])
    original = release.file_sha256
    monkeypatch.setattr(release, "file_sha256", lambda path: "0" * 64 if path == target else original(path))
    with pytest.raises(release.DevelopmentReleaseV2Error, match="module raw drift"):
        release.validate_runtime_dependencies(config)


def test_c04_is_distinct_and_exact_countermodel_set() -> None:
    permutation = {"observed": 0.0, "statistics": [0.0] * 10000, "tail_count": 10000, "p_value": 1.0}
    result = release.exact_countermodels(_valid_profile_models(), permutation, "1" * 64)
    assert set(result) == release._COUNTERMODEL_KEYS
    assert result["C04_STRATIFIED_EXCHANGEABILITY_NULL"] != result["C00_FLRW_RADIAL_BULK_SHEAR_NULL"]
    missing = _valid_profile_models()
    missing.pop("C03_SINGLE_DOMINANT_VOID")
    with pytest.raises(release.DevelopmentReleaseV2Error, match="exact-set"):
        release.exact_countermodels(missing, permutation, "1" * 64)


def test_exact_22897_development_coverage_rejects_missing_duplicate_and_sealed() -> None:
    ledger = release.load_identifier_ledger()
    rows = _metadata_rows(ledger)
    assert release.validate_exact_development_coverage(rows, ledger)["count"] == 22897
    with pytest.raises(release.DevelopmentReleaseV2Error, match="22897"):
        release.validate_exact_development_coverage(rows[:-1], ledger)
    duplicate = rows[:-1] + [dict(rows[0])]
    with pytest.raises(release.DevelopmentReleaseV2Error, match="duplicate"):
        release.validate_exact_development_coverage(duplicate, ledger)
    sealed = next(entry for entry in ledger if entry["role"] == "validation")
    replaced = rows[:-1] + [{"identifier": sealed["identifier"], "source_index": sealed["source_index"], "bucket": sealed["bucket"], "role": sealed["role"]}]
    with pytest.raises(release.DevelopmentReleaseV2Error, match="coverage"):
        release.validate_exact_development_coverage(replaced, ledger)


def test_permutation_recomputes_observed_tail_plus_one_and_exact_count() -> None:
    primary = {"one_sided_statistic": 1.0}
    statistics = [0.0] * 9999 + [1.0]
    valid = {"observed": 1.0, "permutation_statistics": statistics, "tail_count": 1, "p_value": 2.0 / 10001.0}
    assert release.validate_permutation(valid, primary)["tail_count"] == 1
    for mutation, match in (
        ({**valid, "permutation_statistics": statistics[:-1]}, "count"),
        ({**valid, "observed": 0.0}, "observed"),
        ({**valid, "tail_count": 0}, "tail"),
        ({**valid, "p_value": 1.0 / 10001.0}, "plus-one"),
    ):
        with pytest.raises(release.DevelopmentReleaseV2Error, match=match):
            release.validate_permutation(mutation, primary)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"stage": "UNKNOWN"}, "stage"),
        ({"reason": "UNKNOWN"}, "reason"),
        ({"missing_count": True}, "key"),
        ({"bool_count": True}, "access count"),
        ({"duplicate": True}, "duplicate"),
        ({"sealed": True}, "sealed"),
        ({"string_id": True}, "integer type"),
    ],
)
def test_failure_sanitizer_adversaries(mutation: dict, match: str) -> None:
    ledger = release.load_identifier_ledger()
    development_ids = {entry["identifier"] for entry in ledger if entry["role"] == "development"}
    first = next(iter(development_ids))
    stage, reason = "CF4_FRAME", "LEDGER_MISMATCH"
    counts = _failure_counts()
    ids = [first]
    if "stage" in mutation:
        stage = mutation["stage"]
    if "reason" in mutation:
        reason = mutation["reason"]
    if mutation.get("missing_count"):
        counts.pop(next(iter(counts)))
    if mutation.get("bool_count"):
        counts[next(iter(counts))] = True
    if mutation.get("duplicate"):
        ids = [first, first]
    if mutation.get("sealed"):
        ids = [next(entry["identifier"] for entry in ledger if entry["role"] == "validation")]
    if mutation.get("string_id"):
        ids = [str(first)]
    with pytest.raises(release.DevelopmentReleaseV2Error, match=match):
        release.sanitize_failure(stage, reason, counts, ids, development_ids)


def test_valid_failure_and_failure_receipt_are_self_hashed() -> None:
    ledger = release.load_identifier_ledger()
    development_ids = {entry["identifier"] for entry in ledger if entry["role"] == "development"}
    failure = release.sanitize_failure("CF4_FRAME", "LEDGER_MISMATCH", _failure_counts(), [min(development_ids)], development_ids)
    receipt = release.build_failure_receipt(failure, "a" * 64, _contract_receipt())
    assert receipt["content_sha256"] == release._self_hash(receipt)
    assert "raw" not in json.dumps(receipt["failure"]).lower()


def test_authorization_is_exact_self_hashed_and_single_use(tmp_path: Path) -> None:
    payload, contract = _authorization()
    authorization = release.validate_authorization_bytes(payload, contract)
    assert authorization["uses_allowed"] == 1
    marker = release.consume_authorization_at(tmp_path, payload, contract)
    assert marker["content_sha256"] == release._self_hash(marker)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="replay"):
        release.consume_authorization_at(tmp_path, payload, contract)


def test_rehashed_authorization_cannot_change_hard_seals_or_uses() -> None:
    payload, contract = _authorization()
    value = json.loads(payload)
    value["hard_seals"] = value["hard_seals"][:-1]
    value["content_sha256"] = release._self_hash(value)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="hard-seal"):
        release.validate_authorization_bytes(release._pretty(value), contract)
    value = json.loads(payload)
    value["uses_allowed"] = 2
    value["content_sha256"] = release._self_hash(value)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="use count"):
        release.validate_authorization_bytes(release._pretty(value), contract)


def test_production_entry_cannot_accept_caller_supplied_authorization() -> None:
    payload, _ = _authorization()
    with pytest.raises(TypeError):
        release.begin_fixed_one_shot(payload)


def _fixture_marker() -> dict:
    marker = {
        "schema": "invariant-open-gravity-void-correlation-development-authorization-consumption-2.0",
        "authorization_id": "a" * 64,
        "authorization_content_sha256": "b" * 64,
        "authorization_raw_sha256": "c" * 64,
        "contract_receipt_content_sha256": "d" * 64,
        "uses_consumed": 1,
        "content_sha256": "",
    }
    marker["content_sha256"] = release._self_hash(marker)
    return marker


def test_one_shot_gate_exact_success_no_second_pass_and_no_direct_promotion(valid_package) -> None:
    ledger = release.load_identifier_ledger()
    gate = release.OneShotDevelopmentGate(ledger, _fixture_marker())
    gate.source_open("CF4_TABLE4", 1)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="sequence"):
        gate.source_open("CF4_TABLE4", 1)
    for entry in ledger:
        gate.cf4_row(entry, entry["role"] == "development")
    gate.source_open("VAST_TABLE1", 0)
    gate.record_vast_table1(2347)
    gate.source_open("VAST_TABLE2", 1)
    gate.record_vast_table2(80080)
    gate.source_open("MASK_U8", 0)
    gate.record_mask()
    gate.record_development_score()
    completion = gate.finalize()
    assert completion.counts == release.expected_success_access_counts()
    with pytest.raises(release.DevelopmentReleaseV2Error, match="repeat"):
        gate.record_development_score()
    with pytest.raises(release.DevelopmentReleaseV2Error, match="already finalized"):
        gate.finalize()
    payloads, authorization, contract = valid_package
    artifacts = {name: payloads[name] for name in release._ARTIFACT_NAMES}
    with pytest.raises(release.DevelopmentReleaseV2Error, match="not consumed"):
        release.promote_fixed_package(artifacts, authorization, contract, completion)


def test_one_shot_gate_rejects_sealed_decode_and_incomplete_coverage() -> None:
    ledger = release.load_identifier_ledger()
    gate = release.OneShotDevelopmentGate(ledger, _fixture_marker())
    gate.source_open("CF4_TABLE4", 1)
    validation_index = next(index for index, entry in enumerate(ledger) if entry["role"] == "validation")
    for entry in ledger[:validation_index]:
        gate.cf4_row(entry, entry["role"] == "development")
    with pytest.raises(release.DevelopmentReleaseV2Error, match="sealed"):
        gate.cf4_row(ledger[validation_index], True)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="coverage incomplete"):
        gate.finalize()


def test_full_final_package_receipt_hashes_roots_counts_and_models(valid_package) -> None:
    payloads, authorization, contract = valid_package
    receipt = release.validate_package_payloads(payloads, authorization, contract)
    assert receipt["content_sha256"] == release._self_hash(receipt)
    assert receipt["counts"] == {"development_rows": 22897, "eligible_primary_rows": 500, "partial_mask_rows": 22397, "permutations": 10000}
    assert set(receipt["countermodels"]) == release._COUNTERMODEL_KEYS
    assert receipt["hard_seals"] == release.load_config()["authorization_contract"]["required_hard_seals"]


def test_rehashed_final_receipt_cannot_hide_stale_ledger_root(valid_package) -> None:
    payloads, authorization, contract = valid_package
    mutated = dict(payloads)
    rows = release._parse_jsonl(mutated["artifacts/development-rows.jsonl"])
    rows[0]["reason_codes"] = ["ANGULAR_MASK_FALSE"]
    body = dict(rows[0])
    body.pop("row_leaf_sha256")
    rows[0]["row_leaf_sha256"] = release.content_sha256(body)
    artifacts = {name: mutated[name] for name in release._ARTIFACT_NAMES}
    artifacts["artifacts/development-rows.jsonl"] = release._jsonl(rows)
    rehashed = release.finalize_development_artifacts(artifacts, authorization, contract)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="ledger root mismatch"):
        release.validate_package_payloads(rehashed, authorization, contract)


def test_rehashed_final_receipt_cannot_omit_summary_countermodel(valid_package) -> None:
    payloads, authorization, contract = valid_package
    artifacts = {name: payloads[name] for name in release._ARTIFACT_NAMES}
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    summary["countermodels"].pop("C04_STRATIFIED_EXCHANGEABILITY_NULL")
    artifacts["artifacts/development-summary.json"] = release._pretty(summary)
    rehashed = release.finalize_development_artifacts(artifacts, authorization, contract)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="summary countermodels"):
        release.validate_package_payloads(rehashed, authorization, contract)


def test_rehashed_coordinated_profile_permutation_and_c04_drift_rejects(valid_package) -> None:
    payloads, authorization, contract = valid_package
    artifacts = {name: payloads[name] for name in release._ARTIFACT_NAMES}
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    countermodels = json.loads(artifacts["artifacts/countermodels.json"])
    summary["profile"]["one_sided_statistic"] = (1.0).hex()
    summary["permutation"]["observed_hex"] = (1.0).hex()
    summary["permutation"]["tail_count"] = 0
    summary["permutation"]["p_value_hex"] = (1.0 / 10001.0).hex()
    countermodels["PRIMARY_UNION_VOID_PATH"]["one_sided_statistic"] = (1.0).hex()
    c04 = countermodels["C04_STRATIFIED_EXCHANGEABILITY_NULL"]
    c04["observed"] = (1.0).hex()
    c04["tail_count"] = 0
    c04["p_value"] = (1.0 / 10001.0).hex()
    artifacts["artifacts/countermodels.json"] = release._pretty(countermodels)
    summary["countermodels"] = countermodels
    summary["roots"]["countermodels_content_sha256"] = release.content_sha256(countermodels)
    artifacts["artifacts/development-summary.json"] = release._pretty(summary)
    rehashed = release.finalize_development_artifacts(artifacts, authorization, contract)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="profile does not exactly replay"):
        release.validate_package_payloads(rehashed, authorization, contract)


def test_file_fsync_and_directory_no_clobber(tmp_path: Path, valid_package, monkeypatch: pytest.MonkeyPatch) -> None:
    payloads, _, _ = valid_package
    staging = tmp_path / "staging"
    staging.mkdir()
    calls = 0
    original = release.os.fsync

    def counted(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        original(descriptor)

    monkeypatch.setattr(release.os, "fsync", counted)
    release._write_payload_tree(staging, payloads)
    assert calls >= len(payloads)
    final = tmp_path / "final"
    release._atomic_directory_promote(staging, final)
    assert (final / "receipt.json").is_file()
    second = tmp_path / "second"
    second.mkdir()
    (second / "sentinel").write_text("new", encoding="utf-8")
    with pytest.raises(release.DevelopmentReleaseV2Error):
        release._atomic_directory_promote(second, final)
    assert (final / "receipt.json").is_file()


def test_fixed_paths_and_promotion_require_consumption_marker(valid_package) -> None:
    payloads, authorization, contract = valid_package
    artifacts = {name: payloads[name] for name in release._ARTIFACT_NAMES}
    assert release.FINAL_DIRECTORY == release.REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v2"
    assert release.STAGING_ROOT == release.REPO_ROOT / "work/open-gravity-void-correlation-development-score-v2-staging"
    with pytest.raises(TypeError):
        release.promote_fixed_package(artifacts, authorization, contract)
    assert not release.FINAL_DIRECTORY.exists()


def test_noncanonical_path_rejects_before_any_open(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = False
    original = Path.open

    def guarded(path: Path, *args, **kwargs):
        nonlocal opened
        opened = True
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="drive path forbidden"):
        release.canonical_file("C:/arbitrary/scientific.dat")
    assert not opened


def test_rehashed_contract_receipt_top_level_mutation_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = release.check_receipt()
    receipt["decision"] = "MUTATED"
    receipt["content_sha256"] = release._self_hash(receipt)
    path = tmp_path / "receipt.json"
    path.write_bytes(release._pretty(receipt))
    monkeypatch.setattr(release, "OUTPUT_PATH", path)
    with pytest.raises(release.DevelopmentReleaseV2Error, match="receipt drift"):
        release.check_receipt()


def test_build_and_check_never_open_scientific_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    source_paths = {(release.REPO_ROOT / source["path"]).resolve() for source in release.v1.load_config()["sources"].values()}
    original = Path.open

    def guarded(path: Path, *args, **kwargs):
        if path.resolve() in source_paths:
            raise AssertionError(f"scientific source opened: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    receipt = release.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())


def test_config_test_and_preservation_mutations_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = release.file_sha256
    monkeypatch.setattr(release, "file_sha256", lambda path: "0" * 64 if path == release.CONFIG_PATH else original(path))
    with pytest.raises(release.DevelopmentReleaseV2Error, match="config raw drift"):
        release.load_config()
    monkeypatch.setattr(release, "file_sha256", lambda path: "0" * 64 if path == release.TEST_PATH else original(path))
    with pytest.raises(release.DevelopmentReleaseV2Error, match="test drift"):
        release.validate_code_pins()


def test_cli_exposes_source_free_commands_only() -> None:
    with pytest.raises(SystemExit):
        release.main(["run-development"])
