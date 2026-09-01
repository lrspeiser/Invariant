from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_correlation_development_release_v1 as release


def _config() -> dict:
    return release.load_config()


def _cf4_line(identifier: int = 12, **overrides: float) -> bytes:
    values: dict[str, float | int] = {
        "1PGC": identifier,
        "DMzp": 30.0,
        "e_DMzp": 0.1,
        "Dist": 10.0,
        "V3k": 1000,
        "RAdeg": 0.0,
        "DEdeg": 0.0,
    }
    values.update(overrides)
    config = _config()
    return release._format_synthetic_record(config["fixed_width_schemas"]["CF4_TABLE4"], values, 157)


def _development_identifiers(count: int) -> list[int]:
    result: list[int] = []
    candidate = 1
    while len(result) < count:
        if release.executor_v3.split_role(candidate)[1] == "development":
            result.append(candidate)
        candidate += 1
    return result


def _sealed_identifier() -> int:
    return next(value for value in range(1, 100) if release.executor_v3.split_role(value)[1] != "development")


def _profile_rows(count: int = 10) -> list[dict]:
    rows = []
    identifiers = _development_identifiers(count)
    for index, identifier in enumerate(identifiers):
        distance = 50.0 + index
        exposure = 5.0 + (index % 4)
        theta = 2.0 * math.pi * index / count
        direction = (math.cos(theta), math.sin(theta), 0.2 + 0.01 * index)
        rows.append(
            {
                "identifier": identifier,
                "source_index": index,
                "bucket": release.executor_v3.split_role(identifier)[0],
                "role": "development",
                "eligible_primary": True,
                "reason_codes": [],
                "cf4": {"1PGC": identifier, "DMzp": 30.0, "e_DMzp": 0.1, "Dist": distance, "V3k": 1000, "RAdeg": 0.0, "DEdeg": 0.0},
                "z_D": 0.01,
                "D_path_Mpc": distance,
                "direction": direction,
                "mask_pixel": True,
                "L_void_Mpc": exposure,
                "L_observed_matter_Mpc": distance - exposure,
                "L_unobserved_Mpc": 0.0,
                "void_fraction": exposure / distance,
                "union_crossings": 2,
                "maximum_chord_Mpc": 0.7 * exposure,
                "observer_endpoint_chord_Mpc": 0.1 * exposure,
                "target_endpoint_chord_Mpc": 0.2 * exposure,
                "y": 4.0 * exposure / 299792.458 + 0.00001 * index,
                "sigma_s": 0.001,
                "nuisance_design_log": release.executor_v3.velocity_to_log_design(
                    release.executor_v3.nuisance_velocity_design(distance, direction)
                ),
                "law_column": exposure / 299792.458,
            }
        )
    return rows


def test_release_chain_and_config_are_exactly_bound() -> None:
    config = _config()
    bindings = release.validate_release_chain(config)
    assert bindings["ids_audit_content_sha256"] == "0e7454add4a8aea6b80cc6316e83fcca30d3546acef05d13f2807876bf530163"
    assert config["release_chain"]["ids_v1"]["independent_audit"]["raw_sha256"] == "bee93e92774d34aad1e9839e0ee41574b9edb1da9b95acdb2b0c11d6e24692de"


def test_build_opens_no_scientific_source(monkeypatch: pytest.MonkeyPatch) -> None:
    source_paths = {(release.REPO_ROOT / value["path"]).resolve() for value in _config()["sources"].values()}
    original = Path.open

    def guarded(path: Path, *args, **kwargs):
        if path.resolve() in source_paths:
            raise AssertionError(f"scientific source opened: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    receipt = release.build_receipt()
    assert receipt["access_accounting"]["scientific_rows_decoded"] == 0
    assert receipt["access_accounting"]["real_scores"] == 0


def test_cf4_development_parser_verifies_hashes_offsets_and_domains() -> None:
    line = _cf4_line()
    ledger = release._synthetic_ledger(line, 12, source_index=4, framed_start=632)
    row = release.parse_cf4_development_record(line, ledger, source_index=4, framed_start=632)
    assert row == {"1PGC": 12, "source_index": 4, "DMzp": 30.0, "e_DMzp": 0.1, "Dist": 10.0, "V3k": 1000, "RAdeg": 0.0, "DEdeg": 0.0}


def test_sealed_role_rejects_before_any_scientific_field(monkeypatch: pytest.MonkeyPatch) -> None:
    identifier = _sealed_identifier()
    line = _cf4_line(identifier)
    ledger = release._synthetic_ledger(line, identifier)
    original = release.parse_field
    calls: list[str] = []

    def traced(payload: bytes, field: dict):
        calls.append(field["name"])
        return original(payload, field)

    monkeypatch.setattr(release, "parse_field", traced)
    with pytest.raises(release.DevelopmentReleaseV1Error, match="sealed role"):
        release.parse_cf4_development_record(line, ledger, source_index=0, framed_start=0)
    assert calls == ["1PGC"]


def test_sealed_role_does_not_validate_nonascii_scientific_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    identifier = _sealed_identifier()
    line = bytearray(_cf4_line(identifier))
    line[120] = 0xFF
    ledger = release._synthetic_ledger(bytes(line[:120] + bytearray(b"X") + line[121:]), identifier)
    payload = release._frame_payload(bytes(line), 157)
    ledger.update(
        {
            "framed_raw_sha256": release._sha256_bytes(bytes(line)),
            "payload_raw_sha256": release._sha256_bytes(payload),
            "identifier_field_raw_sha256": release._sha256_bytes(payload[:7]),
            "opaque_tail_raw_sha256": release._sha256_bytes(payload[7:]),
        }
    )
    original = release.parse_field
    calls: list[str] = []

    def traced(payload: bytes, field: dict):
        calls.append(field["name"])
        return original(payload, field)

    monkeypatch.setattr(release, "parse_field", traced)
    with pytest.raises(release.DevelopmentReleaseV1Error, match="sealed role"):
        release.parse_cf4_development_record(bytes(line), ledger, source_index=0, framed_start=0)
    assert calls == ["1PGC"]


def test_opaque_tail_mutation_rejects_before_identifier_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    line = bytearray(_cf4_line())
    ledger = release._synthetic_ledger(bytes(line), 12)
    line[120] = ord("X")
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("decoder reached")

    monkeypatch.setattr(release, "parse_field", forbidden)
    with pytest.raises(release.DevelopmentReleaseV1Error, match="framed hash mismatch"):
        release.parse_cf4_development_record(bytes(line), ledger, source_index=0, framed_start=0)
    assert calls == 0


@pytest.mark.parametrize("token", [b" 1e3  ", b" NaN  ", b"\t1.0  ", b"   1   "])
def test_strict_float_grammar_rejects_noncontract_tokens(token: bytes) -> None:
    field = {"name": "x", "start": 1, "end": len(token), "format": f"F{len(token)}.1", "required": True}
    with pytest.raises(release.DevelopmentReleaseV1Error):
        release.parse_field(token, field)


def test_record_framing_accepts_lf_and_crlf_only() -> None:
    payload = b" " * 4
    assert release.record_payload(payload + b"\n", 4) == payload
    assert release.record_payload(payload + b"\r\n", 4) == payload
    for bad in (payload, payload + b"\r", payload + b"\n\n", b" \n  \n"):
        with pytest.raises(release.DevelopmentReleaseV1Error):
            release.record_payload(bad, 4)


def test_vast_parsers_and_join_apply_coordinate_probe_and_h_once() -> None:
    config = _config()
    table1_values = {"Cosmo": "Planck2018", "x": 1.0, "y": 0.0, "z": 0.0, "Rad": 1.0, "void": 2, "edge": 0, "s": 1.0, "RAdeg": 0.0, "DEdeg": 0.0, "Reff": 1.0}
    table2_values = {"Cosmo": "Planck2018", "x": 2.0, "y": 0.0, "z": 0.0, "Rad": 1.0, "void": 2}
    table1 = release.parse_vast_table1_record(release._format_synthetic_record(config["fixed_width_schemas"]["VAST_TABLE1"], table1_values, 181))
    table2 = release.parse_vast_table2_record(release._format_synthetic_record(config["fixed_width_schemas"]["VAST_TABLE2"], table2_values, 105))
    geometry = release.prepare_vast_geometry([table1], [table2])
    assert geometry["retained"] == 1
    assert geometry["spheres_Mpc"][0][0][0] == 2.0 / 0.674
    assert geometry["spheres_Mpc"][0][1] == 1.0 / 0.674


def test_vast_duplicate_signed_zero_and_missing_join_fail_closed() -> None:
    table1 = [{"Cosmo": "Planck2018", "void": 1, "edge": 0, "x": 1.0, "y": 0.0, "z": 0.0, "s": 1.0, "RAdeg": 0.0, "DEdeg": 0.0}]
    sphere = {"Cosmo": "Planck2018", "void": 1, "x": 2.0, "y": 0.0, "z": 0.0, "Rad": 1.0}
    signed_zero = dict(sphere, y=-0.0)
    with pytest.raises(release.executor_v3.VoidExecutorV3Error, match="duplicate VAST_TABLE2"):
        release.prepare_vast_geometry(table1, [sphere, signed_zero])
    with pytest.raises(release.executor_v3.VoidExecutorV3Error, match="unmatched VAST sphere"):
        release.prepare_vast_geometry(table1, [dict(sphere, void=2)])


def test_coordinate_probe_failure_is_global() -> None:
    table1 = [{"Cosmo": "Planck2018", "void": 1, "edge": 0, "x": 1.1, "y": 0.0, "z": 0.0, "s": 1.0, "RAdeg": 0.0, "DEdeg": 0.0}]
    with pytest.raises(release.DevelopmentReleaseV1Error, match="coordinate probe"):
        release.prepare_vast_geometry(table1, [])


def test_geometry_union_eligibility_and_partial_mask_are_retained() -> None:
    parsed = release.parse_cf4_development_record(_cf4_line(), release._synthetic_ledger(_cf4_line(), 12), source_index=0, framed_start=0)
    spheres = [((4.0, 0.0, 0.0), 2.0), ((6.0, 0.0, 0.0), 2.0)]
    full = release.derive_development_row(parsed, bytes([1]) * 64800, spheres)
    partial = release.derive_development_row(parsed, bytes(64800), spheres)
    assert full["eligible_primary"] is True
    assert math.isclose(full["L_void_Mpc"], 6.0, abs_tol=1e-12)
    assert full["union_crossings"] == 1
    assert partial["eligible_primary"] is False
    assert partial["reason_codes"] == ["ANGULAR_MASK_FALSE", "UNOBSERVED_PATH"]
    assert partial["L_void_Mpc"] == 0.0


def test_likelihood_mapping_and_uncertainty_are_finite() -> None:
    line = _cf4_line()
    row = release.parse_cf4_development_record(line, release._synthetic_ledger(line, 12), source_index=0, framed_start=0)
    derived = release.derive_development_row(row, bytes([1]) * 64800, [])
    expected_y = math.log1p(1000.0 / 299792.458) - math.log1p(derived["z_D"])
    assert derived["y"] == expected_y
    assert derived["sigma_s"] > 250.0 / 299792.458
    assert len(derived["nuisance_design_log"]) == 9


def test_cwd_independent_planck_inversion_matches_frozen_geometry_v3() -> None:
    for distance in (0.0, 10.0, 100.0, 500.0):
        assert release.luminosity_to_comoving_hinv(distance) == release.geometry_v3.luminosity_to_comoving_hinv(distance)


def test_countermodels_use_same_frozen_profile_surface() -> None:
    scores = release.score_countermodels(_profile_rows())
    assert set(scores) == {
        "PRIMARY_UNION_VOID_PATH",
        "C00_FLRW_RADIAL_BULK_SHEAR_NULL",
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID",
        "C02_TARGET_ENDPOINT_LOCAL_VOID",
        "C03_SINGLE_DOMINANT_VOID",
    }
    assert scores["C00_FLRW_RADIAL_BULK_SHEAR_NULL"]["delta_chi2"] == 0.0
    assert all(value["delta_chi2"] >= 0.0 for value in scores.values())


def test_retained_profile_grid_equals_public_v3_summary() -> None:
    rows = _profile_rows()
    details = release.profile_grid_details(rows)
    assert len(details["profiles"]) == 161
    assert details["summary"] == release.score_exposure(rows, "L_void_Mpc")


def test_pcg64_permutation_reference_is_deterministic() -> None:
    first = release._permutation_reference(_profile_rows(), 3)
    second = release._permutation_reference(_profile_rows(), 3)
    assert first == second
    assert len(first["permutation_statistics"]) == 3
    assert first["p_value"] == (1 + first["tail_count"]) / 4


def test_thresholds_advance_anomaly_without_countermodel_or_physics_veto() -> None:
    primary = {"best_delta_H": 20.0, "one_sided_statistic": 12.0, "delta_chi2": 12.0}
    permutation = {"p_value": 0.005}
    countermodels = {
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID": {"delta_chi2": 15.0},
        "C02_TARGET_ENDPOINT_LOCAL_VOID": {"delta_chi2": 2.0},
        "C03_SINGLE_DOMINANT_VOID": {"delta_chi2": 3.0},
    }
    result = release.classify_development(primary, permutation, countermodels, 500)
    assert result["request_validation"] is True
    assert result["empirical_label"] == "COUNTERMODEL_DEGENERATE_ANOMALY"
    assert result["grid_boundary"] is True
    assert result["physics_veto_applied"] is False


def test_per_object_ledger_uses_hex_and_self_hashes() -> None:
    rows = _profile_rows()
    details = release.profile_grid_details(rows)
    ledger = release.development_ledger_rows(rows, details)
    assert [row["identifier"] for row in ledger] == sorted(row["identifier"] for row in ledger)
    assert all(row["role"] == "development" and row["primary_prediction_hex"].startswith(("0x", "-0x")) for row in ledger)
    for row in ledger:
        leaf = row.pop("row_leaf_sha256")
        assert leaf == release.content_sha256(row)


def test_all_frozen_future_artifacts_serialize_source_free() -> None:
    rows = _profile_rows(500)
    details = release.profile_grid_details(rows)
    permutation = {
        "observed": details["summary"]["one_sided_statistic"],
        "permutation_statistics": [0.0] * 10000,
        "tail_count": 0,
        "p_value": 1.0 / 10001.0,
    }
    countermodels = {
        "C00_FLRW_RADIAL_BULK_SHEAR_NULL": {"delta_chi2": 0.0},
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID": {"delta_chi2": 0.0},
        "C02_TARGET_ENDPOINT_LOCAL_VOID": {"delta_chi2": 0.0},
        "C03_SINGLE_DOMINANT_VOID": {"delta_chi2": 0.0},
    }
    artifacts = release.assemble_development_artifacts(rows, details, permutation, countermodels, {"scientific_rows": 500}, [])
    assert set(artifacts) == {
        "artifacts/development-rows.jsonl",
        "artifacts/profile-grid.jsonl",
        "artifacts/permutation-statistics.jsonl",
        "artifacts/countermodels.json",
        "artifacts/failures.json",
        "artifacts/development-summary.json",
    }
    assert artifacts["artifacts/profile-grid.jsonl"].count(b"\n") == 161
    assert artifacts["artifacts/permutation-statistics.jsonl"].count(b"\n") == 10000
    assert artifacts["artifacts/development-rows.jsonl"].count(b"\n") == 500


def test_artifact_assembly_rejects_unsanitized_failure() -> None:
    rows = _profile_rows(500)
    details = release.profile_grid_details(rows)
    permutation = {
        "observed": 0.0,
        "permutation_statistics": [0.0] * 10000,
        "tail_count": 10000,
        "p_value": 1.0,
    }
    countermodels = {
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID": {"delta_chi2": 0.0},
        "C02_TARGET_ENDPOINT_LOCAL_VOID": {"delta_chi2": 0.0},
        "C03_SINGLE_DOMINANT_VOID": {"delta_chi2": 0.0},
    }
    with pytest.raises(release.DevelopmentReleaseV1Error, match="unsanitized"):
        release.assemble_development_artifacts(rows, details, permutation, countermodels, {}, [{"raw_token": "forbidden"}])


def test_transaction_promotes_complete_tree(tmp_path: Path) -> None:
    final = tmp_path / "final"
    staging = tmp_path / "staging"
    result = release.transactional_promote(final, {"a.json": b"{}\n", "nested/b.bin": b"123"}, staging)
    assert result == "PROMOTED_COMPLETE"
    assert (final / "a.json").read_bytes() == b"{}\n"
    assert (final / "nested/b.bin").read_bytes() == b"123"
    assert list(staging.iterdir()) == []


def test_transaction_failure_cleans_only_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = tmp_path / "final"
    staging = tmp_path / "staging"
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(release, "file_sha256", lambda path: "wrong")
    with pytest.raises(release.DevelopmentReleaseV1Error, match="hash mismatch"):
        release.transactional_promote(final, {"a": b"x"}, staging)
    assert not final.exists()
    assert list(staging.iterdir()) == []
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "a/./b"])
def test_transaction_rejects_unsafe_artifact_names(name: str, tmp_path: Path) -> None:
    with pytest.raises(release.DevelopmentReleaseV1Error):
        release.transactional_promote(tmp_path / "final", {name: b"x"}, tmp_path / "staging")
    assert not (tmp_path / "final").exists()


def test_failure_record_is_sanitized_and_canonical() -> None:
    record = release.sanitized_failure_record("GEOMETRY", "JOIN_FAILURE", {"rows": 2}, [12])
    assert record == {"stage": "GEOMETRY", "reason_code": "JOIN_FAILURE", "access_counts": {"rows": 2}, "authorized_development_ids": [12]}
    assert "raw" not in json.dumps(record).lower()


def test_config_and_chain_mutations_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = release.file_sha256

    def mutated(path: Path) -> str:
        if path == release.CONFIG_PATH:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(release, "file_sha256", mutated)
    with pytest.raises(release.DevelopmentReleaseV1Error, match="config raw drift"):
        release.load_config()


def test_ids_audit_and_test_pin_mutations_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    audit_path = release.canonical_relative_path(config["release_chain"]["ids_v1"]["independent_audit"]["path"])
    original = release.file_sha256

    def audit_mutated(path: Path) -> str:
        if path == audit_path:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(release, "file_sha256", audit_mutated)
    with pytest.raises(release.DevelopmentReleaseV1Error, match="raw binding drift"):
        release.validate_release_chain(config)
    monkeypatch.setattr(release, "file_sha256", lambda path: "0" * 64 if path == release.TEST_PATH else original(path))
    with pytest.raises(release.DevelopmentReleaseV1Error, match="test pin drift"):
        release.validate_code_pins()


def test_receipt_freezes_outputs_thresholds_and_all_hard_seals() -> None:
    receipt = release.build_receipt()
    assert receipt["status"] == "PASS_SOURCE_FREE_DEVELOPMENT_RELEASE_CONTRACT_AWAIT_INDEPENDENT_AUDIT"
    assert receipt["thresholds"]["orthodoxy"].startswith("no established-physics")
    assert receipt["access_intent"]["hard_seals"] == [
        "validation CF4 scientific fields",
        "confirmation CF4 scientific fields",
        "Pantheon+ data",
        "Pantheon+ covariance",
    ]
    assert receipt["outputs"]["permutations"]["rows"] == 10000
    assert receipt["content_sha256"] == release._self_hash(receipt)


def test_cli_exposes_contract_commands_only() -> None:
    with pytest.raises(SystemExit):
        release.main(["open-development"])
