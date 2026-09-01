from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_void_correlation_executor_contract_v3 as contract


def synthetic_cf4_line(ending: bytes = b"\n") -> bytes:
    payload = bytearray(b" " * 157)
    payload[0:7] = b"  12345"
    payload[8:14] = b"35.000"
    payload[15:20] = b"0.100"
    payload[21:26] = b"100.0"
    payload[39:44] = b" 7000"
    payload[83:91] = b"120.0000"
    payload[92:100] = b" 20.0000"
    return bytes(payload) + ending


def field_map() -> dict[str, dict[str, object]]:
    return {row["name"]: row for row in contract.load_config()["fixed_width_schemas"]["CF4_TABLE4"]}


def profile_fixture(sign: float = 1.0) -> tuple[list[float], list[float], list[float], list[float], list[tuple[float, float, float]], list[float], list[int]]:
    identifiers = list(range(101, 113))
    luminosity_distances = [40.0 + 2.0 * index for index in range(12)]
    path_distances = [35.0 + 1.8 * index for index in range(12)]
    directions = [(1.0, 0.11 * (index + 1), 0.07 * (index % 4 + 1)) for index in range(12)]
    l_void = [1.0, 8.0, 2.0, 10.0, 4.0, 7.0, 3.0, 11.0, 5.0, 9.0, 6.0, 12.0]
    sigma = [2.0e-5] * 12
    y = [sign * 6.0 * value / 299792.458 for value in l_void]
    return y, sigma, luminosity_distances, path_distances, directions, l_void, identifiers


def test_canonical_paths_are_cwd_independent_and_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = contract.CONFIG_PATH.resolve()
    monkeypatch.chdir(tmp_path)
    assert contract.canonical_bound_path("configs/open_gravity_void_correlation_executor_contract_v3.json") == expected
    for invalid in (
        "",
        "../configs/open_gravity_void_correlation_executor_contract_v3.json",
        "configs\\open_gravity_void_correlation_executor_contract_v3.json",
        "C:/Windows/win.ini",
        "/etc/passwd",
    ):
        with pytest.raises(contract.VoidExecutorV3Error):
            contract.canonical_bound_path(invalid)


def test_arbitrary_path_is_rejected_before_any_filesystem_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_is_file(_path: Path) -> bool:
        raise AssertionError("filesystem probe occurred before grammar rejection")

    monkeypatch.setattr(Path, "is_file", forbidden_is_file)
    with pytest.raises(contract.VoidExecutorV3Error, match="absolute"):
        contract.canonical_bound_path("C:/outside/secret.dat")


def test_fixed_width_schemas_are_exact_and_nonoverlapping() -> None:
    config = contract.load_config()
    for name, fields in config["fixed_width_schemas"].items():
        contract.validate_schema(name, fields, config["inputs"][name]["record_length"])
    bad = [dict(row) for row in config["fixed_width_schemas"]["CF4_TABLE4"]]
    bad[1]["start"] = 7
    with pytest.raises(contract.VoidExecutorV3Error, match="overlap"):
        contract.validate_schema("BAD", bad, 157)


def test_strict_record_and_numeric_grammar_accepts_only_frozen_forms() -> None:
    fields = field_map()
    payload = contract.synthetic_record_payload(synthetic_cf4_line(), 157)
    assert contract.parse_synthetic_field(payload, fields["1PGC"]) == 12345
    assert contract.parse_synthetic_field(payload, fields["DMzp"]) == 35.0
    assert contract.parse_synthetic_field(payload, fields["V3k"]) == 7000
    assert contract.synthetic_record_payload(synthetic_cf4_line(b"\r\n"), 157) == payload

    for ending in (b"", b"\r", b"\n\n", b"\r\r\n"):
        with pytest.raises(contract.VoidExecutorV3Error):
            contract.synthetic_record_payload(synthetic_cf4_line(ending), 157)

    invalid_floats = (b"   nan", b"   inf", b"  1e02", b"   100", b" 1,000", b" 1\t.0")
    for token in invalid_floats:
        changed = bytearray(payload)
        changed[8:14] = token
        with pytest.raises(contract.VoidExecutorV3Error):
            contract.parse_synthetic_field(bytes(changed), fields["DMzp"])

    invalid_integers = (b"  1.0", b"  1e2", b"  1 2", b"  + 1")
    for token in invalid_integers:
        changed = bytearray(payload)
        changed[39:44] = token
        with pytest.raises(contract.VoidExecutorV3Error):
            contract.parse_synthetic_field(bytes(changed), fields["V3k"])

    blank = bytearray(payload)
    blank[21:26] = b"     "
    with pytest.raises(contract.VoidExecutorV3Error, match="required field missing"):
        contract.parse_synthetic_field(bytes(blank), fields["Dist"])


def test_identifier_canonicalization_and_split_are_exact() -> None:
    assert contract.canonical_identifier(123) == b"123"
    assert contract.split_role(123) == contract.split_role(int("000123"))
    bucket, role = contract.split_role(123)
    assert 0 <= bucket <= 9
    assert role in {"development", "validation", "confirmation"}
    for invalid in (0, -1, True, 1.0):
        with pytest.raises(contract.VoidExecutorV3Error):
            contract.canonical_identifier(invalid)  # type: ignore[arg-type]


def test_synthetic_split_arithmetic_is_exact_and_meets_raw_stage_minima() -> None:
    identifiers = list(range(1, 10_001))
    counts = {"development": 0, "validation": 0, "confirmation": 0}
    for identifier in identifiers:
        bucket, role = contract.split_role(identifier)
        expected = int.from_bytes(hashlib.sha256(str(identifier).encode("ascii")).digest()[:8], "big") % 10
        assert bucket == expected
        counts[role] += 1
    assert sum(counts.values()) == len(identifiers)
    contract.validate_stage_counts("IDS_PARTITIONED", counts)


def test_stf_basis_is_symmetric_trace_free_orthonormal_and_matches_columns() -> None:
    basis = contract.stf_basis()
    contract.validate_stf_basis(basis)
    assert basis.shape == (5, 3, 3)
    direction = contract.normalize_direction((2.0, -3.0, 4.0))
    columns = contract.shear_quadratic_columns(direction)
    direct = tuple(float(np.asarray(direction) @ matrix @ np.asarray(direction)) for matrix in basis)
    assert columns == pytest.approx(direct, rel=0.0, abs=5e-16)
    gram = np.einsum("aij,bij->ab", basis, basis)
    assert gram == pytest.approx(np.eye(5), rel=0.0, abs=5e-16)


def test_adversarial_direction_public_entries_normalize_once_and_share_exact_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = (-0.09524089298036276, 0.11954477216099191, 0.8484211680474587)
    expected_direction_hex = ["bfbc484bb6b3f3f5", "3fc1bff2ecc41247", "3fef7e36f36e012a"]
    expected_shear_hex = ["bf7458020b289b21", "bfe8e559b47078be", "bf962f8fbace0e32", "bfc3ae8d3e92d731", "3fc8b44e40432083"]
    once = contract.normalize_direction(raw)
    assert [struct.pack(">d", value).hex() for value in once] == expected_direction_hex

    original = contract.normalize_direction
    calls: list[tuple[float, float, float]] = []

    def traced(direction: tuple[float, float, float]) -> tuple[float, float, float]:
        calls.append(tuple(direction))
        return original(direction)

    monkeypatch.setattr(contract, "normalize_direction", traced)
    shear = contract.shear_quadratic_columns(raw)
    assert calls == [raw]
    assert [struct.pack(">d", value).hex() for value in shear] == expected_shear_hex

    calls.clear()
    design = contract.nuisance_velocity_design(17.0, raw)
    assert calls == [raw]
    assert tuple(design[1:4]) == once
    assert tuple(design[4:]) == tuple(17.0 * value for value in shear)

    calls.clear()
    assert contract._shear_quadratic_columns_from_normalized(once) == shear
    assert contract._nuisance_velocity_design_from_normalized(17.0, once) == design
    assert calls == []


def test_profile_path_normalizes_each_raw_direction_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    y, sigma, _luminosity_distances, path_distances, directions, l_void, identifiers = profile_fixture(1.0)
    adversarial = (-0.09524089298036276, 0.11954477216099191, 0.8484211680474587)
    directions[0] = adversarial
    original = contract.normalize_direction
    calls: list[tuple[float, float, float]] = []

    def traced(direction: tuple[float, float, float]) -> tuple[float, float, float]:
        calls.append(tuple(direction))
        return original(direction)

    monkeypatch.setattr(contract, "normalize_direction", traced)
    result = contract.profile_at_delta(y, sigma, path_distances, directions, l_void, identifiers, 0.0)
    assert math.isfinite(result["chi2"])
    assert len(calls) == len(identifiers)
    assert calls[0] == adversarial
    assert calls == [tuple(direction) for direction in directions]

    calls.clear()
    grid = contract.profile_grid(y, sigma, path_distances, directions, l_void, identifiers)
    assert math.isfinite(grid["best_chi2"])
    assert len(calls) == len(identifiers)
    assert calls == [tuple(direction) for direction in directions]


def test_velocity_design_maps_to_log_redshift_by_dividing_each_column_once() -> None:
    velocity = contract.nuisance_velocity_design(100.0, (1.0, 2.0, 3.0))
    mapped = contract.velocity_to_log_design(velocity)
    assert len(velocity) == len(mapped) == 9
    for raw, log_value in zip(velocity, mapped, strict=True):
        assert log_value == raw / 299792.458
    assert contract.observed_log_redshift(0.0) == 0.0
    with pytest.raises(contract.VoidExecutorV3Error, match="domain"):
        contract.observed_log_redshift(-299792.458)


def test_equal_count_distance_strata_use_identifier_to_break_all_ties() -> None:
    identifiers = list(range(23, 0, -1))
    labels = contract.distance_strata([100.0] * 23, identifiers)
    by_identifier = {identifier: labels[index] for index, identifier in enumerate(identifiers)}
    ordered_labels = [by_identifier[identifier] for identifier in range(1, 24)]
    assert ordered_labels == [(10 * rank) // 23 for rank in range(23)]
    sizes = [labels.count(index) for index in range(10)]
    assert min(sizes) == 2
    assert max(sizes) == 3
    with pytest.raises(contract.VoidExecutorV3Error, match="duplicate"):
        contract.distance_strata([1.0] * 10, [1] * 10)


def test_duplicate_semantics_are_table_specific_and_unmatched_spheres_reject() -> None:
    contract.validate_cf4_duplicate_keys([1, 2, 3])
    with pytest.raises(contract.VoidExecutorV3Error, match="duplicate CF4"):
        contract.validate_cf4_duplicate_keys([1, 2, 1])
    table1 = [("Planck2018", 1, 0), ("Planck2018", 2, 1)]
    table2 = [
        ("Planck2018", 1, 1.0, 2.0, 3.0, 4.0),
        ("Planck2018", 1, 2.0, 2.0, 3.0, 4.0),
    ]
    counts = contract.validate_vast_duplicate_keys(table1, table2)
    assert counts == {"retained": 2, "excluded_edge": 0}
    edge_counts = contract.validate_vast_duplicate_keys(table1, [("Planck2018", 2, 1.0, 2.0, 3.0, 4.0)])
    assert edge_counts == {"retained": 0, "excluded_edge": 1}
    with pytest.raises(contract.VoidExecutorV3Error, match="sphere"):
        contract.validate_vast_duplicate_keys(table1, table2 + [table2[0]])
    with pytest.raises(contract.VoidExecutorV3Error, match="unmatched"):
        contract.validate_vast_duplicate_keys(table1, [("Planck2018", 3, 1.0, 2.0, 3.0, 4.0)])


def test_minimum_counts_are_staged_without_peeking_at_holdout_eligibility() -> None:
    contract.validate_stage_counts("IDS_PARTITIONED", {"development": 500, "validation": 150, "confirmation": 150})
    contract.validate_stage_counts("DEVELOPMENT_OPEN", eligible_development=500)
    with pytest.raises(contract.VoidExecutorV3Error, match="raw-ID minimum"):
        contract.validate_stage_counts("IDS_PARTITIONED", {"development": 499, "validation": 150, "confirmation": 150})
    with pytest.raises(contract.VoidExecutorV3Error, match="eligibility forbidden"):
        contract.validate_stage_counts("IDS_PARTITIONED", {"development": 500, "validation": 150, "confirmation": 150}, eligible_development=999)
    for stage in ("VALIDATION_OPEN", "CONFIRMATION_OPEN", "PANTHEON_OPEN"):
        with pytest.raises(contract.VoidExecutorV3Error, match="not authorized"):
            contract.validate_stage_counts(stage)


def test_profile_solver_is_order_invariant_and_recovers_positive_synthetic_grid_point() -> None:
    y, sigma, _luminosity_distances, path_distances, directions, l_void, identifiers = profile_fixture(1.0)
    first = contract.profile_grid(y, sigma, path_distances, directions, l_void, identifiers)
    order = [5, 0, 9, 1, 11, 2, 8, 3, 10, 4, 7, 6]
    second = contract.profile_grid(
        [y[index] for index in order],
        [sigma[index] for index in order],
        [path_distances[index] for index in order],
        [directions[index] for index in order],
        [l_void[index] for index in order],
        [identifiers[index] for index in order],
    )
    assert first == second
    assert first["best_delta_H"] > 0.0
    assert first["one_sided_statistic"] == first["delta_chi2"]


def test_profile_grid_tie_rule_prefers_zero_then_one_sided_negative_is_zero() -> None:
    y, sigma, _luminosity_distances, path_distances, directions, l_void, identifiers = profile_fixture(1.0)
    tied = contract.profile_grid([0.0] * 12, sigma, path_distances, directions, [0.0] * 12, identifiers)
    assert tied["best_delta_H"] == 0.0
    assert len(tied["tied_delta_H"]) == 161
    assert tied["one_sided_statistic"] == 0.0
    negative = contract.profile_grid([-value for value in y], sigma, path_distances, directions, l_void, identifiers)
    assert negative["best_delta_H"] < 0.0
    assert negative["delta_chi2"] > 0.0
    assert negative["one_sided_statistic"] == 0.0


def test_permutation_rule_is_reproducible_inclusive_and_plus_one() -> None:
    y, sigma, luminosity_distances, path_distances, directions, l_void, identifiers = profile_fixture(1.0)
    first = contract.synthetic_permutation_test(y, sigma, luminosity_distances, path_distances, directions, l_void, identifiers, 3)
    second = contract.synthetic_permutation_test(y, sigma, luminosity_distances, path_distances, directions, l_void, identifiers, 3)
    assert first == second
    assert first["tail_count"] == sum(value >= first["observed"] for value in first["permutation_statistics"])
    assert first["p_value"] == (1 + first["tail_count"]) / 4
    assert 0.25 <= first["p_value"] <= 1.0


def test_exact_pcg64_permutation_call_order_is_frozen() -> None:
    generator = np.random.Generator(np.random.PCG64(902104729))
    assert contract._pcg64_permutation_orders(generator, [5, 3]) == [[4, 0, 3, 1, 2], [0, 1, 2]]
    assert generator.bit_generator.random_raw(4).tolist() == [
        6125280640169825694,
        13722761692446950137,
        14658893663403773969,
        16732217884544078908,
    ]


def test_receipt_is_deterministic_hash_only_and_rows_unopened() -> None:
    contract.validate_code_pins()
    first = contract.build_receipt()
    second = contract.build_receipt()
    assert first == second
    assert first["content_sha256"] == contract._self_hash(first)
    assert first["status"] == "PASS_NORMALIZE_ONCE_REPAIR_MUTATION_FREEZE_ROWS_UNOPENED_AWAIT_INDEPENDENT_AUDIT"
    assert first["next_gate"].startswith("A_DIFFERENT_AGENT")
    assert first["predecessor_audit_block"]["status"] == "BLOCK_INDEPENDENT_AUDIT_NORMALIZATION_COUNT_MISMATCH"
    assert first["mutation_freeze"]["audit_block_raw_sha256"] == "d6e681f4e725c11c2d88f2f3c72975482002955aa65cd9fed970bbd2ce11c5ab"
    assert all(source["operation"] == "STREAMING_RAW_SHA256_ONLY" for source in first["source_bindings"].values())
    assert first["access_accounting"] == {
        "source_files_hashed": 8,
        "source_files_decompressed": 0,
        "scientific_rows_decoded": 0,
        "identifier_rows_decoded": 0,
        "response_values_inspected": 0,
        "real_scores": 0,
        "validation_rows_decoded": 0,
        "confirmation_rows_decoded": 0,
        "pantheon_rows_decoded": 0,
    }
    assert set(contract.load_config()["split_and_access"]["implemented_commands"]) == {"build", "check", "status"}
    assert len(contract.load_config()["inputs"]) == 8


def test_build_and_check_never_call_decompression_or_row_parsers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sealed row/decompression function called")

    monkeypatch.setattr(gzip, "open", forbidden)
    monkeypatch.setattr(contract, "synthetic_record_payload", forbidden)
    monkeypatch.setattr(contract, "parse_synthetic_field", forbidden)
    monkeypatch.setattr(contract, "split_role", forbidden)
    receipt = contract.build_receipt()
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(contract, "OUTPUT_PATH", output)
    assert contract.check_receipt() == receipt


def test_top_level_config_mutation_fails_even_if_attacker_rehashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mutated = json.loads(contract.CONFIG_PATH.read_text(encoding="utf-8"))
    mutated["status"] = "ATTACKER_REHASHED_STATUS"
    path = tmp_path / "mutated-config.json"
    path.write_text(json.dumps(mutated, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(contract, "CONFIG_PATH", path)
    monkeypatch.setattr(contract, "_CONFIG_RAW_SHA256", contract.file_sha256(path))
    monkeypatch.setattr(contract, "_CONFIG_CONTENT_SHA256", contract.content_sha256(mutated))
    with pytest.raises(contract.VoidExecutorV3Error, match="status drift"):
        contract.load_config()


def test_top_level_receipt_mutation_fails_even_with_new_self_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mutated = contract.build_receipt()
    mutated["status"] = "ATTACKER_REHASHED_STATUS"
    mutated["content_sha256"] = contract._self_hash(mutated)
    path = tmp_path / "mutated-receipt.json"
    path.write_text(json.dumps(mutated, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(contract, "OUTPUT_PATH", path)
    with pytest.raises(contract.VoidExecutorV3Error, match="receipt drift"):
        contract.check_receipt()


def test_nonfinite_profile_inputs_fail_closed() -> None:
    y, sigma, _luminosity_distances, path_distances, directions, l_void, identifiers = profile_fixture(1.0)
    bad_y = list(y)
    bad_y[0] = math.nan
    with pytest.raises(contract.VoidExecutorV3Error, match="nonfinite response"):
        contract.profile_at_delta(bad_y, sigma, path_distances, directions, l_void, identifiers, 0.0)
    bad_sigma = list(sigma)
    bad_sigma[0] = 0.0
    with pytest.raises(contract.VoidExecutorV3Error, match="uncertainty"):
        contract.profile_at_delta(y, bad_sigma, path_distances, directions, l_void, identifiers, 0.0)
    bad_void = list(l_void)
    bad_void[0] = path_distances[0] + 1.0
    with pytest.raises(contract.VoidExecutorV3Error, match="void path"):
        contract.profile_at_delta(y, sigma, path_distances, directions, bad_void, identifiers, 0.0)
