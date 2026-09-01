from __future__ import annotations

import math

import pytest

from sigma_theory_compiler import open_gravity_void_correlation_executor_contract_v1 as contract


def synthetic_cf4_line() -> bytes:
    line = bytearray(b" " * 157)
    line[0:7] = b"  12345"
    line[8:14] = b"35.000"
    line[15:20] = b"0.100"
    line[21:26] = b"100.0"
    line[39:44] = b" 7000"
    line[83:91] = b"120.0000"
    line[92:100] = b" 20.0000"
    return bytes(line)


def test_exact_schemas_are_ordered_nonoverlapping_and_bounded() -> None:
    config = contract.load_config()
    for name, fields in config["fixed_width_schemas"].items():
        contract.validate_schema(name, fields, config["inputs"][name]["record_length"])


def test_identifier_only_phase_reads_no_response_bytes() -> None:
    line = synthetic_cf4_line()
    assert contract.identifier_only_from_synthetic_line(line) == 12345
    changed_response = bytearray(line)
    changed_response[39:44] = b"-9999"
    assert contract.identifier_only_from_synthetic_line(bytes(changed_response)) == 12345


def test_synthetic_parser_distinguishes_mag_mpc_and_response() -> None:
    config = contract.load_config()
    line = synthetic_cf4_line()
    fields = {row["name"]: row for row in config["fixed_width_schemas"]["CF4_TABLE4"]}
    assert contract.parse_synthetic_field(line, fields["DMzp"]) == 35.0
    assert fields["DMzp"]["unit"] == "mag"
    assert contract.parse_synthetic_field(line, fields["Dist"]) == 100.0
    assert fields["Dist"]["role"] == "sole_mpc_input"
    assert contract.parse_synthetic_field(line, fields["V3k"]) == 7000
    assert fields["V3k"]["role"] == "response"


def test_missing_and_nonfinite_required_values_fail_closed() -> None:
    config = contract.load_config()
    fields = {row["name"]: row for row in config["fixed_width_schemas"]["CF4_TABLE4"]}
    blank = bytearray(synthetic_cf4_line())
    blank[21:26] = b"     "
    with pytest.raises(contract.VoidExecutorContractError, match="required field missing"):
        contract.parse_synthetic_field(bytes(blank), fields["Dist"])
    nonfinite = bytearray(synthetic_cf4_line())
    nonfinite[21:26] = b"  nan"
    with pytest.raises(contract.VoidExecutorContractError, match="nonfinite"):
        contract.parse_synthetic_field(bytes(nonfinite), fields["Dist"])


def test_group_split_and_confirmation_isolation_are_exact() -> None:
    bucket, role = contract.split_role(12345)
    assert 0 <= bucket <= 9
    assert role in {"development", "validation", "sealed_confirmation"}
    access = contract.load_config()["split_and_access"]
    assert "not authorized" in access["CONFIRMATION_OPEN"]
    assert access["minimum_primary_counts"] == {"development": 500, "validation": 150, "confirmation": 150}


def test_likelihood_and_nuisance_contract_are_finite_and_same_design() -> None:
    config = contract.load_config()
    likelihood = config["response_likelihood"]
    assert "Dist*e_DMzp" in likelihood["distance_sigma"]
    assert "250/c" in likelihood["variance"]
    assert likelihood["permutation"].endswith("902104729")
    assert config["flow_nuisance"]["same_design"].startswith("identical")
    assert math.isfinite(299792.458)


def test_contract_receipt_is_deterministic_and_zero_access() -> None:
    contract.validate_code_pins()
    first = contract.build_receipt()
    second = contract.build_receipt()
    assert first == second
    assert first["content_sha256"] == contract._self_hash(first)
    assert first["access_accounting"] == {"scientific_rows_decoded": 0, "identifier_rows_decoded": 0, "response_values_inspected": 0, "real_scores": 0}
    assert first["decision"] == "AWAIT_INDEPENDENT_REAUDIT_BEFORE_ANY_ROW_ACCESS"
    assert contract.check_receipt() == first
