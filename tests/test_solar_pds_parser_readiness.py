from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from sigma_theory_compiler.solar_pds_parser_readiness import (
    PDSContractError,
    build_readiness_artifact,
    decode_record,
    decode_rsr_record,
    decode_tdf_tracking_record,
    normalized_table_layout,
    parse_pds3_label,
    parse_sce1_cumulative_index,
    select_direct_labels,
    sha256_bytes,
    unpack_rsr_iq_words,
    write_readiness_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/solar_pds_parser_readiness.json"
ARTIFACT_PATH = REPO_ROOT / "runs/engine/solar-pds-parser-readiness.json"


def _fixed_catalog_record(
    volume: str, label_path: str, product_id: str, start: str, stop: str
) -> bytes:
    raw = bytearray(b" " * 201)
    raw[-2:] = b"\r\n"

    def put(start_byte: int, length: int, value: str) -> None:
        encoded = value.encode("ascii")
        assert len(encoded) <= length
        raw[start_byte - 1 : start_byte - 1 + len(encoded)] = encoded

    put(2, 9, volume)
    put(14, 50, label_path)
    put(67, 31, product_id)
    put(143, 19, start)
    put(163, 19, stop)
    return bytes(raw)


def _tdf_label() -> bytes:
    return b"""PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 288
FILE_RECORDS = 1
OBJECT = TDF5_TABLE
  ROW_BYTES = 288
  ROW_SUFFIX_BYTES = 0
  OBJECT = COLUMN
    NAME = \"RECORD HEADER\"
    START_BYTE = 1
    BYTES = 9
    DATA_TYPE = MSB_BIT_STRING
    OBJECT = BIT_COLUMN
      NAME = \"RECORD FORMAT\"
      START_BIT = 5
      BITS = 32
      BIT_DATA_TYPE = MSB_INTEGER
    END_OBJECT = BIT_COLUMN
    OBJECT = BIT_COLUMN
      NAME = \"RECORD TYPE\"
      START_BIT = 41
      BITS = 32
      BIT_DATA_TYPE = MSB_INTEGER
    END_OBJECT = BIT_COLUMN
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = \"DATE-TIME BLOCK\"
    START_BYTE = 10
    BYTES = 9
    DATA_TYPE = MSB_BIT_STRING
    UNIT = \"UTC CALENDAR COMPONENTS\"
    OBJECT = BIT_COLUMN
      NAME = \"YEAR\"
      START_BIT = 1
      BITS = 12
      BIT_DATA_TYPE = MSB_UNSIGNED_INTEGER
    END_OBJECT = BIT_COLUMN
    OBJECT = BIT_COLUMN
      NAME = \"DOY\"
      START_BIT = 13
      BITS = 16
      BIT_DATA_TYPE = MSB_UNSIGNED_INTEGER
    END_OBJECT = BIT_COLUMN
    OBJECT = BIT_COLUMN
      NAME = \"HOUR\"
      START_BIT = 29
      BITS = 8
      BIT_DATA_TYPE = MSB_UNSIGNED_INTEGER
    END_OBJECT = BIT_COLUMN
    OBJECT = BIT_COLUMN
      NAME = \"MINUTE\"
      START_BIT = 37
      BITS = 8
      BIT_DATA_TYPE = MSB_UNSIGNED_INTEGER
    END_OBJECT = BIT_COLUMN
    OBJECT = BIT_COLUMN
      NAME = \"SECOND\"
      START_BIT = 45
      BITS = 8
      BIT_DATA_TYPE = MSB_UNSIGNED_INTEGER
    END_OBJECT = BIT_COLUMN
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = \"DATA TYPE BLOCK\"
    START_BYTE = 19
    BYTES = 4
    DATA_TYPE = MSB_BIT_STRING
    OBJECT = BIT_COLUMN
      NAME = \"STATION ID\"
      START_BIT = 1
      BITS = 10
      BIT_DATA_TYPE = MSB_UNSIGNED_INTEGER
    END_OBJECT = BIT_COLUMN
    OBJECT = BIT_COLUMN
      NAME = \"DOWNLINK FREQUENCY BAND\"
      START_BIT = 11
      BITS = 8
      BIT_DATA_TYPE = MSB_UNSIGNED_INTEGER
    END_OBJECT = BIT_COLUMN
    OBJECT = BIT_COLUMN
      NAME = \"SAMPLE DATA TYPE ID\"
      START_BIT = 19
      BITS = 6
      BIT_DATA_TYPE = MSB_UNSIGNED_INTEGER
    END_OBJECT = BIT_COLUMN
    OBJECT = BIT_COLUMN
      NAME = \"SIGNED CONTROL\"
      START_BIT = 25
      BITS = 8
      BIT_DATA_TYPE = MSB_INTEGER
    END_OBJECT = BIT_COLUMN
  END_OBJECT = COLUMN
END_OBJECT = TDF5_TABLE
END
"""


def _set_msb_bits(payload: bytearray, start_bit: int, bits: int, value: int) -> None:
    if value < 0:
        value += 1 << bits
    whole = int.from_bytes(payload, "big")
    shift = len(payload) * 8 - (start_bit - 1 + bits)
    mask = ((1 << bits) - 1) << shift
    whole = (whole & ~mask) | ((value << shift) & mask)
    payload[:] = whole.to_bytes(len(payload), "big")


def _tdf_record() -> bytes:
    payload = bytearray(288)
    for start, bits, value in (
        (5, 32, 8),
        (41, 32, 90),
        (73, 12, 102),
        (85, 16, 180),
        (101, 8, 13),
        (109, 8, 11),
        (117, 8, 25),
        (145, 10, 25),
        (155, 8, 2),
        (163, 6, 5),
        (169, 8, -2),
    ):
        _set_msb_bits(payload, start, bits, value)
    return bytes(payload)


def _rsr_label(*, sample_bytes: int = 4000) -> bytes:
    return f"""PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 4260
FILE_RECORDS = 1
OBJECT = TABLE
  ROW_BYTES = 4260
  ROW_SUFFIX_BYTES = 0
  OBJECT = COLUMN
    NAME = \"SAMPLE RESOLUTION\"
    START_BYTE = 69
    BYTES = 1
    DATA_TYPE = MSB_UNSIGNED_INTEGER
    UNIT = \"BIT\"
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = \"SAMPLE RATE\"
    START_BYTE = 71
    BYTES = 2
    DATA_TYPE = MSB_UNSIGNED_INTEGER
    UNIT = \"KILOSAMPLE PER SECOND\"
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = \"SFDU YEAR\"
    START_BYTE = 77
    BYTES = 2
    DATA_TYPE = MSB_UNSIGNED_INTEGER
    UNIT = \"UTC YEAR\"
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = \"SFDU DAY OF YEAR\"
    START_BYTE = 79
    BYTES = 2
    DATA_TYPE = MSB_UNSIGNED_INTEGER
    UNIT = \"UTC DAY\"
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = \"SFDU SECOND\"
    START_BYTE = 81
    BYTES = 8
    DATA_TYPE = IEEE_REAL
    UNIT = \"SECOND\"
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = \"DATA CHDO LENGTH\"
    START_BYTE = 259
    BYTES = 2
    DATA_TYPE = MSB_UNSIGNED_INTEGER
    UNIT = \"BYTE\"
  END_OBJECT = COLUMN
  OBJECT = COLUMN
    NAME = \"SAMPLE WORDS\"
    START_BYTE = 261
    BYTES = {sample_bytes}
    ITEMS = 1000
    ITEM_BYTES = 4
    ITEM_OFFSET = 4
    DATA_TYPE = MSB_UNSIGNED_INTEGER
    UNIT = \"DIMENSIONLESS ADC CODE\"
  END_OBJECT = COLUMN
END_OBJECT = TABLE
END
""".encode("ascii")


def _rsr_record() -> bytes:
    payload = bytearray(4260)
    payload[68] = 16
    payload[70:72] = (1).to_bytes(2, "big")
    payload[76:78] = (2005).to_bytes(2, "big")
    payload[78:80] = (336).to_bytes(2, "big")
    payload[80:88] = struct.pack(">d", 7800.0)
    payload[258:260] = (4000).to_bytes(2, "big")
    # Exact first four words from the authoritative 5336021a.rsr example.
    payload[260:276] = bytes.fromhex("2aea145d2bc7116b2a8110e7295b1039")
    return bytes(payload)


def test_fixed_width_metadata_selection_is_deterministic_and_target_free() -> None:
    records = []
    for kind in ("TDF", "RSR"):
        for index in range(6):
            records.append(
                _fixed_catalog_record(
                    f"CORS_{21 + index:04d}",
                    f"SCE1_{157 + index:03d}/{kind}/FILE{index}.LBL",
                    f"FILE{index}.{kind}",
                    f"2002-{157 + index:03d}T00:00:00.0",
                    f"2002-{157 + index:03d}T01:00:00.0",
                )
            )
    payload = b"".join(reversed(records))
    rows = parse_sce1_cumulative_index(payload, expected_sha256=sha256_bytes(payload))
    selected = select_direct_labels(rows)
    assert len(selected) == 12
    assert [row.record_class for row in selected].count("TDF") == 6
    assert [row.record_class for row in selected].count("RSR") == 6
    assert selected == select_direct_labels(reversed(rows))
    assert all(not hasattr(row, "target_value") for row in selected)

    mutated = bytearray(payload)
    mutated[20] ^= 1
    with pytest.raises(PDSContractError, match="SHA-256 mismatch"):
        parse_sce1_cumulative_index(bytes(mutated), expected_sha256=sha256_bytes(payload))


def test_tdf_parser_preserves_bit_semantics_time_and_signedness() -> None:
    label = _tdf_label()
    root = parse_pds3_label(label, expected_sha256=sha256_bytes(label))
    layout = normalized_table_layout(root, object_kind="TDF5_TABLE")
    record = _tdf_record()
    result = decode_tdf_tracking_record(layout, record, expected_sha256=sha256_bytes(record))

    assert result["utc_time_tag"] == {
        "day_of_year": 180,
        "hour": 13,
        "minute": 11,
        "second": 25,
        "time_system": "UTC",
        "year": 2002,
    }
    assert result["fields"]["STATION ID"]["value"] == 25
    assert result["fields"]["SAMPLE DATA TYPE ID"]["value"] == 5
    signed = decode_record(
        layout,
        record,
        expected_sha256=sha256_bytes(record),
        requested_fields=["SIGNED CONTROL"],
    )
    assert signed["SIGNED CONTROL"] == {
        "bits": 8,
        "data_type": "MSB_INTEGER",
        "endianness": "MSB_first",
        "unit": "N/A",
        "value": -2,
    }


def test_tdf_negative_controls_fail_closed() -> None:
    label = _tdf_label()
    layout = normalized_table_layout(
        parse_pds3_label(label, expected_sha256=sha256_bytes(label)), object_kind="TDF5_TABLE"
    )
    record = _tdf_record()
    mutated = bytearray(record)
    mutated[18] ^= 0x80
    with pytest.raises(PDSContractError, match="record SHA-256 mismatch"):
        decode_tdf_tracking_record(layout, bytes(mutated), expected_sha256=sha256_bytes(record))
    with pytest.raises(PDSContractError, match="truncation"):
        decode_tdf_tracking_record(layout, record[:-1], expected_sha256=sha256_bytes(record[:-1]))
    with pytest.raises(PDSContractError, match="label SHA-256 mismatch"):
        parse_pds3_label(label + b" ", expected_sha256=sha256_bytes(label))

    invalid = label.replace(b"BITS = 32", b"BITS = 80", 1)
    invalid_root = parse_pds3_label(invalid, expected_sha256=sha256_bytes(invalid))
    with pytest.raises(PDSContractError, match="outside parent"):
        normalized_table_layout(invalid_root, object_kind="TDF5_TABLE")


def test_rsr_known_answer_and_units_are_exact() -> None:
    label = _rsr_label()
    layout = normalized_table_layout(
        parse_pds3_label(label, expected_sha256=sha256_bytes(label)), object_kind="TABLE"
    )
    record = _rsr_record()
    result = decode_rsr_record(
        layout, record, expected_sha256=sha256_bytes(record), maximum_output_samples=4
    )
    assert result["utc_time_tag"] == {"day_of_year": 336, "second_of_day": 7800.0, "year": 2005}
    assert result["sample_resolution_bits"] == 16
    assert result["sample_rate_kilosample_per_second"] == 1
    assert [(item["i"], item["q"]) for item in result["samples"]] == [
        (10427, 21973),
        (8919, 22415),
        (8655, 21763),
        (8307, 21175),
    ]
    assert result["signedness"] == "twos_complement_per_sample_before_2k_plus_1_bias"
    assert result["output_units"].startswith("dimensionless")


def test_rsr_lower_resolution_time_order_and_negative_controls() -> None:
    # In each 16-bit word, the least-significant n-bit chunk is earlier.
    samples = unpack_rsr_iq_words(
        bytes.fromhex("fe0102ff"), sample_resolution_bits=8, maximum_output_samples=2
    )
    assert [(item["i"], item["q"]) for item in samples] == [(-1, 3), (5, -3)]

    label = _rsr_label()
    layout = normalized_table_layout(
        parse_pds3_label(label, expected_sha256=sha256_bytes(label)), object_kind="TABLE"
    )
    record = _rsr_record()
    mutated = bytearray(record)
    mutated[260] ^= 1
    with pytest.raises(PDSContractError, match="record SHA-256 mismatch"):
        decode_rsr_record(layout, bytes(mutated), expected_sha256=sha256_bytes(record))
    with pytest.raises(PDSContractError, match="truncation"):
        decode_rsr_record(layout, record[:-1], expected_sha256=sha256_bytes(record[:-1]))

    bad_length = bytearray(record)
    bad_length[258:260] = (3996).to_bytes(2, "big")
    with pytest.raises(PDSContractError, match="CHDO length"):
        decode_rsr_record(layout, bytes(bad_length), expected_sha256=sha256_bytes(bad_length))
    bad_rate = bytearray(record)
    bad_rate[70:72] = (2).to_bytes(2, "big")
    with pytest.raises(PDSContractError, match="rate, resolution"):
        decode_rsr_record(layout, bytes(bad_rate), expected_sha256=sha256_bytes(bad_rate))

    invalid_label = _rsr_label(sample_bytes=4001)
    invalid_root = parse_pds3_label(
        invalid_label, expected_sha256=sha256_bytes(invalid_label)
    )
    with pytest.raises(PDSContractError, match="outside the row"):
        normalized_table_layout(invalid_root, object_kind="TABLE")


def test_readiness_artifact_is_deterministic_sealed_and_unregistered(tmp_path: Path) -> None:
    first = build_readiness_artifact(REPO_ROOT, CONFIG_PATH)
    second = build_readiness_artifact(REPO_ROOT, CONFIG_PATH)
    assert first == second
    assert first["filled_registration_field_count"] == 2
    assert first["remaining_registration_field_count"] == 7
    assert set(first["filled_registration_fields"]) == {
        "verified_ATDF_TDF_parser_sha256",
        "verified_RSR_parser_sha256",
    }
    assert first["data_eligibility"]["observational_data_opened"] is False
    assert first["data_eligibility"]["target_values_accessed"] is False
    assert first["observational_authorization"] is False
    assert first["descriptor_registration_status"] == "blocked_unregistered"
    assert "selected_primary_file_root_sha256" in first["remaining_registration_fields"]
    assert "registered_real_source_interval_instantiation_certificate_sha256" in first[
        "remaining_registration_fields"
    ]

    output = tmp_path / "status.json"
    assert write_readiness_artifact(REPO_ROOT, CONFIG_PATH, output) == first
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_checked_in_artifact_matches_and_config_tamper_fails(tmp_path: Path) -> None:
    assert json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")) == build_readiness_artifact(
        REPO_ROOT, CONFIG_PATH
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["data_eligibility"]["candidate_use_authorized"] = True
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(PDSContractError, match="eligibility"):
        build_readiness_artifact(REPO_ROOT, tampered)
