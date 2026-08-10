"""Fail-closed PDS3 binary-layout parsers for the sealed Solar protocol.

This module deliberately separates metadata/label work from primary-record access.
It parses a small, explicit subset of PDS3 labels, validates binary layouts, and
decodes caller-supplied bytes only when their expected SHA-256 is provided.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any


class PDSContractError(ValueError):
    """Raised when a label, layout, or payload fails closed."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KEY_VALUE_RE = re.compile(r"^\s*([A-Z][A-Z0-9_^]*)\s*=\s*(.*?)\s*$")
_TIME_RE = re.compile(r"^\d{4}-\d{3}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$")
_DIRECT_CLASSES = ("TDF", "RSR")
_SUPPORTED_SCALARS = {
    "CHARACTER",
    "MSB_UNSIGNED_INTEGER",
    "MSB_INTEGER",
    "IEEE_REAL",
}
_SUPPORTED_BITS = {"MSB_UNSIGNED_INTEGER", "MSB_INTEGER"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(payload.encode("ascii"))


def _require_sha256(value: str, *, name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise PDSContractError(f"{name} is not a lowercase SHA-256")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _integer(value: str, *, name: str) -> int:
    value = _unquote(value)
    if not re.fullmatch(r"[0-9]+", value):
        raise PDSContractError(f"{name} must be a nonnegative decimal integer")
    return int(value)


@dataclass(frozen=True)
class CatalogRow:
    volume: str
    label_path: str
    product_id: str
    start_time: str
    stop_time: str
    record_class: str

    def identity(self) -> dict[str, str]:
        return {
            "label_path": self.label_path,
            "product_id": self.product_id,
            "record_class": self.record_class,
            "start_time": self.start_time,
            "stop_time": self.stop_time,
            "volume": self.volume,
        }


def parse_sce1_cumulative_index(
    payload: bytes,
    *,
    expected_sha256: str,
    record_bytes: int = 201,
) -> tuple[CatalogRow, ...]:
    """Parse the seven fixed-width metadata columns without opening products."""

    _require_sha256(expected_sha256, name="expected_sha256")
    if sha256_bytes(payload) != expected_sha256:
        raise PDSContractError("cumulative-index SHA-256 mismatch")
    if record_bytes != 201 or not payload or len(payload) % record_bytes:
        raise PDSContractError("cumulative-index fixed-record layout mismatch")

    def text(raw: bytes, start_byte: int, length: int) -> str:
        try:
            value = raw[start_byte - 1 : start_byte - 1 + length].decode("ascii")
        except UnicodeDecodeError as exc:
            raise PDSContractError("cumulative-index field is not ASCII") from exc
        return value.strip().strip('"').strip()

    rows: list[CatalogRow] = []
    for offset in range(0, len(payload), record_bytes):
        raw = payload[offset : offset + record_bytes]
        if raw[-2:] != b"\r\n":
            raise PDSContractError("cumulative-index record lacks CRLF terminator")
        volume = text(raw, 2, 9)
        label_path = text(raw, 14, 50)
        product_id = text(raw, 67, 31)
        start_time = text(raw, 143, 19)
        stop_time = text(raw, 163, 19)
        record_class = next(
            (kind for kind in _DIRECT_CLASSES if f"/{kind}/" in label_path.upper()),
            "OTHER",
        )
        if not re.fullmatch(r"CORS_\d{4}", volume):
            raise PDSContractError("invalid volume identifier in cumulative index")
        if not _TIME_RE.fullmatch(start_time) or not _TIME_RE.fullmatch(stop_time):
            raise PDSContractError("invalid metadata time tag in cumulative index")
        if ".." in label_path or label_path.startswith(("/", "\\")):
            raise PDSContractError("unsafe label path in cumulative index")
        rows.append(
            CatalogRow(volume, label_path, product_id, start_time, stop_time, record_class)
        )
    return tuple(rows)


def select_direct_labels(
    rows: Iterable[CatalogRow], *, per_class: int = 6
) -> tuple[CatalogRow, ...]:
    """Select time-spanning label identities using metadata only.

    Each direct class is sorted independently.  ``per_class`` evenly spaced
    integer indices, including both endpoints, are selected.  Ties are broken
    by the complete identity tuple and never by measurement values.
    """

    if not 1 <= per_class <= 32:
        raise PDSContractError("per_class must be in [1, 32]")
    rows = tuple(rows)
    selected: list[CatalogRow] = []
    for record_class in _DIRECT_CLASSES:
        candidates = sorted(
            (row for row in rows if row.record_class == record_class),
            key=lambda row: (
                row.start_time,
                row.stop_time,
                row.volume,
                row.label_path,
                row.product_id,
            ),
        )
        if len(candidates) < per_class:
            raise PDSContractError(f"insufficient {record_class} metadata candidates")
        denominator = per_class - 1
        indices = (
            [0]
            if denominator == 0
            else [(i * (len(candidates) - 1)) // denominator for i in range(per_class)]
        )
        if len(set(indices)) != len(indices):
            raise PDSContractError("selection indices are not unique")
        selected.extend(candidates[index] for index in indices)
    return tuple(selected)


@dataclass
class PVLObject:
    kind: str
    attributes: dict[str, str] = field(default_factory=dict)
    children: list[PVLObject] = field(default_factory=list)


def parse_pds3_label(payload: bytes, *, expected_sha256: str) -> PVLObject:
    """Parse only the structural PDS3/PVL subset needed for binary layouts."""

    _require_sha256(expected_sha256, name="expected_sha256")
    if sha256_bytes(payload) != expected_sha256:
        raise PDSContractError("PDS label SHA-256 mismatch")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PDSContractError("PDS3 label must be ASCII") from exc
    if "PDS_VERSION_ID" not in text or "PDS3" not in text:
        raise PDSContractError("not a PDS3 label")

    root = PVLObject("ROOT")
    stack = [root]
    for line in text.splitlines():
        match = _KEY_VALUE_RE.match(line)
        if match is None:
            continue
        key, raw_value = match.groups()
        if key == "OBJECT":
            kind = _unquote(raw_value)
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", kind):
                raise PDSContractError("invalid PDS object kind")
            node = PVLObject(kind)
            stack[-1].children.append(node)
            stack.append(node)
            continue
        if key == "END_OBJECT":
            kind = _unquote(raw_value)
            if len(stack) == 1 or stack[-1].kind != kind:
                raise PDSContractError("PDS object nesting mismatch")
            stack.pop()
            continue
        if key in stack[-1].attributes:
            raise PDSContractError(f"duplicate PDS attribute {key}")
        stack[-1].attributes[key] = raw_value
    if len(stack) != 1:
        raise PDSContractError("unterminated PDS object")
    return root


def _objects(root: PVLObject, kind: str) -> list[PVLObject]:
    result: list[PVLObject] = []
    pending = list(root.children)
    while pending:
        node = pending.pop(0)
        if node.kind == kind:
            result.append(node)
        pending[0:0] = node.children
    return result


def _attribute(node: PVLObject, key: str) -> str:
    try:
        return node.attributes[key]
    except KeyError as exc:
        raise PDSContractError(f"missing {key} in {node.kind}") from exc


def normalized_table_layout(root: PVLObject, *, object_kind: str) -> dict[str, Any]:
    """Validate and normalize one TABLE/TDF object for deterministic hashing."""

    candidates = _objects(root, object_kind)
    if len(candidates) != 1:
        raise PDSContractError(f"expected exactly one {object_kind} object")
    table = candidates[0]
    record_bytes = _integer(_attribute(root, "RECORD_BYTES"), name="RECORD_BYTES")
    row_bytes = _integer(_attribute(table, "ROW_BYTES"), name="ROW_BYTES")
    suffix_bytes = _integer(table.attributes.get("ROW_SUFFIX_BYTES", "0"), name="ROW_SUFFIX_BYTES")
    if row_bytes + suffix_bytes != record_bytes:
        raise PDSContractError("table row plus suffix does not equal record size")
    fields: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for column in (child for child in table.children if child.kind == "COLUMN"):
        name = _unquote(_attribute(column, "NAME"))
        start_byte = _integer(_attribute(column, "START_BYTE"), name=f"{name}.START_BYTE")
        byte_length = _integer(_attribute(column, "BYTES"), name=f"{name}.BYTES")
        data_type = _unquote(_attribute(column, "DATA_TYPE"))
        unit = _unquote(column.attributes.get("UNIT", "N/A"))
        if start_byte < 1 or byte_length < 1 or start_byte - 1 + byte_length > row_bytes:
            raise PDSContractError(f"field {name} is outside the row")
        interval = (start_byte - 1, start_byte - 1 + byte_length)
        if any(interval[0] < end and start < interval[1] for start, end in occupied):
            raise PDSContractError(f"field {name} overlaps another field")
        occupied.append(interval)
        bit_fields: list[dict[str, Any]] = []
        for bit_column in (child for child in column.children if child.kind == "BIT_COLUMN"):
            bit_name = _unquote(_attribute(bit_column, "NAME"))
            start_bit = _integer(
                _attribute(bit_column, "START_BIT"), name=f"{name}.{bit_name}.START_BIT"
            )
            bits = _integer(_attribute(bit_column, "BITS"), name=f"{name}.{bit_name}.BITS")
            bit_type = _unquote(_attribute(bit_column, "BIT_DATA_TYPE"))
            if bit_type not in _SUPPORTED_BITS:
                raise PDSContractError(f"unsupported bit type {bit_type}")
            if start_bit < 1 or bits < 1 or start_bit - 1 + bits > byte_length * 8:
                raise PDSContractError(f"bit field {bit_name} is outside parent field")
            bit_fields.append(
                {
                    "bits": bits,
                    "data_type": bit_type,
                    "name": bit_name,
                    "start_bit": start_bit,
                }
            )
        if bit_fields:
            bit_intervals = sorted(
                (item["start_bit"] - 1, item["start_bit"] - 1 + item["bits"])
                for item in bit_fields
            )
            if any(a_end > b_start for (_, a_end), (b_start, _) in pairwise(bit_intervals)):
                raise PDSContractError(f"overlapping bit fields in {name}")
        elif data_type not in _SUPPORTED_SCALARS:
            raise PDSContractError(f"unsupported scalar type {data_type}")
        item_count = _integer(column.attributes.get("ITEMS", "1"), name=f"{name}.ITEMS")
        item_bytes = _integer(
            column.attributes.get("ITEM_BYTES", str(byte_length)), name=f"{name}.ITEM_BYTES"
        )
        item_offset = _integer(
            column.attributes.get("ITEM_OFFSET", str(item_bytes)), name=f"{name}.ITEM_OFFSET"
        )
        if item_count > 1 and (item_bytes < 1 or item_offset < item_bytes):
            raise PDSContractError(f"invalid item layout in {name}")
        if item_count > 1 and (item_count - 1) * item_offset + item_bytes > byte_length:
            raise PDSContractError(f"item layout exceeds field {name}")
        fields.append(
            {
                "bit_fields": bit_fields,
                "byte_length": byte_length,
                "data_type": data_type,
                "item_bytes": item_bytes,
                "item_count": item_count,
                "item_offset": item_offset,
                "name": name,
                "start_byte": start_byte,
                "unit": unit,
            }
        )
    if not fields:
        raise PDSContractError("table contains no fields")
    return {
        "endianness": "MSB_first",
        "fields": fields,
        "object_kind": object_kind,
        "record_bytes": record_bytes,
        "row_bytes": row_bytes,
        "row_suffix_bytes": suffix_bytes,
    }


def _twos_complement(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _decode_scalar(data: bytes, data_type: str) -> Any:
    if data_type == "CHARACTER":
        try:
            return data.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PDSContractError("CHARACTER field is not ASCII") from exc
    if data_type == "MSB_UNSIGNED_INTEGER":
        return int.from_bytes(data, "big", signed=False)
    if data_type == "MSB_INTEGER":
        return int.from_bytes(data, "big", signed=True)
    if data_type == "IEEE_REAL":
        if len(data) == 4:
            return struct.unpack(">f", data)[0]
        if len(data) == 8:
            return struct.unpack(">d", data)[0]
        raise PDSContractError("IEEE_REAL must be four or eight bytes")
    raise PDSContractError(f"unsupported scalar type {data_type}")


def decode_record(
    layout: Mapping[str, Any],
    payload: bytes,
    *,
    expected_sha256: str,
    requested_fields: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Decode an exact record and retain units/type/endian metadata."""

    _require_sha256(expected_sha256, name="expected_sha256")
    if sha256_bytes(payload) != expected_sha256:
        raise PDSContractError("record SHA-256 mismatch")
    if len(payload) != int(layout["record_bytes"]):
        raise PDSContractError("record truncation or length mismatch")
    wanted = None if requested_fields is None else set(requested_fields)
    decoded: dict[str, dict[str, Any]] = {}
    for field_spec in layout["fields"]:
        name = str(field_spec["name"])
        start = int(field_spec["start_byte"]) - 1
        end = start + int(field_spec["byte_length"])
        raw = payload[start:end]
        bit_fields = field_spec["bit_fields"]
        if bit_fields:
            bit_string = int.from_bytes(raw, "big", signed=False)
            total_bits = len(raw) * 8
            for bit_spec in bit_fields:
                bit_name = str(bit_spec["name"])
                if wanted is not None and bit_name not in wanted:
                    continue
                bits = int(bit_spec["bits"])
                shift = total_bits - (int(bit_spec["start_bit"]) - 1 + bits)
                value = (bit_string >> shift) & ((1 << bits) - 1)
                if bit_spec["data_type"] == "MSB_INTEGER":
                    value = _twos_complement(value, bits)
                if bit_name in decoded:
                    raise PDSContractError(f"ambiguous duplicate decoded field {bit_name}")
                decoded[bit_name] = {
                    "bits": bits,
                    "data_type": bit_spec["data_type"],
                    "endianness": "MSB_first",
                    "unit": field_spec["unit"],
                    "value": value,
                }
            continue
        if wanted is not None and name not in wanted:
            continue
        if int(field_spec["item_count"]) == 1:
            value = _decode_scalar(raw, str(field_spec["data_type"]))
        else:
            value = []
            for index in range(int(field_spec["item_count"])):
                item_start = index * int(field_spec["item_offset"])
                item_end = item_start + int(field_spec["item_bytes"])
                value.append(_decode_scalar(raw[item_start:item_end], str(field_spec["data_type"])))
        decoded[name] = {
            "byte_length": int(field_spec["byte_length"]),
            "data_type": field_spec["data_type"],
            "endianness": "MSB_first",
            "unit": field_spec["unit"],
            "value": value,
        }
    if wanted is not None and wanted != set(decoded):
        missing = sorted(wanted - set(decoded))
        raise PDSContractError(f"requested fields missing or ambiguous: {missing}")
    return decoded


def _tdf_utc_tag(decoded: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    year = 1900 + int(decoded["YEAR"]["value"])
    day = int(decoded["DOY"]["value"])
    hour = int(decoded["HOUR"]["value"])
    minute = int(decoded["MINUTE"]["value"])
    second = int(decoded["SECOND"]["value"])
    if not 1900 <= year <= 2099 or not 1 <= day <= 366:
        raise PDSContractError("invalid TDF UTC year/day tag")
    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 60:
        raise PDSContractError("invalid TDF UTC clock tag")
    return {
        "day_of_year": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "time_system": "UTC",
        "year": year,
    }


def decode_tdf_tracking_record(
    layout: Mapping[str, Any], payload: bytes, *, expected_sha256: str
) -> dict[str, Any]:
    """Decode the exact unsigned MSB TDF tracking time/type identity fields."""

    requested = [
        "DOY",
        "DOWNLINK FREQUENCY BAND",
        "HOUR",
        "MINUTE",
        "RECORD FORMAT",
        "RECORD TYPE",
        "SAMPLE DATA TYPE ID",
        "SECOND",
        "STATION ID",
        "YEAR",
    ]
    decoded = decode_record(
        layout, payload, expected_sha256=expected_sha256, requested_fields=requested
    )
    return {
        "fields": decoded,
        "signedness": "label_declared_per_field",
        "utc_time_tag": _tdf_utc_tag(decoded),
    }


def unpack_rsr_iq_words(
    sample_words: bytes,
    *,
    sample_resolution_bits: int,
    maximum_output_samples: int,
) -> tuple[dict[str, int], ...]:
    """Unpack Q-then-I words, LSB-to-MSB time order, and RSR 2*k+1 bias."""

    if sample_resolution_bits not in {1, 2, 4, 8, 16}:
        raise PDSContractError("unsupported RSR sample resolution")
    if not sample_words or len(sample_words) % 4:
        raise PDSContractError("truncated RSR sample word")
    if not 1 <= maximum_output_samples <= 1_000_000:
        raise PDSContractError("maximum_output_samples is outside bounded range")
    mask = (1 << sample_resolution_bits) - 1
    per_word = 16 // sample_resolution_bits
    result: list[dict[str, int]] = []
    for offset in range(0, len(sample_words), 4):
        q_word = int.from_bytes(sample_words[offset : offset + 2], "big", signed=False)
        i_word = int.from_bytes(sample_words[offset + 2 : offset + 4], "big", signed=False)
        for index in range(per_word):
            shift = index * sample_resolution_bits
            q_raw = _twos_complement((q_word >> shift) & mask, sample_resolution_bits)
            i_raw = _twos_complement((i_word >> shift) & mask, sample_resolution_bits)
            if len(result) < maximum_output_samples:
                result.append(
                    {
                        "i": 2 * i_raw + 1,
                        "i_raw_twos_complement": i_raw,
                        "q": 2 * q_raw + 1,
                        "q_raw_twos_complement": q_raw,
                    }
                )
    return tuple(result)


def decode_rsr_record(
    layout: Mapping[str, Any],
    payload: bytes,
    *,
    expected_sha256: str,
    maximum_output_samples: int = 32,
) -> dict[str, Any]:
    """Decode and cross-check one label-bound RSR record."""

    names = {
        "DATA CHDO LENGTH",
        "SAMPLE RATE",
        "SAMPLE RESOLUTION",
        "SFDU DAY OF YEAR",
        "SFDU SECOND",
        "SFDU YEAR",
    }
    decoded = decode_record(
        layout, payload, expected_sha256=expected_sha256, requested_fields=sorted(names)
    )
    resolution = int(decoded["SAMPLE RESOLUTION"]["value"])
    rate_ksps = int(decoded["SAMPLE RATE"]["value"])
    sample_field = next(item for item in layout["fields"] if item["name"] == "SAMPLE WORDS")
    start = int(sample_field["start_byte"]) - 1
    sample_words = payload[start : start + int(sample_field["byte_length"])]
    if int(decoded["DATA CHDO LENGTH"]["value"]) != len(sample_words):
        raise PDSContractError("RSR data-CHDO length disagrees with label layout")
    expected_samples = len(sample_words) // 4 * (16 // resolution)
    if rate_ksps * 1000 != expected_samples:
        raise PDSContractError("RSR sample rate, resolution, and record layout disagree")
    samples = unpack_rsr_iq_words(
        sample_words,
        sample_resolution_bits=resolution,
        maximum_output_samples=maximum_output_samples,
    )
    second = float(decoded["SFDU SECOND"]["value"])
    if not math.isfinite(second) or not 0.0 <= second <= 86400.0:
        raise PDSContractError("invalid RSR UTC second-of-day")
    day = int(decoded["SFDU DAY OF YEAR"]["value"])
    year = int(decoded["SFDU YEAR"]["value"])
    if not 1900 <= year <= 3000 or not 1 <= day <= 366:
        raise PDSContractError("invalid RSR UTC year/day tag")
    return {
        "declared_sample_count": expected_samples,
        "endianness": "MSB_first_words_LSB_to_MSB_sample_time_order",
        "output_units": "dimensionless_odd_integer_ADC_code_after_2k_plus_1_bias",
        "sample_rate_kilosample_per_second": rate_ksps,
        "sample_resolution_bits": resolution,
        "samples": samples,
        "signedness": "twos_complement_per_sample_before_2k_plus_1_bias",
        "utc_time_tag": {"day_of_year": day, "second_of_day": second, "year": year},
    }


def _validate_hash_binding(repo_root: Path, binding: Mapping[str, Any]) -> None:
    path = repo_root / str(binding["path"])
    expected = str(binding["file_sha256"])
    _require_sha256(expected, name=f"{binding['path']}.file_sha256")
    if not path.is_file() or sha256_file(path) != expected:
        raise PDSContractError(f"local binding mismatch: {binding['path']}")


def build_readiness_artifact(repo_root: Path, config_path: Path) -> dict[str, Any]:
    """Build a portable parser-readiness record without network or target access."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "sigma-solar-pds-parser-readiness-config-1.0":
        raise PDSContractError("unexpected parser-readiness config schema")
    for binding in config["local_bindings"].values():
        _validate_hash_binding(repo_root, binding)
    eligibility = config["data_eligibility"]
    expected_seals = {
        "candidate_use_authorized": False,
        "dark_matter_or_halo_inputs": False,
        "observational_data_opened": False,
        "paid_llm_calls": False,
        "redshift_distance_inputs": False,
        "target_values_accessed": False,
    }
    if eligibility != expected_seals:
        raise PDSContractError("data-eligibility seals differ from the closed contract")
    selection = config["metadata_selection"]
    if selection["selection_basis"] != "hash_verified_fixed_width_catalog_metadata_only":
        raise PDSContractError("selection basis is not metadata-only")
    selected = selection["selected_label_identities"]
    class_counts = {
        kind: sum(item["record_class"] == kind for item in selected) for kind in _DIRECT_CLASSES
    }
    if len(selected) != 12 or class_counts != {"TDF": 6, "RSR": 6}:
        raise PDSContractError("expected six TDF and six RSR label identities")
    if any(item.get("primary_record_sha256") is not None for item in selected):
        raise PDSContractError("primary record hash unexpectedly present")
    label_hashes = [item["label_sha256"] for item in selected]
    if len(set(label_hashes)) != 12:
        raise PDSContractError("selected label hashes are not unique")
    for digest in label_hashes:
        _require_sha256(digest, name="selected label SHA-256")
    documentation = config["authoritative_documentation_sources"]
    for name, binding in documentation.items():
        _require_sha256(str(binding["sha256"]), name=f"{name}.sha256")
        if not str(binding["url"]).startswith("https://pds-geosciences.wustl.edu/"):
            raise PDSContractError("documentation source is not authoritative PDS HTTPS")

    parser_source_sha256 = sha256_file(repo_root / "src/sigma_theory_compiler/solar_pds_parser_readiness.py")
    parser_contract = {
        "authoritative_documentation_sources": config["authoritative_documentation_sources"],
        "normalized_layout_roots": config["normalized_layout_roots"],
        "parser_source_sha256": parser_source_sha256,
        "selected_label_hashes": label_hashes,
        "synthetic_and_known_answer_vectors": config["verification_vectors"],
    }
    tdf_verification_sha256 = canonical_sha256({"class": "ATDF_TDF", **parser_contract})
    rsr_verification_sha256 = canonical_sha256({"class": "RSR", **parser_contract})
    filled = {
        "verified_ATDF_TDF_parser_sha256": tdf_verification_sha256,
        "verified_RSR_parser_sha256": rsr_verification_sha256,
    }
    absent = [
        "registered_real_source_interval_instantiation_certificate_sha256",
        "selected_primary_file_root_sha256",
        "selected_PDS_label_and_calibration_file_root_sha256",
        "raw_to_calibrated_transform_and_covariance_implementation_sha256",
        "tracking_session_split_commitment_sha256",
        "training_only_initial_state_checkpoint_sha256",
        "reviewed_candidate_solar_evaluator_descriptor_sha256",
    ]
    artifact: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "authoritative_documentation_root_sha256": canonical_sha256(documentation),
        "data_eligibility": eligibility,
        "descriptor_registration_status": "blocked_unregistered",
        "filled_registration_field_count": len(filled),
        "filled_registration_fields": filled,
        "interpretation": (
            "Detached-label metadata and parser mechanics are frozen and verified against "
            "authoritative documentation examples plus synthetic negative controls. No SCE1 "
            "primary record or target value was opened. Production remains ineligible until "
            "the seven absent hashes, including actual primary/calibration roots, exist."
        ),
        "metadata_selection": {
            "catalog_label_sha256": selection["catalog_label_sha256"],
            "catalog_table_sha256": selection["catalog_table_sha256"],
            "selected_identity_count": len(selected),
            "selected_identity_root_sha256": canonical_sha256(selected),
            "selected_label_file_root_sha256": canonical_sha256(
                [{"path": item["label_path"], "sha256": item["label_sha256"]} for item in selected]
            ),
            "detached_label_metadata_access_count": len(selected),
            "primary_record_access_count": 0,
            "target_values_accessed": False,
        },
        "normalized_layout_roots": config["normalized_layout_roots"],
        "observational_authorization": False,
        "parser_source_sha256": parser_source_sha256,
        "remaining_registration_field_count": len(absent),
        "remaining_registration_fields": absent,
        "schema_version": "sigma-solar-pds-parser-readiness-1.0",
        "source_bindings": config["local_bindings"],
        "status": "parser_ready_labels_selected_primary_records_sealed",
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact


def write_readiness_artifact(repo_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = build_readiness_artifact(repo_root, config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return artifact
