"""Source-free release contract for one future Lane-9 development run.

The CLI in this module never opens a CF4, VAST, mask, or Pantheon+ source.  Its
parsers accept caller-supplied bytes so that the complete future decode and
score surface can be audited on synthetic records before a separate receipt
authorizes any scientific access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from . import open_gravity_void_correlation_executor_contract_v3 as executor_v3
from . import open_gravity_void_correlation_ids_partition_v1 as ids_v1
from . import open_gravity_void_geometry_source_completion_v2 as geometry_v2
from . import open_gravity_void_geometry_source_completion_v3 as geometry_v3
from . import open_gravity_void_gravitational_load_v4 as law_v4

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_correlation_development_release_v1.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_correlation_development_release_v1.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_correlation_development_release_v1.py"
OUTPUT_PATH = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v1/receipt.json"
_CONFIG_RAW_SHA256 = "b46884194ead305eb1bf8e26c97a6a7b6ece9b820e11d09cf4beb8f000df7e79"
_CONFIG_CONTENT_SHA256 = "6d3af19ffc4df56fe2462004498325d46daa3e63a6f1b5d592224bd1a7b9dc72"
_MODULE_SEMANTIC_SHA256 = "4d3e5f400677df75edad746173c4433d54ce752a789e0777919c1b76fd566a48"
_TEST_RAW_SHA256 = "e1d88a1d252705878be85e9373e435f99086b0ed263d57b09b7f8eab3cc3ce8f"

_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}
_I_TOKEN = re.compile(rb" *[+-]?[0-9]+ *\Z")
_F_TOKEN = re.compile(rb" *[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+) *\Z")


class DevelopmentReleaseV1Error(RuntimeError):
    """Fail-closed contract violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentReleaseV1Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if any(stripped.startswith(f'{name} = "') for name in _SELF_CONSTANTS):
            continue
        lines.append(line)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def validate_code_pins() -> None:
    _require(MODULE_PATH.is_file(), "module missing")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test pin drift")


def canonical_relative_path(value: str) -> Path:
    _require(isinstance(value, str) and bool(value), "empty bound path")
    _require("\\" not in value and not re.match(r"^[A-Za-z]:", value), "non-POSIX bound path")
    _require(all(part not in {"", ".", ".."} for part in value.split("/")), "unnormalized bound path")
    pure = PurePosixPath(value)
    _require(not pure.is_absolute(), "absolute bound path")
    root = REPO_ROOT.resolve(strict=True)
    candidate = root / Path(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        _require(cursor.exists(), "bound path missing")
        _require(not cursor.is_symlink(), "symlink in bound path")
    target = candidate.resolve(strict=True)
    _require(target != root and root in target.parents, "bound path escapes repository")
    _require(target.is_file(), "bound path is not a regular file")
    return target


def validate_source_path_strings(config: Mapping[str, Any]) -> None:
    """Validate source path grammar without resolving, stating, or opening a source."""
    for source in config["sources"].values():
        value = source["path"]
        _require(isinstance(value, str) and value and "\\" not in value, "invalid source path")
        _require(all(part not in {"", ".", ".."} for part in value.split("/")), "unsafe source path")
        pure = PurePosixPath(value)
        _require(not pure.is_absolute(), "unsafe source path")


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config content drift")
    _require(value["status"] == "DRAFT_SOURCE_FREE_DEVELOPMENT_RELEASE_CONTRACT_AWAIT_INDEPENDENT_AUDIT", "config status drift")
    validate_source_path_strings(value)
    return value


def _load_json_binding(section: Mapping[str, Any], *, self_hashed: bool = False) -> dict[str, Any]:
    path = canonical_relative_path(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], f"raw binding drift: {section['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if "content_sha256" in section:
        _require(value.get("content_sha256", content_sha256(value)) == section["content_sha256"], f"content binding drift: {section['path']}")
    if self_hashed:
        _require(value.get("content_sha256") == _self_hash(value), f"receipt self-hash drift: {section['path']}")
    if "status" in section:
        _require(value.get("status") == section["status"], f"status binding drift: {section['path']}")
    if "decision" in section:
        _require(value.get("decision") == section["decision"], f"decision binding drift: {section['path']}")
    return value


def validate_release_chain(config: Mapping[str, Any]) -> dict[str, str]:
    chain = config["release_chain"]
    executor = chain["executor_v3"]
    executor_config = _load_json_binding(executor["config"])
    _require(content_sha256(executor_config) == executor["config"]["content_sha256"], "executor config content drift")
    executor_module = canonical_relative_path(executor["module"]["path"])
    _require(file_sha256(executor_module) == executor["module"]["raw_sha256"], "executor module raw drift")
    _require(executor_v3.module_semantic_sha256(executor_module) == executor["module"]["semantic_sha256"], "executor module semantic drift")
    _require(file_sha256(canonical_relative_path(executor["test"]["path"])) == executor["test"]["raw_sha256"], "executor test drift")
    executor_receipt = _load_json_binding(executor["receipt"], self_hashed=True)
    executor_audit = _load_json_binding(executor["independent_audit"], self_hashed=True)

    ids = chain["ids_v1"]
    ids_config = _load_json_binding(ids["config"])
    _require(content_sha256(ids_config) == ids["config"]["content_sha256"], "IDs config content drift")
    ids_module = canonical_relative_path(ids["module"]["path"])
    _require(file_sha256(ids_module) == ids["module"]["raw_sha256"], "IDs module raw drift")
    _require(ids_v1.module_semantic_sha256(ids_module) == ids["module"]["semantic_sha256"], "IDs module semantic drift")
    _require(file_sha256(canonical_relative_path(ids["test"]["path"])) == ids["test"]["raw_sha256"], "IDs test drift")
    ledger_path = canonical_relative_path(ids["identifier_ledger"]["path"])
    _require(file_sha256(ledger_path) == ids["identifier_ledger"]["raw_sha256"], "ID ledger raw drift")
    ledger_values = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    _require(len(ledger_values) == ids["identifier_ledger"]["rows"], "ID ledger row-count drift")
    _require(content_sha256(ledger_values) == ids["identifier_ledger"]["content_sha256"], "ID ledger content drift")
    for name in ("failure_ledger", "summary"):
        value = _load_json_binding(ids[name])
        _require(content_sha256(value) == ids[name]["content_sha256"], f"IDs {name} content drift")
    ids_receipt = _load_json_binding(ids["receipt"], self_hashed=True)
    ids_audit = _load_json_binding(ids["independent_audit"], self_hashed=True)
    _require(ids_audit["authorization_scope"]["allowed_next_stage"] == "DEVELOPMENT_RELEASE_ONLY", "ID audit scope drift")
    _require(ids_audit["zero_scientific_access_statement"], "ID audit access statement missing")
    return {
        "executor_receipt_content_sha256": executor_receipt["content_sha256"],
        "executor_audit_content_sha256": executor_audit["content_sha256"],
        "ids_receipt_content_sha256": ids_receipt["content_sha256"],
        "ids_audit_content_sha256": ids_audit["content_sha256"],
        "identifier_ledger_raw_sha256": ids["identifier_ledger"]["raw_sha256"],
    }


def validate_schema(fields: Sequence[Mapping[str, Any]], record_length: int) -> None:
    names: set[str] = set()
    previous_end = 0
    for field in fields:
        name = str(field["name"])
        start, end = int(field["start"]), int(field["end"])
        _require(name not in names and previous_end < start <= end <= record_length, "invalid schema order")
        match = re.fullmatch(r"([AIF])([1-9][0-9]*)(?:\.([0-9]+))?", str(field["format"]))
        _require(match is not None and int(match.group(2)) == end - start + 1, "invalid schema format")
        _require((match.group(1) == "F") == (match.group(3) is not None), "invalid schema decimals")
        _require(field.get("required") is True, "all release fields must be required")
        names.add(name)
        previous_end = end


def _frame_payload(line: bytes, record_length: int) -> bytes:
    _require(isinstance(line, bytes), "record must be bytes")
    _require(line.endswith(b"\n"), "record lacks terminal LF")
    payload = line[:-1]
    if payload.endswith(b"\r"):
        payload = payload[:-1]
    _require(b"\r" not in payload and b"\n" not in payload, "embedded or repeated line ending")
    _require(len(payload) == record_length, "record length mismatch")
    return payload


def record_payload(line: bytes, record_length: int) -> bytes:
    payload = _frame_payload(line, record_length)
    _require(all(byte < 128 for byte in payload), "non-ASCII record")
    return payload


def parse_field(payload: bytes, field: Mapping[str, Any]) -> str | int | float:
    start, end = int(field["start"]) - 1, int(field["end"])
    _require(len(payload) >= end, "payload too short")
    token = payload[start:end]
    _require(all(byte < 128 for byte in token), "non-ASCII field")
    _require(not all(byte == 0x20 for byte in token), "required field missing")
    kind = str(field["format"])[0]
    if kind == "A":
        _require(all(0x20 <= byte <= 0x7E for byte in token), "nonprintable A field")
        return token.decode("ascii").rstrip(" ")
    _require(all(byte == 0x20 or 0x21 <= byte <= 0x7E for byte in token), "control byte in numeric field")
    if kind == "I":
        _require(_I_TOKEN.fullmatch(token) is not None, "invalid I token")
        return int(token.decode("ascii"))
    _require(kind == "F" and _F_TOKEN.fullmatch(token) is not None, "invalid F token")
    result = float(token.decode("ascii"))
    _require(math.isfinite(result), "nonfinite F token")
    return result


def parse_record(payload: bytes, fields: Sequence[Mapping[str, Any]]) -> dict[str, str | int | float]:
    return {str(field["name"]): parse_field(payload, field) for field in fields}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_cf4_development_record(
    line: bytes,
    ledger_entry: Mapping[str, Any],
    *,
    source_index: int,
    framed_start: int,
) -> dict[str, int | float]:
    """Verify the audited row and role before decoding any scientific field."""
    config = load_config()
    fields = config["fixed_width_schemas"]["CF4_TABLE4"]
    payload = _frame_payload(line, int(config["sources"]["CF4_TABLE4"]["record_length"]))
    line_ending_bytes = len(line) - len(payload)
    framed_end = framed_start + len(line)
    payload_end = framed_start + len(payload)
    _require(int(ledger_entry["source_index"]) == source_index, "source-index mismatch")
    _require(int(ledger_entry["framed_start"]) == framed_start, "framed-start mismatch")
    _require(int(ledger_entry["framed_end_exclusive"]) == framed_end, "framed-end mismatch")
    _require(int(ledger_entry["payload_start"]) == framed_start, "payload-start mismatch")
    _require(int(ledger_entry["payload_end_exclusive"]) == payload_end, "payload-end mismatch")
    _require(int(ledger_entry["line_ending_bytes"]) == line_ending_bytes, "line-ending mismatch")
    _require(_sha256_bytes(line) == ledger_entry["framed_raw_sha256"], "framed hash mismatch")
    _require(_sha256_bytes(payload) == ledger_entry["payload_raw_sha256"], "payload hash mismatch")
    _require(_sha256_bytes(payload[:7]) == ledger_entry["identifier_field_raw_sha256"], "identifier hash mismatch")
    _require(_sha256_bytes(payload[7:]) == ledger_entry["opaque_tail_raw_sha256"], "opaque-tail hash mismatch")
    identifier = parse_field(payload, fields[0])
    _require(isinstance(identifier, int) and identifier > 0, "invalid 1PGC")
    bucket, role = executor_v3.split_role(identifier)
    _require(identifier == int(ledger_entry["identifier"]), "identifier mismatch")
    _require(str(identifier) == ledger_entry["canonical_identifier"], "canonical identifier mismatch")
    _require(bucket == int(ledger_entry["bucket"]) and role == ledger_entry["role"], "split mismatch")
    _require(role == "development", "sealed role rejected before scientific decode")
    _require(all(byte < 128 for byte in payload), "non-ASCII development record")
    values = {"1PGC": identifier, "source_index": source_index}
    for field in fields[1:]:
        values[str(field["name"])] = parse_field(payload, field)
    _validate_cf4_domains(values)
    return values  # type: ignore[return-value]


def _validate_cf4_domains(row: Mapping[str, Any]) -> None:
    _require(int(row["1PGC"]) > 0, "nonpositive 1PGC")
    _require(math.isfinite(float(row["DMzp"])), "invalid DMzp")
    _require(math.isfinite(float(row["e_DMzp"])) and float(row["e_DMzp"]) > 0.0, "invalid e_DMzp")
    _require(math.isfinite(float(row["Dist"])) and float(row["Dist"]) > 0.0, "invalid Dist")
    _require(int(row["V3k"]) > -299792.458, "invalid V3k")
    _require(0.0 <= float(row["RAdeg"]) < 360.0, "invalid CF4 RA")
    _require(-90.0 <= float(row["DEdeg"]) < 90.0, "invalid CF4 Dec")
    geometry_v3.validate_cf4_distance(float(row["DMzp"]), float(row["Dist"]))


def parse_vast_table1_record(line: bytes) -> dict[str, str | int | float]:
    config = load_config()
    payload = record_payload(line, int(config["sources"]["VAST_TABLE1"]["record_length"]))
    row = parse_record(payload, config["fixed_width_schemas"]["VAST_TABLE1"])
    _require(float(row["Rad"]) > 0.0 and int(row["void"]) > 0, "invalid VAST table1 radius or void")
    _require(int(row["edge"]) in (0, 1), "invalid VAST edge")
    _require(float(row["s"]) >= 0.0 and float(row["Reff"]) > 0.0, "invalid VAST table1 distance")
    _require(0.0 <= float(row["RAdeg"]) < 360.0 and -90.0 <= float(row["DEdeg"]) <= 90.0, "invalid VAST table1 angle")
    return row


def parse_vast_table2_record(line: bytes) -> dict[str, str | int | float]:
    config = load_config()
    payload = record_payload(line, int(config["sources"]["VAST_TABLE2"]["record_length"]))
    row = parse_record(payload, config["fixed_width_schemas"]["VAST_TABLE2"])
    _require(float(row["Rad"]) > 0.0 and int(row["void"]) > 0, "invalid VAST table2 radius or void")
    return row


def prepare_vast_geometry(
    table1_rows: Sequence[Mapping[str, Any]],
    table2_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = executor_v3.validate_vast_duplicate_keys(
        [(str(row["Cosmo"]), int(row["void"]), int(row["edge"])) for row in table1_rows],
        [
            (str(row["Cosmo"]), int(row["void"]), float(row["x"]), float(row["y"]), float(row["z"]), float(row["Rad"]))
            for row in table2_rows
        ],
    )
    edge_by_key: dict[tuple[str, int], int] = {}
    for row in table1_rows:
        if str(row["Cosmo"]) != "Planck2018":
            continue
        key = (str(row["Cosmo"]), int(row["void"]))
        edge_by_key[key] = int(row["edge"])
        expected = geometry_v3.radec_to_xyz(float(row["RAdeg"]), float(row["DEdeg"]), float(row["s"]))
        observed = (float(row["x"]), float(row["y"]), float(row["z"]))
        _require(
            all(math.isclose(observed[index], float(expected[index]), rel_tol=1e-9, abs_tol=1e-8) for index in range(3)),
            "VAST table1 coordinate probe failed",
        )
    h = 0.674
    spheres: list[tuple[tuple[float, float, float], float]] = []
    for row in table2_rows:
        key = (str(row["Cosmo"]), int(row["void"]))
        if key[0] != "Planck2018" or edge_by_key[key] == 1:
            continue
        spheres.append(((float(row["x"]) / h, float(row["y"]) / h, float(row["z"]) / h), float(row["Rad"]) / h))
    _require(len(spheres) == summary["retained"], "retained-sphere count drift")
    return {"spheres_Mpc": spheres, "retained": summary["retained"], "excluded_edge": summary["excluded_edge"]}


def validate_mask(mask_u8: bytes) -> None:
    _require(isinstance(mask_u8, bytes) and len(mask_u8) == 64800, "invalid mask length")
    _require(all(value in (0, 1) for value in mask_u8), "invalid mask value")


def _planck_comoving_mpc(redshift: float) -> float:
    _require(math.isfinite(redshift) and redshift >= 0.0, "invalid Planck redshift")
    nodes, weights = np.polynomial.legendre.leggauss(64)
    sample = 0.5 * redshift * (nodes + 1.0)
    expansion = np.sqrt(0.315 * (1.0 + sample) ** 3 + 0.685)
    result = (299792.458 / 67.4) * 0.5 * redshift * float(np.sum(weights / expansion))
    _require(math.isfinite(result), "nonfinite comoving distance")
    return result


def luminosity_to_comoving_hinv(distance_luminosity_mpc: float) -> tuple[float, float]:
    """Cwd-independent byte-for-byte algorithmic copy of geometry-v3's frozen inversion."""
    distance = float(distance_luminosity_mpc)
    _require(math.isfinite(distance) and distance >= 0.0, "invalid luminosity distance")
    if distance == 0.0:
        return 0.0, 0.0
    low, high = 0.0, 0.2

    def luminosity(redshift: float) -> float:
        return (1.0 + redshift) * _planck_comoving_mpc(redshift)

    while luminosity(high) < distance:
        high *= 2.0
        _require(high <= 4.0, "distance outside frozen inversion")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if luminosity(middle) < distance:
            low = middle
        else:
            high = middle
    redshift = 0.5 * (low + high)
    return redshift, 0.674 * _planck_comoving_mpc(redshift)


def _intersect_intervals(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersections = [
        (max(a, c), min(b, d))
        for a, b in law_v4.union_intervals(left)
        for c, d in law_v4.union_intervals(right)
        if min(b, d) > max(a, c)
    ]
    return law_v4.union_intervals(intersections)


def derive_development_row(
    cf4_row: Mapping[str, Any],
    mask_u8: bytes,
    spheres_mpc: Sequence[tuple[Sequence[float], float]],
) -> dict[str, Any]:
    _validate_cf4_domains(cf4_row)
    validate_mask(mask_u8)
    identifier = int(cf4_row["1PGC"])
    bucket, role = executor_v3.split_role(identifier)
    _require(role == "development", "derive called on sealed role")
    z_d, distance_hinv = luminosity_to_comoving_hinv(float(cf4_row["Dist"]))
    h = 0.674
    distance_mpc = distance_hinv / h
    direction_array = geometry_v3.radec_to_xyz(float(cf4_row["RAdeg"]), float(cf4_row["DEdeg"]), 1.0)
    direction = tuple(float(value) for value in direction_array)
    angular_mask = geometry_v2.mask_contains(mask_u8, float(cf4_row["RAdeg"]), float(cf4_row["DEdeg"]))
    radial_limit_mpc = 332.3856506347656 / h
    observed_intervals = [(0.0, min(distance_mpc, radial_limit_mpc))] if angular_mask else []
    void_intervals = law_v4.ray_sphere_intervals(direction, distance_mpc, spheres_mpc)
    observed_void = _intersect_intervals(void_intervals, observed_intervals)
    partition = law_v4.path_partition(direction, distance_mpc, spheres_mpc, observed_intervals)
    l_void = float(partition["L_void"])
    _require(math.isclose(l_void, math.fsum(stop - start for start, stop in observed_void), rel_tol=0.0, abs_tol=1e-10), "void interval mismatch")
    maximum_chord = max((stop - start for start, stop in observed_void), default=0.0)
    observer_chord = observed_void[0][1] - observed_void[0][0] if observed_void and observed_void[0][0] == 0.0 else 0.0
    target_chord = observed_void[-1][1] - observed_void[-1][0] if observed_void and observed_void[-1][1] == distance_mpc else 0.0
    primary = angular_mask and 0.0 <= distance_hinv <= 332.3856506347656 and float(partition["L_unobserved"]) <= 1e-10
    reasons: list[str] = []
    if not angular_mask:
        reasons.append("ANGULAR_MASK_FALSE")
    if not 0.0 <= distance_hinv <= 332.3856506347656:
        reasons.append("OUTSIDE_RADIAL_MASK")
    if float(partition["L_unobserved"]) > 1e-10:
        reasons.append("UNOBSERVED_PATH")
    c = 299792.458
    observed = executor_v3.observed_log_redshift(float(cf4_row["V3k"]), c)
    baseline = math.log1p(z_d)
    response = observed - baseline
    sigma_distance = (math.log(10.0) / 5.0) * float(cf4_row["Dist"]) * float(cf4_row["e_DMzp"])
    expansion = math.sqrt(0.315 * (1.0 + z_d) ** 3 + 0.685)
    derivative = 1.0 / ((1.0 + z_d) * (distance_mpc + (1.0 + z_d) * c / (67.4 * expansion)))
    sigma_s = math.sqrt((derivative * sigma_distance) ** 2 + (250.0 / c) ** 2)
    nuisance_velocity = executor_v3.nuisance_velocity_design(distance_mpc, direction)
    nuisance_log = executor_v3.velocity_to_log_design(nuisance_velocity, c)
    values = [z_d, distance_hinv, distance_mpc, response, sigma_s, l_void, maximum_chord, observer_chord, target_chord, *direction, *nuisance_log]
    _require(all(math.isfinite(value) for value in values) and sigma_s > 0.0, "nonfinite derived row")
    return {
        "identifier": identifier,
        "source_index": int(cf4_row["source_index"]),
        "bucket": bucket,
        "role": role,
        "eligible_primary": primary,
        "reason_codes": reasons,
        "cf4": dict(cf4_row),
        "z_D": z_d,
        "D_path_hinv_Mpc": distance_hinv,
        "D_path_Mpc": distance_mpc,
        "direction": direction,
        "mask_pixel": angular_mask,
        "L_void_Mpc": l_void,
        "L_observed_matter_Mpc": float(partition["L_observed_matter"]),
        "L_unobserved_Mpc": float(partition["L_unobserved"]),
        "void_fraction": l_void / distance_mpc,
        "union_crossings": len(observed_void),
        "maximum_chord_Mpc": maximum_chord,
        "observer_endpoint_chord_Mpc": observer_chord,
        "target_endpoint_chord_Mpc": target_chord,
        "y": response,
        "sigma_s": sigma_s,
        "nuisance_design_log": nuisance_log,
        "law_column": l_void / c,
    }


def _primary_arrays(rows: Sequence[Mapping[str, Any]], exposure_key: str) -> tuple[list[float], ...]:
    primary = sorted((row for row in rows if bool(row["eligible_primary"])), key=lambda row: int(row["identifier"]))
    _require(len(primary) >= 10, "insufficient synthetic primary rows")
    identifiers = [int(row["identifier"]) for row in primary]
    executor_v3.validate_cf4_duplicate_keys(identifiers)
    return (
        [float(row["y"]) for row in primary],
        [float(row["sigma_s"]) for row in primary],
        [float(row["cf4"]["Dist"]) for row in primary],
        [float(row["D_path_Mpc"]) for row in primary],
        [tuple(float(value) for value in row["direction"]) for row in primary],
        [float(row[exposure_key]) for row in primary],
        identifiers,
    )


def score_exposure(rows: Sequence[Mapping[str, Any]], exposure_key: str) -> dict[str, Any]:
    y, sigma, _, path, directions, exposure, identifiers = _primary_arrays(rows, exposure_key)
    return executor_v3.profile_grid(y, sigma, path, directions, exposure, identifiers)


def profile_grid_details(rows: Sequence[Mapping[str, Any]], exposure_key: str = "L_void_Mpc") -> dict[str, Any]:
    """Run the exact v3 grid with one prepared design and retain all 161 rows."""
    y, sigma, _, path, directions, exposure, identifiers = _primary_arrays(rows, exposure_key)
    prepared = executor_v3._prepare_profile(y, sigma, path, directions, exposure, identifiers)
    profiles = [executor_v3._profile_prepared(prepared, delta) for delta in executor_v3.delta_h_grid()]
    raw_minimum = min(float(row["chi2"]) for row in profiles)
    tied = [
        row
        for row in profiles
        if float(row["chi2"]) - raw_minimum <= executor_v3._tie_tolerance(float(row["chi2"]), raw_minimum)
    ]
    best = min(tied, key=lambda row: (abs(float(row["delta_H"])), float(row["delta_H"])))
    null = profiles[80]
    _require(float(null["delta_H"]) == 0.0, "grid null index drift")
    delta_chi2 = float(null["chi2"]) - float(best["chi2"])
    if delta_chi2 < 0.0:
        _require(
            abs(delta_chi2) <= executor_v3._tie_tolerance(float(null["chi2"]), float(best["chi2"])),
            "negative delta-chi-square",
        )
        delta_chi2 = 0.0
    statistic = delta_chi2 if float(best["delta_H"]) > 0.0 else 0.0
    summary = {
        "best_delta_H": float(best["delta_H"]),
        "best_chi2": float(best["chi2"]),
        "null_chi2": float(null["chi2"]),
        "delta_chi2": delta_chi2,
        "one_sided_statistic": statistic,
        "tied_delta_H": [float(row["delta_H"]) for row in tied],
    }
    return {"summary": summary, "profiles": profiles}


def score_countermodels(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    primary = score_exposure(rows, "L_void_Mpc")
    return {
        "PRIMARY_UNION_VOID_PATH": primary,
        "C00_FLRW_RADIAL_BULK_SHEAR_NULL": {"delta_chi2": 0.0, "best_delta_H": 0.0, "null_chi2": primary["null_chi2"]},
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID": score_exposure(rows, "observer_endpoint_chord_Mpc"),
        "C02_TARGET_ENDPOINT_LOCAL_VOID": score_exposure(rows, "target_endpoint_chord_Mpc"),
        "C03_SINGLE_DOMINANT_VOID": score_exposure(rows, "maximum_chord_Mpc"),
    }


def _permutation_reference(rows: Sequence[Mapping[str, Any]], count: int) -> dict[str, Any]:
    y, sigma, luminosity, path, directions, exposure, identifiers = _primary_arrays(rows, "L_void_Mpc")
    return executor_v3.synthetic_permutation_test(
        y,
        sigma,
        luminosity,
        path,
        directions,
        exposure,
        identifiers,
        count,
        seed=902104729,
    )


def development_permutation_test(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(np.__version__ == "2.2.6", "NumPy version drift")
    return _permutation_reference(rows, 10000)


def classify_development(
    primary: Mapping[str, Any],
    permutation: Mapping[str, Any],
    countermodels: Mapping[str, Mapping[str, Any]],
    eligible_count: int,
) -> dict[str, Any]:
    _require(eligible_count >= 500, "development eligible minimum failed")
    best = float(primary["best_delta_H"])
    statistic = float(primary["one_sided_statistic"])
    p_value = float(permutation["p_value"])
    request_validation = best > 0.0 and statistic > 0.0 and p_value <= 0.01
    comparator_max = max(float(countermodels[key]["delta_chi2"]) for key in (
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID",
        "C02_TARGET_ENDPOINT_LOCAL_VOID",
        "C03_SINGLE_DOMINANT_VOID",
    ))
    path_specific = request_validation and float(primary["delta_chi2"]) >= comparator_max + 2.0
    if request_validation:
        empirical = "PATH_ACCUMULATION_SPECIFIC" if path_specific else "COUNTERMODEL_DEGENERATE_ANOMALY"
    elif best > 0.0 and p_value <= 0.05:
        empirical = "DIRECTIONAL_NEAR_MISS"
    elif best < 0.0:
        empirical = "OPPOSITE_SIGN_COUNTEREXAMPLE"
    else:
        empirical = "NO_DIRECTIONAL_SIGNAL"
    return {
        "request_validation": request_validation,
        "empirical_label": empirical,
        "path_accumulation_specific": path_specific,
        "grid_boundary": best in (-20.0, 20.0),
        "comparator_max_delta_chi2": comparator_max,
        "physics_veto_applied": False,
    }


def _hex(value: float) -> str:
    result = float(value)
    _require(math.isfinite(result), "cannot encode nonfinite float")
    return result.hex()


def development_ledger_rows(rows: Sequence[Mapping[str, Any]], profile_details: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary = sorted((row for row in rows if bool(row["eligible_primary"])), key=lambda row: int(row["identifier"]))
    if primary:
        profiles = list(profile_details["profiles"])
        _require(len(profiles) == 161 and float(profiles[80]["delta_H"]) == 0.0, "invalid retained profile grid")
        best_delta_h = float(profile_details["summary"]["best_delta_H"])
        best_rows = [row for row in profiles if float(row["delta_H"]) == best_delta_h]
        _require(len(best_rows) == 1, "best profile row missing")
        null = profiles[80]
        best = best_rows[0]
        null_beta = [float(value) for value in null["beta"]]
        best_beta = [float(value) for value in best["beta"]]
    else:
        best_delta_h, null_beta, best_beta = 0.0, [], []
    output: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["identifier"])):
        eligible = bool(row["eligible_primary"])
        record: dict[str, Any] = {
            "identifier": int(row["identifier"]),
            "source_index": int(row["source_index"]),
            "bucket": int(row["bucket"]),
            "role": str(row["role"]),
            "eligibility": "PRIMARY" if eligible else "PARTIAL_MASK_EXCLUDED",
            "reason_codes": list(row["reason_codes"]),
            "parsed_development_values_hex": {
                key: (_hex(value) if isinstance(value, float) else value)
                for key, value in row["cf4"].items()
            },
            "z_D_hex": _hex(row["z_D"]),
            "D_path_Mpc_hex": _hex(row["D_path_Mpc"]),
            "direction_hex": [_hex(value) for value in row["direction"]],
            "mask_pixel": bool(row["mask_pixel"]),
            "geometry_hex": {
                key: _hex(row[key])
                for key in (
                    "L_void_Mpc",
                    "L_observed_matter_Mpc",
                    "L_unobserved_Mpc",
                    "void_fraction",
                    "maximum_chord_Mpc",
                    "observer_endpoint_chord_Mpc",
                    "target_endpoint_chord_Mpc",
                )
            },
            "union_crossings": int(row["union_crossings"]),
            "y_hex": _hex(row["y"]),
            "sigma_s_hex": _hex(row["sigma_s"]),
            "nuisance_design_hex": [_hex(value) for value in row["nuisance_design_log"]],
            "law_column_hex": _hex(row["law_column"]),
            "null_prediction_hex": None,
            "primary_prediction_hex": None,
            "null_residual_hex": None,
            "primary_residual_hex": None,
        }
        if eligible:
            design = [float(value) for value in row["nuisance_design_log"]]
            null_prediction = math.fsum(design[index] * null_beta[index] for index in range(9))
            primary_prediction = float(best_delta_h) * float(row["law_column"]) + math.fsum(
                design[index] * best_beta[index] for index in range(9)
            )
            record.update({
                "null_prediction_hex": _hex(null_prediction),
                "primary_prediction_hex": _hex(primary_prediction),
                "null_residual_hex": _hex(float(row["y"]) - null_prediction),
                "primary_residual_hex": _hex(float(row["y"]) - primary_prediction),
            })
        record["row_leaf_sha256"] = content_sha256(record)
        output.append(record)
    return output


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _hex_tree(value: Any) -> Any:
    if isinstance(value, float):
        return _hex(value)
    if isinstance(value, dict):
        return {str(key): _hex_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hex_tree(item) for item in value]
    return value


def _leaf_root(ledger: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in ledger:
        leaf = str(row["row_leaf_sha256"])
        _require(re.fullmatch(r"[0-9a-f]{64}", leaf) is not None, "invalid ledger leaf")
        digest.update(bytes.fromhex(leaf))
    return digest.hexdigest()


def _validated_failures(failures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in failures:
        _require(
            set(row) == {"stage", "reason_code", "access_counts", "authorized_development_ids"},
            "unsanitized failure shape",
        )
        sanitized = sanitized_failure_record(
            str(row["stage"]),
            str(row["reason_code"]),
            row["access_counts"],
            row["authorized_development_ids"],
        )
        _require(dict(row) == sanitized, "noncanonical failure record")
        result.append(sanitized)
    return result


def assemble_development_artifacts(
    rows: Sequence[Mapping[str, Any]],
    profile_details: Mapping[str, Any],
    permutation: Mapping[str, Any],
    countermodels: Mapping[str, Mapping[str, Any]],
    access_counts: Mapping[str, int],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    """Create every frozen future output except the audit-bound final receipt."""
    eligible_count = sum(bool(row["eligible_primary"]) for row in rows)
    classification = classify_development(profile_details["summary"], permutation, countermodels, eligible_count)
    ledger = development_ledger_rows(rows, profile_details)
    sanitized_failures = _validated_failures(failures)
    grid_rows = [
        {
            "grid_index": index,
            "delta_H_hex": _hex(profile["delta_H"]),
            "chi2_hex": _hex(profile["chi2"]),
            "beta_hex": [_hex(value) for value in profile["beta"]],
        }
        for index, profile in enumerate(profile_details["profiles"])
    ]
    permutation_rows = [
        {"permutation_index": index, "statistic_hex": _hex(statistic)}
        for index, statistic in enumerate(permutation["permutation_statistics"])
    ]
    _require(len(grid_rows) == 161, "profile artifact row-count drift")
    _require(len(permutation_rows) == 10000, "permutation artifact row-count drift")
    summary = {
        "development_rows": len(rows),
        "eligible_primary_rows": eligible_count,
        "partial_mask_rows": len(rows) - eligible_count,
        "development_ledger_root_sha256": _leaf_root(ledger),
        "profile": _hex_tree(profile_details["summary"]),
        "permutation": {
            "observed_hex": _hex(permutation["observed"]),
            "tail_count": int(permutation["tail_count"]),
            "p_value_hex": _hex(permutation["p_value"]),
        },
        "classification": _hex_tree(classification),
        "access_counts": {str(key): int(value) for key, value in sorted(access_counts.items())},
        "claim_ceiling": load_config()["claim_ceiling"],
    }
    return {
        "artifacts/development-rows.jsonl": _jsonl(ledger),
        "artifacts/profile-grid.jsonl": _jsonl(grid_rows),
        "artifacts/permutation-statistics.jsonl": _jsonl(permutation_rows),
        "artifacts/countermodels.json": _pretty(_hex_tree(countermodels)),
        "artifacts/failures.json": _pretty(sanitized_failures),
        "artifacts/development-summary.json": _pretty(summary),
    }


def _safe_artifact_name(value: str) -> Path:
    _require(isinstance(value, str) and value and "\\" not in value, "invalid artifact name")
    _require(all(part not in {"", ".", ".."} for part in value.split("/")), "unsafe artifact name")
    pure = PurePosixPath(value)
    _require(not pure.is_absolute(), "unsafe artifact name")
    return Path(*pure.parts)


def transactional_promote(final_directory: Path, artifacts: Mapping[str, bytes], staging_root: Path) -> str:
    """Source-free transactional writer used by the future audited executor."""
    _require(bool(artifacts), "empty artifact transaction")
    final = final_directory.resolve()
    root = staging_root.resolve()
    _require(not final.exists(), "final directory already exists")
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="run-", dir=root)).resolve()
    _require(root in staging.parents, "staging containment failure")
    try:
        for name, payload in sorted(artifacts.items()):
            _require(isinstance(payload, bytes), "artifact payload must be bytes")
            relative = _safe_artifact_name(name)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _require(_sha256_bytes(payload) == file_sha256(target), "staged artifact hash mismatch")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        return "PROMOTED_COMPLETE"
    except Exception:
        if staging.exists():
            _require(root in staging.parents, "cleanup containment failure")
            shutil.rmtree(staging)
        raise


def sanitized_failure_record(stage: str, reason_code: str, access_counts: Mapping[str, int], identifiers: Sequence[int]) -> dict[str, Any]:
    _require(re.fullmatch(r"[A-Z0-9_]+", stage) is not None, "invalid failure stage")
    _require(re.fullmatch(r"[A-Z0-9_]+", reason_code) is not None, "invalid reason code")
    ids = sorted(int(value) for value in identifiers)
    _require(all(value > 0 for value in ids), "invalid failure identifier")
    counts = {str(key): int(value) for key, value in sorted(access_counts.items())}
    _require(all(value >= 0 for value in counts.values()), "invalid access count")
    return {"stage": stage, "reason_code": reason_code, "access_counts": counts, "authorized_development_ids": ids}


def _format_synthetic_record(fields: Sequence[Mapping[str, Any]], values: Mapping[str, Any], length: int) -> bytes:
    payload = bytearray(b" " * length)
    for field in fields:
        name = str(field["name"])
        width = int(field["end"]) - int(field["start"]) + 1
        kind = str(field["format"])[0]
        if kind == "A":
            token = str(values[name]).ljust(width)
        elif kind == "I":
            token = f"{int(values[name]):>{width}d}"
        else:
            decimals = int(str(field["format"]).split(".")[1])
            token = f"{float(values[name]):>{width}.{decimals}f}"
        _require(len(token) == width and token.isascii(), "synthetic field overflow")
        payload[int(field["start"]) - 1 : int(field["end"])] = token.encode("ascii")
    return bytes(payload) + b"\n"


def _synthetic_ledger(line: bytes, identifier: int, source_index: int = 0, framed_start: int = 0) -> dict[str, Any]:
    payload = record_payload(line, 157)
    bucket, role = executor_v3.split_role(identifier)
    body = {
        "bucket": bucket,
        "canonical_identifier": str(identifier),
        "framed_end_exclusive": framed_start + len(line),
        "framed_raw_sha256": _sha256_bytes(line),
        "framed_start": framed_start,
        "identifier": identifier,
        "identifier_field_raw_sha256": _sha256_bytes(payload[:7]),
        "line_ending_bytes": len(line) - len(payload),
        "opaque_tail_raw_sha256": _sha256_bytes(payload[7:]),
        "payload_end_exclusive": framed_start + len(payload),
        "payload_raw_sha256": _sha256_bytes(payload),
        "payload_start": framed_start,
        "role": role,
        "source_index": source_index,
    }
    body["leaf_sha256"] = content_sha256(body)
    return body


def conformance_gates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    for name, fields in config["fixed_width_schemas"].items():
        validate_schema(fields, int(config["sources"][name]["record_length"]))
    cf4_values = {"1PGC": 12, "DMzp": 30.0, "e_DMzp": 0.1, "Dist": 10.0, "V3k": 1000, "RAdeg": 0.0, "DEdeg": 0.0}
    cf4_line = _format_synthetic_record(config["fixed_width_schemas"]["CF4_TABLE4"], cf4_values, 157)
    parsed_cf4 = parse_cf4_development_record(cf4_line, _synthetic_ledger(cf4_line, 12), source_index=0, framed_start=0)
    t1_values = {"Cosmo": "Planck2018", "x": 1.0, "y": 0.0, "z": 0.0, "Rad": 1.0, "void": 1, "edge": 0, "s": 1.0, "RAdeg": 0.0, "DEdeg": 0.0, "Reff": 1.0}
    t2_values = {"Cosmo": "Planck2018", "x": 5.0, "y": 0.0, "z": 0.0, "Rad": 2.0, "void": 1}
    t1 = parse_vast_table1_record(_format_synthetic_record(config["fixed_width_schemas"]["VAST_TABLE1"], t1_values, 181))
    t2 = parse_vast_table2_record(_format_synthetic_record(config["fixed_width_schemas"]["VAST_TABLE2"], t2_values, 105))
    geometry = prepare_vast_geometry([t1], [t2])
    derived = derive_development_row(parsed_cf4, bytes([1]) * 64800, geometry["spheres_Mpc"])
    failure = sanitized_failure_record("PARSE", "SYNTHETIC_FAILURE", {"scientific_rows": 0}, [12])
    return [
        {"check_id": "EXACT_THREE_SCHEMA_GRAMMARS", "passed": True},
        {"check_id": "DEVELOPMENT_ROLE_BEFORE_SCIENCE_SURFACE", "passed": parsed_cf4["1PGC"] == 12},
        {"check_id": "VAST_JOIN_COORDINATE_AND_UNIT_SURFACE", "passed": geometry["retained"] == 1 and geometry["excluded_edge"] == 0},
        {"check_id": "GEOMETRY_ELIGIBILITY_LIKELIHOOD_SURFACE", "passed": derived["eligible_primary"] and derived["sigma_s"] > 0.0},
        {"check_id": "SANITIZED_FAILURE_SURFACE", "passed": set(failure) == {"stage", "reason_code", "access_counts", "authorized_development_ids"}},
        {"check_id": "NO_REAL_SCIENTIFIC_ACCESS", "passed": all(value == 0 for value in config["access_intent"]["current_packet"].values())},
    ]


def build_receipt() -> dict[str, Any]:
    validate_code_pins()
    config = load_config()
    bindings = validate_release_chain(config)
    gates = conformance_gates(config)
    _require(all(row["passed"] for row in gates), "source-free conformance failure")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-release-receipt-1.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "release_chain": config["release_chain"],
        "release_bindings": bindings,
        "sources_binding_only": config["sources"],
        "canonical_paths": config["canonical_paths"],
        "parsers": {"fixed_width_schemas": config["fixed_width_schemas"], "strict_grammar": config["strict_grammar"], "scientific_domains": config["scientific_domains"]},
        "role_gate": config["role_gate"],
        "geometry_join": config["geometry_join"],
        "eligibility": config["eligibility"],
        "likelihood": config["likelihood"],
        "countermodels": config["countermodels"],
        "profile_and_permutation": config["profile_and_permutation"],
        "thresholds": config["thresholds"],
        "outputs": config["outputs"],
        "failure_transaction": config["failure_transaction"],
        "access_intent": config["access_intent"],
        "claim_ceiling": config["claim_ceiling"],
        "runtime": config["runtime"],
        "conformance_gates": gates,
        "mutation_freeze": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
        },
        "access_accounting": config["access_intent"]["current_packet"],
        "next_gate": config["next_gate"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    handle, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "CREATED"


def write_receipt() -> str:
    receipt = build_receipt()
    return _atomic_no_clobber(OUTPUT_PATH, _pretty(receipt))


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected = build_receipt()
    _require(observed == expected, "receipt drift")
    _require(observed["content_sha256"] == _self_hash(observed), "receipt self-hash drift")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID_SOURCE_FREE_NO_SCIENTIFIC_ACCESS")
    else:
        print(check_receipt()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
