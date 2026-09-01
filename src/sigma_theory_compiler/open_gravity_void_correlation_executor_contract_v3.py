"""Normalize-once repair of the Lane-9 CF4 x VAST executor contract v2.

This module deliberately has no command that decompresses or parses a bound
scientific source.  Its numerical routines accept caller-supplied synthetic
arrays so the frozen executor semantics can be tested before a separate audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_correlation_executor_contract_v3.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_correlation_executor_contract_v3.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_correlation_executor_contract_v3.py"
OUTPUT_PATH = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-executor-contract-v3/receipt.json"
_CONFIG_RAW_SHA256 = "5ba4e48c444c14b4f355f516ec664d449c4fd8d2a9d332676d0ffbf40f620a04"
_CONFIG_CONTENT_SHA256 = "b3b80f518dcb91595a13ff5a1d0ec5943e96a50f16b185640e085fe586cd28b3"
_MODULE_SEMANTIC_SHA256 = "e7e88c647487a3598418340cb340b88c40d716372f203eb4b3e9bca77f8cfe86"
_TEST_RAW_SHA256 = "edb9e480f7454b9fc8e94ace36aef981d17712718ba8a3424c53afe88ac09540"

_I_TOKEN = re.compile(rb" *[+-]?[0-9]+ *\Z", re.ASCII)
_F_TOKEN = re.compile(rb" *[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+) *\Z", re.ASCII)
_PIN_NAMES = (
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
)


class VoidExecutorV3Error(RuntimeError):
    """Raised when a frozen v3 invariant changes or a sealed action is attempted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidExecutorV3Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    for name in _PIN_NAMES:
        marker = f'{name} = "'
        start = text.index(marker) + len(marker)
        end = text.index('"', start)
        text = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(text.encode()).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def validate_code_pins() -> None:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test pin drift")


def canonical_bound_path(relative: str) -> Path:
    """Resolve one normalized repository-relative POSIX file without aliases."""
    _require(isinstance(relative, str) and relative != "", "empty bound path")
    _require("\\" not in relative, "bound path must use POSIX separators")
    _require(":" not in relative, "absolute or drive-qualified bound path")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute(), "absolute bound path")
    _require(pure.drive == "" and pure.root == "", "rooted bound path")
    _require(all(part not in ("", ".", "..") for part in pure.parts), "noncanonical path component")
    _require(pure.as_posix() == relative, "noncanonical path spelling")
    cursor = REPO_ROOT
    for part in pure.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), "symlink in bound path")
    _require(cursor.is_file(), "bound path is not a regular file")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise VoidExecutorV3Error("bound path escapes repository") from error
    _require(resolved == cursor.absolute(), "bound path aliases physical target")
    return resolved


def load_config() -> dict[str, Any]:
    validate_code_pins()
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config semantic drift")
    _require(value["status"] == "DRAFT_NORMALIZE_ONCE_REPAIR_MUTATION_FREEZE_ROWS_UNOPENED_AWAIT_INDEPENDENT_AUDIT", "status drift")
    _require(value["decision"] == "NORMALIZE_ONCE_REPAIR_MUTATION_FREEZE_REQUESTED_AWAIT_SEPARATE_AGENT_AUDIT_NO_ROW_ACCESS", "decision drift")
    _require(np.__version__ == value["runtime"]["numpy_exact"], "NumPy version drift")
    expected_access = {
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
    _require(value["access_accounting"] == expected_access, "access-accounting drift")
    return value


def _validate_bound_receipt(section: Mapping[str, Any], expected_status: str) -> dict[str, Any]:
    path = canonical_bound_path(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], "bound receipt raw drift")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    _require(receipt.get("content_sha256") == section["content_sha256"], "bound receipt content pin drift")
    _require(receipt.get("content_sha256") == _self_hash(receipt), "bound receipt self-hash drift")
    _require(receipt.get("status") == expected_status, "bound receipt status drift")
    return receipt


def _validate_audit_block(section: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_bound_path(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], "audit-block raw drift")
    audit = json.loads(path.read_text(encoding="utf-8"))
    _require(audit.get("status") == section["status"], "audit-block status drift")
    _require(audit.get("counterexample", {}).get("direction") == section["counterexample_direction"], "audit counterexample drift")
    _require(audit.get("scientific_access", {}).get("scientific_rows_decoded") == 0, "blocked packet opened rows")
    return audit


def validate_schema(name: str, fields: Sequence[Mapping[str, Any]], record_length: int) -> None:
    seen: set[str] = set()
    last_end = 0
    for field in fields:
        field_name = str(field["name"])
        start, end = int(field["start"]), int(field["end"])
        _require(field_name not in seen, f"duplicate field: {name}.{field_name}")
        _require(1 <= start <= end <= record_length, f"field outside record: {name}.{field_name}")
        _require(start > last_end, f"field overlap/order: {name}.{field_name}")
        match = re.fullmatch(r"([AIF])([1-9][0-9]*)(?:\.([0-9]+))?", str(field["format"]))
        _require(match is not None, f"invalid format: {name}.{field_name}")
        kind, width, decimals = match.group(1), int(match.group(2)), match.group(3)
        _require(width == end - start + 1, f"format width mismatch: {name}.{field_name}")
        _require((kind == "F") == (decimals is not None), f"format decimal mismatch: {name}.{field_name}")
        seen.add(field_name)
        last_end = end


def synthetic_record_payload(line: bytes, record_length: int) -> bytes:
    """Apply the frozen line grammar to caller-supplied synthetic bytes only."""
    _require(isinstance(line, bytes), "record must be bytes")
    _require(line.endswith(b"\n"), "record lacks terminal LF")
    payload = line[:-1]
    if payload.endswith(b"\r"):
        payload = payload[:-1]
    _require(b"\r" not in payload and b"\n" not in payload, "embedded or repeated line ending")
    _require(len(payload) == record_length, "record length mismatch")
    _require(all(byte < 128 for byte in payload), "non-ASCII record")
    return payload


def parse_synthetic_field(payload: bytes, field: Mapping[str, Any]) -> str | int | float | None:
    """Parse one field from a synthetic payload; this function accepts no path."""
    start, end = int(field["start"]) - 1, int(field["end"])
    _require(len(payload) >= end, "synthetic payload too short")
    token = payload[start:end]
    _require(all(byte < 128 for byte in token), "non-ASCII field")
    if token and all(byte == 0x20 for byte in token):
        _require(not bool(field.get("required", False)), "required field missing")
        return None
    kind = str(field["format"])[0]
    if kind == "A":
        _require(all(0x20 <= byte <= 0x7E for byte in token), "nonprintable A field")
        return token.decode("ascii").rstrip(" ")
    _require(all(byte == 0x20 or 0x21 <= byte <= 0x7E for byte in token), "control byte in numeric field")
    if kind == "I":
        _require(_I_TOKEN.fullmatch(token) is not None, "invalid I token")
        return int(token.decode("ascii"))
    _require(kind == "F", "unsupported field kind")
    _require(_F_TOKEN.fullmatch(token) is not None, "invalid F token")
    result = float(token.decode("ascii"))
    _require(math.isfinite(result), "nonfinite F token")
    return result


def canonical_identifier(identifier: int) -> bytes:
    _require(isinstance(identifier, int) and not isinstance(identifier, bool), "identifier must be int")
    _require(identifier > 0, "identifier must be positive")
    return str(identifier).encode("ascii")


def split_role(identifier: int) -> tuple[int, str]:
    digest = hashlib.sha256(canonical_identifier(identifier)).digest()
    bucket = int.from_bytes(digest[:8], "big", signed=False) % 10
    role = "development" if bucket <= 5 else "validation" if bucket <= 7 else "confirmation"
    return bucket, role


def stf_basis() -> np.ndarray:
    return np.asarray(load_config()["stf_shear"]["basis"], dtype=np.float64)


def validate_stf_basis(basis: np.ndarray | None = None) -> None:
    array = stf_basis() if basis is None else np.asarray(basis, dtype=np.float64)
    _require(array.shape == (5, 3, 3) and bool(np.all(np.isfinite(array))), "invalid STF basis shape")
    for index, matrix in enumerate(array):
        _require(bool(np.array_equal(matrix, matrix.T)), f"STF basis {index} is not symmetric")
        _require(abs(float(np.trace(matrix))) <= 4e-16, f"STF basis {index} is not trace-free")
    gram = np.einsum("aij,bij->ab", array, array)
    _require(bool(np.allclose(gram, np.eye(5), rtol=0.0, atol=5e-16)), "STF basis is not Frobenius-orthonormal")


def _raw_finite_direction(direction: Sequence[float]) -> tuple[float, float, float]:
    _require(len(direction) == 3, "direction must have three components")
    values = tuple(float(value) for value in direction)
    _require(all(math.isfinite(value) for value in values), "nonfinite direction")
    return values


def normalize_direction(direction: Sequence[float]) -> tuple[float, float, float]:
    values = _raw_finite_direction(direction)
    norm = math.sqrt(math.fsum(value * value for value in values))
    _require(math.isfinite(norm) and norm > 0.0, "zero or invalid direction")
    normalized = tuple(value / norm for value in values)
    _require(all(math.isfinite(value) for value in normalized), "nonfinite normalized direction")
    return normalized


def _binary64_hex(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


def _shear_quadratic_columns_from_normalized(
    normalized_direction: tuple[float, float, float],
) -> tuple[float, float, float, float, float]:
    nx, ny, nz = _raw_finite_direction(normalized_direction)
    return (
        (nx * nx - ny * ny) / math.sqrt(2.0),
        (nx * nx + ny * ny - 2.0 * nz * nz) / math.sqrt(6.0),
        math.sqrt(2.0) * nx * ny,
        math.sqrt(2.0) * nx * nz,
        math.sqrt(2.0) * ny * nz,
    )


def shear_quadratic_columns(direction: Sequence[float]) -> tuple[float, float, float, float, float]:
    normalized = normalize_direction(direction)
    return _shear_quadratic_columns_from_normalized(normalized)


def _nuisance_velocity_design_from_normalized(
    distance_mpc: float,
    normalized_direction: tuple[float, float, float],
) -> tuple[float, ...]:
    distance = float(distance_mpc)
    _require(math.isfinite(distance) and distance >= 0.0, "invalid distance")
    nx, ny, nz = _raw_finite_direction(normalized_direction)
    q = _shear_quadratic_columns_from_normalized((nx, ny, nz))
    result = (distance, nx, ny, nz, *(distance * value for value in q))
    _require(len(result) == 9 and all(math.isfinite(value) for value in result), "invalid nuisance design")
    return result


def nuisance_velocity_design(distance_mpc: float, direction: Sequence[float]) -> tuple[float, ...]:
    normalized = normalize_direction(direction)
    return _nuisance_velocity_design_from_normalized(distance_mpc, normalized)


def velocity_to_log_design(velocity_design: Sequence[float], c_km_s: float = 299792.458) -> tuple[float, ...]:
    c_value = float(c_km_s)
    _require(math.isfinite(c_value) and c_value > 0.0, "invalid speed of light")
    result = tuple(float(value) / c_value for value in velocity_design)
    _require(all(math.isfinite(value) for value in result), "invalid log-redshift design")
    return result


def observed_log_redshift(v3k_km_s: float, c_km_s: float = 299792.458) -> float:
    velocity, c_value = float(v3k_km_s), float(c_km_s)
    _require(math.isfinite(velocity) and math.isfinite(c_value) and c_value > 0.0, "invalid observed velocity")
    _require(velocity > -c_value, "log1p domain failure")
    result = math.log1p(velocity / c_value)
    _require(math.isfinite(result), "nonfinite observed log redshift")
    return result


def validate_cf4_duplicate_keys(identifiers: Sequence[int]) -> None:
    keys = [int(identifier) for identifier in identifiers]
    _require(all(key > 0 for key in keys), "invalid CF4 key")
    _require(len(keys) == len(set(keys)), "duplicate CF4 1PGC")


def validate_vast_duplicate_keys(
    table1_rows: Sequence[tuple[str, int, int]],
    table2_rows: Sequence[tuple[str, int, float, float, float, float]],
) -> dict[str, int]:
    planck_rows = [(str(cosmo), int(void), int(edge)) for cosmo, void, edge in table1_rows if str(cosmo) == "Planck2018"]
    _require(all(edge in (0, 1) for _, _, edge in planck_rows), "invalid VAST edge flag")
    maximal = [(cosmo, void) for cosmo, void, _ in planck_rows]
    _require(len(maximal) == len(set(maximal)), "duplicate VAST_TABLE1 key")
    edge_by_key = {(cosmo, void): edge for cosmo, void, edge in planck_rows}
    sphere_keys: list[tuple[str, int, float, float, float, float]] = []
    for cosmo, void, x, y, z, radius in table2_rows:
        key = (str(cosmo), int(void), float(x), float(y), float(z), float(radius))
        if key[0] != "Planck2018":
            continue
        _require(all(math.isfinite(value) for value in key[2:]), "nonfinite sphere key")
        _require((key[0], key[1]) in edge_by_key, "unmatched VAST sphere")
        sphere_keys.append(key)
    _require(len(sphere_keys) == len(set(sphere_keys)), "duplicate VAST_TABLE2 sphere key")
    retained = sum(edge_by_key[(key[0], key[1])] == 0 for key in sphere_keys)
    excluded_edge = len(sphere_keys) - retained
    return {"retained": retained, "excluded_edge": excluded_edge}


def distance_strata(distances_mpc: Sequence[float], identifiers: Sequence[int], strata_count: int = 10) -> list[int]:
    _require(len(distances_mpc) == len(identifiers), "strata length mismatch")
    count = len(identifiers)
    _require(isinstance(strata_count, int) and strata_count > 0 and count >= strata_count, "insufficient rows for strata")
    validate_cf4_duplicate_keys(identifiers)
    distances = [float(value) for value in distances_mpc]
    _require(all(math.isfinite(value) and value >= 0.0 for value in distances), "invalid stratum distance")
    order = sorted(range(count), key=lambda index: (distances[index], int(identifiers[index])))
    labels = [-1] * count
    for rank, index in enumerate(order):
        labels[index] = (strata_count * rank) // count
    sizes = [labels.count(stratum) for stratum in range(strata_count)]
    _require(min(sizes) > 0 and max(sizes) - min(sizes) <= 1, "unequal distance strata")
    return labels


def validate_stage_counts(
    stage: str,
    raw_counts: Mapping[str, int] | None = None,
    eligible_development: int | None = None,
) -> None:
    if stage == "IDS_PARTITIONED":
        _require(raw_counts is not None, "raw counts missing")
        thresholds = {"development": 500, "validation": 150, "confirmation": 150}
        _require(set(raw_counts) == set(thresholds), "raw-count role mismatch")
        _require(all(isinstance(raw_counts[role], int) and raw_counts[role] >= threshold for role, threshold in thresholds.items()), "raw-ID minimum failed")
        _require(eligible_development is None, "eligibility forbidden at ID stage")
        return
    if stage == "DEVELOPMENT_OPEN":
        _require(raw_counts is None, "raw counts not reinterpreted at development stage")
        _require(isinstance(eligible_development, int) and eligible_development >= 500, "development eligible minimum failed")
        return
    if stage in {"VALIDATION_OPEN", "CONFIRMATION_OPEN", "PANTHEON_OPEN"}:
        raise VoidExecutorV3Error(f"{stage} is not authorized by v3")
    raise VoidExecutorV3Error("unknown access stage")


def _ordered_inputs(
    y: Sequence[float],
    sigma: Sequence[float],
    path_distances: Sequence[float],
    directions: Sequence[Sequence[float]],
    l_void: Sequence[float],
    identifiers: Sequence[int],
) -> tuple[list[float], list[float], list[float], list[tuple[float, float, float]], list[float], list[int]]:
    size = len(identifiers)
    _require(size > 0 and all(len(values) == size for values in (y, sigma, path_distances, directions, l_void)), "profile length mismatch")
    validate_cf4_duplicate_keys(identifiers)
    order = sorted(range(size), key=lambda index: int(identifiers[index]))
    ids = [int(identifiers[index]) for index in order]
    y_values = [float(y[index]) for index in order]
    sigma_values = [float(sigma[index]) for index in order]
    distance_values = [float(path_distances[index]) for index in order]
    direction_values = [_raw_finite_direction(directions[index]) for index in order]
    void_values = [float(l_void[index]) for index in order]
    _require(all(math.isfinite(value) for value in y_values), "nonfinite response")
    _require(all(math.isfinite(value) and value > 0.0 for value in sigma_values), "invalid uncertainty")
    _require(all(math.isfinite(value) and value >= 0.0 for value in distance_values), "invalid distance")
    _require(all(math.isfinite(value) and 0.0 <= value <= distance_values[index] + 1e-10 for index, value in enumerate(void_values)), "invalid void path")
    return y_values, sigma_values, distance_values, direction_values, void_values, ids


def _cholesky_solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    size = len(vector)
    _require(size > 0 and len(matrix) == size and all(len(row) == size for row in matrix), "normal-system shape mismatch")
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            subtotal = math.fsum(lower[row][index] * lower[column][index] for index in range(column))
            if row == column:
                diagonal = float(matrix[row][row]) - subtotal
                _require(math.isfinite(diagonal) and diagonal > 0.0, "nonpositive Cholesky diagonal")
                lower[row][column] = math.sqrt(diagonal)
            else:
                value = (float(matrix[row][column]) - subtotal) / lower[column][column]
                _require(math.isfinite(value), "nonfinite Cholesky value")
                lower[row][column] = value
    forward = [0.0] * size
    for row in range(size):
        subtotal = math.fsum(lower[row][column] * forward[column] for column in range(row))
        forward[row] = (float(vector[row]) - subtotal) / lower[row][row]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        subtotal = math.fsum(lower[column][row] * solution[column] for column in range(row + 1, size))
        solution[row] = (forward[row] - subtotal) / lower[row][row]
    _require(all(math.isfinite(value) for value in solution), "nonfinite profile solution")
    return solution


def _prepare_profile(
    y: Sequence[float],
    sigma: Sequence[float],
    path_distances: Sequence[float],
    directions: Sequence[Sequence[float]],
    l_void: Sequence[float],
    identifiers: Sequence[int],
    c_km_s: float = 299792.458,
) -> dict[str, Any]:
    y_values, sigma_values, distance_values, direction_values, void_values, ids = _ordered_inputs(
        y, sigma, path_distances, directions, l_void, identifiers
    )
    c_value = float(c_km_s)
    _require(math.isfinite(c_value) and c_value > 0.0, "invalid c")
    design = [velocity_to_log_design(nuisance_velocity_design(distance_values[index], direction_values[index]), c_value) for index in range(len(ids))]
    weights = [1.0 / (value * value) for value in sigma_values]
    law_column = [value / c_value for value in void_values]
    priors = [20.0, 500.0, 500.0, 500.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    normal = [[0.0] * 9 for _ in range(9)]
    for row in range(9):
        for column in range(row + 1):
            value = math.fsum(weights[index] * design[index][row] * design[index][column] for index in range(len(ids)))
            if row == column:
                value += 1.0 / (priors[row] * priors[row])
            normal[row][column] = value
            normal[column][row] = value
    return {
        "y": y_values,
        "sigma": sigma_values,
        "design": design,
        "weights": weights,
        "law_column": law_column,
        "priors": priors,
        "normal": normal,
        "row_ids": ids,
    }


def _profile_prepared(prepared: Mapping[str, Any], delta_h: float) -> dict[str, Any]:
    delta = float(delta_h)
    _require(math.isfinite(delta), "invalid law parameter")
    y_values = prepared["y"]
    sigma_values = prepared["sigma"]
    design = prepared["design"]
    weights = prepared["weights"]
    law_column = prepared["law_column"]
    priors = prepared["priors"]
    normal = prepared["normal"]
    ids = prepared["row_ids"]
    residual_without_nuisance = [y_values[index] - delta * law_column[index] for index in range(len(ids))]
    rhs = [
        math.fsum(weights[index] * design[index][row] * residual_without_nuisance[index] for index in range(len(ids)))
        for row in range(9)
    ]
    beta = _cholesky_solve(normal, rhs)
    data_terms = []
    for index in range(len(ids)):
        fitted = math.fsum(design[index][column] * beta[column] for column in range(9))
        standardized = (residual_without_nuisance[index] - fitted) / sigma_values[index]
        data_terms.append(standardized * standardized)
    prior_terms = [(beta[column] / priors[column]) ** 2 for column in range(9)]
    chi2 = math.fsum((*data_terms, *prior_terms))
    _require(math.isfinite(chi2) and chi2 >= 0.0, "invalid profile objective")
    return {"delta_H": delta, "chi2": chi2, "beta": beta, "row_ids": ids}


def profile_at_delta(
    y: Sequence[float],
    sigma: Sequence[float],
    path_distances: Sequence[float],
    directions: Sequence[Sequence[float]],
    l_void: Sequence[float],
    identifiers: Sequence[int],
    delta_h: float,
    c_km_s: float = 299792.458,
) -> dict[str, Any]:
    prepared = _prepare_profile(y, sigma, path_distances, directions, l_void, identifiers, c_km_s)
    return _profile_prepared(prepared, delta_h)


def delta_h_grid() -> list[float]:
    return [-20.0 + 0.25 * index for index in range(161)]


def _tie_tolerance(first: float, second: float) -> float:
    return 8.0 * (2.0 ** -52) * max(1.0, abs(first), abs(second))


def profile_grid(
    y: Sequence[float],
    sigma: Sequence[float],
    path_distances: Sequence[float],
    directions: Sequence[Sequence[float]],
    l_void: Sequence[float],
    identifiers: Sequence[int],
) -> dict[str, Any]:
    prepared = _prepare_profile(y, sigma, path_distances, directions, l_void, identifiers)
    profiles = [_profile_prepared(prepared, delta) for delta in delta_h_grid()]
    raw_minimum = min(float(row["chi2"]) for row in profiles)
    tied = [row for row in profiles if float(row["chi2"]) - raw_minimum <= _tie_tolerance(float(row["chi2"]), raw_minimum)]
    best = min(tied, key=lambda row: (abs(float(row["delta_H"])), float(row["delta_H"])))
    null = profiles[80]
    _require(float(null["delta_H"]) == 0.0, "grid null index drift")
    delta_chi2 = float(null["chi2"]) - float(best["chi2"])
    if delta_chi2 < 0.0:
        _require(abs(delta_chi2) <= _tie_tolerance(float(null["chi2"]), float(best["chi2"])), "negative delta-chi-square")
        delta_chi2 = 0.0
    statistic = delta_chi2 if float(best["delta_H"]) > 0.0 else 0.0
    return {
        "best_delta_H": float(best["delta_H"]),
        "best_chi2": float(best["chi2"]),
        "null_chi2": float(null["chi2"]),
        "delta_chi2": delta_chi2,
        "one_sided_statistic": statistic,
        "tied_delta_H": [float(row["delta_H"]) for row in tied],
    }


def _pcg64_permutation_orders(generator: np.random.Generator, group_sizes: Sequence[int]) -> list[list[int]]:
    orders: list[list[int]] = []
    for size in group_sizes:
        _require(isinstance(size, int) and size > 0, "invalid permutation group size")
        orders.append([int(value) for value in generator.permutation(size).tolist()])
    return orders


def synthetic_permutation_test(
    y: Sequence[float],
    sigma: Sequence[float],
    luminosity_distances: Sequence[float],
    path_distances: Sequence[float],
    directions: Sequence[Sequence[float]],
    l_void: Sequence[float],
    identifiers: Sequence[int],
    permutations: int,
    seed: int = 902104729,
) -> dict[str, Any]:
    """Exercise the frozen permutation rule on synthetic inputs only."""
    _require(isinstance(permutations, int) and permutations > 0, "invalid permutation count")
    _require(len(luminosity_distances) == len(path_distances), "luminosity/path distance length mismatch")
    labels = distance_strata(luminosity_distances, identifiers, 10)
    generator = np.random.Generator(np.random.PCG64(seed))
    observed = float(profile_grid(y, sigma, path_distances, directions, l_void, identifiers)["one_sided_statistic"])
    permuted_statistics: list[float] = []
    for _ in range(permutations):
        permuted = [float(value) for value in l_void]
        index_groups = [
            sorted(
                (index for index, label in enumerate(labels) if label == stratum),
                key=lambda index: int(identifiers[index]),
            )
            for stratum in range(10)
        ]
        orders = _pcg64_permutation_orders(generator, [len(indexes) for indexes in index_groups])
        for stratum in range(10):
            indexes = index_groups[stratum]
            values = [float(l_void[index]) for index in indexes]
            for target_position, source_position in enumerate(orders[stratum]):
                permuted[indexes[target_position]] = values[int(source_position)]
        statistic = float(profile_grid(y, sigma, path_distances, directions, permuted, identifiers)["one_sided_statistic"])
        _require(math.isfinite(statistic), "nonfinite permutation statistic")
        permuted_statistics.append(statistic)
    tail = sum(statistic >= observed for statistic in permuted_statistics)
    return {
        "observed": observed,
        "permutation_statistics": permuted_statistics,
        "tail_count": tail,
        "p_value": (1 + tail) / (permutations + 1),
    }


def _bind_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bound: dict[str, dict[str, Any]] = {}
    for name, source in config["inputs"].items():
        path = canonical_bound_path(str(source["path"]))
        size = path.stat().st_size
        _require(size == int(source["bytes"]), f"input byte-count drift: {name}")
        observed_hash = file_sha256(path)
        _require(observed_hash == source["sha256"], f"opaque input hash drift: {name}")
        bound[name] = {
            "path": source["path"],
            "bytes": size,
            "sha256": observed_hash,
            "access_class": source["access_class"],
            "operation": "STREAMING_RAW_SHA256_ONLY",
        }
    return bound


def _synthetic_profile_fixture() -> tuple[list[float], list[float], list[float], list[float], list[tuple[float, float, float]], list[float], list[int]]:
    identifiers = list(range(1, 13))
    luminosity_distances = [40.0 + 2.0 * index for index in range(12)]
    path_distances = [35.0 + 1.8 * index for index in range(12)]
    directions = [(1.0, 0.1 * (index + 1), 0.05 * (index % 3 + 1)) for index in range(12)]
    l_void = [4.0 + (index % 5) for index in range(12)]
    sigma = [0.0008] * 12
    y = [6.0 * l_void[index] / 299792.458 for index in range(12)]
    return y, sigma, luminosity_distances, path_distances, directions, l_void, identifiers


def conformance_gates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for name, fields in config["fixed_width_schemas"].items():
        validate_schema(name, fields, int(config["inputs"][name]["record_length"]))
    gates.append({"check_id": "FIXED_WIDTH_SCHEMAS", "passed": True})

    validate_stf_basis()
    q = shear_quadratic_columns((1.0, 0.0, 0.0))
    gates.append({"check_id": "STF_ORTHONORMAL_AND_ORDERED", "passed": q[0] > 0.0 and q[1] > 0.0})

    adversarial = config["stf_shear"]["adversarial_regression"]
    once = normalize_direction(adversarial["raw"])
    adversarial_q = shear_quadratic_columns(adversarial["raw"])
    adversarial_design = nuisance_velocity_design(17.0, adversarial["raw"])
    expected_direction_hex = adversarial["once_normalized_binary64_hex"]
    expected_shear_hex = adversarial["once_shear_binary64_hex"]
    normalize_once_passed = (
        [_binary64_hex(value) for value in once] == expected_direction_hex
        and [_binary64_hex(value) for value in adversarial_q] == expected_shear_hex
        and tuple(adversarial_design[1:4]) == once
        and tuple(adversarial_design[4:]) == tuple(17.0 * value for value in adversarial_q)
        and _shear_quadratic_columns_from_normalized(once) == adversarial_q
    )
    gates.append({"check_id": "NORMALIZE_ONCE_ADVERSARIAL_BINARY64", "passed": normalize_once_passed})

    design = nuisance_velocity_design(10.0, (1.0, 0.0, 0.0))
    mapped = velocity_to_log_design(design)
    gates.append({"check_id": "VELOCITY_TO_LOG_DIVIDE_C_ONCE", "passed": mapped[0] == design[0] / 299792.458})

    labels = distance_strata([1.0] * 10, list(range(10, 0, -1)))
    gates.append({"check_id": "DISTANCE_TIES_TOTAL_ORDER", "passed": sorted(labels) == list(range(10))})

    validate_cf4_duplicate_keys([1, 2, 3])
    validate_vast_duplicate_keys([("Planck2018", 1, 0)], [("Planck2018", 1, 1.0, 2.0, 3.0, 4.0)])
    duplicate_failures = 0
    for call in (
        lambda: validate_cf4_duplicate_keys([1, 1]),
        lambda: validate_vast_duplicate_keys([("Planck2018", 1, 0), ("Planck2018", 1, 1)], []),
        lambda: validate_vast_duplicate_keys([("Planck2018", 1, 0)], [("Planck2018", 2, 1.0, 2.0, 3.0, 4.0)]),
    ):
        try:
            call()
        except VoidExecutorV3Error:
            duplicate_failures += 1
    gates.append({"check_id": "DUPLICATE_AND_JOIN_FAIL_CLOSED", "passed": duplicate_failures == 3})

    validate_stage_counts("IDS_PARTITIONED", {"development": 500, "validation": 150, "confirmation": 150})
    validate_stage_counts("DEVELOPMENT_OPEN", eligible_development=500)
    sealed_failures = 0
    for stage in ("VALIDATION_OPEN", "CONFIRMATION_OPEN", "PANTHEON_OPEN"):
        try:
            validate_stage_counts(stage)
        except VoidExecutorV3Error:
            sealed_failures += 1
    gates.append({"check_id": "STAGED_COUNTS_WITH_SEALED_HOLDOUTS", "passed": sealed_failures == 3})

    y, sigma, luminosity_distances, path_distances, directions, l_void, identifiers = _synthetic_profile_fixture()
    profile = profile_grid(y, sigma, path_distances, directions, l_void, identifiers)
    gates.append({"check_id": "PROFILE_SOLVER_GRID_AND_TIES", "passed": profile["best_delta_H"] in delta_h_grid() and profile["delta_chi2"] >= 0.0})

    first = synthetic_permutation_test(y, sigma, luminosity_distances, path_distances, directions, l_void, identifiers, 2)
    second = synthetic_permutation_test(y, sigma, luminosity_distances, path_distances, directions, l_void, identifiers, 2)
    gates.append({"check_id": "PCG64_PERMUTATION_TAIL_PLUS_ONE", "passed": first == second and first["p_value"] == (1 + first["tail_count"]) / 3})

    _require(all(bool(gate["passed"]) for gate in gates), "synthetic conformance gate failed")
    return gates


def build_receipt() -> dict[str, Any]:
    config = load_config()
    predecessor = _validate_bound_receipt(config["predecessor"], "PASS_MUTATION_FREEZE_SYNTHETIC_CONFORMANCE_ROWS_UNOPENED_AWAIT_INDEPENDENT_AUDIT")
    audit_block = _validate_audit_block(config["audit_block"])
    law = _validate_bound_receipt(config["law"], "PASS_FAIL_CLOSED_IDENTIFIABLE_LAW_ROWS_UNOPENED")
    geometry = _validate_bound_receipt(config["geometry"], "PASS_DISTANCE_TYPED_FAIL_CLOSED_ROWS_UNOPENED")
    source_bindings = _bind_inputs(config)
    gates = conformance_gates(config)
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-executor-contract-receipt-3.0",
        "package_id": config["package_id"],
        "status": "PASS_NORMALIZE_ONCE_REPAIR_MUTATION_FREEZE_ROWS_UNOPENED_AWAIT_INDEPENDENT_AUDIT",
        "decision": config["decision"],
        "predecessor_content_sha256": predecessor["content_sha256"],
        "predecessor_audit_block": {
            "status": audit_block["status"],
            "raw_sha256": config["audit_block"]["raw_sha256"],
            "counterexample": audit_block["counterexample"],
        },
        "law_content_sha256": law["content_sha256"],
        "geometry_content_sha256": geometry["content_sha256"],
        "v1_blocker_dispositions": config["v1_blocker_dispositions"],
        "canonical_paths": config["canonical_paths"],
        "strict_record_grammar": config["strict_record_grammar"],
        "stf_shear": config["stf_shear"],
        "likelihood": config["likelihood"],
        "profile_solver": config["profile_solver"],
        "distance_strata": config["distance_strata"],
        "permutation_test": config["permutation_test"],
        "duplicate_semantics": config["duplicate_semantics"],
        "split_and_access": config["split_and_access"],
        "scoring_and_roles": config["scoring_and_roles"],
        "geometry_join": config["geometry_join"],
        "excluded_and_remaining": config["excluded_and_remaining"],
        "source_bindings": source_bindings,
        "conformance_gates": gates,
        "access_accounting": config["access_accounting"],
        "mutation_freeze": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
            "predecessor_raw_sha256": config["predecessor"]["raw_sha256"],
            "audit_block_raw_sha256": config["audit_block"]["raw_sha256"],
            "law_raw_sha256": config["law"]["raw_sha256"],
            "geometry_raw_sha256": config["geometry"]["raw_sha256"],
        },
        "next_gate": "A_DIFFERENT_AGENT_MUST_AUDIT_THIS_EXACT_NORMALIZE_ONCE_REPAIR_AND_HASHES_BEFORE_IDS_PARTITIONED",
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt() -> str:
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode() + b"\n"
    return _atomic_no_clobber(OUTPUT_PATH, payload)


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
    else:
        receipt = check_receipt()
        print("VALID" if args.command == "check" else receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
