"""Source-free, audit-block repair for the Lane-9 development release."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import marshal
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import open_gravity_void_correlation_development_release_v1 as v1
from . import open_gravity_void_correlation_executor_contract_v3 as executor_v3
from . import open_gravity_void_correlation_ids_partition_v1 as ids_v1
from . import open_gravity_void_geometry_source_completion_v2 as geometry_v2
from . import open_gravity_void_geometry_source_completion_v3 as geometry_v3
from . import open_gravity_void_gravitational_load_v3 as law_v3
from . import open_gravity_void_gravitational_load_v4 as law_v4

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_correlation_development_release_v2.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_correlation_development_release_v2.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_correlation_development_release_v2.py"
OUTPUT_PATH = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v2/receipt.json"
FINAL_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v2"
STAGING_ROOT = REPO_ROOT / "work/open-gravity-void-correlation-development-score-v2-staging"
FAILURE_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v2-failures"
CONSUMPTION_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v2-authorization-consumed"
_CONFIG_RAW_SHA256 = "c1e146d9bd775f50748a986bc3212b0e1048c42887f3f217a4a9a5b5cb1f48ec"
_CONFIG_CONTENT_SHA256 = "aa3bc243d63b8c21b22d30450927189b5a909844ccd494702919cb5475769b45"
_MODULE_SEMANTIC_SHA256 = "e16e065b2c696466ddf57ed7b987fbbadb04d1f4d7ee0941f69c3a9cd5594233"
_TEST_RAW_SHA256 = "f49845823f2aea4bfc01012354a19ce60f71b916d2dd5ec8b2113d3c3d8a889a"

_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_KEYS = {
    "bucket",
    "canonical_identifier",
    "framed_end_exclusive",
    "framed_raw_sha256",
    "framed_start",
    "identifier",
    "identifier_field_raw_sha256",
    "leaf_sha256",
    "line_ending_bytes",
    "opaque_tail_raw_sha256",
    "payload_end_exclusive",
    "payload_raw_sha256",
    "payload_start",
    "role",
    "source_index",
}
_ARTIFACT_NAMES = {
    "artifacts/development-rows.jsonl",
    "artifacts/profile-grid.jsonl",
    "artifacts/permutation-statistics.jsonl",
    "artifacts/countermodels.json",
    "artifacts/failures.json",
    "artifacts/development-summary.json",
}
_COUNTERMODEL_KEYS = {
    "PRIMARY_UNION_VOID_PATH",
    "C00_FLRW_RADIAL_BULK_SHEAR_NULL",
    "C01_OBSERVER_ENDPOINT_LOCAL_VOID",
    "C02_TARGET_ENDPOINT_LOCAL_VOID",
    "C03_SINGLE_DOMINANT_VOID",
    "C04_STRATIFIED_EXCHANGEABILITY_NULL",
}
_DEVELOPMENT_ROW_KEYS = {
    "identifier",
    "source_index",
    "bucket",
    "role",
    "eligibility",
    "reason_codes",
    "parsed_development_values_hex",
    "z_D_hex",
    "D_path_Mpc_hex",
    "direction_hex",
    "mask_pixel",
    "geometry_hex",
    "union_crossings",
    "y_hex",
    "sigma_s_hex",
    "nuisance_design_hex",
    "law_column_hex",
    "null_prediction_hex",
    "primary_prediction_hex",
    "null_residual_hex",
    "primary_residual_hex",
    "row_leaf_sha256",
}

_RUNTIME_CALLABLES = {
    "geometry_v2.mask_index": geometry_v2.mask_index,
    "geometry_v2.mask_contains": geometry_v2.mask_contains,
    "geometry_v2.radec_to_xyz": geometry_v2.radec_to_xyz,
    "geometry_v2._comoving_mpc": geometry_v2._comoving_mpc,
    "geometry_v2.luminosity_to_comoving_hinv": geometry_v2.luminosity_to_comoving_hinv,
    "geometry_v3.validate_cf4_distance": geometry_v3.validate_cf4_distance,
    "geometry_v3.mask_index": geometry_v3.mask_index,
    "geometry_v3.radec_to_xyz": geometry_v3.radec_to_xyz,
    "geometry_v3.luminosity_to_comoving_hinv": geometry_v3.luminosity_to_comoving_hinv,
    "law_v4.union_intervals": law_v4.union_intervals,
    "law_v4.ray_sphere_intervals": law_v4.ray_sphere_intervals,
    "law_v4.path_partition": law_v4.path_partition,
    "law_v3.union_intervals": law_v3.union_intervals,
    "law_v3.ray_sphere_intervals": law_v3.ray_sphere_intervals,
    "law_v3.path_partition": law_v3.path_partition,
}
_RUNTIME_LOOKUP = {
    "geometry_v2.mask_index": lambda: geometry_v2.mask_index,
    "geometry_v2.mask_contains": lambda: geometry_v2.mask_contains,
    "geometry_v2.radec_to_xyz": lambda: geometry_v2.radec_to_xyz,
    "geometry_v2._comoving_mpc": lambda: geometry_v2._comoving_mpc,
    "geometry_v2.luminosity_to_comoving_hinv": lambda: geometry_v2.luminosity_to_comoving_hinv,
    "geometry_v3.validate_cf4_distance": lambda: geometry_v3.validate_cf4_distance,
    "geometry_v3.mask_index": lambda: geometry_v3.mask_index,
    "geometry_v3.radec_to_xyz": lambda: geometry_v3.radec_to_xyz,
    "geometry_v3.luminosity_to_comoving_hinv": lambda: geometry_v3.luminosity_to_comoving_hinv,
    "law_v4.union_intervals": lambda: law_v4.union_intervals,
    "law_v4.ray_sphere_intervals": lambda: law_v4.ray_sphere_intervals,
    "law_v4.path_partition": lambda: law_v4.path_partition,
    "law_v3.union_intervals": lambda: law_v3.union_intervals,
    "law_v3.ray_sphere_intervals": lambda: law_v3.ray_sphere_intervals,
    "law_v3.path_partition": lambda: law_v3.path_partition,
}
_PINNED_RUNTIME_NAMES = {
    "v1": (
        v1,
        (
            "_frame_payload",
            "record_payload",
            "parse_field",
            "parse_record",
            "parse_cf4_development_record",
            "_validate_cf4_domains",
            "parse_vast_table1_record",
            "parse_vast_table2_record",
            "prepare_vast_geometry",
            "validate_mask",
            "luminosity_to_comoving_hinv",
            "derive_development_row",
            "score_exposure",
            "profile_grid_details",
            "score_countermodels",
            "development_permutation_test",
            "classify_development",
            "development_ledger_rows",
        ),
    ),
    "executor_v3": (
        executor_v3,
        (
            "split_role",
            "observed_log_redshift",
            "normalize_direction",
            "nuisance_velocity_design",
            "velocity_to_log_design",
            "validate_cf4_duplicate_keys",
            "_ordered_inputs",
            "_prepare_profile",
            "_profile_prepared",
            "_cholesky_solve",
            "_tie_tolerance",
            "delta_h_grid",
            "profile_at_delta",
            "profile_grid",
            "distance_strata",
            "_pcg64_permutation_orders",
            "synthetic_permutation_test",
            "stf_basis",
            "validate_stf_basis",
            "shear_quadratic_columns",
        ),
    ),
}
for _runtime_prefix, (_runtime_module, _runtime_names) in _PINNED_RUNTIME_NAMES.items():
    for _runtime_name in _runtime_names:
        _runtime_key = f"{_runtime_prefix}.{_runtime_name}"
        _RUNTIME_CALLABLES[_runtime_key] = getattr(_runtime_module, _runtime_name)
        _RUNTIME_LOOKUP[_runtime_key] = lambda module=_runtime_module, name=_runtime_name: getattr(module, name)


class DevelopmentReleaseV2Error(RuntimeError):
    """Fail-closed v2 release violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentReleaseV2Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    lines = []
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


def _code_sha256(function: Any) -> str:
    _require(callable(function) and hasattr(function, "__code__"), "runtime callable missing code")
    return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()


def canonical_file(relative: str) -> Path:
    _require(isinstance(relative, str) and relative and "\\" not in relative, "invalid canonical path")
    _require(not re.match(r"^[A-Za-z]:", relative), "drive path forbidden")
    _require(all(part not in {"", ".", ".."} for part in relative.split("/")), "unnormalized canonical path")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute(), "absolute canonical path")
    root = REPO_ROOT.resolve(strict=True)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        _require(cursor.exists() and not cursor.is_symlink(), "missing or symlink canonical component")
    target = cursor.resolve(strict=True)
    _require(root in target.parents and target.is_file(), "canonical target invalid")
    return target


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "v2 config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "v2 config content drift")
    _require(value["status"] == "DRAFT_SOURCE_FREE_AUDIT_BLOCK_REPAIR_AWAIT_INDEPENDENT_REAUDIT", "v2 config status drift")
    return value


def validate_code_pins() -> None:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "v2 module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "v2 test drift")


def _json_binding(section: Mapping[str, Any], *, receipt: bool = False) -> dict[str, Any]:
    path = canonical_file(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], f"raw binding drift: {section['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if "content_sha256" in section:
        observed = value.get("content_sha256", content_sha256(value))
        _require(observed == section["content_sha256"], f"content binding drift: {section['path']}")
    if receipt:
        _require(value.get("content_sha256") == _self_hash(value), f"receipt self-hash drift: {section['path']}")
    if "status" in section:
        _require(value.get("status") == section["status"], f"status binding drift: {section['path']}")
    return value


def _module_binding(section: Mapping[str, Any], semantic_function: Any) -> None:
    path = canonical_file(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], f"module raw drift: {section['path']}")
    _require(semantic_function(path) == section["semantic_sha256"], f"module semantic drift: {section['path']}")


def validate_blocked_v1(config: Mapping[str, Any]) -> dict[str, str]:
    blocked = config["blocked_v1"]
    config_value = _json_binding(blocked["config"])
    _require(content_sha256(config_value) == blocked["config"]["content_sha256"], "blocked v1 config content drift")
    _module_binding(blocked["module"], v1.module_semantic_sha256)
    _require(file_sha256(canonical_file(blocked["test"]["path"])) == blocked["test"]["raw_sha256"], "blocked v1 test drift")
    receipt = _json_binding(blocked["receipt"], receipt=True)
    for name in ("config", "module", "test", "receipt"):
        preserved = canonical_file(blocked[name]["preserved_path"])
        original = canonical_file(blocked[name]["path"])
        _require(file_sha256(preserved) == blocked[name]["raw_sha256"], f"blocked preservation drift: {name}")
        _require(preserved.read_bytes() == original.read_bytes(), f"blocked preservation not byte-exact: {name}")
    return {"v1_receipt_content_sha256": receipt["content_sha256"]}


def validate_runtime_dependencies(config: Mapping[str, Any]) -> dict[str, str]:
    dependencies = config["runtime_dependencies"]
    modules = {
        "executor_v3": (executor_v3, executor_v3.module_semantic_sha256),
        "geometry_v2": (geometry_v2, geometry_v2.module_semantic_sha256),
        "geometry_v3": (geometry_v3, geometry_v3.module_semantic_sha256),
        "law_v4": (law_v4, law_v4.module_semantic_sha256),
        "law_v3": (law_v3, law_v3.module_semantic_sha256),
    }
    result: dict[str, str] = {}
    for name, (module, semantic_function) in modules.items():
        section = dependencies[name]
        config_value = _json_binding(section["config"])
        _require(content_sha256(config_value) == section["config"]["content_sha256"], f"{name} config content drift")
        _module_binding(section["module"], semantic_function)
        _require(file_sha256(canonical_file(section["test"]["path"])) == section["test"]["raw_sha256"], f"{name} test drift")
        receipt = _json_binding(section["receipt"], receipt=True)
        result[f"{name}_receipt_content_sha256"] = receipt["content_sha256"]
        if "independent_audit" in section:
            audit = _json_binding(section["independent_audit"], receipt=True)
            result[f"{name}_audit_content_sha256"] = audit["content_sha256"]
        _require(module is not None, f"{name} module unavailable")
    expected_fingerprints = dependencies["callable_code_sha256"]
    _require(set(expected_fingerprints) == set(_RUNTIME_CALLABLES), "runtime callable set drift")
    for name, original in _RUNTIME_CALLABLES.items():
        current = _RUNTIME_LOOKUP[name]()
        _require(current is original, f"runtime callable identity drift: {name}")
        _require(_code_sha256(current) == expected_fingerprints[name], f"runtime callable code drift: {name}")
    _require(
        v1.executor_v3 is executor_v3
        and v1.geometry_v2 is geometry_v2
        and v1.geometry_v3 is geometry_v3
        and v1.law_v4 is law_v4
        and law_v4.v3 is law_v3,
        "runtime dependency wiring drift",
    )
    _require(v1.np.__version__ == "2.2.6", "NumPy runtime version drift")
    return result


def validate_ids_binding(config: Mapping[str, Any]) -> dict[str, str]:
    section = config["ids_binding"]
    config_value = _json_binding(section["config"])
    _require(content_sha256(config_value) == section["config"]["content_sha256"], "IDs config content drift")
    _module_binding(section["module"], ids_v1.module_semantic_sha256)
    _require(file_sha256(canonical_file(section["test"]["path"])) == section["test"]["raw_sha256"], "IDs test drift")
    receipt = _json_binding(section["receipt"], receipt=True)
    audit = _json_binding(section["audit"], receipt=True)
    return {"ids_receipt_content_sha256": receipt["content_sha256"], "ids_audit_content_sha256": audit["content_sha256"]}


def validate_ledger_entry(entry: Mapping[str, Any]) -> None:
    _require(set(entry) == _LEDGER_KEYS, "ledger exact-key mismatch")
    integer_keys = {
        "bucket",
        "framed_end_exclusive",
        "framed_start",
        "identifier",
        "line_ending_bytes",
        "payload_end_exclusive",
        "payload_start",
        "source_index",
    }
    for key in integer_keys:
        _require(isinstance(entry[key], int) and not isinstance(entry[key], bool), f"ledger integer type: {key}")
    for key in (
        "framed_raw_sha256",
        "identifier_field_raw_sha256",
        "leaf_sha256",
        "opaque_tail_raw_sha256",
        "payload_raw_sha256",
    ):
        _require(isinstance(entry[key], str) and _HEX64.fullmatch(entry[key]) is not None, f"ledger hash syntax: {key}")
    identifier = int(entry["identifier"])
    _require(identifier > 0 and entry["canonical_identifier"] == str(identifier), "ledger canonical identifier")
    bucket, role = v1.executor_v3.split_role(identifier)
    _require(int(entry["bucket"]) == bucket and entry["role"] == role, "ledger split drift")
    _require(0 <= int(entry["bucket"]) <= 9 and role in {"development", "validation", "confirmation"}, "ledger role invalid")
    _require(int(entry["line_ending_bytes"]) in (1, 2), "ledger line ending invalid")
    _require(int(entry["payload_start"]) == int(entry["framed_start"]), "ledger payload start drift")
    _require(int(entry["payload_end_exclusive"]) - int(entry["payload_start"]) == 157, "ledger payload length drift")
    _require(
        int(entry["framed_end_exclusive"]) == int(entry["payload_end_exclusive"]) + int(entry["line_ending_bytes"]),
        "ledger framed end drift",
    )
    body = dict(entry)
    leaf = str(body.pop("leaf_sha256"))
    _require(leaf != "0" * 64, "stale zero ledger leaf")
    _require(content_sha256(body) == leaf, "ledger leaf mismatch")


def load_identifier_ledger(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    active = load_config() if config is None else config
    section = active["ids_binding"]["identifier_ledger"]
    path = canonical_file(section["path"])
    _require(file_sha256(path) == section["raw_sha256"], "identifier ledger raw drift")
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(len(values) == section["rows"], "identifier ledger count drift")
    _require(content_sha256(values) == section["content_sha256"], "identifier ledger content drift")
    prior_end = 0
    identifiers: set[int] = set()
    development = 0
    for index, entry in enumerate(values):
        validate_ledger_entry(entry)
        _require(entry["source_index"] == index, "ledger source-index sequence")
        _require(entry["framed_start"] == prior_end, "ledger offset continuity")
        prior_end = int(entry["framed_end_exclusive"])
        identifier = int(entry["identifier"])
        _require(identifier not in identifiers, "ledger duplicate identifier")
        identifiers.add(identifier)
        development += entry["role"] == "development"
    _require(development == section["development_rows"] == 22897, "exact development ledger count drift")
    return values


def parse_cf4_development_record_v2(
    line: bytes,
    ledger_entry: Mapping[str, Any],
    *,
    source_index: int,
    framed_start: int,
) -> dict[str, int | float]:
    validate_ledger_entry(ledger_entry)
    payload = v1._frame_payload(line, 157)
    _require(source_index == ledger_entry["source_index"] and framed_start == ledger_entry["framed_start"], "ledger row position mismatch")
    _require(framed_start + len(line) == ledger_entry["framed_end_exclusive"], "ledger full-row offset mismatch")
    _require(bytes_sha256(line) == ledger_entry["framed_raw_sha256"], "ledger full-row hash mismatch")
    _require(bytes_sha256(payload) == ledger_entry["payload_raw_sha256"], "ledger payload hash mismatch")
    _require(bytes_sha256(payload[:7]) == ledger_entry["identifier_field_raw_sha256"], "ledger ID-slice hash mismatch")
    _require(bytes_sha256(payload[7:]) == ledger_entry["opaque_tail_raw_sha256"], "ledger tail hash mismatch")
    return v1.parse_cf4_development_record(
        line,
        ledger_entry,
        source_index=source_index,
        framed_start=framed_start,
    )


def validate_exact_development_coverage(
    rows: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        (int(entry["identifier"]), int(entry["source_index"]), int(entry["bucket"]), str(entry["role"]))
        for entry in ledger
        if entry["role"] == "development"
    }
    _require(len(expected) == 22897, "expected development coverage drift")
    observed_rows = [
        (int(row["identifier"]), int(row["source_index"]), int(row["bucket"]), str(row["role"]))
        for row in rows
    ]
    _require(len(observed_rows) == 22897, "development output row count must equal 22897")
    _require(len(set(observed_rows)) == len(observed_rows), "duplicate development output row")
    _require(set(observed_rows) == expected, "development output coverage mismatch")
    identifiers = [row[0] for row in observed_rows]
    _require(len(set(identifiers)) == 22897, "development IDs not globally unique")
    return {"count": 22897, "identifier_root_sha256": _simple_root(str(value).encode("ascii") for value in sorted(identifiers))}


def _simple_root(values: Sequence[bytes] | Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(hashlib.sha256(value).digest())
    return digest.hexdigest()


def _leaf_root(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        leaf = str(row["row_leaf_sha256"])
        _require(_HEX64.fullmatch(leaf) is not None, "invalid row leaf")
        body = dict(row)
        body.pop("row_leaf_sha256")
        _require(content_sha256(body) == leaf, "row leaf mismatch")
        digest.update(bytes.fromhex(leaf))
    return digest.hexdigest()


def validate_permutation(
    permutation: Mapping[str, Any],
    primary_summary: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        set(permutation) == {"observed", "permutation_statistics", "tail_count", "p_value"},
        "permutation exact-key mismatch",
    )
    observed = float(permutation["observed"])
    expected_observed = float(primary_summary["one_sided_statistic"])
    _require(math.isfinite(observed) and observed == expected_observed, "permutation observed mismatch")
    statistics = [float(value) for value in permutation["permutation_statistics"]]
    _require(len(statistics) == 10000 and all(math.isfinite(value) for value in statistics), "permutation count or finite failure")
    tail = sum(value >= observed for value in statistics)
    _require(isinstance(permutation["tail_count"], int) and not isinstance(permutation["tail_count"], bool), "tail count type")
    _require(int(permutation["tail_count"]) == tail, "permutation tail mismatch")
    p_value = (1 + tail) / 10001
    _require(float(permutation["p_value"]) == p_value, "permutation plus-one p mismatch")
    return {"observed": observed, "statistics": statistics, "tail_count": tail, "p_value": p_value}


def exact_countermodels(
    profile_scores: Mapping[str, Mapping[str, Any]],
    permutation: Mapping[str, Any],
    permutation_root: str,
) -> dict[str, Any]:
    expected_profiles = _COUNTERMODEL_KEYS - {"C04_STRATIFIED_EXCHANGEABILITY_NULL"}
    _require(set(profile_scores) == expected_profiles, "profile countermodel exact-set mismatch")
    full_profile_keys = {"best_delta_H", "best_chi2", "null_chi2", "delta_chi2", "one_sided_statistic", "tied_delta_H"}
    for key in expected_profiles - {"C00_FLRW_RADIAL_BULK_SHEAR_NULL"}:
        _require(set(profile_scores[key]) == full_profile_keys, f"countermodel profile schema: {key}")
    _require(
        set(profile_scores["C00_FLRW_RADIAL_BULK_SHEAR_NULL"]) == {"delta_chi2", "best_delta_H", "null_chi2"},
        "C00 profile schema",
    )
    result = {name: dict(profile_scores[name]) for name in sorted(expected_profiles)}
    result["C04_STRATIFIED_EXCHANGEABILITY_NULL"] = {
        "kind": "STRATIFIED_EXCHANGEABILITY_NULL",
        "seed": 902104729,
        "permutations": 10000,
        "observed": float(permutation["observed"]),
        "tail_count": int(permutation["tail_count"]),
        "p_value": float(permutation["p_value"]),
        "permutation_root_sha256": permutation_root,
    }
    _require(set(result) == _COUNTERMODEL_KEYS, "countermodel exact-set drift")
    _require(result["C04_STRATIFIED_EXCHANGEABILITY_NULL"] != result["C00_FLRW_RADIAL_BULK_SHEAR_NULL"], "C04 aliases C00")
    return result


def _hex(value: float) -> str:
    observed = float(value)
    _require(math.isfinite(observed), "nonfinite float serialization")
    return observed.hex()


def _hex_tree(value: Any) -> Any:
    if isinstance(value, float):
        return _hex(value)
    if isinstance(value, dict):
        return {str(key): _hex_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hex_tree(item) for item in value]
    return value


def _row_with_leaf(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    _require("row_leaf_sha256" not in result, "preexisting row leaf")
    result["row_leaf_sha256"] = content_sha256(result)
    return result


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _parse_jsonl(payload: bytes) -> list[dict[str, Any]]:
    _require(payload.endswith(b"\n"), "JSONL terminal newline missing")
    lines = payload.splitlines()
    values = [json.loads(line) for line in lines]
    _require(all(isinstance(value, dict) for value in values), "JSONL row not object")
    return values


def _require_finite_hex(value: Any, label: str) -> float:
    _require(isinstance(value, str), f"{label} not hex string")
    try:
        observed = float.fromhex(value)
    except ValueError as error:
        raise DevelopmentReleaseV2Error(f"{label} invalid float.hex") from error
    _require(math.isfinite(observed) and observed.hex() == value, f"{label} noncanonical or nonfinite float.hex")
    return observed


def _validate_development_output_row(row: Mapping[str, Any]) -> None:
    _require(set(row) == _DEVELOPMENT_ROW_KEYS, "development row schema mismatch")
    for key in ("identifier", "source_index", "bucket", "union_crossings"):
        _require(isinstance(row[key], int) and not isinstance(row[key], bool), f"development integer type: {key}")
    _require(row["role"] == "development", "nondevelopment output role")
    _require(row["eligibility"] in {"PRIMARY", "PARTIAL_MASK_EXCLUDED"}, "eligibility enum")
    _require(isinstance(row["reason_codes"], list) and all(reason in {"ANGULAR_MASK_FALSE", "OUTSIDE_RADIAL_MASK", "UNOBSERVED_PATH"} for reason in row["reason_codes"]), "reason-code schema")
    _require(isinstance(row["mask_pixel"], bool), "mask pixel type")
    _require(isinstance(row["parsed_development_values_hex"], dict), "parsed values schema")
    _require(
        set(row["parsed_development_values_hex"])
        == {"1PGC", "source_index", "DMzp", "e_DMzp", "Dist", "V3k", "RAdeg", "DEdeg"},
        "parsed value key set",
    )
    for key in ("1PGC", "source_index", "V3k"):
        _require(isinstance(row["parsed_development_values_hex"][key], int) and not isinstance(row["parsed_development_values_hex"][key], bool), f"parsed integer type: {key}")
    for key in ("DMzp", "e_DMzp", "Dist", "RAdeg", "DEdeg"):
        _require_finite_hex(row["parsed_development_values_hex"][key], f"parsed {key}")
    for key in ("z_D_hex", "D_path_Mpc_hex", "y_hex", "sigma_s_hex", "law_column_hex"):
        _require_finite_hex(row[key], key)
    _require(isinstance(row["direction_hex"], list) and len(row["direction_hex"]) == 3, "direction schema")
    _require(isinstance(row["nuisance_design_hex"], list) and len(row["nuisance_design_hex"]) == 9, "nuisance schema")
    for index, value in enumerate(row["direction_hex"]):
        _require_finite_hex(value, f"direction {index}")
    for index, value in enumerate(row["nuisance_design_hex"]):
        _require_finite_hex(value, f"nuisance {index}")
    geometry_keys = {
        "L_void_Mpc",
        "L_observed_matter_Mpc",
        "L_unobserved_Mpc",
        "void_fraction",
        "maximum_chord_Mpc",
        "observer_endpoint_chord_Mpc",
        "target_endpoint_chord_Mpc",
    }
    _require(isinstance(row["geometry_hex"], dict) and set(row["geometry_hex"]) == geometry_keys, "geometry schema")
    for key, value in row["geometry_hex"].items():
        _require_finite_hex(value, f"geometry {key}")
    prediction_keys = ("null_prediction_hex", "primary_prediction_hex", "null_residual_hex", "primary_residual_hex")
    if row["eligibility"] == "PRIMARY":
        _require(all(row[key] is not None for key in prediction_keys), "primary prediction missing")
        for key in prediction_keys:
            _require_finite_hex(row[key], key)
    else:
        _require(all(row[key] is None for key in prediction_keys), "partial-mask prediction leak")


def _reconstruct_scored_rows(ledger_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct only already-emitted development values for an exact score replay."""
    result: list[dict[str, Any]] = []
    float_cf4 = {"DMzp", "e_DMzp", "Dist", "RAdeg", "DEdeg"}
    for row in ledger_rows:
        parsed = row["parsed_development_values_hex"]
        cf4 = {
            key: _require_finite_hex(value, f"replay CF4 {key}") if key in float_cf4 else value
            for key, value in parsed.items()
        }
        geometry = {
            key: _require_finite_hex(value, f"replay geometry {key}")
            for key, value in row["geometry_hex"].items()
        }
        result.append(
            {
                "identifier": row["identifier"],
                "source_index": row["source_index"],
                "bucket": row["bucket"],
                "role": row["role"],
                "eligible_primary": row["eligibility"] == "PRIMARY",
                "reason_codes": list(row["reason_codes"]),
                "cf4": cf4,
                "z_D": _require_finite_hex(row["z_D_hex"], "replay z_D"),
                "D_path_Mpc": _require_finite_hex(row["D_path_Mpc_hex"], "replay path"),
                "direction": tuple(
                    _require_finite_hex(value, "replay direction") for value in row["direction_hex"]
                ),
                "mask_pixel": row["mask_pixel"],
                **geometry,
                "union_crossings": row["union_crossings"],
                "y": _require_finite_hex(row["y_hex"], "replay response"),
                "sigma_s": _require_finite_hex(row["sigma_s_hex"], "replay sigma"),
                "nuisance_design_log": tuple(
                    _require_finite_hex(value, "replay nuisance") for value in row["nuisance_design_hex"]
                ),
                "law_column": _require_finite_hex(row["law_column_hex"], "replay law column"),
            }
        )
    return result


def sanitize_failure(
    stage: str,
    reason: str,
    access_counts: Mapping[str, Any],
    identifiers: Sequence[int],
    audited_development_ids: set[int],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active = load_config() if config is None else config
    contract = active["failure_contract"]
    _require(stage in contract["stages"], "failure stage outside enum")
    _require(reason in contract["reasons"], "failure reason outside enum")
    _require(set(access_counts) == set(contract["access_count_keys"]), "failure access-count key mismatch")
    counts: dict[str, int] = {}
    for key in contract["access_count_keys"]:
        value = access_counts[key]
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"invalid failure access count: {key}")
        counts[key] = value
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in identifiers),
        "failure identifier integer type",
    )
    ids = list(identifiers)
    _require(len(ids) == len(set(ids)), "duplicate failure identifier")
    _require(all(value > 0 and v1.executor_v3.split_role(value)[1] == "development" for value in ids), "sealed failure identifier")
    _require(set(ids).issubset(audited_development_ids), "failure identifier absent from audited development set")
    return {
        "stage": stage,
        "reason_code": reason,
        "access_counts": counts,
        "authorized_development_ids": sorted(ids),
    }


def expected_success_access_counts() -> dict[str, int]:
    return {
        "authorization_consumptions": 1,
        "cf4_source_opens": 1,
        "cf4_gzip_passes": 1,
        "cf4_identifier_rows_reverified": 38053,
        "cf4_development_scientific_rows_decoded": 22897,
        "cf4_validation_scientific_rows_decoded": 0,
        "cf4_confirmation_scientific_rows_decoded": 0,
        "vast_table1_source_opens": 1,
        "vast_table1_rows_decoded": 2347,
        "vast_table2_source_opens": 1,
        "vast_table2_gzip_passes": 1,
        "vast_table2_rows_decoded": 80080,
        "mask_source_opens": 1,
        "pantheon_source_opens": 0,
        "development_scores": 1,
    }


def assemble_development_artifacts_v2(
    rows: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    profile_details: Mapping[str, Any],
    permutation: Mapping[str, Any],
    profile_countermodels: Mapping[str, Mapping[str, Any]],
    access_counts: Mapping[str, int],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    coverage = validate_exact_development_coverage(rows, ledger)
    _require(dict(access_counts) == expected_success_access_counts(), "success access counts not exact")
    primary_summary = profile_details["summary"]
    checked_permutation = validate_permutation(permutation, primary_summary)
    development_rows = v1.development_ledger_rows(rows, profile_details)
    _require(len(development_rows) == 22897, "development ledger output count drift")
    profile_rows = [
        _row_with_leaf(
            {
                "grid_index": index,
                "delta_H_hex": _hex(profile["delta_H"]),
                "chi2_hex": _hex(profile["chi2"]),
                "beta_hex": [_hex(value) for value in profile["beta"]],
            }
        )
        for index, profile in enumerate(profile_details["profiles"])
    ]
    _require(len(profile_rows) == 161 and [row["grid_index"] for row in profile_rows] == list(range(161)), "profile grid schema drift")
    permutation_rows = [
        _row_with_leaf({"permutation_index": index, "statistic_hex": _hex(value)})
        for index, value in enumerate(checked_permutation["statistics"])
    ]
    _require(len(permutation_rows) == 10000, "permutation artifact count drift")
    permutation_root = _leaf_root(permutation_rows)
    countermodels = exact_countermodels(profile_countermodels, checked_permutation, permutation_root)
    classification = v1.classify_development(
        primary_summary,
        permutation,
        profile_countermodels,
        sum(bool(row["eligible_primary"]) for row in rows),
    )
    countermodels_payload = _pretty(_hex_tree(countermodels))
    audited_ids = {int(entry["identifier"]) for entry in ledger if entry["role"] == "development"}
    sanitized_failures = [
        sanitize_failure(
            str(row["stage"]),
            str(row["reason_code"]),
            row["access_counts"],
            row["authorized_development_ids"],
            audited_ids,
        )
        for row in failures
    ]
    failures_payload = _pretty(sanitized_failures)
    eligible_rows = [row for row in development_rows if row["eligibility"] == "PRIMARY"]
    partial_rows = [row for row in development_rows if row["eligibility"] == "PARTIAL_MASK_EXCLUDED"]
    roots = {
        "development_ledger_root_sha256": _leaf_root(development_rows),
        "eligible_primary_root_sha256": _leaf_root(eligible_rows),
        "partial_mask_root_sha256": _leaf_root(partial_rows),
        "profile_grid_root_sha256": _leaf_root(profile_rows),
        "permutation_root_sha256": permutation_root,
        "countermodels_content_sha256": content_sha256(json.loads(countermodels_payload)),
        "failures_content_sha256": content_sha256(sanitized_failures),
    }
    summary = {
        "schema": "invariant-open-gravity-void-correlation-development-summary-2.0",
        "development_rows": 22897,
        "eligible_primary_rows": len(eligible_rows),
        "partial_mask_rows": len(partial_rows),
        "coverage": coverage,
        "profile": _hex_tree(primary_summary),
        "permutation": {
            "observed_hex": _hex(checked_permutation["observed"]),
            "tail_count": checked_permutation["tail_count"],
            "p_value_hex": _hex(checked_permutation["p_value"]),
            "rows": 10000,
        },
        "countermodels": _hex_tree(countermodels),
        "classification": _hex_tree(classification),
        "roots": roots,
        "access_counts": {key: int(value) for key, value in sorted(access_counts.items())},
        "claim_ceiling": v1.load_config()["claim_ceiling"],
    }
    return {
        "artifacts/development-rows.jsonl": _jsonl(development_rows),
        "artifacts/profile-grid.jsonl": _jsonl(profile_rows),
        "artifacts/permutation-statistics.jsonl": _jsonl(permutation_rows),
        "artifacts/countermodels.json": countermodels_payload,
        "artifacts/failures.json": failures_payload,
        "artifacts/development-summary.json": _pretty(summary),
    }


def _authorization_binding(contract_receipt: Mapping[str, Any]) -> dict[str, str]:
    return {
        "config_raw_sha256": contract_receipt["mutation_freeze"]["config_raw_sha256"],
        "config_content_sha256": contract_receipt["mutation_freeze"]["config_content_sha256"],
        "module_raw_sha256": contract_receipt["mutation_freeze"]["module_raw_sha256"],
        "module_semantic_sha256": contract_receipt["mutation_freeze"]["module_semantic_sha256"],
        "test_raw_sha256": contract_receipt["mutation_freeze"]["test_raw_sha256"],
        "receipt_raw_sha256": file_sha256(OUTPUT_PATH),
        "receipt_content_sha256": contract_receipt["content_sha256"],
    }


def validate_authorization_bytes(payload: bytes, contract_receipt: Mapping[str, Any]) -> dict[str, Any]:
    config = load_config()
    authorization = json.loads(payload)
    required = {
        "schema",
        "status",
        "decision",
        "authorization_id",
        "uses_allowed",
        "hard_seals",
        "contract_binding",
        "content_sha256",
    }
    _require(set(authorization) == required, "authorization exact-key mismatch")
    contract = config["authorization_contract"]
    _require(authorization["schema"] == contract["schema"], "authorization schema drift")
    _require(authorization["status"] == contract["status"] and authorization["decision"] == contract["decision"], "authorization decision drift")
    _require(_HEX64.fullmatch(str(authorization["authorization_id"])) is not None, "authorization ID syntax")
    _require(authorization["uses_allowed"] == 1 and not isinstance(authorization["uses_allowed"], bool), "authorization use count")
    _require(authorization["hard_seals"] == contract["required_hard_seals"], "authorization hard-seal drift")
    _require(authorization["contract_binding"] == _authorization_binding(contract_receipt), "authorization contract binding drift")
    _require(authorization["content_sha256"] == _self_hash(authorization), "authorization self-hash drift")
    authorization["raw_sha256"] = bytes_sha256(payload)
    return authorization


def _artifact_content(name: str, payload: bytes) -> str:
    if name.endswith(".jsonl"):
        return content_sha256(_parse_jsonl(payload))
    return content_sha256(json.loads(payload))


def build_final_development_receipt(
    artifacts: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(artifacts) == _ARTIFACT_NAMES, "final artifact exact-set mismatch")
    authorization = validate_authorization_bytes(authorization_payload, contract_receipt)
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    _require(summary["development_rows"] == 22897 and summary["permutation"]["rows"] == 10000, "final summary count drift")
    artifact_index = {
        name: {
            "bytes": len(payload),
            "raw_sha256": bytes_sha256(payload),
            "content_sha256": _artifact_content(name, payload),
        }
        for name, payload in sorted(artifacts.items())
    }
    config = load_config()
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-score-receipt-2.0",
        "package_id": "open-gravity-void-correlation-development-score-v2",
        "status": "PASS_DEVELOPMENT_SCORE_FROZEN_VALIDATION_CONFIRMATION_PANTHEON_SEALED",
        "decision": summary["classification"]["empirical_label"],
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "raw_sha256": authorization["raw_sha256"],
            "content_sha256": authorization["content_sha256"],
            "uses_allowed": 1,
        },
        "release_chain": {
            "blocked_v1": config["blocked_v1"],
            "ids_binding": config["ids_binding"],
            "runtime_dependencies": config["runtime_dependencies"],
            "development_release_v2": _authorization_binding(contract_receipt),
        },
        "artifacts": artifact_index,
        "roots": summary["roots"],
        "counts": {
            "development_rows": summary["development_rows"],
            "eligible_primary_rows": summary["eligible_primary_rows"],
            "partial_mask_rows": summary["partial_mask_rows"],
            "permutations": summary["permutation"]["rows"],
        },
        "countermodels": summary["countermodels"],
        "access_counts": summary["access_counts"],
        "hard_seals": config["authorization_contract"]["required_hard_seals"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def finalize_development_artifacts(
    artifacts: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> dict[str, bytes]:
    receipt = build_final_development_receipt(artifacts, authorization_payload, contract_receipt)
    return {**artifacts, "receipt.json": _pretty(receipt)}


def validate_package_payloads(
    payloads: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(payloads) == _ARTIFACT_NAMES | {"receipt.json"}, "package file-set mismatch")
    artifacts = {name: payloads[name] for name in _ARTIFACT_NAMES}
    expected = build_final_development_receipt(artifacts, authorization_payload, contract_receipt)
    observed = json.loads(payloads["receipt.json"])
    _require(observed == expected and observed["content_sha256"] == _self_hash(observed), "final receipt mismatch")
    for name, index in observed["artifacts"].items():
        _require(index["raw_sha256"] == bytes_sha256(payloads[name]), f"artifact raw mismatch: {name}")
        _require(index["content_sha256"] == _artifact_content(name, payloads[name]), f"artifact content mismatch: {name}")
    summary = json.loads(payloads["artifacts/development-summary.json"])
    _require(
        set(summary)
        == {
            "schema",
            "development_rows",
            "eligible_primary_rows",
            "partial_mask_rows",
            "coverage",
            "profile",
            "permutation",
            "countermodels",
            "classification",
            "roots",
            "access_counts",
            "claim_ceiling",
        },
        "summary exact-key mismatch",
    )
    _require(summary["schema"] == "invariant-open-gravity-void-correlation-development-summary-2.0", "summary schema drift")
    _require(set(summary["coverage"]) == {"count", "identifier_root_sha256"} and summary["coverage"]["count"] == 22897, "summary coverage schema")
    _require(
        set(summary["profile"])
        == {"best_delta_H", "best_chi2", "null_chi2", "delta_chi2", "one_sided_statistic", "tied_delta_H"},
        "summary profile schema",
    )
    _require(set(summary["permutation"]) == {"observed_hex", "tail_count", "p_value_hex", "rows"}, "summary permutation schema")
    _require(
        set(summary["classification"])
        == {"request_validation", "empirical_label", "path_accumulation_specific", "grid_boundary", "comparator_max_delta_chi2", "physics_veto_applied"},
        "summary classification schema",
    )
    ledger = _parse_jsonl(payloads["artifacts/development-rows.jsonl"])
    grid = _parse_jsonl(payloads["artifacts/profile-grid.jsonl"])
    permutations = _parse_jsonl(payloads["artifacts/permutation-statistics.jsonl"])
    _require(len(ledger) == 22897 and len(grid) == 161 and len(permutations) == 10000, "package row-count mismatch")
    for row in ledger:
        _validate_development_output_row(row)
    _require(all(set(row) == {"grid_index", "delta_H_hex", "chi2_hex", "beta_hex", "row_leaf_sha256"} for row in grid), "profile row schema mismatch")
    _require(all(set(row) == {"permutation_index", "statistic_hex", "row_leaf_sha256"} for row in permutations), "permutation row schema mismatch")
    _require([row["grid_index"] for row in grid] == list(range(161)), "profile index sequence mismatch")
    _require([row["permutation_index"] for row in permutations] == list(range(10000)), "permutation index sequence mismatch")
    for row in grid:
        _require_finite_hex(row["delta_H_hex"], "grid delta")
        _require_finite_hex(row["chi2_hex"], "grid chi2")
        _require(isinstance(row["beta_hex"], list) and len(row["beta_hex"]) == 9, "grid beta schema")
        for value in row["beta_hex"]:
            _require_finite_hex(value, "grid beta")
    for row in permutations:
        _require_finite_hex(row["statistic_hex"], "permutation statistic")
    expected_root_keys = set(load_config()["output_contract"]["roots"])
    _require(set(summary["roots"]) == expected_root_keys, "summary root exact-set mismatch")
    _require(_leaf_root(ledger) == summary["roots"]["development_ledger_root_sha256"], "ledger root mismatch")
    _require(_leaf_root(grid) == summary["roots"]["profile_grid_root_sha256"], "profile root mismatch")
    _require(_leaf_root(permutations) == summary["roots"]["permutation_root_sha256"], "permutation root mismatch")
    eligible = [row for row in ledger if row["eligibility"] == "PRIMARY"]
    partial = [row for row in ledger if row["eligibility"] == "PARTIAL_MASK_EXCLUDED"]
    _require(len(eligible) == summary["eligible_primary_rows"] and len(partial) == summary["partial_mask_rows"], "eligibility count mismatch")
    _require(_leaf_root(eligible) == summary["roots"]["eligible_primary_root_sha256"], "eligible root mismatch")
    _require(_leaf_root(partial) == summary["roots"]["partial_mask_root_sha256"], "partial root mismatch")
    audited_ledger = load_identifier_ledger()
    observed_coverage = validate_exact_development_coverage(ledger, audited_ledger)
    _require(summary["coverage"] == observed_coverage, "summary coverage root mismatch")
    replay_rows = _reconstruct_scored_rows(ledger)
    replay_details = v1.profile_grid_details(replay_rows)
    replay_grid = [
        _row_with_leaf(
            {
                "grid_index": index,
                "delta_H_hex": _hex(profile["delta_H"]),
                "chi2_hex": _hex(profile["chi2"]),
                "beta_hex": [_hex(value) for value in profile["beta"]],
            }
        )
        for index, profile in enumerate(replay_details["profiles"])
    ]
    _require(grid == replay_grid, "profile grid does not exactly replay from per-object ledger")
    _require(summary["profile"] == _hex_tree(replay_details["summary"]), "summary profile does not exactly replay")
    replay_ledger = v1.development_ledger_rows(replay_rows, replay_details)
    _require(ledger == replay_ledger, "per-object predictions or residuals do not exactly replay")
    replay_models = v1.score_countermodels(replay_rows)
    countermodels = json.loads(payloads["artifacts/countermodels.json"])
    _require(set(countermodels) == _COUNTERMODEL_KEYS, "countermodel file set mismatch")
    _require(content_sha256(countermodels) == summary["roots"]["countermodels_content_sha256"], "countermodel content root mismatch")
    statistics = [_require_finite_hex(row["statistic_hex"], "permutation statistic") for row in permutations]
    _require(all(math.isfinite(value) for value in statistics), "permutation artifact nonfinite")
    observed_statistic = _require_finite_hex(summary["permutation"]["observed_hex"], "summary observed")
    p_value = _require_finite_hex(summary["permutation"]["p_value_hex"], "summary p")
    checked_permutation = validate_permutation(
        {
            "observed": observed_statistic,
            "permutation_statistics": statistics,
            "tail_count": summary["permutation"]["tail_count"],
            "p_value": p_value,
        },
        replay_details["summary"],
    )
    expected_countermodels = _hex_tree(
        exact_countermodels(
            replay_models,
            checked_permutation,
            summary["roots"]["permutation_root_sha256"],
        )
    )
    _require(countermodels == expected_countermodels, "countermodels do not exactly replay from per-object ledger")
    expected_classification = _hex_tree(
        v1.classify_development(
            replay_details["summary"],
            checked_permutation,
            replay_models,
            len(eligible),
        )
    )
    _require(summary["classification"] == expected_classification, "classification does not exactly replay")
    failures = json.loads(payloads["artifacts/failures.json"])
    _require(isinstance(failures, list), "failures artifact not a list")
    _require(content_sha256(failures) == summary["roots"]["failures_content_sha256"], "failure content root mismatch")
    audited_ids = {int(entry["identifier"]) for entry in audited_ledger if entry["role"] == "development"}
    for failure in failures:
        _require(
            isinstance(failure, dict)
            and set(failure) == {"stage", "reason_code", "access_counts", "authorized_development_ids"},
            "failure record exact-key mismatch",
        )
        canonical_failure = sanitize_failure(
            failure["stage"],
            failure["reason_code"],
            failure["access_counts"],
            failure["authorized_development_ids"],
            audited_ids,
        )
        _require(failure == canonical_failure, "failure record noncanonical")
    _require(summary["access_counts"] == expected_success_access_counts(), "summary access counts mismatch")
    _require(summary["countermodels"] == countermodels, "summary countermodels incomplete")
    _require(observed["roots"] == summary["roots"], "receipt roots differ from summary")
    _require(observed["countermodels"] == countermodels, "receipt countermodels differ from summary")
    return observed


def _validate_fixed_directory(path: Path, expected: Path) -> None:
    _require(path.is_absolute() and path == expected, "noncanonical fixed directory")
    root = REPO_ROOT.resolve(strict=True)
    _require(not REPO_ROOT.is_symlink() and REPO_ROOT == root, "repository root is redirected")
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise DevelopmentReleaseV2Error("fixed directory escapes repository") from error
    cursor = REPO_ROOT
    for index, part in enumerate(relative.parts):
        cursor = cursor / part
        _require(not cursor.is_symlink(), "fixed path contains symlink")
        if cursor.exists():
            _require(cursor.is_dir(), "fixed path component is not directory")
        elif index < len(relative.parts) - 1:
            break
    _require(path.resolve(strict=False) == path, "fixed directory resolves noncanonically")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory_posix(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_directory_promote(staging: Path, final: Path) -> None:
    _require(staging.is_dir() and not staging.is_symlink(), "invalid staging directory")
    _require(not final.exists() and not final.is_symlink(), "final directory already exists or is redirected")
    if os.name != "nt":
        final.mkdir()
        try:
            os.rename(staging, final)
        except Exception:
            if final.exists() and not any(final.iterdir()):
                final.rmdir()
            raise
        _fsync_directory_posix(final.parent)
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file_ex = kernel32.MoveFileExW
    move_file_ex.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    move_file_ex.restype = ctypes.c_int
    movefile_write_through = 0x00000008
    success = move_file_ex(str(staging), str(final), movefile_write_through)
    _require(bool(success), f"MoveFileExW failed: {ctypes.get_last_error()}")


def _write_payload_tree(staging: Path, payloads: Mapping[str, bytes]) -> None:
    for name, payload in sorted(payloads.items()):
        _require(name in _ARTIFACT_NAMES | {"receipt.json"}, "unexpected payload path")
        target = staging / Path(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _require(file_sha256(target) == bytes_sha256(payload), "staged file hash mismatch")
    _fsync_directory_posix(staging / "artifacts")
    _fsync_directory_posix(staging)


def _read_fixed_package(directory: Path) -> dict[str, bytes]:
    _require(directory.is_dir() and not directory.is_symlink(), "package directory invalid or redirected")
    expected_paths = _ARTIFACT_NAMES | {"receipt.json"}
    observed_entries = list(directory.rglob("*"))
    _require(all(not path.is_symlink() for path in observed_entries), "package contains symlink")
    observed_paths = {path.relative_to(directory).as_posix() for path in observed_entries if path.is_file()}
    observed_directories = {path.relative_to(directory).as_posix() for path in observed_entries if path.is_dir()}
    _require(observed_paths == expected_paths, "existing package file-set mismatch")
    _require(observed_directories == {"artifacts"}, "existing package directory-set mismatch")
    result: dict[str, bytes] = {}
    for name in expected_paths:
        target = directory / Path(*PurePosixPath(name).parts)
        _require(target.is_file() and not target.is_symlink(), "package target invalid or redirected")
        result[name] = target.read_bytes()
    return result


_COMPLETION_SECRET = object()


class _OneShotCompletion:
    """Unforgeable-by-contract completion emitted only after exact gate finalization."""

    def __init__(self, secret: object, authorization_id: str, counts: Mapping[str, int]) -> None:
        _require(secret is _COMPLETION_SECRET, "invalid one-shot completion constructor")
        self._secret = secret
        self._authorization_id = authorization_id
        self._counts = dict(counts)
        self._used = False

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def consume(self, authorization_id: str) -> dict[str, int]:
        _require(self._secret is _COMPLETION_SECRET and not self._used, "one-shot completion replay")
        _require(self._authorization_id == authorization_id, "one-shot completion authorization mismatch")
        _require(self._counts == expected_success_access_counts(), "one-shot completion count mismatch")
        self._used = True
        return dict(self._counts)


def promote_fixed_package(
    artifacts: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
    completion: _OneShotCompletion,
) -> str:
    _validate_fixed_directory(FINAL_DIRECTORY, REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v2")
    _validate_fixed_directory(STAGING_ROOT, REPO_ROOT / "work/open-gravity-void-correlation-development-score-v2-staging")
    authorization = validate_authorization_bytes(authorization_payload, contract_receipt)
    _require(isinstance(completion, _OneShotCompletion), "missing one-shot completion")
    completion.consume(authorization["authorization_id"])
    _validate_fixed_directory(
        CONSUMPTION_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v2-authorization-consumed",
    )
    marker_path = CONSUMPTION_DIRECTORY / f"{authorization['authorization_id']}.json"
    _require(marker_path.is_file() and not marker_path.is_symlink(), "authorization not consumed before promotion")
    observed_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    _require(observed_marker == _consumption_marker(authorization, contract_receipt), "authorization consumption marker drift")
    _require(observed_marker["content_sha256"] == _self_hash(observed_marker), "authorization marker self-hash drift")
    payloads = finalize_development_artifacts(artifacts, authorization_payload, contract_receipt)
    if FINAL_DIRECTORY.exists():
        observed = _read_fixed_package(FINAL_DIRECTORY)
        validate_package_payloads(observed, authorization_payload, contract_receipt)
        _require(observed == payloads, "existing package differs")
        return "EXISTING_IDENTICAL"
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="run-", dir=STAGING_ROOT)).resolve()
    _require(STAGING_ROOT.resolve() in staging.parents, "staging containment failure")
    try:
        _write_payload_tree(staging, payloads)
        validate_package_payloads(_read_fixed_package(staging), authorization_payload, contract_receipt)
        FINAL_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
        _atomic_directory_promote(staging, FINAL_DIRECTORY)
        _require(not staging.exists(), "staging survived promotion")
        return "PROMOTED_COMPLETE"
    except Exception:
        if staging.exists():
            _require(STAGING_ROOT.resolve() in staging.parents, "cleanup containment failure")
            shutil.rmtree(staging)
        raise


def build_failure_receipt(
    failure: Mapping[str, Any],
    authorization_id: str,
    contract_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _require(_HEX64.fullmatch(authorization_id) is not None, "failure authorization ID syntax")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-failure-receipt-2.0",
        "status": "RETAINED_DEVELOPMENT_FAILURE_NO_PARTIAL_SUCCESS",
        "authorization_id": authorization_id,
        "failure": dict(failure),
        "contract_binding": _authorization_binding(contract_receipt),
        "hard_seals_preserved": load_config()["authorization_contract"]["required_hard_seals"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def write_fixed_failure_receipt(
    failure: Mapping[str, Any],
    authorization_id: str,
    contract_receipt: Mapping[str, Any],
) -> str:
    _validate_fixed_directory(
        FAILURE_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v2-failures",
    )
    _require(dict(contract_receipt) == check_receipt(), "failure receipt contract binding drift")
    audited_ids = {
        int(entry["identifier"])
        for entry in load_identifier_ledger()
        if entry["role"] == "development"
    }
    _require(
        isinstance(failure, Mapping)
        and set(failure) == {"stage", "reason_code", "access_counts", "authorized_development_ids"},
        "failure receipt input schema",
    )
    canonical_failure = sanitize_failure(
        failure["stage"],
        failure["reason_code"],
        failure["access_counts"],
        failure["authorized_development_ids"],
        audited_ids,
    )
    _require(dict(failure) == canonical_failure, "failure receipt input not canonical")
    receipt = build_failure_receipt(canonical_failure, authorization_id, contract_receipt)
    payload = _pretty(receipt)
    FAILURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _validate_fixed_directory(
        FAILURE_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v2-failures",
    )
    target = FAILURE_DIRECTORY / f"{authorization_id}-{receipt['content_sha256']}.json"
    if target.exists():
        _require(target.is_file() and not target.is_symlink(), "existing failure receipt redirected")
        _require(target.read_bytes() == payload, "existing failure receipt differs")
        return "EXISTING_IDENTICAL"
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory_posix(FAILURE_DIRECTORY)
    return "CREATED"


def _consumption_marker(authorization: Mapping[str, Any], contract_receipt: Mapping[str, Any]) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-authorization-consumption-2.0",
        "authorization_id": authorization["authorization_id"],
        "authorization_content_sha256": authorization["content_sha256"],
        "authorization_raw_sha256": authorization["raw_sha256"],
        "contract_receipt_content_sha256": contract_receipt["content_sha256"],
        "uses_consumed": 1,
        "content_sha256": "",
    }
    marker["content_sha256"] = _self_hash(marker)
    return marker


def consume_authorization_at(
    directory: Path,
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = validate_authorization_bytes(authorization_payload, contract_receipt)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{authorization['authorization_id']}.json"
    marker = _consumption_marker(authorization, contract_receipt)
    _require(not target.exists(), "authorization replay")
    with target.open("xb") as handle:
        handle.write(_pretty(marker))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory_posix(directory)
    return marker


def consume_fixed_authorization(
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_fixed_directory(
        CONSUMPTION_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v2-authorization-consumed",
    )
    marker = consume_authorization_at(CONSUMPTION_DIRECTORY, authorization_payload, contract_receipt)
    _validate_fixed_directory(
        CONSUMPTION_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v2-authorization-consumed",
    )
    return marker


class OneShotDevelopmentGate:
    """State machine that makes a second pass or sealed scientific decode impossible."""

    def __init__(self, ledger: Sequence[Mapping[str, Any]], consumption_marker: Mapping[str, Any]) -> None:
        _require(
            set(consumption_marker)
            == {
                "schema",
                "authorization_id",
                "authorization_content_sha256",
                "authorization_raw_sha256",
                "contract_receipt_content_sha256",
                "uses_consumed",
                "content_sha256",
            }
            and consumption_marker.get("schema")
            == "invariant-open-gravity-void-correlation-development-authorization-consumption-2.0"
            and _HEX64.fullmatch(str(consumption_marker.get("authorization_id"))) is not None
            and consumption_marker.get("uses_consumed") == 1
            and consumption_marker.get("content_sha256") == _self_hash(consumption_marker),
            "authorization not validly consumed",
        )
        _require(len(ledger) == 38053, "one-shot ledger count")
        self.ledger = list(ledger)
        self.next_source = 0
        self.next_cf4 = 0
        self.development_decoded = 0
        self.validation_decoded = 0
        self.confirmation_decoded = 0
        self.vast1_rows = 0
        self.vast2_rows = 0
        self.mask_loads = 0
        self.scores = 0
        self.authorization_consumptions = 1
        self.authorization_id = str(consumption_marker["authorization_id"])
        self.finalized = False

    def source_open(self, name: str, gzip_passes: int) -> None:
        sequence = ["CF4_TABLE4", "VAST_TABLE1", "VAST_TABLE2", "MASK_U8"]
        _require(self.next_source < len(sequence) and name == sequence[self.next_source], "source sequence or repeat")
        if name == "VAST_TABLE1":
            _require(self.next_cf4 == 38053, "VAST table1 opened before CF4 coverage")
        elif name == "VAST_TABLE2":
            _require(self.vast1_rows == 2347, "VAST table2 opened before table1 completion")
        elif name == "MASK_U8":
            _require(self.vast2_rows == 80080, "mask opened before table2 completion")
        expected_gzip = 1 if name in {"CF4_TABLE4", "VAST_TABLE2"} else 0
        _require(gzip_passes == expected_gzip, "source gzip-pass count")
        self.next_source += 1

    def cf4_row(self, entry: Mapping[str, Any], scientific_decoded: bool) -> None:
        _require(self.next_source == 1, "CF4 row outside sole pass")
        _require(self.next_cf4 < 38053 and dict(entry) == self.ledger[self.next_cf4], "CF4 exact coverage/order")
        validate_ledger_entry(entry)
        role = str(entry["role"])
        if role == "development":
            _require(scientific_decoded is True, "development row not decoded once")
            self.development_decoded += 1
        else:
            _require(scientific_decoded is False, "sealed role scientific decode")
            if role == "validation":
                self.validation_decoded += int(scientific_decoded)
            else:
                self.confirmation_decoded += int(scientific_decoded)
        self.next_cf4 += 1

    def record_vast_table1(self, rows: int) -> None:
        _require(self.next_source == 2 and self.next_cf4 == 38053 and self.vast1_rows == 0, "VAST table1 ordering or repeat")
        _require(rows == 2347, "VAST table1 count")
        self.vast1_rows = rows

    def record_vast_table2(self, rows: int) -> None:
        _require(self.next_source == 3 and self.vast1_rows == 2347 and self.vast2_rows == 0, "VAST table2 ordering or repeat")
        _require(rows == 80080, "VAST table2 count")
        self.vast2_rows = rows

    def record_mask(self) -> None:
        _require(self.next_source == 4 and self.vast2_rows == 80080 and self.mask_loads == 0, "mask ordering or repeat")
        self.mask_loads = 1

    def record_development_score(self) -> None:
        _require(self.mask_loads == 1 and self.scores == 0, "score ordering or repeat")
        self.scores = 1

    def finalize(self) -> _OneShotCompletion:
        _require(not self.finalized, "one-shot gate already finalized")
        _require(self.next_source == 4 and self.next_cf4 == 38053, "one-shot source or CF4 coverage incomplete")
        _require(self.development_decoded == 22897, "one-shot development decode count")
        _require(self.validation_decoded == self.confirmation_decoded == 0, "one-shot sealed decode count")
        _require(self.vast1_rows == 2347 and self.vast2_rows == 80080 and self.mask_loads == 1, "one-shot shared source count")
        _require(self.scores == 1 and self.authorization_consumptions == 1, "one-shot score or authorization count")
        counts = {
            "authorization_consumptions": 1,
            "cf4_source_opens": 1,
            "cf4_gzip_passes": 1,
            "cf4_identifier_rows_reverified": 38053,
            "cf4_development_scientific_rows_decoded": 22897,
            "cf4_validation_scientific_rows_decoded": 0,
            "cf4_confirmation_scientific_rows_decoded": 0,
            "vast_table1_source_opens": 1,
            "vast_table1_rows_decoded": 2347,
            "vast_table2_source_opens": 1,
            "vast_table2_gzip_passes": 1,
            "vast_table2_rows_decoded": 80080,
            "mask_source_opens": 1,
            "pantheon_source_opens": 0,
            "development_scores": 1,
        }
        self.finalized = True
        return _OneShotCompletion(_COMPLETION_SECRET, self.authorization_id, counts)


class OneShotExecutionSession:
    """Consumed-authorization owner; exactly one success or one retained failure."""

    def __init__(
        self,
        authorization_payload: bytes,
        contract_receipt: Mapping[str, Any],
        ledger: Sequence[Mapping[str, Any]],
        marker: Mapping[str, Any],
    ) -> None:
        self.authorization_payload = authorization_payload
        self.contract_receipt = dict(contract_receipt)
        self.authorization = validate_authorization_bytes(authorization_payload, contract_receipt)
        self.audited_development_ids = {
            int(entry["identifier"]) for entry in ledger if entry["role"] == "development"
        }
        self.gate = OneShotDevelopmentGate(ledger, marker)
        self.closed = False

    def abort(
        self,
        stage: str,
        reason: str,
        access_counts: Mapping[str, Any],
        identifiers: Sequence[int],
    ) -> str:
        _require(not self.closed, "one-shot session already closed")
        failure = sanitize_failure(
            stage,
            reason,
            access_counts,
            identifiers,
            self.audited_development_ids,
        )
        result = write_fixed_failure_receipt(
            failure,
            self.authorization["authorization_id"],
            self.contract_receipt,
        )
        self.closed = True
        return result

    def succeed(self, artifacts: Mapping[str, bytes]) -> str:
        _require(not self.closed, "one-shot session already closed")
        completion = self.gate.finalize()
        _require(completion.counts == expected_success_access_counts(), "one-shot final count drift")
        try:
            result = promote_fixed_package(
                artifacts,
                self.authorization_payload,
                self.contract_receipt,
                completion,
            )
        except Exception:
            self.closed = True
            failure = sanitize_failure(
                "PROMOTION",
                "PROMOTION_FAILURE",
                completion.counts,
                [],
                self.audited_development_ids,
            )
            write_fixed_failure_receipt(
                failure,
                self.authorization["authorization_id"],
                self.contract_receipt,
            )
            raise
        self.closed = True
        return result


def begin_fixed_one_shot() -> OneShotExecutionSession:
    """The only production entry: read and consume the fixed audit receipt before source open."""
    contract_receipt = check_receipt()
    authorization_path = canonical_file(load_config()["authorization_contract"]["future_path"])
    authorization_payload = authorization_path.read_bytes()
    authorization = validate_authorization_bytes(authorization_payload, contract_receipt)
    marker = consume_fixed_authorization(authorization_payload, contract_receipt)
    _require(marker["authorization_id"] == authorization["authorization_id"], "consumption marker authorization drift")
    ledger = load_identifier_ledger()
    return OneShotExecutionSession(authorization_payload, contract_receipt, ledger, marker)


def conformance_gates(config: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    first = dict(ledger[0])
    stale = dict(first)
    stale["leaf_sha256"] = "0" * 64
    stale_rejected = False
    try:
        validate_ledger_entry(stale)
    except DevelopmentReleaseV2Error:
        stale_rejected = True
    development_metadata = [
        {
            "identifier": entry["identifier"],
            "source_index": entry["source_index"],
            "bucket": entry["bucket"],
            "role": entry["role"],
        }
        for entry in ledger
        if entry["role"] == "development"
    ]
    coverage = validate_exact_development_coverage(development_metadata, ledger)
    permutation = {"observed": 0.0, "permutation_statistics": [0.0] * 10000, "tail_count": 10000, "p_value": 1.0}
    checked = validate_permutation(permutation, {"one_sided_statistic": 0.0})
    profile_template = {
        "best_delta_H": 0.0,
        "best_chi2": 1.0,
        "null_chi2": 1.0,
        "delta_chi2": 0.0,
        "one_sided_statistic": 0.0,
        "tied_delta_H": [0.0],
    }
    profile_models = {
        "PRIMARY_UNION_VOID_PATH": dict(profile_template),
        "C00_FLRW_RADIAL_BULK_SHEAR_NULL": {"delta_chi2": 0.0, "best_delta_H": 0.0, "null_chi2": 1.0},
        "C01_OBSERVER_ENDPOINT_LOCAL_VOID": dict(profile_template),
        "C02_TARGET_ENDPOINT_LOCAL_VOID": dict(profile_template),
        "C03_SINGLE_DOMINANT_VOID": dict(profile_template),
    }
    models = exact_countermodels(profile_models, checked, "1" * 64)
    counts = {key: 0 for key in config["failure_contract"]["access_count_keys"]}
    audited_ids = {int(entry["identifier"]) for entry in ledger if entry["role"] == "development"}
    failure = sanitize_failure("CF4_FRAME", "LEDGER_MISMATCH", counts, [development_metadata[0]["identifier"]], audited_ids, config)
    fixture_marker: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-authorization-consumption-2.0",
        "authorization_id": "f" * 64,
        "authorization_content_sha256": "e" * 64,
        "authorization_raw_sha256": "d" * 64,
        "contract_receipt_content_sha256": "c" * 64,
        "uses_consumed": 1,
        "content_sha256": "",
    }
    fixture_marker["content_sha256"] = _self_hash(fixture_marker)
    gate = OneShotDevelopmentGate(ledger, fixture_marker)
    gate.source_open("CF4_TABLE4", 1)
    for entry in ledger:
        gate.cf4_row(entry, scientific_decoded=entry["role"] == "development")
    gate.source_open("VAST_TABLE1", 0)
    gate.record_vast_table1(2347)
    gate.source_open("VAST_TABLE2", 1)
    gate.record_vast_table2(80080)
    gate.source_open("MASK_U8", 0)
    gate.record_mask()
    gate.record_development_score()
    one_shot = gate.finalize().counts
    return [
        {"check_id": "STALE_ZERO_LEAF_REJECTED", "passed": stale_rejected},
        {"check_id": "EXACT_38053_LEDGER_SEQUENCE_AND_22897_DEVELOPMENT_COVERAGE", "passed": coverage["count"] == 22897},
        {"check_id": "EXACT_10000_PERMUTATION_RECOMPUTATION", "passed": checked["tail_count"] == 10000 and checked["p_value"] == 1.0},
        {"check_id": "C04_DISTINCT_EXACT_COUNTERMODEL_SET", "passed": set(models) == _COUNTERMODEL_KEYS and models["C04_STRATIFIED_EXCHANGEABILITY_NULL"] != models["C00_FLRW_RADIAL_BULK_SHEAR_NULL"]},
        {"check_id": "STRICT_FAILURE_SANITIZER", "passed": failure["reason_code"] == "LEDGER_MISMATCH"},
        {"check_id": "ONE_SHOT_EXACT_COUNTS_AND_HARD_SEALS", "passed": one_shot["development_scores"] == 1 and one_shot["cf4_validation_scientific_rows_decoded"] == 0},
        {"check_id": "NO_REAL_SCIENTIFIC_ACCESS", "passed": all(value == 0 for value in config["current_access_accounting"].values())},
    ]


def build_receipt() -> dict[str, Any]:
    validate_code_pins()
    config = load_config()
    blocked = validate_blocked_v1(config)
    ids = validate_ids_binding(config)
    runtime = validate_runtime_dependencies(config)
    ledger = load_identifier_ledger(config)
    gates = conformance_gates(config, ledger)
    _require(all(gate["passed"] for gate in gates), "v2 source-free conformance failure")
    development_entries = [entry for entry in ledger if entry["role"] == "development"]
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-release-receipt-2.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "blocked_v1": config["blocked_v1"],
        "audit_findings": config["audit_findings"],
        "bindings": {**blocked, **ids, **runtime},
        "runtime_dependencies": config["runtime_dependencies"],
        "ledger_contract": config["ledger_contract"],
        "ledger_counts": {"all": len(ledger), "development": len(development_entries)},
        "ledger_development_root_sha256": _simple_root(
            str(entry["leaf_sha256"]).encode("ascii") for entry in development_entries
        ),
        "countermodel_contract": config["countermodel_contract"],
        "permutation_contract": config["permutation_contract"],
        "output_contract": config["output_contract"],
        "failure_contract": config["failure_contract"],
        "authorization_contract": config["authorization_contract"],
        "one_shot_contract": config["one_shot_contract"],
        "conformance_gates": gates,
        "access_accounting": config["current_access_accounting"],
        "mutation_freeze": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
        },
        "next_gate": config["next_gate"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing contract receipt differs")
        return "EXISTING_IDENTICAL"
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "CREATED"


def write_receipt() -> str:
    return _atomic_no_clobber(OUTPUT_PATH, _pretty(build_receipt()))


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected = build_receipt()
    _require(observed == expected and observed["content_sha256"] == _self_hash(observed), "v2 receipt drift")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID_V2_SOURCE_FREE_NO_SCIENTIFIC_ACCESS")
    else:
        print(check_receipt()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
