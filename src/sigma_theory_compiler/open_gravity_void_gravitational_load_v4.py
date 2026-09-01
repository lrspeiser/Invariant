"""Fail-closed, identifiable appendix to Lane-9 law v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import open_gravity_void_geometry_source_completion_v3 as geometry_v3
from . import open_gravity_void_gravitational_load_v3 as v3

CONFIG_PATH = Path("configs/open_gravity_void_gravitational_load_v4.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_gravitational_load_v4.py")
TEST_PATH = Path("tests/test_open_gravity_void_gravitational_load_v4.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-gravitational-load-v4/receipt.json")
_CONFIG_RAW_SHA256 = "11a6ec2b4e3e03d015091259a75eed490f280162ed3f85ba8ee04a1269736acf"
_CONFIG_CONTENT_SHA256 = "555c108bdb89992f886c50abcb566fc3ac5ee736e1a08ac810cbf3d15b2b0867"
_MODULE_SEMANTIC_SHA256 = "d6f8de533000ac1f20f9bf7e82af422222e8eef92a7eb98c0f7c36acda01059d"
_TEST_RAW_SHA256 = "69d5ec26797e41e97b1ab641bfa343c2965f415ec40df79458ad995a27aac51a"


class VoidLoadV4Error(RuntimeError):
    """Raised when a v4 law invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidLoadV4Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    for name in ("_CONFIG_RAW_SHA256", "_CONFIG_CONTENT_SHA256", "_MODULE_SEMANTIC_SHA256", "_TEST_RAW_SHA256"):
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


def load_config() -> dict[str, Any]:
    validate_code_pins()
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config semantic drift")
    _require(value["status"] == "FROZEN_FAIL_CLOSED_IDENTIFIABLE_LAW_ROWS_UNOPENED", "status drift")
    _require(value["identifiable_law"]["only_law_parameter"] == "delta_H", "law widened")
    _require("H_m*D/c" in value["identifiable_law"]["absorbed_baseline"], "baseline changed")
    _require(value["access_accounting"]["scientific_rows_decoded"] == 0, "rows opened")
    return value


def _finite_vector(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    _require(array.shape == (3,) and bool(np.all(np.isfinite(array))), f"invalid {label}")
    return array


def union_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    checked: list[tuple[float, float]] = []
    for start, stop in intervals:
        _require(math.isfinite(start) and math.isfinite(stop), "nonfinite interval")
        _require(stop >= start, "reversed interval")
        checked.append((start, stop))
    result = v3.union_intervals(checked)
    _require(all(math.isfinite(value) for row in result for value in row), "nonfinite interval output")
    return result


def ray_sphere_intervals(direction: Any, distance: float, spheres: Sequence[tuple[Any, float]]) -> list[tuple[float, float]]:
    ray = _finite_vector(direction, "ray direction")
    _require(math.isfinite(distance) and distance >= 0.0, "invalid path distance")
    checked: list[tuple[np.ndarray, float]] = []
    for center, radius in spheres:
        center_array = _finite_vector(center, "sphere center")
        _require(math.isfinite(radius) and radius > 0.0, "invalid sphere radius")
        checked.append((center_array, radius))
    try:
        return union_intervals(v3.ray_sphere_intervals(ray, distance, checked))
    except v3.VoidLoadV3Error as error:
        raise VoidLoadV4Error(str(error)) from error


def path_partition(direction: Any, distance: float, spheres: Sequence[tuple[Any, float]], observed_intervals: Sequence[tuple[float, float]]) -> dict[str, float]:
    ray = _finite_vector(direction, "ray direction")
    _require(math.isfinite(distance) and distance >= 0.0, "invalid path distance")
    checked_intervals = union_intervals(observed_intervals)
    checked_spheres: list[tuple[np.ndarray, float]] = []
    for center, radius in spheres:
        center_array = _finite_vector(center, "sphere center")
        _require(math.isfinite(radius) and radius > 0.0, "invalid sphere radius")
        checked_spheres.append((center_array, radius))
    try:
        result = v3.path_partition(ray, distance, checked_spheres, checked_intervals)
    except v3.VoidLoadV3Error as error:
        raise VoidLoadV4Error(str(error)) from error
    _require(all(math.isfinite(value) for value in result.values()), "nonfinite path partition")
    return result


def identifiable_void_prediction(delta_h: float, l_void: float, c: float) -> tuple[float, float]:
    _require(all(math.isfinite(value) for value in (delta_h, l_void, c)), "nonfinite prediction input")
    _require(l_void >= 0.0 and c > 0.0, "invalid prediction domain")
    log_redshift = (delta_h / c) * l_void
    velocity = delta_h * l_void
    _require(math.isfinite(log_redshift) and math.isfinite(velocity), "nonfinite prediction")
    return log_redshift, velocity


def observed_log_redshift(v3k: float, c: float) -> float:
    _require(math.isfinite(v3k) and math.isfinite(c) and c > 0.0, "nonfinite observed coordinate")
    try:
        result = v3.observed_log_redshift(v3k, c)
    except v3.VoidLoadV3Error as error:
        raise VoidLoadV4Error(str(error)) from error
    _require(math.isfinite(result), "nonfinite observed log redshift")
    return result


def _bind_packages(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor = config["predecessor"]
    geometry = config["geometry"]
    _require(file_sha256(Path(predecessor["path"])) == predecessor["raw_sha256"], "predecessor raw drift")
    _require(file_sha256(Path(geometry["path"])) == geometry["raw_sha256"], "geometry raw drift")
    predecessor_value = v3.check_package()
    geometry_value = geometry_v3.check_receipt()
    _require(predecessor_value["content_sha256"] == predecessor["content_sha256"], "predecessor content drift")
    _require(geometry_value["content_sha256"] == geometry["content_sha256"], "geometry content drift")
    return predecessor_value, geometry_value


def gates() -> list[dict[str, Any]]:
    failures = 0
    bad_calls = (
        lambda: ray_sphere_intervals([math.nan, 0.0, 0.0], 1.0, []),
        lambda: ray_sphere_intervals([1.0, 0.0, 0.0], math.inf, []),
        lambda: ray_sphere_intervals([1.0, 0.0, 0.0], 1.0, [([0.0, 0.0, 0.0], math.nan)]),
        lambda: union_intervals([(0.0, math.inf)]),
        lambda: identifiable_void_prediction(math.nan, 1.0, 1.0),
        lambda: identifiable_void_prediction(1e308, 1e308, 1.0),
    )
    for call in bad_calls:
        try:
            call()
        except VoidLoadV4Error:
            failures += 1
    logz, velocity = identifiable_void_prediction(0.02, 7.6, 1.0)
    return [
        {"check_id": "NONFINITE_FAIL_CLOSED", "passed": failures == len(bad_calls), "diagnostic": float(failures)},
        {"check_id": "DELTA_H_SIGN_UNITS", "passed": logz > 0.0 and velocity == 0.02 * 7.6, "diagnostic": abs(velocity - 0.02 * 7.6)},
        {"check_id": "CODE_PINS_ENFORCED", "passed": module_semantic_sha256() == _MODULE_SEMANTIC_SHA256 and file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "diagnostic": 0.0},
        {"check_id": "ROWS_UNOPENED", "passed": True, "diagnostic": 0.0},
    ]


def build_receipt() -> dict[str, Any]:
    config = load_config()
    predecessor, geometry = _bind_packages(config)
    checks = gates()
    _require(all(row["passed"] for row in checks), "law repair gate failed")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-gravitational-load-receipt-4.0",
        "package_id": config["package_id"],
        "status": "PASS_FAIL_CLOSED_IDENTIFIABLE_LAW_ROWS_UNOPENED",
        "decision": "DRAFT_FIXED_EXECUTOR_CONTRACT_THEN_INDEPENDENT_REAUDIT",
        "predecessor_content_sha256": predecessor["content_sha256"],
        "geometry_content_sha256": geometry["content_sha256"],
        "identifiable_law": config["identifiable_law"],
        "gates": checks,
        "access_accounting": config["access_accounting"],
        "claim_boundary": config["claim_boundary"],
        "bindings": {"config_raw_sha256": file_sha256(CONFIG_PATH), "config_content_sha256": content_sha256(config), "module_raw_sha256": file_sha256(MODULE_PATH), "module_semantic_sha256": module_semantic_sha256(), "test_raw_sha256": file_sha256(TEST_PATH), "predecessor_raw_sha256": config["predecessor"]["raw_sha256"], "geometry_raw_sha256": config["geometry"]["raw_sha256"]},
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
    return _atomic_no_clobber(OUTPUT_PATH, json.dumps(build_receipt(), sort_keys=True, indent=2).encode() + b"\n")


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(observed == build_receipt() and observed["content_sha256"] == _self_hash(observed), "receipt drift")
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
