"""Fail-closed, distance-typed appendix to Lane-9 geometry v2."""

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

from . import open_gravity_void_geometry_source_completion_v2 as v2

CONFIG_PATH = Path("configs/open_gravity_void_geometry_source_completion_v3.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_geometry_source_completion_v3.py")
TEST_PATH = Path("tests/test_open_gravity_void_geometry_source_completion_v3.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-geometry-source-completion-v3/receipt.json")
_CONFIG_RAW_SHA256 = "05a8ec88a9d37db28f3a9e9d991108e5b3388dac2cf55e1017f6b7d431e45c36"
_CONFIG_CONTENT_SHA256 = "8a389d86254ae1d1fd18f462d540daf937b10b514a8ba7d1f47b01d46f34ace4"
_MODULE_SEMANTIC_SHA256 = "8f0d20a1788322fdb26cce3af17c7e7669aa4f55d894ce9eadabb8249e2e8128"
_TEST_RAW_SHA256 = "e68bac67a32aae5e8aee530a66845f0ad3600ef54595966d910711f21bd28771"


class VoidGeometryV3Error(RuntimeError):
    """Raised when a v3 geometry invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidGeometryV3Error(message)


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
    _require(file_sha256(MODULE_PATH) != "", "module missing")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test pin drift")


def load_config() -> dict[str, Any]:
    validate_code_pins()
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config semantic drift")
    _require(value["status"] == "FROZEN_DISTANCE_TYPED_FAIL_CLOSED_ROWS_UNOPENED", "status drift")
    _require(value["access_accounting"]["scientific_rows_decoded"] == 0, "rows opened")
    return value


def validate_cf4_distance(dmzp_mag: float, dist_mpc: float) -> dict[str, float]:
    _require(math.isfinite(dmzp_mag) and math.isfinite(dist_mpc), "nonfinite CF4 distance")
    _require(dist_mpc > 0.0, "nonpositive CF4 distance")
    formula = 10.0 ** ((dmzp_mag - 25.0) / 5.0)
    _require(math.isfinite(formula), "distance formula overflow")
    tolerance = 0.05 + formula * (10.0 ** (0.0005 / 5.0) - 1.0)
    difference = abs(dist_mpc - formula)
    _require(difference <= tolerance + 1e-12, "DMzp/Dist catalog-rounding mismatch")
    return {"DMzp_mag": dmzp_mag, "Dist_Mpc": dist_mpc, "formula_Mpc": formula, "tolerance_Mpc": tolerance, "difference_Mpc": difference}


def mask_index(ra_deg: float, dec_deg: float) -> tuple[int, int]:
    _require(math.isfinite(ra_deg) and math.isfinite(dec_deg), "nonfinite coordinate")
    try:
        return v2.mask_index(ra_deg, dec_deg)
    except v2.VoidGeometryV2Error as error:
        raise VoidGeometryV3Error(str(error)) from error


def radec_to_xyz(ra_deg: float, dec_deg: float, radius: float) -> np.ndarray:
    _require(all(math.isfinite(value) for value in (ra_deg, dec_deg, radius)), "nonfinite coordinate or radius")
    _require(radius >= 0.0, "negative radius")
    try:
        result = v2.radec_to_xyz(ra_deg, dec_deg, radius)
    except v2.VoidGeometryV2Error as error:
        raise VoidGeometryV3Error(str(error)) from error
    _require(bool(np.all(np.isfinite(result))), "nonfinite transformed coordinate")
    return result


def luminosity_to_comoving_hinv(dist_mpc: float) -> tuple[float, float]:
    _require(math.isfinite(dist_mpc) and dist_mpc >= 0.0, "invalid luminosity distance")
    config_v2 = v2.load_config()
    try:
        result = v2.luminosity_to_comoving_hinv(dist_mpc, config_v2)
    except v2.VoidGeometryV2Error as error:
        raise VoidGeometryV3Error(str(error)) from error
    _require(all(math.isfinite(value) for value in result), "nonfinite distance inversion")
    return result


def _bind_predecessor(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = config["predecessor"]
    path = Path(predecessor["path"])
    _require(path.is_file() and file_sha256(path) == predecessor["raw_sha256"], "predecessor raw drift")
    observed = v2.check_package()
    _require(observed["content_sha256"] == predecessor["content_sha256"], "predecessor content drift")
    return observed


def gates() -> list[dict[str, Any]]:
    sample_mu = 35.0
    sample_dist = round(10.0 ** ((sample_mu - 25.0) / 5.0), 1)
    row = validate_cf4_distance(sample_mu, sample_dist)
    rejected = 0
    for call in (
        lambda: validate_cf4_distance(math.nan, 10.0),
        lambda: validate_cf4_distance(30.0, math.inf),
        lambda: radec_to_xyz(math.nan, 0.0, 1.0),
        lambda: radec_to_xyz(0.0, 0.0, math.inf),
        lambda: luminosity_to_comoving_hinv(math.nan),
    ):
        try:
            call()
        except VoidGeometryV3Error:
            rejected += 1
    return [
        {"check_id": "CF4_DISTANCE_ROUNDING_IDENTITY", "passed": row["difference_Mpc"] <= row["tolerance_Mpc"], "diagnostic": row["difference_Mpc"]},
        {"check_id": "NONFINITE_FAIL_CLOSED", "passed": rejected == 5, "diagnostic": float(rejected)},
        {"check_id": "CODE_PINS_ENFORCED", "passed": module_semantic_sha256() == _MODULE_SEMANTIC_SHA256 and file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "diagnostic": 0.0},
        {"check_id": "ROWS_UNOPENED", "passed": True, "diagnostic": 0.0},
    ]


def build_receipt() -> dict[str, Any]:
    config = load_config()
    predecessor = _bind_predecessor(config)
    checks = gates()
    _require(all(row["passed"] for row in checks), "repair gate failed")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-geometry-source-completion-receipt-3.0",
        "package_id": config["package_id"],
        "status": "PASS_DISTANCE_TYPED_FAIL_CLOSED_ROWS_UNOPENED",
        "decision": "GEOMETRY_READY_FOR_INDEPENDENT_REAUDIT_BEFORE_EXECUTOR",
        "predecessor_content_sha256": predecessor["content_sha256"],
        "cf4_distance_columns": config["cf4_distance_columns"],
        "canonical_mask": config["canonical_mask"],
        "gates": checks,
        "access_accounting": config["access_accounting"],
        "claim_boundary": config["claim_boundary"],
        "bindings": {"config_raw_sha256": file_sha256(CONFIG_PATH), "config_content_sha256": content_sha256(config), "module_raw_sha256": file_sha256(MODULE_PATH), "module_semantic_sha256": module_semantic_sha256(), "test_raw_sha256": file_sha256(TEST_PATH), "predecessor_raw_sha256": config["predecessor"]["raw_sha256"]},
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
