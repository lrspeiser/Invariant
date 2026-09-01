"""Canonical, non-executing VAST mask and distance/coordinate contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickletools
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_void_geometry_source_completion_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_geometry_source_completion_v2.py")
TEST_PATH = Path("tests/test_open_gravity_void_geometry_source_completion_v2.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-geometry-source-completion-v2/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
_CONFIG_RAW_SHA256 = "180121d1253c8e14138c5c02423490e8f4f6cd2319e8ee33e40bd4208a916eef"
_CONFIG_CONTENT_SHA256 = "307a0976a9913e54b40d664643f19b1b124f53fcae5c0aa1689a4e3c1823f573"
_MODULE_SEMANTIC_SHA256 = "40c36acebaa25350975c59316936ab42ad6661c6d67ae4309db432e46bb3ba8c"
_TEST_RAW_SHA256 = "55805783f23dac804731411ffa7966e5598d8faf862fab20086a27016c61c4c4"


class VoidGeometryV2Error(RuntimeError):
    """Raised when an immutable geometry or source invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidGeometryV2Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    for name in (
        "_CONFIG_RAW_SHA256",
        "_CONFIG_CONTENT_SHA256",
        "_MODULE_SEMANTIC_SHA256",
        "_TEST_RAW_SHA256",
    ):
        marker = f'{name} = "'
        start = text.index(marker) + len(marker)
        end = text.index('"', start)
        text = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(text.encode()).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(type(value) is dict, "config must be object")
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config semantic drift")
    _require(value["schema"].endswith("2.0"), "schema drift")
    _require(value["status"] == "FROZEN_CORRELATION_ONLY_GEOMETRY_ROWS_UNOPENED", "status drift")
    _require(value["scope"]["scientific_rows_decoded"] == 0, "science rows opened")
    _require("mechanism inference" in value["scope"]["forbidden"], "scope widened")
    return value


def _validate_receipt_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(row["path"])
    _require(path.is_file() and file_sha256(path) == row["raw_sha256"], "receipt raw drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value["content_sha256"] == row["content_sha256"], "receipt content drift")
    _require(_self_hash(value) == row["content_sha256"], "receipt self-hash invalid")
    return value


def _trace_rows(data: bytes) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for opcode, argument, position in pickletools.genops(data):
        normalized: Any = argument
        if isinstance(argument, bytes):
            normalized = {"bytes": len(argument), "sha256": hashlib.sha256(argument).hexdigest()}
        rows.append([position, opcode.name, normalized])
    return rows


def canonical_mask(config: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """Extract two exact payloads from one exact pickle trace without executing it."""
    mask = config["mask"]
    path = Path(mask["local_path"])
    data = path.read_bytes()
    _require(len(data) == mask["bytes"] and file_sha256(path) == mask["sha256"], "mask drift")
    rows = _trace_rows(data)
    _require(len(rows) == 108, "opcode count drift")
    _require(hashlib.sha256(_canonical(rows)).hexdigest() == mask["opcode_trace_sha256"], "trace drift")
    forbidden = {"GLOBAL", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX"}
    _require(not ({row[1] for row in rows} & forbidden), "executable pickle opcode present")
    by_position = {position: (opcode, argument) for opcode, argument, position in pickletools.genops(data)}
    opcode, boolean_payload = by_position[143]
    _require(opcode.name == "BINBYTES" and isinstance(boolean_payload, bytes), "mask payload moved")
    _require(hashlib.sha256(boolean_payload).hexdigest() == mask["boolean_payload_sha256"], "mask payload drift")
    _require(len(boolean_payload) == 360 * 180 and set(boolean_payload) <= {0, 1}, "mask payload invalid")
    opcode, float_payload = by_position[65014]
    _require(opcode.name == "SHORT_BINBYTES" and isinstance(float_payload, bytes), "limits moved")
    _require(hashlib.sha256(float_payload).hexdigest() == mask["float_payload_sha256"], "limits drift")
    limits = list(struct.unpack("<2f", float_payload))
    _require(by_position[87][1] == 360 and by_position[90][1] == 180, "shape operands moved")
    _require(by_position[64952][1] == 1, "resolution operand moved")
    _require(limits == mask["dist_limits_h_inverse_mpc"], "distance limits drift")
    metadata = {
        "shape": [360, 180],
        "layout": "C-order uint8; index i*180+j",
        "true_pixels": sum(boolean_payload),
        "resolution_per_degree": 1,
        "dist_limits_h_inverse_mpc": limits,
        "source_pickle_sha256": mask["sha256"],
        "payload_sha256": hashlib.sha256(boolean_payload).hexdigest(),
        "pickle_executed": False,
    }
    _require(metadata["true_pixels"] == mask["true_pixels"], "mask population drift")
    return boolean_payload, metadata


def mask_index(ra_deg: float, dec_deg: float, resolution: int = 1) -> tuple[int, int]:
    _require(math.isfinite(ra_deg) and math.isfinite(dec_deg), "nonfinite angle")
    _require(-90.0 <= dec_deg < 90.0 and resolution == 1, "angle outside frozen mask")
    ra = ra_deg % 360.0
    return math.floor(ra * resolution), math.floor((dec_deg + 90.0) * resolution)


def mask_contains(mask_u8: bytes, ra_deg: float, dec_deg: float) -> bool:
    i, j = mask_index(ra_deg, dec_deg)
    return bool(mask_u8[i * 180 + j])


def radec_to_xyz(ra_deg: float, dec_deg: float, radius: float) -> np.ndarray:
    _require(radius >= 0.0 and math.isfinite(radius), "invalid radius")
    ra = math.radians(ra_deg % 360.0)
    dec = math.radians(dec_deg)
    _require(-math.pi / 2 <= dec <= math.pi / 2, "invalid declination")
    return radius * np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])


def _comoving_mpc(z: float, *, h0: float, omega_m: float, c_km_s: float) -> float:
    _require(z >= 0.0, "negative redshift")
    nodes, weights = np.polynomial.legendre.leggauss(64)
    sample = 0.5 * z * (nodes + 1.0)
    expansion = np.sqrt(omega_m * (1.0 + sample) ** 3 + (1.0 - omega_m))
    return (c_km_s / h0) * 0.5 * z * float(np.sum(weights / expansion))


def luminosity_to_comoving_hinv(distance_luminosity_mpc: float, config: Mapping[str, Any]) -> tuple[float, float]:
    _require(distance_luminosity_mpc >= 0.0, "negative luminosity distance")
    cosmology = config["distance_contract"]["cosmology"]
    h0 = float(cosmology["H0_km_s_Mpc"])
    omega_m = float(cosmology["Omega_m"])
    c_km_s = float(cosmology["c_km_s"])
    if distance_luminosity_mpc == 0.0:
        return 0.0, 0.0
    low, high = 0.0, 0.2
    def dl(redshift: float) -> float:
        return (1.0 + redshift) * _comoving_mpc(redshift, h0=h0, omega_m=omega_m, c_km_s=c_km_s)
    while dl(high) < distance_luminosity_mpc:
        high *= 2.0
        _require(high <= 4.0, "distance outside frozen inversion")
    for _ in range(80):
        mid = 0.5 * (low + high)
        if dl(mid) < distance_luminosity_mpc:
            low = mid
        else:
            high = mid
    z = 0.5 * (low + high)
    dc = _comoving_mpc(z, h0=h0, omega_m=omega_m, c_km_s=c_km_s)
    return z, (h0 / 100.0) * dc


def build_receipt() -> tuple[dict[str, Any], dict[Path, bytes]]:
    config = load_config()
    blocked = _validate_receipt_binding(config["blocked_predecessor"])
    source = _validate_receipt_binding(config["source_packet"])
    _require(source["source_bundle_root_sha256"] == config["source_packet"]["source_bundle_root_sha256"], "source root drift")
    mask_u8, metadata = canonical_mask(config)
    payloads = {ARTIFACT_DIR / "mask-u8.bin": mask_u8, ARTIFACT_DIR / "mask-metadata.json": _pretty(metadata)}
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-geometry-source-completion-receipt-2.0",
        "package_id": config["package_id"],
        "status": "PASS_CORRELATION_ONLY_GEOMETRY_CANONICAL_ROWS_UNOPENED",
        "decision": "READY_TO_FREEZE_CORRELATION_ONLY_EXECUTOR_NOT_MECHANISM_CLAIM",
        "counterevidence_receipt_content_sha256": blocked["content_sha256"],
        "source_bundle_root_sha256": source["source_bundle_root_sha256"],
        "mask_metadata": metadata,
        "coordinate_contract": config["coordinate_contract"],
        "distance_contract": config["distance_contract"],
        "version_boundary": config["version_boundary"],
        "scope": config["scope"],
        "bindings": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
            "blocked_predecessor_raw_sha256": config["blocked_predecessor"]["raw_sha256"],
        },
        "artifact_index": [{"path": path.as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} for path, payload in sorted(payloads.items(), key=lambda row: row[0].as_posix())],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt, payloads


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"existing output differs: {path}")
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


def write_package() -> str:
    receipt, payloads = build_receipt()
    for path, payload in payloads.items():
        _atomic_no_clobber(path, payload)
    return _atomic_no_clobber(OUTPUT_PATH, _pretty(receipt))


def check_package() -> dict[str, Any]:
    _require(OUTPUT_PATH.is_file(), "receipt missing")
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected, payloads = build_receipt()
    _require(observed == expected and observed["content_sha256"] == _self_hash(observed), "receipt drift")
    for path, payload in payloads.items():
        _require(path.is_file() and path.read_bytes() == payload, f"artifact drift: {path}")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_package())
    else:
        receipt = check_package()
        print("VALID" if args.command == "check" else receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
