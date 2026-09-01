from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from astropy.io import fits

CONFIG_PATH = Path("configs/open_gravity_rg_things_matched_pair_2d_kinematics_source_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_matched_pair_2d_kinematics_source_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_matched_pair_2d_kinematics_source_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-matched-pair-2d-kinematics-source-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-matched-pair-2d-kinematics-source-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-things-matched-pair-2d-kinematics-source-receipt-1.0"
_CONFIG_RAW_SHA256 = "cfe5fed17ee8d19c1bc2ea21bd852a99d29ca603e53d723a0f140aa3e9bf697c"
_CONFIG_CONTENT_SHA256 = "0c0ac3e37f788f24df25a6ba3a29e63c42324dc8440f84fd2459bf09f02fa322"
_MODULE_SEMANTIC_SHA256 = "09eb69f90b4fd26ad4eb15b44a96224725f55233aff224a2fa26e16332f02463"
_TEST_RAW_SHA256 = "391a1b3d8b490bdb53af79810d3531c12cac6456e1b34c43353336e684c88df4"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")
_BEAM_PATTERN = re.compile(
    r"CLEAN BMAJ=\s*([0-9.+\-Ee]+) BMIN=\s*([0-9.+\-Ee]+) BPA=\s*([0-9.+\-Ee]+)"
)


class KinematicsSourceError(RuntimeError):
    """Raised when the matched-pair source packet fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KinematicsSourceError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _MODULE_PIN_PATTERN.subn(
        b'_MODULE_SEMANTIC_SHA256 = "' + b"0" * 64 + b'"', path.read_bytes()
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path == _ROOT or _ROOT in path.parents, "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KinematicsSourceError(f"invalid {label}") from exc
    _require(type(payload) is dict, f"{label} must be an object")
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "SEALED_REAL_PUBLIC_RESPONSE_BYTES_BENCHMARK_PENDING",
        "status changed",
    )
    selection = config["selection_binding"]
    _require(selection["selected_pair"] == ["NGC2976", "NGC4214"], "pair changed")
    _require(selection["selection_used_things_velocity_pixels"] is False, "selection leak")
    _require(len(config["primary_sources"]) == 2, "source-paper ledger changed")
    files = config["files"]
    _require(len(files) == 4, "file inventory changed")
    _require(sum(int(row["bytes"]) for row in files) == 16974720, "byte ceiling changed")
    _require(
        {(row["object_id"], row["role"]) for row in files}
        == {
            ("NGC2976", "HI_MOM1_NATURAL_VELOCITY_FIELD"),
            ("NGC2976", "HI_MOM2_NATURAL_VELOCITY_DISPERSION"),
            ("NGC4214", "HI_MOM1_NATURAL_VELOCITY_FIELD"),
            ("NGC4214", "HI_MOM2_NATURAL_VELOCITY_DISPERSION"),
        },
        "file roles changed",
    )
    gate = config["future_builder_gate"]
    _require(gate["real_public_source_data_present"] is True, "real source lost")
    _require(gate["primary_measurement_and_method_papers_present"] is True, "paper lost")
    _require(gate["independent_solver_benchmarks_passed"] is False, "benchmark overclaim")
    _require(gate["response_scoring_allowed"] is False, "premature scoring enabled")
    _require(gate["general_3d_claim_allowed"] is False, "3D overclaim")
    _require(len(gate["required_before_pixel_decode"]) == 6, "predecode gate changed")
    access = config["access_accounting"]
    _require(access["head_calls"] == 4 and access["get_calls"] == 4, "network count changed")
    _require(access["network_bytes"] == 16974720, "network bytes changed")
    _require(access["velocity_pixel_values_decoded"] == 0, "velocity values decoded")
    _require(access["dispersion_pixel_values_decoded"] == 0, "dispersion values decoded")
    _require(access["scores_computed"] == 0, "score overclaim")
    claims = config["claim_boundary"]
    _require(claims["exact_public_response_bytes_sealed"] is True, "byte claim lost")
    for key in (
        "velocity_pixels_analyzed",
        "solver_validated",
        "scientific_fit_tested",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim overreach: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output changed")


def _validate_package() -> None:
    if _MODULE_SEMANTIC_SHA256 != "0" * 64:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
    if _TEST_RAW_SHA256 != "0" * 64:
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    if _CONFIG_RAW_SHA256 != "0" * 64:
        _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def _validate_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["selection_binding"]
    for role in ("config", "module", "test", "receipt"):
        path = _repo_path(binding[f"diagnostic_{role}_path"])
        _require(path.is_file(), f"diagnostic {role} missing")
        _require(
            file_sha256(path) == binding[f"diagnostic_{role}_raw_sha256"],
            f"diagnostic {role} changed",
        )
    receipt = _read_json(_repo_path(binding["diagnostic_receipt_path"]), "diagnostic receipt")
    _require(
        receipt["content_sha256"] == binding["diagnostic_receipt_content_sha256"],
        "diagnostic receipt content changed",
    )
    pair = receipt["matched_pair"]
    _require(
        [pair["anchor_object"], pair["source_nearest_neighbor"]] == binding["selected_pair"],
        "matched pair changed",
    )
    _require(pair["opposite_support_direction"] is True, "pair discriminator changed")
    return receipt


def _beam_from_header(header: fits.Header) -> tuple[float, float, float]:
    history = header.get("HISTORY", [])
    rows = [history] if isinstance(history, str) else list(history)
    matches = _BEAM_PATTERN.findall(" ".join(rows))
    _require(len(matches) == 1, "publisher beam history changed")
    return tuple(float(value) for value in matches[0])


def _header_metadata(config: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    path = _repo_path(row["relative_path"])
    _require(path.is_file(), "source file missing")
    _require(path.stat().st_size == int(row["bytes"]), "source byte count changed")
    _require(file_sha256(path) == row["sha256"], "source file hash changed")
    header = fits.getheader(path, 0)
    expected = config["fits_header_contract"]
    for key in ("naxis", "naxis1", "naxis2", "naxis3", "naxis4", "bitpix"):
        _require(int(header[key.upper()]) == int(expected[key]), f"FITS {key} changed")
    for key in ("bunit", "ctype1", "ctype2"):
        _require(str(header[key.upper()]) == expected[key], f"FITS {key} changed")
    for key in ("cdelt1_deg", "cdelt2_deg"):
        header_key = key.split("_")[0].upper()
        _require(float(header[header_key]) == float(expected[key]), f"FITS {key} changed")
    _require(str(header["OBJECT"]) == row["object_id"], "FITS object changed")
    beam = _beam_from_header(header)
    declared = (
        float(row["beam_major_deg"]),
        float(row["beam_minor_deg"]),
        float(row["beam_position_angle_deg"]),
    )
    _require(beam == declared, "publisher beam changed")
    return {
        "object_id": row["object_id"],
        "role": row["role"],
        "relative_path": row["relative_path"],
        "bytes": row["bytes"],
        "sha256": row["sha256"],
        "shape": [
            int(header["NAXIS4"]),
            int(header["NAXIS3"]),
            int(header["NAXIS2"]),
            int(header["NAXIS1"]),
        ],
        "bitpix": int(header["BITPIX"]),
        "bunit": str(header["BUNIT"]),
        "wcs": {
            "ctype1": str(header["CTYPE1"]),
            "ctype2": str(header["CTYPE2"]),
            "crpix1": float(header["CRPIX1"]),
            "crpix2": float(header["CRPIX2"]),
            "crval1": float(header["CRVAL1"]),
            "crval2": float(header["CRVAL2"]),
            "cdelt1": float(header["CDELT1"]),
            "cdelt2": float(header["CDELT2"]),
        },
        "beam": {
            "major_deg": beam[0],
            "minor_deg": beam[1],
            "position_angle_deg": beam[2],
        },
        "data_values_read": 0,
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    diagnostic = _validate_selection(config)
    files = [_header_metadata(config, row) for row in config["files"]]
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_EXACT_THINGS_MATCHED_PAIR_BYTES_SOURCE_READY_SOLVER_BENCHMARK_PENDING",
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "diagnostic_receipt_content_sha256": diagnostic["content_sha256"],
        "primary_sources": config["primary_sources"],
        "files": files,
        "file_count": len(files),
        "byte_count": sum(int(row["bytes"]) for row in files),
        "future_builder_gate": config["future_builder_gate"],
        "access_accounting": config["access_accounting"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    _require(dict(payload) == build_receipt(config), "receipt differs from rebuild")


def _output_path() -> Path:
    path = _repo_path(OUTPUT_PATH)
    _require(path == (_ROOT / OUTPUT_PATH).resolve(), "output path changed")
    return path


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing nonidentical overwrite")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent nonidentical receipt")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt() -> str:
    config = load_config()
    return _atomic_no_clobber(_output_path(), canonical_bytes(build_receipt(config)) + b"\n")


def validate_receipt() -> None:
    config = load_config()
    path = _output_path()
    _require(path.is_file(), "receipt missing")
    validate_receipt_payload(config, _read_json(path, "receipt"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"), nargs="?", default="check")
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt(load_config())
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "file_count": receipt["file_count"],
                    "byte_count": receipt["byte_count"],
                    "velocity_pixel_values_decoded": receipt["access_accounting"][
                        "velocity_pixel_values_decoded"
                    ],
                    "solver_benchmarks_passed": receipt["future_builder_gate"][
                        "independent_solver_benchmarks_passed"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
