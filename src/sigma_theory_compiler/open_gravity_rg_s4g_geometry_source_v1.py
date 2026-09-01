"""Validate public S4G geometry metadata for response-blind 3D source lifts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_rg_s4g_geometry_source_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_rg_s4g_geometry_source_v1.py")
TEST_PATH = Path("tests/test_open_gravity_rg_s4g_geometry_source_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-rg-s4g-geometry-source-v1/receipt.json")

_CONFIG_RAW_SHA256 = "16bbe54ae9410a47fd1cb8109f07cdee578398ca8d147fa3e972ad9f0df9dbb3"
_CONFIG_CONTENT_SHA256 = "00d77bc3eb7df8fa6f21385ae43f935437bffb8d0341714a37f4002d81680b53"
_MODULE_SEMANTIC_SHA256 = "3517ed7e4cb76b1a543b5a0f9b0e6944978d8b75123bb9cf759ad5c3900e79e1"
_TEST_RAW_SHA256 = "46fd1fcb57a66a0e47fe7138078afc09de2e63c00cfde97540bfa406225b399e"
_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f*]{64}("\r?\n)')
_SCHEMA = "invariant-open-gravity-rg-s4g-geometry-source-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-s4g-geometry-source-receipt-1.0"
_OBJECTS = ("NGC2903", "NGC2976", "NGC3198", "NGC3521", "NGC4214")


class GeometrySourceError(RuntimeError):
    """Raised when the frozen public-geometry contract changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometrySourceError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: Path | str) -> Path:
    root = _root().resolve()
    candidate = (root / relative).resolve()
    _require(candidate == root or root in candidate.parents, "path escaped repository")
    return candidate


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    clean = dict(value) if type(value) is dict else value
    if type(clean) is dict:
        clean.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(clean)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", path.read_bytes())
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeometrySourceError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")
    return config


def validate_config(config: dict[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "SOURCE_ONLY_FIVE_OBJECT_PUBLIC_GEOMETRY_VALIDATED",
        "status changed",
    )
    _require([row["object_id"] for row in config["objects"]] == list(_OBJECTS), "objects changed")
    _require(
        [row["orientation_flag"] for row in config["objects"]] == ["ok", "ok", "ok", "u", "u"],
        "orientation flags changed",
    )
    _require(
        config["future_builder_contract"]["fit_geometry_to_response"] is False,
        "response geometry fitting enabled",
    )
    _require(
        config["future_builder_contract"]["retain_all_geometry_failures"] is True,
        "failure retention lost",
    )
    claims = config["claims"]
    _require(claims["public_geometry_source_validated"] is True, "source claim lost")
    _require(claims["five_object_geometry_records_bound"] is True, "record claim lost")
    _require(claims["all_orientations_reliable"] is False, "orientation overclaim")
    _require(
        not any(
            claims[key]
            for key in (
                "three_dimensional_sources_built",
                "response_scored",
                "refracted_gravity_supported",
                "publication_or_discovery_claim",
            )
        ),
        "claim ceiling exceeded",
    )
    access = config["access_state"]
    _require(access["metadata_files_opened"] == 2, "metadata count changed")
    _require(access["metadata_bytes_opened"] == 347186, "metadata bytes changed")
    _require(
        access["response_files_opened"]
        == access["response_rows_opened"]
        == access["scores_computed"]
        == 0,
        "response access enabled",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def _parse_catalog(row: str) -> dict[str, Any]:
    def optional_float(value: str) -> float | None:
        return float(value) if value.strip() else None

    return {
        "object_id": row[0:10].strip(),
        "ra_deg": float(row[13:22]),
        "dec_deg": float(row[23:32]),
        "semi_major_25p5_arcsec": optional_float(row[34:39]),
        "catalog_position_angle_deg": optional_float(row[40:46]),
        "catalog_ellipticity": optional_float(row[47:52]),
        "distance_mpc": optional_float(row[119:126]),
    }


def _parse_pipeline(row: str) -> dict[str, Any]:
    return {
        "object_id": row[0:10].strip(),
        "pipeline4_center_x_pixel": float(row[11:18]),
        "pipeline4_center_y_pixel": float(row[19:26]),
        "outer_position_angle_deg": float(row[27:32]),
        "outer_position_angle_sd_deg": float(row[33:37]),
        "outer_ellipticity": float(row[38:43]),
        "outer_ellipticity_sd": float(row[44:49]),
        "orientation_flag": row[60:62].strip(),
    }


def _load_sources(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    source_evidence: dict[str, Any] = {}
    for source in config["sources"]:
        path = _repo_path(source["path"])
        raw = path.read_bytes()
        _require(
            len(raw) == source["raw_bytes"]
            and hashlib.sha256(raw).hexdigest() == source["raw_sha256"],
            "source bytes changed",
        )
        decoded = gzip.decompress(raw) if source["path"].endswith(".gz") else raw
        _require(
            len(decoded) == source["decoded_bytes"]
            and hashlib.sha256(decoded).hexdigest() == source["decoded_sha256"],
            "decoded source changed",
        )
        rows = decoded.decode("ascii").splitlines()
        _require(len(rows) == source["record_count"], "source record count changed")
        parser = _parse_catalog if source["source_id"] == "S4G_CATALOG_V2" else _parse_pipeline
        parsed[source["source_id"]] = {
            value["object_id"]: value
            for row in rows
            if len(row) >= (126 if parser is _parse_catalog else 84)
            for value in (parser(row),)
        }
        source_evidence[source["source_id"]] = {
            "raw_bytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_bytes": len(decoded),
            "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
            "record_count": len(rows),
        }
    return parsed, source_evidence


def _inclination_deg(ellipticity: float, intrinsic_q: float) -> float:
    observed_q = 1.0 - ellipticity
    if observed_q <= intrinsic_q:
        return 90.0
    cosine2 = (observed_q * observed_q - intrinsic_q * intrinsic_q) / (
        1.0 - intrinsic_q * intrinsic_q
    )
    return math.degrees(math.acos(math.sqrt(max(min(cosine2, 1.0), 0.0))))


def build_receipt(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    parsed, sources = _load_sources(config)
    objects = []
    for expected in config["objects"]:
        object_id = expected["object_id"]
        catalog = parsed["S4G_CATALOG_V2"][object_id]
        pipeline = parsed["S4G_PIPELINE4_OUTER_GEOMETRY"][object_id]
        for key, value in {**catalog, **pipeline}.items():
            if key != "object_id":
                _require(
                    expected[key] == value, f"published geometry mismatch for {object_id}:{key}"
                )
        cells = [
            {
                "intrinsic_axis_ratio_q0": q0,
                "inclination_deg": _inclination_deg(float(expected["outer_ellipticity"]), q0),
                "role": "MODEL_GEOMETRY_SENSITIVITY",
            }
            for q0 in config["future_builder_contract"]["intrinsic_thickness_cells"]
        ]
        objects.append(
            {
                "object_id": object_id,
                "orientation_flag": expected["orientation_flag"],
                "disposition": expected["disposition"],
                "source_record_sha256": content_sha256({"catalog": catalog, "pipeline4": pipeline}),
                "inclination_model_cells": cells,
            }
        )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "PASS_PUBLIC_GEOMETRY_READY_FOR_RESPONSE_BLIND_MODEL_LIFT_WITH_TWO_UNCERTAIN_ORIENTATIONS",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "source_evidence": sources,
        "objects": objects,
        "object_count": len(objects),
        "reliable_orientation_count": sum(row["orientation_flag"] == "ok" for row in objects),
        "uncertain_orientation_count": sum(row["orientation_flag"] != "ok" for row in objects),
        "future_builder_contract": config["future_builder_contract"],
        "claims": config["claims"],
        "access_state": config["access_state"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent output differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt() -> str:
    return _atomic_no_clobber(
        _repo_path(OUTPUT_PATH), canonical_bytes(build_receipt(load_config()))
    )


def check_receipt() -> str:
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    _require(
        path.read_bytes() == canonical_bytes(build_receipt(load_config())),
        "receipt does not rebuild",
    )
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        print(check_receipt())
    else:
        config = load_config()
        print(
            json.dumps(
                {"status": config["status"], "output_exists": _repo_path(OUTPUT_PATH).exists()},
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
