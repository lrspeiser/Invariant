"""Validate and seal the source-only three-galaxy model-lifted 3-D pilot inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

CONFIG_PATH = Path(
    "configs/open_gravity_phangs_things_model_lifted_3d_pilot_source_acquisition_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_phangs_things_model_lifted_3d_pilot_source_acquisition_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_phangs_things_model_lifted_3d_pilot_source_acquisition_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-phangs-things-model-lifted-3d-pilot-source-acquisition-v1/receipt.json"
)

_CONFIG_RAW_SHA256 = "7d9c9870f7273e4c817e18b6942bfca6eae3dd604e5c7dfa2912b8806c19a5e7"
_CONFIG_CONTENT_SHA256 = "f51512bbd8406f1e1b3914ebbcefe7f2e23aaff4311aa6ba3d15df14d4c023ea"
_MODULE_SEMANTIC_SHA256 = "f32aac91c56c7e1a15ddc0a0ca9d2c2a13c326975e5a61d8c4d2db4fed2fb493"
_TEST_RAW_SHA256 = "f9336855964f9b7c0105f3c22b72b1a74545a48b3ce48e706032b52c6dab7f96"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')
_BEAM_PATTERN = re.compile(
    r"CLEAN BMAJ=\s*([0-9.+\-Ee]+) BMIN=\s*([0-9.+\-Ee]+) BPA=\s*([0-9.+\-Ee]+)"
)
_SCHEMA = "invariant-open-gravity-phangs-things-model-lifted-3d-pilot-source-acquisition-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-phangs-things-model-lifted-3d-pilot-source-acquisition-receipt-1.0"
)


class SourceAcquisitionError(RuntimeError):
    """Raised when the frozen source acquisition contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAcquisitionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: Path | str) -> Path:
    root = _root().resolve()
    candidate = (root / relative).resolve()
    _require(candidate == root or root in candidate.parents, "path escaped repository")
    return candidate


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized, count = _MODULE_PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", raw)
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAcquisitionError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def load_config() -> dict[str, Any]:
    config_path = _repo_path(CONFIG_PATH)
    _require(file_sha256(config_path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(config_path, "config")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "SOURCE_ACQUIRED_AND_SCHEMA_VALIDATED_DEVELOPMENT_ONLY",
        "status changed",
    )
    inventory = config["inventory_contract"]
    _require(inventory["objects"] == ["NGC2903", "NGC3351", "NGC3627"], "objects changed")
    _require(inventory["file_count"] == 21, "file count changed")
    _require(inventory["scientific_payload_bytes"] == 74_030_400, "payload bytes changed")
    _require(inventory["image_pixels_inspected"] == 18_026_980, "pixel count changed")
    _require(inventory["finite_image_pixels"] == 16_452_117, "finite pixel count changed")
    transport = config["transport_accounting"]
    _require(transport["successful_source_gets"] == 21, "successful GET count changed")
    _require(transport["failed_redirect_gets"] == 1, "failed GET hidden")
    _require(transport["failed_redirect_body_bytes"] == 252, "failed GET bytes hidden")
    _require(transport["total_get_attempts"] == 22, "total GET attempts changed")
    _require(transport["total_network_body_bytes"] == 74_030_652, "network bytes changed")
    _require(transport["redirects_followed"] == 0, "redirect followed")
    _require(transport["retries"] == 0, "retry claimed")
    boundary = config["scientific_boundary"]
    _require(boundary["source_images_opened"] is True, "source access hidden")
    _require(
        boundary["velocity_or_rotation_response_opened_by_this_package"] is False, "response opened"
    )
    _require(boundary["response_rows_opened"] == 0, "response rows changed")
    _require(boundary["scores_computed"] == 0 and boundary["models_fit"] == 0, "science executed")
    _require(boundary["development_only"] is True, "development boundary lost")
    claims = config["claims"]
    _require(claims["exact_source_bytes_and_fits_schemas_validated"] is True, "source claim lost")
    _require(
        not any(
            value
            for key, value in claims.items()
            if key != "exact_source_bytes_and_fits_schemas_validated"
        ),
        "claim ceiling exceeded",
    )


def _validate_package_files() -> None:
    _require(
        module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256, "module changed"
    )
    _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def _load_predecessor(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor = config["predecessor"]
    for role in ("config", "module", "test", "receipt"):
        path = _repo_path(predecessor[f"{role}_path"])
        _require(
            file_sha256(path) == predecessor[f"{role}_raw_sha256"], f"predecessor {role} changed"
        )
    prior_config = _read_json(_repo_path(predecessor["config_path"]), "predecessor config")
    prior_receipt = _read_json(_repo_path(predecessor["receipt_path"]), "predecessor receipt")
    _require(
        prior_receipt["content_sha256"] == predecessor["receipt_content_sha256"],
        "predecessor receipt content changed",
    )
    return prior_config, prior_receipt


def _things_beam(header: fits.Header) -> list[float]:
    history = header.get("HISTORY", [])
    if isinstance(history, str):
        history = [history]
    matches = [match for line in history if (match := _BEAM_PATTERN.search(str(line)))]
    _require(len(matches) == 1, "THINGS must contain exactly one AIPS CLEAN beam")
    beam = [float(value) for value in matches[0].groups()]
    _require(beam[0] > 0.0 and beam[1] > 0.0 and math.isfinite(beam[2]), "invalid THINGS beam")
    _require("BMAJ" not in header and "BMIN" not in header, "THINGS beam storage changed")
    return beam


def _source_filename(row: dict[str, Any]) -> str:
    return f"{row['object_id']}__{row['survey']}__{row['role']}.fits"


def build_inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    _validate_package_files()
    prior_config, _ = _load_predecessor(config)
    root = _repo_path(config["private_source_root"])
    _require(root.is_dir(), "private source root missing")
    forbidden = tuple(config["fits_contract"]["response_bearing_products_forbidden"])
    rows: list[dict[str, Any]] = []
    for source in prior_config["source_files"]:
        name = _source_filename(source)
        _require(
            not any(token in name.upper() for token in forbidden), "response-bearing file admitted"
        )
        path = (root / name).resolve()
        _require(path.parent == root, "source path escaped private root")
        _require(path.is_file(), f"source file missing: {name}")
        _require(path.stat().st_size == source["bytes"], f"source byte count changed: {name}")
        with fits.open(path, memmap=True, do_not_scale_image_data=False) as hdus:
            _require(len(hdus) == 1, f"unexpected HDU inventory: {name}")
            _require(hdus[0].data is not None, f"image missing: {name}")
            header = hdus[0].header
            array = np.asarray(hdus[0].data)
            finite_pixels = int(np.isfinite(array).sum())
            beam_source: str | None = None
            beam: list[float] | None = None
            if source["survey"] == "S4G_P5":
                _require(array.ndim == 2, f"S4G dimensionality changed: {name}")
                _require(
                    header.get("BUNIT") == config["fits_contract"]["s4g_bunit"],
                    f"S4G unit changed: {name}",
                )
                _require(
                    str(header.get("CTYPE1", "")).startswith("RA---TAN"), f"S4G WCS changed: {name}"
                )
                _require(
                    str(header.get("CTYPE2", "")).startswith("DEC--TAN"), f"S4G WCS changed: {name}"
                )
            elif source["survey"] == "THINGS":
                _require(list(array.shape) == [1, 1, 1024, 1024], f"THINGS shape changed: {name}")
                _require(
                    header.get("BUNIT") == config["fits_contract"]["things_bunit"],
                    f"THINGS unit changed: {name}",
                )
                _require(
                    header.get("RESTFREQ") == config["fits_contract"]["things_rest_hz"],
                    f"THINGS frequency changed: {name}",
                )
                beam = _things_beam(header)
                beam_source = "AIPS_HISTORY"
            else:
                _require(source["survey"] == "PHANGS_ALMA", f"unknown survey: {name}")
                _require(
                    array.ndim == 4 and list(array.shape[:2]) == [1, 1],
                    f"PHANGS shape changed: {name}",
                )
                _require(
                    header.get("BUNIT") == config["fits_contract"]["phangs_bunit"],
                    f"PHANGS unit changed: {name}",
                )
                _require(
                    header.get("RESTFRQ") == config["fits_contract"]["phangs_rest_hz"],
                    f"PHANGS frequency changed: {name}",
                )
                beam = [float(header["BMAJ"]), float(header["BMIN"]), float(header.get("BPA", 0.0))]
                _require(beam[0] > 0.0 and beam[1] > 0.0, f"invalid PHANGS beam: {name}")
                beam_source = "HEADER"
            rows.append(
                {
                    "id": f"{source['object_id']}:{source['survey']}:{source['role']}",
                    "relative_private_path": path.relative_to(_root()).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "shape": list(array.shape),
                    "bunit": str(header.get("BUNIT", "")),
                    "ctype1": str(header.get("CTYPE1", "")),
                    "ctype2": str(header.get("CTYPE2", "")),
                    "rest_hz": header.get("RESTFRQ", header.get("RESTFREQ")),
                    "beam_source": beam_source,
                    "beam_deg": beam,
                    "finite_pixels": finite_pixels,
                    "total_pixels": int(array.size),
                }
            )
    inventory = config["inventory_contract"]
    _require(len(rows) == inventory["file_count"], "inventory file count changed")
    _require(
        sum(row["bytes"] for row in rows) == inventory["scientific_payload_bytes"],
        "inventory bytes changed",
    )
    _require(
        sum(row["total_pixels"] for row in rows) == inventory["image_pixels_inspected"],
        "pixel count changed",
    )
    _require(
        sum(row["finite_pixels"] for row in rows) == inventory["finite_image_pixels"],
        "finite pixel count changed",
    )
    _require(
        content_sha256(rows) == inventory["ordered_inventory_root_sha256"], "inventory root changed"
    )
    sha_root = hashlib.sha256("\n".join(row["sha256"] for row in rows).encode("ascii")).hexdigest()
    _require(sha_root == inventory["ordered_file_sha_root_sha256"], "file SHA root changed")
    return rows


def build_receipt(config: dict[str, Any]) -> dict[str, Any]:
    rows = build_inventory(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "SOURCE_READY_FOR_MODEL_LIFTED_3D_DERIVATION_DEVELOPMENT_ONLY",
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "predecessor": config["predecessor"],
        "inventory": rows,
        "inventory_summary": config["inventory_contract"],
        "transport_accounting": config["transport_accounting"],
        "fits_contract": config["fits_contract"],
        "scientific_boundary": config["scientific_boundary"],
        "claims": config["claims"],
        "access_state": {
            "source_payload_files_downloaded": 21,
            "source_payload_bytes_downloaded": 74_030_400,
            "scientific_image_pixels_read": 18_026_980,
            "response_rows_read": 0,
            "scores_computed": 0,
            "models_fit": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(receipt: dict[str, Any], config: dict[str, Any]) -> None:
    _require(receipt == build_receipt(config), "receipt does not exactly rebuild")


def _receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    output = _repo_path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _receipt_bytes(receipt)
    if output.exists():
        _require(output.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        if os.name != "nt":
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except FileExistsError as exc:
        raise SourceAcquisitionError("receipt appeared concurrently") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "CREATED"


def check_receipt() -> str:
    config = load_config()
    output = _repo_path(OUTPUT_PATH)
    _require(output.is_file(), "receipt missing")
    validate_receipt(_read_json(output, "receipt"), config)
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
        rows = build_inventory(config)
        print(
            json.dumps(
                {
                    "status": config["status"],
                    "files": len(rows),
                    "source_bytes": sum(row["bytes"] for row in rows),
                    "pixels": sum(row["total_pixels"] for row in rows),
                    "response_rows": 0,
                    "scores": 0,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
