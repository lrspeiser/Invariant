"""Validate and seal the source-only 12-galaxy outer-disk expansion inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from sigma_theory_compiler import (
    open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_preflight_v1 as preflight,
)

CONFIG_PATH = Path(
    "configs/open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_source_acquisition_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_source_acquisition_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_refracted_gravity_things_heracles_sparc_3d_expansion_source_acquisition_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-refracted-gravity-things-heracles-sparc-3d-expansion-source-acquisition-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "a55b4ae9e262783f0622b59411fe23aaae4ec7ffa56184cc66dca257bde457c1"
_CONFIG_CONTENT_SHA256 = "24d41a4eaab28ea1e765b1b0ced813889fda5c7cfd26c6039f18fb6ae1fd9785"
_MODULE_SEMANTIC_SHA256 = "7b6ed9726b39bfc1de5dfe2dbf16c317329cc482260c7bb249fe40c217a0b9d1"
_TEST_RAW_SHA256 = "c80e8f8917b7ee07a96548f0af36ec057c8a5aa4728ade6f2c44e7db3df2f771"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')
_BEAM_PATTERN = re.compile(
    r"CLEAN BMAJ=\s*([0-9.+\-Ee]+) BMIN=\s*([0-9.+\-Ee]+) BPA=\s*([0-9.+\-Ee]+)"
)
_SCHEMA = (
    "invariant-open-gravity-refracted-gravity-things-heracles-sparc-3d-expansion-"
    "source-acquisition-1.0"
)
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-refracted-gravity-things-heracles-sparc-3d-expansion-"
    "source-acquisition-receipt-1.0"
)


class ExpansionSourceError(RuntimeError):
    """Raised when the source-only acquisition contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpansionSourceError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized, count = _MODULE_PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", raw)
    _require(count == 1, "module semantic pin structure changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path.is_relative_to(_ROOT), "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpansionSourceError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _source_filename(row: Mapping[str, Any]) -> str:
    suffix = ".fits.gz" if row["url"].endswith(".fits.gz") else ".fits"
    return f"{row['object_id']}__{row['survey']}__{row['role']}{suffix}"


def _beam(header: fits.Header) -> tuple[str | None, list[float] | None]:
    if "BMAJ" in header and "BMIN" in header:
        beam = [float(header["BMAJ"]), float(header["BMIN"]), float(header.get("BPA", 0.0))]
        _require(beam[0] > 0.0 and beam[1] > 0.0, "invalid header beam")
        return "HEADER", beam
    history = header.get("HISTORY", [])
    if isinstance(history, str):
        history = [history]
    matches = [match for line in history if (match := _BEAM_PATTERN.search(str(line)))]
    if not matches:
        return None, None
    _require(len(matches) == 1, "ambiguous AIPS beam")
    beam = [float(value) for value in matches[0].groups()]
    _require(beam[0] > 0.0 and beam[1] > 0.0 and math.isfinite(beam[2]), "invalid AIPS beam")
    return "AIPS_HISTORY", beam


def _wcs_signature(header: fits.Header) -> tuple[Any, ...]:
    keys = (
        "NAXIS1",
        "NAXIS2",
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CDELT1",
        "CDELT2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
    )
    return tuple(header.get(key) for key in keys)


def validate_config(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "predecessor",
        "private_source_root",
        "inventory_contract",
        "transport_accounting",
        "fits_contract",
        "builder_admission",
        "scientific_boundary",
        "claims",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected_keys, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"]
        == "open-gravity-refracted-gravity-things-heracles-sparc-3d-expansion-source-acquisition-v1",
        "package ID changed",
    )
    _require(
        config["status"]
        == "SOURCE_ONLY_12_OBJECT_77_FILE_ACQUISITION_AND_FITS_SCHEMA_VALIDATION_PASS",
        "status changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    inventory = config["inventory_contract"]
    _require(inventory["objects"] == list(preflight._OBJECTS), "objects changed")
    _require(inventory["file_count"] == 77, "file count changed")
    _require(inventory["compressed_source_bytes"] == 351_461_418, "bytes changed")
    _require(inventory["image_pixels_inspected"] == 87_444_699, "pixel count changed")
    _require(inventory["finite_image_pixels"] == 60_869_171, "finite count changed")
    _require(
        inventory["survey_role_counts"]
        == {"S4G_P5": 15, "SINGS_IRAC1": 14, "THINGS": 24, "HERACLES": 24},
        "survey counts changed",
    )
    transport = config["transport_accounting"]
    _require(transport["successful_source_gets"] == 77, "GET count changed")
    _require(transport["failed_gets"] == 0, "failed GET count changed")
    _require(transport["redirects_followed"] == 0, "redirect claimed")
    _require(transport["retries"] == 0, "retry claimed")
    _require(transport["network_body_bytes"] == 351_461_418, "network bytes changed")
    _require(transport["response_or_velocity_gets"] == 0, "response GET claimed")
    _require(transport["paid_cost_usd"] == 0.0, "cost changed")
    boundary = config["scientific_boundary"]
    _require(boundary["source_images_opened"], "source access hidden")
    _require(boundary["source_pixels_inspected"], "pixel access hidden")
    _require(not boundary["velocity_or_rotation_response_opened"], "response opened")
    _require(boundary["response_rows_opened"] == 0, "response row count changed")
    _require(boundary["scores_computed"] == 0, "scores computed")
    _require(boundary["models_fit"] == 0, "model fit claimed")
    _require(boundary["selection_events"] == 0, "selection claimed")
    claims = config["claims"]
    _require(claims["exact_source_bytes_and_fits_schemas_validated"], "source claim lost")
    _require(
        not any(
            claims[key] for key in claims if key != "exact_source_bytes_and_fits_schemas_validated"
        ),
        "claim ceiling exceeded",
    )


def _load_predecessor(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = config["predecessor"]
    for role in ("config", "module", "test", "receipt"):
        path = _repo_path(predecessor[f"{role}_path"])
        _require(
            file_sha256(path) == predecessor[f"{role}_raw_sha256"],
            f"predecessor {role} changed",
        )
    prior_receipt = _read_json(_repo_path(predecessor["receipt_path"]), "preflight receipt")
    _require(
        prior_receipt["content_sha256"] == predecessor["receipt_content_sha256"],
        "preflight receipt content changed",
    )
    return _read_json(_repo_path(predecessor["config_path"]), "preflight config")


def load_config() -> dict[str, Any]:
    _require(file_sha256(_repo_path(CONFIG_PATH)) == _CONFIG_RAW_SHA256, "config bytes changed")
    _require(
        module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
        "module changed",
    )
    _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")
    config = _read_json(_repo_path(CONFIG_PATH), "source acquisition config")
    validate_config(config)
    _load_predecessor(config)
    return config


def build_inventory(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    prior_config = _load_predecessor(config)
    sources = preflight.flatten_source_files(prior_config)
    private_root = _repo_path(config["private_source_root"])
    _require(private_root.is_dir(), "private source root missing")
    rows: list[dict[str, Any]] = []
    wcs_by_object_role: dict[tuple[str, str], tuple[Any, ...]] = {}
    shape_by_object_role: dict[tuple[str, str], tuple[int, ...]] = {}
    for source in sources:
        name = _source_filename(source)
        forbidden = tuple(config["fits_contract"]["forbidden_filename_tokens"])
        _require(not any(token in name.upper() for token in forbidden), "response source admitted")
        path = (private_root / name).resolve()
        _require(path.parent == private_root, "private source path escaped")
        _require(path.is_file(), f"source missing: {name}")
        _require(path.stat().st_size == source["bytes"], f"source bytes changed: {name}")
        with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdus:
            _require(len(hdus) == 1 and hdus[0].data is not None, f"HDU inventory changed: {name}")
            header = hdus[0].header
            array = np.asarray(hdus[0].data)
            beam_source, beam = _beam(header)
            contract = config["fits_contract"]
            if source["survey"] == "S4G_P5":
                survey_contract = contract["s4g"]
                _require(array.ndim == survey_contract["ndim"], f"S4G dimensions changed: {name}")
                _require(
                    header.get("BUNIT") == survey_contract["bunit"], f"S4G unit changed: {name}"
                )
                _require(
                    str(header.get("CTYPE1", "")).startswith(survey_contract["ctype1_prefix"]),
                    "S4G CTYPE1 changed",
                )
                _require(
                    str(header.get("CTYPE2", "")).startswith(survey_contract["ctype2_prefix"]),
                    "S4G CTYPE2 changed",
                )
            elif source["survey"] == "SINGS_IRAC1":
                survey_contract = contract["sings"]
                _require(array.ndim == survey_contract["ndim"], f"SINGS dimensions changed: {name}")
                _require(
                    header.get("BUNIT") == survey_contract["bunit"], f"SINGS unit changed: {name}"
                )
                _require(
                    str(header.get("CTYPE1", "")).startswith(survey_contract["ctype1_prefix"]),
                    "SINGS CTYPE1 changed",
                )
                _require(
                    str(header.get("CTYPE2", "")).startswith(survey_contract["ctype2_prefix"]),
                    "SINGS CTYPE2 changed",
                )
                pair_key = (source["object_id"], "SINGS")
                if pair_key in wcs_by_object_role:
                    _require(
                        wcs_by_object_role[pair_key] == _wcs_signature(header),
                        "SINGS pair WCS changed",
                    )
                    _require(
                        shape_by_object_role[pair_key] == tuple(array.shape),
                        "SINGS pair shape changed",
                    )
                else:
                    wcs_by_object_role[pair_key] = _wcs_signature(header)
                    shape_by_object_role[pair_key] = tuple(array.shape)
            elif source["survey"] == "THINGS":
                survey_contract = contract["things"]
                _require(
                    list(array.shape) == survey_contract["shape"], f"THINGS shape changed: {name}"
                )
                _require(
                    header.get("BUNIT") == survey_contract["bunit"], f"THINGS unit changed: {name}"
                )
                _require(
                    str(header.get("CTYPE1", "")).startswith(survey_contract["ctype1_prefix"]),
                    "THINGS CTYPE1 changed",
                )
                _require(
                    str(header.get("CTYPE2", "")).startswith(survey_contract["ctype2_prefix"]),
                    "THINGS CTYPE2 changed",
                )
                _require(
                    header.get("RESTFREQ") in survey_contract["rest_hz_allowed"],
                    "THINGS rest frequency changed",
                )
                _require(
                    beam_source == survey_contract["beam_source"], "THINGS beam source changed"
                )
            else:
                _require(source["survey"] == "HERACLES", "unknown source survey")
                survey_contract = contract["heracles"]
                _require(
                    array.ndim == survey_contract["ndim"], f"HERACLES dimensions changed: {name}"
                )
                _require(
                    header.get("BUNIT") == survey_contract["bunit"],
                    f"HERACLES unit changed: {name}",
                )
                _require(
                    str(header.get("CTYPE1", "")).startswith(survey_contract["ctype1_prefix"]),
                    "HERACLES CTYPE1 changed",
                )
                _require(
                    str(header.get("CTYPE2", "")).startswith(survey_contract["ctype2_prefix"]),
                    "HERACLES CTYPE2 changed",
                )
                if header.get("RESTFREQ") is not None:
                    _require(
                        header.get("RESTFREQ") == survey_contract["rest_hz_if_present"],
                        "HERACLES rest frequency changed",
                    )
                pair_key = (source["object_id"], "HERACLES")
                if pair_key in wcs_by_object_role:
                    _require(
                        wcs_by_object_role[pair_key] == _wcs_signature(header),
                        "HERACLES pair WCS changed",
                    )
                    _require(
                        shape_by_object_role[pair_key] == tuple(array.shape),
                        "HERACLES pair shape changed",
                    )
                else:
                    wcs_by_object_role[pair_key] = _wcs_signature(header)
                    shape_by_object_role[pair_key] = tuple(array.shape)
            row = {
                "object_id": source["object_id"],
                "survey": source["survey"],
                "role": source["role"],
                "url": source["url"],
                "relative_path": path.relative_to(_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "hdu_count": len(hdus),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bunit": str(header.get("BUNIT", "")),
                "ctype1": str(header.get("CTYPE1", "")),
                "ctype2": str(header.get("CTYPE2", "")),
                "rest_hz": header.get("RESTFRQ", header.get("RESTFREQ")),
                "beam_source": beam_source,
                "beam_deg": beam,
                "total_pixels": int(array.size),
                "finite_pixels": int(np.isfinite(array).sum()),
                "decompressed_sha256": decompressed_sha256(path),
            }
            row["row_sha256"] = content_sha256(row)
            rows.append(row)
    inventory = config["inventory_contract"]
    _require(len(rows) == inventory["file_count"], "inventory file count changed")
    _require(
        sum(row["bytes"] for row in rows) == inventory["compressed_source_bytes"],
        "inventory bytes changed",
    )
    _require(
        sum(row["total_pixels"] for row in rows) == inventory["image_pixels_inspected"],
        "inventory pixel count changed",
    )
    _require(
        sum(row["finite_pixels"] for row in rows) == inventory["finite_image_pixels"],
        "inventory finite count changed",
    )
    _require(
        Counter(row["survey"] for row in rows) == inventory["survey_role_counts"],
        "inventory survey counts changed",
    )
    file_root = hashlib.sha256("\n".join(row["sha256"] for row in rows).encode("ascii")).hexdigest()
    decompressed_root = hashlib.sha256(
        "\n".join(row["decompressed_sha256"] for row in rows).encode("ascii")
    ).hexdigest()
    _require(file_root == inventory["ordered_file_sha_root_sha256"], "file SHA root changed")
    _require(
        decompressed_root == inventory["ordered_decompressed_sha_root_sha256"],
        "decompressed SHA root changed",
    )
    _require(content_sha256(rows) == inventory["ordered_record_root_sha256"], "record root changed")
    return rows


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = build_inventory(config)
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "PASS_EXACT_SOURCE_BYTES_AND_FITS_SCHEMAS_READY_FOR_RESPONSE_BLIND_BUILDERS",
        "package_bindings": {
            "config_path": CONFIG_PATH.as_posix(),
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_path": MODULE_PATH.as_posix(),
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_path": TEST_PATH.as_posix(),
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "predecessor": dict(config["predecessor"]),
        "inventory": {
            "object_count": 12,
            "file_count": len(rows),
            "compressed_source_bytes": sum(row["bytes"] for row in rows),
            "image_pixels_inspected": sum(row["total_pixels"] for row in rows),
            "finite_image_pixels": sum(row["finite_pixels"] for row in rows),
            "ordered_file_sha_root_sha256": config["inventory_contract"][
                "ordered_file_sha_root_sha256"
            ],
            "ordered_decompressed_sha_root_sha256": config["inventory_contract"][
                "ordered_decompressed_sha_root_sha256"
            ],
            "ordered_record_root_sha256": content_sha256(rows),
            "records": rows,
        },
        "builder_admission": dict(config["builder_admission"]),
        "transport_accounting": dict(config["transport_accounting"]),
        "checks": {
            "exact_77_files": True,
            "exact_351461418_compressed_bytes": True,
            "all_raw_and_decompressed_hashes_sealed": True,
            "single_primary_image_hdu_only": True,
            "survey_units_dimensions_wcs_and_beams_validated": True,
            "paired_sings_flux_weight_wcs_exact": True,
            "paired_heracles_mom0_emom0_wcs_exact": True,
            "no_velocity_map_cube_or_rotation_response": True,
            "no_scoring_or_model_fit": True,
        },
        "scientific_boundary": dict(config["scientific_boundary"]),
        "claims": dict(config["claims"]),
        "next_action": (
            "validate the frozen SINGS IRAC1 flux-to-mass builder on manufactured images and the "
            "three SINGS/S4G overlap galaxies before any response scoring"
        ),
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require(type(receipt) is dict, "receipt is not an object")
    supplied = dict(receipt)
    claimed = supplied.pop("content_sha256", None)
    _require(type(claimed) is str and len(claimed) == 64, "receipt hash missing")
    _require(content_sha256(supplied) == claimed, "receipt self-hash mismatch")
    _require(dict(receipt) == build_receipt(config), "receipt does not exactly rebuild")


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    output = _repo_path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output.exists():
        validate_receipt(_read_json(output, "stored receipt"), config)
        _require(output.read_text(encoding="utf-8") == payload, "stored receipt bytes changed")
        return "EXISTING_IDENTICAL"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False, newline="\n"
        ) as handle:
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
        raise ExpansionSourceError("receipt appeared concurrently") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "CREATED"


def check_receipt() -> str:
    config = load_config()
    output = _repo_path(OUTPUT_PATH)
    _require(output.exists(), "receipt missing")
    validate_receipt(_read_json(output, "stored receipt"), config)
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
        receipt = build_receipt(load_config())
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "decision": receipt["decision"],
                    "objects": receipt["inventory"]["object_count"],
                    "source_files": receipt["inventory"]["file_count"],
                    "source_bytes": receipt["inventory"]["compressed_source_bytes"],
                    "source_pixels_inspected": receipt["inventory"]["image_pixels_inspected"],
                    "response_rows_opened": receipt["scientific_boundary"]["response_rows_opened"],
                    "scores_computed": receipt["scientific_boundary"]["scores_computed"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
