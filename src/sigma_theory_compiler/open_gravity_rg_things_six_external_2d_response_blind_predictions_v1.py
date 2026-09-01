"""Build response-blind two-dimensional predictions for six external THINGS galaxies."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
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
    open_gravity_rg_holmberg_ii_things_2d_response_blind_predictions_v1 as method,
)

CONFIG_PATH = Path(
    "configs/open_gravity_rg_things_six_external_2d_response_blind_predictions_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/"
    "open_gravity_rg_things_six_external_2d_response_blind_predictions_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_rg_things_six_external_2d_response_blind_predictions_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-things-six-external-2d-response-blind-predictions-v1/receipt.json"
)
PRIVATE_DIRECTORY = Path(
    "work/private/open-gravity-rg-things-six-external-2d-response-blind-predictions-v1"
)
PRIVATE_MANIFEST_PATH = PRIVATE_DIRECTORY / "manifest.json"

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-six-external-2d-response-blind-predictions-1.0"
_CELL_SCHEMA = "invariant-open-gravity-rg-things-six-external-2d-response-blind-prediction-cell-1.0"
_MANIFEST_SCHEMA = (
    "invariant-open-gravity-rg-things-six-external-2d-response-blind-private-manifest-1.0"
)
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-rg-things-six-external-2d-response-blind-predictions-receipt-1.0"
)
_CANDIDATES = method._CANDIDATES
_RESOLUTIONS = ("ROBUST", "NATURAL")
_CONFIG_RAW_SHA256 = "b9996e29a691183151c486ff7219a66f8521e935b0bb875de842b31781ab4890"
_CONFIG_CONTENT_SHA256 = "b6de63072d1711bff3f566e164b24572333bc633d7e59ee53f553c59b85c8f70"
_MODULE_SEMANTIC_SHA256 = "c72e92ef68651e41a28500c0e6341c94b94f2824ce7233a19e9b575ace1f6b4b"
_TEST_RAW_SHA256 = "b5458a07c723255b4d3ed8bc934592a123724aa6c3af2aaef0b558392fe40ae8"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class SixGalaxyPredictionError(RuntimeError):
    """Raised when a source, numerical, response-blind, or seal gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SixGalaxyPredictionError(message)


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


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    normalized, count = _MODULE_PIN_PATTERN.subn(
        rb"\g<1>" + b"0" * 64 + rb"\g<2>", path.read_bytes()
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path == _ROOT or _ROOT in path.parents, "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SixGalaxyPredictionError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _package_bindings() -> dict[str, str]:
    return {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"]
        == "FROZEN_RESPONSE_BLIND_SIX_EXTERNAL_GALAXY_2D_FOUR_LAW_PREDICTION_BUILD",
        "status changed",
    )
    roles = [row["role"] for row in config["predecessor_bindings"]]
    _require(
        roles
        == [
            "SIX_EXTERNAL_2D_REPLICATION_PREFLIGHT",
            "SEVEN_HOLDOUT_SOURCE_BUILDER",
            "HOLMBERG_RESPONSE_BLIND_2D_METHOD",
            "AUDITED_3D_DST_PCG_MECHANICS",
            "PUBLISHED_CONTROL_FORMULAS",
            "AUDITED_2D_WCS_BEAM_PROJECTION",
        ],
        "predecessor inventory changed",
    )
    candidates = config["candidate_contract"]
    _require(candidates["candidate_ids"] == list(_CANDIDATES), "candidates changed")
    _require(candidates["a0_m_s2"] == 1.2e-10, "a0 changed")
    _require(
        candidates["refracted_gravity_parameters"]
        == {
            "published_parameter_id": "DISKMASS_UNIVERSAL_MEDIAN",
            "epsilon_0": 0.661,
            "Q": 1.79,
            "log10_rho_c_g_cm3": -24.54,
        },
        "RG parameters changed",
    )
    for key in ("response_parameter_fitting", "response_parameter_tuning", "best_cell_selection"):
        _require(candidates[key] is False, f"forbidden selection enabled: {key}")
    _require(candidates["same_fixed_parameters_all_objects"] is True, "shared constants removed")
    source = config["source_contract"]
    object_order = ["NGC2841", "IC2574", "DDO154", "NGC5055", "NGC6946", "NGC7331"]
    _require(source["object_order"] == object_order, "objects changed")
    _require(sum(source["expected_built_cells_by_object"].values()) == 15, "cell count changed")
    _require(sum(source["expected_failed_conversions_by_object"].values()) == 3, "failures changed")
    _require(source["source_cells"] == 15, "source cells changed")
    _require(source["retained_source_failures"] == 3, "failure retention changed")
    _require(source["model_lift_label"] == "MODEL_LIFTED_2P5D", "3D overclaim")
    _require(source["response_values_used"] is False, "response entered source")
    headers = config["response_header_contracts"]
    _require(list(headers) == object_order, "header object order changed")
    for object_id, contract in headers.items():
        _require(contract["shape"] == [1, 1, 1024, 1024], f"shape changed: {object_id}")
        _require(contract["bunit"] == "METR/SEC", f"unit changed: {object_id}")
        _require(contract["ctype"] == ["RA---SIN", "DEC--SIN"], f"WCS changed: {object_id}")
        _require(contract["crpix"] == [512.0, 513.0], f"CRPIX changed: {object_id}")
        _require(
            contract["cdelt_deg"] == [-0.0004166666768, 0.0004166666768],
            f"pixel scale changed: {object_id}",
        )
        _require(len(contract["natural_beam_deg"]) == 3, "natural beam changed")
        _require(len(contract["robust_beam_deg"]) == 3, "robust beam changed")
    grid = config["grid_contract"]
    _require(grid["solver_half_box_kpc"] == 30.0, "box changed")
    _require(grid["fine_nodes_per_axis"] == 241, "fine grid changed")
    _require(grid["convergence_nodes_per_axis"] == 193, "convergence grid changed")
    _require(grid["minimum_radius_kpc"] == 0.5, "minimum radius changed")
    _require(grid["maximum_radius_kpc"] == 15.0, "maximum radius changed")
    operator = config["operator_contract"]
    _require(operator["pcg_relative_tolerance"] == 1e-10, "PCG tolerance changed")
    _require(operator["pcg_absolute_tolerance"] == 0.0, "PCG absolute tolerance changed")
    _require(operator["pcg_max_iterations"] == 100, "PCG ceiling changed")
    _require(operator["maximum_solver_relative_residual"] == 1e-8, "residual gate changed")
    _require(operator["maximum_source_mass_relative_error"] == 2e-9, "mass gate changed")
    _require(operator["maximum_local_relative_difference"] == 0.05, "local gate changed")
    projection = config["projection_contract"]
    _require(
        projection["minimum_natural_eligible_intensity_fraction"] == 0.99,
        "coverage gate changed",
    )
    _require(projection["response_values_used"] is False, "response entered projection")
    execution = config["execution_contract"]
    _require(
        execution
        == {
            "objects": 6,
            "source_cells": 15,
            "retained_source_failures": 3,
            "field_solver_runs": 60,
            "candidate_resolution_predictions": 120,
            "private_arrays_per_cell": 13,
            "private_array_files": 195,
            "predecessor_response_assets_sealed": 24,
            "response_assets_read_in_this_build": 0,
            "response_headers_opened_in_this_build": 0,
            "response_pixels_opened": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "execution accounting changed",
    )
    private = config["private_output"]
    _require(private["directory"] == PRIVATE_DIRECTORY.as_posix(), "private directory changed")
    _require(private["manifest"] == PRIVATE_MANIFEST_PATH.as_posix(), "manifest changed")
    _require(len(private["array_roles"]) == 13, "private roles changed")
    _require(all(value == 0 for value in config["response_boundary"].values()), "response leak")
    claims = config["claim_boundary"]
    for key in (
        "response_pixels_opened",
        "response_blind_predictions_built",
        "two_dimensional_score_completed",
        "refracted_gravity_replication_established",
        "inclination_crossover_generalized",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim promoted before build: {key}")
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


def _load_predecessors(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["predecessor_bindings"]:
        receipt: dict[str, Any] | None = None
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "predecessor artifact missing")
            _require(file_sha256(path) == artifact["sha256"], "predecessor artifact changed")
            if artifact["path"].endswith("receipt.json"):
                receipt = _read_json(path, "predecessor receipt")
        _require(receipt is not None, "predecessor receipt missing")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            "predecessor receipt content changed",
        )
        if "required_status" in binding:
            _require(receipt["status"] == binding["required_status"], "preflight status changed")
        receipts[binding["role"]] = receipt
    preflight_binding = config["predecessor_bindings"][0]
    receipts["SIX_EXTERNAL_2D_REPLICATION_PREFLIGHT"]["_config"] = _read_json(
        _repo_path(preflight_binding["artifacts"][0]["path"]), "preflight config"
    )
    source_binding = config["predecessor_bindings"][1]
    receipts["SEVEN_HOLDOUT_SOURCE_BUILDER"]["_config"] = _read_json(
        _repo_path(source_binding["artifacts"][0]["path"]), "source config"
    )
    return receipts


def cell_run_id(source_cell: Mapping[str, Any]) -> str:
    return method.cell_run_id(source_cell)


def _source_inventory(
    config: Mapping[str, Any], source_receipt: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    source = config["source_contract"]
    object_order = source["object_order"]
    all_rows = [row for row in source_receipt["source_cells"] if row["object_id"] in object_order]
    _require(len(all_rows) == 18, "external source inventory changed")
    built = [row for row in all_rows if row["disposition"] == "SOURCE_MAP_BUILT_RESPONSE_BLIND"]
    failed = [row for row in all_rows if row["disposition"] != "SOURCE_MAP_BUILT_RESPONSE_BLIND"]
    _require(len(built) == 15, "built source cell count changed")
    _require(len(failed) == 3, "source failure count changed")
    _require(
        {row["disposition"] for row in failed}
        == {"SOURCE_CONVERSION_FAILED_UNPHYSICAL_FASTICA_COLOR_RETAINED"},
        "source failure disposition changed",
    )
    built_counts = Counter(row["object_id"] for row in built)
    failed_counts = Counter(row["object_id"] for row in failed)
    _require(dict(built_counts) == source["expected_built_cells_by_object"], "built cells changed")
    expected_failed = {
        key: value
        for key, value in source["expected_failed_conversions_by_object"].items()
        if value
    }
    _require(dict(failed_counts) == expected_failed, "failed cells changed")
    conversion_order = {
        value: index for index, value in enumerate(source["stellar_conversion_cells"])
    }
    object_index = {value: index for index, value in enumerate(object_order)}
    all_rows.sort(
        key=lambda row: (
            object_index[row["object_id"]],
            conversion_order[row["conversion_cell_id"]],
        )
    )
    built = [row for row in all_rows if row["disposition"] == "SOURCE_MAP_BUILT_RESPONSE_BLIND"]
    failed = [row for row in all_rows if row["disposition"] != "SOURCE_MAP_BUILT_RESPONSE_BLIND"]
    _require(len({cell_run_id(row) for row in built}) == 15, "source cell IDs duplicated")
    for row in built:
        _require(row["model_lift_label"] == "MODEL_LIFTED_2P5D", "source dimension changed")
    return built, failed


def _header_from_contract(contract: Mapping[str, Any]) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 4
    header["NAXIS1"] = int(contract["shape"][3])
    header["NAXIS2"] = int(contract["shape"][2])
    header["NAXIS3"] = int(contract["shape"][1])
    header["NAXIS4"] = int(contract["shape"][0])
    header["BUNIT"] = contract["bunit"]
    header["CTYPE1"] = contract["ctype"][0]
    header["CTYPE2"] = contract["ctype"][1]
    header["CRVAL1"] = float(contract["crval_deg"][0])
    header["CRVAL2"] = float(contract["crval_deg"][1])
    header["CRPIX1"] = float(contract["crpix"][0])
    header["CRPIX2"] = float(contract["crpix"][1])
    header["CDELT1"] = float(contract["cdelt_deg"][0])
    header["CDELT2"] = float(contract["cdelt_deg"][1])
    return header


def _validate_preflight_metadata(
    config: Mapping[str, Any], preflight_receipt: Mapping[str, Any]
) -> None:
    object_order = config["source_contract"]["object_order"]
    _require(
        [row["object_id"] for row in preflight_receipt["objects"]] == object_order,
        "preflight objects changed",
    )
    assets = preflight_receipt["response_assets"]
    _require(len(assets) == 24, "preflight response inventory changed")
    expected_keys = {
        (object_id, resolution, observable)
        for object_id in object_order
        for resolution in ("NATURAL", "ROBUST")
        for observable in ("MOM1", "MOM2")
    }
    _require(
        {(row["object_id"], row["resolution"], row["observable"]) for row in assets}
        == expected_keys,
        "preflight response keys changed",
    )
    for row in assets:
        contract = config["response_header_contracts"][row["object_id"]]
        _require(row["shape"] == [1024, 1024], "preflight shape changed")
        _require(row["wcs_ctype"] == contract["ctype"], "preflight WCS changed")
        expected_beam = contract[f"{row['resolution'].lower()}_beam_deg"]
        _require(row["beam_deg"] == expected_beam, "preflight beam changed")
        _require(row["pixel_values_decoded"] == 0, "preflight decoded response")
    _require(
        preflight_receipt["acquisition_accounting"]["response_pixels_decoded"] == 0, "response leak"
    )


def _build_cell_arrays(
    config: Mapping[str, Any],
    source_cell: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cell_config = dict(config)
    contract = config["response_header_contracts"][source_cell["object_id"]]
    cell_config["response_header_contract"] = contract
    header = _header_from_contract(contract)
    method._validate_header(header, contract)
    return method._build_cell_arrays(
        cell_config,
        source_cell,
        source_receipt,
        source_config,
        header,
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


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
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent output differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _private_path(relative: str) -> Path:
    private = _repo_path(PRIVATE_DIRECTORY)
    path = (private / relative).resolve()
    _require(path == private or private in path.parents, "private path escaped")
    return path


def _cell_payload_path(source_cell: Mapping[str, Any]) -> Path:
    return _private_path(f"{cell_run_id(source_cell)}__cell.json")


def _array_relative_path(source_cell: Mapping[str, Any], role: str) -> str:
    _require(re.fullmatch(r"[A-Z0-9_]+(?:__[A-Z]+)?|[a-z_]+", role) is not None, "bad role")
    return f"{cell_run_id(source_cell)}__{role}.npy"


def _validate_array_rows(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    _require(
        [row["role"] for row in rows] == config["private_output"]["array_roles"],
        "array roles changed",
    )
    for row in rows:
        path = _private_path(row["relative_path"])
        _require(path.is_file(), "prediction array missing")
        _require(path.stat().st_size == row["bytes"], "prediction array size changed")
        _require(file_sha256(path) == row["file_sha256"], "prediction array bytes changed")
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise SixGalaxyPredictionError("invalid prediction array") from exc
        _require(list(array.shape) == row["shape"], "prediction shape changed")
        _require(str(array.dtype) == row["dtype"], "prediction dtype changed")
        _require(array_sha256(array) == row["array_sha256"], "prediction values changed")


def _build_cell_payload(
    config: Mapping[str, Any],
    source_cell: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    serialized: dict[str, bytes] = {}
    for role in config["private_output"]["array_roles"]:
        _require(role in arrays, "prediction array role missing")
        payload = _npy_bytes(arrays[role])
        relative = _array_relative_path(source_cell, role)
        serialized[relative] = payload
        rows.append(
            {
                "role": role,
                "relative_path": relative,
                "bytes": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "array_sha256": array_sha256(arrays[role]),
                "shape": list(arrays[role].shape),
                "dtype": str(arrays[role].dtype),
            }
        )
    cell: dict[str, Any] = {
        "schema": _CELL_SCHEMA,
        "package_id": config["package_id"],
        "package_bindings": _package_bindings(),
        "cell_run_id": cell_run_id(source_cell),
        "object_id": source_cell["object_id"],
        "conversion_cell_id": source_cell["conversion_cell_id"],
        "geometry": source_cell["geometry"],
        "source_profile_sha256": source_cell["profile_sha256"],
        "model_lift_label": source_cell["model_lift_label"],
        "arrays": rows,
        "diagnostics": diagnostics,
        "response_boundary": config["response_boundary"],
    }
    cell["content_sha256"] = content_sha256(cell)
    return cell, serialized


def validate_cell_payload(
    config: Mapping[str, Any], source_cell: Mapping[str, Any], cell: Mapping[str, Any]
) -> None:
    _require(cell["schema"] == _CELL_SCHEMA, "cell schema changed")
    _require(cell["package_id"] == config["package_id"], "cell package changed")
    _require(cell["package_bindings"] == _package_bindings(), "cell package seal changed")
    _require(cell["cell_run_id"] == cell_run_id(source_cell), "cell ID changed")
    _require(cell["object_id"] == source_cell["object_id"], "cell object changed")
    _require(cell["conversion_cell_id"] == source_cell["conversion_cell_id"], "conversion changed")
    _require(cell["geometry"] == source_cell["geometry"], "geometry changed")
    _require(cell["source_profile_sha256"] == source_cell["profile_sha256"], "source changed")
    _require(cell["model_lift_label"] == "MODEL_LIFTED_2P5D", "dimension overclaim")
    _require(cell["response_boundary"] == config["response_boundary"], "response leak")
    copy = dict(cell)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "cell content hash changed")
    _validate_array_rows(config, cell["arrays"])


def write_cell(cell_id: str) -> str:
    config = load_config()
    predecessors = _load_predecessors(config)
    preflight = predecessors["SIX_EXTERNAL_2D_REPLICATION_PREFLIGHT"]
    _validate_preflight_metadata(config, preflight)
    source_receipt = predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"]
    source_config = source_receipt["_config"]
    source_cells, _ = _source_inventory(config, source_receipt)
    source_cell = next((row for row in source_cells if cell_run_id(row) == cell_id), None)
    _require(source_cell is not None, "unknown source cell")
    cell_path = _cell_payload_path(source_cell)
    if cell_path.is_file():
        cell = _read_json(cell_path, "prediction cell")
        validate_cell_payload(config, source_cell, cell)
        return "EXISTING_VALID"
    arrays, diagnostics = _build_cell_arrays(config, source_cell, source_receipt, source_config)
    cell, serialized = _build_cell_payload(config, source_cell, arrays, diagnostics)
    statuses = [
        _atomic_no_clobber(_private_path(relative), payload)
        for relative, payload in serialized.items()
    ]
    statuses.append(_atomic_no_clobber(cell_path, canonical_bytes(cell) + b"\n"))
    del arrays
    gc.collect()
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def _load_completed_cells(
    config: Mapping[str, Any], source_cells: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source_cell in source_cells:
        path = _cell_payload_path(source_cell)
        _require(path.is_file(), f"prediction cell missing: {cell_run_id(source_cell)}")
        cell = _read_json(path, "prediction cell")
        validate_cell_payload(config, source_cell, cell)
        output.append(cell)
    return output


def _source_failure_rows(failed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "object_id": row["object_id"],
            "conversion_cell_id": row["conversion_cell_id"],
            "geometry_variant_id": row["geometry"]["geometry_variant_id"],
            "disposition": row["disposition"],
            "failure_evidence": row["failure_evidence"],
        }
        for row in failed
    ]


def build_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessors = _load_predecessors(config)
    _validate_preflight_metadata(config, predecessors["SIX_EXTERNAL_2D_REPLICATION_PREFLIGHT"])
    source_cells, failed = _source_inventory(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
    cells = _load_completed_cells(config, source_cells)
    arrays = [
        dict(row) | {"cell_run_id": cell["cell_run_id"]} for cell in cells for row in cell["arrays"]
    ]
    manifest: dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "package_id": config["package_id"],
        "package_bindings": _package_bindings(),
        "object_count": 6,
        "cell_count": len(cells),
        "retained_source_failures": _source_failure_rows(failed),
        "array_file_count": len(arrays),
        "candidate_resolution_prediction_count": len(cells) * len(_CANDIDATES) * len(_RESOLUTIONS),
        "cell_content_sha256": [cell["content_sha256"] for cell in cells],
        "arrays": arrays,
        "response_boundary": config["response_boundary"],
    }
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def validate_manifest(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    _require(manifest["schema"] == _MANIFEST_SCHEMA, "manifest schema changed")
    _require(manifest["package_id"] == config["package_id"], "manifest package changed")
    _require(manifest["package_bindings"] == _package_bindings(), "manifest seal changed")
    _require(manifest["object_count"] == 6, "manifest object count changed")
    _require(manifest["cell_count"] == 15, "manifest cell count changed")
    _require(len(manifest["retained_source_failures"]) == 3, "manifest failures changed")
    _require(manifest["array_file_count"] == 195, "manifest array count changed")
    _require(
        manifest["candidate_resolution_prediction_count"] == 120, "manifest predictions changed"
    )
    _require(manifest["response_boundary"] == config["response_boundary"], "manifest response leak")
    copy = dict(manifest)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "manifest content hash changed")


def write_manifest() -> str:
    config = load_config()
    manifest = build_manifest(config)
    validate_manifest(config, manifest)
    return _atomic_no_clobber(_repo_path(PRIVATE_MANIFEST_PATH), canonical_bytes(manifest) + b"\n")


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _repo_path(PRIVATE_MANIFEST_PATH)
    _require(manifest_path.is_file(), "private manifest missing")
    manifest = _read_json(manifest_path, "private manifest")
    validate_manifest(config, manifest)
    predecessors = _load_predecessors(config)
    _validate_preflight_metadata(config, predecessors["SIX_EXTERNAL_2D_REPLICATION_PREFLIGHT"])
    source_cells, failed = _source_inventory(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
    cells = _load_completed_cells(config, source_cells)
    claims = dict(config["claim_boundary"])
    claims["response_blind_predictions_built"] = True
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_RESPONSE_BLIND_120_SIX_EXTERNAL_GALAXY_PREDICTIONS_SEALED",
        "decision": "READY_TO_OPEN_SEALED_THINGS_RESPONSE_FOR_FIXED_SIX_GALAXY_2D_SCORE",
        "package_bindings": _package_bindings(),
        "predecessor_receipt_content_sha256": {
            binding["role"]: binding["receipt_content_sha256"]
            for binding in config["predecessor_bindings"]
        },
        "objects": config["source_contract"]["object_order"],
        "source_cells": 15,
        "retained_source_failures": _source_failure_rows(failed),
        "candidate_ids": list(_CANDIDATES),
        "response_resolutions": list(_RESOLUTIONS),
        "candidate_resolution_predictions": 120,
        "private_manifest": {
            "path": PRIVATE_MANIFEST_PATH.as_posix(),
            "raw_sha256": file_sha256(manifest_path),
            "content_sha256": manifest["content_sha256"],
            "array_file_count": manifest["array_file_count"],
        },
        "cell_summaries": [
            {
                "cell_run_id": cell["cell_run_id"],
                "content_sha256": cell["content_sha256"],
                "object_id": cell["object_id"],
                "conversion_cell_id": cell["conversion_cell_id"],
                "geometry_variant_id": cell["geometry"]["geometry_variant_id"],
                "inclination_deg": cell["geometry"]["inclination_deg"],
                "all_solver_gates_pass": cell["diagnostics"]["all_solver_gates_pass"],
                "robust_eligible_pixels": cell["diagnostics"]["robust_eligible_pixels"],
                "natural_eligible_pixels": cell["diagnostics"]["natural_eligible_pixels"],
            }
            for cell in cells
        ],
        "all_solver_gates_pass": all(
            cell["diagnostics"]["all_solver_gates_pass"] for cell in cells
        ),
        "response_boundary": config["response_boundary"],
        "execution_accounting": config["execution_contract"],
        "claim_boundary": claims,
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(dict(receipt) == build_receipt(config), "receipt differs from deterministic rebuild")


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    validate_receipt(config, receipt)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n")


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    receipt = _read_json(path, "receipt")
    validate_receipt(config, receipt)
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    predecessors = _load_predecessors(config)
    cells, failed = _source_inventory(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
    completed = sum(_cell_payload_path(cell).is_file() for cell in cells)
    return {
        "package_id": config["package_id"],
        "completed_cells": completed,
        "required_cells": len(cells),
        "retained_source_failures": len(failed),
        "manifest_exists": _repo_path(PRIVATE_MANIFEST_PATH).is_file(),
        "receipt_exists": _repo_path(OUTPUT_PATH).is_file(),
        "response_assets_read_in_this_build": 0,
        "response_pixels_opened": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    cell = sub.add_parser("write-cell")
    cell.add_argument("--cell-id", required=True)
    sub.add_parser("write-all")
    sub.add_parser("write-manifest")
    sub.add_parser("write-receipt")
    sub.add_parser("check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "status":
        print(json.dumps(status(), sort_keys=True))
    elif arguments.command == "write-cell":
        print(write_cell(arguments.cell_id))
    elif arguments.command == "write-all":
        config = load_config()
        predecessors = _load_predecessors(config)
        cells, _ = _source_inventory(config, predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"])
        for cell in cells:
            print(f"{cell_run_id(cell)} {write_cell(cell_run_id(cell))}", flush=True)
    elif arguments.command == "write-manifest":
        print(write_manifest())
    elif arguments.command == "write-receipt":
        print(write_receipt())
    else:
        print(check_receipt())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
