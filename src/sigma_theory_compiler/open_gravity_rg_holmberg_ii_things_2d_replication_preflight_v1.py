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

CONFIG_PATH = Path("configs/open_gravity_rg_holmberg_ii_things_2d_replication_preflight_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_holmberg_ii_things_2d_replication_preflight_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_holmberg_ii_things_2d_replication_preflight_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-holmberg-ii-things-2d-replication-preflight-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-things-2d-replication-preflight-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-rg-holmberg-ii-things-2d-replication-preflight-receipt-1.0"
)
_CONFIG_RAW_SHA256 = "827d741fc8d1d7dd4d161152bb92e4a90afb523d3a53a3855c6b65f3d49becb9"
_CONFIG_CONTENT_SHA256 = "92b7491dd8919e823a118297ca6944c5ad163c0660a2f53ec844f3d9f0789bed"
_MODULE_SEMANTIC_SHA256 = "08a2fc6c058c63b7e60bc51ba8f20aac03ed708a15e59b52b45a3cd7df3c886b"
_TEST_RAW_SHA256 = "67cc6aa1fff400776a77c3c1e07a30d8dc91ab383e6498d2c641c267072ffec3"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')
_BEAM_PATTERN = re.compile(
    r"AIPS\s+CLEAN\s+BMAJ=\s*([-+0-9.Ee]+)\s+BMIN=\s*([-+0-9.Ee]+)\s+BPA=\s*([-+0-9.Ee]+)"
)


class ReplicationPreflightError(RuntimeError):
    """Raised when the source-matched Holmberg II preflight fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplicationPreflightError(message)


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
    raw = path.read_bytes()
    normalized, count = _MODULE_PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", raw)
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    candidate = (_ROOT / relative).resolve()
    _require(candidate == _ROOT or _ROOT in candidate.parents, "path escaped repository")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplicationPreflightError(f"invalid {label}") from exc
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
        config["status"] == "FROZEN_SOURCE_MATCHED_2D_REPLICATION_BYTES_SEALED_PIXELS_UNOPENED",
        "status changed",
    )
    sources = config["primary_sources"]
    _require(len(sources) == 4, "primary-source count changed")
    _require(all(row["url"].startswith("https://") for row in sources), "source URL changed")
    assets = config["response_assets"]
    _require(len(assets) == 4, "response inventory changed")
    _require(len({row["role"] for row in assets}) == 4, "response role repeated")
    _require(sum(row["bytes"] for row in assets) == 16_951_680, "response bytes changed")
    _require(
        {(row["resolution"], row["observable"]) for row in assets}
        == {("NATURAL", "MOM1"), ("NATURAL", "MOM2"), ("ROBUST", "MOM1"), ("ROBUST", "MOM2")},
        "response roles changed",
    )
    header = config["exact_header_contract"]
    _require(header["shape"] == [1, 1, 1024, 1024], "header shape changed")
    _require(header["bunit"] == "METR/SEC", "header unit changed")
    _require(header["data_array_opened_for_header_validation"] is False, "data access claimed")
    prediction = config["prediction_contract"]
    _require(prediction["source_cells"] == 9, "source-cell count changed")
    _require(prediction["candidate_resolution_cells"] == 72, "candidate coverage changed")
    _require(prediction["inclination_cells_deg"] == [27.0, 38.0, 49.0], "inclinations changed")
    _require(prediction["response_pixels_used_to_build_predictions"] is False, "blindness lost")
    _require(prediction["best_geometry_or_conversion_selection"] is False, "selection enabled")
    _require(prediction["parameters_fitted"] == 0, "parameter fitting enabled")
    _require(prediction["parameters_tuned"] == 0, "parameter tuning enabled")
    score = config["future_score_contract"]
    _require(score["all_18_source_resolution_cells_reported"] is True, "score coverage reduced")
    _require(score["model_specific_systemic_offset_primary"] is False, "model offset enabled")
    _require(score["response_tuning_calls"] == 0, "response tuning enabled")
    _require(score["per_model_sign_selection"] is False, "sign selection enabled")
    _require(score["source_geometry_reselection"] is False, "geometry selection enabled")
    _require(score["p_values_computed"] is False, "p-value overclaim")
    _require(score["retain_every_failure_and_counterexample"] is True, "failures hidden")
    access = config["preparation_access_disclosure"]
    _require(access["fits_headers_opened"] == 4, "header access changed")
    _require(access["velocity_or_dispersion_pixels_decoded"] == 0, "pixels decoded")
    _require(access["scientific_scores_computed"] == 0, "score already computed")
    _require(
        access["candidate_or_parameter_changes_after_header_access"] == 0, "post-access change"
    )
    _require(
        access["header_extrema_used_for_candidate_geometry_mask_or_score_design"] is False,
        "header leakage",
    )
    _require(len(config["scientific_limitations"]) == 6, "limitations changed")
    _require(all(value is False for value in config["claim_boundary"].values()), "claim promoted")
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


def _validate_binding(binding: Mapping[str, Any], receipt_label: str) -> dict[str, Any]:
    path_keys = [key for key in binding if key.endswith("_path")]
    _require(len(path_keys) == 4, f"{receipt_label} binding paths changed")
    for path_key in path_keys:
        hash_key = path_key.replace("_path", "_sha256")
        path = _repo_path(binding[path_key])
        _require(path.is_file(), f"{receipt_label} artifact missing")
        _require(file_sha256(path) == binding[hash_key], f"{receipt_label} artifact changed")
    receipt = _read_json(_repo_path(binding["receipt_path"]), f"{receipt_label} receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"],
        f"{receipt_label} content changed",
    )
    return receipt


def _beam_from_history(header: fits.Header) -> list[float]:
    matches = []
    history = header.get("HISTORY", [])
    if isinstance(history, str):
        history = [history]
    for line in history:
        match = _BEAM_PATTERN.search(str(line))
        if match:
            matches.append([float(match.group(index)) for index in (1, 2, 3)])
    _require(len(matches) >= 1, "beam history missing")
    return matches[-1]


def _header_summary(path: Path) -> dict[str, Any]:
    try:
        header = fits.getheader(path, 0)
    except OSError as exc:
        raise ReplicationPreflightError("invalid FITS header") from exc
    return {
        "object": str(header["OBJECT"]),
        "shape": [
            int(header.get("NAXIS4", 1)),
            int(header.get("NAXIS3", 1)),
            int(header["NAXIS2"]),
            int(header["NAXIS1"]),
        ],
        "bunit": str(header["BUNIT"]),
        "ctype": [str(header["CTYPE1"]), str(header["CTYPE2"])],
        "crval_deg": [float(header["CRVAL1"]), float(header["CRVAL2"])],
        "crpix": [float(header["CRPIX1"]), float(header["CRPIX2"])],
        "cdelt_deg": [float(header["CDELT1"]), float(header["CDELT2"])],
        "beam_deg": _beam_from_history(header),
        "data_array_opened": False,
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    trigger = _validate_binding(config["trigger_binding"], "trigger")
    _require(
        trigger["robustness_findings"]["signal_classification"]
        == config["trigger_binding"]["required_finding"],
        "trigger finding changed",
    )
    source = _validate_binding(config["source_builder_binding"], "source")
    source_cells = [
        row
        for row in source["source_cells"]
        if row["object_id"] == config["source_builder_binding"]["required_object_id"]
        and row["disposition"] == "SOURCE_MAP_BUILT_RESPONSE_BLIND"
    ]
    _require(len(source_cells) == 9, "Holmberg II source cells changed")
    _require(
        all(
            row["model_lift_label"] == config["source_builder_binding"]["required_model_lift_label"]
            for row in source_cells
        ),
        "source model-lift label changed",
    )

    header_contract = config["exact_header_contract"]
    asset_rows = []
    for asset in config["response_assets"]:
        path = _repo_path(asset["relative_path"])
        _require(path.is_file(), "response asset missing")
        _require(path.stat().st_size == asset["bytes"], "response asset bytes changed")
        _require(file_sha256(path) == asset["sha256"], "response asset hash changed")
        summary = _header_summary(path)
        for key in ("object", "shape", "bunit", "ctype", "crval_deg", "crpix", "cdelt_deg"):
            _require(summary[key] == header_contract[key], f"response header changed: {key}")
        expected_beam = (
            header_contract["natural_beam_deg"]
            if asset["resolution"] == "NATURAL"
            else header_contract["robust_beam_deg"]
        )
        _require(summary["beam_deg"] == expected_beam, "response beam changed")
        asset_rows.append(
            {
                "role": asset["role"],
                "relative_path": asset["relative_path"],
                "bytes": asset["bytes"],
                "sha256": asset["sha256"],
                "header": summary,
            }
        )

    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "READY_RESPONSE_BLIND_2D_PREDICTION_BUILD",
        "package_bindings": _package_bindings(),
        "trigger_receipt_content_sha256": trigger["content_sha256"],
        "source_receipt_content_sha256": source["content_sha256"],
        "response_asset_count": len(asset_rows),
        "response_network_bytes": sum(row["bytes"] for row in asset_rows),
        "response_assets": asset_rows,
        "holmberg_ii_source_cell_ids": sorted(
            f"{row['conversion_cell_id']}__{row['geometry']['geometry_variant_id']}"
            for row in source_cells
        ),
        "prediction_contract": config["prediction_contract"],
        "future_score_contract": config["future_score_contract"],
        "preparation_access_disclosure": config["preparation_access_disclosure"],
        "scientific_limitations": config["scientific_limitations"],
        "claim_boundary": {
            **config["claim_boundary"],
            "response_assets_sealed": True,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    validate_receipt(config, receipt, recompute=False)
    return receipt


def validate_receipt(
    config: Mapping[str, Any], receipt: Mapping[str, Any], *, recompute: bool = True
) -> None:
    _require(receipt["schema"] == _RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt["package_id"] == config["package_id"], "receipt package changed")
    _require(receipt["package_bindings"] == _package_bindings(), "receipt package seal changed")
    _require(receipt["status"] == "READY_RESPONSE_BLIND_2D_PREDICTION_BUILD", "status changed")
    _require(receipt["response_asset_count"] == 4, "asset count changed")
    _require(receipt["response_network_bytes"] == 16_951_680, "network bytes changed")
    _require(len(receipt["holmberg_ii_source_cell_ids"]) == 9, "source-cell ledger changed")
    _require(
        receipt["preparation_access_disclosure"] == config["preparation_access_disclosure"],
        "access changed",
    )
    claims = receipt["claim_boundary"]
    _require(claims["response_assets_sealed"] is True, "asset seal lost")
    for key, value in claims.items():
        if key != "response_assets_sealed":
            _require(value is False, f"claim overpromoted: {key}")
    copy = dict(receipt)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "receipt content hash changed")
    if recompute:
        _require(dict(receipt) == build_receipt(config), "receipt rebuild differs")


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


def write_receipt() -> str:
    config = load_config()
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(build_receipt(config)))


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    receipt = _read_json(path, "receipt")
    validate_receipt(config, receipt)
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
                    "response_asset_count": receipt["response_asset_count"],
                    "source_cell_count": len(receipt["holmberg_ii_source_cell_ids"]),
                    "velocity_or_dispersion_pixels_decoded": receipt[
                        "preparation_access_disclosure"
                    ]["velocity_or_dispersion_pixels_decoded"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
