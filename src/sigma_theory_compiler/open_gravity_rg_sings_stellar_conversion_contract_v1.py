from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_rg_sings_stellar_conversion_contract_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_sings_stellar_conversion_contract_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_sings_stellar_conversion_contract_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-rg-sings-stellar-conversion-contract-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-sings-stellar-conversion-contract-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-sings-stellar-conversion-contract-receipt-1.0"
_CONFIG_RAW_SHA256 = "7ab1a435e03270db9098989da8692bacec18adb22428c926d854c1f81d9ce8dc"
_CONFIG_CONTENT_SHA256 = "b84c6847dfb96bf9954fff33d7c980e975d3019429a2931d8c6b1f5c7e5a379d"
_MODULE_SEMANTIC_SHA256 = "bf343fca09ffea6488058263c48e43d224294f72eedf2ffad91b4388bc754861"
_TEST_RAW_SHA256 = "c0be3d38a69a1c2410310acc94b16643ac3bbea5d3200f4b485dbc7b6e39ba7f"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")


class StellarConversionContractError(RuntimeError):
    """Raised when the frozen SINGS conversion contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StellarConversionContractError(message)


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StellarConversionContractError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_SOURCE_ONLY_THREE_CELL_STELLAR_CONVERSION_CONTRACT",
        "status changed",
    )
    _require(
        config["overlap_object_order"] == ["NGC2976", "NGC3198", "NGC3521"],
        "overlap objects changed",
    )
    _require(
        config["holdout_object_order"]
        == ["UGC04305", "NGC2841", "IC2574", "DDO154", "NGC5055", "NGC6946", "NGC7331"],
        "holdout objects changed",
    )
    _require(
        [cell["cell_id"] for cell in config["conversion_cells"]]
        == ["IRAC1_FIXED_ML0P6", "IRAC1_GLOBAL_COLOR_ML", "IRAC1_IRAC2_FASTICA36"],
        "conversion cells changed",
    )
    admission = config["admission"]
    _require(len(admission["primary_papers"]) == 4, "paper gate changed")
    _require(len(admission["independent_benchmarks"]) == 3, "benchmark gate changed")
    _require(admission["response_data_used"] is False, "response admission changed")
    score = config["response_score_contract"]
    _require(score["primary_cell"] is None, "source cell preselected")
    _require(score["all_three_cells_scored"] is True, "cell coverage reduced")
    _require(score["cell_selected_by_response"] is False, "response selection enabled")
    _require(score["failure_or_counterexample_prunes_object"] is False, "pruning enabled")
    _require(score["stable_sign_label_requires_all_three_cells"] is True, "stability gate changed")
    _require(score["minority_cells_retained"] is True, "minority cell lost")
    _require(score["new_rg_parameters_fitted"] is False, "RG tuning enabled")
    for value in config["access_scope"].values():
        _require(value == 0, "new access enabled")
    claims = config["claim_boundary"]
    _require(claims["source_conversion_uncertainty_frozen"] is True, "uncertainty gate lost")
    for key in (
        "any_conversion_is_s4g_equivalent",
        "fastica_is_published_pipeline_reproduction",
        "stellar_mass_is_directly_observed",
        "general_3d_validated",
        "gravity_result_established",
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
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def _load_bound_json(
    binding: Mapping[str, Any], path_key: str, raw_key: str, label: str
) -> dict[str, Any]:
    path = _repo_path(binding[path_key])
    _require(path.is_file(), f"{label} missing")
    _require(file_sha256(path) == binding[raw_key], f"{label} changed")
    value = _read_json(path, label)
    if "content_sha256" in binding:
        _require(value["content_sha256"] == binding["content_sha256"], f"{label} content changed")
    elif "result_content_sha256" in binding:
        _require(
            value["content_sha256"] == binding["result_content_sha256"],
            f"{label} content changed",
        )
    return value


def _load_bindings(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bindings = config["bindings"]
    acquisition = bindings["source_acquisition"]
    for role in ("config", "module", "test", "receipt"):
        path = _repo_path(acquisition[f"{role}_path"])
        _require(path.is_file(), f"source acquisition {role} missing")
        _require(
            file_sha256(path) == acquisition[f"{role}_raw_sha256"],
            f"source acquisition {role} changed",
        )
    acquisition_receipt = _read_json(
        _repo_path(acquisition["receipt_path"]), "source acquisition receipt"
    )
    _require(
        acquisition_receipt["content_sha256"] == acquisition["receipt_content_sha256"],
        "source acquisition content changed",
    )
    _require(
        acquisition_receipt["decision"]
        == "PASS_EXACT_SOURCE_BYTES_AND_FITS_SCHEMAS_READY_FOR_RESPONSE_BLIND_BUILDERS",
        "source acquisition no longer admitted",
    )
    irac2 = _load_bound_json(bindings["irac2_inventory"], "path", "raw_sha256", "IRAC2 inventory")
    outputs: dict[str, dict[str, Any]] = {
        "acquisition": acquisition_receipt,
        "irac2": irac2,
    }
    for key in ("raw_irac1_overlap", "global_color_overlap", "fastica_overlap"):
        binding = bindings[key]
        script = _repo_path(binding["script_path"])
        _require(script.is_file(), f"{key} script missing")
        _require(file_sha256(script) == binding["script_raw_sha256"], f"{key} script changed")
        outputs[key] = _load_bound_json(
            binding, "result_path", "result_raw_sha256", f"{key} result"
        )
    return outputs


def effective_ml(color_mag: float) -> float:
    value = 10.0 ** (-0.339 * color_mag - 0.336)
    _require(math.isfinite(value) and value > 0.0, "invalid effective M/L")
    return value


def irac_color(sum_f36: float, sum_f45: float) -> float:
    _require(sum_f36 > 0.0 and sum_f45 > 0.0, "invalid IRAC flux")
    return -2.5 * math.log10((sum_f36 / sum_f45) * (179.7 / 280.9))


def _ratio45_over36(color_mag: float) -> float:
    return (179.7 / 280.9) * 10.0 ** (0.4 * color_mag)


def reconstruct_two_component(
    star_f36: float, dust_f36: float, star_color: float, dust_color: float
) -> dict[str, float]:
    _require(star_f36 >= 0.0 and dust_f36 >= 0.0, "negative component")
    star_f45 = star_f36 * _ratio45_over36(star_color)
    dust_f45 = dust_f36 * _ratio45_over36(dust_color)
    return {
        "f36": star_f36 + dust_f36,
        "f45": star_f45 + dust_f45,
        "star_f36": star_f36,
        "dust_f36": dust_f36,
    }


def _core_raw_records(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for row in result["records"]:
        if row["wcs_mode"] == "CORE_TAN_PRIMARY":
            records[row["object_id"]] = row
    return records


def _overlap_records(
    config: Mapping[str, Any], bound: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    raw = _core_raw_records(bound["raw_irac1_overlap"])
    color = {row["object_id"]: row for row in bound["global_color_overlap"]["records"]}
    fastica = {row["object_id"]: row for row in bound["fastica_overlap"]["records"]}
    _require(set(raw) == set(config["overlap_object_order"]), "raw overlap set changed")
    _require(set(color) == set(config["overlap_object_order"]), "color overlap set changed")
    _require(set(fastica) == set(config["overlap_object_order"]), "FastICA overlap set changed")
    records = []
    for object_id in config["overlap_object_order"]:
        raw_row = raw[object_id]
        color_row = color[object_id]
        fastica_row = fastica[object_id]
        records.append(
            {
                "object_id": object_id,
                "raw_irac1": {
                    "integrated_over_s4g_stellar": float(
                        raw_row["sings_vs_s4g_stellar"][
                            "integrated_observed_per_reference_flux_ratio"
                        ]
                    ),
                    "integrated_over_s4g_stellar_plus_nonstellar": float(
                        raw_row["sings_vs_s4g_stellar_plus_nonstellar"][
                            "integrated_observed_per_reference_flux_ratio"
                        ]
                    ),
                },
                "fixed_ml0p6": {
                    "fractional_error_vs_clean_fixed_ml": float(
                        color_row["raw_fixed0p6_vs_clean_fixed0p6_fractional_error"]
                    )
                },
                "global_color_ml": {
                    "observed_color_mag": float(
                        color_row["observed_global_color_3p6_minus_4p5_mag"]
                    ),
                    "effective_ml": float(color_row["published_effective_ml"]),
                    "fractional_error_vs_clean_color_ml": float(
                        color_row["raw_global_color_vs_clean_color_fractional_error"]
                    ),
                },
                "fastica36": {
                    "stellar_integrated_over_s4g": float(
                        fastica_row["stellar_benchmark"]["integrated_candidate_over_reference"]
                    ),
                    "stellar_pearson_r": float(fastica_row["stellar_benchmark"]["pearson_r"]),
                    "stellar_spearman_r": float(fastica_row["stellar_benchmark"]["spearman_r"]),
                    "exact_reconstruction_max_abs_mjy_sr": float(
                        fastica_row["exact_reconstruction_max_abs_mjy_sr"]
                    ),
                },
            }
        )
    return records


def _without_content_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "content_sha256"}


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bound = _load_bindings(config)
    overlap = _overlap_records(config, bound)
    inventory = bound["irac2"]
    _require(inventory["file_count"] == 20, "IRAC2 file count changed")
    _require(inventory["object_count"] == 10, "IRAC2 object count changed")
    _require(inventory["production_fallback_objects"] == 7, "holdout count changed")
    inventory_objects = {row["object_id"] for row in inventory["records"]}
    _require(
        inventory_objects == set(config["overlap_object_order"] + config["holdout_object_order"]),
        "IRAC2 object inventory changed",
    )
    raw_nonstellar_errors = [
        abs(row["raw_irac1"]["integrated_over_s4g_stellar_plus_nonstellar"] - 1.0)
        for row in overlap
    ]
    fixed_errors = [
        abs(row["fixed_ml0p6"]["fractional_error_vs_clean_fixed_ml"]) for row in overlap
    ]
    color_errors = [
        abs(row["global_color_ml"]["fractional_error_vs_clean_color_ml"]) for row in overlap
    ]
    fastica_errors = [abs(row["fastica36"]["stellar_integrated_over_s4g"] - 1.0) for row in overlap]
    report: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "SOURCE_ONLY_CONVERSION_UNCERTAINTY_FROZEN_SEVEN_HOLDOUTS_READY_FOR_BUILD",
        "decision": "PASS_RETAIN_ALL_THREE_STELLAR_CONVERSION_CELLS_BEFORE_NEW_RESPONSE",
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
        "source_acquisition_receipt_content_sha256": config["bindings"]["source_acquisition"][
            "receipt_content_sha256"
        ],
        "irac2_inventory_content_sha256": config["bindings"]["irac2_inventory"]["content_sha256"],
        "overlap_records": overlap,
        "benchmark_summary": {
            "raw_irac1_max_abs_integrated_error_vs_stellar_plus_nonstellar": max(
                raw_nonstellar_errors
            ),
            "fixed_ml0p6_max_abs_mass_error_vs_clean_fixed_ml": max(fixed_errors),
            "global_color_ml_max_abs_mass_error_vs_clean_color_ml": max(color_errors),
            "fastica36_max_abs_integrated_stellar_error_vs_s4g": max(fastica_errors),
            "fastica36_min_stellar_pearson_r": min(
                row["fastica36"]["stellar_pearson_r"] for row in overlap
            ),
            "raw_irac1_tracks_stellar_plus_nonstellar_better_than_stellar_only": all(
                abs(row["raw_irac1"]["integrated_over_s4g_stellar_plus_nonstellar"] - 1.0)
                < abs(row["raw_irac1"]["integrated_over_s4g_stellar"] - 1.0)
                for row in overlap
            ),
            "single_conversion_is_s4g_equivalent": False,
        },
        "holdout_admission": {
            "object_order": config["holdout_object_order"],
            "object_count": len(config["holdout_object_order"]),
            "irac1_irac2_file_count": 28,
            "all_three_source_cells_required": True,
            "response_opened": False,
        },
        "response_score_contract": config["response_score_contract"],
        "interpretation": "Raw SINGS IRAC1 closely recovers S4G stellar-plus-nonstellar light, proving that nonstellar morphology is material. Global color corrects only total normalization, while two-band FastICA changes spatial structure but is not uniformly S4G-equivalent. All three cells must therefore be carried into the seven-object gravity test rather than selecting the cell that scores best.",
        "access_accounting": config["access_scope"],
        "claim_boundary": config["claim_boundary"],
    }
    report["content_sha256"] = content_sha256(report)
    return report


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    _require(type(receipt) is dict, "receipt must be an object")
    _require(receipt["schema"] == _RECEIPT_SCHEMA, "receipt schema changed")
    _require(
        receipt["content_sha256"] == content_sha256(_without_content_hash(receipt)),
        "receipt self-hash changed",
    )
    _require(dict(receipt) == build_receipt(), "receipt differs from deterministic rebuild")


def write_receipt() -> str:
    receipt = build_receipt()
    path = _repo_path(OUTPUT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(receipt) + b"\n"
    if path.exists():
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, path)
    except FileExistsError:
        _require(path.read_bytes() == payload, "receipt race differs")
        return "EXISTING_IDENTICAL"
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return "CREATED"


def check_receipt() -> str:
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    validate_receipt(_read_json(path, "receipt"))
    return "VALID"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        print(check_receipt())
    else:
        receipt = build_receipt()
        print(receipt["status"])
        print(receipt["decision"])
        print(json.dumps(receipt["benchmark_summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
