from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_rg_five_galaxy_morphology_diagnostic_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_five_galaxy_morphology_diagnostic_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_five_galaxy_morphology_diagnostic_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-rg-five-galaxy-morphology-diagnostic-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-five-galaxy-morphology-diagnostic-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-five-galaxy-morphology-diagnostic-receipt-1.0"
_CONFIG_RAW_SHA256 = "f7a46ec3032f50016900fd7191569572c76df4407924477fc42c801861ebd7d1"
_CONFIG_CONTENT_SHA256 = "2accd660b4f599b26743e4022da2009154e34b05f54815ae27663fe77b4f3585"
_MODULE_SEMANTIC_SHA256 = "62f8df9e4f030c4bd875676df25a5cadc2c4af27273e651c33984c97fbab405a"
_TEST_RAW_SHA256 = "ab570ccc73566165bcf616d3c7f504ae2385a51431d4746feff25fe9241fb591"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")


class MorphologyDiagnosticError(RuntimeError):
    """Raised when the frozen morphology diagnostic fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MorphologyDiagnosticError(message)


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
        raise MorphologyDiagnosticError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["status"] == "FROZEN_POST_HOC_HYPOTHESIS_GENERATOR", "status changed")
    _require(
        config["object_order"] == ["NGC2903", "NGC2976", "NGC3198", "NGC3521", "NGC4214"],
        "objects changed",
    )
    statistics = config["statistical_contract"]
    _require(statistics["permutations"] == 120, "permutation contract changed")
    _require(statistics["feature_count"] == 11, "feature count changed")
    _require(statistics["new_formula_fit"] is False, "formula fitting enabled")
    _require(statistics["threshold_tuning"] is False, "threshold tuning enabled")
    _require(len(config["continuous_feature_ids"]) == 11, "feature ledger changed")
    pair = config["matched_pair_contract"]
    _require(pair["anchor_object"] == "NGC2976", "anchor changed")
    _require(
        pair["pair_selected_only_for_post_hoc_hypothesis_generation"] is True,
        "pair overclaim",
    )
    next_test = config["next_test_contract"]
    _require(
        next_test["missing_velocity_field_disposition"] == "SOURCE_BLOCKED", "source gate lost"
    )
    _require(len(next_test["independent_benchmarks"]) == 3, "benchmark gate changed")
    _require(next_test["general_3d_claim_allowed"] is False, "3D overclaim")
    for value in config["access_scope"].values():
        _require(value == 0, "new access enabled")
    claims = config["claim_boundary"]
    _require(claims["post_hoc_hypothesis_generation_only"] is True, "scope changed")
    for key in (
        "five_objects_can_establish_morphology_law",
        "source_feature_association_is_causal",
        "matched_pair_is_confirmation",
        "new_theory_supported",
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


def _load_bindings(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    receipts = []
    for label in ("source_builder", "fixed_score"):
        binding = config["bindings"][label]
        for role in ("config", "module", "test", "receipt"):
            path = _repo_path(binding[f"{role}_path"])
            _require(path.is_file(), f"{label} {role} missing")
            _require(
                file_sha256(path) == binding[f"{role}_raw_sha256"],
                f"{label} {role} changed",
            )
        receipt = _read_json(_repo_path(binding["receipt_path"]), f"{label} receipt")
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            f"{label} receipt content changed",
        )
        receipts.append(receipt)
    source, score = receipts
    _require(source["claims"]["scientific_response_scored"] is False, "source response leak")
    _require(score["adjudication"]["family_eliminated"] is False, "score overclaim")
    return source, score


def _feature_rows(
    config: Mapping[str, Any], source: Mapping[str, Any], score: Mapping[str, Any]
) -> list[dict[str, Any]]:
    improvements = {
        row["object_id"]: float(row["fractional_improvement"])
        for row in score["adjudication"]["object_comparisons"]
    }
    source_by_id = {row["object_id"]: row for row in source["object_summaries"]}
    rows = []
    for object_id in config["object_order"]:
        source_row = source_by_id[object_id]
        primary = source_row["primary_summary"]
        matched = primary["matched_acceleration"]
        gas = float(primary["co_helium_mass_msun"]) + float(primary["hi_helium_mass_msun"])
        stellar = float(primary["stellar_mass_msun"])
        features = {
            "log10_stellar_mass_msun": math.log10(stellar),
            "gas_mass_fraction": gas / (gas + stellar),
            "log10_rhalf_pc": math.log10(float(source_row["rhalf_pc"])),
            "log10_rho_midplane_msun_pc3": math.log10(float(matched["rho_midplane_msun_pc3"])),
            "log10_sigma_b_msun_pc2": math.log10(float(matched["sigma_b_msun_pc2"])),
            "log10_potential_depth_c2": math.log10(float(matched["potential_depth_c2"])),
            "log10_g_b_m_s2": math.log10(float(matched["g_b_m_s2"])),
            "radial_force_rms_asymmetry": float(matched["radial_force_rms_asymmetry"]),
            "radial_force_fourier_m1_m4": float(matched["radial_force_fourier_m1_m4"]),
            "inclination_deg": float(source_row["central_geometry"]["inclination_deg"]),
            "ellipticity": float(source_row["central_geometry"]["ellipticity"]),
        }
        _require(list(features) == config["continuous_feature_ids"], "feature order changed")
        rows.append(
            {
                "object_id": object_id,
                "orientation_flag": source_row["orientation_flag"],
                "fixed_rg_fractional_improvement": improvements[object_id],
                "features": features,
            }
        )
    return rows


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    _require(math.isfinite(value), "invalid Spearman statistic")
    return value


def _exact_permutation_p(
    feature: Sequence[float], response: Sequence[float]
) -> tuple[float, float]:
    observed = _spearman(feature, response)
    exceed = 0
    total = 0
    for permutation in itertools.permutations(response):
        total += 1
        if abs(_spearman(feature, permutation)) + 1.0e-15 >= abs(observed):
            exceed += 1
    _require(total == 120, "permutation count changed")
    return observed, exceed / total


def _correlations(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    response = [float(row["fixed_rg_fractional_improvement"]) for row in rows]
    results = []
    feature_count = int(config["statistical_contract"]["feature_count"])
    alpha = float(config["statistical_contract"]["familywise_alpha"])
    for feature_id in config["continuous_feature_ids"]:
        feature = [float(row["features"][feature_id]) for row in rows]
        rho, p_value = _exact_permutation_p(feature, response)
        adjusted = min(1.0, p_value * feature_count)
        results.append(
            {
                "feature_id": feature_id,
                "spearman_rho": rho,
                "exact_two_sided_p": p_value,
                "bonferroni_p": adjusted,
                "familywise_significant": adjusted < alpha,
            }
        )
    return sorted(results, key=lambda row: (-abs(float(row["spearman_rho"])), row["feature_id"]))


def _matched_pair(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    feature_ids = config["matched_pair_contract"]["source_only_distance_features"]
    matrix = np.asarray([[float(row["features"][key]) for key in feature_ids] for row in rows])
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    _require(bool(np.all(scale > 0.0)), "degenerate pair feature")
    standardized = (matrix - mean) / scale
    anchor_id = config["matched_pair_contract"]["anchor_object"]
    anchor_index = next(index for index, row in enumerate(rows) if row["object_id"] == anchor_id)
    distances = np.linalg.norm(standardized - standardized[anchor_index], axis=1)
    distances[anchor_index] = np.inf
    neighbor_index = int(np.argmin(distances))
    anchor = rows[anchor_index]
    neighbor = rows[neighbor_index]
    return {
        "anchor_object": anchor_id,
        "source_nearest_neighbor": neighbor["object_id"],
        "standardized_source_distance": float(distances[neighbor_index]),
        "anchor_fractional_improvement": anchor["fixed_rg_fractional_improvement"],
        "neighbor_fractional_improvement": neighbor["fixed_rg_fractional_improvement"],
        "opposite_support_direction": (float(anchor["fixed_rg_fractional_improvement"]) > 0.0)
        != (float(neighbor["fixed_rg_fractional_improvement"]) > 0.0),
        "interpretation": "A nearest source-feature neighbor with the opposite response direction would rule out simple compactness/density/potential depth as a sufficient explanation, while leaving orientation, gas structure, response systematics, and richer field morphology open.",
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    source, score = _load_bindings(config)
    rows = _feature_rows(config, source, score)
    correlations = _correlations(config, rows)
    pair = _matched_pair(config, rows)
    significant = [row["feature_id"] for row in correlations if row["familywise_significant"]]
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "POST_HOC_SOURCE_DIAGNOSTIC_NO_FAMILYWISE_FEATURE_SIGNAL",
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "source_receipt_content_sha256": source["content_sha256"],
        "score_receipt_content_sha256": score["content_sha256"],
        "object_feature_rows": rows,
        "correlations": correlations,
        "familywise_significant_features": significant,
        "matched_pair": pair,
        "diagnostic_conclusions": {
            "single_source_feature_explains_fixed_rg_outcome": bool(significant),
            "simple_density_potential_compactness_is_sufficient": False,
            "orientation_or_gas_structure_is_established": False,
            "resolved_2d_velocity_test_is_warranted": True,
            "reason": "The five-object sample has no multiplicity-corrected association, and the source-nearest compact/shallow pair has opposite RG response directions.",
        },
        "next_test_contract": config["next_test_contract"],
        "access_accounting": config["access_scope"],
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
                    "strongest_feature": receipt["correlations"][0]["feature_id"],
                    "strongest_rho": receipt["correlations"][0]["spearman_rho"],
                    "familywise_significant_features": receipt["familywise_significant_features"],
                    "source_nearest_pair": [
                        receipt["matched_pair"]["anchor_object"],
                        receipt["matched_pair"]["source_nearest_neighbor"],
                    ],
                    "opposite_pair_direction": receipt["matched_pair"][
                        "opposite_support_direction"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
