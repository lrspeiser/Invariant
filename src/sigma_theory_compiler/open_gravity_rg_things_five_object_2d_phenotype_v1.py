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

CONFIG_PATH = Path("configs/open_gravity_rg_things_five_object_2d_phenotype_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_things_five_object_2d_phenotype_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_things_five_object_2d_phenotype_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-rg-things-five-object-2d-phenotype-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-things-five-object-2d-phenotype-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-things-five-object-2d-phenotype-receipt-1.0"
_CONFIG_RAW_SHA256 = "30b6af2d75628ca507abb7f68966b51c306712b2a11c65b3ecbc1ae9ed33df84"
_CONFIG_CONTENT_SHA256 = "e3e4bcde8bdcbf5857d18663f91c4ba1b199195dc144231021e8c7cc81c4640a"
_MODULE_SEMANTIC_SHA256 = "20cb10fb086b73c8c00479bb6ecfc3624e7be2fc0345dad374b4a28961a87e20"
_TEST_RAW_SHA256 = "c6552d697274a73d969be8fd91d7930ef3d95ef1e331341f13d3ab67059a4ad1"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")


class PhenotypeDiagnosticError(RuntimeError):
    """Raised when the frozen resolved-phenotype diagnostic fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhenotypeDiagnosticError(message)


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
        raise PhenotypeDiagnosticError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_POST_RESPONSE_EXPLORATORY_DIAGNOSTIC",
        "status changed",
    )
    _require(
        config["object_order"] == ["NGC2903", "NGC2976", "NGC3198", "NGC3521", "NGC4214"],
        "objects changed",
    )
    _require(
        config["response_axes"]
        == [
            "rg_fractional_improvement_over_newton",
            "rg_fractional_improvement_over_rar",
        ],
        "response axes changed",
    )
    statistics = config["statistical_contract"]
    _require(statistics["permutations_per_test"] == 120, "permutations changed")
    _require(statistics["feature_count"] == 11, "feature count changed")
    _require(statistics["response_axis_count"] == 2, "response count changed")
    _require(statistics["family_count"] == 22, "family count changed")
    for key in ("new_formula_fit", "classifier_fit", "threshold_tuning", "object_pruning"):
        _require(statistics[key] is False, f"forbidden fitting enabled: {key}")
    _require(len(config["continuous_feature_ids"]) == 11, "feature ledger changed")
    admission = config["admission_rule"]
    _require(len(admission["primary_measurement_papers"]) == 3, "paper gate changed")
    _require(
        len(admission["independent_known_answer_benchmarks"]) == 3,
        "benchmark gate changed",
    )
    _require(admission["general_3d_validation"] is False, "3D overclaim")
    next_test = config["next_test_contract"]
    _require(next_test["source_before_builder"] is True, "source-before-builder gate lost")
    _require(next_test["required_primary_papers"] is True, "paper gate lost")
    _require(
        next_test["required_independent_benchmarks"] is True,
        "benchmark gate lost",
    )
    _require(next_test["minimum_new_response_blind_objects"] == 5, "holdout size changed")
    _require(next_test["general_3d_claim_allowed"] is False, "next-test 3D overclaim")
    for value in config["access_scope"].values():
        _require(value == 0, "new access enabled")
    claims = config["claim_boundary"]
    _require(claims["post_response_exploratory_diagnostic_only"] is True, "scope changed")
    for key in (
        "mechanism_imprint_is_novel_evidence",
        "five_objects_establish_a_subclass",
        "source_feature_is_causal",
        "general_3d_validated",
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
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def _load_bindings(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    receipts = []
    for label in ("source_builder", "resolved_score"):
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
    _require(
        score["decision"]
        == "NGC2976_RG_SIGNAL_DID_NOT_GENERALIZE_TO_PREREGISTERED_FIVE_OBJECT_GATE",
        "resolved decision changed",
    )
    return source, score


def _feature_rows(
    config: Mapping[str, Any], source: Mapping[str, Any], score: Mapping[str, Any]
) -> list[dict[str, Any]]:
    source_by_id = {row["object_id"]: row for row in source["object_summaries"]}
    score_by_id = {row["object_id"]: row for row in score["objects"]}
    _require(set(source_by_id) == set(config["object_order"]), "source object set changed")
    _require(set(score_by_id) == set(config["object_order"]), "score object set changed")
    rows = []
    for object_id in config["object_order"]:
        source_row = source_by_id[object_id]
        score_row = score_by_id[object_id]
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
        responses = {axis: float(score_row[axis]) for axis in config["response_axes"]}
        newton = responses["rg_fractional_improvement_over_newton"]
        rar = responses["rg_fractional_improvement_over_rar"]
        if newton > 0.0 and rar > 0.0:
            outcome = "BEATS_BOTH"
        elif newton > 0.0:
            outcome = "BEATS_NEWTON_ONLY"
        elif rar > 0.0:
            outcome = "BEATS_RAR_ONLY"
        else:
            outcome = "BEATS_NEITHER"
        rows.append(
            {
                "object_id": object_id,
                "features": features,
                "responses": responses,
                "outcome_partition": outcome,
                "sealed_strict_object_gate": bool(score_row["rg_broader_signal_object_gate"]),
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
    results = []
    family_count = int(config["statistical_contract"]["family_count"])
    alpha = float(config["statistical_contract"]["familywise_alpha"])
    for response_axis in config["response_axes"]:
        response = [float(row["responses"][response_axis]) for row in rows]
        for feature_id in config["continuous_feature_ids"]:
            feature = [float(row["features"][feature_id]) for row in rows]
            rho, p_value = _exact_permutation_p(feature, response)
            adjusted = min(1.0, p_value * family_count)
            results.append(
                {
                    "response_axis": response_axis,
                    "feature_id": feature_id,
                    "spearman_rho": rho,
                    "exact_two_sided_p": p_value,
                    "bonferroni_p": adjusted,
                    "familywise_significant": adjusted <= alpha,
                }
            )
    return sorted(
        results, key=lambda row: (row["response_axis"], row["exact_two_sided_p"], row["feature_id"])
    )


def _without_content_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "content_sha256"}


def build_receipt() -> dict[str, Any]:
    config = load_config()
    source, score = _load_bindings(config)
    rows = _feature_rows(config, source, score)
    correlations = _correlations(config, rows)
    mechanism = next(
        row
        for row in correlations
        if row["feature_id"] == config["mechanism_diagnostic"]["feature_id"]
        and row["response_axis"] == config["mechanism_diagnostic"]["response_axis"]
    )
    outcome_counts = {
        label: sum(row["outcome_partition"] == label for row in rows)
        for label in ("BEATS_BOTH", "BEATS_NEWTON_ONLY", "BEATS_RAR_ONLY", "BEATS_NEITHER")
    }
    familywise = [row for row in correlations if row["familywise_significant"]]
    rar_familywise = [
        row for row in familywise if row["response_axis"] == "rg_fractional_improvement_over_rar"
    ]
    report: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "POST_RESPONSE_DENSITY_IMPRINT_NO_FAMILYWISE_RAR_DISCRIMINATOR",
        "decision": "MECHANISM_IMPRINT_DETECTED_SUBCLASS_NOT_IDENTIFIED_EXPAND_RESPONSE_BLIND_SAMPLE",
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
        "source_receipt_content_sha256": config["bindings"]["source_builder"][
            "receipt_content_sha256"
        ],
        "score_receipt_content_sha256": config["bindings"]["resolved_score"][
            "receipt_content_sha256"
        ],
        "object_feature_rows": rows,
        "outcome_counts": outcome_counts,
        "correlations": correlations,
        "familywise_significant_associations": familywise,
        "mechanism_density_imprint": {
            **mechanism,
            "interpretation": config["mechanism_diagnostic"]["interpretation_if_negative"],
            "exact_monotonic_rank_relation": math.isclose(
                mechanism["spearman_rho"], -1.0, rel_tol=0.0, abs_tol=1.0e-15
            ),
            "post_response": True,
        },
        "diagnostic_conclusions": {
            "rg_beats_newton_objects": [
                row["object_id"]
                for row in rows
                if row["responses"]["rg_fractional_improvement_over_newton"] > 0.0
            ],
            "rg_beats_rar_objects": [
                row["object_id"]
                for row in rows
                if row["responses"]["rg_fractional_improvement_over_rar"] > 0.0
            ],
            "rg_beats_both_objects": [
                row["object_id"] for row in rows if row["outcome_partition"] == "BEATS_BOTH"
            ],
            "density_dependent_gain_over_newton_is_visible": mechanism["spearman_rho"] < 0.0,
            "density_imprint_is_independent_rg_evidence": False,
            "source_feature_explains_rg_vs_rar_variation": bool(rar_familywise),
            "publishable_subclass_identified": False,
            "interpretation": "The fixed RG law adds acceleration in the density ordering its mechanism predicts, but the five-galaxy variation against RAR is not explained by any frozen scalar source descriptor. The remaining lead is spatial/radial field structure, not a fitted one-number phenotype.",
        },
        "next_test_contract": config["next_test_contract"],
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
    expected = build_receipt()
    _require(dict(receipt) == expected, "receipt differs from deterministic rebuild")


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
        print(json.dumps(receipt["outcome_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
