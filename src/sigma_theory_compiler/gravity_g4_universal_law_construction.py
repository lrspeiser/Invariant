"""Construct a compact universal galaxy relation without opening confirmation data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g0_experiment import score_predictions
from .gravity_g0_experiment import validate_receipt as validate_g0_receipt
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g3_meta_law import _fold_map, prepare_packets
from .gravity_g3_meta_law_v2 import load_config as load_g3_config
from .gravity_g3_meta_law_v2 import validate_receipt as validate_g3_receipt
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-g4-universal-construction-receipt-1.0"
CONFIG_SCHEMA = "invariant-gravity-g4-universal-construction-config-1.0"
CONFIG_PATH = "configs/gravity_g4_universal_law_construction.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g4_universal_law_construction.py"
TEST_PATH = "tests/test_gravity_g4_universal_law_construction.py"
OUTPUT_PATH = "runs/gravity/g4/universal-galaxy-law-construction-v1.json"

FEATURE_IDS = (
    "log_y",
    "log_r_over_disk_peak",
    "gas_fraction",
    "disk_fraction",
    "bulge_fraction",
    "baryon_log_slope",
)


class GravityG4ConstructionError(ValueError):
    """The G4 construction contract or evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load G4 construction and validate its G3 and G0 predecessors."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG4ConstructionError("G4 construction config schema changed")
    predecessor = config.get("predecessor_binding", {})
    predecessor_path = root / str(predecessor.get("path"))
    if _file_sha256(predecessor_path) != predecessor.get("file_sha256"):
        raise GravityG4ConstructionError("G4 G3 predecessor file changed")
    g3 = _load_json(predecessor_path)
    validate_g3_receipt(g3, root=root)
    if (
        g3.get("content_sha256") != predecessor.get("content_sha256")
        or g3.get("decision") != predecessor.get("required_decision")
        or g3.get("counts", {}).get("predicted_galaxies")
        != predecessor.get("required_galaxies")
    ):
        raise GravityG4ConstructionError("G4 G3 predecessor identity changed")
    g0_binding = config.get("g0_binding", {})
    g0_path = root / str(g0_binding.get("path"))
    if _file_sha256(g0_path) != g0_binding.get("file_sha256"):
        raise GravityG4ConstructionError("G4 G0 predecessor file changed")
    g0 = _load_json(g0_path)
    validate_g0_receipt(g0, root=root)
    if (
        g0.get("content_sha256") != g0_binding.get("content_sha256")
        or g0.get("decision") != g0_binding.get("required_decision")
    ):
        raise GravityG4ConstructionError("G4 G0 predecessor identity changed")
    family = config.get("formula_family", {})
    if tuple(family.get("allowed_dimensionless_local_features", ())) != FEATURE_IDS:
        raise GravityG4ConstructionError("G4 feature grammar changed")
    if family.get("term_count") != 27:
        raise GravityG4ConstructionError("G4 term count changed")
    if family.get("per_galaxy_gravitational_constants") != 0:
        raise GravityG4ConstructionError("G4 permits local gravitational constants")
    disclosure = config.get("diagnostic_disclosure", {})
    if (
        disclosure.get(
            "grammar_selected_after_inspecting_g3_feature_importance_and_compactness_diagnostics"
        )
        is not True
        or disclosure.get("result_is_independent_confirmation") is not False
    ):
        raise GravityG4ConstructionError("G4 diagnostic disclosure changed")
    if config.get("admission", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG4ConstructionError("G4 construction permits confirmation access")
    return config


def formula_terms() -> list[tuple[str, tuple[str, ...]]]:
    """Return the complete declared one-term grammar in canonical order."""

    terms = [(name, (name,)) for name in FEATURE_IDS]
    terms.extend(
        (f"{left}*{right}", (left, right))
        for left_index, left in enumerate(FEATURE_IDS)
        for right in FEATURE_IDS[left_index:]
    )
    if len(terms) != 27:
        raise GravityG4ConstructionError("G4 generated the wrong number of terms")
    return terms


def term_values(packet: Mapping[str, Any], factors: Sequence[str]) -> np.ndarray:
    """Evaluate a dimensionless local term without target access."""

    value = np.ones(packet["galaxy"].count, dtype=np.float64)
    for factor in factors:
        value = value * np.asarray(packet["features"][factor], dtype=np.float64)
    if np.any(~np.isfinite(value)):
        raise GravityG4ConstructionError("G4 term is non-finite")
    return value


def fit_coefficients(
    packets: Sequence[Mapping[str, Any]], factors: Sequence[str]
) -> np.ndarray:
    """Fit the two universal correction coefficients by the frozen linearization."""

    term = np.concatenate([term_values(packet, factors) for packet in packets])
    target = np.concatenate([packet["delta"] for packet in packets])
    radius = np.concatenate([packet["arrays"]["radius"] for packet in packets])
    observed = np.concatenate([packet["arrays"]["vobs"] for packet in packets])
    sigma = np.concatenate([packet["arrays"]["sigma"] for packet in packets])
    a0 = float(packets[0]["a0"])
    weight = (radius * a0 / (2.0 * observed * sigma)) ** 2
    design = np.column_stack([np.ones(len(term), dtype=np.float64), term])
    sqrt_weight = np.sqrt(weight)
    coefficients = np.linalg.lstsq(
        design * sqrt_weight[:, None],
        target * sqrt_weight,
        rcond=None,
    )[0]
    if coefficients.shape != (2,) or np.any(~np.isfinite(coefficients)):
        raise GravityG4ConstructionError("G4 coefficient fit failed")
    return coefficients


def predict(
    packet: Mapping[str, Any],
    factors: Sequence[str],
    coefficients: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Evaluate a universal formula on one galaxy."""

    delta = coefficients[0] + coefficients[1] * term_values(packet, factors)
    prediction2 = (
        packet["rar2"]
        + packet["arrays"]["radius"] * float(packet["a0"]) * delta
    )
    invalid = int(np.sum(~np.isfinite(prediction2) | (prediction2 <= 0)))
    prediction = np.sqrt(np.maximum(prediction2, np.finfo(np.float64).tiny))
    return prediction, invalid


def _stratum_assignments(
    packets: Sequence[Mapping[str, Any]], bins: int
) -> dict[str, dict[str, int]]:
    values = {
        "median_log_y": {
            packet["galaxy"].name: float(np.median(packet["features"]["log_y"]))
            for packet in packets
        },
        "median_gas_fraction": {
            packet["galaxy"].name: float(
                np.median(packet["features"]["gas_fraction"])
            )
            for packet in packets
        },
        "median_bulge_fraction": {
            packet["galaxy"].name: float(
                np.median(packet["features"]["bulge_fraction"])
            )
            for packet in packets
        },
    }
    assignments: dict[str, dict[str, int]] = {}
    for dimension, rows in values.items():
        ordered = sorted(rows, key=lambda name: (rows[name], name))
        assignments[dimension] = {
            name: min(bins - 1, rank * bins // len(ordered))
            for rank, name in enumerate(ordered)
        }
    return assignments


def build_receipt(root: Path, *, term_limit: int | None = None) -> dict[str, Any]:
    """Search the finite G4 construction grammar on exploration galaxies only."""

    root = root.resolve()
    config = load_config(root)
    g3_config = load_g3_config(root)
    packets = prepare_packets(root, g3_config)
    if len(packets) != int(config["predecessor_binding"]["required_galaxies"]):
        raise GravityG4ConstructionError("G4 exploration population changed")
    folds = int(config["whole_galaxy_cross_validation"]["folds"])
    assignments = _fold_map(
        [packet["galaxy"].name for packet in packets],
        str(config["whole_galaxy_cross_validation"]["salt"]),
        folds,
    )
    terms = formula_terms()
    if term_limit is not None:
        terms = terms[: max(0, min(term_limit, len(terms)))]
    shrinkages = [float(value) for value in config["formula_family"]["shrinkage_grid"]]
    cells = []
    for term_id, factors in terms:
        fold_coefficients = {}
        for fold in range(folds):
            training = [
                packet
                for packet in packets
                if assignments[packet["galaxy"].name] != fold
            ]
            fold_coefficients[fold] = fit_coefficients(training, factors)
        for shrinkage in shrinkages:
            chi_square = 0.0
            invalid = 0
            for packet in packets:
                fold = assignments[packet["galaxy"].name]
                coefficients = fold_coefficients[fold] * shrinkage
                prediction, bad = predict(packet, factors, coefficients)
                invalid += bad
                chi_square += float(
                    score_predictions(
                        prediction,
                        packet["arrays"]["vobs"],
                        packet["arrays"]["sigma"],
                    )["chi_square"]
                )
            cells.append(
                {
                    "chi_square": _metric(chi_square),
                    "invalid_prediction2": invalid,
                    "shrinkage": _metric(shrinkage),
                    "term_degree": len(factors),
                    "term_id": term_id,
                }
            )
    eligible = [cell for cell in cells if cell["invalid_prediction2"] == 0]
    selected = min(
        eligible,
        key=lambda cell: (
            float(cell["chi_square"]),
            cell["term_degree"],
            cell["term_id"],
            float(cell["shrinkage"]),
        ),
        default=None,
    )
    selected_factors = (
        () if selected is None else next(row[1] for row in terms if row[0] == selected["term_id"])
    )
    per_galaxy = []
    selected_fold_coefficients = {}
    if selected is not None:
        shrinkage = float(selected["shrinkage"])
        for fold in range(folds):
            training = [
                packet
                for packet in packets
                if assignments[packet["galaxy"].name] != fold
            ]
            selected_fold_coefficients[fold] = (
                fit_coefficients(training, selected_factors) * shrinkage
            )
        for packet in sorted(packets, key=lambda row: row["galaxy"].name):
            fold = assignments[packet["galaxy"].name]
            prediction, bad = predict(
                packet,
                selected_factors,
                selected_fold_coefficients[fold],
            )
            candidate_score = score_predictions(
                prediction,
                packet["arrays"]["vobs"],
                packet["arrays"]["sigma"],
            )
            rar_score = score_predictions(
                np.sqrt(packet["rar2"]),
                packet["arrays"]["vobs"],
                packet["arrays"]["sigma"],
            )
            per_galaxy.append(
                {
                    "candidate_prediction_sha256": canonical_sha256(
                        [format(float(value), ".15e") for value in prediction]
                    ),
                    "candidate_score": candidate_score,
                    "fold": fold,
                    "galaxy": packet["galaxy"].name,
                    "invalid_prediction2": bad,
                    "point_count": packet["galaxy"].count,
                    "rar_score": rar_score,
                }
            )
    candidate_chi = sum(
        float(row["candidate_score"]["chi_square"]) for row in per_galaxy
    )
    rar_chi = sum(float(row["rar_score"]["chi_square"]) for row in per_galaxy)
    g0 = _load_json(root / str(config["g0_binding"]["path"]))
    baseline_aggregate = g0["baseline_replay"]["aggregate"]
    nfw_chi = float(baseline_aggregate["nfw_halo_ceiling"]["chi_square"])
    newtonian_chi = float(baseline_aggregate["newtonian_baryons"]["chi_square"])
    point_count = sum(row["point_count"] for row in per_galaxy)
    assignments_by_stratum = _stratum_assignments(
        packets, int(config["stratification"]["bins_per_dimension"])
    )
    strata = []
    by_name = {row["galaxy"]: row for row in per_galaxy}
    for dimension, bins_by_name in assignments_by_stratum.items():
        for bin_id in range(int(config["stratification"]["bins_per_dimension"])):
            names = sorted(name for name, value in bins_by_name.items() if value == bin_id)
            stratum_candidate = sum(
                float(by_name[name]["candidate_score"]["chi_square"]) for name in names
            )
            stratum_rar = sum(
                float(by_name[name]["rar_score"]["chi_square"]) for name in names
            )
            strata.append(
                {
                    "bin": bin_id,
                    "candidate_chi_square": _metric(stratum_candidate),
                    "dimension": dimension,
                    "fractional_gain_over_rar": _metric(
                        1.0 - stratum_candidate / stratum_rar
                    ),
                    "galaxies": len(names),
                    "rar_chi_square": _metric(stratum_rar),
                }
            )
    admission = config["admission"]
    minimum_gain = float(admission["minimum_fractional_chi_square_gain_over_empirical_rar"])
    maximum_stratum_regression = float(
        admission["maximum_fractional_chi_square_regression_vs_rar_in_any_stratum"]
    )
    nfw_limit = nfw_chi + float(
        admission["nfw_ceiling_slack_chi_square_per_point"]
    ) * point_count
    full_run = term_limit is None and len(terms) == int(config["formula_family"]["term_count"])
    gate_checks = {
        "all_139_exploration_galaxies_predicted_once": len(per_galaxy) == 139,
        "all_predictions_positive_and_finite": all(
            row["invalid_prediction2"] == 0 for row in per_galaxy
        ),
        "beats_newtonian_baryons": candidate_chi < newtonian_chi,
        "beats_rar_by_minimum_fraction": (
            rar_chi > 0 and 1.0 - candidate_chi / rar_chi >= minimum_gain
        ),
        "compact_family_fully_searched": full_run,
        "no_stratum_regresses_beyond_limit": all(
            float(row["fractional_gain_over_rar"]) >= -maximum_stratum_regression
            for row in strata
        ),
        "per_galaxy_gravitational_constants_zero": (
            config["formula_family"]["per_galaxy_gravitational_constants"] == 0
        ),
        "within_nfw_performance_ceiling": candidate_chi <= nfw_limit,
    }
    passed = all(gate_checks.values())
    final_coefficients = None
    if selected is not None:
        final_coefficients = fit_coefficients(packets, selected_factors) * float(
            selected["shrinkage"]
        )
    origin_status = (
        "KNOWN_FAMILY"
        if selected_factors and set(selected_factors) == {"log_y"}
        else "COMBINATION"
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G4_CONSTRUCTION",
        "decision": (
            "PASS_G4_EXPLORATION_FREEZE" if passed else "BLOCK_G4_EXPLORATION_CONSTRUCTION"
        ),
        "claims": {
            "alternative_to_gr_discovered": False,
            "compact_universal_empirical_relation_constructed": selected is not None,
            "confirmation_authorized": passed,
            "confirmation_galaxy_evaluated": False,
            "historical_novelty_established": False,
            "independent_confirmation_completed": False,
            "zero_local_gravitational_constants": True,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "candidate_formula_cells": len(cells),
            "candidate_galaxy_evaluations": len(cells) * len(packets),
            "candidate_point_evaluations": len(cells)
            * sum(packet["galaxy"].count for packet in packets),
            "confirmation_evaluator_accesses": 0,
            "exploration_galaxies": len(per_galaxy),
            "exploration_points": point_count,
            "formula_terms": len(terms),
            "shrinkages_per_term": len(shrinkages),
        },
        "diagnostic_disclosure": config["diagnostic_disclosure"],
        "family_result": {
            "all_cells": sorted(
                cells,
                key=lambda row: (
                    float(row["chi_square"]),
                    row["term_degree"],
                    row["term_id"],
                    float(row["shrinkage"]),
                ),
            ),
            "exhausted_without_full_gate_survivor": not passed and full_run,
        },
        "gate_checks": gate_checks,
        "galaxies": per_galaxy,
        "predecessors": {
            "g0": config["g0_binding"],
            "g3": config["predecessor_binding"],
        },
        "scores": {
            "candidate_chi_square": _metric(candidate_chi),
            "empirical_rar_chi_square": _metric(rar_chi),
            "fractional_gain_over_empirical_rar": _metric(
                1.0 - candidate_chi / rar_chi
            ),
            "newtonian_baryons_chi_square": _metric(newtonian_chi),
            "nfw_ceiling_chi_square": _metric(nfw_chi),
            "nfw_ceiling_limit_with_slack": _metric(nfw_limit),
            "nfw_ceiling_excess": _metric(candidate_chi - nfw_limit),
        },
        "selected_formula": None
        if selected is None
        else {
            "authoritative_origin_status": origin_status,
            "c0": _metric(float(final_coefficients[0])),
            "c1": _metric(float(final_coefficients[1])),
            "equation": "V^2=V_RAR^2+r*g_dagger*(C0+C1*term)",
            "factors": list(selected_factors),
            "per_galaxy_gravitational_constants": 0,
            "proposer_origin_label": (
                "known_family_instance"
                if origin_status == "KNOWN_FAMILY"
                else "new_combination_of_known_ideas"
            ),
            "selection_chi_square": selected["chi_square"],
            "shrinkage": selected["shrinkage"],
            "term_id": selected["term_id"],
            "universal_correction_constants": 2,
        },
        "strata": strata,
        "limitations": [
            "The grammar and shrinkage range were chosen after G3 diagnostics on the same exploration population; cross-validation scores are model-development evidence.",
            "The base is the known empirical RAR relation, so an acceleration-only winner is a known-family recalibration rather than a new gravity theory.",
            "SPARC random-error chi-square omits correlated inclination and distance systematics.",
            "A blocked NFW ceiling keeps confirmation locked even if the compact relation improves RAR.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Validate a checked G4 construction receipt and its fail-closed claims."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG4ConstructionError("G4 receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG4ConstructionError("G4 receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG4ConstructionError("G4 receipt config binding changed")
    for key, path in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG4ConstructionError(f"G4 {key} binding changed")
    counts = receipt.get("counts", {})
    claims = receipt.get("claims", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG4ConstructionError("G4 construction records confirmation access")
    if claims.get("historical_novelty_established") is not False:
        raise GravityG4ConstructionError("G4 construction overstates novelty")
    if claims.get("independent_confirmation_completed") is not False:
        raise GravityG4ConstructionError("G4 construction overstates confirmation")
    passed = receipt.get("decision") == "PASS_G4_EXPLORATION_FREEZE"
    if passed and (
        not all(receipt.get("gate_checks", {}).values())
        or claims.get("confirmation_authorized") is not True
    ):
        raise GravityG4ConstructionError("G4 PASS is unsupported")
    if not passed and claims.get("confirmation_authorized") is not False:
        raise GravityG4ConstructionError("blocked G4 authorizes confirmation")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG4ConstructionError(
                f"refusing to overwrite immutable G4 receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--term-limit", type=int)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        validate_receipt(_load_json(root / OUTPUT_PATH), root=root)
        return 0
    receipt = build_receipt(root, term_limit=args.term_limit)
    if args.term_limit is None:
        _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "decision": receipt["decision"],
                "selected_formula": receipt["selected_formula"],
                "scores": receipt["scores"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"] == "PASS_G4_EXPLORATION_FREEZE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG4ConstructionError",
    "build_receipt",
    "fit_coefficients",
    "formula_terms",
    "load_config",
    "predict",
    "term_values",
    "validate_receipt",
]
