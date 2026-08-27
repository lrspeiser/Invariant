"""Repair G4 construction with published SPARC surface-brightness profiles."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g0_experiment import score_predictions
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g3_meta_law import _fold_map, prepare_packets
from .gravity_g3_meta_law_v2 import load_config as load_g3_config
from .gravity_g4_universal_law_construction import (
    _stratum_assignments,
    fit_coefficients,
    predict,
)
from .gravity_g4_universal_law_construction import (
    validate_receipt as validate_g4_v1_receipt,
)
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_surface_brightness import load_asset

SCHEMA = "invariant-gravity-g4-photometric-construction-receipt-2.0"
CONFIG_SCHEMA = "invariant-gravity-g4-photometric-construction-config-2.0"
CONFIG_PATH = "configs/gravity_g4_photometric_law_construction.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g4_photometric_law_construction.py"
TEST_PATH = "tests/test_gravity_g4_photometric_law_construction.py"
OUTPUT_PATH = "runs/gravity/g4/universal-galaxy-law-construction-v2-photometric.json"

FEATURE_IDS = (
    "log_y",
    "log_r_over_disk_peak",
    "gas_fraction",
    "disk_fraction",
    "bulge_fraction",
    "baryon_log_slope",
    "log1p_sb_disk",
    "log1p_sb_bulge",
    "log1p_sb_total",
    "sb_disk_fraction",
    "sb_bulge_fraction",
    "sb_log_slope",
)


class GravityG4PhotometricError(ValueError):
    """The G4 photometric repair or its evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the repair and validate its blocked predecessor and photometry."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG4PhotometricError("G4 photometric config schema changed")
    predecessor = config.get("predecessor_binding", {})
    predecessor_path = root / str(predecessor.get("path"))
    if _file_sha256(predecessor_path) != predecessor.get("file_sha256"):
        raise GravityG4PhotometricError("G4 photometric predecessor file changed")
    prior = _load_json(predecessor_path)
    validate_g4_v1_receipt(prior, root=root)
    if (
        prior.get("content_sha256") != predecessor.get("content_sha256")
        or prior.get("decision") != predecessor.get("required_decision")
    ):
        raise GravityG4PhotometricError("G4 photometric predecessor changed")
    binding = config.get("surface_brightness_binding", {})
    asset_path = root / str(binding.get("path"))
    if _file_sha256(asset_path) != binding.get("file_sha256"):
        raise GravityG4PhotometricError("G4 surface-brightness file changed")
    asset = load_asset(root)
    if (
        asset.get("content_sha256") != binding.get("content_sha256")
        or asset.get("counts", {}).get("exploration_galaxies")
        != binding.get("required_galaxies")
        or asset.get("counts", {}).get("exploration_rows")
        != binding.get("required_rows")
        or asset.get("counts", {}).get("confirmation_rows")
        != binding.get("required_confirmation_rows")
    ):
        raise GravityG4PhotometricError("G4 surface-brightness identity changed")
    family = config.get("formula_family", {})
    if tuple(family.get("allowed_dimensionless_local_features", ())) != FEATURE_IDS:
        raise GravityG4PhotometricError("G4 photometric feature grammar changed")
    if family.get("term_count") != 90:
        raise GravityG4PhotometricError("G4 photometric term count changed")
    if family.get("per_galaxy_gravitational_constants") != 0:
        raise GravityG4PhotometricError("G4 photometric grammar permits local constants")
    disclosure = config.get("diagnostic_disclosure", {})
    if (
        disclosure.get("photometric_grammar_selected_after_inspecting_same_exploration_data")
        is not True
        or disclosure.get("result_is_independent_confirmation") is not False
    ):
        raise GravityG4PhotometricError("G4 photometric disclosure changed")
    if config.get("admission", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG4PhotometricError("G4 photometric repair permits confirmation")
    return config


def formula_terms() -> list[tuple[str, tuple[str, ...]]]:
    terms = [(name, (name,)) for name in FEATURE_IDS]
    terms.extend(
        (f"{left}*{right}", (left, right))
        for left_index, left in enumerate(FEATURE_IDS)
        for right in FEATURE_IDS[left_index:]
    )
    if len(terms) != 90:
        raise GravityG4PhotometricError("wrong G4 photometric term count")
    return terms


def prepare_photometric_packets(root: Path) -> list[dict[str, Any]]:
    """Join target-blind photometry to exploration packets by sealed galaxy name."""

    packets = prepare_packets(root, load_g3_config(root))
    asset = load_asset(root)
    by_name = {row["galaxy"]: row for row in asset["galaxies"]}
    enriched = []
    for packet in packets:
        name = packet["galaxy"].name
        source = by_name[name]
        brightness = np.asarray(source["rows"], dtype=np.float64)
        if brightness.shape != (packet["galaxy"].count, 2):
            raise GravityG4PhotometricError(f"photometry row mismatch for {name}")
        disk = brightness[:, 0]
        bulge = brightness[:, 1]
        total = disk + bulge
        denominator = np.where(total > 0, total, 1.0)
        log_total = np.log1p(total)
        radius = packet["arrays"]["radius"]
        additions = {
            "log1p_sb_disk": np.log1p(disk),
            "log1p_sb_bulge": np.log1p(bulge),
            "log1p_sb_total": log_total,
            "sb_disk_fraction": np.where(total > 0, disk / denominator, 0.0),
            "sb_bulge_fraction": np.where(total > 0, bulge / denominator, 0.0),
            "sb_log_slope": np.gradient(log_total, np.log(radius)),
        }
        if any(np.any(~np.isfinite(value)) for value in additions.values()):
            raise GravityG4PhotometricError(f"invalid photometric feature for {name}")
        row = dict(packet)
        row["features"] = {**packet["features"], **additions}
        enriched.append(row)
    return enriched


def build_receipt(root: Path, *, term_limit: int | None = None) -> dict[str, Any]:
    """Exhaust the compact photometric family without opening confirmation."""

    root = root.resolve()
    config = load_config(root)
    packets = prepare_photometric_packets(root)
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
        fold_coefficients = {
            fold: fit_coefficients(
                [
                    packet
                    for packet in packets
                    if assignments[packet["galaxy"].name] != fold
                ],
                factors,
            )
            for fold in range(folds)
        }
        for shrinkage in shrinkages:
            chi_square = 0.0
            invalid = 0
            for packet in packets:
                coefficients = (
                    fold_coefficients[assignments[packet["galaxy"].name]] * shrinkage
                )
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
    factors = (
        () if selected is None else next(row[1] for row in terms if row[0] == selected["term_id"])
    )
    per_galaxy = []
    if selected is not None:
        shrinkage = float(selected["shrinkage"])
        fold_coefficients = {
            fold: fit_coefficients(
                [
                    packet
                    for packet in packets
                    if assignments[packet["galaxy"].name] != fold
                ],
                factors,
            )
            * shrinkage
            for fold in range(folds)
        }
        for packet in sorted(packets, key=lambda row: row["galaxy"].name):
            fold = assignments[packet["galaxy"].name]
            prediction, bad = predict(packet, factors, fold_coefficients[fold])
            per_galaxy.append(
                {
                    "candidate_prediction_sha256": canonical_sha256(
                        [format(float(value), ".15e") for value in prediction]
                    ),
                    "candidate_score": score_predictions(
                        prediction,
                        packet["arrays"]["vobs"],
                        packet["arrays"]["sigma"],
                    ),
                    "fold": fold,
                    "galaxy": packet["galaxy"].name,
                    "invalid_prediction2": bad,
                    "point_count": packet["galaxy"].count,
                    "rar_score": score_predictions(
                        np.sqrt(packet["rar2"]),
                        packet["arrays"]["vobs"],
                        packet["arrays"]["sigma"],
                    ),
                }
            )
    candidate_chi = sum(
        float(row["candidate_score"]["chi_square"]) for row in per_galaxy
    )
    rar_chi = sum(float(row["rar_score"]["chi_square"]) for row in per_galaxy)
    prior = _load_json(root / str(config["predecessor_binding"]["path"]))
    nfw_chi = float(prior["scores"]["nfw_ceiling_chi_square"])
    newtonian_chi = float(prior["scores"]["newtonian_baryons_chi_square"])
    point_count = sum(row["point_count"] for row in per_galaxy)
    stratum_map = _stratum_assignments(
        packets, int(config["stratification"]["bins_per_dimension"])
    )
    by_name = {row["galaxy"]: row for row in per_galaxy}
    strata = []
    for dimension, bins_by_name in stratum_map.items():
        for bin_id in range(int(config["stratification"]["bins_per_dimension"])):
            names = sorted(name for name, value in bins_by_name.items() if value == bin_id)
            candidate = sum(
                float(by_name[name]["candidate_score"]["chi_square"]) for name in names
            )
            rar = sum(float(by_name[name]["rar_score"]["chi_square"]) for name in names)
            strata.append(
                {
                    "bin": bin_id,
                    "candidate_chi_square": _metric(candidate),
                    "dimension": dimension,
                    "fractional_gain_over_rar": _metric(1.0 - candidate / rar),
                    "galaxies": len(names),
                    "rar_chi_square": _metric(rar),
                }
            )
    admission = config["admission"]
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
            rar_chi > 0
            and 1.0 - candidate_chi / rar_chi
            >= float(admission["minimum_fractional_chi_square_gain_over_empirical_rar"])
        ),
        "compact_photometric_family_fully_searched": full_run,
        "no_stratum_regresses_beyond_limit": all(
            float(row["fractional_gain_over_rar"])
            >= -float(
                admission["maximum_fractional_chi_square_regression_vs_rar_in_any_stratum"]
            )
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
        final_coefficients = fit_coefficients(packets, factors) * float(
            selected["shrinkage"]
        )
    uses_photometry = any(factor.startswith("sb_") or "sb_" in factor for factor in factors)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G4_CONSTRUCTION_REPAIR",
        "decision": (
            "PASS_G4_PHOTOMETRIC_EXPLORATION_FREEZE"
            if passed
            else "BLOCK_G4_PHOTOMETRIC_CONSTRUCTION"
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
            "photometric_features": 6,
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
        "predecessor": config["predecessor_binding"],
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
            "authoritative_origin_status": (
                "COMBINATION" if uses_photometry else "KNOWN_FAMILY"
            ),
            "c0": _metric(float(final_coefficients[0])),
            "c1": _metric(float(final_coefficients[1])),
            "equation": "V^2=V_RAR^2+r*g_dagger*(C0+C1*term)",
            "factors": list(factors),
            "per_galaxy_gravitational_constants": 0,
            "proposer_origin_label": (
                "new_combination_of_known_ideas"
                if uses_photometry
                else "known_family_instance"
            ),
            "selection_chi_square": selected["chi_square"],
            "shrinkage": selected["shrinkage"],
            "term_id": selected["term_id"],
            "universal_correction_constants": 2,
        },
        "strata": strata,
        "surface_brightness": config["surface_brightness_binding"],
        "limitations": [
            "This photometric grammar was designed after inspecting the same exploration data and is not independent validation.",
            "Surface brightness is a real missing baryonic observable, but a phenomenological correlation is not a gravity derivation.",
            "A COMBINATION label means a mix of known inputs, not historical novelty.",
            "A blocked NFW ceiling keeps all confirmation targets locked.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "source": _binding(root, SOURCE_PATH),
            "surface_brightness": _binding(
                root, str(config["surface_brightness_binding"]["path"])
            ),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG4PhotometricError("G4 photometric receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG4PhotometricError("G4 photometric receipt seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG4PhotometricError("G4 photometric config binding changed")
    paths = {
        "config": CONFIG_PATH,
        "source": SOURCE_PATH,
        "surface_brightness": str(config["surface_brightness_binding"]["path"]),
        "test": TEST_PATH,
    }
    for key, path in paths.items():
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG4PhotometricError(f"G4 photometric {key} binding changed")
    counts = receipt.get("counts", {})
    claims = receipt.get("claims", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG4PhotometricError("G4 photometric run records confirmation access")
    if claims.get("historical_novelty_established") is not False:
        raise GravityG4PhotometricError("G4 photometric run overstates novelty")
    if claims.get("independent_confirmation_completed") is not False:
        raise GravityG4PhotometricError("G4 photometric run overstates confirmation")
    passed = receipt.get("decision") == "PASS_G4_PHOTOMETRIC_EXPLORATION_FREEZE"
    if passed and (
        not all(receipt.get("gate_checks", {}).values())
        or claims.get("confirmation_authorized") is not True
    ):
        raise GravityG4PhotometricError("G4 photometric PASS is unsupported")
    if not passed and claims.get("confirmation_authorized") is not False:
        raise GravityG4PhotometricError("blocked photometric G4 authorizes confirmation")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG4PhotometricError(
                f"refusing to overwrite immutable G4 photometric receipt: {path}"
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
                "scores": receipt["scores"],
                "selected_formula": receipt["selected_formula"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"] == "PASS_G4_PHOTOMETRIC_EXPLORATION_FREEZE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG4PhotometricError",
    "build_receipt",
    "formula_terms",
    "load_config",
    "prepare_photometric_packets",
    "validate_receipt",
]
