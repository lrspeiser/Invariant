"""Test a finite nonlocal radial-profile grammar on real SPARC exploration data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g0_experiment import score_predictions
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g3_meta_law import _fold_map
from .gravity_g4_photometric_law_construction import (
    prepare_photometric_packets,
)
from .gravity_g4_photometric_law_construction import (
    validate_receipt as validate_predecessor_receipt,
)
from .gravity_g4_universal_law_construction import _stratum_assignments
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-g4-nonlocal-profile-receipt-3.0"
CONFIG_SCHEMA = "invariant-gravity-g4-nonlocal-profile-config-3.0"
CONFIG_PATH = "configs/gravity_g4_nonlocal_profile_law_construction.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/gravity_g4_nonlocal_profile_law_construction.py"
)
TEST_PATH = "tests/test_gravity_g4_nonlocal_profile_law_construction.py"
OUTPUT_PATH = "runs/gravity/g4/universal-galaxy-law-construction-v3-nonlocal-profile.json"

SOURCE_FIELDS = (
    "log_y",
    "log1p_sb_total",
    "gas_fraction",
    "bulge_fraction",
    "baryon_log_slope",
    "sb_log_slope",
)
KERNEL_SHAPES = (
    "symmetric_exponential",
    "symmetric_gaussian",
    "symmetric_cauchy",
    "interior_exponential",
    "exterior_exponential",
)
LOG_RADIUS_SCALES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
REPRESENTATIONS = ("weighted_mean", "mean_minus_local")


class GravityG4NonlocalError(ValueError):
    """The G4 nonlocal contract, computation, or evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the contract and validate every predecessor and data binding."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG4NonlocalError("G4 nonlocal config schema changed")
    predecessor = config.get("predecessor_binding", {})
    predecessor_path = root / str(predecessor.get("path"))
    if _file_sha256(predecessor_path) != predecessor.get("file_sha256"):
        raise GravityG4NonlocalError("G4 nonlocal predecessor file changed")
    prior = _load_json(predecessor_path)
    validate_predecessor_receipt(prior, root=root)
    if (
        prior.get("content_sha256") != predecessor.get("content_sha256")
        or prior.get("decision") != predecessor.get("required_decision")
    ):
        raise GravityG4NonlocalError("G4 nonlocal predecessor changed")
    surface = config.get("surface_brightness_binding", {})
    if _file_sha256(root / str(surface.get("path"))) != surface.get("file_sha256"):
        raise GravityG4NonlocalError("G4 nonlocal surface-brightness file changed")
    for evidence in config.get("prior_nonlocal_evidence", ()):
        path = root / str(evidence.get("path"))
        if _file_sha256(path) != evidence.get("file_sha256"):
            raise GravityG4NonlocalError("G4 prior nonlocal evidence changed")
        if _load_json(path).get("content_sha256") != evidence.get("content_sha256"):
            raise GravityG4NonlocalError("G4 prior nonlocal content changed")
    family = config.get("formula_family", {})
    if (
        tuple(family.get("source_fields", ())) != SOURCE_FIELDS
        or tuple(family.get("kernel_shapes", ())) != KERNEL_SHAPES
        or tuple(float(value) for value in family.get("log_radius_scales", ()))
        != LOG_RADIUS_SCALES
        or tuple(family.get("representations", ())) != REPRESENTATIONS
        or family.get("feature_count") != 360
        or family.get("per_galaxy_gravitational_constants") != 0
    ):
        raise GravityG4NonlocalError("G4 nonlocal grammar changed")
    cascade = family.get("cascade", {})
    if (
        cascade.get("top_univariate_features_retained") != 24
        or cascade.get("formula_structures") != 636
    ):
        raise GravityG4NonlocalError("G4 nonlocal cascade changed")
    if config.get("admission", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG4NonlocalError("G4 nonlocal contract permits confirmation")
    if config.get("origin_assessment", {}).get("historical_novelty_claimed") is not False:
        raise GravityG4NonlocalError("G4 nonlocal contract overstates novelty")
    return config


def feature_specs() -> list[dict[str, Any]]:
    """Enumerate all 360 typed nonlocal features in stable ordinal order."""

    specs = []
    for source in SOURCE_FIELDS:
        for kernel in KERNEL_SHAPES:
            for scale in LOG_RADIUS_SCALES:
                for representation in REPRESENTATIONS:
                    scale_id = format(scale, "g").replace(".", "p")
                    specs.append(
                        {
                            "feature_id": (
                                f"{source}__{kernel}__ell_{scale_id}__{representation}"
                            ),
                            "kernel": kernel,
                            "log_radius_scale": scale,
                            "representation": representation,
                            "source": source,
                        }
                    )
    if len(specs) != 360 or len({row["feature_id"] for row in specs}) != 360:
        raise GravityG4NonlocalError("G4 nonlocal feature enumeration failed")
    return specs


def _log_radius_cell_widths(log_radius: np.ndarray) -> np.ndarray:
    if len(log_radius) < 2 or np.any(np.diff(log_radius) <= 0):
        raise GravityG4NonlocalError("G4 nonlocal radii are not strictly increasing")
    edges = np.empty(len(log_radius) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (log_radius[:-1] + log_radius[1:])
    edges[0] = log_radius[0] - 0.5 * (log_radius[1] - log_radius[0])
    edges[-1] = log_radius[-1] + 0.5 * (log_radius[-1] - log_radius[-2])
    widths = np.diff(edges)
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise GravityG4NonlocalError("G4 nonlocal quadrature is invalid")
    return widths


def _kernel_matrix(
    log_radius: np.ndarray, widths: np.ndarray, kernel: str, scale: float
) -> np.ndarray:
    separation = log_radius[None, :] - log_radius[:, None]
    normalized = separation / scale
    if kernel == "symmetric_exponential":
        values = np.exp(-np.abs(normalized))
    elif kernel == "symmetric_gaussian":
        values = np.exp(-0.5 * normalized**2)
    elif kernel == "symmetric_cauchy":
        values = 1.0 / (1.0 + normalized**2)
    elif kernel == "interior_exponential":
        values = np.exp(-np.abs(normalized)) * (separation <= 0)
    elif kernel == "exterior_exponential":
        values = np.exp(-np.abs(normalized)) * (separation >= 0)
    else:
        raise GravityG4NonlocalError(f"unknown G4 nonlocal kernel: {kernel}")
    weighted = values * widths[None, :]
    denominator = np.sum(weighted, axis=1)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise GravityG4NonlocalError("G4 nonlocal kernel normalization failed")
    return weighted / denominator[:, None]


def materialize_nonlocal_features(packet: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compute the feature bank from radii and baryonic inputs, never targets."""

    specs = feature_specs()
    radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
    log_radius = np.log(radius)
    widths = _log_radius_cell_widths(log_radius)
    matrices = {
        (kernel, scale): _kernel_matrix(log_radius, widths, kernel, scale)
        for kernel in KERNEL_SHAPES
        for scale in LOG_RADIUS_SCALES
    }
    features: dict[str, np.ndarray] = {}
    for spec in specs:
        local = np.asarray(packet["features"][spec["source"]], dtype=np.float64)
        mean = matrices[(spec["kernel"], spec["log_radius_scale"])] @ local
        value = mean if spec["representation"] == "weighted_mean" else mean - local
        if np.any(~np.isfinite(value)):
            raise GravityG4NonlocalError("G4 nonlocal feature is non-finite")
        features[spec["feature_id"]] = value
    return features


def prepare_nonlocal_packets(root: Path) -> list[dict[str, Any]]:
    """Materialize target-blind nonlocal features inside each exploration galaxy."""

    packets = sorted(
        prepare_photometric_packets(root), key=lambda packet: packet["galaxy"].name
    )
    enriched = []
    for packet in packets:
        row = dict(packet)
        row["nonlocal_features"] = materialize_nonlocal_features(packet)
        enriched.append(row)
    return enriched


def _flatten(
    packets: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
) -> dict[str, Any]:
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for packet in packets:
        count = packet["galaxy"].count
        slices[packet["galaxy"].name] = (offset, offset + count)
        offset += count
    radius = np.concatenate([packet["arrays"]["radius"] for packet in packets])
    observed = np.concatenate([packet["arrays"]["vobs"] for packet in packets])
    sigma = np.concatenate([packet["arrays"]["sigma"] for packet in packets])
    a0 = float(packets[0]["a0"])
    return {
        "a0": a0,
        "delta": np.concatenate([packet["delta"] for packet in packets]),
        "fold": np.concatenate(
            [
                np.full(packet["galaxy"].count, assignments[packet["galaxy"].name])
                for packet in packets
            ]
        ),
        "observed": observed,
        "radius_a0": radius * a0,
        "rar2": np.concatenate([packet["rar2"] for packet in packets]),
        "sigma": sigma,
        "slices": slices,
        "weight": (radius * a0 / (2.0 * observed * sigma)) ** 2,
        "x": np.column_stack(
            [
                np.concatenate(
                    [packet["nonlocal_features"][spec["feature_id"]] for packet in packets]
                )
                for spec in specs
            ]
        ),
    }


def _fit_coefficients(
    flat: Mapping[str, Any], indices: Sequence[int], training: np.ndarray
) -> np.ndarray:
    x = np.asarray(flat["x"])[training][:, indices]
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    sqrt_weight = np.sqrt(np.asarray(flat["weight"])[training])
    coefficients = np.linalg.lstsq(
        design * sqrt_weight[:, None],
        np.asarray(flat["delta"])[training] * sqrt_weight,
        rcond=None,
    )[0]
    if np.any(~np.isfinite(coefficients)):
        raise GravityG4NonlocalError("G4 nonlocal coefficient fit failed")
    return coefficients


def _oof_delta(
    flat: Mapping[str, Any], indices: Sequence[int], folds: int
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    fold_ids = np.asarray(flat["fold"])
    x = np.asarray(flat["x"])[:, indices]
    prediction = np.empty(len(fold_ids), dtype=np.float64)
    coefficients = {}
    for fold in range(folds):
        training = fold_ids != fold
        testing = ~training
        coefficients[fold] = _fit_coefficients(flat, indices, training)
        design = np.column_stack(
            [np.ones(int(np.sum(testing)), dtype=np.float64), x[testing]]
        )
        prediction[testing] = design @ coefficients[fold]
    return prediction, coefficients


def _score_structure(
    flat: Mapping[str, Any],
    indices: Sequence[int],
    feature_ids: Sequence[str],
    folds: int,
    shrinkages: np.ndarray,
) -> list[dict[str, Any]]:
    delta, _coefficients = _oof_delta(flat, indices, folds)
    prediction2 = np.asarray(flat["rar2"])[None, :] + (
        np.asarray(flat["radius_a0"])[None, :]
        * shrinkages[:, None]
        * delta[None, :]
    )
    invalid = np.sum(~np.isfinite(prediction2) | (prediction2 <= 0), axis=1)
    velocity = np.sqrt(np.maximum(prediction2, np.finfo(np.float64).tiny))
    chi_square = np.sum(
        ((velocity - np.asarray(flat["observed"])[None, :]) / np.asarray(flat["sigma"])[None, :])
        ** 2,
        axis=1,
    )
    return [
        {
            "chi_square": _metric(float(chi_square[index])),
            "feature_ids": list(feature_ids),
            "invalid_prediction2": int(invalid[index]),
            "shrinkage": _metric(float(shrinkage)),
            "universal_correction_constants": len(indices) + 1,
        }
        for index, shrinkage in enumerate(shrinkages)
    ]


def _cell_key(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(cell["chi_square"]),
        cell["universal_correction_constants"],
        tuple(cell["feature_ids"]),
        float(cell["shrinkage"]),
    )


def build_receipt(root: Path, *, feature_limit: int | None = None) -> dict[str, Any]:
    """Exhaust the declared nonlocal cascade without opening confirmation."""

    root = root.resolve()
    config = load_config(root)
    packets = prepare_nonlocal_packets(root)
    all_specs = feature_specs()
    specs = (
        all_specs
        if feature_limit is None
        else all_specs[: max(0, min(feature_limit, len(all_specs)))]
    )
    folds = int(config["whole_galaxy_cross_validation"]["folds"])
    assignments = _fold_map(
        [packet["galaxy"].name for packet in packets],
        str(config["whole_galaxy_cross_validation"]["salt"]),
        folds,
    )
    flat = _flatten(packets, specs, assignments)
    shrinkages = np.asarray(
        [float(value) for value in config["formula_family"]["shrinkage_grid"]],
        dtype=np.float64,
    )
    cells = []
    univariate_best = []
    for index, spec in enumerate(specs):
        rows = _score_structure(
            flat, (index,), (spec["feature_id"],), folds, shrinkages
        )
        cells.extend(rows)
        univariate_best.append(min(rows, key=_cell_key))
    retained_count = min(
        int(
            config["formula_family"]["cascade"][
                "top_univariate_features_retained"
            ]
        ),
        len(specs),
    )
    retained_ids = [
        row["feature_ids"][0]
        for row in sorted(univariate_best, key=_cell_key)[:retained_count]
    ]
    index_by_id = {spec["feature_id"]: index for index, spec in enumerate(specs)}
    for left, right in combinations(retained_ids, 2):
        cells.extend(
            _score_structure(
                flat,
                (index_by_id[left], index_by_id[right]),
                (left, right),
                folds,
                shrinkages,
            )
        )
    eligible = [row for row in cells if row["invalid_prediction2"] == 0]
    selected = min(eligible, key=_cell_key, default=None)
    per_galaxy = []
    final_coefficients = None
    direct_refit_chi_square = None
    selected_fold_coefficients: dict[int, np.ndarray] = {}
    if selected is not None:
        selected_indices = tuple(index_by_id[value] for value in selected["feature_ids"])
        delta, selected_fold_coefficients = _oof_delta(flat, selected_indices, folds)
        shrinkage = float(selected["shrinkage"])
        prediction2 = np.asarray(flat["rar2"]) + np.asarray(flat["radius_a0"]) * (
            shrinkage * delta
        )
        prediction = np.sqrt(np.maximum(prediction2, np.finfo(np.float64).tiny))
        for packet in packets:
            name = packet["galaxy"].name
            start, stop = flat["slices"][name]
            candidate = prediction[start:stop]
            bad = int(
                np.sum(~np.isfinite(prediction2[start:stop]) | (prediction2[start:stop] <= 0))
            )
            per_galaxy.append(
                {
                    "candidate_prediction_sha256": canonical_sha256(
                        [format(float(value), ".15e") for value in candidate]
                    ),
                    "candidate_score": score_predictions(
                        candidate,
                        packet["arrays"]["vobs"],
                        packet["arrays"]["sigma"],
                    ),
                    "fold": assignments[name],
                    "galaxy": name,
                    "invalid_prediction2": bad,
                    "point_count": packet["galaxy"].count,
                    "rar_score": score_predictions(
                        np.sqrt(packet["rar2"]),
                        packet["arrays"]["vobs"],
                        packet["arrays"]["sigma"],
                    ),
                }
            )
        training = np.ones(len(flat["fold"]), dtype=bool)
        final_coefficients = (
            _fit_coefficients(flat, selected_indices, training) * shrinkage
        )
        direct_design = np.column_stack(
            [
                np.ones(len(flat["fold"]), dtype=np.float64),
                np.asarray(flat["x"])[:, selected_indices],
            ]
        )
        direct2 = np.asarray(flat["rar2"]) + np.asarray(flat["radius_a0"]) * (
            direct_design @ final_coefficients
        )
        direct_prediction = np.sqrt(np.maximum(direct2, np.finfo(np.float64).tiny))
        direct_refit_chi_square = float(
            np.sum(
                (
                    (direct_prediction - np.asarray(flat["observed"]))
                    / np.asarray(flat["sigma"])
                )
                ** 2
            )
        )
    candidate_chi = sum(
        float(row["candidate_score"]["chi_square"]) for row in per_galaxy
    )
    rar_chi = sum(float(row["rar_score"]["chi_square"]) for row in per_galaxy)
    predecessor = _load_json(root / str(config["predecessor_binding"]["path"]))
    nfw_chi = float(predecessor["scores"]["nfw_ceiling_chi_square"])
    newtonian_chi = float(predecessor["scores"]["newtonian_baryons_chi_square"])
    point_count = sum(row["point_count"] for row in per_galaxy)
    by_name = {row["galaxy"]: row for row in per_galaxy}
    strata = []
    stratum_map = _stratum_assignments(
        packets, int(config["stratification"]["bins_per_dimension"])
    )
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
    full_run = feature_limit is None and len(specs) == 360
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
        "full_nonlocal_cascade_searched": full_run,
        "no_stratum_regresses_beyond_limit": all(
            float(row["fractional_gain_over_rar"])
            >= -float(
                admission[
                    "maximum_fractional_chi_square_regression_vs_rar_in_any_stratum"
                ]
            )
            for row in strata
        ),
        "per_galaxy_gravitational_constants_zero": (
            config["formula_family"]["per_galaxy_gravitational_constants"] == 0
        ),
        "within_nfw_performance_ceiling": candidate_chi <= nfw_limit,
    }
    passed = all(gate_checks.values())
    spec_by_id = {row["feature_id"]: row for row in specs}
    structures = len(specs) + retained_count * (retained_count - 1) // 2
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G4_NONLOCAL_PROFILE_CONSTRUCTION",
        "decision": (
            "PASS_G4_NONLOCAL_PROFILE_EXPLORATION_FREEZE"
            if passed
            else "BLOCK_G4_NONLOCAL_PROFILE_CONSTRUCTION"
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
            "candidate_point_evaluations": len(cells) * point_count,
            "confirmation_evaluator_accesses": 0,
            "exploration_galaxies": len(per_galaxy),
            "exploration_points": point_count,
            "formula_structures": structures,
            "nonlocal_features": len(specs),
            "pair_structures": retained_count * (retained_count - 1) // 2,
            "retained_univariate_features": retained_count,
            "shrinkages_per_structure": len(shrinkages),
        },
        "diagnostic_disclosure": config["diagnostic_disclosure"],
        "family_result": {
            "all_cells": sorted(cells, key=_cell_key),
            "exhausted_without_full_gate_survivor": not passed and full_run,
            "retained_univariate_feature_ids": retained_ids,
        },
        "gate_checks": gate_checks,
        "galaxies": per_galaxy,
        "origin_assessment": config["origin_assessment"],
        "predecessor": config["predecessor_binding"],
        "prior_nonlocal_evidence": config["prior_nonlocal_evidence"],
        "scores": {
            "candidate_chi_square": _metric(candidate_chi),
            "direct_all_exploration_refit_chi_square": (
                None
                if direct_refit_chi_square is None
                else _metric(direct_refit_chi_square)
            ),
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
            "authoritative_origin_status": "COMBINATION",
            "equation": "V^2=V_RAR^2+r*g_dagger*(C0+sum_m C_m*Phi_m(r))",
            "features": [
                {
                    **spec_by_id[value],
                    "log_radius_scale": _metric(
                        float(spec_by_id[value]["log_radius_scale"])
                    ),
                }
                for value in selected["feature_ids"]
            ],
            "final_universal_coefficients": [
                _metric(float(value)) for value in final_coefficients
            ],
            "fold_universal_coefficients_before_shrinkage": {
                str(fold): [_metric(float(value)) for value in coefficients]
                for fold, coefficients in selected_fold_coefficients.items()
            },
            "per_galaxy_gravitational_constants": 0,
            "proposer_origin_label": "new_combination_of_known_ideas",
            "selection_chi_square": selected["chi_square"],
            "shrinkage": selected["shrinkage"],
            "universal_correction_constants": selected[
                "universal_correction_constants"
            ],
        },
        "strata": strata,
        "limitations": [
            "This scale-free radial operator is a phenomenological grammar, not a derived field equation.",
            "The grammar was designed after inspecting earlier failures on the same exploration population.",
            "A normalized one-dimensional log-radius kernel is not yet a three-dimensional causal Green function.",
            "COMBINATION means a new tested mix of known ideas, not historical novelty.",
            "A blocked NFW ceiling keeps every confirmation target locked.",
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
        raise GravityG4NonlocalError("G4 nonlocal receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG4NonlocalError("G4 nonlocal receipt seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG4NonlocalError("G4 nonlocal config binding changed")
    paths = {
        "config": CONFIG_PATH,
        "source": SOURCE_PATH,
        "surface_brightness": str(config["surface_brightness_binding"]["path"]),
        "test": TEST_PATH,
    }
    for key, path in paths.items():
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG4NonlocalError(f"G4 nonlocal {key} binding changed")
    counts = receipt.get("counts", {})
    claims = receipt.get("claims", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG4NonlocalError("G4 nonlocal run records confirmation access")
    if claims.get("historical_novelty_established") is not False:
        raise GravityG4NonlocalError("G4 nonlocal run overstates novelty")
    if claims.get("independent_confirmation_completed") is not False:
        raise GravityG4NonlocalError("G4 nonlocal run overstates confirmation")
    passed = receipt.get("decision") == "PASS_G4_NONLOCAL_PROFILE_EXPLORATION_FREEZE"
    if passed and (
        not all(receipt.get("gate_checks", {}).values())
        or claims.get("confirmation_authorized") is not True
    ):
        raise GravityG4NonlocalError("G4 nonlocal PASS is unsupported")
    if not passed and claims.get("confirmation_authorized") is not False:
        raise GravityG4NonlocalError("blocked G4 nonlocal run authorizes confirmation")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG4NonlocalError(
                f"refusing to overwrite immutable G4 nonlocal receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--feature-limit", type=int)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        validate_receipt(_load_json(root / OUTPUT_PATH), root=root)
        return 0
    receipt = build_receipt(root, feature_limit=args.feature_limit)
    if args.feature_limit is None:
        _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "counts": receipt["counts"],
                "decision": receipt["decision"],
                "scores": receipt["scores"],
                "selected_formula": receipt["selected_formula"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"] == "PASS_G4_NONLOCAL_PROFILE_EXPLORATION_FREEZE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OUTPUT_PATH",
    "GravityG4NonlocalError",
    "build_receipt",
    "feature_specs",
    "load_config",
    "materialize_nonlocal_features",
    "prepare_nonlocal_packets",
    "validate_receipt",
]
