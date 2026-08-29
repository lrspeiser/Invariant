"""Item 44 dimensionless scale-hierarchy search across S4TM and CLASH."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item43_cosmological_boundary import (
    _feature_arrays as _s4tm_feature_arrays,
    _read_vizier_rows,
    expansion_ratio,
)

CONFIG_PATH = Path("configs/gravity_item44_scale_hierarchy_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")


class GravityItem44Error(RuntimeError):
    """Raised when an Item 44 hierarchy, exposure, or evaluation gate fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != "invariant-gravity-item44-scale-hierarchy-config-1.0"
        or int(config.get("item", -1)) != 44
    ):
        raise GravityItem44Error("unexpected Item 44 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem44Error("stable gravity goal changed")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem44Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-43-cosmological-boundary-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem44Error("Item 43 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem44Error("Item 43 decision changed")
    policy = load_counterexample_policy(root / POLICY_PATH)
    discovery = config["discovery_policy"]
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem44Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem44Error("count-only rejection entered Item 44")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem44Error("finite empirical family pruning entered Item 44")
    if policy["empirical_evidence"]["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem44Error("executable counterexample policy changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem44Error("raw candidate count changed")
    if int(generator["cells_per_niche"]) != 65536:
        raise GravityItem44Error("niche capacity changed")
    if list(generator["grid_shape_per_niche"]) != [16, 16, 16, 16]:
        raise GravityItem44Error("candidate grid changed")
    if len(generator["niches"]) != 4 or int(generator["post_evaluation_cells"]) != 0:
        raise GravityItem44Error("hierarchy niche boundary changed")
    if not bool(config["scope"]["all_empirical_responses_already_exposed"]):
        raise GravityItem44Error("Item 44 exposure disclosure changed")
    if bool(config["scope"]["fresh_confirmation_claim_allowed"]):
        raise GravityItem44Error("fresh confirmation entered retrospective Item 44")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem44Error("paid calls entered Item 44")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    cells = int(config["candidate_generator"]["cells_per_niche"])
    candidate_id = np.arange(total, dtype=np.int64)
    lane = (candidate_id // cells).astype(np.int8)
    local = candidate_id % cells
    return {
        "candidate_id": candidate_id,
        "lane": lane,
        "amplitude_index": ((local // 4096) % 16).astype(np.int8),
        "exponent_index": ((local // 256) % 16).astype(np.int8),
        "transition_index": ((local // 16) % 16).astype(np.int8),
        "shape_index": (local % 16).astype(np.int8),
    }


def _candidate_parameters(
    candidates: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grids = config["candidate_generator"]["parameter_grids"]
    return (
        np.asarray(grids["amplitude"])[np.asarray(candidates["amplitude_index"], int)],
        np.asarray(grids["acceleration_exponent"])[np.asarray(candidates["exponent_index"], int)],
        np.asarray(grids["transition_u"])[np.asarray(candidates["transition_index"], int)],
        np.asarray(grids["hierarchy_shape"])[np.asarray(candidates["shape_index"], int)],
    )


def hierarchy_coordinates(
    radius_kpc: np.ndarray,
    baryonic_size_kpc: np.ndarray,
    enclosed_mass_msun: np.ndarray,
    redshift: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    radius = np.asarray(radius_kpc, dtype=np.float64)
    size = np.asarray(baryonic_size_kpc, dtype=np.float64)
    mass = np.asarray(enclosed_mass_msun, dtype=np.float64)
    z = np.asarray(redshift, dtype=np.float64)
    if np.any(radius <= 0.0) or np.any(size <= 0.0) or np.any(mass <= 0.0):
        raise GravityItem44Error("hierarchy scales must be positive")
    constants = config["constants"]
    cosmology = config["fiducial_cosmology"]
    g = float(constants["gravitational_constant_kpc_km2_s2_msun"])
    c = float(cosmology["speed_of_light_km_s"])
    a0_kpc = float(constants["acceleration_scale_m_s2"]) * float(
        constants["kpc_to_km"]
    ) / 1000.0
    transition = np.sqrt(g * mass / a0_kpc)
    schwarzschild = 2.0 * g * mass / (c * c)
    curvature = np.sqrt(np.power(radius, 3.0) / schwarzschild)
    acceleration_wavelength = np.full_like(radius, c * c / a0_kpc)
    e = expansion_ratio(z, {"fiducial_cosmology": cosmology})
    horizon = c / (float(cosmology["hubble_constant_km_s_mpc"]) * e) * 1000.0
    ratios = np.stack(
        (
            size / transition,
            radius / transition,
            curvature / np.sqrt(acceleration_wavelength * horizon),
            (size / transition)
            * (curvature / acceleration_wavelength)
            / (radius / horizon),
        ),
        axis=0,
    )
    coordinate = 1.0 / (1.0 + np.abs(np.log10(np.maximum(ratios, 1e-300))))
    return coordinate, {
        "transition_length_kpc": transition,
        "schwarzschild_length_kpc": schwarzschild,
        "curvature_radius_kpc": curvature,
        "acceleration_wavelength_kpc": acceleration_wavelength,
        "horizon_radius_kpc": horizon,
        "raw_ratios": ratios,
    }


def candidate_multiplier(
    candidates: Mapping[str, np.ndarray],
    u: np.ndarray,
    hierarchy: np.ndarray,
    config: Mapping[str, Any],
    *,
    scale_free: bool = False,
) -> np.ndarray:
    amplitude, exponent, transition, shape = _candidate_parameters(candidates, config)
    lane = np.asarray(candidates["lane"], dtype=int)
    h = np.ones((len(lane), len(u)), dtype=np.float64) if scale_free else hierarchy[lane]
    response = 0.05 + 0.95 * np.power(h, shape[:, None])
    return 1.0 + amplitude[:, None] * np.power(
        np.asarray(u)[None, :], -exponent[:, None]
    ) / (1.0 + np.asarray(u)[None, :] / transition[:, None]) * response


def admissible_candidates(
    config: Mapping[str, Any], *, batch_size: int = 4096
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    gate = config["candidate_generator"]["admissibility"]
    u = np.logspace(
        float(gate["probe_log10_u_min"]),
        float(gate["probe_log10_u_max"]),
        int(gate["probe_points"]),
    )
    h_values = np.asarray(gate["probe_hierarchy_coordinates"], dtype=np.float64)
    keep_parts: list[np.ndarray] = []
    signatures: set[bytes] = set()
    rejection: Counter[str] = Counter()
    for begin in range(0, len(raw["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(raw["candidate_id"]))
        rows = {key: value[begin:end] for key, value in raw.items()}
        amplitude, exponent, transition, shape = _candidate_parameters(rows, config)
        response = 0.05 + 0.95 * np.power(
            h_values[None, :, None], shape[:, None, None]
        )
        multiplier = 1.0 + amplitude[:, None, None] * np.power(
            u[None, None, :], -exponent[:, None, None]
        ) / (1.0 + u[None, None, :] / transition[:, None, None]) * response
        finite = np.all(np.isfinite(multiplier), axis=(1, 2))
        bounded = finite & np.all(multiplier >= float(gate["minimum_multiplier"]), axis=(1, 2))
        bounded &= np.all(multiplier <= float(gate["maximum_multiplier"]), axis=(1, 2))
        local = bounded & (
            np.max(np.log10(multiplier[:, :, -1]), axis=1)
            <= float(gate["maximum_high_acceleration_log10_deviation"])
        )
        material = local & (
            np.min(multiplier[:, :, 0], axis=1)
            >= float(gate["minimum_low_acceleration_multiplier"])
        )
        span = np.ptp(np.log10(multiplier), axis=1)
        hierarchy_material = material & (
            np.max(span, axis=1) >= float(gate["minimum_hierarchy_log10_effect_span"])
        )
        monotone = hierarchy_material & np.all(
            np.diff(multiplier, axis=2) <= float(gate["monotone_nonincreasing_tolerance"]),
            axis=(1, 2),
        )
        rejection["nonfinite"] += int(np.sum(~finite))
        rejection["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection["no_local_limit"] += int(np.sum(bounded & ~local))
        rejection["immaterial_low_acceleration"] += int(np.sum(local & ~material))
        rejection["hierarchy_immaterial"] += int(np.sum(material & ~hierarchy_material))
        rejection["nonmonotone"] += int(np.sum(hierarchy_material & ~monotone))
        local_keep = np.flatnonzero(monotone)
        keep_parts.append(local_keep + begin)
        signature = np.round(
            np.log10(multiplier[local_keep]), int(gate["behavior_signature_decimals"])
        )
        for row in signature:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    admitted = {key: value[keep] for key, value in raw.items()}
    return admitted, {
        "raw_candidates": len(raw["candidate_id"]),
        "admitted_candidates": len(keep),
        "rejected_candidates": len(raw["candidate_id"]) - len(keep),
        "admitted_by_lane": {
            str(lane): int(np.sum(admitted["lane"] == lane)) for lane in range(4)
        },
        "behavioral_equivalence_classes": len(signatures),
        "rejection_counts_nonexclusive": dict(sorted(rejection.items())),
    }


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem44Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition, shape = _candidate_parameters(row, config)
    lane = int(row["lane"][0])
    niche = config["candidate_generator"]["niches"][lane]
    return {
        "candidate_id": candidate_id,
        "lane_id": lane,
        "lane": niche["name"],
        "creativity_label": niche["creativity_label"],
        "coordinate": niche["coordinate"],
        "parameters": {
            "amplitude": float(amplitude[0]),
            "acceleration_exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
            "hierarchy_shape": float(shape[0]),
        },
    }


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    admitted, audit = admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item44-candidate-manifest-1.0",
            "item": 44,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "all_empirical_responses_already_exposed": True,
            "response_values_used_during_formula_generation": 0,
            "confirmation_accessed": False,
            "paid_model_calls": 0,
            **audit,
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "niches": config["candidate_generator"]["niches"],
            "claim_boundaries": [
                "transition, curvature, carrier, and horizon scales are known ingredients",
                "the four-scale cross ratio is only a potentially new observational synthesis",
                "algebraic non-equivalence does not establish historical novelty",
                "the retrospective data can generate a lead but cannot confirm one",
            ],
        }
    )


def build_exposure_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item44-exposure-manifest-1.0",
            "item": 44,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "datasets": [
                {
                    "id": "S4TM_ITEM43_EXPLORATION",
                    "objects": 28,
                    "response_status": "already exposed in Item 43",
                    "role": "retrospective joint development",
                },
                {
                    "id": "CLASH_ACCELERATION",
                    "objects": 20,
                    "points": 84,
                    "response_status": "already exposed before Item 44",
                    "role": "retrospective joint development",
                },
            ],
            "sealed_data": {
                "item43_s4tm_confirmation_lenses": 7,
                "access_authorized": False,
                "response_rows_read": 0,
            },
            "rules": [
                "no result may be described as fresh confirmation",
                "select formulas only by balanced whole-object cross-validation",
                "preserve all empirical mismatches and never use one as a veto",
            ],
        }
    )


def write_freeze_manifests(root: Path) -> list[Path]:
    config = load_config(root)
    paths = [
        _source_path(root, config, "candidate_manifest"),
        _source_path(root, config, "exposure_manifest"),
    ]
    _write_json(paths[0], build_candidate_manifest(root))
    _write_json(paths[1], build_exposure_manifest(root))
    return paths


def _clash_redshifts(root: Path, config: Mapping[str, Any]) -> dict[str, float]:
    metadata = _read_vizier_rows(
        root / "runs/gravity/roadmap/item-02-shape-anisotropy-v1-source/clash_clusters.tsv"
    )
    by_id = {str(row["ID"]).strip(): float(row["z"]) for row in metadata}
    aliases = {
        "A209": "A209", "A383": "A383", "A611": "A611", "A2261": "A2261",
        "MACS0329": "M0329", "MACS0416": "M0416", "MACS0429": "M0429",
        "MACS0647": "M0647", "MACS0717": "M0717", "MACS0744": "M0744",
        "MACS1115": "M1115", "MACS1149": "M1149", "MACS1206": "M1206",
        "MACS1720": "M1720", "MACS1931": "M1931", "MS2137": "MS2137",
        "RXJ1347": "RXJ1347", "RXJ1532": "M1532", "RXJ2129": "RXJ2129",
        "RXJ2248": "RXJ2248",
    }
    return {name: by_id[value] for name, value in aliases.items()}


def build_joint_features(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item43_config = _read_json(root / "configs/gravity_item43_cosmological_boundary_v1.json")
    predictors = _read_json(
        root / "runs/gravity/roadmap/item-43-cosmological-boundary-v1-source/s4tm-predictors.json"
    )
    sample = _read_json(
        root / "runs/gravity/roadmap/item-43-cosmological-boundary-v1-source/sample-manifest.json"
    )
    responses = _read_json(
        root / "runs/gravity/roadmap/item-43-cosmological-boundary-v1-source/s4tm-exploration-responses.json"
    )
    predictor_by_name = {row["target"]: row for row in predictors["records"]}
    response_by_name = {
        row["target"]: float(row["log10_einstein_mass_msun"])
        for row in responses["records"]
    }
    sample_rows = [row for row in sample["objects"] if row["role"] == "exploration"]
    s4_records = [predictor_by_name[row["target"]] for row in sample_rows]
    s4_features = _s4tm_feature_arrays(s4_records, item43_config)
    distance = np.asarray(s4_features["radius_kpc"]) / np.asarray(
        [float(row["einstein_radius_arcsec"]) for row in s4_records]
    )
    s4_size = distance * np.asarray(
        [float(row["effective_radius_arcsec"]) for row in s4_records]
    )
    s4_mass = np.power(10.0, np.asarray(s4_features["log_mbar"]))
    s4_hierarchy, s4_scales = hierarchy_coordinates(
        np.asarray(s4_features["radius_kpc"]),
        s4_size,
        s4_mass,
        np.asarray(s4_features["z"]),
        config,
    )
    records: list[dict[str, Any]] = []
    for index, sample_row in enumerate(sample_rows):
        records.append(
            {
                "population": "S4TM",
                "object": sample_row["target"],
                "fold": int(sample_row["outer_fold"]),
                "radius_kpc": float(s4_features["radius_kpc"][index]),
                "baryonic_size_kpc": float(s4_size[index]),
                "redshift": float(s4_features["z"][index]),
                "log10_baryonic_quantity": float(s4_features["log_mbar"][index]),
                "log10_observed_quantity": response_by_name[sample_row["target"]],
                "log10_uncertainty": 0.20615528128088303,
                "u": float(s4_features["u"][index]),
                "hierarchy": [float(value) for value in s4_hierarchy[:, index]],
                "scale_values_kpc": {
                    key: float(value[index]) for key, value in s4_scales.items() if key != "raw_ratios"
                },
                "raw_scale_ratios": [float(value) for value in s4_scales["raw_ratios"][:, index]],
            }
        )
    clash_rows = _read_vizier_rows(
        root / "runs/gravity/g4/cluster-lensing-exploration-v7-source/fig2.tsv"
    )
    z_by_name = _clash_redshifts(root, config)
    cluster_names = sorted({str(row["AName"]).strip() for row in clash_rows})
    cluster_fold = {
        name: index % int(config["evaluation"]["outer_folds"])
        for index, name in enumerate(
            sorted(cluster_names, key=lambda name: _split_hash(name, config["evaluation"]["cluster_fold_salt"]))
        )
    }
    g = float(config["constants"]["gravitational_constant_kpc_km2_s2_msun"])
    conversion = float(config["constants"]["kpc_to_km"]) / 1000.0
    by_cluster: dict[str, list[dict[str, str]]] = {name: [] for name in cluster_names}
    for row in clash_rows:
        by_cluster[str(row["AName"]).strip()].append(row)
    size_by_cluster: dict[str, float] = {}
    for name, rows in by_cluster.items():
        ordered = sorted(rows, key=lambda row: float(row["Rad"]))
        radii = np.asarray([float(row["Rad"]) for row in ordered])
        masses = np.power(10.0, [float(row["log(gbar)"]) for row in ordered]) * conversion * np.square(radii) / g
        masses = np.maximum.accumulate(masses)
        size_by_cluster[name] = float(np.interp(0.5 * masses[-1], masses, radii))
    for row in clash_rows:
        name = str(row["AName"]).strip()
        radius = float(row["Rad"])
        loggbar = float(row["log(gbar)"])
        mass = 10.0**loggbar * conversion * radius * radius / g
        hierarchy, scales = hierarchy_coordinates(
            np.asarray([radius]),
            np.asarray([size_by_cluster[name]]),
            np.asarray([mass]),
            np.asarray([z_by_name[name]]),
            config,
        )
        records.append(
            {
                "population": "CLASH",
                "object": name,
                "fold": cluster_fold[name],
                "radius_kpc": radius,
                "baryonic_size_kpc": size_by_cluster[name],
                "redshift": z_by_name[name],
                "log10_baryonic_quantity": loggbar,
                "log10_observed_quantity": float(row["log(gtot)"]),
                "log10_uncertainty": math.sqrt(
                    float(row["e_log(gbar)"]) ** 2 + float(row["e_log(gtot)"]) ** 2
                ),
                "u": 10.0**loggbar / float(config["constants"]["acceleration_scale_m_s2"]),
                "hierarchy": [float(value) for value in hierarchy[:, 0]],
                "scale_values_kpc": {
                    key: float(value[0]) for key, value in scales.items() if key != "raw_ratios"
                },
                "raw_scale_ratios": [float(value) for value in scales["raw_ratios"][:, 0]],
            }
        )
    h = np.asarray([row["hierarchy"] for row in records])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item44-joint-features-1.0",
            "item": 44,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "records": records,
            "counts": {
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "total_points": len(records),
                "sealed_confirmation_rows": 0,
                "paid_model_calls": 0,
            },
            "hierarchy_ranges": {
                str(lane): {
                    "minimum": float(np.min(h[:, lane])),
                    "maximum": float(np.max(h[:, lane])),
                }
                for lane in range(4)
            },
            "lineage": config["data_roles"],
        }
    )


def write_joint_features(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "feature_receipt")
    _write_json(path, build_joint_features(root))
    return path


def _arrays(feature_doc: Mapping[str, Any]) -> dict[str, Any]:
    rows = feature_doc["records"]
    return {
        "population": np.asarray([row["population"] for row in rows]),
        "object": np.asarray([row["object"] for row in rows]),
        "fold": np.asarray([int(row["fold"]) for row in rows]),
        "base": np.asarray([float(row["log10_baryonic_quantity"]) for row in rows]),
        "target": np.asarray([float(row["log10_observed_quantity"]) for row in rows]),
        "sigma": np.asarray([float(row["log10_uncertainty"]) for row in rows]),
        "u": np.asarray([float(row["u"]) for row in rows]),
        "hierarchy": np.asarray([row["hierarchy"] for row in rows], dtype=np.float64).T,
        "raw_ratios": np.asarray(
            [row["raw_scale_ratios"] for row in rows], dtype=np.float64
        ).T,
        "radius": np.asarray([float(row["radius_kpc"]) for row in rows]),
        "size": np.asarray([float(row["baryonic_size_kpc"]) for row in rows]),
        "redshift": np.asarray([float(row["redshift"]) for row in rows]),
    }


def _mass_scale_variant(
    arrays: Mapping[str, Any], population: str, shift_dex: float
) -> dict[str, Any]:
    varied = {
        key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        for key, value in arrays.items()
    }
    mask = varied["population"] == population
    factor = 10.0**shift_dex
    varied["base"][mask] += shift_dex
    varied["u"][mask] *= factor
    ratio_factors = np.asarray(
        [factor**-0.5, factor**-0.5, factor**-0.5, factor**-1.0]
    )
    varied["raw_ratios"][:, mask] *= ratio_factors[:, None]
    varied["hierarchy"][:, mask] = 1.0 / (
        1.0
        + np.abs(
            np.log10(np.maximum(varied["raw_ratios"][:, mask], 1e-300))
        )
    )
    return varied


def _object_weights(arrays: Mapping[str, Any], train: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(arrays["target"]), dtype=np.float64)
    for population in ("S4TM", "CLASH"):
        mask = train & (arrays["population"] == population)
        objects = sorted(set(arrays["object"][mask].tolist()))
        for name in objects:
            points = mask & (arrays["object"] == name)
            weights[points] = 0.5 / len(objects) / int(np.sum(points))
    return weights


def _candidate_subset(candidates: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in candidates.items()}


def _best_candidate(
    candidates: Mapping[str, np.ndarray], arrays: Mapping[str, Any], train: np.ndarray,
    config: Mapping[str, Any], *, scale_free: bool = False,
) -> tuple[int, float, str, int]:
    weights_np = _object_weights(arrays, train)
    indices = np.flatnonzero(train)
    backend = "numpy_cpu"
    xp: Any = np
    try:
        import cupy as cp

        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            xp = cp
            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            backend = "cupy_cuda_" + (name.decode() if isinstance(name, bytes) else str(name))
    except Exception:
        xp = np
    u = xp.asarray(arrays["u"][indices])
    hierarchy = xp.asarray(arrays["hierarchy"][:, indices])
    residual = xp.asarray(arrays["target"][indices] - arrays["base"][indices])
    sigma = xp.asarray(arrays["sigma"][indices])
    weights = xp.asarray(weights_np[indices])
    grids = config["candidate_generator"]["parameter_grids"]
    amplitude_grid = xp.asarray(grids["amplitude"])
    exponent_grid = xp.asarray(grids["acceleration_exponent"])
    transition_grid = xp.asarray(grids["transition_u"])
    shape_grid = xp.asarray(grids["hierarchy_shape"])
    best_loss = math.inf
    best_index = -1
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(candidates["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(candidates["candidate_id"]))
        lane = xp.asarray(np.asarray(candidates["lane"])[begin:end], dtype=xp.int64)
        aa = amplitude_grid[xp.asarray(np.asarray(candidates["amplitude_index"])[begin:end])]
        pp = exponent_grid[xp.asarray(np.asarray(candidates["exponent_index"])[begin:end])]
        tt = transition_grid[xp.asarray(np.asarray(candidates["transition_index"])[begin:end])]
        ss = shape_grid[xp.asarray(np.asarray(candidates["shape_index"])[begin:end])]
        h = xp.ones((end - begin, len(indices))) if scale_free else hierarchy[lane]
        response = 0.05 + 0.95 * xp.power(h, ss[:, None])
        multiplier = 1.0 + aa[:, None] * xp.power(u[None, :], -pp[:, None]) / (
            1.0 + u[None, :] / tt[:, None]
        ) * response
        errors = xp.square((xp.log10(multiplier) - residual[None, :]) / sigma[None, :])
        losses = xp.sum(errors * weights[None, :], axis=1)
        local = int(xp.argmin(losses).item())
        loss = float(losses[local].item())
        if loss < best_loss:
            best_loss = loss
            best_index = begin + local
    return (
        int(np.asarray(candidates["candidate_id"])[best_index]),
        best_loss,
        backend,
        len(candidates["candidate_id"]) * len(indices),
    )


def _predict(
    candidate_id: int, arrays: Mapping[str, Any], config: Mapping[str, Any], *,
    scale_free: bool = False,
) -> np.ndarray:
    raw = generate_raw_candidates(config)
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    multiplier = candidate_multiplier(
        row, arrays["u"], arrays["hierarchy"], config, scale_free=scale_free
    )[0]
    return arrays["base"] + np.log10(multiplier)


def _score(arrays: Mapping[str, Any], prediction: np.ndarray) -> dict[str, Any]:
    error = np.square((prediction - arrays["target"]) / arrays["sigma"])
    populations: dict[str, Any] = {}
    object_losses: dict[str, float] = {}
    for population in ("S4TM", "CLASH"):
        mask = arrays["population"] == population
        names = sorted(set(arrays["object"][mask].tolist()))
        losses = []
        for name in names:
            value = float(np.mean(error[mask & (arrays["object"] == name)]))
            object_losses[f"{population}:{name}"] = value
            losses.append(value)
        populations[population] = {"loss": float(np.mean(losses)), "objects": len(names)}
    return {
        "balanced_loss": 0.5 * (populations["S4TM"]["loss"] + populations["CLASH"]["loss"]),
        "populations": populations,
        "object_losses": object_losses,
    }


def _ordinary_crossfit(arrays: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
    logu = np.log10(arrays["u"])
    x = np.column_stack(
        (
            logu,
            np.log10(arrays["radius"]),
            np.log10(arrays["size"]),
            arrays["redshift"],
            arrays["hierarchy"].T,
            (arrays["population"] == "CLASH").astype(float),
        )
    )
    residual = arrays["target"] - arrays["base"]
    result = np.empty(len(residual), dtype=np.float64)
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        weights = _object_weights(arrays, train)[train]
        mean = np.average(x[train], axis=0, weights=weights)
        scale = np.sqrt(np.average(np.square(x[train] - mean), axis=0, weights=weights))
        scale[scale < 1e-12] = 1.0
        design = np.column_stack((np.ones(np.sum(train)), (x[train] - mean) / scale))
        test_design = np.column_stack((np.ones(np.sum(test)), (x[test] - mean) / scale))
        root_w = np.sqrt(weights / np.sum(weights))
        weighted = design * root_w[:, None]
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            weighted.T @ weighted + penalty, weighted.T @ (residual[train] * root_w)
        )
        result[test] = arrays["base"][test] + test_design @ coefficients
    return result


def _paired_p(diff: np.ndarray, config: Mapping[str, Any]) -> float:
    rng = np.random.default_rng(int(config["evaluation"]["permutation_seed"]))
    observed = abs(float(np.mean(diff)))
    exceed = 0
    count = int(config["evaluation"]["paired_sign_flip_permutations"])
    for _ in range(count):
        value = abs(float(np.mean(diff * rng.choice((-1.0, 1.0), len(diff)))))
        exceed += int(value >= observed - 1e-15)
    return (exceed + 1.0) / (count + 1.0)


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    feature_doc = _read_json(_source_path(root, config, "feature_receipt"))
    arrays = _arrays(feature_doc)
    admitted, admission = admissible_candidates(config)
    scale_free_mask = (
        (np.asarray(admitted["lane"]) == 0)
        & (np.asarray(admitted["shape_index"]) == 0)
    )
    scale_free = _candidate_subset(admitted, scale_free_mask)
    candidate_oof = np.empty(len(arrays["target"]), dtype=np.float64)
    control_oof = np.empty(len(arrays["target"]), dtype=np.float64)
    ledger: list[dict[str, Any]] = []
    fold_candidate: dict[int, int] = {}
    fold_control: dict[int, int] = {}
    backends: set[str] = set()
    evaluations = 0
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        candidate_id, train_loss, backend, count = _best_candidate(
            admitted, arrays, train, config
        )
        control_id, control_loss, control_backend, control_count = _best_candidate(
            scale_free, arrays, train, config, scale_free=True
        )
        candidate_oof[test] = _predict(candidate_id, arrays, config)[test]
        control_oof[test] = _predict(control_id, arrays, config, scale_free=True)[test]
        evaluations += count + control_count
        backends.update((backend, control_backend))
        fold_candidate[fold] = candidate_id
        fold_control[fold] = control_id
        ledger.append(
            {
                "fold": fold,
                "selected_candidate": decode_candidate(candidate_id, config),
                "training_balanced_loss": train_loss,
                "selected_scale_free_control": decode_candidate(control_id, config),
                "scale_free_training_balanced_loss": control_loss,
                "heldout_s4tm_objects": sorted(set(arrays["object"][test & (arrays["population"] == "S4TM")].tolist())),
                "heldout_clash_objects": sorted(set(arrays["object"][test & (arrays["population"] == "CLASH")].tolist())),
            }
        )
    all_rows = np.ones(len(arrays["target"]), dtype=bool)
    selected_id, selected_loss, backend, count = _best_candidate(
        admitted, arrays, all_rows, config
    )
    control_id, control_loss, control_backend, control_count = _best_candidate(
        scale_free, arrays, all_rows, config, scale_free=True
    )
    evaluations += count + control_count
    backends.update((backend, control_backend))
    selected_prediction = _predict(selected_id, arrays, config)
    cpu_loss = _score(arrays, selected_prediction)["balanced_loss"]
    cpu_gpu_difference = abs(float(cpu_loss) - selected_loss)
    if cpu_gpu_difference > float(config["evaluation"]["cpu_gpu_tolerance"]):
        raise GravityItem44Error("CPU/GPU selected loss cross-check failed")
    newton = arrays["base"].copy()
    mond = arrays["base"] + np.log10(
        1.0 / (1.0 - np.exp(-np.sqrt(arrays["u"])))
    )
    ordinary = _ordinary_crossfit(arrays, config)
    scores = {
        "scale_hierarchy": _score(arrays, candidate_oof),
        "matched_scale_free": _score(arrays, control_oof),
        "baryonic_newton": _score(arrays, newton),
        "mond_rar": _score(arrays, mond),
        "ordinary_ridge": _score(arrays, ordinary),
    }
    controls = ("matched_scale_free", "baryonic_newton", "mond_rar", "ordinary_ridge")
    strongest = min(controls, key=lambda name: scores[name]["balanced_loss"])
    candidate_objects = scores["scale_hierarchy"]["object_losses"]
    control_objects = scores[strongest]["object_losses"]
    object_keys = sorted(candidate_objects)
    diff = np.asarray([control_objects[key] - candidate_objects[key] for key in object_keys])
    raw = diff < 0.0
    stable = raw.copy()
    systematic_scores: dict[str, Any] = {}
    for variant_name, population, shift in (
        ("s4tm_stellar_mass_minus_0.25_dex", "S4TM", -0.25),
        ("s4tm_stellar_mass_plus_0.25_dex", "S4TM", 0.25),
        ("clash_baryonic_scale_minus_0.10_dex", "CLASH", -0.10),
        ("clash_baryonic_scale_plus_0.10_dex", "CLASH", 0.10),
    ):
        varied = _mass_scale_variant(arrays, population, shift)
        candidate_variant = np.empty(len(varied["target"]), dtype=np.float64)
        control_variant = np.empty(len(varied["target"]), dtype=np.float64)
        for fold in range(int(config["evaluation"]["outer_folds"])):
            test = varied["fold"] == fold
            candidate_variant[test] = _predict(
                fold_candidate[fold], varied, config
            )[test]
            control_variant[test] = _predict(
                fold_control[fold], varied, config, scale_free=True
            )[test]
        candidate_variant_score = _score(varied, candidate_variant)
        control_variant_score = _score(varied, control_variant)
        systematic_scores[variant_name] = {
            "scale_hierarchy": candidate_variant_score,
            "matched_scale_free": control_variant_score,
        }
        for index, key in enumerate(object_keys):
            stable[index] &= (
                candidate_variant_score["object_losses"][key]
                > control_variant_score["object_losses"][key]
            )
    leave_one = [float(np.mean(np.delete(diff, index))) for index in range(len(diff))]
    trim_count = max(1, int(len(diff) * float(config["evaluation"]["robust_trim_fraction"])))
    trimmed = np.sort(diff)[trim_count:-trim_count]
    improvement = 100.0 * (
        scores[strongest]["balanced_loss"] - scores["scale_hierarchy"]["balanced_loss"]
    ) / scores[strongest]["balanced_loss"]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(object_keys),
        "raw_counterexample_count": int(np.sum(raw)),
        "quality_verified_counterexample_count": int(np.sum(raw)),
        "uncertainty_resolved_counterexample_count": int(np.sum(stable)),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": False,
        "strongest_baseline_failed": bool(improvement <= 0.0),
        "leave_one_changes_sign": bool((min(leave_one) <= 0.0) != (float(np.mean(diff)) <= 0.0)),
        "trim_changes_sign": bool((float(np.mean(trimmed)) <= 0.0) != (float(np.mean(diff)) <= 0.0)),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item44-joint-evaluation-1.0",
            "item": 44,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selected_candidate": decode_candidate(selected_id, config),
            "selected_full_data_balanced_training_loss": selected_loss,
            "selected_scale_free_control": decode_candidate(control_id, config),
            "scale_free_full_data_balanced_training_loss": control_loss,
            "fold_ledger": ledger,
            "scores": scores,
            "strongest_control": strongest,
            "aggregate_improvement_percent": improvement,
            "paired_sign_flip_p": _paired_p(diff, config),
            "robustness": {
                "leave_one_min_mean_control_minus_candidate_loss": min(leave_one),
                "leave_one_max_mean_control_minus_candidate_loss": max(leave_one),
                "trimmed_mean_control_minus_candidate_loss": float(np.mean(trimmed)),
            },
            "counterexamples": [
                {
                    "object": key,
                    "raw_counterexample": bool(raw[index]),
                    "uncertainty_resolved_counterexample": bool(stable[index]),
                }
                for index, key in enumerate(object_keys)
            ],
            "systematic_scores": systematic_scores,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": policy,
            "compute": {
                "backends": sorted(backends),
                "candidate_point_fold_evaluations": evaluations,
                "cpu_gpu_selected_loss_absolute_difference": cpu_gpu_difference,
                "admission": admission,
            },
            "implementation_audit": {
                "timing": "after the first retrospective nominal evaluation",
                "repair": "The first evaluator omitted the required mass-scale uncertainty audit; fixed-fold candidates were replayed under the frozen S4TM plus/minus 0.25 dex and CLASH plus/minus 0.10 dex mass-scale variants.",
                "formula_space_changed": False,
                "fold_selection_changed": False,
                "nominal_candidate_or_score_changed": False,
                "post_evaluation_candidate_cells_added": 0,
            },
            "counts": {
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "limitations": [
                "Both empirical datasets were exposed before Item 44, so cross-validation limits target leakage but cannot create fresh confirmation.",
                "S4TM and CLASH lens quantities are model-derived summaries rather than raw image likelihoods.",
                "The cluster baryonic size is a spherical-equivalent half-enclosed-mass radius inferred from the published gbar profile.",
                "Uncertainty-resolved counts cover only four frozen global mass-scale shifts; they do not replace independent raw-data systematics or fresh replication.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    candidate = _read_json(_source_path(root, config, "candidate_manifest"))
    exposure = _read_json(_source_path(root, config, "exposure_manifest"))
    features = _read_json(_source_path(root, config, "feature_receipt"))
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    scores = evaluation["scores"]
    gates = {
        "beats_scale_free_s4tm": scores["scale_hierarchy"]["populations"]["S4TM"]["loss"] < scores["matched_scale_free"]["populations"]["S4TM"]["loss"],
        "beats_scale_free_clash": scores["scale_hierarchy"]["populations"]["CLASH"]["loss"] < scores["matched_scale_free"]["populations"]["CLASH"]["loss"],
        "beats_ordinary_ridge_balanced": scores["scale_hierarchy"]["balanced_loss"] < scores["ordinary_ridge"]["balanced_loss"],
        "paired_p_passes": float(evaluation["paired_sign_flip_p"]) <= float(config["gates"]["paired_p_maximum"]),
        "leave_one_stable": float(evaluation["robustness"]["leave_one_min_mean_control_minus_candidate_loss"]) > 0.0,
        "trim_stable": float(evaluation["robustness"]["trimmed_mean_control_minus_candidate_loss"]) > 0.0,
        "mass_scale_audits_all_improve": all(
            audit["scale_hierarchy"]["balanced_loss"]
            < audit["matched_scale_free"]["balanced_loss"]
            for audit in evaluation["systematic_scores"].values()
        ),
        "confirmation_rows_zero": int(evaluation["counts"]["sealed_confirmation_rows"]) == 0,
        "post_evaluation_candidates_zero": int(evaluation["counts"]["post_evaluation_candidate_cells"]) == 0,
        "fresh_confirmation_available": False,
    }
    empirical_lead = all(
        gates[key]
        for key in (
            "beats_scale_free_s4tm", "beats_scale_free_clash", "beats_ordinary_ridge_balanced",
            "paired_p_passes", "leave_one_stable", "trim_stable",
            "mass_scale_audits_all_improve",
        )
    )
    decision = (
        "RETROSPECTIVE_ITEM44_SCALE_HIERARCHY_LEAD_REQUIRES_FRESH_TEST"
        if empirical_lead
        else "NONPROMOTED_ITEM44_SCALE_HIERARCHY_RESULT_RETAINED"
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item44-scale-hierarchy-result-1.0",
            "item": 44,
            "goal": "GRAVITY_ROADMAP_ITEM_44_SCALE_HIERARCHY",
            "decision": decision,
            "selected_candidate": evaluation["selected_candidate"],
            "scores": scores,
            "strongest_control": evaluation["strongest_control"],
            "aggregate_improvement_percent": evaluation["aggregate_improvement_percent"],
            "paired_sign_flip_p": evaluation["paired_sign_flip_p"],
            "gates": gates,
            "counterexample_policy_assessment": evaluation["counterexample_policy_assessment"],
            "counts": {
                "raw_candidates": candidate["raw_candidates"],
                "admitted_candidates": candidate["admitted_candidates"],
                "s4tm_lenses": features["counts"]["s4tm_lenses"],
                "clash_clusters": features["counts"]["clash_clusters"],
                "clash_points": features["counts"]["clash_points"],
                "candidate_point_fold_evaluations": evaluation["compute"]["candidate_point_fold_evaluations"],
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "source_bindings": {
                "config": {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)},
                "candidate_manifest": {"path": str(_source_path(root, config, "candidate_manifest").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "candidate_manifest"))},
                "exposure_manifest": {"path": str(_source_path(root, config, "exposure_manifest").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "exposure_manifest"))},
                "features": {"path": str(_source_path(root, config, "feature_receipt").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "feature_receipt"))},
                "evaluation": {"path": str(_source_path(root, config, "evaluation_result").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "evaluation_result"))},
            },
            "claims": {
                "roadmap_item_44_complete": True,
                "fresh_confirmation_completed": False,
                "scale_hierarchy_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "covariant_theory_established": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": "Preserve the selected hierarchy and every mismatch; if it is a retrospective lead, preregister an unchanged fresh lens/cluster test, then advance to Item 45 universal interaction variables.",
            "exposure": exposure,
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    checks = {
        "candidate_manifest": _read_json(_source_path(root, config, "candidate_manifest")) == build_candidate_manifest(root),
        "exposure_manifest": _read_json(_source_path(root, config, "exposure_manifest")) == build_exposure_manifest(root),
        "feature_receipt": _read_json(_source_path(root, config, "feature_receipt")) == build_joint_features(root),
        "evaluation_result": _read_json(_source_path(root, config, "evaluation_result")) == build_evaluation_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"])) == build_aggregate_result(root),
    }
    return {"valid": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "features", "evaluate", "aggregate", "replay"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.command == "freeze":
        result: Any = [str(path) for path in write_freeze_manifests(root)]
    elif args.command == "features":
        result = str(write_joint_features(root))
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["valid"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
