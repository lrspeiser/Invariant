"""Item 47 typed scalar projections of local and nonlocal operator classes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
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
from sigma_theory_compiler.gravity_item45_universal_interactions import (
    _best_candidate,
    _item44_oof,
    _ordinary_crossfit,
    _paired_p,
    _predict as _item45_predict,
    _score,
    _variant_arrays as _item45_variant_arrays,
    load_config as _load_item45_config,
)
from sigma_theory_compiler.gravity_item46_dimensionless_generator import (
    _evaluation_arrays as _item46_evaluation_arrays,
    _physical_log_values as _item46_physical_log_values,
    _predict as _item46_predict,
    load_config as _load_item46_config,
    pi_vectors as _item46_pi_vectors,
)

CONFIG_PATH = Path("configs/gravity_item47_operator_generator_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM44_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-scale-features.json"
)
ITEM45_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-45-universal-interactions-v1-source/joint-evaluation-result.json"
)
ITEM46_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-46-dimensionless-generator-v1-source/joint-evaluation-result.json"
)
S4TM_PREDICTOR_PATH = Path(
    "runs/gravity/roadmap/item-43-cosmological-boundary-v1-source/s4tm-predictors.json"
)
CLASH_SHAPE_PATH = Path(
    "runs/gravity/roadmap/item-02-shape-anisotropy-v1-source/clash_xray_morphology_500kpc.tsv"
)


class GravityItem47Error(RuntimeError):
    """Raised when an Item 47 operator, support, leakage, or evaluation gate fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item47-operator-generator-config-1.0"
        or int(config.get("item", -1)) != 47
    ):
        raise GravityItem47Error("unexpected Item 47 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem47Error("stable gravity goal changed")
    if re.fullmatch(r"[0-9a-f]{40}", str(config["scientific_freeze_commit"])) is None:
        raise GravityItem47Error("Item 47 scientific freeze is not bound to a commit")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem47Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-46-dimensionless-generator-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem47Error("Item 46 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem47Error("Item 46 decision binding changed")
    if int(predecessor["selected_candidate"]["candidate_id"]) != int(required["selected_candidate_id"]):
        raise GravityItem47Error("Item 46 candidate binding changed")
    discovery = config["discovery_policy"]
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem47Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem47Error("count-only rejection entered Item 47")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem47Error("finite empirical family pruning entered Item 47")
    policy = load_counterexample_policy(root / POLICY_PATH)
    if policy["empirical_evidence"]["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem47Error("executable counterexample policy changed")
    operators = config["operator_generator"]
    if (
        len(operators["operator_classes"]) != 6
        or int(operators["recipes_per_class"]) != 16
        or int(operators["operator_recipes"]) != 96
        or any(len(operators["class_sources"][name]) != 4 for name in operators["operator_classes"])
    ):
        raise GravityItem47Error("operator grammar capacity changed")
    generator = config["candidate_generator"]
    if (
        int(generator["operator_recipes"]) != 96
        or int(generator["cells_per_recipe"]) != 4096
        or int(generator["cells_per_operator_class"]) != 65536
        or int(generator["raw_candidate_cells"]) != 393216
        or int(generator["post_evaluation_cells"]) != 0
    ):
        raise GravityItem47Error("candidate capacity changed")
    if int(operators["response_values_allowed_in_operator_generation"]) != 0:
        raise GravityItem47Error("responses entered operator generation")
    if bool(config["scope"]["measured_baryonic_history_available"]):
        raise GravityItem47Error("unmeasured history was promoted to measured history")
    if bool(config["scope"]["fresh_confirmation_claim_allowed"]):
        raise GravityItem47Error("fresh confirmation entered retrospective Item 47")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem47Error("paid calls entered Item 47")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def operator_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    generator = config["operator_generator"]
    catalogs = []
    for class_id, class_name in enumerate(generator["operator_classes"]):
        sources = generator["class_sources"][class_name]
        if class_name in {"local"}:
            scales = generator["local_soft_scales"]
        elif class_name == "tensor_scalar":
            scales = generator["tensor_soft_scales"]
        elif class_name == "causal_history":
            scales = generator["history_tau_over_t0"]
        else:
            scales = generator["radial_kernel_scales_log_radius"]
        for source_id, source in enumerate(sources):
            for scale_id, scale in enumerate(scales):
                recipe_id = len(catalogs)
                catalogs.append(
                    {
                        "recipe_id": recipe_id,
                        "operator_class_id": class_id,
                        "operator_class": class_name,
                        "source_id": source_id,
                        "source": source,
                        "scale_id": scale_id,
                        "scale": float(scale),
                        "support_rule": generator["support_rules"][class_name],
                        "creativity_label": generator["creativity_labels"][class_name],
                        "historical_novelty_claimed": False,
                    }
                )
    if len(catalogs) != 96:
        raise GravityItem47Error("operator catalog does not contain 96 recipes")
    return catalogs


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    candidate_id = np.arange(total, dtype=np.int64)
    recipe = (candidate_id // per_recipe).astype(np.int16)
    local = candidate_id % per_recipe
    return {
        "candidate_id": candidate_id,
        "recipe": recipe,
        "amplitude_index": ((local // 256) % 16).astype(np.int8),
        "exponent_index": ((local // 16) % 16).astype(np.int8),
        "transition_index": (local % 16).astype(np.int8),
    }


def _candidate_parameters(
    candidates: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grids = config["candidate_generator"]["parameter_grids"]
    return (
        np.asarray(grids["amplitude"])[np.asarray(candidates["amplitude_index"], int)],
        np.asarray(grids["acceleration_exponent"])[np.asarray(candidates["exponent_index"], int)],
        np.asarray(grids["transition_u"])[np.asarray(candidates["transition_index"], int)],
    )


def admissible_candidates(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    gate = config["candidate_generator"]["admissibility"]
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    local_id = np.arange(per_recipe, dtype=np.int64)
    local = {
        "candidate_id": local_id,
        "recipe": np.zeros(per_recipe, dtype=np.int16),
        "amplitude_index": ((local_id // 256) % 16).astype(np.int8),
        "exponent_index": ((local_id // 16) % 16).astype(np.int8),
        "transition_index": (local_id % 16).astype(np.int8),
    }
    amplitude, exponent, transition = _candidate_parameters(local, config)
    u = np.logspace(float(gate["probe_log10_u_min"]), float(gate["probe_log10_u_max"]), int(gate["probe_points"]))
    h = np.asarray(gate["probe_operator_coordinates"], dtype=float)
    multiplier = 1.0 + amplitude[:, None, None] * np.power(
        u[None, None, :], -exponent[:, None, None]
    ) / (1.0 + u[None, None, :] / transition[:, None, None]) * (0.05 + 0.95 * h[None, :, None])
    finite = np.all(np.isfinite(multiplier), axis=(1, 2))
    bounded = finite & np.all(multiplier >= float(gate["minimum_multiplier"]), axis=(1, 2))
    bounded &= np.all(multiplier <= float(gate["maximum_multiplier"]), axis=(1, 2))
    local_limit = bounded & (
        np.max(np.log10(multiplier[:, :, -1]), axis=1)
        <= float(gate["maximum_high_acceleration_log10_deviation"])
    )
    material = local_limit & (
        np.min(multiplier[:, :, 0], axis=1) >= float(gate["minimum_low_acceleration_multiplier"])
    )
    monotone = material & np.all(
        np.diff(multiplier, axis=2) <= float(gate["monotone_nonincreasing_tolerance"]), axis=(1, 2)
    )
    kept_local = np.flatnonzero(monotone)
    recipes = int(config["candidate_generator"]["operator_recipes"])
    keep = np.concatenate([kept_local + recipe * per_recipe for recipe in range(recipes)])
    raw = generate_raw_candidates(config)
    admitted = {key: value[keep] for key, value in raw.items()}
    signatures = {
        hashlib.blake2b(row.tobytes(), digest_size=16).digest()
        for row in np.round(np.log10(multiplier[kept_local]), int(gate["behavior_signature_decimals"]))
    }
    local_counts = {
        "nonfinite": int(np.sum(~finite)),
        "out_of_bounds": int(np.sum(finite & ~bounded)),
        "no_local_limit": int(np.sum(bounded & ~local_limit)),
        "immaterial_low_acceleration": int(np.sum(local_limit & ~material)),
        "nonmonotone": int(np.sum(material & ~monotone)),
    }
    return admitted, {
        "raw_candidates": len(raw["candidate_id"]),
        "admitted_candidates": len(keep),
        "rejected_candidates": len(raw["candidate_id"]) - len(keep),
        "admitted_per_operator_recipe": len(kept_local),
        "admitted_by_operator_class": {
            name: int(np.sum(np.asarray(admitted["recipe"]) // 16 == class_id))
            for class_id, name in enumerate(config["operator_generator"]["operator_classes"])
        },
        "generic_formula_behavior_classes": len(signatures),
        "symbolic_operator_recipes": recipes,
        "rejection_counts_nonexclusive": {key: value * recipes for key, value in local_counts.items()},
    }


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    if candidate_id < 0 or candidate_id >= total:
        raise GravityItem47Error("candidate id outside frozen grid")
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    recipe_id = candidate_id // per_recipe
    local = candidate_id % per_recipe
    row = {
        "amplitude_index": np.asarray([(local // 256) % 16]),
        "exponent_index": np.asarray([(local // 16) % 16]),
        "transition_index": np.asarray([local % 16]),
    }
    amplitude, exponent, transition = _candidate_parameters(row, config)
    return {
        **operator_catalog(config)[recipe_id],
        "candidate_id": candidate_id,
        "coordinate_expression": "H=0.5+0.5*I_operator/(1+abs(I_operator))",
        "parameters": {
            "amplitude": float(amplitude[0]),
            "acceleration_exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
        },
    }


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    admitted, audit = admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item47-candidate-manifest-1.0",
            "item": 47,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "response_values_used_during_operator_or_formula_generation": 0,
            "confirmation_accessed": False,
            "paid_model_calls": 0,
            **audit,
            "candidate_id_sha256": _sha256_bytes(np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()),
            "operator_catalog": operator_catalog(config),
            "claim_boundaries": [
                "local, differential, integral, exterior-kernel, tensor, and history labels describe scalar observational projections, not complete covariant operators",
                "the causal-history lane has no measured source history and uses a declared constant-state closure",
                "tensor recipes use only a rotation-invariant axis-ratio amplitude and cannot test directional lensing",
                "no algebraic or behavioral distinction establishes historical novelty",
                "retrospective grouped validation can generate a lead but cannot confirm one",
            ],
        }
    )


def build_exposure_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item47-exposure-manifest-1.0",
            "item": 47,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "datasets": [
                {"id": "S4TM_ITEM43_EXPLORATION", "objects": 28, "response_status": "already exposed", "role": "retrospective operator development"},
                {"id": "CLASH_ACCELERATION", "objects": 20, "points": 84, "response_status": "already exposed", "role": "retrospective operator development"},
            ],
            "sealed_data": {"item43_s4tm_confirmation_lenses": 7, "access_authorized": False, "response_rows_read": 0},
            "rules": [
                "no result may be described as fresh confirmation",
                "construct profiles, shapes, supports, and operator coordinates without responses",
                "preserve every mismatch; neither one counterexample nor its count is a veto",
            ],
        }
    )


def write_freeze_manifests(root: Path) -> list[Path]:
    config = load_config(root)
    paths = [_source_path(root, config, "candidate_manifest"), _source_path(root, config, "exposure_manifest")]
    _write_json(paths[0], build_candidate_manifest(root))
    _write_json(paths[1], build_exposure_manifest(root))
    return paths


def _devaucouleurs_fraction(radius_over_re: np.ndarray) -> np.ndarray:
    x = 7.669249442500804 * np.power(np.maximum(radius_over_re, 1e-300), 0.25)
    series = np.zeros_like(x)
    term = np.ones_like(x)
    for k in range(8):
        if k > 0:
            term = term * x / k
        series += term
    return np.maximum(1.0 - np.exp(-x) * series, 1e-300)


def _weighted_derivatives(
    radius: np.ndarray, mass: np.ndarray, evaluation_radius: float, bandwidth: float
) -> tuple[float, float]:
    x = np.log(np.asarray(radius, dtype=float))
    y = np.log(np.maximum(np.asarray(mass, dtype=float), 1e-300))
    delta = x - math.log(float(evaluation_radius))
    weights = np.exp(-0.5 * np.square(delta / float(bandwidth)))
    keep = weights > 1e-8
    if int(np.sum(keep)) < 3:
        keep = np.argsort(np.abs(delta))[: min(3, len(delta))]
        mask = np.zeros(len(delta), dtype=bool)
        mask[keep] = True
        keep = mask
    design = np.column_stack((np.ones(np.sum(keep)), delta[keep], np.square(delta[keep])))
    root_weight = np.sqrt(weights[keep] + 1e-12)
    coefficients = np.linalg.lstsq(
        design * root_weight[:, None], y[keep] * root_weight, rcond=None
    )[0]
    return float(coefficients[1]), float(2.0 * coefficients[2])


def _profile_slopes(radius: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.gradient(np.log(np.maximum(mass, 1e-300)), np.log(radius), edge_order=1)


def _operator_raw_values(
    profile_radius: np.ndarray,
    profile_mass: np.ndarray,
    *,
    radius: float,
    baryonic_size: float,
    u: float,
    axis_ratio: float,
    age_over_t0: float,
    config: Mapping[str, Any],
) -> np.ndarray:
    pr = np.asarray(profile_radius, dtype=float)
    pm = np.maximum.accumulate(np.asarray(profile_mass, dtype=float))
    if len(pr) < 3 or np.any(pr <= 0.0) or np.any(pm <= 0.0) or not np.all(np.diff(pr) > 0.0):
        raise GravityItem47Error("profile must have at least three positive ordered nodes")
    geometry = math.log10(radius / baryonic_size)
    logu = math.log10(u)
    anisotropy = (1.0 - axis_ratio) / (1.0 + axis_ratio)
    q_fixed, curvature_fixed = _weighted_derivatives(pr, pm, radius, 0.6)
    values: list[float] = []
    generator = config["operator_generator"]
    local_sources = (logu, geometry, q_fixed - 1.0, anisotropy)
    for source in local_sources:
        for scale in generator["local_soft_scales"]:
            values.append(source / float(scale))
    for source_id in range(4):
        for bandwidth in generator["radial_kernel_scales_log_radius"]:
            q, curvature = _weighted_derivatives(pr, pm, radius, float(bandwidth))
            differential = (q, q - 2.0, curvature, curvature + q * (q - 1.0))[source_id]
            values.append(differential)
    shells = np.diff(np.concatenate(([0.0], pm)))
    log_ratio = np.log(pr / radius)
    inside = pr <= radius * (1.0 + 1e-12)
    outside = pr > radius * (1.0 + 1e-12)
    m_current = float(np.interp(radius, pr, pm))
    g_profile = pm / np.square(pr)
    g_current = m_current / (radius * radius)
    for source_id in range(4):
        for scale in generator["radial_kernel_scales_log_radius"]:
            weights = np.exp(-np.abs(log_ratio) / float(scale))
            iw = weights[inside] * shells[inside]
            if source_id == 0:
                value = float(np.sum(iw) / m_current)
            elif source_id == 1:
                value = float(radius * np.sum(iw / pr[inside]) / m_current)
            elif source_id == 2:
                denom = float(np.sum(iw))
                value = 0.0 if denom <= 0.0 else float(np.sum(iw * np.log(g_profile[inside] / g_current)) / denom)
            else:
                sub_r = pr[inside]
                sub_m = pm[inside]
                if len(sub_r) < 2:
                    value = 0.0
                else:
                    slope = _profile_slopes(sub_r, sub_m)
                    denom = float(np.sum(iw))
                    value = 0.0 if denom <= 0.0 else float(np.sum(iw * (slope - slope[-1])) / denom)
            values.append(value)
    for source_id in range(4):
        for scale in generator["radial_kernel_scales_log_radius"]:
            weights = np.exp(-np.abs(log_ratio) / float(scale))
            ow = weights[outside] * shells[outside]
            if not np.any(outside) or float(np.sum(ow)) <= 0.0:
                value = 0.0
            elif source_id == 0:
                value = float(np.sum(ow) / m_current)
            elif source_id == 1:
                value = float(radius * np.sum(ow / pr[outside]) / m_current)
            elif source_id == 2:
                value = float(np.sum(ow * np.log(g_profile[outside] / g_current)) / np.sum(ow))
            else:
                sub_r = pr[outside]
                sub_m = pm[outside]
                if len(sub_r) < 2:
                    value = 0.0
                else:
                    slope = _profile_slopes(sub_r, sub_m)
                    value = float(np.sum(ow * (slope - slope[0])) / np.sum(ow))
            values.append(value)
    tensor_sources = (q_fixed, logu, geometry, curvature_fixed)
    for source in tensor_sources:
        for scale in generator["tensor_soft_scales"]:
            values.append(anisotropy * source / float(scale))
    history_sources = (
        1.0,
        logu / (1.0 + abs(logu)),
        geometry / (1.0 + abs(geometry)),
        (q_fixed - 1.0) / (1.0 + abs(q_fixed - 1.0)),
    )
    for source_id, source in enumerate(history_sources):
        del source_id
        for tau in generator["history_tau_over_t0"]:
            memory = 1.0 - math.exp(-age_over_t0 / float(tau))
            centered_memory = 2.0 * memory - 1.0
            values.append(centered_memory * source)
    result = np.asarray(values, dtype=float)
    if result.shape != (96,) or not np.all(np.isfinite(result)):
        raise GravityItem47Error("operator evaluation did not produce 96 finite scalars")
    return result


def _age_ratio(redshift: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    cosmology = config["fiducial_cosmology"]
    om = float(cosmology["omega_matter"])
    ol = float(cosmology["omega_lambda"])
    scale = math.sqrt(ol / om)
    z = np.asarray(redshift, dtype=float)
    return np.arcsinh(scale / np.power(1.0 + z, 1.5)) / math.asinh(scale)


def _parse_tsv(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t"))


def _shape_by_object(root: Path, arrays: Mapping[str, Any]) -> dict[str, float]:
    predictor = _read_json(root / S4TM_PREDICTOR_PATH)
    result = {str(row["target"]): float(row["light_axis_ratio"]) for row in predictor["records"]}
    clash = {row["source_name"]: float(row["axis_ratio"]) for row in _parse_tsv(root / CLASH_SHAPE_PATH)}
    aliases = {
        "A209": "A209", "A383": "A383", "A611": "A611", "A2261": "A2261",
        "MACS0329": "0329-02", "MACS0416": "0416-24", "MACS0429": "0429-02",
        "MACS0647": "0647+70", "MACS0717": "0717+37", "MACS0744": "0744+39",
        "MACS1115": "1115+01", "MACS1149": "1149+22", "MACS1206": "1206-08",
        "MACS1720": "1720+35", "MACS1931": "1931-26", "MS2137": "MS2137",
        "RXJ1347": "1347-1145", "RXJ1532": "1532+30", "RXJ2129": "2129+0005",
        "RXJ2248": "2248-44",
    }
    result.update({name: clash[source] for name, source in aliases.items()})
    required = set(np.asarray(arrays["object"]).tolist())
    if required - set(result):
        raise GravityItem47Error(f"shape predictors missing for {sorted(required - set(result))}")
    if any(not 0.0 < result[name] <= 1.0 for name in required):
        raise GravityItem47Error("axis ratios must be in (0,1]")
    return result


def _profiles(arrays: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    populations = np.asarray(arrays["population"])
    names = np.asarray(arrays["object"])
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in sorted(set(names.tolist())):
        indices = np.flatnonzero(names == name)
        if populations[indices[0]] == "S4TM":
            size = float(arrays["size"][indices[0]])
            radius = size * np.logspace(-3.0, 2.0, 128)
            mass = _devaucouleurs_fraction(radius / size)
        else:
            order = indices[np.argsort(np.asarray(arrays["radius"])[indices])]
            radius = np.asarray(arrays["radius"])[order].astype(float)
            mass = np.maximum.accumulate(np.asarray(arrays["u"])[order] * np.square(radius))
        result[name] = (radius, mass)
    return result


def operator_bank_from_arrays(
    arrays: Mapping[str, Any], shapes: Mapping[str, float], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    profiles = _profiles(arrays)
    ages = _age_ratio(np.asarray(arrays["redshift"]), config)
    raw = np.empty((len(arrays["object"]), 96), dtype=float)
    for index, name in enumerate(arrays["object"]):
        profile_radius, profile_mass = profiles[str(name)]
        raw[index] = _operator_raw_values(
            profile_radius,
            profile_mass,
            radius=float(arrays["radius"][index]),
            baryonic_size=float(arrays["size"][index]),
            u=float(arrays["u"][index]),
            axis_ratio=float(shapes[str(name)]),
            age_over_t0=float(ages[index]),
            config=config,
        )
    coordinate = 0.5 + 0.5 * raw / (1.0 + np.abs(raw))
    if not np.all(np.isfinite(coordinate)) or np.any(coordinate <= 0.0) or np.any(coordinate >= 1.0):
        raise GravityItem47Error("operator coordinate map failed")
    return raw, coordinate


def build_operator_features_from_sources(
    item44: Mapping[str, Any],
    item46: Mapping[str, Any],
    shapes: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rows44 = item44["records"]
    rows46 = item46["records"]
    if len(rows44) != len(rows46):
        raise GravityItem47Error("Item 44/46 source row counts differ")
    arrays = {
        "population": np.asarray([row["population"] for row in rows44]),
        "object": np.asarray([row["object"] for row in rows44]),
        "fold": np.asarray([int(row["fold"]) for row in rows44]),
        "radius": np.asarray([float(row["radius_kpc"]) for row in rows44]),
        "size": np.asarray([float(row["baryonic_size_kpc"]) for row in rows44]),
        "redshift": np.asarray([float(row["redshift"]) for row in rows44]),
        "u": np.asarray([float(row["u"]) for row in rows44]),
    }
    for index, (left, right) in enumerate(zip(rows44, rows46, strict=True)):
        if int(right["source_row_index"]) != index or left["population"] != right["population"] or left["object"] != right["object"]:
            raise GravityItem47Error("Item 44/46 row alignment changed")
    raw, bank = operator_bank_from_arrays(arrays, shapes, config)
    records = [
        {
            "source_row_index": index,
            "population": str(arrays["population"][index]),
            "object": str(arrays["object"][index]),
            "fold": int(arrays["fold"][index]),
            "axis_ratio": float(shapes[str(arrays["object"][index])]),
            "raw_operator_values": [float(value) for value in raw[index]],
            "operator_coordinates": [float(value) for value in bank[index]],
        }
        for index in range(len(arrays["object"]))
    ]
    hashes = [
        hashlib.sha256(np.round(bank[:, index], 12).astype("<f8").tobytes()).hexdigest()
        for index in range(bank.shape[1])
    ]
    lineage = [
        {
            "population": row["population"], "object": row["object"], "fold": row["fold"],
            "radius_kpc": row["radius_kpc"], "baryonic_size_kpc": row["baryonic_size_kpc"],
            "redshift": row["redshift"], "u": row["u"], "axis_ratio": shapes[str(row["object"])],
        }
        for row in rows44
    ]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item47-operator-features-1.0",
            "item": 47,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_blind_source_lineage_sha256": _sha256_bytes(_canonical_bytes(lineage)),
            "response_fields_read_by_feature_builder": [],
            "response_values_used": 0,
            "records": records,
            "counts": {
                "s4tm_lenses": int(np.sum(arrays["population"] == "S4TM")),
                "clash_clusters": len(set(arrays["object"][arrays["population"] == "CLASH"].tolist())),
                "clash_points": int(np.sum(arrays["population"] == "CLASH")),
                "total_points": len(records), "operator_recipes": bank.shape[1],
                "sealed_confirmation_rows": 0, "paid_model_calls": 0,
            },
            "dataset_behavior": {
                "unique_operator_coordinate_hashes": len(set(hashes)),
                "duplicate_symbolic_recipes_on_development_predictors": len(hashes) - len(set(hashes)),
                "operator_coordinate_sha256": hashes,
            },
            "profile_contract": config["profile_contract"],
            "operator_catalog": operator_catalog(config),
            "lineage": config["data_roles"],
        }
    )


def build_operator_features(root: Path) -> dict[str, Any]:
    config = load_config(root)
    item44 = _read_json(root / ITEM44_FEATURE_PATH)
    item46 = _read_json(root / "runs/gravity/roadmap/item-46-dimensionless-generator-v1-source/dimensionless-features.json")
    dummy_arrays = {"object": np.asarray([row["object"] for row in item44["records"]])}
    shapes = _shape_by_object(root, dummy_arrays)
    return build_operator_features_from_sources(item44, item46, shapes, config)


def write_operator_features(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "feature_receipt")
    _write_json(path, build_operator_features(root))
    return path


def _evaluation_arrays(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    config46 = _load_item46_config(root)
    arrays = _item46_evaluation_arrays(root, config46)
    feature = _read_json(_source_path(root, config, "feature_receipt"))
    rows = feature["records"]
    if len(rows) != len(arrays["target"]):
        raise GravityItem47Error("operator feature row count changed")
    shapes: dict[str, float] = {}
    for index, row in enumerate(rows):
        if int(row["source_row_index"]) != index or row["population"] != arrays["population"][index] or row["object"] != arrays["object"][index]:
            raise GravityItem47Error("operator feature/source alignment changed")
        shapes[str(row["object"])] = float(row["axis_ratio"])
    arrays["operator_bank"] = np.asarray([row["operator_coordinates"] for row in rows]).T
    return arrays, shapes


def _candidate_subset(candidates: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in candidates.items()}


def _predict(candidate_id: int, arrays: Mapping[str, Any], config: Mapping[str, Any], *, bank_key: str) -> np.ndarray:
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    recipe = candidate_id // per_recipe
    local = candidate_id % per_recipe
    row = {
        "amplitude_index": np.asarray([(local // 256) % 16]),
        "exponent_index": np.asarray([(local // 16) % 16]),
        "transition_index": np.asarray([local % 16]),
    }
    amplitude, exponent, transition = _candidate_parameters(row, config)
    h = np.asarray(arrays[bank_key])[recipe]
    multiplier = 1.0 + amplitude[0] * np.power(arrays["u"], -exponent[0]) / (
        1.0 + arrays["u"] / transition[0]
    ) * (0.05 + 0.95 * h)
    return arrays["base"] + np.log10(multiplier)


def _fixed_oof(fold_ids: Mapping[int, int], arrays: Mapping[str, Any], config: Mapping[str, Any], bank_key: str) -> np.ndarray:
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _predict(candidate_id, arrays, config, bank_key=bank_key)[test]
    return prediction


def _item45_oof(root: Path, arrays: Mapping[str, Any]) -> tuple[np.ndarray, dict[int, int]]:
    config45 = _load_item45_config(root)
    evaluation = _read_json(root / ITEM45_EVALUATION_PATH)
    fold_ids = {int(row["fold"]): int(row["selected_interaction"]["candidate_id"]) for row in evaluation["fold_ledger"]}
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _item45_predict(candidate_id, arrays, config45, bank_key="interaction_bank")[test]
    return prediction, fold_ids


def _item46_oof(root: Path, arrays: Mapping[str, Any]) -> tuple[np.ndarray, dict[int, int]]:
    config46 = _load_item46_config(root)
    evaluation = _read_json(root / ITEM46_EVALUATION_PATH)
    fold_ids = {int(row["fold"]): int(row["selected_pi"]["candidate_id"]) for row in evaluation["fold_ledger"]}
    prediction = np.empty(len(arrays["target"]), dtype=float)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _item46_predict(candidate_id, arrays, config46, bank_key="pi_bank")[test]
    return prediction, fold_ids


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    arrays, shapes = _evaluation_arrays(root, config)
    admitted, admission = admissible_candidates(config)
    local_candidates = _candidate_subset(admitted, np.asarray(admitted["recipe"]) < 16)
    scale_candidates = _candidate_subset(admitted, np.asarray(admitted["recipe"]) == 0)
    scale_arrays = dict(arrays)
    scale_arrays["scale_free_bank"] = np.ones((96, len(arrays["target"])))
    candidate_oof = np.empty(len(arrays["target"]), dtype=float)
    local_oof = np.empty(len(arrays["target"]), dtype=float)
    fold_candidate: dict[int, int] = {}
    fold_local: dict[int, int] = {}
    fold_scale: dict[int, int] = {}
    ledger = []
    backends: set[str] = set()
    evaluations = 0
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        candidate_id, train_loss, backend, count = _best_candidate(admitted, arrays, train, config, bank_key="operator_bank")
        local_id, local_loss, local_backend, local_count = _best_candidate(local_candidates, arrays, train, config, bank_key="operator_bank")
        scale_id, scale_loss, scale_backend, scale_count = _best_candidate(scale_candidates, scale_arrays, train, config, bank_key="scale_free_bank")
        candidate_oof[test] = _predict(candidate_id, arrays, config, bank_key="operator_bank")[test]
        local_oof[test] = _predict(local_id, arrays, config, bank_key="operator_bank")[test]
        fold_candidate[fold] = candidate_id
        fold_local[fold] = local_id
        fold_scale[fold] = scale_id
        evaluations += count + local_count + scale_count
        backends.update((backend, local_backend, scale_backend))
        ledger.append(
            {
                "fold": fold,
                "selected_operator": decode_candidate(candidate_id, config),
                "operator_training_balanced_loss": train_loss,
                "selected_local_control": decode_candidate(local_id, config),
                "local_training_balanced_loss": local_loss,
                "selected_scale_free_candidate_id": scale_id,
                "scale_free_training_balanced_loss": scale_loss,
                "heldout_s4tm_objects": sorted(set(arrays["object"][test & (arrays["population"] == "S4TM")].tolist())),
                "heldout_clash_objects": sorted(set(arrays["object"][test & (arrays["population"] == "CLASH")].tolist())),
            }
        )
    scale_oof = _fixed_oof(fold_scale, scale_arrays, config, "scale_free_bank")
    all_rows = np.ones(len(arrays["target"]), dtype=bool)
    selected_id, selected_loss, backend, count = _best_candidate(admitted, arrays, all_rows, config, bank_key="operator_bank")
    selected_local, selected_local_loss, local_backend, local_count = _best_candidate(local_candidates, arrays, all_rows, config, bank_key="operator_bank")
    evaluations += count + local_count
    backends.update((backend, local_backend))
    cpu_loss = _score(arrays, _predict(selected_id, arrays, config, bank_key="operator_bank"))["balanced_loss"]
    cpu_gpu_difference = abs(float(cpu_loss) - selected_loss)
    if cpu_gpu_difference > float(config["evaluation"]["cpu_gpu_tolerance"]):
        raise GravityItem47Error("CPU/GPU selected loss cross-check failed")
    item45_oof, fold_item45 = _item45_oof(root, arrays)
    item46_oof, fold_item46 = _item46_oof(root, arrays)
    item44_oof, fold_item44 = _item44_oof(root, arrays)
    scores = {
        "operator_generator": _score(arrays, candidate_oof),
        "item45_universal_interaction": _score(arrays, item45_oof),
        "item46_dimensionless_generator": _score(arrays, item46_oof),
        "item44_scale_hierarchy": _score(arrays, item44_oof),
        "matched_local_operator": _score(arrays, local_oof),
        "matched_scale_free": _score(arrays, scale_oof),
        "baryonic_newton": _score(arrays, arrays["base"]),
        "mond_rar": _score(arrays, arrays["base"] + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(arrays["u"]))))),
        "ordinary_ridge": _score(arrays, _ordinary_crossfit(arrays, config)),
    }
    controls = tuple(name for name in scores if name != "operator_generator")
    strongest = min(controls, key=lambda name: scores[name]["balanced_loss"])
    candidate_objects = scores["operator_generator"]["object_losses"]
    control_objects = scores[strongest]["object_losses"]
    object_keys = sorted(candidate_objects)
    diff = np.asarray([control_objects[key] - candidate_objects[key] for key in object_keys])
    raw_counterexample = diff < 0.0
    stable_counterexample = raw_counterexample.copy()
    systematic_scores: dict[str, Any] = {}
    config44 = _read_json(root / "configs/gravity_item44_scale_hierarchy_v1.json")
    config45 = _load_item45_config(root)
    config46 = _load_item46_config(root)
    from sigma_theory_compiler.gravity_item44_scale_hierarchy import _predict as item44_predict

    for variant_name, population, shift in config["evaluation"]["mass_scale_variants"]:
        varied = _item45_variant_arrays(arrays, str(population), float(shift), config45)
        varied["pi_bank"] = (
            1.0 / (1.0 + np.abs(_item46_physical_log_values(varied, config46) @ np.asarray(_item46_pi_vectors(config46), dtype=float).T))
        ).T
        varied["operator_bank"] = operator_bank_from_arrays(varied, shapes, config)[1].T
        varied_scale = dict(varied)
        varied_scale["scale_free_bank"] = np.ones((96, len(varied["target"])))
        candidate_variant = _fixed_oof(fold_candidate, varied, config, "operator_bank")
        local_variant = _fixed_oof(fold_local, varied, config, "operator_bank")
        scale_variant = _fixed_oof(fold_scale, varied_scale, config, "scale_free_bank")
        item45_variant = np.empty(len(varied["target"]), dtype=float)
        item46_variant = np.empty(len(varied["target"]), dtype=float)
        item44_variant = np.empty(len(varied["target"]), dtype=float)
        for fold in range(int(config["evaluation"]["outer_folds"])):
            test = varied["fold"] == fold
            item45_variant[test] = _item45_predict(fold_item45[fold], varied, config45, bank_key="interaction_bank")[test]
            item46_variant[test] = _item46_predict(fold_item46[fold], varied, config46, bank_key="pi_bank")[test]
            item44_variant[test] = item44_predict(fold_item44[fold], varied, config44)[test]
        variants = {
            "operator_generator": _score(varied, candidate_variant),
            "item45_universal_interaction": _score(varied, item45_variant),
            "item46_dimensionless_generator": _score(varied, item46_variant),
            "item44_scale_hierarchy": _score(varied, item44_variant),
            "matched_local_operator": _score(varied, local_variant),
            "matched_scale_free": _score(varied, scale_variant),
            "baryonic_newton": _score(varied, varied["base"]),
            "mond_rar": _score(varied, varied["base"] + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(varied["u"]))))),
            "ordinary_ridge": _score(varied, _ordinary_crossfit(varied, config)),
        }
        systematic_scores[str(variant_name)] = {
            "operator_generator": variants["operator_generator"],
            "strongest_control_name": strongest,
            "strongest_control": variants[strongest],
        }
        for index, key in enumerate(object_keys):
            stable_counterexample[index] &= variants["operator_generator"]["object_losses"][key] > variants[strongest]["object_losses"][key]
    leave_one = [float(np.mean(np.delete(diff, index))) for index in range(len(diff))]
    trim_count = max(1, int(len(diff) * float(config["evaluation"]["robust_trim_fraction"])))
    trimmed = np.sort(diff)[trim_count:-trim_count]
    improvement = 100.0 * (scores[strongest]["balanced_loss"] - scores["operator_generator"]["balanced_loss"]) / scores[strongest]["balanced_loss"]
    policy_report = {
        "evidence_kind": "empirical", "evaluable_objects": len(object_keys),
        "raw_counterexample_count": int(np.sum(raw_counterexample)),
        "quality_verified_counterexample_count": int(np.sum(raw_counterexample)),
        "uncertainty_resolved_counterexample_count": int(np.sum(stable_counterexample)),
        "independent_failure_strata": 0, "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement, "quality_gate_passed": False,
        "strongest_baseline_failed": bool(improvement <= 0.0),
        "leave_one_changes_sign": bool((min(leave_one) <= 0.0) != (float(np.mean(diff)) <= 0.0)),
        "trim_changes_sign": bool((float(np.mean(trimmed)) <= 0.0) != (float(np.mean(diff)) <= 0.0)),
        "object_level_records_preserved": True, "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = assess_counterexample_evidence(policy_report, load_counterexample_policy(root / POLICY_PATH))
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item47-joint-evaluation-1.0", "item": 47,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selected_candidate": decode_candidate(selected_id, config),
            "selected_full_data_balanced_training_loss": selected_loss,
            "selected_local_control": decode_candidate(selected_local, config),
            "selected_local_full_data_balanced_training_loss": selected_local_loss,
            "fold_ledger": ledger, "scores": scores, "strongest_control": strongest,
            "aggregate_improvement_percent": improvement, "paired_sign_flip_p": _paired_p(diff, config),
            "robustness": {
                "leave_one_min_mean_control_minus_candidate_loss": min(leave_one),
                "leave_one_max_mean_control_minus_candidate_loss": max(leave_one),
                "trimmed_mean_control_minus_candidate_loss": float(np.mean(trimmed)),
            },
            "counterexamples": [
                {"object": key, "raw_counterexample": bool(raw_counterexample[index]), "uncertainty_resolved_counterexample": bool(stable_counterexample[index])}
                for index, key in enumerate(object_keys)
            ],
            "systematic_scores": systematic_scores,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": policy,
            "compute": {
                "backends": sorted(backends), "candidate_point_fold_evaluations": evaluations,
                "cpu_gpu_selected_loss_absolute_difference": cpu_gpu_difference, "admission": admission,
            },
            "counts": {"s4tm_lenses": 28, "clash_clusters": 20, "clash_points": 84, "sealed_confirmation_rows": 0, "post_evaluation_candidate_cells": 0, "paid_model_calls": 0},
            "limitations": [
                "All responses were exposed before Item 47; grouped cross-validation cannot create fresh confirmation.",
                "Every operator is reduced to a scalar weak-field coordinate and is not a covariant field operator or action.",
                "S4TM uses an analytic projected de Vaucouleurs profile; CLASH exterior integrals stop at the last published point and can be identically zero there.",
                "The tensor lane uses only axis-ratio amplitude, not orientation-resolved tensor lensing.",
                "The history lane has no measured past baryonic state and uses a constant-current-state closure.",
                "Four global mass shifts do not exhaust baryonic, profile, shape, lens-model, or selection uncertainty and cannot prune a family.",
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
    systematics = evaluation["systematic_scores"]
    gates = {
        "beats_item45_s4tm": scores["operator_generator"]["populations"]["S4TM"]["loss"] < scores["item45_universal_interaction"]["populations"]["S4TM"]["loss"],
        "beats_item45_clash": scores["operator_generator"]["populations"]["CLASH"]["loss"] < scores["item45_universal_interaction"]["populations"]["CLASH"]["loss"],
        "beats_local_operator_balanced": scores["operator_generator"]["balanced_loss"] < scores["matched_local_operator"]["balanced_loss"],
        "beats_ordinary_ridge_balanced": scores["operator_generator"]["balanced_loss"] < scores["ordinary_ridge"]["balanced_loss"],
        "paired_p_passes": float(evaluation["paired_sign_flip_p"]) <= float(config["gates"]["paired_p_maximum"]),
        "leave_one_stable": float(evaluation["robustness"]["leave_one_min_mean_control_minus_candidate_loss"]) > 0.0,
        "trim_stable": float(evaluation["robustness"]["trimmed_mean_control_minus_candidate_loss"]) > 0.0,
        "mass_scale_audits_not_all_reverse": any(value["operator_generator"]["balanced_loss"] < value["strongest_control"]["balanced_loss"] for value in systematics.values()),
        "confirmation_rows_zero": int(evaluation["counts"]["sealed_confirmation_rows"]) == 0,
        "post_evaluation_candidates_zero": int(evaluation["counts"]["post_evaluation_candidate_cells"]) == 0,
        "fresh_confirmation_available": False,
    }
    empirical_lead = all(
        gates[key]
        for key in (
            "beats_item45_s4tm", "beats_item45_clash", "beats_local_operator_balanced",
            "beats_ordinary_ridge_balanced", "paired_p_passes", "leave_one_stable",
            "trim_stable", "mass_scale_audits_not_all_reverse",
        )
    )
    decision = "RETROSPECTIVE_ITEM47_OPERATOR_LEAD_REQUIRES_FRESH_TEST" if empirical_lead else "NONPROMOTED_ITEM47_OPERATOR_RESULT_RETAINED"
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item47-operator-generator-result-1.0", "item": 47,
            "goal": "GRAVITY_ROADMAP_ITEM_47_OPERATOR_GENERATOR", "decision": decision,
            "selected_candidate": evaluation["selected_candidate"], "scores": scores,
            "strongest_control": evaluation["strongest_control"],
            "aggregate_improvement_percent": evaluation["aggregate_improvement_percent"],
            "paired_sign_flip_p": evaluation["paired_sign_flip_p"], "gates": gates,
            "counterexample_policy_assessment": evaluation["counterexample_policy_assessment"],
            "counts": {
                "raw_candidates": candidate["raw_candidates"], "admitted_candidates": candidate["admitted_candidates"],
                "symbolic_operator_recipes": candidate["symbolic_operator_recipes"],
                "unique_operator_behaviors_on_development_data": features["dataset_behavior"]["unique_operator_coordinate_hashes"],
                "s4tm_lenses": features["counts"]["s4tm_lenses"], "clash_clusters": features["counts"]["clash_clusters"],
                "clash_points": features["counts"]["clash_points"],
                "candidate_point_fold_evaluations": evaluation["compute"]["candidate_point_fold_evaluations"],
                "sealed_confirmation_rows": 0, "post_evaluation_candidate_cells": 0, "paid_model_calls": 0,
            },
            "source_bindings": {
                "config": {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)},
                "candidate_manifest": {"path": str(_source_path(root, config, "candidate_manifest").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "candidate_manifest"))},
                "exposure_manifest": {"path": str(_source_path(root, config, "exposure_manifest").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "exposure_manifest"))},
                "features": {"path": str(_source_path(root, config, "feature_receipt").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "feature_receipt"))},
                "evaluation": {"path": str(_source_path(root, config, "evaluation_result").relative_to(root)), "sha256": _sha256_file(_source_path(root, config, "evaluation_result"))},
            },
            "claims": {
                "roadmap_item_47_complete": True, "fresh_confirmation_completed": False,
                "operator_law_established": False, "alternative_to_gr_established": False,
                "dark_matter_eliminated": False, "historical_novelty_established": False,
                "covariant_theory_established": False, "measured_history_tested": False,
                "formula_family_pruned": False, "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": "Preserve every operator, selected clue, equivalence, and mismatch; require fresh unchanged replication for confirmation, then advance to Item 48 action generation.",
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
        "feature_receipt": _read_json(_source_path(root, config, "feature_receipt")) == build_operator_features(root),
        "evaluation_result": _read_json(_source_path(root, config, "evaluation_result")) == build_evaluation_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"])) == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write-freeze", "write-features", "evaluate", "aggregate", "replay"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write-freeze":
        result: Any = [str(path) for path in write_freeze_manifests(root)]
    elif args.command == "write-features":
        result = str(write_operator_features(root))
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
