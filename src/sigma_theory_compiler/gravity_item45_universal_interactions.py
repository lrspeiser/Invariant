"""Item 45 response-blind universal interaction search across S4TM and CLASH."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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
from sigma_theory_compiler.gravity_item44_scale_hierarchy import (
    _arrays as _item44_arrays,
    _mass_scale_variant as _item44_mass_scale_variant,
    _predict as _item44_predict,
    load_config as _load_item44_config,
)

CONFIG_PATH = Path("configs/gravity_item45_universal_interactions_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM44_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-scale-features.json"
)
ITEM44_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-evaluation-result.json"
)
AXES = ("geometry", "density", "gradient", "time", "environment", "field")


class GravityItem45Error(RuntimeError):
    """Raised when an Item 45 freeze, leakage, or evaluation gate fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item45-universal-interactions-config-1.0"
        or int(config.get("item", -1)) != 45
    ):
        raise GravityItem45Error("unexpected Item 45 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem45Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem45Error("Item 45 scientific freeze is not bound to a commit")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem45Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-44-scale-hierarchy-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem45Error("Item 44 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem45Error("Item 44 decision binding changed")
    if int(predecessor["selected_candidate"]["candidate_id"]) != int(
        required["selected_candidate_id"]
    ):
        raise GravityItem45Error("Item 44 selected candidate binding changed")
    discovery = config["discovery_policy"]
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem45Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem45Error("count-only rejection entered Item 45")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem45Error("finite empirical family pruning entered Item 45")
    policy = load_counterexample_policy(root / POLICY_PATH)
    if policy["empirical_evidence"]["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem45Error("executable counterexample policy changed")
    generator = config["candidate_generator"]
    if (
        int(generator["raw_candidate_cells"]) != 262144
        or int(generator["interaction_recipes"]) != 64
        or int(generator["cells_per_recipe"]) != 4096
        or int(generator["cells_per_niche"]) != 65536
    ):
        raise GravityItem45Error("candidate capacity changed")
    if len(generator["niches"]) != 4 or int(generator["post_evaluation_cells"]) != 0:
        raise GravityItem45Error("interaction niche boundary changed")
    if tuple(config["primitive_contract"]["axes"]) != AXES:
        raise GravityItem45Error("primitive axis contract changed")
    if int(config["primitive_contract"]["response_values_allowed_in_feature_synthesis"]) != 0:
        raise GravityItem45Error("responses entered feature synthesis")
    if not bool(config["scope"]["all_empirical_responses_already_exposed"]):
        raise GravityItem45Error("retrospective exposure disclosure changed")
    if bool(config["scope"]["fresh_confirmation_claim_allowed"]):
        raise GravityItem45Error("fresh confirmation entered retrospective Item 45")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem45Error("paid calls entered Item 45")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def recipe_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    generator = config["candidate_generator"]
    recipes: list[dict[str, Any]] = []
    blocks = (
        ("pair_products", "product", generator["pair_product_terms"]),
        ("signed_contrasts", "contrast", generator["signed_contrast_terms"]),
        ("gated_resonances", "gated", generator["gated_resonance_terms"]),
        ("triple_closures", "product", generator["triple_closure_terms"]),
    )
    for niche_id, (name, operator, terms) in enumerate(blocks):
        creativity = generator["niches"][niche_id]["creativity_label"]
        if len(terms) != 16:
            raise GravityItem45Error(f"niche {name} does not contain 16 recipes")
        for operands in terms:
            recipe_id = len(recipes)
            if operator == "contrast":
                expression = f"{operands[0]}-{operands[1]}"
            elif operator == "gated":
                expression = f"{operands[0]}*tanh(2*{operands[1]})"
            else:
                expression = "*".join(operands)
            recipes.append(
                {
                    "recipe_id": recipe_id,
                    "niche_id": niche_id,
                    "niche": name,
                    "operator": operator,
                    "operands": list(operands),
                    "interaction_expression": expression,
                    "coordinate_expression": f"H=0.5+0.5*tanh(2*({expression}))",
                    "creativity_label": creativity,
                }
            )
    if len(recipes) != 64:
        raise GravityItem45Error("interaction catalog does not contain 64 recipes")
    return recipes


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    candidate_id = np.arange(total, dtype=np.int64)
    recipe = (candidate_id // per_recipe).astype(np.int8)
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
    h = np.asarray(gate["probe_interaction_coordinates"], dtype=np.float64)
    keep_parts: list[np.ndarray] = []
    rejection: Counter[str] = Counter()
    signatures: set[bytes] = set()
    for begin in range(0, len(raw["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(raw["candidate_id"]))
        rows = {key: value[begin:end] for key, value in raw.items()}
        amplitude, exponent, transition = _candidate_parameters(rows, config)
        multiplier = 1.0 + amplitude[:, None, None] * np.power(
            u[None, None, :], -exponent[:, None, None]
        ) / (1.0 + u[None, None, :] / transition[:, None, None]) * (
            0.05 + 0.95 * h[None, :, None]
        )
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
        monotone = material & np.all(
            np.diff(multiplier, axis=2)
            <= float(gate["monotone_nonincreasing_tolerance"]),
            axis=(1, 2),
        )
        rejection["nonfinite"] += int(np.sum(~finite))
        rejection["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection["no_local_limit"] += int(np.sum(bounded & ~local))
        rejection["immaterial_low_acceleration"] += int(np.sum(local & ~material))
        rejection["nonmonotone"] += int(np.sum(material & ~monotone))
        selected = np.flatnonzero(monotone)
        keep_parts.append(selected + begin)
        signature = np.round(
            np.log10(multiplier[selected]), int(gate["behavior_signature_decimals"])
        )
        for row in signature:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    admitted = {key: value[keep] for key, value in raw.items()}
    return admitted, {
        "raw_candidates": len(raw["candidate_id"]),
        "admitted_candidates": len(keep),
        "rejected_candidates": len(raw["candidate_id"]) - len(keep),
        "admitted_by_niche": {
            str(niche): int(np.sum(np.asarray(admitted["recipe"]) // 16 == niche))
            for niche in range(4)
        },
        "generic_formula_behavior_classes": len(signatures),
        "symbolic_interaction_recipes": 64,
        "rejection_counts_nonexclusive": dict(sorted(rejection.items())),
    }


def decode_candidate(
    candidate_id: int, config: Mapping[str, Any], *, main_effect: bool = False
) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem45Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition = _candidate_parameters(row, config)
    recipe_id = int(row["recipe"][0])
    if main_effect:
        transforms = config["candidate_generator"]["matched_main_effect_control"]["transforms"]
        axis = AXES[recipe_id % len(AXES)]
        transform = transforms[recipe_id // len(AXES)]
        recipe = {
            "recipe_id": recipe_id,
            "niche": "matched_unary_main_effect_control",
            "interaction_expression": f"{transform}({axis})",
            "coordinate_expression": f"H=0.5+0.5*tanh(2*{transform}({axis}))",
            "creativity_label": "known_unary_feature_transformation_control",
        }
    else:
        recipe = recipe_catalog(config)[recipe_id]
    return {
        **recipe,
        "candidate_id": candidate_id,
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
            "schema_version": "invariant-gravity-item45-candidate-manifest-1.0",
            "item": 45,
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
            "recipe_catalog": recipe_catalog(config),
            "matched_main_effect_candidate_cells": int(
                config["candidate_generator"]["matched_main_effect_control"]["candidate_cells"]
            ),
            "claim_boundaries": [
                "products and contrasts are standard algebraic feature engineering, not new formulas by themselves",
                "gated and triple recipes are potentially new observational syntheses only; historical novelty is untested",
                "behavioral difference on this finite dataset cannot establish mathematical or physical novelty",
                "the retrospective data can generate a lead but cannot confirm one",
            ],
        }
    )


def build_exposure_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item45-exposure-manifest-1.0",
            "item": 45,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "datasets": [
                {
                    "id": "S4TM_ITEM43_EXPLORATION",
                    "objects": 28,
                    "response_status": "already exposed before Item 45",
                    "role": "retrospective interaction development",
                },
                {
                    "id": "CLASH_ACCELERATION",
                    "objects": 20,
                    "points": 84,
                    "response_status": "already exposed before Item 45",
                    "role": "retrospective interaction development",
                },
            ],
            "sealed_data": {
                "item43_s4tm_confirmation_lenses": 7,
                "access_authorized": False,
                "response_rows_read": 0,
            },
            "rules": [
                "no result may be described as fresh confirmation",
                "feature synthesis is response-blind and formula selection is whole-object cross-validated",
                "preserve every mismatch; one counterexample and counterexample count alone are never vetoes",
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


def _cosmic_age_ratio(redshift: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    cosmology = config["fiducial_cosmology"]
    om = float(cosmology["omega_matter"])
    ol = float(cosmology["omega_lambda"])
    z = np.asarray(redshift, dtype=np.float64)
    scale = math.sqrt(ol / om)
    age = np.arcsinh(scale / np.power(1.0 + z, 1.5))
    age0 = math.asinh(scale)
    return age / age0


def _devaucouleurs_enclosed_slope(radius_over_re: np.ndarray) -> np.ndarray:
    x = 7.669249442500804 * np.power(np.maximum(radius_over_re, 1e-300), 0.25)
    series = np.zeros_like(x)
    term = np.ones_like(x)
    for k in range(8):
        if k > 0:
            term = term * x / k
        series += term
    enclosed = 1.0 - np.exp(-x) * series
    derivative = np.exp(-x) * np.power(x, 8.0) / (4.0 * math.factorial(7))
    return derivative / np.maximum(enclosed, 1e-300)


def _response_blind_source_arrays(feature_doc: Mapping[str, Any]) -> dict[str, Any]:
    rows = feature_doc["records"]
    return {
        "population": np.asarray([str(row["population"]) for row in rows]),
        "object": np.asarray([str(row["object"]) for row in rows]),
        "fold": np.asarray([int(row["fold"]) for row in rows]),
        "radius": np.asarray([float(row["radius_kpc"]) for row in rows]),
        "size": np.asarray([float(row["baryonic_size_kpc"]) for row in rows]),
        "redshift": np.asarray([float(row["redshift"]) for row in rows]),
        "u": np.asarray([float(row["u"]) for row in rows]),
        "horizon": np.asarray(
            [float(row["scale_values_kpc"]["horizon_radius_kpc"]) for row in rows]
        ),
        "schwarzschild": np.asarray(
            [float(row["scale_values_kpc"]["schwarzschild_length_kpc"]) for row in rows]
        ),
    }


def primitive_coordinates(
    arrays: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    population = np.asarray(arrays["population"])
    names = np.asarray(arrays["object"])
    radius = np.asarray(arrays["radius"], dtype=np.float64)
    size = np.asarray(arrays["size"], dtype=np.float64)
    u = np.asarray(arrays["u"], dtype=np.float64)
    gradient = np.empty(len(radius), dtype=np.float64)
    s4 = population == "S4TM"
    gradient[s4] = _devaucouleurs_enclosed_slope(radius[s4] / size[s4])
    for name in sorted(set(names[population == "CLASH"].tolist())):
        indices = np.flatnonzero((population == "CLASH") & (names == name))
        order = indices[np.argsort(radius[indices])]
        log_radius = np.log(radius[order])
        log_mass = np.log(u[order]) + 2.0 * log_radius
        slope = np.gradient(log_mass, log_radius, edge_order=1)
        gradient[order] = slope
    raw = np.column_stack(
        (
            np.log10(radius / size),
            np.log10(u),
            gradient - 1.0,
            np.log10(1.0 / _cosmic_age_ratio(np.asarray(arrays["redshift"]), config)),
            np.log10(radius / np.asarray(arrays["horizon"])),
            np.log10(np.asarray(arrays["schwarzschild"]) / radius),
        )
    )
    scales = np.asarray(
        [float(config["primitive_contract"]["fixed_soft_scales"][axis]) for axis in AXES]
    )
    normalized = raw / (scales[None, :] + np.abs(raw))
    if not np.all(np.isfinite(normalized)) or np.any(np.abs(normalized) >= 1.0):
        raise GravityItem45Error("primitive normalization failed")
    return raw, normalized


def interaction_bank(
    normalized: np.ndarray, config: Mapping[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    index = {axis: i for i, axis in enumerate(AXES)}
    columns: list[np.ndarray] = []
    catalog = recipe_catalog(config)
    for recipe in catalog:
        values = [normalized[:, index[axis]] for axis in recipe["operands"]]
        if recipe["operator"] == "contrast":
            interaction = values[0] - values[1]
        elif recipe["operator"] == "gated":
            interaction = values[0] * np.tanh(2.0 * values[1])
        else:
            interaction = np.prod(np.stack(values), axis=0)
        columns.append(0.5 + 0.5 * np.tanh(2.0 * interaction))
    return np.column_stack(columns), catalog


def _unary_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return values
    if transform == "negative":
        return -values
    if transform == "absolute":
        return np.abs(values)
    if transform == "negative_absolute":
        return -np.abs(values)
    if transform == "square":
        return np.square(values)
    if transform == "negative_square":
        return -np.square(values)
    if transform == "cube":
        return np.power(values, 3.0)
    if transform == "tanh_double":
        return np.tanh(2.0 * values)
    if transform == "sine_half_pi":
        return np.sin(0.5 * math.pi * values)
    if transform == "signed_square_root":
        return np.sign(values) * np.sqrt(np.abs(values))
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise GravityItem45Error(f"unknown unary transform: {transform}")


def main_effect_bank(normalized: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    transforms = config["candidate_generator"]["matched_main_effect_control"]["transforms"]
    columns = []
    for recipe_id in range(64):
        axis = recipe_id % len(AXES)
        transform = transforms[recipe_id // len(AXES)]
        value = _unary_transform(normalized[:, axis], transform)
        columns.append(0.5 + 0.5 * np.tanh(2.0 * value))
    return np.column_stack(columns)


def build_interaction_features_from_item44(
    feature_doc: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    arrays = _response_blind_source_arrays(feature_doc)
    raw, normalized = primitive_coordinates(arrays, config)
    bank, catalog = interaction_bank(normalized, config)
    main = main_effect_bank(normalized, config)
    records = []
    for i in range(len(arrays["object"])):
        records.append(
            {
                "source_row_index": i,
                "population": str(arrays["population"][i]),
                "object": str(arrays["object"][i]),
                "fold": int(arrays["fold"][i]),
                "raw_primitives": {axis: float(raw[i, j]) for j, axis in enumerate(AXES)},
                "normalized_primitives": {
                    axis: float(normalized[i, j]) for j, axis in enumerate(AXES)
                },
                "interaction_coordinates": [float(value) for value in bank[i]],
                "main_effect_coordinates": [float(value) for value in main[i]],
            }
        )
    hashes = [
        hashlib.sha256(np.round(bank[:, i], 12).astype("<f8").tobytes()).hexdigest()
        for i in range(bank.shape[1])
    ]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item45-interaction-features-1.0",
            "item": 45,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_blind_source_lineage_sha256": _sha256_bytes(
                _canonical_bytes(
                    [
                        {
                            "population": row["population"],
                            "object": row["object"],
                            "fold": row["fold"],
                            "radius_kpc": row["radius_kpc"],
                            "baryonic_size_kpc": row["baryonic_size_kpc"],
                            "redshift": row["redshift"],
                            "u": row["u"],
                            "horizon_radius_kpc": row["scale_values_kpc"][
                                "horizon_radius_kpc"
                            ],
                            "schwarzschild_length_kpc": row["scale_values_kpc"][
                                "schwarzschild_length_kpc"
                            ],
                        }
                        for row in feature_doc["records"]
                    ]
                )
            ),
            "response_fields_read_by_feature_builder": [],
            "response_values_used": 0,
            "records": records,
            "counts": {
                "s4tm_lenses": int(np.sum(arrays["population"] == "S4TM")),
                "clash_clusters": len(set(arrays["object"][arrays["population"] == "CLASH"].tolist())),
                "clash_points": int(np.sum(arrays["population"] == "CLASH")),
                "total_points": len(records),
                "primitive_axes": len(AXES),
                "interaction_recipes": bank.shape[1],
                "matched_main_effect_recipes": main.shape[1],
                "sealed_confirmation_rows": 0,
                "paid_model_calls": 0,
            },
            "primitive_ranges": {
                axis: {
                    "raw_minimum": float(np.min(raw[:, j])),
                    "raw_maximum": float(np.max(raw[:, j])),
                    "normalized_minimum": float(np.min(normalized[:, j])),
                    "normalized_maximum": float(np.max(normalized[:, j])),
                }
                for j, axis in enumerate(AXES)
            },
            "dataset_behavior": {
                "unique_interaction_coordinate_hashes": len(set(hashes)),
                "recipe_coordinate_sha256": hashes,
            },
            "recipe_catalog": catalog,
            "lineage": config["data_roles"],
        }
    )


def build_interaction_features(root: Path) -> dict[str, Any]:
    config = load_config(root)
    feature_doc = _read_json(root / ITEM44_FEATURE_PATH)
    return build_interaction_features_from_item44(feature_doc, config)


def write_interaction_features(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "feature_receipt")
    _write_json(path, build_interaction_features(root))
    return path


def _evaluation_arrays(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = _read_json(root / ITEM44_FEATURE_PATH)
    feature_doc = _read_json(_source_path(root, config, "feature_receipt"))
    arrays = _item44_arrays(source)
    source_blind = _response_blind_source_arrays(source)
    arrays.update(source_blind)
    rows = feature_doc["records"]
    if len(rows) != len(arrays["target"]):
        raise GravityItem45Error("interaction feature row count changed")
    for i, row in enumerate(rows):
        if (
            int(row["source_row_index"]) != i
            or row["population"] != arrays["population"][i]
            or row["object"] != arrays["object"][i]
            or int(row["fold"]) != int(arrays["fold"][i])
        ):
            raise GravityItem45Error("interaction feature/source alignment changed")
    arrays["primitives"] = np.asarray(
        [[row["normalized_primitives"][axis] for axis in AXES] for row in rows]
    )
    arrays["interaction_bank"] = np.asarray(
        [row["interaction_coordinates"] for row in rows]
    ).T
    arrays["main_effect_bank"] = np.asarray(
        [row["main_effect_coordinates"] for row in rows]
    ).T
    return arrays


def _variant_arrays(
    arrays: Mapping[str, Any], population: str, shift_dex: float, config: Mapping[str, Any]
) -> dict[str, Any]:
    varied = _item44_mass_scale_variant(arrays, population, shift_dex)
    mask = varied["population"] == population
    varied["schwarzschild"][mask] *= 10.0**shift_dex
    raw, normalized = primitive_coordinates(varied, config)
    del raw
    varied["primitives"] = normalized
    varied["interaction_bank"] = interaction_bank(normalized, config)[0].T
    varied["main_effect_bank"] = main_effect_bank(normalized, config).T
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


def _candidate_subset(
    candidates: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in candidates.items()}


def _best_candidate(
    candidates: Mapping[str, np.ndarray],
    arrays: Mapping[str, Any],
    train: np.ndarray,
    config: Mapping[str, Any],
    *,
    bank_key: str,
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
    bank = xp.asarray(arrays[bank_key][:, indices])
    residual = xp.asarray(arrays["target"][indices] - arrays["base"][indices])
    sigma = xp.asarray(arrays["sigma"][indices])
    weights = xp.asarray(weights_np[indices])
    grids = config["candidate_generator"]["parameter_grids"]
    amplitude_grid = xp.asarray(grids["amplitude"])
    exponent_grid = xp.asarray(grids["acceleration_exponent"])
    transition_grid = xp.asarray(grids["transition_u"])
    best_loss = math.inf
    best_index = -1
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(candidates["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(candidates["candidate_id"]))
        recipe = xp.asarray(np.asarray(candidates["recipe"])[begin:end], dtype=xp.int64)
        aa = amplitude_grid[xp.asarray(np.asarray(candidates["amplitude_index"])[begin:end])]
        pp = exponent_grid[xp.asarray(np.asarray(candidates["exponent_index"])[begin:end])]
        tt = transition_grid[xp.asarray(np.asarray(candidates["transition_index"])[begin:end])]
        h = bank[recipe]
        multiplier = 1.0 + aa[:, None] * xp.power(u[None, :], -pp[:, None]) / (
            1.0 + u[None, :] / tt[:, None]
        ) * (0.05 + 0.95 * h)
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
    candidate_id: int,
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    bank_key: str,
) -> np.ndarray:
    raw = generate_raw_candidates(config)
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition = _candidate_parameters(row, config)
    h = np.asarray(arrays[bank_key])[int(row["recipe"][0])]
    multiplier = 1.0 + amplitude[0] * np.power(arrays["u"], -exponent[0]) / (
        1.0 + arrays["u"] / transition[0]
    ) * (0.05 + 0.95 * h)
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
    x = np.column_stack(
        (
            np.log10(arrays["u"]),
            arrays["primitives"],
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


def _item44_oof(
    root: Path, arrays: Mapping[str, Any]
) -> tuple[np.ndarray, dict[int, int]]:
    config44 = _load_item44_config(root)
    evaluation44 = _read_json(root / ITEM44_EVALUATION_PATH)
    fold_ids = {
        int(row["fold"]): int(row["selected_candidate"]["candidate_id"])
        for row in evaluation44["fold_ledger"]
    }
    prediction = np.empty(len(arrays["target"]), dtype=np.float64)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _item44_predict(candidate_id, arrays, config44)[test]
    return prediction, fold_ids


def _fixed_oof(
    fold_ids: Mapping[int, int],
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
    bank_key: str,
) -> np.ndarray:
    prediction = np.empty(len(arrays["target"]), dtype=np.float64)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _predict(candidate_id, arrays, config, bank_key=bank_key)[test]
    return prediction


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    arrays = _evaluation_arrays(root, config)
    admitted, admission = admissible_candidates(config)
    scale_free = _candidate_subset(admitted, np.asarray(admitted["recipe"]) == 0)
    scale_free_arrays = dict(arrays)
    scale_free_arrays["scale_free_bank"] = np.ones((64, len(arrays["target"])))
    candidate_oof = np.empty(len(arrays["target"]), dtype=np.float64)
    main_oof = np.empty(len(arrays["target"]), dtype=np.float64)
    scale_free_oof = np.empty(len(arrays["target"]), dtype=np.float64)
    ledger: list[dict[str, Any]] = []
    fold_candidate: dict[int, int] = {}
    fold_main: dict[int, int] = {}
    fold_scale: dict[int, int] = {}
    backends: set[str] = set()
    evaluations = 0
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        candidate_id, train_loss, backend, count = _best_candidate(
            admitted, arrays, train, config, bank_key="interaction_bank"
        )
        main_id, main_loss, main_backend, main_count = _best_candidate(
            admitted, arrays, train, config, bank_key="main_effect_bank"
        )
        scale_id, scale_loss, scale_backend, scale_count = _best_candidate(
            scale_free, scale_free_arrays, train, config, bank_key="scale_free_bank"
        )
        candidate_oof[test] = _predict(
            candidate_id, arrays, config, bank_key="interaction_bank"
        )[test]
        main_oof[test] = _predict(main_id, arrays, config, bank_key="main_effect_bank")[test]
        scale_free_oof[test] = _predict(
            scale_id, scale_free_arrays, config, bank_key="scale_free_bank"
        )[test]
        fold_candidate[fold] = candidate_id
        fold_main[fold] = main_id
        fold_scale[fold] = scale_id
        evaluations += count + main_count + scale_count
        backends.update((backend, main_backend, scale_backend))
        ledger.append(
            {
                "fold": fold,
                "selected_interaction": decode_candidate(candidate_id, config),
                "interaction_training_balanced_loss": train_loss,
                "selected_main_effect": decode_candidate(main_id, config, main_effect=True),
                "main_effect_training_balanced_loss": main_loss,
                "selected_scale_free": decode_candidate(scale_id, config, main_effect=True),
                "scale_free_training_balanced_loss": scale_loss,
                "heldout_s4tm_objects": sorted(
                    set(arrays["object"][test & (arrays["population"] == "S4TM")].tolist())
                ),
                "heldout_clash_objects": sorted(
                    set(arrays["object"][test & (arrays["population"] == "CLASH")].tolist())
                ),
            }
        )
    all_rows = np.ones(len(arrays["target"]), dtype=bool)
    selected_id, selected_loss, backend, count = _best_candidate(
        admitted, arrays, all_rows, config, bank_key="interaction_bank"
    )
    selected_main, selected_main_loss, main_backend, main_count = _best_candidate(
        admitted, arrays, all_rows, config, bank_key="main_effect_bank"
    )
    evaluations += count + main_count
    backends.update((backend, main_backend))
    selected_prediction = _predict(selected_id, arrays, config, bank_key="interaction_bank")
    cpu_loss = _score(arrays, selected_prediction)["balanced_loss"]
    cpu_gpu_difference = abs(float(cpu_loss) - selected_loss)
    if cpu_gpu_difference > float(config["evaluation"]["cpu_gpu_tolerance"]):
        raise GravityItem45Error("CPU/GPU selected loss cross-check failed")
    item44_oof, fold_item44 = _item44_oof(root, arrays)
    scale_free_oof = _fixed_oof(fold_scale, scale_free_arrays, config, "scale_free_bank")
    newton = arrays["base"].copy()
    mond = arrays["base"] + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(arrays["u"]))))
    ordinary = _ordinary_crossfit(arrays, config)
    scores = {
        "universal_interaction": _score(arrays, candidate_oof),
        "item44_scale_hierarchy": _score(arrays, item44_oof),
        "matched_main_effect": _score(arrays, main_oof),
        "matched_scale_free": _score(arrays, scale_free_oof),
        "baryonic_newton": _score(arrays, newton),
        "mond_rar": _score(arrays, mond),
        "ordinary_ridge": _score(arrays, ordinary),
    }
    controls = (
        "item44_scale_hierarchy",
        "matched_main_effect",
        "matched_scale_free",
        "baryonic_newton",
        "mond_rar",
        "ordinary_ridge",
    )
    strongest = min(controls, key=lambda name: scores[name]["balanced_loss"])
    candidate_objects = scores["universal_interaction"]["object_losses"]
    control_objects = scores[strongest]["object_losses"]
    object_keys = sorted(candidate_objects)
    diff = np.asarray([control_objects[key] - candidate_objects[key] for key in object_keys])
    raw_counterexample = diff < 0.0
    stable_counterexample = raw_counterexample.copy()
    systematic_scores: dict[str, Any] = {}
    for variant_name, population, shift in config["evaluation"]["mass_scale_variants"]:
        varied = _variant_arrays(arrays, str(population), float(shift), config)
        varied_scale_free = dict(varied)
        varied_scale_free["scale_free_bank"] = np.ones((64, len(varied["target"])))
        candidate_variant = _fixed_oof(fold_candidate, varied, config, "interaction_bank")
        main_variant = _fixed_oof(fold_main, varied, config, "main_effect_bank")
        scale_variant = _fixed_oof(fold_scale, varied_scale_free, config, "scale_free_bank")
        item44_variant = np.empty(len(varied["target"]), dtype=np.float64)
        config44 = _load_item44_config(root)
        for fold, candidate_id in fold_item44.items():
            test = varied["fold"] == fold
            item44_variant[test] = _item44_predict(candidate_id, varied, config44)[test]
        variants = {
            "universal_interaction": _score(varied, candidate_variant),
            "item44_scale_hierarchy": _score(varied, item44_variant),
            "matched_main_effect": _score(varied, main_variant),
            "matched_scale_free": _score(varied, scale_variant),
            "baryonic_newton": _score(varied, varied["base"]),
            "mond_rar": _score(
                varied,
                varied["base"]
                + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(varied["u"])))),
            ),
            "ordinary_ridge": _score(varied, _ordinary_crossfit(varied, config)),
        }
        systematic_scores[str(variant_name)] = {
            "universal_interaction": variants["universal_interaction"],
            "strongest_control_name": strongest,
            "strongest_control": variants[strongest],
        }
        for i, key in enumerate(object_keys):
            stable_counterexample[i] &= (
                variants["universal_interaction"]["object_losses"][key]
                > variants[strongest]["object_losses"][key]
            )
    leave_one = [float(np.mean(np.delete(diff, i))) for i in range(len(diff))]
    trim_count = max(1, int(len(diff) * float(config["evaluation"]["robust_trim_fraction"])))
    trimmed = np.sort(diff)[trim_count:-trim_count]
    improvement = 100.0 * (
        scores[strongest]["balanced_loss"] - scores["universal_interaction"]["balanced_loss"]
    ) / scores[strongest]["balanced_loss"]
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(object_keys),
        "raw_counterexample_count": int(np.sum(raw_counterexample)),
        "quality_verified_counterexample_count": int(np.sum(raw_counterexample)),
        "uncertainty_resolved_counterexample_count": int(np.sum(stable_counterexample)),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": False,
        "strongest_baseline_failed": bool(improvement <= 0.0),
        "leave_one_changes_sign": bool(
            (min(leave_one) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "trim_changes_sign": bool(
            (float(np.mean(trimmed)) <= 0.0) != (float(np.mean(diff)) <= 0.0)
        ),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item45-joint-evaluation-1.0",
            "item": 45,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selected_candidate": decode_candidate(selected_id, config),
            "selected_full_data_balanced_training_loss": selected_loss,
            "selected_main_effect": decode_candidate(selected_main, config, main_effect=True),
            "selected_main_effect_full_data_balanced_training_loss": selected_main_loss,
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
                    "raw_counterexample": bool(raw_counterexample[i]),
                    "uncertainty_resolved_counterexample": bool(stable_counterexample[i]),
                }
                for i, key in enumerate(object_keys)
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
            "counts": {
                "s4tm_lenses": 28,
                "clash_clusters": 20,
                "clash_points": 84,
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "limitations": [
                "Both empirical datasets were exposed before Item 45, so grouped cross-validation limits row leakage but cannot create fresh confirmation.",
                "S4TM and CLASH lens quantities are model-derived summaries rather than raw image likelihoods.",
                "The density axis is a dimensionless acceleration/surface-density proxy and the environment axis is horizon occupancy, not a direct local-density measurement.",
                "The S4TM radial gradient is imposed by a projected de Vaucouleurs profile; CLASH gradients are finite differences of published model-derived profiles.",
                "Uncertainty-resolved counts cover only four global mass-scale shifts and do not justify pruning any formula family.",
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
        "beats_item44_s4tm": scores["universal_interaction"]["populations"]["S4TM"]["loss"]
        < scores["item44_scale_hierarchy"]["populations"]["S4TM"]["loss"],
        "beats_item44_clash": scores["universal_interaction"]["populations"]["CLASH"]["loss"]
        < scores["item44_scale_hierarchy"]["populations"]["CLASH"]["loss"],
        "beats_matched_main_effect_balanced": scores["universal_interaction"]["balanced_loss"]
        < scores["matched_main_effect"]["balanced_loss"],
        "beats_ordinary_ridge_balanced": scores["universal_interaction"]["balanced_loss"]
        < scores["ordinary_ridge"]["balanced_loss"],
        "paired_p_passes": float(evaluation["paired_sign_flip_p"])
        <= float(config["gates"]["paired_p_maximum"]),
        "leave_one_stable": float(
            evaluation["robustness"]["leave_one_min_mean_control_minus_candidate_loss"]
        )
        > 0.0,
        "trim_stable": float(
            evaluation["robustness"]["trimmed_mean_control_minus_candidate_loss"]
        )
        > 0.0,
        "mass_scale_audits_not_all_reverse": any(
            value["universal_interaction"]["balanced_loss"]
            < value["strongest_control"]["balanced_loss"]
            for value in systematics.values()
        ),
        "confirmation_rows_zero": int(evaluation["counts"]["sealed_confirmation_rows"]) == 0,
        "post_evaluation_candidates_zero": int(
            evaluation["counts"]["post_evaluation_candidate_cells"]
        )
        == 0,
        "fresh_confirmation_available": False,
    }
    empirical_lead = all(
        gates[key]
        for key in (
            "beats_item44_s4tm",
            "beats_item44_clash",
            "beats_matched_main_effect_balanced",
            "beats_ordinary_ridge_balanced",
            "paired_p_passes",
            "leave_one_stable",
            "trim_stable",
            "mass_scale_audits_not_all_reverse",
        )
    )
    decision = (
        "RETROSPECTIVE_ITEM45_UNIVERSAL_INTERACTION_LEAD_REQUIRES_FRESH_TEST"
        if empirical_lead
        else "NONPROMOTED_ITEM45_UNIVERSAL_INTERACTION_RESULT_RETAINED"
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item45-universal-interaction-result-1.0",
            "item": 45,
            "goal": "GRAVITY_ROADMAP_ITEM_45_UNIVERSAL_INTERACTION_VARIABLES",
            "decision": decision,
            "selected_candidate": evaluation["selected_candidate"],
            "scores": scores,
            "strongest_control": evaluation["strongest_control"],
            "aggregate_improvement_percent": evaluation["aggregate_improvement_percent"],
            "paired_sign_flip_p": evaluation["paired_sign_flip_p"],
            "gates": gates,
            "counterexample_policy_assessment": evaluation[
                "counterexample_policy_assessment"
            ],
            "counts": {
                "raw_candidates": candidate["raw_candidates"],
                "admitted_candidates": candidate["admitted_candidates"],
                "interaction_recipes": features["counts"]["interaction_recipes"],
                "unique_interaction_behaviors_on_development_data": features[
                    "dataset_behavior"
                ]["unique_interaction_coordinate_hashes"],
                "s4tm_lenses": features["counts"]["s4tm_lenses"],
                "clash_clusters": features["counts"]["clash_clusters"],
                "clash_points": features["counts"]["clash_points"],
                "candidate_point_fold_evaluations": evaluation["compute"][
                    "candidate_point_fold_evaluations"
                ],
                "sealed_confirmation_rows": 0,
                "post_evaluation_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "source_bindings": {
                "config": {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)},
                "candidate_manifest": {
                    "path": str(_source_path(root, config, "candidate_manifest").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "candidate_manifest")),
                },
                "exposure_manifest": {
                    "path": str(_source_path(root, config, "exposure_manifest").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "exposure_manifest")),
                },
                "features": {
                    "path": str(_source_path(root, config, "feature_receipt").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "feature_receipt")),
                },
                "evaluation": {
                    "path": str(_source_path(root, config, "evaluation_result").relative_to(root)),
                    "sha256": _sha256_file(_source_path(root, config, "evaluation_result")),
                },
            },
            "claims": {
                "roadmap_item_45_complete": True,
                "fresh_confirmation_completed": False,
                "universal_interaction_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "covariant_theory_established": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": "Preserve every selected interaction and mismatch; require unchanged fresh data before confirmation, and advance to Item 46 dimensional-group generation.",
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
        "candidate_manifest": _read_json(_source_path(root, config, "candidate_manifest"))
        == build_candidate_manifest(root),
        "exposure_manifest": _read_json(_source_path(root, config, "exposure_manifest"))
        == build_exposure_manifest(root),
        "feature_receipt": _read_json(_source_path(root, config, "feature_receipt"))
        == build_interaction_features(root),
        "evaluation_result": _read_json(_source_path(root, config, "evaluation_result"))
        == build_evaluation_result(root),
        "aggregate_result": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("write-freeze", "write-features", "evaluate", "aggregate", "replay"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write-freeze":
        result: Any = [str(path) for path in write_freeze_manifests(root)]
    elif args.command == "write-features":
        result = str(write_interaction_features(root))
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
