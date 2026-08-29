"""Item 46 exact Buckingham-Pi generator and retrospective gravity evaluation."""

from __future__ import annotations

import argparse
import hashlib
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
    _evaluation_arrays as _item45_evaluation_arrays,
    _item44_oof,
    _ordinary_crossfit,
    _paired_p,
    _predict as _item45_predict,
    _score,
    _variant_arrays as _item45_variant_arrays,
    load_config as _load_item45_config,
)
from sigma_theory_compiler.symmetry_dimension_derivation import (
    _expression,
    _rank_fraction,
    _sympy_rank_and_basis,
    enumerate_primitive_invariants,
)

CONFIG_PATH = Path("configs/gravity_item46_dimensionless_generator_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM44_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-44-scale-hierarchy-v1-source/joint-scale-features.json"
)
ITEM45_FEATURE_PATH = Path(
    "runs/gravity/roadmap/item-45-universal-interactions-v1-source/interaction-features.json"
)
ITEM45_EVALUATION_PATH = Path(
    "runs/gravity/roadmap/item-45-universal-interactions-v1-source/joint-evaluation-result.json"
)


class GravityItem46Error(RuntimeError):
    """Raised when an Item 46 dimensional, leakage, or evaluation gate fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def _dimension_rows(config: Mapping[str, Any]) -> list[list[int]]:
    variables = config["dimensional_declaration"]["variables"]
    axes = config["dimensional_declaration"]["dimension_axes"]
    return [
        [int(variable["dimensions"][axis]) for variable in variables]
        for axis in range(len(axes))
    ]


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item46-dimensionless-generator-config-1.0"
        or int(config.get("item", -1)) != 46
    ):
        raise GravityItem46Error("unexpected Item 46 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem46Error("stable gravity goal changed")
    if re.fullmatch(r"[0-9a-f]{40}", str(config["scientific_freeze_commit"])) is None:
        raise GravityItem46Error("Item 46 scientific freeze is not bound to a commit")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem46Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-45-universal-interactions-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem46Error("Item 45 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem46Error("Item 45 decision binding changed")
    if int(predecessor["selected_candidate"]["candidate_id"]) != int(
        required["selected_candidate_id"]
    ):
        raise GravityItem46Error("Item 45 candidate binding changed")
    discovery = config["discovery_policy"]
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem46Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem46Error("count-only rejection entered Item 46")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem46Error("finite empirical family pruning entered Item 46")
    policy = load_counterexample_policy(root / POLICY_PATH)
    if policy["empirical_evidence"]["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem46Error("executable counterexample policy changed")
    declaration = config["dimensional_declaration"]
    variables = declaration["variables"]
    if len(variables) != 9 or any(not bool(variable["positive"]) for variable in variables):
        raise GravityItem46Error("positive physical-variable declaration changed")
    rows = _dimension_rows(config)
    fraction_rank = _rank_fraction(rows)
    sympy_rank, _ = _sympy_rank_and_basis(rows)
    if fraction_rank != 3 or sympy_rank != 3 or fraction_rank != sympy_rank:
        raise GravityItem46Error("exact dimensional rank changed")
    if len(variables) - fraction_rank != int(declaration["expected_nullity"]):
        raise GravityItem46Error("dimensional nullity changed")
    generator = config["candidate_generator"]
    if (
        int(generator["pi_recipes"]) != 184
        or int(generator["cells_per_recipe"]) != 4096
        or int(generator["raw_candidate_cells"]) != 753664
        or int(generator["post_evaluation_cells"]) != 0
    ):
        raise GravityItem46Error("Pi candidate capacity changed")
    if int(declaration["response_values_allowed_in_group_generation"]) != 0:
        raise GravityItem46Error("responses entered Pi generation")
    if not bool(config["scope"]["all_empirical_responses_already_exposed"]):
        raise GravityItem46Error("retrospective disclosure changed")
    if bool(config["scope"]["fresh_confirmation_claim_allowed"]):
        raise GravityItem46Error("fresh confirmation entered retrospective Item 46")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem46Error("paid calls entered Item 46")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def pi_vectors(config: Mapping[str, Any]) -> list[tuple[int, ...]]:
    declaration = config["dimensional_declaration"]
    enumeration = declaration["enumeration"]
    vectors = enumerate_primitive_invariants(
        _dimension_rows(config),
        variable_count=len(declaration["variables"]),
        maximum_absolute_exponent=int(enumeration["maximum_absolute_exponent"]),
        maximum_l1_norm=int(enumeration["maximum_l1_norm"]),
    )
    if len(vectors) != int(enumeration["expected_primitive_pi_groups"]):
        raise GravityItem46Error("bounded Pi enumeration changed")
    if _rank_fraction(vectors) != int(enumeration["expected_spanning_rank"]):
        raise GravityItem46Error("bounded Pi groups no longer span the exact nullspace")
    if any(any(sum(row[j] * vector[j] for j in range(len(vector))) for row in _dimension_rows(config)) for vector in vectors):
        raise GravityItem46Error("dimensionful vector entered Pi catalog")
    return vectors


def primitive_basis_indices(config: Mapping[str, Any]) -> list[int]:
    selected: list[tuple[int, ...]] = []
    indices: list[int] = []
    for index, vector in enumerate(pi_vectors(config)):
        if _rank_fraction([*selected, vector]) > len(selected):
            selected.append(vector)
            indices.append(index)
        if len(selected) == int(config["dimensional_declaration"]["expected_nullity"]):
            break
    if len(indices) != 6:
        raise GravityItem46Error("minimum-complexity Pi basis changed")
    return indices


def pi_catalog(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = [variable["name"] for variable in config["dimensional_declaration"]["variables"]]
    basis = set(primitive_basis_indices(config))
    rows = _dimension_rows(config)
    labels = config["candidate_generator"]["creativity_labels"]
    catalog = []
    for recipe_id, vector in enumerate(pi_vectors(config)):
        mutation = list(vector)
        mutation[0] += 1
        residual = [sum(row[j] * mutation[j] for j in range(len(vector))) for row in rows]
        if not any(residual):
            raise GravityItem46Error("dimension-breaking mutation was admitted")
        catalog.append(
            {
                "recipe_id": recipe_id,
                "exponents": list(vector),
                "expression": _expression(names, vector),
                "l1_complexity": sum(abs(value) for value in vector),
                "primitive_basis_member": recipe_id in basis,
                "creativity_label": labels[
                    "primitive_basis" if recipe_id in basis else "bounded_combination"
                ],
                "dimension_residual": [0, 0, 0],
                "negative_control": {
                    "mutation": "increment_R_exponent_by_one",
                    "mutated_exponents": mutation,
                    "dimension_residual": residual,
                    "rejected": True,
                },
            }
        )
    return catalog


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
    u = np.logspace(
        float(gate["probe_log10_u_min"]),
        float(gate["probe_log10_u_max"]),
        int(gate["probe_points"]),
    )
    h = np.asarray(gate["probe_pi_coordinates"], dtype=np.float64)
    multiplier = 1.0 + amplitude[:, None, None] * np.power(
        u[None, None, :], -exponent[:, None, None]
    ) / (1.0 + u[None, None, :] / transition[:, None, None]) * (
        0.05 + 0.95 * h[None, :, None]
    )
    finite = np.all(np.isfinite(multiplier), axis=(1, 2))
    bounded = finite & np.all(multiplier >= float(gate["minimum_multiplier"]), axis=(1, 2))
    bounded &= np.all(multiplier <= float(gate["maximum_multiplier"]), axis=(1, 2))
    local_limit = bounded & (
        np.max(np.log10(multiplier[:, :, -1]), axis=1)
        <= float(gate["maximum_high_acceleration_log10_deviation"])
    )
    material = local_limit & (
        np.min(multiplier[:, :, 0], axis=1)
        >= float(gate["minimum_low_acceleration_multiplier"])
    )
    monotone = material & np.all(
        np.diff(multiplier, axis=2) <= float(gate["monotone_nonincreasing_tolerance"]),
        axis=(1, 2),
    )
    kept_local = np.flatnonzero(monotone)
    recipes = int(config["candidate_generator"]["pi_recipes"])
    keep = np.concatenate([kept_local + recipe * per_recipe for recipe in range(recipes)])
    raw = generate_raw_candidates(config)
    admitted = {key: value[keep] for key, value in raw.items()}
    signatures = {
        hashlib.blake2b(row.tobytes(), digest_size=16).digest()
        for row in np.round(
            np.log10(multiplier[kept_local]), int(gate["behavior_signature_decimals"])
        )
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
        "admitted_per_pi_recipe": len(kept_local),
        "generic_formula_behavior_classes": len(signatures),
        "symbolic_pi_recipes": recipes,
        "rejection_counts_nonexclusive": {
            key: value * recipes for key, value in local_counts.items()
        },
    }


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    if candidate_id < 0 or candidate_id >= total:
        raise GravityItem46Error("candidate id outside frozen grid")
    raw = generate_raw_candidates(config)
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition = _candidate_parameters(row, config)
    recipe = pi_catalog(config)[int(row["recipe"][0])]
    return {
        **recipe,
        "candidate_id": candidate_id,
        "coordinate_expression": f"H=1/[1+|log10({recipe['expression']})|]",
        "parameters": {
            "amplitude": float(amplitude[0]),
            "acceleration_exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
        },
    }


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    admitted, audit = admissible_candidates(config)
    rows = _dimension_rows(config)
    sympy_rank, sympy_basis = _sympy_rank_and_basis(rows)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item46-candidate-manifest-1.0",
            "item": 46,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "response_values_used_during_group_or_formula_generation": 0,
            "confirmation_accessed": False,
            "paid_model_calls": 0,
            **audit,
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "dimension_matrix": rows,
            "exact_algebra": {
                "fraction_rank": _rank_fraction(rows),
                "sympy_rank": sympy_rank,
                "sympy_nullspace_basis": [list(vector) for vector in sympy_basis],
                "bounded_catalog_spanning_rank": _rank_fraction(pi_vectors(config)),
                "agreement": True,
            },
            "primitive_basis_recipe_ids": primitive_basis_indices(config),
            "pi_catalog": pi_catalog(config),
            "claim_boundaries": [
                "Buckingham-Pi admissibility does not determine the free function",
                "generated monomials are algebraic combinations and carry no historical novelty claim",
                "dataset behavior can identify duplicates but cannot prove physical equivalence globally",
                "retrospective grouped validation can generate a lead but cannot confirm one",
            ],
        }
    )


def build_exposure_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item46-exposure-manifest-1.0",
            "item": 46,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "datasets": [
                {"id": "S4TM_ITEM43_EXPLORATION", "objects": 28, "response_status": "already exposed", "role": "retrospective Pi development"},
                {"id": "CLASH_ACCELERATION", "objects": 20, "points": 84, "response_status": "already exposed", "role": "retrospective Pi development"},
            ],
            "sealed_data": {"item43_s4tm_confirmation_lenses": 7, "access_authorized": False, "response_rows_read": 0},
            "rules": [
                "no result may be described as fresh confirmation",
                "derive every Pi coordinate without response values before grouped formula selection",
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


def _background(redshift: np.ndarray, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cosmology = config["fiducial_cosmology"]
    om = float(cosmology["omega_matter"])
    ol = float(cosmology["omega_lambda"])
    h0 = float(cosmology["hubble_constant_km_s_mpc"]) * 1000.0 / 3.0856775814913673e22
    z = np.asarray(redshift, dtype=np.float64)
    hubble = h0 * np.sqrt(om * np.power(1.0 + z, 3.0) + ol)
    age = 2.0 / (3.0 * h0 * math.sqrt(ol)) * np.arcsinh(
        math.sqrt(ol / om) / np.power(1.0 + z, 1.5)
    )
    return hubble, age


def _physical_log_values(arrays: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
    constants = config["si_constants"]
    radius = np.asarray(arrays["radius"], dtype=np.float64) * float(constants["kpc_to_m"])
    size = np.asarray(arrays["size"], dtype=np.float64) * float(constants["kpc_to_m"])
    a0 = float(constants["acceleration_scale_m_s2"])
    grav = float(constants["gravitational_constant_m3_kg_s2"])
    speed = float(constants["speed_of_light_m_s"])
    u = np.asarray(arrays["u"], dtype=np.float64)
    mass = u * a0 * np.square(radius) / grav
    hubble, age = _background(np.asarray(arrays["redshift"]), config)
    slope = np.asarray(arrays["mass_slope"], dtype=np.float64)
    values = np.column_stack(
        (
            radius,
            size,
            mass,
            np.full_like(radius, grav),
            np.full_like(radius, a0),
            np.full_like(radius, speed),
            hubble,
            age,
            slope,
        )
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise GravityItem46Error("Pi variables must be finite and positive")
    return np.log10(values)


def _response_blind_arrays(
    item44: Mapping[str, Any], item45: Mapping[str, Any]
) -> dict[str, Any]:
    rows44 = item44["records"]
    rows45 = item45["records"]
    if len(rows44) != len(rows45):
        raise GravityItem46Error("Item 44/45 response-blind row counts differ")
    arrays: dict[str, Any] = {
        "population": np.asarray([row["population"] for row in rows44]),
        "object": np.asarray([row["object"] for row in rows44]),
        "fold": np.asarray([int(row["fold"]) for row in rows44]),
        "radius": np.asarray([float(row["radius_kpc"]) for row in rows44]),
        "size": np.asarray([float(row["baryonic_size_kpc"]) for row in rows44]),
        "redshift": np.asarray([float(row["redshift"]) for row in rows44]),
        "u": np.asarray([float(row["u"]) for row in rows44]),
    }
    for index, (left, right) in enumerate(zip(rows44, rows45, strict=True)):
        if (
            int(right["source_row_index"]) != index
            or left["population"] != right["population"]
            or left["object"] != right["object"]
        ):
            raise GravityItem46Error("Item 44/45 response-blind row alignment changed")
    raw_gradient = np.asarray([float(row["raw_primitives"]["gradient"]) for row in rows45])
    arrays["mass_slope"] = raw_gradient + 1.0
    return arrays


def build_dimensionless_features_from_sources(
    item44: Mapping[str, Any], item45: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    arrays = _response_blind_arrays(item44, item45)
    log_values = _physical_log_values(arrays, config)
    vectors = np.asarray(pi_vectors(config), dtype=np.float64)
    log_pi = log_values @ vectors.T
    bank = 1.0 / (1.0 + np.abs(log_pi))
    if not np.all(np.isfinite(bank)) or np.any(bank <= 0.0) or np.any(bank > 1.0):
        raise GravityItem46Error("Pi coordinate construction failed")
    records = []
    for index in range(len(arrays["object"])):
        records.append(
            {
                "source_row_index": index,
                "population": str(arrays["population"][index]),
                "object": str(arrays["object"][index]),
                "fold": int(arrays["fold"][index]),
                "mass_slope": float(arrays["mass_slope"][index]),
                "log10_pi": [float(value) for value in log_pi[index]],
                "pi_coordinates": [float(value) for value in bank[index]],
            }
        )
    hashes = [
        hashlib.sha256(np.round(bank[:, index], 12).astype("<f8").tobytes()).hexdigest()
        for index in range(bank.shape[1])
    ]
    lineage = [
        {
            "population": row44["population"],
            "object": row44["object"],
            "fold": row44["fold"],
            "radius_kpc": row44["radius_kpc"],
            "baryonic_size_kpc": row44["baryonic_size_kpc"],
            "redshift": row44["redshift"],
            "u": row44["u"],
            "mass_slope": row45["raw_primitives"]["gradient"] + 1.0,
        }
        for row44, row45 in zip(item44["records"], item45["records"], strict=True)
    ]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item46-dimensionless-features-1.0",
            "item": 46,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_blind_source_lineage_sha256": _sha256_bytes(_canonical_bytes(lineage)),
            "response_fields_read_by_feature_builder": [],
            "response_values_used": 0,
            "records": records,
            "counts": {
                "s4tm_lenses": int(np.sum(arrays["population"] == "S4TM")),
                "clash_clusters": len(set(arrays["object"][arrays["population"] == "CLASH"].tolist())),
                "clash_points": int(np.sum(arrays["population"] == "CLASH")),
                "total_points": len(records),
                "pi_recipes": bank.shape[1],
                "sealed_confirmation_rows": 0,
                "paid_model_calls": 0,
            },
            "dataset_behavior": {
                "unique_pi_coordinate_hashes": len(set(hashes)),
                "duplicate_symbolic_recipes_on_development_predictors": len(hashes) - len(set(hashes)),
                "pi_coordinate_sha256": hashes,
            },
            "pi_catalog": pi_catalog(config),
            "lineage": config["data_roles"],
        }
    )


def build_dimensionless_features(root: Path) -> dict[str, Any]:
    config = load_config(root)
    return build_dimensionless_features_from_sources(
        _read_json(root / ITEM44_FEATURE_PATH),
        _read_json(root / ITEM45_FEATURE_PATH),
        config,
    )


def write_dimensionless_features(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "feature_receipt")
    _write_json(path, build_dimensionless_features(root))
    return path


def _evaluation_arrays(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    arrays = _item45_evaluation_arrays(root, _load_item45_config(root))
    feature = _read_json(_source_path(root, config, "feature_receipt"))
    rows = feature["records"]
    if len(rows) != len(arrays["target"]):
        raise GravityItem46Error("Pi feature row count changed")
    for index, row in enumerate(rows):
        if (
            int(row["source_row_index"]) != index
            or row["population"] != arrays["population"][index]
            or row["object"] != arrays["object"][index]
        ):
            raise GravityItem46Error("Pi feature/source alignment changed")
    arrays["mass_slope"] = np.asarray([float(row["mass_slope"]) for row in rows])
    arrays["pi_bank"] = np.asarray([row["pi_coordinates"] for row in rows]).T
    return arrays


def _variant_arrays(
    arrays: Mapping[str, Any], population: str, shift: float, config: Mapping[str, Any]
) -> dict[str, Any]:
    varied = _item45_variant_arrays(arrays, population, shift, _load_item45_config(Path.cwd()))
    log_values = _physical_log_values(varied, config)
    varied["pi_bank"] = 1.0 / (
        1.0 + np.abs(log_values @ np.asarray(pi_vectors(config), dtype=np.float64).T)
    ).T
    return varied


def _candidate_subset(candidates: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in candidates.items()}


def _predict(
    candidate_id: int, arrays: Mapping[str, Any], config: Mapping[str, Any], *, bank_key: str
) -> np.ndarray:
    per_recipe = int(config["candidate_generator"]["cells_per_recipe"])
    recipe = candidate_id // per_recipe
    local = candidate_id % per_recipe
    row = {
        "candidate_id": np.asarray([candidate_id]),
        "recipe": np.asarray([recipe]),
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


def _fixed_oof(
    fold_ids: Mapping[int, int], arrays: Mapping[str, Any], config: Mapping[str, Any], bank_key: str
) -> np.ndarray:
    prediction = np.empty(len(arrays["target"]), dtype=np.float64)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _predict(candidate_id, arrays, config, bank_key=bank_key)[test]
    return prediction


def _item45_oof(root: Path, arrays: Mapping[str, Any]) -> tuple[np.ndarray, dict[int, int]]:
    config45 = _load_item45_config(root)
    evaluation = _read_json(root / ITEM45_EVALUATION_PATH)
    fold_ids = {
        int(row["fold"]): int(row["selected_interaction"]["candidate_id"])
        for row in evaluation["fold_ledger"]
    }
    prediction = np.empty(len(arrays["target"]), dtype=np.float64)
    for fold, candidate_id in fold_ids.items():
        test = arrays["fold"] == fold
        prediction[test] = _item45_predict(
            candidate_id, arrays, config45, bank_key="interaction_bank"
        )[test]
    return prediction, fold_ids


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    arrays = _evaluation_arrays(root, config)
    admitted, admission = admissible_candidates(config)
    basis_ids = set(primitive_basis_indices(config))
    basis_candidates = _candidate_subset(
        admitted, np.asarray([int(value) in basis_ids for value in admitted["recipe"]])
    )
    scale_candidates = _candidate_subset(admitted, np.asarray(admitted["recipe"]) == 0)
    scale_arrays = dict(arrays)
    scale_arrays["scale_free_bank"] = np.ones(
        (int(config["candidate_generator"]["pi_recipes"]), len(arrays["target"]))
    )
    candidate_oof = np.empty(len(arrays["target"]), dtype=np.float64)
    basis_oof = np.empty(len(arrays["target"]), dtype=np.float64)
    fold_candidate: dict[int, int] = {}
    fold_basis: dict[int, int] = {}
    fold_scale: dict[int, int] = {}
    ledger = []
    backends: set[str] = set()
    evaluations = 0
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = arrays["fold"] != fold
        test = ~train
        candidate_id, train_loss, backend, count = _best_candidate(
            admitted, arrays, train, config, bank_key="pi_bank"
        )
        basis_id, basis_loss, basis_backend, basis_count = _best_candidate(
            basis_candidates, arrays, train, config, bank_key="pi_bank"
        )
        scale_id, scale_loss, scale_backend, scale_count = _best_candidate(
            scale_candidates, scale_arrays, train, config, bank_key="scale_free_bank"
        )
        candidate_oof[test] = _predict(candidate_id, arrays, config, bank_key="pi_bank")[test]
        basis_oof[test] = _predict(basis_id, arrays, config, bank_key="pi_bank")[test]
        fold_candidate[fold] = candidate_id
        fold_basis[fold] = basis_id
        fold_scale[fold] = scale_id
        evaluations += count + basis_count + scale_count
        backends.update((backend, basis_backend, scale_backend))
        ledger.append(
            {
                "fold": fold,
                "selected_pi": decode_candidate(candidate_id, config),
                "pi_training_balanced_loss": train_loss,
                "selected_primitive_basis": decode_candidate(basis_id, config),
                "primitive_basis_training_balanced_loss": basis_loss,
                "selected_scale_free_candidate_id": scale_id,
                "scale_free_training_balanced_loss": scale_loss,
                "heldout_s4tm_objects": sorted(set(arrays["object"][test & (arrays["population"] == "S4TM")].tolist())),
                "heldout_clash_objects": sorted(set(arrays["object"][test & (arrays["population"] == "CLASH")].tolist())),
            }
        )
    scale_oof = _fixed_oof(fold_scale, scale_arrays, config, "scale_free_bank")
    all_rows = np.ones(len(arrays["target"]), dtype=bool)
    selected_id, selected_loss, backend, count = _best_candidate(
        admitted, arrays, all_rows, config, bank_key="pi_bank"
    )
    selected_basis, selected_basis_loss, basis_backend, basis_count = _best_candidate(
        basis_candidates, arrays, all_rows, config, bank_key="pi_bank"
    )
    evaluations += count + basis_count
    backends.update((backend, basis_backend))
    cpu_loss = _score(arrays, _predict(selected_id, arrays, config, bank_key="pi_bank"))[
        "balanced_loss"
    ]
    cpu_gpu_difference = abs(float(cpu_loss) - selected_loss)
    if cpu_gpu_difference > float(config["evaluation"]["cpu_gpu_tolerance"]):
        raise GravityItem46Error("CPU/GPU selected loss cross-check failed")
    item45_oof, fold_item45 = _item45_oof(root, arrays)
    item44_oof, fold_item44 = _item44_oof(root, arrays)
    ordinary = _ordinary_crossfit(arrays, config)
    scores = {
        "dimensionless_generator": _score(arrays, candidate_oof),
        "item45_universal_interaction": _score(arrays, item45_oof),
        "item44_scale_hierarchy": _score(arrays, item44_oof),
        "primitive_pi_basis": _score(arrays, basis_oof),
        "matched_scale_free": _score(arrays, scale_oof),
        "baryonic_newton": _score(arrays, arrays["base"]),
        "mond_rar": _score(
            arrays,
            arrays["base"] + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(arrays["u"])))),
        ),
        "ordinary_ridge": _score(arrays, ordinary),
    }
    controls = tuple(name for name in scores if name != "dimensionless_generator")
    strongest = min(controls, key=lambda name: scores[name]["balanced_loss"])
    candidate_objects = scores["dimensionless_generator"]["object_losses"]
    control_objects = scores[strongest]["object_losses"]
    object_keys = sorted(candidate_objects)
    diff = np.asarray([control_objects[key] - candidate_objects[key] for key in object_keys])
    raw_counterexample = diff < 0.0
    stable_counterexample = raw_counterexample.copy()
    systematic_scores: dict[str, Any] = {}
    config44 = _read_json(root / "configs/gravity_item44_scale_hierarchy_v1.json")
    config45 = _load_item45_config(root)
    for variant_name, population, shift in config["evaluation"]["mass_scale_variants"]:
        varied = _item45_variant_arrays(arrays, str(population), float(shift), config45)
        log_values = _physical_log_values(varied, config)
        varied["pi_bank"] = (
            1.0 / (1.0 + np.abs(log_values @ np.asarray(pi_vectors(config), dtype=float).T))
        ).T
        varied_scale = dict(varied)
        varied_scale["scale_free_bank"] = np.ones(
            (int(config["candidate_generator"]["pi_recipes"]), len(varied["target"]))
        )
        candidate_variant = _fixed_oof(fold_candidate, varied, config, "pi_bank")
        basis_variant = _fixed_oof(fold_basis, varied, config, "pi_bank")
        scale_variant = _fixed_oof(fold_scale, varied_scale, config, "scale_free_bank")
        item45_variant = np.empty(len(varied["target"]), dtype=float)
        item44_variant = np.empty(len(varied["target"]), dtype=float)
        from sigma_theory_compiler.gravity_item44_scale_hierarchy import _predict as item44_predict

        for fold in range(int(config["evaluation"]["outer_folds"])):
            test = varied["fold"] == fold
            item45_variant[test] = _item45_predict(
                fold_item45[fold], varied, config45, bank_key="interaction_bank"
            )[test]
            item44_variant[test] = item44_predict(fold_item44[fold], varied, config44)[test]
        variants = {
            "dimensionless_generator": _score(varied, candidate_variant),
            "item45_universal_interaction": _score(varied, item45_variant),
            "item44_scale_hierarchy": _score(varied, item44_variant),
            "primitive_pi_basis": _score(varied, basis_variant),
            "matched_scale_free": _score(varied, scale_variant),
            "baryonic_newton": _score(varied, varied["base"]),
            "mond_rar": _score(varied, varied["base"] + np.log10(1.0 / (1.0 - np.exp(-np.sqrt(varied["u"]))))),
            "ordinary_ridge": _score(varied, _ordinary_crossfit(varied, config)),
        }
        systematic_scores[str(variant_name)] = {
            "dimensionless_generator": variants["dimensionless_generator"],
            "strongest_control_name": strongest,
            "strongest_control": variants[strongest],
        }
        for index, key in enumerate(object_keys):
            stable_counterexample[index] &= (
                variants["dimensionless_generator"]["object_losses"][key]
                > variants[strongest]["object_losses"][key]
            )
    leave_one = [float(np.mean(np.delete(diff, index))) for index in range(len(diff))]
    trim_count = max(1, int(len(diff) * float(config["evaluation"]["robust_trim_fraction"])))
    trimmed = np.sort(diff)[trim_count:-trim_count]
    improvement = 100.0 * (
        scores[strongest]["balanced_loss"] - scores["dimensionless_generator"]["balanced_loss"]
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
        "leave_one_changes_sign": bool((min(leave_one) <= 0.0) != (float(np.mean(diff)) <= 0.0)),
        "trim_changes_sign": bool((float(np.mean(trimmed)) <= 0.0) != (float(np.mean(diff)) <= 0.0)),
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = assess_counterexample_evidence(policy_report, load_counterexample_policy(root / POLICY_PATH))
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item46-joint-evaluation-1.0",
            "item": 46,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selected_candidate": decode_candidate(selected_id, config),
            "selected_full_data_balanced_training_loss": selected_loss,
            "selected_primitive_basis": decode_candidate(selected_basis, config),
            "selected_primitive_basis_full_data_balanced_training_loss": selected_basis_loss,
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
                {"object": key, "raw_counterexample": bool(raw_counterexample[index]), "uncertainty_resolved_counterexample": bool(stable_counterexample[index])}
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
            "counts": {"s4tm_lenses": 28, "clash_clusters": 20, "clash_points": 84, "sealed_confirmation_rows": 0, "post_evaluation_candidate_cells": 0, "paid_model_calls": 0},
            "limitations": [
                "All empirical responses were exposed before Item 46; grouped cross-validation cannot create fresh confirmation.",
                "Buckingham-Pi analysis generates admissible coordinates but leaves their free functional relationship undetermined.",
                "The enclosed mass is reconstructed from the same response-blind g_bar and radius predictors, creating exact behavioral dependencies that are reported rather than hidden.",
                "S4TM and CLASH lens summaries are model-derived, and the radial mass slope inherits the Item 45 profile assumptions.",
                "Four global mass-scale shifts do not exhaust baryonic, geometric, selection, or lens-model uncertainties and cannot prune a family.",
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
        "beats_item45_s4tm": scores["dimensionless_generator"]["populations"]["S4TM"]["loss"] < scores["item45_universal_interaction"]["populations"]["S4TM"]["loss"],
        "beats_item45_clash": scores["dimensionless_generator"]["populations"]["CLASH"]["loss"] < scores["item45_universal_interaction"]["populations"]["CLASH"]["loss"],
        "beats_primitive_basis_balanced": scores["dimensionless_generator"]["balanced_loss"] < scores["primitive_pi_basis"]["balanced_loss"],
        "beats_ordinary_ridge_balanced": scores["dimensionless_generator"]["balanced_loss"] < scores["ordinary_ridge"]["balanced_loss"],
        "paired_p_passes": float(evaluation["paired_sign_flip_p"]) <= float(config["gates"]["paired_p_maximum"]),
        "leave_one_stable": float(evaluation["robustness"]["leave_one_min_mean_control_minus_candidate_loss"]) > 0.0,
        "trim_stable": float(evaluation["robustness"]["trimmed_mean_control_minus_candidate_loss"]) > 0.0,
        "mass_scale_audits_not_all_reverse": any(value["dimensionless_generator"]["balanced_loss"] < value["strongest_control"]["balanced_loss"] for value in systematics.values()),
        "confirmation_rows_zero": int(evaluation["counts"]["sealed_confirmation_rows"]) == 0,
        "post_evaluation_candidates_zero": int(evaluation["counts"]["post_evaluation_candidate_cells"]) == 0,
        "fresh_confirmation_available": False,
    }
    empirical_lead = all(
        gates[key]
        for key in (
            "beats_item45_s4tm", "beats_item45_clash", "beats_primitive_basis_balanced",
            "beats_ordinary_ridge_balanced", "paired_p_passes", "leave_one_stable",
            "trim_stable", "mass_scale_audits_not_all_reverse",
        )
    )
    decision = (
        "RETROSPECTIVE_ITEM46_DIMENSIONLESS_GENERATOR_LEAD_REQUIRES_FRESH_TEST"
        if empirical_lead
        else "NONPROMOTED_ITEM46_DIMENSIONLESS_GENERATOR_RESULT_RETAINED"
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item46-dimensionless-generator-result-1.0",
            "item": 46,
            "goal": "GRAVITY_ROADMAP_ITEM_46_DIMENSIONLESS_GENERATOR",
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
                "symbolic_pi_recipes": candidate["symbolic_pi_recipes"],
                "unique_pi_behaviors_on_development_data": features["dataset_behavior"]["unique_pi_coordinate_hashes"],
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
                "roadmap_item_46_complete": True,
                "fresh_confirmation_completed": False,
                "dimensionless_law_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "covariant_theory_established": False,
                "formula_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": "Preserve every Pi coordinate, selected lead, and mismatch; require fresh unchanged replication for confirmation, then advance to Item 47 operator generation.",
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
        "feature_receipt": _read_json(_source_path(root, config, "feature_receipt")) == build_dimensionless_features(root),
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
        result = str(write_dimensionless_features(root))
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
