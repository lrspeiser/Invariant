"""Frozen Item 36 screened extra-dimensional radial search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _git,
    _improvement,
    _mse,
    _read_json,
    _read_tsv,
    _require_ancestor,
    _sha256_bytes,
    _sha256_file,
    _to_numpy,
    _verify_content_hash,
    _write_json,
    _write_tsv,
)
from sigma_theory_compiler.gravity_item29_nonlinear_self_interaction import _backend
from sigma_theory_compiler.gravity_item30_screening_mechanisms import (
    _candidate_digest,
    _minimum_separations_arcsec,
    _ridge_oof,
)
from sigma_theory_compiler.gravity_item32_boundary_focusing import GravityItem32Error
from sigma_theory_compiler.gravity_item35_modified_inertia import (
    GravityItem35Error,
    _hmac_rank,
    _maps_payload,
    _response_record,
    _robust_comparison,
    _source_feature_rows,
)
from sigma_theory_compiler.gravity_item35_modified_inertia import (
    _fresh_pool as _item35_fresh_pool,
)
from sigma_theory_compiler.gravity_item35_modified_inertia import (
    load_config as load_item35_config,
)

CONFIG_PATH = Path("configs/gravity_item36_extra_dimensions_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item36_extra_dimensions.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
TEST_PATH = Path("tests/test_gravity_item36_extra_dimensions.py")
_ADMISSIBLE_CACHE: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


class GravityItem36Error(RuntimeError):
    """Raised when an Item 36 freeze, dimensional, leakage, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if (
        config.get("schema_version") != "invariant-gravity-item36-extra-dimensions-config-1.0"
        or int(config.get("item", -1)) != 36
    ):
        raise GravityItem36Error("unexpected Item 36 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem36Error("stable gravity goal changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem36Error("raw candidate boundary changed")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem36Error("post-response candidates entered Item 36")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem36Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem36Error("paid calls are outside Item 36")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem36Error("equal-viability policy changed")
    if not bool(
        config["discovery_policy"]["single_empirical_counterexample_is_not_a_formula_family_veto"]
    ):
        raise GravityItem36Error("counterexample policy changed")
    if bool(config["gates"]["single_empirical_counterexample_is_veto"]):
        raise GravityItem36Error("one-object empirical veto entered Item 36")
    if bool(config["gates"]["single_object_sensitive_formula_may_promote"]):
        raise GravityItem36Error("single-object-sensitive promotion entered Item 36")
    if sum(bool(row["action_track_eligible"]) for row in generator["niches"]) != 4:
        raise GravityItem36Error("action eligibility allocation changed")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem36Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    value["source_feature_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "source_features",
        "source_feature_manifest",
        "exploration_responses",
        "response_source_manifest",
        "compute_manifest",
    )
    return {key: base / str(config["paths"][key]) for key in keys}


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen) != _contract_digest(config):
        raise GravityItem36Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes) or _sha256_bytes(frozen_module) != _sha256_file(
        root / MODULE_PATH
    ):
        raise GravityItem36Error("Item 36 module differs from scientific freeze")


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in ("predictors", "predictor_source_manifest", "sample_manifest", "candidate_manifest"):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem36Error(f"{key} differs from sample freeze")


def verify_source_feature_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["source_feature_freeze_commit"])
    _require_ancestor(root, commit, "source feature freeze")
    paths = _source_paths(root, config)
    for key in ("source_features", "source_feature_manifest"):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem36Error(f"{key} differs from source feature freeze")


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    radices = {
        "polarity": len(generator["polarities"]),
        "amplitude": len(generator["amplitudes"]),
        "scale": len(generator["crossover_scales_kpc"]),
        "width": len(generator["transition_widths"]),
        "power": len(generator["powers"]),
        "coupling": len(generator["baryon_couplings"]),
        "dimension": len(generator["dimension_shifts"]),
        "side": len(generator["phase_sides"]),
    }
    per_niche = int(generator["raw_candidate_cells"]) // 4
    if int(np.prod(list(radices.values()))) != per_niche:
        raise GravityItem36Error("mixed-radix grammar does not fill each niche exactly")
    pieces: dict[str, list[np.ndarray]] = {"niche": []} | {key: [] for key in radices}
    for niche in range(4):
        working = np.arange(per_niche, dtype=np.int64)
        decoded: dict[str, np.ndarray] = {}
        for key, radix in reversed(list(radices.items())):
            decoded[key] = (working % radix).astype(np.int16)
            working //= radix
        if np.any(working != 0):
            raise GravityItem36Error("candidate decoder overflow")
        pieces["niche"].append(np.full(per_niche, niche, dtype=np.int16))
        for key in radices:
            pieces[key].append(decoded[key])
    arrays = {key: np.concatenate(values) for key, values in pieces.items()}
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    order = random.permutation(len(arrays["niche"]))
    return {key: values[order] for key, values in arrays.items()}


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    index = {key: arrays[key][begin:end] for key in arrays}
    return {
        "niche": xp.asarray(index["niche"]),
        "polarity": xp.asarray(np.asarray(generator["polarities"])[index["polarity"]]),
        "amplitude": xp.asarray(np.asarray(generator["amplitudes"])[index["amplitude"]]),
        "scale": xp.asarray(np.asarray(generator["crossover_scales_kpc"])[index["scale"]]),
        "width": xp.asarray(np.asarray(generator["transition_widths"])[index["width"]]),
        "power": xp.asarray(np.asarray(generator["powers"])[index["power"]]),
        "coupling": xp.asarray(np.asarray(generator["baryon_couplings"])[index["coupling"]]),
        "dimension": xp.asarray(np.asarray(generator["dimension_shifts"])[index["dimension"]]),
        "side": xp.asarray(np.asarray(generator["phase_sides"])[index["side"]]),
    }


def _gravity_multiplier(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    predictors: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> tuple[Any, Any]:
    values = _candidate_values(config, arrays, begin, end, xp)
    shape = (-1, 1)
    niche = values["niche"].reshape(shape)
    polarity = values["polarity"].reshape(shape)
    amplitude = values["amplitude"].reshape(shape)
    scale = values["scale"].reshape(shape)
    width = values["width"].reshape(shape)
    power = values["power"].reshape(shape)
    coupling = values["coupling"].reshape(shape)
    dimension = values["dimension"].reshape(shape)
    side = values["side"].reshape(shape)
    radius = xp.asarray(predictors["radius_kpc"])[None, :]
    log_acceleration = xp.asarray(predictors["log_acceleration"])[None, :]
    source_eta = xp.asarray(predictors["source_nonaxisymmetry"])[None, :]
    radial_slope = xp.asarray(predictors["radial_source_slope"])[None, :]
    log_surface_density = xp.asarray(predictors["log_surface_density"])[None, :]
    log_a0 = math.log10(float(config["physics"]["constants"]["a0_m_s2"]))
    acceleration_coordinate = (log_acceleration - log_a0) / width
    low_acceleration = 1.0 / (1.0 + xp.power(10.0, power * acceleration_coordinate))
    branch = xp.clip(0.5 * (1.0 + side * (2.0 * low_acceleration - 1.0)), 0.0, 1.0)
    local_center = float(config["physics"]["universal_local_shield_log10_acceleration_m_s2"])
    local_power = float(config["physics"]["universal_local_shield_power"])
    local_shield = 1.0 / (1.0 + xp.power(10.0, local_power * (log_acceleration - local_center)))
    radial_ratio = xp.maximum(radius / scale, 1e-12)
    compact_q = xp.exp(-radial_ratio)
    compact = (compact_q + dimension * compact_q**2) / (1.0 + dimension)
    warped = 1.0 / (1.0 + xp.power(radial_ratio, power))
    cascading = xp.power(radial_ratio, power) / (1.0 + xp.power(radial_ratio, power))
    morphology = xp.clip(
        (1.0 + coupling * source_eta)
        / (1.0 + coupling)
        * xp.exp(-xp.abs(radial_slope - dimension) / xp.maximum(width, 1e-8)),
        0.0,
        1.0,
    )
    density_relocalization = 1.0 / (
        1.0 + xp.power(10.0, (log_surface_density - 8.5) / xp.maximum(width, 1e-8))
    )
    spectral = xp.clip(
        cascading
        * (dimension / (1.0 + dimension))
        * (morphology + coupling * density_relocalization)
        / (1.0 + coupling),
        0.0,
        1.0,
    )
    activation = (
        branch
        * local_shield
        * xp.where(
            niche == 0,
            compact,
            xp.where(
                niche == 1,
                warped * (1.0 + coupling * source_eta) / (1.0 + coupling),
                xp.where(niche == 2, cascading, spectral),
            ),
        )
    )
    activation = xp.clip(activation, 0.0, 1.0)
    multiplier = 1.0 + polarity * amplitude * activation
    return multiplier, activation


def _candidate_delta_log10_speed(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    predictors: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    multiplier, _ = _gravity_multiplier(config, arrays, predictors, begin, end, xp)
    return 0.5 * xp.log10(multiplier)


def _adversarial_predictors() -> dict[str, np.ndarray]:
    index = np.arange(80)
    return {
        "radius_kpc": np.geomspace(0.05, 50.0, 80),
        "log_acceleration": np.linspace(-13.5, -8.2, 80),
        "source_nonaxisymmetry": 0.01 + 0.8 * ((index * 17) % 80) / 79.0,
        "radial_source_slope": 0.1 + 2.9 * ((index * 29) % 80) / 79.0,
        "log_surface_density": 6.0 + 5.0 * ((index * 37) % 80) / 79.0,
    }


def _local_predictors(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "radius_kpc": np.asarray([float(config["physics"]["constants"]["one_au_kpc"])]),
        "log_acceleration": np.asarray(
            [math.log10(float(config["physics"]["constants"]["one_au_acceleration_m_s2"]))]
        ),
        "source_nonaxisymmetry": np.asarray([0.0]),
        "radial_source_slope": np.asarray([1.0]),
        "log_surface_density": np.asarray([12.0]),
    }


def _admissible_candidates(
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_key = _sha256_bytes(
        _canonical_bytes(
            {
                "candidate_generator": config["candidate_generator"],
                "admissibility": config["admissibility"],
                "physics": config["physics"],
            }
        )
    )
    if cache_key in _ADMISSIBLE_CACHE:
        return _ADMISSIBLE_CACHE[cache_key]
    raw = generate_raw_candidates(config)
    domain = _adversarial_predictors()
    local = _local_predictors(config)
    count = len(raw["niche"])
    keep = np.zeros(count, dtype=bool)
    minimum = np.full(count, np.nan)
    maximum = np.full(count, np.nan)
    material = np.full(count, np.nan)
    local_response = np.full(count, np.nan)
    batch = int(config["evaluation"]["candidate_batch_size"])
    gates = config["admissibility"]
    for begin in range(0, count, batch):
        end = min(begin + batch, count)
        multiplier, _ = _gravity_multiplier(config, raw, domain, begin, end, np)
        local_multiplier, _ = _gravity_multiplier(config, raw, local, begin, end, np)
        delta = 0.5 * np.log10(multiplier)
        minimum[begin:end] = np.min(multiplier, axis=1)
        maximum[begin:end] = np.max(multiplier, axis=1)
        material[begin:end] = np.max(np.abs(delta), axis=1)
        local_response[begin:end] = np.abs(local_multiplier[:, 0] - 1.0)
        keep[begin:end] = (
            np.all(np.isfinite(delta), axis=1)
            & (minimum[begin:end] >= float(gates["minimum_gravity_multiplier"]))
            & (maximum[begin:end] <= float(gates["maximum_gravity_multiplier"]))
            & (material[begin:end] >= float(gates["minimum_material_delta_log10_speed"]))
            & (
                local_response[begin:end]
                <= float(gates["maximum_local_fractional_gravity_response"])
            )
            & (material[begin:end] <= float(gates["maximum_absolute_delta_log10_speed"]))
        )
    arrays = {key: values[keep] for key, values in raw.items()}
    signatures = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        signatures.append(
            np.round(
                _candidate_delta_log10_speed(config, arrays, domain, begin, end, np),
                int(gates["behavioral_equivalence_precision_decimal_places"]),
            )
        )
    behavior = np.concatenate(signatures) if signatures else np.empty((0, 80))
    classes = len(np.unique(behavior, axis=0))
    raw_counts = Counter(int(value) for value in raw["niche"])
    admitted_counts = Counter(int(value) for value in arrays["niche"])
    audit = {
        "raw_candidates": count,
        "raw_per_niche": {str(key): raw_counts[key] for key in range(4)},
        "admissible_candidates": len(arrays["niche"]),
        "admissible_per_niche": {str(key): admitted_counts[key] for key in range(4)},
        "raw_candidate_digest": _candidate_digest(raw),
        "admissible_candidate_digest": _candidate_digest(arrays),
        "exact_parameter_signatures": len(
            np.unique(np.column_stack([arrays[key] for key in sorted(arrays)]), axis=0)
        ),
        "behavioral_equivalence_classes_adversarial": classes,
        "behavioral_duplicate_cells_adversarial": len(arrays["niche"]) - classes,
        "minimum_admitted_gravity_multiplier": float(np.min(minimum[keep])),
        "maximum_admitted_gravity_multiplier": float(np.max(maximum[keep])),
        "minimum_admitted_material_delta_log10_speed": float(np.min(material[keep])),
        "maximum_admitted_local_fractional_gravity_response": float(np.max(local_response[keep])),
        "maximum_admitted_absolute_delta_log10_speed": float(np.max(material[keep])),
    }
    generator = config["candidate_generator"]
    for expected_key, observed_key in (
        ("expected_raw_candidate_digest", "raw_candidate_digest"),
        ("expected_admissible_candidate_digest", "admissible_candidate_digest"),
        ("expected_admissible_candidates", "admissible_candidates"),
        (
            "expected_behavioral_equivalence_classes_adversarial",
            "behavioral_equivalence_classes_adversarial",
        ),
    ):
        expected = generator.get(expected_key)
        if expected not in (None, "TO_BE_MEASURED", -1) and audit[observed_key] != expected:
            raise GravityItem36Error(f"candidate invariant changed: {expected_key}")
    expected_niches = generator.get("expected_admissible_per_niche")
    if (
        expected_niches
        and all(int(value) >= 0 for value in expected_niches.values())
        and audit["admissible_per_niche"] != expected_niches
    ):
        raise GravityItem36Error("admissible niche counts changed")
    _ADMISSIBLE_CACHE[cache_key] = arrays, audit
    return arrays, audit


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item36-candidate-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "niches": config["candidate_generator"]["niches"],
            "action_proxy": config["physics"]["action_proxy"],
            "variation_proxy": config["physics"]["variation_proxy"],
            "four_dimensional_limit": config["physics"]["four_dimensional_limit"],
            "historical_novelty_claimed": False,
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "post_response_cells": 0,
            "audit": audit,
        }
    )


def _fresh_pool(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item35_config = load_item35_config(root)
    inherited, inherited_audit = _item35_fresh_pool(root, item35_config)
    independence = config["independence"]
    if len(inherited) != int(independence["expected_item35_fresh_disk_pool"]):
        raise GravityItem36Error("Item 35 fresh disk pool changed")
    item35_sample = _read_json(root / str(config["sources"]["item35_sample_manifest"]))
    _verify_content_hash(item35_sample, "Item 35 sample manifest")
    roles = item35_sample["objects"]
    if len(roles) != int(independence["expected_item35_roles"]):
        raise GravityItem36Error("Item 35 role count changed")
    role_ids = {str(row["plateifu"]) for row in roles}
    coordinates = np.asarray([[float(row["ra"]), float(row["dec"])] for row in roles])
    post_identity = [row for row in inherited if str(row["plateifu"]) not in role_ids]
    if len(post_identity) != int(independence["expected_post_identity_pool"]):
        raise GravityItem36Error("Item 36 identity exclusion count changed")
    separations = _minimum_separations_arcsec(post_identity, coordinates)
    veto = float(independence["coordinate_veto_arcsec"])
    excluded_coordinates = int(np.count_nonzero(separations <= veto))
    if excluded_coordinates != int(independence["expected_additional_coordinate_exclusions"]):
        raise GravityItem36Error("Item 36 coordinate exclusion count changed")
    fresh = []
    for source, separation in zip(post_identity, separations, strict=True):
        if separation <= veto:
            continue
        row = dict(source)
        row["minimum_item35_role_separation_arcsec"] = float(separation)
        fresh.append(row)
    fresh.sort(key=lambda row: str(row["plateifu"]))
    if len(fresh) != int(independence["expected_fresh_pool"]):
        raise GravityItem36Error("Item 36 fresh pool changed")
    return fresh, {
        "item35_fresh_disk_pool": len(inherited),
        "item35_predecessor_audit": inherited_audit,
        "item35_roles": len(roles),
        "post_identity_pool": len(post_identity),
        "additional_coordinate_exclusions": excluded_coordinates,
        "fresh_pool": len(fresh),
    }


def _sample_manifest(
    config: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    sample = config["sample"]
    mass_median = float(np.median([float(row["log_stellar_mass"]) for row in pool]))
    size_median = float(np.median([float(row["log_half_light_radius"]) for row in pool]))
    cells: dict[str, list[dict[str, Any]]] = {
        f"m{mass}-r{size}": [] for mass in range(2) for size in range(2)
    }
    for source in pool:
        row = dict(source)
        mass_bin = int(float(row["log_stellar_mass"]) >= mass_median)
        size_bin = int(float(row["log_half_light_radius"]) >= size_median)
        cell = f"m{mass_bin}-r{size_bin}"
        row.update({"mass_bin": mass_bin, "size_bin": size_bin, "sample_cell": cell})
        cells[cell].append(row)
    capacities = {key: len(values) for key, values in cells.items()}
    if capacities != {key: int(value) for key, value in sample["expected_cell_capacities"].items()}:
        raise GravityItem36Error("Item 36 response-blind cell capacities changed")
    objects = []
    cell_counts = {}
    selected_count = int(sample["selected_per_mass_size_cell"])
    confirmation_count = int(sample["confirmation_per_cell"])
    for cell, values in sorted(cells.items()):
        selected = sorted(
            values,
            key=lambda row: _hmac_rank(str(sample["role_key"]), f"select|{row['plateifu']}"),
        )[:selected_count]
        confirmation_ids = {
            str(row["plateifu"])
            for row in sorted(
                selected,
                key=lambda row: _hmac_rank(
                    str(sample["role_key"]), f"confirmation|{row['plateifu']}"
                ),
            )[:confirmation_count]
        }
        exploration = sorted(
            (row for row in selected if str(row["plateifu"]) not in confirmation_ids),
            key=lambda row: _hmac_rank(str(sample["fold_key"]), str(row["plateifu"])),
        )
        folds = {
            str(row["plateifu"]): index % int(sample["outer_folds"])
            for index, row in enumerate(exploration)
        }
        for row in selected:
            identity = str(row["plateifu"])
            is_confirmation = identity in confirmation_ids
            row.update(
                {
                    "role": "reserved_confirmation" if is_confirmation else "exploration",
                    "outer_fold": None if is_confirmation else folds[identity],
                    "source_map_read": False,
                    "velocity_response_read": False,
                }
            )
            objects.append(row)
        cell_counts[cell] = {
            "eligible": len(values),
            "selected": len(selected),
            "exploration": len(exploration),
            "reserved_confirmation": len(confirmation_ids),
        }
    objects.sort(key=lambda row: str(row["plateifu"]))
    roles = Counter(str(row["role"]) for row in objects)
    fold_counts = Counter(int(row["outer_fold"]) for row in objects if row["role"] == "exploration")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item36-sample-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "mass_median_log10_msun": f"{mass_median:.12e}",
            "size_median_log10_kpc": f"{size_median:.12e}",
            "objects": objects,
            "selected_cell_counts": cell_counts,
            "fold_counts_exploration": {
                str(key): fold_counts[key] for key in range(int(sample["outer_folds"]))
            },
            "counts": {
                "fresh_pool": len(pool),
                "selected": len(objects),
                "exploration": roles["exploration"],
                "reserved_confirmation": roles["reserved_confirmation"],
                "source_map_rows_read": 0,
                "velocity_response_rows_read": 0,
            },
            "claims": {
                "target_blind": True,
                "confirmation_values_read": 0,
                "object_identity_used_as_numeric_feature": False,
            },
        }
    )


def prepare_predictors(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    pool, predecessor_audit = _fresh_pool(root, config)
    sample_manifest = _sample_manifest(config, pool)
    sample = config["sample"]
    expected_counts = {
        "fresh_pool": int(config["independence"]["expected_fresh_pool"]),
        "selected": int(sample["expected_selected"]),
        "exploration": int(sample["expected_exploration"]),
        "reserved_confirmation": int(sample["expected_confirmation"]),
        "source_map_rows_read": 0,
        "velocity_response_rows_read": 0,
    }
    if sample_manifest["counts"] != expected_counts:
        raise GravityItem36Error("frozen Item 36 sample counts changed")
    columns = [
        *list(pool[0]),
        "mass_bin",
        "size_bin",
        "sample_cell",
        "role",
        "outer_fold",
        "source_map_read",
        "velocity_response_read",
    ]
    _write_tsv(paths["predictors"], sample_manifest["objects"], columns)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item36-predictor-source-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "item35_sample": {
                "path": config["sources"]["item35_sample_manifest"],
                "sha256": _sha256_file(root / str(config["sources"]["item35_sample_manifest"])),
            },
            "predecessor_audit": predecessor_audit,
            "counts": {
                "fresh_pool": len(pool),
                "selected": len(sample_manifest["objects"]),
                "response_columns_read": 0,
                "source_map_rows_read": 0,
            },
            "claims": {
                "predecessor_exclusion_before_roles": True,
                "sample_target_blind": True,
                "confirmation_values_read": 0,
                "post_response_formula_cells": 0,
            },
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], _candidate_manifest(config))
    return paths


def _add_radial_source_predictors(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["annulus_index"]))
    if len(ordered) != 4 or [int(row["annulus_index"]) for row in ordered] != [0, 1, 2, 3]:
        raise GravityItem36Error("source map does not contain four frozen annuli")
    log_radius = np.log(np.asarray([float(row["radius_kpc"]) for row in ordered], dtype=np.float64))
    log_mass = np.log(
        np.asarray(
            [float(row["enclosed_stellar_mass_msun"]) for row in ordered],
            dtype=np.float64,
        )
    )
    slope = np.gradient(log_mass, log_radius)
    output = []
    for row, value in zip(ordered, slope, strict=True):
        updated = dict(row)
        updated["radial_source_slope"] = float(value)
        updated["localization_coordinate"] = float(
            np.clip(
                0.4 * float(row["source_nonaxisymmetry"])
                + 0.3 * abs(float(value) - 1.0) / 2.0
                + 0.3 / (1.0 + 10.0 ** (float(row["log_surface_density"]) - 8.5)),
                0.0,
                1.0,
            )
        )
        output.append(updated)
    return output


def _source_control_spec(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    labels = (
        "weighted_radius_re",
        "radius_kpc",
        "log_acceleration",
        "radial_source_slope",
        "source_nonaxisymmetry",
        "localization_coordinate",
    )
    probabilities = [float(value) for value in config["evaluation"]["radial_control_quantiles"]]
    output = {}
    for label in labels:
        values = np.asarray([float(row[label]) for row in rows], dtype=np.float64)
        output[label] = {
            "center": float(np.median(values)),
            "scale": max(float(np.std(values)), 1e-8),
            "knots": [float(value) for value in np.quantile(values, probabilities)],
        }
    return output


def acquire_source_maps(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample_manifest = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample_manifest, "Item 36 sample manifest")
    exploration = sorted(
        (row for row in sample_manifest["objects"] if row["role"] == "exploration"),
        key=lambda row: str(row["plateifu"]),
    )
    confirmations = {
        str(row["plateifu"])
        for row in sample_manifest["objects"]
        if row["role"] == "reserved_confirmation"
    }
    source_rows: list[dict[str, Any]] = []
    files = []
    failures = []
    for sample_row in exploration:
        identity = str(sample_row["plateifu"])
        if identity in confirmations:
            raise GravityItem36Error("confirmation identity entered source-map access")
        try:
            payload, filename, url, headers = _maps_payload(root, config, identity)
            file_record = {
                "plateifu": identity,
                "file_name": filename,
                "url": url,
                "file_bytes": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "cache": headers.get("cache", "miss"),
            }
            rows = _add_radial_source_predictors(_source_feature_rows(payload, sample_row, config))
        except (GravityItem32Error, GravityItem35Error, GravityItem36Error) as exc:
            failures.append({"plateifu": identity, "reason": str(exc)})
            continue
        source_rows.extend(rows)
        files.append({**file_record, "source_extension_checksums_verified": True})
    if not source_rows:
        raise GravityItem36Error("no Item 36 source-only MAPS features were extracted")
    touched = {str(row["plateifu"]) for row in [*files, *failures]}
    if confirmations & touched:
        raise GravityItem36Error("confirmation MAPS entered Item 36 source phase")
    source_rows.sort(key=lambda row: (str(row["plateifu"]), int(row["annulus_index"])))
    new_columns = [
        "annulus",
        "annulus_index",
        "source_pixels",
        "weighted_radius_re",
        "flux_m1",
        "flux_m2",
        "flux_m3",
        "source_nonaxisymmetry",
        "flux_asymmetry",
        "centroid_offset_re",
        "radial_concentration",
        "radius_kpc",
        "enclosed_stellar_mass_msun",
        "enclosed_stellar_mass_fraction",
        "log_baryonic_speed_km_s",
        "log_acceleration",
        "omega_Gyr_inverse",
        "log_omega_Gyr_inverse",
        "mode_frequency_ratio",
        "age_gyr_proxy",
        "vertical_to_orbital_frequency",
        "radial_source_slope",
        "localization_coordinate",
    ]
    _write_tsv(paths["source_features"], source_rows, [*list(exploration[0]), *new_columns])
    complete_galaxies = len({str(row["plateifu"]) for row in source_rows})
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item36-source-feature-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "data_model": config["sources"]["maps"]["data_model"],
            "files": files,
            "failures": failures,
            "source_features_sha256": _sha256_file(paths["source_features"]),
            "ordinary_radial_control_spec": _source_control_spec(source_rows, config),
            "derived_predictor_contract": config["map_source"]["derived_predictors"],
            "counts": {
                "exploration_maps_attempted": len(exploration),
                "exploration_maps_parsed": complete_galaxies,
                "source_feature_rows": len(source_rows),
                "source_map_failures": len(failures),
                "source_extensions_read_per_parsed_map": len(
                    config["sources"]["maps"]["source_extensions"]
                ),
                "response_extensions_read": 0,
                "velocity_arrays_read": 0,
                "confirmation_maps_downloaded": 0,
                "post_response_formula_cells": 0,
            },
            "claims": {
                "source_only": True,
                "velocity_response_blind": True,
                "radial_thresholds_frozen_before_response": True,
                "confirmation_opened": False,
            },
        }
    )
    _write_json(paths["source_feature_manifest"], manifest)
    return paths


def acquire_responses(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    verify_source_feature_freeze(root, config)
    paths = _source_paths(root, config)
    source_manifest = _read_json(paths["source_feature_manifest"])
    _verify_content_hash(source_manifest, "Item 36 source feature manifest")
    if int(source_manifest["counts"]["velocity_arrays_read"]) != 0:
        raise GravityItem36Error("velocity response entered source-feature freeze")
    sample_manifest = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample_manifest, "Item 36 sample manifest")
    sample_by_id = {
        str(row["plateifu"]): row
        for row in sample_manifest["objects"]
        if row["role"] == "exploration"
    }
    confirmations = {
        str(row["plateifu"])
        for row in sample_manifest["objects"]
        if row["role"] == "reserved_confirmation"
    }
    cache = root / str(config["sources"]["maps"]["raw_cache"])
    records = []
    failures = []
    files = []
    for file_record in source_manifest["files"]:
        identity = str(file_record["plateifu"])
        if identity in confirmations or identity not in sample_by_id:
            raise GravityItem36Error("nonexploration identity entered response access")
        path = cache / str(file_record["file_name"])
        if not path.is_file():
            raise GravityItem36Error(f"source-frozen MAPS payload missing: {identity}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != str(file_record["file_sha256"]):
            raise GravityItem36Error(f"source-frozen MAPS checksum changed: {identity}")
        try:
            record = _response_record(payload, sample_by_id[identity], config)
        except (GravityItem32Error, GravityItem35Error, GravityItem36Error) as exc:
            failures.append({"plateifu": identity, "reason": str(exc)})
            continue
        records.append(record)
        files.append(
            {
                "plateifu": identity,
                "file_name": file_record["file_name"],
                "file_bytes": len(payload),
                "file_sha256": digest,
                "response_extension_checksums_verified": True,
            }
        )
    records.sort(key=lambda row: str(row["plateifu"]))
    responses = _content_hashed(
        {
            "schema_version": "invariant-gravity-item36-exploration-responses-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "source_feature_freeze_commit": config["source_feature_freeze_commit"],
            "records": records,
            "failures": failures,
            "counts": {
                "source_complete": len(records),
                "Halpha_complete": sum(bool(row["Halpha_quality_pass"]) for row in records),
                "stellar_complete": sum(bool(row["stellar_quality_pass"]) for row in records),
                "both_tracer_complete": sum(
                    bool(row["Halpha_quality_pass"]) and bool(row["stellar_quality_pass"])
                    for row in records
                ),
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
            },
            "claims": {
                "confirmation_opened": False,
                "candidate_generation_preceded_response": True,
                "source_features_frozen_before_response": True,
            },
        }
    )
    response_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item36-response-source-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "source_feature_freeze_commit": config["source_feature_freeze_commit"],
            "source_feature_manifest_sha256": _sha256_file(paths["source_feature_manifest"]),
            "source_features_sha256": _sha256_file(paths["source_features"]),
            "files": files,
            "failures": failures,
            "counts": {
                "cached_exploration_payloads_read": len(files),
                "new_downloads": 0,
                "confirmation_maps_downloaded": 0,
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
        }
    )
    _write_json(paths["exploration_responses"], responses)
    _write_json(paths["response_source_manifest"], response_manifest)
    return paths


def _load_experiment_data(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    for key in (
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "source_feature_manifest",
        "exploration_responses",
        "response_source_manifest",
    ):
        value = _read_json(paths[key])
        _verify_content_hash(value, f"Item 36 {key}")
    source_manifest = _read_json(paths["source_feature_manifest"])
    responses = _read_json(paths["exploration_responses"])
    if int(source_manifest["counts"]["velocity_arrays_read"]) != 0:
        raise GravityItem36Error("response leaked into source feature manifest")
    if int(responses["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem36Error("confirmation response entered Item 36")
    features = _read_tsv(paths["source_features"])
    feature_by_key = {(str(row["plateifu"]), int(row["annulus_index"])): row for row in features}
    primary = []
    transfer = []
    for response in responses["records"]:
        identity = str(response["plateifu"])
        if bool(response["Halpha_quality_pass"]):
            for annulus_index, annulus in enumerate(response["Halpha_annuli"]):
                key = (identity, annulus_index)
                if key not in feature_by_key:
                    raise GravityItem36Error("Halpha response lacks frozen source feature")
                primary.append(
                    {
                        **feature_by_key[key],
                        "target_Halpha_circular": float(annulus["log10_circular_speed_km_s"]),
                    }
                )
        if bool(response["Halpha_quality_pass"]) and bool(response["stellar_quality_pass"]):
            for annulus_index, annulus in enumerate(response["stellar_annuli"]):
                key = (identity, annulus_index)
                if key not in feature_by_key:
                    raise GravityItem36Error("stellar response lacks frozen source feature")
                transfer.append(
                    {
                        **feature_by_key[key],
                        "target_stellar_circular": float(annulus["log10_circular_speed_km_s"]),
                    }
                )
    primary.sort(key=lambda row: (str(row["plateifu"]), int(row["annulus_index"])))
    transfer.sort(key=lambda row: (str(row["plateifu"]), int(row["annulus_index"])))
    return primary, transfer, responses


def _candidate_predictors(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    labels = (
        "radius_kpc",
        "log_acceleration",
        "source_nonaxisymmetry",
        "radial_source_slope",
        "log_surface_density",
    )
    return {
        label: np.asarray([float(row[label]) for row in rows], dtype=np.float64) for label in labels
    }


def _build_candidate_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    predictors = _candidate_predictors(rows)
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(_candidate_delta_log10_speed(config, arrays, predictors, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _screen_candidates(
    delta: Any,
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    outer_folds = int(config["sample"]["outer_folds"])
    if {int(value) for value in folds} != set(range(outer_folds)):
        raise GravityItem36Error("response-complete folds are incomplete")
    target_xp = xp.asarray(target)
    base_xp = xp.asarray(base)
    prediction = np.empty_like(target)
    indices = []
    losses = []
    offsets = []
    raw_offsets = []
    bounds = config["physics"]["shared_mass_proxy_scale_bounds"]
    lower = 0.5 * math.log10(float(bounds[0]))
    upper = 0.5 * math.log10(float(bounds[1]))
    for fold in range(outer_folds):
        train_np = np.where(folds != fold)[0]
        held_np = np.where(folds == fold)[0]
        train = xp.asarray(train_np)
        residual = target_xp[train] - base_xp[train]
        values = delta[:, train]
        raw = xp.mean(residual[None, :] - values, axis=1)
        fitted = xp.clip(raw, lower, upper)
        loss = xp.mean(xp.square(residual[None, :] - values - fitted[:, None]), axis=1)
        index = int(_to_numpy(xp.argmin(loss), xp))
        indices.append(index)
        losses.append(float(_to_numpy(loss[index], xp)))
        offset = float(_to_numpy(fitted[index], xp))
        offsets.append(offset)
        raw_offsets.append(float(_to_numpy(raw[index], xp)))
        prediction[held_np] = (
            base[held_np] + _to_numpy(delta[index, xp.asarray(held_np)], xp) + offset
        )
    return {
        "prediction": prediction,
        "selected_indices": indices,
        "selected_train_losses": losses,
        "offsets": offsets,
        "raw_offsets": raw_offsets,
    }


def _select_full_candidate(
    delta: Any,
    target: np.ndarray,
    base: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> tuple[int, float, float]:
    residual = xp.asarray(target - base)
    bounds = config["physics"]["shared_mass_proxy_scale_bounds"]
    lower = 0.5 * math.log10(float(bounds[0]))
    upper = 0.5 * math.log10(float(bounds[1]))
    raw = xp.mean(residual[None, :] - delta, axis=1)
    fitted = xp.clip(raw, lower, upper)
    loss = xp.mean(xp.square(residual[None, :] - delta - fitted[:, None]), axis=1)
    index = int(_to_numpy(xp.argmin(loss), xp))
    return index, float(_to_numpy(fitted[index], xp)), float(_to_numpy(loss[index], xp))


def _normalized_columns(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> np.ndarray:
    columns = []
    for label in labels:
        values = np.asarray([float(row[label]) for row in rows], dtype=np.float64)
        columns.append((values - np.median(values)) / max(float(np.std(values)), 1e-8))
    return np.column_stack(columns)


def _baseline_designs(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    fixed = []
    for label, normalization in config["evaluation"]["fixed_feature_normalization"].items():
        center, scale = (float(value) for value in normalization)
        fixed.append(
            (np.asarray([float(row[label]) for row in rows], dtype=np.float64) - center) / scale
        )
    fixed_matrix = np.column_stack(fixed)
    annulus_index = np.asarray([int(row["annulus_index"]) for row in rows])
    annulus_one_hot = np.column_stack(
        [(annulus_index == index).astype(float) for index in range(1, 4)]
    )
    structural = np.column_stack([fixed_matrix[:, :8], annulus_one_hot])
    morphology = _normalized_columns(
        rows,
        (
            "weighted_radius_re",
            "flux_m1",
            "flux_m2",
            "flux_m3",
            "source_nonaxisymmetry",
            "radial_source_slope",
            "localization_coordinate",
        ),
    )
    flexible_base = np.column_stack([fixed_matrix, morphology, annulus_one_hot])
    interactions = np.column_stack(
        [
            flexible_base[:, 0] * flexible_base[:, 1],
            flexible_base[:, 0] * flexible_base[:, 3],
            flexible_base[:, 3] * flexible_base[:, 4],
            flexible_base[:, -5] * flexible_base[:, -4],
            flexible_base[:, 8] * flexible_base[:, -6],
        ]
    )
    flexible = np.column_stack([flexible_base, flexible_base**2, interactions])
    spec = source_manifest["ordinary_radial_control_spec"]
    smooth = []
    hinges = []
    for label in spec:
        values = np.asarray([float(row[label]) for row in rows], dtype=np.float64)
        center = float(spec[label]["center"])
        scale = float(spec[label]["scale"])
        smooth.append((values - center) / scale)
        for knot in spec[label]["knots"]:
            hinges.append(np.maximum(values - float(knot), 0.0) / scale)
            hinges.append(np.maximum(float(knot) - values, 0.0) / scale)
    ordinary = np.column_stack([flexible, *smooth, *hinges])
    return {"structural": structural, "flexible": flexible, "ordinary_radial": ordinary}


def _offset_oof(
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    prediction = np.empty_like(target)
    bounds = config["physics"]["shared_mass_proxy_scale_bounds"]
    lower = 0.5 * math.log10(float(bounds[0]))
    upper = 0.5 * math.log10(float(bounds[1]))
    for fold in range(int(config["sample"]["outer_folds"])):
        train = folds != fold
        held = folds == fold
        offset = float(np.clip(np.mean(target[train] - base[train]), lower, upper))
        prediction[held] = base[held] + offset
    return prediction


def _baseline_predictions(
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    designs = _baseline_designs(rows, config, source_manifest)
    output = {"baryonic_source": _offset_oof(target, base, folds, config)}
    alpha = {
        "structural": float(config["evaluation"]["ridge_alpha_structural"]),
        "flexible": float(config["evaluation"]["ridge_alpha_flexible"]),
        "ordinary_radial": float(config["evaluation"]["ridge_alpha_radial"]),
    }
    for label, design in designs.items():
        output[label] = base + _ridge_oof(
            target - base,
            folds,
            design,
            alpha[label],
            int(config["sample"]["outer_folds"]),
        )
    return output


def _candidate_record(
    index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(arrays["niche"][index])
    definition = config["candidate_generator"]["niches"][niche]
    return {
        "admissible_index": index,
        "niche_index": niche,
        "niche": definition["id"],
        "creativity_label": definition["creativity_label"],
        "action_track_eligible": bool(definition["action_track_eligible"]),
        "polarity": float(values["polarity"][0]),
        "amplitude": float(values["amplitude"][0]),
        "crossover_scale_kpc": float(values["scale"][0]),
        "transition_width": float(values["width"][0]),
        "power": float(values["power"][0]),
        "baryon_coupling": float(values["coupling"][0]),
        "dimension_shift": float(values["dimension"][0]),
        "phase_side": float(values["side"][0]),
        "equivalence_boundary": definition["equivalence"],
    }


def _transfer_selected_candidates(
    delta: Any,
    primary_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    primary_index = {
        (str(row["plateifu"]), int(row["annulus_index"])): index
        for index, row in enumerate(primary_rows)
    }
    indices = np.asarray(
        [primary_index[(str(row["plateifu"]), int(row["annulus_index"]))] for row in transfer_rows],
        dtype=int,
    )
    target = np.asarray(
        [float(row["target_stellar_circular"]) for row in transfer_rows], dtype=np.float64
    )
    base = np.asarray(
        [float(row["log_baryonic_speed_km_s"]) for row in transfer_rows], dtype=np.float64
    )
    folds = np.asarray([int(row["outer_fold"]) for row in transfer_rows], dtype=int)
    prediction = np.empty_like(target)
    offsets = []
    bounds = config["physics"]["shared_mass_proxy_scale_bounds"]
    lower = 0.5 * math.log10(float(bounds[0]))
    upper = 0.5 * math.log10(float(bounds[1]))
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        if not len(train) or not len(held):
            raise GravityItem36Error("stellar transfer folds are incomplete")
        correction = _to_numpy(delta[int(selected_indices[fold]), xp.asarray(indices)], xp)
        offset = float(
            np.clip(np.mean(target[train] - base[train] - correction[train]), lower, upper)
        )
        offsets.append(offset)
        prediction[held] = base[held] + correction[held] + offset
    return {
        "target": target,
        "base": base,
        "folds": folds,
        "prediction": prediction,
        "offsets": offsets,
        "selected_indices_from_Halpha": [int(value) for value in selected_indices],
        "formula_reselection_on_stellar": False,
    }


def _permuted_target(
    target: np.ndarray,
    reference: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    random: np.random.Generator,
) -> np.ndarray:
    residual = target - reference
    shuffled = np.empty_like(residual)
    groups = np.asarray([f"{row['sample_cell']}|{row['annulus']}" for row in rows], dtype=object)
    for group in sorted(set(groups.tolist())):
        indices = np.where(groups == group)[0]
        shuffled[indices] = residual[random.permutation(indices)]
    return reference + shuffled


def _robust_by_galaxy(
    target: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    identities = sorted({str(row["plateifu"]) for row in rows})
    candidate_error = []
    reference_error = []
    for identity in identities:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if str(row["plateifu"]) == identity]
        )
        candidate_error.append(float(np.mean(np.square(target[indices] - candidate[indices]))))
        reference_error.append(float(np.mean(np.square(target[indices] - reference[indices]))))
    candidate_values = np.asarray(candidate_error)
    reference_values = np.asarray(reference_error)
    comparative = candidate_values - reference_values
    full = _improvement(float(np.mean(reference_values)), float(np.mean(candidate_values)))
    order = np.argsort(np.abs(comparative))[::-1]
    leave = np.ones(len(identities), dtype=bool)
    leave[int(order[0])] = False
    leave_improvement = _improvement(
        float(np.mean(reference_values[leave])), float(np.mean(candidate_values[leave]))
    )
    trim_count = max(
        1,
        math.floor(
            float(config["evaluation"]["robust_comparative_trim_fraction"]) * len(identities)
        ),
    )
    trimmed = np.ones(len(identities), dtype=bool)
    trimmed[order[:trim_count]] = False
    trim_improvement = _improvement(
        float(np.mean(reference_values[trimmed])), float(np.mean(candidate_values[trimmed]))
    )
    sensitive = bool(
        (full >= 0.0) != (leave_improvement >= 0.0) or (full >= 0.0) != (trim_improvement >= 0.0)
    )
    return {
        "galaxies": len(identities),
        "counterexample_galaxies": int(np.count_nonzero(candidate_values > reference_values)),
        "counterexample_fraction": float(np.mean(candidate_values > reference_values)),
        "full_improvement": full,
        "single_most_influential_identity": identities[int(order[0])],
        "leave_one_most_influential_improvement": leave_improvement,
        "leave_one_changes_improvement_sign": bool((full >= 0.0) != (leave_improvement >= 0.0)),
        "trim_fraction": float(config["evaluation"]["robust_comparative_trim_fraction"]),
        "trimmed_galaxies": trim_count,
        "trimmed_improvement": trim_improvement,
        "trim_changes_improvement_sign": bool((full >= 0.0) != (trim_improvement >= 0.0)),
        "single_object_sensitive": sensitive,
        "single_counterexample_is_veto": False,
    }


def _slice_results(
    target: np.ndarray,
    candidate: np.ndarray,
    references: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    definitions = {
        "sample_cell": [str(row["sample_cell"]) for row in rows],
        "annulus": [str(row["annulus"]) for row in rows],
        "mass_bin": [str(row["mass_bin"]) for row in rows],
        "size_bin": [str(row["size_bin"]) for row in rows],
    }
    output = []
    for dimension, labels in definitions.items():
        values = np.asarray(labels, dtype=object)
        for label in sorted(set(labels)):
            selected = values == label
            candidate_mse = _mse(target[selected], candidate[selected])
            output.append(
                {
                    "dimension": dimension,
                    "value": label,
                    "observations": int(np.count_nonzero(selected)),
                    "galaxies": len(
                        {
                            str(row["plateifu"])
                            for row, keep in zip(rows, selected, strict=True)
                            if keep
                        }
                    ),
                    "candidate_mse": candidate_mse,
                    "improvement_vs_baryonic_source": _improvement(
                        _mse(target[selected], references["baryonic_source"][selected]),
                        candidate_mse,
                    ),
                    "improvement_vs_ordinary_radial": _improvement(
                        _mse(target[selected], references["ordinary_radial"][selected]),
                        candidate_mse,
                    ),
                }
            )
    return output


def _synthetic_controls(
    delta: Any,
    base: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    xp: Any,
) -> dict[str, Any]:
    injections = []
    for niche in range(4):
        indices = np.where(arrays["niche"] == niche)[0]
        values = delta[xp.asarray(indices)]
        variance = xp.var(values, axis=1)
        injection_index = int(indices[int(_to_numpy(xp.argmax(variance), xp))])
        target = base + _to_numpy(delta[injection_index], xp)
        selected = _screen_candidates(delta, target, base, folds, config, xp)
        selected_niches = [int(arrays["niche"][index]) for index in selected["selected_indices"]]
        injections.append(
            {
                "injection_index": injection_index,
                "injection_niche": niche,
                "selected_niches": selected_niches,
                "exact_niche_recovered_all_folds": all(value == niche for value in selected_niches),
                "candidate_mse": _mse(target, selected["prediction"]),
                "transfer_reselected_formula": False,
            }
        )
    newtonian = _screen_candidates(delta, base.copy(), base, folds, config, xp)
    baseline = _offset_oof(base.copy(), base, folds, config)
    candidate_mse = _mse(base, newtonian["prediction"])
    baseline_mse = _mse(base, baseline)
    return {
        "injections": injections,
        "all_injected_niches_recovered": all(
            row["exact_niche_recovered_all_folds"] for row in injections
        ),
        "all_injected_niches_transfer_unchanged": True,
        "Newtonian_candidate_mse": candidate_mse,
        "Newtonian_baseline_mse": baseline_mse,
        "Newtonian_control_candidate_improves": candidate_mse < baseline_mse - 1e-16,
    }


def _evaluate(
    root: Path,
    config: Mapping[str, Any],
    primary_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(primary_rows) < 4 * int(config["sample"]["outer_folds"]):
        raise GravityItem36Error("too few Halpha response-complete observations")
    arrays, candidate_audit = _admissible_candidates(config)
    target = np.asarray(
        [float(row["target_Halpha_circular"]) for row in primary_rows], dtype=np.float64
    )
    base = np.asarray(
        [float(row["log_baryonic_speed_km_s"]) for row in primary_rows], dtype=np.float64
    )
    folds = np.asarray([int(row["outer_fold"]) for row in primary_rows], dtype=int)
    paths = _source_paths(root, config)
    source_manifest = _read_json(paths["source_feature_manifest"])
    _verify_content_hash(source_manifest, "Item 36 source feature manifest")
    xp, backend, device = _backend()
    xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    delta = _build_candidate_matrix(config, arrays, primary_rows, xp)
    xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    crosscheck = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["niche"]))
    cpu = _candidate_delta_log10_speed(
        config, arrays, _candidate_predictors(primary_rows), 0, crosscheck, np
    )
    gpu = _to_numpy(delta[:crosscheck], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu - gpu)))
    observed = _screen_candidates(delta, target, base, folds, config, xp)
    full_index, full_offset, full_loss = _select_full_candidate(delta, target, base, config, xp)
    baselines = _baseline_predictions(target, base, folds, primary_rows, config, source_manifest)
    candidate_mse = _mse(target, observed["prediction"])
    baseline_mse = {key: _mse(target, value) for key, value in baselines.items()}
    improvements = {key: _improvement(value, candidate_mse) for key, value in baseline_mse.items()}
    selected_records = [
        _candidate_record(int(index), config, arrays) for index in observed["selected_indices"]
    ]
    niche_counts = Counter(int(row["niche_index"]) for row in selected_records)
    modal_niche, modal_count = niche_counts.most_common(1)[0]
    transfer = _transfer_selected_candidates(
        delta, primary_rows, transfer_rows, observed["selected_indices"], config, xp
    )
    transfer_baselines = _baseline_predictions(
        transfer["target"],
        transfer["base"],
        transfer["folds"],
        transfer_rows,
        config,
        source_manifest,
    )
    transfer_candidate_mse = _mse(transfer["target"], transfer["prediction"])
    transfer_baseline_mse = {
        key: _mse(transfer["target"], value) for key, value in transfer_baselines.items()
    }
    transfer_improvements = {
        key: _improvement(value, transfer_candidate_mse)
        for key, value in transfer_baseline_mse.items()
    }
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    observed_statistic = improvements["ordinary_radial"]
    null_improvements = []
    for _ in range(int(config["evaluation"]["permutation_trials"])):
        null_target = _permuted_target(target, baselines["ordinary_radial"], primary_rows, random)
        null_selected = _screen_candidates(delta, null_target, base, folds, config, xp)
        null_baselines = _baseline_predictions(
            null_target, base, folds, primary_rows, config, source_manifest
        )
        null_improvements.append(
            _improvement(
                _mse(null_target, null_baselines["ordinary_radial"]),
                _mse(null_target, null_selected["prediction"]),
            )
        )
    p_value = (1.0 + sum(value >= observed_statistic for value in null_improvements)) / (
        len(null_improvements) + 1.0
    )
    robust_observations = _robust_comparison(
        target, observed["prediction"], baselines["ordinary_radial"], primary_rows, config
    )
    robust_galaxies = _robust_by_galaxy(
        target,
        observed["prediction"],
        baselines["ordinary_radial"],
        primary_rows,
        config,
    )
    robust_transfer = _robust_by_galaxy(
        transfer["target"],
        transfer["prediction"],
        transfer_baselines["ordinary_radial"],
        transfer_rows,
        config,
    )
    slices = _slice_results(target, observed["prediction"], baselines, primary_rows)
    synthetic = _synthetic_controls(delta, base, folds, config, arrays, xp)
    Halpha_galaxies = len({str(row["plateifu"]) for row in primary_rows})
    stellar_galaxies = len({str(row["plateifu"]) for row in transfer_rows})
    quality_limited = [dict(row) for row in responses["failures"]]
    for row in responses["records"]:
        reasons = []
        if not bool(row["Halpha_quality_pass"]):
            reasons.append({"channel": "Halpha", "reason": row["Halpha_quality_reason"]})
        if not bool(row["stellar_quality_pass"]):
            reasons.append({"channel": "stellar", "reason": row["stellar_quality_reason"]})
        if reasons:
            quality_limited.append({"plateifu": row["plateifu"], "channel_reasons": reasons})
    quality = {
        "Halpha_complete_galaxies": Halpha_galaxies,
        "Halpha_quality_retention_fraction": Halpha_galaxies
        / float(config["sample"]["expected_exploration"]),
        "stellar_transfer_complete_galaxies": stellar_galaxies,
        "Halpha_minimum_pass": Halpha_galaxies
        >= int(config["sample"]["minimum_complete_Halpha_galaxies"]),
        "Halpha_fraction_pass": Halpha_galaxies / float(config["sample"]["expected_exploration"])
        >= float(config["sample"]["minimum_Halpha_quality_retention_fraction"]),
        "stellar_minimum_pass": stellar_galaxies
        >= int(config["sample"]["minimum_complete_stellar_transfer_galaxies"]),
        "missing_or_quality_limited_measurements": quality_limited,
    }
    quality["all_pass"] = bool(
        quality["Halpha_minimum_pass"]
        and quality["Halpha_fraction_pass"]
        and quality["stellar_minimum_pass"]
    )
    gates = config["gates"]
    broad_baryonic = [
        float(row["improvement_vs_baryonic_source"])
        for row in slices
        if row["dimension"] in ("sample_cell", "annulus")
    ]
    sensitive = bool(
        robust_galaxies["single_object_sensitive"] or robust_transfer["single_object_sensitive"]
    )
    universal_checks = {
        "quality": quality["all_pass"],
        "improvement_vs_baryonic_source": improvements["baryonic_source"]
        >= float(gates["minimum_improvement_vs_baryonic_source"]),
        "improvement_vs_structural": improvements["structural"]
        >= float(gates["minimum_improvement_vs_structural"]),
        "improvement_vs_flexible": improvements["flexible"]
        >= float(gates["minimum_improvement_vs_flexible"]),
        "improvement_vs_ordinary_radial": improvements["ordinary_radial"]
        >= float(gates["minimum_improvement_vs_ordinary_radial"]),
        "all_broad_slices_vs_baryonic": min(broad_baryonic)
        >= float(gates["minimum_each_broad_slice_improvement_vs_baryonic_source"]),
        "stellar_transfer_vs_ordinary_radial": transfer_improvements["ordinary_radial"]
        >= float(gates["minimum_stellar_transfer_improvement_vs_ordinary_radial"]),
        "selection_aware_p": p_value <= float(gates["maximum_selection_aware_permutation_p"]),
        "same_niche_folds": modal_count >= int(gates["minimum_same_niche_folds"]),
        "selected_action_eligible": all(
            bool(row["action_track_eligible"]) for row in selected_records
        ),
        "synthetic_injections": bool(synthetic["all_injected_niches_recovered"]),
        "Newtonian_control": not bool(synthetic["Newtonian_control_candidate_improves"]),
        "cpu_gpu": cpu_gpu_max <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        "not_single_object_sensitive": not sensitive,
    }
    phenomenon_checks = {
        "quality": quality["all_pass"],
        "improvement_vs_ordinary_radial": improvements["ordinary_radial"]
        >= float(gates["phenomenon_minimum_improvement_vs_ordinary_radial"]),
        "stellar_transfer": transfer_improvements["ordinary_radial"]
        >= float(gates["phenomenon_minimum_stellar_transfer_improvement"]),
        "selection_aware_p": p_value <= float(gates["phenomenon_maximum_selection_aware_p"]),
        "synthetic_injections": bool(synthetic["all_injected_niches_recovered"]),
        "cpu_gpu": cpu_gpu_max <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        "not_single_object_sensitive": not sensitive,
    }
    partial_slices = [
        row
        for row in slices
        if float(row["improvement_vs_ordinary_radial"])
        >= float(gates["partial_minimum_slice_improvement_vs_ordinary_radial"])
    ]
    universal_pass = all(bool(value) for value in universal_checks.values())
    phenomenon_pass = all(bool(value) for value in phenomenon_checks.values())
    if not quality["all_pass"]:
        decision = "INCONCLUSIVE_ITEM36_RESPONSE_QUALITY"
    elif universal_pass:
        decision = "ADVANCE_ITEM36_EXTRA_DIMENSIONAL_CANDIDATE"
    elif phenomenon_pass:
        decision = "RETAIN_ITEM36_RADIAL_PHENOMENON_SIGNAL"
    elif partial_slices:
        decision = "RETAIN_ITEM36_PARTIAL_SLICE_SIGNAL"
    else:
        decision = "NO_ITEM36_EXTRA_DIMENSIONAL_LEAD"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item36-extra-dimensions-result-1.0",
            "item": 36,
            "title": config["title"],
            "decision": decision,
            "claim_ceiling": config["scope"]["claim_ceiling"],
            "sample": {
                "Halpha_galaxies": Halpha_galaxies,
                "Halpha_observations": len(primary_rows),
                "stellar_transfer_galaxies": stellar_galaxies,
                "stellar_transfer_observations": len(transfer_rows),
                "confirmation_values_read": 0,
            },
            "quality": quality,
            "candidate_space": candidate_audit,
            "selected_candidates_by_fold": selected_records,
            "selected_niche_counts": {
                str(key): value for key, value in sorted(niche_counts.items())
            },
            "modal_selected_niche": int(modal_niche),
            "full_exploration_selected_candidate": _candidate_record(full_index, config, arrays),
            "full_exploration_offset": full_offset,
            "full_exploration_training_loss": full_loss,
            "primary_Halpha": {
                "candidate_mse": candidate_mse,
                "baseline_mse": baseline_mse,
                "improvements": improvements,
                "selection_aware_permutation_p": p_value,
                "null_improvements": null_improvements,
                "robust_observations": robust_observations,
                "robust_galaxies": robust_galaxies,
                "single_object_sensitive": bool(robust_galaxies["single_object_sensitive"]),
            },
            "stellar_unchanged_transfer": {
                "candidate_mse": transfer_candidate_mse,
                "baseline_mse": transfer_baseline_mse,
                "improvements": transfer_improvements,
                "selected_indices_from_Halpha": transfer["selected_indices_from_Halpha"],
                "formula_reselection_on_stellar": False,
                "robust_galaxies": robust_transfer,
                "single_object_sensitive": bool(robust_transfer["single_object_sensitive"]),
            },
            "broad_slices": slices,
            "partial_positive_slices": partial_slices,
            "synthetic_controls": synthetic,
            "action_and_ontology": {
                "selected_action_track_eligible": all(
                    bool(row["action_track_eligible"]) for row in selected_records
                ),
                "positive_Green_spectrum_all_admitted": (
                    candidate_audit["minimum_admitted_gravity_multiplier"]
                    >= float(config["admissibility"]["minimum_gravity_multiplier"])
                ),
                "maximum_local_fractional_gravity_response": candidate_audit[
                    "maximum_admitted_local_fractional_gravity_response"
                ],
                "screening_is_added_not_standard_unscreened_ADD_RS_DGP": True,
                "complete_5D_solution_proved": False,
                "relativistic_completion_proved": False,
                "lensing_law_proved": False,
                "historical_novelty_claimed": False,
            },
            "universal_track_checks": universal_checks,
            "universal_track_pass": universal_pass,
            "phenomenon_track_checks": phenomenon_checks,
            "phenomenon_track_pass": phenomenon_pass,
            "paper_claim_authorized": False,
            "counterexample_policy": {
                "single_empirical_counterexample_is_veto": False,
                "single_object_sensitive_formula_may_promote": False,
                "hard_theoretical_violation_can_be_single_case_veto": True,
                "formula_family_rejected_from_one_empirical_object": False,
            },
            "compute": {
                "backend": backend,
                "device": device,
                "candidate_matrix_seconds": matrix_seconds,
                "cpu_gpu_max_absolute_difference": cpu_gpu_max,
                "cpu_gpu_tolerance": config["evaluation"]["cpu_gpu_tolerance"],
                "raw_candidate_cells": int(config["candidate_generator"]["raw_candidate_cells"]),
                "admissible_candidate_cells": len(arrays["niche"]),
                "permutation_trials": int(config["evaluation"]["permutation_trials"]),
                "approximate_candidate_observation_evaluations": int(
                    len(arrays["niche"])
                    * len(primary_rows)
                    * (1 + int(config["evaluation"]["permutation_trials"]))
                ),
                "confirmation_calls": 0,
                "paid_model_calls": 0,
                "estimated_api_spend_usd": 0.0,
            },
            "next_step": (
                "Keep confirmations sealed; any paper claim requires unchanged fresh replication. "
                "Advance the ordered roadmap to Item 37 after recording this exploration."
            ),
        }
    )
    compute_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item36-compute-manifest-1.0",
            "result_content_sha256": result["content_sha256"],
            "compute": result["compute"],
            "confirmation_values_read": 0,
            "post_response_candidate_cells": 0,
            "paid_model_calls": 0,
        }
    )
    return result, compute_manifest


def run_experiment(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    verify_source_feature_freeze(root, config)
    primary, transfer, responses = _load_experiment_data(root, config)
    result, compute_manifest = _evaluate(root, config, primary, transfer, responses)
    paths = _source_paths(root, config)
    result_path = root / str(config["paths"]["result"])
    _write_json(paths["compute_manifest"], compute_manifest)
    _write_json(result_path, result)
    return result_path


def validate_checked(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    verify_source_feature_freeze(root, config)
    paths = _source_paths(root, config)
    for key in (
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "source_feature_manifest",
        "exploration_responses",
        "response_source_manifest",
        "compute_manifest",
    ):
        manifest = _read_json(paths[key])
        _verify_content_hash(manifest, f"Item 36 {key}")
    result = _read_json(root / str(config["paths"]["result"]))
    _verify_content_hash(result, "Item 36 result")
    if result.get("schema_version") != "invariant-gravity-item36-extra-dimensions-result-1.0":
        raise GravityItem36Error("unexpected Item 36 result schema")
    if int(result.get("item", -1)) != 36:
        raise GravityItem36Error("unexpected Item 36 result item")
    allowed = {
        "INCONCLUSIVE_ITEM36_RESPONSE_QUALITY",
        "ADVANCE_ITEM36_EXTRA_DIMENSIONAL_CANDIDATE",
        "RETAIN_ITEM36_RADIAL_PHENOMENON_SIGNAL",
        "RETAIN_ITEM36_PARTIAL_SLICE_SIGNAL",
        "NO_ITEM36_EXTRA_DIMENSIONAL_LEAD",
    }
    if result.get("decision") not in allowed:
        raise GravityItem36Error("unexpected Item 36 decision")
    if int(result["sample"]["confirmation_values_read"]) != 0:
        raise GravityItem36Error("confirmation value entered Item 36 result")
    policy = result["counterexample_policy"]
    if bool(policy["single_empirical_counterexample_is_veto"]):
        raise GravityItem36Error("one-object empirical veto entered Item 36 result")
    if bool(policy["single_object_sensitive_formula_may_promote"]):
        raise GravityItem36Error("single-object-sensitive promotion entered Item 36 result")
    if bool(policy["formula_family_rejected_from_one_empirical_object"]):
        raise GravityItem36Error("formula family rejected from one empirical object")
    if (
        bool(result["primary_Halpha"]["single_object_sensitive"])
        or bool(result["stellar_unchanged_transfer"]["single_object_sensitive"])
    ) and (bool(result["universal_track_pass"]) or bool(result["phenomenon_track_pass"])):
        raise GravityItem36Error("single-object-sensitive formula was promoted")
    if bool(result["paper_claim_authorized"]):
        raise GravityItem36Error("unreplicated paper claim entered Item 36")
    if int(result["compute"]["paid_model_calls"]) != 0:
        raise GravityItem36Error("paid model call entered Item 36")
    if int(result["compute"]["confirmation_calls"]) != 0:
        raise GravityItem36Error("confirmation computation entered Item 36")
    if float(result["compute"]["cpu_gpu_max_absolute_difference"]) > float(
        result["compute"]["cpu_gpu_tolerance"]
    ):
        raise GravityItem36Error("Item 36 CPU/GPU crosscheck failed")
    sample = _read_json(paths["sample_manifest"])
    if sample["counts"] != {
        "fresh_pool": int(config["independence"]["expected_fresh_pool"]),
        "selected": int(config["sample"]["expected_selected"]),
        "exploration": int(config["sample"]["expected_exploration"]),
        "reserved_confirmation": int(config["sample"]["expected_confirmation"]),
        "source_map_rows_read": 0,
        "velocity_response_rows_read": 0,
    }:
        raise GravityItem36Error("Item 36 sample counts changed")
    source = _read_json(paths["source_feature_manifest"])
    if int(source["counts"]["response_extensions_read"]) != 0:
        raise GravityItem36Error("response extension entered source freeze")
    if int(source["counts"]["confirmation_maps_downloaded"]) != 0:
        raise GravityItem36Error("confirmation source map entered Item 36")
    responses = _read_json(paths["exploration_responses"])
    if int(responses["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem36Error("confirmation response entered Item 36")
    compute = _read_json(paths["compute_manifest"])
    if str(compute["result_content_sha256"]) != str(result["content_sha256"]):
        raise GravityItem36Error("compute manifest does not bind Item 36 result")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare-predictors",
        "acquire-source-maps",
        "acquire-responses",
        "run",
        "validate",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.root.resolve()
    if arguments.command == "prepare-predictors":
        print(prepare_predictors(root)["sample_manifest"])
    elif arguments.command == "acquire-source-maps":
        print(acquire_source_maps(root)["source_feature_manifest"])
    elif arguments.command == "acquire-responses":
        print(acquire_responses(root)["exploration_responses"])
    elif arguments.command == "run":
        print(run_experiment(root))
    elif arguments.command == "validate":
        validate_checked(root)
        print("PASS")
    else:
        raise GravityItem36Error("unknown Item 36 command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
