"""Frozen Item 37 action-level alternative-geometry search."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
import io
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
from sigma_theory_compiler.gravity_item32_boundary_focusing import (
    GravityItem32Error,
    _channel_index,
    _maps_payload,
    _quadrant_count,
    _unique_stellar_measurements,
)
from sigma_theory_compiler.gravity_item34_condensate_superfluid import (
    _robust_comparison,
)
from sigma_theory_compiler.gravity_item36_extra_dimensions import (
    _fresh_pool as _item36_fresh_pool,
)
from sigma_theory_compiler.gravity_item36_extra_dimensions import (
    load_config as load_item36_config,
)

CONFIG_PATH = Path("configs/gravity_item37_alternative_geometry_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item37_alternative_geometry.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
TEST_PATH = Path("tests/test_gravity_item37_alternative_geometry.py")
_ADMISSIBLE_CACHE: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


class GravityItem37Error(RuntimeError):
    """Raised when an Item 37 freeze, action, leakage, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item37-alternative-geometry-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 37:
        raise GravityItem37Error("unexpected Item 37 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem37Error("stable gravity goal changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem37Error("raw candidate boundary changed")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem37Error("post-response candidates entered Item 37")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem37Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem37Error("paid calls are outside Item 37")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem37Error("equal-viability policy changed")
    if not bool(
        config["discovery_policy"]["single_empirical_counterexample_is_not_a_formula_family_veto"]
    ):
        raise GravityItem37Error("counterexample policy changed")
    if bool(config["gates"]["single_empirical_counterexample_is_veto"]):
        raise GravityItem37Error("empirical single-counterexample veto entered Item 37")
    if bool(config["gates"]["single_object_sensitive_formula_may_promote"]):
        raise GravityItem37Error("single-object-sensitive promotion entered Item 37")
    if sum(bool(row["action_track_eligible"]) for row in generator["niches"]) != 4:
        raise GravityItem37Error("action eligibility allocation changed")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem37Error(f"scientific dependency changed: {relative}")
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
        raise GravityItem37Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes) or _sha256_bytes(frozen_module) != _sha256_file(
        root / MODULE_PATH
    ):
        raise GravityItem37Error("Item 37 module differs from scientific freeze")


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in ("predictors", "predictor_source_manifest", "sample_manifest", "candidate_manifest"):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem37Error(f"{key} differs from sample freeze")


def verify_source_feature_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["source_feature_freeze_commit"])
    _require_ancestor(root, commit, "source feature freeze")
    paths = _source_paths(root, config)
    for key in ("source_features", "source_feature_manifest"):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem37Error(f"{key} differs from source feature freeze")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    radices = {
        "polarity": len(generator["polarities"]),
        "amplitude": len(generator["amplitudes"]),
        "scale": len(generator["scale_values"]),
        "width": len(generator["transition_widths"]),
        "power": len(generator["powers"]),
        "coupling": len(generator["baryon_couplings"]),
        "latent": len(generator["geometry_mixings"]),
        "side": len(generator["branch_sides"]),
    }
    per_niche = int(generator["raw_candidate_cells"]) // 4
    if int(np.prod(list(radices.values()))) != per_niche:
        raise GravityItem37Error("mixed-radix grammar does not fill each niche exactly")
    pieces: dict[str, list[np.ndarray]] = {"niche": []} | {key: [] for key in radices}
    for niche in range(4):
        working = np.arange(per_niche, dtype=np.int64)
        decoded: dict[str, np.ndarray] = {}
        for key, radix in reversed(list(radices.items())):
            decoded[key] = (working % radix).astype(np.int16)
            working //= radix
        if np.any(working != 0):
            raise GravityItem37Error("candidate decoder overflow")
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
        "scale": xp.asarray(np.asarray(generator["scale_values"])[index["scale"]]),
        "width": xp.asarray(np.asarray(generator["transition_widths"])[index["width"]]),
        "power": xp.asarray(np.asarray(generator["powers"])[index["power"]]),
        "coupling": xp.asarray(np.asarray(generator["baryon_couplings"])[index["coupling"]]),
        "latent": xp.asarray(np.asarray(generator["geometry_mixings"])[index["latent"]]),
        "side": xp.asarray(np.asarray(generator["branch_sides"])[index["side"]]),
    }


def _action_kernel(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    predictors: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> tuple[Any, Any, Any, Any]:
    values = _candidate_values(config, arrays, begin, end, xp)
    shape = (-1, 1)
    niche = values["niche"].reshape(shape)
    polarity = values["polarity"].reshape(shape)
    amplitude = values["amplitude"].reshape(shape)
    scale = values["scale"].reshape(shape)
    width = values["width"].reshape(shape)
    power = values["power"].reshape(shape)
    coupling = values["coupling"].reshape(shape)
    latent = values["latent"].reshape(shape)
    side = values["side"].reshape(shape)
    log_acceleration = xp.asarray(predictors["log_acceleration"])[None, :]
    source_eta = xp.asarray(predictors["source_nonaxisymmetry"])[None, :]
    mode_frequency = xp.asarray(predictors["mode_frequency_ratio"])[None, :]
    torsion = xp.asarray(predictors["torsion_proxy"])[None, :]
    nonmetricity = xp.asarray(predictors["nonmetricity_proxy"])[None, :]
    finsler = xp.asarray(predictors["Finsler_anisotropy_proxy"])[None, :]
    affine = xp.asarray(predictors["affine_holonomy_proxy"])[None, :]
    log_a0 = math.log10(float(config["physics"]["constants"]["a0_m_s2"]))

    log_acceleration_scale = log_a0 + xp.log10(scale)
    acceleration_coordinate = (log_acceleration - log_acceleration_scale) / width
    low_acceleration = 1.0 / (1.0 + xp.power(10.0, power * acceleration_coordinate))
    acceleration_branch = xp.clip(0.5 * (1.0 + side * (2.0 * low_acceleration - 1.0)), 0.0, 1.0)
    local_center = float(config["physics"]["universal_local_shield_log10_acceleration_m_s2"])
    local_power = float(config["physics"]["universal_local_shield_power"])
    local_shield = 1.0 / (1.0 + xp.power(10.0, local_power * (log_acceleration - local_center)))
    common = acceleration_branch * local_shield

    mode_excess = xp.clip(0.5 * (mode_frequency - 1.0), 0.0, 1.0)

    def mixed_feature(feature: Any) -> Any:
        shaped = xp.power(xp.clip(feature, 0.0, 1.0), power)
        return (1.0 - latent) + latent * shaped

    torsion_circular = common * mixed_feature(torsion)
    torsion_mode = common * mixed_feature(
        xp.clip(torsion * (1.0 + coupling * mode_excess) / (1.0 + coupling), 0.0, 1.0)
    )

    nonmetricity_circular = common * mixed_feature(nonmetricity)
    nonmetricity_mode = common * mixed_feature(
        xp.clip(
            nonmetricity * (1.0 + coupling * xp.abs(nonmetricity - torsion)),
            0.0,
            1.0,
        )
    )

    finsler_circular = common * mixed_feature(1.0 - 0.5 * finsler)
    finsler_mode = common * mixed_feature(
        xp.clip(finsler * (1.0 + coupling * mode_excess), 0.0, 1.0)
    )

    affine_circular = common * mixed_feature(affine)
    affine_mode = common * mixed_feature(
        xp.clip(
            affine * (1.0 + coupling * source_eta * mode_excess)
            + 0.25 * coupling * xp.sqrt(xp.maximum(torsion * nonmetricity, 0.0)),
            0.0,
            1.0,
        )
    )

    circular_activation = xp.where(
        niche == 0,
        torsion_circular,
        xp.where(
            niche == 1,
            nonmetricity_circular,
            xp.where(niche == 2, finsler_circular, affine_circular),
        ),
    )
    mode_activation = xp.where(
        niche == 0,
        torsion_mode,
        xp.where(
            niche == 1,
            nonmetricity_mode,
            xp.where(niche == 2, finsler_mode, affine_mode),
        ),
    )
    response_circular = 1.0 + polarity * amplitude * circular_activation
    response_mode = 1.0 + polarity * amplitude * mode_activation
    return response_circular, response_mode, circular_activation, mode_activation


def _candidate_deltas(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    predictors: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    response_circular, response_mode, _, _ = _action_kernel(
        config, arrays, predictors, begin, end, xp
    )
    source_eta = xp.asarray(predictors["source_nonaxisymmetry"])[None, :]
    circular = 0.5 * xp.log10(response_circular)
    ratio = response_mode / response_circular
    noncircular = xp.log10(1.0 + source_eta * ratio) - xp.log10(1.0 + source_eta)
    return xp.stack((circular, noncircular), axis=2)


def _adversarial_predictors() -> dict[str, np.ndarray]:
    index = np.arange(64)
    log_acceleration = np.linspace(-13.5, -8.2, 64)
    source = 0.01 + 0.8 * ((index * 17) % 64) / 63.0
    mode = 1.0 + 2.0 * ((index * 29) % 64) / 63.0
    return {
        "log_acceleration": log_acceleration,
        "source_nonaxisymmetry": source,
        "mode_frequency_ratio": mode,
        "torsion_proxy": 0.01 + 0.98 * ((index * 37) % 64) / 63.0,
        "nonmetricity_proxy": 0.01 + 0.98 * ((index * 43) % 64) / 63.0,
        "Finsler_anisotropy_proxy": 0.01 + 0.98 * ((index * 47) % 64) / 63.0,
        "affine_holonomy_proxy": 0.01 + 0.98 * ((index * 53) % 64) / 63.0,
    }


def _local_predictors(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "log_acceleration": np.asarray(
            [math.log10(float(config["physics"]["constants"]["one_au_acceleration_m_s2"]))]
        ),
        "source_nonaxisymmetry": np.asarray([0.01]),
        "mode_frequency_ratio": np.asarray([2.0]),
        "torsion_proxy": np.asarray([0.01]),
        "nonmetricity_proxy": np.asarray([0.01]),
        "Finsler_anisotropy_proxy": np.asarray([0.01]),
        "affine_holonomy_proxy": np.asarray([0.01]),
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
    minimum_response = np.full(count, np.nan)
    maximum_response = np.full(count, np.nan)
    material = np.full(count, np.nan)
    local_response = np.full(count, np.nan)
    maximum_circular = np.full(count, np.nan)
    maximum_noncircular = np.full(count, np.nan)
    batch = int(config["evaluation"]["candidate_batch_size"])
    gates = config["admissibility"]
    for begin in range(0, count, batch):
        end = min(begin + batch, count)
        response_circular, response_mode, _, _ = _action_kernel(config, raw, domain, begin, end, np)
        delta = _candidate_deltas(config, raw, domain, begin, end, np)
        local_circular, local_mode, _, _ = _action_kernel(config, raw, local, begin, end, np)
        minimum_response[begin:end] = np.minimum(
            np.min(response_circular, axis=1), np.min(response_mode, axis=1)
        )
        maximum_response[begin:end] = np.maximum(
            np.max(response_circular, axis=1), np.max(response_mode, axis=1)
        )
        maximum_circular[begin:end] = np.max(np.abs(delta[:, :, 0]), axis=1)
        maximum_noncircular[begin:end] = np.max(np.abs(delta[:, :, 1]), axis=1)
        material[begin:end] = np.maximum(
            maximum_circular[begin:end], maximum_noncircular[begin:end]
        )
        local_response[begin:end] = np.maximum(
            np.abs(local_circular[:, 0] - 1.0), np.abs(local_mode[:, 0] - 1.0)
        )
        keep[begin:end] = (
            np.all(np.isfinite(delta), axis=(1, 2))
            & (minimum_response[begin:end] >= float(gates["minimum_response_eigenvalue"]))
            & (maximum_response[begin:end] <= float(gates["maximum_response_eigenvalue"]))
            & (
                material[begin:end]
                >= float(gates["minimum_material_circular_or_non_circular_response"])
            )
            & (
                local_response[begin:end]
                <= float(gates["maximum_local_fractional_geometry_response"])
            )
            & (
                maximum_circular[begin:end]
                <= float(gates["maximum_absolute_circular_delta_log10_speed"])
            )
            & (
                maximum_noncircular[begin:end]
                <= float(gates["maximum_absolute_noncircular_delta_log10_ratio"])
            )
        )
    arrays = {key: values[keep] for key, values in raw.items()}
    signatures = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        delta = _candidate_deltas(config, arrays, domain, begin, end, np)
        signatures.append(
            np.round(
                delta.reshape(end - begin, -1),
                int(gates["behavioral_equivalence_precision_decimal_places"]),
            )
        )
    behavior = np.concatenate(signatures) if signatures else np.empty((0, 128))
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
        "minimum_admitted_response_eigenvalue": float(np.min(minimum_response[keep])),
        "maximum_admitted_response_eigenvalue": float(np.max(maximum_response[keep])),
        "minimum_admitted_material_response": float(np.min(material[keep])),
        "maximum_admitted_local_fractional_geometry_response": float(np.max(local_response[keep])),
        "maximum_admitted_circular_delta_log10_speed": float(np.max(maximum_circular[keep])),
        "maximum_admitted_noncircular_delta_log10_ratio": float(np.max(maximum_noncircular[keep])),
    }
    generator = config["candidate_generator"]
    checks = (
        ("expected_raw_candidate_digest", "raw_candidate_digest"),
        ("expected_admissible_candidate_digest", "admissible_candidate_digest"),
        ("expected_admissible_candidates", "admissible_candidates"),
        (
            "expected_behavioral_equivalence_classes_adversarial",
            "behavioral_equivalence_classes_adversarial",
        ),
    )
    for expected_key, observed_key in checks:
        expected = generator.get(expected_key)
        if expected not in (None, "TO_BE_MEASURED", -1) and audit[observed_key] != expected:
            raise GravityItem37Error(f"candidate invariant changed: {expected_key}")
    expected_niches = generator.get("expected_admissible_per_niche")
    if (
        expected_niches
        and all(int(value) >= 0 for value in expected_niches.values())
        and audit["admissible_per_niche"] != expected_niches
    ):
        raise GravityItem37Error("admissible niche counts changed")
    _ADMISSIBLE_CACHE[cache_key] = arrays, audit
    return arrays, audit


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item37-candidate-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "niches": config["candidate_generator"]["niches"],
            "action_proxy": config["physics"]["action_proxy"],
            "variation_proxy": config["physics"]["variation_proxy"],
            "conservation_proxy": config["physics"]["conservation_proxy"],
            "historical_novelty_claimed": False,
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "post_response_cells": 0,
            "audit": audit,
        }
    )


def _fresh_pool(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item36_config = load_item36_config(root)
    inherited, inherited_audit = _item36_fresh_pool(root, item36_config)
    independence = config["independence"]
    if len(inherited) != int(independence["expected_item36_fresh_disk_pool"]):
        raise GravityItem37Error("Item 36 fresh disk pool changed")
    item36_sample = _read_json(root / str(config["sources"]["item36_sample_manifest"]))
    _verify_content_hash(item36_sample, "Item 36 sample manifest")
    roles = item36_sample["objects"]
    if len(roles) != int(independence["expected_item36_roles"]):
        raise GravityItem37Error("Item 36 role count changed")
    role_ids = {str(row["plateifu"]) for row in roles}
    coordinates = np.asarray([[float(row["ra"]), float(row["dec"])] for row in roles])
    post_identity = [row for row in inherited if str(row["plateifu"]) not in role_ids]
    if len(post_identity) != int(independence["expected_post_identity_pool"]):
        raise GravityItem37Error("Item 37 identity exclusion count changed")
    separations = _minimum_separations_arcsec(post_identity, coordinates)
    veto = float(independence["coordinate_veto_arcsec"])
    excluded_coordinates = int(np.count_nonzero(separations <= veto))
    if excluded_coordinates != int(independence["expected_additional_coordinate_exclusions"]):
        raise GravityItem37Error("Item 37 coordinate exclusion count changed")
    fresh = []
    for source, separation in zip(post_identity, separations, strict=True):
        if separation <= veto:
            continue
        row = dict(source)
        row["minimum_item36_role_separation_arcsec"] = float(separation)
        fresh.append(row)
    fresh.sort(key=lambda row: str(row["plateifu"]))
    if len(fresh) != int(independence["expected_fresh_pool"]):
        raise GravityItem37Error("Item 37 fresh pool changed")
    if len(fresh) != int(config["sample"]["expected_fresh_disk_pool"]):
        raise GravityItem37Error("Item 37 disk predictor pool changed")
    audit = {
        "item36_fresh_disk_pool": len(inherited),
        "item36_predecessor_audit": inherited_audit,
        "item36_roles": len(roles),
        "post_identity_pool": len(post_identity),
        "additional_coordinate_exclusions": excluded_coordinates,
        "fresh_disk_pool": len(fresh),
    }
    return fresh, audit


def _sample_manifest(
    config: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    sample = config["sample"]
    ranked_mass = sorted(
        pool,
        key=lambda row: (float(row["log_stellar_mass"]), str(row["plateifu"])),
    )
    if len(ranked_mass) % 4:
        raise GravityItem37Error("Item 37 mass quartiles are not exactly divisible")
    quartile_size = len(ranked_mass) // 4
    cells: dict[str, list[dict[str, Any]]] = {f"mq{quartile}": [] for quartile in range(4)}
    for rank, source in enumerate(ranked_mass):
        row = dict(source)
        mass_quartile = min(rank // quartile_size, 3)
        cell = f"mq{mass_quartile}"
        row.update({"mass_quartile": mass_quartile, "sample_cell": cell})
        cells[cell].append(row)
    capacities = {key: len(values) for key, values in cells.items()}
    expected = {key: int(value) for key, value in sample["expected_cell_capacities"].items()}
    if capacities != expected:
        raise GravityItem37Error("Item 37 response-blind cell capacities changed")
    objects = []
    cell_counts = {}
    selected_count = int(sample["selected_per_mass_quartile"])
    confirmation_count = int(sample["confirmation_per_cell"])
    for cell, values in sorted(cells.items()):
        ranked = sorted(
            values,
            key=lambda row: _hmac_rank(str(sample["role_key"]), f"select|{row['plateifu']}"),
        )
        selected = ranked[:selected_count]
        confirmation_ids = {
            str(row["plateifu"])
            for row in sorted(
                selected,
                key=lambda row: _hmac_rank(
                    str(sample["role_key"]), f"confirmation|{row['plateifu']}"
                ),
            )[:confirmation_count]
        }
        exploration = [row for row in selected if str(row["plateifu"]) not in confirmation_ids]
        exploration = sorted(
            exploration,
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
            "schema_version": "invariant-gravity-item37-sample-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "mass_quartile_boundaries_log10_msun": [
                f"{float(ranked_mass[index]['log_stellar_mass']):.12e}"
                for index in (quartile_size, 2 * quartile_size, 3 * quartile_size)
            ],
            "objects": objects,
            "selected_cell_counts": cell_counts,
            "fold_counts_exploration": {
                str(key): fold_counts[key] for key in range(int(sample["outer_folds"]))
            },
            "counts": {
                "fresh_disk_pool": len(pool),
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
        "fresh_disk_pool": int(sample["expected_fresh_disk_pool"]),
        "selected": int(sample["expected_selected"]),
        "exploration": int(sample["expected_exploration"]),
        "reserved_confirmation": int(sample["expected_confirmation"]),
        "source_map_rows_read": 0,
        "velocity_response_rows_read": 0,
    }
    if sample_manifest["counts"] != expected_counts:
        raise GravityItem37Error("frozen Item 37 sample counts changed")
    columns = [
        *list(pool[0]),
        "mass_quartile",
        "sample_cell",
        "role",
        "outer_fold",
        "source_map_read",
        "velocity_response_read",
    ]
    _write_tsv(paths["predictors"], sample_manifest["objects"], columns)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item37-predictor-source-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "item36_sample": {
                "path": config["sources"]["item36_sample_manifest"],
                "sha256": _sha256_file(root / str(config["sources"]["item36_sample_manifest"])),
            },
            "predecessor_audit": predecessor_audit,
            "counts": {
                "fresh_disk_pool": len(pool),
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


def _verified_hdus(
    compressed_payload: bytes,
    sample_row: Mapping[str, Any],
    config: Mapping[str, Any],
    required_extensions: Sequence[str],
) -> tuple[Any, Any]:
    try:
        from astropy.io import fits

        payload = gzip.decompress(compressed_payload)
        hdus = fits.open(io.BytesIO(payload), memmap=False)
        primary = hdus[0].header
        source = config["sources"]["maps"]
        for key, expected in source["required_primary_headers"].items():
            if str(primary.get(key, "")).strip() != str(expected):
                hdus.close()
                raise GravityItem37Error(f"MaNGA MAPS header {key} changed")
        if str(primary.get("PLATEIFU", "")).strip() != str(sample_row["plateifu"]):
            hdus.close()
            raise GravityItem37Error("MaNGA MAPS plateifu changed")
        if str(primary.get("MANGAID", "")).strip().upper() != str(sample_row["mangaid"]).upper():
            hdus.close()
            raise GravityItem37Error("MaNGA MAPS mangaid changed")
        required = [hdus[str(name)] for name in required_extensions]
        if bool(source["fits_checksum_required"]):
            for hdu in [hdus[0], *required]:
                if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                    hdus.close()
                    raise GravityItem37Error("MaNGA MAPS FITS checksum failed")
        return hdus, primary
    except GravityItem37Error:
        raise
    except (OSError, ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
        raise GravityItem37Error("invalid MaNGA MAPS FITS") from exc


def _source_feature_rows(
    compressed_payload: bytes,
    sample_row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    extensions = tuple(str(value) for value in config["sources"]["maps"]["source_extensions"])
    hdus, _ = _verified_hdus(compressed_payload, sample_row, config, extensions)
    try:
        channels = config["sources"]["maps"]["channels"]
        coordinates = hdus["SPX_ELLCOO"]
        radius = np.asarray(
            coordinates.data[_channel_index(coordinates, str(channels["radius_re"]))],
            dtype=np.float64,
        )
        azimuth = np.asarray(
            coordinates.data[_channel_index(coordinates, str(channels["spaxel_azimuth"]))],
            dtype=np.float64,
        )
        flux = np.asarray(hdus["SPX_MFLUX"].data, dtype=np.float64)
        flux_ivar = np.asarray(hdus["SPX_MFLUX_IVAR"].data, dtype=np.float64)
        snr = np.asarray(hdus["SPX_SNR"].data, dtype=np.float64)
    finally:
        hdus.close()
    source = config["map_source"]
    valid = (
        np.isfinite(radius)
        & np.isfinite(azimuth)
        & np.isfinite(flux)
        & np.isfinite(flux_ivar)
        & np.isfinite(snr)
        & (flux > 0)
        & (flux_ivar > 0)
        & (snr >= float(source["minimum_spaxel_snr"]))
    )
    domain = valid & (radius >= 0.0) & (radius < 1.5)
    if not np.any(domain):
        raise GravityItem37Error("no valid continuum source domain")
    rotated_flux = np.rot90(flux, 2)
    rotated_valid = np.rot90(valid, 2)
    common = domain & rotated_valid & np.isfinite(rotated_flux)
    flux_asymmetry = float(
        np.sum(np.abs(flux[common] - rotated_flux[common]))
        / max(2.0 * np.sum(np.abs(flux[common])), 1e-12)
    )
    radians = np.radians(azimuth[domain])
    domain_weights = flux[domain]
    domain_total = float(np.sum(domain_weights))
    centroid_x = float(np.sum(domain_weights * radius[domain] * np.cos(radians)) / domain_total)
    centroid_y = float(np.sum(domain_weights * radius[domain] * np.sin(radians)) / domain_total)
    centroid_offset = float(math.hypot(centroid_x, centroid_y))
    inner_flux = float(np.sum(flux[domain & (radius < 0.5)]))
    radial_concentration = inner_flux / max(domain_total, 1e-12)

    try:
        from scipy.special import gammainc, gammaincinv
    except ImportError as exc:
        raise GravityItem37Error("Item 37 requires scipy for the frozen Sersic source") from exc
    mass = float(sample_row["stellar_mass_msun"])
    radius_re_kpc = float(sample_row["half_light_radius_kpc"])
    sersic = float(sample_row["sersic_index"])
    gravitational = float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
    acceleration_conversion = 1.0e6 / float(config["physics"]["constants"]["kpc_to_m"])
    frequency_conversion = float(config["physics"]["constants"]["km_s_kpc_to_Gyr_inverse"])
    b_n = float(gammaincinv(2.0 * sersic, 0.5))
    density_global = 3.0 * mass / (4.0 * math.pi * radius_re_kpc**3)
    vertical_frequency = math.sqrt(4.0 * math.pi * gravitational * density_global)
    dn4000 = float(sample_row["dn4000"])
    age_proxy = float(np.clip(0.5 + 5.0 * (dn4000 - 1.0), 0.1, 10.0))
    rows = []
    for annulus_index, (label, bounds) in enumerate(
        zip(source["annulus_labels"], source["annuli_re"], strict=True)
    ):
        lower, upper = (float(value) for value in bounds)
        selected = valid & (radius >= lower) & (radius < upper)
        count = int(np.count_nonzero(selected))
        if count < int(source["minimum_source_pixels_per_annulus"]):
            raise GravityItem37Error(f"insufficient continuum source pixels in {label}")
        weights = flux[selected]
        total = float(np.sum(weights))
        radius_value = float(np.sum(weights * radius[selected]) / total)
        theta = np.radians(azimuth[selected])
        modes = [
            float(abs(np.sum(weights * np.exp(1j * order * theta))) / total)
            for order in range(1, 4)
        ]
        nonaxisymmetry = float(np.clip(math.sqrt(sum(value * value for value in modes)), 0.0, 1.0))
        mode_total = sum(modes)
        mode_frequency_ratio = (
            float(sum((index + 1) * value for index, value in enumerate(modes)) / mode_total)
            if mode_total > 1e-12
            else 1.0
        )
        radius_kpc = radius_value * radius_re_kpc
        enclosed_fraction = float(gammainc(2.0 * sersic, b_n * radius_value ** (1.0 / sersic)))
        enclosed_mass = mass * enclosed_fraction
        baryonic_speed = math.sqrt(gravitational * enclosed_mass / radius_kpc)
        acceleration = gravitational * enclosed_mass / radius_kpc**2 * acceleration_conversion
        orbital_frequency = baryonic_speed / radius_kpc * frequency_conversion
        row = dict(sample_row)
        row.update(
            {
                "annulus": str(label),
                "annulus_index": annulus_index,
                "source_pixels": count,
                "weighted_radius_re": radius_value,
                "flux_m1": modes[0],
                "flux_m2": modes[1],
                "flux_m3": modes[2],
                "source_nonaxisymmetry": nonaxisymmetry,
                "flux_asymmetry": flux_asymmetry,
                "centroid_offset_re": centroid_offset,
                "radial_concentration": radial_concentration,
                "radius_kpc": radius_kpc,
                "enclosed_stellar_mass_msun": enclosed_mass,
                "enclosed_stellar_mass_fraction": enclosed_fraction,
                "log_baryonic_speed_km_s": math.log10(baryonic_speed),
                "log_acceleration": math.log10(acceleration),
                "omega_Gyr_inverse": orbital_frequency,
                "log_omega_Gyr_inverse": math.log10(orbital_frequency),
                "mode_frequency_ratio": mode_frequency_ratio,
                "age_gyr_proxy": age_proxy,
                "vertical_to_orbital_frequency": vertical_frequency / orbital_frequency,
            }
        )
        rows.append(row)
    return rows


def _add_geometry_source_predictors(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(rows) != 2:
        raise GravityItem37Error("Item 37 geometry predictors require exactly two annuli")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["annulus_index"]))
    log_radius = np.log10(
        np.maximum([float(row["radius_kpc"]) for row in ordered], 1e-8)
    )
    log_mass = np.log10(
        np.maximum([float(row["enclosed_stellar_mass_msun"]) for row in ordered], 1.0)
    )
    denominator = max(float(log_radius[1] - log_radius[0]), 1e-8)
    enclosed_slope = float(np.clip((log_mass[1] - log_mass[0]) / denominator, 0.0, 3.0))
    for row in ordered:
        source_eta = float(np.clip(float(row["source_nonaxisymmetry"]), 0.0, 1.0))
        radius_coordinate = float(np.clip(float(row["weighted_radius_re"]) / 1.5, 0.0, 1.0))
        slope_coordinate = enclosed_slope / 3.0
        torsion = float(
            np.clip(
                source_eta * (0.5 + 0.5 * radius_coordinate) * (0.5 + 0.5 * slope_coordinate),
                0.0,
                1.0,
            )
        )
        nonmetricity = float(np.clip(abs(2.0 - enclosed_slope) / 2.0, 0.0, 1.0))
        flattening = float(np.clip((1.0 - float(row["axis_ratio"])) / 0.7, 0.0, 1.0))
        finsler = float(np.clip(source_eta * flattening, 0.0, 1.0))
        centroid = float(np.clip(float(row["centroid_offset_re"]) / 0.5, 0.0, 1.0))
        affine = float(
            np.clip(math.sqrt(max(torsion * nonmetricity, 0.0)) * (0.5 + 0.5 * centroid), 0.0, 1.0)
        )
        row.update(
            {
                "radial_enclosed_mass_slope": enclosed_slope,
                "torsion_proxy": torsion,
                "nonmetricity_proxy": nonmetricity,
                "Finsler_anisotropy_proxy": finsler,
                "affine_holonomy_proxy": affine,
            }
        )
    return ordered


def _source_control_spec(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    labels = (
        "log_acceleration",
        "weighted_radius_re",
        "radial_enclosed_mass_slope",
        "mode_frequency_ratio",
        "source_nonaxisymmetry",
        "torsion_proxy",
        "nonmetricity_proxy",
        "Finsler_anisotropy_proxy",
        "affine_holonomy_proxy",
    )
    probabilities = [float(value) for value in config["evaluation"]["geometry_control_quantiles"]]
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
    _verify_content_hash(sample_manifest, "Item 37 sample manifest")
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
            raise GravityItem37Error("confirmation identity entered source-map access")
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
            rows = _add_geometry_source_predictors(
                _source_feature_rows(payload, sample_row, config)
            )
        except (GravityItem32Error, GravityItem37Error) as exc:
            failures.append({"plateifu": identity, "reason": str(exc)})
            continue
        source_rows.extend(rows)
        files.append({**file_record, "source_extension_checksums_verified": True})
    if not source_rows:
        raise GravityItem37Error("no Item 37 source-only MAPS features were extracted")
    touched = {str(row["plateifu"]) for row in [*files, *failures]}
    if confirmations & touched:
        raise GravityItem37Error("confirmation MAPS entered Item 37 source phase")
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
        "radial_enclosed_mass_slope",
        "torsion_proxy",
        "nonmetricity_proxy",
        "Finsler_anisotropy_proxy",
        "affine_holonomy_proxy",
    ]
    columns = [*list(exploration[0]), *new_columns]
    _write_tsv(paths["source_features"], source_rows, columns)
    complete_galaxies = len({str(row["plateifu"]) for row in source_rows})
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item37-source-feature-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "data_model": config["sources"]["maps"]["data_model"],
            "files": files,
            "failures": failures,
            "source_features_sha256": _sha256_file(paths["source_features"]),
            "ordinary_geometry_control_spec": _source_control_spec(source_rows, config),
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
                "geometry_thresholds_frozen_before_response": True,
                "confirmation_opened": False,
            },
        }
    )
    _write_json(paths["source_feature_manifest"], manifest)
    return paths


def _inclination_sine(axis_ratio: float, config: Mapping[str, Any]) -> float:
    intrinsic = float(config["response"]["intrinsic_disk_axis_ratio"])
    cosine_squared = float(np.clip((axis_ratio**2 - intrinsic**2) / (1.0 - intrinsic**2), 0.0, 1.0))
    return max(
        math.sqrt(max(1.0 - cosine_squared, 0.0)),
        float(config["response"]["minimum_sine_inclination"]),
    )


def _harmonic_fit_order_three(
    velocity: np.ndarray,
    radius: np.ndarray,
    azimuth_degrees: np.ndarray,
    inverse_variance: np.ndarray,
    bounds: Sequence[float],
    minimum_count: int,
    minimum_quadrants: int,
    minimum_circular_amplitude: float,
    maximum_condition: float,
    maximum_noncircular_ratio: float,
    sine_inclination: float,
    channel: str,
    annulus: str,
) -> dict[str, Any]:
    lower, upper = (float(value) for value in bounds)
    selected = (
        np.isfinite(velocity)
        & np.isfinite(radius)
        & np.isfinite(azimuth_degrees)
        & np.isfinite(inverse_variance)
        & (inverse_variance > 0)
        & (radius >= lower)
        & (radius < upper)
    )
    count = int(np.count_nonzero(selected))
    if count < minimum_count:
        raise GravityItem37Error(f"insufficient {channel} measurements in {annulus}")
    azimuth = azimuth_degrees[selected]
    quadrants = _quadrant_count(azimuth)
    if quadrants < minimum_quadrants:
        raise GravityItem37Error(f"insufficient {channel} azimuth coverage in {annulus}")
    theta = np.radians(azimuth)
    design = np.column_stack(
        [
            np.ones(count),
            np.cos(theta),
            np.sin(theta),
            np.cos(2.0 * theta),
            np.sin(2.0 * theta),
            np.cos(3.0 * theta),
            np.sin(3.0 * theta),
        ]
    )
    weights = inverse_variance[selected]
    normal = design.T @ (weights[:, None] * design)
    condition = float(np.linalg.cond(normal))
    if not math.isfinite(condition) or condition > maximum_condition:
        raise GravityItem37Error(f"ill-conditioned {channel} harmonic fit in {annulus}")
    coefficient = np.linalg.solve(normal, design.T @ (weights * velocity[selected]))
    circular_los = abs(float(coefficient[1]))
    if circular_los < minimum_circular_amplitude:
        raise GravityItem37Error(f"weak {channel} circular amplitude in {annulus}")
    noncircular_amplitude = float(np.linalg.norm(coefficient[[2, 3, 4, 5, 6]]))
    ratio = noncircular_amplitude / circular_los
    if not math.isfinite(ratio) or ratio > maximum_noncircular_ratio:
        raise GravityItem37Error(f"extreme {channel} noncircular ratio in {annulus}")
    circular_speed = circular_los / sine_inclination
    return {
        "annulus": annulus,
        "measurements": count,
        "azimuth_quadrants": quadrants,
        "condition_number": condition,
        "coefficients_km_s": [float(value) for value in coefficient],
        "circular_los_amplitude_km_s": circular_los,
        "circular_speed_km_s": circular_speed,
        "noncircular_amplitude_km_s": noncircular_amplitude,
        "noncircular_ratio": ratio,
        "log10_circular_speed_km_s": math.log10(circular_speed),
        "log10p_noncircular_ratio": math.log10(1.0 + ratio),
    }


def _response_annuli(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    sample_row: Mapping[str, Any],
    config: Mapping[str, Any],
    channel: str,
) -> tuple[bool, list[dict[str, Any]], str | None]:
    response = config["response"]
    prefix = "stellar" if channel == "stellar" else "Halpha"
    try:
        summaries = [
            _harmonic_fit_order_three(
                *values,
                bounds,
                int(response[f"minimum_{prefix}_bins_per_annulus"])
                if channel == "stellar"
                else int(response["minimum_Halpha_spaxels_per_annulus"]),
                int(response[f"minimum_{prefix}_azimuth_quadrants"]),
                float(response[f"minimum_{prefix}_circular_amplitude_km_s"]),
                float(response["maximum_design_condition_number"]),
                float(response["maximum_non_circular_ratio"]),
                _inclination_sine(float(sample_row["axis_ratio"]), config),
                channel,
                str(label),
            )
            for label, bounds in zip(
                config["map_source"]["annulus_labels"],
                config["map_source"]["annuli_re"],
                strict=True,
            )
        ]
    except GravityItem37Error as exc:
        return False, [], str(exc)
    return True, summaries, None


def _response_record(
    compressed_payload: bytes,
    sample_row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    extensions = tuple(
        dict.fromkeys(["SPX_ELLCOO", *config["sources"]["maps"]["response_extensions"]])
    )
    hdus, primary = _verified_hdus(compressed_payload, sample_row, config, extensions)
    try:
        channels = config["sources"]["maps"]["channels"]
        spaxel_coordinates = hdus["SPX_ELLCOO"]
        bin_coordinates = hdus["BIN_LWELLCOO"]
        radius = np.asarray(
            spaxel_coordinates.data[_channel_index(spaxel_coordinates, str(channels["radius_re"]))],
            dtype=np.float64,
        )
        azimuth = np.asarray(
            spaxel_coordinates.data[
                _channel_index(spaxel_coordinates, str(channels["spaxel_azimuth"]))
            ],
            dtype=np.float64,
        )
        stellar_radius = np.asarray(
            bin_coordinates.data[_channel_index(bin_coordinates, str(channels["radius_re"]))],
            dtype=np.float64,
        )
        stellar_azimuth = np.asarray(
            bin_coordinates.data[_channel_index(bin_coordinates, str(channels["bin_azimuth"]))],
            dtype=np.float64,
        )
        bin_hdu = hdus["BINID"]
        stellar_bin = np.asarray(
            bin_hdu.data[_channel_index(bin_hdu, str(channels["stellar_bin_id"]))],
            dtype=np.int64,
        )
        stellar_velocity = np.asarray(hdus["STELLAR_VEL"].data, dtype=np.float64)
        stellar_ivar = np.asarray(hdus["STELLAR_VEL_IVAR"].data, dtype=np.float64)
        stellar_mask = np.asarray(hdus["STELLAR_VEL_MASK"].data, dtype=np.int64)
        stellar_snr = np.asarray(hdus["BIN_SNR"].data, dtype=np.float64)
        stellar_fom = hdus["STELLAR_FOM"]
        stellar_rchi2 = np.asarray(
            stellar_fom.data[_channel_index(stellar_fom, str(channels["stellar_rchi2"]))],
            dtype=np.float64,
        )
        response = config["response"]
        stellar_valid = (
            (stellar_bin >= 0)
            & (stellar_mask == 0)
            & (stellar_ivar > 0)
            & (stellar_snr >= float(response["minimum_stellar_bin_snr"]))
            & (stellar_rchi2 >= 0)
            & (stellar_rchi2 <= float(response["maximum_stellar_rchi2"]))
            & np.isfinite(stellar_velocity)
            & np.isfinite(stellar_radius)
            & np.isfinite(stellar_azimuth)
            & np.isfinite(stellar_ivar)
            & np.isfinite(stellar_snr)
            & np.isfinite(stellar_rchi2)
        )
        stellar_values = _unique_stellar_measurements(
            stellar_bin,
            stellar_valid,
            stellar_ivar,
            stellar_velocity,
            stellar_radius,
            stellar_azimuth,
        )
        halpha_channel = _channel_index(hdus["EMLINE_GVEL"], str(channels["halpha"]))
        halpha_velocity = np.asarray(hdus["EMLINE_GVEL"].data[halpha_channel], dtype=np.float64)
        halpha_ivar = np.asarray(hdus["EMLINE_GVEL_IVAR"].data[halpha_channel], dtype=np.float64)
        halpha_mask = np.asarray(hdus["EMLINE_GVEL_MASK"].data[halpha_channel], dtype=np.int64)
        halpha_anr = np.asarray(hdus["EMLINE_GANR"].data[halpha_channel], dtype=np.float64)
        halpha_ew = np.asarray(hdus["EMLINE_GEW"].data[halpha_channel], dtype=np.float64)
        halpha_ew_mask = np.asarray(hdus["EMLINE_GEW_MASK"].data[halpha_channel], dtype=np.int64)
        halpha_rchi2 = np.asarray(hdus["EMLINE_LFOM"].data[halpha_channel], dtype=np.float64)
        halpha_valid = (
            (halpha_mask == 0)
            & (halpha_ew_mask == 0)
            & (halpha_ivar > 0)
            & (halpha_anr >= float(response["minimum_Halpha_anr"]))
            & (halpha_ew >= float(response["minimum_Halpha_ew_angstrom"]))
            & (halpha_rchi2 >= 0)
            & (halpha_rchi2 <= float(response["maximum_Halpha_rchi2"]))
            & np.isfinite(halpha_velocity)
            & np.isfinite(radius)
            & np.isfinite(azimuth)
            & np.isfinite(halpha_ivar)
            & np.isfinite(halpha_anr)
            & np.isfinite(halpha_ew)
            & np.isfinite(halpha_rchi2)
        )
        halpha_values = (
            halpha_velocity[halpha_valid],
            radius[halpha_valid],
            azimuth[halpha_valid],
            halpha_ivar[halpha_valid],
        )
        drp3qual = int(primary.get("DRP3QUAL", -1))
        dapqual = int(primary.get("DAPQUAL", -1))
    finally:
        hdus.close()
    stellar_pass, stellar_annuli, stellar_reason = _response_annuli(
        stellar_values, sample_row, config, "stellar"
    )
    halpha_pass, halpha_annuli, halpha_reason = _response_annuli(
        halpha_values, sample_row, config, "Halpha"
    )
    return {
        "plateifu": sample_row["plateifu"],
        "mangaid": sample_row["mangaid"],
        "drp3qual": drp3qual,
        "dapqual": dapqual,
        "sine_inclination": _inclination_sine(float(sample_row["axis_ratio"]), config),
        "stellar_quality_pass": stellar_pass,
        "stellar_quality_reason": stellar_reason,
        "stellar_annuli": stellar_annuli,
        "Halpha_quality_pass": halpha_pass,
        "Halpha_quality_reason": halpha_reason,
        "Halpha_annuli": halpha_annuli,
    }


def acquire_responses(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    verify_source_feature_freeze(root, config)
    paths = _source_paths(root, config)
    source_manifest = _read_json(paths["source_feature_manifest"])
    _verify_content_hash(source_manifest, "Item 37 source feature manifest")
    if int(source_manifest["counts"]["velocity_arrays_read"]) != 0:
        raise GravityItem37Error("velocity response entered source-feature freeze")
    sample_manifest = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample_manifest, "Item 37 sample manifest")
    sample_by_id = {
        str(row["plateifu"]): row
        for row in sample_manifest["objects"]
        if row["role"] == "exploration"
    }
    confirmation = {
        str(row["plateifu"])
        for row in sample_manifest["objects"]
        if row["role"] == "reserved_confirmation"
    }
    records = []
    failures = []
    files = []
    cache = root / str(config["sources"]["maps"]["raw_cache"])
    for file_record in source_manifest["files"]:
        identity = str(file_record["plateifu"])
        if identity in confirmation or identity not in sample_by_id:
            raise GravityItem37Error("nonexploration identity entered response access")
        path = cache / str(file_record["file_name"])
        if not path.is_file():
            raise GravityItem37Error(f"source-frozen MAPS payload missing: {identity}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != str(file_record["file_sha256"]):
            raise GravityItem37Error(f"source-frozen MAPS checksum changed: {identity}")
        try:
            record = _response_record(payload, sample_by_id[identity], config)
        except GravityItem37Error as exc:
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
            "schema_version": "invariant-gravity-item37-exploration-responses-1.0",
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
            "schema_version": "invariant-gravity-item37-response-source-manifest-1.0",
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
        _verify_content_hash(value, f"Item 37 {key}")
    source_manifest = _read_json(paths["source_feature_manifest"])
    responses = _read_json(paths["exploration_responses"])
    if int(source_manifest["counts"]["velocity_arrays_read"]) != 0:
        raise GravityItem37Error("response leaked into source feature manifest")
    if int(responses["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem37Error("confirmation response entered Item 37")
    source_rows = _read_tsv(paths["source_features"])
    feature_by_key = {(str(row["plateifu"]), int(row["annulus_index"])): row for row in source_rows}
    response_by_id = {str(row["plateifu"]): row for row in responses["records"]}
    primary_rows = []
    transfer_rows = []
    for identity in sorted(response_by_id):
        response = response_by_id[identity]
        if bool(response["Halpha_quality_pass"]):
            for annulus_index, annulus in enumerate(response["Halpha_annuli"]):
                key = (identity, annulus_index)
                if key not in feature_by_key:
                    raise GravityItem37Error("Halpha response lacks frozen source feature")
                primary_rows.append(
                    {
                        **feature_by_key[key],
                        "target_circular": float(annulus["log10_circular_speed_km_s"]),
                        "target_noncircular": float(annulus["log10p_noncircular_ratio"]),
                    }
                )
        if bool(response["Halpha_quality_pass"]) and bool(response["stellar_quality_pass"]):
            for annulus_index, annulus in enumerate(response["stellar_annuli"]):
                key = (identity, annulus_index)
                if key not in feature_by_key:
                    raise GravityItem37Error("stellar response lacks frozen source feature")
                transfer_rows.append(
                    {
                        **feature_by_key[key],
                        "target_stellar_noncircular": float(annulus["log10p_noncircular_ratio"]),
                    }
                )
    primary_rows.sort(key=lambda row: (str(row["plateifu"]), int(row["annulus_index"])))
    transfer_rows.sort(key=lambda row: (str(row["plateifu"]), int(row["annulus_index"])))
    return primary_rows, transfer_rows, responses


def _candidate_predictors(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    labels = (
        "log_acceleration",
        "source_nonaxisymmetry",
        "mode_frequency_ratio",
        "torsion_proxy",
        "nonmetricity_proxy",
        "Finsler_anisotropy_proxy",
        "affine_holonomy_proxy",
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
        pieces.append(_candidate_deltas(config, arrays, predictors, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _base_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.column_stack(
        [
            [float(row["log_baryonic_speed_km_s"]) for row in rows],
            [math.log10(1.0 + float(row["source_nonaxisymmetry"])) for row in rows],
        ]
    )


def _offset_bounds(config: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    mass = config["physics"]["shared_mass_proxy_scale_bounds"]
    noncircular = config["physics"]["noncircular_source_scale_bounds"]
    return (
        (0.5 * math.log10(float(mass[0])), 0.5 * math.log10(float(mass[1]))),
        (math.log10(float(noncircular[0])), math.log10(float(noncircular[1]))),
    )


def _screen_joint_candidates(
    delta: Any,
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    outer_folds = int(config["sample"]["outer_folds"])
    if {int(value) for value in folds} != set(range(outer_folds)):
        raise GravityItem37Error("response-complete folds are incomplete")
    target_xp = xp.asarray(target)
    base_xp = xp.asarray(base)
    prediction = np.empty_like(target)
    selected_indices = []
    selected_train_losses = []
    fitted_offsets = []
    raw_offsets = []
    bounds = _offset_bounds(config)
    for fold in range(outer_folds):
        train_np = np.where(folds != fold)[0]
        held_np = np.where(folds == fold)[0]
        train = xp.asarray(train_np)
        residual = target_xp[train] - base_xp[train]
        channel_losses = []
        channel_offsets = []
        channel_raw_offsets = []
        for channel in range(2):
            values = delta[:, train, channel]
            raw = xp.mean(residual[:, channel][None, :] - values, axis=1)
            fitted = xp.clip(raw, bounds[channel][0], bounds[channel][1])
            loss = xp.mean(
                xp.square(residual[:, channel][None, :] - values - fitted[:, None]),
                axis=1,
            )
            channel_losses.append(loss)
            channel_offsets.append(fitted)
            channel_raw_offsets.append(raw)
        joint_loss = 0.5 * (channel_losses[0] + channel_losses[1])
        index = int(_to_numpy(xp.argmin(joint_loss), xp))
        selected_indices.append(index)
        selected_train_losses.append(float(_to_numpy(joint_loss[index], xp)))
        fold_offsets = [float(_to_numpy(values[index], xp)) for values in channel_offsets]
        fold_raw = [float(_to_numpy(values[index], xp)) for values in channel_raw_offsets]
        fitted_offsets.append(fold_offsets)
        raw_offsets.append(fold_raw)
        correction = _to_numpy(delta[index, xp.asarray(held_np), :], xp)
        prediction[held_np] = base[held_np] + correction + np.asarray(fold_offsets)[None, :]
    return {
        "prediction": prediction,
        "selected_indices": selected_indices,
        "selected_train_losses": selected_train_losses,
        "fitted_offsets": fitted_offsets,
        "raw_offsets": raw_offsets,
    }


def _normalized_columns(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> np.ndarray:
    columns = []
    for label in labels:
        values = np.asarray([float(row[label]) for row in rows], dtype=np.float64)
        center = float(np.median(values))
        scale = max(float(np.std(values)), 1e-8)
        columns.append((values - center) / scale)
    return np.column_stack(columns)


def _baseline_designs(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    normalization = config["evaluation"]["fixed_feature_normalization"]
    fixed = []
    for label, values in normalization.items():
        center, scale = (float(value) for value in values)
        fixed.append(
            (np.asarray([float(row[label]) for row in rows], dtype=np.float64) - center) / scale
        )
    fixed_matrix = np.column_stack(fixed)
    annulus = np.asarray([-1.0 if int(row["annulus_index"]) == 0 else 1.0 for row in rows])[:, None]
    structural = np.column_stack([fixed_matrix[:, :8], annulus])
    morphology_labels = (
        "weighted_radius_re",
        "flux_m1",
        "flux_m2",
        "flux_m3",
        "source_nonaxisymmetry",
    )
    morphology = _normalized_columns(rows, morphology_labels)
    flexible_base = np.column_stack([fixed_matrix, morphology, annulus])
    interactions = np.column_stack(
        [
            flexible_base[:, 0] * flexible_base[:, 1],
            flexible_base[:, 0] * flexible_base[:, 3],
            flexible_base[:, 3] * flexible_base[:, 4],
            flexible_base[:, -2] * flexible_base[:, -1],
            flexible_base[:, 8] * flexible_base[:, -2],
        ]
    )
    flexible = np.column_stack([flexible_base, flexible_base**2, interactions])
    spec = source_manifest["ordinary_geometry_control_spec"]
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
    geometry = np.column_stack([flexible, *smooth, *hinges])
    return {"structural": structural, "flexible": flexible, "ordinary_geometry": geometry}


def _offset_oof(
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    prediction = np.empty_like(target)
    bounds = _offset_bounds(config)
    for fold in range(int(config["sample"]["outer_folds"])):
        train = folds != fold
        held = folds == fold
        for channel in range(target.shape[1]):
            offset = float(
                np.clip(
                    np.mean(target[train, channel] - base[train, channel]),
                    bounds[channel][0],
                    bounds[channel][1],
                )
            )
            prediction[held, channel] = base[held, channel] + offset
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
        "ordinary_geometry": float(config["evaluation"]["ridge_alpha_geometry"]),
    }
    for label, design in designs.items():
        prediction = np.empty_like(target)
        for channel in range(target.shape[1]):
            residual_prediction = _ridge_oof(
                target[:, channel] - base[:, channel],
                folds,
                design,
                alpha[label],
                int(config["sample"]["outer_folds"]),
            )
            prediction[:, channel] = base[:, channel] + residual_prediction
        output[label] = prediction
    return output


def _joint_mse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.mean(np.square(target - prediction), axis=0)))


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
        "shared_acceleration_geometry_scale": float(values["scale"][0]),
        "transition_width": float(values["width"][0]),
        "power": float(values["power"][0]),
        "baryon_coupling": float(values["coupling"][0]),
        "geometry_mixing": float(values["latent"][0]),
        "branch_side": float(values["side"][0]),
        "equivalence_boundary": definition["equivalence"],
    }


def _robust_joint_by_galaxy(
    target: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    identities = sorted({str(row["plateifu"]) for row in rows})
    candidate_errors = []
    reference_errors = []
    for identity in identities:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if str(row["plateifu"]) == identity],
            dtype=int,
        )
        candidate_errors.append(float(np.mean(np.square(target[indices] - candidate[indices]))))
        reference_errors.append(float(np.mean(np.square(target[indices] - reference[indices]))))
    candidate_error = np.asarray(candidate_errors)
    reference_error = np.asarray(reference_errors)
    comparative = candidate_error - reference_error
    full = _improvement(float(np.mean(reference_error)), float(np.mean(candidate_error)))
    order = np.argsort(np.abs(comparative))[::-1]
    leave = np.ones(len(identities), dtype=bool)
    leave[int(order[0])] = False
    leave_improvement = _improvement(
        float(np.mean(reference_error[leave])), float(np.mean(candidate_error[leave]))
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
        float(np.mean(reference_error[trimmed])), float(np.mean(candidate_error[trimmed]))
    )
    return {
        "galaxies": len(identities),
        "counterexample_galaxies": int(np.count_nonzero(candidate_error > reference_error)),
        "counterexample_fraction": float(np.mean(candidate_error > reference_error)),
        "full_improvement": full,
        "single_most_influential_identity": identities[int(order[0])],
        "leave_one_most_influential_improvement": leave_improvement,
        "leave_one_changes_improvement_sign": bool((full >= 0.0) != (leave_improvement >= 0.0)),
        "trim_fraction": float(config["evaluation"]["robust_comparative_trim_fraction"]),
        "trimmed_galaxies": trim_count,
        "trimmed_improvement": trim_improvement,
        "trim_changes_improvement_sign": bool((full >= 0.0) != (trim_improvement >= 0.0)),
        "single_object_sensitive": bool(
            (full >= 0.0) != (leave_improvement >= 0.0)
            or (full >= 0.0) != (trim_improvement >= 0.0)
        ),
        "single_counterexample_is_veto": False,
    }


def _select_full_candidate(
    delta: Any,
    target: np.ndarray,
    base: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> tuple[int, list[float], float]:
    residual = xp.asarray(target - base)
    bounds = _offset_bounds(config)
    losses = []
    offsets = []
    for channel in range(2):
        values = delta[:, :, channel]
        raw = xp.mean(residual[:, channel][None, :] - values, axis=1)
        fitted = xp.clip(raw, bounds[channel][0], bounds[channel][1])
        losses.append(
            xp.mean(
                xp.square(residual[:, channel][None, :] - values - fitted[:, None]),
                axis=1,
            )
        )
        offsets.append(fitted)
    joint = 0.5 * (losses[0] + losses[1])
    index = int(_to_numpy(xp.argmin(joint), xp))
    return (
        index,
        [float(_to_numpy(values[index], xp)) for values in offsets],
        float(_to_numpy(joint[index], xp)),
    )


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
        [float(row["target_stellar_noncircular"]) for row in transfer_rows], dtype=np.float64
    )
    base = np.asarray(
        [math.log10(1.0 + float(row["source_nonaxisymmetry"])) for row in transfer_rows],
        dtype=np.float64,
    )
    folds = np.asarray([int(row["outer_fold"]) for row in transfer_rows], dtype=int)
    prediction = np.empty_like(target)
    offsets = []
    raw_offsets = []
    noncircular_bounds = _offset_bounds(config)[1]
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        if not len(train) or not len(held):
            raise GravityItem37Error("stellar transfer folds are incomplete")
        candidate_index = int(selected_indices[fold])
        correction = _to_numpy(delta[candidate_index, xp.asarray(indices), 1], xp)
        raw = float(np.mean(target[train] - base[train] - correction[train]))
        fitted = float(np.clip(raw, *noncircular_bounds))
        raw_offsets.append(raw)
        offsets.append(fitted)
        prediction[held] = base[held] + correction[held] + fitted
    return {
        "target": target,
        "base": base,
        "folds": folds,
        "prediction": prediction,
        "offsets": offsets,
        "raw_offsets": raw_offsets,
        "selected_indices_from_Halpha": [int(value) for value in selected_indices],
        "formula_reselection_on_stellar": False,
    }


def _scalar_baselines(
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    designs = _baseline_designs(rows, config, source_manifest)
    prediction = np.empty_like(target)
    bounds = _offset_bounds(config)[1]
    for fold in range(int(config["sample"]["outer_folds"])):
        train = folds != fold
        held = folds == fold
        offset = float(np.clip(np.mean(target[train] - base[train]), *bounds))
        prediction[held] = base[held] + offset
    output = {"baryonic_source": prediction}
    alpha = {
        "structural": float(config["evaluation"]["ridge_alpha_structural"]),
        "flexible": float(config["evaluation"]["ridge_alpha_flexible"]),
        "ordinary_geometry": float(config["evaluation"]["ridge_alpha_geometry"]),
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


def _permuted_joint_target(
    target: np.ndarray,
    reference: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    random: np.random.Generator,
) -> np.ndarray:
    residual = target - reference
    shuffled = np.empty_like(residual)
    groups = np.asarray([f"{row['sample_cell']}|{row['annulus']}" for row in rows], dtype=object)
    for channel in range(2):
        for group in sorted(set(groups.tolist())):
            indices = np.where(groups == group)[0]
            shuffled[indices, channel] = residual[random.permutation(indices), channel]
    return reference + shuffled


def _slice_results(
    target: np.ndarray,
    candidate: np.ndarray,
    references: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    definitions = {
        "sample_cell": [str(row["sample_cell"]) for row in rows],
        "annulus": [str(row["annulus"]) for row in rows],
        "mass_quartile": [str(row["mass_quartile"]) for row in rows],
    }
    output = []
    for dimension, labels in definitions.items():
        values = np.asarray(labels, dtype=object)
        for label in sorted(set(labels)):
            selected = values == label
            candidate_mse = _joint_mse(target[selected], candidate[selected])
            reference_mse = {
                key: _joint_mse(target[selected], prediction[selected])
                for key, prediction in references.items()
            }
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
                        reference_mse["baryonic_source"], candidate_mse
                    ),
                    "improvement_vs_ordinary_geometry": _improvement(
                        reference_mse["ordinary_geometry"], candidate_mse
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
        if not len(indices):
            raise GravityItem37Error(f"missing synthetic niche {niche}")
        values = delta[xp.asarray(indices)]
        variance = xp.var(values.reshape(len(indices), -1), axis=1)
        injection_index = int(indices[int(_to_numpy(xp.argmax(variance), xp))])
        target = base + _to_numpy(delta[injection_index], xp)
        selected = _screen_joint_candidates(delta, target, base, folds, config, xp)
        selected_niches = [int(arrays["niche"][index]) for index in selected["selected_indices"]]
        injections.append(
            {
                "injection_index": injection_index,
                "injection_niche": niche,
                "selected_niches": selected_niches,
                "exact_niche_recovered_all_folds": all(value == niche for value in selected_niches),
                "candidate_mse": _joint_mse(target, selected["prediction"]),
                "transfer_reselected_formula": False,
            }
        )
    newtonian = _screen_joint_candidates(delta, base.copy(), base, folds, config, xp)
    baseline = _offset_oof(base.copy(), base, folds, config)
    newtonian_candidate_mse = _joint_mse(base, newtonian["prediction"])
    newtonian_baseline_mse = _joint_mse(base, baseline)
    return {
        "injections": injections,
        "all_injected_niches_recovered": all(
            row["exact_niche_recovered_all_folds"] for row in injections
        ),
        "all_injected_niches_transfer_unchanged": True,
        "GR_candidate_mse": newtonian_candidate_mse,
        "GR_baseline_mse": newtonian_baseline_mse,
        "GR_control_candidate_improves": (
            newtonian_candidate_mse < newtonian_baseline_mse - 1e-16
        ),
    }


def _evaluate(
    root: Path,
    config: Mapping[str, Any],
    primary_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(primary_rows) < 2 * int(config["sample"]["outer_folds"]):
        raise GravityItem37Error("too few Halpha response-complete observations")
    arrays, candidate_audit = _admissible_candidates(config)
    target = np.column_stack(
        [
            [float(row["target_circular"]) for row in primary_rows],
            [float(row["target_noncircular"]) for row in primary_rows],
        ]
    )
    base = _base_matrix(primary_rows)
    folds = np.asarray([int(row["outer_fold"]) for row in primary_rows], dtype=int)
    paths = _source_paths(root, config)
    source_manifest = _read_json(paths["source_feature_manifest"])
    _verify_content_hash(source_manifest, "Item 37 source feature manifest")

    xp, backend, device = _backend()
    xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    delta = _build_candidate_matrix(config, arrays, primary_rows, xp)
    xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    crosscheck = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["niche"]))
    cpu = _candidate_deltas(config, arrays, _candidate_predictors(primary_rows), 0, crosscheck, np)
    gpu = _to_numpy(delta[:crosscheck], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu - gpu)))

    observed = _screen_joint_candidates(delta, target, base, folds, config, xp)
    full_index, full_offsets, full_loss = _select_full_candidate(delta, target, base, config, xp)
    baselines = _baseline_predictions(target, base, folds, primary_rows, config, source_manifest)
    candidate_mse = _joint_mse(target, observed["prediction"])
    baseline_mse = {key: _joint_mse(target, prediction) for key, prediction in baselines.items()}
    improvements = {key: _improvement(value, candidate_mse) for key, value in baseline_mse.items()}
    selected_records = [
        _candidate_record(int(index), config, arrays) for index in observed["selected_indices"]
    ]
    selected_niches = [int(row["niche_index"]) for row in selected_records]
    niche_counts = Counter(selected_niches)
    modal_niche, modal_count = niche_counts.most_common(1)[0]

    transfer = _transfer_selected_candidates(
        delta, primary_rows, transfer_rows, observed["selected_indices"], config, xp
    )
    transfer_baselines = _scalar_baselines(
        transfer["target"],
        transfer["base"],
        transfer["folds"],
        transfer_rows,
        config,
        source_manifest,
    )
    transfer_candidate_mse = _mse(transfer["target"], transfer["prediction"])
    transfer_baseline_mse = {
        key: _mse(transfer["target"], prediction) for key, prediction in transfer_baselines.items()
    }
    transfer_improvements = {
        key: _improvement(value, transfer_candidate_mse)
        for key, value in transfer_baseline_mse.items()
    }

    observed_statistic = improvements["ordinary_geometry"]
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    null_improvements = []
    for _ in range(int(config["evaluation"]["permutation_trials"])):
        null_target = _permuted_joint_target(
            target, baselines["ordinary_geometry"], primary_rows, random
        )
        null_selected = _screen_joint_candidates(delta, null_target, base, folds, config, xp)
        null_baselines = _baseline_predictions(
            null_target, base, folds, primary_rows, config, source_manifest
        )
        null_improvements.append(
            _improvement(
                _joint_mse(null_target, null_baselines["ordinary_geometry"]),
                _joint_mse(null_target, null_selected["prediction"]),
            )
        )
    p_value = (1.0 + sum(value >= observed_statistic for value in null_improvements)) / (
        len(null_improvements) + 1.0
    )
    synthetic = _synthetic_controls(delta, base, folds, config, arrays, xp)
    slices = _slice_results(target, observed["prediction"], baselines, primary_rows)
    flat_rows = []
    for row in primary_rows:
        flat_rows.extend(
            [
                {**row, "plateifu": f"{row['plateifu']}:{row['annulus']}:circular"},
                {**row, "plateifu": f"{row['plateifu']}:{row['annulus']}:noncircular"},
            ]
        )
    robust_observations = _robust_comparison(
        target.reshape(-1),
        observed["prediction"].reshape(-1),
        baselines["ordinary_geometry"].reshape(-1),
        flat_rows,
        config,
    )
    robust_galaxies = _robust_joint_by_galaxy(
        target,
        observed["prediction"],
        baselines["ordinary_geometry"],
        primary_rows,
        config,
    )
    transfer_matrix_target = transfer["target"][:, None]
    transfer_matrix_candidate = transfer["prediction"][:, None]
    transfer_matrix_reference = transfer_baselines["ordinary_geometry"][:, None]
    robust_transfer = _robust_joint_by_galaxy(
        transfer_matrix_target,
        transfer_matrix_candidate,
        transfer_matrix_reference,
        transfer_rows,
        config,
    )

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
    action_eligible = all(bool(row["action_track_eligible"]) for row in selected_records)
    not_single_object_sensitive = not bool(
        robust_galaxies["single_object_sensitive"]
        or robust_transfer["single_object_sensitive"]
    )
    universal_checks = {
        "quality": quality["all_pass"],
        "improvement_vs_baryonic_source": improvements["baryonic_source"]
        >= float(gates["minimum_improvement_vs_baryonic_source"]),
        "improvement_vs_structural": improvements["structural"]
        >= float(gates["minimum_improvement_vs_structural"]),
        "improvement_vs_flexible": improvements["flexible"]
        >= float(gates["minimum_improvement_vs_flexible"]),
        "improvement_vs_ordinary_geometry": improvements["ordinary_geometry"]
        >= float(gates["minimum_improvement_vs_ordinary_geometry"]),
        "all_broad_slices_vs_baryonic": min(broad_baryonic)
        >= float(gates["minimum_each_broad_slice_improvement_vs_baryonic_source"]),
        "stellar_transfer_vs_ordinary_geometry": transfer_improvements["ordinary_geometry"]
        >= float(gates["minimum_stellar_transfer_improvement_vs_ordinary_geometry"]),
        "selection_aware_p": p_value <= float(gates["maximum_selection_aware_permutation_p"]),
        "same_niche_folds": modal_count >= int(gates["minimum_same_niche_folds"]),
        "selected_action_eligible": action_eligible,
        "synthetic_injections": bool(synthetic["all_injected_niches_recovered"]),
        "GR_control": not bool(synthetic["GR_control_candidate_improves"]),
        "cpu_gpu": cpu_gpu_max <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        "not_single_object_sensitive": not_single_object_sensitive,
    }
    phenomenon_checks = {
        "quality": quality["all_pass"],
        "improvement_vs_ordinary_geometry": improvements["ordinary_geometry"]
        >= float(gates["phenomenon_minimum_improvement_vs_ordinary_geometry"]),
        "stellar_transfer": transfer_improvements["ordinary_geometry"]
        >= float(gates["phenomenon_minimum_stellar_transfer_improvement"]),
        "selection_aware_p": p_value <= float(gates["phenomenon_maximum_selection_aware_p"]),
        "synthetic_injections": bool(synthetic["all_injected_niches_recovered"]),
        "cpu_gpu": cpu_gpu_max <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        "not_single_object_sensitive": not_single_object_sensitive,
    }
    partial_slices = [
        row
        for row in slices
        if float(row["improvement_vs_ordinary_geometry"])
        >= float(gates["partial_minimum_slice_improvement_vs_ordinary_geometry"])
    ]
    universal_pass = all(bool(value) for value in universal_checks.values())
    phenomenon_pass = all(bool(value) for value in phenomenon_checks.values())
    if not quality["all_pass"]:
        decision = "INCONCLUSIVE_ITEM37_RESPONSE_QUALITY"
    elif universal_pass:
        decision = "ADVANCE_ITEM37_ALTERNATIVE_GEOMETRY_CANDIDATE"
    elif phenomenon_pass:
        decision = "RETAIN_ITEM37_NONCIRCULAR_PHENOMENON_SIGNAL"
    elif partial_slices:
        decision = "RETAIN_ITEM37_PARTIAL_SLICE_SIGNAL"
    else:
        decision = "NO_ITEM37_ALTERNATIVE_GEOMETRY_LEAD"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item37-alternative-geometry-result-1.0",
            "item": 37,
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
            "full_exploration_offsets": full_offsets,
            "full_exploration_training_loss": full_loss,
            "primary_Halpha": {
                "candidate_mse": candidate_mse,
                "baseline_mse": baseline_mse,
                "improvements": improvements,
                "selection_aware_permutation_p": p_value,
                "null_improvements": null_improvements,
                "robust_observation_channels": robust_observations,
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
                "selected_action_track_eligible": action_eligible,
                "positive_bounded_response_all_admitted": (
                    candidate_audit["minimum_admitted_response_eigenvalue"]
                    >= float(config["admissibility"]["minimum_response_eigenvalue"])
                ),
                "maximum_local_fractional_geometry_response": candidate_audit[
                    "maximum_admitted_local_fractional_geometry_response"
                ],
                "explicit_GR_equivalence_boundary": True,
                "mass_and_composition_independent_kernel": True,
                "quadratic_conservation_proxy_only": True,
                "complete_connection_field_equations_proved": False,
                "lensing_law_proved": False,
                "relativistic_completion_proved": False,
                "historical_novelty_claimed": False,
            },
            "universal_track_checks": universal_checks,
            "universal_track_pass": universal_pass,
            "phenomenon_track_checks": phenomenon_checks,
            "phenomenon_track_pass": phenomenon_pass,
            "paper_claim_authorized": False,
            "counterexample_policy": {
                "single_empirical_counterexample_is_veto": False,
                "hard_theoretical_violation_can_be_single_case_veto": True,
                "formula_family_rejected_from_one_empirical_object": False,
                "single_object_sensitive_formula_may_promote": False,
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
                "approximate_candidate_observation_channel_evaluations": int(
                    len(arrays["niche"])
                    * len(primary_rows)
                    * 2
                    * (1 + int(config["evaluation"]["permutation_trials"]))
                ),
                "confirmation_calls": 0,
                "paid_model_calls": 0,
                "estimated_api_spend_usd": 0.0,
            },
            "next_step": (
                "Keep confirmations sealed; any paper claim requires an unchanged fresh replication. "
                "Advance the ordered roadmap to Item 38 after recording this exploration."
            ),
        }
    )
    compute_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item37-compute-manifest-1.0",
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
    primary_rows, transfer_rows, responses = _load_experiment_data(root, config)
    result, compute_manifest = _evaluate(root, config, primary_rows, transfer_rows, responses)
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
        _verify_content_hash(manifest, f"Item 37 {key}")
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "Item 37 result")
    if result.get("schema_version") != "invariant-gravity-item37-alternative-geometry-result-1.0":
        raise GravityItem37Error("unexpected Item 37 result schema")
    if int(result.get("item", -1)) != 37:
        raise GravityItem37Error("unexpected Item 37 result item")
    allowed = {
        "INCONCLUSIVE_ITEM37_RESPONSE_QUALITY",
        "ADVANCE_ITEM37_ALTERNATIVE_GEOMETRY_CANDIDATE",
        "RETAIN_ITEM37_NONCIRCULAR_PHENOMENON_SIGNAL",
        "RETAIN_ITEM37_PARTIAL_SLICE_SIGNAL",
        "NO_ITEM37_ALTERNATIVE_GEOMETRY_LEAD",
    }
    if result.get("decision") not in allowed:
        raise GravityItem37Error("unexpected Item 37 decision")
    if int(result["sample"]["confirmation_values_read"]) != 0:
        raise GravityItem37Error("confirmation value entered Item 37 result")
    if bool(result["counterexample_policy"]["single_empirical_counterexample_is_veto"]):
        raise GravityItem37Error("one-object empirical veto entered Item 37 result")
    if bool(result["counterexample_policy"]["formula_family_rejected_from_one_empirical_object"]):
        raise GravityItem37Error("formula family rejected from one empirical object")
    if bool(result["paper_claim_authorized"]):
        raise GravityItem37Error("unreplicated paper claim entered Item 37")
    if int(result["compute"]["paid_model_calls"]) != 0:
        raise GravityItem37Error("paid model call entered Item 37")
    if int(result["compute"]["confirmation_calls"]) != 0:
        raise GravityItem37Error("confirmation computation entered Item 37")
    if float(result["compute"]["cpu_gpu_max_absolute_difference"]) > float(
        result["compute"]["cpu_gpu_tolerance"]
    ):
        raise GravityItem37Error("Item 37 CPU/GPU crosscheck failed")
    sample_manifest = _read_json(paths["sample_manifest"])
    if sample_manifest["counts"] != {
        "fresh_disk_pool": int(config["sample"]["expected_fresh_disk_pool"]),
        "selected": int(config["sample"]["expected_selected"]),
        "exploration": int(config["sample"]["expected_exploration"]),
        "reserved_confirmation": int(config["sample"]["expected_confirmation"]),
        "source_map_rows_read": 0,
        "velocity_response_rows_read": 0,
    }:
        raise GravityItem37Error("Item 37 sample counts changed")
    source_manifest = _read_json(paths["source_feature_manifest"])
    if int(source_manifest["counts"]["response_extensions_read"]) != 0:
        raise GravityItem37Error("response extension entered source freeze")
    if int(source_manifest["counts"]["confirmation_maps_downloaded"]) != 0:
        raise GravityItem37Error("confirmation source map entered Item 37")
    responses = _read_json(paths["exploration_responses"])
    if int(responses["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem37Error("confirmation response entered Item 37")
    compute = _read_json(paths["compute_manifest"])
    if str(compute["result_content_sha256"]) != str(result["content_sha256"]):
        raise GravityItem37Error("compute manifest does not bind Item 37 result")


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
        paths = prepare_predictors(root)
        print(paths["sample_manifest"])
    elif arguments.command == "acquire-source-maps":
        paths = acquire_source_maps(root)
        print(paths["source_feature_manifest"])
    elif arguments.command == "acquire-responses":
        paths = acquire_responses(root)
        print(paths["exploration_responses"])
    elif arguments.command == "run":
        print(run_experiment(root))
    elif arguments.command == "validate":
        validate_checked(root)
        print("PASS")
    else:
        raise GravityItem37Error("unknown Item 37 command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
