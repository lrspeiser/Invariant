"""Frozen Item 33 universal critical-variable and phase-transition search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import time
import urllib.parse
import urllib.request
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
    _baryonic_virial_prediction,
    _baseline_predictions,
    _candidate_digest,
    _design_matrix,
    _feature_matrix,
    _ridge_oof,
    _screen_candidate_matrix,
    _virial_oof,
)

CONFIG_PATH = Path("configs/gravity_item33_phase_transition_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item33_phase_transition.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
_ADMISSIBLE_CACHE: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


class GravityItem33Error(RuntimeError):
    """Raised when an Item 33 freeze, leakage, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item33-phase-transition-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 33:
        raise GravityItem33Error("unexpected Item 33 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem33Error("stable gravity goal changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem33Error("raw candidate boundary changed")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem33Error("post-response candidates entered Item 33")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem33Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem33Error("paid calls are outside Item 33")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem33Error("equal-viability policy changed")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem33Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "exploration_responses",
        "response_source_manifest",
        "compute_manifest",
    )
    return {key: base / str(config["paths"][key]) for key in keys}


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem33Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem33Error("could not read frozen Item 33 module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem33Error("Item 33 module differs from scientific freeze")


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in ("predictors", "predictor_source_manifest", "sample_manifest", "candidate_manifest"):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem33Error(f"{key} differs from sample freeze")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    radices = {
        "polarity": len(generator["polarities"]),
        "amplitude": len(generator["amplitudes"]),
        "threshold": int(generator["threshold_cells"]),
        "width": len(generator["transition_widths"]),
        "critical": len(generator["critical_exponents"]),
        "secondary": len(generator["secondary_couplings"]),
        "latent": len(generator["latent_strengths"]),
        "side": len(generator["phase_sides"]),
    }
    per_niche = int(generator["raw_candidate_cells"]) // 4
    if int(np.prod(list(radices.values()))) != per_niche:
        raise GravityItem33Error("mixed-radix grammar does not fill each niche exactly")
    pieces: dict[str, list[np.ndarray]] = {"niche": []} | {key: [] for key in radices}
    for niche in range(4):
        working = np.arange(per_niche, dtype=np.int64)
        decoded: dict[str, np.ndarray] = {}
        for key, radix in reversed(list(radices.items())):
            decoded[key] = (working % radix).astype(np.int16)
            working //= radix
        if np.any(working != 0):
            raise GravityItem33Error("candidate decoder overflow")
        pieces["niche"].append(np.full(per_niche, niche, dtype=np.int16))
        for key in radices:
            pieces[key].append(decoded[key])
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    order = random.permutation(len(arrays["niche"]))
    return {key: value[order] for key, value in arrays.items()}


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    index = {key: arrays[key][begin:end] for key in arrays}
    threshold = index["threshold"]
    return {
        "niche": xp.asarray(index["niche"]),
        "polarity": xp.asarray(np.asarray(generator["polarities"])[index["polarity"]]),
        "amplitude": xp.asarray(np.asarray(generator["amplitudes"])[index["amplitude"]]),
        "threshold_index": xp.asarray(threshold),
        "acceleration_threshold": xp.asarray(
            np.asarray(generator["acceleration_log10_thresholds_m_s2"])[threshold]
        ),
        "density_threshold": xp.asarray(
            np.asarray(generator["density_log10_thresholds_msun_kpc3"])[threshold]
        ),
        "environment_threshold": xp.asarray(
            np.asarray(generator["environment_q_lss_thresholds"])[threshold]
        ),
        "organization_threshold": xp.asarray(
            np.asarray(generator["organization_thresholds"])[threshold]
        ),
        "width": xp.asarray(np.asarray(generator["transition_widths"])[index["width"]]),
        "critical": xp.asarray(np.asarray(generator["critical_exponents"])[index["critical"]]),
        "secondary": xp.asarray(np.asarray(generator["secondary_couplings"])[index["secondary"]]),
        "latent": xp.asarray(np.asarray(generator["latent_strengths"])[index["latent"]]),
        "side": xp.asarray(np.asarray(generator["phase_sides"])[index["side"]]),
    }


def _phase_predictors(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "log_acceleration": np.log10(
            np.asarray([float(row["internal_acceleration_m_s2"]) for row in rows])
        ),
        "log_density": np.asarray([float(row["log_mean_stellar_density"]) for row in rows]),
        "log_potential": np.asarray([float(row["log_dimensionless_potential"]) for row in rows]),
        "q_lss": np.asarray([float(row["gema_q_lss"]) for row in rows]),
        "age": (np.asarray([float(row["dn4000"]) for row in rows]) - 1.6) / 0.3,
        "log_surface_density": np.asarray([float(row["log_surface_density"]) for row in rows]),
        "axis_ratio": np.asarray([float(row["axis_ratio"]) for row in rows]),
    }


def _candidate_activation(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    predictors: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
    *,
    return_convergence: bool = False,
) -> Any:
    values = _candidate_values(config, arrays, begin, end, xp)
    shape = (-1, 1)
    niche = values["niche"].reshape(shape)
    side = values["side"].reshape(shape)
    width = values["width"].reshape(shape)
    critical = values["critical"].reshape(shape)
    secondary = values["secondary"].reshape(shape)
    latent = values["latent"].reshape(shape)
    log_acceleration = xp.asarray(predictors["log_acceleration"])[None, :]
    log_density = xp.asarray(predictors["log_density"])[None, :]
    log_potential = xp.asarray(predictors["log_potential"])[None, :]
    q_lss = xp.asarray(predictors["q_lss"])[None, :]
    age = xp.asarray(predictors["age"])[None, :]
    log_surface = xp.asarray(predictors["log_surface_density"])[None, :]
    axis_ratio = xp.asarray(predictors["axis_ratio"])[None, :]

    acceleration_distance = (
        log_acceleration
        - values["acceleration_threshold"].reshape(shape)
        + secondary * (q_lss + 2.1) / 2.0
    ) / width
    density_distance = (
        log_density - values["density_threshold"].reshape(shape) + secondary * (log_potential + 6.5)
    ) / width
    history_distance = (
        q_lss
        - values["environment_threshold"].reshape(shape)
        + secondary * (log_acceleration + 10.5)
        + 0.35 * secondary * age
    ) / width
    acceleration_distance -= 0.5 * latent * xp.tanh(acceleration_distance)
    density_distance -= 0.5 * latent * xp.tanh(density_distance)
    history_distance -= 0.7 * latent * xp.tanh(age) * side
    acceleration_phase = xp.power(0.5 * (1.0 - xp.tanh(side * acceleration_distance)), critical)
    density_phase = xp.power(0.5 * (1.0 - xp.tanh(side * density_distance)), critical)
    history_phase = xp.power(0.5 * (1.0 - xp.tanh(side * history_distance)), critical)

    organization = (
        (log_surface - 8.8) / 1.2
        + secondary * ((axis_ratio - 0.65) / 0.25 + 0.5 * age)
        - values["organization_threshold"].reshape(shape)
    ) / width
    order = side * xp.ones_like(organization)
    coupling = 0.8 + 0.8 * latent
    iterations = int(config["admissibility"]["landau_fixed_point_iterations"])
    for _ in range(iterations):
        order = xp.tanh(organization + coupling * order)
    next_order = xp.tanh(organization + coupling * order)
    convergence = xp.max(xp.abs(next_order - order), axis=1)
    landau_phase = xp.power(xp.clip(0.5 * (1.0 - side * order), 0.0, 1.0), critical)
    activation = xp.where(
        niche == 0,
        acceleration_phase,
        xp.where(niche == 1, density_phase, xp.where(niche == 2, history_phase, landau_phase)),
    )
    shield_center = float(config["physics"]["universal_local_shield_log10_acceleration_m_s2"])
    shield_power = float(config["physics"]["universal_local_shield_power"])
    local_shield = 1.0 / (1.0 + xp.power(10.0, shield_power * (log_acceleration - shield_center)))
    activation = xp.clip(activation * local_shield, 0.0, 1.0)
    if return_convergence:
        return activation, xp.where(values["niche"] == 3, convergence, 0.0)
    return activation


def _candidate_delta_log10_sigma(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    predictors: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    values = _candidate_values(config, arrays, begin, end, xp)
    activation = _candidate_activation(config, arrays, predictors, begin, end, xp)
    mu = 1.0 + values["polarity"][:, None] * values["amplitude"][:, None] * activation
    return 0.5 * xp.log10(mu)


def _adversarial_predictors() -> dict[str, np.ndarray]:
    index = np.arange(64)
    return {
        "log_acceleration": np.linspace(-13.0, -8.5, 64),
        "log_density": 5.0 + 5.5 * ((index * 17) % 64) / 63.0,
        "log_potential": -8.5 + 4.0 * ((index * 29) % 64) / 63.0,
        "q_lss": -5.0 + 7.0 * ((index * 37) % 64) / 63.0,
        "age": -2.0 + 4.0 * ((index * 43) % 64) / 63.0,
        "log_surface_density": 6.5 + 4.5 * ((index * 47) % 64) / 63.0,
        "axis_ratio": 0.25 + 0.7 * ((index * 53) % 64) / 63.0,
    }


def _local_predictors(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "log_acceleration": np.asarray(
            [math.log10(float(config["physics"]["constants"]["one_au_acceleration_m_s2"]))]
        ),
        "log_density": np.asarray([12.0]),
        "log_potential": np.asarray([-8.0]),
        "q_lss": np.asarray([-2.0]),
        "age": np.asarray([0.0]),
        "log_surface_density": np.asarray([10.0]),
        "axis_ratio": np.asarray([1.0]),
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
    minimum_mu = np.full(count, np.nan)
    maximum_mu = np.full(count, np.nan)
    material_response = np.full(count, np.nan)
    phase_span = np.full(count, np.nan)
    local_response = np.full(count, np.nan)
    convergence = np.full(count, np.nan)
    batch = int(config["evaluation"]["candidate_batch_size"])
    gates = config["admissibility"]
    for begin in range(0, count, batch):
        end = min(begin + batch, count)
        values = _candidate_values(config, raw, begin, end, np)
        activation, convergence_batch = _candidate_activation(
            config, raw, domain, begin, end, np, return_convergence=True
        )
        mu = 1.0 + values["polarity"][:, None] * values["amplitude"][:, None] * activation
        local_activation = _candidate_activation(config, raw, local, begin, end, np)[:, 0]
        local_mu = 1.0 + values["polarity"] * values["amplitude"] * local_activation
        minimum_mu[begin:end] = np.min(mu, axis=1)
        maximum_mu[begin:end] = np.max(mu, axis=1)
        material_response[begin:end] = np.max(np.abs(mu - 1.0), axis=1)
        phase_span[begin:end] = np.max(activation, axis=1) - np.min(activation, axis=1)
        local_response[begin:end] = np.abs(local_mu - 1.0)
        convergence[begin:end] = convergence_batch
        keep[begin:end] = (
            np.all(np.isfinite(mu), axis=1)
            & (minimum_mu[begin:end] >= float(gates["minimum_mu"]))
            & (maximum_mu[begin:end] <= float(gates["maximum_mu"]))
            & (material_response[begin:end] >= float(gates["minimum_material_fractional_response"]))
            & (phase_span[begin:end] >= float(gates["minimum_phase_activation_span"]))
            & (local_response[begin:end] <= float(gates["maximum_local_fractional_response"]))
            & (convergence[begin:end] <= float(gates["landau_fixed_point_tolerance"]))
        )
    arrays = {key: value[keep] for key, value in raw.items()}
    signature_parts = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        delta = _candidate_delta_log10_sigma(config, arrays, domain, begin, end, np)
        signature_parts.append(
            np.round(delta, int(gates["behavioral_equivalence_precision_decimal_places"]))
        )
    signatures = np.concatenate(signature_parts) if signature_parts else np.empty((0, 64))
    classes = len(np.unique(signatures, axis=0))
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
        "minimum_admitted_mu": float(np.min(minimum_mu[keep])),
        "maximum_admitted_mu": float(np.max(maximum_mu[keep])),
        "minimum_admitted_material_fractional_response": float(np.min(material_response[keep])),
        "minimum_admitted_phase_activation_span": float(np.min(phase_span[keep])),
        "maximum_admitted_local_fractional_response": float(np.max(local_response[keep])),
        "maximum_admitted_landau_fixed_point_difference": float(np.max(convergence[keep])),
    }
    generator = config["candidate_generator"]
    expected_fields = (
        ("expected_raw_candidate_digest", "raw_candidate_digest"),
        ("expected_admissible_candidate_digest", "admissible_candidate_digest"),
        ("expected_admissible_candidates", "admissible_candidates"),
        (
            "expected_behavioral_equivalence_classes_adversarial",
            "behavioral_equivalence_classes_adversarial",
        ),
    )
    for expected_key, audit_key in expected_fields:
        expected = generator.get(expected_key)
        if expected not in (None, "TO_BE_MEASURED", -1) and audit[audit_key] != expected:
            raise GravityItem33Error(f"candidate invariant changed: {expected_key}")
    expected_niches = generator.get("expected_admissible_per_niche")
    if (
        expected_niches
        and all(int(value) >= 0 for value in expected_niches.values())
        and audit["admissible_per_niche"] != expected_niches
    ):
        raise GravityItem33Error("admissible niche counts changed")
    _ADMISSIBLE_CACHE[cache_key] = arrays, audit
    return arrays, audit


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item33-candidate-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "niches": config["candidate_generator"]["niches"],
            "historical_novelty_claimed": False,
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "post_response_cells": 0,
            "audit": audit,
        }
    )


def _fresh_pool(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, str]], set[str]]:
    inherited = _read_tsv(root / str(config["sources"]["inherited_predictors"]))
    if len(inherited) != int(config["sample"]["expected_inherited_predictors"]):
        raise GravityItem33Error("inherited predictor count changed")
    prior_ids: set[str] = set()
    for key in ("item30_sample_manifest", "item31_sample_manifest", "item32_sample_manifest"):
        manifest = _read_json(root / str(config["sources"][key]))
        _verify_content_hash(manifest, key)
        prior_ids.update(str(row["plateifu"]) for row in manifest["objects"])
    pool = [
        row
        for row in inherited
        if str(row["plateifu"]) not in prior_ids
        and float(row["snr_med_g"]) >= float(config["sample"]["predictor_minimum_snr_med_g"])
    ]
    if len(pool) != int(config["sample"]["expected_fresh_predictor_pool"]):
        raise GravityItem33Error("fresh predictor pool changed")
    return pool, prior_ids


def _sample_manifest(
    config: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    sample = config["sample"]
    acceleration_median = float(
        np.median([float(row["internal_acceleration_m_s2"]) for row in pool])
    )
    environment_median = float(np.median([float(row["gema_q_lss"]) for row in pool]))
    cells: dict[str, list[dict[str, Any]]] = {
        f"g{acceleration}-env{environment}": []
        for acceleration in range(2)
        for environment in range(2)
    }
    for source in pool:
        row = dict(source)
        acceleration_bin = int(float(row["internal_acceleration_m_s2"]) >= acceleration_median)
        environment_bin = int(float(row["gema_q_lss"]) >= environment_median)
        cell = f"g{acceleration_bin}-env{environment_bin}"
        row.update(
            {
                "acceleration_bin": acceleration_bin,
                "environment_bin": environment_bin,
                "sample_cell": cell,
            }
        )
        cells[cell].append(row)
    objects = []
    cell_counts = {}
    selected_count = int(sample["selected_per_acceleration_environment_cell"])
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
                    "response_read": False,
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
            "schema_version": "invariant-gravity-item33-sample-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "acceleration_median_m_s2": f"{acceleration_median:.12e}",
            "environment_median_q_lss": f"{environment_median:.12e}",
            "objects": objects,
            "selected_cell_counts": cell_counts,
            "fold_counts_exploration": {
                str(key): fold_counts[key] for key in range(int(sample["outer_folds"]))
            },
            "counts": {
                "fresh_predictor_pool": len(pool),
                "selected": len(objects),
                "exploration": roles["exploration"],
                "reserved_confirmation": roles["reserved_confirmation"],
                "response_rows_read": 0,
            },
            "claims": {
                "response_values_read": 0,
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
    pool, prior_ids = _fresh_pool(root, config)
    sample_manifest = _sample_manifest(config, pool)
    sample = config["sample"]
    expected_counts = {
        "fresh_predictor_pool": int(sample["expected_fresh_predictor_pool"]),
        "selected": int(sample["expected_selected"]),
        "exploration": int(sample["expected_exploration"]),
        "reserved_confirmation": int(sample["expected_confirmation"]),
        "response_rows_read": 0,
    }
    if sample_manifest["counts"] != expected_counts:
        raise GravityItem33Error("frozen sample counts changed")
    inherited = _read_tsv(root / str(config["sources"]["inherited_predictors"]))
    columns = [
        *list(inherited[0]),
        "acceleration_bin",
        "environment_bin",
        "sample_cell",
        "role",
        "outer_fold",
        "response_read",
    ]
    _write_tsv(paths["predictors"], sample_manifest["objects"], columns)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item33-predictor-source-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "inherited_predictor_path": config["sources"]["inherited_predictors"],
            "inherited_predictor_sha256": _sha256_file(
                root / str(config["sources"]["inherited_predictors"])
            ),
            "predecessor_sample_sha256": {
                key: _sha256_file(root / str(config["sources"][key]))
                for key in (
                    "item30_sample_manifest",
                    "item31_sample_manifest",
                    "item32_sample_manifest",
                )
            },
            "counts": {
                "inherited_predictors": len(inherited),
                "predecessor_roles_excluded": len(prior_ids),
                "fresh_predictor_pool": len(pool),
                "selected": len(sample_manifest["objects"]),
                "response_columns_read": 0,
            },
            "claims": {
                "target_blind": True,
                "confirmation_values_read": 0,
                "post_response_formula_cells": 0,
            },
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], _candidate_manifest(config))
    return paths


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item33/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not body:
        raise GravityItem33Error(f"empty source response: {url}")
    return body, headers


def _response_query(config: Mapping[str, Any], identities: Sequence[str]) -> str:
    quoted = ",".join("'" + str(value).replace("'", "''") + "'" for value in identities)
    columns = ", ".join("d." + str(value) for value in config["sources"]["response_columns"])
    return (
        f"SELECT {columns} FROM {config['sources']['dap_table']} AS d "
        f"WHERE d.daptype='{config['sources']['daptype']}' AND d.plateifu IN ({quoted}) "
        "ORDER BY d.plateifu"
    )


def _skyserver_query(config: Mapping[str, Any], query: str) -> tuple[bytes, str]:
    parameters = urllib.parse.urlencode({"cmd": query, "format": "csv"})
    url = str(config["sources"]["skyserver_endpoint"]) + "?" + parameters
    payload, _ = _download(url)
    return payload, url


def _parse_skyserver_csv(payload: bytes) -> tuple[list[dict[str, str]], list[str]]:
    lines = payload.decode("utf-8-sig", errors="strict").splitlines()
    comments = [line.strip() for line in lines if line.strip().startswith("#")]
    table_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    if not table_lines:
        raise GravityItem33Error("empty SkyServer CSV after comment filtering")
    reader = csv.DictReader(io.StringIO("\n".join(table_lines)))
    rows = [
        {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
        for row in reader
    ]
    if reader.fieldnames == ["error_message"]:
        message = rows[0]["error_message"] if rows else "unknown SkyServer error"
        raise GravityItem33Error(message)
    return rows, comments


def acquire_responses(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "Item 33 sample manifest")
    exploration = sorted(
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "exploration"
    )
    confirmations = {
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if len(exploration) != int(config["sample"]["expected_exploration"]):
        raise GravityItem33Error("Item 33 exploration role count changed before query")
    if len(confirmations) != int(config["sample"]["expected_confirmation"]):
        raise GravityItem33Error("Item 33 confirmation role count changed before query")

    chunks: list[dict[str, Any]] = []
    all_rows: list[dict[str, str]] = []
    observed_comments: set[str] = set()
    chunk_size = int(config["sources"]["response_chunk_size"])
    expected_columns = tuple(str(value) for value in config["sources"]["response_columns"])
    for begin in range(0, len(exploration), chunk_size):
        identities = exploration[begin : begin + chunk_size]
        query = _response_query(config, identities)
        payload, url = _skyserver_query(config, query)
        rows, comments = _parse_skyserver_csv(payload)
        observed_comments.update(comments)
        if rows and tuple(rows[0].keys()) != expected_columns:
            raise GravityItem33Error("MaNGA response schema changed")
        returned = {row["plateifu"] for row in rows}
        if returned & confirmations:
            raise GravityItem33Error("confirmation response entered Item 33 acquisition")
        if not returned <= set(identities):
            raise GravityItem33Error("unrequested MaNGA response entered Item 33")
        all_rows.extend(rows)
        chunks.append(
            {
                "begin": begin,
                "requested": len(identities),
                "returned": len(rows),
                "comment_lines": comments,
                "query_sha256": _sha256_bytes(query.encode()),
                "payload_sha256": _sha256_bytes(payload),
                "url_sha256": _sha256_bytes(url.encode()),
            }
        )
    if len({row["plateifu"] for row in all_rows}) != len(all_rows):
        raise GravityItem33Error("duplicate MaNGA response row")
    all_rows.sort(key=lambda row: row["plateifu"])
    _write_tsv(paths["exploration_responses"], all_rows, expected_columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item33-response-source-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "endpoint": config["sources"]["skyserver_endpoint"],
            "daptype": config["sources"]["daptype"],
            "response_columns": list(expected_columns),
            "observed_comment_lines": sorted(observed_comments),
            "counts": {
                "exploration_identities_requested": len(exploration),
                "response_rows_returned": len(all_rows),
                "confirmation_identities_requested": 0,
                "confirmation_values_read": 0,
                "paid_api_calls": 0,
            },
            "chunks": chunks,
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["response_source_manifest"]


def _finite(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise GravityItem33Error(f"invalid response value {key}") from error
    if not np.isfinite(value):
        raise GravityItem33Error(f"nonfinite response value {key}")
    return value


def _load_response_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictor_manifest = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    response_manifest = _read_json(paths["response_source_manifest"])
    for value, label in (
        (predictor_manifest, "predictor manifest"),
        (sample, "sample manifest"),
        (candidates, "candidate manifest"),
        (response_manifest, "response manifest"),
    ):
        _verify_content_hash(value, label)
    if _sha256_file(paths["exploration_responses"]) != response_manifest["response_file"]["sha256"]:
        raise GravityItem33Error("Item 33 response file changed")
    if int(response_manifest["counts"]["confirmation_values_read"]) != 0:
        raise GravityItem33Error("Item 33 response manifest opened confirmations")

    sample_rows = {
        str(row["plateifu"]): row for row in sample["objects"] if row["role"] == "exploration"
    }
    response_rows = {row["plateifu"]: row for row in _read_tsv(paths["exploration_responses"])}
    quality = config["quality"]
    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for plateifu, predictor in sorted(sample_rows.items()):
        response = response_rows.get(plateifu)
        reasons = []
        if response is None:
            failures.append({"plateifu": plateifu, "reasons": ["missing_response_row"]})
            continue
        try:
            sigma = _finite(response, "stellar_sigma_1re")
            rchi2 = _finite(response, "stellar_rchi2_1re")
            velocity_low = _finite(response, "stellar_vel_lo_clip")
            velocity_high = _finite(response, "stellar_vel_hi_clip")
        except GravityItem33Error:
            failures.append({"plateifu": plateifu, "reasons": ["incomplete_response"]})
            continue
        span = velocity_high - velocity_low
        if float(predictor["snr_med_g"]) < float(quality["minimum_snr_med_g"]):
            reasons.append("low_predictor_snr")
        if rchi2 > float(quality["maximum_stellar_rchi2_1re"]):
            reasons.append("stellar_rchi2")
        if not (
            float(quality["minimum_stellar_sigma_km_s"])
            <= sigma
            <= float(quality["maximum_stellar_sigma_km_s"])
        ):
            reasons.append("stellar_sigma")
        if not (
            float(quality["minimum_stellar_velocity_span_km_s"])
            <= span
            <= float(quality["maximum_stellar_velocity_span_km_s"])
        ):
            reasons.append("stellar_velocity_span")
        if reasons:
            failures.append({"plateifu": plateifu, "reasons": reasons})
            continue
        row = dict(predictor)
        row.update(
            {
                "stellar_sigma_1re_km_s": sigma,
                "stellar_rchi2_1re": rchi2,
                "stellar_velocity_span_km_s": span,
                "y_log10_sigma": math.log10(sigma),
            }
        )
        valid.append(row)
    extraction = _content_hashed(
        {
            "schema_version": "invariant-gravity-item33-extraction-1.0",
            "exploration_roles": len(sample_rows),
            "response_rows": len(response_rows),
            "quality_passing": len(valid),
            "quality_failures": failures,
            "failure_reason_counts": dict(
                sorted(Counter(reason for row in failures for reason in row["reasons"]).items())
            ),
            "confirmation_values_read": 0,
            "failed_identity_replacement": False,
        }
    )
    return valid, response_manifest, extraction


def _build_candidate_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    predictors = {key: xp.asarray(value) for key, value in _phase_predictors(rows).items()}
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(_candidate_delta_log10_sigma(config, arrays, predictors, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _critical_feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    predictors = _phase_predictors(rows)
    return np.column_stack(
        [
            predictors["log_acceleration"],
            predictors["log_density"],
            predictors["log_potential"],
            predictors["q_lss"],
            predictors["log_surface_density"],
            predictors["age"],
            predictors["axis_ratio"],
        ]
    )


def _ordinary_change_point_design(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, list[float]]]:
    base_features = _feature_matrix(rows, config)
    critical = _critical_feature_matrix(rows)
    quantiles = np.asarray(config["evaluation"]["change_point_quantiles"], dtype=np.float64)
    labels = (
        "log_acceleration",
        "log_density",
        "log_potential",
        "q_lss",
        "log_surface_density",
        "age",
        "axis_ratio",
    )
    hinges = []
    thresholds: dict[str, list[float]] = {}
    for column, label in enumerate(labels):
        values = critical[:, column]
        knots = np.quantile(values, quantiles)
        thresholds[label] = [float(value) for value in knots]
        scale = max(float(np.std(values)), 1e-12)
        for knot in knots:
            hinges.append(np.maximum(values - knot, 0.0) / scale)
            hinges.append(np.maximum(knot - values, 0.0) / scale)
    ordinary = _design_matrix(base_features, flexible=True)
    return np.column_stack([ordinary, *hinges]), thresholds


def _all_baseline_predictions(
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    predictions = _baseline_predictions(target, base, folds, rows, config)
    change_design, thresholds = _ordinary_change_point_design(rows, config)
    predictions["ordinary_change_point"] = _ridge_oof(
        target,
        folds,
        change_design,
        float(config["evaluation"]["ridge_alpha_change_point"]),
        int(config["sample"]["outer_folds"]),
    )
    return predictions, thresholds


def _candidate_record(
    index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(arrays["niche"][index])
    record: dict[str, Any] = {
        "admissible_index": index,
        "niche_index": niche,
        "niche": config["candidate_generator"]["niches"][niche]["id"],
        "creativity_label": config["candidate_generator"]["niches"][niche]["creativity_label"],
        "polarity": float(values["polarity"][0]),
        "amplitude": float(values["amplitude"][0]),
        "threshold_index": int(values["threshold_index"][0]),
        "transition_width": float(values["width"][0]),
        "critical_exponent": float(values["critical"][0]),
        "secondary_coupling": float(values["secondary"][0]),
        "latent_strength": float(values["latent"][0]),
        "phase_side": float(values["side"][0]),
    }
    if niche == 0:
        record["log10_acceleration_threshold_m_s2"] = float(values["acceleration_threshold"][0])
    elif niche == 1:
        record["log10_density_threshold_msun_kpc3"] = float(values["density_threshold"][0])
    elif niche == 2:
        record["environment_q_lss_threshold"] = float(values["environment_threshold"][0])
    else:
        record["organization_threshold"] = float(values["organization_threshold"][0])
    return record


def _synthetic_controls(
    delta: Any,
    base: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    xp: Any,
) -> dict[str, Any]:
    injection_results = []
    for niche in range(4):
        indices = np.where(arrays["niche"] == niche)[0]
        if not len(indices):
            raise GravityItem33Error(f"missing synthetic injection niche {niche}")
        niche_values = delta[xp.asarray(indices)]
        variance = xp.var(niche_values, axis=1)
        index = int(indices[int(_to_numpy(xp.argmax(variance), xp))])
        target = base + _to_numpy(delta[index], xp)
        selected = _screen_candidate_matrix(delta, target, base, folds, config, xp)
        selected_niches = [int(arrays["niche"][value]) for value in selected["selected_indices"]]
        injection_results.append(
            {
                "injection_index": index,
                "injection_niche": niche,
                "selected_niches": selected_niches,
                "exact_niche_recovered_all_folds": all(value == niche for value in selected_niches),
                "candidate_mse": _mse(target, selected["prediction"]),
            }
        )
    gr_target = base.copy()
    gr_candidate = _screen_candidate_matrix(delta, gr_target, base, folds, config, xp)
    gr_baseline = _virial_oof(gr_target, base, folds, config)
    candidate_mse = _mse(gr_target, gr_candidate["prediction"])
    baseline_mse = _mse(gr_target, gr_baseline)
    return {
        "injections": injection_results,
        "all_injected_niches_recovered": all(
            row["exact_niche_recovered_all_folds"] for row in injection_results
        ),
        "GR_candidate_mse": candidate_mse,
        "GR_baseline_mse": baseline_mse,
        "GR_control_candidate_improves": candidate_mse < baseline_mse - 1e-16,
    }


def _cell_residual_permutation(
    target: np.ndarray,
    reference: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    random: np.random.Generator,
) -> np.ndarray:
    residual = target - reference
    shuffled = residual.copy()
    cells = np.asarray([str(row["sample_cell"]) for row in rows])
    for cell in sorted(set(cells.tolist())):
        indices = np.where(cells == cell)[0]
        shuffled[indices] = residual[random.permutation(indices)]
    return reference + shuffled


def _evaluate(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) < int(config["sample"]["outer_folds"]):
        raise GravityItem33Error("too few Item 33 response-complete galaxies")
    arrays, candidate_audit = _admissible_candidates(config)
    target = np.asarray([float(row["y_log10_sigma"]) for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
    expected_folds = set(range(int(config["sample"]["outer_folds"])))
    if set(folds.tolist()) != expected_folds:
        raise GravityItem33Error("Item 33 response-complete folds are incomplete")
    base = _baryonic_virial_prediction(rows, config)
    xp, backend, device = _backend()
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    delta = _build_candidate_matrix(config, arrays, rows, xp)
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start

    crosscheck = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["niche"]))
    cpu_delta = _candidate_delta_log10_sigma(
        config, arrays, _phase_predictors(rows), 0, crosscheck, np
    )
    gpu_delta = _to_numpy(delta[:crosscheck], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu_delta - gpu_delta)))

    observed = _screen_candidate_matrix(delta, target, base, folds, config, xp)
    baselines, change_thresholds = _all_baseline_predictions(target, base, folds, rows, config)
    candidate_mse = _mse(target, observed["prediction"])
    baseline_mse = {key: _mse(target, value) for key, value in baselines.items()}
    observed_statistic = _improvement(baseline_mse["ordinary_change_point"], candidate_mse)

    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    trials = int(config["evaluation"]["permutation_trials"])
    null_improvements = []
    for _ in range(trials):
        null_target = _cell_residual_permutation(
            target, baselines["ordinary_change_point"], rows, random
        )
        null_selected = _screen_candidate_matrix(delta, null_target, base, folds, config, xp)
        null_baselines, _ = _all_baseline_predictions(null_target, base, folds, rows, config)
        null_improvements.append(
            _improvement(
                _mse(null_target, null_baselines["ordinary_change_point"]),
                _mse(null_target, null_selected["prediction"]),
            )
        )
    p_value = (1.0 + sum(value >= observed_statistic for value in null_improvements)) / (
        trials + 1.0
    )
    controls = _synthetic_controls(delta, base, folds, config, arrays, xp)
    selected_records = [
        _candidate_record(index, config, arrays) for index in observed["selected_indices"]
    ]
    selected_niches = [int(arrays["niche"][index]) for index in observed["selected_indices"]]
    niche_counts = Counter(selected_niches)

    mass = np.asarray([float(row["stellar_mass_msun"]) for row in rows])
    acceleration = np.asarray([float(row["internal_acceleration_m_s2"]) for row in rows])
    density = np.asarray([float(row["log_mean_stellar_density"]) for row in rows])
    q_lss = np.asarray([float(row["gema_q_lss"]) for row in rows])
    age = np.asarray([float(row["dn4000"]) for row in rows])
    slice_masks = {
        "low_mass": mass <= np.median(mass),
        "high_mass": mass > np.median(mass),
        "low_acceleration": acceleration <= np.median(acceleration),
        "high_acceleration": acceleration > np.median(acceleration),
        "low_density": density <= np.median(density),
        "high_density": density > np.median(density),
        "low_external_tide": q_lss <= np.median(q_lss),
        "high_external_tide": q_lss > np.median(q_lss),
        "younger_spectral_clock": age <= np.median(age),
        "older_spectral_clock": age > np.median(age),
    }
    slices: dict[str, Any] = {}
    for label, mask in slice_masks.items():
        indices = np.where(mask)[0]
        value: dict[str, Any] = {
            "objects": len(indices),
            "candidate_mse": _mse(target, observed["prediction"], indices),
        }
        for baseline_name, prediction in baselines.items():
            mse = _mse(target, prediction, indices)
            value[f"{baseline_name}_mse"] = mse
            value[f"improvement_vs_{baseline_name}"] = _improvement(mse, value["candidate_mse"])
        slices[label] = value
    object_counterexamples = int(
        np.count_nonzero(
            (target - observed["prediction"]) ** 2
            > (target - baselines["ordinary_change_point"]) ** 2
        )
    )
    required = int(config["sample"]["minimum_complete_exploration_objects"])
    fraction = len(rows) / int(config["sample"]["expected_exploration"])
    quality_pass = len(rows) >= required and fraction >= float(
        config["sample"]["minimum_quality_retention_fraction"]
    )
    gates = config["gates"]
    stable_niche = max(niche_counts.values()) >= int(gates["minimum_same_niche_folds"])
    controls_pass = (
        bool(controls["all_injected_niches_recovered"])
        and not bool(controls["GR_control_candidate_improves"])
        and cpu_gpu_max <= 1e-11
    )
    universal_gates = {
        "response_quality": quality_pass,
        "confirmation_values_read_zero": int(gates["confirmation_values_read"]) == 0,
        "post_response_cells_zero": int(gates["post_response_candidate_cells"]) == 0,
        "improvement_vs_baryonic_virial": _improvement(
            baseline_mse["baryonic_virial"], candidate_mse
        )
        >= float(gates["minimum_improvement_vs_baryonic_virial"]),
        "improvement_vs_structural": _improvement(baseline_mse["structural_ridge"], candidate_mse)
        >= float(gates["minimum_improvement_vs_structural"]),
        "improvement_vs_flexible": _improvement(baseline_mse["flexible_nuisance"], candidate_mse)
        >= float(gates["minimum_improvement_vs_flexible"]),
        "improvement_vs_change_point": observed_statistic
        >= float(gates["minimum_improvement_vs_change_point"]),
        "each_broad_half_improves_baryonic_virial": all(
            value["improvement_vs_baryonic_virial"]
            >= float(gates["minimum_each_broad_half_improvement_vs_baryonic_virial"])
            for value in slices.values()
        ),
        "selection_aware_permutation": p_value
        <= float(gates["maximum_selection_aware_permutation_p"]),
        "stable_niche": stable_niche,
        "all_injected_niches_recovered": bool(controls["all_injected_niches_recovered"]),
        "known_GR_control": not bool(controls["GR_control_candidate_improves"]),
        "cpu_gpu_agreement": cpu_gpu_max <= 1e-11,
        "local_limit": candidate_audit["maximum_admitted_local_fractional_response"]
        <= float(config["admissibility"]["maximum_local_fractional_response"]),
        "landau_fixed_point": candidate_audit["maximum_admitted_landau_fixed_point_difference"]
        <= float(config["admissibility"]["landau_fixed_point_tolerance"]),
    }
    phenomenon_gates = {
        "response_quality": quality_pass,
        "improvement_vs_change_point": observed_statistic
        >= float(gates["phenomenon_minimum_improvement_vs_change_point"]),
        "selection_aware_permutation": p_value
        <= float(gates["phenomenon_maximum_selection_aware_p"]),
        "stable_niche": stable_niche,
        "controls": controls_pass,
    }
    partial_slices = [
        label
        for label, value in slices.items()
        if value["improvement_vs_ordinary_change_point"]
        >= float(gates["partial_minimum_slice_improvement_vs_change_point"])
    ]
    universal_pass = all(universal_gates.values())
    phenomenon_pass = all(phenomenon_gates.values())
    if not quality_pass:
        decision = "INCONCLUSIVE_ITEM33_QUALITY"
    elif universal_pass:
        decision = "PASS_ITEM33_EXPLORATION_UNIVERSAL"
    elif phenomenon_pass:
        decision = "PASS_ITEM33_EXPLORATION_PHENOMENON_LEAD_PENDING_REPLICATION"
    elif partial_slices:
        decision = "BOTH_FORMAL_TRACKS_NOT_PROMOTED_SCOPED_PARTIAL_PATTERN_RETAINED"
    else:
        decision = "SCOPED_ITEM33_REJECT"

    scientific = {
        "decision": decision,
        "quality": {
            "complete_exploration_objects": len(rows),
            "minimum_required": required,
            "retention_fraction": fraction,
            "pass": quality_pass,
        },
        "universal_gravity_track": {
            "decision": "PASS_EXPLORATION" if universal_pass else "NOT_PROMOTED",
            "gates": universal_gates,
        },
        "phenomenon_publication_track": {
            "decision": "PASS_EXPLORATION" if phenomenon_pass else "NOT_PROMOTED",
            "gates": phenomenon_gates,
            "paper_claim_authorized": False,
            "unchanged_fresh_replication_required": True,
        },
        "partial_track": {"retained_slices": partial_slices, "paper_claim_authorized": False},
        "metrics": {
            "candidate_mse": candidate_mse,
            "baseline_mse": baseline_mse,
            "improvement_vs_baryonic_virial": _improvement(
                baseline_mse["baryonic_virial"], candidate_mse
            ),
            "improvement_vs_structural": _improvement(
                baseline_mse["structural_ridge"], candidate_mse
            ),
            "improvement_vs_flexible": _improvement(
                baseline_mse["flexible_nuisance"], candidate_mse
            ),
            "improvement_vs_change_point": observed_statistic,
            "selection_aware_permutation_p": p_value,
            "maximum_null_improvement": max(null_improvements),
            "object_counterexamples_vs_change_point": object_counterexamples,
        },
        "permutation": {
            "strategy": config["evaluation"]["permutation_strategy"],
            "trials": trials,
            "null_improvements": null_improvements,
        },
        "ordinary_change_point_thresholds": change_thresholds,
        "broad_slices": slices,
        "selected_candidates": selected_records,
        "selected_niche_counts": {str(key): niche_counts[key] for key in range(4)},
        "controls": controls,
        "candidate_audit": candidate_audit,
        "failure_space": {
            "raw_cells": candidate_audit["raw_candidates"],
            "inadmissible_cells": candidate_audit["raw_candidates"]
            - candidate_audit["admissible_candidates"],
            "admissible_cells": candidate_audit["admissible_candidates"],
            "behavioral_equivalence_classes_adversarial": candidate_audit[
                "behavioral_equivalence_classes_adversarial"
            ],
            "object_counterexamples_vs_change_point": object_counterexamples,
            "negative_or_partial_families_are_retained": True,
        },
    }
    training_per_search = int(
        len(arrays["niche"])
        * sum(
            np.count_nonzero(folds != fold) for fold in range(int(config["sample"]["outer_folds"]))
        )
    )
    compute = {
        "backend": backend,
        "device": device,
        "candidate_matrix_seconds": matrix_seconds,
        "candidate_cells": len(arrays["niche"]),
        "candidate_observable_matrix_values": int(np.prod(delta.shape)),
        "candidate_training_residual_evaluations_observed": training_per_search,
        "candidate_training_residual_evaluations_with_nulls": training_per_search * (trials + 1),
        "cpu_crosscheck_candidates": crosscheck,
        "cpu_gpu_max_abs_difference": cpu_gpu_max,
        "permutation_trials": trials,
        "paid_api_calls": 0,
    }
    return scientific, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    response_manifest: Mapping[str, Any],
    extraction: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    predictor = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    test_path = root / "tests/test_gravity_item33_phase_transition.py"
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item33-phase-transition-result-1.0",
            "item": 33,
            "title": config["title"],
            "decision": scientific["decision"],
            "hypothesis": config["hypothesis"],
            "scientific": scientific,
            "compute": compute,
            "extraction": extraction,
            "theory": {
                "sources": config["sources"]["theory_sources"],
                "families": config["candidate_generator"]["niches"],
                "field_projection": config["physics"]["field_projection"],
                "dynamical_baseline": config["physics"]["dynamical_baseline"],
                "stability_scope": config["physics"]["stability_scope"],
            },
            "frozen_boundary": {
                "stable_goal_sha256": config["stable_goal_sha256"],
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "predictor_manifest_content_sha256": predictor["content_sha256"],
                "sample_manifest_content_sha256": sample["content_sha256"],
                "candidate_manifest_content_sha256": candidates["content_sha256"],
                "response_manifest_content_sha256": response_manifest["content_sha256"],
                "response_file_sha256": response_manifest["response_file"]["sha256"],
                "complete_response_objects": len(rows),
                "confirmation_response_values_read": 0,
                "post_response_formula_generation": False,
                "paid_api_calls": 0,
            },
            "source_bindings": {
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "test_sha256": _sha256_file(test_path) if test_path.exists() else None,
            },
            "claim_boundary": [
                config["scope"]["claim_ceiling"],
                "This cross-sectional test cannot observe a temporal phase transition, causal hysteresis, metastability, or spontaneous scalarization.",
                "The Landau fixed point is an observable proxy, not an action-derived order parameter, covariant field solution, or historical novelty claim.",
                "The acceleration and density branches overlap known MOND-like, symmetron, chameleon, and screening transition families.",
                "The matched ordinary change-point ridge is the primary phenomenon control; beating weaker smooth baselines alone is not phase-transition evidence.",
                "Stellar mass-to-light ratios, orbital anisotropy, stellar populations, Sersic virial projection, and environment catalogs can produce ordinary critical-looking correlations.",
                "This integrated stellar-dispersion observable cannot establish resolved rotation curves, gravitational slip, direct lensing, clusters, cosmology, stability, or an alternative to GR.",
                "A positive result is exploration evidence only; confirmations stay sealed and an unchanged fresh replication is mandatory for a paper claim.",
            ],
        }
    )


def run_experiment(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest, extraction = _load_response_rows(root, config)
    scientific, compute = _evaluate(config, rows)
    paths = _source_paths(root, config)
    compute_manifest = _content_hashed(
        {"schema_version": "invariant-gravity-item33-compute-1.0", **compute}
    )
    _write_json(paths["compute_manifest"], compute_manifest)
    receipt = _build_receipt(root, config, rows, response_manifest, extraction, scientific, compute)
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, receipt)
    return result_path


def validate_checked(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    for key in (
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "response_source_manifest",
        "compute_manifest",
    ):
        _verify_content_hash(_read_json(paths[key]), key)
    predictor = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    response = _read_json(paths["response_source_manifest"])
    if int(predictor["counts"]["response_columns_read"]) != 0:
        raise GravityItem33Error("Item 33 predictor freeze contains response values")
    if int(sample["counts"]["reserved_confirmation"]) != int(
        config["sample"]["expected_confirmation"]
    ):
        raise GravityItem33Error("Item 33 confirmation allocation changed")
    if int(sample["counts"]["exploration"]) != int(config["sample"]["expected_exploration"]):
        raise GravityItem33Error("Item 33 exploration allocation changed")
    if int(candidates["post_response_cells"]) != 0:
        raise GravityItem33Error("Item 33 candidate manifest contains post-response cells")
    if int(response["counts"]["confirmation_values_read"]) != 0:
        raise GravityItem33Error("Item 33 response acquisition opened confirmations")
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "Item 33 result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem33Error("checked Item 33 result opened confirmation responses")
    if bool(result["frozen_boundary"]["post_response_formula_generation"]):
        raise GravityItem33Error("checked Item 33 result contains post-response generation")
    if (
        _sha256_file(paths["exploration_responses"])
        != result["frozen_boundary"]["response_file_sha256"]
    ):
        raise GravityItem33Error("checked Item 33 response file changed")
    if result["source_bindings"]["config_sha256"] != _sha256_file(root / CONFIG_PATH):
        raise GravityItem33Error("checked Item 33 config changed")
    if result["source_bindings"]["module_sha256"] != _sha256_file(root / MODULE_PATH):
        raise GravityItem33Error("checked Item 33 module changed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-predictors")
    sub.add_parser("acquire-responses")
    sub.add_parser("run")
    sub.add_parser("validate-checked")
    sub.add_parser("show-candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare-predictors":
        print(prepare_predictors(root)["sample_manifest"].as_posix())
    elif args.command == "acquire-responses":
        print(acquire_responses(root).as_posix())
    elif args.command == "run":
        print(run_experiment(root).as_posix())
    elif args.command == "validate-checked":
        validate_checked(root)
        print("PASS")
    else:
        print(json.dumps(_candidate_manifest(load_config(root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
