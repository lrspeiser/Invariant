"""Frozen Item 31 vacuum-polarization and gravitational-permittivity experiment.

The experiment inherits only response-blind, predecessor-vetoed MaNGA/GEMA
predictors from Item 30.  Every Item 30 exploration and confirmation identity is
excluded before Item 31 roles are assigned.  Stellar-dispersion responses are
queried only for committed exploration roles after both freezes are bound.
"""

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
    _screen_candidate_matrix,
    _virial_oof,
)
from sigma_theory_compiler.gravity_item30_screening_mechanisms import (
    generate_raw_candidates as _generate_mixed_radix_candidates,
)

CONFIG_PATH = Path("configs/gravity_item31_vacuum_permittivity_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item31_vacuum_permittivity.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
_ADMISSIBLE_CACHE: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


class GravityItem31Error(RuntimeError):
    """Raised when an Item 31 freeze, leakage, or replay invariant is violated."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item31-vacuum-permittivity-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 31:
        raise GravityItem31Error("unexpected Item 31 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem31Error("stable gravity goal changed")
    if int(config["candidate_generator"]["raw_candidate_cells"]) != 262144:
        raise GravityItem31Error("raw candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem31Error("post-response candidates entered Item 31")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem31Error("equal-viability policy changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem31Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem31Error("paid calls are outside Item 31")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem31Error(f"scientific dependency changed: {relative}")
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
        raise GravityItem31Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem31Error("could not read frozen Item 31 module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem31Error("Item 31 module differs from scientific freeze")


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in (
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem31Error(f"{key} differs from sample freeze")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return _generate_mixed_radix_candidates(config)


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    index = {key: arrays[key][begin:end] for key in arrays}
    return {
        "niche": xp.asarray(index["niche"]),
        "polarity": xp.asarray(np.asarray(generator["polarities"])[index["polarity"]]),
        "amplitude": xp.asarray(np.asarray(generator["amplitudes"])[index["amplitude"]]),
        "threshold_index": xp.asarray(index["threshold"]),
        "sharpness": xp.asarray(np.asarray(generator["sharpness"])[index["sharpness"]]),
        "environment": xp.asarray(
            np.asarray(generator["environment_couplings"])[index["environment"]]
        ),
        "shape": xp.asarray(np.asarray(generator["shape_powers"])[index["shape"]]),
        "scale": xp.asarray(np.asarray(generator["scale_factors"])[index["scale"]]),
        "acceleration_threshold": xp.asarray(
            np.asarray(generator["log10_acceleration_thresholds_m_s2"])[index["threshold"]]
        ),
        "density_threshold": xp.asarray(
            np.asarray(generator["log10_density_thresholds_msun_kpc3"])[index["threshold"]]
        ),
    }


def _candidate_activation(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    predictors: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    values = _candidate_values(config, arrays, begin, end, xp)
    column = (-1, 1)
    niche = values["niche"].reshape(column)
    sharp = values["sharpness"].reshape(column)
    shape = values["shape"].reshape(column)
    scale = values["scale"].reshape(column)
    coupling = values["environment"].reshape(column)

    acceleration = xp.maximum(xp.asarray(predictors["acceleration_m_s2"])[None, :], 1e-300)
    density = xp.maximum(xp.asarray(predictors["density_msun_kpc3"])[None, :], 1e-300)
    q_lss = xp.clip(xp.asarray(predictors["q_lss"])[None, :], -8.0, 2.0)
    eta_k = xp.clip(xp.asarray(predictors["eta_k"])[None, :], -3.0, 4.0)
    dnn = xp.maximum(xp.asarray(predictors["dnn_mpc"])[None, :], 1e-6)

    tide = xp.power(10.0, q_lss)
    crowding = xp.power(10.0, eta_k - 1.0)
    density_effective = density * (1.0 + coupling * (tide + crowding))
    acceleration_effective = acceleration * (1.0 + coupling * tide)

    density_c = xp.power(10.0, values["density_threshold"].reshape(column)) * scale
    acceleration_c = xp.power(10.0, values["acceleration_threshold"].reshape(column)) * scale
    low_density = 1.0 / (1.0 + xp.power(density_effective / density_c, sharp))
    low_acceleration = 1.0 / (1.0 + xp.power(acceleration_effective / acceleration_c, sharp))
    density_permittivity = xp.power(low_density, shape)
    acceleration_polarization = xp.power(low_acceleration, shape)

    environment_weight = coupling / (1.0 + coupling)
    separation_isolation = dnn / (dnn + scale)
    field_isolation = 1.0 / (1.0 + tide + crowding)
    constitutive_kernel = xp.sqrt(xp.clip(separation_isolation * field_isolation, 0.0, 1.0))
    environment_factor = (1.0 - environment_weight) + environment_weight * constitutive_kernel
    nonlocal_polarization = xp.power(low_acceleration, shape) * xp.power(environment_factor, sharp)

    frustrated_density_c = xp.power(10.0, values["density_threshold"].reshape(column)) / xp.maximum(
        scale, 1e-300
    )
    frustrated_low_density = 1.0 / (1.0 + xp.power(density / frustrated_density_c, sharp))
    frustrated = xp.power(xp.abs(low_acceleration - frustrated_low_density), shape)
    frustrated *= xp.power(environment_factor, sharp)

    return xp.where(
        niche == 0,
        density_permittivity,
        xp.where(
            niche == 1,
            acceleration_polarization,
            xp.where(niche == 2, nonlocal_polarization, frustrated),
        ),
    )


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


def _adversarial_predictors(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    count = int(config["admissibility"]["adversarial_points"])
    log_mass = np.repeat(np.linspace(8.0, 12.0, 8), 8)
    log_radius = np.tile(np.linspace(-0.5, 1.5, 8), 8)
    if len(log_mass) != count:
        raise GravityItem31Error("adversarial point count changed")
    constants = config["physics"]["constants"]
    mass = 10.0**log_mass
    radius = 10.0**log_radius
    acceleration = (
        float(constants["G_kpc_km2_s2_Msun"])
        * mass
        / radius**2
        * 1000.0**2
        / float(constants["kpc_to_m"])
    )
    density = mass / (4.0 * math.pi * radius**3 / 3.0)
    return {
        "acceleration_m_s2": acceleration,
        "density_msun_kpc3": density,
        "q_lss": np.tile(np.asarray([-8.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]), 8),
        "eta_k": np.repeat(np.linspace(-2.0, 3.0, 8), 8),
        "dnn_mpc": np.tile(np.geomspace(0.01, 3.0, 8), 8),
    }


def _local_predictors(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    local = config["physics"]["local_reference"]
    return {
        "acceleration_m_s2": np.asarray([float(local["acceleration_m_s2"])]),
        "density_msun_kpc3": np.asarray([float(local["mean_density_msun_kpc3"])]),
        "q_lss": np.asarray([float(local["q_lss"])]),
        "eta_k": np.asarray([float(local["eta_k"])]),
        "dnn_mpc": np.asarray([float(local["dnn_mpc"])]),
    }


def _probe(
    acceleration: float, density: float, q_lss: float = -8.0, eta_k: float = -2.0
) -> dict[str, np.ndarray]:
    return {
        "acceleration_m_s2": np.asarray([acceleration]),
        "density_msun_kpc3": np.asarray([density]),
        "q_lss": np.asarray([q_lss]),
        "eta_k": np.asarray([eta_k]),
        "dnn_mpc": np.asarray([3.0]),
    }


def _admissible_candidates(
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_key = _sha256_bytes(
        _canonical_bytes(
            {
                "candidate_generator": config["candidate_generator"],
                "admissibility": config["admissibility"],
                "local_reference": config["physics"]["local_reference"],
            }
        )
    )
    if cache_key in _ADMISSIBLE_CACHE:
        return _ADMISSIBLE_CACHE[cache_key]
    raw = generate_raw_candidates(config)
    domain = _adversarial_predictors(config)
    local = _local_predictors(config)
    low = _probe(1e-13, 1e4)
    high = _probe(1e-5, 1e12, q_lss=2.0, eta_k=3.0)
    mismatch_low_g = _probe(1e-13, 1e12)
    mismatch_low_rho = _probe(1e-5, 1e4)

    keep = np.zeros(len(raw["niche"]), dtype=bool)
    local_fraction = np.full(len(keep), np.nan)
    minimum_mu = np.full(len(keep), np.nan)
    maximum_mu = np.full(len(keep), np.nan)
    direction_contrast = np.full(len(keep), np.nan)
    behavioral_digests: set[bytes] = set()
    gates = config["admissibility"]
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(keep), batch):
        end = min(begin + batch, len(keep))
        values = _candidate_values(config, raw, begin, end, np)
        activation = _candidate_activation(config, raw, domain, begin, end, np)
        mu = 1.0 + values["polarity"][:, None] * values["amplitude"][:, None] * activation
        finite = np.all(np.isfinite(mu), axis=1)
        minimum_mu[begin:end] = np.nanmin(mu, axis=1)
        maximum_mu[begin:end] = np.nanmax(mu, axis=1)
        bounded = (minimum_mu[begin:end] >= float(gates["minimum_mu"])) & (
            maximum_mu[begin:end] <= float(gates["maximum_mu"])
        )
        material = np.max(np.abs(mu - 1.0), axis=1) >= float(gates["minimum_effective_response"])
        local_activation = _candidate_activation(config, raw, local, begin, end, np)[:, 0]
        local_mu = 1.0 + values["polarity"] * values["amplitude"] * local_activation
        local_fraction[begin:end] = np.abs(local_mu - 1.0)
        local_pass = local_fraction[begin:end] <= float(gates["maximum_local_fractional_response"])

        low_activation = _candidate_activation(config, raw, low, begin, end, np)[:, 0]
        high_activation = _candidate_activation(config, raw, high, begin, end, np)[:, 0]
        mismatch_a = _candidate_activation(config, raw, mismatch_low_g, begin, end, np)[:, 0]
        mismatch_b = _candidate_activation(config, raw, mismatch_low_rho, begin, end, np)[:, 0]
        known_contrast = low_activation - high_activation
        mismatch_contrast = np.maximum(mismatch_a, mismatch_b) - np.maximum(
            low_activation, high_activation
        )
        is_frustrated = raw["niche"][begin:end] == 3
        direction_contrast[begin:end] = np.where(is_frustrated, mismatch_contrast, known_contrast)
        direction = np.where(
            is_frustrated,
            mismatch_contrast >= float(gates["minimum_frustrated_mismatch_contrast"]),
            known_contrast >= float(gates["minimum_known_family_low_state_contrast"]),
        )
        batch_keep = finite & bounded & material & local_pass & direction
        keep[begin:end] = batch_keep
        rounded_behavior = np.round(
            np.column_stack([mu[batch_keep], local_mu[batch_keep]]), decimals=12
        )
        for row in rounded_behavior:
            behavioral_digests.add(hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest())

    arrays = {key: value[keep] for key, value in raw.items()}
    raw_counts = Counter(int(value) for value in raw["niche"])
    admitted_counts = Counter(int(value) for value in arrays["niche"])
    signature = np.column_stack([arrays[key] for key in sorted(arrays)])
    audit = {
        "raw_candidates": len(raw["niche"]),
        "raw_per_niche": {str(key): raw_counts[key] for key in range(4)},
        "admissible_candidates": len(arrays["niche"]),
        "admissible_per_niche": {str(key): admitted_counts[key] for key in range(4)},
        "raw_candidate_digest": _candidate_digest(raw),
        "admissible_candidate_digest": _candidate_digest(arrays),
        "exact_parameter_signatures": len(np.unique(signature, axis=0)),
        "behavioral_equivalence_classes_adversarial": len(behavioral_digests),
        "behavioral_duplicate_cells_adversarial": len(arrays["niche"]) - len(behavioral_digests),
        "behavioral_equivalence_precision_decimal_places": 12,
        "maximum_admitted_local_fractional_response": float(np.max(local_fraction[keep])),
        "minimum_admitted_mu": float(np.min(minimum_mu[keep])),
        "maximum_admitted_mu": float(np.max(maximum_mu[keep])),
        "minimum_admitted_direction_contrast": float(np.min(direction_contrast[keep])),
    }
    generator = config["candidate_generator"]
    expected = generator.get("expected_raw_candidate_digest")
    if expected not in (None, "TO_BE_MEASURED") and audit["raw_candidate_digest"] != expected:
        raise GravityItem31Error("raw Item 31 candidate digest changed")
    expected = generator.get("expected_admissible_candidate_digest")
    if (
        expected not in (None, "TO_BE_MEASURED")
        and audit["admissible_candidate_digest"] != expected
    ):
        raise GravityItem31Error("admissible Item 31 candidate digest changed")
    expected_count = int(generator.get("expected_admissible_candidates", -1))
    if expected_count >= 0 and audit["admissible_candidates"] != expected_count:
        raise GravityItem31Error("admissible Item 31 candidate count changed")
    expected_niches = generator.get("expected_admissible_per_niche")
    if (
        expected_niches
        and all(int(value) >= 0 for value in expected_niches.values())
        and audit["admissible_per_niche"] != expected_niches
    ):
        raise GravityItem31Error("admissible Item 31 niche counts changed")
    expected_classes = generator.get("expected_behavioral_equivalence_classes_adversarial")
    if expected_classes is not None and audit["behavioral_equivalence_classes_adversarial"] != int(
        expected_classes
    ):
        raise GravityItem31Error("Item 31 behavioral equivalence classes changed")
    expected_duplicates = generator.get("expected_behavioral_duplicate_cells_adversarial")
    if expected_duplicates is not None and audit["behavioral_duplicate_cells_adversarial"] != int(
        expected_duplicates
    ):
        raise GravityItem31Error("Item 31 behavioral duplicate count changed")
    _ADMISSIBLE_CACHE[cache_key] = (arrays, audit)
    return arrays, audit


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item31-candidate-manifest-1.0",
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "niches": config["candidate_generator"]["niches"],
            "audit": audit,
            "post_response_cells": 0,
            "response_values_read": 0,
            "historical_novelty_claimed": False,
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
        }
    )


def _sample_manifest(
    config: Mapping[str, Any], predictors: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    acceleration_median = float(
        np.median([float(row["internal_acceleration_m_s2"]) for row in predictors])
    )
    environment_median = float(np.median([float(row["gema_q_lss"]) for row in predictors]))
    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for source in predictors:
        row = dict(source)
        acceleration_bin = int(float(row["internal_acceleration_m_s2"]) > acceleration_median)
        environment_bin = int(float(row["gema_q_lss"]) > environment_median)
        row["acceleration_bin"] = acceleration_bin
        row["environment_bin"] = environment_bin
        row["sample_cell"] = f"g{acceleration_bin}-env{environment_bin}"
        cells.setdefault((acceleration_bin, environment_bin), []).append(row)

    expected_capacities = {
        str(key): int(value) for key, value in config["sample"]["expected_cell_capacities"].items()
    }
    observed_capacities = {
        f"g{cell[0]}-env{cell[1]}": len(values) for cell, values in sorted(cells.items())
    }
    if observed_capacities != expected_capacities:
        raise GravityItem31Error("Item 31 response-blind cell capacities changed")

    selected: list[dict[str, Any]] = []
    selected_cell_counts: dict[str, Any] = {}
    fold_counts = Counter()
    per_cell = int(config["sample"]["selected_per_acceleration_environment_cell"])
    confirmations = int(config["sample"]["confirmation_per_cell"])
    role_key = str(config["sample"]["role_key"])
    fold_key = str(config["sample"]["fold_key"])
    outer_folds = int(config["sample"]["outer_folds"])
    for cell in sorted(cells):
        ordered = sorted(cells[cell], key=lambda row: _hmac_rank(role_key, str(row["plateifu"])))
        if len(ordered) < per_cell:
            raise GravityItem31Error(f"Item 31 sample cell cannot fill allocation: {cell}")
        chosen = ordered[:per_cell]
        confirmation_ids = {
            str(row["plateifu"])
            for row in sorted(
                chosen,
                key=lambda row: _hmac_rank(role_key + "-confirmation", str(row["plateifu"])),
            )[:confirmations]
        }
        exploration = [row for row in chosen if str(row["plateifu"]) not in confirmation_ids]
        exploration = sorted(
            exploration, key=lambda row: _hmac_rank(fold_key, str(row["plateifu"]))
        )
        for ordinal, row in enumerate(exploration):
            row["role"] = "exploration"
            row["outer_fold"] = ordinal % outer_folds
            row["response_read"] = False
            fold_counts[row["outer_fold"]] += 1
            selected.append(row)
        for row in chosen:
            if str(row["plateifu"]) not in confirmation_ids:
                continue
            row["role"] = "reserved_confirmation"
            row["outer_fold"] = None
            row["response_read"] = False
            selected.append(row)
        label = f"g{cell[0]}-env{cell[1]}"
        selected_cell_counts[label] = {
            "eligible": len(cells[cell]),
            "selected": len(chosen),
            "exploration": len(exploration),
            "reserved_confirmation": len(confirmation_ids),
        }

    selected = sorted(selected, key=lambda row: str(row["plateifu"]))
    counts = Counter(str(row["role"]) for row in selected)
    expected = config["sample"]
    if len(selected) != int(expected["expected_selected"]):
        raise GravityItem31Error("Item 31 selected sample count changed")
    if counts["exploration"] != int(expected["expected_exploration"]):
        raise GravityItem31Error("Item 31 exploration count changed")
    if counts["reserved_confirmation"] != int(expected["expected_confirmation"]):
        raise GravityItem31Error("Item 31 confirmation count changed")
    if any(fold_counts[fold] != 32 for fold in range(outer_folds)):
        raise GravityItem31Error("Item 31 fold balance changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item31-sample-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "counts": {
                "fresh_predictor_pool": len(predictors),
                "selected": len(selected),
                "exploration": counts["exploration"],
                "reserved_confirmation": counts["reserved_confirmation"],
                "response_rows_read": 0,
            },
            "acceleration_median_m_s2": f"{acceleration_median:.12e}",
            "environment_median": f"{environment_median:.12e}",
            "eligible_cell_capacities": observed_capacities,
            "selected_cell_counts": selected_cell_counts,
            "fold_counts_exploration": {str(key): fold_counts[key] for key in range(outer_folds)},
            "objects": selected,
            "claims": {
                "response_values_read": 0,
                "confirmation_values_read": 0,
                "item30_roles_excluded": int(config["independence"]["item30_roles_excluded"]),
                "object_identity_used_as_numeric_feature": False,
                "failed_identity_replacement": False,
            },
        }
    )


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    inherited_predictor_path = root / str(config["sources"]["inherited_predictors"])
    inherited_manifest_path = root / str(config["sources"]["inherited_predictor_manifest"])
    predecessor_sample_path = root / str(config["sources"]["item30_sample_manifest"])
    inherited_manifest = _read_json(inherited_manifest_path)
    predecessor_sample = _read_json(predecessor_sample_path)
    _verify_content_hash(inherited_manifest, "Item 30 inherited predictor manifest")
    _verify_content_hash(predecessor_sample, "Item 30 predecessor sample")
    if int(inherited_manifest["counts"]["response_columns_requested"]) != 0:
        raise GravityItem31Error("response column entered inherited predictors")

    inherited = _read_tsv(inherited_predictor_path)
    if len(inherited) != int(config["sample"]["expected_inherited_predictors"]):
        raise GravityItem31Error("inherited Item 30 predictor count changed")
    predecessor_roles = {str(row["plateifu"]) for row in predecessor_sample["objects"]}
    if len(predecessor_roles) != int(config["sample"]["expected_item30_roles_excluded"]):
        raise GravityItem31Error("Item 30 role exclusion count changed")
    after_role_veto = [row for row in inherited if str(row["plateifu"]) not in predecessor_roles]
    snr_floor = float(config["sample"]["predictor_minimum_snr_med_g"])
    eligible = sorted(
        [row for row in after_role_veto if float(row["snr_med_g"]) >= snr_floor],
        key=lambda row: str(row["plateifu"]),
    )
    if len(eligible) != int(config["sample"]["expected_fresh_predictor_pool"]):
        raise GravityItem31Error("fresh Item 31 predictor pool changed")
    if not eligible:
        raise GravityItem31Error("empty Item 31 predictor pool")

    _write_tsv(paths["predictors"], eligible, tuple(eligible[0].keys()))
    sample = _sample_manifest(config, eligible)
    _write_json(paths["sample_manifest"], sample)
    candidates = _candidate_manifest(config)
    _write_json(paths["candidate_manifest"], candidates)
    source_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item31-predictor-source-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sources": {
                "inherited_predictors": str(config["sources"]["inherited_predictors"]),
                "inherited_predictors_sha256": _sha256_file(inherited_predictor_path),
                "inherited_predictor_manifest": str(
                    config["sources"]["inherited_predictor_manifest"]
                ),
                "inherited_predictor_manifest_sha256": _sha256_file(inherited_manifest_path),
                "item30_sample_manifest": str(config["sources"]["item30_sample_manifest"]),
                "item30_sample_manifest_sha256": _sha256_file(predecessor_sample_path),
            },
            "counts": {
                "inherited_predictors": len(inherited),
                "item30_roles_excluded": len(predecessor_roles),
                "remaining_after_role_veto": len(after_role_veto),
                "low_snr_excluded": len(after_role_veto) - len(eligible),
                "fresh_predictor_pool": len(eligible),
                "response_columns_requested": 0,
                "response_rows_read": 0,
                "confirmation_values_read": 0,
                "paid_api_calls": 0,
            },
            "files": {"predictors_sha256": _sha256_file(paths["predictors"])},
            "claims": {
                "predecessor_exclusion_before_role_assignment": True,
                "snr_filter_response_blind": True,
                "response_opened": False,
                "sample_target_blind": True,
                "confirmation_opened": False,
            },
        }
    )
    _write_json(paths["predictor_source_manifest"], source_manifest)
    return paths


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item31/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not body:
        raise GravityItem31Error(f"empty source response: {url}")
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
        raise GravityItem31Error("empty SkyServer CSV after comment filtering")
    reader = csv.DictReader(io.StringIO("\n".join(table_lines)))
    rows = [
        {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
        for row in reader
    ]
    if reader.fieldnames == ["error_message"]:
        message = rows[0]["error_message"] if rows else "unknown SkyServer error"
        raise GravityItem31Error(message)
    return rows, comments


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "Item 31 sample manifest")
    exploration = sorted(
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "exploration"
    )
    confirmations = {
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if len(exploration) != int(config["sample"]["expected_exploration"]):
        raise GravityItem31Error("Item 31 exploration role count changed before query")

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
            raise GravityItem31Error("MaNGA response schema changed")
        returned = {row["plateifu"] for row in rows}
        if returned & confirmations:
            raise GravityItem31Error("confirmation response entered Item 31 acquisition")
        if not returned <= set(identities):
            raise GravityItem31Error("unrequested MaNGA response entered Item 31")
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
        raise GravityItem31Error("duplicate MaNGA response row")
    all_rows = sorted(all_rows, key=lambda row: row["plateifu"])
    _write_tsv(paths["exploration_responses"], all_rows, expected_columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item31-response-source-1.0",
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
        raise GravityItem31Error(f"invalid response value {key}") from error
    if not np.isfinite(value):
        raise GravityItem31Error(f"nonfinite response value {key}")
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
        raise GravityItem31Error("Item 31 response file changed")
    if int(response_manifest["counts"]["confirmation_values_read"]) != 0:
        raise GravityItem31Error("Item 31 response manifest opened confirmations")

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
        except GravityItem31Error:
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
            "schema_version": "invariant-gravity-item31-extraction-1.0",
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


def _permittivity_predictors(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        "acceleration_m_s2": np.asarray(
            [float(row["internal_acceleration_m_s2"]) for row in rows], dtype=np.float64
        ),
        "density_msun_kpc3": np.asarray(
            [float(row["mean_stellar_density_msun_kpc3"]) for row in rows],
            dtype=np.float64,
        ),
        "q_lss": np.asarray([float(row["gema_q_lss"]) for row in rows], dtype=np.float64),
        "eta_k": np.asarray([float(row["gema_eta_k"]) for row in rows], dtype=np.float64),
        "dnn_mpc": np.asarray([float(row["gema_dnn_mpc"]) for row in rows], dtype=np.float64),
    }


def _build_candidate_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    predictors = {key: xp.asarray(value) for key, value in _permittivity_predictors(rows).items()}
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(_candidate_delta_log10_sigma(config, arrays, predictors, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _candidate_record(
    index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(arrays["niche"][index])
    record = {
        "admissible_index": index,
        "niche_index": niche,
        "niche": config["candidate_generator"]["niches"][niche]["id"],
        "creativity_label": config["candidate_generator"]["niches"][niche]["creativity_label"],
        "polarity": float(values["polarity"][0]),
        "amplitude": float(values["amplitude"][0]),
        "threshold_index": int(values["threshold_index"][0]),
        "sharpness": float(values["sharpness"][0]),
        "environment_coupling": float(values["environment"][0]),
        "shape_power": float(values["shape"][0]),
        "scale_factor": float(values["scale"][0]),
        "log10_acceleration_threshold_m_s2": float(values["acceleration_threshold"][0]),
        "log10_density_threshold_msun_kpc3": float(values["density_threshold"][0]),
    }
    return record


def _synthetic_controls_item31(
    delta: Any,
    base: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    xp: Any,
) -> dict[str, Any]:
    injection_results = []
    injection_indices = []
    maximum_environment_index = len(config["candidate_generator"]["environment_couplings"]) - 1
    for niche in range(4):
        mask = arrays["niche"] == niche
        # The nonlocal control must exercise its constitutive-environment term rather
        # than an environment=0 cell that is deliberately equivalent to niche 1.
        if niche == 2:
            mask &= arrays["environment"] == maximum_environment_index
        indices = np.where(mask)[0]
        if not len(indices):
            raise GravityItem31Error(f"missing synthetic injection niche {niche}")
        niche_values = delta[xp.asarray(indices)]
        variance = xp.var(niche_values, axis=1)
        injection_indices.append(int(indices[int(_to_numpy(xp.argmax(variance), xp))]))
    for index in injection_indices:
        target = base + _to_numpy(delta[index], xp)
        selected = _screen_candidate_matrix(delta, target, base, folds, config, xp)
        selected_niches = [int(arrays["niche"][value]) for value in selected["selected_indices"]]
        injection_results.append(
            {
                "injection_index": index,
                "injection_niche": int(arrays["niche"][index]),
                "injection_environment_index": int(arrays["environment"][index]),
                "selected_niches": selected_niches,
                "exact_niche_recovered_all_folds": all(
                    value == int(arrays["niche"][index]) for value in selected_niches
                ),
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
        "nonlocal_injection_requires_maximum_environment_coupling": True,
        "all_injected_niches_recovered": all(
            row["exact_niche_recovered_all_folds"] for row in injection_results
        ),
        "GR_candidate_mse": candidate_mse,
        "GR_baseline_mse": baseline_mse,
        "GR_control_candidate_improves": candidate_mse < baseline_mse - 1e-16,
    }


def _evaluate(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) < int(config["sample"]["outer_folds"]):
        raise GravityItem31Error("too few Item 31 response-complete galaxies")
    arrays, candidate_audit = _admissible_candidates(config)
    target = np.asarray([float(row["y_log10_sigma"]) for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
    expected_folds = set(range(int(config["sample"]["outer_folds"])))
    if set(folds.tolist()) != expected_folds:
        raise GravityItem31Error("Item 31 response-complete folds are incomplete")
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
        config, arrays, _permittivity_predictors(rows), 0, crosscheck, np
    )
    gpu_delta = _to_numpy(delta[:crosscheck], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu_delta - gpu_delta)))

    observed = _screen_candidate_matrix(delta, target, base, folds, config, xp)
    baselines = _baseline_predictions(target, base, folds, rows, config)
    candidate_mse = _mse(target, observed["prediction"])
    baseline_mse = {key: _mse(target, value) for key, value in baselines.items()}
    observed_statistic = _improvement(baseline_mse["flexible_nuisance"], candidate_mse)

    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    trials = int(config["evaluation"]["permutation_trials"])
    null_improvements = []
    for _ in range(trials):
        null_target = target[random.permutation(len(target))]
        null_selected = _screen_candidate_matrix(delta, null_target, base, folds, config, xp)
        null_flexible = _baseline_predictions(null_target, base, folds, rows, config)[
            "flexible_nuisance"
        ]
        null_improvements.append(
            _improvement(
                _mse(null_target, null_flexible),
                _mse(null_target, null_selected["prediction"]),
            )
        )
    p_value = (1.0 + sum(value >= observed_statistic for value in null_improvements)) / (
        trials + 1.0
    )
    controls = _synthetic_controls_item31(delta, base, folds, config, arrays, xp)
    selected_records = [
        _candidate_record(index, config, arrays) for index in observed["selected_indices"]
    ]
    selected_niches = [int(arrays["niche"][index]) for index in observed["selected_indices"]]
    niche_counts = Counter(selected_niches)

    mass = np.asarray([float(row["stellar_mass_msun"]) for row in rows])
    acceleration = np.asarray([float(row["internal_acceleration_m_s2"]) for row in rows])
    density = np.asarray([float(row["log_mean_stellar_density"]) for row in rows])
    q_lss = np.asarray([float(row["gema_q_lss"]) for row in rows])
    slice_masks = {
        "low_mass": mass <= np.median(mass),
        "high_mass": mass > np.median(mass),
        "low_acceleration": acceleration <= np.median(acceleration),
        "high_acceleration": acceleration > np.median(acceleration),
        "low_density": density <= np.median(density),
        "high_density": density > np.median(density),
        "low_external_tide": q_lss <= np.median(q_lss),
        "high_external_tide": q_lss > np.median(q_lss),
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
            (target - observed["prediction"]) ** 2 > (target - baselines["flexible_nuisance"]) ** 2
        )
    )
    required = int(config["sample"]["minimum_complete_exploration_objects"])
    fraction = len(rows) / int(config["sample"]["expected_exploration"])
    quality_pass = len(rows) >= required and fraction >= float(
        config["sample"]["minimum_quality_retention_fraction"]
    )
    gates = config["gates"]
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
        "improvement_vs_flexible": observed_statistic
        >= float(gates["minimum_improvement_vs_flexible"]),
        "each_broad_half_improves_baryonic_virial": all(
            value["improvement_vs_baryonic_virial"]
            >= float(gates["minimum_each_broad_half_improvement_vs_baryonic_virial"])
            for value in slices.values()
        ),
        "selection_aware_permutation": p_value
        <= float(gates["maximum_selection_aware_permutation_p"]),
        "stable_niche": max(niche_counts.values()) >= int(gates["minimum_same_niche_folds"]),
        "all_injected_niches_recovered": bool(controls["all_injected_niches_recovered"]),
        "known_GR_control": not bool(controls["GR_control_candidate_improves"]),
        "cpu_gpu_agreement": cpu_gpu_max <= 1e-11,
        "local_limit": candidate_audit["maximum_admitted_local_fractional_response"]
        <= float(config["admissibility"]["maximum_local_fractional_response"]),
    }
    phenomenon_gates = {
        "response_quality": quality_pass,
        "improvement_vs_flexible": observed_statistic
        >= float(gates["phenomenon_minimum_improvement_vs_flexible"]),
        "selection_aware_permutation": p_value
        <= float(gates["phenomenon_maximum_selection_aware_p"]),
        "stable_niche": universal_gates["stable_niche"],
        "controls": universal_gates["all_injected_niches_recovered"]
        and universal_gates["known_GR_control"]
        and universal_gates["cpu_gpu_agreement"],
    }
    partial_slices = [
        label
        for label, value in slices.items()
        if value["improvement_vs_flexible_nuisance"]
        >= float(gates["partial_minimum_slice_improvement_vs_flexible"])
    ]
    universal_pass = all(universal_gates.values())
    phenomenon_pass = all(phenomenon_gates.values())
    if not quality_pass:
        decision = "INCONCLUSIVE_ITEM31_QUALITY"
    elif universal_pass:
        decision = "PASS_ITEM31_EXPLORATION_UNIVERSAL"
    elif phenomenon_pass:
        decision = "NONPROMOTED_ITEM31_PHENOMENON_LEAD"
    elif partial_slices:
        decision = "SCOPED_ITEM31_PARTIAL_PATTERN_RETAINED"
    else:
        decision = "SCOPED_ITEM31_REJECT"

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
        "partial_track": {
            "retained_slices": partial_slices,
            "paper_claim_authorized": False,
        },
        "metrics": {
            "candidate_mse": candidate_mse,
            "baseline_mse": baseline_mse,
            "improvement_vs_baryonic_virial": _improvement(
                baseline_mse["baryonic_virial"], candidate_mse
            ),
            "improvement_vs_structural": _improvement(
                baseline_mse["structural_ridge"], candidate_mse
            ),
            "improvement_vs_flexible": observed_statistic,
            "selection_aware_permutation_p": p_value,
            "maximum_null_improvement": max(null_improvements),
            "object_counterexamples_vs_flexible": object_counterexamples,
        },
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
            "object_counterexamples_vs_flexible": object_counterexamples,
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
    test_path = root / "tests/test_gravity_item31_vacuum_permittivity.py"
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item31-vacuum-permittivity-result-1.0",
            "item": 31,
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
                "The tested epsilon_eff is an integrated phenomenological projection. It is not a measured vacuum material property or a solution of a covariant field equation.",
                "The stellar-mass proxy, Sersic virial map, orbital anisotropy, stellar populations, and projected GEMA environment carry ordinary astrophysical and catalog systematics.",
                "The acceleration-polarization branch overlaps known MOND-like and dipolar-medium behavior; a positive fit would not establish historical novelty or literal vacuum polarization.",
                "The nonlocal branch uses a bounded one-object neighbor contrast and is not a causal constitutive spacetime kernel.",
                "This single integrated motion observable cannot establish gravitational slip, direct lensing, clusters, cosmology, a covariant completion, or an alternative to GR.",
                "A positive result is exploration evidence only; confirmations stay sealed and an unchanged fresh replication is mandatory for a paper claim.",
            ],
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest, extraction = _load_response_rows(root, config)
    scientific, compute = _evaluate(config, rows)
    paths = _source_paths(root, config)
    compute_manifest = _content_hashed(
        {"schema_version": "invariant-gravity-item31-compute-1.0", **compute}
    )
    _write_json(paths["compute_manifest"], compute_manifest)
    receipt = _build_receipt(root, config, rows, response_manifest, extraction, scientific, compute)
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, receipt)
    return result_path


def validate_checked(root: Path) -> None:
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
    response = _read_json(paths["response_source_manifest"])
    if int(predictor["counts"]["response_rows_read"]) != 0:
        raise GravityItem31Error("Item 31 predictor freeze contains response values")
    if int(sample["counts"]["reserved_confirmation"]) != int(
        config["sample"]["expected_confirmation"]
    ):
        raise GravityItem31Error("Item 31 confirmation allocation changed")
    if int(response["counts"]["confirmation_values_read"]) != 0:
        raise GravityItem31Error("Item 31 response acquisition opened confirmations")
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "Item 31 result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem31Error("checked Item 31 result opened confirmation responses")
    if bool(result["frozen_boundary"]["post_response_formula_generation"]):
        raise GravityItem31Error("checked Item 31 result contains post-response generation")
    if (
        _sha256_file(paths["exploration_responses"])
        != result["frozen_boundary"]["response_file_sha256"]
    ):
        raise GravityItem31Error("checked Item 31 response file changed")


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
