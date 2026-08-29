"""Frozen Item 34 condensate and superfluid two-tracer search."""

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
    _candidate_digest,
    _minimum_separations_arcsec,
    _ridge_oof,
    _screen_candidate_matrix,
    _validate_legacy_content_hash,
    _virial_oof,
)

CONFIG_PATH = Path("configs/gravity_item34_condensate_superfluid_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item34_condensate_superfluid.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
_ADMISSIBLE_CACHE: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


class GravityItem34Error(RuntimeError):
    """Raised when an Item 34 freeze, leakage, ontology, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item34-condensate-superfluid-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 34:
        raise GravityItem34Error("unexpected Item 34 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem34Error("stable gravity goal changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem34Error("raw candidate boundary changed")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem34Error("post-response candidates entered Item 34")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem34Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem34Error("paid calls are outside Item 34")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem34Error("equal-viability policy changed")
    if not bool(
        config["discovery_policy"][
            "single_empirical_counterexample_is_not_a_formula_family_veto"
        ]
    ):
        raise GravityItem34Error("single-counterexample retention policy changed")
    if bool(config["gates"]["single_empirical_counterexample_is_veto"]):
        raise GravityItem34Error("empirical single-counterexample veto entered Item 34")
    niches = generator["niches"]
    if sum(bool(row["gravity_track_eligible"]) for row in niches) != 2:
        raise GravityItem34Error("matter ontology allocation changed")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem34Error(f"scientific dependency changed: {relative}")
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
        raise GravityItem34Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem34Error("could not read frozen Item 34 module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem34Error("Item 34 module differs from scientific freeze")


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in ("predictors", "predictor_source_manifest", "sample_manifest", "candidate_manifest"):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem34Error(f"{key} differs from sample freeze")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    radices = {
        "polarity": len(generator["polarities"]),
        "amplitude": len(generator["amplitudes"]),
        "coherence": len(generator["coherence_lengths_kpc"]),
        "width": len(generator["transition_widths"]),
        "power": len(generator["powers"]),
        "coupling": len(generator["baryon_couplings"]),
        "latent": len(generator["latent_strengths"]),
        "side": len(generator["phase_sides"]),
        "mode": len(generator["profile_modes"]),
    }
    per_niche = int(generator["raw_candidate_cells"]) // 4
    if int(np.prod(list(radices.values()))) != per_niche:
        raise GravityItem34Error("mixed-radix grammar does not fill each niche exactly")
    pieces: dict[str, list[np.ndarray]] = {"niche": []} | {key: [] for key in radices}
    for niche in range(4):
        working = np.arange(per_niche, dtype=np.int64)
        decoded: dict[str, np.ndarray] = {}
        for key, radix in reversed(list(radices.items())):
            decoded[key] = (working % radix).astype(np.int16)
            working //= radix
        if np.any(working != 0):
            raise GravityItem34Error("candidate decoder overflow")
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
    return {
        "niche": xp.asarray(index["niche"]),
        "polarity": xp.asarray(np.asarray(generator["polarities"])[index["polarity"]]),
        "amplitude": xp.asarray(np.asarray(generator["amplitudes"])[index["amplitude"]]),
        "coherence_length": xp.asarray(
            np.asarray(generator["coherence_lengths_kpc"])[index["coherence"]]
        ),
        "width": xp.asarray(np.asarray(generator["transition_widths"])[index["width"]]),
        "power": xp.asarray(np.asarray(generator["powers"])[index["power"]]),
        "coupling": xp.asarray(np.asarray(generator["baryon_couplings"])[index["coupling"]]),
        "latent": xp.asarray(np.asarray(generator["latent_strengths"])[index["latent"]]),
        "side": xp.asarray(np.asarray(generator["phase_sides"])[index["side"]]),
        "mode": xp.asarray(np.asarray(generator["profile_modes"])[index["mode"]]),
    }


def _coherence_predictors(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    log_mass = np.asarray([float(row["log_stellar_mass"]) for row in rows], dtype=np.float64)
    log_radius = np.asarray([float(row["log_half_light_radius"]) for row in rows], dtype=np.float64)
    log_density = np.asarray(
        [float(row["log_mean_stellar_density"]) for row in rows], dtype=np.float64
    )
    log_vbar = np.asarray([float(row["log_baryonic_speed_km_s"]) for row in rows])
    return {
        "log_acceleration": np.log10(
            np.asarray([float(row["internal_acceleration_m_s2"]) for row in rows])
        ),
        "log_density": log_density,
        "log_radius": log_radius,
        "log_mass": log_mass,
        "log_surface_density": np.asarray([float(row["log_surface_density"]) for row in rows]),
        "log_vbar": log_vbar,
        "phase_space": log_density - 3.0 * (log_vbar - 2.0),
        "age": (np.asarray([float(row["dn4000"]) for row in rows]) - 1.5) / 0.35,
        "sfr": (np.asarray([float(row["log_specific_sfr"]) for row in rows]) + 11.0) / 1.5,
        "axis_ratio": np.asarray([float(row["axis_ratio"]) for row in rows]),
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
    shape = (-1, 1)
    niche = values["niche"].reshape(shape)
    side = values["side"].reshape(shape)
    width = values["width"].reshape(shape)
    power = values["power"].reshape(shape)
    coupling = values["coupling"].reshape(shape)
    latent = values["latent"].reshape(shape)
    mode = values["mode"].reshape(shape)
    coherence = values["coherence_length"].reshape(shape)
    log_acceleration = xp.asarray(predictors["log_acceleration"])[None, :]
    log_density = xp.asarray(predictors["log_density"])[None, :]
    log_radius = xp.asarray(predictors["log_radius"])[None, :]
    log_mass = xp.asarray(predictors["log_mass"])[None, :]
    log_surface = xp.asarray(predictors["log_surface_density"])[None, :]
    log_vbar = xp.asarray(predictors["log_vbar"])[None, :]
    phase_space = xp.asarray(predictors["phase_space"])[None, :]
    age = xp.asarray(predictors["age"])[None, :]
    sfr = xp.asarray(predictors["sfr"])[None, :]

    log_lambda_db = xp.log10(coherence) - (log_vbar - 2.0)
    occupation = (
        phase_space
        - 8.0
        + 3.0 * log_lambda_db
        + coupling * (age - 0.5 * sfr)
        + 0.25 * mode * (log_density - 8.0)
    ) / width
    condensed = xp.power(0.5 * (1.0 + xp.tanh(side * occupation)), power)
    log_a0 = math.log10(float(config["physics"]["constants"]["a0_m_s2"]))
    phonon = xp.sqrt(1.0 / (1.0 + xp.power(10.0, log_acceleration - log_a0)))
    superfluid = condensed * xp.power(phonon, 1.0 / (1.0 + latent))

    radius = xp.power(10.0, log_radius)
    core_radius = coherence * xp.power(
        10.0, coupling * (10.0 - log_mass) * (1.0 + latent) * (1.0 + 0.5 * mode)
    )
    soliton_profile = xp.power(1.0 + 0.091 * xp.square(radius / core_radius), -8.0)
    soliton = xp.power(
        xp.clip(0.5 * (1.0 + side * (2.0 * soliton_profile - 1.0)), 0.0, 1.0),
        power,
    )

    healing_length = coherence * xp.power(
        10.0, coupling * (log_surface - 8.5) + 0.25 * mode * (log_density - 8.0)
    )
    healing_ratio = radius / healing_length
    helmholtz = 1.0 - (1.0 + healing_ratio) * xp.exp(-healing_ratio)
    coherent = xp.power(
        xp.clip(0.5 * (1.0 + side * (2.0 * helmholtz - 1.0)), 0.0, 1.0),
        power,
    ) * xp.power(phonon, 1.0 / (1.0 + latent))

    mismatch = (
        log_radius
        - xp.log10(coherence)
        + coupling * (age - 0.5 * sfr)
        + 0.25 * latent * (log_surface - 8.5)
        + 0.5 * mode * (xp.asarray(predictors["axis_ratio"])[None, :] - 0.6)
    ) / width
    locked_window = xp.exp(-xp.power(xp.abs(mismatch), power))
    locked = xp.clip(0.5 * (1.0 + side * (2.0 * locked_window - 1.0)), 0.0, 1.0)

    activation = xp.where(
        niche == 0,
        superfluid,
        xp.where(niche == 1, soliton, xp.where(niche == 2, coherent, locked)),
    )
    shield_center = float(config["physics"]["universal_local_shield_log10_acceleration_m_s2"])
    shield_power = float(config["physics"]["universal_local_shield_power"])
    local_shield = 1.0 / (1.0 + xp.power(10.0, shield_power * (log_acceleration - shield_center)))
    return xp.clip(activation * local_shield, 0.0, 1.0)


def _candidate_delta_log10_velocity(
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
    log_mass = 8.0 + 3.5 * ((index * 17) % 64) / 63.0
    log_radius = -0.5 + 1.8 * ((index * 29) % 64) / 63.0
    log_vbar = 0.5 * (math.log10(4.30091e-6) + log_mass - log_radius)
    log_density = log_mass - math.log10(4.0 * math.pi / 3.0) - 3.0 * log_radius
    return {
        "log_acceleration": np.linspace(-13.0, -8.5, 64),
        "log_density": log_density,
        "log_radius": log_radius,
        "log_mass": log_mass,
        "log_surface_density": 6.5 + 4.0 * ((index * 37) % 64) / 63.0,
        "log_vbar": log_vbar,
        "phase_space": log_density - 3.0 * (log_vbar - 2.0),
        "age": -2.0 + 4.0 * ((index * 43) % 64) / 63.0,
        "sfr": -2.0 + 4.0 * ((index * 47) % 64) / 63.0,
        "axis_ratio": 0.25 + 0.7 * ((index * 53) % 64) / 63.0,
    }


def _local_predictors(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    log_radius = math.log10(4.848136811e-9)
    log_mass = 0.0
    log_vbar = 0.5 * (math.log10(4.30091e-6) + log_mass - log_radius)
    log_density = 25.0
    return {
        "log_acceleration": np.asarray(
            [math.log10(float(config["physics"]["constants"]["one_au_acceleration_m_s2"]))]
        ),
        "log_density": np.asarray([log_density]),
        "log_radius": np.asarray([log_radius]),
        "log_mass": np.asarray([log_mass]),
        "log_surface_density": np.asarray([20.0]),
        "log_vbar": np.asarray([log_vbar]),
        "phase_space": np.asarray([log_density - 3.0 * (log_vbar - 2.0)]),
        "age": np.asarray([0.0]),
        "sfr": np.asarray([0.0]),
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
    activation_span = np.full(count, np.nan)
    local_response = np.full(count, np.nan)
    batch = int(config["evaluation"]["candidate_batch_size"])
    gates = config["admissibility"]
    for begin in range(0, count, batch):
        end = min(begin + batch, count)
        values = _candidate_values(config, raw, begin, end, np)
        activation = _candidate_activation(config, raw, domain, begin, end, np)
        mu = 1.0 + values["polarity"][:, None] * values["amplitude"][:, None] * activation
        local_activation = _candidate_activation(config, raw, local, begin, end, np)[:, 0]
        local_mu = 1.0 + values["polarity"] * values["amplitude"] * local_activation
        minimum_mu[begin:end] = np.min(mu, axis=1)
        maximum_mu[begin:end] = np.max(mu, axis=1)
        material_response[begin:end] = np.max(np.abs(mu - 1.0), axis=1)
        activation_span[begin:end] = np.max(activation, axis=1) - np.min(activation, axis=1)
        local_response[begin:end] = np.abs(local_mu - 1.0)
        keep[begin:end] = (
            np.all(np.isfinite(mu), axis=1)
            & (minimum_mu[begin:end] >= float(gates["minimum_mu"]))
            & (maximum_mu[begin:end] <= float(gates["maximum_mu"]))
            & (material_response[begin:end] >= float(gates["minimum_material_fractional_response"]))
            & (activation_span[begin:end] >= float(gates["minimum_activation_span"]))
            & (local_response[begin:end] <= float(gates["maximum_local_fractional_response"]))
        )
    arrays = {key: value[keep] for key, value in raw.items()}
    signature_parts = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        delta = _candidate_delta_log10_velocity(config, arrays, domain, begin, end, np)
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
        "minimum_admitted_activation_span": float(np.min(activation_span[keep])),
        "maximum_admitted_local_fractional_response": float(np.max(local_response[keep])),
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
            raise GravityItem34Error(f"candidate invariant changed: {expected_key}")
    expected_niches = generator.get("expected_admissible_per_niche")
    if (
        expected_niches
        and all(int(value) >= 0 for value in expected_niches.values())
        and audit["admissible_per_niche"] != expected_niches
    ):
        raise GravityItem34Error("admissible niche counts changed")
    _ADMISSIBLE_CACHE[cache_key] = arrays, audit
    return arrays, audit


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item34-candidate-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "niches": config["candidate_generator"]["niches"],
            "historical_novelty_claimed": False,
            "ontology_counts_raw": {
                "hidden_matter_required": 131072,
                "baryon_sourced_gravitational_sector": 131072,
            },
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "post_response_cells": 0,
            "audit": audit,
        }
    )


def _fresh_pool(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictor_source = _read_json(root / str(config["sources"]["manga_predictor_source"]))
    _validate_legacy_content_hash(predictor_source, "Item 34 frozen MaNGA predictor source")
    if int(predictor_source["counts"]["response_columns_requested"]) != 0:
        raise GravityItem34Error("response column entered inherited MaNGA predictors")
    records = predictor_source["records"]
    independence = config["independence"]
    if len(records) != int(independence["expected_source_predictors"]):
        raise GravityItem34Error("inherited MaNGA predictor count changed")

    coordinate_rows = _read_tsv(root / str(config["sources"]["predecessor_coordinates"]))
    identity_rows = _read_tsv(root / str(config["sources"]["predecessor_identities"]))
    plateifus = {row["value"] for row in identity_rows if row["kind"] == "plateifu"}
    mangaids = {row["value"] for row in identity_rows if row["kind"] == "mangaid"}
    coordinates = [
        [float(row["ra_deg"]), float(row["dec_deg"])] for row in coordinate_rows
    ]
    role_receipts = []
    for relative in config["sources"]["current_predecessor_samples"]:
        path = root / str(relative)
        manifest = _read_json(path)
        _verify_content_hash(manifest, str(relative))
        objects = manifest["objects"]
        role_receipts.append(
            {"path": str(relative), "sha256": _sha256_file(path), "objects": len(objects)}
        )
        for row in objects:
            plateifus.add(str(row["plateifu"]))
            if row.get("mangaid") is not None:
                mangaids.add(str(row["mangaid"]))
            ra = row.get("ra", row.get("objra", row.get("ra_deg")))
            dec = row.get("dec", row.get("objdec", row.get("dec_deg")))
            if ra is None or dec is None:
                raise GravityItem34Error("current MaNGA predecessor role lacks coordinates")
            coordinates.append([float(ra), float(dec)])
    expected_counts = {
        "plateifu_ids": int(independence["expected_predecessor_plateifu_ids"]),
        "mangaids": int(independence["expected_predecessor_mangaids"]),
        "coordinate_rows": int(independence["expected_coordinate_rows"]),
    }
    observed_counts = {
        "plateifu_ids": len(plateifus),
        "mangaids": len(mangaids),
        "coordinate_rows": len(coordinates),
    }
    if observed_counts != expected_counts:
        raise GravityItem34Error("Item 34 predecessor union changed")
    separations = _minimum_separations_arcsec(records, np.asarray(coordinates, dtype=np.float64))
    coordinate_veto = float(independence["coordinate_veto_arcsec"])
    fresh: list[dict[str, Any]] = []
    exclusions = Counter()
    for source, separation in zip(records, separations, strict=True):
        if str(source["plateifu"]) in plateifus or str(source["mangaid"]) in mangaids:
            exclusions["predecessor_identity"] += 1
            continue
        if separation <= coordinate_veto:
            exclusions["predecessor_coordinate"] += 1
            continue
        row = dict(source)
        row["minimum_predecessor_separation_arcsec"] = float(separation)
        fresh.append(row)
    if exclusions != Counter(
        {
            "predecessor_identity": int(independence["expected_identity_exclusions"]),
            "predecessor_coordinate": int(independence["expected_coordinate_exclusions"]),
        }
    ):
        raise GravityItem34Error("Item 34 predecessor exclusion counts changed")
    if len(fresh) != int(independence["expected_fresh_before_quality"]):
        raise GravityItem34Error("Item 34 fresh pre-quality count changed")

    constants = config["physics"]["constants"]
    pool = []
    for source in fresh:
        if float(source["snr_med_g"]) < float(config["sample"]["predictor_minimum_snr_med_g"]):
            continue
        if float(source["sersic_index"]) <= 0.0 or float(source["axis_ratio"]) <= 0.0:
            continue
        mass = 10.0 ** float(source["log_stellar_mass"])
        radius = 10.0 ** float(source["log_half_light_radius"])
        speed = math.sqrt(float(constants["G_kpc_km2_s2_Msun"]) * mass / radius)
        acceleration = (
            float(constants["G_kpc_km2_s2_Msun"])
            * mass
            / radius**2
            * 1000.0**2
            / float(constants["kpc_to_m"])
        )
        density = mass / (4.0 * math.pi * radius**3 / 3.0)
        row = dict(source)
        row.update(
            {
                "stellar_mass_msun": mass,
                "half_light_radius_kpc": radius,
                "log_baryonic_speed_km_s": math.log10(speed),
                "internal_acceleration_m_s2": acceleration,
                "mean_stellar_density_msun_kpc3": density,
                "log_mean_stellar_density": math.log10(density),
                "phase_space_proxy": math.log10(density) - 3.0 * (math.log10(speed) - 2.0),
            }
        )
        pool.append(row)
    pool.sort(key=lambda row: str(row["plateifu"]))
    if len(pool) != int(config["sample"]["expected_fresh_predictor_pool"]):
        raise GravityItem34Error("Item 34 fresh predictor pool changed")
    audit = {
        "base_predecessor_coordinates": len(coordinate_rows),
        "base_predecessor_identity_rows": len(identity_rows),
        "predecessor_union": observed_counts,
        "current_role_receipts": role_receipts,
        "exclusions": dict(sorted(exclusions.items())),
        "fresh_before_quality": len(fresh),
        "fresh_predictor_pool": len(pool),
    }
    return pool, audit


def _coherence_control_values(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    predictors = _coherence_predictors(rows)
    return {
        "phase_space": predictors["phase_space"],
        "log_acceleration": predictors["log_acceleration"],
        "log_radius": predictors["log_radius"],
        "log_surface_density": predictors["log_surface_density"],
        "age": predictors["age"],
    }


def _coherence_control_spec(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    quantiles = np.asarray(config["evaluation"]["coherence_quantiles"], dtype=np.float64)
    answer: dict[str, dict[str, Any]] = {}
    for label, values in _coherence_control_values(rows).items():
        center = float(np.mean(values))
        scale = max(float(np.std(values)), 1e-12)
        answer[label] = {
            "center": center,
            "scale": scale,
            "knots": [float(value) for value in np.quantile(values, quantiles)],
        }
    return answer


def _sample_manifest(
    config: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    sample = config["sample"]
    mass_median = float(np.median([float(row["log_stellar_mass"]) for row in pool]))
    surface_median = float(np.median([float(row["log_surface_density"]) for row in pool]))
    cells: dict[str, list[dict[str, Any]]] = {
        f"m{mass}-s{surface}": [] for mass in range(2) for surface in range(2)
    }
    for source in pool:
        row = dict(source)
        mass_bin = int(float(row["log_stellar_mass"]) >= mass_median)
        surface_bin = int(float(row["log_surface_density"]) >= surface_median)
        cell = f"m{mass_bin}-s{surface_bin}"
        row.update({"mass_bin": mass_bin, "surface_bin": surface_bin, "sample_cell": cell})
        cells[cell].append(row)
    capacities = {key: len(value) for key, value in cells.items()}
    if capacities != {key: int(value) for key, value in sample["expected_cell_capacities"].items()}:
        raise GravityItem34Error("Item 34 response-blind cell capacities changed")

    objects = []
    cell_counts = {}
    selected_count = int(sample["selected_per_mass_surface_cell"])
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
    fold_counts = Counter(
        int(row["outer_fold"]) for row in objects if row["role"] == "exploration"
    )
    exploration_predictors = [row for row in objects if row["role"] == "exploration"]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item34-sample-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "mass_median_log10_msun": f"{mass_median:.12e}",
            "surface_density_median_log10_msun_kpc2": f"{surface_median:.12e}",
            "ordinary_coherence_control_spec": _coherence_control_spec(
                exploration_predictors, config
            ),
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
    pool, predecessor_audit = _fresh_pool(root, config)
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
        raise GravityItem34Error("frozen Item 34 sample counts changed")
    columns = [
        *list(pool[0]),
        "mass_bin",
        "surface_bin",
        "sample_cell",
        "role",
        "outer_fold",
        "response_read",
    ]
    _write_tsv(paths["predictors"], sample_manifest["objects"], columns)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item34-predictor-source-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "manga_predictor_source": {
                "path": config["sources"]["manga_predictor_source"],
                "sha256": _sha256_file(root / str(config["sources"]["manga_predictor_source"])),
                "response_columns_read": 0,
            },
            "predecessor_audit": predecessor_audit,
            "counts": {
                "source_predictors": int(config["independence"]["expected_source_predictors"]),
                "fresh_predictor_pool": len(pool),
                "selected": len(sample_manifest["objects"]),
                "response_columns_read": 0,
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


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item34/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not body:
        raise GravityItem34Error(f"empty source response: {url}")
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
        raise GravityItem34Error("empty SkyServer CSV after comment filtering")
    reader = csv.DictReader(io.StringIO("\n".join(table_lines)))
    rows = [
        {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
        for row in reader
    ]
    if reader.fieldnames == ["error_message"]:
        message = rows[0]["error_message"] if rows else "unknown SkyServer error"
        raise GravityItem34Error(message)
    return rows, comments


def acquire_responses(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "Item 34 sample manifest")
    exploration = sorted(
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "exploration"
    )
    confirmations = {
        str(row["plateifu"])
        for row in sample["objects"]
        if row["role"] == "reserved_confirmation"
    }
    if len(exploration) != int(config["sample"]["expected_exploration"]):
        raise GravityItem34Error("Item 34 exploration role count changed before query")
    if len(confirmations) != int(config["sample"]["expected_confirmation"]):
        raise GravityItem34Error("Item 34 confirmation role count changed before query")
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
            raise GravityItem34Error("MaNGA response schema changed")
        returned = {row["plateifu"] for row in rows}
        if returned & confirmations:
            raise GravityItem34Error("confirmation response entered Item 34 acquisition")
        if not returned <= set(identities):
            raise GravityItem34Error("unrequested MaNGA response entered Item 34")
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
        raise GravityItem34Error("duplicate MaNGA response row")
    all_rows.sort(key=lambda row: row["plateifu"])
    _write_tsv(paths["exploration_responses"], all_rows, expected_columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item34-response-source-1.0",
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
        raise GravityItem34Error(f"invalid response value {key}") from error
    if not np.isfinite(value):
        raise GravityItem34Error(f"nonfinite response value {key}")
    return value


def _halpha_target(row: Mapping[str, Any], span: float, config: Mapping[str, Any]) -> float:
    q0 = float(config["physics"]["intrinsic_disk_axis_ratio"])
    q = float(row["axis_ratio"])
    cos2 = np.clip((q * q - q0 * q0) / (1.0 - q0 * q0), 0.0, 1.0)
    sine = max(math.sqrt(1.0 - float(cos2)), float(config["physics"]["minimum_transfer_sine_inclination"]))
    return math.log10(0.5 * span / sine)


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
    if _sha256_file(paths["exploration_responses"]) != response_manifest["response_file"][
        "sha256"
    ]:
        raise GravityItem34Error("Item 34 response file changed")
    if int(response_manifest["counts"]["confirmation_values_read"]) != 0:
        raise GravityItem34Error("Item 34 response manifest opened confirmations")
    sample_rows = {
        str(row["plateifu"]): row
        for row in sample["objects"]
        if row["role"] == "exploration"
    }
    response_rows = {row["plateifu"]: row for row in _read_tsv(paths["exploration_responses"])}
    quality = config["quality"]
    valid: list[dict[str, Any]] = []
    primary_failures: list[dict[str, Any]] = []
    transfer_failures: list[dict[str, Any]] = []
    for plateifu, predictor in sorted(sample_rows.items()):
        response = response_rows.get(plateifu)
        primary_reasons = []
        if response is None:
            primary_failures.append({"plateifu": plateifu, "reasons": ["missing_response_row"]})
            continue
        try:
            sigma = _finite(response, "stellar_sigma_1re")
            rchi2 = _finite(response, "stellar_rchi2_1re")
            stellar_low = _finite(response, "stellar_vel_lo_clip")
            stellar_high = _finite(response, "stellar_vel_hi_clip")
        except GravityItem34Error:
            primary_failures.append({"plateifu": plateifu, "reasons": ["incomplete_stellar"]})
            continue
        stellar_span = stellar_high - stellar_low
        if float(predictor["snr_med_g"]) < float(quality["minimum_snr_med_g"]):
            primary_reasons.append("low_predictor_snr")
        if rchi2 > float(quality["maximum_stellar_rchi2_1re"]):
            primary_reasons.append("stellar_rchi2")
        if not (
            float(quality["minimum_stellar_sigma_km_s"])
            <= sigma
            <= float(quality["maximum_stellar_sigma_km_s"])
        ):
            primary_reasons.append("stellar_sigma")
        if not (
            float(quality["minimum_stellar_velocity_span_km_s"])
            <= stellar_span
            <= float(quality["maximum_stellar_velocity_span_km_s"])
        ):
            primary_reasons.append("stellar_velocity_span")
        if primary_reasons:
            primary_failures.append({"plateifu": plateifu, "reasons": primary_reasons})
            continue
        row = dict(predictor)
        row.update(
            {
                "stellar_sigma_1re_km_s": sigma,
                "stellar_rchi2_1re": rchi2,
                "stellar_velocity_span_km_s": stellar_span,
                "y_log10_stellar_sigma": math.log10(sigma),
                "Halpha_transfer_valid": False,
                "y_log10_Halpha_speed": None,
            }
        )
        transfer_reasons = []
        try:
            gas_low = _finite(response, "ha_gvel_lo_clip")
            gas_high = _finite(response, "ha_gvel_hi_clip")
            gas_span = gas_high - gas_low
        except GravityItem34Error:
            gas_span = math.nan
            transfer_reasons.append("incomplete_Halpha")
        if np.isfinite(gas_span) and not (
            float(quality["minimum_Halpha_velocity_span_km_s"])
            <= gas_span
            <= float(quality["maximum_Halpha_velocity_span_km_s"])
        ):
            transfer_reasons.append("Halpha_velocity_span")
        if float(row["sersic_index"]) > float(quality["maximum_transfer_sersic_index"]):
            transfer_reasons.append("transfer_sersic")
        if not (
            float(quality["minimum_transfer_axis_ratio"])
            <= float(row["axis_ratio"])
            <= float(quality["maximum_transfer_axis_ratio"])
        ):
            transfer_reasons.append("transfer_axis_ratio")
        if transfer_reasons:
            transfer_failures.append({"plateifu": plateifu, "reasons": transfer_reasons})
        else:
            row["Halpha_transfer_valid"] = True
            row["Halpha_velocity_span_km_s"] = gas_span
            row["y_log10_Halpha_speed"] = _halpha_target(row, gas_span, config)
        valid.append(row)
    extraction = _content_hashed(
        {
            "schema_version": "invariant-gravity-item34-extraction-1.0",
            "exploration_roles": len(sample_rows),
            "response_rows": len(response_rows),
            "stellar_quality_passing": len(valid),
            "Halpha_transfer_passing": sum(bool(row["Halpha_transfer_valid"]) for row in valid),
            "primary_quality_failures": primary_failures,
            "primary_failure_reason_counts": dict(
                sorted(
                    Counter(
                        reason for row in primary_failures for reason in row["reasons"]
                    ).items()
                )
            ),
            "transfer_quality_failures": transfer_failures,
            "transfer_failure_reason_counts": dict(
                sorted(
                    Counter(
                        reason for row in transfer_failures for reason in row["reasons"]
                    ).items()
                )
            ),
            "confirmation_values_read": 0,
            "failed_identity_replacement": False,
            "ordinary_coherence_spec_is_response_blind": True,
        }
    )
    return valid, response_manifest, extraction


def _feature_matrix(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> np.ndarray:
    normalization = config["evaluation"]["fixed_feature_normalization"]
    raw = {
        "log_stellar_mass": [float(row["log_stellar_mass"]) for row in rows],
        "log_half_light_radius": [float(row["log_half_light_radius"]) for row in rows],
        "log_surface_density": [float(row["log_surface_density"]) for row in rows],
        "sersic_index": [float(row["sersic_index"]) for row in rows],
        "axis_ratio": [float(row["axis_ratio"]) for row in rows],
        "g_minus_r_color": [float(row["g_minus_r_color"]) for row in rows],
        "redshift": [float(row["redshift"]) for row in rows],
        "log_snr": [float(row["log_snr"]) for row in rows],
        "dn4000": [float(row["dn4000"]) for row in rows],
        "balmer_mean": [
            0.5 * (float(row["hdelta_a"]) + float(row["hgamma_a"])) for row in rows
        ],
        "log_specific_sfr": [float(row["log_specific_sfr"]) for row in rows],
        "signed_log_halpha_ew": [
            math.copysign(
                math.log10(1.0 + abs(float(row["halpha_ew"]))), float(row["halpha_ew"])
            )
            for row in rows
        ],
    }
    columns = []
    for key in normalization:
        center, scale = (float(value) for value in normalization[key])
        columns.append((np.asarray(raw[key], dtype=np.float64) - center) / scale)
    return np.column_stack(columns)


def _design_matrix(features: np.ndarray, flexible: bool) -> np.ndarray:
    if not flexible:
        return features[:, :8]
    pieces = [features, features * features]
    interactions = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (4, 3),
        (5, 0),
        (6, 0),
        (6, 1),
        (8, 0),
        (8, 2),
        (9, 8),
        (10, 2),
        (10, 8),
        (11, 2),
        (11, 8),
        (10, 11),
    )
    pieces.extend((features[:, left] * features[:, right])[:, None] for left, right in interactions)
    return np.column_stack(pieces)


def _ordinary_coherence_design(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    spec: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    features = _feature_matrix(rows, config)
    values = _coherence_control_values(rows)
    expected = set(values)
    if set(spec) != expected:
        raise GravityItem34Error("ordinary coherence control specification changed")
    smooth = []
    hinges = []
    for label in values:
        column = np.asarray(values[label], dtype=np.float64)
        center = float(spec[label]["center"])
        scale = float(spec[label]["scale"])
        normalized = (column - center) / scale
        smooth.append(normalized)
        for knot in spec[label]["knots"]:
            hinges.append(np.maximum(column - float(knot), 0.0) / scale)
            hinges.append(np.maximum(float(knot) - column, 0.0) / scale)
    smooth_matrix = np.column_stack(smooth)
    matched_interactions = np.column_stack(
        [
            smooth_matrix[:, 0] * smooth_matrix[:, 4],
            smooth_matrix[:, 1] * smooth_matrix[:, 3],
            smooth_matrix[:, 2] * smooth_matrix[:, 3],
            smooth_matrix[:, 0] * smooth_matrix[:, 1],
            smooth_matrix[:, 2] * smooth_matrix[:, 4],
        ]
    )
    return np.column_stack(
        [_design_matrix(features, flexible=True), smooth_matrix, matched_interactions, *hinges]
    )


def _baseline_predictions(
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    coherence_spec: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    features = _feature_matrix(rows, config)
    outer_folds = int(config["sample"]["outer_folds"])
    coherence_design = _ordinary_coherence_design(rows, config, coherence_spec)
    return {
        "baryonic_virial": _virial_oof(target, base, folds, config),
        "structural_ridge": _ridge_oof(
            target,
            folds,
            _design_matrix(features, flexible=False),
            float(config["evaluation"]["ridge_alpha_structural"]),
            outer_folds,
        ),
        "flexible_nuisance": _ridge_oof(
            target,
            folds,
            _design_matrix(features, flexible=True),
            float(config["evaluation"]["ridge_alpha_flexible"]),
            outer_folds,
        ),
        "ordinary_coherence": _ridge_oof(
            target,
            folds,
            coherence_design,
            float(config["evaluation"]["ridge_alpha_coherence"]),
            outer_folds,
        ),
    }


def _build_candidate_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    predictors = {key: xp.asarray(value) for key, value in _coherence_predictors(rows).items()}
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(
            _candidate_delta_log10_velocity(config, arrays, predictors, begin, end, xp)
        )
    return xp.concatenate(pieces, axis=0)


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
        "ontology": definition["ontology"],
        "gravity_track_eligible": bool(definition["gravity_track_eligible"]),
        "polarity": float(values["polarity"][0]),
        "amplitude": float(values["amplitude"][0]),
        "coherence_length_kpc": float(values["coherence_length"][0]),
        "transition_width": float(values["width"][0]),
        "power": float(values["power"][0]),
        "baryon_coupling": float(values["coupling"][0]),
        "latent_strength": float(values["latent"][0]),
        "phase_side": float(values["side"][0]),
        "profile_mode": float(values["mode"][0]),
    }


def _transfer_selected_candidates(
    delta: Any,
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    selected_indices: Sequence[int],
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    prediction = np.empty_like(target)
    offsets = []
    raw_offsets = []
    bounds = config["physics"]["shared_mass_proxy_scale_bounds"]
    lower = 0.5 * math.log10(float(bounds[0]))
    upper = 0.5 * math.log10(float(bounds[1]))
    outer_folds = int(config["sample"]["outer_folds"])
    if len(selected_indices) != outer_folds:
        raise GravityItem34Error("stellar candidate selection does not cover every fold")
    for fold in range(outer_folds):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        if not len(train) or not len(held):
            raise GravityItem34Error("Halpha transfer folds are incomplete")
        index = int(selected_indices[fold])
        correction = _to_numpy(delta[index], xp)
        raw = float(np.mean(target[train] - base[train] - correction[train]))
        fitted = float(np.clip(raw, lower, upper))
        raw_offsets.append(raw)
        offsets.append(fitted)
        prediction[held] = base[held] + correction[held] + fitted
    return {
        "prediction": prediction,
        "selected_indices_from_stellar": [int(value) for value in selected_indices],
        "log10_speed_offsets": offsets,
        "raw_log10_speed_offsets": raw_offsets,
        "formula_reselection_on_Halpha": False,
    }


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
            raise GravityItem34Error(f"missing synthetic injection niche {niche}")
        niche_values = delta[xp.asarray(indices)]
        variance = xp.var(niche_values, axis=1)
        index = int(indices[int(_to_numpy(xp.argmax(variance), xp))])
        target = base + _to_numpy(delta[index], xp)
        selected = _screen_candidate_matrix(delta, target, base, folds, config, xp)
        selected_niches = [int(arrays["niche"][value]) for value in selected["selected_indices"]]
        transferred = _transfer_selected_candidates(
            delta,
            target,
            base,
            folds,
            selected["selected_indices"],
            config,
            xp,
        )
        injection_results.append(
            {
                "injection_index": index,
                "injection_niche": niche,
                "selected_niches": selected_niches,
                "exact_niche_recovered_all_folds": all(value == niche for value in selected_niches),
                "candidate_mse": _mse(target, selected["prediction"]),
                "unchanged_transfer_mse": _mse(target, transferred["prediction"]),
                "transfer_reselected_formula": transferred["formula_reselection_on_Halpha"],
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
        "all_injected_niches_transfer_unchanged": all(
            not row["transfer_reselected_formula"] for row in injection_results
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


def _robust_comparison(
    target: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_error = np.square(target - candidate)
    reference_error = np.square(target - reference)
    comparative = candidate_error - reference_error
    full_improvement = _improvement(float(np.mean(reference_error)), float(np.mean(candidate_error)))
    influence_order = np.argsort(np.abs(comparative))[::-1]
    worst = int(influence_order[0])
    leave_one = np.ones(len(target), dtype=bool)
    leave_one[worst] = False
    leave_one_improvement = _improvement(
        float(np.mean(reference_error[leave_one])), float(np.mean(candidate_error[leave_one]))
    )
    trim_fraction = float(config["evaluation"]["robust_comparative_trim_fraction"])
    trim_count = max(1, math.floor(trim_fraction * len(target)))
    trimmed = np.ones(len(target), dtype=bool)
    trimmed[influence_order[:trim_count]] = False
    trimmed_improvement = _improvement(
        float(np.mean(reference_error[trimmed])), float(np.mean(candidate_error[trimmed]))
    )
    counterexamples = candidate_error > reference_error
    identity = str(rows[worst].get("plateifu", worst))
    return {
        "objects": len(target),
        "counterexamples": int(np.count_nonzero(counterexamples)),
        "counterexample_fraction": float(np.mean(counterexamples)),
        "full_improvement": full_improvement,
        "single_most_influential_identity": identity,
        "single_most_influential_comparative_squared_error": float(comparative[worst]),
        "leave_one_most_influential_improvement": leave_one_improvement,
        "leave_one_changes_improvement_sign": bool(
            (full_improvement >= 0.0) != (leave_one_improvement >= 0.0)
        ),
        "trim_fraction": trim_fraction,
        "trimmed_objects": trim_count,
        "trimmed_improvement": trimmed_improvement,
        "trim_changes_improvement_sign": bool(
            (full_improvement >= 0.0) != (trimmed_improvement >= 0.0)
        ),
        "single_counterexample_is_veto": False,
    }


def _evaluate(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    coherence_spec: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outer_folds = int(config["sample"]["outer_folds"])
    if len(rows) < outer_folds:
        raise GravityItem34Error("too few Item 34 response-complete galaxies")
    arrays, candidate_audit = _admissible_candidates(config)
    target = np.asarray([float(row["y_log10_stellar_sigma"]) for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
    if set(folds.tolist()) != set(range(outer_folds)):
        raise GravityItem34Error("Item 34 stellar response-complete folds are incomplete")
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
    cpu_delta = _candidate_delta_log10_velocity(
        config, arrays, _coherence_predictors(rows), 0, crosscheck, np
    )
    gpu_delta = _to_numpy(delta[:crosscheck], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu_delta - gpu_delta)))

    observed = _screen_candidate_matrix(delta, target, base, folds, config, xp)
    baselines = _baseline_predictions(target, base, folds, rows, config, coherence_spec)
    candidate_mse = _mse(target, observed["prediction"])
    baseline_mse = {key: _mse(target, prediction) for key, prediction in baselines.items()}
    observed_statistic = _improvement(baseline_mse["ordinary_coherence"], candidate_mse)

    transfer_indices = np.asarray(
        [index for index, row in enumerate(rows) if bool(row["Halpha_transfer_valid"])],
        dtype=np.int64,
    )
    transfer_rows = [rows[int(index)] for index in transfer_indices]
    transfer_target = np.asarray(
        [float(row["y_log10_Halpha_speed"]) for row in transfer_rows], dtype=np.float64
    )
    transfer_folds = folds[transfer_indices]
    if len(transfer_indices) and set(transfer_folds.tolist()) != set(range(outer_folds)):
        raise GravityItem34Error("Item 34 Halpha transfer folds are incomplete")
    transfer_base = np.asarray(
        [float(row["log_baryonic_speed_km_s"]) for row in transfer_rows], dtype=np.float64
    )
    transfer_delta = delta[:, xp.asarray(transfer_indices)]
    transferred = _transfer_selected_candidates(
        transfer_delta,
        transfer_target,
        transfer_base,
        transfer_folds,
        observed["selected_indices"],
        config,
        xp,
    )
    transfer_baselines = _baseline_predictions(
        transfer_target,
        transfer_base,
        transfer_folds,
        transfer_rows,
        config,
        coherence_spec,
    )
    transfer_candidate_mse = _mse(transfer_target, transferred["prediction"])
    transfer_baseline_mse = {
        key: _mse(transfer_target, prediction) for key, prediction in transfer_baselines.items()
    }
    transfer_improvement_vs_coherence = _improvement(
        transfer_baseline_mse["ordinary_coherence"], transfer_candidate_mse
    )

    random = np.random.Generator(
        np.random.PCG64(int(config["evaluation"]["permutation_seed"]))
    )
    trials = int(config["evaluation"]["permutation_trials"])
    null_improvements = []
    for _ in range(trials):
        null_target = _cell_residual_permutation(
            target, baselines["ordinary_coherence"], rows, random
        )
        null_selected = _screen_candidate_matrix(delta, null_target, base, folds, config, xp)
        null_baselines = _baseline_predictions(
            null_target, base, folds, rows, config, coherence_spec
        )
        null_improvements.append(
            _improvement(
                _mse(null_target, null_baselines["ordinary_coherence"]),
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
    selected_niches = [record["niche_index"] for record in selected_records]
    niche_counts = Counter(selected_niches)
    ontology_counts = Counter(record["ontology"] for record in selected_records)

    mass = np.asarray([float(row["stellar_mass_msun"]) for row in rows])
    surface = np.asarray([float(row["log_surface_density"]) for row in rows])
    acceleration = np.asarray([float(row["internal_acceleration_m_s2"]) for row in rows])
    age = np.asarray([float(row["dn4000"]) for row in rows])
    phase_space = np.asarray([float(row["phase_space_proxy"]) for row in rows])
    slice_masks = {
        "low_mass": mass <= np.median(mass),
        "high_mass": mass > np.median(mass),
        "low_surface_density": surface <= np.median(surface),
        "high_surface_density": surface > np.median(surface),
        "low_acceleration": acceleration <= np.median(acceleration),
        "high_acceleration": acceleration > np.median(acceleration),
        "younger_spectral_clock": age <= np.median(age),
        "older_spectral_clock": age > np.median(age),
        "low_phase_space_proxy": phase_space <= np.median(phase_space),
        "high_phase_space_proxy": phase_space > np.median(phase_space),
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
            value[f"improvement_vs_{baseline_name}"] = _improvement(
                mse, value["candidate_mse"]
            )
        slices[label] = value

    stellar_robustness = _robust_comparison(
        target,
        observed["prediction"],
        baselines["ordinary_coherence"],
        rows,
        config,
    )
    transfer_robustness = _robust_comparison(
        transfer_target,
        transferred["prediction"],
        transfer_baselines["ordinary_coherence"],
        transfer_rows,
        config,
    )
    object_counterexamples = int(stellar_robustness["counterexamples"])
    transfer_counterexamples = int(transfer_robustness["counterexamples"])
    robustness_sign_consistent = not any(
        (
            stellar_robustness["leave_one_changes_improvement_sign"],
            stellar_robustness["trim_changes_improvement_sign"],
            transfer_robustness["leave_one_changes_improvement_sign"],
            transfer_robustness["trim_changes_improvement_sign"],
        )
    )
    stellar_fraction = len(rows) / int(config["sample"]["expected_exploration"])
    stellar_quality_pass = len(rows) >= int(config["sample"]["minimum_complete_stellar_exploration"])
    stellar_quality_pass = stellar_quality_pass and stellar_fraction >= float(
        config["sample"]["minimum_stellar_retention_fraction"]
    )
    transfer_quality_pass = len(transfer_rows) >= int(
        config["sample"]["minimum_complete_Halpha_transfer"]
    )

    gates = config["gates"]
    stable_niche = max(niche_counts.values()) >= int(gates["minimum_same_niche_folds"])
    selected_ontology_eligible = all(record["gravity_track_eligible"] for record in selected_records)
    cpu_gpu_pass = cpu_gpu_max <= float(config["evaluation"]["cpu_gpu_tolerance"])
    controls_pass = (
        bool(controls["all_injected_niches_recovered"])
        and bool(controls["all_injected_niches_transfer_unchanged"])
        and not bool(controls["GR_control_candidate_improves"])
        and cpu_gpu_pass
    )
    universal_gates = {
        "stellar_response_quality": stellar_quality_pass,
        "Halpha_transfer_quality": transfer_quality_pass,
        "confirmation_values_read_zero": int(gates["confirmation_values_read"]) == 0,
        "post_response_cells_zero": int(gates["post_response_candidate_cells"]) == 0,
        "improvement_vs_baryonic_virial": _improvement(
            baseline_mse["baryonic_virial"], candidate_mse
        )
        >= float(gates["minimum_improvement_vs_baryonic_virial"]),
        "improvement_vs_structural": _improvement(
            baseline_mse["structural_ridge"], candidate_mse
        )
        >= float(gates["minimum_improvement_vs_structural"]),
        "improvement_vs_flexible": _improvement(
            baseline_mse["flexible_nuisance"], candidate_mse
        )
        >= float(gates["minimum_improvement_vs_flexible"]),
        "improvement_vs_ordinary_coherence": observed_statistic
        >= float(gates["minimum_improvement_vs_ordinary_coherence"]),
        "each_broad_half_improves_baryonic_virial": all(
            value["improvement_vs_baryonic_virial"]
            >= float(gates["minimum_each_broad_half_improvement_vs_baryonic_virial"])
            for value in slices.values()
        ),
        "Halpha_transfer_improves_ordinary_coherence": transfer_improvement_vs_coherence
        >= float(gates["minimum_Halpha_transfer_improvement_vs_ordinary_coherence"]),
        "selection_aware_permutation": p_value
        <= float(gates["maximum_selection_aware_permutation_p"]),
        "stable_niche": stable_niche,
        "robustness_sign_consistent": robustness_sign_consistent,
        "selected_ontology_is_gravity_eligible": selected_ontology_eligible,
        "all_injected_niches_recovered": bool(controls["all_injected_niches_recovered"]),
        "known_GR_control": not bool(controls["GR_control_candidate_improves"]),
        "cpu_gpu_agreement": cpu_gpu_pass,
        "local_limit": candidate_audit["maximum_admitted_local_fractional_response"]
        <= float(config["admissibility"]["maximum_local_fractional_response"]),
        "positive_bounded_stability_proxy": candidate_audit["minimum_admitted_mu"]
        >= float(config["admissibility"]["minimum_mu"])
        and candidate_audit["maximum_admitted_mu"]
        <= float(config["admissibility"]["maximum_mu"]),
    }
    phenomenon_gates = {
        "stellar_response_quality": stellar_quality_pass,
        "Halpha_transfer_quality": transfer_quality_pass,
        "improvement_vs_ordinary_coherence": observed_statistic
        >= float(gates["phenomenon_minimum_improvement_vs_ordinary_coherence"]),
        "Halpha_transfer_improvement": transfer_improvement_vs_coherence
        >= float(gates["phenomenon_minimum_Halpha_transfer_improvement"]),
        "selection_aware_permutation": p_value
        <= float(gates["phenomenon_maximum_selection_aware_p"]),
        "stable_niche": stable_niche,
        "robustness_sign_consistent": robustness_sign_consistent,
        "controls": controls_pass,
    }
    partial_slices = [
        label
        for label, value in slices.items()
        if value["improvement_vs_ordinary_coherence"]
        >= float(gates["partial_minimum_slice_improvement_vs_ordinary_coherence"])
    ]
    universal_pass = all(universal_gates.values())
    phenomenon_pass = all(phenomenon_gates.values())
    if not stellar_quality_pass or not transfer_quality_pass:
        decision = "INCONCLUSIVE_ITEM34_TWO_TRACER_QUALITY"
    elif universal_pass:
        decision = "PASS_ITEM34_EXPLORATION_UNIVERSAL"
    elif phenomenon_pass:
        decision = "PASS_ITEM34_EXPLORATION_PHENOMENON_LEAD_PENDING_REPLICATION"
    elif partial_slices:
        decision = "BOTH_FORMAL_TRACKS_NOT_PROMOTED_SCOPED_PARTIAL_PATTERN_RETAINED"
    elif not robustness_sign_consistent:
        decision = "BOTH_FORMAL_TRACKS_NOT_PROMOTED_ROBUSTNESS_SENSITIVE_PATTERN_RETAINED"
    else:
        decision = "SCOPED_ITEM34_REJECT"

    scientific = {
        "decision": decision,
        "quality": {
            "stellar_complete_exploration_objects": len(rows),
            "stellar_minimum_required": int(
                config["sample"]["minimum_complete_stellar_exploration"]
            ),
            "stellar_retention_fraction": stellar_fraction,
            "stellar_pass": stellar_quality_pass,
            "Halpha_transfer_objects": len(transfer_rows),
            "Halpha_minimum_required": int(config["sample"]["minimum_complete_Halpha_transfer"]),
            "Halpha_pass": transfer_quality_pass,
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
        "stellar_metrics": {
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
            "improvement_vs_ordinary_coherence": observed_statistic,
            "selection_aware_permutation_p": p_value,
            "maximum_null_improvement": max(null_improvements),
            "object_counterexamples_vs_ordinary_coherence": object_counterexamples,
            "counterexample_robustness": stellar_robustness,
        },
        "Halpha_transfer_metrics": {
            "formula_reselection": False,
            "candidate_mse": transfer_candidate_mse,
            "baseline_mse": transfer_baseline_mse,
            "improvement_vs_ordinary_coherence": transfer_improvement_vs_coherence,
            "object_counterexamples_vs_ordinary_coherence": transfer_counterexamples,
            "counterexample_robustness": transfer_robustness,
            "selected_indices_from_stellar": transferred["selected_indices_from_stellar"],
            "log10_speed_offsets": transferred["log10_speed_offsets"],
        },
        "permutation": {
            "strategy": config["evaluation"]["permutation_strategy"],
            "trials": trials,
            "null_improvements": null_improvements,
        },
        "ordinary_coherence_control_spec": coherence_spec,
        "broad_slices": slices,
        "selected_candidates": selected_records,
        "selected_niche_counts": {str(key): niche_counts[key] for key in range(4)},
        "selected_ontology_counts": dict(sorted(ontology_counts.items())),
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
            "stellar_counterexamples_vs_ordinary_coherence": object_counterexamples,
            "Halpha_counterexamples_vs_ordinary_coherence": transfer_counterexamples,
            "negative_or_partial_families_are_retained": True,
        },
    }
    training_per_search = int(
        len(arrays["niche"])
        * sum(np.count_nonzero(folds != fold) for fold in range(outer_folds))
    )
    compute = {
        "backend": backend,
        "device": device,
        "candidate_matrix_seconds": matrix_seconds,
        "candidate_cells": len(arrays["niche"]),
        "candidate_observable_matrix_values": int(np.prod(delta.shape)),
        "candidate_training_residual_evaluations_observed": training_per_search,
        "candidate_training_residual_evaluations_with_nulls": training_per_search
        * (trials + 1),
        "Halpha_transfer_candidate_values": int(len(arrays["niche"]) * len(transfer_rows)),
        "cpu_crosscheck_candidates": crosscheck,
        "cpu_gpu_max_abs_difference": cpu_gpu_max,
        "cpu_gpu_tolerance": float(config["evaluation"]["cpu_gpu_tolerance"]),
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
    test_path = root / "tests/test_gravity_item34_condensate_superfluid.py"
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item34-condensate-superfluid-result-1.0",
            "item": 34,
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
                "hidden_matter_is_not_relabelled_as_gravity": True,
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
                "complete_stellar_response_objects": len(rows),
                "confirmation_response_values_read": 0,
                "post_response_formula_generation": False,
                "Halpha_formula_reselection": False,
                "paid_api_calls": 0,
            },
            "source_bindings": {
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "test_sha256": _sha256_file(test_path) if test_path.exists() else None,
            },
            "claim_boundary": [
                config["scope"]["claim_ceiling"],
                "The superfluid and soliton controls require hidden stress-energy and cannot count as no-dark-matter gravity even if they fit.",
                "The baryon-sourced coherent and locked-phase projections are bounded integrated kernels, not action-derived covariant field theories.",
                "No condensate density, boson mass, coherence map, vortex, interference fringe, or phase occupation was directly measured.",
                "The matched ordinary phase-space/coherence hinge model is the primary phenomenon control; beating weaker baryonic or structural baselines is insufficient.",
                "The Halpha transfer reuses the stellar-selected formula without formula or parameter-cell reselection and fits only a training-fold speed offset.",
                "Stellar populations, mass-to-light ratios, anisotropy, inclination, gas flows, and integrated-aperture projection can mimic coherence-like correlations.",
                "This experiment does not establish resolved rotation curves, gravitational slip, direct lensing, clusters, conservation, stability, cosmology, or an alternative to GR.",
                "A positive result is exploration evidence only; confirmations stay sealed and unchanged fresh replication is mandatory for a paper claim.",
            ],
        }
    )


def run_experiment(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest, extraction = _load_response_rows(root, config)
    sample = _read_json(_source_paths(root, config)["sample_manifest"])
    coherence_spec = sample["ordinary_coherence_control_spec"]
    scientific, compute = _evaluate(config, rows, coherence_spec)
    paths = _source_paths(root, config)
    compute_manifest = _content_hashed(
        {"schema_version": "invariant-gravity-item34-compute-1.0", **compute}
    )
    _write_json(paths["compute_manifest"], compute_manifest)
    receipt = _build_receipt(
        root, config, rows, response_manifest, extraction, scientific, compute
    )
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
        raise GravityItem34Error("Item 34 predictor freeze contains response values")
    if int(sample["counts"]["reserved_confirmation"]) != int(
        config["sample"]["expected_confirmation"]
    ):
        raise GravityItem34Error("Item 34 confirmation allocation changed")
    if int(sample["counts"]["exploration"]) != int(config["sample"]["expected_exploration"]):
        raise GravityItem34Error("Item 34 exploration allocation changed")
    if int(candidates["post_response_cells"]) != 0:
        raise GravityItem34Error("Item 34 candidate manifest contains post-response cells")
    if int(response["counts"]["confirmation_values_read"]) != 0:
        raise GravityItem34Error("Item 34 response acquisition opened confirmations")
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "Item 34 result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem34Error("checked Item 34 result opened confirmation responses")
    if bool(result["frozen_boundary"]["post_response_formula_generation"]):
        raise GravityItem34Error("checked Item 34 result contains post-response generation")
    if bool(result["frozen_boundary"]["Halpha_formula_reselection"]):
        raise GravityItem34Error("checked Item 34 result reselected on Halpha")
    if _sha256_file(paths["exploration_responses"]) != result["frozen_boundary"][
        "response_file_sha256"
    ]:
        raise GravityItem34Error("checked Item 34 response file changed")
    if result["source_bindings"]["config_sha256"] != _sha256_file(root / CONFIG_PATH):
        raise GravityItem34Error("checked Item 34 config changed")
    if result["source_bindings"]["module_sha256"] != _sha256_file(root / MODULE_PATH):
        raise GravityItem34Error("checked Item 34 module changed")
    test_path = root / "tests/test_gravity_item34_condensate_superfluid.py"
    if result["source_bindings"]["test_sha256"] != _sha256_file(test_path):
        raise GravityItem34Error("checked Item 34 test changed")


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
