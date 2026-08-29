"""Frozen Item 29 nonlinear gravitational self-interaction experiment.

Only response-independent SL2S identities, photometry, geometry, and stellar-mass
proxies may be used to freeze candidates and sample roles.  Velocity dispersions
and Einstein radii are acquired for exploration roles only after both freezes.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    _angular_diameter_distances,
    _parse_vizier_tsv,
    hernquist_projected_mass_fraction,
)
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

CONFIG_PATH = Path("configs/gravity_item29_nonlinear_self_interaction_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item29_nonlinear_self_interaction.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
_ADMISSIBLE_CACHE: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


class GravityItem29Error(RuntimeError):
    """Raised when an Item 29 freeze, leakage, or replay invariant is violated."""


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item29/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not body:
        raise GravityItem29Error(f"empty source response: {url}")
    return body, headers


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item29-nonlinear-self-interaction-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 29:
        raise GravityItem29Error("unexpected Item 29 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem29Error("stable gravity goal changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem29Error("raw candidate boundary changed")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem29Error("post-response candidates entered Item 29")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem29Error("equal-viability policy changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem29Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem29Error("paid calls are outside Item 29")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem29Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem29Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem29Error("could not read frozen Item 29 module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem29Error("Item 29 module differs from scientific freeze")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "identity_catalog",
        "photometry_catalog",
        "stellar_mass_catalog",
        "predictors",
        "predecessor_audit",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "exploration_responses",
        "response_source_manifest",
        "compute_manifest",
    )
    return {key: base / str(config["paths"][key]) for key in keys}


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in (
        "identity_catalog",
        "photometry_catalog",
        "stellar_mass_catalog",
        "predictors",
        "predecessor_audit",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem29Error(f"{key} differs from sample freeze")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()


def _canonical_identity(value: str) -> str:
    answer = "".join(char for char in str(value).upper() if char.isalnum() or char in "+-")
    for prefix in ("SDSS", "SL2S", "HSC"):
        if answer.startswith(prefix):
            answer = answer[len(prefix) :]
            break
    return answer.removeprefix("J")


def _short_sky_key(value: str) -> str:
    canonical = _canonical_identity(value)
    for sign in ("+", "-"):
        if sign not in canonical:
            continue
        ra, dec = canonical.split(sign, 1)
        if len(ra) >= 4 and len(dec) >= 4 and ra[:4].isdigit() and dec[:4].isdigit():
            return f"{ra[:4]}{sign}{dec[:4]}"
    return canonical


def _coordinates_from_name(value: str) -> tuple[float, float]:
    canonical = _canonical_identity(value)
    sign = "+" if "+" in canonical else "-" if "-" in canonical else ""
    if not sign:
        raise GravityItem29Error(f"identity has no coordinate sign: {value}")
    ra_text, dec_text = canonical.split(sign, 1)
    if len(ra_text) < 6 or not ra_text[:6].isdigit() or not dec_text.isdigit():
        raise GravityItem29Error(f"identity cannot be parsed as coordinates: {value}")
    dec_text = dec_text.rjust(6, "0")
    if len(dec_text) < 6:
        raise GravityItem29Error(f"declination cannot be parsed: {value}")
    ra = 15.0 * (
        int(ra_text[:2]) + int(ra_text[2:4]) / 60.0 + int(ra_text[4:6]) / 3600.0
    )
    dec = int(dec_text[:2]) + int(dec_text[2:4]) / 60.0 + int(dec_text[4:6]) / 3600.0
    return ra, dec if sign == "+" else -dec


def _angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = math.radians(ra2 - ra1)
    de1, de2 = math.radians(dec1), math.radians(dec2)
    cosine = math.sin(de1) * math.sin(de2) + math.cos(de1) * math.cos(de2) * math.cos(dra)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 3600.0


def _manifest_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        objects = value.get("objects")
        if isinstance(objects, list):
            for row in objects:
                if isinstance(row, Mapping):
                    yield row
        for key, child in value.items():
            if key != "objects":
                yield from _manifest_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _manifest_objects(child)


def _predecessor_entries(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative in config["sources"]["predecessor_sample_manifests"]:
        manifest = _read_json(root / str(relative))
        for row in _manifest_objects(manifest):
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            ra = row.get("catalog_ra_deg", row.get("ra_deg"))
            dec = row.get("catalog_dec_deg", row.get("dec_deg"))
            try:
                coordinates = (float(ra), float(dec))
            except (TypeError, ValueError):
                try:
                    coordinates = _coordinates_from_name(name)
                except GravityItem29Error:
                    coordinates = (None, None)
            entries.append(
                {
                    "manifest": str(relative),
                    "name": name,
                    "canonical_identity": _canonical_identity(name),
                    "short_sky_key": _short_sky_key(name),
                    "ra_deg": coordinates[0],
                    "dec_deg": coordinates[1],
                }
            )
    return entries


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["raw_candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    parameter_radices = {
        "amplitude": len(generator["amplitudes"]),
        "acceleration_scale": len(generator["acceleration_scales_m_s2"]),
        "acceleration_power": len(generator["acceleration_powers"]),
        "radius_scale": len(generator["radius_scales_re"]),
        "radius_power": len(generator["radius_powers"]),
        "feedback_eta": len(generator["feedback_eta"]),
    }
    structural_count = int(np.prod(list(parameter_radices.values())))
    structures_per_polarity = count // 8
    pieces: dict[str, list[np.ndarray]] = {
        "niche": [],
        "polarity": [],
        **{key: [] for key in parameter_radices},
    }
    for niche_index in range(4):
        selected = random.permutation(structural_count)[:structures_per_polarity]
        ordinals = np.tile(selected, 2)
        pieces["niche"].append(
            np.full(2 * structures_per_polarity, niche_index, dtype=np.int16)
        )
        pieces["polarity"].append(
            np.repeat(np.arange(2, dtype=np.int16), structures_per_polarity)
        )
        working = ordinals.copy()
        decoded: dict[str, np.ndarray] = {}
        for key, radix in reversed(list(parameter_radices.items())):
            decoded[key] = (working % radix).astype(np.int16)
            working //= radix
        if np.any(working != 0):
            raise GravityItem29Error("mixed-radix candidate decoder overflow")
        for key in parameter_radices:
            pieces[key].append(decoded[key])
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    permutation = random.permutation(count)
    return {key: value[permutation] for key, value in arrays.items()}


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    return {
        "niche": xp.asarray(arrays["niche"][begin:end]),
        "polarity": xp.asarray(
            np.asarray(generator["polarities"])[arrays["polarity"][begin:end]]
        ),
        "amplitude": xp.asarray(
            np.asarray(generator["amplitudes"])[arrays["amplitude"][begin:end]]
        ),
        "acceleration_scale": xp.asarray(
            np.asarray(generator["acceleration_scales_m_s2"])[
                arrays["acceleration_scale"][begin:end]
            ]
        ),
        "acceleration_power": xp.asarray(
            np.asarray(generator["acceleration_powers"])[
                arrays["acceleration_power"][begin:end]
            ]
        ),
        "radius_scale": xp.asarray(
            np.asarray(generator["radius_scales_re"])[arrays["radius_scale"][begin:end]]
        ),
        "radius_power": xp.asarray(
            np.asarray(generator["radius_powers"])[arrays["radius_power"][begin:end]]
        ),
        "feedback_eta": xp.asarray(
            np.asarray(generator["feedback_eta"])[arrays["feedback_eta"][begin:end]]
        ),
    }


def _candidate_log_response(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    accelerations: Any,
    radius_ratios: Any,
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    values = _candidate_values(config, arrays, begin, end, xp)
    shape = (-1,) + (1,) * xp.asarray(accelerations).ndim
    niche = values["niche"].reshape(shape)
    polarity = values["polarity"].reshape(shape)
    amplitude = values["amplitude"].reshape(shape)
    scale = values["acceleration_scale"].reshape(shape)
    power = values["acceleration_power"].reshape(shape)
    radial_scale = values["radius_scale"].reshape(shape)
    radial_power = values["radius_power"].reshape(shape)
    eta = values["feedback_eta"].reshape(shape)
    acceleration = xp.maximum(xp.asarray(accelerations)[None, ...], 1e-300)
    radius = xp.maximum(xp.asarray(radius_ratios)[None, ...], 1e-12)
    geometry = 1.0 / (1.0 + xp.power(radial_scale / radius, radial_power))
    z = xp.minimum(xp.power(scale / acceleration, power) * geometry, 1e100)
    q = z / (1.0 + z)

    kinetic_driver = (q + eta * q * q) / (1.0 + eta)
    kinetic_denominator = 1.0 - polarity * amplitude * kinetic_driver
    kinetic = xp.where(
        kinetic_denominator > 0.0,
        -xp.log(xp.maximum(kinetic_denominator, 1e-300)),
        xp.nan,
    )

    source_driver = (q + eta * q * q * q) / (1.0 + eta)
    coupling = 0.24 * polarity * amplitude * source_driver
    discriminant = 1.0 - 4.0 * coupling
    field_mu = xp.where(
        discriminant > 0.0,
        2.0 / (1.0 + xp.sqrt(xp.maximum(discriminant, 0.0))),
        xp.nan,
    )
    field_energy = xp.log(field_mu)

    hessian_z = z * (1.0 + eta * geometry)
    branch = 2.0 / (1.0 + xp.sqrt(1.0 + 4.0 * hessian_z))
    hessian = polarity * amplitude * (1.0 - branch)

    feedback = xp.zeros_like(z)
    for _ in range(int(config["candidate_generator"]["feedback_iterations"])):
        feedback = xp.tanh(z + eta * feedback)
    bounded = polarity * amplitude * feedback

    return xp.where(
        niche == 0,
        kinetic,
        xp.where(niche == 1, field_energy, xp.where(niche == 2, hessian, bounded)),
    )


def _admissible_candidates(
    config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_key = _sha256_bytes(
        _canonical_bytes(
            {
                "candidate_generator": config["candidate_generator"],
                "admissibility": config["admissibility"],
                "evaluation_batch": config["evaluation"]["candidate_batch_size"],
                "local_gravity": config["physics"]["constants"][
                    "solar_gravity_at_1au_m_s2"
                ],
            }
        )
    )
    if cache_key in _ADMISSIBLE_CACHE:
        return _ADMISSIBLE_CACHE[cache_key]
    raw = generate_raw_candidates(config)
    admissibility = config["admissibility"]
    g_grid = np.asarray(admissibility["field_acceleration_grid_m_s2"], dtype=np.float64)
    r_grid = np.asarray(admissibility["radius_ratio_grid"], dtype=np.float64)
    accelerations = np.tile(g_grid, len(r_grid))
    radii = np.repeat(r_grid, len(g_grid))
    keep = np.zeros(len(raw["niche"]), dtype=bool)
    local_maximum = 0.0
    batch = int(config["evaluation"]["candidate_batch_size"])
    log_g_step = np.diff(np.log(g_grid))[None, None, :]
    minimum_slope = float(admissibility["minimum_effective_acceleration_log_slope"])
    for begin in range(0, len(keep), batch):
        end = min(begin + batch, len(keep))
        response = _candidate_log_response(
            config, raw, accelerations, radii, begin, end, np
        ).reshape(end - begin, len(r_grid), len(g_grid))
        finite = np.all(np.isfinite(response), axis=(1, 2))
        bounded = np.max(np.abs(response), axis=(1, 2)) <= float(
            admissibility["maximum_absolute_log_response"]
        )
        effective = response + np.log(g_grid)[None, None, :]
        monotone = np.all(np.diff(effective, axis=2) >= minimum_slope * log_g_step, axis=(1, 2))
        material = np.max(np.abs(np.expm1(response)), axis=(1, 2)) >= float(
            admissibility["minimum_material_fractional_response"]
        )
        local = _candidate_log_response(
            config,
            raw,
            np.asarray([config["physics"]["constants"]["solar_gravity_at_1au_m_s2"]]),
            np.asarray([admissibility["local_radius_ratio_worst_case"]]),
            begin,
            end,
            np,
        ).reshape(-1)
        local_fraction = np.abs(np.expm1(local))
        local_maximum = max(local_maximum, float(np.nanmax(local_fraction)))
        local_pass = np.isfinite(local_fraction) & (
            local_fraction <= float(admissibility["maximum_local_fractional_response_at_1au"])
        )
        keep[begin:end] = finite & bounded & monotone & material & local_pass
    arrays = {key: value[keep] for key, value in raw.items()}
    raw_counts = Counter(int(value) for value in raw["niche"])
    admitted_counts = Counter(int(value) for value in arrays["niche"])
    matrix = np.column_stack([arrays[key] for key in sorted(arrays)])
    audit = {
        "raw_candidates": len(raw["niche"]),
        "raw_per_niche": {str(key): raw_counts[key] for key in range(4)},
        "admissible_candidates": len(arrays["niche"]),
        "admissible_per_niche": {str(key): admitted_counts[key] for key in range(4)},
        "raw_candidate_digest": _candidate_digest(raw),
        "admissible_candidate_digest": _candidate_digest(arrays),
        "exact_parameter_signatures": len(np.unique(matrix, axis=0)),
        "maximum_raw_local_fractional_response": local_maximum,
        "maximum_admitted_local_fractional_response": float(
            np.max(
                np.abs(
                    np.expm1(
                        _candidate_log_response(
                            config,
                            arrays,
                            np.asarray(
                                [config["physics"]["constants"][
                                    "solar_gravity_at_1au_m_s2"
                                ]]
                            ),
                            np.asarray([admissibility["local_radius_ratio_worst_case"]]),
                            0,
                            len(arrays["niche"]),
                            np,
                        )
                    )
                )
            )
        ),
        "filters_are_response_independent": True,
        "real_positive_monotone_branch_required": True,
    }
    generator = config["candidate_generator"]
    expected = (
        (audit["raw_candidate_digest"], generator["expected_raw_candidate_digest"], "raw digest"),
        (
            audit["admissible_candidate_digest"],
            generator["expected_admissible_candidate_digest"],
            "admissible digest",
        ),
        (
            audit["admissible_candidates"],
            int(generator["expected_admissible_candidates"]),
            "admissible count",
        ),
        (
            audit["admissible_per_niche"],
            generator["expected_admissible_per_niche"],
            "admissible niche counts",
        ),
    )
    for actual, wanted, label in expected:
        if wanted in ("TO_BE_FROZEN", -1, {}):
            continue
        if actual != wanted:
            raise GravityItem29Error(f"{label} changed: {actual} != {wanted}")
    _ADMISSIBLE_CACHE[cache_key] = (arrays, audit)
    return arrays, audit


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    arrays, audit = _admissible_candidates(config)
    injections = [
        int(value)
        for value in config["candidate_generator"]["synthetic_injection_admissible_indices"]
    ]
    if any(index < 0 or index >= len(arrays["niche"]) for index in injections):
        raise GravityItem29Error("synthetic injection index is outside admissible candidates")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item29-nonlinear-candidates-1.0",
            "response_values_read": 0,
            "post_response_candidate_cells": 0,
            "candidate_generator": config["candidate_generator"],
            "admissibility": config["admissibility"],
            "audit": audit,
            "synthetic_injection_admissible_indices": injections,
            "synthetic_injection_niches": [int(arrays["niche"][index]) for index in injections],
            "creativity_and_provenance_labels": config["candidate_generator"]["niches"],
        }
    )


def _build_sample(
    predictors: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    sample = config["sample"]
    selected = sorted(
        (dict(row) for row in predictors),
        key=lambda row: (float(row["log10_stellar_mass_msun"]), str(row["name"])),
    )
    if len(selected) != int(sample["expected_selected"]):
        raise GravityItem29Error("predictor-selected sample size changed")
    strata = int(sample["mass_strata"])
    for index, row in enumerate(selected):
        row["mass_stratum"] = min(strata - 1, index * strata // len(selected))
    for stratum in range(strata):
        group = [row for row in selected if int(row["mass_stratum"]) == stratum]
        ranked = sorted(
            group, key=lambda row: _hmac_rank(str(sample["role_key"]), str(row["name"]))
        )
        held = {
            str(row["name"])
            for row in ranked[: int(sample["confirmation_per_stratum"])]
        }
        for row in group:
            row["role"] = "confirmation" if str(row["name"]) in held else "exploration"
    exploration = [row for row in selected if row["role"] == "exploration"]
    exploration.sort(key=lambda row: _hmac_rank(str(sample["fold_key"]), str(row["name"])))
    for index, row in enumerate(exploration):
        row["fold"] = index % int(sample["outer_folds"])
    for row in selected:
        if row["role"] == "confirmation":
            row["fold"] = None
    if len(exploration) != int(sample["expected_exploration"]):
        raise GravityItem29Error("exploration role count changed")
    if len(selected) - len(exploration) != int(sample["expected_confirmation"]):
        raise GravityItem29Error("confirmation role count changed")
    selected.sort(key=lambda row: str(row["name"]))
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item29-nonlinear-sample-1.0",
            "selection_rule": sample["rule"],
            "selection_used_response_values": False,
            "response_values_read": 0,
            "confirmation_opened": False,
            "counts": {
                "selected": len(selected),
                "exploration": len(exploration),
                "confirmation": len(selected) - len(exploration),
            },
            "fold_counts": dict(
                sorted(Counter(str(row["fold"]) for row in exploration).items())
            ),
            "mass_stratum_counts": dict(
                sorted(Counter(str(row["mass_stratum"]) for row in selected).items())
            ),
            "objects": selected,
        }
    )


def _float_required(row: Mapping[str, Any], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(key)
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError(key)
    return answer


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    sources = config["sources"]
    downloads: dict[str, tuple[bytes, dict[str, str]]] = {}
    for key in ("identity", "photometry", "stellar_mass"):
        downloads[key] = _download(str(sources[f"{key}_query"]))
    identity_body, _ = downloads["identity"]
    photometry_body, _ = downloads["photometry"]
    stellar_mass_body, _ = downloads["stellar_mass"]
    paths["identity_catalog"].parent.mkdir(parents=True, exist_ok=True)
    paths["identity_catalog"].write_bytes(identity_body)
    paths["photometry_catalog"].write_bytes(photometry_body)
    paths["stellar_mass_catalog"].write_bytes(stellar_mass_body)
    identities = _parse_vizier_tsv(identity_body, tuple(sources["identity_columns"]))
    photometry = _parse_vizier_tsv(photometry_body, tuple(sources["photometry_columns"]))
    stellar_mass = _parse_vizier_tsv(stellar_mass_body, tuple(sources["stellar_mass_columns"]))
    expected_counts = (
        (len(identities), int(sources["expected_identity_rows"]), "identity rows"),
        (len(photometry), int(sources["expected_photometry_rows"]), "photometry rows"),
        (len(stellar_mass), int(sources["expected_stellar_mass_rows"]), "stellar-mass rows"),
    )
    for actual, expected, label in expected_counts:
        if actual != expected:
            raise GravityItem29Error(f"{label} changed: {actual} != {expected}")
    survey_rows = [row for row in identities if str(row["Survey"]).strip() == sources["survey"]]
    if len(survey_rows) != int(sources["expected_survey_rows"]):
        raise GravityItem29Error("SL2S identity count changed")
    photometry_by_name = {
        _canonical_identity(str(row["SL2S"])): row for row in photometry
    }
    mass_by_name = {_canonical_identity(str(row["SL2S"])): row for row in stellar_mass}
    predecessor = _predecessor_entries(root, config)
    predecessor_keys = {str(row["short_sky_key"]) for row in predecessor}
    incomplete: list[dict[str, Any]] = []
    name_exclusions: list[dict[str, Any]] = []
    coordinate_exclusions: list[dict[str, Any]] = []
    predictors: list[dict[str, Any]] = []
    for identity in survey_rows:
        name = str(identity["Name"]).strip()
        canonical = _canonical_identity(name)
        phot = photometry_by_name.get(canonical)
        mass = mass_by_name.get(canonical)
        try:
            if phot is None or mass is None:
                raise ValueError("join")
            reff_arcsec = _float_required(phot, "Reff")
            axis_ratio = _float_required(phot, "q")
            gmag = _float_required(phot, "gmag")
            rmag = _float_required(phot, "rmag")
            imag = _float_required(phot, "imag")
            zmag = _float_required(phot, "zmag")
            position_angle = _float_required(phot, "PA")
            logmass = _float_required(mass, "logMC")
            mass_error = _float_required(mass, "e_logMC")
            z_lens = _float_required(identity, "zl")
            z_source = _float_required(identity, "zs")
            if not (reff_arcsec > 0 and 0 < axis_ratio <= 1 and z_source > z_lens > 0):
                raise ValueError("domain")
            ra, dec = _coordinates_from_name(name)
        except (GravityItem29Error, ValueError) as error:
            incomplete.append({"name": name, "reason": str(error)})
            continue
        short_key = _short_sky_key(name)
        matching_names = [row for row in predecessor if row["short_sky_key"] == short_key]
        if short_key in predecessor_keys:
            name_exclusions.append(
                {"name": name, "short_sky_key": short_key, "matches": matching_names}
            )
            continue
        coordinate_matches = []
        for row in predecessor:
            if row["ra_deg"] is None or row["dec_deg"] is None:
                continue
            separation = _angular_separation_arcsec(
                ra, dec, float(row["ra_deg"]), float(row["dec_deg"])
            )
            if separation <= float(sources["coordinate_veto_arcsec"]):
                coordinate_matches.append({**row, "separation_arcsec": separation})
        if coordinate_matches:
            coordinate_exclusions.append({"name": name, "matches": coordinate_matches})
            continue
        d_lens, _, _ = _angular_diameter_distances(z_lens, z_source, config)
        reff_kpc = (
            reff_arcsec * float(config["physics"]["constants"]["arcsec_to_radian"]) * d_lens
        )
        stellar_mass = 10.0**logmass
        predictors.append(
            {
                "name": name,
                "canonical_identity": canonical,
                "short_sky_key": short_key,
                "survey": "SL2S",
                "ra_deg": ra,
                "dec_deg": dec,
                "z_lens": z_lens,
                "z_source": z_source,
                "reff_arcsec": reff_arcsec,
                "reff_kpc": reff_kpc,
                "axis_ratio": axis_ratio,
                "position_angle_deg": position_angle,
                "g_minus_i": gmag - imag,
                "r_minus_z": rmag - zmag,
                "imag_ab": imag,
                "log10_stellar_mass_msun": logmass,
                "log10_stellar_mass_error_dex": mass_error,
                "stellar_mass_proxy_msun": stellar_mass,
                "stellar_surface_density_proxy_msun_kpc2": stellar_mass
                / (2.0 * math.pi * reff_kpc**2),
            }
        )
    checks = (
        (len(incomplete), int(sources["expected_incomplete_predictor_rows"]), "incomplete"),
        (len(name_exclusions), int(sources["expected_predecessor_name_exclusions"]), "name exclusions"),
        (
            len(coordinate_exclusions),
            int(sources["expected_predecessor_coordinate_exclusions_after_name"]),
            "coordinate exclusions",
        ),
        (len(predictors), int(sources["expected_predictor_eligible"]), "eligible predictors"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise GravityItem29Error(f"{label} changed: {actual} != {expected}")
    predictors.sort(key=lambda row: str(row["name"]))
    _write_tsv(paths["predictors"], predictors, list(predictors[0]))
    predecessor_audit = _content_hashed(
        {
            "schema_version": "invariant-gravity-item29-predecessor-audit-1.0",
            "response_values_read": 0,
            "canonical_rule": "survey prefixes and leading J removed; HHMM+/-DDMM short sky key plus a three-arcsecond coordinate veto",
            "predecessor_manifests": sources["predecessor_sample_manifests"],
            "predecessor_entries": len(predecessor),
            "incomplete_predictors": sorted(incomplete, key=lambda row: row["name"]),
            "name_exclusions": sorted(name_exclusions, key=lambda row: row["name"]),
            "coordinate_exclusions": sorted(
                coordinate_exclusions, key=lambda row: row["name"]
            ),
            "eligible_names": [row["name"] for row in predictors],
        }
    )
    _write_json(paths["predecessor_audit"], predecessor_audit)
    sample_manifest = _build_sample(predictors, config)
    candidate_manifest = _candidate_manifest(config)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item29-predictor-source-1.0",
            "response_values_read": 0,
            "selection_used_response_values": False,
            "sources": {
                key: {
                    "url": sources[f"{key}_query"],
                    "bytes": len(downloads[key][0]),
                    "sha256": _sha256_bytes(downloads[key][0]),
                    "last_modified": downloads[key][1].get("last-modified"),
                    "etag": downloads[key][1].get("etag"),
                    "approved_columns": sources[f"{key}_columns"],
                }
                for key in ("identity", "photometry", "stellar_mass")
            },
            "counts": {
                "identity_rows": len(identities),
                "SL2S_rows": len(survey_rows),
                "incomplete": len(incomplete),
                "predecessor_name_exclusions": len(name_exclusions),
                "predecessor_coordinate_exclusions": len(coordinate_exclusions),
                "eligible": len(predictors),
            },
            "predictor_file": {
                "path": paths["predictors"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["predictors"]),
            },
            "predecessor_audit_sha256": predecessor_audit["content_sha256"],
            "claim_boundary": config["scope"]["claim_ceiling"],
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], candidate_manifest)
    return paths


def _load_prepared(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictor = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    for value, label in (
        (predictor, "predictor manifest"),
        (sample, "sample manifest"),
        (candidates, "candidate manifest"),
    ):
        _verify_content_hash(value, label)
        if int(value.get("response_values_read", 0)) != 0:
            raise GravityItem29Error(f"{label} contains response values")
    if _sha256_file(paths["predictors"]) != predictor["predictor_file"]["sha256"]:
        raise GravityItem29Error("predictor TSV changed")
    _, audit = _admissible_candidates(config)
    if audit["admissible_candidate_digest"] != candidates["audit"][
        "admissible_candidate_digest"
    ]:
        raise GravityItem29Error("candidate arrays changed")
    return predictor, sample, candidates


def _response_url(config: Mapping[str, Any], name: str) -> str:
    params = {
        "-source": "J/MNRAS/488/3745/tablea1",
        "-out": ",".join(config["sources"]["response_columns"]),
        "Name": name,
        "-out.max": "3",
    }
    return str(config["sources"]["response_query_base"]) + "?" + urllib.parse.urlencode(params)


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    _, sample, candidates = _load_prepared(root, config)
    if int(candidates["post_response_candidate_cells"]) != 0:
        raise GravityItem29Error("post-response candidates detected")
    columns = tuple(config["sources"]["response_columns"])
    responses: list[dict[str, str]] = []
    receipts: list[dict[str, Any]] = []
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    for row in exploration:
        name = str(row["name"])
        url = _response_url(config, name)
        body, headers = _download(url)
        parsed = _parse_vizier_tsv(body, columns)
        exact = [
            item
            for item in parsed
            if _canonical_identity(str(item["Name"])) == _canonical_identity(name)
        ]
        if len(exact) != 1:
            raise GravityItem29Error(f"response query did not return one exact row: {name}")
        responses.append(exact[0])
        receipts.append(
            {
                "name": name,
                "url": url,
                "bytes": len(body),
                "sha256": _sha256_bytes(body),
                "last_modified": headers.get("last-modified"),
                "etag": headers.get("etag"),
            }
        )
    paths = _source_paths(root, config)
    responses.sort(key=lambda row: str(row["Name"]))
    _write_tsv(paths["exploration_responses"], responses, columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item29-response-source-1.0",
            "response_columns_queried": list(columns),
            "exploration_response_rows": len(responses),
            "confirmation_response_values_read": 0,
            "post_response_candidate_cells": 0,
            "per_object_receipts": receipts,
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["exploration_responses"]


def _hernquist_3d_fraction(radius_over_a: float) -> float:
    return radius_over_a**2 / (1.0 + radius_over_a) ** 2


def _load_response_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    _, sample, _ = _load_prepared(root, config)
    manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(manifest, "response manifest")
    if int(manifest["confirmation_response_values_read"]) != 0:
        raise GravityItem29Error("confirmation response was opened")
    if _sha256_file(paths["exploration_responses"]) != manifest["response_file"]["sha256"]:
        raise GravityItem29Error("response TSV changed")
    responses = {
        _canonical_identity(str(row["Name"])): row
        for row in _read_tsv(paths["exploration_responses"])
    }
    constants = config["physics"]["constants"]
    rows: list[dict[str, Any]] = []
    for predictor in sample["objects"]:
        if predictor["role"] != "exploration":
            continue
        observed = responses.get(_canonical_identity(str(predictor["name"])))
        if observed is None:
            continue
        try:
            sigma = _float_required(observed, "sigap")
            sigma_error = _float_required(observed, "e_sigap")
            theta = _float_required(observed, "thetaE")
            z_lens = float(predictor["z_lens"])
            z_source = float(predictor["z_source"])
            reff = float(predictor["reff_kpc"])
            mass = float(predictor["stellar_mass_proxy_msun"])
        except (TypeError, ValueError):
            continue
        if not (
            sigma > 0
            and sigma_error > 0
            and theta > 0
            and z_source > z_lens > 0
            and reff > 0
            and mass > 0
        ):
            continue
        d_lens, d_source, d_lens_source = _angular_diameter_distances(
            z_lens, z_source, config
        )
        rein = theta * float(constants["arcsec_to_radian"]) * d_lens
        scale_radius = reff / float(config["physics"]["hernquist_re_over_a"])
        projected_fraction = float(hernquist_projected_mass_fraction(rein / scale_radius))
        dyn_fraction = _hernquist_3d_fraction(reff / scale_radius)
        gravitational = float(constants["G_kpc_km2_s2_Msun"])
        c = float(constants["c_km_s"])
        sigma_critical = (c**2 / (4.0 * math.pi * gravitational)) * (
            d_source / (d_lens * d_lens_source)
        )
        required_lens_mass = math.pi * rein**2 * sigma_critical
        required_dyn_mass = (
            float(config["physics"]["dynamical_virial_coefficient"])
            * reff
            * sigma**2
            / gravitational
        )
        g_si = float(constants["G_si"])
        mass_kg = mass * float(constants["msun_to_kg"])
        kpc_to_m = float(constants["kpc_to_m"])
        g_dyn = g_si * mass_kg * dyn_fraction / (reff * kpc_to_m) ** 2
        g_lens = g_si * mass_kg * projected_fraction / (rein * kpc_to_m) ** 2
        rows.append(
            {
                **predictor,
                "fold": int(predictor["fold"]),
                "sigma_km_s": sigma,
                "sigma_error_km_s": sigma_error,
                "theta_ein_arcsec": theta,
                "rein_kpc": rein,
                "projected_fraction_at_rein": projected_fraction,
                "dynamical_fraction_at_reff": dyn_fraction,
                "g_dyn_m_s2": g_dyn,
                "g_lens_m_s2": g_lens,
                "radius_ratio_dyn": 1.0,
                "radius_ratio_lens": rein / reff,
                "y_dyn": math.log(required_dyn_mass / mass),
                "y_lens": math.log(required_lens_mass / (mass * projected_fraction)),
            }
        )
    rows.sort(key=lambda row: str(row["name"]))
    return rows, manifest


def _backend() -> tuple[Any, str, str]:
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return cp, "gpu_cupy", str(name)
    except (ImportError, RuntimeError, OSError) as error:
        raise GravityItem29Error(f"Item 29 requires the frozen CUDA lane: {error}") from error


def _build_log_response_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    accelerations = xp.asarray(
        np.asarray(
            [[row["g_dyn_m_s2"], row["g_lens_m_s2"]] for row in rows],
            dtype=np.float64,
        )
    )
    radii = xp.asarray(
        np.asarray(
            [[row["radius_ratio_dyn"], row["radius_ratio_lens"]] for row in rows],
            dtype=np.float64,
        )
    )
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(
            _candidate_log_response(config, arrays, accelerations, radii, begin, end, xp)
        )
    return xp.concatenate(pieces, axis=0)


def _screen(
    log_response: Any,
    target: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    bounds = tuple(float(value) for value in config["physics"]["shared_mass_proxy_scale_bounds"])
    target_device = xp.asarray(target)
    prediction = np.empty_like(target)
    selected: list[int] = []
    offsets: list[float] = []
    raw_offsets: list[float] = []
    training_mse: list[float] = []
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        residual = target_device[None, train, :] - log_response[:, train, :]
        raw = xp.mean(residual, axis=(1, 2))
        fitted = xp.clip(raw, math.log(bounds[0]), math.log(bounds[1]))
        mse = xp.mean((residual - fitted[:, None, None]) ** 2, axis=(1, 2))
        index = int(_to_numpy(xp.argmin(mse), xp))
        selected.append(index)
        raw_offsets.append(float(_to_numpy(raw[index], xp)))
        offsets.append(float(_to_numpy(fitted[index], xp)))
        training_mse.append(float(_to_numpy(mse[index], xp)))
        prediction[held] = _to_numpy(log_response[index, held, :], xp) + offsets[-1]
    return {
        "prediction": prediction,
        "selected_indices": selected,
        "log_mass_proxy_offsets": offsets,
        "raw_log_mass_proxy_offsets": raw_offsets,
        "training_mse": training_mse,
    }


def _feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                math.log(float(row["stellar_mass_proxy_msun"])),
                math.log(float(row["reff_kpc"])),
                math.log(float(row["stellar_surface_density_proxy_msun_kpc2"])),
                float(row["z_lens"]),
                float(row["z_source"]),
                float(row["g_minus_i"]),
                float(row["r_minus_z"]),
                float(row["axis_ratio"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _fit_offset(values: np.ndarray, bounds: tuple[float, float]) -> float:
    return float(np.clip(np.mean(values), math.log(bounds[0]), math.log(bounds[1])))


def _baseline_predictions(
    target: np.ndarray,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    bounds = tuple(float(value) for value in config["physics"]["shared_mass_proxy_scale_bounds"])
    shared = np.empty_like(target)
    separate = np.empty_like(target)
    flexible = np.empty_like(target)
    feature = _feature_matrix(rows)
    alpha = float(config["evaluation"]["ridge_alpha"])
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        shared[held] = _fit_offset(target[train].reshape(-1), bounds)
        for channel in range(2):
            separate[held, channel] = _fit_offset(target[train, channel], bounds)
        mean = feature[train].mean(axis=0)
        scale = feature[train].std(axis=0)
        scale[scale == 0] = 1.0
        train_design = np.column_stack([np.ones(len(train)), (feature[train] - mean) / scale])
        held_design = np.column_stack([np.ones(len(held)), (feature[held] - mean) / scale])
        penalty = np.diag([0.0] + [alpha] * feature.shape[1])
        for channel in range(2):
            coefficient = np.linalg.solve(
                train_design.T @ train_design + penalty,
                train_design.T @ target[train, channel],
            )
            flexible[held, channel] = held_design @ coefficient
    return {"shared_GR": shared, "separate_calibration": separate, "flexible_nuisance": flexible}


def _candidate_record(
    index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(arrays["niche"][index])
    return {
        "admissible_candidate_index": index,
        "niche_index": niche,
        "niche": config["candidate_generator"]["niches"][niche],
        "polarity": float(values["polarity"][0]),
        "amplitude": float(values["amplitude"][0]),
        "acceleration_scale_m_s2": float(values["acceleration_scale"][0]),
        "acceleration_power": float(values["acceleration_power"][0]),
        "radius_scale_re": float(values["radius_scale"][0]),
        "radius_power": float(values["radius_power"][0]),
        "feedback_eta": float(values["feedback_eta"][0]),
    }


def _synthetic_controls(
    log_response: Any,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    xp: Any,
) -> dict[str, Any]:
    injections = [
        int(value)
        for value in config["candidate_generator"]["synthetic_injection_admissible_indices"]
    ]
    injection_results = []
    for index in injections:
        target = math.log(1.2) + _to_numpy(log_response[index], xp)
        selected = _screen(log_response, target, folds, config, xp)
        selected_niches = [int(arrays["niche"][value]) for value in selected["selected_indices"]]
        injection_results.append(
            {
                "injection_index": index,
                "injection_niche": int(arrays["niche"][index]),
                "selected_niches": selected_niches,
                "exact_niche_recovered_all_folds": all(
                    value == int(arrays["niche"][index]) for value in selected_niches
                ),
                "candidate_mse": _mse(target, selected["prediction"]),
            }
        )
    gr_target = np.zeros((len(rows), 2), dtype=np.float64)
    gr_selected = _screen(log_response, gr_target, folds, config, xp)
    gr_baseline = _baseline_predictions(gr_target, folds, rows, config)["shared_GR"]
    gr_candidate_mse = _mse(gr_target, gr_selected["prediction"])
    gr_baseline_mse = _mse(gr_target, gr_baseline)
    return {
        "injections": injection_results,
        "all_injected_niches_recovered": all(
            row["exact_niche_recovered_all_folds"] for row in injection_results
        ),
        "GR_candidate_mse": gr_candidate_mse,
        "GR_baseline_mse": gr_baseline_mse,
        "GR_control_prefers_nonzero_self_interaction": gr_candidate_mse
        < gr_baseline_mse - 1e-16,
    }


def _weighted_mse(
    target: np.ndarray,
    prediction: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float:
    mass_error = math.log(10.0) * float(config["evaluation"]["stellar_mass_systematic_dex"])
    lens_fraction = float(config["evaluation"]["lens_radius_fractional_uncertainty"])
    errors = []
    for row in rows:
        sigma_fraction = float(row["sigma_error_km_s"]) / float(row["sigma_km_s"])
        errors.append(
            [math.hypot(2.0 * sigma_fraction, mass_error), math.hypot(2.0 * lens_fraction, mass_error)]
        )
    weights = 1.0 / np.asarray(errors) ** 2
    return float(np.sum(weights * (target - prediction) ** 2) / np.sum(weights))


def _evaluate(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(rows) < int(config["sample"]["outer_folds"]):
        raise GravityItem29Error("too few response-complete objects for diagnostic folds")
    arrays, candidate_audit = _admissible_candidates(config)
    xp, backend, device = _backend()
    target = np.asarray([[row["y_dyn"], row["y_lens"]] for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if set(folds.tolist()) != set(range(int(config["sample"]["outer_folds"]))):
        raise GravityItem29Error("response-complete folds are incomplete")
    xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    log_response = _build_log_response_matrix(config, arrays, rows, xp)
    xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    crosscheck = int(config["evaluation"]["cpu_crosscheck_candidates"])
    accelerations = np.asarray(
        [[row["g_dyn_m_s2"], row["g_lens_m_s2"]] for row in rows], dtype=np.float64
    )
    radii = np.asarray(
        [[row["radius_ratio_dyn"], row["radius_ratio_lens"]] for row in rows],
        dtype=np.float64,
    )
    cpu = _candidate_log_response(config, arrays, accelerations, radii, 0, crosscheck, np)
    gpu = _to_numpy(log_response[:crosscheck], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu - gpu)))
    observed = _screen(log_response, target, folds, config, xp)
    baselines = _baseline_predictions(target, folds, rows, config)
    candidate_mse = _mse(target, observed["prediction"])
    baseline_mse = {key: _mse(target, value) for key, value in baselines.items()}
    observed_statistic = _improvement(baseline_mse["flexible_nuisance"], candidate_mse)
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    null_improvements = []
    trials = int(config["evaluation"]["permutation_trials"])
    for _ in range(trials):
        permutation = random.permutation(len(target))
        null_target = target[permutation]
        null_selected = _screen(log_response, null_target, folds, config, xp)
        null_flexible = _baseline_predictions(null_target, folds, rows, config)[
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
    controls = _synthetic_controls(log_response, folds, rows, config, arrays, xp)
    selected_records = [
        _candidate_record(index, config, arrays) for index in observed["selected_indices"]
    ]
    selected_niches = [int(arrays["niche"][index]) for index in observed["selected_indices"]]
    niche_counts = Counter(selected_niches)
    channel_metrics: dict[str, Any] = {}
    for channel, label in enumerate(("dynamics", "lensing")):
        value = {"candidate_mse": _mse(target[:, channel], observed["prediction"][:, channel])}
        for key, baseline in baselines.items():
            mse = _mse(target[:, channel], baseline[:, channel])
            value[f"{key}_mse"] = mse
            value[f"improvement_vs_{key}"] = _improvement(mse, value["candidate_mse"])
        channel_metrics[label] = value
    slices: dict[str, Any] = {}
    slice_values = {
        "low_mass": np.asarray(
            [float(row["stellar_mass_proxy_msun"]) for row in rows]
        )
        <= np.median([float(row["stellar_mass_proxy_msun"]) for row in rows]),
        "high_mass": np.asarray(
            [float(row["stellar_mass_proxy_msun"]) for row in rows]
        )
        > np.median([float(row["stellar_mass_proxy_msun"]) for row in rows]),
        "small_re": np.asarray([float(row["reff_kpc"]) for row in rows])
        <= np.median([float(row["reff_kpc"]) for row in rows]),
        "large_re": np.asarray([float(row["reff_kpc"]) for row in rows])
        > np.median([float(row["reff_kpc"]) for row in rows]),
        "low_redshift": np.asarray([float(row["z_lens"]) for row in rows])
        <= np.median([float(row["z_lens"]) for row in rows]),
        "high_redshift": np.asarray([float(row["z_lens"]) for row in rows])
        > np.median([float(row["z_lens"]) for row in rows]),
    }
    for label, mask in slice_values.items():
        indices = np.where(mask)[0]
        cand = _mse(target, observed["prediction"], indices)
        flex = _mse(target, baselines["flexible_nuisance"], indices)
        shared = _mse(target, baselines["shared_GR"], indices)
        slices[label] = {
            "objects": len(indices),
            "candidate_mse": cand,
            "improvement_vs_shared_GR": _improvement(shared, cand),
            "improvement_vs_flexible": _improvement(flex, cand),
        }
    object_counterexamples = int(
        np.count_nonzero(
            np.mean((target - observed["prediction"]) ** 2, axis=1)
            > np.mean((target - baselines["flexible_nuisance"]) ** 2, axis=1)
        )
    )
    quality_pass = len(rows) >= int(config["sample"]["minimum_complete_exploration_objects"])
    gates = config["gates"]
    universal_gates = {
        "response_quality": quality_pass,
        "confirmation_values_read_zero": int(gates["confirmation_values_read"]) == 0,
        "post_response_cells_zero": int(gates["post_response_candidate_cells"]) == 0,
        "joint_improvement_vs_shared_GR": _improvement(
            baseline_mse["shared_GR"], candidate_mse
        )
        >= float(gates["minimum_joint_mse_improvement_vs_shared_GR"]),
        "joint_improvement_vs_separate_calibration": _improvement(
            baseline_mse["separate_calibration"], candidate_mse
        )
        >= float(gates["minimum_joint_mse_improvement_vs_separate_calibration"]),
        "joint_improvement_vs_flexible": observed_statistic
        >= float(gates["minimum_joint_mse_improvement_vs_flexible_nuisance"]),
        "each_channel_improves_shared_GR": all(
            value["improvement_vs_shared_GR"]
            >= float(gates["minimum_each_channel_improvement_vs_shared_GR"])
            for value in channel_metrics.values()
        ),
        "each_broad_half_improves_shared_GR": all(
            value["improvement_vs_shared_GR"]
            >= float(gates["minimum_each_broad_half_improvement_vs_shared_GR"])
            for value in slices.values()
        ),
        "selection_aware_permutation": p_value
        <= float(gates["maximum_selection_aware_permutation_p"]),
        "stable_niche": max(niche_counts.values())
        >= int(gates["minimum_same_niche_folds"]),
        "all_injected_niches_recovered": bool(controls["all_injected_niches_recovered"]),
        "known_GR_control": not bool(
            controls["GR_control_prefers_nonzero_self_interaction"]
        ),
        "cpu_gpu_agreement": cpu_gpu_max <= 1e-11,
        "local_limit": candidate_audit["maximum_admitted_local_fractional_response"]
        <= float(config["admissibility"]["maximum_local_fractional_response_at_1au"]),
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
        if value["improvement_vs_flexible"]
        >= float(gates["partial_minimum_slice_improvement_vs_flexible"])
    ]
    universal_pass = all(universal_gates.values())
    phenomenon_pass = all(phenomenon_gates.values())
    if not quality_pass:
        decision = "INCONCLUSIVE_QUALITY"
    elif universal_pass:
        decision = "PASS_ITEM29_EXPLORATION_UNIVERSAL"
    elif phenomenon_pass:
        decision = "NONPROMOTED_POSITIVE_PHENOMENON_LEAD"
    elif partial_slices:
        decision = "SCOPED_PARTIAL_PATTERN_RETAINED"
    else:
        decision = "SCOPED_REJECT"
    scientific = {
        "decision": decision,
        "quality": {
            "complete_exploration_objects": len(rows),
            "minimum_required": int(config["sample"]["minimum_complete_exploration_objects"]),
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
            "improvement_vs_shared_GR": _improvement(baseline_mse["shared_GR"], candidate_mse),
            "improvement_vs_separate_calibration": _improvement(
                baseline_mse["separate_calibration"], candidate_mse
            ),
            "improvement_vs_flexible": observed_statistic,
            "selection_aware_permutation_p": p_value,
            "maximum_null_improvement": max(null_improvements),
            "weighted_candidate_mse": _weighted_mse(
                target, observed["prediction"], rows, config
            ),
            "weighted_flexible_mse": _weighted_mse(
                target, baselines["flexible_nuisance"], rows, config
            ),
            "object_counterexamples_vs_flexible": object_counterexamples,
        },
        "channels": channel_metrics,
        "broad_slices": slices,
        "selected_candidates": selected_records,
        "selected_niche_counts": {str(key): value for key, value in sorted(niche_counts.items())},
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
    training_values_per_search = int(
        sum(np.count_nonzero(folds != fold) for fold in range(config["sample"]["outer_folds"]))
        * 2
        * len(arrays["niche"])
    )
    compute = {
        "backend": backend,
        "device": device,
        "candidate_matrix_seconds": matrix_seconds,
        "candidate_cells": len(arrays["niche"]),
        "candidate_observable_matrix_values": int(np.prod(log_response.shape)),
        "candidate_training_residual_evaluations_observed": training_values_per_search,
        "candidate_training_residual_evaluations_with_nulls": training_values_per_search
        * (trials + 1),
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
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    predictor, sample, candidates = _load_prepared(root, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item29-nonlinear-self-interaction-result-1.0",
            "item": 29,
            "title": config["title"],
            "decision": scientific["decision"],
            "hypothesis": config["hypothesis"],
            "scientific": scientific,
            "compute": compute,
            "theory": {
                "sources": config["sources"]["theory_sources"],
                "families": config["candidate_generator"]["niches"],
                "matter_response": config["physics"]["matter_response"],
                "light_response": config["physics"]["light_response"],
                "stability_scope": config["physics"]["stability_scope"],
            },
            "frozen_boundary": {
                "stable_goal_sha256": config["stable_goal_sha256"],
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "predictor_manifest_sha256": predictor["content_sha256"],
                "sample_manifest_sha256": sample["content_sha256"],
                "candidate_manifest_sha256": candidates["content_sha256"],
                "response_manifest_sha256": response_manifest["content_sha256"],
                "response_file_sha256": response_manifest["response_file"]["sha256"],
                "complete_response_objects": len(rows),
                "confirmation_response_values_read": 0,
                "post_response_formula_generation": False,
                "paid_api_calls": 0,
            },
            "claim_boundary": [
                config["scope"]["claim_ceiling"],
                "The stellar mass proxy and spherical Hernquist mapping carry IMF, anisotropy, environment, and profile systematics.",
                "Observed Einstein radius is the evaluation radius; this is not a direct image or shear likelihood.",
                "A positive result would be an exploration lead only; confirmations remain sealed and an unchanged fresh replication is mandatory.",
                "Historical novelty is not inferred from a creativity label or behavioral non-equivalence.",
            ],
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_response_rows(root, config)
    scientific, compute = _evaluate(config, rows)
    paths = _source_paths(root, config)
    compute_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item29-compute-1.0",
            **compute,
        }
    )
    _write_json(paths["compute_manifest"], compute_manifest)
    receipt = _build_receipt(root, config, rows, response_manifest, scientific, compute)
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
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem29Error("checked result opened a confirmation response")
    if bool(result["frozen_boundary"]["post_response_formula_generation"]):
        raise GravityItem29Error("checked result contains post-response generation")
    if _sha256_file(paths["exploration_responses"]) != result["frozen_boundary"][
        "response_file_sha256"
    ]:
        raise GravityItem29Error("checked response file changed")


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
