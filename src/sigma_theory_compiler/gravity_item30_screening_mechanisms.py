"""Frozen Item 30 screening-mechanism experiment.

Only response-independent MaNGA structure and GEMA baryonic-environment
predictors may define candidates, exclusions, sample roles, and folds.  Central
stellar-dispersion responses are queried for exploration roles only after the
scientific and sample freezes are both committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item12_manga_dynamical_age import (
    _coordinates as _item12_legacy_coordinates,
)
from sigma_theory_compiler.gravity_item12_manga_dynamical_age import (
    _validate_content_hash as _validate_legacy_content_hash,
)
from sigma_theory_compiler.gravity_item16_s4tm_qed_field import _parse_vizier_tsv
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

CONFIG_PATH = Path("configs/gravity_item30_screening_mechanisms_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item30_screening_mechanisms.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
_ADMISSIBLE_CACHE: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}


class GravityItem30Error(RuntimeError):
    """Raised when an Item 30 freeze, leakage, or replay invariant is violated."""


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant-Item30/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not body:
        raise GravityItem30Error(f"empty source response: {url}")
    return body, headers


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item30-screening-mechanisms-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 30:
        raise GravityItem30Error("unexpected Item 30 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem30Error("stable gravity goal changed")
    if int(config["candidate_generator"]["raw_candidate_cells"]) != 262144:
        raise GravityItem30Error("raw candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem30Error("post-response candidates entered Item 30")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem30Error("equal-viability policy changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem30Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem30Error("paid calls are outside Item 30")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem30Error(f"scientific dependency changed: {relative}")
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
        raise GravityItem30Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem30Error("could not read frozen Item 30 module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem30Error("Item 30 module differs from scientific freeze")
    paths = _source_paths(root, config)
    for key in (
        "ugc_catalog",
        "predecessor_coordinates",
        "predecessor_identities",
        "predecessor_audit",
    ):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem30Error(f"{key} differs from scientific freeze")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "gema_catalog",
        "ugc_catalog",
        "predecessor_coordinates",
        "predecessor_identities",
        "predecessor_audit",
        "predictors",
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
        "gema_catalog",
        "ugc_catalog",
        "predecessor_coordinates",
        "predecessor_identities",
        "predecessor_audit",
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        relative = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem30Error(f"{key} differs from sample freeze")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()


def _manifest_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("role") is not None:
            yield value
        for child in value.values():
            yield from _manifest_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _manifest_objects(child)


def _full_coordinate_from_text(value: str) -> tuple[float, float] | None:
    text = str(value).upper().replace("SDSS", "").replace("J", "")
    match = re.search(r"(\d{6}(?:\.\d+)?)([+-])(\d{6}(?:\.\d+)?)", text)
    if match is None:
        return None
    ra_text, sign, dec_text = match.groups()
    ra = 15.0 * (float(ra_text[:2]) + float(ra_text[2:4]) / 60.0 + float(ra_text[4:]) / 3600.0)
    dec = float(dec_text[:2]) + float(dec_text[2:4]) / 60.0 + float(dec_text[4:]) / 3600.0
    return ra, dec if sign == "+" else -dec


def _parse_ugc(payload: bytes) -> list[dict[str, Any]]:
    rows = _parse_vizier_tsv(payload, ("UGC", "_RAJ2000", "_DEJ2000"))
    answer = []
    for row in rows:
        try:
            ugc = int(row["UGC"])
            ra = float(row["_RAJ2000"])
            dec = float(row["_DEJ2000"])
        except (KeyError, TypeError, ValueError):
            continue
        answer.append({"ugc": ugc, "ra_deg": ra, "dec_deg": dec})
    if len(answer) < 12000:
        raise GravityItem30Error("UGC coordinate catalog is unexpectedly incomplete")
    return answer


def build_predecessor_audit(root: Path) -> dict[str, Path]:
    """Write the response-blind cross-survey predecessor catalog before science freeze."""
    config = load_config(root)
    paths = _source_paths(root, config)
    paths["predecessor_coordinates"].parent.mkdir(parents=True, exist_ok=True)
    ugc_payload, ugc_headers = _download(str(config["sources"]["ugc_url"]))
    paths["ugc_catalog"].write_bytes(ugc_payload)
    ugc_rows = _parse_ugc(ugc_payload)
    ugc_coordinates = {row["ugc"]: (row["ra_deg"], row["dec_deg"]) for row in ugc_rows}

    legacy_config = _read_json(root / str(config["sources"]["legacy_coordinate_config"]))
    legacy = _item12_legacy_coordinates(root, legacy_config)
    coordinates: list[dict[str, Any]] = [
        {
            "ra_deg": float(row[0]),
            "dec_deg": float(row[1]),
            "source": "item12_legacy_predecessor_coordinate_union",
            "identity": f"legacy-{index}",
        }
        for index, row in enumerate(legacy)
    ]
    identities: list[dict[str, str]] = []
    source_receipts = []
    ugc_roles: set[int] = set()
    item24_roles: set[str] = set()
    for relative in config["sources"]["predecessor_sample_manifests"]:
        path = root / str(relative)
        manifest = _read_json(path)
        objects = list(_manifest_objects(manifest))
        source_receipts.append(
            {"path": str(relative), "sha256": _sha256_file(path), "role_objects": len(objects)}
        )
        for row in objects:
            mangaid = row.get("mangaid", row.get("manga_id"))
            plateifu = row.get("plateifu")
            if mangaid:
                identities.append(
                    {"kind": "mangaid", "value": str(mangaid).strip(), "source": str(relative)}
                )
            if plateifu:
                identities.append(
                    {"kind": "plateifu", "value": str(plateifu).strip(), "source": str(relative)}
                )
            if "item-24-temporal-lapse" in str(relative) and row.get("lane") == "galaxy_motion":
                item24_roles.add(str(row.get("identity", "")).strip())
            ugc = row.get("ugc")
            if ugc is not None:
                try:
                    ugc_roles.add(int(ugc))
                except (TypeError, ValueError):
                    pass
            added = False
            for ra_key, dec_key in (
                ("ra", "dec"),
                ("ra_deg", "dec_deg"),
                ("catalog_ra_deg", "catalog_dec_deg"),
            ):
                try:
                    ra = float(row[ra_key])
                    dec = float(row[dec_key])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    np.isfinite(ra)
                    and np.isfinite(dec)
                    and 0.0 <= ra <= 360.0
                    and -90.0 <= dec <= 90.0
                ):
                    coordinates.append(
                        {
                            "ra_deg": ra,
                            "dec_deg": dec,
                            "source": str(relative),
                            "identity": str(
                                mangaid or plateifu or row.get("name", row.get("identity", ""))
                            ),
                        }
                    )
                    added = True
                    break
            if added:
                continue
            for key in ("sdss", "name", "display_name", "identity"):
                parsed = _full_coordinate_from_text(str(row.get(key, "")))
                if parsed is not None:
                    coordinates.append(
                        {
                            "ra_deg": parsed[0],
                            "dec_deg": parsed[1],
                            "source": str(relative),
                            "identity": str(row.get(key, "")),
                        }
                    )
                    break

    for ugc in sorted(ugc_roles):
        if ugc not in ugc_coordinates:
            raise GravityItem30Error(f"UGC predecessor missing from coordinate catalog: {ugc}")
        ra, dec = ugc_coordinates[ugc]
        coordinates.append(
            {
                "ra_deg": ra,
                "dec_deg": dec,
                "source": str(config["sources"]["ugc_catalog"]),
                "identity": f"UGC{ugc:05d}",
            }
        )

    item24_path = root / str(config["sources"]["item24_galaxy_predictors"])
    item24_rows = _read_tsv(item24_path)
    joined_item24 = 0
    for row in item24_rows:
        if str(row.get("agc", "")).strip() not in item24_roles:
            continue
        coordinates.append(
            {
                "ra_deg": float(row["ra_deg"]),
                "dec_deg": float(row["dec_deg"]),
                "source": str(config["sources"]["item24_galaxy_predictors"]),
                "identity": f"AGC{str(row['agc']).strip()}",
            }
        )
        joined_item24 += 1
    if joined_item24 != len(item24_roles):
        raise GravityItem30Error("Item 24 role-to-coordinate join is incomplete")

    coordinate_rows = sorted(
        coordinates,
        key=lambda row: (
            float(row["ra_deg"]),
            float(row["dec_deg"]),
            str(row["source"]),
            str(row["identity"]),
        ),
    )
    identity_rows = sorted(
        {tuple(row[key] for key in ("kind", "value", "source")) for row in identities}
    )
    _write_tsv(
        paths["predecessor_coordinates"],
        [
            {
                "ra_deg": f"{float(row['ra_deg']):.12e}",
                "dec_deg": f"{float(row['dec_deg']):.12e}",
                "source": row["source"],
                "identity": row["identity"],
            }
            for row in coordinate_rows
        ],
        ("ra_deg", "dec_deg", "source", "identity"),
    )
    _write_tsv(
        paths["predecessor_identities"],
        [{"kind": row[0], "value": row[1], "source": row[2]} for row in identity_rows],
        ("kind", "value", "source"),
    )
    audit = _content_hashed(
        {
            "schema_version": "invariant-gravity-item30-predecessor-audit-1.0",
            "response_values_read": 0,
            "paid_api_calls": 0,
            "legacy_coordinates": len(legacy),
            "explicit_and_resolved_coordinates": len(coordinate_rows) - len(legacy),
            "total_coordinate_rows": len(coordinate_rows),
            "unique_manga_identities": len(identity_rows),
            "ugc_role_coordinates": len(ugc_roles),
            "item24_role_coordinates": joined_item24,
            "predecessor_sources": source_receipts,
            "item24_predictor_source": {
                "path": str(config["sources"]["item24_galaxy_predictors"]),
                "sha256": _sha256_file(item24_path),
            },
            "ugc_source": {
                "url": config["sources"]["ugc_url"],
                "payload_sha256": _sha256_bytes(ugc_payload),
                "rows": len(ugc_rows),
                "headers": ugc_headers,
            },
            "coordinate_file_sha256": _sha256_file(paths["predecessor_coordinates"]),
            "identity_file_sha256": _sha256_file(paths["predecessor_identities"]),
        }
    )
    _write_json(paths["predecessor_audit"], audit)
    return paths


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    per_niche = int(generator["raw_candidate_cells"]) // 4
    radices = {
        "polarity": len(generator["polarities"]),
        "amplitude": len(generator["amplitudes"]),
        "threshold": int(generator["threshold_cells"]),
        "sharpness": len(generator["sharpness"]),
        "environment": len(generator["environment_couplings"]),
        "shape": len(generator["shape_powers"]),
        "scale": len(generator["scale_factors"]),
    }
    if int(np.prod(list(radices.values()))) != per_niche:
        raise GravityItem30Error("Item 30 mixed-radix grammar does not fill each niche exactly")
    pieces: dict[str, list[np.ndarray]] = {"niche": []} | {key: [] for key in radices}
    for niche in range(4):
        working = np.arange(per_niche, dtype=np.int64)
        decoded: dict[str, np.ndarray] = {}
        for key, radix in reversed(list(radices.items())):
            decoded[key] = (working % radix).astype(np.int16)
            working //= radix
        if np.any(working != 0):
            raise GravityItem30Error("Item 30 candidate decoder overflow")
        pieces["niche"].append(np.full(per_niche, niche, dtype=np.int16))
        for key in radices:
            pieces[key].append(decoded[key])
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    order = random.permutation(len(arrays["niche"]))
    return {key: value[order] for key, value in arrays.items()}


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
        "chameleon_threshold": xp.asarray(
            np.asarray(generator["chameleon_log10_phi_thresholds"])[index["threshold"]]
        ),
        "symmetron_threshold": xp.asarray(
            np.asarray(generator["symmetron_log10_density_thresholds_msun_kpc3"])[
                index["threshold"]
            ]
        ),
        "vainshtein_crossover": xp.asarray(
            np.asarray(generator["vainshtein_log10_crossover_kpc"])[index["threshold"]]
        ),
        "hybrid_phi_threshold": xp.asarray(
            np.asarray(generator["hybrid_log10_phi_thresholds"])[index["threshold"]]
        ),
        "hybrid_density_threshold": xp.asarray(
            np.asarray(generator["hybrid_log10_density_base_thresholds"])[index["threshold"]]
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
    shape = (-1, 1)
    niche = values["niche"].reshape(shape)
    sharp = values["sharpness"].reshape(shape)
    environment = values["environment"].reshape(shape)
    shape_power = values["shape"].reshape(shape)
    scale = values["scale"].reshape(shape)
    phi = xp.maximum(xp.asarray(predictors["dimensionless_potential"])[None, :], 1e-300)
    density = xp.maximum(xp.asarray(predictors["density_msun_kpc3"])[None, :], 1e-300)
    mass = xp.maximum(xp.asarray(predictors["mass_msun"])[None, :], 1e-300)
    radius = xp.maximum(xp.asarray(predictors["radius_kpc"])[None, :], 1e-300)
    q_lss = xp.clip(xp.asarray(predictors["q_lss"])[None, :], -12.0, 4.0)
    eta_k = xp.clip(xp.asarray(predictors["eta_k"])[None, :], -4.0, 5.0)

    chameleon_phi_c = xp.power(10.0, values["chameleon_threshold"].reshape(shape)) * scale
    phi_effective = phi * (1.0 + environment * xp.power(10.0, q_lss))
    low_phi = 1.0 / (1.0 + xp.power(phi_effective / chameleon_phi_c, sharp))
    chameleon = 1.0 - xp.power(xp.maximum(1.0 - low_phi, 0.0), shape_power)

    symmetron_rho_c = xp.power(10.0, values["symmetron_threshold"].reshape(shape)) * scale
    density_effective = density * (1.0 + environment * xp.power(10.0, eta_k - 1.0))
    symmetron = xp.power(
        1.0 / (1.0 + xp.power(density_effective / symmetron_rho_c, sharp)),
        shape_power,
    )

    constants = config["physics"]["constants"]
    schwarzschild_radius = (
        2.0 * float(constants["G_kpc_km2_s2_Msun"]) * mass / float(constants["c_km_s"]) ** 2
    )
    crossover = xp.power(10.0, values["vainshtein_crossover"].reshape(shape)) * scale
    vainshtein_radius = xp.power(schwarzschild_radius * crossover * crossover, 1.0 / 3.0)
    vainshtein_radius *= xp.power(1.0 + environment * xp.power(10.0, q_lss), 1.0 / 3.0)
    radius_ratio = xp.maximum(radius / vainshtein_radius, 1e-300)
    vainshtein = xp.power(
        2.0 / (1.0 + xp.sqrt(1.0 + xp.power(radius_ratio, -sharp))),
        shape_power,
    )

    hybrid_phi_c = xp.power(10.0, values["hybrid_phi_threshold"].reshape(shape)) * scale
    hybrid_rho_c = xp.power(10.0, values["hybrid_density_threshold"].reshape(shape)) / scale
    hybrid_low_phi = 1.0 / (1.0 + xp.power(phi / hybrid_phi_c, sharp))
    hybrid_low_rho = 1.0 / (1.0 + xp.power(density / hybrid_rho_c, shape_power))
    joint = xp.power(xp.maximum(hybrid_low_phi * hybrid_low_rho, 0.0), 0.5 * (sharp + shape_power))
    competing_screen = 0.5 * ((1.0 - hybrid_low_phi) + (1.0 - hybrid_low_rho))
    nonseparable = joint / xp.maximum(joint + competing_screen, 1e-300)
    isolation = 1.0 / (1.0 + environment * (xp.power(10.0, q_lss) + xp.power(10.0, eta_k - 1.0)))
    hybrid = nonseparable * isolation

    return xp.where(
        niche == 0,
        chameleon,
        xp.where(niche == 1, symmetron, xp.where(niche == 2, vainshtein, hybrid)),
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
        raise GravityItem30Error("adversarial point count changed")
    constants = config["physics"]["constants"]
    mass = 10.0**log_mass
    radius = 10.0**log_radius
    potential = (
        float(constants["G_kpc_km2_s2_Msun"]) * mass / radius / float(constants["c_km_s"]) ** 2
    )
    density = mass / (4.0 * math.pi * radius**3 / 3.0)
    return {
        "dimensionless_potential": potential,
        "density_msun_kpc3": density,
        "mass_msun": mass,
        "radius_kpc": radius,
        "q_lss": np.tile(np.asarray([-8.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]), 8),
        "eta_k": np.repeat(np.linspace(-2.0, 3.0, 8), 8),
    }


def _local_predictors(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    local = config["physics"]["local_reference"]
    return {
        "dimensionless_potential": np.asarray([float(local["dimensionless_total_potential"])]),
        "density_msun_kpc3": np.asarray([float(local["mean_density_msun_kpc3"])]),
        "mass_msun": np.asarray([float(local["mass_msun"])]),
        "radius_kpc": np.asarray([float(local["radius_kpc"])]),
        "q_lss": np.asarray([float(local["q_lss"])]),
        "eta_k": np.asarray([float(local["eta_k"])]),
    }


def _admissible_candidates(
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_key = _sha256_bytes(
        _canonical_bytes(
            {
                "candidate_generator": config["candidate_generator"],
                "admissibility": config["admissibility"],
                "physics": config["physics"]["local_reference"],
            }
        )
    )
    if cache_key in _ADMISSIBLE_CACHE:
        return _ADMISSIBLE_CACHE[cache_key]
    raw = generate_raw_candidates(config)
    domain = _adversarial_predictors(config)
    local = _local_predictors(config)
    keep = np.zeros(len(raw["niche"]), dtype=bool)
    local_fraction = np.full(len(keep), np.nan)
    maximum_mu = np.full(len(keep), np.nan)
    minimum_mu = np.full(len(keep), np.nan)
    screening_drop = np.full(len(keep), np.nan)
    batch = int(config["evaluation"]["candidate_batch_size"])
    gates = config["admissibility"]
    low = {
        "dimensionless_potential": np.asarray([1e-10]),
        "density_msun_kpc3": np.asarray([1e4]),
        "mass_msun": np.asarray([1e9]),
        "radius_kpc": np.asarray([30.0]),
        "q_lss": np.asarray([-8.0]),
        "eta_k": np.asarray([-2.0]),
    }
    high = {
        "dimensionless_potential": np.asarray([1e-5]),
        "density_msun_kpc3": np.asarray([1e11]),
        "mass_msun": np.asarray([1e12]),
        "radius_kpc": np.asarray([0.1]),
        "q_lss": np.asarray([2.0]),
        "eta_k": np.asarray([3.0]),
    }
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
        screening_drop[begin:end] = low_activation - high_activation
        direction = screening_drop[begin:end] >= float(gates["minimum_known_family_screening_drop"])
        keep[begin:end] = finite & bounded & material & local_pass & direction
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
        "maximum_admitted_local_fractional_response": float(np.max(local_fraction[keep])),
        "minimum_admitted_mu": float(np.min(minimum_mu[keep])),
        "maximum_admitted_mu": float(np.max(maximum_mu[keep])),
        "minimum_admitted_screening_drop": float(np.min(screening_drop[keep])),
    }
    generator = config["candidate_generator"]
    expected = generator.get("expected_raw_candidate_digest")
    if expected is not None and audit["raw_candidate_digest"] != expected:
        raise GravityItem30Error("raw Item 30 candidate digest changed")
    expected = generator.get("expected_admissible_candidate_digest")
    if expected is not None and audit["admissible_candidate_digest"] != expected:
        raise GravityItem30Error("admissible Item 30 candidate digest changed")
    expected = generator.get("expected_admissible_candidates")
    if expected is not None and audit["admissible_candidates"] != int(expected):
        raise GravityItem30Error("admissible Item 30 candidate count changed")
    expected = generator.get("expected_admissible_per_niche")
    if expected is not None and audit["admissible_per_niche"] != expected:
        raise GravityItem30Error("admissible Item 30 niche counts changed")
    _ADMISSIBLE_CACHE[cache_key] = (arrays, audit)
    return arrays, audit


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item30-candidate-manifest-1.0",
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


def _minimum_separations_arcsec(
    rows: Sequence[Mapping[str, Any]], coordinates: np.ndarray
) -> np.ndarray:
    answer = np.full(len(rows), np.inf, dtype=np.float64)
    predecessor_ra = np.radians(coordinates[:, 0])
    predecessor_dec = np.radians(coordinates[:, 1])
    for index, row in enumerate(rows):
        ra = math.radians(float(row["ra"]))
        dec = math.radians(float(row["dec"]))
        cosine = math.sin(dec) * np.sin(predecessor_dec) + math.cos(dec) * np.cos(
            predecessor_dec
        ) * np.cos(ra - predecessor_ra)
        answer[index] = float(np.min(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) * 3600.0))
    return answer


def _serialize_predictor(row: Mapping[str, Any]) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (float, np.floating)):
            answer[key] = f"{float(value):.12e}"
        elif isinstance(value, (int, np.integer)):
            answer[key] = int(value)
        else:
            answer[key] = value
    return answer


def _sample_manifest(
    config: Mapping[str, Any], predictors: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    phi_median = float(np.median([float(row["log_dimensionless_potential"]) for row in predictors]))
    q_median = float(np.median([float(row["gema_q_lss"]) for row in predictors]))
    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for source in predictors:
        row = dict(source)
        phi_bin = int(float(row["log_dimensionless_potential"]) > phi_median)
        environment_bin = int(float(row["gema_q_lss"]) > q_median)
        row["compactness_bin"] = phi_bin
        row["environment_bin"] = environment_bin
        row["sample_cell"] = f"phi{phi_bin}-env{environment_bin}"
        cells.setdefault((phi_bin, environment_bin), []).append(row)
    selected: list[dict[str, Any]] = []
    selected_cell_counts = {}
    fold_counts = Counter()
    per_cell = int(config["sample"]["selected_per_compactness_environment_cell"])
    confirmations = int(config["sample"]["confirmation_per_cell"])
    role_key = str(config["sample"]["role_key"])
    fold_key = str(config["sample"]["fold_key"])
    outer_folds = int(config["sample"]["outer_folds"])
    for cell in sorted(cells):
        ordered = sorted(cells[cell], key=lambda row: _hmac_rank(role_key, str(row["plateifu"])))
        if len(ordered) < per_cell:
            raise GravityItem30Error(f"Item 30 sample cell cannot fill frozen allocation: {cell}")
        chosen = ordered[:per_cell]
        confirmation_ids = {
            row["plateifu"]
            for row in sorted(
                chosen,
                key=lambda row: _hmac_rank(role_key + "-confirmation", str(row["plateifu"])),
            )[:confirmations]
        }
        exploration = [row for row in chosen if row["plateifu"] not in confirmation_ids]
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
            if row["plateifu"] not in confirmation_ids:
                continue
            row["role"] = "reserved_confirmation"
            row["outer_fold"] = None
            row["response_read"] = False
            selected.append(row)
        selected_cell_counts[f"phi{cell[0]}-env{cell[1]}"] = {
            "eligible": len(cells[cell]),
            "selected": len(chosen),
            "exploration": len(exploration),
            "reserved_confirmation": len(confirmation_ids),
        }
    selected = sorted(selected, key=lambda row: str(row["plateifu"]))
    counts = Counter(str(row["role"]) for row in selected)
    expected = config["sample"]
    if len(selected) != int(expected["expected_selected"]):
        raise GravityItem30Error("Item 30 selected sample count changed")
    if counts["exploration"] != int(expected["expected_exploration"]):
        raise GravityItem30Error("Item 30 exploration count changed")
    if counts["reserved_confirmation"] != int(expected["expected_confirmation"]):
        raise GravityItem30Error("Item 30 confirmation count changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item30-sample-manifest-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "counts": {
                "environment_eligible": len(predictors),
                "selected": len(selected),
                "exploration": counts["exploration"],
                "reserved_confirmation": counts["reserved_confirmation"],
                "response_rows_read": 0,
            },
            "compactness_median": f"{phi_median:.12e}",
            "environment_median": f"{q_median:.12e}",
            "selected_cell_counts": selected_cell_counts,
            "fold_counts_exploration": {str(key): fold_counts[key] for key in range(outer_folds)},
            "objects": selected,
            "claims": {
                "response_values_read": 0,
                "confirmation_values_read": 0,
                "object_identity_used_as_numeric_feature": False,
                "failed_identity_replacement": False,
            },
        }
    )


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    for key in ("predecessor_coordinates", "predecessor_identities", "predecessor_audit"):
        if not paths[key].exists():
            raise GravityItem30Error(f"frozen predecessor artifact is missing: {key}")
    _verify_content_hash(_read_json(paths["predecessor_audit"]), "predecessor audit")
    predecessor_coordinates = _read_tsv(paths["predecessor_coordinates"])
    coordinate_array = np.asarray(
        [[float(row["ra_deg"]), float(row["dec_deg"])] for row in predecessor_coordinates],
        dtype=np.float64,
    )
    predecessor_identities = _read_tsv(paths["predecessor_identities"])
    mangaids = {row["value"] for row in predecessor_identities if row["kind"] == "mangaid"}
    plateifus = {row["value"] for row in predecessor_identities if row["kind"] == "plateifu"}

    predictor_path = root / str(config["sources"]["manga_predictor_source"])
    predictor_source = _read_json(predictor_path)
    # Item 12 predates the newline-terminated receipt convention used by Item 30.
    # Its embedded digest is still independently bound by the exact file SHA-256 in
    # ``scientific_dependencies`` above; validate the embedded digest with Item 12's
    # canonicalizer rather than silently rewriting the immutable predecessor receipt.
    _validate_legacy_content_hash(predictor_source, "frozen response-blind MaNGA predictor source")
    if int(predictor_source["counts"]["response_columns_requested"]) != 0:
        raise GravityItem30Error("response column entered inherited MaNGA predictors")
    records = predictor_source["records"]
    if len(records) != int(config["sample"]["expected_valid_manga_predictors"]):
        raise GravityItem30Error("inherited MaNGA predictor count changed")
    separations = _minimum_separations_arcsec(records, coordinate_array)
    veto = float(config["independence"]["coordinate_veto_arcsec"])
    fresh: list[dict[str, Any]] = []
    exclusion_counts = Counter()
    for row, separation in zip(records, separations, strict=True):
        if str(row["mangaid"]) in mangaids or str(row["plateifu"]) in plateifus:
            exclusion_counts["predecessor_identity"] += 1
            continue
        if separation <= veto:
            exclusion_counts["predecessor_coordinate"] += 1
            continue
        value = dict(row)
        value["minimum_predecessor_separation_arcsec"] = float(separation)
        fresh.append(value)
    expected_fresh = config["sample"].get("expected_fresh_after_identity_and_coordinate_veto")
    if expected_fresh is not None and len(fresh) != int(expected_fresh):
        raise GravityItem30Error("fresh Item 30 predictor count changed")

    gema_payload, gema_headers = _download(str(config["sources"]["gema_url"]))
    if _sha256_bytes(gema_payload) != str(config["sources"]["gema_sha256"]):
        raise GravityItem30Error("official GEMA FITS checksum changed")
    paths["gema_catalog"].write_bytes(gema_payload)
    from astropy.io import fits

    with fits.open(io.BytesIO(gema_payload), memmap=False) as hdus:
        completeness_hdu = hdus[str(config["sources"]["gema_hdu_completeness"])]
        environment_hdu = hdus[str(config["sources"]["gema_hdu_environment"])]
        completeness = {str(row["mangaid"]).strip(): row for row in completeness_hdu.data}
        environment = {str(row["mangaid"]).strip(): row for row in environment_hdu.data}
    constants = config["physics"]["constants"]
    eligible: list[dict[str, Any]] = []
    minimum_completeness = float(
        config["sample"]["environment_minimum_5mpc_redshift_completeness_percent"]
    )
    for row in fresh:
        mangaid = str(row["mangaid"])
        comp = completeness.get(mangaid)
        env = environment.get(mangaid)
        if comp is None or env is None:
            exclusion_counts["gema_join"] += 1
            continue
        if bool(config["sample"]["require_5mpc_footprint"]) and int(comp["footprint_5"]) != 1:
            exclusion_counts["outside_5mpc_footprint"] += 1
            continue
        zcomp = float(comp["zcomp_5Mpc"])
        if not np.isfinite(zcomp) or zcomp < minimum_completeness:
            exclusion_counts["low_5mpc_redshift_completeness"] += 1
            continue
        env_values = np.asarray(
            [env["Q_nn"], env["dnn"], env["Q_lss"], env["eta_k"], env["dkn"]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(env_values)) or np.any(env_values <= -900.0):
            exclusion_counts["incomplete_environment"] += 1
            continue
        mass = 10.0 ** float(row["log_stellar_mass"])
        radius = 10.0 ** float(row["log_half_light_radius"])
        potential = (
            float(constants["G_kpc_km2_s2_Msun"]) * mass / radius / float(constants["c_km_s"]) ** 2
        )
        acceleration = (
            float(constants["G_kpc_km2_s2_Msun"])
            * mass
            / radius**2
            * 1000.0**2
            / float(constants["kpc_to_m"])
        )
        density = mass / (4.0 * math.pi * radius**3 / 3.0)
        value = dict(row)
        value.update(
            {
                "stellar_mass_msun": mass,
                "half_light_radius_kpc": radius,
                "dimensionless_potential": potential,
                "log_dimensionless_potential": math.log10(potential),
                "internal_acceleration_m_s2": acceleration,
                "mean_stellar_density_msun_kpc3": density,
                "log_mean_stellar_density": math.log10(density),
                "gema_zcomp_1mpc_percent": float(comp["zcomp_1Mpc"]),
                "gema_zcomp_5mpc_percent": zcomp,
                "gema_footprint_1mpc": int(comp["footprint_1"]),
                "gema_footprint_5mpc": int(comp["footprint_5"]),
                "gema_nneigh": int(env["nneigh"]),
                "gema_q_nn": float(env["Q_nn"]),
                "gema_dnn_mpc": float(env["dnn"]),
                "gema_q_lss": float(env["Q_lss"]),
                "gema_eta_k": float(env["eta_k"]),
                "gema_dkn_mpc": float(env["dkn"]),
            }
        )
        eligible.append(_serialize_predictor(value))
    eligible = sorted(eligible, key=lambda row: str(row["plateifu"]))
    expected_eligible = config["sample"].get("expected_environment_eligible")
    if expected_eligible is not None and len(eligible) != int(expected_eligible):
        raise GravityItem30Error("environment-eligible Item 30 count changed")

    predictor_fields = tuple(eligible[0].keys())
    _write_tsv(paths["predictors"], eligible, predictor_fields)
    sample = _sample_manifest(config, eligible)
    _write_json(paths["sample_manifest"], sample)
    candidate = _candidate_manifest(config)
    _write_json(paths["candidate_manifest"], candidate)
    source_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item30-predictor-source-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "source": {
                "manga_predictor_path": str(config["sources"]["manga_predictor_source"]),
                "manga_predictor_sha256": _sha256_file(predictor_path),
                "manga_predictor_content_sha256": predictor_source["content_sha256"],
                "gema_url": config["sources"]["gema_url"],
                "gema_version": config["sources"]["gema_version"],
                "gema_sha256": _sha256_bytes(gema_payload),
                "gema_headers": gema_headers,
                "gema_hdus": [
                    config["sources"]["gema_hdu_completeness"],
                    config["sources"]["gema_hdu_environment"],
                ],
            },
            "counts": {
                "valid_manga_predictors": len(records),
                "fresh_after_veto": len(fresh),
                "environment_eligible": len(eligible),
                "exclusions": dict(sorted(exclusion_counts.items())),
                "response_columns_requested": 0,
                "response_rows_read": 0,
                "confirmation_values_read": 0,
                "paid_api_calls": 0,
            },
            "files": {
                "gema_catalog_sha256": _sha256_file(paths["gema_catalog"]),
                "predictors_sha256": _sha256_file(paths["predictors"]),
                "predecessor_coordinates_sha256": _sha256_file(paths["predecessor_coordinates"]),
                "predecessor_identities_sha256": _sha256_file(paths["predecessor_identities"]),
            },
            "claims": {
                "response_opened": False,
                "sample_target_blind": True,
                "confirmation_opened": False,
            },
        }
    )
    _write_json(paths["predictor_source_manifest"], source_manifest)
    return paths


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


def _parse_skyserver_csv(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig").strip()
    if not text:
        raise GravityItem30Error("empty SkyServer CSV")
    rows = list(csv.DictReader(io.StringIO(text)))
    if rows and "error_message" in rows[0]:
        raise GravityItem30Error("SkyServer error: " + str(rows[0]["error_message"]))
    return [{str(key): str(value or "").strip() for key, value in row.items()} for row in rows]


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "Item 30 sample manifest")
    exploration = sorted(
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "exploration"
    )
    confirmations = {
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if len(exploration) != int(config["sample"]["expected_exploration"]):
        raise GravityItem30Error("Item 30 exploration role count changed before response query")
    chunks = []
    all_rows: list[dict[str, str]] = []
    chunk_size = int(config["sources"]["response_chunk_size"])
    expected_columns = tuple(str(value) for value in config["sources"]["response_columns"])
    for begin in range(0, len(exploration), chunk_size):
        identities = exploration[begin : begin + chunk_size]
        query = _response_query(config, identities)
        payload, url = _skyserver_query(config, query)
        rows = _parse_skyserver_csv(payload)
        if rows and tuple(rows[0].keys()) != expected_columns:
            raise GravityItem30Error("MaNGA response schema changed")
        returned = {row["plateifu"] for row in rows}
        if returned & confirmations:
            raise GravityItem30Error("confirmation response entered Item 30 acquisition")
        if not returned <= set(identities):
            raise GravityItem30Error("unrequested MaNGA response entered Item 30")
        all_rows.extend(rows)
        chunks.append(
            {
                "begin": begin,
                "requested": len(identities),
                "returned": len(rows),
                "query_sha256": _sha256_bytes(query.encode()),
                "payload_sha256": _sha256_bytes(payload),
                "url_sha256": _sha256_bytes(url.encode()),
            }
        )
    if len({row["plateifu"] for row in all_rows}) != len(all_rows):
        raise GravityItem30Error("duplicate MaNGA response row")
    all_rows = sorted(all_rows, key=lambda row: row["plateifu"])
    _write_tsv(paths["exploration_responses"], all_rows, expected_columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item30-response-source-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "endpoint": config["sources"]["skyserver_endpoint"],
            "daptype": config["sources"]["daptype"],
            "response_columns": list(expected_columns),
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
        raise GravityItem30Error(f"invalid response value {key}") from error
    if not np.isfinite(value):
        raise GravityItem30Error(f"nonfinite response value {key}")
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
        raise GravityItem30Error("Item 30 response file changed")
    sample_rows = {
        str(row["plateifu"]): row for row in sample["objects"] if row["role"] == "exploration"
    }
    response_rows = {row["plateifu"]: row for row in _read_tsv(paths["exploration_responses"])}
    quality = config["quality"]
    valid: list[dict[str, Any]] = []
    failures = []
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
        except GravityItem30Error:
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
            "schema_version": "invariant-gravity-item30-extraction-1.0",
            "exploration_roles": len(sample_rows),
            "response_rows": len(response_rows),
            "quality_passing": len(valid),
            "quality_failures": failures,
            "confirmation_values_read": 0,
            "failed_identity_replacement": False,
        }
    )
    return valid, response_manifest, extraction


def _screening_predictors(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "dimensionless_potential": np.asarray(
            [float(row["dimensionless_potential"]) for row in rows], dtype=np.float64
        ),
        "density_msun_kpc3": np.asarray(
            [float(row["mean_stellar_density_msun_kpc3"]) for row in rows], dtype=np.float64
        ),
        "mass_msun": np.asarray(
            [float(row["stellar_mass_msun"]) for row in rows], dtype=np.float64
        ),
        "radius_kpc": np.asarray(
            [float(row["half_light_radius_kpc"]) for row in rows], dtype=np.float64
        ),
        "q_lss": np.asarray([float(row["gema_q_lss"]) for row in rows], dtype=np.float64),
        "eta_k": np.asarray([float(row["gema_eta_k"]) for row in rows], dtype=np.float64),
    }


def _baryonic_virial_prediction(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> np.ndarray:
    constant = float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
    answer = []
    for row in rows:
        n = float(row["sersic_index"])
        coefficient = max(0.25, 8.87 - 0.831 * n + 0.0241 * n * n)
        sigma_squared = (
            constant
            * float(row["stellar_mass_msun"])
            / (coefficient * float(row["half_light_radius_kpc"]))
        )
        answer.append(0.5 * math.log10(sigma_squared))
    return np.asarray(answer, dtype=np.float64)


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
        "balmer_mean": [0.5 * (float(row["hdelta_a"]) + float(row["hgamma_a"])) for row in rows],
        "log_specific_sfr": [float(row["log_specific_sfr"]) for row in rows],
        "signed_log_halpha_ew": [
            math.copysign(math.log10(1.0 + abs(float(row["halpha_ew"]))), float(row["halpha_ew"]))
            for row in rows
        ],
        "log_dimensionless_potential": [float(row["log_dimensionless_potential"]) for row in rows],
        "gema_q_lss": [float(row["gema_q_lss"]) for row in rows],
        "gema_eta_k": [float(row["gema_eta_k"]) for row in rows],
        "log_gema_dnn_mpc": [math.log10(float(row["gema_dnn_mpc"])) for row in rows],
        "gema_zcomp_5mpc_fraction": [float(row["gema_zcomp_5mpc_percent"]) / 100.0 for row in rows],
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
        (0, 3),
        (2, 3),
        (8, 2),
        (8, 0),
        (5, 0),
        (12, 13),
        (14, 2),
        (13, 0),
        (14, 0),
        (6, 0),
        (6, 1),
        (4, 3),
        (13, 14),
        (9, 8),
        (10, 8),
        (11, 8),
    )
    pieces.extend((features[:, left] * features[:, right])[:, None] for left, right in interactions)
    return np.column_stack(pieces)


def _ridge_oof(
    target: np.ndarray,
    folds: np.ndarray,
    design: np.ndarray,
    alpha: float,
    outer_folds: int,
) -> np.ndarray:
    prediction = np.empty_like(target)
    matrix = np.column_stack([np.ones(len(design)), design])
    penalty = np.eye(matrix.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    for fold in range(outer_folds):
        train = folds != fold
        held = folds == fold
        gram = matrix[train].T @ matrix[train] + penalty
        coefficients = np.linalg.solve(gram, matrix[train].T @ target[train])
        prediction[held] = matrix[held] @ coefficients
    return prediction


def _virial_oof(
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
) -> dict[str, np.ndarray]:
    features = _feature_matrix(rows, config)
    outer_folds = int(config["sample"]["outer_folds"])
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
    }


def _build_candidate_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    predictors = {key: xp.asarray(value) for key, value in _screening_predictors(rows).items()}
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(_candidate_delta_log10_sigma(config, arrays, predictors, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _screen_candidate_matrix(
    delta: Any,
    target: np.ndarray,
    base: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    prediction = np.empty_like(target)
    selected = []
    offsets = []
    raw_offsets = []
    training_mse = []
    bounds = config["physics"]["shared_mass_proxy_scale_bounds"]
    lower = 0.5 * math.log10(float(bounds[0]))
    upper = 0.5 * math.log10(float(bounds[1]))
    residual = target - base
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        values = xp.asarray(residual[train])
        candidate = delta[:, train]
        mean_candidate = xp.mean(candidate, axis=1)
        mean_candidate_squared = xp.mean(candidate * candidate, axis=1)
        mean_response = float(np.mean(residual[train]))
        mean_response_squared = float(np.mean(residual[train] ** 2))
        mean_product = candidate @ values / len(train)
        raw = mean_response - mean_candidate
        fitted = xp.clip(raw, lower, upper)
        mse = (
            mean_response_squared
            - 2.0 * mean_product
            + mean_candidate_squared
            - 2.0 * fitted * (mean_response - mean_candidate)
            + fitted * fitted
        )
        index = int(_to_numpy(xp.argmin(mse), xp))
        selected.append(index)
        raw_offsets.append(float(_to_numpy(raw[index], xp)))
        offsets.append(float(_to_numpy(fitted[index], xp)))
        training_mse.append(float(_to_numpy(mse[index], xp)))
        prediction[held] = base[held] + _to_numpy(delta[index, held], xp) + offsets[-1]
    return {
        "prediction": prediction,
        "selected_indices": selected,
        "log10_sigma_offsets": offsets,
        "raw_log10_sigma_offsets": raw_offsets,
        "training_mse": training_mse,
    }


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
    }
    if niche == 0:
        record["log10_phi_threshold"] = float(values["chameleon_threshold"][0])
    elif niche == 1:
        record["log10_density_threshold"] = float(values["symmetron_threshold"][0])
    elif niche == 2:
        record["log10_crossover_kpc"] = float(values["vainshtein_crossover"][0])
    else:
        record["log10_phi_threshold"] = float(values["hybrid_phi_threshold"][0])
        record["log10_density_threshold"] = float(values["hybrid_density_threshold"][0])
    return record


def _synthetic_controls(
    delta: Any,
    base: np.ndarray,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    xp: Any,
) -> dict[str, Any]:
    injection_results = []
    injection_indices = []
    for niche in range(4):
        indices = np.where(arrays["niche"] == niche)[0]
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
        raise GravityItem30Error("too few Item 30 response-complete galaxies")
    arrays, candidate_audit = _admissible_candidates(config)
    target = np.asarray([float(row["y_log10_sigma"]) for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64)
    expected_folds = set(range(int(config["sample"]["outer_folds"])))
    if set(folds.tolist()) != expected_folds:
        raise GravityItem30Error("Item 30 response-complete folds are incomplete")
    base = _baryonic_virial_prediction(rows, config)
    xp, backend, device = _backend()
    xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    delta = _build_candidate_matrix(config, arrays, rows, xp)
    xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    crosscheck = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["niche"]))
    cpu_delta = _candidate_delta_log10_sigma(
        config, arrays, _screening_predictors(rows), 0, crosscheck, np
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
    controls = _synthetic_controls(delta, base, folds, rows, config, arrays, xp)
    selected_records = [
        _candidate_record(index, config, arrays) for index in observed["selected_indices"]
    ]
    selected_niches = [int(arrays["niche"][index]) for index in observed["selected_indices"]]
    niche_counts = Counter(selected_niches)

    mass = np.asarray([float(row["stellar_mass_msun"]) for row in rows])
    potential = np.asarray([float(row["log_dimensionless_potential"]) for row in rows])
    q_lss = np.asarray([float(row["gema_q_lss"]) for row in rows])
    eta_k = np.asarray([float(row["gema_eta_k"]) for row in rows])
    slice_masks = {
        "low_mass": mass <= np.median(mass),
        "high_mass": mass > np.median(mass),
        "low_potential": potential <= np.median(potential),
        "high_potential": potential > np.median(potential),
        "low_external_tide": q_lss <= np.median(q_lss),
        "high_external_tide": q_lss > np.median(q_lss),
        "low_projected_density": eta_k <= np.median(eta_k),
        "high_projected_density": eta_k > np.median(eta_k),
    }
    slices = {}
    for label, mask in slice_masks.items():
        indices = np.where(mask)[0]
        value = {
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
        if value["improvement_vs_flexible"]
        >= float(gates["partial_minimum_slice_improvement_vs_flexible"])
    ]
    universal_pass = all(universal_gates.values())
    phenomenon_pass = all(phenomenon_gates.values())
    if not quality_pass:
        decision = "INCONCLUSIVE_ITEM30_QUALITY"
    elif universal_pass:
        decision = "PASS_ITEM30_EXPLORATION_UNIVERSAL"
    elif phenomenon_pass:
        decision = "NONPROMOTED_ITEM30_PHENOMENON_LEAD"
    elif partial_slices:
        decision = "SCOPED_ITEM30_PARTIAL_PATTERN_RETAINED"
    else:
        decision = "SCOPED_ITEM30_REJECT"
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
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item30-screening-result-1.0",
            "item": 30,
            "title": config["title"],
            "decision": scientific["decision"],
            "hypothesis": config["hypothesis"],
            "scientific": scientific,
            "compute": compute,
            "extraction": extraction,
            "theory": {
                "sources": config["sources"]["theory_sources"],
                "families": config["candidate_generator"]["niches"],
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
            "claim_boundary": [
                config["scope"]["claim_ceiling"],
                "The stellar mass, Sersic virial mapping, anisotropy, stellar populations, and GEMA projected environment proxies carry ordinary astrophysical and catalog systematics.",
                "GEMA Q_LSS is a baryonic tidal-strength proxy and eta_k is a projected-neighbor density; neither is a direct scalar-field value or a dark-matter-free three-dimensional potential reconstruction.",
                "This single integrated motion observable cannot establish gravitational slip, lensing, cluster behavior, a covariant completion, or an alternative to GR.",
                "A positive result is exploration evidence only; confirmations stay sealed and unchanged fresh replication is mandatory.",
                "Historical novelty is not inferred from a creativity label or behavioral non-equivalence.",
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
        {"schema_version": "invariant-gravity-item30-compute-1.0", **compute}
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
        "predecessor_audit",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "response_source_manifest",
        "compute_manifest",
    ):
        _verify_content_hash(_read_json(paths[key]), key)
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "Item 30 result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem30Error("checked Item 30 result opened confirmation responses")
    if bool(result["frozen_boundary"]["post_response_formula_generation"]):
        raise GravityItem30Error("checked Item 30 result contains post-response generation")
    if (
        _sha256_file(paths["exploration_responses"])
        != result["frozen_boundary"]["response_file_sha256"]
    ):
        raise GravityItem30Error("checked Item 30 response file changed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-predecessors")
    sub.add_parser("prepare-predictors")
    sub.add_parser("acquire-responses")
    sub.add_parser("run")
    sub.add_parser("validate-checked")
    sub.add_parser("show-candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "audit-predecessors":
        print(build_predecessor_audit(root)["predecessor_audit"].as_posix())
    elif args.command == "prepare-predictors":
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
