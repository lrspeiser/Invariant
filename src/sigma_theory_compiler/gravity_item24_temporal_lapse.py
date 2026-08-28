"""Frozen Item 24 temporal-metric search on fresh motion and photon-delay data."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import subprocess
import time
import urllib.error
import urllib.parse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    _angular_diameter_distances,
    _parse_vizier_tsv,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _angular_separation_arcsec,
    _backend,
    _download,
    _legacy_url,
    _nearest_legacy_row,
    _read_tsv,
    _to_numpy,
    _write_tsv,
)

CONFIG_PATH = Path("configs/gravity_item24_temporal_lapse_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item24_temporal_lapse.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem24Error(RuntimeError):
    """Raised when an Item 24 freeze, leakage, or replay invariant is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _content_hashed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_sha256", None)
    result["content_sha256"] = _sha256_bytes(_canonical_bytes(result))
    return result


def _verify_content_hash(payload: Mapping[str, Any], label: str) -> None:
    expected = payload.get("content_sha256")
    if not isinstance(expected, str):
        raise GravityItem24Error(f"{label} has no content hash")
    body = dict(payload)
    body.pop("content_sha256", None)
    if _sha256_bytes(_canonical_bytes(body)) != expected:
        raise GravityItem24Error(f"{label} content hash changed")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityItem24Error(f"expected JSON object: {path}")
    return value


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem24Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem24Error(f"{label} is not an ancestor of HEAD")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if (
        config.get("schema_version")
        != "invariant-gravity-item24-temporal-lapse-config-1.0"
        or int(config.get("item", -1)) != 24
    ):
        raise GravityItem24Error("unexpected Item 24 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem24Error("stable gravity goal changed")
    if int(config["candidate_generator"]["raw_candidate_cells"]) != 262144:
        raise GravityItem24Error("raw candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem24Error("post-response candidates entered Item 24")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem24Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem24Error("paid calls are outside Item 24")
    policy = config["discovery_policy"]
    if not bool(policy["equal_initial_viability"]):
        raise GravityItem24Error("equal-viability policy changed")
    if not bool(policy["age_or_history_is_not_privileged"]):
        raise GravityItem24Error("history was privileged")
    for relative, digest in config["dependency_sha256"].items():
        if _sha256_file(root / str(relative)) != str(digest):
            raise GravityItem24Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    value.pop("implementation_correction_commit", None)
    value.pop("implementation_correction_scope", None)
    return _sha256_bytes(_canonical_bytes(value))


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen) != _contract_digest(config):
        raise GravityItem24Error("scientific contract differs from frozen commit")
    module_commit = str(config.get("implementation_correction_commit", commit))
    if module_commit.startswith("TO_BE_BOUND"):
        raise GravityItem24Error("implementation correction has not been bound")
    _require_ancestor(root, module_commit, "implementation correction")
    module = _git(root, "show", f"{module_commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(module, bytes) or _sha256_bytes(module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem24Error("Item 24 module differs from scientific freeze")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "galaxy_predictors",
        "lens_predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "galaxy_exploration_responses",
        "lens_exploration_responses",
        "response_source_manifest",
        "compute_manifest",
    )
    return {key: base / str(config["paths"][key]) for key in keys}


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in (
        "galaxy_predictors",
        "lens_predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        repo_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repo_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem24Error(f"{key} differs from sample freeze")


def _hmac_rank(key: str, value: str) -> str:
    return hmac.new(key.encode(), value.encode(), hashlib.sha256).hexdigest()


def _normal_identity(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _sexagesimal_ra(value: str) -> float:
    parts = str(value).strip().replace(":", " ").split()
    if len(parts) != 3:
        raise GravityItem24Error(f"invalid RA: {value}")
    return 15.0 * (float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0)


def _sexagesimal_dec(value: str) -> float:
    parts = str(value).strip().replace(":", " ").split()
    if len(parts) != 3:
        raise GravityItem24Error(f"invalid Dec: {value}")
    sign = -1.0 if parts[0].startswith("-") else 1.0
    return sign * (abs(float(parts[0])) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0)


def _query_url(source: str, columns: Sequence[str], **conditions: Any) -> str:
    params: dict[str, str] = {
        "-source": source,
        "-out": ",".join(columns),
        "-out.max": "unlimited",
    }
    params.update({str(key): str(value) for key, value in conditions.items()})
    return "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?" + urllib.parse.urlencode(params)


def _predecessor_exclusions(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    names: set[str] = set()
    agcs: set[int] = set()
    coordinates: list[tuple[float, float, str]] = []
    for path in sorted(root.glob(str(config["sources"]["predecessor_sample_glob"]))):
        try:
            manifest = _read_json(path)
        except (OSError, json.JSONDecodeError, GravityItem24Error):
            continue
        for row in manifest.get("objects", []):
            if not isinstance(row, Mapping):
                continue
            for key in ("name", "other_name", "normalized_identity"):
                if row.get(key):
                    names.add(_normal_identity(str(row[key])))
            if row.get("ugc") is not None:
                names.add(_normal_identity(f"UGC{int(row['ugc']):05d}"))
            if row.get("agc") is not None:
                agcs.add(int(row["agc"]))
            for ra_key, dec_key in (
                ("ra_deg", "dec_deg"),
                ("ra", "dec"),
                ("catalog_ra_deg", "catalog_dec_deg"),
            ):
                if row.get(ra_key) is not None and row.get(dec_key) is not None:
                    try:
                        coordinates.append(
                            (float(row[ra_key]), float(row[dec_key]), path.parent.name)
                        )
                    except (TypeError, ValueError):
                        pass
    return {"names": names, "agcs": agcs, "coordinates": coordinates}


def _minimum_separation(
    ra: float, dec: float, coordinates: Sequence[tuple[float, float, str]]
) -> float:
    nearby = [
        _angular_separation_arcsec(ra, dec, old_ra, old_dec)
        for old_ra, old_dec, _ in coordinates
        if abs(dec - old_dec) < 1.0 and abs(ra - old_ra) * math.cos(math.radians(dec)) < 2.0
    ]
    return min(nearby, default=1.0e9)


def _alfalfa_safe_rows(
    rows: Sequence[Mapping[str, str]], root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exclusions = _predecessor_exclusions(root, config)
    quality = config["alfalfa_quality"]
    accepted: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw in rows:
        counts["catalog_rows"] += 1
        try:
            agc = int(raw["AGC"])
            ra = _sexagesimal_ra(raw["RAO"])
            dec = _sexagesimal_dec(raw["DEO"])
            distance = float(raw["Dist"])
            distance_error = float(raw["e_Dist"])
            log_hi = float(raw["logMHI"])
            log_hi_error = float(raw["e_logMHI"])
            snr = float(raw["SNR"])
            velocity = float(raw["Vhel"])
            hi_code = int(raw["HI"])
        except (KeyError, TypeError, ValueError, GravityItem24Error):
            counts["incomplete_predictors"] += 1
            continue
        if hi_code != int(quality["required_HI_code"]):
            counts["HI_code"] += 1
            continue
        if snr < float(quality["minimum_SNR"]):
            counts["SNR"] += 1
            continue
        if not float(quality["minimum_distance_Mpc"]) <= distance <= float(
            quality["maximum_distance_Mpc"]
        ):
            counts["distance"] += 1
            continue
        if distance_error / distance > float(quality["maximum_fractional_distance_error"]):
            counts["distance_error"] += 1
            continue
        if log_hi_error > float(quality["maximum_log_HI_mass_error_dex"]):
            counts["HI_mass_error"] += 1
            continue
        if not float(quality["minimum_log_HI_mass"]) <= log_hi <= float(
            quality["maximum_log_HI_mass"]
        ):
            counts["HI_mass"] += 1
            continue
        if velocity < float(quality["minimum_heliocentric_velocity_km_s"]):
            counts["velocity"] += 1
            continue
        identity_set = {
            _normal_identity(str(raw.get("Name", ""))),
            _normal_identity(f"AGC{agc}"),
            _normal_identity(f"UGC{agc:05d}") if agc < 100000 else "",
        }
        if agc in exclusions["agcs"] or any(
            value and value in exclusions["names"] for value in identity_set
        ):
            counts["predecessor_name_or_AGC"] += 1
            continue
        separation = _minimum_separation(ra, dec, exclusions["coordinates"])
        if separation <= float(config["sources"]["predecessor_coordinate_veto_arcsec"]):
            counts["predecessor_coordinate"] += 1
            continue
        accepted.append(
            {
                "agc": agc,
                "name": str(raw.get("Name", "")).strip(),
                "ra_deg": ra,
                "dec_deg": dec,
                "heliocentric_velocity_km_s": velocity,
                "HI_flux_Jy_km_s": float(raw["HIflux"]),
                "HI_flux_error_Jy_km_s": float(raw["e_HIflux"]),
                "HI_SNR": snr,
                "distance_Mpc": distance,
                "distance_error_Mpc": distance_error,
                "log_HI_mass": log_hi,
                "log_HI_mass_error": log_hi_error,
                "minimum_predecessor_separation_arcsec": separation,
            }
        )
    counts["safe_eligible"] = len(accepted)
    return accepted, {
        "counts": dict(sorted(counts.items())),
        "predecessor_names": len(exclusions["names"]),
        "predecessor_AGCs": len(exclusions["agcs"]),
        "predecessor_coordinates": len(exclusions["coordinates"]),
    }


def _prelegacy_selection(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: float(row["log_HI_mass"]))
    strata = int(config["sample"]["galaxy_mass_strata"])
    count = int(config["alfalfa_quality"]["prelegacy_objects_per_mass_stratum"])
    selected: list[dict[str, Any]] = []
    for stratum in range(strata):
        begin = round(stratum * len(ordered) / strata)
        end = round((stratum + 1) * len(ordered) / strata)
        ranked = sorted(
            ordered[begin:end],
            key=lambda row: _hmac_rank(
                "invariant-item24-temporal-lapse-prelegacy-v1", str(row["agc"])
            ),
        )[:count]
        for row in ranked:
            row["mass_stratum"] = stratum
        selected.extend(ranked)
    return selected


def _axis_ratio(tractor: Mapping[str, Any]) -> float:
    ellipticity = min(
        math.hypot(float(tractor["shape_e1"]), float(tractor["shape_e2"])), 0.999
    )
    return (1.0 - ellipticity) / (1.0 + ellipticity)


def _galaxy_predictor(
    identity: Mapping[str, Any], tractor: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    distance_kpc = float(identity["distance_Mpc"]) * 1000.0
    modulus = 5.0 * math.log10(distance_kpc * 100.0)
    magnitudes: dict[str, float] = {}
    for band in ("g", "r", "z"):
        corrected = float(tractor[f"flux_{band}"]) / float(tractor[f"mw_transmission_{band}"])
        magnitudes[band] = 22.5 - 2.5 * math.log10(corrected)
    absolute_z = magnitudes["z"] - modulus
    luminosity = 10.0 ** (
        -0.4
        * (absolute_z - float(config["physics"]["constants"]["absolute_z_solar_AB"]))
    )
    reff_kpc = (
        float(tractor["shape_r"])
        * float(config["physics"]["constants"]["arcsec_to_radian"])
        * distance_kpc
    )
    return {
        **dict(identity),
        "tractor_release": int(tractor["release"]),
        "tractor_brickid": int(tractor["brickid"]),
        "tractor_brickname": str(tractor["brickname"]),
        "tractor_objid": int(tractor["objid"]),
        "tractor_type": str(tractor["type"]),
        "tractor_ra_deg": float(tractor["ra"]),
        "tractor_dec_deg": float(tractor["dec"]),
        "center_separation_arcsec": float(tractor["center_separation_arcsec"]),
        "reff_arcsec": float(tractor["shape_r"]),
        "reff_kpc": reff_kpc,
        "axis_ratio": _axis_ratio(tractor),
        "sersic": float(tractor["sersic"]),
        "z_luminosity_Lsun": luminosity,
        "g_minus_r": magnitudes["g"] - magnitudes["r"],
        "r_minus_z": magnitudes["r"] - magnitudes["z"],
        "max_fracflux_grz": max(float(tractor[f"fracflux_{b}"]) for b in ("g", "r", "z")),
        "max_fracmasked_grz": max(
            float(tractor[f"fracmasked_{b}"]) for b in ("g", "r", "z")
        ),
        "shape_signal_to_noise": float(tractor["shape_signal_to_noise"]),
        "minimum_flux_signal_to_noise_grz": min(
            float(tractor[f"flux_{b}"]) * math.sqrt(float(tractor[f"flux_ivar_{b}"]))
            for b in ("g", "r", "z")
        ),
    }


def _coordinate_core(value: str) -> str:
    match = re.search(r"J?(\d{6})(?:\.\d+)?([+-]\d{6})(?:\.\d+)?", str(value).upper())
    return f"J{match.group(1)}{match.group(2)}" if match else _normal_identity(value)


def _coordinate_from_sdss_name(value: str) -> tuple[float, float] | None:
    match = re.search(
        r"J?(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)",
        str(value).upper(),
    )
    if match is None:
        return None
    ra = 15.0 * (int(match[1]) + int(match[2]) / 60.0 + float(match[3]) / 3600.0)
    dec = int(match[5]) + int(match[6]) / 60.0 + float(match[7]) / 3600.0
    return ra, dec if match[4] == "+" else -dec


def _lens_predictors(
    cosmo_rows: Sequence[Mapping[str, str]],
    sqls_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sqls = [
        (coordinate, row)
        for row in sqls_rows
        if (coordinate := _coordinate_from_sdss_name(str(row["SDSS"]))) is not None
    ]
    exposed = {_normal_identity(value) for value in config["scope"]["preview_exposed_lenses_excluded"]}
    results: list[dict[str, Any]] = []
    for row in cosmo_rows:
        name = str(row["Name"]).strip()
        if _normal_identity(name) in exposed:
            continue
        file_name = str(row["FileName"]).strip()
        if not file_name.startswith(str(config["sources"]["required_lens_light_curve_family"]) + "/"):
            continue
        ra = _sexagesimal_ra(str(row["RAJ2000"]))
        dec = _sexagesimal_dec(str(row["DEJ2000"]))
        ranked = sorted(
            (
                _angular_separation_arcsec(ra, dec, coordinate[0], coordinate[1]),
                candidate,
            )
            for coordinate, candidate in sqls
        )
        if not ranked or ranked[0][0] > 3.0:
            continue
        match = ranked[0][1]
        core = _coordinate_core(str(match["SDSS"]))
        try:
            values = {
                "z_source": float(match["zs"]),
                "z_lens": float(match["zl"]),
                "quasar_i_mag": float(match["imag"]),
                "image_separation_arcsec": float(match["thetaM"]),
                "image_flux_ratio": float(match["Irat"]),
                "lens_i_mag": float(match["Imag"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if not (
            values["z_source"] > values["z_lens"] > 0.0
            and values["image_separation_arcsec"] > 0.0
            and 0.0 < values["image_flux_ratio"] < 1.0
        ):
            continue
        results.append(
            {
                "name": name,
                "normalized_identity": _normal_identity(name),
                "coordinate_core": core,
                "ra_deg": ra,
                "dec_deg": dec,
                "light_curve_file": file_name,
                **values,
            }
        )
    return sorted(results, key=lambda row: str(row["name"]))


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["raw_candidate_cells"])
    per = int(config["discovery_policy"]["equal_raw_capacity_per_mechanism"])
    if count != 4 * per:
        raise GravityItem24Error("mechanism capacity is not equal")
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    arrays: dict[str, np.ndarray] = {
        "niche": np.repeat(np.arange(4, dtype=np.int8), per),
    }
    for key, values in (
        ("amplitude", generator["amplitudes"]),
        ("polarity", generator["polarities"]),
        ("transition_acceleration", generator["transition_acceleration_mps2"]),
        ("transition_power", generator["transition_powers"]),
        ("slip", generator["spatial_slip_coefficients"]),
        ("compactness_threshold", generator["compactness_thresholds"]),
        ("memory_multiplier", generator["memory_dynamical_time_multipliers"]),
        ("resonance_frequency", generator["resonance_log_frequencies"]),
        ("resonance_phase", generator["resonance_phases_rad"]),
    ):
        arrays[key] = random.integers(0, len(values), count, dtype=np.int16)
    return arrays


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    result = {"niche": xp.asarray(arrays["niche"][begin:end])}
    for array_key, config_key in (
        ("amplitude", "amplitudes"),
        ("polarity", "polarities"),
        ("transition_acceleration", "transition_acceleration_mps2"),
        ("transition_power", "transition_powers"),
        ("slip", "spatial_slip_coefficients"),
        ("compactness_threshold", "compactness_thresholds"),
        ("memory_multiplier", "memory_dynamical_time_multipliers"),
        ("resonance_frequency", "resonance_log_frequencies"),
        ("resonance_phase", "resonance_phases_rad"),
    ):
        choices = xp.asarray(generator[config_key], dtype=xp.float64)
        result[array_key] = choices[xp.asarray(arrays[array_key][begin:end])]
    return result


def _raw_candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def _mechanism_H(
    values: Mapping[str, Any], acceleration: Any, compactness: Any, tau_ratio: Any, xp: Any
) -> Any:
    niche = values["niche"][:, None]
    amplitude = values["polarity"][:, None] * values["amplitude"][:, None]
    screen = 1.0 / (
        1.0
        + (acceleration[None, :] / values["transition_acceleration"][:, None])
        ** values["transition_power"][:, None]
    )
    compact = 1.0 / (
        1.0
        + (compactness[None, :] / values["compactness_threshold"][:, None])
        ** values["transition_power"][:, None]
    )
    settled = 1.0 - xp.exp(
        -1.0 / xp.maximum(values["memory_multiplier"][:, None] * tau_ratio[None, :], 1e-30)
    )
    phase = (
        values["resonance_frequency"][:, None]
        * xp.log(1.0 / xp.maximum(tau_ratio[None, :], 1e-30))
        + values["resonance_phase"][:, None]
    )
    resonance = 0.5 * (1.0 + xp.cos(phase))
    factor = xp.where(niche == 0, 1.0, xp.where(niche == 1, compact, xp.where(niche == 2, settled, resonance)))
    return amplitude * screen * factor


def _responses_from_H(values: Mapping[str, Any], H: Any, xp: Any) -> tuple[Any, Any]:
    motion = 1.0 + H
    photon = 1.0 + 0.5 * (1.0 + values["slip"][:, None]) * H
    return motion, photon


def _admissible_candidates(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    values = _candidate_values(config, raw, 0, len(raw["niche"]), np)
    solar = config["physics"]["solar_system"]
    acceleration = np.asarray([float(solar["acceleration_m_s2"])])
    compactness = np.asarray([float(solar["compactness"])])
    tau_ratio = np.asarray([float(solar["dynamical_to_age_ratio"])])
    H_solar = _mechanism_H(values, acceleration, compactness, tau_ratio, np)
    local_motion, local_photon = _responses_from_H(values, H_solar, np)
    local_motion = local_motion[:, 0]
    local_photon = local_photon[:, 0]
    negative_safe = ~((values["polarity"] < 0.0) & (values["amplitude"] >= 1.0))
    finite_local = np.isfinite(local_motion) & np.isfinite(local_photon)
    local_ok = (
        (np.abs(local_motion - 1.0) <= float(config["gates"]["maximum_local_motion_fractional_deviation"]))
        & (np.abs(local_photon - 1.0) <= float(config["gates"]["maximum_local_photon_fractional_deviation"]))
    )
    # A response-independent physical-domain grid prevents hidden lapse sign failures.
    g_grid, c_grid, t_grid = np.meshgrid(
        np.logspace(-14, -8, 7), np.logspace(-10, -5, 6), np.logspace(-7, -2, 6), indexing="ij"
    )
    domain_ok = np.ones(len(raw["niche"]), dtype=bool)
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(raw["niche"]), batch):
        end = min(begin + batch, len(raw["niche"]))
        part = _candidate_values(config, raw, begin, end, np)
        H = _mechanism_H(part, g_grid.ravel(), c_grid.ravel(), t_grid.ravel(), np)
        motion, photon = _responses_from_H(part, H, np)
        domain_ok[begin:end] = (
            np.all(np.isfinite(motion) & (motion > 0.05) & (motion < 100.0), axis=1)
            & np.all(np.isfinite(photon) & (photon > 0.05) & (photon < 100.0), axis=1)
        )
    keep = negative_safe & finite_local & local_ok & domain_ok
    arrays = {key: value[keep] for key, value in raw.items()}
    counts = Counter(int(value) for value in arrays["niche"])
    return arrays, {
        "raw_cells": len(raw["niche"]),
        "raw_niche_counts": {str(index): int(np.count_nonzero(raw["niche"] == index)) for index in range(4)},
        "admissible_cells": len(arrays["niche"]),
        "admissible_niche_counts": {str(index): counts.get(index, 0) for index in range(4)},
        "raw_candidate_digest": _raw_candidate_digest(raw),
        "admissible_candidate_digest": _raw_candidate_digest(arrays),
        "maximum_admitted_local_motion_deviation": float(np.max(np.abs(local_motion[keep] - 1.0))),
        "maximum_admitted_local_photon_deviation": float(np.max(np.abs(local_photon[keep] - 1.0))),
        "causal_kernel_cells": int(np.count_nonzero(arrays["niche"] == 2)),
    }


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item24-temporal-candidates-1.0",
            "generator": config["candidate_generator"],
            "audit": audit,
            "responses_open_when_generated": False,
            "published_delay_answers_read": False,
        }
    )


def _build_sample(
    galaxy_predictors: Sequence[Mapping[str, Any]],
    lens_predictors: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    sample = config["sample"]
    role_key = str(sample["role_key"])
    fold_key = str(sample["fold_key"])
    objects: list[dict[str, Any]] = []
    galaxy_selected: list[dict[str, Any]] = []
    for stratum in range(int(sample["galaxy_mass_strata"])):
        eligible = [dict(row) for row in galaxy_predictors if int(row["mass_stratum"]) == stratum]
        ranked = sorted(eligible, key=lambda row: _hmac_rank(role_key, f"galaxy:{row['agc']}"))
        needed = int(sample["galaxies_per_stratum"])
        if len(ranked) < needed:
            raise GravityItem24Error(f"galaxy stratum {stratum} has {len(ranked)}, needs {needed}")
        chosen = ranked[:needed]
        confirmations = {
            int(row["agc"])
            for row in chosen[: int(sample["galaxy_confirmation_per_stratum"])]
        }
        galaxy_selected.extend(chosen)
        exploration = [row for row in chosen if int(row["agc"]) not in confirmations]
        exploration.sort(key=lambda row: _hmac_rank(fold_key, f"galaxy:{row['agc']}"))
        folds = {int(row["agc"]): index % int(sample["outer_folds"]) for index, row in enumerate(exploration)}
        for row in chosen:
            confirmation = int(row["agc"]) in confirmations
            objects.append(
                {
                    "lane": "galaxy_motion",
                    "identity": str(row["agc"]),
                    "display_name": str(row.get("name", "")) or f"AGC {row['agc']}",
                    "mass_stratum": stratum,
                    "role": "confirmation" if confirmation else "exploration",
                    "fold": None if confirmation else folds[int(row["agc"])],
                    "role_rank": _hmac_rank(role_key, f"galaxy:{row['agc']}"),
                    "response_read": False,
                }
            )
    lenses = sorted((dict(row) for row in lens_predictors), key=lambda row: float(row["z_lens"]))
    lens_selected: list[dict[str, Any]] = []
    lens_records: list[tuple[dict[str, Any], int, bool]] = []
    for stratum in range(int(sample["lens_redshift_strata"])):
        begin = round(stratum * len(lenses) / int(sample["lens_redshift_strata"]))
        end = round((stratum + 1) * len(lenses) / int(sample["lens_redshift_strata"]))
        group = sorted(
            lenses[begin:end], key=lambda row: _hmac_rank(role_key, f"lens:{row['name']}")
        )
        needed = int(sample["lenses_per_stratum"])
        if len(group) != needed:
            raise GravityItem24Error(f"lens stratum {stratum} has {len(group)}, needs {needed}")
        lens_selected.extend(group)
        confirmations = {
            str(row["name"]) for row in group[: int(sample["lens_confirmation_per_stratum"])]
        }
        for row in group:
            confirmation = str(row["name"]) in confirmations
            lens_records.append((row, stratum, confirmation))
    lens_exploration = sorted(
        (row for row, _, confirmation in lens_records if not confirmation),
        key=lambda row: _hmac_rank(fold_key, f"lens:{row['name']}"),
    )
    lens_folds = {
        str(row["name"]): index % int(sample["outer_folds"])
        for index, row in enumerate(lens_exploration)
    }
    for row, stratum, confirmation in lens_records:
        objects.append(
            {
                "lane": "photon_delay",
                "identity": str(row["name"]),
                "display_name": str(row["name"]),
                "redshift_stratum": stratum,
                "role": "confirmation" if confirmation else "exploration",
                "fold": None if confirmation else lens_folds[str(row["name"])],
                "role_rank": _hmac_rank(role_key, f"lens:{row['name']}"),
                "response_read": False,
            }
        )
    objects.sort(key=lambda row: (str(row["lane"]), str(row["identity"])))
    counts = Counter(f"{row['lane']}:{row['role']}" for row in objects)
    fold_counts = Counter(
        f"{row['lane']}:{row['fold']}" for row in objects if row["role"] == "exploration"
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item24-temporal-sample-1.0",
            "selection_rule": sample["rule"],
            "objects": objects,
            "counts": dict(sorted(counts.items())),
            "fold_counts": dict(sorted(fold_counts.items())),
            "targets_read_when_frozen": False,
            "published_delay_answers_read": False,
        }
    )


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    if str(config["scientific_freeze_commit"]).startswith("TO_BE_BOUND"):
        raise GravityItem24Error("bind the scientific freeze before predictor preparation")
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["galaxy_predictors"].parent.mkdir(parents=True, exist_ok=True)
    source_receipts: list[dict[str, Any]] = []

    alfalfa_url = _query_url(
        str(config["sources"]["alfalfa_catalog"]).split()[1],
        config["sources"]["alfalfa_predictor_columns"],
    )
    alfalfa_body, alfalfa_headers = _download(alfalfa_url)
    alfalfa_rows = _parse_vizier_tsv(alfalfa_body, config["sources"]["alfalfa_predictor_columns"])
    safe_rows, exclusion_audit = _alfalfa_safe_rows(alfalfa_rows, root, config)
    prelegacy = _prelegacy_selection(safe_rows, config)

    def fetch_legacy(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, list[str], dict[str, Any]]:
        url = _legacy_url(config, float(row["ra_deg"]), float(row["dec_deg"]))
        try:
            body, headers = _download(url)
            tractor, failures = _nearest_legacy_row(
                body, float(row["ra_deg"]), float(row["dec_deg"]), config
            )
            receipt = {
                "agc": int(row["agc"]),
                "url": url,
                "sha256": _sha256_bytes(body),
                "bytes": len(body),
                "etag": headers.get("etag"),
                "last_modified": headers.get("last-modified"),
                "failures": failures,
            }
            return dict(row), tractor, failures, receipt
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as error:  # pragma: no cover
            return dict(row), None, [f"download:{type(error).__name__}"], {
                "agc": int(row["agc"]), "url": url, "error": type(error).__name__
            }

    galaxy_predictors: list[dict[str, Any]] = []
    legacy_receipts: list[dict[str, Any]] = []
    legacy_failures: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=16) as executor:
        for index, (identity, tractor, failures, receipt) in enumerate(
            executor.map(fetch_legacy, prelegacy), 1
        ):
            legacy_receipts.append(receipt)
            if tractor is None or failures:
                legacy_failures.update(failures)
            else:
                predictor = _galaxy_predictor(identity, tractor, config)
                q = float(predictor["axis_ratio"])
                if not float(config["predictor_quality"]["minimum_axis_ratio"]) <= q <= float(
                    config["predictor_quality"]["maximum_axis_ratio"]
                ):
                    legacy_failures["axis_ratio"] += 1
                else:
                    galaxy_predictors.append(predictor)
            if index % 50 == 0:
                print(f"Item 24 Legacy predictors {index}/{len(prelegacy)}", flush=True)

    cosmo_url = _query_url(
        str(config["sources"]["cosmograil_catalog"]).split()[1],
        config["sources"]["cosmograil_query_columns"],
    )
    cosmo_body, cosmo_headers = _download(cosmo_url)
    cosmo_rows = _parse_vizier_tsv(cosmo_body, config["sources"]["cosmograil_query_columns"])
    sqls_rows: list[dict[str, str]] = []
    sqls_receipts: list[dict[str, Any]] = []
    for source in config["sources"]["sqls_catalogs"]:
        url = _query_url(str(source), config["sources"]["sqls_predictor_columns"])
        body, headers = _download(url)
        sqls_rows.extend(_parse_vizier_tsv(body, config["sources"]["sqls_predictor_columns"]))
        sqls_receipts.append(
            {"source": source, "url": url, "sha256": _sha256_bytes(body), "bytes": len(body), "etag": headers.get("etag")}
        )
    lens_predictors = _lens_predictors(cosmo_rows, sqls_rows, config)
    if len(cosmo_rows) != int(config["sources"]["expected_cosmograil_rows"]):
        raise GravityItem24Error(f"COSMOGRAIL row count changed: {len(cosmo_rows)}")
    if len(lens_predictors) != int(
        config["sources"]["expected_safe_sqls_lcab_matches_after_preview_exclusion"]
    ):
        raise GravityItem24Error(f"safe lens count changed: {len(lens_predictors)}")

    galaxy_columns = sorted({key for row in galaxy_predictors for key in row})
    lens_columns = sorted({key for row in lens_predictors for key in row})
    _write_tsv(paths["galaxy_predictors"], galaxy_predictors, galaxy_columns)
    _write_tsv(paths["lens_predictors"], lens_predictors, lens_columns)
    source_receipts.extend(
        [
            {
                "source": config["sources"]["alfalfa_catalog"],
                "url": alfalfa_url,
                "sha256": _sha256_bytes(alfalfa_body),
                "bytes": len(alfalfa_body),
                "etag": alfalfa_headers.get("etag"),
            },
            {
                "source": config["sources"]["cosmograil_catalog"],
                "url": cosmo_url,
                "sha256": _sha256_bytes(cosmo_body),
                "bytes": len(cosmo_body),
                "etag": cosmo_headers.get("etag"),
            },
            *sqls_receipts,
        ]
    )
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item24-temporal-predictors-1.0",
            "response_columns_queried": [],
            "published_delay_answers_read": False,
            "source_receipts": source_receipts,
            "alfalfa_exclusion_audit": exclusion_audit,
            "prelegacy_selected": len(prelegacy),
            "legacy_quality_eligible": len(galaxy_predictors),
            "legacy_failure_counts": dict(sorted(legacy_failures.items())),
            "legacy_query_receipts": legacy_receipts,
            "safe_lens_predictors": len(lens_predictors),
            "galaxy_predictor_file": {"path": paths["galaxy_predictors"].relative_to(root).as_posix(), "sha256": _sha256_file(paths["galaxy_predictors"]), "rows": len(galaxy_predictors)},
            "lens_predictor_file": {"path": paths["lens_predictors"].relative_to(root).as_posix(), "sha256": _sha256_file(paths["lens_predictors"]), "rows": len(lens_predictors)},
        }
    )
    sample_manifest = _build_sample(galaxy_predictors, lens_predictors, config)
    candidate_manifest = _candidate_manifest(config)
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], candidate_manifest)
    return paths


def _parse_lcab(data: bytes) -> dict[str, np.ndarray]:
    rows: list[tuple[float, float, float, float, float, str]] = []
    for line in data.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            rows.append(
                (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), str(parts[5]))
            )
        except ValueError:
            continue
    if not rows:
        raise GravityItem24Error("empty lcab light curve")
    return {
        "time": np.asarray([row[0] for row in rows]),
        "a": np.asarray([row[1] for row in rows]),
        "ea": np.asarray([row[2] for row in rows]),
        "b": np.asarray([row[3] for row in rows]),
        "eb": np.asarray([row[4] for row in rows]),
        "telescope": np.asarray([row[5] for row in rows], dtype=object),
    }


def _season_ids(time_values: np.ndarray, gap: float) -> np.ndarray:
    order = np.argsort(time_values)
    ids = np.zeros(len(time_values), dtype=np.int32)
    season = 0
    previous = float(time_values[order[0]])
    for index in order:
        current = float(time_values[index])
        if current - previous > gap:
            season += 1
        ids[index] = season
        previous = current
    return ids


def _delay_score(
    curve: Mapping[str, np.ndarray], lag: float, indices: np.ndarray, config: Mapping[str, Any]
) -> tuple[float, int]:
    time_values = curve["time"]
    query = time_values[indices] + lag
    position = np.searchsorted(time_values, query)
    valid = (position > 0) & (position < len(time_values))
    left = np.clip(position - 1, 0, len(time_values) - 1)
    right = np.clip(position, 0, len(time_values) - 1)
    gap = time_values[right] - time_values[left]
    valid &= gap <= float(config["delay_estimator"]["maximum_interpolation_gap_days"])
    if np.count_nonzero(valid) < 20:
        return float("inf"), int(np.count_nonzero(valid))
    idx = indices[valid]
    q = query[valid]
    lo, hi = left[valid], right[valid]
    fraction = (q - time_values[lo]) / np.maximum(time_values[hi] - time_values[lo], 1e-12)
    interp_b = curve["b"][lo] * (1.0 - fraction) + curve["b"][hi] * fraction
    interp_var = curve["eb"][lo] ** 2 * (1.0 - fraction) ** 2 + curve["eb"][hi] ** 2 * fraction**2
    residual = curve["a"][idx] - interp_b
    variance = curve["ea"][idx] ** 2 + interp_var + 1e-5
    # Telescope offsets plus one global slow microlensing slope are nuisance-only.
    telescopes = sorted({str(value) for value in curve["telescope"][idx]})
    centered_time = (time_values[idx] - np.mean(time_values[idx])) / max(np.ptp(time_values[idx]), 1.0)
    design = np.column_stack(
        [*[np.asarray(curve["telescope"][idx] == telescope, dtype=float) for telescope in telescopes], centered_time]
    )
    weight = 1.0 / variance
    coefficient = np.linalg.lstsq(
        design.T @ (weight[:, None] * design) + np.eye(design.shape[1]) * 1e-8,
        design.T @ (weight * residual),
        rcond=None,
    )[0]
    centered = residual - design @ coefficient
    delta = float(config["delay_estimator"]["huber_delta_magnitude"])
    absolute = np.abs(centered)
    huber = np.where(absolute <= delta, 0.5 * centered**2, delta * (absolute - 0.5 * delta))
    return float(np.average(huber, weights=weight)), len(centered)


def _estimate_delay(data: bytes, config: Mapping[str, Any], seed_offset: int = 0) -> dict[str, Any]:
    curve = _parse_lcab(data)
    estimator = config["delay_estimator"]
    lags = np.arange(
        float(estimator["lag_min_days"]),
        float(estimator["lag_max_days"]) + 0.5 * float(estimator["lag_step_days"]),
        float(estimator["lag_step_days"]),
    )
    all_indices = np.arange(len(curve["time"]), dtype=np.int32)
    scores = np.asarray([_delay_score(curve, lag, all_indices, config)[0] for lag in lags])
    best_index = int(np.argmin(scores))
    best_lag = float(lags[best_index])
    zero_index = int(np.argmin(np.abs(lags)))
    seasons = _season_ids(curve["time"], float(estimator["season_gap_days"]))
    unique_seasons = np.unique(seasons)
    random = np.random.Generator(
        np.random.PCG64(int(estimator["bootstrap_seed"]) + int(seed_offset))
    )
    bootstrap: list[float] = []
    for _ in range(int(estimator["bootstrap_trials"])):
        chosen = random.choice(unique_seasons, size=len(unique_seasons), replace=True)
        indices = np.concatenate([np.where(seasons == value)[0] for value in chosen])
        trial_scores = np.asarray([_delay_score(curve, lag, indices, config)[0] for lag in lags])
        if np.any(np.isfinite(trial_scores)):
            bootstrap.append(float(lags[int(np.argmin(trial_scores))]))
    bootstrap_values = np.asarray(bootstrap)
    median = float(np.median(bootstrap_values)) if len(bootstrap_values) else float("nan")
    mad = (
        float(np.median(np.abs(bootstrap_values - median))) if len(bootstrap_values) else float("inf")
    )
    variability_ratio = min(
        float(np.std(curve["a"]) / max(np.median(curve["ea"]), 1e-6)),
        float(np.std(curve["b"]) / max(np.median(curve["eb"]), 1e-6)),
    )
    zero_score = float(scores[zero_index])
    score_improvement = (zero_score - float(scores[best_index])) / zero_score if zero_score > 0 else 0.0
    quality = {
        "minimum_rows": len(curve["time"]) >= int(estimator["minimum_rows"]),
        "minimum_time_span": float(np.ptp(curve["time"])) >= float(estimator["minimum_time_span_days"]),
        "minimum_variability": variability_ratio >= float(estimator["minimum_variability_to_error_ratio"]),
        "nonzero_delay": abs(best_lag) >= float(estimator["minimum_absolute_delay_days"]),
        "bootstrap_precision": mad <= float(estimator["maximum_bootstrap_MAD_days"]),
        "beats_zero_lag": score_improvement >= float(estimator["minimum_best_vs_zero_score_improvement"]),
    }
    return {
        "delay_days": abs(best_lag),
        "signed_delay_days": best_lag,
        "bootstrap_median_signed_days": median,
        "bootstrap_MAD_days": mad,
        "score_best": float(scores[best_index]),
        "score_zero": zero_score,
        "score_improvement_vs_zero": score_improvement,
        "rows": len(curve["time"]),
        "time_span_days": float(np.ptp(curve["time"])),
        "seasons": len(unique_seasons),
        "variability_to_error_ratio": variability_ratio,
        "quality_gates": quality,
        "quality_pass": all(quality.values()),
        "bootstrap_delay_sha256": _sha256_bytes(np.asarray(bootstrap_values, dtype="<f8").tobytes()),
    }


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "sample manifest")
    galaxy_roles = {
        int(row["identity"]): row
        for row in sample["objects"]
        if row["lane"] == "galaxy_motion" and row["role"] == "exploration"
    }
    lens_roles = {
        str(row["identity"]): row
        for row in sample["objects"]
        if row["lane"] == "photon_delay" and row["role"] == "exploration"
    }

    def fetch_width(agc: int) -> tuple[dict[str, Any], dict[str, Any]]:
        url = _query_url(
            str(config["sources"]["alfalfa_catalog"]).split()[1],
            config["sources"]["alfalfa_response_columns"],
            AGC=agc,
        )
        body, headers = _download(url)
        rows = _parse_vizier_tsv(body, config["sources"]["alfalfa_response_columns"])
        exact = [row for row in rows if int(row["AGC"]) == agc]
        response: dict[str, Any] = {"agc": agc, "W50_km_s": "", "W50_error_km_s": ""}
        if len(exact) == 1:
            response["W50_km_s"] = exact[0].get("W50", "")
            response["W50_error_km_s"] = exact[0].get("e_W50", "")
        return response, {
            "agc": agc, "url": url, "sha256": _sha256_bytes(body), "bytes": len(body), "etag": headers.get("etag")
        }

    galaxy_responses: list[dict[str, Any]] = []
    galaxy_receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for response, receipt in executor.map(fetch_width, sorted(galaxy_roles)):
            galaxy_responses.append(response)
            galaxy_receipts.append(receipt)
    galaxy_responses.sort(key=lambda row: int(row["agc"]))
    _write_tsv(paths["galaxy_exploration_responses"], galaxy_responses, ["agc", "W50_km_s", "W50_error_km_s"])

    lens_predictors = {row["name"]: row for row in _read_tsv(paths["lens_predictors"])}

    def fetch_curve(item: tuple[int, str]) -> tuple[dict[str, Any], dict[str, Any]]:
        index, name = item
        predictor = lens_predictors[name]
        url = str(config["sources"]["cosmograil_raw_base_url"]) + str(predictor["light_curve_file"])
        body, headers = _download(url)
        estimate = _estimate_delay(body, config, index)
        return {"name": name, **estimate}, {
            "name": name,
            "url": url,
            "sha256": _sha256_bytes(body),
            "bytes": len(body),
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
            "published_delay_answer_read": False,
        }

    lens_responses: list[dict[str, Any]] = []
    lens_receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for response, receipt in executor.map(fetch_curve, enumerate(sorted(lens_roles))):
            lens_responses.append(response)
            lens_receipts.append(receipt)
    lens_responses.sort(key=lambda row: str(row["name"]))
    lens_columns = [
        "name", "delay_days", "signed_delay_days", "bootstrap_median_signed_days", "bootstrap_MAD_days",
        "score_best", "score_zero", "score_improvement_vs_zero", "rows", "time_span_days", "seasons",
        "variability_to_error_ratio", "quality_pass", "bootstrap_delay_sha256",
    ]
    _write_tsv(paths["lens_exploration_responses"], lens_responses, lens_columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item24-temporal-responses-1.0",
            "galaxy_exploration_queries": galaxy_receipts,
            "lens_exploration_queries": lens_receipts,
            "galaxy_response_file": {"path": paths["galaxy_exploration_responses"].relative_to(root).as_posix(), "sha256": _sha256_file(paths["galaxy_exploration_responses"]), "rows": len(galaxy_responses)},
            "lens_response_file": {"path": paths["lens_exploration_responses"].relative_to(root).as_posix(), "sha256": _sha256_file(paths["lens_exploration_responses"]), "rows": len(lens_responses)},
            "confirmation_queries": 0,
            "confirmation_values_read": 0,
            "published_delay_answers_read": False,
            "post_response_candidate_cells": 0,
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["response_source_manifest"]


def _cosmic_age_Gyr(redshift: float, config: Mapping[str, Any]) -> float:
    cosmology = config["physics"]["cosmology"]
    h0 = float(cosmology["H0_km_s_Mpc"])
    omega_m = float(cosmology["omega_m"])
    omega_l = float(cosmology["omega_lambda"])
    scale = np.logspace(-5, math.log10(1.0 / (1.0 + redshift)), 20000)
    integrand = 1.0 / (scale * np.sqrt(omega_m / scale**3 + omega_l))
    integral = float(np.trapezoid(integrand, scale))
    h0_seconds = h0 / 3.0856775814913673e19
    return integral / h0_seconds / float(config["physics"]["constants"]["seconds_per_Gyr"])


def _system_features(
    mass_msun: float, radius_kpc: float, redshift: float, config: Mapping[str, Any]
) -> tuple[float, float, float]:
    gravitational = float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
    c = float(config["physics"]["constants"]["c_km_s"])
    acceleration_km2_s2_kpc = gravitational * mass_msun / radius_kpc**2
    acceleration = acceleration_km2_s2_kpc * 3.240779289444365e-14
    compactness = gravitational * mass_msun / (radius_kpc * c**2)
    tau_seconds = math.sqrt(radius_kpc**3 / (gravitational * mass_msun)) * 3.0856775814913673e16
    age_seconds = _cosmic_age_Gyr(redshift, config) * float(
        config["physics"]["constants"]["seconds_per_Gyr"]
    )
    return acceleration, compactness, tau_seconds / age_seconds


def _inclination_sine(axis_ratio: float, config: Mapping[str, Any]) -> float:
    q0 = float(config["physics"]["constants"]["intrinsic_disk_axis_ratio"])
    cos2 = max(0.0, min(1.0, (axis_ratio**2 - q0**2) / (1.0 - q0**2)))
    return math.sqrt(max(1.0 - cos2, 1e-6))


def _sis_delay_days(predictor: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    zl, zs = float(predictor["z_lens"]), float(predictor["z_source"])
    dl, ds, dls = _angular_diameter_distances(zl, zs, config)
    theta_e = 0.5 * float(predictor["image_separation_arcsec"]) * float(
        config["physics"]["constants"]["arcsec_to_radian"]
    )
    ratio = float(predictor["image_flux_ratio"])
    beta = theta_e * (1.0 - ratio) / (1.0 + ratio)
    d_delta_kpc = dl * ds / dls
    c = float(config["physics"]["constants"]["c_km_s"])
    seconds = (1.0 + zl) * d_delta_kpc * 3.0856775814913673e16 / c * 2.0 * theta_e * beta
    return seconds / float(config["physics"]["constants"]["seconds_per_day"])


def _load_evaluation_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    response_manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(sample, "sample manifest")
    _verify_content_hash(response_manifest, "response manifest")
    galaxy_predictors = {int(row["agc"]): row for row in _read_tsv(paths["galaxy_predictors"])}
    lens_predictors = {str(row["name"]): row for row in _read_tsv(paths["lens_predictors"])}
    galaxy_responses = {int(row["agc"]): row for row in _read_tsv(paths["galaxy_exploration_responses"])}
    lens_responses = {str(row["name"]): row for row in _read_tsv(paths["lens_exploration_responses"])}
    roles = {(row["lane"], str(row["identity"])): row for row in sample["objects"]}
    rows: list[dict[str, Any]] = []
    for agc, response in galaxy_responses.items():
        role = roles[("galaxy_motion", str(agc))]
        predictor = galaxy_predictors[agc]
        try:
            width = float(response["W50_km_s"])
            width_error = float(response["W50_error_km_s"])
        except (TypeError, ValueError):
            continue
        q = float(predictor["axis_ratio"])
        sin_i = _inclination_sine(q, config)
        rotation = width / (2.0 * sin_i)
        radius = float(config["physics"]["trigger_radius_in_effective_radii"]) * float(
            predictor["reff_kpc"]
        )
        stellar = float(config["physics"]["fiducial_stellar_mass_to_light_z"]) * float(
            predictor["z_luminosity_Lsun"]
        )
        gas = float(config["physics"]["constants"]["helium_HI_multiplier"]) * 10.0 ** float(
            predictor["log_HI_mass"]
        )
        enclosed = (
            float(config["physics"]["galaxy_stellar_enclosed_fraction"]) * stellar
            + float(config["physics"]["galaxy_HI_enclosed_fraction"]) * gas
        )
        gravitational = float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
        vbar = math.sqrt(gravitational * enclosed / radius)
        acceleration, compactness, tau_ratio = _system_features(enclosed, radius, 0.0, config)
        rows.append(
            {
                "lane": 0,
                "identity": f"AGC {agc}",
                "fold": int(role["fold"]),
                "target": 2.0 * math.log(rotation / vbar),
                "acceleration": acceleration,
                "compactness": compactness,
                "tau_ratio": tau_ratio,
                "response_fractional_error": width_error / width,
                "log_mass": math.log(enclosed),
                "log_radius": math.log(radius),
                "log_distance": math.log(float(predictor["distance_Mpc"])),
                "axis_ratio": q,
                "color": float(predictor["r_minus_z"]),
                "quality": math.log1p(float(predictor["HI_SNR"])),
            }
        )
    for name, response in lens_responses.items():
        if str(response["quality_pass"]).lower() not in {"true", "1"}:
            continue
        role = roles[("photon_delay", name)]
        predictor = lens_predictors[name]
        zl, zs = float(predictor["z_lens"]), float(predictor["z_source"])
        dl, _, _ = _angular_diameter_distances(zl, zs, config)
        radius = (
            0.5
            * float(predictor["image_separation_arcsec"])
            * float(config["physics"]["constants"]["arcsec_to_radian"])
            * dl
        )
        luminosity_distance = dl * (1.0 + zl) ** 2
        modulus = 5.0 * math.log10(luminosity_distance * 100.0)
        absolute_i = float(predictor["lens_i_mag"]) - modulus + 2.5 * math.log10(1.0 + zl)
        luminosity = 10.0 ** (
            -0.4
            * (absolute_i - float(config["physics"]["constants"]["absolute_i_solar_AB"]))
        )
        mass = float(config["physics"]["fiducial_lens_mass_to_light_i"]) * luminosity
        acceleration, compactness, tau_ratio = _system_features(mass, radius, zl, config)
        baseline = _sis_delay_days(predictor, config)
        delay = float(response["delay_days"])
        rows.append(
            {
                "lane": 1,
                "identity": name,
                "fold": int(role["fold"]),
                "target": math.log(delay / baseline),
                "acceleration": acceleration,
                "compactness": compactness,
                "tau_ratio": tau_ratio,
                "response_fractional_error": max(
                    float(response["bootstrap_MAD_days"]) / max(delay, 1.0), 0.05
                ),
                "log_mass": math.log(mass),
                "log_radius": math.log(radius),
                "log_distance": math.log(dl),
                "axis_ratio": float(predictor["image_flux_ratio"]),
                "color": zl,
                "quality": math.log1p(float(response["variability_to_error_ratio"])),
                "source_redshift": zs,
                "image_separation_arcsec": float(predictor["image_separation_arcsec"]),
            }
        )
    rows.sort(key=lambda row: (int(row["lane"]), str(row["identity"])))
    return rows, response_manifest


def _candidate_log_mu(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], begin: int, end: int, xp: Any
) -> Any:
    values = _candidate_values(config, arrays, begin, end, xp)
    acceleration = xp.asarray([float(row["acceleration"]) for row in rows])
    compactness = xp.asarray([float(row["compactness"]) for row in rows])
    tau_ratio = xp.asarray([float(row["tau_ratio"]) for row in rows])
    H = _mechanism_H(values, acceleration, compactness, tau_ratio, xp)
    solar = config["physics"]["solar_system"]
    H_solar = _mechanism_H(
        values,
        xp.asarray([float(solar["acceleration_m_s2"])]),
        xp.asarray([float(solar["compactness"])]),
        xp.asarray([float(solar["dynamical_to_age_ratio"])]),
        xp,
    )
    motion, photon = _responses_from_H(values, H, xp)
    local_motion, local_photon = _responses_from_H(values, H_solar, xp)
    motion = motion / local_motion
    photon = photon / local_photon
    lane = xp.asarray([int(row["lane"]) for row in rows])[None, :]
    selected = xp.where(lane == 0, motion, photon)
    return xp.log(selected)


def _build_log_mu_matrix(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], xp: Any
) -> Any:
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(_candidate_log_mu(config, arrays, rows, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _lane_mean_mse(residual: Any, lane: np.ndarray, xp: Any) -> Any:
    values = []
    for channel in (0, 1):
        indices = np.where(lane == channel)[0]
        values.append(xp.mean(residual[:, indices] ** 2, axis=1))
    return 0.5 * (values[0] + values[1])


def _screen_log_mu(
    log_mu: Any, target: np.ndarray, folds: np.ndarray, lane: np.ndarray, config: Mapping[str, Any], xp: Any
) -> dict[str, Any]:
    target_device = xp.asarray(target)
    prediction = np.empty_like(target)
    selected: list[int] = []
    offsets: list[dict[str, float]] = []
    training_mse: list[float] = []
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        residual = target_device[None, train] - log_mu[:, train]
        centered = residual.copy()
        fold_offsets: dict[int, Any] = {}
        for channel in (0, 1):
            local = np.where(lane[train] == channel)[0]
            offset = xp.mean(residual[:, local], axis=1)
            centered[:, local] -= offset[:, None]
            fold_offsets[channel] = offset
        mse = _lane_mean_mse(centered, lane[train], xp)
        index = int(_to_numpy(xp.argmin(mse), xp))
        selected.append(index)
        training_mse.append(float(_to_numpy(mse[index], xp)))
        offsets.append({str(channel): float(_to_numpy(fold_offsets[channel][index], xp)) for channel in (0, 1)})
        for channel in (0, 1):
            local_held = held[lane[held] == channel]
            prediction[local_held] = _to_numpy(log_mu[index, local_held], xp) + offsets[-1][str(channel)]
    return {"prediction": prediction, "selected_indices": selected, "offsets": offsets, "training_mse": training_mse}


def _screen_separate_formula(
    log_mu: Any,
    target: np.ndarray,
    folds: np.ndarray,
    lane: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    target_device = xp.asarray(target)
    prediction = np.empty_like(target)
    selected: dict[str, list[int]] = {"0": [], "1": []}
    for fold in range(int(config["sample"]["outer_folds"])):
        for channel in (0, 1):
            train = np.where((folds != fold) & (lane == channel))[0]
            held = np.where((folds == fold) & (lane == channel))[0]
            residual = target_device[None, train] - log_mu[:, train]
            offset = xp.mean(residual, axis=1)
            mse = xp.mean((residual - offset[:, None]) ** 2, axis=1)
            index = int(_to_numpy(xp.argmin(mse), xp))
            selected[str(channel)].append(index)
            prediction[held] = _to_numpy(log_mu[index, held], xp) + float(
                _to_numpy(offset[index], xp)
            )
    return {"prediction": prediction, "selected_indices_by_lane": selected}


def _feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row["log_mass"]), float(row["log_radius"]), float(row["log_distance"]),
                math.log(float(row["acceleration"])), math.log(float(row["compactness"])),
                math.log(float(row["tau_ratio"])), float(row["axis_ratio"]), float(row["color"]),
                float(row["quality"]),
            ]
            for row in rows
        ], dtype=np.float64,
    )


def _baseline_predictions(
    target: np.ndarray, folds: np.ndarray, lane: np.ndarray, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    calibrated = np.empty_like(target)
    flexible = np.empty_like(target)
    features = _feature_matrix(rows)
    alpha = float(config["evaluation"]["ridge_alpha"])
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        for channel in (0, 1):
            local_train = train[lane[train] == channel]
            local_held = held[lane[held] == channel]
            calibrated[local_held] = float(np.mean(target[local_train]))
            mean = features[local_train].mean(axis=0)
            scale = features[local_train].std(axis=0)
            scale[scale == 0.0] = 1.0
            design = np.column_stack([np.ones(len(local_train)), (features[local_train] - mean) / scale])
            held_design = np.column_stack([np.ones(len(local_held)), (features[local_held] - mean) / scale])
            penalty = np.diag([0.0] + [alpha] * features.shape[1])
            coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ target[local_train])
            flexible[local_held] = held_design @ coefficient
    return {"calibrated_GR": calibrated, "flexible_nuisance": flexible}


def _balanced_mse(target: np.ndarray, prediction: np.ndarray, lane: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is None:
        indices = np.arange(len(target))
    values = []
    for channel in (0, 1):
        local = indices[lane[indices] == channel]
        values.append(float(np.mean((target[local] - prediction[local]) ** 2)))
    return 0.5 * sum(values)


def _improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference if reference > 0.0 else float("-inf")


def _selected_cell(index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(arrays["niche"][index])
    return {
        "candidate_index": index,
        "niche": config["candidate_generator"]["niches"][niche],
        "amplitude": float(values["amplitude"][0]),
        "polarity": float(values["polarity"][0]),
        "transition_acceleration_mps2": float(values["transition_acceleration"][0]),
        "transition_power": float(values["transition_power"][0]),
        "spatial_slip_coefficient": float(values["slip"][0]),
        "compactness_threshold": float(values["compactness_threshold"][0]),
        "memory_dynamical_time_multiplier": float(values["memory_multiplier"][0]),
        "resonance_log_frequency": float(values["resonance_frequency"][0]),
        "resonance_phase_rad": float(values["resonance_phase"][0]),
    }


def _synthetic_controls(
    log_mu: Any, folds: np.ndarray, lane: np.ndarray, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], xp: Any
) -> dict[str, Any]:
    niche = np.where(arrays["niche"] == 2)[0]
    niche_variance = xp.var(log_mu[xp.asarray(niche)], axis=1)
    injection_index = int(niche[int(_to_numpy(xp.argmax(niche_variance), xp))])
    injection = _to_numpy(log_mu[injection_index], xp) + np.where(lane == 0, 0.2, -0.1)
    selected = _screen_log_mu(log_mu, injection, folds, lane, config, xp)
    selected_niches = [int(arrays["niche"][index]) for index in selected["selected_indices"]]
    gr_target = np.where(lane == 0, 0.2, -0.1)
    gr = _screen_log_mu(log_mu, gr_target, folds, lane, config, xp)
    amplitudes = [abs(_selected_cell(index, config, arrays)["amplitude"]) for index in gr["selected_indices"]]
    centered_variation = []
    for index in gr["selected_indices"]:
        values = _to_numpy(log_mu[index], xp)
        centered_variation.append(
            max(float(np.std(values[lane == channel])) for channel in (0, 1))
        )
    return {
        "injection_candidate_index": injection_index,
        "injection_selected_niches": selected_niches,
        "injection_exact_niche_recovered_all_folds": all(value == 2 for value in selected_niches),
        "GR_control_selected_amplitudes": amplitudes,
        "GR_control_selected_centered_log_response_std": centered_variation,
        "GR_control_prefers_material_response": any(value > 1e-4 for value in centered_variation),
    }


def _evaluate(
    root: Path, config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], record_compute: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    galaxy_count = sum(int(row["lane"]) == 0 for row in rows)
    lens_count = sum(int(row["lane"]) == 1 for row in rows)
    if galaxy_count < int(config["sample"]["minimum_valid_exploration_galaxies"]):
        raise GravityItem24Error(f"only {galaxy_count} valid exploration galaxies")
    if lens_count < int(config["sample"]["minimum_valid_exploration_lenses"]):
        raise GravityItem24Error(f"only {lens_count} valid exploration lenses")
    arrays, admissibility = _admissible_candidates(config)
    xp, backend, device = _backend()
    start = time.perf_counter()
    log_mu = _build_log_mu_matrix(config, arrays, rows, xp)
    sample_domain_valid = _to_numpy(xp.all(xp.isfinite(log_mu), axis=1), xp).astype(bool)
    removed_by_sample_domain = int(np.count_nonzero(~sample_domain_valid))
    log_mu = log_mu[xp.asarray(sample_domain_valid)]
    arrays = {key: value[sample_domain_valid] for key, value in arrays.items()}
    admissibility["sample_domain_nonfinite_cells_removed"] = removed_by_sample_domain
    admissibility["sample_domain_admissible_cells"] = len(arrays["niche"])
    admissibility["sample_domain_admissible_digest"] = _raw_candidate_digest(arrays)
    xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    cpu_count = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["niche"]))
    cpu = _candidate_log_mu(config, arrays, rows, 0, cpu_count, np)
    gpu = _to_numpy(log_mu[:cpu_count], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu - gpu)))
    target = np.asarray([float(row["target"]) for row in rows])
    folds = np.asarray([int(row["fold"]) for row in rows])
    lane = np.asarray([int(row["lane"]) for row in rows])
    controls = _synthetic_controls(log_mu, folds, lane, config, arrays, xp)
    start = time.perf_counter()
    selected = _screen_log_mu(log_mu, target, folds, lane, config, xp)
    baselines = _baseline_predictions(target, folds, lane, rows, config)
    separate_formula = _screen_separate_formula(log_mu, target, folds, lane, config, xp)
    baselines["separate_formula"] = separate_formula["prediction"]
    candidate_mse = _balanced_mse(target, selected["prediction"], lane)
    baseline_mse = {key: _balanced_mse(target, value, lane) for key, value in baselines.items()}
    observed = _improvement(baseline_mse["flexible_nuisance"], candidate_mse)
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    trials = int(config["evaluation"]["permutation_trials"])
    null_statistics: list[float] = []
    for trial in range(trials):
        permuted = target.copy()
        for channel in (0, 1):
            indices = np.where(lane == channel)[0]
            permuted[indices] = target[random.permutation(indices)]
        null_selected = _screen_log_mu(log_mu, permuted, folds, lane, config, xp)
        null_base = _baseline_predictions(permuted, folds, lane, rows, config)["flexible_nuisance"]
        null_statistics.append(
            _improvement(
                _balanced_mse(permuted, null_base, lane),
                _balanced_mse(permuted, null_selected["prediction"], lane),
            )
        )
        if record_compute and (trial + 1) % 25 == 0:
            print(f"Item 24 selection-aware null {trial + 1}/{trials}", flush=True)
    xp.cuda.Stream.null.synchronize()
    screen_seconds = time.perf_counter() - start
    raw_p = (1 + sum(value >= observed for value in null_statistics)) / (trials + 1)
    guarded_p = 1.0 if observed <= 0.0 else raw_p
    cells = [_selected_cell(index, config, arrays) for index in selected["selected_indices"]]
    niche_counts = Counter(str(cell["niche"]["id"]) for cell in cells)
    channel_metrics: dict[str, Any] = {}
    for channel, label in ((0, "galaxy_motion"), (1, "photon_delay")):
        indices = np.where(lane == channel)[0]
        candidate = float(np.mean((target[indices] - selected["prediction"][indices]) ** 2))
        channel_metrics[label] = {
            "objects": len(indices),
            "candidate_mse": candidate,
            **{
                f"{key}_mse": float(np.mean((target[indices] - value[indices]) ** 2))
                for key, value in baselines.items()
            },
        }
        for key in baselines:
            channel_metrics[label][f"improvement_vs_{key}"] = _improvement(
                channel_metrics[label][f"{key}_mse"], candidate
            )
    strata: dict[str, Any] = {}
    for channel, label in ((0, "galaxy"), (1, "lens")):
        lane_indices = np.where(lane == channel)[0]
        for field in ("acceleration", "compactness", "tau_ratio"):
            values = np.asarray([float(row[field]) for row in rows])
            median = float(np.median(values[lane_indices]))
            for side, indices in (
                ("low", lane_indices[values[lane_indices] <= median]),
                ("high", lane_indices[values[lane_indices] > median]),
            ):
                candidate = float(np.mean((target[indices] - selected["prediction"][indices]) ** 2))
                gr = float(np.mean((target[indices] - baselines["calibrated_GR"][indices]) ** 2))
                flexible = float(np.mean((target[indices] - baselines["flexible_nuisance"][indices]) ** 2))
                strata[f"{label}_{field}_{side}"] = {
                    "objects": len(indices), "candidate_mse": candidate, "calibrated_GR_mse": gr,
                    "flexible_nuisance_mse": flexible, "improvement_vs_calibrated_GR": _improvement(gr, candidate),
                    "improvement_vs_flexible_nuisance": _improvement(flexible, candidate),
                }
    same_niche = max(niche_counts.values(), default=0)
    gates = config["gates"]
    universal_gates = {
        "minimum_valid_galaxies": galaxy_count >= int(config["sample"]["minimum_valid_exploration_galaxies"]),
        "minimum_valid_lenses": lens_count >= int(config["sample"]["minimum_valid_exploration_lenses"]),
        "confirmations_sealed": True,
        "published_delay_answers_unread": True,
        "post_response_candidates_zero": int(config["candidate_generator"]["post_response_cells"]) == 0,
        "local_motion_limit": admissibility["maximum_admitted_local_motion_deviation"] <= float(gates["maximum_local_motion_fractional_deviation"]),
        "local_photon_limit": admissibility["maximum_admitted_local_photon_deviation"] <= float(gates["maximum_local_photon_fractional_deviation"]),
        "causal_positive_domain": bool(np.all(np.isfinite(_to_numpy(log_mu, xp)))),
        "synthetic_injection": bool(controls["injection_exact_niche_recovered_all_folds"]),
        "known_GR_control": not bool(controls["GR_control_prefers_material_response"]),
        "joint_improvement_vs_calibrated_GR": _improvement(baseline_mse["calibrated_GR"], candidate_mse) >= float(gates["minimum_joint_improvement_vs_calibrated_GR"]),
        "joint_improvement_vs_flexible_nuisance": observed > float(gates["minimum_joint_improvement_vs_flexible_nuisance"]),
        "both_channels_improve_vs_calibrated_GR": all(value["improvement_vs_calibrated_GR"] > float(gates["minimum_each_channel_improvement_vs_calibrated_GR"]) for value in channel_metrics.values()),
        "all_broad_halves_improve_vs_calibrated_GR": all(value["improvement_vs_calibrated_GR"] > float(gates["minimum_each_broad_half_improvement_vs_calibrated_GR"]) for value in strata.values()),
        "selection_aware_permutation": guarded_p <= float(gates["maximum_selection_aware_permutation_p"]),
        "same_niche_stability": same_niche >= int(gates["minimum_same_niche_folds"]),
    }
    phenomenon_channels = {
        key: value["improvement_vs_flexible_nuisance"]
        >= float(gates["phenomenon_minimum_channel_improvement_vs_flexible"])
        for key, value in channel_metrics.items()
    }
    phenomenon_gates = {
        "at_least_one_channel_beats_flexible_by_margin": any(phenomenon_channels.values()),
        "selection_aware_significance": guarded_p <= float(gates["maximum_selection_aware_permutation_p"]),
        "same_niche_stability": same_niche >= int(gates["minimum_same_niche_folds"]),
        "pipeline_controls": bool(controls["injection_exact_niche_recovered_all_folds"]) and not bool(controls["GR_control_prefers_material_response"]),
        "fresh_replication_required_before_paper": True,
    }
    universal_advance = all(universal_gates.values())
    phenomenon_lead = all(value for key, value in phenomenon_gates.items() if key != "fresh_replication_required_before_paper")
    counterexamples = [
        str(row["identity"])
        for index, row in enumerate(rows)
        if (target[index] - selected["prediction"][index]) ** 2
        > (target[index] - baselines["flexible_nuisance"][index]) ** 2
    ]
    training_evaluations = len(arrays["niche"]) * sum(
        int(np.count_nonzero(folds != fold)) for fold in range(int(config["sample"]["outer_folds"]))
    )
    compute = {
        "schema_version": "invariant-gravity-item24-temporal-compute-1.0",
        "backend": backend, "device": device, "numpy_version": np.__version__,
        "cupy_version": getattr(xp, "__version__", None),
        "raw_candidate_cells": int(config["candidate_generator"]["raw_candidate_cells"]),
        "generic_domain_admissible_candidate_cells": admissibility["admissible_cells"],
        "admissible_candidate_cells": len(arrays["niche"]),
        "sample_domain_nonfinite_cells_removed": removed_by_sample_domain,
        "objects": len(rows), "galaxies": galaxy_count, "lenses": lens_count,
        "candidate_observable_matrix_values": int(np.prod(log_mu.shape)),
        "candidate_training_residual_evaluations_observed": training_evaluations,
        "candidate_training_residual_evaluations_with_nulls": training_evaluations * (trials + 1),
        "matrix_build_seconds_observed": matrix_seconds, "screen_and_null_seconds_observed": screen_seconds,
        "cpu_gpu_max_absolute_log_mu_difference": cpu_gpu_max,
    }
    scientific = {
        "decision": "PASS_ITEM24_TEMPORAL_EXPLORATION" if universal_advance else "REJECT_ITEM24_TEMPORAL_EXPLORATION",
        "track_decisions": {
            "universal_gravity": "ADVANCE" if universal_advance else "DO_NOT_ADVANCE",
            "phenomenon_publication": "REPLICATION_LEAD" if phenomenon_lead else "NO_EMPIRICAL_LEAD",
            "paper_claim_now": False,
        },
        "counts": {
            "valid_exploration_galaxies": galaxy_count, "valid_exploration_lenses": lens_count,
            "raw_candidate_cells": int(config["candidate_generator"]["raw_candidate_cells"]),
            "generic_domain_admissible_candidate_cells": admissibility["admissible_cells"],
            "admissible_candidate_cells": len(arrays["niche"]),
            "sample_domain_nonfinite_cells_removed": removed_by_sample_domain,
            "post_response_candidate_cells": 0,
            "permutation_trials": trials, "counterexamples_vs_flexible": len(counterexamples),
        },
        "primary_metrics": {
            "candidate_balanced_mse": candidate_mse,
            **{f"{key}_balanced_mse": value for key, value in baseline_mse.items()},
            "improvement_vs_calibrated_GR": _improvement(baseline_mse["calibrated_GR"], candidate_mse),
            "improvement_vs_flexible_nuisance": observed,
            "selection_aware_raw_permutation_p": raw_p,
            "selection_aware_guarded_permutation_p": guarded_p,
        },
        "channel_metrics": channel_metrics,
        "stratum_metrics": strata,
        "outer_selections": [
            {
                "fold": fold, "cell": cells[fold], "training_mse": selected["training_mse"][fold],
                "lane_offsets": selected["offsets"][fold],
                "heldout_objects": [str(rows[index]["identity"]) for index in np.where(folds == fold)[0]],
            }
            for fold in range(int(config["sample"]["outer_folds"]))
        ],
        "selection_stability": {"niche_counts": dict(sorted(niche_counts.items())), "maximum_same_niche_folds": same_niche, "exact_candidate_indices": selected["selected_indices"]},
        "separate_formula_selection": separate_formula["selected_indices_by_lane"],
        "null_distribution": {
            "statistic": "OOF balanced improvement versus fixed flexible nuisance models",
            "observed": observed, "minimum": min(null_statistics), "median": float(np.median(null_statistics)),
            "maximum": max(null_statistics), "sha256": _sha256_bytes(np.asarray(null_statistics, dtype="<f8").tobytes()),
        },
        "controls": {**controls, **admissibility, "cpu_gpu_max_absolute_log_mu_difference": cpu_gpu_max},
        "universal_gates": universal_gates,
        "phenomenon_publication_gates": {**phenomenon_gates, "channel_margin_pass": phenomenon_channels},
        "counterexample_identities_vs_flexible": counterexamples,
    }
    del log_mu
    xp.get_default_memory_pool().free_all_blocks()
    return scientific, compute


def _build_receipt(
    root: Path, config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], response_manifest: Mapping[str, Any], scientific: Mapping[str, Any], compute: Mapping[str, Any]
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    predictors = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item24-temporal-lapse-receipt-1.0",
            "item": 24, "title": config["title"], "hypothesis": config["hypothesis"],
            "discovery_policy": config["discovery_policy"],
            "mathematical_definition": {
                "metric": config["theory"]["metric"], "potential_closure": config["theory"]["potential_closure"],
                "motion_response": config["theory"]["motion_response"], "photon_response": config["theory"]["photon_response"],
                "system_clocks": config["theory"]["system_clocks"], "causality": config["theory"]["causality"],
                "galaxy_target": "2 log[(W50/(2 sin i))/sqrt(G M_b(<2.2Re)/(2.2Re))]",
                "photon_target": "log[abs(raw-light-curve delay)/SIS geometry-plus-Shapiro delay]",
            },
            "provenance_and_creativity_labels": config["candidate_generator"]["niches"],
            "equivalence_audit": {
                "known_rewrite_control": "screened_static_lapse",
                "non_equivalent_search_regions": ["compactness_dependent_clock", "causal_settling_memory", "screened_temporal_resonance"],
                "historical_novelty_claimed": False,
                "boundaries": config["candidate_generator"]["equivalence_boundaries"],
            },
            "stability_scope": config["theory"]["claim_limits"], "claim_boundary": config["scope"]["claim_ceiling"],
            "source_bindings": {
                "predictor_manifest_sha256": predictors["content_sha256"], "sample_manifest_sha256": sample["content_sha256"],
                "candidate_manifest_sha256": candidates["content_sha256"], "response_manifest_sha256": response_manifest["content_sha256"],
                "galaxy_response_file_sha256": response_manifest["galaxy_response_file"]["sha256"],
                "lens_response_file_sha256": response_manifest["lens_response_file"]["sha256"],
                "observable_lineage": config["sources"]["observable_lineage"],
            },
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"], "sample_freeze_commit": config["sample_freeze_commit"],
                "implementation_correction_commit": config["implementation_correction_commit"],
                "implementation_correction_scope": config["implementation_correction_scope"],
                "stable_goal_sha256": config["stable_goal_sha256"], "confirmation_opened": False,
                "confirmation_response_values_read": 0, "published_delay_answers_read": False,
                "post_response_formula_generation": False,
            },
            "baselines": {
                "calibrated_GR": config["evaluation"]["baseline_calibrated_GR"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
                "separate_formula": config["evaluation"]["baseline_separate_formula"],
            },
            "scientific_result": scientific,
            "compute_and_api_cost": {**compute, "paid_model_calls": 0, "paid_api_spend_usd": 0.0},
            "counterexamples_and_limitations": [
                "ALFALFA W50 is a global HI linewidth, not a resolved rotation curve; inclination and turbulent broadening can dominate individual residuals.",
                "Legacy optical light and a universal stellar mass-to-light scale are incomplete baryon models, especially for dust, molecular gas, and extended HI geometry.",
                "The SIS photon baseline uses catalog image separation and flux ratio and is not a precision mass-sheet, environment, or microlensing model.",
                "The raw-delay estimator is deliberately independent of published answers but simpler than COSMOGRAIL's spline and Gaussian-process analyses.",
                "Only quality-passing exploration delays enter; the frozen quality gates may make the lens lane small.",
                "One positive lane or regime can be retained as a paper-track lead, but no paper claim is allowed without unchanged fresh replication.",
                "No sealed confirmation response or published delay answer was opened.",
            ],
            "exact_next_action": "Preserve every Item 24 branch and result under the equal-viability two-track policy, then advance the numbered roadmap to Item 25 gravity coupled to time while separately preregistering unchanged fresh replication for any phenomenon lead.",
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(), "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(), "module_sha256": _sha256_file(root / MODULE_PATH),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
                "valid_object_identities": [str(row["identity"]) for row in rows],
            },
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_evaluation_rows(root, config)
    scientific, compute_raw = _evaluate(root, config, rows, record_compute=True)
    paths = _source_paths(root, config)
    compute = _content_hashed(compute_raw)
    _write_json(paths["compute_manifest"], compute)
    result = _build_receipt(root, config, rows, response_manifest, scientific, compute)
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, result)
    return result_path


def validate_result(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    for key in ("predictor_source_manifest", "sample_manifest", "candidate_manifest", "response_source_manifest", "compute_manifest"):
        _verify_content_hash(_read_json(paths[key]), key)
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem24Error("result opened confirmation data")
    if bool(result["frozen_boundary"]["published_delay_answers_read"]):
        raise GravityItem24Error("result read published delay answers")
    if bool(result["equivalence_audit"]["historical_novelty_claimed"]):
        raise GravityItem24Error("result made an unauthorized novelty claim")
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-predictors")
    sub.add_parser("acquire-responses")
    sub.add_parser("run")
    sub.add_parser("validate")
    sub.add_parser("show-candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "prepare-predictors":
        print(prepare_predictors(root)["sample_manifest"].as_posix())
    elif args.command == "acquire-responses":
        print(acquire_responses(root).as_posix())
    elif args.command == "run":
        print(run_experiment(root).as_posix())
    elif args.command == "validate":
        print(validate_result(root).as_posix())
    else:
        print(json.dumps(_candidate_manifest(load_config(root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
