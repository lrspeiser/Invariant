"""Frozen direct-cooling/weak-lensing experiment for gravity roadmap Item 15."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_item11_neargalcat_external_field import _ridge_fit, _ridge_predict
from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item15_accept_lc2_timescale_ratios_v2.json"
SCIENTIFIC_FREEZE_COMMIT = "e007ac9e3b13237f650482e899d87fd6706d1c27"
SAMPLE_FREEZE_COMMIT = "18b1dc9d1c0813e304621ea368956c93a1cbe3c2"


class GravityItem15AcceptLC2Error(RuntimeError):
    """Raised when the frozen Item 15 attempt-2 boundary or replay drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem15AcceptLC2Error("non-finite Item 15 metric")
    return f"{float(value):.12e}"


def _serialize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return _metric(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def _content_hashed(value: dict[str, Any]) -> dict[str, Any]:
    result = _serialize(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _validate_content_hash(value: Mapping[str, Any], label: str) -> None:
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem15AcceptLC2Error(f"{label} content hash changed")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item15-accept-lc2-timescale-config-2.0"
    ):
        raise GravityItem15AcceptLC2Error("unexpected Item 15 attempt-2 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem15AcceptLC2Error("stable gravity roadmap changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem15AcceptLC2Error("Item 15 attempt-1 receipt changed")
    previous = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if previous.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem15AcceptLC2Error("Item 15 attempt-1 content changed")
    if previous.get("decision") != predecessor["required_decision"]:
        raise GravityItem15AcceptLC2Error("Item 15 attempt 1 did not require attempt 2")
    for entry in config["independence"]["prior_cluster_sources"]:
        if _sha256_file(root / entry["path"]) != entry["file_sha256"]:
            raise GravityItem15AcceptLC2Error(
                f"prior cluster identity source changed: {entry['path']}"
            )
    authorization = config["authorization"]
    forbidden = (
        "paid_model_calls_allowed",
        "lc2_response_query_allowed_before_sample_freeze",
        "confirmation_response_query_allowed",
        "post_response_candidate_generation_allowed",
        "lc2_m500_as_predictor_allowed",
        "accept_hydrostatic_mass_as_predictor_allowed",
        "object_identity_as_numeric_feature_allowed",
    )
    if any(bool(authorization[key]) for key in forbidden):
        raise GravityItem15AcceptLC2Error("Item 15 attempt-2 authorization changed")
    if not bool(authorization["exploration_response_query_allowed_after_sample_freeze"]):
        raise GravityItem15AcceptLC2Error("exploration response authorization changed")
    access = config["prefreeze_access"]
    if int(access["lc2_m500_response_rows_read_for_frozen_attempt2_objects"]) != 0:
        raise GravityItem15AcceptLC2Error("LC2 response entered the prefreeze design")
    incident = access["incident"]
    if not (
        int(incident["numeric_response_rows_accidentally_displayed"]) == 6
        and int(incident["paper_sample_objects"]) == 100
        and int(incident["accept_lc2_overlap_objects_excluded_by_20_arcmin_coordinate_rule"])
        == len(incident["overlapping_accept_names"])
        == 47
    ):
        raise GravityItem15AcceptLC2Error("Herbonnet response-access remedy changed")
    generator = config["candidate_generator"]
    if int(generator["candidate_cells"]) != 262144 or int(generator["post_response_cells"]) != 0:
        raise GravityItem15AcceptLC2Error("candidate boundary changed")
    sample = config["sample"]
    rows = sample["eligible_lc2_rows"]
    if not (
        len(rows) == int(sample["eligible_objects"]) == 23
        and int(sample["exploration_objects"]) == 18
        and int(sample["confirmation_objects"]) == 5
    ):
        raise GravityItem15AcceptLC2Error("sample count boundary changed")
    names = [str(row[0]) for row in rows]
    if len(names) != len(set(names)):
        raise GravityItem15AcceptLC2Error("eligible ACCEPT identities duplicated")
    if set(names) != set(config["sources"]["accept"]["profile_snapshot_by_accept_name"]):
        raise GravityItem15AcceptLC2Error("profile snapshot inventory changed")
    if int(sample["prefreeze_predictor_audit"]["response_values_read"]) != 0:
        raise GravityItem15AcceptLC2Error("response entered predictor audit")
    claims = config["claim_boundaries"]
    if not bool(claims["direct_hot_gas_cooling_time_tested"]):
        raise GravityItem15AcceptLC2Error("direct cooling coverage was removed")
    if any(
        bool(value) for key, value in claims.items() if key != "direct_hot_gas_cooling_time_tested"
    ):
        raise GravityItem15AcceptLC2Error("Item 15 attempt-2 config overclaims")
    return config


def _require_scientific_freeze() -> None:
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem15AcceptLC2Error("Item 15 attempt-2 scientific freeze is not bound")


def _require_sample_freeze() -> None:
    if SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem15AcceptLC2Error("Item 15 attempt-2 sample freeze is not bound")


def _fetch(url: str, *, attempts: int, timeout: int = 90) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if payload:
                return payload
            error = RuntimeError("empty response")
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(min(1.0 + attempt, 5.0))
    raise GravityItem15AcceptLC2Error(f"source acquisition failed: {url}: {error}")


def _normalize_name(value: str) -> str:
    text = value.upper().replace("ABELL", "A").replace("ZWICKY", "ZW")
    text = text.replace("ZWCL", "ZW").replace("RX J", "RXJ").replace("MACS J", "MACS")
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return re.sub(r"^A0+(\d+)$", r"A\1", text)


def _hms_to_degrees(value: str) -> float:
    hours, minutes, seconds = (float(part) for part in value.split(":"))
    return 15.0 * (hours + minutes / 60.0 + seconds / 3600.0)


def _dms_to_degrees(value: str) -> float:
    sign = -1.0 if value.startswith("-") else 1.0
    degrees, minutes, seconds = (float(part) for part in value.lstrip("+-").split(":"))
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def _separation_arcmin(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cosine = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * 60.0


def parse_accept_metadata(payload: bytes) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem15AcceptLC2Error("ACCEPT metadata is not UTF-8") from exc
    rows = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 13:
            raise GravityItem15AcceptLC2Error("unexpected ACCEPT metadata row")
        rows.append(
            {
                "accept_name": fields[0],
                "ra": _hms_to_degrees(fields[1]),
                "dec": _dms_to_degrees(fields[2]),
                "redshift": float(fields[3]),
                "k0": float(fields[4]),
                "k100": float(fields[5]),
                "entropy_alpha": float(fields[6]),
                "global_temperature_kev": float(fields[7]),
                "lbol_1e44_erg_s": float(fields[8]),
            }
        )
    return rows


PROFILE_FIELDS = (
    "radius_inner_mpc",
    "radius_outer_mpc",
    "electron_density_cm3",
    "electron_density_error_cm3",
    "entropy_interpolated_kev_cm2",
    "entropy_flat_kev_cm2",
    "entropy_error_kev_cm2",
    "pressure_interpolated",
    "pressure_flat",
    "pressure_error",
    "hydrostatic_mass",
    "hydrostatic_mass_error",
    "temperature_kev",
    "temperature_error_kev",
    "cooling_function",
    "cooling_time_isobaric_gyr",
    "cooling_time_isobaric_error_gyr",
    "cooling_time_isochoric_gyr",
    "cooling_time_isochoric_error_gyr",
)


def parse_accept_profile(payload: bytes) -> list[dict[str, float]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem15AcceptLC2Error("ACCEPT profile is not UTF-8") from exc
    rows = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 20:
            raise GravityItem15AcceptLC2Error(f"unexpected ACCEPT profile row width: {len(values)}")
        rows.append(dict(zip(PROFILE_FIELDS, map(float, values[1:]), strict=True)))
    rows.sort(key=lambda row: row["radius_inner_mpc"])
    if not rows:
        raise GravityItem15AcceptLC2Error("empty ACCEPT profile")
    return rows


def _shell_value(rows: Sequence[Mapping[str, float]], radius_mpc: float, field: str) -> float:
    enclosing = [
        row
        for row in rows
        if float(row["radius_inner_mpc"]) <= radius_mpc <= float(row["radius_outer_mpc"])
    ]
    if len(enclosing) != 1:
        raise GravityItem15AcceptLC2Error(
            f"radius {radius_mpc} Mpc has {len(enclosing)} enclosing ACCEPT shells"
        )
    value = float(enclosing[0][field])
    if not math.isfinite(value) or value <= 0:
        raise GravityItem15AcceptLC2Error(f"invalid ACCEPT {field} at {radius_mpc} Mpc")
    return value


def _gas_mass_msun(
    rows: Sequence[Mapping[str, float]], radius_mpc: float, config: Mapping[str, Any]
) -> float:
    constants = config["physical_constants"]
    mass_kg = 0.0
    for row in rows:
        lo = max(0.0, float(row["radius_inner_mpc"]))
        hi = min(radius_mpc, float(row["radius_outer_mpc"]))
        density = float(row["electron_density_cm3"])
        if hi <= lo or density <= 0:
            continue
        mpc_m = float(constants["mpc_m"])
        volume = 4.0 * math.pi / 3.0 * ((hi * mpc_m) ** 3 - (lo * mpc_m) ** 3)
        mass_kg += (
            float(constants["mean_mass_per_electron_proton_units"])
            * float(constants["proton_mass_kg"])
            * density
            * 1.0e6
            * volume
        )
    mass = mass_kg / float(constants["solar_mass_kg"])
    if not math.isfinite(mass) or mass <= 0:
        raise GravityItem15AcceptLC2Error("invalid integrated ACCEPT gas mass")
    return mass


def cosmic_age_gyr(redshift: float, config: Mapping[str, Any]) -> float:
    if not math.isfinite(redshift) or redshift < 0:
        raise GravityItem15AcceptLC2Error("invalid cluster redshift")
    constants = config["physical_constants"]
    omega_m = float(constants["omega_matter"])
    omega_l = float(constants["omega_lambda"])
    age = 2.0 * float(constants["hubble_time_gyr"]) / (3.0 * math.sqrt(omega_l))
    age *= math.asinh(math.sqrt(omega_l / omega_m) / (1.0 + redshift) ** 1.5)
    return age


def derive_cluster_features(
    profile: Sequence[Mapping[str, float]],
    metadata: Mapping[str, Any],
    lc2_metadata: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive predictor-only clocks; this function has no lensing-response argument."""

    constants = config["physical_constants"]
    age = cosmic_age_gyr(float(metadata["redshift"]), config)
    features: dict[str, Any] = {
        **metadata,
        **lc2_metadata,
        "log_k0_plus1": math.log10(1.0 + float(metadata["k0"])),
        "log_k100": math.log10(float(metadata["k100"])),
        "log_global_temperature": math.log10(float(metadata["global_temperature_kev"])),
        # ACCEPT publishes zero when the bolometric-luminosity entry is unavailable.
        # log1p preserves those objects and lets the baseline control missing/low Lbol
        # without converting the sentinel into a response-based exclusion.
        "log_lbol": math.log10(1.0 + float(metadata["lbol_1e44_erg_s"])),
        "cosmic_age_gyr": age,
        "log_cosmic_age_gyr": math.log10(age),
    }
    ratios: dict[str, list[float]] = {
        "cooling_freefall": [],
        "cooling_crossing": [],
        "cooling_cosmic": [],
        "crossing_freefall": [],
    }
    for radius_kpc in config["timescale_features"]["radii_kpc"]:
        suffix = str(int(float(radius_kpc)))
        radius_mpc = float(radius_kpc) / 1000.0
        ne = _shell_value(profile, radius_mpc, "electron_density_cm3")
        temperature = _shell_value(profile, radius_mpc, "temperature_kev")
        cooling = _shell_value(profile, radius_mpc, "cooling_time_isochoric_gyr")
        cooling_isobaric = _shell_value(profile, radius_mpc, "cooling_time_isobaric_gyr")
        if not math.isclose(cooling_isobaric / cooling, 5.0 / 3.0, rel_tol=2e-3):
            raise GravityItem15AcceptLC2Error("ACCEPT cooling-time equivalence changed")
        gas_mass = _gas_mass_msun(profile, radius_mpc, config)
        radius_m = radius_mpc * float(constants["mpc_m"])
        mass_kg = gas_mass * float(constants["solar_mass_kg"])
        freefall_seconds = math.sqrt(
            2.0 * radius_m**3 / (float(constants["gravitational_constant_si"]) * mass_kg)
        )
        sound_speed = math.sqrt(
            float(constants["adiabatic_index"])
            * temperature
            * float(constants["kev_joule"])
            / (
                float(constants["mean_particle_mass_proton_units"])
                * float(constants["proton_mass_kg"])
            )
        )
        crossing_seconds = radius_m / sound_speed
        freefall = freefall_seconds / float(constants["gyr_seconds"])
        crossing = crossing_seconds / float(constants["gyr_seconds"])
        values = {
            f"ne{suffix}": ne,
            f"temperature{suffix}_kev": temperature,
            f"tcool{suffix}_gyr": cooling,
            f"mgas{suffix}_msun": gas_mass,
            f"tff_baryon{suffix}_gyr": freefall,
            f"tsound{suffix}_gyr": crossing,
            f"log_ne{suffix}": math.log10(ne),
            f"log_temperature{suffix}": math.log10(temperature),
            f"log_tcool{suffix}": math.log10(cooling),
            f"log_mgas{suffix}": math.log10(gas_mass),
            f"log_tff_baryon{suffix}": math.log10(freefall),
            f"log_tsound{suffix}": math.log10(crossing),
            f"log_tcool_tff{suffix}": math.log10(cooling / freefall),
            f"log_tcool_tsound{suffix}": math.log10(cooling / crossing),
            f"log_tcool_cosmic{suffix}": math.log10(cooling / age),
            f"log_tsound_tff{suffix}": math.log10(crossing / freefall),
        }
        features.update(values)
        ratios["cooling_freefall"].append(values[f"log_tcool_tff{suffix}"])
        ratios["cooling_crossing"].append(values[f"log_tcool_tsound{suffix}"])
        ratios["cooling_cosmic"].append(values[f"log_tcool_cosmic{suffix}"])
        ratios["crossing_freefall"].append(values[f"log_tsound_tff{suffix}"])
    radial_log_span = math.log10(100.0 / 20.0)
    features["gas_concentration20_100"] = features["log_mgas20"] - features["log_mgas100"]
    features["cooling_gradient20_100"] = (
        features["log_tcool100"] - features["log_tcool20"]
    ) / radial_log_span
    features["temperature_gradient20_100"] = (
        features["log_temperature100"] - features["log_temperature20"]
    ) / radial_log_span
    features["entropy_contrast"] = features["log_k100"] - features["log_k0_plus1"]
    normalized = []
    fixed = config["evaluation"]["fixed_ratio_normalization"]
    for key in ("cooling_freefall", "cooling_crossing", "cooling_cosmic", "crossing_freefall"):
        center, scale = (float(value) for value in fixed[key])
        normalized.append((ratios[key][1] - center) / scale)
    array = np.asarray(normalized, dtype=np.float64)
    features["clock_hierarchy_span"] = float(np.max(array) - np.min(array))
    shifted = array - np.max(array)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    features["clock_hierarchy_entropy"] = float(
        -np.sum(weights * np.log(weights)) / math.log(len(weights))
    )
    author = str(lc2_metadata["lc2_author"])
    features["lc2_source_applegate"] = 1.0 if author == "applegate+14" else 0.0
    features["lc2_source_okabe"] = 1.0 if author.startswith("okabe") else 0.0
    features["lc2_source_merten"] = 1.0 if author == "merten+15" else 0.0
    numeric = [float(value) for value in features.values() if isinstance(value, (int, float))]
    if any(not math.isfinite(value) for value in numeric):
        raise GravityItem15AcceptLC2Error("non-finite derived cluster feature")
    return features


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    scale_min, scale_max = (float(value) for value in generator["scale_log_uniform"])
    power_min, power_max = (float(value) for value in generator["power_log_uniform"])
    return {
        "family": random.integers(0, len(generator["families"]), count, dtype=np.int8),
        "radius": random.integers(0, 3, count, dtype=np.int8),
        "threshold": random.uniform(*generator["threshold_uniform"], count),
        "scale": np.exp(random.uniform(math.log(scale_min), math.log(scale_max), count)),
        "power": np.exp(random.uniform(math.log(power_min), math.log(power_max), count)),
        "phase": random.uniform(*generator["phase_uniform"], count),
        "modulation": random.integers(0, len(generator["modulations"]), count, dtype=np.int8),
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("family", "radius", "threshold", "scale", "power", "phase", "modulation"):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _profile_payload(root: Path, accept_name: str, config: Mapping[str, Any]) -> tuple[bytes, str]:
    timestamp, original = config["sources"]["accept"]["profile_snapshot_by_accept_name"][
        accept_name
    ]
    url = f"https://web.archive.org/web/{timestamp}id_/{original}"
    cache = root / config["sources"]["accept"]["raw_cache"] / f"{accept_name}.dat"
    if cache.exists():
        payload = cache.read_bytes()
        parse_accept_profile(payload)
        return payload, url
    payload = _fetch(
        url,
        attempts=int(config["sources"]["accept"]["download_attempts"]),
    )
    parse_accept_profile(payload)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    return payload, url


def write_prepared_sources(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    _require_scientific_freeze()
    config = load_config(root)
    metadata_payload = _fetch(config["sources"]["accept"]["metadata_url"], attempts=4)
    metadata_rows = parse_accept_metadata(metadata_payload)
    if len(metadata_rows) != int(config["sources"]["accept"]["metadata_expected_rows"]):
        raise GravityItem15AcceptLC2Error("ACCEPT metadata row count changed")
    metadata_by_name = {str(row["accept_name"]): row for row in metadata_rows}
    records = []
    source_receipts = []
    eligible = config["sample"]["eligible_lc2_rows"]
    for accept_name, lc2_name, author, bibcode in eligible:
        if accept_name not in metadata_by_name:
            raise GravityItem15AcceptLC2Error(f"missing ACCEPT metadata: {accept_name}")
        payload, url = _profile_payload(root, accept_name, config)
        profile = parse_accept_profile(payload)
        feature = derive_cluster_features(
            profile,
            metadata_by_name[accept_name],
            {"lc2_name": lc2_name, "lc2_author": author, "lc2_bibcode": bibcode},
            config,
        )
        feature["profile_url"] = url
        feature["profile_sha256"] = hashlib.sha256(payload).hexdigest()
        feature["profile_bytes"] = len(payload)
        feature["profile_bins"] = len(profile)
        records.append(feature)
        source_receipts.append(
            {
                "accept_name": accept_name,
                "url": url,
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "file_bytes": len(payload),
                "profile_bins": len(profile),
            }
        )
    if len(records) != 23:
        raise GravityItem15AcceptLC2Error("response-blind eligible predictor inventory changed")
    cooling_order = sorted(records, key=lambda row: (float(row["log_tcool20"]), row["accept_name"]))
    strata = {
        str(row["accept_name"]): ("short_cooling" if index < 12 else "long_cooling")
        for index, row in enumerate(cooling_order)
    }
    sample_config = config["sample"]
    sample_objects = []
    feature_by_name = {str(row["accept_name"]): row for row in records}
    for stratum in ("short_cooling", "long_cooling"):
        candidates = [row for row in records if strata[str(row["accept_name"])] == stratum]
        ordered = sorted(
            candidates,
            key=lambda row: _split_hash(
                f"role|{row['accept_name']}", str(sample_config["split_salt"])
            ),
        )
        exploration_count = int(sample_config["exploration_per_stratum"][stratum])
        exploration_names = {str(row["accept_name"]) for row in ordered[:exploration_count]}
        fold_order = sorted(
            ordered[:exploration_count],
            key=lambda row: _split_hash(
                f"fold|{row['accept_name']}", str(sample_config["fold_salt"])
            ),
        )
        fold_by_name = {
            str(row["accept_name"]): index % int(config["evaluation"]["outer_folds"])
            for index, row in enumerate(fold_order)
        }
        for row in candidates:
            name = str(row["accept_name"])
            role = "exploration" if name in exploration_names else "reserved_confirmation"
            sample_objects.append(
                {
                    "accept_name": name,
                    "lc2_name": row["lc2_name"],
                    "lc2_author": row["lc2_author"],
                    "lc2_bibcode": row["lc2_bibcode"],
                    "ra": row["ra"],
                    "dec": row["dec"],
                    "redshift": row["redshift"],
                    "cooling_stratum": stratum,
                    "role": role,
                    "outer_fold": fold_by_name.get(name),
                    "selection_digest": _split_hash(
                        f"selected|{name}", str(sample_config["split_salt"])
                    ),
                }
            )
    roles = Counter(str(row["role"]) for row in sample_objects)
    folds = Counter(
        int(row["outer_fold"]) for row in sample_objects if row["role"] == "exploration"
    )
    if roles != {"exploration": 18, "reserved_confirmation": 5} or min(folds.values()) < 2:
        raise GravityItem15AcceptLC2Error("frozen sample role or fold balance failed")
    sample = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-accept-lc2-sample-2.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "objects": sorted(sample_objects, key=lambda row: (row["role"], row["accept_name"])),
            "counts": {
                "eligible": 23,
                "exploration": 18,
                "reserved_confirmation": 5,
                "response_rows_read": 0,
                "herbonnet_sample_objects_excluded": 100,
            },
            "fold_counts_exploration": {str(key): value for key, value in sorted(folds.items())},
            "cooling_stratum_counts": dict(sorted(Counter(strata.values()).items())),
            "selection_used_lc2_mass_response": False,
            "claims": {"confirmation_opened": False},
        }
    )
    predictors = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-accept-predictors-2.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "accept_metadata_receipt": {
                "url": config["sources"]["accept"]["metadata_url"],
                "file_sha256": hashlib.sha256(metadata_payload).hexdigest(),
                "file_bytes": len(metadata_payload),
                "parsed_rows": len(metadata_rows),
            },
            "profile_receipts": sorted(source_receipts, key=lambda row: row["accept_name"]),
            "records": sorted(
                [_serialize(feature_by_name[str(row["accept_name"])]) for row in sample_objects],
                key=lambda row: row["accept_name"],
            ),
            "counts": {"records": 23, "lc2_mass_response_rows": 0, "paid_model_calls": 0},
            "claims": {
                "complete_baryonic_mass_measured": False,
                "causal_timescale_mechanism_established": False,
            },
        }
    )
    arrays = generate_candidates(config)
    families = config["candidate_generator"]["families"]
    family_counts = Counter(families[int(value)]["id"] for value in arrays["family"])
    origin_counts = Counter(families[int(value)]["origin_status"] for value in arrays["family"])
    qualifying_counts = Counter(
        "qualifying" if families[int(value)]["qualifying"] else "rewrite_control"
        for value in arrays["family"]
    )
    candidates = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-accept-lc2-candidates-2.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "candidate_digest_sha256": _candidate_digest(arrays),
            "family_counts": dict(sorted(family_counts.items())),
            "origin_status_counts": dict(sorted(origin_counts.items())),
            "qualifying_counts": dict(sorted(qualifying_counts.items())),
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "counts": {
                "candidate_cells": len(arrays["family"]),
                "response_rows_read": 0,
                "post_response_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {"historical_novelty_established": False},
        }
    )
    paths = tuple(
        root / config["outputs"][key]
        for key in ("sample_manifest", "predictor_source", "candidate_manifest")
    )
    for path, artifact in zip(paths, (sample, predictors, candidates), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    return paths


def validate_prepared_sources(
    sample: Mapping[str, Any],
    predictors: Mapping[str, Any],
    candidates: Mapping[str, Any],
    root: Path,
) -> None:
    config = load_config(root)
    for value, label in (
        (sample, "Item 15 attempt-2 sample"),
        (predictors, "Item 15 attempt-2 predictors"),
        (candidates, "Item 15 attempt-2 candidates"),
    ):
        _validate_content_hash(value, label)
        if value["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
            raise GravityItem15AcceptLC2Error(f"{label} scientific binding changed")
    objects = sample["objects"]
    roles = Counter(str(row["role"]) for row in objects)
    if len(objects) != 23 or roles != {"exploration": 18, "reserved_confirmation": 5}:
        raise GravityItem15AcceptLC2Error("prepared sample counts changed")
    names = [str(row["accept_name"]) for row in objects]
    expected = {str(row[0]) for row in config["sample"]["eligible_lc2_rows"]}
    if len(names) != len(set(names)) or set(names) != expected:
        raise GravityItem15AcceptLC2Error("prepared sample identity set changed")
    if int(sample["counts"]["response_rows_read"]) != 0:
        raise GravityItem15AcceptLC2Error("response entered prepared sample")
    predictor_names = [str(row["accept_name"]) for row in predictors["records"]]
    if len(predictor_names) != 23 or set(predictor_names) != expected:
        raise GravityItem15AcceptLC2Error("prepared predictor identity set changed")
    if int(predictors["counts"]["lc2_mass_response_rows"]) != 0:
        raise GravityItem15AcceptLC2Error("response entered prepared predictors")
    arrays = generate_candidates(config)
    if candidates["candidate_digest_sha256"] != _candidate_digest(arrays):
        raise GravityItem15AcceptLC2Error("prepared candidate digest changed")
    if int(candidates["counts"]["candidate_cells"]) != 262144:
        raise GravityItem15AcceptLC2Error("prepared candidate count changed")
    if int(candidates["counts"]["post_response_cells"]) != 0:
        raise GravityItem15AcceptLC2Error("post-response candidate entered prepared source")
    for artifact in (sample, predictors, candidates):
        if any(bool(value) for value in artifact["claims"].values()):
            raise GravityItem15AcceptLC2Error("prepared source contains an overclaim")


def _load_prepared(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(root)
    values = tuple(
        json.loads((root / config["outputs"][key]).read_text(encoding="utf-8"))
        for key in ("sample_manifest", "predictor_source", "candidate_manifest")
    )
    validate_prepared_sources(*values, root)
    return values


def _lc2_query_url(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    source = config["sources"]["lc2"]
    parameters = [
        ("-source", source["catalog"]),
        ("-out.max", "20"),
        ("-out", ",".join(source["response_columns"])),
        ("Name", str(row["lc2_name"])),
        ("Author", str(row["lc2_author"])),
        ("BibCode", str(row["lc2_bibcode"])),
    ]
    return f"{source['vizier_endpoint']}?{urllib.parse.urlencode(parameters)}"


def _vizier_dict_rows(payload: bytes) -> list[dict[str, str]]:
    try:
        data_lines = [
            line
            for line in payload.decode("utf-8").splitlines()
            if line and not line.startswith("#")
        ]
    except UnicodeDecodeError as exc:
        raise GravityItem15AcceptLC2Error("LC2 response is not UTF-8") from exc
    if not data_lines:
        return []
    return [dict(row) for row in csv.DictReader(io.StringIO("\n".join(data_lines)), delimiter="\t")]


def parse_lc2_response(payload: bytes, sample_row: Mapping[str, Any]) -> dict[str, Any]:
    matches = []
    for raw in _vizier_dict_rows(payload):
        try:
            mass = float((raw.get("M500") or "").strip())
            error = float((raw.get("e_M500") or "").strip())
            ra = float((raw.get("_RAJ2000") or "").strip())
            dec = float((raw.get("_DEJ2000") or "").strip())
            redshift = float((raw.get("z") or "").strip())
        except ValueError:
            continue
        if (
            (raw.get("Author") or "").strip() == str(sample_row["lc2_author"])
            and (raw.get("BibCode") or "").strip() == str(sample_row["lc2_bibcode"])
            and _normalize_name((raw.get("Name") or "").strip())
            == _normalize_name(str(sample_row["lc2_name"]))
        ):
            matches.append(
                {
                    "accept_name": sample_row["accept_name"],
                    "lc2_name": (raw.get("Name") or "").strip(),
                    "lc2_name_ned": (raw.get("NameNED") or "").strip(),
                    "lc2_author": (raw.get("Author") or "").strip(),
                    "lc2_bibcode": (raw.get("BibCode") or "").strip(),
                    "lc2_ra": ra,
                    "lc2_dec": dec,
                    "lc2_redshift": redshift,
                    "m500_1e14_msun": mass,
                    "m500_error_1e14_msun": error,
                }
            )
    if len(matches) != 1:
        raise GravityItem15AcceptLC2Error(
            f"expected one frozen LC2 response row, found {len(matches)}"
        )
    match = matches[0]
    if (
        _separation_arcmin(
            float(sample_row["ra"]),
            float(sample_row["dec"]),
            float(match["lc2_ra"]),
            float(match["lc2_dec"]),
        )
        > 5.0
        or abs(float(sample_row["redshift"]) - float(match["lc2_redshift"])) > 0.02
    ):
        raise GravityItem15AcceptLC2Error("LC2 response identity crossmatch changed")
    return match


def write_response_source(root: Path) -> Path:
    root = root.resolve()
    _require_sample_freeze()
    config = load_config(root)
    sample, _, _ = _load_prepared(root)
    exploration = sorted(
        (row for row in sample["objects"] if row["role"] == "exploration"),
        key=lambda row: str(row["accept_name"]),
    )
    confirmations = {
        str(row["accept_name"])
        for row in sample["objects"]
        if row["role"] == "reserved_confirmation"
    }
    records = []
    failures = []
    cache_dir = root / "work/gravity/item-15-accept-lc2-response-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in exploration:
        name = str(row["accept_name"])
        if name in confirmations:
            raise GravityItem15AcceptLC2Error("confirmation entered LC2 acquisition")
        url = _lc2_query_url(row, config)
        cache = cache_dir / f"{name}.tsv"
        try:
            payload = cache.read_bytes() if cache.exists() else _fetch(url, attempts=4)
            if not cache.exists():
                cache.write_bytes(payload)
            response = parse_lc2_response(payload, row)
            records.append(
                {
                    **response,
                    "query_url": url,
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "payload_bytes": len(payload),
                }
            )
        except GravityItem15AcceptLC2Error as exc:
            failures.append({"accept_name": name, "query_url": url, "reason": str(exc)})
    observed = {str(row["accept_name"]) for row in [*records, *failures]}
    if confirmations & observed or observed != {str(row["accept_name"]) for row in exploration}:
        raise GravityItem15AcceptLC2Error("LC2 response identity boundary changed")
    source = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-lc2-response-source-2.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "catalog": config["sources"]["lc2"]["catalog"],
            "records": sorted(records, key=lambda row: row["accept_name"]),
            "failures": sorted(failures, key=lambda row: row["accept_name"]),
            "counts": {
                "exploration_response_objects_attempted": len(exploration),
                "exploration_response_objects_parsed": len(records),
                "exploration_response_failures": len(failures),
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {"confirmation_opened": False},
        }
    )
    path = root / config["outputs"]["response_source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(source) + b"\n")
    return path


def validate_response_source(source: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(source, "Item 15 attempt-2 response")
    if source["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem15AcceptLC2Error("response scientific binding changed")
    if source["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem15AcceptLC2Error("response sample binding changed")
    for key in ("confirmation_response_rows", "post_response_formula_cells", "paid_model_calls"):
        if int(source["counts"][key]) != 0:
            raise GravityItem15AcceptLC2Error(f"forbidden response count changed: {key}")
    sample, _, _ = _load_prepared(root)
    exploration = {
        str(row["accept_name"]) for row in sample["objects"] if row["role"] == "exploration"
    }
    confirmation = {
        str(row["accept_name"])
        for row in sample["objects"]
        if row["role"] == "reserved_confirmation"
    }
    observed = [str(row["accept_name"]) for row in [*source["records"], *source["failures"]]]
    if len(observed) != len(set(observed)) or set(observed) != exploration:
        raise GravityItem15AcceptLC2Error("response exploration identity set changed")
    if confirmation & set(observed):
        raise GravityItem15AcceptLC2Error("confirmation LC2 response opened")
    if any(bool(value) for value in source["claims"].values()):
        raise GravityItem15AcceptLC2Error("response source contains an overclaim")


def extract_rows(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    sample, predictors, _ = _load_prepared(root)
    response = json.loads((root / config["outputs"]["response_source"]).read_text("utf-8"))
    validate_response_source(response, root)
    sample_by_name = {
        str(row["accept_name"]): row for row in sample["objects"] if row["role"] == "exploration"
    }
    predictor_by_name = {str(row["accept_name"]): row for row in predictors["records"]}
    records = []
    failures = [
        {"accept_name": row["accept_name"], "reasons": [row["reason"]]}
        for row in response["failures"]
    ]
    maximum_error = float(config["quality"]["maximum_lc2_fractional_m500_error"])
    unit = float(config["sources"]["lc2"]["m500_unit_msun"])
    for raw in response["records"]:
        name = str(raw["accept_name"])
        predictor = predictor_by_name[name]
        sample_row = sample_by_name[name]
        mass = float(raw["m500_1e14_msun"])
        error = float(raw["m500_error_1e14_msun"])
        reasons = []
        if not math.isfinite(mass) or mass <= 0:
            reasons.append("nonpositive_m500")
        if not math.isfinite(error) or error <= 0:
            reasons.append("nonpositive_m500_error")
        if mass > 0 and error / mass > maximum_error:
            reasons.append("m500_fractional_error")
        if reasons:
            target = None
            sigma = None
        else:
            target = math.log10(mass * unit / float(predictor["mgas100_msun"]))
            sigma = error / (mass * math.log(10.0))
        record = {
            **predictor,
            **raw,
            "outer_fold": sample_row["outer_fold"],
            "cooling_stratum": sample_row["cooling_stratum"],
            "target_log_m500_to_mgas100": target,
            "target_log_error": sigma,
            "quality_pass": not reasons,
            "quality_failure_reasons": reasons,
        }
        records.append(_serialize(record))
        if reasons:
            failures.append({"accept_name": name, "reasons": reasons})
    passing = [row for row in records if row["quality_pass"]]
    expected = int(config["sample"]["exploration_objects"])
    retention = len(passing) / expected
    fold_counts = Counter(int(row["outer_fold"]) for row in passing)
    stratum_counts = Counter(str(row["cooling_stratum"]) for row in passing)
    quality_pass = (
        len(passing) >= int(config["quality"]["minimum_quality_passing_exploration_clusters"])
        and retention >= float(config["quality"]["minimum_quality_retention_fraction"])
        and len(fold_counts) == int(config["evaluation"]["outer_folds"])
        and min(fold_counts.values(), default=0)
        >= int(config["quality"]["minimum_quality_passing_per_outer_fold"])
        and min(stratum_counts.values(), default=0)
        >= int(config["quality"]["minimum_quality_passing_per_cooling_stratum"])
    )
    summary = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-accept-lc2-extraction-2.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": (
                "PASS_ITEM15_ACCEPT_LC2_QUALITY"
                if quality_pass
                else "INCONCLUSIVE_ITEM15_ACCEPT_LC2_QUALITY"
            ),
            "records": sorted(records, key=lambda row: row["accept_name"]),
            "failures": sorted(failures, key=lambda row: row["accept_name"]),
            "fold_counts_passing": {str(key): value for key, value in sorted(fold_counts.items())},
            "cooling_stratum_counts_passing": dict(sorted(stratum_counts.items())),
            "counts": {
                "exploration_attempted": expected,
                "quality_passing_clusters": len(passing),
                "quality_failed_clusters": expected - len(passing),
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "quality_retention_fraction": retention,
            "claims": {"confirmation_opened": False},
        }
    )
    path = root / config["outputs"]["extraction_summary"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return path


def _candidate_data(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    def column(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    design = np.column_stack([column(key) for key in config["evaluation"]["baseline_features"]])
    modulation = config["evaluation"]["fixed_modulation_normalization"]

    def normalized(key: str, field: str) -> np.ndarray:
        center, scale = (float(value) for value in modulation[key])
        return (column(field) - center) / scale

    return {
        "rows": list(rows),
        "y": column("target_log_m500_to_mgas100"),
        "sigma": column("target_log_error"),
        "folds": np.asarray([int(row["outer_fold"]) for row in rows]),
        "design": design,
        "cooling_freefall": np.stack(
            [column(f"log_tcool_tff{radius}") for radius in (20, 50, 100)]
        ),
        "cooling_crossing": np.stack(
            [column(f"log_tcool_tsound{radius}") for radius in (20, 50, 100)]
        ),
        "cooling_cosmic": np.stack(
            [column(f"log_tcool_cosmic{radius}") for radius in (20, 50, 100)]
        ),
        "crossing_freefall": np.stack(
            [column(f"log_tsound_tff{radius}") for radius in (20, 50, 100)]
        ),
        "hierarchy_span": column("clock_hierarchy_span"),
        "hierarchy_entropy": column("clock_hierarchy_entropy"),
        "core_entropy_modulation": normalized("core_entropy", "log_k0_plus1"),
        "temperature_modulation": normalized("temperature_100", "log_temperature100"),
        "gas_concentration_modulation": normalized("gas_concentration", "gas_concentration20_100"),
        "redshift_modulation": normalized("redshift", "redshift"),
        "temperature": column("log_temperature100"),
        "redshift": column("redshift"),
        "applegate": column("lc2_source_applegate"),
        "cooling_half": np.asarray(
            [1.0 if row["cooling_stratum"] == "long_cooling" else -1.0 for row in rows]
        ),
    }


def _candidate_components(
    arrays: Mapping[str, np.ndarray],
    data: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    family = xp.asarray(arrays["family"][begin:end], dtype=xp.int32)[:, None]
    radius = xp.asarray(arrays["radius"][begin:end], dtype=xp.int32)
    threshold = xp.asarray(arrays["threshold"][begin:end], dtype=xp.float64)[:, None]
    scale = xp.asarray(arrays["scale"][begin:end], dtype=xp.float64)[:, None]
    power = xp.asarray(arrays["power"][begin:end], dtype=xp.float64)[:, None]
    phase = xp.asarray(arrays["phase"][begin:end], dtype=xp.float64)[:, None]
    modulation_index = xp.asarray(arrays["modulation"][begin:end], dtype=xp.int32)[:, None]

    def matrix(key: str) -> Any:
        return xp.asarray(data[key], dtype=xp.float64)

    cooling_freefall = matrix("cooling_freefall")
    cooling_crossing = matrix("cooling_crossing")
    cooling_cosmic = matrix("cooling_cosmic")
    selected_freefall = cooling_freefall[radius, :]
    selected_crossing = cooling_crossing[radius, :]
    selected_cosmic = cooling_cosmic[radius, :]

    def signed_power(value: Any) -> Any:
        z = (value - threshold) / scale
        magnitude = xp.abs(z) ** power
        return xp.sign(z) * magnitude / (1.0 + magnitude)

    direct_freefall = signed_power(selected_freefall)
    direct_crossing = signed_power(selected_crossing)
    direct_cosmic = signed_power(selected_cosmic)
    freefall_gradient = cooling_freefall[2][None, :] - cooling_freefall[0][None, :]
    crossing_gradient = cooling_crossing[2][None, :] - cooling_crossing[0][None, :]
    phase_gradient = signed_power(freefall_gradient)
    competition = signed_power(selected_freefall * selected_crossing)
    hierarchy_span = signed_power(matrix("hierarchy_span")[None, :])
    entropy = xp.exp(-xp.abs((matrix("hierarchy_entropy")[None, :] - threshold) / scale)) ** power
    shell = xp.exp(-0.5 * ((selected_freefall - threshold) / scale) ** 2) ** power
    interference = signed_power(freefall_gradient * crossing_gradient)
    resonance = (
        xp.exp(
            -0.5
            * ((cooling_freefall[1][None, :] - cooling_crossing[1][None, :] - threshold) / scale)
            ** 2
        )
        ** power
    )
    phase_locked = xp.tanh(
        (selected_freefall - threshold)
        * xp.cos(phase + power * selected_crossing)
        / xp.maximum(scale, 1e-12)
    )
    log_periodic = xp.cos(
        phase + power * xp.log1p(xp.abs(freefall_gradient)) / xp.maximum(scale, 1e-12)
    )
    component = xp.where(family == 0, direct_freefall, direct_crossing)
    component = xp.where(family == 2, direct_cosmic, component)
    component = xp.where(family == 3, phase_gradient, component)
    component = xp.where(family == 4, competition, component)
    component = xp.where(family == 5, hierarchy_span, component)
    component = xp.where(family == 6, entropy, component)
    component = xp.where(family == 7, shell, component)
    component = xp.where(family == 8, interference, component)
    component = xp.where(family == 9, resonance, component)
    component = xp.where(family == 10, phase_locked, component)
    component = xp.where(family == 11, log_periodic, component)
    modulations = xp.stack(
        (
            xp.ones_like(matrix("core_entropy_modulation")),
            xp.tanh(matrix("core_entropy_modulation")),
            xp.tanh(matrix("temperature_modulation")),
            xp.tanh(matrix("gas_concentration_modulation")),
            xp.tanh(matrix("redshift_modulation")),
        )
    )
    selected_modulation = modulations[modulation_index[:, 0], :]
    result = component * selected_modulation
    return xp.clip(xp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def _backend() -> tuple[Any, str, str | None]:
    try:
        import cupy as xp

        if int(xp.cuda.runtime.getDeviceCount()) < 1:
            raise RuntimeError("no CUDA device")
        return xp, "gpu_cupy", xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except (ImportError, RuntimeError):
        return np, "cpu_numpy", None


def _component_matrix(
    arrays: Mapping[str, np.ndarray], data: Mapping[str, Any], config: Mapping[str, Any], xp: Any
) -> tuple[Any, float]:
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["family"]), batch):
        end = min(begin + batch, len(arrays["family"]))
        pieces.append(_candidate_components(arrays, data, begin, end, xp))
    matrix = xp.concatenate(pieces, axis=0)
    count = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["family"]))
    cpu = _candidate_components(arrays, data, 0, count, np)
    observed = xp.asnumpy(matrix[:count]) if xp is not np else np.asarray(matrix[:count])
    return matrix, float(np.max(np.abs(cpu - observed)))


def _nested_select_matrix(
    y: np.ndarray,
    data: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    components: Any,
    config: Mapping[str, Any],
    xp: Any,
    *,
    include_records: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    folds = np.asarray(data["folds"])
    design = np.asarray(data["design"])
    baseline = np.full(len(y), np.nan)
    full = np.full(len(y), np.nan)
    records = []
    alpha = float(config["evaluation"]["ridge_alpha"])
    coefficient_ridge = float(config["evaluation"]["timescale_coefficient_ridge"])
    outer_folds = int(config["evaluation"]["outer_folds"])
    for outer in range(outer_folds):
        scores = xp.zeros(len(arrays["family"]), dtype=xp.float64)
        valid_component = xp.ones(len(arrays["family"]), dtype=xp.bool_)
        inner_count = 0
        for inner in [value for value in range(outer_folds) if value != outer]:
            train = (folds != outer) & (folds != inner)
            validation = folds == inner
            model = _ridge_fit(design[train], y[train], alpha)
            train_residual = y[train] - _ridge_predict(model, design[train])
            validation_residual = y[validation] - _ridge_predict(model, design[validation])
            train_component = components[:, train]
            validation_component = components[:, validation]
            mean = xp.mean(train_component, axis=1)
            std = xp.std(train_component, axis=1)
            valid_component &= std > 1e-8
            std = xp.maximum(std, 1e-12)
            standardized = (train_component - mean[:, None]) / std[:, None]
            coefficient = xp.sum(standardized * xp.asarray(train_residual)[None, :], axis=1) / (
                xp.sum(standardized**2, axis=1) + coefficient_ridge
            )
            residual = (
                xp.asarray(validation_residual)[None, :]
                - coefficient[:, None] * (validation_component - mean[:, None]) / std[:, None]
            )
            scores += xp.mean(residual**2, axis=1)
            inner_count += 1
        scores = scores / inner_count
        scores = xp.where(valid_component, scores, xp.inf)
        selected = int(xp.asnumpy(xp.argmin(scores)) if xp is not np else np.argmin(scores))
        train = folds != outer
        test = folds == outer
        model = _ridge_fit(design[train], y[train], alpha)
        train_base = _ridge_predict(model, design[train])
        test_base = _ridge_predict(model, design[test])
        selected_component = (
            xp.asnumpy(components[selected]) if xp is not np else np.asarray(components[selected])
        )
        mean = float(np.mean(selected_component[train]))
        std = max(float(np.std(selected_component[train])), 1e-12)
        standardized = (selected_component[train] - mean) / std
        coefficient = float(
            np.sum(standardized * (y[train] - train_base))
            / (np.sum(standardized**2) + coefficient_ridge)
        )
        baseline[test] = test_base
        full[test] = test_base + coefficient * (selected_component[test] - mean) / std
        if include_records:
            family = config["candidate_generator"]["families"][int(arrays["family"][selected])]
            score = scores[selected]
            score_value = float(xp.asnumpy(score) if xp is not np else score)
            records.append(
                {
                    "outer_fold": outer,
                    "selected_ordinal": selected,
                    "selected_family": family["id"],
                    "origin_status": family["origin_status"],
                    "qualifying": bool(family["qualifying"]),
                    "radius_kpc": config["candidate_generator"]["radius_choices_kpc"][
                        int(arrays["radius"][selected])
                    ],
                    "threshold": _metric(float(arrays["threshold"][selected])),
                    "scale": _metric(float(arrays["scale"][selected])),
                    "power": _metric(float(arrays["power"][selected])),
                    "phase": _metric(float(arrays["phase"][selected])),
                    "modulation": config["candidate_generator"]["modulations"][
                        int(arrays["modulation"][selected])
                    ],
                    "inner_mse": _metric(score_value),
                    "fitted_universal_coefficient": _metric(coefficient),
                    "test_clusters": int(np.sum(test)),
                }
            )
    if np.any(~np.isfinite(baseline)) or np.any(~np.isfinite(full)):
        raise GravityItem15AcceptLC2Error("nested Item 15 predictions are incomplete")
    return baseline, full, records


def _metrics(
    y: np.ndarray, prediction: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, str]:
    residual2 = (y - prediction) ** 2
    mse = (
        float(np.average(residual2, weights=weights))
        if weights is not None
        else float(np.mean(residual2))
    )
    mean = float(np.average(y, weights=weights)) if weights is not None else float(np.mean(y))
    variance = (
        float(np.average((y - mean) ** 2, weights=weights))
        if weights is not None
        else float(np.var(y))
    )
    return {"mse": _metric(mse), "r2": _metric(1.0 - mse / variance if variance > 0 else 0.0)}


def _full_selection_permutation(
    observed_gain: float,
    data: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    components: Any,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    count = int(config["evaluation"]["full_nested_label_permutations"])
    seed = int(
        hashlib.sha256(str(config["evaluation"]["permutation_salt"]).encode()).hexdigest()[:16],
        16,
    )
    random = np.random.default_rng(seed)
    y = np.asarray(data["y"])
    null = []
    for _ in range(count):
        shuffled = random.permutation(y)
        baseline, full, _ = _nested_select_matrix(
            shuffled, data, arrays, components, config, xp, include_records=False
        )
        null.append(float(np.mean((shuffled - baseline) ** 2 - (shuffled - full) ** 2)))
    values = np.asarray(null)
    return {
        "permutations": count,
        "selection_repeated_inside_each_permutation": True,
        "observed_mean_mse_gain": _metric(observed_gain),
        "p_value": _metric((1 + int(np.sum(values >= observed_gain))) / (count + 1)),
        "null_gain_quantiles": {
            "q05": _metric(float(np.quantile(values, 0.05))),
            "q50": _metric(float(np.quantile(values, 0.5))),
            "q95": _metric(float(np.quantile(values, 0.95))),
        },
    }


def _load_data(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary = json.loads((root / config["outputs"]["extraction_summary"]).read_text("utf-8"))
    _validate_content_hash(summary, "Item 15 attempt-2 extraction")
    if int(summary["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem15AcceptLC2Error("confirmation entered extraction")
    rows = [row for row in summary["records"] if row["quality_pass"]]
    if not rows:
        raise GravityItem15AcceptLC2Error("no quality-passing Item 15 clusters")
    data = _candidate_data(rows, config)
    data["summary"] = summary
    return data


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    data = _load_data(root, config)
    arrays = generate_candidates(config)
    xp, backend, device = _backend()
    started = time.perf_counter()
    components, crosscheck = _component_matrix(arrays, data, config, xp)
    baseline, full, selections = _nested_select_matrix(
        np.asarray(data["y"]), data, arrays, components, config, xp, include_records=True
    )
    y = np.asarray(data["y"])
    baseline_metrics = _metrics(y, baseline)
    full_metrics = _metrics(y, full)
    baseline_mse = float(baseline_metrics["mse"])
    full_mse = float(full_metrics["mse"])
    relative = (baseline_mse - full_mse) / baseline_mse
    differences = (y - baseline) ** 2 - (y - full) ** 2
    permutation = _full_selection_permutation(
        float(np.mean(differences)), data, arrays, components, config, xp
    )
    inverse_variance = 1.0 / np.maximum(np.asarray(data["sigma"]) ** 2, 1e-8)
    cap = float(config["evaluation"]["error_weight_cap_ratio"])
    inverse_variance = np.minimum(inverse_variance, np.min(inverse_variance) * cap)
    weighted_baseline = _metrics(y, baseline, inverse_variance)
    weighted_full = _metrics(y, full, inverse_variance)
    dimensions = {
        "cooling_half": (np.asarray(data["cooling_half"]), 0.0),
        "temperature_half": (
            np.asarray(data["temperature"]),
            float(np.median(data["temperature"])),
        ),
        "redshift_half": (np.asarray(data["redshift"]), float(np.median(data["redshift"]))),
        "applegate_vs_other": (np.asarray(data["applegate"]), 0.5),
    }
    strata = []
    stratum_pass = {}
    for dimension, (values, split) in dimensions.items():
        gains = []
        for label, mask in (
            ("low_or_other", values <= split),
            ("high_or_applegate", values > split),
        ):
            control = float(np.mean((y[mask] - baseline[mask]) ** 2))
            proposed = float(np.mean((y[mask] - full[mask]) ** 2))
            gain = control - proposed
            gains.append(gain)
            strata.append(
                {
                    "dimension": dimension,
                    "stratum": label,
                    "clusters": int(np.sum(mask)),
                    "control_mse": _metric(control),
                    "full_model_mse": _metric(proposed),
                    "timescale_mse_gain": _metric(gain),
                }
            )
        stratum_pass[dimension] = all(value > 0 for value in gains)
    coefficients = np.asarray([float(row["fitted_universal_coefficient"]) for row in selections])
    sign_agreement = max(int(np.sum(coefficients > 0)), int(np.sum(coefficients < 0)))
    qualifying_folds = sum(bool(row["qualifying"]) for row in selections)
    summary = data["summary"]
    gates = {
        "quality_count_and_fraction_pass": summary["decision"] == "PASS_ITEM15_ACCEPT_LC2_QUALITY",
        "fresh_identity_and_confirmation_boundary_pass": int(
            summary["counts"]["confirmation_response_rows"]
        )
        == 0,
        "candidate_count_exact": len(arrays["family"]) == 262144,
        "full_model_r2_positive": float(full_metrics["r2"]) > 0,
        "timescale_beats_strong_source_baseline": full_mse < baseline_mse,
        "relative_mse_improvement_at_least": relative
        >= float(config["admission"]["relative_mse_improvement_at_least"]),
        "full_selection_permutation_p_at_most": float(permutation["p_value"])
        <= float(config["admission"]["full_selection_permutation_p_at_most"]),
        "error_weighted_mse_improves": float(weighted_full["mse"])
        < float(weighted_baseline["mse"]),
        "gain_positive_in_both_cooling_halves": stratum_pass["cooling_half"],
        "gain_positive_in_both_temperature_halves": stratum_pass["temperature_half"],
        "gain_positive_in_both_redshift_halves": stratum_pass["redshift_half"],
        "gain_positive_in_applegate_and_other_sources": stratum_pass["applegate_vs_other"],
        "qualifying_family_selected_in_folds_at_least": qualifying_folds
        >= int(config["admission"]["qualifying_family_selected_in_folds_at_least"]),
        "coefficient_sign_agreement_folds_at_least": sign_agreement
        >= int(config["admission"]["coefficient_sign_agreement_folds_at_least"]),
        "post_response_formula_generation_zero": True,
    }
    decision = (
        "PASS_ITEM15_ACCEPT_LC2_TIMESCALE_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM15_ACCEPT_LC2_TIMESCALE_EXPLORATION"
    )
    if not gates["quality_count_and_fraction_pass"]:
        decision = "INCONCLUSIVE_ITEM15_ACCEPT_LC2_QUALITY"
    if xp is not np:
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    input_paths = {
        key: root / config["outputs"][key]
        for key in (
            "sample_manifest",
            "predictor_source",
            "candidate_manifest",
            "response_source",
            "extraction_summary",
        )
    }
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-accept-lc2-result-2.0",
            "goal": config["goal"],
            "item_number": 15,
            "attempt_number": 2,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": decision,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "attempt_scope": config["scientific_contract"]["attempt_scope"],
            "response_boundary": config["scientific_contract"]["interpretation_boundary"],
            "counts": {
                "candidate_cells": len(arrays["family"]),
                "quality_passing_clusters": summary["counts"]["quality_passing_clusters"],
                "quality_failed_clusters": summary["counts"]["quality_failed_clusters"],
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "inputs": {key + "_sha256": _sha256_file(path) for key, path in input_paths.items()},
            "primary_lensing_to_gas_log_mass_ratio": {
                "strong_source_variable_baseline": baseline_metrics,
                "selected_timescale_full_model": full_metrics,
                "relative_mse_improvement": _metric(relative),
                "outer_fold_selections": selections,
            },
            "measurement_error_weighted_robustness": {
                "baseline": weighted_baseline,
                "full_model": weighted_full,
                "candidate_reselection": False,
                "weight_cap_ratio": _metric(cap),
            },
            "full_selection_permutation": permutation,
            "strata": strata,
            "gate_checks": gates,
            "gate_counts": {"passed": sum(gates.values()), "required": len(gates)},
            "compute": {
                "backend": backend,
                "device": device,
                "cupy_version": getattr(xp, "__version__", None) if xp is not np else None,
                "elapsed_seconds": _metric(elapsed),
                "candidate_cells": len(arrays["family"]),
                "clusters": len(y),
                "outer_folds": int(config["evaluation"]["outer_folds"]),
                "full_nested_label_permutations": int(
                    config["evaluation"]["full_nested_label_permutations"]
                ),
                "candidate_scalar_score_evaluations_observed": len(arrays["family"])
                * len(y)
                * 5
                * 4,
                "candidate_scalar_score_evaluations_with_null": len(arrays["family"])
                * len(y)
                * 5
                * 4
                * (1 + int(config["evaluation"]["full_nested_label_permutations"])),
                "cpu_gpu_max_component_difference": _metric(crosscheck),
            },
            "equivalence_boundary": config["timescale_features"]["equivalence_boundary"],
            "limitations": {
                "sample_is_small": True,
                "weak_lensing_catalog_is_heterogeneous_and_model_dependent": True,
                "response_is_direct_shear_or_image_observable": False,
                "gas_mass_within_100kpc_is_complete_baryonic_mass": False,
                "baryon_only_freefall_omits_stars": True,
                "causal_timescale_or_modified_gravity_established": False,
                "confirmation_opened": False,
                "item15_broad_synthesis_complete": False,
            },
            "claims": config["claim_boundaries"],
        }
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(receipt, "Item 15 attempt-2 result")
    config = load_config(root)
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem15AcceptLC2Error("result scientific binding changed")
    if receipt["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem15AcceptLC2Error("result sample binding changed")
    if int(receipt["counts"]["candidate_cells"]) != int(
        config["candidate_generator"]["candidate_cells"]
    ):
        raise GravityItem15AcceptLC2Error("result candidate count changed")
    for key in ("confirmation_response_rows", "post_response_formula_cells", "paid_model_calls"):
        if int(receipt["counts"][key]) != 0:
            raise GravityItem15AcceptLC2Error(f"forbidden result count changed: {key}")
    if not bool(
        receipt["full_selection_permutation"]["selection_repeated_inside_each_permutation"]
    ):
        raise GravityItem15AcceptLC2Error("full-selection null changed")
    if not bool(receipt["claims"]["direct_hot_gas_cooling_time_tested"]):
        raise GravityItem15AcceptLC2Error("result removed direct-cooling coverage")
    if any(
        bool(value)
        for key, value in receipt["claims"].items()
        if key != "direct_hot_gas_cooling_time_tested"
    ):
        raise GravityItem15AcceptLC2Error("result contains an overclaim")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root)
    path = root / config["outputs"]["result"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    config = load_config(root)
    path = root / config["outputs"]["result"]
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "acquire-response", "extract", "run", "check")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    if arguments.command == "prepare":
        paths = write_prepared_sources(arguments.root)
        print("\n".join(str(path) for path in paths))
    elif arguments.command == "acquire-response":
        print(write_response_source(arguments.root))
    elif arguments.command == "extract":
        print(extract_rows(arguments.root))
    elif arguments.command == "run":
        print(write_receipt(arguments.root))
    else:
        check_receipt(arguments.root)
        print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
