"""Frozen Item 39 holographic/boundary-gravity search machinery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import (
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

CONFIG_PATH = Path("configs/gravity_item39_holographic_boundary_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")


class GravityItem39Error(RuntimeError):
    """Raised when an Item 39 freeze, leakage boundary, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != ("invariant-gravity-item39-holographic-boundary-config-1.0")
        or int(config.get("item", -1)) != 39
    ):
        raise GravityItem39Error("unexpected Item 39 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem39Error("stable gravity goal changed")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem39Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-38-emergent-gravity-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem39Error("Item 38 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem39Error("Item 38 decision changed")

    policy = load_counterexample_policy(root / POLICY_PATH)
    empirical = policy["empirical_evidence"]
    discovery = config["discovery_policy"]
    if not bool(discovery["equal_initial_viability"]):
        raise GravityItem39Error("equal initial viability changed")
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem39Error("single-counterexample retention changed")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem39Error("count-only empirical rejection entered Item 39")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem39Error("finite empirical family pruning entered Item 39")
    if bool(discovery["single_object_sensitive_formula_may_promote"]):
        raise GravityItem39Error("single-object-sensitive promotion entered Item 39")
    if empirical["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem39Error("executable counterexample policy changed")

    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem39Error("raw candidate count changed")
    if int(generator["cells_per_niche"]) != 65536:
        raise GravityItem39Error("per-niche capacity changed")
    if list(generator["grid_shape_per_niche"]) != [16, 16, 16, 16]:
        raise GravityItem39Error("candidate grid shape changed")
    if len(generator["niches"]) != 4:
        raise GravityItem39Error("Item 39 requires four equal niches")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem39Error("post-response candidates entered Item 39")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem39Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem39Error("paid model calls entered Item 39")
    metric = config["weak_field_metric_contract"]
    if metric["gravitational_slip"] != "Phi=Psi":
        raise GravityItem39Error("motion/light metric closure changed")
    if not bool(config["independence"]["exclude_every_item10_sample_name_and_coordinate"]):
        raise GravityItem39Error("Item 10 role exclusion changed")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    for key in ("scientific_freeze_commit", "predictor_freeze_commit", "sample_freeze_commit"):
        value[key] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _query_csv(
    endpoint: str,
    query: str,
    *,
    dialect: str,
    user_agent: str,
    timeout: int = 240,
) -> tuple[bytes, list[dict[str, str]]]:
    if dialect == "tap_adql":
        parameters = urllib.parse.urlencode(
            {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
        )
        url = f"{endpoint}?{parameters}"
        headers = {"User-Agent": user_agent}
    elif dialect == "datalab_sql":
        parameters = urllib.parse.urlencode(
            {
                "sql": query,
                "ofmt": "csv",
                "out": "None",
                "async": "False",
                "drop": "False",
                "profile": "default",
            }
        )
        url = f"{endpoint}?{parameters}"
        headers = {
            "User-Agent": user_agent,
            "X-DL-AuthToken": "anonymous.0.0.anon_access",
            "X-DL-TimeoutRequest": str(timeout),
            "X-DL-ClientVersion": "Invariant-Item39",
            "X-DL-OriginIP": "0.0.0.0",
            "X-DL-OriginHost": "localhost",
        }
    else:
        raise GravityItem39Error(f"unknown query dialect: {dialect}")
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if not payload:
        raise GravityItem39Error("empty catalogue query response")
    text = payload.decode("utf-8-sig", errors="strict")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise GravityItem39Error("catalogue CSV has no header")
    rows = [
        {str(key): str(value).strip() for key, value in row.items()}
        for row in reader
        if row and any(str(value).strip() for value in row.values())
    ]
    return payload, rows


def _parse_vector(value: str) -> np.ndarray:
    try:
        result = np.asarray(
            [float(item.strip()) for item in str(value).split(",")], dtype=np.float64
        )
    except ValueError as exc:
        raise GravityItem39Error("invalid catalogue vector") from exc
    if not len(result) or np.any(~np.isfinite(result)):
        raise GravityItem39Error("empty or non-finite catalogue vector")
    return result


def _as_float(value: Any, *, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _distance_mpc(frequency_hz: float, config: Mapping[str, Any]) -> float:
    constants = config["constants"]
    redshift = float(constants["hi_rest_frequency_hz"]) / frequency_hz - 1.0
    distance = (
        float(constants["speed_of_light_km_s"])
        * redshift
        / float(constants["hubble_constant_km_s_mpc"])
    )
    if not math.isfinite(distance) or distance <= 0.0 or redshift >= 0.1:
        raise GravityItem39Error("invalid WALLABY Hubble distance")
    return distance


def _separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1_rad, dec1_rad, ra2_rad, dec2_rad = map(math.radians, (ra1, dec1, ra2, dec2))
    haversine = math.sin((dec2_rad - dec1_rad) / 2.0) ** 2
    haversine += math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin((ra2_rad - ra1_rad) / 2.0) ** 2
    return 2.0 * math.asin(math.sqrt(min(max(haversine, 0.0), 1.0))) * 206264.80624709636


def _edge_crossing(radius: np.ndarray, sigma: np.ndarray, threshold: float) -> float | None:
    candidates = np.flatnonzero((sigma[:-1] >= threshold) & (sigma[1:] < threshold))
    if not len(candidates):
        return None
    index = int(candidates[-1])
    y0 = math.log(max(float(sigma[index]), 1e-12))
    y1 = math.log(max(float(sigma[index + 1]), 1e-12))
    fraction = (math.log(threshold) - y0) / (y1 - y0) if y1 != y0 else 0.5
    return float(radius[index] + np.clip(fraction, 0.0, 1.0) * (radius[index + 1] - radius[index]))


def _release_rank(release: str, config: Mapping[str, Any]) -> tuple[int, str]:
    priority = list(config["independence"]["release_priority"])
    try:
        rank = priority.index(release)
    except ValueError:
        rank = len(priority)
    return rank, release


def _serialize_profile(
    kinematic: Mapping[str, str], source: Mapping[str, str], config: Mapping[str, Any]
) -> dict[str, Any]:
    radius_arcsec = _parse_vector(kinematic["Rad_SD"])
    sigma = _parse_vector(kinematic["SD_model"])
    sigma_error = _parse_vector(kinematic["e_SD_model"])
    if not (len(radius_arcsec) == len(sigma) == len(sigma_error)):
        raise GravityItem39Error("WALLABY surface-profile vector lengths differ")
    order = np.argsort(radius_arcsec)
    radius_arcsec = radius_arcsec[order]
    sigma = sigma[order]
    sigma_error = sigma_error[order]
    valid = (radius_arcsec > 0.0) & (sigma > 0.0) & (sigma_error >= 0.0)
    radius_arcsec = radius_arcsec[valid]
    sigma = sigma[valid]
    sigma_error = sigma_error[valid]
    if len(radius_arcsec) < 6 or np.any(np.diff(radius_arcsec) <= 0.0):
        raise GravityItem39Error("insufficient unique positive HI profile radii")
    distance = _as_float(source.get("dist_h"))
    if not math.isfinite(distance) or distance <= 0.0:
        distance = _distance_mpc(float(kinematic["freq"]), config)
    radius_kpc = (
        radius_arcsec * distance * 1000.0 / float(config["constants"]["arcseconds_per_radian"])
    )
    edges = np.empty(len(radius_kpc) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (radius_kpc[:-1] + radius_kpc[1:])
    edges[-1] = radius_kpc[-1] + 0.5 * (radius_kpc[-1] - radius_kpc[-2])
    annulus_pc2 = math.pi * np.diff(np.square(edges * 1000.0))
    raw_shell_mass = np.maximum(sigma * annulus_pc2, 0.0)
    raw_total = float(np.sum(raw_shell_mass))
    if not math.isfinite(raw_total) or raw_total <= 0.0:
        raise GravityItem39Error("invalid integrated HI profile mass")
    log_hi = _as_float(source.get("log_m_hi_corr"))
    if not math.isfinite(log_hi):
        log_hi = _as_float(source.get("log_m_hi"))
    if not math.isfinite(log_hi):
        raise GravityItem39Error("missing official HI mass")
    hi_mass = 10.0**log_hi
    cumulative_hi = np.cumsum(raw_shell_mass) * (hi_mass / raw_total)
    r_hi = _edge_crossing(radius_kpc, sigma, 1.0)
    separation = _separation_arcsec(
        float(kinematic["ra"]),
        float(kinematic["dec"]),
        float(source["ra"]),
        float(source["dec"]),
    )
    return {
        "name": str(kinematic["name"]),
        "ra": f"{float(kinematic['ra']):.12e}",
        "dec": f"{float(kinematic['dec']):.12e}",
        "frequency_hz": f"{float(kinematic['freq']):.12e}",
        "distance_mpc": f"{distance:.12e}",
        "team_release": str(kinematic["team_release"]),
        "team_release_kin": str(kinematic["team_release_kin"]),
        "radius_arcsec": [f"{value:.12e}" for value in radius_arcsec],
        "radius_kpc": [f"{value:.12e}" for value in radius_kpc],
        "surface_density_hi_msun_pc2": [f"{value:.12e}" for value in sigma],
        "surface_density_error_hi_msun_pc2": [f"{value:.12e}" for value in sigma_error],
        "cumulative_hi_mass_msun": [f"{value:.12e}" for value in cumulative_hi],
        "hi_mass_msun": f"{hi_mass:.12e}",
        "screen_radius_kpc": f"{float(radius_kpc[-1]):.12e}",
        "r_hi_q1_kpc": None if r_hi is None else f"{r_hi:.12e}",
        "source_match_separation_arcsec": f"{separation:.12e}",
        "source_qflag": str(source.get("qflag", "")),
    }


def _exposure_scope(root: Path, config: Mapping[str, Any]) -> tuple[set[str], np.ndarray]:
    manifest = _read_json(_source_path(root, config, "exposure_manifest"))
    names = {str(value) for value in manifest["excluded_names"]}
    coordinates = np.asarray(
        [(float(row["ra"]), float(row["dec"])) for row in manifest["excluded_coordinates"]],
        dtype=np.float64,
    )
    return names, coordinates


def _minimum_separation_arcsec(ra: float, dec: float, coordinates: np.ndarray) -> float:
    return min(_separation_arcsec(ra, dec, float(row[0]), float(row[1])) for row in coordinates)


def write_wallaby_predictor_source(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    if str(config["scientific_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem39Error("Item 39 scientific freeze is not bound")
    check_freeze(root)
    wallaby = config["data_sources"]["wallaby"]
    kin_columns = ",".join(wallaby["kinematic_predictor_columns"])
    src_columns = ",".join(wallaby["source_predictor_columns"])
    kin_query = f"SELECT {kin_columns} FROM {wallaby['kinematic_table']} ORDER BY name"
    src_query = f"SELECT {src_columns} FROM {wallaby['source_table']} ORDER BY name"
    kin_payload, kin_rows = _query_csv(
        str(wallaby["tap_sync_endpoint"]),
        kin_query,
        dialect="tap_adql",
        user_agent="Invariant/Item39-WALLABY-Predictors",
    )
    src_payload, src_rows = _query_csv(
        str(wallaby["tap_sync_endpoint"]),
        src_query,
        dialect="tap_adql",
        user_agent="Invariant/Item39-WALLABY-Predictors",
    )
    if any(set(row) != set(wallaby["kinematic_predictor_columns"]) for row in kin_rows):
        raise GravityItem39Error("WALLABY kinematic predictor schema changed")
    if any(set(row) != set(wallaby["source_predictor_columns"]) for row in src_rows):
        raise GravityItem39Error("WALLABY source predictor schema changed")

    source_by_name: dict[str, list[dict[str, str]]] = {}
    for row in src_rows:
        source_by_name.setdefault(str(row["name"]), []).append(row)
    excluded_names, excluded_coordinates = _exposure_scope(root, config)
    failures: list[dict[str, Any]] = []
    eligible: list[tuple[dict[str, str], dict[str, str]]] = []
    for row in kin_rows:
        name = str(row["name"])
        ra, dec = float(row["ra"]), float(row["dec"])
        separation = _minimum_separation_arcsec(ra, dec, excluded_coordinates)
        if name in excluded_names or separation < float(
            config["independence"]["coordinate_exclusion_arcseconds"]
        ):
            failures.append(
                {
                    "name": name,
                    "team_release_kin": str(row["team_release_kin"]),
                    "reason": "item10_name_or_coordinate_exclusion",
                    "minimum_item10_separation_arcsec": f"{separation:.12e}",
                }
            )
            continue
        options = source_by_name.get(name, [])
        if not options:
            failures.append(
                {
                    "name": name,
                    "team_release_kin": str(row["team_release_kin"]),
                    "reason": "no_source_catalogue_name_match",
                }
            )
            continue
        source = min(
            options,
            key=lambda candidate: _separation_arcsec(
                ra, dec, float(candidate["ra"]), float(candidate["dec"])
            ),
        )
        source_separation = _separation_arcsec(ra, dec, float(source["ra"]), float(source["dec"]))
        if source_separation > 30.0:
            failures.append(
                {
                    "name": name,
                    "team_release_kin": str(row["team_release_kin"]),
                    "reason": "source_catalogue_match_over_30_arcsec",
                    "source_separation_arcsec": f"{source_separation:.12e}",
                }
            )
            continue
        eligible.append((row, source))

    by_name: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
    for pair in eligible:
        by_name.setdefault(str(pair[0]["name"]), []).append(pair)
    records: list[dict[str, Any]] = []
    duplicate_releases: dict[str, list[str]] = {}
    for name, pairs in sorted(by_name.items()):
        ordered = sorted(
            pairs,
            key=lambda pair: _release_rank(str(pair[0]["team_release_kin"]), config),
        )
        if len(ordered) > 1:
            duplicate_releases[name] = [str(pair[0]["team_release_kin"]) for pair in ordered]
        selected = ordered[0]
        try:
            records.append(_serialize_profile(selected[0], selected[1], config))
        except GravityItem39Error as exc:
            failures.append(
                {
                    "name": name,
                    "team_release_kin": str(selected[0]["team_release_kin"]),
                    "reason": str(exc),
                }
            )
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-wallaby-predictor-source-1.0",
            "item": 39,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "queries": {"kinematic": kin_query, "source": src_query},
            "payload_sha256": {
                "kinematic": hashlib.sha256(kin_payload).hexdigest(),
                "source": hashlib.sha256(src_payload).hexdigest(),
            },
            "records": records,
            "failures": failures,
            "duplicate_release_resolution": duplicate_releases,
            "counts": {
                "kinematic_catalogue_rows": len(kin_rows),
                "source_catalogue_rows": len(src_rows),
                "fresh_unique_predictor_profiles": len(records),
                "excluded_or_invalid_rows": len(failures),
                "response_columns_requested": 0,
                "response_rows_read": 0,
                "confirmation_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "fresh_response_opened": False,
                "item10_confirmation_opened": False,
            },
        }
    )
    path = _source_path(root, config, "wallaby_predictor_source")
    _write_json(path, manifest)
    return path


def validate_wallaby_predictor_source(root: Path, source: Mapping[str, Any]) -> None:
    config = load_config(root)
    copy_value = dict(source)
    digest = copy_value.pop("content_sha256", None)
    if digest != _sha256_bytes(_canonical_bytes(copy_value)):
        raise GravityItem39Error("WALLABY predictor content hash changed")
    if source["scientific_freeze_commit"] != config["scientific_freeze_commit"]:
        raise GravityItem39Error("WALLABY predictor freeze binding changed")
    if int(source["counts"]["response_columns_requested"]) != 0:
        raise GravityItem39Error("response column entered predictor source")
    if int(source["counts"]["response_rows_read"]) != 0:
        raise GravityItem39Error("response row entered predictor source")


def _legacy_query(
    targets: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    legacy = config["data_sources"]["legacy_dr10"]
    radius_degrees = float(legacy["match_radius_arcsec"]) / 3600.0
    columns = ",".join(str(value) for value in legacy["columns"])
    all_rows: list[dict[str, str]] = []
    receipts: list[dict[str, Any]] = []
    for start in range(0, len(targets), 12):
        chunk = targets[start : start + 12]
        clauses = [
            f"q3c_radial_query(ra,dec,{float(row['ra']):.12f},{float(row['dec']):.12f},{radius_degrees:.12f})"
            for row in chunk
        ]
        query = f"SELECT {columns} FROM {legacy['table']} WHERE " + " OR ".join(
            f"({clause})" for clause in clauses
        )
        payload, rows = _query_csv(
            str(legacy["query_manager_endpoint"]),
            query,
            dialect="datalab_sql",
            user_agent="Invariant/Item39-LegacyDR10-Predictors",
        )
        all_rows.extend(rows)
        receipts.append(
            {
                "target_start": start,
                "target_count": len(chunk),
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "returned_rows": len(rows),
            }
        )
    return all_rows, receipts


def _select_legacy_match(
    target: Mapping[str, Any], candidates: list[Mapping[str, str]], config: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    legacy = config["data_sources"]["legacy_dr10"]
    ra, dec = float(target["ra"]), float(target["dec"])
    evaluated: list[tuple[float, float, Mapping[str, str]]] = []
    for row in candidates:
        separation = _separation_arcsec(
            ra, dec, _as_float(row.get("ra")), _as_float(row.get("dec"))
        )
        if not math.isfinite(separation) or separation > float(legacy["match_radius_arcsec"]):
            continue
        source_type = str(row.get("type", "")).strip().upper()
        flux_g, flux_i = _as_float(row.get("flux_g")), _as_float(row.get("flux_i"))
        transmission_g = _as_float(row.get("mw_transmission_g"))
        transmission_i = _as_float(row.get("mw_transmission_i"))
        shape_r = _as_float(row.get("shape_r"))
        ivar_g, ivar_i = _as_float(row.get("flux_ivar_g")), _as_float(row.get("flux_ivar_i"))
        if source_type not in {"REX", "EXP", "DEV", "SER"}:
            continue
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                flux_g,
                flux_i,
                transmission_g,
                transmission_i,
                shape_r,
                ivar_g,
                ivar_i,
            )
        ):
            continue
        corrected_r = _as_float(row.get("flux_r")) / max(
            _as_float(row.get("mw_transmission_r"), default=1.0), 1e-12
        )
        if not math.isfinite(corrected_r) or corrected_r <= 0.0:
            corrected_r = flux_i / transmission_i
        score = separation / 5.0 - 0.05 * math.log10(max(corrected_r, 1e-12))
        evaluated.append((score, separation, row))
    if not evaluated:
        return None, ["no_finite_extended_gi_match"]
    _, separation, row = min(evaluated, key=lambda value: (value[0], value[1]))
    if separation > 10.0:
        return None, ["best_match_over_10_arcsec"]
    flux_g = float(row["flux_g"]) / float(row["mw_transmission_g"])
    flux_i = float(row["flux_i"]) / float(row["mw_transmission_i"])
    mag_g = 22.5 - 2.5 * math.log10(flux_g)
    mag_i = 22.5 - 2.5 * math.log10(flux_i)
    distance = float(target["distance_mpc"])
    absolute_i = mag_i - (5.0 * math.log10(distance) + 25.0)
    log_stellar_mass = 1.15 + 0.70 * (mag_g - mag_i) - 0.4 * absolute_i
    effective_radius_kpc = (
        float(row["shape_r"])
        * distance
        * 1000.0
        / float(config["constants"]["arcseconds_per_radian"])
    )
    e1 = _as_float(row.get("shape_e1"), default=0.0)
    e2 = _as_float(row.get("shape_e2"), default=0.0)
    ellipticity = min(math.hypot(e1, e2), 0.999999)
    axis_ratio = (1.0 - ellipticity) / (1.0 + ellipticity)
    selected = {
        "galaxy": str(target["name"]),
        "team_release_kin": str(target["team_release_kin"]),
        "target_ra": str(target["ra"]),
        "target_dec": str(target["dec"]),
        "match_ra": f"{float(row['ra']):.12e}",
        "match_dec": f"{float(row['dec']):.12e}",
        "match_separation_arcsec": f"{separation:.12e}",
        "type": str(row["type"]),
        "flux_g_corrected_nanomaggy": f"{flux_g:.12e}",
        "flux_i_corrected_nanomaggy": f"{flux_i:.12e}",
        "g_minus_i_ab": f"{(mag_g - mag_i):.12e}",
        "absolute_i_ab": f"{absolute_i:.12e}",
        "log10_stellar_mass_msun": f"{log_stellar_mass:.12e}",
        "stellar_mass_msun": f"{10.0**log_stellar_mass:.12e}",
        "effective_radius_kpc": f"{effective_radius_kpc:.12e}",
        "axis_ratio": f"{axis_ratio:.12e}",
        "sersic": str(row.get("sersic", "")),
        "valid_candidates_within_15_arcsec": len(evaluated),
    }
    return selected, []


def write_legacy_predictor_source(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    wallaby_path = _source_path(root, config, "wallaby_predictor_source")
    wallaby = _read_json(wallaby_path)
    validate_wallaby_predictor_source(root, wallaby)
    targets = list(wallaby["records"])
    candidates, query_receipts = _legacy_query(targets, config)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for target in targets:
        selected, reasons = _select_legacy_match(target, candidates, config)
        if selected is None:
            failures.append(
                {
                    "galaxy": str(target["name"]),
                    "team_release_kin": str(target["team_release_kin"]),
                    "reasons": reasons,
                }
            )
        else:
            records.append(selected)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-legacy-dr10-predictor-source-1.0",
            "item": 39,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "wallaby_predictor_sha256": _sha256_file(wallaby_path),
            "query_receipts": query_receipts,
            "records": records,
            "failures": failures,
            "counts": {
                "wallaby_targets": len(targets),
                "raw_legacy_rows_returned": len(candidates),
                "selected_optical_matches": len(records),
                "failed_optical_matches": len(failures),
                "response_columns_requested": 0,
                "response_rows_read": 0,
                "paid_model_calls": 0,
            },
            "systematic_boundaries": {
                "stellar_mass_systematic_dex": config["quality"]["stellar_mass_systematic_dex"],
                "k_correction_applied": False,
                "resolved_surface_photometry_used": False,
                "claim": "catalogue color-M/L and model half-light radii are source-only approximations, not the 45-estimator AutoProf masses in the WALLABY scaling paper",
            },
        }
    )
    path = _source_path(root, config, "legacy_predictor_source")
    _write_json(path, manifest)
    return path


def validate_legacy_predictor_source(root: Path, source: Mapping[str, Any]) -> None:
    config = load_config(root)
    copy_value = dict(source)
    digest = copy_value.pop("content_sha256", None)
    if digest != _sha256_bytes(_canonical_bytes(copy_value)):
        raise GravityItem39Error("Legacy DR10 predictor content hash changed")
    wallaby_path = _source_path(root, config, "wallaby_predictor_source")
    if source["wallaby_predictor_sha256"] != _sha256_file(wallaby_path):
        raise GravityItem39Error("Legacy-to-WALLABY predictor binding changed")
    if int(source["counts"]["response_columns_requested"]) != 0:
        raise GravityItem39Error("response column entered Legacy predictor source")
    if int(source["counts"]["response_rows_read"]) != 0:
        raise GravityItem39Error("response row entered Legacy predictor source")


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def _predictor_quality_reasons(
    wallaby: Mapping[str, Any], optical: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
    quality = config["predictor_quality"]
    checks = {
        "optical_match_separation": _as_float(optical["match_separation_arcsec"])
        <= float(quality["maximum_optical_match_separation_arcsec"]),
        "g_minus_i": float(quality["minimum_g_minus_i_ab"])
        <= _as_float(optical["g_minus_i_ab"])
        <= float(quality["maximum_g_minus_i_ab"]),
        "stellar_mass": float(quality["minimum_log10_stellar_mass_msun"])
        <= _as_float(optical["log10_stellar_mass_msun"])
        <= float(quality["maximum_log10_stellar_mass_msun"]),
        "effective_radius": float(quality["minimum_effective_radius_kpc"])
        <= _as_float(optical["effective_radius_kpc"])
        <= float(quality["maximum_effective_radius_kpc"]),
        "axis_ratio": _as_float(optical["axis_ratio"]) >= float(quality["minimum_axis_ratio"]),
        "screen_radius": _as_float(wallaby["screen_radius_kpc"])
        >= float(quality["minimum_screen_radius_kpc"]),
    }
    return [name for name, passed in checks.items() if not passed]


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    if str(config["predictor_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem39Error("Item 39 predictor freeze is not bound")
    wallaby_path = _source_path(root, config, "wallaby_predictor_source")
    legacy_path = _source_path(root, config, "legacy_predictor_source")
    wallaby = _read_json(wallaby_path)
    legacy = _read_json(legacy_path)
    validate_wallaby_predictor_source(root, wallaby)
    validate_legacy_predictor_source(root, legacy)
    optical_by_key = {
        (str(row["galaxy"]), str(row["team_release_kin"])): row for row in legacy["records"]
    }
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in wallaby["records"]:
        key = (str(row["name"]), str(row["team_release_kin"]))
        optical = optical_by_key.get(key)
        if optical is None:
            rejected.append(
                {
                    "name": key[0],
                    "team_release_kin": key[1],
                    "reasons": ["no_accepted_optical_match"],
                }
            )
            continue
        reasons = _predictor_quality_reasons(row, optical, config)
        if reasons:
            rejected.append({"name": key[0], "team_release_kin": key[1], "reasons": reasons})
            continue
        stellar_mass = float(optical["stellar_mass_msun"])
        gas_mass = float(config["constants"]["helium_mass_factor"]) * float(row["hi_mass_msun"])
        total_baryonic = stellar_mass + gas_mass
        screen_radius = float(row["screen_radius_kpc"])
        effective_radius = float(optical["effective_radius_kpc"])
        eligible.append(
            {
                "name": key[0],
                "team_release_kin": key[1],
                "ra": row["ra"],
                "dec": row["dec"],
                "distance_mpc": row["distance_mpc"],
                "hi_mass_msun": row["hi_mass_msun"],
                "stellar_mass_msun": optical["stellar_mass_msun"],
                "total_baryonic_mass_msun": f"{total_baryonic:.12e}",
                "gas_fraction": f"{(gas_mass / total_baryonic):.12e}",
                "screen_radius_kpc": row["screen_radius_kpc"],
                "effective_radius_kpc": optical["effective_radius_kpc"],
                "effective_to_screen_ratio": f"{(effective_radius / screen_radius):.12e}",
                "axis_ratio": optical["axis_ratio"],
                "optical_match_separation_arcsec": optical["match_separation_arcsec"],
            }
        )
    if not eligible:
        raise GravityItem39Error("no Item 39 source-only sample survived")

    log_mass = np.log10(np.asarray([float(row["total_baryonic_mass_msun"]) for row in eligible]))
    mass_edges = np.quantile(log_mass, [0.25, 0.5, 0.75])
    ratios = np.asarray([float(row["effective_to_screen_ratio"]) for row in eligible])
    ratio_median = float(np.median(ratios))
    cells: dict[str, list[dict[str, Any]]] = {}
    salt = str(config["sample_boundary"]["split_salt"])
    for row, mass, ratio in zip(eligible, log_mass, ratios, strict=True):
        mass_bin = int(np.searchsorted(mass_edges, mass, side="right"))
        ratio_bin = int(ratio > ratio_median)
        cell = f"mass{mass_bin}|screen_ratio{ratio_bin}"
        enriched = dict(row)
        enriched["source_cell"] = cell
        identity = f"{row['name']}|{row['team_release_kin']}"
        enriched["selection_hash"] = _split_hash(f"select|{identity}", salt)
        enriched["confirmation_hash"] = _split_hash(f"confirm|{identity}", salt)
        cells.setdefault(cell, []).append(enriched)
    for rows in cells.values():
        rows.sort(key=lambda row: str(row["selection_hash"]))

    maximum_exploration = int(config["sample_boundary"]["maximum_exploration_galaxies"])
    fraction = float(config["sample_boundary"]["confirmation_fraction"])
    desired_total = math.ceil(maximum_exploration / (1.0 - fraction))
    desired_total = min(desired_total, len(eligible))
    selected: list[dict[str, Any]] = []
    cell_names = sorted(cells)
    round_index = 0
    while len(selected) < desired_total:
        added = False
        for cell in cell_names:
            if round_index < len(cells[cell]) and len(selected) < desired_total:
                selected.append(cells[cell][round_index])
                added = True
        if not added:
            break
        round_index += 1

    confirmation_count = max(
        int(config["sample_boundary"]["minimum_reserved_confirmation_galaxies"]),
        round(len(selected) * fraction),
    )
    confirmation_count = min(confirmation_count, max(len(selected) - 1, 0))
    selected_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        selected_by_cell.setdefault(str(row["source_cell"]), []).append(row)
    quotas = {
        cell: min(len(rows), math.floor(len(rows) * fraction))
        for cell, rows in selected_by_cell.items()
    }
    remaining = confirmation_count - sum(quotas.values())
    order = sorted(
        selected_by_cell,
        key=lambda cell: (
            -(len(selected_by_cell[cell]) * fraction - quotas[cell]),
            _split_hash(f"quota|{cell}", salt),
        ),
    )
    while remaining > 0:
        progressed = False
        for cell in order:
            if remaining <= 0:
                break
            if quotas[cell] < len(selected_by_cell[cell]):
                quotas[cell] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    objects: list[dict[str, Any]] = []
    outer_folds = int(config["evaluation"]["outer_folds"])
    fold_salt = str(config["evaluation"]["fold_salt"])
    for cell, rows in sorted(selected_by_cell.items()):
        ordered = sorted(rows, key=lambda row: str(row["confirmation_hash"]))
        confirmations = {
            (str(row["name"]), str(row["team_release_kin"])) for row in ordered[: quotas[cell]]
        }
        for row in rows:
            identity = f"{row['name']}|{row['team_release_kin']}"
            role = (
                "reserved_confirmation"
                if (str(row["name"]), str(row["team_release_kin"])) in confirmations
                else "exploration"
            )
            output = {
                key: value
                for key, value in row.items()
                if key not in {"selection_hash", "confirmation_hash"}
            }
            output["role"] = role
            output["outer_fold"] = int(_split_hash(identity, fold_salt), 16) % outer_folds
            output["response_read"] = False
            objects.append(output)
    objects.sort(key=lambda row: (str(row["name"]), str(row["team_release_kin"])))
    role_counts = Counter(str(row["role"]) for row in objects)
    cell_counts: dict[str, dict[str, int]] = {}
    for row in objects:
        cell_counts.setdefault(
            str(row["source_cell"]), {"exploration": 0, "reserved_confirmation": 0}
        )[str(row["role"])] += 1
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-sample-manifest-1.0",
            "item": 39,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "predictor_freeze_commit": config["predictor_freeze_commit"],
            "wallaby_predictor_sha256": _sha256_file(wallaby_path),
            "legacy_predictor_sha256": _sha256_file(legacy_path),
            "predictor_quality": config["predictor_quality"],
            "mass_quartile_edges_log10_msun": [f"{value:.12e}" for value in mass_edges],
            "effective_to_screen_ratio_median": f"{ratio_median:.12e}",
            "objects": objects,
            "rejected_predictors": rejected,
            "cells": cell_counts,
            "counts": {
                "wallaby_predictor_profiles": len(wallaby["records"]),
                "legacy_optical_matches": len(legacy["records"]),
                "source_quality_eligible": len(eligible),
                "source_quality_rejected": len(rejected),
                "selected_total": len(objects),
                "exploration": role_counts["exploration"],
                "reserved_confirmation": role_counts["reserved_confirmation"],
                "response_rows_read": 0,
                "confirmation_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "response_opened": False,
                "confirmation_opened": False,
                "historical_novelty_established": False,
            },
        }
    )


def validate_sample_manifest(root: Path, sample: Mapping[str, Any]) -> None:
    copy_value = dict(sample)
    digest = copy_value.pop("content_sha256", None)
    if digest != _sha256_bytes(_canonical_bytes(copy_value)):
        raise GravityItem39Error("Item 39 sample content hash changed")
    if sample != build_sample_manifest(root):
        raise GravityItem39Error("Item 39 target-blind sample drifted")
    if int(sample["counts"]["response_rows_read"]) != 0:
        raise GravityItem39Error("response entered target-blind sample")
    if int(sample["counts"]["confirmation_rows_read"]) != 0:
        raise GravityItem39Error("confirmation entered target-blind sample")


def write_sample_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "sample_manifest")
    _write_json(path, build_sample_manifest(root))
    return path


def write_wallaby_response_source(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    if str(config["sample_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem39Error("Item 39 sample freeze is not bound")
    sample_path = _source_path(root, config, "sample_manifest")
    sample = _read_json(sample_path)
    validate_sample_manifest(root, sample)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    confirmations = {
        (str(row["name"]), str(row["team_release_kin"]))
        for row in sample["objects"]
        if row["role"] == "reserved_confirmation"
    }
    conditions = []
    for row in exploration:
        name = str(row["name"]).replace("'", "''")
        release = str(row["team_release_kin"]).replace("'", "''")
        conditions.append(f"(name='{name}' AND team_release_kin='{release}')")
    wallaby = config["data_sources"]["wallaby"]
    columns = ",".join(wallaby["response_columns"])
    query = (
        f"SELECT {columns} FROM {wallaby['kinematic_table']} WHERE "
        + " OR ".join(conditions)
        + " ORDER BY name,team_release_kin"
    )
    payload, rows = _query_csv(
        str(wallaby["tap_sync_endpoint"]),
        query,
        dialect="tap_adql",
        user_agent="Invariant/Item39-WALLABY-Exploration-Responses",
    )
    expected_columns = set(wallaby["response_columns"])
    if any(set(row) != expected_columns for row in rows):
        raise GravityItem39Error("WALLABY response schema changed")
    expected = {(str(row["name"]), str(row["team_release_kin"])) for row in exploration}
    returned = {(str(row["name"]), str(row["team_release_kin"])) for row in rows}
    if returned != expected or len(rows) != len(returned):
        raise GravityItem39Error("WALLABY exploration response scope changed")
    if returned & confirmations:
        raise GravityItem39Error("WALLABY confirmation response was returned")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-wallaby-exploration-response-1.0",
            "item": 39,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "predictor_freeze_commit": config["predictor_freeze_commit"],
            "sample_freeze_commit": config["sample_freeze_commit"],
            "sample_manifest_sha256": _sha256_file(sample_path),
            "query_identity_count": len(expected),
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "records": rows,
            "counts": {
                "exploration_response_rows": len(rows),
                "confirmation_response_rows": 0,
                "post_response_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "confirmation_opened": False,
                "response_scope_repaired_after_access": False,
            },
        }
    )
    path = _source_path(root, config, "wallaby_response_source")
    _write_json(path, manifest)
    return path


def validate_wallaby_response_source(root: Path, source: Mapping[str, Any]) -> None:
    config = load_config(root)
    copy_value = dict(source)
    digest = copy_value.pop("content_sha256", None)
    if digest != _sha256_bytes(_canonical_bytes(copy_value)):
        raise GravityItem39Error("WALLABY response content hash changed")
    if source["sample_freeze_commit"] != config["sample_freeze_commit"]:
        raise GravityItem39Error("WALLABY response sample binding changed")
    if int(source["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem39Error("confirmation response entered Item 39")
    if int(source["counts"]["post_response_candidate_cells"]) != 0:
        raise GravityItem39Error("post-response candidate entered Item 39")


def _deserialize_wallaby_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in (
        "ra",
        "dec",
        "frequency_hz",
        "distance_mpc",
        "hi_mass_msun",
        "screen_radius_kpc",
    ):
        result[key] = float(row[key])
    for key in (
        "radius_arcsec",
        "radius_kpc",
        "surface_density_hi_msun_pc2",
        "surface_density_error_hi_msun_pc2",
        "cumulative_hi_mass_msun",
    ):
        result[key] = np.asarray(row[key], dtype=np.float64)
    return result


def _write_tsv(path: Path, fields: list[str], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _stellar_enclosed_mass(total_mass: float, radius: np.ndarray, scale: float) -> np.ndarray:
    y = np.maximum(np.asarray(radius, dtype=np.float64) / scale, 0.0)
    return total_mass * (1.0 - (1.0 + y) * np.exp(-y))


def extract_wallaby_profiles(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    wallaby_path = _source_path(root, config, "wallaby_predictor_source")
    legacy_path = _source_path(root, config, "legacy_predictor_source")
    sample_path = _source_path(root, config, "sample_manifest")
    response_path = _source_path(root, config, "wallaby_response_source")
    wallaby = _read_json(wallaby_path)
    legacy = _read_json(legacy_path)
    sample = _read_json(sample_path)
    response = _read_json(response_path)
    validate_wallaby_predictor_source(root, wallaby)
    validate_legacy_predictor_source(root, legacy)
    validate_sample_manifest(root, sample)
    validate_wallaby_response_source(root, response)
    profiles = {
        (str(row["name"]), str(row["team_release_kin"])): _deserialize_wallaby_profile(row)
        for row in wallaby["records"]
    }
    optical = {(str(row["galaxy"]), str(row["team_release_kin"])): row for row in legacy["records"]}
    samples = {
        (str(row["name"]), str(row["team_release_kin"])): row
        for row in sample["objects"]
        if row["role"] == "exploration"
    }
    feature_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    galaxy_receipts: list[dict[str, Any]] = []
    constants = config["constants"]
    quality = config["quality"]
    g_constant = float(constants["gravitational_constant_kpc_km2_s2_msun"])
    acceleration_conversion = 1.0e6 / 3.085677581491367e19
    a0 = float(constants["acceleration_scale_m_s2"])
    for raw in response["records"]:
        key = (str(raw["name"]), str(raw["team_release_kin"]))
        if key not in samples or key not in profiles or key not in optical:
            raise GravityItem39Error(f"response identity lacks frozen predictors: {key}")
        profile = profiles[key]
        optical_row = optical[key]
        sample_row = samples[key]
        reasons: list[str] = []
        try:
            inclination = float(raw["Inc_model"])
            qflag = float(raw["QFlag_model"])
            radius_arcsec = _parse_vector(raw["Rad"])
            velocity = _parse_vector(raw["Vrot_model"])
            velocity_error = _parse_vector(raw["e_Vrot_model"])
            inclination_error = _parse_vector(raw["e_Vrot_model_inc"])
            if not (
                len(radius_arcsec) == len(velocity) == len(velocity_error) == len(inclination_error)
            ):
                raise GravityItem39Error("rotation response vector lengths differ")
        except (ValueError, GravityItem39Error) as exc:
            galaxy_receipts.append(
                {
                    "name": key[0],
                    "team_release_kin": key[1],
                    "quality_pass": False,
                    "quality_failure_reasons": [f"parser:{exc}"],
                    "raw_rotation_points": 0,
                    "accepted_rotation_points": 0,
                }
            )
            continue
        if qflag != float(quality["required_qflag_model"]):
            reasons.append("qflag")
        if not (
            float(quality["minimum_inclination_degrees"])
            <= inclination
            <= float(quality["maximum_inclination_degrees"])
        ):
            reasons.append("inclination")
        radius_kpc = (
            radius_arcsec
            * profile["distance_mpc"]
            * 1000.0
            / float(constants["arcseconds_per_radian"])
        )
        total_error = np.sqrt(
            np.maximum(velocity_error, 0.0) ** 2 + np.maximum(inclination_error, 0.0) ** 2
        )
        within = (radius_kpc > 0.0) & (radius_kpc <= profile["screen_radius_kpc"])
        valid = within & (velocity >= float(quality["minimum_speed_km_s"]))
        valid &= total_error / np.maximum(velocity, 1e-12) <= float(
            quality["maximum_fractional_speed_error"]
        )
        if int(np.sum(valid)) < int(quality["minimum_rotation_points"]):
            reasons.append("insufficient_rotation_points")
        if float(np.mean(within)) < float(quality["minimum_fraction_response_radii_within_screen"]):
            reasons.append("screen_overlap")
        passed = not reasons
        accepted = int(np.sum(valid)) if passed else 0
        if passed:
            indices = np.flatnonzero(valid)
            accepted_radius = radius_kpc[indices]
            stellar_mass = float(optical_row["stellar_mass_msun"])
            scale_radius = float(optical_row["effective_radius_kpc"]) / 1.678
            gas_enclosed = np.interp(
                accepted_radius,
                profile["radius_kpc"],
                profile["cumulative_hi_mass_msun"],
                left=0.0,
                right=profile["hi_mass_msun"],
            ) * float(constants["helium_mass_factor"])
            stellar_enclosed = _stellar_enclosed_mass(stellar_mass, accepted_radius, scale_radius)
            baryonic_enclosed = gas_enclosed + stellar_enclosed
            total_baryonic = stellar_mass + float(constants["helium_mass_factor"]) * float(
                profile["hi_mass_msun"]
            )
            enclosed_fraction = np.clip(baryonic_enclosed / total_baryonic, 1e-8, 1.0)
            source_radius = profile["radius_kpc"]
            source_gas = profile["cumulative_hi_mass_msun"] * float(constants["helium_mass_factor"])
            source_stars = _stellar_enclosed_mass(stellar_mass, source_radius, scale_radius)
            source_baryonic = np.maximum(source_gas + source_stars, 1e-12)
            source_slope = np.gradient(
                np.log(source_baryonic), np.log(np.maximum(source_radius, 1e-12))
            )
            enclosed_slope = np.interp(
                accepted_radius,
                source_radius,
                source_slope,
                left=source_slope[0],
                right=source_slope[-1],
            )
            local_sigma = np.interp(
                accepted_radius,
                profile["radius_kpc"],
                profile["surface_density_hi_msun_pc2"],
            )
            gbar_km2_s2_kpc = g_constant * baryonic_enclosed / np.square(accepted_radius)
            gbar_m_s2 = gbar_km2_s2_kpc * acceleration_conversion
            u = gbar_m_s2 / a0
            vbar = np.sqrt(g_constant * baryonic_enclosed / accepted_radius)
            x = accepted_radius / profile["screen_radius_kpc"]
            h_values = boundary_coordinates(enclosed_fraction, x, enclosed_slope)
            for output_index, source_index in enumerate(indices):
                feature_rows.append(
                    {
                        "galaxy": key[0],
                        "team_release_kin": key[1],
                        "point_index": output_index,
                        "outer_fold": sample_row["outer_fold"],
                        "source_cell": sample_row["source_cell"],
                        "radius_kpc": f"{accepted_radius[output_index]:.12e}",
                        "radius_over_screen": f"{x[output_index]:.12e}",
                        "local_hi_surface_density": f"{local_sigma[output_index]:.12e}",
                        "enclosed_baryonic_mass_msun": f"{baryonic_enclosed[output_index]:.12e}",
                        "enclosed_fraction": f"{enclosed_fraction[output_index]:.12e}",
                        "enclosed_log_slope": f"{enclosed_slope[output_index]:.12e}",
                        "gbar_m_s2": f"{gbar_m_s2[output_index]:.12e}",
                        "u": f"{u[output_index]:.12e}",
                        "vbar_km_s": f"{vbar[output_index]:.12e}",
                        "h_equipartition": f"{h_values[0, output_index]:.12e}",
                        "h_quasilocal": f"{h_values[1, output_index]:.12e}",
                        "h_wedge": f"{h_values[2, output_index]:.12e}",
                        "h_flow": f"{h_values[3, output_index]:.12e}",
                        "total_baryonic_mass_msun": sample_row["total_baryonic_mass_msun"],
                        "gas_fraction": sample_row["gas_fraction"],
                        "effective_to_screen_ratio": sample_row["effective_to_screen_ratio"],
                        "axis_ratio": sample_row["axis_ratio"],
                        "distance_mpc": sample_row["distance_mpc"],
                        "inclination_degrees": f"{inclination:.12e}",
                    }
                )
                response_rows.append(
                    {
                        "galaxy": key[0],
                        "team_release_kin": key[1],
                        "point_index": output_index,
                        "observed_speed_km_s": f"{velocity[source_index]:.12e}",
                        "observed_speed_error_km_s": f"{total_error[source_index]:.12e}",
                    }
                )
        galaxy_receipts.append(
            {
                "name": key[0],
                "team_release_kin": key[1],
                "outer_fold": sample_row["outer_fold"],
                "source_cell": sample_row["source_cell"],
                "quality_pass": passed,
                "quality_failure_reasons": reasons,
                "raw_rotation_points": len(radius_arcsec),
                "accepted_rotation_points": accepted,
                "inclination_degrees": f"{inclination:.12e}",
                "qflag_model": f"{qflag:.12e}",
            }
        )
    feature_fields = [
        "galaxy",
        "team_release_kin",
        "point_index",
        "outer_fold",
        "source_cell",
        "radius_kpc",
        "radius_over_screen",
        "local_hi_surface_density",
        "enclosed_baryonic_mass_msun",
        "enclosed_fraction",
        "enclosed_log_slope",
        "gbar_m_s2",
        "u",
        "vbar_km_s",
        "h_equipartition",
        "h_quasilocal",
        "h_wedge",
        "h_flow",
        "total_baryonic_mass_msun",
        "gas_fraction",
        "effective_to_screen_ratio",
        "axis_ratio",
        "distance_mpc",
        "inclination_degrees",
    ]
    response_fields = [
        "galaxy",
        "team_release_kin",
        "point_index",
        "observed_speed_km_s",
        "observed_speed_error_km_s",
    ]
    feature_path = _source_path(root, config, "point_features")
    rotation_path = _source_path(root, config, "rotation_responses")
    summary_path = _source_path(root, config, "extraction_summary")
    _write_tsv(feature_path, feature_fields, feature_rows)
    _write_tsv(rotation_path, response_fields, response_rows)
    passing = sum(bool(row["quality_pass"]) for row in galaxy_receipts)
    exploration_count = int(sample["counts"]["exploration"])
    quality_count_pass = passing >= int(quality["minimum_quality_passing_exploration_galaxies"])
    quality_fraction_pass = passing / max(exploration_count, 1) >= float(
        quality["minimum_quality_retention_fraction"]
    )
    summary = _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-extraction-summary-1.0",
            "item": 39,
            "sample_freeze_commit": config["sample_freeze_commit"],
            "sample_manifest_sha256": _sha256_file(sample_path),
            "response_source_sha256": _sha256_file(response_path),
            "point_features_sha256": _sha256_file(feature_path),
            "rotation_responses_sha256": _sha256_file(rotation_path),
            "galaxies": galaxy_receipts,
            "counts": {
                "exploration_response_rows": len(response["records"]),
                "quality_passing_galaxies": passing,
                "quality_failing_galaxies": exploration_count - passing,
                "accepted_rotation_points": len(response_rows),
                "confirmation_response_rows": 0,
                "post_response_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "quality": {
                "count_gate_passed": quality_count_pass,
                "retention_fraction_gate_passed": quality_fraction_pass,
                "overall_passed": quality_count_pass and quality_fraction_pass,
            },
            "claims": {
                "confirmation_opened": False,
                "failed_identity_replacement": False,
            },
        }
    )
    _write_json(summary_path, summary)
    return {
        "point_features": feature_path,
        "rotation_responses": rotation_path,
        "extraction_summary": summary_path,
    }


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    shape = tuple(int(value) for value in config["candidate_generator"]["grid_shape_per_niche"])
    indices = np.indices(shape, dtype=np.int16).reshape(4, -1).T
    cells = indices.shape[0]
    lane = np.repeat(np.arange(4, dtype=np.int8), cells)
    tiled = np.tile(indices, (4, 1))
    return {
        "candidate_id": np.arange(4 * cells, dtype=np.int64),
        "lane": lane,
        "amplitude_index": tiled[:, 0],
        "exponent_index": tiled[:, 1],
        "transition_index": tiled[:, 2],
        "shape_index": tiled[:, 3],
    }


def _candidate_parameters(
    candidates: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grids = config["candidate_generator"]["parameter_grids"]
    amplitude = np.asarray(grids["amplitude"], dtype=np.float64)[
        np.asarray(candidates["amplitude_index"], dtype=np.int64)
    ]
    exponent = np.asarray(grids["exponent"], dtype=np.float64)[
        np.asarray(candidates["exponent_index"], dtype=np.int64)
    ]
    transition = np.asarray(grids["transition_u"], dtype=np.float64)[
        np.asarray(candidates["transition_index"], dtype=np.int64)
    ]
    shape = np.asarray(grids["shape"], dtype=np.float64)[
        np.asarray(candidates["shape_index"], dtype=np.int64)
    ]
    return amplitude, exponent, transition, shape


def boundary_coordinates(
    enclosed_fraction: np.ndarray,
    radius_over_screen: np.ndarray,
    enclosed_log_slope: np.ndarray,
) -> np.ndarray:
    """Return the four frozen dimensionless nested-screen coordinates."""

    fraction = np.asarray(enclosed_fraction, dtype=np.float64)
    radius = np.asarray(radius_over_screen, dtype=np.float64)
    slope = np.asarray(enclosed_log_slope, dtype=np.float64)
    if not (fraction.shape == radius.shape == slope.shape) or fraction.ndim != 1:
        raise GravityItem39Error("boundary inputs must be aligned vectors")
    if (
        np.any(~np.isfinite(fraction))
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(slope))
    ):
        raise GravityItem39Error("boundary inputs must be finite")
    if np.any((fraction < 0.0) | (fraction > 1.0)) or np.any(radius <= 0.0):
        raise GravityItem39Error("boundary inputs outside frozen domains")
    floor = 1e-12
    area = np.square(radius)
    equipartition = np.abs(fraction - area) / (fraction + area + floor)
    compactness = fraction / radius
    quasilocal = np.abs(compactness - 1.0) / (compactness + 1.0 + floor)
    mass_cross = np.clip(4.0 * fraction * (1.0 - fraction), 0.0, 1.0)
    radial_cross = np.clip(4.0 * radius / np.square(1.0 + radius), 0.0, 1.0)
    wedge = np.sqrt(mass_cross * radial_cross)
    flow = np.abs(slope - 2.0) / (np.abs(slope) + 2.0 + floor)
    return np.stack((equipartition, quasilocal, wedge, flow), axis=0)


def predict_multiplier(
    candidates: Mapping[str, np.ndarray],
    u: np.ndarray,
    enclosed_fraction: np.ndarray,
    radius_over_screen: np.ndarray,
    enclosed_log_slope: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Return the universal motion/lensing multiplier for candidate rows."""

    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or np.any(~np.isfinite(u)) or np.any(u <= 0.0):
        raise GravityItem39Error("u must be a finite positive vector")
    coordinates = boundary_coordinates(enclosed_fraction, radius_over_screen, enclosed_log_slope)
    if coordinates.shape[1] != len(u):
        raise GravityItem39Error("boundary and acceleration vectors differ")
    lane = np.asarray(candidates["lane"], dtype=np.int8)
    amplitude, exponent, transition, shape = _candidate_parameters(candidates, config)
    result = np.empty((len(lane), len(u)), dtype=np.float64)
    uu = u[None, :]
    for lane_id in range(4):
        mask = lane == lane_id
        if not np.any(mask):
            continue
        aa = amplitude[mask, None]
        pp = exponent[mask, None]
        tt = transition[mask, None]
        ss = shape[mask, None]
        hh = coordinates[lane_id][None, :]
        envelope = aa * np.power(uu, -pp) * np.power(1.0 + np.power(uu / tt, ss), -1.0 / ss)
        if lane_id == 0:
            boundary = 0.05 + 0.95 * np.power(hh, ss)
        elif lane_id == 1:
            boundary = 0.05 + 0.95 * np.power(hh / (0.25 + hh), ss)
        elif lane_id == 2:
            boundary = 0.05 + 0.95 * np.power(np.sin(0.5 * np.pi * hh), ss)
        else:
            boundary = 0.05 + 0.95 * (1.0 - np.exp(-ss * hh))
        result[mask] = 1.0 + envelope * boundary
    return result


def metric_observables(g_bar: np.ndarray, multiplier: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the frozen zero-slip weak-field closure to motion and light."""

    g_bar = np.asarray(g_bar, dtype=np.float64)
    multiplier = np.asarray(multiplier, dtype=np.float64)
    if g_bar.shape != multiplier.shape:
        raise GravityItem39Error("metric inputs differ")
    field = g_bar * multiplier
    return {
        "g_dynamics": field,
        "grad_phi": field,
        "grad_psi": field,
        "lensing_integrand_grad_phi_plus_psi": 2.0 * field,
    }


def fixed_control_multiplier(name: str, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if name == "baryonic_newton":
        return np.ones_like(u)
    if name == "mond_RAR":
        return 1.0 / (-np.expm1(-np.sqrt(u)))
    if name == "item38_selected":
        return 1.0 + 2.75 * np.power(u, -0.45) * np.power(1.0 + np.power(u / 0.01, 0.5), -2.0)
    raise GravityItem39Error(f"unknown fixed control: {name}")


def admissible_candidates(
    config: Mapping[str, Any], *, batch_size: int = 4096
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    admission = config["candidate_generator"]["admissibility"]
    u = np.logspace(
        float(admission["probe_log10_u_min"]),
        float(admission["probe_log10_u_max"]),
        int(admission["probe_points"]),
    )
    fraction = np.clip(0.05 + 0.9 / (1.0 + np.exp(-np.linspace(-3.0, 3.0, len(u)))), 0, 1)
    radius = np.logspace(-1.0, 0.15, len(u))
    slope = 0.4 + 2.4 * np.square(np.sin(np.linspace(0.0, np.pi, len(u))))
    keep_parts: list[np.ndarray] = []
    rejection_counts: Counter[str] = Counter()
    signatures: set[bytes] = set()
    decimals = int(admission["behavior_signature_decimals"])
    total = len(raw["candidate_id"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        rows = {key: value[start:stop] for key, value in raw.items()}
        values = predict_multiplier(rows, u, fraction, radius, slope, config)
        finite = np.all(np.isfinite(values), axis=1)
        bounded = finite & np.all(values >= float(admission["minimum_multiplier"]), axis=1)
        bounded &= np.all(values <= float(admission["maximum_multiplier"]), axis=1)
        local = bounded & (
            np.abs(np.log10(values[:, -1]))
            <= float(admission["maximum_high_acceleration_log10_deviation"])
        )
        material = local & (
            values[:, 16] >= float(admission["minimum_low_acceleration_multiplier"])
        )
        monotone = material & np.all(
            np.diff(values, axis=1) <= float(admission["monotone_nonincreasing_tolerance"]),
            axis=1,
        )
        test_u = np.asarray([0.01, 0.01], dtype=np.float64)
        low_boundary = predict_multiplier(
            rows,
            test_u,
            np.asarray([0.02, 0.85]),
            np.asarray([0.15, 0.95]),
            np.asarray([2.0, 0.4]),
            config,
        )
        boundary_span = np.abs(np.diff(np.log10(low_boundary), axis=1)[:, 0])
        boundary_material = monotone & (
            boundary_span >= float(admission["minimum_boundary_log10_span"])
        )
        rejection_counts["nonfinite"] += int(np.sum(~finite))
        rejection_counts["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection_counts["no_local_limit"] += int(np.sum(bounded & ~local))
        rejection_counts["immaterial_low_acceleration"] += int(np.sum(local & ~material))
        rejection_counts["nonmonotone"] += int(np.sum(material & ~monotone))
        rejection_counts["ordinary_rewrite_boundary_immaterial"] += int(
            np.sum(monotone & ~boundary_material)
        )
        admitted_local = np.flatnonzero(boundary_material)
        keep_parts.append(admitted_local + start)
        rounded = np.round(np.log10(values[admitted_local]), decimals=decimals)
        for row in rounded:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    admitted = {key: value[keep] for key, value in raw.items()}
    audit = {
        "raw_candidates": total,
        "admitted_candidates": len(keep),
        "rejected_candidates": int(total - len(keep)),
        "admitted_by_lane": {str(lane): int(np.sum(admitted["lane"] == lane)) for lane in range(4)},
        "behavioral_equivalence_classes": len(signatures),
        "rejection_counts_nonexclusive": dict(sorted(rejection_counts.items())),
    }
    return admitted, audit


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem39Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition, shape = _candidate_parameters(row, config)
    lane_id = int(row["lane"][0])
    niche = config["candidate_generator"]["niches"][lane_id]
    return {
        "candidate_id": candidate_id,
        "lane_id": lane_id,
        "lane": niche["name"],
        "creativity_label": niche["creativity_label"],
        "boundary_coordinate": niche["boundary_coordinate"],
        "parameters": {
            "amplitude": float(amplitude[0]),
            "exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
            "shape": float(shape[0]),
        },
    }


def build_candidate_manifest(root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config(root))
    admitted, audit = admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-candidate-manifest-1.0",
            "item": 39,
            "config_contract_sha256": _contract_digest(config),
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_accessed": False,
            "confirmation_accessed": False,
            "paid_api_calls": 0,
            "raw_candidates": audit["raw_candidates"],
            "admitted_candidates": audit["admitted_candidates"],
            "admitted_by_lane": audit["admitted_by_lane"],
            "behavioral_equivalence_classes": audit["behavioral_equivalence_classes"],
            "rejection_counts_nonexclusive": audit["rejection_counts_nonexclusive"],
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "niches": config["candidate_generator"]["niches"],
            "metric_contract": config["weak_field_metric_contract"],
            "claim_boundaries": [
                "candidate mappings are weak-field phenomenological projections, not complete theories",
                "screen, quasilocal, and MOND-like ideas are known prior art",
                "potentially new synthesis does not establish historical novelty",
                "behavioral distinction from controls must be demonstrated on held-out data",
                "no response, confirmation value, or paid model call entered generation",
            ],
        }
    )


def build_exposure_manifest(root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config(root))
    path = root / str(config["independence"]["item10_sample_path"])
    item10 = _read_json(path)
    names = sorted({str(row["name"]) for row in item10["objects"]})
    coordinates = sorted(
        {(f"{float(row['ra']):.12e}", f"{float(row['dec']):.12e}") for row in item10["objects"]}
    )
    role_counts = Counter(str(row["role"]) for row in item10["objects"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-predecessor-exposure-1.0",
            "item": 39,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "source_path": str(config["independence"]["item10_sample_path"]),
            "source_sha256": _sha256_file(path),
            "excluded_names": names,
            "excluded_coordinates": [{"ra": ra, "dec": dec} for ra, dec in coordinates],
            "coordinate_exclusion_arcseconds": float(
                config["independence"]["coordinate_exclusion_arcseconds"]
            ),
            "role_counts": dict(sorted(role_counts.items())),
            "rules": [
                "exclude every Item 10 exploration identity",
                "exclude every Item 10 reserved-confirmation identity",
                "exclude coordinate neighbors before response access",
                "never repair a failed sample identity after response access",
            ],
            "response_values_read_while_building": 0,
        }
    )


def write_freeze_manifests(root: Path) -> dict[str, Path]:
    config = load_config(root)
    candidate_path = _source_path(root, config, "candidate_manifest")
    exposure_path = _source_path(root, config, "exposure_manifest")
    _write_json(candidate_path, build_candidate_manifest(root, config))
    _write_json(exposure_path, build_exposure_manifest(root, config))
    return {"candidate_manifest": candidate_path, "exposure_manifest": exposure_path}


def check_freeze(root: Path) -> dict[str, Any]:
    config = load_config(root)
    candidate_path = _source_path(root, config, "candidate_manifest")
    exposure_path = _source_path(root, config, "exposure_manifest")
    candidate = _read_json(candidate_path)
    exposure = _read_json(exposure_path)
    if candidate != build_candidate_manifest(root, config):
        raise GravityItem39Error("candidate manifest drifted")
    if exposure != build_exposure_manifest(root, config):
        raise GravityItem39Error("exposure manifest drifted")
    return {
        "status": "ITEM39_SCIENTIFIC_FREEZE_VALID",
        "candidate_manifest_sha256": _sha256_file(candidate_path),
        "exposure_manifest_sha256": _sha256_file(exposure_path),
        "response_rows_read": 0,
        "confirmation_rows_read": 0,
        "paid_api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "check-freeze",
            "wallaby-predictors",
            "legacy-predictors",
            "sample",
            "responses",
            "extract",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "freeze":
        print(
            json.dumps(
                {key: str(value) for key, value in write_freeze_manifests(args.root).items()},
                sort_keys=True,
            )
        )
    elif args.command == "check-freeze":
        print(json.dumps(check_freeze(args.root), sort_keys=True))
    elif args.command == "wallaby-predictors":
        print(write_wallaby_predictor_source(args.root))
    elif args.command == "legacy-predictors":
        print(write_legacy_predictor_source(args.root))
    elif args.command == "sample":
        print(write_sample_manifest(args.root))
    elif args.command == "responses":
        print(write_wallaby_response_source(args.root))
    else:
        print(
            json.dumps(
                {key: str(value) for key, value in extract_wallaby_profiles(args.root).items()},
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
