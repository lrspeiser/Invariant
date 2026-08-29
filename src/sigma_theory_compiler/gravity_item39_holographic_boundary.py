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
        choices=("freeze", "check-freeze", "wallaby-predictors", "legacy-predictors"),
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
    else:
        print(write_legacy_predictor_source(args.root))


if __name__ == "__main__":
    main()
