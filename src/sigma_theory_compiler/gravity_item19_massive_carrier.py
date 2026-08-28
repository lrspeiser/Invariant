"""Frozen Item 19 massive-carrier experiment on fresh DiskMass galaxies.

The experiment keeps two mechanism niches separate.  A positive-energy scalar supplies an
attractive Yukawa force; a positive-energy Proca field supplies a repulsive finite-range force
between like universal charges.  Both are normalized to the measured short-distance Newton
constant and convolved with exponential stellar and gas disks.  The observable repulsive branch
is explicitly labeled as an algebraic rewrite of Item 16's subtracted-Yukawa response, rather than
as a new formula.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import time
import urllib.parse
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import iv, j1, kv

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    _backend,
    _canonical_bytes,
    _content_hashed,
    _download,
    _format_float,
    _hmac_rank,
    _improvement,
    _mse,
    _parse_vizier_tsv,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _to_numpy,
    _verify_content_hash,
    _write_json,
)
from sigma_theory_compiler.gravity_item18_diskmass_antiscreening import (
    _as_float,
    _as_int,
    _disk_velocity_sq,
)

CONFIG_PATH = Path("configs/gravity_item19_massive_carrier_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item19_massive_carrier.py")
DEPENDENCY_PATHS = (
    Path("src/sigma_theory_compiler/gravity_item16_s4tm_qed_field.py"),
    Path("src/sigma_theory_compiler/gravity_item18_diskmass_antiscreening.py"),
)
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem19Error(RuntimeError):
    """Raised when an Item 19 freeze, physics, or replay invariant is violated."""


def load_config(root: Path) -> dict[str, Any]:
    """Load the frozen contract and enforce its nonnegotiable scope."""

    config = _read_json(root / CONFIG_PATH)
    if config.get("schema_version") != "invariant-gravity-item19-massive-carrier-config-1.0":
        raise GravityItem19Error("unexpected Item 19 config schema")
    if int(config.get("item", -1)) != 19:
        raise GravityItem19Error("Item 19 config changed item number")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem19Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem19Error("paid API calls are outside Item 19")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem19Error("post-response cells entered Item 19")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem19Error("stable gravity goal changed")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem19Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem19Error(f"{label} is not an ancestor of HEAD")


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem19Error("scientific contract differs from frozen commit")
    for path in (MODULE_PATH, *DEPENDENCY_PATHS):
        frozen = _git(root, "show", f"{commit}:{path.as_posix()}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(root / path):
            raise GravityItem19Error(f"Item 19 dependency differs from freeze: {path}")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "predictor_diskmass",
        "predictor_springob",
        "predictor_kinematic",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "kernel_table",
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
        "predictor_diskmass",
        "predictor_springob",
        "predictor_kinematic",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "kernel_table",
    ):
        repository_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repository_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem19Error(f"{key} differs from sample freeze")


def _point_kernel(u: np.ndarray) -> np.ndarray:
    return (1.0 + u) * np.exp(-u)


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Generate and locally filter the exact PCG64 carrier programs."""

    generator = config["candidate_generator"]
    count = int(generator["raw_parameter_cells"])
    rng = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    family = rng.integers(0, len(generator["families"]), count, dtype=np.int8)
    attractive = family < 2
    two_carriers = family % 2 == 1
    coupling_uniform = rng.random(count)
    attractive_log = (
        float(generator["attractive_total_coupling_log10_min"])
        + (
            float(generator["attractive_total_coupling_log10_max"])
            - float(generator["attractive_total_coupling_log10_min"])
        )
        * coupling_uniform
    )
    repulsive_log = (
        float(generator["repulsive_total_coupling_log10_min"])
        + (
            math.log10(float(generator["repulsive_total_coupling_max"]))
            - float(generator["repulsive_total_coupling_log10_min"])
        )
        * coupling_uniform
    )
    total = np.power(10.0, np.where(attractive, attractive_log, repulsive_log))
    lambda_min = float(generator["lambda_log10_kpc_min"])
    lambda_max = float(generator["lambda_log10_kpc_max"])
    log_lambda1 = rng.uniform(lambda_min, lambda_max, count)
    log_ratio = rng.uniform(
        float(generator["secondary_range_ratio_log10_min"]),
        float(generator["secondary_range_ratio_log10_max"]),
        count,
    )
    log_lambda2 = np.clip(log_lambda1 + log_ratio, lambda_min, lambda_max)
    secondary_fraction = rng.uniform(
        float(generator["secondary_weight_fraction_min"]),
        float(generator["secondary_weight_fraction_max"]),
        count,
    )
    secondary_fraction = np.where(two_carriers, secondary_fraction, 0.0)
    alpha1 = total * (1.0 - secondary_fraction)
    alpha2 = total * secondary_fraction
    swap = two_carriers & (log_lambda2 < log_lambda1)
    saved_log = log_lambda1.copy()
    saved_alpha = alpha1.copy()
    log_lambda1[swap] = log_lambda2[swap]
    log_lambda2[swap] = saved_log[swap]
    alpha1[swap] = alpha2[swap]
    alpha2[swap] = saved_alpha[swap]
    log_lambda2[~two_carriers] = log_lambda1[~two_carriers]
    alpha2[~two_carriers] = 0.0
    lambda1 = np.power(10.0, log_lambda1)
    lambda2 = np.power(10.0, log_lambda2)
    sign = np.where(attractive, 1.0, -1.0)
    stellar_grid = np.linspace(
        float(generator["stellar_mass_to_light_min"]),
        float(generator["stellar_mass_to_light_max"]),
        int(generator["stellar_mass_to_light_count"]),
    )
    gas_grid = np.asarray(generator["gas_mass_scales"], dtype=np.float64)
    stellar = stellar_grid[rng.integers(0, len(stellar_grid), count)]
    gas = gas_grid[rng.integers(0, len(gas_grid), count)]
    denominator = 1.0 + sign * total
    filters = config["pre_response_filters"]
    constants = config["physics"]["constants"]
    kpc_m = float(constants["kpc_m"])
    lab_m = float(filters["laboratory_reference_separation_m"])
    lab_u1 = lab_m / (lambda1 * kpc_m)
    lab_u2 = lab_m / (lambda2 * kpc_m)
    lab_mu = (
        1.0 + sign * (alpha1 * _point_kernel(lab_u1) + alpha2 * _point_kernel(lab_u2))
    ) / denominator
    solar_max = np.zeros(count, dtype=np.float64)
    for radius_au in filters["solar_probe_AU"]:
        radius_m = float(radius_au) * float(constants["AU_m"])
        u1 = radius_m / (lambda1 * kpc_m)
        u2 = radius_m / (lambda2 * kpc_m)
        mu = (1.0 + sign * (alpha1 * _point_kernel(u1) + alpha2 * _point_kernel(u2))) / denominator
        solar_max = np.maximum(solar_max, np.abs(mu - 1.0))
    far_mu = 1.0 / denominator
    keep = (
        (denominator >= float(filters["minimum_short_distance_denominator"]))
        & (np.abs(lab_mu - 1.0) <= float(filters["maximum_laboratory_normalization_error"]))
        & (solar_max <= float(filters["maximum_solar_fractional_deviation"]))
        & (far_mu <= float(filters["maximum_large_distance_enhancement"]))
        & np.isfinite(far_mu)
        & (lambda1 > 0.0)
        & (lambda2 > 0.0)
    )
    arrays = {
        "family": family.astype(np.int16),
        "sign": sign,
        "alpha_total": total,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "lambda1_kpc": lambda1,
        "lambda2_kpc": lambda2,
        "stellar_mass_to_light": stellar,
        "gas_mass_scale": gas,
        "maximum_solar_fractional_deviation": solar_max,
    }
    return {key: np.ascontiguousarray(value[keep]) for key, value in arrays.items()}


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype="<f8")
        digest.update(key.encode() + b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _newton_disk_shape(x: float) -> float:
    y = x / 2.0
    return float(x / 2.0 * (iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)))


def _yukawa_disk_ratio(x: float, a: float, config: Mapping[str, Any]) -> tuple[float, float]:
    """Return g_Y/g_N for an exponential disk at x=R/Rd and a=Rd/lambda."""

    newton = _newton_disk_shape(x)
    if a == 0.0:
        effective_a = 0.0
    elif a >= 1.0e4:
        return float(math.exp(-x) / (a * newton)), 0.0
    else:
        effective_a = a

    def integrand(q: float) -> float:
        denominator = math.sqrt(q * q + effective_a * effective_a) * (1.0 + q * q) ** 1.5
        return q * q * float(j1(q * x)) / denominator

    table = config["kernel_table"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value, error = quad(
            integrand,
            0.0,
            np.inf,
            epsabs=float(table["quadrature_epsabs"]),
            epsrel=float(table["quadrature_epsrel"]),
            limit=int(table["quadrature_limit"]),
        )
    return float(value / newton), float(error / newton)


def _prepare_kernel_table(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    table = config["kernel_table"]
    log_a = np.linspace(
        float(table["log10_Rd_over_lambda_min"]),
        float(table["log10_Rd_over_lambda_max"]),
        int(table["grid_count"]),
    )
    x_values = np.asarray(table["component_x_R_over_Rd"], dtype=np.float64)
    ratios = np.empty((len(x_values), len(log_a)), dtype=np.float64)
    max_error = 0.0
    raw_min, raw_max = math.inf, -math.inf
    for x_index, x in enumerate(x_values):
        for a_index, log_value in enumerate(log_a):
            ratio, error = _yukawa_disk_ratio(float(x), 10.0 ** float(log_value), config)
            ratios[x_index, a_index] = ratio
            raw_min = min(raw_min, ratio)
            raw_max = max(raw_max, ratio)
            max_error = max(max_error, error)
    if raw_min < -1.0e-5 or raw_max > 1.0 + 1.0e-5:
        raise GravityItem19Error(f"Yukawa disk ratio escaped [0,1]: {raw_min}, {raw_max}")
    ratios = np.clip(ratios, 0.0, 1.0)
    massless_errors = []
    for x in x_values:
        ratio, _ = _yukawa_disk_ratio(float(x), 0.0, config)
        massless_errors.append(abs(ratio - 1.0))
    if max(massless_errors) > float(table["maximum_massless_relative_error"]):
        raise GravityItem19Error("massless disk-kernel limit failed")
    rng = np.random.Generator(np.random.PCG64(int(table["validation_seed"])))
    validation = []
    for x in x_values:
        for log_value in rng.uniform(
            float(table["log10_Rd_over_lambda_min"]),
            float(table["log10_Rd_over_lambda_max"]),
            int(table["validation_points"]),
        ):
            direct, _ = _yukawa_disk_ratio(float(x), 10.0 ** float(log_value), config)
            interpolated = float(np.interp(log_value, log_a, ratios[np.argmin(abs(x_values - x))]))
            scaled_error = abs(interpolated - direct) / max(abs(direct), 1.0e-6)
            validation.append(scaled_error)
    if max(validation) > float(table["maximum_interpolation_relative_error"]):
        raise GravityItem19Error(f"disk-kernel interpolation failed: {max(validation)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, log10_a=log_a, x_values=x_values, ratios=ratios)
    return {
        "raw_min_ratio": raw_min,
        "raw_max_ratio": raw_max,
        "maximum_reported_quadrature_error_ratio": max_error,
        "maximum_massless_relative_error": max(massless_errors),
        "maximum_interpolation_scaled_error": max(validation),
        "positive_spectral_disk_bound_verified": raw_min >= -1.0e-5 and raw_max <= 1.0 + 1.0e-5,
    }


def _build_sample(
    diskmass_rows: Sequence[Mapping[str, str]],
    springob_rows: Sequence[Mapping[str, str]],
    kinematic_rows: Sequence[Mapping[str, str]],
    root: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    diskmass = {_as_int(row.get("UGC", "")): row for row in diskmass_rows}
    springob = {_as_int(row.get("UGC/AGC", "")): row for row in springob_rows}
    kinematic = {_as_int(row.get("UGC", "")): row for row in kinematic_rows}
    joined: list[dict[str, Any]] = []
    maximum_delta = float(config["sample"]["maximum_velocity_mismatch_km_s"])
    minimum_snr = float(config["sample"]["minimum_springob_SNR"])
    constants = config["physics"]["constants"]
    for ugc in sorted(set(diskmass) & set(springob) & set(kinematic)):
        if ugc is None:
            continue
        d, s, k = diskmass[ugc], springob[ugc], kinematic[ugc]
        values = {
            "hrv_d": _as_float(d.get("HRV", "")),
            "hrv_s": _as_float(s.get("HRV", "")),
            "h_r": _as_float(d.get("hR", "")),
            "mu0": _as_float(d.get("mu0", "")),
            "color": _as_float(d.get("B-K", "")),
            "flux": _as_float(s.get("Sabs", "")),
            "flux_error": _as_float(s.get("e_Sabs", "")),
            "snr": _as_float(s.get("SNR", "")),
            "distance": _as_float(k.get("Dist", "")),
            "k_abs": _as_float(k.get("KMag", "")),
        }
        if any(value is None for value in values.values()):
            continue
        if (
            abs(float(values["hrv_d"]) - float(values["hrv_s"])) > maximum_delta
            or float(values["snr"]) < minimum_snr
            or min(
                float(values["h_r"]),
                float(values["flux"]),
                float(values["flux_error"]),
                float(values["distance"]),
            )
            <= 0.0
        ):
            continue
        luminosity = 10.0 ** (-0.4 * (float(values["k_abs"]) - float(constants["M_K_sun"])))
        hi_mass = (
            float(constants["HI_mass_coefficient"])
            * float(values["distance"]) ** 2
            * float(values["flux"])
        )
        joined.append(
            {
                "ugc": int(ugc),
                "name": f"UGC{int(ugc):05d}",
                "type": str(d.get("Type", "")),
                "hR_arcsec": float(values["h_r"]),
                "mu0_R_mag_arcsec2": float(values["mu0"]),
                "B_minus_K": float(values["color"]),
                "K_abs_mag": float(values["k_abs"]),
                "K_luminosity_Lsun": luminosity,
                "distance_2025_Mpc": float(values["distance"]),
                "springob_Sabs_Jy_km_s": float(values["flux"]),
                "springob_e_Sabs_Jy_km_s": float(values["flux_error"]),
                "springob_SNR": float(values["snr"]),
                "springob_telescope": str(s.get("Tel", "")),
                "HI_mass_Msun": hi_mass,
                "velocity_match_km_s": abs(float(values["hrv_d"]) - float(values["hrv_s"])),
                "mass_proxy_Msun": 0.5 * luminosity + 1.4 * hi_mass,
            }
        )
    if len(joined) != int(config["sources"]["expected_predictor_join_before_exclusions"]):
        raise GravityItem19Error(f"predictor join changed: found {len(joined)}")
    item18_path = root / str(config["sources"]["item18_sample_manifest"])
    if _sha256_file(item18_path) != str(config["sources"]["item18_sample_manifest_sha256"]):
        raise GravityItem19Error("Item 18 identity manifest changed")
    item18 = _read_json(item18_path)
    item18_ids = {int(row["ugc"]) for row in item18["objects"]}
    earlier_ids = {int(value) for value in config["sources"]["predecessor_exposed_ugc"]}
    earlier_ids.update(int(value) for value in config["sources"]["search_snippet_exposed_ugc"])
    item18_count = sum(row["ugc"] in item18_ids for row in joined)
    earlier_count = sum(row["ugc"] in earlier_ids for row in joined)
    if item18_count != int(config["sources"]["expected_item18_identity_exclusions_in_join"]):
        raise GravityItem19Error("Item 18 exclusion overlap count changed")
    if earlier_count != int(config["sources"]["expected_earlier_exposed_exclusions_in_join"]):
        raise GravityItem19Error("earlier exposed overlap count changed")
    objects = [row for row in joined if row["ugc"] not in item18_ids | earlier_ids]
    if len(objects) != int(config["sources"]["expected_prefreeze_clean_eligible"]):
        raise GravityItem19Error(f"expected 20 fresh objects, found {len(objects)}")
    objects.sort(key=lambda row: (row["mass_proxy_Msun"], row["name"]))
    confirmation: set[str] = set()
    begin = 0
    for stratum, size in enumerate(config["sample"]["mass_stratum_sizes"]):
        group = objects[begin : begin + int(size)]
        begin += int(size)
        for row in group:
            row["mass_stratum"] = stratum
            row["role_rank"] = _hmac_rank(str(config["sample"]["role_key"]), row["name"])
        confirmation.add(min(group, key=lambda row: row["role_rank"])["name"])
    if begin != len(objects):
        raise GravityItem19Error("mass strata do not exhaust fresh objects")
    exploration = [row for row in objects if row["name"] not in confirmation]
    fold_ranked = sorted(
        exploration,
        key=lambda row: _hmac_rank(str(config["sample"]["fold_key"]), row["name"]),
    )
    fold_by_name = {
        row["name"]: index % int(config["sample"]["outer_folds"])
        for index, row in enumerate(fold_ranked)
    }
    for row in objects:
        if row["name"] in confirmation:
            row["role"] = "reserved_confirmation"
            row["outer_fold"] = None
        else:
            row["role"] = "exploration"
            row["outer_fold"] = fold_by_name[row["name"]]
    counts = {
        "joined_before_exclusions": len(joined),
        "excluded_item18_identities": item18_count,
        "excluded_earlier_exposed_identities": earlier_count,
        "eligible": len(objects),
        "exploration": len(exploration),
        "reserved_confirmation": len(confirmation),
    }
    return sorted(objects, key=lambda row: row["name"]), counts


def _prior_ugc_hits(root: Path, names: Sequence[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    base = root / "runs" / "gravity" / "roadmap"
    candidates = list(base.glob("item-*.json"))
    candidates.extend(path for path in base.rglob("*response*") if path.is_file())
    for path in candidates:
        if "item-19-" in path.as_posix():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").upper()
        except OSError:
            continue
        for name in names:
            if name.upper() in text:
                hits.setdefault(name, []).append(path.relative_to(root).as_posix())
    return hits


def _carrier_mass_bounds(
    arrays: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> dict[str, float]:
    constants = config["physics"]["constants"]
    conversion = float(constants["hbar_c_eV_m"]) / float(constants["kpc_m"])
    ranges = np.concatenate([arrays["lambda1_kpc"], arrays["lambda2_kpc"]])
    masses = conversion / ranges
    return {"minimum_mass_eV": float(np.min(masses)), "maximum_mass_eV": float(np.max(masses))}


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictor_diskmass"].parent.mkdir(parents=True, exist_ok=True)
    query_map = dict(config["sources"]["predictor_queries"])
    bodies: dict[str, bytes] = {}
    headers: dict[str, dict[str, str]] = {}
    for key, url in query_map.items():
        bodies[key], headers[key] = _download(str(url))
    columns = {
        "diskmass": ("UGC", "Type", "HRV", "Dist", "KMag", "B-K", "mu0", "hR", "Sel", "Ha", "HI"),
        "springob": ("UGC/AGC", "OName", "Sabs", "e_Sabs", "SNR", "HRV", "Tel"),
        "kinematic_predictors": (
            "UGC",
            "Kmag",
            "e_Kmag",
            "Dist",
            "e_Dist",
            "Ak",
            "Kcor",
            "KMag",
            "e_KMag",
        ),
    }
    rows = {key: _parse_vizier_tsv(bodies[key], columns[key]) for key in columns}
    for key, expected in config["sources"]["expected_rows"].items():
        if len(rows[key]) != int(expected):
            raise GravityItem19Error(f"{key} source row count changed")
    file_keys = {
        "diskmass": "predictor_diskmass",
        "springob": "predictor_springob",
        "kinematic_predictors": "predictor_kinematic",
    }
    for source_key, path_key in file_keys.items():
        paths[path_key].write_bytes(bodies[source_key])
    objects, counts = _build_sample(
        rows["diskmass"], rows["springob"], rows["kinematic_predictors"], root, config
    )
    prior_hits = _prior_ugc_hits(root, [str(row["name"]) for row in objects])
    if prior_hits:
        raise GravityItem19Error(f"remaining UGC identities overlap prior responses: {prior_hits}")
    exploration = [row for row in objects if row["role"] == "exploration"]
    confirmation = [row for row in objects if row["role"] == "reserved_confirmation"]
    kernel_validation = _prepare_kernel_table(paths["kernel_table"], config)
    arrays = generate_candidates(config)
    injection_target = (
        (arrays["family"] == 2).astype(float) * 100.0
        - abs(np.log10(arrays["lambda1_kpc"]) - 1.0)
        - abs(arrays["alpha_total"] - 0.4)
        - abs(arrays["stellar_mass_to_light"] - 0.5)
        - abs(arrays["gas_mass_scale"] - 1.5)
    )
    injection_index = int(np.argmax(injection_target))
    sample = _content_hashed(
        {
            "schema_version": "invariant-gravity-item19-massive-carrier-sample-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selection_used_2025_response_values": False,
            "counts": {**counts, "response_values_read": 0},
            "prior_response_identity_hits": {},
            "fold_counts_exploration": dict(
                sorted(Counter(str(row["outer_fold"]) for row in exploration).items())
            ),
            "objects": [
                {
                    key: (_format_float(value) if isinstance(value, float) else value)
                    for key, value in row.items()
                }
                for row in objects
            ],
            "excluded_identity_disclosure": {
                "item18_all_roles_manifest": config["sources"]["item18_sample_manifest"],
                "item18_all_roles_count": 43,
                "2013_target_adjacent_values_seen": config["sources"]["predecessor_exposed_ugc"],
                "2025_response_rows_seen_in_search_snippet": config["sources"][
                    "search_snippet_exposed_ugc"
                ],
            },
            "claims": {"confirmation_opened": False},
        }
    )
    candidate_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item19-massive-carrier-candidates-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "candidate_digest_sha256": _candidate_digest(arrays),
            "counts": {
                "raw_parameter_cells": int(config["candidate_generator"]["raw_parameter_cells"]),
                "physics_admissible_cells": len(arrays["family"]),
                "exact_parameter_equivalence_classes": len(arrays["family"]),
                "mechanism_niches": len(config["candidate_generator"]["families"]),
                "post_response_cells": 0,
            },
            "family_counts": {
                str(config["candidate_generator"]["families"][index]): int(
                    np.sum(arrays["family"] == index)
                )
                for index in range(len(config["candidate_generator"]["families"]))
            },
            "maximum_admitted_solar_fractional_deviation": float(
                np.max(arrays["maximum_solar_fractional_deviation"])
            ),
            "minimum_short_distance_denominator": float(
                np.min(1.0 + arrays["sign"] * arrays["alpha_total"])
            ),
            "carrier_mass_bounds": _carrier_mass_bounds(arrays, config),
            "kernel_table": {
                "path": paths["kernel_table"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["kernel_table"]),
                "validation": kernel_validation,
            },
            "synthetic_injection_index": injection_index,
            "creativity_labels": config["candidate_generator"]["creativity_labels"],
            "known_equivalence": config["candidate_generator"]["known_equivalence"],
            "historical_novelty_claimed": False,
            "positive_spectral_failure_certificate": config["physics"][
                "positive_spectral_failure_certificate"
            ],
            "pre_response_filters": config["pre_response_filters"],
            "response_values_read": 0,
        }
    )
    files = []
    for source_key, path_key in file_keys.items():
        files.append(
            {
                "source": source_key,
                "path": paths[path_key].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths[path_key]),
                "rows": len(rows[source_key]),
                "selected_columns": list(columns[source_key]),
                "last_modified": headers[source_key].get("last-modified"),
            }
        )
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item19-predictors-1.0",
            "queries": query_map,
            "files": files,
            "response_columns_requested": [],
            "forbidden_columns_read": [],
            "HI_line_width_read": False,
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample)
    _write_json(paths["candidate_manifest"], candidate_manifest)
    if len(exploration) != int(config["sample"]["exploration_count"]) or len(confirmation) != int(
        config["sample"]["confirmation_count"]
    ):
        raise GravityItem19Error("sample role counts changed")
    return paths


def _load_prepared(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictor = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    for payload, label in (
        (predictor, "predictor manifest"),
        (sample, "sample manifest"),
        (candidates, "candidate manifest"),
    ):
        _verify_content_hash(payload, label)
    for item in predictor["files"]:
        if _sha256_file(root / str(item["path"])) != str(item["sha256"]):
            raise GravityItem19Error(f"predictor file changed: {item['path']}")
    if _sha256_file(paths["kernel_table"]) != str(candidates["kernel_table"]["sha256"]):
        raise GravityItem19Error("kernel table changed")
    arrays = generate_candidates(config)
    if _candidate_digest(arrays) != candidates["candidate_digest_sha256"]:
        raise GravityItem19Error("candidate digest changed")
    if int(sample["counts"]["response_values_read"]) != 0 or bool(
        sample["claims"]["confirmation_opened"]
    ):
        raise GravityItem19Error("sample freeze contains response access")
    return predictor, sample, candidates


def _response_url(config: Mapping[str, Any], ugc: int) -> str:
    query = urllib.parse.urlencode(
        [
            ("-source", "J/ApJS/276/59/sample"),
            ("-out", ",".join(config["sources"]["response_columns"])),
            ("-out.max", "unlimited"),
            ("UGC", str(int(ugc))),
        ]
    )
    return f"{config['sources']['response_query_base']}?{query}"


def fetch_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    _, sample, candidates = _load_prepared(root, config)
    paths = _source_paths(root, config)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    confirmation = {
        int(row["ugc"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    output: list[dict[str, str]] = []
    receipts: list[dict[str, Any]] = []
    columns = tuple(config["sources"]["response_columns"])
    for item in exploration:
        ugc = int(item["ugc"])
        if ugc in confirmation:
            raise GravityItem19Error("confirmation response requested")
        url = _response_url(config, ugc)
        body, headers = _download(url)
        rows = _parse_vizier_tsv(body, columns)
        if len(rows) != 1 or int(rows[0]["UGC"]) != ugc:
            raise GravityItem19Error(f"response identity mismatch for UGC {ugc}")
        output.append(dict(rows[0]))
        receipts.append(
            {
                "ugc": ugc,
                "query_sha256": _sha256_bytes(url.encode()),
                "response_sha256": _sha256_bytes(body),
                "last_modified": headers.get("last-modified"),
            }
        )
    output.sort(key=lambda row: int(row["UGC"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)
    paths["exploration_responses"].write_text(buffer.getvalue(), encoding="utf-8", newline="")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item19-response-source-1.0",
            "sample_freeze_commit": config["sample_freeze_commit"],
            "requested_objects": len(exploration),
            "returned_objects": len(output),
            "confirmation_objects_requested": 0,
            "confirmation_response_values_read": 0,
            "post_response_candidate_cells": int(candidates["counts"]["post_response_cells"]),
            "columns": list(columns),
            "excluded_columns": config["sources"]["excluded_response_columns"],
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
            "per_object_source_receipts": receipts,
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["exploration_responses"]


def _load_rows(
    root: Path, config: Mapping[str, Any], gas_scale_h_r: float | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    _, sample, _ = _load_prepared(root, config)
    manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(manifest, "response source manifest")
    if int(manifest["confirmation_response_values_read"]) != 0:
        raise GravityItem19Error("confirmation response opened")
    if _sha256_file(paths["exploration_responses"]) != str(manifest["response_file"]["sha256"]):
        raise GravityItem19Error("response file changed")
    with paths["exploration_responses"].open(encoding="utf-8", newline="") as handle:
        response = {int(row["UGC"]): row for row in csv.DictReader(handle, delimiter="\t")}
    q = config["quality"]
    scale_ratio = float(gas_scale_h_r or config["physics"]["primary_atomic_gas_scale_hR"])
    arcsec = float(config["physics"]["constants"]["arcsec_to_radian"])
    rows: list[dict[str, Any]] = []
    for predictor in sample["objects"]:
        if predictor["role"] != "exploration":
            continue
        ugc = int(predictor["ugc"])
        observed = response.get(ugc)
        if observed is None:
            continue
        values = {
            key: _as_float(observed.get(key, ""))
            for key in (
                "N",
                "sigMod",
                "inc",
                "e_inc",
                "Vrot",
                "e_Vrot",
                "hrot",
                "e_hrot",
                "Aphi",
                "Arc",
            )
        }
        required = (
            values["Vrot"],
            values["e_Vrot"],
            values["hrot"],
            values["e_hrot"],
            values["sigMod"],
        )
        quality = all(value is not None for value in required)
        if quality and bool(q["require_positive_finite_rotation_parameters"]):
            quality = (
                min(float(value) for value in required) >= 0.0
                and float(values["Vrot"]) > 0.0
                and float(values["hrot"]) > 0.0
            )
        quality = quality and float(predictor["springob_SNR"]) >= float(q["minimum_springob_SNR"])
        low_flag = str(observed.get("f_inc", "")).strip().lower()
        if bool(q["reject_low_inclination_flag"]):
            quality = quality and low_flag not in {"l", "low"}
        if values["Aphi"] is not None:
            quality = quality and float(values["Aphi"]) <= float(q["maximum_Aphi"])
        if values["Arc"] is not None:
            quality = quality and float(values["Arc"]) <= float(q["maximum_Arc"])
        if not quality:
            continue
        h_r_arcsec = float(predictor["hR_arcsec"])
        x = float(config["physics"]["evaluation_radius_hR"]) * h_r_arcsec / float(values["hrot"])
        v_obs = float(values["Vrot"]) * math.tanh(x)
        if v_obs <= 0.0:
            continue
        distance_kpc = float(predictor["distance_2025_Mpc"]) * 1000.0
        h_r_kpc = h_r_arcsec * arcsec * distance_kpc
        radius_kpc = float(config["physics"]["evaluation_radius_hR"]) * h_r_kpc
        luminosity = float(predictor["K_luminosity_Lsun"])
        hi_mass = float(predictor["HI_mass_Msun"])
        star_v2_unit = float(_disk_velocity_sq(luminosity, h_r_kpc, radius_kpc))
        gas_v2_unit = float(_disk_velocity_sq(1.4 * hi_mass, scale_ratio * h_r_kpc, radius_kpc))
        sech2 = 1.0 / math.cosh(x) ** 2 if x < 350.0 else 0.0
        hrot_fraction = (
            x * sech2 / max(math.tanh(x), 1.0e-12) * float(values["e_hrot"]) / float(values["hrot"])
        )
        sigma_log_v = math.sqrt(
            (float(values["e_Vrot"]) / float(values["Vrot"])) ** 2
            + hrot_fraction**2
            + (float(values["sigMod"]) / max(v_obs, 1.0e-12)) ** 2
        )
        rows.append(
            {
                "ugc": ugc,
                "name": predictor["name"],
                "fold": int(predictor["outer_fold"]),
                "mass_stratum": int(predictor["mass_stratum"]),
                "log_v_obs": math.log(v_obs),
                "sigma_log_v": sigma_log_v,
                "radius_kpc": radius_kpc,
                "star_v2_unit": star_v2_unit,
                "gas_v2_unit": gas_v2_unit,
                "K_luminosity_Lsun": luminosity,
                "HI_mass_Msun": hi_mass,
                "hR_kpc": h_r_kpc,
                "gas_scale_hR": scale_ratio,
                "mu0": float(predictor["mu0_R_mag_arcsec2"]),
                "B_minus_K": float(predictor["B_minus_K"]),
                "gas_fraction_proxy": 1.4 * hi_mass / (0.5 * luminosity + 1.4 * hi_mass),
            }
        )
    return sorted(rows, key=lambda row: row["name"]), manifest


def _load_kernel_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key], dtype=np.float64) for key in payload.files}


def _kernel_values(
    x_value: float,
    disk_scales: Any,
    lambdas: Any,
    table: Mapping[str, np.ndarray],
    xp: Any,
) -> Any:
    x_values = np.asarray(table["x_values"])
    index = int(np.argmin(abs(x_values - x_value)))
    if abs(float(x_values[index]) - x_value) > 1.0e-10:
        raise GravityItem19Error(f"kernel table lacks x={x_value}")
    log_a = xp.log10(disk_scales / lambdas)
    flat = xp.interp(
        log_a.reshape(-1),
        xp.asarray(table["log10_a"]),
        xp.asarray(table["ratios"][index]),
    )
    return flat.reshape(log_a.shape)


def _prediction_matrix(
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    table: Mapping[str, np.ndarray],
    xp: Any,
    batch_size: int,
) -> Any:
    count = len(arrays["family"])
    output = xp.empty((count, len(rows)), dtype=xp.float64)
    h_r = xp.asarray([row["hR_kpc"] for row in rows], dtype=xp.float64)[None, :]
    star_unit = xp.asarray([row["star_v2_unit"] for row in rows], dtype=xp.float64)[None, :]
    gas_unit = xp.asarray([row["gas_v2_unit"] for row in rows], dtype=xp.float64)[None, :]
    gas_scale_h_r = float(rows[0]["gas_scale_hR"])
    x_gas = float(2.2 / gas_scale_h_r)
    for begin in range(0, count, batch_size):
        end = min(begin + batch_size, count)
        sign = xp.asarray(arrays["sign"][begin:end, None])
        alpha1 = xp.asarray(arrays["alpha1"][begin:end, None])
        alpha2 = xp.asarray(arrays["alpha2"][begin:end, None])
        total = xp.asarray(arrays["alpha_total"][begin:end, None])
        lambda1 = xp.asarray(arrays["lambda1_kpc"][begin:end, None])
        lambda2 = xp.asarray(arrays["lambda2_kpc"][begin:end, None])
        stellar = xp.asarray(arrays["stellar_mass_to_light"][begin:end, None])
        gas = xp.asarray(arrays["gas_mass_scale"][begin:end, None])
        star = stellar * star_unit
        gas_term = gas * gas_unit
        ratio_s1 = _kernel_values(2.2, h_r, lambda1, table, xp)
        ratio_s2 = _kernel_values(2.2, h_r, lambda2, table, xp)
        ratio_g1 = _kernel_values(x_gas, gas_scale_h_r * h_r, lambda1, table, xp)
        ratio_g2 = _kernel_values(x_gas, gas_scale_h_r * h_r, lambda2, table, xp)
        massive1 = star * ratio_s1 + gas_term * ratio_g1
        massive2 = star * ratio_s2 + gas_term * ratio_g2
        v2 = (star + gas_term + sign * (alpha1 * massive1 + alpha2 * massive2)) / (
            1.0 + sign * total
        )
        if bool(_to_numpy(xp.any(v2 <= 0.0), xp)):
            raise GravityItem19Error("candidate produced nonpositive disk force")
        output[begin:end] = 0.5 * xp.log(v2)
    return output


def _gr_matrix(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], xp: Any
) -> tuple[Any, np.ndarray, np.ndarray]:
    generator = config["candidate_generator"]
    stellar = np.linspace(
        float(generator["stellar_mass_to_light_min"]),
        float(generator["stellar_mass_to_light_max"]),
        int(generator["stellar_mass_to_light_count"]),
    )
    gas = np.asarray(generator["gas_mass_scales"], dtype=np.float64)
    ss, gg = np.meshgrid(stellar, gas, indexing="ij")
    sf, gf = ss.reshape(-1), gg.reshape(-1)
    matrix = 0.5 * xp.log(
        xp.asarray(sf[:, None]) * xp.asarray([row["star_v2_unit"] for row in rows])[None, :]
        + xp.asarray(gf[:, None]) * xp.asarray([row["gas_v2_unit"] for row in rows])[None, :]
    )
    return matrix, sf, gf


def _oof_select(matrix: Any, y: Any, folds: np.ndarray, xp: Any) -> tuple[np.ndarray, list[int]]:
    prediction = xp.empty(len(folds), dtype=xp.float64)
    selected: list[int] = []
    for fold in sorted({int(value) for value in folds}):
        train = xp.asarray(folds != fold)
        test = xp.asarray(folds == fold)
        if not bool(_to_numpy(xp.any(test), xp)) or not bool(_to_numpy(xp.any(train), xp)):
            raise GravityItem19Error("empty train or held-out fold")
        loss = xp.mean((matrix[:, train] - y[train][None, :]) ** 2, axis=1)
        index = int(_to_numpy(xp.argmin(loss), xp))
        prediction[test] = matrix[index, test]
        selected.append(index)
    return _to_numpy(prediction, xp), selected


def _ridge_oof(rows: Sequence[Mapping[str, Any]], y: np.ndarray, alpha: float) -> np.ndarray:
    features = np.asarray(
        [
            [
                math.log(row["K_luminosity_Lsun"]),
                math.log(row["HI_mass_Msun"]),
                math.log(row["hR_kpc"]),
                row["mu0"],
                row["B_minus_K"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    folds = np.asarray([row["fold"] for row in rows])
    output = np.empty(len(rows))
    for fold in sorted({int(value) for value in folds}):
        train, test = folds != fold, folds == fold
        mean, std = features[train].mean(axis=0), features[train].std(axis=0)
        std[std == 0.0] = 1.0
        x_train = np.column_stack([np.ones(train.sum()), (features[train] - mean) / std])
        x_test = np.column_stack([np.ones(test.sum()), (features[test] - mean) / std])
        penalty = np.eye(x_train.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y[train])
        output[test] = x_test @ coefficient
    return output


def _btfr_oof(rows: Sequence[Mapping[str, Any]], y: np.ndarray) -> np.ndarray:
    x = 0.25 * np.log(
        np.asarray([0.5 * row["K_luminosity_Lsun"] + 1.4 * row["HI_mass_Msun"] for row in rows])
    )
    folds = np.asarray([row["fold"] for row in rows])
    output = np.empty(len(rows))
    for fold in sorted({int(value) for value in folds}):
        train, test = folds != fold, folds == fold
        output[test] = x[test] + float(np.mean(y[train] - x[train]))
    return output


def _slice_improvements(
    rows: Sequence[Mapping[str, Any]],
    candidate: np.ndarray,
    baseline: np.ndarray,
    y: np.ndarray,
    key: str,
) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    median = float(np.median(values))
    return {
        label: _improvement(_mse(y[mask], baseline[mask]), _mse(y[mask], candidate[mask]))
        for label, mask in (("low", values <= median), ("high", values > median))
    }


def _selected_cell(
    index: int, arrays: Mapping[str, np.ndarray], config: Mapping[str, Any], fold: int
) -> dict[str, Any]:
    family_index = int(arrays["family"][index])
    label = str(config["candidate_generator"]["families"][family_index])
    conversion = float(config["physics"]["constants"]["hbar_c_eV_m"]) / float(
        config["physics"]["constants"]["kpc_m"]
    )
    return {
        "fold": fold,
        "index": index,
        "family": label,
        "polarity": "attractive" if float(arrays["sign"][index]) > 0.0 else "repulsive",
        "creativity_label": config["candidate_generator"]["creativity_labels"][label],
        "alpha_total": float(arrays["alpha_total"][index]),
        "alpha1": float(arrays["alpha1"][index]),
        "alpha2": float(arrays["alpha2"][index]),
        "lambda1_kpc": float(arrays["lambda1_kpc"][index]),
        "lambda2_kpc": float(arrays["lambda2_kpc"][index]),
        "mass1_eV": conversion / float(arrays["lambda1_kpc"][index]),
        "mass2_eV": conversion / float(arrays["lambda2_kpc"][index]),
        "stellar_mass_to_light": float(arrays["stellar_mass_to_light"][index]),
        "gas_mass_scale": float(arrays["gas_mass_scale"][index]),
        "maximum_solar_fractional_deviation": float(
            arrays["maximum_solar_fractional_deviation"][index]
        ),
    }


def _synthetic_controls(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    table: Mapping[str, np.ndarray],
    injection_index: int,
    xp: Any,
) -> dict[str, Any]:
    rows = []
    for index in range(48):
        h_r = 0.4 * math.exp(index / 15.0)
        rows.append(
            {
                "fold": index % int(config["sample"]["outer_folds"]),
                "hR_kpc": h_r,
                "gas_scale_hR": float(config["physics"]["primary_atomic_gas_scale_hR"]),
                "star_v2_unit": 1200.0 + 95.0 * index + 170.0 * math.sin(index / 3.0),
                "gas_v2_unit": 500.0 + 28.0 * index + 90.0 * math.cos(index / 5.0),
            }
        )
    matrix = _prediction_matrix(
        arrays, rows, table, xp, int(config["evaluation"]["candidate_batch_size"])
    )
    gr, _, _ = _gr_matrix(config, rows, xp)
    folds = np.asarray([row["fold"] for row in rows])
    truth_gr = _to_numpy(gr[37], xp) + 1.0e-6 * np.sin(np.arange(len(rows)))
    candidate_gr, _ = _oof_select(matrix, xp.asarray(truth_gr), folds, xp)
    baseline_gr, _ = _oof_select(gr, xp.asarray(truth_gr), folds, xp)
    gr_improvement = _improvement(_mse(truth_gr, baseline_gr), _mse(truth_gr, candidate_gr))
    truth_injected = _to_numpy(matrix[injection_index], xp)
    candidate_injected, selected = _oof_select(matrix, xp.asarray(truth_injected), folds, xp)
    baseline_injected, _ = _oof_select(gr, xp.asarray(truth_injected), folds, xp)
    injected_improvement = _improvement(
        _mse(truth_injected, baseline_injected), _mse(truth_injected, candidate_injected)
    )
    target_polarity = float(arrays["sign"][injection_index])
    target_log_range = math.log10(float(arrays["lambda1_kpc"][injection_index]))
    polarity_recovered = all(float(arrays["sign"][index]) == target_polarity for index in selected)
    ranges_recovered = all(
        abs(math.log10(float(arrays["lambda1_kpc"][index])) - target_log_range) <= 0.75
        for index in selected
    )
    return {
        "known_GR": {"candidate_improvement_vs_GR": gr_improvement, "pass": gr_improvement <= 0.0},
        "massive_carrier_injection": {
            "injection_index": injection_index,
            "injection_family": config["candidate_generator"]["families"][
                int(arrays["family"][injection_index])
            ],
            "candidate_improvement_vs_GR": injected_improvement,
            "selected_indices": selected,
            "polarity_recovered": polarity_recovered,
            "primary_range_recovered_within_0p75dex": ranges_recovered,
            "pass": injected_improvement > 0.5 and polarity_recovered and ranges_recovered,
        },
    }


def _positive_spectral_certificate(
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    table: Mapping[str, np.ndarray],
    xp: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    attractive_indices = np.flatnonzero(arrays["sign"] > 0.0)
    subset_indices = attractive_indices[: min(4096, len(attractive_indices))]
    subset = {key: value[subset_indices] for key, value in arrays.items()}
    candidate = _prediction_matrix(
        subset, rows, table, xp, int(config["evaluation"]["candidate_batch_size"])
    )
    same_calibration = 0.5 * xp.log(
        xp.asarray(subset["stellar_mass_to_light"][:, None])
        * xp.asarray([row["star_v2_unit"] for row in rows])[None, :]
        + xp.asarray(subset["gas_mass_scale"][:, None])
        * xp.asarray([row["gas_v2_unit"] for row in rows])[None, :]
    )
    maximum_excess = float(_to_numpy(xp.max(candidate - same_calibration), xp))
    derivative_grid = np.logspace(-12.0, 12.0, 4097)
    derivative = -derivative_grid * np.exp(-derivative_grid)
    table_min = float(np.min(table["ratios"]))
    table_max = float(np.max(table["ratios"]))
    passed = (
        float(np.max(derivative)) <= 1.0e-15
        and table_min >= 0.0
        and table_max <= 1.0
        and maximum_excess <= 1.0e-12
    )
    return {
        "analytic_point_kernel_derivative": "df/du=-u*exp(-u)<=0",
        "numerical_derivative_grid_maximum": float(np.max(derivative)),
        "disk_kernel_ratio_minimum": table_min,
        "disk_kernel_ratio_maximum": table_max,
        "attractive_candidate_cells_checked": len(subset_indices),
        "maximum_log_velocity_excess_over_same_calibration_GR": maximum_excess,
        "scope": "arbitrary nonnegative point-source Yukawa spectral mixtures; and arbitrary nonnegative mixtures of the frozen exponential-disk kernels at the evaluated component radii",
        "pass": passed,
    }


def run_experiment(root: Path) -> Path:
    start = time.perf_counter()
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    _, sample, candidate_manifest = _load_prepared(root, config)
    rows, response_manifest = _load_rows(root, config)
    if not rows:
        raise GravityItem19Error("no valid exploration rows")
    quality_pass = len(rows) >= int(
        config["gravity_track_gates"]["minimum_complete_exploration_objects"]
    )
    folds = np.asarray([row["fold"] for row in rows], dtype=int)
    fold_quality_pass = len(set(folds)) == int(config["sample"]["outer_folds"])
    quality_pass = quality_pass and fold_quality_pass
    arrays = generate_candidates(config)
    paths = _source_paths(root, config)
    table = _load_kernel_table(paths["kernel_table"])
    xp, backend, device = _backend()
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    matrix_start = time.perf_counter()
    matrix = _prediction_matrix(arrays, rows, table, xp, batch_size)
    matrix_seconds = time.perf_counter() - matrix_start
    gr_matrix, gr_stellar, gr_gas = _gr_matrix(config, rows, xp)
    y_np = np.asarray([row["log_v_obs"] for row in rows], dtype=np.float64)
    y = xp.asarray(y_np)
    candidate_prediction, selected = _oof_select(matrix, y, folds, xp)
    gr_prediction, selected_gr = _oof_select(gr_matrix, y, folds, xp)
    fixed_index = int(np.argmin(abs(gr_stellar - 0.5) + abs(gr_gas - 1.0)))
    fixed_prediction = _to_numpy(gr_matrix[fixed_index], xp)
    btfr = _btfr_oof(rows, y_np)
    flexible = _ridge_oof(rows, y_np, float(config["evaluation"]["ridge_alpha"]))
    predictions = {
        "candidate": candidate_prediction,
        "fixed_GR": fixed_prediction,
        "calibrated_GR": gr_prediction,
        "baryonic_TF": btfr,
        "flexible_nuisance": flexible,
    }
    losses = {key: _mse(y_np, value) for key, value in predictions.items()}
    improvements = {
        key: _improvement(losses[key], losses["candidate"]) for key in losses if key != "candidate"
    }
    ordinary_names = ("calibrated_GR", "baryonic_TF", "flexible_nuisance")
    strongest_name = min(ordinary_names, key=lambda name: losses[name])
    strongest_prediction = predictions[strongest_name]
    strongest_improvement = _improvement(losses[strongest_name], losses["candidate"])
    selected_cells = [
        _selected_cell(index, arrays, config, fold) for fold, index in enumerate(selected)
    ]
    mass_slices_gr = _slice_improvements(
        rows, candidate_prediction, gr_prediction, y_np, "mass_stratum"
    )
    gas_slices_gr = _slice_improvements(
        rows, candidate_prediction, gr_prediction, y_np, "gas_fraction_proxy"
    )
    mass_slices_strongest = _slice_improvements(
        rows, candidate_prediction, strongest_prediction, y_np, "mass_stratum"
    )
    rng = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    residual = y_np - gr_prediction
    observed_gain = improvements["calibrated_GR"]
    null_gains = []
    null_start = time.perf_counter()
    for _ in range(int(config["evaluation"]["permutation_trials"])):
        permutation = rng.permutation(len(rows))
        y_null_np = gr_prediction + residual[permutation]
        candidate_null, _ = _oof_select(matrix, xp.asarray(y_null_np), folds, xp)
        gr_null, _ = _oof_select(gr_matrix, xp.asarray(y_null_np), folds, xp)
        null_gains.append(_improvement(_mse(y_null_np, gr_null), _mse(y_null_np, candidate_null)))
    null_seconds = time.perf_counter() - null_start
    p_value = (1.0 + sum(value >= observed_gain for value in null_gains)) / (1.0 + len(null_gains))
    robustness = {}
    for gas_scale in config["physics"]["gas_scale_robustness_hR"]:
        alt_rows, _ = _load_rows(root, config, float(gas_scale))
        if [row["name"] for row in alt_rows] != [row["name"] for row in rows]:
            raise GravityItem19Error("gas-scale robustness changed quality sample")
        alt_matrix = _prediction_matrix(arrays, alt_rows, table, xp, batch_size)
        alt_gr, _, _ = _gr_matrix(config, alt_rows, xp)
        alt_candidate = np.empty(len(rows))
        alt_baseline = np.empty(len(rows))
        for fold, (candidate_index, gr_index) in enumerate(zip(selected, selected_gr, strict=True)):
            test = folds == fold
            alt_candidate[test] = _to_numpy(alt_matrix[candidate_index, xp.asarray(test)], xp)
            alt_baseline[test] = _to_numpy(alt_gr[gr_index, xp.asarray(test)], xp)
        robustness[str(gas_scale)] = _improvement(
            _mse(y_np, alt_baseline), _mse(y_np, alt_candidate)
        )
        del alt_matrix
    controls = _synthetic_controls(
        config,
        arrays,
        table,
        int(candidate_manifest["synthetic_injection_index"]),
        xp,
    )
    certificate = _positive_spectral_certificate(arrays, rows, table, xp, config)
    polarity_count = max(Counter(cell["polarity"] for cell in selected_cells).values())
    primary_logs = np.asarray([math.log10(cell["lambda1_kpc"]) for cell in selected_cells])
    median_log = float(np.median(primary_logs))
    clustered = int(
        np.sum(
            np.abs(primary_logs - median_log)
            <= float(
                config["gravity_track_gates"]["maximum_primary_range_distance_from_median_dex"]
            )
        )
    )
    gravity_config = config["gravity_track_gates"]
    gravity_gates = {
        "minimum_complete_exploration_objects_and_all_folds": quality_pass,
        "confirmation_values_read": int(response_manifest["confirmation_response_values_read"])
        == 0,
        "post_response_candidate_cells": int(candidate_manifest["counts"]["post_response_cells"])
        == 0,
        "improvement_vs_fixed_GR": improvements["fixed_GR"]
        >= float(gravity_config["minimum_mse_improvement_vs_fixed_GR"]),
        "improvement_vs_calibrated_GR": improvements["calibrated_GR"]
        >= float(gravity_config["minimum_mse_improvement_vs_calibrated_GR"]),
        "improvement_vs_baryonic_TF": improvements["baryonic_TF"]
        >= float(gravity_config["minimum_mse_improvement_vs_baryonic_TF"]),
        "improvement_vs_flexible_nuisance": improvements["flexible_nuisance"]
        >= float(gravity_config["minimum_mse_improvement_vs_flexible_nuisance"]),
        "both_mass_halves_vs_calibrated_GR": min(mass_slices_gr.values())
        >= float(gravity_config["minimum_each_mass_half_improvement_vs_calibrated_GR"]),
        "both_gas_halves_vs_calibrated_GR": min(gas_slices_gr.values())
        >= float(gravity_config["minimum_each_gas_fraction_half_improvement_vs_calibrated_GR"]),
        "selection_aware_permutation": p_value
        <= float(gravity_config["maximum_selection_aware_permutation_p"]),
        "same_polarity_folds": polarity_count >= int(gravity_config["minimum_same_polarity_folds"]),
        "primary_range_clustered_folds": clustered
        >= int(gravity_config["minimum_range_clustered_folds"]),
        "gas_scale_robustness": min(robustness.values()) >= 0.0,
        "known_GR_control": bool(controls["known_GR"]["pass"]),
        "synthetic_massive_carrier_control": bool(controls["massive_carrier_injection"]["pass"]),
        "positive_spectral_failure_certificate": bool(certificate["pass"]),
    }
    publication_config = config["publication_track_gates"]
    publication_gates = {
        "minimum_complete_exploration_objects_and_all_folds": quality_pass,
        "improvement_vs_strongest_ordinary_baseline": strongest_improvement
        >= float(publication_config["minimum_improvement_vs_strongest_ordinary_baseline"]),
        "selection_aware_permutation": p_value
        <= float(publication_config["maximum_selection_aware_permutation_p"]),
        "same_polarity_folds": polarity_count
        >= int(publication_config["minimum_same_polarity_folds"]),
        "both_mass_halves_vs_strongest_ordinary_baseline": min(mass_slices_strongest.values())
        >= float(
            publication_config["minimum_each_mass_half_improvement_vs_strongest_ordinary_baseline"]
        ),
        "gas_scale_robustness": min(robustness.values()) >= 0.0,
        "fresh_confirmation_still_required": True,
    }
    gravity_decision = (
        "PASS_ITEM19_GRAVITY_EXPLORATION"
        if all(gravity_gates.values())
        else (
            "INCONCLUSIVE_ITEM19_QUALITY"
            if not quality_pass
            else "REJECT_ITEM19_MASSIVE_CARRIER_GRAVITY_EXPLORATION"
        )
    )
    publication_decision = (
        "RETAIN_ITEM19_PHENOMENON_FOR_INDEPENDENT_CONFIRMATION"
        if all(publication_gates.values())
        else (
            "INCONCLUSIVE_ITEM19_PUBLICATION_QUALITY"
            if not quality_pass
            else "NO_ITEM19_EMPIRICAL_PUBLICATION_LEAD"
        )
    )
    cpu_subset = {key: value[: min(1024, len(value))] for key, value in arrays.items()}
    cpu_matrix = _prediction_matrix(cpu_subset, rows, table, np, batch_size)
    gpu_subset = _to_numpy(matrix[: len(cpu_subset["family"])], xp)
    maximum_cpu_backend_difference = float(np.max(abs(cpu_matrix - gpu_subset)))
    compute = _content_hashed(
        {
            "schema_version": "invariant-gravity-item19-compute-1.0",
            "backend": backend,
            "device": device,
            "raw_parameter_cells": int(config["candidate_generator"]["raw_parameter_cells"]),
            "physics_admissible_cells": len(arrays["family"]),
            "candidate_observable_values": int(len(arrays["family"]) * len(rows)),
            "null_inclusive_training_residual_evaluations": int(
                (1 + len(null_gains))
                * int(config["sample"]["outer_folds"])
                * len(arrays["family"])
                * max(1, len(rows) - len(rows) // int(config["sample"]["outer_folds"]))
            ),
            "matrix_seconds": matrix_seconds,
            "null_screen_seconds": null_seconds,
            "elapsed_seconds": time.perf_counter() - start,
            "maximum_cpu_backend_log_prediction_difference": maximum_cpu_backend_difference,
            "paid_api_calls": 0,
            "api_spend_usd": 0.0,
        }
    )
    _write_json(paths["compute_manifest"], compute)
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item19-massive-carrier-result-1.0",
            "item": 19,
            "hypothesis": config["hypothesis"],
            "two_track_policy": {
                "universal_gravity": config["scope"]["universal_gravity_track"],
                "phenomenon_publication": config["scope"]["phenomenon_publication_track"],
            },
            "provenance_and_creativity_labels": config["candidate_generator"]["creativity_labels"],
            "known_equivalence": config["candidate_generator"]["known_equivalence"],
            "historical_novelty_claimed": False,
            "mathematical_definition": {
                "scalar_action": config["physics"]["scalar_action"],
                "vector_action": config["physics"]["vector_action"],
                "point_potential": config["physics"]["static_point_potential"],
                "disk_law": config["physics"]["disk_law"],
                "disk_transform": config["physics"]["disk_yukawa_transform"],
            },
            "dimensional_and_symmetry_checks": {
                "r_over_lambda_dimensionless": True,
                "carrier_mass_times_range_equals_hbar_over_c": True,
                "couplings_dimensionless": True,
                "static_parity_even_rotationally_invariant": True,
                "laboratory_and_solar_filters_applied_before_response": True,
                "full_covariant_stability_and_current_consistency_proved": False,
            },
            "positive_spectral_failure_certificate": certificate,
            "data_source_receipt": {
                "predictor_manifest": paths["predictor_source_manifest"]
                .relative_to(root)
                .as_posix(),
                "sample_manifest": paths["sample_manifest"].relative_to(root).as_posix(),
                "response_manifest": paths["response_source_manifest"].relative_to(root).as_posix(),
                "exploration_available": int(sample["counts"]["exploration"]),
                "exploration_quality_valid": len(rows),
                "quality_valid_by_fold": dict(
                    sorted(Counter(str(value) for value in folds).items())
                ),
                "reserved_confirmation": int(sample["counts"]["reserved_confirmation"]),
                "confirmation_opened": 0,
                "prefreeze_contamination_excluded": sample["excluded_identity_disclosure"],
            },
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "post_response_candidate_cells": 0,
            },
            "candidate_counts": candidate_manifest["counts"],
            "candidate_family_counts": candidate_manifest["family_counts"],
            "carrier_mass_bounds": candidate_manifest["carrier_mass_bounds"],
            "baselines": config["evaluation"],
            "losses": losses,
            "improvements": improvements,
            "strongest_ordinary_baseline": strongest_name,
            "improvement_vs_strongest_ordinary_baseline": strongest_improvement,
            "selected_cells_by_fold": selected_cells,
            "selected_GR_calibrations_by_fold": [
                {
                    "fold": fold,
                    "stellar_mass_to_light": float(gr_stellar[index]),
                    "gas_mass_scale": float(gr_gas[index]),
                }
                for fold, index in enumerate(selected_gr)
            ],
            "robustness": {
                "mass_halves_improvement_vs_calibrated_GR": mass_slices_gr,
                "gas_fraction_halves_improvement_vs_calibrated_GR": gas_slices_gr,
                "mass_halves_improvement_vs_strongest_ordinary_baseline": mass_slices_strongest,
                "gas_disk_scale_replays_improvement_vs_calibrated_GR": robustness,
            },
            "selection_aware_permutation": {
                "trials": len(null_gains),
                "p_value": p_value,
                "null_gain_quantiles": {
                    str(q): float(np.quantile(null_gains, q)) for q in (0.05, 0.5, 0.95)
                },
            },
            "controls": controls,
            "gravity_track_gates": gravity_gates,
            "gravity_track_gates_passed": sum(gravity_gates.values()),
            "gravity_track_gates_total": len(gravity_gates),
            "publication_track_gates": publication_gates,
            "publication_track_gates_passed": sum(publication_gates.values()),
            "publication_track_gates_total": len(publication_gates),
            "gravity_track_result": gravity_decision,
            "publication_track_result": publication_decision,
            "counterexamples_vs_calibrated_GR": [
                row["name"]
                for row, candidate_value, baseline_value, observed in zip(
                    rows, candidate_prediction, gr_prediction, y_np, strict=True
                )
                if (candidate_value - observed) ** 2 > (baseline_value - observed) ** 2
            ],
            "claim_limit": config["scope"]["claim_ceiling"],
            "compute_manifest": paths["compute_manifest"].relative_to(root).as_posix(),
            "compute": compute,
            "exact_next_action": "Advance the numbered roadmap to Item 20 massive phases without opening the four Item 19 confirmations. Independently retain any publication-track lead that passed its frozen gates; preserve the positive-spectral failure certificate and exact Item16 rewrite equivalence regardless of empirical outcome.",
        }
    )
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, result)
    if hasattr(xp, "get_default_memory_pool"):
        xp.get_default_memory_pool().free_all_blocks()
    return result_path


def validate_result(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    path = root / str(config["paths"]["result"])
    result = _read_json(path)
    _verify_content_hash(result, "Item 19 result")
    paths = _source_paths(root, config)
    for key in ("response_source_manifest", "compute_manifest"):
        payload = _read_json(paths[key])
        _verify_content_hash(payload, key)
    if int(result["data_source_receipt"]["confirmation_opened"]) != 0:
        raise GravityItem19Error("result opened confirmation data")
    if int(result["frozen_boundary"]["post_response_candidate_cells"]) != 0:
        raise GravityItem19Error("result contains post-response candidates")
    if bool(result["historical_novelty_claimed"]):
        raise GravityItem19Error("known Yukawa family was mislabeled historically novel")
    if not bool(result["positive_spectral_failure_certificate"]["pass"]):
        raise GravityItem19Error("positive-spectral certificate failed")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-predictors", "fetch-responses", "run", "check"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare-predictors":
        paths = prepare_predictors(root)
        print(paths["sample_manifest"])
    elif args.command == "fetch-responses":
        print(fetch_responses(root))
    elif args.command == "run":
        print(run_experiment(root))
    else:
        print(validate_result(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
