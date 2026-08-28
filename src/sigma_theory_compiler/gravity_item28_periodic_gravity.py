"""Frozen Item 28 equal-capacity periodic-gravity search on fresh GHASP data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import time
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _backend,
    _download,
    _read_tsv,
    _to_numpy,
    _write_tsv,
)
from sigma_theory_compiler.gravity_item24_temporal_lapse import (
    _angular_separation_arcsec,
    _hmac_rank,
)
from sigma_theory_compiler.gravity_item25_time_varying_g import (
    _canonical_bytes,
    _content_hashed,
    _fit_candidate_predictions,
    _improvement,
    _linear_predict,
    _mse,
    _read_json,
    _ridge_predict,
    _select_candidate,
    _sha256_bytes,
    _sha256_file,
    _verify_content_hash,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item28_periodic_gravity_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item28_periodic_gravity.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem28Error(RuntimeError):
    """Raised when an Item 28 freeze, leakage, or replay invariant is violated."""


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem28Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem28Error(f"{label} is not an ancestor of HEAD")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if (
        config.get("schema_version")
        != "invariant-gravity-item28-periodic-gravity-config-1.0"
        or int(config.get("item", -1)) != 28
    ):
        raise GravityItem28Error("unexpected Item 28 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem28Error("stable gravity goal changed")
    if int(config["candidate_generator"]["raw_candidate_cells"]) != 262144:
        raise GravityItem28Error("raw candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem28Error("post-response candidates entered Item 28")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem28Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem28Error("paid calls are outside Item 28")
    policy = config["discovery_policy"]
    if not bool(policy["equal_initial_viability"]):
        raise GravityItem28Error("equal-viability policy changed")
    if not bool(policy["age_or_history_is_not_privileged"]):
        raise GravityItem28Error("age or history was privileged")
    if not bool(policy["partial_results_are_not_pruned"]):
        raise GravityItem28Error("partial-result preservation changed")
    for relative, digest in config["dependency_sha256"].items():
        if _sha256_file(root / str(relative)) != str(digest):
            raise GravityItem28Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    value.pop("implementation_correction_commit", None)
    value.pop("implementation_correction_scope", None)
    value.pop("implementation_correction_history", None)
    value.pop("response_access_incident", None)
    return _sha256_bytes(_canonical_bytes(value))


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen) != _contract_digest(config):
        raise GravityItem28Error("scientific contract differs from frozen commit")
    module_commit = str(config.get("implementation_correction_commit", commit))
    _require_ancestor(root, module_commit, "implementation correction")
    module = _git(root, "show", f"{module_commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(module, bytes) or _sha256_bytes(module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem28Error("Item 28 module differs from scientific freeze")


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


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in ("predictors", "predictor_source_manifest", "sample_manifest", "candidate_manifest"):
        repo_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repo_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem28Error(f"{key} differs from sample freeze")


def _choice(random: np.random.Generator, values: Sequence[Any], count: int) -> np.ndarray:
    source = np.asarray(values)
    return source[random.integers(0, len(source), size=count)]


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    per = int(config["discovery_policy"]["equal_raw_capacity_per_mechanism"])
    if int(generator["raw_candidate_cells"]) != 4 * per:
        raise GravityItem28Error("mechanism capacity is not equal")
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    niches = np.repeat(np.arange(4, dtype=np.int8), per)
    arrays: dict[str, np.ndarray] = {
        "niche": niches,
        "amplitude": _choice(random, generator["amplitudes"], 4 * per).astype(float),
        "polarity": _choice(random, generator["polarities"], 4 * per).astype(float),
        "phase": _choice(random, generator["phases_rad"], 4 * per).astype(float),
        "harmonic": _choice(random, generator["second_harmonic_weights"], 4 * per).astype(float),
        "envelope": _choice(random, generator["envelope_disk_scales"], 4 * per).astype(float),
        "transition": _choice(random, generator["acceleration_transitions_m_s2"], 4 * per).astype(float),
        "accel_power": _choice(random, generator["acceleration_powers"], 4 * per).astype(float),
        "coupling": _choice(random, generator["phase_couplings"], 4 * per).astype(float),
        "wave": np.zeros(4 * per, dtype=float),
    }
    wave_sources = (
        generator["spatial_wavelengths_kpc"],
        generator["orbital_periods_myr"],
        generator["log_angular_frequencies"],
        generator["phase_radial_periods_disk_scale"],
    )
    for niche, values in enumerate(wave_sources):
        begin, end = niche * per, (niche + 1) * per
        arrays["wave"][begin:end] = _choice(random, values, per)
    return arrays


def _raw_candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _candidate_values(arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any) -> dict[str, Any]:
    return {key: xp.asarray(value[begin:end])[:, None] for key, value in arrays.items()}


def _admissible_candidates(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    physics = config["physics"]
    batch = int(config["evaluation"]["candidate_batch_size"])
    admitted: list[np.ndarray] = []
    local_max = 0.0
    bounded_failures = 0
    cycle_failures = 0
    local_failures = 0
    for begin in range(0, len(raw["niche"]), batch):
        end = min(begin + batch, len(raw["niche"]))
        niche = raw["niche"][begin:end]
        amp = raw["amplitude"][begin:end] * (1.0 + np.abs(raw["harmonic"][begin:end]))
        mu_low = np.exp(-amp)
        mu_high = np.exp(amp)
        bounded = (mu_low >= float(physics["minimum_mu_on_domain"])) & (
            mu_high <= float(physics["maximum_mu_on_domain"])
        )
        wave = raw["wave"][begin:end]
        span = np.zeros(len(niche), dtype=float)
        span[niche == 0] = 2.0 * math.pi * (30.0 - 0.2) / wave[niche == 0]
        span[niche == 1] = 2.0 * math.pi * (1000.0 - 10.0) / wave[niche == 1]
        span[niche == 2] = wave[niche == 2] * math.log(4.0 / 0.4)
        span[niche == 3] = 2.0 * math.pi * (4.0 - 0.4) / wave[niche == 3]
        cycles = span >= math.pi
        transition = raw["transition"][begin:end]
        power = raw["accel_power"][begin:end]
        window = 1.0 / (
            1.0
            + (float(physics["solar_acceleration_m_s2"]) / transition) ** power
        )
        local_response = np.expm1(amp * window)
        local = local_response <= float(physics["maximum_local_fractional_response"])
        keep = bounded & cycles & local
        bounded_failures += int(np.count_nonzero(~bounded))
        cycle_failures += int(np.count_nonzero(~cycles))
        local_failures += int(np.count_nonzero(~local))
        if np.any(keep):
            local_max = max(local_max, float(np.max(local_response[keep])))
            admitted.append(np.arange(begin, end, dtype=np.int64)[keep])
    indices = np.concatenate(admitted)
    arrays = {key: value[indices] for key, value in raw.items()}
    audit = {
        "raw_candidates": len(raw["niche"]),
        "raw_niche_counts": {str(n): int(np.count_nonzero(raw["niche"] == n)) for n in range(4)},
        "admissible_candidates": len(indices),
        "admissible_niche_counts": {str(n): int(np.count_nonzero(arrays["niche"] == n)) for n in range(4)},
        "raw_candidate_digest": _raw_candidate_digest(raw),
        "admissible_candidate_digest": _raw_candidate_digest(arrays),
        "bounded_failure_cells": bounded_failures,
        "cycle_span_failure_cells": cycle_failures,
        "local_limit_failure_cells": local_failures,
        "maximum_admitted_local_fractional_response": local_max,
        "minimum_admitted_mu": float(np.min(np.exp(-arrays["amplitude"] * (1.0 + arrays["harmonic"])))),
        "maximum_admitted_mu": float(np.max(np.exp(arrays["amplitude"] * (1.0 + arrays["harmonic"])))),
    }
    if any(int(audit["admissible_niche_counts"][str(n)]) == 0 for n in range(4)):
        raise GravityItem28Error("an equal-capacity niche has no admissible cells")
    return arrays, audit


def _suggest_injections(arrays: Mapping[str, np.ndarray]) -> list[int]:
    output: list[int] = []
    for niche in range(4):
        indices = np.where(
            (arrays["niche"] == niche)
            & (arrays["amplitude"] >= 0.1)
            & (arrays["amplitude"] <= 0.6)
            & (arrays["harmonic"] <= 0.25)
        )[0]
        if len(indices) == 0:
            raise GravityItem28Error(f"no target-blind injection for niche {niche}")
        output.append(int(indices[len(indices) // 2]))
    return output


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    arrays, audit = _admissible_candidates(config)
    injections = [int(value) for value in config["candidate_generator"]["synthetic_injection_admissible_indices"]]
    if injections != _suggest_injections(arrays):
        raise GravityItem28Error(f"frozen injections changed; expected {_suggest_injections(arrays)}")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item28-periodic-candidates-1.0",
            "generator": config["candidate_generator"],
            "audit": audit,
            "synthetic_injection_admissible_indices": injections,
            "responses_open_when_generated": False,
            "post_response_candidate_cells": 0,
        }
    )


def _normal_identity(value: str) -> str:
    text = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    match = re.fullmatch(r"(UGC|NGC|IC)0*(\d+)([A-Z]?)", text)
    if match:
        return f"{match.group(1)}{int(match.group(2))}{match.group(3)}"
    return text


def _vizier_rows(body: bytes) -> list[dict[str, str]]:
    lines = [
        line
        for line in body.decode("utf-8").splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
    output: list[dict[str, str]] = []
    for row in reader:
        cleaned = {str(key).strip(): str(value or "").strip() for key, value in row.items()}
        first = next(iter(cleaned.values()), "")
        if not first or first.startswith("-"):
            continue
        output.append(cleaned)
    return output


def _download_source(url: str) -> tuple[bytes, dict[str, Any]]:
    body, headers = _download(url)
    return body, {
        "url": url,
        "sha256": _sha256_bytes(body),
        "bytes": len(body),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
    }


def _predecessor_exclusions(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    names: set[str] = set()
    coordinates: list[tuple[float, float, str]] = []
    files = 0
    for path in sorted(root.glob(str(config["sources"]["predecessor_sample_glob"]))):
        if path.parent.name.startswith("item-28-"):
            continue
        try:
            manifest = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        files += 1
        for row in manifest.get("objects", []):
            if not isinstance(row, Mapping):
                continue
            for key in ("identity", "normalized_identity", "name", "other_name"):
                value = row.get(key)
                if value and not (key == "identity" and str(value).isdigit()):
                    names.add(_normal_identity(str(value)))
            for prefix, key in (("UGC", "ugc"), ("NGC", "ngc")):
                if row.get(key) not in (None, ""):
                    try:
                        names.add(_normal_identity(f"{prefix}{int(row[key])}"))
                    except (TypeError, ValueError):
                        pass
            for ra_key, dec_key in (
                ("ra_deg", "dec_deg"),
                ("ra", "dec"),
                ("catalog_ra_deg", "catalog_dec_deg"),
            ):
                if row.get(ra_key) is not None and row.get(dec_key) is not None:
                    try:
                        coordinates.append((float(row[ra_key]), float(row[dec_key]), path.parent.name))
                    except (TypeError, ValueError):
                        pass
    return {"names": names, "coordinates": coordinates, "files": files}


def _optional_float(row: Mapping[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bulge_fraction(rows: Sequence[Mapping[str, str]], disk: Mapping[str, str]) -> tuple[float, float, float]:
    from scipy.special import gamma

    mu0 = float(disk["mu0"])
    hi = float(disk["hi"])
    disk_flux = 2.0 * math.pi * 10.0 ** (-0.4 * mu0) * hi**2
    bulge_flux = 0.0
    bulge_re = 0.0
    bulge_n = 1.0
    for row in rows:
        if str(row.get("Class", "")).lower() != "bulge":
            continue
        mue = _optional_float(row, "mue")
        reff = _optional_float(row, "reff")
        n = _optional_float(row, "n")
        if mue is None or reff is None or n is None or reff <= 0.0 or n <= 0.0:
            continue
        bn = max(0.1, 2.0 * n - 1.0 / 3.0 + 0.009876 / n)
        flux = (
            2.0
            * math.pi
            * 10.0 ** (-0.4 * mue)
            * reff**2
            * n
            * math.exp(bn)
            * float(gamma(2.0 * n))
            / bn ** (2.0 * n)
        )
        if flux > bulge_flux:
            bulge_flux, bulge_re, bulge_n = flux, reff, n
    fraction = bulge_flux / (bulge_flux + disk_flux) if bulge_flux > 0.0 else 0.0
    return float(np.clip(fraction, 0.0, 0.95)), bulge_re, bulge_n


def _predictor_rows(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = config["sources"]
    requested = [
        ("photometric_sample", str(sources["photometric_sample_url"])),
        ("integrated_photometry", str(sources["integrated_photometry_url"])),
        ("decomposition", str(sources["decomposition_url"])),
    ]
    requested.extend((f"curve_names:{key}", str(url)) for key, url in sources["curve_name_urls"].items())
    bodies: dict[str, bytes] = {}
    receipts: dict[str, Any] = {}
    for label, url in requested:
        body, receipt = _download_source(url)
        bodies[label], receipts[label] = body, receipt
    sample_rows = _vizier_rows(bodies["photometric_sample"])
    integrated_rows = _vizier_rows(bodies["integrated_photometry"])
    decomposition_rows = _vizier_rows(bodies["decomposition"])
    if len(sample_rows) != int(sources["expected_photometric_sample_rows"]):
        raise GravityItem28Error(f"photometric sample row count changed: {len(sample_rows)}")
    if len(integrated_rows) != int(sources["expected_integrated_photometry_rows"]):
        raise GravityItem28Error(f"integrated photometry row count changed: {len(integrated_rows)}")
    if len(decomposition_rows) != int(sources["expected_decomposition_rows"]):
        raise GravityItem28Error(f"decomposition row count changed: {len(decomposition_rows)}")
    curve_maps: dict[str, dict[str, str]] = {}
    for catalog in sources["response_catalog_priority"]:
        label = f"curve_names:{catalog}"
        mapping: dict[str, str] = {}
        for row in _vizier_rows(bodies[label]):
            mapping.setdefault(_normal_identity(row["Name"]), row["Name"])
        curve_maps[str(catalog)] = mapping
    expected_curve_counts = (
        int(sources["expected_curve_identities_390"]),
        int(sources["expected_curve_identities_388"]),
    )
    actual_curve_counts = tuple(len(curve_maps[str(catalog)]) for catalog in sources["response_catalog_priority"])
    if actual_curve_counts != expected_curve_counts:
        raise GravityItem28Error(f"curve identity counts changed: {actual_curve_counts}")
    curve_union = set().union(*(mapping.keys() for mapping in curve_maps.values()))
    if len(curve_union) != int(sources["expected_curve_identity_union"]):
        raise GravityItem28Error("curve identity union changed")
    sample_by_name = {_normal_identity(row["Name"]): row for row in sample_rows}
    integrated_by_name: dict[str, dict[str, str]] = {}
    for row in integrated_rows:
        identity = _normal_identity(row["Name"])
        score = (
            row.get("Phot") == "1",
            row.get("Data") == "OHP",
            bool(row.get("Rmagtot")),
            bool(row.get("i")),
        )
        if identity not in integrated_by_name:
            integrated_by_name[identity] = row
        else:
            previous = integrated_by_name[identity]
            previous_score = (
                previous.get("Phot") == "1",
                previous.get("Data") == "OHP",
                bool(previous.get("Rmagtot")),
                bool(previous.get("i")),
            )
            if score > previous_score:
                integrated_by_name[identity] = row
    decomposition_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in decomposition_rows:
        decomposition_by_name[_normal_identity(row["Name"])].append(row)
    disk_by_name: dict[str, dict[str, str]] = {}
    for identity, rows in decomposition_by_name.items():
        disks = [row for row in rows if row.get("hi") and row.get("mu0")]
        if disks:
            disk_by_name[identity] = disks[0]
    intersection = curve_union & sample_by_name.keys() & integrated_by_name.keys() & disk_by_name.keys()
    if len(intersection) != int(sources["expected_predictor_curve_intersection"]):
        raise GravityItem28Error(f"predictor/curve intersection changed: {len(intersection)}")
    exclusions = _predecessor_exclusions(root, config)
    quality = config["predictor_quality"]
    output: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    name_overlaps = 0
    coordinate_overlaps = 0
    for identity in sorted(intersection):
        sample = sample_by_name[identity]
        phot = integrated_by_name[identity]
        disk = disk_by_name[identity]
        try:
            ra = float(sample["_RAJ2000"])
            dec = float(sample["_DEJ2000"])
            distance = float(sample["Dist"])
            seeing = float(sample["Seeing"])
            inclination = float(phot["i"])
            total_magnitude = float(phot["Rmagtot"])
            disk_scale_arcsec = float(disk["hi"])
            disk_mu0 = float(disk["mu0"])
        except (KeyError, TypeError, ValueError):
            failures["missing_required_predictor"] += 1
            continue
        reason: str | None = None
        if int(phot.get("Phot", "-1")) != int(quality["require_photometry_flag"]):
            reason = "photometry_flag"
        elif not float(quality["minimum_inclination_deg"]) <= inclination <= float(quality["maximum_inclination_deg"]):
            reason = "inclination"
        elif not float(quality["minimum_distance_mpc"]) <= distance <= float(quality["maximum_distance_mpc"]):
            reason = "distance"
        elif not float(quality["minimum_total_R_magnitude"]) <= total_magnitude <= float(quality["maximum_total_R_magnitude"]):
            reason = "total_magnitude"
        elif not float(quality["minimum_disk_scale_arcsec"]) <= disk_scale_arcsec <= float(quality["maximum_disk_scale_arcsec"]):
            reason = "disk_scale"
        elif disk_scale_arcsec < float(quality["minimum_disk_scale_to_seeing"]) * seeing:
            reason = "photometric_resolution"
        elif not float(quality["minimum_disk_central_surface_brightness"]) <= disk_mu0 <= float(quality["maximum_disk_central_surface_brightness"]):
            reason = "surface_brightness"
        elif identity in exclusions["names"]:
            name_overlaps += 1
            reason = "predecessor_name"
        else:
            nearest = min(
                (
                    _angular_separation_arcsec(ra, dec, prior_ra, prior_dec)
                    for prior_ra, prior_dec, _ in exclusions["coordinates"]
                ),
                default=float("inf"),
            )
            if nearest <= float(sources["predecessor_coordinate_veto_arcsec"]):
                coordinate_overlaps += 1
                reason = "predecessor_coordinate"
        if reason is not None:
            failures[reason] += 1
            continue
        absolute_R = total_magnitude - 5.0 * math.log10(distance * 1e6 / 10.0)
        luminosity = 10.0 ** (-0.4 * (absolute_R - float(quality["solar_absolute_R_magnitude"])))
        stellar_mass = luminosity * float(quality["fixed_stellar_mass_to_light_R"])
        disk_scale_kpc = disk_scale_arcsec * distance * 1000.0 / 206265.0
        bulge_fraction, bulge_re_arcsec, bulge_n = _bulge_fraction(decomposition_by_name[identity], disk)
        selected_catalog = next(
            str(catalog)
            for catalog in sources["response_catalog_priority"]
            if identity in curve_maps[str(catalog)]
        )
        decomposition = decomposition_by_name[identity]
        output.append(
            {
                "identity": identity,
                "name": sample["Name"].strip(),
                "ra_deg": ra,
                "dec_deg": dec,
                "distance_mpc": distance,
                "seeing_arcsec": seeing,
                "morphological_type": sample.get("Mtype", ""),
                "ttype": float(sample.get("TType") or 0.0),
                "inclination_deg": inclination,
                "position_angle_deg": float(phot.get("PA") or 0.0),
                "total_R_magnitude": total_magnitude,
                "absolute_R_magnitude": absolute_R,
                "log_stellar_mass_proxy": math.log10(stellar_mass),
                "disk_scale_arcsec": disk_scale_arcsec,
                "disk_scale_kpc": disk_scale_kpc,
                "disk_mu0_R": disk_mu0,
                "bulge_fraction_proxy": bulge_fraction,
                "bulge_re_kpc": bulge_re_arcsec * distance * 1000.0 / 206265.0,
                "bulge_sersic_n": bulge_n,
                "bar_component_count": sum(str(row.get("Class", "")).lower() == "bar" for row in decomposition),
                "disk_break_present": int(bool(disk.get("rB"))),
                "disk_break_radius_scale": (float(disk["rB"]) / disk_scale_arcsec if disk.get("rB") else 0.0),
                "response_catalog": selected_catalog,
                "response_query_name": curve_maps[selected_catalog][identity],
            }
        )
    audit = {
        "source_receipts": receipts,
        "photometric_sample_rows": len(sample_rows),
        "integrated_photometry_rows": len(integrated_rows),
        "decomposition_rows": len(decomposition_rows),
        "curve_identity_counts": {catalog: len(mapping) for catalog, mapping in curve_maps.items()},
        "curve_identity_union": len(curve_union),
        "predictor_curve_intersection": len(intersection),
        "predecessor_sample_manifests": exclusions["files"],
        "predecessor_names": len(exclusions["names"]),
        "predecessor_coordinates": len(exclusions["coordinates"]),
        "name_overlaps": name_overlaps,
        "coordinate_overlaps": coordinate_overlaps,
        "quality_failures": dict(sorted(failures.items())),
        "safe_predictor_eligible": len(output),
        "response_columns_read": [],
        "curve_value_rows_read": 0,
        "published_halo_or_mass_model_columns_read": [],
    }
    return sorted(output, key=lambda row: str(row["identity"])), audit


def _build_sample(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    sample = config["sample"]
    ordered = sorted(rows, key=lambda row: (float(row["log_stellar_mass_proxy"]), str(row["identity"])))
    groups = np.array_split(np.asarray(ordered, dtype=object), int(sample["mass_strata"]))
    objects: list[dict[str, Any]] = []
    for stratum, values in enumerate(groups):
        group = [dict(value) for value in values.tolist()]
        ranked = sorted(
            group,
            key=lambda row: _hmac_rank(str(sample["role_key"]), f"ghasp:{row['identity']}"),
        )
        confirmations = {str(row["identity"]) for row in ranked[: int(sample["confirmation_per_stratum"])]}
        exploration = sorted(
            [row for row in ranked if str(row["identity"]) not in confirmations],
            key=lambda row: _hmac_rank(str(sample["fold_key"]), f"ghasp:{row['identity']}"),
        )
        folds = {
            str(row["identity"]): int((index + stratum) % int(sample["outer_folds"]))
            for index, row in enumerate(exploration)
        }
        for row in group:
            identity = str(row["identity"])
            role = "confirmation" if identity in confirmations else "exploration"
            objects.append(
                {
                    "identity": identity,
                    "name": row["name"],
                    "role": role,
                    "mass_stratum": stratum,
                    "outer_fold": None if role == "confirmation" else folds[identity],
                    "ra_deg": float(row["ra_deg"]),
                    "dec_deg": float(row["dec_deg"]),
                    "response_catalog": row["response_catalog"],
                    "response_query_name": row["response_query_name"],
                    "role_rank_sha256": _hmac_rank(str(sample["role_key"]), f"ghasp:{identity}"),
                }
            )
    roles = Counter(str(row["role"]) for row in objects)
    folds = Counter(int(row["outer_fold"]) for row in objects if row["role"] == "exploration")
    if len(objects) != int(sample["expected_safe_predictor_eligible"]):
        raise GravityItem28Error("safe predictor count changed")
    if roles["exploration"] != int(sample["expected_exploration"]) or roles["confirmation"] != int(sample["expected_confirmation"]):
        raise GravityItem28Error("sample role counts changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item28-periodic-sample-1.0",
            "selection_rule": sample["rule"],
            "response_columns_read": [],
            "confirmation_response_values_read": 0,
            "objects": sorted(objects, key=lambda row: str(row["identity"])),
            "role_counts": dict(sorted(roles.items())),
            "fold_counts": {str(key): value for key, value in sorted(folds.items())},
        }
    )


def audit_predictors(root: Path) -> dict[str, Any]:
    config = load_config(root)
    rows, audit = _predictor_rows(root, config)
    return {"eligible": len(rows), "audit": audit, "suggested_injections": _suggest_injections(_admissible_candidates(config)[0])}


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictors"].parent.mkdir(parents=True, exist_ok=True)
    rows, audit = _predictor_rows(root, config)
    if len(rows) != int(config["sample"]["expected_safe_predictor_eligible"]):
        raise GravityItem28Error(f"safe predictor count changed: {len(rows)}")
    _write_tsv(paths["predictors"], rows, list(rows[0]))
    _write_json(
        paths["predictor_source_manifest"],
        _content_hashed(
            {
                "schema_version": "invariant-gravity-item28-periodic-predictors-1.0",
                "audit": audit,
                "predictor_file": {
                    "path": paths["predictors"].relative_to(root).as_posix(),
                    "sha256": _sha256_file(paths["predictors"]),
                    "rows": len(rows),
                },
            }
        ),
    )
    _write_json(paths["sample_manifest"], _build_sample(rows, config))
    _write_json(paths["candidate_manifest"], _candidate_manifest(config))
    return paths


def _interpolate_side(points: Sequence[tuple[float, float]], grid: float, maximum_gap: float) -> float:
    values = sorted(points)
    for x, velocity in values:
        if abs(x - grid) <= 1e-10:
            return velocity
    lower = [(x, velocity) for x, velocity in values if x < grid]
    upper = [(x, velocity) for x, velocity in values if x > grid]
    if not lower or not upper:
        return float("nan")
    x0, y0 = lower[-1]
    x1, y1 = upper[0]
    if x1 - x0 > maximum_gap:
        return float("nan")
    return y0 + (y1 - y0) * (grid - x0) / (x1 - x0)


def _curve_summary(body: bytes, predictor: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extraction = config["response_extraction"]
    rows = _vizier_rows(body)
    good: list[tuple[float, float, str]] = []
    rejected: Counter[str] = Counter()
    scale = float(predictor["disk_scale_kpc"])
    for row in rows:
        try:
            radius = float(row["r"])
            velocity = abs(float(row["Vrot"]))
            error = float(row["e_Vrot"])
            bins = int(row["Nbins"])
            side = str(row["Side"]).lower()
        except (KeyError, TypeError, ValueError):
            rejected["parse"] += 1
            continue
        x = radius / scale
        reason: str | None = None
        if side not in ("a", "r"):
            reason = "side"
        elif not float(extraction["minimum_radius_disk_scale"]) <= x <= float(extraction["maximum_radius_disk_scale"]):
            reason = "radius"
        elif not float(extraction["minimum_rotation_km_s"]) <= velocity <= float(extraction["maximum_rotation_km_s"]):
            reason = "velocity"
        elif error > float(extraction["maximum_rotation_error_km_s"]) or error / velocity > float(extraction["maximum_fractional_rotation_error"]):
            reason = "uncertainty"
        elif bins < int(extraction["minimum_velocity_bins"]):
            reason = "velocity_bins"
        if reason is not None:
            rejected[reason] += 1
        else:
            good.append((x, velocity, side))
    side_points = {
        side: [(x, velocity) for x, velocity, value_side in good if value_side == side]
        for side in ("a", "r")
    }
    failure: str | None = None
    if len(good) < int(extraction["minimum_raw_points"]):
        failure = "minimum_raw_points"
    elif any(len(side_points[side]) < int(extraction["minimum_points_each_side"]) for side in ("a", "r")):
        failure = "minimum_points_each_side"
    records: list[dict[str, Any]] = []
    if failure is None:
        for radius_label, grid in enumerate(extraction["radial_grid_disk_scale"]):
            approaching = _interpolate_side(side_points["a"], float(grid), float(extraction["maximum_interpolation_gap_disk_scale"]))
            receding = _interpolate_side(side_points["r"], float(grid), float(extraction["maximum_interpolation_gap_disk_scale"]))
            if math.isfinite(approaching) and math.isfinite(receding):
                records.append(
                    {
                        "radius_label": radius_label,
                        "radius_disk_scale": float(grid),
                        "radius_kpc": float(grid) * scale,
                        "primary_velocity_km_s": 0.5 * (approaching + receding),
                        "approaching_velocity_km_s": approaching,
                        "receding_velocity_km_s": receding,
                        "side_fractional_difference": abs(approaching - receding) / (0.5 * (approaching + receding)),
                    }
                )
        if len(records) < int(extraction["minimum_primary_grid_points"]):
            failure = "minimum_primary_grid_points"
            records = []
    return records, {
        "source_rows": len(rows),
        "good_raw_points": len(good),
        "approaching_raw_points": len(side_points["a"]),
        "receding_raw_points": len(side_points["r"]),
        "primary_grid_points": len(records),
        "rejected": dict(sorted(rejected.items())),
        "failure": failure,
    }


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    predictors = {str(row["identity"]): row for row in _read_tsv(paths["predictors"])}
    sample = _read_json(paths["sample_manifest"])
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    if len(exploration) != int(config["sample"]["expected_exploration"]):
        raise GravityItem28Error("exploration boundary changed")
    if any(row["role"] == "confirmation" for row in exploration):
        raise GravityItem28Error("confirmation entered response acquisition")

    def acquire(row: Mapping[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
        encoded = urllib.parse.quote(str(row["response_query_name"]), safe="")
        url = str(config["sources"]["response_query_template"]).format(
            catalog=row["response_catalog"], encoded_name=encoded
        )
        body, receipt = _download_source(url)
        return dict(row), body, receipt

    with ThreadPoolExecutor(max_workers=8) as pool:
        acquired = list(pool.map(acquire, exploration))
    records: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    valid_galaxies = 0
    for sample_row, body, receipt in acquired:
        identity = str(sample_row["identity"])
        predictor = predictors[identity]
        summary, audit = _curve_summary(body, predictor, config)
        receipts.append({"identity": identity, **receipt, "audit": audit})
        if audit["failure"] is not None:
            failures[str(audit["failure"])] += 1
            continue
        valid_galaxies += 1
        for value in summary:
            records.append(
                {
                    "identity": identity,
                    "name": predictor["name"],
                    "fold": int(sample_row["outer_fold"]),
                    "mass_stratum": int(sample_row["mass_stratum"]),
                    **value,
                }
            )
    if not records:
        raise GravityItem28Error("no exploration curve points passed the frozen extraction")
    _write_tsv(paths["exploration_responses"], records, list(records[0]))
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item28-periodic-responses-1.0",
            "exploration_queries": len(exploration),
            "confirmation_queries": 0,
            "confirmation_values_read": 0,
            "response_columns_read": ["Name", "r", "e_r", "r2", "e_r2", "Vrot", "e_Vrot", "Nbins", "Side"],
            "valid_galaxies": valid_galaxies,
            "valid_curve_points": len(records),
            "quality_failures": dict(sorted(failures.items())),
            "source_receipts": sorted(receipts, key=lambda row: str(row["identity"])),
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
                "rows": len(records),
            },
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["exploration_responses"]


def _row_physics(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[float, float, float]:
    from scipy.special import gammainc, iv, kv

    radius = float(row["radius_kpc"])
    scale = float(row["disk_scale_kpc"])
    mass = 10.0 ** float(row["log_stellar_mass_proxy"])
    bulge_fraction = float(row["bulge_fraction_proxy"])
    disk_mass = mass * (1.0 - bulge_fraction)
    bulge_mass = mass * bulge_fraction
    y = max(radius / (2.0 * scale), 1e-5)
    bessel = float(iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y))
    disk_v2 = max(0.0, 2.0 * float(config["physics"]["G_kpc_km2_s2_Msun"]) * disk_mass / scale * y**2 * bessel)
    bulge_v2 = 0.0
    reff = float(row["bulge_re_kpc"])
    n = float(row["bulge_sersic_n"])
    if bulge_mass > 0.0 and reff > 0.0 and n > 0.0:
        bn = max(0.1, 2.0 * n - 1.0 / 3.0 + 0.009876 / n)
        enclosed = float(gammainc(2.0 * n, bn * (radius / reff) ** (1.0 / n)))
        bulge_v2 = float(config["physics"]["G_kpc_km2_s2_Msun"]) * bulge_mass * enclosed / radius
    velocity = math.sqrt(max(disk_v2 + bulge_v2, 1.0))
    acceleration = velocity**2 * 1e6 / float(config["physics"]["meters_per_kpc"]) / radius
    orbital_myr = (
        2.0
        * math.pi
        * radius
        * float(config["physics"]["meters_per_kpc"])
        / 1000.0
        / velocity
        / float(config["physics"]["seconds_per_myr"])
    )
    return velocity, acceleration, orbital_myr


def _phase_matrix(
    config: Mapping[str, Any],
    values: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    radius = xp.asarray([float(row["radius_kpc"]) for row in rows])[None, :]
    x = xp.asarray([float(row["radius_disk_scale"]) for row in rows])[None, :]
    physics = [_row_physics(row, config) for row in rows]
    acceleration = xp.asarray([value[1] for value in physics])[None, :]
    orbital = xp.asarray([value[2] for value in physics])[None, :]
    niche = values["niche"]
    wave = values["wave"]
    phase = xp.zeros((len(niche), len(rows)), dtype=xp.float64)
    phase = xp.where(niche == 0, 2.0 * math.pi * radius / wave + values["phase"], phase)
    phase = xp.where(niche == 1, 2.0 * math.pi * orbital / wave + values["phase"], phase)
    phase = xp.where(niche == 2, wave * xp.log(x) + values["phase"], phase)
    coupled = (
        2.0 * math.pi * x / wave
        + values["coupling"] * xp.log(acceleration / float(config["physics"]["reference_acceleration_m_s2"]))
        + values["phase"]
    )
    return xp.where(niche == 3, coupled, phase)


def _build_term_matrix(config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    physics = [_row_physics(row, config) for row in rows]
    acceleration = np.asarray([value[1] for value in physics])[None, :]
    x = np.asarray([float(row["radius_disk_scale"]) for row in rows])[None, :]
    pieces: list[np.ndarray] = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        values = _candidate_values(arrays, begin, end, np)
        phase = _phase_matrix(config, values, rows, np)
        window = 1.0 / (1.0 + (acceleration / values["transition"]) ** values["accel_power"])
        envelope = np.exp(-x / values["envelope"])
        oscillation = np.sin(phase) + values["harmonic"] * np.sin(2.0 * phase)
        log_mu = values["polarity"] * values["amplitude"] * window * envelope * oscillation
        pieces.append(0.5 * log_mu / math.log(10.0))
    return np.concatenate(pieces, axis=0)


def _base_design(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> np.ndarray:
    velocity = np.asarray([_row_physics(row, config)[0] for row in rows])
    return np.column_stack([np.ones(len(rows)), np.log10(velocity)])


def _flex_design(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> np.ndarray:
    physics = [_row_physics(row, config) for row in rows]
    logv = np.log10([value[0] for value in physics])
    logg = np.log10([value[1] for value in physics]) + 10.0
    logx = np.log10([float(row["radius_disk_scale"]) for row in rows])
    mass = np.asarray([float(row["log_stellar_mass_proxy"]) for row in rows]) - 10.0
    mu0 = np.asarray([float(row["disk_mu0_R"]) for row in rows]) - 21.0
    inc = np.asarray([float(row["inclination_deg"]) for row in rows]) / 60.0
    ttype = np.asarray([float(row["ttype"]) for row in rows]) / 10.0
    bulge = np.asarray([float(row["bulge_fraction_proxy"]) for row in rows])
    bar = np.asarray([float(row["bar_component_count"]) for row in rows])
    broken = np.asarray([float(row["disk_break_present"]) for row in rows])
    resolution = np.asarray([float(row["seeing_arcsec"]) / float(row["disk_scale_arcsec"]) for row in rows])
    side_difference = np.asarray([float(row["side_fractional_difference"]) for row in rows])
    return np.column_stack(
        [
            logv,
            logv**2,
            logg,
            logg**2,
            logx,
            logx**2,
            logx**3,
            mass,
            mu0,
            inc,
            ttype,
            bulge,
            bar,
            broken,
            resolution,
            side_difference,
            logx * mass,
            logx * mu0,
            logg * bulge,
            logx * resolution,
        ]
    )


def _target(rows: Sequence[Mapping[str, Any]], label: str) -> np.ndarray:
    key = {
        "primary": "primary_velocity_km_s",
        "approaching": "approaching_velocity_km_s",
        "receding": "receding_velocity_km_s",
    }[label]
    return np.log10([float(row[key]) for row in rows])


def _oof_search(
    xp: Any,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    folds: np.ndarray,
    term_matrix: np.ndarray,
) -> dict[str, Any]:
    base = _base_design(rows, config)
    flexible = _flex_design(rows, config)
    output = {key: np.full(len(rows), np.nan) for key in ("candidate", "base", "flexible")}
    selected: list[int] = []
    evaluations = 0
    for outer in sorted({int(value) for value in folds}):
        train = np.where(folds != outer)[0]
        test = np.where(folds == outer)[0]
        index, count = _select_candidate(xp, config, base, target, folds, outer, term_matrix)
        selected.append(index)
        evaluations += count
        terms = xp.asarray(term_matrix[index : index + 1])
        prediction = _fit_candidate_predictions(xp, base, target, terms, train, test)
        output["candidate"][test] = _to_numpy(prediction[0], xp)
        output["base"][test] = _linear_predict(base, target, train, test)
        output["flexible"][test] = _ridge_predict(
            flexible, target, train, test, float(config["evaluation"]["ridge_alpha"])
        )
    return {**output, "selected": selected, "residual_evaluations": evaluations}


def _fixed_oof(
    xp: Any,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    folds: np.ndarray,
    term_matrix: np.ndarray,
    selected_by_fold: Mapping[int, int],
) -> dict[str, np.ndarray]:
    base = _base_design(rows, config)
    flexible = _flex_design(rows, config)
    output = {key: np.full(len(rows), np.nan) for key in ("candidate", "base", "flexible")}
    for outer in sorted({int(value) for value in folds}):
        train = np.where(folds != outer)[0]
        test = np.where(folds == outer)[0]
        index = int(selected_by_fold[outer])
        terms = xp.asarray(term_matrix[index : index + 1])
        output["candidate"][test] = _to_numpy(
            _fit_candidate_predictions(xp, base, target, terms, train, test)[0], xp
        )
        output["base"][test] = _linear_predict(base, target, train, test)
        output["flexible"][test] = _ridge_predict(
            flexible, target, train, test, float(config["evaluation"]["ridge_alpha"])
        )
    return output


def _candidate_record(config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    niche = int(arrays["niche"][index])
    return {
        "index": index,
        "niche": niche,
        "niche_id": config["candidate_generator"]["niches"][niche]["id"],
        "creativity_label": config["candidate_generator"]["niches"][niche]["creativity_label"],
        "amplitude": float(arrays["amplitude"][index]),
        "polarity": float(arrays["polarity"][index]),
        "wave": float(arrays["wave"][index]),
        "phase_rad": float(arrays["phase"][index]),
        "phase_coupling": float(arrays["coupling"][index]),
        "second_harmonic_weight": float(arrays["harmonic"][index]),
        "envelope_disk_scales": float(arrays["envelope"][index]),
        "acceleration_transition_m_s2": float(arrays["transition"][index]),
        "acceleration_power": float(arrays["accel_power"][index]),
    }


def _load_joined_rows(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictors = {str(row["identity"]): row for row in _read_tsv(paths["predictors"])}
    responses = _read_tsv(paths["exploration_responses"])
    sample = _read_json(paths["sample_manifest"])
    roles = {str(row["identity"]): row for row in sample["objects"]}
    rows: list[dict[str, Any]] = []
    for response in responses:
        identity = str(response["identity"])
        role = roles[identity]
        if role["role"] != "exploration":
            raise GravityItem28Error("confirmation response entered evaluation")
        rows.append(
            {
                **predictors[identity],
                **response,
                "fold": int(role["outer_fold"]),
                "mass_stratum": int(role["mass_stratum"]),
            }
        )
    manifest = _read_json(paths["response_source_manifest"])
    valid_identities = {str(row["identity"]) for row in rows}
    quality = {
        "frozen_exploration_galaxies": int(config["sample"]["expected_exploration"]),
        "valid_galaxies": len(valid_identities),
        "valid_curve_points": len(rows),
        "minimum_valid_exploration_galaxies": int(config["sample"]["minimum_valid_exploration_galaxies"]),
        "minimum_valid_curve_points": int(config["sample"]["minimum_valid_curve_points"]),
        "formal_quality_pass": (
            len(valid_identities) >= int(config["sample"]["minimum_valid_exploration_galaxies"])
            and len(rows) >= int(config["sample"]["minimum_valid_curve_points"])
        ),
        "fold_galaxy_counts": dict(sorted(Counter(int(roles[value]["outer_fold"]) for value in valid_identities).items())),
        "response_manifest_valid_galaxies": int(manifest["valid_galaxies"]),
    }
    return sorted(rows, key=lambda row: (str(row["identity"]), int(row["radius_label"]))), quality


def _slice_metrics(target: np.ndarray, observed: Mapping[str, Any], slices: Mapping[str, np.ndarray]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, indices in slices.items():
        candidate = _mse(target, observed["candidate"], indices)
        base = _mse(target, observed["base"], indices)
        flexible = _mse(target, observed["flexible"], indices)
        output[label] = {
            "points": len(indices),
            "candidate_mse": candidate,
            "baryonic_mse": base,
            "flexible_nuisance_mse": flexible,
            "improvement_vs_baryonic": _improvement(base, candidate),
            "improvement_vs_flexible_nuisance": _improvement(flexible, candidate),
        }
    return output


def _common_rule(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    niches = Counter(int(record["niche"]) for record in records)
    dominant, same_niche = niches.most_common(1)[0]
    relevant = [record for record in records if int(record["niche"]) == dominant]
    ratio = float(config["gates"]["maximum_common_wave_ratio"])
    wave_count = max(
        sum(max(float(a["wave"]), float(b["wave"])) / min(float(a["wave"]), float(b["wave"])) <= ratio for b in relevant)
        for a in relevant
    )
    maximum_arc = float(config["gates"]["maximum_common_phase_arc_rad"])
    phase_count = max(
        sum(
            abs(math.atan2(math.sin(float(a["phase_rad"]) - float(b["phase_rad"])), math.cos(float(a["phase_rad"]) - float(b["phase_rad"])))) <= maximum_arc
            for b in relevant
        )
        for a in relevant
    )
    return {
        "dominant_niche": dominant,
        "dominant_niche_id": config["candidate_generator"]["niches"][dominant]["id"],
        "same_niche_folds": same_niche,
        "common_wave_folds": wave_count,
        "common_phase_folds": phase_count,
        "wave_ratio_threshold": ratio,
        "phase_arc_threshold_rad": maximum_arc,
        "pass": (
            same_niche >= int(config["gates"]["minimum_same_niche_folds"])
            and wave_count >= int(config["gates"]["minimum_common_wave_folds"])
            and phase_count >= int(config["gates"]["minimum_common_phase_folds"])
        ),
    }


def _evaluate(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], formal_quality_pass: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    xp, backend, device = _backend()
    started = time.perf_counter()
    arrays, candidate_audit = _admissible_candidates(config)
    terms = _build_term_matrix(config, arrays, rows)
    folds = np.asarray([int(row["fold"]) for row in rows])
    target = _target(rows, "primary")
    observed = _oof_search(xp, config, rows, target, folds, terms)
    residual_evaluations = int(observed["residual_evaluations"])
    candidate_mse = _mse(target, observed["candidate"])
    base_mse = _mse(target, observed["base"])
    flexible_mse = _mse(target, observed["flexible"])
    observed_improvement = _improvement(base_mse, candidate_mse)

    base_full = _base_design(rows, config)
    base_prediction = base_full @ np.linalg.lstsq(base_full, target, rcond=None)[0]
    residual = target - base_prediction
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    strata = np.asarray([int(row["mass_stratum"]) for row in rows])
    radii = np.asarray([int(row["radius_label"]) for row in rows])
    null_improvements: list[float] = []
    for trial in range(int(config["evaluation"]["permutation_trials"])):
        permuted = residual.copy()
        for stratum in sorted(set(strata.tolist())):
            for radius in sorted(set(radii.tolist())):
                indices = np.where((strata == stratum) & (radii == radius))[0]
                if len(indices) > 1:
                    permuted[indices] = random.permutation(permuted[indices])
        null_target = base_prediction + permuted
        null = _oof_search(xp, config, rows, null_target, folds, terms)
        null_improvements.append(
            _improvement(_mse(null_target, null["base"]), _mse(null_target, null["candidate"]))
        )
        residual_evaluations += int(null["residual_evaluations"])
        if (trial + 1) % 10 == 0:
            print(f"Item 28 selection-aware nulls {trial + 1}/{config['evaluation']['permutation_trials']}", flush=True)
    permutation_p = (1.0 + sum(value >= observed_improvement for value in null_improvements)) / (len(null_improvements) + 1.0)

    injections = [int(value) for value in config["candidate_generator"]["synthetic_injection_admissible_indices"]]
    synthetic: list[dict[str, Any]] = []
    for niche, injection in enumerate(injections):
        if int(arrays["niche"][injection]) != niche:
            raise GravityItem28Error("injection niche changed")
        injected_target = base_prediction + terms[injection]
        replay = _oof_search(xp, config, rows, injected_target, folds, terms)
        selected_niches = [int(arrays["niche"][index]) for index in replay["selected"]]
        recovered = int(np.count_nonzero(np.asarray(selected_niches) == niche))
        synthetic.append(
            {
                "injected_niche": niche,
                "injected_index": injection,
                "selected_niches": selected_niches,
                "selected_niche_folds": recovered,
                "pass": recovered >= int(config["gates"]["minimum_same_niche_folds"]),
            }
        )
        residual_evaluations += int(replay["residual_evaluations"])
    zero = _oof_search(xp, config, rows, base_prediction, folds, terms)
    zero_improvement = _improvement(_mse(base_prediction, zero["base"]), _mse(base_prediction, zero["candidate"]))
    residual_evaluations += int(zero["residual_evaluations"])
    zero_pass = zero_improvement <= float(config["gates"]["known_zero_signal_control_maximum_material_improvement"])

    selected_by_fold = {
        outer: index
        for outer, index in zip(sorted({int(value) for value in folds}), observed["selected"], strict=True)
    }
    side_metrics: dict[str, Any] = {}
    for label in ("approaching", "receding"):
        side_target = _target(rows, label)
        prediction = _fixed_oof(xp, config, rows, side_target, folds, terms, selected_by_fold)
        side_candidate = _mse(side_target, prediction["candidate"])
        side_base = _mse(side_target, prediction["base"])
        side_flexible = _mse(side_target, prediction["flexible"])
        side_metrics[label] = {
            "points": len(rows),
            "candidate_mse": side_candidate,
            "baryonic_mse": side_base,
            "flexible_nuisance_mse": side_flexible,
            "improvement_vs_baryonic": _improvement(side_base, side_candidate),
            "improvement_vs_flexible_nuisance": _improvement(side_flexible, side_candidate),
        }

    mass = np.asarray([float(row["log_stellar_mass_proxy"]) for row in rows])
    brightness = np.asarray([float(row["disk_mu0_R"]) for row in rows])
    radius = np.asarray([float(row["radius_disk_scale"]) for row in rows])
    slices = {
        "inner_radius": np.where(radius <= np.median(radius))[0],
        "outer_radius": np.where(radius > np.median(radius))[0],
        "low_stellar_mass": np.where(mass <= np.median(mass))[0],
        "high_stellar_mass": np.where(mass > np.median(mass))[0],
        "high_surface_brightness": np.where(brightness <= np.median(brightness))[0],
        "low_surface_brightness": np.where(brightness > np.median(brightness))[0],
    }
    slice_metrics = _slice_metrics(target, observed, slices)
    selected_records = [_candidate_record(config, arrays, int(index)) for index in observed["selected"]]
    common = _common_rule(selected_records, config)
    by_identity: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_identity[str(row["identity"])].append(index)
    counterexamples = sum(
        _mse(target, observed["candidate"], np.asarray(indices))
        > _mse(target, observed["flexible"], np.asarray(indices))
        for indices in by_identity.values()
    )
    universal = all(
        [
            formal_quality_pass,
            observed_improvement >= float(config["gates"]["minimum_improvement_vs_baryonic"]),
            _improvement(flexible_mse, candidate_mse) >= float(config["gates"]["minimum_improvement_vs_flexible_nuisance"]),
            all(side_metrics[label]["improvement_vs_baryonic"] >= float(config["gates"]["minimum_each_side_improvement_vs_baryonic"]) for label in side_metrics),
            all(slice_metrics[label]["improvement_vs_baryonic"] >= float(config["gates"]["minimum_each_radial_half_improvement_vs_baryonic"]) for label in ("inner_radius", "outer_radius")),
            all(slice_metrics[label]["improvement_vs_baryonic"] >= float(config["gates"]["minimum_each_mass_half_improvement_vs_baryonic"]) for label in ("low_stellar_mass", "high_stellar_mass")),
            all(slice_metrics[label]["improvement_vs_baryonic"] >= float(config["gates"]["minimum_each_surface_brightness_half_improvement_vs_baryonic"]) for label in ("high_surface_brightness", "low_surface_brightness")),
            permutation_p <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            common["pass"],
            all(value["pass"] for value in synthetic),
            zero_pass,
        ]
    )
    phenomenon = all(
        [
            formal_quality_pass,
            _improvement(flexible_mse, candidate_mse) >= float(config["gates"]["phenomenon_minimum_improvement_vs_flexible"]),
            permutation_p <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            common["pass"],
            all(slice_metrics[label]["improvement_vs_flexible_nuisance"] >= 0.0 for label in ("inner_radius", "outer_radius")),
        ]
    )
    strongest_slice = max(slice_metrics, key=lambda label: slice_metrics[label]["improvement_vs_flexible_nuisance"])
    partial = all(
        [
            formal_quality_pass,
            permutation_p <= float(config["gates"]["partial_lead_maximum_selection_aware_p"]),
            common["same_niche_folds"] >= 3,
            slice_metrics[strongest_slice]["improvement_vs_flexible_nuisance"]
            >= float(config["gates"]["partial_lead_minimum_slice_improvement_vs_flexible"]),
        ]
    )
    cpu_terms = terms[np.asarray(observed["selected"])]
    gpu_terms = _to_numpy(xp.asarray(cpu_terms), xp)
    cpu_gpu_max = float(np.max(np.abs(cpu_terms - gpu_terms)))
    scientific = {
        "valid_galaxies": len(by_identity),
        "valid_curve_points": len(rows),
        "formal_quality_pass": formal_quality_pass,
        "candidate_audit": candidate_audit,
        "metrics": {
            "candidate_mse": candidate_mse,
            "baryonic_mse": base_mse,
            "flexible_nuisance_mse": flexible_mse,
            "improvement_vs_baryonic": observed_improvement,
            "improvement_vs_flexible_nuisance": _improvement(flexible_mse, candidate_mse),
            "selection_aware_permutation_p": permutation_p,
            "null_improvement_minimum": float(np.min(null_improvements)),
            "null_improvement_median": float(np.median(null_improvements)),
            "null_improvement_maximum": float(np.max(null_improvements)),
            "galaxy_counterexamples_vs_flexible": int(counterexamples),
        },
        "two_sided_replays": side_metrics,
        "slice_metrics": slice_metrics,
        "selected_folds": selected_records,
        "common_wavelength_phase_rule": common,
        "controls": {
            "synthetic_niche_recovery": synthetic,
            "synthetic_all_pass": all(value["pass"] for value in synthetic),
            "zero_signal_control_improvement": zero_improvement,
            "zero_signal_control_pass": zero_pass,
            "cpu_gpu_max_absolute_difference": cpu_gpu_max,
            "cpu_gpu_pass": cpu_gpu_max <= 1e-12,
        },
        "strongest_preregistered_slice": strongest_slice,
        "scoped_partial_replication_lead": partial,
        "universal_gravity_track_pass": universal,
        "phenomenon_publication_track_pass": phenomenon,
        "paper_claim_allowed": False,
        "formal_status": (
            "INCONCLUSIVE_QUALITY"
            if not formal_quality_pass
            else "PASS_EXPLORATION_BOTH_TRACKS"
            if universal and phenomenon
            else "PASS_EXPLORATION_UNIVERSAL_ONLY"
            if universal
            else "PASS_EXPLORATION_PHENOMENON_LEAD"
            if phenomenon
            else "SCOPED_PARTIAL_REPLICATION_LEAD"
            if partial
            else "SCOPED_REJECT_BOTH_TRACKS"
        ),
    }
    compute = {
        "schema_version": "invariant-gravity-item28-periodic-compute-1.0",
        "backend": backend,
        "device": device,
        "raw_candidates": int(config["candidate_generator"]["raw_candidate_cells"]),
        "admissible_candidates": len(arrays["niche"]),
        "training_residual_evaluations": residual_evaluations,
        "permutation_trials": len(null_improvements),
        "synthetic_full_searches": 4,
        "zero_signal_control_full_searches": 1,
        "wall_seconds": time.perf_counter() - started,
        "paid_model_calls": 0,
        "paid_api_spend_usd": 0.0,
    }
    return scientific, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    response_manifest: Mapping[str, Any],
    quality: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    valid_identities = sorted({str(row["identity"]) for row in rows})
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item28-periodic-result-1.0",
            "item": 28,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "discovery_policy": config["discovery_policy"],
            "theory_and_equivalence_audit": config["theory"],
            "observable_lineage": config["sources"]["observable_lineage"],
            "frozen_boundary": {
                "stable_goal_sha256": config["stable_goal_sha256"],
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "implementation_correction_commit": config.get("implementation_correction_commit"),
                "implementation_correction_scope": config.get("implementation_correction_scope"),
                "implementation_correction_history": config.get("implementation_correction_history", []),
                "response_access_incident": config.get("response_access_incident"),
                "confirmation_response_values_read": int(response_manifest["confirmation_values_read"]),
                "post_response_formula_generation": False,
            },
            "sample": {
                "valid_exploration_identities": valid_identities,
                "confirmation_identities_remain_sealed": int(config["sample"]["expected_confirmation"]),
                "quality_audit": dict(quality),
            },
            "baselines": {
                "baryonic": config["evaluation"]["baseline_baryonic"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
            },
            "scientific_result": dict(scientific),
            "compute_and_api_cost": dict(compute),
            "counterexamples_and_limitations": config["theory"]["claim_limits"],
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "predictor_manifest_path": paths["predictor_source_manifest"].relative_to(root).as_posix(),
                "response_manifest_path": paths["response_source_manifest"].relative_to(root).as_posix(),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
            },
            "exact_next_action": "Preserve every periodic family and scoped result under the equal-viability two-track policy; require unchanged independent replication before any paper claim; advance to Item 29 dimensional leakage without privileging or pruning age, periodicity, or any other distinct mechanism.",
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    for key in ("predictors", "predictor_source_manifest", "sample_manifest", "candidate_manifest", "exploration_responses", "response_source_manifest"):
        if not paths[key].exists():
            raise GravityItem28Error(f"missing frozen artifact: {key}")
    rows, quality = _load_joined_rows(root, config)
    scientific, compute_raw = _evaluate(config, rows, bool(quality["formal_quality_pass"]))
    compute = _content_hashed(compute_raw)
    _write_json(paths["compute_manifest"], compute)
    response_manifest = _read_json(paths["response_source_manifest"])
    result = _build_receipt(root, config, rows, response_manifest, quality, scientific, compute)
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
    response = _read_json(paths["response_source_manifest"])
    if int(response["confirmation_values_read"]) != 0:
        raise GravityItem28Error("confirmation boundary was opened")
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem28Error("result contains confirmation response values")
    if bool(result["scientific_result"]["paper_claim_allowed"]):
        raise GravityItem28Error("exploration result made a paper claim")
    if int(result["compute_and_api_cost"]["paid_model_calls"]) != 0:
        raise GravityItem28Error("paid calls entered Item 28")
    return result_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-predictors")
    sub.add_parser("prepare-predictors")
    sub.add_parser("acquire-responses")
    sub.add_parser("run")
    sub.add_parser("validate")
    sub.add_parser("show-candidates")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path.cwd()
    if args.command == "audit-predictors":
        print(json.dumps(audit_predictors(root), indent=2, sort_keys=True))
    elif args.command == "prepare-predictors":
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
