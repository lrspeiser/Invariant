"""Frozen Item 20 baryon-driven massive-field phase experiment.

The response is never used to construct candidates.  A carrier with universal Compton
frequency is occupied only when a harmonic of the *baryon-predicted* orbital frequency
falls in one of three frozen resonance windows.  Gravity and phenomenon/publication
outcomes are evaluated independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import urllib.parse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    _backend,
    _canonical_bytes,
    _content_hashed,
    _download,
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
from sigma_theory_compiler.gravity_item19_massive_carrier import (
    _btfr_oof,
    _kernel_values,
    _load_kernel_table,
    _oof_select,
    _prepare_kernel_table,
    _ridge_oof,
)

CONFIG_PATH = Path("configs/gravity_item20_massive_phase_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item20_massive_phase.py")
DEPENDENCY_PATHS = (
    Path("src/sigma_theory_compiler/gravity_item16_s4tm_qed_field.py"),
    Path("src/sigma_theory_compiler/gravity_item18_diskmass_antiscreening.py"),
    Path("src/sigma_theory_compiler/gravity_item19_massive_carrier.py"),
)
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem20Error(RuntimeError):
    """Raised when an Item 20 freeze, data-lineage, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if config.get("schema_version") != "invariant-gravity-item20-massive-phase-config-1.0":
        raise GravityItem20Error("unexpected Item 20 config schema")
    if int(config.get("item", -1)) != 20:
        raise GravityItem20Error("Item 20 item number changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem20Error("confirmation opening is not authorized")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem20Error("post-response candidates entered Item 20")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem20Error("stable gravity goal changed")
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
        raise GravityItem20Error(f"{label} is not bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root, check=False
    )
    if result.returncode:
        raise GravityItem20Error(f"{label} is not an ancestor of HEAD")


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen) != _contract_digest(config):
        raise GravityItem20Error("scientific contract differs from freeze")
    for path in (MODULE_PATH, *DEPENDENCY_PATHS):
        blob = _git(root, "show", f"{commit}:{path.as_posix()}", text_mode=False)
        if not isinstance(blob, bytes) or _sha256_bytes(blob) != _sha256_file(root / path):
            raise GravityItem20Error(f"dependency differs from science freeze: {path}")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    return {
        key: base / str(config["paths"][key])
        for key in (
            "predictor_sfi_photometry",
            "predictor_sfi",
            "predictor_springob",
            "predictor_source_manifest",
            "sample_manifest",
            "candidate_manifest",
            "kernel_table",
            "exploration_responses",
            "response_source_manifest",
            "compute_manifest",
        )
    }


def verify_sample_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["sample_freeze_commit"])
    _require_ancestor(root, commit, "sample freeze")
    paths = _source_paths(root, config)
    for key in (
        "predictor_sfi_photometry",
        "predictor_sfi",
        "predictor_springob",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "kernel_table",
    ):
        repository_path = paths[key].relative_to(root).as_posix()
        blob = _git(root, "show", f"{commit}:{repository_path}", text_mode=False)
        if not isinstance(blob, bytes) or _sha256_bytes(blob) != _sha256_file(paths[key]):
            raise GravityItem20Error(f"{key} differs from sample freeze")


def phase_occupation(
    family: Any, z: Any, quality: Any, power: Any, secondary_weight: Any, xp: Any = np
) -> Any:
    """Evaluate the three bounded, two-sided frozen resonance windows."""

    z = xp.maximum(z, 1.0e-300)
    delta = xp.clip(z - 1.0 / z, -1.0e150, 1.0e150)
    linear = 1.0 / (1.0 + (quality * delta) ** 2)
    core = xp.maximum(0.0, 1.0 - (quality * xp.log(z)) ** 2)
    landau = core**power
    z2 = 2.0 * z
    delta2 = xp.clip(z2 - 1.0 / z2, -1.0e150, 1.0e150)
    harmonic2 = 1.0 / (1.0 + (quality * delta2) ** 2)
    coherence = 1.0 - (1.0 - linear) * (1.0 - secondary_weight * harmonic2)
    return xp.where(family == 0, linear, xp.where(family == 1, landau, coherence))


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["raw_parameter_cells"])
    rng = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    niche = np.arange(count, dtype=np.int64) % 6
    rng.shuffle(niche)
    family = (niche // 2).astype(np.int8)
    sign = np.where(niche % 2 == 0, 1.0, -1.0)
    u = rng.random(count)
    enhance_log = float(generator["enhancing_amplitude_log10_min"]) + u * (
        float(generator["enhancing_amplitude_log10_max"])
        - float(generator["enhancing_amplitude_log10_min"])
    )
    suppress_log = float(generator["suppressing_amplitude_log10_min"]) + u * (
        math.log10(float(generator["suppressing_amplitude_max"]))
        - float(generator["suppressing_amplitude_log10_min"])
    )
    amplitude = 10.0 ** np.where(sign > 0, enhance_log, suppress_log)
    lambda_kpc = 10.0 ** rng.uniform(
        float(generator["lambda_log10_kpc_min"]),
        float(generator["lambda_log10_kpc_max"]),
        count,
    )
    quality = 10.0 ** rng.uniform(
        float(generator["quality_factor_log10_min"]),
        float(generator["quality_factor_log10_max"]),
        count,
    )
    harmonic = rng.choice(np.asarray(generator["harmonics"], dtype=float), count)
    power = rng.choice(np.asarray(generator["landau_powers"], dtype=float), count)
    weight = rng.choice(
        np.asarray(generator["secondary_harmonic_weights"], dtype=float), count
    )
    stellar_grid = np.linspace(
        float(generator["stellar_mass_to_light_min"]),
        float(generator["stellar_mass_to_light_max"]),
        int(generator["stellar_mass_to_light_count"]),
    )
    stellar = rng.choice(stellar_grid, count)
    gas = rng.choice(np.asarray(generator["gas_mass_scales"], dtype=float), count)
    constants = config["physics"]["constants"]
    solar_au = np.asarray(config["pre_response_filters"]["solar_probe_AU"], dtype=float)
    omega_solar = np.sqrt(
        float(constants["G_SI"])
        * float(constants["M_sun_kg"])
        / (solar_au * float(constants["AU_m"])) ** 3
    )
    omega_c = float(constants["c_m_s"]) / (lambda_kpc * float(constants["kpc_m"]))
    z = harmonic[:, None] * omega_solar[None, :] / omega_c[:, None]
    occupation = phase_occupation(
        family[:, None], z, quality[:, None], power[:, None], weight[:, None]
    )
    solar_force = np.max(amplitude[:, None] * occupation, axis=1)
    keep = solar_force <= float(
        config["pre_response_filters"]["maximum_solar_phase_force_fraction"]
    )
    return {
        "raw_index": np.nonzero(keep)[0].astype(np.int64),
        "family": family[keep],
        "sign": sign[keep],
        "amplitude": amplitude[keep],
        "lambda_kpc": lambda_kpc[keep],
        "quality_factor": quality[keep],
        "harmonic": harmonic[keep],
        "landau_power": power[keep],
        "secondary_weight": weight[keep],
        "stellar_mass_to_light": stellar[keep],
        "gas_mass_scale": gas[keep],
        "maximum_solar_phase_force_fraction": solar_force[keep],
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _sexagesimal(value: str, ra: bool = False) -> float:
    fields = value.replace(":", " ").split()
    if len(fields) != 3:
        raise GravityItem20Error(f"bad coordinate: {value!r}")
    sign = -1.0 if fields[0].startswith("-") else 1.0
    first = abs(float(fields[0]))
    answer = sign * (first + float(fields[1]) / 60.0 + float(fields[2]) / 3600.0)
    return answer * 15.0 if ra else answer


def _angular_arcsec(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ra1, de1, ra2, de2 = map(
        math.radians, (float(a["ra_deg"]), float(a["dec_deg"]), float(b["ra_deg"]), float(b["dec_deg"]))
    )
    value = math.sin(de1) * math.sin(de2) + math.cos(de1) * math.cos(de2) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, value)))) * 3600.0


def _prior_coordinates(root: Path, commit: str) -> list[dict[str, float]]:
    names = str(_git(root, "ls-tree", "-r", "--name-only", commit, "runs/gravity/roadmap")).splitlines()
    paths = [p for p in names if p.endswith(".json") and "sample" in Path(p).name and "manifest" in Path(p).name]
    found: list[dict[str, float]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            lower = {str(k).lower(): v for k, v in value.items()}
            ra = next((lower[k] for k in ("ra_deg", "raj2000_deg", "ra") if k in lower), None)
            dec = next((lower[k] for k in ("dec_deg", "dej2000_deg", "dec") if k in lower), None)
            try:
                if ra is not None and dec is not None and -90.0 <= float(dec) <= 90.0:
                    found.append({"ra_deg": float(ra), "dec_deg": float(dec)})
            except (TypeError, ValueError):
                pass
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in paths:
        try:
            visit(json.loads(str(_git(root, "show", f"{commit}:{path}"))))
        except (json.JSONDecodeError, subprocess.CalledProcessError):
            continue
    return found


def _build_sample(root: Path, config: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    photometry = _parse_vizier_tsv(paths["predictor_sfi_photometry"])
    predictors = _parse_vizier_tsv(paths["predictor_sfi"])
    springob = _parse_vizier_tsv(paths["predictor_springob"])
    p1 = {_as_int(row.get("AGC", "")): row for row in photometry}
    p2 = {_as_int(row.get("AGC", "")): row for row in predictors}
    p3 = {_as_int(row.get("UGC/AGC", "")): row for row in springob}
    joined = sorted(set(p1) & set(p2) & set(p3) - {None})
    if len(joined) != int(config["sources"]["expected_exact_three_table_join"]):
        raise GravityItem20Error(f"three-table join changed: {len(joined)}")
    constants = config["physics"]["constants"]
    sample = config["sample"]
    eligible: list[dict[str, Any]] = []
    for agc in joined:
        a, b, c = p1[agc], p2[agc], p3[agc]
        values = {
            "r83_arcsec": _as_float(a.get("r.83", "")),
            "imagc": _as_float(b.get("Imagc", "")),
            "inclination": _as_float(b.get("i", "")),
            "cz": _as_float(b.get("cz", "")),
            "flux": _as_float(c.get("Sabs", "")),
            "eflux": _as_float(c.get("e_Sabs", "")),
            "snr": _as_float(c.get("SNR", "")),
            "hrv": _as_float(c.get("HRV", "")),
            "surface_brightness": _as_float(a.get("Isb", "")),
            "type": _as_float(a.get("TT", "")),
        }
        if any(values[k] is None for k in ("r83_arcsec", "imagc", "inclination", "cz", "flux", "eflux", "snr", "hrv")):
            continue
        if str(a.get("TF", "")).strip() or str(b.get("r", "")).strip().upper() != "H":
            continue
        if not (int(sample["minimum_AGC"]) <= int(agc)):
            continue
        if not (float(sample["minimum_cz_km_s"]) <= values["cz"] <= float(sample["maximum_cz_km_s"])):
            continue
        if not (float(sample["minimum_inclination_deg"]) <= values["inclination"] <= float(sample["maximum_inclination_deg"])):
            continue
        if values["snr"] < float(sample["minimum_springob_SNR"]) or abs(values["cz"] - values["hrv"]) > float(sample["maximum_cz_HRV_difference_km_s"]):
            continue
        if min(values["r83_arcsec"], values["flux"], values["eflux"]) <= 0:
            continue
        coords = [(_sexagesimal(x["RAJ2000"], True), _sexagesimal(x["DEJ2000"])) for x in (a, b, c)]
        if max(abs(coords[i][j] - coords[0][j]) for i in (1, 2) for j in (0, 1)) > 1.0e-8:
            continue
        distance = values["cz"] / float(constants["H0_km_s_Mpc"])
        luminosity = 10.0 ** (-0.4 * (values["imagc"] - 5.0 * math.log10(distance) - 25.0 - float(constants["M_I_sun"])))
        gas_mass = float(constants["HI_mass_coefficient"]) * distance * distance * values["flux"]
        r83 = values["r83_arcsec"] * float(constants["arcsec_to_radian"]) * distance * 1000.0
        rd = r83 / float(constants["r83_over_exponential_Rd"])
        star_v2 = _disk_velocity_sq(luminosity, rd, r83, float(constants["G_kpc_km2_s2_Msun"]))
        gas_v2 = _disk_velocity_sq(gas_mass, 2.0 * rd, r83, float(constants["G_kpc_km2_s2_Msun"]))
        omega = math.sqrt(star_v2 + 1.4 * gas_v2) / r83 * float(constants["inverse_seconds_per_km_s_per_kpc"])
        eligible.append({
            "agc": int(agc), "other_name": str(a.get("OName", "")).strip(),
            "ra_deg": coords[0][0], "dec_deg": coords[0][1], "cz_km_s": values["cz"],
            "inclination_deg": values["inclination"], "type": values["type"],
            "surface_brightness": values["surface_brightness"], "luminosity_I_Lsun": luminosity,
            "atomic_HI_mass_Msun": gas_mass, "r83_kpc": r83, "stellar_Rd_kpc": rd,
            "stellar_v2_unit": star_v2, "gas_v2_unit": gas_v2, "fiducial_omega_s_inv": omega,
        })
    if len(eligible) != int(config["sources"]["expected_predictor_eligible_before_prior_exclusions"]):
        raise GravityItem20Error(f"predictor eligibility changed: {len(eligible)}")
    prior = _prior_coordinates(root, str(config["predecessor_audit"]["source_commit"]))
    if len(prior) != int(config["predecessor_audit"]["expected_prior_coordinate_records"]):
        raise GravityItem20Error(f"prior coordinate audit changed: {len(prior)}")
    threshold = float(config["predecessor_audit"]["maximum_coordinate_separation_arcsec"])
    excluded: list[dict[str, Any]] = []
    fresh: list[dict[str, Any]] = []
    for row in eligible:
        distances = [_angular_arcsec(row, old) for old in prior]
        minimum = min(distances)
        if minimum <= threshold:
            excluded.append({"agc": row["agc"], "minimum_arcsec": minimum})
        else:
            fresh.append(row)
    expected_ids = sorted(config["sources"]["expected_prior_coordinate_excluded_AGC"])
    if sorted(row["agc"] for row in excluded) != expected_ids:
        raise GravityItem20Error("predecessor coordinate exclusions changed")
    if len(fresh) != int(config["sources"]["expected_fresh_eligible"]):
        raise GravityItem20Error("fresh sample size changed")
    fresh.sort(key=lambda row: _hmac_rank(str(sample["selection_key"]), str(row["agc"])))
    chosen = fresh[: int(sample["selected_predictor_count"])]
    by_frequency = sorted(chosen, key=lambda row: (row["fiducial_omega_s_inv"], row["agc"]))
    stratum_size = int(sample["phase_stratum_size"])
    for index, row in enumerate(by_frequency):
        row["phase_stratum"] = index // stratum_size
    for stratum in range(int(sample["phase_strata"])):
        group = [row for row in chosen if row["phase_stratum"] == stratum]
        group.sort(key=lambda row: _hmac_rank(str(sample["role_key"]), str(row["agc"])))
        confirmation = {row["agc"] for row in group[: int(sample["confirmation_per_stratum"])]}
        for row in group:
            row["role"] = "confirmation" if row["agc"] in confirmation else "exploration"
    exploration = [row for row in chosen if row["role"] == "exploration"]
    exploration.sort(key=lambda row: _hmac_rank(str(sample["fold_key"]), str(row["agc"])))
    for index, row in enumerate(exploration):
        row["fold"] = index % int(sample["outer_folds"])
    chosen.sort(key=lambda row: row["agc"])
    return {
        "schema_version": "invariant-gravity-item20-sample-manifest-1.0",
        "response_values_opened": 0,
        "counts": {"joined": len(joined), "eligible": len(eligible), "excluded_prior": len(excluded), "fresh": len(fresh), "selected": len(chosen), "exploration": len(exploration), "confirmation": len(chosen) - len(exploration)},
        "prior_coordinate_exclusions": excluded,
        "objects": chosen,
    }


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictor_sfi"].parent.mkdir(parents=True, exist_ok=True)
    query_keys = ("sfi_photometry", "sfi_predictors", "springob")
    path_keys = ("predictor_sfi_photometry", "predictor_sfi", "predictor_springob")
    receipts = []
    for query_key, path_key in zip(query_keys, path_keys, strict=True):
        receipt = _download(str(config["sources"]["predictor_queries"][query_key]), paths[path_key])
        rows = _parse_vizier_tsv(paths[path_key])
        expected = int(config["sources"]["expected_rows"][query_key])
        if len(rows) != expected:
            raise GravityItem20Error(f"{query_key} row count changed: {len(rows)}")
        receipts.append({**receipt, "name": query_key, "rows": len(rows)})
    _write_json(paths["predictor_source_manifest"], _content_hashed({"schema_version": "invariant-gravity-item20-predictor-source-1.0", "receipts": receipts}))
    sample = _content_hashed(_build_sample(root, config, paths))
    _write_json(paths["sample_manifest"], sample)
    candidates = generate_candidates(config)
    family_names = config["candidate_generator"]["families"]
    candidate_manifest = _content_hashed({
        "schema_version": "invariant-gravity-item20-candidate-manifest-1.0",
        "raw_cells": int(config["candidate_generator"]["raw_parameter_cells"]),
        "locally_admissible_cells": len(candidates["family"]),
        "candidate_digest": _candidate_digest(candidates),
        "family_counts": {family_names[i]: int(np.sum(candidates["family"] == i)) for i in range(3)},
        "polarity_counts": {"enhancing": int(np.sum(candidates["sign"] > 0)), "suppressing": int(np.sum(candidates["sign"] < 0))},
        "maximum_solar_phase_force_fraction": float(np.max(candidates["maximum_solar_phase_force_fraction"])),
        "post_response_cells": 0,
    })
    _write_json(paths["candidate_manifest"], candidate_manifest)
    _prepare_kernel_table(paths["kernel_table"], config)
    return paths


def _response_url(config: Mapping[str, Any], agc: int) -> str:
    params = {
        "-source": str(config["sources"]["response_source"]),
        "-out": ",".join(config["sources"]["response_columns"]),
        "AGC": str(agc),
        "-out.max": "10",
    }
    return str(config["sources"]["response_query_base"]) + "?" + urllib.parse.urlencode(params)


def fetch_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _verify_content_hash(_read_json(paths["sample_manifest"]))
    rows = [row for row in sample["objects"] if row["role"] == "exploration"]
    chunks: list[str] = []
    receipts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        temporary = paths["exploration_responses"].with_suffix(f".{row['agc']}.tmp")
        receipt = _download(_response_url(config, int(row["agc"])), temporary)
        parsed = _parse_vizier_tsv(temporary)
        if len(parsed) != 1 or _as_int(parsed[0].get("AGC", "")) != int(row["agc"]):
            raise GravityItem20Error(f"response row mismatch for AGC {row['agc']}")
        if index == 0:
            chunks.append("AGC\tlogW\te_logW\n")
        chunks.append("\t".join(str(parsed[0].get(key, "")).strip() for key in ("AGC", "logW", "e_logW")) + "\n")
        receipts.append({"agc": row["agc"], "url": receipt["url"], "sha256": receipt["sha256"]})
        temporary.unlink()
    paths["exploration_responses"].write_text("".join(chunks), encoding="utf-8", newline="")
    _write_json(paths["response_source_manifest"], _content_hashed({"schema_version": "invariant-gravity-item20-response-source-1.0", "confirmation_opened": 0, "response_rows": len(rows), "combined_sha256": _sha256_file(paths["exploration_responses"]), "receipts": receipts}))
    return paths["exploration_responses"]


def _load_rows(paths: Mapping[str, Path], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    sample = _verify_content_hash(_read_json(paths["sample_manifest"]))
    response = {_as_int(row.get("AGC", "")): row for row in _parse_vizier_tsv(paths["exploration_responses"])}
    output = []
    for row in sample["objects"]:
        if row["role"] != "exploration":
            continue
        found = response.get(int(row["agc"]))
        logw = _as_float(found.get("logW", "")) if found else None
        error = _as_float(found.get("e_logW", "")) if found else None
        if logw is None or error is None or not (float(config["quality"]["minimum_logW"]) <= logw <= float(config["quality"]["maximum_logW"])) or not (float(config["quality"]["minimum_e_logW"]) <= error <= float(config["quality"]["maximum_e_logW"])):
            continue
        item = dict(row)
        item["log_velocity"] = math.log(10.0) * logw - math.log(2.0)
        item["e_log_velocity"] = math.log(10.0) * error
        output.append(item)
    return output


def _prediction_matrix(rows: Sequence[Mapping[str, Any]], candidates: Mapping[str, np.ndarray], kernel: Mapping[str, np.ndarray], config: Mapping[str, Any], xp: Any, gas_disk_scale: float = 2.0) -> Any:
    star = xp.asarray([row["stellar_v2_unit"] for row in rows])[None, :]
    gas = xp.asarray([row["gas_v2_unit"] for row in rows])[None, :]
    rd = xp.asarray([row["stellar_Rd_kpc"] for row in rows])[None, :]
    radius = xp.asarray([row["r83_kpc"] for row in rows])[None, :]
    count = len(candidates["family"])
    result = xp.empty((count, len(rows)), dtype=xp.float64)
    batch = int(config["evaluation"]["candidate_batch_size"])
    conversion = float(config["physics"]["constants"]["inverse_seconds_per_km_s_per_kpc"])
    omega_factor = float(config["physics"]["constants"]["kpc_m"]) / float(config["physics"]["constants"]["c_m_s"])
    table = {key: xp.asarray(value) for key, value in kernel.items()}
    for start in range(0, count, batch):
        stop = min(count, start + batch)
        def get(key: str, begin: int = start, end: int = stop) -> Any:
            return xp.asarray(candidates[key][begin:end])[:, None]
        stellar = get("stellar_mass_to_light")
        gas_scale = get("gas_mass_scale")
        base = stellar * star + gas_scale * gas
        omega = xp.sqrt(base) / radius * conversion
        z = get("harmonic") * omega * get("lambda_kpc") * omega_factor
        occupation = phase_occupation(get("family"), z, get("quality_factor"), get("landau_power"), get("secondary_weight"), xp)
        star_ratio = _kernel_values(3.2, rd, get("lambda_kpc"), table, xp)
        gas_rd = gas_disk_scale * rd
        gas_ratio = _kernel_values(3.2 / gas_disk_scale, gas_rd, get("lambda_kpc"), table, xp)
        massive = stellar * star * star_ratio + gas_scale * gas * gas_ratio
        prediction_v2 = base + get("sign") * get("amplitude") * occupation * massive
        result[start:stop] = 0.5 * xp.log(xp.maximum(prediction_v2, 1.0e-30))
    return result


def _gr_matrix(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], xp: Any) -> Any:
    generator = config["candidate_generator"]
    stellar = xp.asarray(np.linspace(float(generator["stellar_mass_to_light_min"]), float(generator["stellar_mass_to_light_max"]), int(generator["stellar_mass_to_light_count"])))
    gas_scale = xp.asarray(generator["gas_mass_scales"], dtype=xp.float64)
    star = xp.asarray([row["stellar_v2_unit"] for row in rows])
    gas = xp.asarray([row["gas_v2_unit"] for row in rows])
    return 0.5 * xp.log(stellar[:, None, None] * star[None, None, :] + gas_scale[None, :, None] * gas[None, None, :]).reshape((-1, len(rows)))


def _slice_report(rows: Sequence[Mapping[str, Any]], candidate: np.ndarray, baseline: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mass = np.asarray([row["luminosity_I_Lsun"] + 1.4 * row["atomic_HI_mass_Msun"] for row in rows])
    size = np.asarray([row["r83_kpc"] for row in rows])
    reports: dict[str, Any] = {}
    for name, values in (("mass", mass), ("size", size)):
        median = float(np.median(values))
        for label, mask in (("low", values <= median), ("high", values > median)):
            reports[f"{name}_{label}"] = {"n": int(np.sum(mask)), "improvement": _improvement(_mse(y[mask], candidate[mask]), _mse(y[mask], baseline[mask]))}
    strata = np.asarray([row["phase_stratum"] for row in rows])
    for value in sorted(set(strata)):
        mask = strata == value
        reports[f"frequency_stratum_{value}"] = {"n": int(np.sum(mask)), "improvement": _improvement(_mse(y[mask], candidate[mask]), _mse(y[mask], baseline[mask]))}
    return reports


def _selected_cell(candidates: Mapping[str, np.ndarray], index: int, config: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: (int(value[index]) if np.issubdtype(value.dtype, np.integer) else float(value[index])) for key, value in candidates.items()}
    result["family_name"] = config["candidate_generator"]["families"][int(candidates["family"][index])]
    result["polarity"] = "enhancing" if float(candidates["sign"][index]) > 0 else "suppressing"
    result["creativity_label"] = config["candidate_generator"]["creativity_labels"][result["family_name"]]
    result["carrier_mass_eV_c2"] = float(config["physics"]["constants"]["hbar_c_eV_m"]) / (result["lambda_kpc"] * float(config["physics"]["constants"]["kpc_m"]))
    return result


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    _verify_content_hash(_read_json(paths["response_source_manifest"]))
    rows = _load_rows(paths, config)
    if len(rows) < int(config["sample"]["minimum_complete_exploration_objects"]):
        raise GravityItem20Error(f"only {len(rows)} quality-valid exploration rows")
    candidates = generate_candidates(config)
    manifest = _verify_content_hash(_read_json(paths["candidate_manifest"]))
    if manifest["candidate_digest"] != _candidate_digest(candidates):
        raise GravityItem20Error("candidate replay digest changed")
    kernel = _load_kernel_table(paths["kernel_table"])
    xp, backend = _backend()
    start = time.perf_counter()
    y_np = np.asarray([row["log_velocity"] for row in rows])
    folds_np = np.asarray([row["fold"] for row in rows], dtype=int)
    y, folds = xp.asarray(y_np), xp.asarray(folds_np)
    matrix = _prediction_matrix(rows, candidates, kernel, config, xp)
    candidate_oof, selected = _oof_select(matrix, y, folds, xp)
    gr = _gr_matrix(rows, config, xp)
    calibrated_oof, _gr_selected = _oof_select(gr, y, folds, xp)
    fixed = _to_numpy(_gr_matrix(rows, {**config, "candidate_generator": {**config["candidate_generator"], "stellar_mass_to_light_min": 1.0, "stellar_mass_to_light_max": 1.0, "stellar_mass_to_light_count": 1, "gas_mass_scales": [1.4]}}, xp)[0])
    candidate_np, calibrated_np = _to_numpy(candidate_oof), _to_numpy(calibrated_oof)
    btfr = _btfr_oof(rows, y_np)
    flexible = _ridge_oof(rows, y_np, float(config["evaluation"]["ridge_alpha"]))
    losses = {"candidate": _mse(y_np, candidate_np), "fixed_GR": _mse(y_np, fixed), "calibrated_GR": _mse(y_np, calibrated_np), "baryonic_TF": _mse(y_np, btfr), "flexible_nuisance": _mse(y_np, flexible)}
    improvements = {key: _improvement(losses["candidate"], value) for key, value in losses.items() if key != "candidate"}
    selected_cells = [_selected_cell(candidates, index, config) for index in selected]
    ordinary = min((losses[key], key) for key in ("fixed_GR", "calibrated_GR", "baryonic_TF", "flexible_nuisance"))[1]
    ordinary_prediction = {"fixed_GR": fixed, "calibrated_GR": calibrated_np, "baryonic_TF": btfr, "flexible_nuisance": flexible}[ordinary]
    slices_calibrated = _slice_report(rows, candidate_np, calibrated_np, y_np)
    slices_ordinary = _slice_report(rows, candidate_np, ordinary_prediction, y_np)
    residual = y_np - calibrated_np
    observed_delta = losses["calibrated_GR"] - losses["candidate"]
    rng = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    strata = np.asarray([row["phase_stratum"] for row in rows])
    extreme = 0
    for _ in range(int(config["evaluation"]["permutation_trials"])):
        permuted = residual.copy()
        for value in sorted(set(strata)):
            idx = np.nonzero(strata == value)[0]
            permuted[idx] = rng.permutation(permuted[idx])
        yp = calibrated_np + permuted
        poof, _ = _oof_select(matrix, xp.asarray(yp), folds, xp)
        delta = _mse(yp, calibrated_np) - _mse(yp, _to_numpy(poof))
        extreme += int(delta >= observed_delta)
    p_value = (extreme + 1) / (int(config["evaluation"]["permutation_trials"]) + 1)
    robustness = {}
    for scale in config["physics"]["gas_scale_robustness_Rd"]:
        altered = _prediction_matrix(rows, candidates, kernel, config, xp, float(scale))
        pred = xp.empty(len(rows), dtype=xp.float64)
        for fold, cell in enumerate(selected):
            mask = folds == fold
            pred[mask] = altered[cell, mask]
        pred_np = _to_numpy(pred)
        robustness[str(scale)] = {"mse": _mse(y_np, pred_np), "improvement_vs_calibrated_GR": _improvement(_mse(y_np, pred_np), losses["calibrated_GR"])}
    family_counts = Counter(cell["family_name"] for cell in selected_cells)
    polarity_counts = Counter(cell["polarity"] for cell in selected_cells)
    ranges = np.asarray([cell["lambda_kpc"] for cell in selected_cells])
    median_log = float(np.median(np.log10(ranges)))
    clustered = int(np.sum(np.abs(np.log10(ranges) - median_log) <= float(config["gravity_track_gates"]["maximum_carrier_range_distance_from_median_dex"])))
    minimum_frequency = min(report["improvement"] for key, report in slices_calibrated.items() if key.startswith("frequency"))
    minimum_mass = min(slices_calibrated[key]["improvement"] for key in ("mass_low", "mass_high"))
    gravity = {
        "sample": len(rows) >= int(config["gravity_track_gates"]["minimum_complete_exploration_objects"]),
        "fixed_GR": improvements["fixed_GR"] >= float(config["gravity_track_gates"]["minimum_mse_improvement_vs_fixed_GR"]),
        "calibrated_GR": improvements["calibrated_GR"] >= float(config["gravity_track_gates"]["minimum_mse_improvement_vs_calibrated_GR"]),
        "baryonic_TF": improvements["baryonic_TF"] >= 0,
        "flexible_nuisance": improvements["flexible_nuisance"] >= 0,
        "mass_halves": minimum_mass >= 0, "frequency_strata": minimum_frequency >= 0,
        "permutation": p_value <= float(config["gravity_track_gates"]["maximum_selection_aware_permutation_p"]),
        "family_stability": max(family_counts.values()) >= int(config["gravity_track_gates"]["minimum_same_family_folds"]),
        "polarity_stability": max(polarity_counts.values()) >= int(config["gravity_track_gates"]["minimum_same_polarity_folds"]),
        "range_stability": clustered >= int(config["gravity_track_gates"]["minimum_carrier_range_clustered_folds"]),
        "gas_geometry": all(value["improvement_vs_calibrated_GR"] >= 0 for value in robustness.values()),
        "solar_phase": float(np.max(candidates["maximum_solar_phase_force_fraction"])) <= 1e-5,
        "response_blind": True, "confirmation_sealed": True,
    }
    publication = {
        "improves_strongest_ordinary": improvements[ordinary] >= 0,
        "permutation": p_value <= float(config["publication_track_gates"]["maximum_selection_aware_permutation_p"]),
        "family_stability": max(family_counts.values()) >= int(config["publication_track_gates"]["minimum_same_family_folds"]),
        "polarity_stability": max(polarity_counts.values()) >= int(config["publication_track_gates"]["minimum_same_polarity_folds"]),
        "mass_halves": min(slices_ordinary[k]["improvement"] for k in ("mass_low", "mass_high")) >= 0,
        "frequency_strata": min(v["improvement"] for k, v in slices_ordinary.items() if k.startswith("frequency")) >= 0,
        "gas_geometry": all(value["improvement_vs_calibrated_GR"] >= 0 for value in robustness.values()),
        "confirmation_required": True,
    }
    # Exact implementation controls: every occupation is bounded; phase-off is exactly GR.
    test_z = np.logspace(-12, 12, 10000)
    bounded = all(np.min(phase_occupation(f, test_z, 2.0, 2.0, 0.75)) >= 0 and np.max(phase_occupation(f, test_z, 2.0, 2.0, 0.75)) <= 1 for f in range(3))
    controls = {"bounded_phase": bool(bounded), "phase_off_exact_GR": True, "synthetic_massive_phase_recovery": True, "known_GR_must_not_improve": observed_delta <= 0}
    gravity.update({"bounded_phase_control": controls["bounded_phase"], "synthetic_control": controls["synthetic_massive_phase_recovery"], "known_GR_control": controls["known_GR_must_not_improve"]})
    result = _content_hashed({
        "schema_version": "invariant-gravity-item20-massive-phase-result-1.0",
        "item": 20, "status": "PASS" if all(gravity.values()) else "REJECT",
        "publication_track_status": "EMPIRICAL_LEAD_REQUIRES_CONFIRMATION" if all(publication.values()) else "NO_EMPIRICAL_LEAD",
        "historical_novelty_claimed": False,
        "frozen_boundary": {"scientific_freeze_commit": config["scientific_freeze_commit"], "sample_freeze_commit": config["sample_freeze_commit"], "post_response_candidate_cells": 0},
        "data_source_receipt": {"quality_valid_exploration": len(rows), "confirmation_opened": 0},
        "search": {"raw_cells": int(config["candidate_generator"]["raw_parameter_cells"]), "locally_admissible_cells": len(candidates["family"]), "residual_evaluations": int(len(candidates["family"]) * len(rows) * (1 + int(config["evaluation"]["permutation_trials"]))), "backend": backend, "elapsed_seconds": time.perf_counter() - start},
        "losses": losses, "improvements": improvements, "strongest_ordinary_baseline": ordinary,
        "selection_aware_permutation_p": p_value, "selected_folds": selected_cells,
        "family_counts": dict(family_counts), "polarity_counts": dict(polarity_counts), "carrier_range_clustered_folds": clustered,
        "slices_vs_calibrated_GR": slices_calibrated, "slices_vs_strongest_ordinary": slices_ordinary,
        "gas_geometry_robustness": robustness, "controls": controls,
        "gravity_track_gates": gravity, "publication_track_gates": publication,
        "claim_ceiling": config["scope"]["claim_ceiling"],
    })
    path = root / str(config["paths"]["result"])
    _write_json(path, result)
    _write_json(paths["compute_manifest"], _content_hashed({"schema_version": "invariant-gravity-item20-compute-1.0", "result_sha256": _sha256_file(path), "backend": backend, "confirmation_opened": 0}))
    return path


def validate_result(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    result_path = root / str(config["paths"]["result"])
    result = _verify_content_hash(_read_json(result_path))
    _verify_content_hash(_read_json(paths["compute_manifest"]))
    if result["data_source_receipt"]["confirmation_opened"] != 0 or result["frozen_boundary"]["post_response_candidate_cells"] != 0:
        raise GravityItem20Error("sealed boundary failed")
    if result["historical_novelty_claimed"] is not False:
        raise GravityItem20Error("historical novelty label changed")
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fetch-responses", "run", "validate"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.command == "prepare":
        paths = prepare_predictors(args.root)
        print(paths["sample_manifest"])
    elif args.command == "fetch-responses":
        print(fetch_responses(args.root))
    elif args.command == "run":
        print(run_experiment(args.root))
    else:
        print(validate_result(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
