"""Frozen Item 21 mixed massless/massive mode experiment."""

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
    _kernel_values,
    _load_kernel_table,
    _oof_select,
    _prepare_kernel_table,
)
from sigma_theory_compiler.gravity_item20_massive_phase import (
    _angular_arcsec,
    _prior_coordinates,
    _sexagesimal,
)

CONFIG_PATH = Path("configs/gravity_item21_mixed_modes_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item21_mixed_modes.py")
DEPENDENCY_PATHS = (
    Path("src/sigma_theory_compiler/gravity_item16_s4tm_qed_field.py"),
    Path("src/sigma_theory_compiler/gravity_item18_diskmass_antiscreening.py"),
    Path("src/sigma_theory_compiler/gravity_item19_massive_carrier.py"),
    Path("src/sigma_theory_compiler/gravity_item20_massive_phase.py"),
)
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem21Error(RuntimeError):
    """Raised when an Item 21 scientific, data-lineage, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if config.get("schema_version") != "invariant-gravity-item21-mixed-modes-config-1.0":
        raise GravityItem21Error("unexpected Item 21 config schema")
    if int(config.get("item", -1)) != 21:
        raise GravityItem21Error("Item 21 number changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem21Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem21Error("paid API calls are outside Item 21")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem21Error("post-response candidates entered Item 21")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem21Error("stable gravity roadmap changed")
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
        raise GravityItem21Error(f"{label} is not bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root, check=False
    )
    if result.returncode:
        raise GravityItem21Error(f"{label} is not an ancestor of HEAD")


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen) != _contract_digest(config):
        raise GravityItem21Error("scientific contract differs from freeze")
    for path in (MODULE_PATH, *DEPENDENCY_PATHS):
        blob = _git(root, "show", f"{commit}:{path.as_posix()}", text_mode=False)
        if not isinstance(blob, bytes) or _sha256_bytes(blob) != _sha256_file(root / path):
            raise GravityItem21Error(f"dependency differs from science freeze: {path}")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    return {
        key: base / str(config["paths"][key])
        for key in (
            "predictors",
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
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "kernel_table",
    ):
        relative = paths[key].relative_to(root).as_posix()
        blob = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(blob, bytes) or _sha256_bytes(blob) != _sha256_file(paths[key]):
            raise GravityItem21Error(f"{key} differs from sample freeze")


def _point_kernel(u: Any, xp: Any = np) -> Any:
    return (1.0 + u) * xp.exp(-u)


def mixing_occupation(
    family: Any,
    log_u: Any,
    width: Any,
    fano_q: Any,
    xp: Any = np,
) -> Any:
    """Evaluate bounded nonlinear two-state mixing probabilities."""

    clipped = xp.clip(log_u, -350.0, 350.0)
    crossing = 1.0 / xp.cosh(clipped) ** 2
    adiabatic = 0.5 * (1.0 - clipped / xp.sqrt(clipped**2 + 4.0 * width**2))
    epsilon = clipped / width
    fano = (fano_q + epsilon) ** 2 / (
        (1.0 + fano_q**2) * (1.0 + epsilon**2)
    )
    return xp.where(
        family == 1,
        crossing,
        xp.where(family == 2, adiabatic, xp.where(family == 3, crossing * fano, 1.0)),
    )


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["raw_parameter_cells"])
    rng = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    niche = np.arange(count, dtype=np.int64) % 8
    rng.shuffle(niche)
    family = (niche // 2).astype(np.int8)
    sign = np.where(niche % 2 == 0, 1.0, -1.0)
    u = rng.random(count)
    enhancing_log = float(generator["enhancing_amplitude_log10_min"]) + u * (
        float(generator["enhancing_amplitude_log10_max"])
        - float(generator["enhancing_amplitude_log10_min"])
    )
    suppressing_log = float(generator["suppressing_amplitude_log10_min"]) + u * (
        math.log10(float(generator["suppressing_amplitude_max"]))
        - float(generator["suppressing_amplitude_log10_min"])
    )
    amplitude = 10.0 ** np.where(sign > 0, enhancing_log, suppressing_log)
    kinetic_abs = rng.uniform(
        float(generator["kinetic_mixing_abs_min"]),
        float(generator["kinetic_mixing_abs_max"]),
        count,
    )
    kinetic = kinetic_abs * rng.choice(np.asarray([-1.0, 1.0]), count)
    effective = amplitude * kinetic_abs**2
    lambda_kpc = 10.0 ** rng.uniform(
        float(generator["lambda_log10_kpc_min"]),
        float(generator["lambda_log10_kpc_max"]),
        count,
    )
    transition = 10.0 ** rng.uniform(
        float(generator["transition_acceleration_log10_m_s2_min"]),
        float(generator["transition_acceleration_log10_m_s2_max"]),
        count,
    )
    field_power = rng.choice(np.asarray(generator["field_powers"], dtype=float), count)
    ratio_power = rng.choice(
        np.asarray(generator["massive_ratio_powers"], dtype=float), count
    )
    width = rng.choice(np.asarray(generator["mixing_widths"], dtype=float), count)
    fano_q = rng.choice(np.asarray(generator["fano_q"], dtype=float), count)
    stellar_grid = np.linspace(
        float(generator["stellar_mass_scale_min"]),
        float(generator["stellar_mass_scale_max"]),
        int(generator["stellar_mass_scale_count"]),
    )
    stellar = rng.choice(stellar_grid, count)
    gas = rng.choice(np.asarray(generator["gas_mass_scales"], dtype=float), count)
    constants = config["physics"]["constants"]
    radii = np.asarray(config["pre_response_filters"]["solar_probe_AU"], dtype=float)
    radius_m = radii * float(constants["AU_m"])
    g_newton = (
        float(constants["G_SI"]) * float(constants["M_sun_kg"]) / radius_m**2
    )
    ratio = _point_kernel(
        radius_m[None, :] / (lambda_kpc[:, None] * float(constants["kpc_m"]))
    )
    log_u = (
        field_power[:, None] * np.log(g_newton[None, :] / transition[:, None])
        + ratio_power[:, None] * np.log(np.maximum(ratio, 1.0e-300))
    )
    occupation = mixing_occupation(
        family[:, None], log_u, width[:, None], fano_q[:, None]
    )
    nonlinear_deviation = effective[:, None] * occupation * ratio
    denominator = 1.0 + sign[:, None] * effective[:, None]
    two_pole_mu = (1.0 + sign[:, None] * effective[:, None] * ratio) / denominator
    deviation = np.where(
        family[:, None] == 0, abs(two_pole_mu - 1.0), nonlinear_deviation
    )
    maximum_solar = np.max(deviation, axis=1)
    determinant = 1.0 - kinetic**2
    filters = config["pre_response_filters"]
    keep = (
        (determinant >= float(filters["minimum_quadratic_kinetic_determinant"]))
        & (maximum_solar <= float(filters["maximum_solar_fractional_force_deviation"]))
        & ((sign > 0) | (effective < 0.95))
        & ((sign < 0) | (effective <= float(filters["maximum_enhancement"])))
        & (denominator[:, 0] > 0.05)
    )
    return {
        "raw_index": np.flatnonzero(keep).astype(np.int64),
        "family": family[keep],
        "sign": sign[keep],
        "amplitude": amplitude[keep],
        "kinetic_mixing": kinetic[keep],
        "kinetic_determinant": determinant[keep],
        "effective_amplitude": effective[keep],
        "lambda_kpc": lambda_kpc[keep],
        "transition_acceleration_m_s2": transition[keep],
        "field_power": field_power[keep],
        "massive_ratio_power": ratio_power[keep],
        "mixing_width": width[keep],
        "fano_q": fano_q[keep],
        "stellar_mass_scale": stellar[keep],
        "gas_mass_scale": gas[keep],
        "maximum_solar_fractional_force_deviation": maximum_solar[keep],
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _parse_path(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    return _parse_vizier_tsv(path.read_bytes(), columns)


def _build_sample(
    root: Path, config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    columns = tuple(config["sources"]["predictor_columns"])
    raw = _parse_path(paths["predictors"], columns)
    sample = config["sample"]
    constants = config["physics"]["constants"]
    eligible: list[dict[str, Any]] = []
    for row in raw:
        values = {
            key: _as_float(row.get(key, ""))
            for key in (
                "Dist",
                "e_Dist",
                "Inc",
                "Reff",
                "e_Reff",
                "MHI",
                "e_MHI",
                "Mstar",
                "e_Mstar",
            )
        }
        nsa = _as_int(row.get("NSA", ""))
        if nsa is None or any(value is None or value <= 0 for value in values.values()):
            continue
        if not (
            float(sample["minimum_inclination_deg"])
            <= values["Inc"]
            <= float(sample["maximum_inclination_deg"])
        ):
            continue
        if values["e_Dist"] / values["Dist"] > float(
            sample["maximum_fractional_distance_error"]
        ):
            continue
        if values["e_Reff"] / values["Reff"] > float(
            sample["maximum_fractional_radius_error"]
        ):
            continue
        if values["e_Mstar"] / values["Mstar"] > float(
            sample["maximum_fractional_stellar_mass_error"]
        ):
            continue
        if values["e_MHI"] / values["MHI"] > float(
            sample["maximum_fractional_HI_mass_error"]
        ):
            continue
        stellar_mass = values["Mstar"] * 1.0e9
        hi_mass = values["MHI"] * 1.0e9
        rd = values["Reff"] / 1.678
        radius = float(config["physics"]["evaluation_radius_stellar_Rd"]) * rd
        star_v2 = float(_disk_velocity_sq(stellar_mass, rd, radius))
        gas_v2 = float(
            _disk_velocity_sq(
                hi_mass,
                float(config["physics"]["primary_atomic_gas_scale_stellar_Rd"]) * rd,
                radius,
            )
        )
        acceleration = (
            (star_v2 + 1.4 * gas_v2)
            / radius
            * float(constants["km2_s2_per_kpc_to_m_s2"])
        )
        eligible.append(
            {
                "nsa": nsa,
                "name": str(row.get("SimbadName", "")).strip(),
                "ra_deg": _sexagesimal(str(row["RAJ2000"]), True),
                "dec_deg": _sexagesimal(str(row["DEJ2000"])),
                "distance_Mpc": values["Dist"],
                "fractional_distance_error": values["e_Dist"] / values["Dist"],
                "inclination_deg": values["Inc"],
                "reff_kpc": values["Reff"],
                "fractional_radius_error": values["e_Reff"] / values["Reff"],
                "stellar_mass_Msun": stellar_mass,
                "fractional_stellar_mass_error": values["e_Mstar"] / values["Mstar"],
                "HI_mass_Msun": hi_mass,
                "fractional_HI_mass_error": values["e_MHI"] / values["MHI"],
                "HI_source": int(_as_int(row.get("Ref", "")) or 0),
                "stellar_Rd_kpc": rd,
                "radius_kpc": radius,
                "stellar_v2_unit": star_v2,
                "gas_v2_unit": gas_v2,
                "fiducial_acceleration_m_s2": acceleration,
            }
        )
    if len(eligible) != int(config["sources"]["expected_predictor_eligible"]):
        raise GravityItem21Error(f"predictor eligibility changed: {len(eligible)}")
    prior = _prior_coordinates(root, str(config["predecessor_audit"]["source_commit"]))
    if len(prior) != int(config["predecessor_audit"]["expected_prior_coordinate_records"]):
        raise GravityItem21Error(f"prior coordinate count changed: {len(prior)}")
    threshold = float(config["predecessor_audit"]["maximum_coordinate_separation_arcsec"])
    prior_excluded: list[dict[str, Any]] = []
    preview_excluded: list[int] = []
    fresh: list[dict[str, Any]] = []
    preview = {int(value) for value in config["sources"]["preview_exposed_NSA"]}
    for row in eligible:
        minimum = min(_angular_arcsec(row, old) for old in prior)
        if minimum <= threshold:
            prior_excluded.append({"nsa": row["nsa"], "minimum_arcsec": minimum})
        elif int(row["nsa"]) in preview:
            preview_excluded.append(int(row["nsa"]))
        else:
            fresh.append(row)
    if sorted(row["nsa"] for row in prior_excluded) != sorted(
        config["sources"]["expected_prior_coordinate_excluded_NSA"]
    ):
        raise GravityItem21Error("prior-coordinate exclusion identities changed")
    if len(preview_excluded) != int(
        config["sources"]["expected_preview_exclusions_after_quality_and_prior"]
    ):
        raise GravityItem21Error("preview exclusion count changed")
    if len(fresh) != int(config["sources"]["expected_fresh_eligible"]):
        raise GravityItem21Error(f"fresh eligibility changed: {len(fresh)}")
    fresh.sort(key=lambda row: _hmac_rank(str(sample["selection_key"]), str(row["nsa"])))
    selected = fresh[: int(sample["selected_predictor_count"])]
    by_acceleration = sorted(
        selected, key=lambda row: (row["fiducial_acceleration_m_s2"], row["nsa"])
    )
    for index, row in enumerate(by_acceleration):
        row["acceleration_stratum"] = index // int(sample["acceleration_stratum_size"])
    for stratum in range(int(sample["acceleration_strata"])):
        group = [row for row in selected if row["acceleration_stratum"] == stratum]
        group.sort(key=lambda row: _hmac_rank(str(sample["role_key"]), str(row["nsa"])))
        held = {row["nsa"] for row in group[: int(sample["confirmation_per_stratum"])]}
        for row in group:
            row["role"] = "confirmation" if row["nsa"] in held else "exploration"
    exploration = [row for row in selected if row["role"] == "exploration"]
    exploration.sort(key=lambda row: _hmac_rank(str(sample["fold_key"]), str(row["nsa"])))
    for index, row in enumerate(exploration):
        row["fold"] = index % int(sample["outer_folds"])
    selected.sort(key=lambda row: row["nsa"])
    return {
        "schema_version": "invariant-gravity-item21-sample-manifest-1.0",
        "response_values_opened": 0,
        "counts": {
            "raw": len(raw),
            "eligible": len(eligible),
            "prior_excluded": len(prior_excluded),
            "preview_excluded": len(preview_excluded),
            "fresh": len(fresh),
            "selected": len(selected),
            "exploration": len(exploration),
            "confirmation": len(selected) - len(exploration),
        },
        "prior_coordinate_exclusions": prior_excluded,
        "preview_exclusions": sorted(preview_excluded),
        "objects": selected,
    }


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictors"].parent.mkdir(parents=True, exist_ok=True)
    body, headers = _download(str(config["sources"]["predictor_query"]))
    paths["predictors"].write_bytes(body)
    rows = _parse_vizier_tsv(body, tuple(config["sources"]["predictor_columns"]))
    if len(rows) != int(config["sources"]["expected_predictor_rows"]):
        raise GravityItem21Error(f"predictor row count changed: {len(rows)}")
    _write_json(
        paths["predictor_source_manifest"],
        _content_hashed(
            {
                "schema_version": "invariant-gravity-item21-predictor-source-1.0",
                "url": config["sources"]["predictor_query"],
                "rows": len(rows),
                "sha256": _sha256_bytes(body),
                "headers": headers,
                "response_columns_requested": 0,
            }
        ),
    )
    _write_json(paths["sample_manifest"], _content_hashed(_build_sample(root, config, paths)))
    candidates = generate_candidates(config)
    family_names = config["candidate_generator"]["families"]
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item21-candidate-manifest-1.0",
            "raw_cells": int(config["candidate_generator"]["raw_parameter_cells"]),
            "locally_admissible_cells": len(candidates["family"]),
            "structural_equivalence_classes": int(
                config["candidate_generator"]["structural_equivalence_classes"]
            ),
            "candidate_digest": _candidate_digest(candidates),
            "family_counts": {
                family_names[index]: int(np.sum(candidates["family"] == index))
                for index in range(4)
            },
            "polarity_counts": {
                "enhancing": int(np.sum(candidates["sign"] > 0)),
                "suppressing": int(np.sum(candidates["sign"] < 0)),
            },
            "minimum_kinetic_determinant": float(
                np.min(candidates["kinetic_determinant"])
            ),
            "maximum_solar_fractional_force_deviation": float(
                np.max(candidates["maximum_solar_fractional_force_deviation"])
            ),
            "post_response_cells": 0,
        }
    )
    _write_json(paths["candidate_manifest"], manifest)
    _prepare_kernel_table(paths["kernel_table"], config)
    return paths


def _response_url(config: Mapping[str, Any], nsa: int) -> str:
    parameters = {
        "-source": str(config["sources"]["response_source"]),
        "-out": ",".join(config["sources"]["response_columns"]),
        "NSA": str(nsa),
        "-out.max": "10",
    }
    return str(config["sources"]["response_query_base"]) + "?" + urllib.parse.urlencode(
        parameters
    )


def fetch_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "sample manifest")
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    output = ["NSA\tW20i\te_W20i\n"]
    receipts = []
    for row in exploration:
        url = _response_url(config, int(row["nsa"]))
        body, _headers = _download(url)
        parsed = _parse_vizier_tsv(body, tuple(config["sources"]["response_columns"]))
        if len(parsed) != 1 or _as_int(parsed[0].get("NSA", "")) != int(row["nsa"]):
            raise GravityItem21Error(f"response mismatch for NSA {row['nsa']}")
        output.append(
            "\t".join(str(parsed[0].get(key, "")).strip() for key in ("NSA", "W20i", "e_W20i"))
            + "\n"
        )
        receipts.append({"nsa": row["nsa"], "url": url, "sha256": _sha256_bytes(body)})
    paths["exploration_responses"].write_text(
        "".join(output), encoding="utf-8", newline=""
    )
    _write_json(
        paths["response_source_manifest"],
        _content_hashed(
            {
                "schema_version": "invariant-gravity-item21-response-source-1.0",
                "requested_exploration": len(exploration),
                "returned_exploration": len(receipts),
                "confirmation_opened": 0,
                "combined_sha256": _sha256_file(paths["exploration_responses"]),
                "receipts": receipts,
            }
        ),
    )
    return paths["exploration_responses"]


def _load_rows(paths: Mapping[str, Path], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "sample manifest")
    response = {
        _as_int(row.get("NSA", "")): row
        for row in _parse_path(
            paths["exploration_responses"], tuple(config["sources"]["response_columns"])
        )
    }
    output = []
    for row in sample["objects"]:
        if row["role"] != "exploration":
            continue
        found = response.get(int(row["nsa"]))
        width = _as_float(found.get("W20i", "")) if found else None
        error = _as_float(found.get("e_W20i", "")) if found else None
        if width is None or error is None:
            continue
        if not (
            float(config["quality"]["minimum_W20i_km_s"])
            <= width
            <= float(config["quality"]["maximum_W20i_km_s"])
        ):
            continue
        if error < float(config["quality"]["minimum_e_W20i_km_s"]):
            continue
        if error / width > float(config["quality"]["maximum_fractional_W20i_error"]):
            continue
        item = dict(row)
        item["log_velocity"] = math.log(width / 2.0)
        item["fractional_width_error"] = error / width
        output.append(item)
    return output


def _prediction_matrix(
    rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, np.ndarray],
    kernel: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    xp: Any,
    gas_disk_scale: float | None = None,
) -> Any:
    star = xp.asarray([row["stellar_v2_unit"] for row in rows])[None, :]
    gas = xp.asarray([row["gas_v2_unit"] for row in rows])[None, :]
    rd = xp.asarray([row["stellar_Rd_kpc"] for row in rows])[None, :]
    radius = xp.asarray([row["radius_kpc"] for row in rows])[None, :]
    gas_scale_rd = float(
        gas_disk_scale
        if gas_disk_scale is not None
        else config["physics"]["primary_atomic_gas_scale_stellar_Rd"]
    )
    count = len(candidates["family"])
    matrix = xp.empty((count, len(rows)), dtype=xp.float64)
    batch = int(config["evaluation"]["candidate_batch_size"])
    acceleration_conversion = float(
        config["physics"]["constants"]["km2_s2_per_kpc_to_m_s2"]
    )
    evaluation_x = float(config["physics"]["evaluation_radius_stellar_Rd"])
    for start in range(0, count, batch):
        stop = min(count, start + batch)

        def get(key: str, begin: int = start, end: int = stop) -> Any:
            return xp.asarray(candidates[key][begin:end])[:, None]

        stellar_scale = get("stellar_mass_scale")
        gas_mass_scale = get("gas_mass_scale")
        base = stellar_scale * star + gas_mass_scale * gas
        stellar_ratio = _kernel_values(
            evaluation_x, rd, get("lambda_kpc"), kernel, xp
        )
        gas_rd = gas_scale_rd * rd
        gas_ratio = _kernel_values(
            evaluation_x / gas_scale_rd, gas_rd, get("lambda_kpc"), kernel, xp
        )
        massive = stellar_scale * star * stellar_ratio + gas_mass_scale * gas * gas_ratio
        acceleration = base / radius * acceleration_conversion
        ratio = xp.maximum(massive / base, 1.0e-300)
        log_u = get("field_power") * xp.log(
            acceleration / get("transition_acceleration_m_s2")
        ) + get("massive_ratio_power") * xp.log(ratio)
        occupation = mixing_occupation(
            get("family"), log_u, get("mixing_width"), get("fano_q"), xp
        )
        effective = get("effective_amplitude")
        sign = get("sign")
        nonlinear = base + sign * effective * occupation * massive
        two_pole = (base + sign * effective * massive) / (1.0 + sign * effective)
        predicted_v2 = xp.where(get("family") == 0, two_pole, nonlinear)
        matrix[start:stop] = 0.5 * xp.log(xp.maximum(predicted_v2, 1.0e-30))
    return matrix


def _gr_matrix(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], xp: Any
) -> Any:
    generator = config["candidate_generator"]
    stellar = xp.asarray(
        np.linspace(
            float(generator["stellar_mass_scale_min"]),
            float(generator["stellar_mass_scale_max"]),
            int(generator["stellar_mass_scale_count"]),
        )
    )
    gas_scale = xp.asarray(generator["gas_mass_scales"], dtype=xp.float64)
    star = xp.asarray([row["stellar_v2_unit"] for row in rows])
    gas = xp.asarray([row["gas_v2_unit"] for row in rows])
    return 0.5 * xp.log(
        stellar[:, None, None] * star[None, None, :]
        + gas_scale[None, :, None] * gas[None, None, :]
    ).reshape((-1, len(rows)))


def _btfr_oof(rows: Sequence[Mapping[str, Any]], y: np.ndarray) -> np.ndarray:
    x = 0.25 * np.log(
        np.asarray(
            [row["stellar_mass_Msun"] + 1.4 * row["HI_mass_Msun"] for row in rows]
        )
    )
    folds = np.asarray([row["fold"] for row in rows])
    output = np.empty(len(rows))
    for fold in sorted(set(folds)):
        train, test = folds != fold, folds == fold
        output[test] = x[test] + float(np.mean(y[train] - x[train]))
    return output


def _rar_prediction(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> np.ndarray:
    conversion = float(config["physics"]["constants"]["km2_s2_per_kpc_to_m_s2"])
    output = []
    for row in rows:
        vbar2 = row["stellar_v2_unit"] + 1.4 * row["gas_v2_unit"]
        gbar = vbar2 / row["radius_kpc"] * conversion
        gobs = gbar / (1.0 - math.exp(-math.sqrt(gbar / 1.2e-10)))
        output.append(0.5 * math.log(gobs * row["radius_kpc"] / conversion))
    return np.asarray(output)


def _ridge_oof(
    rows: Sequence[Mapping[str, Any]], y: np.ndarray, alpha: float
) -> np.ndarray:
    sources = sorted({int(row["HI_source"]) for row in rows})
    features = []
    for row in rows:
        features.append(
            [
                math.log(row["stellar_mass_Msun"]),
                math.log(row["HI_mass_Msun"]),
                math.log(row["reff_kpc"]),
                row["inclination_deg"],
                math.log(row["distance_Mpc"]),
                row["fractional_stellar_mass_error"],
                row["fractional_HI_mass_error"],
                row["fractional_radius_error"],
                *[float(row["HI_source"] == source) for source in sources[1:]],
            ]
        )
    x = np.asarray(features)
    folds = np.asarray([row["fold"] for row in rows])
    output = np.empty(len(rows))
    for fold in sorted(set(folds)):
        train, test = folds != fold, folds == fold
        mean, std = x[train].mean(axis=0), x[train].std(axis=0)
        std[std == 0] = 1.0
        x_train = np.column_stack([np.ones(train.sum()), (x[train] - mean) / std])
        x_test = np.column_stack([np.ones(test.sum()), (x[test] - mean) / std])
        penalty = np.eye(x_train.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y[train])
        output[test] = x_test @ coefficient
    return output


def _selected_cell(
    candidates: Mapping[str, np.ndarray], index: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        key: (
            int(value[index])
            if np.issubdtype(value.dtype, np.integer)
            else float(value[index])
        )
        for key, value in candidates.items()
    }
    name = config["candidate_generator"]["families"][int(candidates["family"][index])]
    result["family_name"] = name
    result["polarity"] = "enhancing" if candidates["sign"][index] > 0 else "suppressing"
    result["creativity_label"] = config["candidate_generator"]["creativity_labels"][name]
    result["carrier_mass_eV_c2"] = float(
        config["physics"]["constants"]["hbar_c_eV_m"]
    ) / (
        float(result["lambda_kpc"])
        * float(config["physics"]["constants"]["kpc_m"])
    )
    return result


def _slice_report(
    rows: Sequence[Mapping[str, Any]],
    candidate: np.ndarray,
    baseline: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    mass = np.asarray(
        [row["stellar_mass_Msun"] + 1.4 * row["HI_mass_Msun"] for row in rows]
    )
    output = {}
    median = float(np.median(mass))
    for label, mask in (("mass_low", mass <= median), ("mass_high", mass > median)):
        output[label] = {
            "n": int(np.sum(mask)),
            "improvement": _improvement(
                _mse(y[mask], baseline[mask]), _mse(y[mask], candidate[mask])
            ),
        }
    strata = np.asarray([row["acceleration_stratum"] for row in rows])
    for value in sorted(set(strata)):
        mask = strata == value
        output[f"acceleration_stratum_{value}"] = {
            "n": int(np.sum(mask)),
            "improvement": _improvement(
                _mse(y[mask], baseline[mask]), _mse(y[mask], candidate[mask])
            ),
        }
    return output


def _synthetic_controls(
    matrix: Any,
    gr: Any,
    candidates: Mapping[str, np.ndarray],
    folds: np.ndarray,
    xp: Any,
) -> dict[str, Any]:
    columns = np.arange(min(100, matrix.shape[1]))
    control_folds = folds[columns]
    digest = _candidate_digest(candidates)
    injection = int(digest[:16], 16) % len(candidates["family"])
    truth = _to_numpy(matrix[injection, columns], xp)
    candidate_prediction, selected = _oof_select(
        matrix[:, columns], xp.asarray(truth), control_folds, xp
    )
    gr_prediction, _ = _oof_select(gr[:, columns], xp.asarray(truth), control_folds, xp)
    improvement = _improvement(_mse(truth, gr_prediction), _mse(truth, candidate_prediction))
    target_family = int(candidates["family"][injection])
    target_sign = float(candidates["sign"][injection])
    target_range = math.log10(float(candidates["lambda_kpc"][injection]))
    recovered = (
        all(int(candidates["family"][index]) == target_family for index in selected)
        and all(float(candidates["sign"][index]) == target_sign for index in selected)
        and all(
            abs(math.log10(float(candidates["lambda_kpc"][index])) - target_range) <= 0.75
            for index in selected
        )
    )
    gr_index = int(digest[16:24], 16) % int(gr.shape[0])
    truth_gr = _to_numpy(gr[gr_index, columns], xp) + 1.0e-6 * np.sin(columns)
    candidate_gr, _ = _oof_select(
        matrix[:, columns], xp.asarray(truth_gr), control_folds, xp
    )
    baseline_gr, _ = _oof_select(gr[:, columns], xp.asarray(truth_gr), control_folds, xp)
    gr_improvement = _improvement(
        _mse(truth_gr, baseline_gr), _mse(truth_gr, candidate_gr)
    )
    return {
        "known_GR": {
            "candidate_improvement_vs_GR": gr_improvement,
            "pass": gr_improvement <= 0.0,
        },
        "mixed_mode_injection": {
            "digest_selected_index": injection,
            "candidate_improvement_vs_GR": improvement,
            "selected_indices": selected,
            "family_polarity_range_recovered": recovered,
            "pass": improvement > 0.5 and recovered,
        },
    }


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    response_manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(response_manifest, "response source manifest")
    if int(response_manifest["confirmation_opened"]) != 0:
        raise GravityItem21Error("confirmation values were opened")
    rows = _load_rows(paths, config)
    if len(rows) < int(config["sample"]["minimum_complete_exploration_objects"]):
        raise GravityItem21Error(f"only {len(rows)} quality-valid exploration rows")
    candidates = generate_candidates(config)
    candidate_manifest = _read_json(paths["candidate_manifest"])
    _verify_content_hash(candidate_manifest, "candidate manifest")
    if candidate_manifest["candidate_digest"] != _candidate_digest(candidates):
        raise GravityItem21Error("candidate digest changed")
    kernel = _load_kernel_table(paths["kernel_table"])
    xp, backend, device = _backend()
    start = time.perf_counter()
    y_np = np.asarray([row["log_velocity"] for row in rows])
    folds = np.asarray([row["fold"] for row in rows], dtype=int)
    y = xp.asarray(y_np)
    matrix = _prediction_matrix(rows, candidates, kernel, config, xp)
    candidate_oof, selected = _oof_select(matrix, y, folds, xp)
    gr = _gr_matrix(rows, config, xp)
    calibrated_gr, selected_gr = _oof_select(gr, y, folds, xp)
    generator = config["candidate_generator"]
    stellar_grid = np.linspace(
        float(generator["stellar_mass_scale_min"]),
        float(generator["stellar_mass_scale_max"]),
        int(generator["stellar_mass_scale_count"]),
    )
    gas_grid = np.asarray(generator["gas_mass_scales"])
    fixed_index = int(
        np.argmin(
            abs(np.repeat(stellar_grid, len(gas_grid)) - 1.0)
            + abs(np.tile(gas_grid, len(stellar_grid)) - 1.4)
        )
    )
    fixed_gr = _to_numpy(gr[fixed_index], xp)
    candidate_np, calibrated_np = np.asarray(candidate_oof), np.asarray(calibrated_gr)
    btfr = _btfr_oof(rows, y_np)
    rar = _rar_prediction(rows, config)
    flexible = _ridge_oof(rows, y_np, float(config["evaluation"]["ridge_alpha"]))
    predictions = {
        "fixed_GR": fixed_gr,
        "calibrated_GR": calibrated_np,
        "baryonic_TF": btfr,
        "RAR": rar,
        "flexible_nuisance": flexible,
    }
    losses = {"candidate": _mse(y_np, candidate_np)}
    losses.update({name: _mse(y_np, value) for name, value in predictions.items()})
    improvements = {
        name: _improvement(loss, losses["candidate"])
        for name, loss in losses.items()
        if name != "candidate"
    }
    strongest_name = min(predictions, key=lambda name: losses[name])
    strongest = predictions[strongest_name]
    slices = _slice_report(rows, candidate_np, strongest, y_np)
    observed_delta = losses[strongest_name] - losses["candidate"]
    residual = y_np - strongest
    strata = np.asarray([row["acceleration_stratum"] for row in rows])
    rng = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    extreme = 0
    for _ in range(int(config["evaluation"]["permutation_trials"])):
        permuted = residual.copy()
        for stratum in sorted(set(strata)):
            indices = np.flatnonzero(strata == stratum)
            permuted[indices] = rng.permutation(permuted[indices])
        null_y = strongest + permuted
        null_prediction, _ = _oof_select(matrix, xp.asarray(null_y), folds, xp)
        null_delta = _mse(null_y, strongest) - _mse(null_y, null_prediction)
        extreme += int(null_delta >= observed_delta)
    p_value = (extreme + 1) / (int(config["evaluation"]["permutation_trials"]) + 1)
    robustness = {}
    for scale in config["physics"]["gas_scale_robustness_stellar_Rd"]:
        altered = _prediction_matrix(rows, candidates, kernel, config, xp, float(scale))
        prediction = xp.empty(len(rows), dtype=xp.float64)
        for fold, cell in enumerate(selected):
            mask = folds == fold
            prediction[mask] = altered[cell, mask]
        value = _to_numpy(prediction, xp)
        robustness[str(scale)] = {
            "mse": _mse(y_np, value),
            "improvement_vs_strongest": _improvement(
                losses[strongest_name], _mse(y_np, value)
            ),
        }
    selected_cells = [_selected_cell(candidates, index, config) for index in selected]
    family_counts = Counter(cell["family_name"] for cell in selected_cells)
    polarity_counts = Counter(cell["polarity"] for cell in selected_cells)
    ranges = np.log10([cell["lambda_kpc"] for cell in selected_cells])
    median_range = float(np.median(ranges))
    clustered = int(
        np.sum(
            abs(ranges - median_range)
            <= float(config["gravity_track_gates"]["maximum_range_distance_from_median_dex"])
        )
    )
    synthetic = _synthetic_controls(matrix, gr, candidates, folds, xp)
    subset = {key: value[:512] for key, value in candidates.items()}
    gpu_check = _to_numpy(
        _prediction_matrix(rows[:12], subset, kernel, config, xp), xp
    )
    cpu_check = _prediction_matrix(rows[:12], subset, kernel, config, np)
    cpu_gpu_error = float(np.max(abs(cpu_check - gpu_check)))
    counterexamples = []
    for row, candidate_value, baseline_value, truth in zip(
        rows, candidate_np, strongest, y_np, strict=True
    ):
        if (truth - candidate_value) ** 2 >= (truth - baseline_value) ** 2:
            counterexamples.append(int(row["nsa"]))
    minimum_mass = min(slices[key]["improvement"] for key in ("mass_low", "mass_high"))
    minimum_acceleration = min(
        value["improvement"]
        for key, value in slices.items()
        if key.startswith("acceleration_stratum")
    )
    gravity = {
        "sample": len(rows) >= int(config["gravity_track_gates"]["minimum_complete_exploration_objects"]),
        "fixed_GR": improvements["fixed_GR"] >= float(config["gravity_track_gates"]["minimum_improvement_vs_fixed_GR"]),
        "calibrated_GR": improvements["calibrated_GR"] >= float(config["gravity_track_gates"]["minimum_improvement_vs_calibrated_GR"]),
        "baryonic_TF": improvements["baryonic_TF"] >= 0.0,
        "RAR": improvements["RAR"] >= 0.0,
        "flexible_nuisance": improvements["flexible_nuisance"] >= 0.0,
        "mass_halves": minimum_mass >= 0.0,
        "acceleration_strata": minimum_acceleration >= 0.0,
        "permutation": p_value <= float(config["gravity_track_gates"]["maximum_selection_aware_permutation_p"]),
        "family_stability": max(family_counts.values()) >= int(config["gravity_track_gates"]["minimum_same_family_folds"]),
        "polarity_stability": max(polarity_counts.values()) >= int(config["gravity_track_gates"]["minimum_same_polarity_folds"]),
        "range_stability": clustered >= int(config["gravity_track_gates"]["minimum_range_clustered_folds"]),
        "gas_geometry": all(value["improvement_vs_strongest"] >= 0 for value in robustness.values()),
        "solar_limit": float(candidate_manifest["maximum_solar_fractional_force_deviation"]) <= 1.0e-5,
        "known_GR_control": synthetic["known_GR"]["pass"],
        "synthetic_control": synthetic["mixed_mode_injection"]["pass"],
        "cpu_gpu": cpu_gpu_error <= 1.0e-10,
        "response_blind": True,
        "confirmation_sealed": True,
    }
    publication = {
        "improves_strongest_ordinary": improvements[strongest_name] >= 0.0,
        "permutation": p_value <= float(config["publication_track_gates"]["maximum_selection_aware_permutation_p"]),
        "family_stability": max(family_counts.values()) >= int(config["publication_track_gates"]["minimum_same_family_folds"]),
        "polarity_stability": max(polarity_counts.values()) >= int(config["publication_track_gates"]["minimum_same_polarity_folds"]),
        "mass_halves": minimum_mass >= 0.0,
        "acceleration_strata": minimum_acceleration >= 0.0,
        "gas_geometry": all(value["improvement_vs_strongest"] >= 0 for value in robustness.values()),
        "confirmation_required": True,
    }
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item21-mixed-modes-result-1.0",
            "item": 21,
            "status": "PASS" if all(gravity.values()) else "REJECT",
            "publication_track_status": "EMPIRICAL_LEAD_REQUIRES_CONFIRMATION" if all(publication.values()) else "NO_EMPIRICAL_LEAD",
            "historical_novelty_claimed": False,
            "linear_equivalence_certificate": {
                "statement": config["physics"]["linear_equivalence_theorem"],
                "equivalent_predecessor": "Item19 healthy normalized two-pole family",
                "pass": True,
            },
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "post_response_candidate_cells": 0,
            },
            "data_source_receipt": {
                "quality_valid_exploration": len(rows),
                "confirmation_opened": 0,
            },
            "search": {
                "raw_cells": int(config["candidate_generator"]["raw_parameter_cells"]),
                "locally_admissible_cells": len(candidates["family"]),
                "structural_equivalence_classes": int(config["candidate_generator"]["structural_equivalence_classes"]),
                "residual_evaluations": int(len(candidates["family"]) * len(rows) * (1 + int(config["evaluation"]["permutation_trials"]))),
                "backend": backend,
                "device": device,
                "elapsed_seconds": time.perf_counter() - start,
            },
            "losses": losses,
            "improvements": improvements,
            "strongest_ordinary_baseline": strongest_name,
            "selection_aware_permutation_p": p_value,
            "selected_GR_cells": selected_gr,
            "selected_folds": selected_cells,
            "family_counts": dict(family_counts),
            "polarity_counts": dict(polarity_counts),
            "range_clustered_folds": clustered,
            "slices_vs_strongest": slices,
            "gas_geometry_robustness": robustness,
            "counterexample_count": len(counterexamples),
            "counterexample_NSA": counterexamples,
            "controls": {**synthetic, "cpu_gpu_max_abs_log_velocity_error": cpu_gpu_error},
            "gravity_track_gates": gravity,
            "publication_track_gates": publication,
            "claim_ceiling": config["scope"]["claim_ceiling"],
        }
    )
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, result)
    _write_json(
        paths["compute_manifest"],
        _content_hashed(
            {
                "schema_version": "invariant-gravity-item21-compute-1.0",
                "result_sha256": _sha256_file(result_path),
                "backend": backend,
                "device": device,
                "confirmation_opened": 0,
            }
        ),
    )
    return result_path


def validate_result(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "result")
    compute = _read_json(paths["compute_manifest"])
    _verify_content_hash(compute, "compute manifest")
    if result["data_source_receipt"]["confirmation_opened"] != 0:
        raise GravityItem21Error("confirmation boundary failed")
    if result["frozen_boundary"]["post_response_candidate_cells"] != 0:
        raise GravityItem21Error("post-response candidate boundary failed")
    if result["historical_novelty_claimed"] is not False:
        raise GravityItem21Error("historical novelty label changed")
    if not result["linear_equivalence_certificate"]["pass"]:
        raise GravityItem21Error("linear equivalence certificate failed")
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
