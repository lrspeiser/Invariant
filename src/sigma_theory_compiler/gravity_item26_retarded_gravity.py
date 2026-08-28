"""Frozen Item 26 causal finite-propagation search on fresh HRS rotation curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    GravityItem16Error,
    _parse_vizier_tsv,
)
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
    _query_url,
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

CONFIG_PATH = Path("configs/gravity_item26_retarded_gravity_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item26_retarded_gravity.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem26Error(RuntimeError):
    """Raised when an Item 26 freeze, leakage, or replay invariant is violated."""


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem26Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem26Error(f"{label} is not an ancestor of HEAD")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if (
        config.get("schema_version")
        != "invariant-gravity-item26-retarded-gravity-config-1.0"
        or int(config.get("item", -1)) != 26
    ):
        raise GravityItem26Error("unexpected Item 26 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem26Error("stable gravity goal changed")
    if int(config["candidate_generator"]["raw_candidate_cells"]) != 262144:
        raise GravityItem26Error("raw candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem26Error("post-response candidates entered Item 26")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem26Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem26Error("paid calls are outside Item 26")
    policy = config["discovery_policy"]
    if not bool(policy["equal_initial_viability"]):
        raise GravityItem26Error("equal-viability policy changed")
    if not bool(policy["age_or_history_is_not_privileged"]):
        raise GravityItem26Error("age or history was privileged")
    for relative, digest in config["dependency_sha256"].items():
        if _sha256_file(root / str(relative)) != str(digest):
            raise GravityItem26Error(f"scientific dependency changed: {relative}")
    return config


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["sample_freeze_commit"] = "<BOUND_COMMIT>"
    value.pop("implementation_correction_commit", None)
    value.pop("implementation_correction_scope", None)
    value.pop("response_access_incident", None)
    return _sha256_bytes(_canonical_bytes(value))


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen) != _contract_digest(config):
        raise GravityItem26Error("scientific contract differs from frozen commit")
    module_commit = str(config.get("implementation_correction_commit", commit))
    _require_ancestor(root, module_commit, "implementation correction")
    module = _git(
        root, "show", f"{module_commit}:{MODULE_PATH.as_posix()}", text_mode=False
    )
    if not isinstance(module, bytes) or _sha256_bytes(module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem26Error("Item 26 module differs from scientific freeze")


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
    for key in (
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        repo_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repo_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem26Error(f"{key} differs from sample freeze")


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["raw_candidate_cells"])
    per = int(config["discovery_policy"]["equal_raw_capacity_per_mechanism"])
    if count != 4 * per:
        raise GravityItem26Error("mechanism capacity is not equal")
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    arrays: dict[str, np.ndarray] = {
        "niche": np.repeat(np.arange(4, dtype=np.int8), per)
    }
    for key, values in (
        ("amplitude", generator["amplitudes"]),
        ("polarity", generator["polarities"]),
        ("radial_transition", generator["radial_transition_reff"]),
        ("radial_power", generator["radial_powers"]),
        ("compensation_power", generator["compensation_powers"]),
        ("speed_fraction", generator["propagation_speed_fractions_c"]),
        ("propagation_transition", generator["propagation_transition_light_years"]),
        ("echo_path", generator["echo_path_multipliers"]),
        ("echo_weight", generator["echo_weights"]),
    ):
        arrays[key] = random.integers(0, len(values), count, dtype=np.int16)
    return arrays


def _candidate_values(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    begin: int,
    end: int,
    xp: Any,
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    result = {"niche": xp.asarray(arrays["niche"][begin:end])}
    for array_key, config_key in (
        ("amplitude", "amplitudes"),
        ("polarity", "polarities"),
        ("radial_transition", "radial_transition_reff"),
        ("radial_power", "radial_powers"),
        ("compensation_power", "compensation_powers"),
        ("speed_fraction", "propagation_speed_fractions_c"),
        ("propagation_transition", "propagation_transition_light_years"),
        ("echo_path", "echo_path_multipliers"),
        ("echo_weight", "echo_weights"),
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


def _log_mu(
    values: Mapping[str, Any],
    specific_growth_per_year: Any,
    light_time_years: Any,
    velocity_fraction_c: Any,
    radius_reff: Any,
    xp: Any,
) -> Any:
    growth = xp.asarray(specific_growth_per_year, dtype=xp.float64)[None, :]
    light_time = xp.asarray(light_time_years, dtype=xp.float64)[None, :]
    velocity_fraction = xp.asarray(velocity_fraction_c, dtype=xp.float64)[None, :]
    radius = xp.asarray(radius_reff, dtype=xp.float64)[None, :]
    niche = values["niche"][:, None]
    signed_amplitude = values["polarity"][:, None] * values["amplitude"][:, None]
    screen = radius ** values["radial_power"][:, None] / (
        radius ** values["radial_power"][:, None]
        + values["radial_transition"][:, None] ** values["radial_power"][:, None]
    )
    direct = growth * light_time
    luminal = direct
    compensated = direct * velocity_fraction ** values["compensation_power"][:, None]
    transition = values["propagation_transition"][:, None]
    beta_effective = values["speed_fraction"][:, None] + (
        1.0 - values["speed_fraction"][:, None]
    ) / (
        1.0
        + (light_time / transition) ** values["radial_power"][:, None]
    )
    finite = direct / beta_effective
    echo = direct * (
        1.0
        + values["echo_weight"][:, None]
        * (values["echo_path"][:, None] - 1.0)
    )
    kernel = xp.where(
        niche == 0,
        luminal,
        xp.where(niche == 1, compensated, xp.where(niche == 2, finite, echo)),
    )
    return signed_amplitude * screen * kernel


def _universal_speed_eligible(values: Mapping[str, Any], xp: Any) -> Any:
    niche = values["niche"]
    return (niche != 2) | (xp.abs(values["speed_fraction"] - 1.0) <= 1e-15)


def _admissible_candidates(
    config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    raw = generate_raw_candidates(config)
    physics = config["physics"]
    batch = int(config["evaluation"]["candidate_batch_size"])
    keep = np.zeros(len(raw["niche"]), dtype=bool)
    universal = np.zeros(len(raw["niche"]), dtype=bool)
    local_deviation = np.full(len(raw["niche"]), np.nan)
    domain_min = np.full(len(raw["niche"]), np.nan)
    domain_max = np.full(len(raw["niche"]), np.nan)
    growth_grid, light_grid, velocity_grid, radius_grid = np.meshgrid(
        np.logspace(-12.0, -8.0, 5),
        np.logspace(2.0, 5.0, 4),
        np.logspace(-4.0, -2.5, 3),
        np.asarray([0.2, 0.8, 1.3, 1.8, 3.0]),
        indexing="ij",
    )
    for begin in range(0, len(raw["niche"]), batch):
        end = min(begin + batch, len(raw["niche"]))
        values = _candidate_values(config, raw, begin, end, np)
        local_log = _log_mu(
            values,
            np.asarray([float(physics["solar_fractional_mass_change_per_year"])]),
            np.asarray([float(physics["AU_light_time_years"])]),
            np.asarray([29.78 / float(physics["c_km_s"])]),
            np.asarray([1.0]),
            np,
        )[:, 0]
        log_domain = _log_mu(
            values,
            growth_grid.ravel(),
            light_grid.ravel(),
            velocity_grid.ravel(),
            radius_grid.ravel(),
            np,
        )
        mu_domain = np.exp(np.clip(log_domain, -100.0, 100.0))
        local_ok = np.abs(np.expm1(local_log)) <= float(
            physics["maximum_local_fractional_response"]
        )
        domain_ok = np.all(np.isfinite(log_domain), axis=1) & np.all(
            (mu_domain >= float(physics["minimum_mu_on_domain"]))
            & (mu_domain <= float(physics["maximum_mu_on_domain"])),
            axis=1,
        )
        causal_ok = (
            (values["speed_fraction"] > 0.0)
            & (values["speed_fraction"] <= 1.0)
            & (values["echo_path"] >= 1.0)
            & (values["echo_weight"] >= 0.0)
        )
        keep[begin:end] = local_ok & domain_ok & causal_ok
        universal[begin:end] = np.asarray(_universal_speed_eligible(values, np), dtype=bool)
        local_deviation[begin:end] = np.abs(np.expm1(local_log))
        domain_min[begin:end] = np.min(mu_domain, axis=1)
        domain_max[begin:end] = np.max(mu_domain, axis=1)
    arrays = {key: value[keep] for key, value in raw.items()}
    universal_kept = universal[keep]
    counts = Counter(int(value) for value in arrays["niche"])
    universal_counts = Counter(
        int(value) for value in arrays["niche"][universal_kept]
    )
    return arrays, {
        "raw_cells": len(raw["niche"]),
        "raw_niche_counts": {
            str(index): int(np.count_nonzero(raw["niche"] == index)) for index in range(4)
        },
        "admissible_cells": len(arrays["niche"]),
        "admissible_niche_counts": {
            str(index): counts.get(index, 0) for index in range(4)
        },
        "universal_speed_eligible_cells": int(np.count_nonzero(universal_kept)),
        "universal_speed_eligible_niche_counts": {
            str(index): universal_counts.get(index, 0) for index in range(4)
        },
        "raw_candidate_digest": _raw_candidate_digest(raw),
        "admissible_candidate_digest": _raw_candidate_digest(arrays),
        "maximum_admitted_local_fractional_response": float(
            np.max(local_deviation[keep])
        ),
        "admitted_domain_mu_range": [
            float(np.min(domain_min[keep])),
            float(np.max(domain_max[keep])),
        ],
        "advanced_support_cells": 0,
        "superluminal_cells": 0,
        "filters_are_response_independent": True,
    }, universal_kept


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    _, audit, _ = _admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item26-retarded-candidates-1.0",
            "generator": config["candidate_generator"],
            "physics_gates": config["physics"],
            "audit": audit,
            "synthetic_injection_admissible_indices": config["candidate_generator"][
                "synthetic_injection_admissible_indices"
            ],
            "synthetic_injection_rule": config["candidate_generator"][
                "synthetic_injection_rule"
            ],
            "responses_open_when_generated": False,
            "post_response_candidate_cells": 0,
        }
    )


def _normal_identity(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _predecessor_exclusions(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    names: set[str] = set()
    coordinates: list[tuple[float, float, str]] = []
    files = 0
    for path in sorted(root.glob(str(config["sources"]["predecessor_sample_glob"]))):
        if path.parent.name.startswith("item-26-"):
            continue
        try:
            manifest = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        files += 1
        for row in manifest.get("objects", []):
            if not isinstance(row, Mapping):
                continue
            for key in ("name", "other_name", "normalized_identity"):
                if row.get(key):
                    names.add(_normal_identity(str(row[key])))
            identity = row.get("identity")
            if isinstance(identity, str) and not identity.isdigit():
                names.add(_normal_identity(identity))
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
                        coordinates.append(
                            (float(row[ra_key]), float(row[dec_key]), path.parent.name)
                        )
                    except (TypeError, ValueError):
                        pass
    return {"names": names, "coordinates": coordinates, "files": files}


def _float(row: Mapping[str, str], key: str) -> float | None:
    try:
        text = str(row[key]).strip()
        return float(text) if text else None
    except (KeyError, TypeError, ValueError):
        return None


def _predictor_rows(
    e1_rows: Sequence[Mapping[str, str]],
    e3_rows: Sequence[Mapping[str, str]],
    exclusions: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    e3 = {int(row["HRS"]): row for row in e3_rows}
    quality = config["predictor_quality"]
    physics = config["physics"]
    output: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "raw_e1_rows": len(e1_rows),
        "raw_e3_rows": len(e3_rows),
        "joined_rows": 0,
        "name_overlaps": [],
        "coordinate_overlaps": [],
        "quality_failures": {},
        "predecessor_manifest_files": int(exclusions["files"]),
        "predecessor_names": len(exclusions["names"]),
        "predecessor_coordinates": len(exclusions["coordinates"]),
    }
    failure_counts: Counter[str] = Counter()
    for e1 in e1_rows:
        identity = int(e1["HRS"])
        e3_row = e3.get(identity)
        if e3_row is None:
            continue
        audit["joined_rows"] += 1
        alternate_names: list[str] = []
        for prefix, key in (("UGC", "UGC"), ("NGC", "NGC")):
            value = str(e1.get(key, "")).strip()
            if value:
                alternate_names.append(_normal_identity(f"{prefix}{int(value)}"))
        overlap = [value for value in alternate_names if value in exclusions["names"]]
        if overlap:
            audit["name_overlaps"].append({"hrs": identity, "matched": overlap})
            continue
        ra = _float(e1, "_RAJ2000")
        dec = _float(e1, "_DEJ2000")
        if ra is None or dec is None:
            failure_counts["coordinate"] += 1
            continue
        coordinate_matches = [
            (sep, label)
            for previous_ra, previous_dec, label in exclusions["coordinates"]
            if (
                sep := _angular_separation_arcsec(
                    ra, dec, previous_ra, previous_dec
                )
            )
            < float(config["sources"]["predecessor_coordinate_veto_arcsec"])
        ]
        if coordinate_matches:
            nearest = min(coordinate_matches)
            audit["coordinate_overlaps"].append(
                {"hrs": identity, "separation_arcsec": nearest[0], "source": nearest[1]}
            )
            continue
        distance = _float(e1, "D")
        reff = _float(e1, "reff")
        d25 = _float(e1, "D25")
        imag = _float(e1, "imag")
        log_mass = _float(e1, "logM*")
        ebv = _float(e1, "E(B-V)")
        virgo_distance = _float(e1, "VirgoD")
        hi_deficiency = _float(e1, "HIDef")
        ellipticity = _float(e3_row, "eps")
        inclination = _float(e3_row, "imorph")
        position_angle = _float(e3_row, "PAmorph")
        flux = _float(e3_row, "FHa")
        coverage = _float(e3_row, "rRC/reff")
        beams = _float(e3_row, "Nbeams")
        checks = {
            "distance": distance is not None
            and float(quality["minimum_distance_Mpc"])
            <= distance
            <= float(quality["maximum_distance_Mpc"]),
            "effective_radius": reff is not None
            and reff >= float(quality["minimum_effective_radius_kpc"]),
            "optical_radius": d25 is not None
            and reff is not None
            and d25 / reff >= float(quality["minimum_optical_to_effective_radius"]),
            "i_magnitude": imag is not None,
            "stellar_mass": log_mass is not None
            and float(quality["minimum_log_stellar_mass"])
            <= log_mass
            <= float(quality["maximum_log_stellar_mass"]),
            "inclination": inclination is not None
            and float(quality["minimum_morphological_inclination_deg"])
            <= inclination
            <= float(quality["maximum_morphological_inclination_deg"]),
            "position_angle": position_angle is not None,
            "Halpha_flux": flux is not None
            and flux >= float(quality["minimum_Halpha_flux_1e16_W_m2"]),
            "curve_coverage": coverage is not None
            and coverage >= float(quality["minimum_curve_coverage_in_reff"]),
            "spatial_beams": beams is not None
            and beams >= float(quality["minimum_spatial_beams"]),
            "ellipticity": ellipticity is not None,
        }
        failures = [key for key, passed in checks.items() if not passed]
        if failures:
            failure_counts.update(failures)
            continue
        assert distance is not None
        assert reff is not None
        assert d25 is not None
        assert imag is not None
        assert log_mass is not None
        assert inclination is not None
        assert position_angle is not None
        assert flux is not None
        assert coverage is not None
        assert beams is not None
        assert ellipticity is not None
        extinction = 10.0 ** (
            0.4 * float(physics["Halpha_extinction_coefficient"]) * float(ebv or 0.0)
        )
        distance_m = distance * float(physics["meters_per_Mpc"])
        luminosity_erg_s = 4.0 * math.pi * distance_m**2 * flux * 1e-16 * 1e7 * extinction
        sfr = float(physics["Halpha_SFR_Msun_per_year_per_erg_s"]) * luminosity_erg_s
        specific_growth = sfr / 10.0**log_mass
        output.append(
            {
                "hrs": identity,
                "ugc": str(e1.get("UGC", "")).strip(),
                "ngc": str(e1.get("NGC", "")).strip(),
                "ra_deg": ra,
                "dec_deg": dec,
                "distance_Mpc": distance,
                "morphology": str(e1.get("Type", "")).strip(),
                "effective_radius_kpc": reff,
                "D25_kpc": d25,
                "i_magnitude": imag,
                "IRAC1_magnitude": _float(e1, "IRAC1") or float("nan"),
                "log_stellar_mass": log_mass,
                "E_BV": float(ebv or 0.0),
                "Virgo_distance_deg": float(virgo_distance or 0.0),
                "environment": str(e1.get("Memb", "")).strip(),
                "HI_deficiency": float(hi_deficiency or 0.0),
                "HI_deficiency_missing": hi_deficiency is None,
                "ellipticity": ellipticity,
                "inclination_deg": inclination,
                "position_angle_deg": position_angle,
                "Halpha_flux_1e16_W_m2": flux,
                "Halpha_SFR_proxy_Msun_year": sfr,
                "specific_growth_per_year": specific_growth,
                "curve_coverage_reff": coverage,
                "spatial_beams": int(beams),
            }
        )
    audit["quality_failures"] = dict(sorted(failure_counts.items()))
    audit["safe_predictor_eligible"] = len(output)
    return sorted(output, key=lambda row: int(row["hrs"])), audit


def _build_sample(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    sample = config["sample"]
    strata = int(sample["mass_strata"])
    ordered = sorted(rows, key=lambda row: (float(row["log_stellar_mass"]), int(row["hrs"])))
    groups = np.array_split(np.asarray(ordered, dtype=object), strata)
    objects: list[dict[str, Any]] = []
    for stratum, values in enumerate(groups):
        group = [dict(value) for value in values.tolist()]
        ranked = sorted(
            group,
            key=lambda row: _hmac_rank(str(sample["role_key"]), f"hrs:{row['hrs']}"),
        )
        confirmation_count = int(sample["confirmation_per_stratum"])
        confirmations = {int(row["hrs"]) for row in ranked[:confirmation_count]}
        exploration = sorted(
            [row for row in ranked if int(row["hrs"]) not in confirmations],
            key=lambda row: _hmac_rank(str(sample["fold_key"]), f"hrs:{row['hrs']}"),
        )
        fold_by_identity = {
            int(row["hrs"]): int((index + stratum) % int(sample["outer_folds"]))
            for index, row in enumerate(exploration)
        }
        for row in group:
            identity = int(row["hrs"])
            role = "confirmation" if identity in confirmations else "exploration"
            objects.append(
                {
                    "identity": identity,
                    "role": role,
                    "mass_stratum": stratum,
                    "outer_fold": None if role == "confirmation" else fold_by_identity[identity],
                    "ra_deg": float(row["ra_deg"]),
                    "dec_deg": float(row["dec_deg"]),
                    "ugc": row["ugc"],
                    "ngc": row["ngc"],
                    "role_rank_sha256": _hmac_rank(
                        str(sample["role_key"]), f"hrs:{identity}"
                    ),
                }
            )
    role_counts = Counter(str(row["role"]) for row in objects)
    fold_counts = Counter(
        int(row["outer_fold"])
        for row in objects
        if row["role"] == "exploration"
    )
    if len(objects) != int(sample["expected_safe_predictor_eligible"]):
        raise GravityItem26Error(f"selected {len(objects)} predictor rows")
    if role_counts["exploration"] != int(sample["expected_exploration"]):
        raise GravityItem26Error("unexpected exploration count")
    if role_counts["confirmation"] != int(sample["expected_confirmation"]):
        raise GravityItem26Error("unexpected confirmation count")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item26-retarded-sample-1.0",
            "selection_rule": sample["rule"],
            "response_columns_read": [],
            "confirmation_response_values_read": 0,
            "objects": sorted(objects, key=lambda row: int(row["identity"])),
            "role_counts": dict(sorted(role_counts.items())),
            "fold_counts": {str(key): value for key, value in sorted(fold_counts.items())},
        }
    )


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictors"].parent.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    all_rows: list[list[dict[str, str]]] = []
    for source_key, columns_key in (
        ("catalog_e1", "predictor_e1_columns"),
        ("catalog_e3", "predictor_e3_columns"),
    ):
        source = str(config["sources"][source_key])
        columns = config["sources"][columns_key]
        url = _query_url(source, columns)
        body, headers = _download(url)
        rows = _parse_vizier_tsv(body, columns)
        if len(rows) != int(config["sources"]["expected_predictor_rows"]):
            raise GravityItem26Error(f"{source} row count changed: {len(rows)}")
        receipts.append(
            {
                "source": source,
                "url": url,
                "sha256": _sha256_bytes(body),
                "bytes": len(body),
                "etag": headers.get("etag"),
                "last_modified": headers.get("last-modified"),
                "columns": columns,
            }
        )
        all_rows.append(rows)
    exclusions = _predecessor_exclusions(root, config)
    rows, exclusion_audit = _predictor_rows(all_rows[0], all_rows[1], exclusions, config)
    if len(rows) != int(config["sample"]["expected_safe_predictor_eligible"]):
        raise GravityItem26Error(f"safe predictor count changed: {len(rows)}")
    columns = [
        "hrs",
        "ugc",
        "ngc",
        "ra_deg",
        "dec_deg",
        "distance_Mpc",
        "morphology",
        "effective_radius_kpc",
        "D25_kpc",
        "i_magnitude",
        "IRAC1_magnitude",
        "log_stellar_mass",
        "E_BV",
        "Virgo_distance_deg",
        "environment",
        "HI_deficiency",
        "HI_deficiency_missing",
        "ellipticity",
        "inclination_deg",
        "position_angle_deg",
        "Halpha_flux_1e16_W_m2",
        "Halpha_SFR_proxy_Msun_year",
        "specific_growth_per_year",
        "curve_coverage_reff",
        "spatial_beams",
    ]
    _write_tsv(paths["predictors"], rows, columns)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item26-retarded-predictors-1.0",
            "source_receipts": receipts,
            "predictor_columns_queried": [
                *config["sources"]["predictor_e1_columns"],
                *config["sources"]["predictor_e3_columns"],
            ],
            "response_columns_queried": [],
            "exclusion_audit": exclusion_audit,
            "predictor_file": {
                "path": paths["predictors"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["predictors"]),
                "rows": len(rows),
            },
        }
    )
    sample_manifest = _build_sample(rows, config)
    candidate_manifest = _candidate_manifest(config)
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], candidate_manifest)
    return paths


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    paths = _source_paths(root, config)
    sample = _read_json(paths["sample_manifest"])
    _verify_content_hash(sample, "sample manifest")
    identities = [
        int(row["identity"]) for row in sample["objects"] if row["role"] == "exploration"
    ]

    def fetch(identity: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        url = _query_url(
            str(config["sources"]["catalog_rotation"]),
            config["sources"]["response_columns"],
            HRS=identity,
        )
        body, headers = _download(url)
        try:
            raw = _parse_vizier_tsv(body, config["sources"]["response_columns"])
        except GravityItem16Error:
            data_lines = [
                line
                for line in body.decode("utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
            if data_lines:
                raise
            raw = []
        if not raw:
            return [], {
                "identity": identity,
                "url": url,
                "sha256": _sha256_bytes(body),
                "bytes": len(body),
                "rows": 0,
                "empty_exact_query": True,
                "etag": headers.get("etag"),
                "last_modified": headers.get("last-modified"),
            }
        if not raw or any(int(row["HRS"]) != identity for row in raw):
            raise GravityItem26Error(f"response query for HRS {identity} failed")
        rows = [
            {
                "hrs": identity,
                "radius_kpc": float(row["r"]),
                "radius_error_kpc": float(row["s_r"]),
                "velocity_km_s": float(row["v"]),
                "velocity_error_km_s": float(row["s_v"]),
                "bins": int(row["Nbins"]),
                "side": str(row["side"]).strip().lower(),
            }
            for row in raw
        ]
        receipt = {
            "identity": identity,
            "url": url,
            "sha256": _sha256_bytes(body),
            "bytes": len(body),
            "rows": len(rows),
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
        }
        return rows, receipt

    response_rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for rows, receipt in executor.map(fetch, identities):
            response_rows.extend(rows)
            receipts.append(receipt)
    response_rows.sort(key=lambda row: (int(row["hrs"]), str(row["side"]), float(row["radius_kpc"])))
    receipts.sort(key=lambda row: int(row["identity"]))
    _write_tsv(
        paths["exploration_responses"],
        response_rows,
        [
            "hrs",
            "radius_kpc",
            "radius_error_kpc",
            "velocity_km_s",
            "velocity_error_km_s",
            "bins",
            "side",
        ],
    )
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item26-retarded-responses-1.0",
            "response_columns_queried": config["sources"]["response_columns"],
            "query_scope": "one exact VizieR HRS query per frozen exploration identity",
            "exploration_identities_read": len(identities),
            "exploration_curve_rows_read": len(response_rows),
            "confirmation_values_read": 0,
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
            "source_receipts": receipts,
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["response_source_manifest"]


def _interpolate_side(
    rows: Sequence[Mapping[str, Any]], target_radius: float, config: Mapping[str, Any]
) -> tuple[float, float] | None:
    extraction = config["response_extraction"]
    clean = sorted(
        [
            row
            for row in rows
            if float(extraction["minimum_velocity_km_s"])
            <= float(row["velocity_km_s"])
            <= float(extraction["maximum_velocity_km_s"])
            and float(row["velocity_error_km_s"]) >= 0.0
        ],
        key=lambda row: float(row["radius_kpc"]),
    )
    if len(clean) < int(extraction["minimum_side_points"]):
        return None
    radius = np.asarray([float(row["radius_kpc"]) for row in clean])
    if target_radius < float(np.min(radius)) or target_radius > float(np.max(radius)):
        return None
    velocity = np.asarray([float(row["velocity_km_s"]) for row in clean])
    error = np.asarray([float(row["velocity_error_km_s"]) for row in clean])
    value = float(np.interp(target_radius, radius, velocity))
    uncertainty = float(np.interp(target_radius, radius, error))
    if uncertainty / max(value, 1e-12) > float(
        extraction["maximum_interpolated_fractional_error"]
    ):
        return None
    return value, uncertainty


def _load_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictors = {int(row["hrs"]): row for row in _read_tsv(paths["predictors"])}
    response_rows: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in _read_tsv(paths["exploration_responses"]):
        response_rows[int(row["hrs"])][str(row["side"]).lower()].append(row)
    sample = _read_json(paths["sample_manifest"])
    response_manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(sample, "sample manifest")
    _verify_content_hash(response_manifest, "response manifest")
    radii = [
        float(config["response_extraction"]["primary_radius_reff"]),
        *[
            float(value)
            for value in config["response_extraction"]["fixed_replay_radii_reff"]
        ],
    ]
    rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for role in sample["objects"]:
        if role["role"] != "exploration":
            continue
        identity = int(role["identity"])
        predictor = predictors[identity]
        sides = response_rows[identity]
        if "a" not in sides or "r" not in sides:
            failures["both_sides"] += 1
            continue
        speeds: dict[str, float] = {}
        errors: dict[str, float] = {}
        asymmetries: dict[str, float] = {}
        valid = True
        for radius_reff in radii:
            radius_kpc = radius_reff * float(predictor["effective_radius_kpc"])
            approaching = _interpolate_side(sides["a"], radius_kpc, config)
            receding = _interpolate_side(sides["r"], radius_kpc, config)
            if approaching is None or receding is None:
                failures[f"radius_{radius_reff:g}"] += 1
                valid = False
                break
            mean_velocity = 0.5 * (approaching[0] + receding[0])
            asymmetry = abs(approaching[0] - receding[0]) / max(mean_velocity, 1e-12)
            if asymmetry > float(
                config["response_extraction"]["maximum_side_fractional_asymmetry"]
            ):
                failures["side_asymmetry"] += 1
                valid = False
                break
            label = f"r{radius_reff:g}"
            speeds[label] = mean_velocity
            errors[label] = 0.5 * math.hypot(approaching[1], receding[1])
            asymmetries[label] = asymmetry
        if not valid:
            continue
        rows.append(
            {
                "identity": identity,
                "fold": int(role["outer_fold"]),
                "mass_stratum": int(role["mass_stratum"]),
                "distance_Mpc": float(predictor["distance_Mpc"]),
                "effective_radius_kpc": float(predictor["effective_radius_kpc"]),
                "D25_kpc": float(predictor["D25_kpc"]),
                "log_stellar_mass": float(predictor["log_stellar_mass"]),
                "specific_growth_per_year": float(predictor["specific_growth_per_year"]),
                "ellipticity": float(predictor["ellipticity"]),
                "inclination_deg": float(predictor["inclination_deg"]),
                "HI_deficiency": float(predictor["HI_deficiency"]),
                "HI_deficiency_missing": str(predictor["HI_deficiency_missing"]).lower()
                == "true",
                "Virgo_distance_deg": float(predictor["Virgo_distance_deg"]),
                "curve_coverage_reff": float(predictor["curve_coverage_reff"]),
                "spatial_beams": int(predictor["spatial_beams"]),
                "speeds": speeds,
                "speed_errors": errors,
                "side_asymmetries": asymmetries,
            }
        )
    if len(rows) < int(config["sample"]["minimum_valid_exploration"]):
        raise GravityItem26Error(
            f"only {len(rows)} exploration galaxies pass response quality; "
            f"minimum is {config['sample']['minimum_valid_exploration']}"
        )
    quality_audit = {
        "frozen_exploration": int(config["sample"]["expected_exploration"]),
        "valid_exploration": len(rows),
        "failure_counts": dict(sorted(failures.items())),
    }
    return sorted(rows, key=lambda row: int(row["identity"])), response_manifest, quality_audit


def _enclosed_stellar_fraction(radius_reff: float) -> float:
    y = 1.678 * radius_reff
    return 1.0 - (1.0 + y) * math.exp(-y)


def _row_physics(
    rows: Sequence[Mapping[str, Any]], radius_reff: float, config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    physics = config["physics"]
    growth = np.asarray([float(row["specific_growth_per_year"]) for row in rows])
    radius_kpc = radius_reff * np.asarray(
        [float(row["effective_radius_kpc"]) for row in rows]
    )
    light_time_years = (
        radius_kpc
        * float(physics["km_per_kpc"])
        / float(physics["c_km_s"])
        / float(physics["seconds_per_year"])
    )
    enclosed = _enclosed_stellar_fraction(radius_reff)
    mass = 10.0 ** np.asarray([float(row["log_stellar_mass"]) for row in rows])
    baryonic_velocity = np.sqrt(
        float(physics["G_kpc_km2_s2_Msun"]) * mass * enclosed / radius_kpc
    )
    velocity_fraction = baryonic_velocity / float(physics["c_km_s"])
    radius_values = np.full(len(rows), radius_reff)
    return growth, light_time_years, velocity_fraction, radius_values


def _build_term_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    radius_reff: float,
) -> np.ndarray:
    growth, light_time, velocity_fraction, radius_values = _row_physics(
        rows, radius_reff, config
    )
    batch = int(config["evaluation"]["candidate_batch_size"])
    pieces: list[np.ndarray] = []
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        values = _candidate_values(config, arrays, begin, end, np)
        pieces.append(
            0.5
            * _log_mu(
                values,
                growth,
                light_time,
                velocity_fraction,
                radius_values,
                np,
            )
            / math.log(10.0)
        )
    return np.concatenate(pieces, axis=0)


def _base_design(
    rows: Sequence[Mapping[str, Any]], radius_reff: float, config: Mapping[str, Any]
) -> np.ndarray:
    _, _, velocity_fraction, _ = _row_physics(rows, radius_reff, config)
    log_velocity = np.log10(velocity_fraction * float(config["physics"]["c_km_s"]))
    return np.column_stack([np.ones(len(rows)), log_velocity - 2.0])


def _flex_design(
    rows: Sequence[Mapping[str, Any]], radius_reff: float, config: Mapping[str, Any]
) -> np.ndarray:
    base = _base_design(rows, radius_reff, config)[:, 1]
    mass = np.asarray([float(row["log_stellar_mass"]) for row in rows]) - 9.5
    size = np.log10(
        np.asarray([float(row["effective_radius_kpc"]) for row in rows])
    )
    activity = np.log10(
        np.maximum(
            np.asarray([float(row["specific_growth_per_year"]) for row in rows]),
            1e-14,
        )
    ) + 10.0
    ellipticity = np.asarray([float(row["ellipticity"]) for row in rows])
    inclination = np.asarray([float(row["inclination_deg"]) for row in rows]) / 90.0
    hi_deficiency = np.asarray([float(row["HI_deficiency"]) for row in rows])
    hi_missing = np.asarray([float(bool(row["HI_deficiency_missing"])) for row in rows])
    distance = np.asarray([float(row["distance_Mpc"]) for row in rows]) / 20.0
    concentration = np.log10(
        np.asarray([float(row["D25_kpc"]) for row in rows])
        / np.asarray([float(row["effective_radius_kpc"]) for row in rows])
    )
    environment = np.log1p(
        np.asarray([float(row["Virgo_distance_deg"]) for row in rows])
    )
    return np.column_stack(
        [
            base,
            mass,
            size,
            activity,
            ellipticity,
            inclination,
            hi_deficiency,
            hi_missing,
            distance,
            concentration,
            environment,
            mass * activity,
            activity * ellipticity,
            mass * concentration,
        ]
    )


def _target(rows: Sequence[Mapping[str, Any]], radius_reff: float) -> np.ndarray:
    label = f"r{radius_reff:g}"
    return np.log10(np.asarray([float(row["speeds"][label]) for row in rows]))


def _oof_search(
    xp: Any,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    folds: np.ndarray,
    term_matrix: np.ndarray,
    radius_reff: float,
) -> dict[str, Any]:
    base = _base_design(rows, radius_reff, config)
    flexible = _flex_design(rows, radius_reff, config)
    prediction_candidate = np.full(len(rows), np.nan)
    prediction_base = np.full(len(rows), np.nan)
    prediction_flexible = np.full(len(rows), np.nan)
    selected: list[int] = []
    residual_evaluations = 0
    for outer in sorted({int(value) for value in folds}):
        test = np.where(folds == outer)[0]
        train = np.where(folds != outer)[0]
        selected_index, evaluations = _select_candidate(
            xp, config, base, target, folds, outer, term_matrix
        )
        selected.append(selected_index)
        residual_evaluations += evaluations
        terms = xp.asarray(term_matrix[selected_index : selected_index + 1])
        candidate = _fit_candidate_predictions(xp, base, target, terms, train, test)
        prediction_candidate[test] = _to_numpy(candidate[0], xp)
        residual_evaluations += len(test)
        prediction_base[test] = _linear_predict(base, target, train, test)
        prediction_flexible[test] = _ridge_predict(
            flexible,
            target,
            train,
            test,
            float(config["evaluation"]["ridge_alpha"]),
        )
    return {
        "candidate": prediction_candidate,
        "base": prediction_base,
        "flexible": prediction_flexible,
        "selected": selected,
        "residual_evaluations": residual_evaluations,
    }


def _fixed_oof(
    xp: Any,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    folds: np.ndarray,
    term_matrix: np.ndarray,
    selected: Sequence[int],
    radius_reff: float,
) -> dict[str, np.ndarray]:
    base = _base_design(rows, radius_reff, config)
    flexible = _flex_design(rows, radius_reff, config)
    candidate_prediction = np.full(len(rows), np.nan)
    base_prediction = np.full(len(rows), np.nan)
    flexible_prediction = np.full(len(rows), np.nan)
    for selected_index, outer in zip(selected, sorted({int(value) for value in folds}), strict=True):
        test = np.where(folds == outer)[0]
        train = np.where(folds != outer)[0]
        terms = xp.asarray(term_matrix[selected_index : selected_index + 1])
        prediction = _fit_candidate_predictions(xp, base, target, terms, train, test)
        candidate_prediction[test] = _to_numpy(prediction[0], xp)
        base_prediction[test] = _linear_predict(base, target, train, test)
        flexible_prediction[test] = _ridge_predict(
            flexible,
            target,
            train,
            test,
            float(config["evaluation"]["ridge_alpha"]),
        )
    return {
        "candidate": candidate_prediction,
        "base": base_prediction,
        "flexible": flexible_prediction,
    }


def _asymmetry_replay(
    rows: Sequence[Mapping[str, Any]],
    folds: np.ndarray,
    term_matrix: np.ndarray,
    selected: Sequence[int],
    radius_reff: float,
) -> dict[str, float]:
    label = f"r{radius_reff:g}"
    target = np.log10(
        np.asarray([float(row["side_asymmetries"][label]) for row in rows]) + 0.02
    )
    candidate_prediction = np.full(len(rows), np.nan)
    intercept_prediction = np.full(len(rows), np.nan)
    for selected_index, outer in zip(selected, sorted({int(value) for value in folds}), strict=True):
        train = np.where(folds != outer)[0]
        test = np.where(folds == outer)[0]
        feature = np.abs(term_matrix[selected_index])
        design = np.column_stack([np.ones(len(rows)), feature])
        candidate_prediction[test] = _linear_predict(design, target, train, test)
        intercept_prediction[test] = np.mean(target[train])
    candidate_mse = _mse(target, candidate_prediction)
    intercept_mse = _mse(target, intercept_prediction)
    return {
        "candidate_mse": candidate_mse,
        "intercept_mse": intercept_mse,
        "improvement_vs_intercept": _improvement(intercept_mse, candidate_mse),
    }


def _candidate_record(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    universal: np.ndarray,
    index: int,
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(values["niche"][0])
    result: dict[str, Any] = {
        "index": index,
        "niche": niche,
        "niche_id": config["candidate_generator"]["niches"][niche]["id"],
        "universal_speed_eligible": bool(universal[index]),
    }
    for key in (
        "amplitude",
        "polarity",
        "radial_transition",
        "radial_power",
        "compensation_power",
        "speed_fraction",
        "propagation_transition",
        "echo_path",
        "echo_weight",
    ):
        result[key] = float(values[key][0])
    return result


def _evaluate(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    xp, backend, device = _backend()
    started = time.perf_counter()
    arrays, candidate_audit, universal = _admissible_candidates(config)
    primary_radius = float(config["response_extraction"]["primary_radius_reff"])
    replay_radii = [
        float(value) for value in config["response_extraction"]["fixed_replay_radii_reff"]
    ]
    term_matrices = {
        radius: _build_term_matrix(config, arrays, rows, radius)
        for radius in [primary_radius, *replay_radii]
    }
    folds = np.asarray([int(row["fold"]) for row in rows])
    target = _target(rows, primary_radius)
    observed = _oof_search(
        xp,
        config,
        rows,
        target,
        folds,
        term_matrices[primary_radius],
        primary_radius,
    )
    residual_evaluations = int(observed["residual_evaluations"])
    candidate_mse = _mse(target, observed["candidate"])
    base_mse = _mse(target, observed["base"])
    flexible_mse = _mse(target, observed["flexible"])
    observed_improvement = _improvement(base_mse, candidate_mse)

    base_full = _base_design(rows, primary_radius, config)
    base_coefficient = np.linalg.lstsq(base_full, target, rcond=None)[0]
    base_full_prediction = base_full @ base_coefficient
    residual = target - base_full_prediction
    random = np.random.Generator(
        np.random.PCG64(int(config["evaluation"]["permutation_seed"]))
    )
    strata = np.asarray([int(row["mass_stratum"]) for row in rows])
    null_improvements: list[float] = []
    for trial in range(int(config["evaluation"]["permutation_trials"])):
        permuted = residual.copy()
        for stratum in sorted({int(value) for value in strata}):
            indices = np.where(strata == stratum)[0]
            permuted[indices] = random.permutation(permuted[indices])
        null_target = base_full_prediction + permuted
        null = _oof_search(
            xp,
            config,
            rows,
            null_target,
            folds,
            term_matrices[primary_radius],
            primary_radius,
        )
        null_candidate_mse = _mse(null_target, null["candidate"])
        null_base_mse = _mse(null_target, null["base"])
        null_improvements.append(_improvement(null_base_mse, null_candidate_mse))
        residual_evaluations += int(null["residual_evaluations"])
        if (trial + 1) % 10 == 0:
            print(
                f"Item 26 selection-aware nulls {trial + 1}/"
                f"{config['evaluation']['permutation_trials']}",
                flush=True,
            )
    permutation_p = (
        1.0 + sum(value >= observed_improvement for value in null_improvements)
    ) / (len(null_improvements) + 1.0)

    synthetic: list[dict[str, Any]] = []
    frozen_injections = [
        int(value)
        for value in config["candidate_generator"][
            "synthetic_injection_admissible_indices"
        ]
    ]
    if len(frozen_injections) != 4:
        raise GravityItem26Error("synthetic injection list changed")
    for niche, injection_index in enumerate(frozen_injections):
        if not 0 <= injection_index < len(arrays["niche"]):
            raise GravityItem26Error("synthetic injection is outside admissible cells")
        if int(arrays["niche"][injection_index]) != niche:
            raise GravityItem26Error("synthetic injection niche changed")
        injected_target = base_full_prediction + term_matrices[primary_radius][injection_index]
        replay = _oof_search(
            xp,
            config,
            rows,
            injected_target,
            folds,
            term_matrices[primary_radius],
            primary_radius,
        )
        selected_niches = [int(arrays["niche"][index]) for index in replay["selected"]]
        niche_count = int(np.count_nonzero(np.asarray(selected_niches) == niche))
        synthetic.append(
            {
                "injected_niche": niche,
                "injected_index": injection_index,
                "selected_niches": selected_niches,
                "selected_niche_folds": niche_count,
                "pass": niche_count >= int(config["gates"]["minimum_same_niche_folds"]),
            }
        )
        residual_evaluations += int(replay["residual_evaluations"])

    constant_target = base_full_prediction
    constant = _oof_search(
        xp,
        config,
        rows,
        constant_target,
        folds,
        term_matrices[primary_radius],
        primary_radius,
    )
    constant_candidate_mse = _mse(constant_target, constant["candidate"])
    constant_base_mse = _mse(constant_target, constant["base"])
    constant_improvement = _improvement(constant_base_mse, constant_candidate_mse)
    constant_pass = constant_improvement <= float(
        config["gates"]["known_instantaneous_control_maximum_material_improvement"]
    )
    residual_evaluations += int(constant["residual_evaluations"])

    radial_replays: dict[str, Any] = {}
    for radius in replay_radii:
        radius_target = _target(rows, radius)
        predictions = _fixed_oof(
            xp,
            config,
            rows,
            radius_target,
            folds,
            term_matrices[radius],
            observed["selected"],
            radius,
        )
        candidate_value = _mse(radius_target, predictions["candidate"])
        base_value = _mse(radius_target, predictions["base"])
        flexible_value = _mse(radius_target, predictions["flexible"])
        radial_replays[f"r{radius:g}"] = {
            "candidate_mse": candidate_value,
            "instantaneous_baryonic_mse": base_value,
            "flexible_nuisance_mse": flexible_value,
            "improvement_vs_instantaneous_baryonic": _improvement(
                base_value, candidate_value
            ),
            "improvement_vs_flexible_nuisance": _improvement(
                flexible_value, candidate_value
            ),
        }

    asymmetry = _asymmetry_replay(
        rows,
        folds,
        term_matrices[primary_radius],
        observed["selected"],
        primary_radius,
    )
    mass = np.asarray([float(row["log_stellar_mass"]) for row in rows])
    activity = np.asarray([float(row["specific_growth_per_year"]) for row in rows])
    slices = {
        "low_stellar_mass": np.where(mass <= np.median(mass))[0],
        "high_stellar_mass": np.where(mass > np.median(mass))[0],
        "low_Halpha_activity": np.where(activity <= np.median(activity))[0],
        "high_Halpha_activity": np.where(activity > np.median(activity))[0],
    }
    slice_metrics: dict[str, Any] = {}
    for label, indices in slices.items():
        candidate_value = _mse(target, observed["candidate"], indices)
        base_value = _mse(target, observed["base"], indices)
        flexible_value = _mse(target, observed["flexible"], indices)
        slice_metrics[label] = {
            "objects": len(indices),
            "candidate_mse": candidate_value,
            "instantaneous_baryonic_mse": base_value,
            "flexible_nuisance_mse": flexible_value,
            "improvement_vs_instantaneous_baryonic": _improvement(
                base_value, candidate_value
            ),
            "improvement_vs_flexible_nuisance": _improvement(
                flexible_value, candidate_value
            ),
        }
    selected_records = [
        _candidate_record(config, arrays, universal, int(index))
        for index in observed["selected"]
    ]
    selected_niches = [int(record["niche"]) for record in selected_records]
    niche_counts = Counter(selected_niches)
    same_niche_folds = max(niche_counts.values())
    all_speed_eligible = all(bool(record["universal_speed_eligible"]) for record in selected_records)
    counterexamples = int(
        np.count_nonzero(
            (target - observed["candidate"]) ** 2
            > (target - observed["flexible"]) ** 2
        )
    )
    mass_halves_pass = all(
        slice_metrics[label]["improvement_vs_instantaneous_baryonic"]
        >= float(config["gates"]["minimum_each_mass_half_improvement_vs_instantaneous"])
        for label in ("low_stellar_mass", "high_stellar_mass")
    )
    activity_halves_pass = all(
        slice_metrics[label]["improvement_vs_instantaneous_baryonic"]
        >= float(config["gates"]["minimum_each_activity_half_improvement_vs_instantaneous"])
        for label in ("low_Halpha_activity", "high_Halpha_activity")
    )
    radial_pass = all(
        value["improvement_vs_instantaneous_baryonic"]
        >= float(config["gates"]["minimum_each_fixed_radius_improvement_vs_instantaneous"])
        for value in radial_replays.values()
    )
    universal_pass = all(
        [
            observed_improvement
            >= float(config["gates"]["minimum_improvement_vs_instantaneous_baryonic"]),
            _improvement(flexible_mse, candidate_mse)
            >= float(config["gates"]["minimum_improvement_vs_flexible_nuisance"]),
            mass_halves_pass,
            activity_halves_pass,
            radial_pass,
            permutation_p <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            same_niche_folds >= int(config["gates"]["minimum_same_niche_folds"]),
            all_speed_eligible,
            all(value["pass"] for value in synthetic),
            constant_pass,
        ]
    )
    phenomenon_pass = all(
        [
            _improvement(flexible_mse, candidate_mse)
            >= float(config["gates"]["phenomenon_minimum_improvement_vs_flexible"]),
            permutation_p <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
            same_niche_folds >= int(config["gates"]["minimum_same_niche_folds"]),
            asymmetry["improvement_vs_intercept"] >= 0.0,
        ]
    )
    cpu_terms = term_matrices[primary_radius][np.asarray(observed["selected"])]
    gpu_terms = _to_numpy(xp.asarray(cpu_terms), xp)
    cpu_gpu_max = float(np.max(np.abs(cpu_terms - gpu_terms)))
    elapsed = time.perf_counter() - started
    scientific = {
        "valid_objects": len(rows),
        "candidate_audit": candidate_audit,
        "metrics": {
            "candidate_mse": candidate_mse,
            "instantaneous_baryonic_mse": base_mse,
            "flexible_nuisance_mse": flexible_mse,
            "improvement_vs_instantaneous_baryonic": observed_improvement,
            "improvement_vs_flexible_nuisance": _improvement(flexible_mse, candidate_mse),
            "selection_aware_permutation_p": permutation_p,
            "null_improvement_minimum": float(np.min(null_improvements)),
            "null_improvement_median": float(np.median(null_improvements)),
            "null_improvement_maximum": float(np.max(null_improvements)),
            "individual_counterexamples_vs_flexible": counterexamples,
        },
        "fixed_radial_replays": radial_replays,
        "side_asymmetry_replay": asymmetry,
        "slice_metrics": slice_metrics,
        "selected_folds": selected_records,
        "selected_niche_counts": {str(key): value for key, value in sorted(niche_counts.items())},
        "same_niche_folds": same_niche_folds,
        "all_selected_folds_universal_speed_eligible": all_speed_eligible,
        "controls": {
            "synthetic_niche_recovery": synthetic,
            "synthetic_all_pass": all(value["pass"] for value in synthetic),
            "instantaneous_control_improvement": constant_improvement,
            "instantaneous_control_pass": constant_pass,
            "cpu_gpu_max_absolute_difference": cpu_gpu_max,
            "cpu_gpu_pass": cpu_gpu_max <= 1e-12,
        },
        "universal_gravity_track_pass": universal_pass,
        "phenomenon_publication_track_pass": phenomenon_pass,
        "paper_claim_allowed": False,
        "formal_status": (
            "PASS_EXPLORATION_BOTH_TRACKS"
            if universal_pass and phenomenon_pass
            else "PASS_EXPLORATION_UNIVERSAL_ONLY"
            if universal_pass
            else "PASS_EXPLORATION_PHENOMENON_LEAD"
            if phenomenon_pass
            else "SCOPED_REJECT_BOTH_TRACKS"
        ),
    }
    compute = {
        "schema_version": "invariant-gravity-item26-retarded-compute-1.0",
        "backend": backend,
        "device": device,
        "admissible_candidates": len(arrays["niche"]),
        "training_residual_evaluations": residual_evaluations,
        "permutation_trials": len(null_improvements),
        "synthetic_full_searches": 4,
        "instantaneous_control_full_searches": 1,
        "wall_seconds": elapsed,
        "paid_model_calls": 0,
        "paid_api_spend_usd": 0.0,
    }
    return scientific, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    response_manifest: Mapping[str, Any],
    quality_audit: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item26-retarded-result-1.0",
            "item": 26,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "discovery_policy": config["discovery_policy"],
            "theory_and_equivalence_audit": config["theory"],
            "observable_lineage": config["sources"]["observable_lineage"],
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "stable_goal_sha256": config["stable_goal_sha256"],
                "implementation_correction_commit": config[
                    "implementation_correction_commit"
                ],
                "implementation_correction_scope": config[
                    "implementation_correction_scope"
                ],
                "response_access_incident": config["response_access_incident"],
                "confirmation_opened": False,
                "confirmation_response_values_read": int(
                    response_manifest["confirmation_values_read"]
                ),
                "post_response_formula_generation": False,
                "advanced_support_used": False,
            },
            "sample": {
                "quality_audit": quality_audit,
                "valid_exploration_identities": [int(row["identity"]) for row in rows],
                "confirmation_identities_remain_sealed": int(
                    config["sample"]["expected_confirmation"]
                ),
            },
            "baselines": {
                "instantaneous_baryonic": config["evaluation"]["baseline_instantaneous_baryonic"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
            },
            "scientific_result": scientific,
            "compute_and_api_cost": compute,
            "counterexamples_and_limitations": config["theory"]["claim_limits"],
            "exact_next_action": "Preserve every Item 26 branch under the equal-viability two-track policy, independently replicate any phenomenon lead without retuning, and advance the numbered roadmap to Item 27 gravitational memory.",
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "predictor_manifest_path": paths["predictor_source_manifest"]
                .relative_to(root)
                .as_posix(),
                "response_manifest_path": paths["response_source_manifest"]
                .relative_to(root)
                .as_posix(),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
            },
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest, quality_audit = _load_rows(root, config)
    scientific, compute_raw = _evaluate(config, rows)
    paths = _source_paths(root, config)
    compute = _content_hashed(compute_raw)
    _write_json(paths["compute_manifest"], compute)
    result = _build_receipt(
        root,
        config,
        rows,
        response_manifest,
        quality_audit,
        scientific,
        compute,
    )
    result_path = root / str(config["paths"]["result"])
    _write_json(result_path, result)
    return result_path


def validate_result(root: Path) -> Path:
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
        raise GravityItem26Error("result opened confirmation data")
    if bool(result["frozen_boundary"]["advanced_support_used"]):
        raise GravityItem26Error("result used advanced support")
    if bool(result["scientific_result"]["paper_claim_allowed"]):
        raise GravityItem26Error("exploration result made a paper claim")
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
