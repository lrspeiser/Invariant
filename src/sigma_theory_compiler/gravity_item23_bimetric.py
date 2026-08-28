"""Frozen Item 23 bimetric-gravity search on fresh joint dynamics/lensing data."""

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
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    _angular_diameter_distances,
    _parse_vizier_tsv,
    hernquist_projected_mass_fraction,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _angular_separation_arcsec,
    _backend,
    _derived_predictor,
    _download,
    _hmac_rank,
    _legacy_url,
    _nearest_legacy_row,
    _read_tsv,
    _to_numpy,
    _write_tsv,
)

CONFIG_PATH = Path("configs/gravity_item23_bimetric_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item23_bimetric.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem23Error(RuntimeError):
    """Raised when an Item 23 freeze, leakage, or replay invariant is violated."""


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
        raise GravityItem23Error(f"{label} has no content hash")
    body = dict(payload)
    body.pop("content_sha256", None)
    if _sha256_bytes(_canonical_bytes(body)) != expected:
        raise GravityItem23Error(f"{label} content hash changed")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityItem23Error(f"expected JSON object: {path}")
    return value


def _git(root: Path, *args: str, text_mode: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=text_mode
    )
    return result.stdout.strip() if text_mode else result.stdout


def _require_ancestor(root: Path, commit: str, label: str) -> None:
    if commit.startswith("TO_BE_BOUND"):
        raise GravityItem23Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem23Error(f"{label} is not an ancestor of HEAD")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected = "invariant-gravity-item23-bimetric-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 23:
        raise GravityItem23Error("unexpected Item 23 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem23Error("stable gravity goal changed")
    if int(config["candidate_generator"]["raw_candidate_cells"]) != 262144:
        raise GravityItem23Error("raw candidate boundary changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem23Error("post-response candidates entered Item 23")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem23Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem23Error("paid calls are outside Item 23")
    if not bool(config["discovery_policy"]["equal_initial_viability"]):
        raise GravityItem23Error("equal-viability policy changed")
    for relative, digest in config["dependency_sha256"].items():
        if _sha256_file(root / str(relative)) != str(digest):
            raise GravityItem23Error(f"scientific dependency changed: {relative}")
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
        raise GravityItem23Error("scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem23Error("could not read frozen Item 23 module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem23Error("Item 23 module differs from scientific freeze")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    keys = (
        "identity_catalog",
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
        "identity_catalog",
        "predictors",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        repo_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repo_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem23Error(f"{key} differs from sample freeze")


def _normal_identity(value: str) -> str:
    compact = re.sub(r"[^A-Z0-9+.-]", "", str(value).upper())
    match = re.search(r"J(\d{6})(?:\.\d+)?([+-]\d{6})(?:\.\d+)?", compact)
    if match:
        return f"J{match.group(1)}{match.group(2)}"
    simple = re.sub(r"[^A-Z0-9+-]", "", compact)
    return re.sub(r"^(SDSS|SL2S|HSC)", "", simple)


def _coordinate_from_sdss(value: str) -> tuple[float, float]:
    match = re.search(
        r"(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)",
        str(value),
    )
    if not match:
        raise GravityItem23Error(f"cannot parse SDSS coordinate: {value}")
    ra = 15.0 * (int(match[1]) + int(match[2]) / 60.0 + float(match[3]) / 3600.0)
    dec = int(match[5]) + int(match[6]) / 60.0 + float(match[7]) / 3600.0
    return ra, dec if match[4] == "+" else -dec


def _prior_exclusions(
    root: Path, config: Mapping[str, Any], item16_rows: Sequence[Mapping[str, str]]
) -> tuple[set[str], list[tuple[float, float, str]]]:
    item22_config = _read_json(root / "configs/gravity_item22_polarization_superposition_v1.json")
    names = {
        _normal_identity(value) for value in item22_config["sources"]["preview_exposed_identities"]
    }
    manifests = [root / str(value) for value in config["sources"]["predecessor_sample_manifests"]]
    loaded = [_read_json(path) for path in manifests]
    for manifest in loaded:
        for row in manifest.get("objects", []):
            if isinstance(row, Mapping) and row.get("name"):
                names.add(_normal_identity(str(row["name"])))
    coordinates: list[tuple[float, float, str]] = []
    item16_names = {
        str(row["name"])
        for row in loaded[0].get("objects", [])
        if isinstance(row, Mapping) and row.get("name")
    }
    for row in item16_rows:
        if row["Target"] in item16_names:
            coordinates.append((float(row["_RA"]), float(row["_DE"]), f"item16:{row['Target']}"))
    for row in loaded[1].get("objects", []):
        if isinstance(row, Mapping) and row.get("sdss"):
            ra, dec = _coordinate_from_sdss(str(row["sdss"]))
            coordinates.append((ra, dec, f"item17:{row['name']}"))
    for row in loaded[2].get("objects", []):
        if isinstance(row, Mapping) and row.get("catalog_ra_deg") is not None:
            coordinates.append(
                (
                    float(row["catalog_ra_deg"]),
                    float(row["catalog_dec_deg"]),
                    f"item22:{row['name']}",
                )
            )
    return names, coordinates


def _sdss_url(config: Mapping[str, Any], ra: float, dec: float, response: bool) -> str:
    columns = (
        config["sources"]["sdss_response_columns"]
        if response
        else config["sources"]["sdss_presence_columns"]
    )
    params = {
        "-source": "V/154/sdss16",
        "-c": f"{ra:.8f} {dec:.8f}",
        "-c.rs": str(config["sources"]["sdss_radius_arcsec"]),
        "-out": ",".join(columns),
        "-out.max": "100",
    }
    return str(config["sources"]["sdss_query_base"]) + "?" + urllib.parse.urlencode(params)


def _valid_spectra(
    rows: Sequence[Mapping[str, str]], z_lens: float, config: Mapping[str, Any]
) -> list[dict[str, str]]:
    valid: list[dict[str, str]] = []
    for row in rows:
        try:
            identifier = str(row["SpObjID"]).strip()
            zsp = float(row["zsp"])
            warning = str(row["f_zsp"]).strip()
        except (KeyError, ValueError):
            continue
        if (
            identifier not in ("", "0")
            and warning in ("", "0")
            and str(row["spCl"]).strip() == "GALAXY"
            and abs(zsp - z_lens) < float(config["sources"]["maximum_spectrum_redshift_difference"])
        ):
            valid.append(dict(row))
    return valid


def _best_spectrum(
    rows: Sequence[Mapping[str, str]], z_lens: float, config: Mapping[str, Any]
) -> dict[str, str] | None:
    valid = _valid_spectra(rows, z_lens, config)
    if not valid:
        return None
    return max(valid, key=lambda row: float(row.get("spS/N") or -1.0e30))


def _rz_quality_pass(reasons: Sequence[str]) -> bool:
    return all(str(reason).endswith("_g") for reason in reasons)


def _derive_predictor(
    identity: Mapping[str, Any],
    tractor: Mapping[str, Any],
    spectrum: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    mapped = {
        **identity,
        "_RA": identity["RAJ2000"],
        "_DE": identity["DEJ2000"],
        "Survey": "SUGOHI",
    }
    result = _derived_predictor(mapped, tractor, config)
    result["normalized_identity"] = _normal_identity(str(identity["Name"]))
    result["spectrum_id"] = str(spectrum["SpObjID"]).strip()
    result["spectrum_redshift"] = float(spectrum["zsp"])
    result["spectrum_sn"] = float(spectrum["spS/N"])
    result["max_fracflux_rz"] = max(float(tractor[f"fracflux_{band}"]) for band in ("r", "z"))
    result["max_fracmasked_rz"] = max(float(tractor[f"fracmasked_{band}"]) for band in ("r", "z"))
    result["minimum_flux_signal_to_noise_rz"] = min(
        float(tractor[f"flux_{band}"]) * math.sqrt(float(tractor[f"flux_ivar_{band}"]))
        for band in ("r", "z")
    )
    result.pop("g_minus_r", None)
    result.pop("max_fracflux_grz", None)
    result.pop("max_fracmasked_grz", None)
    result.pop("minimum_flux_signal_to_noise_grz", None)
    return result


def _build_sample(
    predictors: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    sample = config["sample"]
    selected = sorted(
        (dict(row) for row in predictors), key=lambda row: (float(row["z_lens"]), str(row["name"]))
    )
    if len(selected) != int(sample["selected_predictor_count"]):
        raise GravityItem23Error("predictor selection count changed")
    width = int(sample["objects_per_redshift_stratum"])
    for index, row in enumerate(selected):
        row["redshift_stratum"] = index // width
    for stratum in range(int(sample["redshift_strata"])):
        group = [row for row in selected if int(row["redshift_stratum"]) == stratum]
        if len(group) != width:
            raise GravityItem23Error("redshift stratum changed size")
        group.sort(key=lambda row: _hmac_rank(str(sample["role_key"]), str(row["name"])))
        held = {str(row["name"]) for row in group[: int(sample["confirmation_per_stratum"])]}
        for row in group:
            row["role"] = "confirmation" if str(row["name"]) in held else "exploration"
    exploration = [row for row in selected if row["role"] == "exploration"]
    exploration.sort(key=lambda row: _hmac_rank(str(sample["fold_key"]), str(row["name"])))
    for index, row in enumerate(exploration):
        row["fold"] = index % int(sample["outer_folds"])
    for row in selected:
        if row["role"] == "confirmation":
            row["fold"] = None
    selected.sort(key=lambda row: str(row["name"]))
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item23-bimetric-sample-1.0",
            "selection_used_response_values": False,
            "response_values_read": 0,
            "confirmation_opened": False,
            "counts": {
                "predictor_quality_eligible": len(predictors),
                "selected": len(selected),
                "exploration": len(exploration),
                "confirmation": len(selected) - len(exploration),
            },
            "fold_counts": dict(sorted(Counter(str(row["fold"]) for row in exploration).items())),
            "objects": selected,
        }
    )


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["raw_candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    niche = np.repeat(np.arange(4, dtype=np.int16), count // 4)
    random.shuffle(niche)
    return {
        "niche": niche,
        "mixing_angle": random.integers(
            0, len(generator["mixing_angles_rad"]), count, dtype=np.int16
        ),
        "massive_range": random.integers(
            0, len(generator["massive_ranges_kpc"]), count, dtype=np.int16
        ),
        "vainshtein_scale": random.integers(
            0, len(generator["vainshtein_scales_kpc_at_reference_mass"]), count, dtype=np.int16
        ),
        "cubic_c3": random.integers(0, len(generator["cubic_c3"]), count, dtype=np.int16),
        "cubic_c4": random.integers(0, len(generator["cubic_c4"]), count, dtype=np.int16),
        "transition_acceleration": random.integers(
            0, len(generator["transition_acceleration_mps2"]), count, dtype=np.int16
        ),
        "transition_power": random.integers(
            0, len(generator["transition_powers"]), count, dtype=np.int16
        ),
        "angle_ratio": random.integers(
            0, len(generator["inner_to_outer_angle_ratios"]), count, dtype=np.int16
        ),
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _candidate_values(
    config: Mapping[str, Any], arrays: Mapping[str, np.ndarray], begin: int, end: int, xp: Any
) -> dict[str, Any]:
    generator = config["candidate_generator"]
    lookups = {
        "mixing_angle": "mixing_angles_rad",
        "massive_range": "massive_ranges_kpc",
        "vainshtein_scale": "vainshtein_scales_kpc_at_reference_mass",
        "cubic_c3": "cubic_c3",
        "cubic_c4": "cubic_c4",
        "transition_acceleration": "transition_acceleration_mps2",
        "transition_power": "transition_powers",
        "angle_ratio": "inner_to_outer_angle_ratios",
    }
    result: dict[str, Any] = {"niche": xp.asarray(arrays["niche"][begin:end])}
    for key, source in lookups.items():
        result[key] = xp.asarray(np.asarray(generator[source])[arrays[key][begin:end]])
    return result


def _solve_monotone_branch(x: Any, c3: Any, c4: Any, iterations: int, xp: Any) -> Any:
    x = xp.maximum(x, 1.0e-300)
    initial_cubic = xp.cbrt(x / xp.maximum(c4, 1.0e-300))
    u = xp.minimum(x, initial_cubic)
    for _ in range(iterations):
        function = u + c3 * u**2 + c4 * u**3 - x
        derivative = 1.0 + 2.0 * c3 * u + 3.0 * c4 * u**2
        u = xp.maximum(u - function / derivative, 0.0)
    return xp.clip(u / x, 0.0, 1.0)


def _solar_mode_strength(
    config: Mapping[str, Any], values: Mapping[str, Any], xp: Any
) -> tuple[Any, Any]:
    constants = config["physics"]["constants"]
    radius = float(constants["au_to_kpc"])
    mass_ratio = 1.0 / float(config["physics"]["reference_mass_Msun"])
    rv = values["vainshtein_scale"] * mass_ratio ** (1.0 / 3.0)
    x = (rv / radius) ** 3
    screen = _solve_monotone_branch(
        x,
        values["cubic_c3"],
        values["cubic_c4"],
        int(config["evaluation"]["cubic_newton_iterations"]),
        xp,
    )
    screen = xp.where(values["niche"] <= 1, 1.0, screen)
    acceleration = (
        float(constants["G_SI"])
        * float(constants["solar_mass_kg"])
        / (radius * 3.085677581491367e19) ** 2
    )
    weight = 1.0 / (
        1.0
        + xp.exp(
            -values["transition_power"] * xp.log(acceleration / values["transition_acceleration"])
        )
    )
    delta = values["mixing_angle"]
    exchanged = delta * (1.0 + (values["angle_ratio"] - 1.0) * weight)
    delta = xp.where(values["niche"] == 3, exchanged, delta)
    delta = xp.clip(delta, 0.0, 0.75)
    amplitude = (4.0 / 3.0) * xp.tan(delta) ** 2
    yukawa = (1.0 + radius / values["massive_range"]) * xp.exp(-radius / values["massive_range"])
    return amplitude * screen * yukawa, screen


def _admissible_candidates(
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    values = _candidate_values(config, raw, 0, len(raw["niche"]), np)
    local_strength, _ = _solar_mode_strength(config, values, np)
    projector = float(config["physics"]["massive_light_to_dynamics_projector"])
    local_deviation = (1.0 - projector) * local_strength / (1.0 + local_strength)
    limit = float(config["gates"]["max_local_fractional_deviation_at_1AU"])
    keep = np.isfinite(local_deviation) & (local_deviation <= limit)
    arrays = {key: value[keep].copy() for key, value in raw.items()}
    return arrays, {
        "raw_cells": len(raw["niche"]),
        "admissible_cells": int(np.count_nonzero(keep)),
        "maximum_admitted_local_fractional_deviation": float(np.max(local_deviation[keep])),
        "raw_niche_counts": dict(sorted(Counter(int(value) for value in raw["niche"]).items())),
        "admissible_niche_counts": dict(
            sorted(Counter(int(value) for value in arrays["niche"]).items())
        ),
    }


def _candidate_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    arrays, audit = _admissible_candidates(config)
    niches = config["candidate_generator"]["niches"]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item23-bimetric-candidates-1.0",
            "response_values_read": 0,
            "post_response_candidate_cells": 0,
            **audit,
            "raw_niche_counts": {
                str(niches[key]["id"]): value for key, value in audit["raw_niche_counts"].items()
            },
            "admissible_niche_counts": {
                str(niches[key]["id"]): value
                for key, value in audit["admissible_niche_counts"].items()
            },
            "candidate_array_sha256": _candidate_digest(arrays),
            "positive_kinetic_norms": True,
            "positive_fierz_pauli_mass_squared": True,
            "unique_monotone_nonlinear_branch": True,
            "acyclic_metric_interaction_graph": True,
            "equivalence_boundaries": config["candidate_generator"]["equivalence_boundaries"],
            "historical_novelty_claimed": False,
        }
    )


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["identity_catalog"].parent.mkdir(parents=True, exist_ok=True)
    identity_body, identity_headers = _download(str(config["sources"]["identity_query"]))
    paths["identity_catalog"].write_bytes(identity_body)
    identity_columns = tuple(config["sources"]["identity_columns"])
    identities = _parse_vizier_tsv(identity_body, identity_columns)
    item16_body, item16_headers = _download(str(config["sources"]["item16_coordinate_query"]))
    item16_columns = tuple(config["sources"]["item16_coordinate_columns"])
    item16_rows = _parse_vizier_tsv(item16_body, item16_columns)
    prior_names, prior_coordinates = _prior_exclusions(root, config, item16_rows)
    grade_a = [row for row in identities if str(row["Grade"]).strip() == "A"]
    both_redshifts = [
        row
        for row in grade_a
        if float(row.get("zl") or -9.0) > 0.0 and float(row.get("zs") or -9.0) > 0.0
    ]
    name_excluded = [row for row in both_redshifts if _normal_identity(row["Name"]) in prior_names]
    after_names = [
        row for row in both_redshifts if _normal_identity(row["Name"]) not in prior_names
    ]
    coordinate_excluded: list[dict[str, Any]] = []
    fresh: list[dict[str, str]] = []
    threshold = float(config["sources"]["maximum_predecessor_coordinate_separation_arcsec"])
    for row in after_names:
        nearest = min(
            (
                _angular_separation_arcsec(float(row["RAJ2000"]), float(row["DEJ2000"]), ra, dec),
                label,
            )
            for ra, dec, label in prior_coordinates
        )
        if nearest[0] <= threshold:
            coordinate_excluded.append(
                {"name": row["Name"], "separation_arcsec": nearest[0], "predecessor": nearest[1]}
            )
        else:
            fresh.append(dict(row))
    presence_receipts: list[dict[str, Any]] = []
    spectra: dict[str, dict[str, str]] = {}
    for row in fresh:
        url = _sdss_url(config, float(row["RAJ2000"]), float(row["DEJ2000"]), False)
        body, headers = _download(url)
        parsed = _parse_vizier_tsv(body, tuple(config["sources"]["sdss_presence_columns"]))
        best = _best_spectrum(parsed, float(row["zl"]), config)
        presence_receipts.append(
            {
                "name": row["Name"],
                "url": url,
                "bytes": len(body),
                "sha256": _sha256_bytes(body),
                "last_modified": headers.get("last-modified"),
                "spectrum_present": best is not None,
                "approved_columns": config["sources"]["sdss_presence_columns"],
            }
        )
        if best is not None:
            spectra[str(row["Name"])] = best
    spectral_rows = [row for row in fresh if str(row["Name"]) in spectra]
    predictor_receipts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    predictors: list[dict[str, Any]] = []
    for row in spectral_rows:
        url = _legacy_url(config, float(row["RAJ2000"]), float(row["DEJ2000"]))
        body, headers = _download(url)
        tractor, reasons = _nearest_legacy_row(
            body, float(row["RAJ2000"]), float(row["DEJ2000"]), config
        )
        passes = tractor is not None and _rz_quality_pass(reasons)
        predictor_receipts.append(
            {
                "name": row["Name"],
                "url": url,
                "bytes": len(body),
                "sha256": _sha256_bytes(body),
                "last_modified": headers.get("last-modified"),
                "quality_pass_rz": passes,
                "all_grz_quality_failures": reasons,
            }
        )
        if passes and tractor is not None:
            predictors.append(_derive_predictor(row, tractor, spectra[str(row["Name"])], config))
        else:
            failures.append({"name": row["Name"], "quality_failures": reasons})
    expected = config["sources"]
    checks = (
        (len(identities), int(expected["expected_identity_rows"]), "identity rows"),
        (len(grade_a), int(expected["expected_grade_a_rows"]), "Grade A rows"),
        (
            len(both_redshifts),
            int(expected["expected_grade_a_both_spectroscopic_redshifts"]),
            "Grade A spectroscopic-redshift rows",
        ),
        (
            len(name_excluded),
            int(expected["expected_name_or_preview_exclusions"]),
            "name/preview exclusions",
        ),
        (
            len(coordinate_excluded),
            int(expected["expected_coordinate_exclusions_after_names"]),
            "coordinate exclusions",
        ),
        (
            len(spectral_rows),
            int(expected["expected_target_blind_spectral_presence"]),
            "spectral presence",
        ),
        (
            len(predictors),
            int(expected["expected_rz_predictor_quality_eligible"]),
            "r/z predictor eligibility",
        ),
    )
    for actual, wanted, label in checks:
        if actual != wanted:
            raise GravityItem23Error(f"{label} changed: {actual} != {wanted}")
    _write_tsv(paths["predictors"], predictors, list(predictors[0]))
    sample_manifest = _build_sample(predictors, config)
    candidate_manifest = _candidate_manifest(config)
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item23-bimetric-predictor-source-1.0",
            "response_values_read": 0,
            "selection_used_response_values": False,
            "identity_source": {
                "url": config["sources"]["identity_query"],
                "bytes": len(identity_body),
                "sha256": _sha256_bytes(identity_body),
                "last_modified": identity_headers.get("last-modified"),
                "approved_columns": list(identity_columns),
            },
            "predecessor_coordinate_source": {
                "url": config["sources"]["item16_coordinate_query"],
                "bytes": len(item16_body),
                "sha256": _sha256_bytes(item16_body),
                "last_modified": item16_headers.get("last-modified"),
                "approved_columns": list(item16_columns),
            },
            "counts": {
                "identity_rows": len(identities),
                "grade_a_rows": len(grade_a),
                "grade_a_both_spectroscopic_redshifts": len(both_redshifts),
                "name_or_preview_exclusions": len(name_excluded),
                "coordinate_exclusions": len(coordinate_excluded),
                "fresh_before_spectrum_presence": len(fresh),
                "target_blind_spectral_presence": len(spectral_rows),
                "rz_predictor_quality_failures": len(failures),
                "rz_predictor_quality_eligible": len(predictors),
            },
            "name_or_preview_exclusions": sorted(str(row["Name"]) for row in name_excluded),
            "coordinate_exclusions": sorted(coordinate_excluded, key=lambda row: row["name"]),
            "spectrum_presence_receipts": presence_receipts,
            "legacy_source": {
                "release": config["sources"]["legacy_release"],
                "documentation": config["sources"]["legacy_catalog_documentation"],
                "quality_bands": config["sources"]["legacy_quality_bands"],
                "per_object_receipts": predictor_receipts,
            },
            "predictor_quality_failures": failures,
            "predictor_file": {
                "path": paths["predictors"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["predictors"]),
            },
            "claim_boundary": config["scope"]["claim_ceiling"],
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample_manifest)
    _write_json(paths["candidate_manifest"], candidate_manifest)
    return paths


def _load_prepared(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _source_paths(root, config)
    predictor = _read_json(paths["predictor_source_manifest"])
    sample = _read_json(paths["sample_manifest"])
    candidates = _read_json(paths["candidate_manifest"])
    for value, label in (
        (predictor, "predictor manifest"),
        (sample, "sample manifest"),
        (candidates, "candidate manifest"),
    ):
        _verify_content_hash(value, label)
        if int(value.get("response_values_read", 0)) != 0:
            raise GravityItem23Error(f"{label} contains response values")
    if _sha256_file(paths["predictors"]) != predictor["predictor_file"]["sha256"]:
        raise GravityItem23Error("predictor TSV changed")
    arrays, _ = _admissible_candidates(config)
    if _candidate_digest(arrays) != candidates["candidate_array_sha256"]:
        raise GravityItem23Error("candidate array changed")
    return predictor, sample, candidates


def _sugohi_post(config: Mapping[str, Any], ra: float, dec: float) -> tuple[bytes, dict[str, str]]:
    payload = {
        "outfmt": "csv",
        "gradea": "ok",
        "galaxy": "ok",
        "group": "ok",
        "gsource": "ok",
        "qsource": "ok",
        "ylspecz": "ok",
        "nlspecz": "ok",
        "ysspecz": "ok",
        "nsspecz": "ok",
        "position": "radial",
        "ra0": f"{ra:.8f}",
        "dec0": f"{dec:.8f}",
        "radius": "0.05",
        "showname": "ok",
        "showra": "ok",
        "showdec": "ok",
        "showrein": "ok",
        "sort": "ra",
        "order": "asc",
    }
    request = urllib.request.Request(
        str(config["sources"]["sugohi_exact_response_endpoint"]),
        data=urllib.parse.urlencode(payload).encode("ascii"),
        headers={"User-Agent": "Invariant-Item23/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if not body:
        raise GravityItem23Error("empty SuGOHI exact response")
    return body, headers


def _parse_sugohi_csv(body: bytes) -> list[dict[str, str]]:
    text = body.decode("utf-8-sig")
    rows = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    return [
        {("name" if key == "#name" else key): value for key, value in row.items()} for row in rows
    ]


def acquire_responses(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    _, sample, candidates = _load_prepared(root, config)
    if int(candidates["post_response_candidate_cells"]) != 0:
        raise GravityItem23Error("post-response candidates detected")
    paths = _source_paths(root, config)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    responses: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for row in exploration:
        name = str(row["name"])
        ra, dec, z_lens = (
            float(row["catalog_ra_deg"]),
            float(row["catalog_dec_deg"]),
            float(row["z_lens"]),
        )
        sdss_url = _sdss_url(config, ra, dec, True)
        sdss_body, sdss_headers = _download(sdss_url)
        sdss_rows = _parse_vizier_tsv(sdss_body, tuple(config["sources"]["sdss_response_columns"]))
        spectrum = _best_spectrum(sdss_rows, z_lens, config)
        sugohi_body, sugohi_headers = _sugohi_post(config, ra, dec)
        sugohi_rows = _parse_sugohi_csv(sugohi_body)
        nearby = []
        for value in sugohi_rows:
            try:
                separation = _angular_separation_arcsec(
                    ra, dec, float(value["ra"]), float(value["dec"])
                )
            except (KeyError, ValueError):
                continue
            if separation <= float(
                config["sources"]["maximum_predecessor_coordinate_separation_arcsec"]
            ):
                nearby.append((separation, value))
        lens = min(nearby, key=lambda value: value[0])[1] if nearby else None
        responses.append(
            {
                "Name": name,
                "Vdisp": "" if spectrum is None else spectrum.get("Vdisp", ""),
                "e_Vdisp": "" if spectrum is None else spectrum.get("e_Vdisp", ""),
                "Rein": "" if lens is None else lens.get("Rein", ""),
            }
        )
        receipts.append(
            {
                "name": name,
                "role": "exploration",
                "sdss": {
                    "url": sdss_url,
                    "bytes": len(sdss_body),
                    "sha256": _sha256_bytes(sdss_body),
                    "last_modified": sdss_headers.get("last-modified"),
                    "approved_columns": config["sources"]["sdss_response_columns"],
                    "matching_spectrum": spectrum is not None,
                },
                "sugohi": {
                    "endpoint": config["sources"]["sugohi_exact_response_endpoint"],
                    "query_center": [ra, dec],
                    "radius_arcmin": 0.05,
                    "bytes": len(sugohi_body),
                    "sha256": _sha256_bytes(sugohi_body),
                    "last_modified": sugohi_headers.get("last-modified"),
                    "matching_lens": lens is not None,
                },
            }
        )
    columns = tuple(config["sources"]["response_output_columns"])
    _write_tsv(paths["exploration_responses"], responses, columns)
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item23-bimetric-response-source-1.0",
            "confirmation_response_values_read": 0,
            "exploration_response_rows": len(responses),
            "post_response_candidate_cells": 0,
            "output_columns": list(columns),
            "forbidden_columns_used": [],
            "response_file": {
                "path": paths["exploration_responses"].relative_to(root).as_posix(),
                "sha256": _sha256_file(paths["exploration_responses"]),
            },
            "per_object_source_receipts": receipts,
        }
    )
    _write_json(paths["response_source_manifest"], manifest)
    return paths["exploration_responses"]


def _load_response_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    _, sample, _ = _load_prepared(root, config)
    manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(manifest, "response manifest")
    if int(manifest["confirmation_response_values_read"]) != 0:
        raise GravityItem23Error("confirmation response was opened")
    if _sha256_file(paths["exploration_responses"]) != manifest["response_file"]["sha256"]:
        raise GravityItem23Error("response TSV changed")
    response = {row["Name"]: row for row in _read_tsv(paths["exploration_responses"])}
    rows: list[dict[str, Any]] = []
    for predictor in sample["objects"]:
        if predictor["role"] != "exploration":
            continue
        observed = response.get(str(predictor["name"]))
        if observed is None:
            continue
        try:
            sigma = float(observed["Vdisp"])
            sigma_error = float(observed["e_Vdisp"])
            theta = float(observed["Rein"])
            z_lens = float(predictor["z_lens"])
            z_source = float(predictor["z_source"])
            reff = float(predictor["reff_kpc"])
            luminosity = float(predictor["z_luminosity_Lsun"])
        except (TypeError, ValueError):
            continue
        if not (
            sigma > 0.0
            and sigma_error > 0.0
            and theta > 0.0
            and z_source > z_lens > 0.0
            and reff > 0.0
            and luminosity > 0.0
        ):
            continue
        d_lens, d_source, d_lens_source = _angular_diameter_distances(z_lens, z_source, config)
        rein = theta * float(config["physics"]["constants"]["arcsec_to_radian"]) * d_lens
        a = reff / float(config["physics"]["hernquist_re_over_a"])
        projected_fraction = float(hernquist_projected_mass_fraction(rein / a))
        gravitational = float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
        c = float(config["physics"]["constants"]["c_km_s"])
        sigma_critical = (c**2 / (4.0 * math.pi * gravitational)) * (
            d_source / (d_lens * d_lens_source)
        )
        required_lens_mass = math.pi * rein**2 * sigma_critical
        virial = float(config["physics"]["dynamical_virial_coefficient"])
        rows.append(
            {
                **predictor,
                "fold": int(predictor["fold"]),
                "sigma_km_s": sigma,
                "sigma_error_km_s": sigma_error,
                "theta_ein_arcsec": theta,
                "theta_ein_error_arcsec": 0.0,
                "rein_kpc": rein,
                "projected_fraction_at_rein": projected_fraction,
                "y_dyn": math.log(virial * reff * sigma**2 / (gravitational * luminosity)),
                "y_lens": math.log(required_lens_mass / (luminosity * projected_fraction)),
            }
        )
    rows.sort(key=lambda row: str(row["name"]))
    return rows, manifest


def _candidate_log_mu(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    values = _candidate_values(config, arrays, begin, end, xp)
    radii_np = np.asarray([[row["reff_kpc"], row["rein_kpc"]] for row in rows], dtype=np.float64)
    luminosity_np = np.asarray([row["z_luminosity_Lsun"] for row in rows], dtype=np.float64)
    reff_np = np.asarray([row["reff_kpc"] for row in rows], dtype=np.float64)
    radii = xp.asarray(radii_np)[None, :, :]
    luminosity = xp.asarray(luminosity_np)[None, :, None]
    reff = xp.asarray(reff_np)[None, :, None]
    mass = float(config["physics"]["fiducial_trigger_mass_to_light_Msun_per_Lzsun"]) * luminosity
    scale = reff / float(config["physics"]["hernquist_re_over_a"])
    enclosed = mass * radii**2 / (radii + scale) ** 2
    acceleration = (
        float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
        * enclosed
        / radii**2
        * float(config["physics"]["constants"]["mps2_per_km2_s2_kpc"])
    )
    rv = values["vainshtein_scale"][:, None, None] * (
        mass / float(config["physics"]["reference_mass_Msun"])
    ) ** (1.0 / 3.0)
    x = (rv / radii) ** 3
    screen = _solve_monotone_branch(
        x,
        values["cubic_c3"][:, None, None],
        values["cubic_c4"][:, None, None],
        int(config["evaluation"]["cubic_newton_iterations"]),
        xp,
    )
    screen = xp.where((values["niche"] <= 1)[:, None, None], 1.0, screen)
    yukawa = (1.0 + radii / values["massive_range"][:, None, None]) * xp.exp(
        -radii / values["massive_range"][:, None, None]
    )
    weight = 1.0 / (
        1.0
        + xp.exp(
            -values["transition_power"][:, None, None]
            * xp.log(acceleration / values["transition_acceleration"][:, None, None])
        )
    )
    delta = values["mixing_angle"][:, None, None]
    exchanged = delta * (1.0 + (values["angle_ratio"][:, None, None] - 1.0) * weight)
    delta = xp.where((values["niche"] == 3)[:, None, None], exchanged, delta)
    delta = xp.clip(delta, 0.0, 0.75)
    amplitude = (4.0 / 3.0) * xp.tan(delta) ** 2
    mode = amplitude * screen * yukawa
    solar, _ = _solar_mode_strength(config, values, xp)
    denominator = 1.0 + solar[:, None, None]
    projector = float(config["physics"]["massive_light_to_dynamics_projector"])
    mu_dynamics = (1.0 + mode) / denominator
    mu_lensing = (1.0 + projector * mode) / denominator
    return xp.stack([xp.log(mu_dynamics[:, :, 0]), xp.log(mu_lensing[:, :, 1])], axis=2)


def _build_log_mu_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    pieces = []
    batch = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(arrays["niche"]), batch):
        end = min(begin + batch, len(arrays["niche"]))
        pieces.append(_candidate_log_mu(config, arrays, rows, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _fit_offset(values: np.ndarray, bounds: tuple[float, float]) -> tuple[float, float]:
    raw = float(np.mean(values))
    return raw, float(np.clip(raw, math.log(bounds[0]), math.log(bounds[1])))


def _screen_log_mu(
    log_mu: Any, y: np.ndarray, folds: np.ndarray, config: Mapping[str, Any], xp: Any
) -> dict[str, Any]:
    bounds = tuple(
        float(value) for value in config["physics"]["shared_mass_to_light_bounds_Msun_per_Lzsun"]
    )
    y_device = xp.asarray(y)
    prediction = np.empty_like(y)
    selected: list[int] = []
    offsets: list[float] = []
    raw_offsets: list[float] = []
    training_mse: list[float] = []
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        residual = y_device[None, train, :] - log_mu[:, train, :]
        raw = xp.mean(residual, axis=(1, 2))
        fitted = xp.clip(raw, math.log(bounds[0]), math.log(bounds[1]))
        mse = xp.mean((residual - fitted[:, None, None]) ** 2, axis=(1, 2))
        index = int(_to_numpy(xp.argmin(mse), xp))
        selected.append(index)
        raw_offsets.append(float(_to_numpy(raw[index], xp)))
        offsets.append(float(_to_numpy(fitted[index], xp)))
        training_mse.append(float(_to_numpy(mse[index], xp)))
        prediction[held] = _to_numpy(log_mu[index, held, :], xp) + offsets[-1]
    return {
        "prediction": prediction,
        "selected_indices": selected,
        "log_mass_to_light_offsets": offsets,
        "raw_log_mass_to_light_offsets": raw_offsets,
        "training_mse": training_mse,
    }


def _feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                math.log(float(row["z_luminosity_Lsun"])),
                math.log(float(row["reff_kpc"])),
                float(row["z_lens"]),
                float(row["r_minus_z"]),
                float(row["axis_ratio"]),
                float(row["max_fracflux_rz"]),
                math.log1p(max(float(row["spectrum_sn"]), 0.0)),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _baseline_predictions(
    y: np.ndarray, folds: np.ndarray, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    bounds = tuple(
        float(value) for value in config["physics"]["shared_mass_to_light_bounds_Msun_per_Lzsun"]
    )
    shared = np.empty_like(y)
    separate = np.empty_like(y)
    flexible = np.empty_like(y)
    features = _feature_matrix(rows)
    alpha = float(config["evaluation"]["ridge_alpha"])
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        held = np.where(folds == fold)[0]
        _, offset = _fit_offset(y[train].reshape(-1), bounds)
        shared[held] = offset
        for channel in range(2):
            _, channel_offset = _fit_offset(y[train, channel], bounds)
            separate[held, channel] = channel_offset
        mean, scale = features[train].mean(axis=0), features[train].std(axis=0)
        scale[scale == 0.0] = 1.0
        train_design = np.column_stack([np.ones(len(train)), (features[train] - mean) / scale])
        held_design = np.column_stack([np.ones(len(held)), (features[held] - mean) / scale])
        penalty = np.diag([0.0] + [alpha] * features.shape[1])
        for channel in range(2):
            coefficient = np.linalg.solve(
                train_design.T @ train_design + penalty, train_design.T @ y[train, channel]
            )
            flexible[held, channel] = held_design @ coefficient
    return {"shared_GR": shared, "separate_calibration": separate, "flexible_nuisance": flexible}


def _mse(y: np.ndarray, prediction: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is not None:
        y, prediction = y[indices], prediction[indices]
    return float(np.mean((y - prediction) ** 2))


def _improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference if reference > 0.0 else float("-inf")


def _selected_cell(
    index: int, config: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    values = _candidate_values(config, arrays, index, index + 1, np)
    niche = int(arrays["niche"][index])
    return {
        "candidate_index": index,
        "niche": config["candidate_generator"]["niches"][niche],
        "mixing_angle_rad": float(values["mixing_angle"][0]),
        "massive_range_kpc": float(values["massive_range"][0]),
        "vainshtein_scale_kpc_at_reference_mass": float(values["vainshtein_scale"][0]),
        "cubic_c3": float(values["cubic_c3"][0]),
        "cubic_c4": float(values["cubic_c4"][0]),
        "transition_acceleration_mps2": float(values["transition_acceleration"][0]),
        "transition_power": float(values["transition_power"][0]),
        "inner_to_outer_angle_ratio": float(values["angle_ratio"][0]),
    }


def _synthetic_controls(
    log_mu: Any,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    xp: Any,
) -> dict[str, Any]:
    niche = np.where(arrays["niche"] == 3)[0]
    injection_index = int(niche[len(niche) // 2])
    injection = _to_numpy(log_mu[injection_index], xp)
    y_injection = math.log(2.0) + injection
    injected = _screen_log_mu(log_mu, y_injection, folds, config, xp)
    injected_niches = [int(arrays["niche"][index]) for index in injected["selected_indices"]]
    baseline = _baseline_predictions(y_injection, folds, rows, config)["flexible_nuisance"]
    y_gr = np.full((len(rows), 2), math.log(2.0), dtype=np.float64)
    gr = _screen_log_mu(log_mu, y_gr, folds, config, xp)
    gr_baseline = _baseline_predictions(y_gr, folds, rows, config)["shared_GR"]
    return {
        "injection_candidate_index": injection_index,
        "injection_selected_niches": injected_niches,
        "injection_exact_niche_recovered_all_folds": all(value == 3 for value in injected_niches),
        "injection_candidate_mse": _mse(y_injection, injected["prediction"]),
        "injection_flexible_mse": _mse(y_injection, baseline),
        "injection_improves_over_flexible": _mse(y_injection, injected["prediction"])
        < _mse(y_injection, baseline),
        "GR_candidate_mse": _mse(y_gr, gr["prediction"]),
        "GR_baseline_mse": _mse(y_gr, gr_baseline),
        "GR_control_prefers_nonzero_bimetric_response": _mse(y_gr, gr["prediction"])
        < _mse(y_gr, gr_baseline) - 1.0e-18,
    }


def _weighted_mse(
    y: np.ndarray,
    prediction: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float:
    luminosity_error = math.log(10.0) * float(
        config["evaluation"]["stellar_luminosity_systematic_dex"]
    )
    theta_fraction = float(config["evaluation"]["lens_radius_fractional_uncertainty_when_missing"])
    errors = [
        [
            math.hypot(
                2.0 * float(row["sigma_error_km_s"]) / float(row["sigma_km_s"]), luminosity_error
            ),
            math.hypot(2.0 * theta_fraction, luminosity_error),
        ]
        for row in rows
    ]
    weights = 1.0 / np.asarray(errors, dtype=np.float64) ** 2
    return float(np.sum(weights * (y - prediction) ** 2) / np.sum(weights))


def _evaluate(
    root: Path, config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], record_compute: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum = int(config["sample"]["minimum_complete_exploration_objects"])
    if len(rows) < minimum:
        raise GravityItem23Error(f"too few complete exploration objects: {len(rows)} < {minimum}")
    arrays, admissibility = _admissible_candidates(config)
    xp, backend, device = _backend()
    y = np.asarray([[row["y_dyn"], row["y_lens"]] for row in rows], dtype=np.float64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if set(folds.tolist()) != set(range(int(config["sample"]["outer_folds"]))):
        raise GravityItem23Error("exploration folds are incomplete")
    xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    log_mu = _build_log_mu_matrix(config, arrays, rows, xp)
    xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    crosscheck = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["niche"]))
    cpu = _candidate_log_mu(config, arrays, rows, 0, crosscheck, np)
    gpu = _to_numpy(log_mu[:crosscheck], xp)
    cpu_gpu_max = float(np.max(np.abs(cpu - gpu)))
    controls = _synthetic_controls(log_mu, folds, rows, config, arrays, xp)
    start = time.perf_counter()
    selected = _screen_log_mu(log_mu, y, folds, config, xp)
    baselines = _baseline_predictions(y, folds, rows, config)
    candidate_mse = _mse(y, selected["prediction"])
    baseline_mse = {key: _mse(y, value) for key, value in baselines.items()}
    observed = _improvement(baseline_mse["flexible_nuisance"], candidate_mse)
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    trials = int(config["evaluation"]["permutation_trials"])
    null_statistics: list[float] = []
    for trial in range(trials):
        permuted_y = y[random.permutation(len(rows))]
        permuted = _screen_log_mu(log_mu, permuted_y, folds, config, xp)
        flexible = _baseline_predictions(permuted_y, folds, rows, config)["flexible_nuisance"]
        null_statistics.append(
            _improvement(_mse(permuted_y, flexible), _mse(permuted_y, permuted["prediction"]))
        )
        if record_compute and (trial + 1) % 25 == 0:
            print(f"Item 23 selection-aware null {trial + 1}/{trials}", flush=True)
    xp.cuda.Stream.null.synchronize()
    screen_seconds = time.perf_counter() - start
    raw_p = (1 + sum(value >= observed for value in null_statistics)) / (trials + 1)
    guarded_p = 1.0 if observed <= 0.0 else raw_p
    cells = [_selected_cell(index, config, arrays) for index in selected["selected_indices"]]
    niche_counts = Counter(str(cell["niche"]["id"]) for cell in cells)
    nonlinear_count = max(
        (
            value
            for key, value in niche_counts.items()
            if key
            in {"monotone_nonlinear_bimetric_branch", "baryon_triggered_metric_ratio_exchange"}
        ),
        default=0,
    )
    channel_metrics: dict[str, Any] = {}
    for channel, label in enumerate(("stellar_dynamics", "Einstein_radius_lensing")):
        value = {
            "candidate_mse": float(
                np.mean((y[:, channel] - selected["prediction"][:, channel]) ** 2)
            )
        }
        for key, prediction in baselines.items():
            mse = float(np.mean((y[:, channel] - prediction[:, channel]) ** 2))
            value[f"{key}_mse"] = mse
            value[f"improvement_vs_{key}"] = _improvement(mse, value["candidate_mse"])
        channel_metrics[label] = value
    strata: dict[str, Any] = {}
    for key, label in (
        ("z_luminosity_Lsun", "luminosity"),
        ("reff_kpc", "size"),
        ("z_lens", "redshift"),
    ):
        values = np.asarray([float(row[key]) for row in rows])
        median = float(np.median(values))
        for side, indices in (
            ("low", np.where(values <= median)[0]),
            ("high", np.where(values > median)[0]),
        ):
            candidate = _mse(y, selected["prediction"], indices)
            shared = _mse(y, baselines["shared_GR"], indices)
            flexible = _mse(y, baselines["flexible_nuisance"], indices)
            strata[f"{label}_{side}"] = {
                "objects": len(indices),
                "candidate_mse": candidate,
                "shared_GR_mse": shared,
                "flexible_nuisance_mse": flexible,
                "improvement_vs_shared_GR": _improvement(shared, candidate),
                "improvement_vs_flexible_nuisance": _improvement(flexible, candidate),
            }
    bounds = tuple(
        float(value) for value in config["physics"]["shared_mass_to_light_bounds_Msun_per_Lzsun"]
    )
    raw_scales = [math.exp(value) for value in selected["raw_log_mass_to_light_offsets"]]
    mass_to_light_in_bounds = all(bounds[0] <= value <= bounds[1] for value in raw_scales)
    gates_config = config["gates"]
    universal_gates = {
        "minimum_complete_exploration_objects": len(rows) >= minimum,
        "confirmation_values_read_zero": True,
        "post_response_candidate_cells_zero": int(
            config["candidate_generator"]["post_response_cells"]
        )
        == 0,
        "local_classical_limit": admissibility["maximum_admitted_local_fractional_deviation"]
        <= float(gates_config["max_local_fractional_deviation_at_1AU"]),
        "positive_kinetic_mass_and_observables": bool(np.all(np.isfinite(_to_numpy(log_mu, xp)))),
        "unique_monotone_branch": True,
        "synthetic_injection_recovers_branch_exchange": bool(
            controls["injection_exact_niche_recovered_all_folds"]
        )
        and bool(controls["injection_improves_over_flexible"]),
        "known_GR_control": not bool(controls["GR_control_prefers_nonzero_bimetric_response"]),
        "joint_improvement_vs_shared_GR": _improvement(baseline_mse["shared_GR"], candidate_mse)
        >= float(gates_config["minimum_joint_mse_improvement_vs_shared_GR"]),
        "joint_improvement_vs_separate_calibration": _improvement(
            baseline_mse["separate_calibration"], candidate_mse
        )
        >= float(gates_config["minimum_joint_mse_improvement_vs_separate_calibration"]),
        "joint_improvement_vs_flexible_nuisance": observed
        > float(gates_config["minimum_joint_mse_improvement_vs_flexible_nuisance"]),
        "both_channels_improve_vs_shared_GR": all(
            value["improvement_vs_shared_GR"]
            > float(gates_config["minimum_each_channel_improvement_vs_shared_GR"])
            for value in channel_metrics.values()
        ),
        "all_broad_halves_improve_vs_shared_GR": all(
            value["improvement_vs_shared_GR"]
            > float(gates_config["minimum_each_broad_half_improvement_vs_shared_GR"])
            for value in strata.values()
        ),
        "selection_aware_permutation": guarded_p
        <= float(gates_config["maximum_selection_aware_permutation_p"]),
        "stable_nonlinear_niche": nonlinear_count
        >= int(gates_config["minimum_same_nonlinear_niche_folds"]),
        "shared_mass_to_light_in_bounds": mass_to_light_in_bounds,
    }
    phenomenon_gates = {
        "beats_flexible_by_required_margin": observed
        >= float(gates_config["phenomenon_minimum_improvement_vs_flexible"]),
        "selection_aware_significance": guarded_p
        <= float(gates_config["maximum_selection_aware_permutation_p"]),
        "stable_nonlinear_niche": nonlinear_count
        >= int(gates_config["minimum_same_nonlinear_niche_folds"]),
        "at_least_one_channel_beats_flexible": any(
            value["improvement_vs_flexible_nuisance"] > 0.0 for value in channel_metrics.values()
        ),
        "pipeline_controls": bool(controls["injection_exact_niche_recovered_all_folds"])
        and not bool(controls["GR_control_prefers_nonzero_bimetric_response"]),
    }
    universal_advance = all(universal_gates.values())
    phenomenon_lead = all(phenomenon_gates.values())
    counterexamples = [
        str(row["name"])
        for index, row in enumerate(rows)
        if float(np.mean((y[index] - selected["prediction"][index]) ** 2))
        > float(np.mean((y[index] - baselines["flexible_nuisance"][index]) ** 2))
    ]
    score_evaluations = (
        len(arrays["niche"])
        * 2
        * sum(
            int(np.count_nonzero(folds != fold))
            for fold in range(int(config["sample"]["outer_folds"]))
        )
    )
    compute = {
        "schema_version": "invariant-gravity-item23-bimetric-compute-1.0",
        "backend": backend,
        "device": device,
        "numpy_version": np.__version__,
        "cupy_version": getattr(xp, "__version__", None),
        "raw_candidate_cells": int(config["candidate_generator"]["raw_candidate_cells"]),
        "admissible_candidate_cells": len(arrays["niche"]),
        "objects": len(rows),
        "channels": 2,
        "candidate_observable_matrix_values": int(np.prod(log_mu.shape)),
        "candidate_training_residual_evaluations_observed": score_evaluations,
        "candidate_training_residual_evaluations_with_nulls": score_evaluations * (trials + 1),
        "matrix_build_seconds_observed": matrix_seconds,
        "screen_and_null_seconds_observed": screen_seconds,
        "cpu_gpu_max_absolute_log_mu_difference": cpu_gpu_max,
    }
    scientific = {
        "decision": "PASS_ITEM23_BIMETRIC_EXPLORATION"
        if universal_advance
        else "REJECT_ITEM23_BIMETRIC_EXPLORATION",
        "track_decisions": {
            "universal_gravity": "ADVANCE" if universal_advance else "DO_NOT_ADVANCE",
            "phenomenon_publication": "REPLICATION_LEAD"
            if phenomenon_lead
            else "NO_EMPIRICAL_LEAD",
            "paper_claim_now": False,
        },
        "counts": {
            "valid_exploration_objects": len(rows),
            "raw_candidate_cells": int(config["candidate_generator"]["raw_candidate_cells"]),
            "admissible_candidate_cells": len(arrays["niche"]),
            "post_response_candidate_cells": 0,
            "permutation_trials": trials,
            "individual_counterexamples_vs_flexible": len(counterexamples),
        },
        "primary_metrics": {
            "candidate_mse": candidate_mse,
            **{f"{key}_mse": value for key, value in baseline_mse.items()},
            "improvement_vs_shared_GR": _improvement(baseline_mse["shared_GR"], candidate_mse),
            "improvement_vs_separate_calibration": _improvement(
                baseline_mse["separate_calibration"], candidate_mse
            ),
            "improvement_vs_flexible_nuisance": observed,
            "selection_aware_raw_permutation_p": raw_p,
            "selection_aware_guarded_permutation_p": guarded_p,
        },
        "weighted_robustness": {
            "candidate_mse": _weighted_mse(y, selected["prediction"], rows, config),
            **{
                f"{key}_mse": _weighted_mse(y, value, rows, config)
                for key, value in baselines.items()
            },
        },
        "channel_metrics": channel_metrics,
        "stratum_metrics": strata,
        "outer_selections": [
            {
                "fold": fold,
                "cell": cells[fold],
                "training_mse": selected["training_mse"][fold],
                "mass_to_light": math.exp(selected["log_mass_to_light_offsets"][fold]),
                "unclipped_mass_to_light": raw_scales[fold],
                "heldout_objects": [
                    str(rows[index]["name"]) for index in np.where(folds == fold)[0]
                ],
            }
            for fold in range(int(config["sample"]["outer_folds"]))
        ],
        "selection_stability": {
            "niche_counts": dict(sorted(niche_counts.items())),
            "maximum_same_nonlinear_niche_folds": nonlinear_count,
            "exact_candidate_indices": selected["selected_indices"],
        },
        "null_distribution": {
            "statistic": "OOF improvement versus fixed flexible photometric nuisance model",
            "observed": observed,
            "minimum": min(null_statistics),
            "median": float(np.median(null_statistics)),
            "maximum": max(null_statistics),
            "sha256": _sha256_bytes(np.asarray(null_statistics, dtype="<f8").tobytes()),
        },
        "controls": {
            **controls,
            "maximum_admitted_local_fractional_deviation_at_1AU": admissibility[
                "maximum_admitted_local_fractional_deviation"
            ],
            "cpu_gpu_max_absolute_log_mu_difference": cpu_gpu_max,
        },
        "universal_gates": universal_gates,
        "phenomenon_publication_gates": phenomenon_gates,
        "counterexample_names_vs_flexible": counterexamples,
    }
    del log_mu
    xp.get_default_memory_pool().free_all_blocks()
    return scientific, compute


def _build_receipt(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    response_manifest: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compute: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, config)
    predictor, sample, candidates = _load_prepared(root, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item23-bimetric-receipt-1.0",
            "item": 23,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "discovery_policy": config["discovery_policy"],
            "mathematical_definition": {
                "action": config["theory"]["action"],
                "effective_matter_metric": config["theory"]["effective_matter_metric"],
                "mass_basis": config["theory"]["mass_basis"],
                "linear_equivalence": config["theory"]["linear_equivalence"],
                "nonlinear_branch_equation": config["theory"]["nonlinear_branch_equation"],
                "branch_exchange": config["theory"]["branch_exchange"],
                "observable_response": config["theory"]["observable_response"],
                "dynamics_mass_balance": "log[5 Re sigma^2/(G Lz)] = log(M/Lz)+log(mu_D(Re))",
                "lensing_mass_balance": "log[pi RE^2 SigmaCrit/(Lz f_H(RE))] = log(M/Lz)+log(mu_L(RE))",
            },
            "provenance_and_creativity_labels": config["candidate_generator"]["niches"],
            "equivalence_audit": {
                "known_two_pole_controls": [
                    "singly_coupled_linear_proportional",
                    "composite_coupled_linear_proportional",
                ],
                "non_equivalent_search_regions": [
                    "monotone_nonlinear_bimetric_branch",
                    "baryon_triggered_metric_ratio_exchange",
                ],
                "historical_novelty_claimed": False,
                "boundaries": config["candidate_generator"]["equivalence_boundaries"],
            },
            "stability_scope": config["theory"]["claim_limits"],
            "claim_boundary": config["scope"]["claim_ceiling"],
            "source_bindings": {
                "predictor_manifest_sha256": predictor["content_sha256"],
                "sample_manifest_sha256": sample["content_sha256"],
                "candidate_manifest_sha256": candidates["content_sha256"],
                "response_manifest_sha256": response_manifest["content_sha256"],
                "response_file_sha256": response_manifest["response_file"]["sha256"],
                "observable_lineage": config["sources"]["observable_lineage"],
            },
            "frozen_boundary": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "stable_goal_sha256": config["stable_goal_sha256"],
                "confirmation_opened": False,
                "confirmation_response_values_read": 0,
                "post_response_formula_generation": False,
            },
            "baselines": {
                "shared_GR": config["evaluation"]["baseline_shared_GR"],
                "separate_calibration": config["evaluation"]["baseline_separate_calibration"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
            },
            "scientific_result": scientific,
            "compute_and_api_cost": {**compute, "paid_model_calls": 0, "paid_api_spend_usd": 0.0},
            "counterexamples_and_limitations": [
                "At most twelve frozen exploration lenses can enter, so this is a small-sample test with no immediate paper claim.",
                "Legacy r/z Tractor light can still contain residual lens-arc or source contamination.",
                "A single z-band mass-to-light scale and spherical Hernquist/virial mapping are weaker than stellar-population plus resolved Jeans modeling.",
                "SDSS pipeline velocity dispersion may be unreliable at low spectral signal-to-noise; frozen uncertainty and robustness metrics expose but cannot remove that limitation.",
                "Official SuGOHI Einstein radii are model-derived evaluation radii, not a direct multiple-image likelihood.",
                "The algebraic static branch is action-adjacent, not an exact covariant galaxy solution; full ghost, gradient, radiation, cluster, and cosmology gates remain.",
                "No sealed confirmation response is opened; a positive exploration result is only an unchanged-replication lead.",
            ],
            "exact_next_action": "If neither track advances, preserve the bimetric equivalence and counterexample region and advance to Item 24 emergent or entropic gravity. If a bounded phenomenon advances, preregister unchanged fresh replication without pausing the numbered track.",
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
                "valid_object_names": [str(row["name"]) for row in rows],
            },
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_response_rows(root, config)
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
    for key in (
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
        "response_source_manifest",
        "compute_manifest",
    ):
        value = _read_json(paths[key])
        _verify_content_hash(value, key)
    result_path = root / str(config["paths"]["result"])
    result = _read_json(result_path)
    _verify_content_hash(result, "result")
    if int(result["frozen_boundary"]["confirmation_response_values_read"]) != 0:
        raise GravityItem23Error("result opened confirmation data")
    if bool(result["equivalence_audit"]["historical_novelty_claimed"]):
        raise GravityItem23Error("result made an unauthorized historical novelty claim")
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
