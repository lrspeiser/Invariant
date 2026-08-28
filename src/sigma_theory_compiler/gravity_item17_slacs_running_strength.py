"""Frozen Item 17 universal distance-running gravity search on fresh SLACS lenses."""

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
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_item16_s4tm_qed_field import (
    _angular_diameter_distances,
    _backend,
    _canonical_bytes,
    _content_hashed,
    _download,
    _fit_offset,
    _format_float,
    _hmac_rank,
    _improvement,
    _mse,
    _parse_vizier_tsv,
    _prior_identity_hits,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _to_numpy,
    _verify_content_hash,
    _write_json,
    hernquist_projected_mass_fraction,
)

CONFIG_PATH = Path("configs/gravity_item17_slacs_running_strength_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item17_slacs_running_strength.py")
DEPENDENCY_PATH = Path("src/sigma_theory_compiler/gravity_item16_s4tm_qed_field.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")


class GravityItem17Error(RuntimeError):
    """Raised when an Item 17 freeze or replay invariant is violated."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    if config.get("schema_version") != "invariant-gravity-item17-slacs-running-strength-config-1.0":
        raise GravityItem17Error("unexpected Item 17 config schema")
    if int(config.get("item", -1)) != 17:
        raise GravityItem17Error("Item 17 config changed item number")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem17Error("post-response running cells entered config")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem17Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem17Error("paid API calls are outside Item 17")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem17Error("stable gravity goal changed")
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
        raise GravityItem17Error(f"{label} has not been bound")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise GravityItem17Error(f"{label} is not an ancestor of HEAD")


def verify_science_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"])
    _require_ancestor(root, commit, "scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem17Error("scientific contract differs from frozen commit")
    for path in (MODULE_PATH, DEPENDENCY_PATH):
        frozen = _git(root, "show", f"{commit}:{path.as_posix()}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(root / path):
            raise GravityItem17Error(f"Item 17 dependency differs from scientific freeze: {path}")


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    return {
        key: base / str(config["paths"][key])
        for key in (
            "predictor_bolton_table4",
            "predictor_grillo_table4",
            "predictor_source_manifest",
            "sample_manifest",
            "candidate_manifest",
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
        "predictor_bolton_table4",
        "predictor_grillo_table4",
        "predictor_source_manifest",
        "sample_manifest",
        "candidate_manifest",
    ):
        repository_path = paths[key].relative_to(root).as_posix()
        frozen = _git(root, "show", f"{commit}:{repository_path}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[key]):
            raise GravityItem17Error(f"{key} differs from sample freeze")


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    positive = np.logspace(
        float(generator["positive_amplitude_log10_min"]),
        float(generator["positive_amplitude_log10_max"]),
        int(generator["positive_amplitude_count"]),
    )
    amplitude = np.concatenate(
        [np.asarray(generator["negative_amplitudes"], dtype=np.float64), positive]
    )
    r0 = np.logspace(
        float(generator["r0_log10_kpc_min"]),
        float(generator["r0_log10_kpc_max"]),
        int(generator["r0_count"]),
    )
    power = np.asarray(generator["powers"], dtype=np.float64)
    saturation = np.asarray(generator["saturation_B"], dtype=np.float64)
    a_grid, r_grid, n_grid, b_grid = np.meshgrid(amplitude, r0, power, saturation, indexing="ij")
    raw = {
        "amplitude": a_grid.reshape(-1),
        "r0_kpc": r_grid.reshape(-1),
        "power": n_grid.reshape(-1),
        "saturation": b_grid.reshape(-1),
    }
    filters = config["pre_response_filters"]
    local_radius = float(filters["local_radius_kpc"])
    domain_max = float(filters["domain_max_kpc"])

    def response(radius: float) -> np.ndarray:
        logarithm = np.log1p((radius / raw["r0_kpc"]) ** raw["power"])
        return 1.0 + raw["amplitude"] * logarithm / (1.0 + raw["saturation"] * logarithm)

    local = response(local_radius)
    far = response(domain_max)
    keep = (
        (np.abs(local - 1.0) <= float(filters["maximum_local_fractional_deviation"]))
        & (local >= float(filters["minimum_G_eff_over_G"]))
        & (far >= float(filters["minimum_G_eff_over_G"]))
        & np.isfinite(local)
        & np.isfinite(far)
    )
    return {key: np.ascontiguousarray(value[keep]) for key, value in raw.items()}


def _raw_candidate_count(config: Mapping[str, Any]) -> int:
    generator = config["candidate_generator"]
    amplitudes = len(generator["negative_amplitudes"]) + int(generator["positive_amplitude_count"])
    return (
        amplitudes
        * int(generator["r0_count"])
        * len(generator["powers"])
        * len(generator["saturation_B"])
    )


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype="<f8")
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _short_name(value: str) -> str:
    text = value.strip().upper().replace("−", "-")
    if not text.startswith("J"):
        text = "J" + text
    return text


def _build_sample(
    bolton_rows: Sequence[Mapping[str, str]],
    grillo_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bolton = {_short_name(str(row["Name"])): row for row in bolton_rows if row.get("Name")}
    objects: list[dict[str, Any]] = []
    mass_models = list(config["physics"]["alternative_stellar_mass_models"])
    for grillo in grillo_rows:
        name = _short_name(str(grillo["SLACS"]))
        source = bolton.get(name)
        if source is None:
            continue
        if str(source["Mph"]) != "E" or str(source["Mul"]) != "S" or str(source["Lens"]) != "A":
            continue
        try:
            values = {
                "z_lens": float(source["zFG"]),
                "z_source": float(source["zBG"]),
                "imag_ab": float(source["Imag"]),
                "galactic_extinction_mag": float(source["AI"]),
                "luminosity_V555_Gsol": float(source["L(V555)"]),
                "reff_arcsec": float(source["Re"]),
                "axis_ratio": float(source["b/a"]),
            }
            masses = {model: float(grillo[model]) * 1e10 for model in mass_models}
        except (TypeError, ValueError):
            continue
        if (
            values["z_source"] <= values["z_lens"]
            or values["z_lens"] <= 0
            or values["reff_arcsec"] <= 0
            or values["axis_ratio"] <= 0
            or any(mass <= 0 or not math.isfinite(mass) for mass in masses.values())
        ):
            continue
        objects.append(
            {
                "name": name,
                "sdss": str(source["SDSS"]),
                **values,
                "stellar_masses_msun": masses,
            }
        )
    expected = int(config["sources"]["expected_joined_objects"])
    if len(objects) != expected:
        raise GravityItem17Error(
            f"expected {expected} predictor-only SLACS joins, found {len(objects)}"
        )
    primary = str(config["physics"]["primary_stellar_mass_model"])
    objects.sort(key=lambda row: (float(row["stellar_masses_msun"][primary]), row["name"]))
    sizes = (14, 14, 14, 15)
    role_key = str(config["sample"]["role_key"])
    confirmation: set[str] = set()
    begin = 0
    for stratum, size in enumerate(sizes):
        group = objects[begin : begin + size]
        begin += size
        for row in group:
            row["mass_stratum"] = stratum
            message = (
                f"{row['name']}|{row['z_lens']:.8f}|{row['z_source']:.8f}|"
                f"{row['stellar_masses_msun'][primary]:.8e}"
            )
            row["role_rank"] = _hmac_rank(role_key, message)
        confirmation.update(
            row["name"] for row in sorted(group, key=lambda row: row["role_rank"])[:3]
        )
    exploration = [row for row in objects if row["name"] not in confirmation]
    fold_key = str(config["sample"]["fold_key"])
    exploration.sort(key=lambda row: _hmac_rank(fold_key, row["name"]))
    for index, row in enumerate(exploration):
        row["role"] = "exploration"
        row["outer_fold"] = index % int(config["sample"]["outer_folds"])
    for row in objects:
        if row["name"] in confirmation:
            row["role"] = "reserved_confirmation"
            row["outer_fold"] = None
    return sorted(objects, key=lambda row: row["name"])


def prepare_predictors(root: Path) -> dict[str, Path]:
    config = load_config(root)
    verify_science_freeze(root, config)
    paths = _source_paths(root, config)
    paths["predictor_bolton_table4"].parent.mkdir(parents=True, exist_ok=True)
    query_bolton = str(config["sources"]["predictor_queries"]["bolton_table4"])
    query_grillo = str(config["sources"]["predictor_queries"]["grillo_table4"])
    bolton_body, bolton_headers = _download(query_bolton)
    grillo_body, grillo_headers = _download(query_grillo)
    bolton_columns = (
        "SDSS",
        "zFG",
        "zBG",
        "Imag",
        "AI",
        "L(V555)",
        "Re",
        "b/a",
        "Mph",
        "Mul",
        "Lens",
        "Name",
    )
    grillo_columns = (
        "SLACS",
        "MSalBC",
        "E_MSalBC",
        "e_MSalBC",
        "MSalM",
        "E_MSalM",
        "e_MSalM",
        "MChaBC",
        "E_MChaBC",
        "e_MChaBC",
        "MKroM",
        "E_MKroM",
        "e_MKroM",
    )
    bolton_rows = _parse_vizier_tsv(bolton_body, bolton_columns)
    grillo_rows = _parse_vizier_tsv(grillo_body, grillo_columns)
    paths["predictor_bolton_table4"].write_bytes(bolton_body)
    paths["predictor_grillo_table4"].write_bytes(grillo_body)
    objects = _build_sample(bolton_rows, grillo_rows, config)
    source_dir = paths["predictor_bolton_table4"].parent
    identities = [row["name"] for row in objects] + [row["sdss"] for row in objects]
    prior_hits = _prior_identity_hits(root, source_dir, identities)
    if prior_hits:
        raise GravityItem17Error(
            f"SLACS identities overlap prior gravity response artifacts: {prior_hits}"
        )
    exploration = [row for row in objects if row["role"] == "exploration"]
    confirmation = [row for row in objects if row["role"] == "reserved_confirmation"]
    sample = _content_hashed(
        {
            "schema_version": "invariant-gravity-item17-slacs-sample-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "selection_used_response_values": False,
            "prior_response_identity_hits": {},
            "counts": {
                "eligible": len(objects),
                "exploration": len(exploration),
                "reserved_confirmation": len(confirmation),
                "response_values_read": 0,
            },
            "fold_counts_exploration": dict(
                sorted(Counter(str(row["outer_fold"]) for row in exploration).items())
            ),
            "objects": [
                {
                    key: (
                        {mass_key: _format_float(mass) for mass_key, mass in value.items()}
                        if key == "stellar_masses_msun"
                        else _format_float(value)
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in row.items()
                }
                for row in objects
            ],
            "claims": {"confirmation_opened": False},
        }
    )
    arrays = generate_candidates(config)
    candidate_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item17-running-candidates-1.0",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "algorithm": config["candidate_generator"]["algorithm"],
            "candidate_digest_sha256": _candidate_digest(arrays),
            "counts": {
                "raw_parameter_cells": _raw_candidate_count(config),
                "physics_admissible_cells": len(arrays["amplitude"]),
                "exact_parameter_equivalence_classes": len(arrays["amplitude"]),
                "post_response_cells": 0,
            },
            "law": config["physics"]["running_law"],
            "creativity_label": config["candidate_generator"]["creativity_label"],
            "pre_response_filters": config["pre_response_filters"],
            "response_values_read": 0,
        }
    )
    predictor_manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item17-slacs-predictors-1.0",
            "catalogs": [
                config["sources"]["bolton_catalog"],
                config["sources"]["grillo_catalog"],
            ],
            "queries": {
                "bolton_table4": {
                    "url": query_bolton,
                    "selected_columns": list(bolton_columns),
                    "rows": len(bolton_rows),
                },
                "grillo_table4": {
                    "url": query_grillo,
                    "selected_columns": list(grillo_columns),
                    "rows": len(grillo_rows),
                },
            },
            "files": [
                {
                    "path": paths["predictor_bolton_table4"].relative_to(root).as_posix(),
                    "sha256": _sha256_file(paths["predictor_bolton_table4"]),
                    "last_modified": bolton_headers.get("last-modified"),
                },
                {
                    "path": paths["predictor_grillo_table4"].relative_to(root).as_posix(),
                    "sha256": _sha256_file(paths["predictor_grillo_table4"]),
                    "last_modified": grillo_headers.get("last-modified"),
                },
            ],
            "response_columns_requested": [],
            "forbidden_columns_read": [],
        }
    )
    _write_json(paths["predictor_source_manifest"], predictor_manifest)
    _write_json(paths["sample_manifest"], sample)
    _write_json(paths["candidate_manifest"], candidate_manifest)
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
    arrays = generate_candidates(config)
    if candidates["candidate_digest_sha256"] != _candidate_digest(arrays):
        raise GravityItem17Error("prepared running candidate digest changed")
    if int(candidates["counts"]["physics_admissible_cells"]) != len(arrays["amplitude"]):
        raise GravityItem17Error("prepared admissible candidate count changed")
    if int(candidates["counts"]["post_response_cells"]) != 0:
        raise GravityItem17Error("prepared manifest admits post-response candidates")
    for item in predictor["files"]:
        if _sha256_file(root / str(item["path"])) != str(item["sha256"]):
            raise GravityItem17Error(f"predictor source changed: {item['path']}")
    if int(sample["counts"]["response_values_read"]) != 0 or bool(
        sample["claims"]["confirmation_opened"]
    ):
        raise GravityItem17Error("sample freeze was contaminated by response values")
    return predictor, sample, candidates


def _response_url(config: Mapping[str, Any], table: str, target_key: str, target: str) -> str:
    columns = ",".join(config["sources"]["response_columns"][table])
    source_table = "table4" if table == "bolton_table4" else "table5"
    query = urllib.parse.urlencode(
        [
            ("-source", f"J/ApJ/682/964/{source_table}"),
            ("-out", columns),
            ("-out.max", "unlimited"),
            (target_key, target),
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
        row["name"] for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    output: list[dict[str, str]] = []
    receipts: list[dict[str, Any]] = []
    for object_row in exploration:
        name = str(object_row["name"])
        sdss = str(object_row["sdss"])
        if name in confirmation:
            raise GravityItem17Error("confirmation response was requested")
        url4 = _response_url(config, "bolton_table4", "SDSS", sdss)
        url5 = _response_url(config, "bolton_table5", "Name", name)
        body4, headers4 = _download(url4)
        body5, headers5 = _download(url5)
        rows4 = _parse_vizier_tsv(body4, ("SDSS", "Name", "sigma", "e_sigma"))
        rows5 = _parse_vizier_tsv(body5, ("Name", "bSIE", "Good?"))
        if len(rows4) != 1 or len(rows5) != 1:
            raise GravityItem17Error(f"response query did not return one row for {name}")
        if _short_name(rows4[0]["Name"]) != name or _short_name(rows5[0]["Name"]) != name:
            raise GravityItem17Error(f"response identity mismatch for {name}")
        output.append(
            {
                "Name": name,
                "sigma": rows4[0]["sigma"],
                "e_sigma": rows4[0]["e_sigma"],
                "bSIE": rows5[0]["bSIE"],
                "Good": rows5[0]["Good?"],
            }
        )
        receipts.append(
            {
                "name": name,
                "table4_query_sha256": _sha256_bytes(url4.encode()),
                "table4_response_sha256": _sha256_bytes(body4),
                "table4_last_modified": headers4.get("last-modified"),
                "table5_query_sha256": _sha256_bytes(url5.encode()),
                "table5_response_sha256": _sha256_bytes(body5),
                "table5_last_modified": headers5.get("last-modified"),
            }
        )
    output.sort(key=lambda row: row["Name"])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("Name", "sigma", "e_sigma", "bSIE", "Good"),
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(output)
    paths["exploration_responses"].write_text(buffer.getvalue(), encoding="utf-8", newline="")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item17-slacs-response-source-1.0",
            "sample_freeze_commit": config["sample_freeze_commit"],
            "requested_objects": len(exploration),
            "returned_objects": len(output),
            "confirmation_objects_requested": 0,
            "confirmation_response_values_read": 0,
            "post_response_candidate_cells": int(candidates["counts"]["post_response_cells"]),
            "columns": ["Name", "sigma", "e_sigma", "bSIE", "Good"],
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


def _load_response_rows(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _source_paths(root, config)
    _, sample, _ = _load_prepared(root, config)
    response_manifest = _read_json(paths["response_source_manifest"])
    _verify_content_hash(response_manifest, "response source manifest")
    if int(response_manifest["confirmation_response_values_read"]) != 0:
        raise GravityItem17Error("confirmation response was opened")
    if _sha256_file(paths["exploration_responses"]) != str(
        response_manifest["response_file"]["sha256"]
    ):
        raise GravityItem17Error("exploration response file changed")
    with paths["exploration_responses"].open(encoding="utf-8", newline="") as handle:
        response = {row["Name"]: row for row in csv.DictReader(handle, delimiter="\t")}
    rows: list[dict[str, Any]] = []
    mass_models = list(config["physics"]["alternative_stellar_mass_models"])
    for predictor in sample["objects"]:
        if predictor["role"] != "exploration":
            continue
        name = str(predictor["name"])
        observed = response.get(name)
        if observed is None:
            continue
        try:
            sigma = float(observed["sigma"])
            e_sigma = float(observed["e_sigma"])
            theta_ein = float(observed["bSIE"])
            z_lens = float(predictor["z_lens"])
            z_source = float(predictor["z_source"])
            reff_arcsec = float(predictor["reff_arcsec"])
            axis_ratio = float(predictor["axis_ratio"])
            stellar_masses = {
                model: float(predictor["stellar_masses_msun"][model]) for model in mass_models
            }
        except (TypeError, ValueError):
            continue
        quality = (
            observed["Good"] == "Yes"
            and sigma > 0
            and e_sigma > 0
            and e_sigma < sigma
            and theta_ein > 0
            and reff_arcsec > 0
            and z_source > z_lens > 0
            and all(value > 0 and math.isfinite(value) for value in stellar_masses.values())
        )
        if not quality:
            continue
        d_lens, d_source, d_lens_source = _angular_diameter_distances(z_lens, z_source, config)
        arcsec = float(config["physics"]["constants"]["arcsec_to_radian"])
        reff_kpc = reff_arcsec * arcsec * d_lens
        rein_kpc = theta_ein * arcsec * d_lens
        re_over_a = float(config["physics"]["hernquist_re_over_a"])
        projected_fraction = float(
            hernquist_projected_mass_fraction(rein_kpc / (reff_kpc / re_over_a))
        )
        c = float(config["physics"]["constants"]["c_km_s"])
        gravitational = float(config["physics"]["constants"]["G_kpc_km2_s2_Msun"])
        sigma_critical = (c**2 / (4.0 * math.pi * gravitational)) * (
            d_source / (d_lens * d_lens_source)
        )
        lens_required_mass = math.pi * rein_kpc**2 * sigma_critical
        virial = float(config["physics"]["dynamical_virial_coefficient"])
        responses: dict[str, list[float]] = {}
        for model, stellar_mass in stellar_masses.items():
            responses[model] = [
                math.log(virial * reff_kpc * sigma**2 / (gravitational * stellar_mass)),
                math.log(lens_required_mass / (stellar_mass * projected_fraction)),
            ]
        rows.append(
            {
                "name": name,
                "fold": int(predictor["outer_fold"]),
                "z_lens": z_lens,
                "z_source": z_source,
                "axis_ratio": axis_ratio,
                "reff_kpc": reff_kpc,
                "rein_kpc": rein_kpc,
                "sigma": sigma,
                "e_sigma": e_sigma,
                "theta_ein_arcsec": theta_ein,
                "stellar_masses_msun": stellar_masses,
                "responses": responses,
            }
        )
    return sorted(rows, key=lambda row: row["name"]), response_manifest


def _candidate_log_mu(
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    amplitude = xp.asarray(arrays["amplitude"][begin:end])[:, None, None]
    r0 = xp.asarray(arrays["r0_kpc"][begin:end])[:, None, None]
    power = xp.asarray(arrays["power"][begin:end])[:, None, None]
    saturation = xp.asarray(arrays["saturation"][begin:end])[:, None, None]
    radii = xp.asarray(
        np.asarray([[row["reff_kpc"], row["rein_kpc"]] for row in rows], dtype=np.float64)
    )[None, :, :]
    logarithm = xp.log1p(xp.power(radii / r0, power))
    response = 1.0 + amplitude * logarithm / (1.0 + saturation * logarithm)
    return xp.log(response)


def _build_log_mu_matrix(
    config: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    xp: Any,
) -> Any:
    batch = int(config["evaluation"]["candidate_batch_size"])
    pieces = []
    for begin in range(0, len(arrays["amplitude"]), batch):
        end = min(begin + batch, len(arrays["amplitude"]))
        pieces.append(_candidate_log_mu(arrays, rows, begin, end, xp))
    return xp.concatenate(pieces, axis=0)


def _screen_log_mu(
    log_mu: Any,
    y: np.ndarray,
    folds: np.ndarray,
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    bounds = tuple(float(value) for value in config["physics"]["shared_stellar_mass_scale_bounds"])
    y_device = xp.asarray(y)
    prediction = np.empty_like(y)
    selected: list[int] = []
    offsets: list[float] = []
    raw_offsets: list[float] = []
    training_mse: list[float] = []
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        heldout = np.where(folds == fold)[0]
        residual = y_device[None, train, :] - log_mu[:, train, :]
        raw = xp.mean(residual, axis=(1, 2))
        fitted = xp.clip(raw, math.log(bounds[0]), math.log(bounds[1]))
        mse = xp.mean((residual - fitted[:, None, None]) ** 2, axis=(1, 2))
        index = int(_to_numpy(xp.argmin(mse), xp))
        selected.append(index)
        raw_offsets.append(float(_to_numpy(raw[index], xp)))
        offsets.append(float(_to_numpy(fitted[index], xp)))
        training_mse.append(float(_to_numpy(mse[index], xp)))
        prediction[heldout] = _to_numpy(log_mu[index, heldout], xp) + offsets[-1]
    return {
        "prediction": prediction,
        "selected_indices": selected,
        "log_mass_offsets": offsets,
        "raw_log_mass_offsets": raw_offsets,
        "training_mse": training_mse,
    }


def _fixed_index_predictions(
    log_mu: Any,
    y: np.ndarray,
    folds: np.ndarray,
    indices: Sequence[int],
    config: Mapping[str, Any],
    xp: Any,
) -> np.ndarray:
    bounds = tuple(float(value) for value in config["physics"]["shared_stellar_mass_scale_bounds"])
    prediction = np.empty_like(y)
    for fold, index in enumerate(indices):
        train = np.where(folds != fold)[0]
        heldout = np.where(folds == fold)[0]
        formula = _to_numpy(log_mu[index], xp)
        _, offset = _fit_offset((y[train] - formula[train]).reshape(-1), bounds)
        prediction[heldout] = formula[heldout] + offset
    return prediction


def _baseline_predictions(
    y: np.ndarray,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    mass_model: str,
) -> dict[str, np.ndarray]:
    bounds = tuple(float(value) for value in config["physics"]["shared_stellar_mass_scale_bounds"])
    shared = np.empty_like(y)
    separate = np.empty_like(y)
    flexible = np.empty_like(y)
    feature = np.asarray(
        [
            [
                math.log10(row["stellar_masses_msun"][mass_model]),
                math.log(row["reff_kpc"]),
                row["z_lens"],
                row["axis_ratio"],
            ]
            for row in rows
        ]
    )
    alpha = float(config["evaluation"]["ridge_alpha"])
    for fold in range(int(config["sample"]["outer_folds"])):
        train = np.where(folds != fold)[0]
        heldout = np.where(folds == fold)[0]
        _, shared_offset = _fit_offset(y[train].reshape(-1), bounds)
        shared[heldout] = shared_offset
        for channel in range(2):
            _, offset = _fit_offset(y[train, channel], bounds)
            separate[heldout, channel] = offset
        mean = feature[train].mean(axis=0)
        scale = feature[train].std(axis=0)
        scale[scale == 0] = 1.0
        train_design = np.column_stack([np.ones(len(train)), (feature[train] - mean) / scale])
        held_design = np.column_stack([np.ones(len(heldout)), (feature[heldout] - mean) / scale])
        penalty = np.diag([0.0] + [alpha] * feature.shape[1])
        for channel in range(2):
            coefficient = np.linalg.solve(
                train_design.T @ train_design + penalty,
                train_design.T @ y[train, channel],
            )
            flexible[heldout, channel] = held_design @ coefficient
    return {"shared_GR": shared, "separate_calibration": separate, "flexible_nuisance": flexible}


def _synthetic_controls(
    log_mu: Any,
    folds: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    index = 17017 % int(log_mu.shape[0])
    injected_formula = _to_numpy(log_mu[index], xp)
    pattern = np.asarray(
        [[math.sin(item + 0.2), math.cos(item + 0.4)] for item in range(len(rows))]
    )
    injected_y = math.log(1.2) + injected_formula + 0.002 * pattern
    screen = _screen_log_mu(log_mu, injected_y, folds, config, xp)
    baseline = _baseline_predictions(
        injected_y, folds, rows, config, str(config["physics"]["primary_stellar_mass_model"])
    )["shared_GR"]
    candidate_mse = _mse(injected_y, screen["prediction"])
    gr_mse = _mse(injected_y, baseline)
    pure_gr = np.full((len(rows), 2), math.log(1.2))
    gr_screen = _screen_log_mu(log_mu, pure_gr, folds, config, xp)
    gr_baseline = _baseline_predictions(
        pure_gr, folds, rows, config, str(config["physics"]["primary_stellar_mass_model"])
    )["shared_GR"]
    return {
        "injection_candidate_index": index,
        "injection_candidate_mse": candidate_mse,
        "injection_GR_mse": gr_mse,
        "injection_improves_over_GR": candidate_mse < gr_mse,
        "GR_candidate_mse": _mse(pure_gr, gr_screen["prediction"]),
        "GR_baseline_mse": _mse(pure_gr, gr_baseline),
        "GR_control_prefers_running": _mse(pure_gr, gr_screen["prediction"])
        < _mse(pure_gr, gr_baseline) - 1e-18,
    }


def _weighted_mse(
    y: np.ndarray,
    prediction: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float:
    mass_error = math.log(10.0) * float(config["evaluation"]["stellar_mass_systematic_dex"])
    lens_error = float(config["evaluation"]["lens_radius_fractional_uncertainty"])
    errors = np.asarray(
        [
            [
                math.hypot(2.0 * row["e_sigma"] / row["sigma"], mass_error),
                math.hypot(2.0 * lens_error, mass_error),
            ]
            for row in rows
        ]
    )
    weights = 1.0 / errors**2
    return float(np.sum(weights * (y - prediction) ** 2) / np.sum(weights))


def _selected_cell(index: int, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "candidate_index": int(index),
        "amplitude_A": _format_float(arrays["amplitude"][index]),
        "amplitude_sign": "grow" if arrays["amplitude"][index] > 0 else "shrink",
        "transition_r0_kpc": _format_float(arrays["r0_kpc"][index]),
        "power_n": _format_float(arrays["power"][index]),
        "saturation_B": _format_float(arrays["saturation"][index]),
        "large_distance_limit_Geff_over_G": _format_float(
            1.0 + arrays["amplitude"][index] / arrays["saturation"][index]
        ),
        "creativity_label": "known_family_combination",
    }


def _evaluation(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    record_compute: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = generate_candidates(config)
    xp, backend, device = _backend()
    if len(rows) < int(config["gates"]["minimum_complete_exploration_objects"]):
        raise GravityItem17Error("too few valid exploration lenses")
    primary = str(config["physics"]["primary_stellar_mass_model"])
    y = np.asarray([row["responses"][primary] for row in rows])
    folds = np.asarray([row["fold"] for row in rows], dtype=np.int64)
    if set(folds.tolist()) != set(range(int(config["sample"]["outer_folds"]))):
        raise GravityItem17Error("exploration folds are incomplete")
    xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    log_mu = _build_log_mu_matrix(config, arrays, rows, xp)
    xp.cuda.Stream.null.synchronize()
    matrix_seconds = time.perf_counter() - start
    crosscheck_count = min(
        int(config["evaluation"]["cpu_crosscheck_candidates"]), len(arrays["amplitude"])
    )
    cpu = _candidate_log_mu(arrays, rows, 0, crosscheck_count, np)
    gpu = _to_numpy(log_mu[:crosscheck_count], xp)
    crosscheck = float(np.max(np.abs(cpu - gpu)))
    controls = _synthetic_controls(log_mu, folds, rows, config, xp)
    start_screen = time.perf_counter()
    selected = _screen_log_mu(log_mu, y, folds, config, xp)
    baselines = _baseline_predictions(y, folds, rows, config, primary)
    candidate_mse = _mse(y, selected["prediction"])
    shared_mse = _mse(y, baselines["shared_GR"])
    separate_mse = _mse(y, baselines["separate_calibration"])
    flexible_mse = _mse(y, baselines["flexible_nuisance"])
    observed_statistic = _improvement(separate_mse, candidate_mse)
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    trials = int(config["evaluation"]["permutation_trials"])
    null: list[float] = []
    for trial in range(trials):
        permuted_y = y[random.permutation(len(rows))]
        permuted = _screen_log_mu(log_mu, permuted_y, folds, config, xp)
        permuted_baseline = _baseline_predictions(permuted_y, folds, rows, config, primary)[
            "separate_calibration"
        ]
        null.append(
            _improvement(
                _mse(permuted_y, permuted_baseline),
                _mse(permuted_y, permuted["prediction"]),
            )
        )
        if record_compute and (trial + 1) % 10 == 0:
            print(f"Item 17 selection-aware null {trial + 1}/{trials}", flush=True)
    xp.cuda.Stream.null.synchronize()
    screen_seconds = time.perf_counter() - start_screen
    permutation_p = (1 + sum(value >= observed_statistic for value in null)) / (trials + 1)
    cells = [_selected_cell(index, arrays) for index in selected["selected_indices"]]
    signs = [cell["amplitude_sign"] for cell in cells]
    log_r0 = np.log10([float(cell["transition_r0_kpc"]) for cell in cells])
    median_r0 = float(np.median(log_r0))
    clustered_folds = int(
        np.sum(
            np.abs(log_r0 - median_r0)
            <= float(config["gates"]["maximum_transition_scale_distance_from_median_dex"])
        )
    )
    channel_metrics: dict[str, Any] = {}
    for channel_index, label in enumerate(("stellar_dynamics", "Einstein_radius_lensing")):
        candidate_channel = float(
            np.mean((y[:, channel_index] - selected["prediction"][:, channel_index]) ** 2)
        )
        shared_channel = float(
            np.mean((y[:, channel_index] - baselines["shared_GR"][:, channel_index]) ** 2)
        )
        separate_channel = float(
            np.mean(
                (y[:, channel_index] - baselines["separate_calibration"][:, channel_index]) ** 2
            )
        )
        channel_metrics[label] = {
            "candidate_mse": candidate_channel,
            "shared_GR_mse": shared_channel,
            "separate_calibration_mse": separate_channel,
            "improvement_vs_shared_GR": _improvement(shared_channel, candidate_channel),
            "improvement_vs_separate_calibration": _improvement(
                separate_channel, candidate_channel
            ),
        }
    strata: dict[str, Any] = {}
    for value, label in (
        (
            np.asarray([math.log10(row["stellar_masses_msun"][primary]) for row in rows]),
            "stellar_mass",
        ),
        (np.asarray([row["reff_kpc"] for row in rows]), "effective_radius"),
    ):
        median = float(np.median(value))
        for side, indices in (
            ("low", np.where(value <= median)[0]),
            ("high", np.where(value > median)[0]),
        ):
            candidate_slice = _mse(y, selected["prediction"], indices)
            shared_slice = _mse(y, baselines["shared_GR"], indices)
            strata[f"{label}_{side}"] = {
                "objects": len(indices),
                "candidate_mse": candidate_slice,
                "shared_GR_mse": shared_slice,
                "improvement_vs_shared_GR": _improvement(shared_slice, candidate_slice),
            }
    alternatives: dict[str, Any] = {}
    for model in config["physics"]["alternative_stellar_mass_models"]:
        model_y = np.asarray([row["responses"][model] for row in rows])
        fixed_prediction = _fixed_index_predictions(
            log_mu,
            model_y,
            folds,
            selected["selected_indices"],
            config,
            xp,
        )
        model_baseline = _baseline_predictions(model_y, folds, rows, config, model)["shared_GR"]
        fixed_mse = _mse(model_y, fixed_prediction)
        model_gr_mse = _mse(model_y, model_baseline)
        alternatives[model] = {
            "candidate_mse": fixed_mse,
            "shared_GR_mse": model_gr_mse,
            "improvement_vs_shared_GR": _improvement(model_gr_mse, fixed_mse),
            "formula_reselected": False,
        }
    alternatives_improving = sum(
        value["improvement_vs_shared_GR"] > 0 for value in alternatives.values()
    )
    bounds = tuple(float(value) for value in config["physics"]["shared_stellar_mass_scale_bounds"])
    raw_scales = [math.exp(value) for value in selected["raw_log_mass_offsets"]]
    gates = {
        "minimum_complete_exploration_objects": len(rows)
        >= int(config["gates"]["minimum_complete_exploration_objects"]),
        "confirmation_values_read_zero": True,
        "post_response_candidate_cells_zero": int(
            config["candidate_generator"]["post_response_cells"]
        )
        == 0,
        "synthetic_injection_recovered": bool(controls["injection_improves_over_GR"]),
        "known_GR_control": not bool(controls["GR_control_prefers_running"]),
        "joint_improvement_vs_shared_GR": _improvement(shared_mse, candidate_mse)
        >= float(config["gates"]["minimum_joint_mse_improvement_vs_shared_GR"]),
        "joint_improvement_vs_separate_calibration": observed_statistic
        >= float(config["gates"]["minimum_joint_mse_improvement_vs_separate_calibration"]),
        "joint_improvement_vs_flexible_nuisance": _improvement(flexible_mse, candidate_mse)
        >= float(config["gates"]["minimum_joint_mse_improvement_vs_flexible_nuisance"]),
        "both_channels_improve_vs_shared_GR": all(
            value["improvement_vs_shared_GR"]
            > float(config["gates"]["minimum_each_channel_improvement_vs_shared_GR"])
            for value in channel_metrics.values()
        ),
        "all_mass_and_size_halves_improve_vs_shared_GR": all(
            value["improvement_vs_shared_GR"]
            > float(config["gates"]["minimum_each_mass_and_size_half_improvement_vs_shared_GR"])
            for value in strata.values()
        ),
        "selection_aware_permutation": permutation_p
        <= float(config["gates"]["maximum_selection_aware_permutation_p"]),
        "amplitude_sign_stable": max(Counter(signs).values())
        >= int(config["gates"]["minimum_same_amplitude_sign_folds"]),
        "transition_scale_stable": clustered_folds
        >= int(config["gates"]["minimum_transition_scale_clustered_folds"]),
        "stellar_population_robustness": alternatives_improving
        >= int(config["gates"]["minimum_alternative_mass_models_improving_vs_shared_GR"]),
        "shared_stellar_mass_scale_in_bounds": all(
            bounds[0] <= value <= bounds[1] for value in raw_scales
        ),
    }
    decision = (
        "PASS_ITEM17_UNIVERSAL_RUNNING_STRENGTH_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM17_UNIVERSAL_RUNNING_STRENGTH_EXPLORATION"
    )
    training_evaluations = (
        len(arrays["amplitude"])
        * 2
        * sum(int(np.count_nonzero(folds != fold)) for fold in range(5))
    )
    compute = {
        "schema_version": "invariant-gravity-item17-compute-1.0",
        "backend": backend,
        "device": device,
        "numpy_version": np.__version__,
        "cupy_version": getattr(xp, "__version__", None),
        "raw_parameter_cells": _raw_candidate_count(config),
        "physics_admissible_cells": len(arrays["amplitude"]),
        "objects": len(rows),
        "channels": 2,
        "candidate_observable_matrix_values": len(arrays["amplitude"]) * len(rows) * 2,
        "candidate_training_residual_evaluations_observed": training_evaluations,
        "candidate_training_residual_evaluations_with_nulls": training_evaluations * (trials + 1),
        "matrix_build_seconds_observed": matrix_seconds,
        "screen_and_null_seconds_observed": screen_seconds,
        "cpu_gpu_max_absolute_log_mu_difference": crosscheck,
    }
    scientific = {
        "decision": decision,
        "counts": {
            "valid_exploration_objects": len(rows),
            "response_channels": 2,
            "raw_parameter_cells": _raw_candidate_count(config),
            "physics_admissible_cells": len(arrays["amplitude"]),
            "post_response_candidate_cells": 0,
            "permutation_trials": trials,
            "passed_gates": sum(bool(value) for value in gates.values()),
            "total_gates": len(gates),
        },
        "primary_metrics": {
            "candidate_mse": candidate_mse,
            "shared_GR_mse": shared_mse,
            "separate_calibration_mse": separate_mse,
            "flexible_nuisance_mse": flexible_mse,
            "improvement_vs_shared_GR": _improvement(shared_mse, candidate_mse),
            "improvement_vs_separate_calibration": observed_statistic,
            "improvement_vs_flexible_nuisance": _improvement(flexible_mse, candidate_mse),
            "selection_aware_permutation_p": permutation_p,
        },
        "weighted_robustness": {
            "candidate_mse": _weighted_mse(y, selected["prediction"], rows, config),
            "shared_GR_mse": _weighted_mse(y, baselines["shared_GR"], rows, config),
            "separate_calibration_mse": _weighted_mse(
                y, baselines["separate_calibration"], rows, config
            ),
        },
        "channel_metrics": channel_metrics,
        "stratum_metrics": strata,
        "stellar_population_replays": alternatives,
        "outer_selections": [
            {
                "fold": fold,
                "cell": cells[fold],
                "training_mse": selected["training_mse"][fold],
                "stellar_mass_scale": math.exp(selected["log_mass_offsets"][fold]),
                "unclipped_stellar_mass_scale": raw_scales[fold],
                "heldout_objects": [rows[index]["name"] for index in np.where(folds == fold)[0]],
            }
            for fold in range(5)
        ],
        "selection_stability": {
            "amplitude_sign_counts": dict(sorted(Counter(signs).items())),
            "transition_log10_r0_kpc": log_r0.tolist(),
            "median_transition_log10_r0_kpc": median_r0,
            "transition_scale_clustered_folds": clustered_folds,
            "exact_candidate_indices": selected["selected_indices"],
        },
        "null_distribution": {
            "observed": observed_statistic,
            "minimum": min(null),
            "median": float(np.median(null)),
            "maximum": max(null),
            "sha256": _sha256_bytes(np.asarray(null, dtype="<f8").tobytes()),
        },
        "pipeline_controls": {**controls, "cpu_gpu_max_absolute_log_mu_difference": crosscheck},
        "gates": gates,
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
            "schema_version": "invariant-gravity-item17-slacs-running-strength-receipt-1.0",
            "item": 17,
            "title": config["title"],
            "hypothesis": config["hypothesis"],
            "provenance_and_creativity_label": config["candidate_generator"]["creativity_label"],
            "mathematical_definition": {
                "running_law": config["physics"]["running_law"],
                "matter_light_relation": config["physics"]["matter_light_relation"],
                "dynamical_response": "log[5 Re sigma^2/(G Mstar)]",
                "lensing_response": "log[pi REin^2 SigmaCrit/(Mstar fHernquist(REin))]",
            },
            "dimensional_and_symmetry_checks": {
                "running_response_dimensionless": True,
                "radius_ratio_dimensionless": True,
                "static_spherical_parity_even": True,
                "short_distance_GR_limit_enforced": True,
                "large_distance_response_bounded": True,
                "positive_Geff_1AU_to_10Mpc": True,
                "fixed_no_slip_matter_light_coupling": True,
                "microscopic_beta_function_derived": False,
                "covariant_action_or_ghost_freedom_proven": False,
            },
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
                "shared_GR": config["evaluation"]["baseline_shared_imf"],
                "separate_calibration": config["evaluation"]["baseline_separate_calibration"],
                "flexible_nuisance": config["evaluation"]["baseline_flexible_nuisance"],
            },
            "scientific_result": scientific,
            "compute_and_api_cost": {**compute, "paid_model_calls": 0, "paid_api_spend_usd": 0.0},
            "counterexamples_and_limitations": [
                "The bounded logarithmic law is a phenomenological known-family combination, not a derived quantum-gravity beta function.",
                "Spherical Hernquist and virial approximations are not resolved orbital or Jeans models.",
                "The SIE Einstein radius is image-model derived rather than a direct image likelihood.",
                "Four stellar-population mass estimates expose but do not eliminate IMF and population-synthesis uncertainty.",
                "No-slip running cannot represent a true scalar/vector polarization difference.",
                "Local positivity and a 1-AU limit do not prove conservation, causality, stability, or a viable cosmology.",
                "No sealed confirmation response is opened.",
            ],
            "result": scientific["decision"].split("_ITEM17")[0],
            "exact_next_action": "Advance to Item 18 gravitational antiscreening with the exact Item 17 running family retained as a known-family comparator; do not open the twelve SLACS confirmations.",
            "reproducibility": {
                "config_path": CONFIG_PATH.as_posix(),
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "module_path": MODULE_PATH.as_posix(),
                "module_sha256": _sha256_file(root / MODULE_PATH),
                "dependency_path": DEPENDENCY_PATH.as_posix(),
                "dependency_sha256": _sha256_file(root / DEPENDENCY_PATH),
                "compute_manifest_path": paths["compute_manifest"].relative_to(root).as_posix(),
                "valid_object_names": [row["name"] for row in rows],
            },
        }
    )


def run_experiment(root: Path) -> Path:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_response_rows(root, config)
    scientific, compute_raw = _evaluation(config, rows, record_compute=True)
    paths = _source_paths(root, config)
    compute = _content_hashed(compute_raw)
    _write_json(paths["compute_manifest"], compute)
    receipt = _build_receipt(root, config, rows, response_manifest, scientific, compute)
    result = root / str(config["paths"]["result"])
    _write_json(result, receipt)
    return result


def replay(root: Path, receipt_path: Path | None = None) -> dict[str, Any]:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    rows, response_manifest = _load_response_rows(root, config)
    paths = _source_paths(root, config)
    compute = _read_json(paths["compute_manifest"])
    _verify_content_hash(compute, "compute manifest")
    scientific, _ = _evaluation(config, rows, record_compute=False)
    rebuilt = _build_receipt(root, config, rows, response_manifest, scientific, compute)
    expected_path = receipt_path or root / str(config["paths"]["result"])
    expected = _read_json(expected_path)
    _verify_content_hash(expected, "Item 17 receipt")
    if _canonical_bytes(rebuilt) != _canonical_bytes(expected):
        raise GravityItem17Error("Item 17 replay differs from committed receipt")
    return {
        "status": "PASS",
        "receipt": expected_path.relative_to(root).as_posix(),
        "content_sha256": rebuilt["content_sha256"],
        "decision": rebuilt["scientific_result"]["decision"],
    }


def verify_pre_response(root: Path) -> dict[str, Any]:
    config = load_config(root)
    verify_science_freeze(root, config)
    verify_sample_freeze(root, config)
    predictor, sample, candidates = _load_prepared(root, config)
    return {
        "status": "PASS",
        "scientific_freeze_commit": config["scientific_freeze_commit"],
        "sample_freeze_commit": config["sample_freeze_commit"],
        "predictor_manifest_sha256": predictor["content_sha256"],
        "sample_manifest_sha256": sample["content_sha256"],
        "candidate_manifest_sha256": candidates["content_sha256"],
        "physics_admissible_cells": candidates["counts"]["physics_admissible_cells"],
        "exploration_objects": sample["counts"]["exploration"],
        "reserved_confirmation_objects": sample["counts"]["reserved_confirmation"],
        "response_values_read": sample["counts"]["response_values_read"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-predictors")
    commands.add_parser("verify-pre-response")
    commands.add_parser("fetch-responses")
    commands.add_parser("run")
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare-predictors":
        paths = prepare_predictors(root)
        print(
            json.dumps(
                {
                    "sample_manifest": paths["sample_manifest"].relative_to(root).as_posix(),
                    "candidate_manifest": paths["candidate_manifest"].relative_to(root).as_posix(),
                    "response_values_read": 0,
                },
                sort_keys=True,
            )
        )
    elif args.command == "verify-pre-response":
        print(json.dumps(verify_pre_response(root), sort_keys=True))
    elif args.command == "fetch-responses":
        print(fetch_responses(root).relative_to(root).as_posix())
    elif args.command == "run":
        print(run_experiment(root).relative_to(root).as_posix())
    elif args.command == "replay":
        receipt = args.receipt
        if receipt is not None and not receipt.is_absolute():
            receipt = root / receipt
        print(json.dumps(replay(root, receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
