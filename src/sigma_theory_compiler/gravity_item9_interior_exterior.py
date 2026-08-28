"""Fresh PROBES test of nonlocal interior/exterior galaxy operators for Item 9."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import gravity_item5_pressure_cross_support as cvcore
from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item9_interior_exterior_probes_v1.json"
SCIENTIFIC_FREEZE_COMMIT = "bcf9df003f8c311ef07f591d46e3140fa3d4b889"


class GravityItem9InteriorExteriorError(RuntimeError):
    """Raised when an Item 9 scientific or source boundary drifts."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _md5_bytes(payload: bytes) -> str:
    return hashlib.md5(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem9InteriorExteriorError("non-finite metric")
    return f"{float(value):.12e}"


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = cvcore._canonicalize_floats(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item9-interior-exterior-probes-config-1.0"
    ):
        raise GravityItem9InteriorExteriorError("unexpected Item 9 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem9InteriorExteriorError("stable roadmap changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem9InteriorExteriorError("Item 8 synthesis file changed")
    predecessor_receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor_receipt.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem9InteriorExteriorError("Item 8 synthesis content changed")
    if predecessor_receipt.get("decision") != predecessor["required_decision"]:
        raise GravityItem9InteriorExteriorError("Item 8 did not authorize Item 9")
    prior = config["prior_lead"]
    prior_path = root / prior["path"]
    if _sha256_file(prior_path) != prior["file_sha256"]:
        raise GravityItem9InteriorExteriorError("prior focusing result file changed")
    prior_receipt = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior_receipt.get("content_sha256") != prior["content_sha256"]:
        raise GravityItem9InteriorExteriorError("prior focusing result content changed")
    if prior_receipt.get("decision") != prior["required_decision"]:
        raise GravityItem9InteriorExteriorError("unexpected prior focusing decision")

    authorization = config["authorization"]
    forbidden_true = (
        "paid_model_calls_allowed",
        "profile_archive_allowed_before_freeze",
        "reserved_confirmation_rotation_profiles_allowed",
        "model_fits_table_allowed",
        "structural_parameters_table_allowed",
        "dynamical_or_dark_mass_allowed",
        "lensing_mass_allowed",
        "post_response_formula_generation_allowed",
    )
    if any(bool(authorization[name]) for name in forbidden_true):
        raise GravityItem9InteriorExteriorError("Item 9 authorization changed")
    if int(config["prefreeze_audit"]["published_rotation_profile_rows_read"]) != 0:
        raise GravityItem9InteriorExteriorError("prefreeze response boundary changed")
    if int(config["prefreeze_audit"]["profile_archive_bytes_downloaded"]) != 0:
        raise GravityItem9InteriorExteriorError("profile archive opened before freeze")
    grammar = config["operator_grammar"]
    expected_operators = (
        (
            len(grammar["surface_brightness_thresholds_lsun_pc2"])
            + len(grammar["acceleration_thresholds_gbar_over_gdagger"])
            + len(grammar["additional_sources"])
        )
        * len(grammar["log_radius_scales"])
        * len(grammar["modes"])
    )
    expected_cells = (
        expected_operators
        * len(grammar["conditions"])
        * len(grammar["logistic_intercepts"])
        * len(grammar["logistic_slopes"])
    )
    if expected_operators != int(grammar["operator_count"]):
        raise GravityItem9InteriorExteriorError("operator count changed")
    if expected_cells != int(grammar["candidate_formula_cells"]):
        raise GravityItem9InteriorExteriorError("candidate cell count changed")
    if config["derivation"]["feature_builder_accepts_observed_velocity"]:
        raise GravityItem9InteriorExteriorError("feature builder cannot accept velocity")
    return config


def _fetch(url: str, *, attempts: int = 3) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except (OSError, TimeoutError, urllib.error.URLError) as exc:  # pragma: no cover
            error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise GravityItem9InteriorExteriorError(f"source acquisition failed: {url}") from error


def parse_metadata(payload: bytes, config: Mapping[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    if _sha256_bytes(payload) != config["sources"]["metadata_sha256"]:
        raise GravityItem9InteriorExteriorError("PROBES metadata SHA-256 changed")
    if _md5_bytes(payload) != config["sources"]["metadata_md5"]:
        raise GravityItem9InteriorExteriorError("PROBES metadata MD5 changed")
    text = payload.decode("utf-8")
    lines = text.splitlines()
    if len(lines) < 3:
        raise GravityItem9InteriorExteriorError("PROBES metadata is empty")
    rows = list(csv.DictReader(lines[1:]))
    allowed = set(config["prefreeze_audit"]["metadata_columns_read"])
    if not allowed.issubset(rows[0]):
        raise GravityItem9InteriorExteriorError("PROBES metadata schema changed")
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            distance = float(row["distance"])
            distance_error = float(row["distance_e"])
            extinction_r = float(row["ext_r"])
        except (TypeError, ValueError):
            continue
        eligible = (
            math.isfinite(distance)
            and distance > 0
            and math.isfinite(distance_error)
            and distance_error >= 0
            and math.isfinite(extinction_r)
            and all(row[f"has_{band}-band"] == "True" for band in ("g", "r", "z", "w1"))
            and row["AutoProf_flags"] == "-"
            and "SPARC" not in row["RC_survey"].upper()
        )
        if not eligible:
            continue
        result.append(
            {
                "name": row["name"],
                "distance_mpc": distance,
                "distance_error_mpc": distance_error,
                "distance_method": row["distance_method"],
                "rc_survey": row["RC_survey"],
                "primary_rc_survey": row["RC_survey"].split("/")[0],
                "extinction_r_magnitude": extinction_r,
                "bands": ["g", "r", "z", "w1"],
                "autoprof_flags": row["AutoProf_flags"],
            }
        )
    result.sort(key=lambda row: str(row["name"]))
    if len(rows) != int(config["prefreeze_audit"]["metadata_rows"]):
        raise GravityItem9InteriorExteriorError("PROBES metadata row count changed")
    if len(result) != int(config["sample"]["eligible_count"]):
        raise GravityItem9InteriorExteriorError("PROBES eligible count changed")
    if len({row["name"] for row in result}) != len(result):
        raise GravityItem9InteriorExteriorError("PROBES galaxy names are not unique")
    return len(rows), result


def acquire_metadata(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    payload = _fetch(str(config["sources"]["metadata_url"]))
    rows, eligible = parse_metadata(payload, config)
    return _seal(
        {
            "schema_version": "invariant-gravity-item9-probes-metadata-source-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM9_RESPONSE_BLIND_METADATA_ACQUISITION",
            "retrieval": {
                "url": config["sources"]["metadata_url"],
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "md5": _md5_bytes(payload),
                "zenodo_record": config["sources"]["zenodo_record"],
                "zenodo_revision": config["sources"]["zenodo_revision"],
            },
            "counts": {
                "catalog_rows": rows,
                "eligible_rows": len(eligible),
                "rotation_profile_rows_read": 0,
                "profile_archive_bytes_downloaded": 0,
                "model_fit_rows_read": 0,
                "structural_parameter_rows_read": 0,
            },
            "allowed_columns": config["prefreeze_audit"]["metadata_columns_read"],
            "records": eligible,
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_metadata_source(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem9InteriorExteriorError("metadata source content hash changed")
    if manifest.get("decision") != "PASS_ITEM9_RESPONSE_BLIND_METADATA_ACQUISITION":
        raise GravityItem9InteriorExteriorError("metadata source decision changed")
    counts = manifest["counts"]
    if int(counts["eligible_rows"]) != int(config["sample"]["eligible_count"]):
        raise GravityItem9InteriorExteriorError("metadata eligible scope changed")
    forbidden_counts = (
        "rotation_profile_rows_read",
        "profile_archive_bytes_downloaded",
        "model_fit_rows_read",
        "structural_parameter_rows_read",
    )
    if any(int(counts[name]) != 0 for name in forbidden_counts):
        raise GravityItem9InteriorExteriorError("metadata source crossed response boundary")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem9InteriorExteriorError("metadata source contains an overclaim")


def write_metadata_source(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    manifest = acquire_metadata(root)
    validate_metadata_source(manifest, config)
    path = root / config["metadata_source_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def _distance_bin(distance: float, boundaries: Sequence[float]) -> int:
    return sum(distance > float(boundary) for boundary in boundaries)


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    source_path = root / config["metadata_source_output"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_metadata_source(source, config)
    sample = config["sample"]
    salt = str(sample["selection_salt"])
    cells: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in source["records"]:
        distance_bin = _distance_bin(
            float(row["distance_mpc"]), sample["distance_bin_upper_mpc"]
        )
        digest = hashlib.sha256(f"{salt}|{row['name']}".encode()).hexdigest()
        cells[(str(row["primary_rc_survey"]), distance_bin)].append((digest, row))
    roles: dict[str, str] = {}
    cell_counts: list[dict[str, Any]] = []
    for (survey, distance_bin), rows in sorted(cells.items()):
        ordered = sorted(rows, key=lambda item: (item[0], str(item[1]["name"])))
        confirmation_count = round(len(ordered) / 4)
        confirmation = {str(row["name"]) for _digest, row in ordered[:confirmation_count]}
        for _digest, row in ordered:
            roles[str(row["name"])] = (
                "reserved_confirmation" if row["name"] in confirmation else "exploration"
            )
        cell_counts.append(
            {
                "primary_rc_survey": survey,
                "distance_bin": distance_bin,
                "all": len(ordered),
                "exploration": len(ordered) - confirmation_count,
                "reserved_confirmation": confirmation_count,
            }
        )
    exploration = sorted(name for name, role in roles.items() if role == "exploration")
    confirmation = sorted(
        name for name, role in roles.items() if role == "reserved_confirmation"
    )
    assignments = cvcore.assign_folds(
        exploration,
        salt=str(sample["outer_fold_salt"]),
        folds=int(config["evaluation"]["outer_folds"]),
    )
    objects = []
    for row in source["records"]:
        name = str(row["name"])
        objects.append(
            {
                **row,
                "role": roles[name],
                "outer_fold": assignments.get(name),
                "distance_bin": _distance_bin(
                    float(row["distance_mpc"]), sample["distance_bin_upper_mpc"]
                ),
                "selection_digest": hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(),
            }
        )
    objects.sort(key=lambda row: (str(row["role"]), str(row["name"])))
    return _seal(
        {
            "schema_version": "invariant-gravity-item9-probes-sample-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM9_RESPONSE_BLIND_PROBES_SAMPLE",
            "metadata_binding": {
                "path": config["metadata_source_output"],
                "file_sha256": _sha256_file(source_path),
                "content_sha256": source["content_sha256"],
            },
            "selection": {
                "eligibility_rule": sample["eligibility_rule"],
                "stratification_rule": sample["stratification_rule"],
                "salt": salt,
                "selection_used_rotation_values": False,
                "selection_used_rotation_point_counts": False,
                "selection_used_profile_archive": False,
            },
            "counts": {
                "eligible": len(objects),
                "exploration": len(exploration),
                "reserved_confirmation": len(confirmation),
                "rotation_profile_rows_read": 0,
                "profile_archive_bytes_downloaded": 0,
            },
            "cells": cell_counts,
            "objects": objects,
            "prefreeze_boundary": {
                "exploration_rotation_profiles_blinded": True,
                "reserved_confirmation_rotation_profiles_blinded": True,
                "model_fits_table_read": False,
                "structural_parameters_table_read": False,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_sample_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem9InteriorExteriorError("sample content hash changed")
    counts = manifest["counts"]
    expected = config["sample"]
    if int(counts["eligible"]) != int(expected["eligible_count"]):
        raise GravityItem9InteriorExteriorError("sample eligible count changed")
    if int(counts["exploration"]) != int(expected["exploration_count"]):
        raise GravityItem9InteriorExteriorError("sample exploration count changed")
    if int(counts["reserved_confirmation"]) != int(expected["reserved_confirmation_count"]):
        raise GravityItem9InteriorExteriorError("sample confirmation count changed")
    if int(counts["rotation_profile_rows_read"]) != 0:
        raise GravityItem9InteriorExteriorError("sample opened rotation rows")
    if int(counts["profile_archive_bytes_downloaded"]) != 0:
        raise GravityItem9InteriorExteriorError("sample downloaded profile archive")
    names = [str(row["name"]) for row in manifest["objects"]]
    if len(names) != len(set(names)):
        raise GravityItem9InteriorExteriorError("sample identities are duplicated")
    roles = Counter(str(row["role"]) for row in manifest["objects"])
    if roles != {
        "exploration": int(expected["exploration_count"]),
        "reserved_confirmation": int(expected["reserved_confirmation_count"]),
    }:
        raise GravityItem9InteriorExteriorError("sample roles changed")
    folds = Counter(
        int(row["outer_fold"])
        for row in manifest["objects"]
        if row["role"] == "exploration"
    )
    if set(folds) != set(range(int(config["evaluation"]["outer_folds"]))):
        raise GravityItem9InteriorExteriorError("outer fold coverage changed")
    if max(folds.values()) - min(folds.values()) > 1:
        raise GravityItem9InteriorExteriorError("outer folds are imbalanced")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem9InteriorExteriorError("sample contains an overclaim")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    manifest = build_sample_manifest(root)
    validate_sample_manifest(manifest, config)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def _number_id(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace(".", "p")


def operator_definitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grammar = config["operator_grammar"]
    sources: list[tuple[str, float | None, str, str]] = []
    for threshold in grammar["surface_brightness_thresholds_lsun_pc2"]:
        sources.append(
            (
                "surface_brightness",
                float(threshold),
                "COMBINATION",
                "prior_repository_baryonic_focusing_combination",
            )
        )
    for threshold in grammar["acceleration_thresholds_gbar_over_gdagger"]:
        sources.append(
            (
                "acceleration",
                float(threshold),
                "COMBINATION",
                "prior_repository_baryonic_focusing_combination",
            )
        )
    labels = {
        "enclosed_mass_fraction": (
            "KNOWN_FAMILY_COMBINATION",
            "known_cumulative_mass_balance_combined_with_nonlocal_kernel",
        ),
        "outer_inverse_radius_potential": (
            "UNRESOLVED",
            "potentially_new_outer_potential_focusing_synthesis",
        ),
        "outer_radius_lever": (
            "UNRESOLVED",
            "potentially_new_outer_lever_arm_synthesis",
        ),
    }
    for source in grammar["additional_sources"]:
        authoritative, proposer = labels[str(source)]
        sources.append((str(source), None, authoritative, proposer))
    result: list[dict[str, Any]] = []
    for source, threshold, authoritative, proposer in sources:
        source_id = source if threshold is None else f"{source}:q{_number_id(threshold)}"
        for scale in grammar["log_radius_scales"]:
            for mode in grammar["modes"]:
                operator_id = (
                    f"focus:{source_id}:ell{_number_id(float(scale))}:{mode}"
                )
                result.append(
                    {
                        "operator_id": operator_id,
                        "source": source,
                        "threshold": threshold,
                        "log_radius_scale": float(scale),
                        "mode": mode,
                        "qualifying": True,
                        "authoritative_origin_status": authoritative,
                        "proposer_origin_label": proposer,
                        "historical_novelty_claimed": False,
                    }
                )
    result.sort(key=lambda row: str(row["operator_id"]))
    if len(result) != int(grammar["operator_count"]):
        raise GravityItem9InteriorExteriorError("generated operator count changed")
    return result


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    grammar = config["operator_grammar"]
    cells: list[dict[str, Any]] = []
    equivalence: dict[tuple[Any, ...], str] = {}
    origin_counts: Counter[str] = Counter()
    prior_count = 0
    prior = config["prior_lead"]["exact_cell"]
    for operator in operator_definitions(config):
        for condition in grammar["conditions"]:
            for intercept in grammar["logistic_intercepts"]:
                for slope in grammar["logistic_slopes"]:
                    equivalence_key = (
                        operator["operator_id"],
                        "constant_condition" if float(slope) == 0 else str(condition),
                        float(intercept),
                        float(slope),
                    )
                    if equivalence_key not in equivalence:
                        equivalence[equivalence_key] = (
                            f"eq-{len(equivalence):05d}-"
                            f"{hashlib.sha256(repr(equivalence_key).encode()).hexdigest()[:12]}"
                        )
                    cell_id = (
                        f"cell:{operator['operator_id']}:{condition}:"
                        f"a{_number_id(float(intercept))}:b{_number_id(float(slope))}"
                    )
                    exact_prior = (
                        operator["source"] == prior["source"]
                        and float(operator["threshold"] or -1) == float(prior["threshold"])
                        and float(operator["log_radius_scale"])
                        == float(prior["log_radius_scale"])
                        and operator["mode"] == prior["mode"]
                        and condition == prior["condition"]
                        and float(intercept) == float(prior["intercept"])
                        and float(slope) == float(prior["slope"])
                    )
                    prior_count += int(exact_prior)
                    origin_counts[str(operator["authoritative_origin_status"])] += 1
                    cells.append(
                        {
                            "ordinal": len(cells),
                            "candidate_id": cell_id,
                            "operator_id": operator["operator_id"],
                            "condition": condition,
                            "intercept": float(intercept),
                            "slope": float(slope),
                            "alpha_max": float(grammar["alpha_max"]),
                            "equivalence_class": equivalence[equivalence_key],
                            "exact_prior_focusing_cell": exact_prior,
                            "qualifying": True,
                            "authoritative_origin_status": operator[
                                "authoritative_origin_status"
                            ],
                            "proposer_origin_label": operator["proposer_origin_label"],
                            "historical_novelty_claimed": False,
                        }
                    )
    if len(cells) != int(grammar["candidate_formula_cells"]):
        raise GravityItem9InteriorExteriorError("generated candidate count changed")
    if prior_count != int(grammar["prior_exact_cell_count"]):
        raise GravityItem9InteriorExteriorError("exact prior cell multiplicity changed")
    return _seal(
        {
            "schema_version": "invariant-gravity-item9-candidate-manifest-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM9_RESPONSE_BLIND_CANDIDATE_GENERATION",
            "grammar": grammar,
            "operators": operator_definitions(config),
            "counts": {
                "operators": len(operator_definitions(config)),
                "candidate_formula_cells": len(cells),
                "declared_equivalence_classes": len(equivalence),
                "exact_prior_focusing_cells": prior_count,
                "post_response_formula_cells": 0,
                "origin_status_cells": dict(sorted(origin_counts.items())),
                "observed_rotation_values_read": 0,
                "profile_archive_bytes_downloaded": 0,
                "paid_model_calls": 0,
            },
            "cells": cells,
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_candidate_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem9InteriorExteriorError("candidate manifest content hash changed")
    counts = manifest["counts"]
    if int(counts["operators"]) != int(config["operator_grammar"]["operator_count"]):
        raise GravityItem9InteriorExteriorError("candidate operator count changed")
    if int(counts["candidate_formula_cells"]) != int(
        config["operator_grammar"]["candidate_formula_cells"]
    ):
        raise GravityItem9InteriorExteriorError("candidate cell count changed")
    if int(counts["exact_prior_focusing_cells"]) != 1:
        raise GravityItem9InteriorExteriorError("prior focusing cell is not unique")
    forbidden = (
        "post_response_formula_cells",
        "observed_rotation_values_read",
        "profile_archive_bytes_downloaded",
        "paid_model_calls",
    )
    if any(int(counts[name]) != 0 for name in forbidden):
        raise GravityItem9InteriorExteriorError("candidate manifest crossed frozen boundary")
    ids = [str(row["candidate_id"]) for row in manifest["cells"]]
    if len(ids) != len(set(ids)):
        raise GravityItem9InteriorExteriorError("candidate IDs are duplicated")
    if any(bool(row["historical_novelty_claimed"]) for row in manifest["cells"]):
        raise GravityItem9InteriorExteriorError("candidate manifest overclaims novelty")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem9InteriorExteriorError("candidate manifest contains an overclaim")


def write_candidate_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    manifest = build_candidate_manifest(root)
    validate_candidate_manifest(manifest, config)
    path = root / config["candidate_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def _profile_rows(payload: bytes, required: Sequence[str]) -> list[dict[str, float]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem9InteriorExteriorError("PROBES profile is not UTF-8") from exc
    if len(lines) < 3:
        raise GravityItem9InteriorExteriorError("PROBES profile is empty")
    reader = csv.DictReader(lines[1:])
    if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
        raise GravityItem9InteriorExteriorError("PROBES profile schema changed")
    result = []
    for raw in reader:
        try:
            row = {field: float(raw[field]) for field in required}
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in row.values()):
            result.append(row)
    return result


def parse_photometry_profile(payload: bytes) -> list[dict[str, float]]:
    return _profile_rows(payload, ("R", "SB", "SB_e", "totmag", "totmag_e", "ellip"))


def parse_rotation_profile(payload: bytes) -> list[dict[str, float]]:
    return _profile_rows(payload, ("R", "V", "V_e"))


def _archive_entries(archive: zipfile.ZipFile) -> dict[str, str]:
    result: dict[str, str] = {}
    for full_name in archive.namelist():
        basename = Path(full_name).name
        if not basename:
            continue
        if basename in result:
            raise GravityItem9InteriorExteriorError("duplicate profile basename in archive")
        result[basename] = full_name
    return result


def acquire_profile_archive(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem9InteriorExteriorError("Item 9 scientific freeze is not bound")
    payload = _fetch(str(config["sources"]["profile_archive_url"]))
    sources = config["sources"]
    if len(payload) != int(sources["profile_archive_bytes"]):
        raise GravityItem9InteriorExteriorError("PROBES profile archive size changed")
    if _md5_bytes(payload) != sources["profile_archive_md5"]:
        raise GravityItem9InteriorExteriorError("PROBES profile archive MD5 changed")
    path = root / config["profile_archive_work_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _weighted_quantile_radius(
    radius: np.ndarray, cumulative_mass: np.ndarray, quantile: float
) -> float:
    target = quantile * float(cumulative_mass[-1])
    return float(np.interp(target, cumulative_mass, radius))


def measure_photometric_predictors(
    *,
    photometry_rows: Sequence[Mapping[str, float]],
    distance_mpc: float,
    extinction_r_magnitude: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build target-blind stellar predictors without a rotation velocity input."""

    if not math.isfinite(distance_mpc) or distance_mpc <= 0:
        raise GravityItem9InteriorExteriorError("invalid galaxy distance")
    required = int(config["quality"]["minimum_photometry_points"])
    rows = sorted(photometry_rows, key=lambda row: float(row["R"]))
    rows = [row for row in rows if float(row["R"]) > 0]
    if len(rows) < required:
        raise GravityItem9InteriorExteriorError("insufficient finite photometry")
    radius_arcsec = np.asarray([row["R"] for row in rows], dtype=np.float64)
    if np.any(np.diff(radius_arcsec) <= 0):
        unique: dict[float, Mapping[str, float]] = {}
        for row in rows:
            unique[float(row["R"])] = row
        rows = [unique[key] for key in sorted(unique)]
        radius_arcsec = np.asarray([row["R"] for row in rows], dtype=np.float64)
    if len(rows) < required:
        raise GravityItem9InteriorExteriorError("insufficient unique photometry radii")
    constants = config["constants"]
    radius_kpc = (
        radius_arcsec
        * distance_mpc
        * 1000.0
        / float(constants["arcseconds_per_radian"])
    )
    surface_brightness = np.asarray([row["SB"] for row in rows], dtype=np.float64)
    corrected_sb = surface_brightness - extinction_r_magnitude
    surface_density = (
        10.0
        ** (
            -0.4
            * (
                corrected_sb
                - float(constants["solar_absolute_r_magnitude_ab"])
                - float(constants["surface_brightness_conversion"])
            )
        )
        * float(constants["fixed_r_band_mass_to_light_msun_per_lsun"])
    )
    distance_modulus = 5.0 * math.log10(distance_mpc) + 25.0
    total_magnitude = np.asarray([row["totmag"] for row in rows], dtype=np.float64)
    absolute_magnitude = total_magnitude - extinction_r_magnitude - distance_modulus
    cumulative_mass = (
        10.0
        ** (
            -0.4
            * (absolute_magnitude - float(constants["solar_absolute_r_magnitude_ab"]))
        )
        * float(constants["fixed_r_band_mass_to_light_msun_per_lsun"])
    )
    cumulative_mass = np.maximum.accumulate(cumulative_mass)
    if np.any(~np.isfinite(cumulative_mass)) or float(cumulative_mass[-1]) <= 0:
        raise GravityItem9InteriorExteriorError("invalid cumulative stellar-light mass")
    if np.any(~np.isfinite(surface_density)) or np.any(surface_density <= 0):
        raise GravityItem9InteriorExteriorError("invalid stellar surface density")
    ellipticity = np.asarray([row["ellip"] for row in rows], dtype=np.float64)
    outer = ellipticity[len(ellipticity) // 2 :]
    axis_ratio = float(np.median(1.0 - outer))
    q0 = float(constants["intrinsic_disk_axis_ratio"])
    cos_squared = float(np.clip((axis_ratio**2 - q0**2) / (1.0 - q0**2), 0.0, 1.0))
    inclination_sine = math.sqrt(max(0.0, 1.0 - cos_squared))
    inclination_degrees = math.degrees(math.asin(inclination_sine))
    r50_kpc = _weighted_quantile_radius(radius_kpc, cumulative_mass, 0.5)
    r90_kpc = _weighted_quantile_radius(radius_kpc, cumulative_mass, 0.9)
    shell_mass = np.diff(np.concatenate(([0.0], cumulative_mass)))
    shell_mass = np.maximum(shell_mass, 0.0)
    return {
        "radius_arcsec": radius_arcsec,
        "radius_kpc": radius_kpc,
        "surface_density": surface_density,
        "cumulative_mass": cumulative_mass,
        "shell_mass": shell_mass,
        "total_mass": float(cumulative_mass[-1]),
        "inclination_sine": inclination_sine,
        "inclination_degrees": inclination_degrees,
        "r50_kpc": r50_kpc,
        "r90_kpc": r90_kpc,
        "radial_concentration": r50_kpc / r90_kpc,
    }


def _kernel_matrix(log_radius: np.ndarray, mode: str, scale: float) -> np.ndarray:
    unique, inverse, multiplicity = np.unique(
        log_radius, return_inverse=True, return_counts=True
    )
    if len(unique) < 2:
        raise GravityItem9InteriorExteriorError("insufficient unique kernel radii")
    edges = np.empty(len(unique) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (unique[:-1] + unique[1:])
    edges[0] = unique[0] - 0.5 * (unique[1] - unique[0])
    edges[-1] = unique[-1] + 0.5 * (unique[-1] - unique[-2])
    unique_widths = np.diff(edges)
    if np.any(~np.isfinite(unique_widths)) or np.any(unique_widths <= 0):
        raise GravityItem9InteriorExteriorError("invalid log-radius quadrature")
    widths = unique_widths[inverse] / multiplicity[inverse]
    delta = log_radius[None, :] - log_radius[:, None]
    if mode == "interior":
        admitted = delta <= 1e-12
    elif mode == "exterior":
        admitted = delta >= -1e-12
    else:  # pragma: no cover - internal programming error
        raise GravityItem9InteriorExteriorError("unknown kernel mode")
    weights = np.exp(-np.abs(delta) / scale) * admitted * widths[None, :]
    denominator = np.sum(weights, axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise GravityItem9InteriorExteriorError("empty nonlocal kernel row")
    return weights / denominator


POINT_FEATURE_FIELDS = (
    "radius_arcsec",
    "radius_kpc",
    "log10_gbar",
    "log10_radius_over_r50",
    "log10_surface_density",
    "enclosed_mass_fraction",
    "log10_total_mass",
    "inclination_sine",
    "log10_distance",
    "newtonian_speed_km_s",
    "rar_speed_km_s",
    "surface_brightness_q10",
    "surface_brightness_q100",
    "surface_brightness_q1000",
    "acceleration_q0p01",
    "acceleration_q0p1",
    "acceleration_q1",
    "enclosed_mass_fraction_source",
    "outer_inverse_radius_potential_source",
    "outer_radius_lever_source",
    "condition_surface_density",
    "condition_compactness",
    "condition_vacuum_fraction",
    "condition_radial_span",
    "condition_concentration",
)


SOURCE_FIELD_BY_DEFINITION = {
    ("surface_brightness", 10.0): "surface_brightness_q10",
    ("surface_brightness", 100.0): "surface_brightness_q100",
    ("surface_brightness", 1000.0): "surface_brightness_q1000",
    ("acceleration", 0.01): "acceleration_q0p01",
    ("acceleration", 0.1): "acceleration_q0p1",
    ("acceleration", 1.0): "acceleration_q1",
    ("enclosed_mass_fraction", None): "enclosed_mass_fraction_source",
    ("outer_inverse_radius_potential", None): "outer_inverse_radius_potential_source",
    ("outer_radius_lever", None): "outer_radius_lever_source",
}


def measure_point_features(
    *,
    predictor: Mapping[str, Any],
    rotation_radius_arcsec: Sequence[float],
    distance_mpc: float,
    config: Mapping[str, Any],
) -> list[dict[str, float]]:
    """Evaluate target-blind features at response radii without response velocities."""

    response_radius_arcsec = np.abs(np.asarray(rotation_radius_arcsec, dtype=np.float64))
    valid = np.isfinite(response_radius_arcsec) & (response_radius_arcsec > 0)
    constants = config["constants"]
    floor = float(constants["dimensionless_floor"])
    radius_kpc = (
        response_radius_arcsec
        * distance_mpc
        * 1000.0
        / float(constants["arcseconds_per_radian"])
    )
    phot_radius_arcsec = np.asarray(predictor["radius_arcsec"])
    valid &= response_radius_arcsec >= float(phot_radius_arcsec[0])
    valid &= response_radius_arcsec <= float(phot_radius_arcsec[-1])
    # Some source curves include a central R=0 row. It is frozen-quality-ineligible,
    # but all target-blind arrays must remain finite until the quality mask discards it.
    safe_radius_kpc = np.where(
        radius_kpc > 0, radius_kpc, float(np.asarray(predictor["radius_kpc"])[0])
    )
    surface_density = np.interp(
        response_radius_arcsec, phot_radius_arcsec, predictor["surface_density"]
    )
    enclosed_mass = np.interp(
        response_radius_arcsec, phot_radius_arcsec, predictor["cumulative_mass"]
    )
    total_mass = float(predictor["total_mass"])
    enclosed_fraction = np.clip(enclosed_mass / total_mass, 0.0, 1.0)
    gravity = float(constants["gravity_kpc_km2_s2_msun"])
    g_dagger = float(constants["g_dagger_km2_s2_kpc"])
    gbar = gravity * enclosed_mass / np.maximum(safe_radius_kpc**2, floor)
    newtonian_v2 = np.maximum(gbar * safe_radius_kpc, floor)
    ratio = np.maximum(gbar / g_dagger, floor)
    rar_acceleration = gbar / np.maximum(1.0 - np.exp(-np.sqrt(ratio)), floor)
    rar_v2 = np.maximum(rar_acceleration * safe_radius_kpc, floor)
    phot_radius_kpc = np.asarray(predictor["radius_kpc"])
    shell_mass = np.asarray(predictor["shell_mass"])
    outer_inverse = np.zeros(len(radius_kpc), dtype=np.float64)
    outer_lever = np.zeros(len(radius_kpc), dtype=np.float64)
    for index, radius in enumerate(safe_radius_kpc):
        mask = phot_radius_kpc > radius
        if np.any(mask):
            outer_inverse[index] = (
                radius
                * float(np.sum(shell_mass[mask] / phot_radius_kpc[mask]))
                / total_mass
            )
            outer_lever[index] = (
                float(np.sum(shell_mass[mask] * phot_radius_kpc[mask] / radius))
                / total_mass
            )
    outer_inverse_source = outer_inverse / (1.0 + outer_inverse)
    outer_lever_source = outer_lever / (1.0 + outer_lever)
    source_values = {
        "surface_brightness_q10": surface_density / (surface_density + 10.0),
        "surface_brightness_q100": surface_density / (surface_density + 100.0),
        "surface_brightness_q1000": surface_density / (surface_density + 1000.0),
        "acceleration_q0p01": ratio / (ratio + 0.01),
        "acceleration_q0p1": ratio / (ratio + 0.1),
        "acceleration_q1": ratio / (ratio + 1.0),
        "enclosed_mass_fraction_source": enclosed_fraction,
        "outer_inverse_radius_potential_source": outer_inverse_source,
        "outer_radius_lever_source": outer_lever_source,
    }
    surface_condition = math.tanh((float(np.median(np.log1p(surface_density))) - 4.0) / 4.0)
    compactness_condition = math.tanh(float(np.median(np.log(ratio))) / 4.0)
    vacuum_condition = 2.0 * float(np.mean(100.0 / (100.0 + surface_density))) - 1.0
    valid_radii = radius_kpc[valid]
    radial_span = (
        float(np.max(valid_radii) / np.min(valid_radii)) if len(valid_radii) else 1.0
    )
    radial_span_condition = math.tanh((math.log(max(radial_span, 1.0)) - 3.0) / 2.0)
    concentration_condition = 2.0 * float(predictor["radial_concentration"]) - 1.0
    conditions = {
        "condition_surface_density": surface_condition,
        "condition_compactness": compactness_condition,
        "condition_vacuum_fraction": vacuum_condition,
        "condition_radial_span": radial_span_condition,
        "condition_concentration": concentration_condition,
    }
    result = []
    for index in range(len(radius_kpc)):
        row = {
            "radius_arcsec": float(response_radius_arcsec[index]),
            "radius_kpc": float(radius_kpc[index]),
            "log10_gbar": math.log10(float(gbar[index])),
            "log10_radius_over_r50": math.log10(
                float(safe_radius_kpc[index]) / float(predictor["r50_kpc"])
            ),
            "log10_surface_density": math.log10(float(surface_density[index])),
            "enclosed_mass_fraction": float(enclosed_fraction[index]),
            "log10_total_mass": math.log10(total_mass),
            "inclination_sine": float(predictor["inclination_sine"]),
            "log10_distance": math.log10(distance_mpc),
            "newtonian_speed_km_s": math.sqrt(float(newtonian_v2[index])),
            "rar_speed_km_s": math.sqrt(float(rar_v2[index])),
            **{name: float(values[index]) for name, values in source_values.items()},
            **conditions,
            "within_photometry": bool(valid[index]),
        }
        if any(not math.isfinite(float(value)) for value in row.values()):
            raise GravityItem9InteriorExteriorError("non-finite point feature")
        result.append(row)
    return result


def _write_tsv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _find_entry(entries: Mapping[str, str], basename: str) -> str:
    try:
        return entries[basename]
    except KeyError as exc:
        raise GravityItem9InteriorExteriorError(f"missing archive entry {basename}") from exc


def _quality_points(
    *,
    response_rows: Sequence[Mapping[str, float]],
    point_features: Sequence[Mapping[str, Any]],
    inclination_sine: float,
    inclination_degrees: float,
    config: Mapping[str, Any],
) -> tuple[list[int], list[str]]:
    quality = config["quality"]
    failures: list[str] = []
    if not (
        float(quality["minimum_inclination_degrees"])
        <= inclination_degrees
        <= float(quality["maximum_inclination_degrees"])
    ):
        failures.append("inclination_outside_frozen_range")
    finite_rotation = sum(
        math.isfinite(float(row["R"])) and abs(float(row["R"])) > 0 for row in response_rows
    )
    within = sum(
        bool(feature["within_photometry"])
        for row, feature in zip(response_rows, point_features, strict=True)
        if math.isfinite(float(row["R"])) and abs(float(row["R"])) > 0
    )
    overlap_fraction = within / finite_rotation if finite_rotation else 0.0
    if overlap_fraction < float(quality["minimum_fraction_rotation_points_within_photometry"]):
        failures.append("insufficient_photometry_rotation_overlap")
    valid_indices = []
    for index, (response, feature) in enumerate(
        zip(response_rows, point_features, strict=True)
    ):
        observed = abs(float(response["V"])) / max(inclination_sine, 1e-12)
        error = float(response["V_e"]) / max(inclination_sine, 1e-12)
        valid = (
            bool(feature["within_photometry"])
            and math.isfinite(observed)
            and observed >= float(quality["minimum_corrected_speed_km_s"])
            and math.isfinite(error)
            and error > 0
            and error / observed <= float(quality["maximum_fractional_speed_error"])
        )
        if valid:
            valid_indices.append(index)
    if len(valid_indices) < int(quality["minimum_rotation_points"]):
        failures.append("insufficient_quality_rotation_points")
    if valid_indices:
        radii = [float(point_features[index]["radius_kpc"]) for index in valid_indices]
        span = max(radii) / min(radii)
        if span < float(quality["minimum_rotation_radial_span"]):
            failures.append("insufficient_rotation_radial_span")
    else:
        failures.append("no_quality_rotation_radius")
    return valid_indices, sorted(set(failures))


FEATURE_TABLE_FIELDS = (
    "galaxy",
    "point_index",
    "outer_fold",
    "primary_rc_survey",
    "distance_bin",
    "mass_stratum",
    *POINT_FEATURE_FIELDS,
)

RESPONSE_TABLE_FIELDS = (
    "galaxy",
    "point_index",
    "radius_arcsec",
    "line_of_sight_velocity_km_s",
    "line_of_sight_error_km_s",
    "observed_speed_km_s",
    "observed_speed_error_km_s",
)


def extract_profiles(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem9InteriorExteriorError("Item 9 scientific freeze is not bound")
    sample_path = root / config["sample_manifest_output"]
    candidate_path = root / config["candidate_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    validate_candidate_manifest(candidates, config)
    archive_path = root / config["profile_archive_work_path"]
    if not archive_path.exists():
        raise GravityItem9InteriorExteriorError("profile archive has not been acquired")
    archive_payload = archive_path.read_bytes()
    sources = config["sources"]
    if len(archive_payload) != int(sources["profile_archive_bytes"]):
        raise GravityItem9InteriorExteriorError("stored profile archive size changed")
    if _md5_bytes(archive_payload) != sources["profile_archive_md5"]:
        raise GravityItem9InteriorExteriorError("stored profile archive MD5 changed")

    feature_rows: list[dict[str, Any]] = []
    response_output_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    galaxy_summaries: list[dict[str, Any]] = []
    predictor_profiles_opened = 0
    exploration_profiles_opened = 0
    rotation_rows_read = 0
    confirmation_response_entries_present = 0
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        entries = _archive_entries(archive)
        for obj in sample["objects"]:
            name = str(obj["name"])
            role = str(obj["role"])
            predictor_basename = f"{name}{sources['predictor_entry_suffix']}"
            response_basename = f"{name}{sources['response_entry_suffix']}"
            if role == "reserved_confirmation" and response_basename in entries:
                confirmation_response_entries_present += 1
            try:
                predictor_entry = _find_entry(entries, predictor_basename)
                predictor_payload = archive.read(predictor_entry)
                predictor_profiles_opened += 1
                photometry = parse_photometry_profile(predictor_payload)
                predictor = measure_photometric_predictors(
                    photometry_rows=photometry,
                    distance_mpc=float(obj["distance_mpc"]),
                    extinction_r_magnitude=float(obj["extinction_r_magnitude"]),
                    config=config,
                )
            except (GravityItem9InteriorExteriorError, KeyError, ValueError) as exc:
                failures.append(
                    {"galaxy": name, "role": role, "stage": "predictor", "reason": str(exc)}
                )
                continue
            record: dict[str, Any] = {
                "galaxy": name,
                "role": role,
                "predictor_entry": predictor_entry,
                "predictor_bytes": len(predictor_payload),
                "predictor_sha256": _sha256_bytes(predictor_payload),
                "response_entry_opened": False,
            }
            if role == "reserved_confirmation":
                source_records.append(record)
                continue
            try:
                response_entry = _find_entry(entries, response_basename)
                response_payload = archive.read(response_entry)
                exploration_profiles_opened += 1
                responses = parse_rotation_profile(response_payload)
                rotation_rows_read += len(responses)
                point_features = measure_point_features(
                    predictor=predictor,
                    rotation_radius_arcsec=[row["R"] for row in responses],
                    distance_mpc=float(obj["distance_mpc"]),
                    config=config,
                )
                valid_indices, reasons = _quality_points(
                    response_rows=responses,
                    point_features=point_features,
                    inclination_sine=float(predictor["inclination_sine"]),
                    inclination_degrees=float(predictor["inclination_degrees"]),
                    config=config,
                )
                record.update(
                    {
                        "response_entry": response_entry,
                        "response_entry_opened": True,
                        "response_bytes": len(response_payload),
                        "response_sha256": _sha256_bytes(response_payload),
                        "response_rows": len(responses),
                    }
                )
            except (GravityItem9InteriorExteriorError, KeyError, ValueError) as exc:
                failures.append(
                    {"galaxy": name, "role": role, "stage": "response", "reason": str(exc)}
                )
                source_records.append(record)
                continue
            source_records.append(record)
            if reasons:
                failures.append(
                    {"galaxy": name, "role": role, "stage": "quality", "reason": reasons}
                )
                continue
            galaxy_summaries.append(
                {
                    "galaxy": name,
                    "outer_fold": int(obj["outer_fold"]),
                    "primary_rc_survey": obj["primary_rc_survey"],
                    "distance_bin": int(obj["distance_bin"]),
                    "distance_mpc": obj["distance_mpc"],
                    "total_mass": predictor["total_mass"],
                    "inclination_degrees": predictor["inclination_degrees"],
                    "quality_points": len(valid_indices),
                }
            )
            for index in valid_indices:
                response = responses[index]
                feature = point_features[index]
                feature_rows.append(
                    {
                        "galaxy": name,
                        "point_index": index,
                        "outer_fold": int(obj["outer_fold"]),
                        "primary_rc_survey": obj["primary_rc_survey"],
                        "distance_bin": int(obj["distance_bin"]),
                        "mass_stratum": "PENDING",
                        **{field: _metric(float(feature[field])) for field in POINT_FEATURE_FIELDS},
                    }
                )
                sine = float(predictor["inclination_sine"])
                response_output_rows.append(
                    {
                        "galaxy": name,
                        "point_index": index,
                        "radius_arcsec": _metric(float(feature["radius_arcsec"])),
                        "line_of_sight_velocity_km_s": _metric(float(response["V"])),
                        "line_of_sight_error_km_s": _metric(float(response["V_e"])),
                        "observed_speed_km_s": _metric(abs(float(response["V"])) / sine),
                        "observed_speed_error_km_s": _metric(float(response["V_e"]) / sine),
                    }
                )

    passing_names = {str(row["galaxy"]) for row in galaxy_summaries}
    mass_median = float(np.median([float(row["total_mass"]) for row in galaxy_summaries]))
    mass_by_name = {
        str(row["galaxy"]): (
            "low_mass" if float(row["total_mass"]) <= mass_median else "high_mass"
        )
        for row in galaxy_summaries
    }
    for row in feature_rows:
        row["mass_stratum"] = mass_by_name[str(row["galaxy"])]
    feature_rows.sort(key=lambda row: (str(row["galaxy"]), int(row["point_index"])))
    response_output_rows.sort(
        key=lambda row: (str(row["galaxy"]), int(row["point_index"]))
    )
    source_records.sort(key=lambda row: (str(row["role"]), str(row["galaxy"])))
    failures.sort(key=lambda row: (str(row["role"]), str(row["galaxy"]), str(row["stage"])))
    galaxy_summaries.sort(key=lambda row: str(row["galaxy"]))
    if len({(row["galaxy"], row["point_index"]) for row in feature_rows}) != len(feature_rows):
        raise GravityItem9InteriorExteriorError("duplicate extracted feature point")
    if {(row["galaxy"], row["point_index"]) for row in feature_rows} != {
        (row["galaxy"], row["point_index"]) for row in response_output_rows
    }:
        raise GravityItem9InteriorExteriorError("feature and response point scopes differ")

    feature_path = root / config["feature_output"]
    response_path = root / config["response_output"]
    _write_tsv(feature_path, FEATURE_TABLE_FIELDS, feature_rows)
    _write_tsv(response_path, RESPONSE_TABLE_FIELDS, response_output_rows)
    profile_source = _seal(
        {
            "schema_version": "invariant-gravity-item9-probes-profile-source-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM9_FROZEN_PROFILE_ENTRY_ACQUISITION",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "archive": {
                "url": sources["profile_archive_url"],
                "bytes": len(archive_payload),
                "md5": _md5_bytes(archive_payload),
                "sha256": _sha256_bytes(archive_payload),
                "work_path": config["profile_archive_work_path"],
                "container_includes_sealed_confirmation_entries": True,
            },
            "bindings": {
                "sample_file_sha256": _sha256_file(sample_path),
                "sample_content_sha256": sample["content_sha256"],
                "candidate_file_sha256": _sha256_file(candidate_path),
                "candidate_content_sha256": candidates["content_sha256"],
            },
            "counts": {
                "predictor_profiles_opened": predictor_profiles_opened,
                "exploration_rotation_profiles_opened": exploration_profiles_opened,
                "exploration_rotation_rows_read": rotation_rows_read,
                "reserved_confirmation_rotation_entries_opened": 0,
                "reserved_confirmation_rotation_entries_present_in_sealed_archive": (
                    confirmation_response_entries_present
                ),
                "model_fit_rows_read": 0,
                "structural_parameter_rows_read": 0,
                "dynamical_or_dark_mass_values_read": 0,
                "lensing_mass_values_read": 0,
                "paid_model_calls": 0,
            },
            "records": source_records,
            "claims": dict(config["claim_boundaries"]),
        }
    )
    profile_source_path = root / config["profile_source_output"]
    profile_source_path.parent.mkdir(parents=True, exist_ok=True)
    profile_source_path.write_bytes(canonical_json_bytes(profile_source) + b"\n")
    summary = _seal(
        {
            "schema_version": "invariant-gravity-item9-probes-profile-extraction-1.0",
            "goal": config["goal"],
            "decision": (
                "PASS_ITEM9_PROFILE_QUALITY_FLOOR"
                if len(passing_names)
                >= int(config["quality"]["minimum_quality_passing_exploration_galaxies"])
                and len(passing_names) / int(config["sample"]["exploration_count"])
                >= float(config["quality"]["minimum_quality_retention_fraction"])
                else "INCONCLUSIVE_ITEM9_PROFILE_QUALITY_FLOOR"
            ),
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "counts": {
                "selected_exploration_galaxies": config["sample"]["exploration_count"],
                "quality_passing_exploration_galaxies": len(passing_names),
                "quality_failed_exploration_galaxies": (
                    int(config["sample"]["exploration_count"]) - len(passing_names)
                ),
                "quality_points": len(feature_rows),
                "reserved_confirmation_galaxies": config["sample"][
                    "reserved_confirmation_count"
                ],
                "reserved_confirmation_rotation_entries_opened": 0,
            },
            "mass_median_msun": _metric(mass_median),
            "failures": failures,
            "galaxies": galaxy_summaries,
            "outputs": {
                "feature_sha256": _sha256_file(feature_path),
                "response_sha256": _sha256_file(response_path),
                "profile_source_sha256": _sha256_file(profile_source_path),
                "profile_source_content_sha256": profile_source["content_sha256"],
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )
    summary_path = root / config["extraction_output"]
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return {
        "profile_source": profile_source_path,
        "features": feature_path,
        "responses": response_path,
        "summary": summary_path,
    }


def validate_profile_source(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem9InteriorExteriorError("profile source content hash changed")
    if manifest["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem9InteriorExteriorError("profile source freeze binding changed")
    counts = manifest["counts"]
    if int(counts["reserved_confirmation_rotation_entries_opened"]) != 0:
        raise GravityItem9InteriorExteriorError("confirmation rotation entry was opened")
    forbidden = (
        "model_fit_rows_read",
        "structural_parameter_rows_read",
        "dynamical_or_dark_mass_values_read",
        "lensing_mass_values_read",
        "paid_model_calls",
    )
    if any(int(counts[name]) != 0 for name in forbidden):
        raise GravityItem9InteriorExteriorError("profile source crossed forbidden boundary")
    for row in manifest["records"]:
        if row["role"] == "reserved_confirmation" and row["response_entry_opened"]:
            raise GravityItem9InteriorExteriorError("confirmation response receipt is open")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem9InteriorExteriorError("profile source contains an overclaim")


def _load_experiment_data(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    feature_path = root / config["feature_output"]
    response_path = root / config["response_output"]
    summary_path = root / config["extraction_output"]
    profile_path = root / config["profile_source_output"]
    candidate_path = root / config["candidate_manifest_output"]
    source = json.loads(profile_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_profile_source(source, config)
    validate_candidate_manifest(candidates, config)
    for manifest, label in ((summary, "extraction"),):
        copy = dict(manifest)
        digest = copy.pop("content_sha256", None)
        if digest != canonical_sha256(copy):
            raise GravityItem9InteriorExteriorError(f"{label} content hash changed")
    if summary["outputs"]["feature_sha256"] != _sha256_file(feature_path):
        raise GravityItem9InteriorExteriorError("feature table changed")
    if summary["outputs"]["response_sha256"] != _sha256_file(response_path):
        raise GravityItem9InteriorExteriorError("response table changed")
    features = _read_tsv(feature_path)
    responses = _read_tsv(response_path)
    if len(features) != len(responses) or not features:
        raise GravityItem9InteriorExteriorError("point table scope changed")
    keys = [(row["galaxy"], row["point_index"]) for row in features]
    if keys != [(row["galaxy"], row["point_index"]) for row in responses]:
        raise GravityItem9InteriorExteriorError("feature-response alignment changed")
    names = sorted({str(row["galaxy"]) for row in features})
    galaxy_index_by_name = {name: index for index, name in enumerate(names)}
    galaxy_index = np.asarray(
        [galaxy_index_by_name[str(row["galaxy"])] for row in features], dtype=np.int64
    )
    if np.any(np.diff(galaxy_index) < 0):
        raise GravityItem9InteriorExteriorError("points are not grouped by galaxy")
    point_counts = np.bincount(galaxy_index, minlength=len(names))
    starts = np.concatenate(([0], np.cumsum(point_counts)[:-1])).astype(np.int64)
    folds_by_name = {
        str(row["galaxy"]): int(row["outer_fold"]) for row in summary["galaxies"]
    }
    folds = np.asarray([folds_by_name[name] for name in names], dtype=np.int64)
    y = np.log10(
        np.asarray([float(row["observed_speed_km_s"]) for row in responses], dtype=np.float64)
    )
    y_error = np.asarray(
        [float(row["observed_speed_error_km_s"]) for row in responses], dtype=np.float64
    )
    feature_arrays = {
        field: np.asarray([float(row[field]) for row in features], dtype=np.float64)
        for field in POINT_FEATURE_FIELDS
    }
    metadata = {
        "primary_rc_survey": [str(row["primary_rc_survey"]) for row in features],
        "distance_bin": [int(row["distance_bin"]) for row in features],
        "mass_stratum": [str(row["mass_stratum"]) for row in features],
    }
    return {
        "features": features,
        "responses": responses,
        "candidate_manifest": candidates,
        "summary": summary,
        "profile_source": source,
        "names": names,
        "galaxy_index": galaxy_index,
        "point_counts": point_counts,
        "starts": starts,
        "folds": folds,
        "y": y,
        "y_error": y_error,
        "feature_arrays": feature_arrays,
        "metadata": metadata,
    }


def build_operator_components(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> np.ndarray:
    operators = data["candidate_manifest"]["operators"]
    arrays = data["feature_arrays"]
    result = np.empty((len(operators), len(data["features"])), dtype=np.float64)
    for galaxy_index, _name in enumerate(data["names"]):
        start = int(data["starts"][galaxy_index])
        stop = start + int(data["point_counts"][galaxy_index])
        log_radius = np.log(np.maximum(arrays["radius_kpc"][start:stop], 1e-12))
        matrices: dict[tuple[str, float], np.ndarray] = {}
        for scale in config["operator_grammar"]["log_radius_scales"]:
            matrices[("interior", float(scale))] = _kernel_matrix(
                log_radius, "interior", float(scale)
            )
            matrices[("exterior", float(scale))] = _kernel_matrix(
                log_radius, "exterior", float(scale)
            )
        for operator_index, operator in enumerate(operators):
            threshold = (
                None if operator["threshold"] is None else float(operator["threshold"])
            )
            key = (str(operator["source"]), threshold)
            try:
                source_field = SOURCE_FIELD_BY_DEFINITION[key]
            except KeyError as exc:
                raise GravityItem9InteriorExteriorError(
                    f"unknown operator source {key}"
                ) from exc
            source = arrays[source_field][start:stop]
            scale = float(operator["log_radius_scale"])
            interior = matrices[("interior", scale)] @ source
            exterior = matrices[("exterior", scale)] @ source
            if operator["mode"] == "interior_minus_exterior_occupancy":
                component = interior - exterior
            elif operator["mode"] == "interior_occupancy_times_exterior_vacuum":
                component = interior * (1.0 - exterior)
            else:
                raise GravityItem9InteriorExteriorError("unknown operator mode")
            if np.any(~np.isfinite(component)):
                raise GravityItem9InteriorExteriorError("non-finite operator component")
            result[operator_index, start:stop] = component
    return result


def _point_weights(data: Mapping[str, Any]) -> np.ndarray:
    counts = np.asarray(data["point_counts"], dtype=np.float64)
    return 1.0 / counts[np.asarray(data["galaxy_index"], dtype=np.int64)]


def _metrics(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict[str, str]:
    normalized = weights / np.sum(weights)
    residual = y - prediction
    mse = float(np.sum(normalized * residual**2))
    mean = float(np.sum(normalized * y))
    variance = float(np.sum(normalized * (y - mean) ** 2))
    r2 = 1.0 - mse / variance if variance > 0 else 0.0
    return {"mse": _metric(mse), "r2": _metric(r2)}


def _ridge_fit(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    normalized = weights / np.sum(weights)
    mean = np.sum(design * normalized[:, None], axis=0)
    centered = design - mean
    scale = np.sqrt(np.sum(centered**2 * normalized[:, None], axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    transformed = centered / scale
    y_mean = float(np.sum(target * normalized))
    y_centered = target - y_mean
    sqrt_weight = np.sqrt(normalized)
    matrix = transformed * sqrt_weight[:, None]
    rhs = y_centered * sqrt_weight
    penalty = alpha * np.eye(transformed.shape[1])
    coefficients = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ rhs)
    return {
        "mean": mean,
        "scale": scale,
        "y_mean": y_mean,
        "coefficients": coefficients,
        "alpha": alpha,
    }


def _ridge_predict(model: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    return float(model["y_mean"]) + (
        (design - np.asarray(model["mean"])) / np.asarray(model["scale"])
    ) @ np.asarray(model["coefficients"])


def _local_control_oof(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays = data["feature_arrays"]
    design = np.column_stack(
        [arrays[field] for field in config["evaluation"]["local_control_features"]]
    )
    y = np.asarray(data["y"])
    galaxy_index = np.asarray(data["galaxy_index"])
    folds = np.asarray(data["folds"])
    point_folds = folds[galaxy_index]
    weights = _point_weights(data)
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    records = []
    for outer in range(int(config["evaluation"]["outer_folds"])):
        train = point_folds != outer
        test = point_folds == outer
        inner_scores = []
        for alpha in config["evaluation"]["ridge_penalties"]:
            fold_scores = []
            for inner in sorted(set(folds) - {outer}):
                inner_train = train & (point_folds != inner)
                inner_test = point_folds == inner
                model = _ridge_fit(
                    design[inner_train], y[inner_train], weights[inner_train], float(alpha)
                )
                inner_prediction = _ridge_predict(model, design[inner_test])
                fold_scores.append(
                    float(_metrics(y[inner_test], inner_prediction, weights[inner_test])["mse"])
                )
            inner_scores.append((float(np.mean(fold_scores)), float(alpha)))
        selected_score, selected_alpha = min(inner_scores, key=lambda item: (item[0], item[1]))
        model = _ridge_fit(design[train], y[train], weights[train], selected_alpha)
        prediction[test] = _ridge_predict(model, design[test])
        rar = np.log10(arrays["rar_speed_km_s"])
        local_train_prediction = np.full(np.sum(train), np.nan, dtype=np.float64)
        train_indices = np.flatnonzero(train)
        for inner in sorted(set(folds) - {outer}):
            inner_train = train & (point_folds != inner)
            inner_test = point_folds == inner
            inner_model = _ridge_fit(
                design[inner_train], y[inner_train], weights[inner_train], selected_alpha
            )
            positions = np.searchsorted(train_indices, np.flatnonzero(inner_test))
            local_train_prediction[positions] = _ridge_predict(inner_model, design[inner_test])
        local_inner_mse = float(
            _metrics(y[train], local_train_prediction, weights[train])["mse"]
        )
        rar_train_mse = float(_metrics(y[train], rar[train], weights[train])["mse"])
        records.append(
            {
                "fold": outer,
                "selected_alpha": _metric(selected_alpha),
                "selected_inner_mse": _metric(selected_score),
                "local_inner_mse": _metric(local_inner_mse),
                "rar_train_mse": _metric(rar_train_mse),
                "test_galaxies": int(np.sum(folds == outer)),
            }
        )
    if np.any(~np.isfinite(prediction)):
        raise GravityItem9InteriorExteriorError("local control OOF prediction is incomplete")
    return prediction, records


def _logistic_xp(value: Any, xp: Any) -> Any:
    return xp.where(
        value >= 0,
        1.0 / (1.0 + xp.exp(-value)),
        xp.exp(value) / (1.0 + xp.exp(value)),
    )


CONDITION_FIELD = {
    "surface_density": "condition_surface_density",
    "compactness": "condition_compactness",
    "vacuum_fraction": "condition_vacuum_fraction",
    "radial_span": "condition_radial_span",
    "concentration": "condition_concentration",
}


def _candidate_galaxy_losses(
    data: Mapping[str, Any],
    components: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    backend = "cpu_numpy"
    gpu_name = None
    try:
        import cupy as xp

        if int(xp.cuda.runtime.getDeviceCount()) < 1:
            raise RuntimeError("no CUDA device")
        backend = "gpu_cupy"
        gpu_name = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except (ImportError, RuntimeError):  # pragma: no cover - CPU fallback
        xp = np
    cells = data["candidate_manifest"]["cells"]
    operators = data["candidate_manifest"]["operators"]
    operator_index = {row["operator_id"]: index for index, row in enumerate(operators)}
    condition_names = list(config["operator_grammar"]["conditions"])
    condition_index = {name: index for index, name in enumerate(condition_names)}
    conditions_by_galaxy = np.empty((len(condition_names), len(data["names"])), dtype=np.float64)
    arrays = data["feature_arrays"]
    for cond_index, name in enumerate(condition_names):
        values = arrays[CONDITION_FIELD[name]]
        for galaxy in range(len(data["names"])):
            start = int(data["starts"][galaxy])
            conditions_by_galaxy[cond_index, galaxy] = float(values[start])
    galaxy_index = np.asarray(data["galaxy_index"], dtype=np.int64)
    starts = np.asarray(data["starts"], dtype=np.int64)
    point_counts = np.asarray(data["point_counts"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    rar_v2 = np.asarray(arrays["rar_speed_km_s"], dtype=np.float64) ** 2
    correction_scale = (
        np.asarray(arrays["radius_kpc"], dtype=np.float64)
        * float(config["constants"]["g_dagger_km2_s2_kpc"])
    )
    x_components = xp.asarray(components)
    x_conditions = xp.asarray(conditions_by_galaxy)
    x_galaxy_index = xp.asarray(galaxy_index)
    x_starts = xp.asarray(starts)
    x_counts = xp.asarray(point_counts)
    x_y = xp.asarray(y)
    x_rar_v2 = xp.asarray(rar_v2)
    x_scale = xp.asarray(correction_scale)
    losses = np.empty((len(cells), len(data["names"])), dtype=np.float64)
    invalid_counts = np.empty(len(cells), dtype=np.int64)
    batch_size = 256
    for begin in range(0, len(cells), batch_size):
        batch = cells[begin : begin + batch_size]
        op_indices = xp.asarray(
            [operator_index[str(row["operator_id"])] for row in batch], dtype=xp.int64
        )
        cond_indices = xp.asarray(
            [condition_index[str(row["condition"])] for row in batch], dtype=xp.int64
        )
        intercept = xp.asarray([float(row["intercept"]) for row in batch])[:, None]
        slope = xp.asarray([float(row["slope"]) for row in batch])[:, None]
        alpha = float(config["operator_grammar"]["alpha_max"]) * _logistic_xp(
            intercept + slope * x_conditions[cond_indices][:, x_galaxy_index], xp
        )
        prediction2 = x_rar_v2[None, :] + alpha * x_scale[None, :] * x_components[op_indices]
        invalid = (~xp.isfinite(prediction2)) | (prediction2 <= 0)
        prediction = 0.5 * xp.log10(xp.maximum(prediction2, 1e-300))
        point_loss = (x_y[None, :] - prediction) ** 2
        point_loss = xp.where(invalid, xp.inf, point_loss)
        galaxy_loss = xp.add.reduceat(point_loss, x_starts, axis=1) / x_counts[None, :]
        if backend == "gpu_cupy":
            losses[begin : begin + len(batch)] = xp.asnumpy(galaxy_loss)
            invalid_counts[begin : begin + len(batch)] = xp.asnumpy(
                xp.sum(invalid, axis=1)
            )
        else:
            losses[begin : begin + len(batch)] = galaxy_loss
            invalid_counts[begin : begin + len(batch)] = np.sum(invalid, axis=1)
    if backend == "gpu_cupy":
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started

    crosscheck_count = min(int(config["evaluation"]["gpu_crosscheck_candidates"]), len(cells))
    maximum_difference = 0.0
    for ordinal in range(crosscheck_count):
        row = cells[ordinal]
        op = operator_index[str(row["operator_id"])]
        cond = condition_index[str(row["condition"])]
        alpha = float(config["operator_grammar"]["alpha_max"]) * (
            1.0
            / (
                1.0
                + np.exp(
                    -(
                        float(row["intercept"])
                        + float(row["slope"])
                        * conditions_by_galaxy[cond, galaxy_index]
                    )
                )
            )
        )
        prediction2 = rar_v2 + alpha * correction_scale * components[op]
        prediction = 0.5 * np.log10(np.maximum(prediction2, 1e-300))
        point_loss = np.where(prediction2 > 0, (y - prediction) ** 2, np.inf)
        expected = np.add.reduceat(point_loss, starts) / point_counts
        finite = np.isfinite(expected) & np.isfinite(losses[ordinal])
        if np.any(finite):
            maximum_difference = max(
                maximum_difference,
                float(np.max(np.abs(expected[finite] - losses[ordinal, finite]))),
            )
        if not np.array_equal(np.isinf(expected), np.isinf(losses[ordinal])):
            raise GravityItem9InteriorExteriorError("GPU/CPU invalid-mask disagreement")
    if maximum_difference > 1e-10:
        raise GravityItem9InteriorExteriorError("GPU/CPU candidate score drift")
    return (
        losses,
        invalid_counts,
        {
            "backend": backend,
            "device": gpu_name,
            "cupy_version": getattr(xp, "__version__", None) if backend == "gpu_cupy" else None,
            "elapsed_seconds": _metric(elapsed),
            "candidate_formula_cells": len(cells),
            "operators": len(operators),
            "galaxies": len(data["names"]),
            "points": len(y),
            "candidate_galaxy_evaluations": len(cells) * len(data["names"]),
            "candidate_point_evaluations": len(cells) * len(y),
            "cpu_crosscheck_candidates": crosscheck_count,
            "cpu_crosscheck_maximum_galaxy_mse_difference": _metric(maximum_difference),
        },
    )


def _candidate_prediction(
    cell: Mapping[str, Any],
    data: Mapping[str, Any],
    components: np.ndarray,
    config: Mapping[str, Any],
    point_mask: np.ndarray | None = None,
) -> np.ndarray:
    cells = data["candidate_manifest"]["cells"]
    del cells
    operators = data["candidate_manifest"]["operators"]
    operator_index = {row["operator_id"]: index for index, row in enumerate(operators)}
    arrays = data["feature_arrays"]
    galaxy_index = np.asarray(data["galaxy_index"])
    condition = arrays[CONDITION_FIELD[str(cell["condition"])]]
    alpha = float(cell["alpha_max"]) / (
        1.0
        + np.exp(
            -(
                float(cell["intercept"])
                + float(cell["slope"]) * condition
            )
        )
    )
    del galaxy_index
    prediction2 = (
        arrays["rar_speed_km_s"] ** 2
        + alpha
        * arrays["radius_kpc"]
        * float(config["constants"]["g_dagger_km2_s2_kpc"])
        * components[operator_index[str(cell["operator_id"])]]
    )
    prediction = 0.5 * np.log10(np.maximum(prediction2, 1e-300))
    if point_mask is not None:
        return prediction[point_mask]
    return prediction


def _select_candidates(
    losses: np.ndarray,
    invalid_counts: np.ndarray,
    data: Mapping[str, Any],
    components: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cells = data["candidate_manifest"]["cells"]
    folds = np.asarray(data["folds"])
    point_folds = folds[np.asarray(data["galaxy_index"])]
    prediction = np.full(len(data["y"]), np.nan, dtype=np.float64)
    selections = []
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train_galaxies = folds != fold
        eligible_loss = np.mean(losses[:, train_galaxies], axis=1)
        eligible_loss = np.where(invalid_counts == 0, eligible_loss, np.inf)
        ordinal = int(np.argmin(eligible_loss))
        if not math.isfinite(float(eligible_loss[ordinal])):
            raise GravityItem9InteriorExteriorError("no finite Item 9 candidate")
        test_points = point_folds == fold
        prediction[test_points] = _candidate_prediction(
            cells[ordinal], data, components, config, test_points
        )
        selections.append(
            {
                "fold": fold,
                "selected_ordinal": ordinal,
                "selected_candidate_id": cells[ordinal]["candidate_id"],
                "selected_equivalence_class": cells[ordinal]["equivalence_class"],
                "authoritative_origin_status": cells[ordinal][
                    "authoritative_origin_status"
                ],
                "exact_prior_focusing_cell": cells[ordinal][
                    "exact_prior_focusing_cell"
                ],
                "training_equal_galaxy_mse": _metric(float(eligible_loss[ordinal])),
                "test_galaxies": int(np.sum(folds == fold)),
            }
        )
    if np.any(~np.isfinite(prediction)):
        raise GravityItem9InteriorExteriorError("candidate OOF prediction is incomplete")
    return prediction, selections


def _galaxy_losses(
    y: np.ndarray, prediction: np.ndarray, starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    return np.add.reduceat((y - prediction) ** 2, starts) / counts


def _stratum_results(
    data: Mapping[str, Any], baseline: np.ndarray, qualifying: np.ndarray
) -> list[dict[str, Any]]:
    y = np.asarray(data["y"])
    starts = np.asarray(data["starts"])
    counts = np.asarray(data["point_counts"], dtype=np.float64)
    baseline_loss = _galaxy_losses(y, baseline, starts, counts)
    qualifying_loss = _galaxy_losses(y, qualifying, starts, counts)
    first_rows = [data["features"][int(start)] for start in starts]
    dimensions = {
        "primary_rc_survey": [str(row["primary_rc_survey"]) for row in first_rows],
        "distance_bin": [str(row["distance_bin"]) for row in first_rows],
        "mass_stratum": [str(row["mass_stratum"]) for row in first_rows],
    }
    result = []
    for dimension, values in dimensions.items():
        for value in sorted(set(values)):
            mask = np.asarray([item == value for item in values], dtype=bool)
            baseline_mse = float(np.mean(baseline_loss[mask]))
            qualifying_mse = float(np.mean(qualifying_loss[mask]))
            result.append(
                {
                    "dimension": dimension,
                    "stratum": value,
                    "galaxies": int(np.sum(mask)),
                    "baseline_mse": _metric(baseline_mse),
                    "qualifying_mse": _metric(qualifying_mse),
                    "qualifying_mse_gain": _metric(baseline_mse - qualifying_mse),
                }
            )
    return result


def _paired_sign_flip(
    differences: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    count = int(config["evaluation"]["paired_sign_flip_permutations"])
    salt = str(config["evaluation"]["permutation_salt"])
    seed = int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
    random = np.random.default_rng(seed)
    observed = float(np.mean(differences))
    null = np.empty(count, dtype=np.float64)
    for index in range(count):
        signs = random.choice(np.asarray([-1.0, 1.0]), size=len(differences))
        null[index] = float(np.mean(differences * signs))
    p_value = (1 + int(np.sum(null >= observed))) / (count + 1)
    return {
        "scheme": "paired whole-galaxy sign flip of held-out MSE gains",
        "permutations": count,
        "observed_mean_mse_gain": _metric(observed),
        "p_value": _metric(p_value),
        "null_gain_quantiles": {
            "q05": _metric(float(np.quantile(null, 0.05))),
            "q50": _metric(float(np.quantile(null, 0.50))),
            "q95": _metric(float(np.quantile(null, 0.95))),
        },
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    data = _load_experiment_data(root, config)
    if data["summary"]["decision"] != "PASS_ITEM9_PROFILE_QUALITY_FLOOR":
        quality_pass = False
    else:
        quality_pass = True
    components = build_operator_components(data, config)
    losses, invalid_counts, compute = _candidate_galaxy_losses(data, components, config)
    qualifying, selections = _select_candidates(
        losses, invalid_counts, data, components, config
    )
    local, local_records = _local_control_oof(data, config)
    arrays = data["feature_arrays"]
    rar = np.log10(arrays["rar_speed_km_s"])
    newtonian = np.log10(arrays["newtonian_speed_km_s"])
    point_folds = np.asarray(data["folds"])[np.asarray(data["galaxy_index"])]
    baseline = np.empty(len(rar), dtype=np.float64)
    baseline_selections = []
    for record in local_records:
        fold = int(record["fold"])
        use_local = float(record["local_inner_mse"]) < float(record["rar_train_mse"])
        mask = point_folds == fold
        baseline[mask] = local[mask] if use_local else rar[mask]
        baseline_selections.append(
            {
                "fold": fold,
                "selected_model": "flexible_local_ridge" if use_local else "fixed_stellar_rar",
                "local_inner_mse": record["local_inner_mse"],
                "rar_train_mse": record["rar_train_mse"],
            }
        )
    cells = data["candidate_manifest"]["cells"]
    prior_cells = [row for row in cells if row["exact_prior_focusing_cell"]]
    if len(prior_cells) != 1:
        raise GravityItem9InteriorExteriorError("exact prior focusing cell scope changed")
    prior_prediction = _candidate_prediction(prior_cells[0], data, components, config)
    weights = _point_weights(data)
    baseline_metrics = _metrics(np.asarray(data["y"]), baseline, weights)
    qualifying_metrics = _metrics(np.asarray(data["y"]), qualifying, weights)
    rar_metrics = _metrics(np.asarray(data["y"]), rar, weights)
    newtonian_metrics = _metrics(np.asarray(data["y"]), newtonian, weights)
    local_metrics = _metrics(np.asarray(data["y"]), local, weights)
    prior_metrics = _metrics(np.asarray(data["y"]), prior_prediction, weights)
    baseline_mse = float(baseline_metrics["mse"])
    qualifying_mse = float(qualifying_metrics["mse"])
    relative_gain = (baseline_mse - qualifying_mse) / baseline_mse
    strata = _stratum_results(data, baseline, qualifying)
    minimum_stratum = int(config["evaluation"]["minimum_galaxies_per_reported_stratum"])

    y = np.asarray(data["y"])
    starts = np.asarray(data["starts"])
    counts = np.asarray(data["point_counts"], dtype=np.float64)
    paired = _paired_sign_flip(
        _galaxy_losses(y, baseline, starts, counts)
        - _galaxy_losses(y, qualifying, starts, counts),
        config,
    )
    gates = {
        "quality_count_and_fraction_pass": quality_pass,
        "qualifying_selector_r2_positive_overall": float(qualifying_metrics["r2"]) > 0,
        "qualifying_selector_beats_fixed_stellar_rar": (
            qualifying_mse < float(rar_metrics["mse"])
        ),
        "qualifying_selector_beats_strongest_local_control": (
            qualifying_mse < baseline_mse
        ),
        "qualifying_relative_mse_improvement_over_strongest_baseline_at_least_0_02": (
            relative_gain >= 0.02
        ),
        "qualifying_improvement_positive_in_all_primary_survey_families": all(
            float(row["qualifying_mse_gain"]) > 0
            for row in strata
            if row["dimension"] == "primary_rc_survey"
            and int(row["galaxies"]) >= minimum_stratum
        ),
        "qualifying_improvement_positive_in_all_distance_bins": all(
            float(row["qualifying_mse_gain"]) > 0
            for row in strata
            if row["dimension"] == "distance_bin"
            and int(row["galaxies"]) >= minimum_stratum
        ),
        "qualifying_improvement_positive_in_both_mass_strata": all(
            float(row["qualifying_mse_gain"]) > 0
            for row in strata
            if row["dimension"] == "mass_stratum"
            and int(row["galaxies"]) >= minimum_stratum
        ),
        "paired_sign_flip_p_at_most_0_05": float(paired["p_value"]) <= 0.05,
        "exact_prior_focusing_cell_improves_over_fixed_stellar_rar": (
            float(prior_metrics["mse"]) < float(rar_metrics["mse"])
        ),
        "reserved_confirmation_rotation_profiles_untouched": (
            int(
                data["profile_source"]["counts"][
                    "reserved_confirmation_rotation_entries_opened"
                ]
            )
            == 0
        ),
        "post_response_formula_generation_zero": (
            int(data["candidate_manifest"]["counts"]["post_response_formula_cells"])
            == 0
        ),
    }
    passed = all(gates.values())
    gate_counts = {"passed": sum(gates.values()), "required": len(gates)}
    selected_ordinals = Counter(int(row["selected_ordinal"]) for row in selections)
    selected_cells = [
        {
            "ordinal": ordinal,
            "outer_folds_selected": count,
            "cell": cells[ordinal],
        }
        for ordinal, count in sorted(selected_ordinals.items())
    ]
    receipt = _seal(
        {
            "schema_version": "invariant-gravity-item9-interior-exterior-result-1.0",
            "goal": config["goal"],
            "item_number": 9,
            "decision": (
                "PASS_ITEM9_INTERIOR_EXTERIOR_EXPLORATION"
                if passed
                else "REJECT_ITEM9_INTERIOR_EXTERIOR_EXPLORATION"
            ),
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "creativity_label": config["scientific_contract"]["creativity_label"],
            "inputs": {
                "config_path": CONFIG_PATH,
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "metadata_source_sha256": _sha256_file(
                    root / config["metadata_source_output"]
                ),
                "sample_manifest_sha256": _sha256_file(
                    root / config["sample_manifest_output"]
                ),
                "candidate_manifest_sha256": _sha256_file(
                    root / config["candidate_manifest_output"]
                ),
                "profile_source_sha256": _sha256_file(
                    root / config["profile_source_output"]
                ),
                "feature_sha256": _sha256_file(root / config["feature_output"]),
                "response_sha256": _sha256_file(root / config["response_output"]),
                "extraction_sha256": _sha256_file(root / config["extraction_output"]),
            },
            "counts": {
                "selected_exploration_galaxies": config["sample"]["exploration_count"],
                "quality_passing_exploration_galaxies": len(data["names"]),
                "quality_failed_exploration_galaxies": (
                    int(config["sample"]["exploration_count"]) - len(data["names"])
                ),
                "exploration_points": len(data["y"]),
                "reserved_confirmation_galaxies": config["sample"][
                    "reserved_confirmation_count"
                ],
                "reserved_confirmation_rotation_entries_opened": 0,
                "operators": data["candidate_manifest"]["counts"]["operators"],
                "candidate_formula_cells": data["candidate_manifest"]["counts"][
                    "candidate_formula_cells"
                ],
                "declared_equivalence_classes": data["candidate_manifest"]["counts"][
                    "declared_equivalence_classes"
                ],
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "compute": compute,
            "primary": {
                "newtonian_stellar_control": newtonian_metrics,
                "fixed_stellar_rar": rar_metrics,
                "flexible_local_control": local_metrics,
                "strongest_baseline_selector": {
                    "metrics": baseline_metrics,
                    "folds": baseline_selections,
                },
                "qualifying_selector": {
                    "metrics": qualifying_metrics,
                    "absolute_mse_improvement_over_strongest_baseline": _metric(
                        baseline_mse - qualifying_mse
                    ),
                    "relative_mse_improvement_over_strongest_baseline": _metric(
                        relative_gain
                    ),
                    "folds": selections,
                    "selected_cells": selected_cells,
                },
                "exact_prior_focusing_cell_replay": {
                    "candidate": prior_cells[0],
                    "metrics": prior_metrics,
                    "absolute_mse_improvement_over_fixed_stellar_rar": _metric(
                        float(rar_metrics["mse"]) - float(prior_metrics["mse"])
                    ),
                    "relative_mse_improvement_over_fixed_stellar_rar": _metric(
                        (float(rar_metrics["mse"]) - float(prior_metrics["mse"]))
                        / float(rar_metrics["mse"])
                    ),
                },
            },
            "local_control_folds": local_records,
            "strata": strata,
            "paired_sign_flip": paired,
            "gate_checks": gates,
            "gate_counts": gate_counts,
            "limitations": {
                "complete_baryonic_mass_used": False,
                "gas_mass_profile_used": False,
                "thin_disk_poisson_field_used": False,
                "spherical_enclosed_stellar_mass_approximation_used": True,
                "profile_archive_container_contains_unopened_confirmation_entries": True,
                "historical_novelty_adjudicated": False,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )
    return receipt


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    config = load_config(root)
    copy = dict(receipt)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem9InteriorExteriorError("Item 9 receipt content hash changed")
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem9InteriorExteriorError("Item 9 receipt freeze binding changed")
    if int(receipt["counts"]["reserved_confirmation_rotation_entries_opened"]) != 0:
        raise GravityItem9InteriorExteriorError("Item 9 confirmation boundary was opened")
    if int(receipt["counts"]["post_response_formula_cells"]) != 0:
        raise GravityItem9InteriorExteriorError("post-response formula cells were generated")
    if int(receipt["counts"]["paid_model_calls"]) != 0:
        raise GravityItem9InteriorExteriorError("unexpected paid model calls")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem9InteriorExteriorError("Item 9 receipt contains an overclaim")
    if int(receipt["counts"]["candidate_formula_cells"]) != int(
        config["operator_grammar"]["candidate_formula_cells"]
    ):
        raise GravityItem9InteriorExteriorError("receipt candidate count changed")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    path = root / config["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    stored = json.loads((root / config["output"]).read_text(encoding="utf-8"))
    rebuilt = build_receipt(root)
    stored_comparable = copy.deepcopy(stored)
    rebuilt_comparable = copy.deepcopy(rebuilt)
    for comparable in (stored_comparable, rebuilt_comparable):
        comparable.pop("content_sha256", None)
        comparable["compute"].pop("elapsed_seconds", None)
    if stored_comparable != rebuilt_comparable:
        raise GravityItem9InteriorExteriorError("Item 9 receipt drifted")
    validate_receipt(stored, root=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "acquire-metadata",
        "sample",
        "candidates",
        "acquire-profiles",
        "extract",
        "run",
        "check",
    ))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "acquire-metadata":
        print(write_metadata_source(args.root))
    elif args.command == "sample":
        print(write_sample_manifest(args.root))
    elif args.command == "candidates":
        print(write_candidate_manifest(args.root))
    elif args.command == "acquire-profiles":
        print(acquire_profile_archive(args.root))
    elif args.command == "extract":
        print(json.dumps({key: str(value) for key, value in extract_profiles(args.root).items()}))
    elif args.command == "run":
        print(write_receipt(args.root))
    else:
        check_receipt(args.root)


if __name__ == "__main__":
    main()
