"""Independent xCOLD-GASS/xGASS replay for gravity-roadmap Item 7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import gravity_item5_pressure_cross_support as cvcore
from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item7_baryonic_composition_xcold_xgass_v2.json"
SCIENTIFIC_FREEZE_COMMIT = "9eb2bf66e5ba6867cca8ffa81f4621c2881717fd"
VIZIER_ENDPOINT = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


class GravityItem7CompositionReplayError(RuntimeError):
    """Raised when the frozen Item 7 replay boundary or result drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem7CompositionReplayError("non-finite metric")
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
        "invariant-gravity-roadmap-item7-baryonic-composition-replay-config-2.0"
    ):
        raise GravityItem7CompositionReplayError("unexpected Item 7 replay config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem7CompositionReplayError("stable roadmap changed")
    attempt = config["attempt_1"]
    attempt_path = root / attempt["path"]
    if _sha256_file(attempt_path) != attempt["file_sha256"]:
        raise GravityItem7CompositionReplayError("Item 7 attempt-1 file changed")
    attempt_receipt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if attempt_receipt.get("content_sha256") != attempt["content_sha256"]:
        raise GravityItem7CompositionReplayError("Item 7 attempt-1 content changed")
    if attempt_receipt.get("decision") != attempt["required_decision"]:
        raise GravityItem7CompositionReplayError("Item 7 attempt 1 is not inconclusive")
    if float(
        attempt_receipt["primary"]["qualifying_selector"][
            "relative_mse_improvement_over_strongest_baseline"
        ]
    ) <= 0:
        raise GravityItem7CompositionReplayError("attempt 1 does not contain a positive lead")
    dependency = config["implementation_dependency"]
    if _sha256_file(root / dependency["path"]) != dependency["file_sha256"]:
        raise GravityItem7CompositionReplayError("Item 7 phase-family dependency changed")

    authorization = config["authorization"]
    forbidden_true = (
        "paid_model_calls_allowed",
        "reserved_confirmation_hi_width_responses_allowed",
        "dark_matter_or_dynamical_mass_allowed_as_predictor",
        "hi_or_co_line_width_allowed_as_predictor",
        "lensing_mass_allowed_as_predictor",
        "phangs_confirmation_opening_allowed",
    )
    if any(bool(authorization[name]) for name in forbidden_true):
        raise GravityItem7CompositionReplayError("Item 7 replay authorization changed")
    sample = config["sample"]
    exploration = [int(value) for value in sample["exploration"]]
    confirmation = [int(value) for value in sample["reserved_confirmation"]]
    if len(exploration) != sample["exploration_count"] or len(set(exploration)) != len(
        exploration
    ):
        raise GravityItem7CompositionReplayError("replay exploration sample changed")
    if len(confirmation) != sample["reserved_confirmation_count"] or len(set(confirmation)) != len(
        confirmation
    ):
        raise GravityItem7CompositionReplayError("replay confirmation sample changed")
    if set(exploration).intersection(confirmation):
        raise GravityItem7CompositionReplayError("replay sample roles overlap")
    if len(exploration) + len(confirmation) != sample["quality_passing_candidates"]:
        raise GravityItem7CompositionReplayError("replay candidate count changed")
    if config["prefreeze_audit"]["hi_width_values_read"] != 0:
        raise GravityItem7CompositionReplayError("prefreeze width boundary changed")
    if config["derivation"]["feature_builder_accepts_hi_width_response"]:
        raise GravityItem7CompositionReplayError("feature builder cannot accept a width")
    composition = config["sources"]["composition"]
    if set(composition["allowed_columns"]).intersection(composition["forbidden_columns"]):
        raise GravityItem7CompositionReplayError("composition source admits a width")
    return config


def build_sample_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    sample = config["sample"]
    exploration = [str(value) for value in sample["exploration"]]
    confirmation = [str(value) for value in sample["reserved_confirmation"]]
    assignments = cvcore.assign_folds(
        exploration,
        salt=str(config["cross_validation"]["fold_salt"]),
        folds=int(config["cross_validation"]["outer_folds"]),
    )
    salt = str(sample["selection_salt"])
    objects: list[dict[str, Any]] = []
    for role, identifiers in (
        ("exploration", exploration),
        ("reserved_confirmation", confirmation),
    ):
        for identifier in identifiers:
            objects.append(
                {
                    "gass_id": int(identifier),
                    "role": role,
                    "outer_fold": assignments.get(identifier),
                    "selection_digest": hashlib.sha256(
                        f"{salt}|{identifier}".encode()
                    ).hexdigest(),
                }
            )
    objects.sort(key=lambda row: (str(row["role"]), int(row["gass_id"])))
    return _seal(
        {
            "schema_version": "invariant-gravity-item7-composition-replay-sample-2.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM7_TARGET_BLIND_XCOLD_XGASS_REPLAY_SAMPLE",
            "selection": {
                "quality_rule": sample["quality_rule"],
                "stratification_rule": sample["stratification_rule"],
                "salt": salt,
                "selection_used_hi_width_response": False,
            },
            "counts": {
                "xcold_rows": config["prefreeze_audit"]["xcold_catalog_rows"],
                "good_hi_detection_metadata_rows": config["prefreeze_audit"][
                    "unique_good_hi_detection_metadata_rows"
                ],
                "quality_overlap": sample["quality_passing_candidates"],
                "exploration": len(exploration),
                "reserved_confirmation": len(confirmation),
            },
            "stratification_cells": sample["cell_counts"],
            "objects": objects,
            "prefreeze_boundary": {
                "hi_width_values_read": 0,
                "reserved_confirmation_predictors_blinded": False,
                "reserved_confirmation_hi_widths_blinded": True,
                "phangs_confirmation_rotation_curves_blinded": True,
                "forbidden_dynamical_or_dark_mass_values_used": 0,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_sample_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem7CompositionReplayError("replay sample content hash changed")
    roles = {"exploration": set(), "reserved_confirmation": set()}
    for row in manifest["objects"]:
        roles[str(row["role"])].add(int(row["gass_id"]))
    if roles["exploration"] != set(config["sample"]["exploration"]):
        raise GravityItem7CompositionReplayError("replay exploration identities changed")
    if roles["reserved_confirmation"] != set(config["sample"]["reserved_confirmation"]):
        raise GravityItem7CompositionReplayError("replay confirmation identities changed")
    boundary = manifest["prefreeze_boundary"]
    if boundary["hi_width_values_read"] != 0:
        raise GravityItem7CompositionReplayError("sample opened HI widths")
    if not boundary["reserved_confirmation_hi_widths_blinded"]:
        raise GravityItem7CompositionReplayError("replay confirmation boundary changed")
    if not boundary["phangs_confirmation_rotation_curves_blinded"]:
        raise GravityItem7CompositionReplayError("PHANGS confirmation boundary changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem7CompositionReplayError("sample contains an overclaim")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return path


def measure_replay_features(
    *,
    log_mstar: float,
    log_mhi: float,
    log_mmol: float,
    half_light_radius_kpc: float,
    concentration_index: float,
    inclination_deg: float,
    hi_signal_to_noise: float,
) -> dict[str, float]:
    """Build target-blind replay features without a line-width argument."""

    values = (
        log_mstar,
        log_mhi,
        log_mmol,
        half_light_radius_kpc,
        concentration_index,
        inclination_deg,
        hi_signal_to_noise,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise GravityItem7CompositionReplayError("non-finite replay observable")
    if (
        half_light_radius_kpc <= 0
        or concentration_index <= 0
        or not (0 < inclination_deg < 90)
        or hi_signal_to_noise <= 0
    ):
        raise GravityItem7CompositionReplayError("invalid replay observable")
    mstar = 10.0**log_mstar
    matomic = 1.36 * 10.0**log_mhi
    mmol = 10.0**log_mmol
    mbar = mstar + matomic + mmol
    fstar = mstar / mbar
    fatomic = matomic / mbar
    fmolecular = mmol / mbar
    fractions = (fstar, fatomic, fmolecular)
    entropy = -sum(value * math.log(value) for value in fractions) / math.log(3.0)
    molecular_to_atomic = math.log10(mmol / matomic)
    gas_to_stars = math.log10((matomic + mmol) / mstar)
    structure = math.log10(concentration_index / 2.6)
    phase_boundary = 4.0 * fatomic * fmolecular
    result = {
        "log_mstar": math.log10(mstar),
        "log_matomic": math.log10(matomic),
        "log_mmol": math.log10(mmol),
        "log_mbar": math.log10(mbar),
        "log_r50": math.log10(half_light_radius_kpc),
        "structure_proxy": structure,
        "inclination": inclination_deg,
        "log_hi_snr": math.log10(hi_signal_to_noise),
        "stellar_fraction": fstar,
        "atomic_fraction": fatomic,
        "molecular_fraction": fmolecular,
        "phase_entropy": entropy,
        "atomic_molecular_boundary": phase_boundary,
        "stellar_gas_boundary": 4.0 * fstar * (fatomic + fmolecular),
        "molecular_to_atomic": molecular_to_atomic,
        "gas_to_stars": gas_to_stars,
        "molecular_atomic_ratio_squared": molecular_to_atomic**2,
        "phase_entropy_squared": entropy**2,
        "phase_entropy_x_structure": entropy * structure,
        "molecular_atomic_x_structure": molecular_to_atomic * structure,
        "gas_stars_x_structure": gas_to_stars * structure,
        "phase_boundary_x_structure": phase_boundary * structure,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise GravityItem7CompositionReplayError("non-finite replay feature")
    return result


FEATURE_NAMES = (
    "log_mstar",
    "log_matomic",
    "log_mmol",
    "log_mbar",
    "log_r50",
    "structure_proxy",
    "inclination",
    "log_hi_snr",
    "stellar_fraction",
    "atomic_fraction",
    "molecular_fraction",
    "phase_entropy",
    "atomic_molecular_boundary",
    "stellar_gas_boundary",
    "molecular_to_atomic",
    "gas_to_stars",
    "molecular_atomic_ratio_squared",
    "phase_entropy_squared",
    "phase_entropy_x_structure",
    "molecular_atomic_x_structure",
    "gas_stars_x_structure",
    "phase_boundary_x_structure",
)


def _query_url(
    catalog_id: str,
    *,
    columns: Sequence[str],
    constraint_name: str | None = None,
    constraint_value: str | None = None,
    max_rows: int = 1000,
) -> str:
    parameters: list[tuple[str, str]] = [("-source", catalog_id)]
    parameters.extend(("-out", str(column)) for column in columns)
    if constraint_name is not None and constraint_value is not None:
        parameters.append((constraint_name, constraint_value))
    parameters.append(("-out.max", str(max_rows)))
    return f"{VIZIER_ENDPOINT}?{urllib.parse.urlencode(parameters)}"


def _fetch(url: str, *, attempts: int = 3) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (OSError, TimeoutError, urllib.error.URLError) as exc:  # pragma: no cover
            error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise GravityItem7CompositionReplayError(f"VizieR acquisition failed: {url}") from error


def _numeric_tsv_rows(payload: bytes, *, fields: int) -> list[list[str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem7CompositionReplayError("VizieR response is not UTF-8") from exc
    rows: list[list[str]] = []
    for line in lines:
        values = [field.strip() for field in line.split("\t")]
        if values and values[0].isdigit():
            if len(values) != fields:
                raise GravityItem7CompositionReplayError("VizieR schema changed")
            rows.append(values)
    return rows


def parse_composition_catalog(payload: bytes) -> dict[int, dict[str, Any]]:
    rows = _numeric_tsv_rows(payload, fields=8)
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not all(row[index] for index in range(2, 8)):
            continue
        identifier = int(row[0])
        result[identifier] = {
            "gass_id": identifier,
            "sdss_id": row[1],
            "inclination_deg": float(row[2]),
            "r50_kpc": float(row[3]),
            "log_mstar": float(row[4]),
            "co_detection_flag": int(row[5]),
            "log_mmol": float(row[6]),
            "concentration_index": float(row[7]),
        }
    return result


def parse_id_catalog(payload: bytes) -> set[int]:
    return {int(row[0]) for row in _numeric_tsv_rows(payload, fields=1)}


def parse_hi_response(
    payload: bytes, *, gass_id: int, source_id: str
) -> dict[str, Any]:
    rows = [row for row in _numeric_tsv_rows(payload, fields=6) if int(row[0]) == gass_id]
    if len(rows) != 1 or not all(rows[0][index] for index in range(1, 6)):
        raise GravityItem7CompositionReplayError("unexpected HI response row")
    row = rows[0]
    return {
        "gass_id": gass_id,
        "source_id": source_id,
        "width_corrected_km_s": float(row[1]),
        "width_error_km_s": float(row[2]),
        "hi_signal_to_noise": float(row[3]),
        "log_mhi": float(row[4]),
        "quality": int(row[5]),
    }


def _retrieval(url: str, payload: bytes) -> dict[str, Any]:
    return {"url": url, "payload_sha256": _sha256_bytes(payload), "bytes": len(payload)}


def acquire_exploration(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem7CompositionReplayError(
            "scientific freeze commit is not bound; HI-width access forbidden"
        )
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    sources = config["sources"]
    composition_url = _query_url(
        sources["composition"]["catalog_id"],
        columns=sources["composition"]["allowed_columns"],
    )
    composition_payload = _fetch(composition_url)
    composition = parse_composition_catalog(composition_payload)

    source_by_id: dict[int, Mapping[str, Any]] = {}
    id_retrievals: list[dict[str, Any]] = []
    for source in sources["hi_releases"]:
        url = _query_url(str(source["catalog_id"]), columns=["GASS"])
        payload = _fetch(url)
        for identifier in parse_id_catalog(payload):
            if identifier in source_by_id:
                raise GravityItem7CompositionReplayError("HI release identifiers overlap")
            source_by_id[identifier] = source
        id_retrievals.append({"source_id": source["id"], **_retrieval(url, payload)})

    records: list[dict[str, Any]] = []
    for identifier in sorted(int(value) for value in config["sample"]["exploration"]):
        if identifier not in composition or identifier not in source_by_id:
            raise GravityItem7CompositionReplayError("frozen replay identifier not found")
        source = source_by_id[identifier]
        response_columns = [
            "GASS",
            "W50c",
            "e_W50",
            "S/N",
            "logMHI",
            str(source["quality_column"]),
        ]
        response_url = _query_url(
            str(source["catalog_id"]),
            columns=response_columns,
            constraint_name="GASS",
            constraint_value=str(identifier),
            max_rows=10,
        )
        response_payload = _fetch(response_url)
        records.append(
            {
                "gass_id": identifier,
                "composition": composition[identifier],
                "hi": parse_hi_response(
                    response_payload, gass_id=identifier, source_id=str(source["id"])
                ),
                "response_retrieval": _retrieval(response_url, response_payload),
            }
        )
    return _seal(
        {
            "schema_version": "invariant-gravity-item7-composition-replay-source-2.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM7_XCOLD_XGASS_EXPLORATION_SOURCE_ACQUISITION",
            "preregistration": {
                "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
                "config_path": CONFIG_PATH,
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "sample_manifest_path": config["sample_manifest_output"],
                "sample_manifest_sha256": _sha256_file(sample_path),
            },
            "global_retrievals": {
                "composition": _retrieval(composition_url, composition_payload),
                "hi_identifier_maps": id_retrievals,
            },
            "boundary": {
                "exploration_primary_response_queries": len(records),
                "exploration_primary_response_rows": len(records),
                "reserved_confirmation_primary_response_queries": 0,
                "phangs_confirmation_response_queries": 0,
                "dynamical_or_dark_mass_values_acquired": 0,
                "lensing_mass_values_acquired": 0,
                "paid_model_calls": 0,
                "hi_mass_and_width_share_spectrum": True,
            },
            "records": records,
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_source_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem7CompositionReplayError("replay source content hash changed")
    if manifest["preregistration"]["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem7CompositionReplayError("replay freeze binding changed")
    if {int(row["gass_id"]) for row in manifest["records"]} != set(
        config["sample"]["exploration"]
    ):
        raise GravityItem7CompositionReplayError("replay source scope changed")
    source_by_id = {
        str(source["id"]): source for source in config["sources"]["hi_releases"]
    }
    for record in manifest["records"]:
        identifier = int(record["gass_id"])
        source = source_by_id[str(record["hi"]["source_id"])]
        parsed = urllib.parse.parse_qs(
            urllib.parse.urlsplit(record["response_retrieval"]["url"]).query
        )
        expected_columns = [
            "GASS",
            "W50c",
            "e_W50",
            "S/N",
            "logMHI",
            str(source["quality_column"]),
        ]
        if (
            parsed.get("-source") != [str(source["catalog_id"])]
            or parsed.get("-out") != expected_columns
            or parsed.get("GASS") != [str(identifier)]
            or parsed.get("-out.max") != ["10"]
        ):
            raise GravityItem7CompositionReplayError(
                "HI response retrieval was not a one-galaxy frozen query"
            )
    boundary = manifest["boundary"]
    if boundary["exploration_primary_response_rows"] != config["sample"]["exploration_count"]:
        raise GravityItem7CompositionReplayError("replay response count changed")
    for field in (
        "reserved_confirmation_primary_response_queries",
        "phangs_confirmation_response_queries",
        "dynamical_or_dark_mass_values_acquired",
        "lensing_mass_values_acquired",
        "paid_model_calls",
    ):
        if int(boundary[field]) != 0:
            raise GravityItem7CompositionReplayError(f"replay boundary opened: {field}")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem7CompositionReplayError("replay source contains an overclaim")


def write_source_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["source_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(acquire_exploration(root)) + b"\n")
    return path


def _features_for_record(record: Mapping[str, Any]) -> dict[str, float]:
    composition = record["composition"]
    hi = record["hi"]
    return measure_replay_features(
        log_mstar=float(composition["log_mstar"]),
        log_mhi=float(hi["log_mhi"]),
        log_mmol=float(composition["log_mmol"]),
        half_light_radius_kpc=float(composition["r50_kpc"]),
        concentration_index=float(composition["concentration_index"]),
        inclination_deg=float(composition["inclination_deg"]),
        hi_signal_to_noise=float(hi["hi_signal_to_noise"]),
    )


def _quality_failure(record: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    quality = config["quality"]
    composition = record["composition"]
    hi = record["hi"]
    inclination = float(composition["inclination_deg"])
    if int(composition["co_detection_flag"]) != int(
        quality["required_co_detection_flag"]
    ):
        return "CO detection flag changed"
    if int(hi["quality"]) != int(quality["required_hi_quality"]):
        return "HI quality flag changed"
    if not (
        float(quality["minimum_inclination_deg"])
        <= inclination
        <= float(quality["maximum_inclination_deg"])
    ):
        return "inclination outside frozen range"
    width = float(hi["width_corrected_km_s"])
    error = float(hi["width_error_km_s"])
    if width <= 0 or error <= 0 or width - error <= 0:
        return "invalid HI width uncertainty"
    if error / width > float(quality["maximum_fractional_width_error"]):
        return "HI fractional width error too large"
    return None


def build_feature_rows(
    source: Mapping[str, Any], *, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_source_manifest(source, config)
    feature_map = {
        int(record["gass_id"]): _features_for_record(record) for record in source["records"]
    }
    ordered_mass = sorted(feature_map, key=lambda key: (feature_map[key]["log_mbar"], key))
    ordered_ratio = sorted(
        feature_map, key=lambda key: (feature_map[key]["molecular_to_atomic"], key)
    )
    mass_quartile = {
        key: f"q{min(3, (4 * index) // len(ordered_mass)) + 1}"
        for index, key in enumerate(ordered_mass)
    }
    mass_stratum = {
        key: ("low_mass" if index < len(ordered_mass) / 2 else "high_mass")
        for index, key in enumerate(ordered_mass)
    }
    ratio_stratum = {
        key: ("low_molecular_ratio" if index < len(ordered_ratio) / 2 else "high_molecular_ratio")
        for index, key in enumerate(ordered_ratio)
    }
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in source["records"]:
        identifier = int(record["gass_id"])
        failure = _quality_failure(record, config)
        if failure is not None:
            failures.append({"gass_id": identifier, "reason": failure})
            continue
        composition = record["composition"]
        hi = record["hi"]
        inclination = math.radians(float(composition["inclination_deg"]))
        width = float(hi["width_corrected_km_s"])
        error = float(hi["width_error_km_s"])
        correction = 2.0 * math.sin(inclination)
        features = feature_map[identifier]
        rows.append(
            {
                "cluster": str(identifier),
                "gass_id": identifier,
                "response_log10_velocity": math.log10(width / correction),
                "response_upper_log10_velocity": math.log10((width + error) / correction),
                "response_lower_log10_velocity": math.log10((width - error) / correction),
                "mass_quartile": mass_quartile[identifier],
                "mass_stratum": mass_stratum[identifier],
                "molecular_ratio_stratum": ratio_stratum[identifier],
                "survey_stratum": (
                    "gass_low" if str(hi["source_id"]) == "gass_low" else "gass_high"
                ),
                **features,
            }
        )
    rows.sort(key=lambda row: int(row["gass_id"]))
    summary = _seal(
        {
            "schema_version": "invariant-gravity-item7-composition-replay-extraction-2.0",
            "goal": config["goal"],
            "decision": (
                "PASS_ITEM7_COMPOSITION_REPLAY_REPRESENTATION_QUALITY"
                if not failures
                else "FAIL_ITEM7_COMPOSITION_REPLAY_REPRESENTATION_QUALITY"
            ),
            "counts": {
                "exploration_galaxies": config["sample"]["exploration_count"],
                "quality_passing_galaxies": len(rows),
                "quality_failures": len(failures),
                "reserved_confirmation_response_accesses": 0,
                "phangs_confirmation_response_accesses": 0,
            },
            "failures": failures,
            "source_manifest_content_sha256": source["content_sha256"],
        }
    )
    return rows, summary


def write_extraction(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    config = load_config(root)
    source = json.loads((root / config["source_manifest_output"]).read_text(encoding="utf-8"))
    rows, summary = build_feature_rows(source, config=config)
    feature_path = root / config["feature_output"]
    summary_path = root / config["extraction_summary_output"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "gass_id",
        "response_log10_velocity",
        "response_upper_log10_velocity",
        "response_lower_log10_velocity",
        "mass_quartile",
        "mass_stratum",
        "molecular_ratio_stratum",
        "survey_stratum",
        *FEATURE_NAMES,
    )
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return feature_path, summary_path


def load_feature_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (root / config["feature_output"]).open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            row: dict[str, Any] = {
                "cluster": raw["gass_id"],
                "gass_id": int(raw["gass_id"]),
                "response_log10_sigma": float(raw["response_log10_velocity"]),
                "mass_quartile": raw["mass_quartile"],
                "mass_stratum": raw["mass_stratum"],
                "molecular_ratio_stratum": raw["molecular_ratio_stratum"],
                "survey_stratum": raw["survey_stratum"],
            }
            for field in (
                "response_log10_velocity",
                "response_upper_log10_velocity",
                "response_lower_log10_velocity",
                *FEATURE_NAMES,
            ):
                row[field] = float(raw[field])
            rows.append(row)
    return rows


def _array(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows])


def _predictions_array(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, float]
) -> np.ndarray:
    return np.asarray([float(predictions[str(row["gass_id"])]) for row in rows])


def _metrics_for_response(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    *,
    response: str = "response_log10_velocity",
) -> dict[str, str]:
    observed = _array(rows, response)
    predicted = _predictions_array(rows, predictions)
    return {
        "mse": _metric(cvcore._mse(observed, predicted)),
        "r2": _metric(cvcore._r2(observed, predicted)),
    }


def _permutation_test(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_models: Sequence[Mapping[str, Any]],
    qualifying_models: Sequence[Mapping[str, Any]],
    observed_improvement: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    count = int(config["cross_validation"]["permutation_count"])
    seed = int.from_bytes(
        hashlib.sha256(str(config["cross_validation"]["permutation_salt"]).encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    original = _array(rows, "response_log10_velocity")
    indices_by_stratum = {
        stratum: [index for index, row in enumerate(rows) if row["mass_quartile"] == stratum]
        for stratum in ("q1", "q2", "q3", "q4")
    }
    null: list[float] = []
    for _ in range(count):
        shuffled = original.copy()
        for indices in indices_by_stratum.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        permuted = [
            dict(
                row,
                response_log10_velocity=float(shuffled[index]),
                response_log10_sigma=float(shuffled[index]),
            )
            for index, row in enumerate(rows)
        ]
        baseline, _ = cvcore._nested_predictions(
            permuted, models=baseline_models, config=config, detailed=False
        )
        qualifying, _ = cvcore._nested_predictions(
            permuted, models=qualifying_models, config=config, detailed=False
        )
        observed = _array(permuted, "response_log10_velocity")
        null.append(
            cvcore._mse(observed, _predictions_array(permuted, baseline))
            - cvcore._mse(observed, _predictions_array(permuted, qualifying))
        )
    p_value = (1 + sum(value >= observed_improvement for value in null)) / (count + 1)
    return {
        "permutations": count,
        "observed_mse_improvement": _metric(observed_improvement),
        "p_value": _metric(p_value),
        "null_improvement_quantiles": {
            "q05": _metric(float(np.quantile(null, 0.05))),
            "q50": _metric(float(np.quantile(null, 0.50))),
            "q95": _metric(float(np.quantile(null, 0.95))),
        },
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample_path = root / config["sample_manifest_output"]
    source_path = root / config["source_manifest_output"]
    feature_path = root / config["feature_output"]
    extraction_path = root / config["extraction_summary_output"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_manifest(source, config)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    rows = load_feature_rows(root, config)
    models = config["model_families"]
    baseline_models = [model for model in models if not model["qualifying"]]
    qualifying_models = [model for model in models if model["qualifying"]]

    unrestricted_predictions, unrestricted_folds = cvcore._nested_predictions(
        rows, models=models, config=config, detailed=True
    )
    baseline_predictions, baseline_folds = cvcore._nested_predictions(
        rows, models=baseline_models, config=config, detailed=True
    )
    qualifying_predictions, qualifying_folds = cvcore._nested_predictions(
        rows, models=qualifying_models, config=config, detailed=True
    )
    observed = _array(rows, "response_log10_velocity")
    baseline_array = _predictions_array(rows, baseline_predictions)
    qualifying_array = _predictions_array(rows, qualifying_predictions)
    baseline_mse = cvcore._mse(observed, baseline_array)
    qualifying_mse = cvcore._mse(observed, qualifying_array)
    improvement = baseline_mse - qualifying_mse
    relative_improvement = improvement / baseline_mse if baseline_mse > 0 else -1.0

    individual_baselines: dict[str, Any] = {}
    beats_each = True
    for model in baseline_models:
        predictions, _ = cvcore._nested_predictions(rows, models=[model], config=config, detailed=False)
        metrics = _metrics_for_response(rows, predictions)
        gain = float(metrics["mse"]) - qualifying_mse
        individual_baselines[str(model["id"])] = {
            **metrics,
            "qualifying_mse_gain": _metric(gain),
        }
        beats_each = beats_each and gain > 0

    strata: dict[str, Any] = {}
    improvements: dict[str, list[float]] = {
        "mass_stratum": [],
        "molecular_ratio_stratum": [],
        "survey_stratum": [],
    }
    for field, labels in (
        ("mass_stratum", ("low_mass", "high_mass")),
        (
            "molecular_ratio_stratum",
            ("low_molecular_ratio", "high_molecular_ratio"),
        ),
        ("survey_stratum", ("gass_high", "gass_low")),
    ):
        for label in labels:
            subset = [row for row in rows if row[field] == label]
            baseline_metrics = _metrics_for_response(subset, baseline_predictions)
            qualifying_metrics = _metrics_for_response(subset, qualifying_predictions)
            gain = float(baseline_metrics["mse"]) - float(qualifying_metrics["mse"])
            improvements[field].append(gain)
            strata[f"{field}:{label}"] = {
                "count": len(subset),
                "baseline": baseline_metrics,
                "qualifying": qualifying_metrics,
                "qualifying_mse_gain": _metric(gain),
            }

    envelopes: dict[str, Any] = {}
    envelope_improvements: list[float] = []
    for field in (
        "response_lower_log10_velocity",
        "response_upper_log10_velocity",
    ):
        baseline_value = cvcore._mse(_array(rows, field), baseline_array)
        qualifying_value = cvcore._mse(_array(rows, field), qualifying_array)
        gain = baseline_value - qualifying_value
        envelope_improvements.append(gain)
        envelopes[field] = {
            "baseline_mse": _metric(baseline_value),
            "qualifying_mse": _metric(qualifying_value),
            "qualifying_mse_gain": _metric(gain),
        }

    permutation = _permutation_test(
        rows,
        baseline_models=baseline_models,
        qualifying_models=qualifying_models,
        observed_improvement=improvement,
        config=config,
    )
    admission = config["exploration_admission"]
    quality_pass = (
        extraction["decision"] == "PASS_ITEM7_COMPOSITION_REPLAY_REPRESENTATION_QUALITY"
        and len(rows) == config["sample"]["exploration_count"]
    )
    gates = {
        "all_96_exploration_galaxies_pass_frozen_quality": quality_pass,
        "unrestricted_selector_qualifying_in_at_least_4_of_5_folds": sum(
            bool(fold["selected_qualifying"]) for fold in unrestricted_folds
        )
        >= int(admission["unrestricted_selector_qualifying_in_at_least_folds"]),
        "qualifying_selector_r2_positive_overall": cvcore._r2(observed, qualifying_array) > 0,
        "qualifying_selector_beats_each_nonqualifying_baseline_overall": beats_each,
        "qualifying_relative_mse_improvement_over_strongest_baseline_at_least_0_02": relative_improvement
        >= float(admission["qualifying_relative_mse_improvement_over_strongest_baseline_at_least"]),
        "qualifying_improvement_positive_in_both_mass_strata": all(
            value > 0 for value in improvements["mass_stratum"]
        ),
        "qualifying_improvement_positive_in_both_molecular_ratio_strata": all(
            value > 0 for value in improvements["molecular_ratio_stratum"]
        ),
        "qualifying_improvement_positive_in_high_and_low_mass_survey_releases": all(
            value > 0 for value in improvements["survey_stratum"]
        ),
        "mass_stratified_permutation_p_at_most_0_05": float(permutation["p_value"])
        <= float(admission["mass_stratified_permutation_p_at_most"]),
        "width_error_envelopes_do_not_reverse_improvement": all(
            value >= 0 for value in envelope_improvements
        ),
        "reserved_confirmation_targets_untouched": source["boundary"][
            "reserved_confirmation_primary_response_queries"
        ]
        == 0
        and source["boundary"]["phangs_confirmation_response_queries"] == 0,
    }
    if not quality_pass:
        decision = "INCONCLUSIVE_ITEM7_COMPOSITION_REPLAY_QUALITY_GATE"
    elif all(gates.values()):
        decision = "PASS_ITEM7_COMPOSITION_REPLAY_REQUIRES_CONFIRMATION_AUTHORIZATION"
    else:
        decision = "REJECT_ITEM7_COMPOSITION_REPLAY_EXPLORATION"
    return _seal(
        {
            "schema_version": "invariant-gravity-item7-composition-replay-result-2.0",
            "goal": config["goal"],
            "item_number": 7,
            "attempt": 2,
            "decision": decision,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "creativity_label": config["scientific_contract"]["creativity_label"],
            "inputs": {
                "config_path": CONFIG_PATH,
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "sample_manifest_path": config["sample_manifest_output"],
                "sample_manifest_sha256": _sha256_file(sample_path),
                "source_manifest_path": config["source_manifest_output"],
                "source_manifest_sha256": _sha256_file(source_path),
                "source_manifest_content_sha256": source["content_sha256"],
                "feature_path": config["feature_output"],
                "feature_sha256": _sha256_file(feature_path),
                "extraction_summary_path": config["extraction_summary_output"],
                "extraction_summary_sha256": _sha256_file(extraction_path),
                "attempt_1_path": config["attempt_1"]["path"],
                "attempt_1_file_sha256": config["attempt_1"]["file_sha256"],
            },
            "counts": {
                "exploration_galaxies": len(rows),
                "reserved_confirmation_galaxies": config["sample"][
                    "reserved_confirmation_count"
                ],
                "reserved_confirmation_target_accesses": 0,
                "phangs_confirmation_target_accesses": 0,
                "paid_model_calls": 0,
                "permutation_nested_cv_runs": config["cross_validation"]["permutation_count"],
            },
            "models": config["model_families"],
            "primary": {
                "unrestricted": {
                    "metrics": _metrics_for_response(rows, unrestricted_predictions),
                    "folds": unrestricted_folds,
                },
                "strongest_nonqualifying_selector": {
                    "metrics": _metrics_for_response(rows, baseline_predictions),
                    "folds": baseline_folds,
                },
                "qualifying_selector": {
                    "metrics": _metrics_for_response(rows, qualifying_predictions),
                    "folds": qualifying_folds,
                    "absolute_mse_improvement_over_strongest_baseline": _metric(improvement),
                    "relative_mse_improvement_over_strongest_baseline": _metric(
                        relative_improvement
                    ),
                },
                "individual_nonqualifying_baselines": individual_baselines,
            },
            "strata": strata,
            "response_error_envelopes": envelopes,
            "permutation": permutation,
            "gate_checks": gates,
            "gate_counts": {"passed": sum(gates.values()), "required": len(gates)},
            "limitations": {
                "hi_mass_and_width_share_spectrum": True,
                "resolved_baryonic_geometry_used": False,
                "same_target_as_phangs_attempt": False,
                "xcold_confirmation_opened": False,
                "phangs_confirmation_opened": False,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(receipt)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem7CompositionReplayError("replay result content hash changed")
    load_config(root)
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem7CompositionReplayError("replay result freeze changed")
    if receipt["counts"]["reserved_confirmation_target_accesses"] != 0:
        raise GravityItem7CompositionReplayError("xGASS confirmation was opened")
    if receipt["counts"]["phangs_confirmation_target_accesses"] != 0:
        raise GravityItem7CompositionReplayError("PHANGS confirmation was opened")
    passed = sum(bool(value) for value in receipt["gate_checks"].values())
    if receipt["gate_counts"] != {"passed": passed, "required": len(receipt["gate_checks"])}:
        raise GravityItem7CompositionReplayError("replay gate count changed")
    passing = receipt["decision"].startswith("PASS_ITEM7_")
    if passing != all(bool(value) for value in receipt["gate_checks"].values()):
        raise GravityItem7CompositionReplayError("replay decision does not match gates")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem7CompositionReplayError("replay result contains an overclaim")
    if receipt != build_receipt(root):
        raise GravityItem7CompositionReplayError("replay result does not replay")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    stored = json.loads((root / config["output"]).read_text(encoding="utf-8"))
    validate_receipt(stored, root=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sample", "acquire", "extract", "run", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "sample":
        print(write_sample_manifest(args.root))
    elif args.command == "acquire":
        print(write_source_manifest(args.root))
    elif args.command == "extract":
        print(*write_extraction(args.root), sep="\n")
    elif args.command == "run":
        print(write_receipt(args.root))
    else:
        check_receipt(args.root)


if __name__ == "__main__":
    main()
