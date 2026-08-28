"""Frozen cluster cross-support experiment for gravity-roadmap Item 5 attempt 2."""

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

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item5_pressure_cross_support_v2.json"
SCIENTIFIC_FREEZE_COMMIT = "e7a861b656cda74e9e92ea63cd864cfcfdf2555e"
VIZIER_ENDPOINT = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


class GravityItem5PressureCrossSupportError(RuntimeError):
    """Raised when the frozen sample, sources, or experiment drift."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem5PressureCrossSupportError("non-finite metric")
    return f"{float(value):.12e}"


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item5-pressure-cross-support-config-2.0"
    ):
        raise GravityItem5PressureCrossSupportError("unexpected Item 5 attempt-2 schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem5PressureCrossSupportError("stable roadmap changed")
    attempt_1 = config["attempt_1"]
    attempt_path = root / attempt_1["path"]
    if _sha256_file(attempt_path) != attempt_1["file_sha256"]:
        raise GravityItem5PressureCrossSupportError("Item 5 attempt-1 file changed")
    attempt_receipt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if attempt_receipt.get("content_sha256") != attempt_1["content_sha256"]:
        raise GravityItem5PressureCrossSupportError("Item 5 attempt-1 content changed")
    if attempt_receipt.get("decision") != attempt_1["required_decision"]:
        raise GravityItem5PressureCrossSupportError("Item 5 attempt 1 is not inconclusive")

    authorization = config["authorization"]
    if authorization["paid_model_calls_allowed"]:
        raise GravityItem5PressureCrossSupportError("paid model calls are forbidden")
    if authorization["reserved_confirmation_velocity_responses_allowed"]:
        raise GravityItem5PressureCrossSupportError("confirmation responses must stay sealed")
    if authorization["inferred_spt_mass_allowed_as_predictor"]:
        raise GravityItem5PressureCrossSupportError("inferred SPT mass must remain forbidden")
    if authorization["sz_equivalent_velocity_dispersion_allowed"]:
        raise GravityItem5PressureCrossSupportError("SZ-equivalent dispersion is circular")

    sources = config["sources"]
    thermal = sources["thermal_predictor"]
    if set(thermal["allowed_columns"]).intersection(thermal["forbidden_columns"]):
        raise GravityItem5PressureCrossSupportError("thermal source admits a forbidden mass")
    for source in sources["robustness_responses"]:
        if set(source["allowed_columns"]).intersection(source.get("forbidden_columns", [])):
            raise GravityItem5PressureCrossSupportError("robustness source admits sigmaSPT")

    sample = config["sample"]
    exploration = [str(value) for value in sample["exploration"]]
    confirmation = [str(value) for value in sample["reserved_confirmation"]]
    if len(exploration) != sample["exploration_count"] or len(set(exploration)) != len(exploration):
        raise GravityItem5PressureCrossSupportError("exploration sample changed")
    if len(confirmation) != sample["reserved_confirmation_count"] or len(set(confirmation)) != len(
        confirmation
    ):
        raise GravityItem5PressureCrossSupportError("confirmation sample changed")
    if set(exploration).intersection(confirmation):
        raise GravityItem5PressureCrossSupportError("sample roles overlap")
    if len(exploration) + len(confirmation) != sample["quality_passing_candidates"]:
        raise GravityItem5PressureCrossSupportError("quality-passing sample count changed")
    act_prefix = {str(value) for value in sample["exploration_act_prefix"]}
    if act_prefix != {"0232-5257", "0235-5121", "0304-4921"}:
        raise GravityItem5PressureCrossSupportError("Bayliss identifier-prefix repair changed")
    if not act_prefix.issubset(exploration):
        raise GravityItem5PressureCrossSupportError("ACT-prefix repair escaped exploration")
    if config["prefreeze_audit"]["response_values_read"] != 0:
        raise GravityItem5PressureCrossSupportError("prefreeze response boundary changed")
    if config["derivation"]["feature_builder_accepts_velocity_response"]:
        raise GravityItem5PressureCrossSupportError("feature builder cannot accept response")
    return config


def assign_folds(names: Sequence[str], *, salt: str, folds: int) -> dict[str, int]:
    if folds < 2 or len(set(names)) < folds:
        raise GravityItem5PressureCrossSupportError("invalid whole-cluster folds")
    ordered = sorted(
        set(names),
        key=lambda name: hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(),
    )
    return {name: index % folds for index, name in enumerate(ordered)}


def build_sample_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    sample = config["sample"]
    exploration = [str(value) for value in sample["exploration"]]
    confirmation = [str(value) for value in sample["reserved_confirmation"]]
    assignments = assign_folds(
        exploration,
        salt=str(config["cross_validation"]["fold_salt"]),
        folds=int(config["cross_validation"]["outer_folds"]),
    )
    salt = str(sample["selection_salt"])
    objects: list[dict[str, Any]] = []
    for role, names in (
        ("exploration", exploration),
        ("reserved_confirmation", confirmation),
    ):
        for name in names:
            objects.append(
                {
                    "cluster": name,
                    "role": role,
                    "outer_fold": assignments.get(name),
                    "selection_digest": hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(),
                }
            )
    objects.sort(key=lambda row: (str(row["role"]), str(row["cluster"])))
    manifest = {
        "schema_version": "invariant-gravity-item5-pressure-cross-support-sample-2.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM5_TARGET_BLIND_SPT_BAYLISS_SAMPLE",
        "selection": {
            "rule": sample["quality_rule"],
            "salt": salt,
            "selection_used_velocity_response": False,
        },
        "counts": {
            "metadata_overlap_candidates": sample["overlap_candidates"],
            "quality_passing_candidates": sample["quality_passing_candidates"],
            "exploration": len(exploration),
            "reserved_confirmation": len(confirmation),
        },
        "objects": objects,
        "prefreeze_boundary": {
            "thermal_predictor_and_quality_metadata_rows_audited": config["prefreeze_audit"][
                "spt_bayliss_overlap"
            ],
            "primary_velocity_response_values_read": 0,
            "robustness_velocity_response_values_read": 0,
            "reserved_confirmation_predictors_blinded": False,
            "reserved_confirmation_velocity_responses_blinded": True,
            "forbidden_mass_values_used": 0,
        },
        "claims": dict(config["claim_boundaries"]),
    }
    return _seal(manifest)


def validate_sample_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem5PressureCrossSupportError("sample content hash changed")
    expected = {
        "metadata_overlap_candidates": config["sample"]["overlap_candidates"],
        "quality_passing_candidates": config["sample"]["quality_passing_candidates"],
        "exploration": config["sample"]["exploration_count"],
        "reserved_confirmation": config["sample"]["reserved_confirmation_count"],
    }
    if manifest["counts"] != expected:
        raise GravityItem5PressureCrossSupportError("sample counts changed")
    roles = {"exploration": set(), "reserved_confirmation": set()}
    for row in manifest["objects"]:
        roles[str(row["role"])].add(str(row["cluster"]))
    if roles["exploration"] != set(config["sample"]["exploration"]):
        raise GravityItem5PressureCrossSupportError("exploration identities changed")
    if roles["reserved_confirmation"] != set(config["sample"]["reserved_confirmation"]):
        raise GravityItem5PressureCrossSupportError("confirmation identities changed")
    boundary = manifest["prefreeze_boundary"]
    if boundary["primary_velocity_response_values_read"] != 0:
        raise GravityItem5PressureCrossSupportError("sample opened primary responses")
    if boundary["robustness_velocity_response_values_read"] != 0:
        raise GravityItem5PressureCrossSupportError("sample opened robustness responses")
    if not boundary["reserved_confirmation_velocity_responses_blinded"]:
        raise GravityItem5PressureCrossSupportError("confirmation response boundary changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem5PressureCrossSupportError("sample contains an overclaim")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return path


def measure_pressure_features(
    *,
    xi: float,
    theta_arcmin: float,
    ysz_1e6_arcmin2: float,
    e_ysz_1e6_arcmin2: float,
    redshift: float,
    beam_floor_arcmin: float = 0.25,
) -> dict[str, float]:
    """Build target-blind thermal-support features from direct catalog observables."""

    values = (xi, theta_arcmin, ysz_1e6_arcmin2, e_ysz_1e6_arcmin2, redshift)
    if not all(math.isfinite(float(value)) for value in values):
        raise GravityItem5PressureCrossSupportError("non-finite pressure observable")
    if xi <= 0 or theta_arcmin < 0 or ysz_1e6_arcmin2 <= 0 or e_ysz_1e6_arcmin2 <= 0:
        raise GravityItem5PressureCrossSupportError("invalid pressure observable")
    if redshift <= 0 or beam_floor_arcmin <= 0:
        raise GravityItem5PressureCrossSupportError("invalid redshift or beam floor")

    omega_m = 0.3
    omega_lambda = 0.7
    hubble = 70.0
    speed_km_s = 299792.458
    grid = np.linspace(0.0, redshift, 4097, dtype=np.float64)
    e_grid = np.sqrt(omega_m * (1.0 + grid) ** 3 + omega_lambda)
    comoving_mpc = speed_km_s / hubble * float(np.trapezoid(1.0 / e_grid, grid))
    angular_diameter_mpc = comoving_mpc / (1.0 + redshift)
    e_z = math.sqrt(omega_m * (1.0 + redshift) ** 3 + omega_lambda)
    arcmin_radian = math.pi / (180.0 * 60.0)
    theta_eff = math.sqrt(theta_arcmin**2 + beam_floor_arcmin**2)
    y_arcmin2 = ysz_1e6_arcmin2 * 1.0e-6
    y_sr = y_arcmin2 * arcmin_radian**2
    extent_mpc = angular_diameter_mpc * theta_eff * arcmin_radian
    y_snr = ysz_1e6_arcmin2 / e_ysz_1e6_arcmin2

    log_y = math.log10(y_arcmin2)
    log_y_snr = math.log10(y_snr)
    log_xi = math.log10(xi)
    log_theta = math.log10(theta_eff)
    log_da = math.log10(angular_diameter_mpc)
    log_ez = math.log10(e_z)
    log_extent = math.log10(extent_mpc)
    thermal_energy = math.log10(y_sr * angular_diameter_mpc**2 * e_z)
    pressure_surface = math.log10(y_arcmin2 / theta_eff**2)
    coherence = math.log10(xi / y_snr)
    aperture_ratio = math.log10(0.75 / theta_eff)
    result = {
        "log_y_sz": log_y,
        "log_y_snr": log_y_snr,
        "log_xi": log_xi,
        "log_theta_eff": log_theta,
        "log_da": log_da,
        "log_ez": log_ez,
        "log1pz": math.log10(1.0 + redshift),
        "thermal_energy_proxy": thermal_energy,
        "log_extent": log_extent,
        "pressure_surface_proxy": pressure_surface,
        "detection_aperture_coherence": coherence,
        "aperture_filter_ratio": aperture_ratio,
        "coherence_x_extent": coherence * log_extent,
        "coherence_x_surface": coherence * pressure_surface,
        "coherence_x_aperture": coherence * aperture_ratio,
        "surface_x_aperture": pressure_surface * aperture_ratio,
        "coherence_squared": coherence**2,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise GravityItem5PressureCrossSupportError("non-finite derived pressure feature")
    return result


def _query_url(
    catalog_id: str,
    *,
    columns: Sequence[str],
    constraint_name: str,
    constraint_value: str,
) -> str:
    parameters: list[tuple[str, str]] = [
        ("-source", catalog_id),
        (constraint_name, constraint_value),
    ]
    parameters.extend(("-out", str(column)) for column in columns)
    parameters.append(("-out.max", "10"))
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
    raise GravityItem5PressureCrossSupportError(f"VizieR acquisition failed: {url}") from error


def _tsv_rows(payload: bytes, identifier_prefixes: Sequence[str]) -> list[list[str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem5PressureCrossSupportError("VizieR response is not UTF-8") from exc
    return [
        [field.strip() for field in line.split("\t")]
        for line in lines
        if any(line.startswith(prefix) for prefix in identifier_prefixes)
    ]


def _one_row(
    payload: bytes,
    *,
    prefixes: Sequence[str],
    expected_fields: int,
    allow_missing: bool = False,
) -> list[str] | None:
    rows = _tsv_rows(payload, prefixes)
    if allow_missing and not rows:
        return None
    if len(rows) != 1 or len(rows[0]) != expected_fields:
        raise GravityItem5PressureCrossSupportError(
            f"unexpected VizieR row count/schema: {len(rows)} rows"
        )
    return rows[0]


def parse_thermal_payload(payload: bytes, *, cluster: str) -> dict[str, Any]:
    row = _one_row(payload, prefixes=(f"J{cluster}",), expected_fields=7)
    assert row is not None
    if row[0] != f"J{cluster}":
        raise GravityItem5PressureCrossSupportError("thermal identifier changed")
    return {
        "cluster": cluster,
        "xi": float(row[1]),
        "theta_arcmin": float(row[2]),
        "ysz_1e6_arcmin2": float(row[3]),
        "e_ysz_1e6_arcmin2": float(row[4]),
        "catalog_redshift": float(row[5]) if row[5] else None,
        "catalog_redshift_note": row[6] or None,
    }


def parse_metadata_payload(payload: bytes, *, cluster: str) -> dict[str, Any]:
    prefix = _bayliss_identifier(cluster)
    row = _one_row(payload, prefixes=(prefix,), expected_fields=8)
    assert row is not None
    if row[0] != prefix:
        raise GravityItem5PressureCrossSupportError("metadata identifier changed")
    return {
        "cluster": cluster,
        "n_spectra": int(row[1]),
        "n_members": int(row[2]),
        "n_passive_poststarburst": int(row[3]),
        "n_star_forming": int(row[4]),
        "redshift": float(row[5]),
        "redshift_error": float(row[6]),
        "references": row[7],
    }


def parse_primary_response_payload(payload: bytes, *, cluster: str) -> dict[str, Any]:
    prefix = _bayliss_identifier(cluster)
    row = _one_row(payload, prefixes=(prefix,), expected_fields=3)
    assert row is not None
    return {
        "cluster": cluster,
        "sigma_km_s": float(row[1]),
        "sigma_error_km_s": float(row[2]),
    }


def parse_robustness_payload(
    payload: bytes,
    *,
    cluster: str,
    source_id: str,
) -> list[dict[str, Any]]:
    row = _one_row(
        payload,
        prefixes=(f"J{cluster}", f"SPT-CLJ{cluster}"),
        expected_fields=5,
        allow_missing=True,
    )
    if row is None:
        return []
    values: list[dict[str, Any]] = []
    if source_id == "J/ApJS/227/3/table2":
        labels = (("biweight", 1, 2), ("gapper", 3, 4))
    elif source_id == "J/ApJ/792/45/table4":
        labels = (("gapper", 1, 2), ("biweight", 3, 4))
    else:
        raise GravityItem5PressureCrossSupportError("unknown robustness source")
    for estimator, value_index, error_index in labels:
        if not row[value_index] or not row[error_index]:
            continue
        values.append(
            {
                "source_id": source_id,
                "estimator": estimator,
                "sigma_km_s": float(row[value_index]),
                "sigma_error_km_s": float(row[error_index]),
            }
        )
    return values


def _retrieval(url: str, payload: bytes) -> dict[str, Any]:
    return {"url": url, "payload_sha256": _sha256_bytes(payload), "bytes": len(payload)}


def _bayliss_identifier(cluster: str) -> str:
    prefix = "ACT-CLJ" if cluster in {"0232-5257", "0235-5121", "0304-4921"} else "SPT-CLJ"
    return f"{prefix}{cluster}"


def acquire_exploration(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem5PressureCrossSupportError(
            "scientific freeze commit is not bound; response access forbidden"
        )
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    exploration = sorted(
        str(row["cluster"]) for row in sample["objects"] if row["role"] == "exploration"
    )
    if set(exploration) != set(config["sample"]["exploration"]):
        raise GravityItem5PressureCrossSupportError("exploration acquisition scope changed")

    sources = config["sources"]
    thermal_source = sources["thermal_predictor"]
    primary_source = sources["primary_response"]
    records: list[dict[str, Any]] = []
    alternative_queries = 0
    alternative_rows = 0
    for cluster in exploration:
        thermal_url = _query_url(
            str(thermal_source["catalog_id"]),
            columns=thermal_source["allowed_columns"],
            constraint_name="SPT-CL",
            constraint_value=f"J{cluster}",
        )
        metadata_url = _query_url(
            str(primary_source["catalog_id"]),
            columns=primary_source["metadata_columns"],
            constraint_name="Cl",
            constraint_value=_bayliss_identifier(cluster),
        )
        response_url = _query_url(
            str(primary_source["catalog_id"]),
            columns=primary_source["response_columns"],
            constraint_name="Cl",
            constraint_value=_bayliss_identifier(cluster),
        )
        thermal_payload = _fetch(thermal_url)
        metadata_payload = _fetch(metadata_url)
        response_payload = _fetch(response_url)
        robustness: list[dict[str, Any]] = []
        robustness_retrievals: list[dict[str, Any]] = []
        for source in sources["robustness_responses"]:
            source_id = str(source["catalog_id"])
            robust_url = _query_url(
                source_id,
                columns=source["allowed_columns"],
                constraint_name="SPT-CL",
                constraint_value=f"J{cluster}",
            )
            robust_payload = _fetch(robust_url)
            parsed = parse_robustness_payload(robust_payload, cluster=cluster, source_id=source_id)
            robustness.extend(parsed)
            alternative_queries += 1
            alternative_rows += len(parsed)
            robustness_retrievals.append(
                {"source_id": source_id, **_retrieval(robust_url, robust_payload)}
            )
        records.append(
            {
                "cluster": cluster,
                "thermal": parse_thermal_payload(thermal_payload, cluster=cluster),
                "metadata": parse_metadata_payload(metadata_payload, cluster=cluster),
                "primary_response": parse_primary_response_payload(
                    response_payload, cluster=cluster
                ),
                "robustness_responses": robustness,
                "retrievals": {
                    "thermal": _retrieval(thermal_url, thermal_payload),
                    "metadata": _retrieval(metadata_url, metadata_payload),
                    "primary_response": _retrieval(response_url, response_payload),
                    "robustness": robustness_retrievals,
                },
            }
        )
    manifest = {
        "schema_version": "invariant-gravity-item5-pressure-cross-support-source-2.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM5_EXPLORATION_SOURCE_ACQUISITION",
        "preregistration": {
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "config_path": CONFIG_PATH,
            "config_sha256": _sha256_file(root / CONFIG_PATH),
            "sample_manifest_path": config["sample_manifest_output"],
            "sample_manifest_sha256": _sha256_file(sample_path),
        },
        "acquisition_history": config["postfreeze_acquisition_audit"],
        "boundary": {
            "exploration_thermal_queries": len(exploration),
            "exploration_metadata_queries": len(exploration),
            "successful_exploration_primary_response_queries": len(exploration),
            "cumulative_exploration_primary_response_queries": len(exploration)
            + int(config["postfreeze_acquisition_audit"]["primary_response_queries_issued"]),
            "exploration_primary_response_rows": len(records),
            "cumulative_primary_response_rows_returned": len(records)
            + int(config["postfreeze_acquisition_audit"]["primary_response_rows_returned"]),
            "successful_exploration_robustness_response_queries": alternative_queries,
            "cumulative_exploration_robustness_response_queries": alternative_queries
            + int(config["postfreeze_acquisition_audit"]["robustness_response_queries_issued"]),
            "exploration_robustness_response_rows": alternative_rows,
            "reserved_confirmation_primary_response_queries": 0,
            "reserved_confirmation_robustness_response_queries": 0,
            "inferred_spt_mass_values_acquired": 0,
            "sigma_spt_values_acquired": 0,
            "paid_model_calls": 0,
        },
        "records": records,
        "claims": dict(config["claim_boundaries"]),
    }
    return _seal(manifest)


def validate_source_manifest(manifest: Mapping[str, Any], *, config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem5PressureCrossSupportError("source content hash changed")
    if manifest["preregistration"]["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem5PressureCrossSupportError("scientific freeze binding changed")
    records = manifest["records"]
    if {str(row["cluster"]) for row in records} != set(config["sample"]["exploration"]):
        raise GravityItem5PressureCrossSupportError("source exploration scope changed")
    boundary = manifest["boundary"]
    if boundary["exploration_primary_response_rows"] != config["sample"]["exploration_count"]:
        raise GravityItem5PressureCrossSupportError("primary response count changed")
    if boundary["reserved_confirmation_primary_response_queries"] != 0:
        raise GravityItem5PressureCrossSupportError("confirmation response was queried")
    if boundary["reserved_confirmation_robustness_response_queries"] != 0:
        raise GravityItem5PressureCrossSupportError("confirmation robustness was queried")
    if boundary["inferred_spt_mass_values_acquired"] != 0:
        raise GravityItem5PressureCrossSupportError("inferred SPT mass was acquired")
    if boundary["sigma_spt_values_acquired"] != 0:
        raise GravityItem5PressureCrossSupportError("sigmaSPT was acquired")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem5PressureCrossSupportError("source contains an overclaim")


def write_source_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["source_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(acquire_exploration(root)) + b"\n")
    return path


def _quality_failure(record: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    thermal = record["thermal"]
    metadata = record["metadata"]
    response = record["primary_response"]
    quality = config["quality"]
    if int(metadata["n_members"]) < int(quality["minimum_member_galaxies"]):
        return "insufficient member galaxies"
    if float(thermal["ysz_1e6_arcmin2"]) / float(thermal["e_ysz_1e6_arcmin2"]) < float(
        quality["minimum_ysz_signal_to_noise"]
    ):
        return "insufficient YSZ signal-to-noise"
    sigma = float(response["sigma_km_s"])
    sigma_error = float(response["sigma_error_km_s"])
    if sigma <= 0 or sigma_error <= 0:
        return "nonpositive primary velocity dispersion or uncertainty"
    if sigma_error / sigma > float(quality["maximum_primary_fractional_sigma_error"]):
        return "primary velocity dispersion uncertainty too large"
    return None


FEATURE_NAMES = (
    "log_y_sz",
    "log_y_snr",
    "log_xi",
    "log_theta_eff",
    "log_da",
    "log_ez",
    "log1pz",
    "thermal_energy_proxy",
    "log_extent",
    "pressure_surface_proxy",
    "detection_aperture_coherence",
    "aperture_filter_ratio",
    "coherence_x_extent",
    "coherence_x_surface",
    "coherence_x_aperture",
    "surface_x_aperture",
    "coherence_squared",
)


def build_feature_rows(
    source: Mapping[str, Any], *, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_source_manifest(source, config=config)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in source["records"]:
        failure = _quality_failure(record, config)
        if failure is not None:
            failures.append({"cluster": str(record["cluster"]), "reason": failure})
            continue
        thermal = record["thermal"]
        metadata = record["metadata"]
        response = record["primary_response"]
        features = measure_pressure_features(
            xi=float(thermal["xi"]),
            theta_arcmin=float(thermal["theta_arcmin"]),
            ysz_1e6_arcmin2=float(thermal["ysz_1e6_arcmin2"]),
            e_ysz_1e6_arcmin2=float(thermal["e_ysz_1e6_arcmin2"]),
            redshift=float(metadata["redshift"]),
        )
        alternatives = [
            {
                "source_id": str(value["source_id"]),
                "estimator": str(value["estimator"]),
                "log10_sigma": math.log10(float(value["sigma_km_s"])),
                "fractional_error": float(value["sigma_error_km_s"]) / float(value["sigma_km_s"]),
            }
            for value in record["robustness_responses"]
            if float(value["sigma_km_s"]) > 0 and float(value["sigma_error_km_s"]) > 0
        ]
        row: dict[str, Any] = {
            "cluster": str(record["cluster"]),
            "redshift": float(metadata["redshift"]),
            "theta_arcmin": float(thermal["theta_arcmin"]),
            "n_members": int(metadata["n_members"]),
            "response_log10_sigma": math.log10(float(response["sigma_km_s"])),
            "response_fractional_error": float(response["sigma_error_km_s"])
            / float(response["sigma_km_s"]),
            "redshift_stratum": "low_z" if float(metadata["redshift"]) < 0.6 else "high_z",
            "extent_stratum": "unresolved"
            if float(thermal["theta_arcmin"]) <= 0.25
            else "resolved",
            "alternative_responses": alternatives,
            **features,
        }
        rows.append(row)
    rows.sort(key=lambda row: str(row["cluster"]))
    summary = {
        "schema_version": "invariant-gravity-item5-pressure-cross-support-extraction-2.0",
        "goal": config["goal"],
        "decision": (
            "PASS_ITEM5_PRESSURE_CROSS_SUPPORT_REPRESENTATION_QUALITY"
            if not failures
            else "FAIL_ITEM5_PRESSURE_CROSS_SUPPORT_REPRESENTATION_QUALITY"
        ),
        "counts": {
            "exploration_clusters": config["sample"]["exploration_count"],
            "quality_passing_clusters": len(rows),
            "quality_failures": len(failures),
            "pooled_alternative_response_rows": sum(
                len(row["alternative_responses"]) for row in rows
            ),
            "reserved_confirmation_response_accesses": 0,
        },
        "failures": failures,
        "source_manifest_content_sha256": source["content_sha256"],
    }
    return rows, _seal(summary)


def write_extraction(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    config = load_config(root)
    source_path = root / config["source_manifest_output"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows, summary = build_feature_rows(source, config=config)
    feature_path = root / config["feature_output"]
    summary_path = root / config["extraction_summary_output"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "cluster",
        "redshift",
        "theta_arcmin",
        "n_members",
        "response_log10_sigma",
        "response_fractional_error",
        "redshift_stratum",
        "extent_stratum",
        *FEATURE_NAMES,
        "alternative_responses_json",
    )
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            output = {name: row[name] for name in fields if name != "alternative_responses_json"}
            output["alternative_responses_json"] = json.dumps(
                row["alternative_responses"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(output)
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return feature_path, summary_path


def load_feature_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / config["feature_output"]
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            row: dict[str, Any] = {
                "cluster": raw["cluster"],
                "redshift_stratum": raw["redshift_stratum"],
                "extent_stratum": raw["extent_stratum"],
                "n_members": int(raw["n_members"]),
                "alternative_responses": json.loads(raw["alternative_responses_json"]),
            }
            for name in (
                "redshift",
                "theta_arcmin",
                "response_log10_sigma",
                "response_fractional_error",
                *FEATURE_NAMES,
            ):
                row[name] = float(raw[name])
            rows.append(row)
    return rows


def _fit(
    training: Sequence[Mapping[str, Any]],
    testing: Sequence[Mapping[str, Any]],
    *,
    model: Mapping[str, Any],
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    features = [str(value) for value in model["features"]]
    fixed = {str(key): float(value) for key, value in model["fixed_coefficients"].items()}
    y = np.asarray([float(row["response_log10_sigma"]) for row in training])
    fixed_train = np.asarray(
        [
            sum(coefficient * float(row[name]) for name, coefficient in fixed.items())
            for row in training
        ]
    )
    fixed_test = np.asarray(
        [
            sum(coefficient * float(row[name]) for name, coefficient in fixed.items())
            for row in testing
        ]
    )
    residual = y - fixed_train
    if not features:
        intercept = float(np.mean(residual))
        return fixed_test + intercept, {
            "intercept": _metric(intercept),
            "coefficients": {},
            "fixed_coefficients": {key: _metric(value) for key, value in fixed.items()},
        }
    x = np.asarray([[float(row[name]) for name in features] for row in training])
    xt = np.asarray([[float(row[name]) for name in features] for row in testing])
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    xs = (x - means) / scales
    xts = (xt - means) / scales
    design = np.column_stack((np.ones(len(xs)), xs))
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ residual)
    predictions = fixed_test + np.column_stack((np.ones(len(xts)), xts)) @ coefficients
    raw_coefficients = coefficients[1:] / scales
    raw_intercept = float(coefficients[0] - np.sum(raw_coefficients * means))
    return predictions, {
        "intercept": _metric(raw_intercept),
        "coefficients": {
            name: _metric(value) for name, value in zip(features, raw_coefficients, strict=True)
        },
        "fixed_coefficients": {key: _metric(value) for key, value in fixed.items()},
    }


def _mse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((observed - predicted) ** 2))


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= 0:
        return float("-inf")
    return 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator


def _candidate_alphas(model: Mapping[str, Any], config: Mapping[str, Any]) -> list[float]:
    return (
        [0.0]
        if not model["features"]
        else [float(value) for value in config["cross_validation"]["ridge_penalties"]]
    )


def _inner_select(
    training: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[Mapping[str, Any]],
    outer_fold: int,
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], float, list[dict[str, Any]]]:
    folds = int(config["cross_validation"]["inner_folds"])
    assignments = assign_folds(
        [str(row["cluster"]) for row in training],
        salt=f"{config['cross_validation']['inner_salt']}|{outer_fold}",
        folds=folds,
    )
    candidates: list[tuple[float, int, float, Mapping[str, Any]]] = []
    diagnostics: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for alpha in _candidate_alphas(model, config):
            observed_all: list[float] = []
            predicted_all: list[float] = []
            for fold in range(folds):
                inner_train = [row for row in training if assignments[str(row["cluster"])] != fold]
                inner_test = [row for row in training if assignments[str(row["cluster"])] == fold]
                predictions, _ = _fit(inner_train, inner_test, model=model, alpha=alpha)
                observed_all.extend(float(row["response_log10_sigma"]) for row in inner_test)
                predicted_all.extend(float(value) for value in predictions)
            mse = _mse(np.asarray(observed_all), np.asarray(predicted_all))
            candidates.append((mse, model_index, alpha, model))
            diagnostics.append(
                {"model": model["id"], "alpha": _metric(alpha), "inner_mse": _metric(mse)}
            )
    mse, _, alpha, model = min(candidates, key=lambda value: (value[0], value[1], value[2]))
    del mse
    return model, alpha, diagnostics


def _nested_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    detailed: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    assignments = assign_folds(
        [str(row["cluster"]) for row in rows],
        salt=str(config["cross_validation"]["fold_salt"]),
        folds=int(config["cross_validation"]["outer_folds"]),
    )
    predictions: dict[str, float] = {}
    fold_records: list[dict[str, Any]] = []
    for fold in range(int(config["cross_validation"]["outer_folds"])):
        training = [row for row in rows if assignments[str(row["cluster"])] != fold]
        testing = [row for row in rows if assignments[str(row["cluster"])] == fold]
        model, alpha, diagnostics = _inner_select(
            training, models=models, outer_fold=fold, config=config
        )
        values, parameters = _fit(training, testing, model=model, alpha=alpha)
        for row, value in zip(testing, values, strict=True):
            predictions[str(row["cluster"])] = float(value)
        record: dict[str, Any] = {
            "fold": fold,
            "test_clusters": sorted(str(row["cluster"]) for row in testing),
            "selected_model": model["id"],
            "selected_qualifying": bool(model["qualifying"]),
            "selected_alpha": _metric(alpha),
            "parameters": parameters,
        }
        if detailed:
            record["inner_candidates"] = diagnostics
        fold_records.append(record)
    if set(predictions) != {str(row["cluster"]) for row in rows}:
        raise GravityItem5PressureCrossSupportError("outer predictions incomplete")
    return predictions, fold_records


def _predicted_array(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, float]
) -> np.ndarray:
    return np.asarray([float(predictions[str(row["cluster"])]) for row in rows])


def _metrics(rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, float]) -> dict[str, str]:
    observed = np.asarray([float(row["response_log10_sigma"]) for row in rows])
    predicted = _predicted_array(rows, predictions)
    return {"mse": _metric(_mse(observed, predicted)), "r2": _metric(_r2(observed, predicted))}


def _permutation_test(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_models: Sequence[Mapping[str, Any]],
    qualifying_models: Sequence[Mapping[str, Any]],
    observed_improvement: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    count = int(config["cross_validation"]["permutation_count"])
    seed_bytes = hashlib.sha256(
        str(config["cross_validation"]["permutation_salt"]).encode()
    ).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
    null: list[float] = []
    indices_by_stratum = {
        stratum: [index for index, row in enumerate(rows) if row["redshift_stratum"] == stratum]
        for stratum in ("low_z", "high_z")
    }
    original = np.asarray([float(row["response_log10_sigma"]) for row in rows])
    for _ in range(count):
        shuffled = original.copy()
        for indices in indices_by_stratum.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        permuted = [
            dict(row, response_log10_sigma=float(shuffled[index])) for index, row in enumerate(rows)
        ]
        baseline, _ = _nested_predictions(
            permuted, models=baseline_models, config=config, detailed=False
        )
        qualifying, _ = _nested_predictions(
            permuted, models=qualifying_models, config=config, detailed=False
        )
        observed = np.asarray([float(row["response_log10_sigma"]) for row in permuted])
        null.append(
            _mse(observed, _predicted_array(permuted, baseline))
            - _mse(observed, _predicted_array(permuted, qualifying))
        )
    p_value = (1 + sum(value >= observed_improvement for value in null)) / (count + 1)
    return {
        "permutations": count,
        "p_value": _metric(p_value),
        "observed_mse_improvement": _metric(observed_improvement),
        "null_improvement_quantiles": {
            "q05": _metric(float(np.quantile(null, 0.05))),
            "q50": _metric(float(np.quantile(null, 0.50))),
            "q95": _metric(float(np.quantile(null, 0.95))),
        },
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    source_path = root / config["source_manifest_output"]
    extraction_path = root / config["extraction_summary_output"]
    feature_path = root / config["feature_output"]
    sample_path = root / config["sample_manifest_output"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_manifest(source, config=config)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    rows = load_feature_rows(root, config)
    models = config["model_families"]
    baseline_models = [model for model in models if not model["qualifying"]]
    qualifying_models = [model for model in models if model["qualifying"]]

    unrestricted_predictions, unrestricted_folds = _nested_predictions(
        rows, models=models, config=config, detailed=True
    )
    baseline_predictions, baseline_folds = _nested_predictions(
        rows, models=baseline_models, config=config, detailed=True
    )
    qualifying_predictions, qualifying_folds = _nested_predictions(
        rows, models=qualifying_models, config=config, detailed=True
    )
    observed = np.asarray([float(row["response_log10_sigma"]) for row in rows])
    baseline_array = _predicted_array(rows, baseline_predictions)
    qualifying_array = _predicted_array(rows, qualifying_predictions)
    baseline_mse = _mse(observed, baseline_array)
    qualifying_mse = _mse(observed, qualifying_array)
    improvement = baseline_mse - qualifying_mse
    relative_improvement = improvement / baseline_mse if baseline_mse > 0 else float("-inf")

    individual_baselines: dict[str, Any] = {}
    beats_each = True
    for model in baseline_models:
        predictions, _ = _nested_predictions(rows, models=[model], config=config, detailed=False)
        metrics = _metrics(rows, predictions)
        model_mse = float(metrics["mse"])
        gain = model_mse - qualifying_mse
        individual_baselines[str(model["id"])] = {**metrics, "qualifying_mse_gain": _metric(gain)}
        beats_each = beats_each and gain > 0

    strata: dict[str, Any] = {}
    for field, labels in (
        ("redshift_stratum", ("low_z", "high_z")),
        ("extent_stratum", ("unresolved", "resolved")),
    ):
        for label in labels:
            subset = [row for row in rows if row[field] == label]
            if len(subset) < 2:
                continue
            key = f"{field}:{label}"
            strata[key] = {
                "count": len(subset),
                "baseline": _metrics(subset, baseline_predictions),
                "qualifying": _metrics(subset, qualifying_predictions),
            }

    alternative_observed: list[float] = []
    alternative_baseline: list[float] = []
    alternative_qualifying: list[float] = []
    for row in rows:
        for alternative in row["alternative_responses"]:
            alternative_observed.append(float(alternative["log10_sigma"]))
            alternative_baseline.append(float(baseline_predictions[str(row["cluster"])]))
            alternative_qualifying.append(float(qualifying_predictions[str(row["cluster"])]))
    alt_observed = np.asarray(alternative_observed)
    alt_baseline = np.asarray(alternative_baseline)
    alt_qualifying = np.asarray(alternative_qualifying)
    if len(alt_observed):
        alt_baseline_mse = _mse(alt_observed, alt_baseline)
        alt_qualifying_mse = _mse(alt_observed, alt_qualifying)
        alt_improvement = alt_baseline_mse - alt_qualifying_mse
    else:
        alt_baseline_mse = alt_qualifying_mse = 0.0
        alt_improvement = -1.0

    permutation = _permutation_test(
        rows,
        baseline_models=baseline_models,
        qualifying_models=qualifying_models,
        observed_improvement=improvement,
        config=config,
    )
    admission = config["exploration_admission"]
    quality_pass = (
        extraction["decision"] == "PASS_ITEM5_PRESSURE_CROSS_SUPPORT_REPRESENTATION_QUALITY"
        and len(rows) == config["sample"]["exploration_count"]
    )
    redshift_r2 = [
        float(strata[f"redshift_stratum:{label}"]["qualifying"]["r2"])
        for label in ("low_z", "high_z")
    ]
    gates = {
        "all_44_exploration_clusters_pass_frozen_quality": quality_pass,
        "unrestricted_selector_qualifying_in_at_least_4_of_5_folds": sum(
            bool(fold["selected_qualifying"]) for fold in unrestricted_folds
        )
        >= int(admission["unrestricted_selector_qualifying_in_at_least_folds"]),
        "qualifying_selector_r2_positive_overall": _r2(observed, qualifying_array) > 0,
        "qualifying_selector_r2_positive_in_both_redshift_strata": all(
            value > 0 for value in redshift_r2
        ),
        "qualifying_selector_beats_each_nonqualifying_baseline_overall": beats_each,
        "qualifying_relative_mse_improvement_over_strongest_baseline_at_least_0_02": relative_improvement
        >= float(admission["qualifying_relative_mse_improvement_over_strongest_baseline_at_least"]),
        "redshift_stratified_permutation_p_at_most_0_05": float(permutation["p_value"])
        <= float(admission["redshift_stratified_permutation_p_at_most"]),
        "pooled_alternative_dispersion_robustness_does_not_reverse_improvement": len(alt_observed)
        >= int(config["quality"]["minimum_pooled_alternative_response_rows"])
        and alt_improvement >= 0,
        "reserved_confirmation_targets_untouched": source["boundary"][
            "reserved_confirmation_primary_response_queries"
        ]
        == 0
        and source["boundary"]["reserved_confirmation_robustness_response_queries"] == 0,
    }
    if not quality_pass:
        decision = "INCONCLUSIVE_ITEM5_PRESSURE_CROSS_SUPPORT_QUALITY_GATE"
    elif all(gates.values()):
        decision = (
            "PASS_ITEM5_PRESSURE_CROSS_SUPPORT_EXPLORATION_REQUIRES_CONFIRMATION_AUTHORIZATION"
        )
    else:
        decision = "REJECT_ITEM5_PRESSURE_CROSS_SUPPORT_EXPLORATION"

    receipt = {
        "schema_version": "invariant-gravity-item5-pressure-cross-support-result-2.0",
        "goal": config["goal"],
        "item_number": 5,
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
        },
        "counts": {
            "exploration_clusters": len(rows),
            "reserved_confirmation_clusters": config["sample"]["reserved_confirmation_count"],
            "reserved_confirmation_target_accesses": 0,
            "pooled_alternative_response_rows": len(alt_observed),
            "paid_model_calls": 0,
            "permutation_nested_cv_runs": int(config["cross_validation"]["permutation_count"]),
        },
        "models": config["model_families"],
        "primary": {
            "unrestricted": {
                "metrics": _metrics(rows, unrestricted_predictions),
                "folds": unrestricted_folds,
            },
            "strongest_nonqualifying_selector": {
                "metrics": _metrics(rows, baseline_predictions),
                "folds": baseline_folds,
            },
            "qualifying_selector": {
                "metrics": _metrics(rows, qualifying_predictions),
                "folds": qualifying_folds,
                "mse_improvement_over_strongest_baseline": _metric(improvement),
                "relative_mse_improvement_over_strongest_baseline": _metric(relative_improvement),
            },
            "individual_nonqualifying_baselines": individual_baselines,
            "strata": strata,
        },
        "alternative_response_robustness": {
            "rows": len(alt_observed),
            "baseline_mse": _metric(alt_baseline_mse),
            "qualifying_mse": _metric(alt_qualifying_mse),
            "qualifying_mse_improvement": _metric(alt_improvement),
        },
        "permutation": permutation,
        "gate_checks": gates,
        "gate_counts": {"passed": sum(gates.values()), "required": len(gates)},
        "claim_boundaries": dict(config["claim_boundaries"]),
        "next_action": (
            "Request explicit authorization before opening any of the 18 confirmation dispersions."
            if decision.startswith("PASS_")
            else "Retain the exact thermal-to-collisionless pressure-coherence counterexamples, synthesize both Item 5 attempts without retuning opened responses, and decide whether Item 5 is complete before Item 6."
        ),
    }
    return _seal(receipt)


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(receipt)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem5PressureCrossSupportError("receipt content hash changed")
    load_config(root)
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem5PressureCrossSupportError("receipt freeze binding changed")
    gates = receipt["gate_checks"]
    expected = (
        "INCONCLUSIVE_ITEM5_PRESSURE_CROSS_SUPPORT_QUALITY_GATE"
        if not gates["all_44_exploration_clusters_pass_frozen_quality"]
        else (
            "PASS_ITEM5_PRESSURE_CROSS_SUPPORT_EXPLORATION_REQUIRES_CONFIRMATION_AUTHORIZATION"
            if all(gates.values())
            else "REJECT_ITEM5_PRESSURE_CROSS_SUPPORT_EXPLORATION"
        )
    )
    if receipt["decision"] != expected:
        raise GravityItem5PressureCrossSupportError("decision does not follow frozen gates")
    if receipt["counts"]["reserved_confirmation_target_accesses"] != 0:
        raise GravityItem5PressureCrossSupportError("receipt opened confirmation")
    if any(bool(value) for value in receipt["claim_boundaries"].values()):
        raise GravityItem5PressureCrossSupportError("receipt contains an overclaim")
    if receipt["inputs"]["config_sha256"] != _sha256_file(root / CONFIG_PATH):
        raise GravityItem5PressureCrossSupportError("receipt config binding changed")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    path = root / config["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def replay(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    stored = json.loads((root / config["output"]).read_text(encoding="utf-8"))
    rebuilt = build_receipt(root)
    if rebuilt != stored:
        raise GravityItem5PressureCrossSupportError("stored receipt does not replay")
    validate_receipt(stored, root=root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("sample", "acquire", "extract", "experiment", "replay"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.root.resolve()
    if arguments.command == "sample":
        print(write_sample_manifest(root))
    elif arguments.command == "acquire":
        print(write_source_manifest(root))
    elif arguments.command == "extract":
        print(*write_extraction(root), sep="\n")
    elif arguments.command == "experiment":
        print(write_receipt(root))
    elif arguments.command == "replay":
        replay(root)
        print("PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
