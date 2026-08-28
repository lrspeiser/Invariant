"""Target-blind 2M++ field-gradient and curvature test for gravity roadmap Item 8."""

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

CONFIG_PATH = "configs/gravity_item8_field_gradients_curvature_2mpp_v1.json"
SCIENTIFIC_FREEZE_COMMIT = "PENDING_ITEM8_FIELD_CURVATURE_SCIENTIFIC_FREEZE"
VIZIER_ENDPOINT = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


class GravityItem8FieldCurvatureError(RuntimeError):
    """Raised when the frozen Item 8 boundary or replay invariant drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem8FieldCurvatureError("non-finite metric")
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
        "invariant-gravity-roadmap-item8-field-gradients-curvature-config-1.0"
    ):
        raise GravityItem8FieldCurvatureError("unexpected Item 8 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem8FieldCurvatureError("stable roadmap changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem8FieldCurvatureError("Item 7 synthesis file changed")
    receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if receipt.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem8FieldCurvatureError("Item 7 synthesis content changed")
    if receipt.get("decision") != predecessor["required_decision"]:
        raise GravityItem8FieldCurvatureError("Item 7 did not authorize Item 8")

    authorization = config["authorization"]
    forbidden_true = (
        "paid_model_calls_allowed",
        "reserved_confirmation_group_dispersion_allowed",
        "member_redshift_allowed",
        "virial_or_halo_mass_allowed_as_predictor",
        "virial_radius_allowed_as_predictor",
        "lensing_mass_allowed_as_predictor",
    )
    if any(bool(authorization[name]) for name in forbidden_true):
        raise GravityItem8FieldCurvatureError("Item 8 authorization changed")
    if int(authorization["direct_lensing_likelihood_evaluations_allowed"]) != 0:
        raise GravityItem8FieldCurvatureError("lensing gate opened")
    if int(config["prefreeze_audit"]["published_group_dispersion_values_read"]) != 0:
        raise GravityItem8FieldCurvatureError("prefreeze response boundary changed")
    if config["derivation"]["feature_builder_accepts_group_dispersion"]:
        raise GravityItem8FieldCurvatureError("feature builder cannot accept dispersion")
    if config["derivation"]["feature_builder_accepts_member_redshift"]:
        raise GravityItem8FieldCurvatureError("feature builder cannot accept member redshift")

    sample = config["sample"]
    exploration = [int(value) for value in sample["exploration"]]
    confirmation = [int(value) for value in sample["reserved_confirmation"]]
    if len(exploration) != sample["exploration_count"] or len(set(exploration)) != len(
        exploration
    ):
        raise GravityItem8FieldCurvatureError("exploration sample changed")
    if len(confirmation) != sample["reserved_confirmation_count"] or len(
        set(confirmation)
    ) != len(confirmation):
        raise GravityItem8FieldCurvatureError("confirmation sample changed")
    if set(exploration).intersection(confirmation):
        raise GravityItem8FieldCurvatureError("sample roles overlap")
    if len(exploration) + len(confirmation) != sample["eligible_count"]:
        raise GravityItem8FieldCurvatureError("eligible count changed")
    sources = config["sources"]
    forbidden = set(sources["forbidden_predictor_columns"])
    if forbidden.intersection(sources["allowed_group_predictor_columns"]):
        raise GravityItem8FieldCurvatureError("group predictor query admits a shortcut")
    if forbidden.intersection(sources["allowed_member_predictor_columns"]):
        raise GravityItem8FieldCurvatureError("member predictor query admits a shortcut")
    accounting = config["candidate_accounting"]
    if len(config["model_families"]) != accounting["preregistered_model_families"]:
        raise GravityItem8FieldCurvatureError("candidate count changed")
    return config


def _richness_stratum(members: int) -> str:
    if 8 <= members <= 11:
        return "8_11"
    if 12 <= members <= 19:
        return "12_19"
    if members >= 20:
        return "20_plus"
    raise GravityItem8FieldCurvatureError("group is below frozen richness")


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
                    "group": int(identifier),
                    "role": role,
                    "outer_fold": assignments.get(identifier),
                    "selection_digest": hashlib.sha256(
                        f"{salt}|{identifier}".encode()
                    ).hexdigest(),
                }
            )
    objects.sort(key=lambda row: (str(row["role"]), int(row["group"])))
    return _seal(
        {
            "schema_version": "invariant-gravity-item8-field-curvature-sample-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM8_TARGET_BLIND_2MPP_SAMPLE",
            "selection": {
                "quality_rule": sample["quality_rule"],
                "stratification_rule": sample["stratification_rule"],
                "salt": salt,
                "selection_used_group_dispersion": False,
                "selection_used_member_redshift": False,
            },
            "counts": {
                "catalog_groups": config["prefreeze_audit"]["group_rows"],
                "catalog_members": config["prefreeze_audit"]["member_rows"],
                "eligible": sample["eligible_count"],
                "exploration": len(exploration),
                "reserved_confirmation": len(confirmation),
            },
            "stratification_cells": sample["cell_counts"],
            "objects": objects,
            "prefreeze_boundary": {
                "published_group_dispersion_values_read": 0,
                "member_redshift_values_read": 0,
                "reserved_confirmation_dispersions_blinded": True,
                "virial_or_halo_mass_values_read": 0,
                "virial_radius_values_read": 0,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_sample_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem8FieldCurvatureError("sample content hash changed")
    roles = {"exploration": set(), "reserved_confirmation": set()}
    for row in manifest["objects"]:
        roles[str(row["role"])].add(int(row["group"]))
    if roles["exploration"] != set(config["sample"]["exploration"]):
        raise GravityItem8FieldCurvatureError("exploration identities changed")
    if roles["reserved_confirmation"] != set(config["sample"]["reserved_confirmation"]):
        raise GravityItem8FieldCurvatureError("confirmation identities changed")
    boundary = manifest["prefreeze_boundary"]
    if int(boundary["published_group_dispersion_values_read"]) != 0:
        raise GravityItem8FieldCurvatureError("sample opened group dispersions")
    if int(boundary["member_redshift_values_read"]) != 0:
        raise GravityItem8FieldCurvatureError("sample opened member redshifts")
    if not boundary["reserved_confirmation_dispersions_blinded"]:
        raise GravityItem8FieldCurvatureError("confirmation boundary changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem8FieldCurvatureError("sample contains an overclaim")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return path


def _query_url(
    catalog_id: str,
    *,
    columns: Sequence[str],
    constraint_name: str | None = None,
    constraint_value: str | None = None,
    max_rows: int = 10000,
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
    raise GravityItem8FieldCurvatureError(f"VizieR acquisition failed: {url}") from error


def _numeric_tsv_rows(payload: bytes, *, fields: int) -> list[list[str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem8FieldCurvatureError("VizieR response is not UTF-8") from exc
    rows: list[list[str]] = []
    for line in lines:
        values = [field.strip() for field in line.split("\t")]
        if values and values[0].isdigit():
            if len(values) != fields:
                raise GravityItem8FieldCurvatureError("VizieR schema changed")
            rows.append(values)
    return rows


def parse_group_predictors(payload: bytes) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in _numeric_tsv_rows(payload, fields=3):
        if not all(row):
            continue
        identifier = int(row[0])
        result[identifier] = {
            "group": identifier,
            "members": int(row[1]),
            "mean_velocity_km_s": float(row[2]),
        }
    return result


def parse_member_predictors(payload: bytes) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for row in _numeric_tsv_rows(payload, fields=5):
        if not all(row):
            continue
        identifier = int(row[0])
        result.setdefault(identifier, []).append(
            {
                "group": identifier,
                "member": int(row[1]),
                "ra_deg": float(row[2]),
                "dec_deg": float(row[3]),
                "k_magnitude": float(row[4]),
            }
        )
    for rows in result.values():
        rows.sort(key=lambda row: int(row["member"]))
    return result


def parse_group_response(payload: bytes, *, expected_group: int) -> float:
    rows = [
        row
        for row in _numeric_tsv_rows(payload, fields=2)
        if int(row[0]) == expected_group
    ]
    if len(rows) != 1 or not rows[0][1]:
        raise GravityItem8FieldCurvatureError("unexpected group response row")
    value = float(rows[0][1])
    if not math.isfinite(value) or value <= 0:
        raise GravityItem8FieldCurvatureError("invalid group dispersion")
    return value


def _retrieval(url: str, payload: bytes) -> dict[str, Any]:
    return {"url": url, "payload_sha256": _sha256_bytes(payload), "bytes": len(payload)}


def acquire_predictors(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    sources = config["sources"]
    group_url = _query_url(
        sources["group_table"], columns=sources["allowed_group_predictor_columns"]
    )
    member_url = _query_url(
        sources["member_table"], columns=sources["allowed_member_predictor_columns"]
    )
    group_payload = _fetch(group_url)
    member_payload = _fetch(member_url)
    groups = parse_group_predictors(group_payload)
    members = parse_member_predictors(member_payload)
    eligible = set(config["sample"]["exploration"]) | set(
        config["sample"]["reserved_confirmation"]
    )
    records: list[dict[str, Any]] = []
    for identifier in sorted(eligible):
        if identifier not in groups or identifier not in members:
            raise GravityItem8FieldCurvatureError("frozen predictor group is missing")
        group = groups[identifier]
        group_members = members[identifier]
        if len(group_members) != int(group["members"]):
            raise GravityItem8FieldCurvatureError("frozen member count changed")
        if int(group["members"]) < int(config["quality"]["minimum_members"]):
            raise GravityItem8FieldCurvatureError("frozen group fell below richness")
        records.append({"group": group, "members": group_members})
    return _seal(
        {
            "schema_version": "invariant-gravity-item8-field-curvature-predictor-source-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM8_TARGET_BLIND_PREDICTOR_ACQUISITION",
            "sample_binding": {
                "path": config["sample_manifest_output"],
                "file_sha256": _sha256_file(sample_path),
                "content_sha256": sample["content_sha256"],
            },
            "retrievals": {
                "group_predictors": _retrieval(group_url, group_payload),
                "member_predictors": _retrieval(member_url, member_payload),
            },
            "boundary": {
                "eligible_groups_acquired": len(records),
                "eligible_member_rows_acquired": sum(len(row["members"]) for row in records),
                "published_group_dispersion_values_acquired": 0,
                "member_redshift_values_acquired": 0,
                "virial_or_halo_mass_values_acquired": 0,
                "virial_radius_values_acquired": 0,
                "lensing_mass_values_acquired": 0,
                "paid_model_calls": 0,
            },
            "records": records,
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_predictor_source(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem8FieldCurvatureError("predictor source content hash changed")
    expected = set(config["sample"]["exploration"]) | set(
        config["sample"]["reserved_confirmation"]
    )
    if {int(row["group"]["group"]) for row in manifest["records"]} != expected:
        raise GravityItem8FieldCurvatureError("predictor source scope changed")
    boundary = manifest["boundary"]
    for field in (
        "published_group_dispersion_values_acquired",
        "member_redshift_values_acquired",
        "virial_or_halo_mass_values_acquired",
        "virial_radius_values_acquired",
        "lensing_mass_values_acquired",
        "paid_model_calls",
    ):
        if int(boundary[field]) != 0:
            raise GravityItem8FieldCurvatureError(f"predictor boundary opened: {field}")
    sources = config["sources"]
    for name, table, columns in (
        (
            "group_predictors",
            sources["group_table"],
            sources["allowed_group_predictor_columns"],
        ),
        (
            "member_predictors",
            sources["member_table"],
            sources["allowed_member_predictor_columns"],
        ),
    ):
        parsed = urllib.parse.parse_qs(
            urllib.parse.urlsplit(manifest["retrievals"][name]["url"]).query
        )
        if (
            parsed.get("-source") != [str(table)]
            or parsed.get("-out") != list(columns)
            or parsed.get("-out.max") != ["10000"]
        ):
            raise GravityItem8FieldCurvatureError("predictor retrieval scope changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem8FieldCurvatureError("predictor source contains an overclaim")


def write_predictor_source(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["predictor_source_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = acquire_predictors(root)
    validate_predictor_source(manifest, config)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def _weighted_radius(radius: np.ndarray, mass: np.ndarray, quantile: float) -> float:
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(mass[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(radius[order[min(index, len(radius) - 1)]])


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log10(1.0 + abs(value)), value)


def _alignment(vector_a: np.ndarray, vector_b: np.ndarray, floor: float) -> float:
    norm = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if norm <= floor:
        return 0.0
    cosine = float(np.dot(vector_a, vector_b) / norm)
    return 2.0 * cosine**2 - 1.0


def measure_field_features(
    *,
    ra_deg: Sequence[float],
    dec_deg: Sequence[float],
    k_magnitude: Sequence[float],
    mean_velocity_km_s: float,
    config: Mapping[str, Any],
    minimum_members: int | None = None,
) -> dict[str, float]:
    """Build target-blind field features without dispersion or member-redshift input."""

    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    kmag = np.asarray(k_magnitude, dtype=np.float64)
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(kmag)
    ra, dec, kmag = ra[valid], dec[valid], kmag[valid]
    required = int(config["quality"]["minimum_members"] if minimum_members is None else minimum_members)
    if len(ra) < required:
        raise GravityItem8FieldCurvatureError("insufficient finite members")
    unique = {(round(float(a), 8), round(float(b), 8)) for a, b in zip(ra, dec, strict=True)}
    if len(unique) < required:
        raise GravityItem8FieldCurvatureError("insufficient unique member positions")
    if not math.isfinite(mean_velocity_km_s) or mean_velocity_km_s <= 0:
        raise GravityItem8FieldCurvatureError("invalid mean group velocity")
    constants = config["constants"]
    distance_mpc = mean_velocity_km_s / float(constants["hubble_km_s_mpc"])
    distance_modulus = 5.0 * math.log10(distance_mpc) + 25.0
    luminosity = 10.0 ** (
        -0.4 * (kmag - distance_modulus - float(constants["solar_absolute_k_magnitude"]))
    )
    mass = luminosity * float(constants["fixed_k_band_mass_to_light_msun_per_lsun"])
    total_mass = float(np.sum(mass))
    if not math.isfinite(total_mass) or total_mass <= 0:
        raise GravityItem8FieldCurvatureError("invalid total K-light mass proxy")

    angle = np.deg2rad(ra)
    mean_ra = math.atan2(
        float(np.sum(mass * np.sin(angle))), float(np.sum(mass * np.cos(angle)))
    )
    delta_ra = np.angle(np.exp(1j * (angle - mean_ra)))
    mean_dec = float(np.sum(mass * np.deg2rad(dec)) / total_mass)
    distance_kpc = distance_mpc * 1000.0
    x = distance_kpc * math.cos(mean_dec) * delta_ra
    y = distance_kpc * (np.deg2rad(dec) - mean_dec)
    x -= float(np.sum(mass * x) / total_mass)
    y -= float(np.sum(mass * y) / total_mass)
    positions = np.column_stack((x, y))
    radius = np.linalg.norm(positions, axis=1)
    r_rms = math.sqrt(float(np.sum(mass * radius**2) / total_mass))
    r50 = _weighted_radius(radius, mass, 0.5)
    r90 = _weighted_radius(radius, mass, 0.9)
    if not all(math.isfinite(value) and value > 0 for value in (r_rms, r50, r90)):
        raise GravityItem8FieldCurvatureError("invalid projected radius")
    epsilon = max(
        float(constants["softening_floor_kpc"]),
        float(constants["softening_fraction_r_rms"]) * r_rms,
    )
    gravity = float(constants["gravity_kpc_km2_s2_msun"])
    floor = float(constants["dimensionless_floor"])

    covariance = (positions.T * mass) @ positions / total_mass
    shape_values, shape_vectors = np.linalg.eigh(covariance)
    shape_values = np.maximum(shape_values, 0.0)
    shape_axis_ratio = math.sqrt(float(shape_values[0] / max(shape_values[1], floor)))
    shape_major = shape_vectors[:, 1]

    squared = radius**2 + epsilon**2
    inv3 = squared ** (-1.5)
    inv5 = squared ** (-2.5)
    inv7 = squared ** (-3.5)
    acceleration_vector = gravity * np.sum(mass[:, None] * positions * inv3[:, None], axis=0)
    hessian = np.zeros((2, 2), dtype=np.float64)
    third = np.zeros((2, 2, 2), dtype=np.float64)
    identity = np.eye(2)
    for member_mass, position, member_inv3, member_inv5, member_inv7 in zip(
        mass, positions, inv3, inv5, inv7, strict=True
    ):
        outer = np.outer(position, position)
        hessian += gravity * member_mass * (3.0 * outer * member_inv5 - identity * member_inv3)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    delta_terms = (
                        (1.0 if i == j else 0.0) * position[k]
                        + (1.0 if i == k else 0.0) * position[j]
                        + (1.0 if j == k else 0.0) * position[i]
                    )
                    third[i, j, k] += gravity * member_mass * (
                        3.0 * delta_terms * member_inv5
                        - 15.0 * position[i] * position[j] * position[k] * member_inv7
                    )
    acceleration_scale = gravity * total_mass / r_rms**2
    tidal_scale = gravity * total_mass / r_rms**3
    third_scale = gravity * total_mass / r_rms**4
    acceleration_norm = float(np.linalg.norm(acceleration_vector) / acceleration_scale)
    tidal = hessian / tidal_scale
    tidal_values, tidal_vectors = np.linalg.eigh(tidal)
    tidal_trace = float(np.trace(tidal))
    tidal_frobenius = float(np.linalg.norm(tidal))
    tidal_determinant = float(np.linalg.det(tidal))
    tidal_anisotropy = float(
        abs(tidal_values[1] - tidal_values[0])
        / (abs(tidal_values[1]) + abs(tidal_values[0]) + floor)
    )
    third_norm = float(np.linalg.norm(third) / third_scale)
    acceleration_alignment = _alignment(acceleration_vector, shape_major, floor)
    tidal_major = tidal_vectors[:, int(np.argmax(np.abs(tidal_values)))]
    tidal_alignment = _alignment(tidal_major, shape_major, floor)

    field_values: list[float] = []
    directions = int(constants["radial_field_directions"])
    for scale in constants["radial_field_scales_r_rms"]:
        magnitudes: list[float] = []
        for index in range(directions):
            theta = 2.0 * math.pi * index / directions
            point = float(scale) * r_rms * np.asarray([math.cos(theta), math.sin(theta)])
            delta = positions - point
            softened = np.sum(delta**2, axis=1) + epsilon**2
            vector = gravity * np.sum(
                mass[:, None] * delta * softened[:, None] ** (-1.5), axis=0
            )
            magnitudes.append(float(np.linalg.norm(vector)))
        field_values.append(float(np.mean(magnitudes)))
    if any(not math.isfinite(value) or value <= 0 for value in field_values):
        raise GravityItem8FieldCurvatureError("invalid radial field profile")
    log_scale = np.log(np.asarray(constants["radial_field_scales_r_rms"], dtype=np.float64))
    log_field = np.log(np.asarray(field_values, dtype=np.float64))
    slopes = np.diff(log_field) / np.diff(log_scale)
    slope_inner = float(slopes[0])
    slope_outer = float(slopes[-1])
    slope_transition = slope_outer - slope_inner
    curvature = slope_transition / float(log_scale[-1] - log_scale[0])

    virial_velocity = math.sqrt(gravity * total_mass / r_rms)
    values = {
        "log10_mass": math.log10(total_mass),
        "log10_r_rms": math.log10(r_rms),
        "log10_r50": math.log10(r50),
        "log10_r90": math.log10(r90),
        "log10_richness": math.log10(len(mass)),
        "log10_distance": math.log10(distance_mpc),
        "log10_virial_velocity": math.log10(virial_velocity),
        "log10_acceleration_amplitude": math.log10(acceleration_scale),
        "log10_tidal_amplitude": math.log10(tidal_scale),
        "shape_axis_ratio": shape_axis_ratio,
        "radial_concentration": r50 / r90,
        "signed_log_center_acceleration_norm": _signed_log1p(acceleration_norm),
        "signed_log_tidal_trace_norm": _signed_log1p(tidal_trace),
        "log10_tidal_frobenius_norm": math.log10(tidal_frobenius + floor),
        "signed_log_tidal_determinant_norm": _signed_log1p(tidal_determinant),
        "tidal_eigenvalue_anisotropy": tidal_anisotropy,
        "log10_third_derivative_norm": math.log10(third_norm + floor),
        "acceleration_shape_alignment": acceleration_alignment,
        "tidal_shape_alignment": tidal_alignment,
        "third_x_tidal_anisotropy": math.log10(third_norm + floor) * tidal_anisotropy,
        "radial_field_slope_inner": slope_inner,
        "radial_field_slope_outer": slope_outer,
        "radial_field_curvature": curvature,
        "radial_slope_transition": slope_transition,
        "r_rms_kpc": r_rms,
        "epsilon_kpc": epsilon,
        "total_mass_msun": total_mass,
    }
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise GravityItem8FieldCurvatureError("non-finite field feature")
    return values


FEATURE_NAMES = (
    "log10_mass",
    "log10_r_rms",
    "log10_r50",
    "log10_r90",
    "log10_richness",
    "log10_distance",
    "log10_virial_velocity",
    "log10_acceleration_amplitude",
    "log10_tidal_amplitude",
    "shape_axis_ratio",
    "radial_concentration",
    "signed_log_center_acceleration_norm",
    "signed_log_tidal_trace_norm",
    "log10_tidal_frobenius_norm",
    "signed_log_tidal_determinant_norm",
    "tidal_eigenvalue_anisotropy",
    "log10_third_derivative_norm",
    "acceleration_shape_alignment",
    "tidal_shape_alignment",
    "third_x_tidal_anisotropy",
    "radial_field_slope_inner",
    "radial_field_slope_outer",
    "radial_field_curvature",
    "radial_slope_transition",
    "r_rms_kpc",
    "epsilon_kpc",
    "total_mass_msun",
)


def _write_feature_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "group",
        "role",
        "outer_fold",
        "richness_stratum",
        "mass_stratum",
        "concentration_stratum",
        "members",
        *FEATURE_NAMES,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row[field]
                    if field in {
                        "group",
                        "role",
                        "outer_fold",
                        "richness_stratum",
                        "mass_stratum",
                        "concentration_stratum",
                        "members",
                    }
                    else _metric(float(row[field]))
                    for field in fields
                }
            )


def extract_predictors(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    config = load_config(root)
    sample = json.loads((root / config["sample_manifest_output"]).read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    source_path = root / config["predictor_source_output"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_predictor_source(source, config)
    roles = {int(row["group"]): row for row in sample["objects"]}
    primary: list[dict[str, Any]] = []
    brightest_removed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in source["records"]:
        group = record["group"]
        identifier = int(group["group"])
        members = record["members"]
        common = {
            "ra_deg": [float(row["ra_deg"]) for row in members],
            "dec_deg": [float(row["dec_deg"]) for row in members],
            "k_magnitude": [float(row["k_magnitude"]) for row in members],
            "mean_velocity_km_s": float(group["mean_velocity_km_s"]),
            "config": config,
        }
        try:
            features = measure_field_features(**common)
            brightest = int(np.argmin(np.asarray(common["k_magnitude"], dtype=np.float64)))
            keep = [index for index in range(len(members)) if index != brightest]
            alternative = measure_field_features(
                ra_deg=[common["ra_deg"][index] for index in keep],
                dec_deg=[common["dec_deg"][index] for index in keep],
                k_magnitude=[common["k_magnitude"][index] for index in keep],
                mean_velocity_km_s=common["mean_velocity_km_s"],
                config=config,
                minimum_members=int(config["quality"]["minimum_members"]) - 1,
            )
        except GravityItem8FieldCurvatureError as exc:
            failures.append({"group": identifier, "reason": str(exc)})
            continue
        object_row = roles[identifier]
        base = {
            "group": identifier,
            "role": str(object_row["role"]),
            "outer_fold": "" if object_row["outer_fold"] is None else int(object_row["outer_fold"]),
            "richness_stratum": _richness_stratum(int(group["members"])),
            "members": int(group["members"]),
        }
        primary.append({**base, **features})
        brightest_removed.append({**base, "members": int(group["members"]) - 1, **alternative})
    primary.sort(key=lambda row: int(row["group"]))
    brightest_removed.sort(key=lambda row: int(row["group"]))
    exploration = [row for row in primary if row["role"] == "exploration"]
    mass_cut = float(np.median([float(row["log10_mass"]) for row in exploration]))
    concentration_cut = float(
        np.median([float(row["radial_concentration"]) for row in exploration])
    )
    for rows in (primary, brightest_removed):
        for row in rows:
            row["mass_stratum"] = (
                "low_mass" if float(row["log10_mass"]) <= mass_cut else "high_mass"
            )
            row["concentration_stratum"] = (
                "low_concentration"
                if float(row["radial_concentration"]) <= concentration_cut
                else "high_concentration"
            )
    feature_path = root / config["feature_output"]
    alternative_path = root / config["brightest_removed_feature_output"]
    _write_feature_table(feature_path, primary)
    _write_feature_table(alternative_path, brightest_removed)
    quality_pass = not failures and len(primary) == config["sample"]["eligible_count"]
    summary = _seal(
        {
            "schema_version": "invariant-gravity-item8-field-curvature-predictor-extraction-1.0",
            "goal": config["goal"],
            "decision": (
                "PASS_ITEM8_TARGET_BLIND_FIELD_REPRESENTATION"
                if quality_pass
                else "FAIL_ITEM8_TARGET_BLIND_FIELD_REPRESENTATION"
            ),
            "counts": {
                "eligible_groups": config["sample"]["eligible_count"],
                "feature_rows": len(primary),
                "brightest_removed_feature_rows": len(brightest_removed),
                "failures": len(failures),
                "published_group_dispersion_values_read": 0,
                "member_redshift_values_read": 0,
            },
            "failures": failures,
            "strata_thresholds": {
                "exploration_log10_mass_median": _metric(mass_cut),
                "exploration_radial_concentration_median": _metric(concentration_cut),
            },
            "inputs": {
                "predictor_source_content_sha256": source["content_sha256"],
            },
            "outputs": {
                "feature_sha256": _sha256_file(feature_path),
                "brightest_removed_feature_sha256": _sha256_file(alternative_path),
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )
    summary_path = root / config["predictor_extraction_output"]
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return feature_path, alternative_path, summary_path


def acquire_responses(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem8FieldCurvatureError(
            "scientific freeze commit is not bound; group-dispersion access forbidden"
        )
    sample_path = root / config["sample_manifest_output"]
    predictor_path = root / config["predictor_source_output"]
    feature_path = root / config["feature_output"]
    extraction_path = root / config["predictor_extraction_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
    validate_predictor_source(predictor, config)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if extraction["decision"] != "PASS_ITEM8_TARGET_BLIND_FIELD_REPRESENTATION":
        raise GravityItem8FieldCurvatureError("predictor representation did not pass")
    records: list[dict[str, Any]] = []
    source = config["sources"]
    for identifier in sorted(int(value) for value in config["sample"]["exploration"]):
        url = _query_url(
            source["group_table"],
            columns=source["response_columns"],
            constraint_name="ID",
            constraint_value=str(identifier),
            max_rows=10,
        )
        payload = _fetch(url)
        records.append(
            {
                "group": identifier,
                "sigma_km_s": parse_group_response(payload, expected_group=identifier),
                "retrieval": _retrieval(url, payload),
            }
        )
    return _seal(
        {
            "schema_version": "invariant-gravity-item8-field-curvature-response-source-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM8_2MPP_EXPLORATION_RESPONSE_ACQUISITION",
            "preregistration": {
                "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
                "config_path": CONFIG_PATH,
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "sample_manifest_sha256": _sha256_file(sample_path),
                "predictor_source_sha256": _sha256_file(predictor_path),
                "feature_sha256": _sha256_file(feature_path),
                "predictor_extraction_sha256": _sha256_file(extraction_path),
            },
            "boundary": {
                "exploration_dispersion_queries": len(records),
                "exploration_dispersion_rows": len(records),
                "reserved_confirmation_dispersion_queries": 0,
                "member_redshift_values_acquired": 0,
                "virial_or_halo_mass_values_acquired": 0,
                "virial_radius_values_acquired": 0,
                "lensing_mass_values_acquired": 0,
                "paid_model_calls": 0,
            },
            "records": records,
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_response_source(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem8FieldCurvatureError("response source content hash changed")
    if manifest["preregistration"]["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem8FieldCurvatureError("response freeze binding changed")
    if {int(row["group"]) for row in manifest["records"]} != set(
        config["sample"]["exploration"]
    ):
        raise GravityItem8FieldCurvatureError("response source scope changed")
    boundary = manifest["boundary"]
    for field in (
        "reserved_confirmation_dispersion_queries",
        "member_redshift_values_acquired",
        "virial_or_halo_mass_values_acquired",
        "virial_radius_values_acquired",
        "lensing_mass_values_acquired",
        "paid_model_calls",
    ):
        if int(boundary[field]) != 0:
            raise GravityItem8FieldCurvatureError(f"response boundary opened: {field}")
    source = config["sources"]
    for record in manifest["records"]:
        identifier = int(record["group"])
        parsed = urllib.parse.parse_qs(
            urllib.parse.urlsplit(record["retrieval"]["url"]).query
        )
        if (
            parsed.get("-source") != [str(source["group_table"])]
            or parsed.get("-out") != list(source["response_columns"])
            or parsed.get("ID") != [str(identifier)]
            or parsed.get("-out.max") != ["10"]
        ):
            raise GravityItem8FieldCurvatureError(
                "group response retrieval was not a one-group frozen query"
            )
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem8FieldCurvatureError("response source contains an overclaim")


def write_response_source(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["response_source_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = acquire_responses(root)
    validate_response_source(manifest, config)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def _load_feature_file(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            identifier = int(raw["group"])
            row: dict[str, Any] = {
                "cluster": str(identifier),
                "group": identifier,
                "role": raw["role"],
                "richness_stratum": raw["richness_stratum"],
                "mass_stratum": raw["mass_stratum"],
                "concentration_stratum": raw["concentration_stratum"],
                "members": int(raw["members"]),
            }
            for field in FEATURE_NAMES:
                row[field] = float(raw[field])
            result[identifier] = row
    return result


def load_exploration_rows(
    root: Path, config: Mapping[str, Any], *, brightest_removed: bool = False
) -> list[dict[str, Any]]:
    feature_name = (
        "brightest_removed_feature_output" if brightest_removed else "feature_output"
    )
    features = _load_feature_file(root / config[feature_name])
    response = json.loads((root / config["response_source_output"]).read_text(encoding="utf-8"))
    validate_response_source(response, config)
    rows: list[dict[str, Any]] = []
    for record in response["records"]:
        identifier = int(record["group"])
        row = dict(features[identifier])
        row["response_log10_sigma"] = math.log10(float(record["sigma_km_s"]))
        rows.append(row)
    rows.sort(key=lambda row: int(row["group"]))
    return rows


def _array(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows])


def _predictions_array(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, float]
) -> np.ndarray:
    return np.asarray([float(predictions[str(row["group"])]) for row in rows])


def _metrics(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, float]
) -> dict[str, str]:
    observed = _array(rows, "response_log10_sigma")
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
    original = _array(rows, "response_log10_sigma")
    labels = sorted({str(row["richness_stratum"]) for row in rows})
    indices_by_stratum = {
        label: [index for index, row in enumerate(rows) if row["richness_stratum"] == label]
        for label in labels
    }
    null: list[float] = []
    for _ in range(count):
        shuffled = original.copy()
        for indices in indices_by_stratum.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        permuted = [
            dict(row, response_log10_sigma=float(shuffled[index]))
            for index, row in enumerate(rows)
        ]
        baseline, _ = cvcore._nested_predictions(
            permuted, models=baseline_models, config=config, detailed=False
        )
        qualifying, _ = cvcore._nested_predictions(
            permuted, models=qualifying_models, config=config, detailed=False
        )
        observed = _array(permuted, "response_log10_sigma")
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


def _selector_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_models: Sequence[Mapping[str, Any]],
    qualifying_models: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline, _ = cvcore._nested_predictions(
        rows, models=baseline_models, config=config, detailed=False
    )
    qualifying, _ = cvcore._nested_predictions(
        rows, models=qualifying_models, config=config, detailed=False
    )
    observed = _array(rows, "response_log10_sigma")
    baseline_mse = cvcore._mse(observed, _predictions_array(rows, baseline))
    qualifying_mse = cvcore._mse(observed, _predictions_array(rows, qualifying))
    return {
        "baseline_mse": _metric(baseline_mse),
        "qualifying_mse": _metric(qualifying_mse),
        "qualifying_mse_gain": _metric(baseline_mse - qualifying_mse),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample_path = root / config["sample_manifest_output"]
    predictor_path = root / config["predictor_source_output"]
    feature_path = root / config["feature_output"]
    alternative_path = root / config["brightest_removed_feature_output"]
    extraction_path = root / config["predictor_extraction_output"]
    response_path = root / config["response_source_output"]
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    validate_response_source(response, config)
    rows = load_exploration_rows(root, config)
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
    observed = _array(rows, "response_log10_sigma")
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
        metrics = _metrics(rows, predictions)
        gain = float(metrics["mse"]) - qualifying_mse
        individual_baselines[str(model["id"])] = {
            **metrics,
            "qualifying_mse_gain": _metric(gain),
        }
        beats_each = beats_each and gain > 0

    strata: dict[str, Any] = {}
    improvements: dict[str, list[float]] = {
        "richness_stratum": [],
        "mass_stratum": [],
        "concentration_stratum": [],
    }
    for field, labels in (
        ("richness_stratum", ("8_11", "12_19", "20_plus")),
        ("mass_stratum", ("low_mass", "high_mass")),
        (
            "concentration_stratum",
            ("low_concentration", "high_concentration"),
        ),
    ):
        for label in labels:
            subset = [row for row in rows if row[field] == label]
            baseline_value = cvcore._mse(
                _array(subset, "response_log10_sigma"),
                _predictions_array(subset, baseline_predictions),
            )
            qualifying_value = cvcore._mse(
                _array(subset, "response_log10_sigma"),
                _predictions_array(subset, qualifying_predictions),
            )
            gain = baseline_value - qualifying_value
            improvements[field].append(gain)
            strata[f"{field}:{label}"] = {
                "count": len(subset),
                "baseline_mse": _metric(baseline_value),
                "qualifying_mse": _metric(qualifying_value),
                "qualifying_mse_gain": _metric(gain),
            }

    alternative_rows = load_exploration_rows(root, config, brightest_removed=True)
    brightest_removed = _selector_comparison(
        alternative_rows,
        baseline_models=baseline_models,
        qualifying_models=qualifying_models,
        config=config,
    )
    permutation = _permutation_test(
        rows,
        baseline_models=baseline_models,
        qualifying_models=qualifying_models,
        observed_improvement=improvement,
        config=config,
    )
    admission = config["exploration_admission"]
    quality_pass = (
        extraction["decision"] == "PASS_ITEM8_TARGET_BLIND_FIELD_REPRESENTATION"
        and len(rows) == config["sample"]["exploration_count"]
    )
    gates = {
        "all_98_exploration_groups_pass_frozen_quality": quality_pass,
        "unrestricted_selector_qualifying_in_at_least_4_of_5_folds": sum(
            bool(fold["selected_qualifying"]) for fold in unrestricted_folds
        )
        >= int(admission["unrestricted_selector_qualifying_in_at_least_folds"]),
        "qualifying_selector_r2_positive_overall": cvcore._r2(observed, qualifying_array) > 0,
        "qualifying_selector_beats_each_nonqualifying_baseline_overall": beats_each,
        "qualifying_relative_mse_improvement_over_strongest_baseline_at_least_0_02": relative_improvement
        >= float(admission["qualifying_relative_mse_improvement_over_strongest_baseline_at_least"]),
        "qualifying_improvement_positive_in_all_richness_strata": all(
            value > 0 for value in improvements["richness_stratum"]
        ),
        "qualifying_improvement_positive_in_both_mass_strata": all(
            value > 0 for value in improvements["mass_stratum"]
        ),
        "qualifying_improvement_positive_in_both_concentration_strata": all(
            value > 0 for value in improvements["concentration_stratum"]
        ),
        "richness_stratified_permutation_p_at_most_0_05": float(permutation["p_value"])
        <= float(admission["richness_stratified_permutation_p_at_most"]),
        "brightest_member_removal_does_not_reverse_improvement": float(
            brightest_removed["qualifying_mse_gain"]
        )
        >= 0,
        "reserved_confirmation_targets_untouched": response["boundary"][
            "reserved_confirmation_dispersion_queries"
        ]
        == 0,
    }
    if not quality_pass:
        decision = "INCONCLUSIVE_ITEM8_FIELD_CURVATURE_QUALITY_GATE"
    elif all(gates.values()):
        decision = "PASS_ITEM8_FIELD_CURVATURE_EXPLORATION_REQUIRES_CONFIRMATION_AUTHORIZATION"
    else:
        decision = "REJECT_ITEM8_FIELD_CURVATURE_EXPLORATION"
    return _seal(
        {
            "schema_version": "invariant-gravity-item8-field-curvature-result-1.0",
            "goal": config["goal"],
            "item_number": 8,
            "decision": decision,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "creativity_label": config["scientific_contract"]["creativity_label"],
            "inputs": {
                "config_path": CONFIG_PATH,
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "sample_manifest_sha256": _sha256_file(sample_path),
                "predictor_source_sha256": _sha256_file(predictor_path),
                "feature_sha256": _sha256_file(feature_path),
                "brightest_removed_feature_sha256": _sha256_file(alternative_path),
                "predictor_extraction_sha256": _sha256_file(extraction_path),
                "response_source_sha256": _sha256_file(response_path),
                "response_source_content_sha256": response["content_sha256"],
            },
            "counts": {
                "exploration_groups": len(rows),
                "reserved_confirmation_groups": config["sample"]["reserved_confirmation_count"],
                "reserved_confirmation_target_accesses": 0,
                "member_redshift_values_acquired": 0,
                "paid_model_calls": 0,
                "permutation_nested_cv_runs": config["cross_validation"]["permutation_count"],
                **config["candidate_accounting"],
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
                    "absolute_mse_improvement_over_strongest_baseline": _metric(improvement),
                    "relative_mse_improvement_over_strongest_baseline": _metric(
                        relative_improvement
                    ),
                },
                "individual_nonqualifying_baselines": individual_baselines,
            },
            "strata": strata,
            "brightest_member_removal": brightest_removed,
            "permutation": permutation,
            "gate_checks": gates,
            "gate_counts": {"passed": sum(gates.values()), "required": len(gates)},
            "limitations": {
                "projected_k_light_is_complete_baryonic_mass": False,
                "redshift_space_membership_is_target_independent": False,
                "three_dimensional_tidal_tensor_used": False,
                "published_sigma_has_reported_uncertainty": False,
                "confirmation_opened": False,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(receipt)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem8FieldCurvatureError("result content hash changed")
    load_config(root)
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem8FieldCurvatureError("result freeze changed")
    if int(receipt["counts"]["reserved_confirmation_target_accesses"]) != 0:
        raise GravityItem8FieldCurvatureError("confirmation was opened")
    if int(receipt["counts"]["member_redshift_values_acquired"]) != 0:
        raise GravityItem8FieldCurvatureError("member redshifts were opened")
    passed = sum(bool(value) for value in receipt["gate_checks"].values())
    if receipt["gate_counts"] != {"passed": passed, "required": len(receipt["gate_checks"])}:
        raise GravityItem8FieldCurvatureError("gate count changed")
    passing = receipt["decision"].startswith("PASS_ITEM8_")
    if passing != all(bool(value) for value in receipt["gate_checks"].values()):
        raise GravityItem8FieldCurvatureError("decision does not match gates")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem8FieldCurvatureError("result contains an overclaim")
    if receipt != build_receipt(root):
        raise GravityItem8FieldCurvatureError("result does not replay")


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
    parser.add_argument(
        "command",
        choices=(
            "sample",
            "acquire-predictors",
            "extract-predictors",
            "acquire-responses",
            "run",
            "check",
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "sample":
        print(write_sample_manifest(args.root))
    elif args.command == "acquire-predictors":
        print(write_predictor_source(args.root))
    elif args.command == "extract-predictors":
        print(*extract_predictors(args.root), sep="\n")
    elif args.command == "acquire-responses":
        print(write_response_source(args.root))
    elif args.command == "run":
        print(write_receipt(args.root))
    else:
        check_receipt(args.root)


if __name__ == "__main__":
    main()
