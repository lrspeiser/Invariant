"""Frozen source, sample, and compactness derivation for gravity roadmap Item 4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item4_baryonic_compactness_groups_v1.json"
EXTRACTION_SUMMARY_PATH = (
    "runs/gravity/roadmap/item-04-baryonic-compactness-v1-source/extraction-summary.json"
)
FREEZE_COMMIT = "125b639f834d38e0561973e6e81f7b27290d9c6a"


class GravityItem4CompactnessError(RuntimeError):
    """Raised when the frozen Item 4 boundary or derivation drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem4CompactnessError("non-finite metric")
    return f"{float(value):.12e}"


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item4-baryonic-compactness-config-1.0"
    ):
        raise GravityItem4CompactnessError("unexpected Item 4 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem4CompactnessError("stable roadmap changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem4CompactnessError("Item 3 synthesis file changed")
    predecessor_receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor_receipt.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem4CompactnessError("Item 3 synthesis content changed")
    if predecessor_receipt.get("decision") != predecessor["required_decision"]:
        raise GravityItem4CompactnessError("Item 3 did not authorize Item 4")
    source = config["source"]
    for prefix in ("item2", "item3"):
        path = root / source[f"{prefix}_exclusion_manifest"]
        if _sha256_file(path) != source[f"{prefix}_exclusion_file_sha256"]:
            raise GravityItem4CompactnessError(f"{prefix} exclusion file changed")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("content_sha256") != source[f"{prefix}_exclusion_content_sha256"]:
            raise GravityItem4CompactnessError(f"{prefix} exclusion content changed")
    if _sha256_file(root / source["metadata_path"]) != source["metadata_file_sha256"]:
        raise GravityItem4CompactnessError("group metadata changed")
    if config["authorization"]["paid_model_calls_allowed"]:
        raise GravityItem4CompactnessError("paid model calls are forbidden")
    if config["authorization"]["reserved_confirmation_member_rows_allowed"]:
        raise GravityItem4CompactnessError("confirmation member access is forbidden")
    return config


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = "Group\tNmemb\tzsp\tLR195\tD10"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem4CompactnessError("metadata header changed") from exc
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(lines[start:], delimiter="\t"):
        try:
            rows.append(
                {
                    "group": int(row["Group"]),
                    "members": int(row["Nmemb"]),
                    "redshift": float(row["zsp"]),
                    "lr195": float(row["LR195"]),
                    "d10": float(row["D10"]),
                }
            )
        except (TypeError, ValueError):
            continue
    return rows


def _prior_groups(root: Path, config: Mapping[str, Any]) -> set[int]:
    groups: set[int] = set()
    for prefix in ("item2", "item3"):
        manifest = json.loads(
            (root / config["source"][f"{prefix}_exclusion_manifest"]).read_text(encoding="utf-8")
        )
        groups.update(int(row["group"]) for row in manifest["objects"])
    return groups


def _stratum(members: int, config: Mapping[str, Any]) -> str | None:
    for stratum in config["sample"]["strata"]:
        if int(stratum["minimum"]) <= members <= int(stratum["maximum"]):
            return str(stratum["id"])
    return None


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample_config = config["sample"]
    lower_z, upper_z = (float(value) for value in sample_config["redshift_range"])
    metadata = _read_metadata(root / config["source"]["metadata_path"])
    eligible_before = [
        row
        for row in metadata
        if lower_z <= row["redshift"] <= upper_z
        and row["members"] >= int(sample_config["minimum_members"])
        and row["lr195"] > 0
        and _stratum(int(row["members"]), config) is not None
    ]
    prior = _prior_groups(root, config)
    remaining = [row for row in eligible_before if int(row["group"]) not in prior]
    by_stratum: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    salt = str(sample_config["selection_salt"])
    for row in remaining:
        by_stratum[str(_stratum(int(row["members"]), config))].append(row)
    objects: list[dict[str, Any]] = []
    for stratum in sample_config["strata"]:
        stratum_id = str(stratum["id"])
        ordered = sorted(
            by_stratum[stratum_id],
            key=lambda row: hashlib.sha256(
                f"{salt}|{stratum_id}|{row['group']}".encode()
            ).hexdigest(),
        )
        exploration = int(stratum["exploration"])
        confirmation = int(stratum["reserved_confirmation"])
        if len(ordered) != exploration + confirmation:
            raise GravityItem4CompactnessError(f"unexpected remaining {stratum_id} count")
        for index, row in enumerate(ordered):
            role = "exploration" if index < exploration else "reserved_confirmation"
            objects.append(
                {
                    "group": int(row["group"]),
                    "members": int(row["members"]),
                    "redshift": f"{float(row['redshift']):.12e}",
                    "lr195": f"{float(row['lr195']):.12e}",
                    "d10": f"{float(row['d10']):.12e}",
                    "richness_stratum": stratum_id,
                    "role": role,
                    "selection_digest": hashlib.sha256(
                        f"{salt}|{stratum_id}|{row['group']}".encode()
                    ).hexdigest(),
                }
            )
    objects.sort(
        key=lambda row: (str(row["richness_stratum"]), str(row["role"]), int(row["group"]))
    )
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item4-compactness-sample-1.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM4_TARGET_BLIND_FRESH_GROUP_SAMPLE",
        "counts": {
            "eligible_before_prior_exclusion": len(eligible_before),
            "prior_group_ids_excluded": len(prior),
            "remaining": len(remaining),
            "exploration": sum(row["role"] == "exploration" for row in objects),
            "reserved_confirmation": sum(row["role"] == "reserved_confirmation" for row in objects),
        },
        "selection_boundary": {
            "member_rows_opened": 0,
            "member_redshifts_read": 0,
            "published_group_velocity_columns_read": 0,
            "confirmation_target_accesses": 0,
        },
        "objects": objects,
        "claims": {
            "alternative_to_gr_established": False,
            "confirmation_opened": False,
            "prior_groups_reused": False,
            "selection_used_response": False,
            "roadmap_item_4_complete": False,
        },
        "content_sha256": None,
    }
    content = dict(manifest)
    content.pop("content_sha256")
    manifest["content_sha256"] = canonical_sha256(content)
    validate_sample_manifest(manifest, config)
    return manifest


def validate_sample_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    content = dict(manifest)
    claimed = content.pop("content_sha256", None)
    if claimed != canonical_sha256(content):
        raise GravityItem4CompactnessError("sample content hash changed")
    expected = config["sample"]
    if manifest["counts"] != {
        "eligible_before_prior_exclusion": expected["expected_eligible_before_prior_exclusion"],
        "prior_group_ids_excluded": expected["expected_prior_group_ids_excluded"],
        "remaining": expected["expected_remaining"],
        "exploration": expected["expected_exploration"],
        "reserved_confirmation": expected["expected_reserved_confirmation"],
    }:
        raise GravityItem4CompactnessError("sample counts changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem4CompactnessError("sample contains overclaim")
    if any(int(value) != 0 for value in manifest["selection_boundary"].values()):
        raise GravityItem4CompactnessError("sample selection leaked member data")
    roles = Counter(str(row["role"]) for row in manifest["objects"])
    if roles != {
        "exploration": expected["expected_exploration"],
        "reserved_confirmation": expected["expected_reserved_confirmation"],
    }:
        raise GravityItem4CompactnessError("sample roles changed")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return path


def _load_sample(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    sample = json.loads((root / config["sample_manifest_output"]).read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    return sample


def _download_member_query(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise GravityItem4CompactnessError(f"member query failed: {url}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def parse_member_payload(payload: bytes, *, expected_group: int) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem4CompactnessError("member response is not UTF-8") from exc
    header = "Group\tGalID\tSpecObjID\tRAJ2000\tDEJ2000\tzsp\tLr"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem4CompactnessError(
            f"member response header changed for group {expected_group}"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line in lines[start + 1 :]:
        fields = line.split("\t")
        if len(fields) != 7 or not fields[0].strip().isdigit():
            continue
        try:
            row = {
                "group": int(fields[0]),
                "galaxy_id": int(fields[1]),
                "specobjid": int(fields[2]),
                "ra_deg": float(fields[3]),
                "dec_deg": float(fields[4]),
                "member_redshift": float(fields[5]),
                "luminosity": float(fields[6]),
            }
        except ValueError:
            continue
        if row["group"] != expected_group:
            raise GravityItem4CompactnessError("query returned another group")
        rows.append(row)
    if not rows or len({row["galaxy_id"] for row in rows}) != len(rows):
        raise GravityItem4CompactnessError(f"invalid member rows: {expected_group}")
    return rows


def _acquire_one(
    row: Mapping[str, Any], *, config: Mapping[str, Any], cache_dir: Path
) -> dict[str, Any]:
    group = int(row["group"])
    url = str(config["source"]["member_query_template"]).format(group=group)
    path = cache_dir / f"members-{group}.tsv"
    payload = _download_member_query(url, path)
    members = parse_member_payload(payload, expected_group=group)
    if len(members) != int(row["members"]):
        raise GravityItem4CompactnessError(f"member count differs from frozen metadata: {group}")
    return {
        "bytes": len(payload),
        "group": group,
        "member_rows": len(members),
        "sha256": _sha256_bytes(payload),
        "url": url,
    }


def acquire_exploration(root: Path, *, cache_dir: Path, workers: int = 8) -> Path:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    sample = _load_sample(root, config)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    confirmation = {
        int(row["group"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if any(int(row["group"]) in confirmation for row in exploration):
        raise GravityItem4CompactnessError("sample roles overlap")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(_acquire_one, row, config=config, cache_dir=cache_dir): int(row["group"])
            for row in exploration
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - retain every frozen source failure
                errors.append(f"{group}: {exc}")
    if errors:
        raise GravityItem4CompactnessError("; ".join(sorted(errors)))
    records.sort(key=lambda row: int(row["group"]))
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item4-compactness-source-1.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM4_EXPLORATION_SOURCE_ACQUISITION",
        "preregistration": {
            "git_commit": FREEZE_COMMIT,
            "member_rows_opened_before_commit": 0,
        },
        "sample_binding": {
            "path": config["sample_manifest_output"],
            "file_sha256": _sha256_file(root / config["sample_manifest_output"]),
            "content_sha256": sample["content_sha256"],
        },
        "boundary": {
            "exploration_groups_acquired": len(records),
            "exploration_target_accesses": len(records),
            "prior_group_target_reuse": 0,
            "published_group_velocity_columns_read": 0,
            "reserved_confirmation_groups_acquired": 0,
            "reserved_confirmation_target_accesses": 0,
        },
        "counts": {
            "bytes": sum(int(row["bytes"]) for row in records),
            "groups": len(records),
            "member_rows": sum(int(row["member_rows"]) for row in records),
        },
        "records": records,
        "claims": {
            "alternative_to_gr_established": False,
            "confirmation_opened": False,
            "roadmap_item_4_complete": False,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_source_manifest(manifest, sample=sample)
    path = root / config["source_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def validate_source_manifest(manifest: Mapping[str, Any], *, sample: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem4CompactnessError("source manifest content hash changed")
    if manifest.get("decision") != "PASS_ITEM4_EXPLORATION_SOURCE_ACQUISITION":
        raise GravityItem4CompactnessError("source acquisition did not pass")
    boundary = manifest["boundary"]
    if any(
        int(boundary[key]) != 0
        for key in (
            "prior_group_target_reuse",
            "published_group_velocity_columns_read",
            "reserved_confirmation_groups_acquired",
            "reserved_confirmation_target_accesses",
        )
    ):
        raise GravityItem4CompactnessError("source boundary changed")
    expected = {int(row["group"]) for row in sample["objects"] if row["role"] == "exploration"}
    if {int(row["group"]) for row in manifest["records"]} != expected:
        raise GravityItem4CompactnessError("source IDs differ from frozen exploration")
    if int(manifest["counts"]["groups"]) != len(expected):
        raise GravityItem4CompactnessError("source group count changed")
    if manifest["preregistration"]["git_commit"] != FREEZE_COMMIT:
        raise GravityItem4CompactnessError("preregistration commit changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem4CompactnessError("source manifest contains overclaim")


def _weighted_quantile_radius(radius: np.ndarray, weight: np.ndarray, quantile: float) -> float:
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(weight[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(radius[order[min(index, len(radius) - 1)]])


def measure_compactness_only(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    luminosity: np.ndarray,
    metadata_redshift: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Measure compactness without accepting member redshifts or dynamics."""

    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    light = np.asarray(luminosity, dtype=np.float64)
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(light) & (light > 0)
    ra, dec, light = ra[valid], dec[valid], light[valid]
    if len(ra) < int(config["sample"]["minimum_members"]):
        raise GravityItem4CompactnessError("insufficient finite compactness members")
    angle = np.deg2rad(ra)
    mean_ra = math.atan2(float(np.sum(light * np.sin(angle))), float(np.sum(light * np.cos(angle))))
    delta_ra = np.angle(np.exp(1j * (angle - mean_ra)))
    mean_dec = float(np.sum(light * np.deg2rad(dec)) / np.sum(light))
    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)
    distance_kpc = float(cosmology.angular_diameter_distance(metadata_redshift).value) * 1000.0
    x = distance_kpc * math.cos(mean_dec) * delta_ra
    y = distance_kpc * (np.deg2rad(dec) - mean_dec)
    x -= float(np.sum(light * x) / np.sum(light))
    y -= float(np.sum(light * y) / np.sum(light))
    radius = np.hypot(x, y)
    if len({(round(float(a), 8), round(float(b), 8)) for a, b in zip(x, y, strict=True)}) < int(
        config["quality"]["minimum_unique_positions"]
    ):
        raise GravityItem4CompactnessError("insufficient unique projected positions")
    constants = config["constants"]
    mass = (
        light
        * float(constants["axes_luminosity_unit_lsun"])
        * float(constants["fixed_r_band_mass_to_light_msun_per_lsun"])
    )
    total_mass = float(np.sum(mass))
    r_rms = math.sqrt(float(np.sum(mass * radius**2) / total_mass))
    r50 = _weighted_quantile_radius(radius, mass, 0.5)
    r90 = _weighted_quantile_radius(radius, mass, 0.9)
    if not all(math.isfinite(value) and value > 0 for value in (r_rms, r50, r90)):
        raise GravityItem4CompactnessError("invalid compactness radius")
    epsilon = max(
        float(constants["pair_softening_floor_kpc"]),
        float(constants["pair_softening_fraction_r_rms"]) * r_rms,
    )
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    separation = np.sqrt(dx**2 + dy**2 + epsilon**2)
    upper = np.triu_indices(len(mass), 1)
    pair_kernel = float(np.sum((mass[:, None] * mass[None, :] / separation)[upper]))
    if not math.isfinite(pair_kernel) or pair_kernel <= 0:
        raise GravityItem4CompactnessError("invalid pair binding kernel")
    member_potential = np.sum(mass[None, :] / separation, axis=1) - mass / epsilon
    center_potential = float(np.sum(mass / np.sqrt(radius**2 + epsilon**2)))
    inner = radius <= r50
    outer = radius > r50
    if not np.any(inner) or not np.any(outer):
        raise GravityItem4CompactnessError("invalid potential radial split")
    inner_potential = float(np.sum(mass[inner] * member_potential[inner]) / np.sum(mass[inner]))
    outer_potential = float(np.sum(mass[outer] * member_potential[outer]) / np.sum(mass[outer]))
    q_pair = pair_kernel * r_rms / total_mass**2
    q_center = center_potential * r_rms / total_mass
    contrast = (inner_potential - outer_potential) * r_rms / total_mass
    potential_dispersion = (
        math.sqrt(
            float(
                np.sum(
                    mass * (member_potential - np.sum(mass * member_potential) / total_mass) ** 2
                )
                / total_mass
            )
        )
        * r_rms
        / total_mass
    )
    gravity = float(constants["gravity_kpc_km2_s2_msun"])
    speed_of_light = float(constants["speed_of_light_km_s"])
    acceleration = gravity * total_mass / r_rms**2 * float(constants["km2_s2_per_kpc_to_m_s2"])
    pair_velocity = math.sqrt(gravity * pair_kernel / total_mass)
    values = {
        "log10_mass": math.log10(total_mass),
        "log10_r_rms": math.log10(r_rms),
        "log10_r50": math.log10(r50),
        "log10_r90": math.log10(r90),
        "log10_compactness_c2": math.log10(gravity * total_mass / (r_rms * speed_of_light**2)),
        "log10_acceleration_compactness": math.log10(
            acceleration / float(constants["transition_acceleration_m_s2"])
        ),
        "log10_pair_virial_velocity": math.log10(pair_velocity),
        "log10_q_pair": math.log10(q_pair),
        "log10_q_center": math.log10(q_center),
        "potential_inner_outer_contrast": contrast,
        "potential_dispersion": potential_dispersion,
        "pair_center_interaction": math.log10(q_pair) * math.log10(q_center),
        "pair_kernel_msun2_per_kpc": pair_kernel,
        "r_rms_kpc": r_rms,
        "epsilon_kpc": epsilon,
        "total_mass_msun": total_mass,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise GravityItem4CompactnessError("non-finite compactness feature")
    return values


def _gapper_dispersion(velocity: np.ndarray) -> float:
    ordered = np.sort(np.asarray(velocity, dtype=np.float64))
    count = ordered.size
    if count < 2:
        raise GravityItem4CompactnessError("insufficient velocities for gapper")
    gaps = np.diff(ordered)
    indices = np.arange(1, count, dtype=np.float64)
    return float(
        math.sqrt(math.pi) * np.sum(indices * (count - indices) * gaps) / (count * (count - 1))
    )


def measure_response_only(
    member_redshift: np.ndarray,
    compactness: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Measure fresh dynamics only after the target-blind features are finalized."""

    redshift = np.asarray(member_redshift, dtype=np.float64)
    redshift = redshift[np.isfinite(redshift)]
    if redshift.size < int(config["sample"]["minimum_members"]):
        raise GravityItem4CompactnessError("insufficient finite member redshifts")
    unique = int(np.unique(redshift).size)
    if unique < int(config["quality"]["minimum_unique_redshifts"]):
        raise GravityItem4CompactnessError("insufficient unique member redshifts")
    median = float(np.median(redshift))
    speed_of_light = float(config["constants"]["speed_of_light_km_s"])
    velocity = speed_of_light * (redshift - median) / (1.0 + median)
    gapper = _gapper_dispersion(velocity)
    mad = 1.4826 * float(np.median(np.abs(velocity - np.median(velocity))))
    if not all(math.isfinite(value) and value > 0 for value in (gapper, mad)):
        raise GravityItem4CompactnessError("nonpositive velocity dispersion")
    leave_one_out = np.asarray(
        [_gapper_dispersion(np.delete(velocity, index)) for index in range(len(velocity))]
    )
    loo_fractional_range = float((np.max(leave_one_out) - np.min(leave_one_out)) / gapper)
    if loo_fractional_range > float(
        config["quality"]["maximum_leave_one_out_gapper_fractional_range"]
    ):
        raise GravityItem4CompactnessError("unstable leave-one-member-out gapper")
    gravity = float(config["constants"]["gravity_kpc_km2_s2_msun"])
    pair_scale_squared = (
        gravity
        * float(compactness["pair_kernel_msun2_per_kpc"])
        / float(compactness["total_mass_msun"])
    )
    eta_pair = gapper**2 / pair_scale_squared
    if not math.isfinite(eta_pair) or eta_pair <= 0:
        raise GravityItem4CompactnessError("invalid pair virial ratio")
    return {
        "log10_sigma_gap": math.log10(gapper),
        "log10_sigma_mad": math.log10(mad),
        "log10_eta_pair": math.log10(eta_pair),
        "sigma_gap_km_s": gapper,
        "sigma_mad_km_s": mad,
        "unique_member_redshifts": unique,
        "leave_one_out_gapper_fractional_range": loo_fractional_range,
    }


def _feature_fieldnames() -> list[str]:
    return [
        "group",
        "richness_stratum",
        "members",
        "metadata_redshift",
        "lr195",
        "d10",
        "log10_richness",
        "log10_mass",
        "log10_r_rms",
        "log10_r50",
        "log10_r90",
        "log10_compactness_c2",
        "log10_acceleration_compactness",
        "log10_pair_virial_velocity",
        "log10_q_pair",
        "log10_q_center",
        "potential_inner_outer_contrast",
        "potential_dispersion",
        "pair_center_interaction",
        "pair_kernel_msun2_per_kpc",
        "r_rms_kpc",
        "epsilon_kpc",
        "total_mass_msun",
        "sigma_gap_km_s",
        "sigma_mad_km_s",
        "unique_member_redshifts",
        "leave_one_out_gapper_fractional_range",
        "log10_sigma_gap",
        "log10_sigma_mad",
        "log10_eta_pair",
    ]


def extract_features(root: Path, *, cache_dir: Path) -> tuple[Path, Path]:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    sample = _load_sample(root, config)
    source_path = root / config["source_manifest_output"]
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_manifest(source_manifest, sample=sample)
    source_by_group = {int(row["group"]): row for row in source_manifest["records"]}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for object_row in sample["objects"]:
        if object_row["role"] != "exploration":
            continue
        group = int(object_row["group"])
        path = cache_dir / f"members-{group}.tsv"
        if _sha256_file(path) != source_by_group[group]["sha256"]:
            raise GravityItem4CompactnessError(f"cached source hash changed: {group}")
        members = parse_member_payload(path.read_bytes(), expected_group=group)
        try:
            # This call cannot accept a member redshift; feature construction finishes
            # before the separate response-only call is made.
            compactness = measure_compactness_only(
                np.asarray([row["ra_deg"] for row in members]),
                np.asarray([row["dec_deg"] for row in members]),
                np.asarray([row["luminosity"] for row in members]),
                float(object_row["redshift"]),
                config,
            )
            response = measure_response_only(
                np.asarray([row["member_redshift"] for row in members]),
                compactness,
                config,
            )
        except GravityItem4CompactnessError as exc:
            failures.append({"group": group, "reason": str(exc)})
            continue
        rows.append(
            {
                "group": group,
                "richness_stratum": str(object_row["richness_stratum"]),
                "members": int(object_row["members"]),
                "metadata_redshift": float(object_row["redshift"]),
                "lr195": float(object_row["lr195"]),
                "d10": float(object_row["d10"]),
                "log10_richness": math.log10(int(object_row["members"])),
                **compactness,
                **response,
            }
        )
    rows.sort(key=lambda row: int(row["group"]))
    fields = _feature_fieldnames()
    feature_path = root / config["feature_output"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect="excel-tab")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        str(row[key])
                        if key == "richness_stratum"
                        else int(row[key])
                        if key in {"group", "members", "unique_member_redshifts"}
                        else _metric(float(row[key]))
                    )
                    for key in fields
                }
            )
    quality_pass = not failures and len(rows) == int(config["sample"]["expected_exploration"])
    summary: dict[str, Any] = {
        "schema_version": "invariant-gravity-item4-compactness-extraction-summary-1.0",
        "goal": config["goal"],
        "decision": (
            "PASS_ITEM4_EXPLORATION_REPRESENTATION_QUALITY"
            if quality_pass
            else "FAIL_ITEM4_EXPLORATION_REPRESENTATION_QUALITY"
        ),
        "counts": {
            "quality_failures": len(failures),
            "quality_passing": len(rows),
            "reserved_confirmation_target_accesses": 0,
            "selected_exploration": int(config["sample"]["expected_exploration"]),
        },
        "failures": failures,
        "leakage_boundary": {
            "compactness_finalized_before_response_function": True,
            "compactness_function_accepts_member_redshift": False,
            "published_group_velocity_columns_read": 0,
            "reserved_confirmation_target_accesses": 0,
        },
        "feature_table": {
            "path": config["feature_output"],
            "rows": len(rows),
            "sha256": _sha256_file(feature_path),
        },
    }
    summary["content_sha256"] = canonical_sha256(summary)
    summary_path = root / EXTRACTION_SUMMARY_PATH
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return feature_path, summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("select", "check-sample", "acquire-exploration", "extract-features"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--write-sample-manifest", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("work/item4-compactness-v1-raw"))
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    command = "select" if args.write_sample_manifest else args.command
    if command == "select":
        print(write_sample_manifest(root))
        return 0
    if command == "acquire-exploration":
        print(acquire_exploration(root, cache_dir=args.cache_dir, workers=args.workers))
        return 0
    if command == "extract-features":
        for path in extract_features(root, cache_dir=args.cache_dir):
            print(path)
        return 0
    if command == "check-sample":
        config = load_config(root)
        stored = _load_sample(root, config)
        if build_sample_manifest(root) != stored:
            raise GravityItem4CompactnessError("sample is not an exact rebuild")
        print(root / config["sample_manifest_output"])
        return 0
    raise GravityItem4CompactnessError("no command selected")


if __name__ == "__main__":
    raise SystemExit(main())
