"""Frozen source, sample, and compactness derivation for gravity roadmap Item 4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item4_baryonic_compactness_groups_v1.json"


class GravityItem4CompactnessError(RuntimeError):
    """Raised when the frozen Item 4 boundary or derivation drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            (root / config["source"][f"{prefix}_exclusion_manifest"]).read_text(
                encoding="utf-8"
            )
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
    objects.sort(key=lambda row: (str(row["richness_stratum"]), str(row["role"]), int(row["group"])))
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item4-compactness-sample-1.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM4_TARGET_BLIND_FRESH_GROUP_SAMPLE",
        "counts": {
            "eligible_before_prior_exclusion": len(eligible_before),
            "prior_group_ids_excluded": len(prior),
            "remaining": len(remaining),
            "exploration": sum(row["role"] == "exploration" for row in objects),
            "reserved_confirmation": sum(
                row["role"] == "reserved_confirmation" for row in objects
            ),
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


def _weighted_quantile_radius(
    radius: np.ndarray, weight: np.ndarray, quantile: float
) -> float:
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
    mean_ra = math.atan2(
        float(np.sum(light * np.sin(angle))), float(np.sum(light * np.cos(angle)))
    )
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
        math.sqrt(float(np.sum(mass * (member_potential - np.sum(mass * member_potential) / total_mass) ** 2) / total_mass))
        * r_rms
        / total_mass
    )
    gravity = float(constants["gravity_kpc_km2_s2_msun"])
    speed_of_light = float(constants["speed_of_light_km_s"])
    acceleration = (
        gravity
        * total_mass
        / r_rms**2
        * float(constants["km2_s2_per_kpc_to_m_s2"])
    )
    pair_velocity = math.sqrt(gravity * pair_kernel / total_mass)
    values = {
        "log10_mass": math.log10(total_mass),
        "log10_r_rms": math.log10(r_rms),
        "log10_r50": math.log10(r50),
        "log10_r90": math.log10(r90),
        "log10_compactness_c2": math.log10(
            gravity * total_mass / (r_rms * speed_of_light**2)
        ),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--write-sample-manifest", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.write_sample_manifest:
        print(write_sample_manifest(args.root))
    else:
        raise GravityItem4CompactnessError("no command selected")


if __name__ == "__main__":
    main()
