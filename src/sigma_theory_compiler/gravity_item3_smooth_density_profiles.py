"""Frozen smooth-profile derivation and source boundary for gravity Item 3 attempt 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item3_smooth_density_profiles_v2.json"
SAMPLE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-03-smooth-density-profiles-v2-source/sample-manifest.json"
)


class GravityItem3SmoothDensityError(RuntimeError):
    """Raised when the frozen attempt-2 derivation or source boundary drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(root: Path) -> dict[str, Any]:
    """Load the frozen contract and verify its immutable predecessors."""

    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item3-smooth-density-profiles-config-2.0"
    ):
        raise GravityItem3SmoothDensityError("unexpected Item 3 attempt-2 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem3SmoothDensityError("stable roadmap binding changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem3SmoothDensityError("attempt-1 receipt file changed")
    predecessor_receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor_receipt.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem3SmoothDensityError("attempt-1 receipt content changed")
    if predecessor_receipt.get("decision") != predecessor["required_decision"]:
        raise GravityItem3SmoothDensityError("attempt-1 decision changed")
    authorization = config["authorization"]
    if authorization["paid_model_calls_allowed"]:
        raise GravityItem3SmoothDensityError("paid model calls are forbidden")
    if authorization["reserved_xcop_confirmation_profile_accesses_allowed"] != 0:
        raise GravityItem3SmoothDensityError("confirmation access budget must be zero")
    if config["cluster_lane"]["reserved_profile_accesses_allowed"] != 0:
        raise GravityItem3SmoothDensityError("cluster confirmation access budget drifted")
    return config


def effective_density_pair(
    radius: np.ndarray,
    q_density: np.ndarray,
    *,
    support_dimension: int,
    gravity_constant: float,
    transition_acceleration: float,
    scale_clip: tuple[float, float] = (0.05, 20.0),
) -> dict[str, np.ndarray]:
    """Convert a smooth radial d-density into common surface/volume sources.

    ``q_density`` has units mass/length**d and ``radius`` uses the matching
    length unit.  ``gravity_constant`` must therefore use compatible mass,
    length, acceleration units.  No dynamics target enters this function.
    """

    radius = np.asarray(radius, dtype=np.float64)
    q_density = np.asarray(q_density, dtype=np.float64)
    if (
        support_dimension not in {2, 3}
        or radius.ndim != 1
        or radius.shape != q_density.shape
        or radius.size < 5
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(q_density))
        or np.any(radius <= 0)
        or np.any(q_density <= 0)
        or np.any(np.diff(radius) <= 0)
        or not math.isfinite(gravity_constant)
        or gravity_constant <= 0
        or not math.isfinite(transition_acceleration)
        or transition_acceleration <= 0
    ):
        raise GravityItem3SmoothDensityError("invalid smooth baryonic profile")
    lower, upper = (float(value) for value in scale_clip)
    if not 0 < lower < upper:
        raise GravityItem3SmoothDensityError("invalid scale-length clip")

    log_gradient = np.gradient(np.log(q_density), radius, edge_order=2)
    raw_h = np.divide(
        1.0,
        np.abs(log_gradient),
        out=np.full_like(log_gradient, np.inf),
        where=np.abs(log_gradient) > 1.0e-15,
    )
    raw_h_over_r = raw_h / radius
    h_over_r = np.clip(raw_h_over_r, lower, upper)
    scale_length = h_over_r * radius
    sigma_eff = q_density * (2.0 * scale_length) ** (support_dimension - 2)
    rho_eff = q_density / (2.0 * scale_length) ** (3 - support_dimension)
    u_surface = (
        math.pi * gravity_constant * sigma_eff / transition_acceleration
    )
    u_volume = (
        4.0
        * math.pi
        * gravity_constant
        * radius
        * rho_eff
        / (3.0 * transition_acceleration)
    )
    if np.any(~np.isfinite(u_surface)) or np.any(~np.isfinite(u_volume)):
        raise GravityItem3SmoothDensityError("non-finite density source")
    if np.any(u_surface <= 0) or np.any(u_volume <= 0):
        raise GravityItem3SmoothDensityError("non-positive density source")

    s = np.log10(u_surface)
    v = np.log10(u_volume)
    mean = 0.5 * (s + v)
    contrast = s - v
    surface_transition = u_surface / (1.0 + u_surface) ** 2
    volume_transition = u_volume / (1.0 + u_volume) ** 2
    transition_sum = surface_transition + volume_transition
    values = {
        "scale_length": scale_length,
        "scale_clip_mask": raw_h_over_r != h_over_r,
        "sigma_eff": sigma_eff,
        "rho_eff": rho_eff,
        "u_surface": u_surface,
        "u_volume": u_volume,
        "s": s,
        "v": v,
        "m": mean,
        "c": contrast,
        "m_x_c": mean * contrast,
        "transition_product": surface_transition * volume_transition,
        "transition_balance": (surface_transition - volume_transition)
        / transition_sum,
    }
    if any(np.any(~np.isfinite(value)) for value in values.values()):
        raise GravityItem3SmoothDensityError("non-finite derived density feature")
    return values


def radial_feature_basis(
    *,
    gbar: np.ndarray,
    density_pair: Mapping[str, np.ndarray],
    transition_acceleration: float,
    population_proxy: float,
) -> dict[str, np.ndarray]:
    """Build the frozen model basis without accepting a dynamics target."""

    gbar = np.asarray(gbar, dtype=np.float64)
    if (
        gbar.ndim != 1
        or np.any(~np.isfinite(gbar))
        or np.any(gbar <= 0)
        or not math.isfinite(population_proxy)
    ):
        raise GravityItem3SmoothDensityError("invalid baryonic acceleration")
    required = {"s", "v", "m", "c", "m_x_c", "transition_product", "transition_balance"}
    if not required.issubset(density_pair):
        raise GravityItem3SmoothDensityError("incomplete density pair")
    if any(np.asarray(density_pair[key]).shape != gbar.shape for key in required):
        raise GravityItem3SmoothDensityError("density/baryonic profile mismatch")
    a = np.log10(gbar / float(transition_acceleration))
    mean = np.asarray(density_pair["m"], dtype=np.float64)
    contrast = np.asarray(density_pair["c"], dtype=np.float64)
    return {
        "a": a,
        "a2": a**2,
        "a3": a**3,
        "s": np.asarray(density_pair["s"], dtype=np.float64),
        "v": np.asarray(density_pair["v"], dtype=np.float64),
        "m": mean,
        "c": contrast,
        "m_x_c": np.asarray(density_pair["m_x_c"], dtype=np.float64),
        "a_x_m": a * mean,
        "a_x_c": a * contrast,
        "transition_product": np.asarray(
            density_pair["transition_product"], dtype=np.float64
        ),
        "transition_balance": np.asarray(
            density_pair["transition_balance"], dtype=np.float64
        ),
        "population_proxy": np.full_like(a, float(population_proxy)),
    }


def _salted_order(names: Sequence[str], *, salt: str, stratum: str) -> list[str]:
    return sorted(
        names,
        key=lambda name: hashlib.sha256(
            f"{salt}|{stratum}|{name}".encode()
        ).hexdigest(),
    )


def build_sample_manifest(root: Path) -> dict[str, Any]:
    """Build the target-blind manifest solely from frozen names and source metadata."""

    root = root.resolve()
    config = load_config(root)
    lane = config["cluster_lane"]
    with_stars = sorted(
        set(lane["exploration_with_stellar_profile"])
        | set(lane["reserved_with_stellar_profile"])
    )
    without_stars = sorted(
        set(lane["exploration_without_stellar_profile"])
        | set(lane["reserved_without_stellar_profile"])
    )
    salt = str(lane["selection_salt"])
    expected_with = _salted_order(
        with_stars, salt=salt, stratum="with_mstar"
    )
    expected_without = _salted_order(
        without_stars, salt=salt, stratum="without_mstar"
    )
    if set(expected_with[:5]) != set(lane["exploration_with_stellar_profile"]):
        raise GravityItem3SmoothDensityError("with-star exploration split drifted")
    if set(expected_with[5:]) != set(lane["reserved_with_stellar_profile"]):
        raise GravityItem3SmoothDensityError("with-star confirmation split drifted")
    if set(expected_without[:3]) != set(lane["exploration_without_stellar_profile"]):
        raise GravityItem3SmoothDensityError("without-star exploration split drifted")
    if set(expected_without[3:]) != set(lane["reserved_without_stellar_profile"]):
        raise GravityItem3SmoothDensityError("without-star confirmation split drifted")

    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item3-smooth-density-sample-manifest-2.0",
        "goal": config["goal"],
        "selection_used_response_values": False,
        "galaxy_development": list(config["galaxy_lane"]["matched_development_objects"]),
        "galaxy_independent_confirmation": [],
        "cluster_exploration": sorted(lane["exploration_objects"]),
        "cluster_reserved_confirmation": sorted(lane["reserved_confirmation_objects"]),
        "cluster_selection_salt": salt,
        "cluster_selection_strata": {
            "with_mstar": {
                "salted_order": expected_with,
                "exploration_count": 5,
                "reserved_count": 2,
            },
            "without_mstar": {
                "salted_order": expected_without,
                "exploration_count": 3,
                "reserved_count": 2,
            },
        },
        "counts": {
            "galaxy_development": len(
                config["galaxy_lane"]["matched_development_objects"]
            ),
            "cluster_exploration": len(lane["exploration_objects"]),
            "cluster_reserved_confirmation": len(lane["reserved_confirmation_objects"]),
        },
        "confirmation_access_budget": 0,
        "content_sha256": None,
    }
    content = dict(manifest)
    content.pop("content_sha256")
    manifest["content_sha256"] = canonical_sha256(content)
    return manifest


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    destination = root / SAMPLE_MANIFEST_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return destination


def verify_local_source_archives(root: Path, cache_dir: Path) -> dict[str, Any]:
    """Verify archive hashes and names without extracting any profile rows."""

    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    xcop_path = cache_dir / "xcop-allfiles.tar.gz"
    sparc_path = cache_dir / "sparc-rotmod.zip"
    if _sha256_file(xcop_path) != config["cluster_lane"]["archive_sha256"]:
        raise GravityItem3SmoothDensityError("X-COP archive hash changed")
    if _sha256_file(sparc_path) != config["galaxy_lane"]["dynamics_catalog"][
        "archive_sha256"
    ]:
        raise GravityItem3SmoothDensityError("SPARC archive hash changed")
    with tarfile.open(xcop_path, mode="r:gz") as archive:
        xcop_names = set(archive.getnames())
    with zipfile.ZipFile(sparc_path) as archive:
        sparc_names = set(archive.namelist())
    clusters = config["cluster_lane"]["exploration_objects"] + config["cluster_lane"][
        "reserved_confirmation_objects"
    ]
    for cluster in clusters:
        for suffix in ("density_L1.fits", "pressure.fits"):
            expected = f"{cluster}/{cluster}_{suffix}"
            if expected not in xcop_names:
                raise GravityItem3SmoothDensityError(f"missing X-COP member: {expected}")
    for galaxy in config["galaxy_lane"]["matched_development_objects"]:
        expected = f"{galaxy}_rotmod.dat"
        if expected not in sparc_names:
            raise GravityItem3SmoothDensityError(f"missing SPARC member: {expected}")
    return {
        "xcop_archive_sha256": _sha256_file(xcop_path),
        "xcop_member_count": len(xcop_names),
        "sparc_archive_sha256": _sha256_file(sparc_path),
        "sparc_member_count": len(sparc_names),
        "profile_rows_opened": 0,
        "confirmation_profiles_opened": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write-sample-manifest")
    verify = subparsers.add_parser("verify-archives")
    verify.add_argument("--cache-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    if args.command == "write-sample-manifest":
        print(write_sample_manifest(root))
    elif args.command == "verify-archives":
        print(json.dumps(verify_local_source_archives(root, args.cache_dir), sort_keys=True))
    else:  # pragma: no cover
        raise GravityItem3SmoothDensityError("unknown command")


if __name__ == "__main__":
    main()
