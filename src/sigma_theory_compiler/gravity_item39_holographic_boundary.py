"""Frozen Item 39 holographic/boundary-gravity search machinery."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import (
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item39_holographic_boundary_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")


class GravityItem39Error(RuntimeError):
    """Raised when an Item 39 freeze, leakage boundary, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != (
        "invariant-gravity-item39-holographic-boundary-config-1.0"
    ) or int(config.get("item", -1)) != 39:
        raise GravityItem39Error("unexpected Item 39 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem39Error("stable gravity goal changed")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem39Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-38-emergent-gravity-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem39Error("Item 38 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem39Error("Item 38 decision changed")

    policy = load_counterexample_policy(root / POLICY_PATH)
    empirical = policy["empirical_evidence"]
    discovery = config["discovery_policy"]
    if not bool(discovery["equal_initial_viability"]):
        raise GravityItem39Error("equal initial viability changed")
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem39Error("single-counterexample retention changed")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem39Error("count-only empirical rejection entered Item 39")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem39Error("finite empirical family pruning entered Item 39")
    if bool(discovery["single_object_sensitive_formula_may_promote"]):
        raise GravityItem39Error("single-object-sensitive promotion entered Item 39")
    if empirical["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem39Error("executable counterexample policy changed")

    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem39Error("raw candidate count changed")
    if int(generator["cells_per_niche"]) != 65536:
        raise GravityItem39Error("per-niche capacity changed")
    if list(generator["grid_shape_per_niche"]) != [16, 16, 16, 16]:
        raise GravityItem39Error("candidate grid shape changed")
    if len(generator["niches"]) != 4:
        raise GravityItem39Error("Item 39 requires four equal niches")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem39Error("post-response candidates entered Item 39")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem39Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem39Error("paid model calls entered Item 39")
    metric = config["weak_field_metric_contract"]
    if metric["gravitational_slip"] != "Phi=Psi":
        raise GravityItem39Error("motion/light metric closure changed")
    if not bool(config["independence"]["exclude_every_item10_sample_name_and_coordinate"]):
        raise GravityItem39Error("Item 10 role exclusion changed")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    for key in ("scientific_freeze_commit", "predictor_freeze_commit", "sample_freeze_commit"):
        value[key] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    shape = tuple(int(value) for value in config["candidate_generator"]["grid_shape_per_niche"])
    indices = np.indices(shape, dtype=np.int16).reshape(4, -1).T
    cells = indices.shape[0]
    lane = np.repeat(np.arange(4, dtype=np.int8), cells)
    tiled = np.tile(indices, (4, 1))
    return {
        "candidate_id": np.arange(4 * cells, dtype=np.int64),
        "lane": lane,
        "amplitude_index": tiled[:, 0],
        "exponent_index": tiled[:, 1],
        "transition_index": tiled[:, 2],
        "shape_index": tiled[:, 3],
    }


def _candidate_parameters(
    candidates: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grids = config["candidate_generator"]["parameter_grids"]
    amplitude = np.asarray(grids["amplitude"], dtype=np.float64)[
        np.asarray(candidates["amplitude_index"], dtype=np.int64)
    ]
    exponent = np.asarray(grids["exponent"], dtype=np.float64)[
        np.asarray(candidates["exponent_index"], dtype=np.int64)
    ]
    transition = np.asarray(grids["transition_u"], dtype=np.float64)[
        np.asarray(candidates["transition_index"], dtype=np.int64)
    ]
    shape = np.asarray(grids["shape"], dtype=np.float64)[
        np.asarray(candidates["shape_index"], dtype=np.int64)
    ]
    return amplitude, exponent, transition, shape


def boundary_coordinates(
    enclosed_fraction: np.ndarray,
    radius_over_screen: np.ndarray,
    enclosed_log_slope: np.ndarray,
) -> np.ndarray:
    """Return the four frozen dimensionless nested-screen coordinates."""

    fraction = np.asarray(enclosed_fraction, dtype=np.float64)
    radius = np.asarray(radius_over_screen, dtype=np.float64)
    slope = np.asarray(enclosed_log_slope, dtype=np.float64)
    if not (fraction.shape == radius.shape == slope.shape) or fraction.ndim != 1:
        raise GravityItem39Error("boundary inputs must be aligned vectors")
    if np.any(~np.isfinite(fraction)) or np.any(~np.isfinite(radius)) or np.any(~np.isfinite(slope)):
        raise GravityItem39Error("boundary inputs must be finite")
    if np.any((fraction < 0.0) | (fraction > 1.0)) or np.any(radius <= 0.0):
        raise GravityItem39Error("boundary inputs outside frozen domains")
    floor = 1e-12
    area = np.square(radius)
    equipartition = np.abs(fraction - area) / (fraction + area + floor)
    compactness = fraction / radius
    quasilocal = np.abs(compactness - 1.0) / (compactness + 1.0 + floor)
    mass_cross = np.clip(4.0 * fraction * (1.0 - fraction), 0.0, 1.0)
    radial_cross = np.clip(4.0 * radius / np.square(1.0 + radius), 0.0, 1.0)
    wedge = np.sqrt(mass_cross * radial_cross)
    flow = np.abs(slope - 2.0) / (np.abs(slope) + 2.0 + floor)
    return np.stack((equipartition, quasilocal, wedge, flow), axis=0)


def predict_multiplier(
    candidates: Mapping[str, np.ndarray],
    u: np.ndarray,
    enclosed_fraction: np.ndarray,
    radius_over_screen: np.ndarray,
    enclosed_log_slope: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Return the universal motion/lensing multiplier for candidate rows."""

    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or np.any(~np.isfinite(u)) or np.any(u <= 0.0):
        raise GravityItem39Error("u must be a finite positive vector")
    coordinates = boundary_coordinates(
        enclosed_fraction, radius_over_screen, enclosed_log_slope
    )
    if coordinates.shape[1] != len(u):
        raise GravityItem39Error("boundary and acceleration vectors differ")
    lane = np.asarray(candidates["lane"], dtype=np.int8)
    amplitude, exponent, transition, shape = _candidate_parameters(candidates, config)
    result = np.empty((len(lane), len(u)), dtype=np.float64)
    uu = u[None, :]
    for lane_id in range(4):
        mask = lane == lane_id
        if not np.any(mask):
            continue
        aa = amplitude[mask, None]
        pp = exponent[mask, None]
        tt = transition[mask, None]
        ss = shape[mask, None]
        hh = coordinates[lane_id][None, :]
        envelope = aa * np.power(uu, -pp) * np.power(
            1.0 + np.power(uu / tt, ss), -1.0 / ss
        )
        if lane_id == 0:
            boundary = 0.05 + 0.95 * np.power(hh, ss)
        elif lane_id == 1:
            boundary = 0.05 + 0.95 * np.power(hh / (0.25 + hh), ss)
        elif lane_id == 2:
            boundary = 0.05 + 0.95 * np.power(np.sin(0.5 * np.pi * hh), ss)
        else:
            boundary = 0.05 + 0.95 * (1.0 - np.exp(-ss * hh))
        result[mask] = 1.0 + envelope * boundary
    return result


def metric_observables(g_bar: np.ndarray, multiplier: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the frozen zero-slip weak-field closure to motion and light."""

    g_bar = np.asarray(g_bar, dtype=np.float64)
    multiplier = np.asarray(multiplier, dtype=np.float64)
    if g_bar.shape != multiplier.shape:
        raise GravityItem39Error("metric inputs differ")
    field = g_bar * multiplier
    return {
        "g_dynamics": field,
        "grad_phi": field,
        "grad_psi": field,
        "lensing_integrand_grad_phi_plus_psi": 2.0 * field,
    }


def fixed_control_multiplier(name: str, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if name == "baryonic_newton":
        return np.ones_like(u)
    if name == "mond_RAR":
        return 1.0 / (-np.expm1(-np.sqrt(u)))
    if name == "item38_selected":
        return 1.0 + 2.75 * np.power(u, -0.45) * np.power(
            1.0 + np.power(u / 0.01, 0.5), -2.0
        )
    raise GravityItem39Error(f"unknown fixed control: {name}")


def admissible_candidates(
    config: Mapping[str, Any], *, batch_size: int = 4096
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    admission = config["candidate_generator"]["admissibility"]
    u = np.logspace(
        float(admission["probe_log10_u_min"]),
        float(admission["probe_log10_u_max"]),
        int(admission["probe_points"]),
    )
    fraction = np.clip(0.05 + 0.9 / (1.0 + np.exp(-np.linspace(-3.0, 3.0, len(u)))), 0, 1)
    radius = np.logspace(-1.0, 0.15, len(u))
    slope = 0.4 + 2.4 * np.square(np.sin(np.linspace(0.0, np.pi, len(u))))
    keep_parts: list[np.ndarray] = []
    rejection_counts: Counter[str] = Counter()
    signatures: set[bytes] = set()
    decimals = int(admission["behavior_signature_decimals"])
    total = len(raw["candidate_id"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        rows = {key: value[start:stop] for key, value in raw.items()}
        values = predict_multiplier(rows, u, fraction, radius, slope, config)
        finite = np.all(np.isfinite(values), axis=1)
        bounded = finite & np.all(values >= float(admission["minimum_multiplier"]), axis=1)
        bounded &= np.all(values <= float(admission["maximum_multiplier"]), axis=1)
        local = bounded & (
            np.abs(np.log10(values[:, -1]))
            <= float(admission["maximum_high_acceleration_log10_deviation"])
        )
        material = local & (
            values[:, 16] >= float(admission["minimum_low_acceleration_multiplier"])
        )
        monotone = material & np.all(
            np.diff(values, axis=1)
            <= float(admission["monotone_nonincreasing_tolerance"]),
            axis=1,
        )
        test_u = np.asarray([0.01, 0.01], dtype=np.float64)
        low_boundary = predict_multiplier(
            rows,
            test_u,
            np.asarray([0.02, 0.85]),
            np.asarray([0.15, 0.95]),
            np.asarray([2.0, 0.4]),
            config,
        )
        boundary_span = np.abs(np.diff(np.log10(low_boundary), axis=1)[:, 0])
        boundary_material = monotone & (
            boundary_span >= float(admission["minimum_boundary_log10_span"])
        )
        rejection_counts["nonfinite"] += int(np.sum(~finite))
        rejection_counts["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection_counts["no_local_limit"] += int(np.sum(bounded & ~local))
        rejection_counts["immaterial_low_acceleration"] += int(np.sum(local & ~material))
        rejection_counts["nonmonotone"] += int(np.sum(material & ~monotone))
        rejection_counts["ordinary_rewrite_boundary_immaterial"] += int(
            np.sum(monotone & ~boundary_material)
        )
        admitted_local = np.flatnonzero(boundary_material)
        keep_parts.append(admitted_local + start)
        rounded = np.round(np.log10(values[admitted_local]), decimals=decimals)
        for row in rounded:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    admitted = {key: value[keep] for key, value in raw.items()}
    audit = {
        "raw_candidates": total,
        "admitted_candidates": len(keep),
        "rejected_candidates": int(total - len(keep)),
        "admitted_by_lane": {
            str(lane): int(np.sum(admitted["lane"] == lane)) for lane in range(4)
        },
        "behavioral_equivalence_classes": len(signatures),
        "rejection_counts_nonexclusive": dict(sorted(rejection_counts.items())),
    }
    return admitted, audit


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem39Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition, shape = _candidate_parameters(row, config)
    lane_id = int(row["lane"][0])
    niche = config["candidate_generator"]["niches"][lane_id]
    return {
        "candidate_id": candidate_id,
        "lane_id": lane_id,
        "lane": niche["name"],
        "creativity_label": niche["creativity_label"],
        "boundary_coordinate": niche["boundary_coordinate"],
        "parameters": {
            "amplitude": float(amplitude[0]),
            "exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
            "shape": float(shape[0]),
        },
    }


def build_candidate_manifest(root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config(root))
    admitted, audit = admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-candidate-manifest-1.0",
            "item": 39,
            "config_contract_sha256": _contract_digest(config),
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_accessed": False,
            "confirmation_accessed": False,
            "paid_api_calls": 0,
            "raw_candidates": audit["raw_candidates"],
            "admitted_candidates": audit["admitted_candidates"],
            "admitted_by_lane": audit["admitted_by_lane"],
            "behavioral_equivalence_classes": audit["behavioral_equivalence_classes"],
            "rejection_counts_nonexclusive": audit["rejection_counts_nonexclusive"],
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "niches": config["candidate_generator"]["niches"],
            "metric_contract": config["weak_field_metric_contract"],
            "claim_boundaries": [
                "candidate mappings are weak-field phenomenological projections, not complete theories",
                "screen, quasilocal, and MOND-like ideas are known prior art",
                "potentially new synthesis does not establish historical novelty",
                "behavioral distinction from controls must be demonstrated on held-out data",
                "no response, confirmation value, or paid model call entered generation",
            ],
        }
    )


def build_exposure_manifest(root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config(root))
    path = root / str(config["independence"]["item10_sample_path"])
    item10 = _read_json(path)
    names = sorted({str(row["name"]) for row in item10["objects"]})
    coordinates = sorted(
        {
            (f"{float(row['ra']):.12e}", f"{float(row['dec']):.12e}")
            for row in item10["objects"]
        }
    )
    role_counts = Counter(str(row["role"]) for row in item10["objects"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-predecessor-exposure-1.0",
            "item": 39,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "source_path": str(config["independence"]["item10_sample_path"]),
            "source_sha256": _sha256_file(path),
            "excluded_names": names,
            "excluded_coordinates": [
                {"ra": ra, "dec": dec} for ra, dec in coordinates
            ],
            "coordinate_exclusion_arcseconds": float(
                config["independence"]["coordinate_exclusion_arcseconds"]
            ),
            "role_counts": dict(sorted(role_counts.items())),
            "rules": [
                "exclude every Item 10 exploration identity",
                "exclude every Item 10 reserved-confirmation identity",
                "exclude coordinate neighbors before response access",
                "never repair a failed sample identity after response access",
            ],
            "response_values_read_while_building": 0,
        }
    )


def write_freeze_manifests(root: Path) -> dict[str, Path]:
    config = load_config(root)
    candidate_path = _source_path(root, config, "candidate_manifest")
    exposure_path = _source_path(root, config, "exposure_manifest")
    _write_json(candidate_path, build_candidate_manifest(root, config))
    _write_json(exposure_path, build_exposure_manifest(root, config))
    return {"candidate_manifest": candidate_path, "exposure_manifest": exposure_path}


def check_freeze(root: Path) -> dict[str, Any]:
    config = load_config(root)
    candidate_path = _source_path(root, config, "candidate_manifest")
    exposure_path = _source_path(root, config, "exposure_manifest")
    candidate = _read_json(candidate_path)
    exposure = _read_json(exposure_path)
    if candidate != build_candidate_manifest(root, config):
        raise GravityItem39Error("candidate manifest drifted")
    if exposure != build_exposure_manifest(root, config):
        raise GravityItem39Error("exposure manifest drifted")
    return {
        "status": "ITEM39_SCIENTIFIC_FREEZE_VALID",
        "candidate_manifest_sha256": _sha256_file(candidate_path),
        "exposure_manifest_sha256": _sha256_file(exposure_path),
        "response_rows_read": 0,
        "confirmation_rows_read": 0,
        "paid_api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check-freeze"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps({key: str(value) for key, value in write_freeze_manifests(args.root).items()}, sort_keys=True))
    else:
        print(json.dumps(check_freeze(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
