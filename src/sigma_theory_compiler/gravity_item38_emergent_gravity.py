"""Frozen Item 38 emergent-gravity candidate and data-boundary machinery."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
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

CONFIG_PATH = Path("configs/gravity_item38_emergent_gravity_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item38_emergent_gravity.py")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
TEST_PATH = Path("tests/test_gravity_item38_emergent_gravity.py")
COUNTEREXAMPLE_POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")


class GravityItem38Error(RuntimeError):
    """Raised when an Item 38 freeze, leakage boundary, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    expected = "invariant-gravity-item38-emergent-gravity-config-1.0"
    if config.get("schema_version") != expected or int(config.get("item", -1)) != 38:
        raise GravityItem38Error("unexpected Item 38 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem38Error("stable gravity goal changed")
    for relative, expected_hash in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected_hash):
            raise GravityItem38Error(f"scientific dependency changed: {relative}")

    policy = load_counterexample_policy(root / COUNTEREXAMPLE_POLICY_PATH)
    empirical = policy["empirical_evidence"]
    discovery = config["discovery_policy"]
    if not bool(discovery["equal_initial_viability"]):
        raise GravityItem38Error("equal initial viability changed")
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem38Error("single-counterexample retention changed")
    if bool(discovery["counterexample_count_alone_is_never_decisive"]) is not True:
        raise GravityItem38Error("count-only rejection entered Item 38")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem38Error("finite empirical family pruning entered Item 38")
    if bool(discovery["single_object_sensitive_formula_may_promote"]):
        raise GravityItem38Error("single-object-sensitive promotion entered Item 38")
    if empirical["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem38Error("executable counterexample policy changed")

    generator = config["candidate_generator"]
    niches = generator["niches"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem38Error("raw candidate count changed")
    if int(generator["cells_per_niche"]) != 65536 or len(niches) != 4:
        raise GravityItem38Error("equal niche capacity changed")
    if list(generator["grid_shape_per_niche"]) != [16, 16, 16, 16]:
        raise GravityItem38Error("candidate grid shape changed")
    if int(generator["post_response_cells"]) != 0:
        raise GravityItem38Error("post-response candidates entered Item 38")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem38Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem38Error("paid API calls are outside Item 38")
    if bool(config["data_source"]["archive_payload_may_be_read_before_binding"]):
        raise GravityItem38Error("early archive payload access entered Item 38")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    value["source_metadata_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    base = root / str(config["paths"]["source_dir"])
    return {
        name: base / str(config["paths"][name])
        for name in (
            "candidate_manifest",
            "source_metadata_manifest",
            "sample_manifest",
            "exploration_response",
            "compute_manifest",
        )
    }


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    shape = tuple(int(value) for value in config["candidate_generator"]["grid_shape_per_niche"])
    indices = np.indices(shape, dtype=np.int16).reshape(4, -1).T
    cells = indices.shape[0]
    lane = np.repeat(np.arange(4, dtype=np.int8), cells)
    tiled = np.tile(indices, (4, 1))
    candidate_id = np.arange(4 * cells, dtype=np.int64)
    return {
        "candidate_id": candidate_id,
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


def predict_multiplier(
    candidates: Mapping[str, np.ndarray], u: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    """Return g_pred/g_bar for candidate rows and dimensionless acceleration columns."""

    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or np.any(~np.isfinite(u)) or np.any(u <= 0.0):
        raise GravityItem38Error("u must be a finite positive vector")
    lane = np.asarray(candidates["lane"], dtype=np.int8)
    amplitude, exponent, transition, shape = _candidate_parameters(candidates, config)
    uu = u[None, :]
    aa = amplitude[:, None]
    pp = exponent[:, None]
    tt = transition[:, None]
    ss = shape[:, None]
    result = np.empty((len(lane), len(u)), dtype=np.float64)

    mask = lane == 0
    if np.any(mask):
        base = np.power(uu, -pp[mask])
        transition_term = np.power(1.0 + np.power(uu / tt[mask], ss[mask]), -0.5 / ss[mask])
        result[mask] = 1.0 + aa[mask] * base * transition_term

    mask = lane == 1
    if np.any(mask):
        entropy = np.log1p(tt[mask] / uu)
        response = np.power(1.0 + aa[mask] * np.power(entropy, pp[mask]), 1.0 / ss[mask])
        result[mask] = response

    mask = lane == 2
    if np.any(mask):
        generalized = np.power(
            1.0 + np.power(aa[mask] * tt[mask] / uu, pp[mask]),
            ss[mask] / pp[mask],
        )
        result[mask] = generalized

    mask = lane == 3
    if np.any(mask):
        information = np.log1p(tt[mask] / uu)
        saturation = 1.0 + np.power(information / ss[mask], ss[mask])
        result[mask] = 1.0 + aa[mask] * np.power(information, pp[mask]) / saturation
    return result


def admissible_candidates(
    config: Mapping[str, Any], *, batch_size: int = 8192
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    admission = config["candidate_generator"]["admissibility"]
    u = np.logspace(
        float(admission["probe_log10_u_min"]),
        float(admission["probe_log10_u_max"]),
        int(admission["probe_points"]),
    )
    keep_parts: list[np.ndarray] = []
    rejection_counts: Counter[str] = Counter()
    signatures: set[bytes] = set()
    decimals = int(admission["behavior_signature_decimals"])
    total = len(raw["candidate_id"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        rows = {key: value[start:stop] for key, value in raw.items()}
        values = predict_multiplier(rows, u, config)
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
        rejection_counts["nonfinite"] += int(np.sum(~finite))
        rejection_counts["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection_counts["no_local_limit"] += int(np.sum(bounded & ~local))
        rejection_counts["immaterial"] += int(np.sum(local & ~material))
        rejection_counts["nonmonotone"] += int(np.sum(material & ~monotone))
        admitted_local = np.flatnonzero(monotone)
        keep_parts.append(admitted_local + start)
        rounded = np.round(np.log10(values[admitted_local]), decimals=decimals)
        for row in rounded:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())

    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    admitted = {key: value[keep] for key, value in raw.items()}
    lane_counts = {
        str(int(lane)): int(np.sum(admitted["lane"] == lane)) for lane in range(4)
    }
    audit = {
        "raw_candidates": total,
        "admitted_candidates": len(keep),
        "rejected_candidates": int(total - len(keep)),
        "admitted_by_lane": lane_counts,
        "behavioral_equivalence_classes": len(signatures),
        "rejection_counts_nonexclusive": dict(sorted(rejection_counts.items())),
        "probe_u": [float(value) for value in u],
    }
    return admitted, audit


def decode_candidate(
    candidate_id: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem38Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition, shape = _candidate_parameters(row, config)
    lane_id = int(row["lane"][0])
    niche = config["candidate_generator"]["niches"][lane_id]
    return {
        "candidate_id": candidate_id,
        "lane_id": lane_id,
        "lane": niche["name"],
        "creativity_label": niche["creativity_label"],
        "micro_to_macro_claim": niche["micro_to_macro_claim"],
        "parameters": {
            "amplitude": float(amplitude[0]),
            "exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
            "shape": float(shape[0]),
        },
    }


def fixed_control_multiplier(name: str, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if name == "baryonic_newton":
        return np.ones_like(u)
    if name == "verlinde_point_mass":
        return 1.0 + np.sqrt(1.0 / (6.0 * u))
    if name == "mond_RAR":
        return 1.0 / (-np.expm1(-np.sqrt(u)))
    raise GravityItem38Error(f"unknown fixed control: {name}")


def build_candidate_manifest(root: Path, output: Path) -> dict[str, Any]:
    config = load_config(root)
    admitted, audit = admissible_candidates(config)
    niches = config["candidate_generator"]["niches"]
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item38-candidate-manifest-1.0",
            "item": 38,
            "config_contract_sha256": _contract_digest(config),
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
            "niches": niches,
            "known_point_mass_control": {
                "formula": config["fixed_controls"]["verlinde_point_mass"],
                "creativity_label": "known_formula_control_not_generated_novelty",
            },
            "claim_boundaries": [
                "candidate formulas are macroscopic projections, not complete microscopic theories",
                "generalized entropy is prior art and cannot be claimed as new",
                "behavioral uniqueness is not historical or mathematical novelty",
                "no response, confirmation profile, or paid model call entered generation",
            ],
        }
    )
    _write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidates = subparsers.add_parser("build-candidates")
    candidates.add_argument("--root", type=Path, default=Path("."))
    candidates.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-candidates":
        result = build_candidate_manifest(args.root.resolve(), args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    raise GravityItem38Error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
