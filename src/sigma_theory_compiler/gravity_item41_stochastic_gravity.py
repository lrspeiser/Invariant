"""Frozen Item 41 stochastic-gravity formula generator and audit boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import load_counterexample_policy
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item41_stochastic_gravity_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")


class GravityItem41Error(RuntimeError):
    """Raised when an Item 41 freeze, stochastic law, or leakage boundary fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item41-stochastic-gravity-config-1.0"
        or int(config.get("item", -1)) != 41
    ):
        raise GravityItem41Error("unexpected Item 41 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem41Error("stable gravity goal changed")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem41Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-40-discrete-network-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem41Error("Item 40 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem41Error("Item 40 decision changed")
    empirical = load_counterexample_policy(root / POLICY_PATH)["empirical_evidence"]
    discovery = config["discovery_policy"]
    if not bool(discovery["equal_initial_viability"]):
        raise GravityItem41Error("equal initial viability changed")
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem41Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem41Error("count-only rejection entered Item 41")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem41Error("finite empirical family pruning entered Item 41")
    if empirical["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem41Error("executable counterexample policy changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem41Error("raw candidate count changed")
    if int(generator["cells_per_niche"]) != 65536:
        raise GravityItem41Error("per-niche candidate count changed")
    if list(generator["grid_shape_per_niche"]) != [16, 16, 16, 16]:
        raise GravityItem41Error("candidate grid shape changed")
    if len(generator["niches"]) != 4 or int(generator["post_response_cells"]) != 0:
        raise GravityItem41Error("stochastic niche or generation boundary changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem41Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem41Error("paid calls entered Item 41")
    if bool(config["scope"]["fresh_confirmation_claimed"]):
        raise GravityItem41Error("retrospective data were called fresh")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    cells = int(config["candidate_generator"]["cells_per_niche"])
    candidate_id = np.arange(total, dtype=np.int64)
    lane = (candidate_id // cells).astype(np.int8)
    local = candidate_id % cells
    return {
        "candidate_id": candidate_id,
        "lane": lane,
        "sigma_index": ((local // 4096) % 16).astype(np.int8),
        "exponent_index": ((local // 256) % 16).astype(np.int8),
        "transition_index": ((local // 16) % 16).astype(np.int8),
        "radial_index": (local % 16).astype(np.int8),
    }


def _candidate_parameters(
    candidates: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grids = config["candidate_generator"]["parameter_grids"]
    sigma = np.asarray(grids["sigma0"])[np.asarray(candidates["sigma_index"], dtype=int)]
    exponent = np.asarray(grids["acceleration_exponent"])[
        np.asarray(candidates["exponent_index"], dtype=int)
    ]
    transition = np.asarray(grids["transition_u"])[
        np.asarray(candidates["transition_index"], dtype=int)
    ]
    radial = np.asarray(grids["radial_scale"])[
        np.asarray(candidates["radial_index"], dtype=int)
    ]
    return sigma, exponent, transition, radial


def stochastic_moments(
    candidates: Mapping[str, np.ndarray],
    u: np.ndarray,
    x: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return drift m and variance S of the one-cell natural-log acceleration increment."""

    u = np.asarray(u, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if u.ndim != 1 or x.shape != u.shape:
        raise GravityItem41Error("stochastic inputs must be aligned vectors")
    if np.any(~np.isfinite(u)) or np.any(u <= 0.0) or np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise GravityItem41Error("stochastic inputs are outside the frozen domain")
    lane = np.asarray(candidates["lane"], dtype=np.int8)
    sigma0, exponent, transition, radial = _candidate_parameters(candidates, config)
    uu = u[None, :]
    xx = x[None, :]
    sig = sigma0[:, None]
    qq = exponent[:, None]
    tt = transition[:, None]
    ll = radial[:, None]
    window = 1.0 / (1.0 + np.power(uu / tt, qq))
    drift = np.zeros((len(lane), len(u)), dtype=np.float64)
    variance = np.zeros_like(drift)
    for lane_id in range(4):
        mask = lane == lane_id
        if not np.any(mask):
            continue
        local_window = window[mask]
        local_sigma = sig[mask]
        if lane_id == 0:
            variance[mask] = np.square(local_sigma) * local_window
        elif lane_id == 1:
            saturation = 1.0 - np.exp(-xx / ll[mask])
            variance[mask] = np.square(local_sigma) * local_window * saturation
            drift[mask] = -0.5 * variance[mask]
        elif lane_id == 2:
            variance[mask] = np.square(local_sigma) * local_window
            drift[mask] = 0.5 * qq[mask] * variance[mask] * (1.0 - local_window)
        else:
            probability = 1.0 - np.exp(-xx / ll[mask])
            probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
            drift[mask] = local_sigma * local_window * (2.0 * probability - 1.0)
            variance[mask] = (
                4.0
                * np.square(local_sigma)
                * np.square(local_window)
                * probability
                * (1.0 - probability)
            )
    return drift, variance


def admissible_candidates(
    config: Mapping[str, Any], *, batch_size: int = 4096
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    gate = config["candidate_generator"]["admissibility"]
    u_grid = np.logspace(
        float(gate["probe_log10_u_min"]),
        float(gate["probe_log10_u_max"]),
        41,
    )
    x_values = np.asarray(gate["probe_x"], dtype=np.float64)
    u = np.tile(u_grid, len(x_values))
    x = np.repeat(x_values, len(u_grid))
    keep_parts: list[np.ndarray] = []
    rejection: Counter[str] = Counter()
    signatures: set[bytes] = set()
    for begin in range(0, len(raw["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(raw["candidate_id"]))
        rows = {key: value[begin:end] for key, value in raw.items()}
        drift, variance = stochastic_moments(rows, u, x, config)
        finite = np.all(np.isfinite(drift) & np.isfinite(variance), axis=1)
        bounded = finite & np.all(variance >= float(gate["minimum_variance"]), axis=1)
        bounded &= np.all(variance <= float(gate["maximum_variance"]), axis=1)
        local_u = np.full(4, float(gate["solar_u"]))
        local_x = np.asarray(gate["probe_x"], dtype=np.float64)
        local_drift, local_variance = stochastic_moments(rows, local_u, local_x, config)
        local = bounded & (
            np.max(np.abs(local_drift), axis=1) <= float(gate["maximum_local_abs_drift"])
        )
        local &= np.max(np.sqrt(local_variance), axis=1) <= float(
            gate["maximum_local_std"]
        )
        low_u = np.full(4, 1e-4)
        low_drift, low_variance = stochastic_moments(rows, low_u, local_x, config)
        material = local & (
            np.max(np.sqrt(low_variance), axis=1)
            >= float(gate["minimum_low_acceleration_std"])
        )
        span = np.ptp(np.log(np.maximum(variance, 1e-300)), axis=1)
        varying = material & (span >= float(gate["minimum_log_variance_span"]))
        rejection["nonfinite"] += int(np.sum(~finite))
        rejection["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection["local_limit"] += int(np.sum(bounded & ~local))
        rejection["immaterial_low_acceleration"] += int(np.sum(local & ~material))
        rejection["variance_immaterial"] += int(np.sum(material & ~varying))
        local_keep = np.flatnonzero(varying)
        keep_parts.append(local_keep + begin)
        signature = np.round(
            np.column_stack((drift[local_keep], np.log(variance[local_keep]))), 6
        )
        for row in signature:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
        del low_drift
    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    admitted = {key: value[keep] for key, value in raw.items()}
    return admitted, {
        "raw_candidates": len(raw["candidate_id"]),
        "admitted_candidates": len(keep),
        "rejected_candidates": len(raw["candidate_id"]) - len(keep),
        "admitted_by_lane": {
            str(lane): int(np.sum(admitted["lane"] == lane)) for lane in range(4)
        },
        "behavioral_equivalence_classes": len(signatures),
        "rejection_counts_nonexclusive": dict(sorted(rejection.items())),
    }


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem41Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    sigma, exponent, transition, radial = _candidate_parameters(row, config)
    lane = int(row["lane"][0])
    niche = config["candidate_generator"]["niches"][lane]
    return {
        "candidate_id": candidate_id,
        "lane_id": lane,
        "lane": niche["name"],
        "creativity_label": niche["creativity_label"],
        "law": niche["law"],
        "parameters": {
            "sigma0": float(sigma[0]),
            "acceleration_exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
            "radial_scale": float(radial[0]),
        },
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_candidate_manifest(root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config(root))
    admitted, audit = admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item41-candidate-manifest-1.0",
            "item": 41,
            "config_contract_sha256": _contract_digest(config),
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_accessed_during_generation": False,
            "confirmation_accessed": False,
            "paid_model_calls": 0,
            **audit,
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "niches": config["candidate_generator"]["niches"],
            "claim_boundaries": [
                "stochastic gravity and multiplicative noise are known idea families",
                "the observational synthesis does not establish historical novelty",
                "candidate generation read no response value despite retrospective data availability",
                "the closure predicts one-cell moments, not a complete stochastic process",
            ],
        }
    )


def build_exposure_manifest(root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config(root))
    response_path = root / str(config["data"]["ghasp_responses"])
    rows = _read_tsv(response_path)
    identities = sorted({str(row["identity"]) for row in rows})
    sample = _read_json(root / str(config["data"]["ghasp_sample"]))
    confirmations = sorted(
        str(row["identity"])
        for row in sample["objects"]
        if str(row["role"]) == "reserved_confirmation"
    )
    if len(identities) != int(config["data"]["ghasp_expected_usable_galaxies"]):
        raise GravityItem41Error("GHASP usable identity count changed")
    if len(rows) != int(config["data"]["ghasp_expected_paired_points"]):
        raise GravityItem41Error("GHASP paired point count changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item41-exposure-manifest-1.0",
            "item": 41,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "role": config["scope"]["ghasp_role"],
            "exposed_exploration_identities": identities,
            "sealed_item28_confirmation_identities": confirmations,
            "counts": {
                "retrospective_exploration_galaxies": len(identities),
                "paired_radial_points": len(rows),
                "sealed_confirmations": len(confirmations),
                "confirmation_values_read": 0,
                "response_values_read_while_generating_candidates": 0,
                "paid_model_calls": 0,
            },
            "rules": [
                "use only Item 28 exploration responses already exposed",
                "open no Item 28 confirmation value",
                "make no fresh-confirmation claim",
                "generate no formula after Item 41 evaluation begins",
            ],
        }
    )


def write_freeze(root: Path) -> dict[str, Path]:
    config = load_config(root)
    candidate = _source_path(root, config, "candidate_manifest")
    exposure = _source_path(root, config, "exposure_manifest")
    _write_json(candidate, build_candidate_manifest(root, config))
    _write_json(exposure, build_exposure_manifest(root, config))
    return {"candidate_manifest": candidate, "exposure_manifest": exposure}


def check_freeze(root: Path) -> dict[str, Any]:
    config = load_config(root)
    candidate = _source_path(root, config, "candidate_manifest")
    exposure = _source_path(root, config, "exposure_manifest")
    if _read_json(candidate) != build_candidate_manifest(root, config):
        raise GravityItem41Error("candidate manifest drifted")
    if _read_json(exposure) != build_exposure_manifest(root, config):
        raise GravityItem41Error("exposure manifest drifted")
    return {
        "status": "ITEM41_SCIENTIFIC_FREEZE_VALID",
        "candidate_manifest_sha256": _sha256_file(candidate),
        "exposure_manifest_sha256": _sha256_file(exposure),
        "confirmation_values_read": 0,
        "paid_model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check-freeze"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "freeze":
        result = {key: str(value) for key, value in write_freeze(args.root).items()}
    else:
        result = check_freeze(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
