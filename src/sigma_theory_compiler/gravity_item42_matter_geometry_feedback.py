"""Frozen Item 42 nonlinear matter-geometry feedback search machinery."""

from __future__ import annotations

import argparse
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
from sigma_theory_compiler.gravity_item39_holographic_boundary import (
    _deserialize_wallaby_profile,
    _parse_vector,
    _query_csv,
    _write_tsv,
)

CONFIG_PATH = Path("configs/gravity_item42_matter_geometry_feedback_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")


class GravityItem42Error(RuntimeError):
    """Raised when an Item 42 freeze, feedback law, or leakage gate fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item42-matter-geometry-feedback-config-1.0"
        or int(config.get("item", -1)) != 42
    ):
        raise GravityItem42Error("unexpected Item 42 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem42Error("stable gravity goal changed")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem42Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-41-stochastic-gravity-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem42Error("Item 41 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem42Error("Item 41 decision changed")
    policy = load_counterexample_policy(root / POLICY_PATH)
    empirical = policy["empirical_evidence"]
    discovery = config["discovery_policy"]
    if not bool(discovery["equal_initial_viability"]):
        raise GravityItem42Error("equal initial viability changed")
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem42Error("one empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem42Error("count-only empirical rejection entered Item 42")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem42Error("finite empirical family pruning entered Item 42")
    if empirical["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem42Error("executable counterexample policy changed")
    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem42Error("raw candidate count changed")
    if int(generator["cells_per_niche"]) != 65536:
        raise GravityItem42Error("per-niche capacity changed")
    if list(generator["grid_shape_per_niche"]) != [16, 16, 16, 16]:
        raise GravityItem42Error("candidate grid shape changed")
    if len(generator["niches"]) != 4 or int(generator["post_response_cells"]) != 0:
        raise GravityItem42Error("feedback niche or response-free boundary changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem42Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem42Error("paid calls entered Item 42")
    if config["weak_field_metric_contract"]["gravitational_slip"] != "Phi=Psi":
        raise GravityItem42Error("motion/light closure changed")
    if not bool(config["independence"]["exclude_every_item39_role"]):
        raise GravityItem42Error("Item 39 role exclusion changed")
    if not bool(config["independence"]["exclude_every_item40_role"]):
        raise GravityItem42Error("Item 40 role exclusion changed")
    transfer = config["cluster_transfer"]
    if bool(transfer["selection_use"]) or bool(transfer["retuning_allowed"]):
        raise GravityItem42Error("CLASH entered selection or retuning")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    for key in ("scientific_freeze_commit", "sample_freeze_commit"):
        value[key] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def generate_raw_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    total = int(config["candidate_generator"]["raw_candidate_cells"])
    cells = int(config["candidate_generator"]["cells_per_niche"])
    candidate_id = np.arange(total, dtype=np.int64)
    lane = (candidate_id // cells).astype(np.int8)
    local = candidate_id % cells
    return {
        "candidate_id": candidate_id,
        "lane": lane,
        "amplitude_index": ((local // 4096) % 16).astype(np.int8),
        "exponent_index": ((local // 256) % 16).astype(np.int8),
        "transition_index": ((local // 16) % 16).astype(np.int8),
        "feedback_index": (local % 16).astype(np.int8),
    }


def _candidate_parameters(
    candidates: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grids = config["candidate_generator"]["parameter_grids"]
    amplitude = np.asarray(grids["amplitude"])[
        np.asarray(candidates["amplitude_index"], dtype=int)
    ]
    exponent = np.asarray(grids["acceleration_exponent"])[
        np.asarray(candidates["exponent_index"], dtype=int)
    ]
    transition = np.asarray(grids["transition_u"])[
        np.asarray(candidates["transition_index"], dtype=int)
    ]
    feedback = np.asarray(grids["feedback_gain"])[
        np.asarray(candidates["feedback_index"], dtype=int)
    ]
    return amplitude, exponent, transition, feedback


def _normalize_positive(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    total = float(np.sum(value))
    if not math_is_finite_positive(total):
        raise GravityItem42Error("feedback source has no positive finite mass")
    return value / total


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def feedback_coordinate(
    radius: np.ndarray,
    cumulative_mass: np.ndarray,
    lane: int,
    feedback_gain: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    radius = np.asarray(radius, dtype=np.float64)
    cumulative_mass = np.asarray(cumulative_mass, dtype=np.float64)
    if radius.ndim != 1 or cumulative_mass.shape != radius.shape or len(radius) < 3:
        raise GravityItem42Error("feedback profile vectors are invalid")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0) or np.any(np.diff(radius) <= 0.0):
        raise GravityItem42Error("feedback radii must be finite, positive, and increasing")
    if np.any(~np.isfinite(cumulative_mass)) or np.any(np.diff(cumulative_mass) < -1e-9):
        raise GravityItem42Error("feedback cumulative mass must be finite and monotone")
    source = _normalize_positive(np.diff(np.concatenate(([0.0], cumulative_mass))))
    x = radius / radius[-1]
    fixed = config["candidate_generator"]["fixed_point"]
    length = float(fixed["kernel_length"])
    kernel = np.exp(-np.abs(x[:, None] - x[None, :]) / length)
    kernel /= np.sum(kernel, axis=1, keepdims=True)
    geometry = kernel @ source
    geometry /= max(float(np.max(geometry)), 1e-15)
    damping = float(fixed["damping"])
    tolerance = float(fixed["convergence_tolerance"])
    clip = float(fixed["exponent_clip"])
    maximum = int(fixed["maximum_iterations"])
    converged = False
    delta = np.inf
    for iteration in range(1, maximum + 1):
        if lane == 0:
            signal = geometry
        elif lane == 1:
            signal = -geometry
        elif lane == 2:
            gradient = np.abs(np.gradient(geometry, x))
            signal = np.tanh(length * gradient)
        elif lane == 3:
            signal = np.tanh((geometry - kernel @ geometry) / 0.1)
        else:
            raise GravityItem42Error("unknown feedback niche")
        exponent = np.clip(float(feedback_gain) * signal, -clip, clip)
        effective_source = _normalize_positive(source * np.exp(exponent))
        proposal = kernel @ effective_source
        proposal /= max(float(np.max(proposal)), 1e-15)
        updated = damping * geometry + (1.0 - damping) * proposal
        delta = float(np.max(np.abs(updated - geometry)))
        geometry = updated
        if delta <= tolerance:
            converged = True
            break
    return geometry, {
        "converged": converged,
        "iterations": iteration,
        "final_max_abs_change": delta,
    }


def feedback_library(
    radius: np.ndarray,
    cumulative_mass: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    gains = config["candidate_generator"]["parameter_grids"]["feedback_gain"]
    result = np.empty((4, len(gains), len(radius)), dtype=np.float64)
    audits: list[dict[str, Any]] = []
    for lane in range(4):
        for index, gain in enumerate(gains):
            coordinate, audit = feedback_coordinate(
                radius, cumulative_mass, lane, float(gain), config
            )
            result[lane, index] = coordinate
            audits.append({"lane": lane, "feedback_index": index, **audit})
    return result, {
        "all_converged": all(bool(row["converged"]) for row in audits),
        "maximum_iterations": max(int(row["iterations"]) for row in audits),
        "maximum_final_change": max(float(row["final_max_abs_change"]) for row in audits),
    }


def _probe_feedback(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = np.linspace(0.05, 1.0, 12)
    shells = [
        np.ones_like(x),
        np.exp(-4.0 * x),
        np.exp(4.0 * (x - 1.0)),
        0.15 + np.exp(-0.5 * np.square((x - 0.55) / 0.12)),
    ]
    libraries = []
    audits = []
    for shell in shells:
        library, audit = feedback_library(x, np.cumsum(shell), config)
        libraries.append(library)
        audits.append(audit)
    stacked = np.stack(libraries, axis=0)
    minimum = np.min(stacked, axis=(0, 3))
    maximum = np.max(stacked, axis=(0, 3))
    base = stacked[:, :, 0:1, :]
    change = np.sqrt(np.mean(np.square(stacked - base), axis=(0, 3)))
    return minimum, maximum, {
        "all_converged": all(bool(row["all_converged"]) for row in audits),
        "maximum_iterations": max(int(row["maximum_iterations"]) for row in audits),
        "coordinate_change": change,
    }


def admissible_candidates(
    config: Mapping[str, Any], *, batch_size: int = 4096
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw = generate_raw_candidates(config)
    minimum_h, maximum_h, feedback_audit = _probe_feedback(config)
    if not bool(feedback_audit["all_converged"]):
        raise GravityItem42Error("a frozen probe feedback system did not converge")
    gate = config["candidate_generator"]["admissibility"]
    u = np.logspace(
        float(gate["probe_log10_u_min"]),
        float(gate["probe_log10_u_max"]),
        int(gate["probe_points"]),
    )
    keep_parts: list[np.ndarray] = []
    rejection: Counter[str] = Counter()
    signatures: set[bytes] = set()
    for begin in range(0, len(raw["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(raw["candidate_id"]))
        rows = {key: value[begin:end] for key, value in raw.items()}
        amplitude, exponent, transition, _ = _candidate_parameters(rows, config)
        lane = np.asarray(rows["lane"], dtype=int)
        feedback_index = np.asarray(rows["feedback_index"], dtype=int)
        h_low = minimum_h[lane, feedback_index]
        h_high = maximum_h[lane, feedback_index]
        aa = amplitude[:, None, None]
        pp = exponent[:, None, None]
        tt = transition[:, None, None]
        uu = u[None, None, :]
        hh = np.stack((h_low, h_high), axis=1)[:, :, None]
        multiplier = 1.0 + aa * np.power(uu, -pp) / (1.0 + uu / tt) * (
            0.05 + 0.95 * hh
        )
        finite = np.all(np.isfinite(multiplier), axis=(1, 2))
        bounded = finite & np.all(
            multiplier >= float(gate["minimum_multiplier"]), axis=(1, 2)
        )
        bounded &= np.all(
            multiplier <= float(gate["maximum_multiplier"]), axis=(1, 2)
        )
        local = bounded & (
            np.max(np.log10(multiplier[:, :, -1]), axis=1)
            <= float(gate["maximum_high_acceleration_log10_deviation"])
        )
        material = local & (
            np.min(multiplier[:, :, 0], axis=1)
            >= float(gate["minimum_low_acceleration_multiplier"])
        )
        coordinate_change = feedback_audit["coordinate_change"][lane, feedback_index]
        feedback_material = material & (
            coordinate_change >= float(gate["minimum_feedback_coordinate_change"])
        )
        monotone = feedback_material & np.all(
            np.diff(multiplier, axis=2)
            <= float(gate["monotone_nonincreasing_tolerance"]),
            axis=(1, 2),
        )
        rejection["nonfinite"] += int(np.sum(~finite))
        rejection["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection["no_local_limit"] += int(np.sum(bounded & ~local))
        rejection["immaterial_low_acceleration"] += int(np.sum(local & ~material))
        rejection["feedback_immaterial"] += int(np.sum(material & ~feedback_material))
        rejection["nonmonotone"] += int(np.sum(feedback_material & ~monotone))
        local_keep = np.flatnonzero(monotone)
        keep_parts.append(local_keep + begin)
        signature = np.round(
            np.log10(multiplier[local_keep]),
            int(gate["behavior_signature_decimals"]),
        )
        for row in signature:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
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
        "probe_feedback_maximum_iterations": int(feedback_audit["maximum_iterations"]),
        "rejection_counts_nonexclusive": dict(sorted(rejection.items())),
    }


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem42Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition, feedback = _candidate_parameters(row, config)
    lane = int(row["lane"][0])
    niche = config["candidate_generator"]["niches"][lane]
    return {
        "candidate_id": candidate_id,
        "lane_id": lane,
        "lane": niche["name"],
        "creativity_label": niche["creativity_label"],
        "source_update": niche["source_update"],
        "parameters": {
            "amplitude": float(amplitude[0]),
            "acceleration_exponent": float(exponent[0]),
            "transition_u": float(transition[0]),
            "feedback_gain": float(feedback[0]),
        },
    }


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    admitted, audit = admissible_candidates(config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item42-candidate-manifest-1.0",
            "item": 42,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "response_accessed_during_generation": False,
            "confirmation_accessed": False,
            "paid_model_calls": 0,
            **audit,
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "niches": config["candidate_generator"]["niches"],
            "claim_boundaries": [
                "curvature-matter feedback and screening are known idea families",
                "the fixed-point observational synthesis does not establish historical novelty",
                "effective source reweighting is not a claim that observed matter physically relocated",
                "zero response values and zero paid model calls entered generation",
            ],
        }
    )


def build_exposure_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    rows: list[dict[str, Any]] = []
    for key, item in (("item39_sample", 39), ("item40_sample", 40)):
        path = root / str(config["data_sources"][key])
        sample = _read_json(path)
        for row in sample["objects"]:
            rows.append(
                {
                    "name": str(row["name"]),
                    "team_release_kin": str(row["team_release_kin"]),
                    "prior_item": item,
                    "prior_role": str(row["role"]),
                }
            )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item42-predecessor-exposure-1.0",
            "item": 42,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "excluded_identities": sorted(
                rows, key=lambda row: (row["name"], row["team_release_kin"], row["prior_item"])
            ),
            "counts": {
                "excluded_prior_role_records": len(rows),
                "response_values_read_while_building": 0,
                "confirmation_values_read": 0,
            },
            "rules": [
                "exclude every Item 39 and Item 40 exploration or confirmation identity",
                "use only response-blind HI profiles with a recorded Legacy optical-match failure",
                "never replace a sample object after response access",
            ],
        }
    )


def write_freeze_manifests(root: Path) -> dict[str, Path]:
    config = load_config(root)
    candidate = _source_path(root, config, "candidate_manifest")
    exposure = _source_path(root, config, "exposure_manifest")
    _write_json(candidate, build_candidate_manifest(root))
    _write_json(exposure, build_exposure_manifest(root))
    return {"candidate_manifest": candidate, "exposure_manifest": exposure}


def check_freeze(root: Path) -> dict[str, Any]:
    config = load_config(root)
    candidate = _source_path(root, config, "candidate_manifest")
    exposure = _source_path(root, config, "exposure_manifest")
    if _read_json(candidate) != build_candidate_manifest(root):
        raise GravityItem42Error("candidate manifest drifted")
    if _read_json(exposure) != build_exposure_manifest(root):
        raise GravityItem42Error("exposure manifest drifted")
    return {
        "status": "ITEM42_SCIENTIFIC_FREEZE_VALID",
        "candidate_manifest_sha256": _sha256_file(candidate),
        "exposure_manifest_sha256": _sha256_file(exposure),
        "response_rows_read": 0,
        "confirmation_rows_read": 0,
        "paid_model_calls": 0,
    }


def _profile_quality_reasons(
    raw: Mapping[str, Any], profile: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
    quality = config["predictor_quality"]
    reasons: list[str] = []
    radius = np.asarray(profile["radius_kpc"], dtype=np.float64)
    cumulative = np.asarray(profile["cumulative_hi_mass_msun"], dtype=np.float64)
    if float(raw["source_qflag"]) != float(quality["required_source_qflag"]):
        reasons.append("source_qflag")
    if len(radius) < int(quality["minimum_profile_nodes"]):
        reasons.append("insufficient_profile_nodes")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0) or np.any(np.diff(radius) <= 0.0):
        reasons.append("invalid_radius")
    if np.any(~np.isfinite(cumulative)) or np.any(cumulative <= 0.0):
        reasons.append("invalid_cumulative_hi_mass")
    if np.any(np.diff(cumulative) < -1e-9):
        reasons.append("nonmonotone_cumulative_hi_mass")
    return reasons


def build_predictor_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    if str(config["scientific_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem42Error("scientific freeze is not commit-bound")
    wallaby_path = root / str(config["data_sources"]["wallaby_predictors"])
    optical_path = root / str(config["data_sources"]["legacy_predictors"])
    wallaby = _read_json(wallaby_path)
    optical = _read_json(optical_path)
    failures = {
        (str(row["galaxy"]), str(row["team_release_kin"]))
        for row in optical["failures"]
    }
    used: set[tuple[str, str]] = set()
    for key in ("item39_sample", "item40_sample"):
        sample = _read_json(root / str(config["data_sources"][key]))
        used.update(
            (str(row["name"]), str(row["team_release_kin"]))
            for row in sample["objects"]
        )
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, raw in enumerate(wallaby["records"]):
        key = (str(raw["name"]), str(raw["team_release_kin"]))
        if key not in failures or key in used:
            continue
        profile = _deserialize_wallaby_profile(raw)
        reasons = _profile_quality_reasons(raw, profile, config)
        record = {
            "name": key[0],
            "team_release_kin": key[1],
            "wallaby_record_index": index,
            "profile_nodes": len(profile["radius_kpc"]),
            "source_qflag": raw["source_qflag"],
        }
        if reasons:
            rejected.append({**record, "reasons": reasons})
        else:
            eligible.append(record)
    eligible.sort(key=lambda row: (row["name"], row["team_release_kin"]))
    rejected.sort(key=lambda row: (row["name"], row["team_release_kin"]))
    expected = int(config["independence"]["expected_response_blind_quality_eligible"])
    if len(eligible) != expected:
        raise GravityItem42Error("Item 42 response-blind eligible count changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item42-predictor-receipt-1.0",
            "item": 42,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "wallaby_predictor_sha256": _sha256_file(wallaby_path),
            "legacy_predictor_sha256": _sha256_file(optical_path),
            "eligible": eligible,
            "rejected": rejected,
            "counts": {
                "optical_match_failure_profiles": len(eligible) + len(rejected),
                "quality_eligible": len(eligible),
                "quality_rejected": len(rejected),
                "response_rows_read": 0,
                "confirmation_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "all_values_are_response_blind_predictors": True,
                "missing_stellar_counterparts_are_disclosed": True,
                "historical_novelty_established": False,
            },
        }
    )


def write_predictor_receipt(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "predictor_receipt")
    _write_json(path, build_predictor_receipt(root))
    return path


def build_sample_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    receipt_path = _source_path(root, config, "predictor_receipt")
    receipt = _read_json(receipt_path)
    if receipt != build_predictor_receipt(root):
        raise GravityItem42Error("predictor receipt drifted")
    wallaby = _read_json(root / str(config["data_sources"]["wallaby_predictors"]))
    salt = str(config["sample_boundary"]["split_salt"])
    confirmation_count = int(config["sample_boundary"]["reserved_confirmation_galaxies"])
    ranked = sorted(
        receipt["eligible"],
        key=lambda row: _split_hash(
            f"{row['name']}|{row['team_release_kin']}", salt
        ),
    )
    confirmations = {
        (str(row["name"]), str(row["team_release_kin"]))
        for row in ranked[:confirmation_count]
    }
    fold_salt = str(config["sample_boundary"]["fold_salt"])
    folds = int(config["sample_boundary"]["outer_folds"])
    objects: list[dict[str, Any]] = []
    for identity in receipt["eligible"]:
        raw = wallaby["records"][int(identity["wallaby_record_index"])]
        profile = _deserialize_wallaby_profile(raw)
        library, audit = feedback_library(
            profile["radius_kpc"], profile["cumulative_hi_mass_msun"], config
        )
        if not bool(audit["all_converged"]):
            raise GravityItem42Error("eligible predictor feedback failed to converge")
        cumulative = profile["cumulative_hi_mass_msun"]
        half_index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
        key = (str(identity["name"]), str(identity["team_release_kin"]))
        value = f"{key[0]}|{key[1]}"
        objects.append(
            {
                **identity,
                "ra": raw["ra"],
                "dec": raw["dec"],
                "distance_mpc": raw["distance_mpc"],
                "hi_mass_msun": raw["hi_mass_msun"],
                "screen_radius_kpc": raw["screen_radius_kpc"],
                "half_hi_mass_radius_fraction": f"{profile['radius_kpc'][half_index] / profile['screen_radius_kpc']:.12e}",
                "maximum_feedback_iterations": audit["maximum_iterations"],
                "feedback_library_cells": int(np.prod(library.shape[:2])),
                "outer_fold": int(_split_hash(value, fold_salt), 16) % folds,
                "role": "reserved_confirmation" if key in confirmations else "exploration",
                "response_read": False,
                "missing_stellar_counterpart": True,
            }
        )
    objects.sort(key=lambda row: (row["name"], row["team_release_kin"]))
    counts = Counter(str(row["role"]) for row in objects)
    if counts["exploration"] != int(
        config["sample_boundary"]["expected_exploration_galaxies"]
    ):
        raise GravityItem42Error("Item 42 exploration count changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item42-sample-manifest-1.0",
            "item": 42,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "predictor_receipt_sha256": _sha256_file(receipt_path),
            "objects": objects,
            "counts": {
                "selected_total": len(objects),
                "exploration": counts["exploration"],
                "reserved_confirmation": counts["reserved_confirmation"],
                "response_rows_read": 0,
                "confirmation_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "response_opened": False,
                "confirmation_opened": False,
                "failed_identity_replacement": False,
                "stellar_counterparts_missing": True,
            },
        }
    )


def write_sample_manifest(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "sample_manifest")
    _write_json(path, build_sample_manifest(root))
    return path


def write_wallaby_response_source(root: Path) -> Path:
    config = load_config(root)
    if str(config["sample_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem42Error("sample freeze is not commit-bound")
    sample_path = _source_path(root, config, "sample_manifest")
    sample = _read_json(sample_path)
    if sample != build_sample_manifest(root):
        raise GravityItem42Error("sample manifest drifted")
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    confirmations = {
        (str(row["name"]), str(row["team_release_kin"]))
        for row in sample["objects"]
        if row["role"] == "reserved_confirmation"
    }
    conditions = []
    for row in exploration:
        name = str(row["name"]).replace("'", "''")
        release = str(row["team_release_kin"]).replace("'", "''")
        conditions.append(f"(name='{name}' AND team_release_kin='{release}')")
    columns = ",".join(config["data_sources"]["response_columns"])
    query = (
        f"SELECT {columns} FROM {config['data_sources']['wallaby_kinematic_table']} WHERE "
        + " OR ".join(conditions)
        + " ORDER BY name,team_release_kin"
    )
    payload, rows = _query_csv(
        str(config["data_sources"]["wallaby_tap_sync_endpoint"]),
        query,
        dialect="tap_adql",
        user_agent="Invariant/Item42-WALLABY-Feedback-Exploration-Responses",
    )
    expected_columns = set(config["data_sources"]["response_columns"])
    if any(set(row) != expected_columns for row in rows):
        raise GravityItem42Error("WALLABY response schema changed")
    expected = {(str(row["name"]), str(row["team_release_kin"])) for row in exploration}
    returned = {(str(row["name"]), str(row["team_release_kin"])) for row in rows}
    if returned != expected or len(rows) != len(returned) or returned & confirmations:
        raise GravityItem42Error("WALLABY response scope changed")
    path = _source_path(root, config, "wallaby_response_source")
    _write_json(
        path,
        _content_hashed(
            {
                "schema_version": "invariant-gravity-item42-wallaby-exploration-response-1.0",
                "item": 42,
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "sample_manifest_sha256": _sha256_file(sample_path),
                "query_identity_count": len(expected),
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "records": rows,
                "counts": {
                    "exploration_response_rows": len(rows),
                    "confirmation_response_rows": 0,
                    "post_response_candidate_cells": 0,
                    "paid_model_calls": 0,
                },
                "claims": {
                    "confirmation_opened": False,
                    "response_scope_repaired_after_access": False,
                },
            }
        ),
    )
    return path


def extract_wallaby_profiles(root: Path) -> dict[str, Path]:
    config = load_config(root)
    wallaby = _read_json(root / str(config["data_sources"]["wallaby_predictors"]))
    sample_path = _source_path(root, config, "sample_manifest")
    response_path = _source_path(root, config, "wallaby_response_source")
    sample = _read_json(sample_path)
    response = _read_json(response_path)
    if sample != build_sample_manifest(root):
        raise GravityItem42Error("sample manifest drifted")
    if response["sample_freeze_commit"] != config["sample_freeze_commit"]:
        raise GravityItem42Error("response/sample binding changed")
    if int(response["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem42Error("confirmation response entered Item 42")
    profiles = {
        (str(row["name"]), str(row["team_release_kin"])): _deserialize_wallaby_profile(row)
        for row in wallaby["records"]
    }
    samples = {
        (str(row["name"]), str(row["team_release_kin"])): row
        for row in sample["objects"]
        if row["role"] == "exploration"
    }
    feature_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    galaxy_receipts: list[dict[str, Any]] = []
    quality = config["response_quality"]
    constants = config["constants"]
    acceleration_conversion = 1.0e6 / 3.085677581491367e19
    h_fields = [f"h_lane{lane}_feedback{gain}" for lane in range(4) for gain in range(16)]
    for raw in response["records"]:
        key = (str(raw["name"]), str(raw["team_release_kin"]))
        if key not in samples or key not in profiles:
            raise GravityItem42Error(f"response lacks frozen predictors: {key}")
        profile = profiles[key]
        sample_row = samples[key]
        reasons: list[str] = []
        try:
            inclination = float(raw["Inc_model"])
            qflag = float(raw["QFlag_model"])
            radius_arcsec = _parse_vector(raw["Rad"])
            velocity = _parse_vector(raw["Vrot_model"])
            velocity_error = _parse_vector(raw["e_Vrot_model"])
            inclination_error = _parse_vector(raw["e_Vrot_model_inc"])
            if not (
                len(radius_arcsec)
                == len(velocity)
                == len(velocity_error)
                == len(inclination_error)
            ):
                raise GravityItem42Error("rotation response vector lengths differ")
        except (ValueError, GravityItem42Error) as exc:
            galaxy_receipts.append(
                {
                    "name": key[0],
                    "team_release_kin": key[1],
                    "quality_pass": False,
                    "quality_failure_reasons": [f"parser:{exc}"],
                    "raw_rotation_points": 0,
                    "accepted_rotation_points": 0,
                    "missing_stellar_counterpart": True,
                }
            )
            continue
        if qflag != float(quality["required_qflag_model"]):
            reasons.append("qflag")
        if not (
            float(quality["minimum_inclination_degrees"])
            <= inclination
            <= float(quality["maximum_inclination_degrees"])
        ):
            reasons.append("inclination")
        radius_kpc = (
            radius_arcsec
            * profile["distance_mpc"]
            * 1000.0
            / float(constants["arcseconds_per_radian"])
        )
        total_error = np.sqrt(
            np.maximum(velocity_error, 0.0) ** 2
            + np.maximum(inclination_error, 0.0) ** 2
        )
        finite = (
            np.isfinite(radius_kpc)
            & np.isfinite(velocity)
            & np.isfinite(total_error)
        )
        within = finite & (radius_kpc > 0.0) & (
            radius_kpc <= profile["screen_radius_kpc"]
        )
        valid = within & (velocity >= float(quality["minimum_speed_km_s"]))
        valid &= total_error / np.maximum(velocity, 1e-12) <= float(
            quality["maximum_fractional_speed_error"]
        )
        if int(np.sum(valid)) < int(quality["minimum_rotation_points"]):
            reasons.append("insufficient_rotation_points")
        if len(within) == 0 or float(np.mean(within)) < float(
            quality["minimum_fraction_response_radii_within_source"]
        ):
            reasons.append("source_overlap")
        passed = not reasons
        accepted_count = int(np.sum(valid)) if passed else 0
        if passed:
            indices = np.flatnonzero(valid)
            accepted_radius = radius_kpc[indices]
            library, audit = feedback_library(
                profile["radius_kpc"], profile["cumulative_hi_mass_msun"], config
            )
            if not bool(audit["all_converged"]):
                raise GravityItem42Error("response-blind feedback library stopped converging")
            interpolated = np.empty((4, 16, len(indices)), dtype=np.float64)
            for lane in range(4):
                for gain in range(16):
                    interpolated[lane, gain] = np.interp(
                        accepted_radius,
                        profile["radius_kpc"],
                        library[lane, gain],
                        left=library[lane, gain, 0],
                        right=library[lane, gain, -1],
                    )
            gas_enclosed = np.interp(
                accepted_radius,
                profile["radius_kpc"],
                profile["cumulative_hi_mass_msun"],
                left=0.0,
                right=profile["hi_mass_msun"],
            ) * float(constants["helium_mass_factor"])
            local_sigma = np.interp(
                accepted_radius,
                profile["radius_kpc"],
                profile["surface_density_hi_msun_pc2"],
            )
            g_constant = float(constants["gravitational_constant_kpc_km2_s2_msun"])
            gbar = g_constant * gas_enclosed / np.square(accepted_radius)
            gbar_m_s2 = gbar * acceleration_conversion
            u = gbar_m_s2 / float(constants["acceleration_scale_m_s2"])
            vbar = np.sqrt(g_constant * gas_enclosed / accepted_radius)
            for output_index, source_index in enumerate(indices):
                feature = {
                    "galaxy": key[0],
                    "team_release_kin": key[1],
                    "point_index": output_index,
                    "outer_fold": sample_row["outer_fold"],
                    "radius_kpc": f"{accepted_radius[output_index]:.12e}",
                    "radius_over_source": f"{accepted_radius[output_index] / profile['screen_radius_kpc']:.12e}",
                    "local_hi_surface_density": f"{local_sigma[output_index]:.12e}",
                    "enclosed_gas_mass_msun": f"{gas_enclosed[output_index]:.12e}",
                    "gbar_m_s2": f"{gbar_m_s2[output_index]:.12e}",
                    "u": f"{u[output_index]:.12e}",
                    "vbar_km_s": f"{vbar[output_index]:.12e}",
                    "total_hi_mass_msun": sample_row["hi_mass_msun"],
                    "half_hi_mass_radius_fraction": sample_row[
                        "half_hi_mass_radius_fraction"
                    ],
                    "distance_mpc": sample_row["distance_mpc"],
                    "inclination_degrees": f"{inclination:.12e}",
                    "missing_stellar_counterpart": True,
                }
                for lane in range(4):
                    for gain in range(16):
                        feature[f"h_lane{lane}_feedback{gain}"] = (
                            f"{interpolated[lane, gain, output_index]:.12e}"
                        )
                feature_rows.append(feature)
                response_rows.append(
                    {
                        "galaxy": key[0],
                        "team_release_kin": key[1],
                        "point_index": output_index,
                        "observed_speed_km_s": f"{velocity[source_index]:.12e}",
                        "observed_speed_error_km_s": f"{total_error[source_index]:.12e}",
                    }
                )
        galaxy_receipts.append(
            {
                "name": key[0],
                "team_release_kin": key[1],
                "outer_fold": sample_row["outer_fold"],
                "quality_pass": passed,
                "quality_failure_reasons": reasons,
                "raw_rotation_points": len(radius_arcsec),
                "accepted_rotation_points": accepted_count,
                "inclination_degrees": f"{inclination:.12e}",
                "qflag_model": f"{qflag:.12e}",
                "missing_stellar_counterpart": True,
            }
        )
    feature_fields = [
        "galaxy",
        "team_release_kin",
        "point_index",
        "outer_fold",
        "radius_kpc",
        "radius_over_source",
        "local_hi_surface_density",
        "enclosed_gas_mass_msun",
        "gbar_m_s2",
        "u",
        "vbar_km_s",
        *h_fields,
        "total_hi_mass_msun",
        "half_hi_mass_radius_fraction",
        "distance_mpc",
        "inclination_degrees",
        "missing_stellar_counterpart",
    ]
    response_fields = [
        "galaxy",
        "team_release_kin",
        "point_index",
        "observed_speed_km_s",
        "observed_speed_error_km_s",
    ]
    feature_path = _source_path(root, config, "point_features")
    rotation_path = _source_path(root, config, "rotation_responses")
    _write_tsv(feature_path, feature_fields, feature_rows)
    _write_tsv(rotation_path, response_fields, response_rows)
    passing = sum(bool(row["quality_pass"]) for row in galaxy_receipts)
    summary_path = _source_path(root, config, "extraction_summary")
    _write_json(
        summary_path,
        _content_hashed(
            {
                "schema_version": "invariant-gravity-item42-extraction-summary-1.0",
                "item": 42,
                "sample_freeze_commit": config["sample_freeze_commit"],
                "sample_manifest_sha256": _sha256_file(sample_path),
                "response_source_sha256": _sha256_file(response_path),
                "point_features_sha256": _sha256_file(feature_path),
                "rotation_responses_sha256": _sha256_file(rotation_path),
                "galaxies": galaxy_receipts,
                "counts": {
                    "exploration_response_rows": len(response["records"]),
                    "confirmation_response_rows": 0,
                    "quality_passing_galaxies": passing,
                    "quality_failing_galaxies": len(galaxy_receipts) - passing,
                    "accepted_rotation_points": len(feature_rows),
                    "post_response_candidate_cells": 0,
                    "paid_model_calls": 0,
                },
                "quality": quality,
                "claims": {
                    "feedback_features_used_rotation_response": False,
                    "confirmation_opened": False,
                    "stellar_counterparts_missing": True,
                    "quality_failures_preserved_without_replacement": True,
                },
            }
        ),
    )
    return {
        "point_features": feature_path,
        "rotation_responses": rotation_path,
        "extraction_summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "check-freeze", "predictors", "sample", "responses", "extract"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "freeze":
        print(
            json.dumps(
                {key: str(value) for key, value in write_freeze_manifests(args.root).items()},
                sort_keys=True,
            )
        )
    elif args.command == "check-freeze":
        print(json.dumps(check_freeze(args.root), sort_keys=True))
    elif args.command == "predictors":
        print(write_predictor_receipt(args.root))
    elif args.command == "sample":
        print(write_sample_manifest(args.root))
    elif args.command == "responses":
        print(write_wallaby_response_source(args.root))
    else:
        print(
            json.dumps(
                {key: str(value) for key, value in extract_wallaby_profiles(args.root).items()},
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
