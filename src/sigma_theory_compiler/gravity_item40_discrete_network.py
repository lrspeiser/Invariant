"""Frozen Item 40 discrete/network-gravity search machinery."""

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
    _candidate_parameters,
    _deserialize_wallaby_profile,
    _parse_vector,
    _predictor_quality_reasons,
    _query_csv,
    _stellar_enclosed_mass,
    _write_tsv,
    boundary_coordinates,
    generate_raw_candidates,
)
from sigma_theory_compiler.gravity_item39_holographic_boundary import (
    load_config as load_item39_config,
)

CONFIG_PATH = Path("configs/gravity_item40_discrete_network_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")


class GravityItem40Error(RuntimeError):
    """Raised when an Item 40 freeze, leakage boundary, or replay invariant fails."""


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config)
    return config


def validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item40-discrete-network-config-1.0"
        or int(config.get("item", -1)) != 40
    ):
        raise GravityItem40Error("unexpected Item 40 config")
    if _sha256_file(root / GOAL_PATH) != str(config["stable_goal_sha256"]):
        raise GravityItem40Error("stable gravity goal changed")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise GravityItem40Error(f"scientific dependency changed: {relative}")
    predecessor = _read_json(root / "runs/gravity/roadmap/item-39-holographic-boundary-v1.json")
    required = config["required_predecessor"]
    if predecessor.get("content_sha256") != required["content_sha256"]:
        raise GravityItem40Error("Item 39 content binding changed")
    if predecessor.get("decision") != required["decision"]:
        raise GravityItem40Error("Item 39 decision changed")

    policy = load_counterexample_policy(root / POLICY_PATH)
    empirical = policy["empirical_evidence"]
    discovery = config["discovery_policy"]
    if not bool(discovery["equal_initial_viability"]):
        raise GravityItem40Error("equal initial viability changed")
    if not bool(discovery["single_empirical_counterexample_is_not_a_formula_or_family_veto"]):
        raise GravityItem40Error("single empirical mismatch became a veto")
    if not bool(discovery["counterexample_count_alone_is_never_decisive"]):
        raise GravityItem40Error("count-only empirical rejection entered Item 40")
    if bool(discovery["finite_empirical_sample_may_prune_family"]):
        raise GravityItem40Error("finite empirical family pruning entered Item 40")
    if empirical["single_counterexample_terminal_rejection_allowed"] is not False:
        raise GravityItem40Error("executable counterexample policy changed")

    generator = config["candidate_generator"]
    if int(generator["raw_candidate_cells"]) != 262144:
        raise GravityItem40Error("raw candidate count changed")
    if int(generator["cells_per_niche"]) != 65536:
        raise GravityItem40Error("per-niche capacity changed")
    if list(generator["grid_shape_per_niche"]) != [16, 16, 16, 16]:
        raise GravityItem40Error("candidate grid shape changed")
    if len(generator["niches"]) != 4 or int(generator["post_response_cells"]) != 0:
        raise GravityItem40Error("Item 40 niche or response-free generation boundary changed")
    if bool(config["scope"]["confirmation_opening_authorized"]):
        raise GravityItem40Error("confirmation opening is not authorized")
    if bool(config["scope"]["paid_api_calls_authorized"]):
        raise GravityItem40Error("paid calls entered Item 40")
    if config["weak_field_metric_contract"]["gravitational_slip"] != "Phi=Psi":
        raise GravityItem40Error("motion/light closure changed")
    if not bool(config["independence"]["exclude_every_item39_role"]):
        raise GravityItem40Error("Item 39 role exclusion changed")
    transfer = config["cluster_transfer"]
    if bool(transfer["selection_use"]) or bool(transfer["retuning_allowed"]):
        raise GravityItem40Error("CLASH entered selection or retuning")
    if int(transfer["post_selection_candidate_cells"]) != 0:
        raise GravityItem40Error("post-selection cluster candidates entered Item 40")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    for key in ("scientific_freeze_commit", "predictor_freeze_commit", "sample_freeze_commit"):
        value[key] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def graph_coordinates(radius: np.ndarray, cumulative_mass: np.ndarray) -> np.ndarray:
    """Return four response-blind graph coordinates on a radial baryonic network."""

    radius = np.asarray(radius, dtype=np.float64)
    cumulative = np.asarray(cumulative_mass, dtype=np.float64)
    if radius.ndim != 1 or cumulative.shape != radius.shape or len(radius) < 3:
        raise GravityItem40Error("graph profile must contain aligned radial vectors")
    if (
        np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(cumulative))
        or np.any(np.diff(radius) <= 0.0)
        or np.any(np.diff(cumulative) < -1e-8 * max(float(cumulative[-1]), 1.0))
        or cumulative[-1] <= 0.0
    ):
        raise GravityItem40Error("graph profile is outside the frozen domain")
    x = radius / radius[-1]
    shell = np.diff(np.concatenate(([0.0], cumulative)))
    shell = np.maximum(shell, 1e-12 * cumulative[-1])
    q = shell / np.sum(shell)
    n = len(radius)

    path = np.zeros((n, n), dtype=np.float64)
    conductance = np.sqrt(q[:-1] * q[1:]) / np.maximum(np.diff(x), 1e-6)
    path[np.arange(n - 1), np.arange(1, n)] = conductance
    path += path.T
    degree = np.sum(path, axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-15))
    laplacian = np.eye(n) - inv_sqrt[:, None] * path * inv_sqrt[None, :]
    eigenvalue, eigenvector = np.linalg.eigh(laplacian)

    fiedler = eigenvector[:, 1]
    fiedler -= float(np.sum(q * fiedler))
    spectral = np.abs(fiedler)
    spectral /= max(float(np.max(spectral)), 1e-15)

    resistance = np.concatenate(([0.0], np.cumsum(1.0 / np.maximum(conductance, 1e-15))))
    positive = resistance[resistance > 0.0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    resistance = 1.0 - np.exp(-resistance / max(scale, 1e-15))
    resistance /= max(float(np.max(resistance)), 1e-15)

    heat = (np.square(eigenvector) * np.exp(-eigenvalue)[None, :]).sum(axis=1)
    heat = (heat - float(np.min(heat))) / max(float(np.ptp(heat)), 1e-15)

    delta = np.abs(x[:, None] - x[None, :])
    nonlocal_weight = np.sqrt(q[:, None] * q[None, :]) * np.exp(-delta / 0.2)
    np.fill_diagonal(nonlocal_weight, 0.0)
    nl_value, nl_vector = np.linalg.eigh(nonlocal_weight)
    scale_nl = max(float(np.max(np.abs(nl_value))), 1e-15)
    communicability_matrix = (nl_vector * np.exp(nl_value / scale_nl)[None, :]) @ nl_vector.T
    communicability = np.abs(communicability_matrix[:, 0])
    communicability = (communicability - float(np.min(communicability))) / max(
        float(np.ptp(communicability)), 1e-15
    )
    result = np.clip(
        np.stack((spectral, resistance, heat, communicability), axis=0), 0.0, 1.0
    )
    if np.any(~np.isfinite(result)):
        raise GravityItem40Error("graph coordinate construction produced nonfinite values")
    return result


def predict_multiplier(
    candidates: Mapping[str, np.ndarray],
    u: np.ndarray,
    graph_feature: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Evaluate the four frozen graph-law niches."""

    u = np.asarray(u, dtype=np.float64)
    features = np.asarray(graph_feature, dtype=np.float64)
    if u.ndim != 1 or features.shape != (4, len(u)):
        raise GravityItem40Error("graph multiplier inputs are not aligned")
    if np.any(~np.isfinite(u)) or np.any(u <= 0.0) or np.any(~np.isfinite(features)):
        raise GravityItem40Error("graph multiplier inputs must be finite")
    if np.any((features < 0.0) | (features > 1.0)):
        raise GravityItem40Error("graph features must be bounded")
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
        envelope = aa * np.power(uu, -pp) * np.power(
            1.0 + np.power(uu / tt, ss), -1.0 / ss
        )
        boundary = 0.05 + 0.95 * np.power(features[lane_id][None, :], ss)
        result[mask] = 1.0 + envelope * boundary
    return result


def fixed_control_multiplier(name: str, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if name == "baryonic_newton":
        return np.ones_like(u)
    if name == "mond_RAR":
        return 1.0 / (-np.expm1(-np.sqrt(u)))
    raise GravityItem40Error(f"unknown fixed control: {name}")


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
    probe = np.vstack(
        [
            np.linspace(0.1, 0.9, len(u)),
            np.linspace(0.9, 0.1, len(u)),
            0.5 + 0.4 * np.sin(np.linspace(0.0, np.pi, len(u))),
            0.5 + 0.4 * np.cos(np.linspace(0.0, np.pi, len(u))),
        ]
    )
    constant_probe = np.full_like(probe, 0.6)
    keep_parts: list[np.ndarray] = []
    rejection_counts: Counter[str] = Counter()
    signatures: set[bytes] = set()
    decimals = int(admission["behavior_signature_decimals"])
    total = len(raw["candidate_id"])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        rows = {key: value[start:stop] for key, value in raw.items()}
        values = predict_multiplier(rows, u, probe, config)
        monotone_values = predict_multiplier(rows, u, constant_probe, config)
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
            np.diff(monotone_values, axis=1)
            <= float(admission["monotone_nonincreasing_tolerance"]),
            axis=1,
        )
        low = np.tile(np.asarray([0.05, 0.95]), (4, 1))
        graph_span = np.abs(
            np.diff(
                np.log10(
                    predict_multiplier(
                        rows, np.asarray([0.01, 0.01]), low, config
                    )
                ),
                axis=1,
            )[:, 0]
        )
        graph_material = monotone & (
            graph_span >= float(admission["minimum_graph_log10_span"])
        )
        rejection_counts["nonfinite"] += int(np.sum(~finite))
        rejection_counts["out_of_bounds"] += int(np.sum(finite & ~bounded))
        rejection_counts["no_local_limit"] += int(np.sum(bounded & ~local))
        rejection_counts["immaterial_low_acceleration"] += int(np.sum(local & ~material))
        rejection_counts["nonmonotone"] += int(np.sum(material & ~monotone))
        rejection_counts["graph_immaterial"] += int(np.sum(monotone & ~graph_material))
        admitted_local = np.flatnonzero(graph_material)
        keep_parts.append(admitted_local + start)
        rounded = np.round(np.log10(values[admitted_local]), decimals=decimals)
        for row in rounded:
            signatures.add(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    keep = np.concatenate(keep_parts) if keep_parts else np.empty(0, dtype=np.int64)
    admitted = {key: value[keep] for key, value in raw.items()}
    return admitted, {
        "raw_candidates": total,
        "admitted_candidates": len(keep),
        "rejected_candidates": int(total - len(keep)),
        "admitted_by_lane": {
            str(lane): int(np.sum(admitted["lane"] == lane)) for lane in range(4)
        },
        "behavioral_equivalence_classes": len(signatures),
        "rejection_counts_nonexclusive": dict(sorted(rejection_counts.items())),
    }


def decode_candidate(candidate_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    raw = generate_raw_candidates(config)
    if candidate_id < 0 or candidate_id >= len(raw["candidate_id"]):
        raise GravityItem40Error("candidate id outside frozen grid")
    row = {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}
    amplitude, exponent, transition, shape = _candidate_parameters(row, config)
    lane_id = int(row["lane"][0])
    niche = config["candidate_generator"]["niches"][lane_id]
    return {
        "candidate_id": candidate_id,
        "lane_id": lane_id,
        "lane": niche["name"],
        "creativity_label": niche["creativity_label"],
        "graph_coordinate": niche["graph_coordinate"],
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
            "schema_version": "invariant-gravity-item40-candidate-manifest-1.0",
            "item": 40,
            "config_contract_sha256": _contract_digest(config),
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "response_accessed": False,
            "confirmation_accessed": False,
            "paid_api_calls": 0,
            **audit,
            "candidate_id_sha256": _sha256_bytes(
                np.asarray(admitted["candidate_id"], dtype="<i8").tobytes()
            ),
            "niches": config["candidate_generator"]["niches"],
            "claim_boundaries": [
                "graph, lattice, resistance, and heat-kernel constructions are known prior art",
                "the baryonic radial-network synthesis may be new but historical novelty is not established",
                "the weak-field multiplier is not a covariant gravity theory",
                "zero responses and zero paid model calls entered generation",
            ],
        }
    )


def build_exposure_manifest(root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or load_config(root))
    sample_path = root / str(config["data_sources"]["item39_sample"])
    sample = _read_json(sample_path)
    identities = sorted(
        (str(row["name"]), str(row["team_release_kin"]), str(row["role"]))
        for row in sample["objects"]
    )
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item40-predecessor-exposure-1.0",
            "item": 40,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "item39_sample_sha256": _sha256_file(sample_path),
            "excluded_identities": [
                {"name": name, "team_release_kin": release, "prior_role": role}
                for name, release, role in identities
            ],
            "counts": {
                "excluded_item39_identities": len(identities),
                "response_values_read_while_building": 0,
                "confirmation_values_read": 0,
            },
            "rules": [
                "exclude all Item 39 exploration identities",
                "exclude all Item 39 reserved-confirmation identities",
                "retain only predictor records with no prior response access",
                "never replace a sample object after response access",
            ],
        }
    )


def write_freeze_manifests(root: Path) -> dict[str, Path]:
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
        raise GravityItem40Error("candidate manifest drifted")
    if _read_json(exposure) != build_exposure_manifest(root, config):
        raise GravityItem40Error("exposure manifest drifted")
    return {
        "status": "ITEM40_SCIENTIFIC_FREEZE_VALID",
        "candidate_manifest_sha256": _sha256_file(candidate),
        "exposure_manifest_sha256": _sha256_file(exposure),
        "response_rows_read": 0,
        "confirmation_rows_read": 0,
        "paid_api_calls": 0,
    }


def build_predictor_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    if str(config["scientific_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem40Error("scientific freeze is not commit-bound")
    item39_config = load_item39_config(root)
    wallaby_path = root / str(config["data_sources"]["wallaby_predictors"])
    optical_path = root / str(config["data_sources"]["legacy_predictors"])
    sample_path = root / str(config["data_sources"]["item39_sample"])
    wallaby = _read_json(wallaby_path)
    optical = _read_json(optical_path)
    sample = _read_json(sample_path)
    used = {(str(row["name"]), str(row["team_release_kin"])) for row in sample["objects"]}
    optical_by_key = {
        (str(row["galaxy"]), str(row["team_release_kin"])): row
        for row in optical["records"]
    }
    unused = [
        row
        for row in wallaby["records"]
        if (str(row["name"]), str(row["team_release_kin"])) not in used
    ]
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in unused:
        key = (str(row["name"]), str(row["team_release_kin"]))
        optical_row = optical_by_key.get(key)
        reasons = (
            ["no_accepted_optical_match"]
            if optical_row is None
            else _predictor_quality_reasons(row, optical_row, item39_config)
        )
        if len(row["radius_kpc"]) < int(config["predictor_quality"]["minimum_graph_nodes"]):
            reasons.append("insufficient_graph_nodes")
        if reasons:
            rejected.append({"name": key[0], "team_release_kin": key[1], "reasons": reasons})
            continue
        eligible.append(
            {
                "name": key[0],
                "team_release_kin": key[1],
                "ra": row["ra"],
                "dec": row["dec"],
                "wallaby_record_index": wallaby["records"].index(row),
                "legacy_record_index": optical["records"].index(optical_row),
                "graph_nodes": len(row["radius_kpc"]),
            }
        )
    if len(unused) != int(config["independence"]["unused_response_blind_predictor_count"]):
        raise GravityItem40Error("unused predictor count changed")
    if len(eligible) != int(config["independence"]["expected_quality_eligible_count"]):
        raise GravityItem40Error("eligible predictor count changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item40-predictor-receipt-1.0",
            "item": 40,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "wallaby_predictor_sha256": _sha256_file(wallaby_path),
            "legacy_predictor_sha256": _sha256_file(optical_path),
            "item39_sample_sha256": _sha256_file(sample_path),
            "eligible": eligible,
            "rejected": rejected,
            "counts": {
                "source_predictors": len(wallaby["records"]),
                "excluded_item39_roles": len(used),
                "unused_response_blind_predictors": len(unused),
                "quality_eligible": len(eligible),
                "quality_rejected": len(rejected),
                "response_rows_read": 0,
                "confirmation_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "all_values_are_predictors": True,
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
    if str(config["predictor_freeze_commit"]).startswith("PENDING_"):
        raise GravityItem40Error("predictor freeze is not commit-bound")
    receipt_path = _source_path(root, config, "predictor_receipt")
    receipt = _read_json(receipt_path)
    if receipt != build_predictor_receipt(root):
        raise GravityItem40Error("predictor receipt drifted")
    wallaby = _read_json(root / str(config["data_sources"]["wallaby_predictors"]))
    optical = _read_json(root / str(config["data_sources"]["legacy_predictors"]))
    eligible: list[dict[str, Any]] = []
    for identity in receipt["eligible"]:
        gas = wallaby["records"][int(identity["wallaby_record_index"])]
        stars = optical["records"][int(identity["legacy_record_index"])]
        stellar_mass = float(stars["stellar_mass_msun"])
        gas_mass = float(config["constants"]["helium_mass_factor"]) * float(gas["hi_mass_msun"])
        total = stellar_mass + gas_mass
        eligible.append(
            {
                "name": identity["name"],
                "team_release_kin": identity["team_release_kin"],
                "ra": identity["ra"],
                "dec": identity["dec"],
                "distance_mpc": gas["distance_mpc"],
                "hi_mass_msun": gas["hi_mass_msun"],
                "stellar_mass_msun": stars["stellar_mass_msun"],
                "effective_radius_kpc": stars["effective_radius_kpc"],
                "screen_radius_kpc": gas["screen_radius_kpc"],
                "total_baryonic_mass_msun": f"{total:.12e}",
                "gas_fraction": f"{gas_mass / total:.12e}",
                "graph_nodes": identity["graph_nodes"],
            }
        )
    mass = np.log10([float(row["total_baryonic_mass_msun"]) for row in eligible])
    gas_fraction = np.asarray([float(row["gas_fraction"]) for row in eligible])
    mass_median = float(np.median(mass))
    gas_median = float(np.median(gas_fraction))
    salt = str(config["sample_boundary"]["split_salt"])
    cells: dict[str, list[dict[str, Any]]] = {}
    for row, mass_value, gas_value in zip(eligible, mass, gas_fraction, strict=True):
        cell = f"mass{int(mass_value > mass_median)}|gas{int(gas_value > gas_median)}"
        enriched = dict(row)
        enriched["source_cell"] = cell
        identity = f"{row['name']}|{row['team_release_kin']}"
        enriched["confirmation_hash"] = _split_hash(f"confirm|{identity}", salt)
        cells.setdefault(cell, []).append(enriched)
    confirmation_count = int(config["sample_boundary"]["minimum_reserved_confirmation_galaxies"])
    ranked = sorted(
        eligible,
        key=lambda row: _split_hash(
            f"confirm|{row['name']}|{row['team_release_kin']}", salt
        ),
    )
    confirmations = {
        (str(row["name"]), str(row["team_release_kin"]))
        for row in ranked[:confirmation_count]
    }
    objects: list[dict[str, Any]] = []
    fold_salt = str(config["evaluation"]["fold_salt"])
    outer_folds = int(config["evaluation"]["outer_folds"])
    for rows in cells.values():
        for row in rows:
            identity = f"{row['name']}|{row['team_release_kin']}"
            output = {key: value for key, value in row.items() if key != "confirmation_hash"}
            output["role"] = (
                "reserved_confirmation"
                if (str(row["name"]), str(row["team_release_kin"])) in confirmations
                else "exploration"
            )
            output["outer_fold"] = int(_split_hash(identity, fold_salt), 16) % outer_folds
            output["response_read"] = False
            objects.append(output)
    objects.sort(key=lambda row: (str(row["name"]), str(row["team_release_kin"])))
    role_counts = Counter(str(row["role"]) for row in objects)
    if role_counts["exploration"] != int(
        config["sample_boundary"]["maximum_exploration_galaxies"]
    ):
        raise GravityItem40Error("exploration count changed")
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item40-sample-manifest-1.0",
            "item": 40,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "predictor_freeze_commit": config["predictor_freeze_commit"],
            "predictor_receipt_sha256": _sha256_file(receipt_path),
            "mass_median_log10_msun": f"{mass_median:.12e}",
            "gas_fraction_median": f"{gas_median:.12e}",
            "objects": objects,
            "counts": {
                "selected_total": len(objects),
                "exploration": role_counts["exploration"],
                "reserved_confirmation": role_counts["reserved_confirmation"],
                "response_rows_read": 0,
                "confirmation_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "response_opened": False,
                "confirmation_opened": False,
                "failed_identity_replacement": False,
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
        raise GravityItem40Error("sample freeze is not commit-bound")
    sample_path = _source_path(root, config, "sample_manifest")
    sample = _read_json(sample_path)
    if sample != build_sample_manifest(root):
        raise GravityItem40Error("sample manifest drifted")
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
        user_agent="Invariant/Item40-WALLABY-Exploration-Responses",
    )
    expected_columns = set(config["data_sources"]["response_columns"])
    if any(set(row) != expected_columns for row in rows):
        raise GravityItem40Error("WALLABY response schema changed")
    expected = {(str(row["name"]), str(row["team_release_kin"])) for row in exploration}
    returned = {(str(row["name"]), str(row["team_release_kin"])) for row in rows}
    if returned != expected or len(rows) != len(returned) or returned & confirmations:
        raise GravityItem40Error("WALLABY response scope changed")
    return_path = _source_path(root, config, "wallaby_response_source")
    _write_json(
        return_path,
        _content_hashed(
            {
                "schema_version": "invariant-gravity-item40-wallaby-exploration-response-1.0",
                "item": 40,
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "predictor_freeze_commit": config["predictor_freeze_commit"],
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
    return return_path


def extract_wallaby_profiles(root: Path) -> dict[str, Path]:
    config = load_config(root)
    wallaby_path = root / str(config["data_sources"]["wallaby_predictors"])
    optical_path = root / str(config["data_sources"]["legacy_predictors"])
    sample_path = _source_path(root, config, "sample_manifest")
    response_path = _source_path(root, config, "wallaby_response_source")
    wallaby = _read_json(wallaby_path)
    optical = _read_json(optical_path)
    sample = _read_json(sample_path)
    response = _read_json(response_path)
    if sample != build_sample_manifest(root):
        raise GravityItem40Error("sample manifest drifted")
    if response["sample_freeze_commit"] != config["sample_freeze_commit"]:
        raise GravityItem40Error("response/sample binding changed")
    if int(response["counts"]["confirmation_response_rows"]) != 0:
        raise GravityItem40Error("confirmation response entered Item 40")
    profiles = {
        (str(row["name"]), str(row["team_release_kin"])): _deserialize_wallaby_profile(row)
        for row in wallaby["records"]
    }
    stars = {
        (str(row["galaxy"]), str(row["team_release_kin"])): row
        for row in optical["records"]
    }
    samples = {
        (str(row["name"]), str(row["team_release_kin"])): row
        for row in sample["objects"]
        if row["role"] == "exploration"
    }
    feature_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    galaxy_receipts: list[dict[str, Any]] = []
    constants = config["constants"]
    quality = config["response_quality"]
    g_constant = float(constants["gravitational_constant_kpc_km2_s2_msun"])
    acceleration_conversion = 1.0e6 / 3.085677581491367e19
    a0 = float(constants["acceleration_scale_m_s2"])
    for raw in response["records"]:
        key = (str(raw["name"]), str(raw["team_release_kin"]))
        if key not in samples or key not in profiles or key not in stars:
            raise GravityItem40Error(f"response lacks frozen predictors: {key}")
        profile = profiles[key]
        optical_row = stars[key]
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
                raise GravityItem40Error("rotation response vector lengths differ")
        except (ValueError, GravityItem40Error) as exc:
            galaxy_receipts.append(
                {
                    "name": key[0],
                    "team_release_kin": key[1],
                    "quality_pass": False,
                    "quality_failure_reasons": [f"parser:{exc}"],
                    "raw_rotation_points": 0,
                    "accepted_rotation_points": 0,
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
            np.maximum(velocity_error, 0.0) ** 2 + np.maximum(inclination_error, 0.0) ** 2
        )
        within = (radius_kpc > 0.0) & (radius_kpc <= profile["screen_radius_kpc"])
        valid = within & (velocity >= float(quality["minimum_speed_km_s"]))
        valid &= total_error / np.maximum(velocity, 1e-12) <= float(
            quality["maximum_fractional_speed_error"]
        )
        if int(np.sum(valid)) < int(quality["minimum_rotation_points"]):
            reasons.append("insufficient_rotation_points")
        if float(np.mean(within)) < float(
            quality["minimum_fraction_response_radii_within_screen"]
        ):
            reasons.append("screen_overlap")
        passed = not reasons
        accepted_count = int(np.sum(valid)) if passed else 0
        if passed:
            indices = np.flatnonzero(valid)
            accepted_radius = radius_kpc[indices]
            stellar_mass = float(optical_row["stellar_mass_msun"])
            scale_radius = float(optical_row["effective_radius_kpc"]) / 1.678
            source_radius = profile["radius_kpc"]
            source_gas = (
                profile["cumulative_hi_mass_msun"] * float(constants["helium_mass_factor"])
            )
            source_stars = _stellar_enclosed_mass(stellar_mass, source_radius, scale_radius)
            source_baryonic = np.maximum(source_gas + source_stars, 1e-12)
            graph = graph_coordinates(source_radius, source_baryonic)
            graph_at_radius = np.vstack(
                [
                    np.interp(
                        accepted_radius,
                        source_radius,
                        lane,
                        left=lane[0],
                        right=lane[-1],
                    )
                    for lane in graph
                ]
            )
            gas_enclosed = (
                np.interp(
                    accepted_radius,
                    source_radius,
                    profile["cumulative_hi_mass_msun"],
                    left=0.0,
                    right=profile["hi_mass_msun"],
                )
                * float(constants["helium_mass_factor"])
            )
            stellar_enclosed = _stellar_enclosed_mass(
                stellar_mass, accepted_radius, scale_radius
            )
            baryonic = gas_enclosed + stellar_enclosed
            total_baryonic = stellar_mass + float(constants["helium_mass_factor"]) * float(
                profile["hi_mass_msun"]
            )
            fraction = np.clip(baryonic / total_baryonic, 1e-8, 1.0)
            source_slope = np.gradient(np.log(source_baryonic), np.log(source_radius))
            slope = np.interp(accepted_radius, source_radius, source_slope)
            local_sigma = np.interp(
                accepted_radius,
                source_radius,
                profile["surface_density_hi_msun_pc2"],
            )
            gbar = g_constant * baryonic / np.square(accepted_radius)
            gbar_m_s2 = gbar * acceleration_conversion
            u = gbar_m_s2 / a0
            vbar = np.sqrt(g_constant * baryonic / accepted_radius)
            x = accepted_radius / profile["screen_radius_kpc"]
            item39_h = boundary_coordinates(fraction, x, slope)
            for output_index, source_index in enumerate(indices):
                feature_rows.append(
                    {
                        "galaxy": key[0],
                        "team_release_kin": key[1],
                        "point_index": output_index,
                        "outer_fold": sample_row["outer_fold"],
                        "source_cell": sample_row["source_cell"],
                        "radius_kpc": f"{accepted_radius[output_index]:.12e}",
                        "radius_over_screen": f"{x[output_index]:.12e}",
                        "local_hi_surface_density": f"{local_sigma[output_index]:.12e}",
                        "enclosed_baryonic_mass_msun": f"{baryonic[output_index]:.12e}",
                        "enclosed_fraction": f"{fraction[output_index]:.12e}",
                        "enclosed_log_slope": f"{slope[output_index]:.12e}",
                        "gbar_m_s2": f"{gbar_m_s2[output_index]:.12e}",
                        "u": f"{u[output_index]:.12e}",
                        "vbar_km_s": f"{vbar[output_index]:.12e}",
                        "h_spectral": f"{graph_at_radius[0, output_index]:.12e}",
                        "h_resistance": f"{graph_at_radius[1, output_index]:.12e}",
                        "h_heat": f"{graph_at_radius[2, output_index]:.12e}",
                        "h_communicability": f"{graph_at_radius[3, output_index]:.12e}",
                        "item39_h_equipartition": f"{item39_h[0, output_index]:.12e}",
                        "item39_h_quasilocal": f"{item39_h[1, output_index]:.12e}",
                        "item39_h_wedge": f"{item39_h[2, output_index]:.12e}",
                        "item39_h_flow": f"{item39_h[3, output_index]:.12e}",
                        "total_baryonic_mass_msun": sample_row[
                            "total_baryonic_mass_msun"
                        ],
                        "gas_fraction": sample_row["gas_fraction"],
                        "effective_radius_kpc": sample_row["effective_radius_kpc"],
                        "distance_mpc": sample_row["distance_mpc"],
                        "inclination_degrees": f"{inclination:.12e}",
                    }
                )
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
                "source_cell": sample_row["source_cell"],
                "quality_pass": passed,
                "quality_failure_reasons": reasons,
                "raw_rotation_points": len(radius_arcsec),
                "accepted_rotation_points": accepted_count,
                "inclination_degrees": f"{inclination:.12e}",
                "qflag_model": f"{qflag:.12e}",
            }
        )
    feature_fields = [
        "galaxy",
        "team_release_kin",
        "point_index",
        "outer_fold",
        "source_cell",
        "radius_kpc",
        "radius_over_screen",
        "local_hi_surface_density",
        "enclosed_baryonic_mass_msun",
        "enclosed_fraction",
        "enclosed_log_slope",
        "gbar_m_s2",
        "u",
        "vbar_km_s",
        "h_spectral",
        "h_resistance",
        "h_heat",
        "h_communicability",
        "item39_h_equipartition",
        "item39_h_quasilocal",
        "item39_h_wedge",
        "item39_h_flow",
        "total_baryonic_mass_msun",
        "gas_fraction",
        "effective_radius_kpc",
        "distance_mpc",
        "inclination_degrees",
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
                "schema_version": "invariant-gravity-item40-extraction-summary-1.0",
                "item": 40,
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
                "quality": config["response_quality"],
                "claims": {
                    "graph_features_used_rotation_response": False,
                    "confirmation_opened": False,
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
