"""G1 pilot v3: baryonic-structure repair for the two v2 counterexamples."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g0_experiment import load_config as load_g0_config
from .gravity_g0_experiment import score_predictions
from .gravity_g1_pilot import (
    FAILURE_NAMES,
    _baseline_contract,
    _best_rows,
    _binding,
    _file_sha256,
    _load_json,
    _merge_best,
    _metric,
)
from .gravity_g1_pilot_v2 import (
    _fit_two_columns,
    _score_batch,
    random_pair_batches,
)
from .gravity_g1_pilot_v2 import (
    validate_receipt as validate_v2_receipt,
)
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import Galaxy, assemble

SCHEMA = "invariant-gravity-g1-pilot-receipt-3.0"
CONFIG_SCHEMA = "invariant-gravity-g1-pilot-config-3.0"
CONFIG_PATH = "configs/gravity_g1_pilot_v3.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g1_pilot_v3.py"
TEST_PATH = "tests/test_gravity_g1_pilot_v3.py"
OUTPUT_PATH = "runs/gravity/g1-pilot/receipt-v3.json"
COMPONENT_COUNT = 8_192
FEATURE_IDS = (
    "log_y",
    "log_r_over_disk_peak",
    "gas_fraction",
    "disk_fraction",
    "bulge_fraction",
    "baryon_log_slope",
    "mass_proxy_fraction",
    "gas_to_disk",
)
WIDTHS = np.asarray([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])


class GravityG1PilotV3Error(ValueError):
    """The v3 feature repair, search, or receipt is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG1PilotV3Error("G1 v3 config schema changed")
    predecessor = config.get("predecessor_binding")
    if not isinstance(predecessor, Mapping):
        raise GravityG1PilotV3Error("G1 v3 has no predecessor binding")
    path = root / str(predecessor.get("path"))
    if _file_sha256(path) != predecessor.get("file_sha256"):
        raise GravityG1PilotV3Error("G1 v3 predecessor file binding changed")
    receipt = _load_json(path)
    validate_v2_receipt(receipt, root=root)
    if receipt.get("content_sha256") != predecessor.get("content_sha256"):
        raise GravityG1PilotV3Error("G1 v3 predecessor content binding changed")
    if receipt.get("decision") != predecessor.get("required_decision"):
        raise GravityG1PilotV3Error("G1 v3 predecessor decision changed")
    covered = {row["galaxy"] for row in receipt["galaxies"] if row["covered"]}
    uncovered = {row["galaxy"] for row in receipt["galaxies"] if not row["covered"]}
    if len(covered) != int(predecessor["required_covered_galaxies"]):
        raise GravityG1PilotV3Error("G1 v3 inherited coverage count changed")
    if uncovered != set(predecessor["required_uncovered_galaxies"]):
        raise GravityG1PilotV3Error("G1 v3 inherited counterexample inventory changed")
    if tuple(config["repair_galaxies"]) != ("UGC11820", "UGC11455"):
        raise GravityG1PilotV3Error("G1 v3 repair galaxy order changed")
    if tuple(item["id"] for item in config["target_blind_features"]) != FEATURE_IDS:
        raise GravityG1PilotV3Error("G1 v3 feature inventory changed")
    if any(int(arm["component_count"]) != COMPONENT_COUNT for arm in config["arms"]):
        raise GravityG1PilotV3Error("G1 v3 component count changed")
    return config


def _center_values(feature_id: str) -> np.ndarray:
    if feature_id in {"gas_fraction", "disk_fraction", "bulge_fraction", "mass_proxy_fraction"}:
        values = np.linspace(-2.0, 3.0, 64)
    elif feature_id == "gas_to_disk":
        values = np.linspace(-4.0, 0.0, 64)
    else:
        values = np.linspace(-8.0, 8.0, 64)
    return np.asarray(sorted(values, key=lambda value: (abs(value), -value)))


def baryonic_features(galaxy: Galaxy, a0: float) -> dict[str, np.ndarray]:
    """Compute the eight declared dimensionless features without reading V_obs or errors."""

    from .gravity_g0_experiment import _galaxy_arrays

    arrays = _galaxy_arrays(galaxy)
    radius = arrays["radius"]
    vbar2 = arrays["vbar2"]
    gas = np.asarray([float(value) for value in galaxy.v_gas])
    disk = np.asarray([float(value) for value in galaxy.v_disk])
    bulge = np.asarray([float(value) for value in galaxy.v_bul])
    parts = np.abs(gas * gas) + 0.5 * disk * disk + 0.7 * bulge * bulge
    disk_peak_radius = radius[int(np.argmax(np.abs(disk)))]
    mass_proxy = vbar2 * radius
    return {
        "log_y": np.log(vbar2 / radius / a0),
        "log_r_over_disk_peak": np.log(radius / disk_peak_radius),
        "gas_fraction": np.abs(gas * gas) / parts,
        "disk_fraction": 0.5 * disk * disk / parts,
        "bulge_fraction": 0.7 * bulge * bulge / parts,
        "baryon_log_slope": np.gradient(np.log(vbar2), np.log(radius)),
        "mass_proxy_fraction": mass_proxy / np.max(mass_proxy),
        "gas_to_disk": np.log((np.abs(gas) + 1.0) / (np.abs(disk) + 1.0)),
    }


def lexicographic_pair_batches(total: int, batch_size: int) -> Iterator[np.ndarray]:
    pair_count = COMPONENT_COUNT * (COMPONENT_COUNT - 1) // 2
    if not 1 <= total <= pair_count:
        raise GravityG1PilotV3Error("v3 structured pair count is outside its grammar")
    produced = 0
    first = 0
    second = 1
    while produced < total:
        take = min(batch_size, total - produced)
        rows: list[np.ndarray] = []
        remaining = take
        while remaining:
            available = COMPONENT_COUNT - second
            width = min(remaining, available)
            rows.append(
                first * COMPONENT_COUNT
                + np.arange(second, second + width, dtype=np.int64)
            )
            second += width
            remaining -= width
            if second == COMPONENT_COUNT:
                first += 1
                second = first + 1
        batch = np.concatenate(rows)
        produced += batch.size
        yield batch


def _feature_matrix(xp: Any, features: Mapping[str, np.ndarray], dtype: Any) -> Any:
    return xp.asarray(np.vstack([features[name] for name in FEATURE_IDS]), dtype=dtype)


def _structured_components(
    xp: Any, component_ids: Any, feature_matrix: Any, dtype: Any
) -> Any:
    q_values = xp.asarray([2.0, 1.0], dtype=dtype)
    widths = xp.asarray(WIDTHS, dtype=dtype)
    centers = xp.asarray(np.vstack([_center_values(name) for name in FEATURE_IDS]), dtype=dtype)
    feature = component_ids // 1024
    local = component_ids % 1024
    center_rank = local // 16
    width = widths[(local // 2) % 8]
    q = q_values[local % 2]
    value = feature_matrix[feature]
    center = centers[feature, center_rank]
    z = xp.abs((value - center[:, None]) / width[:, None])
    return xp.exp(-(z ** q[:, None]))


def _skew_components(xp: Any, component_ids: Any, feature_matrix: Any, dtype: Any) -> Any:
    kappa_values = xp.asarray([-0.5, 0.5], dtype=dtype)
    widths = xp.asarray(WIDTHS, dtype=dtype)
    centers = xp.asarray(np.vstack([_center_values(name) for name in FEATURE_IDS]), dtype=dtype)
    feature = component_ids // 1024
    local = component_ids % 1024
    center_rank = local // 16
    width = widths[(local // 2) % 8]
    kappa = kappa_values[local % 2]
    value = feature_matrix[feature]
    center = centers[feature, center_rank]
    z = (value - center[:, None]) / width[:, None]
    return xp.exp(-(z * z)) * (dtype(1) + kappa[:, None] * xp.tanh(z))


def _creative_components(xp: Any, component_ids: Any, feature_matrix: Any, dtype: Any) -> Any:
    orientations = xp.asarray([-1.0, 1.0], dtype=dtype)
    exponents = xp.asarray([0.5, 1.0, 2.0, 3.0], dtype=dtype)
    scales = xp.asarray(np.geomspace(1.0 / 16.0, 16.0, 32), dtype=dtype)
    feature = component_ids // 1024
    local = component_ids % 1024
    orientation = orientations[local % 2]
    exponent = exponents[(local // 2) % 4]
    scale = scales[(local // 8) % 32]
    family = local // 256
    value = feature_matrix[feature]
    signed = orientation[:, None] * value
    magnitude = xp.abs(value) ** exponent[:, None]
    sigmoid = dtype(1) / (dtype(1) + xp.exp(xp.clip(scale[:, None] * signed, -60, 60)))
    localized = xp.exp(-scale[:, None] * magnitude)
    arctan = dtype(0.5) + orientation[:, None] * xp.arctan(scale[:, None] * value) / dtype(
        np.pi
    )
    positive_side = xp.maximum(signed, dtype(0)) ** exponent[:, None]
    saturated = dtype(1) - xp.exp(-scale[:, None] * positive_side)
    return xp.where(
        family[:, None] == 0,
        sigmoid,
        xp.where(family[:, None] == 1, localized, xp.where(family[:, None] == 2, arctan, saturated)),
    )


def _basis_pair(
    xp: Any,
    arm: str,
    ordinals: Any,
    feature_matrix: Any,
    dtype: Any,
) -> tuple[Any, Any, Any]:
    first = ordinals // COMPONENT_COUNT
    second = ordinals % COMPONENT_COUNT
    if arm == "structured_occam":
        phi1 = _structured_components(xp, first, feature_matrix, dtype)
        phi2 = _structured_components(xp, second, feature_matrix, dtype)
    elif arm == "pseudorandom_permutation":
        phi1 = _skew_components(xp, first, feature_matrix, dtype)
        phi2 = _skew_components(xp, second, feature_matrix, dtype)
    elif arm == "creativity_guided":
        phi1 = _creative_components(xp, first, feature_matrix, dtype)
        phi2 = _creative_components(xp, second, feature_matrix, dtype)
    else:
        raise GravityG1PilotV3Error(f"unknown v3 arm: {arm}")
    valid = xp.all(xp.isfinite(phi1) & xp.isfinite(phi2), axis=1)
    return phi1, phi2, valid


def _component_metadata(arm: str, component_id: int) -> dict[str, Any]:
    feature_index = component_id // 1024
    local = component_id % 1024
    feature_id = FEATURE_IDS[feature_index]
    if arm == "structured_occam":
        return {
            "center": _metric(float(_center_values(feature_id)[local // 16])),
            "family": "generalized_feature_rbf",
            "feature": feature_id,
            "q": _metric((2.0, 1.0)[local % 2]),
            "width": _metric(float(WIDTHS[(local // 2) % 8])),
        }
    if arm == "pseudorandom_permutation":
        return {
            "center": _metric(float(_center_values(feature_id)[local // 16])),
            "family": "skew_feature_rbf",
            "feature": feature_id,
            "kappa": _metric((-0.5, 0.5)[local % 2]),
            "width": _metric(float(WIDTHS[(local // 2) % 8])),
        }
    families = (
        "sigmoid_transition",
        "localized_exponential",
        "arctan_switch",
        "saturating_exponential",
    )
    return {
        "exponent": _metric((0.5, 1.0, 2.0, 3.0)[(local // 2) % 4]),
        "family": families[local // 256],
        "feature": feature_id,
        "llm_origin_assessment": "new_combination_of_known_ideas",
        "orientation": (-1, 1)[local % 2],
        "scale": _metric(float(np.geomspace(1.0 / 16.0, 16.0, 32)[(local // 8) % 32])),
    }


def replay_candidate(
    galaxy: Galaxy,
    arm: str,
    ordinal: int,
    config: Mapping[str, Any],
    g0_config: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if contract is None:
        contract = _baseline_contract(galaxy, g0_config)
    arrays = contract["arrays"]
    a0 = float(
        next(item for item in g0_config["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    features = baryonic_features(galaxy, a0)
    matrix = _feature_matrix(np, features, np.float64)
    phi1, phi2, valid = _basis_pair(
        np, arm, np.asarray([ordinal], dtype=np.int64), matrix, np.float64
    )
    if not bool(valid[0]):
        return {"admitted": False, "failure": "invalid_domain", "ordinal": ordinal}
    column1 = arrays["radius"] * phi1[0]
    column2 = arrays["radius"] * phi2[0]
    predictions = np.empty_like(arrays["vobs"])
    folds = []
    failure_set: set[str] = set()
    all_pass = True
    for row in contract["fold_rows"]:
        fold = row["fold"]
        try:
            coefficient1, coefficient2 = _fit_two_columns(
                column1, column2, arrays, fold.training
            )
        except Exception:  # noqa: BLE001 - any singular replay is a typed rejection
            return {"admitted": False, "failure": "ill_conditioned_fold", "ordinal": ordinal}
        held = np.asarray(fold.holdout, dtype=np.int64)
        prediction2 = (
            arrays["vbar2"][held]
            + coefficient1 * column1[held]
            + coefficient2 * column2[held]
        )
        if np.any(~np.isfinite(prediction2)) or np.any(prediction2 <= 0):
            return {"admitted": False, "failure": "nonpositive_heldout_v2", "ordinal": ordinal}
        predictions[held] = np.sqrt(prediction2)
        score = score_predictions(predictions[held], arrays["vobs"][held], arrays["sigma"][held])
        thresholds = row["thresholds"]
        coverage_count = round(float(score["coverage_two_sigma"]) * len(fold.holdout))
        checks = {
            "beats_newtonian": float(score["chi_square"]) < thresholds["newtonian"],
            "beats_wrong_law": float(score["chi_square"]) < thresholds["wrong"],
            "meets_empirical_rar": float(score["chi_square"]) <= thresholds["rar"],
            "meets_nfw_ceiling": float(score["chi_square"]) <= thresholds["nfw"],
            "meets_two_sigma_coverage": coverage_count >= thresholds["coverage_count"],
        }
        for name, passed in checks.items():
            if not passed:
                failure_set.add(name)
        all_pass &= all(checks.values())
        folds.append(
            {
                "A_km2_s2_kpc": _metric(coefficient1),
                "B_km2_s2_kpc": _metric(coefficient2),
                "checks": checks,
                "fold_id": fold.fold_id,
                "held_out_indices": list(fold.holdout),
                "score": score,
            }
        )
    aggregate_score = score_predictions(predictions, arrays["vobs"], arrays["sigma"])
    baseline = contract["aggregate"]
    coverage_count = round(float(aggregate_score["coverage_two_sigma"]) * galaxy.count)
    aggregate_checks = {
        "beats_newtonian": float(aggregate_score["chi_square"])
        < float(baseline["newtonian_baryons"]["chi_square"]),
        "beats_wrong_law": float(aggregate_score["chi_square"])
        < float(baseline["wrong_high_acceleration_boost"]["chi_square"]),
        "meets_empirical_rar": float(aggregate_score["chi_square"])
        <= float(baseline["empirical_rar"]["chi_square"]),
        "meets_nfw_ceiling": float(aggregate_score["chi_square"])
        <= float(baseline["nfw_halo_ceiling"]["chi_square"]) + 2.0 * galaxy.count,
        "meets_two_sigma_coverage": coverage_count
        >= math.ceil(
            min(0.9, float(baseline["nfw_halo_ceiling"]["coverage_two_sigma"]))
            * galaxy.count
            - 1e-12
        ),
    }
    for name, passed in aggregate_checks.items():
        if not passed:
            failure_set.add(f"aggregate_{name}")
    all_pass &= all(aggregate_checks.values())
    first = ordinal // COMPONENT_COUNT
    second = ordinal % COMPONENT_COUNT
    components = [_component_metadata(arm, first), _component_metadata(arm, second)]
    ir = {
        "arm": arm,
        "components": components,
        "local_constants": ["A_km2_s2_kpc", "B_km2_s2_kpc"],
        "ordinal": ordinal,
        "shell": "vbar2+r*(A*phi1(feature1)+B*phi2(feature2))",
    }
    formula_bytes = len(canonical_json_bytes(ir))
    grammar_bits = math.ceil(math.log2(COMPONENT_COUNT * (COMPONENT_COUNT - 1) // 2))
    return {
        "admitted": all_pass,
        "aggregate_checks": aggregate_checks,
        "aggregate_score": aggregate_score,
        "components": components,
        "description_length": {
            "canonical_formula_bytes": formula_bytes,
            "grammar_address_bits": grammar_bits,
            "local_constant_bits": 128,
            "total_bits": 8 * formula_bytes + 128 + grammar_bits,
        },
        "failure_obligations": sorted(failure_set),
        "folds": folds,
        "formula": "V_pred^2=V_bar^2+r*(A*phi_1(feature_1)+B*phi_2(feature_2))",
        "ordinal": ordinal,
        "prediction_sha256": canonical_sha256(
            [format(float(value), ".15e") for value in predictions]
        ),
    }


def _batches(arm: Mapping[str, Any], total: int, batch_size: int) -> Iterator[np.ndarray]:
    if arm["id"] == "structured_occam":
        return lexicographic_pair_batches(total, batch_size)
    return random_pair_batches(
        total,
        batch_size,
        component_count=COMPONENT_COUNT,
        seed=str(arm["seed"]),
    )


def search_arm_galaxy(
    galaxy: Galaxy,
    arm: Mapping[str, Any],
    config: Mapping[str, Any],
    g0_config: Mapping[str, Any],
    *,
    candidate_count: int,
    use_gpu: bool,
) -> dict[str, Any]:
    prefilter = config["gpu_prefilter"]
    batch_size = int(arm.get("chunk_size", 65_536))
    retain = int(prefilter["retained_candidates_per_arm_galaxy_for_cpu_replay"])
    contract = _baseline_contract(galaxy, g0_config)
    a0 = float(
        next(item for item in g0_config["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    features = baryonic_features(galaxy, a0)
    if use_gpu:
        import cupy as xp

        device = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        xp = np
        device = "cpu-numpy"
    feature_matrix = _feature_matrix(xp, features, xp.float32)
    failure_counts = np.zeros(len(FAILURE_NAMES), dtype=np.int64)
    best_valid: list[tuple[float, int]] = []
    best_prefilter: list[tuple[float, int]] = []
    evaluated = 0
    started = time.perf_counter()
    for host_ordinals in _batches(arm, candidate_count, batch_size):
        ordinals = xp.asarray(host_ordinals)
        phi1, phi2, valid = _basis_pair(
            xp, str(arm["id"]), ordinals, feature_matrix, xp.float32
        )
        scores, survivor, reasons = _score_batch(
            xp,
            phi1,
            phi2,
            valid,
            contract,
            dtype=xp.float32,
            relative_slack=float(prefilter["relative_score_slack"]),
            absolute_slack=float(prefilter["absolute_score_slack"]),
            coverage_slack=int(prefilter["coverage_count_slack"]),
        )
        counts = xp.bincount(reasons, minlength=len(FAILURE_NAMES))
        failure_counts += counts.get() if use_gpu else counts
        best_valid = _merge_best(best_valid, _best_rows(xp, scores, valid, ordinals, 16), 64)
        best_prefilter = _merge_best(
            best_prefilter, _best_rows(xp, scores, survivor, ordinals, 64), retain
        )
        evaluated += int(host_ordinals.size)
    if use_gpu:
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    if evaluated != candidate_count or int(np.sum(failure_counts)) != candidate_count:
        raise GravityG1PilotV3Error("G1 v3 trial accounting changed")
    replayed = [
        replay_candidate(
            galaxy,
            str(arm["id"]),
            ordinal,
            config,
            g0_config,
            contract=contract,
        )
        for _, ordinal in best_prefilter
    ]
    admitted = [item for item in replayed if item["admitted"]]
    admitted.sort(
        key=lambda item: (
            float(item["aggregate_score"]["chi_square"]),
            int(item["description_length"]["total_bits"]),
            int(item["ordinal"]),
        )
    )
    diagnostics = []
    by_ordinal = {int(item["ordinal"]): item for item in replayed}
    for score, ordinal in best_valid[:16]:
        item = by_ordinal.get(ordinal)
        if item is None:
            item = replay_candidate(
                galaxy,
                str(arm["id"]),
                ordinal,
                config,
                g0_config,
                contract=contract,
            )
        diagnostics.append(
            {
                "admitted": item["admitted"],
                "cpu_fp64_chi_square": item.get("aggregate_score", {}).get("chi_square"),
                "gpu_fp32_chi_square": _metric(score),
                "ordinal": ordinal,
            }
        )
    return {
        "admitted_count_among_cpu_replays": len(admitted),
        "arm": arm["id"],
        "candidate_count": candidate_count,
        "candidates_per_second": _metric(candidate_count / elapsed),
        "confirmation_evaluator_access_count": 0,
        "cpu_fp64_admitted_pareto": admitted[:64],
        "cpu_replay_count": len(replayed),
        "device": device,
        "elapsed_seconds": _metric(elapsed),
        "failure_ledger": {
            name: int(count) for name, count in zip(FAILURE_NAMES, failure_counts, strict=True)
        },
        "top_domain_valid_diagnostics": diagnostics,
    }


def build_receipt(
    root: Path, *, candidate_count: int | None = None, use_gpu: bool = True
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    g0_config = load_g0_config(root)
    population = assemble(root)
    by_name = {galaxy.name: galaxy for galaxy in population.exploration}
    predecessor = _load_json(root / config["predecessor_binding"]["path"])
    inherited = sorted(row["galaxy"] for row in predecessor["galaxies"] if row["covered"])
    repair_rows = []
    total_candidates = 0
    for name in config["repair_galaxies"]:
        trials = []
        for arm in config["arms"]:
            count = int(
                arm["candidate_count_per_repair_galaxy"]
                if candidate_count is None
                else candidate_count
            )
            trial = search_arm_galaxy(
                by_name[name],
                arm,
                config,
                g0_config,
                candidate_count=count,
                use_gpu=use_gpu,
            )
            trials.append(trial)
            total_candidates += count
        admitted = [
            candidate
            for trial in trials
            for candidate in trial["cpu_fp64_admitted_pareto"]
        ]
        admitted.sort(
            key=lambda item: (
                float(item["aggregate_score"]["chi_square"]),
                int(item["description_length"]["total_bits"]),
                int(item["ordinal"]),
            )
        )
        repair_rows.append(
            {
                "covered": bool(admitted),
                "galaxy": name,
                "point_count": by_name[name].count,
                "retained_pareto": admitted[:64],
                "trials": trials,
            }
        )
    full_run = (
        candidate_count is None
        and use_gpu
        and all(
            trial["candidate_count"] == 10_000_000
            for row in repair_rows
            for trial in row["trials"]
        )
    )
    repaired = [row["galaxy"] for row in repair_rows if row["covered"]]
    union = sorted(set(inherited) | set(repaired))
    passed = full_run and len(union) == 12 and len(repaired) == 2
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G1_PILOT_V3_REPAIR",
        "decision": (
            "PASS_G1_PILOT_UNION_12_OF_12"
            if passed
            else "BLOCK_G1_PILOT_V3_REPAIR_UNCOVERED_OR_INCOMPLETE"
        ),
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "formula_is_universal": False,
            "historical_novelty_established": False,
            "llm_labels_used_for_pruning": False,
            "pilot_authorizes_139_galaxy_scaleout": passed,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "confirmation_evaluator_accesses": 0,
            "cumulative_pilot_candidate_galaxy_trials": 450_000_000
            if full_run
            else 390_000_000 + total_candidates,
            "inherited_v2_covered_galaxies": len(inherited),
            "new_v3_candidate_galaxy_trials": total_candidates,
            "new_v3_covered_galaxies": len(repaired),
            "union_covered_pilot_galaxies": len(union),
        },
        "inherited_v2_covered_galaxies": inherited,
        "repair_galaxies": repair_rows,
        "union_covered_galaxies": union,
        "limitations": [
            "Pilot coverage is a union of distinct local formulas, not one shared law.",
            "Every local formula has two fitted acceleration coefficients and discrete structural choices; G3 must remove local gravitational freedom.",
            "Baryonic feature normalizations are deterministic and target-blind but may encode object structure that will need whole-galaxy population validation.",
            "A 12/12 pilot PASS authorizes scaleout only; it is not G1 completion, evidence against dark matter, or an alternative to GR.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG1PilotV3Error("G1 v3 receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG1PilotV3Error("G1 v3 receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG1PilotV3Error("G1 v3 receipt config binding changed")
    bindings = receipt.get("source_bindings", {})
    for key, relative in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if bindings.get(key) != _binding(root, relative):
            raise GravityG1PilotV3Error(f"G1 v3 {key} binding changed")
    if receipt.get("counts", {}).get("confirmation_evaluator_accesses") != 0:
        raise GravityG1PilotV3Error("G1 v3 receipt reports confirmation access")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise GravityG1PilotV3Error(f"refusing to overwrite a different G1 v3 receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.validate_checked:
        validate_receipt(_load_json(output), root=root)
        return 0
    receipt = build_receipt(
        root, candidate_count=args.candidate_count, use_gpu=not args.cpu_only
    )
    write_immutable(output, receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["decision"] == "PASS_G1_PILOT_UNION_12_OF_12" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG1PilotV3Error",
    "baryonic_features",
    "build_receipt",
    "lexicographic_pair_batches",
    "load_config",
    "replay_candidate",
    "search_arm_galaxy",
    "validate_receipt",
]
