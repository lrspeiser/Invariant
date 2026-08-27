"""Counterexample-driven G1 repair using RAR-base baryonic residual formulas."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g0_experiment import (
    _empirical_rar,
    score_predictions,
)
from .gravity_g0_experiment import (
    load_config as load_g0_config,
)
from .gravity_g1_atlas import (
    load_config as load_atlas_config,
)
from .gravity_g1_atlas import (
    validate_checkpoint,
)
from .gravity_g1_pilot import (
    FAILURE_NAMES,
    _baseline_contract,
    _best_rows,
    _binding,
    _load_json,
    _merge_best,
    _metric,
)
from .gravity_g1_pilot_v2 import _fit_two_columns
from .gravity_g1_pilot_v3 import (
    COMPONENT_COUNT,
    _basis_pair,
    _batches,
    _component_metadata,
    _feature_matrix,
    baryonic_features,
)
from .gravity_g1_pilot_v3 import (
    load_config as load_v3_config,
)
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import Galaxy, assemble

SCHEMA = "invariant-gravity-g1-atlas-repair-receipt-2.0"
CONFIG_SCHEMA = "invariant-gravity-g1-atlas-repair-config-2.0"
CONFIG_PATH = "configs/gravity_g1_atlas_repair_v2.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g1_atlas_repair_v2.py"
TEST_PATH = "tests/test_gravity_g1_atlas_repair_v2.py"
OUTPUT_PATH = "runs/gravity/g1-atlas/repair-v2.json"


class GravityG1AtlasRepairError(ValueError):
    """The G1 counterexample repair or its evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load and structurally validate the repair contract."""

    config = _load_json(root.resolve() / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG1AtlasRepairError("G1 repair config schema changed")
    if config.get("repair_galaxies") != ["NGC2955"]:
        raise GravityG1AtlasRepairError("G1 repair target changed")
    segments = config.get("segments")
    if not isinstance(segments, list) or len(segments) != 3:
        raise GravityG1AtlasRepairError("G1 repair segments changed")
    total = sum(int(row["candidate_count"]) for row in segments)
    if total != int(config["candidate_budget_per_repair_galaxy"]):
        raise GravityG1AtlasRepairError("G1 repair candidate budget does not sum exactly")
    shell = config.get("candidate_shell", {})
    if shell.get("maximum_local_constants") != 2 or shell.get("proposal_reads_vobs") is not False:
        raise GravityG1AtlasRepairError("G1 repair candidate-shell boundary changed")
    if config.get("admission", {}).get("confirmation_evaluator_accesses_allowed") != 0:
        raise GravityG1AtlasRepairError("G1 repair permits confirmation access")
    return config


def predecessor_summary(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every v1 checkpoint and reproduce its declared counterexample root."""

    root = root.resolve()
    predecessor = config["predecessor_binding"]
    atlas_config = load_atlas_config(root)
    galaxies = list(assemble(root).exploration)
    if len(galaxies) != int(predecessor["checkpoint_count"]):
        raise GravityG1AtlasRepairError("G1 predecessor population changed")
    root_rows: list[dict[str, str]] = []
    covered = 0
    uncovered: list[str] = []
    directory = root / str(predecessor["checkpoint_directory"])
    for galaxy in galaxies:
        checkpoint = _load_json(directory / f"{galaxy.name}.json")
        validate_checkpoint(
            checkpoint,
            root=root,
            config=atlas_config,
            expected_galaxy=galaxy.name,
        )
        root_rows.append(
            {
                "galaxy": str(checkpoint["galaxy"]),
                "content_sha256": str(checkpoint["content_sha256"]),
            }
        )
        if checkpoint["covered"]:
            covered += 1
        else:
            uncovered.append(galaxy.name)
    if canonical_sha256(root_rows) != predecessor["checkpoint_content_root_sha256"]:
        raise GravityG1AtlasRepairError("G1 predecessor checkpoint root changed")
    if covered != int(predecessor["required_covered_galaxies"]):
        raise GravityG1AtlasRepairError("G1 predecessor coverage changed")
    if uncovered != list(predecessor["required_uncovered_galaxies"]):
        raise GravityG1AtlasRepairError("G1 predecessor counterexamples changed")
    return {
        "checkpoint_content_root_sha256": canonical_sha256(root_rows),
        "covered_galaxies": covered,
        "uncovered_galaxies": uncovered,
    }


def _rar_base_v2(arrays: Mapping[str, np.ndarray], a0: float) -> np.ndarray:
    return _empirical_rar(arrays["radius"], arrays["vbar2"], a0) ** 2


def _score_batch(
    xp: Any,
    phi1: Any,
    phi2: Any,
    domain_valid: Any,
    contract: Mapping[str, Any],
    base_v2: np.ndarray,
    *,
    dtype: Any,
    relative_slack: float,
    absolute_slack: float,
    coverage_slack: int,
) -> tuple[Any, Any, Any]:
    arrays = contract["arrays"]
    radius = xp.asarray(arrays["radius"], dtype=dtype)
    base = xp.asarray(base_v2, dtype=dtype)
    vobs = xp.asarray(arrays["vobs"], dtype=dtype)
    sigma = xp.asarray(arrays["sigma"], dtype=dtype)
    column1 = radius[None, :] * phi1
    column2 = radius[None, :] * phi2
    target = vobs * vobs - base
    sigma_v2 = 2 * vobs * sigma + sigma * sigma
    weights = 1.0 / xp.maximum(sigma_v2, dtype(np.finfo(np.float32).tiny)) ** 2
    total_chi = xp.zeros(phi1.shape[0], dtype=dtype)
    total_coverage = xp.zeros(phi1.shape[0], dtype=xp.int16)
    valid = domain_valid.copy()
    passes_newton = domain_valid.copy()
    passes_wrong = domain_valid.copy()
    passes_rar = domain_valid.copy()
    passes_nfw = domain_valid.copy()
    passes_coverage = domain_valid.copy()
    for row in contract["fold_rows"]:
        fold = row["fold"]
        train = xp.asarray(fold.training, dtype=xp.int64)
        held = xp.asarray(fold.holdout, dtype=xp.int64)
        weighted1 = column1[:, train] * weights[train][None, :]
        weighted2 = column2[:, train] * weights[train][None, :]
        m11 = xp.sum(weighted1 * column1[:, train], axis=1)
        m22 = xp.sum(weighted2 * column2[:, train], axis=1)
        m12 = xp.sum(weighted1 * column2[:, train], axis=1)
        b1 = xp.sum(weighted1 * target[train][None, :], axis=1)
        b2 = xp.sum(weighted2 * target[train][None, :], axis=1)
        determinant = m11 * m22 - m12 * m12
        conditioned = determinant > dtype(1e-10) * m11 * m22
        safe_det = xp.where(conditioned, determinant, dtype(1))
        coefficient1 = (b1 * m22 - b2 * m12) / safe_det
        coefficient2 = (b2 * m11 - b1 * m12) / safe_det
        prediction2 = (
            base[held][None, :]
            + coefficient1[:, None] * column1[:, held]
            + coefficient2[:, None] * column2[:, held]
        )
        positive = xp.all(xp.isfinite(prediction2) & (prediction2 > dtype(0)), axis=1)
        fold_valid = conditioned & positive
        valid &= fold_valid
        prediction = xp.sqrt(xp.where(prediction2 > dtype(0), prediction2, dtype(1)))
        residual = (prediction - vobs[held][None, :]) / sigma[held][None, :]
        residual = xp.where(xp.isfinite(residual), residual, dtype(1e12))
        fold_chi = xp.sum(residual * residual, axis=1)
        fold_coverage = xp.sum(xp.abs(residual) <= dtype(2), axis=1).astype(xp.int16)
        total_chi += fold_chi
        total_coverage += fold_coverage
        thresholds = row["thresholds"]
        slack = lambda value: dtype(value * (1.0 + relative_slack) + absolute_slack)
        passes_newton &= fold_valid & (fold_chi < slack(thresholds["newtonian"]))
        passes_wrong &= fold_valid & (fold_chi < slack(thresholds["wrong"]))
        passes_rar &= fold_valid & (fold_chi <= slack(thresholds["rar"]))
        passes_nfw &= fold_valid & (fold_chi <= slack(thresholds["nfw"]))
        passes_coverage &= fold_valid & (
            fold_coverage >= max(0, int(thresholds["coverage_count"]) - coverage_slack)
        )
    aggregate = contract["aggregate"]
    point_count = arrays["vobs"].size
    required_coverage = math.ceil(
        min(0.9, float(aggregate["nfw_halo_ceiling"]["coverage_two_sigma"]))
        * point_count
        - 1e-12
    )
    slack = lambda value: dtype(value * (1.0 + relative_slack) + absolute_slack)
    passes_newton &= total_chi < slack(float(aggregate["newtonian_baryons"]["chi_square"]))
    passes_wrong &= total_chi < slack(
        float(aggregate["wrong_high_acceleration_boost"]["chi_square"])
    )
    passes_rar &= total_chi <= slack(float(aggregate["empirical_rar"]["chi_square"]))
    passes_nfw &= total_chi <= slack(
        float(aggregate["nfw_halo_ceiling"]["chi_square"]) + 2.0 * point_count
    )
    passes_coverage &= total_coverage >= max(0, required_coverage - coverage_slack)
    survivor = valid & passes_newton & passes_wrong & passes_rar & passes_nfw & passes_coverage
    reason = xp.full(phi1.shape[0], 6, dtype=xp.int8)
    reason = xp.where(~passes_coverage, 5, reason)
    reason = xp.where(~passes_nfw, 4, reason)
    reason = xp.where(~passes_rar, 3, reason)
    reason = xp.where(~passes_wrong, 2, reason)
    reason = xp.where(~passes_newton, 1, reason)
    reason = xp.where(~valid, 0, reason)
    return total_chi, survivor, reason


def replay_candidate(
    galaxy: Galaxy,
    arm: str,
    ordinal: int,
    g0_config: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Make the authoritative CPU-FP64 decision for one repair candidate."""

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
    base_v2 = _rar_base_v2(arrays, a0)
    column1 = arrays["radius"] * phi1[0]
    column2 = arrays["radius"] * phi2[0]
    fit_arrays = dict(arrays)
    fit_arrays["vbar2"] = base_v2
    predictions = np.empty_like(arrays["vobs"])
    folds: list[dict[str, Any]] = []
    failure_set: set[str] = set()
    all_pass = True
    for row in contract["fold_rows"]:
        fold = row["fold"]
        try:
            coefficient1, coefficient2 = _fit_two_columns(
                column1, column2, fit_arrays, fold.training
            )
        except Exception:  # noqa: BLE001 - singular fits are typed rejections
            return {"admitted": False, "failure": "ill_conditioned_fold", "ordinal": ordinal}
        held = np.asarray(fold.holdout, dtype=np.int64)
        prediction2 = (
            base_v2[held]
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
        "base": "empirical_RAR",
        "components": components,
        "local_constants": ["A_km2_s2_kpc", "B_km2_s2_kpc"],
        "ordinal": ordinal,
        "shell": "rar2+r*(A*phi1(feature1)+B*phi2(feature2))",
    }
    formula_bytes = len(canonical_json_bytes(ir))
    grammar_bits = math.ceil(math.log2(COMPONENT_COUNT * (COMPONENT_COUNT - 1) // 2))
    return {
        "admitted": all_pass,
        "aggregate_checks": aggregate_checks,
        "aggregate_score": aggregate_score,
        "base_family": "empirical_RAR_MOND_phenomenology",
        "components": components,
        "description_length": {
            "canonical_formula_bytes": formula_bytes,
            "grammar_address_bits": grammar_bits,
            "local_constant_bits": 128,
            "total_bits": 8 * formula_bytes + 128 + grammar_bits,
        },
        "failure_obligations": sorted(failure_set),
        "formula": "V_pred^2=V_RAR^2+r*(A*phi_1(feature_1)+B*phi_2(feature_2))",
        "folds": folds,
        "historical_novelty_established": False,
        "origin_assessment": "new_combination_of_known_ideas",
        "ordinal": ordinal,
        "prediction_sha256": canonical_sha256(
            [format(float(value), ".15e") for value in predictions]
        ),
    }


def search_segment(
    galaxy: Galaxy,
    segment: Mapping[str, Any],
    config: Mapping[str, Any],
    g0_config: Mapping[str, Any],
    v3_config: Mapping[str, Any],
    *,
    candidate_count: int,
    use_gpu: bool,
) -> dict[str, Any]:
    """Search one declared RAR-residual segment."""

    prefilter = config["gpu_prefilter"]
    arm_id = str(segment["v3_arm"])
    arm = next(row for row in v3_config["arms"] if row["id"] == arm_id)
    batch_size = int(arm.get("chunk_size", 65_536))
    retain = int(prefilter["retained_candidates_per_segment_for_cpu_replay"])
    contract = _baseline_contract(galaxy, g0_config)
    arrays = contract["arrays"]
    a0 = float(
        next(item for item in g0_config["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    base_v2 = _rar_base_v2(arrays, a0)
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
            xp, arm_id, ordinals, feature_matrix, xp.float32
        )
        scores, survivor, reasons = _score_batch(
            xp,
            phi1,
            phi2,
            valid,
            contract,
            base_v2,
            dtype=xp.float32,
            relative_slack=float(prefilter["relative_score_slack"]),
            absolute_slack=float(prefilter["absolute_score_slack"]),
            coverage_slack=int(prefilter["coverage_count_slack"]),
        )
        counts = xp.bincount(reasons, minlength=len(FAILURE_NAMES))
        failure_counts += counts.get() if use_gpu else counts
        best_valid = _merge_best(best_valid, _best_rows(xp, scores, valid, ordinals, 16), 64)
        best_prefilter = _merge_best(
            best_prefilter,
            _best_rows(xp, scores, survivor, ordinals, 64),
            retain,
        )
        evaluated += int(host_ordinals.size)
    if use_gpu:
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    if evaluated != candidate_count or int(np.sum(failure_counts)) != candidate_count:
        raise GravityG1AtlasRepairError("G1 repair trial accounting changed")
    replayed = [
        replay_candidate(galaxy, arm_id, ordinal, g0_config, contract=contract)
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
            item = replay_candidate(galaxy, arm_id, ordinal, g0_config, contract=contract)
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
        "arm": arm_id,
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
        "segment_id": segment["id"],
        "top_domain_valid_diagnostics": diagnostics,
    }


def build_receipt(
    root: Path,
    *,
    candidate_count_override: int | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    """Run the repair and combine it with the sealed v1 coverage."""

    root = root.resolve()
    config = load_config(root)
    predecessor = predecessor_summary(root, config)
    g0_config = load_g0_config(root)
    v3_config = load_v3_config(root)
    galaxies = {galaxy.name: galaxy for galaxy in assemble(root).exploration}
    galaxy = galaxies["NGC2955"]
    trials = []
    for segment in config["segments"]:
        count = (
            int(segment["candidate_count"])
            if candidate_count_override is None
            else candidate_count_override
        )
        trials.append(
            search_segment(
                galaxy,
                segment,
                config,
                g0_config,
                v3_config,
                candidate_count=count,
                use_gpu=use_gpu,
            )
        )
    admitted = [
        {**candidate, "segment_id": trial["segment_id"]}
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
    new_trials = sum(int(trial["candidate_count"]) for trial in trials)
    full_run = (
        candidate_count_override is None
        and use_gpu
        and new_trials == int(config["candidate_budget_per_repair_galaxy"])
        and all(
            int(trial["candidate_count"]) == int(segment["candidate_count"])
            for trial, segment in zip(trials, config["segments"], strict=True)
        )
    )
    passed = full_run and bool(admitted)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G1",
        "decision": "PASS_G1_ATLAS_UNION_139_OF_139" if passed else "BLOCK_G1_REPAIR",
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "formula_is_universal": False,
            "g2_equivalence_authorized": passed,
            "historical_novelty_established": False,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "cumulative_candidate_galaxy_trials": 13_900_000_000 + new_trials,
            "new_repair_candidate_galaxy_trials": new_trials,
            "confirmation_evaluator_accesses": 0,
            "union_covered_galaxies": predecessor["covered_galaxies"] + int(bool(admitted)),
            "union_exploration_galaxies": 139,
        },
        "lineage_assessment": {
            "base": "known_family_instance",
            "construction": "new_combination_of_known_ideas",
            "authoritative_for_novelty": False,
            "note": "A known RAR base plus searched baryonic residual features; G2 must adjudicate equivalence and prior art.",
        },
        "predecessor": predecessor,
        "repair": {
            "candidate_count": new_trials,
            "covered": bool(admitted),
            "galaxy": galaxy.name,
            "point_count": galaxy.count,
            "retained_pareto": admitted[:64],
            "trials": trials,
        },
        "limitations": [
            "The repair was designed after observing the exploration-only NGC2955 failure ledger.",
            "RAR is a known empirical/MOND phenomenology baseline, not an independently discovered law.",
            "The residual construction still has two coefficients fitted locally inside each galaxy.",
            "Union coverage authorizes equivalence analysis, not a universal gravity or no-dark-matter claim.",
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
    """Validate a checked repair receipt and its current source bindings."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG1AtlasRepairError("G1 repair receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG1AtlasRepairError("G1 repair receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG1AtlasRepairError("G1 repair config binding changed")
    for key, path in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG1AtlasRepairError(f"G1 repair {key} binding changed")
    predecessor = predecessor_summary(root, config)
    if receipt.get("predecessor") != predecessor:
        raise GravityG1AtlasRepairError("G1 repair predecessor evidence changed")
    counts = receipt.get("counts", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG1AtlasRepairError("G1 repair records confirmation access")
    if receipt.get("claims", {}).get("historical_novelty_established") is not False:
        raise GravityG1AtlasRepairError("G1 repair overstates novelty")
    passed = receipt.get("decision") == "PASS_G1_ATLAS_UNION_139_OF_139"
    if passed and (
        counts.get("new_repair_candidate_galaxy_trials") != 100_000_000
        or counts.get("union_covered_galaxies") != 139
        or receipt.get("repair", {}).get("covered") is not True
    ):
        raise GravityG1AtlasRepairError("G1 repair PASS is unsupported")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG1AtlasRepairError(f"refusing to overwrite immutable receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        receipt = _load_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root=root)
        return 0
    receipt = build_receipt(
        root,
        candidate_count_override=args.candidate_count,
        use_gpu=not args.cpu_only,
    )
    if args.candidate_count is None and not args.cpu_only:
        _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "covered": receipt["repair"]["covered"],
                "decision": receipt["decision"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"] == "PASS_G1_ATLAS_UNION_139_OF_139" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG1AtlasRepairError",
    "build_receipt",
    "load_config",
    "predecessor_summary",
    "replay_candidate",
    "search_segment",
    "validate_receipt",
]
