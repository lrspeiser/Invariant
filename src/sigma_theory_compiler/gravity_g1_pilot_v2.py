"""G1 pilot v2: counterexample-guided two-kernel local formula search."""

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
    select_pilot_galaxies,
)
from .gravity_g1_pilot import (
    validate_receipt as validate_v1_receipt,
)
from .pseudorandom_ordinal import PseudorandomChunkSchedule
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import Galaxy, assemble

SCHEMA = "invariant-gravity-g1-pilot-receipt-2.0"
CONFIG_SCHEMA = "invariant-gravity-g1-pilot-config-2.0"
CONFIG_PATH = "configs/gravity_g1_pilot_v2.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g1_pilot_v2.py"
TEST_PATH = "tests/test_gravity_g1_pilot_v2.py"
OUTPUT_PATH = "runs/gravity/g1-pilot/receipt-v2.json"

STRUCTURED_COMPONENTS = 4_608
KERNEL_COMPONENTS = 8_192


class GravityG1PilotV2Error(ValueError):
    """The v2 counterexample repair, search, or receipt is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG1PilotV2Error("G1 v2 config schema changed")
    predecessor = config.get("predecessor_binding")
    if not isinstance(predecessor, Mapping):
        raise GravityG1PilotV2Error("G1 v2 has no predecessor binding")
    path = root / str(predecessor.get("path"))
    if _file_sha256(path) != predecessor.get("file_sha256"):
        raise GravityG1PilotV2Error("G1 v2 predecessor file binding changed")
    receipt = _load_json(path)
    validate_v1_receipt(receipt, root=root)
    if receipt.get("content_sha256") != predecessor.get("content_sha256"):
        raise GravityG1PilotV2Error("G1 v2 predecessor content binding changed")
    if receipt.get("decision") != predecessor.get("required_decision"):
        raise GravityG1PilotV2Error("G1 v2 predecessor is not the declared counterexample")
    population = assemble(root)
    selected = select_pilot_galaxies(population)
    if selected != tuple(config["pilot_galaxies_in_selection_order"]):
        raise GravityG1PilotV2Error("G1 v2 pilot selection changed")
    grammars = config.get("component_grammars")
    if not isinstance(grammars, list) or [item.get("arm") for item in grammars] != [
        "structured_occam",
        "pseudorandom_permutation",
        "creativity_guided",
    ]:
        raise GravityG1PilotV2Error("G1 v2 grammar inventory changed")
    expected = (STRUCTURED_COMPONENTS, KERNEL_COMPONENTS, KERNEL_COMPONENTS)
    if tuple(int(item["component_count"]) for item in grammars) != expected:
        raise GravityG1PilotV2Error("G1 v2 component counts changed")
    return config


def structured_pair_batches(total: int, batch_size: int) -> Iterator[np.ndarray]:
    """Canonical lexicographic pairs i<j from the symmetric kernel bank."""

    pair_count = STRUCTURED_COMPONENTS * (STRUCTURED_COMPONENTS - 1) // 2
    if not 1 <= total <= pair_count:
        raise GravityG1PilotV2Error("structured pair count is outside its grammar")
    produced = 0
    first = 0
    second = 1
    while produced < total:
        take = min(batch_size, total - produced)
        rows: list[np.ndarray] = []
        remaining = take
        while remaining:
            available = STRUCTURED_COMPONENTS - second
            width = min(remaining, available)
            values = first * STRUCTURED_COMPONENTS + np.arange(
                second, second + width, dtype=np.int64
            )
            rows.append(values)
            second += width
            remaining -= width
            if second == STRUCTURED_COMPONENTS:
                first += 1
                second = first + 1
        batch = np.concatenate(rows)
        produced += batch.size
        yield batch


def random_pair_batches(
    total: int, batch_size: int, *, component_count: int, seed: str
) -> Iterator[np.ndarray]:
    """Collision-free pseudorandom ordered storage, filtered to canonical i<j pairs."""

    pair_count = component_count * (component_count - 1) // 2
    if not 1 <= total <= pair_count:
        raise GravityG1PilotV2Error("random pair count is outside its grammar")
    schedule = PseudorandomChunkSchedule(component_count**2, batch_size, seed)
    produced = 0
    for chunk in schedule.iter():
        raw = np.arange(
            chunk["start_ordinal"], chunk["stop_ordinal_exclusive"], dtype=np.int64
        )
        first = raw // component_count
        second = raw % component_count
        canonical = raw[first < second]
        canonical = canonical[: total - produced]
        if canonical.size:
            produced += canonical.size
            yield canonical
        if produced == total:
            return
    raise GravityG1PilotV2Error("random pair schedule under-ran")


def _ordered_centers(count: int) -> np.ndarray:
    values = np.linspace(-8.0, 8.0, count, dtype=np.float64)
    return np.asarray(sorted(values, key=lambda value: (abs(value), -value)))


def _structured_components(xp: Any, component_ids: Any, log_y: Any, dtype: Any) -> Any:
    q_values = xp.asarray([2.0, 1.5, 1.0, 3.0], dtype=dtype)
    widths = xp.asarray(np.geomspace(0.25, 8.0, 16), dtype=dtype)
    centers = xp.asarray(_ordered_centers(72), dtype=dtype)
    q = q_values[component_ids % 4]
    width = widths[(component_ids // 4) % 16]
    center = centers[component_ids // 64]
    z = xp.abs((log_y[None, :] - center[:, None]) / width[:, None])
    return xp.exp(-(z ** q[:, None]))


def _skew_components(xp: Any, component_ids: Any, log_y: Any, dtype: Any) -> Any:
    kappa_values = xp.asarray([-0.5, 0.5], dtype=dtype)
    q_values = xp.asarray([2.0, 1.5, 1.0, 3.0], dtype=dtype)
    widths = xp.asarray(np.geomspace(0.25, 8.0, 16), dtype=dtype)
    centers = xp.asarray(_ordered_centers(64), dtype=dtype)
    kappa = kappa_values[component_ids % 2]
    q = q_values[(component_ids // 2) % 4]
    width = widths[(component_ids // 8) % 16]
    center = centers[component_ids // 128]
    z = (log_y[None, :] - center[:, None]) / width[:, None]
    return xp.exp(-(xp.abs(z) ** q[:, None])) * (
        dtype(1) + kappa[:, None] * xp.tanh(z)
    )


def _creative_components(xp: Any, component_ids: Any, y: Any, dtype: Any) -> Any:
    shape_values = xp.asarray([0.5, 1.0, 2.0, 3.0], dtype=dtype)
    exponent_values = xp.asarray([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0], dtype=dtype)
    scales = xp.asarray(np.geomspace(1.0 / 64.0, 64.0, 64), dtype=dtype)
    shape = shape_values[component_ids % 4]
    exponent = exponent_values[(component_ids // 4) % 8]
    scale = scales[(component_ids // 32) % 64]
    family = component_ids // 2048
    powered = y[None, :] ** exponent[:, None]
    inverse_powered = dtype(1) / powered
    t = dtype(1) / (dtype(1) + xp.sqrt(y[None, :] / scale[:, None]))
    transition = t**exponent[:, None] * (dtype(1) - t) ** shape[:, None]
    prony = xp.exp(-scale[:, None] * powered)
    arctan = ((dtype(2) / dtype(np.pi)) * xp.arctan(scale[:, None] * inverse_powered)) ** (
        shape[:, None]
    )
    saturated = (dtype(1) - xp.exp(-scale[:, None] * inverse_powered)) ** shape[:, None]
    return xp.where(
        family[:, None] == 0,
        transition,
        xp.where(family[:, None] == 1, prony, xp.where(family[:, None] == 2, arctan, saturated)),
    )


def _basis_pair(
    xp: Any, arm: str, ordinals: Any, y: Any, dtype: Any
) -> tuple[Any, Any, Any]:
    component_count = STRUCTURED_COMPONENTS if arm == "structured_occam" else KERNEL_COMPONENTS
    first = ordinals // component_count
    second = ordinals % component_count
    if arm == "structured_occam":
        phi1 = _structured_components(xp, first, xp.log(y.astype(dtype)), dtype)
        phi2 = _structured_components(xp, second, xp.log(y.astype(dtype)), dtype)
    elif arm == "pseudorandom_permutation":
        phi1 = _skew_components(xp, first, xp.log(y.astype(dtype)), dtype)
        phi2 = _skew_components(xp, second, xp.log(y.astype(dtype)), dtype)
    elif arm == "creativity_guided":
        phi1 = _creative_components(xp, first, y.astype(dtype), dtype)
        phi2 = _creative_components(xp, second, y.astype(dtype), dtype)
    else:
        raise GravityG1PilotV2Error(f"unknown v2 arm: {arm}")
    valid = xp.all(
        xp.isfinite(phi1) & xp.isfinite(phi2) & (phi1 >= dtype(0)) & (phi2 >= dtype(0)),
        axis=1,
    )
    return phi1, phi2, valid


def _score_batch(
    xp: Any,
    phi1: Any,
    phi2: Any,
    domain_valid: Any,
    contract: Mapping[str, Any],
    *,
    dtype: Any,
    relative_slack: float,
    absolute_slack: float,
    coverage_slack: int,
) -> tuple[Any, Any, Any]:
    arrays = contract["arrays"]
    radius = xp.asarray(arrays["radius"], dtype=dtype)
    vbar2 = xp.asarray(arrays["vbar2"], dtype=dtype)
    vobs = xp.asarray(arrays["vobs"], dtype=dtype)
    sigma = xp.asarray(arrays["sigma"], dtype=dtype)
    column1 = radius[None, :] * phi1
    column2 = radius[None, :] * phi2
    target = vobs * vobs - vbar2
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
            vbar2[held][None, :]
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
            fold_coverage
            >= max(0, int(thresholds["coverage_count"]) - coverage_slack)
        )
    aggregate = contract["aggregate"]
    point_count = arrays["vobs"].size
    required_coverage = math.ceil(
        min(0.9, float(aggregate["nfw_halo_ceiling"]["coverage_two_sigma"]))
        * point_count
        - 1e-12
    )
    slack = lambda value: dtype(value * (1.0 + relative_slack) + absolute_slack)
    passes_newton &= total_chi < slack(
        float(aggregate["newtonian_baryons"]["chi_square"])
    )
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


def _numpy_pair(arm: str, ordinal: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi1, phi2, valid = _basis_pair(
        np, arm, np.asarray([ordinal], dtype=np.int64), y, np.float64
    )
    if not bool(valid[0]):
        raise GravityG1PilotV2Error("CPU replay reached an invalid component pair")
    return phi1[0], phi2[0]


def _fit_two_columns(
    column1: np.ndarray,
    column2: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    training: Sequence[int],
) -> tuple[float, float]:
    train = np.asarray(training, dtype=np.int64)
    target = arrays["vobs"][train] ** 2 - arrays["vbar2"][train]
    sigma_v2 = (
        2.0 * arrays["vobs"][train] * arrays["sigma"][train] + arrays["sigma"][train] ** 2
    )
    weights = 1.0 / np.maximum(sigma_v2, np.finfo(np.float64).tiny) ** 2
    matrix = np.column_stack((column1[train], column2[train]))
    gram = matrix.T @ (weights[:, None] * matrix)
    determinant = float(np.linalg.det(gram))
    if determinant <= 1e-10 * float(gram[0, 0] * gram[1, 1]):
        raise GravityG1PilotV2Error("two-kernel fit is ill-conditioned")
    rhs = matrix.T @ (weights * target)
    values = np.linalg.solve(gram, rhs)
    return float(values[0]), float(values[1])


def _component_metadata(arm: str, component_id: int) -> dict[str, Any]:
    if arm == "structured_occam":
        q = (2.0, 1.5, 1.0, 3.0)[component_id % 4]
        width = np.geomspace(0.25, 8.0, 16)[(component_id // 4) % 16]
        center = _ordered_centers(72)[component_id // 64]
        return {
            "family": "symmetric_generalized_log_rbf",
            "mu": _metric(float(center)),
            "q": _metric(q),
            "width": _metric(float(width)),
        }
    if arm == "pseudorandom_permutation":
        kappa = (-0.5, 0.5)[component_id % 2]
        q = (2.0, 1.5, 1.0, 3.0)[(component_id // 2) % 4]
        width = np.geomspace(0.25, 8.0, 16)[(component_id // 8) % 16]
        center = _ordered_centers(64)[component_id // 128]
        return {
            "family": "skew_generalized_log_rbf",
            "kappa": _metric(kappa),
            "mu": _metric(float(center)),
            "q": _metric(q),
            "width": _metric(float(width)),
        }
    families = (
        ("generalized_transition_product", "H01_gen_transition", "cross_domain_synthesis"),
        ("prony_exponential", "H02_exp_basis", "cross_domain_synthesis"),
        ("arctan_switch", "H05_arctan", "proposed_new_construction"),
        ("saturating_exponential", "H06_sat_exp", "cross_domain_synthesis"),
    )
    family_id = component_id // 2048
    scale = np.geomspace(1.0 / 64.0, 64.0, 64)[(component_id // 32) % 64]
    exponent = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)[
        (component_id // 4) % 8
    ]
    shape = (0.5, 1.0, 2.0, 3.0)[component_id % 4]
    family, source, origin = families[family_id]
    return {
        "exponent": _metric(exponent),
        "family": family,
        "llm_origin_assessment": origin,
        "scale": _metric(float(scale)),
        "shape": _metric(shape),
        "source_hypothesis": source,
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
    y = arrays["vbar2"] / arrays["radius"] / float(
        config["candidate_shell"]["a0_km2_s2_kpc"]
    )
    try:
        phi1, phi2 = _numpy_pair(arm, ordinal, y)
    except GravityG1PilotV2Error:
        return {"admitted": False, "failure": "invalid_domain", "ordinal": ordinal}
    column1 = arrays["radius"] * phi1
    column2 = arrays["radius"] * phi2
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
        except GravityG1PilotV2Error:
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
    component_count = STRUCTURED_COMPONENTS if arm == "structured_occam" else KERNEL_COMPONENTS
    first = ordinal // component_count
    second = ordinal % component_count
    first_metadata = _component_metadata(arm, first)
    second_metadata = _component_metadata(arm, second)
    ir = {
        "arm": arm,
        "components": [first_metadata, second_metadata],
        "local_constants": ["A_km2_s2_kpc", "B_km2_s2_kpc"],
        "ordinal": ordinal,
        "shell": "vbar2+r*(A*phi1(y)+B*phi2(y))",
    }
    formula_bytes = len(canonical_json_bytes(ir))
    grammar = next(item for item in config["component_grammars"] if item["arm"] == arm)
    grammar_bits = math.ceil(math.log2(int(grammar["pair_count"])))
    try:
        full_a, full_b = _fit_two_columns(
            column1, column2, arrays, tuple(range(galaxy.count))
        )
    except GravityG1PilotV2Error:
        full_a, full_b = math.nan, math.nan
    return {
        "admitted": all_pass,
        "aggregate_checks": aggregate_checks,
        "aggregate_score": aggregate_score,
        "components": [first_metadata, second_metadata],
        "description_length": {
            "canonical_formula_bytes": formula_bytes,
            "grammar_address_bits": grammar_bits,
            "local_constant_bits": 128,
            "total_bits": 8 * formula_bytes + 128 + grammar_bits,
        },
        "failure_obligations": sorted(failure_set),
        "folds": folds,
        "formula": "V_pred^2=V_bar^2+r*(A*phi_1(y)+B*phi_2(y))",
        "full_data_constants_for_reporting_only": {
            "A_km2_s2_kpc": None if not math.isfinite(full_a) else _metric(full_a),
            "B_km2_s2_kpc": None if not math.isfinite(full_b) else _metric(full_b),
        },
        "ordinal": ordinal,
        "prediction_sha256": canonical_sha256(
            [format(float(value), ".15e") for value in predictions]
        ),
    }


def _batches(grammar: Mapping[str, Any], total: int, batch_size: int) -> Iterator[np.ndarray]:
    arm = str(grammar["arm"])
    if arm == "structured_occam":
        return structured_pair_batches(total, batch_size)
    return random_pair_batches(
        total,
        batch_size,
        component_count=int(grammar["component_count"]),
        seed=str(grammar["seed"]),
    )


def search_arm_galaxy(
    galaxy: Galaxy,
    grammar: Mapping[str, Any],
    config: Mapping[str, Any],
    g0_config: Mapping[str, Any],
    *,
    candidate_count: int,
    use_gpu: bool,
) -> dict[str, Any]:
    prefilter = config["gpu_prefilter"]
    batch_size = int(grammar.get("chunk_size", 65_536))
    retain = int(prefilter["retained_candidates_per_arm_galaxy_for_cpu_replay"])
    contract = _baseline_contract(galaxy, g0_config)
    arrays = contract["arrays"]
    y_numpy = arrays["vbar2"] / arrays["radius"] / float(
        config["candidate_shell"]["a0_km2_s2_kpc"]
    )
    if use_gpu:
        import cupy as xp

        device = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        xp = np
        device = "cpu-numpy"
    y = xp.asarray(y_numpy)
    failure_counts = np.zeros(len(FAILURE_NAMES), dtype=np.int64)
    best_valid: list[tuple[float, int]] = []
    best_prefilter: list[tuple[float, int]] = []
    evaluated = 0
    started = time.perf_counter()
    for host_ordinals in _batches(grammar, candidate_count, batch_size):
        ordinals = xp.asarray(host_ordinals)
        phi1, phi2, valid = _basis_pair(xp, str(grammar["arm"]), ordinals, y, xp.float32)
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
            best_prefilter, _best_rows(xp, scores, survivor, ordinals, 32), retain
        )
        evaluated += int(host_ordinals.size)
    if use_gpu:
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    if evaluated != candidate_count or int(np.sum(failure_counts)) != candidate_count:
        raise GravityG1PilotV2Error("G1 v2 trial accounting changed")
    replayed = [
        replay_candidate(
            galaxy,
            str(grammar["arm"]),
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
    diagnostic_rows = []
    by_ordinal = {int(item["ordinal"]): item for item in replayed}
    for score, ordinal in best_valid[:16]:
        item = by_ordinal.get(ordinal)
        if item is None:
            item = replay_candidate(
                galaxy,
                str(grammar["arm"]),
                ordinal,
                config,
                g0_config,
                contract=contract,
            )
        diagnostic_rows.append(
            {
                "admitted": item["admitted"],
                "cpu_fp64_chi_square": item.get("aggregate_score", {}).get("chi_square"),
                "gpu_fp32_chi_square": _metric(score),
                "ordinal": ordinal,
            }
        )
    return {
        "admitted_count_among_cpu_replays": len(admitted),
        "arm": grammar["arm"],
        "candidate_count": candidate_count,
        "candidates_per_second": _metric(candidate_count / elapsed),
        "confirmation_evaluator_access_count": 0,
        "cpu_fp64_admitted_pareto": admitted[:32],
        "cpu_replay_count": len(replayed),
        "device": device,
        "elapsed_seconds": _metric(elapsed),
        "failure_ledger": {
            name: int(count) for name, count in zip(FAILURE_NAMES, failure_counts, strict=True)
        },
        "top_domain_valid_diagnostics": diagnostic_rows,
    }


def build_receipt(
    root: Path,
    *,
    candidate_count: int | None = None,
    galaxy_limit: int | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    g0_config = load_g0_config(root)
    population = assemble(root)
    by_name = {galaxy.name: galaxy for galaxy in population.exploration}
    declared_names = list(config["pilot_galaxies_in_selection_order"])
    selected_names = declared_names[:galaxy_limit] if galaxy_limit is not None else declared_names
    if not selected_names:
        raise GravityG1PilotV2Error("G1 v2 selected no galaxies")
    galaxy_rows = []
    total_candidates = 0
    for name in selected_names:
        trials = []
        for grammar in config["component_grammars"]:
            count = int(
                grammar["candidate_count_per_galaxy"]
                if candidate_count is None
                else candidate_count
            )
            trial = search_arm_galaxy(
                by_name[name],
                grammar,
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
        galaxy_rows.append(
            {
                "covered": bool(admitted),
                "galaxy": name,
                "point_count": by_name[name].count,
                "retained_pareto": admitted[:64],
                "trials": trials,
            }
        )
    full_run = (
        selected_names == declared_names
        and candidate_count is None
        and use_gpu
        and all(
            trial["candidate_count"] == 10_000_000
            for galaxy in galaxy_rows
            for trial in galaxy["trials"]
        )
    )
    covered = sum(row["covered"] for row in galaxy_rows)
    passed = full_run and covered == 12
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G1_PILOT_V2",
        "decision": (
            "PASS_G1_PILOT_V2_12_OF_12"
            if passed
            else "BLOCK_G1_PILOT_V2_UNCOVERED_OR_INCOMPLETE"
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
            "candidate_galaxy_trials": total_candidates,
            "confirmation_evaluator_accesses": 0,
            "covered_pilot_galaxies": covered,
            "pilot_galaxies": len(selected_names),
        },
        "galaxies": galaxy_rows,
        "limitations": [
            "Two signed local acceleration coefficients make this a diagnostic atlas formula, not a universal law.",
            "Kernel components are functions of baryonic acceleration; an admitted pair is predictive within the frozen folds but is not a field equation.",
            "Provider origin labels are non-authoritative lineage and do not establish novelty.",
            "A pilot PASS covers twelve exploration galaxies only and requires a 139-galaxy scaleout before G2.",
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
        raise GravityG1PilotV2Error("G1 v2 receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG1PilotV2Error("G1 v2 receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG1PilotV2Error("G1 v2 receipt config binding changed")
    bindings = receipt.get("source_bindings", {})
    for key, relative in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if bindings.get(key) != _binding(root, relative):
            raise GravityG1PilotV2Error(f"G1 v2 {key} binding changed")
    if receipt.get("counts", {}).get("confirmation_evaluator_accesses") != 0:
        raise GravityG1PilotV2Error("G1 v2 receipt reports confirmation access")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise GravityG1PilotV2Error(f"refusing to overwrite a different G1 v2 receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--galaxy-limit", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.validate_checked:
        validate_receipt(_load_json(output), root=root)
        return 0
    receipt = build_receipt(
        root,
        candidate_count=args.candidate_count,
        galaxy_limit=args.galaxy_limit,
        use_gpu=not args.cpu_only,
    )
    write_immutable(output, receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["decision"] == "PASS_G1_PILOT_V2_12_OF_12" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG1PilotV2Error",
    "build_receipt",
    "load_config",
    "random_pair_batches",
    "replay_candidate",
    "search_arm_galaxy",
    "structured_pair_batches",
    "validate_receipt",
]
