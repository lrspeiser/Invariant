"""G1 pilot: three-arm local gravity-formula search on twelve SPARC galaxies.

The proposer never receives confirmation galaxies.  Candidate structure and one local
diagnostic acceleration are evaluated by contiguous radial cross-validation.  GPU float32 is
only a slack prefilter; CPU float64 replay makes every admission decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_baryonic_interpolation_screen import (
    FAMILY_SIZE,
    _digits_from_ordinals,
    decode_ordinal,
    render_candidate,
)
from .gravity_g0_experiment import (
    GravityG0Error,
    _empirical_rar,
    _galaxy_arrays,
    _newtonian,
    _nfw_out_of_fold,
    _wrong_high_acceleration,
    radial_folds,
    score_predictions,
)
from .gravity_g0_experiment import (
    load_config as load_g0_config,
)
from .gravity_g0_experiment import (
    validate_receipt as validate_g0_receipt,
)
from .pseudorandom_ordinal import PseudorandomChunkSchedule
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import Galaxy, Population, assemble

SCHEMA = "invariant-gravity-g1-pilot-receipt-1.0"
CONFIG_SCHEMA = "invariant-gravity-g1-pilot-config-1.0"
CONFIG_PATH = "configs/gravity_g1_pilot.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g1_pilot.py"
TEST_PATH = "tests/test_gravity_g1_pilot.py"
OUTPUT_PATH = "runs/gravity/g1-pilot/receipt-v1.json"
FIRST_GALAXY_OUTPUT_PATH = "runs/gravity/g1-pilot/first-galaxy-counterexample-v1.json"

COEFFICIENT_SLOTS = 10
CREATIVE_SLOTS = 8
CREATIVE_FAMILIES = 4
CREATIVE_RADIX_SIZE = 7**CREATIVE_SLOTS
CREATIVE_SIZE = CREATIVE_FAMILIES * CREATIVE_RADIX_SIZE
FAILURE_NAMES = (
    "invalid_domain",
    "fails_newtonian",
    "fails_wrong_law",
    "fails_empirical_rar",
    "fails_nfw_ceiling",
    "fails_two_sigma_coverage",
    "gpu_slack_survivor",
)


class GravityG1PilotError(ValueError):
    """The frozen G1 pilot contract, search, or receipt is inconsistent."""


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise GravityG1PilotError(f"JSON root is not an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_file_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(value):
        raise GravityG1PilotError("non-finite value reached a G1 receipt boundary")
    return format(value, ".12e")


def _binding(root: Path, relative: str) -> dict[str, str]:
    return {"normalized_text_sha256": _normalized_file_sha256(root / relative), "path": relative}


def _population_features(population: Population) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rows: list[np.ndarray] = []
    for galaxy in population.exploration:
        arrays = _galaxy_arrays(galaxy)
        radius = arrays["radius"]
        vbar2 = arrays["vbar2"]
        gas = np.asarray([float(value) for value in galaxy.v_gas])
        disk = np.asarray([float(value) for value in galaxy.v_disk])
        bulge = np.asarray([float(value) for value in galaxy.v_bul])
        component_sum = np.abs(gas * gas) + 0.5 * disk * disk + 0.7 * bulge * bulge
        names.append(galaxy.name)
        rows.append(
            np.asarray(
                [
                    np.log1p(galaxy.count),
                    np.log(float(galaxy.distance_mpc)),
                    np.log(float(radius.max() / radius.min())),
                    np.log(float(np.median(vbar2 / radius))),
                    float(np.sum(np.abs(gas * gas)) / np.sum(component_sum)),
                    float(np.sum(0.7 * bulge * bulge) / np.sum(component_sum)),
                    np.log(float(np.median(0.5 * disk * disk / radius)) + 1e-12),
                ],
                dtype=np.float64,
            )
        )
    return names, np.vstack(rows)


def select_pilot_galaxies(population: Population, count: int = 12) -> tuple[str, ...]:
    """Target-blind deterministic farthest-point selection in baryonic feature space."""

    names, features = _population_features(population)
    if not 1 <= count <= len(names):
        raise GravityG1PilotError("pilot galaxy count is outside the exploration population")
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    scale = np.where(mad > 0, 1.4826 * mad, np.std(features, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise GravityG1PilotError("pilot feature standardization is singular")
    standardized = (features - median) / scale
    first = max(
        range(len(names)),
        key=lambda index: (float(np.sum(standardized[index] ** 2)), names[index]),
    )
    selected = [first]
    while len(selected) < count:
        remaining = [index for index in range(len(names)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                float(
                    min(
                        np.sum((standardized[index] - standardized[chosen]) ** 2)
                        for chosen in selected
                    )
                ),
                names[index],
            ),
        )
        selected.append(next_index)
    return tuple(names[index] for index in selected)


def load_config(root: Path) -> Mapping[str, Any]:
    """Load G1 and fail closed unless its G0 PASS and pilot selection replay."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG1PilotError("G1 pilot config schema changed")
    g0_binding = config.get("g0_binding")
    if not isinstance(g0_binding, Mapping):
        raise GravityG1PilotError("G1 pilot has no G0 binding")
    g0_path = root / str(g0_binding.get("path"))
    if _file_sha256(g0_path) != g0_binding.get("file_sha256"):
        raise GravityG1PilotError("G1 pilot G0 file binding changed")
    g0_receipt = _load_json(g0_path)
    try:
        validate_g0_receipt(g0_receipt, root=root)
        load_g0_config(root)
    except GravityG0Error as error:
        raise GravityG1PilotError("G1 pilot inherited an invalid G0 contract") from error
    if g0_receipt.get("content_sha256") != g0_binding.get("content_sha256"):
        raise GravityG1PilotError("G1 pilot G0 content binding changed")
    if g0_receipt.get("decision") != g0_binding.get("required_decision"):
        raise GravityG1PilotError("G1 pilot is not authorized by G0")
    population = assemble(root)
    selected = select_pilot_galaxies(population)
    declared = tuple(config["pilot_selection"]["galaxies_in_selection_order"])
    if selected != declared:
        raise GravityG1PilotError("target-blind pilot selection replay changed")
    arms = config.get("arms")
    if not isinstance(arms, list) or [item.get("id") for item in arms] != [
        "structured_occam",
        "pseudorandom_permutation",
        "creativity_guided",
    ]:
        raise GravityG1PilotError("G1 pilot arm inventory changed")
    if int(arms[2]["canonical_space_size"]) != CREATIVE_SIZE:
        raise GravityG1PilotError("G1 creative grammar size changed")
    return config


def _decode_radix(values: np.ndarray, *, base: int, slots: int, offset: int) -> np.ndarray:
    work = values.copy()
    result = np.empty((values.size, slots), dtype=np.int8)
    for slot in range(slots):
        result[:, slot] = (work % base).astype(np.int8) + offset
        work //= base
    return result


def _encode_parent_ordinals(beta: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    result = beta.astype(np.int64) * 7**COEFFICIENT_SLOTS
    place = 1
    for slot in range(COEFFICIENT_SLOTS):
        result += (coefficients[:, slot].astype(np.int64) + 3) * place
        place *= 7
    return result


def structured_batches(total: int, batch_size: int) -> Iterator[np.ndarray]:
    """Complexity shells {-1,0,1}, then new points in {-2,...,2}."""

    if not 1 <= total <= FAMILY_SIZE:
        raise GravityG1PilotError("structured candidate count is outside the parent grammar")
    produced = 0
    inner_size = 4 * 3**COEFFICIENT_SLOTS
    for start in range(0, min(total, inner_size), batch_size):
        stop = min(total, inner_size, start + batch_size)
        positions = np.arange(start, stop, dtype=np.int64)
        beta = positions // 3**COEFFICIENT_SLOTS
        coefficients = _decode_radix(
            positions % 3**COEFFICIENT_SLOTS,
            base=3,
            slots=COEFFICIENT_SLOTS,
            offset=-1,
        )
        ordinals = _encode_parent_ordinals(beta, coefficients)
        produced += ordinals.size
        yield ordinals
    raw = 0
    outer_size = 4 * 5**COEFFICIENT_SLOTS
    while produced < total and raw < outer_size:
        positions = np.arange(raw, min(raw + batch_size, outer_size), dtype=np.int64)
        raw += positions.size
        beta = positions // 5**COEFFICIENT_SLOTS
        coefficients = _decode_radix(
            positions % 5**COEFFICIENT_SLOTS,
            base=5,
            slots=COEFFICIENT_SLOTS,
            offset=-2,
        )
        outer = np.any(np.abs(coefficients) == 2, axis=1)
        ordinals = _encode_parent_ordinals(beta[outer], coefficients[outer])
        ordinals = ordinals[: total - produced]
        if ordinals.size:
            produced += ordinals.size
            yield ordinals
    if produced != total:
        raise GravityG1PilotError("structured schedule under-ran its declared count")


def pseudorandom_rational_batches(
    total: int, batch_size: int, seed: str
) -> Iterator[np.ndarray]:
    """Pseudorandom chunks from the disjoint |coefficient|=3 outer shell."""

    schedule = PseudorandomChunkSchedule(FAMILY_SIZE, batch_size, seed)
    produced = 0
    for chunk in schedule.iter():
        ordinals = np.arange(
            chunk["start_ordinal"], chunk["stop_ordinal_exclusive"], dtype=np.int64
        )
        _, coefficients = _digits_from_ordinals(np, ordinals)
        ordinals = ordinals[np.any(np.abs(coefficients) == 3, axis=1)]
        ordinals = ordinals[: total - produced]
        if ordinals.size:
            produced += ordinals.size
            yield ordinals
        if produced == total:
            return
    raise GravityG1PilotError("pseudorandom rational schedule under-ran its declared count")


def creativity_batches(total: int, batch_size: int, seed: str) -> Iterator[np.ndarray]:
    """Pseudorandom nonoverlapping chunks over the Claude-seeded basis grammar."""

    if not 1 <= total <= CREATIVE_SIZE:
        raise GravityG1PilotError("creative candidate count is outside its grammar")
    schedule = PseudorandomChunkSchedule(CREATIVE_SIZE, batch_size, seed)
    produced = 0
    for chunk in schedule.iter():
        ordinals = np.arange(
            chunk["start_ordinal"], chunk["stop_ordinal_exclusive"], dtype=np.int64
        )
        ordinals = ordinals[: total - produced]
        produced += ordinals.size
        yield ordinals
        if produced == total:
            return
    raise GravityG1PilotError("creative schedule under-ran its declared count")


def _rational_h(xp: Any, ordinals: Any, y: Any, dtype: Any) -> tuple[Any, Any]:
    beta_index, coefficients = _digits_from_ordinals(xp, ordinals)
    u = y.astype(dtype) ** dtype(-0.5)
    numerator = xp.ones((ordinals.size, y.size), dtype=dtype)
    denominator = xp.ones_like(numerator)
    power = xp.ones_like(u, dtype=dtype)
    for slot in range(5):
        power *= u
        numerator += coefficients[:, slot, None].astype(dtype) * power[None, :]
        denominator += coefficients[:, 5 + slot, None].astype(dtype) * power[None, :]
    valid = xp.all(xp.isfinite(denominator) & (xp.abs(denominator) > dtype(1e-12)), axis=1)
    ratio = numerator / xp.where(xp.abs(denominator) > dtype(1e-12), denominator, dtype(1))
    valid &= xp.all(xp.isfinite(ratio) & (ratio > dtype(1e-12)), axis=1)
    safe = xp.where(ratio > dtype(1e-12), ratio, dtype(1))
    h = xp.where(
        beta_index[:, None] == 0,
        xp.cbrt(safe),
        xp.where(
            beta_index[:, None] == 1,
            xp.sqrt(safe),
            xp.where(beta_index[:, None] == 2, safe, safe * safe),
        ),
    )
    valid &= xp.all(xp.isfinite(h), axis=1)
    return h, valid


def _creative_basis_numpy(y: np.ndarray) -> np.ndarray:
    basis = np.empty((CREATIVE_FAMILIES, CREATIVE_SLOTS, y.size), dtype=np.float64)
    sqrt_y = np.sqrt(y)
    u = 1.0 / sqrt_y
    t = 1.0 / (1.0 + sqrt_y)
    for slot in range(CREATIVE_SLOTS):
        basis[0, slot] = math.comb(7, slot) * t**slot * (1.0 - t) ** (7 - slot)
        basis[1, slot] = np.exp(-slot * sqrt_y)
        scale = 2.0 ** (slot - 3)
        basis[2, slot] = (2.0 / np.pi) * np.arctan(scale * u)
        basis[3, slot] = (1.0 - np.exp(-scale * u)) ** (1 + slot % 4)
    return basis


def _creative_h(
    xp: Any, ordinals: Any, creative_basis: Any, dtype: Any
) -> tuple[Any, Any]:
    family = (ordinals // CREATIVE_RADIX_SIZE).astype(xp.int8)
    work = (ordinals % CREATIVE_RADIX_SIZE).astype(xp.int64)
    coefficients = xp.empty((ordinals.size, CREATIVE_SLOTS), dtype=xp.int8)
    for slot in range(CREATIVE_SLOTS):
        coefficients[:, slot] = (work % 7 - 3).astype(xp.int8)
        work //= 7
    z = xp.empty((ordinals.size, creative_basis.shape[2]), dtype=dtype)
    for family_id in range(CREATIVE_FAMILIES):
        indices = xp.nonzero(family == family_id)[0]
        if int(indices.size):
            z[indices] = coefficients[indices].astype(dtype) @ creative_basis[family_id].astype(
                dtype
            )
    h = xp.log1p(xp.exp(-xp.abs(z))) + xp.maximum(z, dtype(0))
    valid = xp.all(xp.isfinite(h) & (h > dtype(1e-12)), axis=1)
    return h, valid


def _fit_amplitude_numpy(
    shape: np.ndarray, arrays: Mapping[str, np.ndarray], training: Sequence[int]
) -> float:
    train = np.asarray(training, dtype=np.int64)
    target = arrays["vobs"][train] ** 2 - arrays["vbar2"][train]
    sigma_v2 = (
        2.0 * arrays["vobs"][train] * arrays["sigma"][train] + arrays["sigma"][train] ** 2
    )
    weights = 1.0 / np.maximum(sigma_v2, np.finfo(np.float64).tiny) ** 2
    denominator = float(np.sum(weights * shape[train] ** 2))
    if denominator == 0.0:
        return 0.0
    return max(0.0, float(np.sum(weights * shape[train] * target) / denominator))


def _baseline_contract(galaxy: Galaxy, g0_config: Mapping[str, Any]) -> dict[str, Any]:
    arrays = _galaxy_arrays(galaxy)
    folds = radial_folds(
        galaxy.count,
        maximum_folds=int(g0_config["radial_holdout"]["maximum_folds"]),
        minimum_training_rows=int(g0_config["radial_holdout"]["minimum_training_rows"]),
    )
    a0 = float(
        next(item for item in g0_config["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    nfw_config = next(
        item for item in g0_config["baselines"] if item["id"] == "nfw_halo_ceiling"
    )
    predictions = {
        "newtonian_baryons": _newtonian(arrays["vbar2"]),
        "empirical_rar": _empirical_rar(arrays["radius"], arrays["vbar2"], a0),
        "wrong_high_acceleration_boost": _wrong_high_acceleration(
            arrays["radius"], arrays["vbar2"], a0
        ),
    }
    predictions["nfw_halo_ceiling"], nfw_fits = _nfw_out_of_fold(
        arrays, folds, int(nfw_config["scale_radius_grid_size"])
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        held = np.asarray(fold.holdout, dtype=np.int64)
        scores = {
            name: score_predictions(values[held], arrays["vobs"][held], arrays["sigma"][held])
            for name, values in predictions.items()
        }
        fold_rows.append(
            {
                "fold": fold,
                "scores": scores,
                "thresholds": {
                    "newtonian": float(scores["newtonian_baryons"]["chi_square"]),
                    "wrong": float(scores["wrong_high_acceleration_boost"]["chi_square"]),
                    "rar": float(scores["empirical_rar"]["chi_square"]),
                    "nfw": float(scores["nfw_halo_ceiling"]["chi_square"])
                    + 2.0 * len(fold.holdout),
                    "coverage_count": math.ceil(
                        min(
                            0.9,
                            float(scores["nfw_halo_ceiling"]["coverage_two_sigma"]),
                        )
                        * len(fold.holdout)
                        - 1e-12
                    ),
                },
            }
        )
    aggregate = {
        name: score_predictions(values, arrays["vobs"], arrays["sigma"])
        for name, values in predictions.items()
    }
    return {
        "aggregate": aggregate,
        "arrays": arrays,
        "folds": folds,
        "fold_rows": fold_rows,
        "nfw_fits": nfw_fits,
        "predictions": predictions,
    }


def _score_batch(
    xp: Any,
    h: Any,
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
    shape = h * radius[None, :]
    target = vobs * vobs - vbar2
    sigma_v2 = 2 * vobs * sigma + sigma * sigma
    weights = 1.0 / xp.maximum(sigma_v2, dtype(np.finfo(np.float32).tiny)) ** 2
    total_chi = xp.zeros(h.shape[0], dtype=dtype)
    total_coverage = xp.zeros(h.shape[0], dtype=xp.int16)
    passes_newton = domain_valid.copy()
    passes_wrong = domain_valid.copy()
    passes_rar = domain_valid.copy()
    passes_nfw = domain_valid.copy()
    passes_coverage = domain_valid.copy()
    for row in contract["fold_rows"]:
        fold = row["fold"]
        train = xp.asarray(fold.training, dtype=xp.int64)
        held = xp.asarray(fold.holdout, dtype=xp.int64)
        denominator = xp.sum(weights[train][None, :] * shape[:, train] ** 2, axis=1)
        numerator = xp.sum(
            weights[train][None, :] * shape[:, train] * target[train][None, :], axis=1
        )
        amplitude = xp.maximum(
            dtype(0),
            numerator / xp.where(denominator > dtype(0), denominator, dtype(1)),
        )
        prediction = xp.sqrt(
            xp.maximum(vbar2[held][None, :] + amplitude[:, None] * shape[:, held], dtype(0))
        )
        residual = (prediction - vobs[held][None, :]) / sigma[held][None, :]
        fold_chi = xp.sum(residual * residual, axis=1)
        fold_coverage = xp.sum(xp.abs(residual) <= dtype(2), axis=1).astype(xp.int16)
        total_chi += fold_chi
        total_coverage += fold_coverage
        thresholds = row["thresholds"]
        slack = lambda value: dtype(value * (1.0 + relative_slack) + absolute_slack)
        passes_newton &= fold_chi < slack(thresholds["newtonian"])
        passes_wrong &= fold_chi < slack(thresholds["wrong"])
        passes_rar &= fold_chi <= slack(thresholds["rar"])
        passes_nfw &= fold_chi <= slack(thresholds["nfw"])
        passes_coverage &= fold_coverage >= max(
            0, int(thresholds["coverage_count"]) - coverage_slack
        )
    aggregate = contract["aggregate"]
    aggregate_coverage_required = math.ceil(
        min(0.9, float(aggregate["nfw_halo_ceiling"]["coverage_two_sigma"]))
        * arrays["vobs"].size
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
        float(aggregate["nfw_halo_ceiling"]["chi_square"])
        + 2.0 * arrays["vobs"].size
    )
    passes_coverage &= total_coverage >= max(
        0, aggregate_coverage_required - coverage_slack
    )
    survivor = (
        passes_newton & passes_wrong & passes_rar & passes_nfw & passes_coverage
    )
    reason = xp.full(h.shape[0], 6, dtype=xp.int8)
    reason = xp.where(~passes_coverage, 5, reason)
    reason = xp.where(~passes_nfw, 4, reason)
    reason = xp.where(~passes_rar, 3, reason)
    reason = xp.where(~passes_wrong, 2, reason)
    reason = xp.where(~passes_newton, 1, reason)
    reason = xp.where(~domain_valid, 0, reason)
    return total_chi, survivor, reason


def _best_rows(
    xp: Any, scores: Any, mask: Any, ordinals: Any, limit: int
) -> list[tuple[float, int]]:
    indices = xp.nonzero(mask & xp.isfinite(scores))[0]
    if not int(indices.size):
        return []
    take = min(limit, int(indices.size))
    local = indices[xp.argpartition(scores[indices], take - 1)[:take]]
    host_scores = scores[local].get() if xp is not np else scores[local]
    host_ordinals = ordinals[local].get() if xp is not np else ordinals[local]
    return sorted(
        (float(score), int(ordinal))
        for score, ordinal in zip(host_scores, host_ordinals, strict=True)
    )


def _merge_best(
    current: list[tuple[float, int]], new: list[tuple[float, int]], limit: int
) -> list[tuple[float, int]]:
    by_ordinal: dict[int, float] = {ordinal: score for score, ordinal in current}
    for score, ordinal in new:
        by_ordinal[ordinal] = min(score, by_ordinal.get(ordinal, math.inf))
    return sorted((score, ordinal) for ordinal, score in by_ordinal.items())[:limit]


def _creative_candidate(ordinal: int, config: Mapping[str, Any]) -> dict[str, Any]:
    if not 0 <= ordinal < CREATIVE_SIZE:
        raise GravityG1PilotError("creative ordinal is outside its grammar")
    family_id = ordinal // CREATIVE_RADIX_SIZE
    value = ordinal % CREATIVE_RADIX_SIZE
    coefficients = []
    for _ in range(CREATIVE_SLOTS):
        coefficients.append(value % 7 - 3)
        value //= 7
    family = config["arms"][2]["basis_families"][family_id]
    return {
        "basis_definition": family["definition"],
        "basis_family": family["name"],
        "basis_family_id": family_id,
        "coefficients": coefficients,
        "origin": family["origin"],
        "source_hypotheses": family["source_hypotheses"],
    }


def _render(arm_id: str, ordinal: int, config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if arm_id in {"structured_occam", "pseudorandom_permutation"}:
        candidate = decode_ordinal(ordinal)
        h_text = render_candidate(candidate).replace("nu(y)", "h(y)")
        return f"V_pred^2 = V_bar^2 + A*r*h(y); {h_text}", candidate
    candidate = _creative_candidate(ordinal, config)
    coefficients = ",".join(str(value) for value in candidate["coefficients"])
    formula = (
        "V_pred^2 = V_bar^2 + A*r*softplus(sum_j(c_j*phi_j(y))); "
        f"basis={candidate['basis_family']}; c=[{coefficients}]"
    )
    return formula, candidate


def replay_candidate(
    galaxy: Galaxy,
    arm_id: str,
    ordinal: int,
    config: Mapping[str, Any],
    g0_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Authoritative CPU-float64 replay with training-only fold constants."""

    contract = _baseline_contract(galaxy, g0_config)
    arrays = contract["arrays"]
    y = arrays["vbar2"] / arrays["radius"] / float(config["candidate_shell"]["a0_km2_s2_kpc"])
    if arm_id in {"structured_occam", "pseudorandom_permutation"}:
        h_matrix, valid = _rational_h(
            np, np.asarray([ordinal]), y, np.float64
        )
    else:
        h_matrix, valid = _creative_h(
            np,
            np.asarray([ordinal]),
            _creative_basis_numpy(y),
            np.float64,
        )
    h = h_matrix[0]
    if not bool(valid[0]):
        return {"admitted": False, "failure": "invalid_domain", "ordinal": ordinal}
    shape = arrays["radius"] * h
    predictions = np.empty_like(arrays["vobs"])
    fold_results = []
    all_pass = True
    failure_set: set[str] = set()
    for row in contract["fold_rows"]:
        fold = row["fold"]
        amplitude = _fit_amplitude_numpy(shape, arrays, fold.training)
        held = np.asarray(fold.holdout, dtype=np.int64)
        predictions[held] = np.sqrt(
            np.maximum(arrays["vbar2"][held] + amplitude * shape[held], 0.0)
        )
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
        fold_results.append(
            {
                "A_km2_s2_kpc": _metric(amplitude),
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
    full_amplitude = _fit_amplitude_numpy(shape, arrays, tuple(range(galaxy.count)))
    formula, candidate = _render(arm_id, ordinal, config)
    ir = {
        "arm": arm_id,
        "candidate": candidate,
        "local_constant": "A_km2_s2_kpc",
        "ordinal": ordinal,
        "shell": "vbar2+A*r*h(y)",
    }
    grammar = next(item for item in config["arms"] if item["id"] == arm_id)
    grammar_size = int(
        grammar.get("canonical_space_size", grammar.get("canonical_parent_space_size"))
    )
    formula_bytes = len(canonical_json_bytes(ir))
    return {
        "admitted": all_pass,
        "aggregate_checks": aggregate_checks,
        "aggregate_score": aggregate_score,
        "candidate": candidate,
        "description_length": {
            "canonical_formula_bytes": formula_bytes,
            "grammar_address_bits": math.ceil(math.log2(grammar_size)),
            "local_constant_bits": 64,
            "total_bits": 8 * formula_bytes + 64 + math.ceil(math.log2(grammar_size)),
        },
        "failure_obligations": sorted(failure_set),
        "folds": fold_results,
        "formula": formula,
        "full_data_A_for_reporting_only_km2_s2_kpc": _metric(full_amplitude),
        "ordinal": ordinal,
        "prediction_sha256": canonical_sha256(
            [format(float(value), ".15e") for value in predictions]
        ),
    }


def _arm_batches(
    arm: Mapping[str, Any], total: int, batch_size: int
) -> Iterator[np.ndarray]:
    arm_id = arm["id"]
    if arm_id == "structured_occam":
        return structured_batches(total, batch_size)
    if arm_id == "pseudorandom_permutation":
        return pseudorandom_rational_batches(total, batch_size, str(arm["seed"]))
    if arm_id == "creativity_guided":
        return creativity_batches(total, batch_size, str(arm["seed"]))
    raise GravityG1PilotError(f"unknown G1 arm: {arm_id}")


def search_arm_galaxy(
    galaxy: Galaxy,
    arm: Mapping[str, Any],
    config: Mapping[str, Any],
    g0_config: Mapping[str, Any],
    *,
    candidate_count: int,
    use_gpu: bool,
) -> dict[str, Any]:
    """Search one arm/galaxy trial and CPU-replay its slack survivors."""

    prefilter = config["gpu_prefilter"]
    batch_size = int(prefilter["batch_size"])
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
    creative_basis = xp.asarray(_creative_basis_numpy(y_numpy))
    failure_counts = np.zeros(len(FAILURE_NAMES), dtype=np.int64)
    best_valid: list[tuple[float, int]] = []
    best_prefilter: list[tuple[float, int]] = []
    evaluated = 0
    started = time.perf_counter()
    for host_ordinals in _arm_batches(arm, candidate_count, batch_size):
        ordinals = xp.asarray(host_ordinals)
        if arm["id"] in {"structured_occam", "pseudorandom_permutation"}:
            h, valid = _rational_h(xp, ordinals, y, xp.float32)
        else:
            h, valid = _creative_h(xp, ordinals, creative_basis, xp.float32)
        scores, survivor, reasons = _score_batch(
            xp,
            h,
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
        raise GravityG1PilotError("G1 arm accounting changed")
    replayed = [
        replay_candidate(galaxy, str(arm["id"]), ordinal, config, g0_config)
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
    replayed_ordinals = {int(item["ordinal"]) for item in replayed}
    for score, ordinal in best_valid[:16]:
        if ordinal in replayed_ordinals:
            item = next(item for item in replayed if int(item["ordinal"]) == ordinal)
        else:
            item = replay_candidate(galaxy, str(arm["id"]), ordinal, config, g0_config)
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
        "arm": arm["id"],
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
    """Run the G1 pilot; only a full 360-million trial can PASS."""

    root = root.resolve()
    config = load_config(root)
    g0_config = load_g0_config(root)
    population = assemble(root)
    by_name = {galaxy.name: galaxy for galaxy in population.exploration}
    declared_names = list(config["pilot_selection"]["galaxies_in_selection_order"])
    selected_names = declared_names[:galaxy_limit] if galaxy_limit is not None else declared_names
    if not selected_names:
        raise GravityG1PilotError("G1 pilot selected no galaxies")
    arm_rows = []
    galaxy_rows = []
    total_candidates = 0
    total_confirmation_accesses = 0
    for name in selected_names:
        galaxy = by_name[name]
        trials = []
        for arm in config["arms"]:
            count = int(
                arm["candidate_count_per_galaxy"] if candidate_count is None else candidate_count
            )
            trial = search_arm_galaxy(
                galaxy,
                arm,
                config,
                g0_config,
                candidate_count=count,
                use_gpu=use_gpu,
            )
            trials.append(trial)
            arm_rows.append(trial)
            total_candidates += count
            total_confirmation_accesses += trial["confirmation_evaluator_access_count"]
        admitted = [
            candidate
            for trial in trials
            for candidate in trial["cpu_fp64_admitted_pareto"]
        ]
        admitted.sort(
            key=lambda item: (
                float(item["aggregate_score"]["chi_square"]),
                int(item["description_length"]["total_bits"]),
                str(item["formula"]),
            )
        )
        galaxy_rows.append(
            {
                "covered": bool(admitted),
                "galaxy": name,
                "point_count": galaxy.count,
                "retained_pareto": admitted[:64],
                "trials": trials,
            }
        )
    declared_per_trial = {
        arm["id"]: int(arm["candidate_count_per_galaxy"]) for arm in config["arms"]
    }
    full_run = (
        selected_names == declared_names
        and candidate_count is None
        and use_gpu
        and all(
            trial["candidate_count"] == declared_per_trial[trial["arm"]]
            for trial in arm_rows
        )
    )
    covered = sum(row["covered"] for row in galaxy_rows)
    passed = (
        full_run
        and covered == len(declared_names)
        and total_confirmation_accesses == 0
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G1_PILOT",
        "decision": (
            "PASS_G1_PILOT_12_OF_12" if passed else "BLOCK_G1_PILOT_UNCOVERED_OR_INCOMPLETE"
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
            "confirmation_evaluator_accesses": total_confirmation_accesses,
            "covered_pilot_galaxies": covered,
            "pilot_galaxies": len(selected_names),
        },
        "galaxies": galaxy_rows,
        "limitations": [
            "G1 local constants are diagnostic and are forbidden from surviving the G3 universal-law gate.",
            "Repeated cross-validation within an exploration galaxy estimates interpolation robustness; G3/G4 whole-galaxy tests are required for population generalization.",
            "An LLM origin assessment is lineage metadata, not a prior-art judgment or novelty claim.",
            "A pilot PASS covers twelve selected exploration galaxies only; it is not a 139-galaxy atlas.",
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
        raise GravityG1PilotError("G1 pilot receipt schema changed")
    supplied = receipt.get("content_sha256")
    body = dict(receipt)
    body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG1PilotError("G1 pilot receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG1PilotError("G1 pilot receipt config binding changed")
    bindings = receipt.get("source_bindings", {})
    for key, relative in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if bindings.get(key) != _binding(root, relative):
            raise GravityG1PilotError(f"G1 pilot {key} binding changed")
    counts = receipt.get("counts", {})
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG1PilotError("G1 pilot receipt reports confirmation access")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise GravityG1PilotError(f"refusing to overwrite a different G1 receipt: {path}")
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
    return 0 if receipt["decision"] == "PASS_G1_PILOT_12_OF_12" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIRST_GALAXY_OUTPUT_PATH",
    "GravityG1PilotError",
    "build_receipt",
    "creativity_batches",
    "load_config",
    "pseudorandom_rational_batches",
    "replay_candidate",
    "search_arm_galaxy",
    "select_pilot_galaxies",
    "structured_batches",
    "validate_receipt",
]
