"""Exhaustive screened-nonlocal boundary grammar over already exposed data.

The raw grammar contains exactly 4**10 candidates. Every raw ordinal is addressed.
Candidates that differ only in a lensing branch, or whose nonlocal amplitude is zero,
are counted before their present dynamics-equivalent behavior is evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_g0_experiment import (
    _galaxy_arrays,
)
from sigma_theory_compiler.gravity_g4_nonlocal_profile_law_construction import (
    _kernel_matrix,
    _log_radius_cell_widths,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item57_independent_galaxy_gate import (
    ITEM5_SOURCE_PATH,
    _parse_existing_target,
    _parse_predictor_surface_density,
    _predictions,
)
from sigma_theory_compiler.gravity_item57_independent_galaxy_gate import (
    load_config as load_item57_config,
)
from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    _cumulative_mass,
    _member_mass,
    prepare_packets,
)
from sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate import (
    load_config as load_item59_config,
)
from sigma_theory_compiler.sparc_full_sample import assemble

CONFIG_PATH = Path("configs/screened_nonlocal_boundary_permutation_v0.json")
ITEM59_RESULT = Path("runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json")
ITEM57_SOURCE = Path("runs/gravity/roadmap/item-57-independent-galaxy-gate-v1-source")


class ScreenedNonlocalPermutationError(RuntimeError):
    """Raised when the frozen grammar, data boundary, or exhaustive traversal changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-screened-nonlocal-boundary-permutation-config-0.1"
        or config.get("status") != "scientific_freeze_before_exposed_response_evaluation"
    ):
        raise ScreenedNonlocalPermutationError("unexpected screened-nonlocal config")
    freeze = str(config.get("scientific_freeze_commit", ""))
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise ScreenedNonlocalPermutationError("scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise ScreenedNonlocalPermutationError("invalid scientific freeze marker")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != str(expected):
            raise ScreenedNonlocalPermutationError(f"scientific dependency changed: {relative}")
    grammar = config["grammar"]
    order = list(map(str, grammar["factor_order"]))
    if len(order) != 10 or len(set(order)) != 10:
        raise ScreenedNonlocalPermutationError("factor order changed")
    if any(len(grammar[name]) != 4 for name in order):
        raise ScreenedNonlocalPermutationError("every grammar factor must have four values")
    expected = math.prod(len(grammar[name]) for name in order)
    if expected != 1_048_576 or expected != int(grammar["raw_ordinal_count"]):
        raise ScreenedNonlocalPermutationError("raw grammar size changed")
    if not grammar["all_factors_exhausted"] or grammar["sampling_allowed"]:
        raise ScreenedNonlocalPermutationError("exhaustive traversal was weakened")
    boundary = config["data_contract"]
    if (
        not boundary["only_previously_exposed_responses"]
        or boundary["new_target_queries_allowed"] != 0
        or boundary["sealed_confirmation_accesses_allowed"] != 0
        or boundary["direct_lensing_target_rows_allowed"] != 0
    ):
        raise ScreenedNonlocalPermutationError("sealed or new response access entered the grammar")
    claims = config["claim_policy"]
    if (
        not claims["evaluation_is_exploratory"]
        or claims["evaluation_is_fresh_confirmation"]
        or claims["former_xcop_confirmation_is_fresh_for_new_grammar"]
        or claims["single_counterexample_terminal"]
        or claims["counterexample_count_alone_terminal"]
        or claims["finite_sample_prunes_family"]
    ):
        raise ScreenedNonlocalPermutationError("claim or counterexample policy changed")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def decode_ordinal(ordinal: int, config: Mapping[str, Any]) -> dict[str, int]:
    total = int(config["grammar"]["raw_ordinal_count"])
    if ordinal < 0 or ordinal >= total:
        raise ScreenedNonlocalPermutationError("candidate ordinal outside the frozen grammar")
    indices: dict[str, int] = {}
    remainder = int(ordinal)
    for name in reversed(config["grammar"]["factor_order"]):
        size = len(config["grammar"][name])
        indices[str(name)] = remainder % size
        remainder //= size
    if remainder:
        raise ScreenedNonlocalPermutationError("candidate ordinal decode overflow")
    return indices


def encode_ordinal(indices: Mapping[str, int], config: Mapping[str, Any]) -> int:
    ordinal = 0
    for name in config["grammar"]["factor_order"]:
        size = len(config["grammar"][name])
        index = int(indices[str(name)])
        if index < 0 or index >= size:
            raise ScreenedNonlocalPermutationError(f"factor index outside grammar: {name}")
        ordinal = ordinal * size + index
    return ordinal


def _raw_stream_digest(config: Mapping[str, Any]) -> str:
    ordinals = np.arange(int(config["grammar"]["raw_ordinal_count"]), dtype="<u4")
    return hashlib.sha256(ordinals.tobytes()).hexdigest()


def behavior_representatives(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return every distinct dynamics behavior and its raw multiplicity.

    Lensing does not enter the exposed dynamics/thermodynamics tests. With alpha=0,
    every kernel and transition choice also disappears exactly.
    """

    names = list(map(str, config["grammar"]["factor_order"]))
    rows: dict[str, list[int]] = {name: [] for name in names}
    ordinals: list[int] = []
    multiplicities: list[int] = []
    for values in itertools.product(range(4), repeat=9):
        indices = dict(zip(names[:-1], values, strict=True))
        indices[names[-1]] = 0
        if indices["alpha"] == 0:
            if any(
                indices[name] != 0
                for name in names[:-1]
                if name not in {"base_law", "alpha"}
            ):
                continue
            multiplicity = 4**8
        else:
            multiplicity = 4
        ordinal = encode_ordinal(indices, config)
        ordinals.append(ordinal)
        multiplicities.append(multiplicity)
        for name in names:
            rows[name].append(indices[name])
    result = {name: np.asarray(values, dtype=np.int16) for name, values in rows.items()}
    result["representative_ordinal"] = np.asarray(ordinals, dtype=np.int64)
    result["raw_multiplicity"] = np.asarray(multiplicities, dtype=np.int64)
    if (
        len(ordinals) != 196_612
        or int(np.sum(result["raw_multiplicity"])) != int(
            config["grammar"]["raw_ordinal_count"]
        )
    ):
        raise ScreenedNonlocalPermutationError("behavioral equivalence accounting changed")
    return result


def _base_factor(y: np.ndarray, base_index: np.ndarray | int) -> np.ndarray:
    safe = np.maximum(np.asarray(y, dtype=float), np.finfo(float).tiny)
    options = np.stack(
        [
            np.ones_like(safe),
            1.0 / -np.expm1(-np.sqrt(safe)),
            0.5 + np.sqrt(0.25 + 1.0 / safe),
            np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / np.square(safe)))),
        ]
    )
    return options[np.asarray(base_index)]


def _transition(
    y: np.ndarray,
    compactness: np.ndarray,
    boundary: np.ndarray,
    screen_y: float,
    screen_power: float,
    compactness_threshold: float,
    environment: Mapping[str, Any],
) -> np.ndarray:
    safe_y = np.maximum(np.asarray(y, dtype=float), np.finfo(float).tiny)
    safe_c = np.maximum(np.asarray(compactness, dtype=float), np.finfo(float).tiny)
    high = 1.0 / (1.0 + np.power(safe_y / screen_y, screen_power))
    log_zeta = (
        float(environment["compactness_power"]) * np.log(safe_c / compactness_threshold)
        + float(environment["boundary_power"]) * np.log1p(np.maximum(boundary, 0.0))
    )
    argument = np.clip(float(environment["activation_power"]) * log_zeta, -700.0, 700.0)
    activation = 1.0 / (1.0 + np.exp(-argument))
    result = high * activation
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ScreenedNonlocalPermutationError("transition left its declared domain")
    return result


def _analytic_arrays(
    representatives: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    grammar = config["grammar"]
    gates = config["analytic_gates"]
    base = representatives["base_law"]
    alpha = np.asarray(grammar["alpha"], dtype=float)[representatives["alpha"]]
    y0 = np.asarray(grammar["occupancy_y0"], dtype=float)[representatives["occupancy_y0"]]
    screen_y = np.asarray(grammar["screen_y"], dtype=float)[representatives["screen_y"]]
    screen_power = np.asarray(grammar["screen_power"], dtype=float)[
        representatives["screen_power"]
    ]
    cstar = np.asarray(grammar["compactness_threshold"], dtype=float)[
        representatives["compactness_threshold"]
    ]
    env_index = representatives["environment_shape"]
    compactness_power = np.asarray(
        [row["compactness_power"] for row in grammar["environment_shape"]], dtype=float
    )[env_index]
    boundary_power = np.asarray(
        [row["boundary_power"] for row in grammar["environment_shape"]], dtype=float
    )[env_index]
    activation_power = np.asarray(
        [row["activation_power"] for row in grammar["environment_shape"]], dtype=float
    )[env_index]

    def probe(y_value: float, c_value: float, b_value: float) -> tuple[np.ndarray, np.ndarray]:
        high = 1.0 / (
            1.0 + np.power(y_value / screen_y, screen_power)
        )
        log_zeta = (
            compactness_power * np.log(c_value / cstar)
            + boundary_power * np.log1p(b_value)
        )
        argument = np.clip(activation_power * log_zeta, -700.0, 700.0)
        transitions = high / (1.0 + np.exp(-argument))
        occupancy = y_value / (y_value + y0)
        factor = _base_factor(np.asarray([y_value]), base).reshape(-1)
        factor *= np.exp(2.0 * alpha * transitions * occupancy)
        return factor, transitions

    local_factor, local_transition = probe(
        float(gates["solar_acceleration_to_a0"]),
        float(gates["solar_compactness"]),
        0.0,
    )
    cluster_factor, cluster_transition = probe(
        float(gates["cluster_probe_acceleration_to_a0"]),
        float(gates["cluster_probe_compactness"]),
        float(gates["cluster_probe_boundary_state"]),
    )
    epsilon_min = np.exp(-2.0 * alpha)
    local_pass = (
        np.abs(local_factor - 1.0) <= float(gates["maximum_local_fractional_deviation"])
    )
    bounded_pass = cluster_factor <= float(gates["maximum_total_acceleration_factor"])
    positive_pass = np.isfinite(epsilon_min) & (epsilon_min > 0.0)
    material = np.exp(
        2.0
        * alpha
        * cluster_transition
        * (
            float(gates["cluster_probe_acceleration_to_a0"])
            / (
                float(gates["cluster_probe_acceleration_to_a0"])
                + y0
            )
        )
    ) - 1.0
    material_pass = material >= float(gates["minimum_nonlocal_materiality"])
    controls = alpha == 0.0
    admitted = local_pass & bounded_pass & positive_pass & (controls | material_pass)
    return {
        "admitted": admitted,
        "bounded_pass": bounded_pass,
        "cluster_factor": cluster_factor,
        "cluster_transition": cluster_transition,
        "control": controls,
        "epsilon_min": epsilon_min,
        "local_factor": local_factor,
        "local_pass": local_pass,
        "local_transition": local_transition,
        "material": material,
        "material_pass": material_pass,
        "positive_pass": positive_pass,
        "qualifying": admitted & ~controls,
    }


def build_preflight(root: Path) -> dict[str, Any]:
    config = load_config(root)
    representatives = behavior_representatives(config)
    analytic = _analytic_arrays(representatives, config)
    multiplicity = representatives["raw_multiplicity"]
    summary = {
        name: {
            "behavior_classes": int(np.count_nonzero(values)),
            "raw_candidates": int(np.sum(multiplicity[values])),
        }
        for name, values in (
            ("local_pass", analytic["local_pass"]),
            ("bounded_pass", analytic["bounded_pass"]),
            ("positive_pass", analytic["positive_pass"]),
            ("material_pass", analytic["material_pass"]),
            ("admitted", analytic["admitted"]),
            ("qualifying", analytic["qualifying"]),
        )
    }
    return _content_hashed(
        {
            "schema_version": "invariant-screened-nonlocal-boundary-preflight-0.1",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "contract_digest": _contract_digest(config),
            "raw_candidate_count": int(config["grammar"]["raw_ordinal_count"]),
            "raw_ordinal_stream_sha256": _raw_stream_digest(config),
            "dynamics_behavior_classes": len(representatives["representative_ordinal"]),
            "exact_zero_amplitude_behavior_classes": int(
                np.count_nonzero(analytic["control"])
            ),
            "lensing_branches_per_nonzero_dynamics_behavior": 4,
            "analytic_gates": summary,
            "data_boundary": {
                "response_rows_read": 0,
                "new_target_queries": 0,
                "sealed_confirmation_accesses": 0,
                "direct_lensing_rows": 0,
            },
            "claims": {
                "full_finite_grammar_exhausted": True,
                "infinite_formula_space_exhausted": False,
                "fresh_confirmation": False,
                "alternative_to_gr_established": False,
            },
        }
    )


def write_preflight(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["source_dir"]) / str(config["paths"]["preflight"])
    _write_json(path, build_preflight(root))
    return path


def _boundary_state(radius_kpc: np.ndarray, gbar: np.ndarray, gravity: float, kpc_m: float) -> np.ndarray:
    radius_m = np.asarray(radius_kpc, dtype=float) * kpc_m
    mass = np.maximum.accumulate(
        np.maximum(gbar, np.finfo(float).tiny) * np.square(radius_m) / gravity
    )
    log_r = np.log(radius_kpc)
    log_m = np.log(np.maximum(mass, np.finfo(float).tiny))
    edge = 2 if len(log_r) >= 3 else 1
    slope = np.gradient(log_m, log_r, edge_order=edge)
    curvature = np.abs(np.gradient(slope, log_r, edge_order=edge))
    if np.any(~np.isfinite(curvature)):
        raise ScreenedNonlocalPermutationError("non-finite baryonic boundary state")
    return curvature


def _feature_banks(
    radius_kpc: np.ndarray,
    gbar: np.ndarray,
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    grammar = config["grammar"]
    constants = config59["constants"]
    a0 = float(constants["transition_acceleration_m_s2"])
    kpc_m = float(constants["kiloparsec_m"])
    gravity = float(constants["gravity_si"])
    c = 299_792_458.0
    radius = np.asarray(radius_kpc, dtype=float)
    safe_gbar = np.maximum(np.asarray(gbar, dtype=float), np.finfo(float).tiny)
    y = safe_gbar / a0
    compactness = safe_gbar * radius * kpc_m / c**2
    boundary = _boundary_state(radius, safe_gbar, gravity, kpc_m)
    base = safe_gbar[None, :] * _base_factor(y, np.arange(4))
    log_radius = np.log(radius)
    widths = _log_radius_cell_widths(log_radius)
    kernels = {
        (kernel_index, scale_index): _kernel_matrix(
            log_radius,
            widths,
            str(kernel),
            float(scale),
        )
        for kernel_index, kernel in enumerate(grammar["kernel"])
        for scale_index, scale in enumerate(grammar["log_radius_scale"])
    }
    x_bank = np.empty((64, len(radius)), dtype=float)
    for kernel_index, y0_index, scale_index in itertools.product(range(4), repeat=3):
        q = y / (y + float(grammar["occupancy_y0"][y0_index]))
        bank_index = kernel_index * 16 + y0_index * 4 + scale_index
        x_bank[bank_index] = kernels[(kernel_index, scale_index)] @ q
    t_bank = np.empty((256, len(radius)), dtype=float)
    for screen_index, power_index, cstar_index, env_index in itertools.product(
        range(4), repeat=4
    ):
        bank_index = screen_index * 64 + power_index * 16 + cstar_index * 4 + env_index
        t_bank[bank_index] = _transition(
            y,
            compactness,
            boundary,
            float(grammar["screen_y"][screen_index]),
            float(grammar["screen_power"][power_index]),
            float(grammar["compactness_threshold"][cstar_index]),
            grammar["environment_shape"][env_index],
        )
    if any(np.any(~np.isfinite(values)) for values in (base, x_bank, t_bank)):
        raise ScreenedNonlocalPermutationError("non-finite feature bank")
    return {
        "base": base,
        "boundary": boundary,
        "compactness": compactness,
        "gbar": safe_gbar,
        "radius_kpc": radius,
        "transition": t_bank,
        "x": x_bank,
        "y": y,
    }


def _profile(
    name: str,
    radius: np.ndarray,
    gbar: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, Any]:
    order = np.argsort(radius)
    arrays = [np.asarray(value, dtype=float)[order] for value in (radius, gbar, observed, sigma)]
    if (
        len(arrays[0]) < 3
        or np.any(np.diff(arrays[0]) <= 0.0)
        or np.any(arrays[1] <= 0.0)
        or np.any(arrays[3] <= 0.0)
    ):
        raise ScreenedNonlocalPermutationError(f"invalid galaxy profile: {name}")
    return {
        "name": name,
        "radius_kpc": arrays[0],
        "gbar": arrays[1],
        "observed": arrays[2],
        "sigma": arrays[3],
    }


def _sparc_profiles(root: Path, config: Mapping[str, Any], config59: Mapping[str, Any]) -> list[dict[str, Any]]:
    population = assemble(root)
    kpc_m = float(config59["constants"]["kiloparsec_m"])
    profiles = []
    rows = 0
    for galaxy in population.exploration:
        arrays = _galaxy_arrays(galaxy)
        gbar = arrays["vbar2"] / arrays["radius"] * 1.0e6 / kpc_m
        profiles.append(
            _profile(
                galaxy.name,
                arrays["radius"],
                gbar,
                arrays["vobs"],
                arrays["sigma"],
            )
        )
        rows += galaxy.count
    boundary = config["data_contract"]
    if (
        len(profiles) != int(boundary["sparc_exploration_galaxies"])
        or rows != int(boundary["sparc_exploration_rows"])
    ):
        raise ScreenedNonlocalPermutationError("SPARC exposed population changed")
    return profiles


def _little_things_profiles(
    root: Path, config: Mapping[str, Any], config59: Mapping[str, Any]
) -> list[dict[str, Any]]:
    config57 = load_item57_config(root)
    photometry_manifest = _read_json(root / ITEM57_SOURCE / "photometry-source-manifest.json")
    photometry = {
        str(record["slug"]): record["parsed"] for record in photometry_manifest["records"]
    }
    source = _read_json(root / ITEM5_SOURCE_PATH)
    source_by_slug = {str(record["galaxy"]): record for record in source["records"]}
    kpc_m = float(config59["constants"]["kiloparsec_m"])
    profiles = []
    rows = 0
    for object_row in config57["little_things"]["exploration_objects"]:
        slug = str(object_row["slug"])
        name = str(object_row["vizier_name"])
        record = source_by_slug[slug]
        density_radius, surface_density = _parse_predictor_surface_density(
            root / str(record["predictor"]["path"])
        )
        target = _parse_existing_target(
            root / str(record["target"]["path"]), expected_name=name
        )
        valid = (
            (target["radius"] > 0.0)
            & (target["radius"] <= density_radius[-1])
            & (target["observed"] >= 0.0)
            & (target["sigma"] > 0.0)
        )
        radius = target["radius"][valid]
        baseline, _ = _predictions(
            radius,
            density_radius,
            surface_density,
            photometry[slug],
            config57,
            "nominal",
        )
        gbar = np.square(baseline["newtonian_baryons"]) / radius * 1.0e6 / kpc_m
        profiles.append(
            _profile(
                slug,
                radius,
                gbar,
                target["observed"][valid],
                target["sigma"][valid],
            )
        )
        rows += len(radius)
    boundary = config["data_contract"]
    if (
        len(profiles) != int(boundary["little_things_exploration_galaxies"])
        or rows != int(boundary["little_things_exploration_rows"])
    ):
        raise ScreenedNonlocalPermutationError("LITTLE THINGS exposed population changed")
    return profiles


def _galaxy_bank(
    profiles: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    count = sum(len(profile["radius_kpc"]) for profile in profiles)
    base = np.empty((4, count), dtype=float)
    x = np.empty((64, count), dtype=float)
    transition = np.empty((256, count), dtype=float)
    observed = np.empty(count, dtype=float)
    sigma = np.empty(count, dtype=float)
    radius_factor = np.empty(count, dtype=float)
    weights = np.empty(count, dtype=float)
    slices = {}
    offset = 0
    kpc_m = float(config59["constants"]["kiloparsec_m"])
    for profile in profiles:
        size = len(profile["radius_kpc"])
        slc = slice(offset, offset + size)
        features = _feature_banks(
            np.asarray(profile["radius_kpc"]),
            np.asarray(profile["gbar"]),
            config,
            config59,
        )
        base[:, slc] = features["base"]
        x[:, slc] = features["x"]
        transition[:, slc] = features["transition"]
        observed[slc] = profile["observed"]
        sigma[slc] = profile["sigma"]
        radius_factor[slc] = np.asarray(profile["radius_kpc"]) * kpc_m / 1.0e6
        weights[slc] = 1.0 / (len(profiles) * size)
        slices[str(profile["name"])] = (offset, offset + size)
        offset += size
    return {
        "base": base,
        "observed": observed,
        "profiles": list(profiles),
        "radius_factor": radius_factor,
        "sigma": sigma,
        "slices": slices,
        "transition": transition,
        "weights": weights,
        "x": x,
    }


def _representative_selection(
    representatives: Mapping[str, np.ndarray], selected: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value)[selected]
        for key, value in representatives.items()
    }


def _behavior_indices(representatives: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x_index = (
        representatives["kernel"].astype(np.int64) * 16
        + representatives["occupancy_y0"].astype(np.int64) * 4
        + representatives["log_radius_scale"].astype(np.int64)
    )
    t_index = (
        representatives["screen_y"].astype(np.int64) * 64
        + representatives["screen_power"].astype(np.int64) * 16
        + representatives["compactness_threshold"].astype(np.int64) * 4
        + representatives["environment_shape"].astype(np.int64)
    )
    return x_index, t_index


def _score_galaxy_gpu(
    bank: Mapping[str, Any],
    representatives: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    import cupy as cp

    started = time.perf_counter()
    batch_size = int(config["scoring"]["gpu_batch_size"])
    alpha_grid = cp.asarray(config["grammar"]["alpha"], dtype=cp.float64)
    base_bank = cp.asarray(bank["base"])
    x_bank = cp.asarray(bank["x"])
    t_bank = cp.asarray(bank["transition"])
    radius_factor = cp.asarray(bank["radius_factor"])
    observed = cp.asarray(bank["observed"])
    sigma = cp.asarray(bank["sigma"])
    weights = cp.asarray(bank["weights"])
    x_index, t_index = _behavior_indices(representatives)
    result = np.empty(len(representatives["base_law"]), dtype=float)
    for start in range(0, len(result), batch_size):
        stop = min(start + batch_size, len(result))
        base_index = cp.asarray(representatives["base_law"][start:stop], dtype=cp.int64)
        alpha_index = cp.asarray(representatives["alpha"][start:stop], dtype=cp.int64)
        xb = cp.asarray(x_index[start:stop], dtype=cp.int64)
        tb = cp.asarray(t_index[start:stop], dtype=cp.int64)
        acceleration = base_bank[base_index] * cp.exp(
            2.0 * alpha_grid[alpha_index, None] * t_bank[tb] * x_bank[xb]
        )
        velocity = cp.sqrt(acceleration * radius_factor[None, :])
        residual = (velocity - observed[None, :]) / sigma[None, :]
        result[start:stop] = cp.asnumpy(cp.sum(cp.square(residual) * weights[None, :], axis=1))
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    del base_bank, x_bank, t_bank
    cp.get_default_memory_pool().free_all_blocks()
    return result, elapsed


def _cluster_packets(
    root: Path,
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> list[dict[str, Any]]:
    packets = prepare_packets(root, config59)
    nuisance = dict(config["fixed_xcop_nuisances"])
    variant = {"nuisances": nuisance}
    constants = config59["constants"]
    prepared = []
    for packet in packets:
        density_radius = np.asarray(packet["density_radius_kpc"], dtype=float)
        ne = np.maximum(np.asarray(packet["ne_cm3"], dtype=float), np.finfo(float).tiny)
        target_radii = [float(row["radius_kpc"]) for row in packet["rows"]]
        calc_radius = np.unique(
            np.asarray(
                [
                    *density_radius[
                        (density_radius >= density_radius[0])
                        & (density_radius <= float(packet["anchor"]["radius_kpc"]))
                    ],
                    *target_radii,
                    float(packet["anchor"]["radius_kpc"]),
                ],
                dtype=float,
            )
        )
        calc_ne = np.exp(
            np.interp(np.log(calc_radius), np.log(density_radius), np.log(ne))
        )
        radius_m = calc_radius * float(constants["kiloparsec_m"])
        rho = (
            calc_ne
            * 1.0e6
            * float(constants["mean_molecular_weight_per_electron"])
            * float(constants["proton_mass_kg"])
        )
        gas_mass = _cumulative_mass(radius_m, rho)
        member_mass = _member_mass(
            packet,
            calc_radius,
            gas_mass,
            variant,
            "nominal",
            config59,
        )
        gbar = (
            float(constants["gravity_si"])
            * (gas_mass + member_mass)
            / np.maximum(np.square(radius_m), np.finfo(float).tiny)
        )
        nonthermal = float(nuisance["outer_nonthermal_fraction"])
        radial_power = float(config59["nuisance_grid"]["nonthermal_radial_power"])
        thermal_fraction = 1.0 - nonthermal * (
            calc_radius / float(packet["r500_kpc"])
        ) ** radial_power
        thermal_fraction = np.clip(thermal_fraction, 0.25, 1.0)
        gradient_prefactor = (
            float(constants["mean_molecular_weight"])
            * float(constants["proton_mass_kg"])
            * calc_ne
            * 1.0e6
            * thermal_fraction
        )
        row_indices = np.asarray(
            [int(np.argmin(np.abs(calc_radius - float(row["radius_kpc"])))) for row in packet["rows"]],
            dtype=np.int64,
        )
        if any(
            not math.isclose(
                float(calc_radius[index]),
                float(row["radius_kpc"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for index, row in zip(row_indices, packet["rows"], strict=True)
        ):
            raise ScreenedNonlocalPermutationError("cluster target radius alignment failed")
        prepared.append(
            {
                "anchor_pressure": float(packet["anchor"]["pressure_kev_cm3"]),
                "cluster": str(packet["cluster"]),
                "conversion": float(
                    constants["kev_per_cubic_centimeter_j_per_cubic_meter"]
                ),
                "cross_calibration": float(
                    nuisance["xray_temperature_cross_calibration"]
                ),
                "dr_m": np.diff(radius_m),
                "features": _feature_banks(calc_radius, gbar, config, config59),
                "gradient_prefactor": gradient_prefactor,
                "row_error": np.asarray([float(row["error"]) for row in packet["rows"]]),
                "row_indices": row_indices,
                "row_ne": calc_ne[row_indices],
                "row_observable": np.asarray(
                    [0 if row["observable"] == "pressure" else 1 for row in packet["rows"]],
                    dtype=np.int8,
                ),
                "row_observed": np.asarray(
                    [float(row["observed"]) for row in packet["rows"]]
                ),
                "row_split": np.asarray([str(row["split"]) for row in packet["rows"]]),
                "rows": list(packet["rows"]),
            }
        )
    if len(prepared) != 12:
        raise ScreenedNonlocalPermutationError("X-COP packet count changed")
    splits = ("development_train", "development_holdout", "confirmation")
    for split in splits:
        groups: dict[tuple[str, str], int] = {}
        for packet in prepared:
            for row in packet["rows"]:
                if row["split"] == split:
                    key = (str(row["cluster"]), str(row["observable"]))
                    groups[key] = groups.get(key, 0) + 1
        for packet in prepared:
            weights = np.zeros(len(packet["rows"]), dtype=float)
            for index, row in enumerate(packet["rows"]):
                if row["split"] == split:
                    key = (str(row["cluster"]), str(row["observable"]))
                    weights[index] = 1.0 / (len(groups) * groups[key])
            packet.setdefault("score_weights", {})[split] = weights
    return prepared


def _score_clusters_gpu(
    packets: Sequence[Mapping[str, Any]],
    representatives: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], float]:
    import cupy as cp

    started = time.perf_counter()
    splits = ("development_train", "development_holdout", "confirmation")
    batch_size = int(config["scoring"]["gpu_batch_size"])
    floor = float(config59["scoring"]["minimum_fractional_error"])
    alpha_grid = cp.asarray(config["grammar"]["alpha"], dtype=cp.float64)
    gpu_packets = []
    for packet in packets:
        features = packet["features"]
        gpu_packets.append(
            {
                "anchor": float(packet["anchor_pressure"]),
                "base": cp.asarray(features["base"]),
                "conversion": float(packet["conversion"]),
                "cross": float(packet["cross_calibration"]),
                "dr": cp.asarray(packet["dr_m"]),
                "gradient": cp.asarray(packet["gradient_prefactor"]),
                "indices": cp.asarray(packet["row_indices"]),
                "is_temperature": cp.asarray(packet["row_observable"]).astype(cp.bool_),
                "ne": cp.asarray(packet["row_ne"]),
                "observed": cp.asarray(packet["row_observed"]),
                "fractional": cp.asarray(
                    np.maximum(
                        packet["row_error"] / packet["row_observed"],
                        floor,
                    )
                ),
                "transition": cp.asarray(features["transition"]),
                "weights": {
                    split: cp.asarray(packet["score_weights"][split]) for split in splits
                },
                "x": cp.asarray(features["x"]),
            }
        )
    count = len(representatives["base_law"])
    result = {split: np.empty(count, dtype=float) for split in splits}
    x_index, t_index = _behavior_indices(representatives)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        base_index = cp.asarray(representatives["base_law"][start:stop], dtype=cp.int64)
        alpha_index = cp.asarray(representatives["alpha"][start:stop], dtype=cp.int64)
        xb = cp.asarray(x_index[start:stop], dtype=cp.int64)
        tb = cp.asarray(t_index[start:stop], dtype=cp.int64)
        scores = {split: cp.zeros(stop - start, dtype=cp.float64) for split in splits}
        for packet in gpu_packets:
            acceleration = packet["base"][base_index] * cp.exp(
                2.0
                * alpha_grid[alpha_index, None]
                * packet["transition"][tb]
                * packet["x"][xb]
            )
            gradient = acceleration * packet["gradient"][None, :]
            intervals = 0.5 * (gradient[:, 1:] + gradient[:, :-1]) * packet["dr"][None, :]
            reverse = cp.cumsum(intervals[:, ::-1], axis=1)[:, ::-1]
            integral = cp.concatenate(
                [reverse, cp.zeros((stop - start, 1), dtype=cp.float64)], axis=1
            )
            pressure = packet["anchor"] + integral / packet["conversion"]
            row_pressure = pressure[:, packet["indices"]]
            prediction = cp.where(
                packet["is_temperature"][None, :],
                row_pressure / packet["ne"][None, :] * packet["cross"],
                row_pressure,
            )
            residual = cp.log(prediction / packet["observed"][None, :])
            standardized_square = cp.square(
                residual / packet["fractional"][None, :]
            )
            for split in splits:
                scores[split] += cp.sum(
                    standardized_square * packet["weights"][split][None, :], axis=1
                )
        for split in splits:
            result[split][start:stop] = cp.asnumpy(scores[split])
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    del gpu_packets
    cp.get_default_memory_pool().free_all_blocks()
    return result, elapsed


def _candidate_record(
    representatives: Mapping[str, np.ndarray],
    index: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    grammar = config["grammar"]
    record = {
        "representative_ordinal": int(representatives["representative_ordinal"][index]),
        "candidate_id": f"SNB-{int(representatives['representative_ordinal'][index]):05x}",
        "raw_multiplicity": int(representatives["raw_multiplicity"][index]),
    }
    for name in grammar["factor_order"]:
        factor_index = int(representatives[name][index])
        record[name] = grammar[name][factor_index]
        record[f"{name}_index"] = factor_index
    return record


def _control_index(
    representatives: Mapping[str, np.ndarray], base_index: int
) -> int:
    matches = np.flatnonzero(
        (representatives["base_law"] == base_index)
        & (representatives["alpha"] == 0)
    )
    if len(matches) != 1:
        raise ScreenedNonlocalPermutationError("base-law control behavior changed")
    return int(matches[0])


def _rank_candidates(
    representatives: Mapping[str, np.ndarray],
    analytic: Mapping[str, np.ndarray],
    sparc: np.ndarray,
    little: np.ndarray,
    xcop: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    item59: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    rar_index = _control_index(representatives, 1)
    baselines = {
        "sparc_rar": float(sparc[rar_index]),
        "little_things_rar": float(little[rar_index]),
        "xcop_item59_development_train": float(
            item59["splits"]["development_train"]["candidate"]["score"]
        ),
    }
    ratios = {
        "sparc": sparc / baselines["sparc_rar"],
        "little_things": little / baselines["little_things_rar"],
        "xcop_development_train": (
            xcop["development_train"] / baselines["xcop_item59_development_train"]
        ),
    }
    worst = np.maximum.reduce(list(ratios.values()))
    summed = np.sum(np.stack(list(ratios.values())), axis=0)
    qualifying = np.asarray(analytic["qualifying"])
    eligible = np.flatnonzero(qualifying & np.isfinite(worst))
    order = np.lexsort(
        (
            representatives["representative_ordinal"][eligible],
            summed[eligible],
            worst[eligible],
        )
    )
    return eligible[order], {**ratios, "worst": worst, "sum": summed}, baselines


def _top_record(
    index: int,
    representatives: Mapping[str, np.ndarray],
    ratios: Mapping[str, np.ndarray],
    sparc: np.ndarray,
    little: np.ndarray,
    xcop: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate": _candidate_record(representatives, index, config),
        "scores": {
            "sparc": float(sparc[index]),
            "little_things": float(little[index]),
            "xcop_development_train": float(xcop["development_train"][index]),
            "xcop_development_holdout": float(xcop["development_holdout"][index]),
            "xcop_former_confirmation_retrospective": float(
                xcop["confirmation"][index]
            ),
        },
        "ratios": {
            key: float(ratios[key][index])
            for key in ("sparc", "little_things", "xcop_development_train", "worst", "sum")
        },
    }


def build_evaluation(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = build_preflight(root)
    preflight_path = (
        root / str(config["paths"]["source_dir"]) / str(config["paths"]["preflight"])
    )
    if not preflight_path.exists() or _read_json(preflight_path) != preflight:
        raise ScreenedNonlocalPermutationError("preflight is missing or changed")
    representatives = behavior_representatives(config)
    analytic = _analytic_arrays(representatives, config)
    evaluated_indices = np.flatnonzero(analytic["admitted"])
    evaluated = _representative_selection(representatives, evaluated_indices)
    config59 = load_item59_config(root)

    sparc_profiles = _sparc_profiles(root, config, config59)
    little_profiles = _little_things_profiles(root, config, config59)
    sparc_bank = _galaxy_bank(sparc_profiles, config, config59)
    little_bank = _galaxy_bank(little_profiles, config, config59)
    sparc_scores_subset, sparc_seconds = _score_galaxy_gpu(sparc_bank, evaluated, config)
    little_scores_subset, little_seconds = _score_galaxy_gpu(little_bank, evaluated, config)
    cluster_packets = _cluster_packets(root, config, config59)
    xcop_subset, xcop_seconds = _score_clusters_gpu(
        cluster_packets, evaluated, config, config59
    )

    size = len(representatives["base_law"])
    sparc_scores = np.full(size, np.inf)
    little_scores = np.full(size, np.inf)
    xcop_scores = {
        split: np.full(size, np.inf)
        for split in ("development_train", "development_holdout", "confirmation")
    }
    sparc_scores[evaluated_indices] = sparc_scores_subset
    little_scores[evaluated_indices] = little_scores_subset
    for split, values in xcop_scores.items():
        values[evaluated_indices] = xcop_subset[split]

    item59 = _read_json(root / ITEM59_RESULT)
    ranked, ratios, baselines = _rank_candidates(
        representatives,
        analytic,
        sparc_scores,
        little_scores,
        xcop_scores,
        config,
        item59,
    )
    if not len(ranked):
        raise ScreenedNonlocalPermutationError("no qualifying candidate survived")
    selected_index = int(ranked[0])
    top = [
        _top_record(
            int(index),
            representatives,
            ratios,
            sparc_scores,
            little_scores,
            xcop_scores,
            config,
        )
        for index in ranked[:25]
    ]
    qualifying = np.asarray(analytic["qualifying"])
    threshold_counts = {
        "beats_or_matches_all_three_incumbents": int(
            np.count_nonzero(
                qualifying
                & (ratios["sparc"] <= 1.0)
                & (ratios["little_things"] <= 1.0)
                & (ratios["xcop_development_train"] <= 1.0)
            )
        ),
        "galaxies_within_5pct_and_xcop_within_20pct": int(
            np.count_nonzero(
                qualifying
                & (ratios["sparc"] <= 1.05)
                & (ratios["little_things"] <= 1.05)
                & (ratios["xcop_development_train"] <= 1.20)
            )
        ),
        "sparc_at_or_better_than_rar": int(
            np.count_nonzero(qualifying & (ratios["sparc"] <= 1.0))
        ),
        "little_things_at_or_better_than_rar": int(
            np.count_nonzero(qualifying & (ratios["little_things"] <= 1.0))
        ),
        "xcop_at_or_better_than_item59_incumbent": int(
            np.count_nonzero(
                qualifying & (ratios["xcop_development_train"] <= 1.0)
            )
        ),
    }
    selected = _top_record(
        selected_index,
        representatives,
        ratios,
        sparc_scores,
        little_scores,
        xcop_scores,
        config,
    )
    selected["lensing_branch_status"] = {
        "selected_from_lensing_data": False,
        "direct_lensing_rows_read": 0,
        "dynamics_equivalent_unranked_branches": list(
            config["grammar"]["lensing_branch"]
        ),
    }
    return _content_hashed(
        {
            "schema_version": "invariant-screened-nonlocal-boundary-exposed-evaluation-0.1",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "trial_type": "retrospective_exploration_on_previously_exposed_responses",
            "preflight_content_sha256": preflight["content_sha256"],
            "counts": {
                "raw_candidates": int(config["grammar"]["raw_ordinal_count"]),
                "dynamics_behavior_classes": size,
                "analytically_admitted_behavior_classes": len(evaluated_indices),
                "analytically_qualifying_behavior_classes": int(
                    np.count_nonzero(analytic["qualifying"])
                ),
                "sparc_galaxies": len(sparc_profiles),
                "sparc_rows": sum(len(row["radius_kpc"]) for row in sparc_profiles),
                "little_things_galaxies": len(little_profiles),
                "little_things_rows": sum(
                    len(row["radius_kpc"]) for row in little_profiles
                ),
                "xcop_clusters": len(cluster_packets),
                "new_target_queries": 0,
                "sealed_confirmation_accesses": 0,
                "direct_lensing_rows": 0,
            },
            "compute": {
                "backend": "cupy_float64",
                "device": "NVIDIA GeForce RTX 5090",
                "batch_size": int(config["scoring"]["gpu_batch_size"]),
                "sparc_seconds": sparc_seconds,
                "little_things_seconds": little_seconds,
                "xcop_seconds": xcop_seconds,
                "total_measured_gpu_scoring_seconds": (
                    sparc_seconds + little_seconds + xcop_seconds
                ),
                "paid_model_calls": 0,
                "paid_api_cost_usd": 0.0,
            },
            "incumbent_scores": baselines,
            "threshold_counts": threshold_counts,
            "selected": selected,
            "top_25": top,
            "decision": (
                "EXPOSED_DATA_DESCENDANT_SELECTED_PENDING_UNCHANGED_GROUP_AND_DIRECT_LENSING_GATES"
            ),
            "claims": {
                "full_finite_grammar_exhausted": True,
                "fresh_confirmation": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "historical_novelty_established": False,
                "lensing_rule_selected": False,
                "single_counterexample_used_as_veto": False,
                "formula_family_pruned": False,
            },
            "next_action": (
                "Freeze the selected dynamics behavior and all four still-equivalent lensing "
                "branches, then test a genuinely new group bridge sample before opening any "
                "direct-lensing target."
            ),
        }
    )


def write_evaluation(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["source_dir"]) / str(config["paths"]["evaluation"])
    _write_json(path, build_evaluation(root))
    return path


def build_aggregate(root: Path) -> dict[str, Any]:
    config = load_config(root)
    evaluation_path = (
        root / str(config["paths"]["source_dir"]) / str(config["paths"]["evaluation"])
    )
    evaluation = _read_json(evaluation_path)
    if (
        evaluation.get("schema_version")
        != "invariant-screened-nonlocal-boundary-exposed-evaluation-0.1"
        or evaluation.get("scientific_freeze_commit") != config["scientific_freeze_commit"]
    ):
        raise ScreenedNonlocalPermutationError("evaluation receipt is missing or changed")
    return _content_hashed(
        {
            "schema_version": "invariant-screened-nonlocal-boundary-result-0.1",
            "goal": "SCREENED_NONLOCAL_BOUNDARY_DESCENDANT_EXHAUSTIVE_EXPLORATION",
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "decision": evaluation["decision"],
            "selected": evaluation["selected"],
            "counts": evaluation["counts"],
            "compute": evaluation["compute"],
            "threshold_counts": evaluation["threshold_counts"],
            "claims": evaluation["claims"],
            "limitations": [
                "Every permutation means every cell of the frozen 4^10 grammar, not every mathematically imaginable theory.",
                "All empirical responses were already exposed before this new grammar, so no score is independent confirmation.",
                "The X-COP calculation retains hydrostatic, spherical, boundary-pressure, member-baryon, and nonthermal-pressure assumptions.",
                "The four lensing branches are exactly indistinguishable without direct lensing targets and remain unranked.",
                "The weak-field formula has an action scaffold, but the covariant causal source map and full stability analysis remain incomplete."
            ],
            "next_action": evaluation["next_action"],
        }
    )


def write_aggregate(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate"])
    _write_json(path, build_aggregate(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    preflight = build_preflight(root)
    evaluation = build_evaluation(root)
    aggregate = build_aggregate(root)
    return {
        "preflight_content_sha256": preflight["content_sha256"],
        "evaluation_content_sha256": evaluation["content_sha256"],
        "aggregate_content_sha256": aggregate["content_sha256"],
        "selected_candidate_id": evaluation["selected"]["candidate"]["candidate_id"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "evaluate", "aggregate", "replay"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        output: Any = str(write_preflight(root))
    elif args.command == "evaluate":
        output = str(write_evaluation(root))
    elif args.command == "aggregate":
        output = str(write_aggregate(root))
    else:
        output = replay(root)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
