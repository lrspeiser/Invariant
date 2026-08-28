"""Generate and test ten weak-field first-principles mechanism lanes."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g0_experiment import score_predictions
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g3_meta_law import _fold_map
from .gravity_g4_conditional_formula_generator import (
    galaxy_conditions,
)
from .gravity_g4_conditional_formula_generator import (
    validate_receipt as validate_predecessor_receipt,
)
from .gravity_g4_nonlocal_profile_law_construction import (
    _kernel_matrix,
    _log_radius_cell_widths,
    prepare_nonlocal_packets,
)
from .gravity_g4_universal_law_construction import _stratum_assignments
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-g4-first-principles-mechanism-receipt-5.0"
CONFIG_SCHEMA = "invariant-gravity-g4-first-principles-mechanism-config-5.0"
CONFIG_PATH = "configs/gravity_g4_first_principles_mechanism_search.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g4_first_principles_mechanism_search.py"
TEST_PATH = "tests/test_gravity_g4_first_principles_mechanism_search.py"
OUTPUT_PATH = "runs/gravity/g4/first-principles-mechanism-search-v5.json"

LANE_IDS = (
    "inverse_action_discovery",
    "baryonic_gravitational_permittivity",
    "auxiliary_focusing_field",
    "geometry_directed_gravity",
    "vacuum_boundary_field",
    "causal_gravitational_memory",
    "orbital_mode_resonance",
    "nonlocal_modified_inertia",
    "multiscale_running_gravity",
    "cross_scale_action_synthesis",
)
SCALES = (0.25, 0.5, 1.0, 2.0)
STELLAR_THRESHOLDS = (10.0, 100.0, 1000.0)
ACCELERATION_THRESHOLDS = (0.01, 0.1, 1.0)
EVOLUTION_TIMES_GYR = (1.0, 3.0, 10.0)
COEFFICIENT_GRID = (-4.0, -2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
KM_S_PER_KPC_TO_GYR_INV = 1.0227121650537077


class GravityG4FirstPrinciplesError(ValueError):
    """The mechanism grammar, computation, or sealed evidence is inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the frozen ten-lane mechanism contract and validate its lineage."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG4FirstPrinciplesError("first-principles config changed")
    predecessor_binding = config.get("predecessor_binding", {})
    predecessor_path = root / str(predecessor_binding.get("path"))
    if _file_sha256(predecessor_path) != predecessor_binding.get("file_sha256"):
        raise GravityG4FirstPrinciplesError("first-principles predecessor file changed")
    predecessor = _load_json(predecessor_path)
    validate_predecessor_receipt(predecessor, root=root)
    if predecessor.get("content_sha256") != predecessor_binding.get(
        "content_sha256"
    ) or predecessor.get("decision") != predecessor_binding.get("required_decision"):
        raise GravityG4FirstPrinciplesError("first-principles predecessor content changed")
    g0_binding = config.get("g0_binding", {})
    g0_path = root / str(g0_binding.get("path"))
    g0 = _load_json(g0_path)
    if (
        _file_sha256(g0_path) != g0_binding.get("file_sha256")
        or g0.get("content_sha256") != g0_binding.get("content_sha256")
        or g0.get("decision") != g0_binding.get("required_decision")
    ):
        raise GravityG4FirstPrinciplesError("first-principles G0 binding changed")
    lane_contract = config.get("mechanism_lanes", ())
    if tuple(row.get("id") for row in lane_contract) != LANE_IDS:
        raise GravityG4FirstPrinciplesError("mechanism lane order changed")
    if sum(int(row["declared_candidates"]) for row in lane_contract) != 281:
        raise GravityG4FirstPrinciplesError("mechanism candidate declaration changed")
    accounting = config.get("candidate_accounting", {})
    if (
        accounting.get("mechanism_candidates") != 280
        or accounting.get("known_positive_controls") != 1
        or accounting.get("total_candidate_structures") != 281
        or accounting.get("coefficient_cells_per_selection") != 3653
        or accounting.get("declared_scoring_point_evaluations") != 49_680_800
    ):
        raise GravityG4FirstPrinciplesError("mechanism candidate accounting changed")
    if tuple(float(value) for value in config.get("coefficient_grid", ())) != COEFFICIENT_GRID:
        raise GravityG4FirstPrinciplesError("mechanism coefficient grid changed")
    population = config.get("population", {})
    if any(
        population.get(key) != 0
        for key in (
            "confirmation_evaluator_accesses_allowed",
            "cluster_evaluator_accesses_allowed",
            "lensing_evaluator_accesses_allowed",
        )
    ):
        raise GravityG4FirstPrinciplesError("mechanism config opens a downstream evaluator")
    if config.get("origin_policy", {}).get("historical_novelty_claimed") is not False:
        raise GravityG4FirstPrinciplesError("mechanism config overstates novelty")
    return config


def _scale_id(value: float) -> str:
    return format(value, "g").replace(".", "p")


def _equation_ir(kind: str, equation: str, *, action_status: str) -> dict[str, Any]:
    return {
        "action_status": action_status,
        "dimension_output": "velocity_squared",
        "equation": equation,
        "kind": kind,
        "source_inputs": "baryonic_mass_geometry_only",
    }


def mechanism_specs() -> list[dict[str, Any]]:
    """Enumerate all 280 mechanisms and one exact known-family positive control."""

    rows: list[dict[str, Any]] = []

    for threshold in ACCELERATION_THRESHOLDS:
        for shape in ("linear_occupancy", "quadratic_occupancy"):
            for smoothing in ("local", "symmetric_ell0p5"):
                rows.append(
                    {
                        "candidate_id": (
                            f"inverse-action:y:q{_scale_id(threshold)}:{shape}:{smoothing}"
                        ),
                        "equation_ir": _equation_ir(
                            "effective_action",
                            "delta S/dPhi=0 for L_eff=-(1+beta*f_b)|grad Phi|^2/(8 pi G)-rho_b Phi",
                            action_status="effective_radial_action_only",
                        ),
                        "lane": "inverse_action_discovery",
                        "origin_label": "new_combination_of_known_ideas",
                        "role": "mechanism",
                        "shape": shape,
                        "smoothing": smoothing,
                        "source": "baryonic_acceleration",
                        "threshold": threshold,
                    }
                )

    for source, thresholds in (
        ("stellar_surface_density", STELLAR_THRESHOLDS),
        ("baryonic_acceleration", ACCELERATION_THRESHOLDS),
    ):
        for threshold in thresholds:
            for scale in SCALES:
                for mode in ("interior_flux", "exterior_vacuum"):
                    rows.append(
                        {
                            "candidate_id": (
                                f"permittivity:{source}:q{_scale_id(threshold)}:"
                                f"ell{_scale_id(scale)}:{mode}"
                            ),
                            "equation_ir": _equation_ir(
                                "elliptic_flux_equation",
                                "div(mu[rho_b] grad Phi)=4 pi G rho_b",
                                action_status="effective_baryon_conditioned_permittivity",
                            ),
                            "lane": "baryonic_gravitational_permittivity",
                            "log_radius_scale": scale,
                            "mode": mode,
                            "origin_label": "known_family_scaffold_or_combination",
                            "role": "mechanism",
                            "source": source,
                            "threshold": threshold,
                        }
                    )

    for source, thresholds in (
        ("stellar_surface_density", STELLAR_THRESHOLDS),
        ("baryonic_acceleration", ACCELERATION_THRESHOLDS),
    ):
        for threshold in thresholds:
            for scale in SCALES:
                for mode in ("screened_source", "screened_contrast"):
                    rows.append(
                        {
                            "candidate_id": (
                                f"aux-field:{source}:q{_scale_id(threshold)}:"
                                f"ell{_scale_id(scale)}:{mode}"
                            ),
                            "equation_ir": _equation_ir(
                                "auxiliary_field",
                                "(1-ell^2 nabla_logr^2) psi=f(rho_b); delta g=beta g_dagger psi",
                                action_status="screened_radial_auxiliary_equation",
                            ),
                            "lane": "auxiliary_focusing_field",
                            "log_radius_scale": scale,
                            "mode": mode,
                            "origin_label": "new_combination_of_known_ideas",
                            "role": "mechanism",
                            "source": source,
                            "threshold": threshold,
                        }
                    )

    for threshold in STELLAR_THRESHOLDS:
        for scale in SCALES:
            for mode in ("gradient_alignment", "curvature_alignment"):
                rows.append(
                    {
                        "candidate_id": (
                            f"geometry-tensor:sb:q{_scale_id(threshold)}:"
                            f"ell{_scale_id(scale)}:{mode}"
                        ),
                        "equation_ir": _equation_ir(
                            "geometry_response_tensor",
                            "nabla_i(K^ij[rho_b] nabla_j Phi)=4 pi G rho_b",
                            action_status="radial_tensor_projection_only",
                        ),
                        "lane": "geometry_directed_gravity",
                        "log_radius_scale": scale,
                        "mode": mode,
                        "origin_label": "proposed_new_construction",
                        "role": "mechanism",
                        "source": "stellar_surface_density",
                        "threshold": threshold,
                    }
                )

    for source, thresholds in (
        ("stellar_surface_density", STELLAR_THRESHOLDS),
        ("baryonic_acceleration", ACCELERATION_THRESHOLDS),
    ):
        for threshold in thresholds:
            for scale in SCALES:
                for mode in (
                    "interior_minus_exterior_occupancy",
                    "interior_occupancy_times_exterior_vacuum",
                ):
                    rows.append(
                        {
                            "candidate_id": (
                                f"vacuum-boundary:{source}:q{_scale_id(threshold)}:"
                                f"ell{_scale_id(scale)}:{mode}"
                            ),
                            "equation_ir": _equation_ir(
                                "boundary_state_field",
                                "delta g=beta g_dagger B[occupied interior, empty exterior]",
                                action_status="phenomenological_boundary_projection",
                            ),
                            "lane": "vacuum_boundary_field",
                            "log_radius_scale": scale,
                            "mode": mode,
                            "origin_label": "new_combination_of_known_ideas",
                            "role": "mechanism",
                            "source": source,
                            "threshold": threshold,
                        }
                    )

    for mode in ("interior_relaxation", "symmetric_relaxation"):
        for scale in SCALES:
            for time_gyr in EVOLUTION_TIMES_GYR:
                rows.append(
                    {
                        "candidate_id": (
                            f"causal-memory:{mode}:ell{_scale_id(scale)}:tgyr{_scale_id(time_gyr)}"
                        ),
                        "equation_ir": _equation_ir(
                            "causal_relaxation",
                            "tau^2 ddot psi+tau dot psi-ell^2 nabla^2 psi+m^2 psi=lambda rho_b",
                            action_status="static_projection_of_time_equation",
                        ),
                        "evolution_time_gyr": time_gyr,
                        "lane": "causal_gravitational_memory",
                        "log_radius_scale": scale,
                        "mode": mode,
                        "origin_label": "new_combination_of_known_ideas",
                        "role": "mechanism",
                    }
                )

    for scale in SCALES:
        for mode_count in (1, 2, 3, 4):
            rows.append(
                {
                    "candidate_id": (f"orbital-modes:ell{_scale_id(scale)}:k{mode_count}"),
                    "equation_ir": _equation_ir(
                        "spectral_mode_equation",
                        "L_baryon u_n=lambda_n u_n; delta g=sum_(n<=k) <y,u_n> u_n",
                        action_status="radial_graph_spectrum_only",
                    ),
                    "lane": "orbital_mode_resonance",
                    "log_radius_scale": scale,
                    "mode_count": mode_count,
                    "origin_label": "proposed_new_construction",
                    "role": "mechanism",
                }
            )

    for mode in ("interior_frequency_memory", "symmetric_frequency_memory"):
        for scale in SCALES:
            for time_gyr in EVOLUTION_TIMES_GYR:
                rows.append(
                    {
                        "candidate_id": (
                            f"modified-inertia:{mode}:ell{_scale_id(scale)}:"
                            f"tgyr{_scale_id(time_gyr)}"
                        ),
                        "equation_ir": _equation_ir(
                            "nonlocal_particle_action",
                            "S_particle=int L[v,a,int K(t-t')a(t')dt']dt",
                            action_status="static_orbital_history_proxy",
                        ),
                        "evolution_time_gyr": time_gyr,
                        "lane": "nonlocal_modified_inertia",
                        "log_radius_scale": scale,
                        "mode": mode,
                        "origin_label": "known_family_scaffold_or_combination",
                        "role": "mechanism",
                    }
                )

    for scale in SCALES:
        for shape in ("log_vacuum", "sqrt_transition", "compactness_flow"):
            rows.append(
                {
                    "candidate_id": f"running-gravity:ell{_scale_id(scale)}:{shape}",
                    "equation_ir": _equation_ir(
                        "radial_renormalization_flow",
                        "d G_eff/d log ell=beta_flow(G_eff, compactness_b, geometry_b)",
                        action_status="bounded_radial_flow_ansatz",
                    ),
                    "lane": "multiscale_running_gravity",
                    "log_radius_scale": scale,
                    "origin_label": "known_family_scaffold_or_combination",
                    "role": "mechanism",
                    "shape": shape,
                }
            )
    rows.append(
        {
            "candidate_id": "known-control:exact-empirical-rar-rewrite",
            "equation_ir": _equation_ir(
                "known_empirical_control",
                "g=g_bar/(1-exp(-sqrt(g_bar/g_dagger)))",
                action_status="known_empirical_relation_not_action_derivation",
            ),
            "lane": "multiscale_running_gravity",
            "origin_label": "known_family_instance",
            "role": "known_positive_control",
            "shape": "exact_rar_rewrite",
        }
    )

    for threshold in ACCELERATION_THRESHOLDS:
        for scale in SCALES:
            for mode in ("permittivity_plus_auxiliary", "vacuum_plus_memory"):
                rows.append(
                    {
                        "candidate_id": (
                            f"cross-scale:y:q{_scale_id(threshold)}:ell{_scale_id(scale)}:{mode}"
                        ),
                        "equation_ir": _equation_ir(
                            "cross_scale_action_packet",
                            "one baryon-sourced field must generate dynamics and lensing at all scales",
                            action_status="galaxy_projection_only_downstream_gates_unopened",
                        ),
                        "lane": "cross_scale_action_synthesis",
                        "log_radius_scale": scale,
                        "mode": mode,
                        "origin_label": "proposed_system_construction",
                        "role": "mechanism",
                        "source": "baryonic_acceleration",
                        "threshold": threshold,
                    }
                )

    counts = Counter(row["lane"] for row in rows)
    expected = dict(zip(LANE_IDS, (12, 48, 48, 24, 48, 24, 16, 24, 13, 24), strict=True))
    if len(rows) != 281 or counts != expected:
        raise GravityG4FirstPrinciplesError("mechanism enumeration changed")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise GravityG4FirstPrinciplesError("mechanism identifiers are not unique")
    return rows


def _packet_context(packet: Mapping[str, Any]) -> dict[str, Any]:
    radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
    log_radius = np.log(radius)
    widths = _log_radius_cell_widths(log_radius)
    matrices = {
        (kind, scale): _kernel_matrix(log_radius, widths, kind, scale)
        for kind in (
            "interior_exponential",
            "exterior_exponential",
            "symmetric_exponential",
        )
        for scale in SCALES
    }
    stellar = np.expm1(np.asarray(packet["features"]["log1p_sb_total"], dtype=np.float64))
    acceleration = np.exp(np.asarray(packet["features"]["log_y"], dtype=np.float64))
    vbar2 = np.asarray(packet["arrays"]["vbar2"], dtype=np.float64)
    context: dict[str, Any] = {
        "a0": float(packet["a0"]),
        "acceleration": acceleration,
        "log_radius": log_radius,
        "matrices": matrices,
        "occupancy": {},
        "radius": radius,
        "spectra": {},
        "stellar": stellar,
        "vbar2": vbar2,
    }
    for source, thresholds in (
        ("stellar_surface_density", STELLAR_THRESHOLDS),
        ("baryonic_acceleration", ACCELERATION_THRESHOLDS),
    ):
        values = stellar if source == "stellar_surface_density" else acceleration
        for threshold in thresholds:
            context["occupancy"][(source, threshold)] = values / (values + threshold)
    return context


def _matrix(context: Mapping[str, Any], direction: str, scale: float) -> np.ndarray:
    return np.asarray(context["matrices"][(f"{direction}_exponential", scale)])


def _cycles(context: Mapping[str, Any], time_gyr: float) -> np.ndarray:
    vbar2 = np.asarray(context["vbar2"])
    radius = np.asarray(context["radius"])
    return (
        time_gyr
        * np.sqrt(np.maximum(vbar2, 0.0))
        / radius
        * KM_S_PER_KPC_TO_GYR_INV
        / (2.0 * np.pi)
    )


def _component_for_spec(
    spec: Mapping[str, Any], packet: Mapping[str, Any], context: dict[str, Any]
) -> np.ndarray:
    lane = str(spec["lane"])
    radius = np.asarray(context["radius"])
    vbar2 = np.asarray(context["vbar2"])
    radius_a0 = radius * float(context["a0"])
    if spec["role"] == "known_positive_control":
        return np.asarray(packet["rar2"], dtype=np.float64) - vbar2

    if lane == "inverse_action_discovery":
        q = np.asarray(context["occupancy"][("baryonic_acceleration", float(spec["threshold"]))])
        feature = q if spec["shape"] == "linear_occupancy" else q**2
        if spec["smoothing"] != "local":
            feature = _matrix(context, "symmetric", 0.5) @ feature
        return vbar2 * feature

    if lane == "baryonic_gravitational_permittivity":
        q = np.asarray(context["occupancy"][(str(spec["source"]), float(spec["threshold"]))])
        scale = float(spec["log_radius_scale"])
        feature = (
            _matrix(context, "interior", scale) @ q
            if spec["mode"] == "interior_flux"
            else 1.0 - _matrix(context, "exterior", scale) @ q
        )
        return vbar2 * feature

    if lane == "auxiliary_focusing_field":
        q = np.asarray(context["occupancy"][(str(spec["source"]), float(spec["threshold"]))])
        psi = _matrix(context, "symmetric", float(spec["log_radius_scale"])) @ q
        feature = psi if spec["mode"] == "screened_source" else psi - q
        return radius_a0 * feature

    if lane == "geometry_directed_gravity":
        q = np.asarray(context["occupancy"][("stellar_surface_density", float(spec["threshold"]))])
        smooth = _matrix(context, "symmetric", float(spec["log_radius_scale"])) @ q
        gradient = np.gradient(smooth, np.asarray(context["log_radius"]))
        if spec["mode"] == "gradient_alignment":
            feature = smooth * np.tanh(-gradient)
        else:
            curvature = np.gradient(gradient, np.asarray(context["log_radius"]))
            feature = smooth * np.tanh(-curvature)
        return radius_a0 * feature

    if lane == "vacuum_boundary_field":
        q = np.asarray(context["occupancy"][(str(spec["source"]), float(spec["threshold"]))])
        scale = float(spec["log_radius_scale"])
        interior = _matrix(context, "interior", scale) @ q
        exterior = _matrix(context, "exterior", scale) @ q
        feature = (
            interior - exterior
            if spec["mode"] == "interior_minus_exterior_occupancy"
            else interior * (1.0 - exterior)
        )
        return radius_a0 * feature

    if lane == "causal_gravitational_memory":
        direction = "interior" if spec["mode"] == "interior_relaxation" else "symmetric"
        mean_v2 = _matrix(context, direction, float(spec["log_radius_scale"])) @ vbar2
        response = 1.0 - np.exp(-_cycles(context, float(spec["evolution_time_gyr"])))
        return (mean_v2 - vbar2) * response

    if lane == "orbital_mode_resonance":
        scale = float(spec["log_radius_scale"])
        if scale not in context["spectra"]:
            kernel = _matrix(context, "symmetric", scale)
            adjacency = 0.5 * (kernel + kernel.T)
            np.fill_diagonal(adjacency, 0.0)
            laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
            _values, vectors = np.linalg.eigh(laplacian)
            context["spectra"][scale] = vectors
        vectors = np.asarray(context["spectra"][scale])
        count = min(int(spec["mode_count"]), max(0, vectors.shape[1] - 1))
        modes = vectors[:, 1 : 1 + count]
        source = np.asarray(context["acceleration"])
        source = source - float(np.mean(source))
        reconstruction = modes @ (modes.T @ source) if count else np.zeros_like(source)
        return radius_a0 * reconstruction

    if lane == "nonlocal_modified_inertia":
        direction = "interior" if spec["mode"] == "interior_frequency_memory" else "symmetric"
        frequency = np.sqrt(np.maximum(vbar2, 0.0)) / radius
        mean_frequency = _matrix(context, direction, float(spec["log_radius_scale"])) @ frequency
        ratio = (mean_frequency - frequency) / np.maximum(frequency, np.finfo(np.float64).tiny)
        response = 1.0 - np.exp(-_cycles(context, float(spec["evolution_time_gyr"])))
        return vbar2 * np.tanh(ratio) * response

    if lane == "multiscale_running_gravity":
        y = np.maximum(np.asarray(context["acceleration"]), np.finfo(np.float64).tiny)
        matrix = _matrix(context, "symmetric", float(spec["log_radius_scale"]))
        if spec["shape"] == "log_vacuum":
            feature = matrix @ np.tanh(np.log1p(1.0 / y) / 4.0)
        elif spec["shape"] == "sqrt_transition":
            feature = matrix @ (np.sqrt(y) / (1.0 + np.sqrt(y)))
        else:
            log_y = np.log(y)
            feature = np.tanh(matrix @ log_y - log_y)
        return radius_a0 * feature

    if lane == "cross_scale_action_synthesis":
        q = np.asarray(context["occupancy"][("baryonic_acceleration", float(spec["threshold"]))])
        scale = float(spec["log_radius_scale"])
        interior = _matrix(context, "interior", scale) @ q
        exterior = _matrix(context, "exterior", scale) @ q
        symmetric = _matrix(context, "symmetric", scale)
        if spec["mode"] == "permittivity_plus_auxiliary":
            return vbar2 * interior + radius_a0 * (symmetric @ q)
        return radius_a0 * interior * (1.0 - exterior) + symmetric @ vbar2 - vbar2
    raise GravityG4FirstPrinciplesError(f"unknown mechanism lane: {lane}")


def materialize_mechanisms(
    packets: Sequence[Mapping[str, Any]], *, candidate_limit: int | None = None
) -> list[dict[str, Any]]:
    """Materialize typed velocity-squared components without reading evaluator targets."""

    specs = mechanism_specs()
    if candidate_limit is not None:
        specs = specs[: max(0, min(candidate_limit, len(specs)))]
    parts: dict[str, list[np.ndarray]] = {row["candidate_id"]: [] for row in specs}
    for packet in packets:
        context = _packet_context(packet)
        for spec in specs:
            component = _component_for_spec(spec, packet, context)
            if component.shape != (packet["galaxy"].count,) or np.any(~np.isfinite(component)):
                raise GravityG4FirstPrinciplesError(
                    f"invalid mechanism component {spec['candidate_id']} on {packet['galaxy'].name}"
                )
            parts[spec["candidate_id"]].append(component)
    return [{**spec, "component_v2": np.concatenate(parts[spec["candidate_id"]])} for spec in specs]


def _flatten(
    packets: Sequence[Mapping[str, Any]], assignments: Mapping[str, int]
) -> dict[str, Any]:
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for packet in packets:
        count = packet["galaxy"].count
        slices[packet["galaxy"].name] = (offset, offset + count)
        offset += count
    return {
        "fold": np.concatenate(
            [
                np.full(packet["galaxy"].count, assignments[packet["galaxy"].name])
                for packet in packets
            ]
        ),
        "observed": np.concatenate([packet["arrays"]["vobs"] for packet in packets]),
        "rar2": np.concatenate([packet["rar2"] for packet in packets]),
        "sigma": np.concatenate([packet["arrays"]["sigma"] for packet in packets]),
        "slices": slices,
        "vbar2": np.concatenate([packet["arrays"]["vbar2"] for packet in packets]),
    }


def _score_prediction2(
    prediction2: np.ndarray, flat: Mapping[str, Any], mask: np.ndarray | None = None
) -> tuple[float, int]:
    if mask is None:
        mask = np.ones(len(prediction2), dtype=bool)
    values = np.asarray(prediction2)[mask]
    invalid = int(np.sum(~np.isfinite(values) | (values <= 0)))
    prediction = np.sqrt(np.maximum(values, np.finfo(np.float64).tiny))
    observed = np.asarray(flat["observed"])[mask]
    sigma = np.asarray(flat["sigma"])[mask]
    chi_square = float(np.sum(((prediction - observed) / sigma) ** 2))
    return chi_square, invalid


def _public_spec(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "component_v2"}


def _score_cells(
    mechanisms: Sequence[Mapping[str, Any]], flat: Mapping[str, Any], mask: np.ndarray
) -> list[dict[str, Any]]:
    cells = []
    base = np.asarray(flat["vbar2"])
    for mechanism in mechanisms:
        component = np.asarray(mechanism["component_v2"])
        for beta in COEFFICIENT_GRID:
            chi_square, invalid = _score_prediction2(base + beta * component, flat, mask)
            cells.append(
                {
                    "beta": _metric(beta),
                    "candidate_id": mechanism["candidate_id"],
                    "chi_square": _metric(chi_square),
                    "invalid_prediction2": invalid,
                    "lane": mechanism["lane"],
                    "origin_label": mechanism["origin_label"],
                    "role": mechanism["role"],
                    "universal_constants": 1,
                }
            )
    return cells


def _cell_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (float(row["chi_square"]), abs(float(row["beta"])), str(row["candidate_id"]))


def _select_cell(
    cells: Sequence[Mapping[str, Any]], *, lane: str | None = None, mechanisms_only: bool = True
) -> dict[str, Any]:
    eligible = [
        row
        for row in cells
        if row["invalid_prediction2"] == 0
        and (lane is None or row["lane"] == lane)
        and (not mechanisms_only or row["role"] == "mechanism")
    ]
    if not eligible:
        raise GravityG4FirstPrinciplesError("no eligible mechanism cell")
    return dict(min(eligible, key=_cell_key))


def _prediction_for_cell(
    cell: Mapping[str, Any],
    mechanism_by_id: Mapping[str, Mapping[str, Any]],
    flat: Mapping[str, Any],
) -> np.ndarray:
    mechanism = mechanism_by_id[str(cell["candidate_id"])]
    return np.asarray(flat["vbar2"]) + float(cell["beta"]) * np.asarray(mechanism["component_v2"])


def _prior_focusing_prediction2(
    packets: Sequence[Mapping[str, Any]], flat: Mapping[str, Any]
) -> np.ndarray:
    components = []
    alphas = []
    for packet in packets:
        context = _packet_context(packet)
        q = np.asarray(context["occupancy"][("stellar_surface_density", 100.0)])
        interior = _matrix(context, "interior", 0.25) @ q
        exterior = _matrix(context, "exterior", 0.25) @ q
        components.append(
            np.asarray(context["radius"]) * float(context["a0"]) * interior * (1.0 - exterior)
        )
        surface_density = galaxy_conditions(packet)["surface_density"]
        alpha = 2.0 / (1.0 + math.exp(-(-2.0 + 4.0 * surface_density)))
        alphas.append(np.full(packet["galaxy"].count, alpha, dtype=np.float64))
    return np.asarray(flat["rar2"]) + np.concatenate(alphas) * np.concatenate(components)


def _score_by_galaxy(
    packets: Sequence[Mapping[str, Any]], prediction2: np.ndarray, flat: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for packet in packets:
        name = packet["galaxy"].name
        start, stop = flat["slices"][name]
        local2 = np.asarray(prediction2)[start:stop]
        local = np.sqrt(np.maximum(local2, np.finfo(np.float64).tiny))
        rows.append(
            {
                "candidate_score": score_predictions(
                    local, packet["arrays"]["vobs"], packet["arrays"]["sigma"]
                ),
                "galaxy": name,
                "invalid_prediction2": int(np.sum(~np.isfinite(local2) | (local2 <= 0))),
                "point_count": packet["galaxy"].count,
                "rar_score": score_predictions(
                    np.sqrt(packet["rar2"]),
                    packet["arrays"]["vobs"],
                    packet["arrays"]["sigma"],
                ),
            }
        )
    return rows


def build_receipt(root: Path, *, candidate_limit: int | None = None) -> dict[str, Any]:
    """Run all ten mechanism lanes using nested whole-galaxy selection."""

    root = root.resolve()
    config = load_config(root)
    packets = sorted(prepare_nonlocal_packets(root), key=lambda packet: packet["galaxy"].name)
    population = config["population"]
    if len(packets) != int(population["exploration_galaxies"]) or sum(
        packet["galaxy"].count for packet in packets
    ) != int(population["exploration_points"]):
        raise GravityG4FirstPrinciplesError("mechanism population changed")
    outer = config["nested_whole_galaxy_evaluation"]
    folds = int(outer["outer_folds"])
    assignments = _fold_map(
        [packet["galaxy"].name for packet in packets], str(outer["outer_salt"]), folds
    )
    flat = _flatten(packets, assignments)
    mechanisms = materialize_mechanisms(packets, candidate_limit=candidate_limit)
    mechanism_by_id = {row["candidate_id"]: row for row in mechanisms}
    point_count = len(flat["observed"])
    fold_ids = np.asarray(flat["fold"])
    oof_global2 = np.full(point_count, np.nan, dtype=np.float64)
    oof_lane2 = {lane: np.full(point_count, np.nan, dtype=np.float64) for lane in LANE_IDS}
    outer_ledger = []
    grid_point_evaluations = 0
    for fold in range(folds):
        training = fold_ids != fold
        heldout = ~training
        cells = _score_cells(mechanisms, flat, training)
        grid_point_evaluations += len(cells) * int(np.sum(training))
        global_cell = _select_cell(cells)
        global_prediction2 = _prediction_for_cell(global_cell, mechanism_by_id, flat)
        oof_global2[heldout] = global_prediction2[heldout]
        held_chi, held_invalid = _score_prediction2(global_prediction2, flat, heldout)
        lane_rows = []
        for lane in LANE_IDS:
            available = any(
                row["lane"] == lane and row["role"] == "mechanism" for row in mechanisms
            )
            if not available:
                continue
            lane_cell = _select_cell(cells, lane=lane)
            lane_prediction2 = _prediction_for_cell(lane_cell, mechanism_by_id, flat)
            oof_lane2[lane][heldout] = lane_prediction2[heldout]
            lane_chi, lane_invalid = _score_prediction2(lane_prediction2, flat, heldout)
            lane_rows.append(
                {
                    "heldout_chi_square": _metric(lane_chi),
                    "heldout_invalid_prediction2": lane_invalid,
                    "lane": lane,
                    "selected_cell": lane_cell,
                }
            )
        outer_ledger.append(
            {
                "fold": fold,
                "heldout_chi_square": _metric(held_chi),
                "heldout_galaxies": int(
                    sum(assignments[packet["galaxy"].name] == fold for packet in packets)
                ),
                "heldout_invalid_prediction2": held_invalid,
                "lane_selections": lane_rows,
                "selected_cell": global_cell,
                "training_galaxies": int(
                    sum(assignments[packet["galaxy"].name] != fold for packet in packets)
                ),
            }
        )
    if np.any(~np.isfinite(oof_global2)):
        raise GravityG4FirstPrinciplesError("nested mechanism predictions are incomplete")

    full_mask = np.ones(point_count, dtype=bool)
    final_cells = _score_cells(mechanisms, flat, full_mask)
    grid_point_evaluations += len(final_cells) * point_count
    final_cell = _select_cell(final_cells)
    final_prediction2 = _prediction_for_cell(final_cell, mechanism_by_id, flat)
    final_chi, final_invalid = _score_prediction2(final_prediction2, flat)
    best_all_cell = _select_cell(final_cells, mechanisms_only=False)
    lane_results = []
    for lane in LANE_IDS:
        available = any(row["lane"] == lane and row["role"] == "mechanism" for row in mechanisms)
        if not available:
            continue
        nested_chi, nested_invalid = _score_prediction2(oof_lane2[lane], flat)
        lane_final = _select_cell(final_cells, lane=lane)
        lane_results.append(
            {
                "final_all_exploration_selection": lane_final,
                "lane": lane,
                "nested_fractional_gain_over_rar": _metric(
                    1.0 - nested_chi / float(_score_prediction2(np.asarray(flat["rar2"]), flat)[0])
                ),
                "nested_oof_chi_square": _metric(nested_chi),
                "nested_oof_invalid_prediction2": nested_invalid,
            }
        )
    lane_results.sort(key=lambda row: float(row["nested_oof_chi_square"]))

    controls_prediction2 = {
        "newtonian_baryons": np.asarray(flat["vbar2"]),
        "empirical_rar": np.asarray(flat["rar2"]),
        "prior_baryonic_focusing_v4": _prior_focusing_prediction2(packets, flat),
    }
    controls = {}
    for control_id, prediction2 in controls_prediction2.items():
        chi, invalid = _score_prediction2(prediction2, flat)
        controls[control_id] = {
            "chi_square": _metric(chi),
            "invalid_prediction2": invalid,
        }
    nested_chi, nested_invalid = _score_prediction2(oof_global2, flat)
    rar_chi = float(controls["empirical_rar"]["chi_square"])
    per_galaxy = _score_by_galaxy(packets, oof_global2, flat)
    for row in per_galaxy:
        row["fold"] = assignments[row["galaxy"]]
        row["selected_cell"] = outer_ledger[row["fold"]]["selected_cell"]
    galaxies_beating_rar = sum(
        float(row["candidate_score"]["chi_square"]) < float(row["rar_score"]["chi_square"])
        for row in per_galaxy
    )
    by_name = {row["galaxy"]: row for row in per_galaxy}
    strata = []
    for dimension, bins_by_name in _stratum_assignments(packets, 4).items():
        for bin_id in range(4):
            names = sorted(name for name, value in bins_by_name.items() if value == bin_id)
            candidate_chi = sum(
                float(by_name[name]["candidate_score"]["chi_square"]) for name in names
            )
            local_rar = sum(float(by_name[name]["rar_score"]["chi_square"]) for name in names)
            strata.append(
                {
                    "bin": bin_id,
                    "candidate_chi_square": _metric(candidate_chi),
                    "dimension": dimension,
                    "fractional_gain_over_rar": _metric(1.0 - candidate_chi / local_rar),
                    "galaxies": len(names),
                    "rar_chi_square": _metric(local_rar),
                }
            )

    predecessor = _load_json(root / str(config["predecessor_binding"]["path"]))
    nfw_chi = float(predecessor["scores"]["nfw_ceiling_chi_square"])
    nfw_limit = (
        nfw_chi + float(config["admission"]["nfw_ceiling_slack_chi_square_per_point"]) * point_count
    )
    selections = [
        (row["selected_cell"]["candidate_id"], row["selected_cell"]["beta"]) for row in outer_ledger
    ]
    selection_counts = Counter(selections)
    complete_run = candidate_limit is None and len(mechanisms) == 281
    admission = config["admission"]
    gate_checks = {
        "all_nested_predictions_positive_and_finite": nested_invalid == 0,
        "beats_rar_by_nested_minimum_fraction": (
            1.0 - nested_chi / rar_chi
            >= float(admission["minimum_nested_fractional_gain_over_rar"])
        ),
        "complete_first_principles_obligations": False,
        "complete_ten_lane_grammar_searched": complete_run,
        "majority_of_individual_galaxies_beat_rar": (
            galaxies_beating_rar / len(per_galaxy)
            >= float(admission["minimum_fraction_of_individual_galaxies_beating_rar"])
        ),
        "no_stratum_regresses_beyond_limit": all(
            float(row["fractional_gain_over_rar"])
            >= -float(admission["maximum_fractional_chi_square_regression_vs_rar_in_any_stratum"])
            for row in strata
        ),
        "outer_selected_formula_identical_in_all_folds": len(selection_counts) == 1,
        "per_galaxy_fitted_gravitational_constants_zero": True,
        "within_nfw_performance_ceiling": nested_chi <= nfw_limit,
    }
    passed = all(gate_checks.values())
    expected_point_evaluations = int(
        config["candidate_accounting"]["declared_scoring_point_evaluations"]
    )
    if complete_run and grid_point_evaluations != expected_point_evaluations:
        raise GravityG4FirstPrinciplesError("mechanism point accounting changed")
    obligations = {
        "causal_initial_value_formulation": "PENDING",
        "cluster_forward_model": "LOCKED_G4_BLOCKED",
        "conservation_identity": "PENDING",
        "covariant_completion": "PENDING",
        "dimensionally_typed_weak_field_equation": "PASS",
        "positive_energy_and_stability": "PENDING",
        "same_field_lensing_prediction": "LOCKED_G4_BLOCKED",
        "solar_system_and_gravitational_wave_limits": "PENDING",
        "target_blind_source": "PASS",
    }
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G4_FIRST_PRINCIPLES_MECHANISM_SEARCH",
        "decision": (
            "PASS_G4_FIRST_PRINCIPLES_MECHANISM_FREEZE"
            if passed
            else "BLOCK_G4_FIRST_PRINCIPLES_MECHANISM_SEARCH"
        ),
        "claims": {
            "alternative_to_gr_discovered": False,
            "cluster_mechanism_validated": False,
            "confirmation_authorized": passed,
            "covariant_first_principles_theory_derived": False,
            "historical_novelty_established": False,
            "lensing_mechanism_validated": False,
            "nested_whole_galaxy_generalization_measured": complete_run,
            "ten_mechanism_lanes_implemented": complete_run,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "controls": controls,
        "counts": {
            "candidate_structures": len(mechanisms),
            "coefficient_cells_per_selection": len(mechanisms) * len(COEFFICIENT_GRID),
            "confirmation_evaluator_accesses": 0,
            "cross_scale_cluster_evaluator_accesses": 0,
            "cross_scale_lensing_evaluator_accesses": 0,
            "exploration_galaxies": len(packets),
            "exploration_points": point_count,
            "galaxies_beating_rar": galaxies_beating_rar,
            "grid_scoring_point_evaluations": grid_point_evaluations,
            "known_positive_controls": sum(
                row["role"] == "known_positive_control" for row in mechanisms
            ),
            "mechanism_candidates": sum(row["role"] == "mechanism" for row in mechanisms),
            "mechanism_lanes": len({row["lane"] for row in mechanisms}),
            "outer_folds": folds,
            "per_galaxy_fitted_gravitational_constants": 0,
        },
        "final_all_exploration_selection": {
            "best_any_cell_including_known_control": best_all_cell,
            "best_mechanism_cell": final_cell,
            "best_mechanism_in_sample_chi_square": _metric(final_chi),
            "best_mechanism_invalid_prediction2": final_invalid,
            "generalization_evidence": False,
        },
        "first_principles_obligations": obligations,
        "gate_checks": gate_checks,
        "galaxies": per_galaxy,
        "lane_results": lane_results,
        "outer_fold_ledger": outer_ledger,
        "outer_selection_stability": {
            "distinct_selected_cells": len(selection_counts),
            "selection_counts": [
                {
                    "beta": beta,
                    "candidate_id": candidate_id,
                    "folds": count,
                }
                for (candidate_id, beta), count in sorted(
                    selection_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ],
        },
        "scores": {
            "fractional_gain_over_empirical_rar": _metric(1.0 - nested_chi / rar_chi),
            "nested_oof_mechanism_pipeline_chi_square": _metric(nested_chi),
            "nfw_ceiling_chi_square": _metric(nfw_chi),
            "nfw_ceiling_excess": _metric(nested_chi - nfw_limit),
            "nfw_ceiling_limit_with_slack": _metric(nfw_limit),
        },
        "strata": strata,
        "limitations": [
            "The outer-fold score estimates the mechanism-selection pipeline; folds may select different formulas and it is not automatically one frozen law.",
            "Every mechanism is currently an effective radial weak-field projection, not a completed covariant action.",
            "SPARC random-error scoring is not a complete systematic covariance likelihood.",
            "Cluster and lensing lanes are represented as obligations only because their evaluators remain locked behind G4.",
            "Proposer origin labels preserve ideas but do not establish historical novelty.",
        ],
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "predecessor": _binding(root, str(config["predecessor_binding"]["path"])),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG4FirstPrinciplesError("mechanism receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityG4FirstPrinciplesError("mechanism receipt seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG4FirstPrinciplesError("mechanism config binding changed")
    expected = {
        "config": CONFIG_PATH,
        "predecessor": str(config["predecessor_binding"]["path"]),
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    for key, path in expected.items():
        if receipt.get("source_bindings", {}).get(key) != _binding(root, path):
            raise GravityG4FirstPrinciplesError(f"mechanism {key} binding changed")
    counts = receipt.get("counts", {})
    if any(
        counts.get(key) != 0
        for key in (
            "confirmation_evaluator_accesses",
            "cross_scale_cluster_evaluator_accesses",
            "cross_scale_lensing_evaluator_accesses",
            "per_galaxy_fitted_gravitational_constants",
        )
    ):
        raise GravityG4FirstPrinciplesError("mechanism receipt violates data or fit lock")
    claims = receipt.get("claims", {})
    if claims.get("historical_novelty_established") is not False:
        raise GravityG4FirstPrinciplesError("mechanism receipt overstates novelty")
    if claims.get("covariant_first_principles_theory_derived") is not False:
        raise GravityG4FirstPrinciplesError("mechanism receipt overstates derivation")
    passed = receipt.get("decision") == "PASS_G4_FIRST_PRINCIPLES_MECHANISM_FREEZE"
    if passed and (
        not all(receipt.get("gate_checks", {}).values())
        or claims.get("confirmation_authorized") is not True
    ):
        raise GravityG4FirstPrinciplesError("mechanism PASS is unsupported")
    if not passed and claims.get("confirmation_authorized") is not False:
        raise GravityG4FirstPrinciplesError("blocked mechanism run authorizes confirmation")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityG4FirstPrinciplesError(
                f"refusing to overwrite immutable mechanism receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.validate_checked:
        validate_receipt(_load_json(root / OUTPUT_PATH), root=root)
        return 0
    receipt = build_receipt(root, candidate_limit=args.candidate_limit)
    if args.candidate_limit is None:
        _write_immutable(root / OUTPUT_PATH, receipt)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "counts": receipt["counts"],
                "decision": receipt["decision"],
                "final_all_exploration_selection": receipt["final_all_exploration_selection"],
                "scores": receipt["scores"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LANE_IDS",
    "OUTPUT_PATH",
    "GravityG4FirstPrinciplesError",
    "build_receipt",
    "load_config",
    "materialize_mechanisms",
    "mechanism_specs",
    "validate_receipt",
]
