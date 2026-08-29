from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc as v1
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v2 as v2
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_sbc_v3.json")
ARTIFACT_DIR = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sbc-v3"
)
RESULT_PATH = ARTIFACT_DIR / "bounded-synthetic-sbc-v3.npz"
RECEIPT_PATH = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sbc-v3.json"
)
CONFIG_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-config-3.0"
RESULT_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-result-3.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-receipt-3.0"

SCENARIOS = json.loads(json.dumps(v2.SCENARIOS))
REFERENCE = json.loads(json.dumps(v2.REFERENCE))
RANK_PROTOCOL = json.loads(json.dumps(v2.RANK_PROTOCOL))
GATES = json.loads(json.dumps(v2.GATES))

TRANSPORT_BLOCKS = [
    {
        "block_id": "direct_dynamics",
        "primitive_indices": [0, 1],
        "quotient_coordinates": [
            "outer_nonthermal_fraction",
            "nonthermal_radial_power",
        ],
        "initial_beta": 0.55,
    },
    {
        "block_id": "direct_boundary_density",
        "primitive_indices": [3, 4],
        "quotient_coordinates": [
            "outer_pressure_boundary_sigma",
            "density_error_sigma",
        ],
        "initial_beta": 0.55,
    },
    {
        "block_id": "direct_source_clumping_acceleration",
        "primitive_indices": [11, 12, 16],
        "quotient_coordinates": [
            "missing_stellar_to_gas_mass_ratio",
            "clumping_amplitude",
            "spherical_acceleration_scale",
        ],
        "initial_beta": 0.45,
    },
    {
        "block_id": "geometry_temperature_quotient",
        "primitive_indices": [2, 13, 14, 15],
        "quotient_coordinates": [
            "projected_gas_geometry_scale",
            "published_stellar_acceleration_scale",
            "temperature_density_calibration_scale",
        ],
        "initial_beta": 0.35,
    },
    {
        "block_id": "stellar_product_pushforward",
        "primitive_indices": [5, 6, 7, 8, 9, 10],
        "quotient_coordinates": ["published_stellar_acceleration_scale"],
        "initial_beta": 0.45,
    },
    {
        "block_id": "global_prior_transport",
        "primitive_indices": list(range(17)),
        "quotient_coordinates": list(sampler.COMPOSITES),
        "initial_beta": 0.18,
    },
]

CANDIDATE_INFERENCE = {
    "replicates": 2,
    "particles_per_replicate": 16,
    "adaptation_sweeps": 64,
    "fixed_kernel_settling_sweeps": 128,
    "retained_sweeps": 1024,
    "thin": 8,
    "retained_snapshots_per_particle_chain": 128,
    "transport_kernel": (
        "sequential_quotient_informed_gaussianized_primitive_pcn_blocks_"
        "metropolis_corrected_by_likelihood_ratio"
    ),
    "transport_blocks": TRANSPORT_BLOCKS,
    "beta_bounds": [0.03, 0.95],
    "target_acceptance": 0.30,
    "adaptation_gain": 0.35,
    "adaptation_schedule": "gain_divided_by_sqrt_one_plus_sweep_then_frozen",
    "stellar_log_step": 0.08,
    "geometry_log_step": 0.03,
    "coupled_log_step": 0.02,
    "orbit_moves": list(sampler.ORBIT_NAMES),
    "structural_change_from_v2": {
        "v2_bounded_correlated_random_walk_removed": True,
        "v2_out_of_bounds_active_proposals_removed": True,
        "gaussianized_prior_reversible_transport_added": True,
        "quotient_informed_blocking_added": True,
        "global_transport_bridge_added": True,
        "adaptation_settling_retention_counts_unchanged": True,
        "reason": (
            "V2 candidate/reference ranks and coverage agreed while maximum Rhat "
            "remained above 1.75 and minimum bulk ESS remained below 50"
        ),
    },
}

KERNEL_CONTROL = {
    "seed": 920000,
    "detailed_balance_pairs_per_block": 4096,
    "maximum_log_flux_residual": 5e-11,
    "prior_invariance_draws": 65536,
    "maximum_uniform_marginal_ks": 0.012,
    "maximum_composite_two_sample_ks": 0.015,
    "maximum_stellar_clip_atom_frequency_difference": 0.005,
    "proof": (
        "For each fixed primitive block, z=Phi^-1(u) has a standard-normal "
        "prior and z'=sqrt(1-beta^2)z+beta*xi is reversible with respect to "
        "that prior. The Metropolis likelihood ratio therefore targets the exact "
        "17-uniform primitive posterior. Each fixed block is invariant; their "
        "fixed sequential composition and the canonical prior-invariant orbit "
        "composition are invariant even though the complete sweep need not be "
        "reversible. Phi maps back to the open unit cube, after which the exact "
        "clipped six-factor stellar pushforward is unchanged."
    ),
}

SEED_LINEAGE = {
    "truth_base": 810000,
    "noise_base": 820000,
    "candidate_sobol_start_base": 930000,
    "candidate_transition_base": 940000,
    "candidate_rank_tie_base": 950000,
    "reference_base_owned_by_bound_v2_implementation": 860000,
    "reference_rank_tie_base": 870000,
    "scenario_stride": 10000,
    "simulation_stride": 10,
    "paired_truth_noise_and_reference_with_v2": True,
    "candidate_seeds_new_and_not_derived_from_v2_results": True,
}

DATA_BOUNDARY = {
    "synthetic_data_only": True,
    "real_development_rows_loaded": 0,
    "real_holdout_rows_loaded": 0,
    "real_confirmation_rows_loaded": 0,
    "real_independent_rows_loaded": 0,
    "network_calls": 0,
    "paid_model_calls": 0,
    "candidate_production_runs": 0,
}

CLAIM_BOUNDARY = {
    "calibrates_synthetic_quotient_inference_only": True,
    "calibrates_full_cluster_forward_model": False,
    "candidate_physics_supported": False,
    "candidate_production_completed": False,
    "CP5_7_through_CP5_10_complete": False,
    "scientific_claim_allowed": False,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return target


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def maximum_call_accounting() -> dict[str, int]:
    simulations = sum(int(row["simulations"]) for row in SCENARIOS)
    chains = int(CANDIDATE_INFERENCE["replicates"]) * int(
        CANDIDATE_INFERENCE["particles_per_replicate"]
    )
    sweeps = (
        int(CANDIDATE_INFERENCE["adaptation_sweeps"])
        + int(CANDIDATE_INFERENCE["fixed_kernel_settling_sweeps"])
        + int(CANDIDATE_INFERENCE["retained_sweeps"])
    )
    candidate = simulations * chains * (1 + sweeps * len(TRANSPORT_BLOCKS))
    reference_per_replicate = int(REFERENCE["particles_per_replicate"]) * (
        1
        + int(REFERENCE["maximum_tempering_stages"])
        * int(REFERENCE["coordinate_mh_sweeps_per_stage"])
    )
    reference = simulations * int(REFERENCE["replicates"]) * reference_per_replicate
    return {
        "simulations": simulations,
        "maximum_candidate_synthetic_likelihood_evaluations": candidate,
        "maximum_reference_synthetic_likelihood_evaluations": reference,
        "maximum_total_synthetic_likelihood_evaluations": candidate + reference,
        "real_forward_model_evaluations": 0,
    }


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if file_sha256(target) != expected_sha256:
        raise RuntimeError("SBC V3 config hash changed")
    config = json.loads(target.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "purpose",
        "implementation_source",
        "implementation_source_normalized_sha256",
        "frozen_evidence_bindings",
        "exact_primitive_priors",
        "primitive_prior_semantics",
        "composite_scales",
        "scenarios",
        "candidate_inference",
        "independent_reference",
        "rank_protocol",
        "gates",
        "kernel_control",
        "seed_lineage",
        "data_boundary",
        "claim_boundary",
        "call_accounting",
        "chronology",
        "output_paths",
    }
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise RuntimeError("SBC V3 config keys changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "separately_preregistered_before_single_bounded_v3_run"
    ):
        raise RuntimeError("SBC V3 config identity changed")
    implementation = confined(ROOT / config["implementation_source"])
    if implementation != Path(__file__).resolve() or normalized_sha256(
        implementation
    ) != config["implementation_source_normalized_sha256"]:
        raise RuntimeError("SBC V3 implementation changed after preregistration")
    for label, row in config["frozen_evidence_bindings"].items():
        bound = confined(ROOT / row["path"])
        if file_sha256(bound) != row["file_sha256"]:
            raise RuntimeError(f"SBC V3 frozen evidence changed: {label}")
    uncertainty_path = ROOT / config["frozen_evidence_bindings"][
        "uncertainty_config"
    ]["path"]
    uncertainty_config = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    if (
        config["exact_primitive_priors"] != uncertainty_config["continuous_priors"]
        or config["exact_primitive_priors"] != sampler.PRIMITIVE_PRIORS
        or len(config["exact_primitive_priors"]) != 17
    ):
        raise RuntimeError("SBC V3 exact 17-primitive prior changed")
    expected = {
        "primitive_prior_semantics": (
            "17_independent_uniform_primitives_with_clipped_six_factor_"
            "stellar_pushforward_clip_0.4_2.5"
        ),
        "composite_scales": v1.COMPOSITE_SCALES.tolist(),
        "scenarios": SCENARIOS,
        "candidate_inference": CANDIDATE_INFERENCE,
        "independent_reference": REFERENCE,
        "rank_protocol": RANK_PROTOCOL,
        "gates": GATES,
        "kernel_control": KERNEL_CONTROL,
        "seed_lineage": SEED_LINEAGE,
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "call_accounting": maximum_call_accounting(),
        "chronology": {
            "v2_mixing_failure_diagnosed_before_v3_design": True,
            "v1_and_v2_files_unchanged": True,
            "structural_kernel_selected_before_v3_run": True,
            "source_config_scenarios_gates_budget_and_seeds_frozen_before_first_run": True,
            "exactly_one_bounded_v3_run_allowed": True,
            "result_written_before_receipt": True,
            "failed_result_retained": True,
            "post_result_threshold_or_kernel_changes_forbidden": True,
        },
        "output_paths": {
            "result": RESULT_PATH.as_posix(),
            "receipt": RECEIPT_PATH.as_posix(),
        },
    }
    for name, value in expected.items():
        if config[name] != value:
            raise RuntimeError(f"frozen SBC V3 object changed: {name}")
    config["_config_sha256"] = expected_sha256
    return config


class CandidateLikelihood:
    def __init__(
        self,
        observation: np.ndarray,
        inverse_covariance: np.ndarray,
        prior_config: dict[str, Any],
    ) -> None:
        self.observation = observation
        self.inverse_covariance = inverse_covariance
        self.prior_config = prior_config
        self.calls = 0

    def __call__(self, unit: np.ndarray) -> float:
        return float(self.batch(unit[None, :])[0])

    def batch(self, units: np.ndarray) -> np.ndarray:
        self.calls += len(units)
        return v1.batch_log_likelihood(
            units, self.observation, self.inverse_covariance, self.prior_config
        )


def candidate_sobol_starts(global_index: int) -> np.ndarray:
    populations = []
    for replicate in range(int(CANDIDATE_INFERENCE["replicates"])):
        seed = (
            int(SEED_LINEAGE["candidate_sobol_start_base"])
            + global_index * int(SEED_LINEAGE["simulation_stride"])
            + replicate
        )
        populations.append(qmc.Sobol(d=17, scramble=True, seed=seed).random_base2(m=4))
    return np.stack(populations)


def pcn_block_proposal(
    particles: np.ndarray,
    primitive_indices: list[int],
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 0.0 < beta < 1.0:
        raise RuntimeError("pCN beta left the open unit interval")
    indices = np.asarray(primitive_indices, dtype=int)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise RuntimeError("pCN primitive block changed")
    epsilon = np.finfo(float).eps
    current = np.clip(particles[:, indices], epsilon, 1.0 - epsilon)
    latent = ndtri(current)
    rho = math.sqrt(1.0 - beta**2)
    proposal_latent = rho * latent + beta * rng.normal(size=latent.shape)
    proposals = particles.copy()
    proposals[:, indices] = ndtr(proposal_latent)
    if np.any(proposals <= 0.0) or np.any(proposals >= 1.0):
        raise RuntimeError("Gaussianized pCN proposal left the open primitive cube")
    return proposals


def pcn_block_transition(
    particles: np.ndarray,
    log_likelihood: np.ndarray,
    evaluator: CandidateLikelihood,
    block: dict[str, Any],
    beta: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    proposals = pcn_block_proposal(
        particles, list(map(int, block["primitive_indices"])), beta, rng
    )
    proposal_log_likelihood = evaluator.batch(proposals)
    accepted = np.log(
        np.maximum(rng.random(len(particles)), np.finfo(float).tiny)
    ) < np.minimum(0.0, proposal_log_likelihood - log_likelihood)
    particles[accepted] = proposals[accepted]
    log_likelihood[accepted] = proposal_log_likelihood[accepted]
    return {
        "attempted": len(particles),
        "evaluated": len(particles),
        "accepted": int(np.count_nonzero(accepted)),
        "out_of_bounds_rejected": 0,
    }


def _log_standard_normal(latent: np.ndarray) -> np.ndarray:
    return -0.5 * np.sum(latent**2, axis=1)


def _log_pcn_density(
    destination: np.ndarray, source: np.ndarray, indices: np.ndarray, beta: float
) -> np.ndarray:
    rho = math.sqrt(1.0 - beta**2)
    residual = destination[:, indices] - rho * source[:, indices]
    return -0.5 * np.sum(residual**2, axis=1) / beta**2


def _analytic_control_log_likelihood(
    latent: np.ndarray, prior_config: dict[str, Any]
) -> np.ndarray:
    units = ndtr(latent)
    values = sampler.composite_values(units, prior_config) / v1.COMPOSITE_SCALES
    center = np.asarray([0.6, 0.5, 0.0, 0.0, 0.7, 0.4, 1.0, 5.0, 1.5, 1.0])
    widths = np.asarray([0.8, 0.8, 1.2, 1.0, 0.8, 0.8, 0.8, 1.5, 1.2, 0.8])
    return -0.5 * np.sum(((values - center) / widths) ** 2, axis=1)


def detailed_balance_control(prior_config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(KERNEL_CONTROL["seed"]))
    count = int(KERNEL_CONTROL["detailed_balance_pairs_per_block"])
    rows = []
    maximum_prior_residual = 0.0
    maximum_posterior_residual = 0.0
    for block in TRANSPORT_BLOCKS:
        beta = float(block["initial_beta"])
        indices = np.asarray(block["primitive_indices"], dtype=int)
        current = rng.normal(size=(count, 17))
        proposed = current.copy()
        rho = math.sqrt(1.0 - beta**2)
        proposed[:, indices] = (
            rho * current[:, indices]
            + beta * rng.normal(size=(count, len(indices)))
        )
        forward_q = _log_pcn_density(proposed, current, indices, beta)
        reverse_q = _log_pcn_density(current, proposed, indices, beta)
        prior_residual = np.max(
            np.abs(
                _log_standard_normal(current)
                + forward_q
                - _log_standard_normal(proposed)
                - reverse_q
            )
        )
        current_ll = _analytic_control_log_likelihood(current, prior_config)
        proposed_ll = _analytic_control_log_likelihood(proposed, prior_config)
        forward_flux = (
            _log_standard_normal(current)
            + current_ll
            + forward_q
            + np.minimum(0.0, proposed_ll - current_ll)
        )
        reverse_flux = (
            _log_standard_normal(proposed)
            + proposed_ll
            + reverse_q
            + np.minimum(0.0, current_ll - proposed_ll)
        )
        posterior_residual = float(np.max(np.abs(forward_flux - reverse_flux)))
        maximum_prior_residual = max(maximum_prior_residual, float(prior_residual))
        maximum_posterior_residual = max(
            maximum_posterior_residual, posterior_residual
        )
        rows.append(
            {
                "block_id": block["block_id"],
                "pairs": count,
                "prior_reversibility_log_residual": float(prior_residual),
                "metropolis_posterior_log_flux_residual": posterior_residual,
            }
        )
    threshold = float(KERNEL_CONTROL["maximum_log_flux_residual"])
    return {
        "proof": KERNEL_CONTROL["proof"],
        "blocks": rows,
        "maximum_prior_reversibility_log_residual": maximum_prior_residual,
        "maximum_metropolis_posterior_log_flux_residual": (
            maximum_posterior_residual
        ),
        "threshold": threshold,
        "passed": bool(
            maximum_prior_residual <= threshold
            and maximum_posterior_residual <= threshold
        ),
    }


def _uniform_ks(values: np.ndarray) -> float:
    ordered = np.sort(values)
    count = len(ordered)
    upper = np.max(np.arange(1, count + 1) / count - ordered)
    lower = np.max(ordered - np.arange(count) / count)
    return float(max(upper, lower))


def _two_sample_ks(left: np.ndarray, right: np.ndarray) -> float:
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    pooled = np.sort(np.concatenate([left_sorted, right_sorted]))
    left_cdf = np.searchsorted(left_sorted, pooled, side="right") / len(left_sorted)
    right_cdf = np.searchsorted(right_sorted, pooled, side="right") / len(
        right_sorted
    )
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _raw_stellar_product(units: np.ndarray, prior_config: dict[str, Any]) -> np.ndarray:
    physical = sampler.physical_values(units, prior_config)
    return (
        physical[:, 5]
        * physical[:, 6]
        * (1.0 + physical[:, 7])
        * (1.0 + physical[:, 8])
        * physical[:, 9]
        * physical[:, 10]
    )


def prior_invariance_control(prior_config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(KERNEL_CONTROL["seed"]) + 1)
    count = int(KERNEL_CONTROL["prior_invariance_draws"])
    initial = rng.random((count, 17))
    transported = initial.copy()
    for block in TRANSPORT_BLOCKS:
        transported = pcn_block_proposal(
            transported,
            list(map(int, block["primitive_indices"])),
            float(block["initial_beta"]),
            rng,
        )
    marginal_ks = np.asarray([_uniform_ks(transported[:, i]) for i in range(17)])
    initial_composite = sampler.composite_values(initial, prior_config)
    transported_composite = sampler.composite_values(transported, prior_config)
    composite_ks = np.asarray(
        [
            _two_sample_ks(initial_composite[:, i], transported_composite[:, i])
            for i in range(len(sampler.COMPOSITES))
        ]
    )
    initial_raw = _raw_stellar_product(initial, prior_config)
    transported_raw = _raw_stellar_product(transported, prior_config)
    atom_rows = []
    maximum_atom_difference = 0.0
    for atom, predicate in (
        ("lower_clip_0p4", lambda values: values <= 0.4),
        ("upper_clip_2p5", lambda values: values >= 2.5),
    ):
        initial_frequency = float(np.mean(predicate(initial_raw)))
        transported_frequency = float(np.mean(predicate(transported_raw)))
        difference = abs(initial_frequency - transported_frequency)
        maximum_atom_difference = max(maximum_atom_difference, difference)
        atom_rows.append(
            {
                "atom": atom,
                "initial_frequency": initial_frequency,
                "transported_frequency": transported_frequency,
                "absolute_difference": difference,
            }
        )
    marginal_threshold = float(KERNEL_CONTROL["maximum_uniform_marginal_ks"])
    composite_threshold = float(
        KERNEL_CONTROL["maximum_composite_two_sample_ks"]
    )
    atom_threshold = float(
        KERNEL_CONTROL["maximum_stellar_clip_atom_frequency_difference"]
    )
    return {
        "draws": count,
        "maximum_uniform_marginal_ks": float(np.max(marginal_ks)),
        "uniform_marginal_ks_threshold": marginal_threshold,
        "maximum_composite_two_sample_ks": float(np.max(composite_ks)),
        "composite_two_sample_ks_threshold": composite_threshold,
        "stellar_clip_atoms": atom_rows,
        "maximum_stellar_clip_atom_frequency_difference": maximum_atom_difference,
        "stellar_clip_atom_frequency_threshold": atom_threshold,
        "all_primitives_covered_by_non_global_blocks": sorted(
            {
                int(index)
                for block in TRANSPORT_BLOCKS[:-1]
                for index in block["primitive_indices"]
            }
        )
        == list(range(17)),
        "global_block_covers_all_primitives": TRANSPORT_BLOCKS[-1][
            "primitive_indices"
        ]
        == list(range(17)),
        "passed": bool(
            np.max(marginal_ks) <= marginal_threshold
            and np.max(composite_ks) <= composite_threshold
            and maximum_atom_difference <= atom_threshold
        ),
    }


def kernel_controls(prior_config: dict[str, Any]) -> dict[str, Any]:
    detailed_balance = detailed_balance_control(prior_config)
    prior_invariance = prior_invariance_control(prior_config)
    orbit = v1.orbit_invariance_control(prior_config)
    passed = bool(detailed_balance["passed"] and prior_invariance["passed"] and orbit["passed"])
    return {
        "detailed_balance": detailed_balance,
        "exact_prior_and_atom_invariance": prior_invariance,
        "canonical_orbit_invariance": orbit,
        "complete_sweep_property": (
            "fixed_composition_of_invariant_kernels_is_invariant; complete sweep "
            "is not claimed reversible"
        ),
        "passed": passed,
    }


def run_candidate_fit(
    observation: np.ndarray,
    inverse_covariance: np.ndarray,
    prior_config: dict[str, Any],
    global_index: int,
) -> tuple[np.ndarray, dict[str, Any], int]:
    particles_by_replicate = candidate_sobol_starts(global_index)
    replicates, particle_count, _ = particles_by_replicate.shape
    evaluator = CandidateLikelihood(observation, inverse_covariance, prior_config)
    likelihood_by_replicate = np.empty((replicates, particle_count))
    for replicate in range(replicates):
        likelihood_by_replicate[replicate] = evaluator.batch(
            particles_by_replicate[replicate]
        )
    retained = int(CANDIDATE_INFERENCE["retained_snapshots_per_particle_chain"])
    traces = np.empty(
        (replicates, particle_count, retained, len(sampler.COMPOSITES))
    )
    block_counts = {
        str(block["block_id"]): {
            "attempted": 0,
            "evaluated": 0,
            "accepted": 0,
            "out_of_bounds_rejected": 0,
        }
        for block in TRANSPORT_BLOCKS
    }
    ending_betas = []
    orbit_attempted = 0
    orbit_accepted = 0
    for replicate in range(replicates):
        particles = particles_by_replicate[replicate].copy()
        likelihood = likelihood_by_replicate[replicate].copy()
        rng = np.random.default_rng(
            int(SEED_LINEAGE["candidate_transition_base"])
            + global_index * int(SEED_LINEAGE["simulation_stride"])
            + replicate
        )
        betas = np.asarray(
            [float(block["initial_beta"]) for block in TRANSPORT_BLOCKS]
        )
        retained_index = 0
        phases = (
            ("adaptation", int(CANDIDATE_INFERENCE["adaptation_sweeps"])),
            (
                "settling",
                int(CANDIDATE_INFERENCE["fixed_kernel_settling_sweeps"]),
            ),
            ("retained", int(CANDIDATE_INFERENCE["retained_sweeps"])),
        )
        for phase, sweeps in phases:
            for sweep in range(sweeps):
                orbit = sampler.orbit_sweep(
                    particles, rng, prior_config, CANDIDATE_INFERENCE
                )
                orbit_attempted += sum(
                    int(orbit[f"{move}_attempted"])
                    for move in sampler.ORBIT_NAMES
                )
                orbit_accepted += sum(
                    int(orbit[f"{move}_accepted"])
                    for move in sampler.ORBIT_NAMES
                )
                for block_index, block in enumerate(TRANSPORT_BLOCKS):
                    transition = pcn_block_transition(
                        particles,
                        likelihood,
                        evaluator,
                        block,
                        float(betas[block_index]),
                        rng,
                    )
                    target = block_counts[str(block["block_id"])]
                    for key in target:
                        target[key] += int(transition[key])
                    if phase == "adaptation":
                        acceptance = transition["accepted"] / transition["attempted"]
                        gain = float(CANDIDATE_INFERENCE["adaptation_gain"]) / math.sqrt(
                            sweep + 1.0
                        )
                        betas[block_index] *= math.exp(
                            gain
                            * (
                                acceptance
                                - float(CANDIDATE_INFERENCE["target_acceptance"])
                            )
                        )
                        betas[block_index] = float(
                            np.clip(
                                betas[block_index],
                                float(CANDIDATE_INFERENCE["beta_bounds"][0]),
                                float(CANDIDATE_INFERENCE["beta_bounds"][1]),
                            )
                        )
                if phase == "retained" and (sweep + 1) % int(
                    CANDIDATE_INFERENCE["thin"]
                ) == 0:
                    traces[replicate, :, retained_index, :] = sampler.composite_values(
                        particles, prior_config
                    )
                    retained_index += 1
        if retained_index != retained:
            raise RuntimeError("SBC V3 candidate retained snapshot accounting changed")
        ending_betas.append(betas.tolist())
    chains = traces.reshape(
        replicates * particle_count, retained, len(sampler.COMPOSITES)
    )
    diagnostics = sampler.rank_split_diagnostics(chains)
    draws = chains.reshape(-1, len(sampler.COMPOSITES))
    valid = diagnostics["valid"]
    finite_rhat = diagnostics["rhat"][valid]
    bulk = diagnostics["bulk_ess"][valid]
    tail = diagnostics["tail_ess"][valid]
    total_attempted = sum(row["attempted"] for row in block_counts.values())
    total_accepted = sum(row["accepted"] for row in block_counts.values())
    return (
        draws,
        {
            "all_coordinates_diagnostic_valid": bool(np.all(valid)),
            "valid_coordinate_count": int(np.count_nonzero(valid)),
            "maximum_rhat": (
                float(np.max(finite_rhat)) if len(finite_rhat) else None
            ),
            "minimum_bulk_ess": float(np.min(bulk)) if len(bulk) else 0.0,
            "minimum_tail_ess": float(np.min(tail)) if len(tail) else 0.0,
            "transport_attempted": total_attempted,
            "transport_evaluated": sum(
                row["evaluated"] for row in block_counts.values()
            ),
            "transport_accepted": total_accepted,
            "transport_acceptance": total_accepted / total_attempted,
            "transport_out_of_bounds_self_loops": sum(
                row["out_of_bounds_rejected"] for row in block_counts.values()
            ),
            "transport_by_block": block_counts,
            "ending_betas_by_replicate": ending_betas,
            "orbit_attempted": orbit_attempted,
            "orbit_accepted": orbit_accepted,
            "initial_likelihoods_recomputed_fresh": replicates * particle_count,
        },
        evaluator.calls,
    )


def write_result(path: Path, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    def writer(handle: Any) -> None:
        np.savez_compressed(
            handle,
            **arrays,
            summary=np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False)),
        )

    sampler._write_then_publish_no_clobber(path, writer, suffix=".npz.tmp")


def run_bounded(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if confined(output) != ROOT / RESULT_PATH:
        raise RuntimeError("bounded SBC V3 output path changed")
    prior_config = uncertainty.load_config(ROOT)
    controls = kernel_controls(prior_config)
    if not controls["passed"]:
        raise RuntimeError("SBC V3 structural kernel controls failed before bounded run")
    levels = list(map(float, RANK_PROTOCOL["coverage_levels"]))
    candidate_ranks = []
    reference_ranks = []
    candidate_coverage_rows = []
    reference_coverage_rows = []
    truth_units = []
    scenario_indices = []
    candidate_ties = []
    reference_tie_mass = []
    candidate_summaries = []
    reference_summaries = []
    candidate_calls = 0
    reference_calls = 0
    global_index = 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        covariance, inverse_covariance = v2.scenario_covariance(scenario)
        covariance_root = np.linalg.cholesky(covariance)
        for simulation_index in range(int(scenario["simulations"])):
            seed_offset = (
                scenario_index * int(SEED_LINEAGE["scenario_stride"])
                + simulation_index * int(SEED_LINEAGE["simulation_stride"])
            )
            truth_unit = np.random.default_rng(
                int(SEED_LINEAGE["truth_base"]) + seed_offset
            ).random(17)
            truth = sampler.composite_values(truth_unit, prior_config)
            truth_normalized = truth / v1.COMPOSITE_SCALES
            noise = np.random.default_rng(
                int(SEED_LINEAGE["noise_base"]) + seed_offset
            ).normal(size=len(sampler.COMPOSITES))
            observation = truth_normalized + covariance_root @ noise
            candidate_draws, candidate_summary, calls = run_candidate_fit(
                observation, inverse_covariance, prior_config, global_index
            )
            candidate_calls += calls
            candidate_rank_values, ties = v1.randomized_integer_ranks(
                candidate_draws,
                truth,
                int(SEED_LINEAGE["candidate_rank_tie_base"]) + seed_offset,
            )
            reference_draws, reference_summary, calls = v2.independent_reference(
                observation, inverse_covariance, prior_config, seed_offset
            )
            reference_calls += calls
            reference_rank_values, tie_mass = v2.reference_rank(
                reference_draws,
                truth,
                int(SEED_LINEAGE["reference_rank_tie_base"]) + seed_offset,
            )
            candidate_ranks.append(candidate_rank_values)
            reference_ranks.append(reference_rank_values)
            candidate_coverage_rows.append(
                v1.coverage_rows(candidate_draws, truth, levels)
            )
            reference_coverage_rows.append(
                v2.reference_coverage(reference_draws, truth, levels)
            )
            truth_units.append(truth_unit)
            scenario_indices.append(scenario_index)
            candidate_ties.append(ties)
            reference_tie_mass.append(tie_mass)
            candidate_summaries.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "simulation_index": simulation_index,
                    **candidate_summary,
                }
            )
            reference_summaries.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "simulation_index": simulation_index,
                    **reference_summary,
                }
            )
            global_index += 1
    candidate_rank_array = np.asarray(candidate_ranks)
    reference_rank_array = np.asarray(reference_ranks)
    candidate_coverage_array = np.asarray(candidate_coverage_rows)
    reference_coverage_array = np.asarray(reference_coverage_rows)
    scenario_index_array = np.asarray(scenario_indices, dtype=int)
    scenario_results = []
    all_scenarios_pass = True
    tolerances = GATES["coverage_tolerances_for_32_simulations"]
    for scenario_index, scenario in enumerate(SCENARIOS):
        selected = scenario_index_array == scenario_index
        candidate_rank_summary = v2.summarize_rank(candidate_rank_array[selected])
        reference_rank_summary = v2.summarize_rank(reference_rank_array[selected])
        candidate_coverage_summary = v2.summarize_coverage(
            candidate_coverage_array[selected]
        )
        reference_coverage_summary = v2.summarize_coverage(
            reference_coverage_array[selected]
        )
        mean_difference = float(
            np.max(
                np.abs(
                    np.mean(candidate_rank_array[selected], axis=0)
                    - np.mean(reference_rank_array[selected], axis=0)
                )
            )
        )
        candidate_rows = [
            row
            for row in candidate_summaries
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        reference_rows = [
            row
            for row in reference_summaries
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        valid_fraction = float(
            np.mean([row["all_coordinates_diagnostic_valid"] for row in candidate_rows])
        )
        finite_rhats = [
            float(row["maximum_rhat"])
            for row in candidate_rows
            if row["maximum_rhat"] is not None
        ]
        candidate_coverage_pass = all(
            candidate_coverage_summary["maximum_absolute_error_by_level"][index]
            <= float(tolerances[str(level)])
            for index, level in enumerate(levels)
        )
        reference_coverage_pass = all(
            reference_coverage_summary["maximum_absolute_error_by_level"][index]
            <= float(tolerances[str(level)])
            for index, level in enumerate(levels)
        )
        scenario_pass = bool(
            candidate_rank_summary["maximum_absolute_mean_rank_z"]
            <= float(GATES["maximum_absolute_candidate_mean_rank_z"])
            and reference_rank_summary["maximum_absolute_mean_rank_z"]
            <= float(GATES["maximum_absolute_reference_mean_rank_z"])
            and mean_difference
            <= float(GATES["maximum_absolute_candidate_reference_mean_rank_difference"])
            and candidate_coverage_pass
            and reference_coverage_pass
            and valid_fraction
            >= float(GATES["minimum_fraction_fits_all_coordinates_diagnostic_valid"])
            and len(finite_rhats) == len(candidate_rows)
            and max(finite_rhats)
            <= float(GATES["maximum_rank_normalized_split_rhat"])
            and min(row["minimum_bulk_ess"] for row in candidate_rows)
            >= float(GATES["minimum_bulk_effective_samples_per_valid_coordinate"])
            and min(row["minimum_tail_ess"] for row in candidate_rows)
            >= float(GATES["minimum_tail_effective_samples_per_valid_coordinate"])
            and min(
                row["minimum_stage_conditional_ess_fraction"]
                for row in reference_rows
            )
            >= float(GATES["reference_minimum_stage_conditional_ess_fraction"])
            and max(
                row["maximum_standardized_replicate_mean_difference"]
                for row in reference_rows
            )
            <= float(GATES["reference_maximum_standardized_replicate_mean_difference"])
            and min(
                row["minimum_final_unique_initial_ancestor_fraction"]
                for row in reference_rows
            )
            >= float(GATES["reference_minimum_unique_initial_ancestor_fraction"])
            and all(row["all_replicates_reached_beta_one"] for row in reference_rows)
        )
        all_scenarios_pass &= scenario_pass
        scenario_results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "simulations": int(np.count_nonzero(selected)),
                "candidate_rank": candidate_rank_summary,
                "reference_rank": reference_rank_summary,
                "candidate_coverage": candidate_coverage_summary,
                "reference_coverage": reference_coverage_summary,
                "maximum_absolute_candidate_reference_mean_rank_difference": (
                    mean_difference
                ),
                "fraction_fits_all_coordinates_diagnostic_valid": valid_fraction,
                "maximum_fit_rhat": max(finite_rhats) if finite_rhats else None,
                "minimum_fit_bulk_ess": min(
                    row["minimum_bulk_ess"] for row in candidate_rows
                ),
                "minimum_fit_tail_ess": min(
                    row["minimum_tail_ess"] for row in candidate_rows
                ),
                "reference_minimum_stage_conditional_ess_fraction": min(
                    row["minimum_stage_conditional_ess_fraction"]
                    for row in reference_rows
                ),
                "reference_maximum_standardized_replicate_mean_difference": max(
                    row["maximum_standardized_replicate_mean_difference"]
                    for row in reference_rows
                ),
                "reference_minimum_unique_initial_ancestor_fraction": min(
                    row["minimum_final_unique_initial_ancestor_fraction"]
                    for row in reference_rows
                ),
                "candidate_coverage_passed": candidate_coverage_pass,
                "reference_coverage_passed": reference_coverage_pass,
                "passed": scenario_pass,
            }
        )
    actual_calls = candidate_calls + reference_calls
    maximum_calls = config["call_accounting"][
        "maximum_total_synthetic_likelihood_evaluations"
    ]
    if actual_calls > maximum_calls:
        raise RuntimeError("SBC V3 exceeded frozen maximum calls")
    passed = bool(all_scenarios_pass and controls["passed"])
    decision = (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V3_PASSED_NOT_PHYSICS_OR_PRODUCTION"
        if passed
        else "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V3_FAILED_RESULT_RETAINED"
    )
    summary = {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "config_sha256": config["_config_sha256"],
        "passed": passed,
        "v2_isolated_failure": {
            "independent_reference_calibration_passed": True,
            "candidate_maximum_rhat_exceeded_1p75": True,
            "candidate_minimum_bulk_ess_below_50": True,
            "v2_result_or_thresholds_changed": False,
        },
        "structural_kernel_change": CANDIDATE_INFERENCE[
            "structural_change_from_v2"
        ],
        "scenario_results": scenario_results,
        "candidate_fit_summaries": candidate_summaries,
        "independent_reference_summaries": reference_summaries,
        "kernel_controls": controls,
        "call_accounting": {
            "actual_candidate_synthetic_likelihood_evaluations": candidate_calls,
            "actual_independent_reference_synthetic_likelihood_evaluations": (
                reference_calls
            ),
            "actual_total_synthetic_likelihood_evaluations": actual_calls,
            "frozen_maximum_total_synthetic_likelihood_evaluations": maximum_calls,
            "real_forward_model_evaluations": 0,
        },
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "chronology": config["chronology"],
    }
    write_result(
        output,
        {
            "truth_units": np.asarray(truth_units),
            "scenario_indices": scenario_index_array,
            "candidate_normalized_ranks": candidate_rank_array,
            "reference_normalized_ranks": reference_rank_array,
            "candidate_coverage": candidate_coverage_array,
            "reference_coverage": reference_coverage_array,
            "candidate_tie_counts": np.asarray(candidate_ties),
            "reference_tie_mass": np.asarray(reference_tie_mass),
        },
        summary,
    )
    return summary


def write_receipt(
    config_path: Path,
    expected_config_sha256: str,
    result_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    result_target = confined(result_path)
    if result_target != ROOT / RESULT_PATH:
        raise RuntimeError("SBC V3 result path changed")
    with np.load(result_target, allow_pickle=False) as archive:
        summary = json.loads(str(archive["summary"].item()))
    if (
        summary["schema_version"] != RESULT_SCHEMA
        or summary["config_sha256"] != expected_config_sha256
        or summary["data_boundary"] != DATA_BOUNDARY
        or summary["claim_boundary"] != CLAIM_BOUNDARY
        or not summary["kernel_controls"]["passed"]
    ):
        raise RuntimeError("SBC V3 result boundary or kernel control changed")
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": (
            "bounded_synthetic_sbc_v3_passed_not_candidate_production"
            if summary["passed"]
            else "bounded_synthetic_sbc_v3_failed_result_retained"
        ),
        "decision": summary["decision"],
        "evidence": {
            "config": artifact_binding(config_path),
            "implementation_source": artifact_binding(
                ROOT / config["implementation_source"]
            ),
            "v2_receipt": config["frozen_evidence_bindings"]["v2_receipt"],
            "v2_result": config["frozen_evidence_bindings"]["v2_result"],
            "canonical_sampler": config["frozen_evidence_bindings"][
                "canonical_sampler_source"
            ],
            "quotient_audit_receipt": config["frozen_evidence_bindings"][
                "quotient_audit_receipt"
            ],
            "bounded_v3_result": artifact_binding(result_target),
        },
        "v2_isolated_failure": summary["v2_isolated_failure"],
        "principled_v3_change": {
            "kernel": CANDIDATE_INFERENCE["transport_kernel"],
            "blocks": TRANSPORT_BLOCKS,
            "structural_change_from_v2": CANDIDATE_INFERENCE[
                "structural_change_from_v2"
            ],
            "sweep_counts_unchanged_from_v2": True,
            "scientific_gates_unchanged_from_v2": GATES == v2.GATES,
        },
        "counts": summary["call_accounting"],
        "scenario_results": summary["scenario_results"],
        "controls": summary["kernel_controls"],
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "V3 calibrates only the same synthetic Gaussian likelihood on the exact nuisance quotient, not the cluster forward model.",
            "The pCN blocks operate on Gaussianized primitive coordinates rather than sampling an explicit quotient density; exact quotient prior preservation follows from the unchanged 17-primitive pushforward.",
            "The V2 independent reference is finite-particle adaptive-tempering SMC, not an analytic posterior.",
            "The paired V2 truth/noise/reference seeds isolate the kernel change but do not constitute an independent second SBC experiment.",
            "A pass cannot support candidate physics, complete CP5.7-CP5.10, or authorize production; a failure is retained without threshold or kernel repair.",
        ],
        "replay": {
            "check": (
                "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_sbc_v3 "
                f"check --config {config_path.as_posix()} "
                f"--expected-config-sha256 {expected_config_sha256} "
                f"--receipt {output.as_posix()}"
            ),
            "candidate_production": "NOT_RUN",
        },
    }
    body["content_sha256"] = content_sha256(body)
    sampler.write_json(output, body)
    return body


def check(
    config_path: Path, expected_config_sha256: str, receipt_path: Path
) -> dict[str, Any]:
    load_config(config_path, expected_config_sha256)
    receipt_target = confined(receipt_path)
    body = json.loads(receipt_target.read_text(encoding="utf-8"))
    unhashed = dict(body)
    observed = unhashed.pop("content_sha256", None)
    if observed != content_sha256(unhashed):
        raise RuntimeError("SBC V3 receipt content hash changed")
    if body["schema_version"] != RECEIPT_SCHEMA:
        raise RuntimeError("SBC V3 receipt schema changed")
    for row in body["evidence"].values():
        target = confined(ROOT / row["path"])
        if file_sha256(target) != row["file_sha256"]:
            raise RuntimeError("SBC V3 receipt evidence changed")
    if body["data_boundary"] != DATA_BOUNDARY or body["claim_boundary"] != CLAIM_BOUNDARY:
        raise RuntimeError("SBC V3 receipt data or claim boundary changed")
    if not body["controls"]["passed"]:
        raise RuntimeError("SBC V3 structural kernel controls changed")
    return {
        "valid": True,
        "passed": body["status"]
        == "bounded_synthetic_sbc_v3_passed_not_candidate_production",
        "config_sha256": expected_config_sha256,
        "receipt_sha256": file_sha256(receipt_target),
        "synthetic_likelihood_evaluations": body["counts"][
            "actual_total_synthetic_likelihood_evaluations"
        ],
        "real_forward_model_evaluations": 0,
        "candidate_production_runs": 0,
        "scientific_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("controls")
    run_command = subparsers.add_parser("run-bounded")
    run_command.add_argument("--config", type=Path, required=True)
    run_command.add_argument("--expected-config-sha256", required=True)
    run_command.add_argument("--output", type=Path, required=True)
    receipt_command = subparsers.add_parser("write-receipt")
    receipt_command.add_argument("--config", type=Path, required=True)
    receipt_command.add_argument("--expected-config-sha256", required=True)
    receipt_command.add_argument("--result", type=Path, required=True)
    receipt_command.add_argument("--output", type=Path, required=True)
    check_command = subparsers.add_parser("check")
    check_command.add_argument("--config", type=Path, required=True)
    check_command.add_argument("--expected-config-sha256", required=True)
    check_command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "controls":
        print(json.dumps(kernel_controls(uncertainty.load_config(ROOT)), sort_keys=True))
        return
    if args.command == "run-bounded":
        config = load_config(args.config, args.expected_config_sha256)
        print(json.dumps(run_bounded(config, args.output), sort_keys=True))
        return
    if args.command == "write-receipt":
        print(
            json.dumps(
                write_receipt(
                    args.config,
                    args.expected_config_sha256,
                    args.result,
                    args.output,
                ),
                sort_keys=True,
            )
        )
        return
    print(
        json.dumps(
            check(args.config, args.expected_config_sha256, args.receipt),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
