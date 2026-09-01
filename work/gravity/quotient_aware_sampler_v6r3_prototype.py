from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_bound_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load hash-bound module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_bound_module(
    "invariant_v6r2_bound_support",
    Path(__file__).with_name("quotient_aware_sampler_v6r2_prototype.py"),
)
kernel = base.v6

SCHEMA = "invariant-gravity-cluster-quotient-sampler-contract-6.3"
UNAUTHORIZED_SCHEMA = "invariant-gravity-quotient-sampler-authorization-6.3-unauthorized"
AUTHORIZED_SCHEMA = "invariant-gravity-quotient-sampler-authorization-6.3-authorized"
APPROVAL_SCHEMA = "invariant-gravity-quotient-sampler-external-approval-6.3"
AUTHORIZATION_CONTROLS_SCHEMA = (
    "invariant-gravity-quotient-sampler-authorization-transition-controls-6.3"
)
COMPOSITES = kernel.COMPOSITES
ACTIVE_INDICES = kernel.ACTIVE_INDICES
ORBIT_NAMES = kernel.ORBIT_NAMES
PRIMITIVE_PRIORS = base.PRIMITIVE_PRIORS
MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS = 1_575_104

SOURCE_PATHS = {
    "v6r2_support": "work/gravity/quotient_aware_sampler_v6r2_prototype.py",
    "v6r1_kernel": "work/gravity/quotient_aware_sampler_v6_prototype.py",
    "uncertainty_config": "configs/gravity_cluster_uncertainty_program_v1.json",
    "uncertainty_module": "src/sigma_theory_compiler/gravity_cluster_uncertainty_program.py",
    "quotient_config": "configs/gravity_cluster_nuisance_quotient_audit_v1.json",
    "quotient_module": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_audit.py",
    "quotient_receipt": "runs/gravity/publication-readiness/nuisance-quotient-audit-v1.json",
    "comparator_module": "src/sigma_theory_compiler/gravity_cluster_comparator_suite.py",
    "comparator_receipt": "runs/gravity/publication-readiness/comparator-suite-v1.json",
    "item59_config": "configs/gravity_item59_xcop_forward_observable_gate_v1.json",
    "item59_module": "src/sigma_theory_compiler/gravity_item59_xcop_forward_observable_gate.py",
    "item59_result": "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json",
}

PRODUCTION_SETTINGS = {
    "replicates": 4,
    "particles": 512,
    "sobol_start_population_role": (
        "four_independently_scrambled_prior_populations_with_every_likelihood_recomputed"
    ),
    "adaptation_sweeps": 128,
    "fixed_kernel_settling_sweeps": 128,
    "retained_sweeps": 512,
    "thin": 2,
    "retained_snapshots_per_particle_chain": 256,
    "covariance_refresh_during_adaptation": 8,
    "initial_active_scale": 0.752622083091612,
    "active_scale_bounds": [0.02, 8.0],
    "target_acceptance": 0.234,
    "adaptation_gain": 0.5,
    "active_kernel": ("symmetric_correlated_gaussian_with_whole_proposal_out_of_bounds_rejection"),
    "active_primitive_indices": list(ACTIVE_INDICES),
    "stellar_log_step": 0.08,
    "geometry_log_step": 0.03,
    "coupled_log_step": 0.02,
    "orbit_validation_cases_per_move_per_replicate": 8,
    "seed": 597000,
}
SMOKE_SETTINGS = {
    **PRODUCTION_SETTINGS,
    "replicates": 2,
    "particles": 32,
    "adaptation_sweeps": 8,
    "fixed_kernel_settling_sweeps": 8,
    "retained_sweeps": 16,
    "thin": 2,
    "retained_snapshots_per_particle_chain": 8,
    "covariance_refresh_during_adaptation": 4,
    "orbit_validation_cases_per_move_per_replicate": 2,
}
START_GENERATION = base.START_GENERATION
DATA_SEAL = base.DATA_SEAL
ORBIT_VALIDATION = base.ORBIT_VALIDATION
DIAGNOSTIC_VALIDATION = base.DIAGNOSTIC_VALIDATION
UNIFORM_TARGET_CONTROL = base.UNIFORM_TARGET_CONTROL
COMPLETION_THRESHOLDS = {
    "maximum_rank_normalized_split_rhat": 1.2,
    "minimum_bulk_effective_samples": 50,
    "minimum_tail_effective_samples": 50,
    "maximum_standardized_between_replicate_median_spread": 0.25,
    "all_10_composite_coordinates_must_pass": True,
    "positive_variance_required_for_every_split_chain": True,
    "sobol_start_to_posterior_shift_is_a_gate": False,
}
MECHANICS_THRESHOLDS = base.MECHANICS_THRESHOLDS
ADJUDICATION = base.ADJUDICATION
AUTHORIZATION_POLICY = {
    "unauthorized_schema": UNAUTHORIZED_SCHEMA,
    "authorized_schema": AUTHORIZED_SCHEMA,
    "external_approval_schema": APPROVAL_SCHEMA,
    "separate_status_and_boundary_validation": True,
    "external_approval_must_bind_all_frozen_artifacts": True,
    "production_authorized_by_default": False,
    "explicit_cli_sentinel_required": True,
    "unauthorized_attempt_fails_before_contract_or_runtime_packet_load": True,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def confined(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return resolved


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def validate_artifact_binding(row: dict[str, Any], label: str) -> Path:
    strict_keys(row, {"path", "file_sha256"}, label)
    target = confined(ROOT / str(row["path"]))
    if not target.is_file() or file_sha256(target) != row["file_sha256"]:
        raise RuntimeError(f"artifact binding missing or tampered: {label}")
    return target


def validate_source_bindings(bindings: dict[str, Any]) -> None:
    strict_keys(bindings, set(SOURCE_PATHS), "source_bindings")
    for name, expected_path in SOURCE_PATHS.items():
        row = bindings[name]
        strict_keys(row, {"path", "file_sha256"}, f"source_bindings.{name}")
        if row["path"] != expected_path:
            raise RuntimeError(f"source binding path changed for {name}")
        validate_artifact_binding(row, f"source_bindings.{name}")


def validate_sampler_settings(
    settings: dict[str, Any], label: str, expected_replicates: int, expected_particles: int
) -> None:
    strict_keys(
        settings,
        {
            "replicates",
            "particles",
            "sobol_start_population_role",
            "adaptation_sweeps",
            "fixed_kernel_settling_sweeps",
            "retained_sweeps",
            "thin",
            "retained_snapshots_per_particle_chain",
            "covariance_refresh_during_adaptation",
            "initial_active_scale",
            "active_scale_bounds",
            "target_acceptance",
            "adaptation_gain",
            "active_kernel",
            "active_primitive_indices",
            "stellar_log_step",
            "geometry_log_step",
            "coupled_log_step",
            "orbit_validation_cases_per_move_per_replicate",
            "seed",
        },
        label,
    )
    expected_role = (
        "four_independently_scrambled_prior_populations_with_every_likelihood_recomputed"
    )
    if (
        int(settings["replicates"]) != expected_replicates
        or int(settings["particles"]) != expected_particles
        or settings["sobol_start_population_role"] != expected_role
        or settings["active_kernel"]
        != "symmetric_correlated_gaussian_with_whole_proposal_out_of_bounds_rejection"
        or tuple(settings["active_primitive_indices"]) != ACTIVE_INDICES
        or int(settings["retained_sweeps"]) % int(settings["thin"])
        or int(settings["retained_snapshots_per_particle_chain"])
        != int(settings["retained_sweeps"]) // int(settings["thin"])
        or int(settings["adaptation_sweeps"]) <= 0
        or int(settings["fixed_kernel_settling_sweeps"]) <= 0
        or int(settings["retained_sweeps"]) <= 0
        or int(settings["orbit_validation_cases_per_move_per_replicate"]) <= 0
    ):
        raise RuntimeError(f"{label} violates frozen sampler semantics")


def maximum_forward_calls(settings: dict[str, Any]) -> int:
    replicates = int(settings["replicates"])
    particles = int(settings["particles"])
    validation = (
        2
        * replicates
        * len(ORBIT_NAMES)
        * int(settings["orbit_validation_cases_per_move_per_replicate"])
    )
    proposals = (
        replicates
        * particles
        * (
            int(settings["adaptation_sweeps"])
            + int(settings["fixed_kernel_settling_sweeps"])
            + int(settings["retained_sweeps"])
        )
    )
    return validation + replicates * particles + proposals


def expected_call_accounting() -> dict[str, Any]:
    production_proposals = 4 * 512 * (128 + 128 + 512)
    return {
        "production_initialization_evaluations": 2048,
        "production_orbit_validation_evaluations": 192,
        "production_maximum_proposal_evaluations": production_proposals,
        "production_maximum_total_forward_evaluations": maximum_forward_calls(PRODUCTION_SETTINGS),
        "smoke_maximum_total_forward_evaluations": maximum_forward_calls(SMOKE_SETTINGS),
        "out_of_bounds_proposals_require_forward_evaluation": False,
        "actual_calls_must_equal_sum_of_reported_call_categories": True,
    }


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    contract_path = confined(path)
    observed_hash = file_sha256(contract_path)
    if observed_hash != expected_sha256:
        raise RuntimeError("contract hash differs from the expected hash")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    strict_keys(
        contract,
        {
            "schema_version",
            "status",
            "purpose",
            "prototype_source",
            "prototype_source_normalized_sha256",
            "source_bindings",
            "exact_primitive_priors",
            "primitive_prior_semantics",
            "train_packet",
            "sobol_start_population",
            "start_generation",
            "family",
            "likelihood_split",
            "data_seal",
            "production_settings",
            "smoke_settings",
            "orbit_validation",
            "diagnostic_validation",
            "uniform_target_invariance_control",
            "completion_thresholds",
            "mechanics_thresholds",
            "call_accounting",
            "adjudication",
            "authorization_policy",
        },
        "contract",
    )
    if contract["schema_version"] != SCHEMA:
        raise RuntimeError("contract schema changed")
    if contract["status"] != "frozen_v6r3_controls_and_smoke_only_production_unauthorized":
        raise RuntimeError("contract status changed")
    if (
        contract["family"] != "cross_scale_boundary"
        or contract["likelihood_split"] != "development_train"
    ):
        raise RuntimeError("family or likelihood split changed")
    prototype = confined(ROOT / str(contract["prototype_source"]))
    if prototype != Path(__file__).resolve():
        raise RuntimeError("contract points to another executable")
    if normalized_sha256(prototype) != contract["prototype_source_normalized_sha256"]:
        raise RuntimeError("V6R3 executable changed after freeze")
    validate_source_bindings(contract["source_bindings"])
    if contract["exact_primitive_priors"] != PRIMITIVE_PRIORS:
        raise RuntimeError("exact 17 primitive priors changed")
    if contract["primitive_prior_semantics"] != (
        "17_independent_uniform_primitives_with_clipped_six_factor_stellar_pushforward_clip_0.4_2.5"
    ):
        raise RuntimeError("primitive prior semantics changed")
    uncertainty_config = json.loads(
        (ROOT / SOURCE_PATHS["uncertainty_config"]).read_text(encoding="utf-8")
    )
    if uncertainty_config["continuous_priors"] != PRIMITIVE_PRIORS:
        raise RuntimeError("hash-bound uncertainty prior definitions changed")
    for name in ("train_packet", "sobol_start_population"):
        validate_artifact_binding(contract[name], name)
    base.load_train_packet(
        ROOT / contract["train_packet"]["path"],
        contract["train_packet"]["file_sha256"],
    )
    base.validate_sobol_starts(
        ROOT / contract["sobol_start_population"]["path"],
        contract["sobol_start_population"]["file_sha256"],
    )
    expected_nested = {
        "start_generation": START_GENERATION,
        "data_seal": DATA_SEAL,
        "production_settings": PRODUCTION_SETTINGS,
        "smoke_settings": SMOKE_SETTINGS,
        "orbit_validation": ORBIT_VALIDATION,
        "diagnostic_validation": DIAGNOSTIC_VALIDATION,
        "uniform_target_invariance_control": UNIFORM_TARGET_CONTROL,
        "completion_thresholds": COMPLETION_THRESHOLDS,
        "mechanics_thresholds": MECHANICS_THRESHOLDS,
        "call_accounting": expected_call_accounting(),
        "adjudication": ADJUDICATION,
        "authorization_policy": AUTHORIZATION_POLICY,
    }
    for name, expected in expected_nested.items():
        if contract[name] != expected:
            raise RuntimeError(f"frozen nested contract object changed: {name}")
    validate_sampler_settings(contract["production_settings"], "production settings", 4, 512)
    validate_sampler_settings(contract["smoke_settings"], "smoke settings", 2, 32)
    contract["_execution_contract_sha256"] = observed_hash
    return contract


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def phase_counter() -> dict[str, int]:
    return {
        "attempted": 0,
        "out_of_bounds_rejected": 0,
        "evaluated": 0,
        "accepted": 0,
    }


def atomic_save_result(
    output: Path,
    *,
    traces: np.ndarray,
    ending_particles: np.ndarray,
    ending_log_likelihood: np.ndarray,
    summary: dict[str, Any],
) -> None:
    output = confined(output)
    if output.exists():
        raise RuntimeError("refusing to overwrite an existing sampler result")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz.tmp", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(
                temporary,
                composite_traces=traces,
                ending_particles=ending_particles,
                ending_log_likelihood=ending_log_likelihood,
                summary=np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False)),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def run_sampler(
    contract: dict[str, Any], settings: dict[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    if confined(output).exists():
        raise RuntimeError("refusing to overwrite an existing sampler result")
    packets = base.load_train_packet(
        ROOT / contract["train_packet"]["path"],
        contract["train_packet"]["file_sha256"],
    )
    config = base.uncertainty.load_config(ROOT)
    config59 = base.item59.load_config(ROOT)
    family = config["candidate_and_control_families"][0]
    if family["family_id"] != contract["family"]:
        raise RuntimeError("loaded candidate family disagrees with contract")
    start_archive = np.load(ROOT / contract["sobol_start_population"]["path"], allow_pickle=False)
    all_start_particles = np.asarray(start_archive["particles"], dtype=float)
    replicates = int(settings["replicates"])
    particles_count = int(settings["particles"])
    if all_start_particles.shape != (4, 512, 17):
        raise RuntimeError("Sobol start population shape changed")
    particles_by_replicate = all_start_particles[:replicates, :particles_count].copy()
    evaluator = kernel.LikelihoodEvaluator(packets, family, config, config59)

    diagnostic_control = base.diagnostic_validation_control(contract["diagnostic_validation"])
    uniform_control = base.uniform_target_invariance_control(
        contract["uniform_target_invariance_control"], config
    )
    if not diagnostic_control["passed"] or not uniform_control["passed"]:
        raise RuntimeError("frozen controls failed before forward sampling")

    validation_rng = np.random.default_rng(int(settings["seed"]) + 900_000)
    orbit_validation = kernel.validate_orbits(
        particles_by_replicate,
        evaluator,
        validation_rng,
        config,
        settings,
        int(settings["orbit_validation_cases_per_move_per_replicate"]),
    )
    orbit_thresholds = contract["orbit_validation"]
    orbit_validation_passed = bool(
        orbit_validation["maximum_absolute_training_log_likelihood_difference"]
        <= float(orbit_thresholds["maximum_absolute_training_log_likelihood_difference"])
        and orbit_validation["maximum_absolute_composite_difference"]
        <= float(orbit_thresholds["maximum_absolute_composite_difference"])
        and all(int(orbit_validation["accepted_cases"][move]) > 0 for move in ORBIT_NAMES)
    )
    if not orbit_validation_passed:
        raise RuntimeError("orbit validation failed frozen invariance thresholds")

    calls_before_initialization = evaluator.calls
    log_likelihood_by_replicate = np.empty((replicates, particles_count))
    for replicate in range(replicates):
        for particle in range(particles_count):
            log_likelihood_by_replicate[replicate, particle] = evaluator(
                particles_by_replicate[replicate, particle]
            )
    initialization_evaluations = evaluator.calls - calls_before_initialization
    if initialization_evaluations != replicates * particles_count:
        raise RuntimeError("not every Sobol starting likelihood was recomputed fresh")

    retained_count = int(settings["retained_sweeps"]) // int(settings["thin"])
    traces = np.empty((replicates, particles_count, retained_count, len(COMPOSITES)), dtype=float)
    ending_particles = np.empty_like(particles_by_replicate)
    ending_log_likelihood = np.empty_like(log_likelihood_by_replicate)
    replicate_summaries = []
    phase_evaluations = {phase: 0 for phase in ("adaptation", "settling", "retained")}

    for replicate in range(replicates):
        particles = particles_by_replicate[replicate].copy()
        log_likelihood = log_likelihood_by_replicate[replicate].copy()
        rng = np.random.default_rng(int(settings["seed"]) + replicate)
        scale = float(settings["initial_active_scale"])
        square_root = kernel.covariance_square_root(particles[:, ACTIVE_INDICES])
        active_counts = {phase: phase_counter() for phase in ("adaptation", "settling", "retained")}
        orbit_counts = {
            phase: {
                f"{move}_{suffix}": 0
                for move in ORBIT_NAMES
                for suffix in ("attempted", "accepted")
            }
            for phase in ("adaptation", "settling", "retained")
        }
        retained_index = 0
        phases = (
            ("adaptation", int(settings["adaptation_sweeps"])),
            ("settling", int(settings["fixed_kernel_settling_sweeps"])),
            ("retained", int(settings["retained_sweeps"])),
        )
        for phase, sweeps in phases:
            for sweep in range(sweeps):
                if (
                    phase == "adaptation"
                    and sweep % int(settings["covariance_refresh_during_adaptation"]) == 0
                ):
                    square_root = kernel.covariance_square_root(particles[:, ACTIVE_INDICES])
                orbit_result = kernel.orbit_sweep(particles, rng, config, settings)
                add_counts(orbit_counts[phase], orbit_result)
                calls_before = evaluator.calls
                transition = kernel.active_transition(
                    particles,
                    log_likelihood,
                    evaluator,
                    rng,
                    square_root,
                    scale,
                )
                add_counts(active_counts[phase], transition)
                phase_evaluations[phase] += evaluator.calls - calls_before
                if phase == "adaptation":
                    rate = transition["accepted"] / transition["attempted"]
                    scale *= math.exp(
                        float(settings["adaptation_gain"])
                        * (rate - float(settings["target_acceptance"]))
                    )
                    scale = float(
                        np.clip(
                            scale,
                            float(settings["active_scale_bounds"][0]),
                            float(settings["active_scale_bounds"][1]),
                        )
                    )
                elif phase == "retained" and (sweep + 1) % int(settings["thin"]) == 0:
                    traces[replicate, :, retained_index, :] = kernel.composite_values(
                        particles, config
                    )
                    retained_index += 1
            if phase == "adaptation":
                square_root = kernel.covariance_square_root(particles[:, ACTIVE_INDICES])
        if retained_index != retained_count:
            raise RuntimeError("retained snapshot accounting changed")
        ending_particles[replicate] = particles
        ending_log_likelihood[replicate] = log_likelihood
        replicate_summaries.append(
            {
                "replicate": replicate,
                "final_active_scale": scale,
                "active_counts": active_counts,
                "orbit_counts": orbit_counts,
            }
        )

    chains = traces.reshape(replicates * particles_count, retained_count, len(COMPOSITES))
    diagnostics = base.rank_split_diagnostics(chains)
    pooled = traces.reshape(-1, len(COMPOSITES))
    pooled_standard_deviation = np.std(pooled, axis=0, ddof=1)
    replicate_medians = np.median(traces, axis=(1, 2))
    median_spread = np.divide(
        np.ptp(replicate_medians, axis=0),
        pooled_standard_deviation,
        out=np.zeros_like(pooled_standard_deviation),
        where=pooled_standard_deviation > np.finfo(float).tiny,
    )
    start_composites = kernel.composite_values(particles_by_replicate, config).reshape(
        -1, len(COMPOSITES)
    )
    descriptive_start_shift = np.divide(
        np.abs(np.median(pooled, axis=0) - np.median(start_composites, axis=0)),
        pooled_standard_deviation,
        out=np.zeros_like(pooled_standard_deviation),
        where=pooled_standard_deviation > np.finfo(float).tiny,
    )

    completion = contract["completion_thresholds"]
    coordinate_pass = (
        diagnostics["valid"]
        & (diagnostics["rhat"] <= float(completion["maximum_rank_normalized_split_rhat"]))
        & (diagnostics["bulk_ess"] >= float(completion["minimum_bulk_effective_samples"]))
        & (diagnostics["tail_ess"] >= float(completion["minimum_tail_effective_samples"]))
        & (
            median_spread
            <= float(completion["maximum_standardized_between_replicate_median_spread"])
        )
    )
    mechanics = contract["mechanics_thresholds"]
    mechanics_rows = []
    mechanics_pass = True
    for row in replicate_summaries:
        active = row["active_counts"]["retained"]
        active_rate = active["accepted"] / active["attempted"]
        active_pass = bool(
            float(mechanics["minimum_retained_active_acceptance"])
            <= active_rate
            <= float(mechanics["maximum_retained_active_acceptance"])
        )
        orbit_rates = {}
        orbit_pass = True
        for move in ORBIT_NAMES:
            counts = row["orbit_counts"]["retained"]
            rate = counts[f"{move}_accepted"] / counts[f"{move}_attempted"]
            orbit_rates[move] = rate
            orbit_pass &= rate >= float(mechanics["minimum_retained_orbit_acceptance"])
        passed = bool(active_pass and orbit_pass)
        mechanics_pass &= passed
        mechanics_rows.append(
            {
                "replicate": row["replicate"],
                "retained_active_acceptance": active_rate,
                "retained_orbit_acceptance": orbit_rates,
                "passed": passed,
            }
        )

    accounting = {
        "orbit_validation_evaluations": int(orbit_validation["evaluations"]),
        "initialization_evaluations": int(initialization_evaluations),
        "adaptation_proposal_evaluations": int(phase_evaluations["adaptation"]),
        "settling_proposal_evaluations": int(phase_evaluations["settling"]),
        "retained_proposal_evaluations": int(phase_evaluations["retained"]),
    }
    accounted_total = sum(accounting.values())
    if accounted_total != evaluator.calls:
        raise RuntimeError(
            f"forward evaluation accounting mismatch: {accounted_total} != {evaluator.calls}"
        )
    maximum_calls = maximum_forward_calls(settings)
    if evaluator.calls > maximum_calls:
        raise RuntimeError("actual forward calls exceeded frozen maximum")
    accounting["total_forward_evaluations"] = int(evaluator.calls)
    accounting["frozen_maximum_forward_evaluations"] = int(maximum_calls)

    gate_pass = bool(np.all(coordinate_pass) and mechanics_pass)
    production_passed = bool(gate_pass and not smoke)
    if smoke:
        decision = "SMOKE_ONLY_NOT_PRODUCTION_ADJUDICATION"
    elif production_passed:
        decision = "CANDIDATE_SAMPLER_PASS_DOWNSTREAM_SBC_NEWTONIAN_AND_SOURCE_COVARIANCE_REQUIRED"
    else:
        decision = "CANDIDATE_SAMPLER_FAIL_FROZEN_GATES_RESULT_RETAINED"
    finite_rhat = np.all(np.isfinite(diagnostics["rhat"]))
    aggregate = {
        "schema_version": "invariant-gravity-cluster-quotient-sampler-result-6.3",
        "mode": "smoke" if smoke else "production",
        "decision": decision,
        "execution_contract_sha256": contract["_execution_contract_sha256"],
        "family_id": family["family_id"],
        "sobol_start_population": {
            "role": settings["sobol_start_population_role"],
            "generation": START_GENERATION,
            "file_sha256": contract["sobol_start_population"]["file_sha256"],
            "posterior_ancestry": False,
            "stored_likelihoods": False,
            "every_initial_likelihood_recomputed_fresh": True,
        },
        "replicates": replicates,
        "particle_chains_per_replicate": particles_count,
        "retained_snapshots_per_chain": retained_count,
        "posterior_draws": len(pooled),
        "controls": {
            "diagnostic_validation": diagnostic_control,
            "uniform_target_invariance": uniform_control,
            "orbit_validation": orbit_validation,
        },
        "orbit_validation_passed": orbit_validation_passed,
        "forward_call_accounting": accounting,
        "maximum_rank_normalized_split_rhat": (
            float(np.max(diagnostics["rhat"])) if finite_rhat else None
        ),
        "minimum_bulk_effective_samples": float(np.min(diagnostics["bulk_ess"])),
        "minimum_tail_effective_samples": float(np.min(diagnostics["tail_ess"])),
        "maximum_standardized_between_replicate_median_spread": float(np.max(median_spread)),
        "maximum_descriptive_sobol_start_to_posterior_median_shift": float(
            np.max(descriptive_start_shift)
        ),
        "descriptive_sobol_start_to_posterior_shift_is_a_gate": False,
        "all_coordinates_positive_variance": bool(np.all(diagnostics["valid"])),
        "all_coordinate_gates_passed": bool(np.all(coordinate_pass)),
        "all_mechanics_gates_passed": bool(mechanics_pass),
        "production_passed": production_passed,
        "parameters": [
            {
                "coordinate": name,
                "diagnostic_valid_positive_variance": bool(diagnostics["valid"][index]),
                "minimum_scaled_within_chain_variance": float(
                    diagnostics["minimum_scaled_within_chain_variance"][index]
                ),
                "rank_normalized_split_rhat": (
                    float(diagnostics["rhat"][index])
                    if np.isfinite(diagnostics["rhat"][index])
                    else None
                ),
                "bulk_effective_samples": float(diagnostics["bulk_ess"][index]),
                "tail_effective_samples": float(diagnostics["tail_ess"][index]),
                "standardized_between_replicate_median_spread": float(median_spread[index]),
                "descriptive_sobol_start_to_posterior_median_shift": float(
                    descriptive_start_shift[index]
                ),
                "passed": bool(coordinate_pass[index]),
            }
            for index, name in enumerate(COMPOSITES)
        ],
        "mechanics": mechanics_rows,
        "replicate_summaries": replicate_summaries,
        "runtime_data_boundary": {
            "packet_sha256": contract["train_packet"]["file_sha256"],
            "allowed_split": "development_train",
            "rows_loaded": 80,
            "holdout_rows_loaded": 0,
            "confirmation_rows_loaded": 0,
            "independent_rows_loaded": 0,
            "canonical_comparator_packet_builder_called_during_sampling": False,
        },
        "downstream_sequence_if_candidate_passes": [
            "simulation_based_calibration",
            "matched_newtonian_control",
            "source_covariance",
        ],
        "claim_boundary": contract["adjudication"],
    }
    atomic_save_result(
        output,
        traces=traces,
        ending_particles=ending_particles,
        ending_log_likelihood=ending_log_likelihood,
        summary=aggregate,
    )
    return aggregate


def controls_receipt(contract: dict[str, Any]) -> dict[str, Any]:
    config = base.uncertainty.load_config(ROOT)
    diagnostic = base.diagnostic_validation_control(contract["diagnostic_validation"])
    uniform = base.uniform_target_invariance_control(
        contract["uniform_target_invariance_control"], config
    )
    return {
        "schema_version": "invariant-gravity-cluster-quotient-sampler-controls-6.3",
        "execution_contract_sha256": contract["_execution_contract_sha256"],
        "prototype_source_normalized_sha256": contract["prototype_source_normalized_sha256"],
        "train_packet_sha256": contract["train_packet"]["file_sha256"],
        "sobol_start_population_sha256": contract["sobol_start_population"]["file_sha256"],
        "diagnostic_validation": diagnostic,
        "uniform_target_invariance": uniform,
        "forward_evaluations": 0,
        "passed": bool(diagnostic["passed"] and uniform["passed"]),
    }


def validate_bound_controls_and_smoke(
    contract_hash: str, controls_path: Path, smoke_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = json.loads(confined(controls_path).read_text(encoding="utf-8"))
    smoke = np.load(confined(smoke_path), allow_pickle=False)
    strict_keys(
        {name: None for name in smoke.files},
        {
            "composite_traces",
            "ending_particles",
            "ending_log_likelihood",
            "summary",
        },
        "smoke archive",
    )
    summary = json.loads(str(smoke["summary"].item()))
    if (
        not controls.get("passed")
        or controls.get("execution_contract_sha256") != contract_hash
        or summary.get("execution_contract_sha256") != contract_hash
        or summary.get("mode") != "smoke"
        or summary.get("orbit_validation_passed") is not True
        or summary.get("sobol_start_population", {}).get(
            "every_initial_likelihood_recomputed_fresh"
        )
        is not True
    ):
        raise RuntimeError("controls or smoke do not satisfy the frozen V6R3 contract")
    return controls, summary


def frozen_artifact_bindings(
    contract: dict[str, Any], contract_path: Path, controls_path: Path, smoke_path: Path
) -> dict[str, dict[str, str]]:
    return {
        "contract": artifact_binding(contract_path),
        "prototype_source": artifact_binding(ROOT / contract["prototype_source"]),
        "train_packet": artifact_binding(ROOT / contract["train_packet"]["path"]),
        "sobol_start_population": artifact_binding(
            ROOT / contract["sobol_start_population"]["path"]
        ),
        "controls": artifact_binding(controls_path),
        "smoke": artifact_binding(smoke_path),
    }


def write_unauthorized_manifest(
    contract_path: Path,
    expected_contract_sha256: str,
    controls_path: Path,
    smoke_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path, expected_contract_sha256)
    validate_bound_controls_and_smoke(expected_contract_sha256, controls_path, smoke_path)
    body = {
        "schema_version": UNAUTHORIZED_SCHEMA,
        "status": "production_unauthorized_controls_and_smoke_bound",
        "artifact_bindings": frozen_artifact_bindings(
            contract, contract_path, controls_path, smoke_path
        ),
        "production_authorization": {
            "authorized": False,
            "approved_by": None,
            "approval_id": None,
            "maximum_forward_evaluations": 0,
        },
        "claim_boundary": {
            "controls_passed": True,
            "bounded_smoke_executed": True,
            "candidate_production_executed": False,
            "candidate_claim_allowed": False,
            "production_execution_allowed": False,
            "newtonian_control_unlocked": False,
            "simulation_based_calibration_unlocked": False,
        },
    }
    write_json(output, body)
    return body


def validate_frozen_artifact_bindings(
    bindings: dict[str, Any], *, load_bound_contract: bool
) -> dict[str, Any] | None:
    strict_keys(
        bindings,
        {
            "contract",
            "prototype_source",
            "train_packet",
            "sobol_start_population",
            "controls",
            "smoke",
        },
        "authorization.artifact_bindings",
    )
    paths = {
        name: validate_artifact_binding(row, f"authorization.artifact_bindings.{name}")
        for name, row in bindings.items()
    }
    if not load_bound_contract:
        return None
    contract = load_contract(paths["contract"], bindings["contract"]["file_sha256"])
    if paths["prototype_source"] != Path(__file__).resolve():
        raise RuntimeError("authorization does not bind this V6R3 executable")
    validate_bound_controls_and_smoke(
        contract["_execution_contract_sha256"], paths["controls"], paths["smoke"]
    )
    if bindings["train_packet"] != contract["train_packet"]:
        raise RuntimeError("authorization train packet differs from contract")
    if bindings["sobol_start_population"] != contract["sobol_start_population"]:
        raise RuntimeError("authorization Sobol population differs from contract")
    return contract


def validate_unauthorized_body(
    body: dict[str, Any], *, load_bound_contract: bool
) -> dict[str, Any] | None:
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "artifact_bindings",
            "production_authorization",
            "claim_boundary",
        },
        "unauthorized manifest",
    )
    if body["schema_version"] != UNAUTHORIZED_SCHEMA:
        raise RuntimeError("unauthorized manifest schema changed")
    if body["status"] != "production_unauthorized_controls_and_smoke_bound":
        raise RuntimeError("unauthorized manifest status changed")
    expected_authorization = {
        "authorized": False,
        "approved_by": None,
        "approval_id": None,
        "maximum_forward_evaluations": 0,
    }
    if body["production_authorization"] != expected_authorization:
        raise RuntimeError("unauthorized production fields changed")
    expected_boundary = {
        "controls_passed": True,
        "bounded_smoke_executed": True,
        "candidate_production_executed": False,
        "candidate_claim_allowed": False,
        "production_execution_allowed": False,
        "newtonian_control_unlocked": False,
        "simulation_based_calibration_unlocked": False,
    }
    if body["claim_boundary"] != expected_boundary:
        raise RuntimeError("unauthorized claim boundary changed")
    return validate_frozen_artifact_bindings(
        body["artifact_bindings"], load_bound_contract=load_bound_contract
    )


def validate_external_approval(
    path: Path,
    expected_sha256: str,
    exact_artifact_bindings: dict[str, Any],
) -> dict[str, Any]:
    approval_path = confined(path)
    if not approval_path.is_file() or file_sha256(approval_path) != expected_sha256:
        raise RuntimeError("external approval record missing or tampered")
    body = json.loads(approval_path.read_text(encoding="utf-8"))
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "approved_by",
            "approval_id",
            "maximum_forward_evaluations",
            "artifact_bindings",
        },
        "external approval record",
    )
    if body["schema_version"] != APPROVAL_SCHEMA:
        raise RuntimeError("external approval schema changed")
    if body["status"] != "explicit_external_production_approval":
        raise RuntimeError("external approval status changed")
    if body["approved_by"] != "Henry":
        raise RuntimeError("external approval must be approved_by Henry")
    if not isinstance(body["approval_id"], str) or not body["approval_id"].strip():
        raise RuntimeError("external approval_id must be nonempty")
    if int(body["maximum_forward_evaluations"]) != MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS:
        raise RuntimeError("external approval maximum call count changed")
    if body["artifact_bindings"] != exact_artifact_bindings:
        raise RuntimeError("external approval does not bind the exact frozen artifacts")
    return body


def promote_authorization(
    unauthorized_path: Path,
    expected_unauthorized_sha256: str,
    approval_path: Path,
    expected_approval_sha256: str,
    output: Path,
) -> dict[str, Any]:
    unauthorized_target = confined(unauthorized_path)
    if file_sha256(unauthorized_target) != expected_unauthorized_sha256:
        raise RuntimeError("unauthorized manifest hash differs from expected hash")
    unauthorized = json.loads(unauthorized_target.read_text(encoding="utf-8"))
    validate_unauthorized_body(unauthorized, load_bound_contract=True)
    approval = validate_external_approval(
        approval_path,
        expected_approval_sha256,
        unauthorized["artifact_bindings"],
    )
    body = {
        "schema_version": AUTHORIZED_SCHEMA,
        "status": "production_explicitly_authorized_by_external_approval",
        "artifact_bindings": unauthorized["artifact_bindings"],
        "external_approval_binding": {
            **artifact_binding(approval_path),
            "approval_id": approval["approval_id"],
        },
        "production_authorization": {
            "authorized": True,
            "approved_by": "Henry",
            "approval_id": approval["approval_id"],
            "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
        },
        "authorized_execution_boundary": {
            "production_execution_allowed": True,
            "candidate_result_exists": False,
            "candidate_claim_allowed_before_result": False,
            "production_result_must_retain_failed_gates": True,
            "simulation_based_calibration_follows_candidate_pass": True,
            "matched_newtonian_control_follows_sbc": True,
            "source_covariance_follows_newtonian": True,
        },
    }
    write_json(output, body)
    return body


def validate_authorized_body(body: dict[str, Any]) -> dict[str, Any]:
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "artifact_bindings",
            "external_approval_binding",
            "production_authorization",
            "authorized_execution_boundary",
        },
        "authorized manifest",
    )
    if body["schema_version"] != AUTHORIZED_SCHEMA:
        raise RuntimeError("authorized manifest schema changed")
    if body["status"] != "production_explicitly_authorized_by_external_approval":
        raise RuntimeError("authorized manifest status changed")
    production = body["production_authorization"]
    strict_keys(
        production,
        {
            "authorized",
            "approved_by",
            "approval_id",
            "maximum_forward_evaluations",
        },
        "authorized manifest.production_authorization",
    )
    if (
        production["authorized"] is not True
        or production["approved_by"] != "Henry"
        or not isinstance(production["approval_id"], str)
        or not production["approval_id"].strip()
        or int(production["maximum_forward_evaluations"]) != MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS
    ):
        raise RuntimeError("authorized production fields are incomplete")
    expected_boundary = {
        "production_execution_allowed": True,
        "candidate_result_exists": False,
        "candidate_claim_allowed_before_result": False,
        "production_result_must_retain_failed_gates": True,
        "simulation_based_calibration_follows_candidate_pass": True,
        "matched_newtonian_control_follows_sbc": True,
        "source_covariance_follows_newtonian": True,
    }
    if body["authorized_execution_boundary"] != expected_boundary:
        raise RuntimeError("authorized execution boundary changed")
    contract = validate_frozen_artifact_bindings(
        body["artifact_bindings"], load_bound_contract=True
    )
    if contract is None:
        raise RuntimeError("authorized contract was not loaded")
    approval_binding = body["external_approval_binding"]
    strict_keys(
        approval_binding,
        {"path", "file_sha256", "approval_id"},
        "authorized manifest.external_approval_binding",
    )
    approval = validate_external_approval(
        ROOT / approval_binding["path"],
        approval_binding["file_sha256"],
        body["artifact_bindings"],
    )
    if (
        approval["approval_id"] != approval_binding["approval_id"]
        or approval["approval_id"] != production["approval_id"]
    ):
        raise RuntimeError("authorized approval identifiers disagree")
    return contract


def validate_authorization(
    path: Path, expected_sha256: str, *, require_production: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authorization_path = confined(path)
    if not authorization_path.is_file() or file_sha256(authorization_path) != expected_sha256:
        raise RuntimeError("authorization manifest hash differs from expected hash")
    body = json.loads(authorization_path.read_text(encoding="utf-8"))
    schema = body.get("schema_version")
    if schema == UNAUTHORIZED_SCHEMA:
        if require_production:
            raise RuntimeError(
                "production is unauthorized; refusal occurs before contract or runtime packet load"
            )
        contract = validate_unauthorized_body(body, load_bound_contract=True)
        return body, contract
    if schema == AUTHORIZED_SCHEMA:
        contract = validate_authorized_body(body)
        return body, contract
    raise RuntimeError("authorization manifest has no recognized V6R3 schema")


def injected_approval_body(bindings: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    body = {
        "schema_version": APPROVAL_SCHEMA,
        "status": "explicit_external_production_approval",
        "approved_by": "Henry",
        "approval_id": "V6R3-AUTHORIZATION-TRANSITION-CONTROL-ONLY",
        "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
        "artifact_bindings": bindings,
    }
    body.update(overrides)
    return body


def authorization_transition_controls(
    unauthorized_path: Path, expected_unauthorized_sha256: str
) -> dict[str, Any]:
    unauthorized_target = confined(unauthorized_path)
    if file_sha256(unauthorized_target) != expected_unauthorized_sha256:
        raise RuntimeError("authorization-control input hash changed")
    unauthorized = json.loads(unauthorized_target.read_text(encoding="utf-8"))
    validate_unauthorized_body(unauthorized, load_bound_contract=True)
    negative_results: dict[str, bool] = {}
    positive_result: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="v6r3-authorization-control-", dir=ROOT / "work" / "gravity"
    ) as temporary_directory:
        temporary = Path(temporary_directory)

        def expect_rejection(name: str, approval: dict[str, Any]) -> None:
            approval_path = temporary / f"{name}-approval.json"
            output_path = temporary / f"{name}-authorized.json"
            write_json(approval_path, approval)
            try:
                promote_authorization(
                    unauthorized_target,
                    expected_unauthorized_sha256,
                    approval_path,
                    file_sha256(approval_path),
                    output_path,
                )
            except RuntimeError:
                negative_results[name] = not output_path.exists()
                return
            negative_results[name] = False

        bindings = unauthorized["artifact_bindings"]
        expect_rejection(
            "wrong_approver", injected_approval_body(bindings, approved_by="Not Henry")
        )
        expect_rejection("empty_approval_id", injected_approval_body(bindings, approval_id=""))
        expect_rejection(
            "wrong_maximum_calls",
            injected_approval_body(
                bindings,
                maximum_forward_evaluations=MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS - 1,
            ),
        )
        tampered_bindings = json.loads(json.dumps(bindings))
        tampered_bindings["smoke"]["file_sha256"] = "0" * 64
        expect_rejection("wrong_artifact_binding", injected_approval_body(tampered_bindings))

        approval_path = temporary / "positive-control-external-approval.json"
        authorized_path = temporary / "positive-control-authorized.json"
        write_json(approval_path, injected_approval_body(bindings))
        promote_authorization(
            unauthorized_target,
            expected_unauthorized_sha256,
            approval_path,
            file_sha256(approval_path),
            authorized_path,
        )
        validated, contract = validate_authorization(
            authorized_path,
            file_sha256(authorized_path),
            require_production=True,
        )
        positive_result = {
            "passed": bool(
                validated["schema_version"] == AUTHORIZED_SCHEMA
                and validated["status"] == "production_explicitly_authorized_by_external_approval"
                and validated["production_authorization"]["authorized"] is True
                and contract is not None
            ),
            "authorized_manifest_sha256": file_sha256(authorized_path),
            "approval_record_sha256": file_sha256(approval_path),
            "production_launched": False,
            "manifest_disposable": True,
        }
    temporary_artifacts_removed = not any(
        (ROOT / "work" / "gravity").glob("v6r3-authorization-control-*")
    )
    passed = bool(
        all(negative_results.values())
        and positive_result["passed"]
        and not positive_result["production_launched"]
        and temporary_artifacts_removed
    )
    return {
        "schema_version": AUTHORIZATION_CONTROLS_SCHEMA,
        "passed": passed,
        "negative_controls": negative_results,
        "positive_disposable_authorized_control": positive_result,
        "temporary_artifacts_removed": temporary_artifacts_removed,
        "production_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("controls", "smoke"):
        command = subparsers.add_parser(name)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--expected-contract-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)

    unauthorized_command = subparsers.add_parser("write-unauthorized")
    unauthorized_command.add_argument("--contract", type=Path, required=True)
    unauthorized_command.add_argument("--expected-contract-sha256", required=True)
    unauthorized_command.add_argument("--controls", type=Path, required=True)
    unauthorized_command.add_argument("--smoke", type=Path, required=True)
    unauthorized_command.add_argument("--output", type=Path, required=True)

    promote_command = subparsers.add_parser("promote-authorization")
    promote_command.add_argument("--unauthorized", type=Path, required=True)
    promote_command.add_argument("--expected-unauthorized-sha256", required=True)
    promote_command.add_argument("--external-approval", type=Path, required=True)
    promote_command.add_argument("--expected-external-approval-sha256", required=True)
    promote_command.add_argument("--output", type=Path, required=True)

    validate_command = subparsers.add_parser("validate-authorization")
    validate_command.add_argument("--authorization", type=Path, required=True)
    validate_command.add_argument("--expected-authorization-sha256", required=True)
    validate_command.add_argument("--require-production", action="store_true")

    transition_command = subparsers.add_parser("authorization-controls")
    transition_command.add_argument("--unauthorized", type=Path, required=True)
    transition_command.add_argument("--expected-unauthorized-sha256", required=True)
    transition_command.add_argument("--output", type=Path, required=True)

    run_command = subparsers.add_parser("run")
    run_command.add_argument("--authorization", type=Path, required=True)
    run_command.add_argument("--expected-authorization-sha256", required=True)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--execute-frozen-production-v6r3", action="store_true")

    args = parser.parse_args()
    if args.command in {"controls", "smoke"}:
        contract = load_contract(args.contract, args.expected_contract_sha256)
        if args.command == "controls":
            result = controls_receipt(contract)
            write_json(args.output, result)
            print(json.dumps(result, sort_keys=True))
            if not result["passed"]:
                raise SystemExit(2)
            return
        result = run_sampler(contract, contract["smoke_settings"], args.output, smoke=True)
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "write-unauthorized":
        result = write_unauthorized_manifest(
            args.contract,
            args.expected_contract_sha256,
            args.controls,
            args.smoke,
            args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "promote-authorization":
        result = promote_authorization(
            args.unauthorized,
            args.expected_unauthorized_sha256,
            args.external_approval,
            args.expected_external_approval_sha256,
            args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "validate-authorization":
        body, contract = validate_authorization(
            args.authorization,
            args.expected_authorization_sha256,
            require_production=args.require_production,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema_version": body["schema_version"],
                    "status": body["status"],
                    "production_authorized": body["production_authorization"]["authorized"],
                    "execution_contract_sha256": (
                        contract["_execution_contract_sha256"] if contract is not None else None
                    ),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "authorization-controls":
        result = authorization_transition_controls(
            args.unauthorized, args.expected_unauthorized_sha256
        )
        write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(2)
        return
    if not args.execute_frozen_production_v6r3:
        raise RuntimeError(
            "production requires the explicit --execute-frozen-production-v6r3 sentinel"
        )
    _authorization, contract = validate_authorization(
        args.authorization,
        args.expected_authorization_sha256,
        require_production=True,
    )
    if contract is None:
        raise RuntimeError("authorized contract was not loaded")
    result = run_sampler(contract, contract["production_settings"], args.output, smoke=False)
    print(json.dumps(result, sort_keys=True))
    if not result["production_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
