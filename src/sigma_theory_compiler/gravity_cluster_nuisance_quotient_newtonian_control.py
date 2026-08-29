from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc as sbc
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_newtonian_control_v1.json")
ARTIFACT_DIR = Path("runs/gravity/publication-readiness/nuisance-quotient-newtonian-control-v1")
PRODUCTION_RESULT_PATH = ARTIFACT_DIR / "matched-newtonian-control-production.npz"
IMPLEMENTATION_RECEIPT = Path(
    "runs/gravity/publication-readiness/"
    "nuisance-quotient-newtonian-control-implementation-v1-final.json"
)
CANDIDATE_CONFIG = Path("configs/gravity_cluster_nuisance_quotient_sampler_v1.json")
CANDIDATE_SOURCE = Path("src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sampler.py")
CANDIDATE_IMPLEMENTATION_RECEIPT = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sampler-implementation-v1.json"
)
SBC_RECEIPT_PATH = Path("runs/gravity/publication-readiness/nuisance-quotient-sbc-v1.json")

CONFIG_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-newtonian-control-config-1.0"
PREDICTOR_SCHEMA = "invariant-gravity-target-blind-newtonian-predictors-1.0"
RESULT_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-control-result-1.0"
CONTROLS_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-controls-1.0"
UNAUTHORIZED_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-authorization-1.0-unauthorized"
AUTHORIZED_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-authorization-1.0-authorized"
APPROVAL_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-external-approval-1.0"
AUTHORIZATION_CONTROLS_SCHEMA = (
    "invariant-gravity-nuisance-quotient-newtonian-authorization-controls-1.0"
)
SBC_GATE_CONTROLS_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-sbc-gate-controls-1.0"
IMPLEMENTATION_SCHEMA = "invariant-gravity-nuisance-quotient-newtonian-implementation-1.0"
SBC_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-receipt-1.0"

EXPECTED_CANDIDATE_CONFIG_SHA256 = (
    "c7222acad7071929fff39c23216e1cf2142b967e96c40434764a115a6cf3fc17"
)
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "975b9f69a614d7d419dcc44ac340c86b27a85e6e9fed7ed63d6f6caff228abcb"
)
EXPECTED_CANDIDATE_IMPLEMENTATION_SHA256 = (
    "d2ed231c2c0d99e2a29422633ad8f58a7f2617ca1113e8e70d6c7964ab80330d"
)
EXPECTED_SBC_CONFIG_SHA256 = "f9a9acf4ee4f1558ce97ab489423c7f43c09bb38af764c7d17de5109433b314c"
EXPECTED_SBC_SOURCE_SHA256 = "0eaad327544454c3540bfd99e87530caf3f0ace5e9d0065798125d98de863ba3"

TRUTH_UNIT = np.full(17, 0.5)
TARGET_SEED = 749_301
PREDICTOR_ROWS = 80
MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS = 1_575_104
FORBIDDEN_PREDICTOR_FIELDS = {
    "cluster",
    "object",
    "survey",
    "class",
    "target",
    "response",
    "outcome",
    "observed",
    "inferred_total_mass",
    "coefficient",
    "holdout",
    "confirmation",
    "independent",
    "split",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def confined(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return resolved


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


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


def predictor_body() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(PREDICTOR_ROWS):
        x = (index + 0.5) / PREDICTOR_ROWS
        basis: list[float] = []
        for column in range(10):
            harmonic = column // 2 + 1
            value = (
                math.sin(2.0 * math.pi * harmonic * x)
                if column % 2 == 0
                else math.cos(2.0 * math.pi * harmonic * x)
            )
            basis.append(float(value))
        rows.append(
            {
                "row_index": index,
                "dimensionless_radius_ratio": float(0.05 + 1.95 * x),
                "dimensionless_newtonian_baryonic_acceleration": float(math.exp(-1.25 + 2.0 * x)),
                "dimensionless_log_uncertainty": float(
                    0.22 + 0.03 * math.sin(2.0 * math.pi * x) ** 2
                ),
                "dimensionless_nuisance_basis": basis,
            }
        )
    body: dict[str, Any] = {
        "schema_version": PREDICTOR_SCHEMA,
        "status": "injected_target_blind_predictors_only_no_empirical_targets",
        "row_count": PREDICTOR_ROWS,
        "predictor_semantics": {
            "newtonian_source": "dimensionless_baryonic_acceleration_only",
            "nuisance_basis_coordinates": list(sampler.COMPOSITES),
            "target_fields_present": False,
            "real_object_or_survey_labels_present": False,
        },
        "rows": rows,
    }
    body["content_sha256"] = canonical_hash(body)
    return body


def build_predictor_packet(output: Path) -> dict[str, Any]:
    body = predictor_body()
    sampler.write_json(output, body)
    return body


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_PREDICTOR_FIELDS:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def load_predictor_packet(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("target-blind predictor packet missing or tampered")
    body = json.loads(target.read_text(encoding="utf-8"))
    if body != predictor_body():
        raise RuntimeError("target-blind predictor packet changed")
    if _contains_forbidden_key(body["rows"]):
        raise RuntimeError("predictor packet contains a forbidden target or object field")
    basis = np.asarray([row["dimensionless_nuisance_basis"] for row in body["rows"]], dtype=float)
    if basis.shape != (80, 10) or np.linalg.matrix_rank(basis) != 10:
        raise RuntimeError("predictor nuisance basis lost rank ten")
    return body


def _predictor_arrays(packet: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = packet["rows"]
    source = np.asarray(
        [row["dimensionless_newtonian_baryonic_acceleration"] for row in rows],
        dtype=float,
    )
    sigma = np.asarray([row["dimensionless_log_uncertainty"] for row in rows], dtype=float)
    basis = np.asarray([row["dimensionless_nuisance_basis"] for row in rows], dtype=float)
    return source, sigma, basis


def _composite_reference(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    reference = sampler.composite_values(TRUTH_UNIT, config)
    scale = np.maximum(np.abs(reference), 0.25)
    return np.asarray(reference, dtype=float), np.asarray(scale, dtype=float)


def forward_log_prediction(
    unit: np.ndarray, packet: dict[str, Any], config: dict[str, Any]
) -> np.ndarray:
    source, _sigma, basis = _predictor_arrays(packet)
    reference, scale = _composite_reference(config)
    composite = np.asarray(sampler.composite_values(unit, config), dtype=float)
    standardized = (composite - reference) / scale
    return np.log(source) + basis @ standardized


def generate_newtonian_log_targets(packet: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    _source, sigma, _basis = _predictor_arrays(packet)
    rng = np.random.default_rng(TARGET_SEED)
    return forward_log_prediction(TRUTH_UNIT, packet, config) + sigma * rng.normal(size=len(sigma))


class SyntheticNewtonianEvaluator:
    def __init__(self, packet: dict[str, Any], config: dict[str, Any]):
        self.packet = packet
        self.config = config
        self.targets = generate_newtonian_log_targets(packet, config)
        self.sigma = _predictor_arrays(packet)[1]
        self.calls = 0

    def __call__(self, unit: np.ndarray) -> float:
        self.calls += 1
        prediction = forward_log_prediction(unit, self.packet, self.config)
        residual = (self.targets - prediction) / self.sigma
        return float(-0.5 * np.sum(residual**2 + np.log(2.0 * math.pi * self.sigma**2)))


def identifiability_audit(packet: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    _source, _sigma, basis = _predictor_arrays(packet)
    reference, scale = _composite_reference(config)
    del reference
    epsilon = 1e-6
    composite_jacobian = np.empty((10, 17), dtype=float)
    for column in range(17):
        high = TRUTH_UNIT.copy()
        low = TRUTH_UNIT.copy()
        high[column] += epsilon
        low[column] -= epsilon
        composite_jacobian[:, column] = (
            sampler.composite_values(high, config) - sampler.composite_values(low, config)
        ) / (2.0 * epsilon)
    observable_jacobian = basis @ np.diag(1.0 / scale) @ composite_jacobian
    composite_rank = int(np.linalg.matrix_rank(basis))
    primitive_rank = int(np.linalg.matrix_rank(observable_jacobian))
    singular = np.linalg.svd(observable_jacobian, compute_uv=False)
    passed = composite_rank == 10 and primitive_rank == 10
    return {
        "passed": passed,
        "predictor_basis_rank": composite_rank,
        "primitive_to_observable_rank": primitive_rank,
        "maximum_observable_nuisance_dimension": 10,
        "primitive_null_dimensions": 17 - primitive_rank,
        "minimum_nonzero_singular_value": float(singular[primitive_rank - 1]),
        "primitive_labels_separately_identified": False,
        "claim_boundary": (
            "only_the_ten_quotient_coordinates_are_locally_identifiable_from_this_injected_design"
        ),
    }


def expected_call_accounting() -> dict[str, Any]:
    return sampler.expected_call_accounting()


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("Newtonian-control contract hash changed")
    body = json.loads(target.read_text(encoding="utf-8"))
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "production_result_path",
            "candidate_bindings",
            "source_bindings",
            "predictor_packet",
            "sobol_start_population",
            "exact_primitive_priors",
            "primitive_prior_semantics",
            "production_settings",
            "smoke_settings",
            "diagnostic_validation",
            "uniform_target_invariance_control",
            "completion_thresholds",
            "recovery_thresholds",
            "mechanics_thresholds",
            "orbit_validation",
            "call_accounting",
            "target_generator",
            "sbc_gate",
            "authorization_policy",
            "claim_boundary",
        },
        "Newtonian-control contract",
    )
    if body["schema_version"] != CONFIG_SCHEMA:
        raise RuntimeError("Newtonian-control schema changed")
    if body["status"] != "synthetic_controls_only_sbc_and_external_approval_required":
        raise RuntimeError("Newtonian-control status changed")
    source = confined(ROOT / body["implementation_source"])
    if source != Path(__file__).resolve():
        raise RuntimeError("contract points to another Newtonian-control executable")
    if normalized_sha256(source) != body["implementation_source_normalized_sha256"]:
        raise RuntimeError("Newtonian-control executable changed after freeze")
    if body["production_result_path"] != PRODUCTION_RESULT_PATH.as_posix():
        raise RuntimeError("Newtonian-control production result path changed")
    expected_candidate = {
        "config": {
            "path": CANDIDATE_CONFIG.as_posix(),
            "file_sha256": EXPECTED_CANDIDATE_CONFIG_SHA256,
        },
        "implementation_source": {
            "path": CANDIDATE_SOURCE.as_posix(),
            "file_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
        },
        "implementation_receipt": {
            "path": CANDIDATE_IMPLEMENTATION_RECEIPT.as_posix(),
            "file_sha256": EXPECTED_CANDIDATE_IMPLEMENTATION_SHA256,
        },
    }
    if body["candidate_bindings"] != expected_candidate:
        raise RuntimeError("candidate bindings changed")
    for name, binding in expected_candidate.items():
        validate_artifact_binding(binding, f"candidate_bindings.{name}")
    for name, binding in body["source_bindings"].items():
        validate_artifact_binding(binding, f"source_bindings.{name}")
    if body["exact_primitive_priors"] != sampler.PRIMITIVE_PRIORS:
        raise RuntimeError("matched 17-prior contract changed")
    uncertainty_config = uncertainty.load_config(ROOT)
    if uncertainty_config["continuous_priors"] != sampler.PRIMITIVE_PRIORS:
        raise RuntimeError("canonical uncertainty priors changed")
    if body["production_settings"] != sampler.PRODUCTION_SETTINGS:
        raise RuntimeError("production mechanics are not matched to candidate")
    if body["smoke_settings"] != sampler.SMOKE_SETTINGS:
        raise RuntimeError("smoke mechanics are not matched to candidate")
    expected_nested = {
        "diagnostic_validation": sampler.DIAGNOSTIC_VALIDATION,
        "uniform_target_invariance_control": sampler.UNIFORM_TARGET_CONTROL,
        "completion_thresholds": sampler.COMPLETION_THRESHOLDS,
        "recovery_thresholds": {
            "minimum_coordinates_with_truth_in_marginal_95_interval": 8,
            "maximum_absolute_posterior_median_z": 3.0,
            "single_injected_dataset_false_selection_rate_measured": False,
        },
        "mechanics_thresholds": sampler.MECHANICS_THRESHOLDS,
        "orbit_validation": sampler.ORBIT_VALIDATION,
        "call_accounting": expected_call_accounting(),
    }
    for name, expected in expected_nested.items():
        if body[name] != expected:
            raise RuntimeError(f"matched frozen object changed: {name}")
    exact_control_policy = {
        "target_generator": {
            "law": "newtonian_baryonic_acceleration_times_quotient_nuisance_pushforward",
            "truth_unit_value_for_all_17_primitives": 0.5,
            "noise_seed": TARGET_SEED,
            "targets_generated_only_after_predictor_packet_validation": True,
            "generated_targets_persisted_as_rows": False,
        },
        "sbc_gate": {
            "required_receipt_path": SBC_RECEIPT_PATH.as_posix(),
            "required_schema": SBC_SCHEMA,
            "required_status": "bounded_synthetic_sbc_passed_not_candidate_production",
            "candidate_production_result_must_pass": False,
            "bounded_synthetic_result_must_pass": True,
            "refusal_before_control_config_or_predictor_load": True,
        },
        "authorization_policy": {
            "external_approval_required_after_sbc": True,
            "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
            "production_authorized_by_default": False,
            "explicit_cli_sentinel_required": True,
            "all_generated_artifacts_atomic_no_clobber": True,
        },
        "claim_boundary": {
            "bounded_synthetic_mechanics_only": True,
            "real_data_claim_allowed": False,
            "candidate_false_selection_rate_measured": False,
            "newtonian_control_complete": False,
            "publication_readiness_changed": False,
        },
    }
    for name, expected in exact_control_policy.items():
        if body[name] != expected:
            raise RuntimeError(f"Newtonian frozen policy changed: {name}")
    predictor = load_predictor_packet(
        ROOT / body["predictor_packet"]["path"],
        body["predictor_packet"]["file_sha256"],
    )
    if predictor["content_sha256"] != body["predictor_packet"]["content_sha256"]:
        raise RuntimeError("target-blind predictor content seal changed")
    sampler.validate_sobol_starts(
        ROOT / body["sobol_start_population"]["path"],
        body["sobol_start_population"]["file_sha256"],
    )
    body["_execution_contract_sha256"] = expected_sha256
    return body


def _phase_counter() -> dict[str, int]:
    return {"attempted": 0, "out_of_bounds_rejected": 0, "evaluated": 0, "accepted": 0}


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def run_synthetic_sampler(
    contract: dict[str, Any],
    settings: dict[str, Any],
    output: Path,
    *,
    smoke: bool,
    sbc_receipt_binding: dict[str, str] | None = None,
    authorization_binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    if smoke and (sbc_receipt_binding is not None or authorization_binding is not None):
        raise RuntimeError("bounded smoke cannot claim to consume production gates")
    if not smoke:
        if sbc_receipt_binding is None or authorization_binding is None:
            raise RuntimeError("production must enter through both frozen gates")
        strict_keys(
            sbc_receipt_binding,
            {"path", "file_sha256"},
            "production SBC receipt binding",
        )
        validate_sbc_receipt(ROOT / sbc_receipt_binding["path"], sbc_receipt_binding["file_sha256"])
        strict_keys(
            authorization_binding,
            {"path", "file_sha256"},
            "production authorization binding",
        )
        validate_authorization(
            ROOT / authorization_binding["path"],
            authorization_binding["file_sha256"],
            require_production=True,
        )
        if confined(output) != ROOT / PRODUCTION_RESULT_PATH:
            raise RuntimeError("Newtonian production output path changed")
    packet = load_predictor_packet(
        ROOT / contract["predictor_packet"]["path"],
        contract["predictor_packet"]["file_sha256"],
    )
    config = uncertainty.load_config(ROOT)
    starts_archive = np.load(ROOT / contract["sobol_start_population"]["path"], allow_pickle=False)
    all_starts = np.asarray(starts_archive["particles"], dtype=float)
    replicates = int(settings["replicates"])
    particle_count = int(settings["particles"])
    particles_by_replicate = all_starts[:replicates, :particle_count].copy()
    evaluator = SyntheticNewtonianEvaluator(packet, config)

    diagnostic = sampler.diagnostic_validation_control(contract["diagnostic_validation"])
    uniform = sampler.prior_recovery_control(contract["uniform_target_invariance_control"], config)
    identification = identifiability_audit(packet, config)
    if not diagnostic["passed"] or not uniform["passed"] or not identification["passed"]:
        raise RuntimeError("bounded Newtonian controls failed before smoke")

    validation_rng = np.random.default_rng(int(settings["seed"]) + 900_000)
    orbit = sampler.validate_orbits(
        particles_by_replicate,
        evaluator,
        validation_rng,
        config,
        settings,
        int(settings["orbit_validation_cases_per_move_per_replicate"]),
    )
    orbit_passed = bool(
        orbit["maximum_absolute_training_log_likelihood_difference"]
        <= float(
            contract["orbit_validation"]["maximum_absolute_training_log_likelihood_difference"]
        )
        and orbit["maximum_absolute_composite_difference"]
        <= float(contract["orbit_validation"]["maximum_absolute_composite_difference"])
        and all(int(orbit["accepted_cases"][name]) > 0 for name in sampler.ORBIT_NAMES)
    )
    if not orbit_passed:
        raise RuntimeError("Newtonian-control orbit validation failed")

    before_initialization = evaluator.calls
    log_likelihood = np.empty((replicates, particle_count), dtype=float)
    for replicate in range(replicates):
        for particle in range(particle_count):
            log_likelihood[replicate, particle] = evaluator(
                particles_by_replicate[replicate, particle]
            )
    initialization_evaluations = evaluator.calls - before_initialization
    retained_count = int(settings["retained_sweeps"]) // int(settings["thin"])
    traces = np.empty(
        (replicates, particle_count, retained_count, len(sampler.COMPOSITES)), dtype=float
    )
    ending_particles = np.empty_like(particles_by_replicate)
    ending_log_likelihood = np.empty_like(log_likelihood)
    phase_evaluations = {phase: 0 for phase in ("adaptation", "settling", "retained")}
    replicate_summaries: list[dict[str, Any]] = []

    for replicate in range(replicates):
        particles = particles_by_replicate[replicate].copy()
        likelihood = log_likelihood[replicate].copy()
        rng = np.random.default_rng(int(settings["seed"]) + replicate)
        scale = float(settings["initial_active_scale"])
        square_root = sampler.covariance_square_root(particles[:, sampler.ACTIVE_INDICES])
        active_counts = {
            phase: _phase_counter() for phase in ("adaptation", "settling", "retained")
        }
        orbit_counts = {
            phase: {
                f"{move}_{suffix}": 0
                for move in sampler.ORBIT_NAMES
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
                    square_root = sampler.covariance_square_root(
                        particles[:, sampler.ACTIVE_INDICES]
                    )
                orbit_result = sampler.orbit_sweep(particles, rng, config, settings)
                _add_counts(orbit_counts[phase], orbit_result)
                calls_before = evaluator.calls
                transition = sampler.active_transition(
                    particles,
                    likelihood,
                    evaluator,
                    rng,
                    square_root,
                    scale,
                )
                _add_counts(active_counts[phase], transition)
                phase_evaluations[phase] += evaluator.calls - calls_before
                if phase == "adaptation":
                    rate = transition["accepted"] / transition["attempted"]
                    scale *= math.exp(
                        float(settings["adaptation_gain"])
                        * (rate - float(settings["target_acceptance"]))
                    )
                    scale = float(np.clip(scale, *map(float, settings["active_scale_bounds"])))
                elif phase == "retained" and (sweep + 1) % int(settings["thin"]) == 0:
                    traces[replicate, :, retained_index, :] = sampler.composite_values(
                        particles, config
                    )
                    retained_index += 1
            if phase == "adaptation":
                square_root = sampler.covariance_square_root(particles[:, sampler.ACTIVE_INDICES])
        ending_particles[replicate] = particles
        ending_log_likelihood[replicate] = likelihood
        replicate_summaries.append(
            {
                "replicate": replicate,
                "final_active_scale": scale,
                "active_counts": active_counts,
                "orbit_counts": orbit_counts,
            }
        )

    chains = traces.reshape(-1, retained_count, len(sampler.COMPOSITES))
    diagnostics = sampler.rank_split_diagnostics(chains)
    pooled = traces.reshape(-1, len(sampler.COMPOSITES))
    pooled_sd = np.std(pooled, axis=0, ddof=1)
    medians = np.median(traces, axis=(1, 2))
    spread = np.divide(
        np.ptp(medians, axis=0),
        pooled_sd,
        out=np.zeros_like(pooled_sd),
        where=pooled_sd > np.finfo(float).tiny,
    )
    completion = contract["completion_thresholds"]
    coordinate_pass = (
        diagnostics["valid"]
        & (diagnostics["rhat"] <= float(completion["maximum_rank_normalized_split_rhat"]))
        & (diagnostics["bulk_ess"] >= float(completion["minimum_bulk_effective_samples"]))
        & (diagnostics["tail_ess"] >= float(completion["minimum_tail_effective_samples"]))
        & (spread <= float(completion["maximum_standardized_between_replicate_median_spread"]))
    )
    mechanics_rows: list[dict[str, Any]] = []
    mechanics_pass = True
    mechanics = contract["mechanics_thresholds"]
    for row in replicate_summaries:
        active = row["active_counts"]["retained"]
        active_rate = active["accepted"] / active["attempted"]
        orbit_rates = {
            move: row["orbit_counts"]["retained"][f"{move}_accepted"]
            / row["orbit_counts"]["retained"][f"{move}_attempted"]
            for move in sampler.ORBIT_NAMES
        }
        passed = bool(
            float(mechanics["minimum_retained_active_acceptance"])
            <= active_rate
            <= float(mechanics["maximum_retained_active_acceptance"])
            and all(
                rate >= float(mechanics["minimum_retained_orbit_acceptance"])
                for rate in orbit_rates.values()
            )
        )
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
        "orbit_validation_evaluations": int(orbit["evaluations"]),
        "initialization_evaluations": int(initialization_evaluations),
        "adaptation_proposal_evaluations": int(phase_evaluations["adaptation"]),
        "settling_proposal_evaluations": int(phase_evaluations["settling"]),
        "retained_proposal_evaluations": int(phase_evaluations["retained"]),
    }
    accounted = sum(accounting.values())
    if accounted != evaluator.calls:
        raise RuntimeError("Newtonian forward-call accounting mismatch")
    maximum_calls = sampler.maximum_forward_calls(settings)
    if accounted > maximum_calls:
        raise RuntimeError("Newtonian forward calls exceeded frozen maximum")
    accounting["total_forward_evaluations"] = accounted
    accounting["frozen_maximum_forward_evaluations"] = maximum_calls
    target_hash = hashlib.sha256(evaluator.targets.astype("<f8").tobytes()).hexdigest()
    truth = np.asarray(sampler.composite_values(TRUTH_UNIT, config), dtype=float)
    lower = np.quantile(pooled, 0.025, axis=0)
    median = np.quantile(pooled, 0.5, axis=0)
    upper = np.quantile(pooled, 0.975, axis=0)
    truth_inside = (lower <= truth) & (truth <= upper)
    median_z = np.divide(
        median - truth,
        pooled_sd,
        out=np.full_like(pooled_sd, np.inf),
        where=pooled_sd > np.finfo(float).tiny,
    )
    recovery_thresholds = contract["recovery_thresholds"]
    recovery_passed = bool(
        int(np.count_nonzero(truth_inside))
        >= int(recovery_thresholds["minimum_coordinates_with_truth_in_marginal_95_interval"])
        and float(np.max(np.abs(median_z)))
        <= float(recovery_thresholds["maximum_absolute_posterior_median_z"])
    )
    production_passed = bool(
        not smoke
        and orbit_passed
        and mechanics_pass
        and np.all(coordinate_pass)
        and recovery_passed
    )
    summary: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "mode": (
            "bounded_injected_newtonian_smoke" if smoke else "matched_newtonian_synthetic_control"
        ),
        "decision": (
            "INJECTED_MECHANICS_SMOKE_ONLY_NOT_SBC_OR_NEWTONIAN_ADJUDICATION"
            if smoke
            else (
                "MATCHED_NEWTONIAN_SYNTHETIC_RECOVERY_CONTROL_PASSED"
                if production_passed
                else "MATCHED_NEWTONIAN_SYNTHETIC_RECOVERY_CONTROL_FAILED_RETAIN_RESULT"
            )
        ),
        "execution_contract_sha256": contract["_execution_contract_sha256"],
        "matched_candidate_mechanics": True,
        "target_generation": {
            "law": "newtonian_baryonic_acceleration_times_quotient_nuisance_pushforward",
            "target_seed": TARGET_SEED,
            "generated_target_sha256": target_hash,
            "targets_persisted_as_rows": False,
            "predictors_target_blind": True,
        },
        "data_boundary": {
            "injected_predictor_rows": PREDICTOR_ROWS,
            "real_development_rows": 0,
            "real_holdout_rows": 0,
            "real_confirmation_rows": 0,
            "real_independent_rows": 0,
            "sbc_gate_bypassed_only_for_bounded_mechanics_smoke": smoke,
        },
        "identifiability": identification,
        "controls": {
            "diagnostic_validation": diagnostic,
            "uniform_target_invariance": uniform,
            "orbit_validation": orbit,
        },
        "orbit_validation_passed": orbit_passed,
        "forward_call_accounting": accounting,
        "all_coordinates_positive_variance": bool(np.all(diagnostics["valid"])),
        "all_coordinate_gates_passed": bool(np.all(coordinate_pass)),
        "all_mechanics_gates_passed": bool(mechanics_pass),
        "synthetic_truth_recovery": {
            "coordinates": list(sampler.COMPOSITES),
            "truth": truth.tolist(),
            "marginal_95_interval_lower": lower.tolist(),
            "posterior_median": median.tolist(),
            "marginal_95_interval_upper": upper.tolist(),
            "truth_inside_marginal_95_interval": truth_inside.tolist(),
            "coordinates_with_truth_inside_marginal_95_interval": int(
                np.count_nonzero(truth_inside)
            ),
            "posterior_median_z": median_z.tolist(),
            "thresholds": recovery_thresholds,
            "passed": recovery_passed,
            "single_dataset_false_selection_rate_measured": False,
        },
        "production_passed": production_passed,
        "sbc_receipt_consumed": sbc_receipt_binding,
        "authorization_consumed": authorization_binding,
        "candidate_production_result_consumed": False,
        "mechanics": mechanics_rows,
        "claim_boundary": {
            "real_data_claim": False,
            "candidate_false_selection_claim": False,
            "newtonian_control_complete": production_passed,
            "single_injected_dataset_establishes_false_selection_rate": False,
            "primitive_labels_separately_identified": False,
            "publication_readiness_changed": False,
        },
    }
    sampler.atomic_save_result(
        output,
        traces=traces,
        ending_particles=ending_particles,
        ending_log_likelihood=ending_log_likelihood,
        summary=summary,
    )
    return summary


def controls_receipt(contract: dict[str, Any]) -> dict[str, Any]:
    packet = load_predictor_packet(
        ROOT / contract["predictor_packet"]["path"],
        contract["predictor_packet"]["file_sha256"],
    )
    config = uncertainty.load_config(ROOT)
    diagnostic = sampler.diagnostic_validation_control(contract["diagnostic_validation"])
    uniform = sampler.prior_recovery_control(contract["uniform_target_invariance_control"], config)
    identification = identifiability_audit(packet, config)
    return {
        "schema_version": CONTROLS_SCHEMA,
        "execution_contract_sha256": contract["_execution_contract_sha256"],
        "matched_candidate_mechanics": True,
        "diagnostic_validation": diagnostic,
        "uniform_target_invariance": uniform,
        "identifiability": identification,
        "real_target_rows_read": 0,
        "model_or_paid_calls": 0,
        "forward_evaluations": 0,
        "passed": bool(diagnostic["passed"] and uniform["passed"] and identification["passed"]),
    }


def validate_bound_controls_and_smoke(
    config_hash: str, controls_path: Path, smoke_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = json.loads(confined(controls_path).read_text(encoding="utf-8"))
    with np.load(confined(smoke_path), allow_pickle=False) as archive:
        if set(archive.files) != {
            "composite_traces",
            "ending_particles",
            "ending_log_likelihood",
            "summary",
        }:
            raise RuntimeError("Newtonian smoke archive keys changed")
        smoke = json.loads(str(archive["summary"].item()))
    if (
        controls.get("schema_version") != CONTROLS_SCHEMA
        or controls.get("execution_contract_sha256") != config_hash
        or controls.get("passed") is not True
        or controls.get("real_target_rows_read") != 0
        or smoke.get("schema_version") != RESULT_SCHEMA
        or smoke.get("execution_contract_sha256") != config_hash
        or smoke.get("mode") != "bounded_injected_newtonian_smoke"
        or smoke.get("production_passed") is not False
        or smoke.get("data_boundary", {}).get("real_development_rows") != 0
    ):
        raise RuntimeError("bounded Newtonian evidence changed")
    return controls, smoke


def frozen_artifact_bindings(
    contract: dict[str, Any], config_path: Path, controls_path: Path, smoke_path: Path
) -> dict[str, Any]:
    predictor = contract["predictor_packet"]
    return {
        "contract": artifact_binding(config_path),
        "implementation_source": artifact_binding(ROOT / contract["implementation_source"]),
        "predictor_packet": {
            "path": predictor["path"],
            "file_sha256": predictor["file_sha256"],
        },
        "sobol_start_population": contract["sobol_start_population"],
        "controls": artifact_binding(controls_path),
        "smoke": artifact_binding(smoke_path),
    }


def write_unauthorized_manifest(
    config_path: Path,
    expected_config_sha256: str,
    controls_path: Path,
    smoke_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract(config_path, expected_config_sha256)
    validate_bound_controls_and_smoke(expected_config_sha256, controls_path, smoke_path)
    body = {
        "schema_version": UNAUTHORIZED_SCHEMA,
        "status": "sbc_and_external_approval_required",
        "artifact_bindings": frozen_artifact_bindings(
            contract, config_path, controls_path, smoke_path
        ),
        "production_authorization": {
            "authorized": False,
            "approved_by": None,
            "approval_id": None,
            "maximum_forward_evaluations": 0,
        },
        "claim_boundary": {
            "candidate_sbc_pass_required": True,
            "external_approval_required": True,
            "production_execution_allowed": False,
            "production_executed": False,
            "newtonian_control_complete": False,
        },
    }
    sampler.write_json(output, body)
    return body


def validate_unauthorized_body(body: dict[str, Any]) -> None:
    if (
        body.get("schema_version") != UNAUTHORIZED_SCHEMA
        or body.get("status") != "sbc_and_external_approval_required"
        or body.get("production_authorization")
        != {
            "authorized": False,
            "approved_by": None,
            "approval_id": None,
            "maximum_forward_evaluations": 0,
        }
        or body.get("claim_boundary", {}).get("production_execution_allowed") is not False
    ):
        raise RuntimeError("unauthorized Newtonian manifest changed")
    for name, row in body["artifact_bindings"].items():
        validate_artifact_binding(row, f"authorization.{name}")


def validate_external_approval(
    path: Path, expected_sha256: str, bindings: dict[str, Any]
) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("Newtonian external approval missing or tampered")
    body = json.loads(target.read_text(encoding="utf-8"))
    if (
        body.get("schema_version") != APPROVAL_SCHEMA
        or body.get("status") != "explicit_external_production_approval"
        or body.get("approved_by") != "Henry"
        or not isinstance(body.get("approval_id"), str)
        or not body["approval_id"].strip()
        or body.get("maximum_forward_evaluations") != MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS
        or body.get("artifact_bindings") != bindings
    ):
        raise RuntimeError("Newtonian external approval is incomplete")
    return body


def promote_authorization(
    unauthorized_path: Path,
    expected_unauthorized_sha256: str,
    approval_path: Path,
    expected_approval_sha256: str,
    output: Path,
) -> dict[str, Any]:
    target = confined(unauthorized_path)
    if file_sha256(target) != expected_unauthorized_sha256:
        raise RuntimeError("unauthorized Newtonian manifest hash changed")
    unauthorized = json.loads(target.read_text(encoding="utf-8"))
    validate_unauthorized_body(unauthorized)
    approval = validate_external_approval(
        approval_path, expected_approval_sha256, unauthorized["artifact_bindings"]
    )
    body = {
        "schema_version": AUTHORIZED_SCHEMA,
        "status": "externally_authorized_but_still_sbc_gated",
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
        "claim_boundary": {
            "candidate_sbc_pass_required": True,
            "external_approval_satisfied": True,
            "production_execution_allowed_only_after_sbc": True,
            "production_executed": False,
            "newtonian_control_complete": False,
        },
    }
    sampler.write_json(output, body)
    return body


def validate_authorization(
    path: Path, expected_sha256: str, *, require_production: bool
) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("Newtonian authorization hash changed")
    body = json.loads(target.read_text(encoding="utf-8"))
    if body.get("schema_version") == UNAUTHORIZED_SCHEMA:
        if require_production:
            raise RuntimeError(
                "Newtonian production is externally unauthorized; refusal before config or predictors"
            )
        validate_unauthorized_body(body)
        return body
    if body.get("schema_version") != AUTHORIZED_SCHEMA:
        raise RuntimeError("Newtonian authorization schema is unrecognized")
    if (
        body.get("status") != "externally_authorized_but_still_sbc_gated"
        or body.get("production_authorization", {}).get("authorized") is not True
        or body["production_authorization"].get("approved_by") != "Henry"
        or body["production_authorization"].get("maximum_forward_evaluations")
        != MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS
    ):
        raise RuntimeError("authorized Newtonian manifest changed")
    for name, row in body["artifact_bindings"].items():
        validate_artifact_binding(row, f"authorization.{name}")
    approval = body["external_approval_binding"]
    validate_external_approval(
        ROOT / approval["path"], approval["file_sha256"], body["artifact_bindings"]
    )
    return body


def _injected_approval(bindings: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    body = {
        "schema_version": APPROVAL_SCHEMA,
        "status": "explicit_external_production_approval",
        "approved_by": "Henry",
        "approval_id": "NEWTONIAN-CONTROL-AUTH-TRANSITION-CONTROL-ONLY",
        "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
        "artifact_bindings": bindings,
    }
    body.update(overrides)
    return body


def authorization_transition_controls(
    unauthorized_path: Path, expected_unauthorized_sha256: str
) -> dict[str, Any]:
    target = confined(unauthorized_path)
    body = json.loads(target.read_text(encoding="utf-8"))
    if file_sha256(target) != expected_unauthorized_sha256:
        raise RuntimeError("authorization-control input hash changed")
    validate_unauthorized_body(body)
    negative: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(
        prefix="newtonian-auth-control-", dir=ROOT / ARTIFACT_DIR
    ) as td:
        directory = Path(td)

        def reject(name: str, approval: dict[str, Any]) -> None:
            approval_path = directory / f"{name}-approval.json"
            output_path = directory / f"{name}-authorized.json"
            sampler.write_json(approval_path, approval)
            try:
                promote_authorization(
                    target,
                    expected_unauthorized_sha256,
                    approval_path,
                    file_sha256(approval_path),
                    output_path,
                )
            except RuntimeError:
                negative[name] = not output_path.exists()
                return
            negative[name] = False

        bindings = body["artifact_bindings"]
        reject("wrong_approver", _injected_approval(bindings, approved_by="not-Henry"))
        reject("empty_id", _injected_approval(bindings, approval_id=""))
        reject(
            "wrong_calls",
            _injected_approval(
                bindings,
                maximum_forward_evaluations=MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS - 1,
            ),
        )
        approval_path = directory / "positive-approval.json"
        authorized_path = directory / "positive-authorized.json"
        sampler.write_json(approval_path, _injected_approval(bindings))
        promote_authorization(
            target,
            expected_unauthorized_sha256,
            approval_path,
            file_sha256(approval_path),
            authorized_path,
        )
        positive = validate_authorization(
            authorized_path, file_sha256(authorized_path), require_production=True
        )
        positive_passed = positive["production_authorization"]["authorized"] is True
    result = {
        "schema_version": AUTHORIZATION_CONTROLS_SCHEMA,
        "passed": bool(all(negative.values()) and positive_passed),
        "negative_controls": negative,
        "positive_disposable_authorized_control": {
            "passed": positive_passed,
            "production_launched": False,
            "still_requires_candidate_sbc": True,
            "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
        },
        "production_runs": 0,
        "temporary_artifacts_removed": True,
    }
    return result


def validate_sbc_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("passing candidate SBC receipt is absent or tampered")
    body = json.loads(target.read_text(encoding="utf-8"))
    unhashed = dict(body)
    observed_content_hash = unhashed.pop("content_sha256", None)
    if (
        body.get("schema_version") != SBC_SCHEMA
        or observed_content_hash != sbc.content_sha256(unhashed)
        or body.get("status") != "bounded_synthetic_sbc_passed_not_candidate_production"
        or body.get("decision") != "BOUNDED_SYNTHETIC_QUOTIENT_SBC_PASSED"
        or body.get("evidence", {}).get("config")
        != {
            "path": sbc.CONFIG_PATH.as_posix(),
            "file_sha256": EXPECTED_SBC_CONFIG_SHA256,
        }
        or body.get("evidence", {}).get("implementation_source")
        != {
            "path": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sbc.py",
            "file_sha256": EXPECTED_SBC_SOURCE_SHA256,
        }
        or body.get("evidence", {}).get("canonical_sampler")
        != {
            "path": CANDIDATE_SOURCE.as_posix(),
            "file_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
        }
        or body.get("data_boundary") != sbc.DATA_BOUNDARY
        or body.get("claim_boundary") != sbc.CLAIM_BOUNDARY
        or body.get("controls", {}).get("orbit_invariance", {}).get("passed") is not True
        or body.get("controls", {}).get("importance_reference_present") is not True
        or body.get("controls", {}).get("randomized_stellar_clipping_tie_ranks") is not True
        or not body.get("scenario_results")
        or not all(row.get("passed") is True for row in body["scenario_results"])
    ):
        raise RuntimeError("candidate SBC receipt did not pass frozen gates")
    checked = sbc.check(ROOT / sbc.CONFIG_PATH, EXPECTED_SBC_CONFIG_SHA256, target)
    if checked.get("passed") is not True:
        raise RuntimeError("canonical SBC checker did not pass the receipt")
    result_path = validate_artifact_binding(
        body["evidence"]["bounded_result"], "SBC bounded synthetic result"
    )
    with np.load(result_path, allow_pickle=False) as archive:
        summary = json.loads(str(archive["summary"].item()))
    if (
        summary.get("schema_version") != sbc.RESULT_SCHEMA
        or summary.get("passed") is not True
        or summary.get("decision") != "BOUNDED_SYNTHETIC_QUOTIENT_SBC_PASSED"
        or summary.get("config_sha256") != EXPECTED_SBC_CONFIG_SHA256
        or summary.get("data_boundary") != sbc.DATA_BOUNDARY
        or summary.get("claim_boundary") != sbc.CLAIM_BOUNDARY
        or summary.get("call_accounting") != body.get("counts")
    ):
        raise RuntimeError("SBC receipt does not bind a passing bounded synthetic result")
    return body


def sbc_gate_controls() -> dict[str, Any]:
    receipt_path = ROOT / SBC_RECEIPT_PATH
    present = receipt_path.is_file()
    rejected = False
    observed_status = None
    observed_sha256 = None
    if present:
        observed_sha256 = file_sha256(receipt_path)
        observed_status = json.loads(receipt_path.read_text(encoding="utf-8")).get("status")
        try:
            validate_sbc_receipt(receipt_path, observed_sha256)
        except RuntimeError:
            rejected = True
    else:
        try:
            validate_sbc_receipt(receipt_path, "0" * 64)
        except RuntimeError:
            rejected = True
    return {
        "schema_version": SBC_GATE_CONTROLS_SCHEMA,
        "passed": rejected,
        "required_sbc_receipt_path": SBC_RECEIPT_PATH.as_posix(),
        "required_sbc_schema": SBC_SCHEMA,
        "required_passing_status": "bounded_synthetic_sbc_passed_not_candidate_production",
        "sbc_receipt_currently_present": present,
        "observed_sbc_receipt_sha256": observed_sha256,
        "observed_sbc_status": observed_status,
        "nonpassing_or_missing_sbc_receipt_rejected": rejected,
        "predictor_rows_read_during_rejection": 0,
        "forward_evaluations_during_rejection": 0,
        "production_runs": 0,
        "newtonian_control_unlocked": False,
    }


def execute_production(
    sbc_path: Path,
    expected_sbc_sha256: str,
    authorization_path: Path,
    expected_authorization_sha256: str,
    config_path: Path,
    expected_config_sha256: str,
    output: Path,
) -> dict[str, Any]:
    # Gate ordering is deliberate: no control config or predictors before a passing SBC.
    validate_sbc_receipt(sbc_path, expected_sbc_sha256)
    validate_authorization(
        authorization_path, expected_authorization_sha256, require_production=True
    )
    if confined(output) != ROOT / PRODUCTION_RESULT_PATH:
        raise RuntimeError("Newtonian production output path changed")
    contract = load_contract(config_path, expected_config_sha256)
    return run_synthetic_sampler(
        contract,
        contract["production_settings"],
        output,
        smoke=False,
        sbc_receipt_binding={
            "path": confined(sbc_path).relative_to(ROOT).as_posix(),
            "file_sha256": expected_sbc_sha256,
        },
        authorization_binding={
            "path": confined(authorization_path).relative_to(ROOT).as_posix(),
            "file_sha256": expected_authorization_sha256,
        },
    )


def implementation_receipt(
    config_path: Path,
    expected_config_sha256: str,
    controls_path: Path,
    smoke_path: Path,
    race_path: Path,
    authorization_controls_path: Path,
    unauthorized_path: Path,
    sbc_controls_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract(config_path, expected_config_sha256)
    controls, smoke = validate_bound_controls_and_smoke(
        expected_config_sha256, controls_path, smoke_path
    )
    race = json.loads(confined(race_path).read_text(encoding="utf-8"))
    auth_controls = json.loads(confined(authorization_controls_path).read_text(encoding="utf-8"))
    sbc_controls = json.loads(confined(sbc_controls_path).read_text(encoding="utf-8"))
    authorization = json.loads(confined(unauthorized_path).read_text(encoding="utf-8"))
    validate_unauthorized_body(authorization)
    if (
        race.get("passed") is not True
        or auth_controls.get("passed") is not True
        or sbc_controls.get("passed") is not True
        or sbc_controls.get("newtonian_control_unlocked") is not False
        or sbc_controls != sbc_gate_controls()
        or (ROOT / PRODUCTION_RESULT_PATH).exists()
    ):
        raise RuntimeError("Newtonian implementation controls are not all passing")
    body: dict[str, Any] = {
        "schema_version": IMPLEMENTATION_SCHEMA,
        "status": "bounded_synthetic_controls_pass_sbc_and_external_approval_required",
        "decision": "NEWTONIAN_CONTROL_PATH_IMPLEMENTED_BUT_LOCKED_BEHIND_CANDIDATE_SBC",
        "evidence": {
            "config": artifact_binding(config_path),
            "implementation_source": artifact_binding(ROOT / contract["implementation_source"]),
            "predictor_packet": artifact_binding(ROOT / contract["predictor_packet"]["path"]),
            "controls": artifact_binding(controls_path),
            "smoke": artifact_binding(smoke_path),
            "atomic_no_clobber_controls": artifact_binding(race_path),
            "authorization_controls": artifact_binding(authorization_controls_path),
            "authorization_current_unauthorized": artifact_binding(unauthorized_path),
            "sbc_gate_controls": artifact_binding(sbc_controls_path),
        },
        "matching": {
            "candidate_priors_exact": contract["exact_primitive_priors"]
            == sampler.PRIMITIVE_PRIORS,
            "candidate_production_mechanics_exact": contract["production_settings"]
            == sampler.PRODUCTION_SETTINGS,
            "candidate_smoke_mechanics_exact": contract["smoke_settings"] == sampler.SMOKE_SETTINGS,
            "candidate_call_accounting_exact": contract["call_accounting"]
            == sampler.expected_call_accounting(),
        },
        "identifiability": controls["identifiability"],
        "data_and_compute": {
            "target_blind_injected_predictor_rows": 80,
            "real_development_rows": 0,
            "real_holdout_rows": 0,
            "real_confirmation_rows": 0,
            "real_independent_rows": 0,
            "bounded_smoke_forward_evaluations": smoke["forward_call_accounting"][
                "total_forward_evaluations"
            ],
            "production_forward_evaluations": 0,
            "paid_or_model_calls": 0,
            "network_calls": 0,
        },
        "gates": {
            "candidate_sbc_receipt_present": sbc_controls.get("sbc_receipt_currently_present"),
            "candidate_sbc_status": sbc_controls.get("observed_sbc_status"),
            "candidate_sbc_passed": False,
            "external_production_authorized": False,
            "newtonian_control_unlocked": False,
            "production_result_present": False,
            "production_result_path": PRODUCTION_RESULT_PATH.as_posix(),
        },
        "claim_boundary": {
            "bounded_mechanics_infrastructure_works": True,
            "candidate_false_selection_rate_measured": False,
            "single_injected_dataset_false_selection_rate_measured": False,
            "newtonian_null_rejected": False,
            "only_ten_quotient_coordinates_identifiable": True,
            "seventeen_primitive_labels_identifiable": False,
            "candidate_validated": False,
            "CP5_tasks_completed": [],
            "publication_readiness_changed": False,
        },
    }
    body["content_sha256"] = canonical_hash(body)
    sampler.write_json(output, body)
    return body


def check_implementation(
    config_path: Path, expected_config_sha256: str, receipt_path: Path
) -> dict[str, Any]:
    load_contract(config_path, expected_config_sha256)
    receipt = json.loads(confined(receipt_path).read_text(encoding="utf-8"))
    unhashed = dict(receipt)
    observed = unhashed.pop("content_sha256")
    if observed != canonical_hash(unhashed):
        raise RuntimeError("Newtonian implementation receipt content changed")
    if (
        receipt.get("schema_version") != IMPLEMENTATION_SCHEMA
        or receipt.get("gates", {}).get("newtonian_control_unlocked") is not False
        or receipt.get("data_and_compute", {}).get("production_forward_evaluations") != 0
        or receipt.get("claim_boundary", {}).get("publication_readiness_changed") is not False
    ):
        raise RuntimeError("Newtonian implementation claim boundary changed")
    for name, row in receipt["evidence"].items():
        validate_artifact_binding(row, f"implementation.evidence.{name}")
    return {
        "valid": True,
        "config_sha256": expected_config_sha256,
        "implementation_receipt_sha256": file_sha256(confined(receipt_path)),
        "sbc_passed": False,
        "production_authorized": False,
        "production_runs": 0,
        "newtonian_control_unlocked": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    predictors = commands.add_parser("build-predictors")
    predictors.add_argument("--output", type=Path, required=True)
    for name in ("controls", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--expected-contract-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    race = commands.add_parser("write-race-controls")
    race.add_argument("--output", type=Path, required=True)
    unauthorized = commands.add_parser("write-unauthorized")
    unauthorized.add_argument("--contract", type=Path, required=True)
    unauthorized.add_argument("--expected-contract-sha256", required=True)
    unauthorized.add_argument("--controls", type=Path, required=True)
    unauthorized.add_argument("--smoke", type=Path, required=True)
    unauthorized.add_argument("--output", type=Path, required=True)
    auth_controls = commands.add_parser("authorization-controls")
    auth_controls.add_argument("--unauthorized", type=Path, required=True)
    auth_controls.add_argument("--expected-unauthorized-sha256", required=True)
    auth_controls.add_argument("--output", type=Path, required=True)
    promote = commands.add_parser("promote-authorization")
    promote.add_argument("--unauthorized", type=Path, required=True)
    promote.add_argument("--expected-unauthorized-sha256", required=True)
    promote.add_argument("--external-approval", type=Path, required=True)
    promote.add_argument("--expected-external-approval-sha256", required=True)
    promote.add_argument("--output", type=Path, required=True)
    sbc_controls = commands.add_parser("sbc-gate-controls")
    sbc_controls.add_argument("--output", type=Path, required=True)
    receipt = commands.add_parser("write-implementation-receipt")
    receipt.add_argument("--contract", type=Path, required=True)
    receipt.add_argument("--expected-contract-sha256", required=True)
    receipt.add_argument("--controls", type=Path, required=True)
    receipt.add_argument("--smoke", type=Path, required=True)
    receipt.add_argument("--race-controls", type=Path, required=True)
    receipt.add_argument("--authorization-controls", type=Path, required=True)
    receipt.add_argument("--unauthorized", type=Path, required=True)
    receipt.add_argument("--sbc-controls", type=Path, required=True)
    receipt.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--expected-config-sha256", required=True)
    check.add_argument("--implementation-receipt", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--sbc-receipt", type=Path, required=True)
    run.add_argument("--expected-sbc-sha256", required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--expected-authorization-sha256", required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--expected-config-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute-frozen-newtonian-control", action="store_true")
    args = parser.parse_args()

    if args.command == "build-predictors":
        result = build_predictor_packet(args.output)
    elif args.command in {"controls", "smoke"}:
        contract = load_contract(args.contract, args.expected_contract_sha256)
        if args.command == "controls":
            result = controls_receipt(contract)
            sampler.write_json(args.output, result)
        else:
            result = run_synthetic_sampler(
                contract, contract["smoke_settings"], args.output, smoke=True
            )
    elif args.command == "write-race-controls":
        result = sampler.write_race_controls()
        sampler.write_json(args.output, result)
    elif args.command == "write-unauthorized":
        result = write_unauthorized_manifest(
            args.contract,
            args.expected_contract_sha256,
            args.controls,
            args.smoke,
            args.output,
        )
    elif args.command == "authorization-controls":
        result = authorization_transition_controls(
            args.unauthorized, args.expected_unauthorized_sha256
        )
        sampler.write_json(args.output, result)
    elif args.command == "promote-authorization":
        result = promote_authorization(
            args.unauthorized,
            args.expected_unauthorized_sha256,
            args.external_approval,
            args.expected_external_approval_sha256,
            args.output,
        )
    elif args.command == "sbc-gate-controls":
        result = sbc_gate_controls()
        sampler.write_json(args.output, result)
    elif args.command == "write-implementation-receipt":
        result = implementation_receipt(
            args.contract,
            args.expected_contract_sha256,
            args.controls,
            args.smoke,
            args.race_controls,
            args.authorization_controls,
            args.unauthorized,
            args.sbc_controls,
            args.output,
        )
    elif args.command == "check":
        result = check_implementation(
            args.config, args.expected_config_sha256, args.implementation_receipt
        )
    else:
        if not args.execute_frozen_newtonian_control:
            raise RuntimeError("explicit Newtonian-control production sentinel required")
        result = execute_production(
            args.sbc_receipt,
            args.expected_sbc_sha256,
            args.authorization,
            args.expected_authorization_sha256,
            args.config,
            args.expected_config_sha256,
            args.output,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
