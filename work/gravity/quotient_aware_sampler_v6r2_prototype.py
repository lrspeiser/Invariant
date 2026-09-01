from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import (
    gravity_item59_xcop_forward_observable_gate as item59,
)


def _load_v6_kernel() -> Any:
    path = Path(__file__).with_name("quotient_aware_sampler_v6_prototype.py")
    spec = importlib.util.spec_from_file_location("invariant_v6r1_bound_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the hash-bound V6R1 kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v6 = _load_v6_kernel()
_LEGACY_PRIOR_CONTROL = v6.prior_recovery_control
SCHEMA = "invariant-gravity-cluster-quotient-sampler-contract-6.2"
PACKET_SCHEMA = "invariant-gravity-development-train-packet-6.2"
AUTHORIZATION_SCHEMA = "invariant-gravity-quotient-sampler-authorization-6.2"
COMPOSITES = v6.COMPOSITES
ACTIVE_INDICES = v6.ACTIVE_INDICES
ORBIT_NAMES = v6.ORBIT_NAMES

PRIMITIVE_PRIORS = [
    {
        "parameter": "outer_nonthermal_fraction",
        "cause": "nonthermal_pressure",
        "distribution": "uniform",
        "low": 0.0,
        "high": 0.5,
    },
    {
        "parameter": "nonthermal_radial_power",
        "cause": "nonthermal_pressure",
        "distribution": "uniform",
        "low": 0.5,
        "high": 2.0,
    },
    {
        "parameter": "xray_temperature_cross_calibration",
        "cause": "cross_calibration",
        "distribution": "uniform",
        "low": 0.85,
        "high": 1.15,
    },
    {
        "parameter": "outer_pressure_boundary_sigma",
        "cause": "boundary_condition",
        "distribution": "uniform",
        "low": -2.0,
        "high": 2.0,
    },
    {
        "parameter": "density_error_sigma",
        "cause": "density_measurement",
        "distribution": "uniform",
        "low": -1.0,
        "high": 1.0,
    },
    {
        "parameter": "bcg_mass_scale",
        "cause": "BCG_stellar_mass",
        "distribution": "uniform",
        "low": 0.75,
        "high": 1.25,
    },
    {
        "parameter": "satellite_mass_scale",
        "cause": "satellite_stellar_mass",
        "distribution": "uniform",
        "low": 0.75,
        "high": 1.25,
    },
    {
        "parameter": "missing_member_fraction",
        "cause": "missing_members",
        "distribution": "uniform",
        "low": 0.0,
        "high": 0.2,
    },
    {
        "parameter": "intracluster_light_fraction",
        "cause": "intracluster_light",
        "distribution": "uniform",
        "low": 0.0,
        "high": 0.3,
    },
    {
        "parameter": "imf_mass_scale",
        "cause": "IMF",
        "distribution": "uniform",
        "low": 0.7,
        "high": 1.3,
    },
    {
        "parameter": "mass_to_light_scale",
        "cause": "stellar_mass_to_light",
        "distribution": "uniform",
        "low": 0.8,
        "high": 1.2,
    },
    {
        "parameter": "missing_stellar_to_gas_mass_ratio",
        "cause": "unmeasured_stellar_profile",
        "distribution": "uniform",
        "low": 0.02,
        "high": 0.3,
    },
    {
        "parameter": "clumping_amplitude",
        "cause": "gas_clumping",
        "distribution": "uniform",
        "low": 0.0,
        "high": 0.3,
    },
    {
        "parameter": "centering_radius_shift",
        "cause": "centering",
        "distribution": "uniform",
        "low": -0.05,
        "high": 0.05,
    },
    {
        "parameter": "projection_density_scale",
        "cause": "projection",
        "distribution": "uniform",
        "low": 0.85,
        "high": 1.15,
    },
    {
        "parameter": "triaxial_radius_scale",
        "cause": "triaxiality",
        "distribution": "uniform",
        "low": 0.85,
        "high": 1.15,
    },
    {
        "parameter": "spherical_acceleration_scale",
        "cause": "spherical_approximation",
        "distribution": "uniform",
        "low": 0.85,
        "high": 1.15,
    },
]

SOURCE_PATHS = {
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
    "source_smc_role": "starting_positions_only_with_all_starting_likelihoods_recomputed",
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
    "active_kernel": "symmetric_correlated_gaussian_with_whole_proposal_out_of_bounds_rejection",
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
START_GENERATION = {
    "engine": "scipy_qmc_sobol",
    "independent_scrambles": 4,
    "dimensions": 17,
    "samples_per_scramble": 512,
    "power_of_two_exponent": 9,
    "scramble": True,
    "scramble_seeds": [596200, 596201, 596202, 596203],
    "posterior_ancestry": False,
    "stored_likelihoods": False,
}
DATA_SEAL = {
    "runtime_packet_allowed_split": "development_train",
    "runtime_packet_required_rows": 80,
    "holdout_may_select_sampler_or_settings": False,
    "same_release_confirmation_rows_allowed": False,
    "independent_target_rows_allowed": False,
    "target_rows_opened": 0,
}
ORBIT_VALIDATION = {
    "moves_tested_separately": ["stellar", "geometry", "coupled"],
    "accepted_cases_required": True,
    "maximum_absolute_training_log_likelihood_difference": 1e-10,
    "maximum_absolute_composite_difference": 1e-12,
    "production_forward_evaluations": 192,
    "smoke_forward_evaluations": 24,
}
DIAGNOSTIC_VALIDATION = {
    "seed": 598100,
    "chains": 8,
    "draws": 512,
    "shifted_chain_offset": 3.0,
    "ar1_rho": 0.8,
    "maximum_iid_rhat": 1.05,
    "minimum_shifted_rhat": 1.2,
    "maximum_ar1_to_iid_ess_ratio": 0.5,
    "minimum_ar1_bulk_ess": 50,
    "minimum_scaled_within_chain_variance": 1e-14,
    "maximum_arviz_rhat_absolute_difference": 1e-12,
    "maximum_arviz_ess_relative_difference": 0.02,
    "constant_chain_must_fail_validity_gate": True,
}
UNIFORM_TARGET_CONTROL = {
    "role": "uniform_target_kernel_invariance_and_correlated_boundary_negative_control",
    "samples": 32768,
    "sweeps": 8,
    "initial_seed": 598001,
    "transition_seed": 598002,
    "proposal_ar1_correlation": 0.9,
    "proposal_scale": 0.45,
    "stellar_log_step": 0.08,
    "geometry_log_step": 0.03,
    "coupled_log_step": 0.02,
    "negative_control": "coordinatewise_reflection_with_the_same_correlated_gaussian_must_fail",
    "thresholds": {
        "maximum_absolute_mean_error": 0.012,
        "maximum_absolute_variance_error": 0.006,
        "maximum_absolute_offdiagonal_correlation": 0.05,
        "maximum_ks_distance_from_uniform": 0.015,
    },
}
COMPLETION_THRESHOLDS = {
    "maximum_rank_normalized_split_rhat": 1.2,
    "minimum_bulk_effective_samples": 50,
    "minimum_tail_effective_samples": 50,
    "maximum_standardized_between_replicate_median_spread": 0.25,
    "all_10_composite_coordinates_must_pass": True,
    "positive_variance_required_for_every_split_chain": True,
    "source_start_to_posterior_shift_is_a_gate": False,
}
MECHANICS_THRESHOLDS = {
    "minimum_retained_active_acceptance": 0.05,
    "maximum_retained_active_acceptance": 0.6,
    "minimum_retained_orbit_acceptance": 0.1,
    "all_replicates_must_pass": True,
}
ADJUDICATION = {
    "candidate_pass_alone_completes_CP5_7_through_CP5_10": False,
    "newtonian_control_required_after_candidate_pass": True,
    "simulation_based_calibration_required_after_candidate_pass": True,
    "source_covariance_required_before_CP5_completion": True,
    "primitive_orbit_labels_may_be_reported_as_separately_identified": False,
    "holdout_selection_allowed": False,
    "threshold_relaxation_after_result": False,
    "failed_run_retained": True,
    "newtonian_control_locked_until_candidate_pass": True,
}
AUTHORIZATION_POLICY = {
    "separate_manifest_required": True,
    "manifest_binds_contract_source_packet_starts_controls_and_smoke": True,
    "production_authorized_by_default": False,
    "explicit_cli_sentinel_required": True,
    "unauthorized_attempt_fails_before_runtime_packet_load": True,
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


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binding_rows() -> dict[str, dict[str, str]]:
    return {
        name: {"path": path, "file_sha256": file_sha256(ROOT / path)}
        for name, path in SOURCE_PATHS.items()
    }


def validate_source_bindings(bindings: dict[str, Any]) -> None:
    strict_keys(bindings, set(SOURCE_PATHS), "source_bindings")
    for name, expected_path in SOURCE_PATHS.items():
        row = bindings[name]
        strict_keys(row, {"path", "file_sha256"}, f"source_bindings.{name}")
        if row["path"] != expected_path:
            raise RuntimeError(f"source binding path changed for {name}")
        path = confined(ROOT / expected_path)
        if not path.is_file() or file_sha256(path) != row["file_sha256"]:
            raise RuntimeError(f"source binding missing or tampered for {name}")


def build_train_packet(output: Path) -> dict[str, Any]:
    config59 = item59.load_config(ROOT)
    source_packets = comparators._development_packets(ROOT, config59)
    packets: list[dict[str, Any]] = []
    for source in source_packets:
        rows = [jsonify(row) for row in source["rows"] if row.get("split") == "development_train"]
        packet = {
            "cluster": source["cluster"],
            "density_radius_kpc": jsonify(source["density_radius_kpc"]),
            "ne_cm3": jsonify(source["ne_cm3"]),
            "ne_error_low_cm3": jsonify(source["ne_error_low_cm3"]),
            "ne_error_high_cm3": jsonify(source["ne_error_high_cm3"]),
            "r500_kpc": jsonify(source["r500_kpc"]),
            "anchor": jsonify(source["anchor"]),
            "rows": rows,
            "stellar": jsonify(source["stellar"]),
        }
        packets.append(packet)
    row_count = sum(len(packet["rows"]) for packet in packets)
    body = {
        "schema_version": PACKET_SCHEMA,
        "status": "generated_train_only_runtime_packet_no_holdout_confirmation_or_independent_rows",
        "generation_scope": {
            "builder_may_read_previously_exposed_development_packets": True,
            "serialized_allowed_split": "development_train",
            "serialized_holdout_rows": 0,
            "serialized_confirmation_rows": 0,
            "serialized_independent_rows": 0,
        },
        "source_bindings": binding_rows(),
        "clusters": sorted(str(packet["cluster"]) for packet in packets),
        "cluster_count": len(packets),
        "row_count": row_count,
        "split_counts": {"development_train": row_count},
        "packets": packets,
    }
    body["content_sha256"] = canonical_hash(body)
    validate_train_packet_body(body)
    write_json(output, body)
    return body


def validate_train_packet_body(body: dict[str, Any]) -> None:
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "generation_scope",
            "source_bindings",
            "clusters",
            "cluster_count",
            "row_count",
            "split_counts",
            "packets",
            "content_sha256",
        },
        "train packet",
    )
    if body["schema_version"] != PACKET_SCHEMA:
        raise RuntimeError("train packet schema changed")
    expected_scope = {
        "builder_may_read_previously_exposed_development_packets": True,
        "serialized_allowed_split": "development_train",
        "serialized_holdout_rows": 0,
        "serialized_confirmation_rows": 0,
        "serialized_independent_rows": 0,
    }
    if body["generation_scope"] != expected_scope:
        raise RuntimeError("train packet generation scope changed")
    validate_source_bindings(body["source_bindings"])
    if int(body["cluster_count"]) != 8 or int(body["row_count"]) != 80:
        raise RuntimeError("train packet cluster or row count changed")
    if body["split_counts"] != {"development_train": 80}:
        raise RuntimeError("train packet split counts changed")
    if len(body["packets"]) != 8:
        raise RuntimeError("train packet count changed")
    required_packet_keys = {
        "cluster",
        "density_radius_kpc",
        "ne_cm3",
        "ne_error_low_cm3",
        "ne_error_high_cm3",
        "r500_kpc",
        "anchor",
        "rows",
        "stellar",
    }
    rows = []
    for index, packet in enumerate(body["packets"]):
        strict_keys(packet, required_packet_keys, f"train packet.packets[{index}]")
        rows.extend(packet["rows"])
    if len(rows) != 80 or any(row.get("split") != "development_train" for row in rows):
        raise RuntimeError("runtime packet contains a forbidden split")
    unhashed = dict(body)
    observed = unhashed.pop("content_sha256")
    if observed != canonical_hash(unhashed):
        raise RuntimeError("train packet content hash changed")


def load_train_packet(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    packet_path = confined(path)
    if not packet_path.is_file() or file_sha256(packet_path) != expected_sha256:
        raise RuntimeError("train-only packet missing or tampered")
    body = json.loads(packet_path.read_text(encoding="utf-8"))
    validate_train_packet_body(body)
    return body["packets"]


def build_sobol_starts(output: Path) -> dict[str, Any]:
    populations = []
    for seed in START_GENERATION["scramble_seeds"]:
        engine = qmc.Sobol(d=17, scramble=True, seed=int(seed))
        populations.append(engine.random_base2(m=9))
    particles = np.stack(populations)
    if particles.shape != (4, 512, 17) or np.any((particles <= 0) | (particles >= 1)):
        raise RuntimeError("Sobol start population invariant failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        particles=particles,
        generation=np.asarray(json.dumps(START_GENERATION, sort_keys=True)),
    )
    return {
        "shape": list(particles.shape),
        "minimum": float(np.min(particles)),
        "maximum": float(np.max(particles)),
        "file_sha256": file_sha256(output),
    }


def validate_sobol_starts(path: Path, expected_sha256: str) -> None:
    start_path = confined(path)
    if not start_path.is_file() or file_sha256(start_path) != expected_sha256:
        raise RuntimeError("Sobol starts missing or tampered")
    loaded = np.load(start_path, allow_pickle=False)
    strict_keys({name: None for name in loaded.files}, {"particles", "generation"}, "Sobol archive")
    particles = np.asarray(loaded["particles"], dtype=float)
    generation = json.loads(str(loaded["generation"].item()))
    if generation != START_GENERATION or particles.shape != (4, 512, 17):
        raise RuntimeError("Sobol start contract changed")
    if np.any(~np.isfinite(particles)) or np.any((particles <= 0) | (particles >= 1)):
        raise RuntimeError("Sobol starts left the open prior cube")


def expected_call_accounting() -> dict[str, Any]:
    production = PRODUCTION_SETTINGS
    smoke = SMOKE_SETTINGS
    production_proposals = (
        4
        * 512
        * (
            int(production["adaptation_sweeps"])
            + int(production["fixed_kernel_settling_sweeps"])
            + int(production["retained_sweeps"])
        )
    )
    return {
        "production_initialization_evaluations": 2048,
        "production_orbit_validation_evaluations": 192,
        "production_maximum_proposal_evaluations": production_proposals,
        "production_maximum_total_forward_evaluations": v6.maximum_forward_calls(production),
        "smoke_maximum_total_forward_evaluations": v6.maximum_forward_calls(smoke),
        "out_of_bounds_proposals_require_forward_evaluation": False,
        "actual_calls_must_equal_sum_of_reported_call_categories": True,
    }


def validate_exact_nested_contract(contract: dict[str, Any]) -> None:
    expected_objects = {
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
    for key, expected in expected_objects.items():
        if contract[key] != expected:
            raise RuntimeError(f"frozen nested contract object changed: {key}")
    v6.validate_sampler_settings(contract["production_settings"], "production settings", 4, 512)
    v6.validate_sampler_settings(contract["smoke_settings"], "smoke settings", 2, 32)


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
            "sobol_starts",
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
    if contract["status"] != "frozen_v6r2_controls_and_smoke_only_production_unauthorized":
        raise RuntimeError("contract status changed")
    if (
        contract["family"] != "cross_scale_boundary"
        or contract["likelihood_split"] != "development_train"
    ):
        raise RuntimeError("family or likelihood split changed")
    source_path = confined(ROOT / str(contract["prototype_source"]))
    if source_path != Path(__file__).resolve():
        raise RuntimeError("contract points to another prototype")
    if normalized_sha256(source_path) != contract["prototype_source_normalized_sha256"]:
        raise RuntimeError("V6R2 prototype changed after freeze")
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
        raise RuntimeError("hash-bound uncertainty config prior definitions changed")
    for label in ("train_packet", "sobol_starts"):
        row = contract[label]
        strict_keys(row, {"path", "file_sha256"}, label)
    load_train_packet(
        ROOT / contract["train_packet"]["path"], contract["train_packet"]["file_sha256"]
    )
    validate_sobol_starts(
        ROOT / contract["sobol_starts"]["path"], contract["sobol_starts"]["file_sha256"]
    )
    validate_exact_nested_contract(contract)
    contract["_execution_contract_sha256"] = observed_hash
    return contract


def rank_split_diagnostics(
    chains: np.ndarray, variance_floor: float | None = None
) -> dict[str, np.ndarray]:
    floor = (
        float(DIAGNOSTIC_VALIDATION["minimum_scaled_within_chain_variance"])
        if variance_floor is None
        else float(variance_floor)
    )
    split = v6.split_chains(chains)
    dimensions = split.shape[2]
    rhat = np.full(dimensions, np.inf)
    bulk_ess = np.zeros(dimensions)
    tail_ess = np.zeros(dimensions)
    minimum_scaled_variance = np.zeros(dimensions)
    valid = np.zeros(dimensions, dtype=bool)
    for dimension in range(dimensions):
        raw = split[:, :, dimension]
        scale = max(1.0, float(np.max(np.abs(raw))))
        chain_variances = np.var(raw, axis=1, ddof=1) / scale**2
        minimum_scaled_variance[dimension] = float(np.min(chain_variances))
        if (
            np.any(~np.isfinite(raw))
            or np.any(~np.isfinite(chain_variances))
            or np.min(chain_variances) <= floor
        ):
            continue
        ranked = v6.rank_normalize(raw)
        folded = v6.rank_normalize(np.abs(raw - np.median(raw)))
        rhat[dimension] = max(v6.classic_rhat(ranked), v6.classic_rhat(folded))
        bulk_ess[dimension] = v6.geyer_ess(ranked)
        low = float(np.quantile(raw, 0.05))
        high = float(np.quantile(raw, 0.95))
        tail_ess[dimension] = min(
            v6.geyer_ess((raw <= low).astype(float)),
            v6.geyer_ess((raw >= high).astype(float)),
        )
        valid[dimension] = bool(
            np.isfinite(rhat[dimension])
            and np.isfinite(bulk_ess[dimension])
            and np.isfinite(tail_ess[dimension])
        )
    return {
        "rhat": rhat,
        "bulk_ess": bulk_ess,
        "tail_ess": tail_ess,
        "valid": valid,
        "minimum_scaled_within_chain_variance": minimum_scaled_variance,
    }


def _arviz_values(chains: np.ndarray) -> dict[str, float | None]:
    raw = chains[:, :, 0]
    with np.errstate(all="ignore"):
        rhat = float(np.asarray(az.rhat(raw, method="rank")))
        bulk = float(np.asarray(az.ess(raw, method="bulk")))
        tail = float(np.asarray(az.ess(raw, method="tail")))
    return {
        "rhat": rhat if np.isfinite(rhat) else None,
        "bulk_ess": bulk if np.isfinite(bulk) else None,
        "tail_ess": tail if np.isfinite(tail) else None,
    }


def _diagnostic_row(chains: np.ndarray, floor: float) -> dict[str, Any]:
    ours = rank_split_diagnostics(chains, floor)
    own = {
        "valid": bool(ours["valid"][0]),
        "rhat": float(ours["rhat"][0]) if np.isfinite(ours["rhat"][0]) else None,
        "bulk_ess": float(ours["bulk_ess"][0]),
        "tail_ess": float(ours["tail_ess"][0]),
        "minimum_scaled_within_chain_variance": float(
            ours["minimum_scaled_within_chain_variance"][0]
        ),
    }
    return {"ours": own, "arviz": _arviz_values(chains)}


def diagnostic_validation_control(settings: dict[str, Any]) -> dict[str, Any]:
    if settings != DIAGNOSTIC_VALIDATION:
        raise RuntimeError("diagnostic validation settings changed")
    rng = np.random.default_rng(int(settings["seed"]))
    chains = int(settings["chains"])
    draws = int(settings["draws"])
    iid = rng.normal(size=(chains, draws, 1))
    shifted = iid.copy()
    shifted[0, :, 0] += float(settings["shifted_chain_offset"])
    rho = float(settings["ar1_rho"])
    ar1 = np.empty((chains, draws, 1))
    ar1[:, 0, 0] = rng.normal(size=chains)
    innovations = rng.normal(scale=math.sqrt(1.0 - rho**2), size=(chains, draws - 1))
    for draw in range(1, draws):
        ar1[:, draw, 0] = rho * ar1[:, draw - 1, 0] + innovations[:, draw - 1]
    constant = np.full((chains, draws, 1), 0.375)
    floor = float(settings["minimum_scaled_within_chain_variance"])
    cases = {
        "iid": _diagnostic_row(iid, floor),
        "shifted_chain": _diagnostic_row(shifted, floor),
        "ar1": _diagnostic_row(ar1, floor),
        "constant_chain_negative_control": _diagnostic_row(constant, floor),
    }
    rhat_differences = []
    ess_relative_differences = []
    for name in ("iid", "shifted_chain", "ar1"):
        own = cases[name]["ours"]
        reference = cases[name]["arviz"]
        rhat_differences.append(abs(float(own["rhat"]) - float(reference["rhat"])))
        for key in ("bulk_ess", "tail_ess"):
            capped_reference = min(float(reference[key]), float(chains * draws))
            ess_relative_differences.append(
                abs(float(own[key]) - capped_reference) / max(capped_reference, 1.0)
            )
    max_rhat = max(rhat_differences)
    max_ess = max(ess_relative_differences)
    iid_ours = cases["iid"]["ours"]
    shifted_ours = cases["shifted_chain"]["ours"]
    ar1_ours = cases["ar1"]["ours"]
    constant_ours = cases["constant_chain_negative_control"]["ours"]
    passed = bool(
        iid_ours["valid"]
        and float(iid_ours["rhat"]) <= float(settings["maximum_iid_rhat"])
        and shifted_ours["valid"]
        and float(shifted_ours["rhat"]) >= float(settings["minimum_shifted_rhat"])
        and ar1_ours["valid"]
        and float(ar1_ours["bulk_ess"])
        < float(iid_ours["bulk_ess"]) * float(settings["maximum_ar1_to_iid_ess_ratio"])
        and float(ar1_ours["bulk_ess"]) >= float(settings["minimum_ar1_bulk_ess"])
        and not constant_ours["valid"]
        and float(constant_ours["bulk_ess"]) == 0.0
        and float(constant_ours["tail_ess"]) == 0.0
        and max_rhat <= float(settings["maximum_arviz_rhat_absolute_difference"])
        and max_ess <= float(settings["maximum_arviz_ess_relative_difference"])
    )
    return {
        "passed": passed,
        "cases": cases,
        "parity": {
            "arviz_version": az.__version__,
            "maximum_rhat_absolute_difference": max_rhat,
            "maximum_ess_relative_difference": max_ess,
            "ess_reference_cap": chains * draws,
            "ess_parity_uses_same_finite_draw_cap_as_implementation": True,
            "rhat_threshold": settings["maximum_arviz_rhat_absolute_difference"],
            "ess_relative_threshold": settings["maximum_arviz_ess_relative_difference"],
        },
    }


def uniform_target_invariance_control(
    settings: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if settings != UNIFORM_TARGET_CONTROL:
        raise RuntimeError("uniform-target control settings changed")
    legacy_settings = dict(settings)
    legacy_settings.pop("role")
    result = _LEGACY_PRIOR_CONTROL(legacy_settings, config)
    result["control_name"] = (
        "uniform_target_kernel_invariance_and_correlated_boundary_negative_control"
    )
    result["likelihood_model"] = "constant_uniform_target_no_forward_model"
    return result


def _adapter_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": contract["family"],
        "source_smc_result": contract["sobol_starts"]["path"],
        "source_smc_sha256": contract["sobol_starts"]["file_sha256"],
        "diagnostic_validation": contract["diagnostic_validation"],
        "prior_recovery_control": contract["uniform_target_invariance_control"],
        "orbit_validation": contract["orbit_validation"],
        "completion_thresholds": {
            key: value
            for key, value in contract["completion_thresholds"].items()
            if key != "positive_variance_required_for_every_split_chain"
        },
        "mechanics_thresholds": contract["mechanics_thresholds"],
        "adjudication": contract["adjudication"],
        "_execution_contract_sha256": contract["_execution_contract_sha256"],
    }


def _rewrite_result(
    output: Path,
    aggregate: dict[str, Any],
    contract: dict[str, Any],
    smoke: bool,
    runtime_packet_provider_calls: int,
) -> dict[str, Any]:
    loaded = np.load(output, allow_pickle=False)
    traces = np.asarray(loaded["composite_traces"], dtype=float)
    ending_particles = np.asarray(loaded["ending_particles"], dtype=float)
    ending_log_likelihood = np.asarray(loaded["ending_log_likelihood"], dtype=float)
    chains = traces.reshape(-1, traces.shape[2], traces.shape[3])
    diagnostics = rank_split_diagnostics(chains)
    thresholds = contract["completion_thresholds"]
    coordinate_passes = []
    for index, parameter in enumerate(aggregate["parameters"]):
        valid = bool(diagnostics["valid"][index])
        rhat = float(diagnostics["rhat"][index])
        bulk = float(diagnostics["bulk_ess"][index])
        tail = float(diagnostics["tail_ess"][index])
        passed = bool(
            valid
            and rhat <= float(thresholds["maximum_rank_normalized_split_rhat"])
            and bulk >= float(thresholds["minimum_bulk_effective_samples"])
            and tail >= float(thresholds["minimum_tail_effective_samples"])
            and float(parameter["standardized_between_replicate_median_spread"])
            <= float(thresholds["maximum_standardized_between_replicate_median_spread"])
        )
        parameter["diagnostic_valid_positive_variance"] = valid
        parameter["minimum_scaled_within_chain_variance"] = float(
            diagnostics["minimum_scaled_within_chain_variance"][index]
        )
        parameter["rank_normalized_split_rhat"] = rhat if np.isfinite(rhat) else None
        parameter["bulk_effective_samples"] = bulk
        parameter["tail_effective_samples"] = tail
        parameter["passed"] = passed
        coordinate_passes.append(passed)
    mechanics_pass = bool(aggregate["all_mechanics_gates_passed"])
    production_passed = bool(all(coordinate_passes) and mechanics_pass and not smoke)
    aggregate["schema_version"] = "invariant-gravity-cluster-quotient-sampler-result-6.2"
    aggregate["decision"] = (
        "SMOKE_ONLY_NOT_PRODUCTION_ADJUDICATION"
        if smoke
        else (
            "CANDIDATE_SAMPLER_PASS_DOWNSTREAM_SBC_NEWTONIAN_AND_SOURCE_COVARIANCE_REQUIRED"
            if production_passed
            else "CANDIDATE_SAMPLER_FAIL_FROZEN_GATES_RESULT_RETAINED"
        )
    )
    aggregate.pop("source_smc_role", None)
    aggregate.pop("source_smc_sha256", None)
    aggregate["start_population"] = {
        "role": "four_independently_scrambled_sobol_prior_populations_no_posterior_ancestry",
        "generation": START_GENERATION,
        "file_sha256": contract["sobol_starts"]["file_sha256"],
        "every_initial_likelihood_recomputed_fresh": (
            int(aggregate["forward_call_accounting"]["initialization_evaluations"])
            == int(aggregate["replicates"]) * int(aggregate["particle_chains_per_replicate"])
        ),
    }
    aggregate["runtime_data_boundary"] = {
        "packet_sha256": contract["train_packet"]["file_sha256"],
        "allowed_split": "development_train",
        "rows_loaded": 80,
        "holdout_rows_loaded": 0,
        "confirmation_rows_loaded": 0,
        "independent_rows_loaded": 0,
        "runtime_packet_provider_calls": runtime_packet_provider_calls,
        "canonical_comparator_packet_builder_called_during_sampling": False,
    }
    controls = aggregate["controls"]
    controls["uniform_target_invariance"] = controls.pop("constant_likelihood_prior_recovery")
    aggregate["maximum_rank_normalized_split_rhat"] = (
        float(np.max(diagnostics["rhat"])) if np.all(np.isfinite(diagnostics["rhat"])) else None
    )
    aggregate["minimum_bulk_effective_samples"] = float(np.min(diagnostics["bulk_ess"]))
    aggregate["minimum_tail_effective_samples"] = float(np.min(diagnostics["tail_ess"]))
    aggregate["all_coordinates_positive_variance"] = bool(np.all(diagnostics["valid"]))
    aggregate["all_coordinate_gates_passed"] = bool(all(coordinate_passes))
    aggregate["production_passed"] = production_passed
    aggregate["downstream_sequence_if_candidate_passes"] = [
        "simulation_based_calibration",
        "matched_newtonian_control",
        "source_covariance",
    ]
    np.savez_compressed(
        output,
        composite_traces=traces,
        ending_particles=ending_particles,
        ending_log_likelihood=ending_log_likelihood,
        summary=np.asarray(json.dumps(aggregate, sort_keys=True, allow_nan=False)),
    )
    return aggregate


def run_sampler(
    contract: dict[str, Any], settings: dict[str, Any], output: Path, smoke: bool
) -> dict[str, Any]:
    packets = load_train_packet(
        ROOT / contract["train_packet"]["path"],
        contract["train_packet"]["file_sha256"],
    )
    provider_calls = 0

    def train_only_provider(_root: Path, _config: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal provider_calls
        provider_calls += 1
        return packets

    original_provider = v6.comparators._development_packets
    original_rank = v6.rank_split_diagnostics
    original_diagnostic = v6.diagnostic_validation_control
    original_prior = v6.prior_recovery_control
    v6.comparators._development_packets = train_only_provider
    v6.rank_split_diagnostics = rank_split_diagnostics
    v6.diagnostic_validation_control = diagnostic_validation_control
    v6.prior_recovery_control = uniform_target_invariance_control
    try:
        aggregate = v6.run_sampler(_adapter_contract(contract), settings, output, smoke=smoke)
    finally:
        v6.comparators._development_packets = original_provider
        v6.rank_split_diagnostics = original_rank
        v6.diagnostic_validation_control = original_diagnostic
        v6.prior_recovery_control = original_prior
    if provider_calls != 1:
        raise RuntimeError("train-only runtime packet provider call count changed")
    return _rewrite_result(output, aggregate, contract, smoke, provider_calls)


def artifact_binding(path: Path) -> dict[str, str]:
    confined_path = confined(path)
    return {
        "path": confined_path.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(confined_path),
    }


def write_authorization(
    contract_path: Path,
    expected_contract_sha256: str,
    controls_path: Path,
    smoke_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path, expected_contract_sha256)
    controls = json.loads(confined(controls_path).read_text(encoding="utf-8"))
    smoke = np.load(confined(smoke_path), allow_pickle=False)
    smoke_summary = json.loads(str(smoke["summary"].item()))
    if (
        not controls.get("passed")
        or controls.get("execution_contract_sha256") != expected_contract_sha256
        or smoke_summary.get("execution_contract_sha256") != expected_contract_sha256
        or smoke_summary.get("mode") != "smoke"
    ):
        raise RuntimeError("control or smoke receipt is not valid for this contract")
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "controls_and_smoke_bound_production_not_authorized",
        "bindings": {
            "contract": artifact_binding(contract_path),
            "prototype_source": artifact_binding(ROOT / contract["prototype_source"]),
            "train_packet": artifact_binding(ROOT / contract["train_packet"]["path"]),
            "sobol_starts": artifact_binding(ROOT / contract["sobol_starts"]["path"]),
            "controls": artifact_binding(controls_path),
            "smoke": artifact_binding(smoke_path),
        },
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
            "newtonian_control_unlocked": False,
            "simulation_based_calibration_unlocked": False,
        },
    }
    write_json(output, body)
    return body


def validate_authorization(
    path: Path, expected_sha256: str, *, require_production: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization_path = confined(path)
    if file_sha256(authorization_path) != expected_sha256:
        raise RuntimeError("authorization manifest hash differs from expected hash")
    body = json.loads(authorization_path.read_text(encoding="utf-8"))
    strict_keys(
        body,
        {"schema_version", "status", "bindings", "production_authorization", "claim_boundary"},
        "authorization",
    )
    if body["schema_version"] != AUTHORIZATION_SCHEMA:
        raise RuntimeError("authorization schema changed")
    if body["status"] != "controls_and_smoke_bound_production_not_authorized":
        raise RuntimeError("authorization status changed")
    expected_claim_boundary = {
        "controls_passed": True,
        "bounded_smoke_executed": True,
        "candidate_production_executed": False,
        "candidate_claim_allowed": False,
        "newtonian_control_unlocked": False,
        "simulation_based_calibration_unlocked": False,
    }
    if body["claim_boundary"] != expected_claim_boundary:
        raise RuntimeError("authorization claim boundary changed")
    production = body["production_authorization"]
    strict_keys(
        production,
        {"authorized", "approved_by", "approval_id", "maximum_forward_evaluations"},
        "authorization.production_authorization",
    )
    if require_production and not production["authorized"]:
        raise RuntimeError(
            "production is not authorized; refusal occurs before contract or runtime packet load"
        )
    if production["authorized"]:
        if (
            production["approved_by"] != "Henry"
            or not isinstance(production["approval_id"], str)
            or not production["approval_id"].strip()
            or int(production["maximum_forward_evaluations"]) != 1_575_104
        ):
            raise RuntimeError("production authorization fields are incomplete")
    else:
        if production != {
            "authorized": False,
            "approved_by": None,
            "approval_id": None,
            "maximum_forward_evaluations": 0,
        }:
            raise RuntimeError("unauthorized manifest fields changed")
    bindings = body["bindings"]
    strict_keys(
        bindings,
        {"contract", "prototype_source", "train_packet", "sobol_starts", "controls", "smoke"},
        "authorization.bindings",
    )
    for name, row in bindings.items():
        strict_keys(row, {"path", "file_sha256"}, f"authorization.bindings.{name}")
        target = confined(ROOT / row["path"])
        if not target.is_file() or file_sha256(target) != row["file_sha256"]:
            raise RuntimeError(f"authorization binding missing or tampered: {name}")
    contract_binding = bindings["contract"]
    contract = load_contract(ROOT / contract_binding["path"], contract_binding["file_sha256"])
    if bindings["prototype_source"]["file_sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("authorization does not bind this executable source")
    controls = json.loads((ROOT / bindings["controls"]["path"]).read_text(encoding="utf-8"))
    smoke = np.load(ROOT / bindings["smoke"]["path"], allow_pickle=False)
    smoke_summary = json.loads(str(smoke["summary"].item()))
    contract_hash = contract["_execution_contract_sha256"]
    if (
        controls.get("execution_contract_sha256") != contract_hash
        or not controls.get("passed")
        or smoke_summary.get("execution_contract_sha256") != contract_hash
        or smoke_summary.get("mode") != "smoke"
    ):
        raise RuntimeError("authorization evidence does not match the contract")
    return body, contract


def controls_receipt(contract: dict[str, Any]) -> dict[str, Any]:
    config = uncertainty.load_config(ROOT)
    diagnostic = diagnostic_validation_control(contract["diagnostic_validation"])
    uniform = uniform_target_invariance_control(
        contract["uniform_target_invariance_control"], config
    )
    result = {
        "schema_version": "invariant-gravity-cluster-quotient-sampler-controls-6.2",
        "execution_contract_sha256": contract["_execution_contract_sha256"],
        "prototype_source_normalized_sha256": contract["prototype_source_normalized_sha256"],
        "train_packet_sha256": contract["train_packet"]["file_sha256"],
        "sobol_starts_sha256": contract["sobol_starts"]["file_sha256"],
        "diagnostic_validation": diagnostic,
        "uniform_target_invariance": uniform,
        "forward_evaluations": 0,
        "passed": bool(diagnostic["passed"] and uniform["passed"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    packet_command = subparsers.add_parser("build-train-packet")
    packet_command.add_argument("--output", type=Path, required=True)

    starts_command = subparsers.add_parser("build-sobol-starts")
    starts_command.add_argument("--output", type=Path, required=True)

    for name in ("controls", "smoke"):
        command = subparsers.add_parser(name)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--expected-contract-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)

    authorization_command = subparsers.add_parser("write-authorization")
    authorization_command.add_argument("--contract", type=Path, required=True)
    authorization_command.add_argument("--expected-contract-sha256", required=True)
    authorization_command.add_argument("--controls", type=Path, required=True)
    authorization_command.add_argument("--smoke", type=Path, required=True)
    authorization_command.add_argument("--output", type=Path, required=True)

    validate_command = subparsers.add_parser("validate-authorization")
    validate_command.add_argument("--authorization", type=Path, required=True)
    validate_command.add_argument("--expected-authorization-sha256", required=True)

    run_command = subparsers.add_parser("run")
    run_command.add_argument("--authorization", type=Path, required=True)
    run_command.add_argument("--expected-authorization-sha256", required=True)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--execute-frozen-production-v6r2", action="store_true")

    args = parser.parse_args()
    if args.command == "build-train-packet":
        result = build_train_packet(args.output)
        print(
            json.dumps(
                {
                    "path": str(args.output),
                    "file_sha256": file_sha256(args.output),
                    "content_sha256": result["content_sha256"],
                    "clusters": result["cluster_count"],
                    "rows": result["row_count"],
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "build-sobol-starts":
        print(json.dumps(build_sobol_starts(args.output), sort_keys=True))
        return
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
    if args.command == "write-authorization":
        result = write_authorization(
            args.contract,
            args.expected_contract_sha256,
            args.controls,
            args.smoke,
            args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "validate-authorization":
        body, contract = validate_authorization(
            args.authorization,
            args.expected_authorization_sha256,
            require_production=False,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "authorization_status": body["status"],
                    "production_authorized": body["production_authorization"]["authorized"],
                    "execution_contract_sha256": contract["_execution_contract_sha256"],
                },
                sort_keys=True,
            )
        )
        return
    if not args.execute_frozen_production_v6r2:
        raise RuntimeError(
            "production requires the explicit --execute-frozen-production-v6r2 sentinel"
        )
    _authorization, contract = validate_authorization(
        args.authorization,
        args.expected_authorization_sha256,
        require_production=True,
    )
    result = run_sampler(contract, contract["production_settings"], args.output, smoke=False)
    print(json.dumps(result, sort_keys=True))
    if not result["production_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
