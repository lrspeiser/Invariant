from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc as v1
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v2 as v2

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_sbc_adjudicator_v1.json")
RECEIPT_PATH = Path("runs/gravity/publication-readiness/nuisance-quotient-sbc-adjudicator-v1.json")
TEST_PATH = Path("tests/test_gravity_cluster_nuisance_quotient_sbc_adjudicator.py")

CONFIG_SCHEMA = "invariant-gravity-nuisance-quotient-sbc-adjudicator-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-nuisance-quotient-sbc-adjudicator-receipt-1.0"
STATUS = "strictly_verified_v1_v2_failed_production_locked"
DECISION = "SBC_V1_AND_V2_FAILED_NO_NEWTONIAN_OR_CANDIDATE_PRODUCTION_UNLOCK"

EXPECTED_SBC_ARTIFACTS = {
    "v1_source": {
        "path": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sbc.py",
        "file_sha256": "0eaad327544454c3540bfd99e87530caf3f0ace5e9d0065798125d98de863ba3",
    },
    "v1_config": {
        "path": "configs/gravity_cluster_nuisance_quotient_sbc_v1.json",
        "file_sha256": "f9a9acf4ee4f1558ce97ab489423c7f43c09bb38af764c7d17de5109433b314c",
    },
    "v1_result": {
        "path": "runs/gravity/publication-readiness/nuisance-quotient-sbc-v1/bounded-synthetic-sbc.npz",
        "file_sha256": "90071721c53e4a635443a652541fc3eeae6c5fd9822bc84a77aa460ecab375f3",
    },
    "v1_receipt": {
        "path": "runs/gravity/publication-readiness/nuisance-quotient-sbc-v1.json",
        "file_sha256": "afb4aa64850fc6449272d8ac2174c7c6c7dfa20c319259c48c5cecbd11dcaa11",
    },
    "v1_test": {
        "path": "tests/test_gravity_cluster_nuisance_quotient_sbc.py",
        "file_sha256": "a3c0bd9d7218ed7036a3df8f30aa92b3a8e84b8f416675d8da0aaf4639eff57a",
    },
    "v2_source": {
        "path": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sbc_v2.py",
        "file_sha256": "93b0a51fa0074e191f2c21fc04ad3e62eb1038a85c225e5cde62d90fade33c4a",
    },
    "v2_config": {
        "path": "configs/gravity_cluster_nuisance_quotient_sbc_v2.json",
        "file_sha256": "5251b7daf22d42c5455e2a5fa3c09ebac8be2ebb0a5f976d3632e53599c7507d",
    },
    "v2_result": {
        "path": "runs/gravity/publication-readiness/nuisance-quotient-sbc-v2/bounded-synthetic-sbc-v2.npz",
        "file_sha256": "6d731650848aba063f0aed8fc835dc65dda4524c63f592ff2045a833fd037053",
    },
    "v2_receipt": {
        "path": "runs/gravity/publication-readiness/nuisance-quotient-sbc-v2.json",
        "file_sha256": "da0db9cf28e97d82de19c65eccbf34cab44a90b62c1a43e2673941e7a17d674e",
    },
    "v2_test": {
        "path": "tests/test_gravity_cluster_nuisance_quotient_sbc_v2.py",
        "file_sha256": "0fb6a209bd29dbdb0fed37110832a666f94d32fd7b38ade93800dc36509138c0",
    },
}
EXPECTED_VERIFIER_TEST_SHA256 = "1bb2a805cd86a84fd865d075ea5e6ede69457bde6e4c5a4d92a204655b885916"

DATA_BOUNDARY = {
    "synthetic_data_only": True,
    "real_development_rows_loaded": 0,
    "real_holdout_rows_loaded": 0,
    "real_confirmation_rows_loaded": 0,
    "real_independent_rows_loaded": 0,
    "network_calls": 0,
    "paid_model_calls": 0,
    "candidate_production_runs": 0,
    "newtonian_control_production_runs": 0,
}
CLAIM_BOUNDARY = {
    "artifact_integrity_verified": True,
    "v1_passed": False,
    "v2_passed": False,
    "candidate_production_may_unlock": False,
    "newtonian_control_may_unlock": False,
    "candidate_physics_supported": False,
    "scientific_claim_allowed": False,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    ).hexdigest()


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


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


def validate_binding(binding: dict[str, Any], label: str) -> Path:
    strict_keys(binding, {"path", "file_sha256"}, label)
    target = confined(ROOT / str(binding["path"]))
    if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
        raise RuntimeError(f"{label} missing or hash changed")
    return target


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("strict SBC adjudicator config hash changed")
    config = json.loads(target.read_text(encoding="utf-8"))
    strict_keys(
        config,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "verifier_test",
            "bound_sbc_artifacts",
            "v1_call_accounting",
            "v2_call_accounting",
            "v1_chronology",
            "v2_chronology",
            "v1_canonical_sampler_binding",
            "v2_canonical_sampler_binding",
            "data_boundary",
            "claim_boundary",
            "receipt_path",
        },
        "strict SBC adjudicator config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_append_only_strict_adjudication"
        or config["bound_sbc_artifacts"] != EXPECTED_SBC_ARTIFACTS
        or config["verifier_test"]
        != {
            "path": TEST_PATH.as_posix(),
            "file_sha256": EXPECTED_VERIFIER_TEST_SHA256,
        }
        or config["data_boundary"] != DATA_BOUNDARY
        or config["claim_boundary"] != CLAIM_BOUNDARY
        or config["v1_call_accounting"] != v1.maximum_call_accounting()
        or config["v2_call_accounting"] != v2.maximum_call_accounting()
        or config["v1_chronology"]
        != {
            "config_and_gates_frozen_before_first_result": True,
            "result_written_before_receipt": True,
            "receipt_requires_hash_bound_result": True,
            "failed_result_retained": True,
            "post_result_threshold_changes_forbidden": True,
        }
        or config["v2_chronology"]
        != {
            "v1_failure_diagnosed_before_v2_design": True,
            "v1_files_unchanged": True,
            "v2_source_config_scenarios_gates_and_budget_frozen_before_first_run": True,
            "v2_result_written_before_receipt": True,
            "failed_v2_result_retained": True,
            "post_result_threshold_changes_forbidden": True,
        }
        or config["v1_canonical_sampler_binding"]
        != {
            "path": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sampler.py",
            "file_sha256": "975b9f69a614d7d419dcc44ac340c86b27a85e6e9fed7ed63d6f6caff228abcb",
        }
        or config["v2_canonical_sampler_binding"] != config["v1_canonical_sampler_binding"]
        or config["receipt_path"] != RECEIPT_PATH.as_posix()
    ):
        raise RuntimeError("strict SBC adjudicator frozen contract changed")
    source = confined(ROOT / config["implementation_source"])
    if (
        source != Path(__file__).resolve()
        or normalized_sha256(source) != config["implementation_source_normalized_sha256"]
    ):
        raise RuntimeError("strict SBC adjudicator source changed")
    validate_binding(config["verifier_test"], "verifier_test")
    for name, binding in config["bound_sbc_artifacts"].items():
        validate_binding(binding, f"bound_sbc_artifacts.{name}")
    config["_config_sha256"] = expected_sha256
    return config


def rank_summary(ranks: np.ndarray, bins: int) -> dict[str, Any]:
    simulations = ranks.shape[0]
    standard_error = math.sqrt(1.0 / (12.0 * simulations))
    mean = np.mean(ranks, axis=0)
    z = (mean - 0.5) / standard_error
    return {
        "mean_normalized_rank": mean.tolist(),
        "mean_rank_z": z.tolist(),
        "maximum_absolute_mean_rank_z": float(np.max(np.abs(z))),
        "histograms": [
            np.histogram(ranks[:, index], bins=bins, range=(0.0, 1.0))[0].tolist()
            for index in range(ranks.shape[1])
        ],
        "bins": bins,
    }


def coverage_summary(rows: np.ndarray, levels: list[float]) -> dict[str, Any]:
    observed = np.mean(rows, axis=0)
    errors = np.abs(observed - np.asarray(levels)[None, :])
    return {
        "levels": levels,
        "observed_by_coordinate": observed.tolist(),
        "maximum_absolute_error_by_level": np.max(errors, axis=0).tolist(),
    }


def expected_truth_units(module: Any) -> np.ndarray:
    rows = []
    for scenario_index, scenario in enumerate(module.SCENARIOS):
        for simulation_index in range(int(scenario["simulations"])):
            offset = scenario_index * int(
                module.SEED_LINEAGE["scenario_stride"]
            ) + simulation_index * int(module.SEED_LINEAGE["simulation_stride"])
            rows.append(
                np.random.default_rng(int(module.SEED_LINEAGE["truth_base"]) + offset).random(17)
            )
    return np.asarray(rows)


def validate_orbit(row: dict[str, Any], threshold: float, label: str) -> bool:
    strict_keys(
        row,
        {"accepted_cases", "maximum_absolute_composite_difference", "passed"},
        label,
    )
    expected = bool(
        row["accepted_cases"] == {"stellar": 8, "geometry": 8, "coupled": 8}
        and float(row["maximum_absolute_composite_difference"]) <= threshold
    )
    if row["passed"] is not expected:
        raise RuntimeError(f"{label} pass flag disagrees with frozen gate")
    return expected


def read_npz(path: Path, expected_keys: set[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target = confined(path)
    with np.load(target, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys | {"summary"}:
            raise RuntimeError("sealed SBC NPZ keys changed")
        arrays = {name: np.asarray(archive[name]) for name in expected_keys}
        summary = json.loads(str(archive["summary"].item()))
    return arrays, summary


def validate_v1_fit(row: dict[str, Any]) -> None:
    strict_keys(
        row,
        {
            "scenario_id",
            "simulation_index",
            "all_coordinates_diagnostic_valid",
            "maximum_rhat",
            "minimum_bulk_ess",
            "minimum_tail_ess",
            "active_evaluated",
            "active_out_of_bounds_self_loops",
            "active_accepted",
            "orbit_attempted",
            "orbit_accepted",
            "initial_likelihoods_recomputed_fresh",
            "importance_effective_samples",
        },
        "V1 fit summary",
    )
    attempts = 32 * (
        int(v1.INFERENCE["adaptation_sweeps"])
        + int(v1.INFERENCE["fixed_kernel_settling_sweeps"])
        + int(v1.INFERENCE["retained_sweeps"])
    )
    if (
        row["initial_likelihoods_recomputed_fresh"] != 32
        or row["active_evaluated"] + row["active_out_of_bounds_self_loops"] != attempts
        or not 0 <= row["active_accepted"] <= row["active_evaluated"]
        or row["orbit_attempted"] != attempts * 3
        or not 0 <= row["orbit_accepted"] <= row["orbit_attempted"]
    ):
        raise RuntimeError("V1 fit mechanics accounting changed")


def validate_v2_candidate_fit(row: dict[str, Any]) -> None:
    strict_keys(
        row,
        {
            "scenario_id",
            "simulation_index",
            "all_coordinates_diagnostic_valid",
            "valid_coordinate_count",
            "maximum_rhat",
            "minimum_bulk_ess",
            "minimum_tail_ess",
            "active_attempted",
            "active_evaluated",
            "active_accepted",
            "active_out_of_bounds_self_loops",
            "orbit_attempted",
            "orbit_accepted",
            "initial_likelihoods_recomputed_fresh",
        },
        "V2 candidate fit summary",
    )
    attempts = 32 * (
        int(v2.CANDIDATE_INFERENCE["adaptation_sweeps"])
        + int(v2.CANDIDATE_INFERENCE["fixed_kernel_settling_sweeps"])
        + int(v2.CANDIDATE_INFERENCE["retained_sweeps"])
    )
    if (
        row["initial_likelihoods_recomputed_fresh"] != 32
        or row["active_attempted"] != attempts
        or row["active_evaluated"] + row["active_out_of_bounds_self_loops"] != attempts
        or not 0 <= row["active_accepted"] <= row["active_evaluated"]
        or row["orbit_attempted"] != attempts * 3
        or not 0 <= row["orbit_accepted"] <= row["orbit_attempted"]
        or not 0 <= row["valid_coordinate_count"] <= 10
        or row["all_coordinates_diagnostic_valid"] is not (row["valid_coordinate_count"] == 10)
    ):
        raise RuntimeError("V2 candidate mechanics accounting changed")


def validate_v2_reference(row: dict[str, Any]) -> None:
    strict_keys(
        row,
        {
            "scenario_id",
            "simulation_index",
            "replicates",
            "maximum_standardized_replicate_mean_difference",
            "minimum_stage_conditional_ess_fraction",
            "minimum_final_unique_initial_ancestor_fraction",
            "all_replicates_reached_beta_one",
            "calls",
        },
        "V2 reference summary",
    )
    if len(row["replicates"]) != 2:
        raise RuntimeError("V2 reference replicate count changed")
    calls = 0
    for replicate_index, replicate in enumerate(row["replicates"]):
        strict_keys(
            replicate,
            {
                "replicate",
                "algorithm_id",
                "terminal_beta",
                "stages",
                "minimum_stage_conditional_ess_fraction",
                "final_unique_initial_ancestor_fraction",
                "stage_rows",
                "calls",
            },
            "V2 reference replicate",
        )
        if (
            replicate["replicate"] != replicate_index
            or replicate["algorithm_id"] != v2.REFERENCE["algorithm_id"]
            or replicate["terminal_beta"] != 1.0
            or replicate["stages"] != len(replicate["stage_rows"])
            or not 1 <= replicate["stages"] <= 16
        ):
            raise RuntimeError("V2 reference replicate identity changed")
        previous_beta = 0.0
        valid_calls = int(v2.REFERENCE["particles_per_replicate"])
        minimum_ess = math.inf
        for stage_index, stage in enumerate(replicate["stage_rows"], start=1):
            strict_keys(
                stage,
                {
                    "stage",
                    "beta",
                    "conditional_ess",
                    "conditional_ess_fraction",
                    "coordinate_mh_acceptance",
                    "out_of_bounds_self_loops",
                    "ending_coordinate_step",
                    "unique_initial_ancestor_fraction",
                },
                "V2 reference stage",
            )
            if (
                stage["stage"] != stage_index
                or not previous_beta < stage["beta"] <= 1.0
                or stage["conditional_ess_fraction"]
                != stage["conditional_ess"] / int(v2.REFERENCE["particles_per_replicate"])
            ):
                raise RuntimeError("V2 reference tempering chronology changed")
            attempted = int(v2.REFERENCE["particles_per_replicate"]) * int(
                v2.REFERENCE["coordinate_mh_sweeps_per_stage"]
            )
            if not 0 <= stage["out_of_bounds_self_loops"] <= attempted:
                raise RuntimeError("V2 reference boundary accounting changed")
            valid_calls += attempted - stage["out_of_bounds_self_loops"]
            minimum_ess = min(minimum_ess, stage["conditional_ess_fraction"])
            previous_beta = stage["beta"]
        if (
            replicate["calls"] != valid_calls
            or replicate["minimum_stage_conditional_ess_fraction"] != minimum_ess
        ):
            raise RuntimeError("V2 reference call or ESS accounting changed")
        calls += replicate["calls"]
    if (
        row["calls"] != calls
        or row["minimum_stage_conditional_ess_fraction"]
        != min(
            replicate["minimum_stage_conditional_ess_fraction"] for replicate in row["replicates"]
        )
        or row["minimum_final_unique_initial_ancestor_fraction"]
        != min(
            replicate["final_unique_initial_ancestor_fraction"] for replicate in row["replicates"]
        )
        or row["all_replicates_reached_beta_one"] is not True
    ):
        raise RuntimeError("V2 reference aggregate accounting changed")


def validate_v1_result(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "truth_units",
        "scenario_indices",
        "candidate_normalized_ranks",
        "oracle_weighted_cdf_ranks",
        "candidate_coverage",
        "oracle_coverage",
        "candidate_tie_counts",
        "oracle_tie_mass",
        "importance_effective_samples",
    }
    arrays, summary = read_npz(path, expected_keys)
    strict_keys(
        summary,
        {
            "schema_version",
            "decision",
            "config_sha256",
            "passed",
            "scenario_summaries",
            "orbit_invariance_control",
            "fit_summaries",
            "call_accounting",
            "data_boundary",
            "claim_boundary",
            "chronology",
        },
        "V1 result summary",
    )
    simulations = sum(int(row["simulations"]) for row in v1.SCENARIOS)
    expected_shapes = {
        "truth_units": (simulations, 17),
        "scenario_indices": (simulations,),
        "candidate_normalized_ranks": (simulations, 10),
        "oracle_weighted_cdf_ranks": (simulations, 10),
        "candidate_coverage": (simulations, 10, 3),
        "oracle_coverage": (simulations, 10, 3),
        "candidate_tie_counts": (simulations, 10),
        "oracle_tie_mass": (simulations, 10),
        "importance_effective_samples": (simulations,),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise RuntimeError("V1 result array shape changed")
    if not np.array_equal(arrays["truth_units"], expected_truth_units(v1)):
        raise RuntimeError("V1 truth seed replay changed")
    if (
        summary["schema_version"] != v1.RESULT_SCHEMA
        or summary["config_sha256"] != config["bound_sbc_artifacts"]["v1_config"]["file_sha256"]
    ):
        raise RuntimeError("V1 result identity changed")
    for row in summary["fit_summaries"]:
        validate_v1_fit(row)
    levels = list(map(float, v1.RANK_PROTOCOL["coverage_levels"]))
    scenario_rows = []
    expected_indices = []
    for scenario_index, scenario in enumerate(v1.SCENARIOS):
        expected_indices.extend([scenario_index] * int(scenario["simulations"]))
        selected = arrays["scenario_indices"] == scenario_index
        fits = [
            row for row in summary["fit_summaries"] if row["scenario_id"] == scenario["scenario_id"]
        ]
        if len(fits) != int(scenario["simulations"]) or [
            row["simulation_index"] for row in fits
        ] != list(range(int(scenario["simulations"]))):
            raise RuntimeError("V1 fit scenario chronology changed")
        candidate_rank = rank_summary(
            arrays["candidate_normalized_ranks"][selected],
            int(v1.RANK_PROTOCOL["rank_histogram_bins"]),
        )
        oracle_rank = rank_summary(
            arrays["oracle_weighted_cdf_ranks"][selected],
            int(v1.RANK_PROTOCOL["rank_histogram_bins"]),
        )
        candidate_coverage = coverage_summary(arrays["candidate_coverage"][selected], levels)
        oracle_coverage = coverage_summary(arrays["oracle_coverage"][selected], levels)
        difference = float(
            np.max(
                np.abs(
                    np.mean(arrays["candidate_normalized_ranks"][selected], axis=0)
                    - np.mean(arrays["oracle_weighted_cdf_ranks"][selected], axis=0)
                )
            )
        )
        valid_fraction = float(np.mean([row["all_coordinates_diagnostic_valid"] for row in fits]))
        finite_rhats = [
            float(row["maximum_rhat"]) for row in fits if row["maximum_rhat"] is not None
        ]
        candidate_coverage_passed = all(
            candidate_coverage["maximum_absolute_error_by_level"][index]
            <= float(v1.GATES["maximum_absolute_candidate_coverage_error"][str(level)])
            for index, level in enumerate(levels)
        )
        oracle_coverage_passed = all(
            oracle_coverage["maximum_absolute_error_by_level"][index]
            <= float(v1.GATES["maximum_absolute_oracle_coverage_error"][str(level)])
            for index, level in enumerate(levels)
        )
        scenario_passed = bool(
            candidate_rank["maximum_absolute_mean_rank_z"]
            <= float(v1.GATES["maximum_absolute_candidate_mean_rank_z"])
            and oracle_rank["maximum_absolute_mean_rank_z"]
            <= float(v1.GATES["maximum_absolute_oracle_mean_rank_z"])
            and difference
            <= float(v1.GATES["maximum_absolute_candidate_oracle_mean_rank_difference"])
            and candidate_coverage_passed
            and oracle_coverage_passed
            and min(row["importance_effective_samples"] for row in fits)
            >= float(v1.GATES["minimum_importance_effective_samples"])
            and valid_fraction
            >= float(v1.GATES["minimum_fraction_fits_all_coordinates_diagnostic_valid"])
            and len(finite_rhats) == len(fits)
            and max(finite_rhats) <= float(v1.GATES["maximum_rank_normalized_split_rhat"])
            and min(row["minimum_bulk_ess"] for row in fits)
            >= float(v1.GATES["minimum_bulk_effective_samples_per_valid_coordinate"])
            and min(row["minimum_tail_ess"] for row in fits)
            >= float(v1.GATES["minimum_tail_effective_samples_per_valid_coordinate"])
        )
        scenario_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "simulations": int(np.count_nonzero(selected)),
                "candidate_rank": candidate_rank,
                "oracle_rank": oracle_rank,
                "candidate_coverage": candidate_coverage,
                "oracle_coverage": oracle_coverage,
                "maximum_absolute_candidate_oracle_mean_rank_difference": difference,
                "minimum_importance_effective_samples": min(
                    row["importance_effective_samples"] for row in fits
                ),
                "fraction_fits_all_coordinates_diagnostic_valid": valid_fraction,
                "maximum_fit_rhat": max(finite_rhats) if finite_rhats else None,
                "minimum_fit_bulk_ess": min(row["minimum_bulk_ess"] for row in fits),
                "minimum_fit_tail_ess": min(row["minimum_tail_ess"] for row in fits),
                "passed": scenario_passed,
            }
        )
    if not np.array_equal(arrays["scenario_indices"], np.asarray(expected_indices)):
        raise RuntimeError("V1 scenario index ordering changed")
    if not np.array_equal(
        arrays["importance_effective_samples"],
        np.asarray([row["importance_effective_samples"] for row in summary["fit_summaries"]]),
    ):
        raise RuntimeError("V1 importance ESS array disagrees with fit summaries")
    if summary["scenario_summaries"] != scenario_rows:
        raise RuntimeError("V1 stored scenario adjudication changed")
    orbit_passed = validate_orbit(
        summary["orbit_invariance_control"],
        float(v1.GATES["maximum_orbit_composite_difference"]),
        "V1 orbit control",
    )
    passed = bool(all(row["passed"] for row in scenario_rows) and orbit_passed)
    decision = (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_PASSED_NOT_PHYSICS_OR_PRODUCTION"
        if passed
        else "BOUNDED_SYNTHETIC_QUOTIENT_SBC_FAILED_RESULT_RETAINED"
    )
    candidate_calls = sum(
        row["initial_likelihoods_recomputed_fresh"] + row["active_evaluated"]
        for row in summary["fit_summaries"]
    )
    oracle_calls = simulations * int(v1.ORACLE["prior_particles_per_simulation"])
    calls = {
        "actual_mcmc_synthetic_likelihood_evaluations": candidate_calls,
        "oracle_synthetic_likelihood_evaluations": oracle_calls,
        "actual_total_synthetic_likelihood_evaluations": candidate_calls + oracle_calls,
        "frozen_maximum_total_synthetic_likelihood_evaluations": config["v1_call_accounting"][
            "maximum_total_synthetic_likelihood_evaluations"
        ],
        "real_forward_model_evaluations": 0,
    }
    if (
        summary["passed"] is not passed
        or summary["decision"] != decision
        or summary["call_accounting"] != calls
        or calls["actual_total_synthetic_likelihood_evaluations"]
        > calls["frozen_maximum_total_synthetic_likelihood_evaluations"]
        or summary["data_boundary"] != v1.DATA_BOUNDARY
        or summary["claim_boundary"] != v1.CLAIM_BOUNDARY
        or summary["chronology"] != config["v1_chronology"]
    ):
        raise RuntimeError("V1 result decision, calls, or boundary changed")
    return summary


def validate_v2_result(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "truth_units",
        "scenario_indices",
        "candidate_normalized_ranks",
        "reference_normalized_ranks",
        "candidate_coverage",
        "reference_coverage",
        "candidate_tie_counts",
        "reference_tie_mass",
    }
    arrays, summary = read_npz(path, expected_keys)
    strict_keys(
        summary,
        {
            "schema_version",
            "decision",
            "config_sha256",
            "passed",
            "v1_diagnosis",
            "scenario_results",
            "candidate_fit_summaries",
            "independent_reference_summaries",
            "orbit_invariance_control",
            "call_accounting",
            "data_boundary",
            "claim_boundary",
            "chronology",
        },
        "V2 result summary",
    )
    simulations = sum(int(row["simulations"]) for row in v2.SCENARIOS)
    expected_shapes = {
        "truth_units": (simulations, 17),
        "scenario_indices": (simulations,),
        "candidate_normalized_ranks": (simulations, 10),
        "reference_normalized_ranks": (simulations, 10),
        "candidate_coverage": (simulations, 10, 3),
        "reference_coverage": (simulations, 10, 3),
        "candidate_tie_counts": (simulations, 10),
        "reference_tie_mass": (simulations, 10),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise RuntimeError("V2 result array shape changed")
    if not np.array_equal(arrays["truth_units"], expected_truth_units(v2)):
        raise RuntimeError("V2 truth seed replay changed")
    if (
        summary["schema_version"] != v2.RESULT_SCHEMA
        or summary["config_sha256"] != config["bound_sbc_artifacts"]["v2_config"]["file_sha256"]
    ):
        raise RuntimeError("V2 result identity changed")
    for row in summary["candidate_fit_summaries"]:
        validate_v2_candidate_fit(row)
    for row in summary["independent_reference_summaries"]:
        validate_v2_reference(row)
    levels = list(map(float, v2.RANK_PROTOCOL["coverage_levels"]))
    tolerances = v2.GATES["coverage_tolerances_for_32_simulations"]
    scenario_rows = []
    expected_indices = []
    for scenario_index, scenario in enumerate(v2.SCENARIOS):
        expected_indices.extend([scenario_index] * int(scenario["simulations"]))
        selected = arrays["scenario_indices"] == scenario_index
        candidate_fits = [
            row
            for row in summary["candidate_fit_summaries"]
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        reference_fits = [
            row
            for row in summary["independent_reference_summaries"]
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        expected_sequence = list(range(int(scenario["simulations"])))
        if (
            len(candidate_fits) != int(scenario["simulations"])
            or len(reference_fits) != int(scenario["simulations"])
            or [row["simulation_index"] for row in candidate_fits] != expected_sequence
            or [row["simulation_index"] for row in reference_fits] != expected_sequence
        ):
            raise RuntimeError("V2 fit scenario chronology changed")
        candidate_rank = rank_summary(
            arrays["candidate_normalized_ranks"][selected],
            int(v2.RANK_PROTOCOL["rank_histogram_bins"]),
        )
        reference_rank = rank_summary(
            arrays["reference_normalized_ranks"][selected],
            int(v2.RANK_PROTOCOL["rank_histogram_bins"]),
        )
        candidate_coverage = coverage_summary(arrays["candidate_coverage"][selected], levels)
        reference_coverage = coverage_summary(arrays["reference_coverage"][selected], levels)
        difference = float(
            np.max(
                np.abs(
                    np.mean(arrays["candidate_normalized_ranks"][selected], axis=0)
                    - np.mean(arrays["reference_normalized_ranks"][selected], axis=0)
                )
            )
        )
        valid_fraction = float(
            np.mean([row["all_coordinates_diagnostic_valid"] for row in candidate_fits])
        )
        finite_rhats = [
            float(row["maximum_rhat"]) for row in candidate_fits if row["maximum_rhat"] is not None
        ]
        candidate_coverage_passed = all(
            candidate_coverage["maximum_absolute_error_by_level"][index]
            <= float(tolerances[str(level)])
            for index, level in enumerate(levels)
        )
        reference_coverage_passed = all(
            reference_coverage["maximum_absolute_error_by_level"][index]
            <= float(tolerances[str(level)])
            for index, level in enumerate(levels)
        )
        scenario_passed = bool(
            candidate_rank["maximum_absolute_mean_rank_z"]
            <= float(v2.GATES["maximum_absolute_candidate_mean_rank_z"])
            and reference_rank["maximum_absolute_mean_rank_z"]
            <= float(v2.GATES["maximum_absolute_reference_mean_rank_z"])
            and difference
            <= float(v2.GATES["maximum_absolute_candidate_reference_mean_rank_difference"])
            and candidate_coverage_passed
            and reference_coverage_passed
            and valid_fraction
            >= float(v2.GATES["minimum_fraction_fits_all_coordinates_diagnostic_valid"])
            and len(finite_rhats) == len(candidate_fits)
            and max(finite_rhats) <= float(v2.GATES["maximum_rank_normalized_split_rhat"])
            and min(row["minimum_bulk_ess"] for row in candidate_fits)
            >= float(v2.GATES["minimum_bulk_effective_samples_per_valid_coordinate"])
            and min(row["minimum_tail_ess"] for row in candidate_fits)
            >= float(v2.GATES["minimum_tail_effective_samples_per_valid_coordinate"])
            and min(row["minimum_stage_conditional_ess_fraction"] for row in reference_fits)
            >= float(v2.GATES["reference_minimum_stage_conditional_ess_fraction"])
            and max(row["maximum_standardized_replicate_mean_difference"] for row in reference_fits)
            <= float(v2.GATES["reference_maximum_standardized_replicate_mean_difference"])
            and min(row["minimum_final_unique_initial_ancestor_fraction"] for row in reference_fits)
            >= float(v2.GATES["reference_minimum_unique_initial_ancestor_fraction"])
            and all(row["all_replicates_reached_beta_one"] for row in reference_fits)
        )
        scenario_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "simulations": int(np.count_nonzero(selected)),
                "candidate_rank": candidate_rank,
                "reference_rank": reference_rank,
                "candidate_coverage": candidate_coverage,
                "reference_coverage": reference_coverage,
                "maximum_absolute_candidate_reference_mean_rank_difference": difference,
                "fraction_fits_all_coordinates_diagnostic_valid": valid_fraction,
                "maximum_fit_rhat": max(finite_rhats) if finite_rhats else None,
                "minimum_fit_bulk_ess": min(row["minimum_bulk_ess"] for row in candidate_fits),
                "minimum_fit_tail_ess": min(row["minimum_tail_ess"] for row in candidate_fits),
                "reference_minimum_stage_conditional_ess_fraction": min(
                    row["minimum_stage_conditional_ess_fraction"] for row in reference_fits
                ),
                "reference_maximum_standardized_replicate_mean_difference": max(
                    row["maximum_standardized_replicate_mean_difference"] for row in reference_fits
                ),
                "reference_minimum_unique_initial_ancestor_fraction": min(
                    row["minimum_final_unique_initial_ancestor_fraction"] for row in reference_fits
                ),
                "candidate_coverage_passed": candidate_coverage_passed,
                "reference_coverage_passed": reference_coverage_passed,
                "passed": scenario_passed,
            }
        )
    if not np.array_equal(arrays["scenario_indices"], np.asarray(expected_indices)):
        raise RuntimeError("V2 scenario index ordering changed")
    if summary["scenario_results"] != scenario_rows:
        raise RuntimeError("V2 stored scenario adjudication changed")
    orbit_passed = validate_orbit(
        summary["orbit_invariance_control"],
        float(v2.GATES["maximum_orbit_composite_difference"]),
        "V2 orbit control",
    )
    passed = bool(all(row["passed"] for row in scenario_rows) and orbit_passed)
    decision = (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V2_PASSED_NOT_PHYSICS_OR_PRODUCTION"
        if passed
        else "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V2_FAILED_RESULT_RETAINED"
    )
    candidate_calls = sum(
        row["initial_likelihoods_recomputed_fresh"] + row["active_evaluated"]
        for row in summary["candidate_fit_summaries"]
    )
    reference_calls = sum(row["calls"] for row in summary["independent_reference_summaries"])
    calls = {
        "actual_candidate_synthetic_likelihood_evaluations": candidate_calls,
        "actual_independent_reference_synthetic_likelihood_evaluations": reference_calls,
        "actual_total_synthetic_likelihood_evaluations": candidate_calls + reference_calls,
        "frozen_maximum_total_synthetic_likelihood_evaluations": config["v2_call_accounting"][
            "maximum_total_synthetic_likelihood_evaluations"
        ],
        "real_forward_model_evaluations": 0,
    }
    if (
        summary["v1_diagnosis"]
        != {
            "short_chain_mixing_failure": True,
            "finite_sobol_importance_collapse": True,
            "v1_result_or_thresholds_changed": False,
        }
        or summary["passed"] is not passed
        or summary["decision"] != decision
        or summary["call_accounting"] != calls
        or calls["actual_total_synthetic_likelihood_evaluations"]
        > calls["frozen_maximum_total_synthetic_likelihood_evaluations"]
        or summary["data_boundary"] != v2.DATA_BOUNDARY
        or summary["claim_boundary"] != v2.CLAIM_BOUNDARY
        or summary["chronology"] != config["v2_chronology"]
    ):
        raise RuntimeError("V2 result decision, calls, or boundary changed")
    return summary


def expected_v1_receipt(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["bound_sbc_artifacts"]
    body = {
        "schema_version": v1.RECEIPT_SCHEMA,
        "status": (
            "bounded_synthetic_sbc_passed_not_candidate_production"
            if summary["passed"]
            else "bounded_synthetic_sbc_failed_result_retained"
        ),
        "decision": summary["decision"],
        "evidence": {
            "config": artifacts["v1_config"],
            "implementation_source": artifacts["v1_source"],
            "canonical_sampler": config["v1_canonical_sampler_binding"],
            "bounded_result": artifacts["v1_result"],
        },
        "counts": summary["call_accounting"],
        "scenario_results": summary["scenario_summaries"],
        "controls": {
            "orbit_invariance": summary["orbit_invariance_control"],
            "importance_reference_present": True,
            "randomized_stellar_clipping_tie_ranks": True,
        },
        "data_boundary": v1.DATA_BOUNDARY,
        "claim_boundary": v1.CLAIM_BOUNDARY,
        "limitations": [
            "This calibrates a synthetic Gaussian likelihood in the exact ten-dimensional nuisance quotient, not the real cluster forward model.",
            "Forty-eight simulations provide a bounded control, not publication-grade SBC precision.",
            "The importance reference is a finite scrambled-Sobol approximation, not an analytic posterior.",
            "Passing cannot support the candidate physics, complete CP5.7-CP5.10, or authorize candidate production.",
        ],
        "replay": {
            "check": (
                "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_sbc "
                "check --config configs/gravity_cluster_nuisance_quotient_sbc_v1.json "
                "--expected-config-sha256 "
                f"{artifacts['v1_config']['file_sha256']} --receipt "
                "runs/gravity/publication-readiness/nuisance-quotient-sbc-v1.json"
            ),
            "candidate_production": "NOT_RUN",
        },
    }
    body["content_sha256"] = content_sha256(body)
    return body


def expected_v2_receipt(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["bound_sbc_artifacts"]
    body = {
        "schema_version": v2.RECEIPT_SCHEMA,
        "status": (
            "bounded_synthetic_sbc_v2_passed_not_candidate_production"
            if summary["passed"]
            else "bounded_synthetic_sbc_v2_failed_result_retained"
        ),
        "decision": summary["decision"],
        "evidence": {
            "config": artifacts["v2_config"],
            "implementation_source": artifacts["v2_source"],
            "canonical_sampler": config["v2_canonical_sampler_binding"],
            "v1_receipt": artifacts["v1_receipt"],
            "v1_result": artifacts["v1_result"],
            "bounded_v2_result": artifacts["v2_result"],
        },
        "v1_diagnosis": summary["v1_diagnosis"],
        "principled_v2_changes": {
            "candidate_chain_lengthening": v2.CANDIDATE_INFERENCE["change_from_v1"],
            "independent_reference_replacement": v2.REFERENCE["change_from_v1"],
            "finite_simulation_threshold_basis": {
                "simulations_per_scenario": 32,
                "familywise_z": v2.GATES["finite_simulation_familywise_z"],
                "continuity_allowance": v2.GATES["finite_simulation_continuity_allowance"],
                "coverage_tolerances": v2.GATES["coverage_tolerances_for_32_simulations"],
            },
        },
        "counts": summary["call_accounting"],
        "scenario_results": summary["scenario_results"],
        "controls": {
            "orbit_invariance": summary["orbit_invariance_control"],
            "reference_algorithm": v2.REFERENCE["algorithm_id"],
            "reference_uses_candidate_transition": False,
            "reference_uses_candidate_orbits": False,
        },
        "data_boundary": v2.DATA_BOUNDARY,
        "claim_boundary": v2.CLAIM_BOUNDARY,
        "limitations": [
            "V2 still calibrates a synthetic Gaussian likelihood in the exact nuisance quotient, not the cluster forward model.",
            "Sixty-four simulations improve but do not provide publication-grade calibration precision.",
            "The independent reference is finite-particle adaptive-tempering SMC and is judged by replicate agreement, ancestry, and conditional ESS gates rather than analytic exactness.",
            "A pass cannot support candidate physics, complete CP5.7-CP5.10, or authorize production; a failure is retained without threshold repair.",
        ],
        "replay": {
            "check": (
                "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_sbc_v2 "
                "check --config configs/gravity_cluster_nuisance_quotient_sbc_v2.json "
                "--expected-config-sha256 "
                f"{artifacts['v2_config']['file_sha256']} --receipt "
                "runs/gravity/publication-readiness/nuisance-quotient-sbc-v2.json"
            ),
            "candidate_production": "NOT_RUN",
        },
    }
    body["content_sha256"] = content_sha256(body)
    return body


def require_exact_receipt(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    strict_keys(actual, set(expected), label)
    if actual != expected:
        raise RuntimeError(f"{label} does not exactly reconstruct from sealed result")


def adjudicate(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["bound_sbc_artifacts"]
    v1_summary = validate_v1_result(ROOT / artifacts["v1_result"]["path"], config)
    v2_summary = validate_v2_result(ROOT / artifacts["v2_result"]["path"], config)
    v1_receipt = json.loads((ROOT / artifacts["v1_receipt"]["path"]).read_text(encoding="utf-8"))
    v2_receipt = json.loads((ROOT / artifacts["v2_receipt"]["path"]).read_text(encoding="utf-8"))
    require_exact_receipt(v1_receipt, expected_v1_receipt(config, v1_summary), "V1 receipt")
    require_exact_receipt(v2_receipt, expected_v2_receipt(config, v2_summary), "V2 receipt")
    if v1_summary["passed"] is not False or v2_summary["passed"] is not False:
        raise RuntimeError("frozen adjudicator is only valid for the two retained failures")
    return {
        "v1": {
            "artifact_valid": True,
            "passed": False,
            "decision": v1_summary["decision"],
            "synthetic_likelihood_evaluations": v1_summary["call_accounting"][
                "actual_total_synthetic_likelihood_evaluations"
            ],
        },
        "v2": {
            "artifact_valid": True,
            "passed": False,
            "decision": v2_summary["decision"],
            "synthetic_likelihood_evaluations": v2_summary["call_accounting"][
                "actual_total_synthetic_likelihood_evaluations"
            ],
        },
        "both_failed": True,
        "candidate_production_unlock": False,
        "newtonian_control_unlock": False,
    }


def expected_verifier_receipt(config: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": STATUS,
        "decision": DECISION,
        "evidence": {
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "implementation_source": artifact_binding(ROOT / config["implementation_source"]),
            "verifier_test": config["verifier_test"],
            "bound_sbc_artifacts": config["bound_sbc_artifacts"],
        },
        "adjudication": adjudicate(config),
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    body["content_sha256"] = content_sha256(body)
    return body


def write_receipt(config_path: Path, expected_config_sha256: str, output: Path) -> dict[str, Any]:
    if confined(config_path) != ROOT / CONFIG_PATH:
        raise RuntimeError("strict SBC adjudicator config path changed")
    if confined(output) != ROOT / RECEIPT_PATH:
        raise RuntimeError("strict SBC adjudicator receipt path changed")
    config = load_config(config_path, expected_config_sha256)
    body = expected_verifier_receipt(config)
    v1.sampler.write_json(output, body)
    return body


def check_receipt(
    config_path: Path, expected_config_sha256: str, receipt_path: Path
) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    target = confined(receipt_path)
    if target != ROOT / RECEIPT_PATH:
        raise RuntimeError("strict SBC adjudicator receipt path changed")
    actual = json.loads(target.read_text(encoding="utf-8"))
    expected = expected_verifier_receipt(config)
    require_exact_receipt(actual, expected, "strict SBC adjudicator receipt")
    return {
        "valid": True,
        "passed": False,
        "decision": DECISION,
        "v1_passed": False,
        "v2_passed": False,
        "candidate_production_unlock": False,
        "newtonian_control_unlock": False,
        "real_rows_loaded": 0,
        "receipt_sha256": file_sha256(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-receipt")
    write.add_argument("--config", type=Path, required=True)
    write.add_argument("--expected-config-sha256", required=True)
    write.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--expected-config-sha256", required=True)
    check.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "write-receipt":
        result = write_receipt(args.config, args.expected_config_sha256, args.output)
    else:
        result = check_receipt(args.config, args.expected_config_sha256, args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
