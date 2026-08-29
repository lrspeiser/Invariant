from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_adjudicator as predecessor
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v2 as v2
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v3 as v3
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_sbc_v3_adjudicator_v1.json")
RECEIPT_PATH = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sbc-v3-adjudicator-v1.json"
)
TEST_PATH = Path("tests/test_gravity_cluster_nuisance_quotient_sbc_v3_adjudicator.py")

CONFIG_SCHEMA = "invariant-gravity-nuisance-quotient-sbc-v3-adjudicator-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-nuisance-quotient-sbc-v3-adjudicator-receipt-1.0"
STATUS = "strictly_verified_v3_synthetic_pass_newtonian_eligible_production_locked"
DECISION = (
    "V3_SYNTHETIC_SBC_PASSED_NEWTONIAN_CONTROL_MAY_UNLOCK_CANDIDATE_PRODUCTION_REMAINS_LOCKED"
)
MACHINE_STATEMENT = (
    "V3 synthetic SBC passed; Newtonian-control may unlock; candidate production remains locked"
)

EXPECTED_V3_ARTIFACTS = {
    "v3_source": {
        "path": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sbc_v3.py",
        "file_sha256": "cb7c6621eaa66372c5aeebd02adcd7e28f374fbed7daabe3220a952724ea6a7f",
    },
    "v3_config": {
        "path": "configs/gravity_cluster_nuisance_quotient_sbc_v3.json",
        "file_sha256": "781278eab67272eaec76cece10c3789b22650bec025dacc0296399dc2868cbd3",
    },
    "v3_result": {
        "path": (
            "runs/gravity/publication-readiness/nuisance-quotient-sbc-v3/"
            "bounded-synthetic-sbc-v3.npz"
        ),
        "file_sha256": "472fe23a3ea8660f0d3e7b23e9da692fb595acb9e0e6049ff5a53c23485a7f92",
    },
    "v3_receipt": {
        "path": "runs/gravity/publication-readiness/nuisance-quotient-sbc-v3.json",
        "file_sha256": "2111cd0dec50db4fea67faaebd2705a345f7d94fe4ab9c8d2cd237a857395298",
    },
    "v3_test": {
        "path": "tests/test_gravity_cluster_nuisance_quotient_sbc_v3.py",
        "file_sha256": "e17fd1ad893cda0cf6a0c312259a8738b310688bfa7aae995e68b45845b52681",
    },
}

EXPECTED_PREDECESSOR_ADJUDICATOR = {
    "source": {
        "path": ("src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sbc_adjudicator.py"),
        "file_sha256": "b92f41560f06df44625a5439933dfcdacd0caa0c50b47cc8f631e3953625f94b",
    },
    "config": {
        "path": "configs/gravity_cluster_nuisance_quotient_sbc_adjudicator_v1.json",
        "file_sha256": "5f65214efc2d488ebe80aafdb104b2cae12dc2689a39a6c5888a707f9475608a",
    },
    "test": {
        "path": "tests/test_gravity_cluster_nuisance_quotient_sbc_adjudicator.py",
        "file_sha256": "1bb2a805cd86a84fd865d075ea5e6ede69457bde6e4c5a4d92a204655b885916",
    },
    "receipt": {
        "path": ("runs/gravity/publication-readiness/nuisance-quotient-sbc-adjudicator-v1.json"),
        "file_sha256": "6de931dfc930f306327eb56b21034a24520ecc96325e062c1f4033c7a28df3ea",
    },
}

EXPECTED_VERIFIER_TEST_SHA256 = "b66c1c0a6db7a3be7c5937c559087fcdff6fd585d832c8aa268728f69daa4ec7"

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
    "v1_passed": False,
    "v2_passed": False,
    "v3_synthetic_sbc_passed": True,
    "newtonian_control_may_unlock": True,
    "candidate_production_may_unlock": False,
    "candidate_physics_supported": False,
    "full_cluster_forward_model_calibrated": False,
    "scientific_claim_allowed": False,
}

DIAGNOSTIC_EVIDENCE_BOUNDARY = {
    "retained_chains_present_in_sealed_npz": False,
    "rank_and_coverage_arrays_present_and_independently_recomputed": True,
    "rhat_and_ess_recomputed_from_retained_chains": False,
    "rhat_and_ess_role": "sealed_summary_only",
    "diagnostic_implementation_bound_and_control_validated": True,
    "limitation": (
        "Retained primitive/composite chains were not stored in the V3 NPZ. "
        "R-hat and ESS can be checked for schema, finiteness, mechanics, "
        "aggregation, and frozen thresholds, but cannot be independently "
        "recomputed without a prohibited rerun."
    ),
}


def file_sha256(path: Path) -> str:
    return predecessor.file_sha256(path)


def normalized_sha256(path: Path) -> str:
    return predecessor.normalized_sha256(path)


def content_sha256(value: dict[str, Any]) -> str:
    return predecessor.content_sha256(value)


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    predecessor.strict_keys(value, expected, label)


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
    if target != ROOT / CONFIG_PATH:
        raise RuntimeError("strict V3 adjudicator config path changed")
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("strict V3 adjudicator config hash changed")
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
            "v3_artifacts",
            "predecessor_adjudicator",
            "predecessor_sbc_artifacts",
            "v3_dependency_policy",
            "data_boundary",
            "claim_boundary",
            "diagnostic_evidence_boundary",
            "receipt_path",
        },
        "strict V3 adjudicator config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_append_only_strict_v3_adjudication"
        or config["v3_artifacts"] != EXPECTED_V3_ARTIFACTS
        or config["predecessor_adjudicator"] != EXPECTED_PREDECESSOR_ADJUDICATOR
        or config["predecessor_sbc_artifacts"] != predecessor.EXPECTED_SBC_ARTIFACTS
        or config["v3_dependency_policy"]
        != "validate_hash_bound_v3_config_and_every_frozen_evidence_binding"
        or config["verifier_test"]
        != {"path": TEST_PATH.as_posix(), "file_sha256": EXPECTED_VERIFIER_TEST_SHA256}
        or config["data_boundary"] != DATA_BOUNDARY
        or config["claim_boundary"] != CLAIM_BOUNDARY
        or config["diagnostic_evidence_boundary"] != DIAGNOSTIC_EVIDENCE_BOUNDARY
        or config["receipt_path"] != RECEIPT_PATH.as_posix()
    ):
        raise RuntimeError("strict V3 adjudicator frozen contract changed")
    source = confined(ROOT / config["implementation_source"])
    if (
        source != Path(__file__).resolve()
        or normalized_sha256(source) != config["implementation_source_normalized_sha256"]
    ):
        raise RuntimeError("strict V3 adjudicator source changed")
    validate_binding(config["verifier_test"], "verifier_test")
    for name, binding in config["v3_artifacts"].items():
        validate_binding(binding, f"v3_artifacts.{name}")
    for name, binding in config["predecessor_adjudicator"].items():
        validate_binding(binding, f"predecessor_adjudicator.{name}")
    for name, binding in config["predecessor_sbc_artifacts"].items():
        validate_binding(binding, f"predecessor_sbc_artifacts.{name}")
    bound_v3_config = config["v3_artifacts"]["v3_config"]
    v3.load_config(ROOT / bound_v3_config["path"], bound_v3_config["file_sha256"])
    predecessor_config = config["predecessor_adjudicator"]["config"]
    predecessor_receipt = config["predecessor_adjudicator"]["receipt"]
    checked = predecessor.check_receipt(
        ROOT / predecessor_config["path"],
        predecessor_config["file_sha256"],
        ROOT / predecessor_receipt["path"],
    )
    if (
        checked["valid"] is not True
        or checked["v1_passed"] is not False
        or checked["v2_passed"] is not False
        or checked["candidate_production_unlock"] is not False
        or checked["newtonian_control_unlock"] is not False
    ):
        raise RuntimeError("predecessor strict V1/V2 adjudication changed")
    config["_config_sha256"] = expected_sha256
    return config


def read_npz(path: Path, expected_keys: set[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys | {"summary"}:
            raise RuntimeError("V3 result NPZ keys changed")
        arrays = {name: archive[name].copy() for name in expected_keys}
        summary = json.loads(str(archive["summary"].item()))
    return arrays, summary


def expected_truth_units() -> tuple[np.ndarray, np.ndarray]:
    truths = []
    scenarios = []
    for scenario_index, scenario in enumerate(v3.SCENARIOS):
        for simulation_index in range(int(scenario["simulations"])):
            seed_offset = scenario_index * int(
                v3.SEED_LINEAGE["scenario_stride"]
            ) + simulation_index * int(v3.SEED_LINEAGE["simulation_stride"])
            truths.append(
                np.random.default_rng(int(v3.SEED_LINEAGE["truth_base"]) + seed_offset).random(17)
            )
            scenarios.append(scenario_index)
    return np.asarray(truths), np.asarray(scenarios, dtype=int)


def validate_candidate_fit(row: dict[str, Any]) -> None:
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
            "transport_attempted",
            "transport_evaluated",
            "transport_accepted",
            "transport_acceptance",
            "transport_out_of_bounds_self_loops",
            "transport_by_block",
            "ending_betas_by_replicate",
            "orbit_attempted",
            "orbit_accepted",
            "initial_likelihoods_recomputed_fresh",
        },
        "V3 candidate fit",
    )
    chains = int(v3.CANDIDATE_INFERENCE["replicates"]) * int(
        v3.CANDIDATE_INFERENCE["particles_per_replicate"]
    )
    sweeps = sum(
        int(v3.CANDIDATE_INFERENCE[name])
        for name in ("adaptation_sweeps", "fixed_kernel_settling_sweeps", "retained_sweeps")
    )
    per_block = chains * sweeps
    block_ids = {str(block["block_id"]) for block in v3.TRANSPORT_BLOCKS}
    strict_keys(row["transport_by_block"], block_ids, "V3 candidate transport blocks")
    for block_id, block_row in row["transport_by_block"].items():
        strict_keys(
            block_row,
            {"attempted", "evaluated", "accepted", "out_of_bounds_rejected"},
            f"V3 candidate transport block {block_id}",
        )
        if (
            block_row["attempted"] != per_block
            or block_row["evaluated"] != per_block
            or not 0 <= block_row["accepted"] <= per_block
            or block_row["out_of_bounds_rejected"] != 0
        ):
            raise RuntimeError("V3 candidate block mechanics changed")
    attempted = per_block * len(v3.TRANSPORT_BLOCKS)
    accepted = sum(block["accepted"] for block in row["transport_by_block"].values())
    beta_rows = row["ending_betas_by_replicate"]
    low, high = map(float, v3.CANDIDATE_INFERENCE["beta_bounds"])
    if (
        row["all_coordinates_diagnostic_valid"] is not True
        or row["valid_coordinate_count"] != len(v3.sampler.COMPOSITES)
        or not isinstance(row["maximum_rhat"], (float, int))
        or not math.isfinite(float(row["maximum_rhat"]))
        or not math.isfinite(float(row["minimum_bulk_ess"]))
        or not math.isfinite(float(row["minimum_tail_ess"]))
        or row["minimum_bulk_ess"] <= 0.0
        or row["minimum_tail_ess"] <= 0.0
        or row["transport_attempted"] != attempted
        or row["transport_evaluated"] != attempted
        or row["transport_accepted"] != accepted
        or row["transport_acceptance"] != accepted / attempted
        or row["transport_out_of_bounds_self_loops"] != 0
        or len(beta_rows) != int(v3.CANDIDATE_INFERENCE["replicates"])
        or any(len(beta_row) != len(v3.TRANSPORT_BLOCKS) for beta_row in beta_rows)
        or any(not low <= float(beta) <= high for beta_row in beta_rows for beta in beta_row)
        or row["orbit_attempted"] != len(v3.sampler.ORBIT_NAMES) * chains * sweeps
        or not 0 <= row["orbit_accepted"] <= row["orbit_attempted"]
        or row["initial_likelihoods_recomputed_fresh"] != chains
    ):
        raise RuntimeError("V3 candidate fit mechanics or sealed diagnostics changed")


def scenario_summaries(
    arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> list[dict[str, Any]]:
    levels = list(map(float, v3.RANK_PROTOCOL["coverage_levels"]))
    tolerances = v3.GATES["coverage_tolerances_for_32_simulations"]
    results = []
    for scenario_index, scenario in enumerate(v3.SCENARIOS):
        selected = arrays["scenario_indices"] == scenario_index
        candidate_rank = predecessor.rank_summary(
            arrays["candidate_normalized_ranks"][selected],
            int(v3.RANK_PROTOCOL["rank_histogram_bins"]),
        )
        reference_rank = predecessor.rank_summary(
            arrays["reference_normalized_ranks"][selected],
            int(v3.RANK_PROTOCOL["rank_histogram_bins"]),
        )
        candidate_coverage = predecessor.coverage_summary(
            arrays["candidate_coverage"][selected], levels
        )
        reference_coverage = predecessor.coverage_summary(
            arrays["reference_coverage"][selected], levels
        )
        mean_difference = float(
            np.max(
                np.abs(
                    np.mean(arrays["candidate_normalized_ranks"][selected], axis=0)
                    - np.mean(arrays["reference_normalized_ranks"][selected], axis=0)
                )
            )
        )
        candidate_rows = [
            row
            for row in summary["candidate_fit_summaries"]
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        reference_rows = [
            row
            for row in summary["independent_reference_summaries"]
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        if (
            len(candidate_rows) != int(scenario["simulations"])
            or len(reference_rows) != int(scenario["simulations"])
            or [row["simulation_index"] for row in candidate_rows]
            != list(range(int(scenario["simulations"])))
            or [row["simulation_index"] for row in reference_rows]
            != list(range(int(scenario["simulations"])))
        ):
            raise RuntimeError("V3 scenario fit chronology changed")
        valid_fraction = float(
            np.mean([row["all_coordinates_diagnostic_valid"] for row in candidate_rows])
        )
        finite_rhats = [float(row["maximum_rhat"]) for row in candidate_rows]
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
        passed = bool(
            candidate_rank["maximum_absolute_mean_rank_z"]
            <= float(v3.GATES["maximum_absolute_candidate_mean_rank_z"])
            and reference_rank["maximum_absolute_mean_rank_z"]
            <= float(v3.GATES["maximum_absolute_reference_mean_rank_z"])
            and mean_difference
            <= float(v3.GATES["maximum_absolute_candidate_reference_mean_rank_difference"])
            and candidate_coverage_passed
            and reference_coverage_passed
            and valid_fraction
            >= float(v3.GATES["minimum_fraction_fits_all_coordinates_diagnostic_valid"])
            and max(finite_rhats) <= float(v3.GATES["maximum_rank_normalized_split_rhat"])
            and min(row["minimum_bulk_ess"] for row in candidate_rows)
            >= float(v3.GATES["minimum_bulk_effective_samples_per_valid_coordinate"])
            and min(row["minimum_tail_ess"] for row in candidate_rows)
            >= float(v3.GATES["minimum_tail_effective_samples_per_valid_coordinate"])
            and min(row["minimum_stage_conditional_ess_fraction"] for row in reference_rows)
            >= float(v3.GATES["reference_minimum_stage_conditional_ess_fraction"])
            and max(row["maximum_standardized_replicate_mean_difference"] for row in reference_rows)
            <= float(v3.GATES["reference_maximum_standardized_replicate_mean_difference"])
            and min(row["minimum_final_unique_initial_ancestor_fraction"] for row in reference_rows)
            >= float(v3.GATES["reference_minimum_unique_initial_ancestor_fraction"])
            and all(row["all_replicates_reached_beta_one"] for row in reference_rows)
        )
        results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "simulations": int(np.count_nonzero(selected)),
                "candidate_rank": candidate_rank,
                "reference_rank": reference_rank,
                "candidate_coverage": candidate_coverage,
                "reference_coverage": reference_coverage,
                "maximum_absolute_candidate_reference_mean_rank_difference": mean_difference,
                "fraction_fits_all_coordinates_diagnostic_valid": valid_fraction,
                "maximum_fit_rhat": max(finite_rhats),
                "minimum_fit_bulk_ess": min(row["minimum_bulk_ess"] for row in candidate_rows),
                "minimum_fit_tail_ess": min(row["minimum_tail_ess"] for row in candidate_rows),
                "reference_minimum_stage_conditional_ess_fraction": min(
                    row["minimum_stage_conditional_ess_fraction"] for row in reference_rows
                ),
                "reference_maximum_standardized_replicate_mean_difference": max(
                    row["maximum_standardized_replicate_mean_difference"] for row in reference_rows
                ),
                "reference_minimum_unique_initial_ancestor_fraction": min(
                    row["minimum_final_unique_initial_ancestor_fraction"] for row in reference_rows
                ),
                "candidate_coverage_passed": candidate_coverage_passed,
                "reference_coverage_passed": reference_coverage_passed,
                "passed": passed,
            }
        )
    return results


def validate_v3_result(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    binding = config["v3_artifacts"]["v3_result"]
    target = confined(path)
    if target != ROOT / binding["path"] or file_sha256(target) != binding["file_sha256"]:
        raise RuntimeError("V3 result binding changed or swapped")
    names = {
        "truth_units",
        "scenario_indices",
        "candidate_normalized_ranks",
        "reference_normalized_ranks",
        "candidate_coverage",
        "reference_coverage",
        "candidate_tie_counts",
        "reference_tie_mass",
    }
    arrays, summary = read_npz(target, names)
    strict_keys(
        summary,
        {
            "schema_version",
            "decision",
            "config_sha256",
            "passed",
            "v2_isolated_failure",
            "structural_kernel_change",
            "scenario_results",
            "candidate_fit_summaries",
            "independent_reference_summaries",
            "kernel_controls",
            "call_accounting",
            "data_boundary",
            "claim_boundary",
            "chronology",
        },
        "V3 result summary",
    )
    simulations = sum(int(row["simulations"]) for row in v3.SCENARIOS)
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
        raise RuntimeError("V3 result array shape changed")
    if (
        not np.all(np.isfinite(arrays["truth_units"]))
        or not np.all((arrays["truth_units"] > 0.0) & (arrays["truth_units"] < 1.0))
        or not np.all(np.isfinite(arrays["candidate_normalized_ranks"]))
        or not np.all(
            (arrays["candidate_normalized_ranks"] > 0.0)
            & (arrays["candidate_normalized_ranks"] < 1.0)
        )
        or not np.all(np.isfinite(arrays["reference_normalized_ranks"]))
        or not np.all(
            (arrays["reference_normalized_ranks"] >= 0.0)
            & (arrays["reference_normalized_ranks"] <= 1.0)
        )
        or arrays["candidate_coverage"].dtype != np.dtype(bool)
        or arrays["reference_coverage"].dtype != np.dtype(bool)
        or not np.all(np.isfinite(arrays["candidate_tie_counts"]))
        or not np.all(arrays["candidate_tie_counts"] >= 0)
        or not np.all(arrays["candidate_tie_counts"] <= 4096)
        or not np.all(arrays["candidate_tie_counts"] == np.floor(arrays["candidate_tie_counts"]))
        or not np.all(np.isfinite(arrays["reference_tie_mass"]))
        or not np.all((arrays["reference_tie_mass"] >= 0.0) & (arrays["reference_tie_mass"] <= 1.0))
    ):
        raise RuntimeError("V3 result array type, range, or finiteness changed")
    truth, scenario_indices = expected_truth_units()
    if not np.array_equal(arrays["truth_units"], truth):
        raise RuntimeError("V3 truth seed replay changed")
    if not np.array_equal(arrays["scenario_indices"], scenario_indices):
        raise RuntimeError("V3 scenario ordering changed")
    v2_binding = config["predecessor_sbc_artifacts"]["v2_result"]
    v2_arrays, v2_summary = predecessor.read_npz(
        ROOT / v2_binding["path"],
        {
            "truth_units",
            "scenario_indices",
            "candidate_normalized_ranks",
            "reference_normalized_ranks",
            "candidate_coverage",
            "reference_coverage",
            "candidate_tie_counts",
            "reference_tie_mass",
        },
    )
    predecessor_config_binding = config["predecessor_adjudicator"]["config"]
    predecessor_contract = predecessor.load_config(
        ROOT / predecessor_config_binding["path"],
        predecessor_config_binding["file_sha256"],
    )
    predecessor.validate_v2_result(ROOT / v2_binding["path"], predecessor_contract)
    for name in (
        "truth_units",
        "scenario_indices",
        "reference_normalized_ranks",
        "reference_coverage",
        "reference_tie_mass",
    ):
        if not np.array_equal(arrays[name], v2_arrays[name]):
            raise RuntimeError(f"V3 paired V2 reference array changed: {name}")
    if summary["independent_reference_summaries"] != v2_summary["independent_reference_summaries"]:
        raise RuntimeError("V3 paired V2 reference summaries changed")
    for row in summary["candidate_fit_summaries"]:
        validate_candidate_fit(row)
    expected_scenarios = scenario_summaries(arrays, summary)
    if summary["scenario_results"] != expected_scenarios:
        raise RuntimeError("V3 scenario summaries do not reconstruct from sealed evidence")
    prior_config = uncertainty.load_config(ROOT)
    expected_controls = v3.kernel_controls(prior_config)
    if summary["kernel_controls"] != expected_controls or expected_controls["passed"] is not True:
        raise RuntimeError("V3 kernel controls do not replay exactly")
    chains = int(v3.CANDIDATE_INFERENCE["replicates"]) * int(
        v3.CANDIDATE_INFERENCE["particles_per_replicate"]
    )
    sweeps = sum(
        int(v3.CANDIDATE_INFERENCE[name])
        for name in ("adaptation_sweeps", "fixed_kernel_settling_sweeps", "retained_sweeps")
    )
    candidate_calls = simulations * (chains + chains * sweeps * len(v3.TRANSPORT_BLOCKS))
    reference_calls = sum(row["calls"] for row in summary["independent_reference_summaries"])
    actual_calls = candidate_calls + reference_calls
    expected_calls = {
        "actual_candidate_synthetic_likelihood_evaluations": candidate_calls,
        "actual_independent_reference_synthetic_likelihood_evaluations": reference_calls,
        "actual_total_synthetic_likelihood_evaluations": actual_calls,
        "frozen_maximum_total_synthetic_likelihood_evaluations": v3.maximum_call_accounting()[
            "maximum_total_synthetic_likelihood_evaluations"
        ],
        "real_forward_model_evaluations": 0,
    }
    passed = bool(all(row["passed"] for row in expected_scenarios) and expected_controls["passed"])
    decision = (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V3_PASSED_NOT_PHYSICS_OR_PRODUCTION"
        if passed
        else "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V3_FAILED_RESULT_RETAINED"
    )
    expected_v2_failure = {
        "independent_reference_calibration_passed": True,
        "candidate_maximum_rhat_exceeded_1p75": True,
        "candidate_minimum_bulk_ess_below_50": True,
        "v2_result_or_thresholds_changed": False,
    }
    expected_config_hash = config["v3_artifacts"]["v3_config"]["file_sha256"]
    v3_config = json.loads((ROOT / config["v3_artifacts"]["v3_config"]["path"]).read_text())
    if (
        summary["schema_version"] != v3.RESULT_SCHEMA
        or summary["config_sha256"] != expected_config_hash
        or summary["v2_isolated_failure"] != expected_v2_failure
        or summary["structural_kernel_change"]
        != v3.CANDIDATE_INFERENCE["structural_change_from_v2"]
        or summary["call_accounting"] != expected_calls
        or summary["data_boundary"] != v3.DATA_BOUNDARY
        or summary["claim_boundary"] != v3.CLAIM_BOUNDARY
        or summary["chronology"] != v3_config["chronology"]
        or summary["passed"] is not passed
        or summary["decision"] != decision
        or passed is not True
    ):
        raise RuntimeError("V3 result identity, calls, boundaries, or decision changed")
    return summary


def expected_v3_receipt(config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["v3_artifacts"]
    body = {
        "schema_version": v3.RECEIPT_SCHEMA,
        "status": "bounded_synthetic_sbc_v3_passed_not_candidate_production",
        "decision": summary["decision"],
        "evidence": {
            "config": artifacts["v3_config"],
            "implementation_source": artifacts["v3_source"],
            "v2_receipt": config["predecessor_sbc_artifacts"]["v2_receipt"],
            "v2_result": config["predecessor_sbc_artifacts"]["v2_result"],
            "canonical_sampler": {
                "path": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sampler.py",
                "file_sha256": "975b9f69a614d7d419dcc44ac340c86b27a85e6e9fed7ed63d6f6caff228abcb",
            },
            "quotient_audit_receipt": {
                "path": "runs/gravity/publication-readiness/nuisance-quotient-audit-v1.json",
                "file_sha256": "65dd1909e11a724e01b9614b1a0bb611d7a3c2f5baf8c54edba9bd27b229af08",
            },
            "bounded_v3_result": artifacts["v3_result"],
        },
        "v2_isolated_failure": summary["v2_isolated_failure"],
        "principled_v3_change": {
            "kernel": v3.CANDIDATE_INFERENCE["transport_kernel"],
            "blocks": v3.TRANSPORT_BLOCKS,
            "structural_change_from_v2": v3.CANDIDATE_INFERENCE["structural_change_from_v2"],
            "sweep_counts_unchanged_from_v2": True,
            "scientific_gates_unchanged_from_v2": v3.GATES == v2.GATES,
        },
        "counts": summary["call_accounting"],
        "scenario_results": summary["scenario_results"],
        "controls": summary["kernel_controls"],
        "data_boundary": v3.DATA_BOUNDARY,
        "claim_boundary": v3.CLAIM_BOUNDARY,
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
                "check --config configs/gravity_cluster_nuisance_quotient_sbc_v3.json "
                "--expected-config-sha256 "
                f"{artifacts['v3_config']['file_sha256']} --receipt "
                "runs/gravity/publication-readiness/nuisance-quotient-sbc-v3.json"
            ),
            "candidate_production": "NOT_RUN",
        },
    }
    body["content_sha256"] = content_sha256(body)
    return body


def require_exact_receipt(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    strict_keys(actual, set(expected), label)
    if actual != expected:
        raise RuntimeError(f"{label} does not exactly reconstruct from sealed V3 evidence")


def adjudicate(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = config["v3_artifacts"]
    summary = validate_v3_result(ROOT / artifacts["v3_result"]["path"], config)
    receipt = json.loads((ROOT / artifacts["v3_receipt"]["path"]).read_text(encoding="utf-8"))
    require_exact_receipt(receipt, expected_v3_receipt(config, summary), "V3 receipt")
    return {
        "predecessor_v1_v2_adjudication_valid": True,
        "v1_passed": False,
        "v2_passed": False,
        "v3_artifact_valid": True,
        "v3_synthetic_sbc_passed": True,
        "v3_synthetic_likelihood_evaluations": summary["call_accounting"][
            "actual_total_synthetic_likelihood_evaluations"
        ],
        "newtonian_control_unlock": True,
        "candidate_production_unlock": False,
        "diagnostics_sealed_summary_only": True,
    }


def expected_verifier_receipt(config: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": STATUS,
        "decision": DECISION,
        "machine_statement": MACHINE_STATEMENT,
        "evidence": {
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "implementation_source": artifact_binding(ROOT / config["implementation_source"]),
            "verifier_test": config["verifier_test"],
            "v3_artifacts": config["v3_artifacts"],
            "predecessor_adjudicator": config["predecessor_adjudicator"],
            "predecessor_sbc_artifacts": config["predecessor_sbc_artifacts"],
        },
        "adjudication": adjudicate(config),
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "diagnostic_evidence_boundary": DIAGNOSTIC_EVIDENCE_BOUNDARY,
    }
    body["content_sha256"] = content_sha256(body)
    return body


def write_receipt(config_path: Path, expected_config_sha256: str, output: Path) -> dict[str, Any]:
    if confined(output) != ROOT / RECEIPT_PATH:
        raise RuntimeError("strict V3 adjudicator receipt path changed")
    config = load_config(config_path, expected_config_sha256)
    body = expected_verifier_receipt(config)
    v3.sampler.write_json(output, body)
    return body


def check_receipt(
    config_path: Path, expected_config_sha256: str, receipt_path: Path
) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    target = confined(receipt_path)
    if target != ROOT / RECEIPT_PATH:
        raise RuntimeError("strict V3 adjudicator receipt path changed")
    actual = json.loads(target.read_text(encoding="utf-8"))
    expected = expected_verifier_receipt(config)
    require_exact_receipt(actual, expected, "strict V3 adjudicator receipt")
    return {
        "valid": True,
        "passed": True,
        "decision": DECISION,
        "machine_statement": MACHINE_STATEMENT,
        "v1_passed": False,
        "v2_passed": False,
        "v3_synthetic_sbc_passed": True,
        "newtonian_control_unlock": True,
        "candidate_production_unlock": False,
        "scientific_claim_allowed": False,
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
