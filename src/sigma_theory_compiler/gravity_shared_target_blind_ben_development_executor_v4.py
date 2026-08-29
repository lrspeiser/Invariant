"""Authorization-gated development scoring for the frozen 60-class B+E+N registry.

The ``preflight`` command is metadata-only.  The ``execute`` command validates an exact
authorization and writes an immutable access-intent record before it opens one development
payload.  An access intent cannot be replayed: an interrupted run requires a successor
contract rather than silently opening the data twice.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from fractions import Fraction
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from sigma_theory_compiler.gravity_shared_target_blind_ben_synthetic_execution import (
    _find_component,
    evaluate_ast,
    expression,
    formula_ast,
    normalize_ast,
    sha256_value,
    validate_ast,
    validate_registry,
)
from sigma_theory_compiler.real_data_gravity_confrontation import Galaxy
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sparc_full_sample import (
    CONFIRMATION_FRACTION,
    FULL_SPLIT_RULE,
    FULL_SPLIT_SALT,
    _decimal,
    admit,
    declare_split,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_shared_target_blind_ben_development_executor_v4.json")
TEST_PATH = Path("tests/test_gravity_shared_target_blind_ben_development_executor_v4.py")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_shared_target_blind_ben_development_executor_v4.py"
)
PREFLIGHT_PATH = Path("runs/gravity/shared-target-blind-ben-development-executor-v4-preflight.json")
AUTHORIZATION_PATH = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/"
    "authorization-current-unauthorized.json"
)
ACCESS_INTENT_PATH = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/access-intent.json"
)
RESULT_PATH = Path("runs/gravity/shared-target-blind-ben-development-executor-v4/result.json")
ADJUDICATION_PATH = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/result-adjudication.json"
)
ACCESS_FAILURE_PATH = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/access-failure.json"
)
PHASE_RECEIPT_DIRECTORY = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/phases"
)
TERMINAL_SUCCESS_PATH = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/terminal-success.json"
)
TERMINAL_STATE_LOCK_PATH = Path(
    "runs/gravity/shared-target-blind-ben-development-executor-v4/terminal-state.lock"
)

CONFIG_SCHEMA = "invariant-gravity-ben-development-executor-config-4.0"
PREFLIGHT_SCHEMA = "invariant-gravity-ben-development-executor-preflight-4.0"
AUTHORIZATION_SCHEMA = "invariant-gravity-ben-development-executor-authorization-4.0"
ACCESS_SCHEMA = "invariant-gravity-ben-development-executor-access-intent-4.0"
RESULT_SCHEMA = "invariant-gravity-ben-development-score-result-4.0"
ADJUDICATION_SCHEMA = "invariant-gravity-ben-development-result-adjudication-4.0"
ACCESS_FAILURE_SCHEMA = "invariant-gravity-ben-development-access-failure-4.0"
PHASE_RECEIPT_SCHEMA = "invariant-gravity-ben-development-phase-receipt-4.0"
TERMINAL_SUCCESS_SCHEMA = "invariant-gravity-ben-development-terminal-success-4.0"
DECISION = "READY_UNAUTHORIZED_ZERO_TARGET_ACCESS"

CONFIG_SECTION_SHA256: dict[str, str] = {
    "ablation_registry": "f490549d6b3ec8fec7950442c595ca408abc4426f55e0aaf3b690bd743684bfe",
    "authorization_gate": "90849359f2627f5e30bac42e9a2e35c4156d56cbb70d67d0675cf917397b0e3b",
    "candidate_registry": "577281c3f36d11d9ac75dadebf241bb984b05df4e75157660d8fa0fce549abec",
    "claim_ceiling": "04a62234da45f74810ee955926c63dffc2da2d17e3164e0de189c66874b46ed1",
    "compute_ceiling": "05c8ffc4686f86bfa794f507c96730c804c67a4b04adbf16f0c2ad3235cc759a",
    "development_populations": "2eed4bb77238ec660a440cd699818c84b4cfca1bec20553b940c4dc384f39209",
    "domain_switch_risk": "f6495f2176c7b0d0f617087866b6e3cb81cea0dabbd77e1154739a2ce1208493",
    "interrupted_run_contract": "e871bdd30c26f20a37be4e1e2067f619e2b36c3cbffb48e78d2a804319185760",
    "output_paths": "5412535a3fe29b865b405551888a54959c8dc7c90e327ec644655dc8c1d05380",
    "result_validation_contract": "cab1b8c4fd145c3b4fda1b71ecd3db8f8bcdf417c1aa399fc5406bab4c619568",
    "runtime_environment_contract": "71f84ad6a460e6e20a47713991630a55062e7018cdf6fdd148c550d1263edd89",
    "runtime_preflight": "b56fca76c13fba701fc5fe6a133e12f6e20a15e2aeff9b6ae38a2e9623a43d7e",
    "selection_contract": "5822a0544e458f3a3e897f55fe281ac63bb1bcc625d3b4e0d9cb846de48573bd",
    "source_bindings": "a23b1c678931400fcdd5b46826fc074d3fb8caa6e3b1d9e4d5406329aea81855",
    "sparc_mapping_and_score": "1f861ee4f34488cbc92e7958ddd783e65920243d7f4c98fec37dd02498fc8b5d",
    "xcop_shape_bridge_and_score": "3e582c6f3490427442b49fc192e12f2ee4226ca3977dc246d5d658cdf566417c",
    "zero_access_chronology": "ceb390531ed30b2971e94771ed89abe9026f2a465e556f48fd23e0ef808f003b",
}
RFC3339_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z$"
)

EXPECTED_XCOP_OBJECTS = [
    "A1644",
    "A1795",
    "A2142",
    "A2255",
    "A2319",
    "A3266",
    "A85",
    "ZW1215",
]
FORBIDDEN_XCOP_OBJECTS = ["A2029", "A3158", "A644", "RXC1825"]
ABLATION_IDS = ["N_zero_ablation", "B_unity_gate_ablation", "A_off_nuisance_ablation"]
PHASES = [
    "access_intent_committed",
    "sparc_development_loaded",
    "xcop_development_loaded",
    "formula_scoring_completed",
    "result_adjudication_passed_in_memory",
    "result_committed",
    "adjudication_committed",
]
XCOP_ROLES = ("density", "pressure", "temperature")
XCOP_HDU = {"density": 2, "pressure": 2, "temperature": 1}
XCOP_COLUMNS = {
    "density": ["RW_X", "NE", "ERR_NE_LO", "ERR_NE_HI"],
    "pressure": ["RW_SZ", "P_SZ", "eP_SZ"],
    "temperature": ["RW_X", "T_X", "eT_X"],
}
ACKNOWLEDGEMENTS = {
    "all_local_sparc_rows_are_development_only_for_this_descendant": True,
    "historical_sparc_score_subset_only": True,
    "xcop_eight_development_objects_only": True,
    "no_confirmation_or_independent_access": True,
    "shape_only_not_absolute_cluster_prediction": True,
    "diagonal_errors_not_full_covariance": True,
    "gas_only_newtonian_cluster_control": True,
    "no_single_counterexample_veto": True,
    "no_formula_family_pruning": True,
    "novelty_labels_non_authoritative": True,
    "no_publication_dark_matter_or_gr_replacement_claim": True,
    "no_network_model_paid_group_or_lensing_access": True,
    "reference_runtime_validation_required_before_scoring": True,
    "numerical_indifference_band_required_for_every_win": True,
    "raw_per_object_losses_not_independently_recomputed_by_adjudicator": True,
}


class BENDevelopmentExecutorV4Error(RuntimeError):
    """Raised before a frozen scope, identity, or access boundary can change."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256((canonical_json(unsigned) + "\n").encode()).hexdigest()


def section_sha256(value: Any) -> str:
    return hashlib.sha256((canonical_json(value) + "\n").encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confined(path: Path) -> Path:
    target = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise BENDevelopmentExecutorV4Error(f"path escaped repository: {path}") from error
    return target


def read_json(path: Path) -> dict[str, Any]:
    target = confined(path)
    if not target.is_file():
        raise BENDevelopmentExecutorV4Error(f"required artifact absent: {path}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BENDevelopmentExecutorV4Error(f"expected JSON object: {path}")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise BENDevelopmentExecutorV4Error(f"{label} keys changed")


def _verify_content_receipt(value: Mapping[str, Any], expected: str, label: str) -> None:
    if value.get("content_sha256") != expected:
        raise BENDevelopmentExecutorV4Error(f"{label} declared content seal changed")
    # These predecessor receipts use several historical content-hash normalizations.  The
    # exact file bytes are already bound and verified above; this check independently
    # requires that the byte-bound artifact still declares the expected semantic seal.


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "status",
        "purpose",
        "implementation_source",
        "verifier_test",
        "source_bindings",
        "candidate_registry",
        "ablation_registry",
        "development_populations",
        "sparc_mapping_and_score",
        "xcop_shape_bridge_and_score",
        "selection_contract",
        "domain_switch_risk",
        "runtime_preflight",
        "runtime_environment_contract",
        "result_validation_contract",
        "interrupted_run_contract",
        "compute_ceiling",
        "authorization_gate",
        "output_paths",
        "zero_access_chronology",
        "claim_ceiling",
    }
    _strict_keys(config, expected, "V4 config")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_unauthorized_zero_target_access"
        or config["implementation_source"] != SOURCE_PATH.as_posix()
        or config["verifier_test"] != TEST_PATH.as_posix()
    ):
        raise BENDevelopmentExecutorV4Error("V4 identity changed")
    if set(CONFIG_SECTION_SHA256) != expected - {
        "schema_version",
        "status",
        "purpose",
        "implementation_source",
        "verifier_test",
    }:
        raise BENDevelopmentExecutorV4Error("frozen section-hash registry is incomplete")
    for section, expected_hash in CONFIG_SECTION_SHA256.items():
        if section_sha256(config[section]) != expected_hash:
            raise BENDevelopmentExecutorV4Error(f"frozen config section changed: {section}")

    expected_binding_labels = {
        "synthetic_config",
        "synthetic_source",
        "sparc_full_sample_source",
        "real_data_gravity_source",
        "sigma_core_source",
        "synthetic_receipt",
        "development_v2_config",
        "development_v2_receipt",
        "shape_v3_config",
        "shape_v3_receipt",
        "sparc_g0_config",
        "sparc_g0_receipt",
        "xcop_item59_config",
        "xcop_source_receipt",
        "item61_sparc_evaluation",
    }
    _strict_keys(config["source_bindings"], expected_binding_labels, "source bindings")
    content_bound = {
        "synthetic_receipt",
        "development_v2_receipt",
        "shape_v3_receipt",
        "sparc_g0_receipt",
        "xcop_source_receipt",
        "item61_sparc_evaluation",
    }
    for label, binding in config["source_bindings"].items():
        expected_binding_keys = {"path", "file_sha256"}
        if label in content_bound:
            expected_binding_keys.add("content_sha256")
        _strict_keys(binding, expected_binding_keys, f"source binding {label}")

    registry = config["candidate_registry"]
    ablations = config["ablation_registry"]
    if (
        registry["registry_content_sha256"]
        != "45966eae73d7641ea982a7eea47aad883a9ff344baf121b91b901c32ef819f19"
        or registry["raw_candidates_frozen"] != 240
        or registry["canonical_full_classes"] != 60
        or registry["raw_members_scored"] is not False
        or registry["novelty_labels_authoritative"] is not False
        or registry["post_response_generation_calls"] != 0
        or registry["post_response_repair_calls"] != 0
        or ablations["ordered_ablation_ids"] != ABLATION_IDS
        or ablations["registered_total"] != 180
        or ablations["unique_ablation_asts"] != 51
        or ablations["duplicate_registered_ablation_instances"] != 129
        or ablations["ablation_asts_overlapping_full_classes"] != 33
        or ablations["unique_asts_across_60_full_plus_180_registered_ablations"] != 78
        or ablations["score_every_registered_ablation"] is not True
    ):
        raise BENDevelopmentExecutorV4Error("candidate or ablation accounting changed")

    populations = config["development_populations"]
    sparc = populations["sparc"]
    xcop = populations["xcop"]
    if (
        sparc["all_local_rows_role"] != "development_only_for_this_descendant"
        or sparc["parsed_container_objects"] != 175
        or sparc["parsed_container_rows"] != 3391
        or sparc["split_confirmation_objects"] != 35
        or sparc["split_confirmation_name_root_sha256"]
        != "1fd327cc974e860af825c9141dd97dbaac59854570dd0a024cb00550fb335ada"
        or sparc["split_exploration_before_admission_objects"] != 140
        or sparc["split_exploration_before_admission_name_root_sha256"]
        != "c8c2f9bcc0a2c6069de3e0f6382bdf60a596923a2336e34552cef598dfe9286e"
        or sparc["objects"] != 139
        or sparc["rows"] != 2720
        or sparc["admitted_score_name_root_sha256"]
        != "655448defd9b6b8092bed4766bd8cd2f7defff204c889092ced08d22e2627057"
        or sparc["admitted_score_name_row_ledger_sha256"]
        != "ea249dc3b71448cabb6ae9f2d0a68040eafe1021e92902311d2390b247b584dc"
        or sparc["rows_outside_subset_scored"] != 0
        or sparc["local_confirmation_role_exists"] is not False
        or xcop["objects"] != EXPECTED_XCOP_OBJECTS
        or xcop["predictor_density_rows"] != 521
        or xcop["response_rows"] != 184
        or xcop["forbidden_objects"] != FORBIDDEN_XCOP_OBJECTS
        or xcop["allowed_roles"] != list(XCOP_ROLES)
        or any(value is not True for value in populations["forbidden"].values())
    ):
        raise BENDevelopmentExecutorV4Error("development row allowlist changed")

    bridge = config["xcop_shape_bridge_and_score"]
    selection = config["selection_contract"]
    if (
        bridge["uses_P500_T500_R500_outer_anchor_or_mass_target"] is not False
        or bridge["matched_nuisance_profile_for_every_candidate_ablation_and_control"] is not True
        or bridge["H_pressure_response_nontrivial_required_before_fit"] is not True
        or bridge["H_temperature_response_nontrivial_required_before_fit"] is not True
        or bridge["absolute_amplitude_identified"] is not False
        or bridge["comparators"]
        != ["gas_only_newtonian_shape", "uniform_acceleration_shape_control"]
        or selection["single_counterexample_terminal"] is not False
        or selection["counterexample_count_alone_terminal"] is not False
        or selection["finite_sample_may_prune_formula_family"] is not False
        or selection["retain_all_failures_and_ties"] is not True
        or selection["every_compared_control_must_be_valid_and_parity_pass"] is not True
        or selection["every_named_ablation_must_be_valid_and_parity_pass_in_both_domains"]
        is not True
        or selection["numerical_indifference_band"]["absolute_tolerance"] != 1.0e-12
        or selection["numerical_indifference_band"]["relative_tolerance"] != 1.0e-10
        or selection["numerical_indifference_band"]["scale_floor"] != 1.0
    ):
        raise BENDevelopmentExecutorV4Error("shape, selection, or counterexample policy changed")
    domain_risk = config["domain_switch_risk"]
    runtime = config["runtime_preflight"]
    environment = config["runtime_environment_contract"]
    result_validation = config["result_validation_contract"]
    interrupted = config["interrupted_run_contract"]
    if (
        domain_risk["xcop_x_geometry_value"] != 1.0
        or domain_risk["xcop_x_geometry_is_constant"] is not True
        or domain_risk["sparc_x_geometry_is_data_derived"] is not True
        or domain_risk["risk_is_evidence_of_new_physics"] is not False
        or domain_risk["risk_blocks_universal_or_novelty_claim"] is not True
        or domain_risk["risk_alone_prunes_formula_family"] is not False
        or runtime["must_complete_before_access_intent"] is not True
        or runtime["minimum_cuda_devices"] != 1
        or runtime["required_device_name_substring"] != "NVIDIA GeForce RTX 5090"
        or runtime["float64_probe_values"] != 16
        or runtime["scientific_payload_access"] is not False
        or environment["policy_id"] != "ben-development-reference-runtime-indifference-v1"
        or environment["reference_environment_validation_required_before_access_intent"] is not True
        or environment["comparison_operator"] != "binary64_numerical_indifference_band"
        or environment["host"]["float_mantissa_bits"] != 53
        or environment["svd_threadpool"]["required_num_threads_during_run"] != 1
        or environment["cuda"]["device_name"] != "NVIDIA GeForce RTX 5090"
        or result_validation["schema_version"] != ADJUDICATION_SCHEMA
        or result_validation["exact_result_schema_required"] is not True
        or result_validation["adjudication_must_pass_before_result_write"] is not True
        or result_validation["reject_if_access_failure_receipt_exists_before_adjudication"]
        is not True
        or result_validation["reload_and_validate_exact_approved_authorization_for_check_result"]
        is not True
        or result_validation[
            "match_authorization_file_sha256_across_access_intent_result_adjudication"
        ]
        is not True
        or result_validation["terminal_success_marker_required_after_runtime_restoration"]
        is not True
        or result_validation["check_result_must_hold_exclusive_terminal_state_lock"] is not True
        or result_validation["raw_per_object_losses_independently_recomputed"] is not False
        or interrupted["terminal_state_failure_receipt_schema"] != ACCESS_FAILURE_SCHEMA
        or interrupted["durable_phase_receipt_schema"] != PHASE_RECEIPT_SCHEMA
        or interrupted["ordered_durable_phases"] != PHASES
        or interrupted["write_atomic_no_clobber_phase_receipt_after_each_completed_phase"]
        is not True
        or interrupted[
            "write_atomic_no_clobber_failure_receipt_on_any_post_intent_or_runtime_restoration_exception"
        ]
        is not True
        or interrupted["persist_raw_exception_class_or_message"] is not False
        or interrupted["fixed_error_code_and_class_allowlist"]
        != {
            "EXTERNAL_INTERRUPT": "interruption",
            "PAYLOAD_READ_FAILURE": "payload_read",
            "PHASE_RECEIPT_PUBLICATION_FAILURE": "phase_receipt_publication",
            "RESULT_PUBLICATION_FAILURE": "result_publication",
            "ADJUDICATION_PUBLICATION_FAILURE": "adjudication_publication",
            "ACCESS_INTENT_PUBLICATION_FAILURE": "access_intent_publication",
            "RUNTIME_RESTORATION_FAILURE": "runtime_restoration",
            "TERMINAL_MARKER_PUBLICATION_FAILURE": "terminal_marker_publication",
            "OUTPUT_PUBLICATION_FAILURE": "output_publication",
            "MEMORY_EXHAUSTION": "resource",
            "CONTRACT_OR_SCORE_VALIDATION_FAILURE": "validation",
            "DETERMINISTIC_EXECUTION_FAILURE": "execution",
            "UNCLASSIFIED_EXECUTION_FAILURE": "internal",
        }
        or interrupted["fixed_failure_operation_allowlist"]
        != [
            "access_intent_publication",
            "sparc_payload_read",
            "xcop_payload_read",
            "phase_receipt_publication",
            "formula_scoring",
            "result_validation",
            "result_publication",
            "adjudication_build",
            "adjudication_publication",
            "runtime_restoration",
            "terminal_marker_publication",
        ]
        or interrupted["last_completed_phase_nullable_until_first_durable_phase_receipt"]
        is not True
        or interrupted["redact_paths_credentials_and_exception_values"] is not True
        or interrupted["authorization_replay_after_failure"] is not False
    ):
        raise BENDevelopmentExecutorV4Error(
            "domain-switch risk, runtime, result validation, or failure policy changed"
        )

    ceiling = config["compute_ceiling"]
    expected_ceiling = {
        "domains": 2,
        "canonical_full_candidates": 60,
        "registered_ablation_variants": 180,
        "domain_specific_comparators": 4,
        "formula_domain_batches_per_backend": 484,
        "cpu_formula_domain_batches": 484,
        "gpu_formula_domain_batches": 484,
        "cpu_gpu_parity_comparisons": 484,
        "sparc_formula_row_cells_per_backend": 658240,
        "xcop_formula_row_cells_per_backend": 126082,
        "total_formula_row_cells_per_backend": 784322,
        "total_formula_row_cells_both_backends": 1568644,
        "xcop_coupled_three_parameter_nuisance_fits": 1936,
        "xcop_zeta_objective_evaluations": 7931792,
        "xcop_analytic_scale_solves": 15863584,
        "maximum_object_score_reductions": 35574,
        "maximum_response_row_score_terms": 702768,
        "candidate_selection_events": 1,
        "result_validation_events": 1,
        "result_adjudication_receipts": 1,
        "maximum_durable_phase_receipts": 7,
        "maximum_access_failure_receipts": 1,
        "maximum_terminal_success_markers": 1,
        "gpu_runtime_preflight_calls": 1,
        "gpu_runtime_preflight_probe_values": 16,
        "threshold_tuning_calls": 0,
        "formula_generation_calls": 0,
        "formula_repair_calls": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "maximum_api_spend_usd": 0.0,
        "maximum_payload_file_open_attempts": 25,
        "maximum_payload_file_successful_reads": 25,
    }
    if ceiling != expected_ceiling:
        raise BENDevelopmentExecutorV4Error("compute ceiling changed")
    gate = config["authorization_gate"]
    if (
        gate["authorization_id"] != "ben-development-score-v4-production-1"
        or gate["authorization_path"] != AUTHORIZATION_PATH.as_posix()
        or gate["current_authorization_expected"] is not False
        or gate["authorization_cannot_expand_scope"] is not True
        or gate["authorization_replay_allowed"] is not False
    ):
        raise BENDevelopmentExecutorV4Error("authorization boundary changed")
    outputs = config["output_paths"]
    if outputs != {
        "preflight_receipt": PREFLIGHT_PATH.as_posix(),
        "authorization": AUTHORIZATION_PATH.as_posix(),
        "access_intent": ACCESS_INTENT_PATH.as_posix(),
        "result": RESULT_PATH.as_posix(),
        "result_adjudication": ADJUDICATION_PATH.as_posix(),
        "access_failure": ACCESS_FAILURE_PATH.as_posix(),
        "phase_receipt_directory": PHASE_RECEIPT_DIRECTORY.as_posix(),
        "terminal_success": TERMINAL_SUCCESS_PATH.as_posix(),
        "terminal_state_lock": TERMINAL_STATE_LOCK_PATH.as_posix(),
    }:
        raise BENDevelopmentExecutorV4Error("output path boundary changed")
    if any(
        value != 0
        for key, value in config["zero_access_chronology"].items()
        if key != "contract_frozen_before_target_access"
    ):
        raise BENDevelopmentExecutorV4Error("preflight claims target access")
    if config["zero_access_chronology"]["contract_frozen_before_target_access"] is not True:
        raise BENDevelopmentExecutorV4Error("preflight chronology changed")


def validate_bound_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate only metadata artifacts.  This function must never open a payload."""

    loaded: dict[str, dict[str, Any]] = {}
    payload_paths = {
        Path(config["development_populations"]["sparc"]["payload_path"]),
        Path(config["development_populations"]["xcop"]["raw_directory"]),
    }
    for label, binding in config["source_bindings"].items():
        path = Path(binding["path"])
        if path in payload_paths or any(parent in payload_paths for parent in path.parents):
            raise BENDevelopmentExecutorV4Error("a metadata binding points at a payload")
        target = confined(path)
        if file_sha256(target) != binding["file_sha256"]:
            raise BENDevelopmentExecutorV4Error(f"bound metadata changed: {label}")
        if target.suffix == ".json":
            value = read_json(path)
            if "content_sha256" in binding:
                _verify_content_receipt(value, binding["content_sha256"], label)
            loaded[label] = value

    synthetic = loaded["synthetic_receipt"]
    registry = synthetic["candidate_registry"]
    validate_registry(registry)
    if registry["content_sha256"] != config["candidate_registry"]["registry_content_sha256"]:
        raise BENDevelopmentExecutorV4Error("candidate registry binding changed")
    v2 = loaded["development_v2_receipt"]
    v3 = loaded["shape_v3_receipt"]
    g0 = loaded["sparc_g0_receipt"]
    item61 = loaded["item61_sparc_evaluation"]
    if (
        v2["claims"]["all_local_sparc_rows_development_only_for_descendant"] is not True
        or v2["claims"]["local_sparc_confirmation_claim_survives"] is not False
        or v3["claims"]["all_local_sparc_rows_development_only"] is not True
        or v3["claims"]["local_sparc_confirmation_claim_survives"] is not False
        or v3["claims"]["predictor_only_xcop_shape_basis_frozen"] is not True
        or v3["claims"]["real_scoring_executed"] is not False
    ):
        raise BENDevelopmentExecutorV4Error("V2/V3 access or mapping lineage changed")
    sparc_contract = config["development_populations"]["sparc"]
    if (
        g0["counts"]["published_galaxies"] != 175
        or g0["counts"]["published_points"] != 3391
        or g0["counts"]["admitted_exploration_galaxies"] != 139
        or g0["counts"]["admitted_exploration_points"] != 2720
        or g0["split"]["salt"] != FULL_SPLIT_SALT
        or g0["split"]["confirmation_count"] != 35
        or g0["split"]["confirmation_name_root_sha256"]
        != sparc_contract["split_confirmation_name_root_sha256"]
        or g0["split"]["exploration_count_before_admission"] != 140
        or g0["split"]["exploration_name_root_sha256"]
        != sparc_contract["split_exploration_before_admission_name_root_sha256"]
        or g0["baseline_replay"]["evaluated_name_root_sha256"]
        != sparc_contract["admitted_score_name_root_sha256"]
    ):
        raise BENDevelopmentExecutorV4Error("G0 SPARC split or count binding changed")
    ledger = [
        {"object": str(row["galaxy"]), "rows": int(row["rows"])}
        for row in sorted(item61["sparc"]["per_object"], key=lambda row: str(row["galaxy"]))
    ]
    if (
        item61["sparc"]["galaxies"] != 139
        or item61["sparc"]["rows"] != 2720
        or len(ledger) != 139
        or sum(row["rows"] for row in ledger) != 2720
    ):
        raise BENDevelopmentExecutorV4Error("Item61 SPARC metadata ledger changed")
    ledger_sha = hashlib.sha256(canonical_json(ledger).encode()).hexdigest()
    name_root = canonical_sha256(sorted(row["object"] for row in ledger))
    if (
        ledger_sha != sparc_contract["admitted_score_name_row_ledger_sha256"]
        or name_root != sparc_contract["admitted_score_name_root_sha256"]
    ):
        raise BENDevelopmentExecutorV4Error("Item61 admitted SPARC ledger seal changed")
    return loaded


def _ast_variables(node: Mapping[str, Any]) -> set[str]:
    if "var" in node:
        return {str(node["var"])}
    if "const" in node:
        return set()
    variables: set[str] = set()
    for child in node["args"]:
        variables.update(_ast_variables(child))
    return variables


def _components_for_representative(registry: Mapping[str, Any], raw_id: str) -> dict[str, Any]:
    raw_rows = {row["raw_id"]: row for row in registry["raw_candidates"]}
    if raw_id not in raw_rows:
        raise BENDevelopmentExecutorV4Error("canonical representative is absent from raw registry")
    ids = raw_rows[raw_id]["component_raw_ids"]
    return {role: _find_component(role, component_id) for role, component_id in ids.items()}


def build_registered_variants(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize 60 full records and 180 named ablations before response access."""

    full: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    for row in sorted(registry["equivalence_classes"], key=lambda item: item["class_id"]):
        full.append(
            {
                "variant_id": f"full:{row['class_id']}",
                "kind": "full",
                "full_class_id": row["class_id"],
                "ablation_id": None,
                "canonical_ast": row["canonical_ast"],
                "canonical_expression": row["canonical_expression"],
                "canonical_expression_sha256": row["canonical_expression_sha256"],
                "representative_raw_id": min(row["raw_member_ids"]),
                "raw_member_count": row["raw_member_count"],
                "provenance_label": row["provenance_label"],
                "provenance_is_authoritative_novelty_finding": False,
            }
        )
        representative = min(row["raw_member_ids"])
        base = _components_for_representative(registry, representative)
        for ablation_id in ABLATION_IDS:
            components = dict(base)
            if ablation_id == "N_zero_ablation":
                components["N_additive_channel"] = _find_component(
                    "N_additive_channel", "N.null_ablation"
                )
            elif ablation_id == "B_unity_gate_ablation":
                components["B_continuous_gate"] = {"ast": {"const": 1.0}}
            elif ablation_id == "A_off_nuisance_ablation":
                components["A_nuisance"] = _find_component("A_nuisance", "A.off")
            ast = normalize_ast(formula_ast(components))
            digest = sha256_value(ast)
            ablations.append(
                {
                    "variant_id": f"ablation:{row['class_id']}:{ablation_id}",
                    "kind": "ablation",
                    "full_class_id": row["class_id"],
                    "ablation_id": ablation_id,
                    "canonical_ast": ast,
                    "canonical_expression": expression(ast),
                    "canonical_expression_sha256": digest,
                    "representative_raw_id": representative,
                    "raw_member_count": 1,
                    "provenance_label": "derived_ablation_not_novelty_assessed",
                    "provenance_is_authoritative_novelty_finding": False,
                }
            )
    variants = full + ablations
    multiplicity = Counter(row["canonical_expression_sha256"] for row in variants)
    for row in variants:
        row["registered_equivalence_multiplicity"] = multiplicity[
            row["canonical_expression_sha256"]
        ]
        row["uses_x_geometry"] = "x_geometry" in _ast_variables(row["canonical_ast"])
        row["constant_xcop_geometry_domain_switch_risk"] = row["uses_x_geometry"]
    ablation_hashes = [row["canonical_expression_sha256"] for row in ablations]
    full_hashes = {row["canonical_expression_sha256"] for row in full}
    if (
        len(full) != 60
        or len(ablations) != 180
        or len(set(ablation_hashes)) != 51
        or len(ablations) - len(set(ablation_hashes)) != 129
        or len(set(ablation_hashes) & full_hashes) != 33
        or len(set(ablation_hashes) | full_hashes) != 78
    ):
        raise BENDevelopmentExecutorV4Error("derived ablation equivalence accounting changed")
    return {
        "full": full,
        "ablations": ablations,
        "variants": variants,
        "accounting": {
            "raw_candidates_frozen": registry["raw_candidate_count"],
            "canonical_full_classes": len(full),
            "registered_ablations": len(ablations),
            "unique_ablation_asts": len(set(ablation_hashes)),
            "duplicate_registered_ablation_instances": len(ablations) - len(set(ablation_hashes)),
            "ablation_asts_overlapping_full_classes": len(set(ablation_hashes) & full_hashes),
            "unique_asts_across_full_and_ablations": len(set(ablation_hashes) | full_hashes),
            "raw_equivalent_members_scored": 0,
            "registered_variants_using_x_geometry": sum(row["uses_x_geometry"] for row in variants),
            "registered_variants_flagged_constant_xcop_geometry_domain_switch_risk": sum(
                row["constant_xcop_geometry_domain_switch_risk"] for row in variants
            ),
        },
    }


def _registered_formula_identity_ledger_sha256(registered: Mapping[str, Any]) -> str:
    ledger = [
        {
            "variant_id": row["variant_id"],
            "kind": row["kind"],
            "full_class_id": row["full_class_id"],
            "ablation_id": row["ablation_id"],
            "canonical_expression_sha256": row["canonical_expression_sha256"],
            "registered_equivalence_multiplicity": row["registered_equivalence_multiplicity"],
            "constant_xcop_geometry_domain_switch_risk": row[
                "constant_xcop_geometry_domain_switch_risk"
            ],
        }
        for row in registered["variants"]
    ]
    return section_sha256(ledger)


def _distribution_identity(distribution_name: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    try:
        distribution = metadata.distribution(distribution_name)
        record = distribution.read_text("RECORD")
    except metadata.PackageNotFoundError as error:
        raise BENDevelopmentExecutorV4Error(
            f"required runtime distribution absent: {distribution_name}"
        ) from error
    if record is None:
        raise BENDevelopmentExecutorV4Error(
            f"runtime distribution RECORD absent: {distribution_name}"
        )
    verified_installed_files = {
        relative_path: file_sha256(Path(distribution.locate_file(relative_path)))
        for relative_path in expected["verified_installed_files"]
    }
    return {
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        "verified_installed_files": verified_installed_files,
    }


def _static_runtime_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate deterministic host/package facts without touching target payloads."""

    contract = config["runtime_environment_contract"]
    windows_release, windows_build, windows_service_pack, windows_build_type = platform.win32_ver()
    host = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable_sha256": file_sha256(Path(sys.executable)),
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "float_mantissa_bits": sys.float_info.mant_dig,
        "windows_release": windows_release,
        "windows_build": windows_build,
        "windows_service_pack": windows_service_pack,
        "windows_build_type": windows_build_type,
        "cpu_processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    packages = {
        name: _distribution_identity(name, expected)
        for name, expected in contract["packages"].items()
    }
    build_dependencies = np.__config__.CONFIG.get("Build Dependencies", {})
    blas = build_dependencies.get("blas", {})
    lapack = build_dependencies.get("lapack", {})
    numpy_build = {
        "blas_name": blas.get("name"),
        "blas_version": blas.get("version"),
        "lapack_name": lapack.get("name"),
        "lapack_version": lapack.get("version"),
        "openblas_configuration": blas.get("openblas configuration"),
    }
    simd = np.__config__.CONFIG.get("SIMD Extensions", {})
    numpy_simd_dispatch = {
        "baseline": simd.get("baseline"),
        "found": simd.get("found"),
        "not_found": simd.get("not found"),
    }
    try:
        from threadpoolctl import threadpool_info
    except ImportError as error:  # pragma: no cover - contract requires the distribution
        raise BENDevelopmentExecutorV4Error("threadpoolctl import failed") from error
    expected_backend = contract["svd_threadpool"]
    pools = [
        row
        for row in threadpool_info()
        if row.get("user_api") == expected_backend["user_api"]
        and row.get("prefix") == expected_backend["prefix"]
        and Path(str(row.get("filepath"))).name == expected_backend["library_filename"]
    ]
    if len(pools) != 1:
        raise BENDevelopmentExecutorV4Error("reference NumPy BLAS/SVD backend is not unique")
    backend = {
        key: pools[0].get(key)
        for key in (
            "user_api",
            "internal_api",
            "prefix",
            "version",
            "threading_layer",
            "architecture",
        )
    }
    backend["library_filename"] = Path(str(pools[0]["filepath"])).name
    backend["library_sha256"] = file_sha256(Path(str(pools[0]["filepath"])))
    if (
        host != contract["host"]
        or packages != contract["packages"]
        or numpy_build != contract["numpy_build"]
        or numpy_simd_dispatch != contract["numpy_simd_dispatch"]
        or backend
        != {
            key: expected_backend[key]
            for key in (
                "user_api",
                "internal_api",
                "prefix",
                "version",
                "threading_layer",
                "architecture",
            )
        }
        | {
            "library_filename": expected_backend["library_filename"],
            "library_sha256": expected_backend["library_sha256"],
        }
    ):
        raise BENDevelopmentExecutorV4Error("validated reference host runtime changed")
    return {
        "policy_id": contract["policy_id"],
        "host": host,
        "packages": packages,
        "numpy_build": numpy_build,
        "numpy_simd_dispatch": numpy_simd_dispatch,
        "svd_backend": backend,
        "scientific_payload_access": False,
    }


def _activate_reference_svd_runtime(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Limit the bound BLAS/SVD backend to one thread for the complete authorized run."""

    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as error:  # pragma: no cover - contract requires the distribution
        raise BENDevelopmentExecutorV4Error("threadpoolctl import failed") from error
    expected = config["runtime_environment_contract"]["svd_threadpool"]
    limiter = threadpool_limits(
        limits=expected["required_num_threads_during_run"], user_api=expected["user_api"]
    )
    pools = [
        row
        for row in threadpool_info()
        if row.get("user_api") == expected["user_api"]
        and row.get("prefix") == expected["prefix"]
        and Path(str(row.get("filepath"))).name == expected["library_filename"]
    ]
    if (
        len(pools) != 1
        or any(
            pools[0].get(key) != expected[key]
            for key in (
                "user_api",
                "internal_api",
                "prefix",
                "version",
                "threading_layer",
                "architecture",
            )
        )
        or pools[0].get("num_threads") != expected["required_num_threads_during_run"]
    ):
        limiter.restore_original_limits()
        raise BENDevelopmentExecutorV4Error("single-thread reference BLAS/SVD activation failed")
    active = {
        "user_api": pools[0]["user_api"],
        "internal_api": pools[0]["internal_api"],
        "prefix": pools[0]["prefix"],
        "version": pools[0]["version"],
        "threading_layer": pools[0]["threading_layer"],
        "architecture": pools[0]["architecture"],
        "num_threads": pools[0]["num_threads"],
    }
    active["library_filename"] = Path(str(pools[0]["filepath"])).name
    active["library_sha256"] = file_sha256(Path(str(pools[0]["filepath"])))
    if (
        active["library_filename"] != expected["library_filename"]
        or active["library_sha256"] != expected["library_sha256"]
    ):
        limiter.restore_original_limits()
        raise BENDevelopmentExecutorV4Error("reference BLAS library identity changed")
    return limiter, active


def _xcop_inventory(
    config: Mapping[str, Any], source_receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_key = {
        (row["cluster"], row["role"]): row
        for row in source_receipt["files"]
        if row["role"] in XCOP_ROLES
    }
    inventory = []
    for cluster in EXPECTED_XCOP_OBJECTS:
        for role in XCOP_ROLES:
            row = by_key.get((cluster, role))
            if row is None:
                raise BENDevelopmentExecutorV4Error(f"missing X-COP metadata: {cluster}:{role}")
            expected_member = (
                f"{cluster}/{cluster}_{'density_L1' if role == 'density' else role}.fits"
            )
            if (
                row["member"] != expected_member
                or row["confirmation_response_opened_after_scientific_freeze"] is not False
            ):
                raise BENDevelopmentExecutorV4Error("X-COP development inventory changed")
            inventory.append(
                {
                    "cluster": cluster,
                    "role": role,
                    "relative_path": (
                        Path(config["development_populations"]["xcop"]["raw_directory"])
                        / row["member"]
                    ).as_posix(),
                    "bytes": row["bytes"],
                    "file_sha256": row["sha256"],
                    "hdu": XCOP_HDU[role],
                    "columns": XCOP_COLUMNS[role],
                }
            )
    if len(inventory) != 24:
        raise BENDevelopmentExecutorV4Error("X-COP inventory is not exactly 24 development files")
    return inventory


def build_preflight() -> dict[str, Any]:
    config = load_config()
    loaded = validate_bound_metadata(config)
    registered = build_registered_variants(loaded["synthetic_receipt"]["candidate_registry"])
    inventory = _xcop_inventory(config, loaded["xcop_source_receipt"])
    static_runtime = _static_runtime_receipt(config)
    body: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "frozen_unauthorized_zero_target_access",
        "decision": DECISION,
        "source_bindings": {
            "config": {
                "path": CONFIG_PATH.as_posix(),
                "file_sha256": file_sha256(confined(CONFIG_PATH)),
            },
            "source": {
                "path": SOURCE_PATH.as_posix(),
                "file_sha256": file_sha256(confined(SOURCE_PATH)),
            },
            "test": {"path": TEST_PATH.as_posix(), "file_sha256": file_sha256(confined(TEST_PATH))},
        },
        "implementation_dependency_file_sha256": {
            label: config["source_bindings"][label]["file_sha256"]
            for label in (
                "synthetic_source",
                "sparc_full_sample_source",
                "real_data_gravity_source",
                "sigma_core_source",
            )
        },
        "bound_metadata_files_validated": len(config["source_bindings"]),
        "config_section_sha256": CONFIG_SECTION_SHA256,
        "immutable_scientific_contract_sha256": {
            section: CONFIG_SECTION_SHA256[section]
            for section in (
                "candidate_registry",
                "ablation_registry",
                "sparc_mapping_and_score",
                "xcop_shape_bridge_and_score",
                "selection_contract",
                "domain_switch_risk",
                "runtime_environment_contract",
                "result_validation_contract",
                "interrupted_run_contract",
                "claim_ceiling",
            )
        },
        "candidate_and_ablation_accounting": registered["accounting"],
        "registered_formula_identity_ledger_sha256": (
            _registered_formula_identity_ledger_sha256(registered)
        ),
        "candidate_registry_content_sha256": config["candidate_registry"][
            "registry_content_sha256"
        ],
        "development_populations": {
            "sparc_container_parsed_objects": 175,
            "sparc_container_parsed_rows": 3391,
            "sparc_objects": 139,
            "sparc_rows": 2720,
            "sparc_role": "development_only",
            "xcop_objects": EXPECTED_XCOP_OBJECTS,
            "xcop_predictor_rows": 521,
            "xcop_response_rows": 184,
            "xcop_role": "development_only",
            "forbidden_xcop_objects": FORBIDDEN_XCOP_OBJECTS,
            "sparc_split_confirmation_name_root_sha256": config["development_populations"]["sparc"][
                "split_confirmation_name_root_sha256"
            ],
            "sparc_split_exploration_name_root_sha256": config["development_populations"]["sparc"][
                "split_exploration_before_admission_name_root_sha256"
            ],
            "sparc_admitted_score_name_root_sha256": config["development_populations"]["sparc"][
                "admitted_score_name_root_sha256"
            ],
            "sparc_admitted_score_name_row_ledger_sha256": config["development_populations"][
                "sparc"
            ]["admitted_score_name_row_ledger_sha256"],
        },
        "xcop_development_file_inventory": inventory,
        "compute_ceiling": config["compute_ceiling"],
        "runtime_preflight": config["runtime_preflight"],
        "runtime_environment_contract": config["runtime_environment_contract"],
        "preparation_static_runtime_validated": static_runtime,
        "result_validation_contract": config["result_validation_contract"],
        "interrupted_run_contract": config["interrupted_run_contract"],
        "domain_switch_risk": {
            **config["domain_switch_risk"],
            "registered_variants_flagged": registered["accounting"][
                "registered_variants_flagged_constant_xcop_geometry_domain_switch_risk"
            ],
        },
        "authorization_contract": {
            "schema_version": AUTHORIZATION_SCHEMA,
            "authorization_id": config["authorization_gate"]["authorization_id"],
            "required_exact_approval_text": config["authorization_gate"]["exact_approval_text"],
            "required_claim_acknowledgements": ACKNOWLEDGEMENTS,
            "authorization_must_precede_payload": True,
            "authorization_replay_allowed": False,
        },
        "zero_access_chronology": config["zero_access_chronology"],
        "claim_ceiling": config["claim_ceiling"],
        "production_executed": False,
        "target_files_opened": 0,
        "target_rows_read": 0,
        "scores_computed": 0,
        "selection_events": 0,
    }
    body["content_sha256"] = content_sha256(body)
    return body


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode()
    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise BENDevelopmentExecutorV4Error(f"refusing to overwrite: {path}") from error
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    """Durably flush the directory entry created by the no-clobber hard link."""

    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return

    # Windows refuses os.open(directory), but a directory handle opened with
    # FILE_FLAG_BACKUP_SEMANTICS and GENERIC_WRITE supports FlushFileBuffers on NTFS.
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(directory.resolve()),
        0x40000000,  # GENERIC_WRITE
        0x00000001 | 0x00000002 | 0x00000004,  # share read/write/delete
        None,
        3,  # OPEN_EXISTING
        0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid:
        raise BENDevelopmentExecutorV4Error(
            f"could not open directory for durability flush: {ctypes.get_last_error()}"
        )
    try:
        if not kernel32.FlushFileBuffers(handle):
            raise BENDevelopmentExecutorV4Error(
                f"directory durability flush failed: {ctypes.get_last_error()}"
            )
    finally:
        kernel32.CloseHandle(handle)


def _acquire_terminal_state_lock(*, create: bool) -> Any:
    """Take the exclusive cross-process lock guarding terminal-state observation."""

    target = confined(TERMINAL_STATE_LOCK_PATH)
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = target.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
            os.fsync(handle.fileno())
    else:
        if not target.is_file():
            raise BENDevelopmentExecutorV4Error("terminal-state lock artifact is absent")
        handle = target.open("r+b")
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - production contract is Windows
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise BENDevelopmentExecutorV4Error("terminal state is still active") from error
    return handle


def _release_terminal_state_lock(handle: Any) -> None:
    try:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - production contract is Windows
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            # Closing the descriptor below is the authoritative OS-level unlock.
            pass
    except (OSError, ValueError):
        pass
    finally:
        try:
            handle.close()
        except OSError:
            pass


def write_preflight() -> Path:
    receipt = build_preflight()
    _atomic_no_clobber(PREFLIGHT_PATH, receipt)
    return confined(PREFLIGHT_PATH)


def validate_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    receipt = read_json(PREFLIGHT_PATH)
    if receipt.get("schema_version") != PREFLIGHT_SCHEMA or receipt.get("decision") != DECISION:
        raise BENDevelopmentExecutorV4Error("preflight identity changed")
    if receipt.get("content_sha256") != content_sha256(receipt):
        raise BENDevelopmentExecutorV4Error("preflight content seal changed")
    expected = build_preflight()
    _strict_keys(receipt, set(expected), "preflight")
    if receipt != expected:
        raise BENDevelopmentExecutorV4Error(
            "stored preflight does not equal deterministic build_preflight"
        )
    return receipt


def authorization_template(
    config: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorization_id": config["authorization_gate"]["authorization_id"],
        "authorization_path": AUTHORIZATION_PATH.as_posix(),
        "authorized": False,
        "approved_by": None,
        "approved_at": None,
        "approval_text": config["authorization_gate"]["exact_approval_text"],
        "config_file_sha256": file_sha256(confined(CONFIG_PATH)),
        "source_file_sha256": file_sha256(confined(SOURCE_PATH)),
        "test_file_sha256": file_sha256(confined(TEST_PATH)),
        "implementation_dependency_file_sha256": {
            label: config["source_bindings"][label]["file_sha256"]
            for label in (
                "synthetic_source",
                "sparc_full_sample_source",
                "real_data_gravity_source",
                "sigma_core_source",
            )
        },
        "preflight_receipt_file_sha256": file_sha256(confined(PREFLIGHT_PATH)),
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "candidate_registry_content_sha256": config["candidate_registry"][
            "registry_content_sha256"
        ],
        "registered_formula_identity_ledger_sha256": preflight[
            "registered_formula_identity_ledger_sha256"
        ],
        "config_section_sha256": CONFIG_SECTION_SHA256,
        "immutable_scientific_contract_sha256": {
            section: CONFIG_SECTION_SHA256[section]
            for section in (
                "candidate_registry",
                "ablation_registry",
                "sparc_mapping_and_score",
                "xcop_shape_bridge_and_score",
                "selection_contract",
                "domain_switch_risk",
                "runtime_environment_contract",
                "result_validation_contract",
                "interrupted_run_contract",
                "claim_ceiling",
            )
        },
        "sparc_role": "development_only",
        "sparc_container_parsed_objects": 175,
        "sparc_container_parsed_rows": 3391,
        "sparc_objects": 139,
        "sparc_rows": 2720,
        "sparc_split_confirmation_name_root_sha256": config["development_populations"]["sparc"][
            "split_confirmation_name_root_sha256"
        ],
        "sparc_split_exploration_name_root_sha256": config["development_populations"]["sparc"][
            "split_exploration_before_admission_name_root_sha256"
        ],
        "sparc_admitted_score_name_root_sha256": config["development_populations"]["sparc"][
            "admitted_score_name_root_sha256"
        ],
        "sparc_admitted_score_name_row_ledger_sha256": config["development_populations"]["sparc"][
            "admitted_score_name_row_ledger_sha256"
        ],
        "xcop_role": "development_only",
        "xcop_objects": EXPECTED_XCOP_OBJECTS,
        "xcop_predictor_rows": 521,
        "xcop_response_rows": 184,
        "compute_ceiling": config["compute_ceiling"],
        "runtime_preflight": config["runtime_preflight"],
        "runtime_environment_contract": config["runtime_environment_contract"],
        "result_validation_contract": config["result_validation_contract"],
        "interrupted_run_contract": config["interrupted_run_contract"],
        "domain_switch_risk": config["domain_switch_risk"],
        "claim_acknowledgements": ACKNOWLEDGEMENTS,
        "access_intent_path": ACCESS_INTENT_PATH.as_posix(),
        "result_path": RESULT_PATH.as_posix(),
        "result_adjudication_path": ADJUDICATION_PATH.as_posix(),
        "access_failure_path": ACCESS_FAILURE_PATH.as_posix(),
        "phase_receipt_directory": PHASE_RECEIPT_DIRECTORY.as_posix(),
        "terminal_success_path": TERMINAL_SUCCESS_PATH.as_posix(),
        "terminal_state_lock_path": TERMINAL_STATE_LOCK_PATH.as_posix(),
    }


def write_unauthorized_template() -> Path:
    config = load_config()
    validate_bound_metadata(config)
    preflight = validate_preflight(config)
    _atomic_no_clobber(AUTHORIZATION_PATH, authorization_template(config, preflight))
    return confined(AUTHORIZATION_PATH)


def validate_authorization(
    path: Path, config: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    if confined(path) != confined(AUTHORIZATION_PATH):
        raise BENDevelopmentExecutorV4Error("authorization must use the exact frozen path")
    supplied = read_json(path)
    expected = authorization_template(config, preflight)
    _strict_keys(supplied, set(expected), "authorization")
    for key, value in expected.items():
        if key in {"authorized", "approved_by", "approved_at"}:
            continue
        if supplied[key] != value:
            raise BENDevelopmentExecutorV4Error(f"authorization scope changed: {key}")
    if supplied["authorized"] is not True:
        raise BENDevelopmentExecutorV4Error("exact production authorization is required")
    if not isinstance(supplied["approved_by"], str) or not supplied["approved_by"].strip():
        raise BENDevelopmentExecutorV4Error("approved_by must identify the approver")
    if (
        not isinstance(supplied["approved_at"], str)
        or RFC3339_UTC.fullmatch(supplied["approved_at"]) is None
    ):
        raise BENDevelopmentExecutorV4Error("approved_at is not an exact RFC3339 UTC instant")
    try:
        instant = datetime.fromisoformat(supplied["approved_at"])
    except ValueError as error:
        raise BENDevelopmentExecutorV4Error("approved_at is not RFC3339 UTC") from error
    if instant.utcoffset() is None or instant.utcoffset().total_seconds() != 0.0:
        raise BENDevelopmentExecutorV4Error("approved_at is not a UTC instant")
    return supplied


def _galaxies_from_payload(payload: Mapping[str, Any]) -> list[Galaxy]:
    galaxies = []
    for entry in payload["galaxies"]:
        columns = list(zip(*entry["rows"], strict=True))
        galaxies.append(
            Galaxy(
                name=entry["name"],
                distance_mpc=entry["distance_mpc"],
                radius=tuple(_decimal(value) for value in columns[0]),
                v_obs=tuple(_decimal(value) for value in columns[1]),
                e_v_obs=tuple(_decimal(value) for value in columns[2]),
                v_gas=tuple(_decimal(value) for value in columns[3]),
                v_disk=tuple(_decimal(value) for value in columns[4]),
                v_bul=tuple(_decimal(value) for value in columns[5]),
                published=tuple(tuple(row) for row in entry["rows"]),
            )
        )
    return galaxies


def _read_payload_bytes(
    relative_path: Path,
    accounting: dict[str, int],
    domain_attempt_key: str,
    domain_completed_key: str,
) -> bytes:
    """Count an attempted payload open before I/O and a completed read only afterward."""

    accounting["payload_file_open_attempts"] = accounting.get("payload_file_open_attempts", 0) + 1
    accounting[domain_attempt_key] = accounting.get(domain_attempt_key, 0) + 1
    raw = confined(relative_path).read_bytes()
    accounting["payload_file_reads_completed"] = (
        accounting.get("payload_file_reads_completed", 0) + 1
    )
    accounting[domain_completed_key] = accounting.get(domain_completed_key, 0) + 1
    return raw


def load_sparc_development(
    config: Mapping[str, Any], access_accounting: dict[str, int] | None = None
) -> dict[str, Any]:
    accounting = access_accounting if access_accounting is not None else {}
    binding = config["development_populations"]["sparc"]
    raw = _read_payload_bytes(
        Path(binding["payload_path"]),
        accounting,
        "sparc_file_open_attempts",
        "sparc_file_reads_completed",
    )
    if hashlib.sha256(raw).hexdigest() != binding["payload_file_sha256"]:
        raise BENDevelopmentExecutorV4Error("SPARC payload file seal changed")
    payload = json.loads(raw.decode("utf-8"))
    validate_dataset(payload)
    if canonical_sha256(payload) != binding["payload_content_sha256"]:
        raise BENDevelopmentExecutorV4Error("SPARC payload semantic seal changed")
    galaxies = _galaxies_from_payload(payload)
    parsed_rows = sum(galaxy.count for galaxy in galaxies)
    if len(galaxies) != 175 or parsed_rows != 3391:
        raise BENDevelopmentExecutorV4Error("SPARC parsed container counts changed")
    accounting["sparc_container_objects_parsed"] = len(galaxies)
    accounting["sparc_container_rows_parsed"] = parsed_rows
    split = declare_split(
        [galaxy.name for galaxy in galaxies],
        count=int(CONFIRMATION_FRACTION * len(galaxies)),
        salt=FULL_SPLIT_SALT,
        rule=FULL_SPLIT_RULE,
    )
    if (
        len(split.confirmation) != 35
        or len(split.exploration) != 140
        or canonical_sha256(sorted(split.confirmation))
        != binding["split_confirmation_name_root_sha256"]
        or canonical_sha256(sorted(split.exploration))
        != binding["split_exploration_before_admission_name_root_sha256"]
    ):
        raise BENDevelopmentExecutorV4Error("SPARC name-only split changed")
    convention = payload["mass_to_light_convention"]
    admitted, _admission = admit(
        galaxies,
        Fraction(convention["disk_3_6um"]),
        Fraction(convention["bulge_3_6um"]),
    )
    allowed = set(split.exploration)
    selected = [galaxy for galaxy in admitted if galaxy.name in allowed]
    if len(selected) != 139 or sum(galaxy.count for galaxy in selected) != 2720:
        raise BENDevelopmentExecutorV4Error("historical SPARC development allowlist changed")
    accounting["sparc_score_objects_loaded"] = len(selected)
    accounting["sparc_score_rows_loaded"] = sum(galaxy.count for galaxy in selected)
    admitted_ledger = [
        {"object": galaxy.name, "rows": galaxy.count}
        for galaxy in sorted(selected, key=lambda galaxy: galaxy.name)
    ]
    if (
        canonical_sha256(sorted(row["object"] for row in admitted_ledger))
        != binding["admitted_score_name_root_sha256"]
        or hashlib.sha256(canonical_json(admitted_ledger).encode()).hexdigest()
        != binding["admitted_score_name_row_ledger_sha256"]
    ):
        raise BENDevelopmentExecutorV4Error("SPARC admitted name/row ledger changed")
    rows: list[dict[str, Any]] = []
    for galaxy in selected:
        radius = np.asarray([float(value) for value in galaxy.radius], dtype=np.float64)
        vobs = np.asarray([float(value) for value in galaxy.v_obs], dtype=np.float64)
        sigma = np.asarray([float(value) for value in galaxy.e_v_obs], dtype=np.float64)
        vgas = np.asarray([float(value) for value in galaxy.v_gas], dtype=np.float64)
        vdisk = np.asarray([float(value) for value in galaxy.v_disk], dtype=np.float64)
        vbul = np.asarray([float(value) for value in galaxy.v_bul], dtype=np.float64)
        vbar2 = vgas * np.abs(vgas) + 0.5 * vdisk**2 + 0.7 * vbul**2
        state_denom = vgas**2 + 0.5 * vdisk**2 + 0.7 * vbul**2
        geometry_denom = 0.5 * vdisk**2 + 0.7 * vbul**2
        if (
            np.any(radius <= 0)
            or np.any(sigma <= 0)
            or np.any(vbar2 <= 0)
            or np.any(state_denom <= 0)
            or np.any(geometry_denom <= 0)
        ):
            raise BENDevelopmentExecutorV4Error(
                f"SPARC predictor denominator invalid: {galaxy.name}"
            )
        predictors = np.column_stack(
            (
                vbar2 / (radius * 3702.81458),
                radius / np.max(radius),
                vgas**2 / state_denom,
                0.7 * vbul**2 / geometry_denom,
            )
        )
        rows.append(
            {
                "object": galaxy.name,
                "rows": galaxy.count,
                "radius": radius,
                "vobs": vobs,
                "sigma": sigma,
                "vbar2": vbar2,
                "predictors": predictors,
            }
        )
    return {
        "objects": rows,
        "payload_file_open_attempts": 1,
        "payload_file_reads_completed": 1,
        "parsed_container_objects": len(galaxies),
        "parsed_container_rows": parsed_rows,
        "scored_objects": len(rows),
        "scored_rows": sum(row["rows"] for row in rows),
        "split_confirmation_objects": len(split.confirmation),
        "split_exploration_before_admission_objects": len(split.exploration),
        "admitted_score_name_root_sha256": binding["admitted_score_name_root_sha256"],
        "admitted_score_name_row_ledger_sha256": binding["admitted_score_name_row_ledger_sha256"],
    }


def _fits_table_from_bytes(raw: bytes, hdu_index: int, expected_columns: Sequence[str]) -> Any:
    with fits.open(io.BytesIO(raw), memmap=False) as handle:
        data = handle[hdu_index].data.copy()
    if list(data.dtype.names or ()) != list(expected_columns):
        raise BENDevelopmentExecutorV4Error("X-COP FITS column schema changed")
    return data


def load_xcop_development(
    config: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    access_accounting: dict[str, int] | None = None,
) -> dict[str, Any]:
    accounting = access_accounting if access_accounting is not None else {}
    loaded: dict[tuple[str, str], Any] = {}
    for record in inventory:
        if record["cluster"] not in EXPECTED_XCOP_OBJECTS or record["role"] not in XCOP_ROLES:
            raise BENDevelopmentExecutorV4Error("X-COP inventory escaped development allowlist")
        raw = _read_payload_bytes(
            Path(record["relative_path"]),
            accounting,
            "xcop_file_open_attempts",
            "xcop_file_reads_completed",
        )
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["file_sha256"]:
            raise BENDevelopmentExecutorV4Error("X-COP payload seal changed")
        loaded[(record["cluster"], record["role"])] = _fits_table_from_bytes(
            raw, record["hdu"], record["columns"]
        )
    clusters: list[dict[str, Any]] = []
    density_rows = 0
    response_rows = 0
    for cluster in EXPECTED_XCOP_OBJECTS:
        density = loaded[(cluster, "density")]
        pressure = loaded[(cluster, "pressure")]
        temperature = loaded[(cluster, "temperature")]
        rw = np.asarray(density["RW_X"], dtype=np.float64)
        ne = np.asarray(density["NE"], dtype=np.float64)
        pr = np.asarray(pressure["RW_SZ"], dtype=np.float64)
        py = np.asarray(pressure["P_SZ"], dtype=np.float64)
        pe = np.asarray(pressure["eP_SZ"], dtype=np.float64)
        tr = np.asarray(temperature["RW_X"], dtype=np.float64)
        ty = np.asarray(temperature["T_X"], dtype=np.float64)
        te = np.asarray(temperature["eT_X"], dtype=np.float64)
        arrays = (rw, ne, pr, py, pe, tr, ty, te)
        if any(np.any(~np.isfinite(value)) or np.any(value <= 0) for value in arrays):
            raise BENDevelopmentExecutorV4Error(f"nonpositive X-COP source: {cluster}")
        if np.any(np.diff(rw) <= 0) or np.any(np.diff(pr) <= 0) or np.any(np.diff(tr) <= 0):
            raise BENDevelopmentExecutorV4Error(f"unordered X-COP radii: {cluster}")
        x = rw / np.max(rw)
        q = ne / np.max(ne)
        log_x = np.log(x)
        log_q = np.log(q)
        slope = np.empty_like(q)
        slope[0] = (log_q[1] - log_q[0]) / (log_x[1] - log_x[0])
        slope[-1] = (log_q[-1] - log_q[-2]) / (log_x[-1] - log_x[-2])
        slope[1:-1] = (log_q[2:] - log_q[:-2]) / (log_x[2:] - log_x[:-2])
        px = pr / np.max(rw)
        tx = tr / np.max(rw)
        if (
            len(px) < 3
            or len(tx) < 3
            or px[0] < x[0]
            or px[-1] > x[-1]
            or tx[0] < x[0]
            or tx[-1] > x[-1]
        ):
            raise BENDevelopmentExecutorV4Error(
                f"X-COP response outside density support: {cluster}"
            )
        clusters.append(
            {
                "object": cluster,
                "x": x,
                "q": q,
                "predictors": np.column_stack((q, x, np.abs(slope), np.ones_like(x))),
                "pressure_x": px,
                "pressure_y": py,
                "pressure_sigma": np.maximum(pe, 0.05 * np.abs(py)),
                "temperature_x": tx,
                "temperature_y": ty,
                "temperature_sigma": np.maximum(te, 0.05 * np.abs(ty)),
            }
        )
        density_rows += len(rw)
        response_rows += len(pr) + len(tr)
    if density_rows != 521 or response_rows != 184:
        raise BENDevelopmentExecutorV4Error("X-COP development row counts changed")
    accounting["xcop_objects_loaded"] = len(clusters)
    accounting["xcop_predictor_rows_loaded"] = density_rows
    accounting["xcop_response_rows_loaded"] = response_rows
    return {
        "objects": clusters,
        "payload_file_open_attempts": len(inventory),
        "payload_file_reads_completed": len(inventory),
        "objects_parsed": len(clusters),
        "predictor_rows_parsed_and_scored": density_rows,
        "response_rows_parsed_and_scored": response_rows,
    }


def _evaluate_ast_xp(node: Mapping[str, Any], predictors: Any, xp: Any) -> Any:
    validate_ast(node)
    if "const" in node:
        return xp.full(predictors.shape[0], float(node["const"]), dtype=xp.float64)
    if "var" in node:
        order = ("x_source", "x_radial", "x_state", "x_geometry")
        return predictors[:, order.index(str(node["var"]))]
    values = [_evaluate_ast_xp(child, predictors, xp) for child in node["args"]]
    name = node["op"]
    if name == "add":
        return values[0] + values[1]
    if name == "subtract":
        return values[0] - values[1]
    if name == "multiply":
        return values[0] * values[1]
    if name == "divide_safe":
        return values[0] / values[1]
    if name == "sqrt_positive":
        return xp.sqrt(values[0])
    if name == "exp_negative":
        return xp.exp(-values[0])
    raise BENDevelopmentExecutorV4Error(f"unknown AST operator: {name}")


def _parity(cpu: np.ndarray, gpu: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    if cpu.shape != gpu.shape:
        return {"pass": False, "max_abs": None, "max_rel": None, "reason": "shape_mismatch"}
    finite = np.all(np.isfinite(cpu)) and np.all(np.isfinite(gpu))
    if not finite:
        return {"pass": False, "max_abs": None, "max_rel": None, "reason": "nonfinite"}
    absolute = np.abs(cpu - gpu)
    relative = absolute / np.maximum(np.maximum(np.abs(cpu), np.abs(gpu)), 1.0)
    max_abs = float(np.max(absolute, initial=0.0))
    max_rel = float(np.max(relative, initial=0.0))
    selection = config["selection_contract"]
    passed = (
        max_abs <= selection["parity_absolute_tolerance"]
        or max_rel <= selection["parity_relative_tolerance"]
    )
    return {
        "pass": bool(passed),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "reason": None if passed else "tolerance",
    }


def _loss(prediction: np.ndarray, observed: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(np.square((prediction - observed) / sigma)))


def _score_sparc_vector(output: np.ndarray, objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if np.any(~np.isfinite(output)) or np.any(output <= 0):
        return {
            "valid": False,
            "domain_loss": None,
            "per_object": [],
            "failures": ["nonpositive_formula_output"],
            "object_score_reductions": 0,
            "response_row_score_terms": 0,
        }
    per_object = []
    cursor = 0
    for row in objects:
        count = row["rows"]
        values = output[cursor : cursor + count]
        prediction = np.sqrt(3702.81458 * values * row["radius"])
        loss = _loss(prediction, row["vobs"], row["sigma"])
        per_object.append(
            {
                "object": row["object"],
                "rows": count,
                "loss": loss,
                "terminal_veto": False,
            }
        )
        cursor += count
    if cursor != len(output):
        raise BENDevelopmentExecutorV4Error("SPARC score row accounting changed")
    domain_loss = float(np.mean([row["loss"] for row in per_object]))
    return {
        "valid": True,
        "domain_loss": domain_loss,
        "per_object": per_object,
        "failures": [],
        "object_score_reductions": len(per_object),
        "response_row_score_terms": cursor,
    }


def _reverse_integral(x: np.ndarray, q: np.ndarray, f: np.ndarray) -> np.ndarray:
    h = np.zeros_like(x)
    for index in range(len(x) - 2, -1, -1):
        h[index] = h[index + 1] + 0.5 * (q[index] * f[index] + q[index + 1] * f[index + 1]) * (
            x[index + 1] - x[index]
        )
    h[-1] = 0.0
    return h


def _shape_gates(x: np.ndarray, q: np.ndarray, f: np.ndarray, h: np.ndarray) -> list[str]:
    failures = []
    scale = max(1.0, float(np.max(np.abs(h))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if np.any(h < -tolerance):
        failures.append("H_negative")
    if h[-1] != 0.0:
        failures.append("H_outer_not_exact_zero")
    derivative = np.diff(h) / np.diff(x)
    expected = -0.5 * (q[:-1] * f[:-1] + q[1:] * f[1:])
    derivative_tolerance = (
        64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(expected))))
    )
    if np.any(derivative >= derivative_tolerance) or not np.allclose(
        derivative, expected, rtol=64.0 * np.finfo(np.float64).eps, atol=derivative_tolerance
    ):
        failures.append("H_derivative_identity")
    if float(np.max(h) - np.min(h)) < math.sqrt(np.finfo(np.float64).eps) * scale:
        failures.append("H_constant")
    return failures


def _nontrivial_response_variation(h: np.ndarray) -> bool:
    scale = max(1.0, float(np.max(np.abs(h))))
    return bool(float(np.max(h) - np.min(h)) >= math.sqrt(np.finfo(np.float64).eps) * scale)


def _fit_shape_nuisance(
    hp: np.ndarray, ht: np.ndarray, qt: np.ndarray, cluster: Mapping[str, Any]
) -> dict[str, Any]:
    zeta = np.arange(4097, dtype=np.float64) / 4096.0
    phi_p = hp[None, :] + zeta[:, None] * (1.0 - hp[None, :])
    phi_t = ht[None, :] + zeta[:, None] * (1.0 - ht[None, :])
    p_y = cluster["pressure_y"]
    p_s = cluster["pressure_sigma"]
    t_y = cluster["temperature_y"]
    t_s = cluster["temperature_sigma"]
    p_w = 1.0 / p_s**2
    t_w = 1.0 / t_s**2
    p_den = np.sum(p_w[None, :] * phi_p**2, axis=1)
    t_design = phi_t / qt[None, :]
    t_den = np.sum(t_w[None, :] * t_design**2, axis=1)
    if np.any(p_den <= 0) or np.any(t_den <= 0):
        return {
            "valid": False,
            "loss": None,
            "failures": ["zero_norm_design"],
            "nuisance_fits": 1,
            "zeta_objective_evaluations": 0,
            "analytic_scale_solves": 0,
            "response_row_score_terms": 0,
            "object_score_reductions": 0,
        }
    b_p = np.maximum(0.0, np.sum(p_w[None, :] * phi_p * p_y[None, :], axis=1) / p_den)
    b_t = np.maximum(0.0, np.sum(t_w[None, :] * t_design * t_y[None, :], axis=1) / t_den)
    p_residual = (b_p[:, None] * phi_p - p_y[None, :]) / p_s[None, :]
    t_residual = (b_t[:, None] * t_design - t_y[None, :]) / t_s[None, :]
    losses = 0.5 * (np.mean(p_residual**2, axis=1) + np.mean(t_residual**2, axis=1))
    index = int(np.argmin(losses))
    zp = float(zeta[index])
    bp = float(b_p[index])
    bt = float(b_t[index])
    pp = bp * phi_p[index]
    failures = []
    if not (zp < 1.0 and bp > 0.0 and bt > 0.0):
        failures.append("neutral_or_nonpositive_nuisance_optimum")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(pp))))
    if np.any(np.diff(pp) > tolerance):
        failures.append("pressure_not_nonincreasing")
    jacobian = np.vstack(
        (
            np.column_stack((bp * (1.0 - hp), phi_p[index], np.zeros_like(hp))),
            np.column_stack((bt * (1.0 - ht) / qt, np.zeros_like(ht), phi_t[index] / qt)),
        )
    )
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms <= 0):
        failures.append("jacobian_zero_column")
        rank = 0
        condition = math.inf
    else:
        singular = np.linalg.svd(jacobian / norms[None, :], compute_uv=False)
        rank = int(np.sum(singular > singular[0] * math.sqrt(np.finfo(np.float64).eps)))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
        if rank != 3 or condition > 67108864.0:
            failures.append("jacobian_rank_or_condition")
    return {
        "valid": not failures,
        "loss": float(losses[index]),
        "zeta": zp,
        "b_P": bp,
        "b_T": bt,
        "jacobian_rank": rank,
        "jacobian_condition": None if not math.isfinite(condition) else condition,
        "failures": failures,
        "nuisance_fits": 1,
        "zeta_objective_evaluations": len(zeta),
        "analytic_scale_solves": 2 * len(zeta),
        "response_row_score_terms": len(p_y) + len(t_y),
        "object_score_reductions": 1,
    }


def _score_xcop_shapes(
    outputs: Sequence[np.ndarray], clusters: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    per_object = []
    failures = []
    nuisance_fits = 0
    zeta_objective_evaluations = 0
    analytic_scale_solves = 0
    response_row_score_terms = 0
    object_score_reductions = 0
    for f_raw, cluster in zip(outputs, clusters, strict=True):
        scoped = []
        if np.any(~np.isfinite(f_raw)) or np.any(f_raw <= 0) or float(np.max(f_raw)) <= 0:
            scoped.append("nonpositive_formula_output")
            per_object.append(
                {
                    "object": cluster["object"],
                    "response_rows": len(cluster["pressure_y"]) + len(cluster["temperature_y"]),
                    "valid": False,
                    "loss": None,
                    "nuisance": {"zeta": None, "b_P": None, "b_T": None},
                    "H_pressure_response_nontrivial": None,
                    "H_temperature_response_nontrivial": None,
                    "jacobian_rank": None,
                    "jacobian_condition": None,
                    "failures": scoped,
                    "terminal_veto": False,
                    "nuisance_fit_executed": False,
                    "zeta_objective_evaluations": 0,
                    "analytic_scale_solves": 0,
                    "response_row_score_terms": 0,
                    "object_score_reductions": 0,
                }
            )
            failures.append(f"{cluster['object']}:nonpositive_formula_output")
            continue
        f = f_raw / np.max(f_raw)
        h = _reverse_integral(cluster["x"], cluster["q"], f)
        scoped.extend(_shape_gates(cluster["x"], cluster["q"], f, h))
        hp = np.interp(cluster["pressure_x"], cluster["x"], h)
        ht = np.interp(cluster["temperature_x"], cluster["x"], h)
        pressure_varies = _nontrivial_response_variation(hp)
        temperature_varies = _nontrivial_response_variation(ht)
        if not pressure_varies:
            scoped.append("H_pressure_response_constant")
        if not temperature_varies:
            scoped.append("H_temperature_response_constant")
        qt = np.exp(
            np.interp(np.log(cluster["temperature_x"]), np.log(cluster["x"]), np.log(cluster["q"]))
        )
        if scoped:
            fit = {
                "valid": False,
                "loss": None,
                "failures": [],
                "nuisance_fits": 0,
                "zeta_objective_evaluations": 0,
                "analytic_scale_solves": 0,
                "response_row_score_terms": 0,
                "object_score_reductions": 0,
            }
        else:
            fit = _fit_shape_nuisance(hp, ht, qt, cluster)
        scoped.extend(fit["failures"])
        nuisance_fits += fit["nuisance_fits"]
        zeta_objective_evaluations += fit["zeta_objective_evaluations"]
        analytic_scale_solves += fit["analytic_scale_solves"]
        response_row_score_terms += fit["response_row_score_terms"]
        object_score_reductions += fit["object_score_reductions"]
        for failure in scoped:
            failures.append(f"{cluster['object']}:{failure}")
        per_object.append(
            {
                "object": cluster["object"],
                "response_rows": len(cluster["pressure_y"]) + len(cluster["temperature_y"]),
                "valid": not scoped,
                "loss": fit.get("loss"),
                "nuisance": {key: fit.get(key) for key in ("zeta", "b_P", "b_T")},
                "H_pressure_response_nontrivial": pressure_varies,
                "H_temperature_response_nontrivial": temperature_varies,
                "jacobian_rank": fit.get("jacobian_rank"),
                "jacobian_condition": fit.get("jacobian_condition"),
                "failures": scoped,
                "terminal_veto": False,
                "nuisance_fit_executed": fit["nuisance_fits"] == 1,
                "zeta_objective_evaluations": fit["zeta_objective_evaluations"],
                "analytic_scale_solves": fit["analytic_scale_solves"],
                "response_row_score_terms": fit["response_row_score_terms"],
                "object_score_reductions": fit["object_score_reductions"],
            }
        )
    valid_losses = [row["loss"] for row in per_object if row["valid"] and row["loss"] is not None]
    valid = len(valid_losses) == len(clusters)
    return {
        "valid": valid,
        "domain_loss": float(np.mean(valid_losses)) if valid else None,
        "valid_object_scores": len(valid_losses),
        "per_object": per_object,
        "failures": failures,
        "formula_family_pruned": False,
        "nuisance_fits": nuisance_fits,
        "zeta_objective_evaluations": zeta_objective_evaluations,
        "analytic_scale_solves": analytic_scale_solves,
        "response_row_score_terms": response_row_score_terms,
        "object_score_reductions": object_score_reductions,
    }


def _split_vector(
    values: np.ndarray, objects: Sequence[Mapping[str, Any]], key: str
) -> list[np.ndarray]:
    output = []
    cursor = 0
    for row in objects:
        count = len(row[key])
        output.append(values[cursor : cursor + count])
        cursor += count
    if cursor != len(values):
        raise BENDevelopmentExecutorV4Error("domain vector split accounting changed")
    return output


def _gas_newtonian_shapes(clusters: Sequence[Mapping[str, Any]], xp: Any) -> Any:
    values = []
    for cluster in clusters:
        x = xp.asarray(cluster["x"], dtype=xp.float64)
        q = xp.asarray(cluster["q"], dtype=xp.float64)
        integrand = q * x**2
        increments = 0.5 * (integrand[:-1] + integrand[1:]) * (x[1:] - x[:-1])
        m = xp.concatenate(
            (xp.asarray([q[0] * x[0] ** 3 / 3.0]), q[0] * x[0] ** 3 / 3.0 + xp.cumsum(increments))
        )
        values.append(m / x**2)
    return xp.concatenate(values)


def _strictly_improves(
    candidate_loss: float, reference_loss: float, config: Mapping[str, Any]
) -> bool:
    """Apply the frozen, target-independent absolute-plus-relative indifference band."""

    if not math.isfinite(candidate_loss) or not math.isfinite(reference_loss):
        return False
    band = config["selection_contract"]["numerical_indifference_band"]
    tolerance = band["absolute_tolerance"] + band["relative_tolerance"] * max(
        abs(candidate_loss), abs(reference_loss), band["scale_floor"]
    )
    return candidate_loss < reference_loss - tolerance


def _build_candidate_decisions(
    registered: Mapping[str, Any],
    sparc_results: Mapping[str, Mapping[str, Any]],
    xcop_results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare binary64 losses through the preregistered indifference band."""

    sparc_control_ids = (
        "control:sparc:newtonian_baryons",
        "control:sparc:empirical_rar",
    )
    xcop_control_ids = (
        "control:xcop:gas_only_newtonian_shape",
        "control:xcop:uniform_acceleration_shape_control",
    )
    controls_valid = all(
        sparc_results[item]["valid"] and sparc_results[item]["domain_loss"] is not None
        for item in sparc_control_ids
    ) and all(
        xcop_results[item]["valid"] and xcop_results[item]["domain_loss"] is not None
        for item in xcop_control_ids
    )
    decisions = []
    for full in registered["full"]:
        class_id = full["full_class_id"]
        full_id = full["variant_id"]
        s_full = sparc_results[full_id]
        x_full = xcop_results[full_id]
        ablation_ids = [f"ablation:{class_id}:{name}" for name in ABLATION_IDS]
        ablations_valid = all(
            sparc_results[item]["valid"]
            and sparc_results[item]["domain_loss"] is not None
            and xcop_results[item]["valid"]
            and xcop_results[item]["domain_loss"] is not None
            for item in ablation_ids
        )
        checks = {
            "sparc_valid_and_parity_pass": s_full["valid"],
            "xcop_valid_and_parity_pass": x_full["valid"],
            "all_compared_controls_valid_and_parity_pass": controls_valid,
            "all_named_ablations_valid_and_parity_pass_in_both_domains": ablations_valid,
            "sparc_beats_newtonian": controls_valid
            and s_full["domain_loss"] is not None
            and _strictly_improves(
                s_full["domain_loss"],
                sparc_results["control:sparc:newtonian_baryons"]["domain_loss"],
                config,
            ),
            "sparc_beats_empirical_rar": controls_valid
            and s_full["domain_loss"] is not None
            and _strictly_improves(
                s_full["domain_loss"],
                sparc_results["control:sparc:empirical_rar"]["domain_loss"],
                config,
            ),
            "xcop_beats_gas_newtonian": controls_valid
            and x_full["domain_loss"] is not None
            and _strictly_improves(
                x_full["domain_loss"],
                xcop_results["control:xcop:gas_only_newtonian_shape"]["domain_loss"],
                config,
            ),
            "xcop_beats_uniform": controls_valid
            and x_full["domain_loss"] is not None
            and _strictly_improves(
                x_full["domain_loss"],
                xcop_results["control:xcop:uniform_acceleration_shape_control"]["domain_loss"],
                config,
            ),
            "sparc_beats_each_valid_ablation": ablations_valid
            and s_full["domain_loss"] is not None
            and all(
                _strictly_improves(
                    s_full["domain_loss"], sparc_results[item]["domain_loss"], config
                )
                for item in ablation_ids
            ),
            "xcop_beats_each_valid_ablation": ablations_valid
            and x_full["domain_loss"] is not None
            and all(
                _strictly_improves(x_full["domain_loss"], xcop_results[item]["domain_loss"], config)
                for item in ablation_ids
            ),
        }
        decisions.append(
            {
                "class_id": class_id,
                "eligible": all(checks.values()),
                "checks": checks,
                "formula_family_pruned": False,
                "single_object_terminal_veto": False,
                "constant_xcop_geometry_domain_switch_risk": full[
                    "constant_xcop_geometry_domain_switch_risk"
                ],
            }
        )
    return decisions


def _execute_scoring(
    config: Mapping[str, Any],
    registered: Mapping[str, Any],
    sparc: Sequence[Mapping[str, Any]],
    xcop: Sequence[Mapping[str, Any]],
    cp: Any,
) -> dict[str, Any]:
    variants = registered["variants"]
    sparc_predictors = np.concatenate([row["predictors"] for row in sparc])
    xcop_predictors = np.concatenate([row["predictors"] for row in xcop])
    sparc_gpu_predictors = cp.asarray(sparc_predictors, dtype=cp.float64)
    xcop_gpu_predictors = cp.asarray(xcop_predictors, dtype=cp.float64)
    sparc_results: dict[str, Any] = {}
    xcop_results: dict[str, Any] = {}
    parity_calls = 0

    for variant in variants:
        ast = variant["canonical_ast"]
        cpu_s = evaluate_ast(ast, sparc_predictors)
        gpu_s = cp.asnumpy(_evaluate_ast_xp(ast, sparc_gpu_predictors, cp))
        parity_s = _parity(cpu_s, gpu_s, config)
        score_s = _score_sparc_vector(cpu_s, sparc)
        score_s["cpu_gpu_parity"] = parity_s
        score_s["valid"] = score_s["valid"] and parity_s["pass"]
        sparc_results[variant["variant_id"]] = score_s
        parity_calls += 1

        cpu_x = evaluate_ast(ast, xcop_predictors)
        gpu_x = cp.asnumpy(_evaluate_ast_xp(ast, xcop_gpu_predictors, cp))
        parity_x = _parity(cpu_x, gpu_x, config)
        score_x = _score_xcop_shapes(_split_vector(cpu_x, xcop, "x"), xcop)
        score_x["cpu_gpu_parity"] = parity_x
        score_x["valid"] = score_x["valid"] and parity_x["pass"]
        xcop_results[variant["variant_id"]] = score_x
        parity_calls += 1

    # Domain-specific controls, with the same CPU/GPU parity and X-COP nuisance profile.
    x_source = sparc_predictors[:, 0]
    controls_s = {
        "control:sparc:newtonian_baryons": x_source,
        "control:sparc:empirical_rar": x_source / (-np.expm1(-np.sqrt(x_source))),
    }
    controls_s_gpu = {
        "control:sparc:newtonian_baryons": sparc_gpu_predictors[:, 0],
        "control:sparc:empirical_rar": sparc_gpu_predictors[:, 0]
        / (-cp.expm1(-cp.sqrt(sparc_gpu_predictors[:, 0]))),
    }
    for key, cpu in controls_s.items():
        gpu = cp.asnumpy(controls_s_gpu[key])
        score = _score_sparc_vector(cpu, sparc)
        score["cpu_gpu_parity"] = _parity(cpu, gpu, config)
        score["valid"] = score["valid"] and score["cpu_gpu_parity"]["pass"]
        sparc_results[key] = score
        parity_calls += 1

    cpu_gas = np.asarray(_gas_newtonian_shapes(xcop, np), dtype=np.float64)
    gpu_gas = cp.asnumpy(_gas_newtonian_shapes(xcop, cp))
    cpu_uniform = np.ones(len(xcop_predictors), dtype=np.float64)
    gpu_uniform = cp.asnumpy(cp.ones(len(xcop_predictors), dtype=cp.float64))
    for key, cpu, gpu in (
        ("control:xcop:gas_only_newtonian_shape", cpu_gas, gpu_gas),
        ("control:xcop:uniform_acceleration_shape_control", cpu_uniform, gpu_uniform),
    ):
        score = _score_xcop_shapes(_split_vector(cpu, xcop, "x"), xcop)
        score["cpu_gpu_parity"] = _parity(cpu, gpu, config)
        score["valid"] = score["valid"] and score["cpu_gpu_parity"]["pass"]
        xcop_results[key] = score
        parity_calls += 1

    decisions = _build_candidate_decisions(registered, sparc_results, xcop_results, config)
    eligible = [row["class_id"] for row in decisions if row["eligible"]]
    selected = eligible[0] if eligible else None
    if parity_calls != 484:
        raise BENDevelopmentExecutorV4Error("CPU/GPU parity accounting changed")
    all_scores = [*sparc_results.values(), *xcop_results.values()]
    xcop_scores = list(xcop_results.values())
    formula_batches = len(sparc_results) + len(xcop_results)
    sparc_formula_row_cells = len(sparc_results) * len(sparc_predictors)
    xcop_formula_row_cells = len(xcop_results) * len(xcop_predictors)
    actual_compute_accounting = {
        "cpu_formula_domain_batches": formula_batches,
        "gpu_formula_domain_batches": formula_batches,
        "cpu_gpu_parity_comparisons": parity_calls,
        "sparc_formula_row_cells_per_backend": sparc_formula_row_cells,
        "xcop_formula_row_cells_per_backend": xcop_formula_row_cells,
        "total_formula_row_cells_per_backend": sparc_formula_row_cells + xcop_formula_row_cells,
        "total_formula_row_cells_both_backends": 2
        * (sparc_formula_row_cells + xcop_formula_row_cells),
        "xcop_coupled_three_parameter_nuisance_fits": sum(
            row["nuisance_fits"] for row in xcop_scores
        ),
        "xcop_zeta_objective_evaluations": sum(
            row["zeta_objective_evaluations"] for row in xcop_scores
        ),
        "xcop_analytic_scale_solves": sum(row["analytic_scale_solves"] for row in xcop_scores),
        "object_score_reductions": sum(row["object_score_reductions"] for row in all_scores),
        "response_row_score_terms": sum(row["response_row_score_terms"] for row in all_scores),
        "candidate_selection_events": 1,
        "gpu_runtime_preflight_calls": 1,
        "gpu_runtime_preflight_probe_values": config["runtime_preflight"]["float64_probe_values"],
        "threshold_tuning_calls": 0,
        "formula_generation_calls": 0,
        "formula_repair_calls": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "api_spend_usd": 0.0,
    }
    ceiling = config["compute_ceiling"]
    exact_pairs = {
        "cpu_formula_domain_batches": "cpu_formula_domain_batches",
        "gpu_formula_domain_batches": "gpu_formula_domain_batches",
        "cpu_gpu_parity_comparisons": "cpu_gpu_parity_comparisons",
        "sparc_formula_row_cells_per_backend": "sparc_formula_row_cells_per_backend",
        "xcop_formula_row_cells_per_backend": "xcop_formula_row_cells_per_backend",
        "total_formula_row_cells_per_backend": "total_formula_row_cells_per_backend",
        "total_formula_row_cells_both_backends": "total_formula_row_cells_both_backends",
        "candidate_selection_events": "candidate_selection_events",
        "gpu_runtime_preflight_calls": "gpu_runtime_preflight_calls",
        "gpu_runtime_preflight_probe_values": "gpu_runtime_preflight_probe_values",
    }
    if any(actual_compute_accounting[key] != ceiling[value] for key, value in exact_pairs.items()):
        raise BENDevelopmentExecutorV4Error("exact compute accounting changed")
    bounded_pairs = {
        "xcop_coupled_three_parameter_nuisance_fits": (
            "xcop_coupled_three_parameter_nuisance_fits"
        ),
        "xcop_zeta_objective_evaluations": "xcop_zeta_objective_evaluations",
        "xcop_analytic_scale_solves": "xcop_analytic_scale_solves",
        "object_score_reductions": "maximum_object_score_reductions",
        "response_row_score_terms": "maximum_response_row_score_terms",
    }
    if any(actual_compute_accounting[key] > ceiling[value] for key, value in bounded_pairs.items()):
        raise BENDevelopmentExecutorV4Error("compute ceiling exceeded")
    return {
        "sparc": sparc_results,
        "xcop": xcop_results,
        "candidate_decisions": decisions,
        "eligible_class_ids": eligible,
        "selected_class_id": selected,
        "selection_events": 1,
        "counterexample_policy": {
            "single_counterexample_terminal": False,
            "counterexample_count_alone_terminal": False,
            "formula_families_pruned": 0,
            "all_scoped_failures_retained": True,
        },
        "actual_compute_accounting": actual_compute_accounting,
    }


def _preflight_gpu(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Validate the reference GPU in FP64 without opening scientific payloads."""

    runtime = config["runtime_preflight"]
    reference_cuda = config["runtime_environment_contract"]["cuda"]
    try:
        import cupy as cp
    except ImportError as error:  # pragma: no cover - depends on production host
        raise BENDevelopmentExecutorV4Error("CuPy is required for GPU preflight") from error
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        cuda_runtime = int(cp.cuda.runtime.runtimeGetVersion())
        cuda_driver = int(cp.cuda.runtime.driverGetVersion())
        if device_count != reference_cuda["device_count"]:
            raise BENDevelopmentExecutorV4Error("GPU preflight device count changed")
        device_id = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = properties["name"]
        device_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        compute_major = int(properties["major"])
        compute_minor = int(properties["minor"])
        if (
            device_name != reference_cuda["device_name"]
            or cuda_runtime != reference_cuda["runtime_version_integer"]
            or cuda_driver != reference_cuda["driver_version_integer"]
            or compute_major != reference_cuda["compute_capability_major"]
            or compute_minor != reference_cuda["compute_capability_minor"]
        ):
            raise BENDevelopmentExecutorV4Error("GPU preflight device identity changed")
        probe_count = int(runtime["float64_probe_values"])
        probe = cp.linspace(0.0, 1.0, probe_count, dtype=cp.float64)
        evaluated = cp.exp(-probe) + cp.sqrt(probe + 1.0)
        cp.cuda.get_current_stream().synchronize()
        host = cp.asnumpy(evaluated)
    except BENDevelopmentExecutorV4Error:
        raise
    except Exception as error:  # pragma: no cover - production-driver dependent
        raise BENDevelopmentExecutorV4Error("GPU FP64 runtime preflight failed") from error
    if (
        host.shape != (probe_count,)
        or host.dtype != np.dtype(np.float64)
        or not np.all(np.isfinite(host))
    ):
        raise BENDevelopmentExecutorV4Error("GPU FP64 probe output changed")
    return cp, {
        "pass": True,
        "scientific_payload_access": False,
        "device_count": device_count,
        "device_id": device_id,
        "device_name": device_name,
        "cuda_runtime_version_integer": cuda_runtime,
        "cuda_driver_version_integer": cuda_driver,
        "compute_capability_major": compute_major,
        "compute_capability_minor": compute_minor,
        "probe_dtype": "float64",
        "probe_values": probe_count,
        "gpu_runtime_preflight_calls": 1,
    }


def _render_result_floats(value: Any) -> Any:
    """Render binary64 only after every banded scientific comparison is complete."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise BENDevelopmentExecutorV4Error("nonfinite value cannot enter final result")
        return format(value, ".17e")
    if isinstance(value, Mapping):
        return {key: _render_result_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_result_floats(item) for item in value]
    if isinstance(value, tuple):
        return [_render_result_floats(item) for item in value]
    return value


def _result_float(value: Any, label: str) -> float:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"-?[0-9]\.[0-9]{17}e[+-][0-9]{2,3}", value) is None
    ):
        raise BENDevelopmentExecutorV4Error(f"{label} is not final binary64 rendering")
    parsed = float(value)
    if not math.isfinite(parsed) or format(parsed, ".17e") != value:
        raise BENDevelopmentExecutorV4Error(f"{label} binary64 round-trip changed")
    return parsed


def _validate_rendered_parity(
    parity: Mapping[str, Any], config: Mapping[str, Any], label: str
) -> bool:
    _strict_keys(parity, {"pass", "max_abs", "max_rel", "reason"}, f"{label} parity")
    if parity["max_abs"] is None or parity["max_rel"] is None:
        if parity["pass"] is not False or parity["reason"] not in {
            "shape_mismatch",
            "nonfinite",
        }:
            raise BENDevelopmentExecutorV4Error(f"{label} parity failure evidence changed")
        return False
    maximum_absolute = _result_float(parity["max_abs"], f"{label}.max_abs")
    maximum_relative = _result_float(parity["max_rel"], f"{label}.max_rel")
    selection = config["selection_contract"]
    expected = (
        maximum_absolute <= selection["parity_absolute_tolerance"]
        or maximum_relative <= selection["parity_relative_tolerance"]
    )
    if parity["pass"] is not expected or parity["reason"] != (None if expected else "tolerance"):
        raise BENDevelopmentExecutorV4Error(f"{label} parity adjudication changed")
    return expected


def _validate_rendered_sparc_score(
    score: Mapping[str, Any], config: Mapping[str, Any], label: str
) -> dict[str, Any]:
    _strict_keys(
        score,
        {
            "valid",
            "domain_loss",
            "per_object",
            "failures",
            "object_score_reductions",
            "response_row_score_terms",
            "cpu_gpu_parity",
        },
        f"{label} SPARC score",
    )
    parity_pass = _validate_rendered_parity(score["cpu_gpu_parity"], config, label)
    losses = []
    rows = 0
    for index, record in enumerate(score["per_object"]):
        _strict_keys(record, {"object", "rows", "loss", "terminal_veto"}, "SPARC object")
        if record["terminal_veto"] is not False or not isinstance(record["rows"], int):
            raise BENDevelopmentExecutorV4Error("SPARC object evidence changed")
        losses.append(_result_float(record["loss"], f"{label}.sparc_object_{index}"))
        rows += record["rows"]
    base_valid = (
        not score["failures"]
        and len(losses) == config["development_populations"]["sparc"]["objects"]
    )
    if losses:
        ledger = [
            {"object": record["object"], "rows": record["rows"]}
            for record in sorted(score["per_object"], key=lambda row: row["object"])
        ]
        sparc_contract = config["development_populations"]["sparc"]
        if (
            hashlib.sha256(canonical_json(ledger).encode()).hexdigest()
            != sparc_contract["admitted_score_name_row_ledger_sha256"]
            or canonical_sha256(sorted(record["object"] for record in ledger))
            != sparc_contract["admitted_score_name_root_sha256"]
        ):
            raise BENDevelopmentExecutorV4Error("SPARC result name/row ledger changed")
    expected_domain = float(np.mean(losses)) if base_valid else None
    observed_domain = (
        None
        if score["domain_loss"] is None
        else _result_float(score["domain_loss"], f"{label}.domain_loss")
    )
    if (
        observed_domain != expected_domain
        or score["valid"] is not (base_valid and parity_pass)
        or score["object_score_reductions"] != len(losses)
        or score["response_row_score_terms"] != rows
    ):
        raise BENDevelopmentExecutorV4Error(f"{label} SPARC aggregate adjudication changed")
    return {
        "valid": score["valid"],
        "domain_loss": observed_domain,
        "object_score_reductions": len(losses),
        "response_row_score_terms": rows,
    }


def _validate_rendered_xcop_score(
    score: Mapping[str, Any], config: Mapping[str, Any], label: str
) -> dict[str, Any]:
    _strict_keys(
        score,
        {
            "valid",
            "domain_loss",
            "valid_object_scores",
            "per_object",
            "failures",
            "formula_family_pruned",
            "nuisance_fits",
            "zeta_objective_evaluations",
            "analytic_scale_solves",
            "response_row_score_terms",
            "object_score_reductions",
            "cpu_gpu_parity",
        },
        f"{label} X-COP score",
    )
    parity_pass = _validate_rendered_parity(score["cpu_gpu_parity"], config, label)
    valid_losses = []
    flattened_failures = []
    nuisance_fits = 0
    zeta_evaluations = 0
    scale_solves = 0
    row_terms = 0
    reductions = 0
    object_keys = {
        "object",
        "response_rows",
        "valid",
        "loss",
        "nuisance",
        "H_pressure_response_nontrivial",
        "H_temperature_response_nontrivial",
        "jacobian_rank",
        "jacobian_condition",
        "failures",
        "terminal_veto",
        "nuisance_fit_executed",
        "zeta_objective_evaluations",
        "analytic_scale_solves",
        "response_row_score_terms",
        "object_score_reductions",
    }
    for index, record in enumerate(score["per_object"]):
        _strict_keys(record, object_keys, "X-COP object")
        if record["terminal_veto"] is not False or record["valid"] is not (not record["failures"]):
            raise BENDevelopmentExecutorV4Error("X-COP object validity evidence changed")
        for failure in record["failures"]:
            flattened_failures.append(f"{record['object']}:{failure}")
        loss = (
            None
            if record["loss"] is None
            else _result_float(record["loss"], f"{label}.xcop_object_{index}")
        )
        _strict_keys(record["nuisance"], {"zeta", "b_P", "b_T"}, "X-COP nuisance")
        for nuisance_name, nuisance_value in record["nuisance"].items():
            if nuisance_value is not None:
                _result_float(nuisance_value, f"{label}.{nuisance_name}_{index}")
        if record["jacobian_condition"] is not None:
            _result_float(record["jacobian_condition"], f"{label}.condition_{index}")
        if not isinstance(record["nuisance_fit_executed"], bool):
            raise BENDevelopmentExecutorV4Error("X-COP nuisance execution flag changed")
        if record["nuisance_fit_executed"]:
            if loss is None:
                expected_evidence = (0, 0, 0, 0)
            else:
                expected_evidence = (
                    4097,
                    8194,
                    record["response_rows"],
                    1,
                )
        else:
            expected_evidence = (0, 0, 0, 0)
        if (
            record["zeta_objective_evaluations"],
            record["analytic_scale_solves"],
            record["response_row_score_terms"],
            record["object_score_reductions"],
        ) != expected_evidence:
            raise BENDevelopmentExecutorV4Error("X-COP retained compute evidence changed")
        if record["valid"] and loss is not None:
            valid_losses.append(loss)
        nuisance_fits += int(record["nuisance_fit_executed"])
        zeta_evaluations += record["zeta_objective_evaluations"]
        scale_solves += record["analytic_scale_solves"]
        row_terms += record["response_row_score_terms"]
        reductions += record["object_score_reductions"]
    base_valid = len(valid_losses) == len(EXPECTED_XCOP_OBJECTS)
    if [record["object"] for record in score["per_object"]] != EXPECTED_XCOP_OBJECTS or sum(
        record["response_rows"] for record in score["per_object"]
    ) != config["development_populations"]["xcop"]["response_rows"]:
        raise BENDevelopmentExecutorV4Error("X-COP result object/row ledger changed")
    expected_domain = float(np.mean(valid_losses)) if base_valid else None
    observed_domain = (
        None
        if score["domain_loss"] is None
        else _result_float(score["domain_loss"], f"{label}.domain_loss")
    )
    if (
        score["failures"] != flattened_failures
        or score["formula_family_pruned"] is not False
        or observed_domain != expected_domain
        or score["valid"] is not (base_valid and parity_pass)
        or score["valid_object_scores"] != len(valid_losses)
        or score["nuisance_fits"] != nuisance_fits
        or score["zeta_objective_evaluations"] != zeta_evaluations
        or score["analytic_scale_solves"] != scale_solves
        or score["response_row_score_terms"] != row_terms
        or score["object_score_reductions"] != reductions
    ):
        raise BENDevelopmentExecutorV4Error(f"{label} X-COP aggregate adjudication changed")
    return {
        "valid": score["valid"],
        "domain_loss": observed_domain,
        "nuisance_fits": nuisance_fits,
        "zeta_objective_evaluations": zeta_evaluations,
        "analytic_scale_solves": scale_solves,
        "response_row_score_terms": row_terms,
        "object_score_reductions": reductions,
    }


def _validate_recorded_runtime_environment(
    receipt: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    _strict_keys(
        receipt,
        {
            "policy_id",
            "host",
            "packages",
            "numpy_build",
            "numpy_simd_dispatch",
            "svd_backend",
            "scientific_payload_access",
            "active_svd_threadpool",
            "gpu",
            "comparison_operator",
            "numerical_indifference_band",
            "reference_environment_validation_pass",
        },
        "runtime environment receipt",
    )
    contract = config["runtime_environment_contract"]
    expected_static = _static_runtime_receipt(config)
    for key, value in expected_static.items():
        if receipt[key] != value:
            raise BENDevelopmentExecutorV4Error("recorded static runtime changed")
    expected_active = {
        "user_api": contract["svd_threadpool"]["user_api"],
        "internal_api": contract["svd_threadpool"]["internal_api"],
        "prefix": contract["svd_threadpool"]["prefix"],
        "version": contract["svd_threadpool"]["version"],
        "threading_layer": contract["svd_threadpool"]["threading_layer"],
        "architecture": contract["svd_threadpool"]["architecture"],
        "library_filename": contract["svd_threadpool"]["library_filename"],
        "library_sha256": contract["svd_threadpool"]["library_sha256"],
        "num_threads": contract["svd_threadpool"]["required_num_threads_during_run"],
    }
    cuda = contract["cuda"]
    expected_gpu = {
        "pass": True,
        "scientific_payload_access": False,
        "device_count": cuda["device_count"],
        "device_id": 0,
        "device_name": cuda["device_name"],
        "cuda_runtime_version_integer": cuda["runtime_version_integer"],
        "cuda_driver_version_integer": cuda["driver_version_integer"],
        "compute_capability_major": cuda["compute_capability_major"],
        "compute_capability_minor": cuda["compute_capability_minor"],
        "probe_dtype": "float64",
        "probe_values": config["runtime_preflight"]["float64_probe_values"],
        "gpu_runtime_preflight_calls": 1,
    }
    if (
        receipt["active_svd_threadpool"] != expected_active
        or receipt["gpu"] != expected_gpu
        or receipt["comparison_operator"] != contract["comparison_operator"]
        or receipt["numerical_indifference_band"]
        != config["selection_contract"]["numerical_indifference_band"]
        or receipt["reference_environment_validation_pass"] is not True
    ):
        raise BENDevelopmentExecutorV4Error("recorded active numerical runtime changed")


def validate_result_document(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    registered: Mapping[str, Any],
    preflight: Mapping[str, Any],
    access_intent: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically adjudicate aggregation/selection from retained result evidence."""

    _strict_keys(
        access_intent,
        {
            "schema_version",
            "authorization_id",
            "authorization_file_sha256",
            "config_file_sha256",
            "preflight_receipt_content_sha256",
            "runtime_environment_receipt",
            "gpu_runtime_preflight",
            "payload_scope",
            "forbidden_scope",
            "payload_file_open_attempts_before_this_record",
            "payload_file_reads_completed_before_this_record",
            "scores_computed_before_this_record",
            "authorization_replay_allowed",
            "content_sha256",
        },
        "access intent",
    )
    if (
        access_intent["schema_version"] != ACCESS_SCHEMA
        or access_intent["content_sha256"] != content_sha256(access_intent)
        or access_intent["preflight_receipt_content_sha256"] != preflight["content_sha256"]
        or access_intent["payload_file_open_attempts_before_this_record"] != 0
        or access_intent["payload_file_reads_completed_before_this_record"] != 0
        or access_intent["scores_computed_before_this_record"] != 0
        or access_intent["authorization_replay_allowed"] is not False
        or access_intent["authorization_id"] != config["authorization_gate"]["authorization_id"]
        or access_intent["config_file_sha256"] != file_sha256(confined(CONFIG_PATH))
        or access_intent["forbidden_scope"] != config["development_populations"]["forbidden"]
        or access_intent["gpu_runtime_preflight"]
        != access_intent["runtime_environment_receipt"]["gpu"]
    ):
        raise BENDevelopmentExecutorV4Error("access intent adjudication changed")
    expected_payload_scope = {
        "sparc_files": 1,
        "sparc_container_parsed_objects": 175,
        "sparc_container_parsed_rows": 3391,
        "sparc_score_objects": 139,
        "sparc_score_rows": 2720,
        "xcop_files": 24,
        "xcop_objects": EXPECTED_XCOP_OBJECTS,
        "xcop_predictor_rows": 521,
        "xcop_response_rows": 184,
    }
    if access_intent["payload_scope"] != expected_payload_scope:
        raise BENDevelopmentExecutorV4Error("access intent payload scope changed")
    _validate_recorded_runtime_environment(access_intent["runtime_environment_receipt"], config)
    result_keys = {
        "schema_version",
        "status",
        "authorization_id",
        "authorization_file_sha256",
        "access_intent_content_sha256",
        "preflight_receipt_content_sha256",
        "candidate_registry_content_sha256",
        "registered_formula_ledger",
        "candidate_and_ablation_accounting",
        "development_populations",
        "runtime_environment_receipt",
        "gpu_runtime_preflight",
        "scores",
        "claim_ceiling",
        "claims",
        "content_sha256",
    }
    _strict_keys(result, result_keys, "result")
    if (
        result["schema_version"] != RESULT_SCHEMA
        or result["status"] != "development_only_scoring_complete"
        or result["content_sha256"] != content_sha256(result)
        or result["preflight_receipt_content_sha256"] != preflight["content_sha256"]
        or result["access_intent_content_sha256"] != access_intent["content_sha256"]
        or result["candidate_registry_content_sha256"]
        != config["candidate_registry"]["registry_content_sha256"]
        or result["candidate_and_ablation_accounting"] != registered["accounting"]
        or result["claim_ceiling"] != config["claim_ceiling"]
        or result["runtime_environment_receipt"] != access_intent["runtime_environment_receipt"]
        or result["gpu_runtime_preflight"] != access_intent["gpu_runtime_preflight"]
        or result["authorization_id"] != access_intent["authorization_id"]
        or result["authorization_file_sha256"] != access_intent["authorization_file_sha256"]
    ):
        raise BENDevelopmentExecutorV4Error("result identity or frozen claim changed")
    expected_ledger = [
        {key: value for key, value in row.items() if key != "canonical_ast"}
        for row in registered["variants"]
    ]
    if result["registered_formula_ledger"] != expected_ledger:
        raise BENDevelopmentExecutorV4Error("result formula identity ledger changed")
    populations = result["development_populations"]
    _strict_keys(populations, {"sparc", "xcop"}, "result populations")
    sparc_population = populations["sparc"]
    xcop_population = populations["xcop"]
    _strict_keys(
        sparc_population,
        {
            "payload_file_open_attempts",
            "payload_file_reads_completed",
            "parsed_container_objects",
            "parsed_container_rows",
            "scored_objects",
            "scored_rows",
            "split_confirmation_objects",
            "split_exploration_before_admission_objects",
            "admitted_score_name_root_sha256",
            "admitted_score_name_row_ledger_sha256",
        },
        "result SPARC population",
    )
    _strict_keys(
        xcop_population,
        {
            "payload_file_open_attempts",
            "payload_file_reads_completed",
            "objects_parsed",
            "predictor_rows_parsed_and_scored",
            "response_rows_parsed_and_scored",
        },
        "result X-COP population",
    )
    sparc_contract = config["development_populations"]["sparc"]
    if sparc_population != {
        "payload_file_open_attempts": 1,
        "payload_file_reads_completed": 1,
        "parsed_container_objects": 175,
        "parsed_container_rows": 3391,
        "scored_objects": 139,
        "scored_rows": 2720,
        "split_confirmation_objects": 35,
        "split_exploration_before_admission_objects": 140,
        "admitted_score_name_root_sha256": sparc_contract["admitted_score_name_root_sha256"],
        "admitted_score_name_row_ledger_sha256": sparc_contract[
            "admitted_score_name_row_ledger_sha256"
        ],
    } or xcop_population != {
        "payload_file_open_attempts": 24,
        "payload_file_reads_completed": 24,
        "objects_parsed": 8,
        "predictor_rows_parsed_and_scored": 521,
        "response_rows_parsed_and_scored": 184,
    }:
        raise BENDevelopmentExecutorV4Error("result development population changed")
    scores = result["scores"]
    _strict_keys(
        scores,
        {
            "sparc",
            "xcop",
            "candidate_decisions",
            "eligible_class_ids",
            "selected_class_id",
            "selection_events",
            "counterexample_policy",
            "actual_compute_accounting",
        },
        "result scores",
    )
    variant_ids = {row["variant_id"] for row in registered["variants"]}
    expected_sparc_ids = variant_ids | {
        "control:sparc:newtonian_baryons",
        "control:sparc:empirical_rar",
    }
    expected_xcop_ids = variant_ids | {
        "control:xcop:gas_only_newtonian_shape",
        "control:xcop:uniform_acceleration_shape_control",
    }
    if set(scores["sparc"]) != expected_sparc_ids or set(scores["xcop"]) != expected_xcop_ids:
        raise BENDevelopmentExecutorV4Error("result score identity ledger changed")
    sparc_adjudicated = {
        key: _validate_rendered_sparc_score(value, config, key)
        for key, value in scores["sparc"].items()
    }
    xcop_adjudicated = {
        key: _validate_rendered_xcop_score(value, config, key)
        for key, value in scores["xcop"].items()
    }
    decisions = _build_candidate_decisions(registered, sparc_adjudicated, xcop_adjudicated, config)
    eligible = [row["class_id"] for row in decisions if row["eligible"]]
    selected = eligible[0] if eligible else None
    if (
        scores["candidate_decisions"] != decisions
        or scores["eligible_class_ids"] != eligible
        or scores["selected_class_id"] != selected
        or scores["selection_events"] != 1
    ):
        raise BENDevelopmentExecutorV4Error("result selection adjudication changed")
    expected_counterexample_policy = {
        "single_counterexample_terminal": False,
        "counterexample_count_alone_terminal": False,
        "formula_families_pruned": 0,
        "all_scoped_failures_retained": True,
    }
    expected_claims = {
        "development_only_score": True,
        "fresh_confirmation": False,
        "absolute_cluster_prediction": False,
        "full_covariance": False,
        "historical_novelty_established": False,
        "dark_matter_eliminated": False,
        "alternative_to_gr_established": False,
        "publication_ready": False,
        "formula_family_pruned": False,
        "raw_per_object_losses_independently_recomputed": False,
    }
    if (
        scores["counterexample_policy"] != expected_counterexample_policy
        or result["claims"] != expected_claims
    ):
        raise BENDevelopmentExecutorV4Error("result counterexample or claim evidence changed")
    all_adjudicated = [*sparc_adjudicated.values(), *xcop_adjudicated.values()]
    expected_accounting = {
        "cpu_formula_domain_batches": 484,
        "gpu_formula_domain_batches": 484,
        "cpu_gpu_parity_comparisons": 484,
        "sparc_formula_row_cells_per_backend": len(scores["sparc"])
        * populations["sparc"]["scored_rows"],
        "xcop_formula_row_cells_per_backend": len(scores["xcop"])
        * populations["xcop"]["predictor_rows_parsed_and_scored"],
        "total_formula_row_cells_per_backend": 784322,
        "total_formula_row_cells_both_backends": 1568644,
        "xcop_coupled_three_parameter_nuisance_fits": sum(
            row["nuisance_fits"] for row in xcop_adjudicated.values()
        ),
        "xcop_zeta_objective_evaluations": sum(
            row["zeta_objective_evaluations"] for row in xcop_adjudicated.values()
        ),
        "xcop_analytic_scale_solves": sum(
            row["analytic_scale_solves"] for row in xcop_adjudicated.values()
        ),
        "object_score_reductions": sum(row["object_score_reductions"] for row in all_adjudicated),
        "response_row_score_terms": sum(row["response_row_score_terms"] for row in all_adjudicated),
        "candidate_selection_events": 1,
        "gpu_runtime_preflight_calls": 1,
        "gpu_runtime_preflight_probe_values": 16,
        "threshold_tuning_calls": 0,
        "formula_generation_calls": 0,
        "formula_repair_calls": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "api_spend_usd": 0.0,
        "payload_file_open_attempts": populations["sparc"]["payload_file_open_attempts"]
        + populations["xcop"]["payload_file_open_attempts"],
        "payload_file_reads_completed": populations["sparc"]["payload_file_reads_completed"]
        + populations["xcop"]["payload_file_reads_completed"],
        "result_validation_events": 1,
    }
    if scores["actual_compute_accounting"] != _render_result_floats(expected_accounting):
        raise BENDevelopmentExecutorV4Error("result actual compute accounting changed")
    return {
        "validation_pass": True,
        "result_validation_events": 1,
        "formula_domain_scores_adjudicated": len(scores["sparc"]) + len(scores["xcop"]),
        "parity_decisions_recomputed": len(scores["sparc"]) + len(scores["xcop"]),
        "domain_aggregates_recomputed": len(scores["sparc"]) + len(scores["xcop"]),
        "candidate_decisions_recomputed": len(decisions),
        "eligible_class_ids": eligible,
        "selected_class_id": selected,
        "actual_compute_accounting": _render_result_floats(expected_accounting),
        "raw_per_object_losses_independently_recomputed": False,
    }


def build_result_adjudication(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    registered: Mapping[str, Any],
    preflight: Mapping[str, Any],
    access_intent: Mapping[str, Any],
    validation_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = (
        dict(validation_evidence)
        if validation_evidence is not None
        else validate_result_document(result, config, registered, preflight, access_intent)
    )
    body: dict[str, Any] = {
        "schema_version": ADJUDICATION_SCHEMA,
        "status": "PASS_DETERMINISTIC_POST_RUN_ADJUDICATION",
        "authorization_id": access_intent["authorization_id"],
        "authorization_file_sha256": access_intent["authorization_file_sha256"],
        "result_path": RESULT_PATH.as_posix(),
        "result_file_sha256": file_sha256(confined(RESULT_PATH)),
        "result_content_sha256": result["content_sha256"],
        "config_file_sha256": file_sha256(confined(CONFIG_PATH)),
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "registered_formula_identity_ledger_sha256": preflight[
            "registered_formula_identity_ledger_sha256"
        ],
        "access_intent_content_sha256": access_intent["content_sha256"],
        "validation_evidence": evidence,
        "claim_ceiling": config["claim_ceiling"],
        "scientific_payload_files_opened_by_adjudicator": 0,
        "authorization_replay_allowed": False,
    }
    body["content_sha256"] = content_sha256(body)
    return body


def _build_terminal_success_marker(
    authorization: Mapping[str, Any],
    authorization_path: Path,
    access_intent: Mapping[str, Any],
    result: Mapping[str, Any],
    adjudication: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": TERMINAL_SUCCESS_SCHEMA,
        "status": "TERMINAL_SUCCESS_AFTER_RUNTIME_RESTORATION",
        "authorization_id": authorization["authorization_id"],
        "authorization_file_sha256": file_sha256(confined(authorization_path)),
        "access_intent_content_sha256": access_intent["content_sha256"],
        "result_content_sha256": result["content_sha256"],
        "adjudication_content_sha256": adjudication["content_sha256"],
        "runtime_restoration_completed": True,
        "access_failure_absent_before_publication": not confined(ACCESS_FAILURE_PATH).exists(),
        "terminal_state_lock_held_during_publication": True,
        "authorization_replay_allowed": False,
    }
    body["content_sha256"] = content_sha256(body)
    return body


def _new_access_accounting() -> dict[str, int]:
    return {
        "payload_file_open_attempts": 0,
        "payload_file_reads_completed": 0,
        "sparc_file_open_attempts": 0,
        "sparc_file_reads_completed": 0,
        "sparc_container_objects_parsed": 0,
        "sparc_container_rows_parsed": 0,
        "sparc_score_objects_loaded": 0,
        "sparc_score_rows_loaded": 0,
        "xcop_file_open_attempts": 0,
        "xcop_file_reads_completed": 0,
        "xcop_objects_loaded": 0,
        "xcop_predictor_rows_loaded": 0,
        "xcop_response_rows_loaded": 0,
        "formula_scoring_started": 0,
        "formula_scoring_completed": 0,
        "result_validations_completed": 0,
        "result_files_committed": 0,
        "adjudication_receipts_committed": 0,
    }


def _write_phase_receipt(
    authorization: Mapping[str, Any],
    authorization_path: Path,
    preflight: Mapping[str, Any],
    access_intent: Mapping[str, Any],
    phase: str,
    access_accounting: Mapping[str, int],
) -> None:
    if phase not in PHASES:
        raise BENDevelopmentExecutorV4Error("unknown durable execution phase")
    ordinal = PHASES.index(phase) + 1
    path = PHASE_RECEIPT_DIRECTORY / f"phase-{ordinal:02d}-{phase}.json"
    body: dict[str, Any] = {
        "schema_version": PHASE_RECEIPT_SCHEMA,
        "status": "DURABLE_EXECUTION_PHASE_COMPLETED",
        "phase_ordinal": ordinal,
        "completed_phase": phase,
        "authorization_id": authorization["authorization_id"],
        "authorization_file_sha256": file_sha256(confined(authorization_path)),
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "access_intent_content_sha256": access_intent["content_sha256"],
        "actual_access_accounting": dict(access_accounting),
        "result_exists": confined(RESULT_PATH).exists(),
        "adjudication_exists": confined(ADJUDICATION_PATH).exists(),
        "authorization_replay_allowed": False,
    }
    body["content_sha256"] = content_sha256(body)
    _atomic_no_clobber(path, body)


def _failure_classification(error: BaseException, failure_operation: str) -> tuple[str, str]:
    """Return only fixed, value-free failure labels safe for durable receipts."""

    if failure_operation == "runtime_restoration":
        return "RUNTIME_RESTORATION_FAILURE", "runtime_restoration"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "EXTERNAL_INTERRUPT", "interruption"
    if isinstance(error, MemoryError):
        return "MEMORY_EXHAUSTION", "resource"
    if isinstance(error, OSError):
        if failure_operation == "access_intent_publication":
            return "ACCESS_INTENT_PUBLICATION_FAILURE", "access_intent_publication"
        if failure_operation in {"sparc_payload_read", "xcop_payload_read"}:
            return "PAYLOAD_READ_FAILURE", "payload_read"
        if failure_operation == "phase_receipt_publication":
            return "PHASE_RECEIPT_PUBLICATION_FAILURE", "phase_receipt_publication"
        if failure_operation == "result_publication":
            return "RESULT_PUBLICATION_FAILURE", "result_publication"
        if failure_operation == "adjudication_publication":
            return "ADJUDICATION_PUBLICATION_FAILURE", "adjudication_publication"
        if failure_operation == "runtime_restoration":
            return "RUNTIME_RESTORATION_FAILURE", "runtime_restoration"
        if failure_operation == "terminal_marker_publication":
            return "TERMINAL_MARKER_PUBLICATION_FAILURE", "terminal_marker_publication"
        if failure_operation in {"result_validation", "adjudication_build"}:
            return "CONTRACT_OR_SCORE_VALIDATION_FAILURE", "validation"
        if failure_operation == "formula_scoring":
            return "DETERMINISTIC_EXECUTION_FAILURE", "execution"
        return "OUTPUT_PUBLICATION_FAILURE", "output_publication"
    if isinstance(error, BENDevelopmentExecutorV4Error):
        if failure_operation == "access_intent_publication":
            return "ACCESS_INTENT_PUBLICATION_FAILURE", "access_intent_publication"
        if failure_operation == "phase_receipt_publication":
            return "PHASE_RECEIPT_PUBLICATION_FAILURE", "phase_receipt_publication"
        if failure_operation == "result_publication":
            return "RESULT_PUBLICATION_FAILURE", "result_publication"
        if failure_operation == "adjudication_publication":
            return "ADJUDICATION_PUBLICATION_FAILURE", "adjudication_publication"
        if failure_operation == "runtime_restoration":
            return "RUNTIME_RESTORATION_FAILURE", "runtime_restoration"
        if failure_operation == "terminal_marker_publication":
            return "TERMINAL_MARKER_PUBLICATION_FAILURE", "terminal_marker_publication"
        return "CONTRACT_OR_SCORE_VALIDATION_FAILURE", "validation"
    if isinstance(error, Exception):
        return "DETERMINISTIC_EXECUTION_FAILURE", "execution"
    return "UNCLASSIFIED_EXECUTION_FAILURE", "internal"


def _write_access_failure_receipt(
    authorization: Mapping[str, Any],
    authorization_path: Path,
    preflight: Mapping[str, Any],
    access_intent: Mapping[str, Any] | None,
    phase: str | None,
    access_accounting: Mapping[str, int],
    failure_operation: str,
    error: BaseException,
) -> None:
    allowed_operations = {
        "access_intent_publication",
        "sparc_payload_read",
        "xcop_payload_read",
        "phase_receipt_publication",
        "formula_scoring",
        "result_validation",
        "result_publication",
        "adjudication_build",
        "adjudication_publication",
        "runtime_restoration",
        "terminal_marker_publication",
    }
    if failure_operation not in allowed_operations:
        raise BENDevelopmentExecutorV4Error("unknown sanitized failure operation")
    error_code, error_class = _failure_classification(error, failure_operation)
    body: dict[str, Any] = {
        "schema_version": ACCESS_FAILURE_SCHEMA,
        "status": "TERMINAL_STATE_FAILURE_SUCCESSOR_REQUIRED",
        "authorization_id": authorization["authorization_id"],
        "authorization_file_sha256": file_sha256(confined(authorization_path)),
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "access_intent_content_sha256": (
            None if access_intent is None else access_intent["content_sha256"]
        ),
        "access_intent_exists": confined(ACCESS_INTENT_PATH).exists(),
        "last_completed_phase": phase,
        "actual_access_accounting": dict(access_accounting),
        "failure_operation": failure_operation,
        "error_code": error_code,
        "error_class": error_class,
        "result_exists": confined(RESULT_PATH).exists(),
        "adjudication_exists": confined(ADJUDICATION_PATH).exists(),
        "authorization_replay_allowed": False,
        "successor_contract_required": True,
    }
    body["content_sha256"] = content_sha256(body)
    _atomic_no_clobber(ACCESS_FAILURE_PATH, body)


def execute(authorization_path: Path) -> Path:
    # Everything through validate_authorization is metadata-only by construction.
    config = load_config()
    loaded = validate_bound_metadata(config)
    preflight = validate_preflight(config)
    authorization = validate_authorization(authorization_path, config, preflight)
    registered = build_registered_variants(loaded["synthetic_receipt"]["candidate_registry"])
    inventory = _xcop_inventory(config, loaded["xcop_source_receipt"])
    if any(
        confined(path).exists()
        for path in (
            ACCESS_INTENT_PATH,
            RESULT_PATH,
            ADJUDICATION_PATH,
            ACCESS_FAILURE_PATH,
            PHASE_RECEIPT_DIRECTORY,
            TERMINAL_SUCCESS_PATH,
        )
    ):
        raise BENDevelopmentExecutorV4Error("authorization replay or output overwrite refused")
    static_runtime = _static_runtime_receipt(config)
    state_lock = _acquire_terminal_state_lock(create=True)
    limiter = None
    access_intent: dict[str, Any] | None = None
    access_accounting = _new_access_accounting()
    phase: str | None = None
    failure_operation = "runtime_preflight"
    pending_error: BaseException | None = None
    pending_traceback: Any = None
    pending_operation = failure_operation
    result: dict[str, Any] | None = None
    adjudication: dict[str, Any] | None = None
    successful_result_path: Path | None = None
    try:
        limiter, svd_runtime = _activate_reference_svd_runtime(config)
        # Host, SVD backend, CUDA/driver, and hardware all pass before the one-shot intent.
        cp, gpu_preflight = _preflight_gpu(config)
        runtime_environment_receipt = {
            **static_runtime,
            "active_svd_threadpool": svd_runtime,
            "gpu": gpu_preflight,
            "comparison_operator": config["runtime_environment_contract"]["comparison_operator"],
            "numerical_indifference_band": config["selection_contract"][
                "numerical_indifference_band"
            ],
            "reference_environment_validation_pass": True,
        }
        access_intent = {
            "schema_version": ACCESS_SCHEMA,
            "authorization_id": authorization["authorization_id"],
            "authorization_file_sha256": file_sha256(confined(authorization_path)),
            "config_file_sha256": file_sha256(confined(CONFIG_PATH)),
            "preflight_receipt_content_sha256": preflight["content_sha256"],
            "runtime_environment_receipt": runtime_environment_receipt,
            "gpu_runtime_preflight": gpu_preflight,
            "payload_scope": {
                "sparc_files": 1,
                "sparc_container_parsed_objects": 175,
                "sparc_container_parsed_rows": 3391,
                "sparc_score_objects": 139,
                "sparc_score_rows": 2720,
                "xcop_files": 24,
                "xcop_objects": EXPECTED_XCOP_OBJECTS,
                "xcop_predictor_rows": 521,
                "xcop_response_rows": 184,
            },
            "forbidden_scope": config["development_populations"]["forbidden"],
            "payload_file_open_attempts_before_this_record": 0,
            "payload_file_reads_completed_before_this_record": 0,
            "scores_computed_before_this_record": 0,
            "authorization_replay_allowed": False,
        }
        access_intent["content_sha256"] = content_sha256(access_intent)

        def record_phase() -> None:
            _write_phase_receipt(
                authorization,
                authorization_path,
                preflight,
                access_intent,
                phase,
                access_accounting,
            )

        try:
            failure_operation = "access_intent_publication"
            _atomic_no_clobber(ACCESS_INTENT_PATH, access_intent)
            phase = "access_intent_committed"
            failure_operation = "phase_receipt_publication"
            record_phase()
            # The first target access in this module occurs here, after durable intent.
            failure_operation = "sparc_payload_read"
            sparc = load_sparc_development(config, access_accounting)
            phase = "sparc_development_loaded"
            failure_operation = "phase_receipt_publication"
            record_phase()
            failure_operation = "xcop_payload_read"
            xcop = load_xcop_development(config, inventory, access_accounting)
            phase = "xcop_development_loaded"
            failure_operation = "phase_receipt_publication"
            record_phase()
            access_accounting["formula_scoring_started"] = 1
            failure_operation = "formula_scoring"
            scores = _execute_scoring(config, registered, sparc["objects"], xcop["objects"], cp)
            access_accounting["formula_scoring_completed"] = 1
            phase = "formula_scoring_completed"
            failure_operation = "phase_receipt_publication"
            record_phase()
            failure_operation = "result_validation"
            payload_file_open_attempts = access_accounting["payload_file_open_attempts"]
            payload_file_reads_completed = access_accounting["payload_file_reads_completed"]
            if (
                payload_file_open_attempts
                > config["compute_ceiling"]["maximum_payload_file_open_attempts"]
                or payload_file_reads_completed
                > config["compute_ceiling"]["maximum_payload_file_successful_reads"]
            ):
                raise BENDevelopmentExecutorV4Error("payload I/O ceiling exceeded")
            scores["actual_compute_accounting"]["payload_file_open_attempts"] = (
                payload_file_open_attempts
            )
            scores["actual_compute_accounting"]["payload_file_reads_completed"] = (
                payload_file_reads_completed
            )
            scores["actual_compute_accounting"]["result_validation_events"] = 1
            public_variants = [
                {key: value for key, value in row.items() if key != "canonical_ast"}
                for row in registered["variants"]
            ]
            result_raw: dict[str, Any] = {
                "schema_version": RESULT_SCHEMA,
                "status": "development_only_scoring_complete",
                "authorization_id": authorization["authorization_id"],
                "authorization_file_sha256": file_sha256(confined(authorization_path)),
                "access_intent_content_sha256": access_intent["content_sha256"],
                "preflight_receipt_content_sha256": preflight["content_sha256"],
                "candidate_registry_content_sha256": config["candidate_registry"][
                    "registry_content_sha256"
                ],
                "registered_formula_ledger": public_variants,
                "candidate_and_ablation_accounting": registered["accounting"],
                "development_populations": {
                    "sparc": {key: value for key, value in sparc.items() if key != "objects"},
                    "xcop": {key: value for key, value in xcop.items() if key != "objects"},
                },
                "runtime_environment_receipt": runtime_environment_receipt,
                "gpu_runtime_preflight": gpu_preflight,
                "scores": scores,
                "claim_ceiling": config["claim_ceiling"],
                "claims": {
                    "development_only_score": True,
                    "fresh_confirmation": False,
                    "absolute_cluster_prediction": False,
                    "full_covariance": False,
                    "historical_novelty_established": False,
                    "dark_matter_eliminated": False,
                    "alternative_to_gr_established": False,
                    "publication_ready": False,
                    "formula_family_pruned": False,
                    "raw_per_object_losses_independently_recomputed": False,
                },
            }
            result = _render_result_floats(result_raw)
            result["content_sha256"] = content_sha256(result)
            validation = validate_result_document(
                result, config, registered, preflight, access_intent
            )
            access_accounting["result_validations_completed"] = 1
            phase = "result_adjudication_passed_in_memory"
            failure_operation = "phase_receipt_publication"
            record_phase()
            failure_operation = "result_publication"
            _atomic_no_clobber(RESULT_PATH, result)
            access_accounting["result_files_committed"] = 1
            phase = "result_committed"
            failure_operation = "phase_receipt_publication"
            record_phase()
            failure_operation = "adjudication_build"
            adjudication = build_result_adjudication(
                result,
                config,
                registered,
                preflight,
                access_intent,
                validation_evidence=validation,
            )
            failure_operation = "adjudication_publication"
            _atomic_no_clobber(ADJUDICATION_PATH, adjudication)
            access_accounting["adjudication_receipts_committed"] = 1
            phase = "adjudication_committed"
            failure_operation = "phase_receipt_publication"
            record_phase()
            successful_result_path = confined(RESULT_PATH)
        except BaseException:  # noqa: TRY203 - outer terminal state records the error
            raise
    except BaseException as error:  # noqa: BLE001 - includes interrupts after intent link
        pending_error = error
        pending_traceback = error.__traceback__
        pending_operation = failure_operation
    finally:
        if limiter is not None:
            try:
                limiter.restore_original_limits()
            except BaseException as error:  # noqa: BLE001 - restoration is terminal evidence
                pending_error = error
                pending_traceback = error.__traceback__
                pending_operation = "runtime_restoration"

    if pending_error is not None:
        try:
            if confined(ACCESS_INTENT_PATH).exists() or pending_operation == "runtime_restoration":
                _write_access_failure_receipt(
                    authorization,
                    authorization_path,
                    preflight,
                    access_intent,
                    phase,
                    access_accounting,
                    pending_operation,
                    pending_error,
                )
        finally:
            _release_terminal_state_lock(state_lock)
        raise pending_error.with_traceback(pending_traceback)

    failure_operation = "terminal_marker_publication"
    try:
        if (
            access_intent is None
            or result is None
            or adjudication is None
            or successful_result_path is None
        ):
            raise BENDevelopmentExecutorV4Error("terminal success evidence is incomplete")
        terminal_success = _build_terminal_success_marker(
            authorization,
            authorization_path,
            access_intent,
            result,
            adjudication,
        )
        _atomic_no_clobber(TERMINAL_SUCCESS_PATH, terminal_success)
    except BaseException as error:
        try:
            _write_access_failure_receipt(
                authorization,
                authorization_path,
                preflight,
                access_intent,
                phase,
                access_accounting,
                failure_operation,
                error,
            )
        finally:
            _release_terminal_state_lock(state_lock)
        raise
    _release_terminal_state_lock(state_lock)
    return successful_result_path


def check_preflight() -> dict[str, Any]:
    config = load_config()
    validate_bound_metadata(config)
    receipt = validate_preflight(config)
    authorization = read_json(AUTHORIZATION_PATH)
    expected = authorization_template(config, receipt)
    if authorization != expected:
        raise BENDevelopmentExecutorV4Error(
            "current authorization is not the sealed false template"
        )
    if any(
        confined(path).exists()
        for path in (
            ACCESS_INTENT_PATH,
            RESULT_PATH,
            ADJUDICATION_PATH,
            ACCESS_FAILURE_PATH,
            PHASE_RECEIPT_DIRECTORY,
            TERMINAL_SUCCESS_PATH,
            TERMINAL_STATE_LOCK_PATH,
        )
    ):
        raise BENDevelopmentExecutorV4Error("preflight is no longer zero-access")
    return {
        "ok": True,
        "decision": DECISION,
        "authorization_id": expected["authorization_id"],
        "authorized": False,
        "target_files_opened": 0,
        "scores_computed": 0,
    }


def _validate_durable_phase_receipts(access_intent: Mapping[str, Any]) -> int:
    previous = _new_access_accounting()
    for ordinal, phase in enumerate(PHASES, start=1):
        path = PHASE_RECEIPT_DIRECTORY / f"phase-{ordinal:02d}-{phase}.json"
        receipt = read_json(path)
        _strict_keys(
            receipt,
            {
                "schema_version",
                "status",
                "phase_ordinal",
                "completed_phase",
                "authorization_id",
                "authorization_file_sha256",
                "preflight_receipt_content_sha256",
                "access_intent_content_sha256",
                "actual_access_accounting",
                "result_exists",
                "adjudication_exists",
                "authorization_replay_allowed",
                "content_sha256",
            },
            "durable phase receipt",
        )
        accounting = receipt["actual_access_accounting"]
        if (
            receipt["schema_version"] != PHASE_RECEIPT_SCHEMA
            or receipt["status"] != "DURABLE_EXECUTION_PHASE_COMPLETED"
            or receipt["phase_ordinal"] != ordinal
            or receipt["completed_phase"] != phase
            or receipt["authorization_id"] != access_intent["authorization_id"]
            or receipt["authorization_file_sha256"] != access_intent["authorization_file_sha256"]
            or receipt["preflight_receipt_content_sha256"]
            != access_intent["preflight_receipt_content_sha256"]
            or receipt["access_intent_content_sha256"] != access_intent["content_sha256"]
            or receipt["authorization_replay_allowed"] is not False
            or receipt["content_sha256"] != content_sha256(receipt)
            or set(accounting) != set(previous)
            or any(accounting[key] < previous[key] for key in previous)
            or receipt["result_exists"] is not (ordinal >= 6)
            or receipt["adjudication_exists"] is not (ordinal >= 7)
        ):
            raise BENDevelopmentExecutorV4Error("durable execution phase evidence changed")
        previous = dict(accounting)
    if previous != {
        "payload_file_open_attempts": 25,
        "payload_file_reads_completed": 25,
        "sparc_file_open_attempts": 1,
        "sparc_file_reads_completed": 1,
        "sparc_container_objects_parsed": 175,
        "sparc_container_rows_parsed": 3391,
        "sparc_score_objects_loaded": 139,
        "sparc_score_rows_loaded": 2720,
        "xcop_file_open_attempts": 24,
        "xcop_file_reads_completed": 24,
        "xcop_objects_loaded": 8,
        "xcop_predictor_rows_loaded": 521,
        "xcop_response_rows_loaded": 184,
        "formula_scoring_started": 1,
        "formula_scoring_completed": 1,
        "result_validations_completed": 1,
        "result_files_committed": 1,
        "adjudication_receipts_committed": 1,
    }:
        raise BENDevelopmentExecutorV4Error("final durable execution accounting changed")
    return len(PHASES)


def _check_result_under_terminal_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the post-run adjudication without opening any scientific payload."""

    if confined(ACCESS_FAILURE_PATH).exists():
        raise BENDevelopmentExecutorV4Error(
            "access-failure receipt exists; successful adjudication is forbidden"
        )
    terminal_success = read_json(TERMINAL_SUCCESS_PATH)
    _strict_keys(
        terminal_success,
        {
            "schema_version",
            "status",
            "authorization_id",
            "authorization_file_sha256",
            "access_intent_content_sha256",
            "result_content_sha256",
            "adjudication_content_sha256",
            "runtime_restoration_completed",
            "access_failure_absent_before_publication",
            "terminal_state_lock_held_during_publication",
            "authorization_replay_allowed",
            "content_sha256",
        },
        "terminal success marker",
    )
    if (
        terminal_success["schema_version"] != TERMINAL_SUCCESS_SCHEMA
        or terminal_success["status"] != "TERMINAL_SUCCESS_AFTER_RUNTIME_RESTORATION"
        or terminal_success["runtime_restoration_completed"] is not True
        or terminal_success["access_failure_absent_before_publication"] is not True
        or terminal_success["terminal_state_lock_held_during_publication"] is not True
        or terminal_success["authorization_replay_allowed"] is not False
        or terminal_success["content_sha256"] != content_sha256(terminal_success)
    ):
        raise BENDevelopmentExecutorV4Error("terminal success marker changed")
    loaded = validate_bound_metadata(config)
    preflight = validate_preflight(config)
    authorization = validate_authorization(AUTHORIZATION_PATH, config, preflight)
    authorization_file_sha256 = file_sha256(confined(AUTHORIZATION_PATH))
    registered = build_registered_variants(loaded["synthetic_receipt"]["candidate_registry"])
    access_intent = read_json(ACCESS_INTENT_PATH)
    result = read_json(RESULT_PATH)
    stored = read_json(ADJUDICATION_PATH)
    if (
        access_intent.get("authorization_id") != authorization["authorization_id"]
        or result.get("authorization_id") != authorization["authorization_id"]
        or stored.get("authorization_id") != authorization["authorization_id"]
        or access_intent.get("authorization_file_sha256") != authorization_file_sha256
        or result.get("authorization_file_sha256") != authorization_file_sha256
        or stored.get("authorization_file_sha256") != authorization_file_sha256
    ):
        raise BENDevelopmentExecutorV4Error(
            "approved authorization file identity changed after execution"
        )
    expected = build_result_adjudication(result, config, registered, preflight, access_intent)
    _strict_keys(stored, set(expected), "result adjudication")
    if stored != expected:
        raise BENDevelopmentExecutorV4Error(
            "stored result adjudication does not equal deterministic rebuild"
        )
    expected_terminal_success = _build_terminal_success_marker(
        authorization,
        AUTHORIZATION_PATH,
        access_intent,
        result,
        stored,
    )
    if terminal_success != expected_terminal_success:
        raise BENDevelopmentExecutorV4Error(
            "terminal success marker does not match approved completed outputs"
        )
    phases_validated = _validate_durable_phase_receipts(access_intent)
    return {
        "ok": True,
        "status": stored["status"],
        "result_content_sha256": result["content_sha256"],
        "adjudication_content_sha256": stored["content_sha256"],
        "scientific_payload_files_opened": 0,
        "durable_phase_receipts_validated": phases_validated,
    }


def check_result() -> dict[str, Any]:
    config = load_config()
    if confined(ACCESS_FAILURE_PATH).exists():
        raise BENDevelopmentExecutorV4Error(
            "access-failure receipt exists; successful adjudication is forbidden"
        )
    state_lock = _acquire_terminal_state_lock(create=False)
    try:
        return _check_result_under_terminal_lock(config)
    finally:
        _release_terminal_state_lock(state_lock)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "write-unauthorized",
            "check-preflight",
            "check-result",
            "execute",
        ),
    )
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(canonical_json({"path": str(write_preflight()), "decision": DECISION}))
        return 0
    if args.command == "write-unauthorized":
        print(canonical_json({"path": str(write_unauthorized_template()), "authorized": False}))
        return 0
    if args.command == "check-preflight":
        print(canonical_json(check_preflight()))
        return 0
    if args.command == "check-result":
        print(canonical_json(check_result()))
        return 0
    if args.authorization is None:
        raise BENDevelopmentExecutorV4Error("--authorization is required for execute")
    print(canonical_json({"path": str(execute(args.authorization))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
