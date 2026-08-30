"""Build the machine-readable development-evidence package for a bounded cluster paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_manuscript_evidence_package_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/manuscript-evidence-package-v1.json")
READINESS_PATH = Path("runs/engine/gravity-cluster-publication-readiness-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-manuscript-evidence-package-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-manuscript-evidence-package-receipt-1.0"
SOURCE_IDS = (
    "item59",
    "comparators",
    "uncertainty",
    "numerical_controls",
    "data_contract",
    "replication_protocol",
    "prior_art",
    "shared_ben_synthetic_execution",
    "xcop_shape_bridge_preflight",
    "mixed_sparc_access_preflight",
    "shared_ben_real_development_preflight_v2",
    "shared_ben_development_executor_v4",
    "group_scale_source_audit",
    "group_scale_bridge_acquisition_v2",
    "group_scale_source_audit_v3",
    "group_scale_xclass_identity_executor_v1",
    "missing_variable_preflight",
    "act_erass_overlap_preflight",
    "act_erass_overlap_executor_v2",
    "predictor_strata_preflight",
    "cluster_strata_development_scoring",
    "matter_lensing_theory_preflight",
    "matter_lensing_symbolic_derivation",
    "matter_lensing_external_metric_principal_symbol",
    "matter_lensing_kinetic_gate_conditional_no_go",
    "matter_lensing_split_gate_source_bound",
    "matter_lensing_universal_conformal_source",
    "matter_lensing_solar_gw_necessary_conditions",
    "matter_lensing_flrw_necessary_conditions",
    "matter_lensing_covariant_field_equations",
    "matter_lensing_adm_constraint_propagation",
    "matter_lensing_scalar_hamiltonian_necessary_conditions",
    "matter_lensing_deep_aqual_transition_tradeoff",
    "shared_formula_scalar_kinetic_reconstruction",
    "shared_quadrature_covariant_action",
    "shared_quadrature_lensing_backreaction",
    "nuisance_quotient_sampler_implementation",
    "nuisance_quotient_sbc_v3_adjudicator",
    "matched_newtonian_control_v2",
    "development_pressure_covariance",
    "a1795_covariance_source_feasibility",
)


class GravityClusterManuscriptPackageError(RuntimeError):
    """Raised when evidence, environment, or claim boundaries change."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterManuscriptPackageError(f"expected JSON object: {path}")
    return value


def _content_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    actual = _sha(body)
    compact_actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if expected not in {actual, compact_actual}:
        raise GravityClusterManuscriptPackageError("bound source content hash changed")
    return str(expected)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterManuscriptPackageError(f"{label} keys changed")


def load_config(root: Path) -> dict[str, Any]:
    value = _read_json(root.resolve() / CONFIG_PATH)
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "package_id",
            "purpose",
            "environment_freeze",
            "source_bindings",
            "included_sections",
            "claim_boundary",
            "output_path",
        },
        "manuscript package config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_development_evidence_only"
        or config["package_id"] != "gravity-cluster-manuscript-evidence-package-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterManuscriptPackageError("manuscript package identity changed")
    environment = config["environment_freeze"]
    if (
        environment["python"] != "3.13.5"
        or environment["numpy"] != "2.2.6"
        or environment["scipy"] != "1.16.1"
        or environment["pytest"] != "8.4.2"
        or len(environment["random_seed_sources"]) != 3
        or len(environment["scientific_freeze_commit"]) != 40
        or len(environment["xcop_archive_sha256"]) != 64
    ):
        raise GravityClusterManuscriptPackageError("environment or source revision changed")
    bindings = config["source_bindings"]
    if tuple(row["source_id"] for row in bindings) != SOURCE_IDS:
        raise GravityClusterManuscriptPackageError("evidence binding inventory changed")
    for row in bindings:
        if set(row) != {"source_id", "path", "file_sha256", "content_sha256"}:
            raise GravityClusterManuscriptPackageError("evidence binding keys changed")
        if len(row["file_sha256"]) != 64 or len(row["content_sha256"]) != 64:
            raise GravityClusterManuscriptPackageError("evidence binding hash changed")
    if len(config["included_sections"]) != 15:
        raise GravityClusterManuscriptPackageError("included evidence sections changed")
    if config["claim_boundary"] != {
        "development_evidence": True,
        "same_release_confirmation": True,
        "independent_replication": False,
        "full_source_covariance": False,
        "bounded_paper_ready": False,
        "physical_mechanism_ready": False,
        "universal_theory_ready": False,
        "historical_novelty_established": False,
        "alternative_to_gr_established": False,
        "dark_matter_eliminated": False,
    }:
        raise GravityClusterManuscriptPackageError("manuscript claim boundary weakened")


def _load_sources(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for binding in config["source_bindings"]:
        path = (root / str(binding["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise GravityClusterManuscriptPackageError("evidence path escaped root") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterManuscriptPackageError(f"evidence file changed: {binding['path']}")
        value = _read_json(path)
        if _content_sha(value) != binding["content_sha256"]:
            raise GravityClusterManuscriptPackageError(
                f"evidence content changed: {binding['path']}"
            )
        result[str(binding["source_id"])] = value
    _validate_new_source_semantics(result)
    return result


def _validate_new_source_semantics(sources: Mapping[str, Mapping[str, Any]]) -> None:
    shape = sources["xcop_shape_bridge_preflight"]
    missing = sources["missing_variable_preflight"]
    group = sources["group_scale_bridge_acquisition_v2"]
    group_v3 = sources["group_scale_source_audit_v3"]
    xclass_executor = sources["group_scale_xclass_identity_executor_v1"]
    act = sources["act_erass_overlap_preflight"]
    act_executor = sources["act_erass_overlap_executor_v2"]
    ben_executor = sources["shared_ben_development_executor_v4"]
    theory = sources["matter_lensing_theory_preflight"]
    symbolic = sources["matter_lensing_symbolic_derivation"]
    external_symbol = sources["matter_lensing_external_metric_principal_symbol"]
    kinetic_no_go = sources["matter_lensing_kinetic_gate_conditional_no_go"]
    source_bound = sources["matter_lensing_split_gate_source_bound"]
    conformal_source = sources["matter_lensing_universal_conformal_source"]
    solar_gw = sources["matter_lensing_solar_gw_necessary_conditions"]
    flrw = sources["matter_lensing_flrw_necessary_conditions"]
    covariant = sources["matter_lensing_covariant_field_equations"]
    adm_constraints = sources["matter_lensing_adm_constraint_propagation"]
    scalar_hamiltonian = sources["matter_lensing_scalar_hamiltonian_necessary_conditions"]
    deep_aqual_transition = sources["matter_lensing_deep_aqual_transition_tradeoff"]
    formula_kinetic = sources["shared_formula_scalar_kinetic_reconstruction"]
    quadrature_action = sources["shared_quadrature_covariant_action"]
    quadrature_lensing = sources["shared_quadrature_lensing_backreaction"]
    if (
        shape["current_authorization"]["authorized"] is not False
        or shape["claims"]["real_scoring_executed"] is not False
        or shape["claims"]["scientific_claim_allowed_now"] is not False
        or shape["claims"]["absolute_pressure_or_temperature_prediction"] is not False
        or shape["production_gate"]["payload_loader_present_in_v3"] is not False
        or shape["production_gate"]["scoring_executor_present_in_v3"] is not False
        or any(
            value != 0
            for key, value in shape["zero_access_chronology"].items()
            if key != "v3_contract_frozen_before_payload_access"
        )
    ):
        raise GravityClusterManuscriptPackageError("X-COP shape bridge ceiling changed")
    if (
        missing["counts"]["defined_proxy_contracts"] != 4
        or missing["counts"]["continuous_measurement_ready_rows"] != 0
        or missing["counts"]["source_blocked_applicable_rows"] != 16
        or missing["counts"]["response_or_target_rows_opened"] != 0
        or missing["counts"]["scientific_scores_computed"] != 0
        or missing["claim_boundary"]["continuous_missing_variables_measured"] is not False
        or missing["claim_boundary"]["cause_identified"] is not False
        or missing["claim_boundary"]["scientific_claim_allowed"] is not False
    ):
        raise GravityClusterManuscriptPackageError("missing-variable ceiling changed")
    if (
        group["counts"]["ready_science_lanes"] != 0
        or group["counts"]["sample_alias_rows_opened"] != 0
        or group["counts"]["scientific_payload_rows_opened"] != 0
        or group["counts"]["target_rows_opened"] != 0
        or group["counts"]["scores_computed"] != 0
        or group["claims"]["CP10_1_complete"] is not False
        or group["claims"]["CP10_2_complete"] is not False
        or group["claims"]["group_scale_bridge_ready"] is not False
        or group["xcop_overlap_contract"]["executed"] is not False
    ):
        raise GravityClusterManuscriptPackageError("group acquisition ceiling changed")
    if (
        act["counts"]["catalog_rows_opened"] != 0
        or act["counts"]["overlap_rows"] != 0
        or act["counts"]["ready_lanes"] != 0
        or act["counts"]["scores_computed"] != 0
        or act["population_gate"]["rule_evaluated"] is not False
        or act["population_gate"]["catalog_overlap_count"] is not None
        or act["claims"]["independent_replication_ready"] is not False
        or act["claims"]["minimum_192_rule_evaluated"] is not False
    ):
        raise GravityClusterManuscriptPackageError("ACT/eRASS ceiling changed")
    if (
        act_executor["status"] != "frozen_unauthorized_executor_not_run"
        or act_executor["decision"] != "PREPARED_NOT_AUTHORIZED_NOT_EXECUTED"
        or act_executor["config_binding"]
        != {
            "content_sha256": "61ed48e0ba6143480757472a203bad74d527975d4426e230993f81405bb74a81",
            "file_sha256": "04d2cc80e21efd688707c036f54da266d20561f24fc34f3fe4bd912fe9387b3a",
            "path": "configs/gravity_cluster_act_dr6_erass1_overlap_executor_v2.json",
        }
        or act_executor["implementation_binding"]
        != {
            "module_file_sha256": "cb3a0aca9014609ac490d8025c35940e6878fa22f4cf352d6cf210ad3c7d277e",
            "module_path": "src/sigma_theory_compiler/gravity_cluster_act_dr6_erass1_overlap_executor.py",
            "test_file_sha256": "4ed1f3e71dabd7c14135743007681f12d047c9444824fd04f8dbcde2390e3b78",
            "test_path": "tests/test_gravity_cluster_act_dr6_erass1_overlap_executor.py",
        }
        or act_executor["current_authorization_binding"]
        != {
            "authorization": False,
            "file_sha256": "22154a9521d680317251e9315be548469357c4c75887ae2382930b7d689404b9",
            "path": "runs/gravity/publication-readiness/act-dr6-erass1-overlap-executor-v2/authorization-current-unauthorized.json",
            "required_status": "UNAUTHORIZED_EXECUTOR_NOT_RUN",
        }
        or act_executor["claims"]
        != {
            "CP3_complete": False,
            "CP7_complete": False,
            "authorized_successor_ready_to_execute": False,
            "catalogs_downloaded": False,
            "central_readiness_changed": False,
            "executor_contract_frozen": True,
            "independent_replication_ready": False,
            "minimum_192_rule_evaluated": False,
            "overlap_count_computed": False,
            "xcop_exclusions_computed": False,
        }
        or act_executor["counts"]
        != {
            "catalog_rows_opened": 0,
            "files_downloaded": 0,
            "forbidden_values_decoded_or_logged": 0,
            "model_or_paid_calls": 0,
            "network_bytes_downloaded": 0,
            "network_calls": 0,
            "sanitized_ledger_rows_emitted": 0,
            "scores_computed": 0,
        }
        or act_executor["access_state"]
        != {
            "authorization": False,
            "authorized_manifest_present": False,
            "catalog_rows_opened": 0,
            "execution_started": False,
            "files_downloaded": 0,
            "forbidden_values_decoded_or_logged": 0,
            "model_or_paid_calls": 0,
            "network_bytes_downloaded": 0,
            "network_calls": 0,
            "result_directory_created": False,
            "sanitized_ledger_rows_emitted": 0,
            "scores_computed": 0,
        }
    ):
        raise GravityClusterManuscriptPackageError("ACT/eRASS executor ceiling changed")
    if (
        ben_executor["status"] != "frozen_unauthorized_zero_target_access"
        or ben_executor["decision"] != "READY_UNAUTHORIZED_ZERO_TARGET_ACCESS"
        or ben_executor["production_executed"] is not False
        or ben_executor["target_files_opened"] != 0
        or ben_executor["target_rows_read"] != 0
        or ben_executor["scores_computed"] != 0
        or ben_executor["selection_events"] != 0
        or ben_executor["source_bindings"]
        != {
            "config": {
                "file_sha256": "ae209d42b60f7f5f5e0d555763f642835eddd3ad841e596722a0ca02cbfb2d9a",
                "path": "configs/gravity_shared_target_blind_ben_development_executor_v4.json",
            },
            "source": {
                "file_sha256": "41fed937b7225d8edcf3e342477de70765557e8516d1d8d812728d79291ec0ba",
                "path": "src/sigma_theory_compiler/gravity_shared_target_blind_ben_development_executor_v4.py",
            },
            "test": {
                "file_sha256": "b428eab416802668c31774f51adec420d4fd2455c2cdab51c07d3eb5dfe00de8",
                "path": "tests/test_gravity_shared_target_blind_ben_development_executor_v4.py",
            },
        }
        or ben_executor["candidate_and_ablation_accounting"]
        != {
            "ablation_asts_overlapping_full_classes": 33,
            "canonical_full_classes": 60,
            "duplicate_registered_ablation_instances": 129,
            "raw_candidates_frozen": 240,
            "raw_equivalent_members_scored": 0,
            "registered_ablations": 180,
            "registered_variants_flagged_constant_xcop_geometry_domain_switch_risk": 117,
            "registered_variants_using_x_geometry": 117,
            "unique_ablation_asts": 51,
            "unique_asts_across_full_and_ablations": 78,
        }
        or ben_executor["config_section_sha256"]["selection_contract"]
        != "5822a0544e458f3a3e897f55fe281ac63bb1bcc625d3b4e0d9cb846de48573bd"
        or ben_executor["zero_access_chronology"]
        != {
            "ablation_scores": 0,
            "authorization_artifacts_accepted": 0,
            "candidate_scores": 0,
            "contract_frozen_before_target_access": True,
            "control_scores": 0,
            "development_payload_file_open_attempts": 0,
            "development_payload_file_reads_completed": 0,
            "model_calls": 0,
            "network_calls": 0,
            "paid_calls": 0,
            "selection_events": 0,
            "sparc_rows_read": 0,
            "xcop_predictor_rows_read": 0,
            "xcop_response_rows_read": 0,
        }
        or ben_executor["runtime_environment_contract"]["policy_id"]
        != "ben-development-reference-runtime-indifference-v1"
        or ben_executor["runtime_environment_contract"]["comparison_operator"]
        != "binary64_numerical_indifference_band"
        or ben_executor["runtime_environment_contract"][
            "reference_environment_validation_required_before_access_intent"
        ]
        is not True
        or ben_executor["runtime_environment_contract"]["tie_rule"]
        != "differences inside or on the absolute-plus-relative band are not a win"
        or "1e-12 plus 1e-10"
        not in ben_executor["authorization_contract"]["required_exact_approval_text"]
        or ben_executor["claim_ceiling"]["reference_runtime_is_fully_frozen"] is not False
        or ben_executor["claim_ceiling"][
            "numerical_indifference_band_removes_all_runtime_variation"
        ]
        is not False
        or ben_executor["claim_ceiling"]["publication_ready"] is not False
        or ben_executor["claim_ceiling"]["fresh_confirmation"] is not False
        or ben_executor["claim_ceiling"]["full_covariance"] is not False
        or ben_executor["claim_ceiling"]["historical_novelty_established"] is not False
        or ben_executor["claim_ceiling"]["alternative_to_gr_established"] is not False
        or ben_executor["claim_ceiling"]["dark_matter_eliminated"] is not False
        or ben_executor["result_validation_contract"][
            "terminal_success_marker_required_after_runtime_restoration"
        ]
        is not True
        or ben_executor["result_validation_contract"][
            "check_result_must_hold_exclusive_terminal_state_lock"
        ]
        is not True
        or ben_executor["result_validation_contract"][
            "reject_if_access_failure_receipt_exists_before_adjudication"
        ]
        is not True
        or ben_executor["interrupted_run_contract"][
            "write_atomic_no_clobber_failure_receipt_on_any_post_intent_or_runtime_restoration_exception"
        ]
        is not True
        or "runtime_restoration"
        not in ben_executor["interrupted_run_contract"]["fixed_failure_operation_allowlist"]
    ):
        raise GravityClusterManuscriptPackageError("B+E+N V4 executor ceiling changed")
    if (
        theory["counts"]["health_gates_total"] != 10
        or theory["counts"]["template_level_gates_passed"] != 1
        or theory["counts"]["health_gates_blocked"] != 9
        or theory["claim_boundary"]["healthy_action_completed"] is not False
        or theory["claim_boundary"]["alternative_to_GR_established"] is not False
        or theory["claim_boundary"]["scientific_claim_allowed"] is not False
        or any(value != 0 for value in theory["zero_access_and_compute"].values())
    ):
        raise GravityClusterManuscriptPackageError("matter+lensing theory ceiling changed")
    if (
        symbolic["counts"]["symbolic_checks"] != 20
        or symbolic["counts"]["symbolic_checks_passed"] != 20
        or symbolic["counts"]["independent_numeric_checks"] != 6
        or symbolic["counts"]["independent_numeric_checks_passed"] != 6
        or symbolic["adjudication"]["H2_general_covariant_scalar_equations"]
        != "UNVERIFIED_STORED_CONTRACT_ONLY"
        or symbolic["claim_boundary"]["full_H2_passed"] is not False
        or symbolic["claim_boundary"]["healthy_action_established"] is not False
        or symbolic["claim_boundary"]["scientific_claim_allowed"] is not False
        or any(value != 0 for value in symbolic["zero_access_and_compute"].values())
    ):
        raise GravityClusterManuscriptPackageError("bounded symbolic ceiling changed")
    if (
        external_symbol["status"]
        != "partial_external_metric_scalar_symbol_derived_designed_obstruction_preserved"
        or external_symbol["decision"]
        != "PARTIAL_H3_SCALAR_EXTERNAL_METRIC_AND_H4_CONSTANT_COEFFICIENT_SYMBOL_DERIVED_U_ONE_THIRD_OBSTRUCTION_PRESERVED"
        or external_symbol["config_binding"]
        != {
            "content_sha256": "5a526c4333ebf666fefca3ca4df5a98e05fa852ffec71792ebc41b5c99193440",
            "file_sha256": "c0c937c1e67df4ab5caa55c1ef20cf16a84f92205a4a085e56457e8009c74903",
            "path": "configs/gravity_matter_lensing_external_metric_principal_symbol_v1.json",
        }
        or external_symbol["implementation_binding"]
        != {
            "source_file_sha256": "c47e7e8f30a135505d3ffeec2623a3a92ec803983cd0c5f35f5d44b584c1de1d",
            "source_path": "src/sigma_theory_compiler/gravity_matter_lensing_external_metric_principal_symbol.py",
            "test_file_sha256": "feb7517e71a588adf925e524badbcc091ad867e3976d336add4e416385e98277",
            "test_path": "tests/test_gravity_matter_lensing_external_metric_principal_symbol.py",
        }
        or external_symbol["counts"]
        != {
            "designed_failures_preserved": 1,
            "gpu_calls": 0,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "numeric_probes": 2,
            "numeric_probes_passed": 2,
            "observational_files_opened": 0,
            "symbolic_checks": 28,
            "symbolic_checks_passed": 28,
        }
        or external_symbol["adjudication"]
        != {
            "EFT_cutoff": False,
            "H3_scalar_external_metric": "PARTIAL_MACHINE_DERIVED_CONSTANT_LOCAL_JETS_WITH_DESIGNED_TIMELIKE_OBSTRUCTION",
            "H4_constant_coefficient": "PARTIAL_MACHINE_DERIVED_ALIGNED_TIMELIKE_AND_SPACELIKE_BLOCKS_WITH_ALGEBRAIC_COMMON_CONE_PRECHECK",
            "disformal_matter_characteristics": False,
            "full_H3": False,
            "full_H4": False,
            "global_strong_hyperbolicity": False,
            "lensing_completion": False,
            "metric_constraints": False,
            "on_shell_backgrounds": False,
            "overall_decision": "PARTIAL_H3_SCALAR_EXTERNAL_METRIC_AND_H4_CONSTANT_COEFFICIENT_SYMBOL_DERIVED_U_ONE_THIRD_OBSTRUCTION_PRESERVED",
        }
        or external_symbol["claim_boundary"]
        != {
            "disformal_matter_system_healthy": False,
            "eft_validity_established": False,
            "full_H3_passed": False,
            "full_H4_passed": False,
            "global_strong_hyperbolicity_established": False,
            "healthy_action_established": False,
            "lensing_predicted": False,
            "metric_scalar_system_healthy": False,
            "observational_support": False,
            "on_shell_background_exists": False,
            "publication_readiness_changed": False,
            "scientific_claim_allowed": False,
        }
        or external_symbol["designed_obstruction"]["sign"]["u>1/3"]
        != "the X_chi contribution is negative"
        or external_symbol["numeric_suite"]["designed_failure_preserved"] is not True
        or any(value != 0 for value in external_symbol["zero_access_and_compute"].values())
    ):
        raise GravityClusterManuscriptPackageError(
            "external-metric principal-symbol ceiling changed"
        )
    if (
        kinetic_no_go["status"]
        != "conditional_timelike_kinetic_gate_no_go_machine_verified_scope_restricted"
        or kinetic_no_go["decision"]
        != "CONDITIONAL_NO_GO_FOR_GLOBALLY_NONNEGATIVE_TIMELIKE_MIXING_IN_SMOOTH_GROWING_KINETIC_GATES_REMEDIES_PREREGISTERED_NOT_VALIDATED"
        or kinetic_no_go["counts"]
        != {
            "bounded_domain_counterexamples": 3,
            "gpu_calls": 0,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "numeric_cases": 5,
            "numeric_cases_passed": 5,
            "observational_files_opened": 0,
            "remedies_preregistered": 5,
            "symbolic_checks": 15,
            "symbolic_checks_passed": 15,
        }
        or kinetic_no_go["adjudication"]["conditional_timelike_mixing_no_go"]
        != "PASS_MACHINE_DERIVED_UNDER_FROZEN_HYPOTHESES"
        or kinetic_no_go["adjudication"]["bounded_domain_nonnegative_examples_exist"] is not True
        or kinetic_no_go["adjudication"]["full_determinant_no_go"] is not False
        or kinetic_no_go["claim_boundary"][
            "conditional_external_metric_timelike_mixing_theorem_established"
        ]
        is not True
        or kinetic_no_go["claim_boundary"]["unconditional_action_no_go_established"] is not False
        or kinetic_no_go["claim_boundary"]["healthy_action_established"] is not False
        or kinetic_no_go["claim_boundary"]["observational_support"] is not False
        or kinetic_no_go["claim_boundary"]["publication_readiness_changed"] is not False
        or kinetic_no_go["analytic_contract"]["not_concluded"]
        != [
            "The full determinant is negative whenever M is negative.",
            "The current action has no healthy bounded parameter/background domain.",
            "Any covariant completion or alternative gate architecture is impossible.",
            "The metric-scalar-disformal matter system is unhealthy on shell.",
        ]
        or any(value != 0 for value in kinetic_no_go["zero_access_and_compute"].values())
        or any(
            remedy["healthy_or_working_claim"] is not False
            for remedy in kinetic_no_go["remedy_preregistration"]
        )
    ):
        raise GravityClusterManuscriptPackageError("conditional kinetic-gate ceiling changed")
    group_v3_lanes = {row["lane_id"]: row for row in group_v3["lane_readiness"]}
    if (
        group_v3["status"] != "frozen_metadata_only_audit_zero_ready_science_lanes"
        or group_v3["counts"]
        != {
            "authoritative_source_records": 14,
            "blocked_lanes": 7,
            "catalog_payload_downloads_by_receipt_builder": 0,
            "future_acquisition_runs": 0,
            "future_pilot_runs": 0,
            "lane_records": 11,
            "model_or_paid_calls": 0,
            "network_calls_by_receipt_builder": 0,
            "partial_lanes": 4,
            "ready_science_lanes": 0,
            "remote_asset_metadata_records": 17,
            "scientific_rows_opened_by_receipt_builder": 0,
            "scores_computed": 0,
        }
        or set(group_v3_lanes)
        != {
            "XCLASS_LOWZ_155",
            "EFEDS_542_RAW_REDUCTION",
            "XGAP_49_XMM",
            "ERASS1_2MRS_619",
            "ACCEPT_239",
            "SUN09_CHANDRA_43",
            "AXES_GLOBAL_CATALOGS",
            "EROSITA_DR2_CATALOG_ONLY",
            "CHEXMATE_CLUSTER_COMPARATOR",
            "LOCUSS_CLUSTER_COMPARATOR",
            "EFEDS_STACKS_997",
        }
        or group_v3_lanes["XCLASS_LOWZ_155"]["role"] != "PREFERRED_RAW_REDUCTION_COHORT"
        or group_v3_lanes["EFEDS_542_RAW_REDUCTION"]["role"] != "BACKUP_COMMON_INSTRUMENT_COHORT"
        or group_v3_lanes["ACCEPT_239"]["documented_objects"] is not None
        or group_v3_lanes["ACCEPT_239"]["reported_counts"]
        != {
            "author_project_overview_sample": 239,
            "current_heasarc_one_row_per_cluster_table": 240,
        }
        or group_v3_lanes["ACCEPT_239"]["population_count_state"]
        != "UNRESOLVED_239_AUTHOR_SAMPLE_VS_240_CURRENT_HEASARC_ROWS"
        or group_v3["future_identity_obsid_acquisition"]["authorized"] is not False
        or group_v3["future_identity_obsid_acquisition"]["executed"] is not False
        or group_v3["future_xclass_five_object_pilot"]["authorized"] is not False
        or group_v3["future_xclass_five_object_pilot"]["executed"] is not False
        or group_v3["xcop_overlap_contract"]["executed"] is not False
        or group_v3["xcop_overlap_contract"]["overlap_count"] is not None
        or group_v3["claims"]["metadata_source_audit_complete"] is not True
        or group_v3["claims"]["observational_authorization"] is not False
        or group_v3["claims"]["group_bridge_ready"] is not False
        or group_v3["claims"]["sample_assembled"] is not False
        or group_v3["claims"]["CP10_1_complete"] is not False
        or group_v3["claims"]["CP10_2_complete"] is not False
        or any(value != 0 for key, value in group_v3["access_chronology"].items() if key != "scope")
    ):
        raise GravityClusterManuscriptPackageError("group-scale V3 source audit changed")
    if (
        xclass_executor["status"]
        != "frozen_executor_preflight_external_authorization_required_unrun"
        or xclass_executor["execution_accounting"]
        != {
            "authorization_manifests_approved": 0,
            "executor_launches": 0,
            "get_attempts": 0,
            "head_calls": 0,
            "identity_rows_decoded": 0,
            "model_or_paid_calls": 0,
            "network_bytes": 0,
            "obsid_mappings": 0,
            "raw_payload_files_created": 0,
            "sanitized_results_published": 0,
            "scientific_values_decoded": 0,
            "scores_computed": 0,
            "xcop_overlap_runs": 0,
        }
        or xclass_executor["source_contract"]["expected_network_bytes"] != 16_895
        or xclass_executor["source_contract"]["expected_rows"] != 155
        or xclass_executor["network_contract"]["get_calls"] != 1
        or xclass_executor["network_contract"]["maximum_network_bytes"] != 16_895
        or xclass_executor["column_contract"]["decode_allowlist"]
        != ["XClass", "RAdeg", "DEdeg", "z"]
        or xclass_executor["authorization_contract"]["authorized_manifest_present_at_freeze"]
        is not False
        or xclass_executor["output_contract"]["access_intent_present_at_freeze"] is not False
        or xclass_executor["output_contract"]["get_attempt_marker_present_at_freeze"] is not False
        or xclass_executor["output_contract"]["result_present_at_freeze"] is not False
        or xclass_executor["obsid_contract"]["obsid_mapping_executed"] is not False
        or xclass_executor["xcop_overlap_contract"]["overlap_executed"] is not False
        or xclass_executor["claims"]["guarded_executor_implemented"] is not True
        or any(
            xclass_executor["claims"][key] is not False
            for key in (
                "CP10_1_complete",
                "CP10_2_complete",
                "candidate_tested_on_groups",
                "five_object_pilot_unlocked",
                "group_bridge_ready",
                "observational_authorization",
                "obsid_mapping_available",
                "publication_claim_supported",
                "scientific_payload_accessed",
                "source_identity_acquired",
                "source_sha256_known",
                "xcop_overlap_known",
            )
        )
    ):
        raise GravityClusterManuscriptPackageError("guarded X-CLASS executor changed")
    if (
        source_bound["status"]
        != "restricted_static_source_ceiling_machine_derived_not_physical_on_shell"
        or source_bound["counts"]["symbolic_checks_passed"] != 17
        or source_bound["counts"]["source_scaling_cases_passed"] != 4
        or source_bound["counts"]["finite_k_probes_passed"] != 4
        or source_bound["adjudication"]["sufficient_source_ceiling_derived"] is not True
        or source_bound["adjudication"]["physical_Q_chi_derived"] is not False
        or source_bound["adjudication"]["physical_on_shell_background"] is not False
        or source_bound["claim_boundary"]["restricted_static_source_bound_established"] is not True
        or any(
            source_bound["claim_boundary"][key] is not False
            for key in source_bound["claim_boundary"]
            if key != "restricted_static_source_bound_established"
        )
        or any(value != 0 for value in source_bound["zero_access_and_compute"].values())
    ):
        raise GravityClusterManuscriptPackageError("split-gate source-bound ceiling changed")
    if (
        conformal_source["status"]
        != "same_action_conformal_source_identity_machine_derived_not_on_shell"
        or conformal_source["counts"]["symbolic_checks_passed"] != 18
        or conformal_source["counts"]["numeric_cases_passed"] != 4
        or conformal_source["adjudication"]["same_action_conformal_Q_identity_derived"] is not True
        or conformal_source["adjudication"]["leading_direct_conformal_lensing_cancellation_derived"]
        is not True
        or conformal_source["adjudication"]["physical_source_profile_established"] is not False
        or conformal_source["adjudication"]["metric_backreaction"] is not False
        or conformal_source["adjudication"]["lensing_prediction"] is not False
        or conformal_source["claim_boundary"]["universal_conformal_source_identity_established"]
        is not True
        or any(
            conformal_source["claim_boundary"][key] is not False
            for key in conformal_source["claim_boundary"]
            if key != "universal_conformal_source_identity_established"
        )
        or any(value != 0 for value in conformal_source["zero_access_and_compute"].values())
    ):
        raise GravityClusterManuscriptPackageError("universal conformal-source ceiling changed")
    if (
        solar_gw["status"]
        != "restricted_necessary_conditions_machine_derived_physical_gates_blocked"
        or solar_gw["counts"]["symbolic_checks_passed"] != 16
        or solar_gw["counts"]["numeric_yukawa_probes_passed"] != 3
        or solar_gw["gate_adjudication"]["solar_necessary_inequality_derived"] is not True
        or solar_gw["gate_adjudication"]["disformal_necessary_inequality_derived"] is not True
        or solar_gw["gate_adjudication"]["solar_gate_passed"] is not False
        or solar_gw["gate_adjudication"]["gw_gate_passed"] is not False
        or solar_gw["claim_boundary"]["restricted_necessary_conditions_established"] is not True
        or any(
            solar_gw["claim_boundary"][key] is not False
            for key in solar_gw["claim_boundary"]
            if key != "restricted_necessary_conditions_established"
        )
        or any(value != 0 for value in solar_gw["zero_access_and_compute"].values())
    ):
        raise GravityClusterManuscriptPackageError("Solar/GW necessary-condition ceiling changed")
    if (
        flrw["status"] != "exact_flat_flrw_equations_machine_derived_cosmological_history_blocked"
        or flrw["counts"]["symbolic_checks_passed"] != 25
        or flrw["counts"]["gate_u_probes_passed"] != 4
        or flrw["counts"]["disformal_q_probes_passed"] != 4
        or flrw["adjudication"]["friedmann_raychaudhuri_derived"] is not True
        or flrw["adjudication"]["gate_limit_obstruction_derived"] is not True
        or flrw["adjudication"]["healthy_late_time_history_exists"] is not False
        or flrw["adjudication"]["perturbation_stability_established"] is not False
        or flrw["adjudication"]["observational_fit_performed"] is not False
        or flrw["claim_boundary"]["restricted_flat_flrw_equations_established"] is not True
        or any(
            flrw["claim_boundary"][key] is not False
            for key in flrw["claim_boundary"]
            if key != "restricted_flat_flrw_equations_established"
        )
        or any(value != 0 for value in flrw["zero_access_and_compute"].values())
    ):
        raise GravityClusterManuscriptPackageError("FLRW necessary-condition ceiling changed")
    if (
        covariant["status"]
        != "covariant_scalar_stress_and_exchange_machine_derived_full_metric_health_blocked"
        or covariant["decision"]
        != "PARTIAL_COVARIANT_SCALAR_STRESS_FIELD_EQUATIONS_AND_EXCHANGE_IDENTITY_DERIVED_FULL_METRIC_DYNAMICS_HEALTH_AND_PHYSICS_UNESTABLISHED"
        or covariant["config_binding"]
        != {
            "content_sha256": "52febf9a9b74d87e8fff208800b59d92258acea262a97817dfc1dbd499e4c894",
            "file_sha256": "e0bd786c41779e47a79b08c4182315669751ac291fce84f49fb9c3d8ee918644",
            "path": "configs/gravity_matter_lensing_covariant_field_equations_v1.json",
        }
        or covariant["implementation_binding"]
        != {
            "source_file_sha256": "13660c4c7884f86a00a9d2f60a8a3d5edf329b7235337dc33450ae57f4d17504",
            "source_path": "src/sigma_theory_compiler/gravity_matter_lensing_covariant_field_equations.py",
            "test_file_sha256": "5878eb4b288eef8c2f321eedd6a3c5e46ff9928d15f971a1686a8402a45f49f7",
            "test_path": "tests/test_gravity_matter_lensing_covariant_field_equations.py",
        }
        or covariant["counts"]
        != {
            "gpu_calls": 0,
            "metric_components_checked": 9,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "numeric_cases": 3,
            "numeric_cases_passed": 3,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "symbolic_checks": 21,
            "symbolic_checks_passed": 21,
        }
        or covariant["adjudication"]["scalar_metric_variation_derived"] is not True
        or covariant["adjudication"]["same_action_exchange_identity_derived"] is not True
        or covariant["adjudication"]["formal_einstein_equation_frozen"] is not True
        or covariant["adjudication"]["full_H2"] is not False
        or covariant["adjudication"]["ADM_constraints_derived"] is not False
        or covariant["adjudication"]["metric_backreaction_solved"] is not False
        or covariant["adjudication"]["lensing_prediction"] is not False
        or covariant["claim_boundary"]["covariant_scalar_stress_and_exchange_established"]
        is not True
        or covariant["claim_boundary"]["formal_same_action_field_equation_contract_established"]
        is not True
        or any(
            covariant["claim_boundary"][key] is not False
            for key in covariant["claim_boundary"]
            if key
            not in {
                "covariant_scalar_stress_and_exchange_established",
                "formal_same_action_field_equation_contract_established",
            }
        )
        or any(value != 0 for value in covariant["zero_access_and_compute"].values())
    ):
        raise GravityClusterManuscriptPackageError("covariant field-equation ceiling changed")
    if (
        adm_constraints["status"]
        != "adm_constraint_propagation_derived_conditional_on_scalar_matter_equations_and_standard_trace_reversed_evolution"
        or adm_constraints["decision"]
        != "CP11_3_COMPLETED_CONDITIONAL_ADM_CONSTRAINT_PROPAGATION_DERIVED_OTHER_THEORY_AND_PHYSICS_GATES_BLOCKED"
        or adm_constraints["config_binding"]
        != {
            "content_sha256": "33f8a84977417af3018ae491382d2b13208758484f5092409c50fa6ef800cf35",
            "file_sha256": "5fdfb1ebdcd4fb513668ad67ac6c7fed3de42698e73ab831830224537d8d8661",
            "path": "configs/gravity_matter_lensing_adm_constraint_propagation_v1.json",
        }
        or adm_constraints["implementation_binding"]
        != {
            "source_file_sha256": "2784eeec6e0e211cb545e1519e623efa77b52add42131af97223f58217139a4c",
            "source_path": "src/sigma_theory_compiler/gravity_matter_lensing_adm_constraint_propagation.py",
            "test_file_sha256": "8e0c13d66dcd331b34650766963590ecd0056554c8e3a06d349fa3ae03a9c8f8",
            "test_path": "tests/test_gravity_matter_lensing_adm_constraint_propagation.py",
        }
        or adm_constraints["counts"]
        != {
            "gpu_calls": 0,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "numeric_cases": 3,
            "numeric_cases_passed": 3,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "symbolic_checks": 18,
            "symbolic_checks_passed": 18,
        }
        or adm_constraints["adjudication"]
        != {
            "CP11_3_complete": True,
            "constraint_preserving_boundary_conditions_instantiated": False,
            "constraint_principal_subsystem_symmetric_hyperbolic": True,
            "constraint_propagation_system_derived": True,
            "einstein_hilbert_boundary_variation_machine_verified": False,
            "full_H2": False,
            "full_H3": False,
            "full_H4": False,
            "full_metric_scalar_matter_system_strongly_hyperbolic": False,
            "global_constraint_propagation": False,
            "hamiltonian_constraint_derived": True,
            "healthy_action": False,
            "lensing_prediction": False,
            "momentum_constraint_derived": True,
            "novelty_established": False,
            "observational_support": False,
            "on_shell_physical_background": False,
            "overall_decision": "CP11_3_COMPLETED_CONDITIONAL_ADM_CONSTRAINT_PROPAGATION_DERIVED_OTHER_THEORY_AND_PHYSICS_GATES_BLOCKED",
            "physical_hamiltonian_positive": False,
            "same_action_exchange_identity_inherited_and_rechecked": True,
            "standard_adm_evolution_representative_derived": True,
        }
        or adm_constraints["claim_boundary"]
        != {
            "CP11_3_complete": True,
            "GW_viability_established": False,
            "Solar_viability_established": False,
            "closed_healthy_theory_established": False,
            "constraint_preserving_boundary_problem_solved": False,
            "cosmology_established": False,
            "energy_momentum_exchange_and_constraint_propagation_established": True,
            "full_H2_established": False,
            "full_characteristic_system_established": False,
            "global_well_posedness_established": False,
            "motion_and_lensing_jointly_predicted": False,
            "novelty_established": False,
            "observational_support": False,
            "on_shell_solution_established": False,
            "physical_hamiltonian_positivity_established": False,
            "publication_readiness_changed": False,
            "scientific_observational_claim_allowed": False,
            "standard_adm_representative_only": True,
        }
        or adm_constraints["zero_access_and_compute"]
        != {
            "GPU_calls": 0,
            "LLM_calls": 0,
            "confirmation_rows_opened": 0,
            "holdout_rows_opened": 0,
            "independent_rows_opened": 0,
            "lensing_rows_opened": 0,
            "network_calls": 0,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "paid_calls": 0,
            "predictor_rows_opened": 0,
            "response_rows_opened": 0,
        }
    ):
        raise GravityClusterManuscriptPackageError("ADM constraint-propagation ceiling changed")
    if (
        scalar_hamiltonian["status"]
        != "restricted_scalar_adm_hamiltonian_and_legendre_conditions_derived_full_cp11_4_blocked"
        or scalar_hamiltonian["decision"]
        != "PARTIAL_SCALAR_ADM_HAMILTONIAN_AND_LEGENDRE_CONDITIONS_DERIVED_CP11_4_FULL_HEALTH_BLOCKED"
        or scalar_hamiltonian["config_binding"]
        != {
            "content_sha256": "907de84f2e288126b494bdafb087196988ff0a88559c526a595ab9ed529942ed",
            "file_sha256": "d36cccadd58ed25a44725a5620aad7e455150cdf653bf316915b6bb384a5ae2e",
            "path": "configs/gravity_matter_lensing_scalar_hamiltonian_necessary_conditions_v1.json",
        }
        or scalar_hamiltonian["implementation_binding"]
        != {
            "source_file_sha256": "25166fabf605751f204a75ba14a86044534aa21caf4502f8dd2242d380141aa0",
            "source_path": "src/sigma_theory_compiler/gravity_matter_lensing_scalar_hamiltonian_necessary_conditions.py",
            "test_file_sha256": "3e8716311db732296967bef81946f4cff94233149093681cec5a48959e118a91",
            "test_path": "tests/test_gravity_matter_lensing_scalar_hamiltonian_necessary_conditions.py",
        }
        or scalar_hamiltonian["counts"]
        != {
            "designed_failures_preserved": 2,
            "gpu_calls": 0,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "numeric_cases": 4,
            "numeric_cases_passed": 4,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "symbolic_checks": 24,
            "symbolic_checks_passed": 24,
        }
        or scalar_hamiltonian["adjudication"]
        != {
            "CP11_4_complete": False,
            "boundary_energy_flux_controlled": False,
            "canonical_stress_energy_identity_derived": True,
            "causal_cone_compatibility_established": False,
            "declared_domain_gradient_stability_proved": False,
            "full_metric_scalar_matter_no_ghost_proof": False,
            "full_system_strong_hyperbolicity": False,
            "general_slice_principal_schur_conditions_derived": True,
            "healthy_action": False,
            "homogeneous_gate_energy_obstruction_reproduced": True,
            "invalid_adm_time_slice_case_preserved": True,
            "legendre_map_and_momentum_convexity_conditions_derived": True,
            "nonlinear_cutoff_established": False,
            "observational_support": False,
            "overall_decision": "PARTIAL_SCALAR_ADM_HAMILTONIAN_AND_LEGENDRE_CONDITIONS_DERIVED_CP11_4_FULL_HEALTH_BLOCKED",
            "physical_hamiltonian_positive": False,
            "positive_principal_negative_energy_case_preserved": True,
            "scalar_canonical_hamiltonian_derived": True,
        }
        or scalar_hamiltonian["claim_boundary"]
        != {
            "CP11_4_complete": False,
            "causality_established": False,
            "full_H2_established": False,
            "full_gradient_stability_established": False,
            "full_hyperbolicity_established": False,
            "full_no_ghost_result_established": False,
            "healthy_action_established": False,
            "homogeneous_gate_energy_obstruction_derived": True,
            "motion_and_lensing_jointly_predicted": False,
            "necessary_legendre_and_slice_health_conditions_derived": True,
            "observational_support": False,
            "on_shell_solution_established": False,
            "physical_hamiltonian_positivity_established": False,
            "publication_readiness_changed": False,
            "restricted_scalar_canonical_hamiltonian_derived": True,
            "scientific_observational_claim_allowed": False,
        }
        or scalar_hamiltonian["zero_access_and_compute"]
        != {
            "GPU_calls": 0,
            "LLM_calls": 0,
            "confirmation_rows_opened": 0,
            "holdout_rows_opened": 0,
            "independent_rows_opened": 0,
            "lensing_rows_opened": 0,
            "network_calls": 0,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "paid_calls": 0,
            "predictor_rows_opened": 0,
            "response_rows_opened": 0,
        }
    ):
        raise GravityClusterManuscriptPackageError("scalar Hamiltonian ceiling changed")
    if (
        deep_aqual_transition["status"]
        != "conditional_exact_deep_aqual_transition_no_go_and_regulated_tradeoff_derived_full_cp11_4_blocked"
        or deep_aqual_transition["decision"]
        != "CONDITIONAL_EXACT_DEEP_AQUAL_TRANSITION_NO_GO_DERIVED_REGULATED_ESCAPE_HAS_ACCURACY_CONE_AND_UNIFORMITY_COSTS_CP11_4_BLOCKED"
        or deep_aqual_transition["config_binding"]
        != {
            "content_sha256": "8d613b3c1d641fa221fc878918b5e989e421e0679264bc89ac6f567a9cde2aa0",
            "file_sha256": "2d5a67b6231c5fafabffdef5b369e9a42c73e179fd943c2c2925373a0c270277",
            "path": "configs/gravity_matter_lensing_deep_aqual_transition_tradeoff_v1.json",
        }
        or deep_aqual_transition["implementation_binding"]
        != {
            "source_file_sha256": "fe68cc6ded344e0723b1aaa77addc3443ddb6dc24c7ea092ca96ab53b1b8f442",
            "source_path": "src/sigma_theory_compiler/gravity_matter_lensing_deep_aqual_transition_tradeoff.py",
            "test_file_sha256": "78d887264be449dad5771f4c4c768fe6f4192bfa535a59387b72bedcac767cfe",
            "test_path": "tests/test_gravity_matter_lensing_deep_aqual_transition_tradeoff.py",
        }
        or deep_aqual_transition["counts"]
        != {
            "gpu_calls": 0,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "numeric_cases": 4,
            "numeric_cases_passed": 4,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "symbolic_checks": 24,
            "symbolic_checks_passed": 24,
        }
        or deep_aqual_transition["adjudication"]
        != {
            "CP11_4_complete": False,
            "causality_established": False,
            "cutoff_established": False,
            "exact_deep_aqual_transition_conditional_no_go_derived": True,
            "exact_deep_aqual_transition_is_C2": False,
            "exact_deep_aqual_transition_is_uniformly_nondegenerate": False,
            "healthy_action": False,
            "observational_support": False,
            "on_shell_solution_established": False,
            "overall_decision": "CONDITIONAL_EXACT_DEEP_AQUAL_TRANSITION_NO_GO_DERIVED_REGULATED_ESCAPE_HAS_ACCURACY_CONE_AND_UNIFORMITY_COSTS_CP11_4_BLOCKED",
            "positive_floor_regulator_preserves_exact_low_gradient_aqual": False,
            "positive_floor_regulator_removes_transition_degeneracy": True,
            "regulated_example_has_finite_timelike_positive_principal_coefficients": True,
            "regulated_example_has_global_unbounded_domain_lower_bound": False,
            "regulated_example_is_subluminal_relative_to_conformal_matter_cone": False,
        }
        or deep_aqual_transition["claim_boundary"]
        != {
            "causality_established": False,
            "conditional_exact_transition_no_go_established": True,
            "exact_phenomenological_mapping_derived": False,
            "full_gradient_stability_established": False,
            "full_hyperbolicity_established": False,
            "full_no_ghost_result_established": False,
            "healthy_action_established": False,
            "motion_and_lensing_jointly_predicted": False,
            "observational_support": False,
            "publication_readiness_changed": False,
            "regulated_transition_example_established": True,
            "scientific_observational_claim_allowed": False,
        }
        or set(deep_aqual_transition["zero_access_and_compute"].values()) != {0}
    ):
        raise GravityClusterManuscriptPackageError("deep-AQUAL transition ceiling changed")
    if (
        formula_kinetic["status"]
        != "formula_to_minimal_scalar_kinetic_map_derived_three_source_only_classes_adjudicated_full_theory_blocked"
        or formula_kinetic["decision"]
        != "MINIMAL_FORMULA_TO_KINETIC_RECONSTRUCTION_DERIVED_ONLY_QUADRATURE_SOURCE_ONLY_CLASS_IS_SINGLE_VALUED_POSITIVE_BUT_CAUSAL_AND_ENDPOINT_GATES_FAIL_FULL_THEORY_BLOCKED"
        or formula_kinetic["config_binding"]
        != {
            "content_sha256": "c53030121ca93220a915241ba34335167a6765afcf6a191e85d0fa92e0263618",
            "file_sha256": "df1e6236612720f8a0a31b4780fa9beff2d1fa3fc00bc32bf0f969a022d4eb41",
            "path": "configs/gravity_shared_formula_scalar_kinetic_reconstruction_v1.json",
        }
        or formula_kinetic["implementation_binding"]
        != {
            "source_file_sha256": "0807449453db372759c5dc711317f7c07c7dc5f13a468387c244c100985b2f26",
            "source_path": "src/sigma_theory_compiler/gravity_shared_formula_scalar_kinetic_reconstruction.py",
            "test_file_sha256": "9659ad5d9a36d1072887b54e5afdaf3dc3655b22f50c881c5281b4a23c7de3f9",
            "test_path": "tests/test_gravity_shared_formula_scalar_kinetic_reconstruction.py",
        }
        or formula_kinetic["counts"]
        != {
            "auxiliary_dependent_formula_classes": 57,
            "canonical_formula_classes": 60,
            "gpu_calls": 0,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "predecessor_bindings": 4,
            "quadrature_numeric_probes": 5,
            "rar_same_excess_witness_points": 2,
            "source_only_formula_classes": 3,
            "symbolic_checks": 16,
            "symbolic_checks_passed": 16,
        }
        or formula_kinetic["adjudication"]
        != {
            "CP11_1_complete": False,
            "CP11_4_complete": False,
            "all_60_classes_structurally_classified": True,
            "auxiliary_dependent_classes_requiring_covariant_completion": 57,
            "candidate_selected_by_observations": False,
            "formula_to_minimal_kinetic_map_derived": True,
            "full_covariant_formula_bridge_derived": False,
            "healthy_action": False,
            "newtonian_control_nontrivial_scalar_map": False,
            "observational_support": False,
            "overall_decision": "MINIMAL_FORMULA_TO_KINETIC_RECONSTRUCTION_DERIVED_ONLY_QUADRATURE_SOURCE_ONLY_CLASS_IS_SINGLE_VALUED_POSITIVE_BUT_CAUSAL_AND_ENDPOINT_GATES_FAIL_FULL_THEORY_BLOCKED",
            "quadrature_minimal_map_causal_relative_to_conformal_matter_cone": False,
            "quadrature_minimal_map_has_global_regular_unbounded_domain": False,
            "quadrature_minimal_map_single_valued_and_locally_positive": True,
            "rar_like_minimal_map_gradient_stable_globally": False,
            "rar_like_minimal_map_single_valued_globally": False,
            "source_only_classes": 3,
        }
        or formula_kinetic["claim_boundary"]
        != {
            "causality_established": False,
            "full_covariant_formula_bridge_established": False,
            "global_hyperbolicity_established": False,
            "healthy_action_established": False,
            "minimal_spherical_formula_to_kinetic_map_established": True,
            "motion_and_lensing_jointly_predicted": False,
            "observational_support": False,
            "one_source_only_class_has_single_valued_positive_minimal_map": True,
            "publication_readiness_changed": False,
            "registry_structural_classification_established": True,
            "scientific_observational_claim_allowed": False,
            "surviving_physical_candidate_selected": False,
        }
        or set(formula_kinetic["zero_access_and_compute"].values()) != {0}
    ):
        raise GravityClusterManuscriptPackageError("formula kinetic reconstruction ceiling changed")
    if (
        quadrature_action["status"]
        != "restricted_spacelike_universal_conformal_quadrature_action_derived_motion_exact_lensing_unsolved_causal_and_endpoint_gates_failed"
        or quadrature_action["decision"]
        != "RESTRICTED_QUADRATURE_UNIVERSAL_CONFORMAL_ACTION_DERIVED_EXACT_MOTION_LAW_AND_STRESS_ESTABLISHED_DIRECT_LENSING_CANCELS_CAUSAL_ENDPOINT_GLOBAL_AND_QUANTITATIVE_LENSING_GATES_FAIL"
        or quadrature_action["config_binding"]
        != {
            "content_sha256": "9467ccf6c7bd2d5512f537eaa3f323a46b861fa10d8ad3b0b1133d8962ea240b",
            "file_sha256": "dca011dce2390ea37a853a44b501d24865185c83a7f7ecf6ab3e0eeaf69fdc64",
            "path": "configs/gravity_shared_quadrature_covariant_action_v1.json",
        }
        or quadrature_action["implementation_binding"]
        != {
            "source_file_sha256": "570d6cab120c9a0c5cf7456a2f2579579acc0388e1efcddaf3bad80fad793197",
            "source_path": "src/sigma_theory_compiler/gravity_shared_quadrature_covariant_action.py",
            "test_file_sha256": "9519cb28f89f5565f1d3b2a2766352a31212914b809f94410447d04516e84d82",
            "test_path": "tests/test_gravity_shared_quadrature_covariant_action.py",
        }
        or quadrature_action["counts"]
        != {
            "gpu_calls": 0,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "numeric_branch_probes": 4,
            "numeric_branch_probes_passed": 4,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "predecessor_bindings": 4,
            "symbolic_checks": 24,
            "symbolic_checks_passed": 24,
        }
        or quadrature_action["adjudication"]
        != {
            "CP11_1_complete": False,
            "CP11_4_complete": False,
            "CP11_8_complete": False,
            "direct_conformal_lensing_shift_cancels": True,
            "finite_gradient_endpoint_regular": False,
            "local_static_energy_density_positive": True,
            "local_static_radial_and_tangential_NEC_nonnegative": True,
            "low_gradient_transition_nondegenerate": False,
            "overall_decision": "RESTRICTED_QUADRATURE_UNIVERSAL_CONFORMAL_ACTION_DERIVED_EXACT_MOTION_LAW_AND_STRESS_ESTABLISHED_DIRECT_LENSING_CANCELS_CAUSAL_ENDPOINT_GLOBAL_AND_QUANTITATIVE_LENSING_GATES_FAIL",
            "quadrature_motion_law_recovered_exactly": True,
            "restricted_spacelike_covariant_action_defined": True,
            "same_action_quantitative_lensing_solution_derived": False,
            "scalar_cone_causal_relative_to_conformal_matter_cone": False,
            "scalar_stress_tensor_derived": True,
            "separate_photon_adjustment_present": False,
            "surviving_empirical_candidate_selected": False,
            "timelike_cosmological_branch_defined": False,
            "universal_massive_matter_and_photon_metric_defined": True,
        }
        or quadrature_action["claim_boundary"]
        != {
            "Solar_System_completion_established": False,
            "causality_established": False,
            "cosmological_completion_established": False,
            "global_hyperbolicity_established": False,
            "gravitational_wave_completion_established": False,
            "healthy_global_action_established": False,
            "historical_novelty_established": False,
            "observational_support": False,
            "publication_readiness_changed": False,
            "quantitative_lensing_prediction_established": False,
            "restricted_quadrature_action_embedding_established": True,
            "same_action_motion_and_lensing_architecture_defined": True,
            "scientific_observational_claim_allowed": False,
        }
        or set(quadrature_action["zero_access_and_compute"].values()) != {0}
    ):
        raise GravityClusterManuscriptPackageError("quadrature action ceiling changed")
    if (
        quadrature_lensing["status"]
        != "restricted_exterior_quadrature_lensing_backreaction_derived_no_data_quantitative_lensing_failed"
        or quadrature_lensing["decision"]
        != "RESTRICTED_QUADRATURE_LENSING_BACKREACTION_DERIVED_DIRECT_CONFORMAL_SHIFT_CANCELS_SCALAR_STRESS_LENSING_IS_ASYMPTOTICALLY_COMPACTNESS_SUPPRESSED_AND_ISOLATED_ENERGY_LOG_DIVERGES_GLOBAL_QUANTITATIVE_LENSING_REMAINS_BLOCKED"
        or quadrature_lensing["config_binding"]
        != {
            "content_sha256": "d68b84e112933c36d26fe4461fe8e48bdfdac79b8a499b3712ae76b0ffb50b04",
            "file_sha256": "3eb1558be624966fb861e860d6f590d8c76ad7862ae25cc29c43489d499bc55a",
            "path": "configs/gravity_shared_quadrature_lensing_backreaction_v1.json",
        }
        or quadrature_lensing["implementation_binding"]
        != {
            "source_file_sha256": "98d3c1c21cf667ad02e4b657b10446c24ce54f7647bc39d829a7a15c0c1f96f5",
            "source_path": "src/sigma_theory_compiler/gravity_shared_quadrature_lensing_backreaction.py",
            "test_file_sha256": "05bbfd6865816ba46d0cfe315ab816fec9321dfbdaec3f20b0590df014a480ff",
            "test_path": "tests/test_gravity_shared_quadrature_lensing_backreaction.py",
        }
        or quadrature_lensing["counts"]
        != {
            "gpu_calls": 0,
            "model_or_paid_calls": 0,
            "network_calls": 0,
            "numeric_probes": 4,
            "numeric_probes_passed": 4,
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "predecessor_artifacts": 4,
            "symbolic_checks": 16,
            "symbolic_checks_passed": 16,
        }
        or quadrature_lensing["adjudication"]
        != {
            "CP11_10_complete": False,
            "CP11_4_complete": False,
            "CP11_8_complete": False,
            "direct_conformal_lensing_shift_cancels": True,
            "external_field_or_cosmological_cutoff_derived": False,
            "finite_isolated_scalar_energy": False,
            "overall_decision": "RESTRICTED_QUADRATURE_LENSING_BACKREACTION_DERIVED_DIRECT_CONFORMAL_SHIFT_CANCELS_SCALAR_STRESS_LENSING_IS_ASYMPTOTICALLY_COMPACTNESS_SUPPRESSED_AND_ISOLATED_ENERGY_LOG_DIVERGES_GLOBAL_QUANTITATIVE_LENSING_REMAINS_BLOCKED",
            "quantitative_observable_lensing_prediction_complete": False,
            "restricted_linearized_metric_backreaction_derived": True,
            "same_action_lensing_matches_scalar_motion_enhancement_asymptotically": False,
            "scalar_lensing_backreaction_compactness_suppressed": True,
            "scalar_stress_lensing_source_nonzero": True,
            "separate_photon_adjustment_present": False,
            "standard_finite_ADM_mass_established": False,
            "unconditional_all_architecture_no_go_established": False,
        }
        or quadrature_lensing["claim_boundary"]
        != {
            "Solar_System_completion_established": False,
            "causality_established": False,
            "cosmological_completion_established": False,
            "finite_total_energy_established": False,
            "gravitational_wave_completion_established": False,
            "healthy_global_action_established": False,
            "historical_novelty_established": False,
            "observational_support": False,
            "publication_readiness_changed": False,
            "restricted_exterior_lensing_backreaction_derived": True,
            "same_action_quantitative_lensing_success": False,
            "scientific_observational_claim_allowed": False,
        }
        or set(quadrature_lensing["zero_access_and_compute"].values()) != {0}
    ):
        raise GravityClusterManuscriptPackageError("quadrature lensing ceiling changed")


def _score_without_rows(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "per_row"}


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sources = _load_sources(root, config)
    item59 = sources["item59"]
    comparators = sources["comparators"]
    uncertainty = sources["uncertainty"]
    numerical = sources["numerical_controls"]
    data_contract = sources["data_contract"]
    protocol = sources["replication_protocol"]
    prior_art = sources["prior_art"]
    ben_synthetic = sources["shared_ben_synthetic_execution"]
    shape_bridge = sources["xcop_shape_bridge_preflight"]
    sparc_incident = sources["mixed_sparc_access_preflight"]
    ben_real_v2 = sources["shared_ben_real_development_preflight_v2"]
    ben_executor = sources["shared_ben_development_executor_v4"]
    group_source = sources["group_scale_source_audit"]
    group_acquisition = sources["group_scale_bridge_acquisition_v2"]
    group_source_v3 = sources["group_scale_source_audit_v3"]
    xclass_executor = sources["group_scale_xclass_identity_executor_v1"]
    missing_variables = sources["missing_variable_preflight"]
    act_overlap = sources["act_erass_overlap_preflight"]
    act_executor = sources["act_erass_overlap_executor_v2"]
    predictor_strata = sources["predictor_strata_preflight"]
    strata_scoring = sources["cluster_strata_development_scoring"]
    theory_preflight = sources["matter_lensing_theory_preflight"]
    symbolic_derivation = sources["matter_lensing_symbolic_derivation"]
    external_symbol = sources["matter_lensing_external_metric_principal_symbol"]
    kinetic_no_go = sources["matter_lensing_kinetic_gate_conditional_no_go"]
    source_bound = sources["matter_lensing_split_gate_source_bound"]
    conformal_source = sources["matter_lensing_universal_conformal_source"]
    solar_gw = sources["matter_lensing_solar_gw_necessary_conditions"]
    flrw = sources["matter_lensing_flrw_necessary_conditions"]
    covariant = sources["matter_lensing_covariant_field_equations"]
    adm_constraints = sources["matter_lensing_adm_constraint_propagation"]
    scalar_hamiltonian = sources["matter_lensing_scalar_hamiltonian_necessary_conditions"]
    deep_aqual_transition = sources["matter_lensing_deep_aqual_transition_tradeoff"]
    formula_kinetic = sources["shared_formula_scalar_kinetic_reconstruction"]
    quadrature_action = sources["shared_quadrature_covariant_action"]
    quadrature_lensing = sources["shared_quadrature_lensing_backreaction"]
    nuisance_sampler = sources["nuisance_quotient_sampler_implementation"]
    quotient_sbc = sources["nuisance_quotient_sbc_v3_adjudicator"]
    newtonian_control = sources["matched_newtonian_control_v2"]
    pressure_covariance = sources["development_pressure_covariance"]
    a1795_feasibility = sources["a1795_covariance_source_feasibility"]
    readiness = _read_json(root / READINESS_PATH)
    _content_sha(readiness)

    per_row = []
    split_summaries = {}
    for split in ("development_train", "development_holdout", "confirmation"):
        candidate = item59["splits"][split]["candidate"]
        split_summaries[split] = {
            "candidate": _score_without_rows(candidate),
            "baselines": item59["splits"][split]["baselines"],
            "improvements": item59["splits"][split]["improvements"],
        }
        for row in candidate["per_row"]:
            per_row.append({"split": split, **row})
    if len(per_row) != 233 or len({row["row_id"] for row in per_row}) != 233:
        raise GravityClusterManuscriptPackageError("per-row Item 59 inventory changed")

    access = {
        "scientific_freeze_commit": item59["scientific_freeze_commit"],
        "confirmation_response_files_opened_after_freeze": item59["counts"][
            "confirmation_response_files_opened_after_freeze"
        ],
        "development_holdout_rows": item59["counts"]["development_holdout_rows"],
        "same_release_confirmation_rows": item59["counts"]["confirmation_rows"],
        "direct_lensing_likelihood_evaluations": item59["counts"][
            "direct_lensing_likelihood_evaluations"
        ],
        "inferred_total_mass_rows": item59["counts"]["inferred_total_mass_rows"],
        "independent_target_rows_opened": readiness["counts"]["independent_target_rows_opened"],
        "independent_observational_authorization": readiness["readiness"][
            "observational_authorization"
        ],
    }
    if (
        access["independent_target_rows_opened"] != 0
        or access["independent_observational_authorization"] is not False
    ):
        raise GravityClusterManuscriptPackageError("independent target seal changed")

    body = {
        "schema_version": RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "decision": "DEVELOPMENT_MANUSCRIPT_EVIDENCE_PACKAGED_NOT_PAPER_READY",
        "config_binding": {"path": CONFIG_PATH.as_posix(), "content_sha256": _sha(config)},
        "completed_goal_evidence": {
            "CP12.2": "environment_dependencies_seeds_hardware_tolerance_scientific_freeze_and_source_archive_revision_frozen",
            "CP12.4": "all_233_candidate_rows_predictions_residuals_object_summaries_and_counterexample_ledgers_packaged",
            "CP12.5": "preflight_same_release_confirmation_and_independent_access_counts_packaged",
            "CP12.7": "ablations_negative_results_nuisance_failures_and_sensitivity_envelopes_packaged",
            "CP12.8": "absolute_and_comparator_relative_performance_packaged_together",
            "CP12.9": "bounded_mechanism_and_universal_claim_tracks_packaged",
        },
        "blocked_goal_evidence": {
            "CP12.1": "no_one_command_primary_figure_and_table_renderer",
            "CP12.3": "independent_source_calibration_covariance_split_and_exclusion_manifests_absent",
            "CP12.6": "no_external_analyst_or_separately_maintained_full_replay",
            "CP12.10": "statistical_cluster_astrophysics_and_modified_gravity_reviews_absent",
            "CP12.11": "eligible_data_licensing_and_public_release_package_incomplete",
            "CP12.12": "bounded_paper_gates_not_passed_and_no_submission_authorized",
        },
        "environment_and_revisions": config["environment_freeze"],
        "nuisance_quotient_sampler_implementation": {
            "decision": nuisance_sampler["decision"],
            "status": nuisance_sampler["status"],
            "implementation_evidence_only": nuisance_sampler["publication_readiness"][
                "implementation_evidence_only"
            ],
            "scientific_claims_added": nuisance_sampler["publication_readiness"][
                "scientific_claims_added"
            ],
            "production_authorized": nuisance_sampler["authorization_and_execution"][
                "production_authorized"
            ],
            "production_launches": nuisance_sampler["authorization_and_execution"][
                "production_launches"
            ],
            "bounded_smoke_forward_evaluations": nuisance_sampler["frozen_mechanics"][
                "bounded_smoke_forward_evaluations"
            ],
            "CP5_status": nuisance_sampler["publication_readiness"]["CP5_status"],
            "CP5_7_through_CP5_10": nuisance_sampler["publication_readiness"][
                "CP5_7_through_CP5_10"
            ],
        },
        "quotient_sampler_calibration_and_newtonian_boundary": {
            "sbc_decision": quotient_sbc["decision"],
            "machine_statement": quotient_sbc["machine_statement"],
            "v1_passed": quotient_sbc["adjudication"]["v1_passed"],
            "v2_passed": quotient_sbc["adjudication"]["v2_passed"],
            "v3_synthetic_sbc_passed": quotient_sbc["adjudication"]["v3_synthetic_sbc_passed"],
            "newtonian_control_unlock": quotient_sbc["adjudication"]["newtonian_control_unlock"],
            "candidate_production_unlock": quotient_sbc["adjudication"][
                "candidate_production_unlock"
            ],
            "retained_chains_present": quotient_sbc["diagnostic_evidence_boundary"][
                "retained_chains_present_in_sealed_npz"
            ],
            "newtonian_package_decision": newtonian_control["decision"],
            "newtonian_external_approval_present": newtonian_control["gates"][
                "external_approval_present"
            ],
            "newtonian_production_runs": newtonian_control["data_boundary"]["production_runs"],
            "newtonian_requested_likelihood_evaluations": newtonian_control["run_request"][
                "maximum_newtonian_control_likelihood_evaluations"
            ],
            "newtonian_maximum_paid_external_cost_usd": newtonian_control["run_request"][
                "maximum_paid_external_cost_usd"
            ],
            "scientific_claim_allowed": False,
        },
        "access_ledger": access,
        "split_summaries": split_summaries,
        "per_row_candidate_predictions": per_row,
        "object_and_counterexample_ledger": {
            "confirmation_cluster_wins": item59["confirmation_cluster_wins"],
            "confirmation_counterexamples": item59["confirmation_counterexamples"],
            "confirmation_counterexamples_by_cluster_observable": item59[
                "confirmation_counterexamples_by_cluster_observable"
            ],
            "counterexample_policy_assessment": item59["counterexample_policy_assessment"],
        },
        "comparators_and_ablations": {
            "ranking": comparators["ranking"],
            "candidate": comparators["candidate"],
            "comparators": comparators["comparators"],
            "ablations": comparators["ablations"],
            "limitations": comparators["limitations"],
        },
        "negative_and_numerical_controls": {
            "synthetic_recovery": numerical["synthetic_recovery"],
            "false_selection": numerical["false_selection"],
            "implementation_agreement": numerical["implementation_agreement"],
            "leakage_mutations": numerical["leakage_mutations"],
            "fold_controls": numerical["fold_controls"],
            "prospective_power_and_stopping": numerical["prospective_power_and_stopping"],
            "limitations": numerical["limitations"],
        },
        "uncertainty_and_alternative_cause_boundary": {
            "decision": uncertainty["decision"],
            "marginalization": uncertainty["marginalization"],
            "covariance_sensitivity": uncertainty["covariance_sensitivity"],
            "missingness_sensitivity": uncertainty["missingness_sensitivity"],
            "observationally_indistinguishable_causes": uncertainty[
                "observational_indistinguishability"
            ]["causes_remaining_indistinguishable_with_current_single_source_diagonal_errors"],
            "source_covariance_blockers": {
                task: uncertainty["blocked_goal_evidence"][task]
                for task in ("CP5.1", "CP5.2", "CP5.3", "CP5.4", "CP5.5", "CP5.6")
            },
            "limitations": uncertainty["limitations"],
        },
        "development_pressure_covariance_boundary": {
            "portable_integrity_decision": pressure_covariance["decision"],
            "reconstruction_decision": pressure_covariance["lineage"]["reconstruction_decision"],
            "scoring_decision": pressure_covariance["lineage"]["scoring_decision"],
            "reconstructed_matrices": pressure_covariance["lineage"]["reconstructed_matrices"],
            "scored_pressure_rows": pressure_covariance["lineage"]["scored_pressure_rows"],
            "CP5_1_status": pressure_covariance["lineage"]["CP5_1_status"],
            "archive_included": pressure_covariance["external_archive_contract"][
                "included_in_portable_package"
            ],
            "archive_license_verified": pressure_covariance["claims"]["archive_license_verified"],
            "a1795_source_decision": a1795_feasibility["decision"],
            "a1795_public_inputs_support_new_reduction": a1795_feasibility["claim_boundary"][
                "public_inputs_exist_for_a_new_a1795_reduction"
            ],
            "a1795_complete_source_packet": a1795_feasibility["adjudication"][
                "complete_public_covariance_source_packet"
            ],
            "CP5_2_through_CP5_6_complete": a1795_feasibility["adjudication"][
                "CP5_2_through_CP5_6_complete"
            ],
            "scientific_result_changed": False,
        },
        "shared_ben_synthetic_and_real_boundary": {
            "synthetic_decision": ben_synthetic["decision"],
            "synthetic_raw_candidates": ben_synthetic["candidate_registry"]["raw_candidate_count"],
            "synthetic_equivalence_classes": ben_synthetic["candidate_registry"][
                "equivalence_class_count"
            ],
            "synthetic_grammar_mechanics_validated": ben_synthetic["claim_boundary"][
                "synthetic_grammar_mechanics_validated"
            ],
            "synthetic_recovery_is_scientific_evidence": ben_synthetic["claim_boundary"][
                "synthetic_recovery_is_scientific_evidence"
            ],
            "real_scientific_evaluation_unlocked": ben_synthetic["claim_boundary"][
                "real_scientific_evaluation_unlocked"
            ],
            "mixed_sparc_incident_decision": sparc_incident["decision"],
            "mixed_sparc_files_opened": sparc_incident["data_boundary"]["mixed_sparc_files_opened"],
            "local_sparc_confirmation_sealed_for_descendant": sparc_incident["claim_boundary"][
                "local_sparc_confirmation_sealed_for_descendant"
            ],
            "incident_real_ben_evaluation_executed": sparc_incident["claim_boundary"][
                "real_ben_evaluation_executed"
            ],
            "v2_decision": ben_real_v2["decision"],
            "all_local_sparc_rows_development_only": ben_real_v2["claims"][
                "all_local_sparc_rows_development_only_for_descendant"
            ],
            "xcop_predictor_input_mapping_ready": ben_real_v2["mapping_decision"][
                "xcop_input_mapping_ready"
            ],
            "xcop_predictor_output_mapping_ready": ben_real_v2["mapping_decision"][
                "xcop_output_mapping_ready"
            ],
            "v2_blocked_before_payload_load": ben_real_v2["mapping_decision"][
                "blocked_before_payload_load"
            ],
            "v2_payload_loader_present": ben_real_v2["production_gate"][
                "payload_loader_present_in_v2"
            ],
            "v2_production_authorized": ben_real_v2["claims"]["production_authorized"],
            "v2_real_scoring_executed": ben_real_v2["claims"]["real_scoring_executed"],
            "scientific_claim_allowed": False,
        },
        "shared_ben_development_executor_v4_boundary": {
            "decision": ben_executor["decision"],
            "production_executed": ben_executor["production_executed"],
            "target_files_opened": ben_executor["target_files_opened"],
            "target_rows_read": ben_executor["target_rows_read"],
            "scores_computed": ben_executor["scores_computed"],
            "canonical_full_classes": ben_executor["candidate_and_ablation_accounting"][
                "canonical_full_classes"
            ],
            "registered_ablations": ben_executor["candidate_and_ablation_accounting"][
                "registered_ablations"
            ],
            "unique_asts_across_full_and_ablations": ben_executor[
                "candidate_and_ablation_accounting"
            ]["unique_asts_across_full_and_ablations"],
            "raw_equivalent_members_scored": ben_executor["candidate_and_ablation_accounting"][
                "raw_equivalent_members_scored"
            ],
            "comparison_operator": ben_executor["runtime_environment_contract"][
                "comparison_operator"
            ],
            "reference_runtime_is_fully_frozen": ben_executor["claim_ceiling"][
                "reference_runtime_is_fully_frozen"
            ],
            "indifference_band_removes_all_runtime_variation": ben_executor["claim_ceiling"][
                "numerical_indifference_band_removes_all_runtime_variation"
            ],
            "terminal_success_marker_required": ben_executor["result_validation_contract"][
                "terminal_success_marker_required_after_runtime_restoration"
            ],
            "publication_ready": ben_executor["claim_ceiling"]["publication_ready"],
        },
        "predictor_shape_and_missing_variable_boundary": {
            "shape_decision": shape_bridge["decision"],
            "shape_predictor_basis_frozen": shape_bridge["claims"][
                "predictor_only_xcop_shape_basis_frozen"
            ],
            "shape_response_profiled_nuisance_required": shape_bridge["claims"][
                "response_profiled_nuisance_required_at_scoring"
            ],
            "shape_production_authorized": shape_bridge["claims"]["production_authorized"],
            "shape_real_scoring_executed": shape_bridge["claims"]["real_scoring_executed"],
            "shape_absolute_prediction_established": shape_bridge["claims"][
                "absolute_pressure_or_temperature_prediction"
            ],
            "shape_payload_files_opened": (
                shape_bridge["zero_access_chronology"]["sparc_payload_files_opened"]
                + shape_bridge["zero_access_chronology"]["xcop_payload_files_opened"]
            ),
            "missing_variable_decision": missing_variables["decision"],
            "variable_families": missing_variables["counts"]["variable_families"],
            "defined_proxy_contracts": missing_variables["counts"]["defined_proxy_contracts"],
            "executable_proxy_only_rows": missing_variables["counts"]["executable_proxy_only_rows"],
            "continuous_measurement_ready_rows": missing_variables["counts"][
                "continuous_measurement_ready_rows"
            ],
            "source_blocked_applicable_rows": missing_variables["counts"][
                "source_blocked_applicable_rows"
            ],
            "source_definition_blocked_variables": missing_variables["counts"][
                "source_definition_blocked_variables"
            ],
            "cause_identified": missing_variables["claim_boundary"]["cause_identified"],
            "scientific_scoring_executed": missing_variables["claim_boundary"][
                "scientific_scoring_executed"
            ],
            "scientific_claim_allowed": False,
        },
        "group_scale_source_boundary": {
            "decision": group_source["decision"],
            "candidate_lanes": group_source["counts"]["candidate_lanes"],
            "ready_lanes": group_source["counts"]["ready_lanes"],
            "payload_rows_opened": group_source["counts"]["payload_rows_opened"],
            "scientific_scores_computed": group_source["counts"]["scientific_scores_computed"],
            "CP10_1_complete": group_source["claims"]["CP10_1_complete"],
            "CP10_2_complete": group_source["claims"]["CP10_2_complete"],
            "public_lane_ready": group_source["claims"]["public_lane_ready"],
            "receipt_builder_zero_row_purity": group_source["claims"][
                "receipt_builder_zero_row_purity"
            ],
            "interactive_audit_zero_row_purity": group_source["claims"][
                "interactive_audit_zero_row_purity"
            ],
            "scientific_result_emitted": group_source["claims"]["scientific_result_emitted"],
            "v3_decision": group_source_v3["decision"],
            "v3_candidate_lanes": group_source_v3["counts"]["lane_records"],
            "v3_partial_lanes": group_source_v3["counts"]["partial_lanes"],
            "v3_blocked_lanes": group_source_v3["counts"]["blocked_lanes"],
            "v3_ready_lanes": group_source_v3["counts"]["ready_science_lanes"],
            "v3_preferred_lane": group_source_v3["future_identity_obsid_acquisition"][
                "preferred_lane"
            ],
            "v3_backup_lane": group_source_v3["future_identity_obsid_acquisition"]["backup_lane"],
            "v3_accept_author_sample": next(
                row for row in group_source_v3["lane_readiness"] if row["lane_id"] == "ACCEPT_239"
            )["reported_counts"]["author_project_overview_sample"],
            "v3_accept_current_table_rows": next(
                row for row in group_source_v3["lane_readiness"] if row["lane_id"] == "ACCEPT_239"
            )["reported_counts"]["current_heasarc_one_row_per_cluster_table"],
            "v3_accept_count_resolved": False,
            "v3_scientific_rows_opened": group_source_v3["counts"][
                "scientific_rows_opened_by_receipt_builder"
            ],
            "v3_observational_authorization": group_source_v3["claims"][
                "observational_authorization"
            ],
        },
        "group_and_act_acquisition_boundary": {
            "group_acquisition_decision": group_acquisition["decision"],
            "group_metadata_manifest_bytes": group_acquisition["counts"][
                "metadata_manifest_bytes_frozen"
            ],
            "group_alias_rows_opened": group_acquisition["counts"]["sample_alias_rows_opened"],
            "group_scientific_payload_rows_opened": group_acquisition["counts"][
                "scientific_payload_rows_opened"
            ],
            "group_ready_science_lanes": group_acquisition["counts"]["ready_science_lanes"],
            "group_overlap_executed": group_acquisition["xcop_overlap_contract"]["executed"],
            "group_CP10_1_complete": group_acquisition["claims"]["CP10_1_complete"],
            "group_CP10_2_complete": group_acquisition["claims"]["CP10_2_complete"],
            "act_overlap_decision": act_overlap["decision"],
            "act_catalog_rows_opened": act_overlap["counts"]["catalog_rows_opened"],
            "act_overlap_rows": act_overlap["counts"]["overlap_rows"],
            "act_profile_target_rows_opened": act_overlap["counts"][
                "profile_thermodynamic_lensing_target_rows_opened"
            ],
            "act_ready_lanes": act_overlap["counts"]["ready_lanes"],
            "act_population_gate_evaluated": act_overlap["population_gate"]["rule_evaluated"],
            "act_catalog_population_gate_passed": act_overlap["claims"][
                "catalog_population_gate_passed"
            ],
            "act_independent_replication_ready": act_overlap["claims"][
                "independent_replication_ready"
            ],
            "act_executor_decision": act_executor["decision"],
            "act_executor_authorized": act_executor["access_state"]["authorization"],
            "act_executor_execution_started": act_executor["access_state"]["execution_started"],
            "act_executor_network_calls": act_executor["access_state"]["network_calls"],
            "act_executor_catalog_rows_opened": act_executor["access_state"]["catalog_rows_opened"],
            "act_executor_overlap_count_computed": act_executor["claims"]["overlap_count_computed"],
            "act_executor_xcop_exclusions_computed": act_executor["claims"][
                "xcop_exclusions_computed"
            ],
            "act_executor_minimum_192_rule_evaluated": act_executor["claims"][
                "minimum_192_rule_evaluated"
            ],
            "xclass_executor_decision": xclass_executor["decision"],
            "xclass_executor_authorized": xclass_executor["claims"]["observational_authorization"],
            "xclass_executor_get_attempts": xclass_executor["execution_accounting"]["get_attempts"],
            "xclass_executor_network_bytes": xclass_executor["execution_accounting"][
                "network_bytes"
            ],
            "xclass_executor_identity_rows": xclass_executor["execution_accounting"][
                "identity_rows_decoded"
            ],
            "xclass_executor_scientific_values": xclass_executor["execution_accounting"][
                "scientific_values_decoded"
            ],
            "xclass_executor_obsid_mapping_available": xclass_executor["claims"][
                "obsid_mapping_available"
            ],
            "xclass_executor_xcop_overlap_known": xclass_executor["claims"]["xcop_overlap_known"],
            "xclass_executor_five_object_pilot_unlocked": xclass_executor["claims"][
                "five_object_pilot_unlocked"
            ],
        },
        "cluster_strata_boundary": {
            "preflight_decision": predictor_strata["decision"],
            "development_clusters": predictor_strata["counts"]["development_clusters"],
            "relaxed_proxy": predictor_strata["counts"]["relaxed_proxy"],
            "disturbed_proxy": predictor_strata["counts"]["disturbed_proxy"],
            "cool_core": predictor_strata["counts"]["cool_core"],
            "non_cool_core": predictor_strata["counts"]["non_cool_core"],
            "CP5_11_predictor_strata_frozen": predictor_strata["readiness"][
                "CP5_11_predictor_definition_and_labels_ready"
            ],
            "preflight_target_or_response_rows_loaded": predictor_strata["data_boundary"][
                "target_or_response_rows_loaded"
            ],
            "scoring_decision": strata_scoring["decision"],
            "candidate_full_covariance_score": strata_scoring["results"]["gates"][
                "candidate_absolute_primary"
            ]["observed"],
            "candidate_absolute_gate_passed": strata_scoring["results"]["gates"][
                "candidate_absolute_primary"
            ]["passed"],
            "nfw_full_covariance_score": strata_scoring["results"]["whole_population"][
                "development_holdout"
            ]["full_covariance"]["nfw_score_equal_cluster_mean"],
            "candidate_full_covariance_advantage": strata_scoring["results"]["gates"][
                "candidate_vs_nfw_primary"
            ]["observed_mean_advantage"],
            "candidate_cluster_wins": strata_scoring["results"]["gates"][
                "candidate_vs_nfw_primary"
            ]["observed_cluster_wins"],
            "minimum_cluster_wins": strata_scoring["results"]["gates"]["candidate_vs_nfw_primary"][
                "minimum_cluster_wins"
            ],
            "candidate_object_win_gate_passed": strata_scoring["results"]["gates"][
                "candidate_vs_nfw_primary"
            ]["passed"],
            "covariance_flips": strata_scoring["results"]["whole_population"][
                "development_holdout"
            ]["positive_diagonal_to_negative_full_clusters"],
            "frozen_stratum_explains_covariance_flips": strata_scoring["results"]["gates"][
                "covariance_flip_explained_by_any_frozen_stratum"
            ]["passed"],
            "A3266_boundary_result_singleton_descriptive_only": strata_scoring["claim_boundary"][
                "A3266_boundary_result_is_singleton_descriptive_only"
            ],
            "new_raw_target_rows_opened": strata_scoring["compute_and_access_accounting"][
                "new_raw_target_rows_opened"
            ],
            "CP5_13_complete": strata_scoring["readiness"]["CP5_13_task_complete"],
            "causal_variable_identified": strata_scoring["claim_boundary"][
                "causal_variable_identified"
            ],
            "scientific_claim_allowed": strata_scoring["claim_boundary"][
                "scientific_claim_allowed"
            ],
        },
        "matter_lensing_theory_boundary": {
            "theory_decision": theory_preflight["decision"],
            "health_gates_total": theory_preflight["counts"]["health_gates_total"],
            "template_level_gates_passed": theory_preflight["counts"][
                "template_level_gates_passed"
            ],
            "health_gates_blocked": theory_preflight["counts"]["health_gates_blocked"],
            "covariant_template_defined": theory_preflight["feasibility_adjudication"][
                "action_is_covariant_template"
            ],
            "universal_matter_photon_metric_declared": theory_preflight["feasibility_adjudication"][
                "one_universal_matter_photon_metric"
            ],
            "healthy_action_completed": theory_preflight["claim_boundary"][
                "healthy_action_completed"
            ],
            "matter_and_lensing_jointly_passed": theory_preflight["feasibility_adjudication"][
                "matter_and_lensing_jointly_passed"
            ],
            "symbolic_decision": symbolic_derivation["decision"],
            "symbolic_checks_passed": symbolic_derivation["counts"]["symbolic_checks_passed"],
            "independent_numeric_checks_passed": symbolic_derivation["counts"][
                "independent_numeric_checks_passed"
            ],
            "bounded_scalar_euler_lagrange": symbolic_derivation["adjudication"][
                "H2_bounded_scalar_euler_lagrange"
            ],
            "general_covariant_scalar_equations": symbolic_derivation["adjudication"][
                "H2_general_covariant_scalar_equations"
            ],
            "full_H2_passed": symbolic_derivation["claim_boundary"]["full_H2_passed"],
            "full_metric_equation_derived": symbolic_derivation["claim_boundary"][
                "full_metric_equation_derived"
            ],
            "observational_files_opened": symbolic_derivation["counts"][
                "observational_files_opened"
            ],
            "external_symbol_decision": external_symbol["decision"],
            "H3_scalar_external_metric": external_symbol["adjudication"][
                "H3_scalar_external_metric"
            ],
            "H4_constant_coefficient": external_symbol["adjudication"]["H4_constant_coefficient"],
            "full_H3_passed": external_symbol["claim_boundary"]["full_H3_passed"],
            "full_H4_passed": external_symbol["claim_boundary"]["full_H4_passed"],
            "designed_u_above_one_third_failure_preserved": external_symbol["numeric_suite"][
                "designed_failure_preserved"
            ],
            "u_above_one_third_gate_contribution": external_symbol["designed_obstruction"]["sign"][
                "u>1/3"
            ],
            "metric_constraints_derived": external_symbol["adjudication"]["metric_constraints"],
            "on_shell_backgrounds_established": external_symbol["adjudication"][
                "on_shell_backgrounds"
            ],
            "global_strong_hyperbolicity_established": external_symbol["claim_boundary"][
                "global_strong_hyperbolicity_established"
            ],
            "kinetic_gate_conditional_no_go_decision": kinetic_no_go["decision"],
            "conditional_timelike_mixing_no_go": kinetic_no_go["adjudication"][
                "conditional_timelike_mixing_no_go"
            ],
            "bounded_domain_nonnegative_examples_exist": kinetic_no_go["adjudication"][
                "bounded_domain_nonnegative_examples_exist"
            ],
            "full_determinant_no_go": kinetic_no_go["adjudication"]["full_determinant_no_go"],
            "unconditional_action_no_go_established": kinetic_no_go["claim_boundary"][
                "unconditional_action_no_go_established"
            ],
            "kinetic_gate_observational_files_opened": kinetic_no_go["counts"][
                "observational_files_opened"
            ],
            "kinetic_gate_observational_support": kinetic_no_go["claim_boundary"][
                "observational_support"
            ],
            "source_bound_decision": source_bound["decision"],
            "restricted_static_source_bound_established": source_bound["claim_boundary"][
                "restricted_static_source_bound_established"
            ],
            "physical_source_law_established": source_bound["claim_boundary"][
                "physical_source_law_established"
            ],
            "physical_on_shell_solution_established": source_bound["claim_boundary"][
                "physical_on_shell_solution_established"
            ],
            "conformal_source_decision": conformal_source["decision"],
            "universal_conformal_source_identity_established": conformal_source["claim_boundary"][
                "universal_conformal_source_identity_established"
            ],
            "physical_source_profile_established": conformal_source["claim_boundary"][
                "physical_source_profile_established"
            ],
            "metric_backreaction_established": conformal_source["claim_boundary"][
                "metric_backreaction_established"
            ],
            "solar_gw_decision": solar_gw["decision"],
            "solar_necessary_conditions_established": solar_gw["claim_boundary"][
                "restricted_necessary_conditions_established"
            ],
            "solar_gate_passed": solar_gw["gate_adjudication"]["solar_gate_passed"],
            "gw_gate_passed": solar_gw["gate_adjudication"]["gw_gate_passed"],
            "flrw_decision": flrw["decision"],
            "restricted_flat_flrw_equations_established": flrw["claim_boundary"][
                "restricted_flat_flrw_equations_established"
            ],
            "flrw_gate_limit_obstruction_derived": flrw["adjudication"][
                "gate_limit_obstruction_derived"
            ],
            "healthy_late_time_history_exists": flrw["adjudication"][
                "healthy_late_time_history_exists"
            ],
            "perturbation_stability_established": flrw["adjudication"][
                "perturbation_stability_established"
            ],
            "cosmological_fit_performed": flrw["adjudication"]["observational_fit_performed"],
            "covariant_field_equation_decision": covariant["decision"],
            "covariant_scalar_stress_and_exchange_established": covariant["claim_boundary"][
                "covariant_scalar_stress_and_exchange_established"
            ],
            "formal_same_action_field_equation_contract_established": covariant["claim_boundary"][
                "formal_same_action_field_equation_contract_established"
            ],
            "einstein_hilbert_curvature_variation_machine_verified": covariant["adjudication"][
                "einstein_hilbert_curvature_variation_machine_verified"
            ],
            "covariant_full_H2": covariant["adjudication"]["full_H2"],
            "covariant_ADM_constraints_derived": covariant["adjudication"][
                "ADM_constraints_derived"
            ],
            "covariant_metric_backreaction_solved": covariant["adjudication"][
                "metric_backreaction_solved"
            ],
            "adm_constraint_decision": adm_constraints["decision"],
            "CP11_3_complete": adm_constraints["claim_boundary"]["CP11_3_complete"],
            "energy_momentum_exchange_and_constraint_propagation_established": adm_constraints[
                "claim_boundary"
            ]["energy_momentum_exchange_and_constraint_propagation_established"],
            "hamiltonian_constraint_derived": adm_constraints["adjudication"][
                "hamiltonian_constraint_derived"
            ],
            "momentum_constraint_derived": adm_constraints["adjudication"][
                "momentum_constraint_derived"
            ],
            "constraint_principal_subsystem_symmetric_hyperbolic": adm_constraints["adjudication"][
                "constraint_principal_subsystem_symmetric_hyperbolic"
            ],
            "standard_adm_representative_only": adm_constraints["claim_boundary"][
                "standard_adm_representative_only"
            ],
            "adm_full_H2": adm_constraints["adjudication"]["full_H2"],
            "adm_full_H3": adm_constraints["adjudication"]["full_H3"],
            "adm_full_H4": adm_constraints["adjudication"]["full_H4"],
            "full_metric_scalar_matter_system_strongly_hyperbolic": adm_constraints["adjudication"][
                "full_metric_scalar_matter_system_strongly_hyperbolic"
            ],
            "physical_hamiltonian_positive": adm_constraints["adjudication"][
                "physical_hamiltonian_positive"
            ],
            "constraint_preserving_boundary_conditions_instantiated": adm_constraints[
                "adjudication"
            ]["constraint_preserving_boundary_conditions_instantiated"],
            "global_constraint_propagation": adm_constraints["adjudication"][
                "global_constraint_propagation"
            ],
            "adm_lensing_prediction": adm_constraints["adjudication"]["lensing_prediction"],
            "adm_observational_support": adm_constraints["adjudication"]["observational_support"],
            "scalar_hamiltonian_decision": scalar_hamiltonian["decision"],
            "restricted_scalar_canonical_hamiltonian_derived": scalar_hamiltonian["claim_boundary"][
                "restricted_scalar_canonical_hamiltonian_derived"
            ],
            "necessary_legendre_and_slice_health_conditions_derived": scalar_hamiltonian[
                "claim_boundary"
            ]["necessary_legendre_and_slice_health_conditions_derived"],
            "homogeneous_gate_energy_obstruction_derived": scalar_hamiltonian["claim_boundary"][
                "homogeneous_gate_energy_obstruction_derived"
            ],
            "positive_principal_negative_energy_case_preserved": scalar_hamiltonian["adjudication"][
                "positive_principal_negative_energy_case_preserved"
            ],
            "invalid_adm_time_slice_case_preserved": scalar_hamiltonian["adjudication"][
                "invalid_adm_time_slice_case_preserved"
            ],
            "scalar_hamiltonian_CP11_4_complete": scalar_hamiltonian["adjudication"][
                "CP11_4_complete"
            ],
            "scalar_physical_hamiltonian_positive": scalar_hamiltonian["adjudication"][
                "physical_hamiltonian_positive"
            ],
            "scalar_full_no_ghost_result": scalar_hamiltonian["claim_boundary"][
                "full_no_ghost_result_established"
            ],
            "scalar_full_gradient_stability": scalar_hamiltonian["claim_boundary"][
                "full_gradient_stability_established"
            ],
            "scalar_full_hyperbolicity": scalar_hamiltonian["claim_boundary"][
                "full_hyperbolicity_established"
            ],
            "scalar_causality_established": scalar_hamiltonian["claim_boundary"][
                "causality_established"
            ],
            "deep_aqual_transition_decision": deep_aqual_transition["decision"],
            "conditional_exact_transition_no_go_established": deep_aqual_transition[
                "claim_boundary"
            ]["conditional_exact_transition_no_go_established"],
            "exact_deep_aqual_transition_is_C2": deep_aqual_transition["adjudication"][
                "exact_deep_aqual_transition_is_C2"
            ],
            "exact_deep_aqual_transition_is_uniformly_nondegenerate": deep_aqual_transition[
                "adjudication"
            ]["exact_deep_aqual_transition_is_uniformly_nondegenerate"],
            "positive_floor_regulator_removes_transition_degeneracy": deep_aqual_transition[
                "adjudication"
            ]["positive_floor_regulator_removes_transition_degeneracy"],
            "positive_floor_regulator_preserves_exact_low_gradient_aqual": deep_aqual_transition[
                "adjudication"
            ]["positive_floor_regulator_preserves_exact_low_gradient_aqual"],
            "regulated_example_is_subluminal_relative_to_conformal_matter_cone": deep_aqual_transition[
                "adjudication"
            ]["regulated_example_is_subluminal_relative_to_conformal_matter_cone"],
            "regulated_example_has_global_unbounded_domain_lower_bound": deep_aqual_transition[
                "adjudication"
            ]["regulated_example_has_global_unbounded_domain_lower_bound"],
            "deep_aqual_transition_CP11_4_complete": deep_aqual_transition["adjudication"][
                "CP11_4_complete"
            ],
            "deep_aqual_transition_healthy_action": deep_aqual_transition["adjudication"][
                "healthy_action"
            ],
            "formula_kinetic_reconstruction_decision": formula_kinetic["decision"],
            "formula_to_minimal_kinetic_map_derived": formula_kinetic["adjudication"][
                "formula_to_minimal_kinetic_map_derived"
            ],
            "formula_registry_classes_classified": formula_kinetic["counts"][
                "canonical_formula_classes"
            ],
            "formula_source_only_classes": formula_kinetic["counts"]["source_only_formula_classes"],
            "formula_auxiliary_dependent_classes": formula_kinetic["counts"][
                "auxiliary_dependent_formula_classes"
            ],
            "quadrature_minimal_map_single_valued_and_locally_positive": formula_kinetic[
                "adjudication"
            ]["quadrature_minimal_map_single_valued_and_locally_positive"],
            "quadrature_minimal_map_causal_relative_to_conformal_matter_cone": formula_kinetic[
                "adjudication"
            ]["quadrature_minimal_map_causal_relative_to_conformal_matter_cone"],
            "quadrature_minimal_map_has_global_regular_unbounded_domain": formula_kinetic[
                "adjudication"
            ]["quadrature_minimal_map_has_global_regular_unbounded_domain"],
            "rar_like_minimal_map_single_valued_globally": formula_kinetic["adjudication"][
                "rar_like_minimal_map_single_valued_globally"
            ],
            "rar_like_minimal_map_gradient_stable_globally": formula_kinetic["adjudication"][
                "rar_like_minimal_map_gradient_stable_globally"
            ],
            "formula_full_covariant_bridge_derived": formula_kinetic["adjudication"][
                "full_covariant_formula_bridge_derived"
            ],
            "formula_surviving_physical_candidate_selected": formula_kinetic["claim_boundary"][
                "surviving_physical_candidate_selected"
            ],
            "formula_kinetic_CP11_1_complete": formula_kinetic["adjudication"]["CP11_1_complete"],
            "formula_kinetic_CP11_4_complete": formula_kinetic["adjudication"]["CP11_4_complete"],
            "quadrature_action_decision": quadrature_action["decision"],
            "restricted_quadrature_action_defined": quadrature_action["adjudication"][
                "restricted_spacelike_covariant_action_defined"
            ],
            "quadrature_motion_law_recovered_exactly": quadrature_action["adjudication"][
                "quadrature_motion_law_recovered_exactly"
            ],
            "quadrature_universal_matter_photon_metric_defined": quadrature_action["adjudication"][
                "universal_massive_matter_and_photon_metric_defined"
            ],
            "quadrature_separate_photon_adjustment_present": quadrature_action["adjudication"][
                "separate_photon_adjustment_present"
            ],
            "quadrature_scalar_stress_tensor_derived": quadrature_action["adjudication"][
                "scalar_stress_tensor_derived"
            ],
            "quadrature_direct_conformal_lensing_shift_cancels": quadrature_action["adjudication"][
                "direct_conformal_lensing_shift_cancels"
            ],
            "quadrature_quantitative_lensing_solution_derived": quadrature_action["adjudication"][
                "same_action_quantitative_lensing_solution_derived"
            ],
            "quadrature_local_static_energy_density_positive": quadrature_action["adjudication"][
                "local_static_energy_density_positive"
            ],
            "quadrature_scalar_cone_causal": quadrature_action["adjudication"][
                "scalar_cone_causal_relative_to_conformal_matter_cone"
            ],
            "quadrature_low_gradient_transition_nondegenerate": quadrature_action["adjudication"][
                "low_gradient_transition_nondegenerate"
            ],
            "quadrature_finite_gradient_endpoint_regular": quadrature_action["adjudication"][
                "finite_gradient_endpoint_regular"
            ],
            "quadrature_timelike_cosmological_branch_defined": quadrature_action["adjudication"][
                "timelike_cosmological_branch_defined"
            ],
            "quadrature_action_CP11_1_complete": quadrature_action["adjudication"][
                "CP11_1_complete"
            ],
            "quadrature_action_CP11_4_complete": quadrature_action["adjudication"][
                "CP11_4_complete"
            ],
            "quadrature_action_CP11_8_complete": quadrature_action["adjudication"][
                "CP11_8_complete"
            ],
            "quadrature_lensing_decision": quadrature_lensing["decision"],
            "quadrature_restricted_lensing_backreaction_derived": quadrature_lensing[
                "adjudication"
            ]["restricted_linearized_metric_backreaction_derived"],
            "quadrature_scalar_stress_lensing_source_nonzero": quadrature_lensing["adjudication"][
                "scalar_stress_lensing_source_nonzero"
            ],
            "quadrature_lensing_backreaction_compactness_suppressed": quadrature_lensing[
                "adjudication"
            ]["scalar_lensing_backreaction_compactness_suppressed"],
            "quadrature_asymptotic_motion_lensing_match": quadrature_lensing["adjudication"][
                "same_action_lensing_matches_scalar_motion_enhancement_asymptotically"
            ],
            "quadrature_finite_isolated_scalar_energy": quadrature_lensing["adjudication"][
                "finite_isolated_scalar_energy"
            ],
            "quadrature_standard_finite_ADM_mass_established": quadrature_lensing["adjudication"][
                "standard_finite_ADM_mass_established"
            ],
            "quadrature_global_quantitative_lensing_success": quadrature_lensing["claim_boundary"][
                "same_action_quantitative_lensing_success"
            ],
            "quadrature_lensing_CP11_8_complete": quadrature_lensing["adjudication"][
                "CP11_8_complete"
            ],
            "quadrature_lensing_CP11_10_complete": quadrature_lensing["adjudication"][
                "CP11_10_complete"
            ],
            "scientific_claim_allowed": False,
        },
        "prior_art_boundary": {
            "decision": prior_art["decision"],
            "candidate_adjudication": prior_art["candidate_adjudication"],
            "closest_behavioral_neighbor": prior_art["closest_behavioral_neighbor"],
        },
        "independent_replication_boundary": {
            "data_contract_decision": data_contract["decision"],
            "protocol_decision": protocol["decision"],
            "data_contract_claims": data_contract["claims"],
            "protocol_claims": protocol["claims"],
            "frozen_decision_summary": protocol["frozen_decision_summary"],
        },
        "claim_tracks": readiness["claim_tracks"],
        "claims": config["claim_boundary"],
        "limitations": sorted(
            {
                *map(str, item59["limitations"]),
                *map(str, comparators["limitations"]),
                *map(str, uncertainty["limitations"]),
                *map(str, numerical["limitations"]),
                "B+E+N recovery and ablation results are synthetic plumbing controls, not empirical evidence.",
                "Mixed SPARC access invalidated local confirmation for this descendant; no real B+E+N score exists.",
                "The V2 real B+E+N preflight blocks before payload access because predictor-only X-COP output mapping is incomplete.",
                "The group-scale metadata audit found zero ready public lanes and completed no CP10 task.",
                "Eight-cluster strata results are exploratory, noncausal, and fail the frozen absolute and object-win gates.",
                "The X-COP shape bridge is predictor-only, unauthorized, unscored, and cannot support an absolute pressure or temperature claim.",
                "The missing-variable registry has four executable proxy rows but zero continuous measurements; sixteen applicable rows remain source-blocked.",
                "The group and ACT/eRASS acquisition paths contain metadata contracts only, with zero catalog or scientific rows and unevaluated population gates.",
                "The guarded ACT/eRASS executor is unauthorized and unrun; executable safety controls do not provide an overlap, X-COP exclusion, or population result.",
                "The two-scalar action is a blocked feasibility template: one of ten template/health gates passes and no healthy matter+lensing theory is established.",
                "The bounded symbolic suite alone verifies only restricted scalar identities and does not independently establish general covariant equations, full H2, metric variation, or joint lensing; the later covariant successor is assessed separately.",
                "The external-metric principal-symbol result is partial H3/H4 evidence on constant local jets and preserves a negative u>1/3 determinant contribution; it establishes neither healthy backgrounds nor a complete theory.",
                "The V4 B+E+N executor freezes 60 canonical classes and 180 registered ablations (78 unique ASTs total), but it is unauthorized and unrun, with zero payload access and zero scores; its reference runtime, indifference band, and terminal-state safeguards are preparation evidence only.",
                "The kinetic-gate theorem is conditional on an unbounded smooth positive gate that is already growing and keeps one timelike mixing term nonnegative; bounded-domain counterexamples remain, no observational data were opened, and no full-action no-go is established.",
                "The V3 group inventory expands to eleven lanes but still has zero ready science lanes; X-CLASS identity acquisition, ObsID mapping, X-COP overlap, and the five-object pilot are absent.",
                "The guarded X-CLASS executor is unauthorized and unrun; its exact one-GET privacy contract is preparation evidence, not an acquired group sample or scientific result.",
                "The split-gate source bound and universal conformal source identity are restricted derivations; no physical source profile, on-shell metric solution, or same-action lensing result is established.",
                "Solar/GW and FLRW packages derive necessary conditions only; both physical gates, perturbation stability, a healthy accelerating history, and every observational fit remain blocked.",
                "The covariant successor machine-derives the scalar stress tensor and same-action exchange identity, but treats the Einstein-Hilbert curvature variation as a standard stored contract; its own receipt predates the separately assessed ADM successor.",
                "The ADM successor completes CP11.3 only for the frozen standard trace-reversed ADM representative and conditionally on solved scalar-matter equations, smooth coefficients, initial constraints, and suitable boundary data; full H2, physical Hamiltonian positivity, full-system hyperbolicity, global propagation, physical solutions, observations, and lensing remain blocked.",
                "The scalar Hamiltonian successor derives the exact scalar ADM Legendre block and local slice conditions but also preserves a positive-principal negative-energy case; the full metric-matter Hamiltonian, lower bound, boundary flux, cutoff, and CP11.4 health proof remain blocked.",
                "The deep-AQUAL transition theorem is conditional on exact scaling reaching X=0; its positive-floor escape is only asymptotically AQUAL, widens the scalar cone, lacks an unbounded-domain lower coefficient bound, and does not complete CP11.4.",
                "The restricted quadrature action exactly embeds the dimensionless motion law in one universal conformal matter/photon metric and derives its scalar stress, but direct conformal lensing shifts cancel, metric backreaction is unsolved, the scalar cone is superluminal, the low-gradient branch is degenerate, the finite-gradient endpoint is singular, and no timelike cosmological branch is defined.",
                "The restricted exterior backreaction calculation finds a nonzero scalar-stress lensing source, but its large-radius response is compactness-suppressed relative to the scalar motion force and the isolated scalar energy grows logarithmically; finite-radius matching and other same-action architectures are not ruled out, while global quantitative lensing remains blocked.",
            }
        ),
        "counts": {
            "per_row_candidate_predictions": len(per_row),
            "clusters": item59["counts"]["clusters"],
            "development_clusters": item59["counts"]["development_clusters"],
            "same_release_confirmation_clusters": item59["counts"]["confirmation_clusters"],
            "comparators": comparators["counts"]["comparators"],
            "ablations": len(comparators["ablations"]),
            "null_trials": numerical["counts"]["null_trials"],
            "covariance_stress_scenarios": uncertainty["counts"][
                "covariance_sensitivity_scenarios"
            ],
            "missingness_stress_scenarios": uncertainty["counts"][
                "missingness_sensitivity_scenarios"
            ],
            "shared_ben_raw_candidates": ben_synthetic["candidate_registry"]["raw_candidate_count"],
            "shared_ben_equivalence_classes": ben_synthetic["candidate_registry"][
                "equivalence_class_count"
            ],
            "shared_ben_real_scores": 0,
            "shared_ben_v4_canonical_full_classes": ben_executor[
                "candidate_and_ablation_accounting"
            ]["canonical_full_classes"],
            "shared_ben_v4_registered_ablations": ben_executor["candidate_and_ablation_accounting"][
                "registered_ablations"
            ],
            "shared_ben_v4_unique_asts": ben_executor["candidate_and_ablation_accounting"][
                "unique_asts_across_full_and_ablations"
            ],
            "shared_ben_v4_scores": ben_executor["scores_computed"],
            "group_scale_ready_lanes": group_source["counts"]["ready_lanes"],
            "group_scale_v3_ready_lanes": group_source_v3["counts"]["ready_science_lanes"],
            "group_scale_v3_lane_records": group_source_v3["counts"]["lane_records"],
            "xclass_identity_rows_opened": xclass_executor["execution_accounting"][
                "identity_rows_decoded"
            ],
            "continuous_missing_variable_measurements": missing_variables["counts"][
                "continuous_measurement_ready_rows"
            ],
            "group_acquisition_scientific_rows": group_acquisition["counts"][
                "scientific_payload_rows_opened"
            ],
            "act_catalog_rows_opened": act_overlap["counts"]["catalog_rows_opened"],
            "act_executor_catalog_rows_opened": act_executor["counts"]["catalog_rows_opened"],
            "act_executor_network_calls": act_executor["counts"]["network_calls"],
            "theory_health_gates_passed": theory_preflight["counts"]["template_level_gates_passed"],
            "theory_health_gates_total": theory_preflight["counts"]["health_gates_total"],
            "symbolic_full_H2_passed": symbolic_derivation["claim_boundary"]["full_H2_passed"],
            "external_symbolic_checks_passed": external_symbol["counts"]["symbolic_checks_passed"],
            "external_designed_failures_preserved": external_symbol["counts"][
                "designed_failures_preserved"
            ],
            "kinetic_gate_symbolic_checks_passed": kinetic_no_go["counts"][
                "symbolic_checks_passed"
            ],
            "kinetic_gate_observational_files_opened": kinetic_no_go["counts"][
                "observational_files_opened"
            ],
            "source_bound_symbolic_checks_passed": source_bound["counts"]["symbolic_checks_passed"],
            "conformal_source_symbolic_checks_passed": conformal_source["counts"][
                "symbolic_checks_passed"
            ],
            "solar_gw_symbolic_checks_passed": solar_gw["counts"]["symbolic_checks_passed"],
            "flrw_symbolic_checks_passed": flrw["counts"]["symbolic_checks_passed"],
            "covariant_symbolic_checks_passed": covariant["counts"]["symbolic_checks_passed"],
            "covariant_numeric_cases_passed": covariant["counts"]["numeric_cases_passed"],
            "adm_constraint_symbolic_checks_passed": adm_constraints["counts"][
                "symbolic_checks_passed"
            ],
            "adm_constraint_numeric_cases_passed": adm_constraints["counts"][
                "numeric_cases_passed"
            ],
            "scalar_hamiltonian_symbolic_checks_passed": scalar_hamiltonian["counts"][
                "symbolic_checks_passed"
            ],
            "scalar_hamiltonian_numeric_cases_passed": scalar_hamiltonian["counts"][
                "numeric_cases_passed"
            ],
            "scalar_hamiltonian_designed_failures_preserved": scalar_hamiltonian["counts"][
                "designed_failures_preserved"
            ],
            "deep_aqual_transition_symbolic_checks_passed": deep_aqual_transition["counts"][
                "symbolic_checks_passed"
            ],
            "deep_aqual_transition_numeric_cases_passed": deep_aqual_transition["counts"][
                "numeric_cases_passed"
            ],
            "formula_kinetic_symbolic_checks_passed": formula_kinetic["counts"][
                "symbolic_checks_passed"
            ],
            "formula_kinetic_quadrature_numeric_probes": formula_kinetic["counts"][
                "quadrature_numeric_probes"
            ],
            "formula_kinetic_rar_witness_points": formula_kinetic["counts"][
                "rar_same_excess_witness_points"
            ],
            "quadrature_action_symbolic_checks_passed": quadrature_action["counts"][
                "symbolic_checks_passed"
            ],
            "quadrature_action_numeric_branch_probes_passed": quadrature_action["counts"][
                "numeric_branch_probes_passed"
            ],
            "quadrature_lensing_symbolic_checks_passed": quadrature_lensing["counts"][
                "symbolic_checks_passed"
            ],
            "quadrature_lensing_numeric_probes_passed": quadrature_lensing["counts"][
                "numeric_probes_passed"
            ],
            "strata_development_clusters": predictor_strata["counts"]["development_clusters"],
            "strata_new_raw_target_rows_opened": strata_scoring["compute_and_access_accounting"][
                "new_raw_target_rows_opened"
            ],
            "independent_target_rows_opened": 0,
        },
        "reproduction": {
            "package_command": "python -m sigma_theory_compiler.gravity_cluster_manuscript_evidence_package write",
            "package_check_command": "python -m sigma_theory_compiler.gravity_cluster_manuscript_evidence_package check",
            "item59_replay_command": "python -m sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate replay",
            "scope": "Recreates this machine-readable evidence package and the bound Item 59 result; it does not yet render every manuscript figure or table.",
        },
        "next_action": "Complete the independent source packet and covariance gates, then add a one-command figure/table renderer and external replay before any bounded-paper submission.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterManuscriptPackageError("manuscript evidence package changed")


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
    elif args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "counts": receipt["counts"],
            "next_action": receipt["next_action"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
