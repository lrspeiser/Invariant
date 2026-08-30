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
    "missing_variable_preflight",
    "act_erass_overlap_preflight",
    "act_erass_overlap_executor_v2",
    "predictor_strata_preflight",
    "cluster_strata_development_scoring",
    "matter_lensing_theory_preflight",
    "matter_lensing_symbolic_derivation",
    "matter_lensing_external_metric_principal_symbol",
    "matter_lensing_kinetic_gate_conditional_no_go",
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
    act = sources["act_erass_overlap_preflight"]
    act_executor = sources["act_erass_overlap_executor_v2"]
    ben_executor = sources["shared_ben_development_executor_v4"]
    theory = sources["matter_lensing_theory_preflight"]
    symbolic = sources["matter_lensing_symbolic_derivation"]
    external_symbol = sources["matter_lensing_external_metric_principal_symbol"]
    kinetic_no_go = sources["matter_lensing_kinetic_gate_conditional_no_go"]
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
    missing_variables = sources["missing_variable_preflight"]
    act_overlap = sources["act_erass_overlap_preflight"]
    act_executor = sources["act_erass_overlap_executor_v2"]
    predictor_strata = sources["predictor_strata_preflight"]
    strata_scoring = sources["cluster_strata_development_scoring"]
    theory_preflight = sources["matter_lensing_theory_preflight"]
    symbolic_derivation = sources["matter_lensing_symbolic_derivation"]
    external_symbol = sources["matter_lensing_external_metric_principal_symbol"]
    kinetic_no_go = sources["matter_lensing_kinetic_gate_conditional_no_go"]
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
                "The bounded symbolic suite verifies only restricted scalar identities; general covariant equations, full H2, metric variation, and joint lensing remain blocked.",
                "The external-metric principal-symbol result is partial H3/H4 evidence on constant local jets and preserves a negative u>1/3 determinant contribution; it establishes neither healthy backgrounds nor a complete theory.",
                "The V4 B+E+N executor freezes 60 canonical classes and 180 registered ablations (78 unique ASTs total), but it is unauthorized and unrun, with zero payload access and zero scores; its reference runtime, indifference band, and terminal-state safeguards are preparation evidence only.",
                "The kinetic-gate theorem is conditional on an unbounded smooth positive gate that is already growing and keeps one timelike mixing term nonnegative; bounded-domain counterexamples remain, no observational data were opened, and no full-action no-go is established.",
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
