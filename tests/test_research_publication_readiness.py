from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from sigma_theory_compiler import research_publication_readiness as readiness

ROOT = Path(__file__).resolve().parents[1]


def test_current_cluster_result_is_retained_but_not_data_or_paper_ready() -> None:
    receipt = readiness.build_receipt(ROOT)
    assert receipt["decision"] == "DEVELOPMENT_CLUSTER_RESULT_RETAINED_BLOCKED_NOT_DATA_READY"
    assert receipt["claim_tracks"]["bounded_empirical_publication"] == {
        "status": "DEVELOPMENT_RESULT_RETAINED_NOT_PUBLICATION_READY",
        "ready": False,
        "missing_requirements": [
            "full_uncertainty_and_covariance",
            "independent_replication",
            "nuisance_and_alternative_cause_robustness",
            "prior_art_and_reproducibility",
        ],
    }
    assert receipt["automatic_findings"]["bounded_cluster_result_retained"] is True
    assert receipt["automatic_findings"]["galaxy_failure_erased_cluster_result"] is False
    assert receipt["automatic_findings"]["galaxy_failure_blocks_universal_promotion"] is True
    assert receipt["automatic_findings"]["shared_ben_synthetic_plumbing_validated"] is True
    assert receipt["automatic_findings"]["shared_ben_real_score_exists"] is False
    assert (
        receipt["automatic_findings"]["local_sparc_confirmation_valid_for_ben_descendant"] is False
    )
    assert receipt["automatic_findings"]["group_scale_ready_lanes"] == 0
    assert receipt["automatic_findings"]["CP5_11_predictor_strata_frozen"] is True
    assert receipt["automatic_findings"]["CP5_13_complete"] is False
    assert receipt["automatic_findings"]["frozen_strata_explain_covariance_flips"] is False
    assert (
        receipt["automatic_findings"][
            "shared_formula_classes_structurally_mapped_to_minimal_scalar_theory"
        ]
        == 60
    )
    assert receipt["automatic_findings"]["shared_formula_source_only_minimal_scalar_classes"] == 3
    assert receipt["automatic_findings"]["shared_formula_full_covariant_bridge_derived"] is False
    assert receipt["automatic_findings"]["shared_quadrature_restricted_action_defined"] is True
    assert receipt["automatic_findings"]["shared_quadrature_exact_motion_law_recovered"] is True
    assert receipt["automatic_findings"]["shared_quadrature_quantitative_lensing_derived"] is False
    assert receipt["automatic_findings"]["shared_quadrature_scalar_cone_causal"] is False
    assert receipt["readiness"]["independent_cluster_data"]["next_gate"] == "CP3"
    assert receipt["readiness"]["independent_cluster_data"]["ready"] is False
    assert receipt["readiness"]["observational_authorization"] is False
    assert receipt["readiness"]["independent_target_rows_opened"] == 0
    assert receipt["counts"]["completed_tasks"] == 61
    assert receipt["counts"]["open_tasks"] == 61
    cp3 = next(gate for gate in receipt["gate_ledger"] if gate["gate_id"] == "CP3")
    assert cp3["completed_task_ids"] == [
        "CP3.1",
        "CP3.2",
        "CP3.3",
        "CP3.4",
        "CP3.7",
        "CP3.8",
    ]
    assert cp3["open_task_ids"] == ["CP3.5", "CP3.6"]
    cp5 = next(gate for gate in receipt["gate_ledger"] if gate["gate_id"] == "CP5")
    assert "CP5.11" in cp5["completed_task_ids"]
    assert "CP5.13" in cp5["open_task_ids"]
    cp12 = next(gate for gate in receipt["gate_ledger"] if gate["gate_id"] == "CP12")
    assert cp12["completed_task_ids"] == [
        "CP12.1",
        "CP12.2",
        "CP12.4",
        "CP12.5",
        "CP12.7",
        "CP12.8",
        "CP12.9",
    ]
    cp11 = next(gate for gate in receipt["gate_ledger"] if gate["gate_id"] == "CP11")
    assert cp11["status"] == "PARTIAL"
    assert cp11["completed_task_ids"] == ["CP11.3"]
    assert "CP11.3" not in cp11["open_task_ids"]


def test_adjacent_domain_failure_does_not_veto_a_complete_bounded_claim() -> None:
    policy = readiness.load_policy(ROOT)
    outcome = {field: True for field in policy["outcome_fields"]}
    outcome["adjacent_domain_failures"] = True
    tracks = readiness.classify_claim_tracks(outcome, policy)
    assert tracks["bounded_empirical_publication"]["ready"] is True
    assert tracks["physical_mechanism"]["ready"] is True
    assert tracks["universal_theory"]["ready"] is False
    assert tracks["universal_theory"]["missing_requirements"] == ["no_adjacent_domain_failures"]


def test_same_release_confirmation_does_not_substitute_for_independent_replication() -> None:
    policy = readiness.load_policy(ROOT)
    outcome = {field: True for field in policy["outcome_fields"]}
    outcome["same_release_confirmation"] = True
    outcome["independent_replication"] = False
    tracks = readiness.classify_claim_tracks(outcome, policy)
    assert tracks["bounded_empirical_publication"]["ready"] is False
    assert tracks["bounded_empirical_publication"]["missing_requirements"] == [
        "independent_replication"
    ]


def test_policy_rejects_erasure_and_weakened_replication_rules() -> None:
    erased = copy.deepcopy(readiness.load_policy(ROOT))
    erased["core_rules"]["adjacent_domain_failure_erases_bounded_result"] = True
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="weakened"):
        readiness.validate_policy(erased)

    same_release = copy.deepcopy(readiness.load_policy(ROOT))
    same_release["core_rules"]["same_release_confirmation_counts_as_independent_replication"] = True
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="weakened"):
        readiness.validate_policy(same_release)


def test_goal_document_and_machine_task_inventory_are_exactly_aligned() -> None:
    policy = readiness.load_policy(ROOT)
    project = readiness.load_project(ROOT, policy)
    document = (ROOT / project["goal_document_binding"]["path"]).read_text(encoding="utf-8")
    documented = set(re.findall(r"\*\*(CP(?:[0-9]|1[0-2])\.[0-9]+)\*\*", document))
    configured = {task_id for gate in project["gates"] for task_id in gate["task_ids"]}
    assert documented == configured
    assert len(configured) == 122
    assert [gate["gate_id"] for gate in project["gates"]] == list(readiness.GATE_ORDER)


def test_goal_checkbox_progress_and_gate_status_fail_closed() -> None:
    policy = readiness.load_policy(ROOT)
    project = readiness.load_project(ROOT, policy)
    path = ROOT / project["goal_document_binding"]["path"]
    partial = copy.deepcopy(project["gates"])
    partial[0]["status"] = "PARTIAL"
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="mixed task"):
        readiness._goal_task_progress(path, partial)

    not_started = copy.deepcopy(project["gates"])
    not_started[0]["status"] = "NOT_STARTED"
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="completed goal"):
        readiness._goal_task_progress(path, not_started)


def test_independent_target_seal_and_evidence_bindings_fail_closed() -> None:
    policy = readiness.load_policy(ROOT)
    opened = copy.deepcopy(readiness.load_project(ROOT, policy))
    opened["independent_data_program"]["independent_target_rows_opened"] = 1
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="target seal"):
        readiness.validate_project(opened, policy, ROOT)

    project = readiness.load_project(ROOT, policy)
    bindings = copy.deepcopy(project["evidence_bindings"])
    bindings[0]["file_sha256"] = "0" * 64
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="evidence file"):
        readiness._load_evidence(ROOT, bindings)


@pytest.mark.parametrize(
    "evidence_id,mutation,match",
    [
        (
            "shared_ben_synthetic_execution",
            lambda value: value["claim_boundary"].__setitem__(
                "synthetic_recovery_is_scientific_evidence", True
            ),
            "synthetic B\\+E\\+N",
        ),
        (
            "mixed_sparc_access_preflight",
            lambda value: value["claim_boundary"].__setitem__(
                "local_sparc_confirmation_sealed_for_descendant", True
            ),
            "mixed SPARC",
        ),
        (
            "shared_ben_real_development_preflight_v2",
            lambda value: value["claims"].__setitem__("real_scoring_executed", True),
            "real B\\+E\\+N V2",
        ),
        (
            "shared_ben_development_executor_v4",
            lambda value: value.__setitem__("scores_computed", 1),
            r"B\+E\+N V4 executor",
        ),
        (
            "shared_ben_development_executor_v4",
            lambda value: value["candidate_and_ablation_accounting"].__setitem__(
                "canonical_full_classes", 61
            ),
            r"B\+E\+N V4 executor",
        ),
        (
            "shared_ben_development_executor_v4",
            lambda value: value["runtime_environment_contract"].__setitem__(
                "comparison_operator", "exact_binary64"
            ),
            r"B\+E\+N V4 executor",
        ),
        (
            "shared_ben_development_executor_v4",
            lambda value: value["result_validation_contract"].__setitem__(
                "terminal_success_marker_required_after_runtime_restoration", False
            ),
            r"B\+E\+N V4 executor",
        ),
        (
            "group_scale_source_audit",
            lambda value: value["counts"].__setitem__("ready_lanes", 1),
            "group-scale",
        ),
        (
            "predictor_strata_preflight",
            lambda value: value["readiness"].__setitem__("CP5_13_task_complete", True),
            "predictor strata",
        ),
        (
            "cluster_strata_development_scoring",
            lambda value: value["results"]["gates"]["candidate_absolute_primary"].__setitem__(
                "passed", True
            ),
            "cluster strata scoring",
        ),
        (
            "xcop_shape_bridge_preflight",
            lambda value: value["claims"].__setitem__("real_scoring_executed", True),
            "X-COP shape bridge",
        ),
        (
            "missing_variable_preflight",
            lambda value: value["counts"].__setitem__("continuous_measurement_ready_rows", 1),
            "missing-variable",
        ),
        (
            "group_scale_bridge_acquisition_v2",
            lambda value: value["counts"].__setitem__("scientific_payload_rows_opened", 1),
            "group-scale bridge acquisition",
        ),
        (
            "act_erass_overlap_preflight",
            lambda value: value["population_gate"].__setitem__("rule_evaluated", True),
            "ACT/eRASS",
        ),
        (
            "act_erass_overlap_executor_v2",
            lambda value: value["access_state"].__setitem__("authorization", True),
            "ACT/eRASS executor",
        ),
        (
            "matter_lensing_theory_preflight",
            lambda value: value["claim_boundary"].__setitem__("healthy_action_completed", True),
            r"matter\+lensing theory",
        ),
        (
            "matter_lensing_symbolic_derivation",
            lambda value: value["claim_boundary"].__setitem__("full_H2_passed", True),
            "bounded symbolic",
        ),
        (
            "matter_lensing_external_metric_principal_symbol",
            lambda value: value["claim_boundary"].__setitem__("full_H3_passed", True),
            "external-metric principal symbol",
        ),
        (
            "matter_lensing_kinetic_gate_conditional_no_go",
            lambda value: value["claim_boundary"].__setitem__(
                "unconditional_action_no_go_established", True
            ),
            "conditional kinetic-gate",
        ),
        (
            "matter_lensing_kinetic_gate_conditional_no_go",
            lambda value: value["counts"].__setitem__("observational_files_opened", 1),
            "conditional kinetic-gate",
        ),
        (
            "group_scale_source_audit_v3",
            lambda value: value["counts"].__setitem__("ready_science_lanes", 1),
            "group-scale V3",
        ),
        (
            "group_scale_xclass_identity_executor_v1",
            lambda value: value["execution_accounting"].__setitem__("identity_rows_decoded", 1),
            "guarded X-CLASS",
        ),
        (
            "matter_lensing_split_gate_source_bound",
            lambda value: value["claim_boundary"].__setitem__(
                "physical_source_law_established", True
            ),
            "source-bound",
        ),
        (
            "matter_lensing_universal_conformal_source",
            lambda value: value["claim_boundary"].__setitem__(
                "metric_backreaction_established", True
            ),
            "conformal-source",
        ),
        (
            "matter_lensing_solar_gw_necessary_conditions",
            lambda value: value["gate_adjudication"].__setitem__("gw_gate_passed", True),
            "Solar/GW",
        ),
        (
            "matter_lensing_flrw_necessary_conditions",
            lambda value: value["adjudication"].__setitem__(
                "perturbation_stability_established", True
            ),
            "FLRW",
        ),
        (
            "matter_lensing_covariant_field_equations",
            lambda value: value["adjudication"].__setitem__("ADM_constraints_derived", True),
            "covariant field-equation",
        ),
        (
            "matter_lensing_adm_constraint_propagation",
            lambda value: value["adjudication"].__setitem__("full_H2", True),
            "ADM constraint-propagation",
        ),
        (
            "matter_lensing_scalar_hamiltonian_necessary_conditions",
            lambda value: value["adjudication"].__setitem__("physical_hamiltonian_positive", True),
            "scalar Hamiltonian",
        ),
        (
            "matter_lensing_deep_aqual_transition_tradeoff",
            lambda value: value["adjudication"].__setitem__("CP11_4_complete", True),
            "deep-AQUAL transition",
        ),
        (
            "shared_formula_scalar_kinetic_reconstruction",
            lambda value: value["adjudication"].__setitem__(
                "full_covariant_formula_bridge_derived", True
            ),
            "formula kinetic reconstruction",
        ),
        (
            "shared_quadrature_covariant_action",
            lambda value: value["adjudication"].__setitem__(
                "same_action_quantitative_lensing_solution_derived", True
            ),
            "quadrature action",
        ),
    ],
)
def test_new_evidence_semantics_fail_closed(evidence_id: str, mutation: object, match: str) -> None:
    policy = readiness.load_policy(ROOT)
    project = readiness.load_project(ROOT, policy)
    evidence = readiness._load_evidence(ROOT, project["evidence_bindings"])
    changed = copy.deepcopy(evidence)
    mutation(changed[evidence_id])  # type: ignore[operator]
    with pytest.raises(readiness.ResearchPublicationReadinessError, match=match):
        readiness._validate_gravity_evidence(changed)


def test_stored_receipt_rebuilds_exactly_and_is_content_bound() -> None:
    stored = json.loads((ROOT / readiness.OUTPUT_PATH).read_text(encoding="utf-8"))
    readiness.validate_receipt(stored, ROOT)
    assert stored == readiness.build_receipt(ROOT)
    assert stored["counts"] == {
        "claim_tracks": 3,
        "gates": 13,
        "tasks": 122,
        "completed_tasks": 61,
        "open_tasks": 61,
        "pass_gates": 4,
        "partial_gates": 6,
        "blocked_gates": 0,
        "not_started_gates": 3,
        "bound_evidence_receipts": 48,
        "independent_target_rows_opened": 0,
    }


def test_semantically_resealed_overclaim_still_fails_against_bound_evidence() -> None:
    stored = readiness.build_receipt(ROOT)
    stored["claims"]["universal_gravity_theory_ready"] = True
    body = {key: value for key, value in stored.items() if key != "content_sha256"}
    stored["content_sha256"] = readiness._sha(body)
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="evidence changed"):
        readiness.validate_receipt(stored, ROOT)
