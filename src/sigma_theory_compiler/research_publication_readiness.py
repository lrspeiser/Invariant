"""Claim-scoped publication readiness for empirical and theoretical research results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POLICY_PATH = Path("configs/research_publication_readiness_policy_v1.json")
PROJECT_PATH = Path("configs/gravity_cluster_publication_readiness_v1.json")
OUTPUT_PATH = Path("runs/engine/gravity-cluster-publication-readiness-v1.json")
POLICY_SCHEMA = "invariant-research-publication-readiness-policy-1.0"
PROJECT_SCHEMA = "invariant-gravity-cluster-publication-readiness-config-1.0"
RECEIPT_SCHEMA = "invariant-research-publication-readiness-receipt-1.0"
TRACK_ORDER = (
    "bounded_empirical_publication",
    "physical_mechanism",
    "universal_theory",
)
GATE_ORDER = tuple(f"CP{index}" for index in range(13))
PRE_DATA_GATES = ("CP0", "CP1", "CP3", "CP4", "CP5", "CP6", "CP7")
BOUNDED_PAPER_GATES = (
    "CP0",
    "CP1",
    "CP2",
    "CP3",
    "CP4",
    "CP5",
    "CP6",
    "CP7",
    "CP8",
    "CP12",
)
VALID_GATE_STATUSES = frozenset({"PASS", "PARTIAL", "BLOCKED", "NOT_STARTED"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GOAL_TASK = re.compile(r"^- \[(?P<mark>[ xX])\] \*\*(?P<task>CP(?:[0-9]|1[0-2])\.[0-9]+)\*\*")


class ResearchPublicationReadinessError(RuntimeError):
    """Raised when a claim, evidence binding, or publication gate fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResearchPublicationReadinessError(f"expected JSON object: {path}")
    return value


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ResearchPublicationReadinessError(f"{label} keys changed")


def _valid_sha(value: Any, label: str) -> str:
    text = str(value)
    if SHA256.fullmatch(text) is None:
        raise ResearchPublicationReadinessError(f"invalid {label} SHA-256")
    return text


def _content_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    actual = _sha(body)
    compact_actual = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if expected is not None and expected not in {actual, compact_actual}:
        raise ResearchPublicationReadinessError("bound evidence content hash changed")
    return str(expected) if expected is not None else actual


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ResearchPublicationReadinessError(f"{label} escaped repository root") from error
    if not path.is_file():
        raise ResearchPublicationReadinessError(f"{label} is missing: {relative}")
    return path


def load_policy(root: Path) -> dict[str, Any]:
    policy = _read_json(root / POLICY_PATH)
    validate_policy(policy)
    return policy


def validate_policy(policy: Mapping[str, Any]) -> None:
    _strict(
        policy,
        {
            "schema_version",
            "status",
            "purpose",
            "core_rules",
            "outcome_fields",
            "claim_tracks",
            "automatic_findings",
            "required_report_sections",
        },
        "publication policy",
    )
    if policy["schema_version"] != POLICY_SCHEMA or policy["status"] != "frozen":
        raise ResearchPublicationReadinessError("publication policy identity changed")
    core = policy["core_rules"]
    required_true = {
        "publication_is_not_universality",
        "adjacent_domain_failure_blocks_universal_promotion",
        "direct_observables_preferred_over_model_derived_truth",
        "matched_flexibility_comparators_required",
        "full_uncertainty_and_covariance_required",
        "nuisance_edge_selection_requires_followup",
        "independent_replication_required_for_bounded_publication_ready",
        "same_fields_must_predict_massive_matter_and_light_for_gravity_claim",
        "historical_novelty_requires_named_human_review",
    }
    required_false = {
        "adjacent_domain_failure_erases_bounded_result",
        "same_release_confirmation_counts_as_independent_replication",
        "single_counterexample_is_terminal",
        "finite_empirical_sample_prunes_broader_family",
    }
    if set(core) != required_true | required_false:
        raise ResearchPublicationReadinessError("publication core rules changed")
    if any(core[name] is not True for name in required_true) or any(
        core[name] is not False for name in required_false
    ):
        raise ResearchPublicationReadinessError("publication core rule weakened")
    fields = list(map(str, policy["outcome_fields"]))
    if len(fields) != 14 or len(set(fields)) != 14:
        raise ResearchPublicationReadinessError("publication outcome fields changed")
    tracks = policy["claim_tracks"]
    if tuple(tracks) != TRACK_ORDER:
        raise ResearchPublicationReadinessError("claim track order changed")
    bounded = tracks["bounded_empirical_publication"]
    if set(bounded["required_true_fields"]) - set(fields):
        raise ResearchPublicationReadinessError("bounded track uses unknown outcome field")
    if bounded["fields_that_do_not_veto_this_track"] != ["adjacent_domain_failures"]:
        raise ResearchPublicationReadinessError("bounded-domain preservation rule changed")
    if tracks["physical_mechanism"]["extends"] != "bounded_empirical_publication":
        raise ResearchPublicationReadinessError("mechanism claim inheritance changed")
    universal = tracks["universal_theory"]
    if (
        universal["extends"] != "physical_mechanism"
        or universal["requires_no_adjacent_domain_failures"] is not True
    ):
        raise ResearchPublicationReadinessError("universal claim inheritance changed")
    for track in TRACK_ORDER:
        required = tracks[track]["required_true_fields"]
        if not required or set(required) - set(fields):
            raise ResearchPublicationReadinessError(f"invalid required fields: {track}")


def _validate_binding(root: Path, binding: Mapping[str, Any], *, content: bool) -> Path:
    expected = (
        {"path", "file_sha256", "content_sha256"}
        if content
        else {
            "path",
            "file_sha256",
        }
    )
    _strict(binding, expected, "source binding")
    path = _under(root, str(binding["path"]), "source binding")
    if _file_sha(path) != _valid_sha(binding["file_sha256"], "file"):
        raise ResearchPublicationReadinessError(f"bound source file changed: {binding['path']}")
    if content:
        value = _read_json(path)
        expected_content = _valid_sha(binding["content_sha256"], "content")
        actual = value.get("content_sha256", _sha(value))
        if actual != expected_content:
            raise ResearchPublicationReadinessError(
                f"bound source content changed: {binding['path']}"
            )
    return path


def _goal_task_progress(
    path: Path, gates: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Read the human checklist and fail if it drifts from the machine inventory."""

    documented: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = GOAL_TASK.match(line)
        if match is None:
            continue
        task_id = match.group("task")
        if task_id in documented:
            raise ResearchPublicationReadinessError(f"duplicate goal task: {task_id}")
        documented[task_id] = match.group("mark").lower() == "x"

    configured = {str(task_id) for gate in gates for task_id in gate["task_ids"]}
    if set(documented) != configured:
        missing = sorted(configured - set(documented))
        extra = sorted(set(documented) - configured)
        raise ResearchPublicationReadinessError(
            f"goal task inventory changed; missing={missing}, extra={extra}"
        )

    progress: dict[str, dict[str, Any]] = {}
    for gate in gates:
        gate_id = str(gate["gate_id"])
        task_ids = list(map(str, gate["task_ids"]))
        completed = [task_id for task_id in task_ids if documented[task_id]]
        opened = [task_id for task_id in task_ids if not documented[task_id]]
        status = str(gate["status"])
        if status == "PASS" and opened:
            raise ResearchPublicationReadinessError(f"PASS gate has open goal tasks: {gate_id}")
        if status == "PARTIAL" and not (completed and opened):
            raise ResearchPublicationReadinessError(
                f"PARTIAL gate lacks mixed task progress: {gate_id}"
            )
        if status == "NOT_STARTED" and completed:
            raise ResearchPublicationReadinessError(
                f"NOT_STARTED gate has completed goal tasks: {gate_id}"
            )
        progress[gate_id] = {
            "task_count": len(task_ids),
            "completed_task_count": len(completed),
            "open_task_count": len(opened),
            "completed_task_ids": completed,
            "open_task_ids": opened,
        }
    return progress


def load_project(root: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    project = _read_json(root / PROJECT_PATH)
    validate_project(project, policy, root)
    return project


def validate_project(project: Mapping[str, Any], policy: Mapping[str, Any], root: Path) -> None:
    _strict(
        project,
        {
            "schema_version",
            "project_id",
            "policy_binding",
            "goal_document_binding",
            "candidate",
            "evidence_bindings",
            "outcome_evidence",
            "claim_language",
            "gates",
            "independent_data_program",
            "output_path",
        },
        "publication project",
    )
    if (
        project["schema_version"] != PROJECT_SCHEMA
        or project["project_id"] != "gravity-cluster-phenomenology-publication-v1"
        or project["output_path"] != str(OUTPUT_PATH).replace("\\", "/")
    ):
        raise ResearchPublicationReadinessError("publication project identity changed")
    policy_path = _validate_binding(root, project["policy_binding"], content=True)
    if policy_path != (root / POLICY_PATH).resolve() or _read_json(policy_path) != policy:
        raise ResearchPublicationReadinessError("publication policy binding changed")
    goal_document = _validate_binding(root, project["goal_document_binding"], content=False)
    candidate = project["candidate"]
    _strict(
        candidate,
        {
            "candidate_id",
            "source_gate",
            "primary_claim_track",
            "domain",
            "formula_refit_allowed_after_independent_target_open",
            "per_cluster_gravity_parameters_allowed",
        },
        "publication candidate",
    )
    if (
        candidate["primary_claim_track"] != "bounded_empirical_publication"
        or candidate["formula_refit_allowed_after_independent_target_open"] is not False
        or candidate["per_cluster_gravity_parameters_allowed"] is not False
    ):
        raise ResearchPublicationReadinessError("candidate claim or freeze weakened")
    fields = set(policy["outcome_fields"])
    if set(project["outcome_evidence"]) != fields or not all(
        isinstance(value, bool) for value in project["outcome_evidence"].values()
    ):
        raise ResearchPublicationReadinessError("outcome evidence contract changed")
    gates = project["gates"]
    if tuple(gate["gate_id"] for gate in gates) != GATE_ORDER:
        raise ResearchPublicationReadinessError("publication gate order changed")
    tasks: list[str] = []
    for gate in gates:
        _strict(
            gate,
            {
                "gate_id",
                "label",
                "status",
                "required_for_predata",
                "required_for_bounded_paper",
                "task_ids",
            },
            f"publication gate {gate.get('gate_id')}",
        )
        gate_id = str(gate["gate_id"])
        if gate["status"] not in VALID_GATE_STATUSES:
            raise ResearchPublicationReadinessError(f"invalid gate status: {gate_id}")
        if gate["required_for_predata"] is not (gate_id in PRE_DATA_GATES):
            raise ResearchPublicationReadinessError(f"pre-data gate policy changed: {gate_id}")
        if gate["required_for_bounded_paper"] is not (gate_id in BOUNDED_PAPER_GATES):
            raise ResearchPublicationReadinessError(f"bounded-paper gate policy changed: {gate_id}")
        gate_tasks = list(map(str, gate["task_ids"]))
        if not gate_tasks or any(not task.startswith(f"{gate_id}.") for task in gate_tasks):
            raise ResearchPublicationReadinessError(f"invalid task IDs: {gate_id}")
        tasks.extend(gate_tasks)
    if len(tasks) != 122 or len(set(tasks)) != 122:
        raise ResearchPublicationReadinessError("publication task inventory changed")
    _goal_task_progress(goal_document, gates)
    language = project["claim_language"]
    if not language["allowed"] or not language["prohibited"]:
        raise ResearchPublicationReadinessError("claim language boundary is empty")
    data = project["independent_data_program"]
    _strict(
        data,
        {
            "source_metadata_only",
            "candidate_lanes",
            "selected_primary_lane",
            "selected_secondary_lane",
            "authorized_target_packets",
            "independent_target_rows_opened",
            "observational_authorization",
        },
        "independent data program",
    )
    if (
        data["source_metadata_only"] is not True
        or len(data["candidate_lanes"]) != 6
        or data["selected_primary_lane"] is not None
        or data["selected_secondary_lane"] is not None
        or data["authorized_target_packets"] != []
        or data["independent_target_rows_opened"] != 0
        or data["observational_authorization"] is not False
    ):
        raise ResearchPublicationReadinessError("independent target seal changed")


def _load_evidence(root: Path, bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_ids = (
        "item59_forward_observable",
        "item60_direct_lensing_readiness",
        "item61_cross_scale_transfer",
        "screened_descendant_adjudication",
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
        "independent_data_contract",
        "act_erass_overlap_preflight",
        "act_erass_overlap_executor_v2",
        "matched_comparator_suite",
        "uncertainty_program",
        "nuisance_sampler_diagnostic",
        "nuisance_identifiability_audit",
        "nuisance_quotient_audit",
        "nuisance_quotient_sampler_implementation",
        "nuisance_quotient_sbc_v3_adjudicator",
        "matched_newtonian_control_v2",
        "development_pressure_covariance",
        "a1795_covariance_source_feasibility",
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
        "numerical_controls",
        "independent_replication_protocol",
        "prior_art_positioning",
        "manuscript_evidence_package",
        "manuscript_artifact_manifest",
    )
    if tuple(binding.get("evidence_id") for binding in bindings) != expected_ids:
        raise ResearchPublicationReadinessError("publication evidence order changed")
    result = {}
    for binding in bindings:
        _strict(
            binding,
            {
                "evidence_id",
                "path",
                "file_sha256",
                "content_sha256",
                "schema_version",
                "decision",
            },
            "publication evidence binding",
        )
        path = _under(root, str(binding["path"]), "publication evidence")
        if _file_sha(path) != _valid_sha(binding["file_sha256"], "evidence file"):
            raise ResearchPublicationReadinessError(f"evidence file changed: {binding['path']}")
        value = _read_json(path)
        if (
            _content_sha(value) != _valid_sha(binding["content_sha256"], "evidence content")
            or value.get("schema_version") != binding["schema_version"]
            or value.get("decision") != binding["decision"]
        ):
            raise ResearchPublicationReadinessError(
                f"evidence semantics changed: {binding['path']}"
            )
        result[str(binding["evidence_id"])] = value
    _validate_gravity_evidence(result)
    return result


def _validate_gravity_evidence(evidence: Mapping[str, Mapping[str, Any]]) -> None:
    item59 = evidence["item59_forward_observable"]
    item60 = evidence["item60_direct_lensing_readiness"]
    item61 = evidence["item61_cross_scale_transfer"]
    descendant = evidence["screened_descendant_adjudication"]
    ben_synthetic = evidence["shared_ben_synthetic_execution"]
    shape_bridge = evidence["xcop_shape_bridge_preflight"]
    sparc_incident = evidence["mixed_sparc_access_preflight"]
    ben_real_v2 = evidence["shared_ben_real_development_preflight_v2"]
    ben_executor = evidence["shared_ben_development_executor_v4"]
    group_source = evidence["group_scale_source_audit"]
    group_acquisition = evidence["group_scale_bridge_acquisition_v2"]
    group_source_v3 = evidence["group_scale_source_audit_v3"]
    xclass_executor = evidence["group_scale_xclass_identity_executor_v1"]
    missing_variables = evidence["missing_variable_preflight"]
    data_contract = evidence["independent_data_contract"]
    act_overlap = evidence["act_erass_overlap_preflight"]
    act_executor = evidence["act_erass_overlap_executor_v2"]
    comparators = evidence["matched_comparator_suite"]
    uncertainty = evidence["uncertainty_program"]
    nuisance_diagnostic = evidence["nuisance_sampler_diagnostic"]
    nuisance_identifiability = evidence["nuisance_identifiability_audit"]
    nuisance_quotient = evidence["nuisance_quotient_audit"]
    nuisance_quotient_sampler = evidence["nuisance_quotient_sampler_implementation"]
    quotient_sbc = evidence["nuisance_quotient_sbc_v3_adjudicator"]
    newtonian_control = evidence["matched_newtonian_control_v2"]
    pressure_covariance = evidence["development_pressure_covariance"]
    a1795_feasibility = evidence["a1795_covariance_source_feasibility"]
    predictor_strata = evidence["predictor_strata_preflight"]
    strata_scoring = evidence["cluster_strata_development_scoring"]
    theory_preflight = evidence["matter_lensing_theory_preflight"]
    symbolic_derivation = evidence["matter_lensing_symbolic_derivation"]
    external_symbol = evidence["matter_lensing_external_metric_principal_symbol"]
    kinetic_no_go = evidence["matter_lensing_kinetic_gate_conditional_no_go"]
    source_bound = evidence["matter_lensing_split_gate_source_bound"]
    conformal_source = evidence["matter_lensing_universal_conformal_source"]
    solar_gw = evidence["matter_lensing_solar_gw_necessary_conditions"]
    flrw = evidence["matter_lensing_flrw_necessary_conditions"]
    covariant = evidence["matter_lensing_covariant_field_equations"]
    adm_constraints = evidence["matter_lensing_adm_constraint_propagation"]
    numerical = evidence["numerical_controls"]
    replication_protocol = evidence["independent_replication_protocol"]
    prior_art = evidence["prior_art_positioning"]
    manuscript_package = evidence["manuscript_evidence_package"]
    manuscript_artifacts = evidence["manuscript_artifact_manifest"]
    if (
        item59["claims"]["xcop_forward_observable_development_gate_passed"] is not True
        or item59["counts"]["clusters"] != 12
        or item59["counts"]["confirmation_clusters"] != 4
        or item59["counts"]["direct_lensing_likelihood_evaluations"] != 0
    ):
        raise ResearchPublicationReadinessError("Item 59 evidence changed")
    if (
        item60["counts"]["direct_target_rows_opened"] != 0
        or item60["claims"]["item59_acceleration_candidate_empirically_rejected"] is not False
    ):
        raise ResearchPublicationReadinessError("Item 60 retention boundary changed")
    if (
        item61["claims"]["item59_cluster_result_rejected"] is not False
        or item61["claims"]["universal_cross_scale_gate_passed"] is not False
        or item61["counts"]["group_transition_objects"] != 0
    ):
        raise ResearchPublicationReadinessError("Item 61 domain boundary changed")
    if (
        descendant["promotion"]["promote_to_fresh_group_gate"] is not False
        or descendant["interpretation"]["broader_theory_family_pruned"] is not False
    ):
        raise ResearchPublicationReadinessError("descendant adjudication changed")
    if (
        ben_synthetic["status"]
        != "bounded_synthetic_ben_execution_passed_not_empirical_or_physical_evidence"
        or ben_synthetic["candidate_registry"]["raw_candidate_count"] != 240
        or ben_synthetic["candidate_registry"]["equivalence_class_count"] != 60
        or ben_synthetic["claim_boundary"]["synthetic_grammar_mechanics_validated"] is not True
        or ben_synthetic["claim_boundary"]["synthetic_recovery_is_scientific_evidence"] is not False
        or ben_synthetic["claim_boundary"]["real_scientific_evaluation_unlocked"] is not False
        or ben_synthetic["claim_boundary"]["candidate_physics_supported"] is not False
        or ben_synthetic["claim_boundary"]["same_action_derived"] is not False
        or ben_synthetic["claim_boundary"]["publication_ready"] is not False
        or any(
            ben_synthetic["data_boundary"][key] != 0
            for key in (
                "real_cluster_rows_read",
                "real_galaxy_rows_read",
                "real_group_rows_read",
                "real_lensing_rows_read",
                "real_formula_scores_computed",
                "confirmation_rows_read",
                "independent_rows_read",
                "sealed_rows_read",
                "network_calls",
                "model_calls",
                "paid_calls",
                "gpu_calls",
            )
        )
        or ben_synthetic["data_boundary"]["real_target_fields_read"] != []
    ):
        raise ResearchPublicationReadinessError("synthetic B+E+N evidence changed")
    if (
        shape_bridge["status"]
        != "v3_predictor_only_shape_basis_frozen_response_profiled_nuisance_no_payload_access"
        or shape_bridge["current_authorization"]
        != {
            "authorized": False,
            "authorized_cpu_formula_domain_batches": 0,
            "authorized_gpu_formula_domain_batches": 0,
            "authorized_paid_calls": 0,
            "authorized_payload_file_opens": 0,
            "path": "runs/gravity/shared-target-blind-ben-xcop-shape-preflight-v3/authorization-v1.json",
        }
        or shape_bridge["production_gate"]["payload_loader_present_in_v3"] is not False
        or shape_bridge["production_gate"]["scoring_executor_present_in_v3"] is not False
        or shape_bridge["claims"]["predictor_only_xcop_shape_basis_frozen"] is not True
        or shape_bridge["claims"]["production_authorized"] is not False
        or shape_bridge["claims"]["real_scoring_executed"] is not False
        or shape_bridge["claims"]["scientific_claim_allowed_now"] is not False
        or shape_bridge["claims"]["candidate_supported_or_refuted"] is not False
        or shape_bridge["claims"]["absolute_pressure_or_temperature_prediction"] is not False
        or shape_bridge["claims"]["parameter_free_target_independent_observable_mapping"]
        is not False
        or shape_bridge["zero_access_chronology"]["v3_contract_frozen_before_payload_access"]
        is not True
        or any(
            value != 0
            for key, value in shape_bridge["zero_access_chronology"].items()
            if key != "v3_contract_frozen_before_payload_access"
        )
    ):
        raise ResearchPublicationReadinessError("X-COP shape bridge evidence changed")
    if (
        sparc_incident["status"] != "blocked_preflight_retained_no_real_evaluation"
        or sparc_incident["claim_boundary"]["local_sparc_confirmation_sealed_for_descendant"]
        is not False
        or sparc_incident["claim_boundary"]["real_ben_evaluation_executed"] is not False
        or sparc_incident["claim_boundary"]["candidate_ranking_created"] is not False
        or sparc_incident["claim_boundary"]["cross_domain_metric_created"] is not False
        or sparc_incident["claim_boundary"]["scientific_claim_allowed"] is not False
        or sparc_incident["data_boundary"]["mixed_sparc_files_opened"] != 1
        or sparc_incident["data_boundary"]["mixed_sparc_file_bytes_read_by_process"] != 247_315
        or any(
            sparc_incident["data_boundary"][key] != 0
            for key in (
                "sparc_real_candidate_score_calls",
                "sparc_real_metric_calls",
                "xcop_target_files_opened",
                "xcop_target_rows_read",
                "xcop_real_candidate_score_calls",
                "xcop_real_metric_calls",
                "group_rows_read",
                "group_score_calls",
                "lensing_rows_read",
                "lensing_score_calls",
                "network_calls",
                "model_calls",
                "paid_calls",
                "gpu_calls",
            )
        )
    ):
        raise ResearchPublicationReadinessError("mixed SPARC access evidence changed")
    if (
        ben_real_v2["status"] != "v2_pre_score_contract_blocked_no_payload_access"
        or ben_real_v2["claims"]["all_local_sparc_rows_development_only_for_descendant"] is not True
        or ben_real_v2["claims"]["local_sparc_confirmation_claim_survives"] is not False
        or ben_real_v2["claims"]["predictor_only_input_mapping_frozen"] is not True
        or ben_real_v2["claims"]["predictor_only_cross_domain_output_mapping_complete"] is not False
        or ben_real_v2["claims"]["production_authorized"] is not False
        or ben_real_v2["claims"]["real_scoring_executed"] is not False
        or ben_real_v2["claims"]["scientific_claim_allowed"] is not False
        or ben_real_v2["mapping_decision"]["blocked_before_payload_load"] is not True
        or ben_real_v2["mapping_decision"]["xcop_input_mapping_ready"] is not True
        or ben_real_v2["mapping_decision"]["xcop_output_mapping_ready"] is not False
        or ben_real_v2["production_gate"]["payload_loader_present_in_v2"] is not False
        or ben_real_v2["zero_access_chronology"]["v2_contract_frozen_before_payload_access"]
        is not True
        or any(
            value != 0
            for key, value in ben_real_v2["zero_access_chronology"].items()
            if key != "v2_contract_frozen_before_payload_access"
        )
    ):
        raise ResearchPublicationReadinessError("real B+E+N V2 preflight changed")
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
        or ben_executor["zero_access_chronology"]["contract_frozen_before_target_access"]
        is not True
        or any(
            value != 0
            for key, value in ben_executor["zero_access_chronology"].items()
            if key != "contract_frozen_before_target_access"
        )
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
        or ben_executor["config_section_sha256"]["selection_contract"]
        != "5822a0544e458f3a3e897f55fe281ac63bb1bcc625d3b4e0d9cb846de48573bd"
        or "1e-12 plus 1e-10"
        not in ben_executor["authorization_contract"]["required_exact_approval_text"]
        or ben_executor["claim_ceiling"]["reference_runtime_is_fully_frozen"] is not False
        or ben_executor["claim_ceiling"][
            "numerical_indifference_band_removes_all_runtime_variation"
        ]
        is not False
        or any(
            ben_executor["claim_ceiling"][key] is not False
            for key in (
                "fresh_confirmation",
                "full_covariance",
                "historical_novelty_established",
                "publication_ready",
                "alternative_to_gr_established",
                "dark_matter_eliminated",
            )
        )
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
        raise ResearchPublicationReadinessError("B+E+N V4 executor evidence changed")
    if (
        group_source["counts"]["candidate_lanes"] != 3
        or group_source["counts"]["ready_lanes"] != 0
        or any(
            group_source["counts"][key] != 0
            for key in (
                "payload_rows_opened",
                "thermodynamic_rows_opened",
                "stellar_baryon_rows_opened",
                "inferred_mass_rows_opened",
                "lensing_rows_opened",
                "scientific_scores_computed",
                "downloads",
                "downloaded_bytes",
                "network_calls_by_receipt_builder",
                "paid_or_model_calls",
            )
        )
        or group_source["claims"]["CP10_1_complete"] is not False
        or group_source["claims"]["CP10_2_complete"] is not False
        or group_source["claims"]["public_lane_ready"] is not False
        or group_source["claims"]["scientific_data_ready"] is not False
        or group_source["claims"]["scientific_result_emitted"] is not False
        or group_source["claims"]["receipt_builder_zero_row_purity"] is not True
        or group_source["claims"]["interactive_audit_zero_row_purity"] is not False
    ):
        raise ResearchPublicationReadinessError("group-scale source audit changed")
    if (
        group_acquisition["status"] != "frozen_metadata_manifest_only_bridge_blocked"
        or group_acquisition["counts"]["metadata_manifests_frozen"] != 1
        or group_acquisition["counts"]["metadata_manifest_bytes_frozen"] != 6278
        or group_acquisition["counts"]["ready_science_lanes"] != 0
        or any(
            group_acquisition["counts"][key] != 0
            for key in (
                "sample_alias_rows_opened",
                "scientific_payload_bytes",
                "scientific_payload_downloads",
                "scientific_payload_rows_opened",
                "scores_computed",
                "target_rows_opened",
                "model_or_paid_calls",
                "network_calls_by_receipt_builder",
            )
        )
        or group_acquisition["claims"]["CP10_1_complete"] is not False
        or group_acquisition["claims"]["CP10_2_complete"] is not False
        or group_acquisition["claims"]["group_scale_bridge_ready"] is not False
        or group_acquisition["claims"]["candidate_tested_on_groups"] is not False
        or group_acquisition["claims"]["publication_claim_supported"] is not False
        or group_acquisition["xcop_overlap_contract"]["executed"] is not False
        or group_acquisition["xcop_overlap_contract"]["input_alias_rows"] != 0
        or group_acquisition["xcop_overlap_contract"]["overlap_count"] is not None
    ):
        raise ResearchPublicationReadinessError("group-scale bridge acquisition changed")
    if (
        missing_variables["status"]
        != "target_blind_defined_proxies_frozen_continuous_sources_or_definitions_blocked"
        or missing_variables["counts"]["variable_families"] != 7
        or missing_variables["counts"]["defined_proxy_contracts"] != 4
        or missing_variables["counts"]["executable_proxy_only_rows"] != 4
        or missing_variables["counts"]["continuous_measurement_ready_rows"] != 0
        or missing_variables["counts"]["source_blocked_applicable_rows"] != 16
        or missing_variables["counts"]["source_definition_blocked_variables"] != 2
        or missing_variables["counts"]["response_or_target_rows_opened"] != 0
        or missing_variables["counts"]["scientific_scores_computed"] != 0
        or missing_variables["claim_boundary"]["all_variable_predictor_contracts_frozen"]
        is not False
        or missing_variables["claim_boundary"]["continuous_missing_variables_measured"] is not False
        or missing_variables["claim_boundary"]["cause_identified"] is not False
        or missing_variables["claim_boundary"]["cross_domain_law_supported"] is not False
        or missing_variables["claim_boundary"]["scientific_scoring_executed"] is not False
        or missing_variables["claim_boundary"]["scientific_claim_allowed"] is not False
        or any(
            missing_variables["chronology_and_access"][key] != 0
            for key in (
                "confirmation_rows_loaded",
                "formula_selection_events",
                "gpu_calls",
                "group_payload_rows_loaded",
                "holdout_rows_loaded",
                "independent_rows_loaded",
                "inferred_mass_rows_loaded",
                "lensing_rows_loaded",
                "model_or_paid_calls",
                "network_calls",
                "new_predictor_source_payload_rows_opened",
                "response_rows_loaded",
                "scientific_scores_computed",
            )
        )
    ):
        raise ResearchPublicationReadinessError("missing-variable preflight evidence changed")
    if (
        data_contract["claims"]["source_metadata_audit_complete"] is not True
        or data_contract["claims"]["independent_source_selected"] is not False
        or data_contract["claims"]["target_rows_accessed"] is not False
        or data_contract["counts"]["fully_ready_lanes"] != 0
        or data_contract["counts"]["candidate_lanes"] != 6
    ):
        raise ResearchPublicationReadinessError("independent data contract changed")
    if (
        act_overlap["status"] != "frozen_metadata_only_unauthorized_catalog_rows_required"
        or act_overlap["counts"]["catalog_bytes_downloaded"] != 0
        or act_overlap["counts"]["catalog_files_downloaded"] != 0
        or act_overlap["counts"]["catalog_rows_opened"] != 0
        or act_overlap["counts"]["overlap_rows"] != 0
        or act_overlap["counts"]["profile_thermodynamic_lensing_target_rows_opened"] != 0
        or act_overlap["counts"]["ready_lanes"] != 0
        or act_overlap["counts"]["scores_computed"] != 0
        or act_overlap["access_and_decision"]["current_catalog_access_authorized"] is not False
        or act_overlap["claims"]["catalog_population_gate_passed"] is not False
        or act_overlap["claims"]["minimum_192_rule_evaluated"] is not False
        or act_overlap["claims"]["overlap_count_computed"] is not False
        or act_overlap["claims"]["independent_replication_ready"] is not False
        or act_overlap["population_gate"]["catalog_overlap_count"] is not None
        or act_overlap["population_gate"]["post_xcop_catalog_upper_bound"] is not None
        or act_overlap["population_gate"]["rule_evaluated"] is not False
        or act_overlap["xcop_exclusion_ledger_contract"]["executed"] is not False
        or act_overlap["xcop_exclusion_ledger_contract"]["input_rows"] != 0
    ):
        raise ResearchPublicationReadinessError("ACT/eRASS overlap preflight changed")
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
        raise ResearchPublicationReadinessError("ACT/eRASS executor evidence changed")
    if (
        comparators["claims"]["matched_comparator_suite_complete"] is not True
        or comparators["claims"]["independent_replication"] is not False
        or comparators["claims"]["full_covariance_used"] is not False
        or comparators["counts"]["comparators"] != 6
        or comparators["counts"]["target_rows_opened"] != 0
    ):
        raise ResearchPublicationReadinessError("matched comparator evidence changed")
    if (
        uncertainty["claims"]["development_nuisance_marginalization_complete"] is not False
        or uncertainty["claims"]["full_source_covariance_complete"] is not False
        or uncertainty["claims"]["independent_replication"] is not False
        or set(uncertainty["completed_goal_evidence"]) != {"CP5.12", "CP5.14"}
        or uncertainty["counts"]["target_rows_opened"] != 0
    ):
        raise ResearchPublicationReadinessError("uncertainty evidence changed")
    if (
        nuisance_diagnostic["completed_goal_evidence"] != {}
        or set(nuisance_diagnostic["blocked_goal_evidence"])
        != {"CP5.7", "CP5.8", "CP5.9", "CP5.10"}
        or nuisance_diagnostic["claims"]["correlation_aware_sampler_materially_improved_mixing"]
        is not True
        or nuisance_diagnostic["claims"]["posterior_sampler_converged"] is not False
        or nuisance_diagnostic["claims"]["development_nuisance_marginalization_complete"]
        is not False
        or nuisance_diagnostic["claims"]["CP5_7_through_CP5_10_complete"] is not False
        or nuisance_diagnostic["counts"]["candidate_forward_evaluations"] != 501636
        or nuisance_diagnostic["counts"]["largest_affine_posterior_draws"] != 115200
        or nuisance_diagnostic["counts"]["parameters_passing_extended_affine_rhat"] != 0
        or nuisance_diagnostic["counts"]["target_rows_opened"] != 0
        or nuisance_diagnostic["counts"]["paid_model_calls"] != 0
    ):
        raise ResearchPublicationReadinessError("nuisance sampler diagnostic changed")
    if (
        nuisance_identifiability["completed_goal_evidence"] != {}
        or set(nuisance_identifiability["blocked_goal_evidence"])
        != {"CP5.7", "CP5.8", "CP5.9", "CP5.10"}
        or nuisance_identifiability["claims"]["tempered_smc_mechanics_passed"] is not True
        or nuisance_identifiability["claims"]["full_posterior_rejuvenation_completed"] is not True
        or nuisance_identifiability["claims"]["more_sampling_alone_supported"] is not False
        or nuisance_identifiability["claims"]["posterior_sampler_converged"] is not False
        or nuisance_identifiability["claims"]["CP5_7_through_CP5_10_complete"] is not False
        or nuisance_identifiability["claims"]["newtonian_control_run"] is not False
        or nuisance_identifiability["counts"]["new_candidate_forward_evaluations"] != 722944
        or nuisance_identifiability["counts"][
            "cumulative_candidate_forward_evaluations_with_predecessor"
        ]
        != 1224580
        or nuisance_identifiability["counts"]["parameters_passing_rejuvenated_rhat"] != 0
        or nuisance_identifiability["counts"]["target_rows_opened"] != 0
        or nuisance_identifiability["counts"]["paid_model_calls"] != 0
    ):
        raise ResearchPublicationReadinessError("nuisance identifiability audit changed")
    if (
        nuisance_quotient["completed_goal_evidence"] != {}
        or set(nuisance_quotient["blocked_goal_evidence"]) != {"CP5.7", "CP5.8", "CP5.9", "CP5.10"}
        or nuisance_quotient["claims"]["maximum_observable_nuisance_dimension"] != 10
        or nuisance_quotient["claims"]["exact_null_dimensions"] != 7
        or nuisance_quotient["claims"]["rank_ten_at_all_frozen_interior_anchors"] is not True
        or nuisance_quotient["claims"]["forward_symmetry_checks_passed"] is not True
        or nuisance_quotient["claims"]["primitive_labels_separately_identified"] is not False
        or nuisance_quotient["claims"]["composite_posterior_converged"] is not False
        or nuisance_quotient["claims"]["CP5_7_through_CP5_10_complete"] is not False
        or nuisance_quotient["claims"]["newtonian_control_run"] is not False
        or nuisance_quotient["counts"]["rank_anchors"] != 16
        or nuisance_quotient["counts"]["frozen_invariance_cases"] != 88
        or nuisance_quotient["counts"]["target_rows_opened"] != 0
        or nuisance_quotient["counts"]["paid_model_calls"] != 0
    ):
        raise ResearchPublicationReadinessError("nuisance quotient audit changed")
    if (
        nuisance_quotient_sampler["status"]
        != "canonical_bounded_controls_and_smoke_only_external_approval_required"
        or nuisance_quotient_sampler["authorization_and_execution"]
        != {
            "production_authorized": False,
            "authorized_manifests_present": False,
            "production_launches": 0,
            "external_approval_required": True,
        }
        or nuisance_quotient_sampler["publication_readiness"]
        != {
            "completed_tasks": 59,
            "open_tasks": 63,
            "total_tasks": 122,
            "CP5_status": "PARTIAL",
            "CP5_7_through_CP5_10": "OPEN",
            "implementation_evidence_only": True,
            "scientific_claims_added": False,
            "candidate_production_claim": False,
        }
        or nuisance_quotient_sampler["frozen_mechanics"]["bounded_smoke_forward_evaluations"] != 852
        or nuisance_quotient_sampler["frozen_mechanics"]["maximum_production_forward_evaluations"]
        != 1_575_104
    ):
        raise ResearchPublicationReadinessError(
            "nuisance quotient sampler implementation evidence changed"
        )
    if (
        quotient_sbc["status"]
        != "strictly_verified_v3_synthetic_pass_newtonian_eligible_production_locked"
        or quotient_sbc["machine_statement"]
        != "V3 synthetic SBC passed; Newtonian-control may unlock; candidate production remains locked"
        or quotient_sbc["adjudication"]["v1_passed"] is not False
        or quotient_sbc["adjudication"]["v2_passed"] is not False
        or quotient_sbc["adjudication"]["v3_synthetic_sbc_passed"] is not True
        or quotient_sbc["adjudication"]["newtonian_control_unlock"] is not True
        or quotient_sbc["adjudication"]["candidate_production_unlock"] is not False
        or quotient_sbc["adjudication"]["v3_synthetic_likelihood_evaluations"] != 24_896_774
        or quotient_sbc["claim_boundary"]["scientific_claim_allowed"] is not False
        or quotient_sbc["diagnostic_evidence_boundary"]["retained_chains_present_in_sealed_npz"]
        is not False
        or quotient_sbc["diagnostic_evidence_boundary"][
            "rhat_and_ess_recomputed_from_retained_chains"
        ]
        is not False
        or quotient_sbc["data_boundary"]
        != {
            "candidate_production_runs": 0,
            "network_calls": 0,
            "newtonian_control_production_runs": 0,
            "paid_model_calls": 0,
            "real_confirmation_rows_loaded": 0,
            "real_development_rows_loaded": 0,
            "real_holdout_rows_loaded": 0,
            "real_independent_rows_loaded": 0,
            "synthetic_data_only": True,
        }
    ):
        raise ResearchPublicationReadinessError("strict quotient SBC evidence changed")
    if (
        newtonian_control["status"] != "package_prepared_strict_v3_pass_external_approval_required"
        or newtonian_control["gates"]
        != {
            "external_approval_present": False,
            "full_matched_newtonian_run_completed": False,
            "production_authorized": False,
            "strict_v3_adjudicator_passed": True,
        }
        or newtonian_control["claim_boundary"]["package_prepared"] is not True
        or newtonian_control["claim_boundary"]["full_matched_newtonian_run_completed"] is not False
        or newtonian_control["claim_boundary"]["scientific_claim_allowed"] is not False
        or newtonian_control["data_boundary"]
        != {
            "network_calls": 0,
            "paid_or_model_calls": 0,
            "production_runs": 0,
            "real_confirmation_rows": 0,
            "real_development_rows": 0,
            "real_holdout_rows": 0,
            "real_independent_rows": 0,
            "synthetic_target_blind_predictor_rows": 80,
        }
        or newtonian_control["run_request"]["maximum_newtonian_control_likelihood_evaluations"]
        != 233_504
        or newtonian_control["run_request"]["maximum_paired_likelihood_evaluations"] != 467_008
        or newtonian_control["run_request"]["maximum_paid_external_cost_usd"] != 0.0
    ):
        raise ResearchPublicationReadinessError("matched Newtonian control evidence changed")
    if (
        pressure_covariance["claims"]["portable_integrity_supported"] is not True
        or pressure_covariance["claims"]["CP5_status_changed"] is not False
        or pressure_covariance["claims"]["scientific_result_changed"] is not False
        or pressure_covariance["claims"]["archive_license_verified"] is not False
        or pressure_covariance["lineage"]
        != {
            "CP5_1_status": "DEVELOPMENT_PRESSURE_COVARIANCE_SCORED_NOT_COMPONENT_COMPLETE",
            "reconstructed_matrices": 8,
            "reconstruction_decision": "DEVELOPMENT_PRESSURE_COVARIANCE_PILOT_RECONSTRUCTIBLE_CP5_REMAINS_PARTIAL",
            "scored_pressure_rows": 54,
            "scoring_decision": "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS",
        }
        or pressure_covariance["counts"]["external_covariance_members_manifested"] != 8
        or pressure_covariance["counts"]["tracked_standalone_pressure_files_verified"] != 8
        or pressure_covariance["counts"]["scientific_payload_rows_read"] != 0
        or pressure_covariance["counts"]["scientific_scores_computed"] != 0
        or pressure_covariance["external_archive_contract"]["included_in_portable_package"]
        is not False
        or pressure_covariance["external_archive_contract"]["required_for_portable_integrity_check"]
        is not False
    ):
        raise ResearchPublicationReadinessError("development pressure covariance evidence changed")
    if (
        a1795_feasibility["status"]
        != "strictly_verified_source_packet_incomplete_cp5_2_through_cp5_6_blocked"
        or a1795_feasibility["adjudication"]["strict_verifier_passed"] is not True
        or a1795_feasibility["adjudication"]["complete_public_covariance_source_packet"]
        is not False
        or a1795_feasibility["adjudication"]["CP5_2_through_CP5_6_complete"] is not False
        or a1795_feasibility["adjudication"]["observation_count"] != 6
        or a1795_feasibility["adjudication"]["planck_product_count"] != 5
        or a1795_feasibility["adjudication"]["planck_public_bytes_manifested"] != 13_314_915_231
        or a1795_feasibility["claim_boundary"]["public_inputs_exist_for_a_new_a1795_reduction"]
        is not True
        or a1795_feasibility["claim_boundary"]["complete_bounded_source_packet_frozen"] is not False
        or a1795_feasibility["claim_boundary"]["publication_claim_supported"] is not False
        or set(a1795_feasibility["cp5_statuses"]) != {"CP5.2", "CP5.3", "CP5.4", "CP5.5", "CP5.6"}
        or not all(
            value.startswith("BLOCKED_") for value in a1795_feasibility["cp5_statuses"].values()
        )
        or any(value != 0 for value in a1795_feasibility["data_boundary"].values())
    ):
        raise ResearchPublicationReadinessError("A1795 covariance feasibility evidence changed")
    if (
        predictor_strata["status"] != "predictor_preflight_pass_scientific_scoring_not_run"
        or predictor_strata["counts"]["development_clusters"] != 8
        or predictor_strata["counts"]["relaxed_proxy"] != 4
        or predictor_strata["counts"]["disturbed_proxy"] != 4
        or predictor_strata["counts"]["cool_core"] != 3
        or predictor_strata["counts"]["non_cool_core"] != 5
        or predictor_strata["counts"]["alternative_causes_mapped"] != 7
        or predictor_strata["counts"]["target_or_response_rows_loaded"] != 0
        or predictor_strata["counts"]["target_scoring_calls"] != 0
        or predictor_strata["readiness"]["CP5_11_predictor_definition_and_labels_ready"] is not True
        or predictor_strata["readiness"]["CP5_11_scientific_stratum_scoring_complete"] is not False
        or predictor_strata["readiness"]["CP5_13_scientific_alternative_cause_comparison_complete"]
        is not False
        or predictor_strata["readiness"]["CP5_13_task_complete"] is not False
        or predictor_strata["claim_boundary"]["cause_identified"] is not False
        or predictor_strata["claim_boundary"]["scientific_claim_allowed"] is not False
        or predictor_strata["data_boundary"]["target_or_response_rows_loaded"] != 0
        or predictor_strata["data_boundary"]["holdout_rows_loaded"] != 0
        or predictor_strata["data_boundary"]["confirmation_rows_loaded"] != 0
        or predictor_strata["data_boundary"]["independent_rows_loaded"] != 0
    ):
        raise ResearchPublicationReadinessError("predictor strata evidence changed")
    strata_gates = strata_scoring["results"]["gates"]
    strata_whole = strata_scoring["results"]["whole_population"]["development_holdout"][
        "full_covariance"
    ]
    if (
        strata_scoring["status"] != "eight_object_exploratory_development_strata_scored"
        or strata_gates["candidate_absolute_primary"]["observed"] != 3.862923367431524
        or strata_gates["candidate_absolute_primary"]["threshold_max"] != 1.0
        or strata_gates["candidate_absolute_primary"]["passed"] is not False
        or strata_gates["candidate_vs_nfw_primary"]["observed_mean_advantage"] != 8.757178295772734
        or strata_gates["candidate_vs_nfw_primary"]["observed_cluster_wins"] != 4
        or strata_gates["candidate_vs_nfw_primary"]["minimum_cluster_wins"] != 5
        or strata_gates["candidate_vs_nfw_primary"]["passed"] is not False
        or strata_gates["covariance_flip_explained_by_any_frozen_stratum"]["passed"] is not False
        or strata_whole["candidate_score_equal_cluster_mean"] != 3.862923367431524
        or strata_whole["nfw_score_equal_cluster_mean"] != 12.62010166320426
        or strata_whole["candidate_advantage_equal_cluster_mean"] != 8.757178295772734
        or strata_whole["candidate_wins"] != 4
        or strata_scoring["results"]["whole_population"]["development_holdout"][
            "positive_diagonal_to_negative_full_clusters"
        ]
        != ["A85", "ZW1215"]
        or strata_scoring["readiness"]["CP5_13_task_complete"] is not False
        or strata_scoring["readiness"]["all_seven_cause_families_scientifically_compared"]
        is not False
        or strata_scoring["claim_boundary"]["strata_explain_covariance_flips"] is not False
        or strata_scoring["claim_boundary"]["causal_variable_identified"] is not False
        or strata_scoring["claim_boundary"]["scientific_claim_allowed"] is not False
        or strata_scoring["claim_boundary"]["A3266_boundary_result_is_singleton_descriptive_only"]
        is not True
        or strata_scoring["compute_and_access_accounting"]["new_raw_target_rows_opened"] != 0
        or strata_scoring["compute_and_access_accounting"]["formula_refits"] != 0
        or strata_scoring["compute_and_access_accounting"]["nuisance_refits"] != 0
        or strata_scoring["compute_and_access_accounting"]["confirmation_rows_opened"] != 0
        or strata_scoring["compute_and_access_accounting"]["independent_rows_opened"] != 0
    ):
        raise ResearchPublicationReadinessError("cluster strata scoring evidence changed")
    if (
        theory_preflight["status"]
        != "blocked_covariant_template_defined_health_and_lensing_not_established"
        or theory_preflight["counts"]["health_gates_total"] != 10
        or theory_preflight["counts"]["template_level_gates_passed"] != 1
        or theory_preflight["counts"]["health_gates_blocked"] != 9
        or theory_preflight["feasibility_adjudication"]["action_is_covariant_template"] is not True
        or theory_preflight["feasibility_adjudication"]["one_universal_matter_photon_metric"]
        is not True
        or theory_preflight["feasibility_adjudication"]["healthy_degrees_of_freedom_proven"]
        is not False
        or theory_preflight["feasibility_adjudication"]["matter_and_lensing_jointly_passed"]
        is not False
        or theory_preflight["feasibility_adjudication"]["theory_feasible_for_observational_scoring"]
        is not False
        or theory_preflight["claim_boundary"]["healthy_action_completed"] is not False
        or theory_preflight["claim_boundary"]["alternative_to_GR_established"] is not False
        or theory_preflight["claim_boundary"]["dark_matter_eliminated"] is not False
        or theory_preflight["claim_boundary"]["scientific_claim_allowed"] is not False
        or any(value != 0 for value in theory_preflight["zero_access_and_compute"].values())
    ):
        raise ResearchPublicationReadinessError("matter+lensing theory preflight changed")
    if (
        symbolic_derivation["status"]
        != "partial_bounded_symbolic_derivation_passed_full_covariant_gates_blocked"
        or symbolic_derivation["counts"]["symbolic_checks"] != 20
        or symbolic_derivation["counts"]["symbolic_checks_passed"] != 20
        or symbolic_derivation["counts"]["independent_numeric_checks"] != 6
        or symbolic_derivation["counts"]["independent_numeric_checks_passed"] != 6
        or symbolic_derivation["adjudication"]["H2_bounded_scalar_euler_lagrange"]
        != "PASS_MACHINE_SYMBOLIC_0P1_FLAT_HOMOGENEOUS_WITH_EXPLICIT_Q_SIGN"
        or symbolic_derivation["adjudication"]["H2_general_covariant_scalar_equations"]
        != "UNVERIFIED_STORED_CONTRACT_ONLY"
        or symbolic_derivation["adjudication"]["H2_full_metric_variation"] != "BLOCKED_NOT_DERIVED"
        or symbolic_derivation["claim_boundary"]["full_H2_passed"] is not False
        or symbolic_derivation["claim_boundary"]["healthy_action_established"] is not False
        or symbolic_derivation["claim_boundary"]["matter_and_lensing_jointly_predicted"]
        is not False
        or symbolic_derivation["claim_boundary"]["scientific_claim_allowed"] is not False
        or any(value != 0 for value in symbolic_derivation["zero_access_and_compute"].values())
    ):
        raise ResearchPublicationReadinessError("bounded symbolic derivation changed")
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
        or external_symbol["numeric_suite"]["probes"][1]["kinetic_determinant_sign"] != "negative"
        or any(value != 0 for value in external_symbol["zero_access_and_compute"].values())
    ):
        raise ResearchPublicationReadinessError("external-metric principal symbol changed")
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
        raise ResearchPublicationReadinessError("conditional kinetic-gate evidence changed")
    group_v3_lanes = {row["lane_id"]: row for row in group_source_v3["lane_readiness"]}
    if (
        group_source_v3["status"] != "frozen_metadata_only_audit_zero_ready_science_lanes"
        or group_source_v3["counts"]
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
        or group_source_v3["future_identity_obsid_acquisition"]["authorized"] is not False
        or group_source_v3["future_identity_obsid_acquisition"]["executed"] is not False
        or group_source_v3["future_xclass_five_object_pilot"]["authorized"] is not False
        or group_source_v3["future_xclass_five_object_pilot"]["executed"] is not False
        or group_source_v3["xcop_overlap_contract"]["executed"] is not False
        or group_source_v3["xcop_overlap_contract"]["overlap_count"] is not None
        or group_source_v3["claims"]["metadata_source_audit_complete"] is not True
        or group_source_v3["claims"]["observational_authorization"] is not False
        or group_source_v3["claims"]["group_bridge_ready"] is not False
        or group_source_v3["claims"]["sample_assembled"] is not False
        or group_source_v3["claims"]["CP10_1_complete"] is not False
        or group_source_v3["claims"]["CP10_2_complete"] is not False
        or any(
            value != 0
            for key, value in group_source_v3["access_chronology"].items()
            if key != "scope"
        )
    ):
        raise ResearchPublicationReadinessError("group-scale V3 source audit changed")
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
        or xclass_executor["column_contract"]["opaque_suffix_exact_bytes"] != 80
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
        raise ResearchPublicationReadinessError("guarded X-CLASS executor evidence changed")
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
        raise ResearchPublicationReadinessError("split-gate source-bound evidence changed")
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
        raise ResearchPublicationReadinessError("universal conformal-source evidence changed")
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
        raise ResearchPublicationReadinessError("Solar/GW necessary-condition evidence changed")
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
        raise ResearchPublicationReadinessError("FLRW necessary-condition evidence changed")
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
        raise ResearchPublicationReadinessError("covariant field-equation evidence changed")
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
        raise ResearchPublicationReadinessError("ADM constraint-propagation evidence changed")
    if (
        numerical["claims"]["all_CP6_tasks_complete"] is not True
        or numerical["claims"]["development_numerical_control_gate_passed"] is not True
        or numerical["claims"]["independent_replication"] is not False
        or numerical["counts"]["item59_variants"] != 2025
        or numerical["counts"]["null_trials"] != 4096
        or numerical["counts"]["target_rows_opened"] != 0
    ):
        raise ResearchPublicationReadinessError("numerical control evidence changed")
    if (
        set(replication_protocol["completed_goal_evidence"])
        != {"CP7.4", "CP7.5", "CP7.6", "CP7.7", "CP7.8", "CP7.10"}
        or set(replication_protocol["blocked_goal_evidence"]) != {"CP7.2", "CP7.3", "CP7.9"}
        or replication_protocol["claims"]["source_selected"] is not False
        or replication_protocol["claims"]["observational_authorization"] is not False
        or replication_protocol["claims"]["target_rows_accessed"] is not False
        or replication_protocol["counts"]["independent_target_rows_opened"] != 0
        or replication_protocol["frozen_decision_summary"]["confirmatory_target_clusters"] != 192
    ):
        raise ResearchPublicationReadinessError("independent replication protocol changed")
    if (
        set(prior_art["completed_goal_evidence"]) != {"CP2.2", "CP2.3", "CP2.4"}
        or set(prior_art["blocked_goal_evidence"]) != {"CP2.5", "CP2.6"}
        or prior_art["claims"]["close_behavioral_equivalent_identified"] is not True
        or prior_art["claims"]["corpus_absence_is_authoritative"] is not False
        or prior_art["claims"]["historical_novelty_established"] is not False
        or prior_art["closest_behavioral_neighbor"]["source_id"]
        != "PENNER_MODIFIED_GRAS_AQUAL_2026"
        or prior_art["counts"]["target_rows_opened"] != 0
    ):
        raise ResearchPublicationReadinessError("prior-art positioning evidence changed")
    if (
        set(manuscript_package["completed_goal_evidence"])
        != {"CP12.2", "CP12.4", "CP12.5", "CP12.7", "CP12.8", "CP12.9"}
        or set(manuscript_package["blocked_goal_evidence"])
        != {"CP12.1", "CP12.3", "CP12.6", "CP12.10", "CP12.11", "CP12.12"}
        or manuscript_package["claims"]["independent_replication"] is not False
        or manuscript_package["claims"]["bounded_paper_ready"] is not False
        or manuscript_package["counts"]["per_row_candidate_predictions"] != 233
        or manuscript_package["counts"]["independent_target_rows_opened"] != 0
    ):
        raise ResearchPublicationReadinessError("manuscript evidence package changed")
    if (
        manuscript_artifacts["completed_goal_evidence"]
        != {
            "CP12.1": "one_command_recreates_all_7_frozen_primary_tables_and_6_frozen_primary_figures"
        }
        or manuscript_artifacts["supersedes_snapshot_blocker"]
        != {
            "source_receipt": "manuscript_evidence_package",
            "goal_task_id": "CP12.1",
            "reason": "The upstream package recorded CP12.1 before this downstream renderer existed.",
        }
        or manuscript_artifacts["counts"]["primary_tables"] != 7
        or manuscript_artifacts["counts"]["primary_figures"] != 6
        or manuscript_artifacts["counts"]["artifacts"] != 13
        or manuscript_artifacts["counts"]["source_candidate_rows"] != 233
        or manuscript_artifacts["counts"]["independent_target_rows_opened"] != 0
        or manuscript_artifacts["claims"]["development_artifacts_reproducible"] is not True
        or manuscript_artifacts["claims"]["every_frozen_primary_table_and_figure_rendered"]
        is not True
        or manuscript_artifacts["claims"]["external_reproduction"] is not False
        or manuscript_artifacts["claims"]["independent_replication"] is not False
        or manuscript_artifacts["claims"]["bounded_paper_ready"] is not False
    ):
        raise ResearchPublicationReadinessError("manuscript artifact evidence changed")


def classify_claim_tracks(outcome: Mapping[str, bool], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Classify bounded, mechanism, and universal claims without cross-track erasure."""

    fields = set(policy["outcome_fields"])
    if set(outcome) != fields or not all(isinstance(value, bool) for value in outcome.values()):
        raise ResearchPublicationReadinessError("claim outcome fields changed")
    tracks = policy["claim_tracks"]
    inherited: list[str] = []
    result = {}
    for track in TRACK_ORDER:
        inherited.extend(map(str, tracks[track]["required_true_fields"]))
        missing = sorted({field for field in inherited if not outcome[field]})
        if track == "universal_theory" and outcome["adjacent_domain_failures"]:
            missing.append("no_adjacent_domain_failures")
        ready = not missing
        if ready:
            status = "READY"
        elif track == "bounded_empirical_publication" and outcome["in_scope_positive_evidence"]:
            status = "DEVELOPMENT_RESULT_RETAINED_NOT_PUBLICATION_READY"
        else:
            status = "BLOCKED"
        result[track] = {
            "status": status,
            "ready": ready,
            "missing_requirements": missing,
        }
    return result


def _gate_readiness(
    gates: Sequence[Mapping[str, Any]], required_ids: Sequence[str]
) -> dict[str, Any]:
    by_id = {str(gate["gate_id"]): gate for gate in gates}
    blockers = [
        {
            "gate_id": gate_id,
            "label": by_id[gate_id]["label"],
            "status": by_id[gate_id]["status"],
        }
        for gate_id in required_ids
        if by_id[gate_id]["status"] != "PASS"
    ]
    return {
        "ready": not blockers,
        "required_gate_ids": list(required_ids),
        "blockers": blockers,
        "next_gate": blockers[0]["gate_id"] if blockers else None,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    policy = load_policy(root)
    project = load_project(root, policy)
    evidence = _load_evidence(root, project["evidence_bindings"])
    tracks = classify_claim_tracks(project["outcome_evidence"], policy)
    predata = _gate_readiness(project["gates"], PRE_DATA_GATES)
    bounded = _gate_readiness(project["gates"], BOUNDED_PAPER_GATES)
    gates = project["gates"]
    goal_path = _under(root, str(project["goal_document_binding"]["path"]), "goal document")
    progress = _goal_task_progress(goal_path, gates)
    if predata["next_gate"] is None:
        next_action = (
            "Keep independent targets sealed and request explicit authorization for the frozen "
            "CP8 replication."
        )
    else:
        next_action = (
            f"Complete {predata['next_gate']} and every remaining pre-data comparator, "
            "covariance, control, and independent-source freeze before requesting target "
            "authorization."
        )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "project_id": project["project_id"],
        "candidate_id": project["candidate"]["candidate_id"],
        "primary_claim_track": project["candidate"]["primary_claim_track"],
        "decision": "DEVELOPMENT_CLUSTER_RESULT_RETAINED_BLOCKED_NOT_DATA_READY",
        "source_bindings": {
            "policy": project["policy_binding"],
            "goal_document": project["goal_document_binding"],
            "evidence": [
                {
                    "evidence_id": binding["evidence_id"],
                    "content_sha256": binding["content_sha256"],
                }
                for binding in project["evidence_bindings"]
            ],
        },
        "claim_tracks": tracks,
        "automatic_findings": {
            "bounded_cluster_result_retained": True,
            "galaxy_failure_erased_cluster_result": False,
            "galaxy_failure_blocks_universal_promotion": True,
            "same_release_xcop_confirmation_is_independent": False,
            "same_release_xcop_classification": (
                "development_confirmation_not_independent_replication"
            ),
            "lensing_candidate_empirically_rejected": False,
            "lensing_target_rows_protected": item60_target_rows(evidence) == 0,
            "screened_descendant_promoted": False,
            "finite_cross_scale_failure_pruned_broader_family": False,
            "nuisance_more_sampling_alone_supported": False,
            "nuisance_identifiability_redesign_required": True,
            "nuisance_exact_composite_coordinates": 10,
            "nuisance_primitive_null_dimensions": 7,
            "nuisance_quotient_sampler_required": True,
            "shared_ben_synthetic_plumbing_validated": True,
            "shared_ben_real_score_exists": False,
            "local_sparc_confirmation_valid_for_ben_descendant": False,
            "group_scale_ready_lanes": 0,
            "CP5_11_predictor_strata_frozen": True,
            "CP5_13_complete": False,
            "frozen_strata_explain_covariance_flips": False,
            "strata_candidate_absolute_gate_passed": False,
            "strata_candidate_object_win_gate_passed": False,
        },
        "readiness": {
            "independent_cluster_data": predata,
            "bounded_cluster_paper": bounded,
            "observational_authorization": False,
            "independent_target_rows_opened": 0,
        },
        "gate_ledger": [
            {
                "gate_id": gate["gate_id"],
                "label": gate["label"],
                "status": gate["status"],
                "required_for_predata": gate["required_for_predata"],
                "required_for_bounded_paper": gate["required_for_bounded_paper"],
                **progress[str(gate["gate_id"])],
            }
            for gate in gates
        ],
        "counts": {
            "claim_tracks": len(TRACK_ORDER),
            "gates": len(gates),
            "tasks": sum(len(gate["task_ids"]) for gate in gates),
            "completed_tasks": sum(
                gate_progress["completed_task_count"] for gate_progress in progress.values()
            ),
            "open_tasks": sum(
                gate_progress["open_task_count"] for gate_progress in progress.values()
            ),
            "pass_gates": sum(gate["status"] == "PASS" for gate in gates),
            "partial_gates": sum(gate["status"] == "PARTIAL" for gate in gates),
            "blocked_gates": sum(gate["status"] == "BLOCKED" for gate in gates),
            "not_started_gates": sum(gate["status"] == "NOT_STARTED" for gate in gates),
            "bound_evidence_receipts": len(evidence),
            "independent_target_rows_opened": 0,
        },
        "claims": {
            "bounded_development_result_retained": True,
            "data_ready_for_independent_cluster_replication": False,
            "bounded_cluster_paper_ready": False,
            "physical_mechanism_ready": False,
            "universal_gravity_theory_ready": False,
            "alternative_to_gr_established": False,
            "dark_matter_eliminated": False,
            "historical_novelty_established": False,
        },
        "next_action": next_action,
    }
    return {**body, "content_sha256": _sha(body)}


def item60_target_rows(evidence: Mapping[str, Mapping[str, Any]]) -> int:
    return int(evidence["item60_direct_lensing_readiness"]["counts"]["direct_target_rows_opened"])


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body):
        raise ResearchPublicationReadinessError("publication readiness receipt hash changed")
    if dict(receipt) != build_receipt(root):
        raise ResearchPublicationReadinessError("publication readiness receipt evidence changed")


def write_receipt(root: Path) -> Path:
    project = load_project(root.resolve(), load_policy(root.resolve()))
    path = root.resolve() / project["output_path"]
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
            "next_action": receipt["next_action"],
            "readiness": receipt["readiness"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
