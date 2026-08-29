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
GOAL_TASK = re.compile(
    r"^- \[(?P<mark>[ xX])\] \*\*(?P<task>CP(?:[0-9]|1[0-2])\.[0-9]+)\*\*"
)


class ResearchPublicationReadinessError(RuntimeError):
    """Raised when a claim, evidence binding, or publication gate fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8") + b"\n"


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
    if expected is not None and expected != actual:
        raise ResearchPublicationReadinessError("bound evidence content hash changed")
    return actual


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
    expected = {"path", "file_sha256", "content_sha256"} if content else {
        "path",
        "file_sha256",
    }
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

    configured = {
        str(task_id)
        for gate in gates
        for task_id in gate["task_ids"]
    }
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
            raise ResearchPublicationReadinessError(
                f"PASS gate has open goal tasks: {gate_id}"
            )
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


def validate_project(
    project: Mapping[str, Any], policy: Mapping[str, Any], root: Path
) -> None:
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
            raise ResearchPublicationReadinessError(
                f"bounded-paper gate policy changed: {gate_id}"
            )
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
        "independent_data_contract",
        "matched_comparator_suite",
        "uncertainty_program",
        "nuisance_sampler_diagnostic",
        "nuisance_identifiability_audit",
        "nuisance_quotient_audit",
        "nuisance_quotient_sampler_implementation",
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
            raise ResearchPublicationReadinessError(f"evidence semantics changed: {binding['path']}")
        result[str(binding["evidence_id"])] = value
    _validate_gravity_evidence(result)
    return result


def _validate_gravity_evidence(evidence: Mapping[str, Mapping[str, Any]]) -> None:
    item59 = evidence["item59_forward_observable"]
    item60 = evidence["item60_direct_lensing_readiness"]
    item61 = evidence["item61_cross_scale_transfer"]
    descendant = evidence["screened_descendant_adjudication"]
    data_contract = evidence["independent_data_contract"]
    comparators = evidence["matched_comparator_suite"]
    uncertainty = evidence["uncertainty_program"]
    nuisance_diagnostic = evidence["nuisance_sampler_diagnostic"]
    nuisance_identifiability = evidence["nuisance_identifiability_audit"]
    nuisance_quotient = evidence["nuisance_quotient_audit"]
    nuisance_quotient_sampler = evidence["nuisance_quotient_sampler_implementation"]
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
        data_contract["claims"]["source_metadata_audit_complete"] is not True
        or data_contract["claims"]["independent_source_selected"] is not False
        or data_contract["claims"]["target_rows_accessed"] is not False
        or data_contract["counts"]["fully_ready_lanes"] != 0
        or data_contract["counts"]["candidate_lanes"] != 6
    ):
        raise ResearchPublicationReadinessError("independent data contract changed")
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
        or set(nuisance_quotient["blocked_goal_evidence"])
        != {"CP5.7", "CP5.8", "CP5.9", "CP5.10"}
        or nuisance_quotient["claims"]["maximum_observable_nuisance_dimension"] != 10
        or nuisance_quotient["claims"]["exact_null_dimensions"] != 7
        or nuisance_quotient["claims"]["rank_ten_at_all_frozen_interior_anchors"]
        is not True
        or nuisance_quotient["claims"]["forward_symmetry_checks_passed"] is not True
        or nuisance_quotient["claims"]["primitive_labels_separately_identified"]
        is not False
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
        or nuisance_quotient_sampler["frozen_mechanics"][
            "bounded_smoke_forward_evaluations"
        ]
        != 852
        or nuisance_quotient_sampler["frozen_mechanics"][
            "maximum_production_forward_evaluations"
        ]
        != 1_575_104
    ):
        raise ResearchPublicationReadinessError(
            "nuisance quotient sampler implementation evidence changed"
        )
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
        or set(replication_protocol["blocked_goal_evidence"])
        != {"CP7.2", "CP7.3", "CP7.9"}
        or replication_protocol["claims"]["source_selected"] is not False
        or replication_protocol["claims"]["observational_authorization"] is not False
        or replication_protocol["claims"]["target_rows_accessed"] is not False
        or replication_protocol["counts"]["independent_target_rows_opened"] != 0
        or replication_protocol["frozen_decision_summary"]["confirmatory_target_clusters"]
        != 192
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


def classify_claim_tracks(
    outcome: Mapping[str, bool], policy: Mapping[str, Any]
) -> dict[str, Any]:
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
    goal_path = _under(
        root, str(project["goal_document_binding"]["path"]), "goal document"
    )
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
                gate_progress["completed_task_count"]
                for gate_progress in progress.values()
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
