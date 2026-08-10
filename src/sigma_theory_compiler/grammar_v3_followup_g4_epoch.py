"""Final reviewed G4 evaluator epoch for the grammar-v3 follow-up queue."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .grammar_v3_followup_g3_epoch import (
    G3_TASK_TYPES,
    GrammarV3FollowupQueueG3Epoch,
)
from .grammar_v3_followup_queue import (
    EVALUATOR_DESCRIPTOR_SCHEMA,
    RESULT_SCHEMA,
    REVIEWED_TASK_TYPES,
    STATUS_SCHEMA,
    _file_sha,
    _load,
    _reject_scalar_truth_score,
    _sha,
    _validate_config,
)
from .persistent_parallel_search import WorkLease
from .promotion_orchestrator import ELIGIBILITY

EPOCH_CONFIG_SCHEMA = "sigma-grammar-v3-followup-queue-config-g4-final-1.0"
G4_TASK_TYPES = {"g4_global_lapse_invertibility", "g4_global_positive_energy"}


def _validate_record(payload: dict[str, Any], record: dict[str, Any]) -> None:
    provenance = record.get("provenance", {})
    if (
        payload.get("task_type") not in G4_TASK_TYPES
        or payload.get("candidate_id") != record.get("seed_id")
        or payload.get("action_sha256") != record.get("action_sha256")
        or record.get("decision") != "pass"
        or record.get("first_missing_premise") is not None
        or provenance.get("data_eligibility") != ELIGIBILITY
        or record.get("solar_bundle", {}).get("generated") is not False
        or record.get("solar_bundle", {}).get("status") != "sealed"
    ):
        raise ValueError("G4 final evaluator candidate, action, or seal binding mismatch")
    gates = record["gate_ledger"]
    if (
        gates["candidate_specific_positive_mass"]["status"] != "pass"
        or gates["global_Einstein_frame_domain"]["status"] != "pass"
        or gates["AF_constraint_and_gauge_formulation"]["status"] != "pass"
        or gates["formal_prerequisite_completion"]["status"] != "pass"
    ):
        raise ValueError("G4 final evaluator premise ledger changed")


def reviewed_g4_global_lapse_evaluator(
    payload: dict[str, Any],
    audit_record: dict[str, Any],
    evaluator_binding_sha256: str,
    audit_binding: dict[str, str],
) -> dict[str, Any]:
    _validate_record(payload, audit_record)
    if payload["task_type"] != "g4_global_lapse_invertibility":
        raise ValueError("G4 lapse evaluator received another task type")
    provenance = audit_record["provenance"]
    return {
        "decision": "pass",
        "blocker": None,
        "evaluator_binding_sha256": evaluator_binding_sha256,
        "audit_binding": audit_binding,
        "audit_candidate_provenance_sha256": provenance["binding_sha256"],
        "action_sha256": provenance["action_sha256"],
        "alternative_formulation_sha256": provenance["alternative_formulation_sha256"],
        "bypass_certificate_sha256": provenance["bypass_certificate_sha256"],
        "resolved_target_blockers": sorted(
            blocker["gate_id"] for blocker in payload["target_blockers"]
        ),
        "global_nonunitary_lapse_status": "pass_in_Einstein_frame_generalized_harmonic_gauge",
        "scientific_candidate_decision_changed": True,
        "data_eligibility": {**ELIGIBILITY, "passed": True},
        "paid_llm_spend_usd": 0.0,
    }


def reviewed_g4_global_positive_energy_evaluator(
    payload: dict[str, Any],
    audit_record: dict[str, Any],
    evaluator_binding_sha256: str,
    audit_binding: dict[str, str],
) -> dict[str, Any]:
    _validate_record(payload, audit_record)
    if payload["task_type"] != "g4_global_positive_energy":
        raise ValueError("G4 positive-energy evaluator received another task type")
    provenance = audit_record["provenance"]
    return {
        "decision": "pass",
        "blocker": None,
        "evaluator_binding_sha256": evaluator_binding_sha256,
        "audit_binding": audit_binding,
        "audit_candidate_provenance_sha256": provenance["binding_sha256"],
        "action_sha256": provenance["action_sha256"],
        "alternative_formulation_sha256": provenance["alternative_formulation_sha256"],
        "bypass_certificate_sha256": provenance["bypass_certificate_sha256"],
        "resolved_target_blockers": sorted(
            blocker["gate_id"] for blocker in payload["target_blockers"]
        ),
        "positive_energy_status": "pass_from_hash_bound_predecessor_in_global_equivalent_frame",
        "remaining_formal_blocker": None,
        "scientific_candidate_decision_changed": True,
        "data_eligibility": {**ELIGIBILITY, "passed": True},
        "paid_llm_spend_usd": 0.0,
    }


def _validate_epoch_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "predecessor_queue_config",
        "pareto_report",
        "reviewed_task_types",
        "reviewed_evaluators",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != EPOCH_CONFIG_SCHEMA:
        raise ValueError("G4 final queue epoch config is invalid")
    if (
        config.get("data_eligibility") != ELIGIBILITY
        or config.get("external_paid_llm_calls") is not False
    ):
        raise ValueError("G4 final queue epoch opened a forbidden input")
    expected = {
        "aether_nonlinear_twist_energy",
        "g2_global_boundary_dirac_contract",
        "g2_global_positive_mass",
        *G3_TASK_TYPES,
        *G4_TASK_TYPES,
    }
    if set(config["reviewed_evaluators"]) != expected:
        raise ValueError("G4 final queue evaluator allowlist changed")


class GrammarV3FollowupQueueG4FinalEpoch(GrammarV3FollowupQueueG3Epoch):
    """Migrate the eight-result G3 epoch and evaluate the final two G4 packets."""

    def __init__(self, coordinator, config: dict[str, Any], project_root: str | Path) -> None:
        _validate_epoch_config(config)
        root = Path(project_root).resolve()
        predecessor_binding = config["predecessor_queue_config"]
        predecessor_path = root / predecessor_binding["path"]
        if (
            not predecessor_path.is_file()
            or _file_sha(predecessor_path) != predecessor_binding["file_sha256"]
        ):
            raise ValueError("G4 final queue predecessor config file mismatch")
        predecessor = _load(predecessor_path)
        if _sha(predecessor) != predecessor_binding["config_sha256"]:
            raise ValueError("G4 final queue predecessor config content mismatch")

        old_binding = predecessor["predecessor_queue_config"]
        old_path = root / old_binding["path"]
        if not old_path.is_file() or _file_sha(old_path) != old_binding["file_sha256"]:
            raise ValueError("G4 final queue base config file mismatch")
        old_config = _load(old_path)
        if _sha(old_config) != old_binding["config_sha256"]:
            raise ValueError("G4 final queue base config content mismatch")
        _validate_config(old_config)

        self.coordinator = coordinator
        self.root = root
        self.config = old_config
        self._load_predecessor_inputs()
        self.evaluators = self._load_reviewed_evaluators()
        self.config = predecessor
        self.evaluators.update(self._load_g3_evaluators())
        self.work_packets = self._build_work_packets()
        predecessor_root = _sha(self.work_packets)
        if predecessor_root != predecessor_binding["queue_registry_root_sha256"]:
            raise ValueError("G4 final queue predecessor registry root mismatch")

        self.config = config
        self.evaluators.update(self._load_g4_evaluators())
        self.work_packets = self._build_work_packets()
        self.queue_registry_root_sha256 = _sha(self.work_packets)
        self._initialize_final_state(predecessor, predecessor_root)
        self.recovered_on_start = self.coordinator.recover_expired()

    def _load_g4_evaluators(self) -> dict[str, dict[str, Any]]:
        loaded = {}
        for task_type in sorted(G4_TASK_TYPES):
            binding = self.config["reviewed_evaluators"][task_type]
            descriptor_path = self.root / binding["descriptor_path"]
            if (
                not descriptor_path.is_file()
                or _file_sha(descriptor_path) != binding["descriptor_file_sha256"]
            ):
                raise ValueError("reviewed G4 evaluator descriptor hash mismatch")
            descriptor = _load(descriptor_path)
            required = {
                "schema_version",
                "evaluator_id",
                "task_type",
                "callback",
                "artifact_path",
                "artifact_sha256",
                "audit_artifact",
                "predecessor_bindings",
                "data_eligibility",
            }
            if (
                set(descriptor) != required
                or descriptor.get("schema_version") != EVALUATOR_DESCRIPTOR_SCHEMA
                or descriptor.get("task_type") != task_type
                or descriptor.get("data_eligibility") != ELIGIBILITY
            ):
                raise ValueError("reviewed G4 evaluator descriptor is invalid")
            callback = (
                reviewed_g4_global_lapse_evaluator
                if task_type == "g4_global_lapse_invertibility"
                else reviewed_g4_global_positive_energy_evaluator
            )
            expected_callback = callback.__module__ + ":" + callback.__name__
            artifact = self.root / descriptor["artifact_path"]
            if (
                descriptor["callback"] != expected_callback
                or artifact.resolve() != Path(__file__).resolve()
                or _file_sha(artifact) != descriptor["artifact_sha256"]
            ):
                raise ValueError("reviewed G4 evaluator callback or source changed")
            audit_binding = descriptor["audit_artifact"]
            audit_path = self.root / audit_binding["path"]
            if not audit_path.is_file() or _file_sha(audit_path) != audit_binding["file_sha256"]:
                raise ValueError("reviewed G4 audit file hash mismatch")
            audit = _load(audit_path)
            audit_body = {key: value for key, value in audit.items() if key != "content_sha256"}
            if (
                audit.get("content_sha256") != audit_binding["content_sha256"]
                or _sha(audit_body) != audit_binding["content_sha256"]
                or audit.get("decision_counts") != {"blocked": 1, "pass": 1}
                or audit.get("target_seed_count") != 2
                or audit.get("full_formal_pass_count") != 1
                or audit.get("nonunitary_bypass_pass_count") != 2
                or audit.get("unitary_chart_obstruction_count") != 2
                or audit.get("observational_data_opened") is not False
                or audit.get("data_eligibility") != ELIGIBILITY
                or audit.get("paid_llm_spend_usd") != 0.0
            ):
                raise ValueError("reviewed G4 audit content or outcome changed")
            records = [item for item in audit["candidate_records"] if item.get("family") == "G4"]
            if len(records) != 1:
                raise ValueError("reviewed G4 audit candidate set changed")
            record = records[0]
            self._validate_g4_predecessors(record, descriptor["predecessor_bindings"])
            loaded[task_type] = {
                "callback": callback,
                "descriptor_binding_sha256": _sha(descriptor),
                "audit_binding": {
                    "file_sha256": audit_binding["file_sha256"],
                    "content_sha256": audit_binding["content_sha256"],
                },
                "records": {record["seed_id"]: record},
            }
        return loaded

    @staticmethod
    def _validate_g4_predecessors(record: dict[str, Any], expected: dict[str, Any]) -> None:
        provenance = record["provenance"]
        actual = {
            "candidate_id": record["seed_id"],
            "action_sha256": record["action_sha256"],
            "candidate_provenance_sha256": provenance["binding_sha256"],
            "predecessor_content_sha256": provenance["predecessor_content_sha256"],
            "predecessor_provenance_sha256": provenance["predecessor_provenance_sha256"],
            "alternative_formulation_sha256": provenance["alternative_formulation_sha256"],
            "bypass_certificate_sha256": provenance["bypass_certificate_sha256"],
        }
        if actual != expected:
            raise ValueError("reviewed G4 predecessor, lapse, or energy evidence mismatch")

    def _initialize_final_state(
        self, predecessor_config: dict[str, Any], predecessor_root: str
    ) -> None:
        expected = {
            "singleton": 1,
            "schema_version": STATUS_SCHEMA,
            "config_sha256": _sha(self.config),
            "pareto_report_content_sha256": self.report["content_sha256"],
            "queue_registry_root_sha256": self.queue_registry_root_sha256,
        }
        predecessor_expected = {
            **expected,
            "config_sha256": _sha(predecessor_config),
            "queue_registry_root_sha256": predecessor_root,
        }
        current = {packet["followup_task_id"]: packet for packet in self.work_packets}
        with self.coordinator.connect() as connection:
            row = connection.execute(
                "SELECT * FROM grammar_v3_followup_adapter WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_followup_adapter VALUES (1,?,?,?,?)",
                    tuple(expected[key] for key in expected if key != "singleton"),
                )
            elif dict(row) == predecessor_expected:
                for stored in connection.execute("SELECT payload_json FROM work").fetchall():
                    payload = json.loads(stored[0])
                    task_id = payload.get("followup_task_id")
                    if task_id not in current or payload != current[task_id]:
                        raise ValueError("G4 final epoch would rewrite completed work lineage")
                connection.execute(
                    "UPDATE grammar_v3_followup_adapter SET config_sha256=?,"
                    "queue_registry_root_sha256=? WHERE singleton=1",
                    (expected["config_sha256"], expected["queue_registry_root_sha256"]),
                )
            elif dict(row) != expected:
                raise ValueError("refusing an unbound G4 final queue epoch transition")

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        payload = self._expected_payload(lease)
        if payload["task_type"] not in G4_TASK_TYPES:
            return super().execute_lease(lease)
        evaluator = self.evaluators.get(payload["task_type"])
        if evaluator is None:
            return super().execute_lease(lease)
        audit_record = evaluator["records"].get(payload["candidate_id"])
        if audit_record is None:
            raise ValueError("reviewed G4 evaluator lacks the queued candidate")
        reviewed_evidence = evaluator["callback"](
            payload,
            audit_record,
            evaluator["descriptor_binding_sha256"],
            evaluator["audit_binding"],
        )
        if (
            reviewed_evidence.get("decision") != "pass"
            or reviewed_evidence.get("blocker") is not None
            or reviewed_evidence.get("scientific_candidate_decision_changed") is not True
            or reviewed_evidence.get("data_eligibility") != {**ELIGIBILITY, "passed": True}
            or reviewed_evidence.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("reviewed G4 pass is not exactly hash-bound and fail-closed")
        result_body = {
            "schema_version": RESULT_SCHEMA,
            "work_id": lease.work_id,
            "followup_task_id": payload["followup_task_id"],
            "followup_lineage_sha256": payload["followup_lineage_sha256"],
            "candidate_id": payload["candidate_id"],
            "task_type": payload["task_type"],
            "decision": "pass",
            "blocker": None,
            "evaluator_invoked": True,
            "reviewed_evidence": reviewed_evidence,
            "scientific_candidate_decision_changed": True,
            "target_blocker_root_sha256": _sha(payload["target_blockers"]),
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**result_body, "content_sha256": _sha(result_body)}

    def status(self) -> dict[str, Any]:
        expected = {packet["followup_task_id"]: packet for packet in self.work_packets}
        records = []
        states: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        evaluator_states: Counter[str] = Counter()
        changed_candidates: set[str] = set()
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT work_id,ordinal,seed,payload_json,state,attempt,result_json,error_text "
                "FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            task_id = payload.get("followup_task_id")
            if task_id not in expected or payload != expected[task_id]:
                raise ValueError("stored grammar-v3 follow-up packet is unregistered")
            state = str(row["state"])
            states[state] += 1
            result_sha = None
            if row["result_json"] is not None:
                result = json.loads(row["result_json"])
                body = {key: item for key, item in result.items() if key != "content_sha256"}
                changed = result.get("scientific_candidate_decision_changed")
                valid_changed = changed is False or (
                    changed is True
                    and payload["task_type"] in G4_TASK_TYPES
                    and result.get("decision") == "pass"
                    and result.get("blocker") is None
                    and result.get("reviewed_evidence", {}).get(
                        "scientific_candidate_decision_changed"
                    )
                    is True
                )
                if (
                    result.get("content_sha256") != _sha(body)
                    or result.get("followup_lineage_sha256") != payload["followup_lineage_sha256"]
                    or not valid_changed
                ):
                    raise ValueError("stored G4 final follow-up result lineage mismatch")
                if changed:
                    changed_candidates.add(payload["candidate_id"])
                decisions[result["decision"]] += 1
                evaluator_states["invoked" if result["evaluator_invoked"] else "missing"] += 1
                result_sha = result["content_sha256"]
            records.append(
                {
                    "work_id": row["work_id"],
                    "followup_task_id": task_id,
                    "followup_lineage_sha256": payload["followup_lineage_sha256"],
                    "candidate_id": payload["candidate_id"],
                    "task_type": payload["task_type"],
                    "pareto_front": payload["pareto_front"],
                    "pareto_axis_values": payload["pareto_axis_values"],
                    "target_blocker_root_sha256": _sha(payload["target_blockers"]),
                    "ordinal": int(row["ordinal"]),
                    "coordinator_seed": int(row["seed"]),
                    "state": state,
                    "attempt": int(row["attempt"]),
                    "result_sha256": result_sha,
                    "error_text": row["error_text"],
                }
            )
        telemetry = self.coordinator.telemetry()
        body = {
            "schema_version": STATUS_SCHEMA,
            "pareto_report_file_sha256": self.config["pareto_report"]["file_sha256"],
            "pareto_report_content_sha256": self.report["content_sha256"],
            "evidence_packet_registry_root_sha256": self.report[
                "evidence_packet_registry_root_sha256"
            ],
            "queue_registry_root_sha256": self.queue_registry_root_sha256,
            "reviewed_task_types": list(REVIEWED_TASK_TYPES),
            "task_count": len(self.work_packets),
            "work_state_counts": dict(sorted(states.items())),
            "followup_decision_counts": dict(sorted(decisions.items())),
            "reviewed_evaluator_invocation_count": evaluator_states["invoked"],
            "missing_evaluator_count": evaluator_states["missing"],
            "candidate_scientific_decisions_changed": len(changed_candidates),
            "work_records": records,
            "work_records_root_sha256": _sha(records),
            "checkpoint_sequence": telemetry["checkpoint_sequence"],
            "recovered_leases": telemetry["recovered_leases"],
            "database_bytes": self._enforce_disk_budget(),
            "budget": dict(self.config["budget"]),
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "interpretation": (
                "The two G4 prerequisite packets pass only through the exact reviewed "
                "Einstein-frame generalized-harmonic audit; all other results retain their "
                "original hash-bound decisions and no observational input is opened."
            ),
        }
        _reject_scalar_truth_score(body)
        return {**body, "content_sha256": _sha(body)}
