"""Reviewed G3 evaluator epoch for the durable grammar-v3 follow-up queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .grammar_v3_followup_queue import (
    EVALUATOR_DESCRIPTOR_SCHEMA,
    STATUS_SCHEMA,
    GrammarV3FollowupQueue,
    _file_sha,
    _load,
    _reject_scalar_truth_score,
    _sha,
    _validate_config,
)
from .promotion_orchestrator import ELIGIBILITY

EPOCH_CONFIG_SCHEMA = "sigma-grammar-v3-followup-queue-config-g3-epoch-1.0"
G3_TASK_TYPES = {"g3_uniform_interval_cell", "g3_global_lapse_dirac_contract"}


def _blocked_gate_ids(record: dict[str, Any]) -> set[str]:
    return {
        gate_id
        for gate_id, gate in record["gate_ledger"].items()
        if gate["status"] != "pass"
    }


def reviewed_g3_componentwise_interval_evaluator(
    payload: dict[str, Any],
    audit_record: dict[str, Any],
    evaluator_binding_sha256: str,
    audit_binding: dict[str, str],
) -> dict[str, Any]:
    provenance = audit_record.get("provenance", {})
    if (
        payload.get("task_type") != "g3_uniform_interval_cell"
        or payload.get("candidate_id") != audit_record.get("seed_id")
        or payload.get("action_sha256") != audit_record.get("action_sha256")
        or audit_record.get("decision") != "blocked"
        or audit_record.get("resolved_predecessor_blocker")
        != "componentwise_normalized_local_jet_box"
        or audit_record.get("first_missing_premise")
        != "candidate_specific_full_Delta_N_operator"
        or provenance.get("data_eligibility") != ELIGIBILITY
        or audit_record.get("solar_bundle") != {"generated": False, "status": "blocked"}
    ):
        raise ValueError("G3 interval evaluator candidate, action, or seal binding mismatch")
    certificate = audit_record["principal_common_cone_certificate"]
    if (
        certificate.get("status") != "pass_uniform_local_jet_box"
        or audit_record["lapse_prerequisite"]["local_full_operator_invertibility"]
        != "blocked"
        or _blocked_gate_ids(audit_record)
        != {
            "complete_candidate_Delta_N",
            "distributed_Dirac_and_global_lapse",
            "formal_prerequisite_completion",
            "global_hamiltonian_energy",
        }
    ):
        raise ValueError("G3 interval evaluator premise ledger changed")
    return {
        "decision": "blocked",
        "blocker": "candidate_specific_full_Delta_N_operator",
        "evaluator_binding_sha256": evaluator_binding_sha256,
        "audit_binding": audit_binding,
        "audit_candidate_provenance_sha256": provenance["binding_sha256"],
        "action_sha256": provenance["action_sha256"],
        "componentwise_domain_sha256": provenance["componentwise_domain_sha256"],
        "principal_certificate_sha256": provenance["principal_certificate_sha256"],
        "lapse_adapter_evidence_sha256": provenance["lapse_adapter_evidence_sha256"],
        "prior_lapse_certificate_sha256": provenance["lapse_certificate_sha256"],
        "resolved_target_blockers": sorted(
            blocker["gate_id"] for blocker in payload["target_blockers"]
        ),
        "scientific_candidate_decision_changed": False,
        "data_eligibility": {**ELIGIBILITY, "passed": True},
        "paid_llm_spend_usd": 0.0,
    }


def reviewed_g3_full_lapse_dirac_evaluator(
    payload: dict[str, Any],
    audit_record: dict[str, Any],
    evaluator_binding_sha256: str,
    audit_binding: dict[str, str],
) -> dict[str, Any]:
    provenance = audit_record.get("provenance", {})
    if (
        payload.get("task_type") != "g3_global_lapse_dirac_contract"
        or payload.get("candidate_id") != audit_record.get("seed_id")
        or payload.get("action_sha256") != audit_record.get("action_sha256")
        or audit_record.get("decision") != "blocked"
        or audit_record.get("resolved_predecessor_blocker")
        != "candidate_specific_full_Delta_N_operator"
        or audit_record.get("first_missing_premise")
        != "asymptotically_flat_or_global_energy_domain"
        or provenance.get("data_eligibility") != ELIGIBILITY
        or audit_record.get("solar_bundle") != {"generated": False, "status": "blocked"}
    ):
        raise ValueError("G3 lapse evaluator candidate, action, or seal binding mismatch")
    if (
        audit_record["coercivity_certificate"]["function_space_result"]["status"]
        != "pass"
        or audit_record["coercivity_certificate"]["Delta_N_lower_bound"][
            "strictly_positive"
        ]
        is not True
        or _blocked_gate_ids(audit_record)
        != {
            "asymptotically_flat_extension",
            "formal_prerequisite_completion",
            "global_hamiltonian_energy",
        }
    ):
        raise ValueError("G3 full lapse-Dirac evaluator premise ledger changed")
    return {
        "decision": "blocked",
        "blocker": "asymptotically_flat_or_global_energy_domain",
        "evaluator_binding_sha256": evaluator_binding_sha256,
        "audit_binding": audit_binding,
        "audit_candidate_provenance_sha256": provenance["binding_sha256"],
        "action_sha256": provenance["action_sha256"],
        "componentwise_predecessor_content_sha256": provenance[
            "predecessor_content_sha256"
        ],
        "componentwise_predecessor_provenance_sha256": provenance[
            "predecessor_provenance_sha256"
        ],
        "full_lapse_derivation_sha256": provenance["derivation_sha256"],
        "coercivity_sha256": provenance["coercivity_sha256"],
        "operator_domain_sha256": provenance["operator_domain_sha256"],
        "prior_lapse_certificate_sha256": provenance[
            "prior_lapse_certificate_sha256"
        ],
        "resolved_target_blockers": sorted(
            blocker["gate_id"] for blocker in payload["target_blockers"]
        ),
        "global_energy_status": "blocked",
        "scientific_candidate_decision_changed": False,
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
        raise ValueError("G3 follow-up queue epoch config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY or config.get(
        "external_paid_llm_calls"
    ) is not False:
        raise ValueError("G3 follow-up queue epoch opened a forbidden input")
    if set(config["reviewed_evaluators"]) != {
        "aether_nonlinear_twist_energy",
        "g2_global_boundary_dirac_contract",
        "g2_global_positive_mass",
        *G3_TASK_TYPES,
    }:
        raise ValueError("G3 follow-up queue epoch evaluator allowlist changed")


class GrammarV3FollowupQueueG3Epoch(GrammarV3FollowupQueue):
    """Migrate the prior durable queue by adding exactly two reviewed G3 evaluators."""

    def __init__(self, coordinator, config: dict[str, Any], project_root: str | Path) -> None:
        _validate_epoch_config(config)
        root = Path(project_root).resolve()
        predecessor_binding = config["predecessor_queue_config"]
        predecessor_path = root / predecessor_binding["path"]
        if (
            not predecessor_path.is_file()
            or _file_sha(predecessor_path) != predecessor_binding["file_sha256"]
        ):
            raise ValueError("G3 queue epoch predecessor config file mismatch")
        predecessor = _load(predecessor_path)
        if _sha(predecessor) != predecessor_binding["config_sha256"]:
            raise ValueError("G3 queue epoch predecessor config content mismatch")
        _validate_config(predecessor)

        self.coordinator = coordinator
        self.config = predecessor
        self.root = root
        self._load_predecessor_inputs()
        self.evaluators = self._load_reviewed_evaluators()
        self.work_packets = self._build_work_packets()
        predecessor_root = _sha(self.work_packets)
        if predecessor_root != predecessor_binding["queue_registry_root_sha256"]:
            raise ValueError("G3 queue epoch predecessor registry root mismatch")

        self.config = config
        self.evaluators.update(self._load_g3_evaluators())
        self.work_packets = self._build_work_packets()
        self.queue_registry_root_sha256 = _sha(self.work_packets)
        self._initialize_epoch_state(predecessor, predecessor_root)
        self.recovered_on_start = self.coordinator.recover_expired()

    def _load_predecessor_inputs(self) -> None:
        descriptor = self.config["pareto_report"]
        self.report_path = self.root / descriptor["path"]
        if not self.report_path.is_file() or _file_sha(self.report_path) != descriptor[
            "file_sha256"
        ]:
            raise ValueError("bound grammar-v3 Pareto report file mismatch")
        self.report = _load(self.report_path)
        body = {key: value for key, value in self.report.items() if key != "content_sha256"}
        if (
            self.report.get("content_sha256") != descriptor["content_sha256"]
            or _sha(body) != descriptor["content_sha256"]
            or self.report.get("evidence_packet_registry_root_sha256")
            != descriptor["evidence_packet_registry_root_sha256"]
            or self.report.get("candidate_decision_counts") != {"blocked": 6}
            or self.report.get("data_eligibility") != {**ELIGIBILITY, "passed": True}
            or self.report.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("bound grammar-v3 Pareto report content or seals changed")
        _reject_scalar_truth_score(self.report)

    def _load_g3_evaluators(self) -> dict[str, dict[str, Any]]:
        loaded: dict[str, dict[str, Any]] = {}
        for task_type in sorted(G3_TASK_TYPES):
            binding = self.config["reviewed_evaluators"][task_type]
            descriptor_path = self.root / binding["descriptor_path"]
            if (
                not descriptor_path.is_file()
                or _file_sha(descriptor_path) != binding["descriptor_file_sha256"]
            ):
                raise ValueError("reviewed G3 evaluator descriptor hash mismatch")
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
                raise ValueError("reviewed G3 evaluator descriptor is invalid")
            if descriptor["callback"].rsplit(":", 1)[-1] == "reviewed_g3_componentwise_interval_evaluator":
                callback = reviewed_g3_componentwise_interval_evaluator
            elif descriptor["callback"].rsplit(":", 1)[-1] == "reviewed_g3_full_lapse_dirac_evaluator":
                callback = reviewed_g3_full_lapse_dirac_evaluator
            else:
                raise ValueError("reviewed G3 evaluator callback is not allowlisted")
            artifact = self.root / descriptor["artifact_path"]
            if artifact.resolve() != Path(__file__).resolve() or _file_sha(artifact) != descriptor[
                "artifact_sha256"
            ]:
                raise ValueError("reviewed G3 evaluator source binding changed")
            audit_binding = descriptor["audit_artifact"]
            audit_path = self.root / audit_binding["path"]
            if not audit_path.is_file() or _file_sha(audit_path) != audit_binding[
                "file_sha256"
            ]:
                raise ValueError("reviewed G3 audit file hash mismatch")
            audit = _load(audit_path)
            audit_body = {key: value for key, value in audit.items() if key != "content_sha256"}
            if (
                audit.get("content_sha256") != audit_binding["content_sha256"]
                or _sha(audit_body) != audit_binding["content_sha256"]
                or audit.get("decision_counts") != {"blocked": 1}
                or audit.get("target_seed_count") != 1
                or audit.get("full_formal_pass_count") != 0
                or audit.get("observational_data_opened") is not False
                or audit.get("data_eligibility") != ELIGIBILITY
                or audit.get("paid_llm_spend_usd") != 0.0
            ):
                raise ValueError("reviewed G3 audit content or outcome changed")
            record = audit["candidate_records"][0]
            self._validate_g3_predecessors(task_type, record, descriptor["predecessor_bindings"])
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
    def _validate_g3_predecessors(
        task_type: str, record: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        provenance = record["provenance"]
        common = {
            "candidate_id": record["seed_id"],
            "action_sha256": record["action_sha256"],
            "candidate_provenance_sha256": provenance["binding_sha256"],
        }
        if any(expected.get(key) != value for key, value in common.items()):
            raise ValueError("reviewed G3 candidate or action predecessor mismatch")
        if task_type == "g3_uniform_interval_cell":
            actual = {
                "predecessor_content_sha256": provenance["predecessor_content_sha256"],
                "predecessor_provenance_sha256": provenance[
                    "predecessor_provenance_sha256"
                ],
                "componentwise_domain_sha256": provenance["componentwise_domain_sha256"],
                "principal_certificate_sha256": provenance["principal_certificate_sha256"],
                "lapse_adapter_evidence_sha256": provenance[
                    "lapse_adapter_evidence_sha256"
                ],
                "lapse_certificate_sha256": provenance["lapse_certificate_sha256"],
            }
        else:
            actual = {
                "predecessor_content_sha256": provenance["predecessor_content_sha256"],
                "predecessor_provenance_sha256": provenance[
                    "predecessor_provenance_sha256"
                ],
                "principal_certificate_sha256": provenance["principal_certificate_sha256"],
                "prior_lapse_certificate_sha256": provenance[
                    "prior_lapse_certificate_sha256"
                ],
                "derivation_sha256": provenance["derivation_sha256"],
                "coercivity_sha256": provenance["coercivity_sha256"],
                "operator_domain_sha256": provenance["operator_domain_sha256"],
            }
        if any(expected.get(key) != value for key, value in actual.items()):
            raise ValueError("reviewed G3 domain, lapse, or predecessor evidence mismatch")

    def _initialize_epoch_state(
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
        current_packets = {packet["followup_task_id"]: packet for packet in self.work_packets}
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
                    if task_id not in current_packets or payload != current_packets[task_id]:
                        raise ValueError("G3 queue epoch would rewrite completed work lineage")
                connection.execute(
                    "UPDATE grammar_v3_followup_adapter SET config_sha256=?,"
                    "queue_registry_root_sha256=? WHERE singleton=1",
                    (expected["config_sha256"], expected["queue_registry_root_sha256"]),
                )
            elif dict(row) != expected:
                raise ValueError("refusing an unbound G3 queue epoch transition")

