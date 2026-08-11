"""Restart-safe execution of the reviewed, sealed G4 galaxy callback."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY
from .reviewed_g4_candidate_galaxy_evaluator import (
    ACTION_SHA256,
    CALLBACK,
    CANDIDATE_ID,
    DESCRIPTOR_FIELD,
    FORMAL_PROVENANCE_SHA256,
    INPUT_CONTRACT,
    OUTPUT_CONTRACT,
    REQUIRED_REGISTRATION_HASHES,
    reviewed_g4_candidate_galaxy_evaluator,
)

CONFIG_SCHEMA = "sigma-grammar-v3-g4-galaxy-reviewed-execution-config-1.0"
PACKET_SCHEMA = "sigma-grammar-v3-g4-galaxy-reviewed-execution-packet-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-g4-galaxy-reviewed-execution-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-g4-galaxy-reviewed-execution-status-1.0"
DESCRIPTOR_SCHEMA = "sigma-candidate-galaxy-evaluator-descriptor-1.0"
BLOCKER = "missing_registered_galaxy_prediction_and_data_contracts"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"bound G4 galaxy execution artifact changed: {binding['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{binding['path']} must contain an object")
    expected = binding.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        actual = _sha(body) if "content_sha256" in value else _sha(value)
        if actual != expected or (
            "content_sha256" in value and value["content_sha256"] != expected
        ):
            raise ValueError(f"bound G4 galaxy execution content changed: {binding['path']}")
    return value


def _expected_coordinator_identity(
    config: dict[str, Any], packet: dict[str, Any]
) -> tuple[str, int]:
    seed_payload = {
        "master_seed": int(config["coordinator"]["determinism"]["master_seed"]),
        "ordinal": 0,
        "lane": "cpu",
        "payload": packet,
    }
    encoded = _canonical(seed_payload).encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
    return f"PSW-{hashlib.sha256(encoded).hexdigest()[:24]}", seed


class GrammarV3G4GalaxyReviewedExecution:
    """Execute one allowlisted readiness callback without opening galaxy data."""

    def __init__(
        self,
        directory: str | Path,
        config: dict[str, Any],
        project_root: str | Path,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.directory = Path(directory).resolve()
        if "campaign-v1" in str(self.directory).lower():
            raise ValueError("refusing to use the live campaign-v1 database")
        self.config = config
        self._validate_config()
        self.followup = self._validate_followup()
        self.dossier = self._validate_dossier()
        self.formal = self._validate_formal_pass()
        self.contract = self._validate_contract()
        self.descriptor = self._validate_descriptor()
        self.readiness = self._validate_readiness()
        self.packet = self._build_packet()
        profile = _load_bound(self.root, self.config["resource_profile"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.coordinator = PersistentParallelSearch(
            self.directory / "reviewed-g4-galaxy.sqlite",
            self.config["coordinator"],
            profile,
        )
        self._bind_adapter_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _validate_config(self) -> None:
        required = {
            "schema_version",
            "candidate",
            "predecessor_followup_queue",
            "candidate_dossier",
            "formal_pass",
            "prediction_bundle_contract",
            "evaluator_descriptor_allowlist",
            "reviewed_readiness",
            "executor_source",
            "resource_profile",
            "coordinator",
            "observational_opening_authorized",
            "external_paid_llm_calls",
            "data_eligibility",
        }
        if set(self.config) != required or self.config.get("schema_version") != CONFIG_SCHEMA:
            raise ValueError("reviewed G4 galaxy execution config is invalid")
        if self.config["candidate"] != {
            "candidate_id": CANDIDATE_ID,
            "role": "generated_candidate",
            "action_sha256": ACTION_SHA256,
        }:
            raise ValueError("reviewed G4 galaxy execution candidate changed")
        if (
            self.config.get("observational_opening_authorized") is not False
            or self.config.get("external_paid_llm_calls") is not False
            or self.config.get("data_eligibility") != ELIGIBILITY
        ):
            raise ValueError("reviewed G4 galaxy execution opened a forbidden input")
        executor = self.config["executor_source"]
        executor_path = self.root / executor.get("path", "")
        if (
            executor.get("path")
            != "src/sigma_theory_compiler/grammar_v3_g4_galaxy_reviewed_execution.py"
            or not executor_path.is_file()
            or _file_sha(executor_path) != executor.get("file_sha256")
        ):
            raise ValueError("reviewed G4 galaxy execution source changed")
        coordinator = self.config["coordinator"]
        if (
            coordinator.get("external_paid_llm_calls") is not False
            or coordinator.get("queue", {}).get("maximum_pending_work") != 1
            or coordinator.get("queue", {}).get("maximum_attempts") != 3
            or coordinator.get("budget", {}).get("maximum_tasks") != 1
            or coordinator.get("cpu", {}).get("maximum_workers") != 1
            or coordinator.get("supervisor", {}).get("cpu_workers") != 1
            or coordinator.get("supervisor", {}).get("gpu_workers") != 0
        ):
            raise ValueError("reviewed G4 galaxy coordinator is not exactly bounded")

    def _validate_followup(self) -> dict[str, str]:
        binding = self.config["predecessor_followup_queue"]
        status = _load_bound(self.root, binding)
        candidate_records = [
            item
            for item in status.get("work_records", [])
            if item.get("candidate_id") == CANDIDATE_ID
        ]
        if (
            status.get("queue_registry_root_sha256")
            != binding["queue_registry_root_sha256"]
            or status.get("work_records_root_sha256")
            != binding["completed_work_records_root_sha256"]
            or {item.get("task_type") for item in candidate_records}
            != {"g4_global_lapse_invertibility", "g4_global_positive_energy"}
            or any(item.get("state") != "succeeded" for item in candidate_records)
            or status.get("observational_data_opened") is not False
            or status.get("paid_llm_spend_usd") != 0.0
        ):
            raise ValueError("reviewed G4 galaxy follow-up predecessor changed")
        return {
            "queue_registry_root_sha256": binding["queue_registry_root_sha256"],
            "completed_work_records_root_sha256": binding[
                "completed_work_records_root_sha256"
            ],
            "content_sha256": binding["content_sha256"],
        }

    def _validate_dossier(self) -> dict[str, str]:
        binding = self.config["candidate_dossier"]
        artifact = _load_bound(self.root, binding)
        records = [
            item
            for item in artifact.get("dossiers", [])
            if item.get("dossier_id") == CANDIDATE_ID
        ]
        if len(records) != 1 or records[0].get("content_sha256") != binding[
            "candidate_dossier_sha256"
        ]:
            raise ValueError("reviewed G4 galaxy dossier changed")
        nodes = {item["node_id"]: item for item in records[0]["hierarchy_nodes"]}
        if (
            records[0].get("overall_status") != "blocked_after_formal_pass"
            or nodes["defining_covariant_action"].get("action_sha256") != ACTION_SHA256
            or nodes["adm_dirac_obligation"].get("status") != "proven"
            or nodes["principal_symbol_obligation"].get("status") != "proven"
            or nodes["global_energy_obligation"].get("status") != "proven"
        ):
            raise ValueError("reviewed G4 galaxy dossier formal hierarchy changed")
        return {
            "artifact_content_sha256": binding["content_sha256"],
            "candidate_dossier_sha256": binding["candidate_dossier_sha256"],
        }

    def _validate_formal_pass(self) -> dict[str, str]:
        binding = self.config["formal_pass"]
        artifact = _load_bound(self.root, binding)
        records = [
            item
            for item in artifact.get("candidate_records", [])
            if item.get("seed_id") == CANDIDATE_ID
        ]
        if (
            len(records) != 1
            or records[0].get("action_sha256") != ACTION_SHA256
            or records[0].get("decision") != "pass"
            or records[0].get("first_missing_premise") is not None
            or records[0].get("provenance", {}).get("binding_sha256")
            != FORMAL_PROVENANCE_SHA256
            or artifact.get("observational_data_opened") is not False
        ):
            raise ValueError("reviewed G4 galaxy formal pass changed")
        return {
            "artifact_content_sha256": binding["content_sha256"],
            "candidate_provenance_sha256": FORMAL_PROVENANCE_SHA256,
        }

    def _validate_contract(self) -> dict[str, str]:
        binding = self.config["prediction_bundle_contract"]
        contract = _load_bound(self.root, binding)
        properties = contract.get("properties", {})
        if (
            contract.get("$id")
            != "sigma://grammar-v3/g4-galaxy-direct-observable-prediction-bundle-1.0"
            or contract.get("additionalProperties") is not False
            or properties.get("candidate_id", {}).get("const") != CANDIDATE_ID
            or properties.get("action_sha256", {}).get("const") != ACTION_SHA256
            or properties.get("input_contract", {}).get("const") != INPUT_CONTRACT
            or properties.get("output_contract", {}).get("const") != OUTPUT_CONTRACT
            or properties.get("object_specific_gravity_parameter_count", {}).get(
                "const"
            )
            != 0
        ):
            raise ValueError("reviewed G4 galaxy bundle contract changed")
        return {
            "file_sha256": binding["file_sha256"],
            "content_sha256": binding["content_sha256"],
            "object_specific_gravity_parameter_count": 0,
        }

    def _validate_descriptor(self) -> dict[str, Any]:
        allowlist = self.config["evaluator_descriptor_allowlist"]
        if not isinstance(allowlist, list) or len(allowlist) != 1:
            raise ValueError("reviewed G4 galaxy evaluator allowlist changed")
        binding = allowlist[0]
        descriptor = _load_bound(self.root, binding)
        expected = {
            "schema_version": DESCRIPTOR_SCHEMA,
            "evaluator_id": "reviewed-g4-candidate-galaxy-readiness-v1",
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "callback": CALLBACK,
            "artifact_path": "src/sigma_theory_compiler/reviewed_g4_candidate_galaxy_evaluator.py",
            "artifact_sha256": binding["callback_source_sha256"],
            "prediction_bundle_contract_path": self.config[
                "prediction_bundle_contract"
            ]["path"],
            "prediction_bundle_contract_file_sha256": self.contract["file_sha256"],
            "prediction_bundle_contract_content_sha256": self.contract[
                "content_sha256"
            ],
            "data_eligibility": ELIGIBILITY,
        }
        source = self.root / descriptor.get("artifact_path", "")
        if (
            descriptor != expected
            or _sha(descriptor) != binding["descriptor_binding_sha256"]
            or not source.is_file()
            or _file_sha(source) != descriptor["artifact_sha256"]
        ):
            raise ValueError("reviewed G4 galaxy evaluator is not exactly allowlisted")
        return descriptor

    def _validate_readiness(self) -> dict[str, Any]:
        binding = self.config["reviewed_readiness"]
        readiness = _load_bound(self.root, binding)
        decision = readiness.get("current_evaluator_decision", {})
        expected_missing = sorted(set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD})
        if (
            readiness.get("candidate", {}).get("candidate_id") != CANDIDATE_ID
            or readiness.get("candidate", {}).get("action_sha256") != ACTION_SHA256
            or readiness.get("descriptor_implementation_ready") is not True
            or readiness.get("prediction_bundle_registered") is not False
            or readiness.get("candidate_use_authorized") is not False
            or readiness.get("observational_data_opened") is not False
            or readiness.get("primary_record_access_count") != 0
            or decision.get("decision") != "blocked"
            or decision.get("blocker") != BLOCKER
            or decision.get("filled_registration_hash_count") != 1
            or decision.get("missing_registration_hashes") != expected_missing
            or readiness.get("implementation_readiness", {}).get(
                "descriptor_binding_sha256"
            )
            != self.config["evaluator_descriptor_allowlist"][0][
                "descriptor_binding_sha256"
            ]
            or readiness.get("dark_matter_or_halo_inputs") is not False
            or readiness.get("redshift_distance_inputs") is not False
        ):
            raise ValueError("reviewed G4 galaxy readiness binding changed")
        return readiness

    def _build_packet(self) -> dict[str, Any]:
        registration = {name: None for name in REQUIRED_REGISTRATION_HASHES}
        registration[DESCRIPTOR_FIELD] = self.config["evaluator_descriptor_allowlist"][0][
            "descriptor_binding_sha256"
        ]
        body = {
            "schema_version": PACKET_SCHEMA,
            "ordinal": 0,
            "candidate": self.config["candidate"],
            "predecessor_followup_queue": self.followup,
            "candidate_dossier": self.dossier,
            "formal_pass": self.formal,
            "prediction_bundle_contract": self.contract,
            "readiness_content_sha256": self.config["reviewed_readiness"][
                "content_sha256"
            ],
            "evaluator_descriptor_binding_sha256": registration[DESCRIPTOR_FIELD],
            "callback": CALLBACK,
            "registration_hashes": registration,
            "prediction_bundle_registered": False,
            "observational_opening_authorized": False,
            "data_eligibility": ELIGIBILITY,
        }
        lineage = _sha(body)
        return {
            **body,
            "followup_task_id": f"G4GALREV-{lineage[:24]}",
            "followup_lineage_sha256": lineage,
        }

    def _bind_adapter_state(self) -> None:
        expected = {
            "schema_version": STATUS_SCHEMA,
            "config_sha256": _sha(self.config),
            "packet_root_sha256": _sha([self.packet]),
        }
        with self.coordinator.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS g4_galaxy_reviewed_adapter ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1),"
                "schema_version TEXT NOT NULL,config_sha256 TEXT NOT NULL,"
                "packet_root_sha256 TEXT NOT NULL)"
            )
            row = connection.execute("SELECT * FROM g4_galaxy_reviewed_adapter").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO g4_galaxy_reviewed_adapter VALUES (1,?,?,?)",
                    tuple(expected.values()),
                )
            elif {key: row[key] for key in expected} != expected:
                raise ValueError("refusing to resume a changed reviewed G4 galaxy execution")

    def enqueue(self) -> dict[str, int]:
        result = self.coordinator.enqueue([self.packet], lane="cpu", max_attempts=3)
        return {**result, "requested": 1}

    def _validate_lease(self, lease: WorkLease) -> None:
        work_id, seed = _expected_coordinator_identity(self.config, self.packet)
        if (
            lease.payload != self.packet
            or lease.ordinal != 0
            or lease.lane != "cpu"
            or lease.work_id != work_id
            or lease.seed != seed
        ):
            raise ValueError("reviewed G4 galaxy lease or lineage changed")

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        self._validate_lease(lease)
        reviewed = reviewed_g4_candidate_galaxy_evaluator(
            {
                "candidate_id": CANDIDATE_ID,
                "action_sha256": ACTION_SHA256,
                "role": "generated_candidate",
                "data_eligibility": dict(ELIGIBILITY),
            },
            {
                "data_eligibility": dict(ELIGIBILITY),
                "observational_opening_authorized": False,
                "registration_hashes": dict(self.packet["registration_hashes"]),
            },
        )
        expected_missing = sorted(set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD})
        if (
            reviewed.get("decision") != "blocked"
            or reviewed.get("blocker") != BLOCKER
            or reviewed.get("filled_registration_hash_count") != 1
            or reviewed.get("missing_registration_hashes") != expected_missing
            or reviewed.get("observational_data_opened") is not False
            or reviewed.get("primary_record_access_count") != 0
            or reviewed.get("dark_matter_or_halo_inputs") is not False
            or reviewed.get("redshift_distance_inputs") is not False
        ):
            raise ValueError("reviewed G4 galaxy callback changed or opened data")
        body = {
            "schema_version": RESULT_SCHEMA,
            "work_id": lease.work_id,
            "followup_task_id": self.packet["followup_task_id"],
            "followup_lineage_sha256": self.packet["followup_lineage_sha256"],
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "decision": "blocked",
            "blocker": BLOCKER,
            "reviewed_evaluator_invoked": True,
            "evaluator_descriptor_binding_sha256": self.packet[
                "evaluator_descriptor_binding_sha256"
            ],
            "filled_registration_hash_count": 1,
            "missing_registration_hashes": expected_missing,
            "prediction_bundle_registered": False,
            "object_specific_gravity_parameter_count": 0,
            "candidate_use_authorized": False,
            "observational_opening_authorized": False,
            "observational_data_opened": False,
            "primary_record_access_count": 0,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}

    def run_bounded(self) -> dict[str, Any]:
        processed = 0
        worker = "reviewed-g4-galaxy-worker"
        lease = self.coordinator.claim("cpu", worker)
        if lease is not None:
            try:
                result = self.execute_lease(lease)
                if not self.coordinator.finish(lease, worker, result):
                    raise RuntimeError("reviewed G4 galaxy lease was lost")
                processed = 1
            except BaseException as error:
                self.coordinator.fail(lease, worker, f"{type(error).__name__}: {error}")
                raise
        if processed:
            self.coordinator.checkpoint()
        return {"processed": processed, "status": self.status()}

    def status(self) -> dict[str, Any]:
        expected_work_id, expected_seed = _expected_coordinator_identity(
            self.config, self.packet
        )
        records: list[dict[str, Any]] = []
        states: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        invoked = 0
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT work_id,ordinal,lane,seed,payload_json,state,attempt,result_json,"
                "error_text FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            if (
                payload != self.packet
                or row["work_id"] != expected_work_id
                or int(row["ordinal"]) != 0
                or row["lane"] != "cpu"
                or int(row["seed"]) != expected_seed
            ):
                raise ValueError("stored reviewed G4 galaxy work lineage changed")
            state = str(row["state"])
            states[state] += 1
            result_sha = None
            if row["result_json"] is not None:
                result = json.loads(row["result_json"])
                body = {key: item for key, item in result.items() if key != "content_sha256"}
                if (
                    result.get("content_sha256") != _sha(body)
                    or result.get("followup_lineage_sha256")
                    != self.packet["followup_lineage_sha256"]
                    or result.get("decision") != "blocked"
                    or result.get("blocker") != BLOCKER
                    or result.get("prediction_bundle_registered") is not False
                    or result.get("object_specific_gravity_parameter_count") != 0
                    or result.get("observational_data_opened") is not False
                ):
                    raise ValueError("stored reviewed G4 galaxy result changed")
                decisions[result["decision"]] += 1
                invoked += 1
                result_sha = result["content_sha256"]
            records.append(
                {
                    "work_id": row["work_id"],
                    "followup_task_id": payload["followup_task_id"],
                    "followup_lineage_sha256": payload["followup_lineage_sha256"],
                    "candidate_id": CANDIDATE_ID,
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
            "candidate_id": CANDIDATE_ID,
            "action_sha256": ACTION_SHA256,
            "predecessor_followup_queue": self.followup,
            "candidate_dossier": self.dossier,
            "formal_pass": self.formal,
            "prediction_bundle_contract": self.contract,
            "evaluator_descriptor_allowlist_count": 1,
            "evaluator_descriptor_binding_sha256": self.packet[
                "evaluator_descriptor_binding_sha256"
            ],
            "queue_registry_root_sha256": _sha([self.packet]),
            "task_count": len(records),
            "work_state_counts": dict(sorted(states.items())),
            "decision_counts": dict(sorted(decisions.items())),
            "reviewed_evaluator_invocation_count": invoked,
            "filled_registration_hash_count": 1,
            "missing_registration_hash_count": 17,
            "missing_registration_hashes": sorted(
                set(REQUIRED_REGISTRATION_HASHES) - {DESCRIPTOR_FIELD}
            ),
            "prediction_bundle_registered": False,
            "object_specific_gravity_parameter_count": 0,
            "work_records": records,
            "work_records_root_sha256": _sha(records),
            "checkpoint_sequence": telemetry["checkpoint_sequence"],
            "recovered_leases": telemetry["recovered_leases"],
            "restart_safe": True,
            "portable_status": True,
            "candidate_use_authorized": False,
            "observational_opening_authorized": False,
            "observational_data_opened": False,
            "primary_record_access_count": 0,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "tracking_target_values_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "interpretation": (
                "The exact reviewed G4 galaxy callback ran under one durable private lease and "
                "reproduced its sealed blocker. No prediction bundle, observation, halo label, "
                "redshift distance, or object-specific gravity parameter was admitted."
            ),
        }
        return {**body, "content_sha256": _sha(body)}

    def export(self) -> dict[str, Any]:
        return self.status()
