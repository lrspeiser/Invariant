"""Hash-only typed-DSL admission from LLM quarantine to a compiler work queue."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_ir import compile_action_spec, load_action_grammar
from .campaign import CampaignStore, ClaimedTask, stable_id
from .campaign_engine import CampaignEngine, WorkerOutcome
from .formal_backend import load_field_contract
from .llm_formula_proposal_adapter import (
    canonical_sha256,
    sha256_file,
    validate_proposal_output,
)

ADMISSION_TASK_TYPE = "reviewed_typed_dsl_admission"
COMPILER_QUEUE_TASK_TYPE = "reviewed_covariant_compiler_admission"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_DSL_RE = re.compile(r"^covariant_action\(([A-Z][A-Z0-9_]*(?:,[A-Z][A-Z0-9_]*)*)\)$")
_FORBIDDEN = re.compile(
    r"dark[-_ ]?matter|\bhalo\b|\bredshift\b|supernova|\bz_b\b|\bT2_m\b|"
    r"\bJ_b\b|\brho_b\b|baryon|non[-_ ]?universal|matter[-_ ]?coupling[-_ ]?exception",
    re.IGNORECASE,
)


class TypedDSLAdmissionError(ValueError):
    """Raised for malformed configuration or unsafe direct API use."""


class _ScientificReject(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class _ScientificBlock(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


@dataclass(frozen=True)
class CallbackBinding:
    callback_id: str
    kind: str
    contract_sha256: str
    source_sha256: str
    version: str = "1.0"

    @property
    def binding_sha256(self) -> str:
        if self.kind not in {"quarantine_resolver", "covariant_compiler"}:
            raise TypedDSLAdmissionError("invalid admission callback kind")
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]{2,127}", self.callback_id):
            raise TypedDSLAdmissionError("invalid admission callback ID")
        if not _SHA_RE.fullmatch(self.contract_sha256) or not _SHA_RE.fullmatch(
            self.source_sha256
        ):
            raise TypedDSLAdmissionError("admission callback is not hash-bound")
        return canonical_sha256(
            {
                "callback_id": self.callback_id,
                "contract_sha256": self.contract_sha256,
                "kind": self.kind,
                "source_sha256": self.source_sha256,
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class RegisteredCallback:
    binding: CallbackBinding
    callback: Callable[..., Any]


class ReviewedAdmissionRegistry:
    def __init__(self, callbacks: tuple[RegisteredCallback, ...] = ()) -> None:
        self._callbacks: dict[tuple[str, str], RegisteredCallback] = {}
        for item in callbacks:
            key = (item.binding.kind, item.binding.callback_id)
            if key in self._callbacks:
                raise TypedDSLAdmissionError("duplicate admission callback")
            _ = item.binding.binding_sha256
            self._callbacks[key] = item

    def resolve(self, kind: str, callback_id: str, binding_sha256: str) -> Callable[..., Any]:
        item = self._callbacks.get((kind, callback_id))
        if item is None:
            raise _ScientificBlock(f"missing_reviewed_{kind}_callback")
        if item.binding.binding_sha256 != binding_sha256:
            raise _ScientificReject(f"{kind}_binding_hash_mismatch")
        return item.callback


@dataclass(frozen=True)
class AdmissionConfig:
    execution_enabled: bool
    grammar_sha256: str
    field_contract_sha256: str
    quarantine_resolver_id: str
    quarantine_resolver_binding_sha256: str
    compiler_callback_id: str
    compiler_callback_binding_sha256: str
    maximum_task_attempts: int
    maximum_admissions: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AdmissionConfig:
        required = {
            "compiler_callback_binding_sha256",
            "compiler_callback_id",
            "execution_enabled",
            "field_contract_sha256",
            "grammar_sha256",
            "maximum_admissions",
            "maximum_task_attempts",
            "quarantine_resolver_binding_sha256",
            "quarantine_resolver_id",
        }
        if set(raw) != required:
            raise TypedDSLAdmissionError("typed-DSL admission config shape mismatch")
        if not isinstance(raw["execution_enabled"], bool):
            raise TypedDSLAdmissionError("execution_enabled must be boolean")
        for key in (
            "grammar_sha256",
            "field_contract_sha256",
            "quarantine_resolver_binding_sha256",
            "compiler_callback_binding_sha256",
        ):
            if not _SHA_RE.fullmatch(str(raw[key])):
                raise TypedDSLAdmissionError(f"invalid {key}")
        attempts = int(raw["maximum_task_attempts"])
        maximum = int(raw["maximum_admissions"])
        if not 1 <= attempts <= 8 or not 1 <= maximum <= 10_000:
            raise TypedDSLAdmissionError("typed-DSL admission budgets are outside bounds")
        return cls(
            execution_enabled=raw["execution_enabled"],
            grammar_sha256=str(raw["grammar_sha256"]),
            field_contract_sha256=str(raw["field_contract_sha256"]),
            quarantine_resolver_id=str(raw["quarantine_resolver_id"]),
            quarantine_resolver_binding_sha256=str(
                raw["quarantine_resolver_binding_sha256"]
            ),
            compiler_callback_id=str(raw["compiler_callback_id"]),
            compiler_callback_binding_sha256=str(raw["compiler_callback_binding_sha256"]),
            maximum_task_attempts=attempts,
            maximum_admissions=maximum,
        )


@dataclass(frozen=True)
class _Evaluation:
    action_ir_sha256: str
    candidate_id: str
    proposal_sha256: str
    output_sha256: str
    work_lineage_sha256: str
    validation_artifact_sha256: str


class TypedDSLCampaignAdmission:
    def __init__(
        self,
        *,
        config: AdmissionConfig,
        grammar_path: Path,
        field_contract_path: Path,
        registry: ReviewedAdmissionRegistry,
    ) -> None:
        if sha256_file(grammar_path) != config.grammar_sha256:
            raise TypedDSLAdmissionError("covariant grammar hash mismatch")
        if sha256_file(field_contract_path) != config.field_contract_sha256:
            raise TypedDSLAdmissionError("field contract hash mismatch")
        self.config = config
        self.grammar = load_action_grammar(grammar_path)
        self.field_contract = load_field_contract(field_contract_path)
        self.registry = registry

    def install(self, engine: CampaignEngine) -> None:
        contract = engine.config.get("scientific_contract", {})
        required_seals = {
            "dark_matter_or_halo_inputs": False,
            "observations_authorized": False,
            "redshift_distance_inputs": False,
        }
        if any(contract.get(key) is not expected for key, expected in required_seals.items()):
            raise TypedDSLAdmissionError("campaign scientific data seals are not closed")
        self._installed_store = engine.store
        engine.handlers[ADMISSION_TASK_TYPE] = self.handle_admission_task
        engine.handlers[COMPILER_QUEUE_TASK_TYPE] = self.handle_compiler_task

    def enqueue_quarantine(
        self,
        store: CampaignStore,
        campaign_id: str,
        *,
        quarantine_receipt: Mapping[str, Any],
        raw_output: Mapping[str, Any],
        selected_proposal_sha256: str,
    ) -> str:
        if quarantine_receipt.get("decision") != "quarantined":
            raise TypedDSLAdmissionError("only quarantined proposal receipts are admissible")
        normalized = validate_proposal_output(raw_output)
        output_sha256 = canonical_sha256(normalized)
        if output_sha256 != quarantine_receipt.get("output_sha256"):
            raise TypedDSLAdmissionError("quarantine output hash mismatch at admission")
        proposal_hashes = [canonical_sha256(item) for item in normalized["proposals"]]
        if selected_proposal_sha256 not in proposal_hashes:
            raise TypedDSLAdmissionError("selected proposal hash is absent from quarantine")
        receipt_hashes = quarantine_receipt.get("proposal_sha256", [])
        if selected_proposal_sha256 not in receipt_hashes:
            raise TypedDSLAdmissionError("selected proposal hash is absent from receipt")
        payload = self._safe_payload(
            request_id=str(quarantine_receipt["request_id"]),
            output_sha256=output_sha256,
            proposal_sha256=selected_proposal_sha256,
            quarantine_lineage_sha256=str(quarantine_receipt["lineage_sha256"]),
        )
        with store.connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE campaign_id=? AND task_type=?",
                (campaign_id, ADMISSION_TASK_TYPE),
            ).fetchone()[0]
        if count >= self.config.maximum_admissions:
            raise TypedDSLAdmissionError("typed-DSL admission task budget exhausted")
        return store.add_task(
            campaign_id,
            ADMISSION_TASK_TYPE,
            stage=92,
            payload=payload,
            max_attempts=self.config.maximum_task_attempts,
            diversity_bucket="typed-dsl-admission",
            idempotency_key=stable_id(
                "DSLADMIT", campaign_id, output_sha256, selected_proposal_sha256
            ),
        )

    def _safe_payload(
        self,
        *,
        request_id: str,
        output_sha256: str,
        proposal_sha256: str,
        quarantine_lineage_sha256: str,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", request_id):
            raise TypedDSLAdmissionError("unsafe request ID")
        if any(
            not _SHA_RE.fullmatch(value)
            for value in (output_sha256, proposal_sha256, quarantine_lineage_sha256)
        ):
            raise TypedDSLAdmissionError("admission payload contains an invalid hash")
        return {
            "compiler_callback_binding_sha256": self.config.compiler_callback_binding_sha256,
            "field_contract_sha256": self.config.field_contract_sha256,
            "grammar_sha256": self.config.grammar_sha256,
            "output_sha256": output_sha256,
            "proposal_sha256": proposal_sha256,
            "quarantine_lineage_sha256": quarantine_lineage_sha256,
            "request_id": request_id,
        }

    def _validate_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        expected = self._safe_payload(
            request_id=str(payload.get("request_id", "")),
            output_sha256=str(payload.get("output_sha256", "")),
            proposal_sha256=str(payload.get("proposal_sha256", "")),
            quarantine_lineage_sha256=str(payload.get("quarantine_lineage_sha256", "")),
        )
        if set(payload) != set(expected) or dict(payload) != expected:
            raise _ScientificReject("admission_payload_or_contract_hash_tamper")
        return expected

    def _resolve_output(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        resolver = self.registry.resolve(
            "quarantine_resolver",
            self.config.quarantine_resolver_id,
            self.config.quarantine_resolver_binding_sha256,
        )
        try:
            raw = resolver(
                {
                    "output_sha256": payload["output_sha256"],
                    "proposal_sha256": payload["proposal_sha256"],
                    "request_id": payload["request_id"],
                }
            )
        except Exception:  # noqa: BLE001 - reviewed boundary errors must persist no bodies
            raise _ScientificBlock("reviewed_quarantine_resolver_failed") from None
        if not isinstance(raw, Mapping):
            raise _ScientificReject("quarantine_resolver_schema_mismatch")
        raw_text = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if _FORBIDDEN.search(raw_text):
            raise _ScientificReject("forbidden_input_or_nonuniversal_matter_seal")
        try:
            normalized = validate_proposal_output(raw)
        except Exception:  # noqa: BLE001 - schema errors are a scientific rejection
            raise _ScientificReject("quarantined_proposal_schema_mismatch") from None
        if canonical_sha256(normalized) != payload["output_sha256"]:
            raise _ScientificReject("quarantine_output_hash_mismatch")
        matches = [
            item
            for item in normalized["proposals"]
            if canonical_sha256(item) == payload["proposal_sha256"]
        ]
        if len(matches) != 1:
            raise _ScientificReject("quarantine_proposal_hash_mismatch")
        return matches[0]

    def _typed_terms(self, proposal: Mapping[str, Any]) -> tuple[str, ...]:
        expression = str(proposal["expression"])
        match = _DSL_RE.fullmatch(expression)
        if match is None:
            raise _ScientificReject("unsupported_typed_dsl_operator")
        terms = tuple(match.group(1).split(","))
        if len(terms) != len(set(terms)):
            raise _ScientificReject("duplicate_typed_dsl_term")
        allowed = {item["id"] for item in self.grammar["term_library"]}
        if unknown := sorted(set(terms) - allowed):
            del unknown
            raise _ScientificReject("unsupported_covariant_term")
        return terms

    def _evaluate(self, payload: Mapping[str, Any]) -> _Evaluation:
        safe = self._validate_payload(payload)
        proposal = self._resolve_output(safe)
        terms = self._typed_terms(proposal)
        compiler = self.registry.resolve(
            "covariant_compiler",
            self.config.compiler_callback_id,
            self.config.compiler_callback_binding_sha256,
        )
        typed_packet = {
            "concept_tags": list(proposal["concept_tags"]),
            "parameters": list(proposal["parameters"]),
            "proposal_sha256": safe["proposal_sha256"],
            "terms": list(terms),
            "typed_dsl_version": "sigma-covariant-action-terms-1.0",
        }
        lineage_input = {
            **safe,
            "typed_packet_sha256": canonical_sha256(typed_packet),
        }
        try:
            compiled = compiler(typed_packet, lineage_input)
        except Exception:  # noqa: BLE001 - reviewed boundary errors must persist no bodies
            raise _ScientificBlock("reviewed_covariant_compiler_failed") from None
        required = {
            "action_spec",
            "proposal_sha256",
            "schema_version",
            "validation_artifact_sha256",
        }
        if not isinstance(compiled, Mapping) or set(compiled) != required:
            raise _ScientificReject("reviewed_compiler_receipt_schema_mismatch")
        if compiled["schema_version"] != "sigma-reviewed-covariant-compiler-callback-1.0":
            raise _ScientificReject("reviewed_compiler_receipt_version_mismatch")
        if compiled["proposal_sha256"] != safe["proposal_sha256"]:
            raise _ScientificReject("reviewed_compiler_proposal_hash_mismatch")
        validation_sha = str(compiled["validation_artifact_sha256"])
        if not _SHA_RE.fullmatch(validation_sha):
            raise _ScientificReject("reviewed_compiler_validation_hash_missing")
        action_spec = compiled["action_spec"]
        if not isinstance(action_spec, Mapping):
            raise _ScientificReject("reviewed_compiler_action_spec_schema_mismatch")
        allowed_action_keys = {
            "coefficients",
            "fields",
            "matter_metric",
            "parameter_domain",
            "role",
            "schema_version",
            "static_dictionary_status",
            "terms",
            "universal_constants",
        }
        if set(action_spec) != allowed_action_keys:
            raise _ScientificReject("reviewed_compiler_action_spec_schema_mismatch")
        action_text = json.dumps(
            action_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        if _FORBIDDEN.search(action_text):
            raise _ScientificReject("forbidden_input_or_nonuniversal_matter_seal")
        if action_spec.get("role") != "candidate":
            raise _ScientificReject("reviewed_compiler_candidate_role_mismatch")
        if tuple(action_spec.get("terms", ())) != terms:
            raise _ScientificReject("reviewed_compiler_typed_terms_mismatch")
        term_library = {item["id"]: item for item in self.grammar["term_library"]}
        required_fields = sorted(
            {field for term in terms for field in term_library[term]["fields"]}
        )
        if sorted(action_spec.get("fields", ())) != required_fields:
            raise _ScientificReject("reviewed_compiler_typed_fields_mismatch")
        if sorted(action_spec.get("universal_constants", ())) != sorted(
            proposal["parameters"]
        ):
            raise _ScientificReject("reviewed_compiler_typed_parameters_mismatch")
        action_ir = compile_action_spec(dict(action_spec), self.grammar, self.field_contract)
        if not action_ir["valid"]:
            raise _ScientificReject("covariant_action_grammar_or_matter_contract_rejected")
        action_ir_sha = str(action_ir["content_sha256"])
        candidate_id = stable_id(
            "CANDLLM",
            safe["proposal_sha256"],
            action_ir_sha,
            self.config.grammar_sha256,
            self.config.field_contract_sha256,
            self.config.compiler_callback_binding_sha256,
        )
        work_lineage = canonical_sha256(
            {
                "action_ir_sha256": action_ir_sha,
                "candidate_id": candidate_id,
                "compiler_callback_binding_sha256": self.config.compiler_callback_binding_sha256,
                "output_sha256": safe["output_sha256"],
                "proposal_sha256": safe["proposal_sha256"],
                "quarantine_lineage_sha256": safe["quarantine_lineage_sha256"],
                "validation_artifact_sha256": validation_sha,
            }
        )
        return _Evaluation(
            action_ir_sha256=action_ir_sha,
            candidate_id=candidate_id,
            proposal_sha256=safe["proposal_sha256"],
            output_sha256=safe["output_sha256"],
            work_lineage_sha256=work_lineage,
            validation_artifact_sha256=validation_sha,
        )

    def handle_admission_task(self, task: ClaimedTask) -> WorkerOutcome:
        if not self.config.execution_enabled:
            return self._decision(task, "block", "typed_dsl_admission_disabled", "deferred")
        try:
            evaluation = self._evaluate(task.payload)
        except _ScientificBlock as error:
            return self._decision(task, "block", error.reason, "deferred")
        except (_ScientificReject, TypedDSLAdmissionError) as error:
            reason = error.reason if isinstance(error, _ScientificReject) else "payload_invalid"
            return self._decision(task, "reject", reason, "succeeded")
        compiler_payload = {
            "action_ir_sha256": evaluation.action_ir_sha256,
            "candidate_id": evaluation.candidate_id,
            "compiler_callback_binding_sha256": self.config.compiler_callback_binding_sha256,
            "field_contract_sha256": self.config.field_contract_sha256,
            "grammar_sha256": self.config.grammar_sha256,
            "output_sha256": evaluation.output_sha256,
            "proposal_sha256": evaluation.proposal_sha256,
            "validation_artifact_sha256": evaluation.validation_artifact_sha256,
            "work_lineage_sha256": evaluation.work_lineage_sha256,
        }
        compiler_task_id = stable_id(
            "TASK",
            stable_id("DSLCOMPILE", task.campaign_id, evaluation.work_lineage_sha256),
        )
        actual_task_id = self._store_for_task(task).add_task(
            task.campaign_id,
            COMPILER_QUEUE_TASK_TYPE,
            stage=93,
            payload=compiler_payload,
            max_attempts=self.config.maximum_task_attempts,
            diversity_bucket="covariant-compiler-admission",
            idempotency_key=stable_id(
                "DSLCOMPILE", task.campaign_id, evaluation.work_lineage_sha256
            ),
        )
        if actual_task_id != compiler_task_id:
            raise TypedDSLAdmissionError("compiler queue deterministic task ID mismatch")
        return WorkerOutcome(
            status="succeeded",
            result={
                **self._evaluation_result(evaluation),
                "compiler_queue_task_id": actual_task_id,
                "decision": "pass",
                "enqueued_count": 1,
            },
        )

    def _store_for_task(self, task: ClaimedTask) -> CampaignStore:
        if not hasattr(self, "_installed_store"):
            raise TypedDSLAdmissionError("admission handler was not installed on a campaign engine")
        store: CampaignStore = self._installed_store
        return store

    def handle_compiler_task(self, task: ClaimedTask) -> WorkerOutcome:
        if not self.config.execution_enabled:
            return self._decision(task, "block", "typed_dsl_admission_disabled", "deferred")
        try:
            evaluation = self._evaluate(
                {
                    "compiler_callback_binding_sha256": task.payload.get(
                        "compiler_callback_binding_sha256"
                    ),
                    "field_contract_sha256": task.payload.get("field_contract_sha256"),
                    "grammar_sha256": task.payload.get("grammar_sha256"),
                    "output_sha256": task.payload.get("output_sha256"),
                    "proposal_sha256": task.payload.get("proposal_sha256"),
                    "quarantine_lineage_sha256": self._quarantine_lineage_for_work(task),
                    "request_id": self._request_id_for_work(task),
                }
            )
        except _ScientificBlock as error:
            return self._decision(task, "block", error.reason, "deferred")
        except (_ScientificReject, TypedDSLAdmissionError) as error:
            reason = error.reason if isinstance(error, _ScientificReject) else "compiler_payload_invalid"
            return self._decision(task, "reject", reason, "succeeded")
        expected = self._evaluation_result(evaluation)
        for key in (
            "action_ir_sha256",
            "candidate_id",
            "output_sha256",
            "proposal_sha256",
            "validation_artifact_sha256",
            "work_lineage_sha256",
        ):
            if task.payload.get(key) != expected[key]:
                return self._decision(task, "reject", "compiler_queue_replay_tamper", "succeeded")
        return WorkerOutcome(
            status="succeeded",
            result={**expected, "decision": "pass", "compiler_admission_completed": True},
        )

    def _request_id_for_work(self, task: ClaimedTask) -> str:
        return self._lookup_admission_payload(task, "request_id")

    def _quarantine_lineage_for_work(self, task: ClaimedTask) -> str:
        return self._lookup_admission_payload(task, "quarantine_lineage_sha256")

    def _lookup_admission_payload(self, task: ClaimedTask, key: str) -> str:
        store = self._store_for_task(task)
        with store.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM tasks WHERE campaign_id=? AND task_type=?",
                (task.campaign_id, ADMISSION_TASK_TYPE),
            ).fetchall()
        matches = [
            json.loads(row["payload_json"])
            for row in rows
            if json.loads(row["payload_json"]).get("proposal_sha256")
            == task.payload.get("proposal_sha256")
        ]
        if len(matches) != 1 or key not in matches[0]:
            raise TypedDSLAdmissionError("compiler work has no unique admission predecessor")
        return str(matches[0][key])

    @staticmethod
    def _evaluation_result(evaluation: _Evaluation) -> dict[str, Any]:
        return {
            "action_ir_sha256": evaluation.action_ir_sha256,
            "candidate_id": evaluation.candidate_id,
            "output_sha256": evaluation.output_sha256,
            "proposal_sha256": evaluation.proposal_sha256,
            "validation_artifact_sha256": evaluation.validation_artifact_sha256,
            "work_lineage_sha256": evaluation.work_lineage_sha256,
        }

    @staticmethod
    def _decision(
        task: ClaimedTask, decision: str, reason: str, status: str
    ) -> WorkerOutcome:
        return WorkerOutcome(
            status=status,
            result={
                "compiler_tasks_enqueued": 0,
                "decision": decision,
                "reason": reason,
                "task_id": task.task_id,
            },
        )


def build_admission_readiness_artifact(repo_root: Path, config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "sigma-typed-dsl-campaign-admission-config-1.0":
        raise TypedDSLAdmissionError("unexpected typed-DSL admission config schema")
    config = AdmissionConfig.from_mapping(raw["admission"])
    if config.execution_enabled:
        raise TypedDSLAdmissionError("checked-in typed-DSL admission must remain disabled")
    grammar_path = repo_root / raw["paths"]["grammar"]
    field_contract_path = repo_root / raw["paths"]["field_contract"]
    if sha256_file(grammar_path) != config.grammar_sha256:
        raise TypedDSLAdmissionError("checked-in grammar binding mismatch")
    if sha256_file(field_contract_path) != config.field_contract_sha256:
        raise TypedDSLAdmissionError("checked-in field-contract binding mismatch")
    if set(raw.get("data_eligibility", {}).values()) != {False}:
        raise TypedDSLAdmissionError("typed-DSL admission data seals must all be closed")
    artifact: dict[str, Any] = {
        "admission_task_type": ADMISSION_TASK_TYPE,
        "compiler_queue_task_type": COMPILER_QUEUE_TASK_TYPE,
        "config_sha256": sha256_file(config_path),
        "data_eligibility": raw["data_eligibility"],
        "default_execution_enabled": False,
        "field_contract_sha256": config.field_contract_sha256,
        "fixture_expected_counts": {"block": 1, "enqueue": 1, "pass": 1, "reject": 9},
        "formula_body_persistence": False,
        "grammar_sha256": config.grammar_sha256,
        "paid_spend_usd": "0.000000",
        "schema_version": "sigma-typed-dsl-campaign-admission-readiness-1.0",
        "source_sha256": sha256_file(
            repo_root / "src/sigma_theory_compiler/typed_dsl_campaign_admission.py"
        ),
        "status": "ready_disabled_hash_only",
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact
