"""Promote completed compiler receipts through equivalence into the candidate registry."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_ir import compile_action_spec, load_action_grammar
from .campaign import CampaignStore, ClaimedTask, canonical_json, stable_id
from .campaign_engine import CampaignEngine, WorkerOutcome
from .equation_universe import EquationUniverse
from .formal_backend import load_field_contract
from .llm_formula_proposal_adapter import canonical_sha256, sha256_file
from .typed_dsl_campaign_admission import (
    ADMISSION_TASK_TYPE,
    COMPILER_QUEUE_TASK_TYPE,
)

TASK_TYPE = "reviewed_compiler_receipt_registry_admission"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN = re.compile(
    r"dark[-_ ]?matter|\bhalo\b|\bredshift\b|supernova|\bz_b\b|\bT2_m\b|"
    r"\bJ_b\b|\brho_b\b|baryon|non[-_ ]?universal",
    re.IGNORECASE,
)


class CompilerReceiptBridgeError(ValueError):
    """Raised for invalid configuration or unsafe direct admission requests."""


class _Reject(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class _Block(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


@dataclass(frozen=True)
class CallbackBinding:
    callback_id: str
    kind: str
    contract_sha256: str
    source_sha256: str

    @property
    def binding_sha256(self) -> str:
        if self.kind not in {"action_receipt_resolver", "next_stage_adapter"}:
            raise CompilerReceiptBridgeError("invalid registry callback kind")
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]{2,127}", self.callback_id):
            raise CompilerReceiptBridgeError("invalid registry callback ID")
        if not _SHA_RE.fullmatch(self.contract_sha256) or not _SHA_RE.fullmatch(
            self.source_sha256
        ):
            raise CompilerReceiptBridgeError("registry callback is not hash-bound")
        return canonical_sha256(
            {
                "callback_id": self.callback_id,
                "contract_sha256": self.contract_sha256,
                "kind": self.kind,
                "source_sha256": self.source_sha256,
            }
        )


@dataclass(frozen=True)
class RegisteredCallback:
    binding: CallbackBinding
    callback: Callable[..., Any]


class ReviewedRegistryCallbacks:
    def __init__(self, callbacks: tuple[RegisteredCallback, ...] = ()) -> None:
        self._callbacks: dict[tuple[str, str], RegisteredCallback] = {}
        for item in callbacks:
            key = (item.binding.kind, item.binding.callback_id)
            if key in self._callbacks:
                raise CompilerReceiptBridgeError("duplicate registry callback")
            _ = item.binding.binding_sha256
            self._callbacks[key] = item

    def resolve(self, kind: str, callback_id: str | None, binding_sha256: str | None):
        if callback_id is None or binding_sha256 is None:
            raise _Block(f"missing_reviewed_{kind}")
        item = self._callbacks.get((kind, callback_id))
        if item is None:
            raise _Block(f"missing_reviewed_{kind}")
        if item.binding.binding_sha256 != binding_sha256:
            raise _Reject(f"{kind}_binding_hash_mismatch")
        return item.callback


@dataclass(frozen=True)
class BridgeConfig:
    execution_enabled: bool
    grammar_sha256: str
    field_contract_sha256: str
    equation_universe_sha256: str
    source_compiler_binding_sha256: str
    action_resolver_id: str
    action_resolver_binding_sha256: str
    next_stage_adapter_id: str | None
    next_stage_adapter_binding_sha256: str | None
    maximum_task_attempts: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BridgeConfig:
        required = {
            "action_resolver_binding_sha256",
            "action_resolver_id",
            "equation_universe_sha256",
            "execution_enabled",
            "field_contract_sha256",
            "grammar_sha256",
            "maximum_task_attempts",
            "next_stage_adapter_binding_sha256",
            "next_stage_adapter_id",
            "source_compiler_binding_sha256",
        }
        if set(raw) != required or not isinstance(raw["execution_enabled"], bool):
            raise CompilerReceiptBridgeError("registry bridge config shape mismatch")
        for key in (
            "grammar_sha256",
            "field_contract_sha256",
            "equation_universe_sha256",
            "source_compiler_binding_sha256",
            "action_resolver_binding_sha256",
        ):
            if not _SHA_RE.fullmatch(str(raw[key])):
                raise CompilerReceiptBridgeError(f"invalid {key}")
        next_id = raw["next_stage_adapter_id"]
        next_sha = raw["next_stage_adapter_binding_sha256"]
        if (next_id is None) != (next_sha is None):
            raise CompilerReceiptBridgeError("next-stage adapter ID and hash must be paired")
        if next_sha is not None and not _SHA_RE.fullmatch(str(next_sha)):
            raise CompilerReceiptBridgeError("invalid next-stage adapter binding")
        attempts = int(raw["maximum_task_attempts"])
        if not 1 <= attempts <= 8:
            raise CompilerReceiptBridgeError("task attempts outside bounds")
        return cls(
            execution_enabled=raw["execution_enabled"],
            grammar_sha256=str(raw["grammar_sha256"]),
            field_contract_sha256=str(raw["field_contract_sha256"]),
            equation_universe_sha256=str(raw["equation_universe_sha256"]),
            source_compiler_binding_sha256=str(raw["source_compiler_binding_sha256"]),
            action_resolver_id=str(raw["action_resolver_id"]),
            action_resolver_binding_sha256=str(raw["action_resolver_binding_sha256"]),
            next_stage_adapter_id=None if next_id is None else str(next_id),
            next_stage_adapter_binding_sha256=None if next_sha is None else str(next_sha),
            maximum_task_attempts=attempts,
        )


class CompilerReceiptRegistryBridge:
    def __init__(
        self,
        *,
        config: BridgeConfig,
        grammar_path: Path,
        field_contract_path: Path,
        equation_universe_path: Path,
        callbacks: ReviewedRegistryCallbacks,
    ) -> None:
        bindings = (
            (grammar_path, config.grammar_sha256, "grammar"),
            (field_contract_path, config.field_contract_sha256, "field contract"),
            (equation_universe_path, config.equation_universe_sha256, "equation universe"),
        )
        for path, expected, label in bindings:
            if not path.is_file() or sha256_file(path) != expected:
                raise CompilerReceiptBridgeError(f"{label} hash mismatch")
        self.config = config
        self.grammar = load_action_grammar(grammar_path)
        self.field_contract = load_field_contract(field_contract_path)
        self.equation_universe = EquationUniverse(equation_universe_path)
        self.callbacks = callbacks

    def install(self, engine: CampaignEngine) -> None:
        contract = engine.config.get("scientific_contract", {})
        required = {
            "dark_matter_or_halo_inputs": False,
            "observations_authorized": False,
            "redshift_distance_inputs": False,
        }
        if any(contract.get(key) is not value for key, value in required.items()):
            raise CompilerReceiptBridgeError("campaign scientific data seals are not closed")
        self.store = engine.store
        engine.handlers[TASK_TYPE] = self.handle_task

    def enqueue_completed_receipt(
        self, store: CampaignStore, campaign_id: str, compiler_task_id: str
    ) -> str:
        receipt = self._compiler_receipt(store, campaign_id, compiler_task_id)
        payload = {
            "action_ir_sha256": receipt["action_ir_sha256"],
            "compiler_candidate_id": receipt["candidate_id"],
            "compiler_task_id": compiler_task_id,
            "field_contract_sha256": self.config.field_contract_sha256,
            "grammar_sha256": self.config.grammar_sha256,
            "output_sha256": receipt["output_sha256"],
            "proposal_sha256": receipt["proposal_sha256"],
            "validation_artifact_sha256": receipt["validation_artifact_sha256"],
            "work_lineage_sha256": receipt["work_lineage_sha256"],
        }
        return store.add_task(
            campaign_id,
            TASK_TYPE,
            stage=94,
            payload=payload,
            max_attempts=self.config.maximum_task_attempts,
            diversity_bucket="compiler-receipt-registry",
            idempotency_key=stable_id("REGISTRY", campaign_id, compiler_task_id, canonical_sha256(payload)),
        )

    def _compiler_receipt(
        self, store: CampaignStore, campaign_id: str, compiler_task_id: str
    ) -> dict[str, Any]:
        with store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE campaign_id=? AND task_id=?",
                (campaign_id, compiler_task_id),
            ).fetchone()
        if (
            row is None
            or row["task_type"] != COMPILER_QUEUE_TASK_TYPE
            or row["status"] != "succeeded"
        ):
            raise CompilerReceiptBridgeError("source compiler task is not completed")
        result = json.loads(row["result_json"])
        payload = json.loads(row["payload_json"])
        hash_fields = {
            "action_ir_sha256",
            "output_sha256",
            "proposal_sha256",
            "validation_artifact_sha256",
            "work_lineage_sha256",
        }
        if result.get("decision") != "pass" or any(
            not _SHA_RE.fullmatch(str(result.get(key, ""))) for key in hash_fields
        ):
            raise CompilerReceiptBridgeError("compiler result receipt is incomplete")
        if not re.fullmatch(r"CANDLLM-[0-9a-f]{24}", str(result.get("candidate_id", ""))):
            raise CompilerReceiptBridgeError("compiler candidate ID is invalid")
        required = hash_fields | {"candidate_id"}
        if any(payload.get(key) != result.get(key) for key in required):
            raise CompilerReceiptBridgeError("compiler payload/result hash mismatch")
        if payload.get("compiler_callback_binding_sha256") != self.config.source_compiler_binding_sha256:
            raise CompilerReceiptBridgeError("source compiler binding mismatch")
        return result

    def _admission_lineage(self, campaign_id: str, proposal_sha256: str) -> dict[str, Any]:
        with self.store.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM tasks WHERE campaign_id=? AND task_type=? "
                "AND status='succeeded'",
                (campaign_id, ADMISSION_TASK_TYPE),
            ).fetchall()
        payloads = [json.loads(row["payload_json"]) for row in rows]
        matches = [item for item in payloads if item.get("proposal_sha256") == proposal_sha256]
        if len(matches) != 1:
            raise _Reject("compiler_receipt_has_no_unique_admission_predecessor")
        return matches[0]

    def _revalidate(self, task: ClaimedTask) -> dict[str, Any]:
        receipt = self._compiler_receipt(self.store, task.campaign_id, task.payload["compiler_task_id"])
        for key in (
            "action_ir_sha256",
            "output_sha256",
            "proposal_sha256",
            "validation_artifact_sha256",
            "work_lineage_sha256",
        ):
            if task.payload.get(key) != receipt[key]:
                raise _Reject("registry_admission_receipt_replay_tamper")
        if task.payload.get("compiler_candidate_id") != receipt["candidate_id"]:
            raise _Reject("registry_admission_candidate_lineage_tamper")
        if task.payload.get("grammar_sha256") != self.config.grammar_sha256 or task.payload.get(
            "field_contract_sha256"
        ) != self.config.field_contract_sha256:
            raise _Reject("registry_admission_contract_hash_tamper")
        predecessor = self._admission_lineage(task.campaign_id, receipt["proposal_sha256"])
        resolver = self.callbacks.resolve(
            "action_receipt_resolver",
            self.config.action_resolver_id,
            self.config.action_resolver_binding_sha256,
        )
        try:
            resolved = resolver(
                {
                    "action_ir_sha256": receipt["action_ir_sha256"],
                    "compiler_task_id": task.payload["compiler_task_id"],
                    "proposal_sha256": receipt["proposal_sha256"],
                    "work_lineage_sha256": receipt["work_lineage_sha256"],
                }
            )
        except Exception:  # noqa: BLE001 - callback errors must not persist bodies
            raise _Block("reviewed_action_receipt_resolver_failed") from None
        if not isinstance(resolved, Mapping) or set(resolved) != {
            "action_spec",
            "proposal_sha256",
            "schema_version",
            "validation_artifact_sha256",
        }:
            raise _Reject("action_receipt_resolver_schema_mismatch")
        if (
            resolved["schema_version"] != "sigma-action-receipt-resolution-1.0"
            or resolved["proposal_sha256"] != receipt["proposal_sha256"]
            or resolved["validation_artifact_sha256"] != receipt["validation_artifact_sha256"]
        ):
            raise _Reject("action_receipt_resolver_lineage_mismatch")
        action_spec = resolved["action_spec"]
        if not isinstance(action_spec, Mapping):
            raise _Reject("resolved_action_schema_mismatch")
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
        if set(action_spec) != allowed_action_keys or action_spec.get("role") != "candidate":
            raise _Reject("resolved_action_schema_mismatch")
        text = json.dumps(action_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if _FORBIDDEN.search(text):
            raise _Reject("forbidden_input_or_nonuniversal_matter_seal")
        action_ir = compile_action_spec(dict(action_spec), self.grammar, self.field_contract)
        if not action_ir["valid"] or action_ir["content_sha256"] != receipt["action_ir_sha256"]:
            raise _Reject("unsupported_or_hash_mismatched_covariant_action")
        expected_compiler_candidate = stable_id(
            "CANDLLM",
            receipt["proposal_sha256"],
            receipt["action_ir_sha256"],
            self.config.grammar_sha256,
            self.config.field_contract_sha256,
            self.config.source_compiler_binding_sha256,
        )
        if expected_compiler_candidate != receipt["candidate_id"]:
            raise _Reject("compiler_candidate_lineage_hash_mismatch")
        expected_work = canonical_sha256(
            {
                "action_ir_sha256": receipt["action_ir_sha256"],
                "candidate_id": receipt["candidate_id"],
                "compiler_callback_binding_sha256": self.config.source_compiler_binding_sha256,
                "output_sha256": receipt["output_sha256"],
                "proposal_sha256": receipt["proposal_sha256"],
                "quarantine_lineage_sha256": predecessor["quarantine_lineage_sha256"],
                "validation_artifact_sha256": receipt["validation_artifact_sha256"],
            }
        )
        if expected_work != receipt["work_lineage_sha256"]:
            raise _Reject("compiler_work_lineage_hash_mismatch")
        canonical = action_ir["canonical"]
        equivalence_key = {
            "action_ir_sha256": receipt["action_ir_sha256"],
            "fields": canonical["fields"],
            "matter_metric": canonical["matter_metric"],
            "schema_version": "sigma-hash-only-action-equivalence-1.0",
            "terms": [item["id"] for item in canonical["terms"]],
            "universal_constants": canonical["universal_constants"],
        }
        prior = self.equation_universe.classify(
            {
                "name": "hash-only compiler receipt",
                "representation": "tensor_dsl",
                "expression": canonical_json(equivalence_key),
                "variables": [],
            }
        )
        return {
            "action_ir_sha256": receipt["action_ir_sha256"],
            "compiler_candidate_id": receipt["candidate_id"],
            "equivalence_key": equivalence_key,
            "equivalence_semantic_sha256": prior["canonical"]["semantic_hash"],
            "field_ids": canonical["fields"],
            "prior_art_classification": prior["classification"],
            "prior_art_semantic_match_ids": sorted(
                item["equation_id"] for item in prior["semantic_matches"]
            ),
            "proposal_sha256": receipt["proposal_sha256"],
            "term_ids": [item["id"] for item in canonical["terms"]],
            "universal_constant_ids": canonical["universal_constants"],
            "work_lineage_sha256": receipt["work_lineage_sha256"],
        }

    def handle_task(self, task: ClaimedTask) -> WorkerOutcome:
        if not self.config.execution_enabled:
            return self._decision(task, "block", "registry_bridge_disabled", "deferred")
        try:
            evaluated = self._revalidate(task)
            if evaluated["prior_art_classification"] == "known_semantic_equivalent":
                return WorkerOutcome(
                    result={
                        **self._public_result(evaluated),
                        "candidate_registry_admitted": False,
                        "decision": "dedup",
                        "next_stage_tasks_enqueued": 0,
                        "novelty_claim_allowed": False,
                        "reason": "known_equivalent_in_equation_universe",
                    }
                )
            with self.store.connect() as connection:
                existing = connection.execute(
                    "SELECT candidate_id FROM candidates WHERE campaign_id=? AND canonical_json LIKE ?",
                    (task.campaign_id, f'%"action_ir_sha256":"{evaluated["action_ir_sha256"]}"%'),
                ).fetchone()
            if existing is not None:
                return WorkerOutcome(
                    result={
                        **self._public_result(evaluated),
                        "candidate_registry_admitted": False,
                        "decision": "dedup",
                        "duplicate_candidate_id": existing["candidate_id"],
                        "next_stage_tasks_enqueued": 0,
                        "novelty_claim_allowed": False,
                        "reason": "exact_action_ir_already_registered",
                    }
                )
            adapter = self.callbacks.resolve(
                "next_stage_adapter",
                self.config.next_stage_adapter_id,
                self.config.next_stage_adapter_binding_sha256,
            )
            candidate_metadata = {
                "action_ir_sha256": evaluated["action_ir_sha256"],
                "compiler_candidate_id": evaluated["compiler_candidate_id"],
                "equivalence_semantic_sha256": evaluated["equivalence_semantic_sha256"],
                "equation_universe_sha256": self.config.equation_universe_sha256,
                "field_ids": evaluated["field_ids"],
                "novelty_claim_allowed": False,
                "prior_art_classification": evaluated["prior_art_classification"],
                "prior_art_semantic_match_ids": evaluated["prior_art_semantic_match_ids"],
                "proposal_sha256": evaluated["proposal_sha256"],
                "schema_version": "sigma-hash-only-candidate-1.0",
                "term_ids": evaluated["term_ids"],
                "universal_constant_ids": evaluated["universal_constant_ids"],
                "work_lineage_sha256": evaluated["work_lineage_sha256"],
            }
            content_sha = hashlib.sha256(canonical_json(candidate_metadata).encode()).hexdigest()
            candidate_id = stable_id("CAND", task.campaign_id, "llm_typed_covariant", content_sha)
            try:
                next_receipt = adapter(candidate_id, dict(candidate_metadata))
            except Exception:  # noqa: BLE001 - callback errors must persist no bodies
                raise _Block("reviewed_next_stage_adapter_failed") from None
            if not isinstance(next_receipt, Mapping) or set(next_receipt) != {
                "adapter_artifact_sha256",
                "candidate_id",
                "payload",
                "schema_version",
                "task_type",
            }:
                raise _Reject("next_stage_adapter_receipt_schema_mismatch")
            if (
                next_receipt["schema_version"] != "sigma-reviewed-next-stage-adapter-1.0"
                or next_receipt["candidate_id"] != candidate_id
                or next_receipt["task_type"] != "policy_validate"
                or next_receipt["payload"] != {"candidate_id": candidate_id}
                or not _SHA_RE.fullmatch(str(next_receipt["adapter_artifact_sha256"]))
            ):
                raise _Reject("next_stage_adapter_receipt_invalid")
            actual_candidate_id = self.store.add_candidate(
                task.campaign_id,
                kind="llm_typed_covariant",
                expression=f"action-ir-sha256:{evaluated['action_ir_sha256']}",
                canonical=candidate_metadata,
                family_id="LLM-TYPED-COVARIANT",
                mechanism_tags=["typed-covariant", "hash-only-registry"],
            )
            if actual_candidate_id != candidate_id:
                raise _Reject("ordinary_candidate_registry_id_mismatch")
            next_task_id = self.store.add_task(
                task.campaign_id,
                "policy_validate",
                stage=0,
                payload={"candidate_id": candidate_id},
                candidate_id=candidate_id,
                priority=80.0,
                diversity_bucket="LLM-TYPED-COVARIANT",
                idempotency_key=stable_id(
                    "NEXTPOLICY", task.campaign_id, candidate_id, next_receipt["adapter_artifact_sha256"]
                ),
            )
            return WorkerOutcome(
                result={
                    **self._public_result(evaluated),
                    "candidate_id": candidate_id,
                    "candidate_registry_admitted": True,
                    "decision": "pass",
                    "next_stage_task_id": next_task_id,
                    "next_stage_tasks_enqueued": 1,
                    "novelty_claim_allowed": False,
                }
            )
        except _Block as error:
            return self._decision(task, "block", error.reason, "deferred")
        except (_Reject, CompilerReceiptBridgeError, KeyError) as error:
            reason = error.reason if isinstance(error, _Reject) else "receipt_or_payload_invalid"
            return self._decision(task, "reject", reason, "succeeded")

    @staticmethod
    def _public_result(evaluated: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: evaluated[key]
            for key in (
                "action_ir_sha256",
                "compiler_candidate_id",
                "equivalence_semantic_sha256",
                "prior_art_classification",
                "prior_art_semantic_match_ids",
                "proposal_sha256",
                "work_lineage_sha256",
            )
        }

    @staticmethod
    def _decision(task: ClaimedTask, decision: str, reason: str, status: str) -> WorkerOutcome:
        return WorkerOutcome(
            status=status,
            result={
                "candidate_registry_admitted": False,
                "decision": decision,
                "next_stage_tasks_enqueued": 0,
                "novelty_claim_allowed": False,
                "reason": reason,
                "task_id": task.task_id,
            },
        )


def build_readiness_artifact(repo_root: Path, config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "sigma-compiler-receipt-registry-bridge-config-1.0":
        raise CompilerReceiptBridgeError("unexpected registry bridge config schema")
    config = BridgeConfig.from_mapping(raw["bridge"])
    if config.execution_enabled or config.next_stage_adapter_id is not None:
        raise CompilerReceiptBridgeError("checked-in registry bridge must remain disabled")
    paths = raw["paths"]
    for key, expected in (
        ("grammar", config.grammar_sha256),
        ("field_contract", config.field_contract_sha256),
        ("equation_universe", config.equation_universe_sha256),
    ):
        if sha256_file(repo_root / paths[key]) != expected:
            raise CompilerReceiptBridgeError(f"checked-in {key} binding mismatch")
    if set(raw.get("data_eligibility", {}).values()) != {False}:
        raise CompilerReceiptBridgeError("registry bridge data seals must remain closed")
    artifact: dict[str, Any] = {
        "candidate_body_persistence": False,
        "config_sha256": sha256_file(config_path),
        "data_eligibility": raw["data_eligibility"],
        "default_execution_enabled": False,
        "fixture_expected_counts": {
            "block": 1,
            "dedup": 1,
            "enqueue": 1,
            "pass": 1,
            "reject": 7,
        },
        "next_stage_adapter_registered": False,
        "novelty_claim_allowed": False,
        "paid_spend_usd": "0.000000",
        "schema_version": "sigma-compiler-receipt-registry-bridge-readiness-1.0",
        "source_sha256": sha256_file(
            repo_root / "src/sigma_theory_compiler/compiler_receipt_registry_bridge.py"
        ),
        "status": "ready_disabled_hash_only",
        "task_type": TASK_TYPE,
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact
