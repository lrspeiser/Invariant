from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_campaign import iter_scalable_seed_specs
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY

ADAPTER_SCHEMA = "sigma-grammar-v3-seed-execution-adapter-1.0"
PAYLOAD_SCHEMA = "sigma-grammar-v3-seed-work-1.0"
CALLBACK_SCHEMA = "sigma-grammar-v3-seed-callback-descriptor-1.0"
CALLBACK_RESULT_SCHEMA = "sigma-grammar-v3-seed-reviewed-result-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-seed-execution-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-seed-execution-status-1.0"
EXPECTED_SEED_SCHEMA = "sigma-covariant-grammar-v3-concrete-seed-1.0"
SeedCallback = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("grammar-v3 manifest must contain a JSON object")
    return value


def _is_sha(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _validate_seed(seed: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "family_id",
        "family_lineage_sha256",
        "parameter_index",
        "parameters",
        "operator_atoms",
        "theory_contract",
        "data_eligibility",
        "seed_id",
        "seed_lineage_sha256",
    }
    if set(seed) != required or seed.get("schema_version") != EXPECTED_SEED_SCHEMA:
        raise ValueError("grammar-v3 concrete seed fields or schema are invalid")
    if seed.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 concrete seed eligibility is not fail-closed")
    body = {key: value for key, value in seed.items() if key not in {"seed_id", "seed_lineage_sha256"}}
    lineage = _sha(body)
    if seed.get("seed_lineage_sha256") != lineage or seed.get("seed_id") != (
        "G3-" + lineage[:24]
    ):
        raise ValueError("grammar-v3 concrete seed lineage mismatch")
    if not _is_sha(seed.get("family_lineage_sha256")):
        raise ValueError("grammar-v3 family lineage is invalid")


def callback_binding(descriptor: dict[str, Any]) -> str:
    required = {
        "schema_version",
        "callback_id",
        "callback",
        "artifact_path",
        "artifact_sha256",
        "data_eligibility",
    }
    if set(descriptor) != required or descriptor.get("schema_version") != CALLBACK_SCHEMA:
        raise ValueError("grammar-v3 callback descriptor fields or schema are invalid")
    if descriptor.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 callback eligibility is not fail-closed")
    artifact = Path(str(descriptor["artifact_path"])).resolve()
    if not artifact.is_file() or _file_sha(artifact) != descriptor.get("artifact_sha256"):
        raise ValueError("grammar-v3 callback artifact hash mismatch")
    return _sha({key: value for key, value in descriptor.items() if key != "artifact_path"})


def _resolve_callback(descriptor: dict[str, Any]) -> SeedCallback:
    module_name, separator, attribute = str(descriptor["callback"]).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("grammar-v3 callback must use module:function syntax")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError("grammar-v3 reviewed callback is not callable")
    source = inspect.getsourcefile(callback)
    if source is None or Path(source).resolve() != Path(descriptor["artifact_path"]).resolve():
        raise ValueError("grammar-v3 callback is not defined by its bound artifact")
    return callback


STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS grammar_v3_seed_adapter (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  manifest_file_sha256 TEXT NOT NULL,
  manifest_content_sha256 TEXT NOT NULL,
  callback_registry_root_sha256 TEXT NOT NULL,
  seed_registry_root_sha256 TEXT NOT NULL,
  adapter_config_sha256 TEXT NOT NULL
);
"""


class GrammarV3SeedExecution:
    """Bounded grammar-v3 seed adapter over the durable parallel-search coordinator."""

    def __init__(
        self,
        coordinator: PersistentParallelSearch,
        manifest_path: str | Path,
        *,
        expected_manifest_file_sha256: str,
        expected_manifest_content_sha256: str,
        callback_descriptor: dict[str, Any] | None = None,
        maximum_seeds: int = 6,
    ) -> None:
        if not 1 <= maximum_seeds <= 6:
            raise ValueError("grammar-v3 seed execution bound must be between one and six")
        self.coordinator = coordinator
        self.manifest_path = Path(manifest_path).resolve()
        if (
            not self.manifest_path.is_file()
            or _file_sha(self.manifest_path) != expected_manifest_file_sha256
        ):
            raise ValueError("grammar-v3 seed manifest file hash mismatch")
        self.manifest = _load(self.manifest_path)
        body = {
            key: value for key, value in self.manifest.items() if key != "content_sha256"
        }
        if (
            self.manifest.get("content_sha256") != _sha(body)
            or self.manifest.get("content_sha256") != expected_manifest_content_sha256
        ):
            raise ValueError("grammar-v3 seed manifest content hash mismatch")
        if self.manifest.get("data_eligibility") != ELIGIBILITY:
            raise ValueError("grammar-v3 seed manifest eligibility is not fail-closed")
        if self.manifest.get("observational_data_opened") is not False:
            raise ValueError("grammar-v3 seed manifest opened observations")
        self.manifest_file_sha256 = expected_manifest_file_sha256
        self.manifest_content_sha256 = expected_manifest_content_sha256
        self.seeds = list(iter_scalable_seed_specs(self.manifest))
        declared = self.manifest.get("scalable_generator_hook", {})
        if (
            len(self.seeds) != int(declared.get("concrete_seed_count", -1))
            or len(self.seeds) != maximum_seeds
            or self.seeds != declared.get("concrete_seeds")
        ):
            raise ValueError("grammar-v3 scalable seed count or ordering mismatch")
        for seed in self.seeds:
            _validate_seed(seed)
        if len({seed["seed_id"] for seed in self.seeds}) != len(self.seeds):
            raise ValueError("grammar-v3 scalable seed ids are not unique")
        self.callback_descriptor = callback_descriptor
        if callback_descriptor is None:
            self.callback: SeedCallback | None = None
            self.callback_binding_sha256: str | None = None
            self.callback_registry_root_sha256 = _sha(
                {"state": "reviewed_candidate_compiler_formal_callback_missing"}
            )
        else:
            self.callback_binding_sha256 = callback_binding(callback_descriptor)
            self.callback = _resolve_callback(callback_descriptor)
            self.callback_registry_root_sha256 = _sha(
                {
                    "callback_id": callback_descriptor["callback_id"],
                    "binding_sha256": self.callback_binding_sha256,
                }
            )
        self.seed_registry_root_sha256 = _sha(
            [
                {
                    "seed_id": seed["seed_id"],
                    "seed_lineage_sha256": seed["seed_lineage_sha256"],
                }
                for seed in self.seeds
            ]
        )
        adapter_config = {
            "schema_version": ADAPTER_SCHEMA,
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_content_sha256": self.manifest_content_sha256,
            "callback_registry_root_sha256": self.callback_registry_root_sha256,
            "seed_registry_root_sha256": self.seed_registry_root_sha256,
            "maximum_seeds": maximum_seeds,
            "data_eligibility": ELIGIBILITY,
            "external_paid_llm_calls": False,
        }
        self.adapter_config_sha256 = _sha(adapter_config)
        self._initialize_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _initialize_state(self) -> None:
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SCHEMA)
            row = connection.execute(
                "SELECT * FROM grammar_v3_seed_adapter WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_seed_adapter VALUES (1,?,?,?,?,?,?)",
                    (
                        ADAPTER_SCHEMA,
                        self.manifest_file_sha256,
                        self.manifest_content_sha256,
                        self.callback_registry_root_sha256,
                        self.seed_registry_root_sha256,
                        self.adapter_config_sha256,
                    ),
                )
            else:
                expected = {
                    "singleton": 1,
                    "schema_version": ADAPTER_SCHEMA,
                    "manifest_file_sha256": self.manifest_file_sha256,
                    "manifest_content_sha256": self.manifest_content_sha256,
                    "callback_registry_root_sha256": self.callback_registry_root_sha256,
                    "seed_registry_root_sha256": self.seed_registry_root_sha256,
                    "adapter_config_sha256": self.adapter_config_sha256,
                }
                if dict(row) != expected:
                    raise ValueError("refusing to resume a changed grammar-v3 seed adapter")
            rows = connection.execute("SELECT payload_json FROM work ORDER BY work_id").fetchall()
            for row in rows:
                payload = json.loads(row[0])
                if payload.get("schema_version") != PAYLOAD_SCHEMA:
                    raise ValueError("grammar-v3 adapter requires a dedicated coordinator database")

    @staticmethod
    def _ordinal(seed: dict[str, Any]) -> int:
        return int(str(seed["seed_lineage_sha256"])[:15], 16)

    def _work_items(self) -> list[dict[str, Any]]:
        items = []
        ordinals: set[int] = set()
        for seed in self.seeds:
            ordinal = self._ordinal(seed)
            if ordinal in ordinals:
                raise ValueError("grammar-v3 deterministic seed ordinal collision")
            ordinals.add(ordinal)
            input_lineage = _sha(
                {
                    "manifest_content_sha256": self.manifest_content_sha256,
                    "seed_lineage_sha256": seed["seed_lineage_sha256"],
                    "callback_registry_root_sha256": self.callback_registry_root_sha256,
                }
            )
            adapter_work_id = "G3W-" + _sha(
                {
                    "manifest_content_sha256": self.manifest_content_sha256,
                    "seed_id": seed["seed_id"],
                    "seed_lineage_sha256": seed["seed_lineage_sha256"],
                }
            )[:24]
            items.append(
                {
                    "schema_version": PAYLOAD_SCHEMA,
                    "ordinal": ordinal,
                    "adapter_work_id": adapter_work_id,
                    "manifest_file_sha256": self.manifest_file_sha256,
                    "manifest_content_sha256": self.manifest_content_sha256,
                    "seed_id": seed["seed_id"],
                    "seed_lineage_sha256": seed["seed_lineage_sha256"],
                    "seed_spec": seed,
                    "input_lineage_sha256": input_lineage,
                    "callback_binding_sha256": self.callback_binding_sha256,
                    "data_eligibility": ELIGIBILITY,
                }
            )
        return items

    def _coordinator_identity(self, payload: dict[str, Any]) -> tuple[str, int]:
        seed_payload = {
            "master_seed": int(self.coordinator.config["determinism"]["master_seed"]),
            "ordinal": int(payload["ordinal"]),
            "lane": "cpu",
            "payload": payload,
        }
        encoded = _canonical(seed_payload).encode()
        digest = hashlib.sha256(encoded).digest()
        coordinator_seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        work_id = f"PSW-{hashlib.sha256(encoded).hexdigest()[:24]}"
        return work_id, coordinator_seed

    def enqueue(self) -> dict[str, Any]:
        admitted = self.coordinator.enqueue(self._work_items(), lane="cpu", max_attempts=3)
        checkpoint = self.coordinator.checkpoint()
        return {
            **admitted,
            "requested": len(self.seeds),
            "seed_registry_root_sha256": self.seed_registry_root_sha256,
            "checkpoint_sha256": checkpoint["content_sha256"],
        }

    def _validate_lease(self, lease: WorkLease) -> dict[str, Any]:
        payload = lease.payload
        expected = {item["seed_id"]: item for item in self._work_items()}
        if payload.get("schema_version") != PAYLOAD_SCHEMA:
            raise ValueError("leased work is not grammar-v3 seed work")
        seed_id = str(payload.get("seed_id"))
        if seed_id not in expected or payload != expected[seed_id]:
            raise ValueError("leased grammar-v3 seed payload or lineage mismatch")
        if lease.ordinal != int(payload["ordinal"]):
            raise ValueError("leased grammar-v3 seed ordinal mismatch")
        expected_work_id, expected_seed = self._coordinator_identity(payload)
        if lease.work_id != expected_work_id or lease.seed != expected_seed:
            raise ValueError("leased grammar-v3 coordinator identity mismatch")
        return payload

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        payload = self._validate_lease(lease)
        seed = payload["seed_spec"]
        if self.callback is None:
            reviewed = {
                "schema_version": CALLBACK_RESULT_SCHEMA,
                "decision": "blocked",
                "candidate_compilation": None,
                "formal_result": None,
                "blocker": "reviewed_candidate_compiler_formal_callback_missing",
                "data_eligibility": ELIGIBILITY,
            }
        else:
            reviewed = self.callback(
                seed,
                {
                    "schema_version": "sigma-grammar-v3-seed-callback-context-1.0",
                    "adapter_work_id": payload["adapter_work_id"],
                    "input_lineage_sha256": payload["input_lineage_sha256"],
                    "manifest_content_sha256": self.manifest_content_sha256,
                    "callback_binding_sha256": self.callback_binding_sha256,
                    "data_eligibility": ELIGIBILITY,
                    "external_paid_llm_calls": False,
                },
            )
        required = {
            "schema_version",
            "decision",
            "candidate_compilation",
            "formal_result",
            "blocker",
            "data_eligibility",
        }
        if set(reviewed) != required or reviewed.get("schema_version") != CALLBACK_RESULT_SCHEMA:
            raise ValueError("reviewed grammar-v3 callback result fields or schema are invalid")
        if reviewed.get("decision") not in {"pass", "reject", "blocked"}:
            raise ValueError("reviewed grammar-v3 callback decision is invalid")
        if reviewed.get("decision") == "blocked" and not reviewed.get("blocker"):
            raise ValueError("blocked grammar-v3 callback result requires a blocker")
        if reviewed.get("data_eligibility") != ELIGIBILITY:
            raise ValueError("reviewed grammar-v3 callback result is not fail-closed")
        reviewed_sha = _sha(reviewed)
        body = {
            "schema_version": RESULT_SCHEMA,
            "work_id": lease.work_id,
            "adapter_work_id": payload["adapter_work_id"],
            "seed_id": payload["seed_id"],
            "seed_lineage_sha256": payload["seed_lineage_sha256"],
            "input_lineage_sha256": payload["input_lineage_sha256"],
            "callback_registry_root_sha256": self.callback_registry_root_sha256,
            "reviewed_result": reviewed,
            "reviewed_result_sha256": reviewed_sha,
            "output_lineage_sha256": _sha(
                {
                    "input_lineage_sha256": payload["input_lineage_sha256"],
                    "reviewed_result_sha256": reviewed_sha,
                }
            ),
            "decision": reviewed["decision"],
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}

    def run_bounded(self, *, maximum_tasks: int = 6, worker_id: str = "grammar-v3-seed") -> dict[str, Any]:
        if not 1 <= maximum_tasks <= 6:
            raise ValueError("grammar-v3 bounded run must execute between one and six tasks")
        recovered = self.coordinator.recover_expired()
        executed = 0
        for _ in range(maximum_tasks):
            lease = self.coordinator.claim("cpu", worker_id)
            if lease is None:
                break
            try:
                result = self.execute_lease(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("grammar-v3 seed work lease was lost before completion")
            except Exception as error:
                self.coordinator.fail(lease, worker_id, f"{type(error).__name__}: {error}")
                raise
            executed += 1
        checkpoint = self.coordinator.checkpoint()
        return {
            "executed": executed,
            "recovered": recovered,
            "checkpoint_sha256": checkpoint["content_sha256"],
            "status": self.status(),
        }

    def status(self) -> dict[str, Any]:
        expected_ids = {item["adapter_work_id"] for item in self._work_items()}
        counts: dict[str, int] = {}
        decisions: dict[str, int] = {}
        records = []
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT work_id,ordinal,seed,payload_json,state,attempt,result_json,error_text "
                "FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            payload = json.loads(row["payload_json"])
            if payload.get("adapter_work_id") not in expected_ids:
                raise ValueError("coordinator contains unregistered grammar-v3 seed work")
            state = str(row["state"])
            expected_work_id, expected_seed = self._coordinator_identity(payload)
            if row["work_id"] != expected_work_id or int(row["seed"]) != expected_seed:
                raise ValueError("stored grammar-v3 coordinator identity mismatch")
            counts[state] = counts.get(state, 0) + 1
            result_sha = None
            output_lineage = None
            if row["result_json"] is not None:
                result = json.loads(row["result_json"])
                body = {key: value for key, value in result.items() if key != "content_sha256"}
                if result.get("content_sha256") != _sha(body):
                    raise ValueError("stored grammar-v3 seed result hash mismatch")
                if result.get("input_lineage_sha256") != payload["input_lineage_sha256"]:
                    raise ValueError("stored grammar-v3 seed result lineage mismatch")
                reviewed_sha = _sha(result.get("reviewed_result"))
                expected_output = _sha(
                    {
                        "input_lineage_sha256": payload["input_lineage_sha256"],
                        "reviewed_result_sha256": reviewed_sha,
                    }
                )
                if (
                    result.get("reviewed_result_sha256") != reviewed_sha
                    or result.get("output_lineage_sha256") != expected_output
                    or result.get("callback_registry_root_sha256")
                    != self.callback_registry_root_sha256
                ):
                    raise ValueError("stored grammar-v3 callback result binding mismatch")
                decision = str(result["decision"])
                decisions[decision] = decisions.get(decision, 0) + 1
                result_sha = result["content_sha256"]
                output_lineage = result["output_lineage_sha256"]
            records.append(
                {
                    "work_id": row["work_id"],
                    "adapter_work_id": payload["adapter_work_id"],
                    "seed_id": payload["seed_id"],
                    "seed_lineage_sha256": payload["seed_lineage_sha256"],
                    "ordinal": int(row["ordinal"]),
                    "coordinator_seed": int(row["seed"]),
                    "state": state,
                    "attempt": int(row["attempt"]),
                    "result_sha256": result_sha,
                    "output_lineage_sha256": output_lineage,
                    "error_text": row["error_text"],
                }
            )
        telemetry = self.coordinator.telemetry()
        body = {
            "schema_version": STATUS_SCHEMA,
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_content_sha256": self.manifest_content_sha256,
            "seed_registry_root_sha256": self.seed_registry_root_sha256,
            "callback_registry_root_sha256": self.callback_registry_root_sha256,
            "seed_count": len(self.seeds),
            "work_state_counts": dict(sorted(counts.items())),
            "decision_counts": dict(sorted(decisions.items())),
            "work_records": records,
            "work_records_root_sha256": _sha(records),
            "checkpoint_sequence": telemetry["checkpoint_sequence"],
            "recovered_leases": telemetry["recovered_leases"],
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "next_scaling_hook": (
                "replace the six-item iterator with bounded parameter-cell chunks while "
                "retaining the same manifest/seed/callback lineage envelope"
            ),
        }
        return {**body, "content_sha256": _sha(body)}
