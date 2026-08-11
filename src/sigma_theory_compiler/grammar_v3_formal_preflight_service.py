from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_compilation_campaign import _compile_action_ir
from .grammar_v3_parameter_cell_compilation_campaign import _action_density_key
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-formal-preflight-service-config-1.0"
PAYLOAD_SCHEMA = "sigma-grammar-v3-formal-preflight-work-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-formal-preflight-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-formal-preflight-status-1.0"

STATE_SQL = """
CREATE TABLE IF NOT EXISTS grammar_v3_formal_preflight_service (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  immutable_config_sha256 TEXT NOT NULL,
  candidate_registry_root_sha256 TEXT NOT NULL,
  callback_registry_root_sha256 TEXT NOT NULL,
  compilation_campaign_content_sha256 TEXT NOT NULL
);
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "execution_enabled",
        "compilation_campaign",
        "parameter_cell_manifest",
        "source_seed_manifest",
        "coordinator_config",
        "resource_profile",
        "reviewed_adapters",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 formal-preflight config is invalid")
    if not isinstance(config.get("execution_enabled"), bool):
        raise TypeError("formal-preflight execution_enabled must be boolean")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("formal-preflight eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("formal-preflight enabled paid LLM calls")
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_tasks",
        "chunk_size",
        "maximum_attempts",
        "maximum_wall_seconds",
        "maximum_disk_bytes",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget["maximum_tasks"]) != 163
        or int(budget["chunk_size"]) != 32
        or not 1 <= int(budget["maximum_attempts"]) <= 3
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 1024 * 1024 <= int(budget["maximum_disk_bytes"]) <= 128 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("formal-preflight budget is invalid or unbounded")
    families = {adapter["family_id"] for adapter in config["reviewed_adapters"]}
    if families != {
        "AETHER_K1234_PARAMETER_CELL",
        "KESSENCE_G2_CONVEX",
        "CUBIC_HORNDESKI_G3_WEAK_CELL",
        "CONFORMAL_G4_PHI_SCALAR_TENSOR",
    }:
        raise ValueError("formal-preflight adapter family registry is incomplete")


class GrammarV3FormalPreflightService:
    """Durable cheap-prerequisite queue for the 163 unique grammar-v3 actions."""

    def __init__(
        self,
        directory: str | Path,
        config: dict[str, Any],
        repo_root: str | Path,
        *,
        unavailable_families: frozenset[str] = frozenset(),
    ) -> None:
        _validate_config(config)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.compilation = self._bound_json("compilation_campaign", content=True)
        self.cell_manifest = self._bound_json("parameter_cell_manifest", content=True)
        self.source_manifest = self._bound_json("source_seed_manifest", content=True)
        self.base_coordinator = self._bound_json("coordinator_config")
        self.resource_profile = self._bound_json("resource_profile")
        self.unavailable_families = unavailable_families
        unknown = unavailable_families - {
            adapter["family_id"] for adapter in config["reviewed_adapters"]
        }
        if unknown:
            raise ValueError("unknown unavailable formal-preflight family")
        self.adapters = self._resolve_adapters()
        self.callback_registry_root_sha256 = _sha(
            [
                {
                    "family_id": descriptor["family_id"],
                    "adapter_id": descriptor["adapter_id"],
                    "callback": descriptor["callback"],
                    "source_file_sha256": descriptor["source_file_sha256"],
                    "state": (
                        "missing"
                        if descriptor["family_id"] in unavailable_families
                        else "reviewed_bound"
                    ),
                }
                for descriptor in config["reviewed_adapters"]
            ]
        )
        self.work_items = self._work_items()
        self.candidate_registry_root_sha256 = _sha(
            [
                [
                    item["candidate_id"],
                    item["action_density_equivalence_sha256"],
                    item["typed_action_ir_sha256"],
                ]
                for item in self.work_items
            ]
        )
        if self.candidate_registry_root_sha256 != self.compilation[
            "unique_candidate_registry_root_sha256"
        ]:
            raise ValueError("formal-preflight candidate registry differs from compilation")
        self.coordinator = PersistentParallelSearch(
            self.directory / "formal-preflight.sqlite",
            self._coordinator_config(),
            self.resource_profile,
        )
        self._initialize_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _binding_path(self, binding: dict[str, Any], label: str) -> Path:
        path = (self.repo_root / binding["path"]).resolve()
        try:
            path.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError(f"formal-preflight {label} path escapes repository") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"formal-preflight {label} file hash mismatch")
        return path

    def _bound_json(self, key: str, *, content: bool = False) -> dict[str, Any]:
        binding = self.config[key]
        path = self._binding_path(binding, key)
        value = _load(path)
        if content:
            body = {name: item for name, item in value.items() if name != "content_sha256"}
            if value.get("content_sha256") != binding["content_sha256"] or _sha(body) != binding[
                "content_sha256"
            ]:
                raise ValueError(f"formal-preflight {key} content hash mismatch")
        return value

    def _resolve_adapters(self) -> dict[str, Any]:
        resolved = {}
        for descriptor in self.config["reviewed_adapters"]:
            required = {
                "family_id",
                "adapter_id",
                "callback",
                "source_path",
                "source_file_sha256",
            }
            if set(descriptor) != required:
                raise ValueError("formal-preflight adapter descriptor fields are invalid")
            source = self._binding_path(
                {"path": descriptor["source_path"], "file_sha256": descriptor["source_file_sha256"]},
                "adapter source",
            )
            module_name, separator, name = descriptor["callback"].partition(":")
            if not separator:
                raise ValueError("formal-preflight adapter callback must use module:function")
            callback = getattr(importlib.import_module(module_name), name, None)
            callback_source = (
                inspect.getsourcefile(inspect.unwrap(callback)) if callable(callback) else None
            )
            if not callable(callback) or callback_source is None or Path(callback_source).resolve() != source:
                raise ValueError("formal-preflight adapter is not defined by its bound source")
            resolved[descriptor["family_id"]] = callback
        return resolved

    def _work_items(self) -> list[dict[str, Any]]:
        cells = list(iter_parameter_cells(self.cell_manifest, self.source_manifest))
        families = {
            family["family_id"]: family
            for family in self.source_manifest["typed_family_seeds"]
            if family["enabled_for_generation"]
        }
        manifest_binding = {
            "parameter_cell_manifest_content_sha256": self.cell_manifest["content_sha256"],
            "parameter_cell_registry_root_sha256": self.cell_manifest[
                "parameter_cell_registry_root_sha256"
            ],
        }
        seen: set[str] = set()
        items = []
        for cell in cells:
            equivalence_sha = _sha(_action_density_key(cell))
            if equivalence_sha in seen:
                continue
            seen.add(equivalence_sha)
            candidate_id = "G3A-" + equivalence_sha[:24]
            pseudo_seed = {
                "seed_id": cell["parameter_cell_id"],
                "seed_lineage_sha256": cell["parameter_cell_lineage_sha256"],
                "family_id": cell["family_id"],
                "family_lineage_sha256": cell["family_lineage_sha256"],
                "theory_contract": cell["theory_contract"],
                "operator_atoms": cell["operator_atoms"],
                "parameters": cell["parameters"],
            }
            action_ir = _compile_action_ir(
                pseudo_seed, families[cell["family_id"]], manifest_binding
            )
            body = {
                "schema_version": PAYLOAD_SCHEMA,
                "ordinal": len(items),
                "candidate_id": candidate_id,
                "typed_action_ir_sha256": action_ir["content_sha256"],
                "action_density_equivalence_sha256": equivalence_sha,
                "representative_cell_id": cell["parameter_cell_id"],
                "representative_cell_lineage_sha256": cell[
                    "parameter_cell_lineage_sha256"
                ],
                "family_id": cell["family_id"],
                "compilation_campaign_content_sha256": self.compilation["content_sha256"],
                "callback_registry_root_sha256": self.callback_registry_root_sha256,
                "data_eligibility": dict(ELIGIBILITY),
            }
            items.append({**body, "input_lineage_sha256": _sha(body)})
        if len(items) != int(self.config["budget"]["maximum_tasks"]):
            raise ValueError("formal-preflight unique candidate count changed")
        return items

    def _coordinator_config(self) -> dict[str, Any]:
        config = json.loads(_canonical(self.base_coordinator))
        maximum = int(self.config["budget"]["maximum_tasks"])
        config["queue"].update(
            maximum_pending_work=maximum,
            maximum_attempts=int(self.config["budget"]["maximum_attempts"]),
            lease_seconds=int(self.config["budget"]["maximum_wall_seconds"]),
            checkpoint_every_completions=int(self.config["budget"]["chunk_size"]),
        )
        config["budget"] = {
            "maximum_tasks": maximum,
            "maximum_wall_seconds": float(self.config["budget"]["maximum_wall_seconds"]),
        }
        config["cpu"]["maximum_workers"] = min(4, maximum)
        config["external_paid_llm_calls"] = False
        return config

    def _initialize_state(self) -> None:
        expected = {
            "singleton": 1,
            "schema_version": STATUS_SCHEMA,
            "immutable_config_sha256": _sha(self.config),
            "candidate_registry_root_sha256": self.candidate_registry_root_sha256,
            "callback_registry_root_sha256": self.callback_registry_root_sha256,
            "compilation_campaign_content_sha256": self.compilation["content_sha256"],
        }
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SQL)
            row = connection.execute(
                "SELECT * FROM grammar_v3_formal_preflight_service WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_formal_preflight_service VALUES (1,?,?,?,?,?)",
                    tuple(expected[key] for key in expected if key != "singleton"),
                )
            elif dict(row) != expected:
                raise ValueError("refusing to resume changed grammar-v3 formal-preflight service")
            for row in connection.execute("SELECT payload_json FROM work"):
                if json.loads(row[0]).get("schema_version") != PAYLOAD_SCHEMA:
                    raise ValueError("formal-preflight requires a dedicated coordinator database")

    def _disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def _enforce_budget(self, started: float | None = None) -> None:
        if self._disk_bytes() > int(self.config["budget"]["maximum_disk_bytes"]):
            raise RuntimeError("formal-preflight disk budget exhausted")
        if started is not None and time.monotonic() - started > float(
            self.config["budget"]["maximum_wall_seconds"]
        ):
            raise TimeoutError("formal-preflight wall budget exhausted")

    def enqueue(self) -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 formal-preflight is disabled by config")
        self._enforce_budget()
        admitted = self.coordinator.enqueue(
            self.work_items,
            lane="cpu",
            max_attempts=int(self.config["budget"]["maximum_attempts"]),
        )
        checkpoint = self.coordinator.checkpoint()
        return {
            **admitted,
            "requested": len(self.work_items),
            "candidate_registry_root_sha256": self.candidate_registry_root_sha256,
            "checkpoint_sha256": checkpoint["content_sha256"],
        }

    @staticmethod
    @lru_cache(maxsize=4)
    def _invoke_adapter(family_id: str, callback: Any) -> dict[str, Any]:
        if family_id == "AETHER_K1234_PARAMETER_CELL":
            evidence = callback()
            passed = evidence.get("passed") is True and evidence.get("noether_residual") == "0"
            stable = {
                "passed": evidence.get("passed"),
                "noether_residual": evidence.get("noether_residual"),
                "declared_point_rank": evidence.get("declared_point_rank"),
                "scope": evidence.get("scope"),
            }
        elif family_id in {"KESSENCE_G2_CONVEX", "CUBIC_HORNDESKI_G3_WEAK_CELL"}:
            passed, evidence = callback()
            stable = {
                "passed": bool(passed),
                "status": evidence.get("status"),
                "scope": evidence.get("scope"),
                "evidence_sha256": _sha(evidence),
            }
        elif family_id == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
            evidence = callback(
                {
                    "schema_version": "sigma-scalar-tensor-pack-1.0",
                    "name": "grammar-v3 formal-preflight fixed-xi G4",
                    "normalization": {
                        "u": "phi/Lambda_phi",
                        "x": "-nabla_phi_squared/(2*Lambda_phi**4)",
                        "Lambda_phi_positive": True,
                    },
                    "coefficients": ["xi"],
                    "functions": {"g2": "x", "g3": "0", "g4": "1/2+xi*u**2"},
                    "derivative_overrides": {"g4_x": "0"},
                    "mutation_axes": [{"coefficient": "xi", "values": ["1/100"]}],
                }
            )
            passed = evidence.get("status") == "pass"
            stable = {
                "passed": passed,
                "status": evidence.get("status"),
                "content_sha256": evidence.get("content_sha256"),
                "typed_normalized_covariant_family": evidence.get("capability_status", {}).get(
                    "typed_normalized_covariant_family"
                ),
            }
        else:
            raise ValueError("unreviewed formal-preflight family")
        body = {
            "family_id": family_id,
            "decision": "pass" if passed else "blocked",
            "evidence_summary": stable,
            "scope": "cheap family prerequisite only; not full ADM or global energy",
        }
        return {**body, "content_sha256": _sha(body)}

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        if lease.ordinal >= len(self.work_items) or lease.payload != self.work_items[lease.ordinal]:
            raise ValueError("formal-preflight leased candidate binding changed")
        payload = lease.payload
        family = payload["family_id"]
        if family in self.unavailable_families:
            adapter = {
                "family_id": family,
                "decision": "blocked",
                "evidence_summary": None,
                "scope": "reviewed family prerequisite adapter missing",
            }
            adapter = {**adapter, "content_sha256": _sha(adapter)}
            blocker = "reviewed_family_preflight_adapter_missing"
        else:
            adapter = self._invoke_adapter(family, self.adapters[family])
            blocker = None if adapter["decision"] == "pass" else "family_prerequisite_not_passed"
        body = {
            "schema_version": RESULT_SCHEMA,
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "input_lineage_sha256": payload["input_lineage_sha256"],
            "callback_registry_root_sha256": self.callback_registry_root_sha256,
            "receipt_binding_gate": "pass",
            "family_prerequisite_gate": adapter["decision"],
            "adapter_evidence": adapter,
            "decision": adapter["decision"],
            "blocker": blocker,
            "expensive_adm_or_global_energy_run": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}

    def run_bounded(self, *, worker_id: str = "grammar-v3-formal-preflight") -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 formal-preflight is disabled by config")
        started = time.monotonic()
        current = self.coordinator.recover_expired()
        recovered = {
            key: int(self.recovered_on_start[key]) + int(current[key])
            for key in ("recovered", "failed")
        }
        self.recovered_on_start = {"recovered": 0, "failed": 0}
        executed = 0
        for _ in range(int(self.config["budget"]["maximum_tasks"])):
            self._enforce_budget(started)
            lease = self.coordinator.claim("cpu", worker_id)
            if lease is None:
                break
            try:
                result = self.execute_lease(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("formal-preflight lease was lost")
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
        work_counts: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        family_decisions: defaultdict[str, Counter[str]] = defaultdict(Counter)
        gate_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        records = []
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT ordinal,payload_json,state,attempt,result_json,error_text FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            ordinal = int(row["ordinal"])
            payload = json.loads(row["payload_json"])
            if ordinal >= len(self.work_items) or payload != self.work_items[ordinal]:
                raise ValueError("stored formal-preflight work payload was tampered")
            work_counts[str(row["state"])] += 1
            result_sha = None
            blocker = None
            if row["result_json"]:
                result = json.loads(row["result_json"])
                body = {key: value for key, value in result.items() if key != "content_sha256"}
                if (
                    result.get("content_sha256") != _sha(body)
                    or result.get("candidate_id") != payload["candidate_id"]
                    or result.get("typed_action_ir_sha256") != payload["typed_action_ir_sha256"]
                    or result.get("input_lineage_sha256") != payload["input_lineage_sha256"]
                ):
                    raise ValueError("stored formal-preflight result binding changed")
                decision = result["decision"]
                decisions[decision] += 1
                family_decisions[payload["family_id"]][decision] += 1
                gate_counts["receipt_binding"][result["receipt_binding_gate"]] += 1
                gate_counts["family_prerequisite"][result["family_prerequisite_gate"]] += 1
                result_sha = result["content_sha256"]
                blocker = result["blocker"]
            records.append(
                {
                    "candidate_id": payload["candidate_id"],
                    "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
                    "family_id": payload["family_id"],
                    "state": row["state"],
                    "attempt": int(row["attempt"]),
                    "result_sha256": result_sha,
                    "blocker": blocker,
                    "error_text": row["error_text"],
                }
            )
        chunk_size = int(self.config["budget"]["chunk_size"])
        chunks = []
        for start in range(0, len(records), chunk_size):
            selected = records[start : start + chunk_size]
            chunk_body = {
                "chunk_index": len(chunks),
                "range": {"start": start, "stop": start + len(selected)},
                "record_root_sha256": _sha(selected),
                "state_counts": dict(sorted(Counter(item["state"] for item in selected).items())),
            }
            chunks.append({**chunk_body, "content_sha256": _sha(chunk_body)})
        body = {
            "schema_version": STATUS_SCHEMA,
            "execution_enabled": self.config["execution_enabled"],
            "compilation_campaign_content_sha256": self.compilation["content_sha256"],
            "candidate_count": len(self.work_items),
            "candidate_registry_root_sha256": self.candidate_registry_root_sha256,
            "callback_registry_root_sha256": self.callback_registry_root_sha256,
            "work_state_counts": dict(sorted(work_counts.items())),
            "decision_counts": dict(sorted(decisions.items())),
            "family_decision_counts": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(family_decisions.items())
            },
            "gate_counts": {
                gate: dict(sorted(counts.items())) for gate, counts in sorted(gate_counts.items())
            },
            "record_registry_root_sha256": _sha(records),
            "chunks": chunks,
            "chunk_registry_root_sha256": _sha(chunks),
            "checkpoint_sequence": self.coordinator.telemetry()["checkpoint_sequence"],
            "disk_bytes": self._disk_bytes(),
            "budget": self.config["budget"],
            "expensive_adm_or_global_energy_run": False,
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
            "next_promotion_hook": (
                "enqueue only preflight-pass candidates into separately reviewed family-specific "
                "ADM/formal campaigns bound to candidate_id+typed_action_ir_sha256"
            ),
        }
        return {**body, "content_sha256": _sha(body)}


def portable_status(status: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in status.items()
        if key not in {"content_sha256", "checkpoint_sequence", "disk_bytes"}
    }
    return {**body, "content_sha256": _sha(body)}
