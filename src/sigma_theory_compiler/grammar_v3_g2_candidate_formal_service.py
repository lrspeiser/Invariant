from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .g2_global_positive_mass_prerequisite_audit import (
    build_g2_global_positive_mass_prerequisite_audit,
)
from .g2_seed_coupled_formal_prerequisite_campaign import (
    build_g2_seed_coupled_formal_prerequisite_campaign,
)
from .grammar_v3_parameter_cell_compilation_campaign import _action_density_key
from .grammar_v3_parameter_cell_manifest_campaign import iter_parameter_cells
from .grammar_v3_promotion_admission_service import GrammarV3PromotionAdmissionService
from .persistent_parallel_search import PersistentParallelSearch, WorkLease
from .promotion_orchestrator import ELIGIBILITY

CONFIG_SCHEMA = "sigma-grammar-v3-g2-candidate-formal-service-config-1.0"
PAYLOAD_SCHEMA = "sigma-grammar-v3-g2-candidate-formal-work-1.0"
RESULT_SCHEMA = "sigma-grammar-v3-g2-candidate-formal-result-1.0"
STATUS_SCHEMA = "sigma-grammar-v3-g2-candidate-formal-status-1.0"

STATE_SQL = """
CREATE TABLE IF NOT EXISTS grammar_v3_g2_candidate_formal_service (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version TEXT NOT NULL,
  immutable_config_sha256 TEXT NOT NULL,
  candidate_registry_root_sha256 TEXT NOT NULL,
  reviewed_adapter_registry_root_sha256 TEXT NOT NULL,
  promotion_status_content_sha256 TEXT NOT NULL
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


def _validate(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "execution_enabled",
        "promotion_status",
        "promotion_config",
        "local_formal_campaign",
        "local_formal_config",
        "positive_mass_audit",
        "positive_mass_config",
        "coordinator_config",
        "resource_profile",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("grammar-v3 G2 candidate formal config is invalid")
    if not isinstance(config.get("execution_enabled"), bool):
        raise TypeError("G2 candidate formal execution_enabled must be boolean")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("G2 candidate formal eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("G2 candidate formal enabled paid LLM calls")
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_tasks",
        "maximum_attempts",
        "maximum_wall_seconds",
        "maximum_disk_bytes",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget["maximum_tasks"]) != 2
        or not 1 <= int(budget["maximum_attempts"]) <= 3
        or not 1 <= float(budget["maximum_wall_seconds"]) <= 300
        or not 1024 * 1024 <= int(budget["maximum_disk_bytes"]) <= 128 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("G2 candidate formal budget is invalid or unbounded")


class GrammarV3G2CandidateFormalService:
    """Reviewed exact G2 prerequisites with global claims kept fail-closed."""

    def __init__(
        self,
        directory: str | Path,
        config: dict[str, Any],
        repo_root: str | Path,
        *,
        missing_adapter_ids: frozenset[str] = frozenset(),
    ) -> None:
        _validate(config)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.promotion_status = self._bound_json("promotion_status", content=True)
        self.promotion_config = self._bound_json("promotion_config")
        self.local_committed = self._bound_json("local_formal_campaign", content=True)
        self.local_config = self._bound_json("local_formal_config")
        self.energy_committed = self._bound_json("positive_mass_audit", content=True)
        self.energy_config = self._bound_json("positive_mass_config")
        self.base_coordinator = self._bound_json("coordinator_config")
        self.resource_profile = self._bound_json("resource_profile")
        self._validate_upstream_status()
        self.missing_adapter_ids = missing_adapter_ids
        allowed_missing = {
            *self.local_committed["adapter_results"],
            "restricted_maximal_slice_positive_mass",
        }
        if missing_adapter_ids - allowed_missing:
            raise ValueError("unknown missing G2 reviewed adapter")
        self.local_rebuilt = build_g2_seed_coupled_formal_prerequisite_campaign(
            self.local_config, self.repo_root
        )
        if self.local_rebuilt != self.local_committed:
            raise ValueError("G2 local formal campaign does not exactly rebuild")
        self.energy_rebuilt = build_g2_global_positive_mass_prerequisite_audit(
            self.energy_config, self.repo_root
        )
        if self.energy_rebuilt != self.energy_committed:
            raise ValueError("G2 positive-mass audit does not exactly rebuild")
        self.reviewed_adapter_registry_root_sha256 = _sha(
            {
                "local_adapter_results": self.local_committed["adapter_results"],
                "positive_mass_core_replay": self.energy_committed[
                    "positive_mass_core_replay"
                ],
                "missing_adapter_ids": sorted(missing_adapter_ids),
            }
        )
        self.promotion = GrammarV3PromotionAdmissionService(
            self.directory / "promotion-attestation",
            self.promotion_config,
            self.repo_root,
        )
        if self.promotion_status["eligible_candidate_registry_root_sha256"] != (
            self.promotion.eligible_candidate_registry_root_sha256
        ):
            raise ValueError("G2 service promotion candidate registry changed")
        self.work_items = self._work_items()
        self.candidate_registry_root_sha256 = _sha(
            [
                [
                    item["candidate_id"],
                    item["typed_action_ir_sha256"],
                    item["preflight_result_sha256"],
                    item["admission_result_sha256"],
                    item["reviewed_local_record_binding_sha256"],
                    item["reviewed_energy_record_binding_sha256"],
                ]
                for item in self.work_items
            ]
        )
        self.coordinator = PersistentParallelSearch(
            self.directory / "g2-candidate-formal.sqlite",
            self._coordinator_config(),
            self.resource_profile,
        )
        self._initialize_state()
        self.recovered_on_start = self.coordinator.recover_expired()

    def _path(self, binding: dict[str, Any], label: str) -> Path:
        path = (self.repo_root / binding["path"]).resolve()
        try:
            path.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError(f"G2 candidate formal {label} path escapes repository") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"G2 candidate formal {label} file hash mismatch")
        return path

    def _bound_json(self, key: str, *, content: bool = False) -> dict[str, Any]:
        binding = self.config[key]
        value = _load(self._path(binding, key))
        if content:
            body = {name: item for name, item in value.items() if name != "content_sha256"}
            if value.get("content_sha256") != binding["content_sha256"] or _sha(body) != binding[
                "content_sha256"
            ]:
                raise ValueError(f"G2 candidate formal {key} content hash mismatch")
        return value

    def _validate_upstream_status(self) -> None:
        status = self.promotion_status
        if (
            status.get("decision_counts") != {"pass": 162}
            or status.get("target_queue_counts", {}).get(
                "grammar_v3_g2_candidate_adm_formal"
            )
            != 2
            or status.get("target_queue_registry_roots", {}).get(
                "grammar_v3_g2_candidate_adm_formal"
            )
            is None
            or status.get("downstream_expensive_execution_started") is not False
            or status.get("data_eligibility") != {**ELIGIBILITY, "passed": True}
        ):
            raise ValueError("G2 promotion-admission status is ineligible")

    def _candidate_g2_expressions(self) -> dict[str, str]:
        cells = list(
            iter_parameter_cells(
                self.promotion.preflight.cell_manifest,
                self.promotion.preflight.source_manifest,
            )
        )
        expressions = {}
        for cell in cells:
            if cell["family_id"] != "KESSENCE_G2_CONVEX":
                continue
            equivalence = _sha(_action_density_key(cell))
            candidate_id = "G3A-" + equivalence[:24]
            expressions.setdefault(candidate_id, cell["parameters"]["G2"])
        if len(expressions) != 2:
            raise ValueError("G2 canonical action expression count changed")
        return expressions

    def _work_items(self) -> list[dict[str, Any]]:
        promotion_by_id = {
            item["candidate_id"]: item
            for item in self.promotion.work_items
            if item["family_id"] == "KESSENCE_G2_CONVEX"
        }
        expressions = self._candidate_g2_expressions()
        local_by_expression = {
            record["candidate_certificate"]["G2"]: record
            for record in self.local_committed["candidate_records"]
        }
        energy_by_coefficient = {
            record["dec_and_boundary_certificate"]["G2"].split("+(", 1)[1].split(")", 1)[0]: record
            for record in self.energy_committed["candidate_records"]
        }
        items = []
        for candidate_id in sorted(promotion_by_id):
            source = promotion_by_id[candidate_id]
            fake = WorkLease(
                work_id="attestation",
                ordinal=int(source["ordinal"]),
                lane="cpu",
                seed=0,
                attempt=1,
                max_attempts=1,
                payload=source,
            )
            admission = self.promotion.execute_lease(fake)
            if admission["decision"] != "pass":
                raise ValueError("G2 promotion admission does not replay as pass")
            expression = expressions[candidate_id]
            local = local_by_expression.get(expression)
            if local is None:
                raise ValueError("new G2 action lacks exact reviewed polynomial predecessor")
            coefficient = local["candidate_certificate"]["quadratic_coefficient"]
            energy = energy_by_coefficient.get(coefficient)
            if energy is None:
                raise ValueError("new G2 action lacks reviewed positive-mass predecessor")
            if (
                local["decision"] != "blocked"
                or energy["decision"] != "blocked"
                or local["candidate_certificate"]["domain"] != "0<=X_phi<=1"
                or energy["first_missing_premise"]
                != "hash_bound_general_nonmaximal_positive_mass_theorem"
            ):
                raise ValueError("reviewed G2 predecessor decision boundary changed")
            body = {
                "schema_version": PAYLOAD_SCHEMA,
                "ordinal": len(items),
                "candidate_id": candidate_id,
                "typed_action_ir_sha256": source["typed_action_ir_sha256"],
                "preflight_result_sha256": source["preflight_result_sha256"],
                "admission_result_sha256": admission["content_sha256"],
                "promotion_g2_queue_root_sha256": self.promotion_status[
                    "target_queue_registry_roots"
                ]["grammar_v3_g2_candidate_adm_formal"],
                "G2": expression,
                "quadratic_coefficient": coefficient,
                "reviewed_seed_id": local["seed_id"],
                "reviewed_local_record_binding_sha256": local["provenance"][
                    "binding_sha256"
                ],
                "reviewed_energy_record_binding_sha256": energy["provenance"][
                    "binding_sha256"
                ],
                "local_campaign_content_sha256": self.local_committed["content_sha256"],
                "positive_mass_audit_content_sha256": self.energy_committed[
                    "content_sha256"
                ],
                "reviewed_adapter_registry_root_sha256": self.reviewed_adapter_registry_root_sha256,
                "data_eligibility": dict(ELIGIBILITY),
            }
            items.append({**body, "input_lineage_sha256": _sha(body)})
        if len(items) != 2:
            raise ValueError("G2 formal service requires exactly two admitted candidates")
        return items

    def _coordinator_config(self) -> dict[str, Any]:
        config = json.loads(_canonical(self.base_coordinator))
        config["queue"].update(
            maximum_pending_work=2,
            maximum_attempts=int(self.config["budget"]["maximum_attempts"]),
            lease_seconds=int(self.config["budget"]["maximum_wall_seconds"]),
            checkpoint_every_completions=1,
        )
        config["budget"] = {
            "maximum_tasks": 2,
            "maximum_wall_seconds": float(self.config["budget"]["maximum_wall_seconds"]),
        }
        config["cpu"]["maximum_workers"] = 2
        config["external_paid_llm_calls"] = False
        return config

    def _initialize_state(self) -> None:
        expected = {
            "singleton": 1,
            "schema_version": STATUS_SCHEMA,
            "immutable_config_sha256": _sha(self.config),
            "candidate_registry_root_sha256": self.candidate_registry_root_sha256,
            "reviewed_adapter_registry_root_sha256": self.reviewed_adapter_registry_root_sha256,
            "promotion_status_content_sha256": self.promotion_status["content_sha256"],
        }
        with self.coordinator.connect() as connection:
            connection.executescript(STATE_SQL)
            row = connection.execute(
                "SELECT * FROM grammar_v3_g2_candidate_formal_service WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO grammar_v3_g2_candidate_formal_service VALUES (1,?,?,?,?,?)",
                    tuple(expected[key] for key in expected if key != "singleton"),
                )
            elif dict(row) != expected:
                raise ValueError("refusing to resume changed G2 candidate formal service")
            for row in connection.execute("SELECT payload_json FROM work"):
                if json.loads(row[0]).get("schema_version") != PAYLOAD_SCHEMA:
                    raise ValueError("G2 candidate formal service requires a dedicated DB")

    def _disk_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.directory.rglob("*") if path.is_file())

    def enqueue(self) -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 G2 candidate formal service is disabled")
        if self._disk_bytes() > int(self.config["budget"]["maximum_disk_bytes"]):
            raise RuntimeError("G2 candidate formal disk budget exhausted")
        admitted = self.coordinator.enqueue(
            self.work_items,
            lane="cpu",
            max_attempts=int(self.config["budget"]["maximum_attempts"]),
        )
        checkpoint = self.coordinator.checkpoint()
        return {**admitted, "requested": 2, "checkpoint_sha256": checkpoint["content_sha256"]}

    def execute_lease(self, lease: WorkLease) -> dict[str, Any]:
        if lease.ordinal >= 2 or lease.payload != self.work_items[lease.ordinal]:
            raise ValueError("G2 candidate formal leased payload changed")
        payload = lease.payload
        local_missing = sorted(
            self.missing_adapter_ids & set(self.local_committed["adapter_results"])
        )
        positive_missing = "restricted_maximal_slice_positive_mass" in self.missing_adapter_ids
        local_status = "blocked" if local_missing else "pass"
        restricted_status = "blocked" if positive_missing else "pass"
        blocker = (
            "reviewed_local_g2_adapter_missing: " + ",".join(local_missing)
            if local_missing
            else "reviewed_restricted_positive_mass_adapter_missing"
            if positive_missing
            else "hash_bound_general_nonmaximal_positive_mass_theorem"
        )
        gates = {
            "candidate_action_preflight_admission_binding": "pass",
            "exact_polynomial_predecessor_equivalence": "pass",
            "covariant_variation_noether": local_status,
            "coupled_adm_primary_and_legendre": local_status,
            "candidate_local_dirac_pair": local_status,
            "principal_symbol": local_status,
            "common_time_cone": local_status,
            "pointwise_hamiltonian": local_status,
            "dominant_energy_condition": local_status,
            "explicit_asymptotically_flat_contract": restricted_status,
            "scalar_boundary_flux": restricted_status,
            "restricted_maximal_slice_positive_mass": restricted_status,
            "general_nonmaximal_positive_mass": "blocked",
            "global_positive_energy": "blocked",
            "full_formal_completion": "blocked",
        }
        body = {
            "schema_version": RESULT_SCHEMA,
            "candidate_id": payload["candidate_id"],
            "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
            "preflight_result_sha256": payload["preflight_result_sha256"],
            "admission_result_sha256": payload["admission_result_sha256"],
            "input_lineage_sha256": payload["input_lineage_sha256"],
            "quadratic_coefficient": payload["quadratic_coefficient"],
            "reviewed_adapter_registry_root_sha256": self.reviewed_adapter_registry_root_sha256,
            "gate_ledger": gates,
            "decision": "blocked",
            "first_missing_premise": blocker,
            "general_nonmaximal_global_positive_mass_proved": False,
            "full_formal_pass": False,
            "negative_total_energy_counterexample_found": False,
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}

    def run_bounded(self, *, worker_id: str = "grammar-v3-g2-formal") -> dict[str, Any]:
        if self.config["execution_enabled"] is not True:
            raise RuntimeError("grammar-v3 G2 candidate formal service is disabled")
        started = time.monotonic()
        current = self.coordinator.recover_expired()
        recovered = {
            key: int(self.recovered_on_start[key]) + int(current[key])
            for key in ("recovered", "failed")
        }
        self.recovered_on_start = {"recovered": 0, "failed": 0}
        executed = 0
        for _ in range(2):
            if time.monotonic() - started > float(self.config["budget"]["maximum_wall_seconds"]):
                raise TimeoutError("G2 candidate formal wall budget exhausted")
            lease = self.coordinator.claim("cpu", worker_id)
            if lease is None:
                break
            try:
                result = self.execute_lease(lease)
                if not self.coordinator.finish(lease, worker_id, result):
                    raise RuntimeError("G2 candidate formal lease was lost")
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
        gate_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        blocker_counts: Counter[str] = Counter()
        records = []
        with self.coordinator.connect() as connection:
            rows = connection.execute(
                "SELECT ordinal,payload_json,state,attempt,result_json,error_text FROM work ORDER BY ordinal"
            ).fetchall()
        for row in rows:
            ordinal = int(row["ordinal"])
            payload = json.loads(row["payload_json"])
            if ordinal >= 2 or payload != self.work_items[ordinal]:
                raise ValueError("stored G2 candidate formal payload was tampered")
            work_counts[str(row["state"])] += 1
            result_sha = blocker = None
            if row["result_json"]:
                result = json.loads(row["result_json"])
                body = {key: value for key, value in result.items() if key != "content_sha256"}
                if (
                    result.get("content_sha256") != _sha(body)
                    or result.get("candidate_id") != payload["candidate_id"]
                    or result.get("typed_action_ir_sha256") != payload["typed_action_ir_sha256"]
                    or result.get("admission_result_sha256") != payload["admission_result_sha256"]
                ):
                    raise ValueError("stored G2 candidate formal result binding changed")
                decisions[result["decision"]] += 1
                for gate, state in result["gate_ledger"].items():
                    gate_counts[gate][state] += 1
                blocker = result["first_missing_premise"]
                blocker_counts[blocker] += 1
                result_sha = result["content_sha256"]
            records.append(
                {
                    "candidate_id": payload["candidate_id"],
                    "typed_action_ir_sha256": payload["typed_action_ir_sha256"],
                    "preflight_result_sha256": payload["preflight_result_sha256"],
                    "admission_result_sha256": payload["admission_result_sha256"],
                    "quadratic_coefficient": payload["quadratic_coefficient"],
                    "state": row["state"],
                    "attempt": int(row["attempt"]),
                    "result_sha256": result_sha,
                    "first_missing_premise": blocker,
                    "error_text": row["error_text"],
                }
            )
        body = {
            "schema_version": STATUS_SCHEMA,
            "execution_enabled": self.config["execution_enabled"],
            "candidate_count": 2,
            "candidate_registry_root_sha256": self.candidate_registry_root_sha256,
            "reviewed_adapter_registry_root_sha256": self.reviewed_adapter_registry_root_sha256,
            "promotion_status_binding": self.config["promotion_status"],
            "local_formal_campaign_binding": self.config["local_formal_campaign"],
            "positive_mass_audit_binding": self.config["positive_mass_audit"],
            "work_state_counts": dict(sorted(work_counts.items())),
            "decision_counts": dict(sorted(decisions.items())),
            "gate_counts": {
                gate: dict(sorted(counts.items())) for gate, counts in sorted(gate_counts.items())
            },
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "record_registry_root_sha256": _sha(records),
            "checkpoint_sequence": self.coordinator.telemetry()["checkpoint_sequence"],
            "disk_bytes": self._disk_bytes(),
            "general_nonmaximal_global_positive_mass_proved": False,
            "full_formal_pass_count": 0,
            "observational_data_opened": False,
            "data_eligibility": {**ELIGIBILITY, "passed": True},
            "paid_llm_spend_usd": 0.0,
        }
        return {**body, "content_sha256": _sha(body)}


def portable_status(status: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in status.items()
        if key not in {"content_sha256", "checkpoint_sequence", "disk_bytes"}
    }
    return {**body, "content_sha256": _sha(body)}
