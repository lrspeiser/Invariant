from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .grammar_v3_seed_compilation_callback import (
    CONFIG_FILE_SHA256,
    RESULT_CONTENT_SHA256,
    RESULT_FILE_SHA256,
)
from .grammar_v3_seed_execution import (
    GrammarV3SeedExecution,
    callback_binding,
)
from .persistent_parallel_search import PersistentParallelSearch
from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-grammar-v3-seed-compilation-execution-status-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def run_grammar_v3_seed_compilation_execution(
    database: str | Path,
    coordinator_config: dict[str, Any],
    resource_profile: dict[str, Any],
    *,
    manifest_path: str | Path,
    manifest_file_sha256: str,
    manifest_content_sha256: str,
    callback_descriptor_path: str | Path,
    callback_descriptor_file_sha256: str,
) -> dict[str, Any]:
    if (
        coordinator_config.get("external_paid_llm_calls") is not False
        or int(coordinator_config.get("budget", {}).get("maximum_tasks", -1)) != 6
        or int(coordinator_config.get("queue", {}).get("maximum_pending_work", -1)) < 6
    ):
        raise ValueError("grammar-v3 compilation execution coordinator is not bounded to six")
    descriptor_path = Path(callback_descriptor_path).resolve()
    if not descriptor_path.is_file() or _file_sha(descriptor_path) != (
        callback_descriptor_file_sha256
    ):
        raise ValueError("grammar-v3 callback descriptor file hash mismatch")
    descriptor = _load(descriptor_path)
    artifact = Path(str(descriptor["artifact_path"]))
    if not artifact.is_absolute():
        artifact = (descriptor_path.parent.parent / artifact).resolve()
    descriptor["artifact_path"] = str(artifact)
    binding = callback_binding(descriptor)

    coordinator = PersistentParallelSearch(database, coordinator_config, resource_profile)
    adapter = GrammarV3SeedExecution(
        coordinator,
        manifest_path,
        expected_manifest_file_sha256=manifest_file_sha256,
        expected_manifest_content_sha256=manifest_content_sha256,
        callback_descriptor=descriptor,
    )
    admitted = adapter.enqueue()
    if admitted["accepted"] != 6 or any(
        admitted[key] != 0 for key in ("duplicate", "backpressured", "budget_rejected")
    ):
        raise ValueError("grammar-v3 compilation execution did not admit exactly six seeds")
    execution = adapter.run_bounded(maximum_tasks=6, worker_id="reviewed-grammar-v3")
    status = execution["status"]
    if (
        status["work_state_counts"] != {"succeeded": 6}
        or status["decision_counts"] != {"blocked": 6}
        or status["observational_data_opened"] is not False
        or status["paid_llm_spend_usd"] != 0.0
        or status["data_eligibility"] != {**ELIGIBILITY, "passed": True}
    ):
        raise ValueError("grammar-v3 compilation execution outcome contract changed")
    work_records = [
        {
            "work_id": record["work_id"],
            "adapter_work_id": record["adapter_work_id"],
            "seed_id": record["seed_id"],
            "seed_lineage_sha256": record["seed_lineage_sha256"],
            "ordinal": record["ordinal"],
            "coordinator_seed": record["coordinator_seed"],
            "state": record["state"],
            "attempt": record["attempt"],
            "result_sha256": record["result_sha256"],
            "output_lineage_sha256": record["output_lineage_sha256"],
        }
        for record in status["work_records"]
    ]
    body = {
        "schema_version": SCHEMA_VERSION,
        "source_bindings": {
            "seed_manifest": {
                "file_sha256": manifest_file_sha256,
                "content_sha256": manifest_content_sha256,
            },
            "candidate_compilation_campaign": {
                "config_file_sha256": CONFIG_FILE_SHA256,
                "result_file_sha256": RESULT_FILE_SHA256,
                "result_content_sha256": RESULT_CONTENT_SHA256,
            },
            "reviewed_callback": {
                "descriptor_file_sha256": callback_descriptor_file_sha256,
                "callback_artifact_file_sha256": descriptor["artifact_sha256"],
                "callback_binding_sha256": binding,
                "callback_registry_root_sha256": status[
                    "callback_registry_root_sha256"
                ],
            },
            "coordinator_config_sha256": _sha(coordinator_config),
            "resource_profile_sha256": _sha(resource_profile),
        },
        "seed_count": status["seed_count"],
        "decision_counts": status["decision_counts"],
        "work_state_counts": status["work_state_counts"],
        "seed_registry_root_sha256": status["seed_registry_root_sha256"],
        "work_records_root_sha256": status["work_records_root_sha256"],
        "portable_result_registry_root_sha256": _sha(work_records),
        "work_records": work_records,
        "checkpoint_sequence": status["checkpoint_sequence"],
        "crash_recovery_contract": (
            "expired coordinator leases replay the identical seed payload and deterministic ids"
        ),
        "observational_data_opened": False,
        "data_eligibility": {**ELIGIBILITY, "passed": True},
        "paid_llm_spend_usd": 0.0,
        "interpretation": (
            "All six grammar-v3 seeds compiled through the reviewed candidate-specific campaign. "
            "Every result remains formally blocked; no Solar or observational gate opened."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
