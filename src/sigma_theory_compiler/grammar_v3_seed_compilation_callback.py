from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from .covariant_grammar_v3_seed_compilation_campaign import (
    build_covariant_grammar_v3_seed_compilation_campaign,
)
from .grammar_v3_seed_execution import CALLBACK_RESULT_SCHEMA
from .promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "covariant_grammar_v3_seed_compilation_campaign.json"
RESULT_PATH = ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-compilation-campaign.json"
CONFIG_FILE_SHA256 = "31ebf4e6637dcc328a17e498a89f1cbfba0b5430952b40771dfe16b18aff0795"
RESULT_FILE_SHA256 = "88c72002d12fd57a8ef79b166363319fe5dc372b52901c1e581bb382fa3e0c21"
RESULT_CONTENT_SHA256 = "b00cd0ab37f8f7f66f0561df68c1d6735ba3853ad7d3124783202648bc804d47"
BOUND_CAMPAIGN_INPUTS = (
    Path("runs/engine/covariant-grammar-v3-seed-manifest.json"),
    Path("runs/formal-controls-v1/formal-controls.json"),
    Path("configs/covariant_action_grammar.json"),
    Path("configs/covariant_field_contract.json"),
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("grammar-v3 compilation campaign file must contain an object")
    return value


def _rebuild_in_isolated_root(config: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sigma-g3-reviewed-callback-") as temporary:
        isolated_root = Path(temporary) / "project"
        for relative_path in BOUND_CAMPAIGN_INPUTS:
            source = ROOT / relative_path
            destination = isolated_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return build_covariant_grammar_v3_seed_compilation_campaign(
            config, isolated_root
        )


@lru_cache(maxsize=1)
def _reviewed_campaign() -> dict[str, dict[str, Any]]:
    if _file_sha(CONFIG_PATH) != CONFIG_FILE_SHA256:
        raise ValueError("grammar-v3 compilation campaign config hash mismatch")
    if _file_sha(RESULT_PATH) != RESULT_FILE_SHA256:
        raise ValueError("grammar-v3 compilation campaign result file hash mismatch")
    config = _load(CONFIG_PATH)
    committed = _load(RESULT_PATH)
    body = {key: value for key, value in committed.items() if key != "content_sha256"}
    if committed.get("content_sha256") != RESULT_CONTENT_SHA256 or _sha(body) != (
        RESULT_CONTENT_SHA256
    ):
        raise ValueError("grammar-v3 compilation campaign result content mismatch")
    rebuilt = _rebuild_in_isolated_root(config)
    if rebuilt != committed:
        raise ValueError("grammar-v3 compilation campaign is not an exact rebuild")
    if (
        rebuilt.get("seed_count") != 6
        or rebuilt.get("decision_counts") != {"blocked": 6}
        or rebuilt.get("solar_bundle_count") != 0
        or rebuilt.get("observational_data_opened") is not False
        or rebuilt.get("data_eligibility") != ELIGIBILITY
    ):
        raise ValueError("grammar-v3 compilation campaign outcome contract changed")
    records = {str(record["seed_id"]): record for record in rebuilt["candidate_records"]}
    if len(records) != 6:
        raise ValueError("grammar-v3 compilation campaign candidate ids are not unique")
    return records


def reviewed_candidate_compiler_formal_callback(
    seed: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    if context.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("grammar-v3 callback context eligibility is not fail-closed")
    if context.get("external_paid_llm_calls") is not False:
        raise ValueError("grammar-v3 callback context enabled paid LLM calls")
    if context.get("manifest_content_sha256") != (
        "e28ad576a68648f11892a3ff1fff5b7e18057f1ba0b27c89d538499e46de171b"
    ):
        raise ValueError("grammar-v3 callback context manifest binding mismatch")
    record = _reviewed_campaign().get(str(seed.get("seed_id")))
    if record is None:
        raise ValueError("grammar-v3 seed is absent from the reviewed compilation campaign")
    if (
        record.get("decision") != "blocked"
        or record.get("provenance", {}).get("seed_lineage_sha256")
        != seed.get("seed_lineage_sha256")
        or record.get("typed_action_ir", {}).get("seed_lineage_sha256")
        != seed.get("seed_lineage_sha256")
    ):
        raise ValueError("grammar-v3 reviewed compilation candidate lineage mismatch")
    compilation = {
        "campaign_result_file_sha256": RESULT_FILE_SHA256,
        "campaign_result_content_sha256": RESULT_CONTENT_SHA256,
        "typed_action_ir": record["typed_action_ir"],
        "parameter_certificate": record["parameter_certificate"],
        "declared_adapter_entrypoints": record["declared_adapter_entrypoints"],
        "invoked_adapter_entrypoints": record["invoked_adapter_entrypoints"],
        "adapter_invocations": record["adapter_invocations"],
        "provenance": record["provenance"],
    }
    formal = {
        "decision": record["decision"],
        "gate_ledger": record["gate_ledger"],
        "solar_known_answer_bundle": record["solar_known_answer_bundle"],
        "observational_data_opened": False,
    }
    return {
        "schema_version": CALLBACK_RESULT_SCHEMA,
        "decision": "blocked",
        "candidate_compilation": compilation,
        "formal_result": formal,
        "blocker": "formal_prerequisites_incomplete",
        "data_eligibility": dict(ELIGIBILITY),
    }
