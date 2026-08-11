from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_parameter_cell_compilation_campaign import (
    build_parameter_cell_compilation_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_parameter_cell_compilation_campaign.json"
ARTIFACT = ROOT / "runs" / "engine" / "grammar-v3-parameter-cell-compilation-campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_256_cells_compile_to_163_unique_actions_with_exact_dedup() -> None:
    campaign = build_parameter_cell_compilation_campaign(_load(CONFIG), ROOT)
    assert campaign["input_parameter_cell_count"] == 256
    assert campaign["compiled_action_ir_count"] == 256
    assert campaign["unique_candidate_count"] == 163
    assert campaign["equivalent_duplicate_count"] == 93
    assert campaign["candidate_decision_counts"] == {
        "pass": 163,
        "reject": 0,
        "blocked": 0,
    }
    assert campaign["cell_disposition_counts"] == {
        "compiled_representative": 163,
        "deduplicated_equivalent": 93,
    }
    assert campaign["family_audit"] == {
        "AETHER_K1234_PARAMETER_CELL": {
            "input_cells": 128,
            "unique_actions": 128,
            "equivalent_duplicates": 0,
        },
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": {
            "input_cells": 32,
            "unique_actions": 1,
            "equivalent_duplicates": 31,
        },
        "CUBIC_HORNDESKI_G3_WEAK_CELL": {
            "input_cells": 32,
            "unique_actions": 32,
            "equivalent_duplicates": 0,
        },
        "KESSENCE_G2_CONVEX": {
            "input_cells": 64,
            "unique_actions": 2,
            "equivalent_duplicates": 62,
        },
    }


def test_all_cheap_policy_gates_pass_without_formal_or_data_opening() -> None:
    campaign = build_parameter_cell_compilation_campaign(_load(CONFIG), ROOT)
    assert campaign["structural_gate_pass_counts"] == {
        name: 256
        for name in (
            "action_policy_bounds",
            "cell_lineage",
            "data_eligibility",
            "field_contract",
            "forbidden_tokens_absent",
            "registered_operators",
            "typed_family",
            "universal_matter_coupling",
        )
    }
    assert campaign["expensive_formal_campaign_run"] is False
    assert campaign["formal_decision_counts"] == {}
    assert campaign["observational_data_opened"] is False
    assert campaign["data_eligibility"] == ELIGIBILITY
    assert campaign["paid_llm_spend_usd"] == 0.0
    assert campaign["negative_control_counts"] == {"reject": 5}


def test_chunks_are_disjoint_and_resume_hook_is_hash_bound() -> None:
    campaign = build_parameter_cell_compilation_campaign(_load(CONFIG), ROOT)
    assert len(campaign["chunks"]) == 8
    assert [chunk["range"] for chunk in campaign["chunks"]] == [
        {"start": start, "stop": start + 32} for start in range(0, 256, 32)
    ]
    assert sum(chunk["compiled_representative_count"] for chunk in campaign["chunks"]) == 163
    assert sum(chunk["deduplicated_equivalent_count"] for chunk in campaign["chunks"]) == 93
    hook = campaign["next_execution_hook"]
    assert hook["chunk_count"] == 8
    assert hook["chunk_size"] == 32
    assert "receipt_registry_root_sha256" in hook["resume_key"]
    assert "missing adapters block" in hook["required_adapter"]


def test_manifest_compiler_and_policy_tamper_fail_closed() -> None:
    for key, message in (
        ("parameter_cell_manifest", "parameter-cell manifest file hash mismatch"),
        ("compiler_semantics", "compiler semantics file hash mismatch"),
        ("field_contract", "field contract file hash mismatch"),
    ):
        config = _load(CONFIG)
        config[key]["file_sha256"] = "0" * 64
        with pytest.raises(ValueError, match=message):
            build_parameter_cell_compilation_campaign(config, ROOT)

    forbidden = _load(CONFIG)
    forbidden["data_eligibility"]["paid_llm_calls"] = True
    with pytest.raises(ValueError, match="not fail-closed"):
        build_parameter_cell_compilation_campaign(forbidden, ROOT)

    overflow = _load(CONFIG)
    overflow["finite_budget"]["maximum_cells"] = 257
    with pytest.raises(ValueError, match="budget is invalid"):
        build_parameter_cell_compilation_campaign(overflow, ROOT)


def test_committed_campaign_is_exact() -> None:
    assert build_parameter_cell_compilation_campaign(_load(CONFIG), ROOT) == _load(ARTIFACT)
