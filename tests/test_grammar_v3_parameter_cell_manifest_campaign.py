from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_parameter_cell_manifest_campaign import (
    build_parameter_cell_manifest,
    iter_parameter_cells,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_parameter_cell_manifest_campaign.json"
SOURCE = ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-manifest.json"
ARTIFACT = ROOT / "runs" / "engine" / "grammar-v3-parameter-cell-manifest.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_256_cell_manifest_reproduces_and_is_disjoint() -> None:
    manifest = build_parameter_cell_manifest(_load(CONFIG), ROOT)
    cells = list(iter_parameter_cells(manifest, _load(SOURCE)))
    assert len(cells) == len({cell["parameter_cell_id"] for cell in cells}) == 256
    assert manifest["family_cell_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 32,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 64,
    }
    assert [chunk["range"] for chunk in manifest["chunks"]] == [
        {"start": start, "stop": start + 32} for start in range(0, 256, 32)
    ]
    assert [cell["ordinal"] for cell in cells] == list(range(256))
    assert manifest["formal_evaluation_performed"] is False
    assert manifest["scientific_decision_counts"] == {}
    assert manifest["data_eligibility"] == ELIGIBILITY
    assert manifest["observational_data_opened"] is False
    assert manifest["paid_llm_spend_usd"] == 0.0


def test_domain_margins_and_existing_evaluator_semantics_are_preserved() -> None:
    manifest = build_parameter_cell_manifest(_load(CONFIG), ROOT)
    cells = list(iter_parameter_cells(manifest, _load(SOURCE)))
    kessence = [cell for cell in cells if cell["family_id"] == "KESSENCE_G2_CONVEX"]
    assert {cell["rational_coordinates"]["alpha"] for cell in kessence} == {
        "1/8",
        "1/4",
    }
    assert all("G2_X>=1" in cell["domain_contract"] for cell in kessence)
    g4 = [
        cell
        for cell in cells
        if cell["family_id"] == "CONFORMAL_G4_PHI_SCALAR_TENSOR"
    ]
    assert {cell["rational_coordinates"]["xi"] for cell in g4} == {"1/100"}
    assert manifest["evaluator_semantics_changed"] is False
    assert "new reviewed candidate-compilation campaign" in manifest[
        "next_execution_hook"
    ]["required_next_adapter"]


def test_negative_controls_reject_duplicate_domain_overflow_and_forbidden_data() -> None:
    manifest = build_parameter_cell_manifest(_load(CONFIG), ROOT)
    assert manifest["negative_control_counts"] == {"reject": 6}
    reasons = {item["reason"] for item in manifest["negative_control_results"]}
    assert reasons == {
        "duplicate_parameter_cell_id",
        "equivalent_parameter_cell_after_rational_normalization",
        "invalid_parameter_domain",
        "finite_cell_budget_overflow",
        "forbidden_data_input",
        "invalid_exact_rational",
    }

    duplicate = _load(CONFIG)
    duplicate["family_grids"]["AETHER_K1234_PARAMETER_CELL"]["c1"] = [
        "1/8",
        "2/16",
        "3/16",
        "1/4",
    ]
    with pytest.raises(ValueError, match="noncanonical exact rational|duplicate or equivalent"):
        build_parameter_cell_manifest(duplicate, ROOT)

    invalid = _load(CONFIG)
    invalid["family_grids"]["KESSENCE_G2_CONVEX"]["alpha"] = ["-1/8", "1/4"]
    with pytest.raises(ValueError, match="outside unchanged evaluator semantics"):
        build_parameter_cell_manifest(invalid, ROOT)

    overflow = _load(CONFIG)
    overflow["finite_budget"]["maximum_cells"] = 513
    with pytest.raises(ValueError, match="finite and defensible"):
        build_parameter_cell_manifest(overflow, ROOT)

    forbidden = _load(CONFIG)
    forbidden["data_eligibility"]["dark_matter_or_halo_inputs"] = True
    with pytest.raises(ValueError, match="not fail-closed"):
        build_parameter_cell_manifest(forbidden, ROOT)


def test_hash_tamper_and_replay_tamper_fail_closed() -> None:
    config = _load(CONFIG)
    config["source_seed_manifest"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_parameter_cell_manifest(config, ROOT)

    manifest = build_parameter_cell_manifest(_load(CONFIG), ROOT)
    manifest["compact_grid_contract"]["family_quotas"]["KESSENCE_G2_CONVEX"] = 63
    with pytest.raises(ValueError, match="manifest hash|compact grid"):
        list(iter_parameter_cells(manifest, _load(SOURCE)))


def test_committed_compact_manifest_is_exact() -> None:
    assert build_parameter_cell_manifest(_load(CONFIG), ROOT) == _load(ARTIFACT)
