from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler import external_creativity_lean_bridge as L

ROOT = Path(__file__).resolve().parents[1]


def test_formula_specific_source_and_closed_premise_manifest() -> None:
    source = (ROOT / L.SOURCE_PATH).read_text(encoding="utf-8")
    assert "theorem recoveredKineticNormalForm" in source
    assert "theorem externalSumSquaresClosedForm" in source
    assert "theorem externalKnownFormulaControls" in source
    assert "sorry" not in source.lower()
    assert "axiom " not in source.lower()
    config = L.adapter_config(ROOT / "missing-lean")
    assert config.target == L.TARGET
    assert "Invariant.externalSumSquaresClosedForm" in config.allowed_premises
    assert "Classical.choice" in config.forbidden_premises


def test_missing_lean_fails_closed_without_a_formal_claim() -> None:
    receipt = L.run_bridge(ROOT, executable=ROOT / "missing-lean", environment={})
    L.validate_receipt(receipt)
    assert receipt["status"] == "BLOCKED_LEAN_UNAVAILABLE_OR_REJECTED"
    assert not receipt["claims"]["known_formula_normal_forms_kernel_checked"]
    assert not receipt["claims"]["novel_formula_established"]
    assert not receipt["claims"]["physical_law_proved"]
