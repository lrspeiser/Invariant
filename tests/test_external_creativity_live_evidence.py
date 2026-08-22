from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import external_creativity_live_evidence as V

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / V.OUTPUT_PATH


def test_promoted_live_evidence_is_sealed_and_claim_neutral() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    V.validate_evidence(evidence)
    assert len(evidence["calls"]) == 8
    assert evidence["usage"]["calls"] == 8
    assert evidence["usage"]["total_tokens"] <= 64000
    assert not evidence["claims"]["credential_material_included"]
    assert not evidence["claims"]["model_output_is_verifier_authority"]
    assert not evidence["claims"]["novel_formula_established"]
    assert not evidence["source_receipt"]["raw_prompts_or_outputs_copied"]
    serialized = json.dumps(evidence, sort_keys=True)
    assert "hypotheses" not in serialized
    assert "rationale" not in serialized
    assert "x-api-key" not in serialized
