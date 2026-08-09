from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_symmetrizer_domain import (
    run_quartic_symmetrizer_domain_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
IR = ROOT / "runs/physics-language/horndeski-l2-l4-polynomial-ir.json"
BINDINGS = (
    ROOT / "runs/physics-language/quartic-linear-x-symbol-campaign/campaign.json"
)
CONFIG = (
    ROOT / "configs/backgrounds/quartic_symmetrizer_uniform_domain_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-symmetrizer-uniform-domain-campaign/campaign.json"
)


def test_all_twelve_quartic_candidates_have_uniform_symmetrizer_boxes() -> None:
    ir = json.loads(IR.read_text(encoding="utf-8"))
    bindings = json.loads(BINDINGS.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    result = run_quartic_symmetrizer_domain_campaign(ir, bindings, config)
    assert result["status"] == (
        "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes"
    ), result
    assert result["counts"] == {
        "selected": 12,
        "uniform_local_jet_strong_hyperbolicity_passed": 12,
        "rejected": 0,
    }
    assert result["content_sha256"] == artifact["content_sha256"]
    assert len(result["certificates"]) == 12
    for certificate in result["certificates"]:
        assert certificate["status"] == (
            "pass_uniform_local_jet_strong_hyperbolicity"
        )
        assert all(
            value is True
            for value in certificate["theorem_binding"].values()
            if isinstance(value, bool)
        )
        bounds = certificate["uniform_matrix_bounds"]
        assert bounds["companion_margin_numeric"] > 0
        assert certificate["domain"]["on_shell_invariance_status"] == "unresolved"
    assert all(
        record["rejected"] for record in result["negative_controls"].values()
    )


def test_binding_hash_mismatch_rejects_before_domain_promotion() -> None:
    ir = json.loads(IR.read_text(encoding="utf-8"))
    bindings = json.loads(BINDINGS.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    corrupted = copy.deepcopy(bindings)
    corrupted["source_ir_sha256"] = "corrupted"
    result = run_quartic_symmetrizer_domain_campaign(ir, corrupted, config)
    assert result["status"] == "reject"
    assert "hash mismatch" in " ".join(result["errors"])
