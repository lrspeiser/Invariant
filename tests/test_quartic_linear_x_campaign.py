from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path

from sigma_theory_compiler.quartic_linear_x_campaign import (
    run_quartic_linear_x_symbol_campaign,
)
from sigma_theory_compiler.scalar_tensor_pack import compile_scalar_tensor_pack

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "configs/operator_packs/horndeski_l2_l4_polynomial.json"
CONFIG = ROOT / "configs/backgrounds/quartic_linear_x_symbol_campaign.json"


@cache
def _compiled_ir() -> dict:
    return compile_scalar_tensor_pack(json.loads(PACK.read_text(encoding="utf-8")))


def test_all_specialized_linear_x_quartic_mutations_bind_exactly() -> None:
    ir = _compiled_ir()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = run_quartic_linear_x_symbol_campaign(ir, config)
    assert result["status"] == (
        "pass_exact_symbol_binding_uniform_symmetrizer_unresolved"
    ), result
    assert result["counts"] == {
        "selected": 12,
        "exactly_bound": 12,
        "canonical_G2": 4,
        "quadratic_kessence_G2": 8,
        "rejected": 0,
    }
    assert len(result["candidates"]) == 12
    assert all(
        item["status"] == "pass_exact_11x11_symbol_binding_symmetrizer_unresolved"
        for item in result["candidates"]
    )
    assert result["proof_controls"]["canonical_linear_X_quartic"][
        "extraction_status"
    ] == "pass"
    assert result["proof_controls"]["quadratic_kessence_extension"][
        "arbitrary_covector_effective_metric_residual"
    ] == "0"
    assert result["negative_controls"][
        "phi_dependent_G4_outside_extracted_symbol"
    ]["rejected"]


def test_phi_dependent_g4_specialization_fails_closed() -> None:
    ir = _compiled_ir()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    corrupted = copy.deepcopy(config)
    corrupted["fixed_coefficients"]["a01"] = "1"
    result = run_quartic_linear_x_symbol_campaign(ir, corrupted)
    assert result["status"] == "reject"
    assert "a01" in " ".join(result["errors"])
