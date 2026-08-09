from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.covariant_variation import (
    render_scalar_cadabra_variation,
    vary_scalar_action_file,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "actions" / "canonical_scalar_control.json"
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"


def test_action_ir_renders_a_bounded_cadabra_scalar_variation() -> None:
    action_ir = compile_action_file(SPEC, GRAMMAR, CONTRACT)
    script = render_scalar_cadabra_variation(action_ir)
    assert "vary(action" in script
    assert "integrate_by_parts(action" in script
    assert "factor_out(action" in script
    assert "J_b" not in script and "z_b" not in script


def test_generated_scalar_variation_executes_in_cadabra(tmp_path) -> None:
    result = vary_scalar_action_file(
        SPEC,
        GRAMMAR,
        CONTRACT,
        tmp_path,
        project_root=ROOT,
    )
    if result["status"] == "unresolved":
        assert not result["backend"]["available"]
    else:
        assert result["status"] == "pass", result["execution"]
        assert result["input_action_sha256"]
        assert Path(result["result_path"]).exists()
