from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.covariant_variation import (
    render_proca_cadabra_variation,
    vary_proca_action_file,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "actions" / "proca_control.json"
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"


def test_proca_action_ir_compiles_and_renders_without_matter_diagnostics() -> None:
    action_ir = compile_action_file(SPEC, GRAMMAR, CONTRACT)
    assert action_ir["valid"], action_ir["errors"]
    script = render_proca_cadabra_variation(action_ir)
    assert "field_strength" in script
    assert "vary(action" in script
    assert "integrate_by_parts(action" in script
    assert "J_b" not in script and "z_b" not in script


def test_generated_proca_variation_executes_in_cadabra(tmp_path) -> None:
    result = vary_proca_action_file(
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
