from pathlib import Path

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.q_variation_ir import compile_q_variation_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"


def _action(name: str) -> dict:
    return compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)


def test_q_completion_has_exact_fixed_metric_vector_variation_only() -> None:
    action = _action("generated_gf_5df8715b319f54cb_static_null_completion.json")
    result = compile_q_variation_ir(action)
    assert result["status"] == "unresolved"
    fixed = result["fixed_metric_vector_variation"]
    assert fixed["status"] == "pass"
    assert fixed["algebraic_controls"] == {
        "projector_and_B_first_variation_residual": "0",
        "acceleration_norm_first_variation_residual": "0",
        "passed": True,
        "arithmetic": "exact rational tensor contractions in signature (-,+,+)",
    }
    assert result["metric_variation"]["status"] == "unresolved"
    assert result["diffeomorphism_noether_identity"]["status"] == "unresolved"
    assert not result["observational_data_opened"]


def test_non_q_control_is_not_applicable_and_passes() -> None:
    result = compile_q_variation_ir(_action("einstein_hilbert_control.json"))
    assert result["status"] == "pass"
    assert not result["applicable"]
