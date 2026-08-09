from pathlib import Path

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.higher_jet_ir import compile_higher_jet_auxiliary_ir

ROOT = Path(__file__).resolve().parents[1]


def _action(name: str) -> dict:
    return compile_action_file(
        ROOT / "configs" / "actions" / name,
        ROOT / "configs" / "covariant_action_grammar.json",
        ROOT / "configs" / "covariant_field_contract.json",
    )


def test_q_action_gets_exact_auxiliary_lift_without_false_dirac_pass() -> None:
    result = compile_higher_jet_auxiliary_ir(
        _action("generated_gf_5df8715b319f54cb_static_null_completion.json")
    )
    assert result["status"] == "unresolved"
    assert result["equivalence_certificate"]["status"] == "pass"
    assert result["lift"]["on_constraint_result"].endswith("every admitted Q power")
    assert result["lift"]["maximum_derivative_order_per_independent_field"] == {
        "b_mu": 1,
        "r^mu": 0,
        "u_mu": 1,
    }
    assert result["required_dirac_work"]
    assert not result["promotion_allowed"]


def test_non_q_action_marks_auxiliary_lift_not_applicable() -> None:
    result = compile_higher_jet_auxiliary_ir(_action("einstein_hilbert_control.json"))
    assert result["status"] == "pass"
    assert not result["applicable"]
    assert result["lift"] is None
