from pathlib import Path

import pytest

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.x_operator_ir import compile_x_operator_ir

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"


@pytest.mark.parametrize(
    "name,exponent",
    [
        ("generated_cv3_x_p1_2_static_null.json", "1/2"),
        ("generated_cv3_x_p2_3_static_null.json", "2/3"),
        ("generated_cv3_x_p3_4_static_null.json", "3/4"),
    ],
)
def test_constant_gradient_x_completions_reject_at_high_background(
    name: str, exponent: str
) -> None:
    action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)
    result = compile_x_operator_ir(action)
    assert action["valid"], action["errors"]
    assert result["status"] == "reject"
    assert result["exponent"] == exponent
    assert result["local_linear_acceleration_hessian"] == "0"
    assert result["unbounded_characteristic_speed"]
    assert result["hamiltonian_derivative_residual"] == "0"
    assert all(item["status"] == "pass" for item in result["sign_certificates"].values())
    assert "unbounded" in result["conclusion"]


def test_non_x_control_is_not_applicable() -> None:
    action = compile_action_file(
        ROOT / "configs" / "actions" / "einstein_hilbert_control.json",
        GRAMMAR,
        CONTRACT,
    )
    result = compile_x_operator_ir(action)
    assert result["status"] == "pass"
    assert not result["applicable"]


@pytest.mark.parametrize(
    "name,maximum",
    [
        ("generated_cv3_x_p1_2_matched.json", "gamma/epsilon"),
        ("generated_cv3_x_p2_3_matched.json", "3*gamma/(4*epsilon)"),
        ("generated_cv3_x_p3_4_matched.json", "2*gamma/(3*epsilon)"),
    ],
)
def test_derivative_matched_completions_fail_generic_shear_legendre_gate(
    name: str, maximum: str
) -> None:
    action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)
    result = compile_x_operator_ir(action)
    assert action["valid"], action["errors"]
    assert result["status"] == "reject"
    assert result["completion_kind"] == "derivative_matched_static_null_K1_plus_K4"
    assert result["completion_exponent_consistent"]
    assert not result["unbounded_characteristic_speed"]
    assert result["global_maximum_speed_squared"] == maximum
    assert result["matter_cone_certificate"]["status"] == "pass"
    assert all(item["status"] == "pass" for item in result["sign_certificates"].values())
    coupled = result["coupled_zero_extrinsic_curvature_legendre"]
    assert coupled["normalized_completion_strength"] == "gamma"
    assert coupled["cross_block"].startswith("zero at K_ij=0")
    witness = result["generic_traceless_curvature_legendre_witness"]
    assert witness["status"] == "reject"
    assert witness["parameter_point_satisfies_declared_domain"]
    assert witness["negative_exactly"]
    assert "rank-changing" in witness["consequence"]
