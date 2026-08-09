from __future__ import annotations

import pytest
import sympy as sp

from sigma_theory_compiler.principal_symbol import (
    analyze_anisotropic_second_order_symbol,
    analyze_isotropic_second_order_symbol,
    analyze_reduced_quadratic_lagrangian_symbol,
    run_anisotropic_principal_symbol_controls,
    run_extracted_principal_symbol_controls,
    run_principal_symbol_controls,
    run_uniform_multifield_block_controls,
    run_uniform_scalar_anisotropy_controls,
)


def test_scalar_and_reduced_proca_are_metric_cone_hyperbolic() -> None:
    scalar = analyze_isotropic_second_order_symbol(sp.eye(1), sp.eye(1))
    proca = analyze_isotropic_second_order_symbol(sp.eye(3), sp.eye(3))
    assert scalar.passed and scalar.speed_squared == (sp.Integer(1),)
    assert proca.passed and proca.speed_squared == (sp.Integer(1),) * 3
    wave_number, omega = sp.symbols("k omega", real=True)
    assert sp.simplify(scalar.principal_polynomial - (wave_number**2 - omega**2)) == 0


def test_pathology_controls_are_separately_identified() -> None:
    ghost = analyze_isotropic_second_order_symbol(sp.diag(1, -1), sp.eye(2))
    gradient = analyze_isotropic_second_order_symbol(sp.eye(2), sp.diag(1, -1))
    fast = analyze_isotropic_second_order_symbol(sp.eye(1), sp.Matrix([[4]]))
    assert not ghost.ghost_free
    assert not gradient.gradient_stable
    assert gradient.real_characteristics
    assert not fast.cone_policy_pass
    assert fast.speed_squared == (sp.Integer(4),)


def test_singular_unreduced_symbol_fails_closed() -> None:
    result = analyze_isotropic_second_order_symbol(sp.diag(0, 1), sp.eye(2))
    assert not result.passed
    assert result.speed_squared == ()


def test_nonsymmetric_quadratic_matrices_are_rejected() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        analyze_isotropic_second_order_symbol(sp.Matrix([[1, 1], [0, 1]]), sp.eye(2))


def test_full_principal_symbol_control_packet_passes() -> None:
    assert run_principal_symbol_controls()["passed"]


def test_anisotropic_symbol_rejects_zero_direction() -> None:
    blocks = [[sp.eye(1) if i == j else sp.zeros(1) for j in range(3)] for i in range(3)]
    with pytest.raises(ValueError, match="zero spatial direction"):
        analyze_anisotropic_second_order_symbol(sp.eye(1), blocks, [(0, 0, 0)])


def test_oblique_direction_catches_cross_gradient_instability() -> None:
    controls = run_anisotropic_principal_symbol_controls()
    assert controls["passed"]
    assert controls["negative_control_axes_pass"]
    failure = controls["negative_control_oblique_failure"]
    assert failure["direction"] == ["1", "-1", "0"]
    assert not failure["gradient_stable"]


def test_reduced_lagrangian_principal_blocks_are_extracted_automatically() -> None:
    velocity = sp.symbols("v")
    gx, gy = sp.symbols("gx gy")
    result = analyze_reduced_quadratic_lagrangian_symbol(
        (velocity**2 - gx**2 - 2 * gy**2) / 2,
        (velocity,),
        ((gx, gy),),
        ((1, 0), (0, 1)),
        maximum_speed_squared=2,
    )
    assert result["passed"]
    assert result["extraction"]["kinetic_matrix"] == "Matrix([[1]])"
    assert result["analysis"]["directional_results"][1]["speed_squared"] == ["2"]


def test_time_space_mixed_principal_block_fails_closed() -> None:
    controls = run_extracted_principal_symbol_controls()
    assert controls["passed"]
    mixed = controls["time_space_mixed_characteristics"]
    assert mixed["status"] == "pass"
    assert mixed["extraction"]["time_space_mixed_present"]
    first_direction = mixed["analysis"]["directional_results"][0]
    assert first_direction["real_characteristics"]
    assert first_direction["complete_eigenbasis"]
    defective = controls["defective_mixed_negative_control"]
    assert not defective["passed"]
    assert not defective["analysis"]["directional_results"][0]["complete_eigenbasis"]


def test_uniform_scalar_anisotropy_uses_full_sphere_eigenvalue_bounds() -> None:
    controls = run_uniform_scalar_anisotropy_controls()
    assert controls["passed"]
    stable = controls["stable_anisotropic"]
    assert stable["minimum_speed_squared"] == "1/4"
    assert stable["maximum_speed_squared"] == "1"
    unstable = controls["off_axis_unstable"]
    assert not unstable["gradient_stable"]
    assert "-1" in unstable["principal_speed_squared_eigenvalues"]


def test_uniform_multifield_block_certificate_is_exact_and_conservative() -> None:
    controls = run_uniform_multifield_block_controls()
    assert controls["passed"]
    stable = controls["stable_two_field"]
    assert stable["status"] == "pass"
    assert stable["uniform_strong_hyperbolicity"]
    off_axis = controls["off_axis_inconclusive_negative_control"]
    assert off_axis["status"] == "unresolved"
    assert not off_axis["gradient_block_positive_semidefinite"]
    superluminal = controls["superluminal_inconclusive_negative_control"]
    assert superluminal["status"] == "unresolved"
    assert not superluminal["cone_block_positive_semidefinite"]
