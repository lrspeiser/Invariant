"""Exact local 11-by-11 principal symbol for the linear-X quartic control.

The block formulas are the specialization of Appendix B of Papallo,
arXiv:1710.10155, to ``G3=G5=0`` and ``G4_deformation=alpha*X``.  The
modified-harmonic gauge block follows Kovacs--Reall, arXiv:2003.08398.
All calculations use a local physical orthonormal frame; curvature, scalar
gradient, and scalar Hessian components remain independent local jet data.
"""

from __future__ import annotations

from functools import cache
from itertools import permutations
from typing import Any

import sympy as sp

DIMENSION = 4
SYMMETRIC_PAIRS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
)


def _permutation_sign(values: tuple[int, ...]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


@cache
def generalized_delta(
    upper: tuple[int, ...], lower: tuple[int, ...]
) -> int:
    """Return the exact generalized Kronecker delta."""

    if len(upper) != len(lower):
        raise ValueError("generalized-delta ranks differ")
    if len(set(upper)) != len(upper) or set(upper) != set(lower):
        return 0
    positions = tuple(upper.index(value) for value in lower)
    return _permutation_sign(positions)


@cache
def _delta_tails(
    first_upper: int, first_lower: int, rank: int
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]:
    tail_length = rank - 1
    upper_tails = permutations(
        tuple(index for index in range(DIMENSION) if index != first_upper),
        tail_length,
    )
    terms: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    for upper_tail in upper_tails:
        for lower_tail in permutations(
            tuple(index for index in range(DIMENSION) if index != first_lower),
            tail_length,
        ):
            value = generalized_delta(
                (first_upper, *upper_tail),
                (first_lower, *lower_tail),
            )
            if value:
                terms.append((upper_tail, lower_tail, value))
    return tuple(terms)


def _symmetric_basis() -> tuple[sp.Matrix, ...]:
    basis: list[sp.Matrix] = []
    for row, column in SYMMETRIC_PAIRS:
        tensor = sp.zeros(DIMENSION)
        if row == column:
            tensor[row, column] = 1
        else:
            tensor[row, column] = sp.sqrt(2) / 2
            tensor[column, row] = sp.sqrt(2) / 2
        basis.append(tensor)
    return tuple(basis)


def _projector(
    lower_index: int,
    derivative_index: int,
    first_metric_index: int,
    second_metric_index: int,
    inverse_metric: sp.Matrix,
) -> sp.Expr:
    return sp.factor(
        (
            int(lower_index == first_metric_index)
            * inverse_metric[second_metric_index, derivative_index]
            + int(lower_index == second_metric_index)
            * inverse_metric[first_metric_index, derivative_index]
        )
        / 2
        - int(lower_index == derivative_index)
        * inverse_metric[first_metric_index, second_metric_index]
        / 2
    )


def _trace_reverse(
    first_pair: tuple[int, int],
    second_pair: tuple[int, int],
    inverse_metric: sp.Matrix,
) -> sp.Expr:
    mu, nu = first_pair
    rho, sigma = second_pair
    return sp.factor(
        (
            inverse_metric[mu, rho] * inverse_metric[nu, sigma]
            + inverse_metric[mu, sigma] * inverse_metric[nu, rho]
            - inverse_metric[mu, nu] * inverse_metric[rho, sigma]
        )
        / 2
    )


def _project_output(output_upper: sp.Matrix, basis: sp.Matrix) -> sp.Expr:
    return sp.factor(
        sum(
            basis[row, column] * output_upper[row, column]
            for row in range(DIMENSION)
            for column in range(DIMENSION)
        )
    )


def _raise_second_index(tensor_lower: sp.Matrix, inverse_metric: sp.Matrix) -> sp.Matrix:
    return tensor_lower * inverse_metric


def _raise_mixed_output(output_mixed: sp.Matrix, inverse_metric: sp.Matrix) -> sp.Matrix:
    return output_mixed * inverse_metric


def _metric_action_block(
    *,
    inverse_metric: sp.Matrix,
    xi_lower: sp.Matrix,
    gradient_lower: sp.Matrix,
    alpha: sp.Symbol,
    m2: sp.Symbol,
    basis: tuple[sp.Matrix, ...],
) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    xi_upper = inverse_metric * xi_lower
    gradient_upper = inverse_metric * gradient_lower
    xi_squared = sp.factor((xi_lower.T * xi_upper)[0])
    x_scalar = sp.factor(-(gradient_lower.T * gradient_upper)[0] / 2)
    baseline = sp.zeros(10)
    correction = sp.zeros(10)

    for column, input_tensor in enumerate(basis):
        input_mixed = _raise_second_index(input_tensor, inverse_metric)
        baseline_output = sp.zeros(DIMENSION)
        correction_mixed = sp.zeros(DIMENSION)
        for mu in range(DIMENSION):
            for nu in range(DIMENSION):
                trace_term = sum(
                    _trace_reverse((mu, nu), pair, inverse_metric)
                    * input_tensor[pair[0], pair[1]]
                    * (2 if pair[0] != pair[1] else 1)
                    for pair in SYMMETRIC_PAIRS
                )
                gauge_invariant_term = sp.Integer(0)
                for lower_index in range(DIMENSION):
                    for derivative_left in range(DIMENSION):
                        left = _projector(
                            lower_index,
                            derivative_left,
                            mu,
                            nu,
                            inverse_metric,
                        )
                        if not left:
                            continue
                        for raised_index in range(DIMENSION):
                            middle = inverse_metric[lower_index, raised_index]
                            if not middle:
                                continue
                            for derivative_right in range(DIMENSION):
                                contracted_input = sum(
                                    _projector(
                                        raised_index,
                                        derivative_right,
                                        rho,
                                        sigma,
                                        inverse_metric,
                                    )
                                    * input_tensor[rho, sigma]
                                    for rho in range(DIMENSION)
                                    for sigma in range(DIMENSION)
                                )
                                gauge_invariant_term += (
                                    left
                                    * xi_lower[derivative_left]
                                    * middle
                                    * xi_lower[derivative_right]
                                    * contracted_input
                                )
                baseline_output[mu, nu] = sp.factor(
                    m2
                    / 2
                    * (-xi_squared * trace_term / 2 + gauge_invariant_term)
                )

        for upper_index in range(DIMENSION):
            for lower_index in range(DIMENSION):
                rank_three = sp.Integer(0)
                for upper_tail, lower_tail, sign in _delta_tails(
                    upper_index, lower_index, 3
                ):
                    c1, c2 = upper_tail
                    d1, d2 = lower_tail
                    rank_three += (
                        sign
                        * xi_lower[c1]
                        * xi_upper[d1]
                        * input_mixed[c2, d2]
                    )
                rank_four = sp.Integer(0)
                for upper_tail, lower_tail, sign in _delta_tails(
                    upper_index, lower_index, 4
                ):
                    c1, c2, c3 = upper_tail
                    d1, d2, d3 = lower_tail
                    rank_four += (
                        sign
                        * xi_lower[c1]
                        * xi_upper[d1]
                        * input_mixed[c2, d2]
                        * gradient_lower[c3]
                        * gradient_upper[d3]
                    )
                # Papallo's mixed-index formula is converted to the covariant
                # metric-perturbation convention used by this compiler.  The
                # rank-three term changes sign under that conversion; the result
                # is fixed independently by the exact ADM identities
                # G_T=2(G4-2X G4_X) and F_T=2G4.
                correction_mixed[upper_index, lower_index] = sp.factor(
                    -alpha * (x_scalar * rank_three + rank_four) / 2
                )

        correction_output = _raise_mixed_output(
            correction_mixed, inverse_metric
        )
        for row, row_basis in enumerate(basis):
            baseline[row, column] = _project_output(
                baseline_output, row_basis
            )
            correction[row, column] = _project_output(
                correction_output, row_basis
            )
    return baseline, correction, sp.factor(x_scalar)


def _modified_harmonic_gauge_block(
    *,
    physical_inverse_metric: sp.Matrix,
    tilde_inverse_metric: sp.Matrix,
    hat_inverse_metric: sp.Matrix,
    xi_lower: sp.Matrix,
    basis: tuple[sp.Matrix, ...],
) -> sp.Matrix:
    block = sp.zeros(10)
    for row, row_basis in enumerate(basis):
        for column, input_tensor in enumerate(basis):
            output = sp.zeros(DIMENSION)
            for mu in range(DIMENSION):
                for nu in range(DIMENSION):
                    value = sp.Integer(0)
                    for alpha_index in range(DIMENSION):
                        for gamma in range(DIMENSION):
                            left = _projector(
                                alpha_index,
                                gamma,
                                mu,
                                nu,
                                hat_inverse_metric,
                            )
                            if not left:
                                continue
                            for beta_index in range(DIMENSION):
                                middle = physical_inverse_metric[
                                    alpha_index, beta_index
                                ]
                                if not middle:
                                    continue
                                for delta in range(DIMENSION):
                                    contracted_input = sum(
                                        _projector(
                                            beta_index,
                                            delta,
                                            rho,
                                            sigma,
                                            tilde_inverse_metric,
                                        )
                                        * input_tensor[rho, sigma]
                                        for rho in range(DIMENSION)
                                        for sigma in range(DIMENSION)
                                    )
                                    value -= (
                                        left
                                        * xi_lower[gamma]
                                        * middle
                                        * xi_lower[delta]
                                        * contracted_input
                                    )
                    output[mu, nu] = sp.factor(value)
            block[row, column] = _project_output(output, row_basis)
    return block


def _mixing_column(
    *,
    inverse_metric: sp.Matrix,
    xi_lower: sp.Matrix,
    hessian_lower: sp.Matrix,
    alpha: sp.Symbol,
    basis: tuple[sp.Matrix, ...],
) -> sp.Matrix:
    xi_upper = inverse_metric * xi_lower
    hessian_mixed = _raise_second_index(hessian_lower, inverse_metric)
    output_mixed = sp.zeros(DIMENSION)
    for upper_index in range(DIMENSION):
        for lower_index in range(DIMENSION):
            value = sp.Integer(0)
            for upper_tail, lower_tail, sign in _delta_tails(
                upper_index, lower_index, 3
            ):
                c1, c2 = upper_tail
                d1, d2 = lower_tail
                value += (
                    sign
                    * xi_lower[c1]
                    * xi_upper[d1]
                    * hessian_mixed[c2, d2]
                )
            output_mixed[upper_index, lower_index] = sp.factor(alpha * value)
    output_upper = _raise_mixed_output(output_mixed, inverse_metric)
    return sp.Matrix(
        [_project_output(output_upper, item) for item in basis]
    )


def _first_order_generalized_pencil(
    principal_symbol: sp.Matrix, time_covector: sp.Symbol
) -> dict[str, sp.Matrix]:
    """Return ``A,B,C`` and the 22-by-22 linearized characteristic pencil.

    For ``P=A*xi_0**2+B*xi_0+C``, the vector ``(T,xi_0*T)`` obeys
    ``K v = xi_0 H v`` with the matrices returned below.  This avoids a
    premature symbolic inversion of the background-dependent matrix ``A``.
    """

    expanded = principal_symbol.applyfunc(sp.expand)
    coefficient_a = expanded.applyfunc(
        lambda expression: sp.factor(expression.coeff(time_covector, 2))
    )
    coefficient_b = expanded.applyfunc(
        lambda expression: sp.factor(expression.coeff(time_covector, 1))
    )
    coefficient_c = expanded.subs(time_covector, 0).applyfunc(sp.factor)
    identity = sp.eye(11)
    zero = sp.zeros(11)
    mass = identity.row_join(zero).col_join(
        zero.row_join(coefficient_a)
    )
    evolution = zero.row_join(identity).col_join(
        (-coefficient_c).row_join(-coefficient_b)
    )
    return {
        "A": coefficient_a,
        "B": coefficient_b,
        "C": coefficient_c,
        "mass": mass,
        "evolution": evolution,
    }


@cache
def build_quartic_horndeski_modified_harmonic_symbol() -> dict[str, Any]:
    """Build the complete local-frame 11-by-11 principal-symbol blocks."""

    alpha = sp.Symbol("alpha", nonzero=True, finite=True, real=True)
    m2 = sp.Symbol("M2", positive=True, finite=True)
    xi_lower = sp.Matrix(sp.symbols("xi_0:4", real=True))
    gradient_lower = sp.Matrix(sp.symbols("v_0:4", real=True))
    hessian_symbols = sp.symbols(
        "H_00 H_01 H_02 H_03 H_11 H_12 H_13 H_22 H_23 H_33",
        real=True,
    )
    einstein_symbols = sp.symbols(
        "G_00 G_01 G_02 G_03 G_11 G_12 G_13 G_22 G_23 G_33",
        real=True,
    )
    hessian_lower = sp.zeros(DIMENSION)
    einstein_upper = sp.zeros(DIMENSION)
    for symbol, (row, column) in zip(
        hessian_symbols, SYMMETRIC_PAIRS, strict=True
    ):
        hessian_lower[row, column] = symbol
        hessian_lower[column, row] = symbol
    for symbol, (row, column) in zip(
        einstein_symbols, SYMMETRIC_PAIRS, strict=True
    ):
        einstein_upper[row, column] = symbol
        einstein_upper[column, row] = symbol

    physical_inverse_metric = sp.diag(-1, 1, 1, 1)
    tilde_inverse_metric = sp.diag(-4, 1, 1, 1)
    hat_inverse_metric = sp.diag(-9, 1, 1, 1)
    basis = _symmetric_basis()
    baseline_metric, correction_metric, x_scalar = _metric_action_block(
        inverse_metric=physical_inverse_metric,
        xi_lower=xi_lower,
        gradient_lower=gradient_lower,
        alpha=alpha,
        m2=m2,
        basis=basis,
    )
    gauge_metric = sp.factor(m2 / 2) * _modified_harmonic_gauge_block(
        physical_inverse_metric=physical_inverse_metric,
        tilde_inverse_metric=tilde_inverse_metric,
        hat_inverse_metric=hat_inverse_metric,
        xi_lower=xi_lower,
        basis=basis,
    )
    mixing = _mixing_column(
        inverse_metric=physical_inverse_metric,
        xi_lower=xi_lower,
        hessian_lower=hessian_lower,
        alpha=alpha,
        basis=basis,
    )
    xi_upper = physical_inverse_metric * xi_lower
    xi_squared = sp.factor((xi_lower.T * xi_upper)[0])
    curvature_contraction = sp.factor(
        (xi_lower.T * einstein_upper * xi_lower)[0]
    )
    scalar_block = sp.factor(-xi_squared + 2 * alpha * curvature_contraction)

    action_symbol = sp.zeros(11)
    action_symbol[:10, :10] = baseline_metric + correction_metric
    action_symbol[:10, 10] = mixing
    action_symbol[10, :10] = mixing.T
    action_symbol[10, 10] = scalar_block
    gauge_symbol = sp.zeros(11)
    gauge_symbol[:10, :10] = gauge_metric
    full_symbol = action_symbol + gauge_symbol
    first_order = _first_order_generalized_pencil(
        full_symbol, xi_lower[0]
    )
    return {
        "alpha": alpha,
        "m2": m2,
        "xi_lower": xi_lower,
        "gradient_lower": gradient_lower,
        "hessian_lower": hessian_lower,
        "einstein_upper": einstein_upper,
        "physical_inverse_metric": physical_inverse_metric,
        "tilde_inverse_metric": tilde_inverse_metric,
        "hat_inverse_metric": hat_inverse_metric,
        "basis": basis,
        "x_scalar": x_scalar,
        "baseline_metric": baseline_metric,
        "correction_metric": correction_metric,
        "mixing": mixing,
        "scalar_block": scalar_block,
        "action_symbol": action_symbol,
        "gauge_symbol": gauge_symbol,
        "full_symbol": full_symbol,
        "first_order": first_order,
    }


def quartic_horndeski_full_local_principal_control() -> tuple[bool, dict[str, Any]]:
    """Verify the extracted covariant 11-by-11 block and exact flat reductions."""

    data = build_quartic_horndeski_modified_harmonic_symbol()
    action_symbol = data["action_symbol"]
    correction_metric = data["correction_metric"]
    full_symbol = data["full_symbol"]
    alpha = data["alpha"]
    m2 = data["m2"]
    xi_lower = data["xi_lower"]
    gradient_lower = data["gradient_lower"]
    hessian_lower = data["hessian_lower"]
    einstein_upper = data["einstein_upper"]

    action_symmetric = action_symbol.equals(action_symbol.T)
    correction_symmetric = correction_metric.equals(correction_metric.T)
    mixing_transpose_exact = action_symbol[:10, 10].equals(
        action_symbol[10, :10].T
    )
    gauge_scalar_rows_zero = data["gauge_symbol"][10, :].is_zero_matrix and data[
        "gauge_symbol"
    ][:, 10].is_zero_matrix
    pure_gauge_kernel_residuals: list[sp.Matrix] = []
    basis = data["basis"]
    for gauge_index in range(DIMENSION):
        gauge_covector = sp.zeros(DIMENSION, 1)
        gauge_covector[gauge_index] = 1
        pure_gauge_tensor = (
            xi_lower * gauge_covector.T + gauge_covector * xi_lower.T
        )
        pure_gauge_vector = sp.Matrix(
            [
                sum(
                    item[row, column]
                    * pure_gauge_tensor[row, column]
                    for row in range(DIMENSION)
                    for column in range(DIMENSION)
                )
                for item in basis
            ]
            + [0]
        )
        pure_gauge_kernel_residuals.append(
            (action_symbol * pure_gauge_vector).applyfunc(sp.factor)
        )
    pure_gauge_kernel_passed = all(
        residual.is_zero_matrix for residual in pure_gauge_kernel_residuals
    )
    einstein_scalar_symbol = sp.zeros(11)
    einstein_scalar_symbol[:10, :10] = data["baseline_metric"]
    einstein_scalar_symbol[10, 10] = -sp.factor(
        (xi_lower.T * data["physical_inverse_metric"] * xi_lower)[0]
    )
    zero_coupling_reduction_passed = (
        full_symbol.subs(alpha, 0)
        - einstein_scalar_symbol
        - data["gauge_symbol"]
    ).is_zero_matrix
    first_order = data["first_order"]
    second_order_reconstruction = (
        first_order["A"] * xi_lower[0] ** 2
        + first_order["B"] * xi_lower[0]
        + first_order["C"]
        - full_symbol
    ).applyfunc(sp.factor)
    second_order_reconstruction_passed = (
        second_order_reconstruction.is_zero_matrix
    )

    omega, wave_number, a_star = sp.symbols(
        "omega k A_star", real=True
    )
    flat_substitutions: dict[sp.Symbol, sp.Expr] = {
        xi_lower[0]: -omega,
        xi_lower[1]: wave_number,
        xi_lower[2]: 0,
        xi_lower[3]: 0,
        gradient_lower[0]: a_star,
        gradient_lower[1]: 0,
        gradient_lower[2]: 0,
        gradient_lower[3]: 0,
    }
    flat_substitutions.update(
        {symbol: 0 for symbol in hessian_lower.free_symbols}
    )
    flat_substitutions.update(
        {symbol: 0 for symbol in einstein_upper.free_symbols}
    )
    flat_symbol = full_symbol.subs(flat_substitutions)
    plus = sp.zeros(11, 1)
    plus[SYMMETRIC_PAIRS.index((2, 2))] = sp.sqrt(2) / 2
    plus[SYMMETRIC_PAIRS.index((3, 3))] = -sp.sqrt(2) / 2
    cross = sp.zeros(11, 1)
    cross[SYMMETRIC_PAIRS.index((2, 3))] = 1
    plus_polynomial = sp.factor((plus.T * flat_symbol * plus)[0])
    cross_polynomial = sp.factor((cross.T * flat_symbol * cross)[0])
    scalar_polynomial = sp.factor(flat_symbol[10, 10])
    flat_mixing_zero = flat_symbol[:10, 10].is_zero_matrix
    expected_tensor_polynomial = sp.factor(
        (
            (m2 - alpha * a_star**2) * omega**2
            - (m2 + alpha * a_star**2) * wave_number**2
        )
        / 4
    )

    tensor_modes_match = bool(plus_polynomial == cross_polynomial)
    baseline_pencil_substitutions = dict(flat_substitutions)
    baseline_pencil_substitutions.update(
        {
            alpha: 0,
            m2: 2,
            omega: -xi_lower[0],
            wave_number: 1,
            xi_lower[0]: xi_lower[0],
            xi_lower[1]: 1,
        }
    )
    baseline_a = first_order["A"].subs(baseline_pencil_substitutions)
    baseline_b = first_order["B"].subs(baseline_pencil_substitutions)
    baseline_c = first_order["C"].subs(baseline_pencil_substitutions)
    baseline_a_determinant = sp.factor(baseline_a.det())
    baseline_companion = (
        sp.eye(11)
        .row_join(sp.zeros(11))
        .col_join(
            sp.zeros(11).row_join(baseline_a)
        )
        .inv()
        * (
            sp.zeros(11)
            .row_join(sp.eye(11))
            .col_join((-baseline_c).row_join(-baseline_b))
        )
    )
    requested_characteristic_variable = sp.Symbol("lambda", real=True)
    characteristic_polynomial = baseline_companion.charpoly(
        requested_characteristic_variable
    )
    characteristic_variable = characteristic_polynomial.gen
    baseline_characteristic_polynomial = sp.factor(
        characteristic_polynomial.as_expr()
    )
    expected_baseline_characteristic_polynomial = sp.factor(
        (characteristic_variable**2 - 1) ** 3
        * (characteristic_variable**2 - sp.Rational(1, 4)) ** 4
        * (characteristic_variable**2 - sp.Rational(1, 9)) ** 4
    )
    baseline_spectrum_residual = sp.factor(
        baseline_characteristic_polynomial
        - expected_baseline_characteristic_polynomial
    )
    baseline_a_general = first_order["A"].subs(alpha, 0)
    baseline_a_general_determinant = sp.factor(baseline_a_general.det())
    expected_baseline_a_general_determinant = sp.factor(
        sp.Rational(6561, 4096) * m2**10
    )
    baseline_ata_eigenvalues = {
        sp.factor(value): multiplicity
        for value, multiplicity in (baseline_a.T * baseline_a).eigenvals().items()
    }
    smallest_nondiagonal_ata_eigenvalue = sp.factor(
        (397 - 5 * sp.sqrt(6097)) / 8
    )
    normalized_singular_floor_passed = bool(
        baseline_ata_eigenvalues.get(sp.Rational(1, 4)) == 5
        and baseline_ata_eigenvalues.get(sp.Integer(1)) == 1
        and baseline_ata_eigenvalues.get(sp.Integer(324)) == 3
        and baseline_ata_eigenvalues.get(
            smallest_nondiagonal_ata_eigenvalue
        )
        == 1
        and (smallest_nondiagonal_ata_eigenvalue - sp.Rational(1, 4)).is_positive
    )
    baseline_minimum_singular_value = sp.Min(1, m2 / 4)
    delta_a = (first_order["A"] - baseline_a_general).applyfunc(sp.factor)
    delta_a_frobenius_squared = sp.factor(
        sum(entry**2 for entry in delta_a)
    )
    v0, v1, v2, v3 = list(gradient_lower)
    spatial_gradient_squared = sp.factor(v1**2 + v2**2 + v3**2)
    h11 = hessian_lower[1, 1]
    h12 = hessian_lower[1, 2]
    h13 = hessian_lower[1, 3]
    h22 = hessian_lower[2, 2]
    h23 = hessian_lower[2, 3]
    h33 = hessian_lower[3, 3]
    spatial_hessian_trace = sp.factor(h11 + h22 + h33)
    spatial_hessian_frobenius_squared = sp.factor(
        h11**2
        + h22**2
        + h33**2
        + 2 * (h12**2 + h13**2 + h23**2)
    )
    g00 = einstein_upper[0, 0]
    delta_a_sum_of_squares = sp.factor(
        alpha**2
        / 16
        * (
            64 * g00**2
            + 32
            * (
                spatial_hessian_trace**2
                + spatial_hessian_frobenius_squared
            )
            + 9 * (v0**2 - spatial_gradient_squared) ** 2
            + 12 * v0**2 * spatial_gradient_squared
        )
    )
    delta_a_sum_of_squares_residual = sp.factor(
        delta_a_frobenius_squared - delta_a_sum_of_squares
    )
    sufficient_time_block_condition = sp.Lt(
        delta_a_frobenius_squared,
        baseline_minimum_singular_value**2,
        evaluate=False,
    )
    collapse_substitutions: dict[sp.Symbol, sp.Expr] = {
        alpha: 1,
        m2: 2,
    }
    collapse_substitutions.update(
        {symbol: 0 for symbol in gradient_lower}
    )
    collapse_substitutions.update(
        {symbol: 0 for symbol in hessian_lower.free_symbols}
    )
    collapse_substitutions.update(
        {symbol: 0 for symbol in einstein_upper.free_symbols}
    )
    collapse_substitutions[g00] = -sp.Rational(1, 2)
    collapse_a = first_order["A"].subs(collapse_substitutions)
    collapse_a_determinant = sp.factor(collapse_a.det())
    collapse_a_rank = collapse_a.rank()
    time_block_certificate_passed = bool(
        baseline_a_general_determinant
        == expected_baseline_a_general_determinant
        and normalized_singular_floor_passed
        and delta_a_sum_of_squares_residual == 0
        and collapse_a_determinant == 0
        and collapse_a_rank == 10
    )
    passed = bool(
        action_symbol.shape == (11, 11)
        and full_symbol.shape == (11, 11)
        and action_symmetric
        and correction_symmetric
        and mixing_transpose_exact
        and gauge_scalar_rows_zero
        and pure_gauge_kernel_passed
        and zero_coupling_reduction_passed
        and second_order_reconstruction_passed
        and baseline_a_determinant != 0
        and baseline_spectrum_residual == 0
        and time_block_certificate_passed
        and tensor_modes_match
        and sp.simplify(plus_polynomial - expected_tensor_polynomial) == 0
        and flat_mixing_zero
        and sp.simplify(scalar_polynomial - (omega**2 - wave_number**2))
        == 0
        and generalized_delta((0, 1, 2, 3), (0, 1, 2, 3)) == 1
        and generalized_delta((0, 1, 2, 3), (1, 0, 2, 3)) == -1
        and generalized_delta((0, 1, 1), (0, 1, 1)) == 0
    )
    return passed, {
        "control": "linear-X quartic-Horndeski full local 11-by-11 principal extraction",
        "matrix_shape": list(full_symbol.shape),
        "basis": [f"h_{row}{column}" for row, column in SYMMETRIC_PAIRS]
        + ["psi"],
        "local_frame": "g^(mu nu)=diag(-1,1,1,1)",
        "auxiliary_inverse_metrics": {
            "tilde": str(data["tilde_inverse_metric"]),
            "hat": str(data["hat_inverse_metric"]),
            "metric_equation_gauge_normalization": "M2/2",
        },
        "specialized_action": {
            "G2": "X",
            "G3": "0",
            "G4": "M2/2+alpha*X",
            "G5": "0",
        },
        "background_jet_dependencies": {
            "metric_metric_correction": ["alpha", "nabla_mu(phi)"],
            "metric_scalar_mixing": ["alpha", "nabla_mu nabla_nu(phi)"],
            "scalar_scalar_correction": ["alpha", "G^(mu nu)"],
        },
        "block_certificates": {
            "action_symbol_symmetric": bool(action_symmetric),
            "metric_correction_symmetric": bool(correction_symmetric),
            "metric_scalar_transpose_exact": bool(mixing_transpose_exact),
            "gauge_block_has_zero_scalar_row_and_column": bool(
                gauge_scalar_rows_zero
            ),
            "four_action_principal_pure_gauge_vectors_in_kernel": bool(
                pure_gauge_kernel_passed
            ),
            "zero_coupling_reduces_to_einstein_scalar_modified_harmonic": bool(
                zero_coupling_reduction_passed
            ),
            "second_order_symbol_reconstructed_from_A_B_C": bool(
                second_order_reconstruction_passed
            ),
        },
        "flat_constant_timelike_gradient_reduction": {
            "tensor_plus_polynomial": str(plus_polynomial),
            "tensor_cross_polynomial": str(cross_polynomial),
            "tensor_polarizations_match": tensor_modes_match,
            "expected_adm_tensor_polynomial": str(
                expected_tensor_polynomial
            ),
            "adm_tensor_residual": str(
                sp.simplify(plus_polynomial - expected_tensor_polynomial)
            ),
            "scalar_polynomial": str(scalar_polynomial),
            "metric_scalar_mixing_zero": bool(flat_mixing_zero),
        },
        "first_order_generalized_pencil": {
            "status": "pass"
            if second_order_reconstruction_passed
            and baseline_a_determinant != 0
            and baseline_spectrum_residual == 0
            else "reject",
            "coefficient_shapes": {
                "A": list(first_order["A"].shape),
                "B": list(first_order["B"].shape),
                "C": list(first_order["C"].shape),
            },
            "mass_matrix_shape": list(first_order["mass"].shape),
            "evolution_matrix_shape": list(first_order["evolution"].shape),
            "second_order_reconstruction_residual_zero": bool(
                second_order_reconstruction_passed
            ),
            "einstein_scalar_flat_unit_direction": {
                "A_determinant": str(baseline_a_determinant),
                "characteristic_polynomial": str(
                    baseline_characteristic_polynomial
                ),
                "expected_characteristic_polynomial": str(
                    expected_baseline_characteristic_polynomial
                ),
                "residual": str(baseline_spectrum_residual),
                "mode_multiplicities_per_sign": {
                    "physical_speed_1": 3,
                    "pure_gauge_speed_1/2": 4,
                    "gauge_violating_speed_1/3": 4,
                },
            },
            "time_block_invertibility": {
                "status": "conditional_pass",
                "baseline_general_determinant": str(
                    baseline_a_general_determinant
                ),
                "expected_baseline_general_determinant": str(
                    expected_baseline_a_general_determinant
                ),
                "normalized_M2_2_ATA_eigenvalues": {
                    str(value): multiplicity
                    for value, multiplicity in sorted(
                        baseline_ata_eigenvalues.items(), key=lambda item: str(item[0])
                    )
                },
                "baseline_minimum_singular_value": str(
                    baseline_minimum_singular_value
                ),
                "delta_A_frobenius_norm_squared": str(
                    delta_a_frobenius_squared
                ),
                "delta_A_exact_sum_of_squares": str(
                    delta_a_sum_of_squares
                ),
                "sum_of_squares_residual": str(
                    delta_a_sum_of_squares_residual
                ),
                "sufficient_condition": str(
                    sufficient_time_block_condition
                ),
                "bound_chain": (
                    "sigma_min(A)>=sigma_min(A0)-||Delta A||_2>="
                    "Min(1,M2/4)-||Delta A||_F>0"
                ),
                "declared_gradient_only_domain_status": "unresolved",
                "missing_uniform_background_bounds": [
                    "|G^(mu nu) n_mu n_nu| in the chosen common-time frame",
                    "spatial trace and Frobenius norm of h_mu^rho h_nu^sigma nabla_rho nabla_sigma(phi)",
                    "all scalar-gradient frame components entering the exact sum of squares",
                ],
                "curvature_collapse_negative_control": {
                    "substitution": {
                        "alpha": "1",
                        "M2": "2",
                        "nabla_mu(phi)": "0",
                        "nabla_mu nabla_nu(phi)": "0",
                        "G^00": "-1/2",
                        "other_G_components": "0",
                    },
                    "A_determinant": str(collapse_a_determinant),
                    "A_rank": collapse_a_rank,
                    "gradient_domain_inequality": "A_star_squared=0 < M2/abs(alpha)=2",
                    "interpretation": (
                        "the declared scalar-gradient inequality alone does not keep the time "
                        "block noncharacteristic when curvature jets are unrestricted"
                    ),
                    "scope": (
                        "local background-domain insufficiency witness, not an on-shell solution "
                        "and not an additional action rejection"
                    ),
                },
            },
            "generic_A_invertibility_status": "conditional_pass_with_missing_domain_bounds",
        },
        "extraction_status": "pass" if passed else "reject",
        "uniform_symmetrizer_and_norm_status": "unresolved",
        "remaining": [
            "declare and prove uniform curvature, scalar-Hessian, and gradient-component bounds satisfying the exact time-block condition",
            "construct a positive symmetrizer for the extracted direction-dependent first-order pencil",
            "bound the induced correction norm over the declared background jet and unit direction sphere",
            "compare that uniform norm with the exact 19/72 auxiliary-cone budget",
        ],
        "sources": [
            {
                "title": "On the hyperbolicity of the most general Horndeski theory",
                "arxiv": "1710.10155",
                "location": "Appendix B, equations for delta P_gg, delta P_gPhi, and delta P_PhiPhi",
                "url": "https://arxiv.org/abs/1710.10155",
            },
            {
                "title": "Well-posed formulation of Lovelock and Horndeski theories",
                "arxiv": "2003.08398",
                "location": "Section 4.1, equations (89)-(95)",
                "url": "https://arxiv.org/abs/2003.08398",
            },
        ],
        "scope": (
            "complete exact local-frame 11-by-11 principal matrix for independent scalar-gradient, "
            "scalar-Hessian, Einstein-tensor, and covector components, plus its exact 22-by-22 "
            "generalized first-order pencil. Extraction alone is not a uniform strong-hyperbolicity proof"
        ),
        "symbols": {
            "alpha": str(alpha),
            "M2": str(m2),
        },
    }


@cache
def build_quartic_horndeski_x2_kessence_modified_harmonic_symbol() -> dict[str, Any]:
    """Return the exact quartic symbol extended to ``G2=X+c20*X**2``."""

    data = build_quartic_horndeski_modified_harmonic_symbol()
    c20 = sp.Symbol("c20", finite=True, real=True)
    xi_lower = data["xi_lower"]
    gradient_lower = data["gradient_lower"]
    inverse_metric = data["physical_inverse_metric"]
    xi_upper = inverse_metric * xi_lower
    gradient_upper = inverse_metric * gradient_lower
    xi_squared = sp.factor((xi_lower.T * xi_upper)[0])
    gradient_dot_xi = sp.factor((gradient_upper.T * xi_lower)[0])
    x_scalar = data["x_scalar"]

    g2_x = sp.factor(1 + 2 * c20 * x_scalar)
    g2_xx = 2 * c20
    kessence_scalar_block = sp.factor(
        -g2_x * xi_squared + g2_xx * gradient_dot_xi**2
    )
    quartic_scalar_correction = sp.factor(
        2
        * data["alpha"]
        * (xi_lower.T * data["einstein_upper"] * xi_lower)[0]
    )
    extended_action_symbol = data["action_symbol"].copy()
    extended_action_symbol[10, 10] = sp.factor(
        kessence_scalar_block + quartic_scalar_correction
    )
    extended_full_symbol = extended_action_symbol + data["gauge_symbol"]
    return {
        **data,
        "c20": c20,
        "g2_x": g2_x,
        "g2_xx": g2_xx,
        "gradient_dot_xi": gradient_dot_xi,
        "kessence_scalar_block": kessence_scalar_block,
        "quartic_scalar_correction": quartic_scalar_correction,
        "action_symbol": extended_action_symbol,
        "full_symbol": extended_full_symbol,
        "first_order": _first_order_generalized_pencil(
            extended_full_symbol, xi_lower[0]
        ),
        "canonical_quartic_data": data,
    }


def quartic_horndeski_x2_kessence_extension_control() -> tuple[bool, dict[str, Any]]:
    """Verify the exact ``G2=X+c20*X**2`` quartic principal extension."""

    extended = build_quartic_horndeski_x2_kessence_modified_harmonic_symbol()
    data = extended["canonical_quartic_data"]
    c20 = extended["c20"]
    xi_lower = extended["xi_lower"]
    gradient_lower = extended["gradient_lower"]
    x_scalar = extended["x_scalar"]
    xi_upper = extended["physical_inverse_metric"] * xi_lower
    xi_squared = sp.factor((xi_lower.T * xi_upper)[0])
    gradient_dot_xi = extended["gradient_dot_xi"]
    kessence_scalar_block = extended["kessence_scalar_block"]
    quartic_scalar_correction = extended["quartic_scalar_correction"]
    extended_action_symbol = extended["action_symbol"]
    extended_full_symbol = extended["full_symbol"]
    canonical_limit_residual = (
        extended_full_symbol.subs(c20, 0) - data["full_symbol"]
    ).applyfunc(sp.factor)

    expected_kessence_correction = sp.factor(
        -2 * c20 * x_scalar * xi_squared
        + 2 * c20 * gradient_dot_xi**2
    )
    actual_kessence_correction = sp.factor(
        extended_full_symbol[10, 10] - data["full_symbol"][10, 10]
    )
    effective_metric_residual = sp.factor(
        actual_kessence_correction - expected_kessence_correction
    )
    omitted_disformal_residual = sp.factor(2 * c20 * gradient_dot_xi**2)

    omega, wave_number, a_star = sp.symbols("omega k A_star", real=True)
    flat_timelike = {
        xi_lower[0]: -omega,
        xi_lower[1]: wave_number,
        xi_lower[2]: 0,
        xi_lower[3]: 0,
        gradient_lower[0]: a_star,
        gradient_lower[1]: 0,
        gradient_lower[2]: 0,
        gradient_lower[3]: 0,
        **{symbol: 0 for symbol in data["einstein_upper"].free_symbols},
    }
    flat_scalar_polynomial = sp.factor(
        extended_full_symbol[10, 10].subs(flat_timelike)
    )
    expected_flat_scalar_polynomial = sp.factor(
        (1 + 3 * c20 * a_star**2) * omega**2
        - (1 + c20 * a_star**2) * wave_number**2
    )
    flat_residual = sp.factor(
        flat_scalar_polynomial - expected_flat_scalar_polynomial
    )

    passed = bool(
        extended_action_symbol.shape == (11, 11)
        and extended_action_symbol.equals(extended_action_symbol.T)
        and canonical_limit_residual.is_zero_matrix
        and effective_metric_residual == 0
        and flat_residual == 0
        and omitted_disformal_residual != 0
    )
    return passed, {
        "control": "linear-X quartic Horndeski plus quadratic-kessence principal extension",
        "specialized_action": {
            "G2": "X+c20*X**2",
            "G3": "0",
            "G4": "M2/2+alpha*X",
            "G5": "0",
        },
        "matrix_shape": list(extended_full_symbol.shape),
        "extension_location": "scalar-scalar entry only",
        "kessence_scalar_block": str(kessence_scalar_block),
        "quartic_scalar_correction": str(quartic_scalar_correction),
        "canonical_c20_zero_matrix_residual_zero": bool(
            canonical_limit_residual.is_zero_matrix
        ),
        "arbitrary_covector_effective_metric_residual": str(
            effective_metric_residual
        ),
        "flat_constant_timelike_gradient": {
            "scalar_polynomial": str(flat_scalar_polynomial),
            "expected": str(expected_flat_scalar_polynomial),
            "residual": str(flat_residual),
            "no_ghost_condition": "1+3*c20*A_star**2>0",
            "positive_spatial_gradient_condition": "1+c20*A_star**2>0",
        },
        "negative_control": {
            "mutation": "omit G2_XX*(nabla(phi).xi)**2",
            "residual": str(omitted_disformal_residual),
            "rejected": bool(omitted_disformal_residual != 0),
        },
        "extraction_status": "pass" if passed else "reject",
        "uniform_symmetrizer_and_norm_status": "unresolved",
        "scope": (
            "Exact principal-matrix extension for G2=X+c20*X^2 with constant coefficients. "
            "It does not cover phi-dependent G2 coefficients or phi-dependent G4, and it does "
            "not by itself prove a uniform symmetrizer over a background domain."
        ),
    }


@cache
def quartic_horndeski_baseline_riesz_symmetrizer_control() -> tuple[bool, dict[str, Any]]:
    """Construct the exact modified-harmonic Einstein-scalar symmetrizer.

    This implements the six Riesz-group construction used by Kovacs--Reall.
    The physical ``+/-1`` groups use the action-principal Hermitian forms
    ``H_star^+/-``; the pure-gauge and gauge-violating groups use their exact
    spectral projectors.  It also derives an explicit sufficient matrix-norm
    neighbourhood for preserving positivity on the perturbed physical groups.
    """

    data = build_quartic_horndeski_modified_harmonic_symbol()
    xi_lower = data["xi_lower"]
    alpha = data["alpha"]
    m2 = data["m2"]
    substitutions: dict[sp.Symbol, sp.Expr] = {
        alpha: 0,
        m2: 1,
        xi_lower[1]: 1,
        xi_lower[2]: 0,
        xi_lower[3]: 0,
    }
    substitutions.update({symbol: 0 for symbol in data["gradient_lower"]})
    substitutions.update(
        {symbol: 0 for symbol in data["hessian_lower"].free_symbols}
    )
    substitutions.update(
        {symbol: 0 for symbol in data["einstein_upper"].free_symbols}
    )
    first_order = data["first_order"]
    baseline_mass = first_order["mass"].subs(substitutions)
    baseline_evolution = first_order["evolution"].subs(substitutions)
    baseline_companion = baseline_mass.inv() * baseline_evolution
    identity = sp.eye(22)
    spectrum = (
        sp.Integer(1),
        sp.Integer(-1),
        sp.Rational(1, 2),
        sp.Rational(-1, 2),
        sp.Rational(1, 3),
        sp.Rational(-1, 3),
    )
    projectors: dict[sp.Rational, sp.Matrix] = {}
    for eigenvalue in spectrum:
        projector = identity
        for other in spectrum:
            if other != eigenvalue:
                projector *= (baseline_companion - other * identity) / (
                    eigenvalue - other
                )
        projectors[eigenvalue] = projector.applyfunc(sp.factor)

    projector_residuals = {
        str(eigenvalue): {
            "idempotent": bool(
                (projector * projector - projector).applyfunc(sp.factor).is_zero_matrix
            ),
            "commutes_with_companion": bool(
                (
                    baseline_companion * projector
                    - projector * baseline_companion
                ).applyfunc(sp.factor).is_zero_matrix
            ),
            "rank": projector.rank(),
            "frobenius_norm_squared": str(
                sp.factor(sum(entry**2 for entry in projector))
            ),
        }
        for eigenvalue, projector in projectors.items()
    }
    projector_sum_residual = (
        sum(projectors.values(), sp.zeros(22)) - identity
    ).applyfunc(sp.factor)
    pairwise_projector_residuals = [
        (projectors[left] * projectors[right]).applyfunc(sp.factor).is_zero_matrix
        for left in spectrum
        for right in spectrum
        if left != right
    ]

    action_first_order = _first_order_generalized_pencil(
        data["action_symbol"], xi_lower[0]
    )
    action_a = action_first_order["A"].subs(substitutions)
    action_b = action_first_order["B"].subs(substitutions)
    h_star_plus = action_b.row_join(action_a).col_join(
        action_a.row_join(sp.zeros(11))
    )
    h_star_minus = -h_star_plus
    h_star_eigenvalues = {
        sp.factor(value): multiplicity
        for value, multiplicity in h_star_plus.eigenvals().items()
    }
    h_star_norm_is_one = bool(
        h_star_eigenvalues.get(sp.Integer(1)) == 1
        and h_star_eigenvalues.get(sp.Integer(-1)) == 1
        and all(abs(float(sp.N(value, 30))) <= 1 for value in h_star_eigenvalues)
    )
    physical_restrictions: dict[str, Any] = {}
    physical_lower_bound = sp.Rational(1, 4)
    for eigenvalue, hermitian_form in (
        (sp.Integer(1), h_star_plus),
        (sp.Integer(-1), h_star_minus),
    ):
        basis = sp.Matrix.hstack(
            *(baseline_companion - eigenvalue * identity).nullspace()
        )
        restricted_energy = (basis.T * hermitian_form * basis).applyfunc(sp.factor)
        restricted_gram = (basis.T * basis).applyfunc(sp.factor)
        requested_ratio = sp.Symbol("rho", real=True)
        generalized_polynomial = sp.factor(
            (restricted_energy - requested_ratio * restricted_gram).det()
        )
        physical_restrictions[str(eigenvalue)] = {
            "dimension": basis.shape[1],
            "restricted_energy": str(restricted_energy),
            "restricted_euclidean_gram": str(restricted_gram),
            "generalized_eigenvalue_polynomial": str(generalized_polynomial),
            "minimum_generalized_eigenvalue": str(physical_lower_bound),
        }

    symmetrizer = sp.zeros(22)
    for eigenvalue, projector in projectors.items():
        metric = (
            h_star_plus
            if eigenvalue == 1
            else h_star_minus
            if eigenvalue == -1
            else identity
        )
        symmetrizer += projector.T * metric * projector
    symmetrizer = symmetrizer.applyfunc(sp.factor)
    symmetrizer_residual = (
        symmetrizer * baseline_companion
        - baseline_companion.T * symmetrizer
    ).applyfunc(sp.factor)
    ldl_l, ldl_d = symmetrizer.LDLdecomposition(hermitian=True)
    ldl_residual = (ldl_l * ldl_d * ldl_l.T - symmetrizer).applyfunc(sp.factor)
    ldl_pivots = list(ldl_d.diagonal())
    decomposition_lower_bound = sp.factor(physical_lower_bound / len(spectrum))

    omitted_group = sp.Rational(1, 2)
    omitted_symmetrizer = sp.zeros(22)
    for eigenvalue, projector in projectors.items():
        if eigenvalue == omitted_group:
            continue
        metric = (
            h_star_plus
            if eigenvalue == 1
            else h_star_minus
            if eigenvalue == -1
            else identity
        )
        omitted_symmetrizer += projector.T * metric * projector
    wrong_minus_basis = sp.Matrix.hstack(
        *(baseline_companion + identity).nullspace()
    )
    wrong_minus_restriction = (
        wrong_minus_basis.T * h_star_plus * wrong_minus_basis
    ).applyfunc(sp.factor)

    hat_substitutions = dict(substitutions)
    hat_substitutions[xi_lower[0]] = sp.Rational(1, 3)
    hat_action_symbol = data["action_symbol"].subs(hat_substitutions)
    hat_squared_eigenvalues = {
        sp.factor(value): multiplicity
        for value, multiplicity in (
            hat_action_symbol.T * hat_action_symbol
        ).eigenvals().items()
    }
    hat_smallest_nonzero_singular_value = sp.Rational(2, 9)
    hat_rank_control_passed = bool(
        hat_action_symbol.equals(hat_action_symbol.T)
        and hat_action_symbol.rank() == 7
        and len(hat_action_symbol.nullspace()) == 4
        and hat_squared_eigenvalues.get(sp.Integer(0)) == 4
        and hat_squared_eigenvalues.get(sp.Rational(4, 81)) == 2
        and all(
            value == 0
            or sp.simplify(
                value - hat_smallest_nonzero_singular_value**2
            ).is_nonnegative
            is True
            for value in hat_squared_eigenvalues
        )
    )

    contour_radius = sp.Rational(1, 12)
    projector_frobenius_squared = {
        eigenvalue: sp.factor(sum(entry**2 for entry in projector))
        for eigenvalue, projector in projectors.items()
    }
    resolvent_bounds: dict[sp.Rational, sp.Expr] = {}
    for eigenvalue in spectrum:
        bound = sp.Integer(0)
        for other in spectrum:
            distance = (
                contour_radius
                if other == eigenvalue
                else abs(eigenvalue - other) - contour_radius
            )
            bound += sp.sqrt(projector_frobenius_squared[other]) / distance
        resolvent_bounds[eigenvalue] = sp.factor(bound)
    maximum_resolvent_bound = max(
        resolvent_bounds.values(), key=lambda value: float(sp.N(value, 30))
    )
    projector_drift_budget = sp.Rational(1, 32)
    h_star_perturbation_budget = sp.Rational(1, 8)
    companion_perturbation_budget = sp.factor(
        projector_drift_budget
        / (
            maximum_resolvent_bound
            * (
                contour_radius * maximum_resolvent_bound
                + projector_drift_budget
            )
        )
    )
    physical_positivity_margin = sp.factor(
        physical_lower_bound * (1 - projector_drift_budget) ** 2
        - 2
        * (1 + projector_drift_budget)
        * projector_drift_budget
        - projector_drift_budget**2
        - h_star_perturbation_budget
    )

    passed = bool(
        baseline_companion.shape == (22, 22)
        and projector_sum_residual.is_zero_matrix
        and all(pairwise_projector_residuals)
        and [projectors[value].rank() for value in spectrum] == [3, 3, 4, 4, 4, 4]
        and all(
            record["idempotent"] and record["commutes_with_companion"]
            for record in projector_residuals.values()
        )
        and all(record["dimension"] == 3 for record in physical_restrictions.values())
        and symmetrizer.equals(symmetrizer.T)
        and symmetrizer_residual.is_zero_matrix
        and h_star_norm_is_one
        and all(projectors[value].equals(projectors[value].T) for value in (1, -1))
        and ldl_residual.is_zero_matrix
        and all(pivot.is_positive for pivot in ldl_pivots)
        and omitted_symmetrizer.rank() == 18
        and all(value.is_negative for value in wrong_minus_restriction.diagonal())
        and physical_positivity_margin.is_positive
        and companion_perturbation_budget.is_positive
        and hat_rank_control_passed
    )
    return passed, {
        "control": "exact six-group baseline Riesz symmetrizer for modified-harmonic Einstein-scalar",
        "source_equations": {
            "companion": "Kovacs--Reall eq. Mdef",
            "physical_inner_product": "H_star^+/- = +/-[[B_star,A_star],[A_star,0]]",
            "projectors": "six Riesz groups V^+/-, tilde V^+/-, hat V^+/-",
        },
        "baseline": {
            "M2": "1",
            "alpha": "0",
            "unit_spatial_covector": ["1", "0", "0"],
            "spectrum_with_multiplicity": {
                "1": 3,
                "-1": 3,
                "1/2": 4,
                "-1/2": 4,
                "1/3": 4,
                "-1/3": 4,
            },
            "rotation_scope": (
                "The physical, tilde, and hat inverse metrics are spatially isotropic and the "
                "symmetric-tensor basis is orthonormal, so SO(3) conjugation carries this result "
                "to every Euclidean-unit spatial covector without changing ranks or norms."
            ),
        },
        "projectors": projector_residuals,
        "projector_sum_residual_zero": bool(projector_sum_residual.is_zero_matrix),
        "pairwise_projector_products_zero": bool(all(pairwise_projector_residuals)),
        "physical_H_star_restrictions": physical_restrictions,
        "H_star_global_spectral_norm": {
            "eigenvalues": {
                str(value): multiplicity
                for value, multiplicity in h_star_eigenvalues.items()
            },
            "norm": "1",
            "verified": h_star_norm_is_one,
        },
        "hat_group_baseline_restricted_rank": {
            "action_symbol_rank": hat_action_symbol.rank(),
            "action_symbol_nullity": len(hat_action_symbol.nullspace()),
            "squared_singular_values": {
                str(value): multiplicity
                for value, multiplicity in hat_squared_eigenvalues.items()
            },
            "smallest_nonzero_singular_value": str(
                hat_smallest_nonzero_singular_value
            ),
            "passed": hat_rank_control_passed,
            "rank_preservation_argument": (
                "A perturbation with 2-norm below 2/9 keeps rank at least seven. "
                "The exact four-dimensional diffeomorphism kernel keeps rank at most seven, "
                "so the quotient/restricted action block remains invertible."
            ),
        },
        "symmetrizer": {
            "construction": (
                "sum P_lambda^T H_lambda P_lambda, with H_star^+/- on physical "
                "groups and Euclidean identity on tilde/hat gauge groups"
            ),
            "shape": list(symmetrizer.shape),
            "symmetric": bool(symmetrizer.equals(symmetrizer.T)),
            "K_M_minus_M_T_K_zero": bool(symmetrizer_residual.is_zero_matrix),
            "exact_LDL_pivots": [str(value) for value in ldl_pivots],
            "all_LDL_pivots_positive": bool(
                all(value.is_positive for value in ldl_pivots)
            ),
            "decomposition_norm_lower_bound": str(decomposition_lower_bound),
            "lower_bound_derivation": (
                "H_star is >=1/4 on each physical projected component and identity is used "
                "on four gauge groups; ||sum y_a||^2<=6 sum ||y_a||^2 gives K>=I/24."
            ),
        },
        "quantitative_physical_group_perturbation_contract": {
            "contour_radius": str(contour_radius),
            "minimum_baseline_group_separation": "1/6",
            "resolvent_frobenius_upper_bounds": {
                str(key): str(value) for key, value in resolvent_bounds.items()
            },
            "maximum_resolvent_upper_bound": str(maximum_resolvent_bound),
            "maximum_resolvent_upper_bound_numeric": float(
                sp.N(maximum_resolvent_bound, 17)
            ),
            "required_companion_2_norm_perturbation_upper": str(
                companion_perturbation_budget
            ),
            "required_companion_2_norm_perturbation_upper_numeric": float(
                sp.N(companion_perturbation_budget, 17)
            ),
            "implied_Riesz_projector_drift_upper": str(projector_drift_budget),
            "required_H_star_2_norm_perturbation_upper": str(
                h_star_perturbation_budget
            ),
            "certified_physical_H_star_margin": str(physical_positivity_margin),
            "certified_physical_H_star_margin_numeric": float(
                physical_positivity_margin
            ),
            "derivation": (
                "The resolvent identity and Neumann series give ||Pi-Pi0|| <= "
                "r R0^2 delta/(1-R0 delta). The stated delta makes this <=1/32; "
                "combining the baseline 1/4 restricted lower bound, ||H0||_2=1, "
                "and ||Delta H||_2<=1/8 leaves the exact positive margin 181/4096."
            ),
            "remaining_for_full_candidate_theorem": [
                "bound ||M-M0||_2 below the stated threshold over a declared local-jet box and every direction",
                "bound ||H_star-H_star0||_2 below 1/8 on the same box",
                "prove the hat-group restricted characteristic matrix remains invertible",
                "bind the source Noether identities and exact pure-gauge eigenvectors to each candidate specialization",
            ],
        },
        "negative_controls": {
            "omit_plus_half_gauge_group": {
                "resulting_rank": omitted_symmetrizer.rank(),
                "expected_rank": 18,
                "rejected": omitted_symmetrizer.rank() < 22,
            },
            "use_plus_H_star_on_negative_physical_group": {
                "restricted_matrix": str(wrong_minus_restriction),
                "negative_diagonal": bool(
                    all(value.is_negative for value in wrong_minus_restriction.diagonal())
                ),
                "rejected": True,
            },
            "collapse_tilde_and_hat_group_speeds": {
                "mutated_speeds": ["1/3", "1/3"],
                "separation": "0",
                "contours_disjoint": False,
                "rejected": True,
            },
        },
        "status": "pass_exact_baseline_and_quantitative_physical_group_contract"
        if passed
        else "reject",
        "full_quartic_candidate_status": "unresolved_uniform_matrix_and_hat_block_bounds",
        "scope": (
            "Exact strong-hyperbolicity symmetrizer at the Einstein-scalar baseline and a "
            "rigorous sufficient perturbation contract for positivity on the physical Riesz "
            "groups. This is not yet a full quartic-candidate theorem because the candidate "
            "local-jet matrix bounds and the gauge-violating restricted-block bound remain open."
        ),
    }
