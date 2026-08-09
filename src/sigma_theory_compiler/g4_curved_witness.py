from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache
from typing import Any

import sympy as sp

DIMENSION = 4
SIGNATURE = (-1, 1, 1, 1)


@dataclass(frozen=True)
class FirstJet:
    """A scalar component and its four covariant derivatives at one normal-frame point."""

    value: Fraction
    derivative: tuple[Fraction, Fraction, Fraction, Fraction]

    @classmethod
    def constant(cls, value: int | Fraction | sp.Expr) -> FirstJet:
        normalized = value if isinstance(value, sp.Basic) else Fraction(value)
        return cls(normalized, (Fraction(0),) * DIMENSION)  # type: ignore[arg-type]

    def __add__(self, other: FirstJet | int | Fraction) -> FirstJet:
        other = _jet(other)
        return FirstJet(
            self.value + other.value,
            tuple(
                self.derivative[index] + other.derivative[index]
                for index in range(DIMENSION)
            ),
        )

    __radd__ = __add__

    def __neg__(self) -> FirstJet:
        return FirstJet(-self.value, tuple(-item for item in self.derivative))

    def __sub__(self, other: FirstJet | int | Fraction) -> FirstJet:
        return self + (-_jet(other))

    def __rsub__(self, other: FirstJet | int | Fraction) -> FirstJet:
        return _jet(other) - self

    def __mul__(self, other: FirstJet | int | Fraction) -> FirstJet:
        other = _jet(other)
        return FirstJet(
            self.value * other.value,
            tuple(
                self.derivative[index] * other.value
                + self.value * other.derivative[index]
                for index in range(DIMENSION)
            ),
        )

    __rmul__ = __mul__

    def __truediv__(self, other: int | Fraction) -> FirstJet:
        divisor = Fraction(other)
        return FirstJet(
            self.value / divisor,
            tuple(item / divisor for item in self.derivative),
        )


def _jet(value: FirstJet | int | Fraction | sp.Expr) -> FirstJet:
    return value if isinstance(value, FirstJet) else FirstJet.constant(value)


def _rational(seed: int, *indices: int) -> Fraction:
    if seed < 0:
        suffix = "_".join(str(index) for index in indices) or "value"
        return sp.Symbol(f"jet_m{abs(seed)}_{suffix}", real=True)  # type: ignore[return-value]
    accumulator = 17 * seed + 11
    for position, index in enumerate(indices, start=1):
        accumulator = (accumulator * 37 + (index + 3) * (13 + 2 * position)) % 1009
    numerator = accumulator % 11 - 5
    if numerator == 0:
        numerator = 1 if accumulator % 2 else -1
    denominator = 2 + (accumulator // 11) % 7
    return Fraction(numerator, denominator)


def _symmetric_metric_derivative(
    seed: int, first: int, second: int, derivative_indices: tuple[int, ...]
) -> Fraction:
    metric_pair = tuple(sorted((first, second)))
    derivatives = tuple(sorted(derivative_indices))
    return _rational(seed, *metric_pair, 9, *derivatives)


def _symmetric_scalar_derivative(seed: int, indices: tuple[int, ...]) -> Fraction:
    return _rational(seed + 101, *tuple(sorted(indices)))


def _gamma_derivative(
    seed: int, upper: int, first: int, second: int, derivative: int
) -> Fraction:
    total = Fraction(0)
    for contracted in range(DIMENSION):
        total += SIGNATURE[upper] * (1 if upper == contracted else 0) * (
            _symmetric_metric_derivative(
                seed, contracted, first, (second, derivative)
            )
            + _symmetric_metric_derivative(
                seed, contracted, second, (first, derivative)
            )
            - _symmetric_metric_derivative(
                seed, first, second, (contracted, derivative)
            )
        )
    return total / 2


def _gamma_second_derivative(
    seed: int,
    upper: int,
    first: int,
    second: int,
    derivative_one: int,
    derivative_two: int,
) -> Fraction:
    total = Fraction(0)
    for contracted in range(DIMENSION):
        total += SIGNATURE[upper] * (1 if upper == contracted else 0) * (
            _symmetric_metric_derivative(
                seed,
                contracted,
                first,
                (second, derivative_one, derivative_two),
            )
            + _symmetric_metric_derivative(
                seed,
                contracted,
                second,
                (first, derivative_one, derivative_two),
            )
            - _symmetric_metric_derivative(
                seed,
                first,
                second,
                (contracted, derivative_one, derivative_two),
            )
        )
    return total / 2


def _function_partial(
    coefficients: dict[tuple[int, int], Fraction],
    phi: Fraction,
    x_value: Fraction,
    phi_order: int,
    x_order: int,
) -> Fraction:
    total = Fraction(0)
    for (phi_power, x_power), coefficient in coefficients.items():
        if phi_power < phi_order or x_power < x_order:
            continue
        phi_factor = math.factorial(phi_power) // math.factorial(
            phi_power - phi_order
        )
        x_factor = math.factorial(x_power) // math.factorial(x_power - x_order)
        total += (
            coefficient
            * phi_factor
            * x_factor
            * phi ** (phi_power - phi_order)
            * x_value ** (x_power - x_order)
        )
    return total


def _build_witness(seed: int) -> dict[str, Any]:
    half = Fraction(1, 2)

    gamma_one = [
        [
            [
                [
                    _gamma_derivative(seed, upper, first, second, derivative)
                    for derivative in range(DIMENSION)
                ]
                for second in range(DIMENSION)
            ]
            for first in range(DIMENSION)
        ]
        for upper in range(DIMENSION)
    ]
    gamma_two = [
        [
            [
                [
                    [
                        _gamma_second_derivative(
                            seed,
                            upper,
                            first,
                            second,
                            derivative_one,
                            derivative_two,
                        )
                        for derivative_two in range(DIMENSION)
                    ]
                    for derivative_one in range(DIMENSION)
                ]
                for second in range(DIMENSION)
            ]
            for first in range(DIMENSION)
        ]
        for upper in range(DIMENSION)
    ]

    riemann_up = [
        [
            [
                [
                    gamma_one[upper][nu][lower][mu]
                    - gamma_one[upper][mu][lower][nu]
                    for nu in range(DIMENSION)
                ]
                for mu in range(DIMENSION)
            ]
            for lower in range(DIMENSION)
        ]
        for upper in range(DIMENSION)
    ]
    derivative_riemann_up = [
        [
            [
                [
                    [
                        gamma_two[upper][nu][lower][mu][derivative]
                        - gamma_two[upper][mu][lower][nu][derivative]
                        for nu in range(DIMENSION)
                    ]
                    for mu in range(DIMENSION)
                ]
                for lower in range(DIMENSION)
            ]
            for upper in range(DIMENSION)
        ]
        for derivative in range(DIMENSION)
    ]
    riemann = [
        [
            [
                [
                    FirstJet(
                        SIGNATURE[first] * riemann_up[first][second][third][fourth],
                        tuple(
                            SIGNATURE[first]
                            * derivative_riemann_up[derivative][first][second][third][fourth]
                            for derivative in range(DIMENSION)
                        ),
                    )
                    for fourth in range(DIMENSION)
                ]
                for third in range(DIMENSION)
            ]
            for second in range(DIMENSION)
        ]
        for first in range(DIMENSION)
    ]
    ricci = [
        [
            FirstJet(
                sum(
                    riemann_up[upper][first][upper][second]
                    for upper in range(DIMENSION)
                ),
                tuple(
                    sum(
                        derivative_riemann_up[derivative][upper][first][upper][second]
                        for upper in range(DIMENSION)
                    )
                    for derivative in range(DIMENSION)
                ),
            )
            for second in range(DIMENSION)
        ]
        for first in range(DIMENSION)
    ]
    curvature = sum(
        SIGNATURE[index] * ricci[index][index] for index in range(DIMENSION)
    )
    einstein = [
        [
            ricci[first][second]
            - (SIGNATURE[first] if first == second else 0) * curvature / 2
            for second in range(DIMENSION)
        ]
        for first in range(DIMENSION)
    ]

    p_values = [
        _symmetric_scalar_derivative(seed, (index,)) for index in range(DIMENSION)
    ]
    h_values = [
        [
            _symmetric_scalar_derivative(seed, (first, second))
            for second in range(DIMENSION)
        ]
        for first in range(DIMENSION)
    ]
    p = [
        FirstJet(
            p_values[index], tuple(h_values[derivative][index] for derivative in range(DIMENSION))
        )
        for index in range(DIMENSION)
    ]
    hessian = [
        [
            FirstJet(
                h_values[first][second],
                tuple(
                    _symmetric_scalar_derivative(seed, (derivative, first, second))
                    - sum(
                        gamma_one[upper][first][second][derivative] * p_values[upper]
                        for upper in range(DIMENSION)
                    )
                    for derivative in range(DIMENSION)
                ),
            )
            for second in range(DIMENSION)
        ]
        for first in range(DIMENSION)
    ]

    x_jet = -sum(
        SIGNATURE[index] * p[index] * p[index] for index in range(DIMENSION)
    ) * half
    # The derivative of q_mu is obtained by differentiating -p^a H_mu_a.
    q = [
        -sum(
            SIGNATURE[index] * p[index] * hessian[mu][index]
            for index in range(DIMENSION)
        )
        for mu in range(DIMENSION)
    ]
    theta = sum(
        SIGNATURE[index] * hessian[index][index] for index in range(DIMENSION)
    )
    hessian_squared = sum(
        SIGNATURE[first]
        * SIGNATURE[second]
        * hessian[first][second]
        * hessian[first][second]
        for first in range(DIMENSION)
        for second in range(DIMENSION)
    )
    hessian_difference = theta * theta - hessian_squared

    phi_value = _rational(seed + 211, 0)
    coefficients = {
        (phi_power, x_power): _rational(seed + 307, phi_power, x_power)
        for phi_power in range(4)
        for x_power in range(4 - phi_power)
    }

    @cache
    def function_jet(phi_order: int, x_order: int) -> FirstJet:
        value = _function_partial(
            coefficients,
            phi_value,
            x_jet.value,
            phi_order,
            x_order,
        )
        next_phi = _function_partial(
            coefficients,
            phi_value,
            x_jet.value,
            phi_order + 1,
            x_order,
        )
        next_x = _function_partial(
            coefficients,
            phi_value,
            x_jet.value,
            phi_order,
            x_order + 1,
        )
        return FirstJet(
            value,
            tuple(
                next_phi * p[index].value + next_x * q[index].value
                for index in range(DIMENSION)
            ),
        )

    function = function_jet(0, 0)
    function_phi = function_jet(1, 0)
    function_x = function_jet(0, 1)
    function_phiphi = function_jet(2, 0)
    function_phix = function_jet(1, 1)
    function_xx = function_jet(0, 2)
    grad_function_x = [
        function_phix * p[index] + function_xx * q[index]
        for index in range(DIMENSION)
    ]

    def raised_p(index: int) -> FirstJet:
        return SIGNATURE[index] * p[index]

    def raised_q(index: int) -> FirstJet:
        return SIGNATURE[index] * q[index]

    def raised_hessian(first: int, second: int) -> FirstJet:
        return SIGNATURE[first] * SIGNATURE[second] * hessian[first][second]

    def raised_ricci(first: int, second: int) -> FirstJet:
        return SIGNATURE[first] * SIGNATURE[second] * ricci[first][second]

    lagrangian_x = function_x * curvature + function_xx * hessian_difference
    current = []
    for mu in range(DIMENSION):
        value = -lagrangian_x * raised_p(mu)
        value += 2 * function_x * sum(
            raised_ricci(mu, nu) * p[nu] for nu in range(DIMENSION)
        )
        value -= 2 * function_xx * (
            theta * raised_q(mu)
            - sum(q[nu] * raised_hessian(mu, nu) for nu in range(DIMENSION))
        )
        value -= 2 * function_phix * (theta * raised_p(mu) + raised_q(mu))
        current.append(value)
    scalar_euler = function_phi * curvature + function_phix * hessian_difference
    scalar_euler -= sum(
        current[index].derivative[index] for index in range(DIMENSION)
    )

    hessian_pp = sum(
        raised_p(first) * raised_p(second) * hessian[first][second]
        for first in range(DIMENSION)
        for second in range(DIMENSION)
    )
    hessian_square_pp = sum(
        raised_p(first)
        * raised_p(second)
        * SIGNATURE[contracted]
        * hessian[first][contracted]
        * hessian[second][contracted]
        for first in range(DIMENSION)
        for second in range(DIMENSION)
        for contracted in range(DIMENSION)
    )
    grad_function_x_dot_p = sum(
        SIGNATURE[index] * grad_function_x[index] * p[index]
        for index in range(DIMENSION)
    )
    ricci_pp = sum(
        raised_p(first) * raised_p(second) * ricci[first][second]
        for first in range(DIMENSION)
        for second in range(DIMENSION)
    )

    metric_euler: list[list[FirstJet]] = []
    for mu in range(DIMENSION):
        row: list[FirstJet] = []
        for nu in range(DIMENSION):
            metric_entry = SIGNATURE[mu] if mu == nu else 0
            value = function * einstein[mu][nu]
            value -= function_x * curvature * p[mu] * p[nu] * half
            value -= function_xx * hessian_difference * p[mu] * p[nu] * half
            value -= function_x * theta * hessian[mu][nu]
            value += function_x * sum(
                SIGNATURE[index]
                * hessian[index][mu]
                * hessian[index][nu]
                for index in range(DIMENSION)
            )
            value += sum(
                SIGNATURE[index]
                * grad_function_x[index]
                * (
                    hessian[index][mu] * p[nu]
                    + hessian[index][nu] * p[mu]
                )
                for index in range(DIMENSION)
            )
            value -= grad_function_x_dot_p * hessian[mu][nu]
            value += metric_entry * (
                function_phi * theta
                - 2 * x_jet * function_phiphi
                - 2 * function_phix * hessian_pp
                + function_xx * hessian_square_pp
                + function_x * hessian_difference * half
            )
            value += function_x * sum(
                SIGNATURE[index]
                * (
                    ricci[index][mu] * p[nu]
                    + ricci[index][nu] * p[mu]
                )
                * p[index]
                for index in range(DIMENSION)
            )
            value -= theta * (
                grad_function_x[mu] * p[nu]
                + grad_function_x[nu] * p[mu]
            )
            value -= metric_entry * (function_x * ricci_pp - grad_function_x_dot_p * theta)
            value += function_x * sum(
                raised_p(first)
                * raised_p(second)
                * riemann[mu][first][nu][second]
                for first in range(DIMENSION)
                for second in range(DIMENSION)
            )
            value -= function_phi * hessian[mu][nu]
            value -= function_phiphi * p[mu] * p[nu]
            value += function_phix * sum(
                SIGNATURE[index]
                * p[index]
                * (
                    hessian[index][mu] * p[nu]
                    + hessian[index][nu] * p[mu]
                )
                for index in range(DIMENSION)
            )
            value -= function_xx * q[mu] * q[nu]
            row.append(value)
        metric_euler.append(row)

    symmetry_residuals = [
        metric_euler[first][second].value - metric_euler[second][first].value
        for first in range(DIMENSION)
        for second in range(first + 1, DIMENSION)
    ]
    noether_residuals = []
    for nu in range(DIMENSION):
        divergence = sum(
            SIGNATURE[mu] * metric_euler[mu][nu].derivative[mu]
            for mu in range(DIMENSION)
        )
        noether_residuals.append(2 * divergence + scalar_euler.value * p[nu].value)

    if seed < 0:
        noether_residuals = [sp.expand(item) for item in noether_residuals]
        symmetry_residuals = [sp.expand(item) for item in symmetry_residuals]

    algebraic_bianchi_residuals = [
        riemann[first][second][third][fourth].value
        + riemann[first][third][fourth][second].value
        + riemann[first][fourth][second][third].value
        for first in range(DIMENSION)
        for second in range(DIMENSION)
        for third in range(DIMENSION)
        for fourth in range(DIMENSION)
    ]
    differential_bianchi_residuals = [
        riemann[first][second][third][fourth].derivative[derivative]
        + riemann[second][derivative][third][fourth].derivative[first]
        + riemann[derivative][first][third][fourth].derivative[second]
        for derivative in range(DIMENSION)
        for first in range(DIMENSION)
        for second in range(DIMENSION)
        for third in range(DIMENSION)
        for fourth in range(DIMENSION)
    ]
    scalar_commutator_residuals = [
        hessian[mu][nu].derivative[derivative]
        - hessian[derivative][nu].derivative[mu]
        + sum(
            SIGNATURE[upper]
            * riemann[upper][nu][derivative][mu].value
            * p[upper].value
            for upper in range(DIMENSION)
        )
        for derivative in range(DIMENSION)
        for mu in range(DIMENSION)
        for nu in range(DIMENSION)
    ]
    if seed < 0:
        scalar_commutator_residuals = [
            sp.expand(item) for item in scalar_commutator_residuals
        ]
    contracted_bianchi_residuals = [
        sum(
            SIGNATURE[mu] * einstein[mu][nu].derivative[mu]
            for mu in range(DIMENSION)
        )
        for nu in range(DIMENSION)
    ]
    corrupted_residuals = []
    for nu in range(DIMENSION):
        omitted_divergence = sum(
            SIGNATURE[mu]
            * (function_xx * q[mu] * q[nu]).derivative[mu]
            for mu in range(DIMENSION)
        )
        corrupted_residuals.append(2 * omitted_divergence)

    weyl_nonzero = False
    for first in range(DIMENSION):
        for second in range(DIMENSION):
            for third in range(DIMENSION):
                for fourth in range(DIMENSION):
                    g_first_third = SIGNATURE[first] if first == third else 0
                    g_first_fourth = SIGNATURE[first] if first == fourth else 0
                    g_second_third = SIGNATURE[second] if second == third else 0
                    g_second_fourth = SIGNATURE[second] if second == fourth else 0
                    weyl = riemann[first][second][third][fourth].value
                    weyl -= (
                        g_first_third * ricci[fourth][second].value
                        - g_first_fourth * ricci[third][second].value
                        - g_second_third * ricci[fourth][first].value
                        + g_second_fourth * ricci[third][first].value
                    ) / 2
                    weyl += curvature.value * (
                        g_first_third * g_second_fourth
                        - g_first_fourth * g_second_third
                    ) / 6
                    weyl_nonzero = weyl_nonzero or weyl != 0

    return {
        "seed": seed,
        "curvature_nonzero": curvature.value != 0,
        "curvature_gradient_nonzero": any(item != 0 for item in curvature.derivative),
        "riemann_component_nonzero": any(
            riemann[first][second][third][fourth].value != 0
            for first in range(DIMENSION)
            for second in range(DIMENSION)
            for third in range(DIMENSION)
            for fourth in range(DIMENSION)
        ),
        "weyl_component_nonzero": weyl_nonzero,
        "algebraic_bianchi_residuals_zero": all(
            item == 0 for item in algebraic_bianchi_residuals
        ),
        "differential_bianchi_residuals_zero": all(
            item == 0 for item in differential_bianchi_residuals
        ),
        "scalar_hessian_commutator_residuals_zero": all(
            item == 0 for item in scalar_commutator_residuals
        ),
        "contracted_bianchi_residuals": [str(item) for item in contracted_bianchi_residuals],
        "metric_symmetry_residuals": [str(item) for item in symmetry_residuals],
        "combined_noether_residuals": [str(item) for item in noether_residuals],
        "omitted_G4_XX_q_mu_q_nu_residuals": [
            str(item) for item in corrupted_residuals
        ],
        "omitted_term_rejected": any(item != 0 for item in corrupted_residuals),
        "function_partial_values": {
            "G4_X": str(function_x.value),
            "G4_XX": str(function_xx.value),
            "G4_phiX": str(function_phix.value),
        },
    }


@lru_cache(maxsize=1)
def generic_g4_curved_rnc_witness_control() -> tuple[bool, dict[str, Any]]:
    """Run exact rational curved normal-frame witnesses for the full G4 identity."""

    witnesses = [_build_witness(seed) for seed in (3, 7, 11)]
    passed = all(
        witness["curvature_nonzero"]
        and witness["curvature_gradient_nonzero"]
        and witness["riemann_component_nonzero"]
        and witness["weyl_component_nonzero"]
        and witness["algebraic_bianchi_residuals_zero"]
        and witness["differential_bianchi_residuals_zero"]
        and witness["scalar_hessian_commutator_residuals_zero"]
        and witness["contracted_bianchi_residuals"] == ["0"] * DIMENSION
        and witness["metric_symmetry_residuals"] == ["0"] * 6
        and witness["combined_noether_residuals"] == ["0"] * DIMENSION
        and witness["omitted_term_rejected"]
        for witness in witnesses
    )
    return passed, {
        "status": "pass" if passed else "fail",
        "dimension": DIMENSION,
        "signature": "(-,+,+,+)",
        "witnesses": witnesses,
        "construction": (
            "exact rational second/third metric Taylor coefficients with vanishing first metric "
            "derivatives; Riemann, covariant Riemann derivative, scalar Hessian, and covariant "
            "third scalar jet are derived from commuting coordinate Taylor derivatives"
        ),
        "verified_identity": "2 nabla^mu H_mu_nu+E_phi nabla_nu(phi)=0",
        "source": "Kobayashi-Yamaguchi-Yokoyama 2011 equations B.4, B.8, and B.12",
        "source_url": "https://arxiv.org/abs/1105.5723",
        "scope": (
            "three exact rational, fully curved four-dimensional local witnesses with nonzero "
            "curvature and curvature gradients; this is strong falsification of the complete "
            "source-form tensor transcription, not a symbolic all-jet theorem"
        ),
        "generic_all_jet_theorem": (
            "proved_by_generic_g4_curved_symbolic_rnc_control"
        ),
    }


@lru_cache(maxsize=1)
def generic_g4_curved_symbolic_rnc_control() -> tuple[bool, dict[str, Any]]:
    """Prove the complete curved G4 identity as a symbolic local-jet polynomial."""

    symbolic = _build_witness(-10000)
    noether_residuals = symbolic["combined_noether_residuals"]
    symmetry_residuals = symbolic["metric_symmetry_residuals"]
    corrupted = symbolic["omitted_G4_XX_q_mu_q_nu_residuals"]
    corrupted_hashes = [
        hashlib.sha256(item.encode("utf-8")).hexdigest() for item in corrupted
    ]
    passed = (
        symmetry_residuals == ["0"] * 6
        and noether_residuals == ["0"] * DIMENSION
        and symbolic["algebraic_bianchi_residuals_zero"]
        and symbolic["differential_bianchi_residuals_zero"]
        and symbolic["scalar_hessian_commutator_residuals_zero"]
        and symbolic["contracted_bianchi_residuals"] == ["0"] * DIMENSION
        and symbolic["omitted_term_rejected"]
    )
    return passed, {
        "status": "pass" if passed else "fail",
        "dimension": DIMENSION,
        "signature": "(-,+,+,+)",
        "proof_kind": "exact symbolic Riemann-normal-coordinate polynomial identity",
        "independent_local_data": {
            "metric_second_taylor_coefficients": 100,
            "metric_third_taylor_coefficients": 200,
            "scalar_gradient_components": 4,
            "scalar_symmetric_hessian_components": 10,
            "scalar_symmetric_coordinate_third_derivatives": 20,
            "G4_local_taylor_coefficients_through_total_order_3": 10,
            "scalar_base_value": 1,
            "total_independent_symbols": 345,
        },
        "geometry_relations_derived_not_assumed": [
            "algebraic Riemann Bianchi identity",
            "differential Riemann Bianchi identity",
            "contracted Bianchi identity",
            "scalar-Hessian curvature commutator",
        ],
        "metric_symmetry_residuals": symmetry_residuals,
        "combined_noether_residuals": noether_residuals,
        "omitted_G4_XX_q_mu_q_nu_negative": {
            "rejected": symbolic["omitted_term_rejected"],
            "nonzero_expression_sha256": corrupted_hashes,
        },
        "verified_identity": "2 nabla^mu H_mu_nu+E_phi nabla_nu(phi)=0",
        "source": "Kobayashi-Yamaguchi-Yokoyama 2011 equations B.4, B.8, and B.12",
        "source_url": "https://arxiv.org/abs/1105.5723",
        "scope": (
            "exact all-local-jet polynomial theorem for the complete source-form curved "
            "nonlinear G4(phi,X) metric/scalar Noether identity in four dimensions; the metric "
            "Euler formula is source-bound and still awaits independent Cadabra variation"
        ),
        "independent_backend_metric_variation": "unresolved",
    }
