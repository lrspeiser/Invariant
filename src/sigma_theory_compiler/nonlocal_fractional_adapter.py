"""Auxiliary-field localization adapter for the nonlocal ``(-Box)^alpha`` operator arm.

``runs/gpu-baryonic-screen/v3-formal-ladder.json`` reports
``BLOCKED-AT-MATERIALIZATION`` for all 71 surviving screened-gravity families and names
``missing_adapter:nonlocal_fractional_operator`` as the blocker that gates every one of
them.  The blocker's own ``adapter_to_build`` text asks for

    "a nonlocal/fractional-operator action IR with its own principal symbol, Cauchy
    formulation, and positivity contract (auxiliary-field localization of (-Box)^alpha,
    or a spectral-representation adapter)"

and ``docs/PHYSICS_CONCEPT_LANGUAGE.md`` adds that an explicitly nonlocal effective action
"requires separate causality, initial-value, and auxiliary-field controls".  This module
supplies exactly that, and nothing more than that.

The construction
----------------

The lift's power arm is the static Green's function of ``(-Box)^alpha`` with
``alpha = 1 - t/2``.  Write ``alpha = n + beta`` with integer ``n >= 0`` and
``0 <= beta < 1``.

*Route A (exact).*  ``beta = 0``.  The operator is a local integer power.  ``n = 1`` is the
ordinary d'Alembertian and needs no auxiliary field at all; ``n = 2`` localizes exactly with
one auxiliary field via ``L = lambda (-Box) phi - lambda^2/2``, whose ``(phi, lambda)``
kinetic matrix is off-diagonal with eigenvalues ``+1`` and ``-1`` -- the Ostrogradsky ghost,
retained and tested rather than truncated.  ``(-Box)^-1`` (Deser-Woodard) localizes with one
auxiliary scalar ``psi`` obeying ``Box psi = phi`` plus its Lagrange multiplier, again with an
off-diagonal ``(psi, xi)`` kinetic matrix carrying one ghost eigenvalue.

*Route B (approximation).*  ``beta != 0``.  The Balakrishnan integral representation

    A^-beta = (sin(pi beta)/pi) Int_0^inf dlambda lambda^-beta (A + lambda)^-1,   0 < beta < 1

is discretized by an ``N``-point Gauss-Jacobi rule.  With ``lambda = mu (1+y)/(1-y)`` the
integrand becomes ``2 mu^(1-beta) (1-y)^(beta-1) (1+y)^-beta [A(1-y) + mu(1+y)]^-1``, whose
weight ``(1-y)^(beta-1) (1+y)^-beta`` is exactly the Gauss-Jacobi weight with parameters
``a = beta-1``, ``b = -beta``, and whose remaining factor is analytic on ``[-1, 1]``.  The rule
returns poles

    m_i^2 = mu (1+y_i)/(1-y_i),    c_i = 2 (sin(pi beta)/pi) mu^(1-beta) w_i/(1-y_i)

so ``A^-beta ~= sum_i c_i (A + m_i^2)^-1``: a finite sum of ordinary massive propagators, each
of which localizes with one canonically normalized auxiliary scalar of mass ``m_i``.  For
``beta = 1/2`` the weight is ``(1-y^2)^-1/2`` exactly, so the rule is Gauss-Chebyshev and the
nodes and weights are closed-form algebraic numbers ``y_i = cos((2i-1) pi / 2N)``,
``w_i = pi/N``.  Sixty-seven of the seventy-one families land there.

The full propagator ``(k^2)^-alpha = (k^2)^-n (k^2)^-beta`` is then expanded in partial
fractions.  For ``n = 1`` this is the crux of the whole exercise:

    (k^2)^-1 (k^2 + m_i^2)^-1 = (1/m_i^2) [ (k^2)^-1 - (k^2 + m_i^2)^-1 ]

so the massless pole carries residue ``+sum_i c_i/m_i^2`` and *every* massive pole carries a
strictly negative residue.  The residues sum to zero exactly, at every ``N``, which is the
discrete form of the Kallen-Lehmann statement that a propagator falling faster than ``1/k^2``
cannot have a positive spectral density.  The ghost is therefore a property of the declared
nonlocal operator, not an artifact of the quadrature.

Honesty rules, enforced structurally
------------------------------------

1. **The finite-pole localization is an approximation of the nonlocal theory, and the
   verdicts it yields are verdicts about the approximant.**  ``CLAIMS`` carries
   ``approximation_is_not_the_nonlocal_theory: true`` and no receipt can be written without
   it.
2. **Every verdict is a convergence study.**  Each family is decided at every ``N`` in
   ``DECLARED_POLE_COUNTS``.  A verdict that flips with ``N`` is reported as
   ``UNSTABLE_UNDER_LOCALIZATION``, never as a result; a verdict that holds at every ``N`` is
   reported as a conditional verdict carrying the exact sentence "holds for every N-pole
   localization tested, N in {...}; the nonlocal limit is not proved".  ``validate_receipt``
   refuses a receipt that claims stability over fewer than the declared ``N`` set.
3. **A rung this repository still cannot execute stays a typed blocker.**  Discharging the
   nonlocal blocker does not discharge the AQUAL inversion, the direct scalar-matter coupling
   the frozen field contract forbids, the cubic-``G3`` weak-field cone, or the UV form factor
   of the same arm.  They travel with every verdict in ``RESIDUAL_BLOCKERS``.
4. **Two kinds of ghost are distinguished and never conflated.**  A
   ``propagator_residue_ghost`` is a negative residue in the two-point function itself and is
   forced by the sum rule at every ``N``.  An ``auxiliary_field_ghost`` is a wrong-sign kinetic
   eigenvalue of the *localized formulation*; the Deser-Woodard ``1/Box`` control is exactly
   that case and is flagged ``localization_artifact_possible: true``.
5. **Controls must fire or the run aborts.**  A local operator through the adapter must
   reproduce the un-adapted ladder verdict rung for rung; ``1/Box`` must reproduce the known
   Deser-Woodard one-auxiliary-field ghost; a deliberately ghost-laden operator must reject;
   and a declared spectral subtraction whose sign flips between ``N = 4`` and ``N = 8`` must be
   classified ``UNSTABLE_UNDER_LOCALIZATION``.

Reused machinery (nothing here re-derives what the tree already proves):

``v3_family_formal_ladder.run_ladder`` / ``ladder_verdict`` / ``sector_verdict``
    The five ladder rungs on the declared screening sector, on the same interval-certified
    on-shell FLRW trajectory, with the same known-answer control actions.
``principal_symbol.analyze_isotropic_second_order_symbol``
    The ghost / gradient / real-characteristic / strong-hyperbolicity / cone decision, applied
    here to the localized multi-field kinetic and gradient blocks.
``action_ir.compile_action_spec``
    The real covariant grammar, used to record exactly which part of the localized tower the
    frozen grammar admits and which part needs an operator decision.

Nothing in this module is a physical validation, and a ``STABLE_PASS`` admits nothing: the
complete lift of every family remains blocked on arms this adapter does not touch.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from . import v3_family_formal_ladder as ladder
from .action_ir import compile_action_spec, load_action_grammar
from .formal_backend import load_field_contract
from .principal_symbol import analyze_isotropic_second_order_symbol
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-nonlocal-fractional-localization-result-1.0"
ACTION_IR_SCHEMA = "sigma-nonlocal-localized-action-ir-1.0"

#: The blocker this adapter discharges, verbatim from the ladder receipt it consumes.
DISCHARGED_BLOCKER = "missing_adapter:nonlocal_fractional_operator"

LADDER_RECEIPT_PATH = "runs/gpu-baryonic-screen/v3-formal-ladder.json"
REPRESENTATIVES_PATH = "runs/gpu-baryonic-screen/v3-family-representatives.json"
RECEIPT_PATH = "runs/gpu-baryonic-screen/nonlocal-localization-v1.json"
ACTION_GRAMMAR_PATH = "configs/covariant_action_grammar.json"
FIELD_CONTRACT_PATH = "configs/covariant_field_contract.json"

#: The declared convergence study.  Frozen: a stability claim is only meaningful relative to
#: the pole counts it was tested at, so ``validate_receipt`` refuses any receipt that claims
#: STABLE_* over a smaller set than this one.
DECLARED_POLE_COUNTS: tuple[int, ...] = (2, 4, 8, 16)

#: Working precision for the Gauss-Jacobi solve, and the precision the receipt emits.  The
#: emitted precision is deliberately far below the working precision so that a last-digit
#: wobble in the root finder can never reach a receipt and break determinism.
WORKING_PRECISION_DIGITS = 60
EMITTED_PRECISION_DIGITS = 30

#: The declared test momenta for the convergence study, as multiples of the reference mass
#: scale ``mu``.  ``k^2 = mu`` is the point at which the quadrature is exact for every ``N``.
DECLARED_TEST_MOMENTA: tuple[str, ...] = ("1/100", "1", "100")

#: Claims block.  Frozen; any change changes the receipt hash and therefore the claim.
CLAIMS: dict[str, bool] = {
    "approximation_is_not_the_nonlocal_theory": True,
    "corpus_absence_establishes_novelty": False,
    "first_principles_derivation_pending": True,
    "formal_pass_is_not_physical_validation": True,
    "real_data_used": False,
    "synthetic_controls_only": True,
}

#: What this adapter does *not* discharge.  Attached to every single verdict so that a
#: STABLE_PASS can never be misread as an admission of the family.
RESIDUAL_BLOCKERS: tuple[str, ...] = (
    "missing_adapter:aqual_nu_to_kessence_inversion",
    "missing_adapter:direct_scalar_matter_coupling",
    "missing_adapter:uv_form_factor_operator",
    "missing_adapter:auxiliary_tower_exceeds_the_frozen_covariant_grammar_field_bound",
    "nonlocal_limit_of_the_finite_pole_localization_unproved",
    "retarded_branch_of_the_fractional_operator_undetermined",
    "global_positive_energy_on_general_nonmaximal_data_unresolved",
    "arbitrary_inhomogeneous_background_principal_symbol_unresolved",
    "first_principles_derivation_of_the_lift_pending",
    "background_domain_forward_invariance_under_nonlinear_evolution_unresolved",
)

#: Typed blockers this module can itself emit.  Each names the adapter that must be built.
BLOCKERS: dict[str, dict[str, str]] = {
    "missing_adapter:integer_order_localization_above_quadratic": {
        "why": (
            "an integer power (-Box)^n with n >= 3 needs an n-step auxiliary chain whose "
            "kinetic block this module does not construct; guessing the chain would invent "
            "the Ostrogradsky mode count rather than derive it"
        ),
        "adapter_to_build": (
            "an explicit n-step auxiliary chain for (-Box)^n with its own kinetic block and "
            "an Ostrogradsky mode-count certificate"
        ),
    },
    "missing_adapter:cubic_g3_uniform_weak_field_cone": {
        "why": (
            "inherited unchanged from the v3 formal ladder: with canonical G3 != 0 the "
            "screening sector leaves the generalized-harmonic k-essence class and the FLRW "
            "certifier returns modified_harmonic_uniform_bound_required"
        ),
        "adapter_to_build": (
            "a candidate-specific uniform weak-field threshold and common scalar/metric cone "
            "bound for the declared braiding coefficient, or a modified-harmonic symmetrizer "
            "for cubic G3"
        ),
    },
}

#: The ladder, in the same order and with the same names the v3 ladder uses, so that the two
#: receipts' rung breakdowns are directly comparable.
LADDER_RUNGS: tuple[str, ...] = tuple(name for name, _ in ladder.LADDER_RUNGS)

#: Localization routes.
ROUTE_LOCAL = "A_integer_order_exact"
ROUTE_INVERSE = "A_inverse_dalembertian_deser_woodard"
ROUTE_QUADRATURE = "B_balakrishnan_finite_pole_quadrature"

#: Ghost kinds.  The distinction is the scientific point of the module and is never elided.
GHOST_NONE = "none"
GHOST_PROPAGATOR = "propagator_residue_ghost"
GHOST_AUXILIARY = "auxiliary_field_ghost"


class NonlocalFractionalAdapterError(ValueError):
    """Raised on malformed input, a broken known-answer control, or receipt tamper."""


# ---------------------------------------------------------------------------
# Exact helpers.  No float ever reaches a receipt.
# ---------------------------------------------------------------------------


def _decimal(value: Any, digits: int = EMITTED_PRECISION_DIGITS) -> str:
    """Render any exact quantity as a decimal string at the declared emitted precision."""

    with mp.workdps(WORKING_PRECISION_DIGITS):
        if isinstance(value, mp.mpf):
            number = +value
        else:
            number = mp.mpf(str(sp.N(sp.sympify(value), WORKING_PRECISION_DIGITS)))
        return mp.nstr(number, digits, strip_zeros=False)


def _exact(expression: Any) -> str:
    """Render an exact algebraic quantity as its sympy text form.

    A value that already carries floating-point atoms -- the generic Gauss-Jacobi nodes, whose
    exact specification is the declared Jacobi polynomial rather than a radical -- is emitted
    at the declared emitted precision instead, so that a working-precision tail can never make
    a receipt nondeterministic.
    """

    parsed = sp.sympify(expression)
    if parsed.atoms(sp.Float):
        return _decimal(parsed)
    return str(parsed)


def _outward_bound(value: Any) -> str:
    """An outward-rounded power-of-ten upper bound on ``|value|``.

    Round-off residuals are real evidence but their trailing digits are not: emitting thirty
    significant figures of arithmetic noise would make a receipt look precise where it is not,
    and would make it hostage to the last bit of a root finder.  A power-of-ten bound rounded
    away from zero says exactly as much as the number actually supports.
    """

    with mp.workdps(WORKING_PRECISION_DIGITS):
        magnitude = abs(mp.mpf(str(sp.N(sp.sympify(value), WORKING_PRECISION_DIGITS))))
        if magnitude == 0:
            return "0"
        return "1e" + str(int(mp.ceil(mp.log10(magnitude))))


def _no_floats(value: Any, path: str = "$") -> None:
    """Fail closed if any float survives into a receipt."""

    if isinstance(value, float):
        raise NonlocalFractionalAdapterError(f"float in receipt at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _no_floats(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _no_floats(item, f"{path}[{index}]")


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(body), "content_sha256": canonical_sha256(body)}


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


# ---------------------------------------------------------------------------
# Inputs: the sealed ladder receipt whose blocker this adapter discharges.
# ---------------------------------------------------------------------------


def load_ladder_receipt(root: str | Path) -> dict[str, Any]:
    """Load ``v3-formal-ladder.json``, verify its seal, and verify that it names our blocker."""

    path = Path(root) / LADDER_RECEIPT_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("schema_version") != ladder.RESULT_SCHEMA:
        raise NonlocalFractionalAdapterError("unexpected v3 formal-ladder schema")
    if value.get("content_sha256") != canonical_sha256(body):
        raise NonlocalFractionalAdapterError("v3 formal-ladder receipt seal does not replay")
    if DISCHARGED_BLOCKER not in value["config"]["blockers"]:
        raise NonlocalFractionalAdapterError("the ladder receipt does not name this blocker")
    blocked = value["counts"]["blocked_by_adapter"].get(DISCHARGED_BLOCKER, 0)
    if blocked != int(value["counts"]["families_in"]):
        raise NonlocalFractionalAdapterError(
            "the ladder receipt does not block every family on this adapter"
        )
    return value


def load_families(root: str | Path, receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load the 71 representatives and bind them to the ladder receipt's own family list."""

    path = Path(root) / REPRESENTATIVES_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise NonlocalFractionalAdapterError("representative artifact seal does not replay")
    families = sorted(value["families"], key=lambda item: int(item["representative_ordinal"]))
    ordinals = [int(item["representative_ordinal"]) for item in families]
    ladder_ordinals = [int(item["representative_ordinal"]) for item in receipt["families"]]
    if ordinals != sorted(ladder_ordinals):
        raise NonlocalFractionalAdapterError(
            "the representative set and the ladder receipt disagree on the family list"
        )
    return families


_ALPHA_PATTERN = re.compile(r"alpha\s*=\s*1\s*-\s*t/2\s*=\s*([^\s;]+)\s*;")


def read_declared_alpha(family: Mapping[str, Any]) -> tuple[Fraction, dict[str, Any]]:
    """Read ``alpha`` off the lift's own text and cross-check it against ``1 - t/2``.

    The exponent is never invented here: it is parsed out of the ``field_theory_ansatz`` the
    screen itself emitted, and the parse must agree with the declared kernel axis ``t``.  A
    disagreement is a fail-closed error, not a silent preference for one of the two.
    """

    components = [
        component
        for component in family["covariant_lift_candidate"]["components"]
        if component["mechanism"] == "nonlocal_propagator_correction"
    ]
    if len(components) != 1:
        raise NonlocalFractionalAdapterError(
            "every v3 lift must carry exactly one nonlocal_propagator_correction component"
        )
    component = components[0]
    match = _ALPHA_PATTERN.search(str(component["field_theory_ansatz"]))
    if match is None:
        raise NonlocalFractionalAdapterError("the nonlocal component does not declare alpha")
    declared = Fraction(match.group(1))
    axis = Fraction(1) - Fraction(str(family["representative_values"]["t"])) / 2
    if declared != axis:
        raise NonlocalFractionalAdapterError(
            f"declared alpha {declared} disagrees with 1 - t/2 = {axis}"
        )
    return declared, _plain(component)


# ---------------------------------------------------------------------------
# The quadrature.  Exact where the weight is Chebyshev, algebraic elsewhere.
# ---------------------------------------------------------------------------


def _jacobi_moment(beta: Fraction, order: int) -> mp.mpf:
    """``Int_-1^1 (1-y)^(beta-1) (1+y)^-beta y^k dy`` in closed Beta-function form."""

    exponent = mp.mpf(beta.numerator) / beta.denominator
    total = mp.mpf(0)
    for index in range(order + 1):
        total += (
            mp.binomial(order, index)
            * mp.mpf(2) ** index
            * (-1) ** (order - index)
            * mp.beta(1 - exponent + index, exponent)
        )
    return total


def gauss_jacobi_rule(beta: Fraction, pole_count: int) -> dict[str, Any]:
    """Nodes and weights of the ``N``-point Gauss-Jacobi rule for weight ``(1-y)^(b-1)(1+y)^-b``.

    ``beta = 1/2`` gives the Chebyshev weight ``(1-y^2)^-1/2`` exactly, so nodes and weights are
    emitted as closed-form algebraic numbers.  For any other ``beta`` the nodes are the roots of
    the exact-rational Jacobi polynomial ``P_N^(beta-1, -beta)``, which is emitted alongside the
    decimal values so that the algebraic specification travels with the approximation.  The
    weights are solved from the exact moment equations and are then *verified* against every
    moment up to degree ``2N-1``; a rule that fails its own moment check aborts the run.
    """

    if not 0 < beta < 1:
        raise NonlocalFractionalAdapterError("the Gauss-Jacobi route needs 0 < beta < 1")
    if pole_count < 1:
        raise NonlocalFractionalAdapterError("a localization needs at least one pole")
    symbol = sp.Symbol("y")
    if beta == Fraction(1, 2):
        exact_nodes = [
            sp.cos(sp.pi * (2 * index - 1) / (2 * pole_count))
            for index in range(1, pole_count + 1)
        ]
        exact_weights = [sp.pi / pole_count] * pole_count
        exactness = "closed_form_gauss_chebyshev"
        minimal_polynomial = str(sp.expand(sp.chebyshevt(pole_count, symbol)))
    else:
        exact_nodes = []
        exact_weights = []
        exactness = "algebraic_root_of_the_declared_jacobi_polynomial"
        minimal_polynomial = str(
            sp.expand(
                sp.jacobi_poly(
                    pole_count,
                    sp.Rational(beta.numerator, beta.denominator) - 1,
                    -sp.Rational(beta.numerator, beta.denominator),
                    symbol,
                )
            )
        )
    with mp.workdps(WORKING_PRECISION_DIGITS):
        if exact_nodes:
            nodes = [mp.mpf(str(sp.N(node, WORKING_PRECISION_DIGITS))) for node in exact_nodes]
            weights = [
                mp.mpf(str(sp.N(weight, WORKING_PRECISION_DIGITS))) for weight in exact_weights
            ]
        else:
            polynomial = sp.Poly(sp.sympify(minimal_polynomial), symbol)
            coefficients = [
                mp.mpf(int(sp.Rational(item).p)) / int(sp.Rational(item).q)
                for item in polynomial.all_coeffs()
            ]
            roots = mp.polyroots(coefficients, maxsteps=200, extraprec=400)
            nodes = sorted(mp.re(root) for root in roots)
            matrix = mp.matrix(pole_count, pole_count)
            for order in range(pole_count):
                for column in range(pole_count):
                    matrix[order, column] = nodes[column] ** order
            solved = mp.lu_solve(
                matrix, mp.matrix([_jacobi_moment(beta, order) for order in range(pole_count)])
            )
            weights = [solved[index] for index in range(pole_count)]
        residual = mp.mpf(0)
        for order in range(2 * pole_count):
            quadrature = mp.fsum(
                weight * node**order for weight, node in zip(weights, nodes, strict=True)
            )
            residual = max(residual, abs(quadrature - _jacobi_moment(beta, order)))
        tolerance = mp.mpf(10) ** (-(WORKING_PRECISION_DIGITS - 20))
        if residual > tolerance:
            raise NonlocalFractionalAdapterError(
                "the Gauss-Jacobi rule failed its own moment control"
            )
        if any(weight <= 0 for weight in weights):
            raise NonlocalFractionalAdapterError("a Gauss-Jacobi weight is not positive")
        if any(not -1 < node < 1 for node in nodes):
            raise NonlocalFractionalAdapterError("a Gauss-Jacobi node left the interval")
    return {
        "pole_count": pole_count,
        "beta": str(beta),
        "weight_function": "(1-y)^(beta-1) (1+y)^(-beta)",
        "exactness": exactness,
        "minimal_polynomial": minimal_polynomial,
        "precision_digits": str(EMITTED_PRECISION_DIGITS),
        "moment_residual_bound_up_to_degree_2N_minus_1": _outward_bound(residual),
        "exact_nodes": [_exact(node) for node in exact_nodes],
        "exact_weights": [_exact(weight) for weight in exact_weights],
        "_nodes": nodes,
        "_weights": weights,
        "_exact_nodes": exact_nodes,
        "_exact_weights": exact_weights,
    }


# ---------------------------------------------------------------------------
# The localization itself.
# ---------------------------------------------------------------------------


def _split_alpha(alpha: Fraction) -> tuple[int, Fraction]:
    integer_part = alpha.numerator // alpha.denominator
    return integer_part, alpha - integer_part


def localize(
    alpha: Fraction,
    reference_mass_squared: Fraction,
    amplitude: Fraction,
    pole_count: int,
    *,
    spectral_subtraction: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Localize ``amplitude * (-Box)^-alpha`` into ``pole_count`` massive auxiliary scalars.

    ``reference_mass_squared`` is ``mu``: the mass-squared scale the quadrature is centred on,
    taken from the declared kernel's own UV form-factor scale ``mu = 1/L2^2``.  The rule is
    exact at ``k^2 = mu`` for every ``N``, which is the module's cheapest known-answer identity.

    ``spectral_subtraction`` is control-only.  It subtracts a declared weight from the residue
    of the pole nearest a declared mass, which is how a sign-indefinite spectral density is
    fed through the same machinery to exercise the ``UNSTABLE_UNDER_LOCALIZATION`` classifier.
    """

    if amplitude <= 0:
        raise NonlocalFractionalAdapterError("the arm amplitude must be positive")
    if reference_mass_squared <= 0:
        raise NonlocalFractionalAdapterError("the reference mass scale must be positive")
    integer_part, beta = _split_alpha(alpha)
    mu = sp.Rational(reference_mass_squared.numerator, reference_mass_squared.denominator)
    weight = sp.Rational(amplitude.numerator, amplitude.denominator)

    if alpha == -1:
        return _localize_inverse_dalembertian(alpha, mu, weight, pole_count)
    if beta == 0:
        return _localize_integer(alpha, integer_part, mu, weight, pole_count)
    if integer_part < 0 or integer_part > 1:
        return {
            "route": ROUTE_QUADRATURE,
            "status": "blocked",
            "blocker": "missing_adapter:integer_order_localization_above_quadratic",
            "alpha": str(alpha),
            "integer_part": str(integer_part),
            "fractional_part": str(beta),
            "pole_count": str(pole_count),
        }
    return _localize_quadrature(
        alpha,
        integer_part,
        beta,
        mu,
        weight,
        pole_count,
        spectral_subtraction=spectral_subtraction,
    )


def _mode(
    identifier: str,
    kinetic_sign: int,
    mass_squared_exact: Any,
    residue_exact: Any,
    *,
    origin: str,
) -> dict[str, Any]:
    residue = sp.sympify(residue_exact)
    magnitude = residue if kinetic_sign > 0 else -residue
    coupling = sp.sqrt(magnitude)
    return {
        "id": identifier,
        "origin": origin,
        "kinetic_sign": str(kinetic_sign),
        "mass_squared_exact": _exact(mass_squared_exact),
        "mass_squared_decimal": _decimal(mass_squared_exact),
        "propagator_residue_exact": _exact(residue),
        "propagator_residue_decimal": _decimal(residue),
        "residue_sign": "+" if kinetic_sign > 0 else "-",
        "source_coupling_exact": _exact(coupling),
        "source_coupling_decimal": _decimal(coupling),
    }


def _localize_integer(
    alpha: Fraction, integer_part: int, mu: sp.Expr, weight: sp.Expr, pole_count: int
) -> dict[str, Any]:
    """``(-Box)^n`` for integer ``n >= 1``: exactly local, no quadrature, no approximation."""

    if integer_part == 1:
        modes = [_mode("chi_0", 1, 0, weight, origin="massless_pole_of_the_local_dalembertian")]
        undiagonalized = None
        ghost_kind = GHOST_NONE
        artifact = False
        note = (
            "alpha = 1 is the ordinary d'Alembertian: the operator is already local, the "
            "adapter introduces no auxiliary field, and the localization is exact"
        )
    elif integer_part == 2:
        modes = [
            _mode("chi_0", 1, 0, weight, origin="higher_derivative_pair_positive_eigenvalue"),
            _mode(
                "lambda_1",
                -1,
                0,
                -weight,
                origin="higher_derivative_pair_ostrogradsky_eigenvalue",
            ),
        ]
        undiagonalized = sp.Matrix([[0, 1], [1, 0]])
        ghost_kind = GHOST_AUXILIARY
        artifact = False
        note = (
            "(-Box)^2 is local and higher derivative.  L = lambda (-Box) phi - lambda^2/2 is "
            "its exact one-auxiliary-field localization; the (phi, lambda) kinetic matrix is "
            "off-diagonal with eigenvalues +1 and -1, which is the Ostrogradsky ghost of the "
            "original higher-derivative theory rather than an artifact of the localization"
        )
    else:
        return {
            "route": ROUTE_LOCAL,
            "status": "blocked",
            "blocker": "missing_adapter:integer_order_localization_above_quadratic",
            "alpha": str(alpha),
            "integer_part": str(integer_part),
            "fractional_part": "0",
            "pole_count": str(pole_count),
        }
    return _finish(
        route=ROUTE_LOCAL,
        alpha=alpha,
        integer_part=integer_part,
        beta=Fraction(0),
        mu=mu,
        weight=weight,
        pole_count=pole_count,
        modes=modes,
        undiagonalized=undiagonalized,
        ghost_kind=ghost_kind,
        artifact=artifact,
        note=note,
        rule=None,
        approximate=False,
    )


def _localize_inverse_dalembertian(
    alpha: Fraction, mu: sp.Expr, weight: sp.Expr, pole_count: int
) -> dict[str, Any]:
    """``(-Box)^-1``: the Deser-Woodard localization, exact, one auxiliary scalar."""

    modes = [
        _mode("psi", 1, 0, weight, origin="deser_woodard_auxiliary_scalar_box_psi_equals_phi"),
        _mode("xi", -1, 0, -weight, origin="deser_woodard_lagrange_multiplier"),
    ]
    return _finish(
        route=ROUTE_INVERSE,
        alpha=alpha,
        integer_part=-1,
        beta=Fraction(0),
        mu=mu,
        weight=weight,
        pole_count=pole_count,
        modes=modes,
        undiagonalized=sp.Matrix([[0, -1], [-1, 0]]),
        ghost_kind=GHOST_AUXILIARY,
        artifact=True,
        note=(
            "1/Box localizes with one auxiliary scalar psi obeying Box psi = phi, enforced by "
            "a Lagrange multiplier xi.  The (psi, xi) kinetic matrix is off-diagonal with "
            "eigenvalues +1 and -1, so the localized formulation carries exactly one ghost.  "
            "This is the known Deser-Woodard localization ghost: the localized system admits "
            "more initial data than the nonlocal theory, whose own solutions are the subspace "
            "with the declared homogeneous-solution-free boundary condition, so the ghost may "
            "be an artifact of the formulation rather than a mode of the nonlocal theory"
        ),
        rule=None,
        approximate=False,
    )


def _localize_quadrature(
    alpha: Fraction,
    integer_part: int,
    beta: Fraction,
    mu: sp.Expr,
    weight: sp.Expr,
    pole_count: int,
    *,
    spectral_subtraction: Mapping[str, str] | None,
) -> dict[str, Any]:
    """The Balakrishnan finite-pole route.  This one is an approximation and says so."""

    rule = gauss_jacobi_rule(beta, pole_count)
    exponent = sp.Rational(beta.numerator, beta.denominator)
    prefactor = 2 * sp.sin(sp.pi * exponent) / sp.pi
    nodes = rule["_exact_nodes"]
    weights = rule["_exact_weights"]
    if not nodes:
        nodes = [sp.Float(node, WORKING_PRECISION_DIGITS) for node in rule["_nodes"]]
        weights = [sp.Float(item, WORKING_PRECISION_DIGITS) for item in rule["_weights"]]
    masses = [mu * (1 + node) / (1 - node) for node in nodes]
    couplings = [
        weight * prefactor * mu ** (1 - exponent) * item / (1 - node)
        for item, node in zip(weights, nodes, strict=True)
    ]
    subtraction_report: dict[str, Any] | None = None
    if spectral_subtraction is not None:
        target = sp.Rational(Fraction(str(spectral_subtraction["mass_squared"])))
        amount = sp.Rational(Fraction(str(spectral_subtraction["weight"])))
        index = min(
            range(len(masses)),
            key=lambda position: abs(float(sp.N(masses[position] - target, 30))),
        )
        couplings[index] = couplings[index] - amount
        subtraction_report = {
            "role": "declared control-only spectral subtraction",
            "target_mass_squared": _exact(target),
            "subtracted_weight": _exact(amount),
            "pole_index": str(index),
            "why": (
                "a subtracted pole makes the spectral density sign-indefinite; whether the "
                "discretized residue at that pole is still positive depends on how much "
                "quadrature weight lands there, which is exactly the N-dependence the "
                "stability classifier exists to catch"
            ),
        }

    modes: list[dict[str, Any]] = []
    if integer_part == 0:
        for index, (mass, coupling) in enumerate(zip(masses, couplings, strict=True)):
            sign = 1 if sp.N(coupling, 30) > 0 else -1
            modes.append(
                _mode(
                    f"chi_{index + 1}",
                    sign,
                    mass,
                    coupling,
                    origin="balakrishnan_quadrature_pole",
                )
            )
    else:
        massless_residue = sum(
            coupling / mass for coupling, mass in zip(couplings, masses, strict=True)
        )
        modes.append(
            _mode(
                "chi_0",
                1 if sp.N(massless_residue, 30) > 0 else -1,
                0,
                massless_residue,
                origin="massless_pole_of_the_integer_factor",
            )
        )
        for index, (mass, coupling) in enumerate(zip(masses, couplings, strict=True)):
            residue = -coupling / mass
            sign = 1 if sp.N(residue, 30) > 0 else -1
            modes.append(
                _mode(
                    f"chi_{index + 1}",
                    sign,
                    mass,
                    residue,
                    origin="partial_fraction_of_the_integer_factor_times_a_quadrature_pole",
                )
            )
    ghost_kind = (
        GHOST_PROPAGATOR
        if any(int(mode["kinetic_sign"]) < 0 for mode in modes)
        else GHOST_NONE
    )
    if integer_part >= 1:
        note = (
            "alpha > 1: (k^2)^-alpha falls faster than 1/k^2, so partial fractions of "
            "(k^2)^-n against each quadrature pole put a strictly negative residue on every "
            "massive pole and the residues sum to zero exactly at every N.  That zero sum is "
            "the discrete Kallen-Lehmann statement: a propagator decaying faster than 1/k^2 "
            "cannot have a positive spectral density.  The ghost is a property of the "
            "declared operator, not of the quadrature"
        )
    else:
        note = (
            "0 < alpha < 1: the Balakrishnan spectral density is positive, so every pole of "
            "the N-pole localization is an ordinary healthy massive scalar.  The total "
            "spectral weight of the approximant is finite while the exact operator's is not, "
            "so the approximant is a different theory in the ultraviolet and the pass is a "
            "statement about each finite-N approximant only"
        )
    return _finish(
        route=ROUTE_QUADRATURE,
        alpha=alpha,
        integer_part=integer_part,
        beta=beta,
        mu=mu,
        weight=weight,
        pole_count=pole_count,
        modes=modes,
        undiagonalized=None,
        ghost_kind=ghost_kind,
        artifact=False,
        note=note,
        rule=rule,
        approximate=True,
        masses=masses,
        couplings=couplings,
        subtraction=subtraction_report,
    )


def _finish(
    *,
    route: str,
    alpha: Fraction,
    integer_part: int,
    beta: Fraction,
    mu: sp.Expr,
    weight: sp.Expr,
    pole_count: int,
    modes: Sequence[Mapping[str, Any]],
    undiagonalized: sp.Matrix | None,
    ghost_kind: str,
    artifact: bool,
    note: str,
    rule: Mapping[str, Any] | None,
    approximate: bool,
    masses: Sequence[sp.Expr] | None = None,
    couplings: Sequence[sp.Expr] | None = None,
    subtraction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the localization block, its principal symbol, and its convergence evidence.

    The kinetic and gradient blocks are always emitted in the canonically normalized basis:
    every mode carries kinetic sign ``+1`` or ``-1`` and its magnitude sits in the source
    coupling.  Routes whose natural presentation is an off-diagonal multiplier pair also carry
    the undiagonalized matrix, because that is the form the standard derivation writes down.
    """

    kinetic = sp.diag(*[int(mode["kinetic_sign"]) for mode in modes])
    gradient = sp.Matrix(kinetic)
    analysis = analyze_isotropic_second_order_symbol(kinetic, gradient)
    residue_sum = sum(sp.sympify(mode["propagator_residue_exact"]) for mode in modes)
    has_ghost = any(int(mode["kinetic_sign"]) < 0 for mode in modes)
    block: dict[str, Any] = {
        "route": route,
        "status": "localized",
        "alpha": str(alpha),
        "integer_part": str(integer_part),
        "fractional_part": str(beta),
        "reference_mass_squared_exact": _exact(mu),
        "reference_mass_squared_decimal": _decimal(mu),
        "arm_amplitude_exact": _exact(weight),
        "pole_count": str(pole_count),
        "is_approximation": approximate,
        "auxiliary_field_count": str(len(modes)),
        "modes": [dict(mode) for mode in modes],
        "kinetic_matrix": str(kinetic),
        "gradient_matrix": str(gradient),
        "principal_symbol": analysis.as_dict(),
        "ghost_pole_count": str(sum(1 for mode in modes if int(mode["kinetic_sign"]) < 0)),
        "ghost_kind": ghost_kind if has_ghost else GHOST_NONE,
        "ghost_localization_artifact_possible": bool(artifact and has_ghost),
        "residue_sign_sequence": "".join(mode["residue_sign"] for mode in modes),
        "residue_sum_rule": {
            "decimal": _decimal(residue_sum),
            "is_zero": bool(residue_sum == 0),
            "statement": (
                "the residues sum to zero exactly at this pole count, which is the discrete "
                "Kallen-Lehmann statement that a propagator falling faster than 1/k^2 cannot "
                "have a positive spectral density: at least one residue must be negative"
                if residue_sum == 0
                else "the residues sum to the finite positive total spectral weight of the "
                "approximant; the exact operator's total weight is infinite, so the "
                "approximant differs from it in the ultraviolet"
            ),
        },
        "note": note,
    }
    if undiagonalized is not None:
        block["undiagonalized_kinetic_matrix"] = str(undiagonalized)
        block["diagonalization_note"] = (
            "the standard derivation writes this route as an off-diagonal multiplier pair; "
            "the rotation chi_pm = (u +/- v)/sqrt(2) puts it in the canonically normalized "
            "basis emitted above without changing the eigenvalues"
        )
    if subtraction is not None:
        block["spectral_subtraction"] = dict(subtraction)
    if rule is not None:
        block["quadrature"] = {
            key: value for key, value in rule.items() if not key.startswith("_")
        }
    if masses is not None and couplings is not None:
        block["total_spectral_weight_decimal"] = _decimal(sum(couplings))
        block["pole_mass_squared_window"] = {
            "minimum_decimal": _decimal(min(masses, key=lambda item: float(sp.N(item, 30)))),
            "maximum_decimal": _decimal(max(masses, key=lambda item: float(sp.N(item, 30)))),
        }
        block["propagator_reproduction"] = _propagator_reproduction(
            alpha, integer_part, beta, mu, weight, masses, couplings
        )
    return block


def _propagator_reproduction(
    alpha: Fraction,
    integer_part: int,
    beta: Fraction,
    mu: sp.Expr,
    weight: sp.Expr,
    masses: Sequence[sp.Expr],
    couplings: Sequence[sp.Expr],
) -> dict[str, Any]:
    """How well the finite-pole form reproduces the operator it approximates."""

    # ``couplings`` already carries the declared arm amplitude, so the exact reference it is
    # compared against has to carry it too; comparing a weighted approximant against an
    # unweighted operator would report the amplitude as a convergence error.
    exponent = sp.Rational(beta.numerator, beta.denominator)
    rows: list[dict[str, str]] = []
    for factor in DECLARED_TEST_MOMENTA:
        momentum = mu * sp.Rational(Fraction(factor))
        approximate = sum(
            coupling / (momentum + mass)
            for coupling, mass in zip(couplings, masses, strict=True)
        )
        exact = weight * momentum ** (-exponent)
        rows.append(
            {
                "k_squared_over_mu": factor,
                "exact_decimal": _decimal(exact),
                "approximant_decimal": _decimal(approximate),
                "relative_error_decimal": _decimal(
                    sp.Abs(approximate - exact) / sp.Abs(exact), 8
                ),
            }
        )
    at_reference = sum(
        coupling / (mu + mass) for coupling, mass in zip(couplings, masses, strict=True)
    ) - weight * mu ** (-exponent)
    return {
        "fractional_factor": f"(-Box)^-{beta}",
        "integer_factor": f"(-Box)^-{integer_part}",
        "exact_at_reference_scale_residual_bound": _outward_bound(at_reference),
        "exact_at_reference_scale_statement": (
            "at k^2 = mu the resolvent factor (1-y)k^2 + mu(1+y) collapses to the constant "
            "2 mu, so the rule integrates a degree-zero polynomial against its own weight and "
            "reproduces mu^-beta exactly for every N; the residual above is the numerical "
            "check of that identity and is the module's cheapest known-answer control"
        ),
        "declared_momenta": list(DECLARED_TEST_MOMENTA),
        "rows": rows,
        "scope": (
            f"pointwise reproduction of the fractional factor of (-Box)^-{alpha} only; the "
            "declared (s/L2)^p (1+s/L2)^-(p+t) ultraviolet form factor is not materialized"
        ),
    }


# ---------------------------------------------------------------------------
# The rungs on the localized action.
# ---------------------------------------------------------------------------


def arm_rung_statuses(localization: Mapping[str, Any]) -> dict[str, Any]:
    """Decide the five ladder rungs on the localized nonlocal arm alone."""

    if localization.get("status") == "blocked":
        return {
            "statuses": {name: "blocked" for name in LADDER_RUNGS},
            "blocker": localization["blocker"],
            "evidence": {"reason": "the arm could not be localized at all"},
        }
    analysis = localization["principal_symbol"]
    ghost = bool(analysis["ghost_free"])
    statuses = {
        "ghost_freedom": "pass" if ghost else "reject",
        "gradient_stability": "pass" if analysis["gradient_stable"] else "reject",
        "tensor_sector": "pass",
        "principal_symbol_hyperbolicity": (
            "pass"
            if analysis["real_characteristics"]
            and analysis["cone_policy_pass"]
            and analysis["strongly_hyperbolic"]
            else "reject"
        ),
        "positive_energy_hamiltonian": "pass" if ghost else "reject",
    }
    return {
        "statuses": statuses,
        "blocker": None,
        "evidence": {
            "kinetic_matrix": localization["kinetic_matrix"],
            "gradient_matrix": localization["gradient_matrix"],
            "mode_speed_squared": analysis["speed_squared"],
            "ghost_pole_count": localization["ghost_pole_count"],
            "ghost_kind": localization["ghost_kind"],
            "residue_sign_sequence": localization["residue_sign_sequence"],
            "residue_sum_rule": localization["residue_sum_rule"],
            "tensor_sector_reason": (
                "the localized auxiliary scalars are minimally coupled and contribute no G4 "
                "term, so G_T and F_T are exactly those of the screening sector"
            ),
            "positive_energy_reason": (
                "a wrong-sign kinetic term has an energy density unbounded below; a "
                "ghost-free tower inherits the screening sector's pointwise energy status"
            ),
        },
    }


def combine_rungs(
    sector_rungs: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Merge the screening-sector rungs with the localized-arm rungs, fail-closed."""

    sector = {rung["rung"]: rung for rung in sector_rungs}
    if list(sector) != list(LADDER_RUNGS):
        raise NonlocalFractionalAdapterError("the sector ladder is not the declared sequence")
    merged: list[dict[str, Any]] = []
    for name in LADDER_RUNGS:
        sector_status = sector[name]["status"]
        arm_status = arm["statuses"][name]
        if "reject" in {sector_status, arm_status}:
            status = "reject"
        elif "blocked" in {sector_status, arm_status}:
            status = "blocked"
        else:
            status = "pass"
        blocker = None
        if status == "blocked":
            blocker = (
                arm["blocker"]
                if arm_status == "blocked"
                else sector[name]["evidence"].get("blocker")
            ) or "missing_adapter:unnamed"
        merged.append(
            {
                "rung": name,
                "status": status,
                "screening_sector_status": sector_status,
                "localized_arm_status": arm_status,
                "blocker": blocker,
            }
        )
    return merged


def localized_verdict(rungs: Sequence[Mapping[str, Any]]) -> str:
    """``LOCALIZED_PASS`` / ``LOCALIZED_REJECT:<rung>`` / ``LOCALIZED_BLOCKED:<code>``."""

    for rung in rungs:
        if rung["status"] == "reject":
            return f"LOCALIZED_REJECT:{rung['rung']}"
    for rung in rungs:
        if rung["status"] == "blocked":
            return f"LOCALIZED_BLOCKED:{rung['blocker']}"
    return "LOCALIZED_PASS"


def classify_stability(verdicts: Mapping[str, str]) -> str:
    """Classify a whole convergence study.  A verdict that moves with ``N`` is not a result."""

    if sorted(int(key) for key in verdicts) != sorted(DECLARED_POLE_COUNTS):
        raise NonlocalFractionalAdapterError(
            "a stability class needs a verdict at every declared pole count"
        )
    distinct = set(verdicts.values())
    if len(distinct) != 1:
        return "UNSTABLE_UNDER_LOCALIZATION"
    only = distinct.pop()
    if only == "LOCALIZED_PASS":
        return "STABLE_PASS"
    if only.startswith("LOCALIZED_REJECT:"):
        return "STABLE_REJECT:" + only.split(":", 1)[1]
    return "STABLE_BLOCKED:" + only.split(":", 1)[1]


def stability_statement(stability: str) -> str:
    counts = ", ".join(str(item) for item in DECLARED_POLE_COUNTS)
    if stability == "UNSTABLE_UNDER_LOCALIZATION":
        return (
            f"the verdict moves with the pole count over N in {{{counts}}}, so the "
            "localization has not decided this family and no verdict is claimed"
        )
    return (
        f"holds for every N-pole localization tested, N in {{{counts}}}; "
        "the nonlocal limit is not proved"
    )


# ---------------------------------------------------------------------------
# The localized action, in the repository's typed form.
# ---------------------------------------------------------------------------


_GRAMMAR_CACHE: dict[str, Any] = {}


def _grammar_and_contract(root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    key = str(Path(root).resolve())
    if key not in _GRAMMAR_CACHE:
        _GRAMMAR_CACHE[key] = (
            load_action_grammar(Path(root) / ACTION_GRAMMAR_PATH),
            load_field_contract(Path(root) / FIELD_CONTRACT_PATH),
        )
    return _GRAMMAR_CACHE[key]


def grammar_admission(root: str | Path, field_count: int) -> dict[str, Any]:
    """Ask the *real* frozen grammar what it admits of an ``field_count``-scalar tower.

    One auxiliary mode compiles: ``EH_R + SCALAR_X + SCALAR_MASS`` is exactly the localized
    mode Lagrangian.  A tower of more than one does not, and the errors the grammar returns
    are recorded verbatim rather than paraphrased -- lifting the bound is an operator decision
    about ``configs/covariant_action_grammar.json`` and ``configs/covariant_field_contract.json``,
    not something this adapter may take.
    """

    grammar, contract = _grammar_and_contract(root)
    fields = ["g_mu_nu", "psi_m"] + (
        ["phi"] if field_count == 1 else [f"chi_{index + 1}" for index in range(field_count)]
    )
    spec = {
        "schema_version": "sigma-action-spec-1.0",
        "role": "candidate",
        "fields": sorted(fields),
        "matter_metric": "g_mu_nu",
        "universal_constants": ["Lambda_phi", "m_phi"],
        "terms": ["EH_R", "SCALAR_X", "SCALAR_MASS"],
        "coefficients": {"EH_R": "1", "SCALAR_X": "1", "SCALAR_MASS": "1"},
        "parameter_domain": {"positive": ["Lambda_phi", "m_phi"]},
        "static_dictionary_status": "derived",
    }
    compiled = compile_action_spec(spec, grammar, contract)
    admission = {
        "declared_field_count": str(field_count),
        "declared_fields": sorted(fields),
        "valid": bool(compiled["valid"]),
        "errors": [str(item) for item in compiled["errors"]],
        "action_ir_content_sha256": str(compiled["content_sha256"]),
    }
    if field_count > 1:
        # The tower's own field ids are not in the contract, so the contract check fires
        # first and the *bound* never gets a chance to speak.  This second probe uses two
        # fields the contract does declare, so the bound itself is on the record.
        bound_probe = compile_action_spec(
            {**spec, "fields": ["A_mu", "g_mu_nu", "phi", "psi_m"]}, grammar, contract
        )
        admission["extra_dynamical_field_bound_probe"] = {
            "declared_fields": ["A_mu", "g_mu_nu", "phi", "psi_m"],
            "valid": bool(bound_probe["valid"]),
            "errors": [str(item) for item in bound_probe["errors"]],
            "why": (
                "configs/covariant_action_grammar.json bounds an action at one extra "
                "dynamical field, so even two contract-declared gravitational-sector fields "
                "are refused; an N-mode auxiliary tower needs a grammar and field-contract "
                "amendment, which is an operator decision this adapter does not take"
            ),
        }
    return admission


def localized_action_ir(
    root: str | Path, localization: Mapping[str, Any], admission: Mapping[str, Any]
) -> dict[str, Any]:
    """The localized multi-field action in the repository's typed term form."""

    terms: list[dict[str, Any]] = [
        {
            "id": "EH_R",
            "coefficient": "1",
            "density": "sqrt(-g) R",
            "fields": ["g_mu_nu"],
            "invariant": None,
            "maximum_derivatives_per_field": 2,
        }
    ]
    for mode in localization["modes"]:
        identifier = mode["id"]
        terms.append(
            {
                "id": f"SCALAR_X:{identifier}",
                "coefficient": mode["kinetic_sign"],
                "density": f"sqrt(-g) Lambda_phi^4 X_{identifier}",
                "fields": ["g_mu_nu", identifier],
                "invariant": f"X_{identifier}",
                "maximum_derivatives_per_field": 1,
            }
        )
        if mode["mass_squared_exact"] != "0":
            terms.append(
                {
                    "id": f"SCALAR_MASS:{identifier}",
                    "coefficient": mode["kinetic_sign"],
                    "density": f"sqrt(-g) m_{identifier}^2 {identifier}^2",
                    "fields": ["g_mu_nu", identifier],
                    "invariant": None,
                    "maximum_derivatives_per_field": 0,
                }
            )
        terms.append(
            {
                "id": f"SOURCE_COUPLING:{identifier}",
                "coefficient": mode["source_coupling_exact"],
                "density": f"sqrt(-g) {identifier} T/Mpl",
                "fields": ["g_mu_nu", identifier, "psi_m"],
                "invariant": None,
                "maximum_derivatives_per_field": 0,
            }
        )
    body = {
        "schema_version": ACTION_IR_SCHEMA,
        "derived_from": f"auxiliary-field localization of (-Box)^{localization['alpha']}",
        "route": localization["route"],
        "pole_count": localization["pole_count"],
        "fields": ["g_mu_nu", "psi_m"] + [mode["id"] for mode in localization["modes"]],
        "auxiliary_fields": [
            {
                "id": mode["id"],
                "kinetic_sign": mode["kinetic_sign"],
                "mass_squared_exact": mode["mass_squared_exact"],
                "mass_squared_decimal": mode["mass_squared_decimal"],
                "source_coupling_exact": mode["source_coupling_exact"],
                "source_coupling_decimal": mode["source_coupling_decimal"],
            }
            for mode in localization["modes"]
        ],
        "terms": terms,
        "principal_symbol": {
            "kinetic_matrix": localization["kinetic_matrix"],
            "gradient_matrix": localization["gradient_matrix"],
            "convention": "L2 = dot(u)^T K dot(u)/2 - partial_i(u)^T G partial_i(u)/2",
            "analysis": localization["principal_symbol"],
        },
        "covariant_action_grammar_admission": dict(admission),
        "matter_coupling_status": (
            "the source coupling is written for completeness and is NOT admitted: "
            "configs/covariant_field_contract.json forbids direct scalar-matter coupling by "
            "rule, so missing_adapter:direct_scalar_matter_coupling stays open"
        ),
    }
    return _seal(body)


# ---------------------------------------------------------------------------
# The three declared controls of docs/PHYSICS_CONCEPT_LANGUAGE.md, per localization.
# ---------------------------------------------------------------------------


def nonlocality_controls(localization: Mapping[str, Any]) -> dict[str, Any]:
    """Causality, initial-value, and auxiliary-field controls for one localized arm."""

    analysis = localization["principal_symbol"]
    speeds = sorted(set(analysis["speed_squared"]))
    return {
        "causality": {
            "status": "pass" if analysis["cone_policy_pass"] else "reject",
            "mode_speed_squared": speeds,
            "statement": (
                "every localized mode propagates on the metric light cone, so the localized "
                "approximant is causal with respect to g_mu_nu under the declared cone policy"
            ),
            "open": (
                "the branch of (-Box)^alpha the localization represents is the Euclidean/"
                "Feynman one; the retarded branch that a causal nonlocal effective action "
                "needs is not determined by this construction"
            ),
        },
        "initial_value": {
            "status": "pass" if analysis["strongly_hyperbolic"] else "unresolved",
            "statement": (
                "the localized system is a finite set of second-order scalars with a "
                "nondegenerate kinetic matrix and a complete real eigenbasis, so its Cauchy "
                "problem is the ordinary one"
            ),
            "open": (
                "the localized system carries more initial data than the nonlocal theory; the "
                "nonlocal theory is the subspace fixed by the declared boundary condition on "
                "the auxiliary fields, and that restriction is not proved here"
            ),
        },
        "auxiliary_field": {
            "status": "pass",
            "retained_mode_count": localization["auxiliary_field_count"],
            "statement": (
                "every mode the localization produces is retained and tested; none is "
                "truncated, and each carries its own kinetic sign, mass, and coupling"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Known-answer controls.  A run whose controls do not fire aborts.
# ---------------------------------------------------------------------------


#: Control-only declarations, each with the answer it must produce.
CONTROL_CASES: dict[str, dict[str, Any]] = {
    "local_dalembertian_passthrough": {
        "role": "positive",
        "alpha": "1",
        "reference_mass_squared": "1",
        "amplitude": "1",
        "expect_stability": "STABLE_PASS",
        "expect_ghosts": "0",
        "why": (
            "alpha = 1 is the ordinary local d'Alembertian.  Fed through the adapter it must "
            "introduce no auxiliary field and must reproduce the un-adapted ladder verdict "
            "rung for rung; anything else means the adapter is not a no-op on a local theory"
        ),
    },
    "deser_woodard_inverse_dalembertian": {
        "role": "known_structure",
        "alpha": "-1",
        "reference_mass_squared": "1",
        "amplitude": "1",
        "expect_stability": "STABLE_REJECT:ghost_freedom",
        "expect_ghosts": "1",
        "why": (
            "1/Box is the Deser-Woodard operator.  Its localization is the known "
            "one-auxiliary-field form Box psi = phi with a Lagrange multiplier, whose "
            "off-diagonal kinetic matrix carries exactly one ghost eigenvalue"
        ),
    },
    "ostrogradsky_quadratic_box": {
        "role": "negative",
        "alpha": "2",
        "reference_mass_squared": "1",
        "amplitude": "1",
        "expect_stability": "STABLE_REJECT:ghost_freedom",
        "expect_ghosts": "1",
        "why": (
            "(-Box)^2 is a deliberately ghost-laden local higher-derivative operator; its "
            "exact one-auxiliary-field localization must reject on ghost freedom"
        ),
    },
    "subtracted_pole_unstable": {
        "role": "negative",
        "alpha": "1/2",
        "reference_mass_squared": "1",
        "amplitude": "1",
        "spectral_subtraction": {"mass_squared": "1", "weight": "1/4"},
        "expect_stability": "UNSTABLE_UNDER_LOCALIZATION",
        "expect_ghosts": None,
        "why": (
            "a declared subtraction of weight 1/4 at m^2 = 1 leaves the nearest discretized "
            "residue positive at N = 2 and N = 4 and drives it negative at N = 8 and N = 16, "
            "so the classifier must refuse to call it either way"
        ),
    },
}


def run_controls(root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drive the adapter with known answers; a control that does not fire aborts the run."""

    sector = ladder.run_ladder(
        root, "canonical_scalar", control=ladder.CONTROL_ACTIONS["canonical_scalar"]
    )
    unadapted = ladder.ladder_verdict({"full_lift_blockers": []}, sector["rungs"])
    if unadapted != "FORMAL_PASS":
        raise NonlocalFractionalAdapterError(
            "the imported canonical-scalar control does not pass the un-adapted ladder"
        )
    unadapted_statuses = {rung["rung"]: rung["status"] for rung in sector["rungs"]}

    results: dict[str, Any] = {}
    for name in sorted(CONTROL_CASES):
        declared = CONTROL_CASES[name]
        verdicts: dict[str, str] = {}
        ghosts: dict[str, str] = {}
        residues: dict[str, str] = {}
        statuses: dict[str, dict[str, str]] = {}
        for pole_count in DECLARED_POLE_COUNTS:
            localization = localize(
                Fraction(declared["alpha"]),
                Fraction(declared["reference_mass_squared"]),
                Fraction(declared["amplitude"]),
                pole_count,
                spectral_subtraction=declared.get("spectral_subtraction"),
            )
            rungs = combine_rungs(sector["rungs"], arm_rung_statuses(localization))
            verdicts[str(pole_count)] = localized_verdict(rungs)
            ghosts[str(pole_count)] = localization["ghost_pole_count"]
            residues[str(pole_count)] = localization["residue_sign_sequence"]
            statuses[str(pole_count)] = {rung["rung"]: rung["status"] for rung in rungs}
        observed = classify_stability(verdicts)
        if observed != declared["expect_stability"]:
            raise NonlocalFractionalAdapterError(
                f"control {name} expected {declared['expect_stability']} but observed {observed}"
            )
        if declared["expect_ghosts"] is not None and set(ghosts.values()) != {
            declared["expect_ghosts"]
        }:
            raise NonlocalFractionalAdapterError(
                f"control {name} expected {declared['expect_ghosts']} ghost poles at every N"
            )
        if name == "local_dalembertian_passthrough":
            for pole_count, observed_statuses in sorted(statuses.items()):
                if observed_statuses != unadapted_statuses:
                    raise NonlocalFractionalAdapterError(
                        "the local pass-through control did not reproduce the un-adapted rungs"
                    )
        results[name] = {
            "role": declared["role"],
            "why": declared["why"],
            "alpha": declared["alpha"],
            "expected_stability": declared["expect_stability"],
            "observed_stability": observed,
            "per_pole_count_verdicts": verdicts,
            "per_pole_count_ghost_poles": ghosts,
            "per_pole_count_residue_signs": residues,
        }
        if "spectral_subtraction" in declared:
            results[name]["spectral_subtraction"] = dict(declared["spectral_subtraction"])
    results["local_dalembertian_passthrough"]["unadapted_ladder_verdict"] = unadapted
    results["local_dalembertian_passthrough"]["unadapted_rung_statuses"] = unadapted_statuses

    quadrature: dict[str, Any] = {}
    for beta in ("1/2", "3/4"):
        rows = {}
        for pole_count in DECLARED_POLE_COUNTS:
            rule = gauss_jacobi_rule(Fraction(beta), pole_count)
            rows[str(pole_count)] = {
                "exactness": rule["exactness"],
                "moment_residual_bound_up_to_degree_2N_minus_1": rule[
                    "moment_residual_bound_up_to_degree_2N_minus_1"
                ],
            }
        quadrature[beta] = rows
    results["gauss_jacobi_moment_controls"] = {
        "role": "positive",
        "why": (
            "an N-point Gauss-Jacobi rule is exact on every polynomial of degree up to 2N-1; "
            "a rule that misses its own moments would silently corrupt every residue"
        ),
        "by_beta": quadrature,
    }
    return results, sector


# ---------------------------------------------------------------------------
# The run over all 71 families.
# ---------------------------------------------------------------------------


_LOCALIZATION_CACHE: dict[str, dict[str, Any]] = {}


def _localization_bundle(
    root: str | Path, alpha: Fraction, mu: Fraction, amplitude: Fraction
) -> dict[str, Any]:
    key = f"{alpha}|{mu}|{amplitude}"
    if key in _LOCALIZATION_CACHE:
        return _LOCALIZATION_CACHE[key]
    by_pole_count: dict[str, Any] = {}
    action_ir: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    for pole_count in DECLARED_POLE_COUNTS:
        localization = localize(alpha, mu, amplitude, pole_count)
        by_pole_count[str(pole_count)] = localization
        admission = grammar_admission(root, len(localization.get("modes", [])) or 1)
        action_ir[str(pole_count)] = localized_action_ir(root, localization, admission)
        controls[str(pole_count)] = nonlocality_controls(localization)
    bundle = {
        "alpha": str(alpha),
        "reference_mass_squared": str(mu),
        "arm_amplitude": str(amplitude),
        "by_pole_count": by_pole_count,
        "localized_action_ir_by_pole_count": action_ir,
        "nonlocality_controls_by_pole_count": controls,
    }
    bundle["localization_id"] = canonical_sha256(_plain(bundle))
    _LOCALIZATION_CACHE[key] = bundle
    return bundle


def _ghost_finding(bundle: Mapping[str, Any]) -> dict[str, Any]:
    first = bundle["by_pole_count"][str(DECLARED_POLE_COUNTS[0])]
    kinds = {bundle["by_pole_count"][str(n)]["ghost_kind"] for n in DECLARED_POLE_COUNTS}
    sums = {
        str(n): bundle["by_pole_count"][str(n)]["residue_sum_rule"]["is_zero"]
        for n in DECLARED_POLE_COUNTS
    }
    return {
        "ghost_kind": min(kinds) if len(kinds) == 1 else "n_dependent",
        "ghost_pole_count_by_pole_count": {
            str(n): bundle["by_pole_count"][str(n)]["ghost_pole_count"]
            for n in DECLARED_POLE_COUNTS
        },
        "residue_sign_sequence_by_pole_count": {
            str(n): bundle["by_pole_count"][str(n)]["residue_sign_sequence"]
            for n in DECLARED_POLE_COUNTS
        },
        "kallen_lehmann_residue_sum_is_zero_by_pole_count": sums,
        "localization_artifact_possible": bool(
            first["ghost_localization_artifact_possible"]
        ),
        "statement": first["note"],
    }


def run_adapter(root: str | Path) -> dict[str, Any]:
    """Run the localization adapter over every v3 family and build the sealed receipt."""

    root = Path(root).resolve()
    ladder_receipt = load_ladder_receipt(root)
    families = load_families(root, ladder_receipt)
    controls, control_sector = run_controls(root)

    sector_cache: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    bundles: dict[str, Any] = {}
    for family in families:
        alpha, component = read_declared_alpha(family)
        values = family["representative_values"]
        screening = str(family["screening_family"])
        if screening not in sector_cache:
            sector_cache[screening] = ladder.run_ladder(root, screening)
        sector = sector_cache[screening]
        length_scale = Fraction(str(values["L2"]))
        bundle = _localization_bundle(
            root, alpha, 1 / (length_scale * length_scale), Fraction(str(values["w_power"]))
        )
        bundles[bundle["localization_id"]] = bundle

        verdicts: dict[str, str] = {}
        rungs_by_count: dict[str, Any] = {}
        for pole_count in DECLARED_POLE_COUNTS:
            localization = bundle["by_pole_count"][str(pole_count)]
            rungs = combine_rungs(sector["rungs"], arm_rung_statuses(localization))
            verdicts[str(pole_count)] = localized_verdict(rungs)
            rungs_by_count[str(pole_count)] = rungs
        stability = classify_stability(verdicts)
        entries.append(
            {
                "representative_ordinal": int(family["representative_ordinal"]),
                "screening_family": screening,
                "size": int(family["size"]),
                "alpha": str(alpha),
                "alpha_source": (
                    "parsed from the lift's own nonlocal component text and cross-checked "
                    "against the declared kernel axis 1 - t/2"
                ),
                "kernel_parameters": {
                    key: str(values[key])
                    for key in ("L1", "L2", "p", "t", "w_power", "w_yukawa", "local", "screen")
                },
                "declared_nonlocal_component": component,
                "localization_id": bundle["localization_id"],
                "sector_id": str(
                    ladder.SECTOR_ANSATZ[screening]["sector_id"]
                ),
                "sector_verdict": ladder.sector_verdict(sector["rungs"]),
                "per_pole_count_verdicts": verdicts,
                "per_pole_count_rungs": {
                    key: {rung["rung"]: rung["status"] for rung in value}
                    for key, value in sorted(rungs_by_count.items())
                },
                "per_pole_count_rung_detail": rungs_by_count,
                "stability": stability,
                "stability_statement": stability_statement(stability),
                "ghost_finding": _ghost_finding(bundle),
                "residual_blockers": list(RESIDUAL_BLOCKERS),
            }
        )
    entries.sort(key=lambda item: item["representative_ordinal"])
    aggregate = _aggregate(entries)

    config = {
        "declared_pole_counts": [str(item) for item in DECLARED_POLE_COUNTS],
        "ladder_rungs": list(LADDER_RUNGS),
        "precision": {
            "working_digits": str(WORKING_PRECISION_DIGITS),
            "emitted_digits": str(EMITTED_PRECISION_DIGITS),
            "policy": (
                "exact rationals and closed-form algebraic numbers where the weight admits "
                "them; otherwise the exact Jacobi polynomial whose roots the nodes are, plus "
                "decimal strings at the declared emitted precision"
            ),
        },
        "reference_mass_scale_rule": (
            "mu = 1/L2^2, the mass-squared scale of the kernel's own declared UV form-factor "
            "length; the Gauss-Jacobi rule is exact at k^2 = mu for every N"
        ),
        "declared_test_momenta_over_mu": list(DECLARED_TEST_MOMENTA),
        "routes": {
            ROUTE_LOCAL: "integer alpha >= 1: exactly local, no approximation",
            ROUTE_INVERSE: "alpha = -1: the Deser-Woodard one-auxiliary-field localization",
            ROUTE_QUADRATURE: (
                "alpha = n + beta with 0 < beta < 1 and n in {0, 1}: N-pole Gauss-Jacobi "
                "discretization of the Balakrishnan spectral representation"
            ),
        },
        "blockers": BLOCKERS,
        "residual_blockers": list(RESIDUAL_BLOCKERS),
        "control_cases": CONTROL_CASES,
        "discharged_blocker": DISCHARGED_BLOCKER,
    }
    body = {
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "auxiliary-field localization of the nonlocal (-Box)^alpha arm carried by all 71 "
            "surviving v3 screened-gravity families, and the v3 formal ladder re-run on the "
            "localized multi-field action at every declared pole count.  Every verdict is a "
            "verdict about a finite-pole approximant of the nonlocal theory, reported as a "
            "convergence study; a verdict that moves with the pole count is reported as "
            "UNSTABLE_UNDER_LOCALIZATION and is not a result"
        ),
        "claims": CLAIMS,
        "assumptions": {
            "approximation_scope": (
                "the finite-pole localization reproduces the declared fractional operator "
                "exactly at k^2 = mu and to the reported relative error elsewhere; it is not "
                "the nonlocal theory and its verdicts are verdicts about the approximant"
            ),
            "uv_form_factor_is_ultraviolet_only": (
                "the lift declares the (s/L2)^p (1+s/L2)^-(p+t) factor as a UV form factor "
                "turning on at L2, so the localization models the infrared operator "
                "(-Box)^alpha alone and assumes the form factor neither adds nor removes "
                "poles below that scale.  An entire-function form factor of the exp(Box/M^2) "
                "type would violate that assumption and is not what the lift declares; "
                "materializing an arbitrary form factor as an operator stays a separate adapter"
            ),
            "sector_and_arm_are_additive": (
                "the lift declares no derivative cross-coupling between the mediator and the "
                "screening scalar, so the localized principal symbol is block diagonal; any "
                "cross-coupling generated by the unpinned matter coupling is not modelled"
            ),
            "ladder_is_a_necessary_condition": (
                "a rejection on the localized arm eliminates the family outright because the "
                "arm is part of every declared lift; a pass admits nothing, because the "
                "complete lift stays blocked on the residual adapters"
            ),
            "cone_policy": "0 <= c_mode^2 <= 1 relative to the physical metric cone",
        },
        "inputs": {
            "formal_ladder_receipt": {
                "path": LADDER_RECEIPT_PATH,
                "content_sha256": ladder_receipt["content_sha256"],
                "config_sha256": ladder_receipt["config_sha256"],
                "families_in": int(ladder_receipt["counts"]["families_in"]),
                "blocked_on_this_adapter": int(
                    ladder_receipt["counts"]["blocked_by_adapter"][DISCHARGED_BLOCKER]
                ),
                "decision": str(ladder_receipt["decision"]),
            },
            "family_representatives": {
                "path": REPRESENTATIVES_PATH,
                "content_sha256": canonical_sha256(
                    {
                        key: item
                        for key, item in json.loads(
                            (root / REPRESENTATIVES_PATH).read_text(encoding="utf-8")
                        ).items()
                        if key != "content_sha256"
                    }
                ),
            },
            "screen_receipt": {
                "path": ladder.SCREEN_RECEIPT_PATH,
                "content_sha256": ladder_receipt["inputs"]["screen_receipt"]["content_sha256"],
            },
        },
        "config": config,
        "controls": controls,
        "control_sector_action_content_sha256": control_sector["action_ir"]["content_sha256"],
        "counts": aggregate,
        "localizations": {key: bundles[key] for key in sorted(bundles)},
        "families": entries,
        "decision": _decision(aggregate),
        "residual_gap_report": _residual_gap_report(aggregate),
    }
    body["config_sha256"] = canonical_sha256(body["config"])
    body = _plain(body)
    _no_floats(body)
    return _seal(body)


def _aggregate(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = list(entries)
    stability_counts: dict[str, int] = {}
    reject_by_rung: dict[str, int] = {}
    blocked_by_code: dict[str, int] = {}
    for entry in families:
        stability = entry["stability"]
        stability_counts[stability] = stability_counts.get(stability, 0) + 1
        if stability.startswith("STABLE_REJECT:"):
            rung = stability.split(":", 1)[1]
            reject_by_rung[rung] = reject_by_rung.get(rung, 0) + 1
        if stability.startswith("STABLE_BLOCKED:"):
            code = stability.split(":", 1)[1]
            blocked_by_code[code] = blocked_by_code.get(code, 0) + 1
    by_screening: dict[str, dict[str, Any]] = {}
    by_alpha: dict[str, dict[str, Any]] = {}
    cross: dict[str, dict[str, int]] = {}
    for entry in families:
        for table, key in (
            (by_screening, entry["screening_family"]),
            (by_alpha, entry["alpha"]),
        ):
            block = table.setdefault(key, {"families": 0, "candidates": 0, "stability": {}})
            block["families"] += 1
            block["candidates"] += entry["size"]
            block["stability"][entry["stability"]] = (
                block["stability"].get(entry["stability"], 0) + 1
            )
        label = f"{entry['screening_family']}|alpha={entry['alpha']}"
        row = cross.setdefault(label, {})
        row[entry["stability"]] = row.get(entry["stability"], 0) + 1
    ghost_families = [
        entry
        for entry in families
        if entry["ghost_finding"]["ghost_kind"] != GHOST_NONE
    ]
    return {
        "families_in": len(families),
        "candidates_in": sum(entry["size"] for entry in families),
        "stable_pass": stability_counts.get("STABLE_PASS", 0),
        "stable_reject": sum(reject_by_rung.values()),
        "stable_blocked": sum(blocked_by_code.values()),
        "unstable_under_localization": stability_counts.get(
            "UNSTABLE_UNDER_LOCALIZATION", 0
        ),
        "stable_reject_by_rung": {key: reject_by_rung[key] for key in sorted(reject_by_rung)},
        "stable_blocked_by_adapter": {
            key: blocked_by_code[key] for key in sorted(blocked_by_code)
        },
        "stability_counts": {key: stability_counts[key] for key in sorted(stability_counts)},
        "by_screening_family": {key: by_screening[key] for key in sorted(by_screening)},
        "by_alpha": {key: by_alpha[key] for key in sorted(by_alpha)},
        "stability_by_screening_family_and_alpha": {key: cross[key] for key in sorted(cross)},
        "families_with_a_ghost": len(ghost_families),
        "ghost_kind_counts": _count(
            entry["ghost_finding"]["ghost_kind"] for entry in families
        ),
        "ghost_localization_artifact_possible": sum(
            1 for entry in families if entry["ghost_finding"]["localization_artifact_possible"]
        ),
        "distinct_localizations": len({entry["localization_id"] for entry in families}),
    }


def _count(values: Any) -> dict[str, int]:
    table: dict[str, int] = {}
    for value in values:
        table[str(value)] = table.get(str(value), 0) + 1
    return {key: table[key] for key in sorted(table)}


def _decision(aggregate: Mapping[str, Any]) -> str:
    rungs = ", ".join(
        f"{count} at {rung}" for rung, count in sorted(aggregate["stable_reject_by_rung"].items())
    ) or "none"
    counts = ", ".join(str(item) for item in DECLARED_POLE_COUNTS)
    return (
        f"NONLOCAL ARM LOCALIZED: all {aggregate['families_in']} v3 families now have a "
        f"materialized nonlocal arm, so missing_adapter:nonlocal_fractional_operator is "
        f"discharged at the level of the finite-pole approximant.  "
        f"{aggregate['stable_reject']} families are eliminated at every declared pole count "
        f"({rungs}), {aggregate['stable_pass']} clear every executable rung at every declared "
        f"pole count, {aggregate['stable_blocked']} remain blocked on a different adapter, and "
        f"{aggregate['unstable_under_localization']} are UNSTABLE_UNDER_LOCALIZATION.  Every "
        f"non-rejecting verdict holds for every N-pole localization tested, N in {{{counts}}}; "
        f"the nonlocal limit is not proved and the complete lift of every family stays blocked "
        f"on the residual adapters"
    )


def _residual_gap_report(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "statement": (
            "this adapter discharges one of the three universal materialization blockers of "
            "the v3 ladder.  It does not touch the AQUAL nu inversion, the direct "
            "scalar-matter coupling the frozen field contract forbids, the cubic-G3 weak-field "
            "cone, or the ultraviolet form factor of the very arm it localizes"
        ),
        "residual_blockers": list(RESIDUAL_BLOCKERS),
        "still_blocked_at_a_rung": aggregate["stable_blocked_by_adapter"],
        "next_adapters": [
            "missing_adapter:aqual_nu_to_kessence_inversion",
            "missing_adapter:cubic_g3_uniform_weak_field_cone",
            "missing_adapter:uv_form_factor_operator",
            "missing_adapter:direct_scalar_matter_coupling",
        ],
        "why_that_order": (
            "the AQUAL inversion is the remaining universal arm; the cubic-G3 cone is what "
            "still blocks every acceleration-screened family at a rung; the UV form factor "
            "bounds the scope of this adapter's own passes; the matter coupling needs a "
            "field-contract amendment and therefore an operator decision"
        ),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal, binding, claim, control, convergence-study, and structural replay; fail closed."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise NonlocalFractionalAdapterError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise NonlocalFractionalAdapterError("receipt seal changed")
    if value.get("claims") != CLAIMS:
        raise NonlocalFractionalAdapterError("claims block changed")
    if not value["claims"].get("approximation_is_not_the_nonlocal_theory"):
        raise NonlocalFractionalAdapterError(
            "a finite-pole localization receipt must carry "
            "approximation_is_not_the_nonlocal_theory"
        )
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise NonlocalFractionalAdapterError("config binding changed")
    _no_floats(body)

    declared = [str(item) for item in DECLARED_POLE_COUNTS]
    if value["config"].get("declared_pole_counts") != declared:
        raise NonlocalFractionalAdapterError("the declared pole-count set changed")
    inputs = value.get("inputs", {})
    ladder_input = inputs.get("formal_ladder_receipt", {})
    if (
        not isinstance(ladder_input.get("content_sha256"), str)
        or len(ladder_input["content_sha256"]) != 64
    ):
        raise NonlocalFractionalAdapterError("the formal-ladder binding is malformed")
    if ladder_input.get("blocked_on_this_adapter") != ladder_input.get("families_in"):
        raise NonlocalFractionalAdapterError(
            "the receipt does not bind a ladder in which every family was blocked here"
        )

    controls = value.get("controls", {})
    if set(controls) != set(CONTROL_CASES) | {"gauss_jacobi_moment_controls"}:
        raise NonlocalFractionalAdapterError("control set changed")
    for name in sorted(CONTROL_CASES):
        entry = controls.get(name, {})
        if entry.get("observed_stability") != CONTROL_CASES[name]["expect_stability"]:
            raise NonlocalFractionalAdapterError(f"control {name} did not fire in the receipt")
        if sorted(entry.get("per_pole_count_verdicts", {})) != sorted(declared):
            raise NonlocalFractionalAdapterError(
                f"control {name} does not carry the declared convergence study"
            )
    if controls["subtracted_pole_unstable"]["observed_stability"] != (
        "UNSTABLE_UNDER_LOCALIZATION"
    ):
        raise NonlocalFractionalAdapterError("the instability control is not reachable")

    families = value.get("families", [])
    if len(families) != int(ladder_input.get("families_in", -1)):
        raise NonlocalFractionalAdapterError("receipt does not carry every family")
    localizations = value.get("localizations", {})
    for entry in families:
        verdicts = entry.get("per_pole_count_verdicts", {})
        if sorted(verdicts) != sorted(declared):
            raise NonlocalFractionalAdapterError(
                "a family does not carry a verdict at every declared pole count"
            )
        stability = entry.get("stability", "")
        if stability != classify_stability(verdicts):
            raise NonlocalFractionalAdapterError("a stability class does not replay")
        if stability.startswith("STABLE") and len(verdicts) < len(DECLARED_POLE_COUNTS):
            raise NonlocalFractionalAdapterError(
                "a STABLE claim needs the full declared convergence study"
            )
        if stability.startswith("STABLE_REJECT:"):
            if stability.split(":", 1)[1] not in LADDER_RUNGS:
                raise NonlocalFractionalAdapterError("a rejection names an unknown rung")
        elif stability.startswith("STABLE_BLOCKED:"):
            if stability.split(":", 1)[1] not in BLOCKERS:
                raise NonlocalFractionalAdapterError("a block names an unknown adapter")
        elif stability not in {"STABLE_PASS", "UNSTABLE_UNDER_LOCALIZATION"}:
            raise NonlocalFractionalAdapterError(f"unknown stability class {stability!r}")
        if entry.get("stability_statement") != stability_statement(stability):
            raise NonlocalFractionalAdapterError("the stability statement was rewritten")
        if list(entry.get("residual_blockers", [])) != list(RESIDUAL_BLOCKERS):
            raise NonlocalFractionalAdapterError("a verdict dropped its residual blockers")
        if entry.get("localization_id") not in localizations:
            raise NonlocalFractionalAdapterError("a family references an unknown localization")
    for key, bundle in sorted(localizations.items()):
        stripped = {
            name: item for name, item in bundle.items() if name != "localization_id"
        }
        if canonical_sha256(stripped) != key:
            raise NonlocalFractionalAdapterError("a localization id does not replay")
        if sorted(bundle.get("by_pole_count", {})) != sorted(declared):
            raise NonlocalFractionalAdapterError(
                "a localization does not carry every declared pole count"
            )
    # The aggregate is the part a reader quotes, so it must be recomputable from the family
    # list it claims to summarize.  A resealed receipt with a doctored headline fails here.
    if _plain(_aggregate(families)) != value.get("counts"):
        raise NonlocalFractionalAdapterError("aggregate counts do not replay from the families")
    if value.get("decision") != _decision(value["counts"]):
        raise NonlocalFractionalAdapterError("the decision line does not replay")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write(value: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise NonlocalFractionalAdapterError("refusing to overwrite an immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Auxiliary-field localization adapter for the nonlocal (-Box)^alpha arm of the "
            "71 surviving v3 screened-gravity families."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.validate_checked:
        target = Path(args.output) if args.output else root / RECEIPT_PATH
        validate_receipt(json.loads(target.read_text(encoding="utf-8")))
        return 0
    result = run_adapter(root)
    if args.output:
        _write(result, args.output)
    print(
        json.dumps(
            {
                "families_in": result["counts"]["families_in"],
                "stable_pass": result["counts"]["stable_pass"],
                "stable_reject": result["counts"]["stable_reject"],
                "stable_reject_by_rung": result["counts"]["stable_reject_by_rung"],
                "stable_blocked": result["counts"]["stable_blocked"],
                "stable_blocked_by_adapter": result["counts"]["stable_blocked_by_adapter"],
                "unstable_under_localization": result["counts"][
                    "unstable_under_localization"
                ],
                "by_alpha": {
                    key: block["stability"]
                    for key, block in result["counts"]["by_alpha"].items()
                },
                "decision": result["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
