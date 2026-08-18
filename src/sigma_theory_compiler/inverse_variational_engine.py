"""The inverse problem of the calculus of variations: equations of motion back to an action.

The discovery pipeline in this repository runs forwards in two disconnected halves.  One half
turns data into an empirical law -- :mod:`.blind_planetary_law_rediscovery_campaign` recovers
``g = A/r^2`` from anonymized columns, :mod:`.tolerance_aware_fitting` decides which laws survive
declared measurement intervals.  The other half starts from a *declared action* and derives
consequences -- :mod:`.derivation_chain_demo` runs ``action -> Euler-Lagrange -> observable`` with
every step recomputed.  Nothing joined them.  A fitted curve and a first-principles theory sat on
either side of a gap that a human always crossed by hand.

This module is that missing middle step, done mechanically.  Given equations of motion, it asks
the question Helmholtz answered in 1887: *does a Lagrangian exist whose Euler-Lagrange equations
are exactly these?*  The answer is not a search or a heuristic.  It is a finite set of identities
on the derivatives of the equations, and they are necessary **and** sufficient on the declared
class.  So the verdict is a decision, not an opinion.

Step 1, the Helmholtz test.  For a system ``E_i(t, q, q', q'') = 0`` in ``n`` coordinates, with
the convention ``E_i = d/dt(dL/dq'_i) - dL/dq_i``, write ``W_ij = dE_i/dq''_j``.  A Lagrangian
exists locally if and only if

    (H1)  W_ij - W_ji = 0
    (H2)  dE_i/dq'_j + dE_j/dq'_i - 2 d/dt W_ij = 0
    (H3)  dE_i/dq_j - dE_j/dq_i - (1/2) d/dt (dE_i/dq'_j - dE_j/dq'_i) = 0

where ``t, q, q', q'', q'''`` are treated as independent coordinates on the third-order jet and a
condition holds only if its residual vanishes *identically* there.  The verdict is VARIATIONAL,
NOT_VARIATIONAL with the exact failing condition and its residual, or OUT_OF_DECLARED_CLASS.  A
NOT_VARIATIONAL verdict is a scientific result in its own right: the law cannot come from a least
action principle, so it carries no Noether theorem, no canonical structure and no Hamiltonian --
in the declared reduction in which it was stated.

Step 2, construction.  When the conditions hold, a Lagrangian is built, by the Volterra/Tonti
homotopy ``L_V = int_0^1 q^k E_k(t, lambda q, lambda q', lambda q'') dlambda`` reduced to first
order by removing its (necessarily total-derivative) acceleration part, or -- when the equations
are not polynomial in the jet variables, because then the homotopy contracts the configuration
through the force's singularity at the origin and its lambda-integral diverges -- by a declared
central-potential ansatz.  Which method guessed the Lagrangian does not matter, because the
construction is then **differentiated back**: ``d/dt(dL/dq'_i) - dL/dq_i - E_i`` must be
identically zero.  That round trip is the proof, and it is a test.

Step 3, integrating factors.  A damped oscillator is not variational.  Multiplied by ``e^(gamma
t)`` it is, and the Lagrangian that falls out is Caldirola-Kanai.  So a NOT_VARIATIONAL verdict is
refined by searching a *declared, finite, logged* space of multipliers ``mu = exp(alpha t) t^b
prod_i q_i^c_i prod_i q'_i^d_i`` and re-running the Helmholtz test on ``mu E_i``.  A miss means
"outside the declared multiplier space", never "impossible", and the space is printed in the
receipt so the reader can see exactly what was searched.

Step 4, consequences.  For a constructed ``L``, a declared finite space of point symmetries is
tested for strict variational invariance, and every symmetry found is converted to its Noether
charge, which is then verified conserved *on shell* by substituting the accelerations solved from
the equations themselves.  This is the payoff: a fitted curve becomes "this law conserves the
following quantities", which the fit never contained.

Honesty rules this module obeys.

1. Nothing is transcribed.  Every verdict, Lagrangian, residual, multiplier and charge is
   recomputed symbolically on each run, and the tests re-derive them independently.
2. The class boundary is declared before the test, not after.  Anything outside it is reported
   OUT_OF_DECLARED_CLASS and is given no variationality verdict.
3. Sufficiency is local.  The Helmholtz conditions certify a Lagrangian on a star-shaped
   neighbourhood in the declared domain; they say nothing about global existence, and a Lagrangian
   is never unique (total derivatives and multipliers change it).
4. A NOT_VARIATIONAL verdict is about the system *as declared*, in the reduction in which it was
   handed to the engine.  It is not a statement about some other formulation of the same physics.
5. Nothing here is novel.  Helmholtz's conditions, the Volterra/Tonti homotopy, the
   Caldirola-Kanai Lagrangian and Noether's theorem are nineteenth- and twentieth-century
   mathematics, reproduced as an engine capability.  The applications are re-readings of laws this
   repository already carries, not new physics.
6. No observational dataset is opened.  The planetary law is read from this repository's own
   sealed blind-rediscovery receipt, whose rows are computed from a declared generative rule.

A PASS means the engine decided variationality, built and verified the Lagrangians and derived the
conserved quantities.  It does not mean any candidate theory is correct, and a VARIATIONAL verdict
for a static central force is structurally automatic -- the content in that case is the
closed-form Lagrangian and its consequences, not the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

from .sigma_core import canonical_json_bytes, canonical_sha256
from .sigma_gravity_candidate_gate import CANDIDATE_CONFIG

#: Result schema.  Bump only with a receipt-shape change.
RESULT_SCHEMA = "invariant-inverse-variational-engine-result-1.0"

#: System IR schema.  The engine's only input format.
SYSTEM_IR_SCHEMA = "invariant-second-order-ode-system-ir-1.0"

#: Committed receipt location.
RECEIPT_PATH = "runs/math/inverse-variational/engine-v1.json"

#: Claims block.  Frozen; any change changes the receipt hash and therefore the claim.
CLAIMS: dict[str, bool] = {
    "construction_verified_by_round_trip": True,
    "helmholtz_verdict_is_within_declared_class": True,
    "not_variational_is_a_finding_not_a_failure": True,
    "novelty_claimed": False,
}

#: The class boundary.  Declared *before* any test runs; anything outside it gets no verdict.
DECLARED_CLASS: dict[str, Any] = {
    "class_id": "second_order_ode_system_affine_in_accelerations",
    "independent_variables": "exactly one, the evolution parameter t",
    "dependent_variables": "n >= 1 generalized coordinates q_1 .. q_n",
    "equation_form": "E_i(t, q, q', q'') = 0 for i = 1..n; exactly one equation per coordinate",
    "euler_lagrange_convention": "E_i = d/dt(dL/dq'_i) - dL/dq_i",
    "regularity_requirements": [
        "each E_i is free of q''' and of every higher jet coordinate",
        "each E_i is affine in the accelerations: d^2 E_i / (dq''_j dq''_k) = 0 for all i, j, k",
        "the acceleration matrix W_ij = dE_i/dq''_j is nonsingular: det W is not identically zero",
    ],
    "jet_treatment": (
        "t, q, q', q'', q''' are independent coordinates on the third-order jet; a Helmholtz "
        "condition is satisfied only when its residual vanishes identically as a function on that "
        "jet, never merely on solutions"
    ),
    "helmholtz_conditions": {
        "H1": "dE_i/dq''_j - dE_j/dq''_i = 0",
        "H2": "dE_i/dq'_j + dE_j/dq'_i - 2 d/dt(dE_i/dq''_j) = 0",
        "H3": "dE_i/dq_j - dE_j/dq_i - (1/2) d/dt(dE_i/dq'_j - dE_j/dq'_i) = 0",
    },
    "independent_index_pairs": "i <= j only; H1 and H3 are antisymmetric and H2 is symmetric",
    "sufficiency": (
        "on this class the three conditions are necessary and sufficient for the local existence "
        "of a Lagrangian; sufficiency is local and star-shaped in the declared domain, and the "
        "Lagrangian is never unique (total time derivatives and constant rescalings change it)"
    ),
    "outside_this_class": [
        "partial differential equations and field theories",
        "ordinary differential equations of order other than two",
        "systems not affine in the accelerations",
        "systems with a singular acceleration matrix (degenerate or constrained systems)",
        "systems whose equation count differs from the coordinate count",
    ],
    "outside_verdict": "OUT_OF_DECLARED_CLASS; no variationality verdict is issued",
}

#: Construction methods, in the declared order in which they are tried, with the declared rule
#: that decides whether each one is even applicable.
CONSTRUCTION_METHODS: tuple[dict[str, str], ...] = (
    {
        "method": "volterra_tonti_homotopy",
        "formula": "L_V = int_0^1 q^k E_k(t, lambda q, lambda q', lambda q'') dlambda, then the "
        "acceleration part of L_V is removed as a total time derivative",
        "applicability": "every equation is polynomial in (q, q', q''); a negative power of a "
        "coordinate makes the lambda-integral through the origin divergent",
    },
    {
        "method": "central_potential_ansatz",
        "formula": "L = W(t)|q'|^2/2 - V(r) with V' = phi and E_i = W q''_i + W' q'_i + "
        "phi(t, r) q_i/r",
        "applicability": "the acceleration matrix is W(t) times the identity and the remaining "
        "force is velocity-free and central",
    },
)

#: The multiplier grammar searched in step 3.  Finite, ordered and printed in the receipt so a
#: miss reads as "outside this space", never as "impossible".
MULTIPLIER_SPACE: dict[str, Any] = {
    "form": "mu(t, q, q') = exp(alpha*t) * t^b * prod_i q_i^c_i * prod_i q'_i^d_i",
    "scalar_only": True,
    "integer_exponent_rule": "|b| + sum_i |c_i| + sum_i |d_i| <= total_degree_bound",
    "total_degree_bound": 2,
    "continuous_parameter": "alpha, a single real unknown solved from the Helmholtz residuals",
    "acceptance": (
        "a candidate is accepted only when every Helmholtz residual of mu*E vanishes identically "
        "after substituting the solved alpha, certified symbolically and re-verified by a full "
        "Helmholtz test plus a Lagrangian round trip on the multiplied system"
    ),
    "rejection_filter": (
        "candidates are first rejected by evaluating the raw residuals at declared exact rational "
        "jet points with the physical parameters left symbolic; a nonzero value there is a proof "
        "of non-vanishing, and only survivors are certified symbolically"
    ),
    "not_searched": [
        "matrix (non-scalar) multipliers M_ij E_j, the general Douglas multiplier problem",
        "multipliers depending on the accelerations",
        "multipliers outside the declared monomial-times-exponential grammar",
        "alpha values that are not rational functions of the declared system parameters",
    ],
}

#: The point-symmetry generator space searched in step 4.  Strict variational invariance only.
NOETHER_GENERATOR_SPACE: dict[str, Any] = {
    "generator_form": "X = tau(t,q) d/dt + xi_i(t,q) d/dq_i, prolonged to first order",
    "invariance_tested": (
        "strict invariance X^(1)(L) + L d/dt(tau) = 0; divergence symmetries, whose right-hand "
        "side is a total derivative rather than zero, are NOT searched"
    ),
    "families": [
        "time translation: tau = 1, xi = 0",
        "space translation along each coordinate: tau = 0, xi = e_k",
        "rotation in each coordinate pair (j,k): tau = 0, xi_j = -q_k, xi_k = q_j",
        "scaling: tau = a t, xi_i = b q_i over the declared integer grid",
    ],
    "scaling_grid": {"a": [0, 1, 2, 3], "b": [-3, -2, -1, 1, 2, 3]},
    "charge_convention": "C = sum_i (xi_i - tau q'_i) dL/dq'_i + tau L",
    "conservation_check": (
        "d/dt C is recomputed on the third-order jet and then evaluated on shell by substituting "
        "the accelerations solved from E = 0; it must simplify to exactly zero"
    ),
    "physical_identification": (
        "a charge is named only when it matches a declared reference quantity up to overall sign: "
        "energy H = sum_i q'_i dL/dq'_i - L, momentum p_k = dL/dq'_k, or angular momentum "
        "J_jk = q_j p_k - q_k p_j; the 'minus_' prefix records that the charge convention above "
        "returns -H for the time-translation generator, and conservation is sign-invariant"
    ),
}

#: Repository artifacts the applications are bound to.  JSON is bound by the SHA-256 of its parsed
#: canonical serialization, so a CRLF checkout cannot move a digest.
BOUND_ARTIFACTS: dict[str, dict[str, str]] = {
    "blind_planetary_newton_world_receipt": {
        "path": "runs/math/blind-planetary-laws/newton_inverse_square_law.json",
        "kind": "json",
        "semantic_sha256": "fe55295f181dcbce2587001bdc9b5e04857735e006d3c01d01bd2be634451fb7",
        "role": "the empirically recovered inverse-square law, frozen before its target was opened",
    },
    "sigma_gravity_candidate_config": {
        "path": "src/sigma_theory_compiler/sigma_gravity_candidate_gate.py",
        "kind": "imported_mapping",
        "semantic_sha256": "f6439d897f2a7e9b7c0dcf0cad0ae5816fc3005eed25e0f282e59d23f0947027",
        "role": "the screened-gravity candidate law and its published parameters",
    },
}

#: How many exact rational jet points reject a multiplier candidate before symbolic certification.
PROBE_POINT_COUNT = 3

#: Declared exact rational values used only to witness that a failing residual is not the zero
#: function.  A witness is evidence of non-vanishing; it is never used to certify vanishing.
WITNESS_PARAMETER_VALUES: dict[str, str] = {
    "A": "2",
    "A_amp": "3/2",
    "G": "1",
    "M": "1",
    "g_dagger": "4/5",
    "gamma": "3/5",
    "omega": "7/4",
    "xi": "1/4",
}

#: Declared exact rational coordinate values for the same witness.
WITNESS_COORDINATE_VALUES: tuple[str, ...] = ("3", "4", "5")

_T = sp.Symbol("t", real=True)
_ALPHA = sp.Symbol("alpha_mult", real=True)
_LAMBDA = sp.Symbol("lambda_homotopy", positive=True)
_RADIUS = sp.Symbol("r", positive=True)


class InverseVariationalEngineError(ValueError):
    """Raised on malformed input, a broken known-answer control, or receipt tamper."""


# ---------------------------------------------------------------------------
# The system IR.  The engine's only input format.
# ---------------------------------------------------------------------------


def system_ir(
    system_id: str,
    coordinates: Sequence[str],
    equations: Sequence[str],
    *,
    parameters: Mapping[str, Mapping[str, bool]] | None = None,
    description: str = "",
    provenance: str = "",
    domain: str = "",
) -> dict[str, Any]:
    """Build a system IR document.  Every field is JSON, so the IR is itself a receipt artifact."""

    return {
        "schema_version": SYSTEM_IR_SCHEMA,
        "system_id": system_id,
        "independent_variable": "t",
        "coordinates": list(coordinates),
        "equations": list(equations),
        "parameters": {
            name: dict(sorted(dict(assumptions).items()))
            for name, assumptions in sorted(dict(parameters or {}).items())
        },
        "description": description,
        "provenance": provenance,
        "domain": domain,
    }


class SecondOrderSystem:
    """A parsed system ``E_i(t, q, q', q'') = 0`` together with its third-order jet coordinates."""

    def __init__(
        self, ir: Mapping[str, Any], equations: Sequence[sp.Expr] | None = None
    ) -> None:
        if ir.get("schema_version") != SYSTEM_IR_SCHEMA:
            raise InverseVariationalEngineError("unexpected system IR schema version")
        names = ir.get("coordinates")
        if not isinstance(names, (list, tuple)) or not names:
            raise InverseVariationalEngineError("a system needs at least one coordinate")
        if len(set(names)) != len(names):
            raise InverseVariationalEngineError("duplicate coordinate name")
        declared = ir.get("equations")
        if not isinstance(declared, (list, tuple)) or len(declared) != len(names):
            raise InverseVariationalEngineError(
                "the declared class needs exactly one equation per coordinate"
            )
        self.system_id = str(ir["system_id"])
        self.coordinates = tuple(str(name) for name in names)
        self.positions = tuple(sp.Symbol(name, real=True) for name in self.coordinates)
        self.velocities = tuple(sp.Symbol(f"d{name}", real=True) for name in self.coordinates)
        self.accelerations = tuple(sp.Symbol(f"dd{name}", real=True) for name in self.coordinates)
        self.jerks = tuple(sp.Symbol(f"ddd{name}", real=True) for name in self.coordinates)
        self.parameters = {
            str(name): sp.Symbol(str(name), **dict(assumptions))
            for name, assumptions in sorted(dict(ir.get("parameters", {})).items())
        }
        self.ir = json.loads(json.dumps(dict(ir)))
        if equations is None:
            self.equations = tuple(
                sp.sympify(str(text), locals=self.expression_locals()) for text in declared
            )
        else:
            if len(equations) != len(names):
                raise InverseVariationalEngineError("equation count changed under substitution")
            self.equations = tuple(equations)

    @property
    def size(self) -> int:
        return len(self.coordinates)

    def expression_locals(self) -> dict[str, Any]:
        """The only names an IR equation string may refer to, besides sympy's own functions."""

        mapping: dict[str, Any] = {"t": _T, "r": _RADIUS}
        for index, name in enumerate(self.coordinates):
            mapping[name] = self.positions[index]
            mapping[f"d{name}"] = self.velocities[index]
            mapping[f"dd{name}"] = self.accelerations[index]
            mapping[f"ddd{name}"] = self.jerks[index]
        mapping.update(self.parameters)
        return mapping

    def with_equations(self, equations: Sequence[sp.Expr], system_id: str) -> SecondOrderSystem:
        """Same coordinates and parameters, different equations.  No string round trip."""

        ir = dict(self.ir)
        ir["system_id"] = system_id
        ir["equations"] = [str(equation) for equation in equations]
        return SecondOrderSystem(ir, equations=list(equations))


# ---------------------------------------------------------------------------
# Jet calculus.
# ---------------------------------------------------------------------------


def total_derivative(expression: sp.Expr, system: SecondOrderSystem) -> sp.Expr:
    """Total ``d/dt`` on the third-order jet, where t, q, q', q'', q''' are independent."""

    if any(expression.has(symbol) for symbol in system.jerks):
        raise InverseVariationalEngineError(
            "total derivative would need fourth-order jet coordinates"
        )
    result = sp.diff(expression, _T)
    for index in range(system.size):
        result += system.velocities[index] * sp.diff(expression, system.positions[index])
        result += system.accelerations[index] * sp.diff(expression, system.velocities[index])
        result += system.jerks[index] * sp.diff(expression, system.accelerations[index])
    return result


def acceleration_matrix(system: SecondOrderSystem) -> sp.Matrix:
    """``W_ij = dE_i/dq''_j``."""

    return sp.Matrix(
        system.size,
        system.size,
        lambda i, j: sp.diff(system.equations[i], system.accelerations[j]),
    )


def class_check(system: SecondOrderSystem) -> dict[str, Any]:
    """Decide membership of :data:`DECLARED_CLASS` before any verdict is issued."""

    reasons: list[str] = []
    for index, equation in enumerate(system.equations):
        if any(equation.has(symbol) for symbol in system.jerks):
            reasons.append(f"equation {index + 1} contains a third-order jet coordinate")
    matrix = acceleration_matrix(system)
    for i in range(system.size):
        for j in range(system.size):
            for k in range(system.size):
                if sp.simplify(sp.diff(matrix[i, j], system.accelerations[k])) != 0:
                    reasons.append(
                        f"equation {i + 1} is not affine in the accelerations: "
                        f"d^2E/(dq''_{j + 1} dq''_{k + 1}) is nonzero"
                    )
    determinant = sp.simplify(matrix.det())
    if determinant == 0:
        reasons.append("the acceleration matrix is singular, so the system is degenerate")
    ordered: list[str] = []
    for reason in reasons:
        if reason not in ordered:
            ordered.append(reason)
    return {
        "in_declared_class": not ordered,
        "reasons_outside": ordered,
        "acceleration_matrix": [
            [str(matrix[i, j]) for j in range(system.size)] for i in range(system.size)
        ],
        "acceleration_matrix_determinant": str(determinant),
    }


# ---------------------------------------------------------------------------
# Step 1: the Helmholtz test.
# ---------------------------------------------------------------------------


def helmholtz_residuals(system: SecondOrderSystem) -> list[tuple[str, int, int, sp.Expr]]:
    """Raw (unsimplified) Helmholtz residuals over the independent index pairs ``i <= j``."""

    equations = system.equations
    out: list[tuple[str, int, int, sp.Expr]] = []
    for i in range(system.size):
        for j in range(i, system.size):
            w_ij = sp.diff(equations[i], system.accelerations[j])
            w_ji = sp.diff(equations[j], system.accelerations[i])
            out.append(("H1", i, j, w_ij - w_ji))
            out.append(
                (
                    "H2",
                    i,
                    j,
                    sp.diff(equations[i], system.velocities[j])
                    + sp.diff(equations[j], system.velocities[i])
                    - 2 * total_derivative(w_ij, system),
                )
            )
            antisymmetric = sp.diff(equations[i], system.velocities[j]) - sp.diff(
                equations[j], system.velocities[i]
            )
            out.append(
                (
                    "H3",
                    i,
                    j,
                    sp.diff(equations[i], system.positions[j])
                    - sp.diff(equations[j], system.positions[i])
                    - sp.Rational(1, 2) * total_derivative(antisymmetric, system),
                )
            )
    return out


def normalized_residual(raw: sp.Expr) -> tuple[sp.Expr, str]:
    """Return ``(simplified, printable)`` for one raw Helmholtz residual.

    The zero test is done by :func:`sympy.simplify` alone, so nothing about the verdict depends
    on the second pass.  A residual that survives is additionally factored purely for the
    receipt, because a factored obstruction says which term produced it: for the coherence
    reduction below it exposes the obstruction as proportional to the coherence scale and to the
    height off the mid-plane, which the expanded form hides.
    """

    simplified = sp.simplify(raw)
    if simplified == 0:
        return simplified, "0"
    return simplified, str(sp.factor(simplified))


def _witness_substitution(system: SecondOrderSystem) -> dict[sp.Symbol, sp.Expr]:
    values: dict[sp.Symbol, sp.Expr] = {_T: sp.Rational(2, 3)}
    for index in range(system.size):
        coordinate = WITNESS_COORDINATE_VALUES[index % len(WITNESS_COORDINATE_VALUES)]
        values[system.positions[index]] = sp.Rational(coordinate)
        values[system.velocities[index]] = sp.Rational(5 + index, 7)
        values[system.accelerations[index]] = sp.Rational(2 + index, 5)
        values[system.jerks[index]] = sp.Rational(4 + index, 9)
    for name, symbol in system.parameters.items():
        if name not in WITNESS_PARAMETER_VALUES:
            raise InverseVariationalEngineError(f"no declared witness value for parameter {name}")
        values[symbol] = sp.Rational(WITNESS_PARAMETER_VALUES[name])
    return values


def helmholtz_test(system: SecondOrderSystem) -> dict[str, Any]:
    """Step 1.  Return the verdict together with every condition and its residual."""

    membership = class_check(system)
    if not membership["in_declared_class"]:
        return {
            "verdict": "OUT_OF_DECLARED_CLASS",
            "class_check": membership,
            "conditions": [],
            "conditions_checked": 0,
            "failing_conditions": [],
        }
    witness = _witness_substitution(system)
    conditions: list[dict[str, Any]] = []
    failing: list[dict[str, Any]] = []
    for kind, i, j, raw in helmholtz_residuals(system):
        residual, printable = normalized_residual(raw)
        satisfied = residual == 0
        entry: dict[str, Any] = {
            "condition": kind,
            "index_i": i + 1,
            "index_j": j + 1,
            "residual": printable,
            "satisfied": satisfied,
        }
        if not satisfied:
            value = sp.simplify(residual.subs(witness, simultaneous=True))
            if value == 0:
                raise InverseVariationalEngineError(
                    "a failing Helmholtz residual vanished at the declared witness point"
                )
            entry["nonvanishing_witness"] = str(value)
            failing.append(dict(entry))
        conditions.append(entry)
    return {
        "verdict": "VARIATIONAL" if not failing else "NOT_VARIATIONAL",
        "class_check": membership,
        "conditions": conditions,
        "conditions_checked": len(conditions),
        "failing_conditions": failing,
    }


# ---------------------------------------------------------------------------
# Step 2: construction, and the round trip that proves it.
# ---------------------------------------------------------------------------


def euler_lagrange(lagrangian: sp.Expr, system: SecondOrderSystem) -> list[sp.Expr]:
    """``E_i = d/dt(dL/dq'_i) - dL/dq_i`` for a first-order ``L``."""

    return [
        total_derivative(sp.diff(lagrangian, system.velocities[i]), system)
        - sp.diff(lagrangian, system.positions[i])
        for i in range(system.size)
    ]


def round_trip_residuals(lagrangian: sp.Expr, system: SecondOrderSystem) -> list[sp.Expr]:
    """Differentiate the constructed Lagrangian back and subtract the input equations."""

    return [
        sp.simplify(derived - system.equations[i])
        for i, derived in enumerate(euler_lagrange(lagrangian, system))
    ]


def _is_polynomial_in_jet(system: SecondOrderSystem) -> bool:
    generators = system.positions + system.velocities + system.accelerations
    for equation in system.equations:
        try:
            sp.Poly(equation, *generators)
        except (sp.PolynomialError, TypeError, AttributeError):
            return False
    return True


def construct_by_homotopy(system: SecondOrderSystem) -> sp.Expr | None:
    """Volterra/Tonti homotopy, reduced to a first-order Lagrangian.  ``None`` if inapplicable."""

    if not _is_polynomial_in_jet(system):
        return None
    scaling: dict[sp.Symbol, sp.Expr] = {}
    for index in range(system.size):
        scaling[system.positions[index]] = _LAMBDA * system.positions[index]
        scaling[system.velocities[index]] = _LAMBDA * system.velocities[index]
        scaling[system.accelerations[index]] = _LAMBDA * system.accelerations[index]
    integrand = sum(
        system.positions[index] * system.equations[index].subs(scaling, simultaneous=True)
        for index in range(system.size)
    )
    volterra = sp.integrate(sp.expand(integrand), (_LAMBDA, 0, 1))
    if volterra.has(sp.Integral, sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None
    volterra = sp.simplify(volterra)
    constant_part = volterra.subs(
        {symbol: 0 for symbol in system.accelerations}, simultaneous=True
    )
    linear_part = [sp.diff(volterra, system.accelerations[i]) for i in range(system.size)]
    remainder = sp.simplify(
        volterra
        - constant_part
        - sum(linear_part[i] * system.accelerations[i] for i in range(system.size))
    )
    if remainder != 0:
        return None
    velocity_scaling = {
        system.velocities[i]: _LAMBDA * system.velocities[i] for i in range(system.size)
    }
    boundary = sp.integrate(
        sum(
            system.velocities[i] * linear_part[i].subs(velocity_scaling, simultaneous=True)
            for i in range(system.size)
        ),
        (_LAMBDA, 0, 1),
    )
    if boundary.has(sp.Integral, sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None
    reduced = (
        constant_part
        - sp.diff(boundary, _T)
        - sum(
            system.velocities[i] * sp.diff(boundary, system.positions[i])
            for i in range(system.size)
        )
    )
    return sp.simplify(-reduced)


def construct_by_central_potential(system: SecondOrderSystem) -> tuple[sp.Expr, sp.Expr] | None:
    """Declared ansatz ``L = W(t)|q'|^2/2 - V``, for a central force with no velocity coupling.

    Applies when ``E_i = W(t) q''_i + W'(t) q'_i + G_i(t, q)`` with ``G_i = phi(t, r) q_i / r``.
    The homotopy cannot reach these systems: contracting the configuration to the origin runs the
    integrand through the force's singularity there, so its lambda-integral diverges.
    """

    weight = sp.simplify(sp.diff(system.equations[0], system.accelerations[0]))
    if weight == 0 or weight.free_symbols - {_T}:
        return None
    for i in range(system.size):
        for j in range(system.size):
            expected = weight if i == j else sp.Integer(0)
            if sp.simplify(sp.diff(system.equations[i], system.accelerations[j]) - expected) != 0:
                return None
    residual_force = [
        sp.simplify(
            system.equations[i]
            - weight * system.accelerations[i]
            - sp.diff(weight, _T) * system.velocities[i]
        )
        for i in range(system.size)
    ]
    for force in residual_force:
        if any(force.has(symbol) for symbol in system.velocities + system.accelerations):
            return None
    radius = sp.sqrt(sum(symbol**2 for symbol in system.positions))
    magnitude = sp.simplify(
        sum(system.positions[i] * residual_force[i] for i in range(system.size)) / radius
    )
    for i in range(system.size):
        if sp.simplify(residual_force[i] - magnitude * system.positions[i] / radius) != 0:
            return None
    ray: dict[sp.Symbol, sp.Expr] = {system.positions[0]: _RADIUS}
    ray.update({system.positions[i]: sp.Integer(0) for i in range(1, system.size)})
    radial = sp.simplify(magnitude.subs(ray, simultaneous=True))
    potential = sp.simplify(sp.integrate(radial, _RADIUS))
    if potential.has(sp.Integral, sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None
    limit_at_infinity = sp.limit(potential, _RADIUS, sp.oo)
    if limit_at_infinity.is_finite:
        potential = sp.simplify(potential - limit_at_infinity)
    kinetic = weight * sum(symbol**2 for symbol in system.velocities) / 2
    return sp.simplify(kinetic - potential.subs(_RADIUS, radius)), potential


def construct_lagrangian(
    system: SecondOrderSystem,
) -> tuple[dict[str, Any], sp.Expr | None, sp.Expr | None]:
    """Step 2.  Build a Lagrangian, then prove it by differentiating back."""

    attempts: list[dict[str, Any]] = []
    for declared in CONSTRUCTION_METHODS:
        method = declared["method"]
        radial_potential: sp.Expr | None = None
        if method == "volterra_tonti_homotopy":
            candidate = construct_by_homotopy(system)
        else:
            built = construct_by_central_potential(system)
            candidate = None if built is None else built[0]
            radial_potential = None if built is None else built[1]
        if candidate is None:
            attempts.append({"method": method, "outcome": "not_applicable"})
            continue
        residuals = round_trip_residuals(candidate, system)
        if any(residual != 0 for residual in residuals):
            attempts.append(
                {
                    "method": method,
                    "outcome": "round_trip_failed",
                    "round_trip_residual": [str(residual) for residual in residuals],
                }
            )
            continue
        attempts.append({"method": method, "outcome": "accepted"})
        report: dict[str, Any] = {
            "constructed": True,
            "method": method,
            "methods_tried": attempts,
            "lagrangian": str(candidate),
            "round_trip_residual": [str(residual) for residual in residuals],
            "round_trip_verified": True,
        }
        if radial_potential is not None:
            report["radial_potential"] = str(radial_potential)
        return report, candidate, radial_potential
    return (
        {
            "constructed": False,
            "method": None,
            "methods_tried": attempts,
            "lagrangian": None,
            "round_trip_residual": None,
            "round_trip_verified": False,
        },
        None,
        None,
    )


# ---------------------------------------------------------------------------
# Step 3: the integrating-factor (multiplier) search.
# ---------------------------------------------------------------------------


def _probe_points(system: SecondOrderSystem) -> list[dict[sp.Symbol, sp.Expr]]:
    points: list[dict[sp.Symbol, sp.Expr]] = []
    for step in range(PROBE_POINT_COUNT):
        point: dict[sp.Symbol, sp.Expr] = {_T: sp.Rational(2 + step, 3)}
        for index in range(system.size):
            point[system.positions[index]] = sp.Rational(3 + index + 2 * step, 4)
            point[system.velocities[index]] = sp.Rational(5 + index + step, 7)
            point[system.accelerations[index]] = sp.Rational(2 + index + step, 5)
            point[system.jerks[index]] = sp.Rational(4 + index + step, 9)
        points.append(point)
    return points


def _alpha_candidates(expression: sp.Expr, time_value: sp.Expr) -> list[sp.Expr] | None:
    """Solve one probed residual for ``alpha``; ``None`` means the candidate is rejected."""

    stripped = sp.expand(sp.simplify(expression / sp.exp(_ALPHA * time_value)))
    if stripped.has(sp.exp):
        return None
    try:
        polynomial = sp.Poly(stripped, _ALPHA)
    except (sp.PolynomialError, TypeError, AttributeError):
        return None
    if not 1 <= polynomial.total_degree() <= 2:
        return None
    try:
        roots = sp.solve(polynomial.as_expr(), _ALPHA, dict=False)
    except (NotImplementedError, TypeError):
        return None
    return sorted((sp.simplify(root) for root in roots), key=str)


def multiplier_exponent_tuples(size: int, bound: int) -> list[tuple[int, ...]]:
    """The declared, finite, ordered multiplier exponent grid."""

    return [
        tuple(entry)
        for entry in itertools.product(range(-bound, bound + 1), repeat=1 + 2 * size)
        if sum(abs(value) for value in entry) <= bound
    ]


def multiplier_expression(system: SecondOrderSystem, exponents: Sequence[int]) -> sp.Expr:
    """``mu = exp(alpha t) t^b prod_i q_i^c_i prod_i q'_i^d_i`` for one exponent tuple."""

    multiplier = sp.exp(_ALPHA * _T) * _T ** exponents[0]
    for index in range(system.size):
        multiplier *= system.positions[index] ** exponents[1 + index]
        multiplier *= system.velocities[index] ** exponents[1 + system.size + index]
    return multiplier


def search_integrating_factors(system: SecondOrderSystem) -> dict[str, Any]:
    """Step 3.  Search the declared space for multipliers that make the system variational."""

    bound = int(MULTIPLIER_SPACE["total_degree_bound"])
    grid = multiplier_exponent_tuples(system.size, bound)
    points = _probe_points(system)
    allowed = set(system.parameters.values())
    found: list[dict[str, Any]] = []
    rejected = 0
    for exponents in grid:
        multiplier = multiplier_expression(system, exponents)
        scaled = system.with_equations(
            [multiplier * equation for equation in system.equations],
            f"{system.system_id}__scaled",
        )
        residuals = [raw for _, _, _, raw in helmholtz_residuals(scaled) if raw != 0]
        candidates: set[sp.Expr] | None = None
        dead = False
        for raw in residuals:
            probed = raw.subs(points[0], simultaneous=True)
            if probed == 0:
                continue
            solved = _alpha_candidates(probed, points[0][_T])
            if solved is None:
                dead = True
                break
            candidates = set(solved) if candidates is None else candidates & set(solved)
            if not candidates:
                dead = True
                break
        if dead or candidates is None:
            rejected += 1
            continue
        accepted = False
        for value in sorted(candidates, key=str):
            if value.free_symbols - allowed:
                continue
            if not all(
                sp.simplify(raw.subs(_ALPHA, value).subs(point, simultaneous=True)) == 0
                for point in points
                for raw in residuals
            ):
                continue
            certified = system.with_equations(
                [
                    sp.simplify(multiplier.subs(_ALPHA, value) * equation)
                    for equation in system.equations
                ],
                f"{system.system_id}__times_multiplier",
            )
            verdict = helmholtz_test(certified)
            if verdict["verdict"] != "VARIATIONAL":
                continue
            report, _, _ = construct_lagrangian(certified)
            if not report["round_trip_verified"]:
                raise InverseVariationalEngineError(
                    "a certified multiplier produced a Lagrangian that failed its round trip"
                )
            found.append(
                {
                    "exponents": {
                        "t": exponents[0],
                        **{
                            system.coordinates[i]: exponents[1 + i] for i in range(system.size)
                        },
                        **{
                            f"d{system.coordinates[i]}": exponents[1 + system.size + i]
                            for i in range(system.size)
                        },
                    },
                    "alpha": str(value),
                    "multiplier": str(sp.simplify(multiplier.subs(_ALPHA, value))),
                    "multiplied_equations": [str(equation) for equation in certified.equations],
                    "helmholtz_verdict": verdict["verdict"],
                    "construction_method": report["method"],
                    "lagrangian": report["lagrangian"],
                    "round_trip_residual": report["round_trip_residual"],
                }
            )
            accepted = True
            break
        if not accepted:
            rejected += 1
    return {
        "space": dict(MULTIPLIER_SPACE),
        "space_size": len(grid),
        "candidates_rejected": rejected,
        "found": found,
        "outcome": "MULTIPLIER_FOUND" if found else "NO_MULTIPLIER_IN_DECLARED_SPACE",
    }


# ---------------------------------------------------------------------------
# Step 4: symmetries and Noether charges.
# ---------------------------------------------------------------------------


def noether_generators(system: SecondOrderSystem) -> list[dict[str, Any]]:
    """The declared finite generator space, in a fixed order."""

    size = system.size
    zero = [sp.Integer(0)] * size
    generators: list[dict[str, Any]] = [
        {"generator_id": "time_translation", "tau": sp.Integer(1), "xi": list(zero)}
    ]
    for index in range(size):
        xi = list(zero)
        xi[index] = sp.Integer(1)
        generators.append(
            {
                "generator_id": f"space_translation_{system.coordinates[index]}",
                "tau": sp.Integer(0),
                "xi": xi,
            }
        )
    for first in range(size):
        for second in range(first + 1, size):
            xi = list(zero)
            xi[first] = -system.positions[second]
            xi[second] = system.positions[first]
            generators.append(
                {
                    "generator_id": (
                        f"rotation_{system.coordinates[first]}_{system.coordinates[second]}"
                    ),
                    "tau": sp.Integer(0),
                    "xi": xi,
                }
            )
    grid = NOETHER_GENERATOR_SPACE["scaling_grid"]
    for a_value in grid["a"]:
        for b_value in grid["b"]:
            generators.append(
                {
                    "generator_id": f"scaling_a{a_value}_b{b_value}",
                    "tau": sp.Integer(a_value) * _T,
                    "xi": [sp.Integer(b_value) * symbol for symbol in system.positions],
                }
            )
    return generators


def invariance_defect(
    lagrangian: sp.Expr, tau: sp.Expr, xi: Sequence[sp.Expr], system: SecondOrderSystem
) -> sp.Expr:
    """``X^(1)(L) + L d/dt(tau)``; zero exactly when the generator is a strict symmetry."""

    defect = tau * sp.diff(lagrangian, _T)
    for index in range(system.size):
        defect += xi[index] * sp.diff(lagrangian, system.positions[index])
        prolonged = total_derivative(xi[index], system) - system.velocities[
            index
        ] * total_derivative(tau, system)
        defect += prolonged * sp.diff(lagrangian, system.velocities[index])
    return sp.simplify(defect + lagrangian * total_derivative(tau, system))


def noether_charge(
    lagrangian: sp.Expr, tau: sp.Expr, xi: Sequence[sp.Expr], system: SecondOrderSystem
) -> sp.Expr:
    """``C = sum_i (xi_i - tau q'_i) dL/dq'_i + tau L``."""

    charge = tau * lagrangian
    for index in range(system.size):
        charge += (xi[index] - tau * system.velocities[index]) * sp.diff(
            lagrangian, system.velocities[index]
        )
    return sp.simplify(charge)


def solve_accelerations(system: SecondOrderSystem) -> dict[sp.Symbol, sp.Expr]:
    """Solve ``E = 0`` for the accelerations; the declared class guarantees this is possible."""

    matrix = acceleration_matrix(system)
    offset = sp.Matrix(
        system.size,
        1,
        lambda i, _: system.equations[i].subs(
            {symbol: 0 for symbol in system.accelerations}, simultaneous=True
        ),
    )
    solution = matrix.LUsolve(-offset)
    return {
        system.accelerations[index]: sp.simplify(solution[index])
        for index in range(system.size)
    }


def identify_charge(
    charge: sp.Expr, lagrangian: sp.Expr, system: SecondOrderSystem
) -> str | None:
    """Name a charge only when it matches a declared reference quantity up to overall sign."""

    references: list[tuple[str, sp.Expr]] = [
        (
            "energy",
            sp.simplify(
                sum(
                    system.velocities[index] * sp.diff(lagrangian, system.velocities[index])
                    for index in range(system.size)
                )
                - lagrangian
            ),
        )
    ]
    for index in range(system.size):
        references.append(
            (
                f"momentum_{system.coordinates[index]}",
                sp.diff(lagrangian, system.velocities[index]),
            )
        )
    for first in range(system.size):
        for second in range(first + 1, system.size):
            references.append(
                (
                    f"angular_momentum_{system.coordinates[first]}_{system.coordinates[second]}",
                    sp.simplify(
                        system.positions[first]
                        * sp.diff(lagrangian, system.velocities[second])
                        - system.positions[second]
                        * sp.diff(lagrangian, system.velocities[first])
                    ),
                )
            )
    for name, reference in references:
        if reference == 0:
            continue
        if sp.simplify(charge - reference) == 0:
            return name
        if sp.simplify(charge + reference) == 0:
            return f"minus_{name}"
    return None


def noether_analysis(lagrangian: sp.Expr, system: SecondOrderSystem) -> dict[str, Any]:
    """Step 4.  Find strict variational symmetries and verify their charges on shell."""

    on_shell = solve_accelerations(system)
    generators = noether_generators(system)
    symmetries: list[dict[str, Any]] = []
    for generator in generators:
        if invariance_defect(lagrangian, generator["tau"], generator["xi"], system) != 0:
            continue
        charge = noether_charge(lagrangian, generator["tau"], generator["xi"], system)
        if charge == 0:
            continue
        conservation = sp.simplify(
            total_derivative(charge, system).subs(on_shell, simultaneous=True)
        )
        if conservation != 0:
            raise InverseVariationalEngineError(
                f"a Noether charge for {generator['generator_id']} is not conserved on shell"
            )
        symmetries.append(
            {
                "generator_id": generator["generator_id"],
                "tau": str(generator["tau"]),
                "xi": [str(component) for component in generator["xi"]],
                "invariance_defect": "0",
                "conserved_quantity": str(charge),
                "physical_identification": identify_charge(charge, lagrangian, system),
                "on_shell_time_derivative": "0",
            }
        )
    return {
        "generator_space": dict(NOETHER_GENERATOR_SPACE),
        "generators_tested": len(generators),
        "symmetries_found": len(symmetries),
        "symmetries": symmetries,
        "conserved_quantity_count": len(symmetries),
    }


# ---------------------------------------------------------------------------
# Reading the empirically recovered law out of the sealed blind-rediscovery receipt.
# ---------------------------------------------------------------------------


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() != path and root.resolve() not in path.parents:
        raise InverseVariationalEngineError(f"bound path escapes repository root: {relative}")
    return path


def read_recovered_planetary_law(root: Path) -> dict[str, Any]:
    """Read the blind campaign's *pre-unseal* candidate: the law recovered from the data alone."""

    binding = BOUND_ARTIFACTS["blind_planetary_newton_world_receipt"]
    path = _resolve(root, binding["path"])
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InverseVariationalEngineError(
            f"cannot read bound artifact: {binding['path']}"
        ) from exc
    if _semantic_sha256(document) != binding["semantic_sha256"]:
        raise InverseVariationalEngineError(f"bound artifact hash mismatch: {binding['path']}")
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    if document.get("content_sha256") != canonical_sha256(body):
        raise InverseVariationalEngineError("the bound blind-rediscovery receipt seal is broken")
    candidate = document["phase_a"]["candidate"]
    if candidate.get("kind") != "power_law":
        raise InverseVariationalEngineError("the recovered candidate is not a power law")
    exponent = int(sp.Integer(sp.sympify(candidate["exponent"])))
    constant = sp.Rational(
        int(candidate["constant"]["numerator"]), int(candidate["constant"]["denominator"])
    )
    if not constant.is_positive:
        raise InverseVariationalEngineError("the recovered amplitude is not positive")
    return {
        "source_receipt": binding["path"],
        "source_semantic_sha256": binding["semantic_sha256"],
        "world_id": document["world_id"],
        "recovered_before_the_target_was_unsealed": True,
        "candidate_statement": document["phase_a"]["candidate_statement"],
        "candidate_source_stage": candidate["source_stage"],
        "public_row_count": len(document["public_rows"]),
        "public_rows_sha256": document["phase_a"]["public_rows_sha256"],
        "exponent": exponent,
        "constant_text": f"Rational({constant.p}, {constant.q})",
        "constant_decimal_20": str(sp.N(constant, 20)),
        "campaign_verdict": document["unseal"]["verdict"],
        "column_meanings": document["unseal"]["column_meanings"],
        "constant_matches_four_pi_squared_to_40_places": bool(
            abs(sp.N(constant - 4 * sp.pi**2, 60)) < sp.Rational(1, 10**40)
        ),
        "declared_lift": (
            "the campaign recovered the scalar relation 'response = A x1^p' between two "
            "anonymized columns.  Lifting it to an equation of motion adds one declared modelling "
            "assumption the data did not force: that the response is the magnitude of a "
            "centripetal acceleration directed along the line to the source, so that "
            "q'' = -A r^p q/r.  Centrality and isotropy are assumptions here, not findings."
        ),
    }


# ---------------------------------------------------------------------------
# The systems the engine is run on.
# ---------------------------------------------------------------------------


def _central_power_law_ir(
    amplitude: str, exponent: int, system_id: str, provenance: str, description: str
) -> dict[str, Any]:
    """``q''_i + A r^(p-1) q_i = 0``: the central law whose acceleration magnitude is ``A r^p``."""

    radius = "sqrt(q1**2 + q2**2 + q3**2)"
    return system_ir(
        system_id,
        ["q1", "q2", "q3"],
        [
            f"ddq{index} + ({amplitude})*({radius})**({exponent - 1})*q{index}"
            for index in (1, 2, 3)
        ],
        parameters={"A": {"positive": True}} if amplitude == "A" else {},
        description=description,
        provenance=provenance,
        domain="r > 0; the source at the origin is excluded",
    )


def sigma_gravity_effective_acceleration(coherence: str, radius: str) -> str:
    """``g_eff = g_N (1 + A_amp C h(g_N))`` with ``g_N = G M / r^2`` and the published ``h``."""

    newtonian = f"G*M/({radius})**2"
    shape = f"sqrt(g_dagger/({newtonian}))*g_dagger/(g_dagger + {newtonian})"
    return f"({newtonian})*(1 + A_amp*({coherence})*({shape}))"


_SPHERICAL_RADIUS = "sqrt(q1**2 + q2**2 + q3**2)"
_MERIDIONAL_RADIUS = "sqrt(qR**2 + qz**2)"

_SIGMA_PARAMETERS = {
    "A_amp": {"positive": True},
    "G": {"positive": True},
    "M": {"positive": True},
    "g_dagger": {"positive": True},
}


def declared_systems(root: Path) -> list[dict[str, Any]]:
    """Every system the engine runs, with the declared expectation each one must meet."""

    recovered = read_recovered_planetary_law(root)
    meridional = sigma_gravity_effective_acceleration("qR/(xi + qR)", _MERIDIONAL_RADIUS)
    return [
        {
            "role": "control",
            "ir": _central_power_law_ir(
                "A",
                -2,
                "newtonian_inverse_square",
                "textbook; the law derivation_chain_demo derives from the vacuum Laplace equation",
                "the Newtonian inverse-square law as a Cartesian equation of motion",
            ),
            "expected": {
                "helmholtz_verdict": "VARIATIONAL",
                "round_trip_verified": True,
                "required_conserved_quantities": [
                    "angular_momentum_q1_q2",
                    "angular_momentum_q1_q3",
                    "angular_momentum_q2_q3",
                    "minus_energy",
                ],
                "forbidden_conserved_quantities": ["momentum_q1", "momentum_q2", "momentum_q3"],
            },
        },
        {
            "role": "control",
            "ir": system_ir(
                "harmonic_oscillator",
                ["q1"],
                ["ddq1 + omega**2*q1"],
                parameters={"omega": {"positive": True}},
                description="the undamped harmonic oscillator",
                provenance="textbook",
                domain="the whole real line",
            ),
            "expected": {
                "helmholtz_verdict": "VARIATIONAL",
                "round_trip_verified": True,
                "required_conserved_quantities": ["minus_energy"],
                "forbidden_conserved_quantities": ["momentum_q1"],
            },
        },
        {
            "role": "control",
            "ir": system_ir(
                "damped_oscillator",
                ["q1"],
                ["ddq1 + gamma*dq1 + omega**2*q1"],
                parameters={"gamma": {"positive": True}, "omega": {"positive": True}},
                description=(
                    "the damped harmonic oscillator: not variational, but variational after "
                    "multiplication by exp(gamma t), which is the sharpest control in this module"
                ),
                provenance="textbook; Caldirola (1941) and Kanai (1948)",
                domain="the whole real line",
            ),
            "finding": (
                "Friction is not a least-action force.  The single Helmholtz condition with any "
                "content in one dimension, H2, fails with residual 2*gamma, which is exactly the "
                "damping rate: the obstruction is the dissipation itself.  Multiplying the "
                "equation by exp(gamma t) removes it and returns the Caldirola-Kanai Lagrangian, "
                "so the system is variational in a rescaled time weight rather than not "
                "variational at all.  The constructed Lagrangian depends explicitly on t, so time "
                "translation is not a symmetry of it and the engine reports no conserved energy -- "
                "which is the correct physics for a damped oscillator, not a gap in the search."
            ),
            "expected": {
                "helmholtz_verdict": "NOT_VARIATIONAL",
                "failing_condition": "H2",
                "failing_residual": "2*gamma",
                "multiplier_outcome": "MULTIPLIER_FOUND",
                "multiplier": "exp(gamma*t)",
                "multiplier_round_trip_verified": True,
                "required_conserved_quantities": [],
            },
        },
        {
            "role": "control",
            "ir": system_ir(
                "nonconservative_planar_curl",
                ["q1", "q2"],
                ["ddq1 - q2", "ddq2 + q1"],
                description=(
                    "a deliberately non-variational planar system: the force field carries a "
                    "constant curl, and no scalar multiplier in the declared space removes it"
                ),
                provenance="constructed here as a negative control",
                domain="the whole plane",
            ),
            "finding": (
                "The force field has constant curl, so H3 fails with residual -2 and no scalar "
                "multiplier in the declared space repairs it.  This is the honest negative: the "
                "verdict names the failing condition and the receipt names the 61 multipliers "
                "that were searched, so the result reads 'no multiplier in this declared space', "
                "never 'no multiplier exists'."
            ),
            "expected": {
                "helmholtz_verdict": "NOT_VARIATIONAL",
                "failing_condition": "H3",
                "failing_residual": "-2",
                "multiplier_outcome": "NO_MULTIPLIER_IN_DECLARED_SPACE",
                "required_conserved_quantities": [],
            },
        },
        {
            "role": "control",
            "ir": system_ir(
                "quadratic_in_acceleration",
                ["q1"],
                ["ddq1**2 - q1"],
                description="quadratic in the acceleration, so outside the declared class",
                provenance="constructed here as a class-boundary control",
                domain="the whole real line",
            ),
            "expected": {
                "helmholtz_verdict": "OUT_OF_DECLARED_CLASS",
                "required_conserved_quantities": [],
            },
        },
        {
            "role": "application",
            "ir": _central_power_law_ir(
                recovered["constant_text"],
                recovered["exponent"],
                "blind_recovered_inverse_square",
                (
                    "recovered from anonymized columns by "
                    "blind_planetary_law_rediscovery_campaign; read from its sealed world receipt "
                    "and lifted to an equation of motion by the declared centrality assumption"
                ),
                "the empirically recovered planetary law, as an equation of motion",
            ),
            "recovered_law": recovered,
            "finding": (
                "The complete backwards demonstration: anonymized rational columns produced a "
                "power law with no physics in it, the declared centrality lift turned that law "
                "into equations of motion, Helmholtz certified them variational, the engine built "
                "L = |q'|^2/2 + A/r and proved it by differentiating back to residual zero, and "
                "Noether then returned conservation of energy and of all three components of "
                "angular momentum.  Those conservation laws are physical content the curve fit "
                "never contained."
            ),
            "expected": {
                "helmholtz_verdict": "VARIATIONAL",
                "round_trip_verified": True,
                "required_conserved_quantities": [
                    "angular_momentum_q1_q2",
                    "angular_momentum_q1_q3",
                    "angular_momentum_q2_q3",
                    "minus_energy",
                ],
                "forbidden_conserved_quantities": ["momentum_q1", "momentum_q2", "momentum_q3"],
            },
        },
        {
            "role": "application",
            "ir": system_ir(
                "sigma_gravity_spherical_quasistatic",
                ["q1", "q2", "q3"],
                [
                    f"ddq{index} + "
                    f"({sigma_gravity_effective_acceleration('1', _SPHERICAL_RADIUS)})"
                    f"*q{index}/({_SPHERICAL_RADIUS})"
                    for index in (1, 2, 3)
                ],
                parameters=_SIGMA_PARAMETERS,
                description=(
                    "the screened-gravity candidate in its quasistatic spherically symmetric "
                    "reduction: a test particle in the static g_eff field of a point source, at "
                    "the declared coherence bound C = 1 that the candidate gate uses for clusters"
                ),
                provenance="sigma_theory_compiler.sigma_gravity_candidate_gate.CANDIDATE_CONFIG",
                domain="r > 0; the source is excluded",
            ),
            "finding": (
                "VARIATIONAL, with a closed-form Lagrangian.  The verdict itself is structurally "
                "automatic -- any force depending only on r is a gradient -- so the content is the "
                "potential the engine integrates out: a Newtonian -G M/r term plus a logarithmic "
                "tail A_amp sqrt(G M g_dagger) log(G M + g_dagger r^2)/2.  Energy and all three "
                "angular momentum components are conserved, and the logarithmic tail makes the "
                "circular speed tend to a constant, whose fourth power is A_amp^2 G M g_dagger: "
                "an asymptotically flat rotation curve with v^4 proportional to the source mass, "
                "which is the baryonic Tully-Fisher shape.  That is a derived consequence of the "
                "constructed action, not a fit, and it is a known property of this class of laws."
            ),
            "expected": {
                "helmholtz_verdict": "VARIATIONAL",
                "round_trip_verified": True,
                "required_conserved_quantities": [
                    "angular_momentum_q1_q2",
                    "angular_momentum_q1_q3",
                    "angular_momentum_q2_q3",
                    "minus_energy",
                ],
                "forbidden_conserved_quantities": ["momentum_q1", "momentum_q2", "momentum_q3"],
            },
        },
        {
            "role": "application",
            "ir": system_ir(
                "sigma_gravity_meridional_coherence",
                ["qR", "qz"],
                [
                    f"ddqR + ({meridional})*qR/({_MERIDIONAL_RADIUS})",
                    f"ddqz + ({meridional})*qz/({_MERIDIONAL_RADIUS})",
                ],
                parameters={**_SIGMA_PARAMETERS, "xi": {"positive": True}},
                description=(
                    "the same candidate with the published disk coherence C = W(R) = R/(xi + R) "
                    "restored, reduced to the meridional (R, z) plane at zero angular momentum"
                ),
                provenance="sigma_theory_compiler.sigma_gravity_candidate_gate.CANDIDATE_CONFIG",
                domain="R > 0 and r > 0",
            ),
            "finding": (
                "NOT VARIATIONAL, and this is the significant result of the module.  Once the "
                "published disk coherence C = W(R) is restored, the effective force is no longer "
                "central: it depends on the cylindrical radius R as well as on the spherical "
                "radius r.  H3 then fails, with the factored residual proportional to "
                "xi * qz / ((qR + xi)^2 (G M + g_dagger r^2)).  Read that obstruction: it is "
                "proportional to the coherence scale xi and to the height qz off the mid-plane, "
                "so it vanishes exactly in the mid-plane and exactly in the limit xi -> 0, which "
                "is the spherical arm above.  A force field with nonzero curl does nonzero work "
                "around a closed loop, admits no potential, and therefore has no conserved energy "
                "and no Hamiltonian in this reduction.  No multiplier in the declared space "
                "repairs it.  The scope of this finding is exact: it concerns the test-particle "
                "force law as published and as reduced here, with the coherence treated as a "
                "fixed external function of position.  A field theory in which the coherence is "
                "instead a derived functional of the source is a different system and is not "
                "tested here."
            ),
            "expected": {
                "helmholtz_verdict": "NOT_VARIATIONAL",
                "failing_condition": "H3",
                "multiplier_outcome": "NO_MULTIPLIER_IN_DECLARED_SPACE",
                "required_conserved_quantities": [],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Derived consequences worth stating in physics terms.
# ---------------------------------------------------------------------------


def data_to_theory_chain(
    recovered: Mapping[str, Any], report: Mapping[str, Any]
) -> list[dict[str, str]]:
    """The end-to-end demonstration, as an ordered list of what each step produced.

    Every entry is read back out of this run's own artifacts; nothing here is a sentence about a
    result that was not computed above.
    """

    charges = [
        f"{entry['physical_identification']}: {entry['conserved_quantity']}"
        for entry in report["noether"]["symmetries"]
        if entry["physical_identification"]
    ]
    return [
        {
            "step": "1_data",
            "produced": (
                f"{recovered['public_row_count']} exact rational rows under neutral column names, "
                f"world {recovered['world_id']}, rows sha256 {recovered['public_rows_sha256']}"
            ),
            "by": "blind_planetary_law_rediscovery_campaign, read from its sealed world receipt",
        },
        {
            "step": "2_empirical_law",
            "produced": recovered["candidate_statement"],
            "by": (
                f"the campaign's {recovered['candidate_source_stage']} stage, frozen before the "
                "target fixture was unsealed"
            ),
        },
        {
            "step": "3_declared_lift_to_equations_of_motion",
            "produced": "; ".join(report["ir"]["equations"]),
            "by": recovered["declared_lift"],
        },
        {
            "step": "4_helmholtz_test",
            "produced": (
                f"{report['helmholtz']['verdict']} after "
                f"{report['helmholtz']['conditions_checked']} conditions, none failing"
            ),
            "by": "the Helmholtz conditions on the third-order jet",
        },
        {
            "step": "5_constructed_lagrangian",
            "produced": str(report["construction"]["lagrangian"]),
            "by": (
                f"{report['construction']['method']}, proven by a round trip with residual "
                f"{report['construction']['round_trip_residual']}"
            ),
        },
        {
            "step": "6_noether_consequences",
            "produced": "; ".join(charges),
            "by": (
                f"{report['noether']['generators_tested']} declared generators tested for strict "
                f"variational invariance; every charge verified conserved on shell"
            ),
        },
    ]


def circular_speed_consequence(radial_potential: sp.Expr) -> dict[str, Any]:
    """Circular-orbit speed from a radial potential, plus its large-radius limit."""

    speed_squared = sp.simplify(_RADIUS * sp.diff(radial_potential, _RADIUS))
    limit = sp.simplify(sp.limit(speed_squared, _RADIUS, sp.oo))
    entry: dict[str, Any] = {
        "derivation": "a circular orbit balances V'(r) against v^2/r, so v_c^2 = r V'(r)",
        "circular_speed_squared": str(speed_squared),
        "large_radius_limit_of_v_squared": str(limit),
        "asymptotically_flat_rotation_curve": bool(limit.is_finite and limit != 0),
    }
    if entry["asymptotically_flat_rotation_curve"]:
        entry["fourth_power_of_the_flat_speed"] = str(sp.simplify(limit**2))
    return entry


# ---------------------------------------------------------------------------
# Receipt assembly.
# ---------------------------------------------------------------------------


def _no_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise InverseVariationalEngineError(f"floating value forbidden at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _no_floats(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _no_floats(child, f"{path}[{index}]")


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(body), "content_sha256": canonical_sha256(body)}


def _check_expectation(
    report: Mapping[str, Any], expected: Mapping[str, Any], names: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare one system's report against its declared expectation.

    ``names`` is the system's own locals map.  It is mandatory: a bare ``sympify('2*gamma')``
    resolves ``gamma`` to sympy's gamma *function*, so an expectation written in terms of a
    declared parameter must be parsed in that parameter's namespace, exactly as the IR was.
    """

    blockers: list[str] = []
    verdict = report["helmholtz"]["verdict"]
    if verdict != expected["helmholtz_verdict"]:
        blockers.append(f"expected verdict {expected['helmholtz_verdict']} but observed {verdict}")
    if "failing_condition" in expected:
        kinds = {entry["condition"] for entry in report["helmholtz"]["failing_conditions"]}
        if kinds != {expected["failing_condition"]}:
            blockers.append(
                f"expected exactly condition {expected['failing_condition']} to fail, observed "
                f"{sorted(kinds)}"
            )
    if "failing_residual" in expected:
        residuals = {entry["residual"] for entry in report["helmholtz"]["failing_conditions"]}
        target = sp.sympify(expected["failing_residual"], locals=dict(names))
        if not any(
            sp.simplify(sp.sympify(text, locals=dict(names)) - target) == 0
            or sp.simplify(sp.sympify(text, locals=dict(names)) + target) == 0
            for text in residuals
        ):
            blockers.append(
                f"expected a failing residual of +/- {expected['failing_residual']}, observed "
                f"{sorted(residuals)}"
            )
    if expected.get("round_trip_verified") and not report["construction"]["round_trip_verified"]:
        blockers.append("the constructed Lagrangian did not round-trip to the input equations")
    if "multiplier_outcome" in expected:
        search = report["integrating_factor_search"]
        outcome = None if search is None else search["outcome"]
        if outcome != expected["multiplier_outcome"]:
            blockers.append(
                f"expected multiplier outcome {expected['multiplier_outcome']}, observed {outcome}"
            )
        elif search is not None and "multiplier" in expected:
            multipliers = {entry["multiplier"] for entry in search["found"]}
            target = sp.sympify(expected["multiplier"], locals=dict(names))
            if not any(
                sp.simplify(sp.sympify(text, locals=dict(names)) - target) == 0
                for text in multipliers
            ):
                blockers.append(
                    f"expected the multiplier {expected['multiplier']}, observed "
                    f"{sorted(multipliers)}"
                )
            if expected.get("multiplier_round_trip_verified") and not all(
                set(entry["round_trip_residual"]) == {"0"} for entry in search["found"]
            ):
                blockers.append("a multiplier Lagrangian did not round-trip")
    charges: set[str] = set()
    if report["noether"] is not None:
        charges = {
            entry["physical_identification"]
            for entry in report["noether"]["symmetries"]
            if entry["physical_identification"]
        }
    missing = sorted(set(expected.get("required_conserved_quantities", [])) - charges)
    if missing:
        blockers.append(f"missing required conserved quantities: {missing}")
    present = sorted(set(expected.get("forbidden_conserved_quantities", [])) & charges)
    if present:
        blockers.append(f"forbidden conserved quantities appeared: {present}")
    return {
        "status": "pass" if not blockers else "fail",
        "first_blocker": blockers[0] if blockers else None,
        "blockers": blockers,
    }


def analyze_system(declared: Mapping[str, Any]) -> dict[str, Any]:
    """Run all four steps on one declared system and check it against its expectation."""

    system = SecondOrderSystem(declared["ir"])
    helmholtz = helmholtz_test(system)
    report: dict[str, Any] = {
        "system_id": system.system_id,
        "role": declared["role"],
        "coordinate_count": system.size,
        "ir": system.ir,
        "class_check": helmholtz["class_check"],
        "helmholtz": {
            "verdict": helmholtz["verdict"],
            "conditions_checked": helmholtz["conditions_checked"],
            "conditions": helmholtz["conditions"],
            "failing_conditions": helmholtz["failing_conditions"],
        },
    }
    if "recovered_law" in declared:
        report["recovered_law"] = declared["recovered_law"]
    if declared.get("finding"):
        report["finding"] = declared["finding"]

    construction: dict[str, Any] = {
        "constructed": False,
        "method": None,
        "methods_tried": [],
        "lagrangian": None,
        "round_trip_residual": None,
        "round_trip_verified": False,
    }
    noether: dict[str, Any] | None = None
    multiplier: dict[str, Any] | None = None
    consequences: dict[str, Any] = {}

    if helmholtz["verdict"] == "VARIATIONAL":
        construction, lagrangian, radial = construct_lagrangian(system)
        if lagrangian is None:
            raise InverseVariationalEngineError(
                f"{system.system_id}: Helmholtz said VARIATIONAL but no Lagrangian round-tripped"
            )
        noether = noether_analysis(lagrangian, system)
        if radial is not None:
            consequences["circular_orbits"] = circular_speed_consequence(radial)
    elif helmholtz["verdict"] == "NOT_VARIATIONAL":
        multiplier = search_integrating_factors(system)

    report["construction"] = construction
    report["integrating_factor_search"] = multiplier
    report["noether"] = noether
    report["consequences"] = consequences
    if "recovered_law" in declared and noether is not None:
        report["data_to_theory_chain"] = data_to_theory_chain(declared["recovered_law"], report)
    report["expected"] = json.loads(json.dumps(dict(declared["expected"])))
    report["expectation_check"] = _check_expectation(
        report, declared["expected"], system.expression_locals()
    )
    if report["expectation_check"]["status"] != "pass":
        raise InverseVariationalEngineError(
            f"{system.system_id}: {report['expectation_check']['first_blocker']}"
        )
    return report


def _counts(systems: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "systems": len(systems),
        "controls": sum(1 for entry in systems if entry["role"] == "control"),
        "applications": sum(1 for entry in systems if entry["role"] == "application"),
        "variational": sum(
            1 for entry in systems if entry["helmholtz"]["verdict"] == "VARIATIONAL"
        ),
        "not_variational": sum(
            1 for entry in systems if entry["helmholtz"]["verdict"] == "NOT_VARIATIONAL"
        ),
        "out_of_declared_class": sum(
            1 for entry in systems if entry["helmholtz"]["verdict"] == "OUT_OF_DECLARED_CLASS"
        ),
        "lagrangians_constructed": sum(
            1 for entry in systems if entry["construction"]["constructed"]
        ),
        "round_trips_verified": sum(
            1 for entry in systems if entry["construction"]["round_trip_verified"]
        ),
        "multiplier_searches_run": sum(
            1 for entry in systems if entry["integrating_factor_search"] is not None
        ),
        "multipliers_found": sum(
            len(entry["integrating_factor_search"]["found"])
            for entry in systems
            if entry["integrating_factor_search"] is not None
        ),
        "helmholtz_conditions_checked": sum(
            entry["helmholtz"]["conditions_checked"] for entry in systems
        ),
        "conserved_quantities": sum(
            entry["noether"]["conserved_quantity_count"]
            for entry in systems
            if entry["noether"] is not None
        ),
    }


def run_inverse_variational_engine(root: str | Path = ".") -> dict[str, Any]:
    """Run every declared system through all four steps and seal the receipt."""

    root_path = Path(root).resolve()
    systems = [analyze_system(entry) for entry in declared_systems(root_path)]
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "claims": dict(CLAIMS),
        "scope": (
            "Given equations of motion, decide whether a Lagrangian exists whose Euler-Lagrange "
            "equations are exactly those, construct one when it does, prove the construction by "
            "differentiating it back, search a declared multiplier space when it does not, and "
            "derive the Noether charges of whatever is built.  Every verdict is about the system "
            "exactly as declared in its IR, on the declared class, in the reduction in which it "
            "was handed to the engine.  Nothing here is novel; the applications are re-readings "
            "of laws this repository already carries, and no observational dataset is opened."
        ),
        "declared_class": json.loads(json.dumps(DECLARED_CLASS)),
        "system_ir_schema": SYSTEM_IR_SCHEMA,
        "config": {
            "construction_methods_in_order": json.loads(json.dumps(list(CONSTRUCTION_METHODS))),
            "multiplier_space": json.loads(json.dumps(MULTIPLIER_SPACE)),
            "noether_generator_space": json.loads(json.dumps(NOETHER_GENERATOR_SPACE)),
            "probe_point_count": PROBE_POINT_COUNT,
            "witness_parameter_values": dict(WITNESS_PARAMETER_VALUES),
            "witness_coordinate_values": list(WITNESS_COORDINATE_VALUES),
            "sigma_gravity_candidate_formula": CANDIDATE_CONFIG["formula"],
            "sigma_gravity_candidate_source": CANDIDATE_CONFIG["source"],
        },
        "inputs": {
            name: {
                "path": BOUND_ARTIFACTS[name]["path"],
                "kind": BOUND_ARTIFACTS[name]["kind"],
                "binding": BOUND_ARTIFACTS[name]["semantic_sha256"],
                "binding_kind": "semantic_sha256",
                "role": BOUND_ARTIFACTS[name]["role"],
            }
            for name in sorted(BOUND_ARTIFACTS)
        },
        "systems": systems,
        "verdicts": {entry["system_id"]: entry["helmholtz"]["verdict"] for entry in systems},
        "counts": _counts(systems),
        "decision": "INVERSE_VARIATIONAL_ANALYSIS_COMPLETE_NO_NOVELTY_CLAIMED",
        "residual_gap_report": {
            "not_established": [
                (
                    "A VARIATIONAL verdict for a static central force is structurally automatic: "
                    "a force that depends only on the radius is always the gradient of a "
                    "potential, so the informative content in those cases is the closed-form "
                    "Lagrangian and its conserved quantities, not the verdict itself."
                ),
                (
                    "The NOT_VARIATIONAL verdict for the coherence-modulated screened-gravity "
                    "reduction is a statement about the test-particle force law as declared, with "
                    "the coherence treated as a fixed external function of position.  It is not a "
                    "statement about any field theory in which the coherence is instead a derived "
                    "functional of the source, and it is not a refutation of that candidate's "
                    "fits to data."
                ),
                (
                    "The multiplier search covers one declared scalar grammar.  The general "
                    "Douglas multiplier problem, with a full matrix multiplier, is not solved "
                    "here, so NO_MULTIPLIER_IN_DECLARED_SPACE never means 'no multiplier exists'."
                ),
                (
                    "Only strict variational point symmetries are searched.  Divergence "
                    "symmetries, generalized (Lie-Backlund) symmetries and hidden symmetries are "
                    "outside the declared generator space, so a system reported with no conserved "
                    "quantity may still have one."
                ),
                (
                    "Sufficiency of the Helmholtz conditions is local and star-shaped in the "
                    "declared domain; the constructed Lagrangian is one representative of an "
                    "infinite equivalence class, not a canonical object."
                ),
                (
                    "Lifting the blindly recovered scalar power law to a three-dimensional "
                    "equation of motion assumes centrality and isotropy.  Those assumptions are "
                    "declared here; the anonymized columns did not contain them."
                ),
                (
                    "No candidate theory is promoted, screened or validated by this module, and "
                    "no observational dataset is opened."
                ),
            ]
        },
    }
    body["config_sha256"] = canonical_sha256(body["config"])
    _no_floats(body)
    return _seal(body)


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Fail-closed validation: seal, claims, declared class and per-system structure."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise InverseVariationalEngineError("unexpected inverse-variational schema version")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise InverseVariationalEngineError("inverse-variational receipt seal changed")
    if value.get("claims") != CLAIMS:
        raise InverseVariationalEngineError("claims block changed")
    if value.get("declared_class") != json.loads(json.dumps(DECLARED_CLASS)):
        raise InverseVariationalEngineError("declared class boundary changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config")):
        raise InverseVariationalEngineError("config binding changed")
    _no_floats(body)

    systems = value.get("systems")
    if not isinstance(systems, list) or not systems:
        raise InverseVariationalEngineError("the receipt carries no systems")
    allowed = {"VARIATIONAL", "NOT_VARIATIONAL", "OUT_OF_DECLARED_CLASS"}
    for entry in systems:
        verdict = entry["helmholtz"]["verdict"]
        if verdict not in allowed:
            raise InverseVariationalEngineError(f"unknown verdict: {verdict}")
        if entry["expectation_check"]["status"] != "pass":
            raise InverseVariationalEngineError("a system did not meet its declared expectation")
        if verdict == "VARIATIONAL":
            if not entry["construction"]["round_trip_verified"]:
                raise InverseVariationalEngineError(
                    "a VARIATIONAL system carries no verified round trip"
                )
            if set(entry["construction"]["round_trip_residual"]) != {"0"}:
                raise InverseVariationalEngineError("a round-trip residual is not zero")
        if verdict == "NOT_VARIATIONAL" and not entry["helmholtz"]["failing_conditions"]:
            raise InverseVariationalEngineError(
                "a NOT_VARIATIONAL system names no failing condition"
            )
        if verdict == "OUT_OF_DECLARED_CLASS" and not entry["class_check"]["reasons_outside"]:
            raise InverseVariationalEngineError(
                "an OUT_OF_DECLARED_CLASS system names no reason"
            )
        for failing in entry["helmholtz"]["failing_conditions"]:
            if failing.get("nonvanishing_witness") in (None, "0"):
                raise InverseVariationalEngineError(
                    "a failing condition carries no non-vanishing witness"
                )
        search = entry["integrating_factor_search"]
        if search is not None:
            for found in search["found"]:
                if set(found["round_trip_residual"]) != {"0"}:
                    raise InverseVariationalEngineError(
                        "a multiplier Lagrangian round-trip residual is not zero"
                    )
                if found["helmholtz_verdict"] != "VARIATIONAL":
                    raise InverseVariationalEngineError(
                        "a reported multiplier did not make the system variational"
                    )
        if entry["noether"] is not None:
            for symmetry in entry["noether"]["symmetries"]:
                if symmetry["invariance_defect"] != "0":
                    raise InverseVariationalEngineError("a reported symmetry is not invariant")
                if symmetry["on_shell_time_derivative"] != "0":
                    raise InverseVariationalEngineError(
                        "a reported conserved quantity is not conserved on shell"
                    )

    if json.loads(json.dumps(_counts(systems))) != value.get("counts"):
        raise InverseVariationalEngineError("aggregate counts do not replay from the systems")
    if value.get("verdicts") != {
        entry["system_id"]: entry["helmholtz"]["verdict"] for entry in systems
    }:
        raise InverseVariationalEngineError("the verdict index does not replay from the systems")


def _write(value: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise InverseVariationalEngineError("refusing to overwrite an immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether declared equations of motion come from a least-action principle, "
            "construct the Lagrangian when they do, and report the conserved quantities."
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
    result = run_inverse_variational_engine(root)
    if args.output:
        _write(result, args.output)
    print(
        json.dumps(
            {
                "verdicts": result["verdicts"],
                "lagrangians_constructed": result["counts"]["lagrangians_constructed"],
                "round_trips_verified": result["counts"]["round_trips_verified"],
                "multipliers_found": result["counts"]["multipliers_found"],
                "conserved_quantities": result["counts"]["conserved_quantities"],
                "decision": result["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
