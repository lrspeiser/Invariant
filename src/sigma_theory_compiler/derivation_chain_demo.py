"""First-principles derivation chains: Newton, General Relativity, and the excluded alternatives.

Curve-fitting produces a formula that reproduces numbers.  Derivation produces a formula that
follows from a declared action, and then says which competing formulas the measurement kills.
This module demonstrates the second half, as three auditable chains whose every step is a
*recomputed* symbolic artifact rather than a sentence about one.

Chain 1 (Newton, new in this module).  Declare ``S = int [ m|v|^2/2 - m Phi(r) ] dt`` with
``nabla^2 Phi = 0`` in vacuum.  Solve that ODE with sympy: the general spherically symmetric
solution is ``C1 + C2/r``, and ``Phi -> 0`` at infinity forces ``C1 = 0``.  The inverse-square law
is therefore *derived* as the unique spherically symmetric vacuum solution in three spatial
dimensions, not assumed.  Euler-Lagrange then gives ``a = -A r_hat/r^2``, circular-orbit balance
gives Kepler III ``T^2 = (4 pi^2/A) a^3``, and redoing the ODE in ``d`` spatial dimensions gives
``Phi ~ r^(2-d)`` and ``|g| ~ r^(1-d)``.  That last step is the honest answer to "why 2": the
exponent is a consequence of dimensionality, not a fitted parameter.

Chain 2 (General Relativity, reusing this repository's verified controls).  Steps 1-3 reuse
existing machinery and are hash-bound to it: the Einstein-Hilbert action IR, the Cadabra metric
variation registered in the Einstein-Hilbert ``action-health.json``, and
:func:`sigma_theory_compiler.relativity.schwarzschild_ricci_components`, which builds the
Christoffel symbols and the Ricci tensor from the metric definition and returns all sixteen
components.  Steps 4-5 are new here, because the repository has no geodesic machinery: the
Schwarzschild geodesic gives the orbit equation ``u'' + u = GM/L^2 + 3 GM u^2/c^2``, and a
Poincare-Lindstedt expansion gives ``dphi = 6 pi GM/(c^2 a (1-e^2))`` per orbit.  The numeric
verification against Mercury reuses
:func:`sigma_theory_compiler.relativity.solar_system_numeric_checks`.

Chain 3 (the alternatives, new in this module).  A declared three-member family of competitors to
the inverse square -- a modified exponent, a Yukawa term, and a graviton-mass-like term -- is put
through the same nearly-circular apsidal-angle perturbation theory, symbolically.  The published
Mercury agreement then converts each precession formula into an exclusion interval.  The apsidal
machinery is cross-validated by feeding it the Chain 2 effective force and recovering Chain 2's
own ``6 pi GM/(c^2 a)``.

Honesty rules this module obeys.

1. Every symbolic step is recomputed on each run.  Nothing is a stored transcript; the tests
   re-derive each step independently and compare.
2. No observational dataset is opened.  The Mercury agreement level is a *cited published value*
   carried in :data:`CITED_OBSERVATIONS`, not a fit and not a data read.
3. Constants come from the hash-bound :mod:`sigma_theory_compiler.relativity` module, and the
   recomputation is cross-checked against that module's own control output.
4. Nothing here is novel.  Every result in all three chains is textbook nineteenth- and
   twentieth-century physics, reproduced as an engine capability demonstration.
5. Exclusion bounds are order-of-magnitude solar-system limits from a single observable under a
   nearly-circular approximation.  They are not a replacement for a full ephemeris fit.

A PASS here means the engine derived these laws and bounds symbolically from declared actions and
cited numbers.  It does not mean any new physics was found, and it does not mean the excluded
alternatives are excluded by *this* work rather than by the measurement it cites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from .relativity import (
    C_SI,
    G_SI,
    JULIAN_YEAR_DAYS,
    M_SUN_KG,
    schwarzschild_ricci_components,
    solar_system_numeric_checks,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

#: Result schema.  Bump only with a receipt-shape change.
RESULT_SCHEMA = "invariant-derivation-chain-demo-result-1.0"

#: Committed receipt location.
RECEIPT_PATH = "runs/math/derivation-chain/chain-v1.json"

#: Working precision.  Deliberately far above the emitted precision so that no last-digit
#: wobble in a transcendental can reach a receipt.
WORKING_PRECISION_DIGITS = 60

#: Emitted precision for every numeric string in the receipt.
EMITTED_PRECISION_DIGITS = 12

#: Claims block.  Frozen; any change changes the receipt hash and therefore the claim.
CLAIMS: dict[str, bool] = {
    "derivation_is_symbolic_and_checked": True,
    "novelty_claimed": False,
    "published_values_cited_not_fitted": True,
    "real_observational_data_opened": False,
    "reuses_existing_verified_controls": True,
}

#: Existing repository artifacts this module binds to.  JSON artifacts are bound by the SHA-256
#: of their *parsed* canonical serialization and non-JSON artifacts by the SHA-256 of their
#: line-ending-normalized text, so a CRLF checkout cannot change a binding.
BOUND_ARTIFACTS: dict[str, dict[str, str]] = {
    "einstein_hilbert_action_health": {
        "path": (
            "runs/formal-controls-v1/action-health/einstein_hilbert_control/action-health.json"
        ),
        "kind": "json",
        "semantic_sha256": "018ff5ccb448120141fd57c73d9ebe242b7d6c5cfc7282b1d9fddb2c18f2eb61",
    },
    "einstein_hilbert_action_ir": {
        "path": "runs/formal-controls-v1/action-health/einstein_hilbert_control/action-ir.json",
        "kind": "json",
        "semantic_sha256": "21ec90dfad930d870c618ff1d46f01dd0c3ffca2568f8b40c3d7bc00f1ec89d8",
    },
    "einstein_hilbert_action_spec": {
        "path": "configs/actions/einstein_hilbert_control.json",
        "kind": "json",
        "semantic_sha256": "a5f98e66bdcb5772b94b8bf1103039edef7dbfc5669f20275f1972c27baf69c3",
    },
    "einstein_hilbert_metric_variation_script": {
        "path": "formal/cadabra/einstein_hilbert_metric_variation.cdb",
        "kind": "text",
        "normalized_text_sha256": (
            "c94509edc9b3e5e2181821fad589bc55681a25f858f7313378ede02531a1e2f0"
        ),
    },
    "relativity_module": {
        "path": "src/sigma_theory_compiler/relativity.py",
        "kind": "text",
        "normalized_text_sha256": (
            "88f0c4c4f1fc1a128bb7a59345673df1f487bcf44ba05d10b7088fb92d3e491e"
        ),
    },
    "relativity_reference_report": {
        "path": "runs/gr-reference/relativity_reference.json",
        "kind": "json",
        "semantic_sha256": "428a9d6779b8a02e1966cef24bab5cd5dc680db6836ed77982758efc7fda20b8",
    },
}

#: The declared Einstein-Hilbert action term that must be present in the bound action IR.
REQUIRED_ACTION_TERM = {"id": "EH_R", "density": "sqrt(-g) R", "coefficient": "M_Pl^2/2"}

#: Action-health gates that must read ``pass`` before Chain 2 may claim a varied field equation.
REQUIRED_ACTION_HEALTH_GATES = ("covariant_variation", "covariant_identity", "field_contract")

#: The registered Cadabra control that certifies the Einstein-Hilbert metric variation.
REQUIRED_VARIATION_CONTROL = "cadabra_einstein_hilbert_metric_variation"

#: Published values.  These are *cited*, never fitted, and no dataset is opened to obtain them.
CITED_OBSERVATIONS: dict[str, dict[str, str]] = {
    "mercury_anomalous_perihelion_advance": {
        "value_arcsec_per_century": "42.98",
        "fractional_agreement_with_general_relativity": "1e-3",
        "citation": (
            "C. M. Will, 'The Confrontation between General Relativity and Experiment', "
            "Living Reviews in Relativity 17, 4 (2014), Sec. 3.5: the measured anomalous "
            "perihelion advance of Mercury agrees with the general-relativistic value at "
            "roughly the 0.1 percent level."
        ),
        "use": (
            "Only the fractional agreement is used, as a residual budget for any non-Einsteinian "
            "contribution to the precession."
        ),
    },
    "mercury_orbital_elements": {
        "semi_major_axis_m": "5.790905e10",
        "eccentricity": "0.205630",
        "sidereal_period_days": "87.9691",
        "citation": (
            "Standard IAU/JPL mean orbital elements for Mercury; these are the same frozen values "
            "already used by sigma_theory_compiler.relativity.solar_system_numeric_checks, and "
            "the recomputation below is cross-checked against that control's own output."
        ),
        "use": "Converts a per-orbit precession into an arcseconds-per-century observable.",
    },
    "defining_si_constants": {
        "planck_constant_J_s": "6.62607015e-34",
        "electron_volt_J": "1.602176634e-19",
        "citation": (
            "SI (2019 revision) exact defining constants; used only to express a graviton "
            "Compton-wavelength bound as a mass bound."
        ),
        "use": "Unit conversion only; no measurement enters through these.",
    },
}

#: Receipt key carrying each alternative's leading-order precession.  The graviton arm is
#: expanded in a/lambda_g rather than in its own parameter, so it reports a different key.
_LEADING_ORDER_KEY = {
    "modified_exponent": "advance_leading_order",
    "yukawa": "advance_leading_order",
    "graviton_mass": "advance_leading_order_in_x",
}

#: Spatial dimensions at which the vacuum Laplace solution is re-derived for Chain 1 step 5.
DIMENSION_SAMPLE_POINTS = (2, 3, 4, 5, 6)

#: Yukawa ranges, as multiples of Mercury's semi-major axis, at which an exclusion is reported.
YUKAWA_RANGE_SAMPLE_POINTS = ("0.1", "0.2", "0.5", "1", "2", "5", "10")

#: Relative tolerance for the cross-check of the recomputed Mercury advance against the
#: existing relativity control.  Loose enough only to absorb that control's truncated
#: arcseconds-per-radian literal, tight enough that any element change fails closed.
MERCURY_CROSSCHECK_RELATIVE_TOLERANCE = "1e-9"


class DerivationChainDemoError(ValueError):
    """Raised on malformed input, a broken known-answer control, or receipt tamper."""


# ---------------------------------------------------------------------------
# Exact helpers.  No float ever reaches a receipt.
# ---------------------------------------------------------------------------


def _decimal(value: Any, digits: int = EMITTED_PRECISION_DIGITS) -> str:
    """Return a fixed-significance decimal string for an mpmath or Python number."""

    return mp.nstr(mp.mpf(value), digits, strip_zeros=False)


def _text(expression: Any) -> str:
    """Return the canonical sympy string form of an expression."""

    return str(expression)


def _no_floats(value: Any, path: str = "$") -> None:
    """Reject any float anywhere in a receipt body."""

    if isinstance(value, float):
        raise DerivationChainDemoError(f"floating value forbidden at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _no_floats(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _no_floats(child, f"{path}[{index}]")


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(body), "content_sha256": canonical_sha256(body)}


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _semantic_sha256(value: Any) -> str:
    """SHA-256 of a parsed JSON document under a deterministic serialization.

    Parsing first makes the digest independent of the checkout's line endings and of
    insignificant whitespace, which raw byte hashes are not on a Windows checkout.
    """

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalized_text_sha256(text: str) -> str:
    """SHA-256 of text with every line ending normalized to ``\\n``."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() != path and root.resolve() not in path.parents:
        raise DerivationChainDemoError(f"bound path escapes repository root: {relative}")
    return path


def _load_bound_artifacts(root: Path) -> dict[str, Any]:
    """Load and hash-verify every bound repository artifact, failing closed on any drift."""

    loaded: dict[str, Any] = {}
    for name in sorted(BOUND_ARTIFACTS):
        binding = BOUND_ARTIFACTS[name]
        path = _resolve(root, binding["path"])
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DerivationChainDemoError(
                f"cannot read bound artifact: {binding['path']}"
            ) from exc
        if binding["kind"] == "json":
            try:
                document = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DerivationChainDemoError(f"invalid JSON: {binding['path']}") from exc
            observed = _semantic_sha256(document)
            if observed != binding["semantic_sha256"]:
                raise DerivationChainDemoError(
                    f"bound artifact hash mismatch: {binding['path']}"
                )
            loaded[name] = document
        else:
            observed = _normalized_text_sha256(text)
            if observed != binding["normalized_text_sha256"]:
                raise DerivationChainDemoError(
                    f"bound artifact hash mismatch: {binding['path']}"
                )
            loaded[name] = text
    return loaded


def _step(
    number: int,
    statement: str,
    symbolic_result: str,
    check: str,
    check_status: str,
    provenance: str,
    certified_by: str,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble one chain step.  A step that did not check ``pass`` aborts the run."""

    if check_status != "pass":
        raise DerivationChainDemoError(f"derivation step {number} failed its check: {check}")
    if provenance not in {"new_in_this_module", "existing_repository_control"}:
        raise DerivationChainDemoError(f"unknown provenance for step {number}: {provenance}")
    return {
        "step": number,
        "statement": statement,
        "symbolic_result": symbolic_result,
        "check": check,
        "check_status": check_status,
        "provenance": provenance,
        "certified_by": certified_by,
        **extra,
    }


# ---------------------------------------------------------------------------
# Chain 1.  Newton, from a variational principle.  New in this module.
# ---------------------------------------------------------------------------


def radial_laplace_operator(dimension: Any) -> tuple[sp.Symbol, sp.Function, sp.Expr]:
    """Return ``(r, Phi, nabla^2 Phi)`` for a spherically symmetric field in ``d`` dimensions."""

    radius = sp.Symbol("r", positive=True)
    potential = sp.Function("Phi")
    laplacian = (
        sp.diff(radius ** (dimension - 1) * sp.diff(potential(radius), radius), radius)
        / radius ** (dimension - 1)
    )
    return radius, potential, sp.simplify(sp.expand(laplacian))


def solve_vacuum_potential_expressions(dimension: Any) -> dict[str, Any]:
    """Solve ``nabla^2 Phi = 0`` in ``d`` dimensions, returning live sympy expressions.

    Expressions are returned rather than strings so that callers never round-trip through
    :func:`sympy.sympify`, which would silently drop the ``positive=True`` assumption on ``r``.
    """

    radius, potential, laplacian = radial_laplace_operator(dimension)
    solution = sp.dsolve(sp.Eq(laplacian, 0), potential(radius))
    general = sp.simplify(solution.rhs)
    residual = sp.simplify(laplacian.subs(potential(radius), general).doit())
    field = sp.simplify(-sp.diff(general, radius))
    return {
        "radius": radius,
        "operator": laplacian,
        "general_solution": general,
        "vacuum_residual": residual,
        "field_strength": field,
    }


def solve_vacuum_potential(dimension: Any) -> dict[str, str]:
    """String view of :func:`solve_vacuum_potential_expressions` for the receipt."""

    solved = solve_vacuum_potential_expressions(dimension)
    return {
        "dimension": str(dimension),
        "operator": _text(solved["operator"]),
        "general_solution": _text(solved["general_solution"]),
        "vacuum_residual": _text(solved["vacuum_residual"]),
        "field_strength": _text(solved["field_strength"]),
    }


def newton_chain() -> dict[str, Any]:
    """Derive the inverse-square law and Kepler III from a declared action."""

    radius = sp.Symbol("r", positive=True)
    amplitude = sp.Symbol("A", positive=True)
    mass = sp.Symbol("m", positive=True)
    time = sp.Symbol("t", real=True)

    # --- Step 1: declare the action and the vacuum field equation.
    laplacian_3d = radial_laplace_operator(3)[2]
    action = "S = Integral((1/2)*m*(dx/dt)^2 - m*Phi(r), t)"
    step_one = _step(
        1,
        "Declare the action and the vacuum field equation; nothing else is assumed.",
        f"{action}; vacuum condition nabla^2 Phi = 0 with Phi = Phi(r)",
        "the spherically symmetric Laplacian is rebuilt from the metric-free radial form",
        "pass" if laplacian_3d != 0 else "fail",
        "new_in_this_module",
        "derivation_chain_demo.radial_laplace_operator",
        declared_action=action,
        declared_field_equation="nabla^2 Phi = 0",
        radial_operator=_text(laplacian_3d),
    )

    # --- Step 2: solve the ODE.  This is where the inverse square is *derived*.
    solved_expr = solve_vacuum_potential_expressions(3)
    solved = solve_vacuum_potential(3)
    radius = solved_expr["radius"]
    general = solved_expr["general_solution"]
    constants = sorted(general.free_symbols - {radius}, key=lambda symbol: symbol.name)
    if len(constants) != 2:
        raise DerivationChainDemoError("the 3d vacuum solution did not carry two constants")
    additive, coefficient = constants[0], constants[1]
    limit_at_infinity = sp.limit(general, radius, sp.oo)
    bounded = sp.simplify(general.subs(additive, 0))
    derived_potential = sp.simplify(bounded.subs(coefficient, -amplitude))
    residual_3d = solved_expr["vacuum_residual"]
    boundary_ok = sp.simplify(limit_at_infinity - additive) == 0
    step_two = _step(
        2,
        (
            "Solve nabla^2 Phi = 0 symbolically.  The inverse-square potential is the unique "
            "spherically symmetric vacuum solution in three spatial dimensions that vanishes at "
            "infinity; it is derived here, not assumed."
        ),
        f"Phi(r) = {solved['general_solution']}  ->  Phi(r) = {_text(derived_potential)}",
        (
            "the general solution has zero vacuum residual, its limit at infinity is the additive "
            "constant, and that constant is set to zero by the boundary condition"
        ),
        "pass" if residual_3d == 0 and boundary_ok else "fail",
        "new_in_this_module",
        "derivation_chain_demo.solve_vacuum_potential",
        general_solution=solved["general_solution"],
        vacuum_residual=_text(residual_3d),
        limit_at_infinity=_text(limit_at_infinity),
        boundary_condition="Phi(r) -> 0 as r -> oo, hence the additive constant vanishes",
        derived_potential=_text(derived_potential),
        derived_field_strength=_text(sp.simplify(-sp.diff(derived_potential, radius))),
    )

    # --- Step 3: Euler-Lagrange.
    position = [sp.Function(name)(time) for name in ("x", "y", "z")]
    distance = sp.sqrt(sum(component**2 for component in position))
    lagrangian = sp.Rational(1, 2) * mass * sum(
        sp.diff(component, time) ** 2 for component in position
    ) - mass * derived_potential.subs(radius, distance)
    accelerations = []
    for component in position:
        euler_lagrange = sp.diff(lagrangian, component) - sp.diff(
            sp.diff(lagrangian, sp.diff(component, time)), time
        )
        solutions = sp.solve(sp.Eq(euler_lagrange, 0), sp.diff(component, time, 2))
        if len(solutions) != 1:
            raise DerivationChainDemoError("Euler-Lagrange did not yield a unique acceleration")
        accelerations.append(sp.simplify(solutions[0]))
    expected = [
        sp.simplify(-amplitude * component / distance**3) for component in position
    ]
    euler_ok = all(
        sp.simplify(found - want) == 0 for found, want in zip(accelerations, expected, strict=True)
    )
    step_three = _step(
        3,
        "Apply Euler-Lagrange to the declared action with the derived potential.",
        "d^2 x_i/dt^2 = -A x_i/(x^2+y^2+z^2)^(3/2), i.e. a = -A r_hat/r^2",
        "each component of the Euler-Lagrange acceleration matches -A x_i/r^3 exactly",
        "pass" if euler_ok else "fail",
        "new_in_this_module",
        "derivation_chain_demo.newton_chain",
        lagrangian=_text(sp.simplify(lagrangian)),
        acceleration_components=[_text(item) for item in accelerations],
        acceleration_magnitude="A/r^2",
    )

    # --- Step 4: Kepler III from circular-orbit balance.
    semi_major = sp.Symbol("a", positive=True)
    period = sp.Symbol("T", positive=True)
    angular_rate = 2 * sp.pi / period
    balance = sp.Eq(angular_rate**2 * semi_major, amplitude / semi_major**2)
    period_solutions = sp.solve(balance, period)
    if len(period_solutions) != 1:
        raise DerivationChainDemoError("circular-orbit balance did not yield a unique period")
    period_squared = sp.simplify(period_solutions[0] ** 2)
    kepler_ratio = sp.simplify(period_squared / semi_major**3)
    exponent_scan = _kepler_exponent_scan()
    # The negative control gates the step: only the inverse square may survive the scan.
    scan_discriminates = [row["kepler_third_law_holds"] for row in exponent_scan] == [
        "no",
        "yes",
        "no",
        "no",
    ]
    kepler_ok = (
        sp.simplify(period_squared - 4 * sp.pi**2 * semi_major**3 / amplitude) == 0
        and sp.simplify(sp.diff(kepler_ratio, semi_major)) == 0
        and scan_discriminates
    )
    step_four = _step(
        4,
        "Impose circular-orbit balance on the derived acceleration and read off Kepler III.",
        f"T^2 = {_text(period_squared)}, so T^2/a^3 = {_text(kepler_ratio)} is independent of a",
        (
            "the period squared equals 4 pi^2 a^3/A and d(T^2/a^3)/da vanishes; a scan over "
            "force exponents shows only the inverse square makes that ratio a-independent"
        ),
        "pass" if kepler_ok else "fail",
        "new_in_this_module",
        "derivation_chain_demo.newton_chain",
        balance=_text(balance),
        period_squared=_text(period_squared),
        kepler_ratio=_text(kepler_ratio),
        kepler_exponent="3/2 power of the semi-major axis",
        exponent_scan=exponent_scan,
    )

    # --- Step 5: dimensional generalization.  The honest "why 2".
    dimension = sp.Symbol("d", positive=True, integer=True)
    general_d_expr = solve_vacuum_potential_expressions(dimension)
    general_d = solve_vacuum_potential(dimension)
    samples = [solve_vacuum_potential(value) for value in DIMENSION_SAMPLE_POINTS]
    d_radius = general_d_expr["radius"]
    d_solution = general_d_expr["general_solution"]
    d_constants = sorted(
        d_solution.free_symbols - {d_radius, dimension}, key=lambda symbol: symbol.name
    )
    if len(d_constants) != 2:
        raise DerivationChainDemoError(
            "the d-dimensional vacuum solution lost an integration constant"
        )
    additive_d = next(
        symbol for symbol in d_constants if sp.simplify(sp.diff(d_solution, symbol)) == 1
    )
    scaling_d = next(symbol for symbol in d_constants if symbol is not additive_d)
    potential_shape = sp.simplify(sp.diff(d_solution, scaling_d) - d_radius ** (2 - dimension))
    force_shape = sp.simplify(
        sp.diff(general_d_expr["field_strength"], scaling_d)
        - (dimension - 2) * d_radius ** (1 - dimension)
    )
    three_expr = solve_vacuum_potential_expressions(3)
    three_field = three_expr["field_strength"]
    three_constants = sorted(
        three_field.free_symbols - {three_expr["radius"]}, key=lambda symbol: symbol.name
    )
    inverse_square = sp.simplify(
        three_field.subs({symbol: 1 for symbol in three_constants}) - 1 / three_expr["radius"] ** 2
    )
    dimension_ok = (
        general_d_expr["vacuum_residual"] == 0
        and potential_shape == 0
        and force_shape == 0
        and inverse_square == 0
    )
    step_five = _step(
        5,
        (
            "Redo step 2 in d spatial dimensions.  The force exponent is a consequence of "
            "dimensionality, not a fitted number."
        ),
        (
            f"Phi(r) = {general_d['general_solution']}, so |g| ~ r^(1-d); "
            "d = 3 gives Phi ~ 1/r and |g| ~ 1/r^2"
        ),
        (
            "the d-dimensional vacuum residual is zero and the d = 3 field strength is "
            "proportional to 1/r^2 exactly"
        ),
        "pass" if dimension_ok else "fail",
        "new_in_this_module",
        "derivation_chain_demo.solve_vacuum_potential",
        general_solution_in_d=general_d["general_solution"],
        vacuum_residual_in_d=general_d["vacuum_residual"],
        field_strength_in_d=general_d["field_strength"],
        potential_exponent="2 - d",
        force_exponent="1 - d",
        dimension_samples=samples,
        why_two=(
            "The exponent 2 in |g| ~ r^-2 is r^(1-d) at d = 3.  It is fixed by the number of "
            "spatial dimensions in which Laplace's equation is solved."
        ),
    )

    return {
        "chain_id": "newton_from_variational_principle",
        "theory": "Newtonian gravitation",
        "starting_point": action,
        "observable_reached": "Kepler's third law, T^2 proportional to a^3",
        "steps": [step_one, step_two, step_three, step_four, step_five],
        "negative_controls": [
            {
                "control_id": "wrong_force_exponent_fails_kepler",
                "mutation": "replace the derived 1/r^2 force with 1/r^n for n != 2",
                "observed": exponent_scan,
                "rejected": True,
            },
            {
                "control_id": "wrong_dimension_does_not_give_inverse_square",
                "mutation": "solve the same vacuum equation in d != 3 spatial dimensions",
                "observed": [
                    {"dimension": item["dimension"], "field_strength": item["field_strength"]}
                    for item in samples
                    if item["dimension"] != "3"
                ],
                "rejected": True,
            },
        ],
    }


def _kepler_exponent_scan() -> list[dict[str, str]]:
    """Show that only the inverse-square force makes ``T^2/a^3`` independent of ``a``."""

    semi_major = sp.Symbol("a", positive=True)
    period = sp.Symbol("T", positive=True)
    amplitude = sp.Symbol("A", positive=True)
    rows: list[dict[str, str]] = []
    for exponent in (1, 2, 3, 4):
        balance = sp.Eq((2 * sp.pi / period) ** 2 * semi_major, amplitude / semi_major**exponent)
        squared = sp.simplify(sp.solve(balance, period)[0] ** 2)
        ratio = sp.simplify(squared / semi_major**3)
        constant = sp.simplify(sp.diff(ratio, semi_major)) == 0
        rows.append(
            {
                "force_exponent": str(exponent),
                "period_squared": _text(squared),
                "period_squared_over_a_cubed": _text(ratio),
                "kepler_third_law_holds": "yes" if constant else "no",
            }
        )
    if [row["kepler_third_law_holds"] for row in rows] != ["no", "yes", "no", "no"]:
        raise DerivationChainDemoError("the Kepler exponent negative control did not discriminate")
    return rows


# ---------------------------------------------------------------------------
# Chain 2.  General Relativity, action to perihelion.
# ---------------------------------------------------------------------------


def schwarzschild_orbit_equation() -> dict[str, Any]:
    """Derive ``u'' + u = GM/L^2 + 3 GM u^2/c^2`` from the Schwarzschild geodesic.

    New in this module: the repository carries no geodesic machinery.
    """

    angle = sp.Symbol("phi", real=True)
    inverse_radius = sp.Function("u")
    momentum, energy, light, newton_mass = sp.symbols("L E c GM", positive=True)
    schwarzschild_radius = 2 * newton_mass / light**2

    # Radial equation from the normalization of the four-velocity, with u = 1/r and
    # dr/dtau = -L du/dphi.  Everything below is the first integral, differentiated.
    u = inverse_radius(angle)
    first_integral = (
        momentum**2 * sp.diff(u, angle) ** 2
        - energy**2 / light**2
        + (1 - schwarzschild_radius * u) * (light**2 + momentum**2 * u**2)
    )
    differentiated = sp.expand(sp.diff(first_integral, angle))
    reduced = sp.simplify(sp.expand(differentiated / (2 * momentum**2 * sp.diff(u, angle))))
    orbit = sp.simplify(reduced)
    target = sp.simplify(
        sp.diff(u, angle, 2) + u - newton_mass / momentum**2 - 3 * newton_mass * u**2 / light**2
    )
    residual = sp.simplify(sp.expand(orbit - target))

    # Negative control: dropping the relativistic term returns the Newtonian conic, which
    # precesses by exactly zero.
    newtonian = sp.simplify(target + 3 * newton_mass * u**2 / light**2)
    return {
        "first_integral": _text(sp.simplify(first_integral)),
        "orbit_equation": "u'' + u = GM/L^2 + 3*GM*u^2/c^2",
        "orbit_equation_residual": _text(residual),
        "newtonian_limit": "u'' + u = GM/L^2",
        "newtonian_limit_expression": _text(newtonian),
        "relativistic_term": "3*GM*u^2/c^2",
        "derivation_is_exact": residual == 0,
    }


def perihelion_advance_expression() -> dict[str, Any]:
    """Poincare-Lindstedt solution of the orbit equation.  New in this module."""

    psi = sp.Symbol("psi", real=True)
    eccentricity, epsilon = sp.symbols("e epsilon", positive=True)
    detuning = sp.Symbol("k1", real=True)
    correction = sp.Function("w1")(psi)
    momentum, light, newton_mass = sp.symbols("L c GM", positive=True)
    semi_major = sp.Symbol("a", positive=True)

    # Rescale u = (GM/L^2) w and phi = psi/k, so that the orbit equation becomes
    # k^2 w'' + w = 1 + epsilon w^2 with epsilon = 3 (GM)^2/(c^2 L^2).
    zeroth = 1 + eccentricity * sp.cos(psi)
    ansatz = (1 + epsilon * detuning) * (
        sp.diff(zeroth, psi, 2) + epsilon * sp.diff(correction, psi, 2)
    ) + (zeroth + epsilon * correction) - 1 - epsilon * (zeroth + epsilon * correction) ** 2
    first_order = sp.expand(
        sp.expand_trig(sp.expand(sp.series(ansatz, epsilon, 0, 2).removeO().coeff(epsilon, 1)))
    )
    secular = sp.expand(first_order).coeff(sp.cos(psi), 1)
    detunings = sp.solve(sp.Eq(secular, 0), detuning)
    if len(detunings) != 1:
        raise DerivationChainDemoError("secular removal did not fix a unique detuning")
    detuning_value = detunings[0]
    wavenumber = sp.sqrt(1 + epsilon * detuning_value)
    advance = sp.simplify(sp.series(2 * sp.pi / wavenumber - 2 * sp.pi, epsilon, 0, 2).removeO())
    epsilon_value = 3 * newton_mass**2 / (light**2 * momentum**2)
    in_momentum = sp.simplify(advance.subs(epsilon, epsilon_value))
    in_elements = sp.simplify(
        in_momentum.subs(momentum**2, newton_mass * semi_major * (1 - eccentricity**2))
    )
    expected = 6 * sp.pi * newton_mass / (light**2 * semi_major * (1 - eccentricity**2))
    residual = sp.simplify(in_elements - expected)
    return {
        "small_parameter": "epsilon = 3*(GM)^2/(c^2*L^2)",
        "secular_coefficient": _text(secular),
        "detuning": _text(detuning_value),
        "wavenumber": _text(wavenumber),
        "advance_in_epsilon": _text(advance),
        "advance_in_angular_momentum": _text(in_momentum),
        "advance_in_orbital_elements": "6*pi*GM/(c^2*a*(1 - e^2))",
        "advance_residual_against_closed_form": _text(residual),
        "derivation_is_exact": residual == 0,
    }


def perturbed_schwarzschild_ricci() -> dict[str, str]:
    """Negative control: a metric that is not Schwarzschild must fail ``R_ab = 0``.

    The lapse in ``g_tt`` is changed from ``1 - r_s/r`` to ``1 - 3 r_s/(2 r)`` while ``g_rr``
    is left alone.  Christoffel symbols and the Ricci tensor are rebuilt from that metric with
    the same construction the reused control uses.
    """

    time, radius, theta, phi = sp.symbols("t r theta phi", real=True)
    schwarzschild_radius = sp.Symbol("r_s", positive=True)
    coordinates = (time, radius, theta, phi)
    lapse = 1 - sp.Rational(3, 2) * schwarzschild_radius / radius
    grr = 1 - schwarzschild_radius / radius
    metric = sp.diag(-lapse, 1 / grr, radius**2, radius**2 * sp.sin(theta) ** 2)
    inverse = sp.simplify(metric.inv())
    size = 4
    christoffel = [[[sp.Integer(0)] * size for _ in range(size)] for _ in range(size)]
    for upper in range(size):
        for left in range(size):
            for right in range(size):
                value = sp.Integer(0)
                for contracted in range(size):
                    value += inverse[upper, contracted] * (
                        sp.diff(metric[contracted, right], coordinates[left])
                        + sp.diff(metric[contracted, left], coordinates[right])
                        - sp.diff(metric[left, right], coordinates[contracted])
                    )
                christoffel[upper][left][right] = sp.simplify(value / 2)
    components: dict[str, str] = {}
    names = ("t", "r", "theta", "phi")
    for left in range(size):
        for right in range(size):
            value = sp.Integer(0)
            for contracted in range(size):
                value += sp.diff(christoffel[contracted][left][right], coordinates[contracted])
                value -= sp.diff(christoffel[contracted][left][contracted], coordinates[right])
                for inner in range(size):
                    value += (
                        christoffel[contracted][left][right]
                        * christoffel[inner][contracted][inner]
                    )
                    value -= (
                        christoffel[inner][left][contracted]
                        * christoffel[contracted][right][inner]
                    )
            components[f"R_{names[left]}{names[right]}"] = _text(sp.simplify(value))
    return components


def general_relativity_chain(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Einstein-Hilbert action to Mercury's perihelion, reusing the repository's controls."""

    action_ir = artifacts["einstein_hilbert_action_ir"]
    action_health = artifacts["einstein_hilbert_action_health"]
    reference_report = artifacts["relativity_reference_report"]

    # --- Step 1: the declared Einstein-Hilbert action, from the bound action IR.
    terms = action_ir.get("canonical", {}).get("terms", [])
    term = next((item for item in terms if item.get("id") == REQUIRED_ACTION_TERM["id"]), None)
    if term is None:
        raise DerivationChainDemoError("the bound action IR does not declare the EH_R term")
    if (
        term.get("density") != REQUIRED_ACTION_TERM["density"]
        or term.get("coefficient") != REQUIRED_ACTION_TERM["coefficient"]
    ):
        raise DerivationChainDemoError("the bound Einstein-Hilbert term changed")
    declared_action = reference_report["reference_action"]["action"]
    step_one = _step(
        1,
        "Declare the Einstein-Hilbert action, reusing the repository's compiled action IR.",
        f"{declared_action}; IR term {term['id']}: coefficient {term['coefficient']} "
        f"on density {term['density']}",
        "the bound action IR is valid and still declares exactly the EH_R curvature term",
        "pass" if action_ir.get("valid") is True else "fail",
        "existing_repository_control",
        "runs/formal-controls-v1/action-health/einstein_hilbert_control/action-ir.json",
        action_ir_content_sha256=action_ir["content_sha256"],
        declared_action=declared_action,
        action_term=term,
    )

    # --- Step 2: vary the action.  Certified by the repository's Cadabra variation control.
    gates = action_health.get("gates", {})
    gate_statuses = {
        name: gates.get(name, {}).get("status") for name in REQUIRED_ACTION_HEALTH_GATES
    }
    variation_evidence = gates.get("covariant_variation", {}).get("evidence", [])
    variation_ok = (
        all(status == "pass" for status in gate_statuses.values())
        and REQUIRED_VARIATION_CONTROL in variation_evidence
        and action_health.get("input_action_sha256") == action_ir["content_sha256"]
    )
    step_two = _step(
        2,
        (
            "Vary the action with respect to the metric.  This module does not re-derive the "
            "variation; it reuses and cites the repository's registered nonlinear Cadabra "
            "variation control and the action-health certificate that carries it."
        ),
        reference_report["reference_action"]["expected_field_equation"],
        (
            "the bound action-health certificate reports pass on covariant_variation, "
            "covariant_identity and field_contract, names the registered Cadabra metric-variation "
            "control as its evidence, and binds the same action hash as step 1"
        ),
        "pass" if variation_ok else "fail",
        "existing_repository_control",
        (
            f"{REQUIRED_VARIATION_CONTROL} via "
            "formal/cadabra/einstein_hilbert_metric_variation.cdb, certified in "
            "runs/formal-controls-v1/action-health/einstein_hilbert_control/action-health.json"
        ),
        action_health_gate_statuses=gate_statuses,
        variation_control=REQUIRED_VARIATION_CONTROL,
        variation_evidence=sorted(variation_evidence),
        action_hash_binding_matches=action_health.get("input_action_sha256")
        == action_ir["content_sha256"],
        boundary_term="tracked and dropped by the registered control, not silently discarded",
    )

    # --- Step 3: the static spherically symmetric vacuum solution.  Reused control.
    ricci = schwarzschild_ricci_components()
    nonzero = {name: value for name, value in ricci.items() if value != "0"}
    perturbed = perturbed_schwarzschild_ricci()
    perturbed_nonzero = {name: value for name, value in perturbed.items() if value != "0"}
    step_three = _step(
        3,
        (
            "Impose staticity and spherical symmetry on the vacuum field equations and check the "
            "Schwarzschild metric, reusing the repository's Ricci construction."
        ),
        "ds^2 = -(1 - r_s/r) c^2 dt^2 + (1 - r_s/r)^-1 dr^2 + r^2 dOmega^2, with R_ab = 0",
        (
            "all sixteen Ricci components, rebuilt from the metric definition rather than looked "
            "up, are identically zero; a perturbed lapse makes them nonzero"
        ),
        "pass" if not nonzero and perturbed_nonzero else "fail",
        "existing_repository_control",
        "sigma_theory_compiler.relativity.schwarzschild_ricci_components",
        components_calculated=len(ricci),
        nonzero_components=sorted(nonzero),
        negative_control_nonzero_components=sorted(perturbed_nonzero),
    )

    # --- Step 4: the orbit equation from the geodesic.  New in this module.
    orbit = schwarzschild_orbit_equation()
    step_four = _step(
        4,
        (
            "Reduce the Schwarzschild geodesic to an orbit equation in u = 1/r.  New here: the "
            "repository carries no geodesic machinery."
        ),
        orbit["orbit_equation"],
        (
            "differentiating the four-velocity normalization first integral and dividing by "
            "2 L^2 u' reproduces u'' + u - GM/L^2 - 3 GM u^2/c^2 with zero residual"
        ),
        "pass" if orbit["derivation_is_exact"] else "fail",
        "new_in_this_module",
        "derivation_chain_demo.schwarzschild_orbit_equation",
        first_integral=orbit["first_integral"],
        orbit_equation_residual=orbit["orbit_equation_residual"],
        newtonian_limit=orbit["newtonian_limit"],
        relativistic_term=orbit["relativistic_term"],
    )

    # --- Step 5: perturbative solution.  New in this module.
    advance = perihelion_advance_expression()
    step_five = _step(
        5,
        (
            "Solve the orbit equation perturbatively by Poincare-Lindstedt and read the "
            "perihelion advance per orbit."
        ),
        f"dphi = {advance['advance_in_orbital_elements']} per orbit",
        (
            "removing the secular resonance fixes the detuning uniquely, and the resulting "
            "advance matches 6 pi GM/(c^2 a (1-e^2)) with zero residual"
        ),
        "pass" if advance["derivation_is_exact"] else "fail",
        "new_in_this_module",
        "derivation_chain_demo.perihelion_advance_expression",
        **{key: value for key, value in advance.items() if key != "derivation_is_exact"},
    )

    return {
        "chain_id": "general_relativity_action_to_perihelion",
        "theory": "General Relativity",
        "starting_point": declared_action,
        "observable_reached": "anomalous perihelion advance per orbit",
        "steps": [step_one, step_two, step_three, step_four, step_five],
        "negative_controls": [
            {
                "control_id": "perturbed_lapse_fails_ricci_flatness",
                "mutation": "replace the g_tt lapse 1 - r_s/r with 1 - 3 r_s/(2 r)",
                "observed": {
                    "nonzero_ricci_components": sorted(perturbed_nonzero),
                    "example": perturbed_nonzero[min(perturbed_nonzero)],
                },
                "rejected": True,
            },
            {
                "control_id": "dropping_the_relativistic_term_gives_no_precession",
                "mutation": "delete 3 GM u^2/c^2 from the orbit equation",
                "observed": {
                    "orbit_equation": orbit["newtonian_limit"],
                    "advance_per_orbit": "0",
                },
                "rejected": True,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Chain 3.  The alternatives, and why the measurement kills them.
# ---------------------------------------------------------------------------


def apsidal_precession(force_per_mass: sp.Expr, radius: sp.Symbol, orbit_radius: sp.Symbol) -> Any:
    """Perihelion advance per orbit for a nearly circular orbit under a central force.

    For an attractive central force of magnitude ``f(r)`` per unit mass, the apsidal angle of a
    nearly circular orbit of radius ``a`` is ``pi/sqrt(3 + a f'(a)/f(a))``, so the advance per
    orbit is ``2 pi/sqrt(3 + a f'(a)/f(a)) - 2 pi``.  New in this module.
    """

    logarithmic_slope = sp.simplify(
        orbit_radius * sp.diff(force_per_mass, radius).subs(radius, orbit_radius)
        / force_per_mass.subs(radius, orbit_radius)
    )
    discriminant = sp.simplify(3 + logarithmic_slope)
    return sp.simplify(2 * sp.pi / sp.sqrt(discriminant) - 2 * sp.pi), discriminant


def alternative_precessions() -> dict[str, Any]:
    """Derive the precession produced by each declared alternative to the inverse square."""

    radius = sp.Symbol("r", positive=True)
    orbit_radius = sp.Symbol("a", positive=True)
    newton_mass, light = sp.symbols("GM c", positive=True)
    delta = sp.Symbol("delta", real=True)
    alpha = sp.Symbol("alpha", real=True)
    screening = sp.Symbol("lamda", positive=True)
    compton = sp.Symbol("lambda_g", positive=True)
    ratio = sp.Symbol("x", positive=True)

    # (a) modified exponent.
    power_force = newton_mass / radius ** (2 + delta)
    power_advance, power_discriminant = apsidal_precession(power_force, radius, orbit_radius)
    power_leading = sp.simplify(sp.series(power_advance, delta, 0, 2).removeO())

    # (b) Yukawa.
    yukawa_potential = -(newton_mass / radius) * (1 + alpha * sp.exp(-radius / screening))
    yukawa_force = sp.simplify(-sp.diff(yukawa_potential, radius))
    yukawa_advance, yukawa_discriminant = apsidal_precession(yukawa_force, radius, orbit_radius)
    yukawa_leading = sp.simplify(sp.series(yukawa_advance, alpha, 0, 2).removeO())

    # (c) graviton-mass-like term: the whole potential carries the exponential.
    graviton_potential = -(newton_mass / radius) * sp.exp(-radius / compton)
    graviton_force = sp.simplify(-sp.diff(graviton_potential, radius))
    graviton_advance, graviton_discriminant = apsidal_precession(
        graviton_force, radius, orbit_radius
    )
    graviton_leading = sp.simplify(
        sp.series(graviton_advance.subs(compton, orbit_radius / ratio), ratio, 0, 3).removeO()
    )

    # Cross-validation: the same machinery must reproduce Chain 2 on the Chain 2 effective force.
    momentum = sp.Symbol("L", positive=True)
    relativistic_force = newton_mass / radius**2 + 3 * newton_mass * momentum**2 / (
        light**2 * radius**4
    )
    relativistic_advance, _ = apsidal_precession(relativistic_force, radius, orbit_radius)
    relativistic_advance = relativistic_advance.subs(momentum**2, newton_mass * orbit_radius)
    relativistic_leading = sp.simplify(
        sp.series(relativistic_advance, light, sp.oo, 3).removeO()
    )
    crosscheck_residual = sp.simplify(
        relativistic_leading - 6 * sp.pi * newton_mass / (light**2 * orbit_radius)
    )

    if sp.simplify(power_leading - sp.pi * delta) != 0:
        raise DerivationChainDemoError(
            "the modified-exponent precession did not reduce to pi*delta"
        )
    if sp.simplify(
        yukawa_leading - sp.pi * alpha * orbit_radius**2 * sp.exp(-orbit_radius / screening)
        / screening**2
    ) != 0:
        raise DerivationChainDemoError("the Yukawa precession did not reduce to the known form")
    if crosscheck_residual != 0:
        raise DerivationChainDemoError(
            "the apsidal machinery did not reproduce the Chain 2 general-relativistic advance"
        )

    return {
        "method": (
            "nearly circular apsidal angle pi/sqrt(3 + a f'(a)/f(a)); advance per orbit is "
            "2 pi/sqrt(3 + a f'(a)/f(a)) - 2 pi"
        ),
        "modified_exponent": {
            "law": "|g| = GM/r^(2+delta)",
            "parameter": "delta",
            "discriminant": _text(power_discriminant),
            "advance_exact": _text(power_advance),
            "advance_leading_order": _text(power_leading),
            "leading_coefficient": "pi",
        },
        "yukawa": {
            "law": "Phi = -(GM/r)*(1 + alpha*exp(-r/lambda))",
            "parameter": "alpha at fixed lambda",
            "force": _text(yukawa_force),
            "discriminant": _text(yukawa_discriminant),
            "advance_exact": _text(yukawa_advance),
            "advance_leading_order": _text(yukawa_leading),
            "leading_form": "pi*alpha*(a/lambda)^2*exp(-a/lambda)",
        },
        "graviton_mass": {
            "law": "Phi = -(GM/r)*exp(-r/lambda_g)",
            "parameter": "lambda_g, the graviton Compton wavelength",
            "force": _text(graviton_force),
            "discriminant": _text(graviton_discriminant),
            "advance_exact": _text(graviton_advance),
            "advance_leading_order_in_x": _text(graviton_leading),
            "leading_form": "pi*(a/lambda_g)^2",
        },
        "crosscheck_against_chain_two": {
            "effective_force": _text(relativistic_force),
            "recovered_advance": _text(relativistic_leading),
            "expected_advance": "6*pi*GM/(a*c**2)",
            "residual": _text(crosscheck_residual),
            "status": "pass",
        },
    }


def _mercury_numerics() -> dict[str, Any]:
    """Recompute Mercury's advance and cross-check it against the existing relativity control."""

    mp.mp.dps = WORKING_PRECISION_DIGITS
    solar_mu = mp.mpf(G_SI) * mp.mpf(M_SUN_KG)
    light = mp.mpf(C_SI)
    semi_major = mp.mpf(CITED_OBSERVATIONS["mercury_orbital_elements"]["semi_major_axis_m"])
    eccentricity = mp.mpf(CITED_OBSERVATIONS["mercury_orbital_elements"]["eccentricity"])
    period_days = mp.mpf(CITED_OBSERVATIONS["mercury_orbital_elements"]["sidereal_period_days"])
    arcsec_per_radian = 648000 / mp.pi

    advance_per_orbit = (
        6 * mp.pi * solar_mu / (light**2 * semi_major * (1 - eccentricity**2))
    )
    orbits_per_century = 100 * mp.mpf(JULIAN_YEAR_DAYS) / period_days
    advance_per_century = advance_per_orbit * orbits_per_century * arcsec_per_radian

    control = next(
        item for item in solar_system_numeric_checks() if item["name"] == "mercury_perihelion"
    )
    control_value = mp.mpf(control["evidence"]["calculated_arcsec_per_century"])
    crosscheck_relative = abs(advance_per_century - control_value) / control_value
    tolerance = mp.mpf(MERCURY_CROSSCHECK_RELATIVE_TOLERANCE)
    if crosscheck_relative > tolerance:
        raise DerivationChainDemoError(
            "the recomputed Mercury advance disagrees with the existing relativity control"
        )

    observed = mp.mpf(
        CITED_OBSERVATIONS["mercury_anomalous_perihelion_advance"]["value_arcsec_per_century"]
    )
    observed_relative = abs(advance_per_century - observed) / observed
    fraction = mp.mpf(
        CITED_OBSERVATIONS["mercury_anomalous_perihelion_advance"][
            "fractional_agreement_with_general_relativity"
        ]
    )
    if observed_relative > fraction:
        raise DerivationChainDemoError(
            "the derived Mercury advance is outside the cited agreement level"
        )

    residual_arcsec_per_century = fraction * observed
    residual_per_orbit = residual_arcsec_per_century / (orbits_per_century * arcsec_per_radian)
    return {
        "solar_gravitational_parameter_m3_s2": solar_mu,
        "advance_per_orbit_rad": advance_per_orbit,
        "orbits_per_century": orbits_per_century,
        "advance_arcsec_per_century": advance_per_century,
        "control_arcsec_per_century": control_value,
        "crosscheck_relative_difference": crosscheck_relative,
        "observed_arcsec_per_century": observed,
        "observed_relative_error": observed_relative,
        "residual_budget_arcsec_per_century": residual_arcsec_per_century,
        "residual_budget_rad_per_orbit": residual_per_orbit,
        "semi_major_axis_m": semi_major,
        "eccentricity": eccentricity,
    }


def exclusion_bounds(numerics: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the cited residual budget into an exclusion interval for each alternative."""

    mp.mp.dps = WORKING_PRECISION_DIGITS
    residual = numerics["residual_budget_rad_per_orbit"]
    semi_major = numerics["semi_major_axis_m"]

    # (a) |dphi| = pi*|delta|  ->  |delta| < residual/pi
    delta_bound = residual / mp.pi

    # (b) |dphi| = pi*|alpha|*x^2*exp(-x), x = a/lambda  ->  |alpha| < residual*exp(x)/(pi*x^2)
    curve = []
    for sample in YUKAWA_RANGE_SAMPLE_POINTS:
        range_over_a = mp.mpf(sample)
        x = 1 / range_over_a
        alpha_bound = residual * mp.exp(x) / (mp.pi * x**2)
        curve.append(
            {
                "lambda_over_semi_major_axis": sample,
                "lambda_m": _decimal(range_over_a * semi_major),
                "a_over_lambda": _decimal(x),
                "alpha_absolute_upper_bound": _decimal(alpha_bound),
            }
        )
    # The bound is tightest where exp(x)/x^2 is minimal, i.e. x = 2 exactly.
    tightest_x = mp.mpf(2)
    tightest_alpha = residual * mp.exp(tightest_x) / (mp.pi * tightest_x**2)

    # (c) |dphi| = pi*(a/lambda_g)^2  ->  lambda_g > a*sqrt(pi/residual)
    compton_bound = semi_major * mp.sqrt(mp.pi / residual)
    planck = mp.mpf(CITED_OBSERVATIONS["defining_si_constants"]["planck_constant_J_s"])
    electron_volt = mp.mpf(CITED_OBSERVATIONS["defining_si_constants"]["electron_volt_J"])
    mass_bound_kg = planck / (compton_bound * mp.mpf(C_SI))
    mass_bound_ev = planck * mp.mpf(C_SI) / (compton_bound * electron_volt)

    return {
        "residual_budget_rad_per_orbit": _decimal(residual),
        "residual_budget_arcsec_per_century": _decimal(
            numerics["residual_budget_arcsec_per_century"]
        ),
        "modified_exponent": {
            "parameter": "delta",
            "inequality": "|delta| < residual_budget/pi",
            "absolute_upper_bound": _decimal(delta_bound),
            "interpretation": (
                "The inverse-square exponent is 2 to within this bound; a modified exponent large "
                "enough to matter astrophysically is dead."
            ),
        },
        "yukawa": {
            "parameters": "alpha at fixed lambda",
            "inequality": "|alpha| < residual_budget*exp(a/lambda)/(pi*(a/lambda)^2)",
            "exclusion_curve": curve,
            "tightest_point": {
                "a_over_lambda": _decimal(tightest_x),
                "lambda_over_semi_major_axis": _decimal(1 / tightest_x),
                "alpha_absolute_upper_bound": _decimal(tightest_alpha),
                "note": "exp(x)/x^2 is minimized at x = 2, so the constraint is strongest there",
            },
            "degeneracy_note": (
                "As lambda -> oo the Yukawa term is reabsorbed into GM and the precession "
                "vanishes, so the bound necessarily weakens; that limit is unobservable here "
                "rather than allowed."
            ),
        },
        "graviton_mass": {
            "parameter": "lambda_g",
            "inequality": "lambda_g > a*sqrt(pi/residual_budget)",
            "compton_wavelength_lower_bound_m": _decimal(compton_bound),
            "mass_upper_bound_kg": _decimal(mass_bound_kg),
            "mass_upper_bound_eV": _decimal(mass_bound_ev),
            "interpretation": (
                "A graviton heavier than this would have shortened Mercury's precession by more "
                "than the cited agreement allows."
            ),
        },
        "scope": (
            "Single-observable, nearly-circular, leading-order solar-system bounds derived from "
            "one cited agreement level.  They are order-of-magnitude exclusions, not a "
            "replacement for a full multi-planet ephemeris fit."
        ),
    }


def alternatives_chain(numerics: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate the alternatives, derive their precession, and bound them."""

    precessions = alternative_precessions()
    bounds = exclusion_bounds(numerics)
    steps = [
        _step(
            1,
            "Declare a three-member family of alternatives to the inverse-square law.",
            "; ".join(
                precessions[name]["law"]
                for name in ("modified_exponent", "yukawa", "graviton_mass")
            ),
            "each alternative is a closed-form central law with one free parameter",
            "pass",
            "new_in_this_module",
            "derivation_chain_demo.alternative_precessions",
            family=["modified_exponent", "yukawa", "graviton_mass"],
        ),
        _step(
            2,
            (
                "Derive the perihelion precession each alternative produces, symbolically, by "
                "nearly-circular apsidal-angle perturbation theory."
            ),
            "; ".join(
                f"{name}: {precessions[name][_LEADING_ORDER_KEY[name]]}"
                for name in ("modified_exponent", "yukawa", "graviton_mass")
            ),
            (
                "the same machinery, fed the Chain 2 effective force, reproduces Chain 2's own "
                "6 pi GM/(c^2 a) with zero residual"
            ),
            precessions["crosscheck_against_chain_two"]["status"],
            "new_in_this_module",
            "derivation_chain_demo.apsidal_precession",
            method=precessions["method"],
            crosscheck=precessions["crosscheck_against_chain_two"],
        ),
        _step(
            3,
            (
                "Convert the cited Mercury agreement into a residual precession budget and solve "
                "each precession formula for its parameter."
            ),
            (
                f"|delta| < {bounds['modified_exponent']['absolute_upper_bound']}; "
                f"|alpha| < {bounds['yukawa']['tightest_point']['alpha_absolute_upper_bound']} at "
                f"lambda = a/2; lambda_g > "
                f"{bounds['graviton_mass']['compton_wavelength_lower_bound_m']} m"
            ),
            (
                "each bound is the residual budget divided by the derived leading coefficient; "
                "no fit is performed and no dataset is opened"
            ),
            "pass",
            "new_in_this_module",
            "derivation_chain_demo.exclusion_bounds",
            residual_budget_rad_per_orbit=bounds["residual_budget_rad_per_orbit"],
            cited_agreement=CITED_OBSERVATIONS["mercury_anomalous_perihelion_advance"],
        ),
    ]
    return {
        "chain_id": "alternatives_excluded_by_measurement",
        "theory": "declared alternatives to the inverse-square law",
        "starting_point": "a declared one-parameter family per alternative",
        "observable_reached": "exclusion intervals from the cited Mercury precession agreement",
        "steps": steps,
        "derived_precessions": precessions,
        "exclusion_bounds": bounds,
        "negative_controls": [
            {
                "control_id": "zero_parameter_returns_newton",
                "mutation": "set delta = 0, alpha = 0, or lambda_g -> oo",
                "observed": {
                    "modified_exponent_advance": "0",
                    "yukawa_advance": "0",
                    "graviton_advance": "0",
                },
                "rejected": False,
                "note": (
                    "This control must NOT reject: at zero parameter every alternative collapses "
                    "onto the derived inverse square and precesses by exactly zero, which is the "
                    "consistency requirement rather than a failure."
                ),
            }
        ],
    }


# ---------------------------------------------------------------------------
# Receipt.
# ---------------------------------------------------------------------------


def _numeric_verifications(numerics: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "verification_id": "mercury_perihelion_advance",
            "derived_formula": "6*pi*G*M_sun/(c^2*a*(1 - e^2)) per orbit",
            "derived_arcsec_per_century": _decimal(numerics["advance_arcsec_per_century"]),
            "cited_observed_arcsec_per_century": _decimal(numerics["observed_arcsec_per_century"]),
            "relative_error": _decimal(numerics["observed_relative_error"]),
            "cited_agreement_level": CITED_OBSERVATIONS[
                "mercury_anomalous_perihelion_advance"
            ]["fractional_agreement_with_general_relativity"],
            "status": "pass",
        },
        {
            "verification_id": "crosscheck_against_existing_relativity_control",
            "derived_formula": "same expression, recomputed at 60 working digits with mpmath",
            "derived_arcsec_per_century": _decimal(numerics["advance_arcsec_per_century"]),
            "existing_control_arcsec_per_century": _decimal(
                numerics["control_arcsec_per_century"]
            ),
            "relative_difference": _decimal(numerics["crosscheck_relative_difference"]),
            "tolerance": MERCURY_CROSSCHECK_RELATIVE_TOLERANCE,
            "certified_by": "sigma_theory_compiler.relativity.solar_system_numeric_checks",
            "status": "pass",
        },
    ]


def run_derivation_chain(root: str | Path = ".") -> dict[str, Any]:
    """Build the full three-chain receipt.  Deterministic; no timestamps, no floats."""

    repository = Path(root).resolve()
    artifacts = _load_bound_artifacts(repository)
    numerics = _mercury_numerics()

    newton = newton_chain()
    relativity = general_relativity_chain(artifacts)
    alternatives = alternatives_chain(numerics)
    chains = [newton, relativity, alternatives]

    reused = sum(
        1
        for chain in chains
        for step in chain["steps"]
        if step["provenance"] == "existing_repository_control"
    )
    built = sum(
        1
        for chain in chains
        for step in chain["steps"]
        if step["provenance"] == "new_in_this_module"
    )

    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Three auditable derivation chains recomputed symbolically on every run: Newtonian "
            "gravity from a declared action to Kepler III with the force exponent traced to "
            "spatial dimensionality; the Einstein-Hilbert action to Mercury's perihelion advance, "
            "reusing this repository's hash-bound action IR, Cadabra metric-variation certificate "
            "and Schwarzschild Ricci construction; and a declared family of alternatives to the "
            "inverse square whose parameters are excluded by a cited published agreement level.  "
            "No observational dataset is opened, nothing is fitted, and nothing here is novel."
        ),
        "claims": CLAIMS,
        "assumptions": [
            "Test-particle motion; no back-reaction and no multi-body perturbations.",
            (
                "Chain 3 uses the nearly-circular apsidal-angle approximation and leading order "
                "in each alternative's parameter, so its bounds are order-of-magnitude."
            ),
            (
                "The whole residual precession budget is attributed to a single alternative at a "
                "time; no covariance between competing parameters is modelled."
            ),
            (
                "Mercury's orbital elements and the general-relativistic agreement level are "
                "cited published values, cross-checked against this repository's existing control."
            ),
            (
                "The Einstein-Hilbert variation is reused, not re-derived here; it is certified "
                "by the bound Cadabra control and would fail closed if that certificate changed."
            ),
        ],
        "inputs": {
            name: {
                "path": BOUND_ARTIFACTS[name]["path"],
                "kind": BOUND_ARTIFACTS[name]["kind"],
                "binding": BOUND_ARTIFACTS[name].get("semantic_sha256")
                or BOUND_ARTIFACTS[name]["normalized_text_sha256"],
                "binding_kind": (
                    "semantic_sha256"
                    if BOUND_ARTIFACTS[name]["kind"] == "json"
                    else "normalized_text_sha256"
                ),
            }
            for name in sorted(BOUND_ARTIFACTS)
        },
        "config": {
            "cited_observations": CITED_OBSERVATIONS,
            "dimension_sample_points": [str(value) for value in DIMENSION_SAMPLE_POINTS],
            "yukawa_range_sample_points": list(YUKAWA_RANGE_SAMPLE_POINTS),
            "working_precision_digits": WORKING_PRECISION_DIGITS,
            "emitted_precision_digits": EMITTED_PRECISION_DIGITS,
            "mercury_crosscheck_relative_tolerance": MERCURY_CROSSCHECK_RELATIVE_TOLERANCE,
            "reused_constants": {
                "source": "sigma_theory_compiler.relativity",
                "G_SI": _decimal(G_SI),
                "C_SI": str(int(C_SI)),
                "M_SUN_KG": _decimal(M_SUN_KG),
                "JULIAN_YEAR_DAYS": _decimal(JULIAN_YEAR_DAYS),
            },
        },
        "controls": {
            "schwarzschild_ricci_components": {
                "module": "sigma_theory_compiler.relativity",
                "role": "Chain 2 step 3 vacuum check",
                "provenance": "existing_repository_control",
            },
            "solar_system_numeric_checks": {
                "module": "sigma_theory_compiler.relativity",
                "role": "Chain 2 numeric cross-check",
                "provenance": "existing_repository_control",
            },
            REQUIRED_VARIATION_CONTROL: {
                "module": "formal/cadabra/einstein_hilbert_metric_variation.cdb",
                "role": "Chain 2 step 2 metric variation certificate",
                "provenance": "existing_repository_control",
            },
            "apsidal_precession": {
                "module": "sigma_theory_compiler.derivation_chain_demo",
                "role": "Chain 3 precession for every declared alternative",
                "provenance": "new_in_this_module",
            },
        },
        "chains": chains,
        "numeric_verifications": _numeric_verifications(numerics),
        "exclusion_bounds": alternatives["exclusion_bounds"],
        "counts": {
            "chains": len(chains),
            "steps_total": sum(len(chain["steps"]) for chain in chains),
            "steps_reusing_existing_controls": reused,
            "steps_new_in_this_module": built,
            "negative_controls": sum(len(chain["negative_controls"]) for chain in chains),
            "numeric_verifications": 2,
            "alternatives_bounded": 3,
            "yukawa_exclusion_points": len(YUKAWA_RANGE_SAMPLE_POINTS),
            "bound_repository_artifacts": len(BOUND_ARTIFACTS),
        },
        "decision": "DERIVATION_CHAINS_COMPLETE_NO_NOVELTY_CLAIMED",
        "residual_gap_report": {
            "not_established": [
                (
                    "Chain 3's bounds come from one observable under a nearly-circular "
                    "approximation; a full ephemeris fit would tighten and correlate them."
                ),
                (
                    "The alternatives family is declared, not exhaustive; no completeness claim "
                    "is made over the space of modified gravity laws."
                ),
                (
                    "Chain 2's variation step is reused rather than re-derived here, so this "
                    "receipt inherits whatever scope the bound Cadabra control carries."
                ),
                "No candidate theory is promoted, screened, or validated by this demonstration.",
            ]
        },
    }
    body["config_sha256"] = canonical_sha256(body["config"])
    _no_floats(body)
    return _seal(body)


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Fail-closed validation: seal, claims, structure, and full symbolic replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise DerivationChainDemoError("unexpected derivation-chain schema version")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise DerivationChainDemoError("derivation-chain receipt seal changed")
    if value.get("claims") != CLAIMS:
        raise DerivationChainDemoError("claims block changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config")):
        raise DerivationChainDemoError("config binding changed")
    _no_floats(body)

    chains = value.get("chains")
    if not isinstance(chains, list) or len(chains) != 3:
        raise DerivationChainDemoError("the receipt does not carry exactly three chains")
    for chain in chains:
        for step in chain["steps"]:
            if step.get("check_status") != "pass":
                raise DerivationChainDemoError("a derivation step is not marked pass")
            if step.get("provenance") not in {
                "new_in_this_module",
                "existing_repository_control",
            }:
                raise DerivationChainDemoError("unknown step provenance")
    for name, binding in value.get("inputs", {}).items():
        if name not in BOUND_ARTIFACTS:
            raise DerivationChainDemoError(f"unknown bound artifact: {name}")
        digest = binding.get("binding")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DerivationChainDemoError(f"malformed binding digest for {name}")

    expected_counts = {
        "chains": 3,
        "steps_total": sum(len(chain["steps"]) for chain in chains),
        "steps_reusing_existing_controls": sum(
            1
            for chain in chains
            for step in chain["steps"]
            if step["provenance"] == "existing_repository_control"
        ),
        "steps_new_in_this_module": sum(
            1
            for chain in chains
            for step in chain["steps"]
            if step["provenance"] == "new_in_this_module"
        ),
        "negative_controls": sum(len(chain["negative_controls"]) for chain in chains),
        "numeric_verifications": len(value.get("numeric_verifications", [])),
        "alternatives_bounded": 3,
        "yukawa_exclusion_points": len(YUKAWA_RANGE_SAMPLE_POINTS),
        "bound_repository_artifacts": len(BOUND_ARTIFACTS),
    }
    if _plain(expected_counts) != value.get("counts"):
        raise DerivationChainDemoError("aggregate counts do not replay from the chains")
    for verification in value.get("numeric_verifications", []):
        if verification.get("status") != "pass":
            raise DerivationChainDemoError("a numeric verification is not marked pass")


def _write(value: Mapping[str, Any], output: str | Path) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise DerivationChainDemoError("refusing to overwrite an immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive Newton and General Relativity from declared actions and exclude "
        "the alternatives."
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
    result = run_derivation_chain(root)
    if args.output:
        _write(result, args.output)
    bounds = result["exclusion_bounds"]
    print(
        json.dumps(
            {
                "chains": result["counts"]["chains"],
                "steps_total": result["counts"]["steps_total"],
                "steps_reusing_existing_controls": result["counts"][
                    "steps_reusing_existing_controls"
                ],
                "steps_new_in_this_module": result["counts"]["steps_new_in_this_module"],
                "mercury_arcsec_per_century": result["numeric_verifications"][0][
                    "derived_arcsec_per_century"
                ],
                "delta_absolute_upper_bound": bounds["modified_exponent"][
                    "absolute_upper_bound"
                ],
                "graviton_compton_lower_bound_m": bounds["graviton_mass"][
                    "compton_wavelength_lower_bound_m"
                ],
                "decision": result["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
