"""Blinded rediscovery campaign over twelve classically solved sequence problems.

Twelve integer sequences whose closed forms or recurrences are classical, solved
mathematics are presented to the discovery engine as raw anonymized data only: each
world exposes exact public rows under the neutral name ``a`` and an opaque world id
(``world_01`` .. ``world_12``).  The classical statements, names, and attributions
live in a separate sealed fixture that Phase A is forbidden to read; the config
carries only salted SHA-256 commitments to those records.

Phase A (chronology sealed) runs the declared stage ladder per world: B1 basis
synthesis; B7 structural repair when B1 blocks; B3 conjecture generation; and, for
the single world with a config-declared public transformation, adjacent-term ratio
rows plus B2 nonlinear coefficient search.  The best surviving statement is frozen
as the candidate, provers are routed (B5 lemma decomposition with in-receipt Lean
for polynomial closed forms; B6 quantified inequalities for surviving monotonicity
and sign statements via a declared nat-scaled companion), and the Phase A root is
sealed.  Only then is the target fixture read — once, atomically.  Every commitment
must open, and each frozen candidate is compared exactly against its classical
target: sympy equivalence over the declared domain for closed forms, exact
coefficient equality for recurrences and ratio laws.  Verdicts are
REDISCOVERED_WITH_PROOF, REDISCOVERED_EXACT, PARTIAL, or MISSED.  Zero candidates
are generated or tuned after unseal.

A world whose classical closed form lies outside the declared rational grammar
(Fibonacci, Lucas, Pell — Binet-type irrational forms) is scored on its recurrence,
and the engine's refusal to claim a closed form there is recorded as correct
behavior, not failure.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import lcm
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import sympy as sp

from .basis_synthesis import LADDER, Term, _solve_unique, synthesize_basis
from .conjecture_generation import STATEMENT_KINDS_V1, generate_conjectures
from .lemma_decomposition import decompose_closed_form_proof
from .nonlinear_coefficient_search import search_nonlinear
from .quantified_inequality_proofs import prove_quantified_inequality
from .sigma_core import canonical_json_bytes, canonical_sha256
from .structural_repair import repair_structure

CONFIG_SCHEMA = "invariant-dozen-blind-rediscovery-config-1.0"
TARGET_SCHEMA = "invariant-dozen-blind-targets-1.0"
RESULT_SCHEMA = "invariant-dozen-blind-rediscovery-result-1.0"
WORLD_SCHEMA = "invariant-dozen-blind-world-receipt-1.0"
CAMPAIGN_ID = "dozen-blind-rediscovery-001"
CONFIG_PATH = "configs/backgrounds/dozen_blind_rediscovery_v1.json"
TARGETS_PATH = "configs/backgrounds/dozen_blind_targets_v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/dozen_blind_rediscovery_campaign.py"
TEST_PATH = "tests/test_dozen_blind_rediscovery_campaign.py"
OUTPUT_PATH = "runs/math/solved-dozen/campaign.json"
WORLD_RECEIPT_DIRECTORY = "runs/math/solved-dozen"
SEQUENCE_ALIAS = "a"
WORLD_COUNT = 12
WORLD_IDS = tuple(f"world_{index:02d}" for index in range(1, WORLD_COUNT + 1))
RATIO_TRANSFORMATION_ID = "adjacent_term_ratio"
MAX_RECURRENCE_ORDER = 3
MAX_COMPANION_SHIFT = 8
NON_CLAIM_NOTE = "closed form outside declared rational grammar, correctly not claimed"
CANDIDATE_SELECTION_RULE = (
    "first_of[b1_closed_form, declared_transformation_b2_law, "
    "b3_linear_recurrence_survivor, b7_repaired_statement]"
)
POLICIES = {
    "atomic_target_unseal_batches": 1,
    "candidate_generation_after_unseal": 0,
    "minimum_rediscoveries_for_pass": 8,
    "target_reads_before_candidate_freeze": 0,
}
CLAIMS = {
    "atomic_target_unseal_batches": 1,
    "closed_forms_claimed_outside_declared_grammar": False,
    "kernel_verified_lean": False,
    "machine_found_targets_unaided": True,
    "novelty_claimed": False,
    "post_unseal_generation": False,
    "rediscovery_of_classical_results": True,
    "target_records_read_before_candidate_freeze": 0,
}
SCOPE = (
    "Blinded rediscovery of twelve classically solved integer-sequence results from raw "
    "anonymized public rows only. Stage inputs are exact rows under the neutral alias 'a' "
    "and opaque world ids; classical statements, names, and attributions stay in a sealed "
    "fixture whose in-process reads are denied until the single post-freeze atomic unseal. "
    "The one derived-row construction (adjacent-term ratios) is declared in the public "
    "config and logged, and every stage grammar is declared and finite, so a BLOCK means "
    "'outside the declared grammar', never 'impossible'. Emitted Lean is exact-locally "
    "checked here and must be independently re-proved by the pinned Lean kernel in CI; "
    "kernel_verified is false in this receipt. Rediscovery of solved mathematics is the "
    "entire claim: no novelty, priority, or scientific significance is asserted."
)


class DozenBlindError(ValueError):
    """Raised on config drift, sealed-chronology violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Path and file helpers
# ---------------------------------------------------------------------------


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise DozenBlindError("path is not portable")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise DozenBlindError("path escapes repository root") from error
    return path


def _file_sha256(path: Path) -> str:
    try:
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as error:
        raise DozenBlindError("bound file unavailable") from error
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DozenBlindError("campaign JSON unavailable") from error
    if not isinstance(value, dict):
        raise DozenBlindError("campaign JSON must be an object")
    return value


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction(value: Mapping[str, Any]) -> Fraction:
    if set(value) != {"numerator", "denominator"}:
        raise DozenBlindError("rational value schema changed")
    return Fraction(value["numerator"], value["denominator"])


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "campaign_id",
        "output_path",
        "policies",
        "schema_version",
        "sequence_alias",
        "target_fixture",
        "world_receipt_directory",
        "worlds",
    }:
        raise DozenBlindError("config keys changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["campaign_id"] != CAMPAIGN_ID
        or config["output_path"] != OUTPUT_PATH
        or config["world_receipt_directory"] != WORLD_RECEIPT_DIRECTORY
        or config["sequence_alias"] != SEQUENCE_ALIAS
    ):
        raise DozenBlindError("config identity changed")
    if config["policies"] != POLICIES:
        raise DozenBlindError("prospective policy changed")
    fixture = config["target_fixture"]
    if (
        not isinstance(fixture, Mapping)
        or set(fixture) != {"content_sha256", "path"}
        or fixture["path"] != TARGETS_PATH
        or not _hex_digest(fixture["content_sha256"])
    ):
        raise DozenBlindError("target fixture binding changed")
    worlds = config["worlds"]
    if not isinstance(worlds, list) or len(worlds) != WORLD_COUNT:
        raise DozenBlindError("world inventory changed")
    for expected_id, world in zip(WORLD_IDS, worlds, strict=True):
        if not isinstance(world, Mapping) or set(world) != {
            "declared_transformations",
            "public_rows",
            "sealed_target_sha256",
            "world_id",
        }:
            raise DozenBlindError("public world schema changed")
        if world["world_id"] != expected_id:
            raise DozenBlindError("anonymized world identity changed")
        if not _hex_digest(world["sealed_target_sha256"]):
            raise DozenBlindError("sealed target commitment malformed")
        rows = world["public_rows"]
        if not isinstance(rows, list) or len(rows) < 10 or len(rows) > 64:
            raise DozenBlindError("public row budget changed")
        previous: int | None = None
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"point", "value"}:
                raise DozenBlindError("public row schema changed")
            point, value = row["point"], row["value"]
            if (
                not isinstance(point, int)
                or isinstance(point, bool)
                or not isinstance(value, int)
                or isinstance(value, bool)
            ):
                raise DozenBlindError("public rows must be exact integers")
            if previous is not None and point != previous + 1:
                raise DozenBlindError("public points must be consecutive")
            previous = point
        transformations = world["declared_transformations"]
        if not isinstance(transformations, list):
            raise DozenBlindError("declared transformation inventory changed")
        for declaration in transformations:
            if not isinstance(declaration, Mapping) or set(declaration) != {
                "route",
                "statement",
                "transformation_id",
            }:
                raise DozenBlindError("declared transformation schema changed")
            if (
                declaration["transformation_id"] != RATIO_TRANSFORMATION_ID
                or declaration["route"] != "nonlinear_coefficient_search"
            ):
                raise DozenBlindError("declared transformation identity changed")


# ---------------------------------------------------------------------------
# Sealed-targets read guard (builtins.open / io.open / pathlib.Path.open)
# ---------------------------------------------------------------------------


class _SealedTargetsGuard:
    """Deny and audit every in-process read of the sealed targets fixture."""

    def __init__(self, root: Path) -> None:
        self._target = _resolve(root, TARGETS_PATH)
        self._attempts = 0
        self._surfaces: list[str] = []
        self._original_builtin_open = builtins.open
        self._original_io_open = io.open
        self._original_path_open = Path.open

    def _deny_if_sealed(self, file: Any, surface: str) -> None:
        try:
            resolved = Path(file).resolve()
        except TypeError:
            return
        if resolved == self._target:
            self._attempts += 1
            self._surfaces.append(surface)
            raise PermissionError("sealed targets fixture is unreadable before unseal")

    def __enter__(self) -> Self:
        def guarded_builtin_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            self._deny_if_sealed(file, "builtins.open")
            return self._original_builtin_open(file, *args, **kwargs)

        def guarded_io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            self._deny_if_sealed(file, "io.open")
            return self._original_io_open(file, *args, **kwargs)

        def guarded_path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
            self._deny_if_sealed(path, "pathlib.Path.open")
            return self._original_io_open(path, *args, **kwargs)

        builtins.open = guarded_builtin_open
        io.open = guarded_io_open
        Path.open = guarded_path_open  # type: ignore[method-assign]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        Path.open = self._original_path_open  # type: ignore[method-assign]
        io.open = self._original_io_open
        builtins.open = self._original_builtin_open

    def certificate(self) -> dict[str, Any]:
        return {
            "attempted_target_reads": self._attempts,
            "denied_content_bytes_exposed": 0,
            "denied_paths": [TARGETS_PATH] if self._attempts else [],
            "denied_surfaces": list(self._surfaces),
            "denied_target_reads": self._attempts,
            "enforcement_scope": (
                "owned_single_threaded_python_file_read_surfaces_"
                "not_an_operating_system_sandbox"
            ),
            "enforcement_surfaces": ["builtins.open", "io.open", "pathlib.Path.open"],
        }


def _warm_symbolic_runtime() -> None:
    """Trigger sympy's lazy solver imports before the sealed guard is entered."""

    n = sp.Symbol("n", integer=True)
    x, y = sp.symbols("x y")
    sp.solve([sp.Eq(x + y, 2), sp.Eq(x * y, 1)], [x, y], dict=True)
    sp.solve([sp.Eq((x * 1 + y) / (1 + 2), 1)], [x], dict=True)
    sp.simplify(sp.expand_func(sp.binomial(n, 2)) - n * (n - 1) / 2)
    sp.nsimplify(sp.Rational(1, 2), rational=True)
    sp.Poly(n**2 + n, n).all_coeffs()
    sp.expand((n + 1) ** 3 - n**3)


# ---------------------------------------------------------------------------
# Symbolic reconstruction of stage outputs
# ---------------------------------------------------------------------------


def _domain_symbol() -> sp.Symbol:
    return sp.Symbol("n", integer=True, nonnegative=True)


def _term_sympy(term: Term, n: sp.Symbol) -> sp.Expr:
    family, params = term.family, term.params
    if family == "monomial":
        return n ** params[0]
    if family == "alternating_monomial":
        return sp.Integer(-1) ** n * n ** params[0]
    if family == "geometric":
        return sp.Integer(params[0]) ** n
    if family == "geometric_monomial":
        return sp.Integer(params[0]) ** n * n ** params[1]
    if family == "reciprocal":
        return n ** (-params[0])
    if family == "shifted_reciprocal":
        return (n + 1) ** (-params[0])
    if family == "harmonic":
        return sp.harmonic(n)
    if family == "factorial":
        return sp.factorial(n)
    if family == "binomial":
        return sp.binomial(n, params[0])
    raise DozenBlindError(f"unsupported term family: {family}")


def _basis_sympy(family_id: str, coefficients: Sequence[Mapping[str, int]]) -> sp.Expr:
    entry = next((row for row in LADDER if row["family_id"] == family_id), None)
    if entry is None:
        raise DozenBlindError("candidate family left the declared ladder")
    n = _domain_symbol()
    total: sp.Expr = sp.Integer(0)
    for coefficient, term in zip(coefficients, entry["terms"], strict=True):
        rational = _fraction(coefficient)
        total += sp.Rational(rational.numerator, rational.denominator) * _term_sympy(term, n)
    return total


def _model_sympy(model_id: str, parameters: Mapping[str, Any]) -> sp.Expr:
    n = _domain_symbol()

    def value(name: str) -> sp.Expr:
        rational = _fraction(parameters[name])
        return sp.Rational(rational.numerator, rational.denominator)

    if model_id == "pure_geometric":
        return value("a") * value("b") ** n
    if model_id == "shifted_geometric":
        return value("a") * value("b") ** n + value("c")
    if model_id == "geometric_linear_argument":
        return value("a") * value("b") ** n * (n + value("c"))
    if model_id == "linear_fractional":
        return (value("a") * n + value("b")) / (n + value("d"))
    if model_id == "reciprocal_affine":
        return value("a") / (value("b") * n + value("c"))
    if model_id == "power_law":
        rational = _fraction(parameters["a"])
        return sp.Rational(rational.numerator, rational.denominator) * n ** parameters["p"]
    raise DozenBlindError(f"unsupported ratio model: {model_id}")


def _polynomial_coefficients(expression: sp.Expr) -> list[Fraction] | None:
    n = _domain_symbol()
    expanded = sp.expand(sp.expand_func(expression))
    polynomial = expanded.as_poly(n)
    if polynomial is None:
        return None
    coefficients: list[Fraction] = []
    for value in reversed(polynomial.all_coeffs()):
        if not isinstance(value, sp.Rational):
            return None
        coefficients.append(Fraction(int(value.p), int(value.q)))
    return coefficients


def _int_coefficients(expression: sp.Expr, n: sp.Symbol) -> list[int]:
    polynomial = sp.Poly(sp.expand(expression), n)
    values = [int(value) for value in reversed(polynomial.all_coeffs())]
    return values or [0]


def _nat_companion(coefficients: Sequence[Fraction]) -> dict[str, Any] | None:
    """Denominator-cleared, index-shifted companion with Nat-safe coefficients.

    ``B5``/``B6`` emit Lean over ``Nat`` and therefore refuse negative integers.  A
    rational polynomial closed form is carried into that grammar by the declared
    companion ``scale * a(n + shift)``: ``scale`` clears denominators and ``shift``
    is the smallest offset making every closed-form and forward-step coefficient
    nonnegative.  No companion within the declared shift budget means no Lean route,
    never an approximate one.
    """

    n = sp.Symbol("n", integer=True)
    scale = lcm(*(value.denominator for value in coefficients))
    polynomial: sp.Expr = sp.Integer(0)
    for degree, value in enumerate(coefficients):
        polynomial += sp.Rational(value.numerator, value.denominator) * n**degree
    for shift in range(MAX_COMPANION_SHIFT + 1):
        shifted = sp.expand(scale * polynomial.subs(n, n + shift))
        step = sp.expand(scale * (polynomial.subs(n, n + shift + 1) - polynomial.subs(n, n + shift)))
        shifted_coefficients = _int_coefficients(shifted, n)
        step_coefficients = _int_coefficients(step, n)
        base_value = shifted_coefficients[0]
        if (
            min(shifted_coefficients) >= 0
            and min(step_coefficients) >= 0
            and base_value >= 0
        ):
            return {
                "base_value": base_value,
                "closed_form_coefficients_ascending": shifted_coefficients,
                "scale": scale,
                "shift": shift,
                "step_coefficients_ascending": step_coefficients,
            }
    return None


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def _ratio_rows(rows: Sequence[Mapping[str, int]]) -> list[dict[str, Any]]:
    lookup = {row["point"]: Fraction(row["value"]) for row in rows}
    derived: list[dict[str, Any]] = []
    for point in sorted(lookup)[:-1]:
        denominator = lookup[point]
        if denominator == 0:
            raise DozenBlindError("ratio transformation undefined at a zero term")
        derived.append({"point": point, "value": _fraction_data(lookup[point + 1] / denominator)})
    return derived


def _minimal_recurrence(
    values: Sequence[Fraction],
) -> tuple[int, list[Fraction]] | None:
    """Smallest-order affine linear recurrence exact on every public row."""

    for order in range(1, MAX_RECURRENCE_ORDER + 1):
        width = order + 1
        matrix = [
            [values[index - step] for step in range(1, order + 1)] + [Fraction(1)]
            for index in range(order, len(values))
        ]
        vector = [values[index] for index in range(order, len(values))]
        if len(matrix) < width:
            continue
        solution = _solve_unique(
            [list(row) for row in matrix[:width]], list(vector[:width]), width
        )
        if solution is None:
            continue
        exact = all(
            sum(
                (solution[step - 1] * values[index - step] for step in range(1, order + 1)),
                Fraction(0),
            )
            + solution[order]
            == values[index]
            for index in range(order, len(values))
        )
        if exact:
            return order, solution
    return None


def _recurrence_statement(order: int, solution: Sequence[Fraction]) -> str:
    terms = " + ".join(f"({solution[step - 1]})*a(n-{step})" for step in range(1, order + 1))
    constant = solution[order]
    return f"a(n) = {terms}" + (f" + ({constant})" if constant != 0 else "")


def _survivors(b3: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["kind"]: row
        for row in b3["conjectures"]
        if row.get("status") == "SURVIVED"
    }


def _select_candidate(
    world: Mapping[str, Any],
    b1: Mapping[str, Any],
    b2: Mapping[str, Any] | None,
    b3: Mapping[str, Any],
    b7: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if b1["decision"] == "PASS":
        result = b1["result"]
        expression = _basis_sympy(result["family_id"], result["coefficients"])
        return {
            "coefficients": result["coefficients"],
            "confirmations": result["confirmations"],
            "expression": result["expression"],
            "family_id": result["family_id"],
            "kind": "closed_form",
            "source_stage": "b1_basis_synthesis",
            "statement": f"a(n) = {result['expression']}",
            "sympy_expression": str(expression),
            "terms": result["terms"],
        }
    if b2 is not None and b2["decision"] == "PASS":
        result = b2["result"]
        expression = _model_sympy(result["model_id"], result["parameters"])
        return {
            "confirmations": result["confirmations"],
            "expression": result["expression"],
            "kind": "ratio_law",
            "model_id": result["model_id"],
            "parameters": result["parameters"],
            "source_stage": "b2_nonlinear_coefficient_search",
            "statement": f"a(n+1)/a(n) = {result['expression']}",
            "sympy_expression": str(expression),
        }
    recurrence_survivor = _survivors(b3).get("linear_recurrence")
    if recurrence_survivor is not None:
        values = [Fraction(row["value"]) for row in world["public_rows"]]
        derived = _minimal_recurrence(values)
        if derived is None:
            raise DozenBlindError("recurrence survivor lost exact full-row normalization")
        order, solution = derived
        statement = _recurrence_statement(order, solution)
        if statement != recurrence_survivor["statement"]:
            raise DozenBlindError("recurrence normalization diverged from the B3 survivor")
        return {
            "coefficients": [_fraction_data(solution[step]) for step in range(order)],
            "confirmations": recurrence_survivor["support"],
            "inhomogeneous_constant": _fraction_data(solution[order]),
            "kind": "linear_recurrence",
            "normalization": (
                "coefficients re-derived at minimal order from the full public rows and "
                "required to render identically to the B3 prefix-discovered survivor"
            ),
            "order": order,
            "seeds": [row["value"] for row in world["public_rows"][:order]],
            "source_stage": "b3_conjecture_generation",
            "statement": statement,
        }
    if b7 is not None and b7["decision"] == "PASS":
        repair = b7["repair"]
        return {
            "confirmations": repair.get("confirmations"),
            "expression": repair.get("expression"),
            "kind": "repaired_statement",
            "repair": repair,
            "source_stage": "b7_structural_repair",
            "statement": f"a(n) = {repair.get('expression')} [{repair.get('strategy')}]",
        }
    return None


# ---------------------------------------------------------------------------
# Prover routing
# ---------------------------------------------------------------------------


def _lean_namespace(world_id: str) -> str:
    return "SolvedDozen" + world_id.title().replace("_", "")


def _route_provers(
    world_id: str,
    candidate: Mapping[str, Any] | None,
    b3: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if candidate is None:
        return [
            {
                "reason": "no candidate survived Phase A",
                "route": "none",
                "status": "NOT_APPLICABLE",
            }
        ]
    if candidate["kind"] != "closed_form":
        return [
            {
                "reason": (
                    f"candidate kind {candidate['kind']} has no declared prover route; "
                    "B5/B6 cover polynomial closed forms only"
                ),
                "route": "none",
                "status": "NOT_APPLICABLE",
            }
        ]
    expression = _basis_sympy(candidate["family_id"], candidate["coefficients"])
    coefficients = _polynomial_coefficients(expression)
    if coefficients is None:
        return [
            {
                "reason": (
                    "closed form is not polynomial over the rationals; the declared "
                    "B5/B6 grammar excludes exponential and special-function terms"
                ),
                "route": "none",
                "status": "NOT_APPLICABLE",
            }
        ]
    companion = _nat_companion(coefficients)
    if companion is None:
        return [
            {
                "reason": "no nonnegative-integer companion within the declared shift budget",
                "route": "b5_lemma_decomposition",
                "status": "NOT_APPLICABLE",
            }
        ]
    namespace = _lean_namespace(world_id)
    routes: list[dict[str, Any]] = []
    b5 = decompose_closed_form_proof(
        {
            "base_value": companion["base_value"],
            "closed_form": companion["closed_form_coefficients_ascending"],
            "namespace": namespace,
            "sequence_name": "companion",
            "step": companion["step_coefficients_ascending"],
        }
    )
    routes.append(
        {
            "companion": companion,
            "companion_statement": (
                f"{companion['scale']}*a(n+{companion['shift']}) satisfies "
                "companion(k+1) = companion(k) + step(k) with the emitted closed form"
            ),
            "decision": b5["decision"],
            "lean_source_emitted": b5["lean_source"] is not None,
            "receipt": b5,
            "receipt_sha256": b5["content_sha256"],
            "route": "b5_lemma_decomposition",
            "status": "ROUTED",
        }
    )
    survivors = _survivors(b3)
    monotone = survivors.get("monotonicity")
    if monotone is not None and monotone["statement"] == "a(n) < a(n+1)":
        b6 = prove_quantified_inequality(
            {
                "coefficients": companion["closed_form_coefficients_ascending"],
                "name": "companionStrictlyIncreasing",
                "namespace": namespace,
                "relation": "monotone_increasing",
            }
        )
        routes.append(
            {
                "companion": companion,
                "decision": b6["decision"],
                "lean_source_emitted": b6["lean_source"] is not None,
                "receipt": b6,
                "receipt_sha256": b6["content_sha256"],
                "relation": "monotone_increasing",
                "route": "b6_quantified_inequality",
                "status": "ROUTED",
                "survivor_statement": monotone["statement"],
            }
        )
    sign = survivors.get("sign")
    if sign is not None and sign["statement"] == "a(n) > 0":
        b6 = prove_quantified_inequality(
            {
                "coefficients": companion["closed_form_coefficients_ascending"],
                "name": "companionNonnegative",
                "namespace": namespace,
                "relation": "nonnegative",
            }
        )
        routes.append(
            {
                "companion": companion,
                "decision": b6["decision"],
                "lean_source_emitted": b6["lean_source"] is not None,
                "note": (
                    "the Nat-typed emission proves nonnegativity; strict positivity of the "
                    "original sequence is not claimed by this route"
                ),
                "receipt": b6,
                "receipt_sha256": b6["content_sha256"],
                "relation": "nonnegative",
                "route": "b6_quantified_inequality",
                "status": "ROUTED",
                "survivor_statement": sign["statement"],
            }
        )
    return routes


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def _stage(stage_id: str, tool: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": receipt["decision"],
        "receipt": dict(receipt),
        "receipt_sha256": receipt["content_sha256"],
        "stage_id": stage_id,
        "tool": tool,
    }


def _run_world_phase_a(world: Mapping[str, Any]) -> dict[str, Any]:
    rows = world["public_rows"]
    stages: list[dict[str, Any]] = []
    b1 = synthesize_basis(rows)
    stages.append(_stage("b1_basis_synthesis", "synthesize_basis", b1))
    b7: dict[str, Any] | None = None
    if b1["decision"] != "PASS":
        b7 = repair_structure(rows)
        stages.append(_stage("b7_structural_repair", "repair_structure", b7))
    # This campaign is sealed: it was run, unsealed, and shipped under the v1
    # statement-kind profile, so its deterministic replay pins that profile.
    b3 = generate_conjectures(rows, statement_kinds=STATEMENT_KINDS_V1)
    stages.append(_stage("b3_conjecture_generation", "generate_conjectures", b3))
    transformation_records: list[dict[str, Any]] = []
    b2: dict[str, Any] | None = None
    for declaration in world["declared_transformations"]:
        derived_rows = _ratio_rows(rows)
        transformation_records.append(
            {
                "declaration": dict(declaration),
                "input_rows_sha256": canonical_sha256(rows),
                "output_rows": derived_rows,
                "output_rows_sha256": canonical_sha256(derived_rows),
                "rows_consumed": len(rows),
                "rows_produced": len(derived_rows),
                "transformation_id": declaration["transformation_id"],
            }
        )
        b2 = search_nonlinear(derived_rows)
        stages.append(_stage("b2_nonlinear_coefficient_search", "search_nonlinear", b2))
    candidate = _select_candidate(world, b1, b2, b3, b7)
    prover_routes = _route_provers(world["world_id"], candidate, b3)
    lean_emitted = any(route.get("lean_source_emitted") is True for route in prover_routes)
    return {
        "candidate": candidate,
        "candidate_selection_rule": CANDIDATE_SELECTION_RULE,
        "candidate_statement": None if candidate is None else candidate["statement"],
        "closed_form_claimed": candidate is not None and candidate["kind"] == "closed_form",
        "lean_emitted": lean_emitted,
        "prover_routes": prover_routes,
        "public_rows_sha256": canonical_sha256(rows),
        "sealed_target_sha256": world["sealed_target_sha256"],
        "stages": stages,
        "transformation_records": transformation_records,
        "world_id": world["world_id"],
    }


# ---------------------------------------------------------------------------
# Atomic unseal
# ---------------------------------------------------------------------------


def _unseal_targets(
    root: Path, config: Mapping[str, Any], phase_a_root: str
) -> tuple[dict[str, dict[str, Any]], str]:
    """Open the sealed fixture once, after Phase A is sealed, and verify every commitment."""

    if not _hex_digest(phase_a_root):
        raise DozenBlindError("target unseal attempted before candidate freeze")
    target_path = _resolve(root, config["target_fixture"]["path"])
    try:
        raw = target_path.read_bytes()
        fixture = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DozenBlindError("target fixture unavailable") from error
    if (
        not isinstance(fixture, dict)
        or canonical_sha256(fixture) != config["target_fixture"]["content_sha256"]
    ):
        raise DozenBlindError("target fixture content changed")
    if set(fixture) != {"schema_version", "targets"} or fixture["schema_version"] != TARGET_SCHEMA:
        raise DozenBlindError("target fixture schema changed")
    targets = fixture["targets"]
    if not isinstance(targets, list) or len(targets) != len(config["worlds"]):
        raise DozenBlindError("atomic target batch incomplete")
    configured = {world["world_id"]: world for world in config["worlds"]}
    by_world: dict[str, dict[str, Any]] = {}
    classical_ids: set[str] = set()
    for target in targets:
        if not isinstance(target, dict) or set(target) != {
            "attribution",
            "classical_id",
            "domain",
            "expression",
            "expression_ast",
            "kind",
            "parameters",
            "salt",
            "world_id",
        }:
            raise DozenBlindError("target record schema changed")
        world_id = target["world_id"]
        if world_id not in configured or world_id in by_world:
            raise DozenBlindError("target world identity changed")
        if canonical_sha256(target) != configured[world_id]["sealed_target_sha256"]:
            raise DozenBlindError("target commitment did not open")
        classical = target["classical_id"]
        if (
            not isinstance(classical, str)
            or not classical.isidentifier()
            or classical in classical_ids
        ):
            raise DozenBlindError("classical identifier malformed")
        classical_ids.add(classical)
        if target["kind"] not in {"closed_form", "linear_recurrence", "ratio_law"}:
            raise DozenBlindError("target kind changed")
        by_world[world_id] = target
    if set(by_world) != set(configured):
        raise DozenBlindError("atomic target world coverage changed")
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return by_world, hashlib.sha256(normalized).hexdigest()


# ---------------------------------------------------------------------------
# Exact comparison
# ---------------------------------------------------------------------------


def _parse_target_expression(text: str) -> sp.Expr:
    n = _domain_symbol()
    try:
        return sp.sympify(text, locals={"binomial": sp.binomial, "n": n})
    except (sp.SympifyError, TypeError, SyntaxError) as error:
        raise DozenBlindError("target expression unparseable") from error


def _equivalent(lhs: sp.Expr, rhs: sp.Expr) -> tuple[bool, str]:
    difference = sp.expand(sp.expand_func(lhs - rhs))
    if difference == 0:
        return True, "0"
    simplified = sp.simplify(difference)
    if simplified == 0:
        return True, "0"
    collapsed = sp.cancel(sp.together(difference))
    return collapsed == 0, str(simplified)


def _compare_candidate(
    candidate: Mapping[str, Any] | None, target: Mapping[str, Any]
) -> dict[str, Any]:
    if candidate is None:
        return {
            "base_verdict": "MISSED",
            "detail": "no candidate survived Phase A",
            "equivalent": False,
            "method": "no_candidate",
        }
    kind = target["kind"]
    if kind == "closed_form":
        if candidate["kind"] != "closed_form":
            return {
                "base_verdict": "MISSED",
                "detail": f"candidate kind {candidate['kind']} is not a closed form",
                "equivalent": False,
                "method": "kind_mismatch",
            }
        discovered = sp.sympify(
            candidate["sympy_expression"], locals={"binomial": sp.binomial, "n": _domain_symbol()}
        )
        expected = _parse_target_expression(target["expression"])
        equivalent, residual = _equivalent(discovered, expected)
        family_match = candidate["family_id"] == target["parameters"]["family"]
        return {
            "base_verdict": (
                "REDISCOVERED" if equivalent else ("PARTIAL" if family_match else "MISSED")
            ),
            "detail": {
                "discovered_sympy": str(discovered),
                "family_match": family_match,
                "residual": residual,
                "target_sympy": str(expected),
            },
            "equivalent": equivalent,
            "method": "sympy_exact_equivalence_over_declared_domain",
        }
    if kind == "linear_recurrence":
        if candidate["kind"] != "linear_recurrence":
            return {
                "base_verdict": "MISSED",
                "detail": f"candidate kind {candidate['kind']} is not a recurrence",
                "equivalent": False,
                "method": "kind_mismatch",
            }
        parameters = target["parameters"]
        order_match = candidate["order"] == parameters["order"]
        coefficients_match = order_match and [
            _fraction(value) for value in candidate["coefficients"]
        ] == [Fraction(value) for value in parameters["coefficients"]]
        constant_match = (
            _fraction(candidate["inhomogeneous_constant"])
            == Fraction(parameters["inhomogeneous_constant"])
        )
        seeds_match = candidate["seeds"] == parameters["seeds"]
        exact = order_match and coefficients_match and constant_match and seeds_match
        return {
            "base_verdict": (
                "REDISCOVERED" if exact else ("PARTIAL" if order_match else "MISSED")
            ),
            "detail": {
                "coefficients_match": coefficients_match,
                "constant_match": constant_match,
                "order_match": order_match,
                "seeds_match": seeds_match,
            },
            "equivalent": exact,
            "method": "exact_recurrence_coefficient_equality",
        }
    if kind == "ratio_law":
        if candidate["kind"] != "ratio_law":
            return {
                "base_verdict": "MISSED",
                "detail": f"candidate kind {candidate['kind']} is not a ratio law",
                "equivalent": False,
                "method": "kind_mismatch",
            }
        parameters = target["parameters"]
        model_match = candidate["model_id"] == parameters["model"]
        names = sorted(set(parameters) - {"model"})
        parameters_match = model_match and all(
            _fraction(candidate["parameters"][name]) == Fraction(parameters[name])
            for name in names
        )
        discovered = sp.sympify(
            candidate["sympy_expression"], locals={"binomial": sp.binomial, "n": _domain_symbol()}
        )
        expected = _parse_target_expression(target["expression"])
        equivalent, residual = _equivalent(discovered, expected)
        exact = parameters_match and equivalent
        return {
            "base_verdict": (
                "REDISCOVERED" if exact else ("PARTIAL" if model_match else "MISSED")
            ),
            "detail": {
                "discovered_sympy": str(discovered),
                "model_match": model_match,
                "parameters_match": parameters_match,
                "residual": residual,
                "target_sympy": str(expected),
            },
            "equivalent": exact,
            "method": "exact_ratio_parameter_equality_and_sympy_equivalence",
        }
    raise DozenBlindError("target kind changed")


def _target_statement(target: Mapping[str, Any]) -> str:
    if target["kind"] == "closed_form":
        return f"a(n) = {target['expression']}"
    if target["kind"] == "linear_recurrence":
        seeds = target["parameters"]["seeds"]
        seed_text = ", ".join(f"a({index}) = {value}" for index, value in enumerate(seeds))
        return f"a(n) = {target['expression']}; {seed_text}"
    return f"a(n+1)/a(n) = {target['expression']}"


def _score_world(
    phase_a_world: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    comparison = _compare_candidate(phase_a_world["candidate"], target)
    verdict = comparison["base_verdict"]
    routed = [
        route
        for route in phase_a_world["prover_routes"]
        if route.get("status") == "ROUTED"
    ]
    provers_green = bool(routed) and all(
        route["decision"] in {"DECOMPOSED", "PROVED_LOCALLY"} for route in routed
    )
    if verdict == "REDISCOVERED":
        verdict = (
            "REDISCOVERED_WITH_PROOF"
            if phase_a_world["lean_emitted"] and provers_green
            else "REDISCOVERED_EXACT"
        )
    note = None
    if (
        target["kind"] == "linear_recurrence"
        and verdict.startswith("REDISCOVERED")
        and not phase_a_world["closed_form_claimed"]
    ):
        note = NON_CLAIM_NOTE
    return {
        "attribution": target["attribution"],
        "classical_id": target["classical_id"],
        "commitment_opened": True,
        "comparison": comparison,
        "note": note,
        "provers_green": provers_green,
        "target_record": dict(target),
        "target_statement": _target_statement(target),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Campaign assembly
# ---------------------------------------------------------------------------


def build_artifacts(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Run the sealed campaign; returns the campaign receipt and per-world receipts."""

    root = root.resolve()
    config = _load_json(config_path or _resolve(root, CONFIG_PATH))
    _validate_config(config)
    _warm_symbolic_runtime()
    guard = _SealedTargetsGuard(root)
    with guard:
        phase_worlds = [_run_world_phase_a(world) for world in config["worlds"]]
        try:
            _resolve(root, TARGETS_PATH).read_bytes()
        except PermissionError:
            pass
        else:
            raise DozenBlindError("pre-unseal target read was not denied")
    denied_probe = guard.certificate()
    if (
        denied_probe["attempted_target_reads"] != 1
        or denied_probe["denied_target_reads"] != 1
        or denied_probe["denied_content_bytes_exposed"] != 0
        or denied_probe["denied_paths"] != [TARGETS_PATH]
    ):
        raise DozenBlindError("sealed-target enforcement boundary changed")
    phase_a_root = canonical_sha256(
        {
            "commitments": [world["sealed_target_sha256"] for world in config["worlds"]],
            "denied_probe": denied_probe,
            "worlds": phase_worlds,
        }
    )
    targets, target_file_sha256 = _unseal_targets(root, config, phase_a_root)

    world_receipts: dict[str, dict[str, Any]] = {}
    world_rows: list[dict[str, Any]] = []
    for world, phase_a_world in zip(config["worlds"], phase_worlds, strict=True):
        target = targets[world["world_id"]]
        unseal = _score_world(phase_a_world, target)
        receipt_body = {
            "campaign_id": CAMPAIGN_ID,
            "classical_id": target["classical_id"],
            "declared_transformations": world["declared_transformations"],
            "phase_a": phase_a_world,
            "phase_a_root": phase_a_root,
            "public_rows": world["public_rows"],
            "schema_version": WORLD_SCHEMA,
            "sealed_target_sha256": world["sealed_target_sha256"],
            "unseal": unseal,
            "world_id": world["world_id"],
        }
        receipt = {**receipt_body, "content_sha256": canonical_sha256(receipt_body)}
        world_receipts[target["classical_id"]] = receipt
        candidate = phase_a_world["candidate"]
        world_rows.append(
            {
                "attribution": target["attribution"],
                "classical_id": target["classical_id"],
                "discovered_statement": phase_a_world["candidate_statement"],
                "holdout_confirmations": (
                    None if candidate is None else candidate["confirmations"]
                ),
                "lean_emitted": phase_a_world["lean_emitted"],
                "note": unseal["note"],
                "prover_trail": [
                    {
                        "decision": route.get("decision"),
                        "lean_source_emitted": route.get("lean_source_emitted", False),
                        "receipt_sha256": route.get("receipt_sha256"),
                        "route": route["route"],
                        "status": route.get("status"),
                    }
                    for route in phase_a_world["prover_routes"]
                ],
                "stage_trail": [
                    {
                        "decision": stage["decision"],
                        "receipt_sha256": stage["receipt_sha256"],
                        "stage_id": stage["stage_id"],
                        "tool": stage["tool"],
                    }
                    for stage in phase_a_world["stages"]
                ],
                "target_statement": unseal["target_statement"],
                "verdict": unseal["verdict"],
                "world_id": world["world_id"],
                "world_receipt_path": f"{WORLD_RECEIPT_DIRECTORY}/{target['classical_id']}.json",
                "world_receipt_sha256": receipt["content_sha256"],
            }
        )

    verdicts = [row["verdict"] for row in world_rows]
    rediscovered = sum(1 for verdict in verdicts if verdict.startswith("REDISCOVERED"))
    counts = {
        "holdout_confirmations_total": sum(
            row["holdout_confirmations"] or 0 for row in world_rows
        ),
        "lean_sources_emitted": sum(
            1
            for phase_a_world in phase_worlds
            for route in phase_a_world["prover_routes"]
            if route.get("lean_source_emitted") is True
        ),
        "missed": sum(1 for verdict in verdicts if verdict == "MISSED"),
        "partial": sum(1 for verdict in verdicts if verdict == "PARTIAL"),
        "post_unseal_generation_events": 0,
        "prover_receipts": sum(
            1
            for phase_a_world in phase_worlds
            for route in phase_a_world["prover_routes"]
            if route.get("status") == "ROUTED"
        ),
        "rediscovered_exact": sum(
            1 for verdict in verdicts if verdict == "REDISCOVERED_EXACT"
        ),
        "rediscovered_total": rediscovered,
        "rediscovered_with_proof": sum(
            1 for verdict in verdicts if verdict == "REDISCOVERED_WITH_PROOF"
        ),
        "stage_receipts": sum(len(phase_a_world["stages"]) for phase_a_world in phase_worlds),
        "target_fixture_reads": 1,
        "target_fixture_reads_denied_before_unseal": 1,
        "worlds": len(world_rows),
    }
    decision = (
        "PASS" if rediscovered >= config["policies"]["minimum_rediscoveries_for_pass"] else "BLOCK"
    )
    chronology = {
        "denied_probe": denied_probe,
        "events": [
            {"event": "config_and_public_rows_loaded", "sequence": 0, "target_reads": 0},
            {"event": "sealed_targets_guard_entered", "sequence": 1, "target_reads": 0},
            {
                "event": "stage_ladder_and_candidates_frozen_all_worlds",
                "sequence": 2,
                "target_reads": 0,
            },
            {"event": "instrumented_denied_probe_recorded", "sequence": 3, "target_reads": 0},
            {
                "event": "phase_a_root_sealed",
                "root_sha256": phase_a_root,
                "sequence": 4,
                "target_reads": 0,
            },
            {"event": "atomic_target_unseal", "sequence": 5, "target_reads": 1},
            {
                "event": "commitments_opened_and_exact_comparison",
                "sequence": 6,
                "target_reads": 1,
            },
        ],
        "phase_a_root": phase_a_root,
        "unseal_batches": 1,
    }
    body = {
        "campaign_id": CAMPAIGN_ID,
        "chronology": chronology,
        "claims": CLAIMS,
        "counts": counts,
        "decision": decision,
        "first_blocker": (
            None if decision == "PASS" else "minimum_rediscovery_threshold_not_met"
        ),
        "policies": dict(config["policies"]),
        "schema_version": RESULT_SCHEMA,
        "scope": SCOPE,
        "sequence_alias": SEQUENCE_ALIAS,
        "source_bindings": {
            "config": {
                "file_sha256": _file_sha256(_resolve(root, CONFIG_PATH)),
                "path": CONFIG_PATH,
            },
            "source": {
                "file_sha256": _file_sha256(_resolve(root, SOURCE_PATH)),
                "path": SOURCE_PATH,
            },
            "target_fixture": {"file_sha256": target_file_sha256, "path": TARGETS_PATH},
            "test": {
                "file_sha256": _file_sha256(_resolve(root, TEST_PATH)),
                "path": TEST_PATH,
            },
        },
        "world_results": world_rows,
    }
    campaign = {**body, "content_sha256": canonical_sha256(body)}
    return {"campaign": campaign, "worlds": world_receipts}


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Build only the campaign receipt (convenience wrapper over build_artifacts)."""

    return build_artifacts(root, config_path)["campaign"]


def validate_artifacts(
    campaign: Mapping[str, Any],
    worlds: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
    config_path: Path | None = None,
) -> None:
    """Reject any tamper or environmental drift by exact deterministic replay."""

    if campaign.get("schema_version") != RESULT_SCHEMA:
        raise DozenBlindError("campaign receipt schema changed")
    campaign_body = {key: item for key, item in campaign.items() if key != "content_sha256"}
    if campaign.get("content_sha256") != canonical_sha256(campaign_body):
        raise DozenBlindError("campaign receipt seal changed")
    for world in worlds.values():
        world_body = {key: item for key, item in world.items() if key != "content_sha256"}
        if world.get("content_sha256") != canonical_sha256(world_body):
            raise DozenBlindError("world receipt seal changed")
    replayed = build_artifacts(root, config_path)
    if dict(campaign) != replayed["campaign"]:
        raise DozenBlindError("campaign receipt exact replay changed")
    if {key: dict(value) for key, value in worlds.items()} != replayed["worlds"]:
        raise DozenBlindError("world receipt exact replay changed")


def validate_campaign(
    value: Mapping[str, Any], *, root: Path, config_path: Path | None = None
) -> None:
    """Validate the campaign receipt alone by exact deterministic replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise DozenBlindError("campaign receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise DozenBlindError("campaign receipt seal changed")
    if dict(value) != build_artifacts(root, config_path)["campaign"]:
        raise DozenBlindError("campaign receipt exact replay changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise DozenBlindError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = _resolve(root, args.output)
    if args.validate_checked:
        campaign = _load_json(output)
        worlds = {
            row["classical_id"]: _load_json(_resolve(root, row["world_receipt_path"]))
            for row in campaign.get("world_results", [])
        }
        validate_artifacts(campaign, worlds, root=root)
        return 0
    artifacts = build_artifacts(root)
    _write_immutable(output, artifacts["campaign"])
    for classical_id, receipt in artifacts["worlds"].items():
        _write_immutable(
            _resolve(root, f"{WORLD_RECEIPT_DIRECTORY}/{classical_id}.json"), receipt
        )
    validate_artifacts(artifacts["campaign"], artifacts["worlds"], root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
