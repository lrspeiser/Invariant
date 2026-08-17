"""Blinded rediscovery of three classical laws from anonymized column data.

Four worlds are handed to the discovery engine as exact rational rows under neutral
column names ``x1 .. xk`` and opaque world ids (``world_01 .. world_04``).  The public
config carries no name, no unit, and no word from the subject matter: a runtime guard
tokenizes the config text and refuses to run if a single term from the declared
forbidden vocabulary appears.  The closed forms, the meaning of each column, and the
attributions live in a separate sealed fixture that Phase A is forbidden to read; the
config commits to it by salted SHA-256 only.

Phase A runs a declared ladder per world.  B1 basis synthesis and B7 structural repair
are tried on the raw ``(index, response)`` rows, B3 proposes typed statements, and then
the campaign's own declared transformation lane runs: a **bounded derived-view search**.
A view is one member of a finite, ordered grammar of column combinations --
``x_response^v * x1^(-u)`` for the two-column worlds, and
``x_response^v * x1^i * (1 - x2^2)^j * x2^k`` for the three-column world.  B1 and B2 are
run on *every* view.  A view is admitted only when B1 returns its ``constant`` family,
because a constant derived column is the one statement in this grammar that does not
depend on the row index, and the config declares the index to be an arbitrary label.
The first admitted view in Occam order is frozen as the candidate law.

Only then is the target fixture opened -- once, atomically, after the Phase A root is
sealed.  Every commitment must open; the sealed generative rule must regenerate the
public rows exactly; and each frozen candidate is compared against its classical target
by sympy equivalence plus exact rational equality of the exponent structure and the
constant.  Verdicts are REDISCOVERED_EXACT, PARTIAL, or MISSED.  Nothing is generated,
retried, or tuned after the unseal.

Claim boundary.  The rows are computed from a declared generative rule anchored to
published values; they are not measurements, and no observational dataset is opened at
run time.  Rediscovery of solved physics is the entire claim: no novelty, priority, or
empirical significance is asserted.  Emitted Lean proves the Nat-typed monomial
companion of the recovered exponent structure -- not the physical law -- and is
exact-locally checked here, never kernel-verified in this receipt.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import comb, gcd
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import sympy as sp

from .basis_synthesis import synthesize_basis
from .conjecture_generation import STATEMENT_KINDS, generate_conjectures
from .lemma_decomposition import decompose_closed_form_proof
from .nonlinear_coefficient_search import search_nonlinear
from .quantified_inequality_proofs import prove_quantified_inequality
from .sigma_core import canonical_json_bytes, canonical_sha256
from .structural_repair import repair_structure

CONFIG_SCHEMA = "invariant-blind-column-world-config-1.0"
TARGET_SCHEMA = "invariant-blind-planetary-targets-1.0"
RESULT_SCHEMA = "invariant-blind-planetary-law-rediscovery-result-1.0"
WORLD_SCHEMA = "invariant-blind-planetary-world-receipt-1.0"
CAMPAIGN_ID = "blind-planetary-laws-001"
CONFIG_PATH = "configs/backgrounds/blind_planetary_public_worlds_v1.json"
TARGETS_PATH = "configs/backgrounds/blind_planetary_targets_v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/blind_planetary_law_rediscovery_campaign.py"
TEST_PATH = "tests/test_blind_planetary_law_rediscovery_campaign.py"
DOC_PATH = "docs/BLIND_PLANETARY_LAW_REDISCOVERY.md"
OUTPUT_PATH = "runs/math/blind-planetary-laws/campaign.json"
WORLD_RECEIPT_DIRECTORY = "runs/math/blind-planetary-laws"

WORLD_IDS = ("world_01", "world_02", "world_03", "world_04")
RESPONSE_ALIAS = "x_response"
ROOT_QUANTIZATION_DECIMAL_PLACES = 12
DECLARED_CONSTANT_DECIMAL_PLACES = 50
PI_60DP = "3.141592653589793238462643383279502884197169399375105820974945"
ADVANCE_COEFFICIENT_NUMERIC_FACTOR = 3888000

#: Bounds of the declared derived-view grammar.  These numbers are the search space:
#: changing one changes what the engine could possibly have found, so a test pins them.
PAIR_PREDICTOR_EXPONENT_BOUND = 6
PAIR_RESPONSE_EXPONENT_BOUND = 3
TRIPLE_PREDICTOR_EXPONENT_BOUND = 2
TRIPLE_RESPONSE_EXPONENT_BOUND = 2

MAX_COMPANION_DEGREE = 6
ADMITTED_FAMILY_ID = "constant"

CANDIDATE_SELECTION_RULE = (
    "the first derived view admitted by B1's constant family, in declared Occam order; "
    "every index-dependent statement -- a B1 raw closed form, a B2 raw model, a B7 "
    "repair, or a non-constant basis fitted to a view -- is recorded in the stage trail "
    "and never selected, because the config declares the row index an arbitrary label"
)
COMPANION_DEGREE_RULE = (
    "sum of the absolute integer exponents on the predictor side, counting the "
    "(1 - x2^2) base as degree two"
)
LEAN_BOUNDARY_NOTE = (
    "the emitted Lean proves the Nat-typed monomial companion n^d of the recovered "
    "exponent structure by induction; it does not prove the recovered relation, and it "
    "is exact-locally checked here, never kernel-verified in this receipt"
)

POLICIES = {
    "atomic_target_unseal_batches": 1,
    "candidate_generation_after_unseal": 0,
    "minimum_rediscoveries_for_pass": 4,
    "target_reads_before_candidate_freeze": 0,
}

STATIC_CLAIMS = {
    "atomic_target_unseal_batches": 1,
    "data_computed_from_declared_model": True,
    "kernel_verified_lean": False,
    "lean_proves_the_recovered_relation_itself": False,
    "novelty_claimed": False,
    "post_unseal_generation": False,
    "published_anchor_values_used_in_sealed_construction": True,
    "real_observational_data_opened": False,
    "rediscovery_of_classical_results": True,
    "target_records_read_before_candidate_freeze": 0,
}

SCOPE = (
    "Blinded rediscovery of three classically solved physical laws from anonymized exact "
    "rational rows only. Stage inputs are neutral columns x1..xk and opaque world ids; the "
    "closed forms, column meanings, provenance, and attributions stay in a sealed fixture "
    "whose in-process reads are denied until the single post-freeze atomic unseal, evidenced "
    "by one instrumented denied probe. The derived-view grammar is declared, finite, and "
    "logged view by view, so a MISSED means 'outside the declared grammar', never "
    "'impossible'. Rows are computed from a declared generative rule anchored to published "
    "values and sealed with the targets; they are not measurements and no observational "
    "dataset is opened at run time, so this receipt is evidence about the engine, not about "
    "the world. Emitted Lean covers the Nat-typed monomial companion of the recovered "
    "exponent structure only and is not kernel-verified here. Rediscovery of solved physics "
    "is the entire claim: no novelty, priority, or empirical significance is asserted."
)

#: Tokens that must never appear in the public config.  The list is deliberately wider
#: than strictly necessary (names, units, and subject-matter words alike), and the check
#: runs at build time, not only in the test suite.
FORBIDDEN_VOCABULARY = (
    "acceleration",
    "anomalous",
    "aphelion",
    "asteroid",
    "astronomical",
    "astronomy",
    "au",
    "axis",
    "bodies",
    "body",
    "celestial",
    "centuries",
    "century",
    "ceres",
    "comet",
    "cube",
    "day",
    "days",
    "degree",
    "degrees",
    "dynamics",
    "earth",
    "einstein",
    "eccentricity",
    "ephemeris",
    "force",
    "gravitation",
    "gravitational",
    "gravity",
    "harmonic",
    "heliocentric",
    "iau",
    "inverse",
    "jpl",
    "jupiter",
    "kepler",
    "keplerian",
    "kilometer",
    "kilometre",
    "km",
    "law",
    "laws",
    "lunar",
    "major",
    "mars",
    "mass",
    "masses",
    "mercury",
    "meter",
    "metre",
    "moon",
    "motion",
    "nasa",
    "neptune",
    "newton",
    "newtonian",
    "nssdc",
    "orbit",
    "orbital",
    "orbits",
    "perihelion",
    "period",
    "periods",
    "physical",
    "physics",
    "planet",
    "planetary",
    "planets",
    "pluto",
    "precession",
    "radian",
    "radians",
    "relativistic",
    "relativity",
    "revolution",
    "saturn",
    "second",
    "seconds",
    "semi",
    "solar",
    "square",
    "star",
    "stellar",
    "sun",
    "unit",
    "units",
    "uranus",
    "velocity",
    "venus",
    "year",
    "years",
    "yr",
)


class BlindPlanetaryLawError(ValueError):
    """Raised on config drift, sealed-chronology violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Path and file helpers
# ---------------------------------------------------------------------------


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BlindPlanetaryLawError("path is not portable")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise BlindPlanetaryLawError("path escapes repository root") from error
    return path


def _file_sha256(path: Path) -> str:
    try:
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as error:
        raise BlindPlanetaryLawError("bound file unavailable") from error
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BlindPlanetaryLawError("campaign JSON unavailable") from error
    if not isinstance(value, dict):
        raise BlindPlanetaryLawError("campaign JSON must be an object")
    return value


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fraction_data(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _fraction(value: Any) -> Fraction:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise BlindPlanetaryLawError("rational value schema changed")
    numerator, denominator = value["numerator"], value["denominator"]
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator <= 0
    ):
        raise BlindPlanetaryLawError("rational value is not an exact positive-denominator pair")
    return Fraction(numerator, denominator)


def _decimal_fraction(text: Any, *, places: int | None = None) -> Fraction:
    """Parse a plain decimal string exactly; optionally pin its decimal-place count."""

    if not isinstance(text, str) or not re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]?\d+)?", text):
        raise BlindPlanetaryLawError("declared decimal string malformed")
    if "e" in text or "E" in text:
        if places is not None:
            raise BlindPlanetaryLawError("exponential form cannot pin decimal places")
        mantissa, _, exponent = text.lower().partition("e")
        return _decimal_fraction(mantissa) * Fraction(10) ** int(exponent)
    whole, _, decimals = text.partition(".")
    if places is not None and len(decimals) != places:
        raise BlindPlanetaryLawError("declared decimal string has the wrong precision")
    sign = -1 if whole.startswith("-") else 1
    digits = int(whole.lstrip("-") + decimals) if (whole.lstrip("-") + decimals) else 0
    return Fraction(sign * digits, 10 ** len(decimals))


def _round_to_places(value: Fraction, places: int) -> Fraction:
    """Round-half-even a Fraction to `places` decimals, exactly."""

    scaled = value * 10**places
    floor = scaled.numerator // scaled.denominator
    remainder = scaled - floor
    half = Fraction(1, 2)
    if remainder > half or (remainder == half and floor % 2 == 1):
        floor += 1
    return Fraction(floor, 10**places)


# ---------------------------------------------------------------------------
# Declared derived-view grammar
# ---------------------------------------------------------------------------


def _pair_views() -> tuple[dict[str, Any], ...]:
    """`z = x_response^v * x1^(-u)`, coprime (u, v), ordered by total degree."""

    raw = [
        {"u": u, "v": v}
        for v in range(1, PAIR_RESPONSE_EXPONENT_BOUND + 1)
        for u in range(-PAIR_PREDICTOR_EXPONENT_BOUND, PAIR_PREDICTOR_EXPONENT_BOUND + 1)
        if u != 0 and gcd(abs(u), v) == 1
    ]
    raw.sort(key=lambda view: (view["v"] + abs(view["u"]), view["v"], view["u"]))
    return tuple(
        {**view, "rank": rank, "view_id": f"u={view['u']};v={view['v']}"}
        for rank, view in enumerate(raw)
    )


def _triple_views() -> tuple[dict[str, Any], ...]:
    """`z = x_response^v * x1^i * (1 - x2^2)^j * x2^k`, ordered by total degree."""

    bound = TRIPLE_PREDICTOR_EXPONENT_BOUND
    raw = [
        {"i": i, "j": j, "k": k, "v": v}
        for v in range(1, TRIPLE_RESPONSE_EXPONENT_BOUND + 1)
        for i in range(-bound, bound + 1)
        for j in range(-bound, bound + 1)
        for k in range(-bound, bound + 1)
    ]
    raw.sort(
        key=lambda view: (
            view["v"] + abs(view["i"]) + abs(view["j"]) + abs(view["k"]),
            view["v"],
            view["i"],
            view["j"],
            view["k"],
        )
    )
    return tuple(
        {
            **view,
            "rank": rank,
            "view_id": f"v={view['v']};i={view['i']};j={view['j']};k={view['k']}",
        }
        for rank, view in enumerate(raw)
    )


VIEW_FAMILIES: dict[str, tuple[dict[str, Any], ...]] = {
    "power_pair": _pair_views(),
    "power_triple": _triple_views(),
}

DECLARED_VIEW_FAMILIES = {
    "power_pair": {
        "bases": ["x1"],
        "exponent_range": [-PAIR_PREDICTOR_EXPONENT_BOUND, PAIR_PREDICTOR_EXPONENT_BOUND],
        "response_exponent_range": [1, PAIR_RESPONSE_EXPONENT_BOUND],
        "template": (
            "z = x_response^v * x1^(-u), reported as x_response = c^(1/v) * x1^(u/v)"
        ),
    },
    "power_triple": {
        "bases": ["x1", "(1 - x2^2)", "x2"],
        "exponent_range": [-TRIPLE_PREDICTOR_EXPONENT_BOUND, TRIPLE_PREDICTOR_EXPONENT_BOUND],
        "response_exponent_range": [1, TRIPLE_RESPONSE_EXPONENT_BOUND],
        "template": (
            "z = x_response^v * x1^i * (1 - x2^2)^j * x2^k, reported as "
            "x_response = (c * x1^(-i) * (1 - x2^2)^(-j) * x2^(-k))^(1/v)"
        ),
    },
}

VIEW_SEARCH_SPACE = {
    "power_pair_views": len(VIEW_FAMILIES["power_pair"]),
    "power_triple_views": len(VIEW_FAMILIES["power_triple"]),
}

WORLD_COLUMN_SHAPE = {
    "power_pair": ("x1", "x2"),
    "power_triple": ("x1", "x2", "x3"),
}


# ---------------------------------------------------------------------------
# Config validation and the forbidden-vocabulary guard
# ---------------------------------------------------------------------------


def config_vocabulary_violations(text: str) -> list[str]:
    """Forbidden tokens present in `text`, tokenized on letter runs only."""

    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return sorted(tokens & set(FORBIDDEN_VOCABULARY))


def _assert_config_vocabulary(config: Mapping[str, Any]) -> None:
    violations = config_vocabulary_violations(canonical_json_bytes(config).decode("utf-8"))
    if violations:
        raise BlindPlanetaryLawError(
            f"public config leaked target vocabulary: {', '.join(violations)}"
        )


def _validate_rows(world: Mapping[str, Any]) -> None:
    columns = WORLD_COLUMN_SHAPE[world["view_family"]]
    if tuple(world["columns"]) != columns:
        raise BlindPlanetaryLawError("world column inventory changed")
    if world["response_column"] != columns[-1]:
        raise BlindPlanetaryLawError("response column changed")
    rows = world["rows"]
    if not isinstance(rows, list) or len(rows) < 8 or len(rows) > 64:
        raise BlindPlanetaryLawError("public row budget changed")
    expected_index = 1
    previous_predictor: Fraction | None = None
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"index", *columns}:
            raise BlindPlanetaryLawError("public row schema changed")
        if row["index"] != expected_index:
            raise BlindPlanetaryLawError("public row index is not consecutive from one")
        expected_index += 1
        predictor = _fraction(row["x1"])
        if previous_predictor is not None and predictor <= previous_predictor:
            raise BlindPlanetaryLawError("public rows are not strictly ascending in x1")
        previous_predictor = predictor
        for column in columns:
            if _fraction(row[column]) <= 0:
                raise BlindPlanetaryLawError("public column values must be positive")


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "data_declaration",
        "policies",
        "schema_version",
        "target_fixture_commitment_sha256",
        "view_families",
        "worlds",
    }:
        raise BlindPlanetaryLawError("config keys changed")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise BlindPlanetaryLawError("config identity changed")
    if config["policies"] != POLICIES:
        raise BlindPlanetaryLawError("prospective policy changed")
    if not _hex_digest(config["target_fixture_commitment_sha256"]):
        raise BlindPlanetaryLawError("target fixture commitment malformed")
    if config["view_families"] != DECLARED_VIEW_FAMILIES:
        raise BlindPlanetaryLawError("declared view grammar changed")
    declaration = config["data_declaration"]
    if not isinstance(declaration, Mapping) or set(declaration) != {
        "boundary",
        "declared_constant_decimal_places",
        "provenance_commitment_sha256",
        "root_quantization_decimal_places",
        "row_index_meaning",
        "values_are_exact_rationals",
    }:
        raise BlindPlanetaryLawError("data declaration schema changed")
    if (
        declaration["declared_constant_decimal_places"] != DECLARED_CONSTANT_DECIMAL_PLACES
        or declaration["root_quantization_decimal_places"] != ROOT_QUANTIZATION_DECIMAL_PLACES
        or declaration["values_are_exact_rationals"] is not True
        or not _hex_digest(declaration["provenance_commitment_sha256"])
    ):
        raise BlindPlanetaryLawError("data declaration content changed")
    worlds = config["worlds"]
    if not isinstance(worlds, list) or len(worlds) != len(WORLD_IDS):
        raise BlindPlanetaryLawError("world inventory changed")
    for expected_id, world in zip(WORLD_IDS, worlds, strict=True):
        if not isinstance(world, Mapping) or set(world) != {
            "columns",
            "response_column",
            "rows",
            "sealed_target_sha256",
            "view_family",
            "world_id",
        }:
            raise BlindPlanetaryLawError("public world schema changed")
        if world["world_id"] != expected_id:
            raise BlindPlanetaryLawError("anonymized world identity changed")
        if world["view_family"] not in VIEW_FAMILIES:
            raise BlindPlanetaryLawError("world view family left the declared grammar")
        if not _hex_digest(world["sealed_target_sha256"]):
            raise BlindPlanetaryLawError("sealed target commitment malformed")
        _validate_rows(world)
    _assert_config_vocabulary(config)


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
    """Trigger every lazy import B2 and the comparator need before the guard closes."""

    search_nonlinear([{"point": point, "value": 2**point} for point in range(1, 7)])
    search_nonlinear(
        [
            {"point": point, "value": {"numerator": point, "denominator": point + 1}}
            for point in range(1, 8)
        ]
    )
    x1, x2 = _domain_symbols()[:2]
    sp.simplify(sp.expand(x1 ** sp.Rational(3, 2) - x1 * sp.sqrt(x1)))
    sp.cancel(sp.together(sp.Rational(1, 3) / (x1 * (1 - x2**2)) - sp.Rational(1, 3)))
    sp.sympify("Rational(1, 2)*x1**Rational(-2, 1)", locals=_sympy_locals())


# ---------------------------------------------------------------------------
# Symbolic surface
# ---------------------------------------------------------------------------


def _domain_symbols() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    return (
        sp.Symbol("x1", positive=True),
        sp.Symbol("x2", positive=True),
        sp.Symbol("x3", positive=True),
    )


def _sympy_locals() -> dict[str, Any]:
    x1, x2, x3 = _domain_symbols()
    return {"Rational": sp.Rational, "x1": x1, "x2": x2, "x3": x3}


def _parse_expression(text: str) -> sp.Expr:
    try:
        return sp.sympify(text, locals=_sympy_locals())
    except (sp.SympifyError, TypeError, SyntaxError) as error:
        raise BlindPlanetaryLawError("expression unparseable") from error


def _equivalent(lhs: sp.Expr, rhs: sp.Expr) -> tuple[bool, str]:
    difference = sp.expand(sp.expand_func(lhs - rhs))
    if difference == 0:
        return True, "0"
    simplified = sp.simplify(difference)
    if simplified == 0:
        return True, "0"
    collapsed = sp.cancel(sp.together(difference))
    return collapsed == 0, str(simplified)


def _rational_text(value: Fraction) -> str:
    return f"Rational({value.numerator}, {value.denominator})"


def _power_text(base: str, exponent: int) -> str:
    if exponent == 1:
        return base
    return f"{base}^({exponent})" if exponent < 0 else f"{base}^{exponent}"


def _pair_law_texts(view: Mapping[str, Any], constant: Fraction) -> dict[str, str]:
    numerator, denominator = view["u"], view["v"]
    exponent = Fraction(numerator, denominator)
    sympy_text = (
        f"({_rational_text(constant)})**Rational(1, {denominator})"
        f"*x1**Rational({numerator}, {denominator})"
    )
    if constant == 1:
        scale = ""
    elif denominator == 1:
        scale = f"({constant})*"
    else:
        scale = f"({constant})^(1/{denominator})*"
    expression = f"{scale}x1^({exponent})"
    response = _power_text("x_response", denominator)
    invariant = (
        f"{response}/{_power_text('x1', numerator)}"
        if numerator > 0
        else f"{response}*{_power_text('x1', -numerator)}"
    )
    return {
        "expression": expression,
        "invariant_expression": invariant,
        "statement": f"x_response = {expression}",
        "sympy_expression": sympy_text,
    }


def _triple_law_texts(view: Mapping[str, Any], constant: Fraction) -> dict[str, str]:
    factors = [_rational_text(constant)]
    above = [str(constant) if constant.denominator == 1 else f"({constant})"]
    below: list[str] = []
    invariant = [_power_text("x_response", view["v"])]
    for exponent, sympy_base, readable_base in (
        (view["i"], "x1", "x1"),
        (view["j"], "(1 - x2**2)", "(1 - x2^2)"),
        (view["k"], "x2", "x2"),
    ):
        if not exponent:
            continue
        factors.append(f"{sympy_base}**({-exponent})")
        invariant.append(_power_text(readable_base, exponent))
        target = below if exponent > 0 else above
        target.append(_power_text(readable_base, abs(exponent)))
    inner_sympy = "*".join(factors)
    inner_readable = "*".join(above)
    if below:
        joined = below[0] if len(below) == 1 else f"({'*'.join(below)})"
        inner_readable = f"{inner_readable}/{joined}"
    body = inner_readable if view["v"] == 1 else f"({inner_readable})^(1/{view['v']})"
    return {
        "expression": body,
        "invariant_expression": "*".join(invariant),
        "statement": f"x_response = {body}",
        "sympy_expression": f"({inner_sympy})**Rational(1, {view['v']})",
    }


def _law_texts(family: str, view: Mapping[str, Any], constant: Fraction) -> dict[str, str]:
    if family == "power_pair":
        return _pair_law_texts(view, constant)
    return _triple_law_texts(view, constant)


def _companion_degree(family: str, view: Mapping[str, Any]) -> int:
    if family == "power_pair":
        return abs(view["u"])
    return abs(view["i"]) + 2 * abs(view["j"]) + abs(view["k"])


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


def _raw_rows(world: Mapping[str, Any]) -> list[dict[str, Any]]:
    response = world["response_column"]
    return [{"point": row["index"], "value": dict(row[response])} for row in world["rows"]]


def _view_rows(
    family: str, view: Mapping[str, Any], world: Mapping[str, Any]
) -> list[dict[str, Any]] | None:
    """Exact derived column for one declared view, or None where it is undefined."""

    response = world["response_column"]
    derived: list[dict[str, Any]] = []
    for row in world["rows"]:
        value = _fraction(row[response]) ** view["v"]
        try:
            if family == "power_pair":
                value *= _fraction(row["x1"]) ** (-view["u"])
            else:
                complement = Fraction(1) - _fraction(row["x2"]) ** 2
                value *= _fraction(row["x1"]) ** view["i"]
                value *= complement ** view["j"]
                value *= _fraction(row["x2"]) ** view["k"]
        except ZeroDivisionError:
            return None
        derived.append({"point": row["index"], "value": _fraction_data(value)})
    return derived


def _constant_value(rows: Sequence[Mapping[str, Any]]) -> Fraction | None:
    values = {_fraction(row["value"]) for row in rows}
    return values.pop() if len(values) == 1 else None


def _run_view_search(world: Mapping[str, Any]) -> dict[str, Any]:
    """Run B1 and B2 on every declared view; log the whole bounded search."""

    family = world["view_family"]
    log: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    for view in VIEW_FAMILIES[family]:
        rows = _view_rows(family, view, world)
        entry: dict[str, Any] = {
            "exponents": {key: view[key] for key in view if key not in {"rank", "view_id"}},
            "rank": view["rank"],
            "view_id": view["view_id"],
        }
        if rows is None:
            log.append({**entry, "status": "UNDEFINED", "reason": "view_undefined_on_a_row"})
            continue
        b1 = synthesize_basis(rows)
        b2 = search_nonlinear(rows)
        b1_result = b1["result"] or {}
        b2_result = b2["result"] or {}
        constant = _constant_value(rows)
        is_admitted = (
            b1["decision"] == "PASS"
            and b1_result.get("family_id") == ADMITTED_FAMILY_ID
            and constant is not None
        )
        entry.update(
            {
                "b1_confirmations": b1_result.get("confirmations"),
                "b1_decision": b1["decision"],
                "b1_family_id": b1_result.get("family_id"),
                "b1_receipt_sha256": b1["content_sha256"],
                "b2_decision": b2["decision"],
                "b2_model_id": b2_result.get("model_id"),
                "b2_receipt_sha256": b2["content_sha256"],
                "constant": None if constant is None else _fraction_data(constant),
                "status": "ADMITTED" if is_admitted else "REJECTED",
            }
        )
        if not is_admitted and b1["decision"] == "PASS":
            entry["reason"] = "index_dependent_basis_is_not_admissible"
        log.append(entry)
        if is_admitted:
            admitted.append(
                {
                    "b1": b1,
                    "b2": b2,
                    "constant": constant,
                    "entry": entry,
                    "rows": rows,
                    "view": view,
                }
            )
    return {"admitted": admitted, "log": log}


def _view_candidate(
    world: Mapping[str, Any], admitted: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    if not admitted:
        return None
    winner = admitted[0]
    family = world["view_family"]
    view, constant = winner["view"], winner["constant"]
    b1_result, b2_result = winner["b1"]["result"], winner["b2"]["result"] or {}
    texts = _law_texts(family, view, constant)
    exponents = {key: view[key] for key in view if key not in {"rank", "view_id"}}
    candidate: dict[str, Any] = {
        **texts,
        "b1_receipt_sha256": winner["b1"]["content_sha256"],
        "b2_corroborates_constancy": (
            b2_result.get("model_id") == "pure_geometric"
            and b2_result.get("status") == "PASS"
            and _fraction(b2_result["parameters"]["b"]) == 1
        ),
        "b2_receipt_sha256": winner["b2"]["content_sha256"],
        "companion_degree": _companion_degree(family, view),
        "confirmations": b1_result["confirmations"],
        "constant": _fraction_data(constant),
        "kind": "power_law" if family == "power_pair" else "derived_power_law",
        "rejected_simpler_views": view["rank"],
        "source_stage": "b4_declared_view_search",
        "view_exponents": exponents,
        "view_family": family,
        "view_id": view["view_id"],
    }
    if family == "power_pair":
        candidate["exponent"] = str(Fraction(view["u"], view["v"]))
    else:
        candidate["derived_column"] = texts["invariant_expression"]
    return candidate


def _survivors(b3: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["kind"]: row for row in b3["conjectures"] if row.get("status") == "SURVIVED"}


def _lean_namespace(world_id: str) -> str:
    return "BlindColumnWorld" + world_id.title().replace("_", "")


def _route_provers(
    world_id: str, candidate: Mapping[str, Any] | None, b3: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if candidate is None:
        return [
            {"reason": "no candidate survived Phase A", "route": "none", "status": "NOT_APPLICABLE"}
        ]
    degree = candidate["companion_degree"]
    if not 1 <= degree <= MAX_COMPANION_DEGREE:
        return [
            {
                "reason": (
                    f"companion degree {degree} is outside the declared B5 Nat-polynomial "
                    "grammar"
                ),
                "route": "b5_lemma_decomposition",
                "status": "NOT_APPLICABLE",
            }
        ]
    closed_form = [0] * degree + [1]
    step = [comb(degree, index) for index in range(degree)]
    namespace = _lean_namespace(world_id)
    companion = {
        "closed_form_coefficients_ascending": closed_form,
        "degree": degree,
        "degree_rule": COMPANION_DEGREE_RULE,
        "statement": f"companion(n) = n^{degree}",
        "step_coefficients_ascending": step,
    }
    routes: list[dict[str, Any]] = []
    b5 = decompose_closed_form_proof(
        {
            "base_value": 0,
            "closed_form": closed_form,
            "namespace": namespace,
            "sequence_name": "companion",
            "step": step,
        }
    )
    routes.append(
        {
            "boundary": LEAN_BOUNDARY_NOTE,
            "companion": companion,
            "decision": b5["decision"],
            "lean_source_emitted": b5["lean_source"] is not None,
            "receipt": b5,
            "receipt_sha256": b5["content_sha256"],
            "route": "b5_lemma_decomposition",
            "status": "ROUTED",
        }
    )
    survivors = _survivors(b3)
    for kind, statement, relation, name in (
        ("monotonicity", "a(n) < a(n+1)", "monotone_increasing", "companionStrictlyIncreasing"),
        ("sign", "a(n) > 0", "nonnegative", "companionNonnegative"),
    ):
        survivor = survivors.get(kind)
        if survivor is None or survivor["statement"] != statement:
            continue
        b6 = prove_quantified_inequality(
            {
                "coefficients": closed_form,
                "name": name,
                "namespace": namespace,
                "relation": relation,
            }
        )
        routes.append(
            {
                "boundary": LEAN_BOUNDARY_NOTE,
                "companion": companion,
                "decision": b6["decision"],
                "lean_source_emitted": b6["lean_source"] is not None,
                "receipt": b6,
                "receipt_sha256": b6["content_sha256"],
                "relation": relation,
                "route": "b6_quantified_inequality",
                "status": "ROUTED",
                "survivor_statement": survivor["statement"],
            }
        )
    return routes


def _run_world_phase_a(world: Mapping[str, Any]) -> dict[str, Any]:
    rows = _raw_rows(world)
    stages: list[dict[str, Any]] = []
    b1 = synthesize_basis(rows)
    stages.append(_stage("b1_basis_synthesis", "synthesize_basis", b1))
    b7: dict[str, Any] | None = None
    if b1["decision"] != "PASS":
        b7 = repair_structure(rows)
        stages.append(_stage("b7_structural_repair", "repair_structure", b7))
    b3 = generate_conjectures(rows, statement_kinds=STATEMENT_KINDS)
    stages.append(_stage("b3_conjecture_generation", "generate_conjectures", b3))
    search = _run_view_search(world)
    stages.append(
        {
            "decision": "PASS" if search["admitted"] else "BLOCK",
            "receipt_sha256": canonical_sha256(search["log"]),
            "stage_id": "b4_declared_view_search",
            "tool": "synthesize_basis+search_nonlinear_over_declared_views",
            "views_admitted": len(search["admitted"]),
            "views_evaluated": len(search["log"]),
        }
    )
    candidate = _view_candidate(world, search["admitted"])
    raw_index_stages = {
        stage["stage_id"]: stage["decision"]
        for stage in stages
        if stage["stage_id"] != "b4_declared_view_search"
    }
    prover_routes = _route_provers(world["world_id"], candidate, b3)
    accepted_receipts = (
        {
            "b1": search["admitted"][0]["b1"],
            "b2": search["admitted"][0]["b2"],
            "derived_rows": search["admitted"][0]["rows"],
        }
        if search["admitted"]
        else None
    )
    return {
        "accepted_view_receipts": accepted_receipts,
        "candidate": candidate,
        "candidate_selection_rule": CANDIDATE_SELECTION_RULE,
        "candidate_statement": None if candidate is None else candidate["statement"],
        "derived_view_search": {
            "family": world["view_family"],
            "log": search["log"],
            "views_admitted": [item["entry"]["view_id"] for item in search["admitted"]],
            "views_declared": len(VIEW_FAMILIES[world["view_family"]]),
            "views_evaluated": len(search["log"]),
        },
        "index_dependent_raw_stage_decisions": raw_index_stages,
        "lean_emitted": any(route.get("lean_source_emitted") is True for route in prover_routes),
        "prover_routes": prover_routes,
        "public_rows_sha256": canonical_sha256(world["rows"]),
        "raw_rows_sha256": canonical_sha256(rows),
        "sealed_target_sha256": world["sealed_target_sha256"],
        "stages": stages,
        "world_id": world["world_id"],
    }


# ---------------------------------------------------------------------------
# Atomic unseal
# ---------------------------------------------------------------------------


def _unseal_targets(
    root: Path, config: Mapping[str, Any], phase_a_root: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    """Open the sealed fixture once, after Phase A is sealed, and open every commitment."""

    if not _hex_digest(phase_a_root):
        raise BlindPlanetaryLawError("target unseal attempted before candidate freeze")
    target_path = _resolve(root, TARGETS_PATH)
    try:
        raw = target_path.read_bytes()
        fixture = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BlindPlanetaryLawError("target fixture unavailable") from error
    if (
        not isinstance(fixture, dict)
        or canonical_sha256(fixture) != config["target_fixture_commitment_sha256"]
    ):
        raise BlindPlanetaryLawError("target fixture content changed")
    if set(fixture) != {"provenance", "schema_version", "targets"} or (
        fixture["schema_version"] != TARGET_SCHEMA
    ):
        raise BlindPlanetaryLawError("target fixture schema changed")
    provenance = fixture["provenance"]
    if (
        not isinstance(provenance, dict)
        or canonical_sha256(provenance)
        != config["data_declaration"]["provenance_commitment_sha256"]
    ):
        raise BlindPlanetaryLawError("provenance commitment did not open")
    targets = fixture["targets"]
    if not isinstance(targets, list) or len(targets) != len(config["worlds"]):
        raise BlindPlanetaryLawError("atomic target batch incomplete")
    configured = {world["world_id"]: world for world in config["worlds"]}
    by_world: dict[str, dict[str, Any]] = {}
    for target in targets:
        if not isinstance(target, dict) or set(target) != {
            "attribution",
            "attribution_year",
            "classical_id",
            "column_meanings",
            "expression",
            "invariant_expression",
            "kind",
            "parameters",
            "salt",
            "statement",
            "world_id",
            "world_label",
        }:
            raise BlindPlanetaryLawError("target record schema changed")
        world_id = target["world_id"]
        if world_id not in configured or world_id in by_world:
            raise BlindPlanetaryLawError("target world identity changed")
        if canonical_sha256(target) != configured[world_id]["sealed_target_sha256"]:
            raise BlindPlanetaryLawError("target commitment did not open")
        classical = target["classical_id"]
        if not isinstance(classical, str) or not classical.isidentifier():
            raise BlindPlanetaryLawError("classical identifier malformed")
        if target["kind"] not in {"derived_power_law", "power_law"}:
            raise BlindPlanetaryLawError("target kind changed")
        by_world[world_id] = target
    if set(by_world) != set(configured):
        raise BlindPlanetaryLawError("atomic target world coverage changed")
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return by_world, provenance, hashlib.sha256(normalized).hexdigest()


# ---------------------------------------------------------------------------
# Generative-rule replay (post-unseal integrity)
# ---------------------------------------------------------------------------


def _regenerate_columns(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild every public column from the sealed rule in exact rational arithmetic."""

    constants = provenance["declared_constants"]
    if constants["pi_60dp"] != PI_60DP:
        raise BlindPlanetaryLawError("declared circle constant changed")
    places = DECLARED_CONSTANT_DECIMAL_PLACES
    pi_value = _decimal_fraction(PI_60DP, places=60)
    four_pi_squared = _decimal_fraction(constants["four_pi_squared_50dp"], places=places)
    if four_pi_squared != _round_to_places(4 * pi_value * pi_value, places):
        raise BlindPlanetaryLawError("declared scale constant is not the stated rounding")
    coefficient = _decimal_fraction(
        constants["perihelion_advance_coefficient_arcsec_au_50dp"], places=places
    )
    expected_coefficient = _round_to_places(
        Fraction(ADVANCE_COEFFICIENT_NUMERIC_FACTOR)
        * _decimal_fraction(constants["heliocentric_gravitational_constant_m3_s2"])
        / (
            _decimal_fraction(constants["speed_of_light_m_s"]) ** 2
            * _decimal_fraction(constants["astronomical_unit_m"])
        ),
        places,
    )
    if coefficient != expected_coefficient:
        raise BlindPlanetaryLawError("declared advance constant is not the stated rounding")

    half_ulp = Fraction(5, 10 ** (ROOT_QUANTIZATION_DECIMAL_PLACES + 1))
    axes: list[Fraction] = []
    periods: list[Fraction] = []
    eccentricities: list[Fraction] = []
    axis_deviation = Fraction(0)
    period_deviation = Fraction(0)
    for anchor in provenance["anchors"]:
        root = _decimal_fraction(
            anchor["quantized_root_of_semi_major_axis"],
            places=ROOT_QUANTIZATION_DECIMAL_PLACES,
        )
        published_axis = _decimal_fraction(anchor["semi_major_axis_au"])
        if not (root - half_ulp) ** 2 <= published_axis <= (root + half_ulp) ** 2:
            raise BlindPlanetaryLawError("quantized root is not the correctly rounded value")
        axis, period = root * root, root**3
        axes.append(axis)
        periods.append(period)
        eccentricities.append(_decimal_fraction(anchor["eccentricity"]))
        axis_deviation = max(axis_deviation, abs(published_axis - axis) / published_axis)
        published_period = _decimal_fraction(anchor["sidereal_orbit_period_yr"])
        period_deviation = max(
            period_deviation, abs(published_period - period) / published_period
        )
    accelerations = [
        four_pi_squared * axis / (period * period)
        for axis, period in zip(axes, periods, strict=True)
    ]
    advances = [
        coefficient / (axis * (1 - eccentricity**2))
        for axis, eccentricity in zip(axes, eccentricities, strict=True)
    ]
    return {
        "axis_deviation": axis_deviation,
        "columns": {
            "world_01": [axes, periods],
            "world_02": [axes, accelerations],
            "world_03": [axes, eccentricities, advances],
            "world_04": [axes, accelerations],
        },
        "period_deviation": period_deviation,
    }


def _verify_generative_rule(
    config: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    rebuilt = _regenerate_columns(provenance)
    rows_verified = 0
    for world in config["worlds"]:
        columns = rebuilt["columns"][world["world_id"]]
        names = WORLD_COLUMN_SHAPE[world["view_family"]]
        if len(columns) != len(names) or len(columns[0]) != len(world["rows"]):
            raise BlindPlanetaryLawError("sealed rule does not cover the public rows")
        for index, row in enumerate(world["rows"]):
            for name, column in zip(names, columns, strict=True):
                if _fraction(row[name]) != column[index]:
                    raise BlindPlanetaryLawError("public rows do not replay from the sealed rule")
            rows_verified += 1
    declared = provenance["fidelity"]
    if _decimal_fraction(
        declared["max_relative_deviation_semi_major_axis"], places=20
    ) != _round_to_places(rebuilt["axis_deviation"], 20) or _decimal_fraction(
        declared["max_relative_deviation_sidereal_period"], places=20
    ) != _round_to_places(rebuilt["period_deviation"], 20):
        raise BlindPlanetaryLawError("declared fidelity bounds do not match the sealed anchors")
    return {
        "anchor_fidelity": {
            "max_relative_deviation_semi_major_axis": declared[
                "max_relative_deviation_semi_major_axis"
            ],
            "max_relative_deviation_sidereal_period": declared[
                "max_relative_deviation_sidereal_period"
            ],
            "note": declared["note"],
        },
        "declared_constants_recomputed": True,
        "public_rows_replayed_from_sealed_rule": True,
        "quantized_roots_are_correctly_rounded": True,
        "rows_verified": rows_verified,
    }


# ---------------------------------------------------------------------------
# Exact comparison
# ---------------------------------------------------------------------------


def _compare_candidate(
    candidate: Mapping[str, Any] | None, target: Mapping[str, Any]
) -> dict[str, Any]:
    if candidate is None:
        return {
            "base_verdict": "MISSED",
            "detail": {"reason": "no candidate survived Phase A"},
            "equivalent": False,
            "method": "no_candidate",
        }
    parameters = target["parameters"]
    if candidate["kind"] != target["kind"]:
        return {
            "base_verdict": "MISSED",
            "detail": {"reason": f"candidate kind {candidate['kind']} is not {target['kind']}"},
            "equivalent": False,
            "method": "kind_mismatch",
        }
    structure_match = (
        candidate["view_family"] == parameters["view_family"]
        and candidate["view_exponents"] == parameters["view_exponents"]
    )
    constant_match = _fraction(candidate["constant"]) == Fraction(parameters["constant"])
    discovered = _parse_expression(candidate["sympy_expression"])
    expected = _parse_expression(target["expression"])
    equivalent, residual = _equivalent(discovered, expected)
    exponent_match = None
    if target["kind"] == "power_law":
        exponent_match = Fraction(candidate["exponent"]) == Fraction(parameters["exponent"])
    derived_column_match = None
    if target["kind"] == "derived_power_law":
        derived_column_match = structure_match
    if structure_match and constant_match and equivalent:
        verdict = "REDISCOVERED_EXACT"
    elif structure_match or exponent_match:
        verdict = "PARTIAL"
    else:
        verdict = "MISSED"
    return {
        "base_verdict": verdict,
        "detail": {
            "constant_match": constant_match,
            "derived_column_match": derived_column_match,
            "discovered_sympy": str(discovered),
            "exponent_match": exponent_match,
            "residual": residual,
            "structure_match": structure_match,
            "target_sympy": str(expected),
        },
        "equivalent": equivalent,
        "method": "sympy_equivalence_plus_exact_rational_structure_equality",
    }


def _score_world(
    phase_a_world: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    comparison = _compare_candidate(phase_a_world["candidate"], target)
    routed = [route for route in phase_a_world["prover_routes"] if route.get("status") == "ROUTED"]
    provers_green = bool(routed) and all(
        route["decision"] in {"DECOMPOSED", "PROVED_LOCALLY"} for route in routed
    )
    return {
        "attribution": target["attribution"],
        "attribution_year": target["attribution_year"],
        "classical_id": target["classical_id"],
        "column_meanings": target["column_meanings"],
        "commitment_opened": True,
        "comparison": comparison,
        "lean_boundary": LEAN_BOUNDARY_NOTE,
        "provers_green": provers_green,
        "target_record": dict(target),
        "target_statement": target["statement"],
        "verdict": comparison["base_verdict"],
        "world_label": target["world_label"],
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
            raise BlindPlanetaryLawError("pre-unseal target read was not denied")
    denied_probe = guard.certificate()
    if (
        denied_probe["attempted_target_reads"] != 1
        or denied_probe["denied_target_reads"] != 1
        or denied_probe["denied_content_bytes_exposed"] != 0
        or denied_probe["denied_paths"] != [TARGETS_PATH]
    ):
        raise BlindPlanetaryLawError("sealed-target enforcement boundary changed")
    phase_a_root = canonical_sha256(
        {
            "commitments": [world["sealed_target_sha256"] for world in config["worlds"]],
            "denied_probe": denied_probe,
            "worlds": phase_worlds,
        }
    )
    targets, provenance, target_file_sha256 = _unseal_targets(root, config, phase_a_root)
    generative_rule = _verify_generative_rule(config, provenance)

    world_receipts: dict[str, dict[str, Any]] = {}
    world_rows: list[dict[str, Any]] = []
    for world, phase_a_world in zip(config["worlds"], phase_worlds, strict=True):
        target = targets[world["world_id"]]
        unseal = _score_world(phase_a_world, target)
        receipt_body = {
            "campaign_id": CAMPAIGN_ID,
            "classical_id": target["classical_id"],
            "phase_a": phase_a_world,
            "phase_a_root": phase_a_root,
            "public_rows": world["rows"],
            "schema_version": WORLD_SCHEMA,
            "sealed_target_sha256": world["sealed_target_sha256"],
            "unseal": unseal,
            "world_id": world["world_id"],
        }
        receipt = {**receipt_body, "content_sha256": canonical_sha256(receipt_body)}
        world_receipts[target["classical_id"]] = receipt
        candidate = phase_a_world["candidate"]
        search = phase_a_world["derived_view_search"]
        world_rows.append(
            {
                "attribution": target["attribution"],
                "attribution_year": target["attribution_year"],
                "classical_id": target["classical_id"],
                "column_meanings": target["column_meanings"],
                "comparison_detail": unseal["comparison"]["detail"],
                "derived_view_search": search,
                "discovered_constant": None if candidate is None else candidate["constant"],
                "discovered_exponent": None if candidate is None else candidate.get("exponent"),
                "discovered_invariant": (
                    None if candidate is None else candidate["invariant_expression"]
                ),
                "discovered_statement": phase_a_world["candidate_statement"],
                "holdout_confirmations": (
                    None if candidate is None else candidate["confirmations"]
                ),
                "lean_emitted": phase_a_world["lean_emitted"],
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
                "rejected_simpler_views": (
                    None if candidate is None else candidate["rejected_simpler_views"]
                ),
                "stage_trail": [
                    {
                        "decision": stage["decision"],
                        "receipt_sha256": stage["receipt_sha256"],
                        "stage_id": stage["stage_id"],
                        "tool": stage["tool"],
                    }
                    for stage in phase_a_world["stages"]
                ],
                "target_statement": target["statement"],
                "verdict": unseal["verdict"],
                "world_id": world["world_id"],
                "world_label": target["world_label"],
                "world_receipt_path": f"{WORLD_RECEIPT_DIRECTORY}/{target['classical_id']}.json",
                "world_receipt_sha256": receipt["content_sha256"],
            }
        )

    verdicts = [row["verdict"] for row in world_rows]
    rediscovered = sum(1 for verdict in verdicts if verdict == "REDISCOVERED_EXACT")
    counts = {
        "holdout_confirmations_total": sum(row["holdout_confirmations"] or 0 for row in world_rows),
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
        "rediscovered_exact": rediscovered,
        "stage_receipts": sum(len(phase_a_world["stages"]) for phase_a_world in phase_worlds),
        "target_fixture_reads": 1,
        "target_fixture_reads_denied_before_unseal": 1,
        "views_admitted": sum(
            len(phase_a_world["derived_view_search"]["views_admitted"])
            for phase_a_world in phase_worlds
        ),
        "views_evaluated": sum(
            phase_a_world["derived_view_search"]["views_evaluated"]
            for phase_a_world in phase_worlds
        ),
        "worlds": len(world_rows),
    }
    decision = (
        "PASS" if rediscovered >= config["policies"]["minimum_rediscoveries_for_pass"] else "BLOCK"
    )
    machine_found = (
        rediscovered == len(world_rows)
        and denied_probe["denied_target_reads"] == 1
        and counts["post_unseal_generation_events"] == 0
    )
    chronology = {
        "denied_probe": denied_probe,
        "events": [
            {"event": "config_and_public_rows_loaded", "sequence": 0, "target_reads": 0},
            {"event": "forbidden_vocabulary_guard_cleared", "sequence": 1, "target_reads": 0},
            {"event": "sealed_targets_guard_entered", "sequence": 2, "target_reads": 0},
            {
                "event": "stage_ladder_view_search_and_candidates_frozen_all_worlds",
                "sequence": 3,
                "target_reads": 0,
            },
            {"event": "instrumented_denied_probe_recorded", "sequence": 4, "target_reads": 0},
            {
                "event": "phase_a_root_sealed",
                "root_sha256": phase_a_root,
                "sequence": 5,
                "target_reads": 0,
            },
            {"event": "atomic_target_unseal", "sequence": 6, "target_reads": 1},
            {"event": "sealed_generative_rule_replayed", "sequence": 7, "target_reads": 1},
            {"event": "commitments_opened_and_exact_comparison", "sequence": 8, "target_reads": 1},
        ],
        "phase_a_root": phase_a_root,
        "unseal_batches": 1,
    }
    body = {
        "campaign_id": CAMPAIGN_ID,
        "candidate_selection_rule": CANDIDATE_SELECTION_RULE,
        "chronology": chronology,
        "claims": {**STATIC_CLAIMS, "machine_found_laws_unaided": machine_found},
        "counts": counts,
        "decision": decision,
        "declared_view_families": DECLARED_VIEW_FAMILIES,
        "first_blocker": None if decision == "PASS" else "minimum_rediscovery_threshold_not_met",
        "generative_rule_verification": generative_rule,
        "policies": dict(config["policies"]),
        "response_alias": RESPONSE_ALIAS,
        "schema_version": RESULT_SCHEMA,
        "scope": SCOPE,
        "source_bindings": {
            "config": {
                "file_sha256": _file_sha256(_resolve(root, CONFIG_PATH)),
                "path": CONFIG_PATH,
            },
            "doc": {"file_sha256": _file_sha256(_resolve(root, DOC_PATH)), "path": DOC_PATH},
            "source": {
                "file_sha256": _file_sha256(_resolve(root, SOURCE_PATH)),
                "path": SOURCE_PATH,
            },
            "target_fixture": {"file_sha256": target_file_sha256, "path": TARGETS_PATH},
            "test": {"file_sha256": _file_sha256(_resolve(root, TEST_PATH)), "path": TEST_PATH},
        },
        "view_search_space": {
            **VIEW_SEARCH_SPACE,
            "total_views_evaluated": counts["views_evaluated"],
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
        raise BlindPlanetaryLawError("campaign receipt schema changed")
    campaign_body = {key: item for key, item in campaign.items() if key != "content_sha256"}
    if campaign.get("content_sha256") != canonical_sha256(campaign_body):
        raise BlindPlanetaryLawError("campaign receipt seal changed")
    for world in worlds.values():
        world_body = {key: item for key, item in world.items() if key != "content_sha256"}
        if world.get("content_sha256") != canonical_sha256(world_body):
            raise BlindPlanetaryLawError("world receipt seal changed")
    replayed = build_artifacts(root, config_path)
    if dict(campaign) != replayed["campaign"]:
        raise BlindPlanetaryLawError("campaign receipt exact replay changed")
    if {key: dict(value) for key, value in worlds.items()} != replayed["worlds"]:
        raise BlindPlanetaryLawError("world receipt exact replay changed")


def validate_campaign(
    value: Mapping[str, Any], *, root: Path, config_path: Path | None = None
) -> None:
    """Validate the campaign receipt alone by exact deterministic replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise BlindPlanetaryLawError("campaign receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise BlindPlanetaryLawError("campaign receipt seal changed")
    if dict(value) != build_artifacts(root, config_path)["campaign"]:
        raise BlindPlanetaryLawError("campaign receipt exact replay changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise BlindPlanetaryLawError("refusing to overwrite immutable receipt")
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
        _write_immutable(_resolve(root, f"{WORLD_RECEIPT_DIRECTORY}/{classical_id}.json"), receipt)
    validate_artifacts(artifacts["campaign"], artifacts["worlds"], root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
