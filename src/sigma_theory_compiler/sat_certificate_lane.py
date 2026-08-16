"""M10 — SAT certificate lane: bounded combinatorial statements with verified certificates.

Some statements the engine cares about are finite enough to settle by brute force: does a
2-coloring of the edges of K_5 avoid a monochromatic triangle, does {1..9} force a
monochromatic 3-term arithmetic progression, does {1..20} force a monochromatic Pythagorean
triple.  This module compiles such declared statements to CNF with an exact, documented
encoder and hands them to a SAT solver — but the solver is never the last word.

**Certificate discipline is the load-bearing part.**

* A SAT outcome carries its own certificate: the model.  Before anything is sealed, the
  assignment is re-verified against *every clause* in pure Python, independently of the
  solver.  A model that fails that check is an integrity failure and raises instead of
  being sealed.  The decision is ``SAT_MODEL_VERIFIED`` and never anything weaker.
* An UNSAT outcome has no such intrinsic certificate.  It is labeled
  ``UNSAT_SOLVER_ASSERTED`` with ``claims.unsat_independently_verified: false`` — unless a
  ``drat-trim`` binary is found on PATH, a DRAT/DRUP proof was actually extracted from the
  solver during the solve, and ``drat-trim`` actually printed ``s VERIFIED``; only then is
  the decision ``UNSAT_DRAT_VERIFIED``.  An empty extracted proof is never submitted for
  verification, and a solver-asserted refutation is never presented as verified.
  Measured platform fact: on Windows builds of python-sat (1.9.dev15 verified here),
  ``with_proof=True`` returns an empty proof for every backend *and* corrupts interpreter
  shutdown, so :data:`PROOF_EXTRACTION_USABLE` is false on ``win32``, proofs are never
  requested there, and every real Windows UNSAT stays honestly solver-asserted; the
  receipt's ``drat.proof_extraction_usable`` field says so.

**Receipts are deterministic certificates.**  The sealed body carries the exact statement
echo, encoder version, documented variable map, solver name/version, caps echo, decision,
claims, and — for SAT — the canonicalized model (restricted to variables ``1..V``, sorted
by variable, unassigned variables sealed as false).  Wall-clock measurements are
environment facts and are deliberately *not* sealed: two solves of the same statement on
the same environment produce byte-identical receipts.  cadical195 exposes no seed through
python-sat and its default configuration is deterministic for a fixed clause list; the
model canonicalization above removes any residual representational drift.

**Caps are hard, watchdog-style.**  ``max_vars`` and ``max_clauses`` are checked against
cheap predicted counts before any clause list is built; ``max_seconds`` is enforced by a
cooperative interrupt timer plus a terminal wall audit after the solve (a result computed
beyond the budget is discarded, never sealed).  A tripped cap yields a sealed
``CAP_TRIPPED:<cap>`` receipt with no SAT/UNSAT claim — a recorded outcome, never silence.

**Known-answer controls are mechanism receipts.**  R(3,3) = 6, W(2,3) = 9, and the small
Pythagorean instances are classical results used to calibrate the lane; their receipts
establish that the machinery works, not new mathematics.  The famous Pythagorean boundary
n = 7825 (Heule, Kullmann & Marek 2016) is far outside this lane's declared caps and is
stated as out of scope in the receipt itself.

**Discovery-scheduler routing note.**  The A2 problem queue registers the machine-form
kind ``bounded_combinatorial_coloring`` with fields ``{statement_kind, n, k}``;
:func:`statement_from_machine_form` converts such a machine form into a lane statement
(``k`` is 0 for kinds that take no ``k``).  The scheduler does not derive stages for this
kind yet; when it does, the natural stage is a single ``sat_certificate`` stage that calls
``decide(statement_from_machine_form(entry["machine_form"]))`` under the epoch watchdog.
No scheduler edits are part of this module.

Claim boundary: a decision covers exactly the declared finite instance and proves nothing
beyond it.  ``validate_receipt`` re-verifies models and re-solves UNSAT instances as a
tamper check on the decision direction, but a re-solve is still a solver assertion — it
never upgrades ``UNSAT_SOLVER_ASSERTED`` to verified, and no receipt field can.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from itertools import combinations
from math import comb, gcd
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-sat-certificate-lane-result-1.0"
ENCODER_VERSION = "sat-lane-encoder-1.0"
SOLVER_BACKEND = "cadical195"
SOLVER_INTERFACE = "python-sat"
DRAT_TRIM_BINARY = "drat-trim"

#: Measured platform fact, not a preference: on Windows builds of python-sat,
#: ``with_proof=True`` yields an empty proof for every backend and corrupts interpreter
#: shutdown (exit code 127 after normal completion), so proofs are never requested there.
PROOF_EXTRACTION_USABLE = sys.platform != "win32"

DECISION_SAT = "SAT_MODEL_VERIFIED"
DECISION_UNSAT_ASSERTED = "UNSAT_SOLVER_ASSERTED"
DECISION_UNSAT_DRAT = "UNSAT_DRAT_VERIFIED"

SIZE_CAP_NAMES = ("max_vars", "max_clauses")
CAP_NAMES = ("max_clauses", "max_seconds", "max_vars")
_DECISIONS = (
    DECISION_SAT,
    DECISION_UNSAT_ASSERTED,
    DECISION_UNSAT_DRAT,
    "CAP_TRIPPED:max_clauses",
    "CAP_TRIPPED:max_seconds",
    "CAP_TRIPPED:max_vars",
)

STATEMENT_KINDS = (
    "generic_cnf",
    "pythagorean_triple_coloring",
    "ramsey_edge_coloring",
    "vdw_arithmetic_progression",
)

#: Machine-form kinds this lane can serve (registered in problem_queue.MACHINE_FORM_KINDS).
MACHINE_FORM_KIND = "bounded_combinatorial_coloring"
MACHINE_FORM_STATEMENT_KINDS = (
    "pythagorean_triple_coloring",
    "ramsey_edge_coloring",
    "vdw_arithmetic_progression",
)

DEFAULT_CAPS = {"max_clauses": 200_000, "max_seconds": 60, "max_vars": 20_000}

#: Absolute limits.  Caller caps beyond these are a configuration error, never a receipt.
SYSTEM_LIMITS = {
    "max_clauses": 10**7,
    "max_dimacs_bytes": 10**7,
    "max_seconds": 3600,
    "max_statement_parameter": 10**6,
    "max_vars": 10**6,
}

LITERATURE = {
    "pythagorean_triple_coloring": {
        "citation": (
            "Heule, Kullmann & Marek, 'Solving and Verifying the Boolean Pythagorean "
            "Triples Problem via Cube-and-Conquer', SAT 2016, LNCS 9710, pp. 228-245"
        ),
        "note": (
            "The known boundary is n = 7825: every 2-coloring of {1..7825} contains a "
            "monochromatic Pythagorean triple, while {1..7824} admits one that does not. "
            "That instance (a ~200 TB DRAT proof in the literature) is far outside this "
            "lane's declared caps and is not attempted here."
        ),
    },
    "ramsey_edge_coloring": {
        "citation": "Greenwood & Gleason, Canadian J. Math. 7 (1955), pp. 1-7 (R(3,3) = 6)",
        "note": (
            "R(3,3) = 6 is classical: K_5 admits a 2-edge-coloring with no monochromatic "
            "triangle and K_6 does not.  Small instances here are known-answer controls."
        ),
    },
    "vdw_arithmetic_progression": {
        "citation": (
            "Landman & Robertson, 'Ramsey Theory on the Integers', AMS (2004) (W(2,3) = 9)"
        ),
        "note": (
            "W(2,3) = 9 is classical: {1..8} admits a 2-coloring with no monochromatic "
            "3-term AP and {1..9} does not.  Small instances here are known-answer controls."
        ),
    },
}

_SCOPE_BASE = (
    "One declared finite combinatorial statement compiled to CNF by a deterministic, "
    "versioned encoder and decided by a SAT solver under declared hard caps (max_vars and "
    "max_clauses checked against predicted counts before any build; max_seconds enforced "
    "by a cooperative interrupt plus a terminal wall audit, with any over-budget result "
    "discarded rather than sealed).  A SAT decision is sealed only after the model is "
    "re-verified against every clause in pure Python, independently of the solver.  An "
    "UNSAT decision is labeled UNSAT_SOLVER_ASSERTED unless a DRAT proof was extracted "
    "during the solve and checked by an external drat-trim binary, in which case it is "
    "labeled UNSAT_DRAT_VERIFIED; an empty extracted proof is never submitted and a "
    "solver-asserted refutation is never presented as verified.  The decision covers "
    "exactly the declared finite instance and proves nothing beyond it."
)

_SCOPE_KIND_NOTES = {
    "generic_cnf": (
        "  The caller supplied this CNF; the lane asserts nothing about what the formula "
        "encodes, only whether it is satisfiable."
    ),
    "pythagorean_triple_coloring": (
        "  Known boundary for this family: n = 7825 — every 2-coloring of {1..7825} "
        "contains a monochromatic Pythagorean triple while {1..7824} does not (Heule, "
        "Kullmann & Marek, 'Solving and Verifying the Boolean Pythagorean Triples Problem "
        "via Cube-and-Conquer', SAT 2016, LNCS 9710, pp. 228-245).  That boundary is out "
        "of this lane's declared budget and is not attempted here; small-n instances are "
        "mechanism receipts, not new mathematics."
    ),
    "ramsey_edge_coloring": (
        "  Known answer for this family: R(3,3) = 6 (Greenwood & Gleason 1955).  Small "
        "instances are known-answer calibration controls, not new mathematics."
    ),
    "vdw_arithmetic_progression": (
        "  Known answer for this family: W(2,3) = 9 (Landman & Robertson 2004).  Small "
        "instances are known-answer calibration controls, not new mathematics."
    ),
}

#: DIMACS for the generic_cnf known-answer control: a one-variable contradiction.
GENERIC_CONTROL_DIMACS = "p cnf 1 2\n1 0\n-1 0\n"

#: The six known-answer controls sealed under runs/math/sat-lane/.
CONTROLS = (
    ("ramsey-r33-n5-sat", {"kind": "ramsey_edge_coloring", "n": 5, "k": 3}, DECISION_SAT),
    ("ramsey-r33-n6-unsat", {"kind": "ramsey_edge_coloring", "n": 6, "k": 3}, "UNSAT"),
    ("vdw-w23-n8-sat", {"kind": "vdw_arithmetic_progression", "n": 8, "k": 3}, DECISION_SAT),
    ("vdw-w23-n9-unsat", {"kind": "vdw_arithmetic_progression", "n": 9, "k": 3}, "UNSAT"),
    ("pythagorean-n20-sat", {"kind": "pythagorean_triple_coloring", "n": 20}, DECISION_SAT),
    ("generic-cnf-contradiction-unsat", {"kind": "generic_cnf"}, "UNSAT"),
)


class SatCertificateLaneError(ValueError):
    """Raised on malformed input, cap misconfiguration, integrity failure, or tamper."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SatCertificateLaneError(message)


def _plain_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SatCertificateLaneError(f"{label} must be a plain integer")
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


def _validate_caps(caps: Any) -> dict[str, int]:
    if not isinstance(caps, Mapping) or set(caps) != set(CAP_NAMES):
        raise SatCertificateLaneError(f"caps must declare exactly {sorted(CAP_NAMES)}")
    result: dict[str, int] = {}
    for name in CAP_NAMES:
        value = _plain_int(caps[name], f"caps.{name}")
        _require(value >= 0, f"caps.{name} must be nonnegative")
        _require(
            value <= SYSTEM_LIMITS[name],
            f"caps.{name} exceeds the system limit {SYSTEM_LIMITS[name]}",
        )
        result[name] = value
    return result


# ---------------------------------------------------------------------------
# Statement normalization (exact echo, deterministic derived text)
# ---------------------------------------------------------------------------


def _bounded_parameter(value: Any, label: str, minimum: int) -> int:
    result = _plain_int(value, label)
    _require(minimum <= result, f"{label} must be at least {minimum}")
    _require(
        result <= SYSTEM_LIMITS["max_statement_parameter"],
        f"{label} exceeds the statement parameter limit "
        f"{SYSTEM_LIMITS['max_statement_parameter']}",
    )
    return result


def _statement_text(norm: Mapping[str, Any]) -> str:
    kind = norm["kind"]
    if kind == "ramsey_edge_coloring":
        return (
            f"every 2-coloring of the edges of K_{norm['n']} contains a monochromatic "
            f"K_{norm['k']}"
        )
    if kind == "vdw_arithmetic_progression":
        return (
            f"every 2-coloring of {{1, ..., {norm['n']}}} contains a monochromatic "
            f"{norm['k']}-term arithmetic progression"
        )
    if kind == "pythagorean_triple_coloring":
        return (
            f"every 2-coloring of {{1, ..., {norm['n']}}} contains a monochromatic "
            f"Pythagorean triple a^2 + b^2 = c^2 with a < b < c"
        )
    return (
        f"the supplied DIMACS CNF with {norm['variables']} variables and "
        f"{norm['clauses']} clauses is satisfiable"
    )


def _parse_dimacs(text: Any) -> tuple[int, list[list[int]], str]:
    """Strict DIMACS parse: one header, exact clause count, literals within bounds."""

    _require(isinstance(text, str) and text.strip() != "", "dimacs_text must be nonempty")
    _require(
        len(text.encode("utf-8")) <= SYSTEM_LIMITS["max_dimacs_bytes"],
        f"dimacs_text exceeds {SYSTEM_LIMITS['max_dimacs_bytes']} bytes",
    )
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    header: tuple[int, int] | None = None
    clauses: list[list[int]] = []
    current: list[int] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            _require(header is None, "duplicate DIMACS header")
            parts = line.split()
            _require(
                len(parts) == 4 and parts[0] == "p" and parts[1] == "cnf",
                "DIMACS header must be 'p cnf <variables> <clauses>'",
            )
            try:
                declared_vars, declared_clauses = int(parts[2]), int(parts[3])
            except ValueError as error:
                raise SatCertificateLaneError("DIMACS header counts must be integers") from error
            _require(
                declared_vars >= 0 and declared_clauses >= 0,
                "DIMACS header counts must be nonnegative",
            )
            header = (declared_vars, declared_clauses)
            continue
        _require(header is not None, "DIMACS clause appears before the header")
        for token in line.split():
            try:
                literal = int(token)
            except ValueError as error:
                raise SatCertificateLaneError(f"non-integer DIMACS token: {token!r}") from error
            if literal == 0:
                clauses.append(current)
                current = []
            else:
                _require(
                    abs(literal) <= header[0],
                    f"DIMACS literal {literal} exceeds the declared variable count",
                )
                current.append(literal)
    _require(header is not None, "DIMACS input has no 'p cnf' header")
    _require(not current, "DIMACS input ends with an unterminated clause")
    _require(
        len(clauses) == header[1],
        f"DIMACS declares {header[1]} clauses but contains {len(clauses)}",
    )
    return header[0], clauses, normalized


def _normalize_statement(statement: Any, dimacs_text: str | None) -> dict[str, Any]:
    """Canonical statement echo.  ``generic_cnf`` derives its echo from the DIMACS text."""

    if not isinstance(statement, Mapping):
        raise SatCertificateLaneError("statement must be a mapping")
    kind = statement.get("kind")
    if kind not in STATEMENT_KINDS:
        raise SatCertificateLaneError(f"unknown statement kind: {kind!r}")
    norm: dict[str, Any] = {"kind": kind}
    if kind == "generic_cnf":
        _require(dimacs_text is not None, "generic_cnf requires dimacs_text")
        allowed = {"kind", "text", "variables", "clauses", "dimacs_sha256"}
        unknown = set(statement) - allowed
        _require(not unknown, f"unknown statement keys for {kind}: {sorted(unknown)}")
        variables, clause_list, normalized_text = _parse_dimacs(dimacs_text)
        norm["variables"] = variables
        norm["clauses"] = len(clause_list)
        norm["dimacs_sha256"] = _sha256_text(normalized_text)
        for name in ("variables", "clauses", "dimacs_sha256"):
            if name in statement:
                _require(
                    statement[name] == norm[name],
                    f"statement.{name} does not match the supplied DIMACS text",
                )
    else:
        _require(dimacs_text is None, "dimacs_text only applies to generic_cnf")
        takes_k = kind != "pythagorean_triple_coloring"
        allowed = {"kind", "n", "text"} | ({"k"} if takes_k else set())
        unknown = set(statement) - allowed
        _require(not unknown, f"unknown statement keys for {kind}: {sorted(unknown)}")
        minimum_n = 2 if kind == "ramsey_edge_coloring" else 1
        norm["n"] = _bounded_parameter(statement.get("n"), "statement.n", minimum_n)
        if takes_k:
            norm["k"] = _bounded_parameter(statement.get("k"), "statement.k", 2)
    text = _statement_text(norm)
    if "text" in statement:
        _require(statement["text"] == text, "statement text does not match its parameters")
    norm["text"] = text
    return norm


# ---------------------------------------------------------------------------
# Encoders (exact CNF construction; the variable map is documented in the receipt)
# ---------------------------------------------------------------------------


def _edge_var(i: int, j: int, n: int) -> int:
    """1-based rank of edge (i, j), 0 <= i < j < n, in lexicographic order."""

    return i * (2 * n - i - 1) // 2 + (j - i)


def _pythagorean_triples(n: int) -> list[tuple[int, int, int]]:
    """All triples a < b < c <= n with a^2 + b^2 = c^2, via Euclid's parametrization."""

    triples: list[tuple[int, int, int]] = []
    m = 2
    while m * m + 1 <= n:
        for j in range(1 + m % 2, m, 2):
            hypotenuse = m * m + j * j
            if hypotenuse > n:
                break
            if gcd(m, j) != 1:
                continue
            leg_a, leg_b = m * m - j * j, 2 * m * j
            small, large = min(leg_a, leg_b), max(leg_a, leg_b)
            multiple = 1
            while multiple * hypotenuse <= n:
                triples.append((small * multiple, large * multiple, hypotenuse * multiple))
                multiple += 1
        m += 1
    triples.sort()
    return triples


def _variable_map(norm: Mapping[str, Any]) -> dict[str, str]:
    kind = norm["kind"]
    if kind == "ramsey_edge_coloring":
        n, k = norm["n"], norm["k"]
        return {
            "scheme": "edge_lexicographic",
            "description": (
                f"variable e(i,j) for each edge {{i,j}} of K_{n} with 0 <= i < j <= {n - 1}: "
                f"index = i*(2*{n}-i-1)/2 + (j-i); true means the edge has color 1 of {{0,1}}. "
                f"For every {k}-vertex subset, one clause forbids all edges color 1 and one "
                "forbids all edges color 0, in that order, subsets in lexicographic order."
            ),
        }
    if kind == "vdw_arithmetic_progression":
        n, k = norm["n"], norm["k"]
        return {
            "scheme": "integer_identity",
            "description": (
                f"variable i for each integer i in {{1, ..., {n}}}; true means color 1 of "
                f"{{0,1}}. For every arithmetic progression a, a+d, ..., a+{k - 1}*d inside "
                "the interval (d ascending, then a ascending), one clause forbids all-color-1 "
                "and one forbids all-color-0, in that order."
            ),
        }
    if kind == "pythagorean_triple_coloring":
        n = norm["n"]
        return {
            "scheme": "integer_identity",
            "description": (
                f"variable i for each integer i in {{1, ..., {n}}}; true means color 1 of "
                "{0,1}. For every Pythagorean triple a < b < c with a^2 + b^2 = c^2 "
                "(ascending), one clause forbids all-color-1 and one forbids all-color-0, "
                "in that order."
            ),
        }
    return {
        "scheme": "dimacs_identity",
        "description": (
            "variables are exactly the DIMACS variables 1..V with polarities as supplied; "
            "the clause list is the DIMACS clause list in file order."
        ),
    }


def _size_plan(norm: Mapping[str, Any], caps: Mapping[str, int]) -> dict[str, Any]:
    """Predicted counts and the first tripped size cap, before any clause list is built.

    ``clauses`` is None only when the variable cap already tripped and counting the
    clauses would itself require the refused enumeration (Pythagorean triples).
    """

    kind = norm["kind"]
    if kind == "ramsey_edge_coloring":
        variables = comb(norm["n"], 2)
        clauses: int | None = 2 * comb(norm["n"], norm["k"])
    elif kind == "vdw_arithmetic_progression":
        n, k = norm["n"], norm["k"]
        variables = n
        progressions = (n - 1) // (k - 1)
        clauses = 2 * (progressions * n - (k - 1) * progressions * (progressions + 1) // 2)
    elif kind == "pythagorean_triple_coloring":
        variables = norm["n"]
        if variables > caps["max_vars"]:
            clauses = None
        else:
            clauses = 2 * len(_pythagorean_triples(norm["n"]))
    else:
        variables = norm["variables"]
        clauses = norm["clauses"]
    tripped: str | None = None
    if variables > caps["max_vars"]:
        tripped = "max_vars"
    elif clauses is not None and clauses > caps["max_clauses"]:
        tripped = "max_clauses"
    return {"variables": variables, "clauses": clauses, "tripped": tripped}


def _build_clauses(norm: Mapping[str, Any]) -> list[list[int]]:
    """Deterministic clause list for a registry statement (never called for generic_cnf)."""

    kind = norm["kind"]
    clauses: list[list[int]] = []
    if kind == "ramsey_edge_coloring":
        n, k = norm["n"], norm["k"]
        for subset in combinations(range(n), k):
            edges = [_edge_var(i, j, n) for i, j in combinations(subset, 2)]
            clauses.append([-v for v in edges])
            clauses.append(list(edges))
    elif kind == "vdw_arithmetic_progression":
        n, k = norm["n"], norm["k"]
        for d in range(1, (n - 1) // (k - 1) + 1):
            for a in range(1, n - (k - 1) * d + 1):
                progression = [a + t * d for t in range(k)]
                clauses.append([-i for i in progression])
                clauses.append(list(progression))
    elif kind == "pythagorean_triple_coloring":
        for a, b, c in _pythagorean_triples(norm["n"]):
            clauses.append([-a, -b, -c])
            clauses.append([a, b, c])
    else:
        raise SatCertificateLaneError("generic_cnf clauses come from the DIMACS text")
    return clauses


def _cnf_sha256(variables: int, clauses: list[list[int]]) -> str:
    return canonical_sha256({"clauses": clauses, "variables": variables})


def _render_dimacs(variables: int, clauses: list[list[int]]) -> str:
    lines = [f"p cnf {variables} {len(clauses)}"]
    lines.extend(" ".join(str(literal) for literal in clause) + " 0" for clause in clauses)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Independent model verification (pure Python; the SAT certificate check)
# ---------------------------------------------------------------------------


def verify_model(
    variables: int, clauses: list[list[int]], model: Any
) -> tuple[bool, dict[str, Any]]:
    """Check a canonical model against every clause, independently of any solver.

    The model must be exactly ``[±1, ±2, ..., ±variables]`` (position ``i`` decides
    variable ``i+1``).  Returns ``(ok, detail)`` where ``detail`` names the first
    violation: a malformed model or the index of the first unsatisfied clause.
    """

    if not isinstance(model, list) or len(model) != variables:
        return False, {"reason": "model_shape", "expected_length": variables}
    assignment: dict[int, bool] = {}
    for index, literal in enumerate(model):
        variable = index + 1
        if not isinstance(literal, int) or isinstance(literal, bool):
            return False, {"reason": "model_shape", "position": index}
        if literal not in (variable, -variable):
            return False, {"reason": "model_shape", "position": index}
        assignment[variable] = literal > 0
    for index, clause in enumerate(clauses):
        satisfied = False
        for literal in clause:
            variable = abs(literal)
            if variable not in assignment:
                return False, {"reason": "literal_out_of_range", "clause_index": index}
            if assignment[variable] == (literal > 0):
                satisfied = True
                break
        if not satisfied:
            return False, {
                "reason": "unsatisfied_clause",
                "clause_index": index,
                "clause": list(clause),
            }
    return True, {"reason": "verified", "clauses_checked": len(clauses)}


def _canonical_model(raw_model: Any, variables: int) -> list[int]:
    """Restrict/pad a solver model to variables 1..V; unassigned variables become false."""

    polarity: dict[int, bool] = {}
    for literal in raw_model or []:
        _require(
            isinstance(literal, int) and not isinstance(literal, bool) and literal != 0,
            "solver model contains a non-literal entry",
        )
        variable = abs(literal)
        _require(variable <= variables, "solver model names an undeclared variable")
        _require(variable not in polarity, "solver model assigns a variable twice")
        polarity[variable] = literal > 0
    return [
        variable if polarity.get(variable, False) else -variable
        for variable in range(1, variables + 1)
    ]


# ---------------------------------------------------------------------------
# Solving under the wall watchdog, and the DRAT verification attempt
# ---------------------------------------------------------------------------


def _now_ns(monotonic_ns: Callable[[], int] | None) -> int:
    value = (monotonic_ns or time.monotonic_ns)()
    if not isinstance(value, int) or isinstance(value, bool):
        raise SatCertificateLaneError("monotonic_ns probe must return an integer")
    return value


def _solver_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - stdlib since 3.8
        return "unknown"
    try:
        return version("python-sat")
    except PackageNotFoundError:  # pragma: no cover - installed under another name
        return "unknown"


def _solve(
    clauses: list[list[int]],
    caps: Mapping[str, int],
    monotonic_ns: Callable[[], int] | None,
    want_proof: bool,
) -> dict[str, Any]:
    """One solve under the wall watchdog.  ``answer`` None means the wall cap tripped."""

    try:
        from pysat.solvers import Solver
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise SatCertificateLaneError(
            "python-sat is required for the SAT certificate lane (pip install python-sat)"
        ) from error
    started = _now_ns(monotonic_ns)
    timer: threading.Timer | None = None
    solver = Solver(name=SOLVER_BACKEND, bootstrap_with=clauses, with_proof=want_proof)
    try:
        if monotonic_ns is None:
            timer = threading.Timer(caps["max_seconds"] + 1, solver.interrupt)
            timer.daemon = True
            timer.start()
        answer = solver.solve_limited(expect_interrupt=True)
        raw_model = solver.get_model() if answer else None
        proof = solver.get_proof() if want_proof and answer is False else None
    finally:
        if timer is not None:
            timer.cancel()
        solver.delete()
    elapsed_ns = _now_ns(monotonic_ns) - started
    _require(elapsed_ns >= 0, "monotonic time went backwards during the solve")
    wall_seconds = elapsed_ns // 1_000_000_000
    if answer is None or wall_seconds > caps["max_seconds"]:
        # No work after a trip: an answer computed beyond the budget is discarded.
        return {"answer": None, "raw_model": None, "proof": None}
    return {"answer": answer, "raw_model": raw_model, "proof": proof}


def _run_drat_trim(binary: str, dimacs_text: str, proof_text: str, timeout: int) -> bool:
    with tempfile.TemporaryDirectory(prefix="sat-lane-drat-") as scratch:
        cnf_path = Path(scratch) / "instance.cnf"
        proof_path = Path(scratch) / "proof.drat"
        cnf_path.write_text(dimacs_text, encoding="utf-8")
        proof_path.write_text(proof_text, encoding="utf-8")
        try:
            completed = subprocess.run(
                [binary, str(cnf_path), str(proof_path)],
                capture_output=True,
                text=True,
                timeout=max(timeout, 1),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
    return "s VERIFIED" in completed.stdout


def _attempt_drat(
    variables: int,
    clauses: list[list[int]],
    proof: Any,
    drat_binary: str | None,
    drat_verifier: Callable[[str, list[str]], bool] | None,
    timeout: int,
) -> dict[str, Any]:
    """The UNSAT verification attempt.  Only a real 's VERIFIED' upgrades the label.

    Callers reach this only when a verification pathway exists and proof extraction is
    usable on this platform, so ``used`` is true; ``verified`` stays false unless a
    nonempty extracted proof actually passes the check.
    """

    lines = [str(line).strip() for line in (proof or [])]
    block = {
        "available": True,
        "proof_extraction_usable": True,
        "proof_lines": len(lines),
        "proof_sha256": None,
        "used": True,
        "verified": False,
    }
    if not lines:
        # An empty extracted proof is never submitted for verification.
        return block
    proof_text = "\n".join(lines) + "\n"
    block["proof_sha256"] = _sha256_text(proof_text)
    dimacs_text = _render_dimacs(variables, clauses)
    if drat_verifier is not None:
        block["verified"] = bool(drat_verifier(dimacs_text, lines))
    else:
        block["verified"] = _run_drat_trim(drat_binary, dimacs_text, proof_text, timeout)
    return block


def _inactive_drat(available: bool) -> dict[str, Any]:
    return {
        "available": available,
        "proof_extraction_usable": PROOF_EXTRACTION_USABLE,
        "proof_lines": 0,
        "proof_sha256": None,
        "used": False,
        "verified": False,
    }


# ---------------------------------------------------------------------------
# Claims, interpretation, and receipt assembly
# ---------------------------------------------------------------------------


def _build_claims(decision: str) -> dict[str, bool]:
    return {
        "decision_covers_declared_finite_instance_only": True,
        "model_independently_verified": decision == DECISION_SAT,
        "new_mathematics_established": False,
        "scalar_truth_or_probability_score": False,
        "unsat_independently_verified": decision == DECISION_UNSAT_DRAT,
        "unsat_solver_asserted_only": decision == DECISION_UNSAT_ASSERTED,
    }


def _interpretation(norm: Mapping[str, Any], decision: str) -> dict[str, Any]:
    kind = norm["kind"]
    if kind == "generic_cnf":
        sat_means = (
            "the supplied CNF has a satisfying assignment; the sealed model is that assignment"
        )
        unsat_means = "the supplied CNF has no satisfying assignment"
    elif kind == "ramsey_edge_coloring":
        n, k = norm["n"], norm["k"]
        sat_means = (
            f"a 2-coloring of the edges of K_{n} with no monochromatic K_{k} exists; the "
            f"sealed model is that coloring, so the universal statement is FALSE at "
            f"n={n}, k={k}"
        )
        unsat_means = (
            f"no such coloring exists, so the universal statement is TRUE at n={n}, k={k}"
        )
    elif kind == "vdw_arithmetic_progression":
        n, k = norm["n"], norm["k"]
        sat_means = (
            f"a 2-coloring of {{1, ..., {n}}} with no monochromatic {k}-term arithmetic "
            f"progression exists; the sealed model is that coloring, so the universal "
            f"statement is FALSE at n={n}, k={k}"
        )
        unsat_means = (
            f"no such coloring exists, so the universal statement is TRUE at n={n}, k={k}"
        )
    else:
        n = norm["n"]
        sat_means = (
            f"a 2-coloring of {{1, ..., {n}}} with no monochromatic Pythagorean triple "
            f"exists; the sealed model is that coloring, so the universal statement is "
            f"FALSE at n={n}"
        )
        unsat_means = f"no such coloring exists, so the universal statement is TRUE at n={n}"
    if decision == DECISION_SAT:
        conclusion = (
            f"decision {DECISION_SAT}: {sat_means}. The model was re-verified against "
            "every clause in pure Python, independently of the solver."
        )
    elif decision == DECISION_UNSAT_ASSERTED:
        conclusion = (
            f"decision {DECISION_UNSAT_ASSERTED}: {unsat_means}. This refutation is "
            f"asserted by {SOLVER_BACKEND} and was NOT independently verified (no DRAT "
            "proof was checked)."
        )
    elif decision == DECISION_UNSAT_DRAT:
        conclusion = (
            f"decision {DECISION_UNSAT_DRAT}: {unsat_means}. The DRAT refutation proof "
            "was independently checked by drat-trim."
        )
    else:
        conclusion = None
    return {
        "cnf_satisfiable_means": sat_means,
        "cnf_unsatisfiable_means": unsat_means,
        "conclusion": conclusion,
    }


def _scope(kind: str) -> str:
    return _SCOPE_BASE + _SCOPE_KIND_NOTES[kind]


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": canonical_sha256(body)}


def _assemble(
    norm: Mapping[str, Any],
    caps: Mapping[str, int],
    decision: str,
    encoding: Mapping[str, Any],
    model: list[int] | None,
    drat: Mapping[str, Any],
    probes: Mapping[str, bool],
) -> dict[str, Any]:
    body = {
        "caps": dict(caps),
        "claims": _build_claims(decision),
        "decision": decision,
        "drat": dict(drat),
        "encoder_version": ENCODER_VERSION,
        "encoding": dict(encoding),
        "interpretation": _interpretation(norm, decision),
        "literature": LITERATURE.get(norm["kind"]),
        "model": model,
        "probes": dict(probes),
        "schema_version": RESULT_SCHEMA,
        "scope": _scope(norm["kind"]),
        "solver": {
            "backend": SOLVER_BACKEND,
            "interface": SOLVER_INTERFACE,
            "version": _solver_version(),
        },
        "statement": dict(norm),
    }
    return _seal(body)


# ---------------------------------------------------------------------------
# The lane driver
# ---------------------------------------------------------------------------


def decide(
    statement: Mapping[str, Any],
    *,
    caps: Mapping[str, Any] | None = None,
    dimacs_text: str | None = None,
    monotonic_ns: Callable[[], int] | None = None,
    drat_verifier: Callable[[str, list[str]], bool] | None = None,
) -> dict[str, Any]:
    """Decide one declared statement under caps and return its sealed receipt.

    ``monotonic_ns`` and ``drat_verifier`` are test-only probe overrides; using either is
    named in the receipt's ``probes`` block, exactly like the A4 watchdog's overrides.
    """

    validated_caps = _validate_caps(DEFAULT_CAPS if caps is None else caps)
    norm = _normalize_statement(statement, dimacs_text)
    probes = {
        "drat_verifier_overridden": drat_verifier is not None,
        "monotonic_ns_overridden": monotonic_ns is not None,
    }
    if drat_verifier is not None:
        drat_binary: str | None = None
        drat_available = True
    else:
        drat_binary = shutil.which(DRAT_TRIM_BINARY)
        drat_available = drat_binary is not None

    plan = _size_plan(norm, validated_caps)
    variable_map = _variable_map(norm)
    if plan["tripped"] is not None:
        encoding = {
            "clauses": plan["clauses"],
            "cnf_sha256": None,
            "variable_map": variable_map,
            "variables": plan["variables"],
        }
        receipt = _assemble(
            norm,
            validated_caps,
            f"CAP_TRIPPED:{plan['tripped']}",
            encoding,
            None,
            _inactive_drat(drat_available),
            probes,
        )
        validate_receipt(receipt, dimacs_text=dimacs_text)
        return receipt

    if norm["kind"] == "generic_cnf":
        variables, clauses, _ = _parse_dimacs(dimacs_text)
    else:
        variables = plan["variables"]
        clauses = _build_clauses(norm)
    _require(
        variables == plan["variables"] and len(clauses) == plan["clauses"],
        "encoder produced counts that disagree with the size plan",
    )
    encoding = {
        "clauses": len(clauses),
        "cnf_sha256": _cnf_sha256(variables, clauses),
        "variable_map": variable_map,
        "variables": variables,
    }

    want_proof = drat_available and PROOF_EXTRACTION_USABLE
    outcome = _solve(clauses, validated_caps, monotonic_ns, want_proof=want_proof)
    if outcome["answer"] is None:
        receipt = _assemble(
            norm,
            validated_caps,
            "CAP_TRIPPED:max_seconds",
            encoding,
            None,
            _inactive_drat(drat_available),
            probes,
        )
        validate_receipt(receipt, dimacs_text=dimacs_text)
        return receipt

    if outcome["answer"]:
        model = _canonical_model(outcome["raw_model"], variables)
        ok, detail = verify_model(variables, clauses, model)
        if not ok:
            raise SatCertificateLaneError(
                "integrity failure: the solver returned a model that fails independent "
                f"verification: {detail}"
            )
        receipt = _assemble(
            norm,
            validated_caps,
            DECISION_SAT,
            encoding,
            model,
            _inactive_drat(drat_available),
            probes,
        )
        validate_receipt(receipt, dimacs_text=dimacs_text)
        return receipt

    if want_proof:
        drat = _attempt_drat(
            variables,
            clauses,
            outcome["proof"],
            drat_binary,
            drat_verifier,
            validated_caps["max_seconds"],
        )
    else:
        drat = _inactive_drat(drat_available)
    decision = DECISION_UNSAT_DRAT if drat["verified"] else DECISION_UNSAT_ASSERTED
    receipt = _assemble(norm, validated_caps, decision, encoding, None, drat, probes)
    validate_receipt(receipt, dimacs_text=dimacs_text)
    return receipt


# ---------------------------------------------------------------------------
# Receipt validation (seal replay + full semantic re-verification)
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEYS = {
    "caps",
    "claims",
    "content_sha256",
    "decision",
    "drat",
    "encoder_version",
    "encoding",
    "interpretation",
    "literature",
    "model",
    "probes",
    "schema_version",
    "scope",
    "solver",
    "statement",
}

_DRAT_KEYS = {
    "available",
    "proof_extraction_usable",
    "proof_lines",
    "proof_sha256",
    "used",
    "verified",
}


def _validate_drat_block(value: Any, decision: str) -> None:
    """Internal coherence only: availability and usability are facts of the sealing run's
    environment, so they are type-checked and cross-checked against the decision, never
    recomputed against the validating machine."""

    if not isinstance(value, Mapping) or set(value) != _DRAT_KEYS:
        raise SatCertificateLaneError("drat block keys changed")
    for name in ("available", "proof_extraction_usable", "used", "verified"):
        _require(isinstance(value[name], bool), f"drat.{name} must be a boolean")
    proof_lines = _plain_int(value["proof_lines"], "drat.proof_lines")
    _require(proof_lines >= 0, "drat.proof_lines must be nonnegative")
    sha = value["proof_sha256"]
    if sha is not None:
        _require(
            isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
            "drat.proof_sha256 must be null or a lowercase SHA-256 digest",
        )
    if value["used"]:
        _require(
            value["available"] and value["proof_extraction_usable"],
            "drat.used requires an available pathway and usable proof extraction",
        )
    if value["verified"]:
        _require(value["used"], "drat.verified requires drat.used")
    if decision == DECISION_UNSAT_DRAT:
        _require(value["verified"], "UNSAT_DRAT_VERIFIED requires drat.verified")
        _require(proof_lines >= 1, "UNSAT_DRAT_VERIFIED requires a nonempty proof")
        _require(sha is not None, "UNSAT_DRAT_VERIFIED requires proof_sha256")
    elif decision == DECISION_UNSAT_ASSERTED:
        _require(value["verified"] is False, "UNSAT_SOLVER_ASSERTED forbids drat.verified")
    else:
        _require(
            value["used"] is False and value["verified"] is False,
            "drat block must be inactive unless the decision is UNSAT",
        )
        _require(
            proof_lines == 0 and sha is None,
            "drat block must carry no proof unless the decision is UNSAT",
        )


def validate_receipt(value: Any, *, dimacs_text: str | None = None) -> None:
    """Reject any structural, echo, claims, certificate, or seal violation.

    Registry statements are re-encoded and ``generic_cnf`` statements re-parsed (the
    original DIMACS text is required for them), so a SAT model is re-verified against the
    real clauses and an UNSAT decision is re-solved as a direction tamper check.  The
    re-solve is itself a solver assertion: it can expose a flipped decision, but it never
    upgrades a solver-asserted refutation to a verified one.
    """

    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_KEYS:
        raise SatCertificateLaneError("receipt top-level keys changed")
    _require(value["schema_version"] == RESULT_SCHEMA, "receipt schema changed")
    _require(value["encoder_version"] == ENCODER_VERSION, "receipt encoder version changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    _require(value["content_sha256"] == canonical_sha256(body), "receipt seal changed")

    caps = _validate_caps(value["caps"])
    statement = value["statement"]
    if not isinstance(statement, Mapping):
        raise SatCertificateLaneError("statement echo must be a mapping")
    kind = statement.get("kind")
    if kind == "generic_cnf":
        _require(
            dimacs_text is not None,
            "generic_cnf receipts can only be validated against the original DIMACS text",
        )
    norm = _normalize_statement(statement, dimacs_text if kind == "generic_cnf" else None)
    _require(dict(statement) == norm, "statement echo is not canonical")

    decision = value["decision"]
    _require(decision in _DECISIONS, f"unknown decision: {decision!r}")
    _require(value["claims"] == _build_claims(decision), "claims do not match the decision")
    _require(value["scope"] == _scope(norm["kind"]), "scope does not match the statement kind")
    _require(
        value["literature"] == LITERATURE.get(norm["kind"]),
        "literature block does not match the statement kind",
    )
    _require(
        value["interpretation"] == _interpretation(norm, decision),
        "interpretation block does not match the statement and decision",
    )

    solver = value["solver"]
    if not isinstance(solver, Mapping) or set(solver) != {"backend", "interface", "version"}:
        raise SatCertificateLaneError("solver block keys changed")
    _require(solver["backend"] == SOLVER_BACKEND, "solver backend changed")
    _require(solver["interface"] == SOLVER_INTERFACE, "solver interface changed")
    _require(
        isinstance(solver["version"], str) and solver["version"].strip() != "",
        "solver version must be a nonempty string",
    )
    probes = value["probes"]
    if not isinstance(probes, Mapping) or set(probes) != {
        "drat_verifier_overridden",
        "monotonic_ns_overridden",
    }:
        raise SatCertificateLaneError("probes block keys changed")
    for name in sorted(probes):
        _require(isinstance(probes[name], bool), f"probes.{name} must be a boolean")
    _validate_drat_block(value["drat"], decision)

    encoding = value["encoding"]
    expected_encoding_keys = {"clauses", "cnf_sha256", "variable_map", "variables"}
    if not isinstance(encoding, Mapping) or set(encoding) != expected_encoding_keys:
        raise SatCertificateLaneError("encoding block keys changed")
    _require(
        encoding["variable_map"] == _variable_map(norm),
        "encoding variable_map does not match the statement",
    )

    plan = _size_plan(norm, caps)
    if decision in ("CAP_TRIPPED:max_vars", "CAP_TRIPPED:max_clauses"):
        _require(
            plan["tripped"] == decision.split(":", 1)[1],
            "size-cap decision does not match the recomputed size plan",
        )
        _require(encoding["cnf_sha256"] is None, "a size-cap trip must not seal a CNF hash")
        _require(
            encoding["variables"] == plan["variables"] and encoding["clauses"] == plan["clauses"],
            "size-cap trip encoding counts do not match the recomputed size plan",
        )
        _require(value["model"] is None, "a cap-tripped receipt must not carry a model")
        return
    _require(plan["tripped"] is None, "decision ignores a tripped size cap")

    if norm["kind"] == "generic_cnf":
        variables, clauses, _ = _parse_dimacs(dimacs_text)
    else:
        variables = plan["variables"]
        clauses = _build_clauses(norm)
    _require(
        encoding["variables"] == variables and encoding["clauses"] == len(clauses),
        "encoding counts do not match the re-encoded statement",
    )
    _require(
        encoding["cnf_sha256"] == _cnf_sha256(variables, clauses),
        "encoding cnf_sha256 does not match the re-encoded statement",
    )

    if decision == "CAP_TRIPPED:max_seconds":
        _require(value["model"] is None, "a cap-tripped receipt must not carry a model")
        return
    if decision == DECISION_SAT:
        ok, detail = verify_model(variables, clauses, value["model"])
        if not ok:
            raise SatCertificateLaneError(
                f"SAT receipt model fails independent verification: {detail}"
            )
        return
    # UNSAT: the model must be absent and the decision direction must survive a re-solve.
    _require(value["model"] is None, "an UNSAT receipt must not carry a model")
    check = _solve(clauses, caps, None, want_proof=False)
    _require(
        check["answer"] is False,
        "UNSAT receipt failed the re-solve direction check (the instance is satisfiable "
        "or the wall cap tripped during validation)",
    )


# ---------------------------------------------------------------------------
# Machine-form hook for the discovery scheduler
# ---------------------------------------------------------------------------


def statement_from_machine_form(machine_form: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an A2 ``bounded_combinatorial_coloring`` machine form to a lane statement.

    ``statement_kind`` must name a registry statement (``generic_cnf`` is not routable
    through the queue: it needs the DIMACS text).  ``k`` must be 0 exactly when the
    statement kind takes no ``k``.
    """

    if not isinstance(machine_form, Mapping):
        raise SatCertificateLaneError("machine_form must be a mapping")
    _require(
        machine_form.get("kind") == MACHINE_FORM_KIND,
        f"machine_form.kind must be {MACHINE_FORM_KIND!r}",
    )
    _require(
        set(machine_form) == {"kind", "statement_kind", "n", "k"},
        "machine_form keys changed for bounded_combinatorial_coloring",
    )
    statement_kind = machine_form["statement_kind"]
    _require(
        statement_kind in MACHINE_FORM_STATEMENT_KINDS,
        f"statement_kind must be one of {sorted(MACHINE_FORM_STATEMENT_KINDS)}",
    )
    n = _plain_int(machine_form["n"], "machine_form.n")
    k = _plain_int(machine_form["k"], "machine_form.k")
    if statement_kind == "pythagorean_triple_coloring":
        _require(k == 0, "k must be 0 for pythagorean_triple_coloring")
        return {"kind": statement_kind, "n": n}
    return {"kind": statement_kind, "n": n, "k": k}


# ---------------------------------------------------------------------------
# Known-answer controls and CLI
# ---------------------------------------------------------------------------


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise SatCertificateLaneError(f"refusing to overwrite immutable receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def run_controls(output_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Run the six known-answer controls and seal their receipts under ``output_dir``.

    A control landing on the wrong side of its known answer is an error, never a receipt:
    these instances calibrate the lane, so a surprising outcome means the lane is broken.
    """

    results: list[tuple[str, dict[str, Any]]] = []
    for name, statement, expected in CONTROLS:
        text = GENERIC_CONTROL_DIMACS if statement["kind"] == "generic_cnf" else None
        receipt = decide(statement, dimacs_text=text)
        decision = receipt["decision"]
        if expected == "UNSAT":
            _require(
                decision in (DECISION_UNSAT_ASSERTED, DECISION_UNSAT_DRAT),
                f"known-answer control {name} expected UNSAT but got {decision}",
            )
        else:
            _require(
                decision == expected,
                f"known-answer control {name} expected {expected} but got {decision}",
            )
        if text is not None:
            cnf_path = output_dir / f"{name}.cnf"
            if cnf_path.exists():
                _require(
                    cnf_path.read_text(encoding="utf-8") == text,
                    f"refusing to overwrite immutable control input: {cnf_path}",
                )
            else:
                cnf_path.parent.mkdir(parents=True, exist_ok=True)
                cnf_path.write_text(text, encoding="utf-8", newline="\n")
        _write_immutable(output_dir / f"{name}.json", receipt)
        results.append((name, receipt))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAT certificate lane (M10).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--statement", help="path to a JSON statement file")
    group.add_argument(
        "--controls",
        action="store_true",
        help="run the six known-answer controls and seal their receipts",
    )
    group.add_argument(
        "--validate",
        action="store_true",
        help="validate an existing receipt (requires --output; --dimacs for generic_cnf)",
    )
    parser.add_argument("--dimacs", help="path to a DIMACS file (generic_cnf only)")
    parser.add_argument("--output", help="path for one sealed receipt")
    parser.add_argument(
        "--output-dir",
        default="runs/math/sat-lane",
        help="directory for control receipts (default: runs/math/sat-lane)",
    )
    args = parser.parse_args(argv)
    if args.controls:
        for name, receipt in run_controls(Path(args.output_dir)):
            print(f"{name}: {receipt['decision']} content_sha256={receipt['content_sha256']}")
        return 0
    dimacs_text = None
    if args.dimacs:
        dimacs_text = Path(args.dimacs).read_text(encoding="utf-8")
    if args.validate:
        if not args.output:
            parser.error("--validate requires --output")
        receipt = json.loads(Path(args.output).read_text(encoding="utf-8"))
        validate_receipt(receipt, dimacs_text=dimacs_text)
        print(f"valid: {receipt['decision']} content_sha256={receipt['content_sha256']}")
        return 0
    statement = json.loads(Path(args.statement).read_text(encoding="utf-8"))
    receipt = decide(statement, dimacs_text=dimacs_text)
    if args.output:
        _write_immutable(Path(args.output), receipt)
    print(f"{receipt['decision']} content_sha256={receipt['content_sha256']}")
    return 0


__all__ = [
    "CONTROLS",
    "DECISION_SAT",
    "DECISION_UNSAT_ASSERTED",
    "DECISION_UNSAT_DRAT",
    "DEFAULT_CAPS",
    "ENCODER_VERSION",
    "GENERIC_CONTROL_DIMACS",
    "LITERATURE",
    "MACHINE_FORM_KIND",
    "PROOF_EXTRACTION_USABLE",
    "RESULT_SCHEMA",
    "SOLVER_BACKEND",
    "STATEMENT_KINDS",
    "SYSTEM_LIMITS",
    "SatCertificateLaneError",
    "decide",
    "main",
    "run_controls",
    "statement_from_machine_form",
    "validate_receipt",
    "verify_model",
]

if __name__ == "__main__":
    raise SystemExit(main())
