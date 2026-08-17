"""Head-to-head against a real discovery: Balmer 1885 (empirical) and Bohr 1913 (derived).

Two races are run against the historical record, and the receipt publishes both side by
side with what each side actually had in hand.

**Race 1, the blind empirical race (what Balmer did).**  The engine is handed four
anonymized rows ``(m, v)`` with ``m = 1..4``.  The public config carries no name, no
unit, and no word from the subject matter: the forbidden-vocabulary runtime guard of
:mod:`sigma_theory_compiler.blind_planetary_law_rediscovery_campaign` is reused and its
list extended, and the check runs at build time, not only in the test suite.  The true
ordinal is *withheld*: the physically meaningful index is ``m + 2``, and the engine has
to recover that offset for itself rather than being handed rows already labelled 3..6.

The engine runs the declared ladder (B1 basis synthesis on the raw rows, B7 structural
repair, B3 conjecture generation) and then its own declared transformation lane: a
bounded derived-view search over the finite grammar

``z = v * (m + s)^i * ((m + s)^2 - c)^j``

for ``s`` in ``[0, 3]``, ``i`` and ``j`` in ``[-4, 4]`` and ``c`` in ``[0, 9]``, with
``c`` pinned to zero when ``j`` is zero.  That is 2916 declared views, every one of them
evaluated, B1-checked in exact rational arithmetic, and logged with the number that
decided it.  A view is admitted when its derived column is constant to within a
*declared relative tolerance*, because these rows are transcribed measurements and not
computed values: exact equality is the wrong test for a nineteenth-century table, and
saying so out loud is cheaper than pretending otherwise.  The first admitted view in
declared Occam order is frozen as the candidate, and nothing later is considered.

The three further rows are pure holdout, and their *values are sealed*: the public
config carries only their indices.  The engine therefore predicts numbers it provably
cannot have seen, which is exactly the shape of the historical event -- Balmer computed
further members of the series from his formula before they were confirmed.

**Race 2, the derivation race (what Bohr did).**  With no data at all, from two declared
postulates -- Coulomb attraction with quantized angular momentum ``L = n*hbar``, and
emission of a single quantum ``h*nu`` on transition -- sympy derives the orbital radius,
the energy levels, the transition relation, and finally the Rydberg constant
``R = 2*pi^2*m_e*e^4*k^2/(h^3*c)`` in closed form.  Only then are CODATA constants
substituted, and the number compared with the measured Rydberg constant.  The loop is
closed by showing that Balmer's empirical ``B`` is ``4/R``, and evaluating that against
Balmer's own 3645.6.

Honesty rules this module obeys.

1. Every symbolic step is recomputed on each run and re-derived independently in the
   tests; nothing is a stored transcript.
2. The blind phase cannot read the sealed fixture: the read guard denies every owned
   file-read surface and one instrumented denied probe is recorded as evidence.  The
   fixture opens once, atomically, after the candidate and the holdout predictions are
   frozen.
3. The tolerances are declared in the public config before the run, and the receipt
   reports a robustness ladder showing what the admitted set would have been at other
   tolerances -- so "the tolerance was tuned" is a checkable accusation, not a shrug.
4. Historical timescales are *cited or declared undocumented*.  No duration of anyone's
   personal effort is estimated, because the primary record does not document it.
5. The engine did not invent the view grammar or the postulates.  Both are human
   declarations; what is measured here is exhaustive, receipted selection inside them.
6. Nothing is novel.  Both results are settled nineteenth- and twentieth-century
   physics, reproduced as an engine-capability demonstration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

from .basis_synthesis import synthesize_basis
from .blind_planetary_law_rediscovery_campaign import (
    FORBIDDEN_VOCABULARY as PLANETARY_FORBIDDEN_VOCABULARY,
)
from .blind_planetary_law_rediscovery_campaign import (
    _decimal_fraction,
    _file_sha256,
    _fraction,
    _fraction_data,
    _hex_digest,
    _load_json,
    _resolve,
    _round_to_places,
    _SealedTargetsGuard,
)
from .conjecture_generation import STATEMENT_KINDS, generate_conjectures
from .nonlinear_coefficient_search import search_nonlinear
from .sigma_core import canonical_json_bytes, canonical_sha256
from .structural_repair import repair_structure

CONFIG_SCHEMA = "invariant-blind-indexed-value-config-1.0"
TARGET_SCHEMA = "invariant-blind-indexed-value-targets-1.0"
RESULT_SCHEMA = "invariant-balmer-bohr-case-study-result-1.0"
RUNTIME_SCHEMA = "invariant-balmer-bohr-case-study-runtime-1.0"
CASE_STUDY_ID = "balmer-bohr-v1"

CONFIG_PATH = "configs/backgrounds/blind_indexed_value_rows_v1.json"
TARGETS_PATH = "configs/backgrounds/blind_indexed_value_targets_v1.json"
SOURCE_PATH = "src/sigma_theory_compiler/balmer_bohr_case_study.py"
TEST_PATH = "tests/test_balmer_bohr_case_study.py"
DOC_PATH = "docs/BALMER_BOHR_CASE_STUDY.md"
RECEIPT_PATH = "runs/math/case-studies/balmer-bohr-v1.json"
RUNTIME_PATH = "runs/math/case-studies/balmer-bohr-v1-runtime.json"

#: Bounds of the declared derived-view grammar.  These four numbers *are* the search
#: space: changing one changes what the engine could possibly have found, so a test
#: pins every one of them together with the resulting view count.
SHIFT_BOUND = (0, 3)
INDEX_EXPONENT_BOUND = (-4, 4)
QUADRATIC_EXPONENT_BOUND = (-4, 4)
OFFSET_BOUND = (0, 9)

#: Decimal places used for every emitted rational-derived decimal string.
SPREAD_DECIMAL_PLACES = 12
RESIDUAL_DECIMAL_PLACES = 6

#: Declared tolerance ladder reported alongside the run.  The admitted set is recomputed
#: at each rung so that the sensitivity of the verdict to the declared tolerance is a
#: published number rather than a promise.
TOLERANCE_ROBUSTNESS_LADDER = ("1e-6", "1e-5", "1e-4", "3e-4", "1e-3", "1e-2", "1e-1")

#: mpmath working precision, far above anything emitted.
WORKING_PRECISION_DIGITS = 60
EMITTED_PRECISION_DIGITS = 14

CANDIDATE_SELECTION_RULE = (
    "the first derived view whose column is constant to within the declared fit tolerance, "
    "in the declared ordering (total weight abs(i) + 2*abs(j), then s, then c, then i, then "
    "j); every index-dependent statement -- a B1 raw closed form, a B7 repair, or a B3 "
    "conjecture over the raw rows -- is recorded in the stage trail and never selected, "
    "because the config declares the row label arbitrary; nothing after the first admitted "
    "view is considered, and nothing is generated after the unseal"
)

VERDICT_RULE = (
    "REDISCOVERED_EXACT requires all three of: the frozen view exponents (s, i, j, c) equal "
    "the sealed ones exactly; the engine's constant, rounded in exact rational arithmetic to "
    "the precision at which the classical constant was published, equals the sealed constant "
    "exactly; and every sealed holdout row is predicted within the declared holdout "
    "tolerance. PARTIAL is an exact structure match that fails one of the other two. MISSED "
    "is anything else, including no admitted view at all."
)

EXACT_ARITHMETIC_NOTE = (
    "B1 basis synthesis is run in exact rational arithmetic on every one of the declared "
    "views, and it refuses all of them -- including the winning one -- because measured rows "
    "are never exactly constant. The tolerance-aware spread test is the entire concession "
    "this case study makes to real measurement, it is declared in the public config before "
    "the run, and its sensitivity is published as a robustness ladder."
)

POLICIES = {
    "candidate_generation_after_unseal": 0,
    "holdout_reads_before_candidate_freeze": 0,
    "minimum_rediscoveries_for_pass": 1,
    "target_reads_before_candidate_freeze": 0,
    "target_unseal_batches": 1,
}

STATIC_CLAIMS = {
    "blinding_enforced_by_runtime_guard": True,
    "derivation_is_symbolic_and_recomputed": True,
    "grammar_and_postulates_are_human_declarations": True,
    "historical_timescales_cited_not_estimated": True,
    "holdout_values_sealed_until_after_prediction": True,
    "novelty_claimed": False,
    "post_unseal_generation": False,
    "real_observational_data_opened": False,
    "rediscovery_of_classical_results": True,
    "target_records_read_before_candidate_freeze": 0,
}

SCOPE = (
    "A head-to-head against one real discovery, run twice. The empirical race gives the "
    "engine the four anonymized numbers Balmer had, with the true ordinal withheld, and asks "
    "it to find the relation and then predict three rows whose values are sealed until after "
    "the prediction is frozen. The derivation race gives it two declared postulates and no "
    "data, and asks it to reach the Rydberg constant symbolically. Both halves rediscover "
    "settled physics: no novelty, priority, or empirical significance is asserted anywhere. "
    "The view grammar and the postulates are human declarations, so a MISSED would mean "
    "'outside the declared grammar', never 'impossible', and a hit means exhaustive receipted "
    "selection inside a declared space, not invention of the space. The transcribed "
    "historical values are published table entries, not a dataset opened at run time, and the "
    "measured wall-clock reported in the head-to-head is a one-time host measurement kept "
    "outside every sealed hash."
)

#: Subject-matter vocabulary that must never appear in the public config.  The planetary
#: campaign's list is reused wholesale and extended; a test pins the containment.
EXTENDED_FORBIDDEN_VOCABULARY = (
    "absorption",
    "angstroem",
    "angstrom",
    "angstroms",
    "atom",
    "atomic",
    "atoms",
    "balmer",
    "bohr",
    "colour",
    "coulomb",
    "diffraction",
    "electron",
    "electrons",
    "emission",
    "emitted",
    "frequency",
    "grating",
    "hydrogen",
    "ionization",
    "ionisation",
    "light",
    "line",
    "lines",
    "molecular",
    "molecule",
    "nanometer",
    "nanometre",
    "nm",
    "nucleus",
    "optical",
    "optics",
    "photon",
    "photons",
    "planck",
    "prism",
    "proton",
    "quanta",
    "quantised",
    "quantized",
    "quantum",
    "radiation",
    "rydberg",
    "series",
    "spectra",
    "spectral",
    "spectrograph",
    "spectroscope",
    "spectroscopy",
    "spectrum",
    "transition",
    "transitions",
    "violet",
    "visible",
    "wave",
    "wavelength",
    "wavelengths",
    "waves",
)

FORBIDDEN_VOCABULARY = tuple(
    sorted(set(PLANETARY_FORBIDDEN_VOCABULARY) | set(EXTENDED_FORBIDDEN_VOCABULARY))
)


class BalmerBohrCaseStudyError(ValueError):
    """Raised on config drift, sealed-chronology violation, or receipt tamper."""


class _CaseStudySealedGuard(_SealedTargetsGuard):
    """The planetary campaign's read guard, pointed at this study's sealed fixture.

    The enforcement surfaces, the denial semantics, the counters and the certificate
    shape are inherited unchanged; only the guarded path differs.  Subclassing rather
    than editing keeps the other campaign's source hash -- and therefore its committed
    receipt -- byte-identical.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._target = _resolve(root, TARGETS_PATH)

    def certificate(self) -> dict[str, Any]:
        certificate = super().certificate()
        if certificate["denied_paths"]:
            certificate["denied_paths"] = [TARGETS_PATH]
        return certificate


# ---------------------------------------------------------------------------
# Exact helpers.  No float ever reaches a receipt.
# ---------------------------------------------------------------------------


def _no_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise BalmerBohrCaseStudyError(f"floating value forbidden at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _no_floats(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _no_floats(child, f"{path}[{index}]")


def _decimal_text(value: Fraction, places: int) -> str:
    """Fixed-point decimal string of an exact rational, round-half-even, no float path."""

    rounded = _round_to_places(value, places)
    scaled = rounded * 10**places
    integer = scaled.numerator // scaled.denominator
    sign = "-" if integer < 0 else ""
    digits = str(abs(integer)).rjust(places + 1, "0")
    if places == 0:
        return f"{sign}{digits}"
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _mp_text(value: Any, digits: int = EMITTED_PRECISION_DIGITS) -> str:
    return mp.nstr(mp.mpf(value), digits, strip_zeros=False)


def _text(expression: Any) -> str:
    return str(expression)


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    _no_floats(body)
    return {**dict(body), "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# The declared derived-view grammar
# ---------------------------------------------------------------------------


def _view_weight(view: Mapping[str, int]) -> int:
    return abs(view["i"]) + 2 * abs(view["j"])


def declared_views() -> tuple[dict[str, Any], ...]:
    """Every view of the declared grammar, in declared Occam order.

    ``z = v * (m + s)^i * ((m + s)^2 - c)^j``.  When ``j`` is zero the trailing factor is
    identically one, so ``c`` is pinned to zero rather than generating ten aliases.
    """

    raw: list[dict[str, int]] = []
    for shift in range(SHIFT_BOUND[0], SHIFT_BOUND[1] + 1):
        for index_exponent in range(INDEX_EXPONENT_BOUND[0], INDEX_EXPONENT_BOUND[1] + 1):
            for quadratic in range(QUADRATIC_EXPONENT_BOUND[0], QUADRATIC_EXPONENT_BOUND[1] + 1):
                offsets = (
                    (0,)
                    if quadratic == 0
                    else tuple(range(OFFSET_BOUND[0], OFFSET_BOUND[1] + 1))
                )
                for offset in offsets:
                    raw.append(
                        {"c": offset, "i": index_exponent, "j": quadratic, "s": shift}
                    )
    raw.sort(key=lambda view: (_view_weight(view), view["s"], view["c"], view["i"], view["j"]))
    return tuple(
        {
            **view,
            "rank": rank,
            "view_id": f"s={view['s']};i={view['i']};j={view['j']};c={view['c']}",
            "weight": _view_weight(view),
        }
        for rank, view in enumerate(raw)
    )


VIEWS = declared_views()

DECLARED_VIEW_FAMILY = {
    "index_exponent_range": list(INDEX_EXPONENT_BOUND),
    "offset_range": list(OFFSET_BOUND),
    "ordering": (
        "by total weight abs(i) + 2*abs(j), then s, then c, then i, then j; the first "
        "admitted view in this order is frozen and nothing later is considered"
    ),
    "quadratic_exponent_range": list(QUADRATIC_EXPONENT_BOUND),
    "shift_range": list(SHIFT_BOUND),
    "template": (
        "z = v * (m + s)^i * ((m + s)^2 - c)^j; when j is zero the offset c is pinned to "
        "zero, because the trailing factor is then identically one"
    ),
    "total_declared_views": len(VIEWS),
}

REJECTION_REASON_LEGEND = {
    "not_constant": (
        "the derived column's relative spread about its exact rational mean exceeded the "
        "declared fit tolerance"
    ),
    "undefined": "the view is undefined on at least one fit row (a zero base under a factor)",
}


# ---------------------------------------------------------------------------
# Config validation and the forbidden-vocabulary guard
# ---------------------------------------------------------------------------


def config_vocabulary_violations(text: str) -> list[str]:
    """Forbidden tokens present in `text`, tokenized on letter runs only.

    Same tokenization as the planetary campaign's guard, over the extended list.
    """

    found = set(re.findall(r"[a-z]+", text.lower()))
    return sorted(found & set(FORBIDDEN_VOCABULARY))


def _assert_config_vocabulary(config: Mapping[str, Any]) -> None:
    violations = config_vocabulary_violations(canonical_json_bytes(config).decode("utf-8"))
    if violations:
        raise BalmerBohrCaseStudyError(
            f"public config leaked target vocabulary: {', '.join(violations)}"
        )


def _validate_rows_block(rows: Mapping[str, Any]) -> None:
    if set(rows) != {
        "case_id",
        "columns",
        "fit_rows",
        "holdout_indices",
        "sealed_holdout_commitment_sha256",
        "sealed_target_sha256",
    }:
        raise BalmerBohrCaseStudyError("public rows block schema changed")
    if tuple(rows["columns"]) != ("m", "v"):
        raise BalmerBohrCaseStudyError("public column inventory changed")
    if not _hex_digest(rows["sealed_target_sha256"]) or not _hex_digest(
        rows["sealed_holdout_commitment_sha256"]
    ):
        raise BalmerBohrCaseStudyError("sealed commitment malformed")
    fit_rows = rows["fit_rows"]
    if not isinstance(fit_rows, list) or not 4 <= len(fit_rows) <= 32:
        raise BalmerBohrCaseStudyError("public fit-row budget changed")
    expected_index = 1
    previous: Fraction | None = None
    for row in fit_rows:
        if not isinstance(row, Mapping) or set(row) != {"m", "v"}:
            raise BalmerBohrCaseStudyError("public row schema changed")
        if row["m"] != expected_index:
            raise BalmerBohrCaseStudyError("public row label is not consecutive from one")
        expected_index += 1
        value = _fraction(row["v"])
        if value <= 0:
            raise BalmerBohrCaseStudyError("public values must be positive")
        if previous is not None and value >= previous:
            raise BalmerBohrCaseStudyError("public rows are not strictly descending in v")
        previous = value
    holdout = rows["holdout_indices"]
    if not isinstance(holdout, list) or not holdout:
        raise BalmerBohrCaseStudyError("holdout index inventory changed")
    for index in holdout:
        if not isinstance(index, int) or isinstance(index, bool) or index != expected_index:
            raise BalmerBohrCaseStudyError("holdout labels do not continue the fit labels")
        expected_index += 1


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "data_declaration",
        "policies",
        "rows",
        "schema_version",
        "target_fixture_commitment_sha256",
        "view_family",
    }:
        raise BalmerBohrCaseStudyError("config keys changed")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise BalmerBohrCaseStudyError("config identity changed")
    if config["policies"] != POLICIES:
        raise BalmerBohrCaseStudyError("prospective policy changed")
    if config["view_family"] != DECLARED_VIEW_FAMILY:
        raise BalmerBohrCaseStudyError("declared view grammar changed")
    if not _hex_digest(config["target_fixture_commitment_sha256"]):
        raise BalmerBohrCaseStudyError("target fixture commitment malformed")
    declaration = config["data_declaration"]
    if not isinstance(declaration, Mapping) or set(declaration) != {
        "boundary",
        "fit_relative_tolerance",
        "holdout_relative_tolerance",
        "provenance_commitment_sha256",
        "quoted_precision_relative_bound",
        "row_index_meaning",
        "scale_systematic_relative_bound",
        "values_are_exact_rationals",
        "values_carry_declared_measurement_uncertainty",
    }:
        raise BalmerBohrCaseStudyError("data declaration schema changed")
    if (
        declaration["values_are_exact_rationals"] is not True
        or declaration["values_carry_declared_measurement_uncertainty"] is not True
        or not _hex_digest(declaration["provenance_commitment_sha256"])
    ):
        raise BalmerBohrCaseStudyError("data declaration content changed")
    fit_tolerance = _decimal_fraction(declaration["fit_relative_tolerance"])
    holdout_tolerance = _decimal_fraction(declaration["holdout_relative_tolerance"])
    if not 0 < fit_tolerance <= holdout_tolerance <= Fraction(1, 100):
        raise BalmerBohrCaseStudyError("declared tolerances left the prospective envelope")
    _validate_rows_block(config["rows"])
    _assert_config_vocabulary(config)


# ---------------------------------------------------------------------------
# Phase A: the blind empirical race
# ---------------------------------------------------------------------------


def _raw_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"point": row["m"], "value": dict(row["v"])} for row in config["rows"]["fit_rows"]]


def _fit_pairs(config: Mapping[str, Any]) -> list[tuple[int, Fraction]]:
    return [(row["m"], _fraction(row["v"])) for row in config["rows"]["fit_rows"]]


def _view_column(view: Mapping[str, Any], pairs: Sequence[tuple[int, Fraction]]) -> list[Fraction]:
    """Exact derived column for one declared view, or an empty list where undefined."""

    column: list[Fraction] = []
    for label, value in pairs:
        shifted = label + view["s"]
        base = shifted * shifted - view["c"]
        if (shifted == 0 and view["i"] != 0) or (base == 0 and view["j"] != 0):
            return []
        column.append(value * Fraction(shifted) ** view["i"] * Fraction(base) ** view["j"])
    return column


def _relative_spread(column: Sequence[Fraction]) -> tuple[Fraction, Fraction] | None:
    """Exact ``(mean, max |z - mean| / |mean|)`` over a derived column."""

    mean = sum(column, Fraction(0)) / len(column)
    if mean == 0:
        return None
    return mean, max(abs(value - mean) for value in column) / abs(mean)


def _run_view_search(
    pairs: Sequence[tuple[int, Fraction]], fit_tolerance: Fraction
) -> dict[str, Any]:
    """Evaluate every declared view, B1-check each in exact arithmetic, and log all of it."""

    log: list[dict[str, Any]] = []
    b1_stream: list[str] = []
    admitted: list[dict[str, Any]] = []
    reasons = {"not_constant": 0, "undefined": 0}
    for view in VIEWS:
        entry: dict[str, Any] = {
            "rank": view["rank"],
            "view_id": view["view_id"],
            "weight": view["weight"],
        }
        column = _view_column(view, pairs)
        if not column:
            reasons["undefined"] += 1
            log.append({**entry, "reason": "undefined", "relative_spread": None, "status": "SKIP"})
            continue
        rows = [
            {"point": label, "value": _fraction_data(value)}
            for (label, _), value in zip(pairs, column, strict=True)
        ]
        b1 = synthesize_basis(rows)
        b1_stream.append(b1["content_sha256"])
        measured = _relative_spread(column)
        if measured is None:
            reasons["undefined"] += 1
            log.append(
                {
                    **entry,
                    "b1_decision": b1["decision"],
                    "reason": "undefined",
                    "relative_spread": None,
                    "status": "SKIP",
                }
            )
            continue
        mean, spread = measured
        is_admitted = spread <= fit_tolerance
        if not is_admitted:
            reasons["not_constant"] += 1
        entry.update(
            {
                "b1_decision": b1["decision"],
                "relative_spread": _decimal_text(spread, SPREAD_DECIMAL_PLACES),
                "status": "ADMITTED" if is_admitted else "REJECTED",
            }
        )
        if not is_admitted:
            entry["reason"] = "not_constant"
        log.append(entry)
        if is_admitted:
            admitted.append(
                {
                    "b1": b1,
                    "column": column,
                    "entry": entry,
                    "mean": mean,
                    "rows": rows,
                    "spread": spread,
                    "view": view,
                }
            )
    return {
        "admitted": admitted,
        "b1_receipt_stream_sha256": canonical_sha256(b1_stream),
        "log": log,
        "reasons": reasons,
    }


def _tolerance_robustness(
    pairs: Sequence[tuple[int, Fraction]], declared: Fraction
) -> list[dict[str, Any]]:
    """What the admitted set would have been at every declared rung of the ladder."""

    ladder: list[dict[str, Any]] = []
    for text in TOLERANCE_ROBUSTNESS_LADDER:
        tolerance = _decimal_fraction(text)
        admitted: list[str] = []
        for view in VIEWS:
            column = _view_column(view, pairs)
            if not column:
                continue
            measured = _relative_spread(column)
            if measured is not None and measured[1] <= tolerance:
                admitted.append(view["view_id"])
        ladder.append(
            {
                "admitted_view_ids": admitted,
                "is_the_declared_tolerance": tolerance == declared,
                "relative_tolerance": text,
                "views_admitted": len(admitted),
            }
        )
    return ladder


def _relation_latex(exponents: Mapping[str, int], index: sp.Symbol, response: str) -> str:
    """LaTeX for ``response = B * base^(-i) * (base^2 - c)^(-j)`` with ``base = index + s``."""

    base = index + exponents["s"]
    scale = sp.Symbol("B", positive=True)
    expression = (
        scale * base ** (-exponents["i"]) * (base**2 - exponents["c"]) ** (-exponents["j"])
    )
    return sp.latex(sp.Eq(sp.Symbol(response, positive=True), expression))


def _invariant_latex(exponents: Mapping[str, int]) -> str:
    label = sp.Symbol("m", positive=True)
    base = label + exponents["s"]
    value = sp.Symbol("v", positive=True)
    derived = value * base ** exponents["i"] * (base**2 - exponents["c"]) ** exponents["j"]
    return sp.latex(sp.Eq(sp.Symbol("z", positive=True), derived))


def _candidate(admitted: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not admitted:
        return None
    winner = admitted[0]
    view, mean = winner["view"], winner["mean"]
    exponents = {key: view[key] for key in ("c", "i", "j", "s")}
    shifted = f"(m + {view['s']})" if view["s"] else "m"
    invariant = f"v*{shifted}^({view['i']})*({shifted}^2 - {view['c']})^({view['j']})"
    statement = (
        f"v = B*{shifted}^({-view['i']})*({shifted}^2 - {view['c']})^({-view['j']}) "
        f"with B = {_decimal_text(mean, 4)}"
    )
    return {
        "constant": _fraction_data(mean),
        "constant_decimal": _decimal_text(mean, 10),
        "invariant_expression": invariant,
        "invariant_latex": _invariant_latex(exponents),
        "kind": "shifted_rational_relation",
        "latex": _relation_latex(exponents, sp.Symbol("m", positive=True), "v"),
        "recovered_index_offset": view["s"],
        "rejected_earlier_views": view["rank"],
        "relative_spread": _decimal_text(winner["spread"], SPREAD_DECIMAL_PLACES),
        "source_stage": "b4_declared_view_search",
        "statement": statement,
        "sympy_expression": (
            f"Rational({mean.numerator}, {mean.denominator})"
            f"*(m + {view['s']})**{-view['i']}"
            f"*((m + {view['s']})**2 - {view['c']})**{-view['j']}"
        ),
        "view_exponents": exponents,
        "view_id": view["view_id"],
        "weight": view["weight"],
    }


def _holdout_predictions(
    candidate: Mapping[str, Any] | None, indices: Sequence[int]
) -> list[dict[str, Any]]:
    """Predict the sealed rows from the frozen candidate, before anything is unsealed."""

    if candidate is None:
        return []
    exponents = candidate["view_exponents"]
    constant = _fraction(candidate["constant"])
    predictions: list[dict[str, Any]] = []
    for index in indices:
        shifted = index + exponents["s"]
        base = shifted * shifted - exponents["c"]
        value = (
            constant * Fraction(shifted) ** (-exponents["i"]) * Fraction(base) ** (-exponents["j"])
        )
        predictions.append(
            {
                "m": index,
                "predicted_decimal": _decimal_text(value, RESIDUAL_DECIMAL_PLACES),
                "predicted_value": _fraction_data(value),
                "shifted_index": shifted,
            }
        )
    return predictions


def _stage(stage_id: str, tool: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": receipt["decision"],
        "receipt_sha256": receipt["content_sha256"],
        "stage_id": stage_id,
        "tool": tool,
    }


def _guarded_stage(
    stage_id: str, tool: str, run: Any, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Run a declared stage, or record a typed blocker where its caps forbid it.

    Four rows is below what several declared stages accept; the honest receipt records
    that refusal with the tool's own message rather than quietly skipping the stage.
    """

    try:
        return _stage(stage_id, tool, run(rows))
    except ValueError as error:
        return {
            "decision": "NOT_APPLICABLE",
            "reason": str(error),
            "receipt_sha256": None,
            "stage_id": stage_id,
            "tool": tool,
        }


def _run_phase_a(config: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the engine does before anything is unsealed."""

    declaration = config["data_declaration"]
    fit_tolerance = _decimal_fraction(declaration["fit_relative_tolerance"])
    pairs = _fit_pairs(config)
    raw = _raw_rows(config)

    stages: list[dict[str, Any]] = []
    b1_raw = synthesize_basis(raw)
    stages.append(_stage("b1_basis_synthesis", "synthesize_basis", b1_raw))
    if b1_raw["decision"] != "PASS":
        stages.append(
            _guarded_stage("b7_structural_repair", "repair_structure", repair_structure, raw)
        )
    stages.append(
        _guarded_stage(
            "b3_conjecture_generation",
            "generate_conjectures",
            lambda rows: generate_conjectures(rows, statement_kinds=STATEMENT_KINDS),
            raw,
        )
    )

    search = _run_view_search(pairs, fit_tolerance)
    stages.append(
        {
            "decision": "PASS" if search["admitted"] else "BLOCK",
            "receipt_sha256": canonical_sha256(search["log"]),
            "stage_id": "b4_declared_view_search",
            "tool": "synthesize_basis_over_declared_views_plus_declared_spread_test",
            "views_admitted": len(search["admitted"]),
            "views_evaluated": len(search["log"]),
        }
    )
    candidate = _candidate(search["admitted"])
    b2: dict[str, Any] | None = None
    if search["admitted"]:
        b2 = search_nonlinear(search["admitted"][0]["rows"])
        stages.append(_stage("b2_nonlinear_coefficient_search", "search_nonlinear", b2))
    predictions = _holdout_predictions(candidate, config["rows"]["holdout_indices"])
    winner = search["admitted"][0] if search["admitted"] else None
    return {
        "b1_on_the_winning_column": (
            None
            if winner is None
            else {
                "decision": winner["b1"]["decision"],
                "first_blocker": winner["b1"].get("first_blocker"),
                "note": EXACT_ARITHMETIC_NOTE,
                "receipt_sha256": winner["b1"]["content_sha256"],
            }
        ),
        "b2_on_the_winning_column": (
            None
            if b2 is None
            else {
                "decision": b2["decision"],
                "model_id": (b2.get("result") or {}).get("model_id"),
                "receipt_sha256": b2["content_sha256"],
            }
        ),
        "candidate": candidate,
        "candidate_selection_rule": CANDIDATE_SELECTION_RULE,
        "declared_view_family": DECLARED_VIEW_FAMILY,
        "derived_column_of_the_winning_view": (
            None
            if winner is None
            else [_decimal_text(value, RESIDUAL_DECIMAL_PLACES) for value in winner["column"]]
        ),
        "fit_relative_tolerance": declaration["fit_relative_tolerance"],
        "fit_rows_sha256": canonical_sha256(config["rows"]["fit_rows"]),
        "holdout_indices": list(config["rows"]["holdout_indices"]),
        "holdout_predictions": predictions,
        "index_dependent_raw_stage_decisions": {
            stage["stage_id"]: stage["decision"]
            for stage in stages
            if stage["stage_id"] != "b4_declared_view_search"
        },
        "public_rows": [
            {"m": label, "v": _fraction_data(value), "v_decimal": _decimal_text(value, 2)}
            for label, value in pairs
        ],
        "search_log": search["log"],
        "search_space": {
            "b1_receipt_stream_sha256": search["b1_receipt_stream_sha256"],
            "rejection_reason_legend": REJECTION_REASON_LEGEND,
            "rejection_reasons": search["reasons"],
            "search_log_sha256": canonical_sha256(search["log"]),
            "total_declared_views": len(VIEWS),
            "views_admitted": len(search["admitted"]),
            "views_evaluated": len(search["log"]) - search["reasons"]["undefined"],
            "views_rejected": search["reasons"]["not_constant"],
            "views_undefined": search["reasons"]["undefined"],
        },
        "stages": stages,
        "tolerance_robustness": _tolerance_robustness(pairs, fit_tolerance),
    }


# ---------------------------------------------------------------------------
# The single atomic unseal
# ---------------------------------------------------------------------------

_TARGET_KEYS = {
    "attribution",
    "attribution_year",
    "classical_id",
    "column_meanings",
    "constant",
    "constant_decimal",
    "constant_name",
    "constant_published_decimal_places",
    "expression",
    "holdout_rows",
    "invariant_expression",
    "kind",
    "label",
    "parameters",
    "salt",
    "statement",
    "target_id",
}


def _unseal(
    root: Path, config: Mapping[str, Any], phase_a_root: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Open the sealed fixture once, after Phase A is sealed, and open every commitment."""

    if not _hex_digest(phase_a_root):
        raise BalmerBohrCaseStudyError("target unseal attempted before candidate freeze")
    path = _resolve(root, TARGETS_PATH)
    try:
        raw = path.read_bytes()
        fixture = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BalmerBohrCaseStudyError("target fixture unavailable") from error
    if (
        not isinstance(fixture, dict)
        or canonical_sha256(fixture) != config["target_fixture_commitment_sha256"]
    ):
        raise BalmerBohrCaseStudyError("target fixture content changed")
    if set(fixture) != {"holdout", "provenance", "schema_version", "target"} or (
        fixture["schema_version"] != TARGET_SCHEMA
    ):
        raise BalmerBohrCaseStudyError("target fixture schema changed")
    rows_block = config["rows"]
    provenance = fixture["provenance"]
    if (
        not isinstance(provenance, dict)
        or canonical_sha256(provenance)
        != config["data_declaration"]["provenance_commitment_sha256"]
    ):
        raise BalmerBohrCaseStudyError("provenance commitment did not open")
    holdout = fixture["holdout"]
    if (
        not isinstance(holdout, dict)
        or canonical_sha256(holdout) != rows_block["sealed_holdout_commitment_sha256"]
    ):
        raise BalmerBohrCaseStudyError("holdout commitment did not open")
    target = fixture["target"]
    if not isinstance(target, dict) or set(target) != _TARGET_KEYS:
        raise BalmerBohrCaseStudyError("target record schema changed")
    if canonical_sha256(target) != rows_block["sealed_target_sha256"]:
        raise BalmerBohrCaseStudyError("target commitment did not open")
    if target["target_id"] != rows_block["case_id"]:
        raise BalmerBohrCaseStudyError("target identity changed")
    if not str(target["classical_id"]).isidentifier():
        raise BalmerBohrCaseStudyError("classical identifier malformed")
    if target["holdout_rows"] != holdout["rows"]:
        raise BalmerBohrCaseStudyError("sealed holdout rows disagree with the sealed target")
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return target, holdout, provenance, hashlib.sha256(normalized).hexdigest()


# ---------------------------------------------------------------------------
# Post-unseal: replay the sealed rule, then score
# ---------------------------------------------------------------------------


def _replay_sealed_rule(
    config: Mapping[str, Any], target: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the sealed closed form on every row and check it against the source table."""

    constant = Fraction(target["constant"])
    offset = target["parameters"]["view_exponents"]["s"]
    rows: list[dict[str, Any]] = []
    worst = Fraction(0)
    public = {row["m"]: _fraction(row["v"]) for row in config["rows"]["fit_rows"]}
    public.update({row["index"]: _fraction(row["v"]) for row in target["holdout_rows"]})
    source_rows = [("fit", row) for row in provenance["fit_source"]["rows"]]
    source_rows += [("holdout", row) for row in provenance["holdout_source"]["rows"]]
    for origin, source in source_rows:
        label = source["index"]
        shifted = label + offset
        if source["quantum_index"] != shifted:
            raise BalmerBohrCaseStudyError("sealed source table disagrees with the sealed offset")
        measured = _decimal_fraction(source["value_1e-10_m"])
        if public.get(label) != measured:
            raise BalmerBohrCaseStudyError("published rows do not replay from the sealed table")
        rule_value = constant * Fraction(shifted**2, shifted**2 - 4)
        relative = abs(rule_value - measured) / measured
        worst = max(worst, relative)
        rows.append(
            {
                "m": label,
                "measured_decimal": _decimal_text(measured, RESIDUAL_DECIMAL_PLACES),
                "relative_residual": _decimal_text(relative, SPREAD_DECIMAL_PLACES),
                "residual_decimal": _decimal_text(rule_value - measured, RESIDUAL_DECIMAL_PLACES),
                "sealed_rule_decimal": _decimal_text(rule_value, RESIDUAL_DECIMAL_PLACES),
                "shifted_index": shifted,
                "source": origin,
            }
        )
    return {
        "max_relative_residual": _decimal_text(worst, SPREAD_DECIMAL_PLACES),
        "note": (
            "The sealed closed form is recomputed here on every row and compared with the "
            "sealed source table; the public rows are shown to be exactly those table entries. "
            "Nothing is fitted at this point -- the candidate was frozen before this ran."
        ),
        "rows": rows,
        "rows_replayed": len(rows),
        "sealed_rule": target["statement"],
    }


def _score(
    phase_a: Mapping[str, Any],
    config: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the frozen candidate and its predictions against the opened target."""

    candidate = phase_a["candidate"]
    holdout_tolerance = _decimal_fraction(
        config["data_declaration"]["holdout_relative_tolerance"]
    )
    if candidate is None:
        return {
            "constant_match": False,
            "holdout": [],
            "holdout_within_tolerance": False,
            "method": "no_candidate",
            "structure_match": False,
            "verdict": "MISSED",
            "verdict_rule": VERDICT_RULE,
        }
    sealed_constant = Fraction(target["constant"])
    places = target["constant_published_decimal_places"]
    found = _fraction(candidate["constant"])
    structure_match = candidate["view_exponents"] == target["parameters"]["view_exponents"]
    rounded = _round_to_places(found, places)
    constant_match = rounded == sealed_constant
    relative_constant_gap = abs(found - sealed_constant) / sealed_constant

    sealed_values = {row["index"]: _fraction(row["v"]) for row in target["holdout_rows"]}
    holdout: list[dict[str, Any]] = []
    worst = Fraction(0)
    for prediction in phase_a["holdout_predictions"]:
        measured = sealed_values[prediction["m"]]
        predicted = _fraction(prediction["predicted_value"])
        relative = abs(predicted - measured) / measured
        worst = max(worst, relative)
        holdout.append(
            {
                "m": prediction["m"],
                "measured_decimal": _decimal_text(measured, RESIDUAL_DECIMAL_PLACES),
                "predicted_decimal": prediction["predicted_decimal"],
                "relative_residual": _decimal_text(relative, SPREAD_DECIMAL_PLACES),
                "residual_decimal": _decimal_text(predicted - measured, RESIDUAL_DECIMAL_PLACES),
                "shifted_index": prediction["shifted_index"],
                "within_declared_tolerance": relative <= holdout_tolerance,
            }
        )
    holdout_ok = bool(holdout) and all(row["within_declared_tolerance"] for row in holdout)

    discovered = sp.sympify(candidate["sympy_expression"], locals=_law_locals())
    expected = sp.sympify(
        target["expression"].replace("B", f"Rational({sealed_constant.numerator}, "
                                          f"{sealed_constant.denominator})"),
        locals=_law_locals(),
    )
    shape_difference = sp.simplify(
        sp.expand(discovered / found - expected / sealed_constant)
    )
    if structure_match and constant_match and holdout_ok:
        verdict = "REDISCOVERED_EXACT"
    elif structure_match:
        verdict = "PARTIAL"
    else:
        verdict = "MISSED"
    sealed_exponents = target["parameters"]["view_exponents"]
    return {
        "attribution": target["attribution"],
        "attribution_year": target["attribution_year"],
        "classical_id": target["classical_id"],
        "classical_latex": _relation_latex(
            sealed_exponents, sp.Symbol("M", positive=True) - sealed_exponents["s"], "lambda"
        ),
        "classical_upper_index_note": (
            f"M = m + {sealed_exponents['s']}, the ordinal the public rows did not carry"
        ),
        "column_meanings": target["column_meanings"],
        "constant_match": constant_match,
        "constant_published_decimal_places": places,
        "discovered_latex": candidate["latex"],
        "constant_relative_gap": _decimal_text(relative_constant_gap, SPREAD_DECIMAL_PLACES),
        "constant_rounded_to_published_places": _decimal_text(rounded, places),
        "discovered_sympy": _text(discovered),
        "holdout": holdout,
        "holdout_max_relative_residual": _decimal_text(worst, SPREAD_DECIMAL_PLACES),
        "holdout_relative_tolerance": config["data_declaration"]["holdout_relative_tolerance"],
        "holdout_within_tolerance": holdout_ok,
        "method": (
            "exact integer equality of the frozen view exponents, exact rational rounding of "
            "the frozen constant to the published precision, sympy equality of the recovered "
            "shape with the classical one, and a declared relative-tolerance test on every "
            "sealed holdout row"
        ),
        "sealed_constant_decimal": target["constant_decimal"],
        "shape_difference": _text(shape_difference),
        "shapes_agree": shape_difference == 0,
        "structure_match": structure_match,
        "target_statement": target["statement"],
        "target_sympy": _text(expected),
        "verdict": verdict,
        "verdict_rule": VERDICT_RULE,
    }


def _law_locals() -> dict[str, Any]:
    return {"Rational": sp.Rational, "m": sp.Symbol("m", positive=True)}


# ---------------------------------------------------------------------------
# Race 2: the derivation, from two declared postulates and no data
# ---------------------------------------------------------------------------

#: The two postulates, declared before anything is derived.  Everything in the chain is a
#: consequence of these plus classical mechanics; nothing else is assumed.
POSTULATES = (
    {
        "id": "P1",
        "statement": (
            "The negative charge moves in a circular orbit about the positive charge under "
            "the Coulomb attraction k*e^2/r^2, and its angular momentum is restricted to "
            "integer multiples of hbar: L = m_e*v*r = n*hbar."
        ),
        "equations": ["m_e*v^2/r = k*e^2/r^2", "m_e*v*r = n*hbar"],
    },
    {
        "id": "P2",
        "statement": (
            "A change between two such states is accompanied by a single quantum of "
            "radiation carrying the whole difference: h*nu = E(n_2) - E(n_1)."
        ),
        "equations": ["h*nu = E(n_2) - E(n_1)", "nu = c/lambda"],
    },
)

#: CODATA 2018 recommended values.  These are *cited*, never fitted, and no dataset is
#: opened to obtain them.  h, e and c are exact by the 2019 SI redefinition.
CITED_CONSTANTS: dict[str, dict[str, str]] = {
    "electron_mass_kg": {
        "value": "9.1093837015e-31",
        "standard_uncertainty": "2.8e-40",
        "citation": "CODATA 2018 recommended value of the electron mass.",
    },
    "elementary_charge_C": {
        "value": "1.602176634e-19",
        "standard_uncertainty": "0",
        "citation": "Exact by the 2019 SI redefinition of the ampere.",
    },
    "planck_constant_J_s": {
        "value": "6.62607015e-34",
        "standard_uncertainty": "0",
        "citation": "Exact by the 2019 SI redefinition of the kilogram.",
    },
    "speed_of_light_m_s": {
        "value": "299792458",
        "standard_uncertainty": "0",
        "citation": "Exact by the 1983 SI definition of the metre.",
    },
    "vacuum_electric_permittivity_F_m": {
        "value": "8.8541878128e-12",
        "standard_uncertainty": "1.3e-21",
        "citation": "CODATA 2018 recommended value of the electric constant.",
    },
    "proton_mass_kg": {
        "value": "1.67262192369e-27",
        "standard_uncertainty": "5.1e-37",
        "citation": (
            "CODATA 2018 recommended value of the proton mass; used only for the "
            "finite-nuclear-mass correction reported after the primary comparison."
        ),
    },
}

#: The measured quantity the derived constant is checked against.
CITED_MEASUREMENT = {
    "rydberg_constant_per_m": {
        "value": "1.0973731568160e7",
        "standard_uncertainty": "2.1e-5",
        "citation": (
            "CODATA 2018 recommended value of the Rydberg constant, R_infinity = "
            "10973731.568160(21) m^-1; the value most often quoted to ten figures is "
            "1.0973731568e7 m^-1 and is reported alongside."
        ),
    },
    "rydberg_constant_per_m_ten_figures": {
        "value": "1.0973731568e7",
        "standard_uncertainty": "0",
        "citation": "The same constant truncated to the ten figures usually quoted.",
    },
}

#: An auxiliary cited constant used for one unit-frame conversion in the closing note.
#: It is not fitted and it is not part of the primary comparison.
CITED_AUXILIARY = {
    "refractive_index_of_standard_air_visible": {
        "value": "1.000277",
        "citation": (
            "Standard value of the refractive index of dry air at 15 C and 101325 Pa over "
            "the visible range (Edlen-type dispersion formulae give 1.00028 near 400 nm). "
            "Used only to note that nineteenth-century tables are air wavelengths while the "
            "Rydberg constant yields vacuum wavenumbers; the conversion is reported, never "
            "adjusted, and the primary comparison does not use it."
        ),
    }
}

_LEADING_CONSTANT_NAMES = (
    "electron_mass_kg",
    "elementary_charge_C",
    "planck_constant_J_s",
    "speed_of_light_m_s",
    "vacuum_electric_permittivity_F_m",
)


def _derivation_symbols() -> dict[str, sp.Symbol]:
    names = "n hbar m_e k e h c r v n_1 n_2 lambda_ B"
    symbols = sp.symbols(names, positive=True)
    return dict(zip(names.split(), symbols, strict=True))


def _step(
    number: int,
    statement: str,
    symbolic_result: str,
    check: str,
    check_status: str,
    **extra: Any,
) -> dict[str, Any]:
    if check_status != "pass":
        raise BalmerBohrCaseStudyError(f"derivation step {number} failed its check: {check}")
    return {
        "check": check,
        "check_status": check_status,
        "statement": statement,
        "step": number,
        "symbolic_result": symbolic_result,
        **extra,
    }


def solve_orbit(quantization_exponent: int = 1) -> dict[str, sp.Expr]:
    """Solve the two postulate-P1 equations for ``(v, r)``.

    ``quantization_exponent`` is the power of ``n`` in the angular-momentum condition.  The
    physical postulate is 1; the negative control passes 2 and must not reproduce ``r ~ n^2``.
    """

    sym = _derivation_symbols()
    n, hbar, m_e, k, e, r, v = (
        sym["n"], sym["hbar"], sym["m_e"], sym["k"], sym["e"], sym["r"], sym["v"]
    )
    solutions = sp.solve(
        [
            sp.Eq(m_e * v**2 / r, k * e**2 / r**2),
            sp.Eq(m_e * v * r, n**quantization_exponent * hbar),
        ],
        [v, r],
        dict=True,
    )
    if len(solutions) != 1:
        raise BalmerBohrCaseStudyError("the orbit conditions did not yield a unique solution")
    return {
        "radius": sp.simplify(solutions[0][r]),
        "speed": sp.simplify(solutions[0][v]),
    }


def rydberg_expression(hbar_power: int = 2) -> sp.Expr:
    """Derive ``R`` from the postulates, with the power of ``hbar`` in ``E_n`` exposed.

    The physical value is 2; the negative control passes 3 and must miss the measured
    constant by many orders of magnitude.
    """

    sym = _derivation_symbols()
    n, hbar, m_e, k, e, h, c = (
        sym["n"], sym["hbar"], sym["m_e"], sym["k"], sym["e"], sym["h"], sym["c"]
    )
    n_1, n_2 = sym["n_1"], sym["n_2"]
    energy = -m_e * k**2 * e**4 / (2 * hbar**hbar_power * n**2)
    difference = sp.simplify(energy.subs(n, n_2) - energy.subs(n, n_1))
    inverse_wavelength = sp.simplify(
        sp.expand(sp.simplify(difference / (h * c)).subs(hbar, h / (2 * sp.pi)))
    )
    return sp.simplify(inverse_wavelength / (1 / n_1**2 - 1 / n_2**2))


def _numeric_rydberg(expression: sp.Expr) -> Any:
    """Substitute the cited constants into a symbolic ``R`` at working precision."""

    mp.mp.dps = WORKING_PRECISION_DIGITS
    sym = _derivation_symbols()
    planck = mp.mpf(CITED_CONSTANTS["planck_constant_J_s"]["value"])
    permittivity = mp.mpf(CITED_CONSTANTS["vacuum_electric_permittivity_F_m"]["value"])
    arguments = {
        "m_e": mp.mpf(CITED_CONSTANTS["electron_mass_kg"]["value"]),
        "e": mp.mpf(CITED_CONSTANTS["elementary_charge_C"]["value"]),
        "h": planck,
        "c": mp.mpf(CITED_CONSTANTS["speed_of_light_m_s"]["value"]),
        "k": 1 / (4 * mp.pi * permittivity),
        "hbar": planck / (2 * mp.pi),
    }
    names = tuple(arguments)
    evaluate = sp.lambdify(
        tuple(sym[name] for name in names), expression, modules="mpmath"
    )
    return mp.mpf(evaluate(*(arguments[name] for name in names)))


def bohr_derivation() -> dict[str, Any]:
    """The whole chain, recomputed: postulates -> r_n -> E_n -> 1/lambda -> R -> B."""

    mp.mp.dps = WORKING_PRECISION_DIGITS
    sym = _derivation_symbols()
    n, hbar, m_e, k, e, h, c = (
        sym["n"], sym["hbar"], sym["m_e"], sym["k"], sym["e"], sym["h"], sym["c"]
    )
    n_1, n_2 = sym["n_1"], sym["n_2"]
    steps: list[dict[str, Any]] = []

    orbit = solve_orbit()
    radius, speed = orbit["radius"], orbit["speed"]
    expected_radius = n**2 * hbar**2 / (m_e * k * e**2)
    steps.append(
        _step(
            1,
            (
                "Solve the postulate-P1 pair for the orbit: Coulomb attraction supplies the "
                "centripetal force and the angular momentum is n*hbar."
            ),
            f"r_n = {_text(radius)}; v_n = {_text(speed)}",
            "the radius returned by the simultaneous solve equals n^2*hbar^2/(m_e*k*e^2)",
            "pass" if sp.simplify(radius - expected_radius) == 0 else "fail",
            bohr_radius_at_n_equals_one=_text(sp.simplify(radius.subs(n, 1))),
            derived_from="P1",
            latex=sp.latex(sp.Eq(sp.Symbol("r_n"), radius)),
        )
    )

    energy = sp.simplify(m_e * speed**2 / 2 - k * e**2 / radius)
    expected_energy = -m_e * k**2 * e**4 / (2 * hbar**2 * n**2)
    steps.append(
        _step(
            2,
            (
                "Add the kinetic and potential terms at that orbit; the total is "
                "negative and falls as 1/n^2, so the states are bound and ordered."
            ),
            f"E_n = {_text(energy)}",
            "the total equals -m_e*k^2*e^4/(2*hbar^2*n^2) symbolically",
            "pass" if sp.simplify(energy - expected_energy) == 0 else "fail",
            derived_from="P1",
            latex=sp.latex(sp.Eq(sp.Symbol("E_n"), energy)),
            virial_ratio=_text(sp.simplify(energy / (-k * e**2 / (2 * radius)))),
        )
    )

    difference = sp.simplify(energy.subs(n, n_2) - energy.subs(n, n_1))
    inverse_wavelength = sp.simplify(
        sp.expand(sp.simplify(difference / (h * c)).subs(hbar, h / (2 * sp.pi)))
    )
    rydberg = rydberg_expression()
    expected_rydberg = 2 * sp.pi**2 * m_e * e**4 * k**2 / (h**3 * c)
    steps.append(
        _step(
            3,
            (
                "Apply postulate P2: the whole difference leaves as one quantum, so "
                "1/lambda = (E(n_2) - E(n_1))/(h*c). Substituting hbar = h/(2*pi) collects "
                "the constants into a single prefactor."
            ),
            f"1/lambda = {_text(inverse_wavelength)}",
            "the prefactor of (1/n_1^2 - 1/n_2^2) is 2*pi^2*m_e*e^4*k^2/(h^3*c)",
            "pass" if sp.simplify(rydberg - expected_rydberg) == 0 else "fail",
            derived_from="P2",
            latex=sp.latex(sp.Eq(1 / sp.Symbol("lambda"), inverse_wavelength)),
            transition_energy=_text(difference),
        )
    )

    steps.append(
        _step(
            4,
            (
                "Read off the Rydberg constant. It is now an expression in constants that "
                "were measured in entirely different experiments -- no wavelength was used "
                "anywhere above."
            ),
            f"R = {_text(rydberg)}",
            "R factorises out of 1/lambda exactly, leaving (1/n_1^2 - 1/n_2^2)",
            "pass"
            if sp.simplify(inverse_wavelength - rydberg * (1 / n_1**2 - 1 / n_2**2)) == 0
            else "fail",
            derived_from="P1+P2",
            latex=sp.latex(sp.Eq(sp.Symbol("R"), rydberg)),
        )
    )
    return {"inverse_wavelength": inverse_wavelength, "rydberg": rydberg, "steps": steps}


def _relative(value: Any, reference: Any) -> Any:
    return abs(mp.mpf(value) - mp.mpf(reference)) / abs(mp.mpf(reference))


def rydberg_numerics(rydberg: sp.Expr) -> dict[str, Any]:
    """Evaluate ``R`` on the cited constants and compare with the measured constant."""

    mp.mp.dps = WORKING_PRECISION_DIGITS
    derived = _numeric_rydberg(rydberg)
    measured = mp.mpf(CITED_MEASUREMENT["rydberg_constant_per_m"]["value"])
    quoted = mp.mpf(CITED_MEASUREMENT["rydberg_constant_per_m_ten_figures"]["value"])
    return {
        "derived_rydberg_per_m": _mp_text(derived),
        "measured_rydberg_per_m": CITED_MEASUREMENT["rydberg_constant_per_m"]["value"],
        "quoted_rydberg_per_m": CITED_MEASUREMENT["rydberg_constant_per_m_ten_figures"]["value"],
        "relative_error_vs_measured": _mp_text(_relative(derived, measured), 6),
        "relative_error_vs_quoted": _mp_text(_relative(derived, quoted), 6),
        "note": (
            "The residual is at the level of the cited input constants' own uncertainties, "
            "which is the most this comparison can show: nothing here measures anything, it "
            "recomputes a published number from other published numbers."
        ),
        "symbolic_form": _text(rydberg),
    }


def close_the_loop(rydberg: sp.Expr, numerics: Mapping[str, Any]) -> dict[str, Any]:
    """Show that Balmer's constant is 4/R, and evaluate that against 3645.6."""

    mp.mp.dps = WORKING_PRECISION_DIGITS
    sym = _derivation_symbols()
    n_1, n_2, B = sym["n_1"], sym["n_2"], sym["B"]
    upper = sp.Symbol("M", positive=True)
    inverse = rydberg * (1 / n_1**2 - 1 / n_2**2)
    balmer_branch = sp.simplify((1 / inverse).subs({n_1: 2, n_2: upper}))
    balmer_form = B * upper**2 / (upper**2 - 4)
    residual = sp.simplify(balmer_branch - balmer_form.subs(B, 4 / rydberg))

    derived = mp.mpf(numerics["derived_rydberg_per_m"])
    constant_m = 4 / derived
    constant_angstrom = constant_m * mp.mpf("1e10")
    balmer_constant = mp.mpf("3645.6")

    electron = mp.mpf(CITED_CONSTANTS["electron_mass_kg"]["value"])
    proton = mp.mpf(CITED_CONSTANTS["proton_mass_kg"]["value"])
    reduced_factor = 1 / (1 + electron / proton)
    finite_mass_constant = constant_angstrom / reduced_factor
    air_index = mp.mpf(CITED_AUXILIARY["refractive_index_of_standard_air_visible"]["value"])
    air_constant = finite_mass_constant / air_index
    return {
        "balmer_branch_symbolic": _text(balmer_branch),
        "constant_from_rydberg_1e-10_m": _mp_text(constant_angstrom),
        "constant_from_rydberg_m": _mp_text(constant_m),
        "constant_identity": "B = 4/R, the n_1 = 2 branch of the derived relation",
        "constant_identity_latex": sp.latex(sp.Eq(B, 4 / sp.Symbol("R", positive=True))),
        "balmer_branch_latex": sp.latex(sp.Eq(sp.Symbol("lambda"), balmer_form)),
        "corrections": [
            {
                "boundary": "cited constants only; no value here is fitted",
                "id": "finite_nuclear_mass",
                "relative_size": _mp_text(electron / proton, 6),
                "resulting_constant_1e-10_m": _mp_text(finite_mass_constant),
                "statement": (
                    "R_infinity assumes an infinitely heavy nucleus. Replacing the electron "
                    "mass by the reduced mass multiplies B by (1 + m_e/m_p)."
                ),
            },
            {
                "boundary": (
                    "auxiliary cited constant, reported for frame agreement only; the primary "
                    "comparison above does not use it"
                ),
                "id": "air_versus_vacuum_frame",
                "relative_size": _mp_text(air_index - 1, 6),
                "resulting_constant_1e-10_m": _mp_text(air_constant),
                "statement": (
                    "The derived constant is a vacuum quantity; the nineteenth-century table "
                    "is air wavelengths, which are smaller by the refractive index."
                ),
            },
        ],
        "published_balmer_constant_1e-10_m": "3645.6",
        "relative_gap_after_both_corrections": _mp_text(
            _relative(air_constant, balmer_constant), 6
        ),
        "relative_gap_vacuum_infinite_mass": _mp_text(
            _relative(constant_angstrom, balmer_constant), 6
        ),
        "residual_of_the_identity": _text(residual),
        "the_identity_holds": residual == 0,
        "note": (
            "The remaining gap after both corrections is of the same order as the systematic "
            "error later found in the wavelength standard the 1868 table was built on -- "
            "roughly one part in seven thousand. That is stated as context, not as a fit: no "
            "parameter anywhere in this chain was adjusted to close it."
        ),
    }


def negative_controls(rydberg: sp.Expr, numerics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Break the derivation on purpose and check that it fails the way it should."""

    mp.mp.dps = WORKING_PRECISION_DIGITS
    sym = _derivation_symbols()
    n, hbar, m_e, k, e = sym["n"], sym["hbar"], sym["m_e"], sym["k"], sym["e"]
    correct_radius = solve_orbit(1)["radius"]
    broken_radius = solve_orbit(2)["radius"]
    correct_power = sp.degree(sp.simplify(correct_radius * m_e * k * e**2 / hbar**2), n)
    broken_power = sp.degree(sp.simplify(broken_radius * m_e * k * e**2 / hbar**2), n)

    broken_rydberg = rydberg_expression(hbar_power=3)
    broken_numeric = _numeric_rydberg(broken_rydberg)
    measured = mp.mpf(CITED_MEASUREMENT["rydberg_constant_per_m"]["value"])
    broken_relative = _relative(broken_numeric, measured)

    controls = [
        {
            "broken_result": f"r_n = {_text(broken_radius)}",
            "correct_result": f"r_n = {_text(correct_radius)}",
            "detected": broken_power != correct_power
            and sp.simplify(broken_radius - correct_radius) != 0,
            "id": "wrong_quantization_exponent",
            "observed_power_of_n": int(broken_power),
            "perturbation": "the angular-momentum condition is set to L = n^2*hbar",
            "required_power_of_n": int(correct_power),
            "why_it_must_fail": (
                "the power of n in the radius is fixed by the quantization rule, so a wrong "
                "exponent changes the whole ladder of orbits"
            ),
        },
        {
            "broken_result": _text(broken_rydberg),
            "broken_value_per_m": _mp_text(broken_numeric, 6),
            "correct_result": _text(rydberg),
            "correct_value_per_m": numerics["derived_rydberg_per_m"],
            "detected": broken_relative > mp.mpf("1e6")
            and sp.simplify(broken_rydberg - rydberg) != 0,
            "id": "wrong_power_of_hbar",
            "perturbation": "the level formula is given hbar^3 instead of hbar^2",
            "relative_error_vs_measured": _mp_text(broken_relative, 6),
            "why_it_must_fail": (
                "hbar carries dimensions, so a wrong power cannot be absorbed by any constant "
                "and the number misses the measurement by tens of orders of magnitude"
            ),
        },
    ]
    for control in controls:
        if control["detected"] is not True:
            raise BalmerBohrCaseStudyError(
                f"negative control did not fire: {control['id']}"
            )
    return controls


def run_derivation() -> dict[str, Any]:
    """Race 2 end to end."""

    chain = bohr_derivation()
    numerics = rydberg_numerics(chain["rydberg"])
    loop = close_the_loop(chain["rydberg"], numerics)
    controls = negative_controls(chain["rydberg"], numerics)
    return {
        "boundary": (
            "Every step is recomputed on each run and re-derived independently in the tests. "
            "No data are read: the constants are cited published values and the measured "
            "Rydberg constant is a citation, not a fit. The postulates are declarations the "
            "engine was given, not propositions it invented."
        ),
        "cited_auxiliary": CITED_AUXILIARY,
        "cited_constants": CITED_CONSTANTS,
        "cited_measurement": CITED_MEASUREMENT,
        "loop_closure": loop,
        "negative_controls": controls,
        "postulates": [dict(postulate) for postulate in POSTULATES],
        "rydberg_numerics": numerics,
        "steps": chain["steps"],
        "symbolic_inverse_wavelength": _text(chain["inverse_wavelength"]),
        "symbolic_rydberg": _text(chain["rydberg"]),
    }


# ---------------------------------------------------------------------------
# The head-to-head block
# ---------------------------------------------------------------------------

WALL_CLOCK_PROTOCOL = (
    "time.perf_counter() around the blind race and around the derivation, taken once on "
    "the build host and written to a separate runtime file. It is deliberately outside "
    "every sealed hash and is not re-measured on replay, so the sealed receipt stays "
    "byte-deterministic. It is not a benchmark and no hardware claim is attached to it."
)

COMPARISON_NOTES = (
    (
        "The engine was handed the grammar. Balmer was not. The shape "
        "v*(m+s)^i*((m+s)^2-c)^j is a human declaration written before the run; what is "
        "measured here is exhaustive, receipted selection inside that space, not the "
        "invention of the space. A MISSED would have meant 'outside the declared grammar', "
        "never 'impossible'."
    ),
    (
        "The engine was told which rows to fit and which to predict, and it was told that "
        "the row label might be offset from the meaningful ordinal. Balmer had to decide "
        "both for himself, from four numbers, with no assurance that any relation existed "
        "at all."
    ),
    (
        "Bohr's postulates are supplied to the engine as declarations. The engine derives "
        "their consequences and checks them; it did not propose them, and proposing them is "
        "the part that was hard."
    ),
    (
        "Wall-clock against human years is not a comparison of difficulty. It is a "
        "comparison of two different activities, and it is reported only because a "
        "head-to-head with an empty column on our side would be less honest, not more."
    ),
    (
        "Every historical duration below is either a cited interval between publications or "
        "the words 'not precisely documented'. No estimate of anyone's personal working "
        "time appears anywhere in this receipt."
    ),
)


def _head_to_head(
    blind: Mapping[str, Any], derivation: Mapping[str, Any], score: Mapping[str, Any]
) -> dict[str, Any]:
    search_space = blind["search_space"]
    candidate = blind["candidate"] or {}
    numerics = derivation["rydberg_numerics"]
    loop = derivation["loop_closure"]
    return {
        "balmer_1885": {
            "human_timescale": {
                "citation": (
                    "A. J. Angstrom, Recherches sur le spectre solaire (Uppsala, 1868); "
                    "J. J. Balmer, 'Notiz uber die Spectrallinien des Wasserstoffs', Annalen "
                    "der Physik und Chemie 261(5):80-87 (1885)."
                ),
                "documented_interval": (
                    "the four wavelengths were in the published literature from 1868 and the "
                    "formula was published in 1885"
                ),
                "documented_interval_years": 17,
                "interval_is_between_publications_not_a_working_time": True,
                "personal_effort_duration": "not precisely documented",
            },
            "inputs_available": (
                "four measured wavelengths of the visible hydrogen lines, quoted to 0.01 in "
                "units of 1e-10 m; no theory of atomic structure, no independent reason to "
                "believe the indices began at three, and no second series to cross-check "
                "against"
            ),
            "method": (
                "hand arithmetic on four numbers, searching for a rational relation between "
                "the wavelengths and a small integer index"
            ),
            "result": (
                "lambda = B*m^2/(m^2 - 4) with B = 3645.6 in units of 1e-10 m and m = 3, 4, "
                "5, 6; further members were then computed from the formula and compared with "
                "the measurements available at the time"
            ),
            "role": "empirical",
        },
        "bohr_1913": {
            "human_timescale": {
                "citation": (
                    "N. Bohr, 'On the Constitution of Atoms and Molecules, Part I', "
                    "Philosophical Magazine Series 6, 26(151):1-25 (July 1913)."
                ),
                "documented_interval": (
                    "Balmer's formula was published in 1885 and its first-principles "
                    "derivation in July 1913"
                ),
                "documented_interval_years": 28,
                "interval_is_between_publications_not_a_working_time": True,
                "personal_effort_duration": "not precisely documented",
            },
            "inputs_available": (
                "no wavelength data were needed for the derivation itself: two postulates, "
                "plus the separately measured values of the electron mass, the elementary "
                "charge, the Planck constant and the speed of light"
            ),
            "method": (
                "Coulomb force balance with angular momentum restricted to integer multiples "
                "of hbar, then a single-quantum emission rule"
            ),
            "result": (
                "1/lambda = R*(1/n_1^2 - 1/n_2^2) with R = 2*pi^2*m_e*e^4*k^2/(h^3*c); "
                "Balmer's series is the n_1 = 2 branch and Balmer's constant is 4/R"
            ),
            "role": "derivation",
        },
        "comparison_notes": list(COMPARISON_NOTES),
        "engine_empirical": {
            "inputs_available": (
                "the same four numbers, anonymized: rows (m, v) with m running 1..4 under a "
                "config that contains no name, no unit and no subject-matter word, checked by "
                "a runtime guard over a declared forbidden vocabulary. The true ordinal was "
                "withheld, so the offset had to be recovered. The three further rows were "
                "given as bare labels with their values sealed."
            ),
            "measured_wall_clock": {
                "boundary": (
                    "one-time host measurement, outside every sealed hash, not re-measured on "
                    "replay"
                ),
                "measurement_path": RUNTIME_PATH,
                "protocol": WALL_CLOCK_PROTOCOL,
            },
            "method": (
                "a declared, finite, published grammar of derived views, every one evaluated "
                "and logged, admitted only by a declared relative-spread test, with the first "
                "admitted view in declared Occam order frozen and nothing later considered"
            ),
            "result": (
                None
                if not candidate
                else (
                    f"{candidate['statement']} -- recovered index offset "
                    f"{candidate['recovered_index_offset']}, verdict {score['verdict']}"
                )
            ),
            "role": "empirical",
            "search_space_size": search_space["total_declared_views"],
            "views_evaluated": search_space["views_evaluated"],
            "views_rejected": search_space["views_rejected"],
            "views_undefined": search_space["views_undefined"],
        },
        "engine_derivation": {
            "inputs_available": (
                "the two postulates as declarations, and the cited CODATA constants for the "
                "numeric evaluation only; no data file is opened and no wavelength is used "
                "anywhere in the symbolic chain"
            ),
            "measured_wall_clock": {
                "boundary": (
                    "one-time host measurement, outside every sealed hash, not re-measured on "
                    "replay"
                ),
                "measurement_path": RUNTIME_PATH,
                "protocol": WALL_CLOCK_PROTOCOL,
            },
            "method": (
                "sympy solves the postulate equations, composes the level formula, applies "
                "the emission rule, and factors out the constant; two negative controls break "
                "the chain on purpose and must fire"
            ),
            "result": (
                f"R = {derivation['symbolic_rydberg']}, evaluating to "
                f"{numerics['derived_rydberg_per_m']} per m, a relative error of "
                f"{numerics['relative_error_vs_measured']} against the measured constant; "
                f"B = 4/R gives {loop['constant_from_rydberg_1e-10_m']} in units of 1e-10 m"
            ),
            "role": "derivation",
            "steps": len(derivation["steps"]),
        },
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_receipt(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Run both races under the seal discipline and return the sealed receipt."""

    root = root.resolve()
    config = _load_json(config_path or _resolve(root, CONFIG_PATH))
    _validate_config(config)

    guard = _CaseStudySealedGuard(root)
    blind_started = time.perf_counter()
    with guard:
        phase_a = _run_phase_a(config)
        try:
            _resolve(root, TARGETS_PATH).read_bytes()
        except PermissionError:
            pass
        else:
            raise BalmerBohrCaseStudyError("pre-unseal target read was not denied")
    blind_seconds = time.perf_counter() - blind_started
    denied_probe = guard.certificate()
    if (
        denied_probe["attempted_target_reads"] != 1
        or denied_probe["denied_target_reads"] != 1
        or denied_probe["denied_content_bytes_exposed"] != 0
        or denied_probe["denied_paths"] != [TARGETS_PATH]
    ):
        raise BalmerBohrCaseStudyError("sealed-target enforcement boundary changed")

    phase_a_root = canonical_sha256(
        {
            "commitments": {
                "holdout": config["rows"]["sealed_holdout_commitment_sha256"],
                "target": config["rows"]["sealed_target_sha256"],
            },
            "denied_probe": denied_probe,
            "phase_a": phase_a,
        }
    )
    target, holdout, provenance, target_file_sha256 = _unseal(root, config, phase_a_root)
    replay = _replay_sealed_rule(config, target, provenance)
    score = _score(phase_a, config, target)

    derivation_started = time.perf_counter()
    derivation = run_derivation()
    derivation_seconds = time.perf_counter() - derivation_started

    counts = {
        "derivation_steps": len(derivation["steps"]),
        "holdout_rows_predicted": len(phase_a["holdout_predictions"]),
        "holdout_rows_within_tolerance": sum(
            1 for row in score["holdout"] if row["within_declared_tolerance"]
        ),
        "negative_controls_fired": len(derivation["negative_controls"]),
        "post_unseal_generation_events": 0,
        "sealed_rule_rows_replayed": replay["rows_replayed"],
        "stage_receipts": len(phase_a["stages"]),
        "target_fixture_reads": 1,
        "target_fixture_reads_denied_before_unseal": 1,
        "total_declared_views": phase_a["search_space"]["total_declared_views"],
        "views_admitted": phase_a["search_space"]["views_admitted"],
        "views_evaluated": phase_a["search_space"]["views_evaluated"],
        "views_rejected": phase_a["search_space"]["views_rejected"],
        "views_undefined": phase_a["search_space"]["views_undefined"],
    }
    rediscovered = 1 if score["verdict"] == "REDISCOVERED_EXACT" else 0
    decision = (
        "PASS"
        if rediscovered >= config["policies"]["minimum_rediscoveries_for_pass"]
        and all(step["check_status"] == "pass" for step in derivation["steps"])
        else "BLOCK"
    )
    chronology = {
        "denied_probe": denied_probe,
        "events": [
            {"event": "config_and_public_rows_loaded", "sequence": 0, "target_reads": 0},
            {"event": "forbidden_vocabulary_guard_cleared", "sequence": 1, "target_reads": 0},
            {"event": "sealed_targets_guard_entered", "sequence": 2, "target_reads": 0},
            {
                "event": "stage_ladder_and_declared_view_search_completed",
                "sequence": 3,
                "target_reads": 0,
            },
            {"event": "candidate_frozen", "sequence": 4, "target_reads": 0},
            {"event": "holdout_predictions_frozen", "sequence": 5, "target_reads": 0},
            {"event": "instrumented_denied_probe_recorded", "sequence": 6, "target_reads": 0},
            {
                "event": "phase_a_root_sealed",
                "root_sha256": phase_a_root,
                "sequence": 7,
                "target_reads": 0,
            },
            {"event": "atomic_target_unseal", "sequence": 8, "target_reads": 1},
            {"event": "sealed_rule_replayed_on_every_row", "sequence": 9, "target_reads": 1},
            {"event": "commitments_opened_and_scored", "sequence": 10, "target_reads": 1},
            {"event": "derivation_race_run_with_no_data", "sequence": 11, "target_reads": 1},
        ],
        "phase_a_root": phase_a_root,
        "unseal_batches": 1,
    }
    body = {
        "blind_race": {
            **phase_a,
            "sealed_holdout_rows": list(holdout["rows"]),
            "sealed_rule_replay": replay,
            "unseal": score,
        },
        "case_study_id": CASE_STUDY_ID,
        "chronology": chronology,
        "claims": {
            **STATIC_CLAIMS,
            "machine_found_the_relation_unaided": bool(
                rediscovered
                and denied_probe["denied_target_reads"] == 1
                and counts["post_unseal_generation_events"] == 0
            ),
        },
        "counts": counts,
        "decision": decision,
        "derivation": derivation,
        "first_blocker": None if decision == "PASS" else "verdict_threshold_not_met",
        "head_to_head": _head_to_head(phase_a, derivation, score),
        "policies": dict(config["policies"]),
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
        "verdict": score["verdict"],
    }
    receipt = _seal(body)
    return {
        "receipt": receipt,
        "timings": {
            "blind_race_seconds": blind_seconds,
            "derivation_seconds": derivation_seconds,
        },
    }


def build_case_study(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Convenience wrapper returning only the sealed receipt."""

    return build_receipt(root, config_path)["receipt"]


def validate_receipt(
    value: Mapping[str, Any], *, root: Path, config_path: Path | None = None
) -> None:
    """Reject any tamper or environmental drift by exact deterministic replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise BalmerBohrCaseStudyError("case-study receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise BalmerBohrCaseStudyError("case-study receipt seal changed")
    if dict(value) != build_case_study(root, config_path):
        raise BalmerBohrCaseStudyError("case-study receipt exact replay changed")


def _runtime_body(timings: Mapping[str, float]) -> dict[str, Any]:
    total = timings["blind_race_seconds"] + timings["derivation_seconds"]
    return {
        "boundary": (
            "A one-time wall-clock measurement of the build host, written once and never "
            "re-measured. It is not part of any sealed hash, it is not a benchmark, and no "
            "hardware or comparative claim is attached to it. It exists so the head-to-head "
            "has a measured number on our side of the table instead of a blank."
        ),
        "measured_seconds": {
            "blind_race": f"{timings['blind_race_seconds']:.3f}",
            "derivation": f"{timings['derivation_seconds']:.3f}",
            "total": f"{total:.3f}",
        },
        "protocol": WALL_CLOCK_PROTOCOL,
        "receipt_path": RECEIPT_PATH,
        "schema_version": RUNTIME_SCHEMA,
    }


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise BalmerBohrCaseStudyError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _write_measurement(path: Path, value: Mapping[str, Any]) -> None:
    """Write the one-time runtime measurement, and leave an existing one alone."""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Balmer/Bohr head-to-head case study.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=RECEIPT_PATH)
    parser.add_argument("--runtime-output", default=RUNTIME_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output = _resolve(root, args.output)
    if args.validate_checked:
        validate_receipt(_load_json(output), root=root)
        return 0
    built = build_receipt(root)
    _write_immutable(output, built["receipt"])
    _write_measurement(_resolve(root, args.runtime_output), _runtime_body(built["timings"]))
    validate_receipt(built["receipt"], root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
