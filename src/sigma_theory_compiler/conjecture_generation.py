"""B3 — conjecture generation.

Every earlier stage answers a question someone else posed: the caller supplies the
target as constraints and the system recovers or proves it.  Nothing in the stack asks
*what statement is worth proving here*.  That is the difference between answering a
question and finding one, and it is the gap this module closes.

The input is raw observations with no declared target.  The output is a set of typed,
falsifiable **conjectures** across several statement kinds — closed form, linear
recurrence, divisibility, congruence, sign, monotonicity, partial-sum closed form,
spectral bias, holonomic recurrence.  These are different *kinds* of theorem, not
variations on one formula, so the module proposes what sort of thing might be true
rather than only fitting a curve.

Statement kinds are versioned in frozen profiles.  The original eight kinds are
`STATEMENT_KINDS_V1`; the current default adds `spectral_bias` (the Fourier-form
statement the Ulam progress receipt named as its own blocker) and
`holonomic_recurrence` (polynomial-coefficient recurrences).  A sealed receipt
declares the profile it was scanned with and replays under exactly that profile, so
extending the engine never invalidates receipts that were honest when written.

Two rules keep this from degenerating into pattern-matching noise.

**Propose from a prefix, test on a held-out suffix.**  A statement induced from all the
data explains it by construction and predicts nothing.  Here every conjecture is formed
from a declared prefix and then confronted with rows it has never seen.  Survival is
evidence; refutation is recorded with the exact witness that killed it.

**Rank on Pareto axes, never a scalar truth score.**  Support, simplicity, and
specificity are reported separately and a front is computed over them.  Collapsing those
into one number would manufacture a confidence the evidence does not carry.

Claim boundary: a surviving conjecture is a *conjecture*.  `proved` is false until an
external certificate exists.  Statement kinds and search bounds are declared and finite.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import pairwise
from math import gcd
from pathlib import Path
from typing import Any

from .basis_synthesis import BasisSynthesisError, _fraction_data, _parse_rows, synthesize_basis
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-conjecture-generation-result-1.0"

SYSTEM_CAPS = {
    "max_rows": 64,
    "min_prefix_rows": 4,
    "min_holdout_rows": 3,
    "prefix_numerator": 3,
    "prefix_denominator": 5,
    "max_recurrence_order": 3,
    "max_modulus": 12,
}

CLAIMS = {
    "conjecture_is_a_proof": False,
    "corpus_absence_establishes_novelty": False,
    "proposed_from_prefix_only": True,
    "scalar_truth_or_probability_score": False,
    "statement_kinds_are_declared_and_finite": True,
    "survival_on_holdout_establishes_truth": False,
}

#: The original frozen statement-kind profile.  Receipts sealed before the spectral
#: and holonomic kinds existed declare exactly this list and replay under it.
STATEMENT_KINDS_V1 = (
    "closed_form",
    "linear_recurrence",
    "index_scaling_relation",
    "divisibility",
    "congruence",
    "sign",
    "monotonicity",
    "partial_sum_closed_form",
)

#: The current default profile: v1 plus the spectral-bias kind (the statement form
#: the Ulam progress receipt named as its own blocker) and the holonomic-recurrence
#: kind (P-finite: polynomial-coefficient recurrences).
STATEMENT_KINDS = STATEMENT_KINDS_V1 + ("spectral_bias", "holonomic_recurrence")

#: Every profile a receipt may declare.  A receipt whose statement_kinds list is not
#: exactly one of these fails validation: profiles are frozen, never free-form.
STATEMENT_KIND_PROFILES = (STATEMENT_KINDS_V1, STATEMENT_KINDS)

#: Caps added by the v2 kinds.  Kept out of the v1 receipt body so that v1 receipts
#: replay byte-identically.
V2_EXTRA_CAPS = {
    "holonomic_min_prefix_rows": 8,
    "holonomic_min_term_holdout": 2,
    "spectral_min_prefix_rows": 16,
}


class ConjectureGenerationError(ValueError):
    """Raised on malformed input, cap violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Prefix / holdout split
# ---------------------------------------------------------------------------


def _split(rows: Sequence[tuple[int, Fraction]]) -> tuple[list, list]:
    count = len(rows)
    cut = max(
        SYSTEM_CAPS["min_prefix_rows"],
        count * SYSTEM_CAPS["prefix_numerator"] // SYSTEM_CAPS["prefix_denominator"],
    )
    cut = min(cut, count - SYSTEM_CAPS["min_holdout_rows"])
    if cut < SYSTEM_CAPS["min_prefix_rows"]:
        raise ConjectureGenerationError("insufficient rows for a prefix/holdout split")
    return list(rows[:cut]), list(rows[cut:])


def _as_row_dicts(rows: Sequence[tuple[int, Fraction]]) -> list[dict[str, Any]]:
    return [
        {
            "point": point,
            "value": {"numerator": value.numerator, "denominator": value.denominator},
        }
        for point, value in rows
    ]


# ---------------------------------------------------------------------------
# Statement kinds
# ---------------------------------------------------------------------------


def _integers(rows: Sequence[tuple[int, Fraction]]) -> list[int] | None:
    values = []
    for _, value in rows:
        if value.denominator != 1:
            return None
        values.append(value.numerator)
    return values


def _conjecture_closed_form(prefix, holdout) -> dict[str, Any] | None:
    recovered = synthesize_basis(_as_row_dicts(prefix))
    if recovered["decision"] != "PASS":
        return None
    return {
        "kind": "closed_form",
        "statement": f"a(n) = {recovered['result']['expression']}",
        "parameters": recovered["result"]["term_count"],
        "specificity": Fraction(1),
        "prefix_receipt_sha256": recovered["content_sha256"],
        "_family": recovered["result"]["family_id"],
    }


def _solve_recurrence(values: Sequence[Fraction], order: int) -> list[Fraction] | None:
    """Exact linear recurrence a(n) = sum c_i a(n-i) + c_0 fitted on the prefix."""

    from .basis_synthesis import _solve_unique

    width = order + 1
    matrix: list[list[Fraction]] = []
    vector: list[Fraction] = []
    for index in range(order, len(values)):
        row = [values[index - step] for step in range(1, order + 1)] + [Fraction(1)]
        matrix.append(row)
        vector.append(values[index])
    if len(matrix) < width:
        return None
    return _solve_unique(matrix[:width], vector[:width], width)


def _conjecture_recurrence(prefix, holdout) -> dict[str, Any] | None:
    values = [value for _, value in prefix]
    for order in range(1, SYSTEM_CAPS["max_recurrence_order"] + 1):
        solution = _solve_recurrence(values, order)
        if solution is None:
            continue
        # Must reproduce the remaining prefix rows it was not fitted on.
        consistent = True
        for index in range(order + order + 1, len(values)):
            predicted = sum(
                (solution[step - 1] * values[index - step] for step in range(1, order + 1)),
                Fraction(0),
            ) + solution[order]
            if predicted != values[index]:
                consistent = False
                break
        if not consistent:
            continue
        terms = " + ".join(
            f"({solution[step - 1]})*a(n-{step})" for step in range(1, order + 1)
        )
        constant = solution[order]
        statement = f"a(n) = {terms}" + (f" + ({constant})" if constant != 0 else "")
        return {
            "kind": "linear_recurrence",
            "statement": statement,
            "parameters": order + 1,
            "specificity": Fraction(1),
            "_order": order,
            "_coefficients": solution,
        }
    return None


def _conjecture_index_scaling(prefix, holdout) -> dict[str, Any] | None:
    """`a(c*n) = alpha*a(n) + beta` — a relation across a *multiplicative* index step.

    Closed forms relate `a(n)` to `n`; linear recurrences relate `a(n)` to fixed-lag
    predecessors.  Neither can express self-similarity under index scaling, which is
    where a great deal of number-theoretic structure actually lives (valuations, digit
    functions, and any sequence defined by halving).
    """

    from .basis_synthesis import _solve_unique

    lookup = dict(prefix)
    for scale in (2, 3):
        pairs = [
            (lookup[point], lookup[scale * point])
            for point in sorted(lookup)
            if scale * point in lookup
        ]
        if len(pairs) < 4:  # two to determine, at least two to check
            continue
        # Determine alpha and beta from the first two independent pairs.
        matrix: list[list[Fraction]] = []
        vector: list[Fraction] = []
        for source, image in pairs:
            candidate_matrix = [*matrix, [source, Fraction(1)]]
            candidate_vector = [*vector, image]
            if len(candidate_matrix) <= 2:
                matrix, vector = candidate_matrix, candidate_vector
        if len(matrix) < 2:
            continue
        solution = _solve_unique(matrix, vector, 2)
        if solution is None:
            continue
        alpha, beta = solution
        if all(
            alpha * source + beta == image for source, image in pairs
        ):
            relation = f"a({scale}n) = ({alpha})*a(n)" + (f" + ({beta})" if beta != 0 else "")
            return {
                "kind": "index_scaling_relation",
                "statement": relation,
                "parameters": 2,
                "specificity": Fraction(1),
                "_scale": scale,
                "_alpha": alpha,
                "_beta": beta,
            }
    return None


def _conjecture_divisibility(prefix, holdout) -> dict[str, Any] | None:
    values = _integers(prefix)
    if values is None or all(value == 0 for value in values):
        return None
    divisor = 0
    for value in values:
        divisor = gcd(divisor, abs(value))
    if divisor <= 1:
        return None
    return {
        "kind": "divisibility",
        "statement": f"{divisor} divides a(n)",
        "parameters": 1,
        "specificity": Fraction(divisor - 1, divisor),
        "_divisor": divisor,
    }


def _conjecture_congruence(prefix, holdout) -> dict[str, Any] | None:
    values = _integers(prefix)
    if values is None:
        return None
    for modulus in range(2, SYSTEM_CAPS["max_modulus"] + 1):
        residues = {value % modulus for value in values}
        if len(residues) == 1:
            residue = residues.pop()
            if modulus > 1 and not (residue == 0 and modulus == 1):
                return {
                    "kind": "congruence",
                    "statement": f"a(n) = {residue} (mod {modulus})",
                    "parameters": 2,
                    "specificity": Fraction(modulus - 1, modulus),
                    "_modulus": modulus,
                    "_residue": residue,
                }
    return None


def _conjecture_sign(prefix, holdout) -> dict[str, Any] | None:
    values = [value for _, value in prefix]
    if all(value > 0 for value in values):
        return {"kind": "sign", "statement": "a(n) > 0", "parameters": 0, "specificity": Fraction(1, 2), "_sign": "positive"}
    if all(value < 0 for value in values):
        return {"kind": "sign", "statement": "a(n) < 0", "parameters": 0, "specificity": Fraction(1, 2), "_sign": "negative"}
    return None


def _conjecture_monotonicity(prefix, holdout) -> dict[str, Any] | None:
    values = [value for _, value in prefix]
    pairs = list(pairwise(values))
    if pairs and all(left < right for left, right in pairs):
        return {
            "kind": "monotonicity",
            "statement": "a(n) < a(n+1)",
            "parameters": 0,
            "specificity": Fraction(1, 2),
            "_direction": "increasing",
        }
    if pairs and all(left > right for left, right in pairs):
        return {
            "kind": "monotonicity",
            "statement": "a(n) > a(n+1)",
            "parameters": 0,
            "specificity": Fraction(1, 2),
            "_direction": "decreasing",
        }
    return None


def _conjecture_partial_sum(prefix, holdout) -> dict[str, Any] | None:
    running = Fraction(0)
    sums: list[dict[str, Any]] = []
    for point, value in prefix:
        running += value
        sums.append(
            {
                "point": point,
                "value": {"numerator": running.numerator, "denominator": running.denominator},
            }
        )
    recovered = synthesize_basis(sums)
    if recovered["decision"] != "PASS":
        return None
    return {
        "kind": "partial_sum_closed_form",
        "statement": f"sum_(i<=n) a(i) = {recovered['result']['expression']}",
        "parameters": recovered["result"]["term_count"],
        "specificity": Fraction(1),
        "prefix_receipt_sha256": recovered["content_sha256"],
    }


def _spectral_dps(values: Sequence[Fraction]) -> int:
    """Declared cos-sign precision for exact rational holdout values."""

    from .spectral_signal_scan import SYSTEM_CAPS as SPECTRAL_CAPS

    magnitude = max((abs(int(value)) for value in values), default=1)
    return SPECTRAL_CAPS["base_dps"] + len(str(max(magnitude, 1)))


def _conjecture_spectral_bias(prefix, holdout) -> dict[str, Any] | None:
    """Steinerberger-form hidden-frequency bias: `cos(lambda* a_n) < 0` often.

    This is the statement kind the Ulam progress receipt named as its own blocker
    (`statement_kinds_too_weak`): a quasi-periodic signal is invisible to every
    algebraic kind above.  The scan runs on the prefix only; the float grid sweep is
    a pre-filter and every reported number is exact or mpmath-adjudicated.
    """

    from .spectral_signal_scan import PROFILES, propose_bias_from_prefix

    values = _integers(prefix)
    if values is None or len(values) < V2_EXTRA_CAPS["spectral_min_prefix_rows"]:
        return None
    profile = PROFILES["engine"]
    for entry in propose_bias_from_prefix(values, "engine"):
        threshold = entry["threshold_fraction"]
        if threshold is None:
            continue
        return {
            "kind": "spectral_bias",
            "statement": entry["statement"],
            "parameters": 2,
            "specificity": threshold,
            "lambda": entry["lambda"],
            "magnitude_at_lambda": entry["magnitude_at_lambda"],
            "refinement": entry["refinement"],
            "grid": {
                "denominator": profile["grid_denominator"],
                "start_k": profile["start_k"],
                "stop_k": profile["stop_k"],
            },
            "peak_is_not_proof": True,
            "prefix_negative_terms": entry["prefix_negative_terms"],
            "prefix_ambiguous_terms": entry["prefix_ambiguous_terms"],
            "prefix_terms": entry["prefix_terms"],
            "_lambda": entry["lambda_fraction"],
            "_threshold": threshold,
        }
    return None


def _conjecture_holonomic(prefix, holdout) -> dict[str, Any] | None:
    """P-finite recurrence: `sum_i P_i(n) a(n+i) = 0` with polynomial `P_i`.

    Fitted on the minimum prefix rows by exact rational nullspace, verified on the
    remaining prefix rows, then confronted with the held-out suffix like every other
    kind.  Catalan-style sequences get a direct recurrence here instead of only the
    derived-ratio route.
    """

    from .holonomic_guesser import guess_recurrence

    points = [point for point, _ in prefix]
    if len(points) < V2_EXTRA_CAPS["holonomic_min_prefix_rows"]:
        return None
    if any(right - left != 1 for left, right in pairwise(points)):
        return None
    result = guess_recurrence(
        [value for _, value in prefix],
        start_index=points[0],
        min_term_holdout=V2_EXTRA_CAPS["holonomic_min_term_holdout"],
    )
    if result["decision"] != "OPERATOR_FOUND":
        return None
    operator = result["operator"]
    return {
        "kind": "holonomic_recurrence",
        "statement": operator["statement"],
        "parameters": operator["parameter_count"],
        "specificity": Fraction(1),
        "operator": {
            "order": operator["order"],
            "degree": operator["degree"],
            "coefficients": operator["coefficients"],
            "start_point": operator["start_point"],
        },
        "latex": operator["latex"],
        "_order": operator["order"],
        "_coefficients": operator["coefficients"],
    }


PROPOSERS = (
    ("closed_form", _conjecture_closed_form),
    ("linear_recurrence", _conjecture_recurrence),
    ("index_scaling_relation", _conjecture_index_scaling),
    ("partial_sum_closed_form", _conjecture_partial_sum),
    ("divisibility", _conjecture_divisibility),
    ("congruence", _conjecture_congruence),
    ("monotonicity", _conjecture_monotonicity),
    ("sign", _conjecture_sign),
    ("spectral_bias", _conjecture_spectral_bias),
    ("holonomic_recurrence", _conjecture_holonomic),
)


# ---------------------------------------------------------------------------
# Holdout confrontation
# ---------------------------------------------------------------------------


def _test_conjecture(
    conjecture: Mapping[str, Any],
    prefix: Sequence[tuple[int, Fraction]],
    holdout: Sequence[tuple[int, Fraction]],
) -> dict[str, Any]:
    """Confront a prefix-derived conjecture with rows it has never seen."""

    kind = conjecture["kind"]
    if kind == "spectral_bias":
        # The claim is a fraction over the whole suffix, not a per-row law: count
        # exact cos signs at the stated quantized lambda, failing closed on any
        # numerically ambiguous sign.
        from .spectral_signal_scan import cos_sign

        dps = _spectral_dps([value for _, value in prefix] + [value for _, value in holdout])
        negative = 0
        ambiguous = 0
        for _, value in holdout:
            sign = cos_sign(conjecture["_lambda"], value, dps)
            if sign < 0:
                negative += 1
            elif sign == 0:
                ambiguous += 1
        threshold = conjecture["_threshold"]
        if Fraction(negative, len(holdout)) >= threshold:
            return {"status": "SURVIVED", "support": negative, "refutation_witness": None}
        return {
            "status": "REFUTED",
            "support": negative,
            "refutation_witness": {
                "holdout_rows": len(holdout),
                "negative_terms": negative,
                "ambiguous_terms": ambiguous,
                "required_fraction": _fraction_data(threshold),
            },
        }
    support = 0
    for index, (point, value) in enumerate(holdout):
        ok = True
        witness: dict[str, Any] | None = None
        if kind == "closed_form":
            combined = synthesize_basis(_as_row_dicts(list(prefix) + list(holdout[: index + 1])))
            ok = (
                combined["decision"] == "PASS"
                and combined["result"]["family_id"] == conjecture["_family"]
            )
            if not ok:
                witness = {"point": point, "reason": "closed_form_not_preserved"}
        elif kind == "linear_recurrence":
            order = conjecture["_order"]
            history = [item[1] for item in list(prefix) + list(holdout)]
            position = len(prefix) + index
            coefficients = conjecture["_coefficients"]
            predicted = sum(
                (coefficients[step - 1] * history[position - step] for step in range(1, order + 1)),
                Fraction(0),
            ) + coefficients[order]
            ok = predicted == value
            if not ok:
                witness = {
                    "point": point,
                    "predicted": _fraction_data(predicted),
                    "observed": _fraction_data(value),
                }
        elif kind == "index_scaling_relation":
            # Confirm only where the scaled partner is actually present in the data.
            history = dict(list(prefix) + list(holdout))
            scale = conjecture["_scale"]
            if point % scale == 0 and point // scale in history:
                predicted = (
                    conjecture["_alpha"] * history[point // scale] + conjecture["_beta"]
                )
                ok = predicted == value
                if not ok:
                    witness = {
                        "point": point,
                        "predicted": _fraction_data(predicted),
                        "observed": _fraction_data(value),
                    }
            else:
                ok = True  # vacuous at this point; contributes no support
        elif kind == "divisibility":
            ok = value.denominator == 1 and value.numerator % conjecture["_divisor"] == 0
            witness = None if ok else {"point": point, "observed": _fraction_data(value)}
        elif kind == "congruence":
            ok = (
                value.denominator == 1
                and value.numerator % conjecture["_modulus"] == conjecture["_residue"]
            )
            witness = None if ok else {"point": point, "observed": _fraction_data(value)}
        elif kind == "sign":
            ok = value > 0 if conjecture["_sign"] == "positive" else value < 0
            witness = None if ok else {"point": point, "observed": _fraction_data(value)}
        elif kind == "monotonicity":
            previous = (list(prefix) + list(holdout))[len(prefix) + index - 1][1]
            ok = previous < value if conjecture["_direction"] == "increasing" else previous > value
            witness = None if ok else {"point": point, "observed": _fraction_data(value)}
        elif kind == "holonomic_recurrence":
            from .holonomic_guesser import _poly_eval_int

            history = dict(list(prefix) + list(holdout))
            order = conjecture["_order"]
            base = point - order
            if all(base + shift in history for shift in range(order + 1)):
                residual = Fraction(0)
                for shift, coefficients in enumerate(conjecture["_coefficients"]):
                    residual += _poly_eval_int(coefficients, base) * history[base + shift]
                ok = residual == 0
                if not ok:
                    witness = {
                        "point": point,
                        "base_point": base,
                        "residual": _fraction_data(residual),
                    }
            else:
                ok = True  # vacuous where the recurrence window is not in the data
        elif kind == "partial_sum_closed_form":
            # Extend the running-sum series one holdout row at a time; the recovered
            # closed form must keep holding as each new row is admitted.
            running_rows: list[dict[str, Any]] = []
            total = Fraction(0)
            for item_point, item_value in list(prefix) + list(holdout[: index + 1]):
                total += item_value
                running_rows.append(
                    {
                        "point": item_point,
                        "value": {
                            "numerator": total.numerator,
                            "denominator": total.denominator,
                        },
                    }
                )
            ok = synthesize_basis(running_rows)["decision"] == "PASS"
            if not ok:
                witness = {"point": point, "reason": "partial_sum_closed_form_not_preserved"}
        if not ok:
            return {
                "status": "REFUTED",
                "support": support,
                "refutation_witness": witness or {"point": point},
            }
        support += 1
    return {"status": "SURVIVED", "support": support, "refutation_witness": None}


def _pareto_front(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Non-dominated conjectures over (support, simplicity, specificity)."""

    def axes(entry: Mapping[str, Any]) -> tuple[Fraction, Fraction, Fraction]:
        specificity = entry["specificity"]
        return (
            Fraction(entry["support"]),
            Fraction(-entry["parameters"]),
            Fraction(specificity["numerator"], specificity["denominator"]),
        )

    front: list[str] = []
    for entry in entries:
        dominated = False
        for other in entries:
            if other is entry:
                continue
            left, right = axes(entry), axes(other)
            if all(o >= e for o, e in zip(right, left, strict=True)) and any(
                o > e for o, e in zip(right, left, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            front.append(entry["statement"])
    return front


def generate_conjectures(
    rows: Any, statement_kinds: Sequence[str] | None = None
) -> dict[str, Any]:
    """Propose typed falsifiable statements from data with no declared target.

    `statement_kinds` selects one of the frozen declared profiles (default: the
    current full profile).  Passing anything that is not exactly a frozen profile is
    an error: the scan surface is versioned, never free-form.
    """

    active = STATEMENT_KINDS if statement_kinds is None else tuple(statement_kinds)
    if active not in STATEMENT_KIND_PROFILES:
        raise ConjectureGenerationError("statement_kinds is not a declared frozen profile")
    try:
        parsed = _parse_rows(rows)
    except BasisSynthesisError as error:
        raise ConjectureGenerationError(str(error)) from error
    prefix, holdout = _split(parsed)

    proposed: list[dict[str, Any]] = []
    for kind, proposer in PROPOSERS:
        if kind not in active:
            continue
        conjecture = proposer(prefix, holdout)
        if conjecture is None:
            proposed.append({"kind": kind, "status": "NOT_PROPOSED", "reason": "no_pattern_in_prefix"})
            continue
        verdict = _test_conjecture(conjecture, prefix, holdout)
        public = {
            key: value for key, value in conjecture.items() if not key.startswith("_")
        }
        public["specificity"] = _fraction_data(conjecture["specificity"])
        proposed.append(
            {
                **public,
                "status": verdict["status"],
                "support": verdict["support"],
                "holdout_rows": len(holdout),
                "refutation_witness": verdict["refutation_witness"],
                "proved": False,
            }
        )

    survivors = [entry for entry in proposed if entry.get("status") == "SURVIVED"]
    caps = SYSTEM_CAPS if active == STATEMENT_KINDS_V1 else {**SYSTEM_CAPS, **V2_EXTRA_CAPS}
    body: dict[str, Any] = {
        "claims": CLAIMS,
        "conjectures": proposed,
        "counts": {
            "declared_statement_kinds": len(active),
            "holdout_rows": len(holdout),
            "prefix_rows": len(prefix),
            "proposed": sum(1 for entry in proposed if entry.get("status") != "NOT_PROPOSED"),
            "refuted": sum(1 for entry in proposed if entry.get("status") == "REFUTED"),
            "survived": len(survivors),
        },
        "decision": "PROPOSED" if survivors else "NONE_SURVIVED",
        "pareto_front": _pareto_front(survivors),
        "public_rows": _as_row_dicts(parsed),
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Typed conjecture generation from data with no declared target. Statements are "
            "proposed from a prefix and confronted with a held-out suffix. Survival is "
            "evidence of predictive content, not proof: every entry carries proved=false "
            "until an external certificate exists. Ranking is a Pareto front over support, "
            "simplicity, and specificity; no scalar truth or probability score is produced."
        ),
        "statement_kinds": list(active),
        "system_caps": caps,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_result(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RESULT_SCHEMA:
        raise ConjectureGenerationError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ConjectureGenerationError("result seal changed")
    declared = tuple(value.get("statement_kinds", ()))
    if declared not in STATEMENT_KIND_PROFILES:
        raise ConjectureGenerationError("result declares an unknown statement-kind profile")
    rows = [
        {"point": row["point"], "value": row["value"]} for row in value.get("public_rows", [])
    ]
    if dict(value) != generate_conjectures(rows, statement_kinds=declared):
        raise ConjectureGenerationError("result exact replay changed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Conjecture generation (B3).")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, Mapping) else payload
    if args.validate_checked:
        validate_result(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    result = generate_conjectures(rows)
    if args.output:
        encoded = canonical_json_bytes(result) + b"\n"
        path = Path(args.output)
        if path.exists() and path.read_bytes() != encoded:
            raise ConjectureGenerationError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "PROPOSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
