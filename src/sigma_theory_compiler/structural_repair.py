"""B7 — structural counterexample repair.

Existing counterexample-guided repair moves coefficients inside a fixed structure.
When the structure itself is wrong the run ends at REJECT, which throws away the most
informative object in the whole pipeline: the residual.  A counterexample does not
only say "wrong", it says *how* wrong, at which points, and by how much.

This module consumes that.  Two declared repair strategies act on structure:

**Residual extension.**  Fit a ladder entry, subtract it from the data, and synthesize
the residual as a problem in its own right.  If the residual has structure, the repaired
statement is `base + residual` — a strictly larger term set chosen *because of* the
counterexample rather than declared up front.

**Domain restriction.**  A statement that fails globally may hold exactly on a declared
sub-domain.  Reporting `holds for n >= 3` is ordinary mathematics; silently reporting it
as a global identity is not.  Restrictions are declared, finite, and always carried in
the result.

The anti-overfitting rule is stricter here than in B1/B2, and it has to be: every repair
spends parameters.  Acceptance requires `rows - total_parameters >= min_confirmations`
counted across the *entire composition*, so a repair chain cannot buy a fit by consuming
the evidence that was supposed to test it.

Claim boundary: strategies, depth, and restrictions are declared and finite.  A BLOCK
means "not repairable within this declared budget", never "unrepairable".
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .basis_synthesis import (
    LADDER,
    BasisSynthesisError,
    Term,
    _fraction_data,
    _parse_rows,
    evaluate_basis,
    fit_entry,
    render_basis,
    synthesize_basis,
)
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-structural-repair-result-1.0"

SYSTEM_CAPS = {
    "max_rows": 64,
    "max_repair_depth": 2,
    "min_confirmations": 2,
    "min_restricted_rows": 5,
}

CLAIMS = {
    "coefficient_only_repair": False,
    "corpus_absence_establishes_novelty": False,
    "global_claim_from_restricted_domain": False,
    "repair_budget_is_declared_and_finite": True,
    "residual_structure_must_itself_confirm_on_holdout": True,
    "unbounded_structural_search": False,
}


class StructuralRepairError(ValueError):
    """Raised on malformed input, cap violation, or receipt tamper."""


# ---------------------------------------------------------------------------
# Declared domain restrictions
# ---------------------------------------------------------------------------


def _restrictions() -> tuple[dict[str, Any], ...]:
    """Declared sub-domains, each with an optional reindexing map.

    A restriction may also *relabel* the surviving points.  That matters whenever the
    structure lives in a transformed index rather than in `n`: on the sparse set
    `n = 2^k` a statement is naturally a statement about `k`, and without reindexing the
    search is looking at the wrong variable and will find nothing.
    """

    entries: list[dict[str, Any]] = []
    for threshold in (1, 2, 3):
        entries.append(
            {
                "restriction_id": f"lower_bound_{threshold}",
                "description": f"n >= {threshold}",
                "predicate": ("lower_bound", threshold),
                "reindex": None,
            }
        )
    entries.append(
        {
            "restriction_id": "even_points",
            "description": "n even",
            "predicate": ("parity", 0),
            "reindex": None,
        }
    )
    entries.append(
        {
            "restriction_id": "odd_points",
            "description": "n odd",
            "predicate": ("parity", 1),
            "reindex": None,
        }
    )
    # Sparse geometric sub-domains, reindexed by the exponent.
    for base in (2, 3):
        entries.append(
            {
                "restriction_id": f"geometric_index_{base}",
                "description": f"n = {base}^k, restated in k",
                "predicate": ("geometric", base),
                "reindex": ("exponent", base),
            }
        )
    # Arithmetic progressions, reindexed by position within the progression.
    for modulus in (2, 3, 4):
        for residue in range(modulus):
            entries.append(
                {
                    "restriction_id": f"progression_{modulus}_{residue}",
                    "description": f"n = {modulus}k + {residue}, restated in k",
                    "predicate": ("progression", (modulus, residue)),
                    "reindex": ("progression", (modulus, residue)),
                }
            )
    return tuple(entries)


RESTRICTIONS = _restrictions()


def _applies(predicate: tuple[str, Any], point: int) -> bool:
    kind, argument = predicate
    if kind == "lower_bound":
        return point >= argument
    if kind == "parity":
        return point % 2 == argument
    if kind == "geometric":
        if point < 1:
            return False
        value = point
        while value % argument == 0:
            value //= argument
        return value == 1
    if kind == "progression":
        modulus, residue = argument
        return point % modulus == residue
    raise StructuralRepairError(f"unsupported restriction: {kind}")


def _reindex(mapping: tuple[str, Any] | None, point: int) -> int | None:
    """Relabel a surviving point, or None when the map does not apply."""

    if mapping is None:
        return point
    kind, argument = mapping
    if kind == "exponent":
        if point < 1:
            return None
        exponent, value = 0, point
        while value % argument == 0:
            value //= argument
            exponent += 1
        return exponent if value == 1 else None
    if kind == "progression":
        modulus, residue = argument
        return (point - residue) // modulus
    raise StructuralRepairError(f"unsupported reindex map: {kind}")


# ---------------------------------------------------------------------------
# Residual extension
# ---------------------------------------------------------------------------


def _confirmed_fit(
    terms: tuple[Term, ...], rows: Sequence[tuple[int, Fraction]]
) -> dict[str, Any] | None:
    """Solve an arbitrary term set and require exact agreement on untouched rows.

    A minimally-fitted base absorbs part of any second component, so subtracting it
    does not leave a clean residual.  Extending the *term set* and re-solving keeps
    the system linear and exactly solvable, which is what makes the repair checkable.
    """

    width = len(terms)
    if len(rows) - width < SYSTEM_CAPS["min_confirmations"]:
        return None
    fitted = fit_entry({"family_id": "union", "terms": terms}, rows)
    if fitted is None:
        return None
    solution, fit_points = fitted
    confirmations = 0
    for point, value in rows:
        predicted = evaluate_basis(terms, solution, point)
        if predicted is None or predicted != value:
            return None
        if point not in set(fit_points):
            confirmations += 1
    if confirmations < SYSTEM_CAPS["min_confirmations"]:
        return None
    return {
        "solution": solution,
        "fit_points": fit_points,
        "confirmations": confirmations,
        "expression": render_basis(terms, solution),
    }


def _try_basis_union(rows: Sequence[tuple[int, Fraction]]) -> dict[str, Any] | None:
    """Repair by extending the term set with a second declared family.

    This is a structural change, not a coefficient change: the counterexample causes
    new terms to enter the statement.  Unions are tried in Occam order so the smallest
    repaired term set wins.
    """

    candidates: list[tuple[int, int, int, str, str, tuple[Term, ...]]] = []
    for left_index, left in enumerate(LADDER):
        for right_index, right in enumerate(LADDER):
            if right_index <= left_index:
                continue
            merged: list[Term] = list(left["terms"])
            seen = {term.source for term in merged}
            for term in right["terms"]:
                if term.source not in seen:
                    merged.append(term)
                    seen.add(term.source)
            if len(merged) <= max(len(left["terms"]), len(right["terms"])):
                continue  # one family already subsumes the other
            candidates.append(
                (
                    len(merged),
                    left_index,
                    right_index,
                    left["family_id"],
                    right["family_id"],
                    tuple(merged),
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    for term_count, _, _, left_id, right_id, terms in candidates:
        confirmed = _confirmed_fit(terms, rows)
        if confirmed is None:
            continue
        return {
            "strategy": "basis_union",
            "status": "PASS",
            "extended_from": [left_id, right_id],
            "terms": [term.source for term in terms],
            "expression": confirmed["expression"],
            "total_parameters": term_count,
            "confirmations": confirmed["confirmations"],
            "fit_points": confirmed["fit_points"],
        }
    return None


# ---------------------------------------------------------------------------
# Domain restriction
# ---------------------------------------------------------------------------


def _try_domain_restriction(rows: Sequence[tuple[int, Fraction]]) -> dict[str, Any] | None:
    """Repair by weakening the hypothesis to a declared sub-domain."""

    for restriction in RESTRICTIONS:
        subset: list[dict[str, Any]] = []
        seen: set[int] = set()
        for point, value in rows:
            if not _applies(restriction["predicate"], point):
                continue
            relabelled = _reindex(restriction.get("reindex"), point)
            if relabelled is None or relabelled in seen:
                continue
            seen.add(relabelled)
            subset.append(
                {
                    "point": relabelled,
                    "value": {"numerator": value.numerator, "denominator": value.denominator},
                }
            )
        if len(subset) < SYSTEM_CAPS["min_restricted_rows"]:
            continue
        recovered = synthesize_basis(subset)
        if recovered["decision"] != "PASS":
            continue
        reindexed = restriction.get("reindex") is not None
        return {
            "strategy": "domain_restriction",
            "status": "PASS",
            "restriction_id": restriction["restriction_id"],
            "restricted_domain": restriction["description"],
            "reindexed": reindexed,
            "index_variable": "k" if reindexed else "n",
            "family_id": recovered["result"]["family_id"],
            "expression": recovered["result"]["expression"].replace("n", "k")
            if reindexed
            else recovered["result"]["expression"],
            "total_parameters": recovered["result"]["term_count"],
            "confirmations": recovered["result"]["confirmations"],
            "rows_in_restricted_domain": len(subset),
            "rows_excluded": len(rows) - len(subset),
            "restricted_receipt_sha256": recovered["content_sha256"],
        }
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def repair_structure(rows: Any) -> dict[str, Any]:
    """Attempt declared structural repairs after an unrepaired synthesis fails."""

    try:
        parsed = _parse_rows(rows)
    except BasisSynthesisError as error:
        raise StructuralRepairError(str(error)) from error

    unrepaired = synthesize_basis(
        [
            {
                "point": point,
                "value": {"numerator": value.numerator, "denominator": value.denominator},
            }
            for point, value in parsed
        ]
    )
    attempted: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None

    if unrepaired["decision"] == "PASS":
        accepted = {
            "strategy": "none_required",
            "status": "PASS",
            "family_id": unrepaired["result"]["family_id"],
            "expression": unrepaired["result"]["expression"],
            "total_parameters": unrepaired["result"]["term_count"],
            "confirmations": unrepaired["result"]["confirmations"],
        }
    else:
        extension = _try_basis_union(parsed)
        if extension is not None:
            accepted = extension
        else:
            attempted.append(
                {
                    "strategy": "basis_union",
                    "status": "REJECT",
                    "reason": "no_confirmed_union_of_two_declared_families",
                }
            )
            restriction = _try_domain_restriction(parsed)
            if restriction is not None:
                accepted = restriction
            else:
                attempted.append(
                    {
                        "strategy": "domain_restriction",
                        "status": "REJECT",
                        "reason": "no_confirmed_structure_on_declared_subdomain",
                    }
                )

    body: dict[str, Any] = {
        "claims": CLAIMS,
        "counts": {
            "declared_restrictions": len(RESTRICTIONS),
            "public_rows": len(parsed),
            "strategies_attempted": len(attempted),
        },
        "decision": "PASS" if accepted else "BLOCK",
        "first_blocker": None if accepted else "no_declared_repair_recovered_structure",
        "public_rows": [
            {"point": point, "value": _fraction_data(value)} for point, value in parsed
        ],
        "rejected_strategies": attempted,
        "repair": accepted,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Structural repair over declared residual-extension and domain-restriction "
            "strategies. A PASS reports the repaired statement together with every "
            "restriction it depends on; a restricted-domain result is never promoted to "
            "a global identity. It does not establish novelty or scientific significance, "
            "and a BLOCK means 'not repairable within this declared budget'."
        ),
        "system_caps": SYSTEM_CAPS,
        "unrepaired_decision": unrepaired["decision"],
        "unrepaired_receipt_sha256": unrepaired["content_sha256"],
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_result(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != RESULT_SCHEMA:
        raise StructuralRepairError("result schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise StructuralRepairError("result seal changed")
    rows = [
        {"point": row["point"], "value": row["value"]} for row in value.get("public_rows", [])
    ]
    if dict(value) != repair_structure(rows):
        raise StructuralRepairError("result exact replay changed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Structural counterexample repair (B7).")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, Mapping) else payload
    if args.validate_checked:
        validate_result(json.loads(Path(args.output).read_text(encoding="utf-8")))
        return 0
    result = repair_structure(rows)
    if args.output:
        encoded = canonical_json_bytes(result) + b"\n"
        path = Path(args.output)
        if path.exists() and path.read_bytes() != encoded:
            raise StructuralRepairError("refusing to overwrite immutable receipt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
