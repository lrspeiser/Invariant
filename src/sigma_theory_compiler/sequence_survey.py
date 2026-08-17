"""DG4/DG6 — point both sequence instruments at every eligible problem in the queue.

The spectral scanner recovered Steinerberger's Ulam frequency to seven decimals and the
holonomic guesser recovers Catalan, Motzkin, derangements and factorials on demand.  Both
had been aimed at a handful of sequences by hand.  A capability that has only ever been
pointed where someone expected an answer cannot say what it has *not* found, so this
module runs both instruments across the whole sealed queue and seals one receipt per
(instrument, problem) — survivors and refutations alike.

Four rules keep the survey honest.

**Ineligibility is typed, never silent.**  A problem that is not scanned records why:
``INELIGIBLE_KIND`` (not a row-producing machine form), ``INELIGIBLE_MISSING_GENERATOR``
(the sealed synthetic holdout, whose generator is deliberately withheld from the discovery
side), or ``INELIGIBLE_INSUFFICIENT_TERMS`` with the exact term count the generator could
produce inside its declared caps.  "Nobody scanned it" is recoverable from the record.

**Builtin knowledge is declared per sequence, before the scan.**
:data:`BUILTIN_KNOWLEDGE` states, for each problem, whether this repository knows of a
published spectral signal and of a published exact term structure.  Ulam's entry names
Steinerberger 2017; the rest say "not to our knowledge" — and that phrase is a statement
about *this repository's builtin knowledge only*.  It is not a literature search, it is
not evidence of absence, and it is explicitly not a novelty claim.

**The discovery-condition flags are computed, not asserted.**  :func:`build_summary`
reads the sealed receipts back off disk and derives DG4's and DG6's conditions from what
they say, crossed with the declared knowledge table.  No caller can hand the summary a
verdict.

**A survivor is a candidate.**  Every surviving spectral bias also carries
``phase_turns``: how many full turns of ``cos(lambda a(n))`` the sequence's own value
range spans at the stated lambda.  Below one turn a negative-cosine majority is a
consequence of the values being bounded and clustered, not of hidden quasi-periodicity,
and the receipt says so.  The flag is a diagnostic on the candidate, never a filter that
quietly deletes it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import factorial
from pathlib import Path
from typing import Any

import mpmath

from .discovery_scheduler import GENERATOR_REGISTRY, _generator_name, _row_count_request
from .holonomic_guesser import SYSTEM_CAPS as HOLONOMIC_CAPS
from .holonomic_guesser import guess_receipt
from .holonomic_guesser import validate_receipt as validate_holonomic_receipt
from .problem_queue import load_queue
from .sigma_core import canonical_json_bytes, canonical_sha256
from .spectral_signal_scan import SYSTEM_CAPS as SPECTRAL_CAPS
from .spectral_signal_scan import cos_sign, lambda_decimal, scan_receipt
from .spectral_signal_scan import validate_receipt as validate_spectral_receipt

SUMMARY_SCHEMA = "invariant-sequence-survey-summary-1.0"

#: DG4's declared eligibility bar: a sequence is scanned spectrally only when the
#: generator can emit at least this many exact terms inside its declared caps.
SPECTRAL_MIN_TERMS = 64

#: DG6 scans everything the guesser will accept; the floor is the guesser's own cap.
HOLONOMIC_MIN_TERMS = HOLONOMIC_CAPS["min_terms"]

#: Grid profile for the survey.  ``dense`` is the ~313k-point sweep over [0.01, pi) that
#: the committed Ulam exemplar used, so this survey's Ulam receipt is comparable to it.
SURVEY_PROFILE = "dense"

#: Exact terms requested per problem.  Each number is at or below both the generator's
#: declared cap and the instruments' row caps, so one row set feeds both instruments and
#: the two receipts for a problem are statements about the same terms.  Ulam is pinned at
#: 5000 to match the committed exemplar exactly.
SURVEY_TERMS: dict[str, int] = {
    "aliquot_276": 6000,
    "catalan_like_recurrence_holdout": 48,
    "collatz_stopping_time": 6000,
    "continued_fraction_e_pattern": 6000,
    "gilbreath_conjecture": 500,
    "lychrel_196": 6000,
    "prime_gap_polynomial": 6000,
    "recaman_coverage": 6000,
    "singmaster_conjecture": 6000,
    "twin_prime_infinitude": 7,
    "ulam_sequence_structure": 5000,
}

_NOT_TO_OUR_KNOWLEDGE = (
    "no published spectral signal for this sequence is present in this repository's "
    "builtin knowledge. That is a statement about this repository only: it is not a "
    "literature search, not evidence of absence, and explicitly not a novelty claim."
)

#: Per-problem builtin knowledge, declared before the scan.  ``published_spectral_signal``
#: and ``known_exact_structure`` are the two facts the DG4 and DG6 discovery conditions
#: turn on; ``None`` means "not present in this repository's builtin knowledge", which is
#: never the same as "does not exist".
BUILTIN_KNOWLEDGE: dict[str, dict[str, str | None]] = {
    "aliquot_276": {
        "published_spectral_signal": None,
        "known_exact_structure": None,
        "sequence": "aliquot trajectory of 276 (OEIS A008892 head), s(n) = sigma(n) - n",
    },
    "catalan_like_recurrence_holdout": {
        "published_spectral_signal": None,
        "known_exact_structure": (
            "a sealed synthetic holdout: the generating recurrence is known to the "
            "sealing side and deliberately withheld from the discovery side"
        ),
        "sequence": "sealed synthetic Catalan-like recurrence",
    },
    "collatz_stopping_time": {
        "published_spectral_signal": None,
        "known_exact_structure": None,
        "sequence": "total stopping time sigma(n) of the Collatz map (OEIS A006577)",
    },
    "continued_fraction_e_pattern": {
        "published_spectral_signal": None,
        "known_exact_structure": (
            "Euler (1737) proved e = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]: the terms follow "
            "the (1, 2k, 1) block pattern exactly (OEIS A003417), so any annihilating "
            "operator here restates a known exact structure"
        ),
        "sequence": "continued fraction terms of e (OEIS A003417)",
    },
    "gilbreath_conjecture": {
        "published_spectral_signal": None,
        "known_exact_structure": (
            "the emitted rows are constant 1: Gilbreath's conjecture is computationally "
            "verified far beyond this window (Odlyzko 1993, to 10^13 rows), so every "
            "leading term here is 1 and any annihilator is the trivial constant operator"
        ),
        "sequence": "leading terms of the iterated absolute prime difference rows",
    },
    "lychrel_196": {
        "published_spectral_signal": None,
        "known_exact_structure": None,
        "sequence": "digit lengths of the base-10 reverse-and-add trajectory of 196",
    },
    "prime_gap_polynomial": {
        "published_spectral_signal": None,
        "known_exact_structure": None,
        "sequence": "prime gaps p(n+1) - p(n) (OEIS A001223)",
    },
    "recaman_coverage": {
        "published_spectral_signal": None,
        "known_exact_structure": None,
        "sequence": "Recaman's sequence (OEIS A005132)",
    },
    "singmaster_conjecture": {
        "published_spectral_signal": None,
        "known_exact_structure": None,
        "sequence": (
            "multiplicity N(t) of t in Pascal's triangle (OEIS A003016), indexed from "
            "row 1 so that row n carries N(n + 1)"
        ),
    },
    "twin_prime_infinitude": {
        "published_spectral_signal": None,
        "known_exact_structure": None,
        "sequence": "pi_2(10^k), twin prime pairs below 10^k (OEIS A007508)",
    },
    "ulam_sequence_structure": {
        "published_spectral_signal": (
            "Steinerberger, Experimental Mathematics 26 (2017): the Ulam sequence has a "
            "hidden frequency lambda ~= 2.5714474995 with cos(lambda a(n)) < 0 for "
            "almost every term. A surviving signal here is a rediscovery, not a finding."
        ),
        "known_exact_structure": None,
        "sequence": "Ulam numbers U(1, 2) (OEIS A002858)",
    },
}

#: The four declared holonomic controls.  Each is a classical P-finite sequence with a
#: textbook operator; if the guesser stops recovering them, the survey's NO_ANNIHILATOR
#: rows stop meaning anything and the run is void.
CONTROL_TERMS = 24

CLAIMS = {
    "builtin_knowledge_absence_establishes_novelty": False,
    "controls_gate_the_negative_results": True,
    "discovery_conditions_computed_from_receipts": True,
    "eligibility_is_typed_never_silent": True,
    "peak_is_not_proof": True,
    "scalar_truth_or_probability_score": False,
    "survival_on_holdout_establishes_truth": False,
}

SUMMARY_SCOPE = (
    "One survey pass: every sealed-queue problem run through the spectral scanner (DG4) "
    "and the holonomic guesser (DG6), or recorded ineligible with a typed reason. The "
    "summary binds each instrument receipt by content hash and computes DG4's and DG6's "
    "discovery conditions from those receipts crossed with a knowledge table declared "
    "before the scan. Completing the survey establishes no novelty, correctness, or "
    "significance; a surviving spectral bias is a candidate signal on a declared finite "
    "grid and an annihilating operator is a guess, not a proof."
)


class SequenceSurveyError(ValueError):
    """Raised on a malformed survey input, a missing receipt, or a summary tamper."""


# ---------------------------------------------------------------------------
# Rows for one queue entry
# ---------------------------------------------------------------------------


def _control_catalan(count: int) -> list[int]:
    terms = [1]
    for n in range(count - 1):
        terms.append(terms[-1] * 2 * (2 * n + 1) // (n + 2))
    return terms


def _control_motzkin(count: int) -> list[int]:
    terms = [1, 1]
    for n in range(1, count - 1):
        terms.append((terms[n] * (2 * n + 3) + terms[n - 1] * 3 * n) // (n + 3))
    return terms


def _control_derangements(count: int) -> list[int]:
    terms = [1, 0]
    for n in range(2, count):
        terms.append((n - 1) * (terms[n - 1] + terms[n - 2]))
    return terms[:count]


#: control id -> (exact terms, the textbook operator the guesser must recover)
CONTROLS: dict[str, tuple[list[int], str]] = {
    "catalan": (_control_catalan(CONTROL_TERMS), "OEIS A000108"),
    "derangements": (_control_derangements(CONTROL_TERMS), "OEIS A000166"),
    "factorial": ([factorial(n) for n in range(CONTROL_TERMS)], "OEIS A000142"),
    "motzkin": (_control_motzkin(CONTROL_TERMS), "OEIS A001006"),
}


def survey_rows(entry: Mapping[str, Any]) -> tuple[list[dict[str, int]], dict[str, Any]]:
    """Exact rows for one queue entry at its declared survey term count.

    Returns ``(rows, provenance)``.  ``rows`` is empty when the entry produces no rows at
    all; ``provenance`` always records the generator, the requested count, and every typed
    truncation the generator emitted.
    """

    machine_form = entry["machine_form"]
    kind = machine_form["kind"]
    if kind not in ("integer_trajectory", "sequence_rows"):
        return [], {
            "generator": None,
            "reason": "INELIGIBLE_KIND",
            "detail": f"machine form kind {kind!r} produces no sequence rows",
            "truncations": [],
        }
    name = _generator_name(machine_form)
    generator = GENERATOR_REGISTRY.get(name)
    if generator is None:
        return [], {
            "generator": name,
            "reason": "INELIGIBLE_MISSING_GENERATOR",
            "detail": f"no row generator is registered for {name!r}",
            "truncations": [],
        }
    requested = SURVEY_TERMS.get(entry["id"], _row_count_request(machine_form))
    key = "max_point" if kind == "sequence_rows" else "max_steps"
    rows, truncations = generator({**machine_form, key: requested})
    return rows, {
        "generator": name,
        "reason": None,
        "detail": None,
        "requested_terms": requested,
        "truncations": [dict(item) for item in truncations],
    }


#: Knowledge for a queue entry that declares no sequence at all (a Diophantine family, a
#: dataset law fit, a module target).  Nothing about a published signal is asserted
#: because no sequence was ever in play.
_NO_SEQUENCE_KNOWLEDGE: dict[str, str | None] = {
    "known_exact_structure": None,
    "published_spectral_signal": None,
    "sequence": None,
}


def _knowledge(problem_id: str, *, required: bool = True) -> dict[str, str | None]:
    """The declared builtin knowledge for a problem.

    ``required`` is true for any problem an instrument actually ran on: a sequence may
    not be scanned unless what this repository knows about it was declared first.
    """

    known = BUILTIN_KNOWLEDGE.get(problem_id)
    if known is None:
        if required:
            raise SequenceSurveyError(f"no builtin knowledge declared for {problem_id!r}")
        return dict(_NO_SEQUENCE_KNOWLEDGE)
    return dict(known)


def spectral_knowledge_note(problem_id: str, *, required: bool = True) -> str | None:
    """The ``builtin_knowledge`` string sealed into this problem's spectral receipt."""

    known = _knowledge(problem_id, required=required)
    if known["sequence"] is None:
        return None
    published = known["published_spectral_signal"]
    return published if published is not None else _NOT_TO_OUR_KNOWLEDGE


# ---------------------------------------------------------------------------
# Running the two instruments
# ---------------------------------------------------------------------------


def _write_immutable(path: Path, value: Mapping[str, Any]) -> bool:
    """Write a sealed receipt.  Returns True when the file was created by this call."""

    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise SequenceSurveyError(f"refusing to overwrite immutable receipt: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return True


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run_spectral_survey(
    root: Path, queue: Mapping[str, Any], *, use_gpu: bool = False
) -> list[dict[str, Any]]:
    """Scan every eligible problem and seal one receipt each under the survey directory."""

    records: list[dict[str, Any]] = []
    for entry in sorted(queue["entries"], key=lambda item: item["id"]):
        problem_id = entry["id"]
        rows, provenance = survey_rows(entry)
        record: dict[str, Any] = {
            "problem_id": problem_id,
            "generator": provenance["generator"],
            "terms": len(rows),
        }
        if provenance["reason"] is not None:
            records.append({**record, "status": provenance["reason"],
                            "detail": provenance["detail"], "receipt_path": None})
            continue
        if len(rows) < SPECTRAL_MIN_TERMS:
            records.append(
                {
                    **record,
                    "status": "INELIGIBLE_INSUFFICIENT_TERMS",
                    "detail": (
                        f"{len(rows)} exact terms are available inside the generator's "
                        f"declared caps; DG4 scans at {SPECTRAL_MIN_TERMS} or more"
                    ),
                    "receipt_path": None,
                }
            )
            continue
        scanned = rows[: SPECTRAL_CAPS["max_rows"]]
        receipt = scan_receipt(
            scanned,
            sequence_label=(
                f"{_knowledge(problem_id)['sequence']}: {len(scanned)} exact terms from "
                f"the {provenance['generator']} generator"
            ),
            profile_name=SURVEY_PROFILE,
            use_gpu=use_gpu,
            builtin_knowledge=spectral_knowledge_note(problem_id),
        )
        path = root / "runs" / "math" / "spectral" / "survey" / f"{problem_id}.json"
        _write_immutable(path, receipt)
        records.append(
            {
                **record,
                "terms": len(scanned),
                "status": "SCANNED",
                "detail": None,
                "receipt_path": _relative(root, path),
            }
        )
    return records


def run_holonomic_survey(root: Path, queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Guess an annihilator for every eligible problem and seal one receipt each."""

    records: list[dict[str, Any]] = []
    for entry in sorted(queue["entries"], key=lambda item: item["id"]):
        problem_id = entry["id"]
        rows, provenance = survey_rows(entry)
        record: dict[str, Any] = {
            "problem_id": problem_id,
            "generator": provenance["generator"],
            "terms": len(rows),
        }
        if provenance["reason"] is not None:
            records.append({**record, "status": provenance["reason"],
                            "detail": provenance["detail"], "receipt_path": None})
            continue
        if len(rows) < HOLONOMIC_MIN_TERMS:
            records.append(
                {
                    **record,
                    "status": "INELIGIBLE_INSUFFICIENT_TERMS",
                    "detail": (
                        f"{len(rows)} exact terms are available inside the generator's "
                        f"declared caps; the guesser needs {HOLONOMIC_MIN_TERMS}"
                    ),
                    "receipt_path": None,
                }
            )
            continue
        guessed = rows[: HOLONOMIC_CAPS["max_terms"]]
        receipt = guess_receipt(
            guessed,
            sequence_label=(
                f"{_knowledge(problem_id)['sequence']}: {len(guessed)} exact terms from "
                f"the {provenance['generator']} generator"
            ),
        )
        path = root / "runs" / "math" / "holonomic" / "survey" / f"{problem_id}.json"
        _write_immutable(path, receipt)
        records.append(
            {
                **record,
                "terms": len(guessed),
                "status": "GUESSED",
                "detail": None,
                "receipt_path": _relative(root, path),
            }
        )
    return records


def run_controls(root: Path) -> list[dict[str, Any]]:
    """Recover the four declared classical operators and seal one receipt each."""

    records: list[dict[str, Any]] = []
    for control_id in sorted(CONTROLS):
        terms, citation = CONTROLS[control_id]
        rows = [{"point": index, "value": value} for index, value in enumerate(terms)]
        receipt = guess_receipt(
            rows, sequence_label=f"holonomic control: {control_id} ({citation})"
        )
        path = root / "runs" / "math" / "holonomic" / "survey" / f"control-{control_id}.json"
        _write_immutable(path, receipt)
        records.append(
            {
                "control_id": control_id,
                "citation": citation,
                "terms": len(rows),
                "receipt_path": _relative(root, path),
            }
        )
    return records


# ---------------------------------------------------------------------------
# Reading the receipts back and computing the discovery conditions
# ---------------------------------------------------------------------------


def _load_sealed(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        raise SequenceSurveyError(f"survey receipt is missing: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


#: A lambda that is a simple rational multiple of pi makes ``cos(lambda a(n))`` a function
#: of ``a(n)`` modulo ``2q`` alone, so the "signal" is a congruence statement about the
#: values rather than a hidden frequency.  Both bounds are declared, not tuned.
PI_RATIO_MAX_DENOMINATOR = 8
PI_RATIO_TOLERANCE = "0.001"


def _pi_ratio(lam: Fraction, dps: int) -> tuple[str, str, bool]:
    """(nearest simple p/q, |lambda - q-th multiple of pi|, within the declared tolerance)."""

    with mpmath.workdps(dps):
        value = mpmath.mpf(lam.numerator) / lam.denominator
        ratio = Fraction(float(value / mpmath.pi())).limit_denominator(
            PI_RATIO_MAX_DENOMINATOR
        )
        target = mpmath.pi() * ratio.numerator / ratio.denominator
        distance = abs(value - target)
        return (
            f"{ratio.numerator}/{ratio.denominator}",
            mpmath.nstr(distance, 10),
            bool(distance < mpmath.mpf(PI_RATIO_TOLERANCE)),
        )


def _phase_turns(lam: Fraction, values: Sequence[int], dps: int) -> tuple[str, bool]:
    """How many full turns of cos(lambda a(n)) the sequence's value range spans.

    Below one turn every term can sit inside a single negative half-period, so a
    cos-negative majority follows from boundedness alone.
    """

    span = max(values) - min(values)
    with mpmath.workdps(dps):
        turns = mpmath.mpf(lam.numerator) / lam.denominator * span / (2 * mpmath.pi())
        return mpmath.nstr(turns, 10), bool(turns < 1)


def _spectral_findings(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """The survivors and refutations one sealed spectral receipt actually states.

    Each entry carries two computed screens against the two ways a cos-negative
    majority arises without any hidden periodicity: a value range too narrow to span a
    full turn of the cosine, and a single dominant value that already accounts for the
    claimed fraction on its own.  Both are diagnostics printed beside the candidate,
    never filters that delete it.
    """

    receipt = _load_sealed(root, record["receipt_path"])
    validate_spectral_receipt(receipt)
    values = [int(row["value"]) for row in receipt["public_rows"]]
    dps = SPECTRAL_CAPS["base_dps"]
    modal_value = max(set(values), key=lambda item: (values.count(item), -item))
    modal_terms = values.count(modal_value)
    survivors: list[dict[str, Any]] = []
    refutations: list[dict[str, Any]] = []
    for peak in receipt["peaks"]:
        if peak["status"] == "NO_BIAS_PROPOSED":
            continue
        lam = Fraction(peak["lambda"])
        turns, bounded = _phase_turns(lam, values, dps)
        required = Fraction(
            peak["holdout"]["required_fraction"]["numerator"],
            peak["holdout"]["required_fraction"]["denominator"],
        )
        modal_negative = cos_sign(lam, Fraction(modal_value), dps) < 0
        pi_ratio, pi_distance, near_pi_ratio = _pi_ratio(lam, dps)
        entry = {
            "lambda": peak["lambda"],
            "lambda_grid_denominator": SPECTRAL_CAPS["quantize_denominator"],
            "magnitude_at_lambda": peak["magnitude_at_lambda"],
            "prefix_negative": {
                "negative_terms": peak["prefix_negative_terms"],
                "terms": peak["prefix_terms"],
            },
            "holdout_negative": {
                "negative_terms": peak["holdout"]["negative_terms"],
                "terms": peak["holdout"]["rows"],
            },
            "required_fraction": peak["holdout"]["required_fraction"],
            "statement": peak["statement"],
            "phase_turns": turns,
            "spans_less_than_one_phase_turn": bounded,
            "modal_value": modal_value,
            "modal_value_terms": modal_terms,
            "modal_value_cos_negative": modal_negative,
            "modal_value_alone_meets_threshold": bool(
                modal_negative and Fraction(modal_terms, len(values)) >= required
            ),
            "distance_to_simple_pi_multiple": pi_distance,
            "lambda_over_pi_simple_ratio": pi_ratio,
            "near_simple_rational_multiple_of_pi": near_pi_ratio,
        }
        (survivors if peak["status"] == "SURVIVED" else refutations).append(entry)
    return {
        "content_sha256": receipt["content_sha256"],
        "decision": receipt["decision"],
        "counts": receipt["counts"],
        "refutations": refutations,
        "survivors": survivors,
    }


def _holonomic_findings(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_sealed(root, record["receipt_path"])
    validate_holonomic_receipt(receipt)
    operator = receipt["operator"]
    return {
        "content_sha256": receipt["content_sha256"],
        "decision": receipt["decision"],
        "operator": None
        if operator is None
        else {
            "coefficients": operator["coefficients"],
            "degree": operator["degree"],
            "fitted_equations": operator["fitted_equations"],
            "order": operator["order"],
            "statement": operator["statement"],
            "verified_equations": operator["verified_equations"],
        },
    }


#: The three declared reasons a surviving bias is not, on its own, interesting.  Each is
#: computed per survivor and printed beside it; the screens narrow what is worth a human
#: prior-art review, and they never remove a candidate from the record.
DG4_SCREENS: tuple[dict[str, str], ...] = (
    {
        "screen_id": "explained_by_known_exact_structure",
        "statement": (
            "the sequence has a published exact term formula in the declared knowledge "
            "table, so a spectral bias restates known structure"
        ),
    },
    {
        "screen_id": "spans_less_than_one_phase_turn",
        "statement": (
            "lambda times the sequence's own value range is under one full turn of the "
            "cosine, so every term can sit inside one negative half-period by boundedness"
        ),
    },
    {
        "screen_id": "modal_value_alone_meets_threshold",
        "statement": (
            "the single most common value already occurs at least as often as the claimed "
            "fraction and has cos(lambda a) < 0, so the count restates the value histogram"
        ),
    },
    {
        "screen_id": "near_simple_rational_multiple_of_pi",
        "statement": (
            f"lambda is within {PI_RATIO_TOLERANCE} of p/q times pi for some q at most "
            f"{PI_RATIO_MAX_DENOMINATOR}, which makes cos(lambda a(n)) a function of "
            "a(n) modulo 2q: the claim is a congruence on the values, not a frequency"
        ),
    },
)


def _unscreened(hits: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Survivors that no declared screen explains away."""

    return [
        item
        for item in hits
        if item["explained_by_known_exact_structure"] is None
        and not item["spans_less_than_one_phase_turn"]
        and not item["modal_value_alone_meets_threshold"]
        and not item["near_simple_rational_multiple_of_pi"]
    ]


def build_summary(
    root: Path,
    spectral_records: Sequence[Mapping[str, Any]],
    holonomic_records: Sequence[Mapping[str, Any]],
    control_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Index both surveys and compute DG4's and DG6's conditions from the receipts."""

    spectral: list[dict[str, Any]] = []
    dg4_hits: list[dict[str, Any]] = []
    for record in spectral_records:
        problem_id = record["problem_id"]
        scanned = record["status"] == "SCANNED"
        known = _knowledge(problem_id, required=scanned)
        row: dict[str, Any] = {
            **{key: record[key] for key in ("problem_id", "generator", "terms", "status")},
            "builtin_knowledge_note": spectral_knowledge_note(problem_id, required=scanned),
            "detail": record["detail"],
            "published_spectral_signal": known["published_spectral_signal"],
            "receipt_path": record["receipt_path"],
        }
        if scanned:
            findings = _spectral_findings(root, record)
            row.update(findings)
            for survivor in findings["survivors"]:
                if known["published_spectral_signal"] is None:
                    dg4_hits.append(
                        {
                            "problem_id": problem_id,
                            **survivor,
                            "explained_by_known_exact_structure": (
                                known["known_exact_structure"]
                            ),
                        }
                    )
        spectral.append(row)

    holonomic: list[dict[str, Any]] = []
    dg6_hits: list[dict[str, Any]] = []
    for record in holonomic_records:
        problem_id = record["problem_id"]
        known = _knowledge(problem_id, required=record["status"] == "GUESSED")
        row = {
            **{key: record[key] for key in ("problem_id", "generator", "terms", "status")},
            "detail": record["detail"],
            "known_exact_structure": known["known_exact_structure"],
            "receipt_path": record["receipt_path"],
        }
        if record["status"] == "GUESSED":
            findings = _holonomic_findings(root, record)
            row.update(findings)
            if findings["decision"] == "OPERATOR_FOUND" and known["known_exact_structure"] is None:
                dg6_hits.append(
                    {"problem_id": problem_id, "operator": findings["operator"]["statement"]}
                )
        holonomic.append(row)

    controls: list[dict[str, Any]] = []
    for record in control_records:
        receipt = _load_sealed(root, record["receipt_path"])
        validate_holonomic_receipt(receipt)
        controls.append(
            {
                **dict(record),
                "content_sha256": receipt["content_sha256"],
                "decision": receipt["decision"],
                "operator_statement": None
                if receipt["operator"] is None
                else receipt["operator"]["statement"],
            }
        )
    controls_fired = bool(controls) and all(
        item["decision"] == "OPERATOR_FOUND" for item in controls
    )

    body: dict[str, Any] = {
        "claims": CLAIMS,
        "controls": {
            "all_recovered": controls_fired,
            "holonomic": controls,
            "spectral": _spectral_control(root, spectral),
        },
        "counts": {
            "holonomic_ineligible": sum(
                1 for item in holonomic if item["status"] != "GUESSED"
            ),
            "holonomic_no_annihilator": sum(
                1 for item in holonomic if item.get("decision") == "NO_ANNIHILATOR"
            ),
            "holonomic_operator_found": sum(
                1 for item in holonomic if item.get("decision") == "OPERATOR_FOUND"
            ),
            "holonomic_surveyed": sum(1 for item in holonomic if item["status"] == "GUESSED"),
            "problems_in_queue": len(spectral),
            "spectral_ineligible": sum(1 for item in spectral if item["status"] != "SCANNED"),
            "spectral_none_proposed": sum(
                1 for item in spectral if item.get("decision") == "NONE_PROPOSED"
            ),
            "spectral_none_survived": sum(
                1 for item in spectral if item.get("decision") == "NONE_SURVIVED"
            ),
            "spectral_surveyed": sum(1 for item in spectral if item["status"] == "SCANNED"),
            "spectral_survivors": sum(
                1 for item in spectral if item.get("decision") == "SPECTRAL_BIAS_SURVIVED"
            ),
        },
        "discovery_conditions": {
            "DG4": {
                "condition": (
                    "a surviving spectral bias, holding on the held-out suffix, on a "
                    "sequence with no published signal in this repository's builtin "
                    "knowledge, with the frequency reported to declared precision"
                ),
                "met": bool(dg4_hits),
                "met_after_declared_screens": bool(_unscreened(dg4_hits)),
                "note": (
                    "meeting this condition produces a candidate signal and nothing more: "
                    "the knowledge table is this repository's builtin knowledge, not a "
                    "literature search, and no candidate here has been prior-art reviewed"
                ),
                "screens": [dict(item) for item in DG4_SCREENS],
                "sequences": sorted({item["problem_id"] for item in dg4_hits}),
                "sequences_after_declared_screens": sorted(
                    {item["problem_id"] for item in _unscreened(dg4_hits)}
                ),
                "survivors": dg4_hits,
            },
            "DG6": {
                "condition": (
                    "an exact annihilating operator for a sequence with no known P-finite "
                    "recurrence or published exact term structure"
                ),
                "met": bool(dg6_hits),
                "note": (
                    "an operator is a guess confirmed on terms the solver never saw, not "
                    "a proof, and the ladder is finite: NO_ANNIHILATOR means 'not in this "
                    "ladder', never 'the sequence is not holonomic'"
                ),
                "operators": dg6_hits,
                "sequences": sorted({item["problem_id"] for item in dg6_hits}),
            },
        },
        "holonomic": holonomic,
        "profile": SURVEY_PROFILE,
        "schema_version": SUMMARY_SCHEMA,
        "scope": SUMMARY_SCOPE,
        "spectral": spectral,
        "spectral_min_terms": SPECTRAL_MIN_TERMS,
        "survey_terms": dict(sorted(SURVEY_TERMS.items())),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


#: The committed exemplar the spectral instrument is checked against: Steinerberger's
#: published Ulam frequency, and the receipt this repository already sealed for it.
ULAM_PUBLISHED_LAMBDA = "2.5714474995"
ULAM_EXEMPLAR_PATH = "runs/math/spectral/ulam-signal-v1.json"
ULAM_CONTROL_TOLERANCE = Fraction(1, 10**5)


def _spectral_control(root: Path, spectral: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Does the survey still recover the one spectral signal we know is published?"""

    row = next(
        (item for item in spectral if item["problem_id"] == "ulam_sequence_structure"), None
    )
    if row is None or row["status"] != "SCANNED" or not row.get("survivors"):
        return {
            "detail": "the Ulam survey scan produced no surviving bias",
            "matches_committed_exemplar": False,
            "published_lambda": ULAM_PUBLISHED_LAMBDA,
            "recovered": False,
        }
    measured = Fraction(row["survivors"][0]["lambda"])
    difference = abs(measured - Fraction(ULAM_PUBLISHED_LAMBDA))
    exemplar = _load_sealed(root, ULAM_EXEMPLAR_PATH)
    exemplar_lambdas = [
        peak["lambda"] for peak in exemplar["peaks"] if peak["status"] == "SURVIVED"
    ]
    return {
        "committed_exemplar": ULAM_EXEMPLAR_PATH,
        "detail": (
            "the survey's strongest Ulam survivor is compared against Steinerberger's "
            "published frequency and against the lambda this repository already sealed; "
            "the instrument is only as trustworthy as this row"
        ),
        "difference_from_published": lambda_decimal(
            Fraction(
                round(difference * SPECTRAL_CAPS["quantize_denominator"]),
                SPECTRAL_CAPS["quantize_denominator"],
            )
        ),
        "matches_committed_exemplar": row["survivors"][0]["lambda"] in exemplar_lambdas,
        "measured_lambda": row["survivors"][0]["lambda"],
        "published_lambda": ULAM_PUBLISHED_LAMBDA,
        "recovered": difference <= ULAM_CONTROL_TOLERANCE,
        "tolerance": lambda_decimal(ULAM_CONTROL_TOLERANCE),
    }


def validate_summary(value: Mapping[str, Any]) -> None:
    """Seal check plus a recheck that every bound receipt is present and unchanged."""

    if value.get("schema_version") != SUMMARY_SCHEMA:
        raise SequenceSurveyError("summary schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise SequenceSurveyError("summary seal changed")
    if value["claims"] != CLAIMS:
        raise SequenceSurveyError("summary claims changed")
    for row in value["spectral"]:
        if (row["receipt_path"] is None) != (row["status"] != "SCANNED"):
            raise SequenceSurveyError("a scanned spectral row must bind a receipt")
    for row in value["holonomic"]:
        if (row["receipt_path"] is None) != (row["status"] != "GUESSED"):
            raise SequenceSurveyError("a guessed holonomic row must bind a receipt")


def recheck_summary_against_receipts(root: Path, value: Mapping[str, Any]) -> None:
    """Re-derive every finding and both discovery flags from the receipts on disk."""

    validate_summary(value)
    rebuilt = build_summary(
        root,
        [
            {key: row[key] for key in ("problem_id", "generator", "terms", "status",
                                       "detail", "receipt_path")}
            for row in value["spectral"]
        ],
        [
            {key: row[key] for key in ("problem_id", "generator", "terms", "status",
                                       "detail", "receipt_path")}
            for row in value["holonomic"]
        ],
        [
            {key: row[key] for key in ("control_id", "citation", "terms", "receipt_path")}
            for row in value["controls"]["holonomic"]
        ],
    )
    if rebuilt != dict(value):
        raise SequenceSurveyError("summary does not recheck against its own receipts")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_survey(
    root: Path, queue_path: Path, *, use_gpu: bool = False
) -> tuple[dict[str, Any], Path]:
    """Run both surveys plus the controls and seal the summary.  Returns it and its path."""

    queue = load_queue(queue_path)
    spectral_records = run_spectral_survey(root, queue, use_gpu=use_gpu)
    holonomic_records = run_holonomic_survey(root, queue)
    control_records = run_controls(root)
    summary = build_summary(root, spectral_records, holonomic_records, control_records)
    path = root / "runs" / "math" / "survey-summary-v1.json"
    _write_immutable(path, summary)
    return summary, path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DG4/DG6 sequence survey.")
    parser.add_argument("--root", default=".", help="repository root for receipt paths")
    parser.add_argument("--queue", default="configs/problem_queue_v3.json")
    parser.add_argument("--gpu", action="store_true", help="use the GPU fp64 pre-filter")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.validate_checked:
        summary = json.loads(
            (root / "runs" / "math" / "survey-summary-v1.json").read_text(encoding="utf-8")
        )
        recheck_summary_against_receipts(root, summary)
        print("SURVEY SUMMARY rechecks against its receipts")
        return 0
    summary, path = run_survey(root, Path(args.queue), use_gpu=args.gpu)
    counts = summary["counts"]
    print(
        f"SURVEY spectral={counts['spectral_surveyed']} survivors="
        f"{counts['spectral_survivors']} holonomic={counts['holonomic_surveyed']} "
        f"operators={counts['holonomic_operator_found']} "
        f"DG4={summary['discovery_conditions']['DG4']['met']} "
        f"DG6={summary['discovery_conditions']['DG6']['met']} -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
