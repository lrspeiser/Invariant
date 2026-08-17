"""Adjudicate continued-fraction survivors against the prior-art corpus.

The inverse-symbolic CF lane produced 214 survivors of a 120-digit holdout and labelled
them against a fifteen-entry built-in table.  That table is far too small to say anything,
so 32 survivors carried the empty label ``NOT_IN_BUILTIN_TABLE``.  This module replaces that
label with a real adjudication against
:mod:`sigma_theory_compiler.cf_prior_art_corpus`, using **exact** tests only, in a fixed
order, recording which test fired:

1. ``exact_pattern_match`` -- the candidate's ``(a_n, b_n)`` pattern is literally a corpus
   pattern and the two continued fractions converge to the same value.
2. ``equivalence_orbit_match`` -- the corpus's own transformation group carries the
   candidate onto a corpus record.  Equivalence is decided in one step by the exact class
   invariant ``r_n = b_n/(a_n a_{n-1})`` rather than by search, and the remaining generators
   (tail shift, even/odd contraction, extension) are searched to a declared depth.  The
   chain is *exhibited and re-verified*, not asserted.
3. ``value_match_with_structural_confirmation`` -- the candidate's constant equals a corpus
   record's value **and** a chain exists.  Value equality on its own is never a ``KNOWN``
   verdict: two different continued fractions can converge to the same constant, and saying
   otherwise would turn "pi is a known number" into "this formula is known".  Value equality
   without a chain is the distinct verdict ``INCONCLUSIVE_VALUE_MATCH``.
4. ``NOT_FOUND_IN_CORPUS`` -- nothing fired.  This is absence from a finite corpus.  It is
   not novelty, and the receipt says so in a sealed claim.

**Controls are run-aborting.**  The same enumeration receipt carries 182 survivors already
labelled ``KNOWN_REDISCOVERED``.  A screen that cannot recover known formulas is not fit to
report an absence, so every one of those 182 is screened too and the run fails below a
declared recovery rate.  The recovery rate is reported explicitly in the receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp

from .cf_prior_art_corpus import (
    OVERRIDE_WINDOW,
    VALUE_MATCH_DIGITS,
    CFPattern,
    CFRecord,
    Corpus,
    CorpusError,
    Mobius,
    Poly,
    SeqSpec,
    load_corpus,
    mobius_apply,
    mobius_of,
    mobius_render,
    normal_form,
    resolve_to_seed,
    seq_from_poly,
    transform_contraction,
    transform_equivalence,
    transform_extension,
    transform_tail_shift,
)
from .inverse_symbolic_engine import constant_value
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-cf-prior-art-adjudication-1.0"

VERDICTS = ("KNOWN", "INCONCLUSIVE_VALUE_MATCH", "NOT_FOUND_IN_CORPUS")

#: Depth of the orbit search over the non-equivalence generators.  Equivalence itself is
#: decided exactly in one step by the class invariant, so this depth counts only tail
#: shifts, contractions, and extensions.
ORBIT_DEPTH = 2

#: Working precision for the adjudication.  Candidate values arrive at 100 digits.
SCREEN_DPS = 110

CONTROL_RECOVERY_THRESHOLD = Fraction(95, 100)

SCREEN_CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "external_fetch_performed": False,
    "human_review_required_before_any_novelty_claim": True,
    "value_match_alone_is_not_membership": True,
}


class ScreenError(ValueError):
    """Raised on malformed input, a failed control, or receipt tamper."""


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One continued-fraction conjecture ``wrap(CF(pattern)) = target``.

    ``alpha``/``beta`` are the enumeration lane's degree-two coefficient vectors and are
    carried for reporting; the screen itself works from ``pattern``, so a candidate outside
    that grammar (a planted disguise, for instance) can be adjudicated too.
    """

    candidate_id: str
    target: str
    pattern: CFPattern
    wrap: Mobius
    cf_value: str
    formula_text: str
    source_label: str
    alpha: tuple[int, int, int] | None = None
    beta: tuple[int, int, int] | None = None

    @staticmethod
    def from_polynomials(
        *,
        candidate_id: str,
        target: str,
        alpha: Sequence[int],
        beta: Sequence[int],
        wrap: Mobius,
        cf_value: str,
        formula_text: str,
        source_label: str,
    ) -> Candidate:
        return Candidate(
            candidate_id=candidate_id,
            target=target,
            pattern=CFPattern(seq_from_poly(Poly.of(*alpha)), seq_from_poly(Poly.of(*beta))),
            wrap=wrap,
            cf_value=cf_value,
            formula_text=formula_text,
            source_label=source_label,
            alpha=tuple(int(item) for item in alpha),  # type: ignore[arg-type]
            beta=tuple(int(item) for item in beta),  # type: ignore[arg-type]
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target": self.target,
            "alpha": None if self.alpha is None else list(self.alpha),
            "beta": None if self.beta is None else list(self.beta),
            "a_pattern": self.pattern.a.key(),
            "b_pattern": self.pattern.b.key(),
            "wrap": [str(item) for item in self.wrap],
            "formula_text": self.formula_text,
            "enumeration_label": self.source_label,
        }


def load_candidates(receipt: Mapping[str, Any]) -> list[Candidate]:
    """All survivors of a cf-enumeration receipt, ordered by ordinal."""

    candidates: list[Candidate] = []
    for survivor in receipt.get("survivors", []):
        candidates.append(
            Candidate.from_polynomials(
                candidate_id=str(survivor["ordinal"]),
                target=str(survivor["target"]),
                alpha=survivor["alpha"],
                beta=survivor["beta"],
                wrap=mobius_of(*(int(item) for item in survivor["mobius"])),
                cf_value=str(survivor["cf_value_at_final_stage"]),
                formula_text=str(survivor["formula_text"]),
                source_label=str(survivor["prior_art"]["label"]),
            )
        )
    candidates.sort(key=lambda item: int(item.candidate_id))
    return candidates


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


def _record_provenance(corpus: Corpus, record: CFRecord) -> list[dict[str, str]]:
    """The corpus record's own chain back to its cited seed."""

    chain = resolve_to_seed(corpus.by_id, record.record_id)
    steps: list[dict[str, str]] = []
    for record_id in chain[1:]:
        entry = corpus.by_id[record_id]
        detail = dict(entry.transform)
        steps.append(
            {
                "record_id": record_id,
                "transformation": detail.get("transformation", ""),
                "detail": detail.get("detail", ""),
            }
        )
    return steps


def exhibit_equivalence(candidate: CFPattern, record: CFPattern) -> dict[str, Any] | None:
    """Build and *verify* the equivalence sequence carrying ``candidate`` onto ``record``.

    The sequence is forced: ``c_n = a_n(record)/a_n(candidate)``.  It is then applied and the
    result compared key-for-key, so a reported chain is a checked chain.
    """

    period = math.lcm(candidate.a.period, record.a.period)
    try:
        terms = tuple(
            record.a.terms[residue % record.a.period]
            * candidate.a.terms[residue % candidate.a.period].reciprocal()
            for residue in range(period)
        )
        overrides: dict[Fraction, Fraction] | dict[int, Fraction] = {}
        for index in range(OVERRIDE_WINDOW):
            denominator = candidate.a.at(index)
            if denominator == 0:
                return None
            overrides[index] = record.a.at(index) / denominator  # type: ignore[index]
        sequence = SeqSpec.build(period, terms, overrides)  # type: ignore[arg-type]
        rebuilt, step = transform_equivalence(candidate, sequence)
    except CorpusError:
        return None
    if rebuilt.key() != record.key():
        return None
    return {
        "transformation": "equivalence",
        "detail": f"c_n = {sequence.key()}",
        "value_map": mobius_render(step),
        "verified": True,
    }


def _orbit_neighbours(
    pattern: CFPattern, cf_value: mp.mpf
) -> list[tuple[CFPattern, mp.mpf, dict[str, str]]]:
    """One step of every declared non-equivalence generator."""

    out: list[tuple[CFPattern, mp.mpf, dict[str, str]]] = []
    for levels in (1, 2, 3):
        try:
            moved, step = transform_tail_shift(pattern, levels)
        except CorpusError:
            continue
        out.append(
            (
                moved,
                mobius_apply(step, cf_value),
                {
                    "transformation": "tail_shift",
                    "detail": f"levels={levels}",
                    "value_map": mobius_render(step),
                },
            )
        )
    for parity in ("even", "odd"):
        try:
            moved = transform_contraction(pattern, parity)
        except CorpusError:
            continue
        out.append(
            (
                moved,
                cf_value,
                {
                    "transformation": f"contract_{parity}",
                    "detail": f"parity={parity}",
                    "value_map": "x",
                },
            )
        )
    try:
        out.append(
            (
                transform_extension(pattern),
                cf_value,
                {
                    "transformation": "extend",
                    "detail": "unit-denominator inverse of the even contraction",
                    "value_map": "x",
                },
            )
        )
    except CorpusError:
        pass
    return out


def _membership(
    corpus: Corpus, pattern: CFPattern, cf_value: mp.mpf
) -> tuple[CFRecord, dict[str, Any] | None] | None:
    """Exact membership of one orbit node: pattern identity, then equivalence class."""

    for record in corpus.lookup_pattern(pattern):
        if _values_agree(mp.mpf(record.cf_value), cf_value):
            return record, None
    try:
        form = normal_form(pattern)
    except CorpusError:
        return None
    for record in corpus.lookup_normal_form(form, cf_value):
        if record.pattern is None:
            continue
        exhibited = exhibit_equivalence(pattern, record.pattern)
        if exhibited is not None:
            return record, exhibited
    return None


def _values_agree(left: mp.mpf, right: mp.mpf, digits: int = VALUE_MATCH_DIGITS) -> bool:
    if not (mp.isfinite(left) and mp.isfinite(right)):
        return False
    scale = max(mp.mpf(1), abs(right))
    return bool(abs(left - right) / scale < mp.mpf(10) ** (-digits))


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


def screen_candidate(
    corpus: Corpus, candidate: Candidate, *, orbit_depth: int = ORBIT_DEPTH
) -> dict[str, Any]:
    """Adjudicate one candidate.  Returns a fully explained verdict record."""

    with mp.workdps(SCREEN_DPS):
        pattern = candidate.pattern
        cf_value = mp.mpf(candidate.cf_value)
        target_value = constant_value(candidate.target)
        reported = mobius_apply(candidate.wrap, cf_value)
        consistent = _values_agree(reported, target_value, digits=90)
        report: dict[str, Any] = {
            **candidate.as_json(),
            "cf_value_100_digits": candidate.cf_value,
            "wrap_reproduces_target_at_90_digits": consistent,
        }

        # Test 1 and 2 -- one exact membership call per orbit node.
        frontier: list[tuple[CFPattern, mp.mpf, list[dict[str, Any]]]] = [(pattern, cf_value, [])]
        seen = {pattern.key()}
        hit: tuple[CFRecord, dict[str, Any] | None] | None = None
        hit_chain: list[dict[str, Any]] = []
        for depth in range(orbit_depth + 1):
            nxt: list[tuple[CFPattern, mp.mpf, list[dict[str, Any]]]] = []
            for node, value, steps in frontier:
                found = _membership(corpus, node, value)
                if found is not None:
                    hit, hit_chain = found, steps
                    break
                if depth < orbit_depth:
                    for moved, moved_value, step in _orbit_neighbours(node, value):
                        if moved.key() in seen:
                            continue
                        seen.add(moved.key())
                        nxt.append((moved, moved_value, [*steps, step]))
            if hit is not None:
                break
            frontier = nxt

        if hit is not None:
            record, equivalence_step = hit
            chain = list(hit_chain)
            if equivalence_step is not None:
                chain.append(equivalence_step)
            test = "exact_pattern_match" if not chain else "equivalence_orbit_match"
            report.update(
                {
                    "verdict": "KNOWN",
                    "test_that_fired": test,
                    "matched_record": _record_summary(corpus, record),
                    "transformation_chain": chain,
                    "chain_verified": True,
                }
            )
            return report

        # Test 3 -- value equality, which is never membership on its own.
        by_reported = corpus.lookup_reported_value(target_value)
        by_cf = corpus.lookup_value(cf_value)
        relatives = corpus.family_relatives(pattern)
        adjacent = [
            {
                "a_constant_offset": delta,
                **_record_summary(corpus, record),
            }
            for delta, record in corpus.adjacent_family_members(pattern)
        ]
        reasons = {
            "exact_pattern_match": "no corpus record carries this exact (a_n, b_n) pattern",
            "equivalence_orbit_match": (
                "the equivalence-class invariant r_n = b_n/(a_n a_{n-1}) of this candidate, "
                f"and of every node reached within {orbit_depth} tail-shift/contraction/"
                "extension steps, is absent from the corpus index"
            ),
        }
        if by_reported or by_cf:
            report.update(
                {
                    "verdict": "INCONCLUSIVE_VALUE_MATCH",
                    "test_that_fired": "value_match_without_structural_confirmation",
                    "value_matches": {
                        "records_with_the_same_reported_value": [
                            _record_summary(corpus, item) for item in by_reported[:3]
                        ],
                        "records_whose_continued_fraction_has_the_same_limit": [
                            _record_summary(corpus, item) for item in by_cf[:3]
                        ],
                        "match_digits": VALUE_MATCH_DIGITS,
                    },
                    "why_no_chain": reasons,
                    "nearest_family_relatives": [
                        _record_summary(corpus, item) for item in relatives
                    ],
                    "adjacent_family_members_in_corpus": adjacent,
                    "note": (
                        "a corpus record converges to the same constant, but no declared "
                        "transformation chain connects the two continued fractions; two "
                        "different continued fractions can share a limit"
                    ),
                }
            )
            return report

        report.update(
            {
                "verdict": "NOT_FOUND_IN_CORPUS",
                "test_that_fired": "no_test_fired",
                "why_no_chain": {
                    **reasons,
                    "value_match_with_structural_confirmation": (
                        "no corpus record reports this value and no corpus continued "
                        f"fraction converges to it within {VALUE_MATCH_DIGITS} digits"
                    ),
                },
                "nearest_family_relatives": [_record_summary(corpus, item) for item in relatives],
                "adjacent_family_members_in_corpus": adjacent,
                "note": (
                    "absence from this finite corpus; this is not a novelty claim and "
                    "requires human prior-art review before any such claim"
                ),
            }
        )
        return report


def _record_summary(corpus: Corpus, record: CFRecord) -> dict[str, Any]:
    seed = corpus.by_id.get(f"seed:{record.seed_id}", record)
    return {
        "record_id": record.record_id,
        "family": record.family,
        "seed_id": record.seed_id,
        "identity": f"{mobius_render(record.wrap)} = {record.value_expr}",
        "pattern_key": record.pattern_key(),
        "value": record.value,
        "citation": record.citation.as_json(),
        "validity_domain": record.validity_domain,
        "seed_identity": seed.value_expr,
        "provenance": _record_provenance(corpus, record),
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_screen(
    receipt: Mapping[str, Any],
    corpus: Corpus,
    *,
    orbit_depth: int = ORBIT_DEPTH,
    receipt_path: str = "runs/math/inverse-symbolic/cf-enumeration-v1.json",
) -> dict[str, Any]:
    """Screen every survivor, enforce the controls, and seal an adjudication receipt."""

    started = time.perf_counter()
    candidates = load_candidates(receipt)
    if not candidates:
        raise ScreenError("input receipt carries no survivors")
    adjudications = [screen_candidate(corpus, item, orbit_depth=orbit_depth) for item in candidates]

    controls = [
        item
        for item in adjudications
        if item["enumeration_label"] == "KNOWN_REDISCOVERED"
    ]
    subjects = [
        item for item in adjudications if item["enumeration_label"] == "NOT_IN_BUILTIN_TABLE"
    ]
    if not controls:
        raise ScreenError("no KNOWN_REDISCOVERED controls in the input receipt")
    recovered = sum(1 for item in controls if item["verdict"] == "KNOWN")
    cited = sum(
        1
        for item in controls
        if item["verdict"] == "KNOWN" and item["matched_record"]["citation"]["reference"]
    )
    rate = Fraction(recovered, len(controls))
    control_block = {
        "labelled_known_rediscovered": len(controls),
        "screened_KNOWN": recovered,
        "screened_KNOWN_with_resolvable_citation": cited,
        "recovery_rate": f"{float(rate):.4f}",
        "threshold": f"{float(CONTROL_RECOVERY_THRESHOLD):.2f}",
        "passed": bool(rate >= CONTROL_RECOVERY_THRESHOLD),
    }
    if not control_block["passed"]:
        raise ScreenError(
            "control recovery rate "
            f"{control_block['recovery_rate']} below the declared threshold "
            f"{control_block['threshold']}: the screen cannot recover known formulas and is "
            "therefore not fit to report any absence"
        )

    by_verdict: dict[str, int] = {name: 0 for name in VERDICTS}
    by_test: dict[str, int] = {}
    by_target: dict[str, dict[str, int]] = {}
    by_confidence: dict[str, int] = {}
    for item in subjects:
        by_verdict[item["verdict"]] += 1
        by_test[item["test_that_fired"]] = by_test.get(item["test_that_fired"], 0) + 1
        bucket = by_target.setdefault(item["target"], {name: 0 for name in VERDICTS})
        bucket[item["verdict"]] += 1
        if item["verdict"] == "KNOWN":
            confidence = item["matched_record"]["citation"]["confidence"]
            by_confidence[confidence] = by_confidence.get(confidence, 0) + 1

    config = {
        "orbit_depth": orbit_depth,
        "value_match_digits": VALUE_MATCH_DIGITS,
        "screen_dps": SCREEN_DPS,
        "control_recovery_threshold": f"{float(CONTROL_RECOVERY_THRESHOLD):.2f}",
        "test_order": [
            "exact_pattern_match",
            "equivalence_orbit_match",
            "value_match_with_structural_confirmation",
            "no_test_fired",
        ],
        "equivalence_decision": (
            "one-step exact class invariant r_n = b_n/(a_n a_{n-1}); the orbit search covers "
            "tail_shift(1..3), contract_even, contract_odd, and extend"
        ),
    }
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "lane": "cf-prior-art-adjudication",
        "claims": SCREEN_CLAIMS,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "input": {
            "receipt": receipt_path,
            "content_sha256": receipt["content_sha256"],
            "result_core_sha256": receipt["result_core_sha256"],
            "survivors": len(candidates),
            "labelled_known_rediscovered": len(controls),
            "labelled_not_in_builtin_table": len(subjects),
        },
        "corpus": {
            "schema_version": corpus.manifest["schema_version"],
            "content_sha256": corpus.manifest["content_sha256"],
            "records_sha256": corpus.manifest["records_sha256"],
            "sqlite_sha256": corpus.manifest.get("sqlite_sha256"),
            "records": corpus.manifest["counts"]["records"],
            "seeds": corpus.manifest["counts"]["seeds"],
            "external_fetch_performed": corpus.manifest["claims"]["external_fetch_performed"],
        },
        "controls": control_block,
        "counts": {
            "by_verdict": by_verdict,
            "by_test_that_fired": dict(sorted(by_test.items())),
            "by_target_constant": dict(sorted(by_target.items())),
            "known_by_citation_confidence": dict(sorted(by_confidence.items())),
        },
        "candidates": subjects,
        "control_summaries": [
            {
                "candidate_id": item["candidate_id"],
                "target": item["target"],
                "verdict": item["verdict"],
                "test_that_fired": item["test_that_fired"],
                "matched_record_id": item.get("matched_record", {}).get("record_id"),
                "citation_reference": item.get("matched_record", {})
                .get("citation", {})
                .get("reference"),
                "citation_confidence": item.get("matched_record", {})
                .get("citation", {})
                .get("confidence"),
            }
            for item in controls
        ],
        "scope": (
            "Exact adjudication of continued-fraction survivors against a 13k-record "
            "prior-art corpus built from independently encoded classical identities and "
            "their declared transformation orbits. KNOWN requires an exhibited, re-verified "
            "transformation chain to a cited seed. Value equality alone yields "
            "INCONCLUSIVE_VALUE_MATCH. NOT_FOUND_IN_CORPUS is absence from a finite corpus "
            "and is never a novelty claim."
        ),
    }
    core = canonical_sha256(body)
    body["result_core_sha256"] = core
    body["measurement"] = {"elapsed_seconds": format(time.perf_counter() - started, ".3f")}
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# Receipt validation
# ---------------------------------------------------------------------------


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seals, claims, verdict vocabulary, count consistency, and the control gate."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise ScreenError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ScreenError("receipt seal changed")
    core_body = {
        key: item
        for key, item in value.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    if value.get("result_core_sha256") != canonical_sha256(core_body):
        raise ScreenError("deterministic core seal changed")
    if value.get("config_sha256") != canonical_sha256(value.get("config", {})):
        raise ScreenError("config binding changed")
    if value.get("claims") != SCREEN_CLAIMS:
        raise ScreenError("claims block changed")
    candidates = value.get("candidates", [])
    if len(candidates) != value["input"]["labelled_not_in_builtin_table"]:
        raise ScreenError("adjudicated candidate count changed")
    counts = {name: 0 for name in VERDICTS}
    for item in candidates:
        if item["verdict"] not in VERDICTS:
            raise ScreenError(f"unknown verdict {item['verdict']!r}")
        counts[item["verdict"]] += 1
        if item["verdict"] == "KNOWN":
            record = item.get("matched_record") or {}
            if not record.get("citation", {}).get("reference"):
                raise ScreenError(f"KNOWN verdict without a citation: {item['candidate_id']}")
            if not item.get("chain_verified"):
                raise ScreenError(f"KNOWN verdict without a verified chain: {item['candidate_id']}")
        else:
            if not item.get("why_no_chain"):
                raise ScreenError(f"non-KNOWN verdict without a reason: {item['candidate_id']}")
    if counts != value["counts"]["by_verdict"]:
        raise ScreenError("verdict counts changed")
    controls = value["controls"]
    if controls["screened_KNOWN"] > controls["labelled_known_rediscovered"]:
        raise ScreenError("control recovery exceeds the control population")
    rate = Fraction(controls["screened_KNOWN"], controls["labelled_known_rediscovered"])
    if bool(rate >= CONTROL_RECOVERY_THRESHOLD) != controls["passed"]:
        raise ScreenError("control gate result changed")
    if not controls["passed"]:
        raise ScreenError("receipt records a failed control gate")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_receipt(result: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise ScreenError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adjudicate continued-fraction survivors against the prior-art corpus."
    )
    parser.add_argument("--input", default="runs/math/inverse-symbolic/cf-enumeration-v1.json")
    parser.add_argument("--database", default="runs/math/prior-art/cf-corpus-v1.sqlite")
    parser.add_argument("--corpus-manifest", default="runs/math/prior-art/cf-corpus-v1-manifest.json")
    parser.add_argument("--output", default="runs/math/prior-art/cf-adjudication-v1.json")
    parser.add_argument("--orbit-depth", type=int, default=ORBIT_DEPTH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    if args.validate_checked:
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        print(json.dumps({"validated": True, "output": args.output}))
        return 0
    receipt = json.loads(Path(args.input).read_text(encoding="utf-8"))
    corpus = load_corpus(args.database, args.corpus_manifest)
    result = run_screen(receipt, corpus, orbit_depth=args.orbit_depth, receipt_path=args.input)
    _write_receipt(result, args.output)
    print(
        json.dumps(
            {
                "candidates_adjudicated": len(result["candidates"]),
                "by_verdict": result["counts"]["by_verdict"],
                "control_recovery_rate": result["controls"]["recovery_rate"],
                "corpus_records": result["corpus"]["records"],
                "output": args.output,
                "content_sha256": result["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
