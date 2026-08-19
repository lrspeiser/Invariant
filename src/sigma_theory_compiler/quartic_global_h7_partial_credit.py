"""Partial credit for the System9 global-H7 lifespan gate: a BLOCK that also says *how far*.

``quartic_candidate_complete_global_h7_lifespan_gate`` returns ``BLOCK_SYSTEM9`` for all twelve
quartic candidates.  Its twelve candidate records are byte-identical apart from the candidate id
and two finite-Sobolev fields, so the gate answers *no* twelve times and offers nothing to steer
by.  Nothing downstream can tell which candidate is closest to the missing primitive.

This module leaves that verdict exactly where it is and adds a second, orthogonal output: a
**structured distance**.  For each candidate it walks one declared, ordered obligation ledger --
the obligations a PASS would require, taken from the gate's own ``exact_remaining_contract`` and
from the flags the global-H7 certificate already publishes -- and records, obligation by
obligation, whether it is met and, when it is not, by how much it is missed.

Two kinds of shortfall appear, and they are kept apart:

*Discrete shortfalls* are exact integers: 594 unfinished lower-DF entries, 11 unproved affine
split residuals, 15 unaccepted full-direction replays.  These are identical across all twelve
candidates and therefore carry **no** ordering information.  Reporting that is the point: it
localizes the gate's blindness.

*Magnitude shortfalls* are the exact closed forms the certificate already carries.  The
principal one is ``A_known``, the certified linear coefficient in

    E7'(t) <= A*E7(t) + D*E7(t)^(3/2),   A = A_known + 2*Lambda*C_L/H7_lower >= A_known,

whose Riccati time-to-threshold obeys, for every admissible nonnegative ``C_L`` and ``C_B``,

    T <= (2/A)*log(z_tube/z0) <= (2/A_known)*log(z_tube/z0),

and at the certificate's own declared bootstrap ceiling ``E7(0) <= tube_energy_threshold/4`` the
ratio ``z_tube/z0`` is exactly 2.  So ``2*log(2)/A_known`` is a ceiling on the conditional
lifespan that is already proved, and a *smaller* ``A_known`` is a *larger* ceiling: more room for
the eventual lifespan to be usefully positive once the missing B7 bound arrives.  That is the
ordering key, and it is fixed by the certificate's own formula rather than chosen after looking.

Every comparison on the certificate path runs in exact rational arithmetic through
:mod:`.exact_real_bracket`.  A strict order between two candidates is emitted only when two exact
rationals separate their closed forms, and both rationals are transcribed into the receipt.  When
the declared precision ladder does not separate a pair, the pair is reported as unseparated and
the two candidates share a rank; no order is invented.

What partial credit is *not*.  A rank is not a proof, not a promotion, and not a claim.  The
verdict on every candidate stays ``BLOCK``; every claim flag stays False; and a mutated input in
which a candidate's lifespan is asserted proved is rejected outright rather than ranked first.
The controls in :func:`negative_controls` all have to fail for the receipt to be issued.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .exact_real_bracket import (
    GREATER,
    LESS,
    UNSEPARATED,
    Bracket,
    bracket_expression,
    compare_expressions,
    log_bracket,
)

RECEIPT_SCHEMA = "sigma-quartic-global-h7-partial-credit-1.0"
CAMPAIGN_ID = "quartic-global-h7-partial-credit-001"
EXPECTED_CANDIDATES = 12

GLOBAL_H7_PATH = "runs/physics-language/quartic-global-h7-energy-campaign/campaign.json"
FINITE_SOBOLEV_PATH = (
    "runs/physics-language/quartic-finite-sobolev-hierarchy-no-go-campaign/campaign.json"
)
BLOCK_GATE_PATH = (
    "runs/physics-language/quartic-candidate-complete-global-h7-lifespan-gate/campaign.json"
)

GLOBAL_H7_SCHEMA = "sigma-quartic-global-h7-energy-campaign-1.0"
GLOBAL_H7_STATUS = (
    "audit_all_12_global_H7_energies_single_source_remainder_lifespans_fail_closed"
)
FINITE_SOBOLEV_SCHEMA = "sigma-quartic-finite-sobolev-hierarchy-no-go-campaign-1.0"
FINITE_SOBOLEV_STATUS = "finite_unmodified_Sobolev_hierarchy_refuted_candidates_blocked"
BLOCK_GATE_SCHEMA = "sigma-quartic-candidate-complete-global-h7-lifespan-gate-1.0"
BLOCK_GATE_DECISION = "BLOCK_SYSTEM9"

#: The declared precision ladder for every exact comparison on the certificate path, in
#: significant bits.  A pair the last rung cannot separate is reported unseparated, never ordered.
DECLARED_LADDER = (64, 128, 256, 512, 1024)

#: The starved ladder the falsification control runs at.  Sixteen bits resolves the coarse split
#: between the coefficient classes but not the tight one, so it must produce strictly fewer tiers.
STARVED_LADDER = (16,)

CLAIMS_POLICY = {
    "global_H7_proved": False,
    "bootstrap_closed": False,
    "lifespan_proved": False,
    "candidate_promoted": False,
    "rank_is_a_proof": False,
    "observation_opened": False,
}


#: Pure caches.  The twelve candidates carry only four distinct closed forms per magnitude, so
#: memoizing on the exact strings turns a quadratic sweep into a handful of real comparisons.
#: Same arguments, same answer -- these change nothing about what is certified.
_BRACKET_MEMO: dict[tuple[str, int], Bracket] = {}
_COMPARE_MEMO: dict[tuple[str, str, tuple[int, ...]], tuple[str, dict[str, Any]]] = {}


def _cached_bracket(expression: str, bits: int) -> Bracket:
    memo_key = (expression, bits)
    if memo_key not in _BRACKET_MEMO:
        _BRACKET_MEMO[memo_key] = bracket_expression(expression, bits)
    return _BRACKET_MEMO[memo_key]


def _cached_compare(
    left: str, right: str, ladder: tuple[int, ...]
) -> tuple[str, dict[str, Any]]:
    memo_key = (left, right, ladder)
    if memo_key not in _COMPARE_MEMO:
        comparison = compare_expressions(left, right, ladder)
        _COMPARE_MEMO[memo_key] = (comparison.verdict, comparison.as_receipt())
    return _COMPARE_MEMO[memo_key]


class PartialCreditError(ValueError):
    """Raised when partial credit is asked for on inputs that do not support it."""


@dataclass(frozen=True)
class Obligation:
    """One entry of the declared ledger a PASS would have to satisfy."""

    key: str
    source: str
    path: tuple[str, ...]
    kind: str
    #: For ``boolean_unmet``: how many primitives are outstanding behind the False flag.
    outstanding: int = 0
    #: For the magnitude kinds: the direction that moves the candidate toward a PASS.
    direction: str = ""
    #: For the magnitude kinds: the ledger obligation this magnitude quantifies.
    attached_to: str = ""
    meaning: str = ""


#: The declared obligation ledger, in the order a PASS has to satisfy it.  The first block is
#: already discharged for every candidate; the second block is what BLOCK is blocking on.
OBLIGATION_LEDGER: tuple[Obligation, ...] = (
    Obligation(
        key="global_H7_energy_equivalence",
        source="global_h7",
        path=("global_H7_energy_equivalence_certified",),
        kind="boolean_met",
        meaning="H7_lower*||U||^2 <= E7 <= H7_upper*||U||^2 with coercivity and finite low modes",
    ),
    Obligation(
        key="global_nonremainder_dyadic_summation",
        source="global_h7",
        path=("global_nonremainder_dyadic_summation_certified",),
        kind="boolean_met",
        meaning="every currently proved shell term summed into one differential inequality",
    ),
    Obligation(
        key="leading_good_unknown_symbol_binding",
        source="global_h7",
        path=("good_unknown_and_source", "leading_good_unknown_symbol_binding_verified"),
        kind="boolean_met",
        meaning="the leading high-coefficient/low-state symbol is bound to P55",
    ),
    Obligation(
        key="leading_derivative_loss_resolved",
        source="global_h7",
        path=("good_unknown_and_source", "leading_derivative_loss_resolved"),
        kind="boolean_met",
        meaning="the leading order derivative loss is cancelled by the good unknown",
    ),
    Obligation(
        key="explicit_remainder_differential_inequality",
        source="global_h7",
        path=("strongest_global_differential_inequality", "proved_with_explicit_remainder"),
        kind="boolean_met",
        meaning="the inequality is proved with a single explicit unresolved functional B7",
    ),
    Obligation(
        key="universal_affine_split",
        source="global_h7",
        path=("good_unknown_and_source", "universal_affine_split_proved"),
        kind="boolean_unmet",
        outstanding=11,
        meaning="11 universal acceleration affine split entry residuals still unproved zero",
    ),
    Obligation(
        key="mixed_multi_index_components",
        source="global_h7",
        path=("good_unknown_and_source", "mixed_multi_index_components_completed"),
        kind="count_from_zero",
        meaning=(
            "mixed D2..D4 multi-index component DAG roots: none completed, and the required "
            "total is not published, so the shortfall recorded here is a floor of 1"
        ),
    ),
    Obligation(
        key="lower_DF_entries",
        source="global_h7",
        path=("good_unknown_and_source", "lower_DF_entries_missing"),
        kind="count_to_zero",
        meaning="lower Frechet derivative entries still missing from the source Jacobian",
    ),
    Obligation(
        key="paralinearization_remainder_bound",
        source="global_h7",
        path=("good_unknown_and_source", "paralinearization_remainder_bound_proved"),
        kind="boolean_unmet",
        outstanding=1,
        meaning="B7 <= C_L*sqrt(Q7) + C_B*Q7 is not derived, so the inequality cannot close",
    ),
    Obligation(
        key="full_tensor_cancellation",
        source="finite_sobolev",
        path=("full_tensor_cancellation_proved",),
        kind="boolean_unmet",
        outstanding=1,
        meaning="no full tensor cancellation of the refuted finite-Sobolev slice",
    ),
    Obligation(
        key="full_direction_completion_obstruction",
        source="block_gate",
        path=("obstruction_branch", "full_direction_completion_obstruction_proved"),
        kind="boolean_unmet",
        outstanding=15,
        meaning=(
            "none of the 15 declared polarization evaluations carries an accepted "
            "full-direction alternative recurrence replay"
        ),
    ),
    Obligation(
        key="closed_global_H7_inequality",
        source="global_h7",
        path=("global_H7_differential_inequality_closed",),
        kind="boolean_unmet",
        outstanding=1,
        meaning="no closed Gronwall inequality",
    ),
    Obligation(
        key="nonlinear_lifespan",
        source="global_h7",
        path=("nonlinear_lifespan_proved",),
        kind="boolean_unmet",
        outstanding=1,
        meaning="no explicit positive lifespan",
    ),
    Obligation(
        key="uncancelled_slice_growth_multiplier",
        source="finite_sobolev",
        path=("absolute_growth_multiplier",),
        kind="integer_magnitude",
        direction="minimize",
        attached_to="full_tensor_cancellation",
        meaning=(
            "|D2|, the multiplier on the uncancelled H^s lower bound |D2|*N*c_packet/2 that a "
            "full tensor cancellation would have to annihilate"
        ),
    ),
    Obligation(
        key="certified_linear_energy_growth",
        source="global_h7",
        path=("strongest_global_differential_inequality", "A_known"),
        kind="magnitude",
        direction="minimize",
        attached_to="closed_global_H7_inequality",
        meaning=(
            "A_known in E7' <= A*E7 + D*E7^(3/2); the conditional lifespan obeys "
            "T <= 2*log(2)/A_known at the certificate's declared bootstrap ceiling"
        ),
    ),
    Obligation(
        key="unresolved_remainder_coefficient",
        source="global_h7",
        path=("strongest_global_differential_inequality", "Gamma_B"),
        kind="magnitude",
        direction="minimize",
        attached_to="paralinearization_remainder_bound",
        meaning="Gamma_B, the weight the unresolved functional B7 carries in the inequality",
    ),
    Obligation(
        key="energy_equivalence_condition_number",
        source="global_h7",
        path=("global_energy", "__ratio__"),
        kind="magnitude",
        direction="minimize",
        attached_to="global_H7_energy_equivalence",
        meaning="H7_upper/H7_lower, the tightness of the certified energy equivalence",
    ),
)

#: The declared ranking keys, applied lexicographically in this order.  Everything before
#: ``uncancelled_slice_growth_multiplier`` is identical across the twelve by construction; it is
#: listed anyway so a future input that does differ there is ordered before the magnitudes.
RANKING_KEYS: tuple[str, ...] = (
    "unmet_obligation_count",
    "discrete_shortfall_total",
    "uncancelled_slice_growth_multiplier",
    "certified_linear_energy_growth",
    "unresolved_remainder_coefficient",
    "energy_equivalence_condition_number",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def ledger_sha256() -> str:
    """A seal over the declared ledger and key order, so neither can be reordered quietly."""

    return _sha(
        {
            "ledger": [
                {
                    "key": item.key,
                    "source": item.source,
                    "path": list(item.path),
                    "kind": item.kind,
                    "outstanding": item.outstanding,
                    "direction": item.direction,
                    "attached_to": item.attached_to,
                }
                for item in OBLIGATION_LEDGER
            ],
            "ranking_keys": list(RANKING_KEYS),
        }
    )


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PartialCreditError(f"partial-credit source is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise PartialCreditError("partial-credit source must be one JSON object")
    return value


def _dig(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for step in path:
        if not isinstance(value, Mapping) or step not in value:
            raise PartialCreditError(f"missing certificate field: {'.'.join(path)}")
        value = value[step]
    return value


def _magnitude_expression(
    obligation: Obligation, certificate: Mapping[str, Any]
) -> str:
    if obligation.path[-1] == "__ratio__":
        energy = _dig(certificate, obligation.path[:-1])
        return f"({energy['H7_upper']})/({energy['H7_lower']})"
    value = _dig(certificate, obligation.path)
    if not isinstance(value, str):
        raise PartialCreditError(f"magnitude {obligation.key} is not a closed form")
    return value


def validate_sources(
    global_h7: Mapping[str, Any],
    finite_sobolev: Mapping[str, Any],
    block_gate: Mapping[str, Any],
) -> None:
    """Refuse partial credit on anything that is not a sealed, fail-closed BLOCK on twelve."""

    if global_h7.get("schema_version") != GLOBAL_H7_SCHEMA:
        raise PartialCreditError("global H7 schema changed")
    if global_h7.get("status") != GLOBAL_H7_STATUS:
        raise PartialCreditError("global H7 status changed")
    if _content_sha(global_h7) != global_h7.get("content_sha256"):
        raise PartialCreditError("global H7 content seal changed")
    if finite_sobolev.get("schema_version") != FINITE_SOBOLEV_SCHEMA:
        raise PartialCreditError("finite-Sobolev schema changed")
    if finite_sobolev.get("decision") != FINITE_SOBOLEV_STATUS:
        raise PartialCreditError("finite-Sobolev decision changed")
    if finite_sobolev.get("decision_counts") != {"pass": 0, "reject": 0, "blocked": 12}:
        raise PartialCreditError("finite-Sobolev decision counts changed")
    if _content_sha(finite_sobolev) != finite_sobolev.get("content_sha256"):
        raise PartialCreditError("finite-Sobolev content seal changed")
    if block_gate.get("schema_version") != BLOCK_GATE_SCHEMA:
        raise PartialCreditError("block gate schema changed")
    if block_gate.get("decision") != BLOCK_GATE_DECISION:
        raise PartialCreditError("block gate decision changed")
    if _content_sha(block_gate) != block_gate.get("content_sha256"):
        raise PartialCreditError("block gate content seal changed")
    if any(block_gate.get("claims", {}).values()):
        raise PartialCreditError("block gate has an opened claim")

    certificates = global_h7.get("certificates")
    records = finite_sobolev.get("candidate_records")
    gate_records = block_gate.get("candidate_records")
    for name, items in (
        ("global H7", certificates),
        ("finite-Sobolev", records),
        ("block gate", gate_records),
    ):
        if not isinstance(items, list) or len(items) != EXPECTED_CANDIDATES:
            raise PartialCreditError(f"{name} candidate set is not {EXPECTED_CANDIDATES}")
    identities = {
        name: sorted(item["candidate_id"] for item in items)
        for name, items in (
            ("global H7", certificates),
            ("finite-Sobolev", records),
            ("block gate", gate_records),
        )
    }
    if len(set(map(tuple, identities.values()))) != 1:
        raise PartialCreditError("candidate identity alignment changed")

    for item in gate_records:
        if item.get("decision") != BLOCK_GATE_DECISION or item.get("completion_grade"):
            raise PartialCreditError("partial credit is defined only on a uniform BLOCK")
    for item in certificates:
        # A candidate that already closed is not a near-miss and must not be ranked as one.
        if (
            item.get("global_H7_differential_inequality_closed")
            or item.get("nonlinear_lifespan_proved")
            or item.get("global_H7_dyadic_sum_applied")
        ):
            raise PartialCreditError("a candidate claims closure; partial credit is void")


def _evaluate_ledger(
    certificate: Mapping[str, Any],
    finite_record: Mapping[str, Any],
    gate_record: Mapping[str, Any],
) -> dict[str, Any]:
    sources = {
        "global_h7": certificate,
        "finite_sobolev": finite_record,
        "block_gate": gate_record,
    }
    met: list[str] = []
    failed: list[dict[str, Any]] = []
    discrete_total = 0
    integer_magnitudes: dict[str, int] = {}
    magnitudes: dict[str, str] = {}
    attached: dict[str, list[str]] = {}

    for obligation in OBLIGATION_LEDGER:
        record = sources[obligation.source]
        if obligation.kind == "magnitude":
            magnitudes[obligation.key] = _magnitude_expression(obligation, record)
            attached.setdefault(obligation.attached_to, []).append(obligation.key)
            continue
        if obligation.kind == "integer_magnitude":
            raw = _dig(record, obligation.path)
            integer_magnitudes[obligation.key] = int(sp.Integer(sp.sympify(raw)))
            attached.setdefault(obligation.attached_to, []).append(obligation.key)
            continue
        value = _dig(record, obligation.path)
        if obligation.kind == "boolean_met":
            if value is not True:
                raise PartialCreditError(
                    f"obligation {obligation.key} was declared met but is not"
                )
            met.append(obligation.key)
            continue
        if obligation.kind == "boolean_unmet":
            if value is not False:
                raise PartialCreditError(
                    f"obligation {obligation.key} was declared unmet but is not"
                )
            shortfall = obligation.outstanding
        elif obligation.kind == "count_to_zero":
            shortfall = int(value)
            if shortfall <= 0:
                raise PartialCreditError(
                    f"obligation {obligation.key} was declared unmet but reached target"
                )
        elif obligation.kind == "count_from_zero":
            if int(value) != 0:
                raise PartialCreditError(
                    f"obligation {obligation.key} was declared unstarted but is not"
                )
            shortfall = 1
        else:  # pragma: no cover - the ledger is a closed set of kinds
            raise PartialCreditError(f"unknown obligation kind: {obligation.kind}")
        discrete_total += shortfall
        failed.append(
            {
                "key": obligation.key,
                "kind": obligation.kind,
                "shortfall": str(shortfall),
                "meaning": obligation.meaning,
            }
        )

    for entry in failed:
        entry["margins"] = sorted(attached.get(entry["key"], []))

    return {
        "obligations_met": met,
        "obligations_failed": failed,
        "met_count": len(met),
        "unmet_obligation_count": len(failed),
        "discrete_shortfall_total": discrete_total,
        "discrete_shortfall_vector": [entry["shortfall"] for entry in failed],
        "integer_magnitudes": integer_magnitudes,
        "magnitudes": magnitudes,
        "margin_attachments": {key: sorted(value) for key, value in attached.items()},
    }


def _compare_candidates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    ladder: tuple[int, ...],
    keys: tuple[str, ...] = RANKING_KEYS,
) -> tuple[int, str, dict[str, Any] | None]:
    """Order two evaluated candidates by the declared keys.

    Returns ``(-1|0|1, deciding_key, witness)``.  A 0 with ``deciding_key == "tie"`` means every
    declared key agreed; a 0 with ``deciding_key == "unseparated"`` means the precision budget ran
    out.  Neither ever becomes a strict order.
    """

    for key in keys:
        if key in ("unmet_obligation_count", "discrete_shortfall_total"):
            a, b = left[key], right[key]
            if a != b:
                return (-1 if a < b else 1, key, {"left": str(a), "right": str(b)})
            continue
        if key in left["integer_magnitudes"]:
            a, b = left["integer_magnitudes"][key], right["integer_magnitudes"][key]
            if a != b:
                return (-1 if a < b else 1, key, {"left": str(a), "right": str(b)})
            continue
        verdict, witness = _cached_compare(
            left["magnitudes"][key], right["magnitudes"][key], ladder
        )
        if verdict == LESS:
            return -1, key, witness
        if verdict == GREATER:
            return 1, key, witness
        if verdict == UNSEPARATED:
            return 0, "unseparated", {"key": key, "verdict": UNSEPARATED}
    return 0, "tie", None


def _tiers(
    evaluated: Mapping[str, Mapping[str, Any]],
    candidate_ids: Sequence[str],
    ladder: tuple[int, ...],
    keys: tuple[str, ...] = RANKING_KEYS,
) -> tuple[list[list[str]], list[dict[str, Any]], list[dict[str, str]]]:
    """Group candidates into tiers no declared key strictly separates.

    Returns ``(tiers, separation_witnesses, unseparated_pairs)``.  Insertion is by exact pairwise
    comparison, so a pair the precision budget cannot decide is recorded and left in one tier
    rather than ordered on a guess.
    """

    order: list[str] = []
    unseparated: list[dict[str, str]] = []
    for candidate_id in candidate_ids:
        position = len(order)
        for index, placed in enumerate(order):
            sign, deciding, witness = _compare_candidates(
                evaluated[candidate_id], evaluated[placed], ladder, keys
            )
            if deciding == "unseparated" and witness is not None:
                pair = sorted((candidate_id, placed))
                entry = {"left": pair[0], "right": pair[1], "key": witness["key"]}
                if entry not in unseparated:
                    unseparated.append(entry)
            if sign < 0:
                position = index
                break
        order.insert(position, candidate_id)

    tiers: list[list[str]] = []
    witnesses: list[dict[str, Any]] = []
    for candidate_id in order:
        if tiers:
            sign, deciding, witness = _compare_candidates(
                evaluated[tiers[-1][0]], evaluated[candidate_id], ladder, keys
            )
            if sign == 0:
                tiers[-1].append(candidate_id)
                continue
            witnesses.append(
                {
                    "nearer": tiers[-1][0],
                    "farther": candidate_id,
                    "deciding_key": deciding,
                    **({"separation": witness} if witness else {}),
                }
            )
        tiers.append([candidate_id])
    for tier in tiers:
        tier.sort()
    return tiers, witnesses, unseparated


def _audit_tier_order(
    evaluated: Mapping[str, Mapping[str, Any]],
    tiers: Sequence[Sequence[str]],
    ladder: tuple[int, ...],
) -> None:
    """Check the emitted order is a real order, not an artifact of the insertion sweep.

    Insertion against a preorder that the precision budget renders intransitive could in
    principle emit a sequence that is not pairwise sorted.  Every pair of tier representatives is
    therefore re-compared here, and every member of a tier is re-checked against its own
    representative.  A published ranking that fails either check is refused.
    """

    representatives = [tier[0] for tier in tiers]
    for index, nearer in enumerate(representatives):
        for farther in representatives[index + 1 :]:
            sign, _key, _witness = _compare_candidates(
                evaluated[nearer], evaluated[farther], ladder
            )
            if sign >= 0:
                raise PartialCreditError(
                    "emitted tier order is not pairwise strict; refusing to publish a ranking"
                )
    for tier in tiers:
        for member in tier[1:]:
            sign, _key, _witness = _compare_candidates(
                evaluated[tier[0]], evaluated[member], ladder
            )
            if sign != 0:
                raise PartialCreditError(
                    "a tier member is separable from its own representative"
                )


def _conditional_lifespan_ceiling(
    a_known: str, bits: int
) -> dict[str, str]:
    """The already-proved ceiling ``2*log(2)/A_known`` on the conditional lifespan."""

    growth = bracket_expression(a_known, bits)
    if not growth.is_positive():
        raise PartialCreditError("A_known bracket is not provably positive")
    two_log_two = log_bracket(Fraction(4), bits)
    ceiling_lo = two_log_two.lo / growth.hi
    ceiling_hi = two_log_two.hi / growth.lo
    return {
        "expression": "2*log(2)/A_known",
        "lo": str(ceiling_lo),
        "hi": str(ceiling_hi),
        "bits": str(bits),
    }


def build_partial_credit(
    global_h7: Mapping[str, Any],
    finite_sobolev: Mapping[str, Any],
    block_gate: Mapping[str, Any],
    ladder: tuple[int, ...] = DECLARED_LADDER,
) -> dict[str, Any]:
    """Attach an exact structured distance to a uniform BLOCK and rank the candidates by it."""

    validate_sources(global_h7, finite_sobolev, block_gate)
    certificates = {item["candidate_id"]: item for item in global_h7["certificates"]}
    finite_records = {item["candidate_id"]: item for item in finite_sobolev["candidate_records"]}
    candidate_ids = sorted(certificates)

    gate_records = {item["candidate_id"]: item for item in block_gate["candidate_records"]}
    evaluated = {
        candidate_id: _evaluate_ledger(
            certificates[candidate_id],
            finite_records[candidate_id],
            gate_records[candidate_id],
        )
        for candidate_id in candidate_ids
    }

    # Every strict comparison is recorded with the rationals that separate it; unseparated pairs
    # are recorded too, and never ordered.
    tiers, witnesses, unseparated = _tiers(evaluated, candidate_ids, ladder)
    _audit_tier_order(evaluated, tiers, ladder)

    rank_of: dict[str, int] = {}
    tier_of: dict[str, int] = {}
    consumed = 0
    for tier_index, tier in enumerate(tiers, start=1):
        for candidate_id in tier:
            rank_of[candidate_id] = consumed + 1
            tier_of[candidate_id] = tier_index
        consumed += len(tier)

    records = []
    for candidate_id in candidate_ids:
        evaluation = evaluated[candidate_id]
        certificate = certificates[candidate_id]
        margins: dict[str, Any] = {}
        for obligation in OBLIGATION_LEDGER:
            if obligation.kind == "integer_magnitude":
                margins[obligation.key] = {
                    "attached_to": obligation.attached_to,
                    "direction": obligation.direction,
                    "kind": obligation.kind,
                    "exact": str(evaluation["integer_magnitudes"][obligation.key]),
                    "meaning": obligation.meaning,
                }
            elif obligation.kind == "magnitude":
                expression = evaluation["magnitudes"][obligation.key]
                bracket = _cached_bracket(expression, ladder[0])
                margins[obligation.key] = {
                    "attached_to": obligation.attached_to,
                    "direction": obligation.direction,
                    "kind": obligation.kind,
                    "exact": expression,
                    "lo": str(bracket.lo),
                    "hi": str(bracket.hi),
                    "bracket_bits": str(ladder[0]),
                    "meaning": obligation.meaning,
                }
        records.append(
            {
                "candidate_id": candidate_id,
                "decision": "BLOCK",
                "coefficients": certificate["coefficients"],
                "rank": rank_of[candidate_id],
                "tier": tier_of[candidate_id],
                "tier_size": len(tiers[tier_of[candidate_id] - 1]),
                "obligations_met": evaluation["obligations_met"],
                "obligations_failed": evaluation["obligations_failed"],
                "met_count": evaluation["met_count"],
                "unmet_obligation_count": evaluation["unmet_obligation_count"],
                "discrete_shortfall_total": str(evaluation["discrete_shortfall_total"]),
                "discrete_shortfall_vector": list(evaluation["discrete_shortfall_vector"]),
                "margins": margins,
                "conditional_lifespan_ceiling": _conditional_lifespan_ceiling(
                    evaluation["magnitudes"]["certified_linear_energy_growth"], ladder[0]
                ),
                "rank_is_a_proof": False,
            }
        )

    # The discrete layer is the integer shortfalls alone.  Keeping it separate from the magnitude
    # layer is what makes the finding legible: if these signatures are all equal, then a gate that
    # counted only unmet obligations would see twelve identical candidates.
    discrete_signatures = {
        (
            record["unmet_obligation_count"],
            record["discrete_shortfall_total"],
            tuple(record["discrete_shortfall_vector"]),
        )
        for record in records
    }
    magnitude_signatures = {
        tuple(record["margins"][key]["exact"] for key in sorted(record["margins"]))
        for record in records
    }
    swapped = tuple(
        key
        for key in (
            "unmet_obligation_count",
            "discrete_shortfall_total",
            "certified_linear_energy_growth",
            "uncancelled_slice_growth_multiplier",
            "unresolved_remainder_coefficient",
            "energy_equivalence_condition_number",
        )
    )
    gradient = {
        "distinct_tiers": len(tiers),
        "gradient_present": len(tiers) > 1,
        "tiers": [list(tier) for tier in tiers],
        "discrete_layer_separates": len(discrete_signatures) > 1,
        "discrete_layer_distinct_signatures": len(discrete_signatures),
        "magnitude_layer_separates": len(magnitude_signatures) > 1,
        "magnitude_layer_distinct_signatures": len(magnitude_signatures),
        "separating_keys": sorted({item["deciding_key"] for item in witnesses}),
        "unseparated_pairs": unseparated,
        "separation_witnesses": witnesses,
        "ordering_ladder_bits": [str(bits) for bits in ladder],
        "key_order_robustness": {
            "swapped_key_order": list(swapped),
            "tiers_under_swapped_key_order": [
                list(tier)
                for tier in _tiers(evaluated, candidate_ids, ladder, swapped)[0]
            ],
        },
    }
    gradient["key_order_robustness"]["ranking_invariant_under_key_swap"] = gradient[
        "key_order_robustness"
    ]["tiers_under_swapped_key_order"] == [list(tier) for tier in tiers]

    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "block_all_12_with_exact_partial_credit_ranking",
        "decision": "BLOCK",
        "claims": dict(CLAIMS_POLICY),
        "obligation_ledger_sha256": ledger_sha256(),
        "ranking_keys": list(RANKING_KEYS),
        "counts": {
            "selected": len(records),
            "blocked": len(records),
            "passed": 0,
            "promoted": 0,
            "distinct_tiers": len(tiers),
            "obligations_declared": len(OBLIGATION_LEDGER),
            "obligations_met_each": records[0]["met_count"],
            "obligations_failed_each": records[0]["unmet_obligation_count"],
        },
        "gradient_audit": gradient,
        "candidate_records": records,
        "upstream_sha256": {
            "global_h7": global_h7["content_sha256"],
            "finite_sobolev_no_go": finite_sobolev["content_sha256"],
            "block_gate": block_gate["content_sha256"],
        },
        "negative_controls": negative_controls(
            global_h7, finite_sobolev, block_gate, ladder
        ),
        "scope": (
            "A ranking of twelve BLOCKed candidates by an exact structured distance to the "
            "System9 completion contract. The verdict on every candidate is unchanged and "
            "remains BLOCK. The rank orders how much certified room each candidate leaves for a "
            "conditional lifespan; it is not a proof, a promotion, or evidence that the "
            "top-ranked candidate is provable. Every discrete shortfall is identical across the "
            "twelve, so the ordering rests entirely on the certificate's own exact magnitudes."
        ),
    }
    result = {**body, "content_sha256": _content_sha(body)}
    validate_receipt(result)
    return result


def negative_controls(
    global_h7: Mapping[str, Any],
    finite_sobolev: Mapping[str, Any],
    block_gate: Mapping[str, Any],
    ladder: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    """Controls that must all fail before a ranking is allowed to be published."""

    def reseal(value: dict[str, Any]) -> None:
        value["content_sha256"] = _content_sha(value)

    def rejects(mutate) -> bool:
        left = copy.deepcopy(dict(global_h7))
        middle = copy.deepcopy(dict(finite_sobolev))
        right = copy.deepcopy(dict(block_gate))
        mutate(left, middle, right)
        try:
            validate_sources(left, middle, right)
        except (KeyError, TypeError, ValueError):
            return True
        return False

    def assert_lifespan(left: dict[str, Any], _m: dict[str, Any], _r: dict[str, Any]) -> None:
        left["certificates"][0]["nonlinear_lifespan_proved"] = True
        reseal(left)

    def promote_gate(_l: dict[str, Any], _m: dict[str, Any], right: dict[str, Any]) -> None:
        right["candidate_records"][0]["decision"] = "PASS"
        reseal(right)

    def drop_candidate(left: dict[str, Any], _m: dict[str, Any], _r: dict[str, Any]) -> None:
        left["certificates"].pop()
        reseal(left)

    def open_claim(_l: dict[str, Any], _m: dict[str, Any], right: dict[str, Any]) -> None:
        right["claims"]["lifespan_proved"] = True
        reseal(right)

    # The gradient detector itself must be falsifiable: flatten the separating magnitudes onto
    # one shared value and the ranking has to collapse to a single tier of twelve.
    flattened = copy.deepcopy(dict(global_h7))
    reference = flattened["certificates"][0]
    for certificate in flattened["certificates"]:
        certificate["strongest_global_differential_inequality"]["A_known"] = reference[
            "strongest_global_differential_inequality"
        ]["A_known"]
        certificate["strongest_global_differential_inequality"]["Gamma_B"] = reference[
            "strongest_global_differential_inequality"
        ]["Gamma_B"]
        certificate["global_energy"]["H7_lower"] = reference["global_energy"]["H7_lower"]
        certificate["global_energy"]["H7_upper"] = reference["global_energy"]["H7_upper"]
    reseal(flattened)
    flat_finite = copy.deepcopy(dict(finite_sobolev))
    for record in flat_finite["candidate_records"]:
        record["absolute_growth_multiplier"] = flat_finite["candidate_records"][0][
            "absolute_growth_multiplier"
        ]
    reseal(flat_finite)
    flat_tiers = len(tier_partition(flattened, flat_finite, block_gate, ladder))

    # And it must not manufacture an order it cannot certify: starve the precision ladder and the
    # tight pairs have to come back unseparated rather than guessed.
    reference_tiers = len(
        tier_partition(global_h7, finite_sobolev, block_gate, DECLARED_LADDER)
    )
    starved_tiers = len(
        tier_partition(global_h7, finite_sobolev, block_gate, STARVED_LADDER)
    )

    return {
        "assert_lifespan_proved_on_one_candidate": {"rejected": rejects(assert_lifespan)},
        "promote_one_gate_record_to_pass": {"rejected": rejects(promote_gate)},
        "drop_one_candidate": {"rejected": rejects(drop_candidate)},
        "open_a_gate_claim": {"rejected": rejects(open_claim)},
        "identical_magnitudes_yield_no_gradient": {
            "rejected": flat_tiers == 1,
            "tiers_when_flattened": str(flat_tiers),
        },
        # A starved ladder may never see more structure than the declared one, and whenever the
        # declared ladder does find a gradient the starved ladder has to lose some of it.  On
        # inputs with no gradient at all there is nothing to lose, and the check says so.
        "starved_precision_yields_no_full_order": {
            "rejected": starved_tiers <= reference_tiers
            and (reference_tiers == 1 or starved_tiers < reference_tiers),
            "applicable": reference_tiers > 1,
            "tiers_at_declared_precision": str(reference_tiers),
            "tiers_at_starved_precision": str(starved_tiers),
        },
    }


def tier_partition(
    global_h7: Mapping[str, Any],
    finite_sobolev: Mapping[str, Any],
    block_gate: Mapping[str, Any],
    ladder: tuple[int, ...] = DECLARED_LADDER,
) -> list[list[str]]:
    """The tiering alone, nearest tier first, with no receipt built around it.

    Exposed so the falsification controls and their tests can ask what the ledger orders at a
    given precision without going through receipt validation.
    """

    certificates = {item["candidate_id"]: item for item in global_h7["certificates"]}
    finite_records = {item["candidate_id"]: item for item in finite_sobolev["candidate_records"]}
    gate_records = {item["candidate_id"]: item for item in block_gate["candidate_records"]}
    candidate_ids = sorted(certificates)
    evaluated = {
        candidate_id: _evaluate_ledger(
            certificates[candidate_id],
            finite_records[candidate_id],
            gate_records[candidate_id],
        )
        for candidate_id in candidate_ids
    }
    tiers, _witnesses, _unseparated = _tiers(evaluated, candidate_ids, ladder)
    return tiers


def validate_receipt(result: Mapping[str, Any]) -> None:
    """The receipt boundary: a rank may never become a pass, a promotion, or a claim."""

    if set(result) != {
        "schema_version",
        "campaign_id",
        "status",
        "decision",
        "claims",
        "obligation_ledger_sha256",
        "ranking_keys",
        "counts",
        "gradient_audit",
        "candidate_records",
        "upstream_sha256",
        "negative_controls",
        "scope",
        "content_sha256",
    }:
        raise PartialCreditError("partial-credit receipt boundary changed")
    if result["schema_version"] != RECEIPT_SCHEMA or result["campaign_id"] != CAMPAIGN_ID:
        raise PartialCreditError("partial-credit receipt identity changed")
    if result["decision"] != "BLOCK":
        raise PartialCreditError("partial credit never changes the verdict")
    if result["obligation_ledger_sha256"] != ledger_sha256():
        raise PartialCreditError("obligation ledger seal changed")
    if list(result["ranking_keys"]) != list(RANKING_KEYS):
        raise PartialCreditError("declared ranking key order changed")
    if any(result["claims"].values()) or set(result["claims"]) != set(CLAIMS_POLICY):
        raise PartialCreditError("partial credit opened a claim")
    if result["content_sha256"] != _content_sha(result):
        raise PartialCreditError("partial-credit content seal changed")
    records = result["candidate_records"]
    if len(records) != EXPECTED_CANDIDATES:
        raise PartialCreditError("partial-credit candidate set changed")
    if result["counts"]["passed"] or result["counts"]["promoted"]:
        raise PartialCreditError("partial credit produced a pass")
    ranks = sorted(record["rank"] for record in records)
    if ranks[0] != 1 or any(rank < 1 or rank > EXPECTED_CANDIDATES for rank in ranks):
        raise PartialCreditError("partial-credit ranks are out of range")
    for record in records:
        if record["decision"] != "BLOCK" or record["rank_is_a_proof"]:
            raise PartialCreditError("a candidate record escaped BLOCK")
        if not record["obligations_failed"]:
            raise PartialCreditError("a BLOCKed candidate reports no failed obligation")
    controls = result["negative_controls"]
    if set(controls) != {
        "assert_lifespan_proved_on_one_candidate",
        "promote_one_gate_record_to_pass",
        "drop_one_candidate",
        "open_a_gate_claim",
        "identical_magnitudes_yield_no_gradient",
        "starved_precision_yields_no_full_order",
    } or not all(control["rejected"] is True for control in controls.values()):
        raise PartialCreditError("partial-credit negative controls changed")
    if result["gradient_audit"]["gradient_present"] and not controls[
        "starved_precision_yields_no_full_order"
    ]["applicable"]:
        raise PartialCreditError("a published gradient was not put through the starved control")


def build_from_root(root: Path, ladder: tuple[int, ...] = DECLARED_LADDER) -> dict[str, Any]:
    root = root.resolve()
    return build_partial_credit(
        _load(root / GLOBAL_H7_PATH),
        _load(root / FINITE_SOBOLEV_PATH),
        _load(root / BLOCK_GATE_PATH),
        ladder,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-global-h7-partial-credit")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_from_root(arguments.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
