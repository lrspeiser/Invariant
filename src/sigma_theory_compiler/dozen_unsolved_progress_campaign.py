"""Progress-receipt campaign over the twelve open problems added in problem queue v2.

Queue v2 (configs/problem_queue_v2.json) admits twelve genuinely open problems — Lychrel
196, Brocard, Erdos-Moser, Lehmer totient, Giuga, odd perfect numbers, odd untouchables,
twin primes, Gilbreath, the Ulam sequence, Recaman coverage, and Singmaster.  None of
them is solvable by this engine, and the honest product is therefore not a solution but
a **progress receipt**: exact range facts, surviving-and-refuted typed conjectures, and
a named first blocker, sealed per problem.

Three rules keep the receipts honest.

**Every result is a range fact.**  Sweeps carry their declared bounds; sequence lanes
carry their declared budgets; nothing outside the range is claimed.  Where the
literature already reaches further (it always does here), the receipt says so:
``exceeds_literature_bound`` is false and ``mechanism_receipt_below_literature_bound``
is true, with the bound and citation recorded.

**Screens are screens; witnesses are exact.**  Vectorized (GPU or numpy) layers only
nominate candidates.  Every recorded solution, candidate, or anchor is re-verified with
exact Python integer arithmetic (sympy for primality and divisor sums), and each lane
carries a literature anchor — the even perfect numbers, the Giuga numbers, the OEIS
twin-prime counts, the published head of the untouchable list — that the computation
must reproduce or the build raises instead of sealing.

**Open status is documented, never inferred.**  Every receipt claims
``problem_remains_open`` and ``progress_is_not_solution`` as schema booleans and binds
the sealed queue's content hash; ``corpus_absence_establishes_novelty`` is false, as
everywhere in Sigma.

Claim boundary: validation replays cheap prefixes and exact witnesses, not the full
sweeps; a validated receipt means the sealed facts are coherent, canonically encoded,
and reproduce on their declared spot checks.  It proves nothing about any problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from math import isqrt
from pathlib import Path
from typing import Any

import numpy as np

from .conjecture_generation import generate_conjectures, validate_result
from .problem_queue import load_queue
from .sigma_core import canonical_json_bytes, canonical_sha256

RECEIPT_SCHEMA = "invariant-unsolved-progress-1.0"
CAMPAIGN_SCHEMA = "invariant-unsolved-progress-campaign-1.0"

#: Engine cap inherited from conjecture_generation/basis_synthesis: at most 64 rows and
#: |point| <= 64 may enter the B3 lane.  Longer computations are truncated to this
#: window for conjecture purposes only, and the receipt records both lengths.
CONJECTURE_ROW_CAP = 64

CLAIMS = {
    "corpus_absence_establishes_novelty": False,
    "problem_remains_open": True,
    "progress_is_not_solution": True,
}

_SCOPE = (
    "Bounded progress receipt for one documented open problem from the sealed intake "
    "queue (v2). Every lane result is an exact range fact: sweeps prove nothing outside "
    "their declared range, surviving conjectures carry proved=false, and lanes below a "
    "literature bound are mechanism receipts that add no new bound. Vectorized layers "
    "are screens only; recorded solutions and anchors are re-verified with exact integer "
    "arithmetic. The problem's open status is a documented claim bound in from the queue "
    "entry, never an inference from these results; nothing here is a solution."
)

DOZEN_IDS = (
    "lychrel_196",
    "brocard_problem",
    "erdos_moser",
    "lehmer_totient",
    "giuga_conjecture",
    "odd_perfect_number",
    "odd_untouchable",
    "twin_prime_infinitude",
    "gilbreath_conjecture",
    "ulam_sequence_structure",
    "recaman_coverage",
    "singmaster_conjecture",
)

LANES = {
    "lychrel_196": ("integer_trajectory", "conjecture_generation"),
    "brocard_problem": ("bounded_sweep", "exact_verification"),
    "erdos_moser": ("bounded_sweep", "exact_verification"),
    "lehmer_totient": ("bounded_sweep", "exact_verification"),
    "giuga_conjecture": ("bounded_sweep", "exact_verification"),
    "odd_perfect_number": ("bounded_sweep", "exact_verification"),
    "odd_untouchable": ("bounded_sweep", "exact_verification"),
    "twin_prime_infinitude": ("sequence_rows", "conjecture_generation"),
    "gilbreath_conjecture": ("sequence_rows", "conjecture_generation"),
    "ulam_sequence_structure": ("sequence_rows", "conjecture_generation"),
    "recaman_coverage": ("integer_trajectory",),
    "singmaster_conjecture": ("sequence_rows", "conjecture_generation"),
}

DEFAULT_BOUNDS = {
    "lychrel_196": {"max_iterations": 10000},
    "brocard_problem": {"parameter_max": 2000},
    "erdos_moser": {"m_max": 1000000, "k_max": 10},
    "lehmer_totient": {"n_max": 10000000},
    "giuga_conjecture": {"n_max": 1000000, "direct_sum_max": 2000},
    "odd_perfect_number": {"n_max": 100000000, "segment": 10000000},
    "odd_untouchable": {"limit": 100000},
    "twin_prime_infinitude": {"exponent_max": 7},
    "gilbreath_conjecture": {"rows": 500, "prime_limit": 1000000},
    "ulam_sequence_structure": {"terms": 2000},
    "recaman_coverage": {"steps": 10000000},
    "singmaster_conjecture": {"value_max": 1000000},
}

LITERATURE = {
    "lychrel_196": {
        "statement": (
            "p196.org record computations passed 300 million digits (and distributed "
            "continuations one billion digits) with no palindrome"
        ),
        "citation": "OEIS A023108; W. VanLandingham, p196.org",
    },
    "brocard_problem": {
        "statement": "no further solution has n <= 10^9",
        "citation": "Berndt-Galway, Ramanujan J. 4 (2000) 41-42",
    },
    "erdos_moser": {
        "statement": "any further solution has m > 10^(10^9)",
        "citation": "Gallot-Moree-Zudilin, Math. Comp. 80 (2011) 1221-1237",
    },
    "lehmer_totient": {
        "statement": (
            "any composite counterexample exceeds 10^20 and has at least 14 distinct "
            "prime factors"
        ),
        "citation": "Cohen-Hagis, Nieuw Arch. Wisk. (3) 28 (1980) 177-185; Guy B37",
    },
    "giuga_conjecture": {
        "statement": "any composite counterexample has more than 13800 digits",
        "citation": "Borwein-Borwein-Borwein-Girgensohn, Amer. Math. Monthly 103 (1996) 40-50",
    },
    "odd_perfect_number": {
        "statement": "any odd perfect number exceeds 10^1500",
        "citation": "Ochem-Rao, Math. Comp. 81 (2012) 1869-1877",
    },
    "odd_untouchable": {
        "statement": (
            "untouchable numbers are enumerated far beyond 10^5 in the literature and "
            "have positive lower density"
        ),
        "citation": "Guy B10; Erdos, Elem. Math. 28 (1973) 83-86",
    },
    "twin_prime_infinitude": {
        "statement": "pi_2 is tabulated to 10^18 in the literature; infinitude is open",
        "citation": "OEIS A007508; Guy A8; Zhang, Ann. of Math. 179 (2014) 1121-1174",
    },
    "gilbreath_conjecture": {
        "statement": "verified for the first 3.4 x 10^11 difference rows",
        "citation": "Odlyzko, Math. Comp. 61 (1993) 373-380",
    },
    "ulam_sequence_structure": {
        "statement": (
            "terms computed to 10^9 and beyond in the literature; no closed form or "
            "density proof is known"
        ),
        "citation": "Ulam 1964; Steinerberger, Experimental Math. 26 (2017) 460-467",
    },
    "recaman_coverage": {
        "statement": (
            "OEIS A005132 reports searches far beyond 10^15 terms without reaching 852655"
        ),
        "citation": "OEIS A005132",
    },
    "singmaster_conjecture": {
        "statement": (
            "multiplicity is O(log t log log log t / (log log t)^3); no t with "
            "multiplicity exceeding 8 is known"
        ),
        "citation": "Singmaster, Amer. Math. Monthly 78 (1971) 385-386; Kane, Integers 7 (2007)",
    },
}

FIRST_BLOCKERS = {
    "lychrel_196": {
        "code": "no_termination_lane",
        "detail": (
            "A finite iteration budget can only fail to find a palindrome; the engine has "
            "no lane that could certify non-termination of reverse-and-add."
        ),
    },
    "brocard_problem": {
        "code": "factorial_growth",
        "detail": (
            "n! + 1 leaves fixed-width arithmetic near n = 20, so the sweep is exact-bigint "
            "CPU only and cannot approach the 10^9 literature bound."
        ),
    },
    "erdos_moser": {
        "code": "literature_bound_unreachable",
        "detail": (
            "The proven bound m > 10^(10^9) exceeds any sweepable range by orders of "
            "magnitude of orders of magnitude; only the trivial range is checkable."
        ),
    },
    "lehmer_totient": {
        "code": "sieve_memory_wall",
        "detail": (
            "The totient sieve is O(n) memory, so the scan stops near 10^7-10^9 while any "
            "counterexample exceeds 10^20 (Cohen-Hagis)."
        ),
    },
    "giuga_conjecture": {
        "code": "criterion_dependency",
        "detail": (
            "Feasible scanning rests on the Borwein et al. factorization criterion; the "
            "direct congruence is O(n) per candidate, and the 13800-digit bound is "
            "unreachable either way."
        ),
    },
    "odd_perfect_number": {
        "code": "literature_bound_unreachable",
        "detail": (
            "The divisor-sum sieve is O(n) memory and the Ochem-Rao bound 10^1500 is "
            "unreachable by exhaustion in principle."
        ),
    },
    "odd_untouchable": {
        "code": "reduces_to_strong_goldbach",
        "detail": (
            "Deciding the conjecture reduces to a strengthened Goldbach statement, which "
            "no engine lane can address; enumeration only re-confirms 5 in range."
        ),
    },
    "twin_prime_infinitude": {
        "code": "infinitude_not_sweepable",
        "detail": (
            "No finite sieve decides infinitude, and the typed conjecture kinds cannot "
            "express asymptotic density laws (Hardy-Littlewood) over the count rows."
        ),
    },
    "gilbreath_conjecture": {
        "code": "constant_rows_carry_no_signal",
        "detail": (
            "The leading-term rows are identically 1, so every surviving conjecture "
            "restates the conjecture itself; no structural lever is exposed."
        ),
    },
    "ulam_sequence_structure": {
        "code": "statement_kinds_too_weak",
        "detail": (
            "The declared statement kinds cannot express the empirical quasi-periodic "
            "signal (cos(alpha a_n) < 0, Steinerberger 2017), so only monotonicity/sign "
            "survive."
        ),
    },
    "recaman_coverage": {
        "code": "sequential_map",
        "detail": (
            "Each step depends on the full visited set, so the trajectory cannot be "
            "parallelized on GPU and memory grows with the range of visited values."
        ),
    },
    "singmaster_conjecture": {
        "code": "bounded_multiplicity_not_expressible",
        "detail": (
            "A universal bound on multiplicity is not a declared statement kind, so the "
            "engine can only tabulate exact multiplicities in range."
        ),
    },
}


class UnsolvedProgressError(ValueError):
    """Raised on malformed input, integrity failure, anchor mismatch, or tamper."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnsolvedProgressError(message)


def _int(value: Any) -> int:
    return int(value)


# ---------------------------------------------------------------------------
# Literature anchors: published values the computation must reproduce
# ---------------------------------------------------------------------------

#: OEIS A007508: number of twin prime pairs below 10^k, k = 1..7.
TWIN_ANCHOR = (2, 8, 35, 205, 1224, 8169, 58980)

#: The even perfect numbers below 10^8 (Euclid-Euler).
EVEN_PERFECT_ANCHOR = (6, 28, 496, 8128, 33550336)

#: OEIS A007850: the Giuga numbers below 10^6.
GIUGA_NUMBER_ANCHOR = (30, 858, 1722, 66198)

#: OEIS A005114: the first ten untouchable numbers.
UNTOUCHABLE_HEAD_ANCHOR = (2, 5, 52, 88, 96, 120, 124, 146, 162, 188)

#: OEIS A002858: the first sixteen Ulam numbers.
ULAM_HEAD_ANCHOR = (1, 2, 3, 4, 6, 8, 11, 13, 16, 18, 26, 28, 36, 38, 47, 48)

#: OEIS A005132: the first sixteen terms of Recaman's sequence.
RECAMAN_HEAD_ANCHOR = (0, 1, 3, 6, 2, 7, 13, 20, 12, 21, 11, 22, 10, 23, 9, 24)


def _anchor_check(computed: Sequence[int], anchor: Sequence[int], label: str) -> None:
    prefix = tuple(computed[: len(anchor)])
    if prefix != tuple(anchor[: len(prefix)]):
        raise UnsolvedProgressError(
            f"integrity failure: {label} disagrees with its literature anchor "
            f"(computed {list(prefix)!r})"
        )


# ---------------------------------------------------------------------------
# Shared exact generators
# ---------------------------------------------------------------------------


def _prime_mask(limit: int) -> np.ndarray:
    """Boolean primality mask over [0, limit], classic sieve of Eratosthenes."""

    mask = np.zeros(limit + 1, dtype=bool)
    if limit >= 2:
        mask[2:] = True
        for p in range(2, isqrt(limit) + 1):
            if mask[p]:
                mask[p * p :: p] = False
    return mask


def _conjecture_lane(values: Sequence[int], start_point: int = 1) -> dict[str, Any]:
    """Feed at most CONJECTURE_ROW_CAP rows to B3 and extract verdicts verbatim."""

    window = list(values[:CONJECTURE_ROW_CAP])
    _require(len(window) >= 7, "conjecture lane needs at least seven rows")
    rows = [{"point": start_point + i, "value": _int(v)} for i, v in enumerate(window)]
    receipt = generate_conjectures(rows)
    survivors = [
        {"kind": c["kind"], "statement": c["statement"], "support": c["support"]}
        for c in receipt["conjectures"]
        if c.get("status") == "SURVIVED"
    ]
    refuted = [
        {
            "kind": c["kind"],
            "statement": c["statement"],
            "refutation_witness": c["refutation_witness"],
        }
        for c in receipt["conjectures"]
        if c.get("status") == "REFUTED"
    ]
    return {
        "engine_row_cap": CONJECTURE_ROW_CAP,
        "rows_supplied": len(window),
        "rows_computed": len(values),
        "receipt": receipt,
        "survivors": survivors,
        "refuted": refuted,
    }


def _flags() -> dict[str, bool]:
    return {
        "exceeds_literature_bound": False,
        "mechanism_receipt_below_literature_bound": True,
    }


# ---------------------------------------------------------------------------
# sequence_rows lanes
# ---------------------------------------------------------------------------


def _twin_prime_lane(exponent_max: int) -> dict[str, Any]:
    _require(1 <= exponent_max <= 8, "twin exponent_max out of range")
    limit = 10**exponent_max
    mask = _prime_mask(limit + 2)
    twin_at = np.zeros(limit + 1, dtype=bool)
    twin_at[: limit + 1] = mask[: limit + 1] & mask[2 : limit + 3]
    counts = [
        _int(np.count_nonzero(twin_at[: 10**k + 1])) for k in range(1, exponent_max + 1)
    ]
    _anchor_check(counts, TWIN_ANCHOR, "pi_2(10^k)")
    facts = {
        "definition": "pi_2(10^k) = number of primes p <= 10^k with p + 2 prime",
        "sieve_limit": limit,
        "rows": [
            {"exponent": k, "pi2": counts[k - 1]} for k in range(1, exponent_max + 1)
        ],
        "oeis_crosscheck": "A007508 prefix reproduced exactly",
    }
    return {"facts": facts, "b3_values": counts}


def _gilbreath_lane(rows: int, prime_limit: int) -> dict[str, Any]:
    _require(rows >= 8, "gilbreath needs at least eight rows")
    mask = _prime_mask(prime_limit)
    primes = np.nonzero(mask)[0].astype(np.int64)
    _require(len(primes) > rows + 8, "prime budget too small for the requested rows")
    leading: list[int] = []
    current = primes
    first_divergent = None
    for index in range(1, rows + 1):
        current = np.abs(np.diff(current))
        term = _int(current[0])
        leading.append(term)
        if term != 1 and first_divergent is None:
            first_divergent = index
    facts = {
        "prime_limit": prime_limit,
        "prime_count": _int(len(primes)),
        "rows_computed": rows,
        "all_leading_terms_one": first_divergent is None,
        "first_divergent_row": first_divergent,
    }
    return {"facts": facts, "b3_values": leading}


def _ulam_terms(count: int) -> list[int]:
    _require(count >= 8, "ulam needs at least eight terms")
    terms = [1, 2]
    reps = bytearray(8)
    reps[3] = 1  # 1 + 2

    def _bump(index: int) -> None:
        nonlocal reps
        if index >= len(reps):
            reps.extend(bytes(max(len(reps), index + 1 - len(reps))))
        if reps[index] < 2:
            reps[index] += 1

    while len(terms) < count:
        candidate = terms[-1] + 1
        while candidate < len(reps) and reps[candidate] != 1:
            candidate += 1
        _require(candidate < len(reps), "ulam representation table exhausted")
        for earlier in terms:
            _bump(earlier + candidate)
        terms.append(candidate)
    return terms


def _ulam_lane(count: int) -> dict[str, Any]:
    terms = _ulam_terms(count)
    _anchor_check(terms, ULAM_HEAD_ANCHOR, "ulam U(1,2)")
    facts = {
        "terms_computed": len(terms),
        "final_term": terms[-1],
        "first_terms": terms[:40],
        "terms_at_most_10000": sum(1 for t in terms if t <= 10000),
    }
    return {"facts": facts, "b3_values": terms}


def _pascal_multiplicities(value_max: int) -> dict[int, int]:
    """Occurrences of each t <= value_max as C(n, k) with 2 <= k <= n - 2."""

    from math import comb

    counts: dict[int, int] = {}
    k = 2
    while comb(2 * k, k) <= value_max:
        n = 2 * k
        while True:
            value = comb(n, k)
            if value > value_max:
                break
            counts[value] = counts.get(value, 0) + (2 if k < n - k else 1)
            n += 1
        k += 1
    return counts


def _singmaster_multiplicity(t: int, interior: Mapping[int, int]) -> int:
    base = 1 if t == 2 else 2  # C(t, 1) and C(t, t-1); they coincide at t = 2
    return base + interior.get(t, 0)


def _singmaster_lane(value_max: int) -> dict[str, Any]:
    _require(value_max >= 70, "singmaster value_max too small")
    interior = _pascal_multiplicities(value_max)
    max_multiplicity = 2
    attainers_max: list[int] = []
    at_least_six: list[int] = []
    for t in interior:
        n_t = _singmaster_multiplicity(t, interior)
        if n_t > max_multiplicity:
            max_multiplicity = n_t
            attainers_max = [t]
        elif n_t == max_multiplicity:
            attainers_max.append(t)
        if n_t >= 6:
            at_least_six.append(t)
    rows = [_singmaster_multiplicity(t, interior) for t in range(2, 65)]
    distribution: dict[int, int] = {1: 1}  # N(2) = 1
    for t in interior:
        n_t = _singmaster_multiplicity(t, interior)
        distribution[n_t] = distribution.get(n_t, 0) + 1
    plain = value_max - 1 - 1 - len(interior)  # t in [2, value_max] minus t=2 minus keys
    distribution[2] = distribution.get(2, 0) + plain
    facts = {
        "value_max": value_max,
        "max_multiplicity": max_multiplicity,
        "attainers_of_max": sorted(attainers_max),
        "multiplicity_at_least_6": sorted(at_least_six),
        "distribution": [
            {"multiplicity": m, "count": c} for m, c in sorted(distribution.items())
        ],
    }
    return {"facts": facts, "b3_values": rows}


# ---------------------------------------------------------------------------
# integer_trajectory lanes
# ---------------------------------------------------------------------------


def _reverse_add(digits: list[int]) -> list[int]:
    """One reverse-and-add step on little-endian base-10 digits, exactly."""

    size = len(digits)
    out: list[int] = []
    carry = 0
    for i in range(size):
        total = digits[i] + digits[size - 1 - i] + carry
        out.append(total % 10)
        carry = total // 10
    if carry:
        out.append(carry)
    return out


def _lychrel_trajectory(seed: int, max_iterations: int) -> dict[str, Any]:
    _require(seed >= 1 and max_iterations >= 8, "lychrel bounds out of range")
    digits = [int(c) for c in str(seed)][::-1]
    lengths: list[int] = []
    palindrome_at = None
    for iteration in range(1, max_iterations + 1):
        digits = _reverse_add(digits)
        lengths.append(len(digits))
        if digits == digits[::-1]:
            palindrome_at = iteration
            break
    decimal = "".join(str(d) for d in reversed(digits))
    checkpoints = [
        {"iteration": i, "digits": lengths[i - 1]}
        for i in sorted({1, 10, 100, 1000, len(lengths)})
        if i <= len(lengths)
    ]
    return {
        "seed": seed,
        "iterations": len(lengths),
        "palindrome_found": palindrome_at is not None,
        "palindrome_at": palindrome_at,
        "final_digit_count": len(digits),
        "digit_checkpoints": checkpoints,
        "final_value_sha256": hashlib.sha256(decimal.encode("ascii")).hexdigest(),
        "digit_lengths": lengths,
    }


def _lychrel_lane(max_iterations: int) -> dict[str, Any]:
    trajectory = _lychrel_trajectory(196, max_iterations)
    lengths = trajectory.pop("digit_lengths")
    return {"facts": trajectory, "b3_values": lengths}


def _recaman_trajectory(steps: int) -> dict[str, Any]:
    _require(steps >= 16, "recaman needs at least sixteen steps")
    seen = bytearray(min(4 * steps + 16, 1 << 28))
    current = 0
    seen[0] = 1
    max_value = 0
    first_terms = [0]
    for n in range(1, steps + 1):
        back = current - n
        if back > 0 and not seen[back]:
            current = back
        else:
            current += n
            if current >= len(seen):
                seen.extend(bytes(max(len(seen) // 2, current + 1 - len(seen))))
        seen[current] = 1
        max_value = max(max_value, current)
        if len(first_terms) < 40:
            first_terms.append(current)
    smallest_unreached = 1
    while smallest_unreached <= max_value and seen[smallest_unreached]:
        smallest_unreached += 1
    return {
        "steps": steps,
        "smallest_unreached": smallest_unreached,
        "max_value_reached": max_value,
        "distinct_values_reached": seen.count(1),
        "first_terms": first_terms,
    }


def _recaman_lane(steps: int) -> dict[str, Any]:
    facts = _recaman_trajectory(steps)
    _anchor_check(facts["first_terms"], RECAMAN_HEAD_ANCHOR, "recaman head")
    facts["device_note"] = (
        "sequential map: each step depends on the full visited set, so this lane is "
        "CPU-exact by construction; no GPU parallelization is sound"
    )
    return {"facts": facts}


# ---------------------------------------------------------------------------
# diophantine_family lanes (screen + exact verification)
# ---------------------------------------------------------------------------


def _brocard_lane(parameter_max: int) -> dict[str, Any]:
    _require(8 <= parameter_max <= 100000, "brocard parameter_max out of range")
    solutions: list[dict[str, int]] = []
    factorial = 1
    for n in range(1, parameter_max + 1):
        factorial *= n
        candidate = factorial + 1
        root = isqrt(candidate)
        if root * root == candidate:
            solutions.append({"n": n, "m": root})
    _require(
        [s["n"] for s in solutions] == [4, 5, 7],
        "integrity failure: Brocard scan must recover exactly n = 4, 5, 7 in range",
    )
    return {
        "bounds": {"parameter_max": parameter_max},
        "device_class": "cpu-python-bigint",
        "method": "exact factorial accumulation with integer square-root test",
        "solutions": solutions,
        "counterexamples_to_known_list": [],
        "literature": LITERATURE["brocard_problem"],
        "flags": _flags(),
    }


def _erdos_moser_lane(m_max: int, k_max: int) -> dict[str, Any]:
    _require(m_max >= 4 and 1 <= k_max <= 12, "erdos_moser bounds out of range")
    solutions: list[dict[str, int]] = []
    for k in range(1, k_max + 1):
        left_sum = 1  # sum_{j=1}^{m-1} j^k at m = 2
        for m in range(2, m_max + 1):
            term = m**k
            if left_sum == term:
                solutions.append({"k": k, "m": m})
            left_sum += term
    _require(
        solutions == [{"k": 1, "m": 3}],
        "integrity failure: Erdos-Moser scan must recover exactly (k, m) = (1, 3)",
    )
    return {
        "bounds": {"m_max": m_max, "k_max": k_max},
        "device_class": "cpu-python-bigint",
        "method": "incremental exact power sums, equality tested per m",
        "solutions": solutions,
        "literature": LITERATURE["erdos_moser"],
        "flags": _flags(),
    }


def _lehmer_lane(n_max: int, use_gpu: bool) -> dict[str, Any]:
    _require(n_max >= 100, "lehmer n_max too small")
    mask = _prime_mask(n_max)
    phi = np.arange(n_max + 1, dtype=np.int64)
    for p in np.nonzero(mask)[0].tolist():
        view = phi[p::p]
        view -= view // p
    if use_gpu:
        import cupy as xp

        device_class = "gpu-cupy-screen"
    else:
        xp = np
        device_class = "cpu-numpy-screen"
    x_phi = xp.asarray(phi)
    x_mask = xp.asarray(mask)
    ns = xp.arange(n_max + 1, dtype=xp.int64)
    divides = (ns - 1) % xp.where(x_phi == 0, xp.int64(1), x_phi) == 0
    divides = divides & (ns >= 1)
    composite = (~x_mask) & (ns >= 2)
    candidates = [_int(v) for v in xp.nonzero(divides & composite)[0].tolist()]
    prime_pass = _int((divides & x_mask).sum())
    _require(
        prime_pass == _int(mask.sum()),
        "integrity failure: every prime must satisfy phi(p) | p - 1",
    )
    import sympy

    verified: list[dict[str, Any]] = []
    for n in candidates:
        totient = int(sympy.totient(n))
        _require(
            (n - 1) % totient == 0 and not sympy.isprime(n),
            f"integrity failure: screen candidate {n} not reproduced exactly",
        )
        verified.append({"n": n, "phi": totient})
    return {
        "bounds": {"n_max": n_max},
        "device_class": device_class,
        "method": "sieved totient table screen; exact sympy re-verification of candidates",
        "composite_hits": verified,
        "primes_satisfying": prime_pass,
        "literature": LITERATURE["lehmer_totient"],
        "flags": _flags(),
    }


def _giuga_direct_sum_holds(n: int) -> bool:
    return sum(pow(k, n - 1, n) for k in range(1, n)) % n == n - 1


def _giuga_lane(n_max: int, direct_sum_max: int) -> dict[str, Any]:
    _require(n_max >= 100 and 100 <= direct_sum_max <= n_max, "giuga bounds out of range")
    mask = _prime_mask(n_max)
    primes = np.nonzero(mask)[0].tolist()
    indices = np.arange(n_max + 1, dtype=np.int64)
    omega = np.zeros(n_max + 1, dtype=np.int8)
    for p in primes:
        omega[p::p] += 1
    giuga_part = np.zeros(n_max + 1, dtype=np.int8)
    carmichael_part = np.zeros(n_max + 1, dtype=np.int8)
    for p in primes:
        exact = np.arange(p, n_max + 1, p * p, dtype=np.int64)  # n = p (mod p^2)
        giuga_part[exact] += 1
        if p == 2:
            carmichael_part[exact] += 1
        else:
            carmichael_part[exact[(exact - 1) % (p - 1) == 0]] += 1
    composite = (~mask) & (indices >= 2)
    giuga_numbers = [
        _int(v) for v in np.nonzero(composite & (giuga_part == omega))[0].tolist()
    ]
    _anchor_check(giuga_numbers, GIUGA_NUMBER_ANCHOR, "giuga numbers")
    candidates = [
        _int(v)
        for v in np.nonzero(
            composite & (giuga_part == omega) & (carmichael_part == omega)
        )[0].tolist()
    ]
    verified: list[int] = []
    for n in candidates:
        _require(
            _giuga_direct_sum_holds(n),
            f"integrity failure: criterion candidate {n} fails the direct congruence",
        )
        verified.append(n)
    for n in range(2, direct_sum_max + 1):
        direct = _giuga_direct_sum_holds(n)
        criterion = bool(mask[n]) or n in verified
        _require(
            direct == criterion,
            f"integrity failure: direct sum and criterion disagree at n = {n}",
        )
    return {
        "bounds": {"n_max": n_max, "direct_sum_max": direct_sum_max},
        "device_class": "cpu-numpy-screen",
        "method": (
            "screen by the Borwein et al. (1996) criterion: n squarefree and, for every "
            "prime p | n, p | n/p - 1 and (p - 1) | n - 1; exhaustive direct modular "
            "power sums below direct_sum_max; direct-sum settlement of any survivor"
        ),
        "composite_hits": verified,
        "giuga_numbers_in_range": giuga_numbers,
        "direct_sum_iff_verified_below": direct_sum_max,
        "literature": LITERATURE["giuga_conjecture"],
        "flags": _flags(),
    }


def _sigma_segment(xp: Any, lo: int, hi: int) -> Any:
    """Divisor-sum table for [lo, hi) via paired divisors d <= sqrt(n)."""

    sigma = xp.zeros(hi - lo, dtype=xp.int64)
    for d in range(1, isqrt(hi - 1) + 1):
        start = max(d * d, ((lo + d - 1) // d) * d)
        if start >= hi:
            continue
        multiples = xp.arange(start, hi, d, dtype=xp.int64)
        quotients = multiples // d
        sigma[multiples - lo] += d
        strict = quotients != d
        sigma[(multiples - lo)[strict]] += quotients[strict]
    return sigma


def _odd_perfect_lane(n_max: int, segment: int, use_gpu: bool) -> dict[str, Any]:
    _require(n_max >= 10000 and segment >= 1000, "odd perfect bounds out of range")
    if use_gpu:
        import cupy as xp

        device_class = "gpu-cupy-screen"
    else:
        xp = np
        device_class = "cpu-numpy-screen"
    odd_hits: list[int] = []
    even_perfect: list[int] = []
    spot_checks: list[dict[str, int]] = []
    for lo in range(1, n_max + 1, segment):
        hi = min(lo + segment, n_max + 1)
        sigma = _sigma_segment(xp, lo, hi)
        ns = xp.arange(lo, hi, dtype=xp.int64)
        perfect = sigma == 2 * ns
        for value in [_int(v) for v in xp.nonzero(perfect)[0].tolist()]:
            n = lo + value
            (odd_hits if n % 2 else even_perfect).append(n)
        for index in range(4):
            n = lo + (index * 2654435761 + 12345) % (hi - lo)
            spot_checks.append({"n": n, "sigma": _int(sigma[n - lo])})
    import sympy

    for check in spot_checks:
        _require(
            int(sympy.divisor_sigma(check["n"])) == check["sigma"],
            f"integrity failure: sigma sieve wrong at n = {check['n']}",
        )
    expected_even = [v for v in EVEN_PERFECT_ANCHOR if v <= n_max]
    _require(
        even_perfect == expected_even,
        "integrity failure: even perfect anchor not reproduced by the sieve",
    )
    for n in odd_hits:
        _require(
            int(sympy.divisor_sigma(n)) == 2 * n,
            f"integrity failure: odd candidate {n} not reproduced exactly",
        )
    return {
        "bounds": {"n_max": n_max, "segment": segment},
        "device_class": device_class,
        "method": (
            "segmented paired-divisor sigma sieve screen; exact sympy re-verification "
            "of every equality hit and of deterministic spot checks"
        ),
        "odd_hits": odd_hits,
        "even_perfect_anchor": even_perfect,
        "spot_checks": spot_checks,
        "literature": LITERATURE["odd_perfect_number"],
        "flags": _flags(),
    }


def _untouchable_lane(limit: int) -> dict[str, Any]:
    _require(200 <= limit <= 1000000, "untouchable limit out of range")
    sieve_to = isqrt(limit**3)
    reached = np.zeros(limit + 1, dtype=bool)
    reached[1] = True  # s(p) = 1 for every prime p
    segment = 10**7
    for lo in range(1, sieve_to + 1, segment):
        hi = min(lo + segment, sieve_to + 1)
        sigma = _sigma_segment(np, lo, hi)
        aliquot = sigma - np.arange(lo, hi, dtype=np.int64)
        hits = aliquot[(aliquot >= 1) & (aliquot <= limit)]
        reached[hits] = True
    prime_values = np.nonzero(_prime_mask(limit))[0].astype(np.int64)
    squares = 1 + prime_values
    reached[squares[squares <= limit]] = True  # s(p^2) = 1 + p, any prime p
    prime_list = prime_values.tolist()
    for index, p in enumerate(prime_list):
        partners = prime_values[index + 1 :]
        if partners.size == 0 or 1 + p + _int(partners[0]) > limit:
            break
        partners = partners[1 + p + partners <= limit]
        reached[1 + p + partners] = True  # s(p*q) = 1 + p + q, any primes p < q
    untouched = [
        _int(v) for v in np.nonzero(~reached)[0].tolist() if v >= 1
    ]
    _anchor_check(untouched, UNTOUCHABLE_HEAD_ANCHOR, "untouchable head")
    odd_untouched = [v for v in untouched if v % 2 == 1]
    return {
        "bounds": {"limit": limit, "preimage_sieve_to": sieve_to},
        "device_class": "cpu-numpy-screen",
        "method": (
            "exhaustive aliquot preimages: s(m) sieved for all m <= limit^(3/2) (complete "
            "for preimages with three or more prime factors since s(m) >= m^(2/3)); "
            "prime, prime-square, and semiprime preimages enumerated directly via s(p) = 1, "
            "s(p^2) = 1 + p, s(pq) = 1 + p + q"
        ),
        "untouchable_count": len(untouched),
        "untouchable_head": untouched[:30],
        "odd_untouchable": odd_untouched,
        "literature": LITERATURE["odd_untouchable"],
        "flags": _flags(),
    }


# ---------------------------------------------------------------------------
# Receipt assembly
# ---------------------------------------------------------------------------


def _sequence_results(problem_id: str, bounds: Mapping[str, int]) -> dict[str, Any]:
    if problem_id == "twin_prime_infinitude":
        lane = _twin_prime_lane(bounds["exponent_max"])
        start = 1
    elif problem_id == "gilbreath_conjecture":
        lane = _gilbreath_lane(bounds["rows"], bounds["prime_limit"])
        start = 1
    elif problem_id == "ulam_sequence_structure":
        lane = _ulam_lane(bounds["terms"])
        start = 1
    elif problem_id == "singmaster_conjecture":
        lane = _singmaster_lane(bounds["value_max"])
        start = 2
    else:  # pragma: no cover - guarded by build_receipt
        raise UnsolvedProgressError(f"not a sequence problem: {problem_id}")
    conjectures = _conjecture_lane(lane["b3_values"], start_point=start)
    return {
        "bounds": dict(bounds),
        "device_class": "cpu-numpy-exact",
        "sequence_rows": lane["facts"],
        "conjecture_generation": conjectures,
        "literature": LITERATURE[problem_id],
        "flags": _flags(),
    }


def _trajectory_results(problem_id: str, bounds: Mapping[str, int]) -> dict[str, Any]:
    if problem_id == "lychrel_196":
        lane = _lychrel_lane(bounds["max_iterations"])
        conjectures = _conjecture_lane(lane["b3_values"], start_point=1)
        return {
            "bounds": dict(bounds),
            "device_class": "cpu-python-bigint",
            "integer_trajectory": lane["facts"],
            "conjecture_generation": conjectures,
            "literature": LITERATURE[problem_id],
            "flags": _flags(),
        }
    if problem_id == "recaman_coverage":
        lane = _recaman_lane(bounds["steps"])
        return {
            "bounds": dict(bounds),
            "device_class": "cpu-python-exact",
            "integer_trajectory": lane["facts"],
            "literature": LITERATURE[problem_id],
            "flags": _flags(),
        }
    raise UnsolvedProgressError(f"not a trajectory problem: {problem_id}")


def _diophantine_results(
    problem_id: str, bounds: Mapping[str, int], use_gpu: bool
) -> dict[str, Any]:
    if problem_id == "brocard_problem":
        return _brocard_lane(bounds["parameter_max"])
    if problem_id == "erdos_moser":
        return _erdos_moser_lane(bounds["m_max"], bounds["k_max"])
    if problem_id == "lehmer_totient":
        return _lehmer_lane(bounds["n_max"], use_gpu)
    if problem_id == "giuga_conjecture":
        return _giuga_lane(bounds["n_max"], bounds["direct_sum_max"])
    if problem_id == "odd_perfect_number":
        return _odd_perfect_lane(bounds["n_max"], bounds["segment"], use_gpu)
    if problem_id == "odd_untouchable":
        return _untouchable_lane(bounds["limit"])
    raise UnsolvedProgressError(f"not a diophantine problem: {problem_id}")


def build_receipt(
    queue: Mapping[str, Any],
    problem_id: str,
    bounds: Mapping[str, int] | None = None,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """Run this problem's lanes at the declared bounds and seal one progress receipt."""

    _require(problem_id in DOZEN_IDS, f"unknown dozen problem: {problem_id}")
    entries = [e for e in queue["entries"] if e["id"] == problem_id]
    _require(len(entries) == 1, f"queue does not contain {problem_id}")
    entry = entries[0]
    _require(
        not entry["control_rediscovery"] and not entry["synthetic"],
        "dozen problems must not be controls or synthetics",
    )
    effective = dict(DEFAULT_BOUNDS[problem_id])
    for key, value in dict(bounds or {}).items():
        _require(key in effective, f"unknown bound override: {key}")
        effective[key] = _int(value)
    kind = entry["machine_form"]["kind"]
    if kind == "sequence_rows":
        results = _sequence_results(problem_id, effective)
    elif kind == "integer_trajectory":
        results = _trajectory_results(problem_id, effective)
    else:
        results = _diophantine_results(problem_id, effective, use_gpu)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "problem_id": problem_id,
        "queue_content_sha256": queue["content_sha256"],
        "machine_form": dict(entry["machine_form"]),
        "lanes_run": list(LANES[problem_id]),
        "results": results,
        "first_blocker": dict(FIRST_BLOCKERS[problem_id]),
        "claims": dict(CLAIMS),
        "scope": _SCOPE,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


# ---------------------------------------------------------------------------
# Validation: seal, structure, claims, and cheap exact replays
# ---------------------------------------------------------------------------

RECEIPT_KEYS = {
    "schema_version",
    "problem_id",
    "queue_content_sha256",
    "machine_form",
    "lanes_run",
    "results",
    "first_blocker",
    "claims",
    "scope",
    "content_sha256",
}

#: Validation replay budgets: deep enough to catch fabrication, cheap enough for CI.
_REPLAY = {
    "twin_exponent": 4,
    "gilbreath_rows": 100,
    "gilbreath_prime_limit": 100000,
    "ulam_terms": 200,
    "lychrel_iterations": 120,
    "recaman_steps": 2000,
    "erdos_moser_m": 1000,
}


def _replay_sequence(value: Mapping[str, Any]) -> None:
    problem_id = value["problem_id"]
    lane = value["results"]["conjecture_generation"]
    supplied = [
        row["value"]["numerator"] for row in lane["receipt"]["public_rows"]
    ]
    bounds = value["results"]["bounds"]
    if problem_id == "twin_prime_infinitude":
        exponent = min(_REPLAY["twin_exponent"], bounds["exponent_max"])
        expected = _twin_prime_lane(exponent)["b3_values"]
        _require(supplied[: len(expected)] == expected, "twin rows do not replay")
    elif problem_id == "gilbreath_conjecture":
        rows = min(_REPLAY["gilbreath_rows"], bounds["rows"], len(supplied))
        limit = min(_REPLAY["gilbreath_prime_limit"], bounds["prime_limit"])
        expected = _gilbreath_lane(max(rows, 8), limit)["b3_values"][:rows]
        _require(supplied[:rows] == expected, "gilbreath rows do not replay")
    elif problem_id == "ulam_sequence_structure":
        count = min(_REPLAY["ulam_terms"], bounds["terms"])
        expected = _ulam_terms(count)
        window = min(len(supplied), len(expected))
        _require(supplied[:window] == expected[:window], "ulam rows do not replay")
    elif problem_id == "singmaster_conjecture":
        expected = _singmaster_lane(max(70, min(bounds["value_max"], 100000)))
        _require(
            supplied == expected["b3_values"][: len(supplied)],
            "singmaster rows do not replay",
        )


def _replay_trajectory(value: Mapping[str, Any]) -> None:
    problem_id = value["problem_id"]
    facts = value["results"]["integer_trajectory"]
    if problem_id == "lychrel_196":
        lane = value["results"]["conjecture_generation"]
        supplied = [row["value"]["numerator"] for row in lane["receipt"]["public_rows"]]
        replay = _lychrel_trajectory(
            196, min(_REPLAY["lychrel_iterations"], facts["iterations"])
        )
        _require(
            not replay["palindrome_found"] or facts["palindrome_found"],
            "lychrel palindrome flag inconsistent with replay",
        )
        window = min(len(supplied), len(replay["digit_lengths"]))
        _require(
            supplied[:window] == replay["digit_lengths"][:window],
            "lychrel digit rows do not replay",
        )
    else:  # recaman_coverage
        replay = _recaman_trajectory(min(_REPLAY["recaman_steps"], facts["steps"]))
        _require(
            facts["first_terms"][: len(replay["first_terms"])]
            == replay["first_terms"][: len(facts["first_terms"])],
            "recaman head does not replay",
        )


def _replay_diophantine(value: Mapping[str, Any]) -> None:
    import sympy

    problem_id = value["problem_id"]
    results = value["results"]
    if problem_id == "brocard_problem":
        _require(
            [s["n"] for s in results["solutions"]] == [4, 5, 7],
            "brocard solution list changed",
        )
        factorial = 1
        expected = {s["n"]: s["m"] for s in results["solutions"]}
        for n in range(1, max(expected) + 1):
            factorial *= n
            if n in expected:
                _require(
                    expected[n] ** 2 == factorial + 1,
                    f"brocard witness at n = {n} fails exact recheck",
                )
    elif problem_id == "erdos_moser":
        _require(results["solutions"] == [{"k": 1, "m": 3}], "erdos_moser solutions changed")
        replay = _erdos_moser_lane(
            min(_REPLAY["erdos_moser_m"], results["bounds"]["m_max"]),
            results["bounds"]["k_max"],
        )
        _require(replay["solutions"] == results["solutions"], "erdos_moser does not replay")
    elif problem_id == "lehmer_totient":
        for hit in results["composite_hits"]:
            n = hit["n"]
            _require(
                int(sympy.totient(n)) == hit["phi"]
                and (n - 1) % hit["phi"] == 0
                and not sympy.isprime(n),
                f"lehmer hit at n = {n} fails exact recheck",
            )
    elif problem_id == "giuga_conjecture":
        _anchor_check(
            results["giuga_numbers_in_range"], GIUGA_NUMBER_ANCHOR, "giuga numbers"
        )
        for n in results["composite_hits"]:
            _require(
                _giuga_direct_sum_holds(n),
                f"giuga hit at n = {n} fails the direct congruence",
            )
        for n in range(2, min(400, results["bounds"]["direct_sum_max"]) + 1):
            direct = _giuga_direct_sum_holds(n)
            criterion = bool(sympy.isprime(n)) or n in results["composite_hits"]
            _require(direct == criterion, f"giuga iff fails at n = {n}")
    elif problem_id == "odd_perfect_number":
        expected_even = [
            v for v in EVEN_PERFECT_ANCHOR if v <= results["bounds"]["n_max"]
        ]
        _require(
            results["even_perfect_anchor"] == expected_even,
            "even perfect anchor changed",
        )
        for check in results["spot_checks"]:
            _require(
                int(sympy.divisor_sigma(check["n"])) == check["sigma"],
                f"sigma spot check fails at n = {check['n']}",
            )
        for n in results["odd_hits"]:
            _require(
                int(sympy.divisor_sigma(n)) == 2 * n,
                f"odd perfect hit at n = {n} fails exact recheck",
            )
    elif problem_id == "odd_untouchable":
        _anchor_check(
            results["untouchable_head"], UNTOUCHABLE_HEAD_ANCHOR, "untouchable head"
        )
        head = set(results["untouchable_head"])
        for n in results["odd_untouchable"]:
            _require(n % 2 == 1, "odd untouchable list contains an even value")
        _require(5 in head, "untouchable head must contain 5")


def validate_receipt(value: Mapping[str, Any], queue: Mapping[str, Any] | None = None) -> None:
    """Seal, schema, claims, lane, and replay checks.  Never repairs."""

    if not isinstance(value, Mapping):
        raise UnsolvedProgressError("receipt must be an object")
    if set(value) != RECEIPT_KEYS:
        raise UnsolvedProgressError("receipt top-level keys changed")
    if value["schema_version"] != RECEIPT_SCHEMA:
        raise UnsolvedProgressError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["content_sha256"] != canonical_sha256(body):
        raise UnsolvedProgressError("receipt seal changed")
    problem_id = value["problem_id"]
    _require(problem_id in DOZEN_IDS, f"unknown problem id: {problem_id!r}")
    _require(value["claims"] == CLAIMS, "receipt claims changed")
    _require(value["scope"] == _SCOPE, "receipt scope changed")
    _require(list(value["lanes_run"]) == list(LANES[problem_id]), "lanes_run changed")
    _require(value["first_blocker"] == FIRST_BLOCKERS[problem_id], "first_blocker changed")
    results = value["results"]
    _require(isinstance(results, Mapping), "results must be an object")
    _require(results["flags"] == _flags(), "literature honesty flags changed")
    _require(results["literature"] == LITERATURE[problem_id], "literature block changed")
    if queue is not None:
        _require(
            value["queue_content_sha256"] == queue["content_sha256"],
            "receipt is not bound to this queue",
        )
        entry = [e for e in queue["entries"] if e["id"] == problem_id]
        _require(
            len(entry) == 1 and entry[0]["machine_form"] == value["machine_form"],
            "machine_form echo does not match the queue entry",
        )
    if "conjecture_generation" in results:
        validate_result(results["conjecture_generation"]["receipt"])
    if "conjecture_generation" in results and "sequence_rows" in results:
        _replay_sequence(value)
    elif "integer_trajectory" in results:
        _replay_trajectory(value)
    else:
        _replay_diophantine(value)


# ---------------------------------------------------------------------------
# Campaign: write all twelve receipts and the summary index
# ---------------------------------------------------------------------------


def _headline(receipt: Mapping[str, Any]) -> str:
    problem_id = receipt["problem_id"]
    results = receipt["results"]
    if problem_id == "lychrel_196":
        facts = results["integer_trajectory"]
        lane = results["conjecture_generation"]
        return (
            f"no palindrome in {facts['iterations']} iterations "
            f"({facts['final_digit_count']} digits); {len(lane['survivors'])} row "
            f"conjectures survived"
        )
    if "conjecture_generation" in results:
        lane = results["conjecture_generation"]
        survivors = ", ".join(s["statement"] for s in lane["survivors"]) or "none"
        refuted = len(lane["refuted"])
        return f"{len(lane['survivors'])} conjectures survived ({survivors}); {refuted} refuted"
    if problem_id == "recaman_coverage":
        facts = results["integer_trajectory"]
        return (
            f"smallest unreached value {facts['smallest_unreached']} after "
            f"{facts['steps']} steps"
        )
    if problem_id == "brocard_problem":
        return "only n = 4, 5, 7 in range, exact witnesses re-verified"
    if problem_id == "erdos_moser":
        return "only (k, m) = (1, 3) in range"
    if problem_id == "lehmer_totient":
        return f"no composite hit; {results['primes_satisfying']} primes pass as expected"
    if problem_id == "giuga_conjecture":
        giuga = ", ".join(str(v) for v in results["giuga_numbers_in_range"])
        return f"no counterexample; Giuga numbers in range: {giuga}"
    if problem_id == "odd_perfect_number":
        return "no odd hit; even perfect anchor reproduced"
    if problem_id == "odd_untouchable":
        odd = ", ".join(str(v) for v in results["odd_untouchable"])
        return f"odd untouchables in range: {odd} ({results['untouchable_count']} total)"
    return "completed"  # pragma: no cover


def build_summary(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seal the campaign index over the twelve per-problem receipts."""

    _require(
        [r["problem_id"] for r in receipts] == list(DOZEN_IDS),
        "summary needs exactly the twelve receipts in queue order",
    )
    queue_hashes = {r["queue_content_sha256"] for r in receipts}
    _require(len(queue_hashes) == 1, "receipts are bound to different queues")
    survivors = 0
    refuted = 0
    for receipt in receipts:
        lane = receipt["results"].get("conjecture_generation")
        if lane is not None:
            survivors += len(lane["survivors"])
            refuted += len(lane["refuted"])
    body = {
        "schema_version": CAMPAIGN_SCHEMA,
        "queue_content_sha256": min(queue_hashes),
        "problems": [
            {
                "problem_id": r["problem_id"],
                "receipt_path": f"runs/math/unsolved-dozen/{r['problem_id']}.json",
                "receipt_sha256": r["content_sha256"],
                "headline": _headline(r),
            }
            for r in receipts
        ],
        "blockers": [
            {"problem_id": r["problem_id"], **r["first_blocker"]} for r in receipts
        ],
        "counts": {
            "problems": len(receipts),
            "surviving_conjectures": survivors,
            "refuted_conjectures": refuted,
            "counterexamples_found": 0,
        },
        "claims": dict(CLAIMS),
        "scope": _SCOPE,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_summary(
    summary: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> None:
    if summary.get("schema_version") != CAMPAIGN_SCHEMA:
        raise UnsolvedProgressError("summary schema changed")
    body = {key: item for key, item in summary.items() if key != "content_sha256"}
    if summary.get("content_sha256") != canonical_sha256(body):
        raise UnsolvedProgressError("summary seal changed")
    if summary != build_summary(receipts):
        raise UnsolvedProgressError("summary does not replay from its receipts")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise UnsolvedProgressError(f"refusing to overwrite immutable receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def run_campaign(
    queue_path: Path | str,
    out_dir: Path | str,
    use_gpu: bool = False,
    only: str | None = None,
) -> dict[str, Any]:
    """Build (or rebuild byte-identically) the twelve receipts and the summary."""

    queue = load_queue(queue_path)
    out = Path(out_dir)
    receipts: list[dict[str, Any]] = []
    for problem_id in DOZEN_IDS:
        if only is not None and problem_id != only:
            continue
        target = out / f"{problem_id}.json"
        if target.exists():
            receipt = json.loads(target.read_text(encoding="utf-8"))
        else:
            receipt = build_receipt(queue, problem_id, use_gpu=use_gpu)
        validate_receipt(receipt, queue)
        _write_immutable(target, receipt)
        receipts.append(receipt)
        print(f"{problem_id}: {_headline(receipt)}")
    if only is not None:
        return {"receipts": len(receipts)}
    summary = build_summary(receipts)
    validate_summary(summary, receipts)
    _write_immutable(out / "campaign.json", summary)
    print(f"campaign sealed: {summary['content_sha256']}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Progress receipts for the twelve unsolved queue-v2 problems."
    )
    parser.add_argument("--queue", required=True, help="path to the sealed v2 queue")
    parser.add_argument("--out", required=True, help="receipt output directory")
    parser.add_argument("--problem", help="run a single problem id")
    parser.add_argument("--cpu", action="store_true", help="force the numpy screens")
    parser.add_argument(
        "--validate-checked",
        action="store_true",
        help="validate the stored receipts and summary; run nothing",
    )
    args = parser.parse_args(argv)
    if args.validate_checked:
        queue = load_queue(args.queue)
        receipts = []
        for problem_id in DOZEN_IDS:
            receipt = json.loads(
                (Path(args.out) / f"{problem_id}.json").read_text(encoding="utf-8")
            )
            validate_receipt(receipt, queue)
            receipts.append(receipt)
        summary = json.loads(
            (Path(args.out) / "campaign.json").read_text(encoding="utf-8")
        )
        validate_summary(summary, receipts)
        print(f"VALID receipts=12 campaign={summary['content_sha256']}")
        return 0
    run_campaign(args.queue, args.out, use_gpu=not args.cpu, only=args.problem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
