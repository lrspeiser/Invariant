"""Generalized exponent-Diophantine GPU sweeper for the queue's FLT-adjacent targets.

The scheduler's discovery receipts carry a typed blocker — ``missing_sweeper:
diophantine_family`` — wherever a queue entry's machine form is an equation rather than
a sequence.  This module is that sweeper.  It attacks the three exponent-Diophantine
targets admitted in problem queue v3 (``beal_conjecture``, ``fermat_catalan``,
``erdos_straus_sweeper_target``) over declared finite boxes, on GPU when available.

The load-bearing discipline is the **fp64 -> exact funnel**.  For power equations
``A^x + B^y = C^z`` the screen works in log space: ``log S = logaddexp(x log A, y log B)``
and, for each declared right-hand exponent, the only integer base that can match is
``round(exp(log S / z))``.  A lane is a *near-hit* when that rounded base sits within a
declared tolerance ``max(eps_rel * C_hat, delta_floor)``.  The box is admitted only
after an a-priori resolution proof (``C_hat`` small enough that fp64 separates
consecutive integers by a wide margin), so every true solution is guaranteed to be
flagged.  fp64 cannot flag less: on *dominated* lanes, where the smaller power falls
below fp64's 2^-52 relative resolution of the larger, the screen has no information
and must hand the lane over, so near-hit volume is driven by those lanes.  Every
near-hit therefore enters the exact CPU layer, which settles it in two exact integer
stages: a congruence check of the candidate identity modulo the Mersenne prime
2^61 - 1 (pure Python modular exponentiation — an exact rejection, since a true
solution satisfies the congruence for every modulus), then full big-integer
confirmation of every congruence survivor.  The receipt records the whole funnel:
lanes screened, fp64 near-hits, exact congruence rejections, exact full
confirmations, exact full rejections.  The screen is never trusted for anything.

Mode honesty rules:

* **beal** — every exactly confirmed solution has ``gcd(A, B, C)`` computed exactly.
  A coprime confirmed solution would be a Beal counterexample: the receipt then says
  ``COUNTEREXAMPLE_CANDIDATE``, carries a mandatory independent pure-Python
  re-verification (iterated multiplication, hand-rolled Euclid), and raises a loud
  claims flag.  The expected outcome is none; known ``gcd > 1`` families such as
  ``3^3 + 6^3 = 3^5`` MUST be found and labeled ``KNOWN_COMMON_FACTOR_FAMILY`` — that
  rediscovery is the validation control, and a box that covers the control but fails
  to find it refuses to seal.
* **fermat_catalan** — coprime ``x^p + y^q = z^r`` with ``1/p + 1/q + 1/r < 1``.  The
  ten known solutions are built in (exactly verified in code); every one reachable in
  the declared box must be rediscovered and labeled ``KNOWN_SOLUTION_REDISCOVERED``.
  Anything else exactly confirmed is ``NEW_TO_BUILTIN_TABLE`` with a prior-art note
  and no novelty claim — corpus absence never establishes novelty.
* **erdos_straus** — witnesses for ``4/n = 1/x + 1/y + 1/z`` over a declared range.
  Four residue classes (11/12 of the range) get parametric witnesses whose defining
  identities are verified symbolically in exact rational arithmetic; the hard class
  ``n = 1 (mod 12)`` runs a GPU-assisted exact search (int64 modular arithmetic with
  proven-in-range intermediates), completed by an exhaustive per-``x`` divisor search
  on the CPU.  Every hard-class witness is re-verified exactly; parametric classes are
  verified on deterministic strided samples.  Full witness tables (~10^7 rows) would
  dwarf the receipt, so storage is counts + samples + schedule-relative extremal
  minimal-x witnesses, and the receipt says exactly that.

Claim boundary: a finite box proves nothing outside itself.  Every documented search
landscape in the literature (Norvig's Beal search notes; the Fermat-Catalan record;
the Erdos-Straus verification to 10^17) exceeds any feasible box here, so all three
receipts are mechanism receipts and their claims blocks say so.  Floats are forbidden
in receipts, as in every Sigma receipt; tolerances appear as strings.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from math import gcd, log
from pathlib import Path
from typing import Any

import numpy as np

from .problem_queue import ProblemQueueError, load_queue
from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-exponent-diophantine-sweep-1.0"

MODE_PROBLEM_IDS = {
    "beal": "beal_conjecture",
    "fermat_catalan": "fermat_catalan",
    "erdos_straus": "erdos_straus_sweeper_target",
}

DECISION_NO_COUNTEREXAMPLE = "NO_COUNTEREXAMPLE_IN_BOX"
DECISION_COUNTEREXAMPLE_CANDIDATE = "COUNTEREXAMPLE_CANDIDATE"
DECISION_KNOWNS_ONLY = "KNOWN_SOLUTIONS_REDISCOVERED_ONLY"
DECISION_NEW_TO_TABLE = "NEW_TO_BUILTIN_TABLE_PRESENT"
DECISION_NO_SOLUTION = "NO_SOLUTION_IN_BOX"
DECISION_NO_UNSOLVABLE = "NO_UNSOLVABLE_N_IN_RANGE"
DECISION_UNSOLVABLE_CANDIDATE = "UNSOLVABLE_CANDIDATE_IN_RANGE"

MODE_DECISIONS = {
    "beal": (DECISION_NO_COUNTEREXAMPLE, DECISION_COUNTEREXAMPLE_CANDIDATE),
    "fermat_catalan": (DECISION_KNOWNS_ONLY, DECISION_NEW_TO_TABLE, DECISION_NO_SOLUTION),
    "erdos_straus": (DECISION_NO_UNSOLVABLE, DECISION_UNSOLVABLE_CANDIDATE),
}

LABEL_FAMILY = "KNOWN_COMMON_FACTOR_FAMILY"
LABEL_CANDIDATE = "COUNTEREXAMPLE_CANDIDATE"
LABEL_KNOWN = "KNOWN_SOLUTION_REDISCOVERED"
LABEL_NEW = "NEW_TO_BUILTIN_TABLE"

#: fp64 screen tolerances.  The screen's absolute error on the recovered base is below
#: ``C_hat * 2e-14`` (log, divide, exp each contribute O(1) ulp on values <= ~100), so
#: ``eps_rel = 1e-13`` keeps a >= 5x capture margin while ``delta_max = 0.125`` keeps a
#: 4x separation margin from the nearest wrong integer.  Both are enforced a priori.
EPS_REL = 1e-13
DELTA_FLOOR = 1e-9
DELTA_MAX_INVERSE = 8  # delta_max = 1/8
FP64_INTEGER_CAP = 2**52

#: Modulus of the exact congruence stage: the Mersenne prime 2^61 - 1.  A true
#: solution satisfies its identity modulo every integer, so congruence rejection is
#: exact; an fp64 mirage survives it with probability ~ 4e-19 and then meets the full
#: big-integer check.
CONGRUENCE_MODULUS = 2**61 - 1

SYSTEM_CAPS = {
    "base_cap": 100000,
    "congruence_modulus": CONGRUENCE_MODULUS,
    "es_extremal_top_k": 20,
    "es_n_max_cap": 10**8,
    "es_sample_size": 64,
    "exp_cap": 40,
    "fp64_integer_cap_log2": 52,
    "near_hit_cap": 50000000,
    "prefilter_delta_floor": "1e-9",
    "prefilter_delta_max": "0.125",
    "prefilter_eps_rel": "1e-13",
    "solution_sample_cap": 2000,
}

BEAL_CONTROL = {"a": 3, "x": 3, "b": 6, "y": 3, "c": 3, "z": 5}

LITERATURE = {
    "beal": {
        "citation": (
            "R. D. Mauldin, Notices Amer. Math. Soc. 44 (1997) 1436-1437; AMS Beal Prize "
            "($1,000,000); P. Norvig, Beal's conjecture: a search for counterexamples, "
            "norvig.com"
        ),
        "documented_search_landscape": (
            "Norvig's computational notes report no counterexample with bases to 250,000 "
            "and exponents to 1,000, nor with bases to 1,000 and exponents to 250,000; any "
            "box swept here sits inside that landscape"
        ),
    },
    "fermat_catalan": {
        "citation": (
            "H. Darmon and A. Granville, Bull. London Math. Soc. 27 (1995) 513-543; "
            "B. Poonen, E. F. Schaefer, M. Stoll, Duke Math. J. 137 (2007) 103-158 "
            "(standard list of the ten known solutions)"
        ),
        "known_solution_count": 10,
    },
    "erdos_straus": {
        "citation": (
            "C. Elsholtz and T. Tao, arXiv:1107.1010, J. Aust. Math. Soc. 94 (2013) "
            "50-105; S. E. Salez, arXiv:1406.6307 (2014): verified for all n <= 10^17"
        ),
        "verified_below": 10**17,
    },
}

_SCOPE = (
    "Generalized exponent-Diophantine sweep of one declared finite box for one queue v3 "
    "target; this module is the diophantine_family sweeper whose absence discovery "
    "receipts record as the typed blocker missing_sweeper:diophantine_family. The fp64 "
    "log-space layer (GPU or numpy) is a screen only, admitted after an a-priori "
    "resolution proof that guarantees every true solution is flagged; every near-hit is "
    "settled with exact Python big-integer arithmetic and the receipt records the "
    "near-hit -> exact funnel. Beal solutions carry exact gcds: a coprime confirmation "
    "is a COUNTEREXAMPLE_CANDIDATE with mandatory independent re-verification, while "
    "known gcd > 1 families are the rediscovery control. Fermat-Catalan confirmations "
    "are matched by power values against the builtin ten-solution table; unmatched "
    "confirmations claim no novelty. Erdos-Straus witnesses come from symbolically "
    "verified parametric identities on four residue classes plus an exact GPU-assisted "
    "search on n = 1 (mod 12), stored as counts, deterministic samples, and "
    "schedule-relative extremal witnesses, never full tables. A finite box proves "
    "nothing outside itself; every mode's box sits below its documented literature "
    "landscape, so these are mechanism receipts and the claims blocks say so."
)


class ExponentDiophantineError(ValueError):
    """Raised on malformed input, unsound boxes, integrity failures, or tamper."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExponentDiophantineError(message)


def _plain_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ExponentDiophantineError(f"{label} must be a plain integer")
    return value


# ---------------------------------------------------------------------------
# Exact integer helpers
# ---------------------------------------------------------------------------


def integer_root(value: int, degree: int) -> int:
    """Exact floor of the degree-th root of a nonnegative integer (Newton, then clamp)."""

    _require(value >= 0 and degree >= 1, "integer_root needs value >= 0 and degree >= 1")
    if value == 0 or degree == 1:
        return value
    root = 1 << (value.bit_length() // degree + 1)
    while True:
        step = ((degree - 1) * root + value // root ** (degree - 1)) // degree
        if step >= root:
            break
        root = step
    while root**degree > value:
        root -= 1
    while (root + 1) ** degree <= value:
        root += 1
    return root


def _gcd3(a: int, b: int, c: int) -> int:
    return gcd(a, gcd(b, c))


def _euclid_gcd(a: int, b: int) -> int:
    """Hand-rolled Euclid, kept separate from math.gcd for independent re-verification."""

    while b:
        a, b = b, a % b
    return a


def _iterated_power(base: int, exponent: int) -> int:
    """Plain repeated multiplication, kept separate from ** for independent re-checks."""

    result = 1
    for _ in range(exponent):
        result *= base
    return result


# ---------------------------------------------------------------------------
# Backend selection and the fp64 log-space screen
# ---------------------------------------------------------------------------


def _backend(use_gpu: bool) -> tuple[Any, str]:
    if use_gpu:
        import cupy

        name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        return cupy, name
    return np, "cpu-numpy"


def _host_array(array: Any) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array)


def screen_log_power(
    xp: Any,
    logs: np.ndarray,
    z_values: Sequence[int],
    *,
    chunk: int,
    near_cap: int,
    pair_cond: Callable[[Any, int, int, int], Any] | None = None,
) -> tuple[np.ndarray, int]:
    """fp64 log-space screen over ordered pair combinations of one power table.

    Returns ``(near_hits, lanes)``: near_hits is an int64 array of rows
    ``(i, j, z, c0)`` with ``i <= j`` indexing the pair table and ``c0`` the rounded
    base candidate.  ``pair_cond(xp, start, stop, z)`` may narrow the lane mask
    (exact integer arithmetic; used for the Fermat-Catalan exponent condition).
    Every lane whose candidate sits within ``max(eps_rel * C_hat, delta_floor)`` is
    handed to the exact layer; the resolution plan proves c0 is the only possible
    integer base, so nothing else needs to leave the device.
    """

    count = int(logs.shape[0])
    device_logs = xp.asarray(logs)
    columns = xp.arange(count, dtype=xp.int64)
    blocks: list[np.ndarray] = []
    near_count = 0
    lanes = 0
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        rows = xp.arange(start, stop, dtype=xp.int64)
        log_sum = xp.logaddexp(device_logs[start:stop, None], device_logs[None, :])
        triangle = columns[None, :] >= rows[:, None]
        if pair_cond is None:
            lanes += sum(count - i for i in range(start, stop)) * len(z_values)
        for z in z_values:
            if pair_cond is None:
                mask = triangle
            else:
                mask = triangle & pair_cond(xp, start, stop, z)
                lanes += int(mask.sum())
            candidate = xp.exp(log_sum / z)
            rounded = xp.rint(candidate)
            fraction = xp.abs(candidate - rounded)
            delta = xp.maximum(candidate * EPS_REL, DELTA_FLOOR)
            hits = mask & (fraction <= delta) & (rounded >= 2.0)
            if bool(hits.any()):
                index = xp.argwhere(hits)
                block = xp.empty((index.shape[0], 4), dtype=xp.int64)
                block[:, 0] = index[:, 0] + start
                block[:, 1] = index[:, 1]
                block[:, 2] = z
                block[:, 3] = rounded[hits].astype(xp.int64)
                blocks.append(_host_array(block))
                near_count += int(index.shape[0])
                _require(near_count <= near_cap, "near-hit count exceeds the declared cap")
    if not blocks:
        return np.empty((0, 4), dtype=np.int64), lanes
    return np.concatenate(blocks, axis=0), lanes


def _resolution_plan(max_sum: int, min_z: int) -> dict[str, int]:
    """A-priori fp64 resolution proof: the largest recoverable base must be far below
    both the integer-separation bound (eps_rel * C_hat <= 1/8) and 2^52."""

    largest_base = integer_root(max_sum, min_z) + 1
    separation_bound = 10**13 // DELTA_MAX_INVERSE
    _require(
        largest_base <= separation_bound and largest_base <= FP64_INTEGER_CAP,
        "declared box exceeds the fp64 screen's a-priori resolution bound; "
        "shrink base_max or raise the minimum right-hand exponent",
    )
    return {"largest_recoverable_base": largest_base, "separation_bound": separation_bound}


# ---------------------------------------------------------------------------
# Queue binding
# ---------------------------------------------------------------------------


def _default_queue_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "problem_queue_v3.json"


def _bind_queue(mode: str, queue_path: Path | str | None) -> str:
    path = Path(queue_path) if queue_path is not None else _default_queue_path()
    try:
        queue = load_queue(path)
    except ProblemQueueError as error:
        raise ExponentDiophantineError(f"queue binding failed: {error}") from error
    wanted = MODE_PROBLEM_IDS[mode]
    matches = [entry for entry in queue["entries"] if entry["id"] == wanted]
    _require(len(matches) == 1, f"queue does not declare the target {wanted}")
    _require(
        matches[0]["machine_form"]["kind"] == "diophantine_family",
        f"queue entry {wanted} is not a diophantine_family target",
    )
    return queue["content_sha256"]


# ---------------------------------------------------------------------------
# Receipt assembly
# ---------------------------------------------------------------------------


def _finish_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    sealed = {**dict(body), "content_sha256": canonical_sha256(body)}
    validate_receipt(sealed)
    return sealed


def _common_body(
    mode: str,
    queue_seal: str,
    box: Mapping[str, Any],
    device: str,
    arithmetic: Mapping[str, str],
    elapsed: float,
    throughput: int,
    prefilter: Mapping[str, Any],
    results: Mapping[str, Any],
    decision: str,
    claims: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "arithmetic": dict(arithmetic),
        "box": dict(box),
        "claims": dict(claims),
        "decision": decision,
        "device": device,
        "elapsed_seconds": format(elapsed, ".3f"),
        "literature": dict(LITERATURE[mode]),
        "mode": mode,
        "prefilter": dict(prefilter),
        "problem_id": MODE_PROBLEM_IDS[mode],
        "queue_content_sha256": queue_seal,
        "results": dict(results),
        "schema_version": RESULT_SCHEMA,
        "scope": _SCOPE,
        "system_caps": dict(SYSTEM_CAPS),
        "throughput_per_second": throughput,
    }


# ---------------------------------------------------------------------------
# Beal mode
# ---------------------------------------------------------------------------


def _power_pairs(base_max: int, exp_lo: int, exp_hi: int) -> list[tuple[int, int]]:
    return [(base, exp) for base in range(1, base_max + 1) for exp in range(exp_lo, exp_hi + 1)]


def _pair_logs(pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.array([exp * log(base) for base, exp in pairs], dtype=np.float64)


def classify_beal_solution(a: int, x: int, b: int, y: int, c: int, z: int) -> tuple[str, int]:
    """Exact gcd classification of an already-confirmed solution."""

    common = _gcd3(a, b, c)
    return (LABEL_FAMILY if common > 1 else LABEL_CANDIDATE), common


def independent_beal_recheck(sol: Mapping[str, int]) -> dict[str, Any]:
    """Second, structurally independent verification of a counterexample candidate."""

    a, x, b, y, c, z = (sol[key] for key in ("a", "x", "b", "y", "c", "z"))
    equation = _iterated_power(a, x) + _iterated_power(b, y) == _iterated_power(c, z)
    euclid = _euclid_gcd(_euclid_gcd(a, b), c)
    return {
        "equation_holds_by_iterated_multiplication": equation,
        "gcd_by_euclid": euclid,
        "gcd_by_math_gcd": _gcd3(a, b, c),
    }


def _near_hit_rows(near: np.ndarray, batch: int = 1 << 20):
    """Iterate near-hit rows in bounded host batches."""

    for start in range(0, near.shape[0], batch):
        yield from near[start : start + batch].tolist()


def _confirm_beal(
    pairs: Sequence[tuple[int, int]], near: np.ndarray
) -> tuple[list[dict[str, int | str]], int, int]:
    modulus = CONGRUENCE_MODULUS
    solutions: list[dict[str, int | str]] = []
    congruence_rejected = 0
    rejected = 0
    for i, j, z, c0 in _near_hit_rows(near):
        a, x = pairs[i]
        b, y = pairs[j]
        if (pow(a, x, modulus) + pow(b, y, modulus) - pow(c0, z, modulus)) % modulus:
            congruence_rejected += 1
            continue
        total = a**x + b**y
        c = integer_root(total, z)
        if c >= 2 and c**z == total:
            label, common = classify_beal_solution(a, x, b, y, c, z)
            solutions.append(
                {"a": a, "x": x, "b": b, "y": y, "c": c, "z": z, "gcd": common, "label": label}
            )
        else:
            rejected += 1
    solutions.sort(key=lambda s: (s["a"], s["x"], s["b"], s["y"], s["z"]))
    return solutions, congruence_rejected, rejected


def _beal_control_reachable(base_max: int, exp_max: int) -> bool:
    return base_max >= 6 and exp_max >= 5


def _beal_claims(results: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "box_decides_conjecture": False,
        "corpus_absence_establishes_novelty": False,
        "counterexample_candidate_present": results["counterexample_candidate_count"] > 0,
        "exceeds_documented_search_landscape": False,
        "known_common_factor_control_found": bool(results["control"]["found"]),
        "mechanism_receipt": True,
        "screen_trusted_without_exact_confirmation": False,
    }


def run_beal_sweep(
    base_max: int,
    exp_max: int,
    *,
    use_gpu: bool = True,
    chunk: int = 256,
    queue_path: Path | str | None = None,
) -> dict[str, Any]:
    """Sweep A^x + B^y = C^z with A, B <= base_max and x, y, z in [3, exp_max]."""

    base_max = _plain_int(base_max, "base_max")
    exp_max = _plain_int(exp_max, "exp_max")
    _require(2 <= base_max <= SYSTEM_CAPS["base_cap"], "base_max out of declared cap")
    _require(3 <= exp_max <= SYSTEM_CAPS["exp_cap"], "exp_max out of declared cap")
    _resolution_plan(2 * base_max**exp_max, 3)
    queue_seal = _bind_queue("beal", queue_path)
    xp, device = _backend(use_gpu)

    started = time.perf_counter()
    pairs = _power_pairs(base_max, 3, exp_max)
    near, lanes = screen_log_power(
        xp,
        _pair_logs(pairs),
        range(3, exp_max + 1),
        chunk=chunk,
        near_cap=SYSTEM_CAPS["near_hit_cap"],
    )
    solutions, congruence_rejected, rejected = _confirm_beal(pairs, near)
    elapsed = time.perf_counter() - started

    candidates = [dict(s) for s in solutions if s["label"] == LABEL_CANDIDATE]
    for candidate in candidates:
        recheck = independent_beal_recheck(candidate)
        _require(
            recheck["equation_holds_by_iterated_multiplication"]
            and recheck["gcd_by_euclid"] == recheck["gcd_by_math_gcd"] == 1,
            "independent re-verification of a counterexample candidate failed",
        )
        candidate["independent_recheck"] = recheck
    families = sum(1 for s in solutions if s["label"] == LABEL_FAMILY)

    control_reachable = _beal_control_reachable(base_max, exp_max)
    control_hits = [
        s for s in solutions if {k: s[k] for k in ("a", "x", "b", "y", "c", "z")} == BEAL_CONTROL
    ]
    _require(
        not control_reachable or bool(control_hits),
        "validation control failed: the box covers 3^3 + 6^3 = 3^5 but the sweep "
        "did not rediscover it",
    )
    cap = SYSTEM_CAPS["solution_sample_cap"]
    results = {
        "common_factor_family_count": families,
        "control": {
            "found": bool(control_hits),
            "reachable": control_reachable,
            "witness": dict(BEAL_CONTROL) if control_hits else None,
        },
        "counterexample_candidate_count": len(candidates),
        "counterexample_candidates": candidates,
        "sample_cap": cap,
        "solution_count": len(solutions),
        "solutions_sample": solutions[:cap],
        "witness_storage": (
            "all exactly confirmed solutions counted; the first sample_cap in "
            "(a, x, b, y, z) order stored; counterexample candidates always stored in full"
        ),
    }
    prefilter = {
        "congruence_rejected": congruence_rejected,
        "delta_floor": SYSTEM_CAPS["prefilter_delta_floor"],
        "eps_rel": SYSTEM_CAPS["prefilter_eps_rel"],
        "exact_confirmed": len(solutions),
        "exact_rejected": rejected,
        "lanes": lanes,
        "near_hits": int(near.shape[0]),
    }
    decision = (
        DECISION_COUNTEREXAMPLE_CANDIDATE if candidates else DECISION_NO_COUNTEREXAMPLE
    )
    body = _common_body(
        "beal",
        queue_seal,
        {"base_max": base_max, "exp_min": 3, "exp_max": exp_max},
        device,
        {"confirm": "python-bigint-exact", "screen": "fp64-log-space"},
        elapsed,
        int(lanes / elapsed) if elapsed > 0 else 0,
        prefilter,
        results,
        decision,
        _beal_claims(results),
    )
    return _finish_receipt(body)


# ---------------------------------------------------------------------------
# Fermat-Catalan mode
# ---------------------------------------------------------------------------


def fc_exponent_condition(p: int, q: int, r: int) -> bool:
    """1/p + 1/q + 1/r < 1, in exact integer arithmetic."""

    return q * r + p * r + p * q < p * q * r


def known_fermat_catalan_table() -> tuple[dict[str, Any], ...]:
    """The ten known solutions, canonical smaller-power-first, exactly verified here."""

    raw = (
        (1, 7, 2, 3, 3, 2),
        (2, 5, 7, 2, 3, 4),
        (13, 2, 7, 3, 2, 9),
        (2, 7, 17, 3, 71, 2),
        (3, 5, 11, 4, 122, 2),
        (17, 7, 76271, 3, 21063928, 2),
        (1414, 3, 2213459, 2, 65, 7),
        (9262, 3, 15312283, 2, 113, 7),
        (43, 8, 96222, 3, 30042907, 2),
        (33, 8, 1549034, 2, 15613, 3),
    )
    table = []
    for x, p, y, q, z, r in raw:
        _require(x**p + y**q == z**r, "builtin Fermat-Catalan table entry does not verify")
        _require(x**p <= y**q, "builtin Fermat-Catalan table entry is not canonical")
        _require(gcd(x, y) == 1, "builtin Fermat-Catalan table entry is not coprime")
        table.append(
            {
                "x": x,
                "p": p,
                "y": y,
                "q": q,
                "z": z,
                "r": r,
                "base_one_exponent_wildcard": x == 1,
                "powers": (x**p, y**q, z**r),
            }
        )
    return tuple(table)


def _fc_known_index(power_small: int, power_large: int, rhs: int) -> int | None:
    for index, entry in enumerate(known_fermat_catalan_table()):
        if entry["powers"] == (power_small, power_large, rhs):
            return index
    return None


def _fc_reachable_known_indices(base_max: int, pq_max: int, r_max: int) -> list[int]:
    reachable = []
    for index, entry in enumerate(known_fermat_catalan_table()):
        if max(entry["x"], entry["y"]) > base_max or entry["q"] > pq_max or entry["r"] > r_max:
            continue
        if entry["base_one_exponent_wildcard"]:
            if any(
                fc_exponent_condition(p, entry["q"], entry["r"]) for p in range(2, pq_max + 1)
            ):
                reachable.append(index)
        elif entry["p"] <= pq_max and fc_exponent_condition(entry["p"], entry["q"], entry["r"]):
            reachable.append(index)
    return reachable


def _confirm_fermat_catalan(
    pairs: Sequence[tuple[int, int]], near: np.ndarray
) -> tuple[list[dict[str, Any]], int, int, int]:
    modulus = CONGRUENCE_MODULUS
    solutions: list[dict[str, Any]] = []
    congruence_rejected = 0
    noncoprime = 0
    rejected = 0
    for i, j, r, z0 in _near_hit_rows(near):
        x, p = pairs[i]
        y, q = pairs[j]
        _require(fc_exponent_condition(p, q, r), "screen emitted a lane outside the condition")
        if (pow(x, p, modulus) + pow(y, q, modulus) - pow(z0, r, modulus)) % modulus:
            congruence_rejected += 1
            continue
        total = x**p + y**q
        z = integer_root(total, r)
        if not (z >= 2 and z**r == total):
            rejected += 1
            continue
        if gcd(x, y) != 1:
            noncoprime += 1
            continue
        if x**p > y**q:
            x, p, y, q = y, q, x, p
        known = _fc_known_index(x**p, y**q, total)
        solutions.append(
            {
                "known_index": known,
                "label": LABEL_KNOWN if known is not None else LABEL_NEW,
                "p": p,
                "q": q,
                "r": r,
                "x": x,
                "y": y,
                "z": z,
            }
        )
    solutions.sort(key=lambda s: (s["x"] ** s["p"], s["x"], s["p"], s["r"]))
    return solutions, congruence_rejected, noncoprime, rejected


def _fc_claims(results: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "all_reachable_known_solutions_rediscovered": True,
        "corpus_absence_establishes_novelty": False,
        "finiteness_decided": False,
        "mechanism_receipt": True,
        "new_to_builtin_table_present": results["new_to_table_count"] > 0,
        "novelty_claimed_for_new_hits": False,
        "screen_trusted_without_exact_confirmation": False,
    }


def run_fermat_catalan_sweep(
    base_max: int,
    pq_max: int,
    r_max: int,
    *,
    use_gpu: bool = True,
    chunk: int = 256,
    queue_path: Path | str | None = None,
) -> dict[str, Any]:
    """Sweep coprime x^p + y^q = z^r with 1/p + 1/q + 1/r < 1 over the declared box."""

    base_max = _plain_int(base_max, "base_max")
    pq_max = _plain_int(pq_max, "pq_max")
    r_max = _plain_int(r_max, "r_max")
    _require(2 <= base_max <= SYSTEM_CAPS["base_cap"], "base_max out of declared cap")
    _require(2 <= pq_max <= SYSTEM_CAPS["exp_cap"], "pq_max out of declared cap")
    _require(2 <= r_max <= SYSTEM_CAPS["exp_cap"], "r_max out of declared cap")
    _resolution_plan(2 * base_max**pq_max, 2)
    queue_seal = _bind_queue("fermat_catalan", queue_path)
    xp, device = _backend(use_gpu)

    started = time.perf_counter()
    pairs = _power_pairs(base_max, 2, pq_max)
    exponents_host = np.array([exp for _, exp in pairs], dtype=np.int64)
    exponents = xp.asarray(exponents_host)

    def condition(xp_: Any, start: int, stop: int, r: int) -> Any:
        p_rows = exponents[start:stop, None]
        q_cols = exponents[None, :]
        return q_cols * r + p_rows * r + p_rows * q_cols < p_rows * q_cols * r

    near, lanes = screen_log_power(
        xp,
        _pair_logs(pairs),
        range(2, r_max + 1),
        chunk=chunk,
        near_cap=SYSTEM_CAPS["near_hit_cap"],
        pair_cond=condition,
    )
    solutions, congruence_rejected, noncoprime, rejected = _confirm_fermat_catalan(pairs, near)
    elapsed = time.perf_counter() - started

    required = _fc_reachable_known_indices(base_max, pq_max, r_max)
    found_known = sorted(
        {s["known_index"] for s in solutions if s["known_index"] is not None}
    )
    missing = [index for index in required if index not in found_known]
    _require(
        not missing,
        "validation control failed: reachable known Fermat-Catalan solutions "
        f"{missing} were not rediscovered",
    )
    known_count = sum(1 for s in solutions if s["label"] == LABEL_KNOWN)
    new_count = len(solutions) - known_count
    _require(len(solutions) <= SYSTEM_CAPS["solution_sample_cap"], "solution list exceeds cap")
    results = {
        "builtin_table_size": len(known_fermat_catalan_table()),
        "found_known_indices": found_known,
        "known_rediscovered_count": known_count,
        "new_to_table_count": new_count,
        "prior_art_note": (
            "a confirmed solution absent from the builtin table is overwhelmingly likely "
            "to appear in prior computational literature; no novelty is claimed and "
            "corpus absence establishes nothing"
        ),
        "required_known_indices": required,
        "solution_count": len(solutions),
        "solutions": solutions,
    }
    prefilter = {
        "congruence_rejected": congruence_rejected,
        "delta_floor": SYSTEM_CAPS["prefilter_delta_floor"],
        "eps_rel": SYSTEM_CAPS["prefilter_eps_rel"],
        "exact_confirmed_coprime": len(solutions),
        "exact_confirmed_noncoprime_excluded": noncoprime,
        "exact_rejected": rejected,
        "lanes": lanes,
        "near_hits": int(near.shape[0]),
    }
    if not solutions:
        decision = DECISION_NO_SOLUTION
    elif new_count:
        decision = DECISION_NEW_TO_TABLE
    else:
        decision = DECISION_KNOWNS_ONLY
    body = _common_body(
        "fermat_catalan",
        queue_seal,
        {"base_max": base_max, "pq_min": 2, "pq_max": pq_max, "r_min": 2, "r_max": r_max},
        device,
        {"confirm": "python-bigint-exact", "screen": "fp64-log-space"},
        elapsed,
        int(lanes / elapsed) if elapsed > 0 else 0,
        prefilter,
        results,
        decision,
        _fc_claims(results),
    )
    return _finish_receipt(body)


# ---------------------------------------------------------------------------
# Erdos-Straus mode
# ---------------------------------------------------------------------------

#: Residue-class partition of [2, n_max] in priority order.  first/stride give the
#: class members in closed form; the witness rules are proven symbolically and spot
#: verified exactly.  The hard class carries no rule: it is searched.
ES_CLASSES = ("even", "mod4_3", "mod3_0", "mod3_2", "hard_1_mod_12")

ES_WITNESS_RULES = {
    "even": "n = 2m: (m, m+1, m*(m+1))",
    "mod4_3": "n = 4k+3: (k+2, (k+1)*(k+2), (k+1)*n)",
    "mod3_0": "n = 3m (odd, n = 1 mod 4): (m, n+1, n*(n+1))",
    "mod3_2": "n = 3k+2 (odd, n = 1 mod 4): (k+1, n, n*(k+1))",
}


def _es_class_geometry(n_max: int) -> dict[str, tuple[int, int, int]]:
    """(first, stride, count) per class over [2, n_max]; requires n_max >= 13."""

    return {
        "even": (2, 2, n_max // 2),
        "mod4_3": (3, 4, (n_max - 3) // 4 + 1),
        "mod3_0": (9, 12, (n_max - 9) // 12 + 1),
        "mod3_2": (5, 12, (n_max - 5) // 12 + 1),
        "hard_1_mod_12": (13, 12, (n_max - 13) // 12 + 1),
    }


def es_parametric_witness(class_name: str, n: int) -> tuple[int, int, int]:
    if class_name == "even":
        m = n // 2
        return (m, m + 1, m * (m + 1))
    if class_name == "mod4_3":
        k = (n - 3) // 4
        return (k + 2, (k + 1) * (k + 2), (k + 1) * n)
    if class_name == "mod3_0":
        return (n // 3, n + 1, n * (n + 1))
    if class_name == "mod3_2":
        k = (n - 2) // 3
        return (k + 1, n, n * (k + 1))
    raise ExponentDiophantineError(f"no parametric witness for class {class_name}")


def es_witness_is_exact(n: int, x: int, y: int, z: int) -> bool:
    """4/n = 1/x + 1/y + 1/z with 1 <= x <= y <= z, in exact integer arithmetic."""

    if not 1 <= x <= y <= z:
        return False
    return n * (y * z + x * z + x * y) == 4 * x * y * z


def es_symbolic_identity_checks() -> dict[str, bool]:
    """Verify each parametric witness rule as an exact rational-function identity."""

    import sympy

    k = sympy.symbols("k", positive=True, integer=True)
    forms = {
        "even": (2 * k, k, k + 1, k * (k + 1)),
        "mod4_3": (4 * k + 3, k + 2, (k + 1) * (k + 2), (k + 1) * (4 * k + 3)),
        "mod3_0": (3 * k, k, 3 * k + 1, 3 * k * (3 * k + 1)),
        "mod3_2": (3 * k + 2, k + 1, 3 * k + 2, (3 * k + 2) * (k + 1)),
    }
    checks = {}
    for name, (n_expr, x_expr, y_expr, z_expr) in forms.items():
        expr = sympy.Rational(4) / n_expr - 1 / x_expr - 1 / y_expr - 1 / z_expr
        checks[name] = sympy.simplify(sympy.together(expr)) == 0
    return checks


def _es_sample_indices(count: int, sample_size: int) -> list[int]:
    if count <= sample_size:
        return list(range(count))
    return sorted({k * (count - 1) // (sample_size - 1) for k in range(sample_size)})


def _es_hard_members(n_max: int) -> np.ndarray:
    return np.arange(13, n_max + 1, 12, dtype=np.int64)


def _es_hard_rounds(
    xp: Any, members: np.ndarray, x_rounds: int, t_rounds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Exact int64 GPU search schedule over the hard class; returns host arrays.

    For n = 1 (mod 4) and x = n//4 + 1 + dx the numerator a = 4x - n = 3 + 4*dx is a
    scalar per round.  With y = ceil(b/a) + t the divisor d = a*y - b stays below
    a*(t+1), so the divisibility test (b*y) % d == 0 reduces to modular products below
    d^2 — every intermediate provably fits int64 for n_max <= the declared cap.
    """

    n = xp.asarray(members)
    resolved = xp.zeros(n.shape, dtype=bool)
    wx = xp.zeros(n.shape, dtype=xp.int64)
    wy = xp.zeros(n.shape, dtype=xp.int64)
    base_x = n // 4 + 1
    lane_tests = 0
    for dx in range(x_rounds):
        if bool(resolved.all()):
            break
        a = 3 + 4 * dx
        x = base_x + dx
        b = n * x
        y_start = (b + a - 1) // a
        for t in range(t_rounds):
            open_lanes = ~resolved
            lane_tests += int(open_lanes.sum())
            y = y_start + t
            d = a * y - b
            safe = xp.where(d > 0, d, xp.int64(1))
            ok = (
                open_lanes
                & (d > 0)
                & (y >= x)
                & (((b % safe) * (y % safe)) % safe == 0)
            )
            wx = xp.where(ok, x, wx)
            wy = xp.where(ok, y, wy)
            resolved = resolved | ok
    return _host_array(wx), _host_array(wy), _host_array(resolved), lane_tests


def _divisors_of_square(factors: Mapping[int, int]) -> list[int]:
    divisors = [1]
    for prime, exponent in factors.items():
        powers = [prime**e for e in range(2 * exponent + 1)]
        divisors = [d * p for d in divisors for p in powers]
    return sorted(divisors)


def es_complete_search(n: int) -> tuple[int, int, int] | None:
    """Exhaustive first-witness search: x over (n/4, 3n/4], divisor pairs of b^2."""

    import sympy

    n_factors = dict(sympy.factorint(n))
    for x in range(n // 4 + 1, (3 * n) // 4 + 1):
        a = 4 * x - n
        b = n * x
        merged = dict(n_factors)
        for prime, exponent in sympy.factorint(x).items():
            merged[prime] = merged.get(prime, 0) + exponent
        for d1 in _divisors_of_square(merged):
            if d1 > b:
                break
            if (b + d1) % a:
                continue
            y = (b + d1) // a
            if y < x:
                continue
            d2 = (b * b) // d1
            if (b + d2) % a:
                continue
            return (x, y, (b + d2) // a)
    return None


def _es_claims(n_max: int, results: Mapping[str, Any]) -> dict[str, bool]:
    exceeds = n_max > LITERATURE["erdos_straus"]["verified_below"]
    return {
        "corpus_absence_establishes_novelty": False,
        "exceeds_literature_bound": exceeds,
        "full_witness_table_stored": False,
        "mechanism_receipt_below_literature_bound": not exceeds,
        "range_decides_conjecture": False,
        "unsolvable_candidate_present": bool(results["unsolvable_candidates"]),
        "witnesses_exact_verified": True,
    }


def run_erdos_straus_sweep(
    n_max: int,
    *,
    use_gpu: bool = True,
    x_rounds: int = 64,
    t_rounds: int = 32,
    queue_path: Path | str | None = None,
) -> dict[str, Any]:
    """Witness sweep of 4/n = 1/x + 1/y + 1/z for every n in [2, n_max]."""

    n_max = _plain_int(n_max, "n_max")
    x_rounds = _plain_int(x_rounds, "x_rounds")
    t_rounds = _plain_int(t_rounds, "t_rounds")
    _require(13 <= n_max <= SYSTEM_CAPS["es_n_max_cap"], "n_max out of declared bounds")
    _require(1 <= x_rounds <= 4096 and 1 <= t_rounds <= 4096, "search schedule out of bounds")
    queue_seal = _bind_queue("erdos_straus", queue_path)
    xp, device = _backend(use_gpu)

    started = time.perf_counter()
    identity_checks = es_symbolic_identity_checks()
    _require(all(identity_checks.values()), "a parametric witness identity failed symbolically")
    geometry = _es_class_geometry(n_max)
    sample_size = SYSTEM_CAPS["es_sample_size"]

    classes: dict[str, dict[str, Any]] = {}
    for name in ES_CLASSES[:-1]:
        first, stride, count = geometry[name]
        sample = []
        for index in _es_sample_indices(count, sample_size):
            n = first + stride * index
            x, y, z = es_parametric_witness(name, n)
            _require(es_witness_is_exact(n, x, y, z), f"parametric witness failed at n={n}")
            sample.append({"n": n, "x": x, "y": y, "z": z})
        classes[name] = {
            "count": count,
            "first_n": first,
            "last_n": first + stride * (count - 1),
            "sample": sample,
            "symbolic_identity_verified": identity_checks[name],
            "witness_rule": ES_WITNESS_RULES[name],
        }

    members = _es_hard_members(n_max)
    wx, wy, resolved, lane_tests = _es_hard_rounds(xp, members, x_rounds, t_rounds)
    hard_witnesses: list[tuple[int, int, int, int]] = []
    cpu_completed = 0
    unsolvable: list[int] = []
    for index in range(members.shape[0]):
        n = int(members[index])
        if bool(resolved[index]):
            x, y = int(wx[index]), int(wy[index])
            b = n * x
            d = (4 * x - n) * y - b
            _require(d > 0 and (b * y) % d == 0, "GPU hard-class hit failed exact division")
            z = (b * y) // d
        else:
            found = es_complete_search(n)
            if found is None:
                unsolvable.append(n)
                continue
            cpu_completed += 1
            x, y, z = found
        _require(es_witness_is_exact(n, x, y, z), f"hard-class witness failed exactly at n={n}")
        hard_witnesses.append((n, x, y, z))

    hard_first, hard_stride, hard_count = geometry["hard_1_mod_12"]
    _require(len(hard_witnesses) + len(unsolvable) == hard_count, "hard-class accounting broke")
    by_n = {w[0]: w for w in hard_witnesses}
    hard_sample = []
    for index in _es_sample_indices(hard_count, sample_size):
        n = hard_first + hard_stride * index
        if n in by_n:
            _, x, y, z = by_n[n]
            hard_sample.append({"n": n, "x": x, "y": y, "z": z})
    offsets = sorted(
        ((x - (n // 4 + 1), n, x, y, z) for n, x, y, z in hard_witnesses),
        key=lambda item: (-item[0], item[1]),
    )
    top_k = SYSTEM_CAPS["es_extremal_top_k"]
    extremal = [
        {"n": n, "x": x, "x_offset": offset, "y": y, "z": z}
        for offset, n, x, y, z in offsets[:top_k]
    ]
    classes["hard_1_mod_12"] = {
        "count": hard_count,
        "extremal_min_x": extremal,
        "first_n": hard_first,
        "last_n": hard_first + hard_stride * (hard_count - 1),
        "max_x_offset": offsets[0][0] if offsets else 0,
        "sample": hard_sample,
        "search_rule": (
            "x ascending from n//4 + 1 with a small-divisor y window per x "
            f"({x_rounds} x-rounds, {t_rounds} y-offsets, exact int64), then a complete "
            "per-x divisor search of b^2 on the CPU; x_offset is schedule-relative"
        ),
    }
    elapsed = time.perf_counter() - started

    class_total = sum(classes[name]["count"] for name in ES_CLASSES)
    _require(class_total == n_max - 1, "residue classes do not cover [2, n_max] exactly")
    results = {
        "classes": classes,
        "coverage": {"class_total": class_total, "expected_total": n_max - 1},
        "unsolvable_candidates": unsolvable,
        "witness_storage": (
            "full witness tables (~n_max rows) are deliberately not stored: counts per "
            "class, deterministic strided samples, and schedule-relative extremal "
            "minimal-x witnesses only; every stored witness is exactly verified and "
            "every hard-class witness was exactly verified during the run"
        ),
    }
    prefilter = {
        "cpu_divisor_completed": cpu_completed,
        "gpu_lane_tests": lane_tests,
        "gpu_resolved": int(resolved.sum()),
        "hard_class_count": hard_count,
    }
    decision = DECISION_UNSOLVABLE_CANDIDATE if unsolvable else DECISION_NO_UNSOLVABLE
    body = _common_body(
        "erdos_straus",
        queue_seal,
        {"gpu_t_rounds": t_rounds, "gpu_x_rounds": x_rounds, "n_min": 2, "n_max": n_max},
        device,
        {"confirm": "python-bigint-exact", "screen": "int64-exact-modular"},
        elapsed,
        int((n_max - 1) / elapsed) if elapsed > 0 else 0,
        prefilter,
        results,
        decision,
        _es_claims(n_max, results),
    )
    return _finish_receipt(body)


# ---------------------------------------------------------------------------
# Receipt validation (fail-closed; exact witness re-verification)
# ---------------------------------------------------------------------------

_TOP_KEYS = {
    "arithmetic",
    "box",
    "claims",
    "content_sha256",
    "decision",
    "device",
    "elapsed_seconds",
    "literature",
    "mode",
    "prefilter",
    "problem_id",
    "queue_content_sha256",
    "results",
    "schema_version",
    "scope",
    "system_caps",
    "throughput_per_second",
}


def _validate_beal(value: Mapping[str, Any]) -> None:
    box = value["box"]
    base_max = _plain_int(box["base_max"], "box.base_max")
    exp_max = _plain_int(box["exp_max"], "box.exp_max")
    _require(box["exp_min"] == 3, "beal box exp_min changed")
    _resolution_plan(2 * base_max**exp_max, 3)
    results = value["results"]
    solutions = results["solutions_sample"]
    _require(
        len(solutions) == min(results["solution_count"], results["sample_cap"]),
        "solutions sample length does not match its count and cap",
    )
    for sol in solutions:
        a, x, b, y, c, z = (
            _plain_int(sol[key], f"solution.{key}") for key in ("a", "x", "b", "y", "c", "z")
        )
        _require(1 <= a <= base_max and 1 <= b <= base_max, "solution bases outside box")
        _require(all(3 <= e <= exp_max for e in (x, y, z)), "solution exponents outside box")
        _require(a**x + b**y == c**z, "stored solution fails exact re-verification")
        label, common = classify_beal_solution(a, x, b, y, c, z)
        _require(sol["gcd"] == common and sol["label"] == label, "solution gcd or label changed")
    candidates = results["counterexample_candidates"]
    _require(
        results["counterexample_candidate_count"] == len(candidates),
        "candidate count does not match the stored candidates",
    )
    for candidate in candidates:
        a, x, b, y, c, z = (candidate[key] for key in ("a", "x", "b", "y", "c", "z"))
        _require(a**x + b**y == c**z, "candidate fails exact re-verification")
        _require(_gcd3(a, b, c) == 1, "candidate is not coprime on exact recheck")
        recheck = independent_beal_recheck(candidate)
        _require(
            candidate["independent_recheck"] == recheck
            and recheck["equation_holds_by_iterated_multiplication"]
            and recheck["gcd_by_euclid"] == 1,
            "candidate independent re-verification changed or fails",
        )
    control = results["control"]
    _require(
        control["reachable"] == _beal_control_reachable(base_max, exp_max),
        "control reachability does not match the box",
    )
    if control["reachable"]:
        _require(control["found"] and control["witness"] == BEAL_CONTROL, "control witness lost")
        witness = control["witness"]
        _require(
            witness["a"] ** witness["x"] + witness["b"] ** witness["y"]
            == witness["c"] ** witness["z"],
            "control witness fails exact re-verification",
        )
    expected_decision = (
        DECISION_COUNTEREXAMPLE_CANDIDATE if candidates else DECISION_NO_COUNTEREXAMPLE
    )
    _require(value["decision"] == expected_decision, "decision does not match the results")
    _require(value["claims"] == _beal_claims(results), "claims do not recompute")
    prefilter = value["prefilter"]
    _require(
        prefilter["near_hits"]
        == prefilter["congruence_rejected"]
        + prefilter["exact_confirmed"]
        + prefilter["exact_rejected"],
        "prefilter funnel does not balance",
    )
    _require(
        prefilter["exact_confirmed"] == results["solution_count"]
        and results["solution_count"]
        == results["common_factor_family_count"] + len(candidates),
        "solution counts do not balance",
    )


def _validate_fermat_catalan(value: Mapping[str, Any]) -> None:
    box = value["box"]
    base_max = _plain_int(box["base_max"], "box.base_max")
    pq_max = _plain_int(box["pq_max"], "box.pq_max")
    r_max = _plain_int(box["r_max"], "box.r_max")
    _require(box["pq_min"] == 2 and box["r_min"] == 2, "fc box minima changed")
    _resolution_plan(2 * base_max**pq_max, 2)
    results = value["results"]
    solutions = results["solutions"]
    _require(results["solution_count"] == len(solutions), "solution count changed")
    known_count = 0
    found_known = set()
    for sol in solutions:
        x, p, y, q, z, r = (
            _plain_int(sol[key], f"solution.{key}") for key in ("x", "p", "y", "q", "z", "r")
        )
        _require(1 <= x <= base_max and 1 <= y <= base_max, "solution bases outside box")
        _require(2 <= p <= pq_max and 2 <= q <= pq_max and 2 <= r <= r_max, "exponents outside")
        _require(fc_exponent_condition(p, q, r), "solution violates the exponent condition")
        _require(gcd(x, y) == 1, "solution is not coprime on exact recheck")
        _require(x**p + y**q == z**r, "stored solution fails exact re-verification")
        _require(x**p <= y**q, "solution is not canonical smaller-power-first")
        known = _fc_known_index(x**p, y**q, z**r)
        _require(sol["known_index"] == known, "solution known_index changed")
        expected_label = LABEL_KNOWN if known is not None else LABEL_NEW
        _require(sol["label"] == expected_label, "solution label changed")
        if known is not None:
            known_count += 1
            found_known.add(known)
    _require(results["known_rediscovered_count"] == known_count, "known count changed")
    _require(
        results["new_to_table_count"] == len(solutions) - known_count, "new count changed"
    )
    _require(
        results["found_known_indices"] == sorted(found_known), "found known indices changed"
    )
    required = _fc_reachable_known_indices(base_max, pq_max, r_max)
    _require(results["required_known_indices"] == required, "required known indices changed")
    _require(
        all(index in found_known for index in required),
        "a reachable known solution is missing from the receipt",
    )
    if not solutions:
        expected_decision = DECISION_NO_SOLUTION
    elif results["new_to_table_count"]:
        expected_decision = DECISION_NEW_TO_TABLE
    else:
        expected_decision = DECISION_KNOWNS_ONLY
    _require(value["decision"] == expected_decision, "decision does not match the results")
    _require(value["claims"] == _fc_claims(results), "claims do not recompute")
    prefilter = value["prefilter"]
    _require(
        prefilter["near_hits"]
        == prefilter["congruence_rejected"]
        + prefilter["exact_confirmed_coprime"]
        + prefilter["exact_confirmed_noncoprime_excluded"]
        + prefilter["exact_rejected"],
        "prefilter funnel does not balance",
    )


def _validate_erdos_straus(value: Mapping[str, Any]) -> None:
    box = value["box"]
    n_max = _plain_int(box["n_max"], "box.n_max")
    _require(box["n_min"] == 2 and 13 <= n_max <= SYSTEM_CAPS["es_n_max_cap"], "es box changed")
    results = value["results"]
    geometry = _es_class_geometry(n_max)
    sample_size = SYSTEM_CAPS["es_sample_size"]
    identity_checks = es_symbolic_identity_checks()
    _require(all(identity_checks.values()), "a parametric identity fails symbolically")
    classes = results["classes"]
    _require(set(classes) == set(ES_CLASSES), "residue class set changed")
    unsolvable = results["unsolvable_candidates"]
    for name in ES_CLASSES:
        block = classes[name]
        first, stride, count = geometry[name]
        _require(block["count"] == count, f"class {name} count changed")
        _require(block["first_n"] == first, f"class {name} first member changed")
        _require(
            block["last_n"] == first + stride * (count - 1), f"class {name} last member changed"
        )
        expected_ns = [first + stride * i for i in _es_sample_indices(count, sample_size)]
        parametric = name != "hard_1_mod_12"
        sample = block["sample"]
        if parametric:
            _require([w["n"] for w in sample] == expected_ns, f"class {name} sample ns changed")
        else:
            _require(
                [w["n"] for w in sample] == [n for n in expected_ns if n not in unsolvable],
                "hard class sample ns changed",
            )
        for witness in sample:
            n, x, y, z = (witness[key] for key in ("n", "x", "y", "z"))
            _require((n - first) % stride == 0 and first <= n <= n_max, "sample n off-class")
            _require(es_witness_is_exact(n, x, y, z), f"stored witness fails exactly at n={n}")
            if parametric:
                _require(
                    (x, y, z) == es_parametric_witness(name, n),
                    f"class {name} witness is not the parametric witness at n={n}",
                )
    hard = classes["hard_1_mod_12"]
    for witness in hard["extremal_min_x"]:
        n, x, y, z = (witness[key] for key in ("n", "x", "y", "z"))
        _require(es_witness_is_exact(n, x, y, z), "extremal witness fails exact recheck")
        _require(witness["x_offset"] == x - (n // 4 + 1), "extremal x_offset changed")
    if hard["extremal_min_x"]:
        _require(
            hard["max_x_offset"] == hard["extremal_min_x"][0]["x_offset"],
            "max_x_offset does not match the extremal table",
        )
    coverage = results["coverage"]
    class_total = sum(classes[name]["count"] for name in ES_CLASSES)
    _require(
        coverage["class_total"] == class_total == n_max - 1
        and coverage["expected_total"] == n_max - 1,
        "coverage accounting changed",
    )
    expected_decision = (
        DECISION_UNSOLVABLE_CANDIDATE if unsolvable else DECISION_NO_UNSOLVABLE
    )
    _require(value["decision"] == expected_decision, "decision does not match the results")
    _require(value["claims"] == _es_claims(n_max, results), "claims do not recompute")
    prefilter = value["prefilter"]
    _require(
        prefilter["hard_class_count"] == geometry["hard_1_mod_12"][2],
        "hard class count changed",
    )
    _require(
        prefilter["gpu_resolved"] + prefilter["cpu_divisor_completed"] + len(unsolvable)
        == prefilter["hard_class_count"],
        "hard-class funnel does not balance",
    )


_MODE_VALIDATORS = {
    "beal": _validate_beal,
    "fermat_catalan": _validate_fermat_catalan,
    "erdos_straus": _validate_erdos_straus,
}


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal, schema, coherence, claims recompute, and exact witness re-verification."""

    if not isinstance(value, Mapping):
        raise ExponentDiophantineError("receipt must be a mapping")
    if set(value) != _TOP_KEYS:
        raise ExponentDiophantineError("receipt top-level keys changed")
    if value["schema_version"] != RESULT_SCHEMA:
        raise ExponentDiophantineError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["content_sha256"] != canonical_sha256(body):
        raise ExponentDiophantineError("receipt seal changed")
    mode = value["mode"]
    if mode not in MODE_PROBLEM_IDS:
        raise ExponentDiophantineError(f"unknown mode: {mode!r}")
    _require(value["problem_id"] == MODE_PROBLEM_IDS[mode], "problem binding changed")
    seal = value["queue_content_sha256"]
    _require(
        isinstance(seal, str) and len(seal) == 64 and all(c in "0123456789abcdef" for c in seal),
        "queue seal must be a lowercase SHA-256 digest",
    )
    _require(value["decision"] in MODE_DECISIONS[mode], "decision is not declared for this mode")
    _require(value["literature"] == LITERATURE[mode], "literature block changed")
    _require(value["system_caps"] == SYSTEM_CAPS, "system caps changed")
    _require(value["scope"] == _SCOPE, "scope changed")
    _MODE_VALIDATORS[mode](value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ExponentDiophantineError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generalized exponent-Diophantine GPU sweeper (queue v3 targets)."
    )
    parser.add_argument("--mode", choices=sorted(MODE_PROBLEM_IDS))
    parser.add_argument("--base-max", type=int)
    parser.add_argument("--exp-max", type=int)
    parser.add_argument("--pq-max", type=int)
    parser.add_argument("--r-max", type=int)
    parser.add_argument("--n-max", type=int)
    parser.add_argument("--x-rounds", type=int, default=64)
    parser.add_argument("--t-rounds", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--cpu", action="store_true", help="force the numpy screen")
    parser.add_argument("--queue", help="path to the sealed problem queue (default: v3)")
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_checked:
        if not args.output:
            parser.error("--validate-checked requires --output")
        validate_receipt(json.loads(Path(args.output).read_text(encoding="utf-8")))
        print(f"VALID {args.output}")
        return 0
    if args.mode is None:
        parser.error("--mode is required to run a sweep")
    use_gpu = not args.cpu
    if args.mode == "beal":
        if args.base_max is None or args.exp_max is None:
            parser.error("beal mode requires --base-max and --exp-max")
        receipt = run_beal_sweep(
            args.base_max, args.exp_max, use_gpu=use_gpu, chunk=args.chunk,
            queue_path=args.queue,
        )
    elif args.mode == "fermat_catalan":
        if args.base_max is None or args.pq_max is None or args.r_max is None:
            parser.error("fermat_catalan mode requires --base-max, --pq-max, and --r-max")
        receipt = run_fermat_catalan_sweep(
            args.base_max, args.pq_max, args.r_max, use_gpu=use_gpu, chunk=args.chunk,
            queue_path=args.queue,
        )
    else:
        if args.n_max is None:
            parser.error("erdos_straus mode requires --n-max")
        receipt = run_erdos_straus_sweep(
            args.n_max, use_gpu=use_gpu, x_rounds=args.x_rounds, t_rounds=args.t_rounds,
            queue_path=args.queue,
        )
    if args.output:
        _write_immutable(Path(args.output), receipt)
    summary = {
        "mode": receipt["mode"],
        "decision": receipt["decision"],
        "box": receipt["box"],
        "prefilter": receipt["prefilter"],
        "device": receipt["device"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "throughput_per_second": receipt["throughput_per_second"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    loud = (DECISION_COUNTEREXAMPLE_CANDIDATE, DECISION_UNSOLVABLE_CANDIDATE)
    return 3 if receipt["decision"] in loud else 0


if __name__ == "__main__":
    raise SystemExit(main())
